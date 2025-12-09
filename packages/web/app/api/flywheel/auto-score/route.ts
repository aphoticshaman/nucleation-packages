import { NextResponse } from 'next/server';
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import { getLFBMClient } from '@/lib/inference/LFBMClient';

// Lazy initialization
let supabase: SupabaseClient | null = null;

function getSupabase() {
  if (!supabase) {
    supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );
  }
  return supabase;
}

interface Prediction {
  id: string;
  domain: string;
  prediction_type: string;
  prediction_content: Record<string, unknown>;
  confidence: number;
  predicted_timeframe_hours: number;
  source_example_ids: string[];
  created_at: string;
}

// Auto-score expired predictions by checking recent events
export async function POST(request: Request) {
  const authHeader = request.headers.get('authorization');
  const cronSecret = process.env.CRON_SECRET;

  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    const isVercelCron = request.headers.get('x-vercel-cron') === '1';
    if (!isVercelCron) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
  }

  try {
    const db = getSupabase();
    const lfbm = getLFBMClient();

    // Get expired unscored predictions
    const { data: predictions, error: fetchError } = await db
      .from('predictions')
      .select('*')
      .eq('outcome_observed', false)
      .lt('expires_at', new Date().toISOString())
      .limit(10);  // Process 10 at a time

    if (fetchError) throw fetchError;
    if (!predictions || predictions.length === 0) {
      return NextResponse.json({
        success: true,
        message: 'No predictions to score',
        scored: 0,
      });
    }

    const results = [];

    for (const pred of predictions as Prediction[]) {
      try {
        // Get recent events in the same domain from the prediction timeframe
        const predictionEnd = new Date(pred.created_at);
        predictionEnd.setHours(predictionEnd.getHours() + pred.predicted_timeframe_hours);

        const { data: recentEvents } = await db
          .from('training_examples')
          .select('input, output, domain')
          .eq('domain', pred.domain)
          .gte('created_at', pred.created_at)
          .lte('created_at', predictionEnd.toISOString())
          .limit(10);

        // Use Haiku to score the prediction
        const prompt = `You are evaluating whether a prediction came true.

PREDICTION (made at ${pred.created_at}):
Domain: ${pred.domain}
Type: ${pred.prediction_type}
Prediction: ${JSON.stringify(pred.prediction_content)}
Confidence: ${pred.confidence}
Timeframe: ${pred.predicted_timeframe_hours} hours

EVENTS THAT OCCURRED IN THAT TIMEFRAME:
${recentEvents?.map(e => `- ${e.input}`).join('\n') || 'No events recorded'}

Score how accurate the prediction was from 0.0 to 1.0:
- 1.0 = Prediction was exactly correct
- 0.75 = Prediction was mostly correct, minor details off
- 0.5 = Prediction was partially correct
- 0.25 = Prediction had some truth but mostly wrong
- 0.0 = Prediction was completely wrong or opposite happened

Respond with ONLY a JSON object:
{"accuracy": 0.X, "reasoning": "brief explanation"}`;

        const lfbmResponse = await lfbm.generateRaw({
          userMessage: prompt,
          max_tokens: 256,
        });

        const jsonMatch = lfbmResponse.match(/\{[\s\S]*\}/);
        const parsed = JSON.parse(jsonMatch?.[0] || '{"accuracy": 0.5, "reasoning": "Could not parse"}');
        const accuracy = Math.max(0, Math.min(1, parsed.accuracy || 0.5));

        // Score the prediction (this updates training example weights)
        await db.rpc('score_prediction', {
          p_prediction_id: pred.id,
          p_accuracy: accuracy,
          p_notes: parsed.reasoning || 'Auto-scored',
        });

        results.push({
          id: pred.id,
          domain: pred.domain,
          accuracy,
          reasoning: parsed.reasoning,
        });
      } catch (e) {
        console.error(`Failed to score prediction ${pred.id}:`, e);
        results.push({
          id: pred.id,
          error: String(e),
        });
      }
    }

    return NextResponse.json({
      success: true,
      scored: results.filter(r => !('error' in r)).length,
      failed: results.filter(r => 'error' in r).length,
      results,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error('Auto-score error:', error);
    return NextResponse.json({ error: 'Failed to auto-score predictions' }, { status: 500 });
  }
}
