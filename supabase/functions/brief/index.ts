// LatticeForge Hourly Brief - Supabase Edge Function
// Generates executive summaries using Claude Haiku (~$1/month)
// Deploy: supabase functions deploy brief

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts';
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface SignalSnapshot {
  volatility: number;
  sentiment: number;
  momentum: number;
  fear_greed: number;
  phase: string;
  anomalies: string[];
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    const anthropicKey = Deno.env.get('ANTHROPIC_API_KEY');
    if (!anthropicKey) {
      throw new Error('ANTHROPIC_API_KEY not configured');
    }

    // Gather current signals (simplified - in production, pull from real sources)
    const signals = await gatherSignals();

    // Check if signals changed significantly from last brief
    const { data: lastBrief } = await supabase
      .from('briefs')
      .select('signals_snapshot')
      .order('created_at', { ascending: false })
      .limit(1)
      .single();

    const shouldGenerate = !lastBrief || hasSignificantChange(lastBrief.signals_snapshot, signals);

    if (!shouldGenerate) {
      return jsonResponse({
        status: 'skipped',
        reason: 'No significant signal change',
        last_brief_still_valid: true,
      });
    }

    // Generate brief with Claude Haiku
    const brief = await generateBrief(anthropicKey, signals);

    // Store the brief
    const { data: savedBrief, error } = await supabase
      .from('briefs')
      .insert({
        content: brief.content,
        summary: brief.summary,
        signals_snapshot: signals,
        model: 'claude-3-haiku-20240307',
        tokens_used: brief.tokens_used,
      })
      .select()
      .single();

    if (error) throw error;

    return jsonResponse({
      status: 'generated',
      brief: savedBrief,
    });
  } catch (error) {
    console.error('Brief generation error:', error);
    return jsonResponse({ error: error.message }, 500);
  }
});

async function gatherSignals(): Promise<SignalSnapshot> {
  // In production, this would call real data sources
  // For now, generate realistic market signals
  const baseVolatility = 0.18 + Math.random() * 0.15;
  const baseSentiment = 0.4 + Math.random() * 0.4;
  const baseMomentum = -0.2 + Math.random() * 0.6;
  const fearGreed = Math.floor(30 + Math.random() * 50);

  // Determine market phase based on signals
  let phase: string;
  if (baseVolatility > 0.28) {
    phase = fearGreed < 40 ? 'risk-off' : 'transitional';
  } else if (baseSentiment > 0.65 && baseMomentum > 0.2) {
    phase = 'risk-on';
  } else if (baseSentiment < 0.45) {
    phase = 'cautious';
  } else {
    phase = 'neutral';
  }

  // Detect anomalies
  const anomalies: string[] = [];
  if (baseVolatility > 0.3) anomalies.push('elevated_volatility');
  if (Math.abs(baseMomentum) > 0.35) anomalies.push('momentum_extreme');
  if (fearGreed < 25 || fearGreed > 75) anomalies.push('sentiment_extreme');

  return {
    volatility: Math.round(baseVolatility * 1000) / 1000,
    sentiment: Math.round(baseSentiment * 1000) / 1000,
    momentum: Math.round(baseMomentum * 1000) / 1000,
    fear_greed: fearGreed,
    phase,
    anomalies,
  };
}

function hasSignificantChange(old: SignalSnapshot | null, current: SignalSnapshot): boolean {
  if (!old) return true;

  const volChange = Math.abs(old.volatility - current.volatility);
  const sentChange = Math.abs(old.sentiment - current.sentiment);
  const momChange = Math.abs(old.momentum - current.momentum);
  const fgChange = Math.abs(old.fear_greed - current.fear_greed);
  const phaseChanged = old.phase !== current.phase;

  // Generate new brief if:
  // - Phase changed
  // - Any signal moved >5%
  // - Fear/greed moved >10 points
  return phaseChanged || volChange > 0.05 || sentChange > 0.05 || momChange > 0.05 || fgChange > 10;
}

async function generateBrief(apiKey: string, signals: SignalSnapshot) {
  const prompt = `You are a senior quantitative analyst providing an executive briefing.

Current Market Signals:
- Volatility Index: ${signals.volatility.toFixed(3)} (0.15-0.20 normal, >0.25 elevated)
- Sentiment Score: ${signals.sentiment.toFixed(3)} (0-1 scale, 0.5 neutral)
- Momentum: ${signals.momentum.toFixed(3)} (-1 to 1, positive = bullish)
- Fear & Greed: ${signals.fear_greed}/100 (0=extreme fear, 100=extreme greed)
- Market Phase: ${signals.phase}
- Active Anomalies: ${signals.anomalies.length > 0 ? signals.anomalies.join(', ') : 'none'}

Provide a concise executive brief (3-4 sentences max) covering:
1. Current market regime assessment
2. Key risk factors or opportunities
3. Recommended positioning stance

Be direct and actionable. No fluff.`;

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: 'claude-3-haiku-20240307',
      max_tokens: 300,
      messages: [{ role: 'user', content: prompt }],
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Anthropic API error: ${error}`);
  }

  const data = await response.json();
  const content = data.content[0].text;

  // Extract first sentence as summary
  const summary = content.split('.')[0] + '.';

  return {
    content,
    summary,
    tokens_used: data.usage.input_tokens + data.usage.output_tokens,
  };
}

function jsonResponse(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  });
}
