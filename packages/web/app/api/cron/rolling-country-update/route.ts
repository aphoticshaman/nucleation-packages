import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import Anthropic from '@anthropic-ai/sdk';

export const runtime = 'edge';

// PRODUCTION-ONLY: Block Anthropic API calls in non-production unless explicitly enabled
function isAnthropicAllowed(): boolean {
  const env = process.env.VERCEL_ENV || process.env.NODE_ENV;
  if (env === 'production') return true;
  if (process.env.ALLOW_ANTHROPIC_IN_DEV === 'true') return true;
  return false;
}

/**
 * CRON: Rolling Country Update System
 *
 * Ensures ALL countries get fresh intel analysis throughout the day,
 * not just when news breaks. This prevents stale data indicators on the map.
 *
 * Strategy:
 * - Runs every 30 minutes (48 times/day)
 * - Updates ~5 countries per run
 * - Prioritizes countries with oldest updated_at timestamps
 * - This ensures ~240 country updates per day (enough for ~200 countries)
 *
 * Vercel cron config (vercel.json):
 * {
 *   "crons": [{
 *     "path": "/api/cron/rolling-country-update",
 *     "schedule": "15,45 * * * *"
 *   }]
 * }
 */

const BATCH_SIZE = 5; // Countries to update per run
const MIN_UPDATE_INTERVAL_HOURS = 12; // Don't update same country within 12 hours

// System prompt for country-specific intel analysis
const COUNTRY_INTEL_SYSTEM_PROMPT = `You are an intelligence analyst providing brief country assessments.

Your task: Given a country name and its current metrics, provide a 1-2 sentence situational awareness update.

Guidelines:
- Be specific to the country
- Focus on actionable intelligence
- Include any notable recent developments if known
- Be factual and objective
- Do not make up specific events or statistics

Output: A single JSON object with:
{
  "summary": "1-2 sentence country situation summary",
  "risk_adjustment": number between -0.1 and 0.1 (adjustment to current risk based on your knowledge),
  "confidence": number between 0.5 and 1.0 (how confident you are in your assessment)
}`;

interface NationUpdate {
  code: string;
  name: string;
  basin_strength: number;
  transition_risk: number;
  regime: number;
  updated_at: string | null;
}

export async function GET(req: Request) {
  const startTime = Date.now();

  // Verify cron secret or Vercel cron header
  const authHeader = req.headers.get('authorization');
  const cronSecret = process.env.CRON_SECRET;
  const isVercelCron = req.headers.get('x-vercel-cron') === '1';

  if (!isVercelCron && cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  // BLOCK non-production Anthropic API calls
  if (!isAnthropicAllowed()) {
    return NextResponse.json({
      success: false,
      error: 'Anthropic API blocked in non-production environment',
      environment: process.env.VERCEL_ENV || process.env.NODE_ENV,
    }, { status: 403 });
  }

  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );

  const results = {
    timestamp: new Date().toISOString(),
    countries_updated: 0,
    countries_skipped: 0,
    errors: [] as string[],
    updates: [] as { code: string; name: string; risk_change: number }[],
  };

  try {
    // Calculate cutoff time - don't update countries updated within MIN_UPDATE_INTERVAL_HOURS
    const cutoffTime = new Date(Date.now() - MIN_UPDATE_INTERVAL_HOURS * 60 * 60 * 1000).toISOString();

    // Get countries needing update (oldest first)
    const { data: staleCountries, error: fetchError } = await supabase
      .from('nations')
      .select('code, name, basin_strength, transition_risk, regime, updated_at')
      .or(`updated_at.is.null,updated_at.lt.${cutoffTime}`)
      .order('updated_at', { ascending: true, nullsFirst: true })
      .limit(BATCH_SIZE);

    if (fetchError) {
      throw new Error(`Failed to fetch countries: ${fetchError.message}`);
    }

    if (!staleCountries || staleCountries.length === 0) {
      return NextResponse.json({
        ...results,
        message: 'All countries are up to date',
        latency_ms: Date.now() - startTime,
      });
    }

    console.log(`[ROLLING UPDATE] Processing ${staleCountries.length} countries`);

    // Initialize Anthropic client
    const anthropic = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY!,
    });

    // Process each country
    for (const country of staleCountries as NationUpdate[]) {
      try {
        // Build prompt for this specific country
        const userPrompt = `Analyze the current situation for: ${country.name} (${country.code})

Current metrics:
- Stability score: ${(country.basin_strength * 100).toFixed(0)}%
- Transition risk: ${(country.transition_risk * 100).toFixed(0)}%
- Regime type: ${country.regime}

Provide your assessment as JSON.`;

        // Call Claude for country-specific intel
        const message = await anthropic.messages.create({
          model: 'claude-haiku-4-5-20251001',
          max_tokens: 256,
          system: COUNTRY_INTEL_SYSTEM_PROMPT,
          messages: [{ role: 'user', content: userPrompt }],
        });

        // Extract response
        const textContent = message.content.find((c) => c.type === 'text');
        if (!textContent || textContent.type !== 'text') {
          throw new Error('No text response');
        }

        // Parse JSON response
        const jsonMatch = textContent.text.match(/\{[\s\S]*\}/);
        if (!jsonMatch) {
          throw new Error('Could not parse response JSON');
        }

        const assessment = JSON.parse(jsonMatch[0]) as {
          summary: string;
          risk_adjustment: number;
          confidence: number;
        };

        // Apply risk adjustment (clamped and weighted by confidence)
        const clampedAdjustment = Math.max(-0.1, Math.min(0.1, assessment.risk_adjustment || 0));
        const weightedAdjustment = clampedAdjustment * (assessment.confidence || 0.5);

        const newTransitionRisk = Math.max(0.02, Math.min(0.95,
          country.transition_risk + weightedAdjustment
        ));

        // Basin strength inversely correlated
        const newBasinStrength = Math.max(0.05, Math.min(0.95,
          country.basin_strength - (weightedAdjustment * 0.5)
        ));

        // Update the country in database
        const { error: updateError } = await supabase
          .from('nations')
          .update({
            transition_risk: Math.round(newTransitionRisk * 1000) / 1000,
            basin_strength: Math.round(newBasinStrength * 1000) / 1000,
            updated_at: new Date().toISOString(),
            metadata: {
              last_intel_summary: assessment.summary,
              last_intel_confidence: assessment.confidence,
              last_intel_timestamp: new Date().toISOString(),
            },
          })
          .eq('code', country.code);

        if (updateError) {
          results.errors.push(`${country.code}: ${updateError.message}`);
        } else {
          results.countries_updated++;
          results.updates.push({
            code: country.code,
            name: country.name,
            risk_change: Math.round(weightedAdjustment * 1000) / 1000,
          });
          console.log(`[ROLLING UPDATE] Updated ${country.name} (${country.code}): risk ${country.transition_risk.toFixed(3)} -> ${newTransitionRisk.toFixed(3)}`);
        }

        // Small delay to avoid rate limits
        await new Promise(resolve => setTimeout(resolve, 500));

      } catch (countryError) {
        const errorMsg = countryError instanceof Error ? countryError.message : 'Unknown error';
        results.errors.push(`${country.code}: ${errorMsg}`);
        console.error(`[ROLLING UPDATE] Error for ${country.code}:`, countryError);
      }
    }

    return NextResponse.json({
      ...results,
      message: `Rolling update complete: ${results.countries_updated}/${staleCountries.length} countries updated`,
      latency_ms: Date.now() - startTime,
    });

  } catch (error) {
    console.error('[ROLLING UPDATE] Fatal error:', error);
    return NextResponse.json(
      {
        error: 'Rolling update failed',
        details: error instanceof Error ? error.message : 'unknown',
        latency_ms: Date.now() - startTime,
      },
      { status: 500 }
    );
  }
}

// Also support POST for manual triggers
export async function POST(req: Request) {
  return GET(req);
}
