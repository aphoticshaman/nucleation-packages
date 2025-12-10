import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { getLFBMClient } from '@/lib/inference/LFBMClient';

export const runtime = 'edge';

/**
 * CRON: Rolling Country Update System
 *
 * BUDGET-AWARE: Designed for $5/day target spend
 * - Only 4 runs/day (every 6 hours) instead of 48
 * - 3 countries per batch = 12 LFBM calls/day max
 * - Estimated cost: ~$0.60/day (assuming 60s cold start per call)
 *
 * Strategy:
 * - Runs every 6 hours (4 times/day)
 * - Updates 3 high-priority countries per run
 * - Uses CPU-first risk calculation, LFBM only for summary
 *
 * Vercel cron config (vercel.json):
 * {
 *   "crons": [{
 *     "path": "/api/cron/rolling-country-update",
 *     "schedule": "0 */6 * * *"
 *   }]
 * }
 */

const BATCH_SIZE = 3; // Reduced from 5 to control costs

// Priority-based update intervals (hours) - high-risk countries get more frequent updates
const RISK_UPDATE_INTERVALS = {
  high: 2,     // High risk (>70%) - update every 2 hours
  medium: 6,   // Medium risk (40-70%) - update every 6 hours
  low: 12,     // Low risk (<40%) - update every 12 hours
};

// Helper to get update interval based on risk level
function getUpdateIntervalHours(transitionRisk: number): number {
  if (transitionRisk > 0.7) return RISK_UPDATE_INTERVALS.high;
  if (transitionRisk > 0.4) return RISK_UPDATE_INTERVALS.medium;
  return RISK_UPDATE_INTERVALS.low;
}

// System prompt for country-specific intel analysis (Elle persona)
const COUNTRY_INTEL_SYSTEM_PROMPT = `You are Elle, LatticeForge's intelligence analyst, providing brief country assessments.

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
    // Priority-based update: fetch all countries and filter by risk-appropriate intervals
    // High-risk countries get updated more frequently than low-risk
    const { data: allCountries, error: fetchError } = await supabase
      .from('nations')
      .select('code, name, basin_strength, transition_risk, regime, updated_at')
      .order('transition_risk', { ascending: false }); // High risk first

    if (fetchError) {
      throw new Error(`Failed to fetch countries: ${fetchError.message}`);
    }

    // Filter countries based on their risk-appropriate update intervals
    const now = Date.now();
    const staleCountries = (allCountries || [])
      .filter((country) => {
        if (!country.updated_at) return true; // Never updated

        const intervalHours = getUpdateIntervalHours(country.transition_risk || 0);
        const cutoffTime = now - (intervalHours * 60 * 60 * 1000);
        const lastUpdate = new Date(country.updated_at).getTime();

        return lastUpdate < cutoffTime;
      })
      // Sort by: high-risk AND most overdue first
      .sort((a, b) => {
        // Priority score = risk level * hours overdue
        const aInterval = getUpdateIntervalHours(a.transition_risk || 0);
        const bInterval = getUpdateIntervalHours(b.transition_risk || 0);
        const aOverdue = a.updated_at ? (now - new Date(a.updated_at).getTime()) / (aInterval * 60 * 60 * 1000) : 999;
        const bOverdue = b.updated_at ? (now - new Date(b.updated_at).getTime()) / (bInterval * 60 * 60 * 1000) : 999;

        // Higher risk + more overdue = higher priority
        const aPriority = (a.transition_risk || 0) * aOverdue;
        const bPriority = (b.transition_risk || 0) * bOverdue;
        return bPriority - aPriority;
      })
      .slice(0, BATCH_SIZE);

    if (!staleCountries || staleCountries.length === 0) {
      return NextResponse.json({
        ...results,
        message: 'All countries are up to date',
        latency_ms: Date.now() - startTime,
      });
    }

    console.log(`[ROLLING UPDATE] Processing ${staleCountries.length} countries`);

    // Initialize LFBM client
    const lfbm = getLFBMClient();

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

        // Call LFBM for country-specific intel
        const lfbmResponse = await lfbm.generateRaw({
          systemPrompt: COUNTRY_INTEL_SYSTEM_PROMPT,
          userMessage: userPrompt,
          max_tokens: 256,
        });

        // Parse JSON response from LFBM
        const jsonMatch = lfbmResponse.match(/\{[\s\S]*\}/);
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
