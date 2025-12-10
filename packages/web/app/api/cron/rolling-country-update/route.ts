import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { getLFBMClient } from '@/lib/inference/LFBMClient';
import {
  ALLIANCE_GRAPH,
  TRADE_DEPENDENCIES,
  ENERGY_DEPENDENCIES,
  CONFLICT_PAIRS,
} from '@/lib/signals/LogicalAgent';

export const runtime = 'edge';

// CRON: Rolling Country Update System
//
// CPU-FIRST ARCHITECTURE: Budget-optimized with LogicalAgent
// - Uses alliance/trade/energy graphs for risk propagation (CPU)
// - Only calls LFBM for conflict-pair countries or truly novel situations
// - Expected: 80% reduction in LLM calls vs previous approach
//
// Budget: $5/day target, $10/day max
// - 4 runs/day (every 6 hours)
// - 3 countries per batch = 12 updates/day
// - CPU-first: ~0-2 LFBM calls/day (down from 12)
// - Estimated cost: ~$0.10/day (vs $0.60/day before)
//
// Schedule in vercel.json: "0 0,6,12,18 * * *"

const BATCH_SIZE = 3; // Countries per batch

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
// Only used for conflict-pair countries or truly novel situations
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

// ============================================================
// CPU-FIRST: Graph-based risk propagation (NO LLM needed)
// ============================================================

// Check if a country is part of an active conflict pair
function isConflictCountry(code: string): boolean {
  return CONFLICT_PAIRS.some(([a, b]) => a === code || b === code);
}

// Get allies of a country from the alliance graph
function getAllies(code: string): string[] {
  return ALLIANCE_GRAPH[code] || [];
}

// Get trade partners
function getTradePartners(code: string): string[] {
  return TRADE_DEPENDENCIES[code] || [];
}

// Get energy suppliers
function getEnergySuppliers(code: string): string[] {
  return ENERGY_DEPENDENCIES[code] || [];
}

// CPU-based risk calculation using graph relationships
function calculateCPURiskAdjustment(
  country: NationUpdate,
  allNationRisks: Map<string, number>
): { adjustment: number; confidence: number; reasoning: string } {
  const code = country.code;
  let adjustment = 0;
  const reasons: string[] = [];

  // 1. Alliance cascade - if allies have high risk, propagate some
  const allies = getAllies(code);
  if (allies.length > 0) {
    const allyRisks = allies
      .map(a => allNationRisks.get(a) || 0)
      .filter(r => r > 0.5);
    if (allyRisks.length > 0) {
      const avgAllyRisk = allyRisks.reduce((a, b) => a + b, 0) / allyRisks.length;
      const allyAdjustment = (avgAllyRisk - country.transition_risk) * 0.2; // 20% propagation
      if (allyAdjustment > 0.02) {
        adjustment += allyAdjustment;
        reasons.push(`Alliance pressure from ${allyRisks.length} high-risk allies`);
      }
    }
  }

  // 2. Trade dependency cascade
  const tradePartners = getTradePartners(code);
  if (tradePartners.length > 0) {
    const partnerRisks = tradePartners
      .map(p => allNationRisks.get(p) || 0)
      .filter(r => r > 0.5);
    if (partnerRisks.length > 0) {
      const avgPartnerRisk = partnerRisks.reduce((a, b) => a + b, 0) / partnerRisks.length;
      const tradeAdjustment = (avgPartnerRisk - 0.5) * 0.1; // 10% trade exposure
      if (tradeAdjustment > 0.01) {
        adjustment += tradeAdjustment;
        reasons.push(`Trade exposure to ${partnerRisks.length} stressed economies`);
      }
    }
  }

  // 3. Energy dependency risk
  const energySuppliers = getEnergySuppliers(code);
  if (energySuppliers.length > 0) {
    const supplierRisks = energySuppliers
      .map(s => allNationRisks.get(s) || 0)
      .filter(r => r > 0.5);
    if (supplierRisks.length > 0) {
      const maxSupplierRisk = Math.max(...supplierRisks);
      const energyAdjustment = (maxSupplierRisk - 0.5) * 0.15; // 15% energy exposure
      if (energyAdjustment > 0.01) {
        adjustment += energyAdjustment;
        reasons.push(`Energy supply risk from ${supplierRisks.length} suppliers`);
      }
    }
  }

  // 4. Conflict pair status - direct involvement
  if (isConflictCountry(code)) {
    // Find the conflict partner
    const partner = CONFLICT_PAIRS.find(([a, b]) => a === code || b === code);
    if (partner) {
      const otherCode = partner[0] === code ? partner[1] : partner[0];
      const otherRisk = allNationRisks.get(otherCode) || 0;
      if (otherRisk > 0.5) {
        adjustment += 0.05; // Conflict pairs get baseline elevation
        reasons.push(`Active conflict axis with ${otherCode}`);
      }
    }
  }

  // 5. Stabilization for low-risk isolated countries
  if (allies.length === 0 && adjustment === 0 && country.transition_risk > 0.3) {
    // No allies, no cascade - slight stabilization
    adjustment = -0.02;
    reasons.push('No external pressure vectors');
  }

  // Clamp adjustment
  adjustment = Math.max(-0.1, Math.min(0.1, adjustment));

  // Confidence based on data availability
  const dataPoints = allies.length + tradePartners.length + energySuppliers.length;
  const confidence = Math.min(0.9, 0.5 + dataPoints * 0.05);

  return {
    adjustment,
    confidence,
    reasoning: reasons.length > 0 ? reasons.join('; ') : 'Baseline monitoring',
  };
}

// Decide if a country needs LLM analysis (expensive) or CPU is sufficient
function needsLLMAnalysis(country: NationUpdate, cpuResult: { adjustment: number; confidence: number }): boolean {
  // Use LLM for:
  // 1. Active conflict countries (need real-time assessment)
  if (isConflictCountry(country.code)) return true;

  // 2. Very high risk countries (>80%) - need detailed assessment
  if (country.transition_risk > 0.8) return true;

  // 3. Large unexpected CPU adjustment (something unusual happening)
  if (Math.abs(cpuResult.adjustment) > 0.08) return true;

  // 4. Low confidence CPU result
  if (cpuResult.confidence < 0.6) return true;

  // Otherwise, CPU result is sufficient
  return false;
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

    console.log(`[ROLLING UPDATE] Processing ${staleCountries.length} countries (CPU-first)`);

    // Build nation risk map for graph-based calculations
    const nationRiskMap = new Map<string, number>();
    for (const c of allCountries || []) {
      nationRiskMap.set(c.code, c.transition_risk || 0);
    }

    // Initialize LFBM client (only used for conflict countries)
    const lfbm = getLFBMClient();
    let lfbmCallCount = 0;
    let cpuOnlyCount = 0;

    // Process each country
    for (const country of staleCountries as NationUpdate[]) {
      try {
        // STEP 1: CPU-first risk calculation using graph relationships
        const cpuResult = calculateCPURiskAdjustment(country, nationRiskMap);

        let assessment: {
          summary: string;
          risk_adjustment: number;
          confidence: number;
        };
        let usedLLM = false;

        // STEP 2: Decide if LLM is needed
        if (needsLLMAnalysis(country, cpuResult)) {
          // LLM PATH - Only for conflict countries or high uncertainty
          usedLLM = true;
          lfbmCallCount++;

          const userPrompt = `Analyze the current situation for: ${country.name} (${country.code})

Current metrics:
- Stability score: ${(country.basin_strength * 100).toFixed(0)}%
- Transition risk: ${(country.transition_risk * 100).toFixed(0)}%
- Regime type: ${country.regime}
${isConflictCountry(country.code) ? '- STATUS: Active conflict zone - provide assessment' : ''}

Provide your assessment as JSON.`;

          try {
            const lfbmResponse = await lfbm.generateRaw({
              systemPrompt: COUNTRY_INTEL_SYSTEM_PROMPT,
              userMessage: userPrompt,
              max_tokens: 256,
            });

            const jsonMatch = lfbmResponse.match(/\{[\s\S]*\}/);
            if (!jsonMatch) {
              throw new Error('Could not parse response JSON');
            }

            assessment = JSON.parse(jsonMatch[0]);
          } catch (llmError) {
            // LLM failed, fall back to CPU result
            console.warn(`[ROLLING UPDATE] LLM failed for ${country.code}, using CPU result`);
            assessment = {
              summary: cpuResult.reasoning,
              risk_adjustment: cpuResult.adjustment,
              confidence: cpuResult.confidence,
            };
            usedLLM = false;
          }
        } else {
          // CPU-ONLY PATH - No LLM call needed, huge cost savings
          cpuOnlyCount++;
          assessment = {
            summary: cpuResult.reasoning,
            risk_adjustment: cpuResult.adjustment,
            confidence: cpuResult.confidence,
          };
        }

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
              last_update_source: usedLLM ? 'lfbm' : 'cpu_graph',
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

    const costSavings = cpuOnlyCount > 0 ? ((cpuOnlyCount / (cpuOnlyCount + lfbmCallCount)) * 100).toFixed(0) : '0';

    return NextResponse.json({
      ...results,
      message: `Rolling update complete: ${results.countries_updated}/${staleCountries.length} countries updated`,
      latency_ms: Date.now() - startTime,
      cpu_first_stats: {
        cpu_only_updates: cpuOnlyCount,
        lfbm_calls: lfbmCallCount,
        cost_savings_percent: `${costSavings}%`,
        estimated_cost: `$${(lfbmCallCount * 0.001).toFixed(4)}`, // ~$0.001 per LFBM call
      },
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
