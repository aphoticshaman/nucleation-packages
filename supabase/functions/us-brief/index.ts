// LatticeForge US Deep Dive Brief - DETERMINISTIC VERSION
// Zero-LLM: All analysis is threshold-based signal processing
// Cost: $0 per brief (was ~$0.045 with Sonnet)
// Deploy: supabase functions deploy us-brief

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts';
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

// Key US indicators to analyze
const US_INDICATORS = [
  'fed_funds_rate',
  'yield_curve_10y2y',
  'vix',
  'oil_wti',
  'usd_eur',
  'us_unemployment',
  'us_cpi',
  'us_m2_money_supply',
  'gdp_growth',
  'inflation',
  'current_account',
  'debt_to_gdp',
];

// Anomaly thresholds with full context
const ANOMALY_CONFIG = {
  vix: {
    label: 'VIX (Fear Index)',
    thresholds: { calm: 15, normal: 20, elevated: 25, high: 30, extreme: 40 },
    interpretation: {
      calm: 'Market complacency - low volatility regime',
      normal: 'Normal market conditions',
      elevated: 'Increased uncertainty - hedging activity rising',
      high: 'Fear spreading - significant volatility',
      extreme: 'Panic conditions - major risk-off event',
    },
  },
  yield_curve_10y2y: {
    label: 'Yield Curve (10Y-2Y)',
    thresholds: { deeply_inverted: -0.5, inverted: 0, flat: 0.5, normal: 1.5, steep: 2.5 },
    interpretation: {
      deeply_inverted: 'Strong recession signal - historically precedes downturns by 12-18 months',
      inverted: 'Recession warning - curve inversion is active',
      flat: 'Caution zone - growth uncertainty',
      normal: 'Healthy curve - normal growth expectations',
      steep: 'Strong growth expected - but watch for inflation',
    },
  },
  us_unemployment: {
    label: 'Unemployment Rate',
    thresholds: { very_low: 3.5, low: 4, moderate: 5, elevated: 6, high: 7, crisis: 10 },
    interpretation: {
      very_low: 'Extremely tight labor market - wage pressure likely',
      low: 'Full employment range - healthy',
      moderate: 'Softening labor market',
      elevated: 'Labor market weakness - potential recession',
      high: 'Significant economic distress',
      crisis: 'Major economic crisis conditions',
    },
  },
  inflation: {
    label: 'Inflation (CPI YoY)',
    thresholds: { deflation: 0, very_low: 1, target: 2, elevated: 3, high: 5, very_high: 8 },
    interpretation: {
      deflation: 'Deflationary risk - Fed may ease aggressively',
      very_low: 'Below target - room for accommodation',
      target: 'At Fed target - optimal conditions',
      elevated: 'Above target - Fed likely vigilant',
      high: 'Problematic inflation - tightening expected',
      very_high: 'Crisis inflation - aggressive tightening',
    },
  },
  debt_to_gdp: {
    label: 'Debt-to-GDP Ratio',
    thresholds: { low: 60, moderate: 90, elevated: 100, high: 120, critical: 140 },
    interpretation: {
      low: 'Fiscal space available',
      moderate: 'Manageable but limiting',
      elevated: 'Fiscal constraints emerging',
      high: 'Significant fiscal risk',
      critical: 'Debt sustainability concerns',
    },
  },
  oil_wti: {
    label: 'Oil Price (WTI)',
    thresholds: { low: 40, normal_low: 60, normal: 80, elevated: 90, high: 110, crisis: 130 },
    interpretation: {
      low: 'Deflationary pressure - demand concerns',
      normal_low: 'Supportive for growth',
      normal: 'Balanced market',
      elevated: 'Inflation pressure building',
      high: 'Economic drag - stagflation risk',
      crisis: 'Supply shock conditions',
    },
  },
  fed_funds_rate: {
    label: 'Fed Funds Rate',
    thresholds: { zero: 0.25, low: 2, neutral: 3, tight: 4, very_tight: 5, restrictive: 6 },
    interpretation: {
      zero: 'Emergency accommodation',
      low: 'Accommodative policy',
      neutral: 'Neutral range',
      tight: 'Restrictive policy',
      very_tight: 'Very restrictive - slowing economy',
      restrictive: 'Highly restrictive - recession risk',
    },
  },
};

// Economic regime classifications
const REGIME_DEFINITIONS = {
  EXPANSION: {
    description: 'Healthy economic growth with controlled inflation',
    indicators: { gdp_growth: '> 2%', unemployment: '< 5%', inflation: '1-3%' },
  },
  LATE_CYCLE: {
    description: 'Growth peaking, inflation rising, Fed tightening',
    indicators: { gdp_growth: '> 1%', unemployment: '< 4.5%', inflation: '> 3%' },
  },
  CONTRACTION: {
    description: 'Economic decline, rising unemployment',
    indicators: { gdp_growth: '< 1%', unemployment: 'rising', inflation: 'falling' },
  },
  RECESSION: {
    description: 'Significant economic decline',
    indicators: { gdp_growth: '< 0%', unemployment: '> 6%' },
  },
  RECOVERY: {
    description: 'Economy rebounding from trough',
    indicators: { gdp_growth: 'rising', unemployment: 'falling', inflation: 'low' },
  },
  STAGFLATION: {
    description: 'Stagnant growth with high inflation',
    indicators: { gdp_growth: '< 2%', inflation: '> 4%', unemployment: '> 5%' },
  },
};

interface USSignals {
  indicators: Record<string, { value: number; date?: string; source: string }>;
  anomalies: Array<{
    indicator: string;
    label: string;
    value: number;
    zone: string;
    severity: 'warning' | 'critical';
    interpretation: string;
  }>;
  regime: {
    classification: keyof typeof REGIME_DEFINITIONS;
    confidence: number;
    supporting_signals: string[];
  };
}

interface USBrief {
  id: string;
  generated_at: string;

  // Current state
  indicators: USSignals['indicators'];
  regime: USSignals['regime'];

  // Anomalies (threshold-triggered)
  anomalies: USSignals['anomalies'];
  anomaly_count: number;
  critical_count: number;

  // Risk assessment (all rule-based)
  risk_factors: Array<{
    risk: string;
    probability: number;
    impact: 'low' | 'medium' | 'high' | 'critical';
    current_indicators: string[];
    warning_signs: string[];
  }>;

  // Rule-based recommendations
  positioning: {
    equity_stance: 'underweight' | 'neutral' | 'overweight';
    duration_stance: 'short' | 'neutral' | 'long';
    sector_tilts: string[];
    hedges: string[];
    rationale: string[];
  };

  // Key events (data-driven)
  watchlist: Array<{
    indicator: string;
    current: number;
    threshold: number;
    direction: 'above' | 'below';
    significance: string;
  }>;

  // Summary
  headline: string;
  bias: 'bullish' | 'bearish' | 'neutral';
  confidence_score: number;
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

    // Gather US signals from database
    const signals = await gatherUSSignals(supabase);

    // Check if we should generate (significant change or scheduled)
    const url = new URL(req.url);
    const force = url.searchParams.get('force') === 'true';

    if (!force && signals.anomalies.length === 0) {
      // Check last brief time
      const { data: lastBrief } = await supabase
        .from('briefs')
        .select('created_at')
        .eq('model', 'latticeforge-fusion-v1')
        .order('created_at', { ascending: false })
        .limit(1)
        .single();

      if (lastBrief) {
        const hoursSinceLastBrief =
          (Date.now() - new Date(lastBrief.created_at).getTime()) / (1000 * 60 * 60);
        if (hoursSinceLastBrief < 6) {
          return jsonResponse({
            status: 'skipped',
            reason: 'No anomalies and last brief was less than 6 hours ago',
            next_scheduled: new Date(
              new Date(lastBrief.created_at).getTime() + 6 * 60 * 60 * 1000
            ).toISOString(),
          });
        }
      }
    }

    // Generate deterministic brief
    const brief = generateDeterministicBrief(signals);

    // Store the brief
    const { data: savedBrief, error } = await supabase
      .from('briefs')
      .insert({
        content: JSON.stringify(brief, null, 2),
        summary: brief.headline,
        signals_snapshot: signals,
        model: 'latticeforge-fusion-v1',
        tokens_used: 0, // Zero LLM tokens
        metadata: {
          type: 'us_deep_dive_deterministic',
          anomaly_count: signals.anomalies.length,
          regime: brief.regime.classification,
          bias: brief.bias,
        },
      })
      .select()
      .single();

    if (error) throw error;

    return jsonResponse({
      status: 'generated',
      brief: savedBrief,
      analysis: brief,
      cost_estimate: '$0.00 (deterministic analysis)',
    });
  } catch (error) {
    console.error('US Brief error:', error);
    return jsonResponse({ error: error.message }, 500);
  }
});

async function gatherUSSignals(supabase: any): Promise<USSignals> {
  // Get latest US indicators from database
  const { data: rawSignals } = await supabase
    .from('country_signals')
    .select('indicator, value, year, source, metadata, updated_at')
    .eq('country_code', 'USA')
    .order('updated_at', { ascending: false });

  const indicators: Record<string, { value: number; date?: string; source: string }> = {};
  const seen = new Set<string>();

  for (const signal of rawSignals || []) {
    if (seen.has(signal.indicator)) continue;
    seen.add(signal.indicator);
    indicators[signal.indicator] = {
      value: parseFloat(signal.value),
      date: signal.metadata?.date || `${signal.year}`,
      source: signal.source,
    };
  }

  // Detect anomalies using thresholds
  const anomalies: USSignals['anomalies'] = [];

  for (const [key, config] of Object.entries(ANOMALY_CONFIG)) {
    const signal = indicators[key];
    if (!signal) continue;

    const value = signal.value;
    const thresholds = config.thresholds as Record<string, number>;

    // Find which zone the value falls into
    let zone = 'unknown';
    let severity: 'warning' | 'critical' = 'warning';

    const sortedThresholds = Object.entries(thresholds).sort((a, b) => a[1] - b[1]);

    for (let i = 0; i < sortedThresholds.length; i++) {
      const [zoneName, threshold] = sortedThresholds[i];
      const nextThreshold = sortedThresholds[i + 1]?.[1] ?? Infinity;

      if (value >= threshold && value < nextThreshold) {
        zone = zoneName;
        break;
      }
    }

    // Determine if this is an anomaly based on zone
    const criticalZones = ['extreme', 'crisis', 'very_high', 'deeply_inverted', 'restrictive', 'critical'];
    const warningZones = ['high', 'elevated', 'inverted', 'very_tight'];

    if (criticalZones.includes(zone)) {
      severity = 'critical';
      anomalies.push({
        indicator: key,
        label: config.label,
        value,
        zone,
        severity,
        interpretation: (config.interpretation as Record<string, string>)[zone] || '',
      });
    } else if (warningZones.includes(zone)) {
      anomalies.push({
        indicator: key,
        label: config.label,
        value,
        zone,
        severity: 'warning',
        interpretation: (config.interpretation as Record<string, string>)[zone] || '',
      });
    }
  }

  // Sort anomalies by severity
  anomalies.sort((a, b) => (a.severity === 'critical' ? -1 : 1));

  // Classify regime
  const regime = classifyRegime(indicators);

  return { indicators, anomalies, regime };
}

function classifyRegime(indicators: Record<string, { value: number }>): USSignals['regime'] {
  const gdp = indicators['gdp_growth']?.value ?? 2;
  const unemployment = indicators['us_unemployment']?.value ?? 5;
  const inflation = indicators['inflation']?.value ?? 2;
  const yieldCurve = indicators['yield_curve_10y2y']?.value ?? 1;

  const supporting: string[] = [];
  let classification: keyof typeof REGIME_DEFINITIONS = 'EXPANSION';
  let confidence = 0.5;

  // Check for recession
  if (gdp < 0) {
    classification = 'RECESSION';
    confidence = 0.9;
    supporting.push(`GDP growth negative (${gdp.toFixed(1)}%)`);
    if (unemployment > 6) {
      supporting.push(`High unemployment (${unemployment.toFixed(1)}%)`);
      confidence = 0.95;
    }
  }
  // Check for stagflation
  else if (gdp < 2 && inflation > 4 && unemployment > 5) {
    classification = 'STAGFLATION';
    confidence = 0.8;
    supporting.push(`Weak growth (${gdp.toFixed(1)}%)`);
    supporting.push(`High inflation (${inflation.toFixed(1)}%)`);
    supporting.push(`Elevated unemployment (${unemployment.toFixed(1)}%)`);
  }
  // Check for late cycle
  else if (unemployment < 4.5 && inflation > 3 && yieldCurve < 0.5) {
    classification = 'LATE_CYCLE';
    confidence = 0.75;
    supporting.push(`Tight labor market (${unemployment.toFixed(1)}%)`);
    supporting.push(`Elevated inflation (${inflation.toFixed(1)}%)`);
    if (yieldCurve < 0) {
      supporting.push('Yield curve inverted');
      confidence = 0.85;
    }
  }
  // Check for contraction
  else if (gdp < 1 && gdp > 0) {
    classification = 'CONTRACTION';
    confidence = 0.65;
    supporting.push(`Slowing growth (${gdp.toFixed(1)}%)`);
  }
  // Check for expansion
  else if (gdp > 2 && unemployment < 5 && inflation < 3.5) {
    classification = 'EXPANSION';
    confidence = 0.8;
    supporting.push(`Healthy growth (${gdp.toFixed(1)}%)`);
    supporting.push(`Low unemployment (${unemployment.toFixed(1)}%)`);
    supporting.push(`Controlled inflation (${inflation.toFixed(1)}%)`);
  }
  // Default to recovery if nothing else fits
  else {
    classification = 'RECOVERY';
    confidence = 0.5;
    supporting.push('Mixed signals - possible recovery');
  }

  return { classification, confidence, supporting_signals: supporting };
}

function generateDeterministicBrief(signals: USSignals): USBrief {
  const { indicators, anomalies, regime } = signals;

  // Build risk factors
  const riskFactors = buildRiskFactors(indicators, anomalies);

  // Generate positioning recommendations
  const positioning = generatePositioning(regime, anomalies, indicators);

  // Build watchlist
  const watchlist = buildWatchlist(indicators);

  // Calculate overall bias
  const criticalCount = anomalies.filter(a => a.severity === 'critical').length;
  const warningCount = anomalies.filter(a => a.severity === 'warning').length;

  let bias: 'bullish' | 'bearish' | 'neutral';
  if (criticalCount >= 2 || ['RECESSION', 'STAGFLATION', 'CONTRACTION'].includes(regime.classification)) {
    bias = 'bearish';
  } else if (criticalCount === 0 && warningCount <= 1 && regime.classification === 'EXPANSION') {
    bias = 'bullish';
  } else {
    bias = 'neutral';
  }

  // Generate headline
  const headline = generateHeadline(regime, anomalies, bias);

  // Confidence based on data quality
  const indicatorCount = Object.keys(indicators).length;
  const confidenceScore = Math.min(0.95, 0.5 + (indicatorCount / 20) * 0.45);

  return {
    id: `us-brief-${Date.now()}`,
    generated_at: new Date().toISOString(),
    indicators,
    regime,
    anomalies,
    anomaly_count: anomalies.length,
    critical_count: criticalCount,
    risk_factors: riskFactors,
    positioning,
    watchlist,
    headline,
    bias,
    confidence_score: confidenceScore,
  };
}

function buildRiskFactors(
  indicators: USSignals['indicators'],
  anomalies: USSignals['anomalies']
): USBrief['risk_factors'] {
  const risks: USBrief['risk_factors'] = [];

  // Recession risk
  const yieldCurve = indicators['yield_curve_10y2y']?.value;
  const hasYieldCurveAnomaly = anomalies.some(a => a.indicator === 'yield_curve_10y2y');

  if (yieldCurve !== undefined && yieldCurve < 0.5) {
    risks.push({
      risk: 'Recession Risk',
      probability: yieldCurve < 0 ? 0.65 : 0.35,
      impact: 'critical',
      current_indicators: [`Yield curve: ${yieldCurve.toFixed(2)}%`],
      warning_signs: [
        'Watch for rising unemployment claims',
        'Monitor ISM manufacturing below 50',
        'Track consumer confidence decline',
      ],
    });
  }

  // Inflation risk
  const inflation = indicators['inflation']?.value;
  const hasInflationAnomaly = anomalies.some(a => a.indicator === 'inflation');

  if (hasInflationAnomaly && inflation !== undefined) {
    risks.push({
      risk: 'Persistent Inflation',
      probability: inflation > 5 ? 0.7 : 0.4,
      impact: 'high',
      current_indicators: [`CPI: ${inflation.toFixed(1)}%`],
      warning_signs: [
        'Watch core services inflation',
        'Monitor wage growth',
        'Track energy prices',
      ],
    });
  }

  // Market volatility risk
  const vix = indicators['vix']?.value;
  if (vix !== undefined && vix > 20) {
    risks.push({
      risk: 'Market Volatility Spike',
      probability: vix > 30 ? 0.6 : 0.3,
      impact: vix > 30 ? 'high' : 'medium',
      current_indicators: [`VIX: ${vix.toFixed(1)}`],
      warning_signs: [
        'Watch credit spreads',
        'Monitor equity put/call ratio',
        'Track overnight funding rates',
      ],
    });
  }

  // Fiscal risk
  const debtGdp = indicators['debt_to_gdp']?.value;
  if (debtGdp !== undefined && debtGdp > 100) {
    risks.push({
      risk: 'Fiscal Sustainability',
      probability: debtGdp > 130 ? 0.4 : 0.2,
      impact: 'high',
      current_indicators: [`Debt/GDP: ${debtGdp.toFixed(0)}%`],
      warning_signs: [
        'Watch treasury auction demand',
        'Monitor interest expense to revenue',
        'Track credit rating actions',
      ],
    });
  }

  // Sort by probability * impact weight
  const impactWeights = { low: 1, medium: 2, high: 3, critical: 4 };
  risks.sort((a, b) => {
    const scoreA = a.probability * impactWeights[a.impact];
    const scoreB = b.probability * impactWeights[b.impact];
    return scoreB - scoreA;
  });

  return risks.slice(0, 4);
}

function generatePositioning(
  regime: USSignals['regime'],
  anomalies: USSignals['anomalies'],
  indicators: USSignals['indicators']
): USBrief['positioning'] {
  const rationale: string[] = [];
  let equityStance: 'underweight' | 'neutral' | 'overweight' = 'neutral';
  let durationStance: 'short' | 'neutral' | 'long' = 'neutral';
  const sectorTilts: string[] = [];
  const hedges: string[] = [];

  const classification = regime.classification;
  const criticalCount = anomalies.filter(a => a.severity === 'critical').length;

  // Equity stance based on regime
  if (classification === 'RECESSION' || classification === 'STAGFLATION') {
    equityStance = 'underweight';
    rationale.push(`${classification} regime favors defensive positioning`);
    sectorTilts.push('Utilities', 'Healthcare', 'Consumer Staples');
    hedges.push('Put protection on indices', 'Long volatility');
  } else if (classification === 'LATE_CYCLE') {
    equityStance = 'neutral';
    rationale.push('Late cycle - selective positioning warranted');
    sectorTilts.push('Quality factor', 'Dividend growers');
    hedges.push('Collar strategies', 'Reduce beta');
  } else if (classification === 'EXPANSION') {
    equityStance = criticalCount > 0 ? 'neutral' : 'overweight';
    rationale.push('Expansion regime supports risk assets');
    sectorTilts.push('Technology', 'Financials', 'Industrials');
  } else {
    equityStance = 'neutral';
    rationale.push('Mixed signals - maintain balanced allocation');
  }

  // Duration stance based on rates/inflation
  const inflation = indicators['inflation']?.value ?? 2;
  const fedFunds = indicators['fed_funds_rate']?.value ?? 3;

  if (inflation > 4 && fedFunds < 5) {
    durationStance = 'short';
    rationale.push('High inflation risk - avoid long duration');
  } else if (inflation < 2 && fedFunds > 4) {
    durationStance = 'long';
    rationale.push('Rate cuts likely - extend duration');
  }

  // Add specific hedges based on anomalies
  for (const anomaly of anomalies) {
    if (anomaly.indicator === 'yield_curve_10y2y' && anomaly.severity === 'critical') {
      hedges.push('Treasury curve steepeners');
    }
    if (anomaly.indicator === 'vix' && anomaly.severity === 'critical') {
      hedges.push('Reduce gross exposure');
    }
  }

  return {
    equity_stance: equityStance,
    duration_stance: durationStance,
    sector_tilts: sectorTilts.slice(0, 4),
    hedges: hedges.slice(0, 3),
    rationale,
  };
}

function buildWatchlist(indicators: USSignals['indicators']): USBrief['watchlist'] {
  const watchlist: USBrief['watchlist'] = [];

  // Yield curve approaching inversion
  const yieldCurve = indicators['yield_curve_10y2y']?.value;
  if (yieldCurve !== undefined && yieldCurve > 0 && yieldCurve < 0.5) {
    watchlist.push({
      indicator: 'Yield Curve',
      current: yieldCurve,
      threshold: 0,
      direction: 'below',
      significance: 'Inversion historically precedes recessions',
    });
  }

  // VIX approaching spike levels
  const vix = indicators['vix']?.value;
  if (vix !== undefined && vix > 18 && vix < 30) {
    watchlist.push({
      indicator: 'VIX',
      current: vix,
      threshold: 30,
      direction: 'above',
      significance: 'Sustained VIX >30 signals major risk-off event',
    });
  }

  // Unemployment rising
  const unemployment = indicators['us_unemployment']?.value;
  if (unemployment !== undefined && unemployment > 4 && unemployment < 5.5) {
    watchlist.push({
      indicator: 'Unemployment',
      current: unemployment,
      threshold: 5.5,
      direction: 'above',
      significance: 'Rising unemployment confirms economic weakness',
    });
  }

  // Inflation approaching critical
  const inflation = indicators['inflation']?.value;
  if (inflation !== undefined && inflation > 3 && inflation < 5) {
    watchlist.push({
      indicator: 'Inflation',
      current: inflation,
      threshold: 5,
      direction: 'above',
      significance: 'Inflation >5% requires aggressive Fed response',
    });
  }

  return watchlist.slice(0, 5);
}

function generateHeadline(
  regime: USSignals['regime'],
  anomalies: USSignals['anomalies'],
  bias: 'bullish' | 'bearish' | 'neutral'
): string {
  const regimeDef = REGIME_DEFINITIONS[regime.classification];
  const criticalCount = anomalies.filter(a => a.severity === 'critical').length;

  if (criticalCount >= 2) {
    return `ELEVATED RISK: ${regime.classification} regime with ${criticalCount} critical signals`;
  }

  if (bias === 'bearish') {
    return `CAUTION: ${regimeDef.description} - ${anomalies.length} warning signals active`;
  }

  if (bias === 'bullish') {
    return `CONSTRUCTIVE: ${regimeDef.description} - favorable conditions`;
  }

  return `MIXED: ${regimeDef.description} - monitor key thresholds`;
}

function jsonResponse(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  });
}
