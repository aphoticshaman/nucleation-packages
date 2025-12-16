// LatticeForge Intelligence Brief - DETERMINISTIC VERSION
// Zero-LLM: All analysis is mathematical signal processing
// Analysts interpret the structured output
// Deploy: supabase functions deploy intel-brief

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts';
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

// Confidence level definitions (IC standard) - assigned by MATH not LLM
const CONFIDENCE_LEVELS = {
  HIGH: {
    label: 'HIGH CONFIDENCE',
    description: 'Multiple independent sources agree; variance < 10%; strong signal strength',
    threshold: 0.85,
  },
  MODERATE: {
    label: 'MODERATE CONFIDENCE',
    description: 'Sources mostly agree; variance 10-25%; adequate signal strength',
    threshold: 0.6,
  },
  LOW: {
    label: 'LOW CONFIDENCE',
    description: 'Sources diverge; variance > 25%; weak signal strength',
    threshold: 0.3,
  },
  INSUFFICIENT: {
    label: 'INSUFFICIENT DATA',
    description: 'Too few data points for meaningful assessment',
    threshold: 0,
  },
};

// Phase state definitions (from phase-transition model)
const PHASE_STATES = {
  CRYSTALLINE: { label: 'Stable', description: 'Low volatility, mean-reverting regime' },
  SUPERCOOLED: { label: 'Metastable', description: 'Appears stable but susceptible to shocks' },
  NUCLEATING: { label: 'Transitioning', description: 'Active regime change in progress' },
  PLASMA: { label: 'Chaotic', description: 'High energy, unpredictable dynamics' },
  ANNEALING: { label: 'Settling', description: 'Post-transition, new equilibrium forming' },
};

// Indicator thresholds for anomaly detection
const THRESHOLDS = {
  fed_funds_rate: { normal: [2, 5], elevated: [5, 7], critical: [7, Infinity] },
  yield_curve_10y2y: { inverted: [-Infinity, 0], flat: [0, 0.5], normal: [0.5, 2.5] },
  vix: { calm: [0, 15], normal: [15, 25], elevated: [25, 35], extreme: [35, Infinity] },
  us_unemployment: { low: [0, 4], normal: [4, 6], elevated: [6, 8], high: [8, Infinity] },
  inflation: { deflation: [-Infinity, 0], low: [0, 2], target: [2, 3], elevated: [3, 5], high: [5, Infinity] },
  debt_to_gdp: { low: [0, 60], moderate: [60, 90], elevated: [90, 120], critical: [120, Infinity] },
  oil_wti: { low: [0, 50], normal: [50, 80], elevated: [80, 100], high: [100, Infinity] },
};

interface SignalAssessment {
  indicator: string;
  value: number;
  source: string;
  date: string;
  zone: string;
  zscore: number | null;
  anomaly: boolean;
  severity: 'normal' | 'warning' | 'critical';
}

interface IntelBrief {
  id: string;
  generated_at: string;
  source_type: 'OSINT';

  // Key metrics
  signals: SignalAssessment[];
  anomaly_count: number;
  critical_count: number;

  // Phase analysis
  phase_state: keyof typeof PHASE_STATES;
  phase_confidence: number;
  temperature: number;
  order_parameter: number;

  // Confidence assessment (mathematically derived)
  overall_confidence: keyof typeof CONFIDENCE_LEVELS;
  confidence_score: number;
  data_quality: {
    source_count: number;
    recency_score: number;
    coverage_score: number;
  };

  // Risk matrix (threshold-based)
  risk_factors: Array<{
    factor: string;
    current_value: number;
    threshold: number;
    distance_to_threshold: number;
    probability_of_breach: number;
    impact: 'low' | 'medium' | 'high' | 'critical';
  }>;

  // Comparative analysis
  global_comparison: Record<string, {
    country: string;
    indicators: Record<string, number>;
    relative_position: 'leading' | 'lagging' | 'aligned';
  }>;

  // Historical context
  historical_context: {
    similar_periods: string[];
    precedent_outcomes: string[];
  };

  // Structured recommendations (rule-based)
  signals_summary: {
    bullish: string[];
    bearish: string[];
    neutral: string[];
  };
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

    // Gather and analyze signals - ALL DETERMINISTIC
    const brief = await generateDeterministicBrief(supabase);

    // Store the brief
    const { data: savedBrief, error } = await supabase
      .from('briefs')
      .insert({
        content: JSON.stringify(brief, null, 2),
        summary: generateStructuredSummary(brief),
        signals_snapshot: brief.signals,
        model: 'latticeforge-fusion-v1',
        tokens_used: 0, // Zero LLM
        metadata: {
          type: 'intel_brief_deterministic',
          source_type: 'OSINT',
          phase_state: brief.phase_state,
          confidence: brief.overall_confidence,
          anomaly_count: brief.anomaly_count,
          critical_count: brief.critical_count,
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
    console.error('Intel Brief error:', error);
    return jsonResponse({ error: error.message }, 500);
  }
});

async function generateDeterministicBrief(supabase: any): Promise<IntelBrief> {
  // Get US signals
  const { data: usSignals } = await supabase
    .from('country_signals')
    .select('*')
    .eq('country_code', 'USA')
    .order('updated_at', { ascending: false });

  // Get global comparison data
  const { data: globalSignals } = await supabase
    .from('country_signals')
    .select('*')
    .in('country_code', ['CHN', 'DEU', 'JPN', 'GBR', 'FRA', 'IND'])
    .order('updated_at', { ascending: false });

  // Process US signals
  const processedSignals = processSignals(usSignals || []);

  // Calculate phase state
  const phaseAnalysis = calculatePhaseState(processedSignals);

  // Calculate confidence
  const confidenceAnalysis = calculateConfidence(processedSignals, usSignals || []);

  // Build risk matrix
  const riskFactors = buildRiskMatrix(processedSignals);

  // Global comparison
  const globalComparison = buildGlobalComparison(globalSignals || []);

  // Categorize signals
  const signalsSummary = categorizeSignals(processedSignals);

  // Historical context (pattern matching, not LLM)
  const historicalContext = matchHistoricalPatterns(processedSignals);

  const anomalyCount = processedSignals.filter(s => s.anomaly).length;
  const criticalCount = processedSignals.filter(s => s.severity === 'critical').length;

  return {
    id: `intel-${Date.now()}`,
    generated_at: new Date().toISOString(),
    source_type: 'OSINT',

    signals: processedSignals,
    anomaly_count: anomalyCount,
    critical_count: criticalCount,

    phase_state: phaseAnalysis.state,
    phase_confidence: phaseAnalysis.confidence,
    temperature: phaseAnalysis.temperature,
    order_parameter: phaseAnalysis.orderParameter,

    overall_confidence: confidenceAnalysis.level,
    confidence_score: confidenceAnalysis.score,
    data_quality: confidenceAnalysis.quality,

    risk_factors: riskFactors,
    global_comparison: globalComparison,
    historical_context: historicalContext,
    signals_summary: signalsSummary,
  };
}

function processSignals(rawSignals: any[]): SignalAssessment[] {
  const seen = new Map<string, any>();

  // Deduplicate, keeping most recent
  for (const signal of rawSignals) {
    if (!seen.has(signal.indicator)) {
      seen.set(signal.indicator, signal);
    }
  }

  const processed: SignalAssessment[] = [];

  for (const [indicator, signal] of seen) {
    const value = parseFloat(signal.value);
    const thresholdDef = THRESHOLDS[indicator as keyof typeof THRESHOLDS];

    let zone = 'unknown';
    let severity: 'normal' | 'warning' | 'critical' = 'normal';
    let anomaly = false;

    if (thresholdDef) {
      // Find which zone the value falls into
      for (const [zoneName, [min, max]] of Object.entries(thresholdDef)) {
        if (value >= min && value < max) {
          zone = zoneName;
          break;
        }
      }

      // Determine severity based on zone
      if (zone === 'critical' || zone === 'extreme' || zone === 'high' || zone === 'inverted') {
        severity = 'critical';
        anomaly = true;
      } else if (zone === 'elevated' || zone === 'deflation') {
        severity = 'warning';
        anomaly = true;
      }
    }

    processed.push({
      indicator,
      value,
      source: signal.source,
      date: signal.metadata?.date || signal.year?.toString() || 'unknown',
      zone,
      zscore: null, // Would calculate if we had historical data
      anomaly,
      severity,
    });
  }

  return processed;
}

function calculatePhaseState(signals: SignalAssessment[]): {
  state: keyof typeof PHASE_STATES;
  confidence: number;
  temperature: number;
  orderParameter: number;
} {
  // Temperature: measure of volatility/energy
  const criticalCount = signals.filter(s => s.severity === 'critical').length;
  const warningCount = signals.filter(s => s.severity === 'warning').length;
  const totalSignals = signals.length || 1;

  const temperature = (criticalCount * 2 + warningCount) / (totalSignals * 2);

  // Order parameter: how structured vs chaotic
  // Higher when signals are consistent (all bullish or all bearish)
  const anomalyRatio = signals.filter(s => s.anomaly).length / totalSignals;
  const orderParameter = 1 - anomalyRatio;

  // Classify phase
  let state: keyof typeof PHASE_STATES;
  let confidence: number;

  if (temperature > 0.7 && orderParameter < 0.3) {
    state = 'PLASMA';
    confidence = 0.8;
  } else if (temperature < 0.2 && orderParameter > 0.8) {
    state = 'CRYSTALLINE';
    confidence = 0.9;
  } else if (criticalCount >= 2) {
    state = 'NUCLEATING';
    confidence = 0.7;
  } else if (temperature < 0.4 && orderParameter > 0.6 && warningCount > 0) {
    state = 'SUPERCOOLED';
    confidence = 0.6;
  } else {
    state = 'ANNEALING';
    confidence = 0.5;
  }

  return { state, confidence, temperature, orderParameter };
}

function calculateConfidence(
  signals: SignalAssessment[],
  rawSignals: any[]
): {
  level: keyof typeof CONFIDENCE_LEVELS;
  score: number;
  quality: { source_count: number; recency_score: number; coverage_score: number };
} {
  // Source diversity
  const uniqueSources = new Set(rawSignals.map(s => s.source)).size;
  const sourceScore = Math.min(1, uniqueSources / 5);

  // Recency (how recent is the data)
  const now = Date.now();
  const avgAge = rawSignals.reduce((sum, s) => {
    const updated = new Date(s.updated_at).getTime();
    return sum + (now - updated);
  }, 0) / (rawSignals.length || 1);
  const avgAgeDays = avgAge / (1000 * 60 * 60 * 24);
  const recencyScore = Math.max(0, 1 - avgAgeDays / 30); // Decays over 30 days

  // Coverage (how many key indicators we have)
  const keyIndicators = Object.keys(THRESHOLDS);
  const coveredIndicators = signals.filter(s => keyIndicators.includes(s.indicator)).length;
  const coverageScore = coveredIndicators / keyIndicators.length;

  // Overall score
  const score = (sourceScore * 0.3 + recencyScore * 0.4 + coverageScore * 0.3);

  // Determine level
  let level: keyof typeof CONFIDENCE_LEVELS;
  if (score >= CONFIDENCE_LEVELS.HIGH.threshold) {
    level = 'HIGH';
  } else if (score >= CONFIDENCE_LEVELS.MODERATE.threshold) {
    level = 'MODERATE';
  } else if (score >= CONFIDENCE_LEVELS.LOW.threshold) {
    level = 'LOW';
  } else {
    level = 'INSUFFICIENT';
  }

  return {
    level,
    score,
    quality: {
      source_count: uniqueSources,
      recency_score: recencyScore,
      coverage_score: coverageScore,
    },
  };
}

function buildRiskMatrix(signals: SignalAssessment[]): IntelBrief['risk_factors'] {
  const risks: IntelBrief['risk_factors'] = [];

  // Define risk scenarios with thresholds
  const riskDefinitions = [
    { factor: 'Yield Curve Inversion', indicator: 'yield_curve_10y2y', threshold: 0, direction: 'below', impact: 'critical' as const },
    { factor: 'VIX Spike', indicator: 'vix', threshold: 30, direction: 'above', impact: 'high' as const },
    { factor: 'Inflation Surge', indicator: 'inflation', threshold: 4, direction: 'above', impact: 'high' as const },
    { factor: 'Unemployment Rise', indicator: 'us_unemployment', threshold: 5.5, direction: 'above', impact: 'high' as const },
    { factor: 'Oil Shock', indicator: 'oil_wti', threshold: 100, direction: 'above', impact: 'medium' as const },
    { factor: 'Debt Crisis', indicator: 'debt_to_gdp', threshold: 130, direction: 'above', impact: 'critical' as const },
  ];

  for (const def of riskDefinitions) {
    const signal = signals.find(s => s.indicator === def.indicator);
    if (!signal) continue;

    const value = signal.value;
    const distance = def.direction === 'above'
      ? def.threshold - value
      : value - def.threshold;

    // Probability increases as we approach threshold
    const normalizedDistance = distance / Math.abs(def.threshold || 1);
    const probability = Math.max(0, Math.min(1, 1 - normalizedDistance));

    risks.push({
      factor: def.factor,
      current_value: value,
      threshold: def.threshold,
      distance_to_threshold: distance,
      probability_of_breach: probability,
      impact: def.impact,
    });
  }

  // Sort by probability * impact
  const impactWeights = { low: 1, medium: 2, high: 3, critical: 4 };
  risks.sort((a, b) => {
    const scoreA = a.probability_of_breach * impactWeights[a.impact];
    const scoreB = b.probability_of_breach * impactWeights[b.impact];
    return scoreB - scoreA;
  });

  return risks.slice(0, 5); // Top 5 risks
}

function buildGlobalComparison(globalSignals: any[]): IntelBrief['global_comparison'] {
  const comparison: IntelBrief['global_comparison'] = {};

  const byCountry: Record<string, any[]> = {};
  for (const signal of globalSignals) {
    if (!byCountry[signal.country_code]) {
      byCountry[signal.country_code] = [];
    }
    byCountry[signal.country_code].push(signal);
  }

  const countryNames: Record<string, string> = {
    CHN: 'China',
    DEU: 'Germany',
    JPN: 'Japan',
    GBR: 'United Kingdom',
    FRA: 'France',
    IND: 'India',
  };

  for (const [code, signals] of Object.entries(byCountry)) {
    const indicators: Record<string, number> = {};
    const seen = new Set<string>();

    for (const s of signals) {
      if (!seen.has(s.indicator)) {
        seen.add(s.indicator);
        indicators[s.indicator] = parseFloat(s.value);
      }
    }

    // Simple relative position based on GDP growth
    const gdp = indicators['gdp_growth'] || 0;
    const position = gdp > 3 ? 'leading' : gdp < 1 ? 'lagging' : 'aligned';

    comparison[code] = {
      country: countryNames[code] || code,
      indicators,
      relative_position: position as 'leading' | 'lagging' | 'aligned',
    };
  }

  return comparison;
}

function categorizeSignals(signals: SignalAssessment[]): IntelBrief['signals_summary'] {
  const bullish: string[] = [];
  const bearish: string[] = [];
  const neutral: string[] = [];

  for (const signal of signals) {
    const { indicator, value, zone } = signal;

    // Rule-based categorization
    if (indicator === 'yield_curve_10y2y') {
      if (value < 0) bearish.push(`Yield curve inverted (${value.toFixed(2)}%) - recession signal`);
      else if (value > 1) bullish.push(`Yield curve positive (${value.toFixed(2)}%)`);
      else neutral.push(`Yield curve flat (${value.toFixed(2)}%)`);
    }

    if (indicator === 'vix') {
      if (value < 15) bullish.push(`VIX low (${value.toFixed(1)}) - market calm`);
      else if (value > 25) bearish.push(`VIX elevated (${value.toFixed(1)}) - fear rising`);
      else neutral.push(`VIX normal (${value.toFixed(1)})`);
    }

    if (indicator === 'us_unemployment') {
      if (value < 4) bullish.push(`Unemployment low (${value.toFixed(1)}%)`);
      else if (value > 5.5) bearish.push(`Unemployment elevated (${value.toFixed(1)}%)`);
      else neutral.push(`Unemployment moderate (${value.toFixed(1)}%)`);
    }

    if (indicator === 'inflation') {
      if (value > 4) bearish.push(`Inflation high (${value.toFixed(1)}%) - policy tightening risk`);
      else if (value < 1) bearish.push(`Inflation very low (${value.toFixed(1)}%) - deflation risk`);
      else if (value >= 2 && value <= 3) bullish.push(`Inflation at target (${value.toFixed(1)}%)`);
      else neutral.push(`Inflation moderate (${value.toFixed(1)}%)`);
    }

    if (indicator === 'gdp_growth') {
      if (value > 3) bullish.push(`GDP growth strong (${value.toFixed(1)}%)`);
      else if (value < 1) bearish.push(`GDP growth weak (${value.toFixed(1)}%)`);
      else neutral.push(`GDP growth moderate (${value.toFixed(1)}%)`);
    }
  }

  return { bullish, bearish, neutral };
}

function matchHistoricalPatterns(signals: SignalAssessment[]): IntelBrief['historical_context'] {
  // Pattern matching based on current conditions
  const patterns: string[] = [];
  const outcomes: string[] = [];

  const yieldCurve = signals.find(s => s.indicator === 'yield_curve_10y2y');
  const inflation = signals.find(s => s.indicator === 'inflation');
  const unemployment = signals.find(s => s.indicator === 'us_unemployment');

  if (yieldCurve && yieldCurve.value < 0) {
    patterns.push('2006-2007 (pre-GFC)', '2019 (pre-COVID)', '1989 (pre-recession)');
    outcomes.push('Yield curve inversions preceded 7 of last 8 recessions (avg 12-18 month lag)');
  }

  if (inflation && inflation.value > 4) {
    patterns.push('1970s stagflation', '2022 inflation spike');
    outcomes.push('High inflation periods typically require extended tight monetary policy');
  }

  if (unemployment && unemployment.value < 4 && inflation && inflation.value > 3) {
    patterns.push('Late 1960s', 'Late 1990s', '2018-2019');
    outcomes.push('Low unemployment + elevated inflation often precedes Fed tightening');
  }

  if (patterns.length === 0) {
    patterns.push('No strong historical matches');
    outcomes.push('Current conditions do not match known crisis patterns');
  }

  return {
    similar_periods: patterns,
    precedent_outcomes: outcomes,
  };
}

function generateStructuredSummary(brief: IntelBrief): string {
  const phase = PHASE_STATES[brief.phase_state];
  const confidence = CONFIDENCE_LEVELS[brief.overall_confidence];

  const parts = [
    `Phase: ${phase.label} (${brief.phase_state})`,
    `Confidence: ${confidence.label} (${(brief.confidence_score * 100).toFixed(0)}%)`,
    `Anomalies: ${brief.anomaly_count} (${brief.critical_count} critical)`,
  ];

  if (brief.signals_summary.bearish.length > brief.signals_summary.bullish.length) {
    parts.push('Bias: BEARISH');
  } else if (brief.signals_summary.bullish.length > brief.signals_summary.bearish.length) {
    parts.push('Bias: BULLISH');
  } else {
    parts.push('Bias: NEUTRAL');
  }

  return parts.join(' | ');
}

function jsonResponse(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  });
}
