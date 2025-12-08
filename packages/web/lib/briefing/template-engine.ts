/**
 * Hybrid Template Engine for Intel Briefings
 *
 * Generates briefing prose from computed metrics WITHOUT LLM calls.
 * Claude is fallback only for:
 * - Critical events requiring nuanced analysis
 * - Novel situations not covered by templates
 * - User explicitly requests "deep analysis"
 *
 * Cost: $0 per request (vs $0.02-0.25 for LLM)
 * Latency: <10ms (vs 2-8s for LLM)
 */

// ============================================================================
// TYPES
// ============================================================================

export interface RiskMetrics {
  riskLevel: number; // 0-100
  trend: 'improving' | 'stable' | 'worsening';
  alertCount: number;
  keyFactors: string[];
}

export interface ComputedMetrics {
  region: string;
  preset: 'global' | 'nato' | 'brics' | 'conflict';
  categories: Record<string, RiskMetrics>;
  topAlerts: Array<{
    category: string;
    severity: 'low' | 'moderate' | 'elevated' | 'high' | 'critical';
    headline: string;
  }>;
  overallRisk: 'low' | 'moderate' | 'elevated' | 'high' | 'critical';
}

export interface BriefingOutput {
  category: string;
  title: string;
  summary: string;
  riskLevel: string;
  trend: string;
  keyPoints: string[];
}

export interface FullBriefing {
  briefings: BriefingOutput[];
  nsm: string; // Next Strategic Move
  generatedBy: 'template-engine' | 'claude';
  timestamp: number;
}

// ============================================================================
// CATEGORY METADATA
// ============================================================================

const CATEGORY_META: Record<
  string,
  {
    title: string;
    domain: string;
    riskDescriptors: Record<string, string>;
    trendDescriptors: Record<string, string>;
    factorTemplates: Record<string, string>;
  }
> = {
  political: {
    title: 'Political Stability',
    domain: 'governance',
    riskDescriptors: {
      low: 'Political institutions remain stable with normal democratic processes.',
      moderate: 'Some political tensions present but within manageable bounds.',
      elevated: 'Heightened political uncertainty requiring close monitoring.',
      high: 'Significant political instability with potential for disruption.',
      critical: 'Severe political crisis with imminent risk of systemic failure.',
    },
    trendDescriptors: {
      improving: 'Stabilization observed as reform efforts gain traction.',
      stable: 'No significant change in political dynamics.',
      worsening: 'Deteriorating conditions as tensions escalate.',
    },
    factorTemplates: {
      basin_weak: 'Institutional foundations showing signs of stress',
      transition_high: 'Elevated probability of regime or policy shifts',
      gdelt_negative: 'Media sentiment trending increasingly critical',
      election_upcoming: 'Electoral cycle introducing uncertainty',
    },
  },
  economic: {
    title: 'Economic Conditions',
    domain: 'economy',
    riskDescriptors: {
      low: 'Economic fundamentals strong with stable growth trajectory.',
      moderate: 'Mixed economic signals but no immediate concerns.',
      elevated: 'Economic pressures building that may require intervention.',
      high: 'Significant economic stress with risk of broader impact.',
      critical: 'Acute economic crisis requiring urgent response.',
    },
    trendDescriptors: {
      improving: 'Recovery indicators strengthening across key metrics.',
      stable: 'Economic conditions holding steady.',
      worsening: 'Mounting pressures on economic stability.',
    },
    factorTemplates: {
      inflation_high: 'Inflationary pressures eroding purchasing power',
      unemployment_high: 'Labor market weakness constraining consumption',
      debt_elevated: 'Sovereign debt levels limiting fiscal flexibility',
      growth_negative: 'Contraction signaling recessionary dynamics',
    },
  },
  security: {
    title: 'Security Environment',
    domain: 'defense',
    riskDescriptors: {
      low: 'Security situation stable with no active threats.',
      moderate: 'Minor security concerns under existing protocols.',
      elevated: 'Increased security alertness warranted.',
      high: 'Active security threats requiring heightened vigilance.',
      critical: 'Imminent security emergency with direct threat to stability.',
    },
    trendDescriptors: {
      improving: 'De-escalation observed as tensions ease.',
      stable: 'Security posture unchanged.',
      worsening: 'Escalating threat environment.',
    },
    factorTemplates: {
      conflict_active: 'Active military operations in progress',
      terrorism_risk: 'Non-state actor threat assessment elevated',
      border_tensions: 'Territorial disputes generating friction',
      arms_buildup: 'Military capacity expansion detected',
    },
  },
  financial: {
    title: 'Financial Markets',
    domain: 'markets',
    riskDescriptors: {
      low: 'Markets functioning normally with healthy liquidity.',
      moderate: 'Some volatility but within historical norms.',
      elevated: 'Market stress indicators flashing caution signals.',
      high: 'Significant market turbulence with contagion risk.',
      critical: 'Financial system under acute stress.',
    },
    trendDescriptors: {
      improving: 'Risk appetite returning as confidence rebuilds.',
      stable: 'Markets trading sideways in consolidation mode.',
      worsening: 'Risk-off sentiment intensifying.',
    },
    factorTemplates: {
      volatility_high: 'Elevated volatility compressing risk appetite',
      credit_tightening: 'Lending conditions restrictive',
      capital_flight: 'Portfolio outflows accelerating',
      currency_pressure: 'Exchange rate weakness adding to instability',
    },
  },
  diplomatic: {
    title: 'Diplomatic Relations',
    domain: 'foreign_policy',
    riskDescriptors: {
      low: 'Diplomatic channels open and functioning effectively.',
      moderate: 'Some diplomatic friction but dialogue continues.',
      elevated: 'Strained relations requiring careful navigation.',
      high: 'Serious diplomatic rupture with limited engagement.',
      critical: 'Near-complete breakdown in diplomatic relations.',
    },
    trendDescriptors: {
      improving: 'Diplomatic thaw underway with renewed engagement.',
      stable: 'Relations at status quo.',
      worsening: 'Diplomatic deterioration accelerating.',
    },
    factorTemplates: {
      sanctions: 'Restrictive measures constraining bilateral ties',
      expulsions: 'Diplomatic personnel reductions signaling discord',
      alliance_stress: 'Partnership commitments under strain',
      summit_scheduled: 'High-level dialogue opportunity approaching',
    },
  },
  energy: {
    title: 'Energy Security',
    domain: 'commodities',
    riskDescriptors: {
      low: 'Energy supplies adequate with diversified sources.',
      moderate: 'Some supply chain concerns but manageable.',
      elevated: 'Energy security pressures building.',
      high: 'Significant supply disruption risk.',
      critical: 'Acute energy crisis threatening economic function.',
    },
    trendDescriptors: {
      improving: 'Supply diversification reducing vulnerability.',
      stable: 'Energy situation unchanged.',
      worsening: 'Supply constraints tightening.',
    },
    factorTemplates: {
      oil_volatile: 'Crude price swings adding to planning uncertainty',
      gas_supply: 'Natural gas availability under pressure',
      transit_risk: 'Energy transport routes facing disruption risk',
      renewable_shift: 'Transition dynamics reshaping market',
    },
  },
  cyber: {
    title: 'Cyber Threats',
    domain: 'technology',
    riskDescriptors: {
      low: 'Cyber threat landscape at baseline levels.',
      moderate: 'Increased scanning and probing activity detected.',
      elevated: 'Targeted cyber campaigns identified.',
      high: 'Active intrusion attempts against critical infrastructure.',
      critical: 'Widespread cyber attack in progress.',
    },
    trendDescriptors: {
      improving: 'Defensive posture strengthened following upgrades.',
      stable: 'Threat activity within normal parameters.',
      worsening: 'Attack surface expanding with new vulnerabilities.',
    },
    factorTemplates: {
      apt_activity: 'Advanced persistent threat groups actively targeting',
      ransomware: 'Ransomware campaigns affecting critical sectors',
      infrastructure: 'Critical infrastructure vulnerabilities identified',
      supply_chain: 'Software supply chain risks elevated',
    },
  },
  trade: {
    title: 'Trade Relations',
    domain: 'commerce',
    riskDescriptors: {
      low: 'Trade flows unimpeded with stable partnerships.',
      moderate: 'Some trade friction but manageable.',
      elevated: 'Trade tensions creating business uncertainty.',
      high: 'Significant trade barriers disrupting commerce.',
      critical: 'Trade war conditions severely impacting flows.',
    },
    trendDescriptors: {
      improving: 'Trade normalization as barriers reduced.',
      stable: 'Trade patterns holding steady.',
      worsening: 'Protectionist measures escalating.',
    },
    factorTemplates: {
      tariffs: 'Tariff measures affecting bilateral trade',
      supply_disruption: 'Supply chain restructuring underway',
      wto_dispute: 'Trade dispute escalation at multilateral level',
      nearshoring: 'Friend-shoring dynamics reshaping flows',
    },
  },
  humanitarian: {
    title: 'Humanitarian Situation',
    domain: 'social',
    riskDescriptors: {
      low: 'Humanitarian conditions stable, basic needs met.',
      moderate: 'Localized humanitarian concerns being addressed.',
      elevated: 'Growing humanitarian pressures in affected areas.',
      high: 'Humanitarian crisis requiring international response.',
      critical: 'Catastrophic humanitarian emergency underway.',
    },
    trendDescriptors: {
      improving: 'Aid delivery improving outcomes.',
      stable: 'Humanitarian situation unchanged.',
      worsening: 'Conditions deteriorating for vulnerable populations.',
    },
    factorTemplates: {
      displacement: 'Forced migration creating refugee flows',
      food_insecurity: 'Food access challenges affecting populations',
      health_crisis: 'Public health emergency straining systems',
      infrastructure_damage: 'Critical infrastructure destruction limiting aid',
    },
  },
  social: {
    title: 'Social Stability',
    domain: 'domestic',
    riskDescriptors: {
      low: 'Social cohesion strong with constructive civic engagement.',
      moderate: 'Some social tensions but dialogue ongoing.',
      elevated: 'Social unrest risks increasing.',
      high: 'Significant civil disturbance and protest activity.',
      critical: 'Widespread social breakdown threatening order.',
    },
    trendDescriptors: {
      improving: 'Social tensions easing through engagement.',
      stable: 'Social dynamics unchanged.',
      worsening: 'Polarization and unrest intensifying.',
    },
    factorTemplates: {
      protests: 'Mass demonstrations reflecting public discontent',
      inequality: 'Economic inequality fueling social friction',
      polarization: 'Political polarization deepening divisions',
      ethnic_tensions: 'Identity-based conflicts emerging',
    },
  },
};

// Fallback for categories not in meta
const DEFAULT_META = {
  title: 'Risk Assessment',
  domain: 'general',
  riskDescriptors: {
    low: 'Risk indicators at baseline levels.',
    moderate: 'Some concerns present requiring monitoring.',
    elevated: 'Elevated risk environment warranting attention.',
    high: 'High risk conditions with potential for escalation.',
    critical: 'Critical situation requiring immediate attention.',
  },
  trendDescriptors: {
    improving: 'Conditions showing improvement.',
    stable: 'Situation unchanged.',
    worsening: 'Conditions deteriorating.',
  },
  factorTemplates: {},
};

// ============================================================================
// RISK LEVEL MAPPING
// ============================================================================

type RiskLevel = 'low' | 'moderate' | 'elevated' | 'high' | 'critical';

function riskScoreToLevel(score: number): RiskLevel {
  if (score <= 20) return 'low';
  if (score <= 40) return 'moderate';
  if (score <= 60) return 'elevated';
  if (score <= 80) return 'high';
  return 'critical';
}

function levelToDisplay(level: string): string {
  const map: Record<string, string> = {
    low: 'LOW',
    moderate: 'MODERATE',
    elevated: 'ELEVATED',
    high: 'HIGH',
    critical: 'CRITICAL',
  };
  return map[level] || level.toUpperCase();
}

function trendToArrow(trend: string): string {
  const map: Record<string, string> = {
    improving: '↓',
    stable: '→',
    worsening: '↑',
  };
  return map[trend] || '→';
}

// ============================================================================
// TEMPLATE GENERATION
// ============================================================================

function generateCategorySummary(
  category: string,
  metrics: RiskMetrics
): string {
  const meta = CATEGORY_META[category] || DEFAULT_META;
  const level = riskScoreToLevel(metrics.riskLevel);

  // Get base descriptor
  let summary = meta.riskDescriptors[level] || DEFAULT_META.riskDescriptors[level];

  // Add trend context
  const trendDesc = meta.trendDescriptors[metrics.trend] || '';
  if (trendDesc) {
    summary += ` ${trendDesc}`;
  }

  return summary;
}

function generateKeyPoints(
  category: string,
  metrics: RiskMetrics
): string[] {
  const meta = CATEGORY_META[category] || DEFAULT_META;
  const points: string[] = [];

  // Include provided key factors
  if (metrics.keyFactors && metrics.keyFactors.length > 0) {
    points.push(...metrics.keyFactors.slice(0, 3));
  }

  // Ensure minimum points
  while (points.length < 2) {
    const level = riskScoreToLevel(metrics.riskLevel);
    if (level === 'low' || level === 'moderate') {
      points.push('Monitoring continues with standard protocols');
    } else {
      points.push('Situation requires enhanced situational awareness');
    }
  }

  return points.slice(0, 4); // Max 4 points
}

function generateCategoryBriefing(
  category: string,
  metrics: RiskMetrics
): BriefingOutput {
  const meta = CATEGORY_META[category] || DEFAULT_META;
  const level = riskScoreToLevel(metrics.riskLevel);

  return {
    category,
    title: meta.title,
    summary: generateCategorySummary(category, metrics),
    riskLevel: `${levelToDisplay(level)} (${metrics.riskLevel}/100)`,
    trend: `${metrics.trend} ${trendToArrow(metrics.trend)}`,
    keyPoints: generateKeyPoints(category, metrics),
  };
}

// ============================================================================
// NSM (NEXT STRATEGIC MOVE) GENERATION
// ============================================================================

interface NsmContext {
  overallRisk: string;
  topCategory: string;
  topCategoryRisk: number;
  preset: string;
  trendBalance: 'improving' | 'stable' | 'worsening';
}

function analyzeNsmContext(metrics: ComputedMetrics): NsmContext {
  // Find highest risk category
  let topCategory = 'political';
  let topCategoryRisk = 0;

  let improvingCount = 0;
  let worseningCount = 0;

  for (const [cat, data] of Object.entries(metrics.categories)) {
    if (data.riskLevel > topCategoryRisk) {
      topCategoryRisk = data.riskLevel;
      topCategory = cat;
    }
    if (data.trend === 'improving') improvingCount++;
    if (data.trend === 'worsening') worseningCount++;
  }

  const trendBalance =
    improvingCount > worseningCount
      ? 'improving'
      : worseningCount > improvingCount
        ? 'worsening'
        : 'stable';

  return {
    overallRisk: metrics.overallRisk,
    topCategory,
    topCategoryRisk,
    preset: metrics.preset,
    trendBalance,
  };
}

const NSM_TEMPLATES: Record<string, Record<string, string> | undefined> = {
  low: {
    global:
      'Maintain standard monitoring posture. Consider opportunistic engagement where bilateral relations can be strengthened.',
    nato:
      'Alliance cohesion remains strong. Focus on capacity building and interoperability exercises.',
    brics:
      'Economic cooperation opportunities available. Monitor for investment entry points.',
    conflict:
      'Window for diplomatic initiatives may be opening. Assess mediation pathways.',
  },
  moderate: {
    global:
      'Increase situational awareness on flagged indicators. Prepare contingency scenarios for key risk areas.',
    nato:
      'Reinforce deterrence messaging while maintaining dialogue channels. Review rapid response readiness.',
    brics:
      'Diversification strategies advisable given mixed signals. Hedge commodity exposures.',
    conflict:
      'De-escalation efforts should be prioritized. Identify credible interlocutors for back-channel engagement.',
  },
  elevated: {
    global:
      'Activate enhanced monitoring protocols. Brief decision-makers on escalation pathways and response options.',
    nato:
      'Forward presence posture may need adjustment. Coordinate intelligence sharing among allies.',
    brics:
      'Risk mitigation takes precedence over opportunity. Review supply chain vulnerabilities.',
    conflict:
      'Escalation management critical. Establish clear red lines while preserving off-ramps.',
  },
  high: {
    global:
      'Crisis protocols warranted. Convene principals for scenario planning. Pre-position response capabilities.',
    nato:
      'Article 4 consultations may be appropriate. Accelerate readiness measures across alliance.',
    brics:
      'Capital preservation mode. Reduce exposure to vulnerable positions. Monitor for contagion.',
    conflict:
      'Active conflict management required. Humanitarian corridors and civilian protection must be prioritized.',
  },
  critical: {
    global:
      'Emergency response activation recommended. All-hands coordination across agencies. Real-time decision cycle.',
    nato:
      'Full alliance alert posture. Collective defense measures under active consideration.',
    brics:
      'Crisis containment priority. Coordinate with multilateral institutions for stabilization.',
    conflict:
      'Immediate ceasefire pressure essential. International intervention framework may be required.',
  },
};

function generateNsm(metrics: ComputedMetrics): string {
  const ctx = analyzeNsmContext(metrics);

  // Get base NSM from templates
  const presetTemplates = NSM_TEMPLATES[ctx.overallRisk] ?? NSM_TEMPLATES['moderate']!;
  let nsm = presetTemplates![ctx.preset] || presetTemplates!['global'];

  // Add category-specific addendum if one category dominates
  if (ctx.topCategoryRisk > 70) {
    const topMeta = CATEGORY_META[ctx.topCategory] || DEFAULT_META;
    nsm += ` Pay particular attention to ${topMeta.domain} developments.`;
  }

  // Add trend context
  if (ctx.trendBalance === 'worsening') {
    nsm += ' Trend indicators suggest situation may deteriorate further without intervention.';
  } else if (ctx.trendBalance === 'improving') {
    nsm += ' Trend indicators cautiously optimistic if current dynamics hold.';
  }

  return nsm;
}

// ============================================================================
// MAIN ENGINE
// ============================================================================

/**
 * Generate full briefing from pre-computed metrics using templates.
 * No LLM call required.
 */
export function generateBriefingFromMetrics(
  metrics: ComputedMetrics
): FullBriefing {
  const briefings: BriefingOutput[] = [];

  // Generate briefing for each category
  for (const [category, categoryMetrics] of Object.entries(metrics.categories)) {
    briefings.push(generateCategoryBriefing(category, categoryMetrics));
  }

  // Sort by risk level descending (critical issues first)
  briefings.sort((a, b) => {
    const aScore = parseInt(a.riskLevel.match(/\d+/)?.[0] || '0');
    const bScore = parseInt(b.riskLevel.match(/\d+/)?.[0] || '0');
    return bScore - aScore;
  });

  return {
    briefings,
    nsm: generateNsm(metrics),
    generatedBy: 'template-engine',
    timestamp: Date.now(),
  };
}

/**
 * Determine if this situation needs Claude (complex synthesis) vs template.
 * Returns true if Claude should be used.
 */
export function shouldUseClaude(metrics: ComputedMetrics): boolean {
  // Critical situations may benefit from nuanced LLM analysis
  if (metrics.overallRisk === 'critical') {
    return true;
  }

  // Many alerts suggest complex multi-factor situation
  const totalAlerts = Object.values(metrics.categories).reduce(
    (sum, c) => sum + c.alertCount,
    0
  );
  if (totalAlerts > 15) {
    return true;
  }

  // If top alerts are critical severity, use Claude
  const criticalAlerts = metrics.topAlerts?.filter(
    (a) => a.severity === 'critical'
  );
  if (criticalAlerts && criticalAlerts.length >= 2) {
    return true;
  }

  // Template engine handles most cases
  return false;
}

/**
 * Get category color for UI rendering
 */
export function getRiskColor(riskLevel: number): string {
  if (riskLevel <= 20) return '#22c55e'; // green
  if (riskLevel <= 40) return '#84cc16'; // lime
  if (riskLevel <= 60) return '#eab308'; // yellow
  if (riskLevel <= 80) return '#f97316'; // orange
  return '#ef4444'; // red
}

/**
 * Format briefing for plain text output (terminal/email)
 */
export function formatBriefingText(briefing: FullBriefing): string {
  const lines: string[] = [];

  lines.push('═══════════════════════════════════════════════════════════════');
  lines.push('                    INTELLIGENCE BRIEFING');
  lines.push('═══════════════════════════════════════════════════════════════');
  lines.push('');

  for (const b of briefing.briefings) {
    lines.push(`▓ ${b.title.toUpperCase()}`);
    lines.push(`  Risk: ${b.riskLevel} | Trend: ${b.trend}`);
    lines.push(`  ${b.summary}`);
    lines.push('  Key Points:');
    for (const point of b.keyPoints) {
      lines.push(`    • ${point}`);
    }
    lines.push('');
  }

  lines.push('───────────────────────────────────────────────────────────────');
  lines.push('NEXT STRATEGIC MOVE:');
  lines.push(briefing.nsm);
  lines.push('───────────────────────────────────────────────────────────────');
  lines.push(`Generated: ${new Date(briefing.timestamp).toISOString()}`);
  lines.push(`Engine: ${briefing.generatedBy}`);

  return lines.join('\n');
}
