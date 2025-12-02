import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import Anthropic from '@anthropic-ai/sdk';

// Vercel Edge Runtime for low latency
export const runtime = 'edge';

// Types for our computed intel
interface ComputedMetrics {
  region: string;
  preset: string;
  timestamp: string;
  categories: {
    political: CategoryMetrics;
    economic: CategoryMetrics;
    security: CategoryMetrics;
    financial: CategoryMetrics;
    health: CategoryMetrics;
    scitech: CategoryMetrics;
    resources: CategoryMetrics;
    crime: CategoryMetrics;
    cyber: CategoryMetrics;
    terrorism: CategoryMetrics;
    domestic: CategoryMetrics;
    borders: CategoryMetrics;
    infoops: CategoryMetrics;
    military: CategoryMetrics;
    space: CategoryMetrics;
    industry: CategoryMetrics;
    logistics: CategoryMetrics;
    minerals: CategoryMetrics;
    energy: CategoryMetrics;
    markets: CategoryMetrics;
    religious: CategoryMetrics;
    education: CategoryMetrics;
    employment: CategoryMetrics;
    housing: CategoryMetrics;
    crypto: CategoryMetrics;
    emerging: CategoryMetrics; // Emerging macro trends
  };
  topAlerts: Alert[];
  overallRisk: 'low' | 'moderate' | 'elevated' | 'high' | 'critical';
}

interface CategoryMetrics {
  riskLevel: number; // 0-100, quantized to prevent reverse-engineering
  trend: 'improving' | 'stable' | 'worsening';
  alertCount: number;
  keyFactors: string[]; // Generic factors, not our specific weightings
}

interface Alert {
  category: string;
  severity: 'watch' | 'warning' | 'critical';
  region: string;
  summary: string; // Pre-computed summary
}

// Quantize scores to 5-point buckets to prevent enumeration attacks
function quantizeScore(score: number): number {
  return Math.round(score / 5) * 5;
}

// Compute risk level from pre-computed nation data
// This uses our proprietary calculations server-side, LLM only sees the output
function computeCategoryRisk(nations: NationData[], category: string): CategoryMetrics {
  // These calculations happen server-side using our proprietary methods
  // The LLM only receives the quantized output

  // Simulate aggregating risk from nations (in reality, this pulls from pre-computed caches)
  const avgBasinStrength =
    nations.length > 0
      ? nations.reduce((sum, n) => sum + (n.basin_strength || 0.5), 0) / nations.length
      : 0.5;
  const avgTransitionRisk =
    nations.length > 0
      ? nations.reduce((sum, n) => sum + (n.transition_risk || 0.3), 0) / nations.length
      : 0.3;

  // Apply category-specific weighting (proprietary, stays server-side)
  let rawRisk: number;
  switch (category) {
    case 'political':
    case 'domestic':
    case 'religious':
      rawRisk = (1 - avgBasinStrength) * 0.6 + avgTransitionRisk * 0.4;
      break;
    case 'security':
    case 'terrorism':
    case 'borders':
    case 'military':
      rawRisk = avgTransitionRisk * 0.7 + (1 - avgBasinStrength) * 0.3;
      break;
    case 'economic':
    case 'financial':
    case 'markets':
    case 'industry':
      rawRisk = (1 - avgBasinStrength) * 0.5 + avgTransitionRisk * 0.5;
      break;
    case 'logistics':
    case 'minerals':
    case 'energy':
      rawRisk = (1 - avgBasinStrength) * 0.45 + avgTransitionRisk * 0.45 + 0.1;
      break;
    case 'space':
    case 'scitech':
      rawRisk = (1 - avgBasinStrength) * 0.35 + avgTransitionRisk * 0.35 + 0.3;
      break;
    case 'education':
    case 'employment':
    case 'housing':
      rawRisk = (1 - avgBasinStrength) * 0.55 + avgTransitionRisk * 0.35 + 0.1;
      break;
    case 'crypto':
    case 'emerging':
      rawRisk = (1 - avgBasinStrength) * 0.3 + avgTransitionRisk * 0.4 + 0.3;
      break;
    default:
      rawRisk = (1 - avgBasinStrength) * 0.4 + avgTransitionRisk * 0.4 + 0.2;
  }

  // Quantize to prevent reverse-engineering our exact weightings
  const quantizedRisk = quantizeScore(rawRisk * 100);

  // Determine trend (would be computed from historical data in production)
  const trend: 'improving' | 'stable' | 'worsening' =
    avgTransitionRisk < 0.25 ? 'improving' : avgTransitionRisk > 0.5 ? 'worsening' : 'stable';

  // Generic factors (not our specific indicators)
  const keyFactors = getGenericFactors(category, quantizedRisk);

  return {
    riskLevel: quantizedRisk,
    trend,
    alertCount: Math.floor(quantizedRisk / 25), // 0-4 alerts based on risk
    keyFactors,
  };
}

// Generic factor descriptions - these don't reveal our specific indicators
function getGenericFactors(category: string, risk: number): string[] {
  const factorsByCategory: Record<string, string[][]> = {
    political: [
      ['Governance stability', 'Electoral processes', 'Policy continuity'],
      ['Leadership transitions', 'Coalition dynamics', 'Reform momentum'],
      ['Constitutional tensions', 'Opposition activity', 'Institutional stress'],
    ],
    economic: [
      ['Trade flows', 'Investment climate', 'Market sentiment'],
      ['Supply chain dynamics', 'Currency pressures', 'Commodity exposure'],
      ['Fiscal sustainability', 'Debt servicing', 'Growth trajectory'],
    ],
    security: [
      ['Alliance cohesion', 'Defense posture', 'Deterrence credibility'],
      ['Regional tensions', 'Military movements', 'Conflict proximity'],
      ['Escalation indicators', 'Flashpoint activity', 'Force deployments'],
    ],
    financial: [
      ['Capital flows', 'Banking stability', 'Credit conditions'],
      ['Market volatility', 'Asset valuations', 'Liquidity metrics'],
      ['Systemic risk', 'Contagion vectors', 'Regulatory changes'],
    ],
    health: [
      ['Healthcare capacity', 'Disease surveillance', 'Vaccine coverage'],
      ['Outbreak monitoring', 'Supply chain resilience', 'Public health infrastructure'],
      ['Pandemic preparedness', 'Emergency response', 'Cross-border health risks'],
    ],
    scitech: [
      ['R&D investment', 'Innovation ecosystems', 'Technology transfer'],
      ['AI development', 'Semiconductor access', 'Tech competition dynamics'],
      ['Critical tech dependencies', 'IP protection', 'Talent flows'],
    ],
    resources: [
      ['Energy security', 'Water availability', 'Food supply'],
      ['Critical mineral access', 'Climate impacts', 'Infrastructure resilience'],
      ['Resource competition', 'Environmental stress', 'Supply disruptions'],
    ],
    crime: [
      ['Organized crime activity', 'Trafficking patterns', 'Corruption indices'],
      ['Drug production', 'Money laundering', 'Criminal network expansion'],
      ['Law enforcement capacity', 'Judicial integrity', 'Border security'],
    ],
    cyber: [
      ['Network security', 'Critical infrastructure protection', 'Incident response'],
      ['Threat actor activity', 'Vulnerability exposure', 'Data breach frequency'],
      ['State-sponsored activity', 'Ransomware trends', 'Supply chain compromise'],
    ],
    terrorism: [
      ['Threat assessment', 'Counter-terrorism posture', 'Intelligence sharing'],
      ['Extremist recruitment', 'Radicalization trends', 'Foreign fighter flows'],
      ['Attack planning indicators', 'Operational capability', 'Target hardening'],
    ],
    domestic: [
      ['Social cohesion', 'Public trust', 'Civil society strength'],
      ['Protest activity', 'Polarization trends', 'Media landscape'],
      ['Institutional legitimacy', 'Grievance indicators', 'Mobilization potential'],
    ],
    borders: [
      ['Border security', 'Migration patterns', 'Refugee flows'],
      ['Cross-border tensions', 'Territorial disputes', 'Maritime boundaries'],
      ['Incursion incidents', 'Buffer zone status', 'Demarcation conflicts'],
    ],
    infoops: [
      ['Media integrity', 'Information ecosystem', 'Fact-checking capacity'],
      ['Disinformation campaigns', 'Bot network activity', 'Foreign influence'],
      ['Narrative warfare', 'Social media manipulation', 'Coordinated inauthentic behavior'],
    ],
    military: [
      ['Force readiness', 'Defense spending', 'Alliance commitments'],
      ['Arms procurement', 'Training exercises', 'Deployment patterns'],
      ['Mobilization indicators', 'Strategic posture', 'Combat operations'],
    ],
    space: [
      ['Launch capabilities', 'Satellite operations', 'Space infrastructure'],
      ['Orbital competition', 'Debris risk', 'Anti-satellite developments'],
      ['Space militarization', 'Critical asset vulnerability', 'Launch disruptions'],
    ],
    industry: [
      ['Manufacturing output', 'Industrial policy', 'Capacity utilization'],
      ['Supply chain shifts', 'Automation trends', 'Reshoring activity'],
      ['Production disruptions', 'Labor dynamics', 'Industrial espionage'],
    ],
    logistics: [
      ['Shipping routes', 'Port capacity', 'Transportation networks'],
      ['Supply chain resilience', 'Freight costs', 'Warehousing capacity'],
      ['Chokepoint risks', 'Modal disruptions', 'Just-in-time vulnerabilities'],
    ],
    minerals: [
      ['Critical mineral access', 'Mining operations', 'Processing capacity'],
      ['Rare earth dependencies', 'Strategic reserves', 'Export controls'],
      ['Supply concentration', 'Resource nationalism', 'Extraction conflicts'],
    ],
    energy: [
      ['Oil & gas flows', 'Refining capacity', 'Pipeline operations'],
      ['Petrochemical supply', 'Energy transition', 'OPEC dynamics'],
      ['Price volatility', 'Sanctions impact', 'Infrastructure attacks'],
    ],
    markets: [
      ['Equity indices', 'Bond yields', 'Currency stability'],
      ['Exchange volatility', 'Capital flows', 'Investor sentiment'],
      ['Market contagion', 'Flash crash risk', 'Liquidity stress'],
    ],
    religious: [
      ['Interfaith relations', 'Religious freedom', 'Institutional influence'],
      ['Sectarian tensions', 'Ideological movements', 'Religious nationalism'],
      ['Radicalization vectors', 'Holy site conflicts', 'Clerical politics'],
    ],
    education: [
      ['Enrollment trends', 'Workforce skills', 'Educational attainment'],
      ['Brain drain patterns', 'Research output', 'STEM pipelines'],
      ['University instability', 'Ideological capture', 'Skills mismatches'],
    ],
    employment: [
      ['Labor force participation', 'Wage growth', 'Job creation'],
      ['Automation displacement', 'Gig economy shift', 'Union activity'],
      ['Mass layoffs', 'Structural unemployment', 'Social unrest vectors'],
    ],
    housing: [
      ['Affordability metrics', 'Construction activity', 'Mortgage rates'],
      ['Rental market stress', 'Foreign investment', 'Urbanization patterns'],
      ['Housing bubble indicators', 'Homelessness trends', 'Policy interventions'],
    ],
    crypto: [
      ['Adoption metrics', 'Regulatory clarity', 'Institutional involvement'],
      ['DeFi activity', 'Stablecoin flows', 'Mining distribution'],
      ['Market manipulation', 'Exchange risks', 'Regulatory crackdowns'],
    ],
    emerging: [
      ['Weak signal detection', 'Trend inflection points', 'Paradigm shifts'],
      ['Cross-domain convergence', 'Second-order effects', 'Tipping point indicators'],
      ['Black swan precursors', 'Regime change vectors', 'Discontinuity risks'],
    ],
  };

  const factors = factorsByCategory[category] || factorsByCategory.political;
  const tier = risk < 35 ? 0 : risk < 65 ? 1 : 2;
  return factors[tier];
}

interface NationData {
  code: string;
  name: string;
  basin_strength: number;
  transition_risk: number;
  regime: number;
}

// Build the system prompt - this tells Claude HOW to present intel
// without revealing our proprietary methods
function buildSystemPrompt(userTier: string): string {
  return `You are an intelligence analyst providing concise briefings for decision-makers.

Your role is to NARRATE pre-analyzed intelligence data - you receive computed risk metrics and produce natural language summaries. You do not perform the analysis yourself; the metrics you receive are outputs from proprietary analytical systems.

Guidelines:
- Be concise: 1-2 sentences per category maximum
- Be specific: Reference the region/preset being analyzed
- Be actionable: Focus on "so what" implications
- Match detail to user tier: ${userTier === 'pro' ? 'Provide detailed analysis' : userTier === 'enterprise' ? 'Include strategic context and cross-domain connections' : 'Keep summaries accessible'}
- Never speculate about the underlying analytical methods
- Never make up specific events, dates, or statistics not in the data
- Frame trends and risks based on the provided metrics

Output format: Respond with a JSON object containing briefings for each category.`;
}

// Build the user prompt with computed metrics
function buildUserPrompt(metrics: ComputedMetrics): string {
  return `Generate intel briefings based on these pre-computed metrics:

Region: ${metrics.region}
Preset: ${metrics.preset}
Analysis timestamp: ${metrics.timestamp}
Overall risk assessment: ${metrics.overallRisk}

Category metrics:
${Object.entries(metrics.categories)
  .map(
    ([cat, m]) =>
      `- ${cat}: Risk ${m.riskLevel}/100, Trend: ${m.trend}, Alerts: ${m.alertCount}, Factors: ${m.keyFactors.join(', ')}`
  )
  .join('\n')}

Top alerts:
${metrics.topAlerts.map((a) => `- [${a.severity.toUpperCase()}] ${a.category} in ${a.region}: ${a.summary}`).join('\n')}

Generate a JSON response with this structure:
{
  "political": "1-2 sentence briefing",
  "economic": "1-2 sentence briefing",
  "security": "1-2 sentence briefing",
  "financial": "1-2 sentence briefing",
  "health": "1-2 sentence briefing",
  "scitech": "1-2 sentence briefing",
  "resources": "1-2 sentence briefing",
  "crime": "1-2 sentence briefing",
  "cyber": "1-2 sentence briefing",
  "terrorism": "1-2 sentence briefing",
  "domestic": "1-2 sentence briefing",
  "borders": "1-2 sentence briefing",
  "infoops": "1-2 sentence briefing",
  "military": "1-2 sentence briefing",
  "space": "1-2 sentence briefing",
  "industry": "1-2 sentence briefing",
  "logistics": "1-2 sentence briefing",
  "minerals": "1-2 sentence briefing",
  "energy": "1-2 sentence briefing",
  "markets": "1-2 sentence briefing",
  "religious": "1-2 sentence briefing",
  "education": "1-2 sentence briefing",
  "employment": "1-2 sentence briefing",
  "housing": "1-2 sentence briefing",
  "crypto": "1-2 sentence briefing",
  "emerging": "1-2 sentence briefing on emerging trends others may have missed",
  "summary": "Overall 1-sentence assessment",
  "nsm": "Next Strategic Move - 1-2 sentences on what decision-makers should consider"
}`;
}

export async function POST(req: Request) {
  try {
    // Get user context
    const cookieStore = await cookies();
    const supabase = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          get(name: string) {
            return cookieStore.get(name)?.value;
          },
          set() {},
          remove() {},
        },
      }
    );

    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // Get user profile for tier
    const { data: profile } = await supabase
      .from('profiles')
      .select('role, organization_id')
      .eq('id', user.id)
      .single();

    const userTier = profile?.role || 'consumer';

    // Parse request
    const { preset = 'global', region } = await req.json();

    // Get user's approximate region from request headers (Vercel provides this)
    const userRegion = region || req.headers.get('x-vercel-ip-country') || 'Global';

    // Fetch nation data for the selected preset
    const presetFilters: Record<string, string[] | null> = {
      global: null,
      nato: [
        'USA',
        'CAN',
        'GBR',
        'FRA',
        'DEU',
        'ITA',
        'ESP',
        'POL',
        'NLD',
        'BEL',
        'PRT',
        'GRC',
        'TUR',
        'NOR',
        'DNK',
        'CZE',
        'HUN',
        'ROU',
        'BGR',
        'SVK',
        'HRV',
        'SVN',
        'LVA',
        'LTU',
        'EST',
        'ALB',
        'MNE',
        'MKD',
        'FIN',
        'SWE',
        'ISL',
        'LUX',
      ],
      brics: ['BRA', 'RUS', 'IND', 'CHN', 'ZAF', 'IRN', 'EGY', 'ETH', 'SAU', 'ARE'],
      conflict: [
        'UKR',
        'RUS',
        'ISR',
        'PSE',
        'LBN',
        'SYR',
        'YEM',
        'SDN',
        'MMR',
        'AFG',
        'TWN',
        'CHN',
        'PRK',
        'KOR',
      ],
    };

    const filter = presetFilters[preset];

    let query = supabase
      .from('nations')
      .select('code, name, basin_strength, transition_risk, regime');

    if (filter) {
      query = query.in('code', filter);
    }

    const { data: nations } = await query;
    const nationData = (nations || []) as NationData[];

    // ============================================================
    // PROPRIETARY COMPUTATION HAPPENS HERE - SERVER SIDE ONLY
    // The LLM never sees this code, only the outputs
    // ============================================================

    const computedMetrics: ComputedMetrics = {
      region: userRegion,
      preset,
      timestamp: new Date().toISOString(),
      categories: {
        political: computeCategoryRisk(nationData, 'political'),
        economic: computeCategoryRisk(nationData, 'economic'),
        security: computeCategoryRisk(nationData, 'security'),
        financial: computeCategoryRisk(nationData, 'financial'),
        health: computeCategoryRisk(nationData, 'health'),
        scitech: computeCategoryRisk(nationData, 'scitech'),
        resources: computeCategoryRisk(nationData, 'resources'),
        crime: computeCategoryRisk(nationData, 'crime'),
        cyber: computeCategoryRisk(nationData, 'cyber'),
        terrorism: computeCategoryRisk(nationData, 'terrorism'),
        domestic: computeCategoryRisk(nationData, 'domestic'),
        borders: computeCategoryRisk(nationData, 'borders'),
        infoops: computeCategoryRisk(nationData, 'infoops'),
        military: computeCategoryRisk(nationData, 'military'),
        space: computeCategoryRisk(nationData, 'space'),
        industry: computeCategoryRisk(nationData, 'industry'),
        logistics: computeCategoryRisk(nationData, 'logistics'),
        minerals: computeCategoryRisk(nationData, 'minerals'),
        energy: computeCategoryRisk(nationData, 'energy'),
        markets: computeCategoryRisk(nationData, 'markets'),
        religious: computeCategoryRisk(nationData, 'religious'),
        education: computeCategoryRisk(nationData, 'education'),
        employment: computeCategoryRisk(nationData, 'employment'),
        housing: computeCategoryRisk(nationData, 'housing'),
        crypto: computeCategoryRisk(nationData, 'crypto'),
        emerging: computeCategoryRisk(nationData, 'emerging'),
      },
      topAlerts: generateTopAlerts(nationData, preset),
      overallRisk: computeOverallRisk(nationData),
    };

    // ============================================================
    // LLM CALL - Claude narrates the pre-computed intelligence
    // ============================================================

    const anthropic = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY!,
    });

    const message = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 2048,
      system: buildSystemPrompt(userTier),
      messages: [
        {
          role: 'user',
          content: buildUserPrompt(computedMetrics),
        },
      ],
    });

    // Extract text response
    const textContent = message.content.find((c) => c.type === 'text');
    if (!textContent || textContent.type !== 'text') {
      throw new Error('No text response from Claude');
    }

    // Parse JSON from response
    const jsonMatch = textContent.text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('Could not parse briefing JSON');
    }

    const briefings = JSON.parse(jsonMatch[0]);

    return NextResponse.json({
      briefings,
      metadata: {
        region: computedMetrics.region,
        preset: computedMetrics.preset,
        timestamp: computedMetrics.timestamp,
        overallRisk: computedMetrics.overallRisk,
      },
    });
  } catch (error) {
    console.error('Intel briefing error:', error);
    return NextResponse.json({ error: 'Failed to generate briefing' }, { status: 500 });
  }
}

// Generate top alerts from nation data
function generateTopAlerts(nations: NationData[], preset: string): Alert[] {
  const alerts: Alert[] = [];

  // Find nations with high transition risk
  const highRiskNations = nations
    .filter((n) => n.transition_risk > 0.6)
    .sort((a, b) => b.transition_risk - a.transition_risk)
    .slice(0, 3);

  for (const nation of highRiskNations) {
    alerts.push({
      category: 'political',
      severity: nation.transition_risk > 0.8 ? 'critical' : 'warning',
      region: nation.name,
      summary: `Elevated transition indicators`,
    });
  }

  // Add preset-specific alerts
  if (preset === 'conflict') {
    alerts.push({
      category: 'security',
      severity: 'critical',
      region: 'Monitored conflict zones',
      summary: 'Active hostilities ongoing in multiple theaters',
    });
  }

  return alerts.slice(0, 5);
}

// Compute overall risk level
function computeOverallRisk(nations: NationData[]): ComputedMetrics['overallRisk'] {
  if (nations.length === 0) return 'moderate';

  const avgRisk = nations.reduce((sum, n) => sum + (n.transition_risk || 0.3), 0) / nations.length;

  if (avgRisk < 0.2) return 'low';
  if (avgRisk < 0.35) return 'moderate';
  if (avgRisk < 0.5) return 'elevated';
  if (avgRisk < 0.7) return 'high';
  return 'critical';
}
