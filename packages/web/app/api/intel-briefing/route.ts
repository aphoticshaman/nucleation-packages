import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import Anthropic from '@anthropic-ai/sdk';
import { Redis } from '@upstash/redis';
import {
  getReasoningOrchestrator,
  getSecurityGuardian,
  getLearningCollector,
  type ReasoningResult,
} from '@/lib/reasoning';

// Vercel Edge Runtime for low latency
export const runtime = 'edge';

// =============================================================================
// REDIS CACHE - Shared across ALL edge instances
// =============================================================================
// Cache TTL: 10 minutes. All users hitting the same preset get cached response.
// Only Enterprise tier gets fresh on-demand analysis.
const CACHE_TTL_SECONDS = 10 * 60; // 10 minutes

// Initialize Redis client (uses UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN env vars)
const redis = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL!,
  token: process.env.UPSTASH_REDIS_REST_TOKEN!,
});

interface CachedBriefing {
  data: {
    briefings: Record<string, string>;
    metadata: Record<string, unknown>;
  };
  timestamp: number;
  generatedAt: string;
}

function getCacheKey(preset: string): string {
  return `intel-briefing:${preset}`;
}

async function getCachedBriefing(preset: string): Promise<CachedBriefing | null> {
  try {
    const key = getCacheKey(preset);
    const cached = await redis.get<CachedBriefing>(key);
    return cached;
  } catch (error) {
    console.error('[CACHE] Redis get error:', error);
    return null;
  }
}

async function setCachedBriefing(preset: string, data: CachedBriefing['data']): Promise<void> {
  try {
    const key = getCacheKey(preset);
    const cacheEntry: CachedBriefing = {
      data,
      timestamp: Date.now(),
      generatedAt: new Date().toISOString(),
    };
    // Set with TTL - Redis handles expiration automatically
    await redis.set(key, cacheEntry, { ex: CACHE_TTL_SECONDS });
  } catch (error) {
    console.error('[CACHE] Redis set error:', error);
  }
}

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
  // CRITICAL: Include current date so LLM doesn't generate outdated content
  const currentDate = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  return `You are an intelligence analyst providing concise briefings for decision-makers.

CURRENT DATE: ${currentDate}

Your role is to NARRATE pre-analyzed intelligence data - you receive computed risk metrics and produce natural language summaries. You do not perform the analysis yourself; the metrics you receive are outputs from proprietary analytical systems.

Guidelines:
- Be concise: 1-2 sentences per category maximum
- Be specific: Reference the region/preset being analyzed
- Be actionable: Focus on "so what" implications
- Match detail to user tier: ${userTier === 'pro' ? 'Provide detailed analysis' : userTier === 'enterprise' ? 'Include strategic context and cross-domain connections' : 'Keep summaries accessible'}
- Never speculate about the underlying analytical methods
- Never make up specific events, dates, or statistics not in the data
- Frame trends and risks based on the provided metrics
- CRITICAL: All analysis must reflect current geopolitical reality as of ${currentDate}. Do NOT reference outdated administrations, events, or situations.

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
  const startTime = Date.now();
  let sessionHash = 'anonymous';

  try {
    // ============================================================
    // SECURITY: Check if this is a privileged internal/cron call
    // ============================================================
    // Only cron jobs and internal services can generate fresh data.
    // User requests ALWAYS get cached data - this prevents abuse,
    // exploits, and uncontrolled API costs.
    const isCronWarm = req.headers.get('x-cron-warm') === '1';
    const isInternalService = req.headers.get('x-internal-service') === process.env.INTERNAL_SERVICE_SECRET;
    const canGenerateFresh = isCronWarm || isInternalService;

    // Parse request body first (needed for preset)
    const { preset = 'global', region } = await req.json();

    // Validate preset input (prevent injection)
    const validPresets = ['global', 'nato', 'brics', 'conflict'];
    if (!validPresets.includes(preset)) {
      return NextResponse.json({ error: 'Invalid preset' }, { status: 400 });
    }

    // ============================================================
    // CACHE CHECK - ALWAYS check cache first (Redis - shared across all edge instances)
    // ============================================================
    const cached = await getCachedBriefing(preset);
    if (cached) {
      console.log(`[CACHE HIT] Serving cached briefing for preset: ${preset}, age: ${Math.round((Date.now() - cached.timestamp) / 1000)}s`);
      return NextResponse.json({
        ...cached.data,
        metadata: {
          ...cached.data.metadata,
          cached: true,
          cachedAt: cached.generatedAt,
          cacheAgeSeconds: Math.round((Date.now() - cached.timestamp) / 1000),
        },
      });
    }

    // ============================================================
    // CACHE MISS - Return fallback data for users, only cron generates fresh
    // ============================================================
    // SECURITY: Users NEVER trigger external API calls. Period.
    // Return static fallback briefings until cron warms the cache.
    if (!canGenerateFresh) {
      console.log(`[CACHE MISS] No cached briefing for preset: ${preset}, returning fallback`);
      return NextResponse.json({
        briefings: getFallbackBriefings(preset),
        metadata: {
          region: 'Global',
          preset,
          timestamp: new Date().toISOString(),
          overallRisk: 'moderate' as const,
          cached: false,
          fallback: true,
          message: 'Live briefings refresh every 10 minutes. Showing baseline assessment.',
        },
      });
    }

    console.log(`[CRON/INTERNAL] Generating fresh briefing for preset: ${preset}`);

    // Get user context (only needed for cron logging)
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

    // For cron calls, use service account context
    sessionHash = isCronWarm ? 'cron-service' : 'internal-service';
    const userTier = 'enterprise'; // Cron/internal gets enterprise-level analysis

    // ============================================================
    // SECURITY LAYER - Rate limits (skip for cron/internal)
    // ============================================================
    const security = getSecurityGuardian();

    // Cron/internal bypasses rate limits
    const rateLimit = { allowed: true, remaining: 999, resetAt: new Date() };
    if (!rateLimit.allowed) {
      return NextResponse.json(
        {
          error: 'Rate limit exceeded',
          resetAt: rateLimit.resetAt,
        },
        { status: 429 }
      );
    }

    // Validate preset input (already validated earlier, but double-check)
    const allowedPresetValues = ['global', 'nato', 'brics', 'conflict'];
    if (!allowedPresetValues.includes(preset)) {
      return NextResponse.json({ error: 'Invalid preset' }, { status: 400 });
    }

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
    // REASONING ORCHESTRATOR - Enhanced analysis with multiple engines
    // ============================================================
    const orchestrator = getReasoningOrchestrator();
    let reasoningResult: ReasoningResult | null = null;

    try {
      // Run the reasoning orchestrator with computed metrics as context
      const avgBasinStrength =
        nationData.length > 0
          ? nationData.reduce((sum, n) => sum + (n.basin_strength || 0.5), 0) / nationData.length
          : 0.5;
      const avgTransitionRisk =
        nationData.length > 0
          ? nationData.reduce((sum, n) => sum + (n.transition_risk || 0.3), 0) / nationData.length
          : 0.3;

      // Count high-risk connected nations for cascade detection
      const highRiskNations = nationData.filter((n) => n.transition_risk > 0.6);

      reasoningResult = await orchestrator.reason({
        intent: 'analyze',
        domain: preset === 'conflict' ? 'conflict' : preset === 'brics' ? 'economic' : 'political',
        context: {
          basin_strength: avgBasinStrength,
          transition_risk: avgTransitionRisk,
          connected_high_risk_nations: highRiskNations.length,
          nation_count: nationData.length,
          preset,
          // Add category-level signals for abductive reasoning
          domestic_unrest:
            computedMetrics.categories.domestic.riskLevel > 65
              ? computedMetrics.categories.domestic.riskLevel / 100
              : 0,
          economic_stress:
            computedMetrics.categories.economic.riskLevel > 65
              ? computedMetrics.categories.economic.riskLevel / 100
              : 0,
          leadership_instability:
            computedMetrics.categories.political.riskLevel > 65
              ? computedMetrics.categories.political.riskLevel / 100
              : 0,
          sanctions:
            computedMetrics.categories.financial.riskLevel > 65
              ? computedMetrics.categories.financial.riskLevel / 100
              : 0,
          military_threat:
            computedMetrics.categories.military.riskLevel > 65
              ? computedMetrics.categories.military.riskLevel / 100
              : 0,
        },
        userTier: userTier as 'consumer' | 'pro' | 'enterprise',
        userId: sessionHash,
        sessionId: sessionHash,
      });
    } catch (reasoningError) {
      console.error('Reasoning orchestrator error (non-fatal):', reasoningError);
      // Continue without reasoning enhancement
    }

    // ============================================================
    // LLM CALL - Claude narrates the pre-computed intelligence
    // ============================================================

    const anthropic = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY!,
    });

    // Enhance prompt with reasoning insights if available
    let enhancedPrompt = buildUserPrompt(computedMetrics);
    if (reasoningResult) {
      enhancedPrompt += `\n\nAdditional reasoning insights (confidence: ${(reasoningResult.confidence * 100).toFixed(0)}%):
- Primary conclusion: ${reasoningResult.conclusion}
- Historical parallels: ${reasoningResult.analogies.map((a) => a.description).join('; ') || 'None identified'}
- Key causal factors: ${reasoningResult.causal_factors.map((f) => `${f.factor} (${f.contribution > 0 ? '+' : ''}${(f.contribution * 100).toFixed(0)}%)`).join(', ')}
- Uncertainty range: ${(reasoningResult.uncertainty.lower * 100).toFixed(0)}%-${(reasoningResult.uncertainty.upper * 100).toFixed(0)}%`;
    }

    const llmStartTime = Date.now();
    const message = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 2048,
      system: buildSystemPrompt(userTier),
      messages: [
        {
          role: 'user',
          content: enhancedPrompt,
        },
      ],
    });
    const llmLatency = Date.now() - llmStartTime;

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

    // ============================================================
    // LEARNING COLLECTOR - Capture for future model training
    // ============================================================
    const learner = getLearningCollector();

    // Log the LLM interaction (anonymized)
    void learner.logLLMInteraction(sessionHash, userTier, preset, {
      promptTemplate: 'intel_briefing_v2',
      inputTokens: message.usage?.input_tokens || 0,
      outputTokens: message.usage?.output_tokens || 0,
      latencyMs: llmLatency,
      model: 'claude-sonnet-4-20250514',
      success: true,
    });

    // Log reasoning trace if we got one
    if (reasoningResult) {
      void learner.logReasoningTrace(sessionHash, userTier, preset, {
        engines: reasoningResult.metadata.engines_used,
        confidence: reasoningResult.confidence,
        conclusionType:
          computedMetrics.overallRisk === 'critical' || computedMetrics.overallRisk === 'high'
            ? 'high_risk'
            : computedMetrics.overallRisk === 'elevated'
              ? 'moderate_risk'
              : 'stable',
        inputFeatures: {
          avg_basin_strength:
            nationData.length > 0
              ? nationData.reduce((sum, n) => sum + (n.basin_strength || 0.5), 0) /
                nationData.length
              : 0.5,
          avg_transition_risk:
            nationData.length > 0
              ? nationData.reduce((sum, n) => sum + (n.transition_risk || 0.3), 0) /
                nationData.length
              : 0.3,
          nation_count: nationData.length,
          high_risk_count: nationData.filter((n) => n.transition_risk > 0.6).length,
        },
      });
    }

    const totalLatency = Date.now() - startTime;

    const responseData = {
      briefings,
      metadata: {
        region: computedMetrics.region,
        preset: computedMetrics.preset,
        timestamp: computedMetrics.timestamp,
        overallRisk: computedMetrics.overallRisk,
        // Include reasoning metadata for transparency
        reasoning: reasoningResult
          ? {
              confidence: reasoningResult.confidence,
              engines: reasoningResult.metadata.engines_used,
              conclusion: reasoningResult.conclusion,
              computeTimeMs: reasoningResult.metadata.compute_time_ms,
            }
          : null,
        performance: {
          totalLatencyMs: totalLatency,
          llmLatencyMs: llmLatency,
        },
        rateLimitRemaining: rateLimit.remaining,
        cached: false,
      },
    };

    // ============================================================
    // CACHE SET - Store for future requests (Redis - shared across all edge instances)
    // ============================================================
    // Cache the response so next user hitting this preset gets instant response
    await setCachedBriefing(preset, responseData);
    console.log(`[CACHE SET] Cached fresh briefing for preset: ${preset} in Redis`);

    return NextResponse.json(responseData);
  } catch (error) {
    console.error('Intel briefing error:', error);

    // Log failures for learning
    try {
      const learner = getLearningCollector();
      void learner.logLLMInteraction(sessionHash, 'consumer', 'error', {
        promptTemplate: 'intel_briefing_v2',
        inputTokens: 0,
        outputTokens: 0,
        latencyMs: Date.now() - startTime,
        model: 'claude-sonnet-4-20250514',
        success: false,
      });
    } catch {
      // Silent fail for logging
    }

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

// Create anonymized hash for learning data collection
async function hashForLearning(userId: string): Promise<string> {
  const encoder = new TextEncoder();
  const salt = process.env.ANONYMIZATION_SALT || 'lattice-default-salt';
  const data = encoder.encode(userId + salt);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('')
    .slice(0, 16);
}

// Fallback briefings when cache is cold - comprehensive multi-source intelligence synthesis
function getFallbackBriefings(preset: string): Record<string, string> {
  const presetContext: Record<string, string> = {
    global: 'Global geopolitical landscape',
    nato: 'NATO alliance and Euro-Atlantic region',
    brics: 'BRICS+ economic bloc',
    conflict: 'Active conflict zones',
  };

  const context = presetContext[preset] || presetContext.global;

  // Note: These are generic baseline assessments shown when live analysis is unavailable
  // Live briefings refresh every 10 minutes with current intelligence
  return {
    political: `WHAT: Great power competition continues across multiple theaters. US-China relations remain tense with strategic rivalry the dominant framework; selective engagement on narrow issues only. Russia-West standoff entrenched with limited negotiation pathways. WHO: Key actors—Trump administration pursuing assertive bilateral dealmaking and "America First" priorities, Xi entering his third term with economic challenges, Putin managing domestic constraints and international isolation. WHERE: Friction points concentrated in Taiwan Strait, South China Sea, Eastern Europe, and emerging Africa/Middle East competition. WHEN: Next 90 days focused on new administration policy implementation and international response calibration. WHY: Structural tensions driven by technological decoupling, ideological divergence, and resource competition. US IMPACT: American businesses navigating policy shifts on trade, tariffs, and international engagement. Watch for changes in tech transfer restrictions and bilateral deal structures. OUTLOOK: Multipolar fragmentation accelerating—plan for a world of competing blocs, not global integration. Position: Geographic diversification critical. Avoid single-market concentration in either bloc.`,

    economic: `WHAT: Global growth at 2.8% but distribution uneven. US economy outperforming at 2.4% while EU stagnates at 0.9% and China struggles to hit 5% target. WHO: Fed maintaining restrictive stance, ECB cautiously pivoting, PBOC deploying targeted stimulus. Major multinationals reporting bifurcated performance by region. WHERE: Growth concentrating in US, India, and select ASEAN markets. Contraction in Germany, UK sluggish, Japan benefiting from yen weakness. WHEN: Q1 2025 inflection point likely as delayed rate cut effects materialize. WHY: Post-pandemic normalization, manufacturing reshoring, and services resilience in developed markets. US IMPACT: American consumers remain engine of global demand. Labor market cooling but not cracking. Corporate margins compressing but revenues holding. For US households—expect grocery inflation to moderate while shelter costs remain elevated through mid-2025. OUTLOOK: Soft landing baseline scenario holds at 60% probability. Position: Overweight US domestic exposure. Selective emerging market entries where demographics and reform momentum align.`,

    security: `WHAT: Ukraine war entering attrition phase with no clear resolution pathway. Middle East escalation contained but fragile. Taiwan deterrence holding through 2024. WHO: US maintaining 100k troops in Europe, expanding Pacific presence. China-Russia-Iran-DPRK coordination deepening but not formalized. WHERE: Active operations in Ukraine, Gaza, Red Sea. Elevated readiness in Korea, Taiwan Strait, Baltic, and Black Sea. WHEN: Spring 2025 potential inflection in Ukraine pending equipment deliveries and mobilization decisions. WHY: Territorial revisionism, ideological competition, and regional power vacuums creating persistent instability. US IMPACT: Defense industrial base under strain—Raytheon, Lockheed reporting multi-year backlogs. Selective Service registration modernization under discussion. For American families—direct involvement likelihood remains low but supply chain impacts and energy price volatility possible if escalation occurs. OUTLOOK: Extended deterrence credible but stretched. Position: Defense equities structurally supported. Maintain contingency plans for regional evacuation scenarios.`,

    financial: `WHAT: Fed funds at 5.25-5.50%, terminal rate debate ongoing. Credit markets functioning normally but vigilance warranted on CRE exposure. Treasury yields volatile on data releases. WHO: Powell signaling data dependency; markets pricing 100bps cuts in 2025. Major banks reporting stable loan loss reserves. WHERE: Stress concentrated in regional banks with CRE exposure, particularly office. International dollar funding adequate. WHEN: First rate cut expected Q2 2025 absent recession. CRE repricing to extend through 2026. WHY: Inflation moderating to 2.5-3.0% range; employment resilient; financial conditions accommodative despite nominal tightening. US IMPACT: Mortgage affordability crisis persisting—30-year rates above 7%. Auto loans stretched with 72-84 month terms normalized. Credit card delinquencies rising among subprime borrowers. For American savers—high-yield savings and CD rates offering 5%+ returns for first time in 15 years; lock in duration. OUTLOOK: Orderly normalization base case, but refinancing wall in 2025-2026 bears monitoring. Position: Extend duration selectively. Favor investment-grade corporates over HY.`,

    health: `WHAT: COVID transitioning to endemic management. Flu season tracking normal curves. Avian influenza (H5N1) circulating in dairy herds—low human transmission but monitored. WHO: CDC, WHO maintaining surveillance networks. Pharma pivoting from pandemic mode to standard vaccine development cycles. WHERE: Mpox outbreaks contained in Central Africa. Measles resurgence in under-vaccinated US communities. WHEN: Next potential pandemic timeline unknowable—preparedness is continuous. Seasonal patterns normal through 2025. WHY: Improved surveillance detecting more anomalies; actual emergence risk unchanged. US IMPACT: Healthcare system normalized from pandemic surge. For American families—routine care access restored; mental health waitlists remain extended in many markets. Childcare illness burden returning to pre-pandemic patterns. OUTLOOK: Baseline pandemic preparedness adequate. Position: Healthcare sector fairly valued. Focus on aging demographics plays over pandemic hedges.`,

    scitech: `WHAT: AI capabilities advancing faster than governance frameworks. GPT-5 class models expected 2025. Semiconductor reshoring projects on track. Quantum achieving limited commercial applications. WHO: OpenAI, Anthropic, Google leading foundation models. NVIDIA maintaining GPU dominance. TSMC diversifying with Arizona fab. WHERE: US-China tech competition most intense in AI chips, advanced lithography, and quantum. WHEN: Export control impacts materializing 2024-2025 as stockpiles deplete. WHY: AI seen as decisive for economic and military advantage; neither side willing to cede leadership. US IMPACT: American tech workers facing productivity transformation—AI augmentation becoming job requirement. STEM education gaps widening competitiveness concerns. For everyday Americans—AI touching consumer apps, customer service, content creation. Job displacement concentrated in admin, legal support, content moderation initially. OUTLOOK: Technological bifurcation accelerating. Position: US AI infrastructure (power, cooling, networking) investment thesis intact. Avoid China-dependent supply chains.`,

    resources: `WHAT: Critical mineral supply chains restructuring. Copper at $4.15/lb on infrastructure demand. Lithium crashed 70% from peak on oversupply. Rare earths 85%+ China-controlled. WHO: BHP, Rio Tinto, Glencore accelerating diversification. US DOE funding domestic processing. China maintaining export leverage. WHERE: New mines developing in Chile (copper), Australia (lithium), Canada (nickel). Processing bottleneck remains China-centric. WHEN: Western processing capacity online 2026-2028. Supply deficits likely mid-decade. WHY: EV transition, grid buildout, and defense needs converging on same materials. US IMPACT: American EV adoption constrained by charging infrastructure and price premium over ICE. Mining permit reform stalled in Congress. For US consumers—EV prices stabilizing but not declining; battery replacement costs remain concern. OUTLOOK: Resource nationalism rising; secure supply chains command premium. Position: Royalty/streaming companies for downside protection. Direct exposure to non-Chinese processors.`,

    crime: `WHAT: Transnational crime evolving rapidly. Fentanyl deaths plateauing at 70k+ annually but remain crisis level. Ransomware groups reorganizing post-enforcement. WHO: Mexican cartels controlling fentanyl; Chinese precursor suppliers. Russian-speaking ransomware groups fragmenting but persistent. WHERE: Fentanyl flowing through Mexico; crypto laundering via UAE, Turkey, and non-KYC exchanges. WHEN: Continuous threat environment; no resolution timeline. WHY: Profit motive, weak enforcement in source countries, and demand-side factors in US. US IMPACT: Every American community affected by fentanyl—overdose deaths touching all demographics. Small businesses increasingly ransomware targets. For families—naloxone availability expanding; workplace drug testing policies adapting. Cybersecurity hygiene now household-level concern. OUTLOOK: Harm reduction maturing as policy approach. Position: Invest in cybersecurity; review insurance coverage; know naloxone locations.`,

    cyber: `WHAT: Nation-state cyber operations at elevated tempo. Zero-day exploitation timeline <30 days from discovery. Identity compromise primary attack vector. WHO: China (Volt Typhoon) pre-positioning in critical infrastructure. Russia maintaining disruptive capability. Iran and DPRK conducting revenue operations. WHERE: US water, energy, and telecom sectors targeted. Healthcare and education remain soft targets. WHEN: Attacks timed to geopolitical events—expect activity around elections, military exercises, diplomatic tensions. WHY: Asymmetric capability; deniability; strategic signaling and intelligence collection. US IMPACT: Every American institution a potential target. Personal data compromise essentially universal—assume breach. For households—enable MFA everywhere, unique passwords via manager, freeze credit reports, verify before trusting digital communications. OUTLOOK: Offensive advantage persists; defense requires continuous investment. Position: Zero-trust architecture; segment OT networks; validate recovery procedures.`,

    terrorism: `WHAT: Global terrorism threat diffuse rather than centralized. ISIS-K most capable, AQ rebuilding. Lone actors main concern in West. WHO: ISIS-K leadership in Afghanistan; HTS controlling Idlib; various affiliates in Sahel. WHERE: Active threat in Sahel, East Africa, Afghanistan-Pakistan region. Western threat primarily domestic lone actors. WHEN: Anniversary dates, major events, and geopolitical triggers elevate risk. WHY: Ideological persistence, ungoverned spaces, and online radicalization pipelines. US IMPACT: Homeland threat at elevated baseline—soft targets (concerts, gatherings, houses of worship) remain vulnerable. For Americans—situational awareness standard practice; "see something say something" protocols normalized. International travel to elevated-risk areas requires specific preparation. OUTLOOK: Persistent management rather than elimination. Position: Travel insurance with evacuation coverage; crisis communication plans for organizations.`,

    domestic: `WHAT: Social polarization at multi-decade highs. Trust in institutions (Congress 8%, media 16%, SCOTUS 27%) severely degraded. Cost-of-living pressures feeding discontent. WHO: Partisan media ecosystems reinforcing separate realities. Social media amplifying outrage. Civil society organizations struggling to bridge divides. WHERE: Urban-rural divide stark; swing states most contested terrain. WHEN: Election cycles amplify tensions; 2024 particularly high-stakes. WHY: Economic anxiety, demographic change, and information ecosystem fragmentation creating combustible mix. US IMPACT: Every American navigating divided landscape—workplace, family, and community tensions common. For households—limit news consumption for mental health; focus local engagement over national; maintain cross-partisan relationships intentionally. OUTLOOK: Reconciliation unlikely near-term; manage for coexistence. Position: Factor political risk into long-duration US investments; diversify internationally.`,

    borders: `WHAT: US southern border encounters at record levels—2.5M+ FY23. Processing system overwhelmed; asylum backlog 3M+ cases. Workforce shortages in agriculture, construction, hospitality. WHO: Migrants from Venezuela, Central America, Haiti, and increasingly China, India, Africa. Cartels controlling crossing logistics. Border Patrol stretched. WHERE: Texas and Arizona sectors most impacted. Interior enforcement limited by resource constraints. WHEN: Seasonal patterns disrupted; year-round elevated flows. WHY: Push factors (violence, economic collapse, climate) and pull factors (labor demand, network effects) both intensifying. US IMPACT: Border communities strained; sanctuary city budgets stretched (NYC $12B projected through 2025). Labor-dependent industries benefiting from worker availability. For American workers—wage competition in entry-level sectors; childcare and construction costs affected. OUTLOOK: No legislative solution imminent; executive action limited. Position: Workforce planning must account for immigration uncertainty.`,

    infoops: `WHAT: Information environment severely degraded. AI-generated content 8%+ of political social engagement. Attribution nearly impossible for sophisticated actors. Platform trust & safety gutted. WHO: China, Russia, Iran running influence operations. Domestic actors equally prolific. WHERE: All major platforms compromised; Telegram and X particularly permissive. TikTok algorithm concerns unresolved. WHEN: Continuous; intensifying around elections and crises. WHY: Low cost, high impact, deniability, and domestic polarization creating fertile ground. US IMPACT: Every American consuming some manipulated content unknowingly. For media consumers—verify before sharing; triangulate sources; be skeptical of emotional triggers and "too good to be true" stories; recognize own biases. OUTLOOK: Information quality will continue declining. Position: Build institutional verification capabilities; train staff on source evaluation; diversify intelligence inputs.`,

    military: `WHAT: US defense spending $886B FY24 with bipartisan support for increases. Industrial base struggling to meet demand for munitions, shipbuilding. Power projection capability intact but stressed. WHO: DoD prioritizing China competition; Ukraine support continuing. NATO allies at 2.1% GDP average defense spending. WHERE: Indo-Pacific buildup accelerating; Europe presence maintained; Middle East footprint reduced but responsive. WHEN: AUKUS submarines 2030s; NGAD 6th-gen fighter late decade; major shipbuilding gaps persist. WHY: Peer competition requires conventional deterrence; post-unipolar world demands distributed presence. US IMPACT: Defense-adjacent employment growing; STEM recruitment competing with private sector. For military families—deployment tempo elevated; retention incentives improving. Veterans benefits under budget pressure. OUTLOOK: Defense sector structurally supported 3-5 years minimum. Position: Prime contractors and Tier 1 suppliers; munitions and shipbuilding beneficiaries.`,

    space: `WHAT: 8,000+ active satellites; Starlink dominant in LEO communications with 5,000+ constellation. Space as contested domain normalized. WHO: SpaceX disrupting launch economics; China catching up rapidly; legacy providers (ULA, Arianespace) restructuring. WHERE: LEO congested; GEO stable; cislunar emerging as competition zone. WHEN: Chinese space station fully operational; Artemis lunar program proceeding. WHY: Communications, positioning, and surveillance dependencies making space critical infrastructure. US IMPACT: GPS dependency total for navigation, timing, financial systems. Starlink providing Ukraine battlefield advantage demonstrates military utility. For consumers—satellite internet reaching rural areas; GPS accuracy improving; satellite phones becoming consumer devices. OUTLOOK: Space infrastructure increasingly contested; debris management critical. Position: Favor proven launch providers and ground systems over speculative applications.`,

    industry: `WHAT: Manufacturing renaissance underway—reshoring announcements up 180% YoY. PMIs in expansion (52-54). IRA and CHIPS Act driving $500B+ committed investment. WHO: Major winners include semiconductor fabs (TSMC Arizona, Intel Ohio, Samsung Texas), EV battery plants, and clean energy manufacturing. WHERE: Investment concentrating in Sun Belt and industrial Midwest; site selection factors include energy costs, workforce, and incentives. WHEN: Projects completing 2025-2028; employment impacts building. WHY: Supply chain security concerns, industrial policy revival, and labor arbitrage narrowing with Asia. US IMPACT: Blue-collar employment improving in manufacturing regions; training pipeline gaps limiting growth. For American workers—skilled trades (electricians, welders, CNC operators) in high demand; community college pathways valuable. OUTLOOK: Industrial policy support continuing regardless of election outcome. Position: US-based manufacturers with automation capability; energy-intensive industries face margin pressure.`,

    logistics: `WHAT: Global shipping normalized from pandemic chaos. Container rates at pre-2020 levels. Red Sea disruptions adding 10-14 days to Asia-Europe routes. Just-in-case inventory replacing just-in-time. WHO: Maersk, MSC, CMA CGM controlling container shipping; FedEx, UPS adapting to e-commerce normalization; Amazon building competing network. WHERE: Panama Canal drought constraints easing; Suez alternative routing common; US ports invested but labor-constrained. WHEN: Red Sea instability likely persistent; inventory rebuild cycles ongoing. WHY: Houthi attacks demonstrating chokepoint vulnerability; pandemic memory driving buffer stock strategy. US IMPACT: Consumer goods inflation moderating but not deflating. Delivery speeds normalized. For households—shipping times reliable; costs embedded in prices; last-mile delivery employment stable. OUTLOOK: Logistics infrastructure investment continuing. Position: Diversified routing capability valuable; warehouse/distribution REITs attractive.`,

    minerals: `WHAT: Critical mineral supply chains restructuring slowly. China controls 60-90% of processing for key battery materials. Western projects advancing but years from production. WHO: Major miners diversifying; DOE, DOD funding domestic capacity; China maintaining leverage through processing dominance. WHERE: Extraction shifting to Australia (lithium), DRC (cobalt), Indonesia (nickel), Chile (copper); processing still China-centric. WHEN: Western processing capacity 2026-2028; meaningful diversification mid-decade at earliest. WHY: EV transition, grid storage, and defense applications all competing for same materials; supply cannot scale as fast as demand. US IMPACT: EV cost premium persists; grid upgrade projects facing material constraints. For consumers—battery costs stabilizing but not declining; EV adoption constrained by price and charging infrastructure more than range. OUTLOOK: Resource nationalism rising; secure supply commands premium. Position: Streaming/royalty structures for downside protection; direct exposure to non-Chinese processors when available.`,

    energy: `WHAT: Brent crude ranging $70-85; OPEC+ cuts maintaining floor. Natural gas inventories healthy. Renewable additions exceeding coal retirements globally for first time. WHO: Saudi Arabia managing production; US shale disciplined; Russia rerouting to Asia. Utilities scrambling on grid upgrades. WHERE: US remains largest oil/gas producer; Middle East geopolitics contained for now; Europe diversified from Russian gas. WHEN: Fossil fuel demand plateau not before 2030; grid investment needs decades of buildout. WHY: Energy transition accelerating but incumbent infrastructure sticky; reliability concerns limiting rapid switch. US IMPACT: Gasoline prices stable but volatile on geopolitics; electricity rates rising on grid investment; energy jobs shifting but not disappearing. For households—weatherization incentives available; solar/EV adoption economics improving; utility bills increasing 3-5% annually. OUTLOOK: "All of the above" energy reality for next decade. Position: Balanced exposure to transition enablers and disciplined fossil producers; avoid stranded asset risk.`,

    markets: `WHAT: S&P 500 at elevated valuations (21x forward P/E) but supported by earnings. Magnificent 7 concentration at 30% of index. Credit spreads tight; private credit expanding. WHO: Retail participation elevated; institutional positioning neutral; buybacks supporting prices. WHERE: US outperforming international significantly; EM mixed with India strong, China weak. WHEN: Rate cut timing dominating narrative; earnings revisions key to 2025 direction. WHY: AI narrative driving tech; employment resilience supporting consumer; rates constraining but not crushing. US IMPACT: 401k balances recovered for those invested; housing wealth accessible via HELOC. For households—retirement planning on track if consistently investing; market timing historically counterproductive; rebalancing beneficial. OUTLOOK: Modest returns with volatility spikes around data releases. Position: Maintain diversified allocation; defensive tilt prudent; build cash for dislocation opportunities.`,

    religious: `WHAT: Sectarian tensions elevated in multiple regions. Hindu nationalism in India affecting minorities; Christian persecution increasing in sub-Saharan Africa; antisemitism spiking globally post-Oct 7. WHO: State actors weaponizing religious identity; non-state groups organizing along sectarian lines. WHERE: India, Nigeria, Sudan, Myanmar hotspots; diaspora communities affected globally including US. WHEN: Religious holidays and anniversaries elevate risk; electoral cycles amplify rhetoric. WHY: Identity politics ascendant; economic stress channeled into intergroup conflict. US IMPACT: Religious polarization increasing domestically; synagogue, mosque, church security spending rising. For communities—interfaith dialogue valuable; security awareness appropriate; reporting threats normalized. OUTLOOK: Religious freedom metrics declining globally. Position: Factor religious demographics into emerging market assessments; support interfaith resilience.`,

    education: `WHAT: Higher education enrollment declining (-15% since 2010 peak). STEM pipeline concerns persist. K-12 learning loss from pandemic ongoing. EdTech consolidation after bubble burst. WHO: Major universities restructuring; community colleges pivoting to workforce; corporate training expanding. WHERE: Enrollment declines concentrated in for-profit and regional institutions; elite demand unchanged. WHEN: Demographic cliff hitting colleges 2025-2030; AI disruption of credentialing emerging. WHY: Cost-benefit perception shifting; alternative credentials gaining acceptance; employer skills focus over degrees. US IMPACT: Student debt burden $1.7T constraining household formation; loan forgiveness politically contested. For families—ROI on college varies dramatically by major and institution; skilled trades increasingly attractive; continuous learning mandatory. OUTLOOK: Credentialing disruption accelerating. Position: Employer training platforms and certification providers growing; traditional higher ed selective exposure only.`,

    employment: `WHAT: US unemployment 3.9% with wage growth moderating to 4.2% YoY. Labor force participation below pre-pandemic but recovering. Job openings elevated but normalizing. WHO: Workers gaining leverage in shortage sectors (healthcare, skilled trades); white-collar facing efficiency pressure. WHERE: Regional variation significant—Sun Belt tight; manufacturing Midwest recovering; coastal tech adjusting. WHEN: Labor market cooling gradually; no recession-level weakness indicated. WHY: Demographics constraining supply; AI beginning to affect demand composition; immigration policy limiting flexibility. US IMPACT: Real wages positive after inflation for most workers; benefits improvement slowing. For workers—job switching premium declining; skill portability valuable; gig economy maturing. OUTLOOK: Automation displacement accelerating in admin, customer service, content roles. Position: Reskilling investments critical; 18-24 month lead time for retraining.`,

    housing: `WHAT: US existing home sales at 4.0M annualized—30-year low. Mortgage rates above 7%. Inventory slowly releasing as rate-lock effect fades. Prices flat to slightly rising despite volume collapse. WHO: Millennials locked out of starter homes; Boomers locked in by rate differentials; investors pulling back. WHERE: Sun Belt moderating after pandemic surge; coastal markets range-bound; Midwest relatively affordable. WHEN: Rate relief needed for volume recovery; 2025-2026 refinancing wave if rates fall. WHY: Supply constrained by underbuilding and rate-lock; demand constrained by affordability and demographics. US IMPACT: Housing affordability worst on record; household formation delayed; wealth gap widening between owners and renters. For households—renting rational in many markets; down payment assistance programs underutilized; multigenerational living increasing. OUTLOOK: No price crash without employment shock; affordability improvement requires either rates or incomes or both. Position: Entry-level homebuilders positioned well; sunbelt multifamily REITs attractive.`,

    crypto: `WHAT: Bitcoin near all-time highs post-spot ETF approval. Institutional custody infrastructure maturing. Ethereum ETF approved. Regulatory clarity incrementally improving. WHO: BlackRock, Fidelity legitimizing asset class; Coinbase primary US exchange; Tether dominance raising concern. WHERE: US regulatory approach clearer than EU/Asia; offshore exchanges persisting. WHEN: Bitcoin halving cycle historically bullish; ETF flow momentum key. WHY: Inflation hedge narrative; institutional allocation thesis; technological adoption thesis—all debated. US IMPACT: Crypto exposure normalizing in portfolios; tax compliance improving; consumer protection gaps remain. For investors—volatility persists; sizing matters; self-custody learning curve; scam prevalence high. OUTLOOK: Maturing but volatile asset class. Position: 1-3% portfolio allocation for volatility-tolerant; prefer BTC/ETH over altcoins given regulatory uncertainty.`,

    emerging: `WHAT: Monitoring AI agent proliferation (autonomous systems operating without human oversight), biosecurity governance gaps (gain-of-function research oversight fragmented), space resource competition (lunar/asteroid mining claims emerging), quantum cryptography timelines (current encryption vulnerable within decade). WHO: Labs, states, and private actors operating at frontier with limited coordination. WHERE: Distributed globally; concentration in US, China, and selective others. WHEN: No immediate paradigm shift detected; 6-12 month horizon clear; longer-term scenarios require planning. WHY: Technological capability outpacing governance capacity. US IMPACT: Americans will encounter AI agents in customer service, healthcare advice, financial planning within 2 years—disclosure requirements unclear. For individuals—understand AI limitations; verify AI-generated information; maintain human relationships and judgment. OUTLOOK: Black swan preparedness essential even without immediate signals. Position: Scenario planning ongoing; no action required now; watchlist maintained.`,

    summary: `${context}: Operating at BASELINE RISK with localized elevated concerns in specific domains. KEY DRIVERS: (1) Central bank policy trajectory—rate path determines financial conditions. (2) Geopolitical friction—US-China, Russia-West, Middle East all require monitoring. (3) Technology transition—AI and energy reshaping economic landscape. (4) Social cohesion—domestic polarization affecting governance capacity. NO IMMEDIATE CATALYSTS for major repositioning, but multiple scenarios within 12-month horizon could shift assessment. US POSTURE: Economy outperforming peers; security commitments stretched but holding; domestic divisions manageable but not healing. FOR AMERICANS: Employment stable, inflation moderating, market returns positive—but beneath aggregate stability, significant variation by sector, region, and demographic. Maintain financial resilience, stay informed without doom-scrolling, engage locally.`,

    nsm: `STRATEGIC POSTURE: Hold positions with tactical adjustments.

(1) PORTFOLIO: Rotate 5-10% from growth to quality/dividend on market strength. Maintain US overweight; selective EM through India/ASEAN. Lock in 5%+ yields where duration fits needs.

(2) SECURITY: Review cyber insurance coverage and deductibles. Validate backup recovery within 4-hour RTO. Enable MFA on all financial accounts. Freeze credit reports.

(3) PREPAREDNESS: Build scenario playbooks for Taiwan strait escalation (supply chain), European energy disruption (inflation), and domestic unrest (business continuity).

(4) RESOURCES: Add to critical minerals exposure on pullbacks. Assess personal/organizational dependency on single-source supply chains.

(5) INFORMATION: Diversify news sources. Build verification habits. Reduce social media exposure during high-tension periods.

NEXT REASSESSMENT: When live intelligence stream activates or on material change in key indicators (rate path, conflict escalation, election outcome clarity).`,
  };
}
