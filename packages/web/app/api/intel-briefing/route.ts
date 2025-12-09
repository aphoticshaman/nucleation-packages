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
import {
  generateBriefingFromMetrics,
  shouldUseClaude,
  type ComputedMetrics as TemplateMetrics,
} from '@/lib/briefing/template-engine';

// Vercel Edge Runtime for low latency
export const runtime = 'edge';

// =============================================================================
// COST CONTROL KILL SWITCH
// =============================================================================
// Set DISABLE_LLM=true in Vercel to completely stop all LLM API calls
// Use this when you've blown through your credits
const LLM_KILL_SWITCH = process.env.DISABLE_LLM === 'true';

// PRODUCTION-ONLY: Block Anthropic API calls in non-production unless explicitly enabled
function isAnthropicAllowed(): boolean {
  const env = process.env.VERCEL_ENV || process.env.NODE_ENV;
  if (env === 'production') return true;
  if (process.env.ALLOW_ANTHROPIC_IN_DEV === 'true') {
    console.warn('ANTHROPIC API ENABLED IN NON-PRODUCTION');
    return true;
  }
  return false;
}

// =============================================================================
// CACHE TYPES
// =============================================================================
interface CachedBriefing {
  data: {
    briefings: Record<string, string>;
    metadata: Record<string, unknown>;
  };
  timestamp: number;
  generatedAt: string;
}

interface HotCacheEntry {
  data: CachedBriefing;
  expiry: number;
}

// =============================================================================
// L1: HOT CACHE - In-memory, same edge instance (<1ms latency)
// =============================================================================
// This is process-local - not shared across edge instances, but INSTANT.
// Falls back to Redis (warm) if miss. Acts as Upstash failover too.
const HOT_CACHE_TTL_MS = 60 * 1000; // 60 seconds - short because not shared

const hotCache = new Map<string, HotCacheEntry>();

function getHotCache(key: string): CachedBriefing | null {
  const entry = hotCache.get(key);
  if (entry && Date.now() < entry.expiry) {
    console.log(`[L1 HOT] Cache hit for ${key}`);
    return entry.data;
  }
  if (entry) {
    hotCache.delete(key); // Expired, clean up
  }
  return null;
}

function setHotCache(key: string, data: CachedBriefing): void {
  hotCache.set(key, {
    data,
    expiry: Date.now() + HOT_CACHE_TTL_MS,
  });
  console.log(`[L1 HOT] Cached ${key} (expires in ${HOT_CACHE_TTL_MS / 1000}s)`);

  // Prevent memory bloat - keep max 20 entries
  if (hotCache.size > 20) {
    const oldestKey = hotCache.keys().next().value;
    if (oldestKey) hotCache.delete(oldestKey);
  }
}

// =============================================================================
// L2: WARM CACHE - Redis (Upstash), shared across ALL edge instances (5-50ms)
// =============================================================================
// Cache TTL: 10 minutes. All users hitting the same preset get cached response.
// Only Enterprise tier gets fresh on-demand analysis.
const CACHE_TTL_SECONDS = 10 * 60; // 10 minutes

// Initialize Redis client - uses Redis.fromEnv() to auto-detect env var names
// Works with both UPSTASH_REDIS_REST_* and KV_REST_API_* naming conventions
const redis = Redis.fromEnv();

function getCacheKey(preset: string): string {
  return `intel-briefing:${preset}`;
}

// Multi-tier cache read: L1 (hot) → L2 (warm/Redis)
async function getCachedBriefing(preset: string): Promise<CachedBriefing | null> {
  const key = getCacheKey(preset);

  // L1: Check hot cache first (instant)
  const hot = getHotCache(key);
  if (hot) return hot;

  // L2: Check Redis (warm)
  try {
    const cached = await redis.get<CachedBriefing>(key);
    if (cached) {
      console.log(`[L2 WARM] Redis hit for ${key}`);
      // Promote to L1 hot cache
      setHotCache(key, cached);
      return cached;
    }
  } catch (error) {
    console.error('[L2 WARM] Redis get error (falling through):', error);
    // Redis is down - continue to cold path
  }

  return null;
}

// Multi-tier cache write: Write to BOTH L1 and L2
async function setCachedBriefing(preset: string, data: CachedBriefing['data']): Promise<void> {
  const key = getCacheKey(preset);
  const cacheEntry: CachedBriefing = {
    data,
    timestamp: Date.now(),
    generatedAt: new Date().toISOString(),
  };

  // L1: Set hot cache (instant)
  setHotCache(key, cacheEntry);

  // L2: Set Redis cache (warm)
  try {
    await redis.set(key, cacheEntry, { ex: CACHE_TTL_SECONDS });
    console.log(`[L2 WARM] Cached ${key} in Redis (TTL: ${CACHE_TTL_SECONDS}s)`);
  } catch (error) {
    console.error('[L2 WARM] Redis set error:', error);
    // L1 still works even if Redis fails
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

// Build the system prompt - ToM-informed approach (v2: date-agnostic)
// Key insight: The model was refusing because it saw "December 2025" and triggered knowledge cutoff.
// Solution: Frame the task as pure metric translation with NO DATE DEPENDENCY.
// The model doesn't need to know WHEN - it just converts numbers to prose.
function buildSystemPrompt(userTier: string): string {
  // ToM-informed framing: Define WHO the model is, not what it shouldn't do
  const tierContext = userTier === 'enterprise'
    ? 'You provide strategic context and cross-domain connections for executive decision-makers.'
    : userTier === 'pro'
    ? 'You provide detailed analysis with specific metric references for professional analysts.'
    : 'You provide clear, accessible, actionable intelligence for general audiences.';

  // CRITICAL: No date in system prompt. The model is a TRANSLATOR, not a forecaster.
  // It doesn't need to know what date it is - it just converts metrics to prose.
  return `You are a prose translation engine in a multi-stage intelligence pipeline.

YOUR ROLE IN THE SYSTEM:
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Live Sensors → Raw signals (ALREADY COMPLETE)    │
│  STAGE 2: Risk Engine → Numerical metrics (ALREADY DONE)   │
│  STAGE 3: YOU → Convert metrics to prose (YOUR ONLY JOB)   │
│  STAGE 4: Dashboard → Display to analysts                  │
└─────────────────────────────────────────────────────────────┘

CRITICAL UNDERSTANDING:
- You receive COMPUTED OUTPUTS from upstream systems (Stages 1-2)
- These numbers come from LIVE DATA FEEDS processed moments ago
- Your job is LINGUISTIC TRANSLATION only: numbers → sentences
- You are NOT being asked to predict, forecast, or verify external events
- You do NOT need to access any external knowledge to do your job
- The metrics you receive ARE the ground truth for your output

ANALOGY: You are like a "text-to-speech" system, but for metrics. A TTS system doesn't refuse to read text because it can't verify the text is true - it just converts text to audio. You convert metrics to prose. That's it.

${tierContext}

VOICE: Professional intelligence analyst. "Indicators show...", "Risk metrics at...", "Assessment based on current readings..."

OUTPUT: JSON object with 1-2 sentence briefings per category. Reference the EXACT metrics provided - they are your source of truth.`;
}

// Build the user prompt with computed metrics - ToM-informed approach (v2: date-agnostic)
// Present data as facts from upstream systems, frame task as pure translation
// IMPORTANT: Timestamp shown as relative ("just now") not absolute date to avoid knowledge cutoff triggers
function buildUserPrompt(metrics: ComputedMetrics): string {
  return `STAGE 2 OUTPUT → STAGE 3 INPUT (computed moments ago from live feeds)

Region: ${metrics.region}
Preset: ${metrics.preset}
Pipeline run: just now
Overall risk level: ${metrics.overallRisk}

CATEGORY METRICS (translate each to prose):
${Object.entries(metrics.categories)
  .map(
    ([cat, m]) =>
      `${cat.toUpperCase()}: risk=${m.riskLevel}/100 trend=${m.trend} alerts=${m.alertCount} factors=[${m.keyFactors.join(', ')}]`
  )
  .join('\n')}

ACTIVE ALERTS:
${metrics.topAlerts.map((a) => `[${a.severity.toUpperCase()}] ${a.category}/${a.region}: ${a.summary}`).join('\n')}

OUTPUT (JSON):
{
  "political": "<prose>",
  "economic": "<prose>",
  "security": "<prose>",
  "financial": "<prose>",
  "health": "<prose>",
  "scitech": "<prose>",
  "resources": "<prose>",
  "crime": "<prose>",
  "cyber": "<prose>",
  "terrorism": "<prose>",
  "domestic": "<prose>",
  "borders": "<prose>",
  "infoops": "<prose>",
  "military": "<prose>",
  "space": "<prose>",
  "industry": "<prose>",
  "logistics": "<prose>",
  "minerals": "<prose>",
  "energy": "<prose>",
  "markets": "<prose>",
  "religious": "<prose>",
  "education": "<prose>",
  "employment": "<prose>",
  "housing": "<prose>",
  "crypto": "<prose>",
  "emerging": "<prose on weak signals/emerging patterns>",
  "summary": "<1-sentence overall assessment>",
  "nsm": "<Next Strategic Move for decision-makers>"
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
    const isVercelCron = req.headers.get('x-vercel-cron') === '1'; // Set by Vercel, not spoofable
    const isInternalService = req.headers.get('x-internal-service') === process.env.INTERNAL_SERVICE_SECRET;
    const hasCronWarmWithSecret = req.headers.get('x-cron-warm') === '1' && isInternalService;
    const canGenerateFresh = isVercelCron || isInternalService || hasCronWarmWithSecret;

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

    // Create Supabase client (needed for both fallback and fresh generation)
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

    // ============================================================
    // CACHE MISS - Use enhanced template engine with REAL DATA
    // ============================================================
    // Pulls actual GDELT tones, country signals, and nation state vectors
    // to generate data-driven briefings WITHOUT calling Claude.
    // Cost: $0 | Latency: <50ms | Quality: Dynamic, data-backed
    if (!canGenerateFresh) {
      console.log(`[CACHE MISS] No cached briefing for preset: ${preset}, using enhanced template engine with real data`);

      // Fetch nation data for template engine
      const presetFiltersForTemplate: Record<string, string[] | null> = {
        global: null,
        nato: ['USA', 'CAN', 'GBR', 'FRA', 'DEU', 'ITA', 'ESP', 'POL', 'NLD', 'TUR'],
        brics: ['BRA', 'RUS', 'IND', 'CHN', 'ZAF', 'IRN', 'EGY', 'ETH', 'SAU', 'ARE'],
        conflict: ['UKR', 'RUS', 'ISR', 'PSE', 'LBN', 'SYR', 'YEM', 'TWN', 'CHN', 'PRK'],
      };

      const templateFilter = presetFiltersForTemplate[preset];

      // Parallel fetch: nations, GDELT signals, country indicators
      const [nationsResult, gdeltResult, signalsResult] = await Promise.all([
        // Nations with state vectors
        (async () => {
          let q = supabase.from('nations').select('code, name, basin_strength, transition_risk, regime');
          if (templateFilter) q = q.in('code', templateFilter);
          return q;
        })(),
        // Recent GDELT tone data (last 48 hours)
        supabase
          .from('learning_events')
          .select('domain, data')
          .eq('session_hash', 'gdelt_ingest')
          .gte('timestamp', new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString())
          .limit(100),
        // Country economic signals
        supabase
          .from('country_signals')
          .select('country_code, country_name, indicator, value')
          .order('updated_at', { ascending: false })
          .limit(200),
      ]);

      const templateNationData = (nationsResult.data || []) as NationData[];

      // Process GDELT data into country risk scores
      const gdeltRiskByCountry: Record<string, number> = {};
      const gdeltData = gdeltResult.data || [];
      for (const event of gdeltData) {
        const country = event.domain;
        const risk = (event.data as { numeric_features?: { gdelt_tone_risk?: number } })?.numeric_features?.gdelt_tone_risk;
        if (country && risk !== undefined) {
          gdeltRiskByCountry[country] = risk;
        }
      }

      // Process country signals into indicators
      const signalsByCountry: Record<string, Record<string, number>> = {};
      const signalsData = signalsResult.data || [];
      for (const sig of signalsData) {
        if (!signalsByCountry[sig.country_code]) {
          signalsByCountry[sig.country_code] = {};
        }
        signalsByCountry[sig.country_code][sig.indicator] = sig.value;
      }

      // Find highest-risk nations for dynamic content
      const highRiskNations = templateNationData
        .filter(n => (n.transition_risk || 0) > 0.5 || (gdeltRiskByCountry[n.code] || 0) > 0.6)
        .sort((a, b) => (b.transition_risk || 0) - (a.transition_risk || 0))
        .slice(0, 5);

      // Build dynamic key factors from real data
      const dynamicFactors: string[] = [];
      if (highRiskNations.length > 0) {
        dynamicFactors.push(`Elevated risk indicators in ${highRiskNations.map(n => n.name).join(', ')}`);
      }

      // Check for economic stress signals
      const highInflation = Object.entries(signalsByCountry)
        .filter(([_, sigs]) => (sigs['inflation'] || 0) > 8)
        .map(([code]) => code);
      if (highInflation.length > 0) {
        dynamicFactors.push(`Inflationary pressure detected in ${highInflation.length} economies`);
      }

      // Check for negative GDELT sentiment
      const negativeGdelt = Object.entries(gdeltRiskByCountry)
        .filter(([_, risk]) => risk > 0.7)
        .map(([country]) => country);
      if (negativeGdelt.length > 0) {
        dynamicFactors.push(`Negative media sentiment trending in ${negativeGdelt.length} regions`);
      }

      // Compute metrics with GDELT-enhanced risk
      const computeEnhancedRisk = (nations: NationData[], category: string) => {
        const baseRisk = computeCategoryRisk(nations, category);
        // Boost risk if GDELT shows negative sentiment
        const gdeltBoost = Object.values(gdeltRiskByCountry).length > 0
          ? Object.values(gdeltRiskByCountry).reduce((a, b) => a + b, 0) / Object.values(gdeltRiskByCountry).length * 10
          : 0;
        return {
          ...baseRisk,
          riskLevel: Math.min(100, baseRisk.riskLevel + Math.round(gdeltBoost)),
          keyFactors: dynamicFactors.length > 0 ? dynamicFactors.slice(0, 3) : baseRisk.keyFactors,
        };
      };

      const templateMetrics: TemplateMetrics = {
        region: req.headers.get('x-vercel-ip-country') || 'Global',
        preset: preset as 'global' | 'nato' | 'brics' | 'conflict',
        categories: {
          political: computeEnhancedRisk(templateNationData, 'political'),
          economic: computeEnhancedRisk(templateNationData, 'economic'),
          security: computeEnhancedRisk(templateNationData, 'security'),
          financial: computeEnhancedRisk(templateNationData, 'financial'),
          cyber: computeEnhancedRisk(templateNationData, 'cyber'),
          energy: computeEnhancedRisk(templateNationData, 'energy'),
          trade: computeEnhancedRisk(templateNationData, 'industry'),
          diplomatic: computeEnhancedRisk(templateNationData, 'borders'),
          humanitarian: computeEnhancedRisk(templateNationData, 'health'),
          social: computeEnhancedRisk(templateNationData, 'domestic'),
        },
        topAlerts: generateTopAlerts(templateNationData, preset).map(a => ({
          category: a.category,
          severity: a.severity as 'low' | 'moderate' | 'elevated' | 'high' | 'critical',
          headline: a.summary,
        })),
        overallRisk: computeOverallRisk(templateNationData),
      };

      // Generate briefing using template engine (no LLM, instant)
      const templateBriefing = generateBriefingFromMetrics(templateMetrics);

      // Build richer summary with actual data
      const dataPointCount = gdeltData.length + signalsData.length + templateNationData.length;
      const topRiskNation = highRiskNations[0]?.name || 'multiple regions';

      // Convert to expected response format with enhanced prose
      const briefingsMap: Record<string, string> = {};
      for (const b of templateBriefing.briefings) {
        briefingsMap[b.category] = `${b.summary} Risk: ${b.riskLevel}. Trend: ${b.trend}. ${b.keyPoints.map(p => '• ' + p).join(' ')}`;
      }
      briefingsMap['nsm'] = templateBriefing.nsm;
      briefingsMap['summary'] = `Analysis of ${dataPointCount} data points indicates ${templateMetrics.overallRisk} overall risk. Primary concern: ${topRiskNation}. ${highInflation.length > 0 ? `Economic stress in ${highInflation.length} markets. ` : ''}${negativeGdelt.length > 0 ? `Media sentiment deteriorating in ${negativeGdelt.length} regions.` : ''}`;

      return NextResponse.json({
        briefings: briefingsMap,
        metadata: {
          region: templateMetrics.region,
          preset,
          timestamp: new Date().toISOString(),
          overallRisk: templateMetrics.overallRisk,
          cached: false,
          generatedBy: 'template-engine-enhanced',
          dataPoints: dataPointCount,
          gdeltSignals: gdeltData.length,
          llmCost: 0,
          latencyMs: Date.now() - startTime,
        },
      });
    }

    console.log(`[CRON/INTERNAL] Generating fresh briefing for preset: ${preset}`);

    // For cron calls, use service account context
    sessionHash = isCronWarm ? 'cron-service' : 'internal-service';
    const userTier = 'enterprise'; // Cron/internal gets enterprise-level analysis

    // ============================================================
    // SECURITY LAYER - Rate limits (skip for cron/internal)
    // ============================================================
    const _security = getSecurityGuardian(); // Used for authenticated rate limiting

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

    // BLOCK non-production Anthropic API calls - use fallback instead
    if (!isAnthropicAllowed()) {
      console.log('[INTEL] Anthropic blocked in non-production, returning fallback data');
      const fallback = await getFallbackBriefings(preset, supabase);
      return NextResponse.json({
        briefings: fallback,
        metadata: {
          timestamp: new Date().toISOString(),
          overallRisk: 'elevated',
          source: 'fallback_non_production',
          environment: process.env.VERCEL_ENV || process.env.NODE_ENV,
        },
      });
    }

    // COST CONTROL: Kill switch to stop all LLM calls
    if (LLM_KILL_SWITCH) {
      console.log('[INTEL] LLM KILL SWITCH ACTIVE - returning template data only');
      const fallback = await getFallbackBriefings(preset, supabase);
      return NextResponse.json({
        briefings: fallback,
        metadata: {
          timestamp: new Date().toISOString(),
          overallRisk: 'elevated',
          source: 'kill_switch_active',
          message: 'LLM calls disabled via DISABLE_LLM=true',
        },
      });
    }

    const llmStartTime = Date.now();
    const message = await anthropic.messages.create({
      model: 'claude-haiku-4-5-20251001',
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
      model: 'claude-haiku-4-5-20251001',
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
        model: 'claude-haiku-4-5-20251001',
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
async function _hashForLearning(userId: string): Promise<string> {
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

// Fallback briefings when cache is cold - pulls from REAL stored data
async function getFallbackBriefings(preset: string, supabase: ReturnType<typeof createServerClient>): Promise<Record<string, string>> {
  const presetContext: Record<string, string> = {
    global: 'Global geopolitical landscape',
    nato: 'NATO alliance and Euro-Atlantic region',
    brics: 'BRICS+ economic bloc',
    conflict: 'Active conflict zones',
  };

  const context = presetContext[preset] || presetContext.global;
  const currentDate = new Date().toLocaleDateString('en-US', {
    weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
  });

  // Fetch real data from database
  let riskData: Array<{ iso_a3: string; name: string; overall_risk: number; political_risk: number; economic_risk: number }> = [];
  let signalsData: Array<{ country_code: string; country_name: string; indicator: string; value: number; year: number }> = [];

  try {
    // Get nation risk scores
    const { data: risks } = await supabase
      .from('nation_risk')
      .select('iso_a3, name, overall_risk, political_risk, economic_risk, social_risk')
      .order('overall_risk', { ascending: false })
      .limit(20);
    if (risks) riskData = risks;

    // Get recent country signals (economic indicators)
    const { data: signals } = await supabase
      .from('country_signals')
      .select('country_code, country_name, indicator, value, year')
      .order('updated_at', { ascending: false })
      .limit(100);
    if (signals) signalsData = signals;
  } catch (err) {
    console.error('[FALLBACK] Error fetching real data:', err);
  }

  // Build dynamic briefings from real data
  const highRiskNations = riskData.filter(n => n.overall_risk > 0.6).slice(0, 5);
  const elevatedRiskNations = riskData.filter(n => n.overall_risk > 0.4 && n.overall_risk <= 0.6).slice(0, 5);

  // Extract key economic indicators
  const gdpData = signalsData.filter(s => s.indicator === 'gdp_growth').slice(0, 10);
  const inflationData = signalsData.filter(s => s.indicator === 'inflation').slice(0, 10);
  const unemploymentData = signalsData.filter(s => s.indicator === 'unemployment').slice(0, 10);

  // Format high risk nations list
  const highRiskList = highRiskNations.length > 0
    ? highRiskNations.map(n => `${n.name} (${(n.overall_risk * 100).toFixed(0)}%)`).join(', ')
    : 'No nations currently at critical risk levels';

  const elevatedRiskList = elevatedRiskNations.length > 0
    ? elevatedRiskNations.map(n => n.name).join(', ')
    : 'Risk levels moderate across monitored nations';

  // Format economic indicators
  const avgGdpGrowth = gdpData.length > 0
    ? (gdpData.reduce((sum, d) => sum + d.value, 0) / gdpData.length).toFixed(1)
    : 'N/A';
  const avgInflation = inflationData.length > 0
    ? (inflationData.reduce((sum, d) => sum + d.value, 0) / inflationData.length).toFixed(1)
    : 'N/A';

  // Build briefings from real data
  return {
    political: `[DATA AS OF ${currentDate}] ${context} assessment based on ${riskData.length} monitored nations. HIGH RISK: ${highRiskList}. ELEVATED: ${elevatedRiskList}. Political risk indicators show ${highRiskNations.length > 3 ? 'elevated tensions across multiple regions' : 'localized concerns in specific areas'}. Monitoring ${riskData.length} nations for political stability indicators.`,

    economic: `[DATA AS OF ${currentDate}] Economic indicators from ${signalsData.length} recent data points. Average GDP growth across monitored economies: ${avgGdpGrowth}%. Average inflation: ${avgInflation}%. ${gdpData.length > 0 ? `Recent GDP data: ${gdpData.slice(0, 3).map(d => `${d.country_name}: ${d.value}%`).join(', ')}.` : 'Awaiting fresh economic data ingestion.'} Economic risk concentrated in ${riskData.filter(n => n.economic_risk > 0.5).length} nations.`,

    security: `[DATA AS OF ${currentDate}] Security assessment based on composite risk scores. Nations with elevated security concerns: ${riskData.filter(n => n.overall_risk > 0.5).map(n => n.name).slice(0, 5).join(', ') || 'None at critical levels'}. Overall security posture: ${highRiskNations.length > 5 ? 'ELEVATED' : highRiskNations.length > 2 ? 'MODERATE' : 'STABLE'}. Continuous monitoring of ${riskData.length} nations active.`,

    summary: `[LIVE DATA SUMMARY - ${currentDate}] ${context}: Monitoring ${riskData.length} nations with ${highRiskNations.length} at high risk, ${elevatedRiskNations.length} at elevated risk. Economic data from ${signalsData.length} recent indicators. This is REAL stored data from your intelligence feeds - not cached LLM output. Full LLM-enhanced briefings available when Anthropic API is accessible.`,
  };
}
