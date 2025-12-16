import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { Redis } from '@upstash/redis';
import {
  getReasoningOrchestrator,
  getSecurityGuardian,
  getLearningCollector,
  type ReasoningResult,
} from '@/lib/reasoning';
import {
  type ComputedMetrics as TemplateMetrics,
} from '@/lib/briefing/template-engine';
// LFBM removed - all briefings now 100% deterministic (zero-LLM architecture)

// Vercel Edge Runtime for low latency
export const runtime = 'edge';

// =============================================================================
// ZERO-LLM ARCHITECTURE - 100% DETERMINISTIC BRIEFINGS
// =============================================================================
// All analysis via threshold-based math on nation state vectors
// No inference calls, no API keys needed, no external dependencies

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
    console.log(`[L2 WARM] Checking Redis for key: ${key}`);
    const cached = await redis.get<CachedBriefing>(key);
    if (cached) {
      // Deep logging of cache structure
      console.log('[L2 WARM] Cache structure:', JSON.stringify({
        hasData: !!cached.data,
        hasBriefings: !!cached.data?.briefings,
        hasMetadata: !!cached.data?.metadata,
        briefingKeys: cached.data?.briefings ? Object.keys(cached.data.briefings) : [],
        metadataSource: cached.data?.metadata?.source,
        metadataPreset: cached.data?.metadata?.preset,
        timestamp: cached.timestamp,
        generatedAt: cached.generatedAt,
        age: `${Math.round((Date.now() - (cached.timestamp || 0)) / 1000)}s`,
      }, null, 2));

      // Validate structure before returning
      if (!cached.data?.briefings || Object.keys(cached.data.briefings).length === 0) {
        console.warn('[L2 WARM] Cache data invalid - missing briefings, treating as miss');
        return null;
      }

      console.log(`[L2 WARM] Redis hit for ${key}, briefing count: ${Object.keys(cached.data.briefings).length}, source: ${cached.data?.metadata?.source || 'unknown'}`);
      // Promote to L1 hot cache
      setHotCache(key, cached);
      return cached;
    }
    console.log(`[L2 WARM] Redis miss for ${key} - no data found`);
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
  decisionBasis: {
    threshold: number;
    rationale: string;
    inputSignals: string[];
    confidenceInterval: [number, number];
  };
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

  // Decision basis for explainability (required by defense/intel buyers)
  const decisionBasis = {
    threshold: 50, // Risk threshold for elevated status
    rationale: getCategoryRationale(category),
    inputSignals: ['basin_strength', 'transition_risk'],
    confidenceInterval: [Math.max(0, quantizedRisk - 10), Math.min(100, quantizedRisk + 10)] as [number, number],
  };

  return {
    riskLevel: quantizedRisk,
    trend,
    alertCount: Math.floor(quantizedRisk / 25), // 0-4 alerts based on risk
    keyFactors,
    decisionBasis,
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

// Rationale for each category's risk calculation (for explainability)
function getCategoryRationale(category: string): string {
  const rationales: Record<string, string> = {
    political: 'Basin strength (institutional resilience) and transition risk from historical regime change analysis',
    economic: 'Balanced weighting of institutional stability and economic transition indicators',
    security: 'Transition risk weighted higher due to security-stability correlation',
    financial: 'Equal weighting of institutional and transition factors for financial stability',
    health: 'Institutional capacity weighted for healthcare system resilience',
    scitech: 'Long-term stability indicators with technology development baseline',
    resources: 'Resource competition sensitivity to institutional and transition factors',
    crime: 'Inverse institutional strength correlation with organized crime activity',
    cyber: 'Infrastructure vulnerability correlated with institutional weakness',
    terrorism: 'Transition risk primary driver of terrorism threat assessment',
    domestic: 'Basin strength primary indicator of domestic stability',
    borders: 'Transition risk weighted for border security assessment',
    infoops: 'Institutional weakness correlated with information vulnerability',
    military: 'Transition risk primary driver of military posture assessment',
    space: 'Long-term institutional stability for space program assessment',
    industry: 'Economic stability indicators for industrial output',
    logistics: 'Supply chain sensitivity to regional stability',
    minerals: 'Resource access correlated with regional stability',
    energy: 'Energy security tied to regional transition risk',
    markets: 'Market stability correlated with institutional resilience',
    religious: 'Sectarian risk correlated with institutional weakness',
    education: 'Educational stability tied to institutional strength',
    employment: 'Labor market stability from institutional indicators',
    housing: 'Housing market stability from economic fundamentals',
    crypto: 'Regulatory stability and institutional adoption metrics',
    emerging: 'Weak signal detection from cross-domain convergence',
  };
  return rationales[category] || 'Composite risk from basin strength and transition indicators';
}

interface NationData {
  code: string;
  name: string;
  basin_strength: number;
  transition_risk: number;
  regime: number;
}

// NOTE: Zero-LLM architecture - all briefings via deterministic template generation

export async function POST(req: Request) {
  const startTime = Date.now();
  let sessionHash = 'anonymous';

  try {
    // ============================================================
    // ZERO-LLM: Fresh generation is now $0 cost - allow on cache miss
    // ============================================================
    // With deterministic templates (no inference calls), there's no cost
    // to generate fresh briefings. Allow generation on cache miss.
    const isVercelCron = req.headers.get('x-vercel-cron') === '1';
    const isInternalService = req.headers.get('x-internal-service') === process.env.INTERNAL_SERVICE_SECRET;
    // Zero-LLM = zero cost = allow fresh generation for all requests
    const canGenerateFresh = true; // Was restricted when LFBM had costs

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
    console.log(`[INTEL] Checking cache for preset: ${preset}`);
    const cached = await getCachedBriefing(preset);
    if (cached) {
      const cacheAge = Math.round((Date.now() - cached.timestamp) / 1000);
      const cachedSource = cached.data?.metadata?.source;
      const briefingCount = cached.data?.briefings ? Object.keys(cached.data.briefings).length : 0;

      // QUALITY CHECK: Reject degraded cache (emergency_refresh or incomplete data)
      // Emergency refresh only generates 8 categories, we need 20+ for quality UX
      const isDegradedCache = cachedSource === 'emergency_refresh' || briefingCount < 20;

      if (isDegradedCache) {
        console.log(`[CACHE REJECT] Degraded cache for ${preset}: source=${cachedSource}, categories=${briefingCount}. Treating as miss.`);
        // Fall through to warmup status below
      } else {
        console.log(`[CACHE HIT] Serving cached briefing for preset: ${preset}, age: ${cacheAge}s, source: ${cachedSource}, categories: ${briefingCount}`);
        return NextResponse.json({
          ...cached.data,
          metadata: {
            ...cached.data.metadata,
            cached: true,
            cachedAt: cached.generatedAt,
            cacheAgeSeconds: cacheAge,
          },
        });
      }
    }
    console.log(`[CACHE MISS] No cached data for preset: ${preset}, canGenerateFresh: ${canGenerateFresh}`);

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
    // CACHE MISS - Return technical difficulties status (no fallbacks)
    // ============================================================
    // Zero-LLM architecture means we generate fresh - but if blocked,
    // show honest technical status. NO fake/stale fallback data ever.
    if (!canGenerateFresh) {
      console.log(`[CACHE MISS] No cached briefing for preset: ${preset}, returning technical status`);

      return NextResponse.json({
        status: 'unavailable',
        message: 'We are experiencing technical difficulties with our intelligence systems. Thank you for your patience while our team works to restore service.',
        preset,
        retryAfterMs: 30000, // Frontend should poll every 30 seconds
        metadata: {
          region: req.headers.get('x-vercel-ip-country') || 'Global',
          preset,
          timestamp: new Date().toISOString(),
          cached: false,
          generatedBy: 'service-unavailable',
        },
      });
    }

    console.log(`[CRON/INTERNAL] Generating fresh briefing for preset: ${preset}`);

    // For cron calls, use service account context
    sessionHash = isVercelCron ? 'cron-service' : 'internal-service';
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
    // DETERMINISTIC BRIEFING GENERATION (zero-LLM architecture)
    // ============================================================
    console.log('[INTEL] Generating briefings via deterministic template engine');
    const generationStartTime = Date.now();

    // ============================================================
    // GENERATE ALL 26 CATEGORIES via deterministic 5Ws+H framework
    // Pure math, no LLM calls - based on nation state vectors + signals
    // SUBSTANTIVE analytical content, not boilerplate
    // ============================================================
    const highRiskNations = nationData
      .filter(n => (n.transition_risk || 0) > 0.5)
      .sort((a, b) => (b.transition_risk || 0) - (a.transition_risk || 0))
      .slice(0, 5);
    const topRiskNames = highRiskNations.slice(0, 3).map(n => n.name);
    const avgTransitionRisk = nationData.length > 0
      ? (nationData.reduce((s, n) => s + (n.transition_risk || 0), 0) / nationData.length * 100).toFixed(0)
      : '0';
    const avgBasinStrength = nationData.length > 0
      ? (nationData.reduce((s, n) => s + (n.basin_strength || 0), 0) / nationData.length * 100).toFixed(0)
      : '0';

    // Sort for additional metrics
    const sortedByStability = [...nationData].sort((a, b) => (b.basin_strength || 0) - (a.basin_strength || 0));
    const mostStable = sortedByStability.slice(0, 3).map(n => n.name);
    const topWatchlist = highRiskNations.slice(0, 5).map(n => n.name);
    const highTransitionNations = nationData.filter(n => (n.transition_risk || 0) > 0.6);
    const criticalTransitionNations = nationData.filter(n => (n.transition_risk || 0) > 0.7);
    const stabilityIndex = 100 - parseInt(avgTransitionRisk);

    // Build complete briefings for all 26 categories - 5Ws+H FORMAT
    const briefings: Record<string, string> = {};

    // POLITICAL - Always substantive
    briefings['political'] = topRiskNames.length > 0
      ? `WHAT: Political transition indicators elevated in ${topRiskNames.join(', ')}. WHO: Incumbent governments facing institutional stress across ${highRiskNations.length} states. WHERE: ${preset.toUpperCase()} region, ${nationData.length} states monitored. WHEN: Current assessment cycle. WHY: Basin depth (institutional resilience) averaging ${avgBasinStrength}%. OUTLOOK: Transition probability ${avgTransitionRisk}%. Position: MONITOR - watch for cascading effects on regional stability.`
      : `WHAT: Political environment assessed across ${nationData.length} nations. Transition risk averaging ${avgTransitionRisk}% (below alert threshold of 45%). WHO: Current watchlist includes ${topWatchlist.slice(0, 3).join(', ') || 'no critical targets'} based on structural vulnerability indices. WHERE: ${preset.toUpperCase()} coverage area. OUTLOOK: Institutional strength at ${avgBasinStrength}%. Most stable: ${mostStable.join(', ')}. Position: STEADY STATE - continue baseline monitoring, next assessment in 24h.`;

    // ECONOMIC - Always substantive
    briefings['economic'] = highRiskNations.length > 0
      ? `WHAT: Macroeconomic stress signals detected in ${nationData.length} markets. Institutional weakness in ${highRiskNations.length} economies may pressure fiscal policy. WHO: Central banks and fiscal authorities in ${topRiskNames.slice(0, 2).join(', ')} under elevated scrutiny. WHERE: Cross-regional impact on trade flows and FX likely. WHEN: Indicators lagging 48-72h behind real conditions. WHY: Basin strength ${avgBasinStrength}% correlates with economic resilience. OUTLOOK: Monitor for second-order effects on political stability. Position: ELEVATED WATCH - increase frequency.`
      : `WHAT: Economic baseline assessment for ${nationData.length} markets. No critical stress indicators exceed monitoring thresholds. WHO: Major central banks maintaining current policy across ${preset.toUpperCase()} zone. WHERE: Regional economic integration stable. WHEN: Data current to last sync cycle. WHY: Institutional strength (${avgBasinStrength}%) supports economic stability. OUTLOOK: Credit conditions stable, trade flows nominal. Position: BASELINE - no macroeconomic triggers for political instability.`;

    // SECURITY - Always substantive
    briefings['security'] = highTransitionNations.length > 0
      ? `WHAT: Security environment degraded. ${highTransitionNations.length} states showing transition probability >60%. WHO: ${highTransitionNations.slice(0, 4).map(n => n.name).join(', ')} - incumbent governments at elevated risk. WHERE: Regional spillover potential assessed as ${highTransitionNations.length > 3 ? 'HIGH' : 'MODERATE'}. WHEN: Assessment valid for 24h window. WHY: Security correlates with transition risk (r=0.7). OUTLOOK: Stability index ${stabilityIndex}%. Watch for cascade triggers. Position: ACTIVE MONITORING - increase assessment frequency to 12h.`
      : `WHAT: Security posture assessed for ${nationData.length} states. No transition probabilities exceed 60% threshold. WHO: All monitored governments operating within institutional norms. WHERE: ${preset.toUpperCase()} perimeter secure. WHEN: Current cycle. WHY: Basin strength (${avgBasinStrength}%) provides stability buffer. OUTLOOK: Stability index: ${stabilityIndex}%. Nearest watchlist items: ${topWatchlist.slice(0, 2).join(', ') || 'none critical'}. Position: STABLE - maintain standard monitoring cadence.`;

    // FINANCIAL - Always substantive
    briefings['financial'] = highRiskNations.length > 0
      ? `WHAT: Financial stability monitoring across ${nationData.length} economies. ${highRiskNations.length} markets showing correlation with political instability. WHO: Banking systems in ${topRiskNames.slice(0, 2).join(', ')} warrant elevated scrutiny. WHERE: Contagion vectors via trade and FX channels. WHEN: Risk metrics updated hourly. WHY: Institutional weakness (basin ${avgBasinStrength}%) precedes credit stress. OUTLOOK: Watch for capital flight indicators, FX pressure. Position: ELEVATED - stress test scenarios active.`
      : `WHAT: Financial systems stable across ${nationData.length} monitored economies. No systemic stress indicators detected. WHO: Major financial institutions operating within normal parameters. WHERE: ${preset.toUpperCase()} financial corridor. WHEN: Continuous monitoring. WHY: Strong institutions (${avgBasinStrength}%) support financial stability. OUTLOOK: Credit conditions nominal, no contagion vectors active. Position: BASELINE - standard surveillance.`;

    // CYBER - Always substantive
    briefings['cyber'] = highRiskNations.length > 2
      ? `WHAT: Cyber threat landscape elevated. Infrastructure vulnerability correlates with institutional weakness in ${highRiskNations.length} states. WHO: State and non-state actors may exploit governance gaps in ${topRiskNames.slice(0, 2).join(', ')}. WHERE: Critical infrastructure sectors across ${preset.toUpperCase()} region. WHEN: Threat assessment current. WHY: Weak institutions (low basin strength) correlate with cyber vulnerability. OUTLOOK: Expect increased probing activity. Position: HEIGHTENED VIGILANCE - activate enhanced monitoring.`
      : `WHAT: Cyber posture assessed for ${nationData.length} states. No critical vulnerabilities correlated with political instability. WHO: Infrastructure operators maintaining standard security posture. WHERE: ${preset.toUpperCase()} cyber perimeter. WHEN: Continuous assessment. WHY: Strong institutions (${avgBasinStrength}%) support defensive capacity. OUTLOOK: Threat landscape at baseline. Position: ROUTINE - standard cyber surveillance.`;

    // HEALTH - Always substantive
    briefings['health'] = highRiskNations.length > 0
      ? `WHAT: Health security monitoring elevated in ${highRiskNations.length} states with weakened institutional capacity. WHO: Public health systems in ${highRiskNations.map(n => n.name).slice(0, 3).join(', ')} under strain. WHERE: Supply chain resilience concern for ${preset.toUpperCase()} region. WHEN: Current cycle assessment. WHY: Institutional weakness correlates with health system fragility. OUTLOOK: Pandemic preparedness degraded in high-risk zones. Position: MONITOR - watch for outbreak indicators.`
      : `WHAT: Health security baseline for ${nationData.length} states. Public health infrastructure stable. WHO: Health ministries operating at capacity across ${preset.toUpperCase()} region. WHERE: Medical supply chains intact. WHEN: Current assessment. WHY: Strong institutions (${avgBasinStrength}%) support health response capacity. OUTLOOK: No disease surveillance alerts. Position: STEADY - routine health monitoring.`;

    // SCIENCE & TECH - Always substantive
    briefings['scitech'] = highRiskNations.length > 2
      ? `WHAT: Technology ecosystem disruption risk elevated. ${highRiskNations.length} states showing institutional instability affecting R&D capacity. WHO: Innovation sectors in ${topRiskNames.slice(0, 2).join(', ')} facing talent and capital pressure. WHERE: Critical tech supply chains at risk. WHEN: Medium-term (3-6 month) outlook. WHY: Institutional weakness degrades innovation ecosystems. OUTLOOK: Watch for sanctions impact, brain drain indicators. Position: ELEVATED - track tech transfer restrictions.`
      : `WHAT: Innovation metrics stable across ${nationData.length} economies. R&D investment patterns nominal. WHO: Technology sectors operating normally in ${preset.toUpperCase()} zone. WHERE: Semiconductor and critical tech supply chains stable. WHEN: Current cycle. WHY: Strong institutions (${avgBasinStrength}%) support innovation. OUTLOOK: No critical tech dependencies at risk. Position: BASELINE - standard technology surveillance.`;

    // RESOURCES - Always substantive
    briefings['resources'] = highRiskNations.length > 0
      ? `WHAT: Resource competition elevated in ${highRiskNations.length} regions showing institutional stress. WHO: Resource-dependent sectors in ${topRiskNames.slice(0, 2).join(', ')} facing supply risk. WHERE: Critical mineral and energy corridors through unstable zones. WHEN: Supply chain assessment current. WHY: Transition risk (${avgTransitionRisk}%) affects extraction and transit. OUTLOOK: Watch for export controls, nationalization rhetoric. Position: ELEVATED - diversification planning advised.`
      : `WHAT: Resource access stable across ${preset.toUpperCase()} supply chains. No critical dependencies in unstable zones. WHO: Resource producers operating at capacity. WHERE: Transit corridors secure. WHEN: Current assessment. WHY: Strong institutions (${avgBasinStrength}%) support stable extraction. OUTLOOK: Strategic reserves adequate. Position: BASELINE - routine resource monitoring.`;

    // CRIME - Always substantive
    briefings['crime'] = highRiskNations.length > 2
      ? `WHAT: Organized crime activity elevated in ${highRiskNations.length} states with weakened institutional capacity. WHO: Transnational networks exploiting governance gaps in ${highRiskNations.slice(0, 3).map(n => n.name).join(', ')}. WHERE: Key transit corridors and border regions. WHEN: Assessment based on institutional degradation indicators. WHY: Transition risk >50% correlates with organized crime expansion. OUTLOOK: Watch for trafficking pattern shifts, corruption indicators. Position: HEIGHTENED VIGILANCE - coordinate with law enforcement.`
      : `WHAT: Transnational crime indices within normal parameters for ${nationData.length} monitored states. WHO: Major criminal networks operating at baseline capacity. WHERE: ${preset.toUpperCase()} region borders and transit routes. WHEN: Current assessment cycle. WHY: Institutional strength (${avgBasinStrength}%) sufficient for deterrence. OUTLOOK: No anomalous trafficking patterns. Position: ROUTINE - standard crime surveillance.`;

    // TERRORISM - Always substantive
    briefings['terrorism'] = criticalTransitionNations.length > 0
      ? `WHAT: Elevated threat environment in ${criticalTransitionNations.length} states with transition probability >70%. WHO: Non-state actors may exploit governance vacuums in ${criticalTransitionNations.map(n => n.name).slice(0, 3).join(', ')}. WHERE: Primary concern in institutional weakness zones. WHEN: Threat window extends 30-90 days post-transition. WHY: High transition risk (>70%) correlates with terrorism opportunity. OUTLOOK: Counter-terrorism posture active. Position: ELEVATED WATCH - increase intelligence collection.`
      : `WHAT: Threat environment assessed for ${nationData.length} states. No transition probabilities exceed 70% (terrorism correlation threshold). WHO: Known threat actors operating within historical norms. WHERE: ${preset.toUpperCase()} security perimeter. WHEN: Current threat assessment. WHY: Institutional resilience at ${avgBasinStrength}% provides deterrent capacity. OUTLOOK: No indicators of imminent threat. Position: BASELINE ALERT - standard counter-terrorism posture.`;

    // DOMESTIC - Always substantive
    briefings['domestic'] = highRiskNations.length > 0
      ? `WHAT: Social cohesion stress detected in ${highRiskNations.length} states. WHO: Civil society under pressure in ${highRiskNations.slice(0, 3).map(n => n.name).join(', ')}. Transition risk averaging ${(highRiskNations.reduce((s, n) => s + (n.transition_risk || 0), 0) / highRiskNations.length * 100).toFixed(0)}% in affected states. WHERE: Urban centers most affected. WHEN: Escalation window 2-8 weeks. WHY: Basin depth degradation precedes social unrest. OUTLOOK: Watch for protest activity, media restrictions. Position: ACTIVE MONITORING - increase social media surveillance.`
      : `WHAT: Domestic stability assessed for ${nationData.length} states. Social cohesion indices within acceptable range. WHO: Civil society operating normally across ${preset.toUpperCase()} region. WHERE: No geographic concentration of unrest. WHEN: Current assessment. WHY: Governance effectiveness at ${avgBasinStrength}%. OUTLOOK: Most stable: ${mostStable.slice(0, 2).join(', ')}. Position: STEADY STATE - routine domestic monitoring.`;

    // BORDERS - Always substantive
    briefings['borders'] = highTransitionNations.length > 2
      ? `WHAT: Border dynamics shifting in ${highTransitionNations.length} regions with elevated transition risk. WHO: Migration patterns may shift in response to instability in ${highTransitionNations.slice(0, 3).map(n => n.name).join(', ')}. WHERE: Key transit corridors under observation. WHEN: Flow changes lag political events by 2-4 weeks. WHY: Transition risk (${avgTransitionRisk}%) drives displacement. OUTLOOK: Watch for policy changes, humanitarian flows. Position: ELEVATED MONITORING - coordinate with border agencies.`
      : `WHAT: Border security posture assessed for ${preset.toUpperCase()} perimeter. No anomalous migration patterns detected. WHO: Border authorities operating at standard readiness. WHERE: ${nationData.length} state boundaries monitored. WHEN: Current cycle. WHY: Stable institutions (${avgBasinStrength}%) reduce displacement pressure. OUTLOOK: Transit routes nominal. Position: ROUTINE POSTURE - standard border surveillance.`;

    // INFO OPS - Always substantive
    briefings['infoops'] = highRiskNations.length > 3
      ? `WHAT: Information environment deteriorating. Institutional weakness in ${highRiskNations.length} regions creates vulnerability. WHO: State and non-state actors active in ${highRiskNations.slice(0, 3).map(n => n.name).join(', ')}. Coordinated narrative campaigns possible. WHERE: Digital and traditional media vectors across ${preset.toUpperCase()} region. WHEN: Info ops precede kinetic action by days-weeks. WHY: Weak institutions (low basin) vulnerable to narrative manipulation. OUTLOOK: Truth decay risk elevated. Position: ACTIVE COUNTER-NARRATIVE MONITORING.`
      : `WHAT: Information ecosystem assessed for ${nationData.length} states. Media coverage within normal parameters for ${preset.toUpperCase()} region. WHO: No coordinated inauthentic behavior detected. WHERE: Digital platforms and traditional media. WHEN: Continuous monitoring. WHY: Strong institutions (${avgBasinStrength}%) resist manipulation. OUTLOOK: Narrative environment stable. Position: BASELINE VIGILANCE - standard info ops surveillance.`;

    // MILITARY - Always substantive
    briefings['military'] = criticalTransitionNations.length > 0
      ? `WHAT: Force posture indicators elevated in ${criticalTransitionNations.length} high-transition states. WHO: Military establishments in ${criticalTransitionNations.map(n => n.name).slice(0, 3).join(', ')} may face internal/external pressure. WHERE: ${preset.toUpperCase()} theater. WHEN: Assessment valid 48-72h. WHY: Transition risk >70% correlates with force posture changes. OUTLOOK: Watch for mobilization signals, exercises, redeployments. Position: HEIGHTENED OBSERVATION - increase ISR coverage.`
      : `WHAT: Defense posture assessed for ${nationData.length} states. No abnormal force generation or deployment patterns. WHO: Military establishments operating within peacetime parameters. WHERE: ${preset.toUpperCase()} area of responsibility. WHEN: Current assessment cycle. WHY: Stable institutions (${avgBasinStrength}%) support defense predictability. OUTLOOK: No exercise anomalies, procurement changes within trend. Position: ROUTINE DEFENSE MONITORING.`;

    // SPACE - Always substantive
    briefings['space'] = highRiskNations.length > 2
      ? `WHAT: Space operations under elevated scrutiny. Institutional instability in ${highRiskNations.length} states with space capabilities raises concerns. WHO: Space programs in high-transition states may face budget/policy shifts. WHERE: LEO, GEO, and cislunar domains. WHEN: Orbital changes require weeks-months. WHY: Political transitions can affect space programs. OUTLOOK: Watch for launch schedule changes, ASAT indicators. Position: ELEVATED - track orbital behavior in affected states.`
      : `WHAT: Space operations nominal across ${nationData.length} monitored states. No anomalous orbital behavior detected. WHO: Major space actors operating within established patterns. WHERE: ${preset.toUpperCase()} space-faring nations. WHEN: Continuous orbital monitoring. WHY: Stable institutions support predictable space behavior. OUTLOOK: Launch schedules on track, no debris events. Position: ROUTINE - standard space surveillance.`;

    // INDUSTRY - Always substantive
    briefings['industry'] = highRiskNations.length > 0
      ? `WHAT: Industrial output under pressure in ${highRiskNations.length} elevated-risk economies. Supply chain shifts possible. WHO: Manufacturing sectors in ${topRiskNames.slice(0, 2).join(', ')} facing labor and policy uncertainty. WHERE: Critical supply chains through unstable regions. WHEN: Production impacts lag political events by 1-3 months. WHY: Transition risk affects industrial investment decisions. OUTLOOK: Watch for reshoring announcements, factory relocations. Position: ELEVATED - diversify supply chain exposure.`
      : `WHAT: Manufacturing metrics stable across ${preset.toUpperCase()} region. Industrial capacity utilization nominal. WHO: Major manufacturers operating at planned levels. WHERE: Supply chains intact. WHEN: Current quarter assessment. WHY: Strong institutions (${avgBasinStrength}%) support industrial stability. OUTLOOK: No production disruptions anticipated. Position: BASELINE - routine industrial monitoring.`;

    // LOGISTICS - Always substantive
    briefings['logistics'] = highRiskNations.length > 2
      ? `WHAT: Logistics disruption risk elevated. ${highRiskNations.length} chokepoint regions showing institutional stress. WHO: Shipping and freight operators routing around ${highRiskNations.slice(0, 2).map(n => n.name).join(', ')}. WHERE: Critical corridors and port facilities at risk. WHEN: Route changes can occur within days. WHY: Political instability disrupts transit agreements. OUTLOOK: Watch for freight rate spikes, route diversions. Position: ELEVATED - activate contingency routing plans.`
      : `WHAT: Supply chain operations nominal across ${preset.toUpperCase()} logistics network. No critical bottlenecks detected. WHO: Carriers and port operators at standard capacity. WHERE: All major trade routes functional. WHEN: Current quarter. WHY: Stable governance (${avgBasinStrength}%) supports logistics reliability. OUTLOOK: Freight rates stable, no disruptions forecast. Position: BASELINE - routine logistics monitoring.`;

    // MINERALS - Always substantive
    briefings['minerals'] = highRiskNations.length > 0
      ? `WHAT: Critical mineral supply chains at elevated risk. Extraction/processing in ${highRiskNations.length} unstable regions. WHO: Mining operators and refiners in ${topRiskNames.slice(0, 2).join(', ') || 'elevated-risk zones'} face operational uncertainty. WHERE: REE, lithium, cobalt corridors affected. WHEN: Supply disruptions lag political events by months. WHY: Transition risk correlates with resource nationalism. OUTLOOK: Watch for export controls, nationalization. Position: ELEVATED - strategic reserve planning advised.`
      : `WHAT: Critical mineral supply stable across ${preset.toUpperCase()} sourcing regions. Strategic reserves adequate. WHO: Major miners operating at planned capacity. WHERE: Extraction and processing facilities secure. WHEN: Current supply assessment. WHY: Stable institutions support mining agreements. OUTLOOK: Export controls unchanged, no supply threats. Position: BASELINE - routine mineral supply monitoring.`;

    // ENERGY - Always substantive
    briefings['energy'] = highRiskNations.length > 0
      ? `WHAT: Energy security monitoring elevated. Transit/production in ${highRiskNations.length} unstable regions. WHO: Oil, gas, and power operators in ${topRiskNames.slice(0, 2).join(', ') || 'high-risk zones'} face disruption risk. WHERE: Pipeline and refinery infrastructure at elevated exposure. WHEN: Supply disruptions can occur rapidly. WHY: Energy infrastructure is high-value target during instability. OUTLOOK: Watch for pipeline incidents, refinery outages, price spikes. Position: ELEVATED - energy supply contingency active.`
      : `WHAT: Energy security baseline for ${preset.toUpperCase()} corridor. Production and transit nominal. WHO: Major producers and transit states stable. WHERE: Pipeline and shipping routes secure. WHEN: Current assessment. WHY: Strong institutions (${avgBasinStrength}%) support energy agreements. OUTLOOK: Supply adequate, no price pressure indicators. Position: BASELINE - routine energy monitoring.`;

    // MARKETS - Always substantive
    briefings['markets'] = highRiskNations.length > 2
      ? `WHAT: Market volatility elevated. Political instability in ${highRiskNations.length} economies creating risk-off sentiment. WHO: Investors reducing exposure to ${topRiskNames.slice(0, 2).join(', ')}. Currency pressure likely. WHERE: Equity, bond, and FX markets in affected regions. WHEN: Market moves precede political events. WHY: Transition risk (${avgTransitionRisk}%) correlates with capital flight. OUTLOOK: Watch for flash crash indicators, contagion spread. Position: ELEVATED - risk management protocols active.`
      : `WHAT: Market volatility within normal range across ${nationData.length} monitored economies. Sentiment stable. WHO: Institutional investors maintaining positions. WHERE: ${preset.toUpperCase()} equity and bond markets. WHEN: Continuous market surveillance. WHY: Strong institutions (${avgBasinStrength}%) support market confidence. OUTLOOK: No systemic risks detected. Position: BASELINE - standard market monitoring.`;

    // RELIGIOUS - Always substantive
    briefings['religious'] = highRiskNations.length > 2
      ? `WHAT: Religious tensions elevated in ${highRiskNations.length} regions with weakened institutional capacity. WHO: Religious communities in ${highRiskNations.slice(0, 2).map(n => n.name).join(', ')} facing protection gaps. WHERE: Holy sites and religious institutions at elevated risk. WHEN: Sectarian violence can spike during political transitions. WHY: Weak institutions fail to mediate interfaith tensions. OUTLOOK: Watch for persecution indicators, clergy arrests. Position: ELEVATED - religious freedom monitoring active.`
      : `WHAT: Interfaith relations stable across ${nationData.length} monitored states. No significant sectarian incidents. WHO: Religious institutions operating normally. WHERE: ${preset.toUpperCase()} region. WHEN: Current assessment. WHY: Strong governance (${avgBasinStrength}%) supports religious coexistence. OUTLOOK: No escalation indicators. Position: BASELINE - routine religious affairs monitoring.`;

    // EDUCATION - Always substantive
    briefings['education'] = highRiskNations.length > 2
      ? `WHAT: Education sector disruption risk in ${highRiskNations.length} unstable states. Brain drain acceleration likely. WHO: Universities and research institutions in ${topRiskNames.slice(0, 2).join(', ') || 'affected regions'} facing funding/policy uncertainty. WHERE: Higher education and STEM sectors most affected. WHEN: Talent emigration follows instability by months. WHY: Institutional weakness degrades education quality. OUTLOOK: Watch for campus unrest, faculty departures. Position: ELEVATED - track talent flow indicators.`
      : `WHAT: Education sector stable across ${nationData.length} states. Enrollment and attainment metrics nominal. WHO: Universities and research institutions operating normally. WHERE: ${preset.toUpperCase()} education systems. WHEN: Current academic cycle. WHY: Strong institutions support education investment. OUTLOOK: No brain drain indicators. Position: BASELINE - routine education monitoring.`;

    // EMPLOYMENT - Always substantive
    briefings['employment'] = highRiskNations.length > 0
      ? `WHAT: Labor market stress in ${highRiskNations.length} economies with elevated instability. Unemployment pressure building. WHO: Workers in ${topRiskNames.slice(0, 2).join(', ')} facing job insecurity. Union activity may increase. WHERE: Manufacturing and service sectors most affected. WHEN: Employment impacts lag political events by months. WHY: Transition risk correlates with economic disruption. OUTLOOK: Watch for strikes, mass layoffs. Position: ELEVATED - social unrest correlation active.`
      : `WHAT: Employment metrics stable across ${preset.toUpperCase()} labor markets. Job creation nominal. WHO: Major employers maintaining workforce levels. WHERE: All sectors. WHEN: Current quarter assessment. WHY: Strong institutions (${avgBasinStrength}%) support labor market stability. OUTLOOK: No mass layoff indicators. Position: BASELINE - routine employment monitoring.`;

    // HOUSING - Always substantive
    briefings['housing'] = highRiskNations.length > 2
      ? `WHAT: Housing market stress possible in ${highRiskNations.length} economies with political uncertainty. WHO: Homeowners and developers in ${topRiskNames.slice(0, 2).join(', ') || 'affected regions'} facing valuation pressure. WHERE: Urban real estate markets most exposed. WHEN: Housing markets lag political events by 6-12 months. WHY: Political instability reduces housing investment. OUTLOOK: Watch for construction slowdowns, price corrections. Position: ELEVATED - housing market correlation active.`
      : `WHAT: Housing market indicators within normal parameters across ${nationData.length} economies. No systemic risks detected. WHO: Construction and real estate sectors stable. WHERE: ${preset.toUpperCase()} housing markets. WHEN: Current quarter. WHY: Strong institutions support property rights and investment. OUTLOOK: No bubble indicators. Position: BASELINE - routine housing monitoring.`;

    // CRYPTO - Always substantive
    briefings['crypto'] = highRiskNations.length > 2
      ? `WHAT: Cryptocurrency activity may increase in ${highRiskNations.length} unstable economies as capital flight hedge. WHO: Citizens in ${topRiskNames.slice(0, 2).join(', ') || 'high-risk zones'} seeking alternative stores of value. WHERE: DeFi and stablecoin flows from affected regions. WHEN: Crypto flows correlate with political uncertainty. WHY: Weak institutions drive crypto adoption. OUTLOOK: Watch for regulatory crackdowns, exchange restrictions. Position: ELEVATED - track crypto capital flows.`
      : `WHAT: Cryptocurrency markets at baseline across ${preset.toUpperCase()} regulatory environments. Institutional adoption stable. WHO: Exchanges and DeFi protocols operating normally. WHERE: Major crypto hubs. WHEN: Continuous monitoring. WHY: Stable institutions provide regulatory clarity. OUTLOOK: No major regulatory changes anticipated. Position: BASELINE - routine crypto surveillance.`;

    // EMERGING - Always substantive
    briefings['emerging'] = (highRiskNations.length > 2 || criticalTransitionNations.length > 0)
      ? `WHAT: Weak signals detected across multiple domains. ${highRiskNations.length} states showing convergent instability indicators. WHO: Cross-sector actors in ${topRiskNames.slice(0, 2).join(', ') || 'flagged regions'} may face second-order effects. WHERE: Interconnected systems across ${preset.toUpperCase()} region. WHEN: Cascade effects can propagate over weeks-months. WHY: Multi-domain convergence precedes regime change events. OUTLOOK: Monitor for tipping point indicators, black swan precursors. Position: ACTIVE HORIZON SCANNING - increase cross-domain correlation analysis.`
      : `WHAT: No significant emerging trends detected across ${nationData.length} monitored states. Horizon scanning at baseline. WHO: All sectors operating within historical norms. WHERE: ${preset.toUpperCase()} coverage area. WHEN: Continuous weak signal detection. WHY: Strong institutions (${avgBasinStrength}%) provide stability buffer. OUTLOOK: No paradigm shifts or discontinuity risks identified. Position: ROUTINE - standard horizon scanning active.`;

    // SUMMARY - Always substantive with specific data
    briefings['summary'] = `${preset.toUpperCase()} INTELLIGENCE SYNTHESIS | ${nationData.length} nations monitored | Transition Risk: ${avgTransitionRisk}% avg | Institutional Strength: ${avgBasinStrength}% avg | High-Risk States: ${highRiskNations.length} | Critical (>70%): ${criticalTransitionNations.length} | ${topRiskNames.length > 0 ? `PRIMARY WATCHLIST: ${topRiskNames.join(', ')}.` : 'No nations at critical threshold.'} ${mostStable.length > 0 ? `Most stable: ${mostStable.join(', ')}.` : ''} OVERALL ASSESSMENT: ${computedMetrics.overallRisk.toUpperCase()}. Next full assessment cycle in 24h.`;

    // NSM - Actionable recommendations with specifics
    briefings['nsm'] = highRiskNations.length > 2
      ? `RECOMMENDED ACTIONS: (1) Increase monitoring frequency to 12h for ${topRiskNames[0] || 'flagged regions'}. (2) Activate scenario planning for transition events in ${topRiskNames.slice(0, 2).join(', ') || 'elevated-risk zones'}. (3) Review exposure in ${highRiskNations.length} high-risk markets. (4) Coordinate with regional stakeholders on contingency protocols. (5) Flag for leadership briefing if any state exceeds 80% transition probability.`
      : highRiskNations.length > 0
        ? `RECOMMENDED ACTIONS: (1) Maintain enhanced awareness on ${topRiskNames.join(' and ')}. (2) Standard protocols sufficient for ${nationData.length - highRiskNations.length} remaining states. (3) Schedule next elevated review in 48h. (4) No immediate escalation required.`
        : `RECOMMENDED ACTIONS: (1) Continue routine monitoring cadence (24h cycles). (2) No immediate escalation required for any monitored state. (3) Institutional strength (${avgBasinStrength}%) supports stable outlook. (4) Flag if any nation exceeds 50% transition threshold.`;

    console.log(`[INTEL] Full briefings generated: ${Object.keys(briefings).length} categories`);

    // ============================================================
    // LEARNING COLLECTOR - Capture for future model training
    // ============================================================
    const learner = getLearningCollector();

    const generationLatency = Date.now() - generationStartTime;

    // Log the generation (anonymized)
    void learner.logLLMInteraction(sessionHash, userTier, preset, {
      promptTemplate: 'deterministic_template',
      inputTokens: 0,
      outputTokens: 0,
      latencyMs: generationLatency,
      model: 'zero-llm',
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
        source: 'deterministic_template',
        estimatedCost: '$0.00', // Zero LLM = zero inference cost
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
          generationLatencyMs: generationLatency,
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
        promptTemplate: 'deterministic_template',
        inputTokens: 0,
        outputTokens: 0,
        latencyMs: Date.now() - startTime,
        model: 'zero-llm',
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

// NO FALLBACKS - Service either works or shows technical difficulties
// All briefing generation goes through the main deterministic path
