import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { Redis } from '@upstash/redis';
import { getLFBMClient, type AnalysisMode } from '@/lib/inference/LFBMClient';

export const runtime = 'edge';
export const maxDuration = 120; // Extended for meta-analysis

/**
 * Emergency Refresh Modes - ALL powered by LFBM (self-hosted vLLM)
 * - realtime: Current metric translation (~$0.001)
 * - historical: Meta-analysis using Qwen's training knowledge (~$0.002)
 * - hybrid: Combine current metrics with historical patterns (~$0.003)
 *
 * NO EXTERNAL LLM DEPENDENCIES - 250x cheaper than Anthropic
 */
type RefreshMode = AnalysisMode;

interface RefreshRequest {
  mode?: RefreshMode;
  // For historical/hybrid modes
  gdeltPeriod?: {
    start: string; // ISO date, GDELT goes back to 2015
    end: string;
  };
  historicalFocus?: string; // e.g., "Arab Spring parallels", "Cold War patterns"
  depth?: 'quick' | 'standard' | 'deep';
}

// Initialize Redis client for cache (same as intel-briefing route)
const redis = Redis.fromEnv();
const REDIS_CACHE_TTL_SECONDS = 6 * 60 * 60; // 6 hours for emergency refresh

// Emergency endpoint to force-refresh intel data via LFBM (self-hosted vLLM)
// Estimated cost: ~$0.001-0.003 per call (250x cheaper than Anthropic)
export async function POST(request: Request) {
  const startTime = Date.now();
  const debugInfo: Record<string, unknown> = {};

  // Parse request body for mode and options
  let body: RefreshRequest = {};
  try {
    body = await request.json();
  } catch {
    // Default to realtime mode if no body
  }
  const mode: RefreshMode = body.mode || 'realtime';
  const depth = body.depth || 'standard';

  try {
    // Check environment variables first
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
    const lfbmEndpoint = process.env.LFBM_ENDPOINT;

    if (!supabaseUrl || !supabaseKey) {
      return NextResponse.json({
        success: false,
        error: 'Missing Supabase configuration',
      }, { status: 500 });
    }

    if (!lfbmEndpoint) {
      return NextResponse.json({
        success: false,
        error: 'Missing LFBM_ENDPOINT - configure self-hosted vLLM endpoint',
      }, { status: 500 });
    }

    // Use service role client for database operations
    const supabase = createClient(supabaseUrl, supabaseKey);

    // STRICT ADMIN AUTH: Use @supabase/ssr for proper cookie-based auth
    const cookieStore = await cookies();
    const authClient = createServerClient(
      supabaseUrl,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          getAll() {
            return cookieStore.getAll();
          },
          setAll() {
            // Read-only for this check
          },
        },
      }
    );

    // Get user from session cookies (handles chunked auth tokens automatically)
    const { data: { user }, error: authError } = await authClient.auth.getUser();

    if (authError || !user) {
      console.log('[EMERGENCY REFRESH] Auth failed:', authError?.message || 'No user');
      return NextResponse.json({
        success: false,
        error: 'Authentication required - please log in'
      }, { status: 401 });
    }

    // Check if user is admin using service role client (bypasses RLS)
    const { data: profile, error: profileError } = await supabase
      .from('profiles')
      .select('role')
      .eq('id', user.id)
      .single();

    if (profileError || !profile || profile.role !== 'admin') {
      console.warn(`[EMERGENCY REFRESH] Non-admin user ${user.id} (${user.email}) attempted access`);
      return NextResponse.json({
        success: false,
        error: 'Admin access required'
      }, { status: 403 });
    }

    console.log(`[EMERGENCY REFRESH] Admin ${user.email} authorized`);
    console.log(`[EMERGENCY REFRESH] Admin ${user.id} initiated ${mode} refresh via LFBM`);

    debugInfo.hasSupabaseUrl = !!supabaseUrl;
    debugInfo.hasSupabaseKey = !!supabaseKey;
    debugInfo.hasLFBMEndpoint = !!lfbmEndpoint;
    debugInfo.adminId = user.id;
    debugInfo.environment = process.env.VERCEL_ENV || process.env.NODE_ENV;
    debugInfo.mode = mode;

    const lfbmClient = getLFBMClient();

    // Fetch nation data (used in all modes)
    const { data: nationRisks } = await supabase
      .from('nations')
      .select('code, name, basin_strength, transition_risk, regime')
      .order('transition_risk', { ascending: false })
      .limit(20);

    const highRiskNations = (nationRisks || [])
      .filter((n: { transition_risk?: number }) => (n.transition_risk || 0) > 0.5)
      .map((n: { code: string; name: string; transition_risk?: number }) =>
        `${n.name} (${n.code}): ${((n.transition_risk || 0) * 100).toFixed(0)}% transition risk`)
      .slice(0, 5);

    // Fetch GDELT signals based on mode
    let gdeltQuery = supabase
      .from('learning_events')
      .select('domain, data, timestamp')
      .eq('session_hash', 'gdelt_ingest');

    // For historical/hybrid mode, use specified period; else last 48h
    if (body.gdeltPeriod && (mode === 'historical' || mode === 'hybrid')) {
      gdeltQuery = gdeltQuery
        .gte('timestamp', body.gdeltPeriod.start)
        .lte('timestamp', body.gdeltPeriod.end);
    } else {
      gdeltQuery = gdeltQuery
        .gte('timestamp', new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString());
    }

    const { data: recentSignals } = await gdeltQuery.limit(100);
    const gdeltSignalCount = recentSignals?.length || 0;

    // Build GDELT summary for historical analysis
    const gdeltSummary = recentSignals?.reduce((acc, s) => {
      const domain = s.domain || 'unknown';
      acc[domain] = (acc[domain] || 0) + 1;
      return acc;
    }, {} as Record<string, number>) || {};

    // ============================================================
    // LFBM: ALL modes powered by self-hosted vLLM (250x cheaper)
    // ============================================================
    console.log(`[EMERGENCY REFRESH] Using LFBM (self-hosted vLLM) for ${mode} mode`);
    const lfbmStartTime = Date.now();

    // Prepare nation data for LFBM
    const nationDataForLFBM = (nationRisks || []).map((n: { code: string; name: string; basin_strength?: number; transition_risk?: number; regime?: number }) => ({
      code: n.code,
      name: n.name,
      basin_strength: n.basin_strength,
      transition_risk: n.transition_risk,
      regime: n.regime,
    }));

    const categoryRisks = {
      political: 50 + highRiskNations.length * 10,
      economic: 45,
      security: 50 + highRiskNations.length * 15,
      cyber: 40,
    };

    const gdeltSignals = {
      count: gdeltSignalCount,
      avg_tone: highRiskNations.length > 0 ? -0.3 : 0.1,
      alerts: highRiskNations.length,
    };

    let briefings: Record<string, string>;

    if (mode === 'historical') {
      // Historical mode - pattern analysis
      const response = await lfbmClient.generateHistoricalAnalysis({
        nations: nationDataForLFBM.map(n => ({
          code: n.code,
          name: n.name,
          risk: n.transition_risk || 0,
          trend: (n.basin_strength || 0) > 0.5 ? 0.1 : -0.1,
        })),
        gdeltSummary,
        focus: body.historicalFocus,
        depth,
      });
      briefings = response.briefings;
    } else if (mode === 'hybrid') {
      // Hybrid mode - current + historical
      const response = await lfbmClient.generateHybridAnalysis(
        nationDataForLFBM,
        gdeltSignals,
        categoryRisks,
        body.historicalFocus
      );
      briefings = response.briefings;
    } else {
      // Realtime mode - pure metric translation
      briefings = await lfbmClient.generateFromMetrics(
        nationDataForLFBM,
        gdeltSignals,
        categoryRisks
      );
    }

    const lfbmLatency = Date.now() - lfbmStartTime;
    console.log(`[EMERGENCY REFRESH] LFBM ${mode} completed in ${lfbmLatency}ms`);

    // Cost estimates for LFBM (250x cheaper than Anthropic)
    const costEstimates: Record<string, string> = {
      'realtime': '$0.001',
      'historical-quick': '$0.001',
      'historical-standard': '$0.002',
      'historical-deep': '$0.003',
      'hybrid-quick': '$0.001',
      'hybrid-standard': '$0.002',
      'hybrid-deep': '$0.003',
    };
    const costKey = mode === 'realtime' ? 'realtime' : `${mode}-${depth}`;

    // Prepare cache data
    const cacheData = {
      briefings,
      metadata: {
        region: 'Global',
        preset: 'global',
        timestamp: new Date().toISOString(),
        overallRisk: 'elevated' as const,
        source: 'emergency_refresh',
        mode,
        depth: mode !== 'realtime' ? depth : undefined,
        model: 'lfbm-vllm',
        estimatedCost: costEstimates[costKey] || '$0.002',
        cached: false,
        gdeltPeriod: body.gdeltPeriod,
        historicalFocus: body.historicalFocus,
      },
    };

    // ============================================================
    // CRITICAL: Write to REDIS cache (used by main intel-briefing API)
    // ============================================================
    // The main /api/intel-briefing endpoint reads from Redis with key 'intel-briefing:global'
    // We MUST write to Redis for the briefings page to pick up our fresh data!
    try {
      const redisCacheKey = 'intel-briefing:global'; // Same format as intel-briefing route

      // FIRST: Delete the old cache to ensure fresh data is used
      await redis.del(redisCacheKey);
      console.log('[EMERGENCY REFRESH] Deleted old cache key:', redisCacheKey);

      // THEN: Write the new data
      await redis.set(redisCacheKey, {
        data: cacheData,
        timestamp: Date.now(),
        generatedAt: new Date().toISOString(),
      }, { ex: REDIS_CACHE_TTL_SECONDS });
      console.log('[EMERGENCY REFRESH] ✅ Written to Redis cache: intel-briefing:global');
    } catch (redisError) {
      console.error('[EMERGENCY REFRESH] Redis cache write failed:', redisError);
      // Continue anyway - we'll also write to Supabase as backup
    }

    // Also cache in Supabase as backup
    const currentDate = new Date().toISOString().split('T')[0];
    const supabaseCacheKey = `emergency_briefing_global_${currentDate}`;
    try {
      // Delete ALL expired cache entries
      await supabase
        .from('briefing_cache')
        .delete()
        .lt('expires_at', new Date().toISOString());

      // Delete any existing emergency refresh cache for this preset
      await supabase
        .from('briefing_cache')
        .delete()
        .like('cache_key', 'emergency_briefing_%');

      // Store fresh data in Supabase
      await supabase
        .from('briefing_cache')
        .upsert({
          cache_key: supabaseCacheKey,
          data: cacheData,
          expires_at: new Date(Date.now() + 6 * 60 * 60 * 1000).toISOString(), // 6 hour expiry
        }, { onConflict: 'cache_key' });

      console.log('[EMERGENCY REFRESH] ✅ Supabase cache updated');
    } catch (cacheError) {
      console.warn('[EMERGENCY REFRESH] Supabase cache storage failed:', cacheError);
    }

    return NextResponse.json({
      success: true,
      message: `Emergency ${mode} refresh completed via LFBM`,
      briefings,
      metadata: cacheData.metadata,
      usage: {
        source: 'lfbm_vllm',
        latencyMs: lfbmLatency,
        estimatedCost: costEstimates[costKey] || '$0.002',
      },
    });

  } catch (error) {
    console.error('Emergency refresh failed:', error);

    // Extract detailed error info
    let errorDetails = 'Unknown error';
    let errorType = 'unknown';

    if (error instanceof Error) {
      errorDetails = error.message;
      errorType = error.name;

      // Check for LFBM-specific errors
      if (error.message.includes('401') || error.message.includes('Unauthorized')) {
        errorDetails = 'LFBM API key is invalid. Please check LFBM_API_KEY in Vercel.';
        errorType = 'auth_error';
      } else if (error.message.includes('not configured')) {
        errorDetails = 'LFBM endpoint not configured. Set LFBM_ENDPOINT in Vercel.';
        errorType = 'config_error';
      } else if (error.message.includes('429') || error.message.includes('rate')) {
        errorDetails = 'RunPod rate limited. Please wait a moment and try again.';
        errorType = 'rate_limit';
      }
    }

    return NextResponse.json({
      success: false,
      error: 'Emergency refresh failed',
      details: errorDetails,
      errorType,
      debug: {
        ...debugInfo,
        latencyMs: Date.now() - startTime,
      },
    }, { status: 500 });
  }
}
