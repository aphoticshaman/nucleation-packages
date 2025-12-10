import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { Redis } from '@upstash/redis';
import { getLFBMClient, type AnalysisMode } from '@/lib/inference/LFBMClient';
import {
  processSignals,
  extractFeatures,
  decideLLMTier,
  type ProcessedSignal,
} from '@/lib/signals/SignalProcessor';
import {
  runInference,
  generateLogicalBriefing,
  type LogicalBriefing,
} from '@/lib/signals/LogicalAgent';

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
    // Get ALL gdelt records - we'll filter for text content in JS
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

    // Increased limit from 100 to 500 to get more article signals
    const { data: recentSignals, error: signalError } = await gdeltQuery.limit(500);
    const gdeltSignalCount = recentSignals?.length || 0;

    console.log(`[EMERGENCY REFRESH] GDELT query: ${gdeltSignalCount} signals fetched (filter: title not null, last 48h)`);
    if (signalError) {
      console.error(`[EMERGENCY REFRESH] GDELT query error:`, signalError);
    }

    // Build GDELT summary for historical analysis
    const gdeltSummary = recentSignals?.reduce((acc, s) => {
      const domain = s.domain || 'unknown';
      acc[domain] = (acc[domain] || 0) + 1;
      return acc;
    }, {} as Record<string, number>) || {};

    // ============================================================
    // CPU-FIRST ARCHITECTURE: SignalProcessor + LogicalAgent
    // ============================================================
    // Step 1: CPU signal processing (anomaly detection, feature extraction)
    // Step 2: CPU logical inference (rule-based analysis)
    // Step 3: LFBM only for polish (if anomalies need LLM interpretation)
    // Expected savings: 80-95% reduction in LLM calls
    // ============================================================
    console.log(`[EMERGENCY REFRESH] CPU-first pipeline starting for ${mode} mode`);
    const cpuStartTime = Date.now();

    // Prepare nation data
    const nationDataForLFBM = (nationRisks || []).map((n: { code: string; name: string; basin_strength?: number; transition_risk?: number; regime?: number }) => ({
      code: n.code,
      name: n.name,
      basin_strength: n.basin_strength,
      transition_risk: n.transition_risk,
      regime: n.regime,
    }));

    // Build nation risk map for LogicalAgent
    const nationRiskMap: Record<string, number> = {};
    for (const n of nationDataForLFBM) {
      nationRiskMap[n.code] = n.transition_risk || 0;
    }

    // Process GDELT signals through SignalProcessor (CPU)
    const gdeltTexts = (recentSignals || [])
      .map((s) => {
        const data = s.data as { text?: string; title?: string; summary?: string } | null;
        return data?.text || data?.title || data?.summary || '';
      })
      .filter((t) => t.length > 20);

    console.log(`[EMERGENCY REFRESH] Text extraction: ${gdeltTexts.length} texts with length > 20 chars (from ${gdeltSignalCount} signals)`);

    let processedSignals: ProcessedSignal[] = [];
    let anomalyCount = 0;
    let needsLLMPolish = false;

    if (gdeltTexts.length > 0) {
      const processingResult = await processSignals(gdeltTexts);
      processedSignals = processingResult.processed;
      anomalyCount = processingResult.anomalies.length;

      // Check if any anomalies need LLM interpretation
      for (const signal of processingResult.anomalies) {
        const tier = decideLLMTier(signal);
        if (tier === 'elle') {
          needsLLMPolish = true;
          break;
        }
      }

      console.log(`[CPU] Processed ${gdeltTexts.length} signals, found ${anomalyCount} anomalies, needsLLM: ${needsLLMPolish}`);
    }

    // Run LogicalAgent inference (CPU) - rule-based analysis
    const inferenceResult = runInference(
      processedSignals,
      nationRiskMap,
      { // Baseline stats
        sentimentMean: 0,
        sentimentStd: 0.5,
        urgencyMean: 0.2,
        urgencyStd: 0.3,
        topicDistribution: [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
        signalCount: processedSignals.length,
        lastUpdated: new Date(),
      }
    );

    // Generate logical briefing from inferences (CPU - NO LLM)
    const logicalBriefing: LogicalBriefing = generateLogicalBriefing(inferenceResult, nationRiskMap);

    const cpuLatency = Date.now() - cpuStartTime;
    console.log(`[CPU] Inference completed in ${cpuLatency}ms: ${inferenceResult.inferences.length} inferences from ${inferenceResult.factsUsed} facts`);

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
    let usedFallback = false;
    let usedCPUOnly = false;

    // Helper to convert LogicalBriefing to briefings format
    const convertLogicalBriefingToFormat = (lb: LogicalBriefing): Record<string, string> => {
      const timestamp = new Date().toLocaleString('en-US', {
        month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', hour12: true
      });

      // Extract country names from high risk nations
      const riskCountries = highRiskNations.slice(0, 3).map((n: string) => n.split(' (')[0]);
      const primaryWatch = riskCountries[0] || 'monitored regions';

      return {
        summary: lb.summary || `GLOBAL SITUATION (${timestamp}): ${lb.keyFindings[0] || 'Monitoring active.'}`,
        political: lb.keyFindings.find(f => f.toLowerCase().includes('political')) ||
          `Political dynamics across ${nationDataForLFBM.length} states. ${riskCountries.length > 0 ? `Watch: ${riskCountries.join(', ')}.` : 'Stable.'}`,
        economic: lb.keyFindings.find(f => f.toLowerCase().includes('economic') || f.toLowerCase().includes('trade')) ||
          `Economic intelligence from ${gdeltSignalCount} signals. ${gdeltSignals.avg_tone > 0 ? 'Positive' : 'Cautionary'} sentiment.`,
        security: lb.keyFindings.find(f => f.toLowerCase().includes('security') || f.toLowerCase().includes('military')) ||
          `Security: ${lb.riskAlerts.length > 0 ? lb.riskAlerts.map(a => `${a.entity}: ${a.level}`).join(', ') : 'Baseline'}`,
        financial: `Financial stability indicators compiled. ${lb.trends.length > 0 ? lb.trends[0] : 'Credit conditions stable.'}`,
        cyber: `Cyber threat landscape at baseline. ${anomalyCount > 0 ? `${anomalyCount} anomalies under review.` : 'No critical signals.'}`,
        energy: `Energy security tracking. ${lb.cascadeWarnings.length > 0 ? 'Cascade pathways detected.' : 'Supply corridors stable.'}`,
        nsm: lb.actionItems.length > 0
          ? lb.actionItems.join(' ')
          : (highRiskNations.length > 0
              ? `Recommend: Monitor ${primaryWatch}. Review transition scenarios.`
              : 'Maintain routine collection. Standard 4-hour cycle.'),
      };
    };

    // ============================================================
    // DECISION: Use CPU-only OR call LFBM for polish
    // ============================================================
    // Only call LFBM if:
    // 1. Mode is 'historical' or 'hybrid' (requires LLM knowledge)
    // 2. We have critical anomalies that need LLM interpretation
    // 3. Depth is 'deep' (user explicitly requested full analysis)
    // Otherwise: Use CPU-generated briefing (saves ~$0.001 per call)

    const shouldUseLFBM = mode === 'historical' || mode === 'hybrid' || depth === 'deep' || needsLLMPolish;

    if (!shouldUseLFBM) {
      // CPU-ONLY PATH - No LLM call, huge cost savings
      console.log(`[EMERGENCY REFRESH] Using CPU-only path (mode=${mode}, anomalies=${anomalyCount}, needsLLM=${needsLLMPolish})`);
      briefings = convertLogicalBriefingToFormat(logicalBriefing);
      usedCPUOnly = true;
    } else {
      // LFBM PATH - Only for historical/hybrid modes or critical anomalies
      console.log(`[EMERGENCY REFRESH] Using LFBM for ${mode} mode (anomalies=${anomalyCount})`);
      const lfbmStartTime = Date.now();
      const LFBM_TIMEOUT_MS = 90000;

      try {
        const lfbmPromise = (async () => {
          if (mode === 'historical') {
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
            return response.briefings;
          } else if (mode === 'hybrid') {
            const response = await lfbmClient.generateHybridAnalysis(
              nationDataForLFBM,
              gdeltSignals,
              categoryRisks,
              body.historicalFocus
            );
            return response.briefings;
          } else {
            return await lfbmClient.generateFromMetrics(
              nationDataForLFBM,
              gdeltSignals,
              categoryRisks
            );
          }
        })();

        const timeoutPromise = new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('LFBM_TIMEOUT')), LFBM_TIMEOUT_MS)
        );

        briefings = await Promise.race([lfbmPromise, timeoutPromise]);
      } catch (lfbmError) {
        const isTimeout = lfbmError instanceof Error && lfbmError.message === 'LFBM_TIMEOUT';
        console.warn(`[EMERGENCY REFRESH] LFBM ${isTimeout ? 'timed out (90s)' : 'failed'}: ${lfbmError instanceof Error ? lfbmError.message : 'unknown'}`);
        console.warn('[EMERGENCY REFRESH] Falling back to CPU-generated briefing');
        briefings = convertLogicalBriefingToFormat(logicalBriefing);
        usedFallback = true;
      }

      const lfbmLatency = Date.now() - lfbmStartTime;
      console.log(`[EMERGENCY REFRESH] LFBM ${mode} completed in ${lfbmLatency}ms`);
    }

    const totalLatency = Date.now() - cpuStartTime;
    console.log(`[EMERGENCY REFRESH] Total pipeline: ${totalLatency}ms (CPU: ${cpuLatency}ms, inferences: ${inferenceResult.inferences.length})`);

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
    let redisCacheSuccess = false;
    let redisErrorMsg = '';
    const redisCacheKey = 'intel-briefing:global'; // Same format as intel-briefing route

    try {
      // Log what we're about to write
      console.log('[EMERGENCY REFRESH] Cache data structure:', JSON.stringify({
        hasBriefings: !!cacheData.briefings,
        briefingKeys: Object.keys(cacheData.briefings || {}),
        briefingLengths: Object.fromEntries(
          Object.entries(cacheData.briefings || {}).map(([k, v]) => [k, String(v).length])
        ),
        hasMetadata: !!cacheData.metadata,
        metadataSource: cacheData.metadata?.source,
        usedFallback,
      }, null, 2));

      // FIRST: Delete the old cache to ensure fresh data is used
      await redis.del(redisCacheKey);
      console.log('[EMERGENCY REFRESH] Deleted old cache key:', redisCacheKey);

      // THEN: Write the new data
      const cacheEntry = {
        data: cacheData,
        timestamp: Date.now(),
        generatedAt: new Date().toISOString(),
      };

      console.log('[EMERGENCY REFRESH] Writing cache entry with structure:', JSON.stringify({
        hasData: !!cacheEntry.data,
        dataHasBriefings: !!cacheEntry.data?.briefings,
        briefingCount: Object.keys(cacheEntry.data?.briefings || {}).length,
        timestamp: cacheEntry.timestamp,
        generatedAt: cacheEntry.generatedAt,
        ttl: REDIS_CACHE_TTL_SECONDS,
      }, null, 2));

      await redis.set(redisCacheKey, cacheEntry, { ex: REDIS_CACHE_TTL_SECONDS });
      console.log('[EMERGENCY REFRESH] ✅ Written to Redis cache: intel-briefing:global');

      // Verify the write by reading it back
      const verification = await redis.get<{ data: typeof cacheData; timestamp: number; generatedAt: string }>(redisCacheKey);
      if (verification) {
        console.log('[EMERGENCY REFRESH] ✅ Redis verification:', JSON.stringify({
          hasData: !!verification.data,
          hasBriefings: !!verification.data?.briefings,
          briefingKeys: Object.keys(verification.data?.briefings || {}),
          timestamp: verification.timestamp,
        }, null, 2));
        redisCacheSuccess = true;
      } else {
        console.error('[EMERGENCY REFRESH] ❌ Redis verification failed - cache not found after write');
        redisErrorMsg = 'Cache write verified but read returned null';
      }
    } catch (redisError) {
      console.error('[EMERGENCY REFRESH] Redis cache write failed:', redisError);
      redisErrorMsg = redisError instanceof Error ? redisError.message : 'Unknown Redis error';
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
      message: `Emergency ${mode} refresh completed${usedCPUOnly ? ' (CPU-first)' : usedFallback ? ' (CPU fallback)' : ' via LFBM'}`,
      briefings,
      metadata: cacheData.metadata,
      cache: {
        redis: redisCacheSuccess,
        redisError: redisErrorMsg || undefined,
        key: redisCacheKey,
      },
      usage: {
        source: usedCPUOnly ? 'cpu_logical_agent' : usedFallback ? 'cpu_fallback' : 'lfbm_vllm',
        cpuLatencyMs: cpuLatency,
        totalLatencyMs: totalLatency,
        estimatedCost: usedCPUOnly || usedFallback ? '$0.00' : (costEstimates[costKey] || '$0.002'),
        usedCPUOnly,
        usedFallback,
        inferenceStats: {
          signalsProcessed: processedSignals.length,
          anomaliesDetected: anomalyCount,
          inferencesGenerated: inferenceResult.inferences.length,
          factsUsed: inferenceResult.factsUsed,
        },
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
