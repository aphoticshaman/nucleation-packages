import { NextResponse } from 'next/server';
import Anthropic from '@anthropic-ai/sdk';
import { createClient } from '@supabase/supabase-js';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { Redis } from '@upstash/redis';
import { getLFBMClient, shouldUseLFBM } from '@/lib/inference/LFBMClient';

export const runtime = 'edge';
export const maxDuration = 120; // Extended for meta-analysis

/**
 * Emergency Refresh Modes:
 * - realtime: Current metric translation (default)
 * - historical: Meta-analysis using Claude's verified historical knowledge
 * - hybrid: Combine current metrics with historical pattern analysis
 */
type RefreshMode = 'realtime' | 'historical' | 'hybrid';

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

// PRODUCTION-ONLY: Block Anthropic API calls in non-production unless explicitly enabled
function isAnthropicAllowed(): { allowed: boolean; reason?: string } {
  const env = process.env.VERCEL_ENV || process.env.NODE_ENV;
  const allowInDev = process.env.ALLOW_ANTHROPIC_IN_DEV === 'true';

  if (env === 'production') {
    return { allowed: true };
  }

  if (allowInDev) {
    console.warn('ANTHROPIC API ENABLED IN NON-PRODUCTION - ALLOW_ANTHROPIC_IN_DEV=true');
    return { allowed: true };
  }

  return {
    allowed: false,
    reason: `Anthropic API blocked in ${env} environment. Set ALLOW_ANTHROPIC_IN_DEV=true to enable.`
  };
}

// Emergency endpoint to force-refresh intel data from Claude API
// Estimated cost: $0.25-0.75 per call (realtime), $0.50-2.00 (historical/hybrid)
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
    const anthropicKey = process.env.ANTHROPIC_API_KEY;

    if (!supabaseUrl || !supabaseKey) {
      return NextResponse.json({
        success: false,
        error: 'Missing Supabase configuration',
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

    console.log(`Admin ${user.id} initiated emergency refresh`);

    // BLOCK non-production Anthropic API calls
    const apiCheck = isAnthropicAllowed();
    if (!apiCheck.allowed) {
      return NextResponse.json({
        success: false,
        error: apiCheck.reason,
        environment: process.env.VERCEL_ENV || process.env.NODE_ENV,
      }, { status: 403 });
    }

    debugInfo.hasSupabaseUrl = !!supabaseUrl;
    debugInfo.hasSupabaseKey = !!supabaseKey;
    debugInfo.hasAnthropicKey = !!anthropicKey;
    debugInfo.anthropicKeyPrefix = anthropicKey ? anthropicKey.substring(0, 10) + '...' : 'MISSING';
    debugInfo.adminId = user.id;
    debugInfo.environment = process.env.VERCEL_ENV || process.env.NODE_ENV;

    if (!anthropicKey) {
      return NextResponse.json({
        success: false,
        error: 'Missing ANTHROPIC_API_KEY - please set this in Vercel environment variables',
        debug: debugInfo,
      }, { status: 500 });
    }

    const anthropic = new Anthropic({
      apiKey: anthropicKey,
    });

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
    // MODE-SPECIFIC PROMPT GENERATION
    // ============================================================
    let systemPrompt: string;
    let userPrompt: string;
    let modelId: string;
    let maxTokens: number;

    if (mode === 'historical') {
      // HISTORICAL MODE: Meta-analysis using Claude's verified knowledge
      // No hallucination risk - we only ask about events within training data
      modelId = depth === 'deep' ? 'claude-sonnet-4-20250514' : 'claude-haiku-4-5-20251001';
      maxTokens = depth === 'deep' ? 8000 : depth === 'standard' ? 4000 : 2000;

      systemPrompt = `You are a senior intelligence historian specializing in pattern analysis.

YOUR EXPERTISE:
You have comprehensive, verified knowledge of world history from ancient civilizations through early 2025.
Your role is to identify historical patterns, precedents, and cycles that inform current analysis.

ANALYSIS FRAMEWORK:
1. PATTERN RECOGNITION: Identify recurring patterns across historical eras
2. PRECEDENT MAPPING: Find historical analogues to current situations
3. CYCLE IDENTIFICATION: Detect geopolitical, economic, and social cycles
4. LESSONS EXTRACTION: Distill actionable insights from history

CRITICAL RULES:
- Ground ALL claims in verifiable historical events with dates
- Clearly distinguish established facts from analytical interpretation
- Acknowledge uncertainty when extrapolating patterns
- Connect historical patterns to the current nation data provided

VOICE: Academic intelligence analysis - precise, well-sourced, actionable.`;

      const historicalFocus = body.historicalFocus || 'geopolitical transitions and power shifts';
      const periodDesc = body.gdeltPeriod
        ? `GDELT period: ${body.gdeltPeriod.start} to ${body.gdeltPeriod.end}`
        : 'No specific GDELT period selected';

      userPrompt = `HISTORICAL META-ANALYSIS REQUEST

FOCUS AREA: ${historicalFocus}

CURRENT NATION RISK DATA:
${highRiskNations.length > 0 ? highRiskNations.join('\n') : 'No nations currently exceed 50% threshold'}

GDELT SIGNAL CONTEXT:
${Object.entries(gdeltSummary).map(([d, c]) => `- ${d}: ${c} signals`).join('\n') || 'No GDELT data for period'}
${periodDesc}

YOUR TASK: Generate a meta-analysis connecting historical patterns to the current data.

Identify:
1. **HISTORICAL PRECEDENTS**: Events from history (with dates) that parallel current nation risks
2. **PATTERN ANALYSIS**: Recurring cycles relevant to the focus area
3. **LESSONS**: What history suggests about likely outcomes
4. **EARLY WARNINGS**: Historical indicators that preceded similar situations

Format as JSON:
{
  "summary": "<executive summary connecting history to current data>",
  "precedents": [
    {"event": "<historical event>", "date": "<when>", "parallel": "<current situation it mirrors>", "outcome": "<what happened then>"}
  ],
  "patterns": [
    {"name": "<pattern name>", "cycle_length": "<if applicable>", "examples": ["<ex1>", "<ex2>"], "current_phase": "<where we are in cycle>"}
  ],
  "lessons": ["<lesson 1>", "<lesson 2>"],
  "warnings": ["<warning indicator 1>", "<warning indicator 2>"],
  "political": "<historical political analysis>",
  "economic": "<historical economic patterns>",
  "security": "<historical security parallels>",
  "military": "<historical military doctrine insights>",
  "nsm": "<Next Strategic Move based on historical lessons>"
}

Respond ONLY with valid JSON.`;

    } else if (mode === 'hybrid') {
      // HYBRID MODE: Combine current metrics with historical context
      modelId = depth === 'deep' ? 'claude-sonnet-4-20250514' : 'claude-haiku-4-5-20251001';
      maxTokens = depth === 'deep' ? 6000 : 4000;

      systemPrompt = `You are a dual-mode intelligence analyst combining real-time metrics with historical perspective.

DUAL ANALYSIS FRAMEWORK:
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: Current Metrics → What the numbers show NOW       │
│  LAYER 2: Historical Context → What similar patterns meant  │
│  LAYER 3: Synthesis → Combined assessment                   │
└─────────────────────────────────────────────────────────────┘

Your job is to:
1. Translate current metrics into prose (numbers → sentences)
2. Overlay historical context where relevant (patterns, precedents)
3. Provide synthesized assessment combining both perspectives

RULES:
- Current metrics are ground truth - describe what they show
- Historical references must cite specific events with dates
- Clearly separate "current data shows" from "historically, this meant"
- Synthesis should integrate both, not favor one over the other`;

      userPrompt = `HYBRID ANALYSIS REQUEST

CURRENT METRICS (from live pipeline):
Nations at elevated risk: ${highRiskNations.length > 0 ? highRiskNations.join(', ') : 'None above threshold'}
GDELT signals: ${gdeltSignalCount} in analysis window
Signal breakdown: ${Object.entries(gdeltSummary).map(([d, c]) => `${d}: ${c}`).join(', ') || 'No signals'}

HISTORICAL FOCUS: ${body.historicalFocus || 'General geopolitical patterns'}

Generate a hybrid briefing combining current metrics with historical context.

Format as JSON:
{
  "summary": "<synthesis of current data + historical perspective>",
  "current_assessment": {
    "political": "<current political metrics>",
    "economic": "<current economic indicators>",
    "security": "<current security posture>"
  },
  "historical_context": {
    "most_relevant_precedent": "<historical parallel with date>",
    "pattern_match": "<what historical pattern this resembles>",
    "historical_outcome": "<what happened in similar situations>"
  },
  "synthesis": {
    "political": "<integrated political analysis>",
    "economic": "<integrated economic analysis>",
    "security": "<integrated security analysis>",
    "military": "<integrated military analysis>",
    "cyber": "<integrated cyber analysis>"
  },
  "risk_assessment": "<overall risk based on current + historical>",
  "nsm": "<Next Strategic Move informed by both analyses>"
}

Respond ONLY with valid JSON.`;

    } else {
      // REALTIME MODE (default): Pure metric translation
      modelId = 'claude-haiku-4-5-20251001';
      maxTokens = 4000;

      systemPrompt = `You are a prose translation engine in a multi-stage intelligence pipeline.

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

VOICE: Professional intelligence analyst. Use phrases like "Risk indicators show...", "Metrics at elevated levels for...", "Assessment based on current readings..."

OUTPUT: Generate actionable intelligence prose for each domain, referencing the specific metrics provided.`;

      userPrompt = `UPSTREAM PIPELINE OUTPUT (computed just now from live feeds):

MONITORED NATIONS WITH ELEVATED RISK:
${highRiskNations.length > 0 ? highRiskNations.join('\n') : 'No nations currently exceed 50% transition risk threshold'}

SIGNAL ACTIVITY: ${gdeltSignalCount} GDELT signals processed in last 48 hours

GLOBAL RISK CONTEXT:
- Total nations monitored: ${nationRisks?.length || 0}
- High-risk count: ${highRiskNations.length}
- Signal density: ${gdeltSignalCount > 30 ? 'HIGH' : gdeltSignalCount > 10 ? 'MODERATE' : 'LOW'}

YOUR TASK: Translate these metrics into executive briefing prose for each domain.

Format your response as JSON with ALL 26 domains:
{
  "political": "<1-2 sentences on political risk metrics>",
  "economic": "<1-2 sentences on economic indicators>",
  "security": "<1-2 sentences on security posture>",
  "financial": "<1-2 sentences on financial stability>",
  "health": "<1-2 sentences on health sector>",
  "scitech": "<1-2 sentences on science/tech>",
  "resources": "<1-2 sentences on resource security>",
  "crime": "<1-2 sentences on crime metrics>",
  "cyber": "<1-2 sentences on cyber threats>",
  "terrorism": "<1-2 sentences on terrorism indicators>",
  "domestic": "<1-2 sentences on domestic stability>",
  "borders": "<1-2 sentences on border/migration>",
  "infoops": "<1-2 sentences on information ops>",
  "military": "<1-2 sentences on military posture>",
  "space": "<1-2 sentences on space sector>",
  "industry": "<1-2 sentences on industrial output>",
  "logistics": "<1-2 sentences on logistics/supply>",
  "minerals": "<1-2 sentences on critical minerals>",
  "energy": "<1-2 sentences on energy security>",
  "markets": "<1-2 sentences on market conditions>",
  "religious": "<1-2 sentences on religious affairs>",
  "education": "<1-2 sentences on education sector>",
  "employment": "<1-2 sentences on labor markets>",
  "housing": "<1-2 sentences on housing metrics>",
  "crypto": "<1-2 sentences on digital assets>",
  "emerging": "<1-2 sentences on emerging trends>",
  "summary": "<2-3 sentence executive summary>",
  "nsm": "<Next Strategic Move recommendation>"
}

IMPORTANT: You are translating NUMBERS to SENTENCES. The metrics are your source of truth.
Reference the SPECIFIC metrics provided. Output ONLY valid JSON.`;
    }

    debugInfo.mode = mode;
    debugInfo.depth = depth;
    debugInfo.model = modelId;

    // ============================================================
    // LFBM: Use self-hosted vLLM for realtime mode (250x cheaper)
    // ============================================================
    if (mode === 'realtime' && shouldUseLFBM()) {
      console.log('[EMERGENCY REFRESH] Using LFBM (self-hosted vLLM) - 250x cheaper!');
      const lfbmClient = getLFBMClient();
      const lfbmStartTime = Date.now();

      try {
        // Prepare nation data for LFBM
        const nationDataForLFBM = (nationRisks || []).map((n: { code: string; name: string; basin_strength?: number; transition_risk?: number; regime?: number }) => ({
          code: n.code,
          name: n.name,
          basin_strength: n.basin_strength,
          transition_risk: n.transition_risk,
          regime: n.regime,
        }));

        const briefings = await lfbmClient.generateFromMetrics(
          nationDataForLFBM,
          {
            count: gdeltSignalCount,
            avg_tone: highRiskNations.length > 0 ? -0.3 : 0.1, // Negative if high risk nations
            alerts: highRiskNations.length,
          },
          {
            political: 50 + highRiskNations.length * 10,
            economic: 45,
            security: 50 + highRiskNations.length * 15,
            cyber: 40,
          }
        );

        const lfbmLatency = Date.now() - lfbmStartTime;
        console.log(`[EMERGENCY REFRESH] LFBM completed in ${lfbmLatency}ms`);

        // Prepare cache data
        const lfbmCacheData = {
          briefings,
          metadata: {
            region: 'Global',
            preset: 'global',
            timestamp: new Date().toISOString(),
            overallRisk: 'elevated' as const,
            source: 'emergency_refresh',
            mode: 'realtime',
            model: 'lfbm-vllm',
            estimatedCost: '$0.001',
            cached: false,
          },
        };

        // Write to Redis cache
        try {
          const redisCacheKey = 'intel-briefing:global';
          await redis.del(redisCacheKey);
          await redis.set(redisCacheKey, {
            data: lfbmCacheData,
            timestamp: Date.now(),
            generatedAt: new Date().toISOString(),
          }, { ex: REDIS_CACHE_TTL_SECONDS });
          console.log('[EMERGENCY REFRESH] ✅ LFBM result cached to Redis');
        } catch (redisError) {
          console.error('[EMERGENCY REFRESH] Redis cache write failed:', redisError);
        }

        return NextResponse.json({
          success: true,
          message: 'Emergency refresh completed via LFBM (250x cheaper!)',
          briefings,
          metadata: lfbmCacheData.metadata,
          usage: {
            source: 'lfbm_vllm',
            latencyMs: lfbmLatency,
            estimatedCost: '$0.001',
          },
        });
      } catch (lfbmError) {
        console.error('[EMERGENCY REFRESH] LFBM failed, falling back to Anthropic:', lfbmError);
        // Fall through to Anthropic
      }
    }

    const briefingResponse = await anthropic.messages.create({
      model: modelId,
      max_tokens: maxTokens,
      system: systemPrompt,
      messages: [{
        role: 'user',
        content: userPrompt
      }],
    });

    // Extract the text response
    const textBlock = briefingResponse.content.find(block => block.type === 'text');
    if (!textBlock || textBlock.type !== 'text') {
      throw new Error('No text response from Claude');
    }

    // Parse the JSON response
    let briefings;
    try {
      // Try to extract JSON from response
      const jsonMatch = textBlock.text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        briefings = JSON.parse(jsonMatch[0]);
      } else {
        throw new Error('No JSON found in response');
      }
    } catch (parseError) {
      console.error('Failed to parse Claude response:', parseError);
      return NextResponse.json({
        error: 'Failed to parse AI response',
        raw: textBlock.text
      }, { status: 500 });
    }

    // Estimate cost based on mode/depth
    const costEstimates: Record<string, string> = {
      'realtime': '$0.25-0.50',
      'historical-quick': '$0.25-0.50',
      'historical-standard': '$0.40-0.75',
      'historical-deep': '$1.00-2.00',
      'hybrid-quick': '$0.30-0.60',
      'hybrid-standard': '$0.50-1.00',
      'hybrid-deep': '$1.50-2.50',
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
        model: modelId,
        estimatedCost: costEstimates[costKey] || '$0.50-1.00',
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
      message: 'Emergency refresh completed',
      briefings,
      metadata: cacheData.metadata,
      usage: {
        inputTokens: briefingResponse.usage.input_tokens,
        outputTokens: briefingResponse.usage.output_tokens,
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

      // Check for Anthropic-specific errors
      if (error.message.includes('401') || error.message.includes('Unauthorized')) {
        errorDetails = 'Anthropic API key is invalid or expired. Please check ANTHROPIC_API_KEY in Vercel.';
        errorType = 'auth_error';
      } else if (error.message.includes('402') || error.message.includes('Payment')) {
        errorDetails = 'Anthropic account has insufficient credits. Please add credits at console.anthropic.com.';
        errorType = 'billing_error';
      } else if (error.message.includes('429') || error.message.includes('rate')) {
        errorDetails = 'Anthropic API rate limited. Please wait a moment and try again.';
        errorType = 'rate_limit';
      } else if (error.message.includes('model')) {
        errorDetails = `Model error: ${error.message}. The model ID may be incorrect.`;
        errorType = 'model_error';
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
