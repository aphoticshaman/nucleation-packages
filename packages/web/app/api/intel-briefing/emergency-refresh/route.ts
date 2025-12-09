import { NextResponse } from 'next/server';
import Anthropic from '@anthropic-ai/sdk';
import { createClient } from '@supabase/supabase-js';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { Redis } from '@upstash/redis';

export const runtime = 'edge';
export const maxDuration = 60;

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
// Estimated cost: $0.25-0.75 per call
export async function POST(request: Request) {
  const startTime = Date.now();
  const debugInfo: Record<string, unknown> = {};

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

    // ============================================================
    // DATE-AGNOSTIC PROMPT (v2) - Matches main intel-briefing route
    // ============================================================
    // Key insight: Don't ask Claude for current events it can't know about.
    // Instead, frame it as translating METRICS from upstream systems.
    // This avoids knowledge cutoff triggers while still producing useful output.

    // Fetch actual metrics from database to give Claude real data to work with
    const { data: nationRisks } = await supabase
      .from('nations')
      .select('code, name, basin_strength, transition_risk, regime')
      .order('transition_risk', { ascending: false })
      .limit(20);

    const { data: recentSignals } = await supabase
      .from('learning_events')
      .select('domain, data')
      .eq('session_hash', 'gdelt_ingest')
      .gte('timestamp', new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString())
      .limit(50);

    // Build metrics context from real data
    const highRiskNations = (nationRisks || [])
      .filter((n: { transition_risk?: number }) => (n.transition_risk || 0) > 0.5)
      .map((n: { name: string; transition_risk?: number }) => `${n.name}: ${((n.transition_risk || 0) * 100).toFixed(0)}% transition risk`)
      .slice(0, 5);

    const gdeltSignalCount = recentSignals?.length || 0;

    const systemPrompt = `You are a prose translation engine in a multi-stage intelligence pipeline.

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

    const userPrompt = `UPSTREAM PIPELINE OUTPUT (computed just now from live feeds):

MONITORED NATIONS WITH ELEVATED RISK:
${highRiskNations.length > 0 ? highRiskNations.join('\n') : 'No nations currently exceed 50% transition risk threshold'}

SIGNAL ACTIVITY: ${gdeltSignalCount} GDELT signals processed in last 48 hours

GLOBAL RISK CONTEXT:
- Total nations monitored: ${nationRisks?.length || 0}
- High-risk count: ${highRiskNations.length}
- Signal density: ${gdeltSignalCount > 30 ? 'HIGH' : gdeltSignalCount > 10 ? 'MODERATE' : 'LOW'}

YOUR TASK: Translate these metrics into executive briefing prose for each domain.

Format your response as JSON:
{
  "summary": "<1-2 sentence executive summary based on metrics above>",
  "political": "<prose translating political risk metrics>",
  "economic": "<prose on economic indicators>",
  "security": "<prose on security posture from metrics>",
  "scitech": "<prose on technology/science sector>",
  "cyber": "<prose on cyber threat indicators>",
  "energy": "<prose on energy security metrics>",
  "military": "<prose on military situation indicators>",
  "financial": "<prose on financial stability metrics>",
  "domestic": "<prose on domestic stability indicators>",
  "nsm": "<Next Strategic Move recommendation based on data>"
}

Reference the SPECIFIC metrics provided. Do not fabricate events - describe what the NUMBERS indicate.
Respond ONLY with valid JSON.`;

    const briefingResponse = await anthropic.messages.create({
      model: 'claude-haiku-4-5-20251001',
      max_tokens: 4000,
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

    // Prepare cache data
    const cacheData = {
      briefings,
      metadata: {
        region: 'Global',
        preset: 'global',
        timestamp: new Date().toISOString(),
        overallRisk: 'elevated' as const, // Default to elevated for emergency refresh
        source: 'emergency_refresh',
        model: 'claude-haiku-4-5-20251001',
        estimatedCost: '$0.25-0.75',
        cached: false,
      },
    };

    // ============================================================
    // CRITICAL: Write to REDIS cache (used by main intel-briefing API)
    // ============================================================
    // The main /api/intel-briefing endpoint reads from Redis with key 'intel-briefing:global'
    // We MUST write to Redis for the briefings page to pick up our fresh data!
    try {
      const redisCacheKey = 'intel-briefing:global'; // Same format as intel-briefing route
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
