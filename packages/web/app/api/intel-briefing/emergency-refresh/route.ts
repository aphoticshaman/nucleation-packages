import { NextResponse } from 'next/server';
import Anthropic from '@anthropic-ai/sdk';
import { createClient } from '@supabase/supabase-js';

export const runtime = 'edge';
export const maxDuration = 60;

// Emergency endpoint to force-refresh intel data from Claude API
// Estimated cost: $1-3 per call
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

    const supabase = createClient(supabaseUrl, supabaseKey);

    // STRICT ADMIN AUTH: Validate user session and check admin role
    // Extract access token from cookie or Authorization header
    const authHeader = request.headers.get('Authorization');
    const cookieHeader = request.headers.get('cookie');

    let accessToken: string | null = null;

    // Try Authorization header first (Bearer token)
    if (authHeader?.startsWith('Bearer ')) {
      accessToken = authHeader.substring(7);
    }

    // Try cookie if no header
    if (!accessToken && cookieHeader) {
      const cookies = Object.fromEntries(
        cookieHeader.split('; ').map(c => c.split('=').map(decodeURIComponent))
      );
      accessToken = cookies['sb-access-token'] || cookies['supabase-auth-token'];
    }

    if (!accessToken) {
      return NextResponse.json({
        success: false,
        error: 'Authentication required - please log in'
      }, { status: 401 });
    }

    // Verify the token and get user
    const { data: { user }, error: authError } = await supabase.auth.getUser(accessToken);

    if (authError || !user) {
      return NextResponse.json({
        success: false,
        error: 'Invalid or expired session - please log in again'
      }, { status: 401 });
    }

    // Check if user is admin
    const { data: profile, error: profileError } = await supabase
      .from('profiles')
      .select('role')
      .eq('id', user.id)
      .single();

    if (profileError || !profile || profile.role !== 'admin') {
      console.warn(`Non-admin user ${user.id} attempted emergency refresh`);
      return NextResponse.json({
        success: false,
        error: 'Admin access required'
      }, { status: 403 });
    }

    console.log(`Admin ${user.id} initiated emergency refresh`);

    debugInfo.hasSupabaseUrl = !!supabaseUrl;
    debugInfo.hasSupabaseKey = !!supabaseKey;
    debugInfo.hasAnthropicKey = !!anthropicKey;
    debugInfo.anthropicKeyPrefix = anthropicKey ? anthropicKey.substring(0, 10) + '...' : 'MISSING';
    debugInfo.adminId = user.id;

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

    const currentDate = new Date().toISOString().split('T')[0];
    const today = new Date();
    const threeDaysAgo = new Date(today.getTime() - 72 * 60 * 60 * 1000);

    // Generate comprehensive briefings for all major domains
    const systemPrompt = `You are an expert intelligence analyst generating executive briefings.
TODAY'S DATE: ${currentDate}
You MUST provide current, accurate information reflecting events of the past 72 hours.

Generate actionable intelligence covering:
1. Political developments (elections, coups, leadership changes, policy shifts)
2. Economic indicators (markets, trade, sanctions, currency movements)
3. Security situations (conflicts, terrorism, military movements)
4. Technology/cyber events (breaches, AI developments, tech regulation)
5. Resource/energy updates (oil, gas, critical minerals, supply chains)

For EACH domain provide:
- WHAT: Key developments (specific, factual)
- WHO: Key actors involved
- WHERE: Geographic focus
- WHEN: Timeline/urgency
- WHY: Root causes and drivers
- US IMPACT: Direct implications for US interests
- OUTLOOK: 24-72 hour projection

Be specific. Name names. Cite dates. Focus on actionable intelligence.
CRITICAL: Use current information as of ${currentDate}. Reference current administration and ongoing events.`;

    const briefingResponse = await anthropic.messages.create({
      model: 'claude-haiku-4-5-20251001',
      max_tokens: 8000,
      system: systemPrompt,
      messages: [{
        role: 'user',
        content: `Generate comprehensive intelligence briefings for ALL domains covering the period from ${threeDaysAgo.toISOString()} to ${today.toISOString()}.

Format your response as JSON with this structure:
{
  "summary": "Executive summary of global situation",
  "political": "WHAT: ... WHO: ... WHERE: ... WHEN: ... WHY: ... US IMPACT: ... OUTLOOK: ...",
  "economic": "WHAT: ... WHO: ... WHERE: ... WHEN: ... WHY: ... US IMPACT: ... OUTLOOK: ...",
  "security": "WHAT: ... WHO: ... WHERE: ... WHEN: ... WHY: ... US IMPACT: ... OUTLOOK: ...",
  "scitech": "WHAT: ... WHO: ... WHERE: ... WHEN: ... WHY: ... US IMPACT: ... OUTLOOK: ...",
  "cyber": "WHAT: ... WHO: ... WHERE: ... WHEN: ... WHY: ... US IMPACT: ... OUTLOOK: ...",
  "energy": "WHAT: ... WHO: ... WHERE: ... WHEN: ... WHY: ... US IMPACT: ... OUTLOOK: ...",
  "military": "WHAT: ... WHO: ... WHERE: ... WHEN: ... WHY: ... US IMPACT: ... OUTLOOK: ...",
  "financial": "WHAT: ... WHO: ... WHERE: ... WHEN: ... WHY: ... US IMPACT: ... OUTLOOK: ...",
  "domestic": "WHAT: ... WHO: ... WHERE: ... WHEN: ... WHY: ... US IMPACT: ... OUTLOOK: ...",
  "nsm": "Recommended strategic actions for decision makers"
}

Respond ONLY with valid JSON.`
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

    // Cache the results in Supabase
    const cacheKey = `emergency_briefing_global_${currentDate}`;
    const cacheData = {
      preset: 'global',
      briefings,
      metadata: {
        timestamp: new Date().toISOString(),
        overallRisk: 'elevated', // Default to elevated for emergency refresh
        source: 'emergency_refresh',
        model: 'claude-haiku-4-5-20251001',
        estimatedCost: '$0.25-0.75',
      },
      generated_at: new Date().toISOString(),
    };

    // PURGE old cache before storing new data
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

      // Store fresh data
      await supabase
        .from('briefing_cache')
        .upsert({
          cache_key: cacheKey,
          data: cacheData,
          expires_at: new Date(Date.now() + 6 * 60 * 60 * 1000).toISOString(), // 6 hour expiry
        }, { onConflict: 'cache_key' });

      console.log('Cache purged and refreshed successfully');
    } catch (cacheError) {
      console.warn('Cache storage failed, continuing without cache:', cacheError);
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
