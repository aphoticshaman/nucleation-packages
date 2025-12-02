// LatticeForge API - Supabase Edge Function
// Deploy: supabase functions deploy api

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
}

interface AuthResult {
  client_id: string
  client_tier: string
  key_id: string
}

serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const url = new URL(req.url)
    const path = url.pathname.replace('/api', '')

    // Create Supabase client with service role (bypasses RLS)
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    // Authenticate API key
    const authHeader = req.headers.get('Authorization')
    if (!authHeader?.startsWith('Bearer ')) {
      return jsonResponse({ error: 'Missing API key' }, 401)
    }

    const apiKey = authHeader.replace('Bearer ', '')
    const { data: authData, error: authError } = await supabase
      .rpc('validate_api_key', { p_key: apiKey })

    if (authError || !authData?.length) {
      return jsonResponse({ error: 'Invalid API key' }, 401)
    }

    const auth: AuthResult = authData[0]

    // Route handling
    switch (true) {
      // Health check
      case path === '/health' || path === '/':
        return jsonResponse({ status: 'ok', tier: auth.client_tier })

      // Verify API key
      case path === '/verify':
        return jsonResponse({ valid: true, tier: auth.client_tier })

      // Get usage
      case path === '/usage':
        return handleUsage(supabase, auth)

      // Fuse signals
      case path === '/signals/fuse' && req.method === 'POST':
        return handleFuse(supabase, auth, await req.json())

      // Detect anomalies
      case path === '/signals/detect':
        return handleDetect(supabase, auth)

      // Get indicators
      case path.startsWith('/indicators/'):
        const indicator = path.replace('/indicators/', '')
        return handleIndicator(supabase, auth, indicator)

      default:
        return jsonResponse({ error: 'Not found' }, 404)
    }
  } catch (error) {
    console.error('API Error:', error)
    return jsonResponse({ error: 'Internal server error' }, 500)
  }
})

// ============================================
// HANDLERS
// ============================================

async function handleUsage(supabase: any, auth: AuthResult) {
  const period = new Date().toISOString().slice(0, 7) + '-01'

  const { data, error } = await supabase
    .from('usage_records')
    .select('signal_tokens, fusion_tokens, analysis_tokens, total_tokens')
    .eq('client_id', auth.client_id)
    .eq('billing_period', period)

  if (error) throw error

  const totals = data.reduce((acc: any, row: any) => ({
    signal_tokens: acc.signal_tokens + row.signal_tokens,
    fusion_tokens: acc.fusion_tokens + row.fusion_tokens,
    analysis_tokens: acc.analysis_tokens + row.analysis_tokens,
    total_tokens: acc.total_tokens + row.total_tokens,
  }), { signal_tokens: 0, fusion_tokens: 0, analysis_tokens: 0, total_tokens: 0 })

  // Get limits
  const { data: limits } = await supabase
    .from('tier_limits')
    .select('*')
    .eq('tier', auth.client_tier)
    .single()

  return jsonResponse({
    tokens_used: totals.total_tokens,
    tokens_limit: limits?.max_total_tokens ?? 15000,
    breakdown: totals,
    period_start: period,
    period_end: getEndOfMonth(period),
  })
}

async function handleFuse(supabase: any, auth: AuthResult, body: any) {
  const sources = body.sources ?? ['SEC', 'FRED']

  // Check usage limits first
  const tokens = sources.length * 20 // 20 tokens per source
  const { data: limitCheck } = await supabase
    .rpc('check_usage_limit', {
      p_client_id: auth.client_id,
      p_operation: 'fusion',
      p_tokens: tokens,
    })

  if (!limitCheck?.[0]?.allowed) {
    return jsonResponse({
      error: 'Usage limit exceeded',
      current: limitCheck?.[0]?.current_usage,
      limit: limitCheck?.[0]?.max_usage,
    }, 429)
  }

  // Record usage
  await supabase.rpc('record_usage', {
    p_client_id: auth.client_id,
    p_api_key_id: auth.key_id,
    p_operation: 'fusion.multi_source',
    p_fusion_tokens: tokens,
    p_metadata: { sources },
  })

  // Generate fused signal (simplified for MVP)
  const signal = generateFusedSignal(sources)

  return jsonResponse(signal)
}

async function handleDetect(supabase: any, auth: AuthResult) {
  // Record usage
  await supabase.rpc('record_usage', {
    p_client_id: auth.client_id,
    p_api_key_id: auth.key_id,
    p_operation: 'analysis.anomaly_fingerprint',
    p_analysis_tokens: 50,
  })

  // Get recent alerts for this client
  const { data: alerts } = await supabase
    .from('alerts')
    .select('*')
    .eq('client_id', auth.client_id)
    .is('read_at', null)
    .order('created_at', { ascending: false })
    .limit(10)

  return jsonResponse(alerts ?? [])
}

async function handleIndicator(supabase: any, auth: AuthResult, indicator: string) {
  // Record usage
  await supabase.rpc('record_usage', {
    p_client_id: auth.client_id,
    p_api_key_id: auth.key_id,
    p_operation: 'source.fetch',
    p_signal_tokens: 10,
    p_metadata: { indicator },
  })

  // Return indicator value (simplified)
  const indicators: Record<string, number> = {
    volatility: 0.23 + Math.random() * 0.1,
    sentiment: 0.6 + Math.random() * 0.2 - 0.1,
    momentum: 0.45 + Math.random() * 0.3,
    fear_greed: 55 + Math.random() * 20,
  }

  return jsonResponse({
    indicator,
    value: indicators[indicator] ?? 0.5,
    timestamp: new Date().toISOString(),
  })
}

// ============================================
// HELPERS
// ============================================

function jsonResponse(data: any, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  })
}

function getEndOfMonth(period: string): string {
  const date = new Date(period)
  date.setMonth(date.getMonth() + 1)
  date.setDate(0)
  return date.toISOString().split('T')[0]
}

function generateFusedSignal(sources: string[]) {
  // Simplified signal generation for MVP
  // In production, this would call the actual engine
  const baseValue = 0.5 + (Math.random() - 0.5) * 0.3

  const regimes = ['risk-on', 'risk-off', 'transitional', 'uncertain'] as const
  const regime = regimes[Math.floor(baseValue * 4)] ?? 'uncertain'

  return {
    value: Math.round(baseValue * 1000) / 1000,
    confidence: 0.7 + Math.random() * 0.2,
    timestamp: new Date().toISOString(),
    sources,
    regime,
    contributors: sources.map(s => ({
      signal: s,
      weight: 1 / sources.length,
      contribution: baseValue / sources.length,
    })),
  }
}
