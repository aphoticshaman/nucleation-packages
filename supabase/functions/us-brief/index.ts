// LatticeForge US Deep Dive Brief - Supabase Edge Function
// Detailed US economic analysis using Claude Sonnet
// Budget: ~$0.045 per brief, 4x daily = ~$5.40/month
// Deploy: supabase functions deploy us-brief

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

// Key US indicators to analyze
const US_INDICATORS = [
  'fed_funds_rate',
  'yield_curve_10y2y',
  'vix',
  'oil_wti',
  'usd_eur',
  'us_unemployment',
  'us_cpi',
  'us_m2_money_supply',
  'gdp_growth',
  'inflation',
  'current_account',
  'debt_to_gdp',
]

// Anomaly thresholds
const ANOMALY_THRESHOLDS = {
  vix: { high: 25, extreme: 35, label: 'VIX (Fear Index)' },
  yield_curve_10y2y: { low: 0, inverted: -0.5, label: 'Yield Curve' },
  us_unemployment: { elevated: 5, high: 7, label: 'Unemployment' },
  inflation: { elevated: 3, high: 5, label: 'Inflation' },
  debt_to_gdp: { elevated: 100, high: 120, label: 'Debt/GDP' },
  oil_wti: { high: 90, extreme: 110, label: 'Oil Price' },
}

interface USSignals {
  indicators: Record<string, { value: number; date?: string; source: string }>;
  anomalies: Array<{ indicator: string; label: string; value: number; severity: 'warning' | 'critical'; message: string }>;
  trends: Array<{ indicator: string; direction: 'up' | 'down' | 'stable'; change?: number }>;
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    const anthropicKey = Deno.env.get('ANTHROPIC_API_KEY')
    if (!anthropicKey) {
      throw new Error('ANTHROPIC_API_KEY not configured')
    }

    // Gather US signals from database
    const signals = await gatherUSSignals(supabase)

    // Check if we should generate (significant change or scheduled)
    const url = new URL(req.url)
    const force = url.searchParams.get('force') === 'true'

    if (!force && signals.anomalies.length === 0) {
      // Check last brief time
      const { data: lastBrief } = await supabase
        .from('briefs')
        .select('created_at')
        .eq('model', 'claude-sonnet-4-20250514')
        .order('created_at', { ascending: false })
        .limit(1)
        .single()

      if (lastBrief) {
        const hoursSinceLastBrief = (Date.now() - new Date(lastBrief.created_at).getTime()) / (1000 * 60 * 60)
        if (hoursSinceLastBrief < 6 && signals.anomalies.length === 0) {
          return jsonResponse({
            status: 'skipped',
            reason: 'No anomalies and last brief was less than 6 hours ago',
            next_scheduled: new Date(new Date(lastBrief.created_at).getTime() + 6 * 60 * 60 * 1000).toISOString(),
          })
        }
      }
    }

    // Generate detailed brief with Sonnet
    const brief = await generateDetailedBrief(anthropicKey, signals)

    // Store the brief
    const { data: savedBrief, error } = await supabase
      .from('briefs')
      .insert({
        content: brief.content,
        summary: brief.summary,
        signals_snapshot: signals,
        model: 'claude-sonnet-4-20250514',
        tokens_used: brief.tokens_used,
        metadata: {
          type: 'us_deep_dive',
          anomaly_count: signals.anomalies.length,
          sections: brief.sections,
        },
      })
      .select()
      .single()

    if (error) throw error

    return jsonResponse({
      status: 'generated',
      brief: savedBrief,
      cost_estimate: `$${(brief.tokens_used * 0.000015).toFixed(4)}`,
    })

  } catch (error) {
    console.error('US Brief error:', error)
    return jsonResponse({ error: error.message }, 500)
  }
})

async function gatherUSSignals(supabase: any): Promise<USSignals> {
  // Get latest US indicators from database
  const { data: rawSignals } = await supabase
    .from('country_signals')
    .select('indicator, value, year, source, metadata, updated_at')
    .eq('country_code', 'USA')
    .order('updated_at', { ascending: false })

  const indicators: Record<string, { value: number; date?: string; source: string }> = {}
  const seen = new Set<string>()

  for (const signal of rawSignals || []) {
    if (seen.has(signal.indicator)) continue
    seen.add(signal.indicator)
    indicators[signal.indicator] = {
      value: parseFloat(signal.value),
      date: signal.metadata?.date || `${signal.year}`,
      source: signal.source,
    }
  }

  // Detect anomalies
  const anomalies: USSignals['anomalies'] = []

  for (const [key, thresholds] of Object.entries(ANOMALY_THRESHOLDS)) {
    const signal = indicators[key]
    if (!signal) continue

    const value = signal.value

    if (key === 'vix') {
      if (value >= thresholds.extreme) {
        anomalies.push({
          indicator: key,
          label: thresholds.label,
          value,
          severity: 'critical',
          message: `VIX at ${value.toFixed(1)} indicates extreme fear/volatility in markets`,
        })
      } else if (value >= thresholds.high) {
        anomalies.push({
          indicator: key,
          label: thresholds.label,
          value,
          severity: 'warning',
          message: `VIX elevated at ${value.toFixed(1)}, above normal range`,
        })
      }
    }

    if (key === 'yield_curve_10y2y') {
      if (value <= thresholds.inverted) {
        anomalies.push({
          indicator: key,
          label: thresholds.label,
          value,
          severity: 'critical',
          message: `Yield curve deeply inverted at ${value.toFixed(2)}%, historically precedes recessions`,
        })
      } else if (value <= thresholds.low) {
        anomalies.push({
          indicator: key,
          label: thresholds.label,
          value,
          severity: 'warning',
          message: `Yield curve flat/inverted at ${value.toFixed(2)}%, recession signal`,
        })
      }
    }

    if (key === 'us_unemployment') {
      if (value >= thresholds.high) {
        anomalies.push({
          indicator: key,
          label: thresholds.label,
          value,
          severity: 'critical',
          message: `Unemployment at ${value.toFixed(1)}%, significantly elevated`,
        })
      } else if (value >= thresholds.elevated) {
        anomalies.push({
          indicator: key,
          label: thresholds.label,
          value,
          severity: 'warning',
          message: `Unemployment rising to ${value.toFixed(1)}%`,
        })
      }
    }

    if (key === 'inflation') {
      if (value >= thresholds.high) {
        anomalies.push({
          indicator: key,
          label: thresholds.label,
          value,
          severity: 'critical',
          message: `Inflation at ${value.toFixed(1)}%, Fed likely to maintain tight policy`,
        })
      } else if (value >= thresholds.elevated) {
        anomalies.push({
          indicator: key,
          label: thresholds.label,
          value,
          severity: 'warning',
          message: `Inflation elevated at ${value.toFixed(1)}%, above 2% target`,
        })
      }
    }

    if (key === 'oil_wti') {
      if (value >= thresholds.extreme) {
        anomalies.push({
          indicator: key,
          label: thresholds.label,
          value,
          severity: 'critical',
          message: `Oil at $${value.toFixed(0)}/barrel, major inflation risk`,
        })
      } else if (value >= thresholds.high) {
        anomalies.push({
          indicator: key,
          label: thresholds.label,
          value,
          severity: 'warning',
          message: `Oil elevated at $${value.toFixed(0)}/barrel`,
        })
      }
    }
  }

  // Sort anomalies by severity
  anomalies.sort((a, b) => (a.severity === 'critical' ? -1 : 1))

  return { indicators, anomalies, trends: [] }
}

async function generateDetailedBrief(apiKey: string, signals: USSignals) {
  const indicatorText = Object.entries(signals.indicators)
    .map(([key, val]) => `- ${key}: ${val.value} (${val.source}, ${val.date})`)
    .join('\n')

  const anomalyText = signals.anomalies.length > 0
    ? signals.anomalies.map(a => `- [${a.severity.toUpperCase()}] ${a.message}`).join('\n')
    : 'No significant anomalies detected.'

  const prompt = `You are a senior macroeconomic analyst at a hedge fund. Generate a detailed US economic briefing.

## Current US Economic Indicators
${indicatorText}

## Detected Anomalies
${anomalyText}

## Your Task
Write a comprehensive briefing with these sections:

### 1. Executive Summary (2-3 sentences)
Key takeaway for decision-makers.

### 2. Current Economic Regime
Characterize the current state: expansion, contraction, stagflation, recovery, etc.
Reference specific indicators that support your assessment.

### 3. Anomaly Analysis
${signals.anomalies.length > 0
  ? 'For each anomaly detected, explain:\n- Root cause hypothesis\n- Historical precedents\n- Potential market implications\n- Timeline for resolution'
  : 'No significant anomalies. Note any indicators approaching warning thresholds.'}

### 4. Risk Matrix
Identify top 3 risks with probability and impact assessment.

### 5. Positioning Recommendations
Specific, actionable recommendations for:
- Equity allocation
- Fixed income duration
- Sector tilts
- Hedging strategies

### 6. Key Events to Watch
Upcoming data releases, Fed meetings, or events that could shift the narrative.

Be direct, quantitative, and actionable. No fluff.`

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 2000,
      messages: [{ role: 'user', content: prompt }],
    }),
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(`Anthropic API error: ${error}`)
  }

  const data = await response.json()
  const content = data.content[0].text

  // Extract executive summary as the summary
  const summaryMatch = content.match(/Executive Summary[:\s]*\n+([\s\S]*?)(?=\n#|$)/i)
  const summary = summaryMatch ? summaryMatch[1].trim().split('\n')[0] : content.split('.')[0] + '.'

  // Count sections
  const sections = (content.match(/^###?\s/gm) || []).length

  return {
    content,
    summary,
    sections,
    tokens_used: data.usage.input_tokens + data.usage.output_tokens,
  }
}

function jsonResponse(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  })
}
