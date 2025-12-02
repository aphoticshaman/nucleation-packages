// LatticeForge Intelligence Brief - Premium Tier
// Intel-community style predictive forecasting with confidence levels
// Model: Claude Opus (~$0.22 per brief, reserved for top tier)
// Deploy: supabase functions deploy intel-brief

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

// Confidence level definitions (IC standard)
const CONFIDENCE_LEVELS = {
  HIGH: 'HIGH CONFIDENCE - Based on high-quality information from multiple independent sources; well-established analytical frameworks; strong historical precedent',
  MODERATE: 'MODERATE CONFIDENCE - Based on credibly sourced information but with some gaps; reasonable analytical basis; some historical precedent',
  LOW: 'LOW CONFIDENCE - Based on limited or fragmentary information; significant gaps in data; weak or no historical precedent; high uncertainty',
  INSUFFICIENT: 'INSUFFICIENT DATA - Lack adequate information to make a meaningful assessment',
  COMPLEX: 'HIGH UNCERTAINTY - Variables are too numerous and dynamic for reliable prediction; multiple plausible outcomes',
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

    // Verify premium tier
    const authHeader = req.headers.get('Authorization')
    if (!authHeader?.startsWith('Bearer ')) {
      return jsonResponse({ error: 'Authorization required' }, 401)
    }

    const token = authHeader.replace('Bearer ', '')

    // Check if user has premium tier (simplified - in production use proper auth)
    const { data: keyData } = await supabase
      .rpc('validate_api_key', { p_key: token })

    if (!keyData?.[0] || keyData[0].client_tier !== 'enterprise') {
      return jsonResponse({
        error: 'Intel Brief requires Enterprise tier',
        upgrade_url: '/pricing',
        current_tier: keyData?.[0]?.client_tier || 'unknown',
      }, 403)
    }

    const anthropicKey = Deno.env.get('ANTHROPIC_API_KEY')
    if (!anthropicKey) {
      throw new Error('ANTHROPIC_API_KEY not configured')
    }

    // Gather comprehensive data
    const signals = await gatherIntelSignals(supabase)

    // Generate intel-style brief with Opus
    const brief = await generateIntelBrief(anthropicKey, signals)

    // Record usage (premium pricing)
    await supabase.rpc('record_usage', {
      p_client_id: keyData[0].client_id,
      p_api_key_id: keyData[0].key_id,
      p_operation: 'analysis.intel_brief',
      p_analysis_tokens: brief.tokens_used,
      p_metadata: { model: 'claude-opus-4-20250514', brief_type: 'intel' },
    })

    // Store the brief
    const { data: savedBrief, error } = await supabase
      .from('briefs')
      .insert({
        content: brief.content,
        summary: brief.summary,
        signals_snapshot: signals,
        model: 'claude-opus-4-20250514',
        tokens_used: brief.tokens_used,
        metadata: {
          type: 'intel_brief',
          classification: 'UNCLASSIFIED // FOUO',
          forecasts: brief.forecasts,
          key_judgments: brief.key_judgments,
        },
      })
      .select()
      .single()

    if (error) throw error

    return jsonResponse({
      status: 'generated',
      brief: savedBrief,
      cost_estimate: `$${(brief.tokens_used * 0.00006).toFixed(4)}`,
    })

  } catch (error) {
    console.error('Intel Brief error:', error)
    return jsonResponse({ error: error.message }, 500)
  }
})

async function gatherIntelSignals(supabase: any) {
  // Get all available US signals
  const { data: usSignals } = await supabase
    .from('country_signals')
    .select('*')
    .eq('country_code', 'USA')
    .order('updated_at', { ascending: false })

  // Get major economy comparisons
  const { data: globalSignals } = await supabase
    .from('country_signals')
    .select('*')
    .in('country_code', ['CHN', 'DEU', 'JPN', 'GBR', 'FRA'])
    .order('updated_at', { ascending: false })

  // Get recent briefs for context
  const { data: recentBriefs } = await supabase
    .from('briefs')
    .select('summary, signals_snapshot, created_at')
    .order('created_at', { ascending: false })
    .limit(5)

  // Deduplicate and structure
  const usIndicators: Record<string, any> = {}
  for (const s of usSignals || []) {
    if (!usIndicators[s.indicator]) {
      usIndicators[s.indicator] = s
    }
  }

  const globalByCountry: Record<string, Record<string, any>> = {}
  for (const s of globalSignals || []) {
    if (!globalByCountry[s.country_code]) {
      globalByCountry[s.country_code] = {}
    }
    if (!globalByCountry[s.country_code][s.indicator]) {
      globalByCountry[s.country_code][s.indicator] = s
    }
  }

  return {
    us: usIndicators,
    global: globalByCountry,
    recent_assessments: recentBriefs || [],
    as_of: new Date().toISOString(),
  }
}

async function generateIntelBrief(apiKey: string, signals: any) {
  const usText = Object.entries(signals.us)
    .map(([key, val]: [string, any]) => `  ${key}: ${val.value} (${val.source})`)
    .join('\n')

  const globalText = Object.entries(signals.global)
    .map(([country, indicators]: [string, any]) => {
      const indText = Object.entries(indicators)
        .map(([k, v]: [string, any]) => `    ${k}: ${v.value}`)
        .join('\n')
      return `  ${country}:\n${indText}`
    })
    .join('\n')

  const recentContext = signals.recent_assessments
    .map((b: any) => `  - ${new Date(b.created_at).toLocaleDateString()}: ${b.summary}`)
    .join('\n')

  const prompt = `You are a senior intelligence analyst producing a National Intelligence Estimate (NIE) style brief for institutional investors.

Your assessments must use Intelligence Community confidence language:
- "We assess with HIGH CONFIDENCE that..." (strong evidence, multiple sources, clear precedent)
- "We judge with MODERATE CONFIDENCE that..." (credible evidence but gaps exist)
- "We assess with LOW CONFIDENCE that..." (fragmentary evidence, significant uncertainty)
- "We lack sufficient information to assess..." (data gaps prevent assessment)
- "The situation is too dynamic to predict with confidence, but we estimate..." (complex scenarios)

## Current US Economic Indicators
${usText || 'Limited data available'}

## Major Economy Comparison
${globalText || 'Limited data available'}

## Recent Assessment Context
${recentContext || 'No prior assessments available'}

## Your Task
Produce an intelligence brief with the following structure:

### CLASSIFICATION
UNCLASSIFIED // FOR OFFICIAL USE ONLY

### KEY JUDGMENTS
List 3-5 key judgments, each prefaced with confidence level. These are the most important takeaways. Example:
- (HIGH CONFIDENCE) The Federal Reserve will maintain current rates through Q1 2025...
- (MODERATE CONFIDENCE) China's economic slowdown will accelerate...

### ECONOMIC INTELLIGENCE ASSESSMENT

#### Current State Assessment
Characterize the current US economic regime with specific indicator references.

#### Near-Term Forecast (30-90 days)
Provide probabilistic forecasts for key variables:
- GDP trajectory: [forecast with confidence level]
- Inflation path: [forecast with confidence level]
- Employment outlook: [forecast with confidence level]
- Fed policy: [forecast with confidence level]

#### Medium-Term Outlook (6-12 months)
Broader trends and structural shifts expected.

### RISK ASSESSMENT MATRIX

| Risk Factor | Probability | Impact | Confidence | Warning Indicators |
|-------------|-------------|--------|------------|-------------------|
| [Risk 1]    | [%]         | [1-5]  | [H/M/L]    | [What to watch]   |

### SCENARIO ANALYSIS
Describe 3 scenarios:
1. **BASE CASE** (probability: X%) - Most likely outcome
2. **UPSIDE SCENARIO** (probability: Y%) - Better than expected
3. **DOWNSIDE SCENARIO** (probability: Z%) - Worse than expected

### STRATEGIC POSITIONING RECOMMENDATIONS
Specific, actionable recommendations with confidence levels.

### INTELLIGENCE GAPS
What additional information would increase confidence in these assessments?

### NEXT ASSESSMENT
When should this estimate be revisited? What events would trigger an update?

---

Use precise language. Quantify when possible. Flag uncertainty explicitly. This is for sophisticated institutional readers who need actionable intelligence, not reassurance.`

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: 'claude-opus-4-20250514',
      max_tokens: 4000,
      messages: [{ role: 'user', content: prompt }],
    }),
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(`Anthropic API error: ${error}`)
  }

  const data = await response.json()
  const content = data.content[0].text

  // Extract key judgments
  const keyJudgmentsMatch = content.match(/KEY JUDGMENTS[\s\S]*?(?=###|$)/i)
  const keyJudgments = keyJudgmentsMatch
    ? keyJudgmentsMatch[0].match(/\([A-Z\s]+CONFIDENCE\)[^\n]+/g) || []
    : []

  // Extract forecasts from near-term section
  const forecastMatch = content.match(/Near-Term Forecast[\s\S]*?(?=###|$)/i)
  const forecasts = forecastMatch ? forecastMatch[0] : ''

  // Extract first key judgment as summary
  const summary = keyJudgments[0] || content.split('.')[0] + '.'

  return {
    content,
    summary: summary.replace(/^\(.*?\)\s*/, ''), // Remove confidence prefix for summary
    key_judgments: keyJudgments,
    forecasts,
    tokens_used: data.usage.input_tokens + data.usage.output_tokens,
  }
}

function jsonResponse(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  })
}
