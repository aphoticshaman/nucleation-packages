import { NextResponse } from 'next/server';
import Anthropic from '@anthropic-ai/sdk';
import { createClient } from '@supabase/supabase-js';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';

export const runtime = 'edge';
export const maxDuration = 120; // Allow longer for deep analysis

/**
 * Historical Meta-Analysis API
 *
 * Leverages Claude's verified historical knowledge (ancient → training cutoff)
 * combined with GDELT data (2015-present) for pattern synthesis.
 *
 * Key design: NO hallucination risk because we only ask Claude about
 * historical events within its training data, not current events.
 */

// Historical eras Claude can confidently analyze
export const HISTORICAL_ERAS = {
  ancient: { label: 'Ancient Civilizations', range: '3000 BCE - 500 CE', topics: ['Egypt', 'Rome', 'Greece', 'Persia', 'China', 'India'] },
  medieval: { label: 'Medieval Period', range: '500 - 1500 CE', topics: ['Byzantine', 'Islamic Golden Age', 'Mongol Empire', 'Crusades', 'Black Death'] },
  earlyModern: { label: 'Early Modern', range: '1500 - 1800', topics: ['Age of Exploration', 'Colonialism', 'Enlightenment', 'Revolutions'] },
  industrial: { label: 'Industrial Era', range: '1800 - 1914', topics: ['Industrial Revolution', 'Nationalism', 'Imperialism', 'Great Game'] },
  worldWars: { label: 'World Wars', range: '1914 - 1945', topics: ['WWI', 'Interwar Period', 'Great Depression', 'WWII', 'Holocaust'] },
  coldWar: { label: 'Cold War', range: '1945 - 1991', topics: ['Nuclear Age', 'Proxy Wars', 'Decolonization', 'Space Race', 'Vietnam'] },
  postColdWar: { label: 'Post-Cold War', range: '1991 - 2010', topics: ['Globalization', 'War on Terror', 'Financial Crisis', 'Rise of China'] },
  modern: { label: 'Recent History', range: '2010 - 2024', topics: ['Arab Spring', 'Syria', 'Crimea', 'COVID-19', 'Ukraine War'] },
} as const;

// Analysis domains for meta-analysis
export const ANALYSIS_DOMAINS = {
  geopolitical: 'Rise and fall of powers, alliance patterns, territorial changes',
  economic: 'Trade routes, currency systems, economic cycles, resource competition',
  military: 'Military doctrine evolution, technology impact, conflict patterns',
  social: 'Population movements, revolutions, cultural shifts, pandemics',
  technological: 'Innovation diffusion, disruptive technologies, knowledge transfer',
  intelligence: 'Espionage evolution, information warfare, deception patterns',
} as const;

type HistoricalEra = keyof typeof HISTORICAL_ERAS;
type AnalysisDomain = keyof typeof ANALYSIS_DOMAINS;

interface MetaAnalysisRequest {
  eras: HistoricalEra[];
  domains: AnalysisDomain[];
  focus?: string; // Optional specific focus (e.g., "Ukraine conflict")
  gdeltPeriod?: {
    start: string; // ISO date
    end: string;
  };
  depth: 'quick' | 'standard' | 'deep';
}

export async function POST(request: Request) {
  const startTime = Date.now();

  try {
    // Parse request body
    const body: MetaAnalysisRequest = await request.json();
    const { eras = ['modern'], domains = ['geopolitical'], focus, gdeltPeriod, depth = 'standard' } = body;

    // Validate
    if (!eras.length || !domains.length) {
      return NextResponse.json({ error: 'Must specify at least one era and domain' }, { status: 400 });
    }

    // Check environment
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
    const anthropicKey = process.env.ANTHROPIC_API_KEY;

    if (!supabaseUrl || !supabaseKey || !anthropicKey) {
      return NextResponse.json({ error: 'Missing configuration' }, { status: 500 });
    }

    const supabase = createClient(supabaseUrl, supabaseKey);

    // Auth check - require at least logged-in user
    const cookieStore = await cookies();
    const authClient = createServerClient(
      supabaseUrl,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          getAll: () => cookieStore.getAll(),
          setAll: () => {},
        },
      }
    );

    const { data: { user }, error: authError } = await authClient.auth.getUser();
    if (authError || !user) {
      return NextResponse.json({ error: 'Authentication required' }, { status: 401 });
    }

    // Check subscription tier for deep analysis
    const { data: profile } = await supabase
      .from('profiles')
      .select('subscription_tier, role')
      .eq('id', user.id)
      .single();

    if (depth === 'deep' && profile?.subscription_tier !== 'enterprise' && profile?.role !== 'admin') {
      return NextResponse.json({
        error: 'Deep analysis requires enterprise tier',
        upgrade: '/pricing'
      }, { status: 403 });
    }

    // Fetch GDELT data if period specified
    let gdeltContext = '';
    if (gdeltPeriod) {
      const { data: gdeltSignals } = await supabase
        .from('learning_events')
        .select('domain, data, timestamp')
        .eq('session_hash', 'gdelt_ingest')
        .gte('timestamp', gdeltPeriod.start)
        .lte('timestamp', gdeltPeriod.end)
        .limit(100);

      if (gdeltSignals?.length) {
        const signalSummary = gdeltSignals.reduce((acc, s) => {
          acc[s.domain] = (acc[s.domain] || 0) + 1;
          return acc;
        }, {} as Record<string, number>);

        gdeltContext = `\n\nGDELT SIGNALS (${gdeltPeriod.start} to ${gdeltPeriod.end}):\n${
          Object.entries(signalSummary)
            .map(([domain, count]) => `- ${domain}: ${count} events`)
            .join('\n')
        }`;
      }
    }

    // Fetch nation context
    const { data: nations } = await supabase
      .from('nations')
      .select('code, name, transition_risk, regime')
      .order('transition_risk', { ascending: false })
      .limit(15);

    const nationContext = nations?.length
      ? `\n\nCURRENT NATION RISK DATA:\n${nations.map(n =>
          `- ${n.name} (${n.code}): ${((n.transition_risk || 0) * 100).toFixed(0)}% transition risk`
        ).join('\n')}`
      : '';

    // Build era context
    const eraContext = eras.map(era => {
      const e = HISTORICAL_ERAS[era];
      return `${e.label} (${e.range}): ${e.topics.join(', ')}`;
    }).join('\n');

    // Build domain context
    const domainContext = domains.map(d =>
      `${d.toUpperCase()}: ${ANALYSIS_DOMAINS[d]}`
    ).join('\n');

    // Token limits by depth
    const tokenLimits = {
      quick: 2000,
      standard: 4000,
      deep: 8000,
    };

    // Build the meta-analysis prompt
    const systemPrompt = `You are a senior intelligence analyst specializing in historical pattern analysis.

YOUR UNIQUE ADVANTAGE:
You have comprehensive knowledge of human history from ancient civilizations through your training cutoff.
This allows you to identify patterns, parallels, and precedents that inform current analysis.

ANALYSIS METHODOLOGY:
1. PATTERN IDENTIFICATION: Find recurring patterns across historical eras
2. PRECEDENT ANALYSIS: Identify historical analogues to current situations
3. CAUSAL CHAINS: Trace how historical decisions led to outcomes
4. CYCLE DETECTION: Identify geopolitical, economic, and social cycles
5. LESSONS EXTRACTION: Distill actionable insights from historical examples

CRITICAL RULES:
- Ground ALL claims in verifiable historical events
- Cite specific examples with dates when possible
- Distinguish between established facts and analytical interpretation
- Acknowledge uncertainty when extrapolating patterns
- NO speculation about events after your training cutoff
- Connect historical patterns to the data provided about current nations

OUTPUT VOICE: Academic intelligence analysis - precise, sourced, actionable.`;

    const userPrompt = `HISTORICAL META-ANALYSIS REQUEST

ERAS TO ANALYZE:
${eraContext}

DOMAINS OF FOCUS:
${domainContext}

${focus ? `SPECIFIC FOCUS: ${focus}` : ''}
${nationContext}
${gdeltContext}

ANALYSIS DEPTH: ${depth.toUpperCase()}

REQUESTED OUTPUT:
Generate a comprehensive meta-analysis covering:

1. **HISTORICAL PATTERNS**: Recurring patterns across the specified eras relevant to the domains
   - Include specific examples with dates
   - Identify cycle lengths where applicable

2. **PRECEDENT ANALYSIS**: Historical analogues to current situations
   - Map historical events to current nation risk profiles
   - Identify which precedents suggest escalation vs. de-escalation

3. **CAUSAL MECHANISMS**: How similar situations have historically resolved
   - Economic pressures → political outcomes
   - Military buildups → conflict/deterrence patterns
   - Alliance shifts → power transitions

4. **LESSONS FOR ANALYSTS**: Actionable insights from historical analysis
   - Early warning indicators based on historical precedent
   - Common failure modes in similar situations
   - Successful intervention patterns

5. **CONFIDENCE ASSESSMENT**: Your confidence level in each pattern identified
   - HIGH: Well-documented with multiple examples
   - MEDIUM: Supported but limited sample size
   - LOW: Speculative pattern requiring more evidence

Format as structured JSON:
{
  "executive_summary": "<2-3 sentence overview>",
  "patterns": [
    {
      "name": "<pattern name>",
      "description": "<description>",
      "historical_examples": ["<example1>", "<example2>"],
      "current_relevance": "<how it applies now>",
      "confidence": "HIGH|MEDIUM|LOW"
    }
  ],
  "precedents": [
    {
      "historical_event": "<event>",
      "date_range": "<dates>",
      "modern_parallel": "<current situation>",
      "outcome_then": "<what happened>",
      "implications_now": "<what it suggests>"
    }
  ],
  "lessons": [
    {
      "insight": "<actionable insight>",
      "basis": "<historical basis>",
      "application": "<how to apply>"
    }
  ],
  "warnings": ["<early warning indicators based on historical patterns>"],
  "confidence_note": "<overall confidence assessment>"
}`;

    const anthropic = new Anthropic({ apiKey: anthropicKey });

    const response = await anthropic.messages.create({
      model: depth === 'deep' ? 'claude-sonnet-4-20250514' : 'claude-haiku-4-5-20251001',
      max_tokens: tokenLimits[depth],
      system: systemPrompt,
      messages: [{ role: 'user', content: userPrompt }],
    });

    // Extract response
    const textBlock = response.content.find(b => b.type === 'text');
    if (!textBlock || textBlock.type !== 'text') {
      throw new Error('No text response from Claude');
    }

    // Parse JSON response
    let analysis;
    try {
      const jsonMatch = textBlock.text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        analysis = JSON.parse(jsonMatch[0]);
      } else {
        throw new Error('No JSON found');
      }
    } catch {
      // Return raw text if JSON parsing fails
      analysis = { raw: textBlock.text };
    }

    // Log usage for billing
    const { data: existingUsage } = await supabase
      .from('api_usage')
      .select('request_count, total_tokens')
      .eq('user_id', user.id)
      .eq('month', new Date().toISOString().slice(0, 7))
      .single();

    if (existingUsage) {
      await supabase
        .from('api_usage')
        .update({
          request_count: existingUsage.request_count + 1,
          total_tokens: existingUsage.total_tokens + response.usage.input_tokens + response.usage.output_tokens,
        })
        .eq('user_id', user.id)
        .eq('month', new Date().toISOString().slice(0, 7));
    }

    return NextResponse.json({
      success: true,
      analysis,
      metadata: {
        eras,
        domains,
        focus,
        depth,
        model: depth === 'deep' ? 'claude-sonnet-4-20250514' : 'claude-haiku-4-5-20251001',
        tokens: {
          input: response.usage.input_tokens,
          output: response.usage.output_tokens,
        },
        latency_ms: Date.now() - startTime,
        gdelt_signals: gdeltContext ? 'included' : 'none',
      },
    });

  } catch (error) {
    console.error('[META-ANALYSIS] Error:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Analysis failed',
    }, { status: 500 });
  }
}

// GET endpoint to retrieve available eras and domains
export async function GET() {
  return NextResponse.json({
    eras: Object.entries(HISTORICAL_ERAS).map(([key, value]) => ({
      id: key,
      ...value,
    })),
    domains: Object.entries(ANALYSIS_DOMAINS).map(([key, value]) => ({
      id: key,
      description: value,
    })),
    depth_options: [
      { id: 'quick', label: 'Quick Analysis', tokens: 2000, model: 'claude-haiku-4-5-20251001', tier: 'free' },
      { id: 'standard', label: 'Standard Analysis', tokens: 4000, model: 'claude-haiku-4-5-20251001', tier: 'pro' },
      { id: 'deep', label: 'Deep Analysis', tokens: 8000, model: 'claude-sonnet-4-20250514', tier: 'enterprise' },
    ],
  });
}
