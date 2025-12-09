import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { getLFBMClient } from '@/lib/inference/LFBMClient';

export const runtime = 'edge';
export const maxDuration = 120; // Allow longer for deep analysis

/**
 * Historical Meta-Analysis API - POWERED BY LFBM (self-hosted vLLM)
 *
 * Leverages Qwen's training knowledge for historical pattern analysis
 * combined with GDELT data (2015-present) for pattern synthesis.
 *
 * NO EXTERNAL LLM DEPENDENCIES - 250x cheaper than Anthropic
 */

// Historical eras the model can analyze (from training data)
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
    const lfbmEndpoint = process.env.LFBM_ENDPOINT;

    if (!supabaseUrl || !supabaseKey) {
      return NextResponse.json({ error: 'Missing Supabase configuration' }, { status: 500 });
    }

    if (!lfbmEndpoint) {
      return NextResponse.json({ error: 'Missing LFBM_ENDPOINT - configure self-hosted vLLM' }, { status: 500 });
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
    let gdeltSummary: Record<string, number> = {};
    if (gdeltPeriod) {
      const { data: gdeltSignals } = await supabase
        .from('learning_events')
        .select('domain, data, timestamp')
        .eq('session_hash', 'gdelt_ingest')
        .gte('timestamp', gdeltPeriod.start)
        .lte('timestamp', gdeltPeriod.end)
        .limit(100);

      if (gdeltSignals?.length) {
        gdeltSummary = gdeltSignals.reduce((acc, s) => {
          acc[s.domain] = (acc[s.domain] || 0) + 1;
          return acc;
        }, {} as Record<string, number>);
      }
    }

    // Fetch nation context
    const { data: nations } = await supabase
      .from('nations')
      .select('code, name, transition_risk, regime')
      .order('transition_risk', { ascending: false })
      .limit(15);

    // Build focus description from eras and domains
    const eraLabels = eras.map(era => HISTORICAL_ERAS[era].label).join(', ');
    const domainLabels = domains.map(d => d).join(', ');
    const focusDescription = focus || `${eraLabels} - ${domainLabels}`;

    // Use LFBM for historical analysis
    const lfbmClient = getLFBMClient();
    const lfbmStartTime = Date.now();

    const response = await lfbmClient.generateHistoricalAnalysis({
      nations: (nations || []).map(n => ({
        code: n.code,
        name: n.name,
        risk: n.transition_risk || 0,
        trend: 0,
      })),
      gdeltSummary,
      focus: focusDescription,
      selectedEras: eras,
      depth,
    });

    const lfbmLatency = Date.now() - lfbmStartTime;

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
          total_tokens: existingUsage.total_tokens + response.tokens_generated,
        })
        .eq('user_id', user.id)
        .eq('month', new Date().toISOString().slice(0, 7));
    }

    return NextResponse.json({
      success: true,
      analysis: response.briefings,
      metadata: {
        eras,
        domains,
        focus,
        depth,
        model: 'lfbm-vllm',
        tokens: {
          output: response.tokens_generated,
        },
        latency_ms: Date.now() - startTime,
        lfbm_latency_ms: lfbmLatency,
        gdelt_signals: Object.keys(gdeltSummary).length > 0 ? 'included' : 'none',
        estimatedCost: depth === 'deep' ? '$0.003' : '$0.002',
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
      { id: 'quick', label: 'Quick Analysis', tokens: 1024, model: 'lfbm-vllm', tier: 'free', cost: '$0.001' },
      { id: 'standard', label: 'Standard Analysis', tokens: 1536, model: 'lfbm-vllm', tier: 'pro', cost: '$0.002' },
      { id: 'deep', label: 'Deep Analysis', tokens: 2048, model: 'lfbm-vllm', tier: 'enterprise', cost: '$0.003' },
    ],
    inference_engine: 'LFBM (self-hosted vLLM on RunPod)',
  });
}
