import { NextResponse } from 'next/server';
import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { createClient } from '@supabase/supabase-js';
import { cookies } from 'next/headers';

// Service role client
function getServiceClient() {
  return createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );
}

interface TrainingExample {
  id: string;
  domain: string;
  input: string;
  output: string;
  quality_score: number;
  weight: number;
  metadata: Record<string, unknown>;
  created_at: string;
}

// GET: Export training data in various formats for RunPod/fine-tuning
export async function GET(request: Request) {
  const supabase = createRouteHandlerClient({ cookies });
  const { searchParams } = new URL(request.url);

  // Check auth - allow admin or cron
  const authHeader = request.headers.get('authorization');
  const cronSecret = process.env.CRON_SECRET;

  let isAuthorized = false;

  if (cronSecret && authHeader === `Bearer ${cronSecret}`) {
    isAuthorized = true;
  } else {
    const { data: { user } } = await supabase.auth.getUser();
    if (user) {
      const { data: profile } = await supabase
        .from('profiles')
        .select('role')
        .eq('id', user.id)
        .single();
      isAuthorized = profile?.role === 'admin';
    }
  }

  if (!isAuthorized) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const format = searchParams.get('format') || 'alpaca'; // alpaca, chatml, sharegpt, jsonl
  const domain = searchParams.get('domain'); // Filter by domain
  const minQuality = parseFloat(searchParams.get('min_quality') || '0.6');
  const limit = parseInt(searchParams.get('limit') || '10000');
  const includeMetadata = searchParams.get('metadata') === 'true';

  const serviceClient = getServiceClient();

  // Fetch training examples
  let query = serviceClient
    .from('training_examples')
    .select('*')
    .gte('quality_score', minQuality)
    .order('weight', { ascending: false })
    .limit(limit);

  if (domain) {
    query = query.eq('domain', domain);
  }

  const { data: examples, error } = await query;

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  if (!examples || examples.length === 0) {
    return NextResponse.json({ error: 'No training data found' }, { status: 404 });
  }

  // Convert to requested format
  let output: string;
  let contentType: string;
  let filename: string;

  const timestamp = new Date().toISOString().split('T')[0];

  switch (format) {
    case 'alpaca': {
      // Alpaca format - good for Llama fine-tuning
      const alpacaData = (examples as TrainingExample[]).map(ex => ({
        instruction: ex.input,
        input: '',
        output: ex.output,
        ...(includeMetadata && {
          domain: ex.domain,
          quality: ex.quality_score,
          weight: ex.weight,
        }),
      }));
      output = JSON.stringify(alpacaData, null, 2);
      contentType = 'application/json';
      filename = `latticeforge-training-alpaca-${timestamp}.json`;
      break;
    }

    case 'chatml': {
      // ChatML format - good for OpenAI-style fine-tuning
      const chatmlData = (examples as TrainingExample[]).map(ex => ({
        messages: [
          {
            role: 'system',
            content: `You are a geopolitical intelligence analyst specializing in ${ex.domain}. Provide accurate, data-driven analysis.`,
          },
          { role: 'user', content: ex.input },
          { role: 'assistant', content: ex.output },
        ],
        ...(includeMetadata && { metadata: { domain: ex.domain, quality: ex.quality_score } }),
      }));
      output = chatmlData.map(d => JSON.stringify(d)).join('\n');
      contentType = 'application/jsonl';
      filename = `latticeforge-training-chatml-${timestamp}.jsonl`;
      break;
    }

    case 'sharegpt': {
      // ShareGPT format - good for various trainers
      const sharegptData = (examples as TrainingExample[]).map(ex => ({
        conversations: [
          { from: 'human', value: ex.input },
          { from: 'gpt', value: ex.output },
        ],
        system: `Geopolitical intelligence analyst - ${ex.domain}`,
        ...(includeMetadata && { domain: ex.domain, quality: ex.quality_score }),
      }));
      output = JSON.stringify(sharegptData, null, 2);
      contentType = 'application/json';
      filename = `latticeforge-training-sharegpt-${timestamp}.json`;
      break;
    }

    case 'jsonl': {
      // Raw JSONL - flexible format
      output = (examples as TrainingExample[])
        .map(ex => JSON.stringify({
          prompt: ex.input,
          completion: ex.output,
          domain: ex.domain,
          quality: ex.quality_score,
          weight: ex.weight,
          id: ex.id,
          created_at: ex.created_at,
        }))
        .join('\n');
      contentType = 'application/jsonl';
      filename = `latticeforge-training-raw-${timestamp}.jsonl`;
      break;
    }

    case 'csv': {
      // CSV format for analysis
      const headers = ['id', 'domain', 'input', 'output', 'quality_score', 'weight', 'created_at'];
      const rows = (examples as TrainingExample[]).map(ex =>
        [
          ex.id,
          ex.domain,
          `"${ex.input.replace(/"/g, '""')}"`,
          `"${ex.output.replace(/"/g, '""')}"`,
          ex.quality_score,
          ex.weight,
          ex.created_at,
        ].join(',')
      );
      output = [headers.join(','), ...rows].join('\n');
      contentType = 'text/csv';
      filename = `latticeforge-training-${timestamp}.csv`;
      break;
    }

    default:
      return NextResponse.json({ error: 'Invalid format' }, { status: 400 });
  }

  // Get stats
  const domainStats: Record<string, number> = {};
  (examples as TrainingExample[]).forEach(ex => {
    domainStats[ex.domain] = (domainStats[ex.domain] || 0) + 1;
  });

  // Return as downloadable file or JSON response
  if (searchParams.get('download') === 'true') {
    return new NextResponse(output, {
      headers: {
        'Content-Type': contentType,
        'Content-Disposition': `attachment; filename="${filename}"`,
      },
    });
  }

  return NextResponse.json({
    success: true,
    format,
    count: examples.length,
    domains: domainStats,
    avgQuality: (examples as TrainingExample[]).reduce((sum, ex) => sum + ex.quality_score, 0) / examples.length,
    data: format === 'jsonl' || format === 'chatml' ? output.split('\n') : JSON.parse(output),
    downloadUrl: `/api/training/export?format=${format}&download=true`,
  });
}

// POST: Generate training data summary/stats
export async function POST(request: Request) {
  const supabase = createRouteHandlerClient({ cookies });

  const { data: { user } } = await supabase.auth.getUser();
  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { data: profile } = await supabase
    .from('profiles')
    .select('role')
    .eq('id', user.id)
    .single();

  if (profile?.role !== 'admin') {
    return NextResponse.json({ error: 'Admin access required' }, { status: 403 });
  }

  const serviceClient = getServiceClient();

  // Get comprehensive stats
  const { data: totalCount } = await serviceClient
    .from('training_examples')
    .select('id', { count: 'exact', head: true });

  const { data: domainCounts } = await serviceClient
    .from('training_examples')
    .select('domain')
    .then(result => {
      if (!result.data) return { data: [] };
      const counts: Record<string, number> = {};
      result.data.forEach((r: { domain: string }) => {
        counts[r.domain] = (counts[r.domain] || 0) + 1;
      });
      return { data: Object.entries(counts).map(([domain, count]) => ({ domain, count })) };
    });

  const { data: qualityStats } = await serviceClient
    .from('training_examples')
    .select('quality_score');

  const avgQuality = qualityStats && qualityStats.length > 0
    ? qualityStats.reduce((sum: number, r: { quality_score: number }) => sum + r.quality_score, 0) / qualityStats.length
    : 0;

  const { data: recentExamples } = await serviceClient
    .from('training_examples')
    .select('created_at')
    .order('created_at', { ascending: false })
    .limit(1);

  const { data: predictions } = await serviceClient
    .from('predictions')
    .select('id', { count: 'exact', head: true });

  return NextResponse.json({
    stats: {
      totalExamples: totalCount,
      domainBreakdown: domainCounts,
      averageQuality: avgQuality.toFixed(3),
      lastGenerated: recentExamples?.[0]?.created_at || null,
      totalPredictions: predictions,
    },
    exportFormats: [
      { format: 'alpaca', description: 'Stanford Alpaca format - great for Llama/Mistral fine-tuning' },
      { format: 'chatml', description: 'ChatML format - OpenAI-style conversations' },
      { format: 'sharegpt', description: 'ShareGPT format - multi-turn conversations' },
      { format: 'jsonl', description: 'Raw JSONL - flexible for custom processing' },
      { format: 'csv', description: 'CSV - for analysis in spreadsheets' },
    ],
    runpodTips: {
      recommendedModel: 'mistralai/Mistral-7B-Instruct-v0.2',
      minExamplesRecommended: 1000,
      estimatedCostPer1000: '$2-5 on RunPod with A100',
      trainingScript: 'Use axolotl or unsloth for efficient fine-tuning',
    },
  });
}
