import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { mapUserTierToPricing, TIER_CAPABILITIES } from '@/lib/doctrine/types';

export const runtime = 'edge';

/**
 * POST /api/export - Export data in various formats
 * Requires: Analyst tier or higher (rawDataExport)
 *
 * Supported formats: json, csv
 * Future: pdf, xlsx (requires additional libraries)
 */
export async function POST(req: Request) {
  const userTier = req.headers.get('x-user-tier') || 'free';
  const pricingTier = mapUserTierToPricing(userTier);

  // Check rawDataExport capability using tier comparison
  const tierRank: Record<string, number> = {
    observer: 0,
    operational: 1,
    integrated: 2,
    stewardship: 3,
  };

  if (tierRank[pricingTier] < 1) { // Operational+ has api_access, but export needs Analyst+
    return NextResponse.json(
      { error: 'Data export requires Analyst tier or higher' },
      { status: 403 }
    );
  }

  try {
    const body = await req.json();
    const { data_type, format = 'json', filters } = body as {
      data_type: 'signals' | 'alerts' | 'briefings' | 'doctrines';
      format: 'json' | 'csv' | 'pdf' | 'xlsx';
      filters?: Record<string, unknown>;
    };

    if (!data_type) {
      return NextResponse.json(
        { error: 'data_type is required' },
        { status: 400 }
      );
    }

    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );

    let data: unknown[] = [];
    let filename = `latticeforge-${data_type}-${Date.now()}`;

    // Fetch data based on type
    switch (data_type) {
      case 'signals': {
        const { data: signals } = await supabase
          .from('learning_events')
          .select('timestamp, domain, data')
          .eq('type', 'signal_observation')
          .order('timestamp', { ascending: false })
          .limit(1000);
        data = signals || [];
        break;
      }
      case 'alerts': {
        const { data: alerts } = await supabase
          .from('learning_events')
          .select('timestamp, domain, data')
          .eq('type', 'signal_observation')
          .gte('timestamp', new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString())
          .order('timestamp', { ascending: false })
          .limit(500);
        data = (alerts || []).filter((e) => {
          const risk = (e.data as Record<string, unknown>)?.numeric_features as Record<string, number> | undefined;
          return (risk?.risk_score || risk?.gdelt_tone_risk || 0) > 0.6;
        });
        break;
      }
      default:
        return NextResponse.json(
          { error: `Unsupported data_type: ${data_type}` },
          { status: 400 }
        );
    }

    // Format data
    if (format === 'json') {
      return NextResponse.json({
        filename: `${filename}.json`,
        data,
        count: data.length,
        exported_at: new Date().toISOString(),
      });
    }

    if (format === 'csv') {
      if (data.length === 0) {
        return NextResponse.json({ csv: '', count: 0 });
      }

      // Simple CSV conversion
      const headers = Object.keys(data[0] as object);
      const csvRows = [
        headers.join(','),
        ...data.map((row) =>
          headers.map((h) => {
            const val = (row as Record<string, unknown>)[h];
            if (typeof val === 'object') return JSON.stringify(val);
            return String(val ?? '');
          }).join(',')
        ),
      ];

      return new NextResponse(csvRows.join('\n'), {
        headers: {
          'Content-Type': 'text/csv',
          'Content-Disposition': `attachment; filename="${filename}.csv"`,
        },
      });
    }

    if (format === 'pdf' || format === 'xlsx') {
      return NextResponse.json({
        error: `${format.toUpperCase()} export coming soon`,
        note: 'PDF/XLSX export requires additional server-side rendering',
        alternative: 'Use JSON or CSV format for now',
      }, { status: 501 });
    }

    return NextResponse.json(
      { error: `Unsupported format: ${format}` },
      { status: 400 }
    );
  } catch (error) {
    console.error('Export error:', error);
    return NextResponse.json(
      { error: 'Export failed' },
      { status: 500 }
    );
  }
}
