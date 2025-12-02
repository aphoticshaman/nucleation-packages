import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

export const runtime = 'edge';

// USGS Earthquake API - completely free
const USGS_API = 'https://earthquake.usgs.gov/fdsnws/event/1/query';

interface USGSFeature {
  id: string;
  properties: {
    mag: number;
    place: string;
    time: number;
    updated: number;
    type: string;
    title: string;
    alert: string | null;
    tsunami: number;
    sig: number; // Significance score
  };
  geometry: {
    coordinates: [number, number, number]; // lon, lat, depth
  };
}

interface USGSResponse {
  type: string;
  metadata: {
    generated: number;
    count: number;
    title: string;
  };
  features: USGSFeature[];
}

// Risk thresholds
const MAGNITUDE_THRESHOLD = 4.5; // Significant earthquakes
const SIG_THRESHOLD = 250; // USGS significance score

/**
 * Ingest USGS earthquake data for natural disaster monitoring
 */
export async function GET(req: Request) {
  const startTime = Date.now();

  // Verify cron secret
  const authHeader = req.headers.get('authorization');
  const cronSecret = process.env.CRON_SECRET;

  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );

  const results = {
    timestamp: new Date().toISOString(),
    earthquakes_fetched: 0,
    significant_events: 0,
    signals_stored: 0,
    errors: [] as string[],
  };

  try {
    // Fetch earthquakes from last 24 hours, magnitude 4.5+
    const params = new URLSearchParams({
      format: 'geojson',
      starttime: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
      minmagnitude: '4.5',
      orderby: 'magnitude',
    });

    const response = await fetch(`${USGS_API}?${params}`, {
      headers: { 'User-Agent': 'LatticeAI/1.0' },
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: `USGS API error: ${response.status}` },
        { status: 502 }
      );
    }

    const data: USGSResponse = await response.json();
    results.earthquakes_fetched = data.features.length;

    // Aggregate by region (approximate country from place string)
    const regionStats: Record<
      string,
      {
        count: number;
        maxMag: number;
        totalSig: number;
        hasTsunami: boolean;
      }
    > = {};

    const significantEvents: Array<{
      id: string;
      magnitude: number;
      place: string;
      time: string;
      significance: number;
      tsunami: boolean;
    }> = [];

    for (const feature of data.features) {
      const props = feature.properties;

      // Extract region from place (e.g., "10km NE of Tokyo, Japan" → "Japan")
      const placeParts = props.place?.split(',') || [];
      const region = placeParts[placeParts.length - 1]?.trim() || 'Unknown';

      if (!regionStats[region]) {
        regionStats[region] = { count: 0, maxMag: 0, totalSig: 0, hasTsunami: false };
      }

      regionStats[region].count++;
      regionStats[region].maxMag = Math.max(regionStats[region].maxMag, props.mag);
      regionStats[region].totalSig += props.sig;
      if (props.tsunami > 0) regionStats[region].hasTsunami = true;

      // Track significant events
      if (props.mag >= MAGNITUDE_THRESHOLD || props.sig >= SIG_THRESHOLD) {
        significantEvents.push({
          id: feature.id,
          magnitude: props.mag,
          place: props.place,
          time: new Date(props.time).toISOString(),
          significance: props.sig,
          tsunami: props.tsunami > 0,
        });
        results.significant_events++;
      }
    }

    // Store aggregated signals
    const numericFeatures: Record<string, number> = {
      total_earthquakes_24h: data.features.length,
      max_magnitude: Math.max(...data.features.map((f) => f.properties.mag), 0),
      total_significance: data.features.reduce((sum, f) => sum + f.properties.sig, 0),
      tsunami_events: data.features.filter((f) => f.properties.tsunami > 0).length,
      regions_affected: Object.keys(regionStats).length,
    };

    // Add region-specific magnitudes
    for (const [region, stats] of Object.entries(regionStats)) {
      const safeRegion = region.toLowerCase().replace(/\s+/g, '_').slice(0, 30);
      numericFeatures[`eq_${safeRegion}_max_mag`] = stats.maxMag;
      numericFeatures[`eq_${safeRegion}_count`] = stats.count;
    }

    // Store in learning_events
    const { error: insertError } = await supabase.from('learning_events').insert({
      type: 'signal_observation',
      timestamp: new Date().toISOString(),
      session_hash: 'usgs_ingest',
      user_tier: 'system',
      domain: 'natural_disaster',
      data: {
        numeric_features: numericFeatures,
        categorical_features: {
          source: 'usgs',
          regions: Object.keys(regionStats),
          significant_events: significantEvents.slice(0, 10), // Top 10
        },
      },
      metadata: {
        source: 'usgs_cron',
        version: '1.0.0',
        environment: process.env.NODE_ENV || 'production',
        usgs_generated: data.metadata.generated,
      },
    });

    if (insertError) {
      results.errors.push(`DB insert: ${insertError.message}`);
    } else {
      results.signals_stored = Object.keys(numericFeatures).length;
    }

    return NextResponse.json({
      ...results,
      latency_ms: Date.now() - startTime,
      regions: Object.keys(regionStats),
    });
  } catch (error) {
    console.error('USGS ingestion error:', error);
    return NextResponse.json(
      {
        error: 'Ingestion failed',
        details: error instanceof Error ? error.message : 'unknown',
        latency_ms: Date.now() - startTime,
      },
      { status: 500 }
    );
  }
}
