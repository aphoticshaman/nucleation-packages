import { NextRequest, NextResponse } from 'next/server';
import { requireAuth } from '@/lib/auth';
import {
  analyzeTransferEntropy,
  computeEdgeWeights,
  type TimeSeriesPoint,
  type TEConfig,
} from '@/lib/physics/transfer-entropy';

export const maxDuration = 60; // Allow longer computation

interface TERequest {
  mode: 'pairwise' | 'all';
  series: Array<{
    id: string;
    name: string;
    data: Array<{ timestamp: number; value: number }>;
  }>;
  config?: Partial<TEConfig>;
}

/**
 * POST /api/compute/transfer-entropy
 *
 * Compute Transfer Entropy for causal edge weighting.
 * Used to determine directed information flow between time series.
 */
export async function POST(request: NextRequest) {
  try {
    await requireAuth();

    const body: TERequest = await request.json();
    const { mode, series, config = {} } = body;

    if (!series || series.length < 2) {
      return NextResponse.json(
        { error: 'At least 2 time series required' },
        { status: 400 }
      );
    }

    // Validate series have enough data
    for (const s of series) {
      if (!s.data || s.data.length < 50) {
        return NextResponse.json(
          { error: `Series ${s.id} has insufficient data (need at least 50 points)` },
          { status: 400 }
        );
      }
    }

    if (mode === 'pairwise' && series.length === 2) {
      // Single pairwise computation
      const result = await analyzeTransferEntropy(
        series[0].id,
        series[1].id,
        series[0].data as TimeSeriesPoint[],
        series[1].data as TimeSeriesPoint[],
        config
      );

      return NextResponse.json({
        mode: 'pairwise',
        result: {
          ...result,
          sourceName: series[0].name,
          targetName: series[1].name,
        },
      });
    }

    // All pairs computation
    const nodes = series.map(s => ({
      id: s.id,
      timeSeries: s.data as TimeSeriesPoint[],
    }));

    const results = await computeEdgeWeights(nodes, config);

    // Add names to results
    const nameMap = new Map(series.map(s => [s.id, s.name]));
    const enrichedResults = results.map(r => ({
      ...r,
      sourceName: nameMap.get(r.sourceId) || r.sourceId,
      targetName: nameMap.get(r.targetId) || r.targetId,
    }));

    // Build adjacency matrix
    const n = series.length;
    const matrix: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
    const idxMap = new Map(series.map((s, i) => [s.id, i]));

    for (const edge of results) {
      const i = idxMap.get(edge.sourceId);
      const j = idxMap.get(edge.targetId);
      if (i !== undefined && j !== undefined) {
        matrix[i][j] = edge.normalizedTE;
      }
    }

    return NextResponse.json({
      mode: 'all',
      edges: enrichedResults,
      matrix,
      labels: series.map(s => s.name),
      summary: {
        totalEdges: enrichedResults.length,
        strongestEdge: enrichedResults[0] || null,
        avgTE: enrichedResults.reduce((s, e) => s + e.transferEntropy, 0) / enrichedResults.length || 0,
      },
    });
  } catch (error) {
    console.error('Transfer Entropy computation error:', error);
    return NextResponse.json(
      { error: 'Computation failed', details: (error as Error).message },
      { status: 500 }
    );
  }
}
