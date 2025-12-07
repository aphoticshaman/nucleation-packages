import { NextRequest, NextResponse } from 'next/server';
import { requireAuth } from '@/lib/auth';
import {
  executePipeline,
  performArchaeology,
  type ConceptPrimitive,
} from '@/lib/prometheus/cognitive-pipeline';
import {
  detectPatternAnomaly,
  detectTemporalBreak,
  buildKnowledgeQuadrant,
  prioritizeNovelties,
} from '@/lib/prometheus/novelty-detection';

interface PROMETHEUSRequest {
  mode: 'archaeology' | 'synthesis' | 'full' | 'novelty';
  targetSubject: string;
  relatedDomains?: string[];
  catalystConcept?: {
    domain: string;
    name: string;
    definition: string;
    axioms: string[];
  };
  // For novelty detection
  observations?: number[];
  existingKnowledge?: string[];
}

/**
 * POST /api/prometheus/extract
 *
 * PROMETHEUS Protocol API - Novel knowledge extraction
 * using the 5-stage cognitive pipeline.
 */
export async function POST(request: NextRequest) {
  try {
    await requireAuth();

    const body: PROMETHEUSRequest = await request.json();
    const {
      mode,
      targetSubject,
      relatedDomains = [],
      catalystConcept,
      observations,
      existingKnowledge = [],
    } = body;

    if (!targetSubject) {
      return NextResponse.json(
        { error: 'Target subject is required' },
        { status: 400 }
      );
    }

    if (mode === 'archaeology') {
      // Stage 1 only: Latent Space Archaeology
      const gradients = performArchaeology(targetSubject, relatedDomains);

      return NextResponse.json({
        mode: 'archaeology',
        targetSubject,
        gradients,
        summary: {
          totalGradients: gradients.length,
          highPriorityCount: gradients.filter(g => g.excavationPriority > 0.7).length,
          unknownKnownsFound: gradients.reduce((sum, g) => sum + g.unknownKnowns.length, 0),
        },
      });
    }

    if (mode === 'novelty') {
      // Novelty detection on provided data
      if (!observations || observations.length < 50) {
        return NextResponse.json(
          { error: 'Need at least 50 observations for novelty detection' },
          { status: 400 }
        );
      }

      const patternAnomalies = detectPatternAnomaly(observations);
      const temporalBreaks = detectTemporalBreak(observations);

      const allSignals = [...patternAnomalies, ...temporalBreaks];
      const prioritized = prioritizeNovelties(allSignals);
      const quadrant = buildKnowledgeQuadrant(allSignals, existingKnowledge);

      return NextResponse.json({
        mode: 'novelty',
        signals: prioritized,
        quadrant,
        summary: {
          totalSignals: allSignals.length,
          patternAnomalies: patternAnomalies.length,
          temporalBreaks: temporalBreaks.length,
          highConfidenceCount: allSignals.filter(s => s.confidence > 0.8).length,
          unknownKnownsDiscovered: quadrant.unknownKnowns.length,
        },
      });
    }

    if (mode === 'synthesis' || mode === 'full') {
      // Full pipeline execution
      if (!catalystConcept) {
        return NextResponse.json(
          { error: 'Catalyst concept required for synthesis' },
          { status: 400 }
        );
      }

      const catalyst: ConceptPrimitive = {
        id: `catalyst_${Date.now()}`,
        domain: catalystConcept.domain,
        name: catalystConcept.name,
        definition: catalystConcept.definition,
        axioms: catalystConcept.axioms,
        connections: [],
      };

      const pipelineState = await executePipeline(
        targetSubject,
        relatedDomains,
        catalyst
      );

      return NextResponse.json({
        mode: mode,
        pipeline: pipelineState,
        summary: {
          artifactsGenerated: pipelineState.artifacts.length,
          validatedArtifacts: pipelineState.artifacts.filter(
            a => a.validationStatus === 'validated'
          ).length,
          outputsGenerated: pipelineState.outputs.length,
          completionPercentage: pipelineState.progress,
        },
      });
    }

    return NextResponse.json({ error: 'Invalid mode' }, { status: 400 });
  } catch (error) {
    console.error('PROMETHEUS extraction error:', error);
    return NextResponse.json(
      { error: 'Extraction failed', details: (error as Error).message },
      { status: 500 }
    );
  }
}

/**
 * GET /api/prometheus/extract
 *
 * Get information about the PROMETHEUS protocol
 */
export async function GET() {
  return NextResponse.json({
    protocol: 'P.R.O.M.E.T.H.E.U.S.',
    fullName: 'Protocol for Recursive Optimization, Meta-Enhanced Theoretical Heuristic Extraction, and Universal Synthesis',
    stages: [
      {
        id: 1,
        name: 'LATENT SPACE ARCHAEOLOGY',
        description: 'Deep scan for unknown knowns - insights that exist implicitly',
      },
      {
        id: 2,
        name: 'NOVEL SYNTHESIS METHOD',
        description: 'Force-fusion of heterogeneous concepts',
      },
      {
        id: 3,
        name: 'RIGOROUS THEORETICAL VALIDATION',
        description: 'Mathematical proof and ablation testing',
      },
      {
        id: 4,
        name: 'XYZA OPERATIONALIZATION',
        description: 'Code implementation following XYZA framework',
      },
      {
        id: 5,
        name: 'OUTPUT GENERATION',
        description: 'Structured deliverable package',
      },
    ],
    endpoints: {
      archaeology: 'POST with mode="archaeology" for Stage 1 only',
      novelty: 'POST with mode="novelty" for anomaly detection',
      synthesis: 'POST with mode="synthesis" for concept fusion',
      full: 'POST with mode="full" for complete pipeline',
    },
  });
}
