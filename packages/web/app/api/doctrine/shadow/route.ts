import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import {
  mapUserTierToPricing,
  TIER_CAPABILITIES,
  DEFAULT_DOCTRINES,
  type ShadowEvaluation
} from '@/lib/doctrine/types';

export const runtime = 'edge';

function getSupabase() {
  return createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );
}

/**
 * POST /api/doctrine/shadow - Run shadow evaluation of proposed doctrine changes
 * Requires: Stewardship tier
 *
 * This evaluates how a proposed doctrine change would have affected historical outputs
 * WITHOUT changing any production data.
 */
export async function POST(req: Request) {
  const userTier = req.headers.get('x-user-tier') || 'free';
  const pricingTier = mapUserTierToPricing(userTier);

  if (!TIER_CAPABILITIES[pricingTier].shadow_evaluate) {
    return NextResponse.json(
      { error: 'Shadow evaluation requires Stewardship tier' },
      { status: 403 }
    );
  }

  try {
    const body = await req.json();
    const {
      doctrine_id,
      proposed_parameters,
      evaluation_days = 7
    } = body as {
      doctrine_id: string;
      proposed_parameters: Record<string, number | string | boolean>;
      evaluation_days?: number;
    };

    if (!doctrine_id || !proposed_parameters) {
      return NextResponse.json(
        { error: 'doctrine_id and proposed_parameters are required' },
        { status: 400 }
      );
    }

    // Find the doctrine (from DB or defaults)
    const defaultDoctrine = DEFAULT_DOCTRINES.find((d, i) =>
      `default-${i}` === doctrine_id || d.name === doctrine_id
    );

    if (!defaultDoctrine) {
      return NextResponse.json(
        { error: 'Doctrine not found' },
        { status: 404 }
      );
    }

    // Fetch historical events for evaluation
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - evaluation_days);

    const supabase = getSupabase();
    const { data: events, error: eventsError } = await supabase
      .from('learning_events')
      .select('id, timestamp, domain, data')
      .eq('type', 'signal_observation')
      .gte('timestamp', startDate.toISOString())
      .order('timestamp', { ascending: false })
      .limit(500);

    if (eventsError) {
      console.error('Events fetch error:', eventsError);
    }

    const historicalEvents = events || [];

    // Run shadow evaluation
    const currentParams = defaultDoctrine.rule_definition.parameters;
    const currentOutputs: Array<{ event_id: string; timestamp: string; domain: string; output_value: number; confidence: number }> = [];
    const proposedOutputs: Array<{ event_id: string; timestamp: string; domain: string; output_value: number; confidence: number }> = [];
    let divergenceCount = 0;

    for (const event of historicalEvents) {
      const eventData = event.data as Record<string, unknown> || {};
      const numericFeatures = eventData.numeric_features as Record<string, number> || {};

      // Compute output with current parameters
      const currentOutput = computeDoctrineOutput(
        defaultDoctrine.rule_definition.type,
        currentParams,
        numericFeatures
      );

      // Compute output with proposed parameters
      const proposedOutput = computeDoctrineOutput(
        defaultDoctrine.rule_definition.type,
        { ...currentParams, ...proposed_parameters },
        numericFeatures
      );

      currentOutputs.push({
        event_id: event.id,
        timestamp: event.timestamp,
        domain: event.domain,
        output_value: currentOutput.value,
        confidence: currentOutput.confidence
      });

      proposedOutputs.push({
        event_id: event.id,
        timestamp: event.timestamp,
        domain: event.domain,
        output_value: proposedOutput.value,
        confidence: proposedOutput.confidence
      });

      // Check for divergence (different classification)
      if (Math.abs(currentOutput.value - proposedOutput.value) > 0.1) {
        divergenceCount++;
      }
    }

    const evaluation: ShadowEvaluation = {
      id: `shadow-${Date.now()}`,
      doctrine_id,
      proposed_changes: { parameters: proposed_parameters },
      evaluation_period: {
        start: startDate.toISOString(),
        end: new Date().toISOString()
      },
      results: {
        total_events_evaluated: historicalEvents.length,
        current_outputs: currentOutputs.slice(0, 20), // Return sample
        proposed_outputs: proposedOutputs.slice(0, 20),
        divergence_count: divergenceCount,
        divergence_rate: historicalEvents.length > 0
          ? divergenceCount / historicalEvents.length
          : 0
      },
      status: 'completed',
      created_at: new Date().toISOString(),
      completed_at: new Date().toISOString()
    };

    return NextResponse.json({
      evaluation,
      summary: {
        doctrine: defaultDoctrine.name,
        events_evaluated: historicalEvents.length,
        divergence_count: divergenceCount,
        divergence_rate: `${(evaluation.results.divergence_rate * 100).toFixed(1)}%`,
        recommendation: divergenceCount > historicalEvents.length * 0.2
          ? 'HIGH_IMPACT - Review carefully before applying'
          : divergenceCount > historicalEvents.length * 0.05
          ? 'MODERATE_IMPACT - Standard review recommended'
          : 'LOW_IMPACT - Safe to proceed'
      }
    });
  } catch (error) {
    console.error('Shadow evaluation error:', error);
    return NextResponse.json(
      { error: 'Shadow evaluation failed' },
      { status: 500 }
    );
  }
}

/**
 * Compute doctrine output based on rule type and parameters
 */
function computeDoctrineOutput(
  type: string,
  params: Record<string, unknown>,
  features: Record<string, number>
): { value: number; confidence: number } {
  switch (type) {
    case 'threshold': {
      const metric = params.metric as string || 'risk_score';
      const stableThreshold = params.stable_threshold as number || 0.7;
      const unstableThreshold = params.unstable_threshold as number || 0.3;
      const value = features[metric] || features.risk_score || features.gdelt_tone_risk || 0.5;

      let classification = 0.5; // neutral
      if (value >= stableThreshold) classification = 0.2; // stable
      else if (value <= unstableThreshold) classification = 0.8; // unstable

      return { value: classification, confidence: 0.8 };
    }

    case 'weight': {
      let weightedSum = 0;
      let totalWeight = 0;

      for (const [key, weight] of Object.entries(params)) {
        if (typeof weight === 'number' && features[key] !== undefined) {
          weightedSum += features[key] * weight;
          totalWeight += weight;
        }
      }

      const value = totalWeight > 0 ? weightedSum / totalWeight : 0.5;
      return { value, confidence: totalWeight > 0 ? 0.9 : 0.5 };
    }

    case 'mapping': {
      // For mapping types, return the raw risk score
      const riskScore = features.risk_score || features.gdelt_tone_risk || 0.5;
      return { value: riskScore, confidence: 0.85 };
    }

    default:
      return { value: 0.5, confidence: 0.5 };
  }
}
