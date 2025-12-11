/**
 * Elle Insight Reports API
 *
 * Endpoints for Elle's autonomous research capture system.
 * Elle creates insights, progresses them through PROMETHEUS stages,
 * and surfaces them to admins only when validated.
 *
 * POST /api/elle/insights - Create new insight (Elle only)
 * GET /api/elle/insights - List insights (admin only)
 * PATCH /api/elle/insights - Update insight stage (Elle or admin)
 */

import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/lib/supabase/server';

// PROMETHEUS stages in order
const STAGES = [
  'latent_archaeology',
  'novel_synthesis',
  'theoretical_validation',
  'xyza_operationalization',
  'output_generation',
] as const;

type InsightStage = (typeof STAGES)[number];

interface CreateInsightRequest {
  title: string;
  target_subject: string;
  summary?: string;
  archaeology_data?: {
    vertical_scan?: string;
    horizontal_scan?: string;
    temporal_scan?: string;
    gradient_of_ignorance: string;
    unknown_knowns: string[];
  };
  trigger_context?: {
    conversation_id?: string;
    message_that_triggered?: string;
    user_query?: string;
  };
  tags?: string[];
}

interface UpdateInsightRequest {
  insight_id: string;
  advance_stage?: boolean;
  stage_data?: Record<string, unknown>;
  confidence_score?: number;
  confidence_type?: 'derived' | 'hypothetical' | 'empirical' | 'contested';
  code_artifact?: {
    filename: string;
    language: string;
    content: string;
    execution_result?: string;
    tests_passed?: boolean;
    test_output?: string;
  };
  impact_analysis?: {
    novelty_claim: string;
    humanity_impact: string;
    ai_impact: string;
    asymmetric_lever: string;
    estimated_value: 'low' | 'medium' | 'high' | 'breakthrough';
  };
}

// POST - Create new insight (Elle initiating autonomous research)
export async function POST(request: NextRequest) {
  try {
    const supabase = await createClient();
    const body: CreateInsightRequest = await request.json();

    // Validate required fields
    if (!body.title || !body.target_subject) {
      return NextResponse.json(
        { error: 'Missing required fields: title, target_subject' },
        { status: 400 }
      );
    }

    // Validate gradient_of_ignorance if archaeology_data provided
    if (body.archaeology_data && !body.archaeology_data.gradient_of_ignorance) {
      return NextResponse.json(
        { error: 'archaeology_data must include gradient_of_ignorance' },
        { status: 400 }
      );
    }

    const { data: insight, error } = await supabase
      .from('insight_reports')
      .insert({
        title: body.title,
        target_subject: body.target_subject,
        summary: body.summary,
        archaeology_data: body.archaeology_data || {},
        trigger_context: body.trigger_context || {},
        tags: body.tags || [],
        current_stage: 'latent_archaeology',
        status: 'in_progress',
        created_by: 'elle',
        stage_timestamps: {
          latent_archaeology: new Date().toISOString(),
        },
      })
      .select()
      .single();

    if (error) {
      console.error('Error creating insight:', error);
      return NextResponse.json({ error: error.message }, { status: 500 });
    }

    return NextResponse.json({
      success: true,
      insight,
      message: `Insight "${body.title}" created. Beginning PROMETHEUS pipeline.`,
    });
  } catch (error) {
    console.error('Insight creation error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// GET - List insights (with filters)
export async function GET(request: NextRequest) {
  try {
    const supabase = await createClient();
    const { searchParams } = new URL(request.url);

    // Parse query params
    const status = searchParams.get('status');
    const stage = searchParams.get('stage');
    const minConfidence = searchParams.get('min_confidence');
    const limit = parseInt(searchParams.get('limit') || '50');
    const offset = parseInt(searchParams.get('offset') || '0');

    // Build query
    let query = supabase
      .from('insight_reports')
      .select('*')
      .order('created_at', { ascending: false })
      .range(offset, offset + limit - 1);

    if (status) {
      query = query.eq('status', status);
    }
    if (stage) {
      query = query.eq('current_stage', stage);
    }
    if (minConfidence) {
      query = query.gte('confidence_score', parseFloat(minConfidence));
    }

    const { data: insights, error, count } = await query;

    if (error) {
      console.error('Error fetching insights:', error);
      return NextResponse.json({ error: error.message }, { status: 500 });
    }

    // Get stats
    const { data: stats } = await supabase.rpc('get_insight_stats');

    return NextResponse.json({
      insights,
      stats,
      pagination: {
        limit,
        offset,
        total: count,
      },
    });
  } catch (error) {
    console.error('Insight fetch error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// PATCH - Update insight (advance stage, add data)
export async function PATCH(request: NextRequest) {
  try {
    const supabase = await createClient();
    const body: UpdateInsightRequest = await request.json();

    if (!body.insight_id) {
      return NextResponse.json(
        { error: 'Missing insight_id' },
        { status: 400 }
      );
    }

    // Get current insight
    const { data: current, error: fetchError } = await supabase
      .from('insight_reports')
      .select('*')
      .eq('id', body.insight_id)
      .single();

    if (fetchError || !current) {
      return NextResponse.json(
        { error: 'Insight not found' },
        { status: 404 }
      );
    }

    // Build update object
    const updates: Record<string, unknown> = {};

    // Advance to next stage if requested
    if (body.advance_stage) {
      const currentIndex = STAGES.indexOf(current.current_stage as InsightStage);
      if (currentIndex < STAGES.length - 1) {
        const nextStage = STAGES[currentIndex + 1];
        updates.current_stage = nextStage;

        // Update stage timestamps
        updates.stage_timestamps = {
          ...(current.stage_timestamps || {}),
          [nextStage]: new Date().toISOString(),
        };

        // If reaching output_generation, mark for review
        if (nextStage === 'output_generation') {
          updates.status = 'awaiting_review';
        }
      }
    }

    // Update stage-specific data
    if (body.stage_data) {
      const stage = (updates.current_stage || current.current_stage) as InsightStage;

      switch (stage) {
        case 'novel_synthesis':
          updates.nsm_data = { ...current.nsm_data, ...body.stage_data };
          break;
        case 'theoretical_validation':
          updates.theoretical_validation = {
            ...current.theoretical_validation,
            ...body.stage_data,
          };
          break;
        case 'xyza_operationalization':
          updates.xyza_data = { ...current.xyza_data, ...body.stage_data };
          break;
      }
    }

    // Update confidence (bounded by epistemic humility)
    if (body.confidence_score !== undefined) {
      // Cap at 0.95 per epistemic bounds
      updates.confidence_score = Math.min(0.95, body.confidence_score);
    }
    if (body.confidence_type) {
      updates.confidence_type = body.confidence_type;
    }

    // Add code artifact
    if (body.code_artifact) {
      const existingArtifacts = current.code_artifacts || [];
      updates.code_artifacts = [...existingArtifacts, body.code_artifact];
    }

    // Update impact analysis
    if (body.impact_analysis) {
      updates.impact_analysis = body.impact_analysis;
    }

    // Perform update
    const { data: updated, error: updateError } = await supabase
      .from('insight_reports')
      .update(updates)
      .eq('id', body.insight_id)
      .select()
      .single();

    if (updateError) {
      console.error('Error updating insight:', updateError);
      return NextResponse.json({ error: updateError.message }, { status: 500 });
    }

    return NextResponse.json({
      success: true,
      insight: updated,
      message: body.advance_stage
        ? `Advanced to stage: ${updates.current_stage}`
        : 'Insight updated',
    });
  } catch (error) {
    console.error('Insight update error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
