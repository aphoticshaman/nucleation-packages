import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';

// Edge runtime for speed
export const runtime = 'edge';

/**
 * Guardian Governance API
 *
 * Human-in-the-loop rule management:
 * - GET: Fetch dashboard data (metrics, proposals, active rules)
 * - POST: Accept/reject proposals, activate rules, rollback
 *
 * All changes require admin role and are logged for audit.
 */

export async function GET(req: Request) {
  try {
    const cookieStore = await cookies();
    const supabase = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          get(name: string) {
            return cookieStore.get(name)?.value;
          },
          set() {},
          remove() {},
        },
      }
    );

    // Verify admin
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { data: profile } = await supabase
      .from('profiles')
      .select('role')
      .eq('id', user.id)
      .single();

    if (!profile || (profile as { role?: string }).role !== 'admin') {
      return NextResponse.json({ error: 'Admin access required' }, { status: 403 });
    }

    // Parse query params
    const url = new URL(req.url);
    const view = url.searchParams.get('view') || 'dashboard';

    if (view === 'dashboard') {
      // Fetch all dashboard data in parallel
      const [
        activeRulesResult,
        latestMetricsResult,
        pendingProposalsResult,
        disagreementsResult,
        recentAuditResult,
      ] = await Promise.all([
        supabase.from('v_guardian_active_rules').select('*'),
        supabase.from('v_guardian_latest_metrics').select('*'),
        supabase.from('v_guardian_pending_proposals').select('*').limit(10),
        supabase.from('v_guardian_disagreements').select('*').limit(20),
        supabase.from('guardian_audit_log').select('*').order('performed_at', { ascending: false }).limit(20),
      ]);

      // Compute aggregate metrics
      const metrics = latestMetricsResult.data || [];
      const globalMetrics = metrics.find(m => m.domain === null) || {};

      return NextResponse.json({
        activeRules: activeRulesResult.data || [],
        ruleCount: (activeRulesResult.data || []).length,

        metrics: {
          global: globalMetrics,
          byDomain: metrics.filter(m => m.domain !== null),
        },

        proposals: {
          pending: pendingProposalsResult.data || [],
          pendingCount: (pendingProposalsResult.data || []).length,
        },

        disagreements: disagreementsResult.data || [],

        auditLog: recentAuditResult.data || [],

        summary: {
          totalEvaluations24h: (globalMetrics as { total_evaluations?: number }).total_evaluations || 0,
          accuracy: (globalMetrics as { accuracy_pct?: number }).accuracy_pct || null,
          hallucinationRate: (globalMetrics as { hallucination_rate_pct?: number }).hallucination_rate_pct || null,
          pendingReviews: (pendingProposalsResult.data || []).length + (disagreementsResult.data || []).length,
        },
      });
    }

    if (view === 'rules') {
      // Fetch all rule versions for a specific rule
      const ruleName = url.searchParams.get('rule_name');
      if (!ruleName) {
        return NextResponse.json({ error: 'rule_name required' }, { status: 400 });
      }

      const { data: versions } = await supabase
        .from('guardian_rules')
        .select('*')
        .eq('rule_name', ruleName)
        .order('version', { ascending: false });

      return NextResponse.json({ ruleName, versions: versions || [] });
    }

    if (view === 'metrics_history') {
      // Fetch metrics over time
      const domain = url.searchParams.get('domain') || null;
      const days = parseInt(url.searchParams.get('days') || '30');

      let query = supabase
        .from('guardian_metrics')
        .select('*')
        .eq('granularity', 'daily')
        .gte('period_start', new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString())
        .order('period_start', { ascending: true });

      if (domain) {
        query = query.eq('domain', domain);
      }

      const { data } = await query;
      return NextResponse.json({ domain, days, metrics: data || [] });
    }

    return NextResponse.json({ error: 'Invalid view' }, { status: 400 });

  } catch (error) {
    console.error('Guardian API error:', error);
    return NextResponse.json({ error: 'Internal error' }, { status: 500 });
  }
}

export async function POST(req: Request) {
  try {
    const cookieStore = await cookies();
    const supabase = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          get(name: string) {
            return cookieStore.get(name)?.value;
          },
          set() {},
          remove() {},
        },
      }
    );

    // Verify admin
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { data: profile } = await supabase
      .from('profiles')
      .select('role')
      .eq('id', user.id)
      .single();

    if (!profile || (profile as { role?: string }).role !== 'admin') {
      return NextResponse.json({ error: 'Admin access required' }, { status: 403 });
    }

    const body = await req.json();
    const { action } = body;

    // ═══════════════════════════════════════════════════════════════════════
    // ACCEPT PROPOSAL - Human approves Elle's suggested rule change
    // ═══════════════════════════════════════════════════════════════════════
    if (action === 'accept_proposal') {
      const { proposalId, notes } = body;

      // Get proposal
      const { data: proposal, error: proposalError } = await supabase
        .from('guardian_rule_proposals')
        .select('*')
        .eq('id', proposalId)
        .single();

      if (proposalError || !proposal) {
        return NextResponse.json({ error: 'Proposal not found' }, { status: 404 });
      }

      const p = proposal as {
        id: string;
        proposed_rule_name: string;
        proposed_domain: string;
        proposed_config: Record<string, unknown>;
        rationale: string;
        status: string;
      };

      if (p.status !== 'pending') {
        return NextResponse.json({ error: 'Proposal already processed' }, { status: 400 });
      }

      // Get current version number for this rule
      const { data: existingRule } = await supabase
        .from('guardian_rules')
        .select('version')
        .eq('rule_name', p.proposed_rule_name)
        .order('version', { ascending: false })
        .limit(1)
        .single();

      const newVersion = ((existingRule as { version?: number })?.version || 0) + 1;

      // Create new rule version
      const { data: newRule, error: createError } = await supabase
        .from('guardian_rules')
        .insert({
          rule_name: p.proposed_rule_name,
          domain: p.proposed_domain,
          rule_type: 'threshold',  // TODO: infer from proposal
          version: newVersion,
          rule_config: p.proposed_config,
          description: `Auto-generated from proposal ${p.id}`,
          rationale: p.rationale,
          is_active: false,  // Will be activated separately
          created_by: user.id,
        })
        .select()
        .single();

      if (createError) {
        console.error('Failed to create rule:', createError);
        return NextResponse.json({ error: 'Failed to create rule' }, { status: 500 });
      }

      // Update proposal status
      await supabase
        .from('guardian_rule_proposals')
        .update({
          status: 'accepted',
          reviewed_at: new Date().toISOString(),
          reviewed_by: user.id,
          review_notes: notes,
          created_rule_id: (newRule as { id: string }).id,
        })
        .eq('id', proposalId);

      // Log audit
      await supabase.from('guardian_audit_log').insert({
        action: 'proposal_accepted',
        entity_type: 'proposal',
        entity_id: proposalId,
        new_value: { rule_id: (newRule as { id: string }).id },
        performed_by: user.id,
        reason: notes || 'Admin approved proposal',
      });

      return NextResponse.json({
        success: true,
        message: `Proposal accepted. Rule v${newVersion} created.`,
        ruleId: (newRule as { id: string }).id,
        version: newVersion,
      });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // REJECT PROPOSAL - Human rejects Elle's suggestion
    // ═══════════════════════════════════════════════════════════════════════
    if (action === 'reject_proposal') {
      const { proposalId, notes } = body;

      await supabase
        .from('guardian_rule_proposals')
        .update({
          status: 'rejected',
          reviewed_at: new Date().toISOString(),
          reviewed_by: user.id,
          review_notes: notes,
        })
        .eq('id', proposalId);

      // Log audit
      await supabase.from('guardian_audit_log').insert({
        action: 'proposal_rejected',
        entity_type: 'proposal',
        entity_id: proposalId,
        performed_by: user.id,
        reason: notes || 'Admin rejected proposal',
      });

      return NextResponse.json({ success: true, message: 'Proposal rejected' });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ACTIVATE RULE - Enable a rule version
    // ═══════════════════════════════════════════════════════════════════════
    if (action === 'activate_rule') {
      const { ruleId } = body;

      // Call the stored function
      const { error } = await supabase.rpc('activate_guardian_rule', {
        p_rule_id: ruleId,
        p_user_id: user.id,
      });

      if (error) {
        console.error('Failed to activate rule:', error);
        return NextResponse.json({ error: 'Failed to activate rule' }, { status: 500 });
      }

      return NextResponse.json({ success: true, message: 'Rule activated' });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ROLLBACK RULE - Revert to previous version
    // ═══════════════════════════════════════════════════════════════════════
    if (action === 'rollback_rule') {
      const { ruleName, notes } = body;

      // Call the stored function
      const { data: previousId, error } = await supabase.rpc('rollback_guardian_rule', {
        p_rule_name: ruleName,
        p_user_id: user.id,
      });

      if (error) {
        console.error('Rollback failed:', error);
        return NextResponse.json({ error: error.message || 'Rollback failed' }, { status: 500 });
      }

      return NextResponse.json({
        success: true,
        message: `Rolled back to previous version`,
        previousRuleId: previousId,
      });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // REVIEW EVALUATION - Mark ground truth for disagreement
    // ═══════════════════════════════════════════════════════════════════════
    if (action === 'review_evaluation') {
      const { evaluationId, groundTruth, notes } = body;

      if (!['elle_correct', 'guardian_correct', 'both_wrong', 'both_correct'].includes(groundTruth)) {
        return NextResponse.json({ error: 'Invalid ground_truth value' }, { status: 400 });
      }

      await supabase
        .from('guardian_evaluations')
        .update({
          ground_truth: groundTruth,
          ground_truth_notes: notes,
          reviewed_at: new Date().toISOString(),
          reviewed_by: user.id,
        })
        .eq('id', evaluationId);

      return NextResponse.json({ success: true, message: 'Evaluation reviewed' });
    }

    return NextResponse.json({ error: 'Invalid action' }, { status: 400 });

  } catch (error) {
    console.error('Guardian API error:', error);
    return NextResponse.json({ error: 'Internal error' }, { status: 500 });
  }
}
