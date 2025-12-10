import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import Anthropic from '@anthropic-ai/sdk';

export const runtime = 'edge';
export const maxDuration = 120; // 2 minutes for AI analysis

// Verify this is a legitimate cron call
function verifyCronAuth(request: Request): boolean {
  const authHeader = request.headers.get('authorization');
  if (authHeader === `Bearer ${process.env.CRON_SECRET}`) {
    return true;
  }
  // Vercel cron jobs also set this header
  const vercelCron = request.headers.get('x-vercel-cron');
  if (vercelCron === '1') {
    return true;
  }
  return false;
}

interface Disagreement {
  id: string;
  domain: string;
  input_summary: string;
  elle_decision: string;
  elle_reasoning: string;
  guardian_decision: string;
  guardian_reasoning: string;
}

interface TrainingLog {
  timestamp: string;
  period: '24h';
  metrics: {
    totalEvaluations: number;
    agreementRate: number;
    elleCorrectRate: number | null;
    guardianCorrectRate: number | null;
    hallucinations: number;
    byDomain: Record<string, { evals: number; agreement: number }>;
  };
  elleSummary: string;
  guardianSummary: string;
  trainingRecommendations: string[];
  proposedRuleChanges: Array<{
    ruleName: string;
    domain: string;
    change: string;
    rationale: string;
    confidence: number;
  }>;
}

/**
 * Guardian Self-Audit Cron Job
 *
 * Runs daily to:
 * 1. Compute metrics from the last 24 hours
 * 2. Have Elle analyze disagreements and propose rule changes
 * 3. Log training recommendations for the next fine-tuning cycle
 *
 * Human-in-the-loop: All proposals require admin approval in the dashboard
 */
export async function GET(request: Request) {
  const startTime = Date.now();

  // Security check
  if (!verifyCronAuth(request)) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  // Initialize clients
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const anthropicKey = process.env.ANTHROPIC_API_KEY;

  if (!supabaseUrl || !supabaseServiceKey) {
    console.error('[GUARDIAN-AUDIT] Missing Supabase credentials');
    return NextResponse.json({ error: 'Server misconfigured' }, { status: 500 });
  }

  const supabase = createClient(supabaseUrl, supabaseServiceKey);
  const anthropic = anthropicKey ? new Anthropic({ apiKey: anthropicKey }) : null;

  try {
    const now = new Date();
    const yesterday = new Date(now.getTime() - 24 * 60 * 60 * 1000);

    // ═══════════════════════════════════════════════════════════════════════════
    // 1. Fetch evaluation data from last 24 hours
    // ═══════════════════════════════════════════════════════════════════════════

    const { data: evaluations, error: evalError } = await supabase
      .from('guardian_evaluations')
      .select('*')
      .gte('created_at', yesterday.toISOString());

    if (evalError) {
      console.error('[GUARDIAN-AUDIT] Error fetching evaluations:', evalError);
      throw evalError;
    }

    const evals = evaluations || [];
    const totalEvaluations = evals.length;

    // Calculate agreement rate
    const agreements = evals.filter((e) => e.elle_decision === e.guardian_decision).length;
    const agreementRate = totalEvaluations > 0 ? (agreements / totalEvaluations) * 100 : 100;

    // Calculate correctness rates (only for reviewed evaluations)
    const reviewed = evals.filter((e) => e.ground_truth !== null);
    const elleCorrect = reviewed.filter(
      (e) => e.ground_truth === 'elle_correct' || e.ground_truth === 'both_correct'
    ).length;
    const guardianCorrect = reviewed.filter(
      (e) => e.ground_truth === 'guardian_correct' || e.ground_truth === 'both_correct'
    ).length;

    const elleCorrectRate = reviewed.length > 0 ? (elleCorrect / reviewed.length) * 100 : null;
    const guardianCorrectRate = reviewed.length > 0 ? (guardianCorrect / reviewed.length) * 100 : null;

    // Count hallucinations (flagged by Guardian)
    const hallucinations = evals.filter((e) => e.guardian_flags?.includes('hallucination')).length;

    // Group by domain
    const byDomain: Record<string, { evals: number; agreement: number }> = {};
    for (const e of evals) {
      const domain = e.domain || 'global';
      if (!byDomain[domain]) {
        byDomain[domain] = { evals: 0, agreement: 0 };
      }
      byDomain[domain].evals++;
      if (e.elle_decision === e.guardian_decision) {
        byDomain[domain].agreement++;
      }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 2. Fetch unresolved disagreements for Elle to analyze
    // ═══════════════════════════════════════════════════════════════════════════

    const { data: disagreements } = await supabase
      .from('guardian_evaluations')
      .select('id, domain, input_summary, elle_decision, elle_reasoning, guardian_decision, guardian_reasoning')
      .is('ground_truth', null)
      .neq('elle_decision', supabase.from('guardian_evaluations').select('guardian_decision'))
      .gte('created_at', yesterday.toISOString())
      .limit(20);

    const unresolvedDisagreements: Disagreement[] = (disagreements || []).filter(
      (d) => d.elle_decision !== d.guardian_decision
    );

    // ═══════════════════════════════════════════════════════════════════════════
    // 3. Have Elle analyze and propose rule changes (if AI available)
    // ═══════════════════════════════════════════════════════════════════════════

    let elleSummary = 'AI analysis not available - Anthropic API key not configured';
    let guardianSummary = 'Symbolic analysis pending human review';
    let trainingRecommendations: string[] = [];
    const proposedRuleChanges: Array<{
      ruleName: string;
      domain: string;
      change: string;
      rationale: string;
      confidence: number;
    }> = [];

    if (anthropic && unresolvedDisagreements.length > 0) {
      try {
        const analysisPrompt = `You are Elle, an AI specialized in geopolitical intelligence analysis. You work alongside Guardian, a symbolic reasoning system. Analyze the following disagreements from the last 24 hours and provide:

1. A summary of patterns you see in the disagreements
2. Specific rule changes that could improve Guardian's accuracy
3. Training recommendations for your own improvement

DISAGREEMENTS:
${unresolvedDisagreements
  .slice(0, 10) // Limit context
  .map(
    (d, i) => `
${i + 1}. Domain: ${d.domain}
   Input: ${d.input_summary}
   Elle decided: ${d.elle_decision} - ${d.elle_reasoning}
   Guardian decided: ${d.guardian_decision} - ${d.guardian_reasoning}
`
  )
  .join('\n')}

METRICS FOR CONTEXT:
- Total evaluations (24h): ${totalEvaluations}
- Agreement rate: ${agreementRate.toFixed(1)}%
- Hallucinations flagged: ${hallucinations}

Respond in JSON format:
{
  "elleSummary": "Your analysis of patterns and issues",
  "trainingRecommendations": ["recommendation 1", "recommendation 2"],
  "proposedRuleChanges": [
    {
      "ruleName": "descriptive_rule_name",
      "domain": "political|economic|security|etc",
      "change": "Description of the proposed rule change",
      "rationale": "Why this change would help",
      "confidence": 0.0-1.0
    }
  ]
}`;

        const response = await anthropic.messages.create({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 2000,
          messages: [{ role: 'user', content: analysisPrompt }],
        });

        const responseText =
          response.content[0].type === 'text' ? response.content[0].text : '';

        // Parse JSON response
        try {
          const jsonMatch = responseText.match(/\{[\s\S]*\}/);
          if (jsonMatch) {
            const parsed = JSON.parse(jsonMatch[0]);
            elleSummary = parsed.elleSummary || elleSummary;
            trainingRecommendations = parsed.trainingRecommendations || [];

            // Add proposed rule changes
            if (parsed.proposedRuleChanges && Array.isArray(parsed.proposedRuleChanges)) {
              for (const change of parsed.proposedRuleChanges) {
                proposedRuleChanges.push({
                  ruleName: change.ruleName || 'unnamed_rule',
                  domain: change.domain || 'global',
                  change: change.change || '',
                  rationale: change.rationale || '',
                  confidence: typeof change.confidence === 'number' ? change.confidence : 0.5,
                });
              }
            }
          }
        } catch (parseErr) {
          console.error('[GUARDIAN-AUDIT] Failed to parse Elle response:', parseErr);
          elleSummary = responseText.slice(0, 500); // Use raw response as summary
        }
      } catch (aiErr) {
        console.error('[GUARDIAN-AUDIT] AI analysis failed:', aiErr);
        elleSummary = `AI analysis failed: ${aiErr}`;
      }
    }

    // Guardian's symbolic summary (rule-based analysis)
    guardianSummary = generateGuardianSummary(byDomain, agreementRate, hallucinations, totalEvaluations);

    // ═══════════════════════════════════════════════════════════════════════════
    // 4. Store rule proposals in the database (for human review)
    // ═══════════════════════════════════════════════════════════════════════════

    for (const proposal of proposedRuleChanges) {
      if (proposal.confidence >= 0.6) {
        // Only propose if confidence >= 60%
        await supabase.from('guardian_rule_proposals').insert({
          proposed_rule_name: proposal.ruleName,
          proposed_domain: proposal.domain,
          proposed_config: { change: proposal.change },
          rationale: proposal.rationale,
          confidence_score: proposal.confidence,
          elle_summary: elleSummary,
          guardian_summary: guardianSummary,
          supporting_evidence: unresolvedDisagreements.slice(0, 5).map((d) => d.input_summary),
          status: 'pending',
        });
      }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 5. Store training log for export
    // ═══════════════════════════════════════════════════════════════════════════

    const trainingLog: TrainingLog = {
      timestamp: now.toISOString(),
      period: '24h',
      metrics: {
        totalEvaluations,
        agreementRate,
        elleCorrectRate,
        guardianCorrectRate,
        hallucinations,
        byDomain,
      },
      elleSummary,
      guardianSummary,
      trainingRecommendations,
      proposedRuleChanges,
    };

    // Store in audit log
    await supabase.from('guardian_audit_log').insert({
      action: 'daily_audit',
      entity_type: 'system',
      entity_id: now.toISOString().split('T')[0],
      new_value: trainingLog,
      reason: `Daily audit: ${totalEvaluations} evals, ${agreementRate.toFixed(1)}% agreement, ${proposedRuleChanges.length} proposals`,
    });

    // ═══════════════════════════════════════════════════════════════════════════
    // 6. Update metrics table
    // ═══════════════════════════════════════════════════════════════════════════

    await supabase.from('guardian_metrics').insert({
      period_start: yesterday.toISOString(),
      period_end: now.toISOString(),
      granularity: 'daily',
      domain: null, // global
      total_evaluations: totalEvaluations,
      agreements,
      accuracy_pct: agreementRate,
      hallucination_rate_pct: totalEvaluations > 0 ? (hallucinations / totalEvaluations) * 100 : 0,
    });

    // Insert domain-specific metrics
    for (const [domain, stats] of Object.entries(byDomain)) {
      await supabase.from('guardian_metrics').insert({
        period_start: yesterday.toISOString(),
        period_end: now.toISOString(),
        granularity: 'daily',
        domain,
        total_evaluations: stats.evals,
        agreements: stats.agreement,
        accuracy_pct: stats.evals > 0 ? (stats.agreement / stats.evals) * 100 : 100,
      });
    }

    const duration = Date.now() - startTime;
    console.log(`[GUARDIAN-AUDIT] Completed in ${duration}ms: ${totalEvaluations} evals, ${proposedRuleChanges.length} proposals`);

    return NextResponse.json({
      success: true,
      metrics: {
        totalEvaluations,
        agreementRate: agreementRate.toFixed(1),
        hallucinations,
        unresolvedDisagreements: unresolvedDisagreements.length,
      },
      proposals: proposedRuleChanges.length,
      trainingRecommendations: trainingRecommendations.length,
      durationMs: duration,
    });
  } catch (error) {
    console.error('[GUARDIAN-AUDIT] Fatal error:', error);
    return NextResponse.json(
      { error: 'Failed to run guardian audit', details: String(error) },
      { status: 500 }
    );
  }
}

/**
 * Generate Guardian's symbolic analysis summary
 */
function generateGuardianSummary(
  byDomain: Record<string, { evals: number; agreement: number }>,
  agreementRate: number,
  hallucinations: number,
  totalEvaluations: number
): string {
  const parts: string[] = [];

  // Overall assessment
  if (agreementRate >= 95) {
    parts.push('System operating within excellent parameters.');
  } else if (agreementRate >= 85) {
    parts.push('System showing good alignment with minor discrepancies.');
  } else if (agreementRate >= 70) {
    parts.push('Moderate disagreement rate detected - review recommended.');
  } else {
    parts.push('HIGH ALERT: Significant disagreement rate requires immediate attention.');
  }

  // Hallucination check
  if (hallucinations > 0) {
    const rate = ((hallucinations / totalEvaluations) * 100).toFixed(1);
    parts.push(`Flagged ${hallucinations} potential hallucinations (${rate}% of evaluations).`);
  }

  // Domain-specific issues
  const weakDomains = Object.entries(byDomain)
    .filter(([, stats]) => stats.evals >= 3 && stats.agreement / stats.evals < 0.8)
    .map(([domain]) => domain);

  if (weakDomains.length > 0) {
    parts.push(`Domains requiring attention: ${weakDomains.join(', ')}.`);
  }

  // Rule recommendation
  if (agreementRate < 85 || hallucinations > 2) {
    parts.push('Recommend reviewing recent rule proposals and considering threshold adjustments.');
  }

  return parts.join(' ');
}
