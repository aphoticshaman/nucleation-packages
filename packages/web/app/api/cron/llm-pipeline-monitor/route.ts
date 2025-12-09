import { NextResponse } from 'next/server';
import {
  getPipelineHealth,
  sendPipelineAlert,
  ALERT_THRESHOLDS,
  type PipelineAlert,
} from '@/lib/monitoring/LLMPipelineMonitor';

// =============================================================================
// CRON: LLM Pipeline Monitor
// =============================================================================
// Runs every 15 minutes to:
// 1. Check RunPod/vLLM endpoint health
// 2. Track costs and usage
// 3. Detect anomalies
// 4. Send alerts for critical issues
//
// Schedule: */15 * * * * (every 15 minutes)

// Track sent alerts to avoid spam (in-memory, resets on cold start)
const sentAlertIds = new Set<string>();

// Minimum time between same-type alerts (1 hour)
const ALERT_COOLDOWN_MS = 60 * 60 * 1000;
const alertCooldowns = new Map<string, number>();

function shouldSendAlert(alert: PipelineAlert): boolean {
  // Don't spam the same alert
  if (sentAlertIds.has(alert.id)) {
    return false;
  }

  // Check cooldown for alert type
  const lastSent = alertCooldowns.get(alert.type);
  if (lastSent && Date.now() - lastSent < ALERT_COOLDOWN_MS) {
    return false;
  }

  return true;
}

function markAlertSent(alert: PipelineAlert): void {
  sentAlertIds.add(alert.id);
  alertCooldowns.set(alert.type, Date.now());

  // Prevent memory leak - clear old IDs after 1000
  if (sentAlertIds.size > 1000) {
    sentAlertIds.clear();
  }
}

export async function GET(request: Request) {
  const startTime = Date.now();

  // Verify cron authorization
  const isVercelCron = request.headers.get('x-vercel-cron') === '1';
  const cronSecret = process.env.CRON_SECRET;
  const authHeader = request.headers.get('authorization');

  if (!isVercelCron && cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    // Get full pipeline health status
    const health = await getPipelineHealth();

    console.log(`[LLM-MONITOR] Pipeline status: ${health.overall}`);
    console.log(`[LLM-MONITOR] RunPod: ${health.runpod.status} (${health.runpod.latencyMs}ms)`);
    console.log(`[LLM-MONITOR] Today's cost: $${health.costs.today.toFixed(4)}`);
    console.log(`[LLM-MONITOR] Alerts detected: ${health.alerts.length}`);

    // Process alerts
    const alertsSent: string[] = [];
    const alertsSkipped: string[] = [];

    for (const alert of health.alerts) {
      if (shouldSendAlert(alert)) {
        const sent = await sendPipelineAlert(alert);
        if (sent) {
          markAlertSent(alert);
          alertsSent.push(alert.type);
          console.log(`[LLM-MONITOR] Sent ${alert.severity} alert: ${alert.message}`);
        }
      } else {
        alertsSkipped.push(alert.type);
      }
    }

    // Build response
    const response = {
      success: true,
      timestamp: new Date().toISOString(),
      latencyMs: Date.now() - startTime,

      health: {
        overall: health.overall,
        runpod: health.runpod.status,
        upstash: health.upstash.status,
      },

      metrics: {
        runpodLatencyMs: health.runpod.latencyMs,
        upstashLatencyMs: health.upstash.latencyMs,
        todayCostUsd: health.costs.today,
        weekCostUsd: health.costs.thisWeek,
        projectedMonthlyCostUsd: health.costs.projectedMonthly,
      },

      thresholds: ALERT_THRESHOLDS,

      alerts: {
        total: health.alerts.length,
        critical: health.alerts.filter(a => a.severity === 'critical').length,
        warning: health.alerts.filter(a => a.severity === 'warning').length,
        sent: alertsSent,
        skipped: alertsSkipped,
      },
    };

    // Return degraded status code if unhealthy
    const statusCode = health.overall === 'down' ? 503 : 200;

    return NextResponse.json(response, { status: statusCode });

  } catch (error) {
    console.error('[LLM-MONITOR] Monitor error:', error);

    // Try to send error alert
    try {
      await sendPipelineAlert({
        id: `monitor-error-${Date.now()}`,
        severity: 'critical',
        type: 'error',
        message: `LLM Pipeline Monitor itself failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date().toISOString(),
        acknowledged: false,
      });
    } catch {
      // Can't do much if even alerting fails
    }

    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Monitor failed',
      timestamp: new Date().toISOString(),
      latencyMs: Date.now() - startTime,
    }, { status: 500 });
  }
}

// POST endpoint for manual health check
export async function POST(request: Request) {
  return GET(request);
}
