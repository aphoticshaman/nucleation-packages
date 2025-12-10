/**
 * LLM Pipeline Monitor
 *
 * Provides observability for the entire LLM inference pipeline:
 * - RunPod vLLM endpoint health
 * - Latency tracking
 * - Error rate monitoring
 * - Cost estimation
 * - Anomaly detection
 *
 * Stores metrics in Upstash Redis for fast access and alerting.
 */

// =============================================================================
// TYPES
// =============================================================================

export interface LLMMetrics {
  timestamp: number;
  endpoint: 'runpod' | 'hf' | 'lfbm';

  // Health
  healthy: boolean;
  statusCode?: number;
  errorMessage?: string;

  // Performance
  latencyMs: number;
  tokensGenerated?: number;
  tokensPerSecond?: number;

  // Cost (estimated)
  estimatedCostUsd: number;
  gpuSecondsUsed?: number;
}

export interface PipelineHealth {
  overall: 'healthy' | 'degraded' | 'down';
  lastCheck: string;

  runpod: {
    status: 'healthy' | 'degraded' | 'down' | 'unknown';
    latencyMs: number;
    errorRate: number; // 0-1
    lastError?: string;
  };

  upstash: {
    status: 'healthy' | 'degraded' | 'down' | 'unknown';
    latencyMs: number;
    hitRate: number; // 0-1
  };

  costs: {
    today: number;
    thisWeek: number;
    thisMonth: number;
    projectedMonthly: number;
    alertThreshold: number;
  };

  alerts: PipelineAlert[];
}

export interface PipelineAlert {
  id: string;
  severity: 'info' | 'warning' | 'critical';
  type: 'cost' | 'latency' | 'error' | 'outage' | 'security' | 'anomaly';
  message: string;
  timestamp: string;
  acknowledged: boolean;
  data?: Record<string, unknown>;
}

// =============================================================================
// CONSTANTS
// =============================================================================

// RunPod pricing (approximate for Qwen 3B on A40)
const RUNPOD_COST_PER_GPU_SECOND = 0.00044; // ~$1.58/hr for A40
const RUNPOD_COLD_START_SECONDS = 30;
const RUNPOD_AVG_INFERENCE_SECONDS = 2;

// Alert thresholds
export const ALERT_THRESHOLDS = {
  dailyCostUsd: 10,
  latencyMs: 5000, // 5 second response time
  errorRatePercent: 10, // 10% error rate
  consecutiveErrors: 3,
};

// Metric retention
const METRICS_TTL_SECONDS = 7 * 24 * 60 * 60; // 7 days

// =============================================================================
// UPSTASH INTEGRATION
// =============================================================================

interface UpstashConfig {
  url: string;
  token: string;
}

function getUpstashConfig(): UpstashConfig | null {
  const url = process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN;

  if (!url || !token) return null;
  return { url, token };
}

async function upstashCommand(
  config: UpstashConfig,
  command: string[]
): Promise<unknown> {
  const response = await fetch(`${config.url}`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${config.token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(command),
  });

  if (!response.ok) {
    throw new Error(`Upstash error: ${response.status}`);
  }

  const data = await response.json();
  return data.result;
}

// =============================================================================
// METRIC STORAGE
// =============================================================================

export async function recordMetric(metric: LLMMetrics): Promise<void> {
  const config = getUpstashConfig();
  if (!config) {
    console.log('[LLM-MONITOR] No Upstash configured, skipping metric storage');
    return;
  }

  const key = `llm:metrics:${metric.endpoint}:${metric.timestamp}`;
  const listKey = `llm:metrics:${metric.endpoint}:list`;

  try {
    // Store individual metric
    await upstashCommand(config, [
      'SET', key, JSON.stringify(metric), 'EX', String(METRICS_TTL_SECONDS)
    ]);

    // Add to time-series list (for querying)
    await upstashCommand(config, [
      'LPUSH', listKey, String(metric.timestamp)
    ]);

    // Trim list to last 1000 entries
    await upstashCommand(config, ['LTRIM', listKey, '0', '999']);

    // Update rolling stats
    await updateRollingStats(config, metric);

  } catch (error) {
    console.error('[LLM-MONITOR] Failed to record metric:', error);
  }
}

async function updateRollingStats(
  config: UpstashConfig,
  metric: LLMMetrics
): Promise<void> {
  const today = new Date().toISOString().split('T')[0];
  const costKey = `llm:costs:${today}`;
  const errorKey = `llm:errors:${metric.endpoint}:count`;
  const requestKey = `llm:requests:${metric.endpoint}:count`;

  // Increment cost
  await upstashCommand(config, [
    'INCRBYFLOAT', costKey, String(metric.estimatedCostUsd)
  ]);
  await upstashCommand(config, ['EXPIRE', costKey, String(METRICS_TTL_SECONDS)]);

  // Track error rate
  await upstashCommand(config, ['INCR', requestKey]);
  if (!metric.healthy) {
    await upstashCommand(config, ['INCR', errorKey]);
  }

  // Set TTL on counters
  await upstashCommand(config, ['EXPIRE', errorKey, '3600']); // 1 hour window
  await upstashCommand(config, ['EXPIRE', requestKey, '3600']);
}

// =============================================================================
// HEALTH CHECKS
// =============================================================================

export async function checkRunPodHealth(): Promise<{
  healthy: boolean;
  latencyMs: number;
  error?: string;
}> {
  const endpoint = process.env.LFBM_ENDPOINT;
  const apiKey = process.env.LFBM_API_KEY;

  if (!endpoint) {
    return { healthy: false, latencyMs: 0, error: 'LFBM_ENDPOINT not configured' };
  }

  // COST OPTIMIZATION: Skip expensive inference calls in non-prod
  // This prevents health checks from triggering RunPod cold starts
  if (process.env.LF_PROD_ENABLE !== 'true') {
    return {
      healthy: true,
      latencyMs: 0,
      error: 'Skipped - LF_PROD_ENABLE not true (cost savings)'
    };
  }

  const startTime = Date.now();

  try {
    // RunPod health check - use /health or a minimal inference
    const isRunPod = endpoint.includes('api.runpod.ai');

    if (isRunPod) {
      // COST OPTIMIZATION: Use status endpoint instead of inference
      // RunPod serverless has a status endpoint that doesn't trigger cold start
      const statusEndpoint = endpoint.replace(/\/(run|runsync)$/, '/health');

      try {
        const response = await fetch(statusEndpoint, {
          method: 'GET',
          headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
        });
        const latencyMs = Date.now() - startTime;

        // If status endpoint works, endpoint is healthy
        if (response.ok) {
          return { healthy: true, latencyMs };
        }
      } catch {
        // Status endpoint might not exist, fall through to inference check
      }

      // Only do inference check if status endpoint failed
      const syncEndpoint = endpoint.replace(/\/run$/, '/runsync');

      const response = await fetch(syncEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
        },
        body: JSON.stringify({
          input: {
            messages: [{ role: 'user', content: 'ping' }],
            max_tokens: 1,
          },
        }),
      });

      const latencyMs = Date.now() - startTime;

      if (!response.ok) {
        const error = await response.text();
        return { healthy: false, latencyMs, error: `HTTP ${response.status}: ${error}` };
      }

      return { healthy: true, latencyMs };
    } else {
      // Direct vLLM - check health endpoint
      const response = await fetch(`${endpoint}/health`);
      const latencyMs = Date.now() - startTime;

      return { healthy: response.ok, latencyMs };
    }
  } catch (error) {
    return {
      healthy: false,
      latencyMs: Date.now() - startTime,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

export async function checkUpstashHealth(): Promise<{
  healthy: boolean;
  latencyMs: number;
  error?: string;
}> {
  const config = getUpstashConfig();
  if (!config) {
    return { healthy: false, latencyMs: 0, error: 'Upstash not configured' };
  }

  const startTime = Date.now();

  try {
    await upstashCommand(config, ['PING']);
    return { healthy: true, latencyMs: Date.now() - startTime };
  } catch (error) {
    return {
      healthy: false,
      latencyMs: Date.now() - startTime,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

// =============================================================================
// COST TRACKING
// =============================================================================

export function estimateInferenceCost(
  latencyMs: number,
  coldStart: boolean = false
): number {
  const inferenceSeconds = latencyMs / 1000;
  const totalSeconds = coldStart
    ? inferenceSeconds + RUNPOD_COLD_START_SECONDS
    : inferenceSeconds;

  return totalSeconds * RUNPOD_COST_PER_GPU_SECOND;
}

export async function getTodayCost(): Promise<number> {
  const config = getUpstashConfig();
  if (!config) return 0;

  const today = new Date().toISOString().split('T')[0];
  const costKey = `llm:costs:${today}`;

  try {
    const result = await upstashCommand(config, ['GET', costKey]);
    return result ? parseFloat(result as string) : 0;
  } catch {
    return 0;
  }
}

export async function getWeekCost(): Promise<number> {
  const config = getUpstashConfig();
  if (!config) return 0;

  let total = 0;
  const now = new Date();

  for (let i = 0; i < 7; i++) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    const dateStr = date.toISOString().split('T')[0];
    const costKey = `llm:costs:${dateStr}`;

    try {
      const result = await upstashCommand(config, ['GET', costKey]);
      if (result) {
        total += parseFloat(result as string);
      }
    } catch {
      // Ignore individual failures
    }
  }

  return total;
}

// =============================================================================
// ALERT DETECTION
// =============================================================================

export async function detectAlerts(): Promise<PipelineAlert[]> {
  const alerts: PipelineAlert[] = [];
  const config = getUpstashConfig();

  // Cost alert
  const todayCost = await getTodayCost();
  if (todayCost > ALERT_THRESHOLDS.dailyCostUsd) {
    alerts.push({
      id: `cost-${Date.now()}`,
      severity: 'critical',
      type: 'cost',
      message: `Daily RunPod cost ($${todayCost.toFixed(2)}) exceeds threshold ($${ALERT_THRESHOLDS.dailyCostUsd})`,
      timestamp: new Date().toISOString(),
      acknowledged: false,
      data: { cost: todayCost, threshold: ALERT_THRESHOLDS.dailyCostUsd },
    });
  }

  // Error rate alert
  if (config) {
    try {
      const errors = await upstashCommand(config, ['GET', 'llm:errors:runpod:count']) as string | null;
      const requests = await upstashCommand(config, ['GET', 'llm:requests:runpod:count']) as string | null;

      if (errors && requests) {
        const errorCount = parseInt(errors, 10);
        const requestCount = parseInt(requests, 10);
        const errorRate = requestCount > 0 ? (errorCount / requestCount) * 100 : 0;

        if (errorRate > ALERT_THRESHOLDS.errorRatePercent) {
          alerts.push({
            id: `error-rate-${Date.now()}`,
            severity: 'warning',
            type: 'error',
            message: `High error rate: ${errorRate.toFixed(1)}% in last hour`,
            timestamp: new Date().toISOString(),
            acknowledged: false,
            data: { errorRate, errorCount, requestCount },
          });
        }
      }
    } catch {
      // Ignore
    }
  }

  // RunPod health check
  const runpodHealth = await checkRunPodHealth();
  if (!runpodHealth.healthy) {
    alerts.push({
      id: `outage-${Date.now()}`,
      severity: 'critical',
      type: 'outage',
      message: `RunPod endpoint unhealthy: ${runpodHealth.error}`,
      timestamp: new Date().toISOString(),
      acknowledged: false,
      data: { error: runpodHealth.error, latencyMs: runpodHealth.latencyMs },
    });
  } else if (runpodHealth.latencyMs > ALERT_THRESHOLDS.latencyMs) {
    alerts.push({
      id: `latency-${Date.now()}`,
      severity: 'warning',
      type: 'latency',
      message: `High latency: ${runpodHealth.latencyMs}ms (threshold: ${ALERT_THRESHOLDS.latencyMs}ms)`,
      timestamp: new Date().toISOString(),
      acknowledged: false,
      data: { latencyMs: runpodHealth.latencyMs, threshold: ALERT_THRESHOLDS.latencyMs },
    });
  }

  return alerts;
}

// =============================================================================
// FULL PIPELINE STATUS
// =============================================================================

export async function getPipelineHealth(): Promise<PipelineHealth> {
  const [runpodHealth, upstashHealth, todayCost, weekCost, alerts] = await Promise.all([
    checkRunPodHealth(),
    checkUpstashHealth(),
    getTodayCost(),
    getWeekCost(),
    detectAlerts(),
  ]);

  // Determine overall status
  let overall: 'healthy' | 'degraded' | 'down' = 'healthy';
  if (!runpodHealth.healthy) {
    overall = 'down';
  } else if (alerts.some(a => a.severity === 'critical')) {
    overall = 'degraded';
  } else if (alerts.some(a => a.severity === 'warning')) {
    overall = 'degraded';
  }

  // Project monthly cost from week average
  const avgDailyCost = weekCost / 7;
  const projectedMonthly = avgDailyCost * 30;

  return {
    overall,
    lastCheck: new Date().toISOString(),

    runpod: {
      status: runpodHealth.healthy ? 'healthy' : 'down',
      latencyMs: runpodHealth.latencyMs,
      errorRate: 0, // Would need to calculate from stored metrics
      lastError: runpodHealth.error,
    },

    upstash: {
      status: upstashHealth.healthy ? 'healthy' : 'down',
      latencyMs: upstashHealth.latencyMs,
      hitRate: 0, // Would need to track cache hits/misses
    },

    costs: {
      today: todayCost,
      thisWeek: weekCost,
      thisMonth: weekCost * 4, // Rough estimate
      projectedMonthly,
      alertThreshold: ALERT_THRESHOLDS.dailyCostUsd,
    },

    alerts,
  };
}

// =============================================================================
// EMAIL ALERTS
// =============================================================================

export async function sendPipelineAlert(alert: PipelineAlert): Promise<boolean> {
  if (!process.env.RESEND_API_KEY) {
    console.warn('[LLM-MONITOR] No RESEND_API_KEY, cannot send alert email');
    return false;
  }

  const severityColors = {
    info: '#3b82f6',
    warning: '#f59e0b',
    critical: '#dc2626',
  };

  try {
    const response = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${process.env.RESEND_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: 'LatticeForge Alerts <alerts@latticeforge.ai>',
        to: ['admin@latticeforge.ai'],
        subject: `[${alert.severity.toUpperCase()}] LLM Pipeline: ${alert.type}`,
        html: `
          <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: ${severityColors[alert.severity]}; color: white; padding: 16px; border-radius: 8px 8px 0 0;">
              <h2 style="margin: 0;">${alert.severity.toUpperCase()}: ${alert.type}</h2>
            </div>
            <div style="border: 1px solid #e5e7eb; border-top: none; padding: 16px; border-radius: 0 0 8px 8px;">
              <p style="font-size: 16px; margin-bottom: 16px;">${alert.message}</p>
              <table style="width: 100%; border-collapse: collapse;">
                <tr>
                  <td style="padding: 8px; border: 1px solid #e5e7eb;"><strong>Time:</strong></td>
                  <td style="padding: 8px; border: 1px solid #e5e7eb;">${alert.timestamp}</td>
                </tr>
                <tr>
                  <td style="padding: 8px; border: 1px solid #e5e7eb;"><strong>Alert ID:</strong></td>
                  <td style="padding: 8px; border: 1px solid #e5e7eb;">${alert.id}</td>
                </tr>
                ${alert.data ? `
                <tr>
                  <td style="padding: 8px; border: 1px solid #e5e7eb;"><strong>Details:</strong></td>
                  <td style="padding: 8px; border: 1px solid #e5e7eb;"><pre style="margin: 0; font-size: 12px;">${JSON.stringify(alert.data, null, 2)}</pre></td>
                </tr>
                ` : ''}
              </table>
              <p style="margin-top: 16px; color: #6b7280; font-size: 12px;">
                This alert was generated by the LatticeForge LLM Pipeline Monitor.
              </p>
            </div>
          </div>
        `,
      }),
    });

    return response.ok;
  } catch (error) {
    console.error('[LLM-MONITOR] Failed to send alert email:', error);
    return false;
  }
}
