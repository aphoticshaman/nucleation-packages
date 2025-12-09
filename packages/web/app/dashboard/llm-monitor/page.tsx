'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';

interface PipelineHealth {
  overall: 'healthy' | 'degraded' | 'down';
  lastCheck: string;
  runpod: {
    status: string;
    latencyMs: number;
    errorRate: number;
    lastError?: string;
  };
  upstash: {
    status: string;
    latencyMs: number;
    hitRate: number;
  };
  costs: {
    today: number;
    thisWeek: number;
    thisMonth: number;
    projectedMonthly: number;
    alertThreshold: number;
  };
  alerts: Array<{
    id: string;
    severity: 'info' | 'warning' | 'critical';
    type: string;
    message: string;
    timestamp: string;
    acknowledged: boolean;
  }>;
}

interface MonitorResponse {
  success: boolean;
  timestamp: string;
  latencyMs: number;
  health: {
    overall: string;
    runpod: string;
    upstash: string;
  };
  metrics: {
    runpodLatencyMs: number;
    upstashLatencyMs: number;
    todayCostUsd: number;
    weekCostUsd: number;
    projectedMonthlyCostUsd: number;
  };
  alerts: {
    total: number;
    critical: number;
    warning: number;
    sent: string[];
    skipped: string[];
  };
  error?: string;
}

export default function LLMMonitorPage() {
  const [health, setHealth] = useState<MonitorResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchHealth = useCallback(async () => {
    try {
      setLoading(true);
      const res = await fetch('/api/cron/llm-pipeline-monitor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });

      const data = await res.json();
      setHealth(data);
      setLastRefresh(new Date());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHealth();

    // Auto-refresh every 30 seconds if enabled
    if (autoRefresh) {
      const interval = setInterval(fetchHealth, 30000);
      return () => clearInterval(interval);
    }
  }, [fetchHealth, autoRefresh]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'bg-green-500';
      case 'degraded':
        return 'bg-amber-500';
      case 'down':
        return 'bg-red-500';
      default:
        return 'bg-slate-500';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-900 border-red-600 text-red-200';
      case 'warning':
        return 'bg-amber-900 border-amber-600 text-amber-200';
      default:
        return 'bg-blue-900 border-blue-600 text-blue-200';
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold">LLM Pipeline Monitor</h1>
            <p className="text-slate-400 text-sm">
              Real-time observability for RunPod, Upstash, and inference pipeline
            </p>
          </div>
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm text-slate-400">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="rounded"
              />
              Auto-refresh (30s)
            </label>
            <button
              onClick={fetchHealth}
              disabled={loading}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm disabled:opacity-50"
            >
              {loading ? 'Checking...' : 'Refresh'}
            </button>
            <Link
              href="/dashboard"
              className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded text-sm"
            >
              Back to Dashboard
            </Link>
          </div>
        </div>

        {error && (
          <div className="bg-red-900 border border-red-600 rounded p-4 mb-6">
            <p className="text-red-200">Error: {error}</p>
          </div>
        )}

        {health && (
          <>
            {/* Overall Status */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
              <div className={`p-4 rounded-lg ${getStatusColor(health.health.overall)} bg-opacity-20 border ${health.health.overall === 'healthy' ? 'border-green-600' : health.health.overall === 'degraded' ? 'border-amber-600' : 'border-red-600'}`}>
                <div className="text-sm text-slate-300">Overall Status</div>
                <div className="text-2xl font-bold uppercase">{health.health.overall}</div>
                <div className="text-xs text-slate-400 mt-1">
                  Last check: {lastRefresh?.toLocaleTimeString()}
                </div>
              </div>

              <div className="p-4 rounded-lg bg-slate-800 border border-slate-700">
                <div className="text-sm text-slate-300">RunPod Latency</div>
                <div className="text-2xl font-bold">
                  {health.metrics.runpodLatencyMs}ms
                </div>
                <div className={`text-xs mt-1 ${health.health.runpod === 'healthy' ? 'text-green-400' : 'text-red-400'}`}>
                  Status: {health.health.runpod}
                </div>
              </div>

              <div className="p-4 rounded-lg bg-slate-800 border border-slate-700">
                <div className="text-sm text-slate-300">Today's Cost</div>
                <div className="text-2xl font-bold">
                  ${health.metrics.todayCostUsd.toFixed(4)}
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  Week: ${health.metrics.weekCostUsd.toFixed(2)}
                </div>
              </div>

              <div className="p-4 rounded-lg bg-slate-800 border border-slate-700">
                <div className="text-sm text-slate-300">Active Alerts</div>
                <div className="text-2xl font-bold">
                  {health.alerts.total}
                </div>
                <div className="text-xs mt-1">
                  <span className="text-red-400">{health.alerts.critical} critical</span>
                  {' • '}
                  <span className="text-amber-400">{health.alerts.warning} warning</span>
                </div>
              </div>
            </div>

            {/* Cost Projection */}
            <div className="bg-slate-800 rounded-lg p-4 mb-6">
              <h2 className="text-lg font-semibold mb-3">Cost Analysis</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <div className="text-slate-400 text-sm">Today</div>
                  <div className="text-xl font-mono">${health.metrics.todayCostUsd.toFixed(4)}</div>
                </div>
                <div>
                  <div className="text-slate-400 text-sm">This Week</div>
                  <div className="text-xl font-mono">${health.metrics.weekCostUsd.toFixed(2)}</div>
                </div>
                <div>
                  <div className="text-slate-400 text-sm">Projected Monthly</div>
                  <div className="text-xl font-mono">${health.metrics.projectedMonthlyCostUsd.toFixed(2)}</div>
                </div>
                <div>
                  <div className="text-slate-400 text-sm">Daily Alert Threshold</div>
                  <div className="text-xl font-mono text-amber-400">$10.00</div>
                </div>
              </div>

              {/* Cost bar */}
              <div className="mt-4">
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>$0</span>
                  <span>${(10).toFixed(2)} threshold</span>
                </div>
                <div className="h-4 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all ${
                      health.metrics.todayCostUsd > 10
                        ? 'bg-red-500'
                        : health.metrics.todayCostUsd > 5
                        ? 'bg-amber-500'
                        : 'bg-green-500'
                    }`}
                    style={{ width: `${Math.min(100, (health.metrics.todayCostUsd / 10) * 100)}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Services Status */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              <div className="bg-slate-800 rounded-lg p-4">
                <h2 className="text-lg font-semibold mb-3">RunPod / vLLM</h2>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Status</span>
                    <span className={health.health.runpod === 'healthy' ? 'text-green-400' : 'text-red-400'}>
                      {health.health.runpod}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Latency</span>
                    <span className="font-mono">{health.metrics.runpodLatencyMs}ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Endpoint</span>
                    <span className="text-xs text-slate-500 truncate max-w-[200px]">
                      {process.env.NEXT_PUBLIC_LFBM_ENDPOINT || 'Configured via env'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="bg-slate-800 rounded-lg p-4">
                <h2 className="text-lg font-semibold mb-3">Upstash Redis</h2>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Status</span>
                    <span className={health.health.upstash === 'healthy' ? 'text-green-400' : 'text-red-400'}>
                      {health.health.upstash}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Latency</span>
                    <span className="font-mono">{health.metrics.upstashLatencyMs}ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Purpose</span>
                    <span className="text-slate-300">Metrics & Cache</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Alerts */}
            {health.alerts.total > 0 && (
              <div className="bg-slate-800 rounded-lg p-4 mb-6">
                <h2 className="text-lg font-semibold mb-3">Active Alerts</h2>
                <div className="space-y-2">
                  {health.alerts.sent.map((alertType, idx) => (
                    <div
                      key={idx}
                      className="p-3 rounded border bg-red-900/50 border-red-600"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-red-200">{alertType}</span>
                        <span className="text-xs text-red-400">Alert sent</span>
                      </div>
                    </div>
                  ))}
                  {health.alerts.skipped.map((alertType, idx) => (
                    <div
                      key={idx}
                      className="p-3 rounded border bg-amber-900/50 border-amber-600"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-amber-200">{alertType}</span>
                        <span className="text-xs text-amber-400">Cooldown active</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Raw Response (Debug) */}
            <details className="bg-slate-800 rounded-lg p-4">
              <summary className="cursor-pointer text-slate-400 text-sm">
                Raw Monitor Response
              </summary>
              <pre className="mt-2 text-xs text-slate-300 overflow-auto">
                {JSON.stringify(health, null, 2)}
              </pre>
            </details>
          </>
        )}

        {!health && !error && loading && (
          <div className="flex items-center justify-center h-64">
            <div className="text-slate-400">Loading pipeline health...</div>
          </div>
        )}
      </div>
    </div>
  );
}
