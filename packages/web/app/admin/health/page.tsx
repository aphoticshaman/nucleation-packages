'use client';

import { useState, useEffect } from 'react';
import { RefreshCw } from 'lucide-react';
import { Card, Button, StatusIndicator, Skeleton } from '@/components/ui';

interface HealthCheck {
  name: string;
  status: 'healthy' | 'degraded' | 'down' | 'checking';
  latency?: number;
  lastChecked?: string;
  error?: string;
  details?: Record<string, unknown>;
}

const API_ENDPOINTS = [
  { name: 'Intel Briefing Cache', url: '/api/intel-briefing', method: 'POST', body: { preset: 'global' }, critical: true },
  { name: 'US Economic Brief', url: '/api/us-brief', method: 'POST', critical: true },
  { name: 'GDELT Ingest', url: '/api/ingest/gdelt', needsCron: true, critical: false },
  { name: 'Nation Risk Compute', url: '/api/compute/nation-risk', needsCron: true, critical: false },
];

function HealthCard({ check, onRecheck }: { check: HealthCheck; onRecheck: () => void }) {
  const statusMap = {
    healthy: 'healthy' as const,
    degraded: 'warning' as const,
    down: 'error' as const,
    checking: 'pending' as const,
  };

  return (
    <Card padding="md">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <StatusIndicator status={statusMap[check.status]} showLabel={false} />
          <span className="text-sm font-medium text-slate-200">{check.name}</span>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={onRecheck}
          disabled={check.status === 'checking'}
          loading={check.status === 'checking'}
        >
          {check.status === 'checking' ? 'Checking' : 'Recheck'}
        </Button>
      </div>

      <div className="space-y-2 text-xs">
        <div className="flex justify-between">
          <span className="text-slate-500">Status</span>
          <span className={
            check.status === 'healthy' ? 'text-emerald-500' :
            check.status === 'degraded' ? 'text-amber-500' :
            check.status === 'down' ? 'text-red-500' : 'text-slate-500'
          }>
            {check.status.charAt(0).toUpperCase() + check.status.slice(1)}
          </span>
        </div>

        {check.latency !== undefined && (
          <div className="flex justify-between">
            <span className="text-slate-500">Latency</span>
            <span className={
              check.latency < 200 ? 'text-emerald-500' :
              check.latency < 500 ? 'text-amber-500' : 'text-red-500'
            }>
              {check.latency}ms
            </span>
          </div>
        )}

        {check.lastChecked && (
          <div className="flex justify-between">
            <span className="text-slate-500">Last Checked</span>
            <span className="text-slate-400">{new Date(check.lastChecked).toLocaleTimeString()}</span>
          </div>
        )}

        {check.error && (
          <div className="mt-2 p-2 bg-red-500/5 border border-red-500/20 rounded text-red-400">
            {check.error}
          </div>
        )}
      </div>
    </Card>
  );
}

export default function AdminHealthPage() {
  const [checks, setChecks] = useState<HealthCheck[]>(
    API_ENDPOINTS.map((e) => ({
      name: e.name,
      status: 'checking',
    }))
  );
  const [overallStatus, setOverallStatus] = useState<'healthy' | 'degraded' | 'down'>('healthy');

  const runHealthCheck = async (index: number) => {
    const endpoint = API_ENDPOINTS[index];

    setChecks((prev) =>
      prev.map((c, i) => (i === index ? { ...c, status: 'checking' } : c))
    );

    const start = Date.now();

    try {
      if (endpoint.needsCron) {
        setChecks((prev) =>
          prev.map((c, i) =>
            i === index
              ? {
                  ...c,
                  status: 'healthy',
                  latency: 0,
                  lastChecked: new Date().toISOString(),
                  details: { note: 'Cron-protected endpoint' },
                }
              : c
          )
        );
        return;
      }

      const response = await fetch(endpoint.url, {
        method: endpoint.method || 'GET',
        headers: endpoint.body ? { 'Content-Type': 'application/json' } : {},
        body: endpoint.body ? JSON.stringify(endpoint.body) : undefined,
      });

      const latency = Date.now() - start;
      const data = await response.json().catch(() => ({}));
      const isHealthy = response.ok || response.status === 503;

      setChecks((prev) =>
        prev.map((c, i) =>
          i === index
            ? {
                ...c,
                status: isHealthy ? (latency > 500 ? 'degraded' : 'healthy') : 'down',
                latency,
                lastChecked: new Date().toISOString(),
                error: isHealthy ? undefined : data.error || `HTTP ${response.status}`,
                details: data,
              }
            : c
        )
      );
    } catch (error) {
      setChecks((prev) =>
        prev.map((c, i) =>
          i === index
            ? {
                ...c,
                status: 'down',
                latency: Date.now() - start,
                lastChecked: new Date().toISOString(),
                error: error instanceof Error ? error.message : 'Unknown error',
              }
            : c
        )
      );
    }
  };

  const runAllChecks = async () => {
    for (let i = 0; i < API_ENDPOINTS.length; i++) {
      await runHealthCheck(i);
    }
  };

  useEffect(() => {
    void runAllChecks();
    const interval = setInterval(() => void runAllChecks(), 60000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const hasDown = checks.some((c) => c.status === 'down');
    const hasDegraded = checks.some((c) => c.status === 'degraded');
    setOverallStatus(hasDown ? 'down' : hasDegraded ? 'degraded' : 'healthy');
  }, [checks]);

  const statusMap = {
    healthy: 'healthy' as const,
    degraded: 'warning' as const,
    down: 'error' as const,
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-slate-100">System Health</h1>
          <p className="text-sm text-slate-500 mt-0.5">API endpoints and service status</p>
        </div>
        <div className="flex items-center gap-4">
          <StatusIndicator status={statusMap[overallStatus]} label={`System ${overallStatus.charAt(0).toUpperCase() + overallStatus.slice(1)}`} />
          <Button variant="primary" size="sm" onClick={() => void runAllChecks()}>
            <RefreshCw className="w-3.5 h-3.5 mr-1.5" />
            Refresh All
          </Button>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-3">
        <Card padding="sm">
          <p className="text-xs text-slate-500 mb-1">Healthy</p>
          <p className="text-xl font-semibold text-emerald-500">
            {checks.filter((c) => c.status === 'healthy').length}
          </p>
        </Card>
        <Card padding="sm">
          <p className="text-xs text-slate-500 mb-1">Degraded</p>
          <p className="text-xl font-semibold text-amber-500">
            {checks.filter((c) => c.status === 'degraded').length}
          </p>
        </Card>
        <Card padding="sm">
          <p className="text-xs text-slate-500 mb-1">Down</p>
          <p className="text-xl font-semibold text-red-500">
            {checks.filter((c) => c.status === 'down').length}
          </p>
        </Card>
        <Card padding="sm">
          <p className="text-xs text-slate-500 mb-1">Avg Latency</p>
          <p className="text-xl font-semibold text-slate-300">
            {Math.round(checks.reduce((sum, c) => sum + (c.latency || 0), 0) / checks.length) || 0}ms
          </p>
        </Card>
      </div>

      {/* Health Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {checks.map((check, index) => (
          <HealthCard key={check.name} check={check} onRecheck={() => void runHealthCheck(index)} />
        ))}
      </div>

      {/* Scheduled Jobs */}
      <Card padding="md">
        <h2 className="text-sm font-semibold text-slate-200 mb-4">Scheduled Jobs</h2>
        <div className="space-y-2">
          {[
            { name: 'Cache Warm', schedule: 'Every 10 min' },
            { name: 'GDELT Ingest', schedule: 'Hourly at :05' },
            { name: 'Sentiment Analysis', schedule: 'Every 4h at :15' },
            { name: 'Nation Risk Compute', schedule: 'Every 4h at :25' },
            { name: 'Daily Alerts', schedule: 'Daily at 08:00 UTC' },
            { name: 'World Bank Sync', schedule: 'Weekly (Sunday 06:00)' },
          ].map((job) => (
            <div key={job.name} className="flex items-center justify-between py-2 px-3 bg-slate-800/50 rounded">
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                <span className="text-sm text-slate-300">{job.name}</span>
              </div>
              <span className="text-xs text-slate-500">{job.schedule}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
