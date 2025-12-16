'use client';

import { useState, useEffect } from 'react';
import { RefreshCw, AlertTriangle, Bell, CheckCircle } from 'lucide-react';
import { Card, Button, EmptyState, Skeleton, SkeletonList } from '@/components/ui';

interface Alert {
  id: string;
  type: 'critical' | 'warning' | 'info';
  category: string;
  title: string;
  message: string;
  nation?: string;
  timestamp: string;
  read: boolean;
}

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAlerts();
  }, []);

  async function fetchAlerts() {
    setLoading(true);
    try {
      const res = await fetch('/api/query/alerts', {
        headers: { 'x-user-tier': 'enterprise_tier' },
      });
      const data = await res.json();
      setAlerts(data.alerts || []);
    } catch (e) {
      console.error('Failed to fetch alerts:', e);
    } finally {
      setLoading(false);
    }
  }

  const criticalCount = alerts.filter(a => a.type === 'critical').length;
  const warningCount = alerts.filter(a => a.type === 'warning').length;

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-lg font-semibold text-slate-100">Risk Alerts</h1>
          <p className="text-sm text-slate-500 mt-0.5">Signals exceeding configured thresholds</p>
        </div>
        <Button variant="secondary" size="sm" onClick={fetchAlerts} loading={loading}>
          <RefreshCw className="w-3.5 h-3.5 mr-1.5" />
          Refresh
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-3">
        {loading ? (
          <>
            <Skeleton className="h-16 rounded-md" />
            <Skeleton className="h-16 rounded-md" />
            <Skeleton className="h-16 rounded-md" />
          </>
        ) : (
          <>
            <Card padding="sm">
              <p className="text-xs text-slate-500 mb-1">Critical</p>
              <p className="text-xl font-semibold text-red-500">{criticalCount}</p>
            </Card>
            <Card padding="sm">
              <p className="text-xs text-slate-500 mb-1">Warning</p>
              <p className="text-xl font-semibold text-amber-500">{warningCount}</p>
            </Card>
            <Card padding="sm">
              <p className="text-xs text-slate-500 mb-1">Total Active</p>
              <p className="text-xl font-semibold text-slate-300">{alerts.length}</p>
            </Card>
          </>
        )}
      </div>

      {/* Alert List */}
      <div className="space-y-2">
        {loading ? (
          <SkeletonList items={5} />
        ) : alerts.length === 0 ? (
          <EmptyState
            icon={CheckCircle}
            title="No active alerts"
            description="All signals are within configured thresholds. System operating normally."
          />
        ) : (
          alerts.map((alert) => (
            <Card
              key={alert.id}
              padding="sm"
              className={`border-l-2 ${
                alert.type === 'critical' ? 'border-l-red-500' : 'border-l-amber-500'
              }`}
            >
              <div className="flex items-start gap-3">
                {alert.type === 'critical' ? (
                  <AlertTriangle className="w-4 h-4 text-red-500 mt-0.5 shrink-0" />
                ) : (
                  <Bell className="w-4 h-4 text-amber-500 mt-0.5 shrink-0" />
                )}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-sm text-slate-200 font-medium">{alert.title}</span>
                    {alert.nation && (
                      <span className="px-1.5 py-0.5 bg-slate-800 border border-slate-700 rounded text-[10px] text-slate-400 uppercase">
                        {alert.nation}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-400">{alert.message}</p>
                  <p className="text-[10px] text-slate-600 mt-1">
                    {new Date(alert.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
            </Card>
          ))
        )}
      </div>
    </div>
  );
}
