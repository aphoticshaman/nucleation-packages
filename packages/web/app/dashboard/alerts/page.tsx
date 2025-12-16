'use client';

import { useState, useEffect } from 'react';
import { RefreshCw, AlertTriangle, Bell, CheckCircle } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';

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
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Risk Alerts</h1>
          <p className="text-slate-400 mt-1">High-risk signals requiring attention</p>
        </div>
        <GlassButton variant="secondary" size="sm" onClick={fetchAlerts}>
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </GlassButton>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <GlassCard className="p-4 text-center">
          <div className="text-3xl font-bold text-red-400">{criticalCount}</div>
          <div className="text-sm text-slate-400">Critical</div>
        </GlassCard>
        <GlassCard className="p-4 text-center">
          <div className="text-3xl font-bold text-amber-400">{warningCount}</div>
          <div className="text-sm text-slate-400">Warnings</div>
        </GlassCard>
        <GlassCard className="p-4 text-center">
          <div className="text-3xl font-bold text-green-400">{alerts.length}</div>
          <div className="text-sm text-slate-400">Total Active</div>
        </GlassCard>
      </div>

      <div className="space-y-2">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <RefreshCw className="w-8 h-8 text-slate-500 animate-spin" />
          </div>
        ) : alerts.length === 0 ? (
          <GlassCard className="p-8 text-center">
            <CheckCircle className="w-10 h-10 text-green-400 mx-auto mb-3" />
            <p className="text-slate-400">No active alerts</p>
          </GlassCard>
        ) : (
          alerts.map((alert) => (
            <GlassCard
              key={alert.id}
              className={`p-4 border-l-4 ${
                alert.type === 'critical' ? 'border-l-red-500' : 'border-l-amber-500'
              }`}
            >
              <div className="flex items-start gap-4">
                {alert.type === 'critical' ? (
                  <AlertTriangle className="w-5 h-5 text-red-400 mt-0.5" />
                ) : (
                  <Bell className="w-5 h-5 text-amber-400 mt-0.5" />
                )}
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-white font-medium">{alert.title}</span>
                    {alert.nation && (
                      <span className="px-2 py-0.5 bg-slate-700 rounded text-xs text-slate-300">
                        {alert.nation}
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-slate-400">{alert.message}</p>
                  <p className="text-xs text-slate-500 mt-1">
                    {new Date(alert.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
            </GlassCard>
          ))
        )}
      </div>
    </div>
  );
}
