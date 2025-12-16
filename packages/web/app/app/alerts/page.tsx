'use client';

import { useState, useEffect } from 'react';
import {
  Bell,
  AlertTriangle,
  TrendingUp,
  Shield,
  Globe,
  Clock,
  Check,
  RefreshCw,
} from 'lucide-react';
import { Card, Button } from '@/components/ui';
import Link from 'next/link';

interface Alert {
  id: string;
  type: 'critical' | 'warning' | 'info';
  category: 'risk' | 'regime' | 'security' | 'briefing';
  title: string;
  message: string;
  nation?: string;
  timestamp: string;
  read: boolean;
}

const typeStyles = {
  critical: {
    bg: 'bg-red-500/20',
    border: 'border-red-500/30',
    text: 'text-red-400',
    icon: <AlertTriangle className="w-5 h-5" />,
  },
  warning: {
    bg: 'bg-amber-500/20',
    border: 'border-amber-500/30',
    text: 'text-amber-400',
    icon: <TrendingUp className="w-5 h-5" />,
  },
  info: {
    bg: 'bg-blue-500/20',
    border: 'border-blue-500/30',
    text: 'text-blue-400',
    icon: <Bell className="w-5 h-5" />,
  },
};

const categoryIcons = {
  risk: <TrendingUp className="w-4 h-4" />,
  regime: <Globe className="w-4 h-4" />,
  security: <Shield className="w-4 h-4" />,
  briefing: <Bell className="w-4 h-4" />,
};

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [filter, setFilter] = useState<'all' | 'unread'>('all');
  const [loading, setLoading] = useState(true);
  const [readAlerts, setReadAlerts] = useState<Set<string>>(new Set());

  useEffect(() => {
    fetchAlerts();
  }, []);

  async function fetchAlerts() {
    setLoading(true);
    try {
      const res = await fetch('/api/query/alerts');
      if (res.ok) {
        const data = await res.json();
        setAlerts(data.alerts || []);
      }
    } catch (e) {
      console.error('Failed to fetch alerts:', e);
    } finally {
      setLoading(false);
    }
  }

  const filteredAlerts = alerts.filter((a) => {
    const isRead = readAlerts.has(a.id);
    if (filter === 'unread') return !isRead;
    return true;
  });

  const unreadCount = alerts.filter((a) => !readAlerts.has(a.id)).length;

  const markAsRead = (id: string) => {
    setReadAlerts((prev) => new Set(prev).add(id));
  };

  const markAllRead = () => {
    setReadAlerts(new Set(alerts.map((a) => a.id)));
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffHours < 1) return 'Just now';
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-white">Alerts</h1>
          <p className="text-slate-400 text-sm mt-1">
            {loading
              ? 'Loading...'
              : unreadCount > 0
                ? `${unreadCount} unread alert${unreadCount > 1 ? 's' : ''}`
                : 'No unread alerts'}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="secondary" size="sm" onClick={fetchAlerts}>
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Link href="/app/notifications">
            <Button variant="secondary" size="sm">
              <Bell className="w-4 h-4 mr-2" />
              Settings
            </Button>
          </Link>
          {unreadCount > 0 && (
            <Button variant="secondary" size="sm" onClick={markAllRead}>
              <Check className="w-4 h-4 mr-2" />
              Mark All Read
            </Button>
          )}
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => setFilter('all')}
          className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
            filter === 'all'
              ? 'bg-white/10 text-white'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          All
        </button>
        <button
          onClick={() => setFilter('unread')}
          className={`px-3 py-1.5 rounded-lg text-sm transition-colors flex items-center gap-2 ${
            filter === 'unread'
              ? 'bg-white/10 text-white'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          Unread
          {unreadCount > 0 && (
            <span className="px-1.5 py-0.5 bg-cyan-500/20 text-cyan-400 text-xs rounded-full">
              {unreadCount}
            </span>
          )}
        </button>
      </div>

      {/* Alerts List */}
      <div className="space-y-3">
        {loading ? (
          <Card className="p-8 text-center">
            <RefreshCw className="w-8 h-8 text-slate-500 mx-auto mb-4 animate-spin" />
            <p className="text-slate-400">Loading alerts...</p>
          </Card>
        ) : filteredAlerts.length === 0 ? (
          <Card className="p-8 text-center">
            <Bell className="w-12 h-12 text-slate-500 mx-auto mb-4" />
            <p className="text-slate-400">
              {filter === 'unread'
                ? "No unread alerts. You're all caught up!"
                : 'No high-risk signals detected in the past 7 days.'}
            </p>
            <Link href="/app/notifications" className="text-cyan-400 text-sm mt-2 inline-block hover:underline">
              Configure alert settings →
            </Link>
          </Card>
        ) : (
          filteredAlerts.map((alert) => {
            const style = typeStyles[alert.type];
            const isRead = readAlerts.has(alert.id);
            return (
              <Card
                key={alert.id}
                className={`p-4 cursor-pointer transition-all hover:bg-white/5 ${
                  !isRead ? 'border-l-2 border-l-cyan-500' : ''
                }`}
                onClick={() => markAsRead(alert.id)}
              >
                <div className="flex items-start gap-4">
                  {/* Icon */}
                  <div className={`p-2 rounded-lg ${style.bg} ${style.text}`}>
                    {style.icon}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`font-medium ${!isRead ? 'text-white' : 'text-slate-300'}`}>
                        {alert.title}
                      </span>
                      {alert.nation && (
                        <span className="px-2 py-0.5 bg-slate-700 text-slate-300 text-xs rounded">
                          {alert.nation}
                        </span>
                      )}
                      <span className="text-slate-500 text-xs flex items-center gap-1">
                        {categoryIcons[alert.category]}
                        {alert.category}
                      </span>
                    </div>
                    <p className={`text-sm ${!isRead ? 'text-slate-300' : 'text-slate-400'}`}>
                      {alert.message}
                    </p>
                    <div className="flex items-center gap-2 mt-2 text-xs text-slate-500">
                      <Clock className="w-3 h-3" />
                      {formatTime(alert.timestamp)}
                      {!isRead && (
                        <span className="w-2 h-2 bg-cyan-500 rounded-full" />
                      )}
                    </div>
                  </div>
                </div>
              </Card>
            );
          })
        )}
      </div>

      {/* Info */}
      <Card className="p-4 border-dashed">
        <div className="flex items-center gap-3 text-slate-400">
          <Shield className="w-5 h-5" />
          <div>
            <p className="text-sm font-medium">Alerts are generated from risk signals</p>
            <p className="text-xs">
              High-risk events from GDELT, USGS, and other sources trigger alerts automatically.
              <Link href="/app/notifications" className="text-cyan-400 ml-1 hover:underline">
                Customize thresholds →
              </Link>
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}
