'use client';

import { useState, useEffect } from 'react';
import { GlassCard } from '@/components/ui/GlassCard';
import {
  TrendingUp,
  Clock,
  Target,
  DollarSign,
  AlertTriangle,
  Download,
  RefreshCw
} from 'lucide-react';

/**
 * ROI Metrics Dashboard
 *
 * Anti-Complaint Spec Section 1.2 Implementation:
 * - Time saved vs manual OSINT
 * - Threats detected before public disclosure
 * - False positive rate with user feedback
 * - Cost per actionable alert
 */

interface ROIMetrics {
  // Time savings
  hoursManualOSINT: number;
  hoursWithLatticeForge: number;
  hoursSaved: number;
  dollarsPerHour: number;

  // Threat detection
  totalThreatsDetected: number;
  threatsBeforePublic: number;
  avgLeadTimeDays: number;

  // Alert quality
  totalAlerts: number;
  alertsActedUpon: number;
  alertsDismissed: number;
  falsePositiveRate: number;

  // Cost efficiency
  subscriptionCost: number;
  costPerAlert: number;
  costPerActionableAlert: number;

  // Period
  periodDays: number;
}

interface ROIDashboardProps {
  metrics?: Partial<ROIMetrics>;
  onExport?: () => void;
}

// Mock data generator for demo
function generateMockMetrics(periodDays: number): ROIMetrics {
  const scaleFactor = periodDays / 30;

  const hoursManualOSINT = Math.round(120 * scaleFactor);
  const hoursWithLatticeForge = Math.round(25 * scaleFactor);
  const totalAlerts = Math.round(45 * scaleFactor);
  const alertsActedUpon = Math.round(38 * scaleFactor);
  const alertsDismissed = totalAlerts - alertsActedUpon;

  return {
    hoursManualOSINT,
    hoursWithLatticeForge,
    hoursSaved: hoursManualOSINT - hoursWithLatticeForge,
    dollarsPerHour: 85,

    totalThreatsDetected: Math.round(12 * scaleFactor),
    threatsBeforePublic: Math.round(8 * scaleFactor),
    avgLeadTimeDays: 3.2,

    totalAlerts,
    alertsActedUpon,
    alertsDismissed,
    falsePositiveRate: alertsDismissed / totalAlerts,

    subscriptionCost: 1499 * (periodDays / 30),
    costPerAlert: 0,
    costPerActionableAlert: 0,

    periodDays,
  };
}

export default function ROIDashboard({ metrics: initialMetrics, onExport }: ROIDashboardProps) {
  const [period, setPeriod] = useState<30 | 90 | 365>(30);
  const [metrics, setMetrics] = useState<ROIMetrics>(() => generateMockMetrics(30));

  useEffect(() => {
    // In production, this would fetch real metrics from the API
    const newMetrics = generateMockMetrics(period);
    newMetrics.costPerAlert = newMetrics.subscriptionCost / newMetrics.totalAlerts;
    newMetrics.costPerActionableAlert = newMetrics.subscriptionCost / newMetrics.alertsActedUpon;
    setMetrics(newMetrics);
  }, [period]);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const timeSavingsValue = metrics.hoursSaved * metrics.dollarsPerHour;
  const roi = ((timeSavingsValue - metrics.subscriptionCost) / metrics.subscriptionCost) * 100;

  const handleExport = () => {
    const report = {
      generatedAt: new Date().toISOString(),
      periodDays: metrics.periodDays,
      timeSavings: {
        hoursManualOSINT: metrics.hoursManualOSINT,
        hoursWithLatticeForge: metrics.hoursWithLatticeForge,
        hoursSaved: metrics.hoursSaved,
        dollarsPerHour: metrics.dollarsPerHour,
        totalSavings: timeSavingsValue,
      },
      threatDetection: {
        totalDetected: metrics.totalThreatsDetected,
        beforePublicDisclosure: metrics.threatsBeforePublic,
        avgLeadTimeDays: metrics.avgLeadTimeDays,
      },
      alertQuality: {
        totalAlerts: metrics.totalAlerts,
        actedUpon: metrics.alertsActedUpon,
        dismissed: metrics.alertsDismissed,
        falsePositiveRate: metrics.falsePositiveRate,
      },
      costEfficiency: {
        subscriptionCost: metrics.subscriptionCost,
        costPerAlert: metrics.costPerAlert,
        costPerActionableAlert: metrics.costPerActionableAlert,
        roi: roi,
      },
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `latticeforge-roi-${period}d-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);

    onExport?.();
  };

  return (
    <GlassCard blur="heavy" className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <TrendingUp className="w-6 h-6 text-green-400" />
          <div>
            <h2 className="text-xl font-bold text-white">Value Demonstration</h2>
            <p className="text-sm text-slate-400">ROI metrics for procurement justification</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {/* Period Selector */}
          <div className="flex bg-black/30 rounded-lg p-1">
            {([30, 90, 365] as const).map((days) => (
              <button
                key={days}
                onClick={() => setPeriod(days)}
                className={`px-3 py-1 rounded-md text-sm transition-colors ${
                  period === days
                    ? 'bg-blue-500 text-white'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {days === 365 ? '1Y' : `${days}D`}
              </button>
            ))}
          </div>
          <button
            onClick={handleExport}
            className="flex items-center gap-2 px-3 py-1.5 bg-black/30 border border-white/[0.08] rounded-lg text-slate-300 hover:text-white hover:border-white/20 transition-colors"
          >
            <Download className="w-4 h-4" />
            Export
          </button>
        </div>
      </div>

      {/* ROI Hero Metric */}
      <div className="mb-6 p-6 bg-gradient-to-br from-green-500/10 to-blue-500/10 rounded-xl border border-green-500/20">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-slate-400 mb-1">Return on Investment</p>
            <p className="text-5xl font-bold text-white">{roi.toFixed(0)}%</p>
            <p className="text-sm text-green-400 mt-1">
              {formatCurrency(timeSavingsValue)} saved vs {formatCurrency(metrics.subscriptionCost)} cost
            </p>
          </div>
          <div className="text-right">
            <p className="text-sm text-slate-400">Net Value Created</p>
            <p className="text-3xl font-bold text-green-400">
              {formatCurrency(timeSavingsValue - metrics.subscriptionCost)}
            </p>
          </div>
        </div>
      </div>

      {/* Metric Cards Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {/* Time Saved */}
        <div className="p-4 bg-black/20 rounded-xl border border-white/[0.04]">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-4 h-4 text-blue-400" />
            <span className="text-sm text-slate-400">Time Saved</span>
          </div>
          <p className="text-2xl font-bold text-white">{metrics.hoursSaved}h</p>
          <p className="text-xs text-slate-500 mt-1">
            {metrics.hoursWithLatticeForge}h vs {metrics.hoursManualOSINT}h manual
          </p>
        </div>

        {/* Threats Before Public */}
        <div className="p-4 bg-black/20 rounded-xl border border-white/[0.04]">
          <div className="flex items-center gap-2 mb-2">
            <Target className="w-4 h-4 text-purple-400" />
            <span className="text-sm text-slate-400">Early Detection</span>
          </div>
          <p className="text-2xl font-bold text-white">{metrics.threatsBeforePublic}</p>
          <p className="text-xs text-slate-500 mt-1">
            Avg {metrics.avgLeadTimeDays.toFixed(1)} days before public disclosure
          </p>
        </div>

        {/* False Positive Rate */}
        <div className="p-4 bg-black/20 rounded-xl border border-white/[0.04]">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            <span className="text-sm text-slate-400">False Positive Rate</span>
          </div>
          <p className="text-2xl font-bold text-white">{formatPercent(metrics.falsePositiveRate)}</p>
          <p className="text-xs text-slate-500 mt-1">
            {metrics.alertsActedUpon} of {metrics.totalAlerts} alerts actioned
          </p>
        </div>

        {/* Cost Per Actionable Alert */}
        <div className="p-4 bg-black/20 rounded-xl border border-white/[0.04]">
          <div className="flex items-center gap-2 mb-2">
            <DollarSign className="w-4 h-4 text-green-400" />
            <span className="text-sm text-slate-400">Cost/Actionable Alert</span>
          </div>
          <p className="text-2xl font-bold text-white">{formatCurrency(metrics.costPerActionableAlert)}</p>
          <p className="text-xs text-slate-500 mt-1">
            {formatCurrency(metrics.costPerAlert)}/total alert
          </p>
        </div>
      </div>

      {/* Alert Breakdown Bar */}
      <div className="mb-4">
        <div className="flex justify-between text-sm mb-2">
          <span className="text-slate-400">Alert Quality Distribution</span>
          <span className="text-slate-400">{metrics.totalAlerts} total alerts</span>
        </div>
        <div className="h-4 bg-black/30 rounded-full overflow-hidden flex">
          <div
            className="bg-green-500 transition-all"
            style={{ width: `${(metrics.alertsActedUpon / metrics.totalAlerts) * 100}%` }}
            title={`${metrics.alertsActedUpon} actioned`}
          />
          <div
            className="bg-amber-500 transition-all"
            style={{ width: `${(metrics.alertsDismissed / metrics.totalAlerts) * 100}%` }}
            title={`${metrics.alertsDismissed} dismissed`}
          />
        </div>
        <div className="flex gap-4 mt-2 text-xs">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-green-500" />
            <span className="text-slate-400">Actioned ({formatPercent(metrics.alertsActedUpon / metrics.totalAlerts)})</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-amber-500" />
            <span className="text-slate-400">Dismissed ({formatPercent(metrics.falsePositiveRate)})</span>
          </span>
        </div>
      </div>

      {/* Time Savings Breakdown */}
      <div className="p-4 bg-black/20 rounded-xl border border-white/[0.04]">
        <h3 className="text-sm font-medium text-white mb-3">Time Savings Breakdown</h3>
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">Manual OSINT (estimated)</span>
            <span className="text-slate-300">{metrics.hoursManualOSINT} hours</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">With LatticeForge</span>
            <span className="text-slate-300">{metrics.hoursWithLatticeForge} hours</span>
          </div>
          <div className="border-t border-white/[0.08] pt-2 flex justify-between text-sm">
            <span className="text-white font-medium">Hours Saved</span>
            <span className="text-green-400 font-medium">{metrics.hoursSaved} hours</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">@ ${metrics.dollarsPerHour}/hour analyst cost</span>
            <span className="text-green-400 font-medium">{formatCurrency(timeSavingsValue)}</span>
          </div>
        </div>
      </div>
    </GlassCard>
  );
}
