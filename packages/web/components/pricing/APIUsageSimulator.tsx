'use client';

import { useState, useMemo } from 'react';
import { GlassCard } from '@/components/ui/GlassCard';
import { Server, Zap, BarChart3, Info } from 'lucide-react';

/**
 * API Usage Simulator
 *
 * Anti-Complaint Spec Section 1.1 Implementation:
 * - API call pricing with usage simulator
 * - Real-time cost estimation
 * - Workload-based recommendations
 */

interface WorkloadProfile {
  name: string;
  description: string;
  dailyCalls: number;
  peakMultiplier: number;
  burstFrequency: 'rare' | 'occasional' | 'frequent';
}

const WORKLOAD_PROFILES: WorkloadProfile[] = [
  {
    name: 'Light',
    description: 'Small team, periodic monitoring',
    dailyCalls: 500,
    peakMultiplier: 2,
    burstFrequency: 'rare',
  },
  {
    name: 'Standard',
    description: 'Regular intelligence operations',
    dailyCalls: 3000,
    peakMultiplier: 3,
    burstFrequency: 'occasional',
  },
  {
    name: 'Heavy',
    description: 'Continuous monitoring, automations',
    dailyCalls: 15000,
    peakMultiplier: 4,
    burstFrequency: 'frequent',
  },
  {
    name: 'Enterprise',
    description: 'High-volume production workloads',
    dailyCalls: 50000,
    peakMultiplier: 5,
    burstFrequency: 'frequent',
  },
];

interface PricingTier {
  name: string;
  monthlyBase: number;
  includedCalls: number;
  overageRate: number; // per 1000 calls
  rateLimitPerMin: number;
  burstAllowance: number;
}

const TIERS: PricingTier[] = [
  {
    name: 'Entry',
    monthlyBase: 299,
    includedCalls: 10000,
    overageRate: 0.10,
    rateLimitPerMin: 60,
    burstAllowance: 120,
  },
  {
    name: 'Team',
    monthlyBase: 1499,
    includedCalls: 100000,
    overageRate: 0.05,
    rateLimitPerMin: 300,
    burstAllowance: 600,
  },
  {
    name: 'Enterprise',
    monthlyBase: 4999,
    includedCalls: 1000000,
    overageRate: 0.02,
    rateLimitPerMin: 1000,
    burstAllowance: 2000,
  },
];

export default function APIUsageSimulator() {
  const [selectedProfile, setSelectedProfile] = useState(1); // Standard
  const [customCalls, setCustomCalls] = useState<number | null>(null);

  const profile = WORKLOAD_PROFILES[selectedProfile];
  const dailyCalls = customCalls ?? profile.dailyCalls;
  const monthlyCalls = dailyCalls * 30;

  const analysis = useMemo(() => {
    return TIERS.map((tier) => {
      const overageCalls = Math.max(0, monthlyCalls - tier.includedCalls);
      const overageCost = (overageCalls / 1000) * tier.overageRate;
      const totalCost = tier.monthlyBase + overageCost;

      // Rate limit analysis
      const peakCallsPerMin = (dailyCalls / (8 * 60)) * profile.peakMultiplier; // Assuming 8hr workday
      const rateLimitOk = peakCallsPerMin <= tier.rateLimitPerMin;
      const burstOk = peakCallsPerMin <= tier.burstAllowance;

      return {
        tier,
        overageCalls,
        overageCost,
        totalCost,
        peakCallsPerMin: Math.round(peakCallsPerMin),
        rateLimitOk,
        burstOk,
        costPerCall: totalCost / monthlyCalls,
        recommended: false, // Will be set below
      };
    });
  }, [monthlyCalls, dailyCalls, profile.peakMultiplier]);

  // Find recommended tier
  const recommendedIdx = analysis.findIndex(
    (a) => a.rateLimitOk && monthlyCalls <= a.tier.includedCalls * 1.2
  );
  if (recommendedIdx >= 0) {
    analysis[recommendedIdx].recommended = true;
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  const formatNumber = (value: number) => {
    return new Intl.NumberFormat('en-US').format(value);
  };

  return (
    <GlassCard blur="heavy" className="p-6">
      <div className="flex items-center gap-3 mb-6">
        <Server className="w-6 h-6 text-purple-400" />
        <div>
          <h2 className="text-xl font-bold text-white">API Usage Simulator</h2>
          <p className="text-sm text-slate-400">Estimate costs based on your workload</p>
        </div>
      </div>

      {/* Workload Profile Selector */}
      <div className="mb-6">
        <label className="block text-sm text-slate-400 mb-2">Select Workload Profile</label>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {WORKLOAD_PROFILES.map((p, idx) => (
            <button
              key={p.name}
              onClick={() => {
                setSelectedProfile(idx);
                setCustomCalls(null);
              }}
              className={`p-3 rounded-lg border text-left transition-all ${
                selectedProfile === idx
                  ? 'border-purple-500/50 bg-purple-500/10'
                  : 'border-white/[0.08] bg-black/20 hover:border-white/20'
              }`}
            >
              <p className="font-medium text-white">{p.name}</p>
              <p className="text-xs text-slate-400">{p.description}</p>
              <p className="text-xs text-purple-400 mt-1">{formatNumber(p.dailyCalls)}/day</p>
            </button>
          ))}
        </div>
      </div>

      {/* Custom Override */}
      <div className="mb-6 flex items-center gap-4">
        <div className="flex-1">
          <label className="block text-sm text-slate-400 mb-1">Custom Daily API Calls</label>
          <input
            type="number"
            min={0}
            placeholder={profile.dailyCalls.toString()}
            value={customCalls ?? ''}
            onChange={(e) => setCustomCalls(e.target.value ? parseInt(e.target.value) : null)}
            className="w-full px-3 py-2 bg-black/30 border border-white/[0.08] rounded-lg text-white focus:outline-none focus:border-purple-500/50"
          />
        </div>
        <div className="pt-6">
          <p className="text-sm text-slate-400">
            Monthly: <span className="text-white font-medium">{formatNumber(monthlyCalls)}</span> calls
          </p>
        </div>
      </div>

      {/* Tier Comparison */}
      <div className="space-y-3">
        {analysis.map((a) => (
          <div
            key={a.tier.name}
            className={`p-4 rounded-xl border ${
              a.recommended
                ? 'border-green-500/30 bg-green-500/5'
                : 'border-white/[0.04] bg-black/20'
            }`}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                <span className={`text-lg font-bold ${a.recommended ? 'text-green-400' : 'text-white'}`}>
                  {a.tier.name}
                </span>
                {a.recommended && (
                  <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded-full">
                    Recommended
                  </span>
                )}
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-white">{formatCurrency(a.totalCost)}</p>
                <p className="text-xs text-slate-400">/month</p>
              </div>
            </div>

            <div className="grid grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-slate-400">Included</p>
                <p className="text-white">{formatNumber(a.tier.includedCalls)}</p>
              </div>
              <div>
                <p className="text-slate-400">Overage</p>
                <p className={a.overageCalls > 0 ? 'text-amber-400' : 'text-slate-500'}>
                  {a.overageCalls > 0 ? `+${formatNumber(a.overageCalls)}` : '-'}
                </p>
              </div>
              <div>
                <p className="text-slate-400">Overage Cost</p>
                <p className={a.overageCost > 0 ? 'text-amber-400' : 'text-slate-500'}>
                  {a.overageCost > 0 ? formatCurrency(a.overageCost) : '-'}
                </p>
              </div>
              <div>
                <p className="text-slate-400">Cost/1K calls</p>
                <p className="text-white">{formatCurrency(a.costPerCall * 1000)}</p>
              </div>
            </div>

            {/* Rate Limit Status */}
            <div className="mt-3 pt-3 border-t border-white/[0.08] flex items-center gap-4 text-xs">
              <div className="flex items-center gap-1">
                <Zap className={`w-3 h-3 ${a.rateLimitOk ? 'text-green-400' : 'text-red-400'}`} />
                <span className="text-slate-400">
                  Rate Limit: {a.tier.rateLimitPerMin}/min
                  <span className={a.rateLimitOk ? 'text-green-400' : 'text-red-400'}>
                    {' '}(peak: {a.peakCallsPerMin}/min)
                  </span>
                </span>
              </div>
              <div className="flex items-center gap-1">
                <BarChart3 className={`w-3 h-3 ${a.burstOk ? 'text-green-400' : 'text-amber-400'}`} />
                <span className="text-slate-400">
                  Burst: {a.tier.burstAllowance}/min
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* API Pricing Note */}
      <div className="mt-6 p-4 bg-black/20 rounded-xl border border-white/[0.04]">
        <div className="flex items-start gap-3">
          <Info className="w-4 h-4 text-blue-400 mt-0.5" />
          <div className="text-sm text-slate-400">
            <p className="mb-2">
              <strong className="text-white">Rate Limits:</strong> Predictable and burst-friendly.
              Temporary bursts up to 2x the standard limit are allowed without throttling.
            </p>
            <p>
              <strong className="text-white">Sandbox:</strong> Full-featured test environment
              included free with all tiers. API versioning follows semantic versioning with
              18-month deprecation windows.
            </p>
          </div>
        </div>
      </div>
    </GlassCard>
  );
}
