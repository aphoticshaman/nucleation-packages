'use client';

import { useState, useMemo } from 'react';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';
import { Calculator, TrendingUp, Download, ChevronDown, ChevronUp } from 'lucide-react';

/**
 * Total Cost of Ownership Calculator
 *
 * Anti-Complaint Spec Section 1.1 Implementation:
 * - 5-year TCO projection
 * - Egress cost estimator with real data volumes
 * - API call pricing simulator
 * - No hidden costs
 */

interface PricingTier {
  name: string;
  monthlyBase: number;
  apiCallsIncluded: number;
  apiOverageRate: number; // per 1000 calls
  dataExportIncluded: number; // GB per month
  dataExportRate: number; // per GB over limit
  seats: number | 'unlimited';
  seatOverageRate: number;
}

const PRICING_TIERS: Record<string, PricingTier> = {
  entry: {
    name: 'Entry',
    monthlyBase: 299,
    apiCallsIncluded: 10000,
    apiOverageRate: 0.10, // $0.10 per 1000
    dataExportIncluded: 10,
    dataExportRate: 0, // Free export per spec
    seats: 1,
    seatOverageRate: 99,
  },
  team: {
    name: 'Team',
    monthlyBase: 1499,
    apiCallsIncluded: 100000,
    apiOverageRate: 0.05,
    dataExportIncluded: 100,
    dataExportRate: 0,
    seats: 10,
    seatOverageRate: 149,
  },
  enterprise: {
    name: 'Enterprise',
    monthlyBase: 4999,
    apiCallsIncluded: 1000000,
    apiOverageRate: 0.02,
    dataExportIncluded: 1000,
    dataExportRate: 0, // Always free per spec
    seats: 'unlimited',
    seatOverageRate: 0,
  },
};

interface TCOInputs {
  tier: string;
  analysts: number;
  monthlyApiCalls: number;
  monthlyDataExportGB: number;
  projectionYears: number;
  annualGrowthRate: number; // percent
}

interface TCOBreakdown {
  year: number;
  baseCost: number;
  seatOverage: number;
  apiOverage: number;
  exportCost: number;
  totalAnnual: number;
  cumulativeTotal: number;
}

export default function TCOCalculator() {
  const [inputs, setInputs] = useState<TCOInputs>({
    tier: 'team',
    analysts: 5,
    monthlyApiCalls: 50000,
    monthlyDataExportGB: 25,
    projectionYears: 5,
    annualGrowthRate: 20,
  });

  const [showAdvanced, setShowAdvanced] = useState(false);

  const calculations = useMemo(() => {
    const tier = PRICING_TIERS[inputs.tier];
    const breakdown: TCOBreakdown[] = [];
    let cumulativeTotal = 0;

    for (let year = 1; year <= inputs.projectionYears; year++) {
      const growthFactor = Math.pow(1 + inputs.annualGrowthRate / 100, year - 1);
      const currentAnalysts = Math.ceil(inputs.analysts * growthFactor);
      const currentApiCalls = Math.ceil(inputs.monthlyApiCalls * growthFactor);
      const currentDataExport = inputs.monthlyDataExportGB * growthFactor;

      // Base cost (12 months)
      const baseCost = tier.monthlyBase * 12;

      // Seat overage
      let seatOverage = 0;
      if (tier.seats !== 'unlimited' && currentAnalysts > tier.seats) {
        seatOverage = (currentAnalysts - tier.seats) * tier.seatOverageRate * 12;
      }

      // API overage
      let apiOverage = 0;
      if (currentApiCalls > tier.apiCallsIncluded) {
        const overageCalls = currentApiCalls - tier.apiCallsIncluded;
        apiOverage = (overageCalls / 1000) * tier.apiOverageRate * 12;
      }

      // Data export cost (always $0 per anti-complaint spec)
      const exportCost = 0;

      const totalAnnual = baseCost + seatOverage + apiOverage + exportCost;
      cumulativeTotal += totalAnnual;

      breakdown.push({
        year,
        baseCost,
        seatOverage,
        apiOverage,
        exportCost,
        totalAnnual,
        cumulativeTotal,
      });
    }

    return breakdown;
  }, [inputs]);

  const totalTCO = calculations[calculations.length - 1]?.cumulativeTotal || 0;

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const handleExport = () => {
    const csv = [
      ['Year', 'Base Cost', 'Seat Overage', 'API Overage', 'Export Cost', 'Annual Total', 'Cumulative Total'],
      ...calculations.map(c => [
        c.year,
        c.baseCost,
        c.seatOverage,
        c.apiOverage,
        c.exportCost,
        c.totalAnnual,
        c.cumulativeTotal,
      ]),
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `latticeforge-tco-${inputs.tier}-${inputs.projectionYears}yr.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <GlassCard blur="heavy" className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Calculator className="w-6 h-6 text-blue-400" />
          <div>
            <h2 className="text-xl font-bold text-white">Total Cost of Ownership Calculator</h2>
            <p className="text-sm text-slate-400">5-year projection with growth modeling</p>
          </div>
        </div>
        <GlassButton variant="secondary" onClick={handleExport}>
          <Download className="w-4 h-4 mr-2" />
          Export CSV
        </GlassButton>
      </div>

      {/* Input Section */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div>
          <label className="block text-sm text-slate-400 mb-1">Pricing Tier</label>
          <select
            value={inputs.tier}
            onChange={(e) => setInputs({ ...inputs, tier: e.target.value })}
            className="w-full px-3 py-2 bg-black/30 border border-white/[0.08] rounded-lg text-white focus:outline-none focus:border-blue-500/50"
          >
            <option value="entry">Entry ($299/mo)</option>
            <option value="team">Team ($1,499/mo)</option>
            <option value="enterprise">Enterprise ($4,999/mo)</option>
          </select>
        </div>

        <div>
          <label className="block text-sm text-slate-400 mb-1">Number of Analysts</label>
          <input
            type="number"
            min={1}
            max={1000}
            value={inputs.analysts}
            onChange={(e) => setInputs({ ...inputs, analysts: parseInt(e.target.value) || 1 })}
            className="w-full px-3 py-2 bg-black/30 border border-white/[0.08] rounded-lg text-white focus:outline-none focus:border-blue-500/50"
          />
        </div>

        <div>
          <label className="block text-sm text-slate-400 mb-1">Monthly API Calls</label>
          <input
            type="number"
            min={0}
            step={1000}
            value={inputs.monthlyApiCalls}
            onChange={(e) => setInputs({ ...inputs, monthlyApiCalls: parseInt(e.target.value) || 0 })}
            className="w-full px-3 py-2 bg-black/30 border border-white/[0.08] rounded-lg text-white focus:outline-none focus:border-blue-500/50"
          />
        </div>

        <div>
          <label className="block text-sm text-slate-400 mb-1">Data Export (GB/mo)</label>
          <input
            type="number"
            min={0}
            value={inputs.monthlyDataExportGB}
            onChange={(e) => setInputs({ ...inputs, monthlyDataExportGB: parseInt(e.target.value) || 0 })}
            className="w-full px-3 py-2 bg-black/30 border border-white/[0.08] rounded-lg text-white focus:outline-none focus:border-blue-500/50"
          />
          <p className="text-xs text-green-400 mt-1">Always $0 - No egress fees</p>
        </div>
      </div>

      {/* Advanced Options */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="flex items-center gap-2 text-sm text-slate-400 hover:text-white mb-4 transition-colors"
      >
        {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        Advanced Options
      </button>

      {showAdvanced && (
        <div className="grid grid-cols-2 gap-4 mb-6 p-4 bg-black/20 rounded-lg border border-white/[0.04]">
          <div>
            <label className="block text-sm text-slate-400 mb-1">Projection Period (Years)</label>
            <input
              type="number"
              min={1}
              max={10}
              value={inputs.projectionYears}
              onChange={(e) => setInputs({ ...inputs, projectionYears: parseInt(e.target.value) || 5 })}
              className="w-full px-3 py-2 bg-black/30 border border-white/[0.08] rounded-lg text-white focus:outline-none focus:border-blue-500/50"
            />
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-1">Annual Growth Rate (%)</label>
            <input
              type="number"
              min={0}
              max={100}
              value={inputs.annualGrowthRate}
              onChange={(e) => setInputs({ ...inputs, annualGrowthRate: parseInt(e.target.value) || 0 })}
              className="w-full px-3 py-2 bg-black/30 border border-white/[0.08] rounded-lg text-white focus:outline-none focus:border-blue-500/50"
            />
          </div>
        </div>
      )}

      {/* TCO Summary */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="p-4 bg-gradient-to-br from-blue-500/10 to-purple-500/10 rounded-xl border border-blue-500/20">
          <p className="text-sm text-slate-400 mb-1">{inputs.projectionYears}-Year Total Cost</p>
          <p className="text-3xl font-bold text-white">{formatCurrency(totalTCO)}</p>
        </div>
        <div className="p-4 bg-black/20 rounded-xl border border-white/[0.04]">
          <p className="text-sm text-slate-400 mb-1">Average Annual Cost</p>
          <p className="text-2xl font-bold text-white">{formatCurrency(totalTCO / inputs.projectionYears)}</p>
        </div>
        <div className="p-4 bg-black/20 rounded-xl border border-white/[0.04]">
          <p className="text-sm text-slate-400 mb-1">Cost Per Analyst/Year (Avg)</p>
          <p className="text-2xl font-bold text-white">
            {formatCurrency(totalTCO / inputs.projectionYears / inputs.analysts)}
          </p>
        </div>
      </div>

      {/* Year-by-Year Breakdown */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-left text-sm text-slate-400 border-b border-white/[0.08]">
              <th className="pb-3 font-medium">Year</th>
              <th className="pb-3 font-medium">Base</th>
              <th className="pb-3 font-medium">Seats</th>
              <th className="pb-3 font-medium">API</th>
              <th className="pb-3 font-medium">Export</th>
              <th className="pb-3 font-medium">Annual</th>
              <th className="pb-3 font-medium">Cumulative</th>
            </tr>
          </thead>
          <tbody>
            {calculations.map((calc) => (
              <tr key={calc.year} className="text-sm border-b border-white/[0.04]">
                <td className="py-3 text-white font-medium">Year {calc.year}</td>
                <td className="py-3 text-slate-300">{formatCurrency(calc.baseCost)}</td>
                <td className="py-3 text-slate-300">
                  {calc.seatOverage > 0 ? formatCurrency(calc.seatOverage) : '-'}
                </td>
                <td className="py-3 text-slate-300">
                  {calc.apiOverage > 0 ? formatCurrency(calc.apiOverage) : '-'}
                </td>
                <td className="py-3 text-green-400">$0</td>
                <td className="py-3 text-white font-medium">{formatCurrency(calc.totalAnnual)}</td>
                <td className="py-3 text-blue-400 font-medium">{formatCurrency(calc.cumulativeTotal)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Anti-Lock-In Guarantees */}
      <div className="mt-6 p-4 bg-green-500/5 rounded-xl border border-green-500/20">
        <div className="flex items-start gap-3">
          <TrendingUp className="w-5 h-5 text-green-400 mt-0.5" />
          <div>
            <h3 className="font-medium text-green-400 mb-2">Anti-Lock-In Guarantees</h3>
            <ul className="text-sm text-slate-300 space-y-1">
              <li>Data export: $0 (JSON, CSV, STIX2.1, MISP formats)</li>
              <li>Maximum commitment: 12 months</li>
              <li>Early termination penalty: 0%</li>
              <li>Post-cancellation data access: 90 days read-only</li>
            </ul>
          </div>
        </div>
      </div>
    </GlassCard>
  );
}
