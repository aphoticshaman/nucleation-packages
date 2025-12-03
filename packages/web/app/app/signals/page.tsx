'use client';

import { useState } from 'react';
import { ALL_FINANCIAL_SOURCES, MARKET_INDICATORS, type FinancialSource } from '@/lib/signals/financialSources';
import { TrendingUp, Fuel, DollarSign, Bitcoin, Building, Globe, Bell, Plus, Settings, BookOpen } from 'lucide-react';
import Glossary from '@/components/Glossary';
import HelpTip from '@/components/HelpTip';

const SIGNAL_CATEGORIES = [
  { id: 'equities', name: 'Equities', icon: TrendingUp, desc: 'Stock market indices' },
  { id: 'commodities', name: 'Commodities', icon: Fuel, desc: 'Oil, gold, minerals' },
  { id: 'forex', name: 'Forex', icon: DollarSign, desc: 'Currency pairs' },
  { id: 'crypto', name: 'Crypto', icon: Bitcoin, desc: 'Digital assets' },
  { id: 'economic', name: 'Economic', icon: Building, desc: 'GDP, inflation, rates' },
  { id: 'geopolitical', name: 'Geopolitical', icon: Globe, desc: 'Risk indicators' },
];

export default function SignalsPage() {
  const [selectedCategory, setSelectedCategory] = useState('equities');
  const [activeAPIs, setActiveAPIs] = useState<string[]>(['yahoo_finance', 'fred']);
  const [showGlossary, setShowGlossary] = useState(false);

  const toggleAPI = (apiId: string) => {
    setActiveAPIs(prev =>
      prev.includes(apiId)
        ? prev.filter(id => id !== apiId)
        : [...prev, apiId]
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Market Signals</h1>
          <p className="text-slate-400 mt-1">Real-time financial data from 12+ sources</p>
        </div>
        <div className="flex items-center gap-4">
          <button
            onClick={() => setShowGlossary(true)}
            className="flex items-center gap-2 px-3 py-2 min-h-[44px] bg-[rgba(18,18,26,0.7)] backdrop-blur-sm rounded-xl border border-white/[0.06] text-slate-400 hover:text-white hover:border-white/[0.12] transition-all"
          >
            <BookOpen className="w-4 h-4" />
            <span className="text-sm">Terms</span>
          </button>
          <div className="text-right">
            <div className="flex items-center gap-2 text-green-400">
              <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              <span className="text-sm font-medium">Live</span>
            </div>
            <p className="text-xs text-slate-500 mt-1">{activeAPIs.length} sources active</p>
          </div>
        </div>
      </div>

      {/* Category tabs */}
      <div className="flex gap-2 overflow-x-auto pb-2 -mx-1 px-1">
        {SIGNAL_CATEGORIES.map((cat) => {
          const Icon = cat.icon;
          return (
            <button
              key={cat.id}
              onClick={() => setSelectedCategory(cat.id)}
              className={`flex items-center gap-2 px-4 py-2.5 min-h-[44px] rounded-xl whitespace-nowrap transition-all ${
                selectedCategory === cat.id
                  ? 'bg-gradient-to-r from-blue-600 to-cyan-500 text-white shadow-lg shadow-blue-500/25'
                  : 'bg-[rgba(18,18,26,0.7)] backdrop-blur-sm text-slate-400 hover:text-white border border-white/[0.06] hover:border-white/[0.12]'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span className="text-sm font-medium">{cat.name}</span>
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Data Sources Panel */}
        <div className="bg-[rgba(18,18,26,0.7)] backdrop-blur-xl rounded-xl border border-white/[0.06] p-4">
          <h2 className="text-lg font-semibold text-white mb-4">Data Sources</h2>
          <div className="space-y-2">
            {ALL_FINANCIAL_SOURCES.map((api: FinancialSource) => (
              <button
                key={api.id}
                onClick={() => toggleAPI(api.id)}
                className={`w-full flex items-center justify-between p-3 rounded-xl transition-all min-h-[60px] ${
                  activeAPIs.includes(api.id)
                    ? 'bg-green-500/10 border border-green-500/30'
                    : 'bg-black/20 border border-white/[0.04] opacity-60 hover:opacity-80'
                }`}
              >
                <div className="text-left">
                  <p className="text-sm font-medium text-white">{api.name}</p>
                  <p className="text-xs text-slate-400">{api.dataTypes.slice(0, 2).join(', ')}</p>
                </div>
                <div className={`w-3 h-3 rounded-full ${activeAPIs.includes(api.id) ? 'bg-green-500' : 'bg-slate-600'}`} />
              </button>
            ))}
          </div>
        </div>

        {/* Main signals view */}
        <div className="lg:col-span-2 space-y-4">
          {/* Quick indicators */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {MARKET_INDICATORS.us_indices.slice(0, 4).map((indicator) => (
              <div key={indicator.symbol} className="bg-[rgba(18,18,26,0.7)] backdrop-blur-sm rounded-xl border border-white/[0.06] p-4">
                <p className="text-xs text-slate-400">{indicator.name}</p>
                <p className="text-xl font-bold text-white mt-1">--</p>
                <p className="text-xs text-slate-500 mt-1">{indicator.symbol}</p>
              </div>
            ))}
          </div>

          {/* Signal Feed */}
          <div className="bg-[rgba(18,18,26,0.7)] backdrop-blur-xl rounded-xl border border-white/[0.06] p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Signal Feed</h3>
              <select className="bg-black/30 border border-white/[0.08] rounded-xl px-3 py-2 text-sm text-white min-h-[44px]">
                <option>Last 24 hours</option>
                <option>Last 7 days</option>
                <option>Last 30 days</option>
              </select>
            </div>

            <div className="h-64 flex items-center justify-center border border-dashed border-white/[0.08] rounded-xl bg-black/20">
              <div className="text-center">
                <Settings className="w-10 h-10 text-slate-500 mx-auto mb-3" />
                <p className="text-slate-400 mb-3">Connect API keys to see live data</p>
                <button className="px-4 py-2.5 min-h-[44px] bg-gradient-to-r from-blue-600 to-cyan-500 text-white rounded-xl text-sm font-medium
                  hover:shadow-[0_0_20px_rgba(59,130,246,0.3)] active:scale-[0.98] transition-all">
                  Configure APIs
                </button>
              </div>
            </div>
          </div>

          {/* Alert rules */}
          <div className="bg-[rgba(18,18,26,0.7)] backdrop-blur-xl rounded-xl border border-white/[0.06] p-4">
            <div className="flex items-center gap-2 mb-3">
              <Bell className="w-5 h-5 text-amber-400" />
              <h3 className="text-lg font-semibold text-white">Alert Rules</h3>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between p-3 bg-black/20 rounded-xl border border-white/[0.04]">
                <div>
                  <p className="text-sm text-white">VIX &gt; 30</p>
                  <p className="text-xs text-slate-400">Volatility spike alert</p>
                </div>
                <span className="text-xs text-amber-400 font-medium px-2 py-1 bg-amber-500/10 rounded-lg">Armed</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-black/20 rounded-xl border border-white/[0.04]">
                <div>
                  <p className="text-sm text-white">DXY ±2% daily</p>
                  <p className="text-xs text-slate-400">Dollar movement</p>
                </div>
                <span className="text-xs text-green-400 font-medium px-2 py-1 bg-green-500/10 rounded-lg">Active</span>
              </div>
              <button className="w-full p-3 min-h-[52px] border border-dashed border-white/[0.08] rounded-xl text-slate-400 hover:text-white hover:border-white/[0.15] transition-all flex items-center justify-center gap-2">
                <Plus className="w-4 h-4" />
                Add Alert Rule
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Glossary Modal */}
      <Glossary
        isOpen={showGlossary}
        onClose={() => setShowGlossary(false)}
        skillLevel="standard"
      />
    </div>
  );
}
