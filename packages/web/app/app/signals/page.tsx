'use client';

import { useState } from 'react';
import { ALL_FINANCIAL_SOURCES, MARKET_INDICATORS, type FinancialSource } from '@/lib/signals/financialSources';

const SIGNAL_CATEGORIES = [
  { id: 'equities', name: 'Equities', icon: '📈', desc: 'Stock market indices' },
  { id: 'commodities', name: 'Commodities', icon: '🛢️', desc: 'Oil, gold, minerals' },
  { id: 'forex', name: 'Forex', icon: '💱', desc: 'Currency pairs' },
  { id: 'crypto', name: 'Crypto', icon: '₿', desc: 'Digital assets' },
  { id: 'economic', name: 'Economic', icon: '🏛️', desc: 'GDP, inflation, rates' },
  { id: 'geopolitical', name: 'Geopolitical', icon: '🌍', desc: 'Risk indicators' },
];

export default function SignalsPage() {
  const [selectedCategory, setSelectedCategory] = useState('equities');
  const [activeAPIs, setActiveAPIs] = useState<string[]>(['yahoo_finance', 'fred']);

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
        <div className="text-right">
          <p className="text-sm text-green-400">● Live</p>
          <p className="text-xs text-slate-500">{activeAPIs.length} sources active</p>
        </div>
      </div>

      {/* Category tabs */}
      <div className="flex gap-2 overflow-x-auto pb-2">
        {SIGNAL_CATEGORIES.map((cat) => (
          <button
            key={cat.id}
            onClick={() => setSelectedCategory(cat.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg whitespace-nowrap transition-colors ${
              selectedCategory === cat.id
                ? 'bg-blue-600 text-white'
                : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
            }`}
          >
            <span>{cat.icon}</span>
            <span className="text-sm">{cat.name}</span>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Data Sources Panel */}
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
          <h2 className="text-lg font-semibold text-white mb-4">Data Sources</h2>
          <div className="space-y-2">
            {ALL_FINANCIAL_SOURCES.map((api: FinancialSource) => (
              <button
                key={api.id}
                onClick={() => toggleAPI(api.id)}
                className={`w-full flex items-center justify-between p-3 rounded-lg transition-colors ${
                  activeAPIs.includes(api.id)
                    ? 'bg-green-900/30 border border-green-700'
                    : 'bg-slate-800 border border-slate-700 opacity-60'
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
              <div key={indicator.symbol} className="bg-slate-900 rounded-xl border border-slate-800 p-4">
                <p className="text-xs text-slate-400">{indicator.name}</p>
                <p className="text-xl font-bold text-white mt-1">--</p>
                <p className="text-xs text-slate-500 mt-1">{indicator.symbol}</p>
              </div>
            ))}
          </div>

          {/* Placeholder for charts */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Signal Feed</h3>
              <select className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-white">
                <option>Last 24 hours</option>
                <option>Last 7 days</option>
                <option>Last 30 days</option>
              </select>
            </div>

            <div className="h-64 flex items-center justify-center border border-dashed border-slate-700 rounded-lg">
              <div className="text-center">
                <p className="text-slate-400">Connect API keys to see live data</p>
                <button className="mt-3 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm">
                  Configure APIs
                </button>
              </div>
            </div>
          </div>

          {/* Alert rules */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
            <h3 className="text-lg font-semibold text-white mb-3">Alert Rules</h3>
            <div className="space-y-2">
              <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                <div>
                  <p className="text-sm text-white">VIX &gt; 30</p>
                  <p className="text-xs text-slate-400">Volatility spike alert</p>
                </div>
                <span className="text-xs text-yellow-400">Armed</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                <div>
                  <p className="text-sm text-white">DXY ±2% daily</p>
                  <p className="text-xs text-slate-400">Dollar movement</p>
                </div>
                <span className="text-xs text-green-400">Active</span>
              </div>
              <button className="w-full p-3 border border-dashed border-slate-700 rounded-lg text-slate-400 hover:text-white hover:border-slate-600 transition-colors">
                + Add Alert Rule
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
