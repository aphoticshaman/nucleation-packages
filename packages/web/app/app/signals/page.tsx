'use client';

import { useState, useEffect } from 'react';
import { ALL_FINANCIAL_SOURCES, MARKET_INDICATORS, type FinancialSource } from '@/lib/signals/financialSources';
import { TrendingUp, Fuel, DollarSign, Bitcoin, Building, Globe, Bell, Plus, BookOpen, AlertTriangle, Newspaper, Activity } from 'lucide-react';
import Glossary from '@/components/Glossary';
import { Card, Button } from '@/components/ui';

import { DataFreshness } from '@/components/DataFreshness';

const SIGNAL_CATEGORIES = [
  { id: 'live', name: 'Live Feed', icon: Activity, desc: 'Real-time ingested signals' },
  { id: 'equities', name: 'Equities', icon: TrendingUp, desc: 'Stock market indices' },
  { id: 'commodities', name: 'Commodities', icon: Fuel, desc: 'Oil, gold, minerals' },
  { id: 'forex', name: 'Forex', icon: DollarSign, desc: 'Currency pairs' },
  { id: 'crypto', name: 'Crypto', icon: Bitcoin, desc: 'Digital assets' },
  { id: 'economic', name: 'Economic', icon: Building, desc: 'GDP, inflation, rates' },
  { id: 'geopolitical', name: 'Geopolitical', icon: Globe, desc: 'Risk indicators' },
];

interface Signal {
  timestamp: string;
  source: string;
  domain: string;
  title?: string;
  summary?: string;
  numeric_features?: Record<string, number>;
}

export default function SignalsPage() {
  const [selectedCategory, setSelectedCategory] = useState('live');
  const [activeAPIs, setActiveAPIs] = useState<string[]>(['yahoo_finance', 'fred']);
  const [showGlossary, setShowGlossary] = useState(false);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loading, setLoading] = useState(false);
  const [signalSource, setSignalSource] = useState<'all' | 'gdelt' | 'usgs' | 'sentiment'>('all');

  useEffect(() => {
    if (selectedCategory === 'live') {
      fetchSignals();
    }
  }, [selectedCategory, signalSource]);

  async function fetchSignals() {
    setLoading(true);
    try {
      const res = await fetch(`/api/query/signals?source=${signalSource}&limit=50`);
      const data = await res.json();
      setSignals(data.signals || []);
    } catch (e) {
      console.error('Failed to fetch signals:', e);
    } finally {
      setLoading(false);
    }
  }

  const toggleAPI = (apiId: string) => {
    setActiveAPIs(prev =>
      prev.includes(apiId)
        ? prev.filter(id => id !== apiId)
        : [...prev, apiId]
    );
  };

  const formatTime = (ts: string) => {
    const date = new Date(ts);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString();
  };

  const getSourceIcon = (source: string) => {
    switch (source) {
      case 'gdelt': return <Newspaper className="w-4 h-4" />;
      case 'usgs': return <AlertTriangle className="w-4 h-4" />;
      case 'sentiment': return <TrendingUp className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  const getSourceColor = (source: string) => {
    switch (source) {
      case 'gdelt': return 'text-blue-400 bg-blue-500/10';
      case 'usgs': return 'text-amber-400 bg-amber-500/10';
      case 'sentiment': return 'text-green-400 bg-green-500/10';
      default: return 'text-slate-400 bg-slate-500/10';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-lg font-bold text-white">Market Signals</h1>
          <p className="text-slate-400 mt-1">Real-time data from GDELT, USGS, and market feeds</p>
        </div>
        <div className="flex items-center gap-4">
          <button
            onClick={() => setShowGlossary(true)}
            className="flex items-center gap-2 px-3 py-2 min-h-[44px] bg-[rgba(18,18,26,0.7)] backdrop-blur-sm rounded-xl border border-white/[0.06] text-slate-400 hover:text-white hover:border-white/[0.12] transition-all"
          >
            <BookOpen className="w-4 h-4" />
            <span className="text-sm">Terms</span>
          </button>
          <DataFreshness />
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

      {selectedCategory === 'live' ? (
        <div className="space-y-4">
          {/* Data Freshness */}
          <Card>
            <DataFreshness />
          </Card>

          {/* Source Filter */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-slate-400">Source:</span>
            {(['all', 'gdelt', 'usgs', 'sentiment'] as const).map((src) => (
              <button
                key={src}
                onClick={() => setSignalSource(src)}
                className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                  signalSource === src
                    ? 'bg-white/10 text-white'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {src === 'all' ? 'All' : src.toUpperCase()}
              </button>
            ))}
            <Button variant="secondary" size="sm" onClick={fetchSignals} className="ml-auto">
              Refresh
            </Button>
          </div>

          {/* Live Signal Feed */}
          <Card>
            <h3 className="text-lg font-semibold text-white mb-4">Signal Feed</h3>

            {loading ? (
              <div className="h-64 flex items-center justify-center">
                <div className="text-slate-400">Loading signals...</div>
              </div>
            ) : signals.length === 0 ? (
              <div className="h-64 flex items-center justify-center border border-dashed border-white/[0.08] rounded-xl bg-black/20">
                <div className="text-center">
                  <Activity className="w-10 h-10 text-slate-500 mx-auto mb-3" />
                  <p className="text-slate-400">No signals yet</p>
                  <p className="text-slate-500 text-sm">Data is ingested hourly via cron jobs</p>
                </div>
              </div>
            ) : (
              <div className="space-y-2 max-h-[500px] overflow-y-auto">
                {signals.map((signal, i) => (
                  <div
                    key={`${signal.timestamp}-${i}`}
                    className="p-3 bg-black/20 rounded-xl border border-white/[0.04] hover:border-white/[0.08] transition-colors"
                  >
                    <div className="flex items-start gap-3">
                      <div className={`p-2 rounded-lg ${getSourceColor(signal.source)}`}>
                        {getSourceIcon(signal.source)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs text-slate-500 uppercase">{signal.source}</span>
                          {signal.domain && signal.domain !== 'global' && (
                            <span className="px-2 py-0.5 bg-slate-700 text-slate-300 text-xs rounded">
                              {signal.domain}
                            </span>
                          )}
                          <span className="text-xs text-slate-500 ml-auto">{formatTime(signal.timestamp)}</span>
                        </div>
                        {signal.title && (
                          <p className="text-sm text-white truncate">{signal.title}</p>
                        )}
                        {signal.summary && (
                          <p className="text-sm text-slate-400 line-clamp-2">{signal.summary}</p>
                        )}
                        {signal.numeric_features && (
                          <div className="flex flex-wrap gap-2 mt-2">
                            {Object.entries(signal.numeric_features).slice(0, 4).map(([key, value]) => (
                              <span key={key} className="text-xs text-slate-500">
                                {key.replace(/_/g, ' ')}: <span className="text-slate-300">{typeof value === 'number' ? value.toFixed(2) : value}</span>
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Data Sources Panel */}
          <Card>
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
          </Card>

          {/* Main signals view */}
          <div className="lg:col-span-2 space-y-4">
            {/* Quick indicators */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {MARKET_INDICATORS.us_indices.slice(0, 4).map((indicator) => (
                <Card key={indicator.symbol}>
                  <p className="text-xs text-slate-400">{indicator.name}</p>
                  <p className="text-xl font-bold text-white mt-1">--</p>
                  <p className="text-xs text-slate-500 mt-1">{indicator.symbol}</p>
                </Card>
              ))}
            </div>

            {/* Alert rules */}
            <Card>
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
            </Card>
          </div>
        </div>
      )}

      {/* Glossary Modal */}
      <Glossary
        isOpen={showGlossary}
        onClose={() => setShowGlossary(false)}
        skillLevel="standard"
      />
    </div>
  );
}
