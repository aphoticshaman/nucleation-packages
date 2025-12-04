'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import {
  NeuralTicker,
  mockTickerItems,
  PulseIndicator,
  PulseIndicatorLarge,
  SignalFeed,
  mockSignals,
  LogicTree,
  mockLogicTreeData,
  CommandPalette,
  useCommandPalette,
} from '@/components/intelligence';
import { GlassCard } from '@/components/ui/GlassCard';

export default function IntelligenceDashboard() {
  const router = useRouter();
  const { isOpen, open, close } = useCommandPalette();

  // Simulated real-time data
  const [lastUpdate, setLastUpdate] = useState<number>(Date.now());
  const [tickerItems, setTickerItems] = useState(mockTickerItems);
  const [signals, setSignals] = useState(mockSignals);
  const [selectedView, setSelectedView] = useState<'feed' | 'logic' | 'map'>('feed');

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setLastUpdate(Date.now());
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  // Handle entity clicks - pivot to dossier
  const handleEntityClick = useCallback((entity: { id: string; text: string; type: string }) => {
    console.log('Entity clicked:', entity);
    // Would navigate to entity dossier
  }, []);

  // Handle signal clicks - show detail
  const handleSignalClick = useCallback((signal: typeof mockSignals[0]) => {
    console.log('Signal clicked:', signal);
    // Would open signal detail panel
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Command Palette */}
      <CommandPalette isOpen={isOpen} onClose={close} />

      {/* Neural Ticker - Component 01 */}
      <NeuralTicker
        items={tickerItems}
        speed={40}
        onItemClick={(item) => console.log('Ticker item:', item)}
      />

      {/* Main Layout */}
      <div className="flex flex-col md:flex-row h-[calc(100dvh-48px)]">
        {/* Desktop Sidebar - hidden on mobile */}
        <aside className="hidden md:flex w-16 bg-[rgba(18,18,26,0.8)] backdrop-blur-xl border-r border-white/[0.06] flex-col items-center py-4 gap-2">
          {/* Logo */}
          <div className="w-11 h-11 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-white font-bold mb-4">
            LF
          </div>

          {/* Nav items */}
          <NavButton
            icon="📡"
            label="Signals"
            isActive={selectedView === 'feed'}
            onClick={() => setSelectedView('feed')}
          />
          <NavButton
            icon="🧠"
            label="Logic"
            isActive={selectedView === 'logic'}
            onClick={() => setSelectedView('logic')}
          />
          <NavButton
            icon="🗺️"
            label="Map"
            isActive={selectedView === 'map'}
            onClick={() => setSelectedView('map')}
          />

          <div className="flex-1" />

          {/* Command palette trigger */}
          <button
            onClick={open}
            className="w-11 h-11 rounded-lg bg-slate-800 hover:bg-slate-700 flex items-center justify-center text-slate-400 hover:text-white transition-colors"
            title="Command Palette (⌘K)"
          >
            ⌘
          </button>

          {/* Settings */}
          <button className="w-11 h-11 rounded-lg bg-slate-800 hover:bg-slate-700 flex items-center justify-center text-slate-400 hover:text-white transition-colors">
            ⚙️
          </button>
        </aside>

        {/* Main Content */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Top Bar */}
          <header className="h-14 px-3 md:px-6 bg-[rgba(18,18,26,0.6)] backdrop-blur-xl border-b border-white/[0.06] flex items-center justify-between">
            <div className="flex items-center gap-2 md:gap-4">
              <h1 className="text-base md:text-lg font-semibold truncate">
                {selectedView === 'feed' && 'Signal Feed'}
                {selectedView === 'logic' && 'Logic Inspector'}
                {selectedView === 'map' && 'Risk Map'}
              </h1>
              <PulseIndicator
                lastMessageTimestamp={lastUpdate}
                showLatency
              />
            </div>

            <div className="flex items-center gap-2 md:gap-3">
              {/* Filter chips - hidden on mobile */}
              <div className="hidden md:flex gap-2">
                <FilterChip label="All Domains" isActive />
                <FilterChip label="Critical Only" />
                <FilterChip label="Last 1h" />
              </div>

              {/* Search - touch-friendly on mobile */}
              <button
                onClick={open}
                className="hidden md:flex items-center gap-2 px-3 py-2 bg-slate-800 rounded-lg text-sm text-slate-400 hover:text-white transition-colors"
              >
                <span>Search...</span>
                <kbd className="px-1.5 py-0.5 bg-slate-700 rounded text-xs">⌘K</kbd>
              </button>

              {/* Mobile filter button */}
              <button className="md:hidden min-h-[44px] min-w-[44px] flex items-center justify-center bg-slate-800 rounded-lg text-slate-400 active:bg-slate-700">
                <span className="text-lg">☰</span>
              </button>
            </div>
          </header>

          {/* Content Area - extra bottom padding on mobile for nav */}
          <div className="flex-1 overflow-hidden p-3 md:p-6 pb-20 md:pb-6">
            {selectedView === 'feed' && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
                {/* Signal Feed - 2 cols */}
                <div className="lg:col-span-2 bg-[rgba(18,18,26,0.6)] backdrop-blur-xl rounded-xl border border-white/[0.06] overflow-hidden">
                  <div className="p-4 border-b border-white/[0.06] flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <h2 className="font-medium">Live Signals</h2>
                      <span className="text-xs text-slate-500">
                        {signals.length} items
                      </span>
                    </div>
                    <PulseIndicatorLarge
                      lastMessageTimestamp={lastUpdate}
                      datasetName="OSINT Stream"
                    />
                  </div>
                  <SignalFeed
                    signals={signals}
                    onEntityClick={handleEntityClick}
                    onSignalClick={handleSignalClick}
                  />
                </div>

                {/* Stats Panel - 1 col */}
                <div className="space-y-4">
                  {/* Domain breakdown */}
                  <div className="bg-[rgba(18,18,26,0.6)] backdrop-blur-xl rounded-xl border border-white/[0.06] p-4">
                    <h3 className="text-sm font-medium text-slate-400 mb-3">Domain Activity</h3>
                    <div className="space-y-2">
                      <DomainBar domain="Cyber" count={23} total={100} color="bg-red-500" />
                      <DomainBar domain="Financial" count={18} total={100} color="bg-green-500" />
                      <DomainBar domain="Defense" count={15} total={100} color="bg-blue-500" />
                      <DomainBar domain="Geopolitical" count={12} total={100} color="bg-amber-500" />
                      <DomainBar domain="Energy" count={8} total={100} color="bg-purple-500" />
                    </div>
                  </div>

                  {/* Risk summary */}
                  <div className="bg-[rgba(18,18,26,0.6)] backdrop-blur-xl rounded-xl border border-white/[0.06] p-4">
                    <h3 className="text-sm font-medium text-slate-400 mb-3">Risk Summary</h3>
                    <div className="grid grid-cols-2 gap-3">
                      <RiskCard level="CRITICAL" count={2} />
                      <RiskCard level="HIGH" count={7} />
                      <RiskCard level="ELEVATED" count={15} />
                      <RiskCard level="LOW" count={34} />
                    </div>
                  </div>

                  {/* Active cascades */}
                  <div className="bg-[rgba(18,18,26,0.6)] backdrop-blur-xl rounded-xl border border-white/[0.06] p-4">
                    <h3 className="text-sm font-medium text-slate-400 mb-3">Active Cascades</h3>
                    <div className="space-y-2">
                      <CascadeItem
                        source="Energy"
                        target="Financial"
                        strength={0.85}
                        lag="24h"
                      />
                      <CascadeItem
                        source="Cyber"
                        target="Healthcare"
                        strength={0.72}
                        lag="4h"
                      />
                      <CascadeItem
                        source="Geopolitical"
                        target="Defense"
                        strength={0.68}
                        lag="12h"
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {selectedView === 'logic' && (
              <div className="h-full">
                <LogicTree
                  data={mockLogicTreeData}
                  onNodeClick={(node) => console.log('Node clicked:', node)}
                  showConfidence
                />
              </div>
            )}

            {selectedView === 'map' && (
              <div className="h-full bg-[rgba(18,18,26,0.6)] backdrop-blur-xl rounded-xl border border-white/[0.06] flex items-center justify-center">
                <div className="text-center text-slate-500">
                  <span className="text-4xl mb-4 block">🗺️</span>
                  <p>Hexbin Risk Map</p>
                  <p className="text-sm">Component 11 - Coming next</p>
                </div>
              </div>
            )}
          </div>
        </main>

        {/* Mobile Bottom Navigation - 44px+ touch targets */}
        <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-[rgba(18,18,26,0.95)] backdrop-blur-xl border-t border-white/[0.06] px-2 pb-safe">
          <div className="flex items-center justify-around h-16">
            <MobileNavButton
              icon="📡"
              label="Signals"
              isActive={selectedView === 'feed'}
              onClick={() => setSelectedView('feed')}
            />
            <MobileNavButton
              icon="🧠"
              label="Logic"
              isActive={selectedView === 'logic'}
              onClick={() => setSelectedView('logic')}
            />
            <MobileNavButton
              icon="🗺️"
              label="Map"
              isActive={selectedView === 'map'}
              onClick={() => setSelectedView('map')}
            />
            <MobileNavButton
              icon="⌘"
              label="Search"
              isActive={false}
              onClick={open}
            />
          </div>
        </nav>
      </div>
    </div>
  );
}

// Helper Components
function NavButton({
  icon,
  label,
  isActive,
  onClick,
}: {
  icon: string;
  label: string;
  isActive: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`
        w-11 h-11 rounded-lg flex items-center justify-center transition-all
        ${isActive
          ? 'bg-cyan-500/20 text-cyan-400 shadow-[0_0_10px_rgba(6,182,212,0.3)] border border-cyan-500/30'
          : 'text-slate-400 hover:bg-black/30 hover:text-white border border-transparent hover:border-white/[0.06]'
        }
      `}
      title={label}
    >
      <span className="text-lg">{icon}</span>
    </button>
  );
}

function MobileNavButton({
  icon,
  label,
  isActive,
  onClick,
}: {
  icon: string;
  label: string;
  isActive: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`
        min-w-[56px] h-12 px-3 rounded-lg flex flex-col items-center justify-center gap-0.5 transition-all
        ${isActive
          ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
          : 'text-slate-400 active:bg-black/30 active:text-white border border-transparent'
        }
      `}
    >
      <span className="text-lg">{icon}</span>
      <span className="text-[10px] font-medium">{label}</span>
    </button>
  );
}

function FilterChip({ label, isActive = false }: { label: string; isActive?: boolean }) {
  return (
    <button
      className={`
        px-3 py-1 rounded-full text-xs font-medium transition-colors
        ${isActive
          ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50'
          : 'bg-black/30 text-slate-400 border border-white/[0.06] hover:border-white/[0.12]'
        }
      `}
    >
      {label}
    </button>
  );
}

function DomainBar({
  domain,
  count,
  total,
  color,
}: {
  domain: string;
  count: number;
  total: number;
  color: string;
}) {
  const pct = (count / total) * 100;
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-slate-400 w-20 truncate">{domain}</span>
      <div className="flex-1 h-2 bg-black/30 rounded-full overflow-hidden border border-white/[0.04]">
        <div
          className={`h-full ${color} transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs text-slate-500 w-8 text-right">{count}</span>
    </div>
  );
}

function RiskCard({ level, count }: { level: string; count: number }) {
  const colors: Record<string, string> = {
    CRITICAL: 'bg-red-500/20 border-red-500/50 text-red-400',
    HIGH: 'bg-orange-500/20 border-orange-500/50 text-orange-400',
    ELEVATED: 'bg-yellow-500/20 border-yellow-500/50 text-yellow-400',
    LOW: 'bg-green-500/20 border-green-500/50 text-green-400',
  };
  return (
    <div className={`p-3 rounded-lg border ${colors[level]}`}>
      <div className="text-2xl font-bold">{count}</div>
      <div className="text-xs opacity-80">{level}</div>
    </div>
  );
}

function CascadeItem({
  source,
  target,
  strength,
  lag,
}: {
  source: string;
  target: string;
  strength: number;
  lag: string;
}) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-amber-400">{source}</span>
      <span className="text-slate-600">→</span>
      <span className="text-cyan-400">{target}</span>
      <div className="flex-1" />
      <span className="text-slate-500">+{lag}</span>
      <div
        className="w-12 h-1.5 bg-black/30 rounded-full overflow-hidden border border-white/[0.04]"
        title={`${(strength * 100).toFixed(0)}% correlation`}
      >
        <div
          className="h-full bg-gradient-to-r from-amber-500 to-cyan-500"
          style={{ width: `${strength * 100}%` }}
        />
      </div>
    </div>
  );
}
