'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { tw, getRiskLabel, getSentimentColor, admiraltyCode } from '@/lib/design-system';

// Entity types with their visual styling
const ENTITY_TYPES = {
  PERSON: { icon: '👤', color: 'bg-blue-500/20 border-blue-500/50 text-blue-300' },
  ORG: { icon: '🏢', color: 'bg-purple-500/20 border-purple-500/50 text-purple-300' },
  LOC: { icon: '📍', color: 'bg-orange-500/20 border-orange-500/50 text-orange-300' },
  WEAPON: { icon: '⚔️', color: 'bg-red-500/20 border-red-500/50 text-red-300' },
  MONEY: { icon: '💰', color: 'bg-green-500/20 border-green-500/50 text-green-300' },
  EVENT: { icon: '📅', color: 'bg-cyan-500/20 border-cyan-500/50 text-cyan-300' },
  TECH: { icon: '🔧', color: 'bg-slate-500/20 border-slate-500/50 text-slate-300' },
} as const;

interface Entity {
  id: string;
  text: string;
  type: keyof typeof ENTITY_TYPES;
  confidence: number;
  linkedId?: string; // Link to knowledge graph
}

interface SignalItem {
  id: string;
  content: string;
  domain: string;
  timestamp: string;
  source: {
    name: string;
    reliability: 'A' | 'B' | 'C' | 'D' | 'E' | 'F';
    credibility: 1 | 2 | 3 | 4 | 5 | 6;
  };
  sentiment: number;
  riskScore: number;
  entities: Entity[];
  isNeural: boolean; // Neural vs Symbolic source
  duplicateCount?: number;
  isPinned?: boolean;
}

interface SignalFeedProps {
  signals: SignalItem[];
  onEntityClick?: (entity: Entity) => void;
  onSignalClick?: (signal: SignalItem) => void;
  lastReadId?: string;
  isLoading?: boolean;
}

// Component 07: Source Reliability Badge
function SourceBadge({
  reliability,
  credibility,
}: {
  reliability: 'A' | 'B' | 'C' | 'D' | 'E' | 'F';
  credibility: 1 | 2 | 3 | 4 | 5 | 6;
}) {
  const grade = `${reliability}${credibility}`;

  // Color based on reliability
  const reliabilityColors: Record<string, string> = {
    A: 'bg-green-500/20 border-green-500/50 text-green-300',
    B: 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300',
    C: 'bg-yellow-500/20 border-yellow-500/50 text-yellow-300',
    D: 'bg-orange-500/20 border-orange-500/50 text-orange-300',
    E: 'bg-red-500/20 border-red-500/50 text-red-300',
    F: 'bg-slate-500/20 border-slate-500/50 text-slate-300',
  };

  return (
    <div className="group relative">
      <div className={`
        inline-flex items-center gap-1 px-2 py-0.5 rounded border text-xs font-mono font-bold
        ${reliabilityColors[reliability]}
      `}>
        <span className="text-[10px]">🛡️</span>
        {grade}
      </div>

      {/* Tooltip */}
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-slate-800 rounded-lg shadow-xl border border-slate-700 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50 w-48">
        <div className="text-xs">
          <div className="font-bold text-slate-200 mb-1">NATO Admiralty Code</div>
          <div className="text-slate-400">
            <span className="text-slate-300">{reliability}:</span> {admiraltyCode.reliability[reliability]}
          </div>
          <div className="text-slate-400">
            <span className="text-slate-300">{credibility}:</span> {admiraltyCode.credibility[credibility]}
          </div>
        </div>
        <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-800" />
      </div>
    </div>
  );
}

// Component 06: Entity Extraction Chips
function EntityChip({
  entity,
  onClick,
}: {
  entity: Entity;
  onClick?: () => void;
}) {
  const config = ENTITY_TYPES[entity.type];

  return (
    <button
      onClick={onClick}
      className={`
        inline-flex items-center gap-1.5 px-2.5 py-1.5 min-h-[32px] rounded-full border text-xs
        transition-all hover:scale-105 active:scale-95 cursor-pointer
        ${config.color}
      `}
    >
      <span className="text-sm">{config.icon}</span>
      <span className="font-medium truncate max-w-[120px] sm:max-w-[140px]">{entity.text}</span>
      {entity.confidence < 0.8 && (
        <span className="text-[10px] opacity-50">?</span>
      )}
    </button>
  );
}

// Single signal card
function SignalCard({
  signal,
  onEntityClick,
  onClick,
  isNew,
}: {
  signal: SignalItem;
  onEntityClick?: (entity: Entity) => void;
  onClick?: () => void;
  isNew?: boolean;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const risk = getRiskLabel(signal.riskScore);
  const sentimentColor = getSentimentColor(signal.sentiment);

  return (
    <div
      onClick={onClick}
      className={`
        relative p-4 rounded-lg border transition-all duration-200 cursor-pointer
        ${signal.isPinned
          ? 'bg-amber-500/10 border-amber-500/30'
          : 'bg-slate-800/50 border-slate-700/50 hover:bg-slate-800/80'
        }
        ${isNew ? 'ring-2 ring-cyan-500/50 animate-pulse' : ''}
      `}
    >
      {/* Sentiment indicator bar */}
      <div
        className="absolute left-0 top-0 bottom-0 w-1 rounded-l-lg"
        style={{ backgroundColor: sentimentColor }}
      />

      {/* Header */}
      <div className="flex items-start gap-3 mb-2">
        {/* Neural/Symbolic indicator */}
        <div
          className={`
            w-2 h-2 rounded-full mt-1.5 flex-shrink-0
            ${signal.isNeural ? 'bg-cyan-400' : 'bg-amber-400'}
          `}
          title={signal.isNeural ? 'Neural inference' : 'Symbolic rule'}
        />

        {/* Main content */}
        <div className="flex-1 min-w-0">
          {/* Domain + timestamp + source */}
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <span className="text-xs font-mono uppercase tracking-wider text-slate-500">
              {signal.domain}
            </span>
            <span className="text-slate-600">•</span>
            <span className="text-xs text-slate-500">
              {new Date(signal.timestamp).toLocaleTimeString()}
            </span>
            <SourceBadge
              reliability={signal.source.reliability}
              credibility={signal.source.credibility}
            />
          </div>

          {/* Content */}
          <p className="text-sm text-slate-200 leading-relaxed">
            {signal.content}
          </p>

          {/* Entities */}
          {signal.entities.length > 0 && (
            <div className="flex flex-wrap gap-2 mt-3">
              {signal.entities.slice(0, isExpanded ? undefined : 4).map(entity => (
                <EntityChip
                  key={entity.id}
                  entity={entity}
                  onClick={() => onEntityClick?.(entity)}
                />
              ))}
              {!isExpanded && signal.entities.length > 4 && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setIsExpanded(true);
                  }}
                  className="text-xs text-slate-500 hover:text-slate-300"
                >
                  +{signal.entities.length - 4} more
                </button>
              )}
            </div>
          )}
        </div>

        {/* Risk badge */}
        <div
          className="px-2 py-1 rounded text-xs font-bold"
          style={{
            color: risk.color,
            backgroundColor: `${risk.color}20`,
          }}
        >
          {risk.level}
        </div>
      </div>

      {/* Duplicate accordion */}
      {signal.duplicateCount && signal.duplicateCount > 0 && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            // Toggle duplicates view
          }}
          className="mt-2 pt-2 border-t border-slate-700/50 w-full text-left text-xs text-slate-500 hover:text-slate-300"
        >
          Show {signal.duplicateCount} similar reports ▼
        </button>
      )}

      {/* Pinned indicator */}
      {signal.isPinned && (
        <div className="absolute top-2 right-2 text-amber-400 text-sm">📌</div>
      )}
    </div>
  );
}

// Component 10: Catch-Up Divider
function CatchUpDivider() {
  return (
    <div className="relative flex items-center my-4">
      <div className="flex-1 h-px bg-gradient-to-r from-transparent via-red-500 to-transparent" />
      <span className="px-3 text-xs font-mono text-red-400 uppercase tracking-wider">
        New Messages Below
      </span>
      <div className="flex-1 h-px bg-gradient-to-r from-transparent via-red-500 to-transparent" />
    </div>
  );
}

// Main Feed Component (Component 04)
export function SignalFeed({
  signals,
  onEntityClick,
  onSignalClick,
  lastReadId,
  isLoading = false,
}: SignalFeedProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [newCount, setNewCount] = useState(0);
  const lastReadIndex = signals.findIndex(s => s.id === lastReadId);

  // Auto-scroll to top when new signals arrive
  const scrollToTop = useCallback(() => {
    containerRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
    setNewCount(0);
  }, []);

  return (
    <div className="relative h-full flex flex-col">
      {/* New items badge */}
      {newCount > 0 && (
        <button
          onClick={scrollToTop}
          className="absolute top-2 left-1/2 -translate-x-1/2 z-10 px-4 py-2 bg-cyan-500/90 backdrop-blur rounded-full text-sm font-medium text-white shadow-lg hover:bg-cyan-500 transition-colors"
        >
          {newCount} new signals ↑
        </button>
      )}

      {/* Feed container */}
      <div
        ref={containerRef}
        className="flex-1 overflow-y-auto space-y-3 p-4"
      >
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
          </div>
        )}

        {signals.map((signal, index) => (
          <div key={signal.id}>
            {/* Catch-up divider */}
            {lastReadIndex === index && index > 0 && <CatchUpDivider />}

            <SignalCard
              signal={signal}
              onEntityClick={onEntityClick}
              onClick={() => onSignalClick?.(signal)}
              isNew={lastReadIndex >= 0 && index < lastReadIndex}
            />
          </div>
        ))}

        {signals.length === 0 && !isLoading && (
          <div className="flex flex-col items-center justify-center py-16 text-slate-500">
            <span className="text-4xl mb-4">📡</span>
            <span className="text-sm">No signals in current filter</span>
          </div>
        )}
      </div>
    </div>
  );
}

// Mock data for testing
export const mockSignals: SignalItem[] = [
  {
    id: '1',
    content: 'Russian naval vessels observed conducting exercises in the Black Sea, with increased activity near Ukrainian territorial waters. Satellite imagery confirms presence of 12 warships.',
    domain: 'defense',
    timestamp: new Date().toISOString(),
    source: { name: 'OSINT Aggregator', reliability: 'B', credibility: 2 },
    sentiment: -0.6,
    riskScore: 0.75,
    entities: [
      { id: 'e1', text: 'Russian Navy', type: 'ORG', confidence: 0.95 },
      { id: 'e2', text: 'Black Sea', type: 'LOC', confidence: 0.99 },
      { id: 'e3', text: 'Ukraine', type: 'LOC', confidence: 0.99 },
    ],
    isNeural: true,
  },
  {
    id: '2',
    content: 'Central Bank of Turkey announces emergency rate hike of 500 basis points, bringing benchmark rate to 45%. Lira strengthens 3% against USD in immediate aftermath.',
    domain: 'financial',
    timestamp: new Date(Date.now() - 300000).toISOString(),
    source: { name: 'Reuters', reliability: 'A', credibility: 1 },
    sentiment: 0.2,
    riskScore: 0.5,
    entities: [
      { id: 'e4', text: 'Central Bank of Turkey', type: 'ORG', confidence: 0.98 },
      { id: 'e5', text: 'Turkish Lira', type: 'MONEY', confidence: 0.95 },
    ],
    isNeural: false,
    isPinned: true,
  },
  {
    id: '3',
    content: 'Major ransomware group "LockBit 4.0" claims breach of European hospital network, threatens to release 2TB of patient data unless €50M ransom paid within 72 hours.',
    domain: 'cyber',
    timestamp: new Date(Date.now() - 600000).toISOString(),
    source: { name: 'Dark Web Monitor', reliability: 'C', credibility: 3 },
    sentiment: -0.9,
    riskScore: 0.9,
    entities: [
      { id: 'e6', text: 'LockBit 4.0', type: 'ORG', confidence: 0.92 },
      { id: 'e7', text: 'European Hospital Network', type: 'ORG', confidence: 0.75 },
      { id: 'e8', text: '€50M', type: 'MONEY', confidence: 0.99 },
    ],
    isNeural: true,
    duplicateCount: 5,
  },
];

export { SourceBadge, EntityChip };
