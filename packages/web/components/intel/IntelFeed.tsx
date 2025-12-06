'use client';

import React, { useState } from 'react';
import {
  Radio,
  Wifi,
  WifiOff,
  ChevronDown,
  ChevronUp,
  Clock,
  Hash,
  Filter,
  Trash2,
} from 'lucide-react';
import { useIntelStream } from '@/lib/hooks/useIntelStream';
import { ContextLayer, ContextLayerColors, ContextLayerLabels } from '@/lib/types/causal';
import type { IntelPacket } from '@/lib/types/causal';

const CATEGORIES = [
  'DEFENSE',
  'CYBER',
  'TECH',
  'POLITICS',
  'HEALTH',
  'FINANCE',
  'SPACE',
  'CORP',
  'AGRI',
  'RESOURCES',
  'HOUSING',
  'EDU',
  'CRIME',
  'ENTERTAINMENT',
] as const;

interface IntelFeedProps {
  /** Initial context layer filter */
  initialContext?: ContextLayer;
  /** Database table to stream from */
  table?: string;
  /** Maximum items to display */
  maxItems?: number;
  /** Optional className */
  className?: string;
}

/**
 * IntelFeed - Real-time streaming intelligence feed
 *
 * Features:
 * - Live Supabase Realtime subscription
 * - Context layer filtering (SURFACE/DEEP/DARK)
 * - Category filtering
 * - Expandable packet details
 * - Coherence score display
 * - Live/Pause toggle
 */
export function IntelFeed({
  initialContext,
  table = 'briefings',
  maxItems = 50,
  className = '',
}: IntelFeedProps) {
  const [contextFilter, setContextFilter] = useState<ContextLayer | undefined>(initialContext);
  const [categoryFilter, setCategoryFilter] = useState<string | 'ALL'>('ALL');
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());

  const { packets, isLive, start, pause, clear, status } = useIntelStream({
    contextFilter,
    categoryFilter: categoryFilter === 'ALL' ? undefined : categoryFilter,
    table,
    maxItems,
    autoStart: true,
  });

  const toggleExpand = (id: string) => {
    setExpandedItems((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleContextChange = (context: ContextLayer | undefined) => {
    setContextFilter(context);
    // Stream will automatically resubscribe with new filter
  };

  return (
    <div className={`flex flex-col h-full bg-slate-950 overflow-hidden ${className}`}>
      {/* Filter Bar */}
      <div className="shrink-0 py-3 px-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
        <div className="flex items-center gap-2 overflow-x-auto flex-1 mr-4">
          {/* Context Layer Filter */}
          <div className="flex items-center text-slate-500 text-xs font-mono mr-2">
            <Filter size={12} className="mr-1" /> LAYER:
          </div>
          <button
            onClick={() => handleContextChange(undefined)}
            className={`px-3 py-1 rounded text-[10px] font-mono font-medium tracking-wide transition-colors whitespace-nowrap border ${
              !contextFilter
                ? 'bg-blue-500/20 text-blue-400 border-blue-500/30'
                : 'bg-slate-800 text-slate-400 border-slate-700 hover:border-slate-600'
            }`}
          >
            ALL
          </button>
          {Object.values(ContextLayer).map((layer) => (
            <button
              key={layer}
              onClick={() => handleContextChange(layer)}
              className={`px-3 py-1 rounded text-[10px] font-mono font-medium tracking-wide transition-colors whitespace-nowrap border ${
                contextFilter === layer
                  ? 'bg-blue-500/20 text-blue-400 border-blue-500/30'
                  : 'bg-slate-800 text-slate-400 border-slate-700 hover:border-slate-600'
              }`}
            >
              {layer}
            </button>
          ))}

          {/* Category Filter */}
          <div className="flex items-center text-slate-500 text-xs font-mono mx-2">|</div>
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-[10px] font-mono text-slate-300 focus:border-blue-500 outline-none"
          >
            <option value="ALL">ALL CATEGORIES</option>
            {CATEGORIES.map((cat) => (
              <option key={cat} value={cat}>
                {cat}
              </option>
            ))}
          </select>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={clear}
            className="p-1.5 rounded text-slate-500 hover:text-slate-300 hover:bg-slate-800 transition-colors"
            title="Clear feed"
          >
            <Trash2 size={14} />
          </button>
          <button
            onClick={() => (isLive ? pause() : start())}
            className={`flex items-center gap-2 px-3 py-1 rounded border text-[10px] font-mono font-bold transition-all ${
              isLive
                ? 'bg-green-500/10 text-green-400 border-green-500/30'
                : 'bg-slate-800 text-slate-400 border-slate-700'
            }`}
          >
            {isLive ? (
              <>
                <Radio size={14} className="animate-pulse" />
                LIVE
              </>
            ) : (
              <>
                <WifiOff size={14} className="opacity-50" />
                PAUSED
              </>
            )}
          </button>
        </div>
      </div>

      {/* Status indicator */}
      {status === 'connecting' && (
        <div className="shrink-0 px-4 py-2 bg-amber-500/10 border-b border-amber-500/20 text-amber-400 text-xs font-mono flex items-center gap-2">
          <Wifi size={12} className="animate-pulse" />
          Connecting to stream...
        </div>
      )}

      {/* Feed Content */}
      <div className="flex-1 overflow-y-auto">
        {packets.length === 0 && (
          <div className="flex flex-col items-center justify-center h-64 text-slate-500">
            <Radio size={32} className="animate-spin mb-2 opacity-50" />
            <span className="font-mono text-xs">
              {isLive ? 'AWAITING SIGNAL...' : 'STREAM PAUSED'}
            </span>
          </div>
        )}

        {packets.map((packet, index) => {
          const isExpanded = expandedItems.has(packet.id);
          const isNew = index === 0 && isLive;
          const contextColor = ContextLayerColors[packet.context] || '#3b82f6';

          return (
            <div
              key={packet.id}
              className={`border-b border-slate-800 transition-all duration-500 ${
                isNew
                  ? 'bg-blue-500/10 animate-pulse'
                  : isExpanded
                  ? 'bg-slate-900/50'
                  : 'hover:bg-slate-900/30'
              }`}
            >
              {/* Header Row */}
              <div
                className="flex items-center py-3 px-4 cursor-pointer group"
                onClick={() => toggleExpand(packet.id)}
              >
                {/* Context indicator */}
                <div
                  className="w-1 h-8 rounded-full mr-3"
                  style={{ backgroundColor: contextColor }}
                />

                {/* Category Pill */}
                <div className="w-24 shrink-0">
                  <span className="text-[10px] font-mono font-bold text-slate-500 bg-slate-800 border border-slate-700 px-1.5 py-0.5 rounded group-hover:border-slate-600 transition-colors">
                    {packet.category}
                  </span>
                </div>

                {/* Headline */}
                <div className="flex-1 min-w-0 pr-4">
                  <div className="flex items-baseline gap-3">
                    <h3
                      className={`font-mono text-sm truncate transition-colors ${
                        isExpanded
                          ? 'text-blue-400 font-bold'
                          : 'text-slate-200 group-hover:text-blue-300'
                      }`}
                    >
                      {packet.header}
                    </h3>
                    <span className="text-xs text-slate-500 hidden sm:inline-block truncate opacity-70">
                      // {packet.summary}
                    </span>
                  </div>
                </div>

                {/* Metadata */}
                <div className="flex items-center gap-4 shrink-0">
                  <div className="flex items-center text-slate-500 text-[10px] font-mono hidden sm:flex">
                    <Clock size={10} className="mr-1" />
                    {new Date(packet.timestamp).toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </div>
                  <div className="flex items-center text-cyan-400 text-[10px] font-mono hidden sm:flex bg-cyan-500/5 px-2 py-0.5 rounded border border-cyan-500/10">
                    <Hash size={10} className="mr-1" />
                    {packet.coherence.toFixed(2)}
                  </div>
                  <div className="text-slate-500 group-hover:text-slate-300 transition-colors">
                    {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </div>
                </div>
              </div>

              {/* Expanded Body */}
              {isExpanded && (
                <div className="px-4 pb-4 pl-8 sm:pl-12 animate-in slide-in-from-top-2 duration-200">
                  <div className="p-4 bg-slate-900/80 rounded border border-slate-700 shadow-inner">
                    {/* Mobile summary */}
                    <div className="sm:hidden mb-3 pb-3 border-b border-slate-700">
                      <span className="text-xs font-bold text-slate-300 block mb-1">SUMMARY</span>
                      <p className="text-sm text-slate-400">{packet.summary}</p>
                    </div>

                    <div className="flex gap-6 flex-col md:flex-row">
                      <div className="flex-1">
                        <span className="text-[10px] font-mono text-blue-400 mb-2 block tracking-widest uppercase">
                          Analysis Body
                        </span>
                        <p className="text-sm text-slate-300 leading-relaxed">{packet.body}</p>
                      </div>

                      <div className="md:w-48 shrink-0 flex flex-col gap-2 pt-2 md:pt-0 md:border-l md:border-slate-700 md:pl-4">
                        <div>
                          <span className="text-[10px] font-mono text-slate-500 block">SOURCE</span>
                          <span className="text-xs font-mono text-slate-300">{packet.source}</span>
                        </div>
                        <div>
                          <span className="text-[10px] font-mono text-slate-500 block">CONTEXT</span>
                          <span className="text-xs font-mono text-slate-300">{packet.context}</span>
                        </div>
                        <div>
                          <span className="text-[10px] font-mono text-slate-500 block">ID</span>
                          <span className="text-xs font-mono text-slate-300 truncate block">
                            {packet.id}
                          </span>
                        </div>
                        <div className="mt-2">
                          <button className="w-full py-1 bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 border border-blue-500/30 rounded text-[10px] font-mono transition-colors">
                            EXPORT PACKET
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer status */}
      <div className="shrink-0 px-4 py-2 border-t border-slate-800 bg-slate-900/50 flex items-center justify-between text-[10px] font-mono text-slate-500">
        <span>
          {packets.length} packets {contextFilter && `(${contextFilter})`}
        </span>
        <span className="flex items-center gap-2">
          <div
            className={`w-1.5 h-1.5 rounded-full ${
              status === 'connected'
                ? 'bg-green-500'
                : status === 'connecting'
                ? 'bg-amber-500 animate-pulse'
                : 'bg-slate-600'
            }`}
          />
          {status.toUpperCase()}
        </span>
      </div>
    </div>
  );
}

export default IntelFeed;
