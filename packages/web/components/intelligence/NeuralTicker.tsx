'use client';

import { useEffect, useState, useRef } from 'react';
import { tw, getSentimentColor, getRiskLabel } from '@/lib/design-system';

interface TickerItem {
  id: string;
  headline: string;
  domain: string;
  sentiment: number;
  riskScore: number;
  timestamp: string;
  isBreaking?: boolean;
  keywords?: string[];
}

interface NeuralTickerProps {
  items: TickerItem[];
  speed?: number; // pixels per second
  onItemClick?: (item: TickerItem) => void;
  isPaused?: boolean;
}

// Critical keywords that halt the ticker
const CRITICAL_KEYWORDS = ['nuclear', 'anthrax', 'bioweapon', 'chemical attack', 'wmd', 'icbm'];

export function NeuralTicker({
  items,
  speed = 50,
  onItemClick,
  isPaused = false,
}: NeuralTickerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [offset, setOffset] = useState(0);
  const [isHalted, setIsHalted] = useState(false);
  const [haltedItem, setHaltedItem] = useState<TickerItem | null>(null);

  // Check for critical keywords
  useEffect(() => {
    for (const item of items) {
      const text = item.headline.toLowerCase();
      const hasCritical = CRITICAL_KEYWORDS.some(kw => text.includes(kw));
      if (hasCritical) {
        setIsHalted(true);
        setHaltedItem(item);
        return;
      }
    }
    setIsHalted(false);
    setHaltedItem(null);
  }, [items]);

  // Scroll animation
  useEffect(() => {
    if (isPaused || isHalted) return;

    const animate = () => {
      setOffset(prev => {
        const container = containerRef.current;
        if (!container) return prev;
        const contentWidth = container.scrollWidth / 2;
        const newOffset = prev - speed / 60;
        return newOffset <= -contentWidth ? 0 : newOffset;
      });
    };

    const frameId = requestAnimationFrame(function tick() {
      animate();
      requestAnimationFrame(tick);
    });

    return () => cancelAnimationFrame(frameId);
  }, [speed, isPaused, isHalted]);

  const renderItem = (item: TickerItem, index: number) => {
    const sentimentColor = getSentimentColor(item.sentiment);
    const risk = getRiskLabel(item.riskScore);
    const isCritical = item.riskScore >= 0.8 || item.isBreaking;

    return (
      <button
        key={`${item.id}-${index}`}
        onClick={() => onItemClick?.(item)}
        className={`
          inline-flex items-center gap-3 px-4 py-2 mx-2
          rounded border transition-all duration-200
          hover:scale-105 cursor-pointer whitespace-nowrap
          ${isCritical
            ? 'border-red-500/70 bg-red-500/20 animate-pulse'
            : 'border-slate-700/50 bg-slate-800/50 hover:bg-slate-700/50'
          }
        `}
        style={{
          boxShadow: isCritical
            ? '0 0 20px rgba(239, 68, 68, 0.3)'
            : `0 0 10px ${sentimentColor}20`,
        }}
      >
        {/* Domain badge */}
        <span className="text-xs font-mono uppercase tracking-wider text-slate-500">
          {item.domain}
        </span>

        {/* Headline */}
        <span className={`font-mono text-sm ${isCritical ? 'text-red-300' : 'text-slate-200'}`}>
          {item.headline}
        </span>

        {/* Risk indicator */}
        <span
          className="text-xs font-bold px-2 py-0.5 rounded"
          style={{
            color: risk.color,
            backgroundColor: `${risk.color}20`,
          }}
        >
          {risk.level}
        </span>

        {/* Sentiment underline glow */}
        <div
          className="absolute bottom-0 left-0 right-0 h-0.5 opacity-50"
          style={{ backgroundColor: sentimentColor }}
        />
      </button>
    );
  };

  // Halted state for critical alerts
  if (isHalted && haltedItem) {
    return (
      <div className="relative w-full h-12 bg-red-950/80 backdrop-blur-xl border-y border-red-500/50 overflow-hidden">
        <div className="absolute inset-0 flex items-center justify-center animate-pulse">
          <div className="flex items-center gap-4">
            <div className="w-3 h-3 rounded-full bg-red-500 animate-ping" />
            <span className="font-mono text-red-300 font-bold uppercase tracking-wider">
              CRITICAL ALERT
            </span>
            <span className="font-mono text-red-100">
              {haltedItem.headline}
            </span>
            <button
              onClick={() => {
                setIsHalted(false);
                setHaltedItem(null);
              }}
              className="px-3 py-1 bg-red-500/30 border border-red-500/50 rounded text-red-200 text-xs hover:bg-red-500/50 transition-colors"
            >
              ACKNOWLEDGE
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative w-full h-12 bg-slate-950/80 backdrop-blur-xl border-y border-slate-800/50 overflow-hidden">
      {/* Gradient masks */}
      <div className="absolute left-0 top-0 bottom-0 w-20 bg-gradient-to-r from-slate-950 to-transparent z-10" />
      <div className="absolute right-0 top-0 bottom-0 w-20 bg-gradient-to-l from-slate-950 to-transparent z-10" />

      {/* Scrolling content */}
      <div
        ref={containerRef}
        className="flex items-center h-full"
        style={{
          transform: `translate3d(${offset}px, 0, 0)`,
          willChange: 'transform',
        }}
      >
        {/* Duplicate items for seamless loop */}
        {items.map((item, i) => renderItem(item, i))}
        {items.map((item, i) => renderItem(item, i + items.length))}
      </div>

      {/* Pause indicator */}
      {isPaused && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-950/50">
          <span className="text-slate-400 font-mono text-sm">PAUSED</span>
        </div>
      )}
    </div>
  );
}

// Example usage data
export const mockTickerItems: TickerItem[] = [
  {
    id: '1',
    headline: 'OPEC+ announces unexpected production cut of 1.2M barrels/day',
    domain: 'energy',
    sentiment: -0.4,
    riskScore: 0.7,
    timestamp: new Date().toISOString(),
  },
  {
    id: '2',
    headline: 'Major ransomware attack targets European banking infrastructure',
    domain: 'cyber',
    sentiment: -0.8,
    riskScore: 0.85,
    timestamp: new Date().toISOString(),
    isBreaking: true,
  },
  {
    id: '3',
    headline: 'Fed signals potential rate pause in upcoming FOMC meeting',
    domain: 'financial',
    sentiment: 0.3,
    riskScore: 0.3,
    timestamp: new Date().toISOString(),
  },
  {
    id: '4',
    headline: 'Satellite imagery reveals new military installations in disputed region',
    domain: 'defense',
    sentiment: -0.5,
    riskScore: 0.65,
    timestamp: new Date().toISOString(),
  },
  {
    id: '5',
    headline: 'Breakthrough in quantum error correction reported by research team',
    domain: 'quantum',
    sentiment: 0.7,
    riskScore: 0.2,
    timestamp: new Date().toISOString(),
  },
];
