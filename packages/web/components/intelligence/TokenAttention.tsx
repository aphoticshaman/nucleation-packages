'use client';

import { useState, useMemo } from 'react';

interface TokenWeight {
  token: string;
  weight: number; // 0-1 attention weight
  startIndex?: number;
  endIndex?: number;
}

interface TokenAttentionProps {
  text: string;
  tokens: TokenWeight[];
  showWeights?: boolean;
  colorScheme?: 'neural' | 'risk' | 'custom';
  customColors?: { low: string; high: string };
  onTokenClick?: (token: TokenWeight) => void;
}

// Component 34: Token Attention Highlighter
export function TokenAttention({
  text,
  tokens,
  showWeights = false,
  colorScheme = 'neural',
  customColors,
  onTokenClick,
}: TokenAttentionProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // Build highlighted segments
  const segments = useMemo(() => {
    // Sort tokens by position if available, otherwise by occurrence in text
    const sortedTokens = tokens
      .map((t, i) => ({
        ...t,
        originalIndex: i,
        pos: t.startIndex ?? text.toLowerCase().indexOf(t.token.toLowerCase()),
      }))
      .filter(t => t.pos !== -1)
      .sort((a, b) => a.pos - b.pos);

    const result: { text: string; token?: TokenWeight; index?: number }[] = [];
    let lastEnd = 0;

    sortedTokens.forEach((token, idx) => {
      const start = token.startIndex ?? text.toLowerCase().indexOf(token.token.toLowerCase(), lastEnd);
      const end = token.endIndex ?? (start + token.token.length);

      // Add unhighlighted text before this token
      if (start > lastEnd) {
        result.push({ text: text.slice(lastEnd, start) });
      }

      // Add highlighted token
      result.push({
        text: text.slice(start, end),
        token: token,
        index: token.originalIndex,
      });

      lastEnd = end;
    });

    // Add remaining text
    if (lastEnd < text.length) {
      result.push({ text: text.slice(lastEnd) });
    }

    return result;
  }, [text, tokens]);

  // Get color for weight
  const getColor = (weight: number, alpha: number = 1) => {
    if (colorScheme === 'neural') {
      // Cyan gradient
      const r = Math.round(6 + (100 - 6) * (1 - weight));
      const g = Math.round(182 + (200 - 182) * (1 - weight));
      const b = Math.round(212 + (230 - 212) * (1 - weight));
      return `rgba(${r}, ${g}, ${b}, ${alpha * weight})`;
    } else if (colorScheme === 'risk') {
      // Yellow to red gradient
      if (weight < 0.5) {
        return `rgba(234, 179, 8, ${alpha * weight * 2})`;
      } else {
        return `rgba(239, 68, 68, ${alpha * (weight - 0.5) * 2 + 0.3})`;
      }
    } else if (customColors) {
      // Interpolate between custom colors
      return `rgba(${Math.round(parseInt(customColors.low.slice(1, 3), 16) * (1 - weight) + parseInt(customColors.high.slice(1, 3), 16) * weight)}, ${Math.round(parseInt(customColors.low.slice(3, 5), 16) * (1 - weight) + parseInt(customColors.high.slice(3, 5), 16) * weight)}, ${Math.round(parseInt(customColors.low.slice(5, 7), 16) * (1 - weight) + parseInt(customColors.high.slice(5, 7), 16) * weight)}, ${alpha * weight})`;
    }
    return 'transparent';
  };

  return (
    <div className="relative">
      {/* Main text with highlighting */}
      <p className="text-sm text-slate-200 leading-relaxed">
        {segments.map((segment, i) => {
          if (!segment.token) {
            return <span key={i}>{segment.text}</span>;
          }

          const isHovered = hoveredIndex === segment.index;
          const weight = segment.token.weight;

          return (
            <span
              key={i}
              onClick={() => onTokenClick?.(segment.token!)}
              onMouseEnter={() => setHoveredIndex(segment.index!)}
              onMouseLeave={() => setHoveredIndex(null)}
              className={`
                relative cursor-pointer transition-all duration-200
                ${onTokenClick ? 'hover:scale-105' : ''}
              `}
              style={{
                backgroundColor: getColor(weight, isHovered ? 1 : 0.6),
                borderRadius: '2px',
                padding: '0 2px',
              }}
            >
              {segment.text}

              {/* Hover tooltip with weight */}
              {isHovered && showWeights && (
                <span
                  className="absolute -top-8 left-1/2 -translate-x-1/2 px-2 py-1 bg-slate-800 rounded text-xs text-cyan-400 font-mono whitespace-nowrap z-10"
                >
                  {(weight * 100).toFixed(1)}%
                </span>
              )}
            </span>
          );
        })}
      </p>

      {/* Weight legend */}
      {showWeights && (
        <div className="mt-4 flex items-center gap-2 text-xs text-slate-500">
          <span>Attention:</span>
          <div className="flex items-center gap-1">
            <span>Low</span>
            <div
              className="w-20 h-2 rounded"
              style={{
                background: colorScheme === 'neural'
                  ? 'linear-gradient(to right, rgba(6, 182, 212, 0.1), rgba(6, 182, 212, 1))'
                  : 'linear-gradient(to right, rgba(234, 179, 8, 0.1), rgba(239, 68, 68, 1))',
              }}
            />
            <span>High</span>
          </div>
        </div>
      )}
    </div>
  );
}

// Heatmap variant for dense text
export function TokenHeatmap({
  tokens,
  maxTokens = 50,
  onTokenClick,
}: {
  tokens: TokenWeight[];
  maxTokens?: number;
  onTokenClick?: (token: TokenWeight) => void;
}) {
  const displayTokens = tokens.slice(0, maxTokens);

  return (
    <div className="flex flex-wrap gap-1">
      {displayTokens.map((token, i) => (
        <button
          key={i}
          onClick={() => onTokenClick?.(token)}
          className="px-2 py-1 rounded text-xs font-mono transition-transform hover:scale-110"
          style={{
            backgroundColor: `rgba(6, 182, 212, ${token.weight})`,
            color: token.weight > 0.5 ? '#0f172a' : '#e2e8f0',
          }}
          title={`${token.token}: ${(token.weight * 100).toFixed(1)}%`}
        >
          {token.token}
        </button>
      ))}
      {tokens.length > maxTokens && (
        <span className="px-2 py-1 text-xs text-slate-500">
          +{tokens.length - maxTokens} more
        </span>
      )}
    </div>
  );
}

// Side-by-side comparison of two attention patterns
export function AttentionComparison({
  text,
  patternA: { tokens: tokensA, label: labelA },
  patternB: { tokens: tokensB, label: labelB },
}: {
  text: string;
  patternA: { tokens: TokenWeight[]; label: string };
  patternB: { tokens: TokenWeight[]; label: string };
}) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-800">
        <div className="text-xs text-slate-400 mb-2">{labelA}</div>
        <TokenAttention text={text} tokens={tokensA} showWeights colorScheme="neural" />
      </div>
      <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-800">
        <div className="text-xs text-slate-400 mb-2">{labelB}</div>
        <TokenAttention text={text} tokens={tokensB} showWeights colorScheme="risk" />
      </div>
    </div>
  );
}

// Example data
export const mockTokenWeights: TokenWeight[] = [
  { token: 'Russian', weight: 0.85 },
  { token: 'naval', weight: 0.72 },
  { token: 'vessels', weight: 0.65 },
  { token: 'Black Sea', weight: 0.91 },
  { token: 'Ukrainian', weight: 0.88 },
  { token: 'territorial', weight: 0.78 },
  { token: 'warships', weight: 0.82 },
];
