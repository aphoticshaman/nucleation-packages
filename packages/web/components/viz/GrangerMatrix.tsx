'use client';

import React, { useMemo } from 'react';
import { ArrowRight, Info } from 'lucide-react';

interface GrangerMatrixProps {
  labels: string[];
  matrix: number[][];  // Normalized Transfer Entropy values (0-1)
  threshold?: number;
  showLabels?: boolean;
  onCellClick?: (source: string, target: string, value: number) => void;
}

/**
 * Granger Causality / Transfer Entropy Matrix Heatmap
 *
 * Visualizes directed information flow between entities
 * using a color-coded adjacency matrix.
 */
export function GrangerMatrix({
  labels,
  matrix,
  threshold = 0.1,
  showLabels = true,
  onCellClick,
}: GrangerMatrixProps) {
  const n = labels.length;

  // Find min/max for color scaling
  const { minVal, maxVal } = useMemo(() => {
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j && matrix[i][j] > 0) {
          min = Math.min(min, matrix[i][j]);
          max = Math.max(max, matrix[i][j]);
        }
      }
    }
    return { minVal: min === Infinity ? 0 : min, maxVal: max === -Infinity ? 1 : max };
  }, [matrix, n]);

  // Color interpolation
  const getColor = (value: number): string => {
    if (value < threshold) return 'transparent';

    const normalized = (value - minVal) / (maxVal - minVal || 1);

    // Cool to hot gradient: dark blue → cyan → yellow → red
    if (normalized < 0.25) {
      const t = normalized / 0.25;
      return `rgb(${Math.round(30 + t * 10)}, ${Math.round(50 + t * 130)}, ${Math.round(100 + t * 100)})`;
    } else if (normalized < 0.5) {
      const t = (normalized - 0.25) / 0.25;
      return `rgb(${Math.round(40 + t * 180)}, ${Math.round(180 + t * 60)}, ${Math.round(200 - t * 100)})`;
    } else if (normalized < 0.75) {
      const t = (normalized - 0.5) / 0.25;
      return `rgb(${Math.round(220 + t * 35)}, ${Math.round(240 - t * 80)}, ${Math.round(100 - t * 80)})`;
    } else {
      const t = (normalized - 0.75) / 0.25;
      return `rgb(${Math.round(255 - t * 20)}, ${Math.round(160 - t * 100)}, ${Math.round(20 - t * 20)})`;
    }
  };

  const cellSize = Math.min(40, Math.max(20, 400 / n));

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-mono font-bold text-slate-300">
            Transfer Entropy Matrix
          </h3>
          <div className="group relative">
            <Info size={14} className="text-slate-500 cursor-help" />
            <div className="absolute left-0 top-6 w-64 p-3 bg-slate-800 border border-slate-700 rounded-lg shadow-xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
              <p className="text-xs text-slate-400">
                Cell (i,j) shows directed information flow from row i to column j.
                Brighter colors indicate stronger causal influence.
              </p>
            </div>
          </div>
        </div>
        <div className="text-xs font-mono text-slate-500">
          Threshold: TE &gt; {threshold}
        </div>
      </div>

      {/* Matrix */}
      <div className="overflow-auto">
        <div className="inline-block">
          {/* Column labels */}
          {showLabels && (
            <div className="flex" style={{ marginLeft: cellSize * 1.5 }}>
              {labels.map((label, j) => (
                <div
                  key={`col-${j}`}
                  className="text-[9px] font-mono text-slate-400 overflow-hidden text-ellipsis whitespace-nowrap transform -rotate-45 origin-left"
                  style={{ width: cellSize, height: cellSize * 1.5 }}
                >
                  {label.slice(0, 8)}
                </div>
              ))}
            </div>
          )}

          {/* Rows */}
          {matrix.map((row, i) => (
            <div key={`row-${i}`} className="flex items-center">
              {/* Row label */}
              {showLabels && (
                <div
                  className="text-[9px] font-mono text-slate-400 text-right pr-2 overflow-hidden text-ellipsis whitespace-nowrap"
                  style={{ width: cellSize * 1.5 }}
                >
                  {labels[i].slice(0, 8)}
                </div>
              )}

              {/* Cells */}
              {row.map((value, j) => (
                <div
                  key={`cell-${i}-${j}`}
                  className={`border border-slate-800 transition-all ${
                    i !== j && value >= threshold
                      ? 'cursor-pointer hover:ring-2 hover:ring-cyan-400 hover:z-10'
                      : ''
                  }`}
                  style={{
                    width: cellSize,
                    height: cellSize,
                    backgroundColor: i === j ? '#0f172a' : getColor(value),
                  }}
                  onClick={() => {
                    if (i !== j && value >= threshold && onCellClick) {
                      onCellClick(labels[i], labels[j], value);
                    }
                  }}
                  title={
                    i === j
                      ? 'Self (diagonal)'
                      : `${labels[i]} → ${labels[j]}: TE=${value.toFixed(3)}`
                  }
                >
                  {/* Show value for significant cells */}
                  {i !== j && value >= threshold && cellSize >= 30 && (
                    <span className="flex items-center justify-center h-full text-[8px] font-mono text-white/80">
                      {(value * 100).toFixed(0)}
                    </span>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Color Legend */}
      <div className="flex items-center gap-4">
        <span className="text-xs font-mono text-slate-500">Weak</span>
        <div
          className="flex-1 h-3 rounded"
          style={{
            background: `linear-gradient(to right,
              rgb(30, 50, 100),
              rgb(40, 180, 200),
              rgb(220, 240, 100),
              rgb(255, 160, 20),
              rgb(235, 60, 0)
            )`,
          }}
        />
        <span className="text-xs font-mono text-slate-500">Strong</span>
      </div>

      {/* Direction indicator */}
      <div className="flex items-center justify-center gap-2 text-xs font-mono text-slate-500">
        <span>Row</span>
        <ArrowRight size={14} />
        <span>Column (Causal Direction)</span>
      </div>
    </div>
  );
}
