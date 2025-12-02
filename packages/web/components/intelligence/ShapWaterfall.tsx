'use client';

import { useMemo } from 'react';

interface ShapFeature {
  name: string;
  value: number; // Positive = increases risk, negative = decreases
  rawValue?: string | number; // The actual feature value
  isNeural?: boolean; // Neural vs Symbolic source
}

interface ShapWaterfallProps {
  features: ShapFeature[];
  baseValue: number;
  finalValue: number;
  label?: string;
  maxFeatures?: number;
}

// Component 32: SHAP Value Waterfall
export function ShapWaterfall({
  features,
  baseValue,
  finalValue,
  label = 'Feature Contributions',
  maxFeatures = 10,
}: ShapWaterfallProps) {
  const { sortedFeatures, maxAbsValue, scale } = useMemo(() => {
    // Sort by absolute value
    const sorted = [...features]
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, maxFeatures);

    const maxAbs = Math.max(...sorted.map(f => Math.abs(f.value)), 0.1);

    return {
      sortedFeatures: sorted,
      maxAbsValue: maxAbs,
      scale: 100 / maxAbs, // Scale to fit in 100px width
    };
  }, [features, maxFeatures]);

  // Calculate running total for waterfall
  const waterfallData = useMemo(() => {
    let runningTotal = baseValue;
    return sortedFeatures.map(feature => {
      const startValue = runningTotal;
      runningTotal += feature.value;
      return {
        ...feature,
        startValue,
        endValue: runningTotal,
      };
    });
  }, [sortedFeatures, baseValue]);

  return (
    <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-slate-300">{label}</h3>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-red-500/50" />
            <span className="text-slate-400">Increases Risk</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-green-500/50" />
            <span className="text-slate-400">Decreases Risk</span>
          </div>
        </div>
      </div>

      {/* Base value */}
      <div className="flex items-center gap-3 mb-2 py-2 border-b border-slate-800">
        <span className="text-xs text-slate-500 w-32">Base Score</span>
        <div className="flex-1 flex items-center justify-center">
          <span className="text-sm font-mono text-slate-400">
            {baseValue.toFixed(2)}
          </span>
        </div>
        <span className="text-xs text-slate-600 w-20" />
      </div>

      {/* Feature bars */}
      <div className="space-y-1">
        {waterfallData.map((feature, index) => {
          const isPositive = feature.value > 0;
          const barWidth = Math.abs(feature.value) * scale;
          const centerOffset = 50; // Center of the container (percentage)

          return (
            <div
              key={feature.name}
              className="flex items-center gap-3 py-1.5 hover:bg-slate-800/30 rounded transition-colors"
            >
              {/* Feature name */}
              <div className="w-32 flex items-center gap-1.5 truncate">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    feature.isNeural ? 'bg-cyan-400' : 'bg-amber-400'
                  }`}
                  title={feature.isNeural ? 'Neural' : 'Symbolic'}
                />
                <span className="text-xs text-slate-300 truncate">
                  {feature.name}
                </span>
              </div>

              {/* Waterfall bar */}
              <div className="flex-1 relative h-6">
                {/* Center line */}
                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-700" />

                {/* Bar */}
                <div
                  className={`
                    absolute top-1 bottom-1 rounded transition-all duration-300
                    ${isPositive
                      ? 'bg-gradient-to-r from-red-600/80 to-red-500/80'
                      : 'bg-gradient-to-l from-green-600/80 to-green-500/80'
                    }
                  `}
                  style={{
                    left: isPositive ? `${centerOffset}%` : `${centerOffset - barWidth}%`,
                    width: `${barWidth}%`,
                    boxShadow: isPositive
                      ? '0 0 10px rgba(239, 68, 68, 0.3)'
                      : '0 0 10px rgba(34, 197, 94, 0.3)',
                  }}
                />

                {/* Value label */}
                <div
                  className={`
                    absolute top-1/2 -translate-y-1/2 text-xs font-mono
                    ${isPositive ? 'text-red-400' : 'text-green-400'}
                  `}
                  style={{
                    left: isPositive
                      ? `${centerOffset + barWidth + 2}%`
                      : undefined,
                    right: !isPositive
                      ? `${100 - centerOffset + barWidth + 2}%`
                      : undefined,
                  }}
                >
                  {isPositive ? '+' : ''}{feature.value.toFixed(3)}
                </div>
              </div>

              {/* Raw value */}
              <span className="text-xs text-slate-500 w-20 text-right truncate">
                {feature.rawValue !== undefined ? String(feature.rawValue) : '-'}
              </span>
            </div>
          );
        })}
      </div>

      {/* Final value */}
      <div className="flex items-center gap-3 mt-2 pt-2 border-t border-slate-800">
        <span className="text-xs text-slate-400 w-32 font-medium">Final Score</span>
        <div className="flex-1 flex items-center justify-center">
          <span
            className={`
              text-lg font-bold font-mono
              ${finalValue > 0.7 ? 'text-red-400' : finalValue > 0.4 ? 'text-yellow-400' : 'text-green-400'}
            `}
          >
            {finalValue.toFixed(3)}
          </span>
        </div>
        <div className="w-20 flex justify-end">
          <span
            className={`
              px-2 py-0.5 rounded text-xs font-bold
              ${finalValue > 0.7
                ? 'bg-red-500/20 text-red-400'
                : finalValue > 0.4
                  ? 'bg-yellow-500/20 text-yellow-400'
                  : 'bg-green-500/20 text-green-400'
              }
            `}
          >
            {finalValue > 0.7 ? 'HIGH' : finalValue > 0.4 ? 'MED' : 'LOW'}
          </span>
        </div>
      </div>
    </div>
  );
}

// Horizontal bar variant for compact display
export function ShapBar({
  feature,
  maxValue = 0.5,
}: {
  feature: ShapFeature;
  maxValue?: number;
}) {
  const isPositive = feature.value > 0;
  const pct = Math.min(Math.abs(feature.value) / maxValue * 50, 50);

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-slate-400 w-24 truncate">{feature.name}</span>
      <div className="flex-1 relative h-4 bg-slate-800 rounded overflow-hidden">
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-600" />
        <div
          className={`
            absolute top-0.5 bottom-0.5 rounded
            ${isPositive ? 'bg-red-500' : 'bg-green-500'}
          `}
          style={{
            left: isPositive ? '50%' : `${50 - pct}%`,
            width: `${pct}%`,
          }}
        />
      </div>
      <span
        className={`text-xs font-mono w-12 text-right ${
          isPositive ? 'text-red-400' : 'text-green-400'
        }`}
      >
        {isPositive ? '+' : ''}{feature.value.toFixed(2)}
      </span>
    </div>
  );
}

// Example data
export const mockShapFeatures: ShapFeature[] = [
  { name: 'Transaction Amount', value: 0.23, rawValue: '$47,500', isNeural: false },
  { name: 'Country Risk', value: 0.18, rawValue: 0.78, isNeural: true },
  { name: 'Velocity (24h)', value: 0.12, rawValue: '12 txns', isNeural: true },
  { name: 'Account Age', value: -0.08, rawValue: '3 years', isNeural: false },
  { name: 'Prior SAR', value: 0.15, rawValue: 'Yes', isNeural: false },
  { name: 'Network Centrality', value: 0.09, rawValue: 0.65, isNeural: true },
  { name: 'Time of Day', value: -0.04, rawValue: '14:32', isNeural: false },
  { name: 'Recipient History', value: -0.11, rawValue: 'Clean', isNeural: true },
];
