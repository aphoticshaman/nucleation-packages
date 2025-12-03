'use client';

import { useMemo } from 'react';

interface XYZAMetrics {
  coherence_x: number;    // X: Phase synchronization [0, 1]
  complexity_y: number;   // Y: Information entropy [0, 1]
  reflection_z: number;   // Z: Meta-cognition [0, 1]
  attunement_a: number;   // A: Human-AI coupling [0, 1]
  combined_score?: number;
  cognitive_level?: string;
}

interface XYZARadarProps {
  metrics: XYZAMetrics;
  size?: number;
  showLabels?: boolean;
  showThresholds?: boolean;
  animated?: boolean;
}

// Key thresholds from cognitive module
const THRESHOLDS = {
  coherence_flow: 0.76,     // X >= 0.76 for flow
  complexity_low: 0.4,      // Y optimal range
  complexity_high: 0.7,
  reflection_aware: 0.5,    // Z >= 0.5 for self-awareness
  attunement_coupled: 0.42, // A >= 0.42 (K_human)
};

const AXIS_CONFIG = [
  { key: 'coherence_x', label: 'Coherence', shortLabel: 'X', color: '#3b82f6', threshold: THRESHOLDS.coherence_flow },
  { key: 'complexity_y', label: 'Complexity', shortLabel: 'Y', color: '#10b981', threshold: 0.55 }, // Middle of optimal
  { key: 'reflection_z', label: 'Reflection', shortLabel: 'Z', color: '#f59e0b', threshold: THRESHOLDS.reflection_aware },
  { key: 'attunement_a', label: 'Attunement', shortLabel: 'A', color: '#8b5cf6', threshold: THRESHOLDS.attunement_coupled },
];

export function XYZARadar({
  metrics,
  size = 200,
  showLabels = true,
  showThresholds = true,
  animated = true,
}: XYZARadarProps) {
  const center = size / 2;
  const radius = (size / 2) - 30; // Leave room for labels

  // Calculate polygon points
  const points = useMemo(() => {
    return AXIS_CONFIG.map((axis, i) => {
      const angle = (i * 2 * Math.PI) / 4 - Math.PI / 2; // Start from top
      const value = metrics[axis.key as keyof XYZAMetrics] as number || 0;
      const r = value * radius;
      return {
        x: center + r * Math.cos(angle),
        y: center + r * Math.sin(angle),
        angle,
        value,
        ...axis,
      };
    });
  }, [metrics, center, radius]);

  // Create SVG path for the polygon
  const polygonPath = points.map((p, i) =>
    `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`
  ).join(' ') + ' Z';

  // Threshold polygon
  const thresholdPath = useMemo(() => {
    return AXIS_CONFIG.map((axis, i) => {
      const angle = (i * 2 * Math.PI) / 4 - Math.PI / 2;
      const r = axis.threshold * radius;
      const x = center + r * Math.cos(angle);
      const y = center + r * Math.sin(angle);
      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
    }).join(' ') + ' Z';
  }, [center, radius]);

  // Get cognitive level color
  const levelColor = useMemo(() => {
    const level = metrics.cognitive_level?.toLowerCase() || '';
    switch (level) {
      case 'peak': return '#22c55e';
      case 'enhanced': return '#3b82f6';
      case 'normal': return '#f59e0b';
      case 'degraded': return '#ef4444';
      default: return '#64748b';
    }
  }, [metrics.cognitive_level]);

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="overflow-visible">
        {/* Background circles */}
        {[0.25, 0.5, 0.75, 1].map((scale) => (
          <circle
            key={scale}
            cx={center}
            cy={center}
            r={radius * scale}
            fill="none"
            stroke="rgba(148, 163, 184, 0.2)"
            strokeWidth="1"
          />
        ))}

        {/* Axis lines */}
        {points.map((p, i) => (
          <line
            key={i}
            x1={center}
            y1={center}
            x2={center + radius * Math.cos(p.angle)}
            y2={center + radius * Math.sin(p.angle)}
            stroke="rgba(148, 163, 184, 0.3)"
            strokeWidth="1"
          />
        ))}

        {/* Threshold polygon */}
        {showThresholds && (
          <path
            d={thresholdPath}
            fill="rgba(59, 130, 246, 0.1)"
            stroke="rgba(59, 130, 246, 0.4)"
            strokeWidth="1"
            strokeDasharray="4 2"
          />
        )}

        {/* Data polygon */}
        <path
          d={polygonPath}
          fill={`${levelColor}20`}
          stroke={levelColor}
          strokeWidth="2"
          className={animated ? 'transition-all duration-500' : ''}
        />

        {/* Data points */}
        {points.map((p, i) => (
          <g key={i}>
            <circle
              cx={p.x}
              cy={p.y}
              r="4"
              fill={p.color}
              className={animated ? 'transition-all duration-500' : ''}
            />
            {/* Glow effect when above threshold */}
            {p.value >= p.threshold && (
              <circle
                cx={p.x}
                cy={p.y}
                r="8"
                fill={p.color}
                opacity="0.3"
                className="animate-pulse"
              />
            )}
          </g>
        ))}

        {/* Labels */}
        {showLabels && points.map((p, i) => {
          const labelRadius = radius + 20;
          const lx = center + labelRadius * Math.cos(p.angle);
          const ly = center + labelRadius * Math.sin(p.angle);
          return (
            <g key={`label-${i}`}>
              <text
                x={lx}
                y={ly}
                textAnchor="middle"
                dominantBaseline="middle"
                className="fill-slate-300 text-xs font-medium"
              >
                {p.shortLabel}
              </text>
              <text
                x={lx}
                y={ly + 12}
                textAnchor="middle"
                dominantBaseline="middle"
                className="fill-slate-500 text-[10px]"
              >
                {(p.value * 100).toFixed(0)}%
              </text>
            </g>
          );
        })}

        {/* Center score */}
        <text
          x={center}
          y={center - 8}
          textAnchor="middle"
          className="fill-white text-lg font-bold"
        >
          {((metrics.combined_score || 0) * 100).toFixed(0)}
        </text>
        <text
          x={center}
          y={center + 10}
          textAnchor="middle"
          className="fill-slate-400 text-[10px] uppercase tracking-wider"
        >
          {metrics.cognitive_level || 'Unknown'}
        </text>
      </svg>
    </div>
  );
}

// Compact inline version
export function XYZAInline({ metrics }: { metrics: XYZAMetrics }) {
  const combined = metrics.combined_score || 0;
  const level = metrics.cognitive_level?.toLowerCase() || 'unknown';

  const levelColors: Record<string, string> = {
    peak: 'text-green-400 bg-green-500/20',
    enhanced: 'text-blue-400 bg-blue-500/20',
    normal: 'text-amber-400 bg-amber-500/20',
    degraded: 'text-red-400 bg-red-500/20',
    unknown: 'text-slate-400 bg-slate-500/20',
  };

  return (
    <div className="flex items-center gap-2">
      <div className="flex items-center gap-1">
        <span className="text-xs text-blue-400">X:{(metrics.coherence_x * 100).toFixed(0)}</span>
        <span className="text-xs text-green-400">Y:{(metrics.complexity_y * 100).toFixed(0)}</span>
        <span className="text-xs text-amber-400">Z:{(metrics.reflection_z * 100).toFixed(0)}</span>
        <span className="text-xs text-purple-400">A:{(metrics.attunement_a * 100).toFixed(0)}</span>
      </div>
      <span className={`px-2 py-0.5 rounded text-xs font-medium ${levelColors[level]}`}>
        {(combined * 100).toFixed(0)}%
      </span>
    </div>
  );
}

export default XYZARadar;
