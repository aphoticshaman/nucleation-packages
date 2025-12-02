'use client';

import { useMemo } from 'react';

interface SparklinePoint {
  value: number;
  predicted?: number; // Neural prediction
  timestamp: number;
}

interface VelocitySparklineProps {
  data: SparklinePoint[];
  width?: number;
  height?: number;
  showPrediction?: boolean;
  alertThreshold?: number; // Sigma threshold for alerts
  label?: string;
}

export function VelocitySparkline({
  data,
  width = 120,
  height = 32,
  showPrediction = true,
  alertThreshold = 2,
  label,
}: VelocitySparklineProps) {
  const { path, predictionPath, isAlerting, stats } = useMemo(() => {
    if (data.length < 2) {
      return { path: '', predictionPath: '', isAlerting: false, stats: null };
    }

    const values = data.map(d => d.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    // Calculate mean and std for anomaly detection
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
    const std = Math.sqrt(variance);

    // Check if latest value is anomalous
    const latest = values[values.length - 1];
    const latestPredicted = data[data.length - 1].predicted;
    const isAlerting = latestPredicted !== undefined
      ? Math.abs(latest - latestPredicted) > alertThreshold * std
      : false;

    // Build SVG path for actual values
    const xStep = width / (data.length - 1);
    const points = data.map((d, i) => {
      const x = i * xStep;
      const y = height - ((d.value - min) / range) * height;
      return `${x},${y}`;
    });
    const path = `M ${points.join(' L ')}`;

    // Build prediction path if available
    let predictionPath = '';
    if (showPrediction) {
      const predPoints = data
        .filter(d => d.predicted !== undefined)
        .map((d, i) => {
          const idx = data.indexOf(d);
          const x = idx * xStep;
          const y = height - ((d.predicted! - min) / range) * height;
          return `${x},${y}`;
        });
      if (predPoints.length > 1) {
        predictionPath = `M ${predPoints.join(' L ')}`;
      }
    }

    return {
      path,
      predictionPath,
      isAlerting,
      stats: { latest, mean, std, trend: latest > mean ? 'up' : 'down' },
    };
  }, [data, width, height, showPrediction, alertThreshold]);

  // Calculate fill area between actual and predicted
  const fillPath = useMemo(() => {
    if (!showPrediction || data.length < 2) return '';

    const values = data.map(d => d.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const xStep = width / (data.length - 1);

    const actualPoints = data.map((d, i) => {
      const x = i * xStep;
      const y = height - ((d.value - min) / range) * height;
      return { x, y };
    });

    const predPoints = data.map((d, i) => {
      const x = i * xStep;
      const pred = d.predicted ?? d.value;
      const y = height - ((pred - min) / range) * height;
      return { x, y };
    });

    // Create closed path: actual line forward, predicted line backward
    const forward = actualPoints.map(p => `${p.x},${p.y}`).join(' L ');
    const backward = [...predPoints].reverse().map(p => `${p.x},${p.y}`).join(' L ');

    return `M ${forward} L ${backward} Z`;
  }, [data, width, height, showPrediction]);

  return (
    <div className="inline-flex items-center gap-2">
      {label && (
        <span className="text-xs text-slate-500 w-16 truncate">{label}</span>
      )}

      <div className="relative">
        {/* Alert glow */}
        {isAlerting && (
          <div
            className="absolute inset-0 animate-pulse rounded"
            style={{
              background: 'radial-gradient(ellipse at center, rgba(239,68,68,0.3) 0%, transparent 70%)',
            }}
          />
        )}

        <svg
          width={width}
          height={height}
          className="overflow-visible"
        >
          {/* Divergence fill area */}
          {fillPath && (
            <path
              d={fillPath}
              fill="rgba(6, 182, 212, 0.1)"
              stroke="none"
            />
          )}

          {/* Prediction line (Neural - Cyan, dashed) */}
          {predictionPath && (
            <path
              d={predictionPath}
              fill="none"
              stroke="#06b6d4"
              strokeWidth={1}
              strokeDasharray="3,2"
              opacity={0.5}
            />
          )}

          {/* Actual line (Symbolic - Amber) */}
          <path
            d={path}
            fill="none"
            stroke={isAlerting ? '#ef4444' : '#f59e0b'}
            strokeWidth={1.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Latest point */}
          <circle
            cx={width}
            cy={data.length > 0
              ? height - ((data[data.length - 1].value - Math.min(...data.map(d => d.value))) /
                (Math.max(...data.map(d => d.value)) - Math.min(...data.map(d => d.value)) || 1)) * height
              : height / 2
            }
            r={3}
            fill={isAlerting ? '#ef4444' : '#f59e0b'}
            className={isAlerting ? 'animate-ping' : ''}
          />
        </svg>
      </div>

      {/* Trend indicator */}
      {stats && (
        <span className={`text-xs font-mono ${
          stats.trend === 'up' ? 'text-red-400' : 'text-green-400'
        }`}>
          {stats.trend === 'up' ? '↑' : '↓'}
        </span>
      )}
    </div>
  );
}

// Data table cell variant
export function SparklineCell({
  data,
  label,
}: {
  data: SparklinePoint[];
  label: string;
}) {
  return (
    <div className="flex items-center justify-between gap-4 px-3 py-2 bg-slate-800/50 rounded">
      <span className="text-sm text-slate-300">{label}</span>
      <VelocitySparkline data={data} width={80} height={24} />
    </div>
  );
}

// Generate mock sparkline data
export function generateMockSparklineData(points: number = 20): SparklinePoint[] {
  const now = Date.now();
  const data: SparklinePoint[] = [];
  let value = 50 + Math.random() * 20;

  for (let i = 0; i < points; i++) {
    const delta = (Math.random() - 0.5) * 10;
    value = Math.max(0, Math.min(100, value + delta));

    // Add some prediction divergence near the end
    const predicted = i > points - 5
      ? value + (Math.random() - 0.3) * 15
      : value + (Math.random() - 0.5) * 3;

    data.push({
      value,
      predicted,
      timestamp: now - (points - i) * 60000, // 1 minute intervals
    });
  }

  return data;
}
