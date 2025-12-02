'use client';

import { useMemo } from 'react';

interface DataPoint {
  timestamp: number;
  value: number;
  predicted?: boolean;
  upperBound?: number;
  lowerBound?: number;
}

interface ConfidenceConeProps {
  data: DataPoint[];
  width?: number;
  height?: number;
  showGrid?: boolean;
  label?: string;
  unit?: string;
  predictionStartIndex?: number; // Where predictions begin
}

// Component 25: Prediction Confidence Cone
export function ConfidenceCone({
  data,
  width = 600,
  height = 200,
  showGrid = true,
  label,
  unit = '',
  predictionStartIndex,
}: ConfidenceConeProps) {
  const padding = { top: 20, right: 40, bottom: 30, left: 50 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Calculate scales
  const { xScale, yScale, paths } = useMemo(() => {
    if (data.length === 0) return { xScale: () => 0, yScale: () => 0, paths: {} };

    const timestamps = data.map(d => d.timestamp);
    const values = data.flatMap(d => [d.value, d.upperBound, d.lowerBound].filter(Boolean) as number[]);

    const xMin = Math.min(...timestamps);
    const xMax = Math.max(...timestamps);
    const yMin = Math.min(...values) * 0.95;
    const yMax = Math.max(...values) * 1.05;

    const xScale = (t: number) => ((t - xMin) / (xMax - xMin)) * chartWidth + padding.left;
    const yScale = (v: number) => chartHeight - ((v - yMin) / (yMax - yMin)) * chartHeight + padding.top;

    // Find prediction start
    const predStart = predictionStartIndex ?? data.findIndex(d => d.predicted);
    const historicalData = predStart > 0 ? data.slice(0, predStart + 1) : data;
    const predictionData = predStart > 0 ? data.slice(predStart) : [];

    // Build SVG paths
    const historicalPath = historicalData
      .map((d, i) => `${i === 0 ? 'M' : 'L'} ${xScale(d.timestamp)} ${yScale(d.value)}`)
      .join(' ');

    const predictionPath = predictionData
      .map((d, i) => `${i === 0 ? 'M' : 'L'} ${xScale(d.timestamp)} ${yScale(d.value)}`)
      .join(' ');

    // Confidence cone (upper and lower bounds)
    const conePoints = predictionData.filter(d => d.upperBound && d.lowerBound);
    const upperPath = conePoints
      .map((d, i) => `${i === 0 ? 'M' : 'L'} ${xScale(d.timestamp)} ${yScale(d.upperBound!)}`)
      .join(' ');
    const lowerPath = conePoints
      .reverse()
      .map((d, i) => `${i === 0 ? 'L' : 'L'} ${xScale(d.timestamp)} ${yScale(d.lowerBound!)}`)
      .join(' ');
    const conePath = conePoints.length > 0 ? `${upperPath} ${lowerPath} Z` : '';

    return {
      xScale,
      yScale,
      paths: {
        historical: historicalPath,
        prediction: predictionPath,
        cone: conePath,
        yMin,
        yMax,
        xMin,
        xMax,
        predStart: predStart > 0 ? xScale(data[predStart].timestamp) : null,
      },
    };
  }, [data, chartWidth, chartHeight, padding, predictionStartIndex]);

  // Format axis labels
  const formatTime = (ts: number) => {
    const date = new Date(ts);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  const formatValue = (v: number) => {
    if (Math.abs(v) >= 1000000) return `${(v / 1000000).toFixed(1)}M`;
    if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}K`;
    return v.toFixed(1);
  };

  return (
    <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-4">
      {label && (
        <div className="mb-4 flex items-center justify-between">
          <span className="text-sm font-medium text-slate-300">{label}</span>
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1">
              <div className="w-4 h-0.5 bg-amber-500" />
              <span className="text-slate-500">Historical</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-4 h-0.5 bg-cyan-500" style={{ strokeDasharray: '4 2' }} />
              <span className="text-slate-500">Predicted</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-4 h-3 bg-cyan-500/20 rounded" />
              <span className="text-slate-500">Confidence</span>
            </div>
          </div>
        </div>
      )}

      <svg width={width} height={height}>
        {/* Grid */}
        {showGrid && (
          <g className="text-slate-800">
            {/* Horizontal grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map(pct => {
              const y = padding.top + chartHeight * pct;
              return (
                <line
                  key={`h-${pct}`}
                  x1={padding.left}
                  y1={y}
                  x2={width - padding.right}
                  y2={y}
                  stroke="currentColor"
                  strokeWidth={0.5}
                />
              );
            })}
            {/* Vertical grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map(pct => {
              const x = padding.left + chartWidth * pct;
              return (
                <line
                  key={`v-${pct}`}
                  x1={x}
                  y1={padding.top}
                  x2={x}
                  y2={height - padding.bottom}
                  stroke="currentColor"
                  strokeWidth={0.5}
                />
              );
            })}
          </g>
        )}

        {/* Prediction zone background */}
        {paths.predStart && (
          <rect
            x={paths.predStart}
            y={padding.top}
            width={width - padding.right - paths.predStart}
            height={chartHeight}
            fill="rgba(6, 182, 212, 0.05)"
          />
        )}

        {/* Prediction boundary line */}
        {paths.predStart && (
          <line
            x1={paths.predStart}
            y1={padding.top}
            x2={paths.predStart}
            y2={height - padding.bottom}
            stroke="#06b6d4"
            strokeWidth={1}
            strokeDasharray="4 4"
          />
        )}

        {/* Confidence cone */}
        {paths.cone && (
          <path
            d={paths.cone}
            fill="url(#cone-gradient)"
            opacity={0.3}
          />
        )}

        {/* Gradient definition */}
        <defs>
          <linearGradient id="cone-gradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#06b6d4" stopOpacity={0.4} />
            <stop offset="50%" stopColor="#06b6d4" stopOpacity={0.1} />
            <stop offset="100%" stopColor="#06b6d4" stopOpacity={0.4} />
          </linearGradient>
        </defs>

        {/* Historical line */}
        {paths.historical && (
          <path
            d={paths.historical}
            fill="none"
            stroke="#f59e0b"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}

        {/* Prediction line */}
        {paths.prediction && (
          <path
            d={paths.prediction}
            fill="none"
            stroke="#06b6d4"
            strokeWidth={2}
            strokeDasharray="6 3"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{
              filter: 'drop-shadow(0 0 4px rgba(6, 182, 212, 0.5))',
            }}
          />
        )}

        {/* Data points */}
        {data.map((d, i) => (
          <circle
            key={i}
            cx={xScale(d.timestamp)}
            cy={yScale(d.value)}
            r={3}
            fill={d.predicted ? '#06b6d4' : '#f59e0b'}
            className="transition-all hover:r-5"
          />
        ))}

        {/* Y-axis labels */}
        {paths.yMin !== undefined && paths.yMax !== undefined && (
          <g className="text-xs fill-slate-500">
            <text x={padding.left - 8} y={padding.top + 4} textAnchor="end">
              {formatValue(paths.yMax)}{unit}
            </text>
            <text x={padding.left - 8} y={height - padding.bottom} textAnchor="end">
              {formatValue(paths.yMin)}{unit}
            </text>
          </g>
        )}

        {/* X-axis labels */}
        {paths.xMin !== undefined && paths.xMax !== undefined && (
          <g className="text-xs fill-slate-500">
            <text x={padding.left} y={height - 8} textAnchor="start">
              {formatTime(paths.xMin)}
            </text>
            <text x={width - padding.right} y={height - 8} textAnchor="end">
              {formatTime(paths.xMax)}
            </text>
            {paths.predStart && (
              <text x={paths.predStart} y={height - 8} textAnchor="middle" className="fill-cyan-500">
                NOW
              </text>
            )}
          </g>
        )}

        {/* "FORECAST" label */}
        {paths.predStart && (
          <text
            x={paths.predStart + 10}
            y={padding.top + 15}
            className="text-xs fill-cyan-500 font-mono"
          >
            FORECAST →
          </text>
        )}
      </svg>
    </div>
  );
}

// Generate mock forecast data
export function generateMockForecast(hours: number = 24): DataPoint[] {
  const now = Date.now();
  const hourMs = 60 * 60 * 1000;
  const data: DataPoint[] = [];

  let value = 50 + Math.random() * 20;
  const splitPoint = Math.floor(hours * 0.6);

  for (let i = 0; i < hours; i++) {
    const timestamp = now - (hours - i - 1) * hourMs;
    const isPredicted = i >= splitPoint;
    const delta = (Math.random() - 0.5) * 5;
    value = Math.max(20, Math.min(100, value + delta));

    if (isPredicted) {
      // Add uncertainty that grows over time
      const uncertainty = (i - splitPoint + 1) * 2;
      data.push({
        timestamp,
        value,
        predicted: true,
        upperBound: value + uncertainty,
        lowerBound: value - uncertainty,
      });
    } else {
      data.push({ timestamp, value, predicted: false });
    }
  }

  return data;
}
