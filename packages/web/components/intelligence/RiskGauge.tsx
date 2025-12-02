'use client';

import { useMemo } from 'react';
import { colors, getRiskLabel } from '@/lib/design-system';

interface RiskGaugeProps {
  score: number; // 0-1
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  animated?: boolean;
}

// Component 18: Risk "Weather" Gauge
export function RiskGauge({
  score,
  label = 'Threat Level',
  size = 'md',
  showLabel = true,
  animated = true,
}: RiskGaugeProps) {
  const risk = getRiskLabel(score);

  const dimensions = {
    sm: { width: 120, height: 70, strokeWidth: 8, fontSize: 14 },
    md: { width: 180, height: 100, strokeWidth: 12, fontSize: 20 },
    lg: { width: 240, height: 130, strokeWidth: 16, fontSize: 28 },
  };

  const dim = dimensions[size];
  const radius = (dim.width - dim.strokeWidth) / 2;
  const circumference = Math.PI * radius; // Half circle

  // Calculate the dash offset for the fill
  const fillOffset = circumference * (1 - score);

  // Gradient stops based on score
  const gradientId = `risk-gradient-${Math.random().toString(36).slice(2)}`;

  return (
    <div className="flex flex-col items-center">
      {showLabel && (
        <span className="text-xs text-slate-500 uppercase tracking-wider mb-2">
          {label}
        </span>
      )}

      <div className="relative" style={{ width: dim.width, height: dim.height }}>
        <svg
          width={dim.width}
          height={dim.height}
          viewBox={`0 0 ${dim.width} ${dim.height}`}
          className="overflow-visible"
        >
          {/* Gradient definition */}
          <defs>
            <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor={colors.risk.low} />
              <stop offset="40%" stopColor={colors.risk.elevated} />
              <stop offset="70%" stopColor={colors.risk.high} />
              <stop offset="100%" stopColor={colors.risk.critical} />
            </linearGradient>

            {/* Glow filter */}
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Background arc */}
          <path
            d={describeArc(dim.width / 2, dim.height, radius, 180, 360)}
            fill="none"
            stroke="#1e293b"
            strokeWidth={dim.strokeWidth}
            strokeLinecap="round"
          />

          {/* Tick marks */}
          {[0, 0.25, 0.5, 0.75, 1].map((tick, i) => {
            const angle = 180 + tick * 180;
            const innerR = radius - dim.strokeWidth / 2 - 4;
            const outerR = radius - dim.strokeWidth / 2 - 10;
            const x1 = dim.width / 2 + innerR * Math.cos((angle * Math.PI) / 180);
            const y1 = dim.height + innerR * Math.sin((angle * Math.PI) / 180);
            const x2 = dim.width / 2 + outerR * Math.cos((angle * Math.PI) / 180);
            const y2 = dim.height + outerR * Math.sin((angle * Math.PI) / 180);
            return (
              <line
                key={i}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="#475569"
                strokeWidth={1}
              />
            );
          })}

          {/* Filled arc */}
          <path
            d={describeArc(dim.width / 2, dim.height, radius, 180, 180 + score * 180)}
            fill="none"
            stroke={`url(#${gradientId})`}
            strokeWidth={dim.strokeWidth}
            strokeLinecap="round"
            filter="url(#glow)"
            className={animated ? 'transition-all duration-1000 ease-out' : ''}
            style={{
              strokeDasharray: circumference,
              strokeDashoffset: animated ? 0 : fillOffset,
            }}
          />

          {/* Needle */}
          <g
            transform={`rotate(${180 + score * 180}, ${dim.width / 2}, ${dim.height})`}
            className={animated ? 'transition-transform duration-1000 ease-out' : ''}
          >
            <line
              x1={dim.width / 2}
              y1={dim.height}
              x2={dim.width / 2}
              y2={dim.height - radius + dim.strokeWidth + 5}
              stroke={risk.color}
              strokeWidth={2}
              strokeLinecap="round"
            />
            <circle
              cx={dim.width / 2}
              cy={dim.height}
              r={4}
              fill={risk.color}
            />
          </g>
        </svg>

        {/* Center label */}
        <div
          className="absolute left-1/2 -translate-x-1/2 text-center"
          style={{ bottom: -dim.fontSize * 0.5 }}
        >
          <div
            className="font-bold font-mono"
            style={{ fontSize: dim.fontSize, color: risk.color }}
          >
            {(score * 100).toFixed(0)}%
          </div>
          <div
            className="text-xs font-medium uppercase tracking-wider"
            style={{ color: risk.color }}
          >
            {risk.level}
          </div>
        </div>
      </div>
    </div>
  );
}

// Helper to create SVG arc path
function describeArc(
  x: number,
  y: number,
  radius: number,
  startAngle: number,
  endAngle: number
): string {
  const start = polarToCartesian(x, y, radius, endAngle);
  const end = polarToCartesian(x, y, radius, startAngle);
  const largeArcFlag = endAngle - startAngle <= 180 ? 0 : 1;

  return [
    'M', start.x, start.y,
    'A', radius, radius, 0, largeArcFlag, 0, end.x, end.y,
  ].join(' ');
}

function polarToCartesian(
  centerX: number,
  centerY: number,
  radius: number,
  angleInDegrees: number
) {
  const angleInRadians = ((angleInDegrees - 90) * Math.PI) / 180.0;
  return {
    x: centerX + radius * Math.cos(angleInRadians),
    y: centerY + radius * Math.sin(angleInRadians),
  };
}

// Mini inline variant for tables/cards
export function RiskGaugeMini({ score }: { score: number }) {
  const risk = getRiskLabel(score);
  const pct = score * 100;

  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
        <div
          className="h-full transition-all duration-500"
          style={{
            width: `${pct}%`,
            backgroundColor: risk.color,
            boxShadow: `0 0 8px ${risk.color}`,
          }}
        />
      </div>
      <span
        className="text-xs font-mono font-bold"
        style={{ color: risk.color }}
      >
        {pct.toFixed(0)}
      </span>
    </div>
  );
}

// DEFCON-style status display
export function DefconGauge({
  level,
  label = 'DEFCON',
}: {
  level: 1 | 2 | 3 | 4 | 5;
  label?: string;
}) {
  const config = {
    1: { color: '#ef4444', label: 'MAXIMUM', sublabel: 'Nuclear war imminent' },
    2: { color: '#f97316', label: 'INCREASED', sublabel: 'Prepare for war' },
    3: { color: '#eab308', label: 'ELEVATED', sublabel: 'Increase readiness' },
    4: { color: '#3b82f6', label: 'ABOVE NORMAL', sublabel: 'Enhanced intelligence' },
    5: { color: '#22c55e', label: 'LOWEST', sublabel: 'Normal readiness' },
  };

  const current = config[level];

  return (
    <div className="bg-slate-900/80 border border-slate-700 rounded-lg p-4">
      <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">
        {label}
      </div>

      <div className="flex items-end gap-1 mb-2">
        {[5, 4, 3, 2, 1].map(l => (
          <div
            key={l}
            className={`
              w-6 transition-all duration-300
              ${l <= level ? 'opacity-30' : ''}
            `}
            style={{
              height: `${(6 - l) * 8}px`,
              backgroundColor: config[l as 1 | 2 | 3 | 4 | 5].color,
              boxShadow: l === level ? `0 0 15px ${current.color}` : 'none',
            }}
          />
        ))}
      </div>

      <div className="flex items-center gap-2">
        <span
          className="text-2xl font-bold font-mono"
          style={{ color: current.color }}
        >
          {level}
        </span>
        <div>
          <div
            className="text-sm font-bold"
            style={{ color: current.color }}
          >
            {current.label}
          </div>
          <div className="text-xs text-slate-500">
            {current.sublabel}
          </div>
        </div>
      </div>
    </div>
  );
}
