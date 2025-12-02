'use client';

import { useMemo } from 'react';

interface ConfidenceDonutProps {
  confidence: number; // 0-1
  uncertainty?: number; // Epistemic uncertainty 0-1 (higher = less certain)
  size?: 'sm' | 'md' | 'lg';
  label?: string;
  showValue?: boolean;
  animated?: boolean;
}

// Component 33: Confidence Donut with Uncertainty Texture
export function ConfidenceDonut({
  confidence,
  uncertainty = 0,
  size = 'md',
  label,
  showValue = true,
  animated = true,
}: ConfidenceDonutProps) {
  const dimensions = {
    sm: { size: 60, stroke: 6, fontSize: 12 },
    md: { size: 100, stroke: 10, fontSize: 18 },
    lg: { size: 150, stroke: 14, fontSize: 28 },
  };

  const dim = dimensions[size];
  const radius = (dim.size - dim.stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const fillOffset = circumference * (1 - confidence);

  // Generate uncertainty texture pattern
  const textureId = useMemo(() => `uncertainty-${Math.random().toString(36).slice(2)}`, []);

  // Color based on confidence level
  const confidenceColor = useMemo(() => {
    if (confidence >= 0.8) return { main: '#22c55e', glow: 'rgba(34, 197, 94, 0.4)' };
    if (confidence >= 0.6) return { main: '#eab308', glow: 'rgba(234, 179, 8, 0.4)' };
    if (confidence >= 0.4) return { main: '#f97316', glow: 'rgba(249, 115, 22, 0.4)' };
    return { main: '#ef4444', glow: 'rgba(239, 68, 68, 0.4)' };
  }, [confidence]);

  // Uncertainty affects the texture density
  const textureOpacity = uncertainty * 0.6;
  const textureScale = 1 + uncertainty * 2;

  return (
    <div className="flex flex-col items-center">
      {label && (
        <span className="text-xs text-slate-500 uppercase tracking-wider mb-2">
          {label}
        </span>
      )}

      <div className="relative" style={{ width: dim.size, height: dim.size }}>
        <svg
          width={dim.size}
          height={dim.size}
          viewBox={`0 0 ${dim.size} ${dim.size}`}
          className="transform -rotate-90"
        >
          {/* Uncertainty texture pattern */}
          <defs>
            <pattern
              id={textureId}
              width={4 * textureScale}
              height={4 * textureScale}
              patternUnits="userSpaceOnUse"
            >
              <circle cx={2} cy={2} r={1} fill="currentColor" opacity={textureOpacity} />
            </pattern>

            {/* Gradient for the fill */}
            <linearGradient id={`gradient-${textureId}`} x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor={confidenceColor.main} />
              <stop offset="100%" stopColor={confidenceColor.main} stopOpacity={0.7} />
            </linearGradient>

            {/* Glow filter */}
            <filter id={`glow-${textureId}`} x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Background circle */}
          <circle
            cx={dim.size / 2}
            cy={dim.size / 2}
            r={radius}
            fill="none"
            stroke="#1e293b"
            strokeWidth={dim.stroke}
          />

          {/* Uncertainty texture overlay on background */}
          {uncertainty > 0.1 && (
            <circle
              cx={dim.size / 2}
              cy={dim.size / 2}
              r={radius}
              fill="none"
              stroke={`url(#${textureId})`}
              strokeWidth={dim.stroke}
              className="text-slate-500"
            />
          )}

          {/* Confidence fill */}
          <circle
            cx={dim.size / 2}
            cy={dim.size / 2}
            r={radius}
            fill="none"
            stroke={`url(#gradient-${textureId})`}
            strokeWidth={dim.stroke}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={fillOffset}
            filter={`url(#glow-${textureId})`}
            className={animated ? 'transition-all duration-1000 ease-out' : ''}
            style={{
              filter: `drop-shadow(0 0 ${dim.stroke}px ${confidenceColor.glow})`,
            }}
          />

          {/* Uncertainty sketch effect on fill */}
          {uncertainty > 0.2 && (
            <circle
              cx={dim.size / 2}
              cy={dim.size / 2}
              r={radius}
              fill="none"
              stroke={`url(#${textureId})`}
              strokeWidth={dim.stroke}
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={fillOffset}
              className="text-white"
            />
          )}
        </svg>

        {/* Center content */}
        {showValue && (
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span
              className="font-bold font-mono"
              style={{ fontSize: dim.fontSize, color: confidenceColor.main }}
            >
              {(confidence * 100).toFixed(0)}%
            </span>
            {uncertainty > 0.1 && size !== 'sm' && (
              <span className="text-xs text-slate-500">
                ±{(uncertainty * 100).toFixed(0)}%
              </span>
            )}
          </div>
        )}
      </div>

      {/* Uncertainty label */}
      {uncertainty > 0.3 && size !== 'sm' && (
        <div className="mt-2 px-2 py-1 bg-slate-800 rounded text-xs text-slate-400">
          ⚠️ High Uncertainty
        </div>
      )}
    </div>
  );
}

// Inline confidence bar variant
export function ConfidenceBar({
  confidence,
  uncertainty = 0,
  label,
  showValue = true,
}: {
  confidence: number;
  uncertainty?: number;
  label?: string;
  showValue?: boolean;
}) {
  const color = confidence >= 0.8 ? 'bg-green-500' :
                confidence >= 0.6 ? 'bg-yellow-500' :
                confidence >= 0.4 ? 'bg-orange-500' : 'bg-red-500';

  return (
    <div className="flex items-center gap-3">
      {label && (
        <span className="text-xs text-slate-400 w-20 truncate">{label}</span>
      )}
      <div className="flex-1 relative h-2 bg-slate-800 rounded-full overflow-hidden">
        {/* Main fill */}
        <div
          className={`h-full ${color} transition-all duration-500`}
          style={{ width: `${confidence * 100}%` }}
        />
        {/* Uncertainty range */}
        {uncertainty > 0.1 && (
          <>
            <div
              className="absolute top-0 bottom-0 bg-white/20"
              style={{
                left: `${Math.max(0, (confidence - uncertainty) * 100)}%`,
                width: `${uncertainty * 200}%`,
              }}
            />
          </>
        )}
      </div>
      {showValue && (
        <span className="text-xs font-mono text-slate-400 w-10 text-right">
          {(confidence * 100).toFixed(0)}%
        </span>
      )}
    </div>
  );
}

// Multiple model confidence comparison
export function ConfidenceComparison({
  models,
}: {
  models: { name: string; confidence: number; uncertainty?: number }[];
}) {
  const avgConfidence = models.reduce((a, m) => a + m.confidence, 0) / models.length;
  const agreement = 1 - (models.reduce((a, m) =>
    a + Math.abs(m.confidence - avgConfidence), 0) / models.length);

  return (
    <div className="bg-slate-900/50 rounded-lg border border-slate-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm text-slate-300">Model Confidence</span>
        <span className={`text-xs px-2 py-0.5 rounded ${
          agreement > 0.8 ? 'bg-green-500/20 text-green-400' :
          agreement > 0.5 ? 'bg-yellow-500/20 text-yellow-400' :
          'bg-red-500/20 text-red-400'
        }`}>
          {(agreement * 100).toFixed(0)}% Agreement
        </span>
      </div>
      <div className="space-y-2">
        {models.map(model => (
          <ConfidenceBar
            key={model.name}
            confidence={model.confidence}
            uncertainty={model.uncertainty}
            label={model.name}
          />
        ))}
      </div>
      <div className="mt-3 pt-3 border-t border-slate-800">
        <ConfidenceBar
          confidence={avgConfidence}
          label="Ensemble"
          showValue
        />
      </div>
    </div>
  );
}
