'use client';

import { useMemo } from 'react';

/**
 * Confidence Display with Uncertainty Interval
 *
 * Anti-Complaint Spec Section 6.1 Implementation:
 * - Score + uncertainty interval (e.g., "78% +/- 12%")
 * - Visual representation of confidence
 * - Color coding for quick assessment
 */

interface ConfidenceDisplayProps {
  confidence: number;          // 0-1
  interval?: [number, number]; // [lower, upper] bounds, 0-1
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  showBar?: boolean;
  className?: string;
}

export default function ConfidenceDisplay({
  confidence,
  interval,
  size = 'md',
  showLabel = true,
  showBar = true,
  className = '',
}: ConfidenceDisplayProps) {
  const { lower, upper, uncertainty, color, bgColor, label } = useMemo(() => {
    // Calculate interval
    const lo = interval?.[0] ?? confidence - 0.1;
    const up = interval?.[1] ?? confidence + 0.1;
    const unc = ((up - lo) / 2) * 100;

    // Determine color based on confidence
    let col: string;
    let bg: string;
    let lbl: string;

    if (confidence >= 0.8) {
      col = 'text-green-400';
      bg = 'bg-green-500';
      lbl = 'High';
    } else if (confidence >= 0.6) {
      col = 'text-amber-400';
      bg = 'bg-amber-500';
      lbl = 'Medium';
    } else {
      col = 'text-red-400';
      bg = 'bg-red-500';
      lbl = 'Low';
    }

    return {
      lower: lo,
      upper: up,
      uncertainty: unc,
      color: col,
      bgColor: bg,
      label: lbl,
    };
  }, [confidence, interval]);

  const confidencePercent = (confidence * 100).toFixed(0);
  const uncertaintyStr = uncertainty.toFixed(0);

  const sizeClasses = {
    sm: {
      text: 'text-sm',
      interval: 'text-xs',
      bar: 'h-1',
    },
    md: {
      text: 'text-lg font-medium',
      interval: 'text-sm',
      bar: 'h-1.5',
    },
    lg: {
      text: 'text-2xl font-bold',
      interval: 'text-base',
      bar: 'h-2',
    },
  };

  const classes = sizeClasses[size];

  return (
    <div className={`${className}`}>
      <div className="flex items-baseline gap-2">
        {/* Main confidence value */}
        <span className={`${classes.text} ${color}`}>
          {confidencePercent}%
        </span>

        {/* Uncertainty interval */}
        {interval && (
          <span className={`${classes.interval} text-slate-500`}>
            +/- {uncertaintyStr}%
          </span>
        )}

        {/* Label */}
        {showLabel && (
          <span className={`${classes.interval} ${color} opacity-80`}>
            ({label})
          </span>
        )}
      </div>

      {/* Visual bar */}
      {showBar && (
        <div className="mt-2 relative">
          {/* Background track */}
          <div className={`w-full ${classes.bar} bg-slate-700 rounded-full overflow-hidden`}>
            {/* Confidence fill */}
            <div
              className={`${classes.bar} ${bgColor} rounded-full transition-all`}
              style={{ width: `${confidence * 100}%` }}
            />
          </div>

          {/* Uncertainty range indicator */}
          {interval && (
            <div
              className="absolute top-1/2 -translate-y-1/2 h-3 border-l border-r border-white/30 bg-white/5 rounded"
              style={{
                left: `${lower * 100}%`,
                width: `${(upper - lower) * 100}%`,
              }}
            />
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Compact confidence badge for inline use
 */
export function ConfidenceBadge({
  confidence,
  interval,
}: {
  confidence: number;
  interval?: [number, number];
}) {
  const { color, bgLight } = useMemo(() => {
    if (confidence >= 0.8) {
      return { color: 'text-green-400', bgLight: 'bg-green-500/10' };
    } else if (confidence >= 0.6) {
      return { color: 'text-amber-400', bgLight: 'bg-amber-500/10' };
    } else {
      return { color: 'text-red-400', bgLight: 'bg-red-500/10' };
    }
  }, [confidence]);

  const percent = (confidence * 100).toFixed(0);
  const uncertainty = interval
    ? ((interval[1] - interval[0]) / 2 * 100).toFixed(0)
    : null;

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded ${bgLight}`}>
      <span className={`text-sm font-medium ${color}`}>{percent}%</span>
      {uncertainty && (
        <span className="text-xs text-slate-500">+/-{uncertainty}%</span>
      )}
    </span>
  );
}

/**
 * Confidence with calibration info
 * Shows if the system is well-calibrated (70% confidence = 70% true)
 */
export function CalibratedConfidence({
  confidence,
  interval,
  calibrationOffset = 0,
  isCalibrated = true,
}: {
  confidence: number;
  interval?: [number, number];
  calibrationOffset?: number; // How far off calibration is (-0.05 to 0.05)
  isCalibrated?: boolean;
}) {
  return (
    <div>
      <ConfidenceDisplay
        confidence={confidence}
        interval={interval}
        showLabel
        showBar
      />

      {/* Calibration status */}
      <div className="mt-1 flex items-center gap-2 text-xs">
        {isCalibrated ? (
          <span className="text-slate-500">
            Calibrated: {(confidence * 100).toFixed(0)}% confidence = ~{(confidence * 100).toFixed(0)}% true
          </span>
        ) : (
          <span className="text-amber-400">
            Under-calibrated by {Math.abs(calibrationOffset * 100).toFixed(0)}%
          </span>
        )}
      </div>
    </div>
  );
}

/**
 * Insufficient evidence display
 * Per spec: Below 60% confidence -> "Insufficient evidence"
 */
export function InsufficientEvidence({ confidence }: { confidence: number }) {
  if (confidence >= 0.6) {
    return <ConfidenceDisplay confidence={confidence} />;
  }

  return (
    <div className="flex items-center gap-2 text-amber-400">
      <svg
        className="w-4 h-4"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
        />
      </svg>
      <span className="font-medium">Insufficient Evidence</span>
      <span className="text-slate-500 text-sm">
        (confidence: {(confidence * 100).toFixed(0)}%)
      </span>
    </div>
  );
}
