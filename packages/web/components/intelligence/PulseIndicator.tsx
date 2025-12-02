'use client';

import { useEffect, useState } from 'react';

type PulseState = 'realtime' | 'lagging' | 'stale' | 'disconnected';

interface PulseIndicatorProps {
  lastMessageTimestamp: number | null;
  label?: string;
  showLatency?: boolean;
}

function useStaleness(lastMessageTimestamp: number | null) {
  const [state, setState] = useState<PulseState>('disconnected');
  const [latencyMs, setLatencyMs] = useState<number>(0);

  useEffect(() => {
    if (!lastMessageTimestamp) {
      setState('disconnected');
      return;
    }

    const checkStaleness = () => {
      const now = Date.now();
      const delta = now - lastMessageTimestamp;
      setLatencyMs(delta);

      if (delta < 1000) {
        setState('realtime');
      } else if (delta < 10000) {
        setState('lagging');
      } else if (delta < 60000) {
        setState('stale');
      } else {
        setState('disconnected');
      }
    };

    checkStaleness();
    const timer = setInterval(checkStaleness, 500);
    return () => clearInterval(timer);
  }, [lastMessageTimestamp]);

  return { state, latencyMs };
}

const stateConfig: Record<PulseState, {
  color: string;
  bgColor: string;
  label: string;
  animationDuration: string;
  glowColor: string;
}> = {
  realtime: {
    color: '#22c55e',
    bgColor: 'rgba(34, 197, 94, 0.2)',
    label: 'LIVE',
    animationDuration: '1s',
    glowColor: 'rgba(34, 197, 94, 0.5)',
  },
  lagging: {
    color: '#f59e0b',
    bgColor: 'rgba(245, 158, 11, 0.2)',
    label: 'DELAYED',
    animationDuration: '2s',
    glowColor: 'rgba(245, 158, 11, 0.3)',
  },
  stale: {
    color: '#6b7280',
    bgColor: 'rgba(107, 114, 128, 0.2)',
    label: 'STALE',
    animationDuration: '0s',
    glowColor: 'transparent',
  },
  disconnected: {
    color: '#ef4444',
    bgColor: 'rgba(239, 68, 68, 0.2)',
    label: 'OFFLINE',
    animationDuration: '0s',
    glowColor: 'transparent',
  },
};

export function PulseIndicator({
  lastMessageTimestamp,
  label,
  showLatency = false,
}: PulseIndicatorProps) {
  const { state, latencyMs } = useStaleness(lastMessageTimestamp);
  const config = stateConfig[state];

  const formatLatency = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${Math.floor(ms / 60000)}m`;
  };

  return (
    <div className="flex items-center gap-2">
      {/* Pulsing orb */}
      <div className="relative">
        <div
          className="w-3 h-3 rounded-full"
          style={{
            backgroundColor: config.color,
            boxShadow: `0 0 10px ${config.glowColor}`,
          }}
        />
        {state !== 'stale' && state !== 'disconnected' && (
          <div
            className="absolute inset-0 w-3 h-3 rounded-full animate-ping"
            style={{
              backgroundColor: config.color,
              animationDuration: config.animationDuration,
            }}
          />
        )}
      </div>

      {/* Label */}
      <span
        className="text-xs font-mono font-medium tracking-wider"
        style={{ color: config.color }}
      >
        {label || config.label}
      </span>

      {/* Latency */}
      {showLatency && state !== 'disconnected' && (
        <span className="text-xs font-mono text-slate-500">
          ({formatLatency(latencyMs)})
        </span>
      )}

      {/* Strikethrough for stale/disconnected */}
      {(state === 'stale' || state === 'disconnected') && (
        <div
          className="absolute left-0 right-0 h-px top-1/2"
          style={{ backgroundColor: config.color }}
        />
      )}
    </div>
  );
}

// Larger variant for dashboard headers
export function PulseIndicatorLarge({
  lastMessageTimestamp,
  datasetName,
}: {
  lastMessageTimestamp: number | null;
  datasetName: string;
}) {
  const { state, latencyMs } = useStaleness(lastMessageTimestamp);
  const config = stateConfig[state];

  return (
    <div
      className="flex items-center gap-3 px-4 py-2 rounded-lg border"
      style={{
        backgroundColor: config.bgColor,
        borderColor: `${config.color}40`,
      }}
    >
      {/* Animated orb */}
      <div className="relative w-4 h-4">
        <div
          className="absolute inset-0 rounded-full"
          style={{
            backgroundColor: config.color,
            boxShadow: `0 0 15px ${config.glowColor}`,
          }}
        />
        {state === 'realtime' && (
          <>
            <div
              className="absolute inset-0 rounded-full animate-ping opacity-75"
              style={{ backgroundColor: config.color }}
            />
            <div
              className="absolute -inset-1 rounded-full animate-pulse opacity-30"
              style={{ backgroundColor: config.color }}
            />
          </>
        )}
      </div>

      {/* Dataset info */}
      <div className="flex flex-col">
        <span className="text-sm font-medium text-slate-200">
          {datasetName}
        </span>
        <span
          className="text-xs font-mono"
          style={{ color: config.color }}
        >
          {config.label}
          {latencyMs > 0 && state !== 'disconnected' && (
            <span className="text-slate-500 ml-2">
              Last update: {latencyMs < 1000 ? 'just now' : `${Math.floor(latencyMs / 1000)}s ago`}
            </span>
          )}
        </span>
      </div>
    </div>
  );
}

// Hook for external use
export { useStaleness };
