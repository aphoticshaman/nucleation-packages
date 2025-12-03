'use client';

import { useMemo } from 'react';

interface FlowState {
  level: string;           // NONE, EMERGING, BUILDING, FLOW, DEEP_FLOW
  R: number;              // Kuramoto order parameter [0, 1]
  dR_dt?: number;         // Rate of change
  is_flow: boolean;
  is_deep_flow: boolean;
  stability: number;      // [0, 1]
  time_in_state_ms: number;
}

interface FlowStateIndicatorProps {
  flowState: FlowState;
  size?: 'sm' | 'md' | 'lg';
  showDetails?: boolean;
  animated?: boolean;
}

// Flow thresholds from cognitive module
const FLOW_THRESHOLDS = {
  NONE: 0.45,
  EMERGING: 0.65,
  BUILDING: 0.76,
  FLOW: 0.76,
  DEEP_FLOW: 0.88,
};

const LEVEL_CONFIG: Record<string, {
  label: string;
  color: string;
  bgColor: string;
  borderColor: string;
  glowColor: string;
  icon: string;
  description: string;
}> = {
  NONE: {
    label: 'No Flow',
    color: 'text-slate-400',
    bgColor: 'bg-slate-800',
    borderColor: 'border-slate-600',
    glowColor: '',
    icon: '○',
    description: 'Scattered cognitive state',
  },
  EMERGING: {
    label: 'Emerging',
    color: 'text-blue-400',
    bgColor: 'bg-blue-900/30',
    borderColor: 'border-blue-500/50',
    glowColor: 'shadow-blue-500/20',
    icon: '◔',
    description: 'Building coherence',
  },
  BUILDING: {
    label: 'Building',
    color: 'text-cyan-400',
    bgColor: 'bg-cyan-900/30',
    borderColor: 'border-cyan-500/50',
    glowColor: 'shadow-cyan-500/30',
    icon: '◑',
    description: 'Approaching flow state',
  },
  FLOW: {
    label: 'Flow',
    color: 'text-green-400',
    bgColor: 'bg-green-900/30',
    borderColor: 'border-green-500/50',
    glowColor: 'shadow-green-500/40',
    icon: '◕',
    description: 'Synchronized cognition',
  },
  DEEP_FLOW: {
    label: 'Deep Flow',
    color: 'text-purple-400',
    bgColor: 'bg-purple-900/30',
    borderColor: 'border-purple-500/50',
    glowColor: 'shadow-purple-500/50',
    icon: '●',
    description: 'Peak cognitive performance',
  },
};

export function FlowStateIndicator({
  flowState,
  size = 'md',
  showDetails = false,
  animated = true,
}: FlowStateIndicatorProps) {
  const config = LEVEL_CONFIG[flowState.level] || LEVEL_CONFIG.NONE;

  const sizeClasses = {
    sm: 'w-16 h-16',
    md: 'w-24 h-24',
    lg: 'w-32 h-32',
  };

  const textSizes = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base',
  };

  // Calculate ring progress
  const progress = Math.min(flowState.R / 1, 1);
  const circumference = 2 * Math.PI * 45; // radius = 45
  const strokeDashoffset = circumference * (1 - progress);

  // Trend indicator
  const trend = useMemo(() => {
    if (!flowState.dR_dt) return 'stable';
    if (flowState.dR_dt > 0.01) return 'rising';
    if (flowState.dR_dt < -0.01) return 'falling';
    return 'stable';
  }, [flowState.dR_dt]);

  const trendIcon = {
    rising: '↑',
    falling: '↓',
    stable: '→',
  };

  const trendColor = {
    rising: 'text-green-400',
    falling: 'text-red-400',
    stable: 'text-slate-400',
  };

  return (
    <div className="flex flex-col items-center gap-2">
      {/* Circular indicator */}
      <div className={`relative ${sizeClasses[size]}`}>
        <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
          {/* Background circle */}
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke="rgba(148, 163, 184, 0.2)"
            strokeWidth="8"
          />
          {/* Threshold marker at R=0.76 */}
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke="rgba(59, 130, 246, 0.3)"
            strokeWidth="8"
            strokeDasharray={`${circumference * 0.76} ${circumference}`}
          />
          {/* Progress circle */}
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke={flowState.is_flow ? '#22c55e' : flowState.R > 0.5 ? '#3b82f6' : '#64748b'}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            className={animated ? 'transition-all duration-500' : ''}
          />
          {/* Glow effect for flow state */}
          {flowState.is_flow && (
            <circle
              cx="50"
              cy="50"
              r="45"
              fill="none"
              stroke={flowState.is_deep_flow ? '#a855f7' : '#22c55e'}
              strokeWidth="12"
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              opacity="0.3"
              className="animate-pulse"
            />
          )}
        </svg>

        {/* Center content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`font-bold ${config.color} ${size === 'lg' ? 'text-2xl' : 'text-xl'}`}>
            {(flowState.R * 100).toFixed(0)}
          </span>
          <span className="text-[10px] text-slate-500 uppercase tracking-wider">
            R
          </span>
        </div>
      </div>

      {/* Label */}
      <div className="flex items-center gap-1">
        <span className={`${config.icon === '●' && flowState.is_deep_flow ? 'animate-pulse' : ''}`}>
          {config.icon}
        </span>
        <span className={`font-medium ${textSizes[size]} ${config.color}`}>
          {config.label}
        </span>
        <span className={`${trendColor[trend]} ${textSizes[size]}`}>
          {trendIcon[trend]}
        </span>
      </div>

      {/* Details */}
      {showDetails && (
        <div className="text-center space-y-1">
          <p className="text-xs text-slate-500">{config.description}</p>
          <div className="flex items-center gap-3 text-xs">
            <span className="text-slate-400">
              Stability: <span className="text-white">{(flowState.stability * 100).toFixed(0)}%</span>
            </span>
            <span className="text-slate-400">
              Time: <span className="text-white">{(flowState.time_in_state_ms / 1000).toFixed(1)}s</span>
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

// Compact bar version
export function FlowStateBar({ flowState }: { flowState: FlowState }) {
  const config = LEVEL_CONFIG[flowState.level] || LEVEL_CONFIG.NONE;
  const width = `${flowState.R * 100}%`;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className={config.color}>{config.label}</span>
        <span className="text-slate-400">R: {(flowState.R * 100).toFixed(0)}%</span>
      </div>
      <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
        {/* Threshold marker */}
        <div className="relative h-full">
          <div
            className="absolute top-0 bottom-0 w-px bg-blue-500/50"
            style={{ left: '76%' }}
          />
          <div
            className={`h-full transition-all duration-300 ${
              flowState.is_flow ? 'bg-green-500' :
              flowState.R > 0.5 ? 'bg-blue-500' :
              'bg-slate-600'
            }`}
            style={{ width }}
          />
        </div>
      </div>
    </div>
  );
}

// Mini indicator for inline use
export function FlowStateBadge({ flowState }: { flowState: FlowState }) {
  const config = LEVEL_CONFIG[flowState.level] || LEVEL_CONFIG.NONE;

  return (
    <span className={`
      inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium
      ${config.bgColor} ${config.color} border ${config.borderColor}
      ${flowState.is_flow ? 'shadow-lg ' + config.glowColor : ''}
    `}>
      <span>{config.icon}</span>
      <span>R:{(flowState.R * 100).toFixed(0)}</span>
    </span>
  );
}

export default FlowStateIndicator;
