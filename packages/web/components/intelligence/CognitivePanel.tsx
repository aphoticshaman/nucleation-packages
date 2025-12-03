'use client';

import { useMemo } from 'react';
import { XYZARadar, XYZAInline } from './XYZARadar';
import { FlowStateIndicator, FlowStateBar, FlowStateBadge } from './FlowStateIndicator';
import { NestedPanel, PanelGrid, TabPanel } from './NestedPanel';

interface CognitiveState {
  xyza: {
    coherence_x: number;
    complexity_y: number;
    reflection_z: number;
    attunement_a: number;
    combined_score: number;
    cognitive_level: string;
  };
  flow_state: {
    level: string;
    R: number;
    dR_dt?: number;
    is_flow: boolean;
    is_deep_flow: boolean;
    stability: number;
    time_in_state_ms: number;
  };
  diagnostics?: string[];
  generation_time_ms?: number;
}

interface CognitivePanelProps {
  state: CognitiveState | null;
  loading?: boolean;
  variant?: 'full' | 'compact' | 'inline';
  showDiagnostics?: boolean;
}

export function CognitivePanel({
  state,
  loading = false,
  variant = 'full',
  showDiagnostics = true,
}: CognitivePanelProps) {
  if (loading) {
    return <CognitivePanelSkeleton variant={variant} />;
  }

  if (!state) {
    return <CognitivePanelEmpty variant={variant} />;
  }

  if (variant === 'inline') {
    return (
      <div className="flex items-center gap-4 flex-wrap">
        <FlowStateBadge flowState={state.flow_state} />
        <XYZAInline metrics={state.xyza} />
      </div>
    );
  }

  if (variant === 'compact') {
    return (
      <div className="bg-slate-900/50 rounded-lg border border-slate-700 p-3 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-slate-300">Cognitive State</span>
          <FlowStateBadge flowState={state.flow_state} />
        </div>
        <FlowStateBar flowState={state.flow_state} />
        <div className="grid grid-cols-4 gap-2 text-center">
          <MetricMini label="X" value={state.xyza.coherence_x} color="blue" />
          <MetricMini label="Y" value={state.xyza.complexity_y} color="green" />
          <MetricMini label="Z" value={state.xyza.reflection_z} color="amber" />
          <MetricMini label="A" value={state.xyza.attunement_a} color="purple" />
        </div>
      </div>
    );
  }

  // Full variant
  return (
    <NestedPanel
      id="cognitive"
      title="Cognitive State"
      icon="🧠"
      badge={state.xyza.cognitive_level}
      badgeColor={getCognitiveLevelColor(state.xyza.cognitive_level)}
      defaultOpen={true}
    >
      <div className="space-y-4">
        {/* Main metrics display */}
        <div className="flex flex-col md:flex-row items-center justify-around gap-6">
          <FlowStateIndicator
            flowState={state.flow_state}
            size="lg"
            showDetails={true}
          />
          <XYZARadar
            metrics={state.xyza}
            size={180}
            showThresholds={true}
          />
        </div>

        {/* Nested detail panels */}
        <PanelGrid cols={2} gap="sm">
          <NestedPanel
            id="flow-details"
            title="Flow Dynamics"
            icon="🌊"
            variant="compact"
            defaultOpen={false}
          >
            <div className="space-y-2 text-sm">
              <MetricRow label="Order Parameter (R)" value={state.flow_state.R} format="percent" />
              <MetricRow label="Rate of Change" value={state.flow_state.dR_dt || 0} format="delta" />
              <MetricRow label="Stability" value={state.flow_state.stability} format="percent" />
              <MetricRow label="Time in State" value={state.flow_state.time_in_state_ms} format="time" />
            </div>
          </NestedPanel>

          <NestedPanel
            id="xyza-details"
            title="XYZA Breakdown"
            icon="📊"
            variant="compact"
            defaultOpen={false}
          >
            <div className="space-y-2 text-sm">
              <MetricRow label="X (Coherence)" value={state.xyza.coherence_x} format="percent" threshold={0.76} />
              <MetricRow label="Y (Complexity)" value={state.xyza.complexity_y} format="percent" optimal={[0.4, 0.7]} />
              <MetricRow label="Z (Reflection)" value={state.xyza.reflection_z} format="percent" threshold={0.5} />
              <MetricRow label="A (Attunement)" value={state.xyza.attunement_a} format="percent" threshold={0.42} />
            </div>
          </NestedPanel>
        </PanelGrid>

        {/* Diagnostics */}
        {showDiagnostics && state.diagnostics && state.diagnostics.length > 0 && (
          <NestedPanel
            id="diagnostics"
            title="Diagnostics"
            icon="🔍"
            variant="compact"
            badge={state.diagnostics.length}
            defaultOpen={false}
          >
            <ul className="space-y-1">
              {state.diagnostics.map((d, i) => (
                <li key={i} className="text-xs text-slate-400 flex items-start gap-2">
                  <span className={getDiagnosticIcon(d)}>{getDiagnosticEmoji(d)}</span>
                  <span>{d}</span>
                </li>
              ))}
            </ul>
          </NestedPanel>
        )}

        {/* Generation metrics */}
        {state.generation_time_ms !== undefined && (
          <div className="text-xs text-slate-500 text-center">
            Generated in {state.generation_time_ms.toFixed(0)}ms
          </div>
        )}
      </div>
    </NestedPanel>
  );
}

// Helper components
function MetricMini({ label, value, color }: { label: string; value: number; color: string }) {
  const colorClasses = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    amber: 'text-amber-400',
    purple: 'text-purple-400',
  };

  return (
    <div>
      <div className={`text-lg font-bold ${colorClasses[color as keyof typeof colorClasses]}`}>
        {(value * 100).toFixed(0)}
      </div>
      <div className="text-[10px] text-slate-500">{label}</div>
    </div>
  );
}

function MetricRow({
  label,
  value,
  format,
  threshold,
  optimal,
}: {
  label: string;
  value: number;
  format: 'percent' | 'delta' | 'time';
  threshold?: number;
  optimal?: [number, number];
}) {
  let displayValue: string;
  let statusColor = 'text-slate-300';

  switch (format) {
    case 'percent':
      displayValue = `${(value * 100).toFixed(1)}%`;
      if (threshold && value >= threshold) statusColor = 'text-green-400';
      else if (threshold && value < threshold * 0.8) statusColor = 'text-amber-400';
      if (optimal) {
        if (value >= optimal[0] && value <= optimal[1]) statusColor = 'text-green-400';
        else statusColor = 'text-amber-400';
      }
      break;
    case 'delta':
      displayValue = value >= 0 ? `+${value.toFixed(4)}` : value.toFixed(4);
      statusColor = value > 0 ? 'text-green-400' : value < 0 ? 'text-red-400' : 'text-slate-400';
      break;
    case 'time':
      displayValue = value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${value.toFixed(0)}ms`;
      break;
    default:
      displayValue = String(value);
  }

  return (
    <div className="flex justify-between items-center">
      <span className="text-slate-400">{label}</span>
      <span className={`font-mono ${statusColor}`}>{displayValue}</span>
    </div>
  );
}

function CognitivePanelSkeleton({ variant }: { variant: string }) {
  if (variant === 'inline') {
    return (
      <div className="flex items-center gap-4">
        <div className="w-20 h-6 bg-slate-800 rounded animate-pulse" />
        <div className="w-32 h-6 bg-slate-800 rounded animate-pulse" />
      </div>
    );
  }

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-700 p-4">
      <div className="h-6 w-32 bg-slate-800 rounded animate-pulse mb-4" />
      <div className="flex justify-around gap-6">
        <div className="w-24 h-24 bg-slate-800 rounded-full animate-pulse" />
        <div className="w-40 h-40 bg-slate-800 rounded animate-pulse" />
      </div>
    </div>
  );
}

function CognitivePanelEmpty({ variant }: { variant: string }) {
  if (variant === 'inline') {
    return <span className="text-xs text-slate-500">No cognitive data</span>;
  }

  return (
    <div className="bg-slate-900/50 rounded-xl border border-slate-700 p-6 text-center">
      <div className="text-4xl mb-2">🧠</div>
      <p className="text-slate-400">Cognitive state not available</p>
      <p className="text-xs text-slate-500 mt-1">Generate analysis to see metrics</p>
    </div>
  );
}

function getCognitiveLevelColor(level: string): 'green' | 'blue' | 'amber' | 'red' | 'slate' {
  switch (level?.toLowerCase()) {
    case 'peak': return 'green';
    case 'enhanced': return 'blue';
    case 'normal': return 'amber';
    case 'degraded': return 'red';
    default: return 'slate';
  }
}

function getDiagnosticEmoji(diagnostic: string): string {
  if (diagnostic.includes('FLOW')) return '🌊';
  if (diagnostic.includes('COHERENCE')) return '🔗';
  if (diagnostic.includes('COMPLEXITY')) return '🔀';
  if (diagnostic.includes('REFLECTION')) return '🪞';
  if (diagnostic.includes('ATTUNEMENT')) return '🤝';
  if (diagnostic.includes('OVERALL')) return '📊';
  return '•';
}

function getDiagnosticIcon(diagnostic: string): string {
  if (diagnostic.includes('LOW') || diagnostic.includes('POOR')) return 'text-amber-400';
  if (diagnostic.includes('OPTIMAL') || diagnostic.includes('FLOW_READY') || diagnostic.includes('WELL')) return 'text-green-400';
  return 'text-slate-400';
}

export default CognitivePanel;
