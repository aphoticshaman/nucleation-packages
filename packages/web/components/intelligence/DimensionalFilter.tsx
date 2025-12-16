'use client';

import { useState, useMemo, useCallback, ReactNode } from 'react';
import type {
  AnalysisFilters,
  TemporalState,
  EntityType,
  RelationType,
  ResourceType,
  ConflictPhase,
  IdeologicalAxis,
  MilitaryPosture,
} from '@/lib/analysis/dimensions';

// ============================================
// Recursive Filter Node Component
// ============================================

interface FilterNode {
  id: string;
  label: string;
  icon?: string;
  type: 'toggle' | 'range' | 'select' | 'multi' | 'group';
  value?: unknown;
  options?: Array<{ value: string; label: string; icon?: string }>;
  range?: { min: number; max: number; step?: number };
  children?: FilterNode[];
  autoCompute?: () => ReactNode; // Auto-computed display
  description?: string;
  defaultExpanded?: boolean;
}

interface FilterNodeProps {
  node: FilterNode;
  depth: number;
  onChange: (id: string, value: unknown) => void;
  values: Record<string, unknown>;
}

function RecursiveFilterNode({ node, depth, onChange, values }: FilterNodeProps) {
  const [expanded, setExpanded] = useState(node.defaultExpanded ?? depth < 2);
  const hasChildren = node.children && node.children.length > 0;

  // Depth-based styling
  const depthStyles = useMemo(() => {
    const colors = [
      'border-blue-500/30 bg-blue-950/20',
      'border-purple-500/30 bg-purple-950/20',
      'border-amber-500/30 bg-amber-950/20',
      'border-emerald-500/30 bg-emerald-950/20',
      'border-rose-500/30 bg-rose-950/20',
    ];
    return colors[depth % colors.length];
  }, [depth]);

  const currentValue = values[node.id];

  // Auto-computed result display
  const autoResult = useMemo(() => {
    if (node.autoCompute) {
      return node.autoCompute();
    }
    return null;
  }, [node]);

  return (
    <div className={`border-l-2 ${depthStyles} rounded-r-lg`}>
      {/* Node header */}
      <div
        className={`flex items-center gap-2 p-2 ${hasChildren ? 'cursor-pointer hover:bg-slate-800/50' : ''}`}
        onClick={() => hasChildren && setExpanded(!expanded)}
      >
        {hasChildren && (
          <span className="text-slate-500 text-xs w-4">
            {expanded ? '▼' : '▶'}
          </span>
        )}
        {!hasChildren && <span className="w-4" />}

        {node.icon && <span className="text-sm">{node.icon}</span>}

        <span className="text-sm text-slate-300 flex-1">{node.label}</span>

        {/* Inline control for leaf nodes */}
        {!hasChildren && (
          <FilterControl
            node={node}
            value={currentValue}
            onChange={(v) => onChange(node.id, v)}
          />
        )}
      </div>

      {/* Description */}
      {node.description && expanded && (
        <p className="text-xs text-slate-500 px-8 pb-2">{node.description}</p>
      )}

      {/* Auto-computed result */}
      {autoResult && expanded && (
        <div className="px-8 pb-2">{autoResult}</div>
      )}

      {/* Recursive children */}
      {hasChildren && expanded && (
        <div className="ml-4 space-y-1 pb-2">
          {node.children!.map((child) => (
            <RecursiveFilterNode
              key={child.id}
              node={child}
              depth={depth + 1}
              onChange={onChange}
              values={values}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================
// Individual Filter Controls
// ============================================

function FilterControl({
  node,
  value,
  onChange,
}: {
  node: FilterNode;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  switch (node.type) {
    case 'toggle':
      return (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onChange(!value);
          }}
          className={`w-8 h-4 rounded-full transition-colors ${
            value ? 'bg-blue-500' : 'bg-slate-700'
          }`}
        >
          <span
            className={`block w-3 h-3 bg-white rounded-full transition-transform ${
              value ? 'translate-x-4' : 'translate-x-0.5'
            }`}
          />
        </button>
      );

    case 'range':
      return (
        <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
          <input
            type="range"
            min={node.range?.min ?? 0}
            max={node.range?.max ?? 100}
            step={node.range?.step ?? 1}
            value={(value as number) ?? node.range?.min ?? 0}
            onChange={(e) => onChange(parseFloat(e.target.value))}
            className="w-20 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
          <span className="text-xs text-slate-400 w-8 text-right">
            {value as number ?? node.range?.min ?? 0}
          </span>
        </div>
      );

    case 'select':
      return (
        <select
          value={(value as string) ?? ''}
          onChange={(e) => {
            e.stopPropagation();
            onChange(e.target.value);
          }}
          onClick={(e) => e.stopPropagation()}
          className="bg-slate-800 text-slate-300 text-xs rounded px-2 py-1 border border-slate-700"
        >
          <option value="">All</option>
          {node.options?.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.icon} {opt.label}
            </option>
          ))}
        </select>
      );

    case 'multi':
      const selected = (value as string[]) ?? [];
      return (
        <div className="flex flex-wrap gap-1" onClick={(e) => e.stopPropagation()}>
          {node.options?.slice(0, 4).map((opt) => (
            <button
              key={opt.value}
              onClick={() => {
                const newVal = selected.includes(opt.value)
                  ? selected.filter((v) => v !== opt.value)
                  : [...selected, opt.value];
                onChange(newVal);
              }}
              className={`text-xs px-1.5 py-0.5 rounded ${
                selected.includes(opt.value)
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-800 text-slate-400'
              }`}
              title={opt.label}
            >
              {opt.icon || opt.label.slice(0, 3)}
            </button>
          ))}
          {(node.options?.length ?? 0) > 4 && (
            <span className="text-xs text-slate-500">+{(node.options?.length ?? 0) - 4}</span>
          )}
        </div>
      );

    default:
      return null;
  }
}

// ============================================
// Auto-Computed Indicators
// ============================================

function AutoIndicator({
  label,
  value,
  max = 100,
  color = 'blue',
  format = 'number',
}: {
  label: string;
  value: number;
  max?: number;
  color?: string;
  format?: 'number' | 'percent' | 'compact';
}) {
  const percent = (value / max) * 100;
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    amber: 'bg-amber-500',
    red: 'bg-red-500',
    purple: 'bg-purple-500',
  };

  const formatValue = () => {
    switch (format) {
      case 'percent':
        return `${value.toFixed(1)}%`;
      case 'compact':
        return value >= 1e9
          ? `${(value / 1e9).toFixed(1)}B`
          : value >= 1e6
            ? `${(value / 1e6).toFixed(1)}M`
            : value >= 1e3
              ? `${(value / 1e3).toFixed(1)}K`
              : value.toFixed(0);
      default:
        return value.toFixed(0);
    }
  };

  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-slate-500 w-20 truncate">{label}</span>
      <div className="flex-1 h-1.5 bg-slate-800 rounded overflow-hidden">
        <div
          className={`h-full ${colorClasses[color as keyof typeof colorClasses] || 'bg-blue-500'} transition-all`}
          style={{ width: `${Math.min(100, percent)}%` }}
        />
      </div>
      <span className="text-slate-400 w-12 text-right">{formatValue()}</span>
    </div>
  );
}

// ============================================
// Pre-Built Filter Trees
// ============================================

function buildTemporalFilters(): FilterNode {
  return {
    id: 'temporal',
    label: 'Temporal Dimension',
    icon: '',
    type: 'group',
    defaultExpanded: true,
    children: [
      {
        id: 'temporal_state',
        label: 'Time Focus',
        icon: '',
        type: 'multi',
        options: [
          { value: 'historical', label: 'Historical', icon: '' },
          { value: 'current', label: 'Current', icon: '' },
          { value: 'projected', label: 'Projected', icon: '' },
        ],
      },
      {
        id: 'temporal_range',
        label: 'Projection Horizon',
        type: 'group',
        children: [
          {
            id: 'temporal_range_days',
            label: 'Days ahead',
            type: 'range',
            range: { min: 7, max: 365, step: 7 },
          },
          {
            id: 'temporal_confidence',
            label: 'Min confidence',
            type: 'range',
            range: { min: 0, max: 100, step: 5 },
            autoCompute: () => (
              <AutoIndicator label="Forecast quality" value={72} color="blue" format="percent" />
            ),
          },
        ],
      },
    ],
  };
}

function buildEntityFilters(): FilterNode {
  return {
    id: 'entities',
    label: 'Entity Filters',
    icon: '',
    type: 'group',
    children: [
      {
        id: 'entity_types',
        label: 'Entity Types',
        type: 'multi',
        options: [
          { value: 'nation', label: 'Nations', icon: '' },
          { value: 'organization', label: 'Organizations', icon: '' },
          { value: 'alliance', label: 'Alliances', icon: '' },
          { value: 'individual', label: 'Individuals', icon: '' },
          { value: 'movement', label: 'Movements', icon: '' },
        ],
      },
      {
        id: 'entity_groups',
        label: 'Regional Groups',
        type: 'group',
        children: [
          {
            id: 'group_nato',
            label: 'NATO',
            icon: '',
            type: 'toggle',
            autoCompute: () => (
              <div className="space-y-1">
                <AutoIndicator label="Cohesion" value={78} color="blue" format="percent" />
                <AutoIndicator label="Readiness" value={65} color="green" format="percent" />
              </div>
            ),
          },
          {
            id: 'group_brics',
            label: 'BRICS+',
            icon: '',
            type: 'toggle',
            autoCompute: () => (
              <div className="space-y-1">
                <AutoIndicator label="Economic power" value={42} color="amber" format="percent" />
                <AutoIndicator label="Growth rate" value={4.2} max={10} color="green" />
              </div>
            ),
          },
          {
            id: 'group_eu',
            label: 'European Union',
            icon: '',
            type: 'toggle',
          },
          {
            id: 'group_asean',
            label: 'ASEAN',
            icon: '',
            type: 'toggle',
          },
          {
            id: 'group_gcc',
            label: 'Gulf States',
            icon: '',
            type: 'toggle',
          },
        ],
      },
    ],
  };
}

function buildRelationFilters(): FilterNode {
  return {
    id: 'relations',
    label: 'Alliance & Hostility',
    icon: '',
    type: 'group',
    children: [
      {
        id: 'relation_type',
        label: 'Relation Type',
        type: 'multi',
        options: [
          { value: 'alliance', label: 'Alliance', icon: '' },
          { value: 'partnership', label: 'Partnership', icon: '' },
          { value: 'friendly', label: 'Friendly', icon: '' },
          { value: 'neutral', label: 'Neutral', icon: '' },
          { value: 'tension', label: 'Tension', icon: '' },
          { value: 'rivalry', label: 'Rivalry', icon: '' },
          { value: 'hostile', label: 'Hostile', icon: '' },
          { value: 'conflict', label: 'Conflict', icon: '' },
        ],
      },
      {
        id: 'relation_basis',
        label: 'Relation Basis',
        type: 'group',
        children: [
          { id: 'basis_military', label: 'Military', icon: '', type: 'toggle' },
          { id: 'basis_economic', label: 'Economic', icon: '', type: 'toggle' },
          { id: 'basis_ideological', label: 'Ideological', icon: '', type: 'toggle' },
          { id: 'basis_religious', label: 'Religious', icon: '', type: 'toggle' },
          { id: 'basis_historical', label: 'Historical', icon: '', type: 'toggle' },
          { id: 'basis_territorial', label: 'Territorial', icon: '', type: 'toggle' },
          { id: 'basis_resource', label: 'Resource', icon: '', type: 'toggle' },
        ],
      },
      {
        id: 'relation_strength',
        label: 'Min Strength',
        type: 'range',
        range: { min: 0, max: 100, step: 10 },
      },
    ],
  };
}

function buildIdeologicalFilters(): FilterNode {
  return {
    id: 'ideological',
    label: 'Ideological Spectrum',
    icon: '',
    type: 'group',
    children: [
      {
        id: 'axis_economic',
        label: 'Economic (Left ↔ Right)',
        type: 'range',
        range: { min: -100, max: 100, step: 10 },
        description: 'State control vs free market',
      },
      {
        id: 'axis_social',
        label: 'Social (Lib ↔ Auth)',
        type: 'range',
        range: { min: -100, max: 100, step: 10 },
        description: 'Individual freedom vs collective order',
      },
      {
        id: 'axis_foreign',
        label: 'Foreign Policy',
        type: 'range',
        range: { min: -100, max: 100, step: 10 },
        description: 'Isolationist vs interventionist',
      },
      {
        id: 'axis_governance',
        label: 'Governance',
        type: 'range',
        range: { min: -100, max: 100, step: 10 },
        description: 'Democratic vs autocratic',
      },
    ],
  };
}

function buildMilitaryFilters(): FilterNode {
  return {
    id: 'military',
    label: 'Military Dimension',
    icon: '',
    type: 'group',
    children: [
      {
        id: 'military_posture',
        label: 'Posture',
        type: 'multi',
        options: [
          { value: 'defensive', label: 'Defensive', icon: '' },
          { value: 'neutral', label: 'Neutral', icon: '' },
          { value: 'forward', label: 'Forward', icon: '' },
          { value: 'aggressive', label: 'Aggressive', icon: '' },
        ],
      },
      {
        id: 'military_capabilities',
        label: 'Capabilities',
        type: 'group',
        children: [
          {
            id: 'nuclear',
            label: 'Nuclear capable',
            icon: '',
            type: 'toggle',
            autoCompute: () => (
              <span className="text-xs text-amber-400">9 nations with nuclear weapons</span>
            ),
          },
          {
            id: 'blue_water_navy',
            label: 'Blue water navy',
            icon: '',
            type: 'toggle',
          },
          {
            id: 'space_capable',
            label: 'Space capable',
            icon: '',
            type: 'toggle',
          },
          {
            id: 'cyber_tier1',
            label: 'Tier 1 cyber',
            icon: '',
            type: 'toggle',
          },
        ],
      },
      {
        id: 'military_power_min',
        label: 'Min Power Index',
        type: 'range',
        range: { min: 0, max: 100, step: 5 },
      },
    ],
  };
}

function buildResourceFilters(): FilterNode {
  return {
    id: 'resources',
    label: 'Resource Dependencies',
    icon: '',
    type: 'group',
    children: [
      {
        id: 'resource_types',
        label: 'Resource Types',
        type: 'group',
        children: [
          {
            id: 'res_energy',
            label: 'Energy',
            icon: '',
            type: 'group',
            children: [
              { id: 'res_oil', label: 'Oil', type: 'toggle' },
              { id: 'res_gas', label: 'Natural Gas', type: 'toggle' },
              { id: 'res_coal', label: 'Coal', type: 'toggle' },
              { id: 'res_uranium', label: 'Uranium', type: 'toggle' },
            ],
          },
          {
            id: 'res_minerals',
            label: 'Critical Minerals',
            icon: '',
            type: 'group',
            children: [
              { id: 'res_rare_earths', label: 'Rare Earths', type: 'toggle' },
              { id: 'res_lithium', label: 'Lithium', type: 'toggle' },
              { id: 'res_cobalt', label: 'Cobalt', type: 'toggle' },
              { id: 'res_copper', label: 'Copper', type: 'toggle' },
            ],
          },
          {
            id: 'res_strategic',
            label: 'Strategic',
            icon: '',
            type: 'group',
            children: [
              { id: 'res_semiconductors', label: 'Semiconductors', type: 'toggle' },
              { id: 'res_pharma', label: 'Pharmaceuticals', type: 'toggle' },
              { id: 'res_food', label: 'Food/Grains', type: 'toggle' },
            ],
          },
        ],
      },
      {
        id: 'dependency_level',
        label: 'Dependency Criticality',
        type: 'select',
        options: [
          { value: 'essential', label: 'Essential' },
          { value: 'important', label: 'Important' },
          { value: 'moderate', label: 'Moderate' },
          { value: 'low', label: 'Low' },
        ],
      },
      {
        id: 'overlap_analysis',
        label: 'Overlap Analysis',
        type: 'group',
        autoCompute: () => (
          <div className="space-y-2 text-xs">
            <div className="text-slate-400">Auto-computed resource overlaps:</div>
            <div className="flex gap-2">
              <span className="bg-red-500/20 text-red-400 px-2 py-1 rounded">
                US-China: 78% rare earth dependency
              </span>
            </div>
            <div className="flex gap-2">
              <span className="bg-amber-500/20 text-amber-400 px-2 py-1 rounded">
                EU-Russia: 42% gas dependency
              </span>
            </div>
          </div>
        ),
        children: [],
      },
    ],
  };
}

function buildConflictFilters(): FilterNode {
  return {
    id: 'conflicts',
    label: 'Conflict Trajectories',
    icon: '',
    type: 'group',
    children: [
      {
        id: 'conflict_phase',
        label: 'Phase',
        type: 'multi',
        options: [
          { value: 'latent', label: 'Latent', icon: '' },
          { value: 'emerging', label: 'Emerging', icon: '' },
          { value: 'escalating', label: 'Escalating', icon: '' },
          { value: 'crisis', label: 'Crisis', icon: '' },
          { value: 'active_conflict', label: 'Active', icon: '' },
          { value: 'de_escalating', label: 'De-escalating', icon: '' },
          { value: 'post_conflict', label: 'Post-conflict', icon: '' },
        ],
      },
      {
        id: 'conflict_drivers',
        label: 'Conflict Drivers',
        type: 'group',
        children: [
          { id: 'driver_territorial', label: 'Territorial', icon: '', type: 'toggle' },
          { id: 'driver_resource', label: 'Resource', icon: '', type: 'toggle' },
          { id: 'driver_ideological', label: 'Ideological', icon: '', type: 'toggle' },
          { id: 'driver_ethnic', label: 'Ethnic', icon: '', type: 'toggle' },
          { id: 'driver_religious', label: 'Religious', icon: '', type: 'toggle' },
          { id: 'driver_proxy', label: 'Proxy', icon: '', type: 'toggle' },
        ],
      },
      {
        id: 'escalation_risk',
        label: 'Min Escalation Risk',
        type: 'range',
        range: { min: 0, max: 100, step: 5 },
        autoCompute: () => (
          <div className="space-y-1">
            <AutoIndicator label="Global tension" value={62} color="amber" format="percent" />
            <AutoIndicator label="Active hotspots" value={14} max={20} color="red" />
          </div>
        ),
      },
    ],
  };
}

function buildDemographicFilters(): FilterNode {
  return {
    id: 'demographics',
    label: 'Population & Trends',
    icon: '',
    type: 'group',
    children: [
      {
        id: 'population_range',
        label: 'Population Range',
        type: 'group',
        children: [
          {
            id: 'pop_min',
            label: 'Minimum (millions)',
            type: 'range',
            range: { min: 0, max: 1000, step: 10 },
          },
          {
            id: 'pop_max',
            label: 'Maximum (millions)',
            type: 'range',
            range: { min: 0, max: 1500, step: 10 },
          },
        ],
      },
      {
        id: 'growth_factors',
        label: 'Growth Factors',
        type: 'group',
        children: [
          {
            id: 'youth_bulge',
            label: 'Youth bulge risk',
            icon: '',
            type: 'toggle',
            description: 'Large young population = instability risk',
          },
          {
            id: 'aging_crisis',
            label: 'Aging crisis',
            icon: '',
            type: 'toggle',
          },
          {
            id: 'migration_pressure',
            label: 'Migration pressure',
            icon: '',
            type: 'toggle',
          },
        ],
      },
      {
        id: 'demographic_summary',
        label: 'Summary',
        type: 'group',
        autoCompute: () => (
          <div className="space-y-1">
            <AutoIndicator label="World pop" value={8.1} max={10} color="blue" />
            <AutoIndicator label="Urban" value={56} color="purple" format="percent" />
            <AutoIndicator label="Med age" value={30.5} max={50} color="green" />
          </div>
        ),
        children: [],
      },
    ],
  };
}

function buildGovernanceFilters(): FilterNode {
  return {
    id: 'governance',
    label: 'Governance & Stability',
    icon: '',
    type: 'group',
    children: [
      {
        id: 'regime_type',
        label: 'Regime Type',
        type: 'multi',
        options: [
          { value: 'democracy', label: 'Democracy', icon: '' },
          { value: 'hybrid', label: 'Hybrid', icon: '' },
          { value: 'authoritarian', label: 'Authoritarian', icon: '' },
          { value: 'failed_state', label: 'Failed State', icon: '' },
        ],
      },
      {
        id: 'gov_metrics',
        label: 'Governance Metrics',
        type: 'group',
        children: [
          {
            id: 'democracy_index_min',
            label: 'Min Democracy Index',
            type: 'range',
            range: { min: 0, max: 10, step: 0.5 },
          },
          {
            id: 'corruption_max',
            label: 'Max Corruption',
            type: 'range',
            range: { min: 0, max: 100, step: 5 },
          },
          {
            id: 'stability_min',
            label: 'Min Stability',
            type: 'range',
            range: { min: 0, max: 100, step: 5 },
          },
        ],
      },
      {
        id: 'succession_risk',
        label: 'Succession Risk',
        type: 'select',
        options: [
          { value: 'imminent', label: 'Imminent' },
          { value: 'high', label: 'High' },
          { value: 'moderate', label: 'Moderate' },
          { value: 'low', label: 'Low' },
        ],
        autoCompute: () => (
          <span className="text-xs text-amber-400">3 nations with imminent succession risk</span>
        ),
      },
    ],
  };
}

// ============================================
// Main Dimensional Filter Component
// ============================================

interface DimensionalFilterProps {
  onChange?: (filters: Record<string, unknown>) => void;
  defaultExpanded?: boolean;
  compact?: boolean;
}

export function DimensionalFilter({
  onChange,
  defaultExpanded = false,
  compact = false,
}: DimensionalFilterProps) {
  const [filterValues, setFilterValues] = useState<Record<string, unknown>>({});
  const [expandedRoot, setExpandedRoot] = useState(defaultExpanded);

  // Build the complete filter tree
  const filterTree = useMemo<FilterNode[]>(() => [
    buildTemporalFilters(),
    buildEntityFilters(),
    buildRelationFilters(),
    buildIdeologicalFilters(),
    buildMilitaryFilters(),
    buildResourceFilters(),
    buildConflictFilters(),
    buildDemographicFilters(),
    buildGovernanceFilters(),
  ], []);

  const handleFilterChange = useCallback((id: string, value: unknown) => {
    setFilterValues((prev) => {
      const next = { ...prev, [id]: value };
      onChange?.(next);
      return next;
    });
  }, [onChange]);

  // Count active filters
  const activeFilterCount = useMemo(() => {
    return Object.values(filterValues).filter((v) => {
      if (Array.isArray(v)) return v.length > 0;
      if (typeof v === 'boolean') return v;
      if (typeof v === 'number') return true;
      if (typeof v === 'string') return v !== '';
      return false;
    }).length;
  }, [filterValues]);

  if (compact) {
    return (
      <div className="bg-slate-900 rounded-lg border border-slate-800 p-2">
        <button
          onClick={() => setExpandedRoot(!expandedRoot)}
          className="w-full flex items-center justify-between text-sm"
        >
          <span className="text-slate-300 flex items-center gap-2">
            <span>🔍</span>
            <span>Multi-Dimensional Filters</span>
            {activeFilterCount > 0 && (
              <span className="bg-blue-600 text-white text-xs px-1.5 py-0.5 rounded-full">
                {activeFilterCount}
              </span>
            )}
          </span>
          <span className="text-slate-500">{expandedRoot ? '▼' : '▶'}</span>
        </button>

        {expandedRoot && (
          <div className="mt-3 space-y-2 max-h-[60vh] overflow-y-auto">
            {filterTree.map((node) => (
              <RecursiveFilterNode
                key={node.id}
                node={node}
                depth={0}
                onChange={handleFilterChange}
                values={filterValues}
              />
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 space-y-3">
      <div className="flex items-center justify-between pb-2 border-b border-slate-800">
        <h3 className="text-sm font-medium text-white flex items-center gap-2">
          <span>🔍</span>
          <span>Multi-Dimensional Analysis</span>
        </h3>
        {activeFilterCount > 0 && (
          <button
            onClick={() => setFilterValues({})}
            className="text-xs text-slate-400 hover:text-white"
          >
            Clear ({activeFilterCount})
          </button>
        )}
      </div>

      <div className="space-y-2 max-h-[70vh] overflow-y-auto pr-2">
        {filterTree.map((node) => (
          <RecursiveFilterNode
            key={node.id}
            node={node}
            depth={0}
            onChange={handleFilterChange}
            values={filterValues}
          />
        ))}
      </div>
    </div>
  );
}

export default DimensionalFilter;
