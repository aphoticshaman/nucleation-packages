'use client';

import { useState, useMemo, useCallback } from 'react';
import { FuzzyNumber, defuzzify } from '@/lib/epistemic-engine';

type ScenarioType = 'baseline' | 'optimistic' | 'pessimistic' | 'black_swan' | 'custom';

interface ScenarioVariable {
  id: string;
  name: string;
  description?: string;
  category: 'political' | 'economic' | 'military' | 'social' | 'environmental';
  baseValue: number;
  modifiedValue: number;
  minValue: number;
  maxValue: number;
  unit?: string;
  sensitivity: 'low' | 'medium' | 'high' | 'critical';
  dependencies?: string[]; // IDs of variables this depends on
}

interface Scenario {
  id: string;
  name: string;
  type: ScenarioType;
  description?: string;
  variables: ScenarioVariable[];
  probability: FuzzyNumber;
  outcomes: ScenarioOutcome[];
  createdAt: string;
  createdBy: string;
  notes?: string;
}

interface ScenarioOutcome {
  id: string;
  description: string;
  impact: 'catastrophic' | 'severe' | 'moderate' | 'minor' | 'negligible';
  probability: number;
  timeframe: string;
  cascadeEffects?: string[];
}

interface ScenarioBuilderProps {
  scenarios: Scenario[];
  baselineVariables: ScenarioVariable[];
  onScenarioCreate?: (scenario: Omit<Scenario, 'id' | 'createdAt'>) => void;
  onScenarioUpdate?: (id: string, changes: Partial<Scenario>) => void;
  onScenarioDelete?: (id: string) => void;
  onRunSimulation?: (scenarioId: string) => void;
  currentUser?: { id: string; name: string };
}

// Component 49: What-If Scenario Builder
export function ScenarioBuilder({
  scenarios,
  baselineVariables,
  onScenarioCreate,
  onScenarioUpdate,
  onScenarioDelete,
  onRunSimulation,
  currentUser = { id: 'user-1', name: 'Analyst' },
}: ScenarioBuilderProps) {
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  const [compareScenarios, setCompareScenarios] = useState<string[]>([]);

  const typeConfig: Record<ScenarioType, { label: string; icon: string; color: string }> = {
    baseline: { label: 'Baseline', icon: '📊', color: 'slate' },
    optimistic: { label: 'Optimistic', icon: '☀', color: 'green' },
    pessimistic: { label: 'Pessimistic', icon: '🌧', color: 'amber' },
    black_swan: { label: 'Black Swan', icon: '🦢', color: 'red' },
    custom: { label: 'Custom', icon: '🔧', color: 'cyan' },
  };

  const categoryConfig: Record<ScenarioVariable['category'], { icon: string; color: string }> = {
    political: { icon: '🏛', color: 'purple' },
    economic: { icon: '💰', color: 'green' },
    military: { icon: '⚔', color: 'red' },
    social: { icon: '👥', color: 'cyan' },
    environmental: { icon: '🌍', color: 'amber' },
  };

  const activeScenario = scenarios.find(s => s.id === selectedScenario);

  // Calculate scenario divergence from baseline
  const calculateDivergence = useCallback((scenario: Scenario) => {
    let totalDivergence = 0;
    let count = 0;

    for (const variable of scenario.variables) {
      const baseVar = baselineVariables.find(v => v.id === variable.id);
      if (baseVar) {
        const range = variable.maxValue - variable.minValue;
        const divergence = Math.abs(variable.modifiedValue - baseVar.baseValue) / range;
        totalDivergence += divergence;
        count++;
      }
    }

    return count > 0 ? totalDivergence / count : 0;
  }, [baselineVariables]);

  return (
    <div className="bg-slate-900/50 rounded-lg border border-slate-700 h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-slate-700">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-medium text-slate-200">Scenario Builder</h2>
            <p className="text-xs text-slate-500">Create and compare what-if scenarios</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setCompareMode(!compareMode)}
              className={`px-3 py-1.5 rounded text-sm transition-colors ${
                compareMode
                  ? 'bg-purple-500/20 text-purple-400'
                  : 'bg-slate-700 text-slate-400 hover:text-slate-200'
              }`}
            >
              ⚖ Compare
            </button>
            <button
              onClick={() => setIsCreating(true)}
              className="px-3 py-1.5 bg-cyan-500/20 text-cyan-400 rounded text-sm font-medium hover:bg-cyan-500/30"
            >
              + New Scenario
            </button>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Scenario list sidebar */}
        <div className="w-72 border-r border-slate-700 overflow-y-auto">
          {scenarios.map(scenario => {
            const config = typeConfig[scenario.type];
            const divergence = calculateDivergence(scenario);
            const prob = defuzzify(scenario.probability);
            const isSelected = selectedScenario === scenario.id;
            const isComparing = compareScenarios.includes(scenario.id);

            return (
              <div
                key={scenario.id}
                onClick={() => {
                  if (compareMode) {
                    setCompareScenarios(prev =>
                      prev.includes(scenario.id)
                        ? prev.filter(id => id !== scenario.id)
                        : [...prev.slice(-1), scenario.id]
                    );
                  } else {
                    setSelectedScenario(scenario.id);
                  }
                }}
                className={`p-3 border-b border-slate-800 cursor-pointer transition-colors ${
                  isSelected ? 'bg-cyan-500/10 border-l-2 border-l-cyan-500' :
                  isComparing ? 'bg-purple-500/10 border-l-2 border-l-purple-500' :
                  'hover:bg-slate-800/50'
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className={`px-1.5 py-0.5 rounded text-xs bg-${config.color}-500/20 text-${config.color}-400`}>
                    {config.icon} {config.label}
                  </span>
                  <span className="text-sm font-medium text-slate-200 truncate">
                    {scenario.name}
                  </span>
                </div>

                <div className="flex items-center justify-between mt-2">
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <span>P: {(prob * 100).toFixed(0)}%</span>
                    <span>Δ: {(divergence * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex items-center gap-1 text-xs text-slate-500">
                    <span>{scenario.outcomes.length} outcomes</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Scenario detail / comparison view */}
        <div className="flex-1 overflow-y-auto">
          {compareMode && compareScenarios.length === 2 ? (
            <ScenarioComparison
              scenarios={scenarios.filter(s => compareScenarios.includes(s.id))}
              baselineVariables={baselineVariables}
              categoryConfig={categoryConfig}
            />
          ) : activeScenario ? (
            <ScenarioDetail
              scenario={activeScenario}
              baselineVariables={baselineVariables}
              categoryConfig={categoryConfig}
              typeConfig={typeConfig}
              onUpdate={(changes) => onScenarioUpdate?.(activeScenario.id, changes)}
              onDelete={() => {
                onScenarioDelete?.(activeScenario.id);
                setSelectedScenario(null);
              }}
              onRunSimulation={() => onRunSimulation?.(activeScenario.id)}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-slate-500">
              {compareMode
                ? 'Select two scenarios to compare'
                : 'Select a scenario to view details'}
            </div>
          )}
        </div>
      </div>

      {/* Create scenario modal */}
      {isCreating && (
        <CreateScenarioModal
          baselineVariables={baselineVariables}
          typeConfig={typeConfig}
          categoryConfig={categoryConfig}
          currentUser={currentUser}
          onClose={() => setIsCreating(false)}
          onCreate={(data) => {
            onScenarioCreate?.(data);
            setIsCreating(false);
          }}
        />
      )}
    </div>
  );
}

// Scenario detail view
function ScenarioDetail({
  scenario,
  baselineVariables,
  categoryConfig,
  typeConfig,
  onUpdate,
  onDelete,
  onRunSimulation,
}: {
  scenario: Scenario;
  baselineVariables: ScenarioVariable[];
  categoryConfig: Record<ScenarioVariable['category'], { icon: string; color: string }>;
  typeConfig: Record<ScenarioType, { label: string; icon: string; color: string }>;
  onUpdate: (changes: Partial<Scenario>) => void;
  onDelete: () => void;
  onRunSimulation: () => void;
}) {
  const config = typeConfig[scenario.type];
  const prob = defuzzify(scenario.probability);

  // Group variables by category
  const groupedVariables = useMemo(() => {
    const groups: Record<string, ScenarioVariable[]> = {};
    for (const variable of scenario.variables) {
      if (!groups[variable.category]) {
        groups[variable.category] = [];
      }
      groups[variable.category].push(variable);
    }
    return groups;
  }, [scenario.variables]);

  const impactColors = {
    catastrophic: 'text-red-400 bg-red-500/10',
    severe: 'text-orange-400 bg-orange-500/10',
    moderate: 'text-amber-400 bg-amber-500/10',
    minor: 'text-yellow-400 bg-yellow-500/10',
    negligible: 'text-slate-400 bg-slate-500/10',
  };

  return (
    <div className="p-4 space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className={`px-2 py-1 rounded bg-${config.color}-500/20 text-${config.color}-400`}>
              {config.icon} {config.label}
            </span>
            <h3 className="text-xl font-medium text-slate-200">{scenario.name}</h3>
          </div>
          {scenario.description && (
            <p className="text-sm text-slate-400">{scenario.description}</p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={onRunSimulation}
            className="px-4 py-2 bg-cyan-500 text-slate-900 rounded font-medium hover:bg-cyan-400 transition-colors"
          >
            ▶ Run Simulation
          </button>
          <button
            onClick={onDelete}
            className="p-2 text-slate-400 hover:text-red-400 transition-colors"
          >
            🗑
          </button>
        </div>
      </div>

      {/* Probability gauge */}
      <div className="bg-slate-800/50 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-slate-400">Scenario Probability</span>
          <span className="text-lg font-bold text-cyan-400">{(prob * 100).toFixed(0)}%</span>
        </div>
        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-cyan-500 to-cyan-400 transition-all"
            style={{ width: `${prob * 100}%` }}
          />
        </div>
        <div className="flex justify-between mt-1 text-xs text-slate-500">
          <span>Low: {(scenario.probability.low * 100).toFixed(0)}%</span>
          <span>Peak: {(scenario.probability.peak * 100).toFixed(0)}%</span>
          <span>High: {(scenario.probability.high * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Variables by category */}
      <div className="space-y-4">
        <h4 className="text-sm font-medium text-slate-300">Modified Variables</h4>
        {Object.entries(groupedVariables).map(([category, variables]) => {
          const catConfig = categoryConfig[category as ScenarioVariable['category']];
          return (
            <div key={category} className="bg-slate-800/30 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-3">
                <span>{catConfig.icon}</span>
                <span className="text-sm font-medium text-slate-300 capitalize">{category}</span>
              </div>
              <div className="space-y-2">
                {variables.map(variable => {
                  const baseline = baselineVariables.find(v => v.id === variable.id);
                  const delta = baseline
                    ? ((variable.modifiedValue - baseline.baseValue) / baseline.baseValue) * 100
                    : 0;

                  return (
                    <div key={variable.id} className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="text-sm text-slate-200">{variable.name}</div>
                        <div className="flex items-center gap-2 mt-1">
                          <div className="flex-1 h-1.5 bg-slate-700 rounded-full">
                            <div
                              className="h-full bg-cyan-500 rounded-full"
                              style={{
                                width: `${((variable.modifiedValue - variable.minValue) / (variable.maxValue - variable.minValue)) * 100}%`,
                              }}
                            />
                          </div>
                        </div>
                      </div>
                      <div className="text-right ml-4">
                        <div className="text-sm font-mono text-slate-200">
                          {variable.modifiedValue.toFixed(1)}{variable.unit || ''}
                        </div>
                        {delta !== 0 && (
                          <div className={`text-xs ${delta > 0 ? 'text-red-400' : 'text-green-400'}`}>
                            {delta > 0 ? '+' : ''}{delta.toFixed(1)}%
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      {/* Outcomes */}
      <div className="space-y-3">
        <h4 className="text-sm font-medium text-slate-300">Projected Outcomes</h4>
        {scenario.outcomes.map(outcome => (
          <div key={outcome.id} className="bg-slate-800/30 rounded-lg p-3">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className={`px-1.5 py-0.5 rounded text-xs ${impactColors[outcome.impact]}`}>
                    {outcome.impact.toUpperCase()}
                  </span>
                  <span className="text-xs text-slate-500">{outcome.timeframe}</span>
                </div>
                <p className="text-sm text-slate-200 mt-1">{outcome.description}</p>
                {outcome.cascadeEffects && outcome.cascadeEffects.length > 0 && (
                  <div className="mt-2 text-xs text-slate-500">
                    Cascade: {outcome.cascadeEffects.join(' → ')}
                  </div>
                )}
              </div>
              <div className="text-right ml-4">
                <div className="text-lg font-bold text-slate-200">
                  {(outcome.probability * 100).toFixed(0)}%
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Notes */}
      {scenario.notes && (
        <div className="bg-slate-800/30 rounded-lg p-3">
          <h4 className="text-sm font-medium text-slate-300 mb-2">Analyst Notes</h4>
          <p className="text-sm text-slate-400">{scenario.notes}</p>
        </div>
      )}
    </div>
  );
}

// Scenario comparison view
function ScenarioComparison({
  scenarios,
  baselineVariables,
  categoryConfig,
}: {
  scenarios: Scenario[];
  baselineVariables: ScenarioVariable[];
  categoryConfig: Record<ScenarioVariable['category'], { icon: string; color: string }>;
}) {
  if (scenarios.length !== 2) return null;

  const [s1, s2] = scenarios;

  return (
    <div className="p-4">
      <h3 className="text-lg font-medium text-slate-200 mb-4">Scenario Comparison</h3>

      {/* Header row */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="text-sm text-slate-400">Variable</div>
        <div className="text-center text-sm font-medium text-cyan-400">{s1.name}</div>
        <div className="text-center text-sm font-medium text-purple-400">{s2.name}</div>
      </div>

      {/* Probability comparison */}
      <div className="grid grid-cols-3 gap-4 p-3 bg-slate-800/30 rounded-lg mb-4">
        <div className="text-sm text-slate-300">Probability</div>
        <div className="text-center text-lg font-bold text-cyan-400">
          {(defuzzify(s1.probability) * 100).toFixed(0)}%
        </div>
        <div className="text-center text-lg font-bold text-purple-400">
          {(defuzzify(s2.probability) * 100).toFixed(0)}%
        </div>
      </div>

      {/* Variable comparison */}
      {baselineVariables.map(baseVar => {
        const v1 = s1.variables.find(v => v.id === baseVar.id);
        const v2 = s2.variables.find(v => v.id === baseVar.id);

        const val1 = v1?.modifiedValue ?? baseVar.baseValue;
        const val2 = v2?.modifiedValue ?? baseVar.baseValue;
        const diff = val2 - val1;

        return (
          <div key={baseVar.id} className="grid grid-cols-3 gap-4 p-2 border-b border-slate-800">
            <div className="text-sm text-slate-300">{baseVar.name}</div>
            <div className="text-center text-sm font-mono text-slate-200">
              {val1.toFixed(1)}{baseVar.unit || ''}
            </div>
            <div className="text-center">
              <span className="text-sm font-mono text-slate-200">
                {val2.toFixed(1)}{baseVar.unit || ''}
              </span>
              {diff !== 0 && (
                <span className={`ml-2 text-xs ${diff > 0 ? 'text-red-400' : 'text-green-400'}`}>
                  ({diff > 0 ? '+' : ''}{diff.toFixed(1)})
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Create scenario modal (simplified)
function CreateScenarioModal({
  baselineVariables,
  typeConfig,
  categoryConfig,
  currentUser,
  onClose,
  onCreate,
}: {
  baselineVariables: ScenarioVariable[];
  typeConfig: Record<ScenarioType, { label: string; icon: string; color: string }>;
  categoryConfig: Record<ScenarioVariable['category'], { icon: string; color: string }>;
  currentUser: { id: string; name: string };
  onClose: () => void;
  onCreate: (data: Omit<Scenario, 'id' | 'createdAt'>) => void;
}) {
  const [name, setName] = useState('');
  const [type, setType] = useState<ScenarioType>('custom');
  const [description, setDescription] = useState('');

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div className="bg-slate-800 rounded-lg p-6 w-[500px]" onClick={(e) => e.stopPropagation()}>
        <h3 className="text-lg font-medium text-slate-200 mb-4">Create Scenario</h3>

        <div className="space-y-4">
          <div>
            <label className="block text-xs text-slate-400 mb-1">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-slate-200"
            />
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-2">Type</label>
            <div className="grid grid-cols-5 gap-2">
              {(Object.entries(typeConfig) as [ScenarioType, typeof typeConfig[ScenarioType]][]).map(([t, config]) => (
                <button
                  key={t}
                  onClick={() => setType(t)}
                  className={`p-2 rounded text-center ${
                    type === t ? 'bg-cyan-500/20 text-cyan-400 ring-1 ring-cyan-500' : 'bg-slate-700 text-slate-400'
                  }`}
                >
                  <div className="text-lg">{config.icon}</div>
                  <div className="text-xs mt-1">{config.label}</div>
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-1">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-slate-200 resize-none"
              rows={2}
            />
          </div>
        </div>

        <div className="flex justify-end gap-2 mt-6">
          <button onClick={onClose} className="px-3 py-1.5 text-sm text-slate-400">
            Cancel
          </button>
          <button
            onClick={() => onCreate({
              name,
              type,
              description,
              variables: baselineVariables.map(v => ({ ...v, modifiedValue: v.baseValue })),
              probability: { low: 0.3, peak: 0.5, high: 0.7, confidence: 0.6 },
              outcomes: [],
              createdBy: currentUser.id,
            })}
            disabled={!name.trim()}
            className="px-4 py-1.5 bg-cyan-500 text-slate-900 rounded text-sm font-medium disabled:opacity-50"
          >
            Create
          </button>
        </div>
      </div>
    </div>
  );
}

// Mock data
export const mockScenarioVariables: ScenarioVariable[] = [
  { id: 'v1', name: 'Oil Price', category: 'economic', baseValue: 80, modifiedValue: 80, minValue: 40, maxValue: 150, unit: '$/bbl', sensitivity: 'high' },
  { id: 'v2', name: 'NATO Cohesion Index', category: 'political', baseValue: 0.75, modifiedValue: 0.75, minValue: 0, maxValue: 1, sensitivity: 'critical' },
  { id: 'v3', name: 'Russian Military Strength', category: 'military', baseValue: 0.65, modifiedValue: 0.65, minValue: 0, maxValue: 1, sensitivity: 'high' },
  { id: 'v4', name: 'European Energy Reserves', category: 'economic', baseValue: 0.6, modifiedValue: 0.6, minValue: 0, maxValue: 1, sensitivity: 'medium' },
  { id: 'v5', name: 'Public Support for Aid', category: 'social', baseValue: 0.55, modifiedValue: 0.55, minValue: 0, maxValue: 1, sensitivity: 'medium' },
];

export const mockScenarios: Scenario[] = [
  {
    id: '1',
    name: 'Prolonged Stalemate',
    type: 'baseline',
    description: 'Current trajectory continues with no major changes',
    variables: mockScenarioVariables,
    probability: { low: 0.4, peak: 0.55, high: 0.7, confidence: 0.7 },
    outcomes: [
      { id: 'o1', description: 'Frozen conflict by 2025', impact: 'moderate', probability: 0.6, timeframe: '12-18 months' },
    ],
    createdAt: '2024-01-01T00:00:00Z',
    createdBy: 'user-1',
  },
  {
    id: '2',
    name: 'NATO Escalation',
    type: 'pessimistic',
    description: 'Direct NATO involvement following Article 5 trigger',
    variables: mockScenarioVariables.map(v =>
      v.id === 'v2' ? { ...v, modifiedValue: 0.95 } :
      v.id === 'v3' ? { ...v, modifiedValue: 0.8 } : v
    ),
    probability: { low: 0.05, peak: 0.12, high: 0.2, confidence: 0.5 },
    outcomes: [
      { id: 'o2', description: 'Conventional conflict in Eastern Europe', impact: 'catastrophic', probability: 0.8, timeframe: '3-6 months', cascadeEffects: ['Global recession', 'Energy crisis', 'Refugee crisis'] },
    ],
    createdAt: '2024-01-05T00:00:00Z',
    createdBy: 'user-1',
  },
];
