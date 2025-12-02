'use client';

import { useState, useMemo } from 'react';

interface AlertThreshold {
  id: string;
  name: string;
  metric: string;
  operator: 'gt' | 'lt' | 'eq' | 'gte' | 'lte' | 'between';
  value: number;
  secondValue?: number; // for 'between'
  severity: 'info' | 'warning' | 'critical';
  enabled: boolean;
  domain?: string;
  notifyChannels: ('email' | 'slack' | 'sms' | 'webhook')[];
  cooldownMinutes: number;
  lastTriggered?: string;
}

interface AlertThresholdsProps {
  thresholds: AlertThreshold[];
  onThresholdChange?: (id: string, changes: Partial<AlertThreshold>) => void;
  onThresholdDelete?: (id: string) => void;
  onThresholdCreate?: (threshold: Omit<AlertThreshold, 'id'>) => void;
  metrics?: { value: string; label: string }[];
  domains?: string[];
}

// Component 44: Alert Threshold Configuration
export function AlertThresholds({
  thresholds,
  onThresholdChange,
  onThresholdDelete,
  onThresholdCreate,
  metrics = defaultMetrics,
  domains = ['military', 'economic', 'cyber', 'political', 'social'],
}: AlertThresholdsProps) {
  const [isCreating, setIsCreating] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [newThreshold, setNewThreshold] = useState<Partial<AlertThreshold>>({
    name: '',
    metric: 'risk_score',
    operator: 'gt',
    value: 0.7,
    severity: 'warning',
    enabled: true,
    notifyChannels: ['slack'],
    cooldownMinutes: 15,
  });

  const operatorLabels = {
    gt: '>',
    lt: '<',
    eq: '=',
    gte: '≥',
    lte: '≤',
    between: '↔',
  };

  const severityConfig = {
    info: { bg: 'bg-cyan-500/20', text: 'text-cyan-400', border: 'border-cyan-500/50' },
    warning: { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/50' },
    critical: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/50' },
  };

  const handleCreate = () => {
    if (!newThreshold.name || !newThreshold.metric) return;

    onThresholdCreate?.(newThreshold as Omit<AlertThreshold, 'id'>);
    setNewThreshold({
      name: '',
      metric: 'risk_score',
      operator: 'gt',
      value: 0.7,
      severity: 'warning',
      enabled: true,
      notifyChannels: ['slack'],
      cooldownMinutes: 15,
    });
    setIsCreating(false);
  };

  return (
    <div className="bg-slate-900/50 rounded-lg border border-slate-700">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-700">
        <div>
          <h3 className="text-sm font-medium text-slate-200">Alert Thresholds</h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Configure automated alerts based on risk metrics
          </p>
        </div>
        <button
          onClick={() => setIsCreating(true)}
          className="px-3 py-1.5 bg-cyan-500/20 text-cyan-400 rounded text-sm font-medium hover:bg-cyan-500/30 transition-colors"
        >
          + New Alert
        </button>
      </div>

      {/* Threshold List */}
      <div className="divide-y divide-slate-800">
        {thresholds.map((threshold) => {
          const sev = severityConfig[threshold.severity];
          const isEditing = editingId === threshold.id;

          return (
            <div
              key={threshold.id}
              className={`p-4 transition-colors ${threshold.enabled ? '' : 'opacity-50'}`}
            >
              {isEditing ? (
                <ThresholdEditor
                  threshold={threshold}
                  metrics={metrics}
                  domains={domains}
                  onSave={(changes) => {
                    onThresholdChange?.(threshold.id, changes);
                    setEditingId(null);
                  }}
                  onCancel={() => setEditingId(null)}
                />
              ) : (
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    {/* Toggle */}
                    <button
                      onClick={() => onThresholdChange?.(threshold.id, { enabled: !threshold.enabled })}
                      className={`w-10 h-5 rounded-full relative transition-colors ${
                        threshold.enabled ? 'bg-cyan-500' : 'bg-slate-700'
                      }`}
                    >
                      <span
                        className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                          threshold.enabled ? 'left-5' : 'left-0.5'
                        }`}
                      />
                    </button>

                    {/* Details */}
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-slate-200">{threshold.name}</span>
                        <span className={`px-1.5 py-0.5 rounded text-xs ${sev.bg} ${sev.text}`}>
                          {threshold.severity.toUpperCase()}
                        </span>
                        {threshold.domain && (
                          <span className="px-1.5 py-0.5 rounded text-xs bg-slate-700 text-slate-400">
                            {threshold.domain}
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-slate-500 mt-1 font-mono">
                        {threshold.metric} {operatorLabels[threshold.operator]} {threshold.value}
                        {threshold.operator === 'between' && ` AND ${threshold.secondValue}`}
                      </div>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-2">
                    {/* Notification channels */}
                    <div className="flex gap-1">
                      {threshold.notifyChannels.map((ch) => (
                        <span
                          key={ch}
                          className="w-6 h-6 rounded flex items-center justify-center bg-slate-800 text-slate-400 text-xs"
                          title={ch}
                        >
                          {ch === 'email' ? '✉' : ch === 'slack' ? '#' : ch === 'sms' ? '📱' : '🔗'}
                        </span>
                      ))}
                    </div>

                    {/* Last triggered */}
                    {threshold.lastTriggered && (
                      <span className="text-xs text-slate-500">
                        Last: {new Date(threshold.lastTriggered).toLocaleDateString()}
                      </span>
                    )}

                    <button
                      onClick={() => setEditingId(threshold.id)}
                      className="p-1.5 text-slate-400 hover:text-slate-200 transition-colors"
                    >
                      ✎
                    </button>
                    <button
                      onClick={() => onThresholdDelete?.(threshold.id)}
                      className="p-1.5 text-slate-400 hover:text-red-400 transition-colors"
                    >
                      ✕
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })}

        {thresholds.length === 0 && !isCreating && (
          <div className="p-8 text-center text-slate-500">
            No alert thresholds configured. Create one to get started.
          </div>
        )}

        {/* Create new threshold */}
        {isCreating && (
          <div className="p-4 bg-slate-800/50">
            <ThresholdEditor
              threshold={newThreshold as AlertThreshold}
              metrics={metrics}
              domains={domains}
              onSave={(changes) => {
                setNewThreshold((prev) => ({ ...prev, ...changes }));
                handleCreate();
              }}
              onCancel={() => setIsCreating(false)}
              isNew
            />
          </div>
        )}
      </div>
    </div>
  );
}

// Threshold editor component
function ThresholdEditor({
  threshold,
  metrics,
  domains,
  onSave,
  onCancel,
  isNew = false,
}: {
  threshold: AlertThreshold | Partial<AlertThreshold>;
  metrics: { value: string; label: string }[];
  domains: string[];
  onSave: (changes: Partial<AlertThreshold>) => void;
  onCancel: () => void;
  isNew?: boolean;
}) {
  const [form, setForm] = useState(threshold);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        {/* Name */}
        <div>
          <label className="block text-xs text-slate-400 mb-1">Name</label>
          <input
            type="text"
            value={form.name || ''}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            className="w-full px-3 py-1.5 bg-slate-800 border border-slate-600 rounded text-sm text-slate-200 focus:border-cyan-500 focus:outline-none"
            placeholder="High Risk Alert"
          />
        </div>

        {/* Metric */}
        <div>
          <label className="block text-xs text-slate-400 mb-1">Metric</label>
          <select
            value={form.metric || 'risk_score'}
            onChange={(e) => setForm({ ...form, metric: e.target.value })}
            className="w-full px-3 py-1.5 bg-slate-800 border border-slate-600 rounded text-sm text-slate-200 focus:border-cyan-500 focus:outline-none"
          >
            {metrics.map((m) => (
              <option key={m.value} value={m.value}>{m.label}</option>
            ))}
          </select>
        </div>

        {/* Operator & Value */}
        <div className="flex gap-2">
          <div className="w-20">
            <label className="block text-xs text-slate-400 mb-1">Operator</label>
            <select
              value={form.operator || 'gt'}
              onChange={(e) => setForm({ ...form, operator: e.target.value as AlertThreshold['operator'] })}
              className="w-full px-2 py-1.5 bg-slate-800 border border-slate-600 rounded text-sm text-slate-200 focus:border-cyan-500 focus:outline-none"
            >
              <option value="gt">&gt;</option>
              <option value="gte">≥</option>
              <option value="lt">&lt;</option>
              <option value="lte">≤</option>
              <option value="eq">=</option>
              <option value="between">↔</option>
            </select>
          </div>
          <div className="flex-1">
            <label className="block text-xs text-slate-400 mb-1">Value</label>
            <input
              type="number"
              step="0.01"
              value={form.value ?? 0.7}
              onChange={(e) => setForm({ ...form, value: parseFloat(e.target.value) })}
              className="w-full px-3 py-1.5 bg-slate-800 border border-slate-600 rounded text-sm text-slate-200 focus:border-cyan-500 focus:outline-none"
            />
          </div>
          {form.operator === 'between' && (
            <div className="flex-1">
              <label className="block text-xs text-slate-400 mb-1">Max</label>
              <input
                type="number"
                step="0.01"
                value={form.secondValue ?? 1}
                onChange={(e) => setForm({ ...form, secondValue: parseFloat(e.target.value) })}
                className="w-full px-3 py-1.5 bg-slate-800 border border-slate-600 rounded text-sm text-slate-200 focus:border-cyan-500 focus:outline-none"
              />
            </div>
          )}
        </div>

        {/* Severity */}
        <div>
          <label className="block text-xs text-slate-400 mb-1">Severity</label>
          <select
            value={form.severity || 'warning'}
            onChange={(e) => setForm({ ...form, severity: e.target.value as AlertThreshold['severity'] })}
            className="w-full px-3 py-1.5 bg-slate-800 border border-slate-600 rounded text-sm text-slate-200 focus:border-cyan-500 focus:outline-none"
          >
            <option value="info">Info</option>
            <option value="warning">Warning</option>
            <option value="critical">Critical</option>
          </select>
        </div>

        {/* Domain */}
        <div>
          <label className="block text-xs text-slate-400 mb-1">Domain (optional)</label>
          <select
            value={form.domain || ''}
            onChange={(e) => setForm({ ...form, domain: e.target.value || undefined })}
            className="w-full px-3 py-1.5 bg-slate-800 border border-slate-600 rounded text-sm text-slate-200 focus:border-cyan-500 focus:outline-none"
          >
            <option value="">All domains</option>
            {domains.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </div>

        {/* Cooldown */}
        <div>
          <label className="block text-xs text-slate-400 mb-1">Cooldown (min)</label>
          <input
            type="number"
            value={form.cooldownMinutes ?? 15}
            onChange={(e) => setForm({ ...form, cooldownMinutes: parseInt(e.target.value) })}
            className="w-full px-3 py-1.5 bg-slate-800 border border-slate-600 rounded text-sm text-slate-200 focus:border-cyan-500 focus:outline-none"
          />
        </div>
      </div>

      {/* Notification channels */}
      <div>
        <label className="block text-xs text-slate-400 mb-2">Notify via</label>
        <div className="flex gap-2">
          {(['email', 'slack', 'sms', 'webhook'] as const).map((ch) => (
            <button
              key={ch}
              onClick={() => {
                const channels = form.notifyChannels || [];
                const newChannels = channels.includes(ch)
                  ? channels.filter((c) => c !== ch)
                  : [...channels, ch];
                setForm({ ...form, notifyChannels: newChannels });
              }}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                form.notifyChannels?.includes(ch)
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50'
                  : 'bg-slate-800 text-slate-400 border border-slate-600'
              }`}
            >
              {ch === 'email' ? '✉ Email' : ch === 'slack' ? '# Slack' : ch === 'sms' ? '📱 SMS' : '🔗 Webhook'}
            </button>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="flex justify-end gap-2 pt-2">
        <button
          onClick={onCancel}
          className="px-3 py-1.5 text-sm text-slate-400 hover:text-slate-200 transition-colors"
        >
          Cancel
        </button>
        <button
          onClick={() => onSave(form)}
          className="px-4 py-1.5 bg-cyan-500 text-slate-900 rounded text-sm font-medium hover:bg-cyan-400 transition-colors"
        >
          {isNew ? 'Create Alert' : 'Save Changes'}
        </button>
      </div>
    </div>
  );
}

// Default metrics
const defaultMetrics = [
  { value: 'risk_score', label: 'Risk Score' },
  { value: 'cascade_probability', label: 'Cascade Probability' },
  { value: 'confidence', label: 'Confidence Level' },
  { value: 'velocity', label: 'Velocity Index' },
  { value: 'sentiment', label: 'Sentiment Score' },
  { value: 'entity_mentions', label: 'Entity Mentions' },
  { value: 'source_count', label: 'Source Count' },
];

// Mock data
export const mockThresholds: AlertThreshold[] = [
  {
    id: '1',
    name: 'Critical Risk Alert',
    metric: 'risk_score',
    operator: 'gte',
    value: 0.85,
    severity: 'critical',
    enabled: true,
    notifyChannels: ['slack', 'email'],
    cooldownMinutes: 5,
    lastTriggered: '2024-01-15T10:30:00Z',
  },
  {
    id: '2',
    name: 'High Cascade Probability',
    metric: 'cascade_probability',
    operator: 'gt',
    value: 0.7,
    severity: 'warning',
    enabled: true,
    domain: 'military',
    notifyChannels: ['slack'],
    cooldownMinutes: 15,
  },
  {
    id: '3',
    name: 'Low Confidence Warning',
    metric: 'confidence',
    operator: 'lt',
    value: 0.4,
    severity: 'info',
    enabled: false,
    notifyChannels: ['email'],
    cooldownMinutes: 60,
  },
];
