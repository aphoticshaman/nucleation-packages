import { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import { SignalsIcon, AlertIcon, CheckIcon } from '../components/Icons';
import type { Session } from '@supabase/supabase-js';

interface DashboardProps {
  session: Session;
}

// Mock data generator
function generateSignalData(points: number) {
  const data = [];
  let value = 50;
  for (let i = 0; i < points; i++) {
    value += (Math.random() - 0.5) * 10;
    value = Math.max(0, Math.min(100, value));
    data.push({
      time: `${i}:00`,
      value: Math.round(value * 10) / 10,
      phase: Math.random() * 0.5 + (value > 60 ? 0.3 : 0),
    });
  }
  return data;
}

const mockSignalData = generateSignalData(24);

const governanceControls = [
  {
    title: 'Access Governance',
    detail: 'SCIM-provisioned roles with hourly reconciliation',
    status: 'healthy',
    meta: 'Zero drift detected',
  },
  {
    title: 'Data Protection',
    detail: 'Column-level encryption and PII redaction live',
    status: 'healthy',
    meta: 'AES-256-GCM at rest',
  },
  {
    title: 'Audit Integrity',
    detail: 'Tamper-evident audit ledger streaming to SIEM',
    status: 'warning',
    meta: 'Syslog relay lag 220ms',
  },
];

const enterpriseReadiness = [
  {
    label: 'SLO Adherence',
    value: '99.94%',
    delta: '+0.12%',
    color: 'text-emerald-400',
    description: 'Error budget intact; priority burn alerts suppressed',
  },
  {
    label: 'Data Residency',
    value: 'Multi-region',
    delta: 'EU & US',
    color: 'text-lattice-400',
    description: 'Active-active storage with residency-aware routing',
  },
  {
    label: 'Compliance',
    value: 'SOC 2 Type II',
    delta: 'Renewal Q3',
    color: 'text-crystal-400',
    description: 'Controls validated; automations ready for next audit',
  },
];

export function Dashboard({ session }: DashboardProps) {
  const [currentPhase, setCurrentPhase] = useState(0.23);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentPhase(Math.random() * 0.4 + 0.1);
      setLastUpdate(new Date());
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const getPhaseClass = (phase: number) => {
    if (phase < 0.1) return 'phase-stable';
    if (phase < 0.3) return 'phase-normal';
    if (phase < 0.5) return 'phase-elevated';
    if (phase < 0.7) return 'phase-high';
    return 'phase-critical';
  };

  const getPhaseLabel = (phase: number) => {
    if (phase < 0.1) return 'STABLE';
    if (phase < 0.3) return 'NORMAL';
    if (phase < 0.5) return 'ELEVATED';
    if (phase < 0.7) return 'HIGH';
    return 'CRITICAL';
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Command Center</h1>
          <p className="text-sm text-lattice-400 mt-1">
            Real-time signal intelligence and phase detection
          </p>
        </div>
        <div className="text-right text-xs text-surface-500">
          <p>Last updated: {lastUpdate.toLocaleTimeString()}</p>
          <p className="font-mono text-[10px] text-surface-600 mt-0.5">{session.user.email}</p>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-4 gap-4">
        {/* Current Phase */}
        <div className="glass-card p-6">
          <div className="flex items-center justify-between mb-4">
            <span className="label">Current Phase</span>
            <span
              className={`px-2 py-0.5 rounded text-xs font-semibold border ${getPhaseClass(currentPhase)}`}
            >
              {getPhaseLabel(currentPhase)}
            </span>
          </div>
          <div className="metric-value">{(currentPhase * 100).toFixed(1)}%</div>
          <div className="metric-label mt-1">Variance Index</div>
        </div>

        {/* Active Sources */}
        <div className="glass-card p-6">
          <div className="flex items-center justify-between mb-4">
            <span className="label">Active Sources</span>
            <CheckIcon className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="metric-value">8</div>
          <div className="metric-label mt-1">
            <span className="metric-delta-positive">+2 this week</span>
          </div>
        </div>

        {/* Signals Processed */}
        <div className="glass-card p-6">
          <div className="flex items-center justify-between mb-4">
            <span className="label">Signals Processed</span>
            <SignalsIcon className="w-4 h-4 text-lattice-400" />
          </div>
          <div className="metric-value">24.8K</div>
          <div className="metric-label mt-1">Last 24 hours</div>
        </div>

        {/* Alerts */}
        <div className="glass-card p-6">
          <div className="flex items-center justify-between mb-4">
            <span className="label">Active Alerts</span>
            <AlertIcon className="w-4 h-4 text-amber-400" />
          </div>
          <div className="metric-value">3</div>
          <div className="metric-label mt-1">
            <span className="metric-delta-negative">1 critical</span>
          </div>
        </div>
      </div>

      {/* Enterprise posture */}
      <div className="grid grid-cols-3 gap-4">
        {enterpriseReadiness.map((item) => (
          <div key={item.label} className="glass-card-hover p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="label">{item.label}</span>
              <span className={`text-xs font-semibold ${item.color}`}>{item.delta}</span>
            </div>
            <div className="flex items-end justify-between">
              <div className="text-2xl font-semibold text-white">{item.value}</div>
              <span className="px-2 py-1 text-[11px] rounded bg-surface-700/80 text-lattice-100 border border-surface-600">
                Enterprise ready
              </span>
            </div>
            <p className="text-sm text-surface-400 mt-2 leading-5">{item.description}</p>
          </div>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-2 gap-6">
        {/* Signal Chart */}
        <div className="glass-card p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-sm font-semibold text-white">Fused Signal</h3>
            <div className="flex items-center gap-4 text-xs">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-lattice-400" />
                Value
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-crystal-400" />
                Phase
              </span>
            </div>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockSignalData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#22222e" />
                <XAxis dataKey="time" stroke="#7dd3fc" fontSize={10} />
                <YAxis stroke="#7dd3fc" fontSize={10} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a24',
                    border: '1px solid #2a2a38',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#0ea5e9"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="phase"
                  stroke="#a855f7"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Phase Heatmap */}
        <div className="glass-card p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-sm font-semibold text-white">Phase Detection History</h3>
            <span className="text-xs text-lattice-400">Last 7 days</span>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={mockSignalData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#22222e" />
                <XAxis dataKey="time" stroke="#7dd3fc" fontSize={10} />
                <YAxis stroke="#7dd3fc" fontSize={10} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a24',
                    border: '1px solid #2a2a38',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                <defs>
                  <linearGradient id="phaseGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <Area type="monotone" dataKey="value" stroke="#0ea5e9" fill="url(#phaseGradient)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Sources Table */}
      <div className="glass-card">
        <div className="p-6 border-b border-surface-600/50">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-white">Active Data Sources</h3>
              <p className="text-xs text-surface-500 mt-1">Telemetry normalized with automated data contracts</p>
            </div>
            <button className="btn-secondary h-9 px-3 text-xs">Add integration</button>
          </div>
        </div>
        <table className="data-grid">
          <thead>
            <tr>
              <th>Source</th>
              <th>Tier</th>
              <th>Status</th>
              <th>Last Fetch</th>
              <th>Latency</th>
              <th>Rate Limit</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="font-medium text-white">SEC EDGAR</td>
              <td>
                <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-xs">
                  Official
                </span>
              </td>
              <td>
                <span className="status-indicator status-healthy inline-block mr-2" /> Healthy
              </td>
              <td className="text-surface-400 mono">12s ago</td>
              <td className="text-emerald-400 mono">45ms</td>
              <td className="text-surface-400">10/sec</td>
            </tr>
            <tr>
              <td className="font-medium text-white">FRED</td>
              <td>
                <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-xs">
                  Official
                </span>
              </td>
              <td>
                <span className="status-indicator status-healthy inline-block mr-2" /> Healthy
              </td>
              <td className="text-surface-400 mono">1m ago</td>
              <td className="text-emerald-400 mono">89ms</td>
              <td className="text-surface-400">120/min</td>
            </tr>
            <tr>
              <td className="font-medium text-white">The Guardian</td>
              <td>
                <span className="px-2 py-0.5 rounded bg-lattice-500/10 text-lattice-400 text-xs">
                  News
                </span>
              </td>
              <td>
                <span className="status-indicator status-healthy inline-block mr-2" /> Healthy
              </td>
              <td className="text-surface-400 mono">5m ago</td>
              <td className="text-amber-400 mono">234ms</td>
              <td className="text-surface-400">12/sec</td>
            </tr>
            <tr>
              <td className="font-medium text-white">Reddit</td>
              <td>
                <span className="px-2 py-0.5 rounded bg-crystal-500/10 text-crystal-400 text-xs">
                  Social
                </span>
              </td>
              <td>
                <span className="status-indicator status-warning inline-block mr-2" /> Degraded
              </td>
              <td className="text-surface-400 mono">2m ago</td>
              <td className="text-amber-400 mono">512ms</td>
              <td className="text-surface-400">100/min</td>
            </tr>
            <tr>
              <td className="font-medium text-white">Bluesky</td>
              <td>
                <span className="px-2 py-0.5 rounded bg-crystal-500/10 text-crystal-400 text-xs">
                  Social
                </span>
              </td>
              <td>
                <span className="status-indicator status-healthy inline-block mr-2" /> Healthy
              </td>
              <td className="text-surface-400 mono">30s ago</td>
              <td className="text-emerald-400 mono">67ms</td>
              <td className="text-surface-400">Unlimited</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Governance & Runbooks */}
      <div className="grid grid-cols-3 gap-4">
        <div className="glass-card p-6 col-span-2">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-white">Operational Governance</h3>
              <p className="text-xs text-surface-500 mt-1">Guardrails tuned for regulated enterprise workloads</p>
            </div>
            <span className="text-xs px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
              Auto-remediation active
            </span>
          </div>
          <div className="space-y-3">
            {governanceControls.map((control) => (
              <div
                key={control.title}
                className="flex items-start justify-between p-4 bg-surface-800/60 rounded-lg border border-surface-700"
              >
                <div>
                  <p className="text-sm font-semibold text-white">{control.title}</p>
                  <p className="text-xs text-surface-400 mt-1">{control.detail}</p>
                  <p className="text-[11px] text-surface-500 mt-1">{control.meta}</p>
                </div>
                <div className="flex items-center gap-2">
                  <span
                    className={`status-indicator ${
                      control.status === 'warning' ? 'status-warning' : 'status-healthy'
                    }`}
                  />
                  <span className="text-xs text-surface-300 capitalize">{control.status}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="glass-card p-6">
          <h3 className="text-sm font-semibold text-white">Critical Runbook</h3>
          <p className="text-xs text-surface-500 mt-1 mb-4">
            Enterprise-ready steps to mitigate signal drift and maintain compliance posture.
          </p>
          <ol className="space-y-3 list-decimal list-inside text-sm text-surface-200">
            <li>
              <span className="font-semibold text-white">Quarantine anomalous sources</span>
              <p className="text-xs text-surface-500">Auto-revoke tokens and isolate collectors with high jitter.</p>
            </li>
            <li>
              <span className="font-semibold text-white">Escalate via PagerDuty</span>
              <p className="text-xs text-surface-500">Route Sev1 alerts to enterprise on-call rotation with context.</p>
            </li>
            <li>
              <span className="font-semibold text-white">Regenerate attestations</span>
              <p className="text-xs text-surface-500">Push compliance evidence to GRC vault and notify auditors.</p>
            </li>
          </ol>
          <button className="btn-primary w-full mt-6 text-sm">Launch automated playbook</button>
        </div>
      </div>

      {/* Footer Attribution */}
      <div className="text-center text-xs text-surface-600 pt-4">
        <p>
          Data sources: SEC EDGAR (Public Domain) • FRED, Federal Reserve Bank of St. Louis •
          Powered by Guardian Open Platform
        </p>
        <p className="mt-1">LatticeForge v1.0.0 • Crystalline Labs LLC © 2025</p>
      </div>
    </div>
  );
}
