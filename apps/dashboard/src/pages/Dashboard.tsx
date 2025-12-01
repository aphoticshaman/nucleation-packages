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

interface DashboardProps {
  apiKey: string;
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

export function Dashboard({ apiKey }: DashboardProps) {
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
          <p className="font-mono text-[10px] text-surface-600 mt-0.5">
            {apiKey.slice(0, 15)}...
          </p>
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
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke="#0ea5e9"
                  fill="url(#phaseGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Sources Table */}
      <div className="glass-card">
        <div className="p-6 border-b border-surface-600/50">
          <h3 className="text-sm font-semibold text-white">Active Data Sources</h3>
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
              <td><span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-xs">Official</span></td>
              <td><span className="status-indicator status-healthy inline-block mr-2" /> Healthy</td>
              <td className="text-surface-400 mono">12s ago</td>
              <td className="text-emerald-400 mono">45ms</td>
              <td className="text-surface-400">10/sec</td>
            </tr>
            <tr>
              <td className="font-medium text-white">FRED</td>
              <td><span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-xs">Official</span></td>
              <td><span className="status-indicator status-healthy inline-block mr-2" /> Healthy</td>
              <td className="text-surface-400 mono">1m ago</td>
              <td className="text-emerald-400 mono">89ms</td>
              <td className="text-surface-400">120/min</td>
            </tr>
            <tr>
              <td className="font-medium text-white">The Guardian</td>
              <td><span className="px-2 py-0.5 rounded bg-lattice-500/10 text-lattice-400 text-xs">News</span></td>
              <td><span className="status-indicator status-healthy inline-block mr-2" /> Healthy</td>
              <td className="text-surface-400 mono">5m ago</td>
              <td className="text-amber-400 mono">234ms</td>
              <td className="text-surface-400">12/sec</td>
            </tr>
            <tr>
              <td className="font-medium text-white">Reddit</td>
              <td><span className="px-2 py-0.5 rounded bg-crystal-500/10 text-crystal-400 text-xs">Social</span></td>
              <td><span className="status-indicator status-warning inline-block mr-2" /> Degraded</td>
              <td className="text-surface-400 mono">2m ago</td>
              <td className="text-amber-400 mono">512ms</td>
              <td className="text-surface-400">100/min</td>
            </tr>
            <tr>
              <td className="font-medium text-white">Bluesky</td>
              <td><span className="px-2 py-0.5 rounded bg-crystal-500/10 text-crystal-400 text-xs">Social</span></td>
              <td><span className="status-indicator status-healthy inline-block mr-2" /> Healthy</td>
              <td className="text-surface-400 mono">30s ago</td>
              <td className="text-emerald-400 mono">67ms</td>
              <td className="text-surface-400">Unlimited</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Footer Attribution */}
      <div className="text-center text-xs text-surface-600 pt-4">
        <p>Data sources: SEC EDGAR (Public Domain) • FRED, Federal Reserve Bank of St. Louis • Powered by Guardian Open Platform</p>
        <p className="mt-1">LatticeForge v1.0.0 • Crystalline Labs LLC © 2025</p>
      </div>
    </div>
  );
}
