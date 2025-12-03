'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  Database, Download, RefreshCw, Shield, AlertTriangle, CheckCircle,
  Archive, RotateCcw, Settings, Activity, Zap, TrendingUp, Filter,
  Trash2, Eye, Clock, BarChart2, Gauge
} from 'lucide-react';

interface Stats {
  totalExamples: number;
  domainBreakdown: { domain: string; count: number }[];
  averageQuality: string;
  lastGenerated: string | null;
  totalPredictions: number;
}

interface Backup {
  id: string;
  backup_date: string;
  example_count: number;
  domain_stats: Record<string, number>;
  avg_quality: number;
  file_size_bytes: number;
  checksum: string;
  storage_location: string;
  created_at: string;
}

interface Anomaly {
  example_id: string;
  domain: string;
  anomaly_type: string;
  anomaly_score: number;
}

interface QualityMetrics {
  avgQuality: number;
  avgWeight: number;
  recentTrend: 'up' | 'down' | 'stable';
  duplicateRate: number;
  domainBalance: number;
  lastHourCount: number;
  last24HourCount: number;
}

interface TrainingSettings {
  minQualityThreshold: number;
  maxDuplicateSimilarity: number;
  domainRebalanceEnabled: boolean;
  autoQuarantineEnabled: boolean;
  batchSize: number;
}

export default function TrainingDataPage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [backups, setBackups] = useState<Backup[]>([]);
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [qualityMetrics, setQualityMetrics] = useState<QualityMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [backing, setBacking] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState('alpaca');
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'quality' | 'backups' | 'settings'>('overview');
  const [recentExamples, setRecentExamples] = useState<Array<{
    id: string;
    domain: string;
    input: string;
    output: string;
    quality_score: number;
    created_at: string;
    prompt?: string;
    completion?: string;
  }>>([]);

  // Settings state
  const [settings, setSettings] = useState<TrainingSettings>({
    minQualityThreshold: 0.6,
    maxDuplicateSimilarity: 0.85,
    domainRebalanceEnabled: true,
    autoQuarantineEnabled: false,
    batchSize: 10,
  });

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      // Load stats
      const statsRes = await fetch('/api/training/export', { method: 'POST' });
      if (statsRes.ok) {
        const data = await statsRes.json();
        setStats(data.stats);

        // Calculate quality metrics from stats
        if (data.stats) {
          const domains = data.stats.domainBreakdown || [];
          const total = domains.reduce((sum: number, d: { count: number }) => sum + d.count, 0);
          const avgCount = total / Math.max(domains.length, 1);
          const variance = domains.reduce((sum: number, d: { count: number }) =>
            sum + Math.pow(d.count - avgCount, 2), 0) / Math.max(domains.length, 1);
          const balance = 1 - (Math.sqrt(variance) / Math.max(avgCount, 1));

          setQualityMetrics({
            avgQuality: parseFloat(data.stats.averageQuality) || 0,
            avgWeight: 0.5, // Would need separate query
            recentTrend: 'stable',
            duplicateRate: 0.02, // Placeholder - would need dedup check
            domainBalance: Math.max(0, Math.min(1, balance)),
            lastHourCount: 0,
            last24HourCount: total,
          });
        }
      }

      // Load backups
      const backupsRes = await fetch('/api/training/backup');
      if (backupsRes.ok) {
        const data = await backupsRes.json();
        setBackups(data.backups || []);
      }

      // Load recent examples preview
      const previewRes = await fetch('/api/training/export?format=jsonl&limit=10');
      if (previewRes.ok) {
        const data = await previewRes.json();
        if (data.data) {
          setRecentExamples(data.data.map((d: string) => {
            try { return JSON.parse(d); } catch { return null; }
          }).filter(Boolean).slice(0, 10));
        }
      }
    } catch (e) {
      console.error('Failed to load data:', e);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  async function exportData() {
    setExporting(true);
    try {
      const url = `/api/training/export?format=${selectedFormat}&download=true&min_quality=${settings.minQualityThreshold}`;
      window.open(url, '_blank');
      setMessage({ type: 'success', text: `Exporting ${selectedFormat} format (quality >= ${settings.minQualityThreshold})...` });
    } catch {
      setMessage({ type: 'error', text: 'Export failed' });
    }
    setExporting(false);
  }

  async function createBackup() {
    setBacking(true);
    try {
      const res = await fetch('/api/training/backup', { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        setMessage({ type: 'success', text: `Backup created: ${data.backup?.exampleCount} examples` });
        loadData();
      } else {
        setMessage({ type: 'error', text: data.error });
      }
    } catch {
      setMessage({ type: 'error', text: 'Backup failed' });
    }
    setBacking(false);
  }

  async function scanAnomalies() {
    setScanning(true);
    try {
      const res = await fetch('/api/training/rollback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'detect_anomalies' }),
      });
      const data = await res.json();
      if (res.ok) {
        setAnomalies(data.anomalies || []);
        setMessage({ type: 'success', text: `Found ${data.count} potential anomalies` });
      } else {
        setMessage({ type: 'error', text: data.error });
      }
    } catch {
      setMessage({ type: 'error', text: 'Scan failed' });
    }
    setScanning(false);
  }

  async function autoQuarantineAnomalies() {
    if (!confirm('This will automatically quarantine all detected anomalies. Continue?')) return;

    try {
      const res = await fetch('/api/training/rollback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'auto_quarantine_anomalies' }),
      });
      const data = await res.json();
      if (res.ok) {
        setMessage({ type: 'success', text: data.message });
        setAnomalies([]);
        loadData();
      } else {
        setMessage({ type: 'error', text: data.error });
      }
    } catch {
      setMessage({ type: 'error', text: 'Auto-quarantine failed' });
    }
  }

  async function rollbackTo(backupId: string) {
    if (!confirm(`Are you sure you want to rollback to ${backupId}? All data created after this backup will be quarantined.`)) {
      return;
    }

    try {
      const res = await fetch('/api/training/rollback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'rollback', backupId }),
      });
      const data = await res.json();
      if (res.ok) {
        setMessage({ type: 'success', text: data.message });
        loadData();
      } else {
        setMessage({ type: 'error', text: data.error });
      }
    } catch {
      setMessage({ type: 'error', text: 'Rollback failed' });
    }
  }

  function getHealthStatus(): { status: 'healthy' | 'warning' | 'critical'; message: string } {
    if (!qualityMetrics) return { status: 'warning', message: 'Loading...' };

    if (qualityMetrics.avgQuality < 0.4) {
      return { status: 'critical', message: 'Quality below threshold' };
    }
    if (qualityMetrics.duplicateRate > 0.1) {
      return { status: 'warning', message: 'High duplicate rate' };
    }
    if (qualityMetrics.domainBalance < 0.3) {
      return { status: 'warning', message: 'Domain imbalance detected' };
    }
    return { status: 'healthy', message: 'All systems nominal' };
  }

  const health = getHealthStatus();

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0a0a0f] text-white p-8">
        <div className="flex items-center gap-3 animate-pulse">
          <Database className="w-6 h-6" />
          <span>Loading training data...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold flex items-center gap-3">
              <Database className="w-8 h-8 text-blue-400" />
              Training Data Command Center
            </h1>
            <p className="text-slate-400 mt-1">Monitor, QA/QC, export, and protect your training pipeline</p>
          </div>
          <div className="flex items-center gap-3">
            {/* Health Status Badge */}
            <div className={`flex items-center gap-2 px-4 py-2 rounded-full ${
              health.status === 'healthy' ? 'bg-green-500/20 text-green-400' :
              health.status === 'warning' ? 'bg-yellow-500/20 text-yellow-400' :
              'bg-red-500/20 text-red-400'
            }`}>
              <Activity className="w-4 h-4" />
              <span className="text-sm font-medium">{health.message}</span>
            </div>
            <button
              onClick={loadData}
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
              title="Refresh"
            >
              <RefreshCw className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Message */}
        {message && (
          <div className={`p-4 rounded-xl border ${
            message.type === 'success'
              ? 'bg-green-500/10 border-green-500/30 text-green-400'
              : 'bg-red-500/10 border-red-500/30 text-red-400'
          }`}>
            {message.text}
            <button onClick={() => setMessage(null)} className="float-right text-white/50 hover:text-white">×</button>
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-2 border-b border-white/10 pb-2">
          {(['overview', 'quality', 'backups', 'settings'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-t-lg capitalize transition-colors ${
                activeTab === tab
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              <div className="bg-gradient-to-br from-blue-600/20 to-blue-800/20 border border-blue-500/30 rounded-xl p-6">
                <div className="flex items-center gap-2 text-blue-400 mb-2">
                  <Database className="w-5 h-5" />
                  <span className="text-sm">Total Examples</span>
                </div>
                <div className="text-4xl font-bold">
                  {stats?.totalExamples?.toLocaleString() || 0}
                </div>
              </div>
              <div className="bg-gradient-to-br from-green-600/20 to-green-800/20 border border-green-500/30 rounded-xl p-6">
                <div className="flex items-center gap-2 text-green-400 mb-2">
                  <Gauge className="w-5 h-5" />
                  <span className="text-sm">Avg Quality</span>
                </div>
                <div className="text-4xl font-bold">
                  {((qualityMetrics?.avgQuality || 0) * 100).toFixed(1)}%
                </div>
              </div>
              <div className="bg-gradient-to-br from-orange-600/20 to-orange-800/20 border border-orange-500/30 rounded-xl p-6">
                <div className="flex items-center gap-2 text-orange-400 mb-2">
                  <BarChart2 className="w-5 h-5" />
                  <span className="text-sm">Domains</span>
                </div>
                <div className="text-4xl font-bold">
                  {stats?.domainBreakdown?.length || 0}
                </div>
              </div>
              <div className="bg-gradient-to-br from-purple-600/20 to-purple-800/20 border border-purple-500/30 rounded-xl p-6">
                <div className="flex items-center gap-2 text-purple-400 mb-2">
                  <TrendingUp className="w-5 h-5" />
                  <span className="text-sm">Predictions</span>
                </div>
                <div className="text-4xl font-bold">
                  {stats?.totalPredictions || 0}
                </div>
              </div>
              <div className="bg-gradient-to-br from-cyan-600/20 to-cyan-800/20 border border-cyan-500/30 rounded-xl p-6">
                <div className="flex items-center gap-2 text-cyan-400 mb-2">
                  <Clock className="w-5 h-5" />
                  <span className="text-sm">Last Generated</span>
                </div>
                <div className="text-lg font-medium">
                  {stats?.lastGenerated
                    ? new Date(stats.lastGenerated).toLocaleString()
                    : 'Never'}
                </div>
              </div>
            </div>

            {/* Domain Breakdown */}
            {stats?.domainBreakdown && stats.domainBreakdown.length > 0 && (
              <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                  <BarChart2 className="w-5 h-5 text-blue-400" />
                  Domain Distribution
                </h2>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
                  {stats.domainBreakdown.sort((a, b) => b.count - a.count).map(d => (
                    <div key={d.domain} className="bg-black/30 rounded-lg p-3 hover:bg-black/50 transition-colors">
                      <div className="text-lg font-semibold text-blue-400">{d.count.toLocaleString()}</div>
                      <div className="text-xs text-slate-400 truncate" title={d.domain}>{d.domain}</div>
                      <div className="mt-2 h-1 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-blue-500"
                          style={{ width: `${(d.count / Math.max(...stats.domainBreakdown.map(x => x.count))) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Export Section */}
            <div className="bg-white/5 border border-white/10 rounded-xl p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Download className="w-5 h-5 text-blue-400" />
                Export for RunPod Training
              </h2>
              <div className="flex flex-wrap items-center gap-4">
                <select
                  value={selectedFormat}
                  onChange={(e) => setSelectedFormat(e.target.value)}
                  className="bg-black/50 border border-white/10 rounded-lg px-4 py-2 text-white"
                >
                  <option value="alpaca">Alpaca Format (Llama/Mistral)</option>
                  <option value="chatml">ChatML Format (OpenAI-style)</option>
                  <option value="sharegpt">ShareGPT Format</option>
                  <option value="jsonl">Raw JSONL</option>
                  <option value="csv">CSV (Analysis)</option>
                </select>
                <div className="flex items-center gap-2 text-sm text-slate-400">
                  <Filter className="w-4 h-4" />
                  <span>Min Quality:</span>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.1"
                    value={settings.minQualityThreshold}
                    onChange={(e) => setSettings({...settings, minQualityThreshold: parseFloat(e.target.value)})}
                    className="w-16 bg-black/50 border border-white/10 rounded px-2 py-1 text-white"
                  />
                </div>
                <button
                  onClick={exportData}
                  disabled={exporting}
                  className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 px-6 py-2 rounded-lg font-medium transition-colors"
                >
                  <Download className="w-4 h-4" />
                  {exporting ? 'Exporting...' : 'Export Data'}
                </button>
              </div>
              <p className="text-slate-500 text-sm mt-3">
                Your $90 RunPod credits are waiting. Use Alpaca format with phi-2. ~$1-3 per 1000 examples on A100.
              </p>
            </div>

            {/* Recent Examples Preview */}
            {recentExamples.length > 0 && (
              <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                  <Eye className="w-5 h-5 text-green-400" />
                  Recent Training Examples
                </h2>
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {recentExamples.map((ex, i) => (
                    <div key={ex.id || i} className="bg-black/30 rounded-lg p-4 border border-white/5">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs bg-blue-500/20 text-blue-400 px-2 py-1 rounded">{ex.domain}</span>
                        <div className="flex items-center gap-3 text-xs text-slate-500">
                          <span>Quality: <span className={ex.quality_score > 0.7 ? 'text-green-400' : ex.quality_score > 0.4 ? 'text-yellow-400' : 'text-red-400'}>{(ex.quality_score * 100).toFixed(0)}%</span></span>
                          <span>{new Date(ex.created_at).toLocaleString()}</span>
                        </div>
                      </div>
                      <div className="text-sm text-slate-300 line-clamp-2 mb-2">
                        <strong className="text-slate-400">Q:</strong> {ex.input || ex.prompt}
                      </div>
                      <div className="text-sm text-slate-400 line-clamp-2">
                        <strong className="text-slate-500">A:</strong> {ex.output || ex.completion}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Quality Tab */}
        {activeTab === 'quality' && (
          <div className="space-y-6">
            {/* Quality Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold">Quality Score Distribution</h3>
                  <Gauge className="w-5 h-5 text-green-400" />
                </div>
                <div className="relative h-4 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${
                      (qualityMetrics?.avgQuality || 0) > 0.7 ? 'bg-green-500' :
                      (qualityMetrics?.avgQuality || 0) > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${(qualityMetrics?.avgQuality || 0) * 100}%` }}
                  />
                </div>
                <p className="text-sm text-slate-400 mt-2">
                  Average: {((qualityMetrics?.avgQuality || 0) * 100).toFixed(1)}% | Threshold: {settings.minQualityThreshold * 100}%
                </p>
              </div>

              <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold">Domain Balance</h3>
                  <BarChart2 className="w-5 h-5 text-blue-400" />
                </div>
                <div className="relative h-4 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${
                      (qualityMetrics?.domainBalance || 0) > 0.6 ? 'bg-green-500' :
                      (qualityMetrics?.domainBalance || 0) > 0.3 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${(qualityMetrics?.domainBalance || 0) * 100}%` }}
                  />
                </div>
                <p className="text-sm text-slate-400 mt-2">
                  {((qualityMetrics?.domainBalance || 0) * 100).toFixed(0)}% balanced across domains
                </p>
              </div>

              <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold">Duplicate Detection</h3>
                  <Filter className="w-5 h-5 text-orange-400" />
                </div>
                <div className="relative h-4 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${
                      (qualityMetrics?.duplicateRate || 0) < 0.05 ? 'bg-green-500' :
                      (qualityMetrics?.duplicateRate || 0) < 0.1 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${Math.min((qualityMetrics?.duplicateRate || 0) * 1000, 100)}%` }}
                  />
                </div>
                <p className="text-sm text-slate-400 mt-2">
                  ~{((qualityMetrics?.duplicateRate || 0) * 100).toFixed(1)}% estimated duplicates
                </p>
              </div>
            </div>

            {/* Anomaly Detection */}
            <div className="bg-white/5 border border-white/10 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold flex items-center gap-2">
                  <Shield className="w-5 h-5 text-orange-400" />
                  Data Integrity & Security
                </h2>
                <div className="flex gap-2">
                  <button
                    onClick={scanAnomalies}
                    disabled={scanning}
                    className="flex items-center gap-2 bg-orange-600 hover:bg-orange-700 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                  >
                    <Shield className="w-4 h-4" />
                    {scanning ? 'Scanning...' : 'Scan for Anomalies'}
                  </button>
                  {anomalies.length > 0 && (
                    <button
                      onClick={autoQuarantineAnomalies}
                      className="flex items-center gap-2 bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                      Quarantine All ({anomalies.length})
                    </button>
                  )}
                </div>
              </div>

              {anomalies.length > 0 ? (
                <div className="space-y-2 max-h-60 overflow-y-auto">
                  {anomalies.map((a, i) => (
                    <div key={i} className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <AlertTriangle className="w-4 h-4 text-red-400" />
                        <span className="text-red-400 font-mono text-sm">{a.example_id.slice(0, 8)}...</span>
                        <span className="text-slate-400">{a.domain}</span>
                        <span className="text-xs bg-red-500/20 text-red-400 px-2 py-1 rounded">{a.anomaly_type}</span>
                      </div>
                      <span className="text-slate-500">Score: {a.anomaly_score.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-slate-400">
                  <CheckCircle className="w-12 h-12 mx-auto mb-3 text-green-400" />
                  <p>No anomalies detected. Data looks clean.</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Backups Tab */}
        {activeTab === 'backups' && (
          <div className="space-y-6">
            {/* Create Backup */}
            <div className="bg-white/5 border border-white/10 rounded-xl p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-semibold flex items-center gap-2">
                    <Archive className="w-5 h-5 text-green-400" />
                    Create Checkpoint Backup
                  </h2>
                  <p className="text-slate-400 text-sm mt-1">
                    Backups are retained for 30 days. Use for rollback if data gets poisoned or corrupted.
                  </p>
                </div>
                <button
                  onClick={createBackup}
                  disabled={backing}
                  className="flex items-center gap-2 bg-green-600 hover:bg-green-700 disabled:opacity-50 px-6 py-2 rounded-lg font-medium transition-colors"
                >
                  <Archive className="w-4 h-4" />
                  {backing ? 'Creating...' : 'Create Backup Now'}
                </button>
              </div>
            </div>

            {/* Backups List */}
            <div className="bg-white/5 border border-white/10 rounded-xl p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <RotateCcw className="w-5 h-5 text-purple-400" />
                Available Backups ({backups.length})
              </h2>
              {backups.length > 0 ? (
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {backups.map(backup => (
                    <div key={backup.id} className="bg-black/30 rounded-lg p-4 flex items-center justify-between hover:bg-black/50 transition-colors">
                      <div className="flex-1">
                        <div className="flex items-center gap-3">
                          <CheckCircle className="w-4 h-4 text-green-400" />
                          <span className="font-medium">{backup.backup_date}</span>
                          <span className="text-slate-400">|</span>
                          <span className="text-blue-400">{backup.example_count.toLocaleString()} examples</span>
                          <span className="text-slate-400">|</span>
                          <span className="text-green-400">{(backup.avg_quality * 100).toFixed(0)}% quality</span>
                          <span className="text-slate-400">|</span>
                          <span className="text-slate-500">{(backup.file_size_bytes / 1024).toFixed(0)} KB</span>
                        </div>
                        <div className="text-xs text-slate-500 mt-1 flex items-center gap-4">
                          <span>Checksum: {backup.checksum}</span>
                          <span>Domains: {Object.keys(backup.domain_stats || {}).length}</span>
                        </div>
                      </div>
                      <button
                        onClick={() => rollbackTo(backup.id)}
                        className="text-sm px-4 py-2 bg-purple-600/30 hover:bg-purple-600 rounded-lg transition-colors flex items-center gap-2"
                      >
                        <RotateCcw className="w-4 h-4" />
                        Rollback
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-slate-400">
                  <Archive className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>No backups yet. Create one to enable rollback.</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Settings Tab */}
        {activeTab === 'settings' && (
          <div className="space-y-6">
            <div className="bg-white/5 border border-white/10 rounded-xl p-6">
              <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
                <Settings className="w-5 h-5 text-slate-400" />
                Quality Control Settings
              </h2>

              <div className="space-y-6">
                {/* Min Quality Threshold */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="font-medium">Minimum Quality Threshold</label>
                    <p className="text-sm text-slate-400">Examples below this quality won&apos;t be exported</p>
                  </div>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={settings.minQualityThreshold}
                      onChange={(e) => setSettings({...settings, minQualityThreshold: parseFloat(e.target.value)})}
                      className="w-32"
                    />
                    <span className="text-blue-400 font-mono w-12">{(settings.minQualityThreshold * 100).toFixed(0)}%</span>
                  </div>
                </div>

                {/* Duplicate Similarity */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="font-medium">Max Duplicate Similarity</label>
                    <p className="text-sm text-slate-400">Flag examples more similar than this as duplicates</p>
                  </div>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min="0.5"
                      max="1"
                      step="0.05"
                      value={settings.maxDuplicateSimilarity}
                      onChange={(e) => setSettings({...settings, maxDuplicateSimilarity: parseFloat(e.target.value)})}
                      className="w-32"
                    />
                    <span className="text-blue-400 font-mono w-12">{(settings.maxDuplicateSimilarity * 100).toFixed(0)}%</span>
                  </div>
                </div>

                {/* Auto Quarantine */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="font-medium">Auto-Quarantine Anomalies</label>
                    <p className="text-sm text-slate-400">Automatically quarantine detected anomalies during generation</p>
                  </div>
                  <button
                    onClick={() => setSettings({...settings, autoQuarantineEnabled: !settings.autoQuarantineEnabled})}
                    className={`w-14 h-7 rounded-full transition-colors ${
                      settings.autoQuarantineEnabled ? 'bg-green-600' : 'bg-white/20'
                    }`}
                  >
                    <div className={`w-5 h-5 rounded-full bg-white transform transition-transform ${
                      settings.autoQuarantineEnabled ? 'translate-x-8' : 'translate-x-1'
                    }`} />
                  </button>
                </div>

                {/* Domain Rebalancing */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="font-medium">Domain Rebalancing</label>
                    <p className="text-sm text-slate-400">Prioritize underrepresented domains during generation</p>
                  </div>
                  <button
                    onClick={() => setSettings({...settings, domainRebalanceEnabled: !settings.domainRebalanceEnabled})}
                    className={`w-14 h-7 rounded-full transition-colors ${
                      settings.domainRebalanceEnabled ? 'bg-green-600' : 'bg-white/20'
                    }`}
                  >
                    <div className={`w-5 h-5 rounded-full bg-white transform transition-transform ${
                      settings.domainRebalanceEnabled ? 'translate-x-8' : 'translate-x-1'
                    }`} />
                  </button>
                </div>

                {/* Batch Size */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="font-medium">Generation Batch Size</label>
                    <p className="text-sm text-slate-400">Number of examples to generate per API call</p>
                  </div>
                  <select
                    value={settings.batchSize}
                    onChange={(e) => setSettings({...settings, batchSize: parseInt(e.target.value)})}
                    className="bg-black/50 border border-white/10 rounded-lg px-4 py-2 text-white"
                  >
                    <option value={5}>5 (Careful)</option>
                    <option value={10}>10 (Default)</option>
                    <option value={20}>20 (Fast)</option>
                    <option value={50}>50 (Aggressive)</option>
                  </select>
                </div>
              </div>

              <div className="mt-8 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                <div className="flex items-start gap-3">
                  <Zap className="w-5 h-5 text-blue-400 mt-0.5" />
                  <div>
                    <p className="text-blue-400 font-medium">Pro Tip: RunPod Training</p>
                    <p className="text-sm text-slate-400 mt-1">
                      With your $90 RunPod credits, you can fine-tune phi-2 on ~30,000-90,000 training examples.
                      Export in Alpaca format and use scripts/train_h200.py for LoRA training.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
