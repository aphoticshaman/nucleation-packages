'use client';

import { useState, useEffect, useCallback } from 'react';
import { GlassCard } from '@/components/ui/GlassCard';

// Types
interface TrainingItem {
  id: string;
  source_type: string;
  source_id: string | null;
  target_system: 'elle' | 'guardian' | 'both';
  domain: string | null;
  input_text: string;
  expected_output: string | null;
  actual_output: string | null;
  training_type: string;
  quality_score: number | null;
  difficulty_level: string | null;
  tags: string[] | null;
  selected_for_export: boolean;
  export_priority: number;
  status: string;
  created_at: string;
  export_count: number;
}

interface Stats {
  total: number;
  elle: number;
  guardian: number;
  selected: number;
  approved: number;
  exported: number;
}

interface AuditEntry {
  id: string;
  action: string;
  performed_by_email: string;
  performed_by_name: string;
  performed_at: string;
  entity_type: string;
  entity_id: string;
  reason: string | null;
  previous_hash: string;
  entry_hash: string;
}

interface IntegrityCheck {
  is_valid: boolean;
  invalid_entries: string[];
  total_entries: number;
  verified_entries: number;
}

export default function TrainingDataPage() {
  // State
  const [items, setItems] = useState<TrainingItem[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [auditLog, setAuditLog] = useState<AuditEntry[]>([]);
  const [integrity, setIntegrity] = useState<IntegrityCheck | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [actionLoading, setActionLoading] = useState(false);

  // Filters
  const [targetFilter, setTargetFilter] = useState<string>('');
  const [domainFilter, setDomainFilter] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [showSelectedOnly, setShowSelectedOnly] = useState(false);

  // Modal state
  const [showExportModal, setShowExportModal] = useState(false);
  const [showAuditModal, setShowAuditModal] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedItem, setSelectedItem] = useState<TrainingItem | null>(null);

  // Export form
  const [exportName, setExportName] = useState('');
  const [exportFormat, setExportFormat] = useState('jsonl');
  const [exportTarget, setExportTarget] = useState('both');
  const [exportNotes, setExportNotes] = useState('');

  // Create form
  const [createForm, setCreateForm] = useState({
    targetSystem: 'both' as 'elle' | 'guardian' | 'both',
    domain: '',
    inputText: '',
    expectedOutput: '',
    trainingType: 'positive',
    qualityScore: 0.8,
    difficultyLevel: 'medium',
    tags: '',
  });

  // Fetch items
  const fetchItems = useCallback(async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams({ view: 'list' });
      if (targetFilter) params.set('target', targetFilter);
      if (domainFilter) params.set('domain', domainFilter);
      if (statusFilter) params.set('status', statusFilter);
      if (showSelectedOnly) params.set('selected', 'true');

      const response = await fetch(`/api/admin/training-items?${params}`);
      const data = await response.json();
      setItems(data.items || []);
    } catch (error) {
      console.error('Error fetching items:', error);
    } finally {
      setLoading(false);
    }
  }, [targetFilter, domainFilter, statusFilter, showSelectedOnly]);

  // Fetch stats
  const fetchStats = useCallback(async () => {
    try {
      const response = await fetch('/api/admin/training-items?view=stats');
      const data = await response.json();
      setStats(data.summary);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  }, []);

  // Fetch audit log
  const fetchAuditLog = useCallback(async () => {
    try {
      const response = await fetch('/api/admin/training-items?view=audit');
      const data = await response.json();
      setAuditLog(data.auditLog || []);
      setIntegrity(data.integrity);
    } catch (error) {
      console.error('Error fetching audit log:', error);
    }
  }, []);

  useEffect(() => {
    fetchItems();
    fetchStats();
  }, [fetchItems, fetchStats]);

  // Toggle item selection (for export)
  const toggleSelection = async (itemId: string, currentlySelected: boolean) => {
    try {
      await fetch('/api/admin/training-items', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'toggle_selection',
          itemId,
          selected: !currentlySelected,
        }),
      });
      fetchItems();
      fetchStats();
    } catch (error) {
      console.error('Error toggling selection:', error);
    }
  };

  // Bulk select
  const handleBulkSelect = async (selected: boolean) => {
    if (selectedIds.size === 0) return;
    setActionLoading(true);
    try {
      await fetch('/api/admin/training-items', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'bulk_select',
          itemIds: Array.from(selectedIds),
          selected,
        }),
      });
      setSelectedIds(new Set());
      fetchItems();
      fetchStats();
    } catch (error) {
      console.error('Error bulk selecting:', error);
    } finally {
      setActionLoading(false);
    }
  };

  // Review item
  const handleReview = async (itemId: string, status: 'approved' | 'rejected', reason?: string) => {
    setActionLoading(true);
    try {
      await fetch('/api/admin/training-items', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'review',
          itemId,
          status,
          reason,
        }),
      });
      setSelectedItem(null);
      fetchItems();
      fetchStats();
    } catch (error) {
      console.error('Error reviewing item:', error);
    } finally {
      setActionLoading(false);
    }
  };

  // Delete item
  const handleDelete = async (itemId: string, reason?: string) => {
    if (!confirm('Are you sure you want to delete this item? This action is logged and auditable.')) return;
    setActionLoading(true);
    try {
      await fetch('/api/admin/training-items', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'delete',
          itemId,
          reason: reason || 'Deleted by admin',
        }),
      });
      setSelectedItem(null);
      fetchItems();
      fetchStats();
    } catch (error) {
      console.error('Error deleting item:', error);
    } finally {
      setActionLoading(false);
    }
  };

  // Export selected items
  const handleExport = async () => {
    if (!exportName) {
      alert('Please provide an export name');
      return;
    }
    setActionLoading(true);
    try {
      const response = await fetch('/api/admin/training-items', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'export',
          exportName,
          targetSystem: exportTarget,
          exportFormat,
          notes: exportNotes,
        }),
      });
      const result = await response.json();

      if (result.success) {
        // Download the export
        const blob = new Blob([result.data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${exportName}.${exportFormat === 'jsonl' ? 'jsonl' : exportFormat === 'csv' ? 'csv' : 'json'}`;
        a.click();
        URL.revokeObjectURL(url);

        alert(`Exported ${result.itemCount} items successfully!`);
        setShowExportModal(false);
        setExportName('');
        setExportNotes('');
        fetchItems();
        fetchStats();
      } else {
        alert(result.error || 'Export failed');
      }
    } catch (error) {
      console.error('Error exporting:', error);
      alert('Export failed');
    } finally {
      setActionLoading(false);
    }
  };

  // Create new item
  const handleCreate = async () => {
    if (!createForm.inputText) {
      alert('Input text is required');
      return;
    }
    setActionLoading(true);
    try {
      const response = await fetch('/api/admin/training-items', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'create',
          targetSystem: createForm.targetSystem,
          domain: createForm.domain || null,
          inputText: createForm.inputText,
          expectedOutput: createForm.expectedOutput || null,
          trainingType: createForm.trainingType,
          qualityScore: createForm.qualityScore,
          difficultyLevel: createForm.difficultyLevel,
          tags: createForm.tags ? createForm.tags.split(',').map(t => t.trim()) : null,
        }),
      });
      const result = await response.json();

      if (result.success) {
        setShowCreateModal(false);
        setCreateForm({
          targetSystem: 'both',
          domain: '',
          inputText: '',
          expectedOutput: '',
          trainingType: 'positive',
          qualityScore: 0.8,
          difficultyLevel: 'medium',
          tags: '',
        });
        fetchItems();
        fetchStats();
      } else {
        alert(result.error || 'Failed to create item');
      }
    } catch (error) {
      console.error('Error creating item:', error);
      alert('Failed to create item');
    } finally {
      setActionLoading(false);
    }
  };

  // Toggle row selection for bulk actions
  const toggleRowSelection = (itemId: string) => {
    const newSelection = new Set(selectedIds);
    if (newSelection.has(itemId)) {
      newSelection.delete(itemId);
    } else {
      newSelection.add(itemId);
    }
    setSelectedIds(newSelection);
  };

  // Select all visible
  const selectAllVisible = () => {
    const allIds = new Set(items.map(i => i.id));
    setSelectedIds(allIds);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-bold text-white">Training Data Manager</h1>
          <p className="text-slate-400">Granular selection of training items for Elle and Guardian</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => { fetchAuditLog(); setShowAuditModal(true); }}
            className="px-4 py-2 bg-white/[0.06] text-slate-300 rounded-lg hover:bg-white/[0.1] transition-colors flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Audit Log
          </button>
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-500 transition-colors flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            Add Item
          </button>
          <button
            onClick={() => setShowExportModal(true)}
            disabled={!stats?.selected}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Export Selected ({stats?.selected || 0})
          </button>
        </div>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <GlassCard blur="heavy" className="text-center">
            <p className="text-2xl font-bold text-white">{stats.total}</p>
            <p className="text-xs text-slate-400">Total Items</p>
          </GlassCard>
          <GlassCard blur="heavy" className="text-center">
            <p className="text-2xl font-bold text-purple-400">{stats.elle}</p>
            <p className="text-xs text-slate-400">Elle Items</p>
          </GlassCard>
          <GlassCard blur="heavy" className="text-center">
            <p className="text-2xl font-bold text-blue-400">{stats.guardian}</p>
            <p className="text-xs text-slate-400">Guardian Items</p>
          </GlassCard>
          <GlassCard blur="heavy" className="text-center">
            <p className="text-2xl font-bold text-yellow-400">{stats.selected}</p>
            <p className="text-xs text-slate-400">Selected</p>
          </GlassCard>
          <GlassCard blur="heavy" className="text-center">
            <p className="text-2xl font-bold text-green-400">{stats.approved}</p>
            <p className="text-xs text-slate-400">Approved</p>
          </GlassCard>
          <GlassCard blur="heavy" className="text-center">
            <p className="text-2xl font-bold text-slate-400">{stats.exported}</p>
            <p className="text-xs text-slate-400">Exported</p>
          </GlassCard>
        </div>
      )}

      {/* Filters */}
      <GlassCard blur="heavy" className="flex flex-wrap gap-4 items-center">
        <select
          value={targetFilter}
          onChange={(e) => setTargetFilter(e.target.value)}
          className="bg-black/40 border border-white/[0.1] rounded-lg px-3 py-2 text-white text-sm"
        >
          <option value="">All Targets</option>
          <option value="elle">Elle Only</option>
          <option value="guardian">Guardian Only</option>
          <option value="both">Both</option>
        </select>

        <select
          value={domainFilter}
          onChange={(e) => setDomainFilter(e.target.value)}
          className="bg-black/40 border border-white/[0.1] rounded-lg px-3 py-2 text-white text-sm"
        >
          <option value="">All Domains</option>
          <option value="political">Political</option>
          <option value="economic">Economic</option>
          <option value="security">Security</option>
          <option value="military">Military</option>
          <option value="cyber">Cyber</option>
          <option value="financial">Financial</option>
        </select>

        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="bg-black/40 border border-white/[0.1] rounded-lg px-3 py-2 text-white text-sm"
        >
          <option value="">All Status</option>
          <option value="pending">Pending</option>
          <option value="approved">Approved</option>
          <option value="rejected">Rejected</option>
          <option value="exported">Exported</option>
        </select>

        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={showSelectedOnly}
            onChange={(e) => setShowSelectedOnly(e.target.checked)}
            className="rounded bg-black/40 border-white/[0.1]"
          />
          Selected only
        </label>

        <div className="flex-1" />

        {selectedIds.size > 0 && (
          <div className="flex gap-2">
            <button
              onClick={() => handleBulkSelect(true)}
              disabled={actionLoading}
              className="px-3 py-1.5 bg-yellow-500/20 text-yellow-400 rounded-lg text-sm hover:bg-yellow-500/30"
            >
              Mark Selected ({selectedIds.size})
            </button>
            <button
              onClick={() => handleBulkSelect(false)}
              disabled={actionLoading}
              className="px-3 py-1.5 bg-slate-500/20 text-slate-400 rounded-lg text-sm hover:bg-slate-500/30"
            >
              Unmark Selected
            </button>
            <button
              onClick={() => setSelectedIds(new Set())}
              className="px-3 py-1.5 bg-white/[0.06] text-slate-400 rounded-lg text-sm hover:bg-white/[0.1]"
            >
              Clear
            </button>
          </div>
        )}
      </GlassCard>

      {/* Items Table */}
      <GlassCard blur="heavy" className="p-0 overflow-hidden">
        <div className="p-4 border-b border-white/[0.06] flex justify-between items-center">
          <h2 className="font-medium text-white">Training Items</h2>
          <button
            onClick={selectAllVisible}
            className="text-sm text-slate-400 hover:text-white"
          >
            Select all visible
          </button>
        </div>

        {loading ? (
          <div className="p-8 text-center text-slate-500">Loading...</div>
        ) : items.length === 0 ? (
          <div className="p-8 text-center text-slate-500">No training items found</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-xs text-slate-500 uppercase tracking-wide border-b border-white/[0.06]">
                  <th className="py-3 px-4 w-10">
                    <input
                      type="checkbox"
                      checked={selectedIds.size === items.length && items.length > 0}
                      onChange={() => selectedIds.size === items.length ? setSelectedIds(new Set()) : selectAllVisible()}
                      className="rounded bg-black/40 border-white/[0.1]"
                    />
                  </th>
                  <th className="py-3 px-4 w-10">Export</th>
                  <th className="py-3 px-4">Target</th>
                  <th className="py-3 px-4">Domain</th>
                  <th className="py-3 px-4">Type</th>
                  <th className="py-3 px-4">Input (preview)</th>
                  <th className="py-3 px-4">Quality</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4">Actions</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <tr
                    key={item.id}
                    className={`border-b border-white/[0.06] hover:bg-white/[0.02] ${selectedIds.has(item.id) ? 'bg-blue-500/10' : ''}`}
                  >
                    <td className="py-3 px-4">
                      <input
                        type="checkbox"
                        checked={selectedIds.has(item.id)}
                        onChange={() => toggleRowSelection(item.id)}
                        className="rounded bg-black/40 border-white/[0.1]"
                      />
                    </td>
                    <td className="py-3 px-4">
                      <input
                        type="checkbox"
                        checked={item.selected_for_export}
                        onChange={() => toggleSelection(item.id, item.selected_for_export)}
                        className="rounded bg-black/40 border-yellow-500/50 text-yellow-500"
                      />
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-0.5 text-xs rounded ${
                        item.target_system === 'elle' ? 'bg-purple-500/20 text-purple-400' :
                        item.target_system === 'guardian' ? 'bg-blue-500/20 text-blue-400' :
                        'bg-green-500/20 text-green-400'
                      }`}>
                        {item.target_system}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-slate-400 text-sm">{item.domain || '-'}</td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-0.5 text-xs rounded ${
                        item.training_type === 'positive' ? 'bg-green-500/20 text-green-400' :
                        item.training_type === 'negative' ? 'bg-red-500/20 text-red-400' :
                        item.training_type === 'correction' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-blue-500/20 text-blue-400'
                      }`}>
                        {item.training_type}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-slate-300 text-sm max-w-[300px]">
                      <span
                        className="cursor-pointer hover:text-white truncate block"
                        onClick={() => setSelectedItem(item)}
                      >
                        {item.input_text.slice(0, 100)}{item.input_text.length > 100 ? '...' : ''}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      {item.quality_score !== null && (
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-2 bg-white/[0.1] rounded-full overflow-hidden">
                            <div
                              className={`h-full ${
                                item.quality_score >= 0.8 ? 'bg-green-500' :
                                item.quality_score >= 0.6 ? 'bg-yellow-500' :
                                'bg-red-500'
                              }`}
                              style={{ width: `${item.quality_score * 100}%` }}
                            />
                          </div>
                          <span className="text-xs text-slate-400">{(item.quality_score * 100).toFixed(0)}%</span>
                        </div>
                      )}
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-0.5 text-xs rounded ${
                        item.status === 'approved' ? 'bg-green-500/20 text-green-400' :
                        item.status === 'rejected' ? 'bg-red-500/20 text-red-400' :
                        item.status === 'exported' ? 'bg-slate-500/20 text-slate-400' :
                        'bg-yellow-500/20 text-yellow-400'
                      }`}>
                        {item.status}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex gap-1">
                        <button
                          onClick={() => setSelectedItem(item)}
                          className="p-1.5 text-slate-400 hover:text-white hover:bg-white/[0.06] rounded"
                          title="View details"
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                          </svg>
                        </button>
                        {item.status === 'pending' && (
                          <>
                            <button
                              onClick={() => handleReview(item.id, 'approved')}
                              className="p-1.5 text-green-400 hover:bg-green-500/20 rounded"
                              title="Approve"
                            >
                              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                              </svg>
                            </button>
                            <button
                              onClick={() => handleReview(item.id, 'rejected')}
                              className="p-1.5 text-red-400 hover:bg-red-500/20 rounded"
                              title="Reject"
                            >
                              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                              </svg>
                            </button>
                          </>
                        )}
                        <button
                          onClick={() => handleDelete(item.id)}
                          className="p-1.5 text-slate-400 hover:text-red-400 hover:bg-red-500/20 rounded"
                          title="Delete"
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </GlassCard>

      {/* Item Detail Modal */}
      {selectedItem && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-[rgba(18,18,26,0.95)] border border-white/[0.1] rounded-2xl max-w-3xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-6 border-b border-white/[0.06] flex justify-between items-start">
              <div>
                <h2 className="text-xl font-bold text-white">Training Item Details</h2>
                <p className="text-sm text-slate-400 mt-1">ID: {selectedItem.id.slice(0, 8)}...</p>
              </div>
              <button
                onClick={() => setSelectedItem(null)}
                className="p-2 text-slate-400 hover:text-white hover:bg-white/[0.06] rounded-lg"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="p-6 space-y-6">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-xs text-slate-500 uppercase">Target</p>
                  <p className="text-white">{selectedItem.target_system}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500 uppercase">Domain</p>
                  <p className="text-white">{selectedItem.domain || '-'}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500 uppercase">Type</p>
                  <p className="text-white">{selectedItem.training_type}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500 uppercase">Status</p>
                  <p className="text-white">{selectedItem.status}</p>
                </div>
              </div>

              <div>
                <p className="text-xs text-slate-500 uppercase mb-2">Input Text</p>
                <p className="text-slate-300 bg-black/40 p-4 rounded-lg whitespace-pre-wrap">{selectedItem.input_text}</p>
              </div>

              {selectedItem.expected_output && (
                <div>
                  <p className="text-xs text-slate-500 uppercase mb-2">Expected Output</p>
                  <p className="text-slate-300 bg-green-500/10 border border-green-500/20 p-4 rounded-lg whitespace-pre-wrap">{selectedItem.expected_output}</p>
                </div>
              )}

              {selectedItem.actual_output && (
                <div>
                  <p className="text-xs text-slate-500 uppercase mb-2">Actual Output (for correction)</p>
                  <p className="text-slate-300 bg-red-500/10 border border-red-500/20 p-4 rounded-lg whitespace-pre-wrap">{selectedItem.actual_output}</p>
                </div>
              )}

              {selectedItem.tags && selectedItem.tags.length > 0 && (
                <div>
                  <p className="text-xs text-slate-500 uppercase mb-2">Tags</p>
                  <div className="flex flex-wrap gap-2">
                    {selectedItem.tags.map((tag, i) => (
                      <span key={i} className="px-2 py-1 bg-white/[0.06] text-slate-300 rounded text-sm">{tag}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="p-6 border-t border-white/[0.06] flex justify-between">
              <button
                onClick={() => handleDelete(selectedItem.id)}
                disabled={actionLoading}
                className="px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 disabled:opacity-50"
              >
                Delete
              </button>
              <div className="flex gap-3">
                {selectedItem.status === 'pending' && (
                  <>
                    <button
                      onClick={() => handleReview(selectedItem.id, 'rejected')}
                      disabled={actionLoading}
                      className="px-4 py-2 bg-slate-500/20 text-slate-400 rounded-lg hover:bg-slate-500/30 disabled:opacity-50"
                    >
                      Reject
                    </button>
                    <button
                      onClick={() => handleReview(selectedItem.id, 'approved')}
                      disabled={actionLoading}
                      className="px-4 py-2 bg-green-500/20 text-green-400 rounded-lg hover:bg-green-500/30 disabled:opacity-50"
                    >
                      Approve
                    </button>
                  </>
                )}
                <button
                  onClick={() => toggleSelection(selectedItem.id, selectedItem.selected_for_export)}
                  className={`px-4 py-2 rounded-lg ${
                    selectedItem.selected_for_export
                      ? 'bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30'
                      : 'bg-white/[0.06] text-slate-300 hover:bg-white/[0.1]'
                  }`}
                >
                  {selectedItem.selected_for_export ? 'Deselect for Export' : 'Select for Export'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Export Modal */}
      {showExportModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-[rgba(18,18,26,0.95)] border border-white/[0.1] rounded-2xl max-w-lg w-full">
            <div className="p-6 border-b border-white/[0.06]">
              <h2 className="text-xl font-bold text-white">Export Training Data</h2>
              <p className="text-sm text-slate-400 mt-1">{stats?.selected || 0} items selected for export</p>
            </div>

            <div className="p-6 space-y-4">
              <div>
                <label className="text-sm text-slate-400 block mb-2">Export Name</label>
                <input
                  type="text"
                  value={exportName}
                  onChange={(e) => setExportName(e.target.value)}
                  placeholder="e.g., elle-training-2024-12-10"
                  className="w-full bg-black/40 border border-white/[0.1] rounded-lg px-4 py-2 text-white placeholder-slate-500"
                />
              </div>

              <div>
                <label className="text-sm text-slate-400 block mb-2">Target System</label>
                <select
                  value={exportTarget}
                  onChange={(e) => setExportTarget(e.target.value)}
                  className="w-full bg-black/40 border border-white/[0.1] rounded-lg px-4 py-2 text-white"
                >
                  <option value="both">Both (Elle + Guardian)</option>
                  <option value="elle">Elle Only</option>
                  <option value="guardian">Guardian Only</option>
                </select>
              </div>

              <div>
                <label className="text-sm text-slate-400 block mb-2">Format</label>
                <select
                  value={exportFormat}
                  onChange={(e) => setExportFormat(e.target.value)}
                  className="w-full bg-black/40 border border-white/[0.1] rounded-lg px-4 py-2 text-white"
                >
                  <option value="jsonl">JSONL (Axolotl/OpenAI)</option>
                  <option value="alpaca">Alpaca Format</option>
                  <option value="sharegpt">ShareGPT Format</option>
                  <option value="csv">CSV</option>
                  <option value="json">JSON</option>
                </select>
              </div>

              <div>
                <label className="text-sm text-slate-400 block mb-2">Notes (logged for audit)</label>
                <textarea
                  value={exportNotes}
                  onChange={(e) => setExportNotes(e.target.value)}
                  placeholder="Purpose of this export, training run ID, etc."
                  className="w-full bg-black/40 border border-white/[0.1] rounded-lg px-4 py-2 text-white placeholder-slate-500"
                  rows={3}
                />
              </div>
            </div>

            <div className="p-6 border-t border-white/[0.06] flex justify-end gap-3">
              <button
                onClick={() => setShowExportModal(false)}
                className="px-4 py-2 bg-white/[0.06] text-slate-300 rounded-lg hover:bg-white/[0.1]"
              >
                Cancel
              </button>
              <button
                onClick={handleExport}
                disabled={actionLoading || !exportName}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500 disabled:opacity-50"
              >
                {actionLoading ? 'Exporting...' : 'Export'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Create Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-[rgba(18,18,26,0.95)] border border-white/[0.1] rounded-2xl max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-6 border-b border-white/[0.06]">
              <h2 className="text-xl font-bold text-white">Add Training Item</h2>
              <p className="text-sm text-slate-400 mt-1">Manually create a training example</p>
            </div>

            <div className="p-6 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm text-slate-400 block mb-2">Target System</label>
                  <select
                    value={createForm.targetSystem}
                    onChange={(e) => setCreateForm({ ...createForm, targetSystem: e.target.value as 'elle' | 'guardian' | 'both' })}
                    className="w-full bg-black/40 border border-white/[0.1] rounded-lg px-4 py-2 text-white"
                  >
                    <option value="both">Both</option>
                    <option value="elle">Elle</option>
                    <option value="guardian">Guardian</option>
                  </select>
                </div>
                <div>
                  <label className="text-sm text-slate-400 block mb-2">Domain</label>
                  <select
                    value={createForm.domain}
                    onChange={(e) => setCreateForm({ ...createForm, domain: e.target.value })}
                    className="w-full bg-black/40 border border-white/[0.1] rounded-lg px-4 py-2 text-white"
                  >
                    <option value="">General</option>
                    <option value="political">Political</option>
                    <option value="economic">Economic</option>
                    <option value="security">Security</option>
                    <option value="military">Military</option>
                    <option value="cyber">Cyber</option>
                    <option value="financial">Financial</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="text-sm text-slate-400 block mb-2">Training Type</label>
                  <select
                    value={createForm.trainingType}
                    onChange={(e) => setCreateForm({ ...createForm, trainingType: e.target.value })}
                    className="w-full bg-black/40 border border-white/[0.1] rounded-lg px-4 py-2 text-white"
                  >
                    <option value="positive">Positive Example</option>
                    <option value="negative">Negative Example</option>
                    <option value="correction">Correction</option>
                    <option value="reinforcement">Reinforcement</option>
                  </select>
                </div>
                <div>
                  <label className="text-sm text-slate-400 block mb-2">Quality Score</label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.1"
                    value={createForm.qualityScore}
                    onChange={(e) => setCreateForm({ ...createForm, qualityScore: parseFloat(e.target.value) })}
                    className="w-full bg-black/40 border border-white/[0.1] rounded-lg px-4 py-2 text-white"
                  />
                </div>
                <div>
                  <label className="text-sm text-slate-400 block mb-2">Difficulty</label>
                  <select
                    value={createForm.difficultyLevel}
                    onChange={(e) => setCreateForm({ ...createForm, difficultyLevel: e.target.value })}
                    className="w-full bg-black/40 border border-white/[0.1] rounded-lg px-4 py-2 text-white"
                  >
                    <option value="easy">Easy</option>
                    <option value="medium">Medium</option>
                    <option value="hard">Hard</option>
                    <option value="edge_case">Edge Case</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="text-sm text-slate-400 block mb-2">Input Text / Prompt</label>
                <textarea
                  value={createForm.inputText}
                  onChange={(e) => setCreateForm({ ...createForm, inputText: e.target.value })}
                  placeholder="The input or prompt for this training example..."
                  className="w-full bg-black/40 border border-white/[0.1] rounded-lg px-4 py-2 text-white placeholder-slate-500"
                  rows={4}
                />
              </div>

              <div>
                <label className="text-sm text-slate-400 block mb-2">Expected Output</label>
                <textarea
                  value={createForm.expectedOutput}
                  onChange={(e) => setCreateForm({ ...createForm, expectedOutput: e.target.value })}
                  placeholder="The correct/expected response..."
                  className="w-full bg-black/40 border border-white/[0.1] rounded-lg px-4 py-2 text-white placeholder-slate-500"
                  rows={4}
                />
              </div>

              <div>
                <label className="text-sm text-slate-400 block mb-2">Tags (comma-separated)</label>
                <input
                  type="text"
                  value={createForm.tags}
                  onChange={(e) => setCreateForm({ ...createForm, tags: e.target.value })}
                  placeholder="e.g., russia, sanctions, energy"
                  className="w-full bg-black/40 border border-white/[0.1] rounded-lg px-4 py-2 text-white placeholder-slate-500"
                />
              </div>
            </div>

            <div className="p-6 border-t border-white/[0.06] flex justify-end gap-3">
              <button
                onClick={() => setShowCreateModal(false)}
                className="px-4 py-2 bg-white/[0.06] text-slate-300 rounded-lg hover:bg-white/[0.1]"
              >
                Cancel
              </button>
              <button
                onClick={handleCreate}
                disabled={actionLoading || !createForm.inputText}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-500 disabled:opacity-50"
              >
                {actionLoading ? 'Creating...' : 'Create Item'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Audit Log Modal */}
      {showAuditModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-[rgba(18,18,26,0.95)] border border-white/[0.1] rounded-2xl max-w-4xl w-full max-h-[80vh] overflow-hidden flex flex-col">
            <div className="p-6 border-b border-white/[0.06] flex justify-between items-start">
              <div>
                <h2 className="text-xl font-bold text-white">Immutable Audit Log</h2>
                <p className="text-sm text-slate-400 mt-1">All training decisions are logged with hash-chain integrity</p>
              </div>
              <button
                onClick={() => setShowAuditModal(false)}
                className="p-2 text-slate-400 hover:text-white hover:bg-white/[0.06] rounded-lg"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Integrity Status */}
            {integrity && (
              <div className={`p-4 border-b border-white/[0.06] ${integrity.is_valid ? 'bg-green-500/10' : 'bg-red-500/10'}`}>
                <div className="flex items-center gap-3">
                  <span className={`w-3 h-3 rounded-full ${integrity.is_valid ? 'bg-green-500' : 'bg-red-500 animate-pulse'}`} />
                  <span className={integrity.is_valid ? 'text-green-400' : 'text-red-400'}>
                    {integrity.is_valid ? 'Audit log integrity verified' : 'INTEGRITY VIOLATION DETECTED'}
                  </span>
                  <span className="text-slate-500 text-sm ml-auto">
                    {integrity.verified_entries} / {integrity.total_entries} entries verified
                  </span>
                </div>
              </div>
            )}

            <div className="flex-1 overflow-y-auto">
              {auditLog.length === 0 ? (
                <div className="p-8 text-center text-slate-500">No audit entries yet</div>
              ) : (
                <div className="divide-y divide-white/[0.06]">
                  {auditLog.map((entry) => (
                    <div key={entry.id} className="p-4 hover:bg-white/[0.02]">
                      <div className="flex justify-between items-start mb-2">
                        <div className="flex items-center gap-2">
                          <span className={`px-2 py-0.5 text-xs rounded ${
                            entry.action.includes('export') ? 'bg-blue-500/20 text-blue-400' :
                            entry.action.includes('select') ? 'bg-yellow-500/20 text-yellow-400' :
                            entry.action.includes('approv') ? 'bg-green-500/20 text-green-400' :
                            entry.action.includes('reject') || entry.action.includes('delet') ? 'bg-red-500/20 text-red-400' :
                            'bg-slate-500/20 text-slate-400'
                          }`}>
                            {entry.action.replace(/_/g, ' ')}
                          </span>
                          <span className="text-slate-400 text-sm">
                            {entry.performed_by_name || entry.performed_by_email}
                          </span>
                        </div>
                        <span className="text-xs text-slate-500">
                          {new Date(entry.performed_at).toLocaleString()}
                        </span>
                      </div>
                      {entry.reason && (
                        <p className="text-sm text-slate-400 mb-2">{entry.reason}</p>
                      )}
                      <div className="flex gap-4 text-xs text-slate-600">
                        <span>Hash: {entry.entry_hash?.slice(0, 16)}...</span>
                        <span>Previous: {entry.previous_hash?.slice(0, 16)}...</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
