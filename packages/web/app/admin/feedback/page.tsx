'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  MessageSquare, Bug, Lightbulb, HelpCircle, FileText,
  RefreshCw, Filter, ChevronDown, Clock, User, ExternalLink,
  CheckCircle, XCircle, AlertTriangle, Loader2, Search,
  ArrowUp, ArrowDown
} from 'lucide-react';

// Types
interface Feedback {
  id: string;
  type: 'bug' | 'idea' | 'question' | 'other';
  title: string;
  description: string;
  status: 'unread' | 'acknowledged' | 'in_progress' | 'resolved' | 'wont_fix' | 'duplicate';
  priority: 'low' | 'normal' | 'high' | 'critical';
  page_url?: string;
  user_agent?: string;
  admin_notes?: string;
  resolution_notes?: string;
  created_at: string;
  updated_at: string;
  user?: {
    email: string;
    full_name?: string;
    role?: string;
    tier?: string;
  };
  assignee?: {
    email: string;
    full_name?: string;
  };
}

interface FeedbackStats {
  unread_count: number;
  acknowledged_count: number;
  in_progress_count: number;
  resolved_count: number;
  bug_count: number;
  idea_count: number;
  critical_count: number;
  high_priority_count: number;
  total_count: number;
}

// Status config
const STATUS_CONFIG = {
  unread: { label: 'Unread', color: 'bg-slate-500', textColor: 'text-slate-300' },
  acknowledged: { label: 'Acknowledged', color: 'bg-blue-500', textColor: 'text-blue-300' },
  in_progress: { label: 'In Progress', color: 'bg-amber-500', textColor: 'text-amber-300' },
  resolved: { label: 'Resolved', color: 'bg-emerald-500', textColor: 'text-emerald-300' },
  wont_fix: { label: "Won't Fix", color: 'bg-red-500', textColor: 'text-red-300' },
  duplicate: { label: 'Duplicate', color: 'bg-purple-500', textColor: 'text-purple-300' },
};

const PRIORITY_CONFIG = {
  critical: { label: 'Critical', color: 'text-red-400', bgColor: 'bg-red-500/20' },
  high: { label: 'High', color: 'text-amber-400', bgColor: 'bg-amber-500/20' },
  normal: { label: 'Normal', color: 'text-blue-400', bgColor: 'bg-blue-500/20' },
  low: { label: 'Low', color: 'text-slate-400', bgColor: 'bg-slate-500/20' },
};

const TYPE_CONFIG = {
  bug: { label: 'Bug', icon: Bug, color: 'text-red-400' },
  idea: { label: 'Idea', icon: Lightbulb, color: 'text-amber-400' },
  question: { label: 'Question', icon: HelpCircle, color: 'text-blue-400' },
  other: { label: 'Other', icon: FileText, color: 'text-slate-400' },
};

export default function FeedbackDashboard() {
  const [feedback, setFeedback] = useState<Feedback[]>([]);
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [typeFilter, setTypeFilter] = useState<string>('');
  const [priorityFilter, setPriorityFilter] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('created_at');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Selected feedback for detail view
  const [selectedFeedback, setSelectedFeedback] = useState<Feedback | null>(null);
  const [updating, setUpdating] = useState(false);

  // Fetch feedback
  const fetchFeedback = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      if (statusFilter) params.set('status', statusFilter);
      if (typeFilter) params.set('type', typeFilter);
      if (priorityFilter) params.set('priority', priorityFilter);
      params.set('sortBy', sortBy);
      params.set('sortOrder', sortOrder);

      const response = await fetch(`/api/feedback?${params.toString()}`, {
        credentials: 'include',
      });

      if (!response.ok) {
        throw new Error('Failed to fetch feedback');
      }

      const data = await response.json();
      setFeedback(data.feedback || []);
      setStats(data.stats);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setLoading(false);
    }
  }, [statusFilter, typeFilter, priorityFilter, sortBy, sortOrder]);

  useEffect(() => {
    void fetchFeedback();
  }, [fetchFeedback]);

  // Update feedback
  const updateFeedback = async (id: string, updates: Partial<Feedback>) => {
    setUpdating(true);
    try {
      const response = await fetch(`/api/feedback/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(updates),
      });

      if (!response.ok) {
        throw new Error('Failed to update feedback');
      }

      // Refresh list
      await fetchFeedback();

      // Update selected if same
      if (selectedFeedback?.id === id) {
        const data = await response.json();
        setSelectedFeedback(data.feedback);
      }
    } catch (err) {
      console.error('Update error:', err);
    } finally {
      setUpdating(false);
    }
  };

  // Filter feedback by search
  const filteredFeedback = feedback.filter((f) => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      f.title.toLowerCase().includes(query) ||
      f.description.toLowerCase().includes(query) ||
      f.user?.email?.toLowerCase().includes(query)
    );
  });

  // Format date
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(hours / 24);

    if (hours < 1) return 'Just now';
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-white flex items-center gap-2 sm:gap-3">
            <MessageSquare className="w-6 h-6 sm:w-7 sm:h-7 text-cyan-400" />
            Feedback Dashboard
          </h1>
          <p className="text-slate-400 text-xs sm:text-sm mt-1">
            Manage bug reports, feature ideas, and user feedback
          </p>
        </div>
        <button
          onClick={() => void fetchFeedback()}
          disabled={loading}
          className="flex items-center justify-center gap-2 px-4 py-2.5 bg-slate-800 hover:bg-slate-700 active:bg-slate-600 text-white rounded-lg transition-colors touch-manipulation min-h-[44px]"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          <div className="bg-slate-800/50 rounded-xl p-4 border border-white/5">
            <p className="text-sm text-slate-400">Unread</p>
            <p className="text-2xl font-bold text-white">{stats.unread_count}</p>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-4 border border-white/5">
            <p className="text-sm text-slate-400">In Progress</p>
            <p className="text-2xl font-bold text-amber-400">{stats.in_progress_count}</p>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-4 border border-white/5">
            <p className="text-sm text-slate-400">Resolved</p>
            <p className="text-2xl font-bold text-emerald-400">{stats.resolved_count}</p>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-4 border border-white/5">
            <p className="text-sm text-slate-400">Bugs</p>
            <p className="text-2xl font-bold text-red-400">{stats.bug_count}</p>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-4 border border-white/5">
            <p className="text-sm text-slate-400">Ideas</p>
            <p className="text-2xl font-bold text-amber-400">{stats.idea_count}</p>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-4 border border-white/5">
            <p className="text-sm text-slate-400">Critical</p>
            <p className="text-2xl font-bold text-red-500">{stats.critical_count}</p>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3 bg-slate-800/30 rounded-xl p-4 border border-white/5">
        <Filter className="w-5 h-5 text-slate-400" />

        {/* Search */}
        <div className="relative flex-1 min-w-0 sm:min-w-[200px] w-full sm:w-auto order-first sm:order-none">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search feedback..."
            className="w-full pl-10 pr-4 py-2.5 bg-slate-800 border border-white/10 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
          />
        </div>

        {/* Status filter */}
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="px-3 py-2 bg-slate-800 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
        >
          <option value="">All Statuses</option>
          {Object.entries(STATUS_CONFIG).map(([key, config]) => (
            <option key={key} value={key}>{config.label}</option>
          ))}
        </select>

        {/* Type filter */}
        <select
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
          className="px-3 py-2 bg-slate-800 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
        >
          <option value="">All Types</option>
          {Object.entries(TYPE_CONFIG).map(([key, config]) => (
            <option key={key} value={key}>{config.label}</option>
          ))}
        </select>

        {/* Priority filter */}
        <select
          value={priorityFilter}
          onChange={(e) => setPriorityFilter(e.target.value)}
          className="px-3 py-2 bg-slate-800 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
        >
          <option value="">All Priorities</option>
          {Object.entries(PRIORITY_CONFIG).map(([key, config]) => (
            <option key={key} value={key}>{config.label}</option>
          ))}
        </select>

        {/* Sort */}
        <button
          onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
          className="flex items-center gap-1 px-3 py-2 bg-slate-800 border border-white/10 rounded-lg text-white hover:bg-slate-700 transition-colors"
        >
          {sortOrder === 'desc' ? <ArrowDown className="w-4 h-4" /> : <ArrowUp className="w-4 h-4" />}
          {sortBy === 'created_at' ? 'Date' : 'Priority'}
        </button>
      </div>

      {/* Error state */}
      {error && (
        <div className="p-4 bg-red-500/20 border border-red-500/30 rounded-lg text-red-300">
          {error}
        </div>
      )}

      {/* Loading state */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
        </div>
      )}

      {/* Feedback List */}
      {!loading && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* List */}
          <div className="space-y-2">
            {filteredFeedback.length === 0 ? (
              <div className="text-center py-12 text-slate-500">
                No feedback found
              </div>
            ) : (
              filteredFeedback.map((item) => {
                const TypeIcon = TYPE_CONFIG[item.type].icon;
                const statusConfig = STATUS_CONFIG[item.status];
                const priorityConfig = PRIORITY_CONFIG[item.priority];

                return (
                  <button
                    key={item.id}
                    onClick={() => setSelectedFeedback(item)}
                    className={`w-full text-left p-4 rounded-xl border transition-all ${
                      selectedFeedback?.id === item.id
                        ? 'bg-cyan-500/10 border-cyan-500/30'
                        : 'bg-slate-800/50 border-white/5 hover:border-white/10'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <TypeIcon className={`w-5 h-5 mt-0.5 ${TYPE_CONFIG[item.type].color}`} />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <h3 className="font-medium text-white truncate">{item.title}</h3>
                          <span className={`text-xs px-2 py-0.5 rounded-full ${priorityConfig.bgColor} ${priorityConfig.color}`}>
                            {priorityConfig.label}
                          </span>
                        </div>
                        <p className="text-sm text-slate-400 line-clamp-2 mt-1">
                          {item.description}
                        </p>
                        <div className="flex items-center gap-3 mt-2 text-xs text-slate-500">
                          <span className={`px-2 py-0.5 rounded ${statusConfig.color}/20 ${statusConfig.textColor}`}>
                            {statusConfig.label}
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {formatDate(item.created_at)}
                          </span>
                          {item.user?.email && (
                            <span className="flex items-center gap-1">
                              <User className="w-3 h-3" />
                              {item.user.email}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </button>
                );
              })
            )}
          </div>

          {/* Detail Panel */}
          <div className="lg:sticky lg:top-4 lg:self-start">
            {selectedFeedback ? (
              <div className="bg-slate-800/50 rounded-xl border border-white/5 p-6 space-y-6">
                {/* Header */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    {(() => {
                      const TypeIcon = TYPE_CONFIG[selectedFeedback.type].icon;
                      return <TypeIcon className={`w-5 h-5 ${TYPE_CONFIG[selectedFeedback.type].color}`} />;
                    })()}
                    <span className="text-sm text-slate-400">{TYPE_CONFIG[selectedFeedback.type].label}</span>
                    <span className="text-slate-600">|</span>
                    <span className="text-xs text-slate-500">#{selectedFeedback.id.slice(0, 8)}</span>
                  </div>
                  <h2 className="text-xl font-semibold text-white">{selectedFeedback.title}</h2>
                </div>

                {/* Status & Priority Controls */}
                <div className="flex flex-wrap gap-3">
                  <select
                    value={selectedFeedback.status}
                    onChange={(e) => void updateFeedback(selectedFeedback.id, { status: e.target.value as Feedback['status'] })}
                    disabled={updating}
                    className="px-3 py-2 bg-slate-700 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-cyan-500/50 disabled:opacity-50"
                  >
                    {Object.entries(STATUS_CONFIG).map(([key, config]) => (
                      <option key={key} value={key}>{config.label}</option>
                    ))}
                  </select>

                  <select
                    value={selectedFeedback.priority}
                    onChange={(e) => void updateFeedback(selectedFeedback.id, { priority: e.target.value as Feedback['priority'] })}
                    disabled={updating}
                    className="px-3 py-2 bg-slate-700 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-cyan-500/50 disabled:opacity-50"
                  >
                    {Object.entries(PRIORITY_CONFIG).map(([key, config]) => (
                      <option key={key} value={key}>{config.label}</option>
                    ))}
                  </select>
                </div>

                {/* Description */}
                <div>
                  <h3 className="text-sm font-medium text-slate-300 mb-2">Description</h3>
                  <p className="text-slate-400 whitespace-pre-wrap">{selectedFeedback.description}</p>
                </div>

                {/* Metadata */}
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-slate-500">Submitted</p>
                    <p className="text-white">{new Date(selectedFeedback.created_at).toLocaleString()}</p>
                  </div>
                  {selectedFeedback.user?.email && (
                    <div>
                      <p className="text-slate-500">User</p>
                      <p className="text-white">{selectedFeedback.user.full_name || selectedFeedback.user.email}</p>
                    </div>
                  )}
                  {selectedFeedback.page_url && (
                    <div className="col-span-2">
                      <p className="text-slate-500">Page</p>
                      <a
                        href={selectedFeedback.page_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-cyan-400 hover:text-cyan-300 flex items-center gap-1"
                      >
                        {selectedFeedback.page_url}
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    </div>
                  )}
                </div>

                {/* Quick Actions */}
                <div className="flex flex-wrap gap-2 pt-4 border-t border-white/10">
                  <button
                    onClick={() => void updateFeedback(selectedFeedback.id, { status: 'acknowledged' })}
                    disabled={updating || selectedFeedback.status !== 'unread'}
                    className="px-3 py-1.5 bg-blue-500/20 text-blue-400 rounded-lg text-sm hover:bg-blue-500/30 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <CheckCircle className="w-4 h-4 inline mr-1" />
                    Acknowledge
                  </button>
                  <button
                    onClick={() => void updateFeedback(selectedFeedback.id, { status: 'in_progress' })}
                    disabled={updating}
                    className="px-3 py-1.5 bg-amber-500/20 text-amber-400 rounded-lg text-sm hover:bg-amber-500/30 disabled:opacity-50 transition-colors"
                  >
                    <AlertTriangle className="w-4 h-4 inline mr-1" />
                    In Progress
                  </button>
                  <button
                    onClick={() => void updateFeedback(selectedFeedback.id, { status: 'resolved' })}
                    disabled={updating}
                    className="px-3 py-1.5 bg-emerald-500/20 text-emerald-400 rounded-lg text-sm hover:bg-emerald-500/30 disabled:opacity-50 transition-colors"
                  >
                    <CheckCircle className="w-4 h-4 inline mr-1" />
                    Resolve
                  </button>
                  <button
                    onClick={() => void updateFeedback(selectedFeedback.id, { status: 'wont_fix' })}
                    disabled={updating}
                    className="px-3 py-1.5 bg-red-500/20 text-red-400 rounded-lg text-sm hover:bg-red-500/30 disabled:opacity-50 transition-colors"
                  >
                    <XCircle className="w-4 h-4 inline mr-1" />
                    Won&apos;t Fix
                  </button>
                </div>
              </div>
            ) : (
              <div className="bg-slate-800/30 rounded-xl border border-white/5 p-12 text-center">
                <MessageSquare className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                <p className="text-slate-500">Select feedback to view details</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
