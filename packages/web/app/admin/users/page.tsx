'use client';

import { useState, useEffect } from 'react';
import { createBrowserClient } from '@supabase/ssr';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';

interface UserProfile {
  id: string;
  email: string;
  full_name: string | null;
  role: 'admin' | 'enterprise' | 'consumer' | 'support';
  tier: 'free' | 'starter' | 'pro' | 'enterprise_tier';
  is_active: boolean;
  is_banned: boolean;
  ban_reason: string | null;
  banned_at: string | null;
  last_seen_at: string | null;
  created_at: string;
  onboarding_completed_at: string | null;
}

const ROLES = ['admin', 'enterprise', 'consumer', 'support'] as const;
const TIERS = ['free', 'starter', 'pro', 'enterprise_tier'] as const;

// Inline dropdown for quick value changes
function InlineSelect<T extends string>({
  value,
  options,
  onChange,
  disabled,
  colorMap,
}: {
  value: T;
  options: readonly T[];
  onChange: (value: T) => void;
  disabled?: boolean;
  colorMap?: Record<string, string>;
}) {
  const getColor = (val: string) => {
    if (colorMap?.[val]) return colorMap[val];
    return 'bg-slate-500/20 text-slate-300';
  };

  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as T)}
      disabled={disabled}
      className={`px-2 py-1 rounded text-xs uppercase cursor-pointer border-0 focus:ring-2 focus:ring-blue-500 ${getColor(value)} ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      {options.map((opt) => (
        <option key={opt} value={opt}>
          {opt}
        </option>
      ))}
    </select>
  );
}

const ROLE_COLORS: Record<string, string> = {
  admin: 'bg-purple-500/20 text-purple-300',
  enterprise: 'bg-blue-500/20 text-blue-300',
  consumer: 'bg-slate-500/20 text-slate-300',
  support: 'bg-cyan-500/20 text-cyan-300',
};

const TIER_COLORS: Record<string, string> = {
  enterprise_tier: 'bg-amber-500/20 text-amber-300',
  pro: 'bg-green-500/20 text-green-300',
  starter: 'bg-blue-500/20 text-blue-300',
  free: 'bg-slate-500/20 text-slate-300',
};

function UserRow({
  user,
  onUpdate,
  onBan,
  onUnban,
}: {
  user: UserProfile;
  onUpdate: (id: string, updates: Partial<UserProfile>) => Promise<void>;
  onBan: (id: string, reason: string) => Promise<void>;
  onUnban: (id: string) => Promise<void>;
}) {
  const [saving, setSaving] = useState(false);
  const [showBanModal, setShowBanModal] = useState(false);
  const [banReason, setBanReason] = useState('');

  const isOnline =
    user.last_seen_at && new Date(user.last_seen_at) > new Date(Date.now() - 15 * 60 * 1000);

  const handleFieldChange = async (field: keyof UserProfile, value: unknown) => {
    setSaving(true);
    await onUpdate(user.id, { [field]: value });
    setSaving(false);
  };

  const handleBan = async () => {
    if (!banReason.trim()) {
      alert('Please provide a reason for the ban');
      return;
    }
    setSaving(true);
    await onBan(user.id, banReason);
    setSaving(false);
    setShowBanModal(false);
    setBanReason('');
  };

  const handleUnban = async () => {
    if (!confirm('Are you sure you want to unban this user?')) return;
    setSaving(true);
    await onUnban(user.id);
    setSaving(false);
  };

  return (
    <>
      <tr className={`border-b border-white/[0.06] hover:bg-white/[0.02] ${user.is_banned ? 'bg-red-500/5' : ''}`}>
        <td className="py-4 px-4">
          <div className="flex items-center gap-3">
            <span className={`w-2 h-2 rounded-full ${user.is_banned ? 'bg-red-500' : isOnline ? 'bg-green-500' : 'bg-slate-600'}`} />
            <div>
              <div className="flex items-center gap-2">
                <p className="text-white font-medium">{user.full_name || 'No name'}</p>
                {user.is_banned && (
                  <span className="px-1.5 py-0.5 rounded text-[10px] bg-red-500/20 text-red-400 uppercase">Banned</span>
                )}
              </div>
              <p className="text-xs text-slate-500">{user.email}</p>
              {user.is_banned && user.ban_reason && (
                <p className="text-xs text-red-400 mt-1">Reason: {user.ban_reason}</p>
              )}
            </div>
          </div>
        </td>
        <td className="py-4 px-4">
          <InlineSelect
            value={user.role}
            options={ROLES}
            onChange={(v) => void handleFieldChange('role', v)}
            disabled={saving || user.is_banned}
            colorMap={ROLE_COLORS}
          />
        </td>
        <td className="py-4 px-4">
          <InlineSelect
            value={user.tier}
            options={TIERS}
            onChange={(v) => void handleFieldChange('tier', v)}
            disabled={saving || user.is_banned}
            colorMap={TIER_COLORS}
          />
        </td>
        <td className="py-4 px-4">
          <button
            onClick={() => void handleFieldChange('is_active', !user.is_active)}
            disabled={saving || user.is_banned}
            className={`px-2 py-1 rounded text-xs transition-colors ${
              user.is_active ? 'bg-green-500/20 text-green-300 hover:bg-green-500/30' : 'bg-red-500/20 text-red-300 hover:bg-red-500/30'
            } ${saving || user.is_banned ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
          >
            {user.is_active ? 'Active' : 'Inactive'}
          </button>
        </td>
        <td className="py-4 px-4 text-slate-400 text-sm">
          {user.last_seen_at ? new Date(user.last_seen_at).toLocaleString() : 'Never'}
        </td>
        <td className="py-4 px-4">
          {user.role !== 'admin' && (
            user.is_banned ? (
              <GlassButton
                variant="primary"
                size="sm"
                onClick={() => void handleUnban()}
                disabled={saving}
              >
                Unban
              </GlassButton>
            ) : (
              <GlassButton
                variant="secondary"
                size="sm"
                onClick={() => setShowBanModal(true)}
                className="text-red-400 hover:text-red-300 border-red-500/30 hover:border-red-500/50"
              >
                Ban
              </GlassButton>
            )
          )}
        </td>
      </tr>

      {/* Ban Modal */}
      {showBanModal && (
        <tr>
          <td colSpan={6} className="p-0">
            <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowBanModal(false)}>
              <div className="bg-slate-900 border border-white/10 rounded-lg p-6 w-full max-w-md" onClick={(e) => e.stopPropagation()}>
                <h3 className="text-lg font-bold text-white mb-4">Ban User</h3>
                <p className="text-slate-400 mb-4">
                  You are about to ban <strong className="text-white">{user.email}</strong>
                </p>
                <div className="mb-4">
                  <label className="block text-sm text-slate-400 mb-2">Reason for ban</label>
                  <textarea
                    value={banReason}
                    onChange={(e) => setBanReason(e.target.value)}
                    placeholder="e.g., Violation of terms of service, spam, abuse..."
                    className="w-full bg-slate-800 text-white rounded-lg px-4 py-2 border border-white/10 focus:border-red-500 focus:outline-none resize-none"
                    rows={3}
                  />
                </div>
                <div className="flex gap-3 justify-end">
                  <GlassButton variant="secondary" onClick={() => setShowBanModal(false)}>
                    Cancel
                  </GlassButton>
                  <GlassButton
                    variant="primary"
                    onClick={() => void handleBan()}
                    disabled={saving || !banReason.trim()}
                    loading={saving}
                    className="bg-red-600 hover:bg-red-700"
                  >
                    Ban User
                  </GlassButton>
                </div>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export default function AdminUsersPage() {
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'admin' | 'enterprise' | 'consumer' | 'banned'>('all');
  const [search, setSearch] = useState('');

  const supabase = createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  );

  useEffect(() => {
    void loadUsers();
  }, []);

  const loadUsers = async () => {
    setLoading(true);
    const { data, error } = await supabase
      .from('profiles')
      .select('*')
      .order('created_at', { ascending: false });

    if (error) {
      console.error('Failed to load users:', error);
    } else {
      setUsers(data || []);
    }
    setLoading(false);
  };

  const updateUser = async (id: string, updates: Partial<UserProfile>) => {
    const { error } = await supabase.from('profiles').update(updates).eq('id', id);

    if (error) {
      console.error('Failed to update user:', error);
      alert('Failed to update user: ' + error.message);
    } else {
      setUsers((prev) => prev.map((u) => (u.id === id ? { ...u, ...updates } : u)));
    }
  };

  const banUser = async (id: string, reason: string) => {
    const { error } = await supabase.from('profiles').update({
      is_banned: true,
      ban_reason: reason,
      banned_at: new Date().toISOString(),
      is_active: false,
    }).eq('id', id);

    if (error) {
      console.error('Failed to ban user:', error);
      alert('Failed to ban user: ' + error.message);
    } else {
      setUsers((prev) => prev.map((u) => (u.id === id ? {
        ...u,
        is_banned: true,
        ban_reason: reason,
        banned_at: new Date().toISOString(),
        is_active: false
      } : u)));
    }
  };

  const unbanUser = async (id: string) => {
    const { error } = await supabase.from('profiles').update({
      is_banned: false,
      ban_reason: null,
      banned_at: null,
      is_active: true,
    }).eq('id', id);

    if (error) {
      console.error('Failed to unban user:', error);
      alert('Failed to unban user: ' + error.message);
    } else {
      setUsers((prev) => prev.map((u) => (u.id === id ? {
        ...u,
        is_banned: false,
        ban_reason: null,
        banned_at: null,
        is_active: true
      } : u)));
    }
  };

  const filteredUsers = users.filter((u) => {
    if (filter === 'banned' && !u.is_banned) return false;
    if (filter !== 'all' && filter !== 'banned' && u.role !== filter) return false;
    if (search) {
      const searchLower = search.toLowerCase();
      return (
        u.email.toLowerCase().includes(searchLower) ||
        u.full_name?.toLowerCase().includes(searchLower)
      );
    }
    return true;
  });

  const stats = {
    total: users.length,
    admins: users.filter((u) => u.role === 'admin').length,
    enterprise: users.filter((u) => u.role === 'enterprise').length,
    consumers: users.filter((u) => u.role === 'consumer').length,
    active: users.filter((u) => u.is_active && !u.is_banned).length,
    banned: users.filter((u) => u.is_banned).length,
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">User Management</h1>
        <p className="text-slate-400">View and manage all user accounts</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-6 gap-4 mb-8">
        <GlassCard compact>
          <p className="text-2xl font-bold text-white">{stats.total}</p>
          <p className="text-sm text-slate-400">Total Users</p>
        </GlassCard>
        <GlassCard compact>
          <p className="text-2xl font-bold text-purple-400">{stats.admins}</p>
          <p className="text-sm text-slate-400">Admins</p>
        </GlassCard>
        <GlassCard compact>
          <p className="text-2xl font-bold text-blue-400">{stats.enterprise}</p>
          <p className="text-sm text-slate-400">Enterprise</p>
        </GlassCard>
        <GlassCard compact>
          <p className="text-2xl font-bold text-slate-300">{stats.consumers}</p>
          <p className="text-sm text-slate-400">Consumers</p>
        </GlassCard>
        <GlassCard compact>
          <p className="text-2xl font-bold text-green-400">{stats.active}</p>
          <p className="text-sm text-slate-400">Active</p>
        </GlassCard>
        <GlassCard compact>
          <p className="text-2xl font-bold text-red-400">{stats.banned}</p>
          <p className="text-sm text-slate-400">Banned</p>
        </GlassCard>
      </div>

      {/* Filters */}
      <div className="flex gap-4 mb-6">
        <input
          type="text"
          placeholder="Search by email or name..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="flex-1 bg-slate-800 text-white rounded-lg px-4 py-2 border border-white/10 focus:border-blue-500 focus:outline-none"
        />
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value as typeof filter)}
          className="bg-slate-800 text-white rounded-lg px-4 py-2 border border-white/10"
        >
          <option value="all">All Roles</option>
          <option value="admin">Admins</option>
          <option value="enterprise">Enterprise</option>
          <option value="consumer">Consumers</option>
          <option value="banned">Banned Users</option>
        </select>
        <GlassButton variant="primary" onClick={() => void loadUsers()}>
          Refresh
        </GlassButton>
      </div>

      {/* Users Table */}
      <GlassCard blur="heavy" className="p-0 overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-slate-400">Loading users...</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-xs text-slate-500 uppercase tracking-wide border-b border-white/[0.06]">
                  <th className="py-3 px-4">User</th>
                  <th className="py-3 px-4">Role</th>
                  <th className="py-3 px-4">Tier</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4">Last Seen</th>
                  <th className="py-3 px-4">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredUsers.map((user) => (
                  <UserRow key={user.id} user={user} onUpdate={updateUser} onBan={banUser} onUnban={unbanUser} />
                ))}
                {filteredUsers.length === 0 && (
                  <tr>
                    <td colSpan={6} className="py-8 text-center text-slate-500">
                      No users found
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </GlassCard>
    </div>
  );
}
