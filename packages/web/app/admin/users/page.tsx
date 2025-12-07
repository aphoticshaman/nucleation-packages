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
  last_seen_at: string | null;
  created_at: string;
  onboarding_completed_at: string | null;
}

const ROLES = ['admin', 'enterprise', 'consumer', 'support'] as const;
const TIERS = ['free', 'starter', 'pro', 'enterprise_tier'] as const;

function UserRow({
  user,
  onUpdate,
}: {
  user: UserProfile;
  onUpdate: (id: string, updates: Partial<UserProfile>) => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [localRole, setLocalRole] = useState(user.role);
  const [localTier, setLocalTier] = useState(user.tier);
  const [localActive, setLocalActive] = useState(user.is_active);

  const isOnline =
    user.last_seen_at && new Date(user.last_seen_at) > new Date(Date.now() - 15 * 60 * 1000);

  const handleSave = async () => {
    setSaving(true);
    await onUpdate(user.id, {
      role: localRole,
      tier: localTier,
      is_active: localActive,
    });
    setSaving(false);
    setEditing(false);
  };

  const handleCancel = () => {
    setLocalRole(user.role);
    setLocalTier(user.tier);
    setLocalActive(user.is_active);
    setEditing(false);
  };

  return (
    <tr className="border-b border-white/[0.06] hover:bg-white/[0.02]">
      <td className="py-4 px-4">
        <div className="flex items-center gap-3">
          <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-green-500' : 'bg-slate-600'}`} />
          <div>
            <p className="text-white font-medium">{user.full_name || 'No name'}</p>
            <p className="text-xs text-slate-500">{user.email}</p>
          </div>
        </div>
      </td>
      <td className="py-4 px-4">
        {editing ? (
          <select
            value={localRole}
            onChange={(e) => setLocalRole(e.target.value as typeof localRole)}
            className="bg-slate-800 text-white text-sm rounded px-2 py-1 border border-white/10"
          >
            {ROLES.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        ) : (
          <span
            className={`px-2 py-1 rounded text-xs uppercase ${
              user.role === 'admin'
                ? 'bg-purple-500/20 text-purple-300'
                : user.role === 'enterprise'
                  ? 'bg-blue-500/20 text-blue-300'
                  : 'bg-slate-500/20 text-slate-300'
            }`}
          >
            {user.role}
          </span>
        )}
      </td>
      <td className="py-4 px-4">
        {editing ? (
          <select
            value={localTier}
            onChange={(e) => setLocalTier(e.target.value as typeof localTier)}
            className="bg-slate-800 text-white text-sm rounded px-2 py-1 border border-white/10"
          >
            {TIERS.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        ) : (
          <span
            className={`px-2 py-1 rounded text-xs uppercase ${
              user.tier === 'enterprise_tier'
                ? 'bg-amber-500/20 text-amber-300'
                : user.tier === 'pro'
                  ? 'bg-green-500/20 text-green-300'
                  : 'bg-slate-500/20 text-slate-300'
            }`}
          >
            {user.tier}
          </span>
        )}
      </td>
      <td className="py-4 px-4">
        {editing ? (
          <button
            onClick={() => setLocalActive(!localActive)}
            className={`px-2 py-1 rounded text-xs ${
              localActive ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'
            }`}
          >
            {localActive ? 'Active' : 'Inactive'}
          </button>
        ) : (
          <span
            className={`px-2 py-1 rounded text-xs ${
              user.is_active ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'
            }`}
          >
            {user.is_active ? 'Active' : 'Inactive'}
          </span>
        )}
      </td>
      <td className="py-4 px-4 text-slate-400 text-sm">
        {user.last_seen_at ? new Date(user.last_seen_at).toLocaleString() : 'Never'}
      </td>
      <td className="py-4 px-4">
        {editing ? (
          <div className="flex gap-2">
            <GlassButton
              variant="primary"
              size="sm"
              onClick={() => void handleSave()}
              disabled={saving}
              loading={saving}
            >
              Save
            </GlassButton>
            <GlassButton
              variant="secondary"
              size="sm"
              onClick={handleCancel}
            >
              Cancel
            </GlassButton>
          </div>
        ) : (
          <GlassButton
            variant="secondary"
            size="sm"
            onClick={() => setEditing(true)}
          >
            Edit
          </GlassButton>
        )}
      </td>
    </tr>
  );
}

export default function AdminUsersPage() {
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'admin' | 'enterprise' | 'consumer'>('all');
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

  const filteredUsers = users.filter((u) => {
    if (filter !== 'all' && u.role !== filter) return false;
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
    active: users.filter((u) => u.is_active).length,
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">User Management</h1>
        <p className="text-slate-400">View and manage all user accounts</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-5 gap-4 mb-8">
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
                  <UserRow key={user.id} user={user} onUpdate={updateUser} />
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
