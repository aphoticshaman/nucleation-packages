'use client';

import { useState, useEffect } from 'react';
import { createBrowserClient } from '@supabase/ssr';
import { Card, Button } from '@/components/ui';

interface BannedIP {
  id: string;
  ip_address: string;
  ip_range: string | null;
  reason: string;
  banned_by: string | null;
  banned_at: string;
  expires_at: string | null;
  is_active: boolean;
  banner?: {
    email: string;
    full_name: string | null;
  };
}

export default function BannedIPsPage() {
  const [bannedIPs, setBannedIPs] = useState<BannedIP[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAddModal, setShowAddModal] = useState(false);
  const [newIP, setNewIP] = useState('');
  const [newReason, setNewReason] = useState('');
  const [duration, setDuration] = useState<'permanent' | '24h' | '7d' | '30d'>('permanent');
  const [saving, setSaving] = useState(false);

  const supabase = createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  );

  useEffect(() => {
    void loadBannedIPs();
  }, []);

  const loadBannedIPs = async () => {
    setLoading(true);
    const { data, error } = await supabase
      .from('banned_ips')
      .select('*, banner:banned_by(email, full_name)')
      .order('banned_at', { ascending: false });

    if (error) {
      console.error('Failed to load banned IPs:', error);
    } else {
      setBannedIPs(data || []);
    }
    setLoading(false);
  };

  const addBannedIP = async () => {
    if (!newIP.trim() || !newReason.trim()) {
      alert('Please fill in all fields');
      return;
    }

    // Validate IP format
    const ipRegex = /^(\d{1,3}\.){3}\d{1,3}(\/\d{1,2})?$/;
    if (!ipRegex.test(newIP.trim())) {
      alert('Invalid IP address format. Use format like 192.168.1.1 or 192.168.1.0/24 for ranges');
      return;
    }

    setSaving(true);

    let expiresAt: string | null = null;
    if (duration !== 'permanent') {
      const hours = duration === '24h' ? 24 : duration === '7d' ? 168 : 720;
      expiresAt = new Date(Date.now() + hours * 60 * 60 * 1000).toISOString();
    }

    const { data: userData } = await supabase.auth.getUser();

    const { error } = await supabase.from('banned_ips').insert({
      ip_address: newIP.trim(),
      reason: newReason.trim(),
      banned_by: userData.user?.id,
      expires_at: expiresAt,
    });

    if (error) {
      console.error('Failed to ban IP:', error);
      alert('Failed to ban IP: ' + error.message);
    } else {
      setShowAddModal(false);
      setNewIP('');
      setNewReason('');
      setDuration('permanent');
      void loadBannedIPs();
    }
    setSaving(false);
  };

  const unbanIP = async (id: string) => {
    if (!confirm('Are you sure you want to unban this IP address?')) return;

    const { error } = await supabase
      .from('banned_ips')
      .update({ is_active: false })
      .eq('id', id);

    if (error) {
      console.error('Failed to unban IP:', error);
      alert('Failed to unban IP: ' + error.message);
    } else {
      setBannedIPs((prev) => prev.map((ip) => (ip.id === id ? { ...ip, is_active: false } : ip)));
    }
  };

  const stats = {
    total: bannedIPs.length,
    active: bannedIPs.filter((ip) => ip.is_active).length,
    permanent: bannedIPs.filter((ip) => ip.is_active && !ip.expires_at).length,
    temporary: bannedIPs.filter((ip) => ip.is_active && ip.expires_at).length,
  };

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-lg font-bold text-white">Banned IP Addresses</h1>
          <p className="text-slate-400">Manage IP-based access restrictions</p>
        </div>
        <Button variant="secondary" onClick={() => setShowAddModal(true)}>
          + Ban IP Address
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        <Card>
          <p className="text-2xl font-bold text-white">{stats.total}</p>
          <p className="text-sm text-slate-400">Total Bans</p>
        </Card>
        <Card>
          <p className="text-2xl font-bold text-red-400">{stats.active}</p>
          <p className="text-sm text-slate-400">Active Bans</p>
        </Card>
        <Card>
          <p className="text-2xl font-bold text-amber-400">{stats.permanent}</p>
          <p className="text-sm text-slate-400">Permanent</p>
        </Card>
        <Card>
          <p className="text-2xl font-bold text-blue-400">{stats.temporary}</p>
          <p className="text-sm text-slate-400">Temporary</p>
        </Card>
      </div>

      {/* Banned IPs Table */}
      <Card className="p-0 overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-slate-400">Loading banned IPs...</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-xs text-slate-500 uppercase tracking-wide border-b border-white/[0.06]">
                  <th className="py-3 px-4">IP Address</th>
                  <th className="py-3 px-4">Reason</th>
                  <th className="py-3 px-4">Banned By</th>
                  <th className="py-3 px-4">Banned At</th>
                  <th className="py-3 px-4">Expires</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4">Actions</th>
                </tr>
              </thead>
              <tbody>
                {bannedIPs.map((ip) => (
                  <tr key={ip.id} className={`border-b border-white/[0.06] hover:bg-white/[0.02] ${!ip.is_active ? 'opacity-50' : ''}`}>
                    <td className="py-4 px-4">
                      <code className="text-amber-400 bg-amber-500/10 px-2 py-1 rounded text-sm font-mono">
                        {ip.ip_address}
                      </code>
                      {ip.ip_range && (
                        <span className="ml-2 text-xs text-slate-500">Range: {ip.ip_range}</span>
                      )}
                    </td>
                    <td className="py-4 px-4 text-slate-300 text-sm max-w-xs truncate">
                      {ip.reason}
                    </td>
                    <td className="py-4 px-4 text-slate-400 text-sm">
                      {ip.banner?.full_name || ip.banner?.email || 'System'}
                    </td>
                    <td className="py-4 px-4 text-slate-400 text-sm">
                      {new Date(ip.banned_at).toLocaleString()}
                    </td>
                    <td className="py-4 px-4 text-sm">
                      {ip.expires_at ? (
                        <span className={new Date(ip.expires_at) < new Date() ? 'text-slate-500' : 'text-blue-400'}>
                          {new Date(ip.expires_at).toLocaleString()}
                        </span>
                      ) : (
                        <span className="text-amber-400">Permanent</span>
                      )}
                    </td>
                    <td className="py-4 px-4">
                      <span className={`px-2 py-1 rounded text-xs ${
                        ip.is_active
                          ? 'bg-red-500/20 text-red-400'
                          : 'bg-slate-500/20 text-slate-400'
                      }`}>
                        {ip.is_active ? 'Active' : 'Inactive'}
                      </span>
                    </td>
                    <td className="py-4 px-4">
                      {ip.is_active && (
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => void unbanIP(ip.id)}
                        >
                          Unban
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
                {bannedIPs.length === 0 && (
                  <tr>
                    <td colSpan={7} className="py-8 text-center text-slate-500">
                      No banned IP addresses
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* Add Ban Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowAddModal(false)}>
          <div className="bg-slate-900 border border-white/10 rounded-md p-6 w-full max-w-md" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-lg font-bold text-white mb-4">Ban IP Address</h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm text-slate-400 mb-2">IP Address</label>
                <input
                  type="text"
                  value={newIP}
                  onChange={(e) => setNewIP(e.target.value)}
                  placeholder="e.g., 192.168.1.1 or 192.168.1.0/24"
                  className="w-full bg-slate-800 text-white rounded-md px-4 py-2 border border-white/10 focus:border-red-500 focus:outline-none font-mono"
                />
              </div>

              <div>
                <label className="block text-sm text-slate-400 mb-2">Reason</label>
                <textarea
                  value={newReason}
                  onChange={(e) => setNewReason(e.target.value)}
                  placeholder="e.g., Brute force attempts, suspicious activity, VPN abuse..."
                  className="w-full bg-slate-800 text-white rounded-md px-4 py-2 border border-white/10 focus:border-red-500 focus:outline-none resize-none"
                  rows={2}
                />
              </div>

              <div>
                <label className="block text-sm text-slate-400 mb-2">Duration</label>
                <select
                  value={duration}
                  onChange={(e) => setDuration(e.target.value as typeof duration)}
                  className="w-full bg-slate-800 text-white rounded-md px-4 py-2 border border-white/10"
                >
                  <option value="permanent">Permanent</option>
                  <option value="24h">24 Hours</option>
                  <option value="7d">7 Days</option>
                  <option value="30d">30 Days</option>
                </select>
              </div>
            </div>

            <div className="flex gap-3 justify-end mt-6">
              <Button variant="secondary" onClick={() => setShowAddModal(false)}>
                Cancel
              </Button>
              <Button
                variant="secondary"
                onClick={() => void addBannedIP()}
                disabled={saving || !newIP.trim() || !newReason.trim()}
                className="bg-red-600 hover:bg-red-700"
              >
                {saving ? 'Saving...' : 'Ban IP'}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
