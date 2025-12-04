'use client';

import { useState, useEffect, Suspense } from 'react';
import dynamic from 'next/dynamic';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';
import { GlassInput, GlassSelect, GlassToggle } from '@/components/ui/GlassInput';
import { supabase } from '@/lib/supabase';

// Plan display component that fetches user role/tier from DB
function PlanCard() {
  const [loading, setLoading] = useState(true);
  const [userRole, setUserRole] = useState<string>('consumer');
  const [userTier, setUserTier] = useState<string>('free');

  useEffect(() => {
    async function fetchUserPlan() {
      try {
        const { data: { user } } = await supabase.auth.getUser();
        if (!user) return;

        const { data: profile } = await supabase
          .from('profiles')
          .select('role, tier')
          .eq('id', user.id)
          .single();

        if (profile) {
          const p = profile as { role?: string; tier?: string };
          setUserRole(p.role || 'consumer');
          setUserTier(p.tier || 'free');
        }
      } catch (err) {
        console.error('Failed to fetch user plan:', err);
      } finally {
        setLoading(false);
      }
    }
    void fetchUserPlan();
  }, []);

  if (loading) {
    return (
      <GlassCard>
        <div className="animate-pulse">
          <div className="h-6 w-24 bg-white/10 rounded mb-2" />
          <div className="h-4 w-48 bg-white/5 rounded" />
        </div>
      </GlassCard>
    );
  }

  // Admin users - show admin status, no upgrade
  if (userRole === 'admin') {
    return (
      <GlassCard className="relative overflow-hidden border-amber-500/30">
        <div
          className="absolute -right-20 -top-20 w-40 h-40 rounded-full opacity-20 pointer-events-none"
          style={{
            background: 'radial-gradient(circle, rgba(245, 158, 11, 0.8) 0%, transparent 70%)',
          }}
        />
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 relative">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-lg font-medium text-white">Administrator</h2>
              <span className="text-[10px] px-2 py-0.5 bg-amber-500/20 rounded-full text-amber-400 uppercase tracking-wider">
                Full Access
              </span>
            </div>
            <p className="text-slate-400 mt-1 text-sm">Unlimited access to all features</p>
          </div>
          <GlassButton
            variant="secondary"
            fullWidthMobile
            onClick={() => (window.location.href = '/admin')}
          >
            Admin Panel
          </GlassButton>
        </div>
      </GlassCard>
    );
  }

  // Paid tiers
  const tierDisplay: Record<string, { name: string; color: string; bg: string; limits: string }> = {
    free: { name: 'Free Plan', color: 'text-slate-300', bg: 'bg-slate-700/50', limits: '10 simulations/day, 5 save slots' },
    starter: { name: 'Pro Plan', color: 'text-blue-400', bg: 'bg-blue-500/20', limits: '50 simulations/day, 25 save slots' },
    pro: { name: 'Team Plan', color: 'text-purple-400', bg: 'bg-purple-500/20', limits: 'Unlimited simulations, API access' },
    enterprise_tier: { name: 'Enterprise', color: 'text-amber-400', bg: 'bg-amber-500/20', limits: 'Custom limits, dedicated support' },
  };

  const display = tierDisplay[userTier] || tierDisplay.free;

  return (
    <GlassCard accent={userTier !== 'free'} glow={userTier !== 'free'} className="relative overflow-hidden">
      <div
        className="absolute -right-20 -top-20 w-40 h-40 rounded-full opacity-20 pointer-events-none"
        style={{
          background: 'radial-gradient(circle, rgba(59, 130, 246, 0.8) 0%, transparent 70%)',
        }}
      />
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 relative">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-medium text-white">{display.name}</h2>
            <span className={`text-[10px] px-2 py-0.5 ${display.bg} rounded-full ${display.color} uppercase tracking-wider`}>
              Current
            </span>
          </div>
          <p className="text-slate-400 mt-1 text-sm">{display.limits}</p>
        </div>
        {userTier === 'free' ? (
          <GlassButton
            variant="primary"
            glow
            fullWidthMobile
            onClick={() => (window.location.href = '/pricing')}
          >
            Upgrade
          </GlassButton>
        ) : (
          <GlassButton
            variant="secondary"
            fullWidthMobile
            onClick={() => (window.location.href = '/pricing')}
          >
            Manage Plan
          </GlassButton>
        )}
      </div>
    </GlassCard>
  );
}

// Dynamic import for integrations (uses useSearchParams)
const IntegrationsSettings = dynamic(
  () => import('@/components/settings/IntegrationsSettings'),
  {
    ssr: false,
    loading: () => (
      <GlassCard>
        <div className="animate-pulse">
          <div className="h-6 w-32 bg-white/10 rounded mb-4" />
          <div className="h-20 bg-white/5 rounded" />
        </div>
      </GlassCard>
    ),
  }
);

export default function ConsumerSettingsPage() {
  const [saving, setSaving] = useState(false);
  const [autoSave, setAutoSave] = useState(false);
  const [emailNotifications, setEmailNotifications] = useState(true);

  const handleSave = async () => {
    setSaving(true);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setSaving(false);
  };

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Header */}
      <div className="pt-2">
        <h1 className="text-2xl sm:text-3xl font-bold text-white">Settings</h1>
        <p className="text-slate-400 mt-1 text-sm sm:text-base">
          Manage your account and preferences
        </p>
      </div>

      {/* Profile */}
      <GlassCard>
        <h2 className="text-lg font-medium text-white mb-5">Profile</h2>
        <div className="space-y-4">
          <GlassInput
            label="Display Name"
            placeholder="Your name"
          />
          <GlassInput
            label="Email"
            type="email"
            disabled
            value="user@example.com"
            hint="Contact support to change email"
          />
        </div>
      </GlassCard>

      {/* Preferences */}
      <GlassCard>
        <h2 className="text-lg font-medium text-white mb-5">Preferences</h2>
        <div className="space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div className="flex-1">
              <p className="text-white text-sm sm:text-base">Default Map Layer</p>
              <p className="text-xs sm:text-sm text-slate-400">Shown when you open the app</p>
            </div>
            <GlassSelect className="sm:w-40">
              <option value="basin">Stability</option>
              <option value="risk">Risk</option>
              <option value="regime">Regimes</option>
            </GlassSelect>
          </div>

          <div className="border-t border-white/[0.06] pt-5">
            <GlassToggle
              checked={autoSave}
              onChange={setAutoSave}
              label="Auto-save simulations"
              description="Save after each run"
            />
          </div>

          <div className="border-t border-white/[0.06] pt-5">
            <GlassToggle
              checked={emailNotifications}
              onChange={setEmailNotifications}
              label="Email notifications"
              description="Updates and announcements"
            />
          </div>
        </div>
      </GlassCard>

      {/* Integrations */}
      <Suspense
        fallback={
          <GlassCard>
            <div className="animate-pulse">
              <div className="h-6 w-32 bg-white/10 rounded mb-4" />
              <div className="h-20 bg-white/5 rounded" />
            </div>
          </GlassCard>
        }
      >
        <IntegrationsSettings />
      </Suspense>

      {/* Usage */}
      <GlassCard>
        <h2 className="text-lg font-medium text-white mb-5">Usage</h2>
        <div className="space-y-5">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-400">Daily simulations</span>
              <span className="text-sm text-white font-medium">7 / 10</span>
            </div>
            <div className="h-2 bg-black/30 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-blue-600 to-blue-400"
                style={{ width: '70%' }}
              />
            </div>
            <p className="text-xs text-slate-500 mt-1.5">Resets at midnight UTC</p>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-400">Saved simulations</span>
              <span className="text-sm text-white font-medium">3 / 5</span>
            </div>
            <div className="h-2 bg-black/30 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-purple-600 to-purple-400"
                style={{ width: '60%' }}
              />
            </div>
          </div>
        </div>
      </GlassCard>

      {/* Plan - fetched from user context */}
      <PlanCard />

      {/* Danger zone */}
      <GlassCard className="border-red-900/30">
        <h2 className="text-lg font-medium text-red-400 mb-4">Danger Zone</h2>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <p className="text-white text-sm sm:text-base">Delete account</p>
            <p className="text-xs sm:text-sm text-slate-400">
              Permanently delete your account and all data
            </p>
          </div>
          <GlassButton variant="danger" fullWidthMobile>
            Delete
          </GlassButton>
        </div>
      </GlassCard>

      {/* Save button - sticky on mobile */}
      <div className="sticky bottom-4 sm:static pt-2 pb-safe">
        <div className="flex justify-end">
          <GlassButton
            variant="primary"
            size="lg"
            glow
            loading={saving}
            fullWidthMobile
            onClick={() => void handleSave()}
          >
            Save Changes
          </GlassButton>
        </div>
      </div>
    </div>
  );
}
