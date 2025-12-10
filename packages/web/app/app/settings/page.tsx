'use client';

import { useState, useEffect, Suspense } from 'react';
import dynamic from 'next/dynamic';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';
import { GlassInput, GlassSelect, GlassToggle } from '@/components/ui/GlassInput';
import { supabase } from '@/lib/supabase';
import { useAccessibility, type AccessibilitySettings } from '@/contexts/AccessibilityContext';

// Colorblind mode options
const COLORBLIND_OPTIONS: { value: AccessibilitySettings['colorblindMode']; label: string; description: string }[] = [
  { value: 'none', label: 'Standard Colors', description: 'Default color scheme' },
  { value: 'deuteranopia', label: 'Deuteranopia', description: 'Red-green colorblindness (most common)' },
  { value: 'protanopia', label: 'Protanopia', description: 'Red colorblindness' },
  { value: 'tritanopia', label: 'Tritanopia', description: 'Blue-yellow colorblindness' },
  { value: 'monochrome', label: 'Monochrome', description: 'Grayscale for maximum contrast' },
];

// Accessibility settings card component
function AccessibilitySettingsCard() {
  // Try to use accessibility context, fall back gracefully
  let accessibilityContext: ReturnType<typeof useAccessibility> | null = null;
  try {
    accessibilityContext = useAccessibility();
  } catch {
    // Context not available
  }

  if (!accessibilityContext) {
    return (
      <GlassCard>
        <h2 className="text-lg font-medium text-white mb-5">Accessibility</h2>
        <p className="text-slate-400 text-sm">
          Accessibility settings are not available on this page.
        </p>
      </GlassCard>
    );
  }

  const { settings, updateSetting, resetSettings, isColorblindMode } = accessibilityContext;

  return (
    <GlassCard>
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-lg font-medium text-white">Accessibility</h2>
        {(isColorblindMode || settings.highContrast || settings.reducedMotion || settings.largeText) && (
          <button
            onClick={resetSettings}
            className="text-xs text-slate-400 hover:text-white transition-colors"
          >
            Reset to defaults
          </button>
        )}
      </div>

      <div className="space-y-5">
        {/* Colorblind Mode */}
        <div>
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div className="flex-1">
              <p className="text-white text-sm sm:text-base">Color Vision</p>
              <p className="text-xs sm:text-sm text-slate-400">
                Adjust colors for your vision type
              </p>
            </div>
            <GlassSelect
              className="sm:w-48"
              value={settings.colorblindMode}
              onChange={(e) => updateSetting('colorblindMode', e.target.value as AccessibilitySettings['colorblindMode'])}
            >
              {COLORBLIND_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </GlassSelect>
          </div>
          {settings.colorblindMode !== 'none' && (
            <div className="mt-3 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
              <div className="flex items-start gap-2">
                <span className="text-blue-400 text-lg">i</span>
                <div>
                  <p className="text-blue-300 text-sm font-medium">
                    {COLORBLIND_OPTIONS.find(o => o.value === settings.colorblindMode)?.label} Mode Active
                  </p>
                  <p className="text-blue-200/70 text-xs mt-0.5">
                    Maps, charts, and risk indicators now use an optimized color palette.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Color Preview */}
        <div className="border-t border-white/[0.06] pt-5">
          <p className="text-white text-sm mb-3">Color Preview</p>
          <div className="flex flex-wrap gap-2">
            {['critical', 'high', 'moderate', 'low', 'safe'].map((level) => {
              const colors = {
                standard: { critical: '#DC2626', high: '#EA580C', moderate: '#CA8A04', low: '#0891B2', safe: '#059669' },
                colorblind: { critical: '#CC3311', high: '#EE7733', moderate: '#CCBB44', low: '#33BBEE', safe: '#0077BB' },
                monochrome: { critical: '#1F2937', high: '#4B5563', moderate: '#6B7280', low: '#9CA3AF', safe: '#D1D5DB' },
              };
              const palette = settings.colorblindMode === 'none' ? 'standard' :
                settings.colorblindMode === 'monochrome' ? 'monochrome' : 'colorblind';
              const color = colors[palette][level as keyof typeof colors.standard];

              return (
                <div key={level} className="flex flex-col items-center">
                  <div
                    className="w-10 h-10 rounded-full border-2 border-white/20"
                    style={{ backgroundColor: color }}
                  />
                  <span className="text-[10px] text-slate-400 mt-1 capitalize">{level}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* High Contrast */}
        <div className="border-t border-white/[0.06] pt-5">
          <GlassToggle
            checked={settings.highContrast}
            onChange={(checked) => updateSetting('highContrast', checked)}
            label="High Contrast"
            description="Increase text and border contrast"
          />
        </div>

        {/* Reduced Motion */}
        <div className="border-t border-white/[0.06] pt-5">
          <GlassToggle
            checked={settings.reducedMotion}
            onChange={(checked) => updateSetting('reducedMotion', checked)}
            label="Reduced Motion"
            description="Minimize animations and transitions"
          />
        </div>

        {/* Large Text */}
        <div className="border-t border-white/[0.06] pt-5">
          <GlassToggle
            checked={settings.largeText}
            onChange={(checked) => updateSetting('largeText', checked)}
            label="Larger Text"
            description="Increase default text size throughout the app"
          />
        </div>

        {/* Simplified UI */}
        <div className="border-t border-white/[0.06] pt-5">
          <GlassToggle
            checked={settings.simplifiedUI}
            onChange={(checked) => updateSetting('simplifiedUI', checked)}
            label="Simplified Interface"
            description="Hide advanced features for a cleaner experience"
          />
        </div>

        {/* Auto Glossary */}
        <div className="border-t border-white/[0.06] pt-5">
          <GlassToggle
            checked={settings.autoGlossary}
            onChange={(checked) => updateSetting('autoGlossary', checked)}
            label="Show Definitions"
            description="Explain technical terms on hover"
          />
        </div>
      </div>
    </GlassCard>
  );
}

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

// Generate time options for the email time picker
function generateTimeOptions() {
  const options = [];
  for (let h = 0; h < 24; h++) {
    const hour = h % 12 || 12;
    const ampm = h < 12 ? 'AM' : 'PM';
    const label = `${hour}:00 ${ampm}`;
    options.push({ value: h.toString().padStart(2, '0'), label });
  }
  return options;
}

const TIME_OPTIONS = generateTimeOptions();

export default function ConsumerSettingsPage() {
  const [saving, setSaving] = useState(false);
  const [autoSave, setAutoSave] = useState(false);
  const [emailNotifications, setEmailNotifications] = useState(true);

  // Daily intel email settings
  const [dailyEmailEnabled, setDailyEmailEnabled] = useState(false);
  const [dailyEmailTime, setDailyEmailTime] = useState('08'); // Default 8 AM UTC
  const [loadingEmailPrefs, setLoadingEmailPrefs] = useState(true);

  // Load email preferences from DB
  useEffect(() => {
    async function loadEmailPrefs() {
      try {
        const { data: { user } } = await supabase.auth.getUser();
        if (!user) return;

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const { data: prefs } = await (supabase as any)
          .from('email_export_preferences')
          .select('enabled, preferred_time')
          .eq('user_id', user.id)
          .single();

        if (prefs) {
          const p = prefs as { enabled?: boolean; preferred_time?: string };
          setDailyEmailEnabled(p.enabled ?? false);
          if (p.preferred_time) {
            // preferred_time is stored as "HH:00" or just hour number
            const hour = parseInt(p.preferred_time.split(':')[0], 10);
            setDailyEmailTime(hour.toString().padStart(2, '0'));
          }
        }
      } catch (err) {
        console.error('Failed to load email preferences:', err);
      } finally {
        setLoadingEmailPrefs(false);
      }
    }
    void loadEmailPrefs();
  }, []);

  // Save email preferences to DB
  const saveEmailPrefs = async (enabled: boolean, time: string) => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      await (supabase as any)
        .from('email_export_preferences')
        .upsert({
          user_id: user.id,
          enabled,
          frequency: 'daily',
          preferred_time: `${time}:00`,
          include_global: true,
          include_watchlist: true,
          format: 'summary',
        }, { onConflict: 'user_id' });
    } catch (err) {
      console.error('Failed to save email preferences:', err);
    }
  };

  const handleEmailToggle = (enabled: boolean) => {
    setDailyEmailEnabled(enabled);
    void saveEmailPrefs(enabled, dailyEmailTime);
  };

  const handleEmailTimeChange = (time: string) => {
    setDailyEmailTime(time);
    void saveEmailPrefs(dailyEmailEnabled, time);
  };

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

      {/* Daily Intel Email */}
      <GlassCard>
        <h2 className="text-lg font-medium text-white mb-5">Daily Intel Brief</h2>
        <div className="space-y-5">
          <GlassToggle
            checked={dailyEmailEnabled}
            onChange={handleEmailToggle}
            label="Daily intelligence email"
            description="Receive a summary of global developments"
            disabled={loadingEmailPrefs}
          />

          {dailyEmailEnabled && (
            <div className="border-t border-white/[0.06] pt-5">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                <div className="flex-1">
                  <p className="text-white text-sm sm:text-base">Delivery time</p>
                  <p className="text-xs sm:text-sm text-slate-400">
                    When to receive your daily briefing (UTC)
                  </p>
                </div>
                <GlassSelect
                  className="sm:w-40"
                  value={dailyEmailTime}
                  onChange={(e) => handleEmailTimeChange(e.target.value)}
                  disabled={loadingEmailPrefs}
                >
                  {TIME_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </GlassSelect>
              </div>
              <p className="text-xs text-slate-500 mt-2">
                Your local time: {new Date(`2024-01-01T${dailyEmailTime}:00:00Z`).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', hour12: true })}
              </p>
            </div>
          )}
        </div>
      </GlassCard>

      {/* Accessibility */}
      <AccessibilitySettingsCard />

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
