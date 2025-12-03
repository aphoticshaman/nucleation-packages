'use client';

import { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'next/navigation';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';
import { GlassSelect, GlassToggle } from '@/components/ui/GlassInput';

interface Integration {
  id: string;
  provider: 'slack' | 'teams' | 'webhook' | 'email';
  enabled: boolean;
  alert_config: {
    channel_id?: string;
    channel_name?: string;
    severities?: string[];
    categories?: string[];
    daily_digest?: boolean;
    digest_time?: string;
  };
  last_sync_at?: string;
  created_at: string;
}

interface AvailableProvider {
  id: string;
  name: string;
  description: string;
  icon: string;
  configured: boolean;
  comingSoon: boolean;
}

interface SlackChannel {
  id: string;
  name: string;
  isPrivate: boolean;
  isMember: boolean;
}

const SEVERITY_OPTIONS = [
  { id: 'low', label: 'Low', color: 'text-green-400', bg: 'bg-green-500/20', border: 'border-green-500/30' },
  { id: 'moderate', label: 'Moderate', color: 'text-blue-400', bg: 'bg-blue-500/20', border: 'border-blue-500/30' },
  { id: 'elevated', label: 'Elevated', color: 'text-yellow-400', bg: 'bg-yellow-500/20', border: 'border-yellow-500/30' },
  { id: 'high', label: 'High', color: 'text-orange-400', bg: 'bg-orange-500/20', border: 'border-orange-500/30' },
  { id: 'critical', label: 'Critical', color: 'text-red-400', bg: 'bg-red-500/20', border: 'border-red-500/30' },
];

export default function IntegrationsSettings() {
  const searchParams = useSearchParams();
  const [loading, setLoading] = useState(true);
  const [integrations, setIntegrations] = useState<Integration[]>([]);
  const [providers, setProviders] = useState<AvailableProvider[]>([]);
  const [channels, setChannels] = useState<SlackChannel[]>([]);
  const [loadingChannels, setLoadingChannels] = useState(false);
  const [testingSlack, setTestingSlack] = useState(false);
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null);
  const [savingSettings, setSavingSettings] = useState(false);
  const [notification, setNotification] = useState<{ type: 'success' | 'error'; message: string } | null>(null);

  const slackIntegration = integrations.find((i) => i.provider === 'slack');

  // Check for OAuth callback status
  useEffect(() => {
    const slackStatus = searchParams.get('slack');
    if (slackStatus === 'success') {
      setNotification({ type: 'success', message: 'Slack connected successfully!' });
      window.history.replaceState({}, '', window.location.pathname);
    } else if (slackStatus === 'denied') {
      setNotification({ type: 'error', message: 'Slack authorization was denied' });
      window.history.replaceState({}, '', window.location.pathname);
    } else if (slackStatus === 'error') {
      const reason = searchParams.get('reason') || 'unknown';
      setNotification({ type: 'error', message: `Slack connection failed: ${reason}` });
      window.history.replaceState({}, '', window.location.pathname);
    }
  }, [searchParams]);

  const fetchIntegrations = useCallback(async () => {
    try {
      const res = await fetch('/api/integrations');
      if (res.ok) {
        const data = await res.json();
        setIntegrations(data.integrations || []);
        setProviders(data.availableProviders || []);
      }
    } catch (error) {
      console.error('Failed to fetch integrations:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchIntegrations();
  }, [fetchIntegrations]);

  useEffect(() => {
    if (slackIntegration) {
      fetchChannels();
    }
  }, [slackIntegration?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchChannels = async () => {
    setLoadingChannels(true);
    try {
      const res = await fetch('/api/integrations/slack/channels');
      if (res.ok) {
        const data = await res.json();
        setChannels(data.channels || []);
      }
    } catch (error) {
      console.error('Failed to fetch channels:', error);
    } finally {
      setLoadingChannels(false);
    }
  };

  const connectSlack = () => {
    window.location.href = '/api/integrations/slack/auth';
  };

  const disconnectSlack = async () => {
    if (!confirm('Are you sure you want to disconnect Slack?')) return;

    try {
      const res = await fetch('/api/integrations/slack/settings', { method: 'DELETE' });
      if (res.ok) {
        setNotification({ type: 'success', message: 'Slack disconnected' });
        await fetchIntegrations();
      } else {
        setNotification({ type: 'error', message: 'Failed to disconnect Slack' });
      }
    } catch {
      setNotification({ type: 'error', message: 'Failed to disconnect Slack' });
    }
  };

  const testSlackConnection = async () => {
    setTestingSlack(true);
    setTestResult(null);

    try {
      const res = await fetch('/api/integrations/slack/test', { method: 'POST' });
      const data = await res.json();

      if (res.ok && data.success) {
        setTestResult({ success: true, message: 'Test message sent!' });
      } else {
        setTestResult({ success: false, message: data.error || 'Failed to send' });
      }
    } catch {
      setTestResult({ success: false, message: 'Network error' });
    } finally {
      setTestingSlack(false);
    }
  };

  const updateSlackSettings = async (updates: Record<string, unknown>) => {
    setSavingSettings(true);
    try {
      const res = await fetch('/api/integrations/slack/settings', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
      });

      if (res.ok) {
        await fetchIntegrations();
      } else {
        setNotification({ type: 'error', message: 'Failed to save settings' });
      }
    } catch {
      setNotification({ type: 'error', message: 'Failed to save settings' });
    } finally {
      setSavingSettings(false);
    }
  };

  const toggleSeverity = (severity: string) => {
    const currentSeverities = slackIntegration?.alert_config?.severities || ['high', 'critical'];
    const newSeverities = currentSeverities.includes(severity)
      ? currentSeverities.filter((s) => s !== severity)
      : [...currentSeverities, severity];
    updateSlackSettings({ severities: newSeverities });
  };

  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => setNotification(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  if (loading) {
    return (
      <GlassCard>
        <div className="animate-pulse">
          <div className="h-6 w-32 bg-white/10 rounded mb-4" />
          <div className="h-20 bg-white/5 rounded" />
        </div>
      </GlassCard>
    );
  }

  return (
    <GlassCard>
      <h2 className="text-lg font-medium text-white mb-5">Integrations</h2>

      {/* Notification */}
      {notification && (
        <div
          className={`mb-5 px-4 py-3 rounded-xl text-sm ${
            notification.type === 'success'
              ? 'bg-green-500/20 border border-green-500/30 text-green-300'
              : 'bg-red-500/20 border border-red-500/30 text-red-300'
          }`}
        >
          {notification.message}
        </div>
      )}

      <div className="space-y-4">
        {/* Slack Integration */}
        <div className="bg-black/20 rounded-xl p-4 border border-white/[0.06]">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-10 h-10 sm:w-12 sm:h-12 bg-[#4A154B] rounded-xl flex items-center justify-center shrink-0">
                <svg className="w-6 h-6 text-white" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M5.042 15.165a2.528 2.528 0 0 1-2.52 2.523A2.528 2.528 0 0 1 0 15.165a2.527 2.527 0 0 1 2.522-2.52h2.52v2.52zM6.313 15.165a2.527 2.527 0 0 1 2.521-2.52 2.527 2.527 0 0 1 2.521 2.52v6.313A2.528 2.528 0 0 1 8.834 24a2.528 2.528 0 0 1-2.521-2.522v-6.313zM8.834 5.042a2.528 2.528 0 0 1-2.521-2.52A2.528 2.528 0 0 1 8.834 0a2.528 2.528 0 0 1 2.521 2.522v2.52H8.834zM8.834 6.313a2.528 2.528 0 0 1 2.521 2.521 2.528 2.528 0 0 1-2.521 2.521H2.522A2.528 2.528 0 0 1 0 8.834a2.528 2.528 0 0 1 2.522-2.521h6.312zM18.956 8.834a2.528 2.528 0 0 1 2.522-2.521A2.528 2.528 0 0 1 24 8.834a2.528 2.528 0 0 1-2.522 2.521h-2.522V8.834zM17.688 8.834a2.528 2.528 0 0 1-2.523 2.521 2.527 2.527 0 0 1-2.52-2.521V2.522A2.527 2.527 0 0 1 15.165 0a2.528 2.528 0 0 1 2.523 2.522v6.312zM15.165 18.956a2.528 2.528 0 0 1 2.523 2.522A2.528 2.528 0 0 1 15.165 24a2.527 2.527 0 0 1-2.52-2.522v-2.522h2.52zM15.165 17.688a2.527 2.527 0 0 1-2.52-2.523 2.526 2.526 0 0 1 2.52-2.52h6.313A2.527 2.527 0 0 1 24 15.165a2.528 2.528 0 0 1-2.522 2.523h-6.313z" />
                </svg>
              </div>
              <div className="min-w-0">
                <h3 className="text-white font-medium">Slack</h3>
                <p className="text-xs sm:text-sm text-slate-400 truncate">
                  {slackIntegration ? 'Connected' : 'Receive alerts in Slack channels'}
                </p>
              </div>
            </div>

            {slackIntegration ? (
              <div className="flex items-center gap-3 w-full sm:w-auto justify-end">
                <GlassToggle
                  checked={slackIntegration.enabled}
                  onChange={(checked) => updateSlackSettings({ enabled: checked })}
                  disabled={savingSettings}
                />
                <button
                  onClick={disconnectSlack}
                  className="text-sm text-red-400 hover:text-red-300 min-h-[44px] px-2"
                >
                  Disconnect
                </button>
              </div>
            ) : (
              <GlassButton
                onClick={connectSlack}
                disabled={!providers.find((p) => p.id === 'slack')?.configured}
                className="w-full sm:w-auto bg-[#4A154B] hover:bg-[#611f69] border-[#4A154B]"
              >
                Connect Slack
              </GlassButton>
            )}
          </div>

          {/* Slack Settings */}
          {slackIntegration && slackIntegration.enabled && (
            <div className="mt-5 pt-5 border-t border-white/[0.06] space-y-5">
              {/* Channel */}
              <GlassSelect
                label="Alert Channel"
                value={slackIntegration.alert_config?.channel_id || ''}
                onChange={(e) => {
                  const channel = channels.find((c) => c.id === e.target.value);
                  updateSlackSettings({
                    channelId: e.target.value,
                    channelName: channel?.name || '',
                  });
                }}
                disabled={loadingChannels || savingSettings}
                hint={loadingChannels ? 'Loading channels...' : undefined}
              >
                <option value="">Select a channel...</option>
                {channels.map((ch) => (
                  <option key={ch.id} value={ch.id}>
                    {ch.isPrivate ? '🔒 ' : '#'}{ch.name}
                  </option>
                ))}
              </GlassSelect>

              {/* Severities */}
              <div>
                <label className="block text-sm text-slate-400 mb-3">Alert Severities</label>
                <div className="flex flex-wrap gap-2">
                  {SEVERITY_OPTIONS.map((sev) => {
                    const isActive = slackIntegration.alert_config?.severities?.includes(sev.id);
                    return (
                      <button
                        key={sev.id}
                        onClick={() => toggleSeverity(sev.id)}
                        disabled={savingSettings}
                        className={`
                          px-3 py-2 rounded-lg text-sm font-medium
                          min-h-[44px] touch-manipulation
                          transition-all duration-150 active:scale-[0.97]
                          ${isActive
                            ? `${sev.bg} ${sev.border} ${sev.color} border`
                            : 'bg-black/20 border border-white/[0.06] text-slate-500 hover:text-slate-300'
                          }
                        `}
                      >
                        {sev.label}
                      </button>
                    );
                  })}
                </div>
                <p className="text-xs text-slate-500 mt-2">
                  Only these severity levels will be sent to Slack
                </p>
              </div>

              {/* Daily digest */}
              <div className="pt-2">
                <GlassToggle
                  checked={slackIntegration.alert_config?.daily_digest || false}
                  onChange={(checked) => updateSlackSettings({ dailyDigest: checked })}
                  disabled={savingSettings}
                  label="Daily Digest"
                  description="Receive a daily summary at a scheduled time"
                />
              </div>

              {/* Test */}
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pt-3 border-t border-white/[0.06]">
                <div className="min-w-0">
                  {testResult ? (
                    <p className={`text-sm ${testResult.success ? 'text-green-400' : 'text-red-400'}`}>
                      {testResult.message}
                    </p>
                  ) : slackIntegration.last_sync_at ? (
                    <p className="text-xs text-slate-500">
                      Last sync: {new Date(slackIntegration.last_sync_at).toLocaleString()}
                    </p>
                  ) : null}
                </div>
                <GlassButton
                  variant="secondary"
                  size="sm"
                  onClick={testSlackConnection}
                  disabled={testingSlack || !slackIntegration.alert_config?.channel_id}
                  loading={testingSlack}
                  className="w-full sm:w-auto"
                >
                  Send Test
                </GlassButton>
              </div>
            </div>
          )}
        </div>

        {/* Coming Soon */}
        {providers
          .filter((p) => p.comingSoon)
          .map((provider) => (
            <div
              key={provider.id}
              className="bg-black/10 rounded-xl p-4 border border-white/[0.04] opacity-50"
            >
              <div className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-white/5 rounded-xl flex items-center justify-center text-slate-500">
                    {provider.id === 'teams' && (
                      <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M20.625 8.073h-6.186V6.188a2.438 2.438 0 0 1 2.438-2.438h1.313a2.438 2.438 0 0 1 2.437 2.438v1.885h-.002zm-6.186 1.625h5.373a.813.813 0 0 1 .813.813v5.563a3.25 3.25 0 0 1-3.25 3.25h-1.625a.813.813 0 0 1-.813-.813V10.01a.503.503 0 0 1 .502-.312zM8.875 4.5a3.25 3.25 0 1 0 0 6.5 3.25 3.25 0 0 0 0-6.5zm-.813 8.125a4.875 4.875 0 0 0-4.875 4.875v1.625a.813.813 0 0 0 .813.813h9.75a.813.813 0 0 0 .813-.813V17.5a4.875 4.875 0 0 0-4.875-4.875h-1.626z" />
                      </svg>
                    )}
                    {provider.id === 'webhook' && (
                      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                      </svg>
                    )}
                    {provider.id === 'email' && (
                      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                      </svg>
                    )}
                  </div>
                  <div>
                    <h3 className="text-slate-400 font-medium text-sm">{provider.name}</h3>
                    <p className="text-xs text-slate-500">{provider.description}</p>
                  </div>
                </div>
                <span className="text-[10px] px-2 py-1 bg-white/5 rounded-full text-slate-500 uppercase tracking-wider shrink-0">
                  Soon
                </span>
              </div>
            </div>
          ))}
      </div>
    </GlassCard>
  );
}
