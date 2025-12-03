'use client';

import { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'next/navigation';

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
  { id: 'low', label: 'Low', color: 'text-green-400' },
  { id: 'moderate', label: 'Moderate', color: 'text-blue-400' },
  { id: 'elevated', label: 'Elevated', color: 'text-yellow-400' },
  { id: 'high', label: 'High', color: 'text-orange-400' },
  { id: 'critical', label: 'Critical', color: 'text-red-400' },
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

  // Get Slack integration if exists
  const slackIntegration = integrations.find((i) => i.provider === 'slack');

  // Check for OAuth callback status in URL
  useEffect(() => {
    const slackStatus = searchParams.get('slack');
    if (slackStatus === 'success') {
      setNotification({ type: 'success', message: 'Slack connected successfully!' });
      // Remove the query param
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

  // Fetch integrations
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

  // Fetch Slack channels when we have a Slack integration
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
    if (!confirm('Are you sure you want to disconnect Slack? You will stop receiving alerts.')) {
      return;
    }

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
        setTestResult({ success: true, message: 'Test message sent successfully!' });
      } else {
        setTestResult({ success: false, message: data.error || 'Failed to send test message' });
      }
    } catch {
      setTestResult({ success: false, message: 'Network error' });
    } finally {
      setTestingSlack(false);
    }
  };

  const updateSlackSettings = async (updates: Partial<{
    enabled: boolean;
    channelId: string;
    channelName: string;
    severities: string[];
    dailyDigest: boolean;
    digestTime: string;
  }>) => {
    setSavingSettings(true);
    try {
      const res = await fetch('/api/integrations/slack/settings', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
      });

      if (res.ok) {
        await fetchIntegrations();
        setNotification({ type: 'success', message: 'Settings saved' });
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

  // Clear notification after 5 seconds
  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => setNotification(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  if (loading) {
    return (
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
        <div className="animate-pulse">
          <div className="h-6 w-32 bg-slate-700 rounded mb-4" />
          <div className="h-20 bg-slate-800 rounded" />
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
      <h2 className="text-lg font-medium text-white mb-6">Integrations</h2>

      {/* Notification banner */}
      {notification && (
        <div
          className={`mb-4 px-4 py-3 rounded-lg ${
            notification.type === 'success'
              ? 'bg-green-900/50 border border-green-700 text-green-300'
              : 'bg-red-900/50 border border-red-700 text-red-300'
          }`}
        >
          {notification.message}
        </div>
      )}

      <div className="space-y-4">
        {/* Slack Integration */}
        <div className="border border-slate-700 rounded-lg p-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-[#4A154B] rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-white" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M5.042 15.165a2.528 2.528 0 0 1-2.52 2.523A2.528 2.528 0 0 1 0 15.165a2.527 2.527 0 0 1 2.522-2.52h2.52v2.52zM6.313 15.165a2.527 2.527 0 0 1 2.521-2.52 2.527 2.527 0 0 1 2.521 2.52v6.313A2.528 2.528 0 0 1 8.834 24a2.528 2.528 0 0 1-2.521-2.522v-6.313zM8.834 5.042a2.528 2.528 0 0 1-2.521-2.52A2.528 2.528 0 0 1 8.834 0a2.528 2.528 0 0 1 2.521 2.522v2.52H8.834zM8.834 6.313a2.528 2.528 0 0 1 2.521 2.521 2.528 2.528 0 0 1-2.521 2.521H2.522A2.528 2.528 0 0 1 0 8.834a2.528 2.528 0 0 1 2.522-2.521h6.312zM18.956 8.834a2.528 2.528 0 0 1 2.522-2.521A2.528 2.528 0 0 1 24 8.834a2.528 2.528 0 0 1-2.522 2.521h-2.522V8.834zM17.688 8.834a2.528 2.528 0 0 1-2.523 2.521 2.527 2.527 0 0 1-2.52-2.521V2.522A2.527 2.527 0 0 1 15.165 0a2.528 2.528 0 0 1 2.523 2.522v6.312zM15.165 18.956a2.528 2.528 0 0 1 2.523 2.522A2.528 2.528 0 0 1 15.165 24a2.527 2.527 0 0 1-2.52-2.522v-2.522h2.52zM15.165 17.688a2.527 2.527 0 0 1-2.52-2.523 2.526 2.526 0 0 1 2.52-2.52h6.313A2.527 2.527 0 0 1 24 15.165a2.528 2.528 0 0 1-2.522 2.523h-6.313z"/>
                </svg>
              </div>
              <div>
                <h3 className="text-white font-medium">Slack</h3>
                <p className="text-sm text-slate-400">
                  {slackIntegration ? 'Connected' : 'Receive alerts in Slack channels'}
                </p>
              </div>
            </div>
            {slackIntegration ? (
              <div className="flex items-center gap-2">
                <button
                  onClick={() => updateSlackSettings({ enabled: !slackIntegration.enabled })}
                  disabled={savingSettings}
                  className={`w-12 h-6 rounded-full relative transition-colors ${
                    slackIntegration.enabled ? 'bg-blue-600' : 'bg-slate-700'
                  }`}
                >
                  <span
                    className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                      slackIntegration.enabled ? 'right-1' : 'left-1'
                    }`}
                  />
                </button>
                <button
                  onClick={disconnectSlack}
                  className="text-sm text-red-400 hover:text-red-300"
                >
                  Disconnect
                </button>
              </div>
            ) : (
              <button
                onClick={connectSlack}
                disabled={!providers.find((p) => p.id === 'slack')?.configured}
                className="px-4 py-2 bg-[#4A154B] text-white rounded-lg hover:bg-[#611f69] disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Connect
              </button>
            )}
          </div>

          {/* Slack Settings (when connected) */}
          {slackIntegration && slackIntegration.enabled && (
            <div className="mt-4 pt-4 border-t border-slate-700 space-y-4">
              {/* Channel selector */}
              <div>
                <label className="block text-sm text-slate-400 mb-2">Alert Channel</label>
                <select
                  value={slackIntegration.alert_config?.channel_id || ''}
                  onChange={(e) => {
                    const channel = channels.find((c) => c.id === e.target.value);
                    updateSlackSettings({
                      channelId: e.target.value,
                      channelName: channel?.name || '',
                    });
                  }}
                  disabled={loadingChannels || savingSettings}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white"
                >
                  <option value="">Select a channel...</option>
                  {channels.map((ch) => (
                    <option key={ch.id} value={ch.id}>
                      {ch.isPrivate ? '🔒 ' : '#'}{ch.name}
                    </option>
                  ))}
                </select>
                {loadingChannels && (
                  <p className="text-xs text-slate-500 mt-1">Loading channels...</p>
                )}
              </div>

              {/* Severity filter */}
              <div>
                <label className="block text-sm text-slate-400 mb-2">
                  Alert Severities
                </label>
                <div className="flex flex-wrap gap-2">
                  {SEVERITY_OPTIONS.map((sev) => {
                    const isActive = slackIntegration.alert_config?.severities?.includes(sev.id);
                    return (
                      <button
                        key={sev.id}
                        onClick={() => toggleSeverity(sev.id)}
                        disabled={savingSettings}
                        className={`px-3 py-1.5 rounded-lg text-sm border transition-colors ${
                          isActive
                            ? `bg-slate-700 border-slate-600 ${sev.color}`
                            : 'bg-slate-800 border-slate-700 text-slate-500 hover:border-slate-600'
                        }`}
                      >
                        {sev.label}
                      </button>
                    );
                  })}
                </div>
                <p className="text-xs text-slate-500 mt-1">
                  Only alerts matching these severities will be sent to Slack
                </p>
              </div>

              {/* Daily digest toggle */}
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-white text-sm">Daily Digest</p>
                  <p className="text-xs text-slate-400">
                    Receive a summary of the day&apos;s intel at a scheduled time
                  </p>
                </div>
                <button
                  onClick={() =>
                    updateSlackSettings({
                      dailyDigest: !slackIntegration.alert_config?.daily_digest,
                    })
                  }
                  disabled={savingSettings}
                  className={`w-12 h-6 rounded-full relative transition-colors ${
                    slackIntegration.alert_config?.daily_digest ? 'bg-blue-600' : 'bg-slate-700'
                  }`}
                >
                  <span
                    className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                      slackIntegration.alert_config?.daily_digest ? 'right-1' : 'left-1'
                    }`}
                  />
                </button>
              </div>

              {/* Test connection */}
              <div className="flex items-center justify-between pt-2">
                <div>
                  {testResult && (
                    <p
                      className={`text-sm ${
                        testResult.success ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      {testResult.message}
                    </p>
                  )}
                  {!testResult && slackIntegration.last_sync_at && (
                    <p className="text-xs text-slate-500">
                      Last sync: {new Date(slackIntegration.last_sync_at).toLocaleString()}
                    </p>
                  )}
                </div>
                <button
                  onClick={testSlackConnection}
                  disabled={testingSlack || !slackIntegration.alert_config?.channel_id}
                  className="px-4 py-2 bg-slate-700 text-white rounded-lg hover:bg-slate-600 disabled:opacity-50"
                >
                  {testingSlack ? 'Sending...' : 'Send Test'}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Coming Soon Integrations */}
        {providers
          .filter((p) => p.comingSoon)
          .map((provider) => (
            <div
              key={provider.id}
              className="border border-slate-700/50 rounded-lg p-4 opacity-60"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-slate-800 rounded-lg flex items-center justify-center text-slate-500">
                    {provider.id === 'teams' && (
                      <svg className="w-6 h-6" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M20.625 8.073h-6.186V6.188a2.438 2.438 0 0 1 2.438-2.438h1.313a2.438 2.438 0 0 1 2.437 2.438v1.885h-.002zm-6.186 1.625h5.373a.813.813 0 0 1 .813.813v5.563a3.25 3.25 0 0 1-3.25 3.25h-1.625a.813.813 0 0 1-.813-.813V10.01a.503.503 0 0 1 .502-.312zM8.875 4.5a3.25 3.25 0 1 0 0 6.5 3.25 3.25 0 0 0 0-6.5zm-.813 8.125a4.875 4.875 0 0 0-4.875 4.875v1.625a.813.813 0 0 0 .813.813h9.75a.813.813 0 0 0 .813-.813V17.5a4.875 4.875 0 0 0-4.875-4.875h-1.626z"/>
                      </svg>
                    )}
                    {provider.id === 'webhook' && (
                      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                      </svg>
                    )}
                    {provider.id === 'email' && (
                      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                      </svg>
                    )}
                  </div>
                  <div>
                    <h3 className="text-slate-400 font-medium">{provider.name}</h3>
                    <p className="text-sm text-slate-500">{provider.description}</p>
                  </div>
                </div>
                <span className="px-3 py-1 bg-slate-800 text-slate-500 rounded-full text-xs">
                  Coming Soon
                </span>
              </div>
            </div>
          ))}
      </div>
    </div>
  );
}
