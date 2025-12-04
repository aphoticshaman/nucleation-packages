'use client';

import { useState } from 'react';
import {
  Bell,
  Mail,
  MessageSquare,
  Smartphone,
  Globe,
  Shield,
  TrendingUp,
  AlertTriangle,
  Clock,
  Volume2,
  VolumeX,
  Save,
} from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';
import { GlassToggle, GlassSelect } from '@/components/ui/GlassInput';

interface NotificationChannel {
  id: string;
  name: string;
  icon: React.ReactNode;
  enabled: boolean;
  connected: boolean;
}

interface NotificationCategory {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  channels: {
    email: boolean;
    push: boolean;
    slack: boolean;
    inApp: boolean;
  };
}

export default function NotificationsPage() {
  const [saving, setSaving] = useState(false);
  const [channels, setChannels] = useState<NotificationChannel[]>([
    { id: 'email', name: 'Email', icon: <Mail className="w-5 h-5" />, enabled: true, connected: true },
    { id: 'push', name: 'Push Notifications', icon: <Smartphone className="w-5 h-5" />, enabled: false, connected: false },
    { id: 'slack', name: 'Slack', icon: <MessageSquare className="w-5 h-5" />, enabled: false, connected: false },
    { id: 'inApp', name: 'In-App', icon: <Bell className="w-5 h-5" />, enabled: true, connected: true },
  ]);

  const [categories, setCategories] = useState<NotificationCategory[]>([
    {
      id: 'critical',
      name: 'Critical Alerts',
      description: 'Urgent situations requiring immediate attention',
      icon: <AlertTriangle className="w-5 h-5" />,
      color: 'text-red-400',
      channels: { email: true, push: true, slack: true, inApp: true },
    },
    {
      id: 'risk',
      name: 'Risk Changes',
      description: 'Significant changes in nation risk scores',
      icon: <TrendingUp className="w-5 h-5" />,
      color: 'text-orange-400',
      channels: { email: true, push: false, slack: true, inApp: true },
    },
    {
      id: 'regime',
      name: 'Regime Shifts',
      description: 'Changes in nation stability classification',
      icon: <Globe className="w-5 h-5" />,
      color: 'text-purple-400',
      channels: { email: true, push: false, slack: true, inApp: true },
    },
    {
      id: 'security',
      name: 'Security Events',
      description: 'Conflict, unrest, or military activity',
      icon: <Shield className="w-5 h-5" />,
      color: 'text-amber-400',
      channels: { email: true, push: true, slack: true, inApp: true },
    },
    {
      id: 'briefings',
      name: 'Intel Briefings',
      description: 'New briefings and analysis reports',
      icon: <Bell className="w-5 h-5" />,
      color: 'text-blue-400',
      channels: { email: true, push: false, slack: false, inApp: true },
    },
  ]);

  const [digestSettings, setDigestSettings] = useState({
    enabled: true,
    frequency: 'daily',
    time: '08:00',
    timezone: 'America/New_York',
  });

  const [quietHours, setQuietHours] = useState({
    enabled: false,
    start: '22:00',
    end: '08:00',
    allowCritical: true,
  });

  const toggleChannel = (channelId: string) => {
    setChannels((prev) =>
      prev.map((ch) =>
        ch.id === channelId ? { ...ch, enabled: !ch.enabled } : ch
      )
    );
  };

  const toggleCategoryChannel = (categoryId: string, channelId: keyof NotificationCategory['channels']) => {
    setCategories((prev) =>
      prev.map((cat) =>
        cat.id === categoryId
          ? { ...cat, channels: { ...cat.channels, [channelId]: !cat.channels[channelId] } }
          : cat
      )
    );
  };

  const handleSave = async () => {
    setSaving(true);
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1500));
    setSaving(false);
  };

  return (
    <div className="space-y-6 max-w-3xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Notification Preferences</h1>
        <p className="text-slate-400 mt-1">
          Control how and when you receive alerts from LatticeForge
        </p>
      </div>

      {/* Notification Channels */}
      <GlassCard blur="heavy">
        <h2 className="text-lg font-semibold text-white mb-4">Notification Channels</h2>
        <div className="space-y-3">
          {channels.map((channel) => (
            <div
              key={channel.id}
              className="flex items-center justify-between p-3 bg-black/20 rounded-xl"
            >
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                  channel.enabled ? 'bg-blue-500/20 text-blue-400' : 'bg-white/5 text-slate-500'
                }`}>
                  {channel.icon}
                </div>
                <div>
                  <p className="text-white font-medium">{channel.name}</p>
                  <p className="text-xs text-slate-500">
                    {channel.connected ? 'Connected' : 'Not connected'}
                  </p>
                </div>
              </div>
              {channel.connected ? (
                <GlassToggle
                  checked={channel.enabled}
                  onChange={() => toggleChannel(channel.id)}
                />
              ) : (
                <GlassButton
                  variant="secondary"
                  size="sm"
                  onClick={() => window.location.href = '/app/integrations'}
                >
                  Connect
                </GlassButton>
              )}
            </div>
          ))}
        </div>
      </GlassCard>

      {/* Alert Categories */}
      <GlassCard blur="heavy">
        <h2 className="text-lg font-semibold text-white mb-4">Alert Categories</h2>
        <p className="text-sm text-slate-400 mb-4">
          Choose which channels receive each type of alert
        </p>

        <div className="space-y-4">
          {categories.map((category) => (
            <div key={category.id} className="p-4 bg-black/20 rounded-xl">
              <div className="flex items-start gap-3 mb-3">
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center bg-white/5 ${category.color}`}>
                  {category.icon}
                </div>
                <div>
                  <p className="text-white font-medium">{category.name}</p>
                  <p className="text-xs text-slate-500">{category.description}</p>
                </div>
              </div>

              <div className="grid grid-cols-4 gap-2">
                {(['email', 'push', 'slack', 'inApp'] as const).map((channelId) => {
                  const channel = channels.find((c) => c.id === channelId);
                  const isEnabled = category.channels[channelId];
                  const isChannelConnected = channel?.connected && channel?.enabled;

                  return (
                    <button
                      key={channelId}
                      onClick={() => isChannelConnected && toggleCategoryChannel(category.id, channelId)}
                      disabled={!isChannelConnected}
                      className={`p-2 rounded-lg text-xs font-medium transition-all ${
                        !isChannelConnected
                          ? 'bg-white/5 text-slate-600 cursor-not-allowed'
                          : isEnabled
                          ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                          : 'bg-white/5 text-slate-400 hover:text-white'
                      }`}
                    >
                      {channel?.name.split(' ')[0]}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </GlassCard>

      {/* Digest Settings */}
      <GlassCard blur="heavy">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">Daily Digest</h2>
          <GlassToggle
            checked={digestSettings.enabled}
            onChange={(checked) => setDigestSettings((prev) => ({ ...prev, enabled: checked }))}
          />
        </div>

        {digestSettings.enabled && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <GlassSelect
                label="Frequency"
                value={digestSettings.frequency}
                onChange={(e) => setDigestSettings((prev) => ({ ...prev, frequency: e.target.value }))}
              >
                <option value="daily">Daily</option>
                <option value="weekly">Weekly</option>
                <option value="realtime">Real-time</option>
              </GlassSelect>

              <div>
                <label className="block text-sm text-slate-400 mb-2">Delivery Time</label>
                <input
                  type="time"
                  value={digestSettings.time}
                  onChange={(e) => setDigestSettings((prev) => ({ ...prev, time: e.target.value }))}
                  className="w-full px-4 py-3 bg-black/30 border border-white/[0.08] rounded-xl text-white focus:border-blue-500/50 focus:outline-none"
                />
              </div>
            </div>

            <GlassSelect
              label="Timezone"
              value={digestSettings.timezone}
              onChange={(e) => setDigestSettings((prev) => ({ ...prev, timezone: e.target.value }))}
            >
              <option value="America/New_York">Eastern Time (ET)</option>
              <option value="America/Chicago">Central Time (CT)</option>
              <option value="America/Denver">Mountain Time (MT)</option>
              <option value="America/Los_Angeles">Pacific Time (PT)</option>
              <option value="UTC">UTC</option>
              <option value="Europe/London">London (GMT)</option>
              <option value="Europe/Paris">Paris (CET)</option>
              <option value="Asia/Tokyo">Tokyo (JST)</option>
            </GlassSelect>
          </div>
        )}
      </GlassCard>

      {/* Quiet Hours */}
      <GlassCard blur="heavy">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
              {quietHours.enabled ? (
                <VolumeX className="w-5 h-5 text-purple-400" />
              ) : (
                <Volume2 className="w-5 h-5 text-purple-400" />
              )}
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">Quiet Hours</h2>
              <p className="text-xs text-slate-500">Pause non-critical notifications</p>
            </div>
          </div>
          <GlassToggle
            checked={quietHours.enabled}
            onChange={(checked) => setQuietHours((prev) => ({ ...prev, enabled: checked }))}
          />
        </div>

        {quietHours.enabled && (
          <div className="space-y-4 pt-4 border-t border-white/[0.06]">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-slate-400 mb-2">Start Time</label>
                <input
                  type="time"
                  value={quietHours.start}
                  onChange={(e) => setQuietHours((prev) => ({ ...prev, start: e.target.value }))}
                  className="w-full px-4 py-3 bg-black/30 border border-white/[0.08] rounded-xl text-white focus:border-blue-500/50 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-2">End Time</label>
                <input
                  type="time"
                  value={quietHours.end}
                  onChange={(e) => setQuietHours((prev) => ({ ...prev, end: e.target.value }))}
                  className="w-full px-4 py-3 bg-black/30 border border-white/[0.08] rounded-xl text-white focus:border-blue-500/50 focus:outline-none"
                />
              </div>
            </div>

            <GlassToggle
              checked={quietHours.allowCritical}
              onChange={(checked) => setQuietHours((prev) => ({ ...prev, allowCritical: checked }))}
              label="Allow critical alerts"
              description="Critical alerts will still come through during quiet hours"
            />
          </div>
        )}
      </GlassCard>

      {/* Save Button */}
      <div className="sticky bottom-4 flex justify-end">
        <GlassButton
          variant="primary"
          size="lg"
          glow
          loading={saving}
          onClick={() => void handleSave()}
        >
          <Save className="w-4 h-4 mr-2" />
          Save Preferences
        </GlassButton>
      </div>
    </div>
  );
}
