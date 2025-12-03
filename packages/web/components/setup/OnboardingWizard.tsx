'use client';

import { useState, useCallback, useMemo } from 'react';
import type { UserPreferences, UserTier } from '@/lib/config/powerUser';
import { DEFAULT_PREFERENCES, TIER_CAPABILITIES } from '@/lib/config/powerUser';

/**
 * LatticeForge Onboarding Wizard
 *
 * Multi-step configuration that makes users FEEL the power:
 * 1. Welcome & Experience Level Selection
 * 2. Interest Areas (what intel matters to you)
 * 3. Dashboard Configuration
 * 4. Notification Preferences
 * 5. Visual Customization
 * 6. Keyboard Shortcuts
 * 7. Advanced Features (for power users)
 * 8. Review & Launch
 */

// ============================================
// Step Definitions
// ============================================

type WizardStep =
  | 'welcome'
  | 'experience'
  | 'interests'
  | 'dashboard'
  | 'notifications'
  | 'visual'
  | 'shortcuts'
  | 'advanced'
  | 'review';

interface StepConfig {
  id: WizardStep;
  title: string;
  subtitle: string;
  icon: string;
  minTier?: UserTier;
}

const STEPS: StepConfig[] = [
  { id: 'welcome', title: 'Welcome', subtitle: 'Let\'s set up your intelligence platform', icon: '👋' },
  { id: 'experience', title: 'Experience Level', subtitle: 'How familiar are you with geopolitical analysis?', icon: '📊' },
  { id: 'interests', title: 'Focus Areas', subtitle: 'What intelligence matters to you?', icon: '🎯' },
  { id: 'dashboard', title: 'Dashboard', subtitle: 'Configure your command center', icon: '📱' },
  { id: 'notifications', title: 'Alerts', subtitle: 'How should we keep you informed?', icon: '🔔' },
  { id: 'visual', title: 'Appearance', subtitle: 'Make it yours', icon: '🎨' },
  { id: 'shortcuts', title: 'Shortcuts', subtitle: 'Power user controls', icon: '⌨️', minTier: 'analyst' },
  { id: 'advanced', title: 'Advanced', subtitle: 'Unlock full potential', icon: '⚙️', minTier: 'strategist' },
  { id: 'review', title: 'Ready', subtitle: 'Review your configuration', icon: '🚀' },
];

// ============================================
// Configuration State
// ============================================

interface OnboardingConfig {
  // Experience
  experienceLevel: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  primaryUseCase: string;
  organizationType?: string;

  // Interests
  regionsFocus: string[];
  categoriesFocus: string[];
  timeHorizon: 'short' | 'medium' | 'long';

  // Dashboard
  layoutPreset: string;
  enabledWidgets: string[];
  widgetSizes: Record<string, 'small' | 'medium' | 'large'>;

  // Notifications
  alertChannels: string[];
  alertFrequency: string;
  quietHours: boolean;

  // Visual
  theme: string;
  density: string;
  mapStyle: string;
  animations: boolean;

  // Shortcuts
  enableKeyboardNav: boolean;
  customShortcuts: Record<string, string[]>;

  // Advanced
  developerMode: boolean;
  apiAccess: boolean;
  dataExports: boolean;
  experimentalFeatures: boolean;
}

const DEFAULT_CONFIG: OnboardingConfig = {
  experienceLevel: 'intermediate',
  primaryUseCase: 'general',
  regionsFocus: ['global'],
  categoriesFocus: ['political', 'economic', 'security'],
  timeHorizon: 'medium',
  layoutPreset: 'balanced',
  enabledWidgets: ['summary', 'map', 'threats', 'trends'],
  widgetSizes: {},
  alertChannels: ['push'],
  alertFrequency: 'daily',
  quietHours: true,
  theme: 'dark',
  density: 'comfortable',
  mapStyle: 'dark',
  animations: true,
  enableKeyboardNav: true,
  customShortcuts: {},
  developerMode: false,
  apiAccess: false,
  dataExports: false,
  experimentalFeatures: false,
};

// ============================================
// Step Components
// ============================================

interface StepProps {
  config: OnboardingConfig;
  onChange: (updates: Partial<OnboardingConfig>) => void;
  userTier: UserTier;
}

function WelcomeStep({ config, onChange, userTier }: StepProps) {
  return (
    <div className="text-center space-y-6 py-8">
      <div className="text-6xl">🌐</div>
      <div>
        <h2 className="text-2xl font-bold text-white mb-2">Welcome to LatticeForge</h2>
        <p className="text-slate-400 max-w-md mx-auto">
          Your intelligence platform for understanding the world. Let's configure it
          to work exactly how you need it.
        </p>
      </div>

      <div className="bg-slate-800/50 rounded-xl p-6 max-w-md mx-auto">
        <h3 className="text-sm font-medium text-white mb-4">Your Plan: {userTier.toUpperCase()}</h3>
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="flex items-center gap-2">
            <span className="text-green-400">✓</span>
            <span className="text-slate-400">
              {TIER_CAPABILITIES[userTier].maxEntities === -1
                ? 'Unlimited entities'
                : `${TIER_CAPABILITIES[userTier].maxEntities} entities`}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-green-400">✓</span>
            <span className="text-slate-400">
              {TIER_CAPABILITIES[userTier].maxSavedViews === -1
                ? 'Unlimited views'
                : `${TIER_CAPABILITIES[userTier].maxSavedViews} saved views`}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className={TIER_CAPABILITIES[userTier].apiAccess ? 'text-green-400' : 'text-slate-600'}>
              {TIER_CAPABILITIES[userTier].apiAccess ? '✓' : '✗'}
            </span>
            <span className="text-slate-400">API Access</span>
          </div>
          <div className="flex items-center gap-2">
            <span className={TIER_CAPABILITIES[userTier].customDashboards ? 'text-green-400' : 'text-slate-600'}>
              {TIER_CAPABILITIES[userTier].customDashboards ? '✓' : '✗'}
            </span>
            <span className="text-slate-400">Custom Dashboards</span>
          </div>
        </div>
      </div>

      <p className="text-xs text-slate-500">
        This setup takes about 2-3 minutes. You can change everything later.
      </p>
    </div>
  );
}

function ExperienceStep({ config, onChange }: StepProps) {
  const levels = [
    {
      id: 'beginner',
      title: 'New to This',
      icon: '🌱',
      desc: 'I want simple explanations and guidance',
      features: ['Simplified views', 'Guided tours', 'Plain language'],
    },
    {
      id: 'intermediate',
      title: 'Some Experience',
      icon: '📊',
      desc: 'I understand the basics, show me more',
      features: ['Standard dashboards', 'Key metrics', 'Contextual help'],
    },
    {
      id: 'advanced',
      title: 'Experienced Analyst',
      icon: '🔬',
      desc: 'I know what I\'m doing, give me power',
      features: ['Full controls', 'Raw data', 'Advanced filters'],
    },
    {
      id: 'expert',
      title: 'Expert / Developer',
      icon: '🛠️',
      desc: 'Show me everything, I\'ll configure it myself',
      features: ['All features', 'API access', 'Custom integrations'],
    },
  ];

  const useCases = [
    { id: 'general', label: 'General awareness', icon: '🌍' },
    { id: 'investment', label: 'Investment decisions', icon: '📈' },
    { id: 'security', label: 'Security monitoring', icon: '🛡️' },
    { id: 'journalism', label: 'Journalism / Research', icon: '📰' },
    { id: 'government', label: 'Government / Policy', icon: '🏛️' },
    { id: 'risk', label: 'Risk management', icon: '⚠️' },
    { id: 'academic', label: 'Academic research', icon: '🎓' },
    { id: 'business', label: 'Business intelligence', icon: '💼' },
  ];

  return (
    <div className="space-y-8">
      {/* Experience Level */}
      <div>
        <h3 className="text-sm font-medium text-slate-300 mb-4">Your Experience Level</h3>
        <div className="grid grid-cols-2 gap-3">
          {levels.map((level) => (
            <button
              key={level.id}
              onClick={() => onChange({ experienceLevel: level.id as OnboardingConfig['experienceLevel'] })}
              className={`text-left p-4 rounded-xl border transition-all ${
                config.experienceLevel === level.id
                  ? 'bg-blue-600 border-blue-500 text-white'
                  : 'bg-slate-900 border-slate-700 hover:border-slate-600'
              }`}
            >
              <div className="text-2xl mb-2">{level.icon}</div>
              <div className="font-medium text-sm">{level.title}</div>
              <div className={`text-xs mt-1 ${config.experienceLevel === level.id ? 'text-blue-200' : 'text-slate-400'}`}>
                {level.desc}
              </div>
              <div className="flex flex-wrap gap-1 mt-3">
                {level.features.map((f) => (
                  <span
                    key={f}
                    className={`text-xs px-2 py-0.5 rounded ${
                      config.experienceLevel === level.id
                        ? 'bg-blue-500/50'
                        : 'bg-slate-800'
                    }`}
                  >
                    {f}
                  </span>
                ))}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Primary Use Case */}
      <div>
        <h3 className="text-sm font-medium text-slate-300 mb-4">What brings you here?</h3>
        <div className="grid grid-cols-4 gap-2">
          {useCases.map((uc) => (
            <button
              key={uc.id}
              onClick={() => onChange({ primaryUseCase: uc.id })}
              className={`p-3 rounded-lg border text-center transition-all ${
                config.primaryUseCase === uc.id
                  ? 'bg-blue-600 border-blue-500'
                  : 'bg-slate-900 border-slate-700 hover:border-slate-600'
              }`}
            >
              <div className="text-xl mb-1">{uc.icon}</div>
              <div className="text-xs">{uc.label}</div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function InterestsStep({ config, onChange }: StepProps) {
  const regions = [
    { id: 'global', label: 'Global', icon: '🌍' },
    { id: 'north_america', label: 'North America', icon: '🇺🇸' },
    { id: 'europe', label: 'Europe', icon: '🇪🇺' },
    { id: 'asia_pacific', label: 'Asia Pacific', icon: '🌏' },
    { id: 'middle_east', label: 'Middle East', icon: '🏜️' },
    { id: 'africa', label: 'Africa', icon: '🌍' },
    { id: 'latin_america', label: 'Latin America', icon: '🌎' },
    { id: 'russia_eurasia', label: 'Russia & Eurasia', icon: '🇷🇺' },
  ];

  const categories = [
    { id: 'political', label: 'Political', icon: '🏛️' },
    { id: 'economic', label: 'Economic', icon: '📈' },
    { id: 'security', label: 'Security', icon: '⚔️' },
    { id: 'military', label: 'Military', icon: '🎖️' },
    { id: 'cyber', label: 'Cyber', icon: '💻' },
    { id: 'terrorism', label: 'Terrorism', icon: '⚡' },
    { id: 'health', label: 'Health', icon: '🏥' },
    { id: 'scitech', label: 'Science & Tech', icon: '🔬' },
    { id: 'resources', label: 'Resources', icon: '⛏️' },
    { id: 'energy', label: 'Energy', icon: '⚡' },
    { id: 'financial', label: 'Financial', icon: '💰' },
    { id: 'space', label: 'Space', icon: '🛰️' },
  ];

  const toggleRegion = (id: string) => {
    const current = config.regionsFocus;
    const updated = current.includes(id)
      ? current.filter((r) => r !== id)
      : [...current, id];
    onChange({ regionsFocus: updated.length > 0 ? updated : ['global'] });
  };

  const toggleCategory = (id: string) => {
    const current = config.categoriesFocus;
    const updated = current.includes(id)
      ? current.filter((c) => c !== id)
      : [...current, id];
    onChange({ categoriesFocus: updated.length > 0 ? updated : ['political'] });
  };

  return (
    <div className="space-y-8">
      {/* Regions */}
      <div>
        <h3 className="text-sm font-medium text-slate-300 mb-2">Geographic Focus</h3>
        <p className="text-xs text-slate-500 mb-4">Select regions you want to monitor closely</p>
        <div className="grid grid-cols-4 gap-2">
          {regions.map((r) => (
            <button
              key={r.id}
              onClick={() => toggleRegion(r.id)}
              className={`p-3 rounded-lg border transition-all ${
                config.regionsFocus.includes(r.id)
                  ? 'bg-blue-600 border-blue-500'
                  : 'bg-slate-900 border-slate-700 hover:border-slate-600'
              }`}
            >
              <div className="text-xl mb-1">{r.icon}</div>
              <div className="text-xs">{r.label}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Categories */}
      <div>
        <h3 className="text-sm font-medium text-slate-300 mb-2">Intelligence Categories</h3>
        <p className="text-xs text-slate-500 mb-4">What types of intel matter most?</p>
        <div className="grid grid-cols-4 gap-2">
          {categories.map((c) => (
            <button
              key={c.id}
              onClick={() => toggleCategory(c.id)}
              className={`p-3 rounded-lg border transition-all ${
                config.categoriesFocus.includes(c.id)
                  ? 'bg-purple-600 border-purple-500'
                  : 'bg-slate-900 border-slate-700 hover:border-slate-600'
              }`}
            >
              <div className="text-xl mb-1">{c.icon}</div>
              <div className="text-xs">{c.label}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Time Horizon */}
      <div>
        <h3 className="text-sm font-medium text-slate-300 mb-2">Analysis Horizon</h3>
        <div className="flex gap-3">
          {[
            { id: 'short', label: 'Short-term (days-weeks)', icon: '⚡' },
            { id: 'medium', label: 'Medium-term (months)', icon: '📅' },
            { id: 'long', label: 'Long-term (years)', icon: '🔮' },
          ].map((h) => (
            <button
              key={h.id}
              onClick={() => onChange({ timeHorizon: h.id as OnboardingConfig['timeHorizon'] })}
              className={`flex-1 p-4 rounded-lg border transition-all ${
                config.timeHorizon === h.id
                  ? 'bg-amber-600 border-amber-500'
                  : 'bg-slate-900 border-slate-700 hover:border-slate-600'
              }`}
            >
              <div className="text-xl mb-2">{h.icon}</div>
              <div className="text-xs">{h.label}</div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function DashboardStep({ config, onChange, userTier }: StepProps) {
  const layouts = [
    {
      id: 'intel_first',
      name: 'Intel First',
      desc: 'Executive summary front and center',
      preview: '📋 Intel | 🗺️ Map (collapsed)',
    },
    {
      id: 'map_focus',
      name: 'Map Focus',
      desc: 'Interactive map as primary view',
      preview: '🗺️ Map | 📋 Side Panel',
    },
    {
      id: 'balanced',
      name: 'Balanced',
      desc: 'Equal weight to all sections',
      preview: '📋 | 🗺️ | 📊',
    },
    {
      id: 'data_heavy',
      name: 'Data Heavy',
      desc: 'Multiple charts and tables',
      preview: '📊 | 📈 | 📋 | 🗺️',
    },
    {
      id: '3d_tree',
      name: '3D Navigator',
      desc: 'Temporal tree visualization',
      preview: '🌳 3D Tree | 📋 Details',
    },
    {
      id: 'custom',
      name: 'Custom Layout',
      desc: 'Build from scratch',
      preview: '🛠️ Drag & drop',
    },
  ];

  const widgets = [
    { id: 'summary', name: 'Executive Summary', icon: '📋', default: true },
    { id: 'map', name: 'Interactive Map', icon: '🗺️', default: true },
    { id: 'threats', name: 'Threat Matrix', icon: '⚠️', default: true },
    { id: 'trends', name: 'Trends Chart', icon: '📈', default: true },
    { id: 'alerts', name: 'Alert Feed', icon: '🔔', default: false },
    { id: 'news', name: 'News Ticker', icon: '📰', default: false },
    { id: 'xyza', name: 'XYZA Metrics', icon: '🧠', default: false },
    { id: 'flow', name: 'Flow State', icon: '🌊', default: false },
    { id: 'timeline', name: 'Event Timeline', icon: '📅', default: false },
    { id: 'relations', name: 'Relation Graph', icon: '🔗', default: false },
    { id: 'resources', name: 'Resource Flow', icon: '⛏️', default: false },
    { id: 'conflicts', name: 'Conflict Tracker', icon: '🔥', default: false },
  ];

  const toggleWidget = (id: string) => {
    const current = config.enabledWidgets;
    const updated = current.includes(id)
      ? current.filter((w) => w !== id)
      : [...current, id];
    onChange({ enabledWidgets: updated });
  };

  return (
    <div className="space-y-8">
      {/* Layout Presets */}
      <div>
        <h3 className="text-sm font-medium text-slate-300 mb-2">Dashboard Layout</h3>
        <p className="text-xs text-slate-500 mb-4">Choose your primary view structure</p>
        <div className="grid grid-cols-3 gap-3">
          {layouts.map((layout) => (
            <button
              key={layout.id}
              onClick={() => onChange({ layoutPreset: layout.id })}
              className={`p-4 rounded-xl border text-left transition-all ${
                config.layoutPreset === layout.id
                  ? 'bg-blue-600 border-blue-500'
                  : 'bg-slate-900 border-slate-700 hover:border-slate-600'
              }`}
            >
              <div className="font-medium text-sm">{layout.name}</div>
              <div className={`text-xs mt-1 ${config.layoutPreset === layout.id ? 'text-blue-200' : 'text-slate-400'}`}>
                {layout.desc}
              </div>
              <div className="text-xs mt-2 font-mono text-slate-500">{layout.preview}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Widget Selection */}
      <div>
        <h3 className="text-sm font-medium text-slate-300 mb-2">
          Widgets
          <span className="text-slate-500 font-normal ml-2">({config.enabledWidgets.length} selected)</span>
        </h3>
        <p className="text-xs text-slate-500 mb-4">Enable the components you want on your dashboard</p>
        <div className="grid grid-cols-4 gap-2">
          {widgets.map((widget) => (
            <button
              key={widget.id}
              onClick={() => toggleWidget(widget.id)}
              className={`p-3 rounded-lg border transition-all ${
                config.enabledWidgets.includes(widget.id)
                  ? 'bg-green-600 border-green-500'
                  : 'bg-slate-900 border-slate-700 hover:border-slate-600'
              }`}
            >
              <div className="text-xl mb-1">{widget.icon}</div>
              <div className="text-xs">{widget.name}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Widget sizes (for power users) */}
      {userTier !== 'explorer' && config.enabledWidgets.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-slate-300 mb-2">Widget Sizes</h3>
          <div className="space-y-2">
            {config.enabledWidgets.slice(0, 6).map((widgetId) => {
              const widget = widgets.find((w) => w.id === widgetId);
              return (
                <div key={widgetId} className="flex items-center justify-between bg-slate-900 rounded-lg p-2">
                  <span className="text-sm text-slate-300">
                    {widget?.icon} {widget?.name}
                  </span>
                  <div className="flex gap-1">
                    {['small', 'medium', 'large'].map((size) => (
                      <button
                        key={size}
                        onClick={() => onChange({
                          widgetSizes: { ...config.widgetSizes, [widgetId]: size as 'small' | 'medium' | 'large' },
                        })}
                        className={`px-2 py-1 text-xs rounded ${
                          (config.widgetSizes[widgetId] || 'medium') === size
                            ? 'bg-blue-600 text-white'
                            : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        {size[0].toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function NotificationsStep({ config, onChange }: StepProps) {
  const channels = [
    { id: 'push', label: 'Push Notifications', icon: '📱', desc: 'Browser/mobile alerts' },
    { id: 'email', label: 'Email', icon: '📧', desc: 'Daily digest or instant' },
    { id: 'sms', label: 'SMS', icon: '💬', desc: 'Text for critical alerts' },
    { id: 'slack', label: 'Slack', icon: '💼', desc: 'Channel integration' },
    { id: 'webhook', label: 'Webhook', icon: '🔗', desc: 'Custom endpoint' },
  ];

  const frequencies = [
    { id: 'realtime', label: 'Real-time', desc: 'Instant notifications' },
    { id: 'hourly', label: 'Hourly digest', desc: 'Batched every hour' },
    { id: 'daily', label: 'Daily digest', desc: 'Once per day summary' },
    { id: 'weekly', label: 'Weekly digest', desc: 'Weekly roundup' },
  ];

  const toggleChannel = (id: string) => {
    const current = config.alertChannels;
    const updated = current.includes(id)
      ? current.filter((c) => c !== id)
      : [...current, id];
    onChange({ alertChannels: updated });
  };

  return (
    <div className="space-y-8">
      {/* Channels */}
      <div>
        <h3 className="text-sm font-medium text-slate-300 mb-2">Alert Channels</h3>
        <p className="text-xs text-slate-500 mb-4">How should we reach you?</p>
        <div className="space-y-2">
          {channels.map((ch) => (
            <button
              key={ch.id}
              onClick={() => toggleChannel(ch.id)}
              className={`w-full flex items-center justify-between p-4 rounded-lg border transition-all ${
                config.alertChannels.includes(ch.id)
                  ? 'bg-green-600/20 border-green-500'
                  : 'bg-slate-900 border-slate-700 hover:border-slate-600'
              }`}
            >
              <div className="flex items-center gap-3">
                <span className="text-xl">{ch.icon}</span>
                <div className="text-left">
                  <div className="text-sm font-medium text-white">{ch.label}</div>
                  <div className="text-xs text-slate-400">{ch.desc}</div>
                </div>
              </div>
              <div className={`w-8 h-4 rounded-full transition-colors ${
                config.alertChannels.includes(ch.id) ? 'bg-green-500' : 'bg-slate-700'
              }`}>
                <div className={`w-3 h-3 bg-white rounded-full mt-0.5 transition-transform ${
                  config.alertChannels.includes(ch.id) ? 'translate-x-4' : 'translate-x-0.5'
                }`} />
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Frequency */}
      <div>
        <h3 className="text-sm font-medium text-slate-300 mb-4">Alert Frequency</h3>
        <div className="grid grid-cols-2 gap-3">
          {frequencies.map((f) => (
            <button
              key={f.id}
              onClick={() => onChange({ alertFrequency: f.id })}
              className={`p-4 rounded-lg border text-left transition-all ${
                config.alertFrequency === f.id
                  ? 'bg-blue-600 border-blue-500'
                  : 'bg-slate-900 border-slate-700 hover:border-slate-600'
              }`}
            >
              <div className="font-medium text-sm">{f.label}</div>
              <div className={`text-xs mt-1 ${config.alertFrequency === f.id ? 'text-blue-200' : 'text-slate-400'}`}>
                {f.desc}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Quiet Hours */}
      <div className="flex items-center justify-between bg-slate-900 rounded-lg p-4 border border-slate-700">
        <div>
          <div className="text-sm font-medium text-white">Enable Quiet Hours</div>
          <div className="text-xs text-slate-400">No alerts 10 PM - 7 AM (your timezone)</div>
        </div>
        <button
          onClick={() => onChange({ quietHours: !config.quietHours })}
          className={`w-12 h-6 rounded-full transition-colors ${
            config.quietHours ? 'bg-blue-500' : 'bg-slate-700'
          }`}
        >
          <div className={`w-5 h-5 bg-white rounded-full mt-0.5 transition-transform ${
            config.quietHours ? 'translate-x-6' : 'translate-x-0.5'
          }`} />
        </button>
      </div>
    </div>
  );
}

function VisualStep({ config, onChange }: StepProps) {
  const themes = [
    { id: 'dark', label: 'Dark', preview: 'bg-slate-900', icon: '🌙' },
    { id: 'light', label: 'Light', preview: 'bg-white', icon: '☀️' },
    { id: 'midnight', label: 'Midnight', preview: 'bg-slate-950', icon: '🌌' },
    { id: 'military', label: 'Military', preview: 'bg-green-950', icon: '🎖️' },
  ];

  const densities = [
    { id: 'compact', label: 'Compact', desc: 'More info, less space' },
    { id: 'comfortable', label: 'Comfortable', desc: 'Balanced spacing' },
    { id: 'spacious', label: 'Spacious', desc: 'Generous whitespace' },
  ];

  const mapStyles = [
    { id: 'dark', label: 'Dark', icon: '🌑' },
    { id: 'satellite', label: 'Satellite', icon: '🛰️' },
    { id: 'terrain', label: 'Terrain', icon: '🏔️' },
    { id: 'political', label: 'Political', icon: '🗺️' },
  ];

  return (
    <div className="space-y-8">
      {/* Theme */}
      <div>
        <h3 className="text-sm font-medium text-slate-300 mb-4">Color Theme</h3>
        <div className="grid grid-cols-4 gap-3">
          {themes.map((t) => (
            <button
              key={t.id}
              onClick={() => onChange({ theme: t.id })}
              className={`p-4 rounded-xl border transition-all ${
                config.theme === t.id
                  ? 'border-blue-500 ring-2 ring-blue-500/50'
                  : 'border-slate-700 hover:border-slate-600'
              }`}
            >
              <div className={`w-full h-12 rounded-lg ${t.preview} mb-2`} />
              <div className="text-sm text-center">
                {t.icon} {t.label}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Density */}
      <div>
        <h3 className="text-sm font-medium text-slate-300 mb-4">Interface Density</h3>
        <div className="grid grid-cols-3 gap-3">
          {densities.map((d) => (
            <button
              key={d.id}
              onClick={() => onChange({ density: d.id })}
              className={`p-4 rounded-lg border text-left transition-all ${
                config.density === d.id
                  ? 'bg-blue-600 border-blue-500'
                  : 'bg-slate-900 border-slate-700 hover:border-slate-600'
              }`}
            >
              <div className="font-medium text-sm">{d.label}</div>
              <div className={`text-xs mt-1 ${config.density === d.id ? 'text-blue-200' : 'text-slate-400'}`}>
                {d.desc}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Map Style */}
      <div>
        <h3 className="text-sm font-medium text-slate-300 mb-4">Default Map Style</h3>
        <div className="grid grid-cols-4 gap-3">
          {mapStyles.map((m) => (
            <button
              key={m.id}
              onClick={() => onChange({ mapStyle: m.id })}
              className={`p-3 rounded-lg border transition-all ${
                config.mapStyle === m.id
                  ? 'bg-purple-600 border-purple-500'
                  : 'bg-slate-900 border-slate-700 hover:border-slate-600'
              }`}
            >
              <div className="text-xl mb-1">{m.icon}</div>
              <div className="text-xs">{m.label}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Animations */}
      <div className="flex items-center justify-between bg-slate-900 rounded-lg p-4 border border-slate-700">
        <div>
          <div className="text-sm font-medium text-white">Enable Animations</div>
          <div className="text-xs text-slate-400">Smooth transitions and effects</div>
        </div>
        <button
          onClick={() => onChange({ animations: !config.animations })}
          className={`w-12 h-6 rounded-full transition-colors ${
            config.animations ? 'bg-blue-500' : 'bg-slate-700'
          }`}
        >
          <div className={`w-5 h-5 bg-white rounded-full mt-0.5 transition-transform ${
            config.animations ? 'translate-x-6' : 'translate-x-0.5'
          }`} />
        </button>
      </div>
    </div>
  );
}

function ShortcutsStep({ config, onChange }: StepProps) {
  const shortcuts = [
    { key: 'search', label: 'Open Search', default: ['Ctrl', 'K'] },
    { key: 'dashboard', label: 'Go to Dashboard', default: ['G', 'D'] },
    { key: 'map', label: 'Go to Map', default: ['G', 'M'] },
    { key: 'refresh', label: 'Refresh Data', default: ['R'] },
    { key: 'export', label: 'Export View', default: ['Ctrl', 'E'] },
    { key: 'fullscreen', label: 'Fullscreen', default: ['F'] },
    { key: 'filters', label: 'Toggle Filters', default: ['F', 'F'] },
    { key: 'help', label: 'Show Help', default: ['?'] },
  ];

  return (
    <div className="space-y-6">
      {/* Enable keyboard nav */}
      <div className="flex items-center justify-between bg-slate-900 rounded-lg p-4 border border-slate-700">
        <div>
          <div className="text-sm font-medium text-white">Enable Keyboard Navigation</div>
          <div className="text-xs text-slate-400">Use shortcuts to navigate faster</div>
        </div>
        <button
          onClick={() => onChange({ enableKeyboardNav: !config.enableKeyboardNav })}
          className={`w-12 h-6 rounded-full transition-colors ${
            config.enableKeyboardNav ? 'bg-blue-500' : 'bg-slate-700'
          }`}
        >
          <div className={`w-5 h-5 bg-white rounded-full mt-0.5 transition-transform ${
            config.enableKeyboardNav ? 'translate-x-6' : 'translate-x-0.5'
          }`} />
        </button>
      </div>

      {/* Shortcut list */}
      {config.enableKeyboardNav && (
        <div>
          <h3 className="text-sm font-medium text-slate-300 mb-4">Default Shortcuts</h3>
          <div className="space-y-2">
            {shortcuts.map((s) => (
              <div
                key={s.key}
                className="flex items-center justify-between bg-slate-900 rounded-lg p-3 border border-slate-700"
              >
                <span className="text-sm text-slate-300">{s.label}</span>
                <div className="flex items-center gap-1">
                  {s.default.map((key, i) => (
                    <span key={i}>
                      <kbd className="px-2 py-1 bg-slate-800 rounded text-xs font-mono">
                        {key}
                      </kbd>
                      {i < s.default.length - 1 && <span className="mx-1 text-slate-600">+</span>}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p className="text-xs text-slate-500 mt-3">
            You can customize these later in Settings → Keyboard Shortcuts
          </p>
        </div>
      )}
    </div>
  );
}

function AdvancedStep({ config, onChange, userTier }: StepProps) {
  const features = [
    {
      id: 'developerMode',
      label: 'Developer Mode',
      desc: 'Show raw data, debug info, and console logs',
      icon: '🛠️',
      tier: 'analyst' as UserTier,
    },
    {
      id: 'apiAccess',
      label: 'API Access',
      desc: 'Enable API key generation and programmatic access',
      icon: '🔑',
      tier: 'analyst' as UserTier,
    },
    {
      id: 'dataExports',
      label: 'Data Exports',
      desc: 'Export raw data in CSV, JSON, or Excel format',
      icon: '📤',
      tier: 'analyst' as UserTier,
    },
    {
      id: 'experimentalFeatures',
      label: 'Experimental Features',
      desc: 'Early access to new features (may be unstable)',
      icon: '🧪',
      tier: 'strategist' as UserTier,
    },
  ];

  const tierOrder: UserTier[] = ['explorer', 'analyst', 'strategist', 'architect'];
  const userTierIdx = tierOrder.indexOf(userTier);

  return (
    <div className="space-y-6">
      <div className="bg-amber-950/30 border border-amber-800 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <span className="text-xl">⚡</span>
          <div>
            <h4 className="font-medium text-amber-300">Power User Features</h4>
            <p className="text-xs text-amber-400/70 mt-1">
              These advanced options unlock the full potential of LatticeForge.
              Enable only what you need.
            </p>
          </div>
        </div>
      </div>

      <div className="space-y-3">
        {features.map((feature) => {
          const featureTierIdx = tierOrder.indexOf(feature.tier);
          const canAccess = userTierIdx >= featureTierIdx;

          return (
            <div
              key={feature.id}
              className={`flex items-center justify-between p-4 rounded-lg border ${
                canAccess
                  ? 'bg-slate-900 border-slate-700'
                  : 'bg-slate-950 border-slate-800 opacity-50'
              }`}
            >
              <div className="flex items-center gap-3">
                <span className="text-xl">{feature.icon}</span>
                <div>
                  <div className="text-sm font-medium text-white flex items-center gap-2">
                    {feature.label}
                    {!canAccess && (
                      <span className="text-xs px-1.5 py-0.5 bg-slate-800 rounded text-slate-500">
                        {feature.tier}+
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-slate-400">{feature.desc}</div>
                </div>
              </div>
              {canAccess ? (
                <button
                  onClick={() => onChange({ [feature.id]: !config[feature.id as keyof OnboardingConfig] })}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    config[feature.id as keyof OnboardingConfig] ? 'bg-green-500' : 'bg-slate-700'
                  }`}
                >
                  <div className={`w-5 h-5 bg-white rounded-full mt-0.5 transition-transform ${
                    config[feature.id as keyof OnboardingConfig] ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              ) : (
                <span className="text-xs text-slate-500">Upgrade to unlock</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ReviewStep({ config, userTier }: StepProps & { config: OnboardingConfig }) {
  return (
    <div className="space-y-6">
      <div className="text-center py-4">
        <div className="text-4xl mb-3">🎉</div>
        <h2 className="text-xl font-bold text-white">You're All Set!</h2>
        <p className="text-slate-400 text-sm mt-1">Here's a summary of your configuration</p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
          <h4 className="text-xs font-medium text-slate-400 uppercase mb-2">Experience</h4>
          <p className="text-white capitalize">{config.experienceLevel}</p>
        </div>
        <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
          <h4 className="text-xs font-medium text-slate-400 uppercase mb-2">Layout</h4>
          <p className="text-white capitalize">{config.layoutPreset.replace('_', ' ')}</p>
        </div>
        <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
          <h4 className="text-xs font-medium text-slate-400 uppercase mb-2">Regions</h4>
          <p className="text-white">{config.regionsFocus.length} selected</p>
        </div>
        <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
          <h4 className="text-xs font-medium text-slate-400 uppercase mb-2">Widgets</h4>
          <p className="text-white">{config.enabledWidgets.length} enabled</p>
        </div>
        <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
          <h4 className="text-xs font-medium text-slate-400 uppercase mb-2">Alerts</h4>
          <p className="text-white">{config.alertChannels.length} channels</p>
        </div>
        <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
          <h4 className="text-xs font-medium text-slate-400 uppercase mb-2">Theme</h4>
          <p className="text-white capitalize">{config.theme}</p>
        </div>
      </div>

      <div className="bg-blue-950/30 border border-blue-800 rounded-lg p-4">
        <p className="text-sm text-blue-300">
          <strong>Pro tip:</strong> You can change any of these settings later from the
          Settings menu (⚙️) in the top navigation bar.
        </p>
      </div>
    </div>
  );
}

// ============================================
// Main Wizard Component
// ============================================

interface OnboardingWizardProps {
  userTier?: UserTier;
  onComplete: (config: OnboardingConfig) => void;
  onSkip?: () => void;
}

export function OnboardingWizard({
  userTier = 'explorer',
  onComplete,
  onSkip,
}: OnboardingWizardProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [config, setConfig] = useState<OnboardingConfig>(DEFAULT_CONFIG);

  // Filter steps based on user tier
  const availableSteps = useMemo(() => {
    const tierOrder: UserTier[] = ['explorer', 'analyst', 'strategist', 'architect'];
    const userTierIdx = tierOrder.indexOf(userTier);

    return STEPS.filter((step) => {
      if (!step.minTier) return true;
      const stepTierIdx = tierOrder.indexOf(step.minTier);
      return userTierIdx >= stepTierIdx;
    });
  }, [userTier]);

  const currentStepConfig = availableSteps[currentStep];

  const updateConfig = useCallback((updates: Partial<OnboardingConfig>) => {
    setConfig((prev) => ({ ...prev, ...updates }));
  }, []);

  const handleNext = () => {
    if (currentStep < availableSteps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onComplete(config);
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const renderStep = () => {
    const props: StepProps = { config, onChange: updateConfig, userTier };

    switch (currentStepConfig.id) {
      case 'welcome': return <WelcomeStep {...props} />;
      case 'experience': return <ExperienceStep {...props} />;
      case 'interests': return <InterestsStep {...props} />;
      case 'dashboard': return <DashboardStep {...props} />;
      case 'notifications': return <NotificationsStep {...props} />;
      case 'visual': return <VisualStep {...props} />;
      case 'shortcuts': return <ShortcutsStep {...props} />;
      case 'advanced': return <AdvancedStep {...props} />;
      case 'review': return <ReviewStep {...props} config={config} />;
      default: return null;
    }
  };

  return (
    <div className="fixed inset-0 bg-slate-950 z-50 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <span className="text-2xl">🌐</span>
          <span className="font-bold text-white">LatticeForge Setup</span>
        </div>
        {onSkip && (
          <button
            onClick={onSkip}
            className="text-sm text-slate-400 hover:text-white"
          >
            Skip for now →
          </button>
        )}
      </div>

      {/* Progress */}
      <div className="px-6 py-3 border-b border-slate-800">
        <div className="flex items-center gap-2 overflow-x-auto">
          {availableSteps.map((step, idx) => (
            <button
              key={step.id}
              onClick={() => idx <= currentStep && setCurrentStep(idx)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg whitespace-nowrap transition-all ${
                idx === currentStep
                  ? 'bg-blue-600 text-white'
                  : idx < currentStep
                    ? 'bg-green-600/20 text-green-400 cursor-pointer'
                    : 'bg-slate-900 text-slate-500'
              }`}
            >
              {idx < currentStep ? (
                <span>✓</span>
              ) : (
                <span className="text-sm">{step.icon}</span>
              )}
              <span className="text-sm">{step.title}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-6 py-8">
          <div className="mb-8">
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <span>{currentStepConfig.icon}</span>
              <span>{currentStepConfig.title}</span>
            </h2>
            <p className="text-slate-400 text-sm mt-1">{currentStepConfig.subtitle}</p>
          </div>

          {renderStep()}
        </div>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between px-6 py-4 border-t border-slate-800">
        <button
          onClick={handleBack}
          disabled={currentStep === 0}
          className={`px-6 py-2 rounded-lg font-medium transition-colors ${
            currentStep === 0
              ? 'bg-slate-800 text-slate-600 cursor-not-allowed'
              : 'bg-slate-800 text-white hover:bg-slate-700'
          }`}
        >
          ← Back
        </button>

        <div className="text-sm text-slate-500">
          Step {currentStep + 1} of {availableSteps.length}
        </div>

        <button
          onClick={handleNext}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-500 transition-colors"
        >
          {currentStep === availableSteps.length - 1 ? 'Launch LatticeForge 🚀' : 'Next →'}
        </button>
      </div>
    </div>
  );
}

export default OnboardingWizard;
