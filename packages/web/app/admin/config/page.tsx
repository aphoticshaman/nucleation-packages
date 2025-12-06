import { requireAdmin } from '@/lib/auth';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';
import { Settings, Server, Database, Globe, Shield, Bell, Save } from 'lucide-react';

export default async function ConfigPage() {
  await requireAdmin();

  const configSections = [
    {
      title: 'General Settings',
      icon: Settings,
      color: 'blue',
      settings: [
        { key: 'app_name', label: 'Application Name', value: 'LatticeForge', type: 'text' },
        { key: 'support_email', label: 'Support Email', value: 'support@latticeforge.ai', type: 'email' },
        { key: 'maintenance_mode', label: 'Maintenance Mode', value: false, type: 'toggle' },
      ]
    },
    {
      title: 'API Configuration',
      icon: Server,
      color: 'green',
      settings: [
        { key: 'rate_limit_default', label: 'Default Rate Limit (req/min)', value: '100', type: 'number' },
        { key: 'rate_limit_pro', label: 'Pro Rate Limit (req/min)', value: '1000', type: 'number' },
        { key: 'rate_limit_enterprise', label: 'Enterprise Rate Limit (req/min)', value: '10000', type: 'number' },
        { key: 'api_timeout', label: 'API Timeout (seconds)', value: '30', type: 'number' },
      ]
    },
    {
      title: 'Database',
      icon: Database,
      color: 'purple',
      settings: [
        { key: 'db_pool_size', label: 'Connection Pool Size', value: '20', type: 'number' },
        { key: 'query_timeout', label: 'Query Timeout (ms)', value: '5000', type: 'number' },
        { key: 'enable_query_logging', label: 'Enable Query Logging', value: true, type: 'toggle' },
      ]
    },
    {
      title: 'Security',
      icon: Shield,
      color: 'red',
      settings: [
        { key: 'session_timeout', label: 'Session Timeout (hours)', value: '24', type: 'number' },
        { key: 'mfa_required', label: 'Require MFA for Admin', value: true, type: 'toggle' },
        { key: 'ip_whitelist_enabled', label: 'IP Whitelist Enabled', value: false, type: 'toggle' },
      ]
    },
  ];

  const featureFlags = [
    { key: 'new_dashboard', label: 'New Dashboard UI', enabled: true, description: 'Enable the redesigned dashboard interface' },
    { key: 'real_time_updates', label: 'Real-time Updates', enabled: true, description: 'WebSocket-based live data streaming' },
    { key: 'advanced_analytics', label: 'Advanced Analytics', enabled: false, description: 'Experimental analytics features' },
    { key: 'beta_api_v2', label: 'API v2 (Beta)', enabled: false, description: 'Next generation API endpoints' },
    { key: 'ai_suggestions', label: 'AI Suggestions', enabled: true, description: 'ML-powered simulation suggestions' },
  ];

  return (
    <div className="pl-72 p-8">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">System Configuration</h1>
          <p className="text-slate-400">Manage platform settings and feature flags</p>
        </div>
        <GlassButton variant="primary" glow>
          <Save className="w-4 h-4 mr-2" />
          Save Changes
        </GlassButton>
      </div>

      {/* Config Sections */}
      <div className="grid grid-cols-2 gap-6 mb-8">
        {configSections.map((section, i) => {
          const IconComponent = section.icon;
          return (
            <GlassCard key={i} blur="heavy">
              <div className="flex items-center gap-3 mb-4">
                <IconComponent className={`w-5 h-5 ${
                  section.color === 'blue' ? 'text-blue-400' :
                  section.color === 'green' ? 'text-green-400' :
                  section.color === 'purple' ? 'text-purple-400' :
                  'text-red-400'
                }`} />
                <h2 className="text-lg font-bold text-white">{section.title}</h2>
              </div>
              <div className="space-y-4">
                {section.settings.map((setting, j) => (
                  <div key={j} className="flex items-center justify-between">
                    <label className="text-sm text-slate-300">{setting.label}</label>
                    {setting.type === 'toggle' ? (
                      <button className={`w-12 h-6 rounded-full transition-colors ${
                        setting.value ? 'bg-blue-500' : 'bg-slate-600'
                      }`}>
                        <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                          setting.value ? 'translate-x-6' : 'translate-x-0.5'
                        }`} />
                      </button>
                    ) : (
                      <input
                        type={setting.type}
                        defaultValue={String(setting.value)}
                        className="w-48 px-3 py-1.5 bg-black/30 border border-white/[0.08] rounded-lg text-white text-sm focus:outline-none focus:border-blue-500/50"
                      />
                    )}
                  </div>
                ))}
              </div>
            </GlassCard>
          );
        })}
      </div>

      {/* Feature Flags */}
      <GlassCard blur="heavy">
        <div className="flex items-center gap-3 mb-4">
          <Bell className="w-5 h-5 text-amber-400" />
          <h2 className="text-lg font-bold text-white">Feature Flags</h2>
        </div>
        <div className="space-y-3">
          {featureFlags.map((flag, i) => (
            <div key={i} className="flex items-center justify-between p-4 bg-black/20 rounded-xl border border-white/[0.04]">
              <div>
                <p className="text-white font-medium">{flag.label}</p>
                <p className="text-sm text-slate-400">{flag.description}</p>
              </div>
              <div className="flex items-center gap-4">
                <span className={`px-2 py-1 rounded text-xs ${
                  flag.enabled ? 'bg-green-500/20 text-green-400' : 'bg-slate-500/20 text-slate-400'
                }`}>
                  {flag.enabled ? 'Enabled' : 'Disabled'}
                </span>
                <button className={`w-12 h-6 rounded-full transition-colors ${
                  flag.enabled ? 'bg-green-500' : 'bg-slate-600'
                }`}>
                  <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                    flag.enabled ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>
            </div>
          ))}
        </div>
      </GlassCard>
    </div>
  );
}
