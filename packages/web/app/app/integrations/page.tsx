'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  Search,
  Filter,
  ExternalLink,
  Check,
  Clock,
  MessageSquare,
  AlertTriangle,
  Webhook,
  Mail,
  Shield,
  Database,
  BarChart3,
  Zap,
  ArrowRight,
} from 'lucide-react';
import { Card, Button } from '@/components/ui';


interface Integration {
  id: string;
  name: string;
  description: string;
  icon: string;
  configured: boolean;
  comingSoon: boolean;
  category: string;
}

const CATEGORY_INFO: Record<string, { name: string; icon: React.ReactNode; color: string }> = {
  messaging: { name: 'Messaging', icon: <MessageSquare className="w-4 h-4" />, color: 'text-blue-400' },
  incident: { name: 'Incident Management', icon: <AlertTriangle className="w-4 h-4" />, color: 'text-orange-400' },
  automation: { name: 'Automation', icon: <Zap className="w-4 h-4" />, color: 'text-yellow-400' },
  ticketing: { name: 'Ticketing', icon: <Database className="w-4 h-4" />, color: 'text-purple-400' },
  siem: { name: 'SIEM', icon: <Shield className="w-4 h-4" />, color: 'text-red-400' },
  monitoring: { name: 'Monitoring', icon: <BarChart3 className="w-4 h-4" />, color: 'text-green-400' },
};

const INTEGRATION_ICONS: Record<string, React.ReactNode> = {
  slack: (
    <svg className="w-6 h-6" viewBox="0 0 24 24" fill="currentColor">
      <path d="M5.042 15.165a2.528 2.528 0 0 1-2.52 2.523A2.528 2.528 0 0 1 0 15.165a2.527 2.527 0 0 1 2.522-2.52h2.52v2.52zM6.313 15.165a2.527 2.527 0 0 1 2.521-2.52 2.527 2.527 0 0 1 2.521 2.52v6.313A2.528 2.528 0 0 1 8.834 24a2.528 2.528 0 0 1-2.521-2.522v-6.313zM8.834 5.042a2.528 2.528 0 0 1-2.521-2.52A2.528 2.528 0 0 1 8.834 0a2.528 2.528 0 0 1 2.521 2.522v2.52H8.834zM8.834 6.313a2.528 2.528 0 0 1 2.521 2.521 2.528 2.528 0 0 1-2.521 2.521H2.522A2.528 2.528 0 0 1 0 8.834a2.528 2.528 0 0 1 2.522-2.521h6.312zM18.956 8.834a2.528 2.528 0 0 1 2.522-2.521A2.528 2.528 0 0 1 24 8.834a2.528 2.528 0 0 1-2.522 2.521h-2.522V8.834zM17.688 8.834a2.528 2.528 0 0 1-2.523 2.521 2.527 2.527 0 0 1-2.52-2.521V2.522A2.527 2.527 0 0 1 15.165 0a2.528 2.528 0 0 1 2.523 2.522v6.312zM15.165 18.956a2.528 2.528 0 0 1 2.523 2.522A2.528 2.528 0 0 1 15.165 24a2.527 2.527 0 0 1-2.52-2.522v-2.522h2.52zM15.165 17.688a2.527 2.527 0 0 1-2.52-2.523 2.526 2.526 0 0 1 2.52-2.52h6.313A2.527 2.527 0 0 1 24 15.165a2.528 2.528 0 0 1-2.522 2.523h-6.313z" />
    </svg>
  ),
  discord: (
    <svg className="w-6 h-6" viewBox="0 0 24 24" fill="currentColor">
      <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z"/>
    </svg>
  ),
  teams: (
    <svg className="w-6 h-6" viewBox="0 0 24 24" fill="currentColor">
      <path d="M20.625 8.073h-6.186V6.188a2.438 2.438 0 0 1 2.438-2.438h1.313a2.438 2.438 0 0 1 2.437 2.438v1.885h-.002zm-6.186 1.625h5.373a.813.813 0 0 1 .813.813v5.563a3.25 3.25 0 0 1-3.25 3.25h-1.625a.813.813 0 0 1-.813-.813V10.01a.503.503 0 0 1 .502-.312zM8.875 4.5a3.25 3.25 0 1 0 0 6.5 3.25 3.25 0 0 0 0-6.5zm-.813 8.125a4.875 4.875 0 0 0-4.875 4.875v1.625a.813.813 0 0 0 .813.813h9.75a.813.813 0 0 0 .813-.813V17.5a4.875 4.875 0 0 0-4.875-4.875h-1.626z" />
    </svg>
  ),
  pagerduty: (
    <svg className="w-6 h-6" viewBox="0 0 24 24" fill="currentColor">
      <path d="M16.965 1.18C15.085.164 13.769 0 10.683 0H3.73v14.55h6.926c2.743 0 4.8-.164 6.61-1.37 1.975-1.303 3.004-3.414 3.004-6.17 0-3.01-1.32-4.898-3.305-5.83zM11.16 10.67H7.208V3.674l3.882-.005c2.915-.003 4.538 1.131 4.538 3.46 0 2.468-1.771 3.541-4.468 3.541zM3.73 24h3.478v-6.12H3.73V24z"/>
    </svg>
  ),
  webhook: <Webhook className="w-6 h-6" />,
  email: <Mail className="w-6 h-6" />,
  opsgenie: (
    <svg className="w-6 h-6" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 0L1.608 6v12L12 24l10.392-6V6L12 0zm0 3.6l7.2 4.16v8.32L12 20.24l-7.2-4.16V7.76L12 3.6z"/>
    </svg>
  ),
  jira: (
    <svg className="w-6 h-6" viewBox="0 0 24 24" fill="currentColor">
      <path d="M11.571 11.513H0a5.218 5.218 0 0 0 5.232 5.215h2.13v2.057A5.215 5.215 0 0 0 12.575 24V12.518a1.005 1.005 0 0 0-1.005-1.005zM5.024 5.721H16.59a5.218 5.218 0 0 0-5.232-5.215H9.228V.449A5.214 5.214 0 0 0 4.015 5.664v.057h1.009zM5.568 11.513a5.218 5.218 0 0 0 5.232 5.215h2.13v2.057A5.215 5.215 0 0 0 18.142 24V12.518a1.005 1.005 0 0 0-1.005-1.005H5.568z"/>
    </svg>
  ),
  servicenow: (
    <svg className="w-6 h-6" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm0 18.75c-3.722 0-6.75-3.028-6.75-6.75S8.278 5.25 12 5.25s6.75 3.028 6.75 6.75-3.028 6.75-6.75 6.75z"/>
    </svg>
  ),
  splunk: (
    <svg className="w-6 h-6" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12.014 9.206l9.457 5.476-2.812 1.628-6.645-3.85-6.645 3.85-2.812-1.628 9.457-5.476zm0-4.412l9.457 5.476-2.812 1.628-6.645-3.85-6.645 3.85-2.812-1.628L12.014 4.794zm0 13.236l6.645-3.85 2.812 1.628-9.457 5.476-9.457-5.476 2.812-1.628 6.645 3.85z"/>
    </svg>
  ),
  datadog: (
    <svg className="w-6 h-6" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12.15 0C5.436 0 0 5.436 0 12.15s5.436 12.15 12.15 12.15 12.15-5.436 12.15-12.15S18.864 0 12.15 0zm5.85 17.1c-1.35.9-3.15 1.35-4.95 1.35-3.15 0-5.85-1.35-7.65-3.6l1.35-1.35c1.35 1.8 3.6 2.7 6.3 2.7 1.35 0 2.7-.45 3.6-.9l1.35 1.8zm.9-3.15c-.45.45-.9.9-1.35 1.35l-1.35-1.8c.45-.45.9-.9 1.35-1.35l1.35 1.8zm-2.25-2.25l-1.35-1.8c.45-.45.9-.9 1.35-1.35l1.35 1.8c-.45.45-.9.9-1.35 1.35z"/>
    </svg>
  ),
};

const BRAND_COLORS: Record<string, string> = {
  slack: 'bg-[#4A154B]',
  discord: 'bg-[#5865F2]',
  teams: 'bg-[#6264A7]',
  pagerduty: 'bg-[#06AC38]',
  webhook: 'bg-gradient-to-br from-blue-600 to-cyan-500',
  email: 'bg-gradient-to-br from-purple-600 to-pink-500',
  opsgenie: 'bg-[#172B4D]',
  jira: 'bg-[#0052CC]',
  servicenow: 'bg-[#62D84E]',
  splunk: 'bg-[#000000]',
  datadog: 'bg-[#632CA6]',
};

export default function IntegrationsPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [integrations, setIntegrations] = useState<Integration[]>([]);
  const [connectedIds, setConnectedIds] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  useEffect(() => {
    async function fetchIntegrations() {
      try {
        const res = await fetch('/api/integrations');
        if (res.ok) {
          const data = await res.json();
          setIntegrations(data.availableProviders || []);
          setConnectedIds((data.integrations || []).map((i: { provider: string }) => i.provider));
        }
      } catch (error) {
        console.error('Failed to fetch integrations:', error);
      } finally {
        setLoading(false);
      }
    }
    void fetchIntegrations();
  }, []);

  const filteredIntegrations = integrations.filter((int) => {
    const matchesSearch = int.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      int.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = !selectedCategory || int.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const categories = Array.from(new Set(integrations.map((i) => i.category)));

  const handleConnect = (integration: Integration) => {
    if (integration.comingSoon) return;

    // Navigate to settings for configuration
    router.push(`/app/settings?connect=${integration.id}`);
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="animate-pulse">
          <div className="h-8 w-48 bg-white/10 rounded mb-2" />
          <div className="h-4 w-96 bg-white/5 rounded" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-48 bg-white/5 rounded-md animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-lg font-bold text-white">Integrations</h1>
          <p className="text-slate-400 mt-1">Connect LatticeForge to your favorite tools</p>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <span className="text-slate-500">{connectedIds.length} connected</span>
          <span className="text-slate-600">•</span>
          <span className="text-slate-500">{integrations.length} available</span>
        </div>
      </div>

      {/* Search and Filter */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="text"
            placeholder="Search integrations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-3 bg-[rgba(18,18,26,0.7)] backdrop-blur-sm border border-white/[0.06] rounded-md text-white placeholder-slate-500 focus:border-blue-500/50 focus:outline-none transition-colors"
          />
        </div>
        <div className="flex gap-2 overflow-x-auto pb-1">
          <button
            onClick={() => setSelectedCategory(null)}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-md text-sm font-medium whitespace-nowrap transition-all min-h-[44px] ${
              !selectedCategory
                ? 'bg-gradient-to-r from-blue-600 to-cyan-500 text-white'
                : 'bg-[rgba(18,18,26,0.7)] text-slate-400 hover:text-white border border-white/[0.06]'
            }`}
          >
            <Filter className="w-4 h-4" />
            All
          </button>
          {categories.map((cat) => {
            const info = CATEGORY_INFO[cat] || { name: cat, icon: null, color: 'text-slate-400' };
            return (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`flex items-center gap-2 px-4 py-2.5 rounded-md text-sm font-medium whitespace-nowrap transition-all min-h-[44px] ${
                  selectedCategory === cat
                    ? 'bg-gradient-to-r from-blue-600 to-cyan-500 text-white'
                    : 'bg-[rgba(18,18,26,0.7)] text-slate-400 hover:text-white border border-white/[0.06]'
                }`}
              >
                {info.icon}
                {info.name}
              </button>
            );
          })}
        </div>
      </div>

      {/* Connected Integrations */}
      {connectedIds.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Check className="w-5 h-5 text-green-400" />
            Connected
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {integrations
              .filter((int) => connectedIds.includes(int.id))
              .map((integration) => (
                <IntegrationCard
                  key={integration.id}
                  integration={integration}
                  isConnected={true}
                  onConnect={() => handleConnect(integration)}
                />
              ))}
          </div>
        </section>
      )}

      {/* Available Integrations */}
      <section>
        <h2 className="text-lg font-semibold text-white mb-4">
          {connectedIds.length > 0 ? 'Available Integrations' : 'All Integrations'}
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredIntegrations
            .filter((int) => !connectedIds.includes(int.id))
            .map((integration) => (
              <IntegrationCard
                key={integration.id}
                integration={integration}
                isConnected={false}
                onConnect={() => handleConnect(integration)}
              />
            ))}
        </div>

        {filteredIntegrations.length === 0 && (
          <Card className="text-center py-12">
            <p className="text-slate-400">No integrations found matching your search.</p>
          </Card>
        )}
      </section>

      {/* Request Integration */}
      <Card className="text-center">
        <p className="text-slate-400 mb-4">Don&apos;t see what you need?</p>
        <a
          href="mailto:contact@crystallinelabs.io?subject=Integration%20Request"
          className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 font-medium"
        >
          Request an integration
          <ExternalLink className="w-4 h-4" />
        </a>
      </Card>
    </div>
  );
}

function IntegrationCard({
  integration,
  isConnected,
  onConnect,
}: {
  integration: Integration;
  isConnected: boolean;
  onConnect: () => void;
}) {
  const icon = INTEGRATION_ICONS[integration.icon] || <Webhook className="w-6 h-6" />;
  const bgColor = BRAND_COLORS[integration.icon] || 'bg-slate-700';
  const categoryInfo = CATEGORY_INFO[integration.category];

  return (
    <Card
     
      className={`relative overflow-hidden transition-all ${
        integration.comingSoon ? 'opacity-60' : 'hover:border-blue-500/30'
      }`}
    >
      {isConnected && (
        <div className="absolute top-3 right-3">
          <span className="flex items-center gap-1 text-xs text-green-400 bg-green-500/10 px-2 py-1 rounded-full">
            <Check className="w-3 h-3" />
            Connected
          </span>
        </div>
      )}

      {integration.comingSoon && (
        <div className="absolute top-3 right-3">
          <span className="flex items-center gap-1 text-xs text-slate-400 bg-white/5 px-2 py-1 rounded-full">
            <Clock className="w-3 h-3" />
            Coming Soon
          </span>
        </div>
      )}

      <div className="flex items-start gap-4">
        <div className={`w-12 h-12 rounded-md flex items-center justify-center text-white shrink-0 ${bgColor}`}>
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-white font-medium">{integration.name}</h3>
          <p className="text-sm text-slate-400 mt-1 line-clamp-2">{integration.description}</p>
          {categoryInfo && (
            <span className={`inline-flex items-center gap-1 text-xs mt-2 ${categoryInfo.color}`}>
              {categoryInfo.icon}
              {categoryInfo.name}
            </span>
          )}
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-white/[0.06]">
        {isConnected ? (
          <Button
            variant="secondary"
            size="sm"
            onClick={onConnect}
            className="w-full"
          >
            Manage
            <ArrowRight className="w-4 h-4 ml-2" />
          </Button>
        ) : integration.comingSoon ? (
          <Button
            variant="secondary"
            size="sm"
            disabled
            className="w-full"
          >
            Coming Soon
          </Button>
        ) : (
          <Button
            variant="primary"
            size="sm"
            onClick={onConnect}
            className="w-full"
          >
            Connect
            <ArrowRight className="w-4 h-4 ml-2" />
          </Button>
        )}
      </div>
    </Card>
  );
}
