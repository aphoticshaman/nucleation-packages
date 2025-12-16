'use client';

import { useState } from 'react';
import {
  Sparkles,
  Bug,
  Wrench,
  Shield,
  Zap,
  Bell,
  ChevronDown,
  ExternalLink,
  Rocket,
} from 'lucide-react';
import { Card } from '@/components/ui';

interface ChangelogEntry {
  version: string;
  date: string;
  title: string;
  description?: string;
  changes: {
    type: 'feature' | 'improvement' | 'fix' | 'security' | 'performance';
    text: string;
  }[];
  breaking?: boolean;
}

const CHANGELOG: ChangelogEntry[] = [
  {
    version: '2.4.0',
    date: '2024-12-04',
    title: 'Integrations Marketplace',
    description: 'Connect LatticeForge to your favorite tools with our new integrations hub.',
    changes: [
      { type: 'feature', text: 'New Integrations marketplace with 11 providers' },
      { type: 'feature', text: 'Discord integration with webhook support' },
      { type: 'feature', text: 'PagerDuty integration for incident management' },
      { type: 'feature', text: 'Microsoft Teams webhook integration' },
      { type: 'feature', text: 'Custom webhook support with HMAC signatures' },
      { type: 'feature', text: 'Zapier templates for automation workflows' },
      { type: 'feature', text: 'API documentation page with code examples' },
      { type: 'feature', text: 'Keyboard shortcuts help modal' },
      { type: 'improvement', text: 'Enhanced glass-morphism UI across all pages' },
      { type: 'security', text: 'Fixed XSS vulnerabilities in export functions' },
    ],
  },
  {
    version: '2.3.0',
    date: '2024-11-28',
    title: 'Intelligence Packages',
    description: 'Build and export comprehensive intelligence packages with multiple formats.',
    changes: [
      { type: 'feature', text: 'PackageBuilder with audience-specific presets' },
      { type: 'feature', text: 'Export to HTML, PDF, DOCX, and PPTX formats' },
      { type: 'feature', text: 'Section reordering via drag and drop' },
      { type: 'feature', text: 'Live preview with print optimization' },
      { type: 'improvement', text: 'Executive briefing templates' },
      { type: 'performance', text: '40% faster package generation' },
    ],
  },
  {
    version: '2.2.0',
    date: '2024-11-15',
    title: 'Cascade Simulation Engine',
    description: 'Model how events propagate across interconnected nations.',
    changes: [
      { type: 'feature', text: 'Interactive cascade simulation with SIR model' },
      { type: 'feature', text: '8 cascade signature types detection' },
      { type: 'feature', text: 'Network graph visualization' },
      { type: 'feature', text: 'Timeline playback controls' },
      { type: 'improvement', text: 'Improved nation relationship modeling' },
      { type: 'fix', text: 'Fixed cascade peak calculation bug' },
    ],
  },
  {
    version: '2.1.0',
    date: '2024-11-01',
    title: 'Real-time Alerts',
    description: 'Never miss critical intelligence with our new alert system.',
    changes: [
      { type: 'feature', text: 'Slack integration with OAuth flow' },
      { type: 'feature', text: 'Customizable alert thresholds' },
      { type: 'feature', text: 'Daily digest summaries' },
      { type: 'feature', text: 'Severity-based filtering' },
      { type: 'improvement', text: 'Email notification templates' },
      { type: 'security', text: 'Enhanced webhook signature verification' },
    ],
  },
  {
    version: '2.0.0',
    date: '2024-10-15',
    title: 'LatticeForge Platform Launch',
    description: 'Major platform overhaul with new features and improved architecture.',
    breaking: true,
    changes: [
      { type: 'feature', text: 'New Next.js 15 architecture' },
      { type: 'feature', text: '3D globe visualization with Three.js' },
      { type: 'feature', text: 'AI-powered executive summaries' },
      { type: 'feature', text: 'Multi-tier pricing (Free, Starter, Pro, Enterprise)' },
      { type: 'feature', text: 'Team collaboration features' },
      { type: 'improvement', text: 'Complete UI redesign with glass-morphism' },
      { type: 'performance', text: '60% faster page loads' },
      { type: 'security', text: 'Row-level security with Supabase' },
    ],
  },
  {
    version: '1.5.0',
    date: '2024-09-01',
    title: 'Risk Analytics',
    changes: [
      { type: 'feature', text: 'Nation risk scoring algorithm' },
      { type: 'feature', text: 'Regime classification (5 phases)' },
      { type: 'feature', text: 'Historical trend analysis' },
      { type: 'improvement', text: 'Data freshness indicators' },
      { type: 'fix', text: 'Fixed risk compounding calculation' },
    ],
  },
  {
    version: '1.4.0',
    date: '2024-08-15',
    title: 'Data Sources Expansion',
    changes: [
      { type: 'feature', text: 'GDELT integration (185+ countries)' },
      { type: 'feature', text: 'World Bank economic indicators' },
      { type: 'feature', text: 'FRED economic data' },
      { type: 'feature', text: 'Treasury yield signals' },
      { type: 'improvement', text: 'Multi-source signal fusion' },
    ],
  },
];

const TYPE_CONFIG: Record<string, { icon: React.ReactNode; color: string; bg: string; label: string }> = {
  feature: {
    icon: <Sparkles className="w-4 h-4" />,
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
    label: 'New',
  },
  improvement: {
    icon: <Wrench className="w-4 h-4" />,
    color: 'text-blue-400',
    bg: 'bg-blue-500/10',
    label: 'Improved',
  },
  fix: {
    icon: <Bug className="w-4 h-4" />,
    color: 'text-amber-400',
    bg: 'bg-amber-500/10',
    label: 'Fixed',
  },
  security: {
    icon: <Shield className="w-4 h-4" />,
    color: 'text-red-400',
    bg: 'bg-red-500/10',
    label: 'Security',
  },
  performance: {
    icon: <Zap className="w-4 h-4" />,
    color: 'text-purple-400',
    bg: 'bg-purple-500/10',
    label: 'Perf',
  },
};

export default function ChangelogPage() {
  const [expandedVersions, setExpandedVersions] = useState<string[]>([CHANGELOG[0].version]);

  const toggleVersion = (version: string) => {
    setExpandedVersions((prev) =>
      prev.includes(version)
        ? prev.filter((v) => v !== version)
        : [...prev, version]
    );
  };

  const latestVersion = CHANGELOG[0];

  return (
    <div className="space-y-6 max-w-3xl mx-auto">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-600 to-pink-500 flex items-center justify-center">
            <Rocket className="w-5 h-5 text-white" />
          </div>
          <h1 className="text-lg font-bold text-white">What&apos;s New</h1>
        </div>
        <p className="text-slate-400">
          Stay up to date with the latest features and improvements
        </p>
      </div>

      {/* Latest version highlight */}
      <Card className="relative overflow-hidden">
        <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-blue-500/20 to-transparent rounded-full -translate-y-1/2 translate-x-1/2" />
        <div className="relative">
          <div className="flex items-center gap-2 mb-2">
            <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs font-medium rounded-full">
              Latest
            </span>
            <span className="text-slate-500 text-sm">v{latestVersion.version}</span>
          </div>
          <h2 className="text-xl font-bold text-white mb-2">{latestVersion.title}</h2>
          {latestVersion.description && (
            <p className="text-slate-400 mb-4">{latestVersion.description}</p>
          )}
          <div className="flex flex-wrap gap-2">
            {latestVersion.changes.slice(0, 4).map((change, idx) => {
              const config = TYPE_CONFIG[change.type];
              return (
                <span
                  key={idx}
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium ${config.bg} ${config.color}`}
                >
                  {config.icon}
                  {change.text.length > 40 ? change.text.slice(0, 40) + '...' : change.text}
                </span>
              );
            })}
            {latestVersion.changes.length > 4 && (
              <span className="text-slate-500 text-xs py-1">
                +{latestVersion.changes.length - 4} more
              </span>
            )}
          </div>
        </div>
      </Card>

      {/* Subscribe */}
      <Card className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-3">
          <Bell className="w-5 h-5 text-blue-400" />
          <div>
            <p className="text-white font-medium">Get notified</p>
            <p className="text-sm text-slate-400">Subscribe to product updates</p>
          </div>
        </div>
        <a
          href="/app/settings"
          className="text-blue-400 hover:text-blue-300 text-sm font-medium flex items-center gap-1"
        >
          Manage notifications
          <ExternalLink className="w-3 h-3" />
        </a>
      </Card>

      {/* All releases */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-4">All Releases</h2>
        <div className="space-y-4">
          {CHANGELOG.map((entry) => (
            <Card key={entry.version}>
              <button
                onClick={() => toggleVersion(entry.version)}
                className="w-full flex items-center justify-between gap-4"
              >
                <div className="flex items-center gap-3">
                  <span className="text-white font-mono font-medium">v{entry.version}</span>
                  {entry.breaking && (
                    <span className="px-2 py-0.5 bg-red-500/20 text-red-400 text-xs font-medium rounded-full">
                      Breaking
                    </span>
                  )}
                  <span className="text-slate-400">{entry.title}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-slate-500 text-sm">{formatDate(entry.date)}</span>
                  <ChevronDown
                    className={`w-4 h-4 text-slate-500 transition-transform ${
                      expandedVersions.includes(entry.version) ? 'rotate-180' : ''
                    }`}
                  />
                </div>
              </button>

              {expandedVersions.includes(entry.version) && (
                <div className="mt-4 pt-4 border-t border-white/[0.06]">
                  {entry.description && (
                    <p className="text-slate-400 mb-4">{entry.description}</p>
                  )}
                  <div className="space-y-2">
                    {entry.changes.map((change, idx) => {
                      const config = TYPE_CONFIG[change.type];
                      return (
                        <div
                          key={idx}
                          className="flex items-start gap-3 p-2 rounded-lg hover:bg-white/[0.02]"
                        >
                          <span
                            className={`shrink-0 w-6 h-6 rounded flex items-center justify-center ${config.bg} ${config.color}`}
                          >
                            {config.icon}
                          </span>
                          <span className="text-slate-300 text-sm">{change.text}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </Card>
          ))}
        </div>
      </div>

      {/* Footer */}
      <Card className="text-center">
        <p className="text-slate-400 mb-2">Looking for older releases?</p>
        <a
          href="https://github.com/latticeforge/changelog"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 font-medium"
        >
          View full changelog on GitHub
          <ExternalLink className="w-4 h-4" />
        </a>
      </Card>
    </div>
  );
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}
