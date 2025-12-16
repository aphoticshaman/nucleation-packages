'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  Radio,
  Signal,
  AlertTriangle,
  GitBranch,
  Network,
  Zap,
  Waves,
  Flame,
  BookOpen,
  Key,
  TrendingUp,
  Link as LinkIcon,
  Users,
  type LucideIcon,
} from 'lucide-react';
import type { UserProfile } from '@/lib/auth';

interface EnterpriseNavProps {
  user: UserProfile;
  org: {
    name: string;
    plan: string;
    api_calls_used: number;
    api_calls_limit: number;
  } | null;
}

interface NavItem {
  href: string;
  label: string;
  fullLabel?: string;
  icon: LucideIcon;
}

const navItems: NavItem[] = [
  { href: '/dashboard', label: 'Overview', icon: LayoutDashboard },
  { href: '/dashboard/intelligence', label: 'Intelligence', icon: Radio },
  { href: '/dashboard/signals', label: 'Signals', fullLabel: 'Live Signals', icon: Signal },
  { href: '/dashboard/alerts', label: 'Alerts', fullLabel: 'Risk Alerts', icon: AlertTriangle },
  { href: '/dashboard/cascades', label: 'Cascades', fullLabel: 'Cascade Analysis', icon: GitBranch },
  { href: '/dashboard/causal', label: 'Causal Graph', icon: Network },
  { href: '/dashboard/regimes', label: 'Regimes', fullLabel: 'Regime Detection', icon: Zap },
  { href: '/dashboard/phase-dynamics', label: 'Phase', fullLabel: 'Phase Dynamics', icon: Waves },
  { href: '/dashboard/prometheus', label: 'PROMETHEUS', fullLabel: 'Knowledge Extraction', icon: Flame },
  { href: '/dashboard/doctrine', label: 'Doctrine', fullLabel: 'Doctrine Registry', icon: BookOpen },
  { href: '/dashboard/api-keys', label: 'API Keys', icon: Key },
  { href: '/dashboard/usage', label: 'Usage', fullLabel: 'Usage & Analytics', icon: TrendingUp },
  { href: '/dashboard/webhooks', label: 'Webhooks', icon: LinkIcon },
  { href: '/dashboard/team', label: 'Team', icon: Users },
];

const docsItems = [
  { href: '/docs/api', label: 'API Reference', external: true },
  { href: '/docs/streams', label: 'Data Streams', external: true },
  { href: '/docs/sdk', label: 'SDK Guide', external: true },
];

export default function EnterpriseNav({ user, org }: EnterpriseNavProps) {
  const pathname = usePathname();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const usagePercent = org ? (org.api_calls_used / org.api_calls_limit) * 100 : 0;

  return (
    <>
      {/* Mobile top bar */}
      <div className="fixed top-0 left-0 right-0 h-14 bg-slate-900 border-b border-slate-800 flex items-center justify-between px-4 z-50 lg:hidden">
        <button
          onClick={() => setSidebarOpen(true)}
          className="p-2 text-slate-400 hover:text-white"
          aria-label="Open menu"
        >
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 6h16M4 12h16M4 18h16"
            />
          </svg>
        </button>
        <div className="flex items-center gap-2">
          <span className="font-semibold text-white text-sm">LatticeForge</span>
          {org && (
            <span className="px-2 py-0.5 bg-blue-600 rounded text-[10px] text-white uppercase">
              {org.plan}
            </span>
          )}
        </div>
        <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-white text-sm font-medium">
          {user.full_name?.[0] || user.email[0].toUpperCase()}
        </div>
      </div>

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <nav
        className={`
        fixed left-0 top-0 h-full bg-slate-900 border-r border-slate-800 flex flex-col z-50
        w-64
        transform transition-transform duration-200 ease-in-out
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}
      >
        {/* Header */}
        <div className="p-4 border-b border-slate-800 flex items-center justify-between">
          <div>
            <h1 className="text-base font-semibold text-white">LatticeForge</h1>
            <p className="text-[10px] text-slate-500 mt-0.5 uppercase tracking-wide">
              Enterprise
            </p>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="lg:hidden p-2 text-slate-400 hover:text-white"
            aria-label="Close menu"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Organization info */}
        {org && (
          <div className="p-4 border-b border-slate-800">
            <p className="text-sm font-medium text-slate-200 truncate">{org.name}</p>
            <div className="flex items-center gap-2 mt-2">
              <span className="px-2 py-0.5 bg-blue-600/20 border border-blue-600/30 rounded text-[10px] text-blue-400 uppercase">
                {org.plan}
              </span>
            </div>
            {/* Usage bar */}
            <div className="mt-3">
              <div className="flex justify-between text-[10px] text-slate-500 mb-1">
                <span>API Usage</span>
                <span>
                  {org.api_calls_used.toLocaleString()} / {org.api_calls_limit.toLocaleString()}
                </span>
              </div>
              <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all ${
                    usagePercent > 80 ? 'bg-amber-500' : 'bg-blue-500'
                  }`}
                  style={{ width: `${Math.min(usagePercent, 100)}%` }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Nav items */}
        <div className="flex-1 overflow-y-auto p-3">
          <div className="space-y-0.5">
            {navItems.map((item) => {
              const isActive =
                pathname === item.href ||
                (item.href !== '/dashboard' && pathname.startsWith(item.href));
              const Icon = item.icon;

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setSidebarOpen(false)}
                  className={`flex items-center gap-3 px-3 py-2 rounded-md transition-colors text-sm ${
                    isActive
                      ? 'bg-blue-600 text-white'
                      : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                  }`}
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  <span>{item.fullLabel || item.label}</span>
                </Link>
              );
            })}
          </div>

          {/* Documentation links */}
          <div className="mt-6">
            <p className="px-3 text-[10px] text-slate-600 uppercase tracking-wide mb-2">
              Documentation
            </p>
            <div className="space-y-0.5">
              {docsItems.map((item) => (
                <a
                  key={item.href}
                  href={item.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-3 px-3 py-2 text-sm text-slate-500 hover:text-slate-300"
                >
                  <span>{item.label}</span>
                  <span className="text-[10px] text-slate-600">↗</span>
                </a>
              ))}
            </div>
          </div>
        </div>

        {/* User section */}
        <div className="p-4 border-t border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-full bg-slate-700 flex items-center justify-center text-white font-medium text-sm">
              {user.full_name?.[0] || user.email[0].toUpperCase()}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm text-slate-200 truncate">
                {user.full_name || user.email}
              </p>
              <p className="text-[10px] text-slate-500">Engineer</p>
            </div>
          </div>
        </div>
      </nav>
    </>
  );
}
