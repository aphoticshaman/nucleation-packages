'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
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

const navItems = [
  { href: '/dashboard', label: 'Overview', icon: '📊' },
  { href: '/dashboard/intelligence', label: 'Intelligence', icon: '📡' },
  { href: '/dashboard/signals', label: 'Signals', fullLabel: 'Live Signals', icon: '📶' },
  { href: '/dashboard/alerts', label: 'Alerts', fullLabel: 'Risk Alerts', icon: '🚨' },
  { href: '/dashboard/cascades', label: 'Cascades', fullLabel: 'Cascade Analysis', icon: '🌀' },
  { href: '/dashboard/causal', label: 'Causal Graph', icon: '🕸️' },
  { href: '/dashboard/regimes', label: 'Regimes', fullLabel: 'Regime Detection', icon: '⚡' },
  { href: '/dashboard/phase-dynamics', label: 'Phase', fullLabel: 'Phase Dynamics', icon: '🌊' },
  { href: '/dashboard/prometheus', label: 'PROMETHEUS', fullLabel: 'Knowledge Extraction', icon: '🔥' },
  { href: '/dashboard/doctrine', label: 'Doctrine', fullLabel: 'Doctrine Registry', icon: '📜' },
  { href: '/dashboard/api-keys', label: 'API Keys', icon: '🔑' },
  { href: '/dashboard/usage', label: 'Usage', fullLabel: 'Usage & Analytics', icon: '📈' },
  { href: '/dashboard/webhooks', label: 'Webhooks', icon: '🔗' },
  { href: '/dashboard/team', label: 'Team', icon: '👥' },
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
          <span className="font-bold text-white">LatticeForge</span>
          {org && (
            <span className="px-2 py-0.5 bg-blue-600 rounded text-xs text-white uppercase">
              {org.plan}
            </span>
          )}
        </div>
        <div className="w-8 h-8 rounded-full bg-emerald-600 flex items-center justify-center text-white text-sm font-bold">
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

      {/* Sidebar - slides in on mobile, fixed on desktop */}
      <nav
        className={`
        fixed left-0 top-0 h-full bg-slate-900 border-r border-slate-800 flex flex-col z-50
        w-72 lg:w-64 2xl:w-72
        transform transition-transform duration-200 ease-in-out
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}
      >
        {/* Header */}
        <div className="p-4 lg:p-6 border-b border-slate-800 flex items-center justify-between">
          <div>
            <h1 className="text-lg lg:text-xl 2xl:text-2xl font-bold text-white">LatticeForge</h1>
            <p className="text-xs 2xl:text-sm text-slate-500 mt-0.5 lg:mt-1">
              Enterprise Dashboard
            </p>
          </div>
          {/* Mobile close button */}
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
            <p className="text-sm 2xl:text-base font-medium text-white truncate">{org.name}</p>
            <div className="flex items-center gap-2 mt-2">
              <span className="px-2 py-0.5 bg-blue-600 rounded text-xs 2xl:text-sm text-white uppercase">
                {org.plan}
              </span>
            </div>
            {/* Usage bar */}
            <div className="mt-3">
              <div className="flex justify-between text-xs 2xl:text-sm text-slate-400 mb-1">
                <span>API Usage</span>
                <span>
                  {org.api_calls_used.toLocaleString()} / {org.api_calls_limit.toLocaleString()}
                </span>
              </div>
              <div className="h-1.5 2xl:h-2 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all ${
                    usagePercent > 80 ? 'bg-yellow-500' : 'bg-blue-500'
                  }`}
                  style={{ width: `${Math.min(usagePercent, 100)}%` }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Nav items */}
        <div className="flex-1 overflow-y-auto p-3 lg:p-4">
          <div className="space-y-1 lg:space-y-2">
            {navItems.map((item) => {
              const isActive =
                pathname === item.href ||
                (item.href !== '/dashboard' && pathname.startsWith(item.href));

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setSidebarOpen(false)}
                  className={`flex items-center gap-3 px-3 lg:px-4 py-2.5 lg:py-3 2xl:py-3.5 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-blue-600 text-white'
                      : 'text-slate-400 hover:bg-slate-800 hover:text-white'
                  }`}
                >
                  <span className="text-lg 2xl:text-xl">{item.icon}</span>
                  <span className="text-sm 2xl:text-base">{item.fullLabel || item.label}</span>
                </Link>
              );
            })}
          </div>

          {/* Documentation links */}
          <div className="mt-6 lg:mt-8">
            <p className="px-3 lg:px-4 text-xs 2xl:text-sm text-slate-500 uppercase tracking-wide mb-2">
              Documentation
            </p>
            <div className="space-y-1">
              {docsItems.map((item) => (
                <a
                  key={item.href}
                  href={item.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-3 px-3 lg:px-4 py-2 text-sm 2xl:text-base text-slate-400 hover:text-white"
                >
                  <span>{item.label}</span>
                  <span className="text-xs">↗</span>
                </a>
              ))}
            </div>
          </div>
        </div>

        {/* User section */}
        <div className="p-4 border-t border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 lg:w-10 lg:h-10 2xl:w-12 2xl:h-12 rounded-full bg-emerald-600 flex items-center justify-center text-white font-bold text-sm 2xl:text-base">
              {user.full_name?.[0] || user.email[0].toUpperCase()}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm 2xl:text-base text-white truncate">
                {user.full_name || user.email}
              </p>
              <p className="text-xs 2xl:text-sm text-slate-500">Engineer</p>
            </div>
          </div>
        </div>
      </nav>
    </>
  );
}
