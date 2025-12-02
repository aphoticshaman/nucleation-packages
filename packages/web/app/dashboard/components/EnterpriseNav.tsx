'use client';

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
  { href: '/dashboard/api-keys', label: 'API Keys', icon: '🔑' },
  { href: '/dashboard/usage', label: 'Usage & Analytics', icon: '📈' },
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
  const usagePercent = org ? (org.api_calls_used / org.api_calls_limit) * 100 : 0;

  return (
    <nav className="fixed left-0 top-0 h-full w-64 bg-slate-900 border-r border-slate-800 flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-slate-800">
        <h1 className="text-xl font-bold text-white">LatticeForge</h1>
        <p className="text-xs text-slate-500 mt-1">Enterprise Dashboard</p>
      </div>

      {/* Organization info */}
      {org && (
        <div className="p-4 border-b border-slate-800">
          <p className="text-sm font-medium text-white">{org.name}</p>
          <div className="flex items-center gap-2 mt-2">
            <span className="px-2 py-0.5 bg-blue-600 rounded text-xs text-white uppercase">
              {org.plan}
            </span>
          </div>
          {/* Usage bar */}
          <div className="mt-3">
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>API Usage</span>
              <span>{org.api_calls_used.toLocaleString()} / {org.api_calls_limit.toLocaleString()}</span>
            </div>
            <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
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
      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-2">
          {navItems.map((item) => {
            const isActive = pathname === item.href ||
              (item.href !== '/dashboard' && pathname.startsWith(item.href));

            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-400 hover:bg-slate-800 hover:text-white'
                }`}
              >
                <span>{item.icon}</span>
                <span>{item.label}</span>
              </Link>
            );
          })}
        </div>

        {/* Documentation links */}
        <div className="mt-8">
          <p className="px-4 text-xs text-slate-500 uppercase tracking-wide mb-2">
            Documentation
          </p>
          <div className="space-y-1">
            {docsItems.map((item) => (
              <a
                key={item.href}
                href={item.href}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-3 px-4 py-2 text-sm text-slate-400 hover:text-white"
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
          <div className="w-10 h-10 rounded-full bg-emerald-600 flex items-center justify-center text-white font-bold">
            {user.full_name?.[0] || user.email[0].toUpperCase()}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm text-white truncate">{user.full_name || user.email}</p>
            <p className="text-xs text-slate-500">Engineer</p>
          </div>
        </div>
      </div>
    </nav>
  );
}
