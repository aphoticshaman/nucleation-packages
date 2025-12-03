'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import type { UserProfile } from '@/lib/auth';
import ViewAsSwitcher from '@/components/admin/ViewAsSwitcher';

interface AdminNavProps {
  user: UserProfile;
}

const navItems = [
  { href: '/admin', label: 'Overview', icon: '📊' },
  { href: '/admin/builder', label: 'Dashboard Builder', icon: '🎨' },
  { href: '/admin/customers', label: 'Customers', icon: '👥' },
  { href: '/admin/billing', label: 'Billing', icon: '💳' },
  { href: '/admin/analytics', label: 'Analytics', icon: '📈' },
  { href: '/admin/pipelines', label: 'Data Pipelines', icon: '🔄' },
  { href: '/admin/models', label: 'ML Models', icon: '🤖' },
  { href: '/admin/compliance', label: 'Compliance', icon: '🛡️' },
  { href: '/admin/config', label: 'System Config', icon: '⚙️' },
];

const quickLinks = [
  { href: '/app', label: 'Consumer App', icon: '👤' },
  { href: '/app/briefings', label: 'Briefings', icon: '📡' },
  { href: '/app/packages', label: 'Packages', icon: '📦' },
  { href: '/app/signals', label: 'Signals', icon: '📈' },
];

export default function AdminNav({ user }: AdminNavProps) {
  const pathname = usePathname();

  return (
    <nav className="fixed left-0 top-0 h-full w-64 bg-slate-900 border-r border-slate-800">
      {/* Header */}
      <div className="p-6 border-b border-slate-800">
        <h1 className="text-xl font-bold text-white">LatticeForge</h1>
        <p className="text-xs text-slate-500 mt-1">Admin Console</p>
      </div>

      {/* Nav items */}
      <div className="p-4 space-y-2 overflow-y-auto max-h-[calc(100vh-400px)]">
        {navItems.map((item) => {
          const isActive =
            pathname === item.href || (item.href !== '/admin' && pathname.startsWith(item.href));

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

      {/* Quick Links - Jump to consumer views */}
      <div className="px-4 py-3 border-t border-slate-800">
        <p className="text-xs text-slate-500 uppercase tracking-wider mb-2 px-4">Quick Jump</p>
        <div className="space-y-1">
          {quickLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="flex items-center gap-2 px-4 py-2 text-sm text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
            >
              <span>{link.icon}</span>
              <span>{link.label}</span>
            </Link>
          ))}
        </div>
      </div>

      {/* View As Switcher */}
      <div className="absolute bottom-24 left-0 right-0 px-4">
        <ViewAsSwitcher />
      </div>

      {/* User section */}
      <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-slate-800">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-blue-600 flex items-center justify-center text-white font-bold">
            {user.full_name?.[0] || user.email[0].toUpperCase()}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm text-white truncate">{user.full_name || user.email}</p>
            <p className="text-xs text-slate-500">Admin</p>
          </div>
        </div>
      </div>
    </nav>
  );
}
