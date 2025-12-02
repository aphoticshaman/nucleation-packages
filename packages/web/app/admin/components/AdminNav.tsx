'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import type { UserProfile } from '@/lib/auth';

interface AdminNavProps {
  user: UserProfile;
}

const navItems = [
  { href: '/admin', label: 'Overview', icon: '📊' },
  { href: '/admin/customers', label: 'Customers', icon: '👥' },
  { href: '/admin/billing', label: 'Billing', icon: '💳' },
  { href: '/admin/config', label: 'System Config', icon: '⚙️' },
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
      <div className="p-4 space-y-2">
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
