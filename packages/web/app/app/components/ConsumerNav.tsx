'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import type { UserProfile } from '@/lib/auth';

interface ConsumerNavProps {
  user: UserProfile;
}

const navItems = [
  { href: '/app', label: 'Explore' },
  { href: '/app/saved', label: 'My Simulations' },
  { href: '/app/settings', label: 'Settings' },
];

export default function ConsumerNav({ user }: ConsumerNavProps) {
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 left-0 right-0 h-16 bg-slate-900 border-b border-slate-800 z-50">
      <div className="max-w-7xl mx-auto px-4 h-full flex items-center justify-between">
        {/* Logo */}
        <Link href="/app" className="flex items-center gap-2">
          <span className="text-xl font-bold text-white">LatticeForge</span>
        </Link>

        {/* Nav links */}
        <div className="flex items-center gap-1">
          {navItems.map((item) => {
            const isActive = pathname === item.href ||
              (item.href !== '/app' && pathname.startsWith(item.href));

            return (
              <Link
                key={item.href}
                href={item.href}
                className={`px-4 py-2 rounded-lg text-sm transition-colors ${
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800'
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </div>

        {/* User menu */}
        <div className="flex items-center gap-4">
          {/* Upgrade CTA */}
          <a
            href="/pricing"
            className="text-sm text-blue-400 hover:text-blue-300 hidden sm:block"
          >
            Upgrade to Enterprise
          </a>

          {/* User avatar */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-white text-sm font-medium">
              {user.full_name?.[0] || user.email[0].toUpperCase()}
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}
