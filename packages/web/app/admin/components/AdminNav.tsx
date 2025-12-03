'use client';

import { useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { usePathname } from 'next/navigation';
import { BookOpen } from 'lucide-react';
import type { UserProfile } from '@/lib/auth';
import ViewAsSwitcher from '@/components/admin/ViewAsSwitcher';
import Glossary from '@/components/Glossary';

interface AdminNavProps {
  user: UserProfile;
}

const navItems = [
  { href: '/admin', label: 'Overview', icon: '📊' },
  { href: '/admin/builder', label: 'Dashboard Builder', icon: '🎨' },
  { href: '/admin/training', label: 'Training Data', icon: '🧠' },
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
  const [showGlossary, setShowGlossary] = useState(false);

  return (
    <>
    <nav className="fixed left-0 top-0 h-full w-72 bg-[rgba(10,10,15,0.95)] backdrop-blur-xl border-r border-white/[0.06] z-50">
      {/* Header */}
      <div className="p-6 border-b border-white/[0.06]">
        <div className="flex items-center gap-3">
          <Image
            src="/images/brand/monogram.png"
            alt="LatticeForge"
            width={36}
            height={36}
            className="w-9 h-9"
          />
          <div>
            <h1 className="text-lg font-bold text-white">LatticeForge</h1>
            <p className="text-[10px] text-orange-400 uppercase tracking-wider font-medium">Admin Console</p>
          </div>
        </div>
      </div>

      {/* Nav items */}
      <div className="p-3 space-y-1 overflow-y-auto max-h-[calc(100vh-380px)]">
        {navItems.map((item) => {
          const isActive =
            pathname === item.href || (item.href !== '/admin' && pathname.startsWith(item.href));

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-150 ${
                isActive
                  ? 'bg-blue-600/90 text-white shadow-[0_0_20px_rgba(59,130,246,0.25)]'
                  : 'text-slate-400 hover:bg-white/[0.05] hover:text-white'
              }`}
            >
              <span className="text-lg">{item.icon}</span>
              <span className="font-medium">{item.label}</span>
              {isActive && <span className="ml-auto w-1.5 h-1.5 rounded-full bg-white" />}
            </Link>
          );
        })}
      </div>

      {/* Quick Links */}
      <div className="px-3 py-3 border-t border-white/[0.06]">
        <p className="text-[10px] text-slate-500 uppercase tracking-wider font-medium mb-2 px-4">Quick Jump</p>
        <div className="space-y-0.5">
          {quickLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="flex items-center gap-2 px-4 py-2.5 text-sm text-slate-400 hover:text-white hover:bg-white/[0.03] rounded-lg transition-all"
            >
              <span>{link.icon}</span>
              <span>{link.label}</span>
              <svg className="w-3 h-3 ml-auto opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </Link>
          ))}
        </div>
        {/* Glossary Button */}
        <button
          onClick={() => setShowGlossary(true)}
          className="flex items-center gap-2 w-full px-4 py-2.5 mt-2 text-sm text-slate-400 hover:text-white hover:bg-white/[0.03] rounded-lg transition-all"
        >
          <BookOpen className="w-4 h-4" />
          <span>Terminology Reference</span>
        </button>
      </div>

      {/* View As Switcher */}
      <div className="absolute bottom-28 left-0 right-0 px-3">
        <div className="bg-black/20 rounded-xl p-3 border border-white/[0.06]">
          <ViewAsSwitcher />
        </div>
      </div>

      {/* User section */}
      <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-white/[0.06] bg-[rgba(10,10,15,0.8)]">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center text-white font-bold shadow-lg">
            {user.full_name?.[0] || user.email[0].toUpperCase()}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm text-white font-medium truncate">{user.full_name || user.email}</p>
            <p className="text-[10px] text-orange-400 uppercase tracking-wider font-medium">Admin</p>
          </div>
          <Link
            href="/api/auth/signout"
            className="p-2 text-slate-500 hover:text-white hover:bg-white/[0.05] rounded-lg transition-all"
            title="Sign out"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
            </svg>
          </Link>
        </div>
      </div>
    </nav>

    {/* Glossary Modal */}
    <Glossary
      isOpen={showGlossary}
      onClose={() => setShowGlossary(false)}
      skillLevel="detailed"
    />
    </>
  );
}
