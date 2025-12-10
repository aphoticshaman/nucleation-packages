'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { usePathname } from 'next/navigation';
import { BookOpen, Menu, X } from 'lucide-react';
import type { UserProfile } from '@/lib/auth';
import ViewAsSwitcher from '@/components/admin/ViewAsSwitcher';
import Glossary from '@/components/Glossary';
import ExecBriefButton from '@/components/ExecBriefButton';

interface AdminNavProps {
  user: UserProfile;
}

const navItems = [
  { href: '/admin', label: 'Overview', icon: '📊' },
  { href: '/admin/feedback', label: 'Feedback', icon: '💬' },
  { href: '/admin/users', label: 'Users', icon: '👤' },
  { href: '/admin/builder', label: 'Builder', icon: '🎨' },
  { href: '/admin/training', label: 'Training', icon: '🧠' },
  { href: '/admin/customers', label: 'Customers', icon: '👥' },
  { href: '/admin/billing', label: 'Billing', icon: '💳' },
  { href: '/admin/analytics', label: 'Analytics', icon: '📈' },
  { href: '/admin/health', label: 'API Health', icon: '🩺' },
  { href: '/admin/pipelines', label: 'Pipelines', icon: '🔄' },
  { href: '/admin/models', label: 'ML Models', icon: '🤖' },
  { href: '/admin/compliance', label: 'Compliance', icon: '🛡️' },
  { href: '/admin/guardian', label: 'Guardian', icon: '🔮' },
  { href: '/admin/config', label: 'Config', icon: '⚙️' },
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
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Lock body scroll when mobile menu is open
  useEffect(() => {
    document.body.style.overflow = mobileMenuOpen ? 'hidden' : '';
    return () => { document.body.style.overflow = ''; };
  }, [mobileMenuOpen]);

  // Close menu on route change
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [pathname]);

  // Get current page name for mobile header
  const currentPage = navItems.find(
    item => pathname === item.href || (item.href !== '/admin' && pathname.startsWith(item.href))
  );

  const SidebarContent = ({ onNavClick }: { onNavClick?: () => void }) => (
    <>
      {/* Header */}
      <div className="p-4 lg:p-6 border-b border-white/[0.06]">
        <div className="flex items-center gap-3">
          <Image
            src="/images/brand/monogram.png"
            alt="LatticeForge"
            width={36}
            height={36}
            className="w-8 h-8 lg:w-9 lg:h-9"
          />
          <div>
            <h1 className="text-base lg:text-lg font-bold text-white">LatticeForge</h1>
            <p className="text-[10px] text-orange-400 uppercase tracking-wider font-medium">Admin Console</p>
          </div>
        </div>
        {/* Exec Brief Button */}
        <div className="mt-4">
          <ExecBriefButton variant="compact" className="w-full justify-center" />
        </div>
      </div>

      {/* Nav items - scrollable */}
      <div className="flex-1 overflow-y-auto p-2 lg:p-3 space-y-0.5 lg:space-y-1">
        {navItems.map((item) => {
          const isActive =
            pathname === item.href || (item.href !== '/admin' && pathname.startsWith(item.href));

          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={onNavClick}
              className={`flex items-center gap-3 px-3 lg:px-4 py-2.5 lg:py-3 rounded-xl transition-all duration-150 touch-manipulation ${
                isActive
                  ? 'bg-blue-600/90 text-white shadow-[0_0_20px_rgba(59,130,246,0.25)]'
                  : 'text-slate-400 hover:bg-white/[0.05] hover:text-white active:bg-white/[0.08]'
              }`}
            >
              <span className="text-lg">{item.icon}</span>
              <span className="font-medium text-sm lg:text-base">{item.label}</span>
              {isActive && <span className="ml-auto w-1.5 h-1.5 rounded-full bg-white" />}
            </Link>
          );
        })}
      </div>

      {/* Quick Links */}
      <div className="px-2 lg:px-3 py-3 border-t border-white/[0.06]">
        <p className="text-[10px] text-slate-500 uppercase tracking-wider font-medium mb-2 px-3 lg:px-4">Quick Jump</p>
        <div className="space-y-0.5">
          {quickLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              onClick={onNavClick}
              className="flex items-center gap-2 px-3 lg:px-4 py-2 lg:py-2.5 text-sm text-slate-400 hover:text-white hover:bg-white/[0.03] active:bg-white/[0.06] rounded-lg transition-all touch-manipulation"
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
          onClick={() => { setShowGlossary(true); onNavClick?.(); }}
          className="flex items-center gap-2 w-full px-3 lg:px-4 py-2 lg:py-2.5 mt-2 text-sm text-slate-400 hover:text-white hover:bg-white/[0.03] active:bg-white/[0.06] rounded-lg transition-all touch-manipulation"
        >
          <BookOpen className="w-4 h-4" />
          <span>Terminology</span>
        </button>
      </div>

      {/* View As Switcher - hidden on very small screens, shown on tablet+ */}
      <div className="hidden sm:block px-2 lg:px-3 pb-3">
        <div className="bg-black/20 rounded-xl p-2 lg:p-3 border border-white/[0.06]">
          <ViewAsSwitcher />
        </div>
      </div>

      {/* User section */}
      <div className="p-3 lg:p-4 border-t border-white/[0.06] bg-[rgba(10,10,15,0.8)]">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 lg:w-10 lg:h-10 rounded-full bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center text-white font-bold shadow-lg text-sm lg:text-base">
            {user.full_name?.[0] || user.email[0].toUpperCase()}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm text-white font-medium truncate">{user.full_name || user.email}</p>
            <p className="text-[10px] text-orange-400 uppercase tracking-wider font-medium">Admin</p>
          </div>
          <Link
            href="/api/auth/signout"
            prefetch={false}
            onClick={onNavClick}
            className="p-2 text-slate-500 hover:text-white hover:bg-white/[0.05] active:bg-white/[0.1] rounded-lg transition-all touch-manipulation min-w-[44px] min-h-[44px] flex items-center justify-center"
            title="Sign out"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
            </svg>
          </Link>
        </div>
      </div>
    </>
  );

  return (
    <>
      {/* Mobile Header Bar - visible only on mobile/tablet */}
      <header className="lg:hidden fixed top-0 left-0 right-0 h-14 bg-[rgba(10,10,15,0.97)] backdrop-blur-xl border-b border-white/[0.08] z-50 flex items-center justify-between px-3">
        <button
          onClick={() => setMobileMenuOpen(true)}
          className="p-2 -ml-2 text-slate-400 hover:text-white active:bg-white/[0.06] rounded-lg transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
          aria-label="Open menu"
        >
          <Menu className="w-6 h-6" />
        </button>

        <div className="flex items-center gap-2">
          <Image
            src="/images/brand/monogram.png"
            alt="LatticeForge"
            width={28}
            height={28}
            className="w-7 h-7"
          />
          <div className="text-center">
            <span className="text-white font-semibold text-sm">
              {currentPage?.label || 'Admin'}
            </span>
            <span className="text-orange-400 text-[10px] block -mt-0.5">Console</span>
          </div>
        </div>

        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center text-white font-bold text-sm shadow-lg">
          {user.full_name?.[0] || user.email[0].toUpperCase()}
        </div>
      </header>

      {/* Mobile Menu Overlay */}
      <div
        className={`
          lg:hidden fixed inset-0 bg-black/60 backdrop-blur-sm z-[60]
          transition-opacity duration-300
          ${mobileMenuOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}
        `}
        onClick={() => setMobileMenuOpen(false)}
        aria-hidden="true"
      />

      {/* Mobile Slide-out Menu */}
      <div
        className={`
          lg:hidden fixed top-0 left-0 bottom-0 w-[280px] sm:w-[300px] z-[70]
          bg-[rgba(10,10,15,0.98)] backdrop-blur-xl
          border-r border-white/[0.08]
          transform transition-transform duration-300 ease-out
          ${mobileMenuOpen ? 'translate-x-0' : '-translate-x-full'}
          flex flex-col
        `}
        style={{ WebkitBackdropFilter: 'blur(24px)' }}
      >
        {/* Close button */}
        <button
          onClick={() => setMobileMenuOpen(false)}
          className="absolute top-3 right-3 p-2 text-slate-400 hover:text-white active:bg-white/[0.06] rounded-lg transition-colors z-10 min-w-[44px] min-h-[44px] flex items-center justify-center"
          aria-label="Close menu"
        >
          <X className="w-5 h-5" />
        </button>

        <SidebarContent onNavClick={() => setMobileMenuOpen(false)} />
      </div>

      {/* Desktop Sidebar - hidden on mobile, fixed on desktop */}
      <nav className="hidden lg:flex lg:flex-col fixed left-0 top-0 h-full w-72 bg-[rgba(10,10,15,0.95)] backdrop-blur-xl border-r border-white/[0.06] z-50">
        <SidebarContent />
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
