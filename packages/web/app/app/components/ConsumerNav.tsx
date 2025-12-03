'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import Image from 'next/image';
import type { UserProfile, UserTier } from '@/lib/auth';

interface ConsumerNavProps {
  user: UserProfile;
}

interface NavItem {
  href: string;
  label: string;
  icon: string;
  minTier?: UserTier;
}

const TIER_ORDER: UserTier[] = ['free', 'starter', 'pro', 'enterprise_tier'];

function hasTierAccess(userTier: UserTier, requiredTier: UserTier): boolean {
  return TIER_ORDER.indexOf(userTier) >= TIER_ORDER.indexOf(requiredTier);
}

function getTierDisplay(tier: UserTier): { name: string; color: string; bg: string } {
  switch (tier) {
    case 'free':
      return { name: 'Free', color: 'text-slate-300', bg: 'bg-slate-700/50' };
    case 'starter':
      return { name: 'Pro', color: 'text-blue-400', bg: 'bg-blue-500/20' };
    case 'pro':
      return { name: 'Team', color: 'text-purple-400', bg: 'bg-purple-500/20' };
    case 'enterprise_tier':
      return { name: 'Enterprise', color: 'text-amber-400', bg: 'bg-amber-500/20' };
    default:
      return { name: 'Free', color: 'text-slate-300', bg: 'bg-slate-700/50' };
  }
}

// Core navigation - mobile-optimized (fewer items visible)
const navItems: NavItem[] = [
  { href: '/app', label: 'Globe', icon: '🌍' },
  { href: '/app/navigator', label: '3D Nav', icon: '🌐', minTier: 'starter' },
  { href: '/app/dashboards', label: 'Dash', icon: '📊' },
  { href: '/app/briefings', label: 'Intel', icon: '📡' },
  { href: '/app/packages', label: 'Packages', icon: '📦' },
  { href: '/app/signals', label: 'Signals', icon: '📈', minTier: 'starter' },
  { href: '/app/alerts', label: 'Alerts', icon: '🔔', minTier: 'starter' },
  { href: '/app/saved', label: 'Saved', icon: '💾' },
  { href: '/app/settings', label: 'Settings', icon: '⚙️' },
];

export default function ConsumerNav({ user }: ConsumerNavProps) {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const tierDisplay = getTierDisplay(user.tier);

  // Enhance navbar on scroll
  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Lock body scroll when mobile menu is open
  useEffect(() => {
    document.body.style.overflow = mobileMenuOpen ? 'hidden' : '';
    return () => { document.body.style.overflow = ''; };
  }, [mobileMenuOpen]);

  const accessibleItems = navItems.filter(
    (item) => !item.minTier || hasTierAccess(user.tier, item.minTier)
  );
  const lockedItems = navItems.filter(
    (item) => item.minTier && !hasTierAccess(user.tier, item.minTier)
  );

  return (
    <>
      {/* Main navbar - glass effect */}
      <nav
        className={`
          fixed top-0 left-0 right-0 h-14 sm:h-16 z-50
          transition-all duration-300
          ${scrolled
            ? 'bg-[rgba(10,10,15,0.95)] backdrop-blur-xl border-b border-white/[0.08] shadow-lg'
            : 'bg-[rgba(10,10,15,0.8)] backdrop-blur-md border-b border-white/[0.05]'
          }
        `}
        style={{
          // Fallback for older mobile browsers
          WebkitBackdropFilter: scrolled ? 'blur(24px)' : 'blur(12px)',
        }}
      >
        <div className="h-full px-3 sm:px-4 lg:px-6 flex items-center justify-between">
          {/* Logo + Tier */}
          <Link href="/app" className="flex items-center gap-2 min-w-0">
            <Image
              src="/images/brand/monogram.png"
              alt="LatticeForge"
              width={28}
              height={28}
              className="w-7 h-7 sm:w-8 sm:h-8"
            />
            <span className="text-lg sm:text-xl font-bold text-white hidden xs:inline">
              LatticeForge
            </span>
            <span
              className={`text-[10px] sm:text-xs px-2 py-0.5 rounded-full ${tierDisplay.bg} ${tierDisplay.color} border border-current/20`}
            >
              {tierDisplay.name}
            </span>
          </Link>

          {/* Desktop navigation */}
          <div className="hidden lg:flex items-center gap-1">
            {accessibleItems.map((item) => {
              const isActive =
                pathname === item.href || (item.href !== '/app' && pathname.startsWith(item.href));

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`
                    px-3 xl:px-4 py-2 rounded-lg text-sm font-medium
                    transition-all duration-150
                    ${isActive
                      ? 'bg-blue-600/90 text-white shadow-[0_0_15px_rgba(59,130,246,0.3)]'
                      : 'text-slate-400 hover:text-white hover:bg-white/[0.06]'
                    }
                  `}
                >
                  {item.label}
                </Link>
              );
            })}
            {lockedItems.length > 0 && user.tier === 'free' && (
              <Link
                href="/pricing"
                className="px-3 py-2 rounded-lg text-sm text-slate-500 hover:text-slate-300 flex items-center gap-1"
              >
                +{lockedItems.length}
                <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z"
                    clipRule="evenodd"
                  />
                </svg>
              </Link>
            )}
          </div>

          {/* Right side actions */}
          <div className="flex items-center gap-2 sm:gap-3">
            {/* Upgrade - desktop only */}
            {user.tier === 'free' && (
              <Link
                href="/pricing"
                className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium
                  bg-gradient-to-r from-blue-600 to-purple-600 text-white
                  shadow-[0_0_15px_rgba(59,130,246,0.3)]
                  hover:shadow-[0_0_20px_rgba(59,130,246,0.4)]
                  transition-all duration-200"
              >
                <span>Upgrade</span>
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </Link>
            )}

            {/* User avatar - touchable on mobile */}
            <button
              className="w-9 h-9 sm:w-10 sm:h-10 rounded-full bg-gradient-to-br from-slate-700 to-slate-800
                flex items-center justify-center text-white text-sm font-medium
                border border-white/10 shadow-inner
                active:scale-95 transition-transform"
              onClick={() => setMobileMenuOpen(true)}
              aria-label="Open menu"
            >
              {user.full_name?.[0] || user.email?.[0]?.toUpperCase() || '?'}
            </button>
          </div>
        </div>
      </nav>

      {/* Mobile menu overlay */}
      <div
        className={`
          fixed inset-0 bg-black/60 backdrop-blur-sm z-[60]
          transition-opacity duration-300
          ${mobileMenuOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}
        `}
        onClick={() => setMobileMenuOpen(false)}
        aria-hidden="true"
      />

      {/* Mobile slide-out menu - full height, glass effect */}
      <div
        className={`
          fixed top-0 right-0 bottom-0 w-[280px] sm:w-[320px] z-[70]
          bg-[rgba(10,10,15,0.97)] backdrop-blur-xl
          border-l border-white/[0.08]
          transform transition-transform duration-300 ease-out
          ${mobileMenuOpen ? 'translate-x-0' : 'translate-x-full'}
          flex flex-col
          safe-area-inset
        `}
        style={{ WebkitBackdropFilter: 'blur(24px)' }}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-white/[0.06]">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-slate-600 to-slate-700
              flex items-center justify-center text-white font-medium border border-white/10">
              {user.full_name?.[0] || user.email?.[0]?.toUpperCase() || '?'}
            </div>
            <div className="min-w-0">
              <p className="text-white text-sm font-medium truncate">
                {user.full_name || 'User'}
              </p>
              <p className="text-slate-500 text-xs truncate">{user.email}</p>
            </div>
          </div>
          <button
            onClick={() => setMobileMenuOpen(false)}
            className="p-2 -mr-2 text-slate-400 hover:text-white active:bg-white/5 rounded-lg
              min-w-[44px] min-h-[44px] flex items-center justify-center"
            aria-label="Close menu"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Tier badge */}
        <div className="px-4 py-3">
          <span className={`inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full ${tierDisplay.bg} ${tierDisplay.color} border border-current/20`}>
            <span className="w-1.5 h-1.5 rounded-full bg-current" />
            {tierDisplay.name} Plan
          </span>
        </div>

        {/* Nav items - scrollable */}
        <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1">
          {accessibleItems.map((item) => {
            const isActive =
              pathname === item.href || (item.href !== '/app' && pathname.startsWith(item.href));

            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileMenuOpen(false)}
                className={`
                  flex items-center gap-3 px-4 py-3.5 rounded-xl
                  min-h-[52px] touch-manipulation
                  transition-all duration-150 active:scale-[0.98]
                  ${isActive
                    ? 'bg-blue-600/90 text-white shadow-[0_0_20px_rgba(59,130,246,0.25)]'
                    : 'text-slate-300 active:bg-white/[0.06]'
                  }
                `}
              >
                <span className="text-xl w-7 text-center">{item.icon}</span>
                <span className="font-medium">{item.label}</span>
                {isActive && (
                  <span className="ml-auto w-1.5 h-1.5 rounded-full bg-white" />
                )}
              </Link>
            );
          })}

          {/* Locked items */}
          {lockedItems.length > 0 && (
            <>
              <div className="pt-3 pb-2 px-4">
                <p className="text-[11px] text-slate-500 uppercase tracking-wider font-medium">
                  Upgrade to unlock
                </p>
              </div>
              {lockedItems.map((item) => (
                <Link
                  key={item.href}
                  href="/pricing"
                  onClick={() => setMobileMenuOpen(false)}
                  className="flex items-center gap-3 px-4 py-3.5 rounded-xl text-slate-500
                    min-h-[52px] touch-manipulation active:bg-white/[0.03]"
                >
                  <span className="text-xl w-7 text-center opacity-40">{item.icon}</span>
                  <span>{item.label}</span>
                  <svg className="ml-auto w-4 h-4 opacity-50" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                </Link>
              ))}
            </>
          )}
        </div>

        {/* Bottom actions - safe area aware */}
        <div className="p-4 border-t border-white/[0.06] space-y-3 pb-safe">
          {user.tier === 'free' ? (
            <Link
              href="/pricing"
              onClick={() => setMobileMenuOpen(false)}
              className="flex items-center justify-center gap-2 w-full py-3.5 rounded-xl
                bg-gradient-to-r from-blue-600 to-purple-600 text-white font-medium
                shadow-[0_0_20px_rgba(59,130,246,0.3)]
                active:scale-[0.98] transition-transform touch-manipulation min-h-[52px]"
            >
              <span>Upgrade to Pro</span>
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </Link>
          ) : (
            <Link
              href="/pricing"
              onClick={() => setMobileMenuOpen(false)}
              className="flex items-center justify-center gap-2 w-full py-3.5 rounded-xl
                bg-white/[0.06] text-slate-300 font-medium border border-white/[0.08]
                active:bg-white/[0.1] active:scale-[0.98] transition-all touch-manipulation min-h-[52px]"
            >
              <span>Manage Subscription</span>
            </Link>
          )}

          <Link
            href="/api/auth/signout"
            onClick={() => setMobileMenuOpen(false)}
            className="flex items-center justify-center gap-2 w-full py-3 rounded-xl
              text-slate-500 text-sm active:text-slate-300 active:bg-white/[0.03]
              transition-colors touch-manipulation min-h-[44px]"
          >
            Sign Out
          </Link>
        </div>
      </div>
    </>
  );
}
