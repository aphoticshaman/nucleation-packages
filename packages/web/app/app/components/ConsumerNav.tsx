'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import type { UserProfile, UserTier } from '@/lib/auth';

interface ConsumerNavProps {
  user: UserProfile;
}

interface NavItem {
  href: string;
  label: string;
  icon: string;
  minTier?: UserTier; // Minimum tier required (undefined = all tiers)
  badge?: string; // Optional badge text
}

// Tier hierarchy for comparison
const TIER_ORDER: UserTier[] = ['free', 'starter', 'pro', 'enterprise_tier'];

function hasTierAccess(userTier: UserTier, requiredTier: UserTier): boolean {
  return TIER_ORDER.indexOf(userTier) >= TIER_ORDER.indexOf(requiredTier);
}

// Get tier display name and color
function getTierDisplay(tier: UserTier): { name: string; color: string } {
  switch (tier) {
    case 'free': return { name: 'Free', color: 'text-slate-400' };
    case 'starter': return { name: 'Pro', color: 'text-blue-400' };
    case 'pro': return { name: 'Team', color: 'text-purple-400' };
    case 'enterprise_tier': return { name: 'Enterprise', color: 'text-amber-400' };
    default: return { name: 'Free', color: 'text-slate-400' };
  }
}

const navItems: NavItem[] = [
  { href: '/app', label: 'Globe', icon: '🌍' },
  { href: '/app/navigator', label: '3D Navigator', icon: '🌐', minTier: 'starter' },
  { href: '/app/dashboards', label: 'Dashboards', icon: '📊' },
  { href: '/app/briefings', label: 'Briefings', icon: '📡' },
  { href: '/app/packages', label: 'Packages', icon: '📦' },
  { href: '/app/signals', label: 'Signals', icon: '📈', minTier: 'starter' },
  { href: '/app/alerts', label: 'Alerts', icon: '🔔', minTier: 'starter' },
  { href: '/app/saved', label: 'Saved', icon: '💾' },
  { href: '/app/settings', label: 'Settings', icon: '⚙️' },
];

export default function ConsumerNav({ user }: ConsumerNavProps) {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const tierDisplay = getTierDisplay(user.tier);

  // Filter nav items based on tier access
  const accessibleItems = navItems.filter(
    (item) => !item.minTier || hasTierAccess(user.tier, item.minTier)
  );
  const lockedItems = navItems.filter(
    (item) => item.minTier && !hasTierAccess(user.tier, item.minTier)
  );

  return (
    <>
      <nav className="fixed top-0 left-0 right-0 h-16 bg-slate-900 border-b border-slate-800 z-50">
        <div className="max-w-7xl mx-auto px-4 h-full flex items-center justify-between">
          {/* Logo + Tier badge */}
          <Link href="/app" className="flex items-center gap-2">
            <span className="text-xl font-bold text-white">LatticeForge</span>
            <span className={`text-xs px-2 py-0.5 rounded-full bg-slate-800 ${tierDisplay.color}`}>
              {tierDisplay.name}
            </span>
          </Link>

          {/* Desktop nav links */}
          <div className="hidden md:flex items-center gap-1">
            {accessibleItems.map((item) => {
              const isActive =
                pathname === item.href || (item.href !== '/app' && pathname.startsWith(item.href));

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
            {/* Show locked items as disabled */}
            {lockedItems.length > 0 && user.tier === 'free' && (
              <Link
                href="/pricing"
                className="px-4 py-2 rounded-lg text-sm text-slate-600 hover:text-slate-400 flex items-center gap-1"
                title="Upgrade to unlock more features"
              >
                <span>+{lockedItems.length} more</span>
                <span>🔒</span>
              </Link>
            )}
          </div>

          {/* Right side */}
          <div className="flex items-center gap-3">
            {/* Upgrade CTA - desktop */}
            <a
              href="/pricing"
              className="text-sm text-blue-400 hover:text-blue-300 hidden lg:block"
            >
              Upgrade
            </a>

            {/* User avatar */}
            <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-white text-sm font-medium">
              {user.full_name?.[0] || user.email?.[0]?.toUpperCase() || '?'}
            </div>

            {/* Mobile menu button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 text-slate-400 hover:text-white"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? (
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              ) : (
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                </svg>
              )}
            </button>
          </div>
        </div>
      </nav>

      {/* Mobile menu overlay */}
      {mobileMenuOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      {/* Mobile slide-out menu */}
      <div
        className={`fixed top-16 right-0 bottom-0 w-64 bg-slate-900 border-l border-slate-800 z-50 transform transition-transform duration-200 ease-in-out md:hidden ${
          mobileMenuOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <div className="p-4 space-y-2">
          {/* Tier badge */}
          <div className="px-4 py-2 mb-2">
            <span className={`text-sm px-3 py-1 rounded-full bg-slate-800 ${tierDisplay.color}`}>
              {tierDisplay.name} Plan
            </span>
          </div>

          {accessibleItems.map((item) => {
            const isActive =
              pathname === item.href || (item.href !== '/app' && pathname.startsWith(item.href));

            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                  isActive ? 'bg-blue-600 text-white' : 'text-slate-300 hover:bg-slate-800'
                }`}
              >
                <span className="text-lg">{item.icon}</span>
                <span>{item.label}</span>
              </Link>
            );
          })}

          {/* Show locked items */}
          {lockedItems.length > 0 && (
            <>
              <hr className="border-slate-800 my-2" />
              <p className="px-4 py-1 text-xs text-slate-500">Upgrade to unlock:</p>
              {lockedItems.map((item) => (
                <Link
                  key={item.href}
                  href="/pricing"
                  onClick={() => setMobileMenuOpen(false)}
                  className="flex items-center gap-3 px-4 py-3 rounded-lg text-slate-500"
                >
                  <span className="text-lg opacity-50">{item.icon}</span>
                  <span>{item.label}</span>
                  <span className="ml-auto">🔒</span>
                </Link>
              ))}
            </>
          )}

          <hr className="border-slate-800 my-4" />

          <a
            href="/pricing"
            onClick={() => setMobileMenuOpen(false)}
            className="flex items-center gap-3 px-4 py-3 rounded-lg text-blue-400 hover:bg-slate-800"
          >
            <span className="text-lg">⬆️</span>
            <span>{user.tier === 'free' ? 'Upgrade to Pro' : 'Manage Subscription'}</span>
          </a>
        </div>

        {/* User info at bottom */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-slate-700 flex items-center justify-center text-white font-medium">
              {user.full_name?.[0] || user.email?.[0]?.toUpperCase() || '?'}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-white text-sm truncate">{user.full_name || 'User'}</p>
              <p className="text-slate-500 text-xs truncate">{user.email}</p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
