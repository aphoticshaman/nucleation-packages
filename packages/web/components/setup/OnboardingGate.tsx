'use client';

import { useState, useEffect } from 'react';
import { OnboardingWizard, type OnboardingConfig } from './OnboardingWizard';
import { authTierToPowerTier, type UserTier, type AuthTier } from '@/lib/config/powerUser';

interface OnboardingGateProps {
  children: React.ReactNode;
  userId: string;
  userTier: UserTier | AuthTier;
  userRole?: string;
  hasCompletedOnboarding: boolean;
}

/**
 * Gates the app behind onboarding for first-time users.
 *
 * Philosophy: Force users to narrow their focus on first login.
 * - Too filtered = no insight, product feels useless
 * - Too broad = overwhelming, product feels unmanageable
 * - Wrong terminology = user doesn't understand, churns
 *
 * The wizard adapts based on their stated use case and experience.
 */
// Check if tier is an auth tier
const isAuthTier = (tier: string): tier is AuthTier => {
  return ['free', 'starter', 'pro', 'enterprise_tier'].includes(tier);
};

export default function OnboardingGate({
  children,
  userId,
  userTier,
  userRole,
  hasCompletedOnboarding,
}: OnboardingGateProps) {
  // Admins skip onboarding entirely - they have full access and know the platform
  const isAdmin = userRole === 'admin';

  // Check localStorage first to avoid showing wizard after skip/complete
  const getInitialShowWizard = () => {
    if (typeof window === 'undefined') return false; // SSR: don't show
    if (isAdmin) return false; // Admins never see wizard
    if (hasCompletedOnboarding) return false; // DB says completed

    // Check localStorage for skip or complete
    const skipped = localStorage.getItem('latticeforge_onboarding_skipped');
    const completed = localStorage.getItem('latticeforge_onboarding_complete');
    if (skipped === 'true' || completed === 'true') return false;

    return true;
  };

  // Start with false on server, hydrate on client
  const [showWizard, setShowWizard] = useState(false);
  const [isCompleting, setIsCompleting] = useState(false);
  const [hydrated, setHydrated] = useState(false);

  // Hydrate on mount
  useEffect(() => {
    setHydrated(true);
    setShowWizard(getInitialShowWizard());
  }, []);

  // Convert auth tier to power tier if needed
  const powerTier: UserTier = isAuthTier(userTier) ? authTierToPowerTier(userTier) : userTier;

  // Sync localStorage with database state and fetch preferences if needed
  useEffect(() => {
    // If DB says completed, sync to localStorage and hide wizard
    if (hasCompletedOnboarding || isAdmin) {
      localStorage.setItem('latticeforge_onboarding_complete', 'true');
      setShowWizard(false);

      // Also fetch saved preferences from DB if not in localStorage
      const localPrefs = localStorage.getItem('latticeforge_user_preferences');
      if (!localPrefs && !isAdmin) {
        fetch('/api/user/preferences')
          .then(res => res.json())
          .then(data => {
            if (data.preferences?.preferences) {
              localStorage.setItem('latticeforge_user_preferences', JSON.stringify(data.preferences.preferences));
            }
          })
          .catch(err => console.error('Failed to fetch preferences:', err));
      }
    }
  }, [hasCompletedOnboarding, isAdmin]);

  const handleComplete = async (config: OnboardingConfig) => {
    setIsCompleting(true);

    try {
      // Save preferences to Supabase
      const response = await fetch('/api/user/preferences', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId,
          onboardingConfig: config,
          onboardingCompletedAt: new Date().toISOString(),
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to save preferences');
      }

      // Also save to localStorage for immediate use
      localStorage.setItem('latticeforge_user_preferences', JSON.stringify(config));
      localStorage.setItem('latticeforge_onboarding_complete', 'true');

      setShowWizard(false);
    } catch (error) {
      console.error('Failed to save onboarding config:', error);
      // Still allow them through even if save fails
      localStorage.setItem('latticeforge_user_preferences', JSON.stringify(config));
      setShowWizard(false);
    } finally {
      setIsCompleting(false);
    }
  };

  // Allow ALL users to skip - onboarding should feel valuable, not forced
  const handleSkip = () => {
    localStorage.setItem('latticeforge_onboarding_skipped', 'true');
    setShowWizard(false);
  };

  if (showWizard) {
    return (
      <OnboardingWizard
        userTier={powerTier}
        onComplete={handleComplete}
        onSkip={handleSkip}
      />
    );
  }

  return <>{children}</>;
}

/**
 * Hook to check if user needs onboarding
 */
export function useNeedsOnboarding(): boolean {
  const [needsOnboarding, setNeedsOnboarding] = useState(false);

  useEffect(() => {
    const completed = localStorage.getItem('latticeforge_onboarding_complete');
    setNeedsOnboarding(completed !== 'true');
  }, []);

  return needsOnboarding;
}

/**
 * Get saved user preferences from onboarding
 */
export function getUserPreferences(): Record<string, unknown> | null {
  if (typeof window === 'undefined') return null;

  const saved = localStorage.getItem('latticeforge_user_preferences');
  if (!saved) return null;

  try {
    return JSON.parse(saved);
  } catch {
    return null;
  }
}
