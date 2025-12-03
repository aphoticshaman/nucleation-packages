'use client';

import { useState, useEffect } from 'react';
import { OnboardingWizard } from './OnboardingWizard';
import type { UserTier } from '@/lib/config/powerUser';

interface OnboardingGateProps {
  children: React.ReactNode;
  userId: string;
  userTier: UserTier;
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
export default function OnboardingGate({
  children,
  userId,
  userTier,
  hasCompletedOnboarding,
}: OnboardingGateProps) {
  const [showWizard, setShowWizard] = useState(!hasCompletedOnboarding);
  const [isCompleting, setIsCompleting] = useState(false);

  // Check localStorage for skip (dev/testing only - should be removed in prod)
  useEffect(() => {
    const skipOnboarding = localStorage.getItem('latticeforge_skip_onboarding');
    if (skipOnboarding === 'true' && process.env.NODE_ENV === 'development') {
      setShowWizard(false);
    }
  }, []);

  const handleComplete = async (config: Record<string, unknown>) => {
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

  // First-time users MUST complete onboarding (no skip on first visit)
  // Return users who skipped can skip again
  const canSkip = hasCompletedOnboarding || localStorage.getItem('latticeforge_onboarding_skipped') === 'true';

  const handleSkip = () => {
    if (canSkip) {
      localStorage.setItem('latticeforge_onboarding_skipped', 'true');
      setShowWizard(false);
    }
  };

  if (showWizard) {
    return (
      <OnboardingWizard
        userTier={userTier}
        onComplete={handleComplete}
        onSkip={canSkip ? handleSkip : undefined}
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
