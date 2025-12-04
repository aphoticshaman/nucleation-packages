'use client';

import { createContext, useContext, useState, useCallback, type ReactNode } from 'react';
import type { UserRole, UserTier } from '@/lib/auth';

// All viewable perspectives for admin
export interface ViewPerspective {
  role: UserRole;
  tier: UserTier;
  label: string;
  description: string;
}

export const VIEW_PERSPECTIVES: ViewPerspective[] = [
  { role: 'admin', tier: 'enterprise_tier', label: 'Admin', description: 'Full system access' },
  { role: 'consumer', tier: 'free', label: 'Free User', description: 'Basic features only' },
  { role: 'consumer', tier: 'starter', label: 'Starter', description: '$49/mo tier' },
  { role: 'consumer', tier: 'pro', label: 'Pro', description: '$199/mo tier' },
  { role: 'consumer', tier: 'enterprise_tier', label: 'Enterprise', description: 'Custom contract' },
  { role: 'enterprise', tier: 'enterprise_tier', label: 'Enterprise Admin', description: 'Org-level access' },
  { role: 'support', tier: 'enterprise_tier', label: 'Support', description: 'Support staff view' },
];

interface ImpersonationState {
  isImpersonating: boolean;
  viewAs: ViewPerspective | null;
  originalRole: UserRole;
  originalTier: UserTier;
}

interface ImpersonationContextValue extends ImpersonationState {
  startImpersonation: (perspective: ViewPerspective) => void;
  stopImpersonation: () => void;
  getEffectiveRole: () => UserRole;
  getEffectiveTier: () => UserTier;
  canAccessFeature: (requiredTier: UserTier) => boolean;
}

const ImpersonationContext = createContext<ImpersonationContextValue | null>(null);

// Tier hierarchy for feature access checks
const TIER_HIERARCHY: Record<UserTier, number> = {
  free: 0,
  starter: 1,
  pro: 2,
  enterprise_tier: 3,
};

interface ImpersonationProviderProps {
  children: ReactNode;
  userRole: UserRole;
  userTier: UserTier;
}

export function ImpersonationProvider({ children, userRole, userTier }: ImpersonationProviderProps) {
  const [state, setState] = useState<ImpersonationState>({
    isImpersonating: false,
    viewAs: null,
    originalRole: userRole,
    originalTier: userTier,
  });

  const startImpersonation = useCallback((perspective: ViewPerspective) => {
    // Only admins can impersonate
    if (userRole !== 'admin') return;

    setState(prev => ({
      ...prev,
      isImpersonating: true,
      viewAs: perspective,
    }));
  }, [userRole]);

  const stopImpersonation = useCallback(() => {
    setState(prev => ({
      ...prev,
      isImpersonating: false,
      viewAs: null,
    }));
  }, []);

  const getEffectiveRole = useCallback((): UserRole => {
    if (state.isImpersonating && state.viewAs) {
      return state.viewAs.role;
    }
    return state.originalRole;
  }, [state]);

  const getEffectiveTier = useCallback((): UserTier => {
    if (state.isImpersonating && state.viewAs) {
      return state.viewAs.tier;
    }
    return state.originalTier;
  }, [state]);

  const canAccessFeature = useCallback((requiredTier: UserTier): boolean => {
    const effectiveTier = getEffectiveTier();
    // Admins always have access when not impersonating
    if (!state.isImpersonating && state.originalRole === 'admin') {
      return true;
    }
    return TIER_HIERARCHY[effectiveTier] >= TIER_HIERARCHY[requiredTier];
  }, [state, getEffectiveTier]);

  return (
    <ImpersonationContext.Provider
      value={{
        ...state,
        startImpersonation,
        stopImpersonation,
        getEffectiveRole,
        getEffectiveTier,
        canAccessFeature,
      }}
    >
      {children}
    </ImpersonationContext.Provider>
  );
}

export function useImpersonation() {
  const context = useContext(ImpersonationContext);
  if (!context) {
    throw new Error('useImpersonation must be used within an ImpersonationProvider');
  }
  return context;
}

// Optional hook that won't throw if not in provider (for non-admin areas)
export function useImpersonationSafe() {
  return useContext(ImpersonationContext);
}
