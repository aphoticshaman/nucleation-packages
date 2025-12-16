/**
 * Tier-based access control utilities
 *
 * Maps internal tiers to pricing tiers and enforces feature gates.
 */

import { getUser, type UserTier } from './auth';
import {
  mapUserTierToPricing,
  TIER_CAPABILITIES,
  type PricingTier
} from './doctrine/types';

export { mapUserTierToPricing, TIER_CAPABILITIES, type PricingTier };

export interface TierCheckResult {
  allowed: boolean;
  userTier: UserTier;
  pricingTier: PricingTier;
  reason?: string;
}

/**
 * Check if user has required tier for a feature
 */
export async function checkTierAccess(
  requiredTier: PricingTier,
  feature?: keyof typeof TIER_CAPABILITIES['observer']
): Promise<TierCheckResult> {
  const user = await getUser();

  if (!user) {
    return {
      allowed: false,
      userTier: 'free',
      pricingTier: 'observer',
      reason: 'Authentication required'
    };
  }

  const pricingTier = mapUserTierToPricing(user.tier);
  const tierRank: Record<PricingTier, number> = {
    observer: 0,
    operational: 1,
    integrated: 2,
    stewardship: 3
  };

  // Check tier rank
  if (tierRank[pricingTier] < tierRank[requiredTier]) {
    return {
      allowed: false,
      userTier: user.tier,
      pricingTier,
      reason: `This feature requires ${requiredTier} tier or higher`
    };
  }

  // Check specific feature if provided
  if (feature && !TIER_CAPABILITIES[pricingTier][feature]) {
    return {
      allowed: false,
      userTier: user.tier,
      pricingTier,
      reason: `Feature '${feature}' not available in your tier`
    };
  }

  return {
    allowed: true,
    userTier: user.tier,
    pricingTier
  };
}

/**
 * Get rate limit for user's tier
 */
export async function getTierRateLimit(): Promise<number> {
  const user = await getUser();
  if (!user) return TIER_CAPABILITIES.observer.rate_limit;

  const pricingTier = mapUserTierToPricing(user.tier);
  return TIER_CAPABILITIES[pricingTier].rate_limit;
}

/**
 * Get all capabilities for user's tier
 */
export async function getUserCapabilities() {
  const user = await getUser();
  if (!user) {
    return {
      tier: 'observer' as PricingTier,
      capabilities: TIER_CAPABILITIES.observer,
      isAuthenticated: false
    };
  }

  const pricingTier = mapUserTierToPricing(user.tier);
  return {
    tier: pricingTier,
    capabilities: TIER_CAPABILITIES[pricingTier],
    isAuthenticated: true,
    user: {
      id: user.id,
      email: user.email,
      role: user.role
    }
  };
}

/**
 * Tier names for display
 */
export const TIER_DISPLAY_NAMES: Record<PricingTier, string> = {
  observer: 'Observer',
  operational: 'Operational Intelligence',
  integrated: 'Integrated Intelligence',
  stewardship: 'Doctrine Stewardship'
};

/**
 * Tier descriptions for upgrade prompts
 */
export const TIER_DESCRIPTIONS: Record<PricingTier, string> = {
  observer: 'Evaluation access with read-only intelligence snapshots',
  operational: 'Usage-based access with alerts and API',
  integrated: 'Enterprise embedding with webhooks and audit trails',
  stewardship: 'Governance access with doctrine registry and shadow evaluation'
};
