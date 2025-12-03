import Stripe from 'stripe';

// ============================================
// STRIPE SETUP CHECKLIST
// ============================================
// Before subscriptions work, you need:
//
// 1. STRIPE_SECRET_KEY - From Stripe Dashboard > Developers > API Keys
// 2. STRIPE_WEBHOOK_SECRET - From Stripe Dashboard > Developers > Webhooks
//    (Create webhook pointing to /api/webhooks/stripe)
// 3. Create Products & Prices in Stripe Dashboard:
//    - Starter ($19/mo) -> copy Price ID to STRIPE_PRICE_STARTER
//    - Pro ($49/mo) -> copy Price ID to STRIPE_PRICE_PRO
//    - Enterprise (custom) -> copy Price ID to STRIPE_PRICE_ENTERPRISE
//
// Test mode price IDs look like: price_1ABC123...
// ============================================

// Lazy-initialized Stripe client (avoids build-time env var access)
let _stripe: Stripe | null = null;

export function isStripeConfigured(): { ok: boolean; missing: string[] } {
  const missing: string[] = [];
  if (!process.env.STRIPE_SECRET_KEY) missing.push('STRIPE_SECRET_KEY');
  if (!process.env.STRIPE_WEBHOOK_SECRET) missing.push('STRIPE_WEBHOOK_SECRET');
  if (!process.env.STRIPE_PRICE_PRO) missing.push('STRIPE_PRICE_PRO');
  if (!process.env.STRIPE_PRICE_TEAM) missing.push('STRIPE_PRICE_TEAM');
  // Enterprise is optional (custom pricing)
  return { ok: missing.length === 0, missing };
}

function getStripe(): Stripe {
  if (!_stripe) {
    const secretKey = process.env.STRIPE_SECRET_KEY;
    if (!secretKey) {
      throw new Error(
        'STRIPE_SECRET_KEY not configured. Add it to your .env.local file. ' +
        'Get it from: https://dashboard.stripe.com/apikeys'
      );
    }
    _stripe = new Stripe(secretKey, {
      apiVersion: '2023-10-16',
      typescript: true,
    });
  }
  return _stripe;
}

// Exported for compatibility with existing code
export const stripe = {
  get webhooks() {
    return getStripe().webhooks;
  },
  get checkout() {
    return getStripe().checkout;
  },
  get billingPortal() {
    return getStripe().billingPortal;
  },
  get subscriptions() {
    return getStripe().subscriptions;
  },
  get customers() {
    return getStripe().customers;
  },
  get subscriptionItems() {
    return getStripe().subscriptionItems;
  },
};

// ============================================
// STRIPE PRODUCTS TO CREATE IN DASHBOARD
// ============================================
// Create these products in Stripe Dashboard:
//
// 1. Product: "LatticeForge Pro" (Individual)
//    - Price ID: price_pro_monthly
//    - Amount: $79/month
//    - Metadata: { tier: "pro", seats: "1" }
//
// 2. Product: "LatticeForge Team" (Per-seat)
//    - Price ID: price_team_monthly
//    - Amount: $59/seat/month
//    - Metadata: { tier: "team", per_seat: "true" }
//
// 3. Product: "LatticeForge Enterprise"
//    - Price ID: price_enterprise_monthly
//    - Amount: Custom (contact sales)
//    - Metadata: { tier: "enterprise" }
//
// COMPETITIVE ANALYSIS:
// - Dataminr: $50k+/year (enterprise only)
// - Recorded Future: $100k+/year
// - Factal: ~$500/mo for small teams
// - Riskline: $50-100/user/mo
// - US: $79/mo individual, $59/seat team = UNDERCUT by 30-50%
// ============================================

// Price IDs (set these in your Stripe dashboard and env vars)
// NOTE: Using getter to ensure runtime evaluation, not build-time
export function getPriceId(plan: 'pro' | 'team' | 'enterprise'): string | undefined {
  switch (plan) {
    case 'pro':
      return process.env.STRIPE_PRICE_PRO;
    case 'team':
      return process.env.STRIPE_PRICE_TEAM;
    case 'enterprise':
      return process.env.STRIPE_PRICE_ENTERPRISE;
  }
}

// Legacy export for compatibility (evaluated at build time - may be undefined)
export const PRICE_IDS = {
  pro: process.env.STRIPE_PRICE_PRO!,
  team: process.env.STRIPE_PRICE_TEAM!,
  enterprise: process.env.STRIPE_PRICE_ENTERPRISE!,
};

// Plan definitions - Bottoms-up SaaS pricing
// Individual pays credit card, spreads to team, enterprise comes to us
export const PLANS = {
  free: {
    name: 'Free',
    price: 0,
    interval: 'month' as const,
    features: [
      '1 region focus only',
      '7-day historical data',
      'Daily briefing (delayed)',
      'Basic dashboard',
      'Community support',
    ],
    limits: {
      regions: 1,
      history_days: 7,
      alerts: 0,
      api_calls: 0,
      saved_views: 3,
      exports: 0,
      team_seats: 1,
    },
  },
  trial: {
    name: 'Pro Trial',
    price: 0,
    interval: 'month' as const,
    trialDays: 14,
    requiresCard: true,
    convertsTo: 'pro',
    features: [
      '14-day free trial of Pro',
      'All Pro features unlocked',
      'No credit card charge until trial ends',
      'Cancel anytime',
    ],
    limits: {
      regions: -1,
      history_days: 90,
      alerts: -1,
      api_calls: 5000,
      saved_views: -1,
      exports: -1,
      team_seats: 1,
    },
  },
  pro: {
    name: 'Pro',
    price: 79,
    interval: 'month' as const,
    popular: true,
    description: 'For individual analysts and researchers',
    features: [
      'All regions & categories',
      '90-day historical data',
      'Real-time alerts (unlimited)',
      'Custom dashboards',
      '5,000 API calls/month',
      'PDF & Excel exports',
      'Email support (24h response)',
    ],
    limits: {
      regions: -1,
      history_days: 90,
      alerts: -1,
      api_calls: 5000,
      saved_views: -1,
      exports: -1,
      team_seats: 1,
    },
  },
  team: {
    name: 'Team',
    price: 59, // per seat
    pricePerSeat: true,
    interval: 'month' as const,
    minSeats: 3,
    description: 'For teams who need to collaborate',
    features: [
      'Everything in Pro',
      '1-year historical data',
      'Shared dashboards & views',
      'Team comments & annotations',
      '10,000 API calls/month per seat',
      'Slack/Teams integration',
      'Priority support (4h response)',
    ],
    limits: {
      regions: -1,
      history_days: 365,
      alerts: -1,
      api_calls: 10000, // per seat
      saved_views: -1,
      exports: -1,
      team_seats: -1, // pay per seat
    },
  },
  enterprise: {
    name: 'Enterprise',
    price: null, // Custom pricing
    interval: 'month' as const,
    description: 'For organizations with advanced needs',
    features: [
      'Everything in Team',
      'Unlimited historical data',
      'SSO (SAML, OIDC)',
      'Audit logs & compliance',
      'Custom data integrations',
      'Dedicated success manager',
      'SLA guarantee (99.9%)',
      'On-premise deployment option',
    ],
    limits: {
      regions: -1,
      history_days: -1,
      alerts: -1,
      api_calls: -1,
      saved_views: -1,
      exports: -1,
      team_seats: -1,
    },
  },
};

// Annual pricing (2 months free)
export const ANNUAL_DISCOUNT = 2 / 12; // ~17% off
export const getAnnualPrice = (monthlyPrice: number) =>
  Math.round(monthlyPrice * 12 * (1 - ANNUAL_DISCOUNT));

export type PlanId = keyof typeof PLANS;

// Trial status helpers
export const TRIAL_DURATION_DAYS = 7;

export function isTrialExpired(trialEndsAt: string | Date | null): boolean {
  if (!trialEndsAt) return true;
  const endDate = new Date(trialEndsAt);
  return endDate < new Date();
}

export function getTrialDaysRemaining(trialEndsAt: string | Date | null): number {
  if (!trialEndsAt) return 0;
  const endDate = new Date(trialEndsAt);
  const now = new Date();
  const diffMs = endDate.getTime() - now.getTime();
  const diffDays = Math.ceil(diffMs / (1000 * 60 * 60 * 24));
  return Math.max(0, diffDays);
}

export function getTrialEndDate(): Date {
  const endDate = new Date();
  endDate.setDate(endDate.getDate() + TRIAL_DURATION_DAYS);
  return endDate;
}

// Create checkout session
export async function createCheckoutSession({
  priceId,
  customerId,
  organizationId,
  successUrl,
  cancelUrl,
}: {
  priceId: string;
  customerId?: string;
  organizationId: string;
  successUrl: string;
  cancelUrl: string;
}) {
  const session = await stripe.checkout.sessions.create({
    mode: 'subscription',
    payment_method_types: ['card'],
    line_items: [
      {
        price: priceId,
        quantity: 1,
      },
    ],
    customer: customerId,
    success_url: successUrl,
    cancel_url: cancelUrl,
    metadata: {
      organization_id: organizationId,
    },
    subscription_data: {
      metadata: {
        organization_id: organizationId,
      },
    },
    allow_promotion_codes: true,
  });

  return session;
}

// Create billing portal session
export async function createBillingPortalSession({
  customerId,
  returnUrl,
}: {
  customerId: string;
  returnUrl: string;
}) {
  const session = await stripe.billingPortal.sessions.create({
    customer: customerId,
    return_url: returnUrl,
  });

  return session;
}

// Get subscription details
export async function getSubscription(subscriptionId: string) {
  const subscription = await stripe.subscriptions.retrieve(subscriptionId, {
    expand: ['items.data.price.product'],
  });

  return subscription;
}

// Cancel subscription
export async function cancelSubscription(subscriptionId: string) {
  const subscription = await stripe.subscriptions.update(subscriptionId, {
    cancel_at_period_end: true,
  });

  return subscription;
}

// Resume subscription
export async function resumeSubscription(subscriptionId: string) {
  const subscription = await stripe.subscriptions.update(subscriptionId, {
    cancel_at_period_end: false,
  });

  return subscription;
}

// Create or get customer
export async function getOrCreateCustomer({
  email,
  name,
  organizationId,
}: {
  email: string;
  name?: string;
  organizationId: string;
}) {
  // Check if customer exists
  const existing = await stripe.customers.list({
    email,
    limit: 1,
  });

  if (existing.data.length > 0) {
    return existing.data[0];
  }

  // Create new customer
  const customer = await stripe.customers.create({
    email,
    name,
    metadata: {
      organization_id: organizationId,
    },
  });

  return customer;
}

// Update usage (for metered billing if needed)
export async function reportUsage({
  subscriptionItemId,
  quantity,
}: {
  subscriptionItemId: string;
  quantity: number;
}) {
  const usageRecord = await stripe.subscriptionItems.createUsageRecord(subscriptionItemId, {
    quantity,
    timestamp: Math.floor(Date.now() / 1000),
    action: 'increment',
  });

  return usageRecord;
}
