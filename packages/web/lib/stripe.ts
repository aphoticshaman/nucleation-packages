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
  if (!process.env.STRIPE_PRICE_STARTER) missing.push('STRIPE_PRICE_STARTER');
  if (!process.env.STRIPE_PRICE_PRO) missing.push('STRIPE_PRICE_PRO');
  if (!process.env.STRIPE_PRICE_ENTERPRISE) missing.push('STRIPE_PRICE_ENTERPRISE');
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
      apiVersion: '2024-11-20.acacia',
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
// 1. Product: "LatticeForge Starter"
//    - Price ID: price_starter_monthly
//    - Amount: $49/month
//    - Metadata: { tier: "starter", api_calls: "1000", seats: "3" }
//
// 2. Product: "LatticeForge Pro"
//    - Price ID: price_pro_monthly
//    - Amount: $199/month
//    - Metadata: { tier: "pro", api_calls: "10000", seats: "10" }
//
// 3. Product: "LatticeForge Enterprise"
//    - Price ID: price_enterprise_monthly
//    - Amount: $499/month (or custom)
//    - Metadata: { tier: "enterprise", api_calls: "100000", seats: "unlimited" }
// ============================================

// Price IDs (set these in your Stripe dashboard and env vars)
export const PRICE_IDS = {
  starter: process.env.STRIPE_PRICE_STARTER!,
  pro: process.env.STRIPE_PRICE_PRO!,
  enterprise: process.env.STRIPE_PRICE_ENTERPRISE!,
};

// Plan definitions
export const PLANS = {
  trial: {
    name: 'Trial',
    price: 0,
    interval: 'month' as const,
    trialDays: 7,
    requiresCard: true, // CC required for trial
    convertsTo: 'starter', // Auto-converts to Starter after trial
    features: [
      '7-day free trial of Starter plan',
      'Full Starter features during trial',
      '1,000 API calls/month',
      '3 team seats',
      'REST API access',
      'Cancel anytime before trial ends',
    ],
    // Same limits as Starter during trial
    limits: {
      simulations_per_day: -1,
      saved_simulations: -1,
      api_calls: 1000,
      team_seats: 3,
    },
  },
  free: {
    name: 'Free',
    price: 0,
    interval: 'month' as const,
    features: [
      '10 simulations per day',
      '5 saved simulations',
      'Basic visualizations',
      'Community support',
    ],
    limits: {
      simulations_per_day: 10,
      saved_simulations: 5,
      api_calls: 0,
      team_seats: 1,
    },
  },
  starter: {
    name: 'Starter',
    price: 19,
    interval: 'month' as const,
    popular: true,
    features: [
      '1,000 API calls/month',
      '3 team seats',
      'REST API access',
      'Basic webhooks',
      'Email support',
    ],
    limits: {
      simulations_per_day: -1, // unlimited
      saved_simulations: -1,
      api_calls: 1000,
      team_seats: 3,
    },
  },
  pro: {
    name: 'Pro',
    price: 49,
    interval: 'month' as const,
    features: [
      '10,000 API calls/month',
      '10 team seats',
      'REST + WebSocket APIs',
      'Real-time streaming',
      'Advanced webhooks',
      'Priority support',
    ],
    limits: {
      simulations_per_day: -1,
      saved_simulations: -1,
      api_calls: 10000,
      team_seats: 10,
    },
  },
  enterprise: {
    name: 'Enterprise',
    price: null, // Custom pricing
    interval: 'month' as const,
    features: [
      'Unlimited API calls',
      'Unlimited team seats',
      'Full API suite',
      'Custom integrations',
      'SLA guarantee',
      'Dedicated support',
      'On-premise option',
    ],
    limits: {
      simulations_per_day: -1,
      saved_simulations: -1,
      api_calls: -1,
      team_seats: -1,
    },
  },
};

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
