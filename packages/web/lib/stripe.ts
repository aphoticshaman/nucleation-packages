import Stripe from 'stripe';

// Lazy-initialized Stripe client (avoids build-time env var access)
let _stripe: Stripe | null = null;

function getStripe(): Stripe {
  if (!_stripe) {
    _stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
      apiVersion: '2023-10-16',
      typescript: true,
    });
  }
  return _stripe;
}

// Exported for compatibility with existing code
export const stripe = {
  get webhooks() { return getStripe().webhooks; },
  get checkout() { return getStripe().checkout; },
  get billingPortal() { return getStripe().billingPortal; },
  get subscriptions() { return getStripe().subscriptions; },
  get customers() { return getStripe().customers; },
  get subscriptionItems() { return getStripe().subscriptionItems; },
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
    features: [
      '7-day free trial',
      '25 simulations per day',
      '10 saved simulations',
      'Basic visualizations',
      'Email support',
    ],
    limits: {
      simulations_per_day: 25,
      saved_simulations: 10,
      api_calls: 0,
      team_seats: 1,
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
  const usageRecord = await stripe.subscriptionItems.createUsageRecord(
    subscriptionItemId,
    {
      quantity,
      timestamp: Math.floor(Date.now() / 1000),
      action: 'increment',
    }
  );

  return usageRecord;
}
