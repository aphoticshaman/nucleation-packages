'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import TierBadge, { TierType } from '@/components/TierBadge';

// Plans shown to users - only show prices for self-serve tiers
const VISIBLE_PLANS: {
  id: string;
  tier: TierType;
  name: string;
  price: number | null;
  priceLabel: string;
  interval: string | null;
  features: string[];
  cta: string;
  popular?: boolean;
  href?: string;
}[] = [
  {
    id: 'free',
    tier: 'trial',
    name: 'Free',
    price: 0,
    priceLabel: '$0',
    interval: 'forever',
    features: [
      '10 simulations per day',
      '5 saved simulations',
      'Basic visualizations',
      'Community support',
    ],
    cta: 'Get Started',
    href: '/signup',
  },
  {
    id: 'starter',
    tier: 'starter',
    name: 'Starter',
    price: 19,
    priceLabel: '$19',
    interval: 'month',
    features: [
      '1,000 API calls/month',
      '3 team seats',
      'REST API access',
      'Basic webhooks',
      'Email support',
    ],
    cta: 'Subscribe',
    popular: true,
  },
  {
    id: 'pro',
    tier: 'pro',
    name: 'Pro',
    price: 49,
    priceLabel: '$49',
    interval: 'month',
    features: [
      '10,000 API calls/month',
      '10 team seats',
      'REST + WebSocket APIs',
      'Real-time streaming',
      'Advanced webhooks',
      'Priority support',
    ],
    cta: 'Subscribe',
  },
  {
    id: 'enterprise',
    tier: 'enterprise',
    name: 'Enterprise',
    price: null, // Hidden
    priceLabel: 'Custom',
    interval: null,
    features: [
      'Unlimited API calls',
      'Unlimited team seats',
      'Full API suite',
      'Custom integrations',
      'SLA guarantee',
      'Dedicated support',
      'On-premise option',
    ],
    cta: 'Contact Sales',
    href: 'mailto:contact@crystallinelabs.io?subject=LatticeForge%20Enterprise%20Inquiry',
  },
];

export default function PricingPage() {
  const router = useRouter();
  const [loading, setLoading] = useState<string | null>(null);

  const handleSelectPlan = async (plan: (typeof VISIBLE_PLANS)[0]) => {
    // Direct link (free or enterprise)
    if (plan.href) {
      if (plan.href.startsWith('mailto:')) {
        window.location.href = plan.href;
      } else {
        router.push(plan.href);
      }
      return;
    }

    // Stripe checkout for paid plans
    setLoading(plan.id);

    try {
      const res = await fetch('/api/checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ planId: plan.id }),
      });

      const { url, error } = await res.json();

      if (error) {
        alert(error);
        return;
      }

      if (url) {
        window.location.href = url;
      }
    } catch {
      alert('Failed to create checkout session');
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 py-20 px-4">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-slate-950/80 backdrop-blur-sm border-b border-slate-800">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <button
            onClick={() => router.push('/app')}
            className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back to App
          </button>
          <button
            onClick={() => router.push('/')}
            className="text-xl font-bold text-white hover:text-blue-400 transition-colors"
          >
            LatticeForge
          </button>
          <button
            onClick={() => router.push('/app')}
            className="text-slate-400 hover:text-white transition-colors"
          >
            Dashboard
          </button>
        </div>
      </nav>

      <div className="max-w-5xl mx-auto pt-8">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-4xl font-bold text-white mb-4">Simple, transparent pricing</h1>
          <p className="text-xl text-slate-400 max-w-2xl mx-auto">
            Start free, scale as you grow. No hidden fees, cancel anytime.
          </p>
        </div>

        {/* Plans grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {VISIBLE_PLANS.map((plan) => (
            <div
              key={plan.id}
              className={`relative bg-slate-900 rounded-2xl border ${
                plan.popular ? 'border-blue-500' : 'border-slate-800'
              } p-6 flex flex-col`}
            >
              {plan.popular && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-blue-600 text-white text-xs font-medium rounded-full">
                  Most Popular
                </div>
              )}

              {/* Tier Badge */}
              <div className="flex justify-center mb-4">
                <TierBadge tier={plan.tier} size="lg" />
              </div>

              <div className="mb-6 text-center">
                <h3 className="text-xl font-bold text-white">{plan.name}</h3>
                <div className="mt-4">
                  <span className="text-4xl font-bold text-white">{plan.priceLabel}</span>
                  {plan.interval && <span className="text-slate-400">/{plan.interval}</span>}
                </div>
              </div>

              <ul className="space-y-3 mb-8 flex-1">
                {plan.features.map((feature, i) => (
                  <li key={i} className="flex items-start gap-3">
                    <svg
                      className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                    <span className="text-slate-300 text-sm">{feature}</span>
                  </li>
                ))}
              </ul>

              <button
                onClick={() => void handleSelectPlan(plan)}
                disabled={loading === plan.id}
                className={`w-full py-3 rounded-lg font-medium transition-colors ${
                  plan.popular
                    ? 'bg-blue-600 text-white hover:bg-blue-500'
                    : plan.id === 'enterprise'
                      ? 'bg-gradient-to-r from-slate-800 to-slate-700 text-white hover:from-slate-700 hover:to-slate-600 border border-slate-600'
                      : 'bg-slate-800 text-white hover:bg-slate-700'
                } ${loading === plan.id ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {loading === plan.id ? 'Loading...' : plan.cta}
              </button>
            </div>
          ))}
        </div>

        {/* FAQ */}
        <div className="mt-20">
          <h2 className="text-2xl font-bold text-white text-center mb-12">
            Frequently Asked Questions
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            <div>
              <h3 className="text-lg font-medium text-white mb-2">
                Can I upgrade or downgrade anytime?
              </h3>
              <p className="text-slate-400">
                Yes, you can change your plan at any time. Upgrades are prorated, and downgrades
                take effect at the next billing cycle.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-medium text-white mb-2">
                What payment methods do you accept?
              </h3>
              <p className="text-slate-400">
                We accept all major credit cards via Stripe. Enterprise customers can pay via
                invoice.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-medium text-white mb-2">
                What happens if I exceed my API limit?
              </h3>
              <p className="text-slate-400">
                You'll receive a warning at 80% usage. If you exceed your limit, API calls will be
                rate-limited until the next billing cycle.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-medium text-white mb-2">
                Do you offer discounts for annual billing?
              </h3>
              <p className="text-slate-400">
                Yes, annual plans get 2 months free. Contact sales for details.
              </p>
            </div>
          </div>
        </div>

        {/* CTA */}
        <div className="mt-20 text-center">
          <p className="text-slate-400 mb-4">Need a custom plan for your organization?</p>
          <a
            href="mailto:contact@crystallinelabs.io?subject=LatticeForge%20Custom%20Plan"
            className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300"
          >
            Contact our sales team
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M17 8l4 4m0 0l-4 4m4-4H3"
              />
            </svg>
          </a>
        </div>
      </div>
    </div>
  );
}
