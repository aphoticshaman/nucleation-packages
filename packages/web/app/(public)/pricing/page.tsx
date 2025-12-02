'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { PLANS, PlanId } from '@/lib/stripe';

const PLAN_ORDER: PlanId[] = ['free', 'starter', 'pro', 'enterprise'];

export default function PricingPage() {
  const router = useRouter();
  const [loading, setLoading] = useState<string | null>(null);

  const handleSelectPlan = async (planId: PlanId) => {
    if (planId === 'free') {
      router.push('/signup');
      return;
    }

    if (planId === 'enterprise') {
      // Contact sales for enterprise
      window.location.href = 'mailto:sales@latticeforge.io?subject=Enterprise%20Inquiry';
      return;
    }

    setLoading(planId);

    try {
      const res = await fetch('/api/checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ planId }),
      });

      const { url, error } = await res.json();

      if (error) {
        alert(error);
        return;
      }

      if (url) {
        window.location.href = url;
      }
    } catch (err) {
      alert('Failed to create checkout session');
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 py-20 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-4xl font-bold text-white mb-4">
            Simple, transparent pricing
          </h1>
          <p className="text-xl text-slate-400 max-w-2xl mx-auto">
            Start free, scale as you grow. No hidden fees, cancel anytime.
          </p>
        </div>

        {/* Plans grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {PLAN_ORDER.map((planId) => {
            const plan = PLANS[planId];
            const isPopular = 'popular' in plan && plan.popular;

            return (
              <div
                key={planId}
                className={`relative bg-slate-900 rounded-2xl border ${
                  isPopular ? 'border-blue-500' : 'border-slate-800'
                } p-8 flex flex-col`}
              >
                {isPopular && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-blue-600 text-white text-xs font-medium rounded-full">
                    Most Popular
                  </div>
                )}

                <div className="mb-6">
                  <h3 className="text-xl font-bold text-white">{plan.name}</h3>
                  <div className="mt-4">
                    <span className="text-4xl font-bold text-white">
                      ${plan.price}
                    </span>
                    {plan.price > 0 && (
                      <span className="text-slate-400">/{plan.interval}</span>
                    )}
                  </div>
                </div>

                <ul className="space-y-4 mb-8 flex-1">
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
                  onClick={() => handleSelectPlan(planId)}
                  disabled={loading === planId}
                  className={`w-full py-3 rounded-lg font-medium transition-colors ${
                    isPopular
                      ? 'bg-blue-600 text-white hover:bg-blue-500'
                      : planId === 'enterprise'
                      ? 'bg-slate-800 text-white hover:bg-slate-700 border border-slate-700'
                      : 'bg-slate-800 text-white hover:bg-slate-700'
                  } ${loading === planId ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  {loading === planId
                    ? 'Loading...'
                    : planId === 'free'
                    ? 'Get Started'
                    : planId === 'enterprise'
                    ? 'Contact Sales'
                    : 'Subscribe'}
                </button>
              </div>
            );
          })}
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
                Yes, you can change your plan at any time. Upgrades are prorated,
                and downgrades take effect at the next billing cycle.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-medium text-white mb-2">
                What payment methods do you accept?
              </h3>
              <p className="text-slate-400">
                We accept all major credit cards via Stripe. Enterprise customers
                can pay via invoice.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-medium text-white mb-2">
                What happens if I exceed my API limit?
              </h3>
              <p className="text-slate-400">
                You'll receive a warning at 80% usage. If you exceed your limit,
                API calls will be rate-limited until the next billing cycle.
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
          <p className="text-slate-400 mb-4">
            Need a custom plan for your organization?
          </p>
          <a
            href="mailto:sales@latticeforge.io"
            className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300"
          >
            Contact our sales team
            <svg
              className="w-4 h-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
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
