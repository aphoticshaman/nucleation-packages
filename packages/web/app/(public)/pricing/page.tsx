'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import TierBadge, { TierType } from '@/components/TierBadge';
import { Check, ArrowLeft, Zap, Shield, Globe, Users } from 'lucide-react';

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
    price: null,
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

const FAQ = [
  {
    q: 'Can I upgrade or downgrade anytime?',
    a: 'Yes, you can change your plan at any time. Upgrades are prorated, and downgrades take effect at the next billing cycle.',
  },
  {
    q: 'What payment methods do you accept?',
    a: 'We accept all major credit cards via Stripe. Enterprise customers can pay via invoice.',
  },
  {
    q: 'What happens if I exceed my API limit?',
    a: "You'll receive a warning at 80% usage. If you exceed your limit, API calls will be rate-limited until the next billing cycle.",
  },
  {
    q: 'Do you offer discounts for annual billing?',
    a: 'Yes, annual plans get 2 months free. Contact sales for details.',
  },
];

export default function PricingPage() {
  const router = useRouter();
  const [loading, setLoading] = useState<string | null>(null);

  const handleSelectPlan = async (plan: (typeof VISIBLE_PLANS)[0]) => {
    if (plan.href) {
      if (plan.href.startsWith('mailto:')) {
        window.location.href = plan.href;
      } else {
        router.push(plan.href);
      }
      return;
    }

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
    <div className="min-h-screen bg-[#0a0a0f] relative">
      {/* Atmospheric background */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        {/* Base gradient */}
        <div
          className="absolute inset-0"
          style={{
            background: 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(59, 130, 246, 0.15) 0%, transparent 50%)',
          }}
        />
        {/* Grid pattern */}
        <div
          className="absolute inset-0 opacity-30"
          style={{
            backgroundImage: `
              linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
              linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px)
            `,
            backgroundSize: '60px 60px',
            maskImage: 'radial-gradient(ellipse at center, black 0%, transparent 70%)',
            WebkitMaskImage: 'radial-gradient(ellipse at center, black 0%, transparent 70%)',
          }}
        />
        {/* Obsidian texture */}
        <div
          className="absolute inset-0 opacity-20 mix-blend-overlay"
          style={{
            backgroundImage: 'url(/images/bg/obsidian.png)',
            backgroundSize: 'cover',
            backgroundPosition: 'center',
          }}
        />
      </div>

      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-[rgba(10,10,15,0.8)] backdrop-blur-xl border-b border-white/[0.06]">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <button
            onClick={() => router.push('/app')}
            className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors min-h-[44px] px-2 -ml-2"
          >
            <ArrowLeft className="w-5 h-5" />
            <span className="hidden sm:inline">Back to App</span>
          </button>
          <button
            onClick={() => router.push('/')}
            className="flex items-center gap-2"
          >
            <Image
              src="/images/brand/monogram.png"
              alt="LatticeForge"
              width={32}
              height={32}
            />
            <span className="text-xl font-bold text-white hover:text-blue-400 transition-colors">
              LatticeForge
            </span>
          </button>
          <button
            onClick={() => router.push('/app')}
            className="text-slate-400 hover:text-white transition-colors min-h-[44px] px-3"
          >
            Dashboard
          </button>
        </div>
      </nav>

      <div className="relative z-10 max-w-6xl mx-auto px-4 pt-28 pb-20">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-white mb-4">
            Simple, transparent pricing
          </h1>
          <p className="text-lg sm:text-xl text-slate-400 max-w-2xl mx-auto">
            Start free, scale as you grow. No hidden fees, cancel anytime.
          </p>
        </div>

        {/* Plans grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 mb-20">
          {VISIBLE_PLANS.map((plan) => (
            <div
              key={plan.id}
              className={`relative bg-[rgba(18,18,26,0.7)] backdrop-blur-xl rounded-2xl border ${
                plan.popular ? 'border-blue-500/50' : 'border-white/[0.06]'
              } p-6 flex flex-col transition-all hover:border-white/[0.12] hover:bg-[rgba(18,18,26,0.8)]`}
            >
              {plan.popular && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-4 py-1.5 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-xs font-medium rounded-full shadow-lg shadow-blue-500/25">
                  Most Popular
                </div>
              )}

              {/* Tier Badge */}
              <div className="flex justify-center mb-4 pt-2">
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
                    <Check className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                    <span className="text-slate-300 text-sm">{feature}</span>
                  </li>
                ))}
              </ul>

              <button
                onClick={() => void handleSelectPlan(plan)}
                disabled={loading === plan.id}
                className={`w-full py-3.5 min-h-[52px] rounded-xl font-medium transition-all touch-manipulation ${
                  plan.popular
                    ? 'bg-gradient-to-r from-blue-600 to-cyan-500 text-white hover:shadow-[0_0_30px_rgba(59,130,246,0.4)] active:scale-[0.98]'
                    : plan.id === 'enterprise'
                      ? 'bg-gradient-to-r from-amber-500/20 to-orange-500/20 text-amber-300 border border-amber-500/30 hover:border-amber-500/50 active:scale-[0.98]'
                      : 'bg-white/[0.06] text-white border border-white/[0.08] hover:bg-white/[0.1] active:scale-[0.98]'
                } ${loading === plan.id ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {loading === plan.id ? 'Loading...' : plan.cta}
              </button>
            </div>
          ))}
        </div>

        {/* Features highlight */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-20">
          {[
            { icon: Zap, title: 'Instant Setup', desc: 'Get started in minutes' },
            { icon: Shield, title: 'Bank-level Security', desc: 'AES-256 encryption' },
            { icon: Globe, title: '195 Countries', desc: 'Global coverage' },
            { icon: Users, title: 'Team Collaboration', desc: 'Share insights' },
          ].map((item, idx) => (
            <div
              key={idx}
              className="bg-[rgba(18,18,26,0.5)] backdrop-blur-sm rounded-xl border border-white/[0.06] p-4 text-center"
            >
              <item.icon className="w-8 h-8 text-blue-400 mx-auto mb-3" />
              <h4 className="text-white font-medium text-sm">{item.title}</h4>
              <p className="text-slate-500 text-xs mt-1">{item.desc}</p>
            </div>
          ))}
        </div>

        {/* FAQ */}
        <div className="mb-20">
          <h2 className="text-2xl font-bold text-white text-center mb-12">
            Frequently Asked Questions
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">
            {FAQ.map((item, idx) => (
              <div
                key={idx}
                className="bg-[rgba(18,18,26,0.5)] backdrop-blur-sm rounded-xl border border-white/[0.06] p-5"
              >
                <h3 className="text-base font-medium text-white mb-2">{item.q}</h3>
                <p className="text-slate-400 text-sm leading-relaxed">{item.a}</p>
              </div>
            ))}
          </div>
        </div>

        {/* CTA */}
        <div className="text-center">
          <p className="text-slate-400 mb-4">Need a custom plan for your organization?</p>
          <a
            href="mailto:contact@crystallinelabs.io?subject=LatticeForge%20Custom%20Plan"
            className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 font-medium min-h-[44px]"
          >
            Contact our sales team
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </a>
        </div>
      </div>
    </div>
  );
}
