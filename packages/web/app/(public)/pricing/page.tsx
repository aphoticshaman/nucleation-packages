'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import TierBadge, { TierType } from '@/components/TierBadge';
import { Check, ArrowLeft, Zap, Shield, Globe, Users, Crown } from 'lucide-react';
import { Card, Button } from '@/components/ui';
import { supabase } from '@/lib/supabase';

interface UserStatus {
  isLoggedIn: boolean;
  role?: string;
  tier?: string;
  orgPlan?: string;
  canPurchase: boolean;
}

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
    href: 'mailto:contact@latticeforge.ai?subject=LatticeForge%20Enterprise%20Inquiry',
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
  const [userStatus, setUserStatus] = useState<UserStatus>({
    isLoggedIn: false,
    canPurchase: true,
  });

  useEffect(() => {
    async function checkUserStatus() {
      const { data: { user } } = await supabase.auth.getUser();

      if (!user) {
        setUserStatus({ isLoggedIn: false, canPurchase: true });
        return;
      }

      const { data: profile } = await supabase
        .from('profiles')
        .select('role, organizations(plan)')
        .eq('id', user.id)
        .single() as { data: { role?: string; organizations?: { plan?: string } | null } | null };

      const role = profile?.role || 'consumer';
      const orgPlan = profile?.organizations?.plan || 'free';

      const blockedRoles = ['admin', 'enterprise', 'support'];
      const canPurchase = !blockedRoles.includes(role) && orgPlan !== 'enterprise';

      setUserStatus({
        isLoggedIn: true,
        role,
        orgPlan,
        canPurchase,
      });
    }

    void checkUserStatus();
  }, []);

  const handleSelectPlan = async (plan: (typeof VISIBLE_PLANS)[0]) => {
    if (!userStatus.canPurchase && userStatus.isLoggedIn) {
      alert('Your account already has full access. No purchase required.');
      return;
    }

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

  const getAccessLevelText = () => {
    if (!userStatus.isLoggedIn) return null;
    if (userStatus.role === 'admin') return 'Administrator';
    if (userStatus.role === 'enterprise') return 'Enterprise';
    if (userStatus.role === 'support') return 'Support Staff';
    if (userStatus.orgPlan === 'enterprise') return 'Enterprise Plan';
    return null;
  };

  return (
    <div className="min-h-screen bg-slate-950 relative">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-slate-900 border-b border-slate-800">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <button
            onClick={() => router.push('/app')}
            className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
            <span className="hidden sm:inline text-sm">Back</span>
          </button>
          <button
            onClick={() => router.push('/')}
            className="flex items-center gap-2"
          >
            <Image
              src="/images/brand/monogram.png"
              alt="LatticeForge"
              width={28}
              height={28}
            />
            <span className="text-base font-semibold text-white">
              LatticeForge
            </span>
          </button>
          <button
            onClick={() => router.push('/app')}
            className="text-slate-400 hover:text-white transition-colors text-sm"
          >
            Dashboard
          </button>
        </div>
      </nav>

      <div className="relative z-10 max-w-6xl mx-auto px-4 pt-24 pb-16">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-2xl sm:text-3xl font-bold text-white mb-3">
            Simple, transparent pricing
          </h1>
          <p className="text-base text-slate-400 max-w-xl mx-auto">
            Start free, scale as you grow. No hidden fees.
          </p>
        </div>

        {/* Full access banner */}
        {!userStatus.canPurchase && userStatus.isLoggedIn && (
          <Card padding="md" className="mb-10 border-amber-600/30 bg-amber-600/5">
            <div className="flex items-center gap-4 text-center sm:text-left flex-col sm:flex-row">
              <div className="flex-shrink-0 p-3 rounded-md bg-amber-600/20">
                <Crown className="w-6 h-6 text-amber-500" />
              </div>
              <div className="flex-1">
                <h3 className="text-base font-semibold text-white mb-1">
                  Full access enabled
                </h3>
                <p className="text-sm text-slate-400">
                  As {getAccessLevelText()}, you have access to all features.
                </p>
              </div>
              <Button
                variant="secondary"
                onClick={() => router.push('/app')}
              >
                Go to Dashboard
              </Button>
            </div>
          </Card>
        )}

        {/* Plans grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 mb-16">
          {VISIBLE_PLANS.map((plan) => (
            <Card
              key={plan.id}
              padding="md"
              className={`relative flex flex-col ${
                plan.popular ? 'border-blue-600/50' : ''
              }`}
            >
              {plan.popular && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-blue-600 text-white text-[10px] font-medium rounded uppercase tracking-wide">
                  Popular
                </div>
              )}

              {/* Tier Badge */}
              <div className="flex justify-center mb-4 pt-2">
                <TierBadge tier={plan.tier} size="lg" />
              </div>

              <div className="mb-5 text-center">
                <h3 className="text-lg font-semibold text-white">{plan.name}</h3>
                <div className="mt-3">
                  <span className="text-3xl font-bold text-white">{plan.priceLabel}</span>
                  {plan.interval && <span className="text-slate-500 text-sm">/{plan.interval}</span>}
                </div>
              </div>

              <ul className="space-y-2.5 mb-6 flex-1">
                {plan.features.map((feature, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <Check className="w-4 h-4 text-emerald-500 flex-shrink-0 mt-0.5" />
                    <span className="text-slate-400 text-sm">{feature}</span>
                  </li>
                ))}
              </ul>

              <Button
                variant={plan.popular ? 'primary' : 'secondary'}
                onClick={() => void handleSelectPlan(plan)}
                disabled={loading === plan.id || (!userStatus.canPurchase && userStatus.isLoggedIn && !plan.href)}
                loading={loading === plan.id}
                className="w-full"
              >
                {!userStatus.canPurchase && userStatus.isLoggedIn && !plan.href ? 'Included' : plan.cta}
              </Button>
            </Card>
          ))}
        </div>

        {/* Features highlight */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-16">
          {[
            { icon: Zap, title: 'Instant Setup', desc: 'Get started in minutes' },
            { icon: Shield, title: 'Bank-level Security', desc: 'AES-256 encryption' },
            { icon: Globe, title: '195 Countries', desc: 'Global coverage' },
            { icon: Users, title: 'Team Collaboration', desc: 'Share insights' },
          ].map((item, idx) => (
            <Card key={idx} padding="sm" className="text-center">
              <item.icon className="w-6 h-6 text-slate-400 mx-auto mb-2" />
              <h4 className="text-white font-medium text-sm">{item.title}</h4>
              <p className="text-slate-500 text-xs mt-0.5">{item.desc}</p>
            </Card>
          ))}
        </div>

        {/* FAQ */}
        <div className="mb-16">
          <h2 className="text-lg font-semibold text-white text-center mb-8">
            Frequently Asked Questions
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-4xl mx-auto">
            {FAQ.map((item, idx) => (
              <Card key={idx} padding="md">
                <h3 className="text-sm font-medium text-white mb-2">{item.q}</h3>
                <p className="text-slate-500 text-xs leading-relaxed">{item.a}</p>
              </Card>
            ))}
          </div>
        </div>

        {/* CTA */}
        <div className="text-center">
          <p className="text-slate-500 text-sm mb-3">Need a custom plan?</p>
          <a
            href="mailto:contact@latticeforge.ai?subject=LatticeForge%20Custom%20Plan"
            className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 text-sm font-medium"
          >
            Contact sales
            <ArrowLeft className="w-3.5 h-3.5 rotate-180" />
          </a>
        </div>
      </div>
    </div>
  );
}
