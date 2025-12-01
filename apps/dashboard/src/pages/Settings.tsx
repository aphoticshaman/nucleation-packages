import { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import type { Session } from '@supabase/supabase-js';
import { SettingsIcon, CheckIcon, AlertIcon } from '../components/Icons';

interface SettingsProps {
  session: Session | null;
  apiKey: string | null;
}

const TIERS = {
  free: {
    name: 'Free',
    price: '$0/month',
    features: ['5,000 API calls/month', 'World Bank + CIA Factbook data', 'Quick briefs (Haiku)', 'Basic anomaly detection'],
  },
  pro: {
    name: 'Pro',
    price: '$49/month',
    features: ['50,000 API calls/month', 'All Free features', 'FRED economic data', 'US deep dive briefs (Sonnet)', 'Priority email support'],
  },
  enterprise: {
    name: 'Enterprise',
    price: '$299/month',
    features: ['Unlimited API calls', 'All Pro features', 'Intel-style predictive briefs (Opus)', 'Real-time data feeds', 'Custom integrations', 'Dedicated support + SLA'],
  },
};

// Stripe not yet configured - set to true when products are created
const STRIPE_CONFIGURED = false;

export function Settings({ session, apiKey }: SettingsProps) {
  const [currentTier, setCurrentTier] = useState<'free' | 'pro' | 'enterprise'>('enterprise'); // Default to enterprise for admin
  const [currentRole, setCurrentRole] = useState<'admin' | 'support' | 'user'>('admin');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Fetch current tier and role from database
    const fetchUserData = async () => {
      if (!session) return;

      const { data } = await supabase
        .from('clients')
        .select('tier, role')
        .eq('user_id', session.user.id)
        .single();

      if (data?.tier) {
        setCurrentTier(data.tier);
      }
      if (data?.role) {
        setCurrentRole(data.role);
      }
    };

    fetchUserData();
  }, [session]);

  const handleUpgrade = async (tier: 'pro' | 'enterprise') => {
    if (!STRIPE_CONFIGURED) {
      alert('Billing not yet configured. Stripe products need to be created first.');
      return;
    }

    if (!session) {
      alert('Please sign in with email to manage billing');
      return;
    }

    setIsLoading(true);

    try {
      const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
      const response = await fetch(`${supabaseUrl}/functions/v1/billing/checkout`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.access_token}`,
        },
        body: JSON.stringify({
          price_id: `${tier}_monthly`,
          success_url: `${window.location.origin}/settings?success=true`,
          cancel_url: `${window.location.origin}/settings?canceled=true`,
        }),
      });

      const data = await response.json();

      if (data.url) {
        window.location.href = data.url;
      } else {
        alert(data.error || 'Failed to create checkout session');
      }
    } catch (err) {
      console.error('Checkout error:', err);
      alert('Failed to start checkout');
    } finally {
      setIsLoading(false);
    }
  };

  const handleManageBilling = async () => {
    if (!STRIPE_CONFIGURED) {
      alert('Billing not yet configured. Stripe products need to be created first.');
      return;
    }

    if (!session) {
      alert('Please sign in with email to manage billing');
      return;
    }

    setIsLoading(true);

    try {
      const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
      const response = await fetch(`${supabaseUrl}/functions/v1/billing/portal`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.access_token}`,
        },
        body: JSON.stringify({
          return_url: `${window.location.origin}/settings`,
        }),
      });

      const data = await response.json();

      if (data.url) {
        window.location.href = data.url;
      } else {
        alert(data.error || 'Failed to open billing portal');
      }
    } catch (err) {
      console.error('Portal error:', err);
      alert('Failed to open billing portal');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Settings</h1>
          <p className="text-sm text-lattice-400 mt-1">
            Manage your account and subscription
          </p>
        </div>
      </div>

      {/* Billing Notice */}
      {!STRIPE_CONFIGURED && (
        <div className="glass-card p-4 border-amber-500/30 bg-amber-500/5">
          <div className="flex items-start gap-3">
            <AlertIcon className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm text-amber-300 font-medium">Billing Not Yet Configured</p>
              <p className="text-xs text-surface-400 mt-1">
                Stripe products and prices need to be created before upgrades work.
                You currently have admin access with all Enterprise features enabled.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Account Info */}
      <div className="glass-card p-6">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <SettingsIcon className="w-5 h-5 text-lattice-400" />
          Account
        </h2>
        <div className="space-y-3">
          <div className="flex justify-between py-2 border-b border-surface-700">
            <span className="text-surface-400">Email</span>
            <span className="text-white">{session?.user?.email || 'API Key Access'}</span>
          </div>
          <div className="flex justify-between py-2 border-b border-surface-700">
            <span className="text-surface-400">Role</span>
            <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
              currentRole === 'admin' ? 'bg-red-500/20 text-red-400' :
              currentRole === 'support' ? 'bg-amber-500/20 text-amber-400' :
              'bg-surface-600/50 text-surface-300'
            }`}>
              {currentRole.charAt(0).toUpperCase() + currentRole.slice(1)}
            </span>
          </div>
          <div className="flex justify-between py-2 border-b border-surface-700">
            <span className="text-surface-400">Current Plan</span>
            <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
              currentTier === 'enterprise' ? 'bg-crystal-500/20 text-crystal-400' :
              currentTier === 'pro' ? 'bg-lattice-500/20 text-lattice-400' :
              'bg-surface-600/50 text-surface-300'
            }`}>
              {TIERS[currentTier].name}
            </span>
          </div>
          <div className="flex justify-between py-2">
            <span className="text-surface-400">API Key</span>
            <span className="text-white font-mono text-sm">
              {apiKey ? `${apiKey.slice(0, 15)}...` : 'Using session auth'}
            </span>
          </div>
        </div>

        {session && currentTier !== 'free' && STRIPE_CONFIGURED && (
          <button
            onClick={handleManageBilling}
            disabled={isLoading}
            className="mt-4 px-4 py-2 text-sm text-lattice-400 border border-lattice-500/30 rounded-lg hover:bg-lattice-500/10 transition-colors disabled:opacity-50"
          >
            Manage Billing
          </button>
        )}
      </div>

      {/* Roles Info */}
      <div className="glass-card p-6">
        <h2 className="text-lg font-semibold text-white mb-4">User Roles</h2>
        <div className="grid grid-cols-3 gap-4">
          <div className="p-4 rounded-lg bg-surface-800/50 border border-surface-700">
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-0.5 rounded text-xs font-semibold bg-red-500/20 text-red-400">Admin</span>
            </div>
            <p className="text-xs text-surface-400">Full access to all features, billing, user management, and system settings.</p>
          </div>
          <div className="p-4 rounded-lg bg-surface-800/50 border border-surface-700">
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-0.5 rounded text-xs font-semibold bg-amber-500/20 text-amber-400">Support</span>
            </div>
            <p className="text-xs text-surface-400">CSA access to view and assist users. Cannot modify billing or system settings.</p>
          </div>
          <div className="p-4 rounded-lg bg-surface-800/50 border border-surface-700">
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-0.5 rounded text-xs font-semibold bg-surface-600/50 text-surface-300">User</span>
            </div>
            <p className="text-xs text-surface-400">Standard customer access based on subscription tier (Free, Pro, or Enterprise).</p>
          </div>
        </div>
      </div>

      {/* Pricing */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-4">Plans & Pricing</h2>
        <div className="grid grid-cols-3 gap-4">
          {(Object.entries(TIERS) as [keyof typeof TIERS, typeof TIERS.free][]).map(([key, tier]) => (
            <div
              key={key}
              className={`glass-card p-6 relative ${
                currentTier === key ? 'border-lattice-500/50' : ''
              }`}
            >
              {currentTier === key && (
                <div className="absolute top-4 right-4">
                  <span className="px-2 py-0.5 rounded text-xs font-semibold bg-lattice-500/20 text-lattice-400">
                    Current
                  </span>
                </div>
              )}

              <h3 className="text-xl font-bold text-white">{tier.name}</h3>
              <p className="text-2xl font-bold text-lattice-400 mt-2">{tier.price}</p>

              <ul className="mt-4 space-y-2">
                {tier.features.map((feature, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-surface-300">
                    <CheckIcon className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                    {feature}
                  </li>
                ))}
              </ul>

              {key !== 'free' && currentTier === 'free' && session && (
                <button
                  onClick={() => handleUpgrade(key as 'pro' | 'enterprise')}
                  disabled={isLoading || !STRIPE_CONFIGURED}
                  className="w-full mt-6 btn-primary disabled:opacity-50"
                >
                  {!STRIPE_CONFIGURED ? 'Coming Soon' : isLoading ? 'Loading...' : `Upgrade to ${tier.name}`}
                </button>
              )}

              {key !== 'free' && !session && (
                <p className="mt-6 text-xs text-surface-500 text-center">
                  Sign in with email to upgrade
                </p>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
