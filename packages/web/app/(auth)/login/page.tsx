'use client';

import { useState, Suspense } from 'react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { supabase } from '@/lib/supabase';
import { Shield, Zap, Brain, Target, Globe, TrendingUp } from 'lucide-react';

function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const redirect = searchParams.get('redirect') || '/app';

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    const { error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });

    if (error) {
      setError(error.message);
      setLoading(false);
      return;
    }

    router.push(redirect);
  };

  const handleOAuthLogin = async (provider: 'google' | 'github') => {
    const { error } = await supabase.auth.signInWithOAuth({
      provider,
      options: {
        redirectTo: `${window.location.origin}/auth/callback?redirect=${redirect}`,
      },
    });

    if (error) {
      setError(error.message);
    }
  };

  return (
    <div className="w-full max-w-md relative z-10">
      {/* Logo */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-white">LatticeForge</h1>
        <p className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400 mt-2">
          Know what happens next.
        </p>
        <p className="text-slate-500 text-sm mt-1">
          AI-powered geopolitical intelligence
        </p>
      </div>

      {/* Form */}
      <div className="bg-slate-900/90 backdrop-blur-xl rounded-xl border border-slate-700/50 p-8 shadow-2xl">
        {error && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-800 rounded-lg text-red-400 text-sm">
            {error}
          </div>
        )}

        {/* OAuth buttons */}
        <div className="space-y-3 mb-6">
          <button
            onClick={() => void handleOAuthLogin('google')}
            className="w-full flex items-center justify-center gap-3 px-4 py-3 bg-white text-slate-900 rounded-lg font-medium hover:bg-slate-100 transition-colors"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24">
              <path
                fill="currentColor"
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
              />
              <path
                fill="currentColor"
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
              />
              <path
                fill="currentColor"
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
              />
              <path
                fill="currentColor"
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
              />
            </svg>
            Continue with Google
          </button>

          <button
            onClick={() => void handleOAuthLogin('github')}
            className="w-full flex items-center justify-center gap-3 px-4 py-3 bg-slate-800 text-white rounded-lg font-medium hover:bg-slate-700 transition-colors"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <path
                fillRule="evenodd"
                d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                clipRule="evenodd"
              />
            </svg>
            Continue with GitHub
          </button>
        </div>

        <div className="relative mb-6">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-slate-700" />
          </div>
          <div className="relative flex justify-center text-sm">
            <span className="px-2 bg-slate-900 text-slate-500">or</span>
          </div>
        </div>

        {/* Email form */}
        <form onSubmit={(e) => void handleLogin(e)} className="space-y-4">
          <div>
            <label className="block text-sm text-slate-400 mb-2">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              placeholder="you@example.com"
              required
            />
          </div>

          <div>
            <label className="block text-sm text-slate-400 mb-2">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              placeholder="••••••••"
              required
            />
          </div>

          <div className="flex items-center justify-between text-sm">
            <label className="flex items-center gap-2 text-slate-400">
              <input type="checkbox" className="rounded" />
              Remember me
            </label>
            <a href="/forgot-password" className="text-blue-400 hover:text-blue-300">
              Forgot password?
            </a>
          </div>

          <button
            type="submit"
            disabled={loading}
            className={`w-full py-3 rounded-lg font-medium transition-all ${
              loading
                ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white hover:from-blue-500 hover:to-cyan-500 shadow-lg shadow-blue-500/25'
            }`}
          >
            {loading ? 'Signing in...' : 'Sign in'}
          </button>
        </form>
      </div>

      {/* Sign up link */}
      <p className="text-center text-slate-400 mt-6">
        Don&apos;t have an account?{' '}
        <Link href="/signup" className="text-blue-400 hover:text-blue-300">
          Sign up free
        </Link>
      </p>
    </div>
  );
}

function LoginFallback() {
  return (
    <div className="w-full max-w-md animate-pulse relative z-10">
      <div className="text-center mb-8">
        <div className="h-9 bg-slate-800 rounded w-48 mx-auto mb-2"></div>
        <div className="h-5 bg-slate-800 rounded w-40 mx-auto"></div>
      </div>
      <div className="bg-slate-900/90 backdrop-blur-xl rounded-xl border border-slate-700/50 p-8">
        <div className="space-y-3 mb-6">
          <div className="h-12 bg-slate-800 rounded-lg"></div>
          <div className="h-12 bg-slate-800 rounded-lg"></div>
        </div>
      </div>
    </div>
  );
}

// Marketing content
const DIFFERENTIATORS = [
  {
    icon: Brain,
    title: 'Cognitive Intelligence',
    description: 'AI that thinks like an analyst, not a search engine. Context-aware reasoning across 200+ dimensions.',
  },
  {
    icon: Target,
    title: '72% Predictive Accuracy',
    description: 'Detect phase transitions 30 days before they happen. Early warning when it matters most.',
  },
  {
    icon: Globe,
    title: 'Global Coverage',
    description: 'Real-time monitoring across 195 countries. Multi-lingual OSINT from 10,000+ sources.',
  },
  {
    icon: Zap,
    title: 'Instant Briefings',
    description: 'Executive summaries on-demand. From raw data to actionable intelligence in seconds.',
  },
  {
    icon: TrendingUp,
    title: 'Temporal Analysis',
    description: 'Navigate intel across time. Historical patterns, current state, projected futures.',
  },
  {
    icon: Shield,
    title: 'Enterprise Security',
    description: 'AES-256 encryption at rest. Your data never trains our models.',
  },
];

export default function LoginPage() {
  return (
    <div className="min-h-screen bg-slate-950 flex">
      {/* Left side - Hero visual with form overlay */}
      <div className="flex-1 relative flex items-center justify-center px-8 py-12">
        {/* Background image */}
        <div className="absolute inset-0 z-0">
          <img
            src="/images/hero/simulation-globe.png"
            alt=""
            className="w-full h-full object-cover opacity-30"
          />
          <div className="absolute inset-0 bg-gradient-to-r from-slate-950 via-slate-950/80 to-slate-950/60" />
        </div>

        {/* Animated gradient accent */}
        <div className="absolute top-1/4 -left-32 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 left-1/4 w-64 h-64 bg-cyan-500/10 rounded-full blur-3xl animate-pulse delay-1000" />

        {/* Form */}
        <Suspense fallback={<LoginFallback />}>
          <LoginForm />
        </Suspense>
      </div>

      {/* Right side - Marketing content (hidden on mobile) */}
      <div className="hidden lg:flex flex-1 flex-col justify-center p-12 xl:p-16 bg-slate-900/50 border-l border-slate-800">
        <div className="max-w-xl">
          {/* Headline */}
          <h2 className="text-3xl xl:text-4xl font-bold text-white mb-4">
            Intelligence that sees
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">
              {' '}around corners
            </span>
          </h2>
          <p className="text-lg text-slate-400 mb-10">
            When headlines are too late and hunches aren&apos;t enough, LatticeForge gives you the
            analytical edge. Understand what&apos;s happening, why it matters, and what comes next.
          </p>

          {/* Feature grid */}
          <div className="grid grid-cols-2 gap-6">
            {DIFFERENTIATORS.map((item, idx) => (
              <div key={idx} className="group">
                <div className="flex items-start gap-3">
                  <div className="p-2 bg-slate-800/50 rounded-lg group-hover:bg-blue-500/20 transition-colors">
                    <item.icon className="w-5 h-5 text-blue-400" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white text-sm">{item.title}</h3>
                    <p className="text-xs text-slate-500 mt-1 leading-relaxed">{item.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Social proof placeholder */}
          <div className="mt-10 pt-8 border-t border-slate-800">
            <p className="text-sm text-slate-500 italic">
              &ldquo;We&apos;ll have a customer quote here soon. For now, know that we built this
              because existing tools weren&apos;t cutting it.&rdquo;
            </p>
            <p className="text-xs text-slate-600 mt-2">- The LatticeForge Team</p>
          </div>

          {/* Trust indicators */}
          <div className="flex items-center gap-6 mt-8">
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <Shield className="w-4 h-4" />
              <span>Encrypted</span>
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <Globe className="w-4 h-4" />
              <span>195 Countries</span>
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <Zap className="w-4 h-4" />
              <span>Real-time</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
