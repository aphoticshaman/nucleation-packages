'use client';

import { useState, Suspense } from 'react';
import Link from 'next/link';
import Image from 'next/image';
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
    <div className="w-full max-w-md relative z-10 px-4 sm:px-0">
      {/* Logo + Tagline */}
      <div className="text-center mb-6 sm:mb-8">
        {/* Monogram - visible on mobile */}
        <div className="flex justify-center mb-4 lg:hidden">
          <Image
            src="/images/brand/monogram.png"
            alt="LatticeForge"
            width={48}
            height={48}
            className="w-12 h-12"
          />
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold text-white">LatticeForge</h1>
        <p className="text-base sm:text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-cyan-400 to-orange-400 mt-2">
          Know what happens next.
        </p>
        <p className="text-slate-500 text-xs sm:text-sm mt-1">
          AI-powered geopolitical intelligence
        </p>
      </div>

      {/* Glass form card */}
      <div className="bg-[rgba(18,18,26,0.8)] backdrop-blur-xl rounded-2xl border border-white/[0.08] p-6 sm:p-8 shadow-2xl">
        {error && (
          <div className="mb-5 p-4 bg-red-500/20 border border-red-500/30 rounded-xl text-red-300 text-sm">
            {error}
          </div>
        )}

        {/* OAuth buttons */}
        <div className="space-y-3 mb-5">
          <button
            onClick={() => void handleOAuthLogin('google')}
            className="w-full flex items-center justify-center gap-3 px-4 py-3.5 min-h-[52px]
              bg-white text-slate-900 rounded-xl font-medium
              hover:bg-slate-100 active:scale-[0.98]
              transition-all touch-manipulation"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24">
              <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
              <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
              <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
              <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
            </svg>
            Continue with Google
          </button>

          <button
            onClick={() => void handleOAuthLogin('github')}
            className="w-full flex items-center justify-center gap-3 px-4 py-3.5 min-h-[52px]
              bg-[rgba(255,255,255,0.06)] text-white rounded-xl font-medium
              border border-white/[0.08]
              hover:bg-[rgba(255,255,255,0.1)] active:scale-[0.98]
              transition-all touch-manipulation"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
            </svg>
            Continue with GitHub
          </button>
        </div>

        <div className="relative mb-5">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-white/[0.08]" />
          </div>
          <div className="relative flex justify-center text-xs">
            <span className="px-3 bg-[rgba(18,18,26,0.8)] text-slate-500">or sign in with email</span>
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
              className="w-full px-4 py-3.5 min-h-[52px] text-base
                bg-black/30 border border-white/[0.08] rounded-xl text-white
                placeholder:text-slate-500
                focus:outline-none focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20
                transition-all"
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
              className="w-full px-4 py-3.5 min-h-[52px] text-base
                bg-black/30 border border-white/[0.08] rounded-xl text-white
                placeholder:text-slate-500
                focus:outline-none focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20
                transition-all"
              placeholder="••••••••"
              required
            />
          </div>

          <div className="flex items-center justify-between text-sm">
            <label className="flex items-center gap-2 text-slate-400 cursor-pointer min-h-[44px]">
              <input type="checkbox" className="rounded bg-black/30 border-white/20" />
              Remember me
            </label>
            <Link href="/forgot-password" className="text-blue-400 hover:text-blue-300 min-h-[44px] flex items-center">
              Forgot password?
            </Link>
          </div>

          <button
            type="submit"
            disabled={loading}
            className={`w-full py-3.5 min-h-[52px] rounded-xl font-medium transition-all touch-manipulation ${
              loading
                ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-blue-600 via-blue-500 to-cyan-500 text-white hover:shadow-[0_0_30px_rgba(59,130,246,0.4)] active:scale-[0.98]'
            }`}
          >
            {loading ? 'Signing in...' : 'Sign in'}
          </button>
        </form>
      </div>

      {/* Sign up link */}
      <p className="text-center text-slate-400 mt-6 text-sm">
        Don&apos;t have an account?{' '}
        <Link href="/signup" className="text-blue-400 hover:text-blue-300 font-medium">
          Sign up free
        </Link>
      </p>
    </div>
  );
}

function LoginFallback() {
  return (
    <div className="w-full max-w-md animate-pulse relative z-10 px-4 sm:px-0">
      <div className="text-center mb-8">
        <div className="h-9 bg-white/10 rounded w-48 mx-auto mb-2" />
        <div className="h-5 bg-white/10 rounded w-40 mx-auto" />
      </div>
      <div className="bg-[rgba(18,18,26,0.8)] backdrop-blur-xl rounded-2xl border border-white/[0.08] p-8">
        <div className="space-y-3 mb-6">
          <div className="h-14 bg-white/10 rounded-xl" />
          <div className="h-14 bg-white/10 rounded-xl" />
        </div>
      </div>
    </div>
  );
}

const DIFFERENTIATORS = [
  { icon: Brain, title: 'AI Analysis', desc: 'Reads thousands of news sources so you don\'t have to.' },
  { icon: Target, title: '72% Accurate', desc: 'Catches 7 out of 10 major changes a month early.' },
  { icon: Globe, title: '195 Countries', desc: 'Every nation, every language, every day.' },
  { icon: Zap, title: 'Daily Briefings', desc: 'Morning summary of what matters right now.' },
  { icon: TrendingUp, title: 'Spot Trends', desc: 'See where things are heading before headlines break.' },
  { icon: Shield, title: 'Private & Secure', desc: 'Your data stays yours. Always encrypted.' },
];

export default function LoginPage() {
  return (
    <div className="min-h-screen bg-[#0a0a0f] flex flex-col lg:flex-row">
      {/* Background effects - visible on all screens */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        {/* Grid pattern */}
        <div
          className="absolute inset-0 opacity-20"
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
      </div>

      {/* Left side - Form */}
      <div className="flex-1 relative flex items-center justify-center py-8 sm:py-12 lg:py-16 min-h-screen lg:min-h-0">
        {/* Hero wireframe sphere - background on mobile */}
        <div className="absolute inset-0 lg:hidden overflow-hidden">
          {/* Orange glow behind sphere */}
          <div
            className="absolute top-[30%] left-1/2 -translate-x-1/2 -translate-y-1/2 w-[350px] h-[350px] rounded-full"
            style={{ background: 'radial-gradient(circle, rgba(251,146,60,0.4) 0%, rgba(251,146,60,0.15) 40%, transparent 70%)' }}
          />
          {/* Wireframe sphere */}
          <div className="absolute top-[30%] left-1/2 -translate-x-1/2 -translate-y-1/2 w-[320px] h-[320px] opacity-50">
            <Image
              src="/images/hero/wireframe-sphere.png"
              alt=""
              fill
              className="object-contain"
              priority
            />
          </div>
        </div>

        {/* Blue accent glow - top left corner */}
        <div className="absolute -top-32 -left-32 w-[500px] h-[500px] opacity-25 pointer-events-none">
          <div className="w-full h-full rounded-full bg-blue-600/40 blur-[120px]" />
        </div>

        <Suspense fallback={<LoginFallback />}>
          <LoginForm />
        </Suspense>
      </div>

      {/* Right side - Hero visual (desktop only) */}
      <div className="hidden lg:flex flex-1 relative items-center justify-center bg-[rgba(10,10,15,0.5)] border-l border-white/[0.05] overflow-hidden">
        {/* Orange glow - core effect */}
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-[55%] w-[500px] h-[500px] xl:w-[600px] xl:h-[600px] rounded-full pointer-events-none"
          style={{
            background: 'radial-gradient(circle, rgba(251,146,60,0.5) 0%, rgba(251,146,60,0.2) 30%, transparent 60%)',
          }}
        />
        {/* Blue accent glow */}
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-[55%] w-[600px] h-[600px] xl:w-[700px] xl:h-[700px] rounded-full pointer-events-none"
          style={{
            background: 'radial-gradient(circle at 30% 70%, rgba(59,130,246,0.15) 0%, transparent 50%)',
          }}
        />
        {/* Large wireframe sphere hero */}
        <div className="relative w-[500px] h-[500px] xl:w-[600px] xl:h-[600px] -mt-16">
          <Image
            src="/images/hero/wireframe-sphere.png"
            alt="LatticeForge Intelligence Network"
            fill
            className="object-contain drop-shadow-2xl"
            priority
          />
        </div>

        {/* Marketing content overlay */}
        <div className="absolute bottom-0 left-0 right-0 p-8 xl:p-12 bg-gradient-to-t from-[rgba(10,10,15,0.95)] via-[rgba(10,10,15,0.8)] to-transparent">
          <div className="max-w-xl">
            <h2 className="text-2xl xl:text-3xl font-bold text-white mb-3">
              See what&apos;s coming
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-orange-400 via-amber-400 to-yellow-400">
                {' '}before it hits the news
              </span>
            </h2>
            <p className="text-slate-400 text-sm mb-6">
              AI watches the world so you can make better decisions.
            </p>

            {/* Feature pills */}
            <div className="flex flex-wrap gap-2">
              {DIFFERENTIATORS.map((item, idx) => (
                <div
                  key={idx}
                  className="flex items-center gap-2 px-3 py-2 bg-white/[0.05] backdrop-blur-sm rounded-lg border border-white/[0.08]"
                >
                  <item.icon className="w-4 h-4 text-orange-400" />
                  <span className="text-xs text-slate-300">{item.title}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Logo in corner */}
        <div className="absolute top-8 right-8">
          <Image
            src="/images/brand/monogram.png"
            alt="LatticeForge"
            width={40}
            height={40}
            className="opacity-60"
          />
        </div>
      </div>
    </div>
  );
}
