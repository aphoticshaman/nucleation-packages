'use client';

import { useState, Suspense } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { supabase } from '@/lib/supabase';

function SignupForm() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmationSent, setConfirmationSent] = useState(false);

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: {
          full_name: fullName,
          role: 'consumer',
        },
      },
    });

    if (error) {
      setError(error.message);
      setLoading(false);
      return;
    }

    // Check if email confirmation is required
    // If user.identities is empty, it means email confirmation is pending
    // If session exists, user is auto-confirmed (e.g., OAuth or email confirmation disabled)
    if (data.session) {
      // User is auto-confirmed, redirect to app
      window.location.href = '/app';
    } else {
      // Email confirmation required - show the confirmation message
      setConfirmationSent(true);
      setLoading(false);
    }
  };

  const handleOAuthSignup = async (provider: 'google' | 'github') => {
    const { error } = await supabase.auth.signInWithOAuth({
      provider,
      options: {
        redirectTo: `${window.location.origin}/auth/callback`,
      },
    });

    if (error) {
      setError(error.message);
    }
  };

  // Show confirmation sent screen
  if (confirmationSent) {
    return (
      <div className="w-full max-w-md relative z-10 px-4 sm:px-0">
        {/* Logo */}
        <div className="text-center mb-6 sm:mb-8">
          <div className="flex justify-center mb-4">
            <Image
              src="/images/brand/monogram.png"
              alt="LatticeForge"
              width={48}
              height={48}
              className="w-12 h-12"
            />
          </div>
        </div>

        {/* Glass confirmation card */}
        <div className="bg-[rgba(18,18,26,0.8)] backdrop-blur-xl rounded-2xl border border-white/[0.08] p-6 sm:p-8 shadow-2xl text-center">
          {/* Email icon */}
          <div className="mx-auto w-16 h-16 rounded-full bg-gradient-to-br from-blue-500/20 to-cyan-500/20 border border-blue-500/30 flex items-center justify-center mb-6">
            <svg className="w-8 h-8 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M21.75 6.75v10.5a2.25 2.25 0 01-2.25 2.25h-15a2.25 2.25 0 01-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0019.5 4.5h-15a2.25 2.25 0 00-2.25 2.25m19.5 0v.243a2.25 2.25 0 01-1.07 1.916l-7.5 4.615a2.25 2.25 0 01-2.36 0L3.32 8.91a2.25 2.25 0 01-1.07-1.916V6.75" />
            </svg>
          </div>

          <h2 className="text-xl sm:text-2xl font-bold text-white mb-3">
            Check your inbox
          </h2>
          <p className="text-slate-400 mb-2">
            We&apos;ve sent a confirmation email to:
          </p>
          <p className="text-blue-400 font-medium mb-6 break-all">
            {email}
          </p>

          <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4 mb-6">
            <p className="text-slate-300 text-sm">
              Click the link in your email to activate your account.
              <span className="text-slate-500 block mt-2">
                Be sure to check your spam folder, just in case!
              </span>
            </p>
          </div>

          <div className="space-y-3">
            <button
              onClick={() => setConfirmationSent(false)}
              className="w-full py-3 text-slate-400 hover:text-white text-sm transition-colors"
            >
              Use a different email
            </button>
            <Link
              href="/login"
              className="block w-full py-3.5 min-h-[52px] rounded-xl font-medium
                bg-gradient-to-r from-blue-600 via-blue-500 to-cyan-500 text-white
                hover:shadow-[0_0_30px_rgba(59,130,246,0.4)] active:scale-[0.98]
                transition-all text-center"
            >
              Go to Sign In
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-md relative z-10 px-4 sm:px-0">
      {/* Logo + Tagline */}
      <div className="text-center mb-6 sm:mb-8">
        <div className="flex justify-center mb-4 lg:hidden">
          <Image
            src="/images/brand/monogram.png"
            alt="LatticeForge"
            width={48}
            height={48}
            className="w-12 h-12"
          />
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold text-white">Create Account</h1>
        <p className="text-base sm:text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-cyan-400 to-orange-400 mt-2">
          Know what happens next.
        </p>
        <p className="text-slate-500 text-xs sm:text-sm mt-1">
          Deterministic geopolitical intelligence
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
            onClick={() => void handleOAuthSignup('google')}
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
            onClick={() => void handleOAuthSignup('github')}
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
            <span className="px-3 bg-[rgba(18,18,26,0.8)] text-slate-500">or sign up with email</span>
          </div>
        </div>

        {/* Email form */}
        <form onSubmit={(e) => void handleSignup(e)} className="space-y-4">
          <div>
            <label className="block text-sm text-slate-400 mb-2">Full Name</label>
            <input
              type="text"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              className="w-full px-4 py-3.5 min-h-[52px] text-base
                bg-black/30 border border-white/[0.08] rounded-xl text-white
                placeholder:text-slate-500
                focus:outline-none focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20
                transition-all"
              placeholder="John Doe"
            />
          </div>

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
              minLength={8}
            />
            <p className="text-xs text-slate-500 mt-1">Minimum 8 characters</p>
          </div>

          <div className="text-sm text-slate-400 min-h-[44px] flex items-center">
            By signing up, you agree to our{' '}
            <Link href="/terms" className="text-blue-400 hover:text-blue-300 mx-1">
              Terms
            </Link>{' '}
            and{' '}
            <Link href="/privacy" className="text-blue-400 hover:text-blue-300 mx-1">
              Privacy Policy
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
            {loading ? 'Creating account...' : 'Create account'}
          </button>
        </form>
      </div>

      {/* Sign in link */}
      <p className="text-center text-slate-400 mt-6 text-sm">
        Already have an account?{' '}
        <Link href="/login" className="text-blue-400 hover:text-blue-300 font-medium">
          Sign in
        </Link>
      </p>
    </div>
  );
}

function SignupFallback() {
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

export default function SignupPage() {
  return (
    <div className="min-h-screen bg-[#0a0a0f] flex flex-col lg:flex-row">
      {/* Background effects */}
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
        {/* Constellation background on mobile */}
        <div className="absolute inset-0 lg:hidden overflow-hidden">
          <div
            className="absolute top-[25%] left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] rounded-full"
            style={{ background: 'radial-gradient(circle, rgba(59,130,246,0.3) 0%, rgba(59,130,246,0.1) 40%, transparent 70%)' }}
          />
          <div className="absolute top-[25%] left-1/2 -translate-x-1/2 -translate-y-1/2 w-[350px] h-[350px] opacity-40">
            <Image
              src="/images/network/constellation.png"
              alt=""
              fill
              className="object-contain"
              priority
            />
          </div>
        </div>

        {/* Blue accent glow */}
        <div className="absolute -top-32 -left-32 w-[500px] h-[500px] opacity-25 pointer-events-none">
          <div className="w-full h-full rounded-full bg-blue-600/40 blur-[120px]" />
        </div>

        <Suspense fallback={<SignupFallback />}>
          <SignupForm />
        </Suspense>
      </div>

      {/* Right side - Hero visual (desktop only) */}
      <div className="hidden lg:flex flex-1 relative items-center justify-center bg-[rgba(10,10,15,0.5)] border-l border-white/[0.05] overflow-hidden">
        {/* Blue glow effect */}
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-[55%] w-[500px] h-[500px] xl:w-[600px] xl:h-[600px] rounded-full pointer-events-none"
          style={{
            background: 'radial-gradient(circle, rgba(59,130,246,0.4) 0%, rgba(59,130,246,0.15) 30%, transparent 60%)',
          }}
        />
        {/* Cyan accent */}
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-[55%] w-[600px] h-[600px] xl:w-[700px] xl:h-[700px] rounded-full pointer-events-none"
          style={{
            background: 'radial-gradient(circle at 70% 30%, rgba(6,182,212,0.15) 0%, transparent 50%)',
          }}
        />
        {/* Constellation hero */}
        <div className="relative w-[450px] h-[450px] xl:w-[550px] xl:h-[550px] -mt-16">
          <Image
            src="/images/network/constellation.png"
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
              See the connections
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-500">
                {' '}others miss
              </span>
            </h2>
            <p className="text-slate-400 text-sm mb-6">
              Map global influence networks. Predict phase transitions before they happen.
            </p>

            {/* Feature pills */}
            <div className="flex flex-wrap gap-2">
              {[
                { icon: '🌐', text: '195 Countries' },
                { icon: '🎯', text: '72% Accuracy' },
                { icon: '⚡', text: 'Real-time Intel' },
                { icon: '🔒', text: 'AES-256 Encrypted' },
              ].map((item, idx) => (
                <div
                  key={idx}
                  className="flex items-center gap-2 px-3 py-2 bg-white/[0.05] backdrop-blur-sm rounded-lg border border-white/[0.08]"
                >
                  <span className="text-sm">{item.icon}</span>
                  <span className="text-xs text-slate-300">{item.text}</span>
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
