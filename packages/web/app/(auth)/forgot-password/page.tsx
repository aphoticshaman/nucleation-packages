'use client';

import { useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { supabase } from '@/lib/supabase';
import { ArrowLeft, Mail, CheckCircle } from 'lucide-react';

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleResetPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/auth/reset-password`,
    });

    if (error) {
      setError(error.message);
      setLoading(false);
      return;
    }

    setSuccess(true);
    setLoading(false);
  };

  if (success) {
    return (
      <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center px-4">
        <div className="w-full max-w-md">
          <div className="text-center mb-8">
            <div className="flex justify-center mb-4">
              <Image
                src="/images/brand/monogram.png"
                alt="LatticeForge"
                width={48}
                height={48}
                className="w-12 h-12"
              />
            </div>
            <h1 className="text-2xl font-bold text-white">Check your email</h1>
          </div>

          <div className="bg-[rgba(18,18,26,0.8)] backdrop-blur-xl rounded-2xl border border-white/[0.08] p-8 text-center">
            <div className="flex justify-center mb-4">
              <div className="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center">
                <CheckCircle className="w-8 h-8 text-green-400" />
              </div>
            </div>
            <p className="text-slate-300 mb-6">
              We&apos;ve sent a password reset link to <span className="text-white font-medium">{email}</span>
            </p>
            <p className="text-slate-500 text-sm mb-6">
              The link will expire in 24 hours. If you don&apos;t see the email, check your spam folder.
            </p>
            <Link
              href="/login"
              className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 font-medium"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to login
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center px-4">
      {/* Background effects */}
      <div className="fixed inset-0 z-0 pointer-events-none">
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

      <div className="w-full max-w-md relative z-10">
        <div className="text-center mb-8">
          <div className="flex justify-center mb-4">
            <Image
              src="/images/brand/monogram.png"
              alt="LatticeForge"
              width={48}
              height={48}
              className="w-12 h-12"
            />
          </div>
          <h1 className="text-2xl font-bold text-white">Reset your password</h1>
          <p className="text-slate-500 text-sm mt-2">
            Enter your email and we&apos;ll send you a reset link
          </p>
        </div>

        <div className="bg-[rgba(18,18,26,0.8)] backdrop-blur-xl rounded-2xl border border-white/[0.08] p-8">
          {error && (
            <div className="mb-5 p-4 bg-red-500/20 border border-red-500/30 rounded-xl text-red-300 text-sm">
              {error}
            </div>
          )}

          <form onSubmit={(e) => void handleResetPassword(e)} className="space-y-4">
            <div>
              <label className="block text-sm text-slate-400 mb-2">Email address</label>
              <div className="relative">
                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full pl-12 pr-4 py-3.5 min-h-[52px] text-base
                    bg-black/30 border border-white/[0.08] rounded-xl text-white
                    placeholder:text-slate-500
                    focus:outline-none focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20
                    transition-all"
                  placeholder="you@example.com"
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className={`w-full py-3.5 min-h-[52px] rounded-xl font-medium transition-all ${
                loading
                  ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                  : 'bg-gradient-to-r from-blue-600 via-blue-500 to-cyan-500 text-white hover:shadow-[0_0_30px_rgba(59,130,246,0.4)] active:scale-[0.98]'
              }`}
            >
              {loading ? 'Sending...' : 'Send reset link'}
            </button>
          </form>
        </div>

        <p className="text-center text-slate-400 mt-6 text-sm">
          <Link href="/login" className="text-blue-400 hover:text-blue-300 font-medium inline-flex items-center gap-1">
            <ArrowLeft className="w-4 h-4" />
            Back to login
          </Link>
        </p>
      </div>
    </div>
  );
}
