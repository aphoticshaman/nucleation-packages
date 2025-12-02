import { useState, useEffect } from 'react';
import { LatticeForgeIcon, LockIcon, ShieldIcon } from './Icons';
import { auth, supabase } from '../lib/supabase';

interface AuthGateProps {
  onAuthenticate: () => void;
}

type AuthMode = 'login' | 'signup' | 'forgot';

export function AuthGate({ onAuthenticate }: AuthGateProps) {
  const [mode, setMode] = useState<AuthMode>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Reset loading state on mount (handles return from failed OAuth redirect)
  useEffect(() => {
    setIsLoading(false);

    // Check URL for OAuth error
    const params = new URLSearchParams(window.location.search);
    const errorParam = params.get('error_description') || params.get('error');
    if (errorParam) {
      setError(decodeURIComponent(errorParam));
      // Clean up URL
      window.history.replaceState({}, '', window.location.pathname);
    }
  }, []);

  const handleEmailAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setMessage(null);
    setIsLoading(true);

    try {
      if (mode === 'login') {
        await auth.signIn(email, password);
        onAuthenticate();
      } else if (mode === 'signup') {
        await auth.signUp(email, password);
        setMessage('Check your email for a confirmation link');
      } else if (mode === 'forgot') {
        const { error } = await supabase.auth.resetPasswordForEmail(email, {
          redirectTo: `${window.location.origin}/auth/reset`,
        });
        if (error) throw error;
        setMessage('Password reset email sent');
      }
    } catch (err: any) {
      setError(err.message || 'Authentication failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleAuth = async () => {
    setError(null);
    setIsLoading(true);

    try {
      const { error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: `${window.location.origin}/auth/callback`,
        },
      });
      if (error) throw error;
    } catch (err: any) {
      setError(err.message || 'Google sign-in failed');
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-surface-900 bg-grid flex items-center justify-center p-4">
      {/* Background effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-lattice-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-crystal-500/10 rounded-full blur-3xl" />
      </div>

      <div className="relative w-full max-w-md">
        {/* Logo and branding */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-surface-800/50 border border-surface-600/50 mb-6 glow-md">
            <LatticeForgeIcon className="w-12 h-12 text-lattice-400" />
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">LatticeForge</h1>
          <p className="text-lattice-400 text-sm">The Crystallization of Meta-Insight</p>
        </div>

        {/* Auth card */}
        <div className="glass-card p-8">
          <div className="flex items-center gap-2 mb-6">
            <LockIcon className="w-5 h-5 text-lattice-500" />
            <h2 className="text-lg font-semibold text-white">
              {mode === 'login' && 'Sign In'}
              {mode === 'signup' && 'Create Account'}
              {mode === 'forgot' && 'Reset Password'}
            </h2>
          </div>

          {/* Auth mode tabs */}
          {mode !== 'forgot' && (
            <div className="flex gap-2 mb-6">
              <button
                onClick={() => {
                  setMode('login');
                  setError(null);
                  setMessage(null);
                }}
                className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-colors ${
                  mode === 'login'
                    ? 'bg-lattice-500/20 text-lattice-400 border border-lattice-500/30'
                    : 'text-surface-400 hover:text-white hover:bg-surface-700/50'
                }`}
              >
                Sign In
              </button>
              <button
                onClick={() => {
                  setMode('signup');
                  setError(null);
                  setMessage(null);
                }}
                className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-colors ${
                  mode === 'signup'
                    ? 'bg-lattice-500/20 text-lattice-400 border border-lattice-500/30'
                    : 'text-surface-400 hover:text-white hover:bg-surface-700/50'
                }`}
              >
                Sign Up
              </button>
            </div>
          )}

          {/* Google OAuth button */}
          {(mode === 'login' || mode === 'signup') && (
            <>
              <button
                onClick={handleGoogleAuth}
                disabled={isLoading}
                className="w-full flex items-center justify-center gap-3 py-3 px-4 rounded-lg bg-white text-gray-800 font-medium hover:bg-gray-100 transition-colors disabled:opacity-50"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24">
                  <path
                    fill="#4285F4"
                    d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                  />
                  <path
                    fill="#34A853"
                    d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                  />
                  <path
                    fill="#FBBC05"
                    d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                  />
                  <path
                    fill="#EA4335"
                    d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                  />
                </svg>
                Continue with Google
              </button>

              <div className="relative my-6">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-surface-600"></div>
                </div>
                <div className="relative flex justify-center text-sm">
                  <span className="px-2 bg-surface-800 text-surface-400">or</span>
                </div>
              </div>
            </>
          )}

          {/* Email/Password form */}
          <form onSubmit={handleEmailAuth}>
            <div className="space-y-4">
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-lattice-300 mb-2">
                  Email
                </label>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@company.com"
                  className="input-field"
                  required
                />
              </div>

              {(mode === 'login' || mode === 'signup') && (
                <div>
                  <label
                    htmlFor="password"
                    className="block text-sm font-medium text-lattice-300 mb-2"
                  >
                    Password
                  </label>
                  <input
                    id="password"
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••••••"
                    className="input-field"
                    minLength={8}
                    required
                  />
                </div>
              )}

              {error && (
                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
                  {error}
                </div>
              )}

              {message && (
                <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm">
                  {message}
                </div>
              )}

              <button
                type="submit"
                disabled={isLoading}
                className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24">
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                        fill="none"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    {mode === 'login'
                      ? 'Signing in...'
                      : mode === 'signup'
                        ? 'Creating account...'
                        : 'Sending...'}
                  </span>
                ) : mode === 'login' ? (
                  'Sign In'
                ) : mode === 'signup' ? (
                  'Create Account'
                ) : (
                  'Send Reset Link'
                )}
              </button>
            </div>
          </form>

          {/* Forgot password link */}
          {mode === 'login' && (
            <button
              onClick={() => {
                setMode('forgot');
                setError(null);
                setMessage(null);
              }}
              className="w-full mt-4 text-sm text-lattice-500 hover:text-lattice-400"
            >
              Forgot your password?
            </button>
          )}

          {/* Back to login link */}
          {mode === 'forgot' && (
            <button
              onClick={() => {
                setMode('login');
                setError(null);
                setMessage(null);
              }}
              className="w-full mt-4 text-sm text-lattice-500 hover:text-lattice-400"
            >
              Back to sign in
            </button>
          )}

          {/* Security badges */}
          <div className="mt-6 pt-6 border-t border-surface-600/50">
            <div className="flex items-center justify-center gap-6 text-xs text-surface-500">
              <div className="flex items-center gap-1.5">
                <ShieldIcon className="w-4 h-4 text-emerald-500" />
                <span>256-bit encryption</span>
              </div>
              <div className="flex items-center gap-1.5">
                <ShieldIcon className="w-4 h-4 text-emerald-500" />
                <span>SOC 2 compliant</span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-xs text-surface-500">
          <p>Crystalline Labs LLC © 2025</p>
          <p className="mt-1">
            <a href="#" className="text-lattice-500 hover:text-lattice-400">
              Contact Sales
            </a>
            {' · '}
            <a href="#" className="text-lattice-500 hover:text-lattice-400">
              Documentation
            </a>
            {' · '}
            <a href="#" className="text-lattice-500 hover:text-lattice-400">
              Privacy
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
