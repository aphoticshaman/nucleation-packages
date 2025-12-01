import { useState } from 'react';
import { LatticeForgeIcon, LockIcon, ShieldIcon } from './Icons';

interface AuthGateProps {
  onAuthenticate: (apiKey: string) => void;
}

export function AuthGate({ onAuthenticate }: AuthGateProps) {
  const [apiKey, setApiKey] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    // Validate API key format
    if (!apiKey.match(/^lf_(live|test)_[A-Za-z0-9_-]{32,64}$/)) {
      setError('Invalid API key format. Keys start with lf_live_ or lf_test_');
      setIsLoading(false);
      return;
    }

    // In production, validate against API
    // For now, accept valid format
    setTimeout(() => {
      onAuthenticate(apiKey);
      setIsLoading(false);
    }, 500);
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
          <p className="text-lattice-400 text-sm">
            The Crystallization of Meta-Insight
          </p>
        </div>

        {/* Auth card */}
        <div className="glass-card p-8">
          <div className="flex items-center gap-2 mb-6">
            <LockIcon className="w-5 h-5 text-lattice-500" />
            <h2 className="text-lg font-semibold text-white">
              Enterprise Access
            </h2>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="space-y-4">
              <div>
                <label htmlFor="apiKey" className="block text-sm font-medium text-lattice-300 mb-2">
                  API Key
                </label>
                <input
                  id="apiKey"
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="lf_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                  className="input-field font-mono text-sm"
                  autoComplete="off"
                  spellCheck={false}
                />
              </div>

              {error && (
                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={isLoading || !apiKey}
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
                    Authenticating...
                  </span>
                ) : (
                  'Access Dashboard'
                )}
              </button>
            </div>
          </form>

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
            <a href="#" className="text-lattice-500 hover:text-lattice-400">Contact Sales</a>
            {' · '}
            <a href="#" className="text-lattice-500 hover:text-lattice-400">Documentation</a>
            {' · '}
            <a href="#" className="text-lattice-500 hover:text-lattice-400">Privacy</a>
          </p>
        </div>
      </div>
    </div>
  );
}
