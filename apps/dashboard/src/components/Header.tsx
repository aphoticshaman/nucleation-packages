import { LatticeForgeIcon } from './Icons';
import type { User } from '@supabase/supabase-js';

interface HeaderProps {
  onLogout: () => void;
  user?: User | null;
}

export function Header({ onLogout, user }: HeaderProps) {
  return (
    <header className="fixed top-0 left-0 right-0 h-16 bg-surface-800/80 backdrop-blur-xl border-b border-surface-600/50 z-50">
      <div className="flex items-center justify-between h-full px-6">
        {/* Logo & Brand */}
        <div className="flex items-center gap-3">
          <LatticeForgeIcon className="w-8 h-8" />
          <div>
            <h1 className="text-lg font-semibold text-white tracking-tight">LatticeForge</h1>
            <p className="text-[10px] text-lattice-400 uppercase tracking-widest -mt-0.5">
              Signal Intelligence
            </p>
          </div>
        </div>

        {/* Center - Status */}
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="status-indicator status-healthy" />
            <span className="text-xs text-lattice-300">All Systems Operational</span>
          </div>
          <div className="text-xs text-surface-500">|</div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-lattice-400">API Latency:</span>
            <span className="text-xs text-emerald-400 font-mono">23ms</span>
          </div>
        </div>

        {/* Right - User */}
        <div className="flex items-center gap-4">
          <div className="text-right">
            <p className="text-sm font-medium text-white">{user?.email || 'API Access'}</p>
            <p className="text-[10px] text-lattice-400">
              {user ? 'Authenticated User' : 'API Key'}
            </p>
          </div>
          <button
            onClick={onLogout}
            className="px-3 py-1.5 text-xs text-lattice-300 hover:text-white hover:bg-surface-700 rounded-lg transition-colors"
          >
            Sign Out
          </button>
        </div>
      </div>
    </header>
  );
}
