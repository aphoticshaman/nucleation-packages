
import React from 'react';
import { ContextLayer } from '../types';
import { Activity, Lock, Globe, Share2, LayoutDashboard, TrendingUp } from 'lucide-react';

interface NavigationProps {
  context: ContextLayer;
  setContext: (c: ContextLayer) => void;
  currentView: 'causal' | 'workbench' | 'predictive';
  setView: (v: 'causal' | 'workbench' | 'predictive') => void;
}

export const Navigation: React.FC<NavigationProps> = ({ context, setContext, currentView, setView }) => {
  const contextItems = [
    { id: ContextLayer.SURFACE, icon: Globe, label: 'SURFACE' },
    { id: ContextLayer.DEEP, icon: Activity, label: 'DEEP' },
    { id: ContextLayer.DARK, icon: Lock, label: 'DARK' },
  ];

  return (
    <nav className="h-full w-20 lg:w-64 flex flex-col justify-between glass-panel border-r-0 border-r-border-default z-20">
      <div>
        {/* Brand Header */}
        <div className="h-20 flex items-center justify-center lg:justify-start lg:px-6 border-b border-border-default bg-surface/50">
          <div className="w-8 h-8 bg-primary/10 rounded flex items-center justify-center border border-primary/30 shadow-[0_0_15px_rgba(56,189,248,0.3)]">
            <Share2 size={18} className="text-primary" />
          </div>
          <span className="hidden lg:block ml-3 font-mono font-bold text-lg tracking-[0.2em] text-text-primary">
            LATTICE
          </span>
        </div>

        {/* View Switcher - LatticeForge Modules */}
        <div className="px-3 py-6 border-b border-border-default">
            <div className="text-[10px] font-mono text-text-muted mb-2 px-3 hidden lg:block tracking-widest uppercase">Modules</div>
            
            <button
              onClick={() => setView('causal')}
              className={`w-full flex items-center px-3 py-2 mb-1 rounded-md transition-all duration-200 ${
                currentView === 'causal' 
                  ? 'bg-primary/20 text-primary border border-primary/30' 
                  : 'text-text-secondary hover:text-text-primary hover:bg-surface-raised'
              }`}
            >
              <Share2 size={18} />
              <span className="hidden lg:block ml-3 font-mono text-xs">CAUSAL CANVAS</span>
            </button>

            <button
              onClick={() => setView('workbench')}
              className={`w-full flex items-center px-3 py-2 mb-1 rounded-md transition-all duration-200 ${
                currentView === 'workbench' 
                  ? 'bg-primary/20 text-primary border border-primary/30' 
                  : 'text-text-secondary hover:text-text-primary hover:bg-surface-raised'
              }`}
            >
              <LayoutDashboard size={18} />
              <span className="hidden lg:block ml-3 font-mono text-xs">WORKBENCH</span>
            </button>

            <button
              onClick={() => setView('predictive')}
              className={`w-full flex items-center px-3 py-2 rounded-md transition-all duration-200 ${
                currentView === 'predictive' 
                  ? 'bg-primary/20 text-primary border border-primary/30' 
                  : 'text-text-secondary hover:text-text-primary hover:bg-surface-raised'
              }`}
            >
              <TrendingUp size={18} />
              <span className="hidden lg:block ml-3 font-mono text-xs">PREDICTIVE</span>
            </button>
        </div>

        {/* Context Switching */}
        <div className="py-6 space-y-1 px-2">
          <div className="text-[10px] font-mono text-text-muted mb-4 px-4 hidden lg:block tracking-widest uppercase">Depth Layer</div>
          {contextItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setContext(item.id)}
              className={`w-full flex items-center px-4 py-3 rounded-md transition-all duration-200 relative group ${
                context === item.id 
                  ? 'bg-surface-raised text-text-primary border-l-2 border-l-primary'
                  : 'text-text-secondary hover:text-text-primary hover:bg-surface-raised/50 border-l-2 border-l-transparent'
              }`}
            >
              <item.icon size={18} className={context === item.id ? 'text-primary' : 'text-text-muted group-hover:text-text-secondary'} />
              <span className="hidden lg:block ml-3 font-mono text-xs font-medium tracking-wide">
                {item.label}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="p-6 border-t border-border-default bg-surface/30">
        <div className="text-[10px] font-mono text-text-muted mb-2 tracking-wide uppercase">System</div>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-status-success rounded-full animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]" />
          <span className="text-xs text-text-secondary font-mono">LATTICE ONLINE</span>
        </div>
      </div>
    </nav>
  );
};
