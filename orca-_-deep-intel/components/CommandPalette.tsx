import React, { useState, useEffect } from 'react';
import { Search, Command, ArrowRight } from 'lucide-react';

export const CommandPalette: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setIsOpen((open) => !open);
      }
    };
    document.addEventListener('keydown', down);
    return () => document.removeEventListener('keydown', down);
  }, []);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh] bg-app/60 backdrop-blur-sm animate-fade-in" onClick={() => setIsOpen(false)}>
      <div 
        className="w-full max-w-2xl glass-panel rounded-xl overflow-hidden shadow-2xl transform transition-all border border-border-muted" 
        onClick={e => e.stopPropagation()}
      >
        {/* Input Field Area with Glassmorphism */}
        <div className="flex items-center px-4 py-4 border-b border-white/10 bg-white/5 backdrop-blur-sm focus-within:bg-white/10 focus-within:ring-1 focus-within:ring-primary/50 transition-all duration-300">
          <Command className="text-primary/70 mr-3" size={20} />
          <input
            autoFocus
            type="text"
            placeholder="Search Intelligence Graph or execute command..."
            className="flex-1 bg-transparent border-none outline-none text-text-primary placeholder-text-muted/60 font-mono text-sm"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className="flex items-center gap-2">
             <span className="text-[10px] font-mono bg-white/5 border border-white/10 px-1.5 py-0.5 rounded text-text-muted/70">ESC</span>
          </div>
        </div>
        
        {/* Results Area */}
        <div className="p-2 bg-surface/50 backdrop-blur-sm">
            {!query && (
                <div className="text-xs font-mono text-text-muted px-2 py-2 tracking-wide">SUGGESTED COMMANDS</div>
            )}
            {['/simulate Taiwan_Strait_Blockade', '/compare EZC SPZ', '/alert threshold:variance > 0.8'].map((cmd, i) => (
                <button key={i} className="w-full flex items-center justify-between px-3 py-2.5 text-sm text-text-secondary hover:bg-primary/10 hover:text-primary rounded-md transition-colors group">
                    <span className="font-mono">{cmd}</span>
                    <ArrowRight size={14} className="opacity-0 group-hover:opacity-100 transition-opacity text-primary" />
                </button>
            ))}
        </div>
      </div>
    </div>
  );
};