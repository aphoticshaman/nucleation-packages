'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { useRouter } from 'next/navigation';

type CommandType = 'navigation' | 'action' | 'filter' | 'search';

interface Command {
  id: string;
  label: string;
  description?: string;
  type: CommandType;
  icon?: string;
  shortcut?: string;
  action: () => void | Promise<void>;
  keywords?: string[];
}

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  commands?: Command[];
  onSearch?: (query: string) => Promise<Command[]>;
}

const typeIcons: Record<CommandType, string> = {
  navigation: '→',
  action: '⚡',
  filter: '⊕',
  search: '⌕',
};

const typeColors: Record<CommandType, string> = {
  navigation: 'text-cyan-400',
  action: 'text-amber-400',
  filter: 'text-green-400',
  search: 'text-slate-400',
};

export function CommandPalette({
  isOpen,
  onClose,
  commands = [],
  onSearch,
}: CommandPaletteProps) {
  const router = useRouter();
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [mode, setMode] = useState<'search' | 'command'>('search');
  const [dynamicResults, setDynamicResults] = useState<Command[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  // Default commands
  const defaultCommands: Command[] = [
    {
      id: 'nav-dashboard',
      label: 'Dashboard',
      description: 'Go to main dashboard',
      type: 'navigation',
      shortcut: 'G D',
      action: () => router.push('/dashboard'),
      keywords: ['home', 'main', 'overview'],
    },
    {
      id: 'nav-signals',
      label: 'Signal Feed',
      description: 'Real-time intelligence stream',
      type: 'navigation',
      shortcut: 'G S',
      action: () => router.push('/dashboard/signals'),
      keywords: ['feed', 'stream', 'events'],
    },
    {
      id: 'nav-map',
      label: 'Risk Map',
      description: 'Geospatial risk visualization',
      type: 'navigation',
      shortcut: 'G M',
      action: () => router.push('/dashboard/map'),
      keywords: ['geo', 'location', 'hexbin'],
    },
    {
      id: 'nav-cascades',
      label: 'Cascade Analysis',
      description: 'Cross-domain cascade detection',
      type: 'navigation',
      shortcut: 'G C',
      action: () => router.push('/dashboard/cascades'),
      keywords: ['domino', 'ripple', 'effect'],
    },
    {
      id: 'nav-explain',
      label: 'Explainability',
      description: 'Logic tree inspector',
      type: 'navigation',
      shortcut: 'G E',
      action: () => router.push('/dashboard/explain'),
      keywords: ['why', 'reason', 'logic'],
    },
    {
      id: 'filter-domain',
      label: 'Filter by Domain',
      description: 'Focus on specific intelligence domain',
      type: 'filter',
      action: () => setMode('command'),
      keywords: ['cyber', 'financial', 'geopolitical'],
    },
    {
      id: 'action-export',
      label: 'Export Report',
      description: 'Generate PDF briefing',
      type: 'action',
      shortcut: '⌘ E',
      action: () => console.log('Export triggered'),
      keywords: ['pdf', 'download', 'report'],
    },
    {
      id: 'action-refresh',
      label: 'Force Refresh',
      description: 'Reload all data feeds',
      type: 'action',
      shortcut: '⌘ R',
      action: () => window.location.reload(),
      keywords: ['reload', 'update'],
    },
  ];

  const allCommands = [...defaultCommands, ...commands];

  // Filter commands based on query
  const filteredCommands = query
    ? allCommands.filter(cmd => {
        const searchText = `${cmd.label} ${cmd.description || ''} ${(cmd.keywords || []).join(' ')}`.toLowerCase();
        return searchText.includes(query.toLowerCase());
      })
    : allCommands.slice(0, 8);

  const displayCommands = dynamicResults.length > 0 ? dynamicResults : filteredCommands;

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex(prev => Math.min(prev + 1, displayCommands.length - 1));
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex(prev => Math.max(prev - 1, 0));
          break;
        case 'Enter':
          e.preventDefault();
          if (displayCommands[selectedIndex]) {
            displayCommands[selectedIndex].action();
            onClose();
          }
          break;
        case 'Escape':
          e.preventDefault();
          onClose();
          break;
        case 'Backspace':
          if (query === '' && mode === 'command') {
            setMode('search');
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, displayCommands, selectedIndex, query, mode, onClose]);

  // Reset on open
  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setSelectedIndex(0);
      setMode('search');
      setDynamicResults([]);
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [isOpen]);

  // Dynamic search
  useEffect(() => {
    if (onSearch && query.length > 2) {
      onSearch(query).then(results => {
        setDynamicResults(results);
        setSelectedIndex(0);
      });
    } else {
      setDynamicResults([]);
    }
  }, [query, onSearch]);

  // Check for command mode trigger
  useEffect(() => {
    if (query.startsWith('>')) {
      setMode('command');
      setQuery(query.slice(1));
    }
  }, [query]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Palette */}
      <div className="relative w-full max-w-2xl mx-4 bg-slate-900/95 backdrop-blur-2xl border border-slate-700/50 rounded-xl shadow-2xl overflow-hidden">
        {/* Input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-slate-800">
          {/* Mode indicator */}
          <span className={`text-lg ${mode === 'command' ? 'text-amber-400' : 'text-slate-500'}`}>
            {mode === 'command' ? '>' : '⌘'}
          </span>

          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={mode === 'command' ? 'Enter command...' : 'Search or type > for commands...'}
            className="flex-1 bg-transparent text-slate-100 placeholder-slate-500 outline-none font-mono text-sm"
          />

          {/* Keyboard hint */}
          <div className="flex items-center gap-1">
            <kbd className="px-1.5 py-0.5 text-xs font-mono bg-slate-800 text-slate-400 rounded">
              ESC
            </kbd>
          </div>
        </div>

        {/* Results */}
        <div className="max-h-80 overflow-y-auto py-2">
          {displayCommands.length === 0 ? (
            <div className="px-4 py-8 text-center text-slate-500">
              No results found for "{query}"
            </div>
          ) : (
            displayCommands.map((cmd, index) => (
              <button
                key={cmd.id}
                onClick={() => {
                  cmd.action();
                  onClose();
                }}
                onMouseEnter={() => setSelectedIndex(index)}
                className={`
                  w-full flex items-center gap-3 px-4 py-2.5 text-left
                  transition-colors duration-75
                  ${index === selectedIndex
                    ? 'bg-slate-800/80'
                    : 'hover:bg-slate-800/40'
                  }
                `}
              >
                {/* Type icon */}
                <span className={`text-lg w-6 text-center ${typeColors[cmd.type]}`}>
                  {typeIcons[cmd.type]}
                </span>

                {/* Label and description */}
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-slate-200 truncate">
                    {cmd.label}
                  </div>
                  {cmd.description && (
                    <div className="text-xs text-slate-500 truncate">
                      {cmd.description}
                    </div>
                  )}
                </div>

                {/* Shortcut */}
                {cmd.shortcut && (
                  <div className="flex items-center gap-1">
                    {cmd.shortcut.split(' ').map((key, i) => (
                      <kbd
                        key={i}
                        className="px-1.5 py-0.5 text-xs font-mono bg-slate-800 text-slate-400 rounded"
                      >
                        {key}
                      </kbd>
                    ))}
                  </div>
                )}
              </button>
            ))
          )}
        </div>

        {/* Footer hints */}
        <div className="px-4 py-2 border-t border-slate-800 flex items-center justify-between text-xs text-slate-500">
          <div className="flex items-center gap-4">
            <span>
              <kbd className="px-1 bg-slate-800 rounded">↑↓</kbd> navigate
            </span>
            <span>
              <kbd className="px-1 bg-slate-800 rounded">↵</kbd> select
            </span>
            <span>
              <kbd className="px-1 bg-slate-800 rounded">{'>'}</kbd> commands
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-cyan-500">●</span> Neural
            <span className="text-amber-500">●</span> Symbolic
          </div>
        </div>
      </div>
    </div>
  );
}

// Global keyboard hook
export function useCommandPalette() {
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl/Cmd + K
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsOpen(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return {
    isOpen,
    open: () => setIsOpen(true),
    close: () => setIsOpen(false),
    toggle: () => setIsOpen(prev => !prev),
  };
}
