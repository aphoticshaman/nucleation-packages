'use client';

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { createPortal } from 'react-dom';
import {
  Search,
  Command,
  Home,
  BarChart3,
  Bell,
  Settings,
  FileText,
  Shield,
  Users,
  Download,
  HelpCircle,
  Keyboard,
  Globe,
  AlertTriangle,
  Zap,
  Database,
  X
} from 'lucide-react';

/**
 * Command Palette (Cmd+K / Ctrl+K)
 *
 * Anti-Complaint Spec Section 8.2 Implementation:
 * - Universal search across all intelligence
 * - Keyboard-first navigation
 * - Action shortcuts
 */

interface CommandItem {
  id: string;
  title: string;
  description?: string;
  icon: React.ReactNode;
  category: 'navigation' | 'action' | 'search' | 'recent';
  shortcut?: string;
  action: () => void;
  keywords?: string[];
}

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
}

function CommandPaletteContent({ isOpen, onClose }: CommandPaletteProps) {
  const router = useRouter();
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Define available commands
  const commands: CommandItem[] = useMemo(() => [
    // Navigation
    {
      id: 'nav-dashboard',
      title: 'Go to Dashboard',
      description: 'Main intelligence dashboard',
      icon: <Home className="w-4 h-4" />,
      category: 'navigation',
      shortcut: 'G D',
      action: () => router.push('/dashboard'),
      keywords: ['home', 'main', 'overview'],
    },
    {
      id: 'nav-signals',
      title: 'Go to Signals',
      description: 'Real-time signal monitoring',
      icon: <Zap className="w-4 h-4" />,
      category: 'navigation',
      shortcut: 'G S',
      action: () => router.push('/app/signals'),
      keywords: ['alerts', 'events', 'real-time'],
    },
    {
      id: 'nav-navigator',
      title: 'Go to Navigator',
      description: 'Explore intelligence data',
      icon: <Globe className="w-4 h-4" />,
      category: 'navigation',
      shortcut: 'G N',
      action: () => router.push('/app/navigator'),
      keywords: ['explore', 'search', 'browse'],
    },
    {
      id: 'nav-analytics',
      title: 'Go to Analytics',
      description: 'Usage and performance metrics',
      icon: <BarChart3 className="w-4 h-4" />,
      category: 'navigation',
      action: () => router.push('/dashboard/usage'),
      keywords: ['metrics', 'stats', 'reports'],
    },
    {
      id: 'nav-notifications',
      title: 'Go to Notifications',
      description: 'Alert preferences and history',
      icon: <Bell className="w-4 h-4" />,
      category: 'navigation',
      action: () => router.push('/app/notifications'),
      keywords: ['alerts', 'preferences'],
    },
    {
      id: 'nav-settings',
      title: 'Go to Settings',
      description: 'Account and app configuration',
      icon: <Settings className="w-4 h-4" />,
      category: 'navigation',
      shortcut: 'G ,',
      action: () => router.push('/app/settings'),
      keywords: ['config', 'preferences', 'account'],
    },
    {
      id: 'nav-team',
      title: 'Go to Team',
      description: 'Manage team members',
      icon: <Users className="w-4 h-4" />,
      category: 'navigation',
      action: () => router.push('/dashboard/team'),
      keywords: ['members', 'users', 'organization'],
    },
    {
      id: 'nav-api-keys',
      title: 'Go to API Keys',
      description: 'Manage API credentials',
      icon: <Shield className="w-4 h-4" />,
      category: 'navigation',
      action: () => router.push('/dashboard/api-keys'),
      keywords: ['tokens', 'credentials', 'auth'],
    },

    // Actions
    {
      id: 'action-export',
      title: 'Export Current View',
      description: 'Download data as CSV or JSON',
      icon: <Download className="w-4 h-4" />,
      category: 'action',
      shortcut: '⌘ E',
      action: () => {
        // Dispatch custom event for export
        window.dispatchEvent(new CustomEvent('latticeforge:export'));
        onClose();
      },
      keywords: ['download', 'csv', 'json', 'save'],
    },
    {
      id: 'action-new-alert',
      title: 'Create Alert Rule',
      description: 'Set up a new monitoring alert',
      icon: <AlertTriangle className="w-4 h-4" />,
      category: 'action',
      action: () => {
        router.push('/app/notifications?create=true');
        onClose();
      },
      keywords: ['monitor', 'watch', 'notification'],
    },
    {
      id: 'action-intel-brief',
      title: 'Generate Intel Brief',
      description: 'Create a new intelligence briefing',
      icon: <FileText className="w-4 h-4" />,
      category: 'action',
      action: () => {
        router.push('/app/briefings/new');
        onClose();
      },
      keywords: ['report', 'summary', 'brief'],
    },
    {
      id: 'action-help',
      title: 'Help & Documentation',
      description: 'View guides and keyboard shortcuts',
      icon: <HelpCircle className="w-4 h-4" />,
      category: 'action',
      shortcut: '⌘ /',
      action: () => {
        window.dispatchEvent(new CustomEvent('latticeforge:help'));
        onClose();
      },
      keywords: ['docs', 'guide', 'support'],
    },
    {
      id: 'action-shortcuts',
      title: 'Keyboard Shortcuts',
      description: 'View all available shortcuts',
      icon: <Keyboard className="w-4 h-4" />,
      category: 'action',
      shortcut: '?',
      action: () => {
        window.dispatchEvent(new CustomEvent('latticeforge:shortcuts'));
        onClose();
      },
      keywords: ['keys', 'hotkeys', 'bindings'],
    },

    // Search shortcuts
    {
      id: 'search-threats',
      title: 'Search Threat Actors',
      description: 'Find threat actor profiles',
      icon: <Database className="w-4 h-4" />,
      category: 'search',
      action: () => {
        router.push('/app/navigator?type=threat_actor');
        onClose();
      },
      keywords: ['apt', 'adversary', 'attacker'],
    },
    {
      id: 'search-indicators',
      title: 'Search Indicators',
      description: 'Find IOCs and indicators',
      icon: <Search className="w-4 h-4" />,
      category: 'search',
      action: () => {
        router.push('/app/navigator?type=indicator');
        onClose();
      },
      keywords: ['ioc', 'hash', 'ip', 'domain'],
    },
  ], [router, onClose]);

  // Filter commands based on query
  const filteredCommands = useMemo(() => {
    if (!query) return commands;

    const lowerQuery = query.toLowerCase();
    return commands.filter((cmd) => {
      const matchTitle = cmd.title.toLowerCase().includes(lowerQuery);
      const matchDesc = cmd.description?.toLowerCase().includes(lowerQuery);
      const matchKeywords = cmd.keywords?.some((k) => k.includes(lowerQuery));
      return matchTitle || matchDesc || matchKeywords;
    });
  }, [commands, query]);

  // Group commands by category
  const groupedCommands = useMemo(() => {
    const groups: Record<string, CommandItem[]> = {
      navigation: [],
      action: [],
      search: [],
      recent: [],
    };

    for (const cmd of filteredCommands) {
      groups[cmd.category].push(cmd);
    }

    return groups;
  }, [filteredCommands]);

  // Reset selection when query changes
  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      inputRef.current?.focus();
      setQuery('');
      setSelectedIndex(0);
    }
  }, [isOpen]);

  // Handle keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex((i) => Math.min(i + 1, filteredCommands.length - 1));
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex((i) => Math.max(i - 1, 0));
          break;
        case 'Enter':
          e.preventDefault();
          if (filteredCommands[selectedIndex]) {
            filteredCommands[selectedIndex].action();
            onClose();
          }
          break;
        case 'Escape':
          e.preventDefault();
          onClose();
          break;
      }
    },
    [filteredCommands, selectedIndex, onClose]
  );

  // Scroll selected item into view
  useEffect(() => {
    const list = listRef.current;
    if (!list) return;

    const selectedEl = list.querySelector(`[data-index="${selectedIndex}"]`);
    if (selectedEl) {
      selectedEl.scrollIntoView({ block: 'nearest' });
    }
  }, [selectedIndex]);

  if (!isOpen) return null;

  const categoryLabels: Record<string, string> = {
    navigation: 'Navigation',
    action: 'Actions',
    search: 'Search',
    recent: 'Recent',
  };

  let globalIndex = 0;

  return (
    <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[20vh]">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Palette */}
      <div className="relative w-full max-w-xl bg-slate-900 border border-white/10 rounded-xl shadow-2xl overflow-hidden">
        {/* Search Input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-white/10">
          <Command className="w-5 h-5 text-slate-400" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a command or search..."
            className="flex-1 bg-transparent text-white placeholder-slate-500 focus:outline-none text-lg"
          />
          <button
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Command List */}
        <div ref={listRef} className="max-h-[60vh] overflow-y-auto p-2">
          {filteredCommands.length === 0 ? (
            <div className="py-8 text-center text-slate-500">
              No commands found for "{query}"
            </div>
          ) : (
            Object.entries(groupedCommands).map(([category, items]) => {
              if (items.length === 0) return null;

              return (
                <div key={category} className="mb-2">
                  <div className="px-3 py-1.5 text-xs font-medium text-slate-500 uppercase tracking-wider">
                    {categoryLabels[category]}
                  </div>
                  {items.map((cmd) => {
                    const currentIndex = globalIndex++;
                    const isSelected = currentIndex === selectedIndex;

                    return (
                      <button
                        key={cmd.id}
                        data-index={currentIndex}
                        onClick={() => {
                          cmd.action();
                          onClose();
                        }}
                        onMouseEnter={() => setSelectedIndex(currentIndex)}
                        className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${
                          isSelected
                            ? 'bg-blue-500/20 text-white'
                            : 'text-slate-300 hover:bg-white/5'
                        }`}
                      >
                        <span
                          className={`${
                            isSelected ? 'text-blue-400' : 'text-slate-500'
                          }`}
                        >
                          {cmd.icon}
                        </span>
                        <div className="flex-1 text-left">
                          <p className="font-medium">{cmd.title}</p>
                          {cmd.description && (
                            <p className="text-sm text-slate-500">
                              {cmd.description}
                            </p>
                          )}
                        </div>
                        {cmd.shortcut && (
                          <span className="text-xs text-slate-500 font-mono bg-slate-800 px-2 py-0.5 rounded">
                            {cmd.shortcut}
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-2 border-t border-white/10 text-xs text-slate-500">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-slate-400">↑</kbd>
              <kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-slate-400">↓</kbd>
              <span className="ml-1">Navigate</span>
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-slate-400">↵</kbd>
              <span className="ml-1">Select</span>
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-slate-400">esc</kbd>
              <span className="ml-1">Close</span>
            </span>
          </div>
          <span>Type to filter</span>
        </div>
      </div>
    </div>
  );
}

export function CommandPaletteProvider({ children }: { children: React.ReactNode }) {
  const [isOpen, setIsOpen] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Global keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd+K / Ctrl+K to open
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsOpen(true);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <>
      {children}
      {mounted &&
        createPortal(
          <CommandPaletteContent isOpen={isOpen} onClose={() => setIsOpen(false)} />,
          document.body
        )}
    </>
  );
}

export default CommandPaletteProvider;
