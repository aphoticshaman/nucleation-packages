'use client';

import { useState, useEffect, useCallback } from 'react';

interface ShortcutCategory {
  name: string;
  icon: string;
  shortcuts: Shortcut[];
}

interface Shortcut {
  keys: string[];
  action: string;
  description: string;
  global?: boolean;
}

interface KeyboardShortcutsProps {
  isOpen: boolean;
  onClose: () => void;
  customShortcuts?: ShortcutCategory[];
  onShortcutTriggered?: (action: string) => void;
}

// Component 45: Keyboard Shortcut Reference Modal
export function KeyboardShortcuts({
  isOpen,
  onClose,
  customShortcuts = [],
  onShortcutTriggered,
}: KeyboardShortcutsProps) {
  const [activeCategory, setActiveCategory] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');

  const defaultCategories: ShortcutCategory[] = [
    {
      name: 'Navigation',
      icon: '⌨',
      shortcuts: [
        { keys: ['?'], action: 'help', description: 'Show keyboard shortcuts', global: true },
        { keys: ['Esc'], action: 'close', description: 'Close modal/panel', global: true },
        { keys: ['G', 'D'], action: 'goto-dashboard', description: 'Go to Dashboard' },
        { keys: ['G', 'M'], action: 'goto-map', description: 'Go to Map View' },
        { keys: ['G', 'N'], action: 'goto-network', description: 'Go to Network Graph' },
        { keys: ['G', 'T'], action: 'goto-timeline', description: 'Go to Timeline' },
        { keys: ['G', 'A'], action: 'goto-alerts', description: 'Go to Alerts' },
      ],
    },
    {
      name: 'Analysis',
      icon: '🔬',
      shortcuts: [
        { keys: ['Ctrl', 'Enter'], action: 'run-analysis', description: 'Run analysis on selection' },
        { keys: ['Ctrl', 'Shift', 'E'], action: 'export-report', description: 'Export current report' },
        { keys: ['Ctrl', 'S'], action: 'save-view', description: 'Save current view state' },
        { keys: ['R'], action: 'refresh-data', description: 'Refresh data' },
        { keys: ['F'], action: 'toggle-fullscreen', description: 'Toggle fullscreen mode' },
        { keys: ['Ctrl', 'K'], action: 'command-palette', description: 'Open command palette', global: true },
      ],
    },
    {
      name: 'Selection',
      icon: '✓',
      shortcuts: [
        { keys: ['Click'], action: 'select-single', description: 'Select item' },
        { keys: ['Ctrl', 'Click'], action: 'select-multi', description: 'Add to selection' },
        { keys: ['Shift', 'Click'], action: 'select-range', description: 'Select range' },
        { keys: ['Ctrl', 'A'], action: 'select-all', description: 'Select all visible' },
        { keys: ['Ctrl', 'D'], action: 'deselect-all', description: 'Clear selection' },
        { keys: ['Delete'], action: 'delete-selected', description: 'Delete selected items' },
      ],
    },
    {
      name: 'View Controls',
      icon: '👁',
      shortcuts: [
        { keys: ['+'], action: 'zoom-in', description: 'Zoom in' },
        { keys: ['-'], action: 'zoom-out', description: 'Zoom out' },
        { keys: ['0'], action: 'zoom-reset', description: 'Reset zoom' },
        { keys: ['←', '→'], action: 'pan-horizontal', description: 'Pan left/right' },
        { keys: ['↑', '↓'], action: 'pan-vertical', description: 'Pan up/down' },
        { keys: ['H'], action: 'toggle-heatmap', description: 'Toggle heatmap overlay' },
        { keys: ['L'], action: 'toggle-labels', description: 'Toggle labels' },
      ],
    },
    {
      name: 'Risk Assessment',
      icon: '⚠',
      shortcuts: [
        { keys: ['1'], action: 'filter-critical', description: 'Show critical only' },
        { keys: ['2'], action: 'filter-high', description: 'Show high and above' },
        { keys: ['3'], action: 'filter-medium', description: 'Show medium and above' },
        { keys: ['4'], action: 'filter-all', description: 'Show all risk levels' },
        { keys: ['Ctrl', 'Shift', 'R'], action: 'recalculate-risk', description: 'Recalculate risk scores' },
        { keys: ['E'], action: 'show-epistemic', description: 'Show epistemic bounds' },
      ],
    },
    {
      name: 'Annotations',
      icon: '📝',
      shortcuts: [
        { keys: ['N'], action: 'new-annotation', description: 'Create annotation at cursor' },
        { keys: ['Ctrl', 'Shift', 'N'], action: 'new-global-note', description: 'Create global note' },
        { keys: ['T'], action: 'toggle-annotations', description: 'Toggle annotation visibility' },
        { keys: ['['], action: 'prev-annotation', description: 'Previous annotation' },
        { keys: [']'], action: 'next-annotation', description: 'Next annotation' },
      ],
    },
    {
      name: 'Watchlists',
      icon: '👁',
      shortcuts: [
        { keys: ['W'], action: 'add-to-watchlist', description: 'Add selection to watchlist' },
        { keys: ['Ctrl', 'W'], action: 'manage-watchlists', description: 'Open watchlist manager' },
        { keys: ['Shift', 'W'], action: 'create-watchlist', description: 'Create new watchlist' },
      ],
    },
    {
      name: 'Time Controls',
      icon: '⏱',
      shortcuts: [
        { keys: ['Space'], action: 'play-pause', description: 'Play/pause timeline' },
        { keys: ['Shift', '←'], action: 'step-back', description: 'Step back in time' },
        { keys: ['Shift', '→'], action: 'step-forward', description: 'Step forward in time' },
        { keys: ['Home'], action: 'goto-start', description: 'Go to earliest data' },
        { keys: ['End'], action: 'goto-end', description: 'Go to latest data' },
      ],
    },
  ];

  const allCategories = [...defaultCategories, ...customShortcuts];

  // Filter shortcuts based on search
  const filteredCategories = searchQuery
    ? allCategories.map(cat => ({
        ...cat,
        shortcuts: cat.shortcuts.filter(
          s =>
            s.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
            s.action.toLowerCase().includes(searchQuery.toLowerCase()) ||
            s.keys.some(k => k.toLowerCase().includes(searchQuery.toLowerCase()))
        ),
      })).filter(cat => cat.shortcuts.length > 0)
    : allCategories;

  // Handle keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      } else if (e.key === 'ArrowUp') {
        setActiveCategory(prev => Math.max(0, prev - 1));
      } else if (e.key === 'ArrowDown') {
        setActiveCategory(prev => Math.min(filteredCategories.length - 1, prev + 1));
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose, filteredCategories.length]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-slate-900 rounded-xl border border-slate-700 shadow-2xl w-full max-w-3xl max-h-[80vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-700">
          <div>
            <h2 className="text-lg font-medium text-slate-200">Keyboard Shortcuts</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Press <kbd className="px-1 py-0.5 bg-slate-800 rounded text-cyan-400">?</kbd> anywhere to show this reference
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-slate-200 transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Search */}
        <div className="p-3 border-b border-slate-800">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search shortcuts..."
            className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-sm text-slate-200 placeholder:text-slate-500 focus:border-cyan-500 focus:outline-none"
            autoFocus
          />
        </div>

        {/* Content */}
        <div className="flex max-h-[calc(80vh-140px)]">
          {/* Category sidebar */}
          <div className="w-48 border-r border-slate-800 overflow-y-auto">
            {filteredCategories.map((cat, i) => (
              <button
                key={cat.name}
                onClick={() => setActiveCategory(i)}
                className={`w-full text-left px-4 py-3 flex items-center gap-2 transition-colors ${
                  activeCategory === i
                    ? 'bg-cyan-500/10 text-cyan-400 border-r-2 border-cyan-500'
                    : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                }`}
              >
                <span>{cat.icon}</span>
                <span className="text-sm">{cat.name}</span>
                <span className="ml-auto text-xs text-slate-500">{cat.shortcuts.length}</span>
              </button>
            ))}
          </div>

          {/* Shortcut list */}
          <div className="flex-1 overflow-y-auto p-4">
            {filteredCategories[activeCategory] && (
              <div className="space-y-2">
                {filteredCategories[activeCategory].shortcuts.map((shortcut) => (
                  <div
                    key={shortcut.action}
                    className="flex items-center justify-between p-2 rounded hover:bg-slate-800/50 group"
                  >
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-1">
                        {shortcut.keys.map((key, i) => (
                          <span key={i} className="flex items-center">
                            <kbd className="min-w-[24px] px-1.5 py-0.5 bg-slate-800 border border-slate-700 rounded text-xs text-slate-300 text-center font-mono">
                              {key}
                            </kbd>
                            {i < shortcut.keys.length - 1 && (
                              <span className="mx-1 text-slate-600">+</span>
                            )}
                          </span>
                        ))}
                      </div>
                      {shortcut.global && (
                        <span className="px-1 py-0.5 bg-cyan-500/20 text-cyan-400 text-xs rounded">
                          Global
                        </span>
                      )}
                    </div>
                    <span className="text-sm text-slate-400 group-hover:text-slate-200">
                      {shortcut.description}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-3 border-t border-slate-800 bg-slate-800/30">
          <div className="flex items-center gap-4 text-xs text-slate-500">
            <span><kbd className="px-1 bg-slate-700 rounded">↑↓</kbd> Navigate</span>
            <span><kbd className="px-1 bg-slate-700 rounded">Esc</kbd> Close</span>
          </div>
          <span className="text-xs text-slate-500">
            {allCategories.reduce((sum, cat) => sum + cat.shortcuts.length, 0)} shortcuts available
          </span>
        </div>
      </div>
    </div>
  );
}

// Hook for registering global shortcuts
export function useKeyboardShortcuts(
  shortcuts: { keys: string[]; handler: () => void }[],
  enabled = true
) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!enabled) return;

      // Ignore if user is typing in an input
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
        return;
      }

      for (const shortcut of shortcuts) {
        const { keys, handler } = shortcut;

        // Check modifier keys
        const needsCtrl = keys.includes('Ctrl');
        const needsShift = keys.includes('Shift');
        const needsAlt = keys.includes('Alt');
        const needsMeta = keys.includes('Meta');

        if (needsCtrl !== e.ctrlKey) continue;
        if (needsShift !== e.shiftKey) continue;
        if (needsAlt !== e.altKey) continue;
        if (needsMeta !== e.metaKey) continue;

        // Check the main key
        const mainKey = keys.find(k => !['Ctrl', 'Shift', 'Alt', 'Meta'].includes(k));
        if (mainKey && e.key.toLowerCase() === mainKey.toLowerCase()) {
          e.preventDefault();
          handler();
          return;
        }
      }
    },
    [shortcuts, enabled]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);
}

// Quick shortcut hint component
export function ShortcutHint({ keys, className = '' }: { keys: string[]; className?: string }) {
  return (
    <span className={`inline-flex items-center gap-0.5 ${className}`}>
      {keys.map((key, i) => (
        <span key={i} className="flex items-center">
          <kbd className="px-1 py-0.5 bg-slate-800/50 border border-slate-700/50 rounded text-xs text-slate-400 font-mono">
            {key}
          </kbd>
          {i < keys.length - 1 && <span className="mx-0.5 text-slate-600 text-xs">+</span>}
        </span>
      ))}
    </span>
  );
}

// Compact shortcut reference for toolbars
export function CompactShortcutRef() {
  const [isOpen, setIsOpen] = useState(false);

  useKeyboardShortcuts([
    { keys: ['?'], handler: () => setIsOpen(true) },
  ]);

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="p-2 text-slate-400 hover:text-cyan-400 transition-colors"
        title="Keyboard shortcuts (?)"
      >
        <span className="text-lg">⌨</span>
      </button>
      <KeyboardShortcuts isOpen={isOpen} onClose={() => setIsOpen(false)} />
    </>
  );
}
