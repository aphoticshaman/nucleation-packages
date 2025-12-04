'use client';

import { useEffect, useCallback } from 'react';
import { X, Command, ArrowUp, ArrowDown, ArrowLeft, ArrowRight } from 'lucide-react';

interface KeyboardShortcutsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const SHORTCUT_GROUPS = [
  {
    title: 'Navigation',
    shortcuts: [
      { keys: ['g', 'd'], description: 'Go to Dashboard' },
      { keys: ['g', 'm'], description: 'Go to Map' },
      { keys: ['g', 'p'], description: 'Go to Packages' },
      { keys: ['g', 'b'], description: 'Go to Briefings' },
      { keys: ['g', 's'], description: 'Go to Settings' },
      { keys: ['g', 'i'], description: 'Go to Integrations' },
    ],
  },
  {
    title: 'Command Palette',
    shortcuts: [
      { keys: ['⌘', 'k'], description: 'Open command palette' },
      { keys: ['⌘', 'p'], description: 'Search nations' },
      { keys: ['⌘', '/'], description: 'Show shortcuts (this modal)' },
      { keys: ['Escape'], description: 'Close modal / cancel' },
    ],
  },
  {
    title: 'Map Controls',
    shortcuts: [
      { keys: ['+'], description: 'Zoom in' },
      { keys: ['-'], description: 'Zoom out' },
      { keys: ['0'], description: 'Reset zoom' },
      { keys: ['↑', '↓', '←', '→'], description: 'Pan map' },
      { keys: ['r'], description: 'Rotate view' },
      { keys: ['f'], description: 'Toggle fullscreen' },
    ],
  },
  {
    title: 'Analysis',
    shortcuts: [
      { keys: ['c'], description: 'Start cascade simulation' },
      { keys: ['t'], description: 'Toggle timeline' },
      { keys: ['l'], description: 'Toggle layer panel' },
      { keys: ['a'], description: 'Add annotation' },
      { keys: ['s'], description: 'Save snapshot' },
    ],
  },
  {
    title: 'Data & Export',
    shortcuts: [
      { keys: ['⌘', 's'], description: 'Save current view' },
      { keys: ['⌘', 'e'], description: 'Export data' },
      { keys: ['⌘', 'r'], description: 'Refresh data' },
      { keys: ['⌘', 'd'], description: 'Download report' },
    ],
  },
  {
    title: 'Alerts',
    shortcuts: [
      { keys: ['n'], description: 'Next alert' },
      { keys: ['p'], description: 'Previous alert' },
      { keys: ['m'], description: 'Mark as read' },
      { keys: ['x'], description: 'Dismiss alert' },
    ],
  },
];

export default function KeyboardShortcutsModal({ isOpen, onClose }: KeyboardShortcutsModalProps) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    },
    [onClose]
  );

  useEffect(() => {
    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
    };
  }, [isOpen, handleKeyDown]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-3xl max-h-[85vh] bg-[rgba(18,18,26,0.95)] backdrop-blur-xl border border-white/[0.08] rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/[0.06]">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center">
              <Command className="w-4 h-4 text-white" />
            </div>
            <h2 className="text-lg font-semibold text-white">Keyboard Shortcuts</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-white/5 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(85vh-80px)]">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {SHORTCUT_GROUPS.map((group) => (
              <div key={group.title}>
                <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-3">
                  {group.title}
                </h3>
                <div className="space-y-2">
                  {group.shortcuts.map((shortcut, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between py-2 px-3 bg-black/20 rounded-lg"
                    >
                      <span className="text-sm text-slate-300">{shortcut.description}</span>
                      <div className="flex items-center gap-1">
                        {shortcut.keys.map((key, keyIdx) => (
                          <span key={keyIdx}>
                            {keyIdx > 0 && (
                              <span className="text-slate-600 mx-1">+</span>
                            )}
                            <KeyBadge keyName={key} />
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Pro tip */}
          <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/20 rounded-xl">
            <p className="text-sm text-blue-300">
              <strong>Pro tip:</strong> Press{' '}
              <KeyBadge keyName="⌘" />{' '}
              <span className="text-blue-400">+</span>{' '}
              <KeyBadge keyName="k" />{' '}
              anywhere to open the command palette and search for any action.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-white/[0.06] flex items-center justify-between text-sm">
          <span className="text-slate-500">
            Press <KeyBadge keyName="?" /> to toggle this modal
          </span>
          <span className="text-slate-500">
            <KeyBadge keyName="Esc" /> to close
          </span>
        </div>
      </div>
    </div>
  );
}

function KeyBadge({ keyName }: { keyName: string }) {
  // Handle arrow keys
  if (keyName === '↑' || keyName === 'ArrowUp') {
    return (
      <span className="inline-flex items-center justify-center w-6 h-6 bg-black/30 border border-white/[0.1] rounded text-slate-300">
        <ArrowUp className="w-3 h-3" />
      </span>
    );
  }
  if (keyName === '↓' || keyName === 'ArrowDown') {
    return (
      <span className="inline-flex items-center justify-center w-6 h-6 bg-black/30 border border-white/[0.1] rounded text-slate-300">
        <ArrowDown className="w-3 h-3" />
      </span>
    );
  }
  if (keyName === '←' || keyName === 'ArrowLeft') {
    return (
      <span className="inline-flex items-center justify-center w-6 h-6 bg-black/30 border border-white/[0.1] rounded text-slate-300">
        <ArrowLeft className="w-3 h-3" />
      </span>
    );
  }
  if (keyName === '→' || keyName === 'ArrowRight') {
    return (
      <span className="inline-flex items-center justify-center w-6 h-6 bg-black/30 border border-white/[0.1] rounded text-slate-300">
        <ArrowRight className="w-3 h-3" />
      </span>
    );
  }
  if (keyName === '⌘') {
    return (
      <span className="inline-flex items-center justify-center w-6 h-6 bg-black/30 border border-white/[0.1] rounded text-slate-300">
        <Command className="w-3 h-3" />
      </span>
    );
  }

  return (
    <span className="inline-flex items-center justify-center min-w-[24px] h-6 px-1.5 bg-black/30 border border-white/[0.1] rounded text-xs font-mono text-slate-300 uppercase">
      {keyName}
    </span>
  );
}
