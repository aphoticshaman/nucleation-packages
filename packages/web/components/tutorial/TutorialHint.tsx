'use client';

/**
 * Tutorial Hint Component
 *
 * Game-style hint popup that:
 * - Can be individually dismissed
 * - Has "learn more" option
 * - Has "don't show again" option
 * - Non-intrusive positioning
 * - Keyboard accessible
 *
 * The user is always in control.
 */

import { useState, useEffect, useCallback } from 'react';
import { X, ChevronRight, HelpCircle, Lightbulb, Keyboard } from 'lucide-react';
import {
  TutorialHint as TutorialHintType,
  TutorialProgress,
  TutorialAction,
  loadTutorialProgress,
  saveTutorialProgress,
  updateHintState,
  getHintState,
  getNextHint,
  HintContext,
} from '@/lib/tutorial/TutorialSystem';

// ============================================
// Hint Popup Component
// ============================================

interface TutorialHintProps {
  hint: TutorialHintType;
  onDismiss: () => void;
  onPermanentDismiss: () => void;
  onAction: (action: TutorialAction) => void;
}

export function TutorialHintPopup({
  hint,
  onDismiss,
  onPermanentDismiss,
  onAction,
}: TutorialHintProps) {
  const [expanded, setExpanded] = useState(false);

  // Keyboard handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onDismiss();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onDismiss]);

  const categoryIcons: Record<string, typeof Lightbulb> = {
    navigation: Keyboard,
    shortcuts: Keyboard,
    default: Lightbulb,
  };

  const Icon = categoryIcons[hint.category] || categoryIcons.default;

  return (
    <div
      className={`
        tutorial-hint
        fixed z-50 max-w-sm
        bg-gray-900/95 backdrop-blur-sm
        border border-blue-500/30 rounded-lg
        shadow-lg shadow-blue-500/10
        animate-in fade-in slide-in-from-bottom-2
        duration-200
        ${getPositionClasses(hint.position, hint.targetArea)}
      `}
      role="dialog"
      aria-labelledby={`hint-title-${hint.id}`}
    >
      {/* Header */}
      <div className="flex items-start gap-3 p-3 border-b border-gray-700/50">
        <div className="p-1.5 bg-blue-500/20 rounded-lg">
          <Icon className="w-4 h-4 text-blue-400" />
        </div>
        <div className="flex-1 min-w-0">
          <h3
            id={`hint-title-${hint.id}`}
            className="font-medium text-white text-sm"
          >
            {hint.title}
          </h3>
          <span className="text-xs text-gray-500 capitalize">
            {hint.category}
          </span>
        </div>
        <button
          onClick={onDismiss}
          className="p-1 text-gray-500 hover:text-gray-300 transition-colors"
          aria-label="Dismiss hint"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Content */}
      <div className="p-3">
        <p className="text-sm text-gray-300">
          {expanded ? hint.fullText : hint.shortText}
        </p>

        {!expanded && hint.fullText !== hint.shortText && (
          <button
            onClick={() => setExpanded(true)}
            className="mt-2 text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
          >
            Tell me more <ChevronRight className="w-3 h-3" />
          </button>
        )}
      </div>

      {/* Actions */}
      <div className="flex flex-wrap gap-2 p-3 pt-0">
        {hint.actions?.map((action, idx) => (
          <HintActionButton
            key={idx}
            action={action}
            onClick={() => {
              if (action.type === 'never-show') {
                onPermanentDismiss();
              } else if (action.type === 'dismiss') {
                onDismiss();
              } else {
                onAction(action);
              }
            }}
          />
        ))}

        {/* Default dismiss if no actions */}
        {!hint.actions?.length && (
          <button
            onClick={onDismiss}
            className="px-3 py-1.5 text-xs bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors"
          >
            Got it
          </button>
        )}
      </div>

      {/* Quick dismiss footer */}
      <div className="px-3 py-2 bg-gray-800/50 border-t border-gray-700/50 rounded-b-lg">
        <button
          onClick={onPermanentDismiss}
          className="text-xs text-gray-500 hover:text-gray-400 transition-colors"
        >
          Don't show this again
        </button>
      </div>
    </div>
  );
}

// ============================================
// Action Button Component
// ============================================

interface HintActionButtonProps {
  action: TutorialAction;
  onClick: () => void;
}

function HintActionButton({ action, onClick }: HintActionButtonProps) {
  const baseClasses = 'px-3 py-1.5 text-xs rounded transition-colors';

  const typeStyles: Record<TutorialAction['type'], string> = {
    dismiss: 'bg-gray-700 hover:bg-gray-600 text-white',
    'learn-more': 'bg-blue-600 hover:bg-blue-500 text-white',
    'try-it': 'bg-green-600 hover:bg-green-500 text-white',
    'never-show': 'bg-transparent hover:bg-gray-800 text-gray-400',
    custom: 'bg-gray-700 hover:bg-gray-600 text-white',
  };

  return (
    <button onClick={onClick} className={`${baseClasses} ${typeStyles[action.type]}`}>
      {action.label}
      {action.shortcut && (
        <kbd className="ml-1.5 px-1 py-0.5 bg-black/30 rounded text-[10px] font-mono">
          {action.shortcut}
        </kbd>
      )}
    </button>
  );
}

// ============================================
// Position Calculator
// ============================================

function getPositionClasses(
  position?: TutorialHintType['position'],
  targetArea?: TutorialHintType['targetArea']
): string {
  // If floating, use target area or default to bottom-right
  if (position === 'floating' || !position) {
    switch (targetArea) {
      case 'top-left':
        return 'top-20 left-4';
      case 'top-right':
        return 'top-20 right-4';
      case 'bottom-left':
        return 'bottom-20 left-4';
      case 'center':
        return 'top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2';
      case 'bottom-right':
      default:
        return 'bottom-20 right-4';
    }
  }

  // For positioned hints, they'll be placed near target element
  // This is handled by the TutorialManager with DOM positioning
  return 'bottom-20 right-4';
}

// ============================================
// Tutorial Manager (Provider)
// ============================================

interface TutorialManagerProps {
  children: React.ReactNode;
  userTier: 'explorer' | 'analyst' | 'strategist' | 'architect';
  userFeatures: string[];
}

export function TutorialManager({
  children,
  userTier,
  userFeatures,
}: TutorialManagerProps) {
  const [progress, setProgress] = useState<TutorialProgress | null>(null);
  const [currentHint, setCurrentHint] = useState<TutorialHintType | null>(null);
  const [idleTime, setIdleTime] = useState(0);

  // Load progress on mount
  useEffect(() => {
    setProgress(loadTutorialProgress());
  }, []);

  // Track idle time
  useEffect(() => {
    let idleTimer: NodeJS.Timeout;
    let interval: NodeJS.Timeout;

    const resetIdle = () => {
      setIdleTime(0);
    };

    const incrementIdle = () => {
      setIdleTime((prev) => prev + 1);
    };

    window.addEventListener('mousemove', resetIdle);
    window.addEventListener('keydown', resetIdle);
    window.addEventListener('click', resetIdle);

    interval = setInterval(incrementIdle, 1000);

    return () => {
      window.removeEventListener('mousemove', resetIdle);
      window.removeEventListener('keydown', resetIdle);
      window.removeEventListener('click', resetIdle);
      clearInterval(interval);
    };
  }, []);

  // Check for hints periodically
  useEffect(() => {
    if (!progress || progress.tutorialMode === 'off') return;
    if (currentHint) return; // Don't show new hint if one is visible

    const checkForHint = () => {
      const context: HintContext = {
        currentPage: typeof window !== 'undefined' ? window.location.pathname : '/',
        visibleWidgets: getVisibleWidgets(),
        userTier,
        userFeatures,
        idleTime,
        sessionActions: [],
      };

      const hint = getNextHint(progress, context);
      if (hint) {
        setCurrentHint(hint);

        // Mark as shown
        const newProgress = updateHintState(progress, hint.id, {
          shown: true,
          showCount: (getHintState(progress, hint.id).showCount || 0) + 1,
          lastShown: Date.now(),
        });
        setProgress(newProgress);
        saveTutorialProgress(newProgress);
      }
    };

    // Check after initial delay, then periodically
    const initialTimer = setTimeout(checkForHint, 3000);
    const interval = setInterval(checkForHint, 30000);

    return () => {
      clearTimeout(initialTimer);
      clearInterval(interval);
    };
  }, [progress, currentHint, idleTime, userTier, userFeatures]);

  const handleDismiss = useCallback(() => {
    if (!progress || !currentHint) return;

    const newProgress = updateHintState(progress, currentHint.id, {
      dismissed: true,
    });
    setProgress(newProgress);
    saveTutorialProgress(newProgress);
    setCurrentHint(null);
  }, [progress, currentHint]);

  const handlePermanentDismiss = useCallback(() => {
    if (!progress || !currentHint) return;

    const newProgress = updateHintState(progress, currentHint.id, {
      permanentlyDismissed: true,
    });
    setProgress(newProgress);
    saveTutorialProgress(newProgress);
    setCurrentHint(null);
  }, [progress, currentHint]);

  const handleAction = useCallback(
    (action: TutorialAction) => {
      if (!progress || !currentHint) return;

      if (action.url) {
        window.open(action.url, '_blank');
      }

      if (action.shortcut) {
        // Could trigger the shortcut here
        console.log(`Trigger shortcut: ${action.shortcut}`);
      }

      if (action.action) {
        action.action();
      }

      // Mark action completed
      const newProgress = updateHintState(progress, currentHint.id, {
        completedAction: action.label,
        dismissed: true,
      });
      setProgress(newProgress);
      saveTutorialProgress(newProgress);
      setCurrentHint(null);
    },
    [progress, currentHint]
  );

  return (
    <>
      {children}

      {currentHint && (
        <TutorialHintPopup
          hint={currentHint}
          onDismiss={handleDismiss}
          onPermanentDismiss={handlePermanentDismiss}
          onAction={handleAction}
        />
      )}
    </>
  );
}

// ============================================
// Helpers
// ============================================

function getVisibleWidgets(): string[] {
  if (typeof document === 'undefined') return [];

  // Check for common widget selectors
  const selectors = [
    '#tree-navigator',
    '#filter-panel',
    '#relation-graph',
    '#threat-matrix',
    '#cognitive-panel',
    '#timeline',
    '#search-bar',
    '#comparison-panel',
    '#exec-summary',
  ];

  return selectors.filter((sel) => {
    const el = document.querySelector(sel);
    return el && el.getBoundingClientRect().height > 0;
  });
}

// ============================================
// Tutorial Settings Panel
// ============================================

interface TutorialSettingsProps {
  onClose: () => void;
}

export function TutorialSettings({ onClose }: TutorialSettingsProps) {
  const [progress, setProgress] = useState<TutorialProgress | null>(null);

  useEffect(() => {
    setProgress(loadTutorialProgress());
  }, []);

  if (!progress) return null;

  const handleModeChange = (mode: 'full' | 'minimal' | 'off') => {
    const newProgress = { ...progress, tutorialMode: mode };
    setProgress(newProgress);
    saveTutorialProgress(newProgress);
  };

  const handleReset = () => {
    if (confirm('Reset all tutorial progress? This will re-enable all hints.')) {
      const fresh = loadTutorialProgress();
      fresh.hints = {};
      fresh.totalHintsShown = 0;
      fresh.totalHintsDismissed = 0;
      fresh.permanentlyDismissedCount = 0;
      setProgress(fresh);
      saveTutorialProgress(fresh);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-gray-900 border border-gray-700 rounded-lg shadow-xl max-w-md w-full mx-4">
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <HelpCircle className="w-5 h-5 text-blue-400" />
            Tutorial Settings
          </h2>
          <button
            onClick={onClose}
            className="p-1 text-gray-500 hover:text-gray-300"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-4 space-y-4">
          {/* Mode Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Tutorial Mode
            </label>
            <div className="space-y-2">
              {[
                {
                  value: 'full',
                  label: 'Full',
                  desc: 'Show all helpful hints as you explore',
                },
                {
                  value: 'minimal',
                  label: 'Minimal',
                  desc: 'Only show critical hints',
                },
                {
                  value: 'off',
                  label: 'Off',
                  desc: "I know what I'm doing",
                },
              ].map((option) => (
                <label
                  key={option.value}
                  className={`
                    flex items-start gap-3 p-3 rounded-lg cursor-pointer
                    border transition-colors
                    ${
                      progress.tutorialMode === option.value
                        ? 'border-blue-500 bg-blue-500/10'
                        : 'border-gray-700 hover:border-gray-600'
                    }
                  `}
                >
                  <input
                    type="radio"
                    name="tutorialMode"
                    value={option.value}
                    checked={progress.tutorialMode === option.value}
                    onChange={() =>
                      handleModeChange(option.value as 'full' | 'minimal' | 'off')
                    }
                    className="mt-1"
                  />
                  <div>
                    <div className="font-medium text-white">{option.label}</div>
                    <div className="text-sm text-gray-400">{option.desc}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Stats */}
          <div className="bg-gray-800/50 rounded-lg p-3">
            <div className="text-sm text-gray-400">Progress</div>
            <div className="mt-2 grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold text-white">
                  {progress.totalHintsShown}
                </div>
                <div className="text-xs text-gray-500">Hints Shown</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-white">
                  {progress.totalHintsDismissed}
                </div>
                <div className="text-xs text-gray-500">Dismissed</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-white">
                  {progress.permanentlyDismissedCount}
                </div>
                <div className="text-xs text-gray-500">Hidden Forever</div>
              </div>
            </div>
          </div>

          {/* Reset */}
          <button
            onClick={handleReset}
            className="w-full py-2 text-sm text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded transition-colors"
          >
            Reset Tutorial Progress
          </button>
        </div>
      </div>
    </div>
  );
}
