/**
 * Game-Style Tutorial System for LatticeForge
 *
 * Non-intrusive hints that respect user agency:
 * - Individually dismissable
 * - "Tell me more" option for curious users
 * - "Don't show again" for power users who know their shit
 * - Context-aware (only shows relevant hints)
 * - Tracks progress without being preachy
 *
 * Philosophy: Assist, don't annoy. The user is in control.
 */

// ============================================
// Types
// ============================================

export type HintId = string;

export type HintCategory =
  | 'navigation'
  | 'filtering'
  | 'visualization'
  | 'analysis'
  | 'shortcuts'
  | 'collaboration'
  | 'export'
  | 'advanced';

export type HintTrigger =
  | 'first-visit'
  | 'first-use'
  | 'idle'
  | 'error'
  | 'achievement'
  | 'contextual'
  | 'manual';

export type HintPriority = 'low' | 'medium' | 'high' | 'critical';

export interface TutorialHint {
  id: HintId;
  category: HintCategory;
  priority: HintPriority;
  trigger: HintTrigger;

  // Content
  title: string;
  shortText: string;
  fullText: string;
  learnMoreUrl?: string;

  // Targeting
  targetSelector?: string; // CSS selector for element to highlight
  targetArea?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' | 'center';
  position?: 'above' | 'below' | 'left' | 'right' | 'floating';

  // Conditions
  showAfterHints?: HintId[]; // Prerequisites
  requiresFeature?: string; // Only show if user has this feature
  minTier?: 'explorer' | 'analyst' | 'strategist' | 'architect';
  maxShowCount?: number; // Max times to show before auto-suppressing

  // Actions
  actions?: TutorialAction[];

  // Meta
  addedInVersion?: string;
}

export interface TutorialAction {
  label: string;
  type: 'dismiss' | 'learn-more' | 'try-it' | 'never-show' | 'custom';
  action?: () => void;
  url?: string;
  shortcut?: string;
}

export interface HintState {
  shown: boolean;
  showCount: number;
  dismissed: boolean;
  permanentlyDismissed: boolean;
  lastShown?: number;
  completedAction?: string;
}

export interface TutorialProgress {
  hints: Record<HintId, HintState>;
  categoriesCompleted: HintCategory[];
  totalHintsShown: number;
  totalHintsDismissed: number;
  permanentlyDismissedCount: number;
  tutorialMode: 'full' | 'minimal' | 'off';
  showShortcuts: boolean;
  lastActivity: number;
}

// ============================================
// Default Hints Library
// ============================================

export const TUTORIAL_HINTS: TutorialHint[] = [
  // === Navigation ===
  {
    id: 'nav-keyboard-basics',
    category: 'navigation',
    priority: 'high',
    trigger: 'first-visit',
    title: 'Keyboard Navigation',
    shortText: 'Press ? anytime to see all keyboard shortcuts.',
    fullText: 'LatticeForge is designed for keyboard power users. Press ? to see available shortcuts. Most actions have a keyboard equivalent. Press / to search anything.',
    position: 'floating',
    actions: [
      { label: 'Got it', type: 'dismiss' },
      { label: 'Show shortcuts', type: 'try-it', shortcut: '?' },
      { label: 'Don\'t remind me', type: 'never-show' },
    ],
  },
  {
    id: 'nav-search',
    category: 'navigation',
    priority: 'high',
    trigger: 'first-visit',
    title: 'Universal Search',
    shortText: 'Press / to search entities, events, reports, and settings.',
    fullText: 'The search bar understands natural language. Try "conflicts in Asia last month" or "entities related to OPEC". You can also search for settings and commands.',
    targetSelector: '#search-bar',
    position: 'below',
    showAfterHints: ['nav-keyboard-basics'],
    actions: [
      { label: 'Try it', type: 'try-it', shortcut: '/' },
      { label: 'Later', type: 'dismiss' },
      { label: 'I know', type: 'never-show' },
    ],
  },
  {
    id: 'nav-3d-tree',
    category: 'navigation',
    priority: 'medium',
    trigger: 'first-use',
    title: '3D Intel Navigator',
    shortText: 'Drag to rotate. Scroll to zoom. Click nodes to expand.',
    fullText: 'The 3D tree shows intel across time. Vertical axis is temporal: past at bottom, future at top. Related items cluster together. Double-click to focus. Press R to reset view.',
    targetSelector: '#tree-navigator',
    position: 'right',
    actions: [
      { label: 'Explore', type: 'dismiss' },
      { label: 'Controls guide', type: 'learn-more', url: '/help/3d-navigator' },
      { label: 'Got it', type: 'never-show' },
    ],
  },

  // === Filtering ===
  {
    id: 'filter-basics',
    category: 'filtering',
    priority: 'high',
    trigger: 'first-use',
    title: 'Stacking Filters',
    shortText: 'Filters stack. Add multiple to narrow results.',
    fullText: 'Click multiple filter values to combine them. Same category = OR logic. Different categories = AND logic. Ctrl+click to select multiple quickly.',
    targetSelector: '#filter-panel',
    position: 'right',
    actions: [
      { label: 'Makes sense', type: 'dismiss' },
      { label: 'Examples', type: 'learn-more', url: '/help/filtering' },
      { label: 'Don\'t show', type: 'never-show' },
    ],
  },
  {
    id: 'filter-save',
    category: 'filtering',
    priority: 'medium',
    trigger: 'contextual',
    title: 'Save This Filter',
    shortText: 'Save filter combinations you use often.',
    fullText: 'Click the save icon to keep this filter combination. Give it a name and access it from the saved filters dropdown. Great for daily workflows.',
    targetSelector: '#filter-save-btn',
    position: 'below',
    showAfterHints: ['filter-basics'],
    minTier: 'analyst',
    actions: [
      { label: 'Save current', type: 'try-it' },
      { label: 'Maybe later', type: 'dismiss' },
      { label: 'I know', type: 'never-show' },
    ],
  },
  {
    id: 'filter-temporal',
    category: 'filtering',
    priority: 'medium',
    trigger: 'contextual',
    title: 'Time Travel',
    shortText: 'Slide the timeline to see historical or projected states.',
    fullText: 'The temporal filter doesn\'t just filter by date - it shows the state of relationships and situations at that point in time. Enable projections to see likely futures.',
    targetSelector: '#temporal-filter',
    position: 'above',
    actions: [
      { label: 'Cool', type: 'dismiss' },
      { label: 'How projections work', type: 'learn-more', url: '/help/temporal-analysis' },
      { label: 'Got it', type: 'never-show' },
    ],
  },

  // === Visualization ===
  {
    id: 'viz-relation-graph',
    category: 'visualization',
    priority: 'medium',
    trigger: 'first-use',
    title: 'Entity Relationships',
    shortText: 'Double-click to expand connections. Edge color shows relationship type.',
    fullText: 'Green = alliance/cooperation. Red = hostile. Yellow = economic. Line thickness = relationship strength. Double-click any node to show its connections. Right-click for details.',
    targetSelector: '#relation-graph',
    position: 'left',
    actions: [
      { label: 'Explore', type: 'dismiss' },
      { label: 'Graph controls', type: 'learn-more', url: '/help/relation-graph' },
      { label: 'Understood', type: 'never-show' },
    ],
  },
  {
    id: 'viz-threat-matrix',
    category: 'visualization',
    priority: 'medium',
    trigger: 'first-use',
    title: 'Reading the Threat Matrix',
    shortText: 'Position shows severity vs probability. Size shows impact.',
    fullText: 'X-axis is probability (left=unlikely, right=likely). Y-axis is severity (bottom=low, top=high). Circle size indicates potential impact. Click any threat for full assessment.',
    targetSelector: '#threat-matrix',
    position: 'below',
    actions: [
      { label: 'Got it', type: 'dismiss' },
      { label: 'Methodology', type: 'learn-more', url: '/help/threat-assessment' },
      { label: 'I know', type: 'never-show' },
    ],
  },
  {
    id: 'viz-cognitive-panel',
    category: 'visualization',
    priority: 'low',
    trigger: 'first-use',
    title: 'Analysis Confidence Metrics',
    shortText: 'XYZA shows how confident the analysis system is.',
    fullText: 'X=Coherence (internal consistency). Y=Complexity (nuance captured). Z=Reflection (depth of analysis). A=Attunement (relevance to context). High scores = more reliable insights.',
    targetSelector: '#cognitive-panel',
    position: 'left',
    minTier: 'analyst',
    actions: [
      { label: 'Interesting', type: 'dismiss' },
      { label: 'Deep dive', type: 'learn-more', url: '/help/cognitive-metrics' },
      { label: 'Skip', type: 'never-show' },
    ],
  },

  // === Analysis ===
  {
    id: 'analysis-compare',
    category: 'analysis',
    priority: 'medium',
    trigger: 'contextual',
    title: 'Side-by-Side Comparison',
    shortText: 'Drag entities here to compare them.',
    fullText: 'Compare countries, organizations, or events side by side. Drag from entity list or right-click and select "Compare". Up to 4 items at once.',
    targetSelector: '#comparison-panel',
    position: 'above',
    minTier: 'analyst',
    actions: [
      { label: 'Try it', type: 'dismiss' },
      { label: 'Comparison guide', type: 'learn-more', url: '/help/comparison' },
      { label: 'I know', type: 'never-show' },
    ],
  },
  {
    id: 'analysis-timeline-zoom',
    category: 'analysis',
    priority: 'low',
    trigger: 'contextual',
    title: 'Timeline Deep Dive',
    shortText: 'Click and drag on the timeline to zoom into a period.',
    fullText: 'Select a time range by clicking and dragging. The view will zoom to show more detail for that period. Double-click to reset.',
    targetSelector: '#timeline',
    position: 'above',
    actions: [
      { label: 'Got it', type: 'dismiss' },
      { label: 'Don\'t show', type: 'never-show' },
    ],
  },

  // === Shortcuts ===
  {
    id: 'shortcut-quick-actions',
    category: 'shortcuts',
    priority: 'medium',
    trigger: 'idle',
    title: 'Quick Actions',
    shortText: 'Press Ctrl+K for quick actions menu.',
    fullText: 'The quick actions menu lets you jump to any feature, run commands, or access settings without navigating menus. Start typing to filter.',
    position: 'floating',
    showAfterHints: ['nav-keyboard-basics'],
    actions: [
      { label: 'Open it', type: 'try-it', shortcut: 'ctrl+k' },
      { label: 'Later', type: 'dismiss' },
      { label: 'I know', type: 'never-show' },
    ],
  },
  {
    id: 'shortcut-focus-mode',
    category: 'shortcuts',
    priority: 'low',
    trigger: 'idle',
    title: 'Focus Mode',
    shortText: 'Press F to toggle focus mode and hide distractions.',
    fullText: 'Focus mode hides navigation, filters, and other chrome to maximize screen space for your current analysis. Press F again to exit.',
    position: 'floating',
    minTier: 'analyst',
    actions: [
      { label: 'Try it', type: 'try-it', shortcut: 'f' },
      { label: 'Maybe later', type: 'dismiss' },
      { label: 'Skip', type: 'never-show' },
    ],
  },

  // === Export ===
  {
    id: 'export-report',
    category: 'export',
    priority: 'low',
    trigger: 'contextual',
    title: 'Export Reports',
    shortText: 'Press E to export the current view as PDF or PowerPoint.',
    fullText: 'Export your current analysis as a formatted report. Choose PDF for reading, PowerPoint for briefings. The export includes all visible data and visualizations.',
    position: 'floating',
    minTier: 'analyst',
    actions: [
      { label: 'Export now', type: 'try-it', shortcut: 'e' },
      { label: 'Later', type: 'dismiss' },
      { label: 'Don\'t remind', type: 'never-show' },
    ],
  },

  // === Advanced ===
  {
    id: 'advanced-api',
    category: 'advanced',
    priority: 'low',
    trigger: 'contextual',
    title: 'API Access Available',
    shortText: 'You can access all this data programmatically.',
    fullText: 'Your tier includes API access. Check the developer docs for endpoints, authentication, and rate limits. Perfect for custom integrations and automated workflows.',
    position: 'floating',
    minTier: 'analyst',
    requiresFeature: 'api-access',
    actions: [
      { label: 'View docs', type: 'learn-more', url: '/developers/api' },
      { label: 'Later', type: 'dismiss' },
      { label: 'I know', type: 'never-show' },
    ],
  },
  {
    id: 'advanced-macros',
    category: 'advanced',
    priority: 'low',
    trigger: 'idle',
    title: 'Automate with Macros',
    shortText: 'Record repetitive actions as reusable macros.',
    fullText: 'Press Ctrl+Shift+R to start recording. Perform your actions, then stop recording. Name your macro and assign a shortcut. Replay anytime.',
    position: 'floating',
    minTier: 'strategist',
    showAfterHints: ['shortcut-quick-actions'],
    actions: [
      { label: 'Start recording', type: 'try-it', shortcut: 'ctrl+shift+r' },
      { label: 'Learn more', type: 'learn-more', url: '/help/macros' },
      { label: 'Not interested', type: 'never-show' },
    ],
  },
];

// ============================================
// Tutorial State Management
// ============================================

const STORAGE_KEY = 'lattice_tutorial_progress';

/**
 * Load tutorial progress from storage
 */
export function loadTutorialProgress(): TutorialProgress {
  if (typeof window === 'undefined') {
    return getDefaultProgress();
  }

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      return { ...getDefaultProgress(), ...parsed };
    }
  } catch {
    // Ignore storage errors
  }

  return getDefaultProgress();
}

/**
 * Save tutorial progress
 */
export function saveTutorialProgress(progress: TutorialProgress): void {
  if (typeof window === 'undefined') return;

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
  } catch {
    // Ignore storage errors
  }
}

/**
 * Get default progress state
 */
function getDefaultProgress(): TutorialProgress {
  return {
    hints: {},
    categoriesCompleted: [],
    totalHintsShown: 0,
    totalHintsDismissed: 0,
    permanentlyDismissedCount: 0,
    tutorialMode: 'full',
    showShortcuts: true,
    lastActivity: Date.now(),
  };
}

/**
 * Get hint state (creating default if needed)
 */
export function getHintState(progress: TutorialProgress, hintId: HintId): HintState {
  return progress.hints[hintId] || {
    shown: false,
    showCount: 0,
    dismissed: false,
    permanentlyDismissed: false,
  };
}

/**
 * Update hint state
 */
export function updateHintState(
  progress: TutorialProgress,
  hintId: HintId,
  update: Partial<HintState>
): TutorialProgress {
  const currentState = getHintState(progress, hintId);
  const newState = { ...currentState, ...update };

  const newProgress = {
    ...progress,
    hints: {
      ...progress.hints,
      [hintId]: newState,
    },
    lastActivity: Date.now(),
  };

  // Update counters
  if (update.shown && !currentState.shown) {
    newProgress.totalHintsShown++;
  }
  if (update.dismissed && !currentState.dismissed) {
    newProgress.totalHintsDismissed++;
  }
  if (update.permanentlyDismissed && !currentState.permanentlyDismissed) {
    newProgress.permanentlyDismissedCount++;
  }

  return newProgress;
}

// ============================================
// Hint Selection Logic
// ============================================

export interface HintContext {
  currentPage: string;
  visibleWidgets: string[];
  userTier: 'explorer' | 'analyst' | 'strategist' | 'architect';
  userFeatures: string[];
  idleTime: number; // seconds
  sessionActions: string[];
}

/**
 * Get the next hint to show based on context
 */
export function getNextHint(
  progress: TutorialProgress,
  context: HintContext
): TutorialHint | null {
  if (progress.tutorialMode === 'off') return null;

  const tierRank: Record<string, number> = {
    explorer: 0,
    analyst: 1,
    strategist: 2,
    architect: 3,
  };
  const userRank = tierRank[context.userTier];

  // Filter eligible hints
  const eligible = TUTORIAL_HINTS.filter(hint => {
    const state = getHintState(progress, hint.id);

    // Already permanently dismissed
    if (state.permanentlyDismissed) return false;

    // Max show count reached
    if (hint.maxShowCount && state.showCount >= hint.maxShowCount) return false;

    // Recently shown (cooldown)
    if (state.lastShown && Date.now() - state.lastShown < 300000) return false; // 5 min cooldown

    // Tier requirement
    if (hint.minTier && tierRank[hint.minTier] > userRank) return false;

    // Feature requirement
    if (hint.requiresFeature && !context.userFeatures.includes(hint.requiresFeature)) return false;

    // Prerequisites
    if (hint.showAfterHints) {
      const prereqsMet = hint.showAfterHints.every(prereq => {
        const prereqState = getHintState(progress, prereq);
        return prereqState.shown;
      });
      if (!prereqsMet) return false;
    }

    // Target visibility
    if (hint.targetSelector && !context.visibleWidgets.includes(hint.targetSelector)) {
      return false;
    }

    // Trigger conditions
    switch (hint.trigger) {
      case 'first-visit':
        return !state.shown;
      case 'first-use':
        // Would need to track widget usage
        return !state.shown && context.visibleWidgets.includes(hint.targetSelector || '');
      case 'idle':
        return context.idleTime > 30 && !state.dismissed;
      case 'contextual':
        return true; // Always eligible if other conditions met
      default:
        return !state.shown;
    }
  });

  if (eligible.length === 0) return null;

  // Prioritize in minimal mode
  if (progress.tutorialMode === 'minimal') {
    const highPriority = eligible.filter(h => h.priority === 'high' || h.priority === 'critical');
    if (highPriority.length > 0) {
      return highPriority[0];
    }
    return null;
  }

  // Sort by priority
  const priorityOrder: Record<HintPriority, number> = {
    critical: 0,
    high: 1,
    medium: 2,
    low: 3,
  };

  eligible.sort((a, b) => {
    // First by priority
    const priorityDiff = priorityOrder[a.priority] - priorityOrder[b.priority];
    if (priorityDiff !== 0) return priorityDiff;

    // Then by show count (prefer unshown)
    const aState = getHintState(progress, a.id);
    const bState = getHintState(progress, b.id);
    return aState.showCount - bState.showCount;
  });

  return eligible[0];
}

/**
 * Check if a category is complete
 */
export function isCategoryComplete(
  progress: TutorialProgress,
  category: HintCategory
): boolean {
  const categoryHints = TUTORIAL_HINTS.filter(h => h.category === category);
  return categoryHints.every(hint => {
    const state = getHintState(progress, hint.id);
    return state.shown || state.permanentlyDismissed;
  });
}

/**
 * Get progress stats for display
 */
export function getTutorialStats(progress: TutorialProgress): {
  totalHints: number;
  hintsShown: number;
  hintsRemaining: number;
  percentComplete: number;
  categoriesComplete: number;
  totalCategories: number;
} {
  const categories = new Set(TUTORIAL_HINTS.map(h => h.category));

  const hintsShown = TUTORIAL_HINTS.filter(hint => {
    const state = getHintState(progress, hint.id);
    return state.shown || state.permanentlyDismissed;
  }).length;

  const categoriesComplete = [...categories].filter(cat =>
    isCategoryComplete(progress, cat)
  ).length;

  return {
    totalHints: TUTORIAL_HINTS.length,
    hintsShown,
    hintsRemaining: TUTORIAL_HINTS.length - hintsShown,
    percentComplete: Math.round((hintsShown / TUTORIAL_HINTS.length) * 100),
    categoriesComplete,
    totalCategories: categories.size,
  };
}

// ============================================
// Batch Operations
// ============================================

/**
 * Permanently dismiss all hints in a category
 */
export function dismissCategory(
  progress: TutorialProgress,
  category: HintCategory
): TutorialProgress {
  let newProgress = { ...progress };

  TUTORIAL_HINTS
    .filter(h => h.category === category)
    .forEach(hint => {
      newProgress = updateHintState(newProgress, hint.id, {
        permanentlyDismissed: true,
      });
    });

  return newProgress;
}

/**
 * Reset tutorial progress (for users who want to re-learn)
 */
export function resetTutorialProgress(): TutorialProgress {
  const fresh = getDefaultProgress();
  saveTutorialProgress(fresh);
  return fresh;
}

/**
 * Set tutorial mode
 */
export function setTutorialMode(
  progress: TutorialProgress,
  mode: 'full' | 'minimal' | 'off'
): TutorialProgress {
  return {
    ...progress,
    tutorialMode: mode,
    lastActivity: Date.now(),
  };
}
