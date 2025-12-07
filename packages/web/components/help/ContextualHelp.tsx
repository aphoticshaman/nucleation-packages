'use client';

/**
 * Contextual Help System
 *
 * Tiny (?) circles that appear next to technical/complex elements.
 * Click or hover to get inline explanation.
 *
 * Respects expertise level:
 * - Beginner: Show all help, verbose explanations
 * - Intermediate: Show for technical stuff, concise explanations
 * - Advanced: Minimal help, expert-level terminology ok
 */

import { useState, useRef, useEffect, createContext, useContext, ReactNode } from 'react';
import { HelpCircle, ExternalLink, ChevronRight } from 'lucide-react';
import { getLexiconEntry, getArticle, LexiconEntry, KnowledgeArticle } from '@/lib/knowledge/KnowledgeBase';

// ============================================
// Types
// ============================================

export type ExpertiseLevel = 'beginner' | 'intermediate' | 'advanced' | 'expert';

export interface HelpContent {
  short: string; // One-liner, shown on hover
  full?: string; // Expanded explanation
  lexiconTerm?: string; // Link to lexicon
  articleSlug?: string; // Link to full article
  expertiseThreshold?: ExpertiseLevel; // Only show to users at or below this level
}

interface ExpertiseContextType {
  level: ExpertiseLevel;
  setLevel: (level: ExpertiseLevel) => void;
  showHelp: boolean;
  setShowHelp: (show: boolean) => void;
}

// ============================================
// Context
// ============================================

const ExpertiseContext = createContext<ExpertiseContextType>({
  level: 'intermediate',
  setLevel: () => {},
  showHelp: true,
  setShowHelp: () => {},
});

export function useExpertise() {
  return useContext(ExpertiseContext);
}

export function ExpertiseProvider({
  children,
  initialLevel = 'intermediate',
}: {
  children: ReactNode;
  initialLevel?: ExpertiseLevel;
}) {
  const [level, setLevel] = useState<ExpertiseLevel>(initialLevel);
  const [showHelp, setShowHelp] = useState(true);

  // Persist to localStorage
  useEffect(() => {
    const stored = localStorage.getItem('lattice_expertise_level');
    if (stored) {
      setLevel(stored as ExpertiseLevel);
    }
    const helpPref = localStorage.getItem('lattice_show_help');
    if (helpPref !== null) {
      setShowHelp(helpPref === 'true');
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('lattice_expertise_level', level);
  }, [level]);

  useEffect(() => {
    localStorage.setItem('lattice_show_help', String(showHelp));
  }, [showHelp]);

  return (
    <ExpertiseContext.Provider value={{ level, setLevel, showHelp, setShowHelp }}>
      {children}
    </ExpertiseContext.Provider>
  );
}

// ============================================
// Expertise Level Descriptions
// ============================================

export const EXPERTISE_LEVELS: Record<
  ExpertiseLevel,
  {
    label: string;
    description: string;
    examples: string[];
  }
> = {
  beginner: {
    label: 'Learning',
    description: 'New to intelligence analysis. Show all explanations and guidance.',
    examples: [
      'Students and researchers',
      'Journalists starting on international beat',
      'Business professionals new to risk analysis',
      'E-1 to E-4 personnel in training',
    ],
  },
  intermediate: {
    label: 'Working Analyst',
    description: 'Familiar with basics. Show help for technical and advanced features.',
    examples: [
      'Junior to mid-level analysts',
      'Experienced journalists',
      'Risk managers with some background',
      'E-5 to E-7 with relevant MOS',
    ],
  },
  advanced: {
    label: 'Senior Analyst',
    description: 'Deep expertise. Minimal hand-holding, show help only for complex features.',
    examples: [
      'Senior analysts and supervisors',
      'IC professionals',
      'Experienced field officers',
      'O-3 to O-5 with relevant background',
    ],
  },
  expert: {
    label: 'Director / Station Chief',
    description: 'Maximum information density. No basic explanations.',
    examples: [
      'Agency directors and division chiefs',
      'Senior policy advisors',
      'Those who wrote the methodology',
      'O-6+ or GS-15+',
    ],
  },
};

// ============================================
// Help Indicator Component (The little ?)
// ============================================

interface HelpIndicatorProps {
  content: HelpContent;
  size?: 'xs' | 'sm' | 'md';
  inline?: boolean; // Inline with text vs absolute positioned
}

export function HelpIndicator({ content, size = 'xs', inline = true }: HelpIndicatorProps) {
  const { level, showHelp } = useExpertise();
  const [isOpen, setIsOpen] = useState(false);
  const [position, setPosition] = useState<'above' | 'below'>('below');
  const triggerRef = useRef<HTMLButtonElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);

  // Check if help should be shown based on expertise
  const shouldShow = showHelp && shouldShowHelp(content.expertiseThreshold, level);

  // Position popup based on viewport
  useEffect(() => {
    if (isOpen && triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      const spaceBelow = window.innerHeight - rect.bottom;
      setPosition(spaceBelow < 200 ? 'above' : 'below');
    }
  }, [isOpen]);

  // Close on outside click
  useEffect(() => {
    if (!isOpen) return;

    const handleClick = (e: MouseEvent) => {
      if (
        popupRef.current &&
        !popupRef.current.contains(e.target as Node) &&
        triggerRef.current &&
        !triggerRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [isOpen]);

  if (!shouldShow) return null;

  const sizeClasses = {
    xs: 'w-3 h-3',
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
  };

  return (
    <span className={`relative ${inline ? 'inline-flex align-middle ml-1' : ''}`}>
      <button
        ref={triggerRef}
        onClick={() => setIsOpen(!isOpen)}
        onMouseEnter={() => !isOpen && setIsOpen(true)}
        className={`
          ${sizeClasses[size]}
          text-gray-500 hover:text-blue-400
          transition-colors cursor-help
          opacity-60 hover:opacity-100
        `}
        aria-label="Help"
        aria-expanded={isOpen}
      >
        <HelpCircle className="w-full h-full" />
      </button>

      {isOpen && (
        <HelpPopup
          ref={popupRef}
          content={content}
          position={position}
          onClose={() => setIsOpen(false)}
        />
      )}
    </span>
  );
}

// ============================================
// Help Popup (Expanded help content)
// ============================================

interface HelpPopupProps {
  content: HelpContent;
  position: 'above' | 'below';
  onClose: () => void;
}

const HelpPopup = ({
  content,
  position,
  onClose: _onClose,
  ref,
}: HelpPopupProps & { ref: React.RefObject<HTMLDivElement | null> }) => {
  const { level: _level } = useExpertise();
  const [showFull, setShowFull] = useState(false);
  const [lexiconEntry, setLexiconEntry] = useState<LexiconEntry | undefined>();
  const [article, setArticle] = useState<KnowledgeArticle | undefined>();

  // Load related content
  useEffect(() => {
    if (content.lexiconTerm) {
      setLexiconEntry(getLexiconEntry(content.lexiconTerm));
    }
    if (content.articleSlug) {
      setArticle(getArticle(content.articleSlug));
    }
  }, [content]);

  const positionClasses = position === 'above' ? 'bottom-full mb-2' : 'top-full mt-2';

  return (
    <div
      ref={ref}
      className={`
        absolute ${positionClasses} left-0 z-50
        w-64 max-w-[90vw]
        bg-gray-900 border border-gray-700 rounded-lg shadow-xl
        animate-in fade-in zoom-in-95 duration-150
      `}
      role="tooltip"
    >
      {/* Short explanation */}
      <div className="p-3">
        <p className="text-sm text-gray-300">{content.short}</p>

        {/* Full explanation (if available and requested) */}
        {content.full && !showFull && (
          <button
            onClick={() => setShowFull(true)}
            className="mt-2 text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
          >
            More detail <ChevronRight className="w-3 h-3" />
          </button>
        )}

        {content.full && showFull && (
          <div className="mt-3 pt-3 border-t border-gray-700">
            <p className="text-sm text-gray-400">{content.full}</p>
          </div>
        )}
      </div>

      {/* Lexicon reference */}
      {lexiconEntry && (
        <div className="px-3 pb-3">
          <div className="p-2 bg-gray-800/50 rounded border border-gray-700/50">
            <div className="text-xs text-gray-500 mb-1">Lexicon</div>
            <div className="text-sm font-medium text-white">{lexiconEntry.term}</div>
            <p className="text-xs text-gray-400 mt-1 line-clamp-2">{lexiconEntry.definition}</p>
          </div>
        </div>
      )}

      {/* Article link */}
      {article && (
        <div className="px-3 pb-3 pt-0">
          <a
            href={`/help/${article.slug}`}
            className="flex items-center gap-2 text-xs text-blue-400 hover:text-blue-300"
          >
            <ExternalLink className="w-3 h-3" />
            Read full article: {article.title}
          </a>
        </div>
      )}

      {/* Close hint for mouse users */}
      <div className="px-3 py-2 bg-gray-800/30 border-t border-gray-700/50 rounded-b-lg">
        <span className="text-xs text-gray-500">Click anywhere to close</span>
      </div>
    </div>
  );
};

// ============================================
// Expertise Level Selector
// ============================================

export function ExpertiseLevelSelector() {
  const { level, setLevel, showHelp, setShowHelp } = useExpertise();

  return (
    <div className="space-y-4">
      {/* Level selection */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Interface Complexity
        </label>
        <div className="grid grid-cols-2 gap-2">
          {(Object.keys(EXPERTISE_LEVELS) as ExpertiseLevel[]).map((key) => {
            const info = EXPERTISE_LEVELS[key];
            return (
              <button
                key={key}
                onClick={() => setLevel(key)}
                className={`
                  p-3 rounded-lg border text-left transition-all
                  ${
                    level === key
                      ? 'border-blue-500 bg-blue-500/10'
                      : 'border-gray-700 hover:border-gray-600'
                  }
                `}
              >
                <div className="font-medium text-white text-sm">{info.label}</div>
                <div className="text-xs text-gray-400 mt-1 line-clamp-2">{info.description}</div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Help toggle */}
      <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
        <div>
          <div className="text-sm font-medium text-white">Show help indicators</div>
          <div className="text-xs text-gray-400">
            Tiny (?) icons next to complex elements
          </div>
        </div>
        <button
          onClick={() => setShowHelp(!showHelp)}
          className={`
            relative w-12 h-6 rounded-full transition-colors
            ${showHelp ? 'bg-blue-600' : 'bg-gray-600'}
          `}
        >
          <span
            className={`
              absolute top-1 w-4 h-4 bg-white rounded-full transition-transform
              ${showHelp ? 'left-7' : 'left-1'}
            `}
          />
        </button>
      </div>

      {/* Current level description */}
      <div className="p-3 bg-gray-800/30 rounded-lg">
        <div className="text-xs text-gray-500 mb-2">Your level: {EXPERTISE_LEVELS[level].label}</div>
        <div className="text-xs text-gray-400">Typical users at this level:</div>
        <ul className="mt-1 space-y-1">
          {EXPERTISE_LEVELS[level].examples.map((example, i) => (
            <li key={i} className="text-xs text-gray-500 flex items-start gap-2">
              <span className="text-gray-600">-</span>
              {example}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

// ============================================
// Helpers
// ============================================

function shouldShowHelp(threshold: ExpertiseLevel | undefined, current: ExpertiseLevel): boolean {
  if (!threshold) return true; // Always show if no threshold

  const levels: ExpertiseLevel[] = ['beginner', 'intermediate', 'advanced', 'expert'];
  const thresholdIndex = levels.indexOf(threshold);
  const currentIndex = levels.indexOf(current);

  // Show help if user's level is at or below the threshold
  return currentIndex <= thresholdIndex;
}

// ============================================
// Pre-built Help Contents
// ============================================

export const HELP_CONTENTS: Record<string, HelpContent> = {
  // XYZA Metrics
  'xyza-coherence': {
    short: 'How internally consistent is the analysis.',
    full: 'Coherence (X) measures whether different parts of the analysis support each other. High coherence means the conclusions follow logically from the evidence.',
    lexiconTerm: 'XYZA Metrics',
    expertiseThreshold: 'intermediate',
  },
  'xyza-complexity': {
    short: 'How much nuance is captured.',
    full: 'Complexity (Y) measures analytical depth. Low complexity might miss important factors. Excessively high might indicate over-complication.',
    lexiconTerm: 'XYZA Metrics',
    expertiseThreshold: 'intermediate',
  },
  'xyza-reflection': {
    short: 'Depth of analytical reasoning.',
    full: 'Reflection (Z) indicates how thoroughly the analysis considered alternatives, biases, and limitations.',
    lexiconTerm: 'XYZA Metrics',
    expertiseThreshold: 'intermediate',
  },
  'xyza-attunement': {
    short: 'How relevant to your current context.',
    full: 'Attunement (A) measures how well the analysis matches what you need. High attunement = highly relevant.',
    lexiconTerm: 'XYZA Metrics',
    expertiseThreshold: 'intermediate',
  },

  // Confidence levels
  'confidence-high': {
    short: 'Strong evidence, multiple sources corroborate.',
    full: 'HIGH confidence means we have strong analytical basis. Not certainty, but reasonable to act on.',
    articleSlug: 'understanding-confidence',
    expertiseThreshold: 'beginner',
  },
  'confidence-moderate': {
    short: 'Reasonable basis but alternatives exist.',
    full: 'MODERATE confidence means good working hypothesis. Watch for confirming/disconfirming evidence.',
    articleSlug: 'understanding-confidence',
    expertiseThreshold: 'beginner',
  },
  'confidence-low': {
    short: 'Limited information, could change with new data.',
    full: 'LOW confidence is valuable as early warning. Don\'t dismiss, but don\'t plan around it either.',
    articleSlug: 'understanding-confidence',
    expertiseThreshold: 'beginner',
  },

  // Threat matrix
  'threat-probability': {
    short: 'Likelihood of this threat occurring.',
    full: 'Based on indicators, historical patterns, and current conditions. Right = more likely.',
    expertiseThreshold: 'beginner',
  },
  'threat-severity': {
    short: 'Impact if this threat occurs.',
    full: 'Considers affected population, economic damage, political consequences. Top = more severe.',
    expertiseThreshold: 'beginner',
  },

  // Filtering
  'filter-temporal': {
    short: 'Filter by time period.',
    full: 'Shows intel from selected time range. Enable projections to see future estimates.',
    lexiconTerm: 'Temporal Analysis',
    expertiseThreshold: 'beginner',
  },
  'filter-entity-type': {
    short: 'Filter by what kind of actor.',
    full: 'Countries, organizations, people, military units, facilities, etc.',
    lexiconTerm: 'Entity',
    expertiseThreshold: 'beginner',
  },

  // Flow state
  'flow-state': {
    short: 'System performance indicator.',
    full: 'Based on Kuramoto synchronization model. High flow = optimal operation. Low flow = potential degradation.',
    lexiconTerm: 'Flow State',
    expertiseThreshold: 'advanced',
  },

  // Relations
  'relation-strength': {
    short: 'How strong is this relationship.',
    full: 'Based on frequency of interaction, treaty obligations, trade volume, historical depth.',
    lexiconTerm: 'Relation',
    expertiseThreshold: 'intermediate',
  },
};

/**
 * Quick helper to create help content inline
 */
export function help(id: keyof typeof HELP_CONTENTS): HelpContent {
  return HELP_CONTENTS[id];
}
