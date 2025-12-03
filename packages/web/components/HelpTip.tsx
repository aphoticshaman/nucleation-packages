'use client';

import { useState, useRef, useEffect } from 'react';
import { GLOSSARY } from '@/lib/glossary';

type SkillLevel = 'simple' | 'standard' | 'detailed';

interface HelpTipProps {
  /** The glossary term to explain */
  term: string;
  /** Current skill level for appropriate definition */
  skillLevel: SkillLevel;
  /** Optional custom content instead of glossary lookup */
  content?: string;
  /** Size of the icon (default: 12) */
  size?: number;
  /** Position of the popup */
  position?: 'top' | 'bottom' | 'left' | 'right';
}

/**
 * Little "?" circle that shows contextual help on click/hover.
 * 10-12px diameter, pops up a toast-like explanation box.
 */
export default function HelpTip({
  term,
  skillLevel,
  content,
  size = 12,
  position = 'top',
}: HelpTipProps) {
  const [isOpen, setIsOpen] = useState(false);
  const tipRef = useRef<HTMLDivElement>(null);

  // Get definition from glossary or use custom content
  const glossaryTerm = GLOSSARY.find(t => t.term.toLowerCase() === term.toLowerCase());
  const definition = content || (glossaryTerm
    ? skillLevel === 'simple'
      ? glossaryTerm.simple
      : skillLevel === 'detailed'
        ? glossaryTerm.detailed
        : glossaryTerm.standard
    : `No definition found for "${term}"`);

  // Close on click outside
  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (e: MouseEvent) => {
      if (tipRef.current && !tipRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  // Position styles for the popup
  const positionStyles = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  };

  return (
    <span ref={tipRef} className="relative inline-flex items-center">
      <button
        onClick={() => setIsOpen(!isOpen)}
        onMouseEnter={() => setIsOpen(true)}
        onMouseLeave={() => setIsOpen(false)}
        className="inline-flex items-center justify-center rounded-full bg-slate-700/60 hover:bg-blue-600/60 text-slate-400 hover:text-white transition-colors cursor-help"
        style={{ width: size, height: size, fontSize: size * 0.65 }}
        aria-label={`Help: ${term}`}
      >
        ?
      </button>

      {isOpen && (
        <div
          className={`absolute z-50 ${positionStyles[position]} w-56 sm:w-64`}
          role="tooltip"
        >
          <div className="bg-[rgba(18,18,26,0.98)] backdrop-blur-xl border border-white/[0.12] rounded-lg shadow-xl p-3">
            {glossaryTerm && (
              <p className="text-xs font-medium text-blue-400 mb-1">{glossaryTerm.term}</p>
            )}
            <p className="text-xs text-slate-300 leading-relaxed">{definition}</p>
            {glossaryTerm?.related && glossaryTerm.related.length > 0 && skillLevel !== 'simple' && (
              <p className="text-[10px] text-slate-500 mt-2">
                See also: {glossaryTerm.related.slice(0, 3).join(', ')}
              </p>
            )}
          </div>
          {/* Arrow */}
          <div
            className={`absolute w-2 h-2 bg-[rgba(18,18,26,0.98)] border-white/[0.12] transform rotate-45 ${
              position === 'top'
                ? 'top-full left-1/2 -translate-x-1/2 -mt-1 border-r border-b'
                : position === 'bottom'
                  ? 'bottom-full left-1/2 -translate-x-1/2 -mb-1 border-l border-t'
                  : position === 'left'
                    ? 'left-full top-1/2 -translate-y-1/2 -ml-1 border-t border-r'
                    : 'right-full top-1/2 -translate-y-1/2 -mr-1 border-b border-l'
            }`}
          />
        </div>
      )}
    </span>
  );
}

/**
 * Inline help tip that wraps text and adds a small indicator.
 * Shows definition on hover/click.
 */
export function InlineHelp({
  term,
  skillLevel,
  children,
}: {
  term: string;
  skillLevel: SkillLevel;
  children: React.ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const glossaryTerm = GLOSSARY.find(t => t.term.toLowerCase() === term.toLowerCase());

  if (!glossaryTerm) return <>{children}</>;

  const definition =
    skillLevel === 'simple'
      ? glossaryTerm.simple
      : skillLevel === 'detailed'
        ? glossaryTerm.detailed
        : glossaryTerm.standard;

  return (
    <span
      className="relative inline-block cursor-help"
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
      onClick={() => setIsOpen(!isOpen)}
    >
      <span className="border-b border-dotted border-blue-400/50 hover:border-blue-400">
        {children}
      </span>
      {isOpen && (
        <span className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 sm:w-64">
          <span className="block bg-[rgba(18,18,26,0.98)] backdrop-blur-xl border border-white/[0.12] rounded-lg shadow-xl p-3">
            <span className="block text-xs font-medium text-blue-400 mb-1">{glossaryTerm.term}</span>
            <span className="block text-xs text-slate-300 leading-relaxed">{definition}</span>
          </span>
        </span>
      )}
    </span>
  );
}
