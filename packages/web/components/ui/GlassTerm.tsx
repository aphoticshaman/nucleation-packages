'use client';

import { useState, useRef, useEffect } from 'react';
import { HelpCircle, BookOpen } from 'lucide-react';
import { GLOSSARY, getTooltipForLevel, type GlossaryTerm } from '@/lib/glossary';

interface GlassTermProps {
  /** The term to look up in the glossary (case-insensitive) */
  term: string;
  /** Override the displayed text (defaults to term) */
  children?: React.ReactNode;
  /** User's skill level for appropriate explanation depth */
  level?: 'simple' | 'standard' | 'detailed';
  /** Show a small help icon next to the term */
  showIcon?: boolean;
  /** Custom styling */
  className?: string;
}

/**
 * GlassTerm - Interactive term tooltip component
 *
 * Wraps jargon terms with hover tooltips that explain concepts at the user's level.
 *
 * Usage:
 * <GlassTerm term="basin">Basin Strength</GlassTerm>
 * <GlassTerm term="Transition Risk" level="detailed" />
 * <GlassTerm term="volatility" showIcon />
 */
export function GlassTerm({
  term,
  children,
  level = 'simple',
  showIcon = false,
  className = ''
}: GlassTermProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [position, setPosition] = useState<'top' | 'bottom'>('top');
  const triggerRef = useRef<HTMLSpanElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  // Find the term in glossary
  const glossaryEntry = GLOSSARY.find(
    g => g.term.toLowerCase() === term.toLowerCase()
  );

  // Get appropriate explanation
  const explanation = glossaryEntry?.[level] || glossaryEntry?.simple;

  // Position tooltip above or below based on viewport space
  useEffect(() => {
    if (isOpen && triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      const spaceAbove = rect.top;
      const spaceBelow = window.innerHeight - rect.bottom;
      setPosition(spaceAbove > spaceBelow ? 'top' : 'bottom');
    }
  }, [isOpen]);

  if (!glossaryEntry) {
    // Term not found - render without tooltip
    return <span className={className}>{children || term}</span>;
  }

  return (
    <span
      ref={triggerRef}
      className={`relative inline-flex items-center gap-1 cursor-help ${className}`}
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
      onFocus={() => setIsOpen(true)}
      onBlur={() => setIsOpen(false)}
      tabIndex={0}
      role="button"
      aria-describedby={`tooltip-${term.replace(/\s+/g, '-')}`}
    >
      <span className="border-b border-dashed border-slate-400 hover:border-blue-400 transition-colors">
        {children || glossaryEntry.term}
      </span>

      {showIcon && (
        <HelpCircle className="w-3 h-3 text-slate-400 hover:text-blue-400" />
      )}

      {/* Tooltip */}
      {isOpen && (
        <div
          ref={tooltipRef}
          id={`tooltip-${term.replace(/\s+/g, '-')}`}
          role="tooltip"
          className={`absolute z-50 w-72 p-4 text-sm
            bg-slate-900/95 backdrop-blur-xl border border-white/10 rounded-xl shadow-2xl
            animate-in fade-in-0 zoom-in-95 duration-200
            ${position === 'top' ? 'bottom-full mb-2' : 'top-full mt-2'}
            left-1/2 -translate-x-1/2`}
        >
          {/* Header */}
          <div className="flex items-center gap-2 mb-2 pb-2 border-b border-white/10">
            <BookOpen className="w-4 h-4 text-blue-400" />
            <span className="font-semibold text-blue-400">{glossaryEntry.term}</span>
          </div>

          {/* Explanation */}
          <p className="text-slate-300 leading-relaxed">{explanation}</p>

          {/* Related terms */}
          {glossaryEntry.related && glossaryEntry.related.length > 0 && (
            <div className="mt-3 pt-2 border-t border-white/10">
              <span className="text-xs text-slate-500">Related: </span>
              <span className="text-xs text-slate-400">
                {glossaryEntry.related.slice(0, 3).join(', ')}
              </span>
            </div>
          )}

          {/* Level indicator */}
          <div className="mt-2 flex items-center gap-1">
            {['simple', 'standard', 'detailed'].map((l) => (
              <div
                key={l}
                className={`w-2 h-2 rounded-full ${
                  l === level ? 'bg-blue-500' : 'bg-slate-600'
                }`}
                title={`${l.charAt(0).toUpperCase() + l.slice(1)} explanation`}
              />
            ))}
            <span className="text-xs text-slate-500 ml-1">
              {level === 'simple' ? 'Basic' : level === 'standard' ? 'Intermediate' : 'Advanced'}
            </span>
          </div>

          {/* Arrow */}
          <div
            className={`absolute left-1/2 -translate-x-1/2 w-3 h-3 bg-slate-900/95 border border-white/10 rotate-45
              ${position === 'top' ? '-bottom-1.5 border-t-0 border-l-0' : '-top-1.5 border-b-0 border-r-0'}`}
          />
        </div>
      )}
    </span>
  );
}

/**
 * TermHighlighter - Automatically wraps known terms in a text block
 *
 * Usage:
 * <TermHighlighter text="The basin strength indicates stability" level="simple" />
 */
export function TermHighlighter({
  text,
  level = 'simple'
}: {
  text: string;
  level?: 'simple' | 'standard' | 'detailed';
}) {
  // Build regex from all glossary terms
  const termPatterns = GLOSSARY.map(g => g.term).sort((a, b) => b.length - a.length);
  const regex = new RegExp(`\\b(${termPatterns.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})\\b`, 'gi');

  // Split text and wrap matches
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    // Add text before match
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    // Add wrapped term
    parts.push(
      <GlassTerm key={match.index} term={match[1]} level={level}>
        {match[0]}
      </GlassTerm>
    );
    lastIndex = regex.lastIndex;
  }

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return <>{parts}</>;
}

export default GlassTerm;
