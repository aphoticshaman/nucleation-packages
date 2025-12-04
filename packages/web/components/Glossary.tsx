'use client';

import { useState, useMemo } from 'react';
import { X, Search, BookOpen, ChevronRight } from 'lucide-react';
import { GLOSSARY, CATEGORY_LABELS, searchGlossary, type GlossaryTerm } from '@/lib/glossary';

type SkillLevel = 'simple' | 'standard' | 'detailed';

interface GlossaryProps {
  isOpen: boolean;
  onClose: () => void;
  skillLevel: SkillLevel;
}

export default function Glossary({ isOpen, onClose, skillLevel }: GlossaryProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<GlossaryTerm['category'] | 'all'>('all');
  const [expandedTerm, setExpandedTerm] = useState<string | null>(null);

  const filteredTerms = useMemo(() => {
    let terms = searchQuery ? searchGlossary(searchQuery) : GLOSSARY;
    if (selectedCategory !== 'all') {
      terms = terms.filter(t => t.category === selectedCategory);
    }
    return terms;
  }, [searchQuery, selectedCategory]);

  const getDefinition = (term: GlossaryTerm) => {
    switch (skillLevel) {
      case 'simple':
        return term.simple;
      case 'detailed':
        return term.detailed;
      default:
        return term.standard;
    }
  };

  const getLevelLabel = () => {
    switch (skillLevel) {
      case 'simple':
        return 'Basic definitions';
      case 'detailed':
        return 'Technical definitions';
      default:
        return 'Analyst definitions';
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-2xl max-h-[85vh] bg-[rgba(18,18,26,0.95)] backdrop-blur-xl rounded-2xl border border-white/[0.08] shadow-2xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-white/[0.06]">
          <div className="flex items-center gap-3">
            <BookOpen className="w-6 h-6 text-blue-400" />
            <div>
              <h2 className="text-lg font-bold text-white">Terminology Reference</h2>
              <p className="text-xs text-slate-400">{getLevelLabel()}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-white hover:bg-white/[0.06] rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Search & Filters */}
        <div className="p-4 border-b border-white/[0.06] space-y-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <input
              type="text"
              placeholder="Search terms..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 bg-black/30 border border-white/[0.08] rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-blue-500/50"
            />
          </div>

          <div className="flex gap-2 overflow-x-auto pb-1">
            <button
              onClick={() => setSelectedCategory('all')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-colors ${
                selectedCategory === 'all'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white/[0.06] text-slate-400 hover:text-white'
              }`}
            >
              All Terms
            </button>
            {Object.entries(CATEGORY_LABELS).map(([key, label]) => (
              <button
                key={key}
                onClick={() => setSelectedCategory(key as GlossaryTerm['category'])}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-colors ${
                  selectedCategory === key
                    ? 'bg-blue-600 text-white'
                    : 'bg-white/[0.06] text-slate-400 hover:text-white'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Terms List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {filteredTerms.length === 0 ? (
            <div className="text-center py-8 text-slate-400">
              No terms found for "{searchQuery}"
            </div>
          ) : (
            filteredTerms.map((term) => (
              <div
                key={term.term}
                className="bg-black/20 rounded-xl border border-white/[0.04] overflow-hidden"
              >
                <button
                  onClick={() => setExpandedTerm(expandedTerm === term.term ? null : term.term)}
                  className="w-full p-4 text-left flex items-start justify-between gap-3 hover:bg-white/[0.02] transition-colors"
                >
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-white">{term.term}</span>
                      <span className="text-xs px-2 py-0.5 rounded bg-white/[0.06] text-slate-400">
                        {CATEGORY_LABELS[term.category]}
                      </span>
                    </div>
                    <p className="text-sm text-slate-400 mt-1 leading-relaxed">
                      {getDefinition(term)}
                    </p>
                  </div>
                  <ChevronRight
                    className={`w-4 h-4 text-slate-500 transition-transform ${
                      expandedTerm === term.term ? 'rotate-90' : ''
                    }`}
                  />
                </button>

                {/* Expanded view - shows all three levels */}
                {expandedTerm === term.term && (
                  <div className="px-4 pb-4 space-y-3 border-t border-white/[0.04] pt-3">
                    {skillLevel !== 'simple' && (
                      <div>
                        <span className="text-xs font-medium text-green-400">Basic:</span>
                        <p className="text-xs text-slate-400 mt-0.5">{term.simple}</p>
                      </div>
                    )}
                    {skillLevel !== 'standard' && (
                      <div>
                        <span className="text-xs font-medium text-blue-400">Analyst:</span>
                        <p className="text-xs text-slate-400 mt-0.5">{term.standard}</p>
                      </div>
                    )}
                    {skillLevel !== 'detailed' && (
                      <div>
                        <span className="text-xs font-medium text-purple-400">Expert:</span>
                        <p className="text-xs text-slate-400 mt-0.5">{term.detailed}</p>
                      </div>
                    )}
                    {term.related && term.related.length > 0 && (
                      <div>
                        <span className="text-xs font-medium text-slate-500">Related:</span>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {term.related.map(r => (
                            <button
                              key={r}
                              onClick={() => setSearchQuery(r)}
                              className="text-xs px-2 py-0.5 rounded bg-white/[0.06] text-slate-400 hover:text-white hover:bg-white/[0.1] transition-colors"
                            >
                              {r}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-white/[0.06] text-center">
          <p className="text-xs text-slate-500">
            {skillLevel === 'simple'
              ? 'Switch to Analyst or Expert mode for technical definitions'
              : skillLevel === 'standard'
                ? 'Switch to Expert mode for full tradecraft terminology'
                : 'Full technical glossary with cross-references'}
          </p>
        </div>
      </div>
    </div>
  );
}

// Inline tooltip component for contextual help
export function TermTooltip({
  term,
  skillLevel,
  children,
}: {
  term: string;
  skillLevel: SkillLevel;
  children: React.ReactNode;
}) {
  const [showTooltip, setShowTooltip] = useState(false);
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
      className="relative cursor-help border-b border-dotted border-slate-500"
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      {children}
      {showTooltip && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-slate-800 border border-white/[0.1] rounded-lg shadow-xl z-50 w-64">
          <p className="text-xs text-white font-medium mb-1">{glossaryTerm.term}</p>
          <p className="text-xs text-slate-400">{definition}</p>
        </div>
      )}
    </span>
  );
}
