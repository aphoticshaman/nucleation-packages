'use client';

import { useState, useRef, useEffect, useMemo } from 'react';

interface JargonTerm {
  term: string;
  definition: string;
  category?: 'military' | 'intelligence' | 'financial' | 'cyber' | 'technical';
  aliases?: string[];
  context?: string; // Disambiguates based on usage
}

// Built-in intelligence/military jargon dictionary
const JARGON_DICTIONARY: JargonTerm[] = [
  // Military
  { term: 'SIGINT', definition: 'Signals Intelligence - intelligence gathered from intercepting signals', category: 'intelligence' },
  { term: 'HUMINT', definition: 'Human Intelligence - intelligence gathered from human sources', category: 'intelligence' },
  { term: 'OSINT', definition: 'Open Source Intelligence - intelligence from publicly available sources', category: 'intelligence' },
  { term: 'GEOINT', definition: 'Geospatial Intelligence - intelligence from satellite and aerial imagery', category: 'intelligence' },
  { term: 'MASINT', definition: 'Measurement and Signature Intelligence - technical intelligence', category: 'intelligence' },
  { term: 'ISR', definition: 'Intelligence, Surveillance, and Reconnaissance', category: 'military' },
  { term: 'C2', definition: 'Command and Control - authority and direction over forces', category: 'military' },
  { term: 'C4ISR', definition: 'Command, Control, Communications, Computers, Intelligence, Surveillance, and Reconnaissance', category: 'military' },
  { term: 'OPSEC', definition: 'Operations Security - protecting critical information', category: 'military' },
  { term: 'ROE', definition: 'Rules of Engagement - directives for when force can be used', category: 'military' },
  { term: 'DEFCON', definition: 'Defense Readiness Condition - alert state for US military', category: 'military' },
  { term: 'SITREP', definition: 'Situation Report - status update on current conditions', category: 'military' },
  { term: 'CONOP', definition: 'Concept of Operations - how an operation will be executed', category: 'military' },
  { term: 'AO', definition: 'Area of Operations - geographic region for military activity', category: 'military' },
  { term: 'ASAT', definition: 'Anti-Satellite Weapon - weapon designed to destroy satellites', category: 'military' },

  // Financial
  { term: 'AML', definition: 'Anti-Money Laundering - regulations to prevent illicit funds', category: 'financial' },
  { term: 'KYC', definition: 'Know Your Customer - identity verification requirements', category: 'financial' },
  { term: 'SAR', definition: 'Suspicious Activity Report - report filed for suspicious transactions', category: 'financial' },
  { term: 'CTR', definition: 'Currency Transaction Report - report for cash transactions over $10,000', category: 'financial' },
  { term: 'SWIFT', definition: 'Society for Worldwide Interbank Financial Telecommunication', category: 'financial' },
  { term: 'BPS', definition: 'Basis Points - 1/100th of a percentage point', category: 'financial', aliases: ['bps', 'bp'] },
  { term: 'VaR', definition: 'Value at Risk - statistical measure of potential loss', category: 'financial' },
  { term: 'FOMC', definition: 'Federal Open Market Committee - sets US monetary policy', category: 'financial' },

  // Cyber
  { term: 'APT', definition: 'Advanced Persistent Threat - sophisticated, long-term cyber attack', category: 'cyber' },
  { term: 'IOC', definition: 'Indicator of Compromise - forensic artifact of intrusion', category: 'cyber' },
  { term: 'TTP', definition: 'Tactics, Techniques, and Procedures - adversary behaviors', category: 'cyber' },
  { term: 'C2', definition: 'Command and Control - infrastructure for controlling malware', category: 'cyber', context: 'cyber' },
  { term: 'RAT', definition: 'Remote Access Trojan - malware for unauthorized remote control', category: 'cyber' },
  { term: 'CVE', definition: 'Common Vulnerabilities and Exposures - standardized vulnerability ID', category: 'cyber' },
  { term: 'CVSS', definition: 'Common Vulnerability Scoring System - severity rating 0-10', category: 'cyber' },
  { term: 'SOC', definition: 'Security Operations Center - facility for monitoring cyber threats', category: 'cyber' },
  { term: 'SIEM', definition: 'Security Information and Event Management - log analysis platform', category: 'cyber' },
  { term: 'EDR', definition: 'Endpoint Detection and Response - endpoint security tool', category: 'cyber' },
  { term: 'XDR', definition: 'Extended Detection and Response - cross-layer security platform', category: 'cyber' },
  { term: 'MITRE ATT&CK', definition: 'Knowledge base of adversary tactics and techniques', category: 'cyber' },

  // Technical
  { term: 'ML', definition: 'Machine Learning - AI systems that learn from data', category: 'technical' },
  { term: 'NLP', definition: 'Natural Language Processing - AI for understanding text', category: 'technical' },
  { term: 'NER', definition: 'Named Entity Recognition - extracting entities from text', category: 'technical' },
  { term: 'LLM', definition: 'Large Language Model - AI model trained on vast text data', category: 'technical' },
  { term: 'RAG', definition: 'Retrieval Augmented Generation - combining LLMs with document retrieval', category: 'technical' },
  { term: 'SHAP', definition: 'SHapley Additive exPlanations - ML explainability method', category: 'technical' },
  { term: 'LIME', definition: 'Local Interpretable Model-agnostic Explanations - ML explainability', category: 'technical' },
];

// Component 08: Jargon Decoder Tooltip
interface JargonDecoderProps {
  children: React.ReactNode;
  customDictionary?: JargonTerm[];
  contextHint?: string; // Helps disambiguate terms
}

export function JargonDecoder({
  children,
  customDictionary = [],
  contextHint,
}: JargonDecoderProps) {
  const dictionary = useMemo(() => [...JARGON_DICTIONARY, ...customDictionary], [customDictionary]);

  // Build regex pattern for all terms
  const pattern = useMemo(() => {
    const terms = dictionary.flatMap(t => [t.term, ...(t.aliases || [])]);
    const escaped = terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    return new RegExp(`\\b(${escaped.join('|')})\\b`, 'gi');
  }, [dictionary]);

  // If children is a string, process it
  if (typeof children === 'string') {
    return <JargonText text={children} dictionary={dictionary} pattern={pattern} contextHint={contextHint} />;
  }

  return <>{children}</>;
}

function JargonText({
  text,
  dictionary,
  pattern,
  contextHint,
}: {
  text: string;
  dictionary: JargonTerm[];
  pattern: RegExp;
  contextHint?: string;
}) {
  const parts: (string | { term: string; match: string })[] = [];
  let lastIndex = 0;
  let match;

  pattern.lastIndex = 0;
  while ((match = pattern.exec(text)) !== null) {
    // Add text before match
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    // Add match
    parts.push({ term: match[0].toUpperCase(), match: match[0] });
    lastIndex = pattern.lastIndex;
  }
  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return (
    <>
      {parts.map((part, i) =>
        typeof part === 'string' ? (
          <span key={i}>{part}</span>
        ) : (
          <JargonTooltip
            key={i}
            term={part.term}
            displayText={part.match}
            dictionary={dictionary}
            contextHint={contextHint}
          />
        )
      )}
    </>
  );
}

function JargonTooltip({
  term,
  displayText,
  dictionary,
  contextHint,
}: {
  term: string;
  displayText: string;
  dictionary: JargonTerm[];
  contextHint?: string;
}) {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const ref = useRef<HTMLSpanElement>(null);

  // Find definition (with context disambiguation)
  const entry = useMemo(() => {
    const matches = dictionary.filter(
      t => t.term.toUpperCase() === term || t.aliases?.some(a => a.toUpperCase() === term)
    );
    if (matches.length === 1) return matches[0];
    if (contextHint) {
      const contextMatch = matches.find(m => m.context === contextHint || m.category === contextHint);
      if (contextMatch) return contextMatch;
    }
    return matches[0];
  }, [term, dictionary, contextHint]);

  const handleMouseEnter = () => {
    if (!ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    setPosition({
      top: rect.bottom + window.scrollY + 8,
      left: rect.left + window.scrollX + rect.width / 2,
    });
    setIsVisible(true);
  };

  if (!entry) return <span>{displayText}</span>;

  const categoryColors: Record<string, string> = {
    military: 'border-amber-500/50 text-amber-400',
    intelligence: 'border-purple-500/50 text-purple-400',
    financial: 'border-green-500/50 text-green-400',
    cyber: 'border-red-500/50 text-red-400',
    technical: 'border-cyan-500/50 text-cyan-400',
  };

  return (
    <>
      <span
        ref={ref}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={() => setIsVisible(false)}
        className="border-b border-dashed border-slate-500 cursor-help hover:border-cyan-400 transition-colors"
      >
        {displayText}
      </span>

      {isVisible && (
        <div
          className="fixed z-50 pointer-events-none"
          style={{
            top: position.top,
            left: position.left,
            transform: 'translateX(-50%)',
          }}
        >
          <div className="bg-slate-800 border border-slate-600 rounded-lg shadow-xl p-3 max-w-xs">
            {/* Arrow */}
            <div className="absolute -top-2 left-1/2 -translate-x-1/2 w-0 h-0 border-l-8 border-r-8 border-b-8 border-transparent border-b-slate-600" />

            {/* Content */}
            <div className="flex items-start gap-2">
              <span className="text-lg font-bold text-cyan-400 font-mono">{entry.term}</span>
              {entry.category && (
                <span className={`text-xs px-1.5 py-0.5 rounded border ${categoryColors[entry.category]}`}>
                  {entry.category}
                </span>
              )}
            </div>
            <p className="text-sm text-slate-300 mt-1">{entry.definition}</p>
          </div>
        </div>
      )}
    </>
  );
}

// Standalone tooltip for manual use
export function TermTooltip({
  term,
  children,
}: {
  term: string;
  children: React.ReactNode;
}) {
  const entry = JARGON_DICTIONARY.find(t => t.term === term);
  if (!entry) return <>{children}</>;

  return (
    <JargonTooltip
      term={term}
      displayText={term}
      dictionary={JARGON_DICTIONARY}
    />
  );
}

// Get definition programmatically
export function getJargonDefinition(term: string): JargonTerm | undefined {
  return JARGON_DICTIONARY.find(
    t => t.term.toUpperCase() === term.toUpperCase() ||
         t.aliases?.some(a => a.toUpperCase() === term.toUpperCase())
  );
}

// Export dictionary for external use
export { JARGON_DICTIONARY };
