'use client';

import { useState, useMemo } from 'react';

interface Citation {
  id: string;
  sourceId: string;
  sourceName: string;
  sourceType: 'news' | 'gov' | 'academic' | 'intel' | 'social' | 'wire';
  url?: string;
  title: string;
  author?: string;
  publishedAt: string;
  accessedAt: string;
  reliability: 'A' | 'B' | 'C' | 'D' | 'E' | 'F'; // NATO Admiralty Code
  credibility: 1 | 2 | 3 | 4 | 5 | 6; // 1=confirmed, 6=cannot be judged
  excerpt?: string;
  tags?: string[];
}

interface CitationManagerProps {
  citations: Citation[];
  onCitationAdd?: (citation: Omit<Citation, 'id' | 'accessedAt'>) => void;
  onCitationRemove?: (id: string) => void;
  onExport?: (format: 'bibtex' | 'apa' | 'json') => void;
  maxDisplay?: number;
}

// Component 35: Citation Manager with NATO Admiralty Codes
export function CitationManager({
  citations,
  onCitationAdd,
  onCitationRemove,
  onExport,
  maxDisplay = 10,
}: CitationManagerProps) {
  const [expanded, setExpanded] = useState(false);
  const [filter, setFilter] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'date' | 'reliability'>('date');

  const reliabilityLabels: Record<Citation['reliability'], string> = {
    A: 'Completely reliable',
    B: 'Usually reliable',
    C: 'Fairly reliable',
    D: 'Not usually reliable',
    E: 'Unreliable',
    F: 'Cannot be judged',
  };

  const credibilityLabels: Record<Citation['credibility'], string> = {
    1: 'Confirmed',
    2: 'Probably true',
    3: 'Possibly true',
    4: 'Doubtful',
    5: 'Improbable',
    6: 'Cannot be judged',
  };

  const sourceTypeIcons: Record<Citation['sourceType'], string> = {
    news: '📰',
    gov: '🏛️',
    academic: '📚',
    intel: '🔐',
    social: '💬',
    wire: '⚡',
  };

  // Filter and sort citations
  const filteredCitations = useMemo(() => {
    let result = citations;

    if (filter !== 'all') {
      result = result.filter((c) => c.sourceType === filter);
    }

    result = [...result].sort((a, b) => {
      if (sortBy === 'date') {
        return new Date(b.publishedAt).getTime() - new Date(a.publishedAt).getTime();
      } else {
        // Sort by reliability (A > B > C...) then credibility (1 > 2 > 3...)
        const relDiff = a.reliability.charCodeAt(0) - b.reliability.charCodeAt(0);
        if (relDiff !== 0) return relDiff;
        return a.credibility - b.credibility;
      }
    });

    return result;
  }, [citations, filter, sortBy]);

  const displayCitations = expanded ? filteredCitations : filteredCitations.slice(0, maxDisplay);

  // Aggregate reliability stats
  const stats = useMemo(() => {
    const byReliability: Record<string, number> = {};
    const byType: Record<string, number> = {};

    citations.forEach((c) => {
      byReliability[c.reliability] = (byReliability[c.reliability] || 0) + 1;
      byType[c.sourceType] = (byType[c.sourceType] || 0) + 1;
    });

    const avgReliability = citations.length > 0
      ? citations.reduce((sum, c) => sum + c.reliability.charCodeAt(0) - 64, 0) / citations.length
      : 0;

    return { byReliability, byType, avgReliability };
  }, [citations]);

  return (
    <div className="bg-slate-900/50 rounded-lg border border-slate-700">
      {/* Header */}
      <div className="p-4 border-b border-slate-700">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <h3 className="text-sm font-medium text-slate-200">Sources</h3>
            <span className="text-xs text-slate-500">{citations.length} citations</span>

            {/* Aggregate reliability badge */}
            <span
              className={`px-2 py-0.5 rounded text-xs font-mono ${
                stats.avgReliability <= 2 ? 'bg-green-500/20 text-green-400' :
                stats.avgReliability <= 3 ? 'bg-yellow-500/20 text-yellow-400' :
                'bg-red-500/20 text-red-400'
              }`}
            >
              Avg: {String.fromCharCode(64 + Math.round(stats.avgReliability))}
            </span>
          </div>

          {/* Export dropdown */}
          {onExport && (
            <div className="relative group">
              <button className="px-2 py-1 text-xs text-slate-400 hover:text-slate-200 transition-colors">
                Export ↓
              </button>
              <div className="absolute right-0 top-full mt-1 hidden group-hover:block bg-slate-800 border border-slate-700 rounded shadow-lg z-10">
                {(['bibtex', 'apa', 'json'] as const).map((fmt) => (
                  <button
                    key={fmt}
                    onClick={() => onExport(fmt)}
                    className="block w-full px-4 py-2 text-left text-xs text-slate-300 hover:bg-slate-700"
                  >
                    {fmt.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Filters */}
        <div className="flex gap-2 text-xs">
          <button
            onClick={() => setFilter('all')}
            className={`px-2 py-1 rounded transition-colors ${
              filter === 'all' ? 'bg-cyan-500/20 text-cyan-400' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            All
          </button>
          {(['news', 'gov', 'intel', 'academic'] as const).map((type) => (
            <button
              key={type}
              onClick={() => setFilter(type)}
              className={`px-2 py-1 rounded transition-colors ${
                filter === type ? 'bg-cyan-500/20 text-cyan-400' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {sourceTypeIcons[type]} {type}
            </button>
          ))}

          <span className="flex-1" />

          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
            className="px-2 py-1 bg-slate-800 border border-slate-600 rounded text-slate-300 focus:outline-none"
          >
            <option value="date">By Date</option>
            <option value="reliability">By Reliability</option>
          </select>
        </div>
      </div>

      {/* Citation list */}
      <div className="divide-y divide-slate-800">
        {displayCitations.map((citation) => (
          <CitationItem
            key={citation.id}
            citation={citation}
            reliabilityLabels={reliabilityLabels}
            credibilityLabels={credibilityLabels}
            sourceTypeIcons={sourceTypeIcons}
            onRemove={onCitationRemove}
          />
        ))}

        {citations.length === 0 && (
          <div className="p-8 text-center text-slate-500 text-sm">
            No citations available
          </div>
        )}
      </div>

      {/* Show more */}
      {filteredCitations.length > maxDisplay && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full py-2 text-xs text-cyan-400 hover:text-cyan-300 transition-colors border-t border-slate-700"
        >
          {expanded ? 'Show less' : `Show ${filteredCitations.length - maxDisplay} more`}
        </button>
      )}
    </div>
  );
}

// Individual citation item
function CitationItem({
  citation,
  reliabilityLabels,
  credibilityLabels,
  sourceTypeIcons,
  onRemove,
}: {
  citation: Citation;
  reliabilityLabels: Record<Citation['reliability'], string>;
  credibilityLabels: Record<Citation['credibility'], string>;
  sourceTypeIcons: Record<Citation['sourceType'], string>;
  onRemove?: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);

  const reliabilityColor = {
    A: 'text-green-400 bg-green-500/20',
    B: 'text-emerald-400 bg-emerald-500/20',
    C: 'text-yellow-400 bg-yellow-500/20',
    D: 'text-orange-400 bg-orange-500/20',
    E: 'text-red-400 bg-red-500/20',
    F: 'text-slate-400 bg-slate-500/20',
  };

  return (
    <div className="p-3 hover:bg-slate-800/30 transition-colors">
      <div className="flex items-start gap-3">
        {/* Reliability/Credibility badge */}
        <div
          className={`flex-shrink-0 w-8 h-8 rounded flex items-center justify-center font-mono font-bold ${reliabilityColor[citation.reliability]}`}
          title={`${reliabilityLabels[citation.reliability]} / ${credibilityLabels[citation.credibility]}`}
        >
          {citation.reliability}{citation.credibility}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start gap-2">
            <span className="text-base">{sourceTypeIcons[citation.sourceType]}</span>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-500">{citation.sourceName}</span>
                <span className="text-xs text-slate-600">•</span>
                <span className="text-xs text-slate-500">
                  {new Date(citation.publishedAt).toLocaleDateString()}
                </span>
              </div>

              {citation.url ? (
                <a
                  href={citation.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-slate-200 hover:text-cyan-400 transition-colors line-clamp-1"
                >
                  {citation.title}
                </a>
              ) : (
                <span className="text-sm text-slate-200 line-clamp-1">{citation.title}</span>
              )}

              {citation.author && (
                <div className="text-xs text-slate-500 mt-0.5">by {citation.author}</div>
              )}

              {/* Excerpt (expandable) */}
              {citation.excerpt && (
                <button
                  onClick={() => setExpanded(!expanded)}
                  className="text-xs text-slate-400 hover:text-slate-300 mt-1"
                >
                  {expanded ? '▼ Hide excerpt' : '▶ Show excerpt'}
                </button>
              )}
              {expanded && citation.excerpt && (
                <blockquote className="mt-2 pl-3 border-l-2 border-slate-600 text-xs text-slate-400 italic">
                  "{citation.excerpt}"
                </blockquote>
              )}

              {/* Tags */}
              {citation.tags && citation.tags.length > 0 && (
                <div className="flex gap-1 mt-2">
                  {citation.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-1.5 py-0.5 bg-slate-800 rounded text-xs text-slate-400"
                    >
                      #{tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Remove button */}
        {onRemove && (
          <button
            onClick={() => onRemove(citation.id)}
            className="text-slate-500 hover:text-red-400 transition-colors"
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
}

// Inline citation reference
export function CitationRef({
  code,
  onClick,
}: {
  code: string; // e.g., "A2" or "B3"
  onClick?: () => void;
}) {
  const reliability = code[0] as Citation['reliability'];

  const color = {
    A: 'text-green-400',
    B: 'text-emerald-400',
    C: 'text-yellow-400',
    D: 'text-orange-400',
    E: 'text-red-400',
    F: 'text-slate-400',
  }[reliability] || 'text-slate-400';

  return (
    <sup
      onClick={onClick}
      className={`font-mono text-xs cursor-pointer hover:underline ${color}`}
    >
      [{code}]
    </sup>
  );
}

// Mock data
export const mockCitations: Citation[] = [
  {
    id: '1',
    sourceId: 'reuters',
    sourceName: 'Reuters',
    sourceType: 'wire',
    url: 'https://reuters.com/article/example',
    title: 'Russia-Ukraine conflict escalates with new missile strikes',
    author: 'Staff Reporter',
    publishedAt: '2024-01-15T10:00:00Z',
    accessedAt: '2024-01-15T10:30:00Z',
    reliability: 'A',
    credibility: 2,
    excerpt: 'Multiple sources confirm the launch of hypersonic missiles targeting infrastructure...',
    tags: ['ukraine', 'russia', 'military'],
  },
  {
    id: '2',
    sourceId: 'us-state',
    sourceName: 'US State Department',
    sourceType: 'gov',
    url: 'https://state.gov/briefing/2024',
    title: 'Press Briefing on Eastern European Security',
    publishedAt: '2024-01-14T15:00:00Z',
    accessedAt: '2024-01-15T09:00:00Z',
    reliability: 'B',
    credibility: 2,
    tags: ['nato', 'security'],
  },
  {
    id: '3',
    sourceId: 'osint-twitter',
    sourceName: '@OSINTAnalyst',
    sourceType: 'social',
    title: 'Thread: Satellite imagery analysis of troop movements',
    publishedAt: '2024-01-15T08:00:00Z',
    accessedAt: '2024-01-15T10:00:00Z',
    reliability: 'C',
    credibility: 3,
    excerpt: 'New satellite images show significant buildup near the border region...',
    tags: ['satellite', 'analysis'],
  },
  {
    id: '4',
    sourceId: 'rand-corp',
    sourceName: 'RAND Corporation',
    sourceType: 'academic',
    url: 'https://rand.org/pubs/research_reports/2024',
    title: 'Assessment of Baltic Security Architecture',
    author: 'Dr. Jane Smith',
    publishedAt: '2024-01-10T00:00:00Z',
    accessedAt: '2024-01-15T11:00:00Z',
    reliability: 'B',
    credibility: 2,
    tags: ['nato', 'baltic', 'research'],
  },
];
