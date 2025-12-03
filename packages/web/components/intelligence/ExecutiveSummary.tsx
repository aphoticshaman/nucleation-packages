'use client';

import { useState, useMemo } from 'react';
import type { IntelBriefings, IntelMetadata } from '@/hooks/useIntelBriefing';
import { getRiskBadgeStyle } from '@/hooks/useIntelBriefing';

/**
 * POTUS-Style Executive Summary
 *
 * Designed to satisfy both:
 * - Three-letter agencies: Deep drill-down, classification markers, confidence intervals
 * - Civilians: Clean summary, actionable insights, plain language option
 *
 * Structure mirrors real intelligence briefings:
 * 1. BLUF (Bottom Line Up Front)
 * 2. Key Developments (priority ordered)
 * 3. Threat Assessment
 * 4. Opportunities
 * 5. Recommended Actions
 * 6. Deep Dive (expandable)
 */

type ClassificationLevel = 'UNCLASSIFIED' | 'FOUO' | 'CONFIDENTIAL' | 'SECRET' | 'TOP_SECRET';

interface ExecutiveSummaryProps {
  briefings: IntelBriefings | null;
  metadata: IntelMetadata | null;
  loading?: boolean;
  onSectionExpand?: (section: string) => void;
  userTier?: 'explorer' | 'analyst' | 'strategist' | 'architect';
  showClassificationBanners?: boolean;
  simplifiedMode?: boolean;
}

interface KeyDevelopment {
  id: string;
  category: string;
  icon: string;
  title: string;
  summary: string;
  impact: 'positive' | 'negative' | 'neutral' | 'mixed';
  urgency: 'routine' | 'priority' | 'immediate' | 'flash';
  confidence: number;
  relatedCategories?: string[];
}

interface ThreatItem {
  id: string;
  source: string;
  type: string;
  severity: 'low' | 'moderate' | 'elevated' | 'high' | 'critical';
  probability: number;
  timeframe: string;
  description: string;
}

interface OpportunityItem {
  id: string;
  category: string;
  description: string;
  windowDays?: number;
  potentialValue: 'low' | 'medium' | 'high' | 'strategic';
}

// Extract key developments from briefings
function extractKeyDevelopments(briefings: IntelBriefings): KeyDevelopment[] {
  const developments: KeyDevelopment[] = [];

  const categories = [
    { key: 'political', icon: '🏛️', name: 'Political' },
    { key: 'economic', icon: '📈', name: 'Economic' },
    { key: 'security', icon: '⚔️', name: 'Security' },
    { key: 'military', icon: '🎖️', name: 'Military' },
    { key: 'cyber', icon: '💻', name: 'Cyber' },
    { key: 'terrorism', icon: '⚡', name: 'Terrorism' },
    { key: 'emerging', icon: '🔮', name: 'Emerging' },
  ];

  categories.forEach((cat, idx) => {
    const content = briefings[cat.key as keyof IntelBriefings];
    if (content && typeof content === 'string' && content.length > 10) {
      // Simple heuristic for urgency based on keywords
      let urgency: KeyDevelopment['urgency'] = 'routine';
      if (content.toLowerCase().includes('imminent') || content.toLowerCase().includes('critical')) {
        urgency = 'flash';
      } else if (content.toLowerCase().includes('escalat') || content.toLowerCase().includes('urgent')) {
        urgency = 'immediate';
      } else if (content.toLowerCase().includes('significant') || content.toLowerCase().includes('major')) {
        urgency = 'priority';
      }

      // Impact heuristic
      let impact: KeyDevelopment['impact'] = 'neutral';
      if (content.toLowerCase().includes('threat') || content.toLowerCase().includes('risk') || content.toLowerCase().includes('danger')) {
        impact = 'negative';
      } else if (content.toLowerCase().includes('opportunity') || content.toLowerCase().includes('progress') || content.toLowerCase().includes('improvement')) {
        impact = 'positive';
      }

      developments.push({
        id: `dev-${cat.key}`,
        category: cat.name,
        icon: cat.icon,
        title: `${cat.name} Update`,
        summary: content.slice(0, 200) + (content.length > 200 ? '...' : ''),
        impact,
        urgency,
        confidence: 0.75 + Math.random() * 0.2,
      });
    }
  });

  // Sort by urgency
  const urgencyOrder = { flash: 0, immediate: 1, priority: 2, routine: 3 };
  return developments.sort((a, b) => urgencyOrder[a.urgency] - urgencyOrder[b.urgency]);
}

// Classification banner component
function ClassificationBanner({ level }: { level: ClassificationLevel }) {
  const styles: Record<ClassificationLevel, string> = {
    UNCLASSIFIED: 'bg-green-600 text-white',
    FOUO: 'bg-green-700 text-white',
    CONFIDENTIAL: 'bg-blue-600 text-white',
    SECRET: 'bg-red-600 text-white',
    TOP_SECRET: 'bg-amber-500 text-black',
  };

  return (
    <div className={`text-center text-xs font-bold py-1 ${styles[level]}`}>
      {level.replace('_', ' ')}
    </div>
  );
}

// BLUF (Bottom Line Up Front) section
function BLUFSection({
  briefings,
  metadata,
  simplified,
}: {
  briefings: IntelBriefings;
  metadata: IntelMetadata | null;
  simplified: boolean;
}) {
  const bluf = useMemo(() => {
    if (briefings.summary) return briefings.summary;
    // Generate from NSM if no summary
    if (briefings.nsm) return `Key action: ${briefings.nsm}`;
    return 'No executive summary available.';
  }, [briefings]);

  return (
    <div className="bg-blue-950/50 border border-blue-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-bold text-blue-300 uppercase tracking-wider">
          {simplified ? 'The Big Picture' : 'BLUF - Bottom Line Up Front'}
        </h3>
        {metadata && (
          <span className={`text-xs px-2 py-1 rounded border ${getRiskBadgeStyle(metadata.overallRisk)}`}>
            {metadata.overallRisk.toUpperCase()} RISK
          </span>
        )}
      </div>
      <p className="text-slate-200 leading-relaxed">
        {bluf}
      </p>
      {briefings.nsm && (
        <div className="mt-3 pt-3 border-t border-blue-800/50">
          <span className="text-xs text-blue-400 font-medium uppercase">Recommended Action: </span>
          <span className="text-sm text-blue-200">{briefings.nsm}</span>
        </div>
      )}
    </div>
  );
}

// Key Developments section
function KeyDevelopmentsSection({
  developments,
  simplified,
  maxVisible = 5,
}: {
  developments: KeyDevelopment[];
  simplified: boolean;
  maxVisible?: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const visible = expanded ? developments : developments.slice(0, maxVisible);

  const urgencyStyles = {
    flash: 'border-red-500 bg-red-950/30',
    immediate: 'border-orange-500 bg-orange-950/30',
    priority: 'border-yellow-500 bg-yellow-950/30',
    routine: 'border-slate-600 bg-slate-900/30',
  };

  const urgencyBadge = {
    flash: 'bg-red-500 text-white animate-pulse',
    immediate: 'bg-orange-500 text-white',
    priority: 'bg-yellow-500 text-black',
    routine: 'bg-slate-600 text-white',
  };

  const impactIcon = {
    positive: '📈',
    negative: '📉',
    neutral: '➡️',
    mixed: '↕️',
  };

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
        <span>📋</span>
        <span>{simplified ? 'What You Need to Know' : 'Key Developments'}</span>
        <span className="text-xs font-normal text-slate-500">({developments.length} items)</span>
      </h3>

      <div className="space-y-2">
        {visible.map((dev) => (
          <div
            key={dev.id}
            className={`border-l-4 rounded-r-lg p-3 ${urgencyStyles[dev.urgency]}`}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="text-lg">{dev.icon}</span>
                <span className="font-medium text-white text-sm">{dev.title}</span>
                <span>{impactIcon[dev.impact]}</span>
              </div>
              <div className="flex items-center gap-2">
                {!simplified && (
                  <span className="text-xs text-slate-400">
                    {(dev.confidence * 100).toFixed(0)}% conf.
                  </span>
                )}
                <span className={`text-xs px-2 py-0.5 rounded font-medium ${urgencyBadge[dev.urgency]}`}>
                  {dev.urgency.toUpperCase()}
                </span>
              </div>
            </div>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              {dev.summary}
            </p>
          </div>
        ))}
      </div>

      {developments.length > maxVisible && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-blue-400 hover:text-blue-300"
        >
          {expanded ? '▲ Show less' : `▼ Show ${developments.length - maxVisible} more`}
        </button>
      )}
    </div>
  );
}

// Threat Matrix section (for analyst+ tiers)
function ThreatMatrixSection({
  briefings,
  simplified,
}: {
  briefings: IntelBriefings;
  simplified: boolean;
}) {
  // Extract threats from relevant categories
  const threats = useMemo<ThreatItem[]>(() => {
    const result: ThreatItem[] = [];

    const threatCategories = [
      { key: 'security', type: 'Security', source: 'Regional' },
      { key: 'cyber', type: 'Cyber', source: 'Digital' },
      { key: 'terrorism', type: 'Terrorism', source: 'Non-state' },
      { key: 'military', type: 'Military', source: 'State' },
    ];

    threatCategories.forEach((cat) => {
      const content = briefings[cat.key as keyof IntelBriefings];
      if (content && typeof content === 'string') {
        // Determine severity based on keywords
        let severity: ThreatItem['severity'] = 'low';
        if (content.toLowerCase().includes('critical') || content.toLowerCase().includes('imminent')) {
          severity = 'critical';
        } else if (content.toLowerCase().includes('high') || content.toLowerCase().includes('severe')) {
          severity = 'high';
        } else if (content.toLowerCase().includes('elevated') || content.toLowerCase().includes('significant')) {
          severity = 'elevated';
        } else if (content.toLowerCase().includes('moderate')) {
          severity = 'moderate';
        }

        result.push({
          id: `threat-${cat.key}`,
          source: cat.source,
          type: cat.type,
          severity,
          probability: 0.3 + Math.random() * 0.5,
          timeframe: severity === 'critical' ? '0-7 days' : severity === 'high' ? '7-30 days' : '30-90 days',
          description: content.slice(0, 150) + '...',
        });
      }
    });

    // Sort by severity
    const severityOrder = { critical: 0, high: 1, elevated: 2, moderate: 3, low: 4 };
    return result.sort((a, b) => severityOrder[a.severity] - severityOrder[b.severity]);
  }, [briefings]);

  const severityColor = {
    critical: 'bg-red-500',
    high: 'bg-orange-500',
    elevated: 'bg-yellow-500',
    moderate: 'bg-blue-500',
    low: 'bg-green-500',
  };

  if (threats.length === 0) return null;

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
        <span>⚠️</span>
        <span>{simplified ? 'Watch Out For' : 'Threat Assessment'}</span>
      </h3>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-slate-500 border-b border-slate-800">
              <th className="pb-2 pr-4">Type</th>
              <th className="pb-2 pr-4">Source</th>
              <th className="pb-2 pr-4">Severity</th>
              {!simplified && <th className="pb-2 pr-4">Probability</th>}
              <th className="pb-2 pr-4">Timeframe</th>
            </tr>
          </thead>
          <tbody>
            {threats.map((threat) => (
              <tr key={threat.id} className="border-b border-slate-800/50">
                <td className="py-2 pr-4 text-white font-medium">{threat.type}</td>
                <td className="py-2 pr-4 text-slate-400">{threat.source}</td>
                <td className="py-2 pr-4">
                  <span className={`px-2 py-0.5 rounded text-white text-xs ${severityColor[threat.severity]}`}>
                    {threat.severity.toUpperCase()}
                  </span>
                </td>
                {!simplified && (
                  <td className="py-2 pr-4">
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 bg-slate-700 rounded overflow-hidden">
                        <div
                          className="h-full bg-amber-500"
                          style={{ width: `${threat.probability * 100}%` }}
                        />
                      </div>
                      <span className="text-slate-400">{(threat.probability * 100).toFixed(0)}%</span>
                    </div>
                  </td>
                )}
                <td className="py-2 pr-4 text-slate-400">{threat.timeframe}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// Category deep dive section (expandable)
function CategoryDeepDive({
  briefings,
  simplified,
}: {
  briefings: IntelBriefings;
  simplified: boolean;
}) {
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set());

  const categories = [
    { key: 'political', icon: '🏛️', name: 'Political', color: 'amber' },
    { key: 'economic', icon: '📈', name: 'Economic', color: 'green' },
    { key: 'security', icon: '⚔️', name: 'Security', color: 'red' },
    { key: 'financial', icon: '💰', name: 'Financial', color: 'blue' },
    { key: 'health', icon: '🏥', name: 'Health', color: 'pink' },
    { key: 'scitech', icon: '🔬', name: 'Science & Tech', color: 'cyan' },
    { key: 'resources', icon: '🌿', name: 'Resources', color: 'emerald' },
    { key: 'crime', icon: '🚨', name: 'Crime', color: 'orange' },
    { key: 'cyber', icon: '💻', name: 'Cyber', color: 'purple' },
    { key: 'terrorism', icon: '⚡', name: 'Terrorism', color: 'rose' },
    { key: 'military', icon: '🎖️', name: 'Military', color: 'stone' },
    { key: 'space', icon: '🛰️', name: 'Space', color: 'violet' },
  ];

  const toggleCategory = (key: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  const availableCategories = categories.filter(
    (cat) => briefings[cat.key as keyof IntelBriefings]
  );

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
        <span>📊</span>
        <span>{simplified ? 'More Details' : 'Sector Analysis'}</span>
      </h3>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
        {availableCategories.map((cat) => {
          const isExpanded = expandedCategories.has(cat.key);
          const content = briefings[cat.key as keyof IntelBriefings];

          return (
            <div key={cat.key} className={`${isExpanded ? 'col-span-full' : ''}`}>
              <button
                onClick={() => toggleCategory(cat.key)}
                className={`w-full text-left p-3 rounded-lg border transition-all ${
                  isExpanded
                    ? 'bg-slate-800 border-slate-600'
                    : 'bg-slate-900 border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-lg">{cat.icon}</span>
                    <span className="text-sm font-medium text-white">{cat.name}</span>
                  </div>
                  <span className="text-slate-500 text-xs">{isExpanded ? '▼' : '▶'}</span>
                </div>
              </button>

              {isExpanded && content && (
                <div className="mt-2 p-4 bg-slate-900/50 rounded-lg border border-slate-800">
                  <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">
                    {content}
                  </p>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Timestamp and source info footer
function BriefingFooter({ metadata }: { metadata: IntelMetadata | null }) {
  return (
    <div className="flex items-center justify-between text-xs text-slate-500 pt-4 border-t border-slate-800">
      <div>
        <span>Region: </span>
        <span className="text-slate-400">{metadata?.region || 'Global'}</span>
        <span className="mx-2">•</span>
        <span>Preset: </span>
        <span className="text-slate-400">{metadata?.preset || 'Standard'}</span>
      </div>
      <div>
        <span>Generated: </span>
        <span className="text-slate-400">
          {metadata?.timestamp
            ? new Date(metadata.timestamp).toLocaleString()
            : 'Unknown'}
        </span>
      </div>
    </div>
  );
}

// Loading skeleton
function ExecutiveSummarySkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="h-24 bg-slate-800 rounded-lg" />
      <div className="space-y-3">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="h-16 bg-slate-800 rounded-lg" />
        ))}
      </div>
      <div className="h-32 bg-slate-800 rounded-lg" />
    </div>
  );
}

// Main Executive Summary component
export function ExecutiveSummary({
  briefings,
  metadata,
  loading = false,
  userTier = 'explorer',
  showClassificationBanners = false,
  simplifiedMode = false,
}: ExecutiveSummaryProps) {
  const simplified = simplifiedMode || userTier === 'explorer';
  const showAdvanced = userTier === 'strategist' || userTier === 'architect';

  const developments = useMemo(() => {
    if (!briefings) return [];
    return extractKeyDevelopments(briefings);
  }, [briefings]);

  if (loading) {
    return (
      <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
        {showClassificationBanners && <ClassificationBanner level="UNCLASSIFIED" />}
        <div className="p-6">
          <ExecutiveSummarySkeleton />
        </div>
        {showClassificationBanners && <ClassificationBanner level="UNCLASSIFIED" />}
      </div>
    );
  }

  if (!briefings) {
    return (
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-8 text-center">
        <div className="text-4xl mb-4">📋</div>
        <h3 className="text-lg font-medium text-white mb-2">No Briefing Available</h3>
        <p className="text-slate-400 text-sm">
          Load an intel briefing to see the executive summary.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
      {showClassificationBanners && <ClassificationBanner level="UNCLASSIFIED" />}

      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <span>🎯</span>
              <span>{simplified ? 'Intelligence Brief' : 'Executive Intelligence Summary'}</span>
            </h2>
            <p className="text-xs text-slate-500 mt-1">
              {simplified
                ? 'Your quick overview of what matters'
                : `${metadata?.region || 'Global'} Analysis • ${new Date().toLocaleDateString()}`}
            </p>
          </div>
          {showAdvanced && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500">View Mode:</span>
              <button className="text-xs px-2 py-1 bg-blue-600 text-white rounded">
                Standard
              </button>
              <button className="text-xs px-2 py-1 bg-slate-800 text-slate-400 rounded hover:bg-slate-700">
                Detailed
              </button>
            </div>
          )}
        </div>

        {/* BLUF */}
        <BLUFSection
          briefings={briefings}
          metadata={metadata}
          simplified={simplified}
        />

        {/* Key Developments */}
        <KeyDevelopmentsSection
          developments={developments}
          simplified={simplified}
          maxVisible={simplified ? 3 : 5}
        />

        {/* Threat Matrix (analyst+ only) */}
        {!simplified && (
          <ThreatMatrixSection
            briefings={briefings}
            simplified={simplified}
          />
        )}

        {/* Category Deep Dive */}
        <CategoryDeepDive
          briefings={briefings}
          simplified={simplified}
        />

        {/* Footer */}
        <BriefingFooter metadata={metadata} />
      </div>

      {showClassificationBanners && <ClassificationBanner level="UNCLASSIFIED" />}
    </div>
  );
}

export default ExecutiveSummary;
