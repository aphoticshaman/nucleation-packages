'use client';

import { useMemo } from 'react';

// NATO Admiralty Code for source evaluation
// Reliability: A-F (source reliability)
// Credibility: 1-6 (information credibility)

type ReliabilityGrade = 'A' | 'B' | 'C' | 'D' | 'E' | 'F';
type CredibilityGrade = '1' | '2' | '3' | '4' | '5' | '6';

interface SourceReliabilityProps {
  reliability: ReliabilityGrade;
  credibility: CredibilityGrade;
  showLabels?: boolean;
  size?: 'sm' | 'md' | 'lg';
  interactive?: boolean;
  onReliabilityChange?: (grade: ReliabilityGrade) => void;
  onCredibilityChange?: (grade: CredibilityGrade) => void;
}

const reliabilityDescriptions: Record<ReliabilityGrade, { label: string; description: string }> = {
  A: { label: 'Completely Reliable', description: 'No doubt of authenticity, trustworthiness, or competency; has a history of complete reliability' },
  B: { label: 'Usually Reliable', description: 'Minor doubt about authenticity, trustworthiness, or competency; has a history of valid information most of the time' },
  C: { label: 'Fairly Reliable', description: 'Doubt of authenticity, trustworthiness, or competency but has provided valid information in the past' },
  D: { label: 'Not Usually Reliable', description: 'Significant doubt about authenticity, trustworthiness, or competency but has provided valid information in the past' },
  E: { label: 'Unreliable', description: 'Lacking in authenticity, trustworthiness, and competency; history of invalid information' },
  F: { label: 'Cannot Be Judged', description: 'No basis for evaluating the reliability of the source' },
};

const credibilityDescriptions: Record<CredibilityGrade, { label: string; description: string }> = {
  '1': { label: 'Confirmed', description: 'Confirmed by other independent sources; logical in itself; consistent with other information on the subject' },
  '2': { label: 'Probably True', description: 'Not confirmed; logical in itself; consistent with other information on the subject' },
  '3': { label: 'Possibly True', description: 'Not confirmed; reasonably logical in itself; agrees with some other information on the subject' },
  '4': { label: 'Doubtfully True', description: 'Not confirmed; possible but not logical; no other information on the subject' },
  '5': { label: 'Improbable', description: 'Not confirmed; not logical in itself; contradicted by other information on the subject' },
  '6': { label: 'Cannot Be Judged', description: 'No basis for evaluating the validity of the information' },
};

// Component 51: NATO Admiralty Code Source Reliability Indicator
export function SourceReliability({
  reliability,
  credibility,
  showLabels = true,
  size = 'md',
  interactive = false,
  onReliabilityChange,
  onCredibilityChange,
}: SourceReliabilityProps) {
  const reliabilityConfig = useMemo(() => {
    const colors: Record<ReliabilityGrade, string> = {
      A: 'bg-green-500 border-green-400',
      B: 'bg-cyan-500 border-cyan-400',
      C: 'bg-amber-500 border-amber-400',
      D: 'bg-orange-500 border-orange-400',
      E: 'bg-red-500 border-red-400',
      F: 'bg-slate-500 border-slate-400',
    };
    return colors;
  }, []);

  const credibilityConfig = useMemo(() => {
    const colors: Record<CredibilityGrade, string> = {
      '1': 'bg-green-500 border-green-400',
      '2': 'bg-cyan-500 border-cyan-400',
      '3': 'bg-amber-500 border-amber-400',
      '4': 'bg-orange-500 border-orange-400',
      '5': 'bg-red-500 border-red-400',
      '6': 'bg-slate-500 border-slate-400',
    };
    return colors;
  }, []);

  const sizeConfig = {
    sm: { badge: 'w-5 h-5 text-xs', text: 'text-xs' },
    md: { badge: 'w-8 h-8 text-sm', text: 'text-sm' },
    lg: { badge: 'w-10 h-10 text-base', text: 'text-base' },
  };

  const sizes = sizeConfig[size];

  return (
    <div className="inline-flex items-center gap-1">
      {/* Reliability badge */}
      <div className="relative group">
        <div
          className={`
            ${sizes.badge} rounded-full flex items-center justify-center
            font-bold text-slate-900 border-2
            ${reliabilityConfig[reliability]}
            ${interactive ? 'cursor-pointer hover:scale-110 transition-transform' : ''}
          `}
          onClick={() => interactive && onReliabilityChange?.((
            String.fromCharCode(reliability.charCodeAt(0) + 1 > 70 ? 65 : reliability.charCodeAt(0) + 1)
          ) as ReliabilityGrade)}
        >
          {reliability}
        </div>

        {/* Tooltip */}
        <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-20">
          <div className="bg-slate-800 border border-slate-600 rounded-lg shadow-xl p-2 min-w-[200px]">
            <div className="text-xs font-medium text-slate-200">
              {reliabilityDescriptions[reliability].label}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              {reliabilityDescriptions[reliability].description}
            </div>
          </div>
        </div>
      </div>

      {/* Credibility badge */}
      <div className="relative group">
        <div
          className={`
            ${sizes.badge} rounded-full flex items-center justify-center
            font-bold text-slate-900 border-2
            ${credibilityConfig[credibility]}
            ${interactive ? 'cursor-pointer hover:scale-110 transition-transform' : ''}
          `}
          onClick={() => interactive && onCredibilityChange?.((
            parseInt(credibility) >= 6 ? '1' : String(parseInt(credibility) + 1)
          ) as CredibilityGrade)}
        >
          {credibility}
        </div>

        {/* Tooltip */}
        <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-20">
          <div className="bg-slate-800 border border-slate-600 rounded-lg shadow-xl p-2 min-w-[200px]">
            <div className="text-xs font-medium text-slate-200">
              {credibilityDescriptions[credibility].label}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              {credibilityDescriptions[credibility].description}
            </div>
          </div>
        </div>
      </div>

      {/* Labels */}
      {showLabels && (
        <span className={`text-slate-400 ml-1 ${sizes.text}`}>
          ({reliabilityDescriptions[reliability].label.split(' ')[0]})
        </span>
      )}
    </div>
  );
}

// Full reliability matrix for reference
export function ReliabilityMatrix({
  onSelect,
  selectedReliability,
  selectedCredibility,
}: {
  onSelect?: (reliability: ReliabilityGrade, credibility: CredibilityGrade) => void;
  selectedReliability?: ReliabilityGrade;
  selectedCredibility?: CredibilityGrade;
}) {
  const reliabilityGrades: ReliabilityGrade[] = ['A', 'B', 'C', 'D', 'E', 'F'];
  const credibilityGrades: CredibilityGrade[] = ['1', '2', '3', '4', '5', '6'];

  const getCellColor = (r: ReliabilityGrade, c: CredibilityGrade) => {
    const rScore = 6 - (r.charCodeAt(0) - 64); // A=6, F=1
    const cScore = 7 - parseInt(c); // 1=6, 6=1
    const combined = (rScore + cScore) / 2;

    if (combined >= 5) return 'bg-green-500/30 hover:bg-green-500/50';
    if (combined >= 4) return 'bg-cyan-500/30 hover:bg-cyan-500/50';
    if (combined >= 3) return 'bg-amber-500/30 hover:bg-amber-500/50';
    if (combined >= 2) return 'bg-orange-500/30 hover:bg-orange-500/50';
    return 'bg-red-500/30 hover:bg-red-500/50';
  };

  return (
    <div className="bg-slate-900/50 rounded-lg border border-slate-700 p-4">
      <h3 className="text-sm font-medium text-slate-200 mb-4">NATO Admiralty Code Matrix</h3>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr>
              <th className="p-2 text-left text-xs text-slate-400" />
              {credibilityGrades.map(c => (
                <th key={c} className="p-2 text-center">
                  <div className="text-sm font-bold text-slate-300">{c}</div>
                  <div className="text-xs text-slate-500">{credibilityDescriptions[c].label}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {reliabilityGrades.map(r => (
              <tr key={r}>
                <td className="p-2">
                  <div className="text-sm font-bold text-slate-300">{r}</div>
                  <div className="text-xs text-slate-500 max-w-[120px] truncate">
                    {reliabilityDescriptions[r].label}
                  </div>
                </td>
                {credibilityGrades.map(c => {
                  const isSelected = selectedReliability === r && selectedCredibility === c;
                  return (
                    <td key={c} className="p-1">
                      <button
                        onClick={() => onSelect?.(r, c)}
                        className={`
                          w-full h-10 rounded font-bold text-slate-200
                          transition-all ${getCellColor(r, c)}
                          ${isSelected ? 'ring-2 ring-cyan-400 scale-105' : ''}
                        `}
                      >
                        {r}{c}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mt-4 pt-4 border-t border-slate-700">
        <span className="text-xs text-slate-500">Confidence:</span>
        {[
          { label: 'High', color: 'bg-green-500' },
          { label: 'Med-High', color: 'bg-cyan-500' },
          { label: 'Medium', color: 'bg-amber-500' },
          { label: 'Med-Low', color: 'bg-orange-500' },
          { label: 'Low', color: 'bg-red-500' },
        ].map(item => (
          <div key={item.label} className="flex items-center gap-1">
            <div className={`w-3 h-3 rounded ${item.color}`} />
            <span className="text-xs text-slate-400">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Inline source reliability display for citations
export function InlineReliability({
  reliability,
  credibility,
  source,
}: {
  reliability: ReliabilityGrade;
  credibility: CredibilityGrade;
  source: string;
}) {
  return (
    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 bg-slate-800 rounded text-xs">
      <span className="text-slate-400">{source}</span>
      <SourceReliability
        reliability={reliability}
        credibility={credibility}
        showLabels={false}
        size="sm"
      />
    </span>
  );
}

// Aggregate reliability score from multiple sources
export function AggregateReliability({
  sources,
}: {
  sources: { reliability: ReliabilityGrade; credibility: CredibilityGrade; weight?: number }[];
}) {
  const aggregateScore = useMemo(() => {
    if (sources.length === 0) return 0;

    let totalWeight = 0;
    let weightedScore = 0;

    for (const source of sources) {
      const rScore = 6 - (source.reliability.charCodeAt(0) - 64);
      const cScore = 7 - parseInt(source.credibility);
      const weight = source.weight ?? 1;

      weightedScore += ((rScore + cScore) / 2) * weight;
      totalWeight += weight;
    }

    return (weightedScore / totalWeight) / 6; // Normalize to 0-1
  }, [sources]);

  const label = aggregateScore >= 0.8 ? 'High Confidence' :
                aggregateScore >= 0.6 ? 'Moderate Confidence' :
                aggregateScore >= 0.4 ? 'Low Confidence' :
                'Very Low Confidence';

  const color = aggregateScore >= 0.8 ? 'text-green-400' :
                aggregateScore >= 0.6 ? 'text-cyan-400' :
                aggregateScore >= 0.4 ? 'text-amber-400' :
                'text-red-400';

  return (
    <div className="flex items-center gap-2">
      <div className="w-20 h-2 bg-slate-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${
            aggregateScore >= 0.8 ? 'bg-green-500' :
            aggregateScore >= 0.6 ? 'bg-cyan-500' :
            aggregateScore >= 0.4 ? 'bg-amber-500' :
            'bg-red-500'
          }`}
          style={{ width: `${aggregateScore * 100}%` }}
        />
      </div>
      <span className={`text-xs font-medium ${color}`}>{label}</span>
      <span className="text-xs text-slate-500">({sources.length} sources)</span>
    </div>
  );
}
