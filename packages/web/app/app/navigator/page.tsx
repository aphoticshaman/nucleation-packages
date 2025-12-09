'use client';

import { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { useIntelBriefing } from '@/hooks/useIntelBriefing';
import { useSupabaseNations } from '@/hooks/useSupabaseNations';
import { briefingsToTreeNodes, type IntelNode, type IntelCategory } from '@/components/intelligence/TreeNavigator3D';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';

// Dynamic import to avoid SSR issues with Three.js
const TreeNavigator3D = dynamic(
  () => import('@/components/intelligence/TreeNavigator3D').then((mod) => mod.TreeNavigator3D),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-[calc(100vh-8rem)] bg-slate-900 rounded-xl flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-4" />
          <p className="text-slate-400">Loading 3D Navigator...</p>
        </div>
      </div>
    ),
  }
);

// Preset filters for nations (same as main dashboard)
const PRESET_FILTERS: Record<string, string[] | null> = {
  global: null,
  nato: ['USA', 'CAN', 'GBR', 'FRA', 'DEU', 'ITA', 'ESP', 'POL', 'NLD', 'UKR', 'GEO', 'JPN', 'KOR', 'AUS'],
  brics: ['BRA', 'RUS', 'IND', 'CHN', 'ZAF', 'IRN', 'EGY', 'ETH', 'SAU', 'ARE', 'ARG', 'MEX', 'TUR', 'IDN'],
  conflict: ['UKR', 'RUS', 'ISR', 'PSE', 'LBN', 'SYR', 'IRN', 'YEM', 'SDN', 'TWN', 'CHN', 'PRK', 'MMR', 'AFG'],
};

// Convert nation risk to category
function riskToCategory(risk: number): 'low' | 'moderate' | 'elevated' | 'high' | 'critical' {
  if (risk > 0.8) return 'critical';
  if (risk > 0.6) return 'high';
  if (risk > 0.4) return 'elevated';
  if (risk > 0.2) return 'moderate';
  return 'low';
}

export default function NavigatorPage() {
  const [selectedPreset, setSelectedPreset] = useState('global');
  const { briefings, metadata, loading: briefingLoading } = useIntelBriefing(selectedPreset, { autoFetch: true });
  const { nations: allNations, loading: nationsLoading } = useSupabaseNations();
  const [selectedNode, setSelectedNode] = useState<IntelNode | null>(null);

  // Filter nations by preset
  const nations = useMemo(() => {
    const filter = PRESET_FILTERS[selectedPreset];
    if (!filter) return allNations.slice(0, 30); // Limit global to top 30 by risk
    return allNations.filter(n => filter.includes(n.code));
  }, [allNations, selectedPreset]);

  // Fuse briefings + nations into unified 3D nodes
  const nodes = useMemo(() => {
    const fusedNodes: IntelNode[] = [];
    const now = metadata?.timestamp ? new Date(metadata.timestamp) : new Date();

    // 1. Briefing category nodes (Elle's analysis)
    if (briefings) {
      const briefingNodes = briefingsToTreeNodes(
        briefings as unknown as Record<string, string>,
        metadata ?? undefined
      );
      fusedNodes.push(...briefingNodes);
    }

    // 2. Nation-level nodes (quantitative risk data)
    // These orbit around their risk category in 3D space
    nations.forEach((nation) => {
      const riskLevel = riskToCategory(nation.transition_risk || 0);
      const stability = nation.basin_strength || 0.5;

      // Determine category based on nation's primary concern
      let category: IntelCategory = 'security';
      if (nation.transition_risk > 0.7) category = 'security';
      else if (nation.regime && nation.regime < 4) category = 'political';
      else category = 'economic';

      fusedNodes.push({
        id: `nation-${nation.code}`,
        label: nation.name || nation.code,
        category,
        content: `${nation.name || nation.code}: Stability ${(stability * 100).toFixed(0)}%, Risk ${((nation.transition_risk || 0) * 100).toFixed(0)}%`,
        timestamp: now,
        temporalState: 'current',
        confidence: stability,
        risk: riskLevel,
        parentId: `${category}-current`,
      });
    });

    // 3. Add aggregate metrics node
    if (nations.length > 0) {
      const avgRisk = nations.reduce((sum, n) => sum + (n.transition_risk || 0), 0) / nations.length;
      const avgStability = nations.reduce((sum, n) => sum + (n.basin_strength || 0.5), 0) / nations.length;

      fusedNodes.push({
        id: 'aggregate-metrics',
        label: `${selectedPreset.toUpperCase()} Metrics`,
        category: 'executive',
        content: `${nations.length} nations analyzed. Avg Risk: ${(avgRisk * 100).toFixed(0)}%, Avg Stability: ${(avgStability * 100).toFixed(0)}%`,
        timestamp: now,
        temporalState: 'current',
        confidence: 0.9,
        risk: riskToCategory(avgRisk),
        parentId: 'executive-summary',
      });
    }

    return fusedNodes;
  }, [briefings, metadata, nations, selectedPreset]);

  const loading = briefingLoading || nationsLoading;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">3D Navigator</h1>
          <p className="text-slate-400 text-sm mt-1">
            Explore geopolitical relationships in 3D space
          </p>
        </div>

        {/* Preset selector */}
        <div className="flex items-center gap-2">
          {['global', 'nato', 'brics', 'conflict'].map((preset) => (
            <GlassButton
              key={preset}
              variant={selectedPreset === preset ? 'primary' : 'ghost'}
              size="sm"
              onClick={() => setSelectedPreset(preset)}
            >
              {preset.charAt(0).toUpperCase() + preset.slice(1)}
            </GlassButton>
          ))}
        </div>

        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span>Scroll to zoom</span>
          <span>•</span>
          <span>Drag to rotate</span>
          <span>•</span>
          <span>Click nodes for details</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* 3D View */}
        <GlassCard blur="heavy" className="lg:col-span-3 overflow-hidden h-[calc(100vh-12rem)] !p-0">
          {loading ? (
            <div className="w-full h-full flex items-center justify-center">
              <div className="text-center">
                <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-4" />
                <p className="text-slate-400">Loading intel data...</p>
              </div>
            </div>
          ) : nodes.length > 0 ? (
            <TreeNavigator3D
              nodes={nodes}
              onNodeSelect={setSelectedNode}
              selectedNodeId={selectedNode?.id}
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <p className="text-slate-500">No data available. Check back soon.</p>
            </div>
          )}
        </GlassCard>

        {/* Details Panel */}
        <GlassCard blur="heavy" className="lg:col-span-1">
          <h3 className="text-white font-medium mb-4">Node Details</h3>
          {selectedNode ? (
            <div className="space-y-3">
              <div>
                <p className="text-xs text-slate-500">
                  {selectedNode.id.startsWith('nation-') ? 'Nation' : 'Category'}
                </p>
                <p className="text-white font-medium">{selectedNode.label}</p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Risk Level</p>
                <p className={`font-semibold ${
                  selectedNode.risk === 'critical' ? 'text-red-400' :
                  selectedNode.risk === 'high' ? 'text-orange-400' :
                  selectedNode.risk === 'elevated' ? 'text-yellow-400' :
                  selectedNode.risk === 'moderate' ? 'text-blue-400' :
                  'text-green-400'
                }`}>
                  {selectedNode.risk?.toUpperCase() || 'MODERATE'}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">
                  {selectedNode.id.startsWith('nation-') ? 'Stability Score' : 'Confidence'}
                </p>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${
                        (selectedNode.confidence || 0.5) > 0.7 ? 'bg-green-500' :
                        (selectedNode.confidence || 0.5) > 0.4 ? 'bg-yellow-500' :
                        'bg-red-500'
                      }`}
                      style={{ width: `${(selectedNode.confidence || 0.5) * 100}%` }}
                    />
                  </div>
                  <span className="text-white text-sm">
                    {((selectedNode.confidence || 0.5) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <div>
                <p className="text-xs text-slate-500">
                  {selectedNode.id.startsWith('nation-') ? 'Metrics' : 'Analysis'}
                </p>
                <p className="text-slate-300 text-sm leading-relaxed">{selectedNode.content}</p>
              </div>
              {selectedNode.parentId && (
                <div>
                  <p className="text-xs text-slate-500">Data Source</p>
                  <p className="text-slate-400 text-xs">
                    {selectedNode.id.startsWith('nation-')
                      ? '📊 Supabase nations table'
                      : '🤖 Elle intel briefing'
                    }
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-4">
              <p className="text-slate-500 text-sm mb-2">Select a node to view details</p>
              <p className="text-slate-600 text-xs">
                {nations.length} nations + {nodes.length - nations.length} briefings loaded
              </p>
            </div>
          )}
        </GlassCard>
      </div>
    </div>
  );
}
