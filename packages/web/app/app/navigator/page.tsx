'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import { useIntelBriefing } from '@/hooks/useIntelBriefing';
import { briefingsToTreeNodes, type IntelNode } from '@/components/intelligence/TreeNavigator3D';
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

export default function NavigatorPage() {
  const [selectedPreset, setSelectedPreset] = useState('global');
  const { briefings, metadata, loading } = useIntelBriefing(selectedPreset, { autoFetch: true });
  const [selectedNode, setSelectedNode] = useState<IntelNode | null>(null);

  // Convert briefings to 3D nodes
  const nodes = briefings
    ? briefingsToTreeNodes(briefings as unknown as Record<string, string>, metadata ?? undefined)
    : [];

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
                <p className="text-xs text-slate-500">Category</p>
                <p className="text-white">{selectedNode.label}</p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Risk Level</p>
                <p className={`${
                  selectedNode.risk === 'critical' ? 'text-red-400' :
                  selectedNode.risk === 'high' ? 'text-orange-400' :
                  selectedNode.risk === 'elevated' ? 'text-yellow-400' :
                  'text-green-400'
                }`}>
                  {selectedNode.risk?.toUpperCase() || 'MODERATE'}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Confidence</p>
                <p className="text-white">{((selectedNode.confidence || 0.5) * 100).toFixed(0)}%</p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Analysis</p>
                <p className="text-slate-300 text-sm">{selectedNode.content}</p>
              </div>
            </div>
          ) : (
            <p className="text-slate-500 text-sm">Select a node to view details</p>
          )}
        </GlassCard>
      </div>
    </div>
  );
}
