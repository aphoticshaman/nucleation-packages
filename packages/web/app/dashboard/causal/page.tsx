'use client';

import { useState, useCallback } from 'react';
import { CausalGraph } from '@/components/causal/CausalGraph';
import type { CausalNode, CausalEdge } from '@/lib/types/causal';
import { Share2, X } from 'lucide-react';
import Link from 'next/link';

// Demo data - realistic geopolitical causal graph
const DEMO_NODES: CausalNode[] = [
  { id: 'n1', label: 'US_FED_RES', type: 'ACTOR', level: 'MACRO', beliefMass: 0.95, activity: 0.8 },
  { id: 'n2', label: 'PRC_PLAN', type: 'ACTOR', level: 'MACRO', beliefMass: 0.88, activity: 0.9 },
  { id: 'n3', label: 'TSMC_FAB', type: 'RESOURCE', level: 'MESO', beliefMass: 0.92, activity: 0.6 },
  { id: 'n4', label: 'STRAIT_BLOCKADE', type: 'EVENT', level: 'MACRO', beliefMass: 0.45, activity: 0.95 },
  { id: 'n5', label: 'GLOBAL_GPU_SUPPLY', type: 'RESOURCE', level: 'MESO', beliefMass: 0.85, activity: 0.7 },
  { id: 'n6', label: 'NVDA_STOCK', type: 'ACTOR', level: 'MICRO', beliefMass: 0.98, activity: 0.85 },
  { id: 'n7', label: 'EU_CHIPS_ACT', type: 'EVENT', level: 'MACRO', beliefMass: 0.75, activity: 0.4 },
  { id: 'n8', label: 'LITHIUM_SPOT', type: 'RESOURCE', level: 'MICRO', beliefMass: 0.82, activity: 0.5 },
  { id: 'n9', label: 'ASEAN_TRADE', type: 'ACTOR', level: 'MACRO', beliefMass: 0.70, activity: 0.3 },
  { id: 'n10', label: 'CYBER_SCADA', type: 'EVENT', level: 'MESO', beliefMass: 0.30, activity: 0.9 },
  { id: 'n11', label: 'UKRAINE_CONFLICT', type: 'EVENT', level: 'MACRO', beliefMass: 0.99, activity: 0.95 },
  { id: 'n12', label: 'NATO_EXPANSION', type: 'EVENT', level: 'MACRO', beliefMass: 0.85, activity: 0.6 },
  { id: 'n13', label: 'ENERGY_PRICES', type: 'RESOURCE', level: 'MESO', beliefMass: 0.92, activity: 0.8 },
  { id: 'n14', label: 'EUR_INFLATION', type: 'EVENT', level: 'MESO', beliefMass: 0.88, activity: 0.7 },
];

const DEMO_EDGES: CausalEdge[] = [
  { source: 'n2', target: 'n4', transferEntropy: 0.9, lag: 2, type: 'CAUSALITY' },
  { source: 'n4', target: 'n3', transferEntropy: 0.85, lag: 1, type: 'INFLUENCE' },
  { source: 'n3', target: 'n5', transferEntropy: 0.95, lag: 0, type: 'CAUSALITY' },
  { source: 'n5', target: 'n6', transferEntropy: 0.8, lag: 0, type: 'CORRELATION' },
  { source: 'n1', target: 'n6', transferEntropy: 0.6, lag: 1, type: 'INFLUENCE' },
  { source: 'n7', target: 'n3', transferEntropy: 0.4, lag: 5, type: 'INFLUENCE' },
  { source: 'n2', target: 'n9', transferEntropy: 0.5, lag: 3, type: 'INFLUENCE' },
  { source: 'n10', target: 'n3', transferEntropy: 0.7, lag: 0, type: 'CAUSALITY' },
  { source: 'n8', target: 'n5', transferEntropy: 0.65, lag: 2, type: 'CAUSALITY' },
  { source: 'n11', target: 'n12', transferEntropy: 0.75, lag: 1, type: 'CAUSALITY' },
  { source: 'n11', target: 'n13', transferEntropy: 0.88, lag: 0, type: 'CAUSALITY' },
  { source: 'n13', target: 'n14', transferEntropy: 0.82, lag: 1, type: 'CAUSALITY' },
  { source: 'n1', target: 'n14', transferEntropy: 0.55, lag: 2, type: 'INFLUENCE' },
  { source: 'n14', target: 'n9', transferEntropy: 0.45, lag: 3, type: 'CORRELATION' },
];

export default function CausalGraphPage() {
  const [selectedNode, setSelectedNode] = useState<CausalNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<CausalNode | null>(null);

  const handleNodeClick = useCallback((node: CausalNode) => {
    setSelectedNode(node);
  }, []);

  const handleNodeHover = useCallback((node: CausalNode | null) => {
    setHoveredNode(node);
  }, []);

  const closePanel = () => setSelectedNode(null);

  return (
    <div className="h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Header */}
      <header className="shrink-0 border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm z-20">
        <div className="px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-14">
            <div className="flex items-center gap-4">
              <Link href="/dashboard" className="text-slate-400 hover:text-white transition-colors text-sm">
                ← Dashboard
              </Link>
              <div className="h-4 w-px bg-slate-700" />
              <h1 className="text-lg font-mono font-bold flex items-center gap-2">
                <Share2 className="w-5 h-5 text-blue-400" />
                Causal Topology
              </h1>
            </div>
            <div className="flex items-center gap-4">
              <Link
                href="/dashboard/regimes"
                className="text-sm text-slate-400 hover:text-blue-400 transition-colors"
              >
                Regime Detection →
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 relative overflow-hidden">
        {/* Graph Canvas */}
        <CausalGraph
          nodes={DEMO_NODES}
          edges={DEMO_EDGES}
          onNodeClick={handleNodeClick}
          onNodeHover={handleNodeHover}
          className="absolute inset-0"
        />

        {/* Hover tooltip */}
        {hoveredNode && !selectedNode && (
          <div className="absolute bottom-4 left-4 bg-slate-900/90 backdrop-blur-sm border border-slate-700 rounded-lg p-3 pointer-events-none z-10 max-w-xs">
            <p className="text-sm font-mono font-bold text-blue-400">{hoveredNode.label}</p>
            <p className="text-xs text-slate-400 mt-1">
              {hoveredNode.type} • {hoveredNode.level}
            </p>
            <p className="text-xs text-slate-500 mt-1">
              Click for details
            </p>
          </div>
        )}

        {/* Selected Node Panel */}
        {selectedNode && (
          <div className="absolute top-4 right-4 bottom-4 w-80 bg-slate-900/95 backdrop-blur-sm border border-slate-700 rounded-lg overflow-hidden z-20 flex flex-col">
            {/* Panel Header */}
            <div className="p-4 border-b border-slate-700 flex items-center justify-between">
              <div>
                <p className="text-sm font-mono font-bold text-blue-400">{selectedNode.label}</p>
                <p className="text-xs text-slate-500">{selectedNode.type} • {selectedNode.level}</p>
              </div>
              <button
                onClick={closePanel}
                className="p-1 text-slate-400 hover:text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Panel Content */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {/* Metrics */}
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-slate-800/50 rounded-lg p-3">
                  <p className="text-[10px] font-mono text-slate-500 uppercase">Belief Mass</p>
                  <p className="text-lg font-mono font-bold text-cyan-400">
                    {(selectedNode.beliefMass * 100).toFixed(0)}%
                  </p>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-3">
                  <p className="text-[10px] font-mono text-slate-500 uppercase">Activity</p>
                  <p className="text-lg font-mono font-bold text-amber-400">
                    {(selectedNode.activity * 100).toFixed(0)}%
                  </p>
                </div>
              </div>

              {/* Connected Edges */}
              <div>
                <h4 className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider mb-2">
                  Outgoing Influences
                </h4>
                <div className="space-y-2">
                  {DEMO_EDGES.filter(e => e.source === selectedNode.id).map((edge, i) => {
                    const target = DEMO_NODES.find(n => n.id === edge.target);
                    return (
                      <div
                        key={i}
                        className="bg-slate-800/50 rounded p-2 flex items-center justify-between"
                      >
                        <span className="text-xs font-mono text-slate-300">
                          → {target?.label || edge.target}
                        </span>
                        <span className="text-[10px] font-mono text-cyan-400">
                          TE: {edge.transferEntropy.toFixed(2)}
                        </span>
                      </div>
                    );
                  })}
                  {DEMO_EDGES.filter(e => e.source === selectedNode.id).length === 0 && (
                    <p className="text-xs text-slate-500 italic">No outgoing edges</p>
                  )}
                </div>
              </div>

              <div>
                <h4 className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider mb-2">
                  Incoming Influences
                </h4>
                <div className="space-y-2">
                  {DEMO_EDGES.filter(e => e.target === selectedNode.id).map((edge, i) => {
                    const source = DEMO_NODES.find(n => n.id === edge.source);
                    return (
                      <div
                        key={i}
                        className="bg-slate-800/50 rounded p-2 flex items-center justify-between"
                      >
                        <span className="text-xs font-mono text-slate-300">
                          ← {source?.label || edge.source}
                        </span>
                        <span className="text-[10px] font-mono text-cyan-400">
                          TE: {edge.transferEntropy.toFixed(2)}
                        </span>
                      </div>
                    );
                  })}
                  {DEMO_EDGES.filter(e => e.target === selectedNode.id).length === 0 && (
                    <p className="text-xs text-slate-500 italic">No incoming edges</p>
                  )}
                </div>
              </div>

              {/* Description */}
              <div className="pt-4 border-t border-slate-700">
                <h4 className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider mb-2">
                  Description
                </h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  {selectedNode.type === 'ACTOR' && 'State or organizational actor with decision-making capability.'}
                  {selectedNode.type === 'EVENT' && 'Discrete occurrence that can trigger cascading effects.'}
                  {selectedNode.type === 'RESOURCE' && 'Physical or abstract resource affecting system dynamics.'}
                  {selectedNode.type === 'HYPOTHESIS' && 'Unconfirmed causal relationship under investigation.'}
                </p>
              </div>
            </div>

            {/* Panel Footer */}
            <div className="p-4 border-t border-slate-700">
              <button className="w-full py-2 bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 border border-blue-500/30 rounded-lg text-xs font-mono transition-colors">
                View Full Dossier
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Bottom Links */}
      <div className="shrink-0 border-t border-slate-800 bg-slate-900/50 backdrop-blur-sm px-4 py-2">
        <div className="flex items-center gap-4 text-xs font-mono text-slate-500">
          <Link href="/dashboard/regimes" className="hover:text-blue-400 transition-colors">
            Regime Detection
          </Link>
          <span>•</span>
          <Link href="/dashboard/intelligence" className="hover:text-blue-400 transition-colors">
            Intelligence Feed
          </Link>
          <span>•</span>
          <Link href="/dashboard" className="hover:text-blue-400 transition-colors">
            Dashboard Home
          </Link>
        </div>
      </div>
    </div>
  );
}
