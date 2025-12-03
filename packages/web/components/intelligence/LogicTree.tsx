'use client';

import { useState, useCallback, useMemo } from 'react';
import { tw, colors } from '@/lib/design-system';

// Node types for neuro-symbolic reasoning
type NodeType = 'neural' | 'symbolic' | 'input' | 'output';

interface LogicNode {
  id: string;
  type: NodeType;
  label: string;
  description?: string;
  confidence?: number; // 0-1 for neural nodes
  rule?: string; // For symbolic nodes
  value?: string | number | boolean;
  isActive: boolean; // Part of the active decision path
  children: string[]; // Child node IDs
}

interface LogicEdge {
  id: string;
  source: string;
  target: string;
  weight?: number; // Connection strength
  isActive: boolean;
}

interface LogicTreeData {
  nodes: LogicNode[];
  edges: LogicEdge[];
  finalDecision: string;
  overallConfidence: number;
}

interface LogicTreeProps {
  data: LogicTreeData;
  onNodeClick?: (node: LogicNode) => void;
  highlightPath?: string[]; // Node IDs to highlight
  showConfidence?: boolean;
}

// Node styling based on type
const nodeStyles: Record<NodeType, {
  bg: string;
  border: string;
  text: string;
  icon: string;
  shape: 'circle' | 'square' | 'diamond';
}> = {
  neural: {
    bg: 'bg-cyan-500/20',
    border: 'border-cyan-500/50',
    text: 'text-cyan-300',
    icon: '🧠',
    shape: 'circle',
  },
  symbolic: {
    bg: 'bg-amber-500/20',
    border: 'border-amber-500/50',
    text: 'text-amber-300',
    icon: '⚡',
    shape: 'square',
  },
  input: {
    bg: 'bg-slate-600/20',
    border: 'border-slate-500/50',
    text: 'text-slate-300',
    icon: '📥',
    shape: 'diamond',
  },
  output: {
    bg: 'bg-green-500/20',
    border: 'border-green-500/50',
    text: 'text-green-300',
    icon: '🎯',
    shape: 'circle',
  },
};

// Single logic node
function LogicNodeComponent({
  node,
  onClick,
  isHighlighted,
  showConfidence,
  level,
}: {
  node: LogicNode;
  onClick?: () => void;
  isHighlighted?: boolean;
  showConfidence?: boolean;
  level: number;
}) {
  const style = nodeStyles[node.type];

  return (
    <div
      onClick={onClick}
      className={`
        relative group cursor-pointer transition-all duration-200
        ${isHighlighted ? 'scale-105' : 'hover:scale-102'}
      `}
    >
      {/* Main node container */}
      <div
        className={`
          relative px-4 py-3 border-2 backdrop-blur-sm
          ${style.bg} ${style.border}
          ${style.shape === 'circle' ? 'rounded-full' : ''}
          ${style.shape === 'square' ? 'rounded-lg' : ''}
          ${style.shape === 'diamond' ? 'rounded-lg rotate-0' : ''}
          ${node.isActive ? 'ring-2 ring-white/30' : 'opacity-60'}
          ${isHighlighted ? 'ring-2 ring-white/50 shadow-lg' : ''}
        `}
        style={isHighlighted ? {
          boxShadow: node.type === 'neural'
            ? `0 0 20px ${colors.neural.glow}`
            : `0 0 20px ${colors.symbolic.glow}`,
        } : {}}
      >
        {/* Type indicator */}
        <div className="flex items-center gap-2">
          <span className="text-lg">{style.icon}</span>
          <div className="flex flex-col">
            <span className={`text-sm font-medium ${style.text}`}>
              {node.label}
            </span>
            {node.description && (
              <span className="text-xs text-slate-400">
                {node.description}
              </span>
            )}
          </div>
        </div>

        {/* Confidence gauge for neural nodes */}
        {showConfidence && node.type === 'neural' && node.confidence !== undefined && (
          <div className="mt-2 flex items-center gap-2">
            <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-cyan-400 transition-all duration-300"
                style={{ width: `${node.confidence * 100}%` }}
              />
            </div>
            <span className="text-xs text-cyan-400 font-mono">
              {(node.confidence * 100).toFixed(0)}%
            </span>
          </div>
        )}

        {/* Rule display for symbolic nodes */}
        {node.type === 'symbolic' && node.rule && (
          <div className="mt-2 px-2 py-1 bg-slate-800/50 rounded text-xs font-mono text-amber-200">
            {node.rule}
          </div>
        )}

        {/* Value display for input nodes */}
        {node.type === 'input' && node.value !== undefined && (
          <div className="mt-1 text-xs text-slate-400 font-mono">
            = {String(node.value)}
          </div>
        )}
      </div>

      {/* Hover tooltip */}
      <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
        <div className="px-3 py-2 bg-slate-800 rounded-lg border border-slate-700 shadow-xl text-xs max-w-xs">
          <div className="font-bold text-slate-200 mb-1">
            {node.type === 'neural' ? 'Neural Inference' : 'Symbolic Rule'}
          </div>
          <div className="text-slate-400">
            {node.description || node.rule || 'Click for details'}
          </div>
        </div>
      </div>
    </div>
  );
}

// Recursive tree renderer
function TreeLevel({
  nodes,
  nodeMap,
  edges,
  onNodeClick,
  highlightPath,
  showConfidence,
  level,
}: {
  nodes: LogicNode[];
  nodeMap: Map<string, LogicNode>;
  edges: LogicEdge[];
  onNodeClick?: (node: LogicNode) => void;
  highlightPath?: string[];
  showConfidence?: boolean;
  level: number;
}) {
  return (
    <div className="flex flex-col gap-4">
      {nodes.map(node => {
        const children = node.children
          .map(id => nodeMap.get(id))
          .filter(Boolean) as LogicNode[];

        return (
          <div key={node.id} className="flex flex-col md:flex-row items-start gap-3 md:gap-8">
            {/* Current node */}
            <LogicNodeComponent
              node={node}
              onClick={() => onNodeClick?.(node)}
              isHighlighted={highlightPath?.includes(node.id)}
              showConfidence={showConfidence}
              level={level}
            />

            {/* Connection line */}
            {children.length > 0 && (
              <div className="flex items-center">
                <div className={`
                  w-8 h-0.5
                  ${node.isActive ? 'bg-slate-500' : 'bg-slate-700'}
                `} />
                <div className="text-slate-500">→</div>
              </div>
            )}

            {/* Children */}
            {children.length > 0 && (
              <TreeLevel
                nodes={children}
                nodeMap={nodeMap}
                edges={edges}
                onNodeClick={onNodeClick}
                highlightPath={highlightPath}
                showConfidence={showConfidence}
                level={level + 1}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

export function LogicTree({
  data,
  onNodeClick,
  highlightPath,
  showConfidence = true,
}: LogicTreeProps) {
  const [selectedNode, setSelectedNode] = useState<LogicNode | null>(null);
  const [isCounterfactualMode, setIsCounterfactualMode] = useState(false);

  const nodeMap = useMemo(() => {
    return new Map(data.nodes.map(n => [n.id, n]));
  }, [data.nodes]);

  // Find root nodes (nodes with no incoming edges)
  const rootNodes = useMemo(() => {
    const targetIds = new Set(data.edges.map(e => e.target));
    return data.nodes.filter(n => !targetIds.has(n.id));
  }, [data.nodes, data.edges]);

  const handleNodeClick = useCallback((node: LogicNode) => {
    setSelectedNode(node);
    onNodeClick?.(node);
  }, [onNodeClick]);

  return (
    <div className="relative bg-slate-900/50 rounded-xl border border-slate-700/50 p-3 md:p-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4 md:mb-6">
        <div className="flex items-center gap-2 md:gap-3">
          <h3 className="text-base md:text-lg font-medium text-slate-200">Logic Tree</h3>
          <div className="px-2 py-1 rounded text-xs font-medium bg-green-500/20 text-green-400">
            {data.finalDecision}
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2 md:gap-3">
          {/* Counterfactual toggle - 44px touch target */}
          <button
            onClick={() => setIsCounterfactualMode(!isCounterfactualMode)}
            className={`
              min-h-[44px] px-3 md:px-4 rounded text-xs font-medium transition-colors
              ${isCounterfactualMode
                ? 'bg-orange-500/20 text-orange-400 border border-orange-500/50'
                : 'bg-slate-800 text-slate-400 border border-slate-700 active:border-slate-500'
              }
            `}
          >
            {isCounterfactualMode ? '↩ Exit' : '🔄 What If?'}
          </button>

          {/* Legend - hidden on smallest screens */}
          <div className="hidden sm:flex items-center gap-2 md:gap-4 text-xs">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full bg-cyan-500/50 border border-cyan-500" />
              <span className="text-slate-400">Neural</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded bg-amber-500/50 border border-amber-500" />
              <span className="text-slate-400">Symbolic</span>
            </div>
          </div>
        </div>
      </div>

      {/* Overall confidence */}
      <div className="mb-4 md:mb-6 flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4">
        <span className="text-xs sm:text-sm text-slate-400">Confidence:</span>
        <div className="flex items-center gap-2 flex-1">
          <div className="flex-1 max-w-xs h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-cyan-500 to-green-500 transition-all duration-500"
              style={{ width: `${data.overallConfidence * 100}%` }}
            />
          </div>
          <span className="text-sm font-mono text-slate-200">
            {(data.overallConfidence * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Tree visualization */}
      <div className="overflow-x-auto pb-4">
        <div className="min-w-max">
          <TreeLevel
            nodes={rootNodes}
            nodeMap={nodeMap}
            edges={data.edges}
            onNodeClick={handleNodeClick}
            highlightPath={highlightPath}
            showConfidence={showConfidence}
            level={0}
          />
        </div>
      </div>

      {/* Selected node details */}
      {selectedNode && (
        <div className="mt-6 p-4 bg-slate-800/50 rounded-lg border border-slate-700/50">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-lg">
                  {nodeStyles[selectedNode.type].icon}
                </span>
                <span className="font-medium text-slate-200">
                  {selectedNode.label}
                </span>
                <span className={`px-2 py-0.5 rounded text-xs ${
                  selectedNode.type === 'neural'
                    ? 'bg-cyan-500/20 text-cyan-400'
                    : 'bg-amber-500/20 text-amber-400'
                }`}>
                  {selectedNode.type}
                </span>
              </div>
              {selectedNode.description && (
                <p className="text-sm text-slate-400 mb-2">
                  {selectedNode.description}
                </p>
              )}
              {selectedNode.rule && (
                <div className="font-mono text-sm text-amber-200 bg-slate-900/50 px-3 py-2 rounded">
                  {selectedNode.rule}
                </div>
              )}
            </div>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-slate-500 hover:text-slate-300"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* Counterfactual mode overlay */}
      {isCounterfactualMode && (
        <div className="absolute inset-0 bg-orange-500/5 border-2 border-orange-500/30 rounded-xl pointer-events-none">
          <div className="absolute top-2 right-2 px-2 py-1 bg-orange-500/20 rounded text-xs text-orange-400">
            COUNTERFACTUAL MODE
          </div>
        </div>
      )}
    </div>
  );
}

// Example data
export const mockLogicTreeData: LogicTreeData = {
  nodes: [
    {
      id: 'input-1',
      type: 'input',
      label: 'Transaction Amount',
      value: '$47,500',
      isActive: true,
      children: ['symbolic-1'],
    },
    {
      id: 'input-2',
      type: 'input',
      label: 'Country Risk Score',
      value: 0.78,
      isActive: true,
      children: ['neural-1'],
    },
    {
      id: 'symbolic-1',
      type: 'symbolic',
      label: 'Threshold Check',
      rule: 'amount > $10,000 → FLAG',
      isActive: true,
      children: ['neural-2'],
    },
    {
      id: 'neural-1',
      type: 'neural',
      label: 'Behavior Pattern',
      description: 'Matches structuring pattern with 89% confidence',
      confidence: 0.89,
      isActive: true,
      children: ['neural-2'],
    },
    {
      id: 'neural-2',
      type: 'neural',
      label: 'Combined Risk Assessment',
      description: 'Aggregated risk from multiple signals',
      confidence: 0.92,
      isActive: true,
      children: ['output-1'],
    },
    {
      id: 'output-1',
      type: 'output',
      label: 'Final Decision',
      description: 'HIGH RISK - Recommend SAR Filing',
      isActive: true,
      children: [],
    },
  ],
  edges: [
    { id: 'e1', source: 'input-1', target: 'symbolic-1', isActive: true },
    { id: 'e2', source: 'input-2', target: 'neural-1', isActive: true },
    { id: 'e3', source: 'symbolic-1', target: 'neural-2', isActive: true },
    { id: 'e4', source: 'neural-1', target: 'neural-2', isActive: true },
    { id: 'e5', source: 'neural-2', target: 'output-1', isActive: true },
  ],
  finalDecision: 'HIGH RISK',
  overallConfidence: 0.92,
};
