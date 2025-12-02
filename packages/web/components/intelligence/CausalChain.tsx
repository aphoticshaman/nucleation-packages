'use client';

import { useState, useMemo, useCallback } from 'react';

interface CausalNode {
  id: string;
  label: string;
  domain: string;
  timestamp: number;
  riskScore: number;
  isNeural: boolean; // Neural inference vs Symbolic fact
  isCounterfactual?: boolean; // Removed in counterfactual mode
}

interface CausalEdge {
  id: string;
  source: string;
  target: string;
  weight: number; // Causal strength 0-1
  lagHours: number; // Time delay
  isActive: boolean;
}

interface CausalChainProps {
  nodes: CausalNode[];
  edges: CausalEdge[];
  onNodeClick?: (node: CausalNode) => void;
  onEdgeClick?: (edge: CausalEdge) => void;
  highlightPath?: string[]; // Node IDs to highlight
  counterfactualMode?: boolean;
  onCounterfactualToggle?: (nodeId: string) => void;
}

// Domain colors
const domainColors: Record<string, string> = {
  geopolitical: '#f59e0b',
  financial: '#22c55e',
  cyber: '#ef4444',
  defense: '#3b82f6',
  energy: '#a855f7',
  health: '#ec4899',
  tech: '#06b6d4',
  climate: '#84cc16',
};

// Component 22: Causal Chain Navigator
export function CausalChain({
  nodes,
  edges,
  onNodeClick,
  onEdgeClick,
  highlightPath = [],
  counterfactualMode = false,
  onCounterfactualToggle,
}: CausalChainProps) {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<string | null>(null);

  // Sort nodes by timestamp for left-to-right layout
  const sortedNodes = useMemo(() => {
    return [...nodes].sort((a, b) => a.timestamp - b.timestamp);
  }, [nodes]);

  // Calculate node positions
  const nodePositions = useMemo(() => {
    const positions: Record<string, { x: number; y: number; level: number }> = {};

    // Group by approximate time buckets
    const timeMin = Math.min(...sortedNodes.map(n => n.timestamp));
    const timeMax = Math.max(...sortedNodes.map(n => n.timestamp));
    const timeRange = timeMax - timeMin || 1;

    // Track vertical positions per column to avoid overlap
    const columnCounts: Record<number, number> = {};

    sortedNodes.forEach(node => {
      const xPct = ((node.timestamp - timeMin) / timeRange);
      const column = Math.floor(xPct * 10);

      columnCounts[column] = (columnCounts[column] || 0) + 1;
      const yOffset = columnCounts[column] - 1;

      positions[node.id] = {
        x: 80 + xPct * 700, // 80-780px range
        y: 80 + yOffset * 100, // Stack vertically
        level: column,
      };
    });

    return positions;
  }, [sortedNodes]);

  // Calculate SVG height based on node positions
  const svgHeight = useMemo(() => {
    const maxY = Math.max(...Object.values(nodePositions).map(p => p.y));
    return Math.max(300, maxY + 100);
  }, [nodePositions]);

  // Get connected nodes for a given node
  const getConnectedNodes = useCallback((nodeId: string) => {
    const connected = new Set<string>();
    edges.forEach(edge => {
      if (edge.source === nodeId) connected.add(edge.target);
      if (edge.target === nodeId) connected.add(edge.source);
    });
    return connected;
  }, [edges]);

  // Render curved edge path
  const renderEdgePath = (edge: CausalEdge) => {
    const sourcePos = nodePositions[edge.source];
    const targetPos = nodePositions[edge.target];
    if (!sourcePos || !targetPos) return null;

    const dx = targetPos.x - sourcePos.x;
    const dy = targetPos.y - sourcePos.y;

    // Bezier control points for curved line
    const cx1 = sourcePos.x + dx * 0.4;
    const cy1 = sourcePos.y;
    const cx2 = sourcePos.x + dx * 0.6;
    const cy2 = targetPos.y;

    const path = `M ${sourcePos.x + 40} ${sourcePos.y}
                  C ${cx1} ${cy1}, ${cx2} ${cy2}, ${targetPos.x - 40} ${targetPos.y}`;

    const isHighlighted = highlightPath.includes(edge.source) && highlightPath.includes(edge.target);
    const isHovered = hoveredEdge === edge.id ||
      hoveredNode === edge.source ||
      hoveredNode === edge.target;

    return (
      <g key={edge.id} className="group">
        {/* Edge path */}
        <path
          d={path}
          fill="none"
          stroke={isHighlighted ? '#06b6d4' : isHovered ? '#64748b' : '#334155'}
          strokeWidth={isHighlighted ? 3 : 2}
          strokeDasharray={edge.isActive ? 'none' : '5,5'}
          className="transition-all duration-200 cursor-pointer"
          onMouseEnter={() => setHoveredEdge(edge.id)}
          onMouseLeave={() => setHoveredEdge(null)}
          onClick={() => onEdgeClick?.(edge)}
          style={{
            filter: isHighlighted ? 'drop-shadow(0 0 8px rgba(6, 182, 212, 0.5))' : 'none',
          }}
        />

        {/* Arrow head */}
        <polygon
          points={`${targetPos.x - 40},${targetPos.y} ${targetPos.x - 50},${targetPos.y - 5} ${targetPos.x - 50},${targetPos.y + 5}`}
          fill={isHighlighted ? '#06b6d4' : '#475569'}
          className="transition-all duration-200"
        />

        {/* Edge label (time lag) */}
        <text
          x={(sourcePos.x + targetPos.x) / 2}
          y={(sourcePos.y + targetPos.y) / 2 - 10}
          textAnchor="middle"
          className="text-xs fill-slate-500 pointer-events-none"
        >
          +{edge.lagHours}h
        </text>

        {/* Weight indicator */}
        <text
          x={(sourcePos.x + targetPos.x) / 2}
          y={(sourcePos.y + targetPos.y) / 2 + 5}
          textAnchor="middle"
          className="text-xs fill-slate-600 font-mono pointer-events-none"
        >
          {(edge.weight * 100).toFixed(0)}%
        </text>
      </g>
    );
  };

  // Render node
  const renderNode = (node: CausalNode) => {
    const pos = nodePositions[node.id];
    if (!pos) return null;

    const color = domainColors[node.domain] || '#64748b';
    const isHighlighted = highlightPath.includes(node.id);
    const isHovered = hoveredNode === node.id;
    const isConnectedToHovered = hoveredNode ? getConnectedNodes(hoveredNode).has(node.id) : false;
    const isRemoved = counterfactualMode && node.isCounterfactual;

    return (
      <g
        key={node.id}
        transform={`translate(${pos.x}, ${pos.y})`}
        className={`cursor-pointer transition-all duration-200 ${isRemoved ? 'opacity-30' : ''}`}
        onMouseEnter={() => setHoveredNode(node.id)}
        onMouseLeave={() => setHoveredNode(null)}
        onClick={() => onNodeClick?.(node)}
      >
        {/* Glow effect for highlighted nodes */}
        {(isHighlighted || isHovered) && (
          <circle
            r={50}
            fill="none"
            stroke={color}
            strokeWidth={2}
            opacity={0.3}
            className="animate-ping"
          />
        )}

        {/* Node shape: Circle for Neural, Square for Symbolic */}
        {node.isNeural ? (
          <circle
            r={35}
            fill={`${color}20`}
            stroke={isHighlighted ? '#06b6d4' : color}
            strokeWidth={isHighlighted ? 3 : 2}
            style={{
              filter: isHighlighted ? `drop-shadow(0 0 10px ${color})` : 'none',
            }}
          />
        ) : (
          <rect
            x={-35}
            y={-35}
            width={70}
            height={70}
            rx={8}
            fill={`${color}20`}
            stroke={isHighlighted ? '#06b6d4' : color}
            strokeWidth={isHighlighted ? 3 : 2}
            style={{
              filter: isHighlighted ? `drop-shadow(0 0 10px ${color})` : 'none',
            }}
          />
        )}

        {/* Type indicator */}
        <circle
          cx={25}
          cy={-25}
          r={8}
          fill={node.isNeural ? '#06b6d4' : '#f59e0b'}
        />

        {/* Node label */}
        <text
          textAnchor="middle"
          dy={5}
          className="text-xs fill-slate-200 font-medium pointer-events-none"
        >
          {node.label.length > 12 ? node.label.slice(0, 12) + '...' : node.label}
        </text>

        {/* Domain label */}
        <text
          textAnchor="middle"
          dy={20}
          className="text-[10px] fill-slate-500 uppercase pointer-events-none"
        >
          {node.domain}
        </text>

        {/* Counterfactual toggle */}
        {counterfactualMode && (
          <g
            transform="translate(30, 25)"
            onClick={(e) => {
              e.stopPropagation();
              onCounterfactualToggle?.(node.id);
            }}
            className="cursor-pointer"
          >
            <circle r={10} fill="#1e293b" stroke="#475569" />
            <text
              textAnchor="middle"
              dy={4}
              className="text-xs fill-slate-400"
            >
              {isRemoved ? '↩' : '✕'}
            </text>
          </g>
        )}

        {/* Strikethrough for removed nodes */}
        {isRemoved && (
          <line
            x1={-40}
            y1={0}
            x2={40}
            y2={0}
            stroke="#ef4444"
            strokeWidth={3}
          />
        )}
      </g>
    );
  };

  return (
    <div className="bg-slate-900/50 rounded-xl border border-slate-800 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-medium text-slate-300">Causal Chain Navigator</h3>
          <span className="text-xs text-slate-500">
            {nodes.length} events • {edges.length} connections
          </span>
        </div>

        <div className="flex items-center gap-4">
          {/* Legend */}
          <div className="flex items-center gap-3 text-xs">
            <div className="flex items-center gap-1">
              <div className="w-4 h-4 rounded-full border-2 border-cyan-500" />
              <span className="text-slate-500">Neural</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-4 h-4 rounded border-2 border-amber-500" />
              <span className="text-slate-500">Symbolic</span>
            </div>
          </div>

          {/* Counterfactual mode indicator */}
          {counterfactualMode && (
            <div className="px-2 py-1 bg-orange-500/20 border border-orange-500/50 rounded text-xs text-orange-400">
              COUNTERFACTUAL MODE
            </div>
          )}
        </div>
      </div>

      {/* SVG Canvas */}
      <div className="overflow-x-auto">
        <svg
          width={900}
          height={svgHeight}
          className="min-w-full"
        >
          {/* Grid background */}
          <defs>
            <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
              <path
                d="M 50 0 L 0 0 0 50"
                fill="none"
                stroke="#1e293b"
                strokeWidth="0.5"
              />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />

          {/* Timeline axis */}
          <line
            x1={60}
            y1={svgHeight - 30}
            x2={840}
            y2={svgHeight - 30}
            stroke="#334155"
            strokeWidth={1}
          />
          <text
            x={450}
            y={svgHeight - 10}
            textAnchor="middle"
            className="text-xs fill-slate-500"
          >
            TIME →
          </text>

          {/* Render edges first (below nodes) */}
          <g>{edges.map(renderEdgePath)}</g>

          {/* Render nodes */}
          <g>{sortedNodes.map(renderNode)}</g>
        </svg>
      </div>

      {/* Footer with stats */}
      <div className="px-4 py-2 border-t border-slate-800 flex items-center justify-between text-xs text-slate-500">
        <span>
          Click nodes to inspect • Hover for connections
        </span>
        <span>
          Avg cascade time: {edges.length > 0
            ? (edges.reduce((a, e) => a + e.lagHours, 0) / edges.length).toFixed(1)
            : 0}h
        </span>
      </div>
    </div>
  );
}

// Example data
export const mockCausalNodes: CausalNode[] = [
  { id: '1', label: 'Oil Price Spike', domain: 'energy', timestamp: Date.now() - 72 * 3600000, riskScore: 0.7, isNeural: false },
  { id: '2', label: 'Inflation Warning', domain: 'financial', timestamp: Date.now() - 48 * 3600000, riskScore: 0.6, isNeural: true },
  { id: '3', label: 'Supply Shortage', domain: 'tech', timestamp: Date.now() - 36 * 3600000, riskScore: 0.5, isNeural: true },
  { id: '4', label: 'Market Volatility', domain: 'financial', timestamp: Date.now() - 24 * 3600000, riskScore: 0.8, isNeural: true },
  { id: '5', label: 'Diplomatic Tensions', domain: 'geopolitical', timestamp: Date.now() - 12 * 3600000, riskScore: 0.75, isNeural: false },
  { id: '6', label: 'Defense Alert', domain: 'defense', timestamp: Date.now(), riskScore: 0.85, isNeural: true },
];

export const mockCausalEdges: CausalEdge[] = [
  { id: 'e1', source: '1', target: '2', weight: 0.85, lagHours: 24, isActive: true },
  { id: 'e2', source: '1', target: '3', weight: 0.65, lagHours: 36, isActive: true },
  { id: 'e3', source: '2', target: '4', weight: 0.78, lagHours: 24, isActive: true },
  { id: 'e4', source: '3', target: '4', weight: 0.55, lagHours: 12, isActive: true },
  { id: 'e5', source: '4', target: '5', weight: 0.7, lagHours: 12, isActive: true },
  { id: 'e6', source: '5', target: '6', weight: 0.82, lagHours: 12, isActive: true },
];
