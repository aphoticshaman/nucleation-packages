'use client';

import { useState, useRef, useEffect, useCallback, useMemo } from 'react';

interface NetworkNode {
  id: string;
  label: string;
  type: 'entity' | 'event' | 'location' | 'organization' | 'concept';
  risk?: number; // 0-1
  weight?: number; // node importance
  x?: number;
  y?: number;
}

interface NetworkEdge {
  source: string;
  target: string;
  weight?: number; // edge strength
  type?: 'causal' | 'correlation' | 'membership' | 'temporal';
  label?: string;
}

interface NetworkGraphProps {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  width?: number;
  height?: number;
  onNodeClick?: (node: NetworkNode) => void;
  onEdgeClick?: (edge: NetworkEdge) => void;
  highlightedNodeId?: string | null;
  interactive?: boolean;
}

// Component 19: Force-Directed Network Graph
export function NetworkGraph({
  nodes,
  edges,
  width = 600,
  height = 400,
  onNodeClick,
  onEdgeClick,
  highlightedNodeId,
  interactive = true,
}: NetworkGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [positions, setPositions] = useState<Record<string, { x: number; y: number }>>({});
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [draggedNode, setDraggedNode] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });

  // Initialize positions with force-directed layout simulation
  useEffect(() => {
    if (nodes.length === 0) return;

    // Initialize random positions
    const newPositions: Record<string, { x: number; y: number }> = {};
    nodes.forEach((node) => {
      newPositions[node.id] = {
        x: node.x ?? width / 2 + (Math.random() - 0.5) * width * 0.6,
        y: node.y ?? height / 2 + (Math.random() - 0.5) * height * 0.6,
      };
    });

    // Simple force-directed layout (100 iterations)
    const iterations = 100;
    const repulsionStrength = 5000;
    const attractionStrength = 0.01;
    const centerGravity = 0.02;

    for (let i = 0; i < iterations; i++) {
      const forces: Record<string, { x: number; y: number }> = {};
      nodes.forEach((n) => {
        forces[n.id] = { x: 0, y: 0 };
      });

      // Repulsion between all nodes
      for (let j = 0; j < nodes.length; j++) {
        for (let k = j + 1; k < nodes.length; k++) {
          const n1 = nodes[j];
          const n2 = nodes[k];
          const p1 = newPositions[n1.id];
          const p2 = newPositions[n2.id];

          const dx = p2.x - p1.x;
          const dy = p2.y - p1.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;

          const force = repulsionStrength / (dist * dist);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;

          forces[n1.id].x -= fx;
          forces[n1.id].y -= fy;
          forces[n2.id].x += fx;
          forces[n2.id].y += fy;
        }
      }

      // Attraction along edges
      edges.forEach((edge) => {
        const p1 = newPositions[edge.source];
        const p2 = newPositions[edge.target];
        if (!p1 || !p2) return;

        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        const force = dist * attractionStrength * (edge.weight ?? 1);
        const fx = dx * force;
        const fy = dy * force;

        forces[edge.source].x += fx;
        forces[edge.source].y += fy;
        forces[edge.target].x -= fx;
        forces[edge.target].y -= fy;
      });

      // Center gravity
      nodes.forEach((node) => {
        const p = newPositions[node.id];
        forces[node.id].x += (width / 2 - p.x) * centerGravity;
        forces[node.id].y += (height / 2 - p.y) * centerGravity;
      });

      // Apply forces with cooling
      const cooling = 1 - i / iterations;
      nodes.forEach((node) => {
        const p = newPositions[node.id];
        p.x += forces[node.id].x * cooling;
        p.y += forces[node.id].y * cooling;

        // Keep in bounds
        const padding = 30;
        p.x = Math.max(padding, Math.min(width - padding, p.x));
        p.y = Math.max(padding, Math.min(height - padding, p.y));
      });
    }

    setPositions(newPositions);
  }, [nodes, edges, width, height]);

  // Node type colors
  const getNodeColor = useCallback((node: NetworkNode) => {
    if (node.risk !== undefined && node.risk >= 0.7) return '#ef4444'; // red

    const typeColors = {
      entity: '#06b6d4',       // cyan
      event: '#f59e0b',        // amber
      location: '#22c55e',     // green
      organization: '#8b5cf6', // purple
      concept: '#6b7280',      // gray
    };
    return typeColors[node.type] || '#6b7280';
  }, []);

  // Edge type styles
  const getEdgeStyle = useCallback((edge: NetworkEdge) => {
    const styles = {
      causal: { stroke: '#f59e0b', dashArray: 'none' },
      correlation: { stroke: '#06b6d4', dashArray: '5,5' },
      membership: { stroke: '#8b5cf6', dashArray: 'none' },
      temporal: { stroke: '#6b7280', dashArray: '2,2' },
    };
    return styles[edge.type || 'correlation'];
  }, []);

  // Drag handlers
  const handleMouseDown = useCallback((e: React.MouseEvent, nodeId: string) => {
    if (!interactive) return;
    const svg = svgRef.current;
    if (!svg) return;

    const rect = svg.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setDraggedNode(nodeId);
    setDragOffset({
      x: x - (positions[nodeId]?.x ?? 0),
      y: y - (positions[nodeId]?.y ?? 0),
    });
  }, [interactive, positions]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!draggedNode) return;

    const svg = svgRef.current;
    if (!svg) return;

    const rect = svg.getBoundingClientRect();
    const x = e.clientX - rect.left - dragOffset.x;
    const y = e.clientY - rect.top - dragOffset.y;

    setPositions((prev) => ({
      ...prev,
      [draggedNode]: {
        x: Math.max(20, Math.min(width - 20, x)),
        y: Math.max(20, Math.min(height - 20, y)),
      },
    }));
  }, [draggedNode, dragOffset, width, height]);

  const handleMouseUp = useCallback(() => {
    setDraggedNode(null);
  }, []);

  // Connected nodes for highlighting
  const connectedNodes = useMemo(() => {
    const target = hoveredNode || highlightedNodeId;
    if (!target) return new Set<string>();

    const connected = new Set<string>([target]);
    edges.forEach((edge) => {
      if (edge.source === target) connected.add(edge.target);
      if (edge.target === target) connected.add(edge.source);
    });
    return connected;
  }, [hoveredNode, highlightedNodeId, edges]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      className="bg-slate-900/50 rounded-lg border border-slate-700"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Gradient definitions */}
      <defs>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Edges */}
      <g>
        {edges.map((edge, i) => {
          const source = positions[edge.source];
          const target = positions[edge.target];
          if (!source || !target) return null;

          const style = getEdgeStyle(edge);
          const isHighlighted = connectedNodes.has(edge.source) && connectedNodes.has(edge.target);
          const opacity = connectedNodes.size > 0 ? (isHighlighted ? 1 : 0.15) : 0.6;

          return (
            <g key={i}>
              <line
                x1={source.x}
                y1={source.y}
                x2={target.x}
                y2={target.y}
                stroke={style.stroke}
                strokeWidth={isHighlighted ? 2 : 1}
                strokeDasharray={style.dashArray}
                opacity={opacity}
                className="transition-all duration-200 cursor-pointer"
                onClick={() => onEdgeClick?.(edge)}
              />
              {edge.label && isHighlighted && (
                <text
                  x={(source.x + target.x) / 2}
                  y={(source.y + target.y) / 2 - 5}
                  textAnchor="middle"
                  className="text-xs fill-slate-400"
                >
                  {edge.label}
                </text>
              )}
            </g>
          );
        })}
      </g>

      {/* Nodes */}
      <g>
        {nodes.map((node) => {
          const pos = positions[node.id];
          if (!pos) return null;

          const color = getNodeColor(node);
          const size = 8 + (node.weight ?? 1) * 4;
          const isHighlighted = connectedNodes.size === 0 || connectedNodes.has(node.id);
          const isHovered = hoveredNode === node.id || highlightedNodeId === node.id;

          return (
            <g
              key={node.id}
              transform={`translate(${pos.x}, ${pos.y})`}
              className={`transition-all duration-200 ${interactive ? 'cursor-pointer' : ''}`}
              onMouseEnter={() => setHoveredNode(node.id)}
              onMouseLeave={() => setHoveredNode(null)}
              onMouseDown={(e) => handleMouseDown(e, node.id)}
              onClick={() => onNodeClick?.(node)}
            >
              {/* Glow effect for highlighted nodes */}
              {isHovered && (
                <circle
                  r={size + 6}
                  fill={color}
                  opacity={0.3}
                  filter="url(#glow)"
                />
              )}

              {/* Main node */}
              <circle
                r={size}
                fill={color}
                opacity={isHighlighted ? 1 : 0.3}
                stroke={isHovered ? '#fff' : 'none'}
                strokeWidth={2}
              />

              {/* Risk ring */}
              {node.risk !== undefined && node.risk > 0.5 && (
                <circle
                  r={size + 3}
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth={1}
                  strokeDasharray={`${node.risk * 20} ${20 - node.risk * 20}`}
                  opacity={isHighlighted ? 0.8 : 0.2}
                />
              )}

              {/* Label */}
              {(isHovered || isHighlighted) && (
                <text
                  y={size + 14}
                  textAnchor="middle"
                  className="text-xs fill-slate-300"
                  style={{ pointerEvents: 'none' }}
                >
                  {node.label}
                </text>
              )}
            </g>
          );
        })}
      </g>

      {/* Legend */}
      <g transform={`translate(${width - 100}, 10)`}>
        <rect x={-5} y={-5} width={95} height={80} fill="rgba(15,23,42,0.8)" rx={4} />
        {[
          { type: 'entity', label: 'Entity', color: '#06b6d4' },
          { type: 'event', label: 'Event', color: '#f59e0b' },
          { type: 'location', label: 'Location', color: '#22c55e' },
          { type: 'organization', label: 'Org', color: '#8b5cf6' },
        ].map((item, i) => (
          <g key={item.type} transform={`translate(5, ${i * 18 + 8})`}>
            <circle r={4} fill={item.color} />
            <text x={12} y={4} className="text-xs fill-slate-400">{item.label}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}

// Mock data for demo
export const mockNetworkData = {
  nodes: [
    { id: 'russia', label: 'Russia', type: 'entity' as const, weight: 2, risk: 0.85 },
    { id: 'ukraine', label: 'Ukraine', type: 'location' as const, weight: 2, risk: 0.92 },
    { id: 'nato', label: 'NATO', type: 'organization' as const, weight: 1.5 },
    { id: 'energy', label: 'Energy Crisis', type: 'event' as const, weight: 1.5, risk: 0.7 },
    { id: 'sanctions', label: 'Sanctions', type: 'concept' as const, weight: 1 },
    { id: 'china', label: 'China', type: 'entity' as const, weight: 1.5 },
    { id: 'taiwan', label: 'Taiwan', type: 'location' as const, risk: 0.65 },
  ],
  edges: [
    { source: 'russia', target: 'ukraine', type: 'causal' as const, label: 'invasion' },
    { source: 'russia', target: 'energy', type: 'causal' as const },
    { source: 'nato', target: 'ukraine', type: 'membership' as const },
    { source: 'sanctions', target: 'russia', type: 'causal' as const },
    { source: 'china', target: 'russia', type: 'correlation' as const },
    { source: 'china', target: 'taiwan', type: 'causal' as const, weight: 0.8 },
    { source: 'energy', target: 'sanctions', type: 'temporal' as const },
  ],
};
