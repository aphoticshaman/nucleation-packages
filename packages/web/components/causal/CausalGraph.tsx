'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Layers, ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';
import type { CausalNode, CausalEdge } from '@/lib/types/causal';

interface CausalGraphProps {
  nodes: CausalNode[];
  edges: CausalEdge[];
  onNodeClick?: (node: CausalNode) => void;
  onNodeHover?: (node: CausalNode | null) => void;
  className?: string;
}

type SemanticLevel = 'ALL' | 'MACRO' | 'MESO' | 'MICRO';

/**
 * CausalGraph - Force-directed causal topology visualization
 *
 * Features:
 * - Force-directed physics (repulsion + spring + center gravity)
 * - Transfer entropy-weighted edges (thickness + opacity)
 * - Dempster-Shafer belief mass halos
 * - Semantic zoom (MACRO/MESO/MICRO)
 * - Flow particles showing information transfer direction
 */
export function CausalGraph({
  nodes: initialNodes,
  edges,
  onNodeClick,
  onNodeHover,
  className = '',
}: CausalGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [nodes, setNodes] = useState<CausalNode[]>([]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [activeLevel, setActiveLevel] = useState<SemanticLevel>('ALL');
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const animationRef = useRef<number>(0);
  const timeRef = useRef(0);

  // Initialize nodes with random positions
  useEffect(() => {
    const width = containerRef.current?.offsetWidth || 800;
    const height = containerRef.current?.offsetHeight || 600;

    const initialized = initialNodes.map((n) => ({
      ...n,
      x: n.x ?? Math.random() * width,
      y: n.y ?? Math.random() * height,
      vx: n.vx ?? 0,
      vy: n.vy ?? 0,
    }));
    setNodes(initialized);
  }, [initialNodes]);

  // Filter nodes based on semantic level
  const visibleNodes = nodes.filter((n) => {
    if (activeLevel === 'ALL') return true;
    if (activeLevel === 'MACRO') return n.level === 'MACRO';
    if (activeLevel === 'MESO') return n.level === 'MACRO' || n.level === 'MESO';
    return true;
  });

  const visibleEdges = edges.filter((e) => {
    const source = visibleNodes.find((n) => n.id === e.source);
    const target = visibleNodes.find((n) => n.id === e.target);
    return source && target;
  });

  // Get node color based on type
  const getNodeColor = useCallback((type: CausalNode['type']): string => {
    switch (type) {
      case 'ACTOR':
        return '59, 130, 246'; // Blue
      case 'EVENT':
        return '239, 68, 68'; // Red
      case 'RESOURCE':
        return '34, 211, 238'; // Cyan
      case 'HYPOTHESIS':
        return '168, 85, 247'; // Purple
      default:
        return '148, 163, 184'; // Slate
    }
  }, []);

  // Physics simulation and rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const render = () => {
      timeRef.current += 0.01;
      const time = timeRef.current;

      // Resize canvas to container
      const rect = container.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      const width = canvas.width;
      const height = canvas.height;

      // Clear with background
      ctx.fillStyle = '#020617';
      ctx.fillRect(0, 0, width, height);

      // Draw grid
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      ctx.beginPath();
      const gridSize = activeLevel === 'MACRO' ? 80 : 40;
      for (let x = 0; x < width; x += gridSize) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
      }
      for (let y = 0; y < height; y += gridSize) {
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
      }
      ctx.stroke();

      // Physics constants (scale with semantic level)
      const kRepel = activeLevel === 'MACRO' ? 3000 : 2000;
      const kSpring = 0.05;
      const damping = 0.85;
      const centerForce = 0.005;

      // Calculate forces
      visibleNodes.forEach((node) => {
        if (node.x === undefined || node.y === undefined) return;
        let fx = 0,
          fy = 0;

        // Repulsion from other nodes (Coulomb-like)
        visibleNodes.forEach((other) => {
          if (node.id === other.id || other.x === undefined || other.y === undefined) return;
          const dx = node.x! - other.x!;
          const dy = node.y! - other.y!;
          const distSq = dx * dx + dy * dy;
          if (distSq > 0) {
            const f = kRepel / Math.sqrt(distSq);
            const angle = Math.atan2(dy, dx);
            fx += Math.cos(angle) * f;
            fy += Math.sin(angle) * f;
          }
        });

        // Center gravity
        const dx = width / 2 - node.x!;
        const dy = height / 2 - node.y!;
        fx += dx * centerForce;
        fy += dy * centerForce;

        // Apply forces
        node.vx = (node.vx || 0) + fx * 0.01;
        node.vy = (node.vy || 0) + fy * 0.01;
      });

      // Spring forces from edges (Hooke's law)
      visibleEdges.forEach((edge) => {
        const source = nodes.find((n) => n.id === edge.source);
        const target = nodes.find((n) => n.id === edge.target);
        if (!source || !target) return;
        if (source.x === undefined || target.x === undefined) return;

        const dx = target.x! - source.x!;
        const dy = target.y! - source.y!;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const restLength = activeLevel === 'MACRO' ? 250 : 150;
        const force = (dist - restLength) * kSpring;

        const angle = Math.atan2(dy, dx);
        const fx = Math.cos(angle) * force;
        const fy = Math.sin(angle) * force;

        source.vx = (source.vx || 0) + fx;
        source.vy = (source.vy || 0) + fy;
        target.vx = (target.vx || 0) - fx;
        target.vy = (target.vy || 0) - fy;
      });

      // Update positions with damping
      visibleNodes.forEach((node) => {
        if (selectedNode === node.id) return; // Don't move selected node

        node.vx = (node.vx || 0) * damping;
        node.vy = (node.vy || 0) * damping;
        node.x = (node.x || 0) + (node.vx || 0);
        node.y = (node.y || 0) + (node.vy || 0);

        // Bounds
        node.x = Math.max(40, Math.min(width - 40, node.x));
        node.y = Math.max(40, Math.min(height - 40, node.y));
      });

      // Apply zoom and pan transform
      ctx.save();
      ctx.translate(pan.x, pan.y);
      ctx.scale(zoom, zoom);

      // Render edges with Transfer Entropy weighting
      visibleEdges.forEach((edge) => {
        const source = nodes.find((n) => n.id === edge.source);
        const target = nodes.find((n) => n.id === edge.target);
        if (!source || !target) return;
        if (source.x === undefined || target.x === undefined) return;

        ctx.beginPath();
        ctx.moveTo(source.x!, source.y!);
        ctx.lineTo(target.x!, target.y!);

        // Style based on transfer entropy magnitude
        const opacity = 0.2 + edge.transferEntropy * 0.8;
        const isCausality = edge.type === 'CAUSALITY';
        ctx.strokeStyle = isCausality
          ? `rgba(34, 211, 238, ${opacity})`
          : `rgba(148, 163, 184, ${opacity})`;
        ctx.lineWidth = 1 + edge.transferEntropy * 3;
        ctx.setLineDash(isCausality ? [] : [4, 4]);
        ctx.stroke();

        // Flow particle (information transfer animation)
        const flowSpeed = time * (0.5 + edge.transferEntropy);
        const flowPos = flowSpeed % 1;
        const px = source.x! + (target.x! - source.x!) * flowPos;
        const py = source.y! + (target.y! - source.y!) * flowPos;

        ctx.beginPath();
        ctx.arc(px, py, 2, 0, Math.PI * 2);
        ctx.fillStyle = '#fff';
        ctx.fill();
      });
      ctx.setLineDash([]);

      // Render nodes with belief mass halos
      visibleNodes.forEach((node) => {
        if (node.x === undefined || node.y === undefined) return;

        const baseRadius =
          node.level === 'MACRO' ? 18 : node.level === 'MESO' ? 12 : 6;
        const pulse = Math.sin(time * 3 + node.x * 0.1) * 0.1 + 1;
        const color = getNodeColor(node.type);
        const isHovered = hoveredNode === node.id;
        const isSelected = selectedNode === node.id;

        // Belief mass halo (Dempster-Shafer uncertainty visualization)
        const haloRadius = baseRadius * 3 * node.beliefMass * pulse;
        const grad = ctx.createRadialGradient(
          node.x,
          node.y,
          baseRadius,
          node.x,
          node.y,
          haloRadius
        );
        grad.addColorStop(0, `rgba(${color}, 0.4)`);
        grad.addColorStop(1, `rgba(${color}, 0)`);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(node.x, node.y, haloRadius, 0, Math.PI * 2);
        ctx.fill();

        // Core node
        ctx.beginPath();
        ctx.arc(node.x, node.y, baseRadius * (isHovered ? 1.2 : 1), 0, Math.PI * 2);
        ctx.fillStyle = `rgb(${color})`;
        ctx.fill();
        ctx.strokeStyle = isSelected ? '#fff' : isHovered ? '#94a3b8' : '#fff';
        ctx.lineWidth = isSelected ? 3 : isHovered ? 2 : 1.5;
        ctx.stroke();

        // Label (progressive disclosure)
        if (node.level !== 'MICRO' || visibleNodes.length < 15 || isHovered) {
          ctx.fillStyle = '#e2e8f0';
          ctx.font =
            node.level === 'MACRO'
              ? 'bold 12px ui-monospace, monospace'
              : '10px ui-monospace, monospace';
          ctx.fillText(node.label, node.x + baseRadius + 8, node.y + 4);

          // Belief mass bar
          ctx.fillStyle = '#1e293b';
          ctx.fillRect(node.x + baseRadius + 8, node.y + 8, 30, 3);
          ctx.fillStyle = `rgb(${color})`;
          ctx.fillRect(node.x + baseRadius + 8, node.y + 8, 30 * node.beliefMass, 3);
        }
      });

      ctx.restore();

      animationRef.current = requestAnimationFrame(render);
    };

    render();

    return () => cancelAnimationFrame(animationRef.current);
  }, [
    nodes,
    edges,
    visibleNodes,
    visibleEdges,
    selectedNode,
    hoveredNode,
    activeLevel,
    zoom,
    pan,
    getNodeColor,
  ]);

  // Mouse interaction handlers
  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left - pan.x) / zoom;
      const y = (e.clientY - rect.top - pan.y) / zoom;

      // Find clicked node
      for (const node of visibleNodes) {
        if (node.x === undefined || node.y === undefined) continue;
        const dx = x - node.x;
        const dy = y - node.y;
        const radius = node.level === 'MACRO' ? 18 : node.level === 'MESO' ? 12 : 6;
        if (dx * dx + dy * dy < radius * radius * 4) {
          setSelectedNode(node.id);
          onNodeClick?.(node);
          return;
        }
      }
      setSelectedNode(null);
    },
    [visibleNodes, zoom, pan, onNodeClick]
  );

  const handleCanvasMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left - pan.x) / zoom;
      const y = (e.clientY - rect.top - pan.y) / zoom;

      // Find hovered node
      for (const node of visibleNodes) {
        if (node.x === undefined || node.y === undefined) continue;
        const dx = x - node.x;
        const dy = y - node.y;
        const radius = node.level === 'MACRO' ? 18 : node.level === 'MESO' ? 12 : 6;
        if (dx * dx + dy * dy < radius * radius * 4) {
          if (hoveredNode !== node.id) {
            setHoveredNode(node.id);
            onNodeHover?.(node);
          }
          return;
        }
      }
      if (hoveredNode) {
        setHoveredNode(null);
        onNodeHover?.(null);
      }
    },
    [visibleNodes, zoom, pan, hoveredNode, onNodeHover]
  );

  const handleReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setSelectedNode(null);
  };

  return (
    <div ref={containerRef} className={`relative w-full h-full bg-slate-950 overflow-hidden ${className}`}>
      <canvas
        ref={canvasRef}
        className="block w-full h-full cursor-crosshair"
        onClick={handleCanvasClick}
        onMouseMove={handleCanvasMouseMove}
        onMouseLeave={() => {
          setHoveredNode(null);
          onNodeHover?.(null);
        }}
      />

      {/* Semantic Depth Controls */}
      <div className="absolute top-4 right-4 flex flex-col gap-2">
        <div className="bg-slate-900/80 backdrop-blur-sm border border-slate-700 p-2 rounded-lg flex flex-col gap-2">
          <div className="text-[10px] font-mono text-slate-500 text-center uppercase tracking-wider mb-1">
            Semantic Depth
          </div>
          {(['MACRO', 'MESO', 'ALL'] as const).map((level) => (
            <button
              key={level}
              onClick={() => setActiveLevel(level)}
              className={`p-2 rounded transition-colors flex items-center justify-between gap-2 text-[10px] font-mono border ${
                activeLevel === level
                  ? 'bg-blue-500/20 border-blue-500 text-blue-400'
                  : 'hover:bg-slate-800 border-transparent text-slate-400'
              }`}
            >
              <Layers size={14} />
              {level === 'MACRO' ? 'STRATEGIC' : level === 'MESO' ? 'OPERATIONAL' : 'TACTICAL'}
            </button>
          ))}
        </div>

        {/* Zoom Controls */}
        <div className="bg-slate-900/80 backdrop-blur-sm border border-slate-700 p-2 rounded-lg flex flex-col gap-1">
          <button
            onClick={() => setZoom((z) => Math.min(z * 1.2, 3))}
            className="p-2 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200"
          >
            <ZoomIn size={16} />
          </button>
          <button
            onClick={() => setZoom((z) => Math.max(z / 1.2, 0.5))}
            className="p-2 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200"
          >
            <ZoomOut size={16} />
          </button>
          <button
            onClick={handleReset}
            className="p-2 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200"
          >
            <RotateCcw size={16} />
          </button>
        </div>
      </div>

      {/* Legend */}
      <div className="absolute top-4 left-4 p-4 bg-slate-900/80 backdrop-blur-sm rounded-lg border border-slate-700 max-w-xs pointer-events-none">
        <h3 className="text-sm font-mono font-bold text-blue-400 mb-2 flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
          CAUSAL TOPOLOGY // {activeLevel}
        </h3>
        <p className="text-[10px] text-slate-500 leading-relaxed mb-3">
          {activeLevel === 'MACRO' && 'Viewing high-level state actors and major events.'}
          {activeLevel === 'MESO' && 'Viewing organizational flows and supply chains.'}
          {activeLevel === 'ALL' && 'Full causal resolution. All nodes and entropy flows.'}
        </p>

        <div className="space-y-2 border-t border-slate-700 pt-3">
          <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400">
            <div className="w-2 h-2 rounded-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.8)]" />
            ACTOR (State/Org)
          </div>
          <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400">
            <div className="w-2 h-2 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)]" />
            EVENT (Conflict/Crisis)
          </div>
          <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400">
            <div className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.8)]" />
            RESOURCE (Supply/Asset)
          </div>
          <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400">
            <div className="w-8 h-0.5 bg-cyan-400" />
            CAUSALITY (Transfer Entropy)
          </div>
          <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400">
            <div className="w-8 h-0.5 border-t border-dashed border-slate-400" />
            CORRELATION (Lagged)
          </div>
        </div>
      </div>

      {/* Node count indicator */}
      <div className="absolute bottom-4 left-4 text-[10px] font-mono text-slate-500">
        {visibleNodes.length} nodes / {visibleEdges.length} edges
      </div>
    </div>
  );
}

export default CausalGraph;
