
import React, { useEffect, useRef, useState } from 'react';
import { CausalNode, CausalEdge } from '../types';
import { MOCK_NODES, MOCK_EDGES } from '../constants';
import { Layers, Maximize, Activity } from 'lucide-react';

export const CausalCanvas: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [nodes, setNodes] = useState<CausalNode[]>([]);
  const [edges, _setEdges] = useState<CausalEdge[]>(MOCK_EDGES);
  const [selectedNode, _setSelectedNode] = useState<string | null>(null);
  const [activeLevel, setActiveLevel] = useState<'ALL' | 'MACRO' | 'MESO' | 'MICRO'>('ALL');

  // Initialize Simulation
  useEffect(() => {
    // Clone to avoid mutation of constants
    const initialNodes = MOCK_NODES.map(n => ({
      ...n,
      x: Math.random() * 800,
      y: Math.random() * 600,
      vx: 0,
      vy: 0
    }));
    setNodes(initialNodes);
  }, []);

  // Filter Logic
  const visibleNodes = nodes.filter(n => {
    if (activeLevel === 'ALL') return true;
    if (activeLevel === 'MACRO') return n.level === 'MACRO';
    if (activeLevel === 'MESO') return n.level === 'MACRO' || n.level === 'MESO'; // Meso shows context
    return true; // Micro shows all
  });

  const visibleEdges = edges.filter(e => {
    const source = visibleNodes.find(n => n.id === e.source);
    const target = visibleNodes.find(n => n.id === e.target);
    return source && target;
  });

  // Physics Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;
    let time = 0;

    const render = () => {
      time += 0.01;
      
      // Resize handling (basic)
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      const width = canvas.width;
      const height = canvas.height;

      // 1. Clear & Background
      ctx.fillStyle = '#020617'; // app background
      ctx.fillRect(0, 0, width, height);
      
      // Grid
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      ctx.beginPath();
      const gridSize = activeLevel === 'MACRO' ? 80 : 40; // Adapt grid to semantic level
      for (let x = 0; x < width; x += gridSize) { ctx.moveTo(x, 0); ctx.lineTo(x, height); }
      for (let y = 0; y < height; y += gridSize) { ctx.moveTo(0, y); ctx.lineTo(width, y); }
      ctx.stroke();

      // 2. Physics Integration
      const kRepel = activeLevel === 'MACRO' ? 3000 : 2000;
      const kSpring = 0.05;
      const damping = 0.85;
      const centerForce = 0.005;

      visibleNodes.forEach(node => {
        if (!node.x || !node.y) return;
        let fx = 0, fy = 0;

        // Repulsion
        visibleNodes.forEach(other => {
          if (node.id === other.id || !other.x || !other.y) return;
          const dx = node.x - other.x;
          const dy = node.y - other.y;
          const distSq = dx*dx + dy*dy;
          if (distSq > 0) {
             const f = kRepel / Math.sqrt(distSq);
             const angle = Math.atan2(dy, dx);
             fx += Math.cos(angle) * f;
             fy += Math.sin(angle) * f;
          }
        });

        // Center Gravity
        const dx = (width / 2) - node.x;
        const dy = (height / 2) - node.y;
        fx += dx * centerForce;
        fy += dy * centerForce;

        // Apply Forces
        node.vx = (node.vx || 0) + fx * 0.01;
        node.vy = (node.vy || 0) + fy * 0.01;
      });

      // Spring Forces (Edges)
      visibleEdges.forEach(edge => {
        const source = nodes.find(n => n.id === edge.source);
        const target = nodes.find(n => n.id === edge.target);
        if (source && target && source.x && source.y && target.x && target.y) {
          const dx = target.x - source.x;
          const dy = target.y - source.y;
          const dist = Math.sqrt(dx*dx + dy*dy);
          const restLength = activeLevel === 'MACRO' ? 250 : 150; 
          const force = (dist - restLength) * kSpring;
          
          const angle = Math.atan2(dy, dx);
          const fx = Math.cos(angle) * force;
          const fy = Math.sin(angle) * force;

          source.vx = (source.vx || 0) + fx;
          source.vy = (source.vy || 0) + fy;
          target.vx = (target.vx || 0) - fx;
          target.vy = (target.vy || 0) - fy;
        }
      });

      // Update Positions
      visibleNodes.forEach(node => {
        if (selectedNode === node.id) return; // Don't move selected node if dragging (simulated)

        node.vx = (node.vx || 0) * damping;
        node.vy = (node.vy || 0) * damping;
        node.x = (node.x || 0) + (node.vx || 0);
        node.y = (node.y || 0) + (node.vy || 0);

        // Bounds
        node.x = Math.max(40, Math.min(width - 40, node.x));
        node.y = Math.max(40, Math.min(height - 40, node.y));
      });

      // 3. Render Edges (Transfer Entropy Flows)
      visibleEdges.forEach(edge => {
        const source = nodes.find(n => n.id === edge.source);
        const target = nodes.find(n => n.id === edge.target);
        if (source && target && source.x && source.y && target.x && target.y) {
          ctx.beginPath();
          ctx.moveTo(source.x, source.y);
          ctx.lineTo(target.x, target.y);
          
          // Style based on TE magnitude
          const opacity = 0.2 + (edge.transferEntropy * 0.8);
          ctx.strokeStyle = edge.type === 'CAUSALITY' ? `rgba(34, 211, 238, ${opacity})` : `rgba(148, 163, 184, ${opacity})`;
          ctx.lineWidth = 1 + (edge.transferEntropy * 3);
          
          if (edge.type === 'CAUSALITY') {
             ctx.setLineDash([]);
          } else {
             ctx.setLineDash([4, 4]); // Dashed for correlation/influence
          }
          ctx.stroke();

          // Flow Particle (Information Transfer)
          const flowSpeed = time * (0.5 + edge.transferEntropy);
          const flowPos = flowSpeed % 1;
          const px = source.x + (target.x - source.x) * flowPos;
          const py = source.y + (target.y - source.y) * flowPos;
          
          ctx.beginPath();
          ctx.arc(px, py, 2, 0, Math.PI * 2);
          ctx.fillStyle = '#fff';
          ctx.fill();
        }
      });
      ctx.setLineDash([]);

      // 4. Render Nodes
      visibleNodes.forEach(node => {
        if (!node.x || !node.y) return;

        // Belief Mass Halo (Dempster-Shafer)
        const baseRadius = node.level === 'MACRO' ? 18 : node.level === 'MESO' ? 12 : 6;
        const pulse = Math.sin(time * 3 + (node.x * 0.1)) * 0.1 + 1; 
        
        // Halo (Uncertainty Visualization)
        const grad = ctx.createRadialGradient(node.x, node.y, baseRadius, node.x, node.y, baseRadius * 3 * node.beliefMass);
        const color = node.type === 'EVENT' ? '239, 68, 68' : node.type === 'ACTOR' ? '59, 130, 246' : '34, 211, 238';
        
        grad.addColorStop(0, `rgba(${color}, 0.4)`);
        grad.addColorStop(1, `rgba(${color}, 0)`);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(node.x, node.y, baseRadius * 3 * node.beliefMass * pulse, 0, Math.PI * 2);
        ctx.fill();

        // Core Node
        ctx.beginPath();
        ctx.arc(node.x, node.y, baseRadius, 0, Math.PI * 2);
        ctx.fillStyle = `rgb(${color})`; 
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = node.id === selectedNode ? 3 : 1.5;
        ctx.stroke();

        // Label (Progressive Disclosure)
        // Always show Macro/Meso. Show Micro only on hover or high zoom (simulated here by list size)
        if (node.level !== 'MICRO' || visibleNodes.length < 15) {
          ctx.fillStyle = '#e2e8f0';
          ctx.font = node.level === 'MACRO' ? 'bold 12px JetBrains Mono' : '10px JetBrains Mono';
          ctx.fillText(node.label, node.x + baseRadius + 8, node.y + 4);
          
          // Show Belief Mass as small bar below label
          ctx.fillStyle = '#1e293b';
          ctx.fillRect(node.x + baseRadius + 8, node.y + 8, 30, 3);
          ctx.fillStyle = `rgb(${color})`;
          ctx.fillRect(node.x + baseRadius + 8, node.y + 8, 30 * node.beliefMass, 3);
        }
      });

      animationId = requestAnimationFrame(render);
    };
    render();

    return () => cancelAnimationFrame(animationId);
  }, [nodes, edges, selectedNode, activeLevel]);

  return (
    <div className="relative w-full h-full bg-app overflow-hidden">
      <canvas 
        ref={canvasRef} 
        className="block w-full h-full cursor-crosshair"
      />
      
      {/* Semantic Layering Controls (The "Lens") */}
      <div className="absolute top-4 right-4 flex flex-col gap-2">
         <div className="glass-panel p-2 rounded-lg flex flex-col gap-2">
            <div className="text-[10px] font-mono text-text-muted text-center uppercase tracking-wider mb-1">Semantic Depth</div>
            <button 
                onClick={() => setActiveLevel('MACRO')}
                className={`p-2 rounded transition-colors flex items-center justify-between gap-2 text-[10px] font-mono border ${activeLevel === 'MACRO' ? 'bg-primary/20 border-primary text-primary' : 'hover:bg-surface-raised border-transparent text-text-secondary'}`}
            >
                <Maximize size={14} /> MACRO (STRATEGIC)
            </button>
            <button 
                onClick={() => setActiveLevel('MESO')}
                className={`p-2 rounded transition-colors flex items-center justify-between gap-2 text-[10px] font-mono border ${activeLevel === 'MESO' ? 'bg-primary/20 border-primary text-primary' : 'hover:bg-surface-raised border-transparent text-text-secondary'}`}
            >
                <Activity size={14} /> MESO (OPERATIONAL)
            </button>
            <button 
                onClick={() => setActiveLevel('ALL')}
                className={`p-2 rounded transition-colors flex items-center justify-between gap-2 text-[10px] font-mono border ${activeLevel === 'ALL' ? 'bg-primary/20 border-primary text-primary' : 'hover:bg-surface-raised border-transparent text-text-secondary'}`}
            >
                <Layers size={14} /> MICRO (TACTICAL)
            </button>
         </div>
      </div>

      {/* Legend / Status Overlay */}
      <div className="absolute top-4 left-4 p-4 glass-panel rounded-lg border border-border-default max-w-xs pointer-events-none">
        <h3 className="text-sm font-mono font-bold text-primary mb-2 flex items-center gap-2">
           <div className={`w-2 h-2 rounded-full bg-primary animate-pulse`}></div>
           CAUSAL TOPOLOGY // {activeLevel}
        </h3>
        <p className="text-[10px] text-text-muted leading-relaxed mb-3">
          {activeLevel === 'MACRO' && "Viewing high-level state actors and major events. Noise filtered."}
          {activeLevel === 'MESO' && "Viewing organizational flows and supply chain dependencies."}
          {activeLevel === 'ALL' && "Full causal resolution. Showing all active nodes and entropy flows."}
        </p>
        
        <div className="space-y-2 border-t border-border-muted pt-3">
            <div className="flex items-center gap-2 text-[10px] font-mono text-text-secondary">
                <div className="w-2 h-2 rounded-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.8)]"></div> 
                ACTOR (State/Org)
            </div>
            <div className="flex items-center gap-2 text-[10px] font-mono text-text-secondary">
                <div className="w-2 h-2 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)]"></div> 
                EVENT (Conflict/Crisis)
            </div>
            <div className="flex items-center gap-2 text-[10px] font-mono text-text-secondary">
                <div className="w-8 h-0.5 bg-cyan-400"></div> 
                CAUSALITY (Transfer Entropy)
            </div>
             <div className="flex items-center gap-2 text-[10px] font-mono text-text-secondary">
                <div className="w-8 h-0.5 border-t border-dashed border-slate-400"></div> 
                CORRELATION (Lagged)
            </div>
        </div>
      </div>
    </div>
  );
};
