import React, { useEffect, useRef, useState } from 'react';
import { ContextLayer, FoldNode } from '../types';

interface FoldingCoreProps {
  context: ContextLayer;
  entropy: number; // 0-1 value driving jitter
}

export const FoldingCore: React.FC<FoldingCoreProps> = ({ context, entropy }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [nodes, setNodes] = useState<FoldNode[]>([]);

  // Initialize Nodes
  useEffect(() => {
    const initialNodes: FoldNode[] = [];
    const count = 60; // "60 detents in the watch"
    for (let i = 0; i < count; i++) {
      initialNodes.push({
        id: i,
        x: Math.random() * 800,
        y: Math.random() * 600,
        vx: 0,
        vy: 0,
        targetX: 400,
        targetY: 300,
        // Design System: Accent (Teal), Primary (Blue), Inverse Text (White)
        color: i % 3 === 0 ? '#14B8A6' : i % 3 === 1 ? '#3B82F6' : '#E5E5E5',
        size: Math.random() * 3 + 1,
        connections: [Math.floor(Math.random() * count), Math.floor(Math.random() * count)],
        data: Math.random().toString(36).substring(7)
      });
    }
    setNodes(initialNodes);
  }, []);

  // Update Targets based on Context ("The Folding Trigger")
  useEffect(() => {
    setNodes(prev => prev.map((node, i) => {
      let tx = 0, ty = 0;
      
      // CMFC Logic: different structures per context
      if (context === ContextLayer.SURFACE) {
        // Map/Grid Structure
        const col = i % 10;
        const row = Math.floor(i / 10);
        tx = 150 + col * 50;
        ty = 150 + row * 50;
      } else if (context === ContextLayer.DEEP) {
        // Orbital/Atom Structure
        const angle = (i / prev.length) * Math.PI * 4; // Double spiral
        const radius = 50 + i * 4;
        tx = 400 + Math.cos(angle) * radius;
        ty = 300 + Math.sin(angle) * radius;
      } else {
        // Network/Chaotic Structure (Enochian Glyphs)
        const cluster = i % 4;
        const cx = cluster === 0 ? 200 : cluster === 1 ? 600 : cluster === 2 ? 400 : 400;
        const cy = cluster === 0 ? 200 : cluster === 1 ? 200 : cluster === 2 ? 500 : 300;
        tx = cx + (Math.random() - 0.5) * 120;
        ty = cy + (Math.random() - 0.5) * 120;
      }

      return { ...node, targetX: tx, targetY: ty };
    }));
  }, [context]);

  // Animation Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;

    const render = () => {
      // Trail effect - Dark BG (#0A0A0A)
      ctx.fillStyle = 'rgba(10, 10, 10, 0.2)'; 
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Physics Update
      nodes.forEach(node => {
        // Spring force to target
        const dx = node.targetX - node.x;
        const dy = node.targetY - node.y;
        
        node.vx += dx * 0.05;
        node.vy += dy * 0.05;
        
        // Entropy Jitter
        const jitter = Math.max(0.01, entropy);
        node.vx += (Math.random() - 0.5) * jitter * 2;
        node.vy += (Math.random() - 0.5) * jitter * 2;

        // Damping
        node.vx *= 0.9;
        node.vy *= 0.9;

        node.x += node.vx;
        node.y += node.vy;

        // Draw Connections - Primary Blue
        ctx.beginPath();
        node.connections.forEach(targetId => {
          const target = nodes[targetId];
          if (target) {
            const dist = Math.hypot(target.x - node.x, target.y - node.y);
            if (dist < 100) {
              // Blue-500 with fade
              ctx.strokeStyle = `rgba(59, 130, 246, ${1 - dist/100})`;
              ctx.lineWidth = 0.5;
              ctx.moveTo(node.x, node.y);
              ctx.lineTo(target.x, target.y);
            }
          }
        });
        ctx.stroke();

        // Draw Node
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.size, 0, Math.PI * 2);
        ctx.fillStyle = node.color;
        ctx.fill();
        
        // Draw "Data" - White
        if (Math.random() > 0.98) {
            ctx.fillStyle = 'rgba(255,255,255,0.7)';
            ctx.font = '10px monospace';
            ctx.fillText(node.data, node.x + 8, node.y - 8);
        }
      });

      animId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, [nodes, entropy]);

  return (
    <div className="w-full h-full relative border border-border-default bg-surface/30 rounded-xl overflow-hidden">
      <div className="absolute top-4 left-4 z-10 pointer-events-none">
        <div className="text-[10px] font-mono text-accent tracking-widest mb-1">
          TOPOLOGY: {context.split(' ')[0]}
        </div>
        <div className="text-[10px] font-mono text-text-muted">
          ENTROPY: {(entropy * 100).toFixed(2)}%
        </div>
      </div>
      <canvas ref={canvasRef} width={800} height={600} className="w-full h-full" />
    </div>
  );
};