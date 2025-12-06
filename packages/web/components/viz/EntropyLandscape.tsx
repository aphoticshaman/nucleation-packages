'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Play, Pause, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react';
import {
  generatePotentialCurve,
  simulateTrajectory,
  createPhaseState,
  type PhaseState,
  type LandauGinzburgConfig,
} from '@/lib/physics/landau-ginzburg';

interface EntropyLandscapeProps {
  config?: Partial<LandauGinzburgConfig>;
  initialState?: PhaseState;
  showTrajectory?: boolean;
  showParticle?: boolean;
  height?: number;
}

/**
 * Interactive Landau-Ginzburg Potential Landscape Visualization
 *
 * Shows the "Mexican hat" potential with real-time particle dynamics
 * for visualizing phase transition basins.
 */
export function EntropyLandscape({
  config = { a: 1.0, b: 1.0 },
  initialState,
  showTrajectory = true,
  showParticle = true,
  height = 300,
}: EntropyLandscapeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [trajectory, setTrajectory] = useState<PhaseState[]>([]);
  const [zoom, setZoom] = useState(1);

  const animationRef = useRef<number | null>(null);
  const stateRef = useRef<PhaseState>(
    initialState || createPhaseState(0.7, 'stable')
  );

  // Generate potential curve
  const potentialCurve = generatePotentialCurve(config, 200);

  // Find Y-axis bounds
  const vMin = Math.min(...potentialCurve.map(p => p.V));
  const vMax = Math.max(...potentialCurve.map(p => p.V));
  const vRange = vMax - vMin;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const render = () => {
      const width = canvas.width;
      const height = canvas.height;
      const padding = 40;

      // Clear
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, width, height);

      // Grid
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = padding; x < width - padding; x += 40) {
        ctx.moveTo(x, padding);
        ctx.lineTo(x, height - padding);
      }
      for (let y = padding; y < height - padding; y += 40) {
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
      }
      ctx.stroke();

      // Axes
      ctx.strokeStyle = '#475569';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(padding, height - padding);
      ctx.lineTo(width - padding, height - padding);
      ctx.moveTo(width / 2, padding);
      ctx.lineTo(width / 2, height - padding);
      ctx.stroke();

      // Map coordinates
      const mapX = (phi: number) =>
        padding + ((phi + 2) / 4) * (width - 2 * padding) * zoom;
      const mapY = (V: number) =>
        height - padding - ((V - vMin) / vRange) * (height - 2 * padding) * 0.8;

      // Draw potential curve
      ctx.beginPath();
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 3;
      let first = true;
      for (const point of potentialCurve) {
        const x = mapX(point.phi);
        const y = mapY(point.V);
        if (x < padding || x > width - padding) continue;
        if (first) {
          ctx.moveTo(x, y);
          first = false;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      // Fill under curve with gradient
      const gradient = ctx.createLinearGradient(0, padding, 0, height - padding);
      gradient.addColorStop(0, 'rgba(59, 130, 246, 0.1)');
      gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');

      ctx.beginPath();
      ctx.fillStyle = gradient;
      first = true;
      for (const point of potentialCurve) {
        const x = mapX(point.phi);
        const y = mapY(point.V);
        if (x < padding || x > width - padding) continue;
        if (first) {
          ctx.moveTo(x, height - padding);
          ctx.lineTo(x, y);
          first = false;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.lineTo(mapX(potentialCurve[potentialCurve.length - 1].phi), height - padding);
      ctx.closePath();
      ctx.fill();

      // Draw trajectory
      if (showTrajectory && trajectory.length > 1) {
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(34, 211, 238, 0.5)';
        ctx.lineWidth = 2;

        trajectory.forEach((state, i) => {
          const x = mapX(state.orderParameter);
          const y = mapY(state.potential);
          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        });
        ctx.stroke();
      }

      // Draw particle
      if (showParticle) {
        const currentState = stateRef.current;
        const px = mapX(currentState.orderParameter);
        const py = mapY(currentState.potential);

        // Glow
        const glow = ctx.createRadialGradient(px, py, 0, px, py, 20);
        glow.addColorStop(0, 'rgba(251, 191, 36, 0.8)');
        glow.addColorStop(1, 'rgba(251, 191, 36, 0)');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(px, py, 20, 0, Math.PI * 2);
        ctx.fill();

        // Core
        ctx.fillStyle = '#fbbf24';
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fill();

        // Velocity indicator
        if (Math.abs(currentState.velocity) > 0.01) {
          ctx.strokeStyle = '#fbbf24';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(px + currentState.velocity * 50, py);
          ctx.stroke();
        }
      }

      // Labels
      ctx.fillStyle = '#94a3b8';
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('φ (Order Parameter)', width / 2, height - 10);

      ctx.save();
      ctx.translate(15, height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('V(φ) Potential', 0, 0);
      ctx.restore();

      // Basin labels
      if (config.a && config.a > 0) {
        ctx.fillStyle = '#3b82f6';
        ctx.font = 'bold 10px JetBrains Mono, monospace';
        ctx.fillText('Basin A', mapX(-0.8), height - padding - 20);
        ctx.fillText('Basin B', mapX(0.8), height - padding - 20);
        ctx.fillStyle = '#ef4444';
        ctx.fillText('Barrier', width / 2, mapY(0) - 10);
      }
    };

    render();

    // Animation loop
    if (isSimulating) {
      const animate = () => {
        // Simulate one step
        const newTrajectory = simulateTrajectory(stateRef.current, 1, config);
        stateRef.current = newTrajectory[newTrajectory.length - 1];

        setTrajectory(prev => {
          const updated = [...prev, stateRef.current];
          return updated.slice(-500); // Keep last 500 points
        });

        render();
        animationRef.current = requestAnimationFrame(animate);
      };
      animationRef.current = requestAnimationFrame(animate);
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [config, trajectory, isSimulating, showTrajectory, showParticle, zoom, potentialCurve, vMin, vRange]);

  const handleReset = () => {
    stateRef.current = initialState || createPhaseState(0.7, 'stable');
    setTrajectory([]);
    setIsSimulating(false);
  };

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={600}
        height={height}
        className="w-full rounded-lg"
        style={{ height }}
      />

      {/* Controls */}
      <div className="absolute top-4 right-4 flex gap-2">
        <button
          onClick={() => setIsSimulating(!isSimulating)}
          className="p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 border border-slate-600 transition-colors"
        >
          {isSimulating ? <Pause size={16} /> : <Play size={16} />}
        </button>
        <button
          onClick={handleReset}
          className="p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 border border-slate-600 transition-colors"
        >
          <RotateCcw size={16} />
        </button>
        <button
          onClick={() => setZoom(z => Math.min(2, z + 0.2))}
          className="p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 border border-slate-600 transition-colors"
        >
          <ZoomIn size={16} />
        </button>
        <button
          onClick={() => setZoom(z => Math.max(0.5, z - 0.2))}
          className="p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 border border-slate-600 transition-colors"
        >
          <ZoomOut size={16} />
        </button>
      </div>

      {/* Status */}
      <div className="absolute bottom-4 left-4 p-3 rounded-lg bg-slate-900/90 border border-slate-700">
        <div className="grid grid-cols-3 gap-4 text-xs font-mono">
          <div>
            <div className="text-slate-500">φ</div>
            <div className="text-white">{stateRef.current.orderParameter.toFixed(3)}</div>
          </div>
          <div>
            <div className="text-slate-500">V(φ)</div>
            <div className="text-white">{stateRef.current.potential.toFixed(3)}</div>
          </div>
          <div>
            <div className="text-slate-500">Basin</div>
            <div className={`${
              stateRef.current.basinId === 'left' ? 'text-blue-400' :
              stateRef.current.basinId === 'right' ? 'text-green-400' :
              'text-red-400'
            }`}>
              {stateRef.current.basinId.toUpperCase()}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
