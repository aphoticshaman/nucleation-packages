'use client';

import React, { useEffect, useRef } from 'react';

interface PhaseBasinVizProps {
  /** Stability parameter (0-1). Higher = deeper potential wells, less jitter */
  stability: number;
  /** Critical regime indicator - changes color scheme and adds warning */
  isCritical?: boolean;
  /** Optional label override */
  label?: string;
  /** Optional className */
  className?: string;
}

/**
 * PhaseBasinViz - Landau-Ginzburg Potential Well Visualization
 *
 * Visualizes system stability using the Landau-Ginzburg potential:
 * V(x) = x⁴ - ax² (Mexican hat / double-well potential)
 *
 * Mathematical interpretation:
 * - When a > 0: Two stable minima (bistability)
 * - As a → 0: Saddle-node bifurcation (phase transition)
 * - When a < 0: Single stable equilibrium
 *
 * The ball oscillates in the potential well with:
 * - Higher stability → deeper wells, smoother oscillation
 * - Lower stability → shallower wells, more chaotic motion
 */
export function PhaseBasinViz({
  stability,
  isCritical = false,
  label = 'Landau-Ginzburg Potential',
  className = '',
}: PhaseBasinVizProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let time = 0;
    const trail: { x: number; y: number; alpha: number }[] = [];

    const render = () => {
      time += 0.05;
      const width = canvas.width;
      const height = canvas.height;

      // Clear with trail effect (semi-transparent background for ghosting)
      ctx.fillStyle = 'rgba(2, 6, 23, 0.2)';
      ctx.fillRect(0, 0, width, height);

      // Colors based on critical state
      const primaryColor = isCritical ? '#f97316' : '#22d3ee'; // Orange vs Cyan
      const primaryRGB = isCritical ? '249, 115, 22' : '34, 211, 238';

      // Draw the potential well curve: V(x) = x⁴ - ax²
      ctx.beginPath();
      ctx.strokeStyle = primaryColor;
      ctx.lineWidth = 2;
      ctx.shadowColor = primaryColor;
      ctx.shadowBlur = 10;

      const points: { x: number; y: number }[] = [];
      for (let x = 0; x <= width; x += 2) {
        // Normalize x to [-2, 2] range
        const nx = (x / width - 0.5) * 4;
        // Effective stability parameter (scaled for visualization)
        const effectiveStability = stability * 4;
        // Landau-Ginzburg potential: V(x) = x⁴ - ax²
        const y = Math.pow(nx, 4) - effectiveStability * Math.pow(nx, 2);
        // Map to canvas coordinates (invert y, scale, offset)
        const plotY = height / 2 + y * 20 + 10;
        points.push({ x, y: plotY });
      }

      if (points.length > 0) {
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
          ctx.lineTo(points[i].x, points[i].y);
        }
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Draw gradient fill under curve
      const gradient = ctx.createLinearGradient(0, 0, 0, height);
      gradient.addColorStop(0, `rgba(${primaryRGB}, 0)`);
      gradient.addColorStop(1, `rgba(${primaryRGB}, 0.1)`);
      ctx.fillStyle = gradient;
      ctx.lineTo(width, height);
      ctx.lineTo(0, height);
      ctx.fill();

      // Ball physics - oscillates in the potential well
      const frequency = stability * 0.15;
      const noise = (Math.random() - 0.5) * (1 - stability) * 30;
      const ballNx = Math.sin(time * frequency) * Math.sqrt(stability) + noise * 0.05;
      const ballX = (ballNx / 4 + 0.5) * width;

      // Calculate ball Y position on the curve
      const nx = (ballX / width - 0.5) * 4;
      const effectiveStability = stability * 4;
      const curveY = Math.pow(nx, 4) - effectiveStability * Math.pow(nx, 2);
      const ballY = height / 2 + curveY * 20 + 10;

      // Trail logic - store positions for ghosting effect
      trail.push({ x: ballX, y: ballY, alpha: 1.0 });
      if (trail.length > 20) trail.shift();

      // Draw trail
      trail.forEach((p, index) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y - 6, 2 + (index / trail.length) * 4, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${primaryRGB}, ${p.alpha * (index / trail.length) * 0.5})`;
        ctx.fill();
      });

      // Draw main ball
      ctx.beginPath();
      ctx.arc(ballX, ballY - 6, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#fff';
      ctx.shadowColor = primaryColor;
      ctx.shadowBlur = 20;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Critical warning flash
      if (isCritical && Math.random() > 0.9) {
        ctx.fillStyle = 'rgba(249, 115, 22, 0.1)';
        ctx.fillRect(0, 0, width, height);
      }

      animationRef.current = requestAnimationFrame(render);
    };

    render();

    return () => cancelAnimationFrame(animationRef.current);
  }, [stability, isCritical]);

  return (
    <div
      className={`relative w-full h-40 bg-black/40 rounded-lg overflow-hidden border border-white/5 shadow-inner ${className}`}
    >
      <canvas ref={canvasRef} width={400} height={160} className="w-full h-full" />

      {/* Label */}
      <div className="absolute top-2 left-2 flex items-center space-x-2">
        <div
          className={`w-1.5 h-1.5 rounded-full ${
            isCritical ? 'bg-orange-500 animate-pulse' : 'bg-cyan-400'
          }`}
        />
        <div className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">
          {label}
        </div>
      </div>

      {/* Stability indicator */}
      <div className="absolute top-2 right-2 text-[10px] font-mono text-slate-500">
        σ = {stability.toFixed(2)}
      </div>

      {/* Critical warning badge */}
      {isCritical && (
        <div className="absolute bottom-2 right-2 text-[10px] font-mono text-orange-500 bg-orange-500/10 px-2 py-0.5 rounded border border-orange-500/30 animate-pulse">
          CRITICAL REGIME
        </div>
      )}
    </div>
  );
}

export default PhaseBasinViz;
