import React, { useEffect, useRef } from 'react';

interface PhaseBasinVizProps {
  stability: number; // 0 to 1
  isCritical: boolean;
}

export const PhaseBasinViz: React.FC<PhaseBasinVizProps> = ({ stability, isCritical }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let time = 0;
    const trail: {x: number, y: number, alpha: number}[] = [];

    const render = () => {
      time += 0.05;
      const width = canvas.width;
      const height = canvas.height;

      // Clear with trail effect
      ctx.fillStyle = 'rgba(2, 6, 23, 0.2)'; // Forge black with opacity for trails
      ctx.fillRect(0, 0, width, height);

      // Draw the Potential Well (Basin)
      ctx.beginPath();
      ctx.strokeStyle = isCritical ? '#f97316' : '#22d3ee';
      ctx.lineWidth = 2;
      ctx.shadowColor = isCritical ? '#f97316' : '#22d3ee';
      ctx.shadowBlur = 10;
      
      const points = [];
      for (let x = 0; x <= width; x += 2) {
        const nx = (x / width - 0.5) * 4;
        const effectiveStability = stability * 4;
        const y = Math.pow(nx, 4) - effectiveStability * Math.pow(nx, 2);
        const plotY = height / 2 + (y * 20) + 10; 
        points.push({x, y: plotY});
      }

      if (points.length > 0) {
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
          ctx.lineTo(points[i].x, points[i].y);
        }
      }
      ctx.stroke();
      
      // Reset shadow for fill
      ctx.shadowBlur = 0;

      // Draw Gradient Fill
      const gradient = ctx.createLinearGradient(0, 0, 0, height);
      gradient.addColorStop(0, isCritical ? 'rgba(249, 115, 22, 0)' : 'rgba(34, 211, 238, 0)');
      gradient.addColorStop(1, isCritical ? 'rgba(249, 115, 22, 0.1)' : 'rgba(34, 211, 238, 0.1)');
      ctx.fillStyle = gradient;
      ctx.lineTo(width, height);
      ctx.lineTo(0, height);
      ctx.fill();

      // Ball Physics
      const frequency = stability * 0.15; 
      const noise = (Math.random() - 0.5) * (1 - stability) * 30; 
      const ballNx = Math.sin(time * frequency) * Math.sqrt(stability) + (noise * 0.05);
      const ballX = (ballNx / 4 + 0.5) * width;
      
      const nx = (ballX / width - 0.5) * 4;
      const effectiveStability = stability * 4;
      const curveY = Math.pow(nx, 4) - effectiveStability * Math.pow(nx, 2);
      const ballY = height / 2 + (curveY * 20) + 10;

      // Trail Logic
      trail.push({x: ballX, y: ballY, alpha: 1.0});
      if (trail.length > 20) trail.shift();

      // Draw Trail
      trail.forEach((p, index) => {
          ctx.beginPath();
          ctx.arc(p.x, p.y - 6, 2 + (index/trail.length)*4, 0, Math.PI * 2);
          ctx.fillStyle = isCritical 
            ? `rgba(249, 115, 22, ${p.alpha * (index/trail.length) * 0.5})` 
            : `rgba(34, 211, 238, ${p.alpha * (index/trail.length) * 0.5})`;
          ctx.fill();
      });

      // Draw Ball
      ctx.beginPath();
      ctx.arc(ballX, ballY - 6, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#fff';
      ctx.shadowColor = isCritical ? '#f97316' : '#22d3ee';
      ctx.shadowBlur = 20;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Critical Warning Overlay
      if (isCritical && Math.random() > 0.9) {
          ctx.fillStyle = 'rgba(249, 115, 22, 0.1)';
          ctx.fillRect(0, 0, width, height);
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => cancelAnimationFrame(animationFrameId);
  }, [stability, isCritical]);

  return (
    <div className="relative w-full h-40 bg-black/40 rounded-lg overflow-hidden border border-white/5 shadow-inner">
        <canvas ref={canvasRef} width={400} height={160} className="w-full h-full" />
        <div className="absolute top-2 left-2 flex items-center space-x-2">
            <div className={`w-1.5 h-1.5 rounded-full ${isCritical ? 'bg-forge-orange animate-pulse' : 'bg-forge-cyan'}`}></div>
            <div className="text-[10px] font-mono text-gray-500 uppercase tracking-wider">
                Landau-Ginzburg Potential
            </div>
        </div>
        {isCritical && (
            <div className="absolute bottom-2 right-2 text-[10px] font-mono text-forge-orange bg-forge-orange/10 px-2 py-0.5 rounded border border-forge-orange/30 animate-pulse">
                CRITICAL REGIME
            </div>
        )}
    </div>
  );
};