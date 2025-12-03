'use client';

import { useState, useMemo, useCallback, useRef, useEffect } from 'react';

interface HexbinPoint {
  id: string;
  lat: number;
  lng: number;
  value: number;
  category?: string;
  label?: string;
  metadata?: Record<string, unknown>;
}

interface HexbinMapProps {
  points: HexbinPoint[];
  width?: number;
  height?: number;
  hexRadius?: number;
  colorScale?: 'risk' | 'intensity' | 'sentiment' | 'custom';
  customColors?: { min: string; mid: string; max: string };
  onHexClick?: (hex: HexbinData) => void;
  onPointClick?: (point: HexbinPoint) => void;
  showLegend?: boolean;
  interactive?: boolean;
  bounds?: { minLat: number; maxLat: number; minLng: number; maxLng: number };
}

interface HexbinData {
  x: number;
  y: number;
  points: HexbinPoint[];
  avgValue: number;
  maxValue: number;
  count: number;
}

// Component 46: Geospatial Hexbin Heatmap
export function HexbinMap({
  points,
  width = 800,
  height = 500,
  hexRadius = 30,
  colorScale = 'risk',
  customColors,
  onHexClick,
  onPointClick,
  showLegend = true,
  interactive = true,
  bounds,
}: HexbinMapProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredHex, setHoveredHex] = useState<HexbinData | null>(null);
  const [selectedHex, setSelectedHex] = useState<HexbinData | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  // Calculate bounds from points if not provided
  const dataBounds = useMemo(() => {
    if (bounds) return bounds;
    if (points.length === 0) {
      return { minLat: -90, maxLat: 90, minLng: -180, maxLng: 180 };
    }

    const lats = points.map(p => p.lat);
    const lngs = points.map(p => p.lng);
    const padding = 5;

    return {
      minLat: Math.min(...lats) - padding,
      maxLat: Math.max(...lats) + padding,
      minLng: Math.min(...lngs) - padding,
      maxLng: Math.max(...lngs) + padding,
    };
  }, [points, bounds]);

  // Project lat/lng to pixel coordinates
  const project = useCallback((lat: number, lng: number) => {
    const x = ((lng - dataBounds.minLng) / (dataBounds.maxLng - dataBounds.minLng)) * width;
    const y = ((dataBounds.maxLat - lat) / (dataBounds.maxLat - dataBounds.minLat)) * height;
    return { x, y };
  }, [dataBounds, width, height]);

  // Generate hexagon path
  const hexPath = useMemo(() => {
    const angles = Array.from({ length: 6 }, (_, i) => (Math.PI / 3) * i - Math.PI / 6);
    const points = angles.map(a => [
      hexRadius * Math.cos(a),
      hexRadius * Math.sin(a),
    ]);
    return `M${points.map(p => p.join(',')).join('L')}Z`;
  }, [hexRadius]);

  // Bin points into hexagons
  const hexbins = useMemo(() => {
    const bins = new Map<string, HexbinData>();
    const hexWidth = hexRadius * 2;
    const hexHeight = hexRadius * Math.sqrt(3);

    for (const point of points) {
      const { x, y } = project(point.lat, point.lng);

      // Calculate hex grid position
      const col = Math.floor(x / (hexWidth * 0.75));
      const row = Math.floor(y / hexHeight + (col % 2) * 0.5);

      // Snap to hex center
      const hexX = col * hexWidth * 0.75 + hexRadius;
      const hexY = row * hexHeight + (col % 2) * hexHeight * 0.5 + hexRadius * 0.866;

      const key = `${col},${row}`;

      if (!bins.has(key)) {
        bins.set(key, {
          x: hexX,
          y: hexY,
          points: [],
          avgValue: 0,
          maxValue: 0,
          count: 0,
        });
      }

      const bin = bins.get(key)!;
      bin.points.push(point);
      bin.count++;
      bin.maxValue = Math.max(bin.maxValue, point.value);
    }

    // Calculate averages
    for (const bin of bins.values()) {
      bin.avgValue = bin.points.reduce((sum, p) => sum + p.value, 0) / bin.count;
    }

    return Array.from(bins.values());
  }, [points, hexRadius, project]);

  // Color scales
  const getColor = useCallback((value: number, max: number) => {
    const normalized = max > 0 ? value / max : 0;

    const scales = {
      risk: {
        min: [34, 197, 94],    // green
        mid: [250, 204, 21],   // yellow
        max: [239, 68, 68],    // red
      },
      intensity: {
        min: [15, 23, 42],     // dark slate
        mid: [6, 182, 212],    // cyan
        max: [255, 255, 255],  // white
      },
      sentiment: {
        min: [239, 68, 68],    // red (negative)
        mid: [100, 116, 139],  // slate (neutral)
        max: [34, 197, 94],    // green (positive)
      },
      custom: customColors ? {
        min: hexToRgb(customColors.min),
        mid: hexToRgb(customColors.mid),
        max: hexToRgb(customColors.max),
      } : { min: [0, 0, 0], mid: [128, 128, 128], max: [255, 255, 255] },
    };

    const scale = scales[colorScale];

    let color: number[];
    if (normalized < 0.5) {
      const t = normalized * 2;
      color = scale.min.map((c, i) => Math.round(c + (scale.mid[i] - c) * t));
    } else {
      const t = (normalized - 0.5) * 2;
      color = scale.mid.map((c, i) => Math.round(c + (scale.max[i] - c) * t));
    }

    return `rgb(${color.join(',')})`;
  }, [colorScale, customColors]);

  const maxValue = useMemo(() => {
    return Math.max(...hexbins.map(h => h.avgValue), 0.001);
  }, [hexbins]);

  // Pan/zoom handlers
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (!interactive) return;
    setIsPanning(true);
    setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  }, [interactive, pan]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isPanning) return;
    setPan({
      x: e.clientX - panStart.x,
      y: e.clientY - panStart.y,
    });
  }, [isPanning, panStart]);

  const handleMouseUp = useCallback(() => {
    setIsPanning(false);
  }, []);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (!interactive) return;
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(prev => Math.max(0.5, Math.min(5, prev * delta)));
  }, [interactive]);

  return (
    <div className="relative bg-slate-900/50 rounded-lg border border-slate-700 overflow-hidden">
      {/* Map SVG */}
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className={interactive ? 'cursor-move' : ''}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      >
        {/* Background grid */}
        <defs>
          <pattern id="hex-grid" width="60" height="52" patternUnits="userSpaceOnUse">
            <path
              d="M30 0 L60 15 L60 45 L30 60 L0 45 L0 15 Z"
              fill="none"
              stroke="rgba(100,116,139,0.1)"
              strokeWidth="0.5"
            />
          </pattern>
        </defs>
        <rect width={width} height={height} fill="url(#hex-grid)" />

        {/* Transformed content */}
        <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
          {/* World outline (simplified) */}
          <path
            d="M50,250 Q150,200 250,220 Q350,190 450,200 Q550,180 650,210 Q750,190 800,250
               Q750,300 650,290 Q550,320 450,300 Q350,330 250,310 Q150,340 50,300 Z"
            fill="none"
            stroke="rgba(100,116,139,0.3)"
            strokeWidth="1"
          />

          {/* Hexbins */}
          {hexbins.map((hex, i) => {
            const isHovered = hoveredHex === hex;
            const isSelected = selectedHex === hex;
            const color = getColor(hex.avgValue, maxValue);
            const opacity = 0.3 + (hex.avgValue / maxValue) * 0.6;

            return (
              <g
                key={i}
                transform={`translate(${hex.x}, ${hex.y})`}
                className="transition-all duration-200"
                onMouseEnter={() => setHoveredHex(hex)}
                onMouseLeave={() => setHoveredHex(null)}
                onClick={() => {
                  setSelectedHex(hex);
                  onHexClick?.(hex);
                }}
              >
                <path
                  d={hexPath}
                  fill={color}
                  opacity={opacity}
                  stroke={isHovered || isSelected ? '#fff' : 'rgba(255,255,255,0.1)'}
                  strokeWidth={isHovered || isSelected ? 2 : 0.5}
                  className="cursor-pointer"
                />
                {/* Count label */}
                {hex.count > 1 && (
                  <text
                    y={4}
                    textAnchor="middle"
                    className="text-xs fill-white font-bold pointer-events-none"
                    style={{ textShadow: '0 0 3px rgba(0,0,0,0.8)' }}
                  >
                    {hex.count}
                  </text>
                )}
              </g>
            );
          })}

          {/* Individual points (show when zoomed in) */}
          {zoom > 2 && points.map(point => {
            const { x, y } = project(point.lat, point.lng);
            return (
              <circle
                key={point.id}
                cx={x}
                cy={y}
                r={4}
                fill={getColor(point.value, maxValue)}
                stroke="#fff"
                strokeWidth={1}
                className="cursor-pointer"
                onClick={(e) => {
                  e.stopPropagation();
                  onPointClick?.(point);
                }}
              >
                <title>{point.label || point.id}</title>
              </circle>
            );
          })}
        </g>

        {/* Zoom controls */}
        {interactive && (
          <g transform={`translate(${width - 50}, 20)`}>
            <rect x={0} y={0} width={30} height={60} rx={4} fill="rgba(15,23,42,0.8)" />
            <text
              x={15} y={22}
              textAnchor="middle"
              className="fill-slate-400 text-lg cursor-pointer select-none"
              onClick={() => setZoom(prev => Math.min(5, prev * 1.2))}
            >
              +
            </text>
            <line x1={5} y1={30} x2={25} y2={30} stroke="rgba(100,116,139,0.5)" />
            <text
              x={15} y={48}
              textAnchor="middle"
              className="fill-slate-400 text-lg cursor-pointer select-none"
              onClick={() => setZoom(prev => Math.max(0.5, prev / 1.2))}
            >
              −
            </text>
          </g>
        )}
      </svg>

      {/* Tooltip */}
      {hoveredHex && (
        <div
          className="absolute z-10 pointer-events-none"
          style={{
            left: hoveredHex.x * zoom + pan.x + 50,
            top: hoveredHex.y * zoom + pan.y - 30,
          }}
        >
          <div className="bg-slate-800 border border-slate-600 rounded-lg shadow-xl p-3 min-w-[150px]">
            <div className="text-sm font-medium text-slate-200">
              {hoveredHex.count} event{hoveredHex.count !== 1 ? 's' : ''}
            </div>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-xs text-slate-400">Avg Risk:</span>
              <span
                className="text-sm font-mono"
                style={{ color: getColor(hoveredHex.avgValue, maxValue) }}
              >
                {(hoveredHex.avgValue * 100).toFixed(0)}%
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">Max Risk:</span>
              <span
                className="text-sm font-mono"
                style={{ color: getColor(hoveredHex.maxValue, maxValue) }}
              >
                {(hoveredHex.maxValue * 100).toFixed(0)}%
              </span>
            </div>
            {hoveredHex.points[0]?.category && (
              <div className="mt-2 text-xs text-slate-500">
                Categories: {[...new Set(hoveredHex.points.map(p => p.category))].join(', ')}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Legend */}
      {showLegend && (
        <div className="absolute bottom-4 left-4 bg-slate-800/90 rounded-lg p-3 border border-slate-700">
          <div className="text-xs text-slate-400 mb-2">Risk Level</div>
          <div className="flex items-center gap-1">
            {[0, 0.25, 0.5, 0.75, 1].map((v, i) => (
              <div
                key={i}
                className="w-6 h-4 rounded-sm"
                style={{ backgroundColor: getColor(v * maxValue, maxValue) }}
              />
            ))}
          </div>
          <div className="flex justify-between text-xs text-slate-500 mt-1">
            <span>Low</span>
            <span>High</span>
          </div>
        </div>
      )}

      {/* Coordinates display */}
      <div className="absolute top-4 left-4 text-xs text-slate-500 font-mono">
        Zoom: {(zoom * 100).toFixed(0)}%
      </div>
    </div>
  );
}

// Helper function
function hexToRgb(hex: string): number[] {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)]
    : [0, 0, 0];
}

// Mock data for demo
export const mockHexbinPoints: HexbinPoint[] = [
  // Eastern Europe cluster
  { id: '1', lat: 50.4501, lng: 30.5234, value: 0.92, category: 'military', label: 'Kyiv' },
  { id: '2', lat: 48.4647, lng: 35.0462, value: 0.88, category: 'military', label: 'Dnipro' },
  { id: '3', lat: 49.8397, lng: 24.0297, value: 0.75, category: 'military', label: 'Lviv' },
  { id: '4', lat: 55.7558, lng: 37.6173, value: 0.65, category: 'political', label: 'Moscow' },
  // Middle East
  { id: '5', lat: 31.7683, lng: 35.2137, value: 0.85, category: 'military', label: 'Jerusalem' },
  { id: '6', lat: 33.8938, lng: 35.5018, value: 0.78, category: 'political', label: 'Beirut' },
  { id: '7', lat: 35.6762, lng: 51.4241, value: 0.72, category: 'nuclear', label: 'Tehran' },
  // East Asia
  { id: '8', lat: 25.0330, lng: 121.5654, value: 0.68, category: 'military', label: 'Taipei' },
  { id: '9', lat: 39.9042, lng: 116.4074, value: 0.45, category: 'economic', label: 'Beijing' },
  { id: '10', lat: 37.5665, lng: 126.9780, value: 0.52, category: 'military', label: 'Seoul' },
  { id: '11', lat: 39.0392, lng: 125.7625, value: 0.82, category: 'nuclear', label: 'Pyongyang' },
  // Africa
  { id: '12', lat: 9.0820, lng: 8.6753, value: 0.58, category: 'insurgency', label: 'Nigeria' },
  { id: '13', lat: 15.5007, lng: 32.5599, value: 0.71, category: 'political', label: 'Khartoum' },
  // South America
  { id: '14', lat: 10.4806, lng: -66.9036, value: 0.55, category: 'economic', label: 'Caracas' },
];
