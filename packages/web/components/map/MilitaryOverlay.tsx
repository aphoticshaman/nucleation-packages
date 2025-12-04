'use client';

import { useMemo } from 'react';
import type {
  MilitaryAsset,
  TerritorialStatus,
  ContestedZone,
  ForceDisposition,
  UnitType,
  UnitSize,
  ThreatLevel,
  SovereigntyStatus,
} from '@/lib/territorial-status';
import {
  DISPOSITION_COLORS,
  THREAT_COLORS,
  UNIT_ICONS,
  SIZE_MODIFIERS,
  SOVEREIGNTY_STYLES,
} from '@/lib/territorial-status';

/**
 * MilitaryOverlay - Blue Force Tracker Style Map Overlays
 *
 * Renders:
 * - Territorial control/sovereignty (transparent country fills with patterns)
 * - Contested zone overlays (crosshatch, stripes)
 * - Military asset symbols (APP-6 style NATO symbols)
 * - Force disposition indicators (friendly/hostile/neutral/unknown)
 */

// ============================================
// SVG PATTERN DEFINITIONS
// ============================================

export function PatternDefs() {
  return (
    <defs>
      {/* Diagonal stripes - for occupied territories */}
      <pattern id="pattern-stripes" patternUnits="userSpaceOnUse" width="10" height="10" patternTransform="rotate(45)">
        <line x1="0" y1="0" x2="0" y2="10" stroke="currentColor" strokeWidth="2" strokeOpacity="0.5" />
      </pattern>

      {/* Dots - for unrecognized territories */}
      <pattern id="pattern-dots" patternUnits="userSpaceOnUse" width="10" height="10">
        <circle cx="5" cy="5" r="1.5" fill="currentColor" fillOpacity="0.5" />
      </pattern>

      {/* Crosshatch - for annexed/heavily contested */}
      <pattern id="pattern-crosshatch" patternUnits="userSpaceOnUse" width="10" height="10">
        <line x1="0" y1="0" x2="10" y2="10" stroke="currentColor" strokeWidth="1" strokeOpacity="0.4" />
        <line x1="10" y1="0" x2="0" y2="10" stroke="currentColor" strokeWidth="1" strokeOpacity="0.4" />
      </pattern>

      {/* Diagonal lines (alternate direction) */}
      <pattern id="pattern-diagonal" patternUnits="userSpaceOnUse" width="10" height="10" patternTransform="rotate(-45)">
        <line x1="0" y1="0" x2="0" y2="10" stroke="currentColor" strokeWidth="1.5" strokeOpacity="0.4" />
      </pattern>

      {/* Horizontal lines - for protectorates */}
      <pattern id="pattern-horizontal" patternUnits="userSpaceOnUse" width="10" height="6">
        <line x1="0" y1="3" x2="10" y2="3" stroke="currentColor" strokeWidth="1" strokeOpacity="0.4" />
      </pattern>
    </defs>
  );
}

// ============================================
// MILITARY SYMBOL COMPONENT (APP-6 STYLE)
// ============================================

interface MilitarySymbolProps {
  asset: MilitaryAsset;
  scale?: number;
  showLabel?: boolean;
  onClick?: (asset: MilitaryAsset) => void;
}

export function MilitarySymbol({ asset, scale = 1, showLabel = true, onClick }: MilitarySymbolProps) {
  const frameColor = DISPOSITION_COLORS[asset.disposition];
  const icon = UNIT_ICONS[asset.unitType];
  const sizeModifier = SIZE_MODIFIERS[asset.unitSize];
  const size = 24 * scale;

  // Frame shapes by disposition (NATO APP-6D standard)
  const frameShape = useMemo(() => {
    switch (asset.disposition) {
      case 'friendly':
        // Rectangle with rounded top
        return `M ${-size/2} ${-size/2} L ${size/2} ${-size/2} L ${size/2} ${size/2} L ${-size/2} ${size/2} Z`;
      case 'hostile':
        // Diamond
        return `M 0 ${-size/2} L ${size/2} 0 L 0 ${size/2} L ${-size/2} 0 Z`;
      case 'neutral':
        // Square
        return `M ${-size/2} ${-size/2} L ${size/2} ${-size/2} L ${size/2} ${size/2} L ${-size/2} ${size/2} Z`;
      case 'unknown':
        // Quatrefoil (cloverleaf)
        const r = size / 3;
        return `M 0 ${-size/2} Q ${size/3} ${-size/3} ${size/2} 0 Q ${size/3} ${size/3} 0 ${size/2} Q ${-size/3} ${size/3} ${-size/2} 0 Q ${-size/3} ${-size/3} 0 ${-size/2} Z`;
      default:
        return `M ${-size/2} ${-size/2} L ${size/2} ${-size/2} L ${size/2} ${size/2} L ${-size/2} ${size/2} Z`;
    }
  }, [asset.disposition, size]);

  // Echelon indicator position (above symbol)
  const echelonY = -size / 2 - 8;

  return (
    <g
      className="military-symbol cursor-pointer hover:opacity-80 transition-opacity"
      onClick={() => onClick?.(asset)}
      transform={`translate(${asset.position.lon}, ${asset.position.lat})`}
    >
      {/* Frame background */}
      <path
        d={frameShape}
        fill={`${frameColor}22`}
        stroke={frameColor}
        strokeWidth={2}
      />

      {/* Unit type icon */}
      <text
        x="0"
        y="4"
        textAnchor="middle"
        fill={frameColor}
        fontSize={size * 0.4}
        fontFamily="monospace"
        fontWeight="bold"
      >
        {icon}
      </text>

      {/* Echelon/size indicator */}
      {sizeModifier && (
        <text
          x="0"
          y={echelonY}
          textAnchor="middle"
          fill={frameColor}
          fontSize={size * 0.35}
          fontFamily="monospace"
          fontWeight="bold"
        >
          {sizeModifier}
        </text>
      )}

      {/* Unit designation label */}
      {showLabel && (
        <text
          x="0"
          y={size / 2 + 12}
          textAnchor="middle"
          fill="white"
          fontSize={10}
          fontFamily="sans-serif"
          className="drop-shadow-md"
        >
          {asset.designation}
        </text>
      )}

      {/* Threat level indicator (small dot) */}
      {asset.threatLevel !== 'none' && (
        <circle
          cx={size / 2 + 4}
          cy={-size / 2 - 4}
          r={4}
          fill={THREAT_COLORS[asset.threatLevel]}
          stroke="white"
          strokeWidth={1}
        />
      )}

      {/* Nuclear capable indicator */}
      {asset.capabilities.includes('nuclear') && (
        <circle
          cx={-size / 2 - 4}
          cy={-size / 2 - 4}
          r={4}
          fill="#FBBF24"
          stroke="black"
          strokeWidth={1}
        />
      )}
    </g>
  );
}

// ============================================
// TERRITORIAL OVERLAY COMPONENT
// ============================================

interface TerritorialOverlayProps {
  territory: TerritorialStatus;
  controllerColor: string;
  onClick?: (territory: TerritorialStatus) => void;
}

export function TerritorialOverlay({ territory, controllerColor, onClick }: TerritorialOverlayProps) {
  const style = SOVEREIGNTY_STYLES[territory.sovereigntyStatus];

  // Pattern ID based on sovereignty status
  const patternId = useMemo(() => {
    switch (style.fillPattern) {
      case 'stripes': return 'url(#pattern-stripes)';
      case 'dots': return 'url(#pattern-dots)';
      case 'crosshatch': return 'url(#pattern-crosshatch)';
      case 'diagonal': return 'url(#pattern-diagonal)';
      default: return undefined;
    }
  }, [style.fillPattern]);

  // Stroke dash pattern
  const strokeDash = useMemo(() => {
    switch (style.strokeStyle) {
      case 'dashed': return '8,4';
      case 'dotted': return '2,4';
      case 'dash-dot': return '8,4,2,4';
      default: return undefined;
    }
  }, [style.strokeStyle]);

  return (
    <g
      className="territorial-overlay cursor-pointer hover:brightness-110 transition-all"
      onClick={() => onClick?.(territory)}
    >
      {/* Base fill */}
      <rect
        x={territory.bounds.west}
        y={territory.bounds.south}
        width={territory.bounds.east - territory.bounds.west}
        height={territory.bounds.north - territory.bounds.south}
        fill={controllerColor}
        fillOpacity={style.fillOpacity || 0.2}
      />

      {/* Pattern overlay */}
      {patternId && (
        <rect
          x={territory.bounds.west}
          y={territory.bounds.south}
          width={territory.bounds.east - territory.bounds.west}
          height={territory.bounds.north - territory.bounds.south}
          fill={patternId}
          style={{ color: controllerColor }}
        />
      )}

      {/* Border */}
      <rect
        x={territory.bounds.west}
        y={territory.bounds.south}
        width={territory.bounds.east - territory.bounds.west}
        height={territory.bounds.north - territory.bounds.south}
        fill="none"
        stroke={territory.isContested ? THREAT_COLORS.high : controllerColor}
        strokeWidth={style.strokeWidth || 2}
        strokeDasharray={strokeDash}
      />

      {/* Contested indicator */}
      {territory.isContested && (
        <text
          x={(territory.bounds.east + territory.bounds.west) / 2}
          y={(territory.bounds.north + territory.bounds.south) / 2}
          textAnchor="middle"
          fill={THREAT_COLORS.critical}
          fontSize={14}
          fontWeight="bold"
          className="animate-pulse"
        >
          CONTESTED
        </text>
      )}

      {/* Territory name */}
      <text
        x={(territory.bounds.east + territory.bounds.west) / 2}
        y={territory.bounds.north + 15}
        textAnchor="middle"
        fill="white"
        fontSize={12}
        fontFamily="sans-serif"
        className="drop-shadow-lg"
      >
        {territory.name}
      </text>
    </g>
  );
}

// ============================================
// CONTESTED ZONE OVERLAY
// ============================================

interface ContestedZoneOverlayProps {
  zone: ContestedZone;
  onClick?: (zone: ContestedZone) => void;
}

export function ContestedZoneOverlay({ zone, onClick }: ContestedZoneOverlayProps) {
  const color = THREAT_COLORS[zone.intensity];

  // Render based on zone type
  if (zone.type === 'point') {
    const [lat, lon, radius = 50] = zone.coordinates as number[];
    return (
      <g onClick={() => onClick?.(zone)} className="cursor-pointer">
        <circle
          cx={lon}
          cy={lat}
          r={radius}
          fill={`${color}33`}
          stroke={color}
          strokeWidth={2}
          strokeDasharray="4,4"
          className="animate-pulse"
        />
        <text
          x={lon}
          y={lat}
          textAnchor="middle"
          fill={color}
          fontSize={10}
          fontWeight="bold"
        >
          {zone.name}
        </text>
      </g>
    );
  }

  if (zone.type === 'line') {
    const points = (zone.coordinates as number[][]).map(([lat, lon]) => `${lon},${lat}`).join(' ');
    return (
      <g onClick={() => onClick?.(zone)} className="cursor-pointer">
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth={3}
          strokeDasharray="8,4"
          className="animate-pulse"
        />
      </g>
    );
  }

  // Polygon
  const points = (zone.coordinates as number[][]).map(([lat, lon]) => `${lon},${lat}`).join(' ');
  return (
    <g onClick={() => onClick?.(zone)} className="cursor-pointer">
      <polygon
        points={points}
        fill="url(#pattern-crosshatch)"
        style={{ color }}
        fillOpacity={0.4}
        stroke={color}
        strokeWidth={2}
        strokeDasharray="6,3"
      />
      <text
        x={(zone.coordinates as number[][])[0]?.[1]}
        y={(zone.coordinates as number[][])[0]?.[0]}
        fill={color}
        fontSize={11}
        fontWeight="bold"
      >
        {zone.name}
      </text>
    </g>
  );
}

// ============================================
// LEGEND COMPONENT
// ============================================

interface LegendProps {
  showDispositions?: boolean;
  showThreats?: boolean;
  showSovereignty?: boolean;
  className?: string;
}

export function MapLegend({
  showDispositions = true,
  showThreats = true,
  showSovereignty = true,
  className = '',
}: LegendProps) {
  return (
    <div className={`bg-black/70 backdrop-blur-sm rounded-lg p-3 text-xs text-white ${className}`}>
      <div className="font-bold mb-2 text-sm border-b border-white/20 pb-1">MAP LEGEND</div>

      {showDispositions && (
        <div className="mb-3">
          <div className="text-gray-400 mb-1">Force Disposition</div>
          <div className="grid grid-cols-2 gap-1">
            {(Object.entries(DISPOSITION_COLORS) as [ForceDisposition, string][]).map(([key, color]) => (
              <div key={key} className="flex items-center gap-1">
                <div className="w-3 h-3 rounded" style={{ backgroundColor: color }} />
                <span className="capitalize">{key}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {showThreats && (
        <div className="mb-3">
          <div className="text-gray-400 mb-1">Threat Level</div>
          <div className="grid grid-cols-2 gap-1">
            {(Object.entries(THREAT_COLORS) as [ThreatLevel, string][]).map(([key, color]) => (
              <div key={key} className="flex items-center gap-1">
                <div className="w-3 h-3 rounded" style={{ backgroundColor: color }} />
                <span className="capitalize">{key.replace('_', ' ')}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {showSovereignty && (
        <div className="mb-2">
          <div className="text-gray-400 mb-1">Territorial Status</div>
          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <div className="w-6 h-3 border border-white" style={{ background: 'repeating-linear-gradient(45deg, transparent, transparent 2px, rgba(255,255,255,0.3) 2px, rgba(255,255,255,0.3) 4px)' }} />
              <span>Occupied</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-6 h-3 border border-white border-dashed" style={{ background: 'repeating-linear-gradient(-45deg, transparent, transparent 2px, rgba(255,255,255,0.2) 2px, rgba(255,255,255,0.2) 4px)' }} />
              <span>Disputed</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-6 h-3 border border-white" style={{ background: 'radial-gradient(circle, rgba(255,255,255,0.3) 1px, transparent 1px)', backgroundSize: '4px 4px' }} />
              <span>Unrecognized</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-6 h-3 border border-white" style={{ background: 'repeating-linear-gradient(45deg, transparent, transparent 2px, rgba(255,255,255,0.3) 2px, rgba(255,255,255,0.3) 4px), repeating-linear-gradient(-45deg, transparent, transparent 2px, rgba(255,255,255,0.3) 2px, rgba(255,255,255,0.3) 4px)' }} />
              <span>Annexed/Contested</span>
            </div>
          </div>
        </div>
      )}

      <div className="text-gray-500 text-[10px] mt-2 border-t border-white/20 pt-1">
        NATO APP-6D Standard Symbology
      </div>
    </div>
  );
}

// ============================================
// UNIT TYPE LEGEND
// ============================================

export function UnitTypeLegend({ className = '' }: { className?: string }) {
  const unitGroups = {
    'Ground Forces': ['infantry', 'armor', 'mechanized', 'artillery', 'rocket_artillery', 'air_defense'] as UnitType[],
    'Air Assets': ['fighter', 'strike', 'bomber', 'helicopter_attack', 'uav'] as UnitType[],
    'Naval': ['carrier', 'submarine', 'surface_combatant', 'amphibious'] as UnitType[],
    'Special': ['cyber', 'electronic_warfare', 'nuclear', 'special_operations'] as UnitType[],
  };

  return (
    <div className={`bg-black/70 backdrop-blur-sm rounded-lg p-3 text-xs text-white ${className}`}>
      <div className="font-bold mb-2 text-sm border-b border-white/20 pb-1">UNIT TYPES</div>

      {Object.entries(unitGroups).map(([group, types]) => (
        <div key={group} className="mb-2">
          <div className="text-gray-400 mb-1">{group}</div>
          <div className="grid grid-cols-2 gap-1">
            {types.map(type => (
              <div key={type} className="flex items-center gap-1">
                <span className="font-mono bg-blue-500/30 px-1 rounded text-[10px]">
                  {UNIT_ICONS[type]}
                </span>
                <span className="capitalize text-[10px]">{type.replace(/_/g, ' ')}</span>
              </div>
            ))}
          </div>
        </div>
      ))}

      <div className="mt-2 border-t border-white/20 pt-1">
        <div className="text-gray-400 mb-1">Unit Size</div>
        <div className="grid grid-cols-3 gap-1 text-[10px]">
          {(['company', 'battalion', 'brigade', 'division', 'corps'] as UnitSize[]).map(size => (
            <div key={size} className="flex items-center gap-1">
              <span className="font-mono">{SIZE_MODIFIERS[size]}</span>
              <span className="capitalize">{size}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ============================================
// TIME SLIDER FOR HISTORICAL VIEW
// ============================================

interface TimeSliderProps {
  startDate: Date;
  endDate: Date;
  currentDate: Date;
  onChange: (date: Date) => void;
  className?: string;
}

export function TimeSlider({ startDate, endDate, currentDate, onChange, className = '' }: TimeSliderProps) {
  const totalDays = Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));
  const currentDays = Math.ceil((currentDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const days = parseInt(e.target.value);
    const newDate = new Date(startDate.getTime() + days * 24 * 60 * 60 * 1000);
    onChange(newDate);
  };

  return (
    <div className={`bg-black/70 backdrop-blur-sm rounded-lg p-3 ${className}`}>
      <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
        <span>{startDate.toLocaleDateString()}</span>
        <span className="text-white font-bold">{currentDate.toLocaleDateString()}</span>
        <span>{endDate.toLocaleDateString()}</span>
      </div>

      <input
        type="range"
        min={0}
        max={totalDays}
        value={currentDays}
        onChange={handleChange}
        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
      />

      <div className="flex justify-center gap-2 mt-2">
        <button
          onClick={() => onChange(new Date(currentDate.getTime() - 24 * 60 * 60 * 1000))}
          className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded"
        >
          -1 Day
        </button>
        <button
          onClick={() => onChange(new Date(currentDate.getTime() - 7 * 24 * 60 * 60 * 1000))}
          className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded"
        >
          -1 Week
        </button>
        <button
          onClick={() => onChange(new Date())}
          className="px-2 py-1 text-xs bg-blue-600 hover:bg-blue-500 rounded"
        >
          Today
        </button>
        <button
          onClick={() => onChange(new Date(currentDate.getTime() + 7 * 24 * 60 * 60 * 1000))}
          className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded"
        >
          +1 Week
        </button>
        <button
          onClick={() => onChange(new Date(currentDate.getTime() + 24 * 60 * 60 * 1000))}
          className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded"
        >
          +1 Day
        </button>
      </div>
    </div>
  );
}

// ============================================
// INFO PANEL FOR SELECTED ITEM
// ============================================

interface InfoPanelProps {
  selectedAsset?: MilitaryAsset | null;
  selectedTerritory?: TerritorialStatus | null;
  selectedZone?: ContestedZone | null;
  onClose: () => void;
  className?: string;
}

export function InfoPanel({
  selectedAsset,
  selectedTerritory,
  selectedZone,
  onClose,
  className = '',
}: InfoPanelProps) {
  if (!selectedAsset && !selectedTerritory && !selectedZone) return null;

  return (
    <div className={`bg-black/80 backdrop-blur-sm rounded-lg p-4 text-white max-w-sm ${className}`}>
      <button
        onClick={onClose}
        className="absolute top-2 right-2 text-gray-400 hover:text-white"
      >
        &times;
      </button>

      {selectedAsset && (
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span
              className="w-4 h-4 rounded"
              style={{ backgroundColor: DISPOSITION_COLORS[selectedAsset.disposition] }}
            />
            <span className="font-bold text-lg">{selectedAsset.designation}</span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-gray-400">Type:</span>
              <span className="ml-1 capitalize">{selectedAsset.unitType.replace(/_/g, ' ')}</span>
            </div>
            <div>
              <span className="text-gray-400">Size:</span>
              <span className="ml-1 capitalize">{selectedAsset.unitSize}</span>
            </div>
            <div>
              <span className="text-gray-400">Strength:</span>
              <span className="ml-1">{selectedAsset.strength.toLocaleString()}</span>
            </div>
            <div>
              <span className="text-gray-400">Readiness:</span>
              <span className="ml-1">{Math.round(selectedAsset.readiness * 100)}%</span>
            </div>
            <div>
              <span className="text-gray-400">Threat:</span>
              <span className="ml-1 capitalize" style={{ color: THREAT_COLORS[selectedAsset.threatLevel] }}>
                {selectedAsset.threatLevel}
              </span>
            </div>
            <div>
              <span className="text-gray-400">Confidence:</span>
              <span className="ml-1">{Math.round(selectedAsset.confidence * 100)}%</span>
            </div>
          </div>

          {selectedAsset.capabilities.length > 0 && (
            <div className="mt-2">
              <span className="text-gray-400 text-sm">Capabilities:</span>
              <div className="flex flex-wrap gap-1 mt-1">
                {selectedAsset.capabilities.map(cap => (
                  <span key={cap} className="px-2 py-0.5 bg-gray-700 rounded text-xs capitalize">
                    {cap.replace(/_/g, ' ')}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {selectedTerritory && (
        <div>
          <div className="font-bold text-lg mb-2">{selectedTerritory.name}</div>

          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Legal Sovereign:</span>
              <span>{selectedTerritory.legalSovereignName}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">De Facto Control:</span>
              <span>{selectedTerritory.deFactoControllerName}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Status:</span>
              <span className="capitalize">{selectedTerritory.sovereigntyStatus}</span>
            </div>

            {selectedTerritory.isDisputed && (
              <div className="mt-2 p-2 bg-yellow-900/30 rounded">
                <div className="text-yellow-400 font-bold text-xs mb-1">DISPUTED TERRITORY</div>
                {selectedTerritory.claims.map((claim, i) => (
                  <div key={i} className="text-xs">
                    <span className="font-bold">{claim.claimantName}:</span>{' '}
                    {Math.round(claim.internationalSupport * 100)}% recognition,{' '}
                    {claim.controlPercentage}% control
                  </div>
                ))}
              </div>
            )}

            {selectedTerritory.isContested && (
              <div className="text-red-400 font-bold text-xs animate-pulse">
                ACTIVE CONFLICT
              </div>
            )}
          </div>
        </div>
      )}

      {selectedZone && (
        <div>
          <div className="font-bold text-lg mb-2">{selectedZone.name}</div>

          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Type:</span>
              <span className="capitalize">{selectedZone.contestType.replace(/_/g, ' ')}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Intensity:</span>
              <span className="capitalize" style={{ color: THREAT_COLORS[selectedZone.intensity] }}>
                {selectedZone.intensity}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Casualty Rate:</span>
              <span className="capitalize">{selectedZone.casualtyRate}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Civilian Impact:</span>
              <span className="capitalize">{selectedZone.civilianImpact}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Started:</span>
              <span>{new Date(selectedZone.startDate).toLocaleDateString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Parties:</span>
              <span>{selectedZone.primaryParties.join(', ')}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================
// MAIN COMPOSITE OVERLAY LAYER
// ============================================

interface MilitaryOverlayLayerProps {
  territories?: TerritorialStatus[];
  assets?: MilitaryAsset[];
  zones?: ContestedZone[];
  asOfDate?: string;
  nationColors?: Record<string, string>;
  onAssetClick?: (asset: MilitaryAsset) => void;
  onTerritoryClick?: (territory: TerritorialStatus) => void;
  onZoneClick?: (zone: ContestedZone) => void;
}

export function MilitaryOverlayLayer({
  territories = [],
  assets = [],
  zones = [],
  asOfDate = new Date().toISOString(),
  nationColors = {},
  onAssetClick,
  onTerritoryClick,
  onZoneClick,
}: MilitaryOverlayLayerProps) {
  // Filter by date if needed
  const filteredTerritories = useMemo(() => {
    const date = new Date(asOfDate);
    return territories.filter(t => {
      const effective = new Date(t.effectiveDate);
      const end = t.endDate ? new Date(t.endDate) : new Date();
      return date >= effective && date <= end;
    });
  }, [territories, asOfDate]);

  return (
    <g className="military-overlay-layer">
      <PatternDefs />

      {/* Territorial overlays (background) */}
      {filteredTerritories.map(territory => (
        <TerritorialOverlay
          key={territory.id}
          territory={territory}
          controllerColor={nationColors[territory.deFactoController] || '#888888'}
          onClick={onTerritoryClick}
        />
      ))}

      {/* Contested zones (middle) */}
      {zones.map(zone => (
        <ContestedZoneOverlay
          key={zone.id}
          zone={zone}
          onClick={onZoneClick}
        />
      ))}

      {/* Military assets (foreground) */}
      {assets.map(asset => (
        <MilitarySymbol
          key={asset.id}
          asset={asset}
          onClick={onAssetClick}
        />
      ))}
    </g>
  );
}
