'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { Loader } from '@googlemaps/js-api-loader';
import { Nation, InfluenceEdge, MapLayer, REGIMES } from '@/types';
import { useAccessibility } from '@/contexts/AccessibilityContext';
import { Globe, Map as MapIcon, Mountain, Satellite } from 'lucide-react';

// Map type options
type MapTypeOption = 'hybrid' | 'satellite' | 'roadmap' | 'terrain';

interface MapTypeConfig {
  id: MapTypeOption;
  label: string;
  icon: typeof Satellite;
  description: string;
}

const MAP_TYPES: MapTypeConfig[] = [
  { id: 'hybrid', label: 'Satellite', icon: Satellite, description: 'Satellite imagery with labels' },
  { id: 'satellite', label: 'Satellite Only', icon: Globe, description: 'Pure satellite imagery' },
  { id: 'roadmap', label: 'Map', icon: MapIcon, description: 'Standard road map' },
  { id: 'terrain', label: 'Terrain', icon: Mountain, description: 'Topographic relief map' },
];

interface AttractorMapProps {
  nations: Nation[];
  edges: InfluenceEdge[];
  layer: MapLayer;
  onNationSelect?: (nation: Nation) => void;
}

// Color scales for different layers
// DISTINCT COLORS - easy to tell apart even at a glance
const BASIN_COLORS = [
  { threshold: 0.15, color: '#DC2626' }, // Critical - bright red
  { threshold: 0.30, color: '#EA580C' }, // Unstable - orange
  { threshold: 0.45, color: '#CA8A04' }, // Moderate - gold/yellow
  { threshold: 0.60, color: '#0891B2' }, // Stable - cyan/teal (distinct from green)
  { threshold: 1.0, color: '#059669' },  // Very stable - emerald green
];

const RISK_COLORS = [
  { threshold: 0.2, color: '#059669' },  // Very low risk - emerald green
  { threshold: 0.4, color: '#0891B2' },  // Low risk - cyan/teal
  { threshold: 0.6, color: '#CA8A04' },  // Moderate - gold/yellow
  { threshold: 0.8, color: '#EA580C' },  // High - orange
  { threshold: 1.0, color: '#DC2626' },  // Critical - bright red
];

// COLORBLIND-FRIENDLY PALETTE (Deuteranopia/Protanopia/Tritanopia safe)
// Uses shapes + patterns in addition to color
const COLORBLIND_BASIN_COLORS = [
  { threshold: 0.15, color: '#CC3311' }, // Critical - vermillion red
  { threshold: 0.30, color: '#EE7733' }, // Unstable - orange
  { threshold: 0.45, color: '#CCBB44' }, // Moderate - yellow
  { threshold: 0.60, color: '#33BBEE' }, // Stable - sky blue
  { threshold: 1.0, color: '#0077BB' },  // Very stable - strong blue
];

const COLORBLIND_RISK_COLORS = [
  { threshold: 0.2, color: '#0077BB' },  // Very low - strong blue
  { threshold: 0.4, color: '#33BBEE' },  // Low - sky blue
  { threshold: 0.6, color: '#CCBB44' },  // Moderate - yellow
  { threshold: 0.8, color: '#EE7733' },  // High - orange
  { threshold: 1.0, color: '#CC3311' },  // Critical - vermillion red
];

// Stale data color - dark gray/black for data older than 24 hours
const STALE_DATA_COLOR = '#1F2937';  // slate-800
const NO_DATA_COLOR = '#111827';     // slate-900 (almost black)

// Helper to check if data is stale (>24 hours old)
function isDataStale(updatedAt?: string): boolean {
  if (!updatedAt) return true;  // No timestamp = stale
  const updated = new Date(updatedAt);
  const now = new Date();
  const hoursDiff = (now.getTime() - updated.getTime()) / (1000 * 60 * 60);
  return hoursDiff > 24;
}

// Get hours since last update
function getHoursSinceUpdate(updatedAt?: string): number | null {
  if (!updatedAt) return null;
  const updated = new Date(updatedAt);
  const now = new Date();
  return Math.round((now.getTime() - updated.getTime()) / (1000 * 60 * 60));
}

function getColorForValue(value: number, scale: typeof BASIN_COLORS): string {
  for (const { threshold, color } of scale) {
    if (value <= threshold) return color;
  }
  return scale[scale.length - 1].color;
}

export default function AttractorMap({ nations, edges, layer, onNationSelect }: AttractorMapProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const [map, setMap] = useState<google.maps.Map | null>(null);
  const [markers, setMarkers] = useState<google.maps.marker.AdvancedMarkerElement[]>([]);
  const [polylines, setPolylines] = useState<google.maps.Polyline[]>([]);
  const [infoWindow, setInfoWindow] = useState<google.maps.InfoWindow | null>(null);
  const [mapType, setMapType] = useState<MapTypeOption>('hybrid'); // Satellite with labels by default
  const [showMapTypeMenu, setShowMapTypeMenu] = useState(false);

  // Handle map type change
  const handleMapTypeChange = useCallback((typeId: MapTypeOption) => {
    if (map) {
      map.setMapTypeId(typeId);
      setMapType(typeId);
      setShowMapTypeMenu(false);
    }
  }, [map]);

  // Get accessibility settings for colorblind mode
  let accessibilitySettings;
  try {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-call
    accessibilitySettings = useAccessibility();
  } catch {
    // Context not available (e.g., in dashboard outside consumer app)
    accessibilitySettings = { isColorblindMode: false, settings: { colorblindMode: 'none' } };
  }
  const isColorblind = accessibilitySettings.isColorblindMode;

  // Select color palettes based on accessibility mode
  const basinColors = isColorblind ? COLORBLIND_BASIN_COLORS : BASIN_COLORS;
  const riskColors = isColorblind ? COLORBLIND_RISK_COLORS : RISK_COLORS;

  // Initialize map
  useEffect(() => {
    const loader = new Loader({
      apiKey: process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY || '',
      version: 'weekly',
      libraries: ['marker'],
    });

    void loader.load().then(() => {
      if (!mapRef.current) return;

      const mapInstance = new google.maps.Map(mapRef.current, {
        center: { lat: 20, lng: 0 },
        zoom: 2,
        minZoom: 2,
        maxZoom: 10,
        // Default to satellite/hybrid view
        mapTypeId: 'hybrid',
        // Use cloud-based styling via Map ID (configured in Google Cloud Console)
        // Note: Map ID styling only applies to roadmap, not satellite
        mapId: process.env.NEXT_PUBLIC_GOOGLE_MAPS_MAP_ID,
        restriction: {
          latLngBounds: {
            north: 85,
            south: -85,
            west: -180,
            east: 180,
          },
          strictBounds: true,
        },
        disableDefaultUI: true,
        zoomControl: true,
        mapTypeControl: false, // We provide our own control
        streetViewControl: false,
        // Enable tilt for 3D effect on satellite
        tilt: 0,
        // Allow rotation
        rotateControl: true,
      });

      setMap(mapInstance);
      setInfoWindow(new google.maps.InfoWindow());
    });

    return () => {
      // Cleanup markers and polylines
      markers.forEach((m) => (m.map = null));
      polylines.forEach((p) => p.setMap(null));
    };
  }, []);

  // Update markers when nations or layer changes
  useEffect(() => {
    if (!map) return;

    // Clear existing markers
    markers.forEach((m) => (m.map = null));

    const newMarkers: google.maps.marker.AdvancedMarkerElement[] = [];

    nations.forEach((nation) => {
      // Check for stale data first (>24 hours old = dark gray/black)
      const stale = isDataStale(nation.updated_at);

      // Determine color based on layer (using accessibility-aware palettes)
      let color: string;
      let size: number;

      // If data is stale, show dark indicator regardless of layer
      if (stale) {
        color = nation.updated_at ? STALE_DATA_COLOR : NO_DATA_COLOR;
        size = 14;
      } else {
        switch (layer) {
          case 'basin':
            color = getColorForValue(nation.basin_strength, basinColors);
            size = 12 + nation.basin_strength * 8;
            break;
          case 'risk':
            color = getColorForValue(nation.transition_risk, riskColors);
            size = 12 + nation.transition_risk * 12;
            break;
          case 'regime':
            color = REGIMES[nation.regime]?.color || '#6B7280';
            size = 16;
            break;
          case 'influence':
            color = isColorblind ? '#0077BB' : '#3B82F6';
            size = 12 + nation.influence_radius * 4;
            break;
          default:
            color = isColorblind ? '#0077BB' : '#3B82F6';
            size = 14;
        }
      }

      // Create marker element
      const markerEl = document.createElement('div');
      markerEl.className = 'nation-marker';
      markerEl.style.width = `${size}px`;
      markerEl.style.height = `${size}px`;
      markerEl.style.backgroundColor = color;
      markerEl.style.borderRadius = '50%';
      markerEl.style.border = '2px solid white';
      markerEl.style.boxShadow = `0 0 ${size}px ${color}80`;
      markerEl.style.cursor = 'pointer';
      markerEl.title = nation.name;

      const marker = new google.maps.marker.AdvancedMarkerElement({
        map,
        position: { lat: nation.lat, lng: nation.lon },
        content: markerEl,
        title: nation.name,
      });

      // Click handler - LAYMAN-FRIENDLY tooltips
      marker.addListener('click', () => {
        if (infoWindow) {
          // Check data freshness
          const hoursSinceUpdate = getHoursSinceUpdate(nation.updated_at);
          const isStale = stale;

          // Simple stability labels (adjusted for lowered thresholds)
          const stabilityLevel = nation.basin_strength >= 0.60 ? 'Very Stable' :
            nation.basin_strength >= 0.45 ? 'Stable' :
            nation.basin_strength >= 0.30 ? 'Some Concerns' :
            nation.basin_strength >= 0.15 ? 'Unstable' : 'In Crisis';

          // Plain English descriptions
          const stabilityDesc = nation.basin_strength >= 0.60 ? 'Strong government, economy working well' :
            nation.basin_strength >= 0.45 ? 'Generally doing okay, minor issues' :
            nation.basin_strength >= 0.30 ? 'Having some problems to watch' :
            nation.basin_strength >= 0.15 ? 'Facing serious challenges' :
            'Major crisis or breakdown happening';

          // Simple risk labels
          const riskLevel = nation.transition_risk >= 0.7 ? 'High Risk' :
            nation.transition_risk >= 0.5 ? 'Elevated' :
            nation.transition_risk >= 0.3 ? 'Moderate' :
            nation.transition_risk >= 0.15 ? 'Low' : 'Very Low';

          // Plain English risk descriptions
          const riskDesc = nation.transition_risk >= 0.7 ? 'Big changes likely soon' :
            nation.transition_risk >= 0.5 ? 'Situation could shift' :
            nation.transition_risk >= 0.3 ? 'Worth keeping an eye on' :
            nation.transition_risk >= 0.15 ? 'Mostly stable outlook' :
            'No major changes expected';

          // Stale data warning banner
          const staleWarning = isStale ? `
            <div style="background: #1F2937; color: #F59E0B; padding: 8px 10px; margin: -14px -14px 14px -14px; border-radius: 4px 4px 0 0; font-size: 12px; font-weight: 600;">
              ⚠️ ${hoursSinceUpdate === null ? 'No data available' : `Data is ${hoursSinceUpdate}+ hours old`} - may be outdated
            </div>
          ` : '';

          // Last updated info
          const lastUpdated = nation.updated_at
            ? new Date(nation.updated_at).toLocaleString()
            : 'Never';

          const content = `
            <div style="color: #0f172a; padding: 14px; min-width: 260px; font-family: system-ui, sans-serif;">
              ${staleWarning}
              <h3 style="margin: 0 0 14px; font-weight: 700; font-size: 18px; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px;">
                ${nation.name}
              </h3>
              <div style="font-size: 14px; line-height: 1.6;">
                <div style="margin-bottom: 14px; ${isStale ? 'opacity: 0.6;' : ''}">
                  <div style="font-weight: 600; color: #374151; font-size: 13px; margin-bottom: 4px;">
                    How Stable?
                  </div>
                  <div style="color: ${isStale ? '#6B7280' : getColorForValue(nation.basin_strength, basinColors)}; font-weight: 700; font-size: 16px;">
                    ${isStale ? 'Unknown' : stabilityLevel}
                  </div>
                  <div style="color: #6b7280; font-size: 13px; margin-top: 2px;">${isStale ? 'Data too old to assess' : stabilityDesc}</div>
                </div>
                <div style="margin-bottom: 14px; ${isStale ? 'opacity: 0.6;' : ''}">
                  <div style="font-weight: 600; color: #374151; font-size: 13px; margin-bottom: 4px;">
                    Chance of Change?
                  </div>
                  <div style="color: ${isStale ? '#6B7280' : getColorForValue(nation.transition_risk, riskColors)}; font-weight: 700; font-size: 16px;">
                    ${isStale ? 'Unknown' : riskLevel}
                  </div>
                  <div style="color: #6b7280; font-size: 13px; margin-top: 2px;">${isStale ? 'Data too old to assess' : riskDesc}</div>
                </div>
                <div>
                  <div style="font-weight: 600; color: #374151; font-size: 13px; margin-bottom: 4px;">
                    Government Type
                  </div>
                  <div style="color: ${REGIMES[nation.regime]?.color || '#6B7280'}; font-weight: 600; font-size: 15px;">
                    ${REGIMES[nation.regime]?.name || 'Unknown'}
                  </div>
                </div>
                <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #e2e8f0; font-size: 11px; color: #9CA3AF;">
                  Last updated: ${lastUpdated}
                </div>
              </div>
            </div>
          `;
          infoWindow.setContent(content);
          infoWindow.open(map, marker);
        }
        onNationSelect?.(nation);
      });

      newMarkers.push(marker);
    });

    setMarkers(newMarkers);
  }, [map, nations, layer, infoWindow, onNationSelect, basinColors, riskColors, isColorblind]);

  // Update influence edges (polylines)
  useEffect(() => {
    if (!map || layer !== 'influence') return;

    // Clear existing polylines
    polylines.forEach((p) => p.setMap(null));

    const newPolylines: google.maps.Polyline[] = [];

    // Create nation lookup
    const nationLookup = new Map(nations.map((n) => [n.code, n]));

    edges.forEach((edge) => {
      const source = nationLookup.get(edge.source_code);
      const target = nationLookup.get(edge.target_code);

      if (!source || !target) return;

      // Color based on esteem
      const hue = edge.esteem > 0 ? 120 : 0; // Green for positive, red for negative
      const saturation = Math.abs(edge.esteem) * 100;
      const color = `hsl(${hue}, ${saturation}%, 50%)`;

      const polyline = new google.maps.Polyline({
        path: [
          { lat: source.lat, lng: source.lon },
          { lat: target.lat, lng: target.lon },
        ],
        geodesic: true,
        strokeColor: color,
        strokeOpacity: Math.min(edge.strength * 2, 0.8),
        strokeWeight: 1 + edge.strength * 3,
        map,
      });

      newPolylines.push(polyline);
    });

    setPolylines(newPolylines);

    return () => {
      newPolylines.forEach((p) => p.setMap(null));
    };
  }, [map, edges, layer, nations]);

  // Clear polylines when not in influence layer
  useEffect(() => {
    if (layer !== 'influence') {
      polylines.forEach((p) => p.setMap(null));
      setPolylines([]);
    }
  }, [layer]);

  return (
    <div className="relative w-full h-full">
      <div ref={mapRef} className="map-container w-full h-full" />

      {/* Map Type Selector */}
      <div className="absolute top-4 right-4 z-10">
        <div className="relative">
          {/* Toggle Button */}
          <button
            onClick={() => setShowMapTypeMenu(!showMapTypeMenu)}
            className="flex items-center gap-2 px-3 py-2 bg-slate-900/90 hover:bg-slate-800 text-white rounded-lg shadow-lg border border-slate-700 backdrop-blur-sm transition-colors"
            title="Change map type"
          >
            {(() => {
              const current = MAP_TYPES.find(t => t.id === mapType);
              const Icon = current?.icon || Satellite;
              return (
                <>
                  <Icon className="w-4 h-4" />
                  <span className="text-sm font-medium">{current?.label || 'Satellite'}</span>
                  <svg className={`w-4 h-4 transition-transform ${showMapTypeMenu ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </>
              );
            })()}
          </button>

          {/* Dropdown Menu */}
          {showMapTypeMenu && (
            <div className="absolute top-full right-0 mt-2 w-56 bg-slate-900/95 rounded-lg shadow-xl border border-slate-700 backdrop-blur-sm overflow-hidden">
              {MAP_TYPES.map((type) => {
                const Icon = type.icon;
                const isActive = mapType === type.id;
                return (
                  <button
                    key={type.id}
                    onClick={() => handleMapTypeChange(type.id)}
                    className={`w-full flex items-center gap-3 px-4 py-3 text-left transition-colors ${
                      isActive
                        ? 'bg-blue-600 text-white'
                        : 'text-slate-200 hover:bg-slate-800'
                    }`}
                  >
                    <Icon className="w-5 h-5 flex-shrink-0" />
                    <div>
                      <div className="text-sm font-medium">{type.label}</div>
                      <div className={`text-xs ${isActive ? 'text-blue-200' : 'text-slate-400'}`}>
                        {type.description}
                      </div>
                    </div>
                    {isActive && (
                      <svg className="w-4 h-4 ml-auto" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Click outside to close menu */}
      {showMapTypeMenu && (
        <div
          className="fixed inset-0 z-0"
          onClick={() => setShowMapTypeMenu(false)}
        />
      )}
    </div>
  );
}
