'use client';

import { useEffect, useRef, useState } from 'react';
import { Loader } from '@googlemaps/js-api-loader';
import { Nation, InfluenceEdge, MapLayer, REGIMES } from '@/types';

interface AttractorMapProps {
  nations: Nation[];
  edges: InfluenceEdge[];
  layer: MapLayer;
  onNationSelect?: (nation: Nation) => void;
}

// Color scales for different layers
const BASIN_COLORS = [
  { threshold: 0.2, color: '#EF4444' }, // Low stability - red
  { threshold: 0.4, color: '#F59E0B' }, // Medium-low - orange
  { threshold: 0.6, color: '#FBBF24' }, // Medium - yellow
  { threshold: 0.8, color: '#84CC16' }, // Medium-high - lime
  { threshold: 1.0, color: '#10B981' }, // High stability - green
];

const RISK_COLORS = [
  { threshold: 0.2, color: '#10B981' }, // Low risk - green
  { threshold: 0.4, color: '#84CC16' },
  { threshold: 0.6, color: '#FBBF24' },
  { threshold: 0.8, color: '#F59E0B' },
  { threshold: 1.0, color: '#EF4444' }, // High risk - red
];

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
        // Use cloud-based styling via Map ID (configured in Google Cloud Console)
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
        mapTypeControl: false,
        streetViewControl: false,
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
      // Determine color based on layer
      let color: string;
      let size: number;

      switch (layer) {
        case 'basin':
          color = getColorForValue(nation.basin_strength, BASIN_COLORS);
          size = 12 + nation.basin_strength * 8;
          break;
        case 'risk':
          color = getColorForValue(nation.transition_risk, RISK_COLORS);
          size = 12 + nation.transition_risk * 12;
          break;
        case 'regime':
          color = REGIMES[nation.regime]?.color || '#6B7280';
          size = 16;
          break;
        case 'influence':
          color = '#3B82F6';
          size = 12 + nation.influence_radius * 4;
          break;
        default:
          color = '#3B82F6';
          size = 14;
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

      // Click handler
      marker.addListener('click', () => {
        if (infoWindow) {
          // Convert basin strength to human-readable stability description
          const stabilityLevel = nation.basin_strength >= 0.8 ? 'Very High' :
            nation.basin_strength >= 0.6 ? 'High' :
            nation.basin_strength >= 0.4 ? 'Moderate' :
            nation.basin_strength >= 0.2 ? 'Low' : 'Critical';

          const stabilityDesc = nation.basin_strength >= 0.8 ? 'Highly resilient institutions with strong governance' :
            nation.basin_strength >= 0.6 ? 'Stable governance with good institutional capacity' :
            nation.basin_strength >= 0.4 ? 'Functioning institutions with some vulnerabilities' :
            nation.basin_strength >= 0.2 ? 'Weakened institutions facing significant pressures' :
            'Fragile state with severe institutional breakdown';

          // Convert risk to human-readable assessment
          const riskLevel = nation.transition_risk >= 0.8 ? 'Critical' :
            nation.transition_risk >= 0.6 ? 'High' :
            nation.transition_risk >= 0.4 ? 'Elevated' :
            nation.transition_risk >= 0.2 ? 'Low' : 'Minimal';

          const riskDesc = nation.transition_risk >= 0.8 ? 'Immediate risk of major political upheaval' :
            nation.transition_risk >= 0.6 ? 'Significant likelihood of destabilizing events' :
            nation.transition_risk >= 0.4 ? 'Moderate chance of political volatility' :
            nation.transition_risk >= 0.2 ? 'Minor instability indicators present' :
            'Stable outlook with no significant risks detected';

          const content = `
            <div style="color: #0f172a; padding: 12px; min-width: 260px; font-family: system-ui, sans-serif;">
              <h3 style="margin: 0 0 12px; font-weight: 600; font-size: 16px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;">
                ${nation.name}
              </h3>
              <div style="font-size: 13px; line-height: 1.7;">
                <div style="margin-bottom: 12px;">
                  <div style="font-weight: 500; color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">
                    Stability Assessment
                  </div>
                  <div style="color: ${getColorForValue(nation.basin_strength, BASIN_COLORS)}; font-weight: 600;">
                    ${stabilityLevel}
                  </div>
                  <div style="color: #475569; font-size: 12px;">${stabilityDesc}</div>
                </div>
                <div style="margin-bottom: 12px;">
                  <div style="font-weight: 500; color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">
                    Transition Risk
                  </div>
                  <div style="color: ${getColorForValue(nation.transition_risk, RISK_COLORS)}; font-weight: 600;">
                    ${riskLevel}
                  </div>
                  <div style="color: #475569; font-size: 12px;">${riskDesc}</div>
                </div>
                <div>
                  <div style="font-weight: 500; color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">
                    Government Type
                  </div>
                  <div style="color: ${REGIMES[nation.regime]?.color || '#6B7280'}; font-weight: 600;">
                    ${REGIMES[nation.regime]?.name || 'Unknown'}
                  </div>
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
  }, [map, nations, layer, infoWindow, onNationSelect]);

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

  return <div ref={mapRef} className="map-container w-full h-full" />;
}
