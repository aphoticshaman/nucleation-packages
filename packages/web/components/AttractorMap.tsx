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
  { threshold: 0.2, color: '#EF4444' },  // Low stability - red
  { threshold: 0.4, color: '#F59E0B' },  // Medium-low - orange
  { threshold: 0.6, color: '#FBBF24' },  // Medium - yellow
  { threshold: 0.8, color: '#84CC16' },  // Medium-high - lime
  { threshold: 1.0, color: '#10B981' },  // High stability - green
];

const RISK_COLORS = [
  { threshold: 0.2, color: '#10B981' },  // Low risk - green
  { threshold: 0.4, color: '#84CC16' },
  { threshold: 0.6, color: '#FBBF24' },
  { threshold: 0.8, color: '#F59E0B' },
  { threshold: 1.0, color: '#EF4444' },  // High risk - red
];

function getColorForValue(value: number, scale: typeof BASIN_COLORS): string {
  for (const { threshold, color } of scale) {
    if (value <= threshold) return color;
  }
  return scale[scale.length - 1].color;
}

export default function AttractorMap({
  nations,
  edges,
  layer,
  onNationSelect,
}: AttractorMapProps) {
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
        // Dark theme styles (can't use with mapId)
        styles: [
          { elementType: 'geometry', stylers: [{ color: '#1e293b' }] },
          { elementType: 'labels.text.stroke', stylers: [{ color: '#1e293b' }] },
          { elementType: 'labels.text.fill', stylers: [{ color: '#94a3b8' }] },
          {
            featureType: 'water',
            elementType: 'geometry',
            stylers: [{ color: '#0f172a' }],
          },
          {
            featureType: 'administrative.country',
            elementType: 'geometry.stroke',
            stylers: [{ color: '#475569' }],
          },
          {
            featureType: 'poi',
            stylers: [{ visibility: 'off' }],
          },
          {
            featureType: 'transit',
            stylers: [{ visibility: 'off' }],
          },
        ],
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
        backgroundColor: '#0f172a',
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
          const content = `
            <div style="color: #0f172a; padding: 8px; min-width: 200px;">
              <h3 style="margin: 0 0 8px; font-weight: bold;">${nation.name} (${nation.code})</h3>
              <div style="font-size: 12px; line-height: 1.6;">
                <div><strong>Basin Strength:</strong> ${(nation.basin_strength * 100).toFixed(1)}%</div>
                <div><strong>Transition Risk:</strong> ${(nation.transition_risk * 100).toFixed(1)}%</div>
                <div><strong>Regime:</strong> ${REGIMES[nation.regime]?.name || 'Unknown'}</div>
                <div><strong>Position:</strong> [${nation.position.map((p) => p.toFixed(2)).join(', ')}]</div>
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

  return (
    <div ref={mapRef} className="map-container w-full h-full" />
  );
}
