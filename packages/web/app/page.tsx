'use client';

import { useState, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import { useWasm } from '@/hooks/useWasm';
import { useSupabaseNations } from '@/hooks/useSupabaseNations';
import ControlPanel from '@/components/ControlPanel';
import AlertBanner from '@/components/AlertBanner';
import type { WasmGeospatialSystem } from '@/lib/wasm';

// Dynamic import for Google Maps (no SSR)
const AttractorMap = dynamic(() => import('@/components/AttractorMap'), {
  ssr: false,
  loading: () => <div className="wasm-loader" />,
});

export default function Home() {
  const { wasm, loading: wasmLoading, error: wasmError } = useWasm();
  const { nations, edges, loading: dataLoading } = useSupabaseNations();
  const [layer, setLayer] = useState<'basin' | 'risk' | 'influence' | 'regime'>('basin');
  const [alertLevel, setAlertLevel] = useState<string>('normal');
  const [isSimulating, setIsSimulating] = useState(false);
  const geoSystemRef = useRef<WasmGeospatialSystem | null>(null);

  // Initialize geospatial system when WASM and nations are ready
  useEffect(() => {
    if (!wasm || nations.length === 0 || geoSystemRef.current) return;

    try {
      // Create geospatial system with 3 dimensions for attractor space
      const geoSystem = new wasm.WasmGeospatialSystem(3);

      // Add nations to the system
      nations.forEach((nation) => {
        const pos = new Float64Array(nation.position ?? [0, 0, 0]);
        geoSystem.add_nation(
          nation.code,
          nation.name,
          nation.lat,
          nation.lon,
          pos,
          nation.regime ?? 0
        );
      });

      // Add edges (esteem relationships)
      edges.forEach((edge) => {
        geoSystem.set_esteem(edge.source_code, edge.target_code, edge.esteem);
      });

      geoSystemRef.current = geoSystem;
    } catch (err) {
      console.error('Failed to initialize geospatial system:', err);
    }
  }, [wasm, nations, edges]);

  // Run simulation step
  const runStep = async () => {
    const geoSystem = geoSystemRef.current;
    if (!geoSystem) return;

    setIsSimulating(true);
    try {
      // Run one step of the simulation
      geoSystem.step();

      // Get GeoJSON for risk assessment
      const riskJson = geoSystem.to_geojson('risk');
      const riskData = JSON.parse(riskJson);

      // Calculate max transition risk from features
      const maxRisk = riskData.features?.reduce((max: number, f: { properties?: { risk?: number } }) => {
        return Math.max(max, f.properties?.risk ?? 0);
      }, 0) ?? 0;

      // Update alert level based on risk
      if (maxRisk > 0.75) {
        setAlertLevel('critical');
      } else if (maxRisk > 0.5) {
        setAlertLevel('warning');
      } else if (maxRisk > 0.25) {
        setAlertLevel('elevated');
      } else {
        setAlertLevel('normal');
      }
    } catch (err) {
      console.error('Simulation step failed:', err);
    } finally {
      setIsSimulating(false);
    }
  };

  if (wasmLoading || dataLoading) {
    return (
      <div className="wasm-loader">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-slate-400">
            {wasmLoading ? 'Loading WASM core...' : 'Fetching nation data...'}
          </p>
        </div>
      </div>
    );
  }

  if (wasmError) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center text-red-400">
          <p className="text-xl mb-2">Failed to load WASM</p>
          <p className="text-sm text-slate-500">{wasmError.message}</p>
        </div>
      </div>
    );
  }

  return (
    <main className="relative h-screen w-full">
      {/* Alert Banner */}
      <AlertBanner level={alertLevel} />

      {/* Map */}
      <AttractorMap
        nations={nations}
        edges={edges}
        layer={layer}
      />

      {/* Control Panel */}
      <ControlPanel
        layer={layer}
        onLayerChange={setLayer}
        onStep={runStep}
        isSimulating={isSimulating}
        alertLevel={alertLevel}
      />

      {/* Info Panel (bottom right) */}
      <div className="absolute bottom-4 right-4 bg-slate-900/90 backdrop-blur-sm rounded-lg p-3 text-xs text-slate-400">
        <p>{nations.length} nations tracked</p>
        <p>{edges.length} influence edges</p>
        <p className="text-blue-400">Powered by Rust/WASM</p>
      </div>
    </main>
  );
}
