'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { useWasm } from '@/hooks/useWasm';
import { useSupabaseNations } from '@/hooks/useSupabaseNations';
import { useAlertLevel, type NationState } from '@/hooks/useAlertLevel';
import ControlPanel from '@/components/ControlPanel';
import AlertBanner from '@/components/AlertBanner';
import type { WasmGeospatialSystem } from '@/public/wasm/latticeforge_core';

// Dynamic import for Google Maps (no SSR)
const AttractorMap = dynamic(() => import('@/components/AttractorMap'), {
  ssr: false,
  loading: () => <div className="wasm-loader" />,
});

export default function Home() {
  const { wasm, loading: wasmLoading, error: wasmError } = useWasm();
  const { nations: dbNations, edges: dbEdges, loading: dataLoading } = useSupabaseNations();
  const [layer, setLayer] = useState<'basin' | 'risk' | 'influence' | 'regime'>('basin');
  const [isSimulating, setIsSimulating] = useState(false);
  const [simTime, setSimTime] = useState(0);

  // Geospatial system state
  const geoSystemRef = useRef<WasmGeospatialSystem | null>(null);
  const [simulatedNations, setSimulatedNations] = useState<NationState[]>([]);

  // Use simulated nations if available, otherwise use database nations
  const nations = simulatedNations.length > 0 ? simulatedNations : dbNations;
  const edges = dbEdges; // Edges come from database or could be computed from geoSystem

  const alertInfo = useAlertLevel(nations);

  // Initialize WasmGeospatialSystem when WASM and nations are ready
  useEffect(() => {
    if (!wasm || dbNations.length === 0 || geoSystemRef.current) return;

    try {
      // Check if WasmGeospatialSystem is available in the WASM module
      if ('WasmGeospatialSystem' in wasm) {
        const GeoSystem = (wasm as { WasmGeospatialSystem: typeof WasmGeospatialSystem })
          .WasmGeospatialSystem;

        // Create geospatial system with default config
        const geoSystem = GeoSystem.with_defaults();

        // Load nations from database into the simulation
        for (const nation of dbNations) {
          geoSystem.add_nation(
            nation.code,
            nation.name,
            nation.lat ?? 0,
            nation.lon ?? 0,
            nation.regime ?? 0
          );
        }

        geoSystemRef.current = geoSystem;
        console.log(`WasmGeospatialSystem initialized with ${dbNations.length} nations`);
      } else {
        console.log('WasmGeospatialSystem not available in WASM module - using static data');
      }
    } catch (err) {
      console.error('Failed to initialize WasmGeospatialSystem:', err);
    }
  }, [wasm, dbNations]);

  // Run simulation step
  const runStep = useCallback(async () => {
    setIsSimulating(true);
    try {
      if (geoSystemRef.current) {
        // Run 10 simulation steps
        geoSystemRef.current.run(10);
        setSimTime(geoSystemRef.current.get_time());

        // Get updated nation states from the simulation
        const updatedNations = geoSystemRef.current.get_all_nations();
        if (updatedNations && Array.isArray(updatedNations)) {
          setSimulatedNations(updatedNations as NationState[]);
        }
      } else {
        // Fallback: simulate a delay for UI feedback
        await new Promise((resolve) => setTimeout(resolve, 300));
        console.log('Simulation step: WasmGeospatialSystem not initialized');
      }
    } finally {
      setIsSimulating(false);
    }
  }, []);

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
      {/* Alert Banner - Dynamic based on nation risk states */}
      <AlertBanner level={alertInfo.level} message={alertInfo.message} />

      {/* Map */}
      <AttractorMap nations={nations} edges={edges} layer={layer} />

      {/* Control Panel */}
      <ControlPanel
        layer={layer}
        onLayerChange={setLayer}
        onStep={() => void runStep()}
        isSimulating={isSimulating}
        alertLevel={alertInfo.level}
      />

      {/* Info Panel (bottom right) */}
      <div className="absolute bottom-4 right-4 bg-slate-900/90 backdrop-blur-sm rounded-lg p-3 text-xs text-slate-400">
        <p>{nations.length} nations tracked</p>
        <p>{edges.length} influence edges</p>
        {simTime > 0 && <p className="text-green-400">Sim time: {simTime.toFixed(2)}</p>}
        <p className="text-blue-400">Powered by Rust/WASM</p>
      </div>
    </main>
  );
}
