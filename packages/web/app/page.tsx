'use client';

import { useState, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import { useWasm } from '@/hooks/useWasm';
import { useSupabaseNations } from '@/hooks/useSupabaseNations';
import ControlPanel from '@/components/ControlPanel';
import AlertBanner from '@/components/AlertBanner';
// Dynamic import for Google Maps (no SSR)
const AttractorMap = dynamic(() => import('@/components/AttractorMap'), {
  ssr: false,
  loading: () => <div className="wasm-loader" />,
});

// Note: WasmGeospatialSystem is not yet implemented in the WASM module.
// The simulation features are disabled until it's added.

export default function Home() {
  const { wasm, loading: wasmLoading, error: wasmError } = useWasm();
  const { nations, edges, loading: dataLoading } = useSupabaseNations();
  const [layer, setLayer] = useState<'basin' | 'risk' | 'influence' | 'regime'>('basin');
  const alertLevel = 'normal'; // TODO: Make dynamic when alert system is implemented
  const [isSimulating, setIsSimulating] = useState(false);
  const geoSystemInitialized = useRef(false);

  // Log when WASM is ready (geospatial system not yet implemented)
  useEffect(() => {
    if (!wasm || nations.length === 0 || geoSystemInitialized.current) return;

    // Mark as initialized to prevent re-running
    geoSystemInitialized.current = true;

    // TODO: Initialize WasmGeospatialSystem when implemented in WASM module
    // For now, the map displays static nation data from Supabase
    console.log('WASM loaded, nations ready. Geospatial simulation pending implementation.');
  }, [wasm, nations, edges]);

  // Run simulation step (placeholder - WasmGeospatialSystem not yet implemented)
  const runStep = async () => {
    // TODO: Implement when WasmGeospatialSystem is added to WASM module
    setIsSimulating(true);
    try {
      // Simulate a delay for UI feedback
      await new Promise(resolve => setTimeout(resolve, 500));
      console.log('Simulation step: WasmGeospatialSystem not yet implemented');
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
        onStep={() => void runStep()}
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
