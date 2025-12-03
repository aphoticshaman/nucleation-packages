'use client';

import { useState, useEffect, useCallback, useRef } from 'react';

// Pulse check for breaking news (runs every 30 seconds)
export interface PulseResult {
  breaking: boolean;
  severity: 'none' | 'significant' | 'major' | 'critical';
  headline?: string;
  cached?: boolean;
}

export interface IntelBriefings {
  political: string;
  economic: string;
  security: string;
  financial: string;
  health: string;
  scitech: string;
  resources: string;
  crime: string;
  cyber: string;
  terrorism: string;
  domestic: string;
  borders: string;
  infoops: string;
  military: string;
  space: string;
  industry: string;
  logistics: string;
  minerals: string;
  energy: string;
  markets: string;
  religious: string;
  education: string;
  employment: string;
  housing: string;
  crypto: string;
  emerging: string;
  summary: string;
  nsm: string; // Next Strategic Move
}

export interface IntelMetadata {
  region: string;
  preset: string;
  timestamp: string;
  overallRisk: 'low' | 'moderate' | 'elevated' | 'high' | 'critical';
}

export interface UseIntelBriefingOptions {
  /** If true, fetch briefing on mount. Default: false (prevents unwanted API calls on page load) */
  autoFetch?: boolean;
  /** If true, enable pulse polling. Default: false (prevents background API calls) */
  enablePulse?: boolean;
}

interface UseIntelBriefingResult {
  briefings: IntelBriefings | null;
  metadata: IntelMetadata | null;
  loading: boolean;
  error: Error | null;
  /** Call this to explicitly fetch/refresh the briefing */
  refetch: () => Promise<void>;
  pulse: PulseResult | null;
  pulseLoading: boolean;
  /** Call this to start pulse polling (if not auto-enabled) */
  startPulse: () => void;
  /** Call this to stop pulse polling */
  stopPulse: () => void;
}

// Cache briefings for 5 minutes (server-side cache is 10 min, this is backup)
const CACHE_TTL = 5 * 60 * 1000;
const cache: Map<
  string,
  { data: { briefings: IntelBriefings; metadata: IntelMetadata }; timestamp: number }
> = new Map();

// Pulse check interval: 30 seconds
const PULSE_INTERVAL = 30 * 1000;

/**
 * Hook for fetching intel briefings.
 *
 * IMPORTANT: By default, this hook does NOT auto-fetch on mount to prevent
 * unwanted Anthropic API calls on page load/refresh. Call `refetch()` explicitly
 * when the user initiates an action.
 *
 * @param preset - The preset to fetch ('global', 'nato', 'brics', 'conflict')
 * @param options - Configuration options
 * @param options.autoFetch - If true, fetch on mount. Default: false
 * @param options.enablePulse - If true, enable background pulse polling. Default: false
 */
export function useIntelBriefing(
  preset: string = 'global',
  options: UseIntelBriefingOptions = {}
): UseIntelBriefingResult {
  const { autoFetch = false, enablePulse = false } = options;

  const [briefings, setBriefings] = useState<IntelBriefings | null>(null);
  const [metadata, setMetadata] = useState<IntelMetadata | null>(null);
  const [loading, setLoading] = useState(false); // Start false - only true when actually fetching
  const [error, setError] = useState<Error | null>(null);
  const [pulse, setPulse] = useState<PulseResult | null>(null);
  const [pulseLoading, setPulseLoading] = useState(false);
  const [pulseEnabled, setPulseEnabled] = useState(enablePulse);
  const pulseIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const hasFetchedRef = useRef(false); // Track if we've ever fetched

  const fetchBriefing = useCallback(async () => {
    const cacheKey = preset;
    const cached = cache.get(cacheKey);

    // Return cached data if still valid
    if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
      setBriefings(cached.data.briefings);
      setMetadata(cached.data.metadata);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/intel-briefing', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ preset }),
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch briefing: ${response.status}`);
      }

      const data = await response.json();

      // Update cache
      cache.set(cacheKey, {
        data: {
          briefings: data.briefings,
          metadata: data.metadata,
        },
        timestamp: Date.now(),
      });

      setBriefings(data.briefings);
      setMetadata(data.metadata);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to fetch briefing'));
    } finally {
      setLoading(false);
    }
  }, [preset]);

  // Only auto-fetch if explicitly enabled (default: false)
  useEffect(() => {
    if (autoFetch && !hasFetchedRef.current) {
      hasFetchedRef.current = true;
      void fetchBriefing();
    }
  }, [autoFetch, fetchBriefing]);

  // Pulse check - runs every 30 seconds for breaking news (only if enabled)
  const checkPulse = useCallback(async () => {
    if (!pulseEnabled) return;

    try {
      setPulseLoading(true);
      const response = await fetch('/api/intel-briefing/pulse');
      if (response.ok) {
        const data = await response.json();
        setPulse(data);

        // If breaking news detected, force refresh briefing (only if we've already fetched once)
        if (hasFetchedRef.current && data.breaking && (data.severity === 'major' || data.severity === 'critical')) {
          console.log('[PULSE] Breaking news detected, refreshing briefing...');
          cache.delete(preset); // Clear local cache
          void fetchBriefing(); // Refresh briefing
        }
      }
    } catch (err) {
      console.error('Pulse check failed:', err);
    } finally {
      setPulseLoading(false);
    }
  }, [preset, fetchBriefing, pulseEnabled]);

  // Start/stop pulse polling based on pulseEnabled state
  useEffect(() => {
    // Clear any existing interval first
    if (pulseIntervalRef.current) {
      clearInterval(pulseIntervalRef.current);
      pulseIntervalRef.current = null;
    }

    // Only start polling if pulse is enabled
    if (pulseEnabled) {
      // Initial pulse check
      void checkPulse();

      // Set up interval for 30-second pulse checks
      pulseIntervalRef.current = setInterval(() => {
        void checkPulse();
      }, PULSE_INTERVAL);
    }

    return () => {
      if (pulseIntervalRef.current) {
        clearInterval(pulseIntervalRef.current);
        pulseIntervalRef.current = null;
      }
    };
  }, [checkPulse, pulseEnabled]);

  // Control functions for pulse polling
  const startPulse = useCallback(() => {
    setPulseEnabled(true);
  }, []);

  const stopPulse = useCallback(() => {
    setPulseEnabled(false);
    setPulse(null);
  }, []);

  return {
    briefings,
    metadata,
    loading,
    error,
    refetch: fetchBriefing,
    pulse,
    pulseLoading,
    startPulse,
    stopPulse,
  };
}

// Risk level color mapping
export function getRiskColor(risk: IntelMetadata['overallRisk']): string {
  switch (risk) {
    case 'low':
      return 'text-green-400';
    case 'moderate':
      return 'text-blue-400';
    case 'elevated':
      return 'text-yellow-400';
    case 'high':
      return 'text-orange-400';
    case 'critical':
      return 'text-red-400';
    default:
      return 'text-slate-400';
  }
}

// Risk level badge styles
export function getRiskBadgeStyle(risk: IntelMetadata['overallRisk']): string {
  switch (risk) {
    case 'low':
      return 'bg-green-500/20 text-green-400 border-green-500/30';
    case 'moderate':
      return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
    case 'elevated':
      return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
    case 'high':
      return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
    case 'critical':
      return 'bg-red-500/20 text-red-400 border-red-500/30';
    default:
      return 'bg-slate-500/20 text-slate-400 border-slate-500/30';
  }
}
