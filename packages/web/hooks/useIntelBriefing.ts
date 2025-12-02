'use client';

import { useState, useEffect, useCallback } from 'react';

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

interface UseIntelBriefingResult {
  briefings: IntelBriefings | null;
  metadata: IntelMetadata | null;
  loading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

// Cache briefings for 5 minutes to reduce API calls
const CACHE_TTL = 5 * 60 * 1000;
const cache: Map<
  string,
  { data: { briefings: IntelBriefings; metadata: IntelMetadata }; timestamp: number }
> = new Map();

export function useIntelBriefing(preset: string = 'global'): UseIntelBriefingResult {
  const [briefings, setBriefings] = useState<IntelBriefings | null>(null);
  const [metadata, setMetadata] = useState<IntelMetadata | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

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

  useEffect(() => {
    void fetchBriefing();
  }, [fetchBriefing]);

  return {
    briefings,
    metadata,
    loading,
    error,
    refetch: fetchBriefing,
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
