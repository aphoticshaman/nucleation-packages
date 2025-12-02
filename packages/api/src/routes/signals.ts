/**
 * Signal Routes
 *
 * Core API endpoints for signal intelligence.
 */

import { Hono } from 'hono';
import type { AuthContext, SignalResponse, ProvenanceRecord } from '../types.js';
import { requirePermission } from '../middleware/auth.js';

const signals = new Hono();

/**
 * GET /signals
 * List available signal sources
 */
signals.get('/', requirePermission('read:signals'), async (c) => {
  const auth = c.get('auth') as AuthContext;

  const sources = [
    {
      id: 'sec-edgar',
      name: 'SEC EDGAR',
      tier: 'official',
      description: 'US securities filings',
      available: true,
    },
    {
      id: 'fred',
      name: 'Federal Reserve Economic Data',
      tier: 'official',
      description: 'Economic indicators',
      available: true,
    },
    {
      id: 'guardian',
      name: 'The Guardian',
      tier: 'news',
      description: 'News articles',
      available: auth.client.tier !== 'free',
    },
    {
      id: 'reddit',
      name: 'Reddit',
      tier: 'social',
      description: 'Social sentiment',
      available: auth.client.tier !== 'free',
    },
    {
      id: 'bluesky',
      name: 'Bluesky',
      tier: 'social',
      description: 'Social sentiment',
      available: auth.client.tier !== 'free',
    },
  ];

  return c.json({
    success: true,
    data: { sources },
    meta: buildMeta(c),
  });
});

/**
 * POST /signals/fetch
 * Fetch signals from specified sources
 */
signals.post('/fetch', requirePermission('read:signals'), async (c) => {
  const auth = c.get('auth') as AuthContext;
  const body = await c.req.json();

  const { sources, normalize, fuse } = body;

  if (!sources || !Array.isArray(sources) || sources.length === 0) {
    return c.json(
      {
        success: false,
        error: {
          code: 'INVALID_REQUEST',
          message: 'sources array is required',
        },
        meta: buildMeta(c),
      },
      400
    );
  }

  // Validate sources against tier
  const tierSources = getTierSources(auth.client.tier);
  const invalidSources = sources.filter((s: string) => !tierSources.includes(s));

  if (invalidSources.length > 0) {
    return c.json(
      {
        success: false,
        error: {
          code: 'SOURCE_NOT_AVAILABLE',
          message: `Sources not available for your tier: ${invalidSources.join(', ')}`,
        },
        meta: buildMeta(c),
      },
      403
    );
  }

  // Mock signal data (replace with real implementation)
  const signalData: Record<string, number[]> = {};
  const provenance: ProvenanceRecord[] = [];

  for (const source of sources) {
    // Generate mock data
    signalData[source] = generateMockSignal(50);
    provenance.push({
      sourceId: source,
      tier: getSourceTier(source),
      fetchedAt: new Date().toISOString(),
      attribution: getSourceAttribution(source),
      hash: generateHash(signalData[source]),
    });
  }

  const response: SignalResponse = {
    signals: signalData,
    provenance,
  };

  // Optionally normalize
  if (normalize) {
    for (const [key, signal] of Object.entries(response.signals)) {
      response.signals[key] = normalizeSignal(signal);
    }
  }

  // Optionally fuse
  if (fuse && Object.keys(response.signals).length > 1) {
    response.fused = fuseSignals(Object.values(response.signals));
    response.phase = calculatePhase(response.fused);
    response.confidence = calculateConfidence(Object.keys(response.signals).length);
  }

  return c.json({
    success: true,
    data: response,
    meta: buildMeta(c),
  });
});

/**
 * GET /signals/:sourceId
 * Get signals from a specific source
 */
signals.get('/:sourceId', requirePermission('read:signals'), async (c) => {
  const auth = c.get('auth') as AuthContext;
  const sourceId = c.req.param('sourceId');

  const tierSources = getTierSources(auth.client.tier);
  if (!tierSources.includes(sourceId)) {
    return c.json(
      {
        success: false,
        error: {
          code: 'SOURCE_NOT_AVAILABLE',
          message: `Source '${sourceId}' not available for your tier`,
        },
        meta: buildMeta(c),
      },
      403
    );
  }

  // Mock single source fetch
  const signal = generateMockSignal(100);

  return c.json({
    success: true,
    data: {
      sourceId,
      signal,
      provenance: {
        sourceId,
        tier: getSourceTier(sourceId),
        fetchedAt: new Date().toISOString(),
        attribution: getSourceAttribution(sourceId),
        hash: generateHash(signal),
      },
    },
    meta: buildMeta(c),
  });
});

/**
 * POST /signals/detect
 * Run phase detection on provided or fetched signals
 */
signals.post('/detect', requirePermission('read:signals'), async (c) => {
  const body = await c.req.json();
  const { signal, windowSize = 10 } = body;

  if (!signal || !Array.isArray(signal)) {
    return c.json(
      {
        success: false,
        error: {
          code: 'INVALID_REQUEST',
          message: 'signal array is required',
        },
        meta: buildMeta(c),
      },
      400
    );
  }

  // Calculate phase detection
  const phases = detectPhases(signal, windowSize);
  const currentPhase = phases[phases.length - 1] ?? 0;

  return c.json({
    success: true,
    data: {
      phases,
      currentPhase,
      windowSize,
      signalLength: signal.length,
      interpretation: interpretPhase(currentPhase),
    },
    meta: buildMeta(c),
  });
});

// Helper functions

function buildMeta(c: any) {
  const auth = c.get('auth') as AuthContext | undefined;
  const rateLimit = c.get('rateLimit') ?? { remaining: -1, reset: 0 };

  return {
    requestId: c.get('requestId') ?? 'unknown',
    timestamp: new Date().toISOString(),
    version: '1.0.0',
    tier: auth?.client.tier ?? 'unknown',
    rateLimit,
  };
}

function getTierSources(tier: string): string[] {
  const sources: Record<string, string[]> = {
    free: ['sec-edgar', 'fred'],
    pro: ['sec-edgar', 'fred', 'guardian', 'newsapi', 'reddit', 'bluesky'],
    enterprise: ['sec-edgar', 'fred', 'guardian', 'newsapi', 'reddit', 'bluesky', 'discord', 'alpha-vantage'],
    government: ['sec-edgar', 'fred', 'guardian', 'newsapi', 'reddit', 'bluesky', 'discord', 'alpha-vantage', 'classified'],
  };
  return sources[tier] ?? sources.free;
}

function getSourceTier(sourceId: string): string {
  const tiers: Record<string, string> = {
    'sec-edgar': 'official',
    fred: 'official',
    guardian: 'news',
    newsapi: 'news',
    reddit: 'social',
    bluesky: 'social',
    discord: 'social',
    'alpha-vantage': 'financial',
  };
  return tiers[sourceId] ?? 'unknown';
}

function getSourceAttribution(sourceId: string): string {
  const attributions: Record<string, string> = {
    'sec-edgar': 'Data from SEC EDGAR',
    fred: 'Source: FRED, Federal Reserve Bank of St. Louis',
    guardian: 'Powered by Guardian Open Platform',
    newsapi: 'Powered by NewsAPI.org',
    reddit: 'Data from Reddit API',
    bluesky: 'Data from Bluesky AT Protocol',
  };
  return attributions[sourceId] ?? `Data from ${sourceId}`;
}

function generateMockSignal(length: number): number[] {
  const signal: number[] = [];
  let value = 100;
  for (let i = 0; i < length; i++) {
    value += (Math.random() - 0.5) * 10;
    signal.push(Math.round(value * 100) / 100);
  }
  return signal;
}

function normalizeSignal(signal: number[]): number[] {
  if (signal.length === 0) return [];
  const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
  const variance = signal.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / signal.length;
  const stdDev = Math.sqrt(variance);
  if (stdDev === 0) return signal.map(() => 0);
  return signal.map((v) => (v - mean) / stdDev);
}

function fuseSignals(signals: number[][]): number[] {
  const minLength = Math.min(...signals.map((s) => s.length));
  const fused: number[] = [];
  for (let i = 0; i < minLength; i++) {
    const sum = signals.reduce((acc, s) => acc + (s[i] ?? 0), 0);
    fused.push(sum / signals.length);
  }
  return fused;
}

function calculatePhase(signal: number[]): number {
  if (signal.length < 2) return 0;
  const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
  const variance = signal.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / signal.length;
  return Math.min(variance / (mean * mean + 1), 1);
}

function calculateConfidence(sourceCount: number): number {
  return Math.min(sourceCount / 5, 1);
}

function detectPhases(signal: number[], windowSize: number): number[] {
  const phases: number[] = [];
  for (let i = windowSize; i <= signal.length; i++) {
    const window = signal.slice(i - windowSize, i);
    phases.push(calculatePhase(window));
  }
  return phases;
}

function interpretPhase(phase: number): string {
  if (phase < 0.1) return 'STABLE - Low variance, calm market conditions';
  if (phase < 0.3) return 'NORMAL - Typical variance levels';
  if (phase < 0.5) return 'ELEVATED - Increased variance, monitor closely';
  if (phase < 0.7) return 'HIGH - Significant variance, potential regime change';
  return 'CRITICAL - Extreme variance, phase transition likely';
}

function generateHash(data: number[]): string {
  let hash = 0;
  const str = JSON.stringify(data);
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }
  return hash.toString(16);
}

export default signals;
