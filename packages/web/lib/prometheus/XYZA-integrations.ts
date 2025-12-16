/**
 * XYZA INTEGRATIONS - PROMETHEUS PROTOCOL
 *
 * Production-ready code integrations connecting NSM insights
 * to LatticeForge infrastructure. Each integration includes:
 * - Pseudocode algorithm
 * - TypeScript implementation
 * - Guardian security hooks
 * - Encrypted logging
 */

import { ElleGuardian, type ElleInteraction, type ElleInteractionType } from '../reasoning/security';
import { encrypt, decrypt, type EncryptedPayload } from '../crypto/encryption';
import {
  detectPatternAnomaly,
  detectTemporalBreak,
  detectCrossDomainIso,
  buildKnowledgeQuadrant,
  prioritizeNovelties,
  type NoveltySignal,
} from './novelty-detection';
import { prometheusEngine } from './prometheus-engine';
import { NSM_INSIGHTS, getCriticalInsights, generateRecommendations } from './NSM-x20-insights';

// ═══════════════════════════════════════════════════════════════════════════════
// XYZA INTEGRATION #1: UNIFIED SIGNAL FUSION ENGINE
// ═══════════════════════════════════════════════════════════════════════════════
/*
PSEUDOCODE:
```
function fuseSignals(signals: Signal[], weights: number[]): FusedSignal
  // NSM Insight #4: Confidence as weight
  weightedSignals = signals.map((s, i) => s.value * s.confidence * weights[i])

  // NSM Insight #9: Gravitational basin clustering
  basins = clusterByProximity(weightedSignals)

  // Select dominant basin (highest mass × density)
  dominant = basins.maxBy(b => b.mass * b.density)

  // NSM Insight #5: Only trust high-magnitude transitions
  if dominant.magnitude < THRESHOLD: return UNCERTAIN

  return FusedSignal(dominant.centroid, dominant.confidence)
```
*/

export interface MultiDomainSignal {
  domain: 'engagement' | 'churn' | 'security' | 'ux' | 'ai' | 'market';
  value: number;
  variance: number;
  confidence: number;
  magnitude: number; // inflection magnitude
  timestamp: number;
}

export interface FusedSignal {
  value: number;
  confidence: number;
  contributingDomains: string[];
  coherenceScore: number; // NSM #2: How aligned are the signals?
  recommendation: string;
}

const DOMAIN_WEIGHTS: Record<string, number> = {
  engagement: 1.0,
  churn: 1.2,
  security: 1.5,
  ux: 0.9,
  ai: 1.1,
  market: 0.8,
};

export function fuseMultiDomainSignals(signals: MultiDomainSignal[]): FusedSignal {
  if (signals.length === 0) {
    return {
      value: 0,
      confidence: 0,
      contributingDomains: [],
      coherenceScore: 0,
      recommendation: 'No signals to fuse',
    };
  }

  // NSM Insight #4: Weight by confidence
  const weightedValues = signals.map(s => ({
    domain: s.domain,
    weighted: s.value * s.confidence * (DOMAIN_WEIGHTS[s.domain] || 1.0),
    confidence: s.confidence,
    magnitude: s.magnitude,
  }));

  // NSM Insight #9: Cluster by proximity (simplified gravitational basins)
  const values = weightedValues.map(w => w.weighted);
  const basins = prometheusEngine.clusterBasins(values);

  if (basins.length === 0) {
    return {
      value: signals.reduce((a, b) => a + b.value, 0) / signals.length,
      confidence: 0.3,
      contributingDomains: signals.map(s => s.domain),
      coherenceScore: 0,
      recommendation: 'Signals too dispersed for confident fusion',
    };
  }

  // Select dominant basin
  let dominant = basins[0];
  for (const basin of basins) {
    const score = basin.mass * Math.pow(basin.density, 0.1);
    if (score > dominant.mass * Math.pow(dominant.density, 0.1)) {
      dominant = basin;
    }
  }

  // NSM Insight #2: Calculate coherence (how many signals agree?)
  const coherenceScore = dominant.mass / signals.length;

  // NSM Insight #5: Check magnitude threshold
  const avgMagnitude = signals.reduce((a, b) => a + b.magnitude, 0) / signals.length;
  const highMagnitude = avgMagnitude > 2.0;

  // Generate recommendation based on pattern
  let recommendation = 'Normal operation';
  if (coherenceScore > 0.7 && highMagnitude) {
    recommendation = 'CRITICAL: Multi-domain phase transition detected. Immediate review required.';
  } else if (coherenceScore > 0.5 && dominant.centroid > 0.7) {
    recommendation = 'WARNING: Elevated signals across multiple domains. Monitor closely.';
  } else if (coherenceScore < 0.3) {
    recommendation = 'Divergent signals - investigate domain-specific causes';
  }

  return {
    value: dominant.centroid,
    confidence: Math.min(coherenceScore, Math.max(...signals.map(s => s.confidence))),
    contributingDomains: signals.map(s => s.domain),
    coherenceScore,
    recommendation,
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// XYZA INTEGRATION #2: ENCRYPTED FLOW STATE PERSISTENCE
// ═══════════════════════════════════════════════════════════════════════════════
/*
PSEUDOCODE:
```
function persistFlowState(userId: string, state: FlowState): void
  // NSM Insight #7: Serializable state
  serialized = state.detectors.map(d => d.serialize())

  // Encrypt with user-specific key
  encrypted = encrypt(JSON.stringify(serialized), deriveKey(userId))

  // Store with integrity hash
  store.set(userId, {
    data: encrypted.ciphertext,
    iv: encrypted.iv,
    salt: encrypted.salt,
    hash: sha256(encrypted.ciphertext)
  })

function restoreFlowState(userId: string): FlowState
  stored = store.get(userId)

  // Verify integrity
  if sha256(stored.data) != stored.hash:
    throw TAMPERED_ERROR

  decrypted = decrypt(stored.data, deriveKey(userId), stored.iv, stored.salt)

  // NSM Insight #7: Deserialize without reprocessing history
  return FlowState.fromSerialized(JSON.parse(decrypted))
```
*/

export interface SerializedFlowState {
  userId: string;
  detectorStates: Record<string, string>; // domain -> serialized detector
  lastUpdated: number;
  eventCount: number;
}

export interface EncryptedFlowState {
  ciphertext: string;
  iv: string;
  salt: string;
  integrityHash: string;
  encryptedAt: string;
  version: number;
}

export async function persistFlowState(
  userId: string,
  detectorStates: Record<string, string>,
  encryptionKey: string
): Promise<EncryptedFlowState> {
  const state: SerializedFlowState = {
    userId,
    detectorStates,
    lastUpdated: Date.now(),
    eventCount: Object.keys(detectorStates).length,
  };

  const encrypted = await encrypt(JSON.stringify(state), encryptionKey);

  return {
    ciphertext: encrypted.ciphertext,
    iv: encrypted.iv,
    salt: encrypted.salt,
    integrityHash: encrypted.integrityHash,
    encryptedAt: encrypted.encryptedAt,
    version: encrypted.version,
  };
}

export async function restoreFlowState(
  encryptedState: EncryptedFlowState,
  encryptionKey: string
): Promise<{ state: SerializedFlowState; verified: boolean; tampered: boolean }> {
  try {
    const decrypted = await decrypt(
      {
        ciphertext: encryptedState.ciphertext,
        iv: encryptedState.iv,
        salt: encryptedState.salt,
        integrityHash: encryptedState.integrityHash,
        encryptedAt: encryptedState.encryptedAt,
        version: encryptedState.version,
      },
      encryptionKey
    );

    return {
      state: JSON.parse(decrypted.plaintext) as SerializedFlowState,
      verified: decrypted.verified,
      tampered: false,
    };
  } catch (error) {
    return {
      state: {
        userId: '',
        detectorStates: {},
        lastUpdated: 0,
        eventCount: 0,
      },
      verified: false,
      tampered: true,
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// XYZA INTEGRATION #3: GUARDIAN-WRAPPED ELLE PIPELINE
// ═══════════════════════════════════════════════════════════════════════════════
/*
PSEUDOCODE:
```
function processElleInteraction(input: UserInput): ElleResponse
  guardian = new ElleGuardian()

  // Pre-filter user input (NSM #16: Detect threat quieting)
  inputFiltered = guardian.filter({
    type: 'user_to_elle',
    content: input.message,
    context: input.context
  })

  if inputFiltered.blocked:
    return BLOCKED_RESPONSE

  // Process with Elle
  response = elle.generate(inputFiltered.filtered)

  // NSM Insight #10: Track reasoning entropy
  entropy = prometheus.calculateEntropy(response.reasoning)

  // Post-filter output
  outputFiltered = guardian.filter({
    type: 'elle_to_user',
    content: response.content,
    entropy: entropy
  })

  // NSM Insight #3: Check for divergence
  divergence = shepherd.getConflictPotential(input.userId, 'elle')
  if divergence > THRESHOLD:
    outputFiltered.content += '\n[Communication clarity check recommended]'

  return outputFiltered
```
*/

export interface ElleRequest {
  userId: string;
  sessionId: string;
  message: string;
  context?: Record<string, unknown>;
  previousMessages?: Array<{ role: string; content: string }>;
}

export interface ElleResponse {
  content: string;
  reasoning?: string;
  entropy?: number;
  confidence?: number;
  divergenceWarning?: boolean;
  blocked?: boolean;
  blockReason?: string;
  guardrails: string[];
}

export interface GuardianPipelineResult {
  request: ElleRequest;
  inputFiltered: {
    blocked: boolean;
    redacted: string;
    risks: string[];
  };
  response: ElleResponse;
  outputFiltered: {
    blocked: boolean;
    redacted: string;
    risks: string[];
  };
  metrics: {
    entropy: number;
    divergence: number;
    processingTime: number;
  };
}

export async function processElleWithGuardian(
  request: ElleRequest,
  generateResponse: (input: string, context: Record<string, unknown>) => Promise<{ content: string; reasoning?: string }>
): Promise<GuardianPipelineResult> {
  const startTime = Date.now();
  const guardian = new ElleGuardian();
  const guardrails: string[] = [];

  // Pre-filter user input - convert context to compatible metadata type
  const safeMetadata: Record<string, string | number | boolean | undefined> = {};
  if (request.context) {
    for (const [key, value] of Object.entries(request.context)) {
      if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' || value === undefined) {
        safeMetadata[key] = value;
      }
    }
  }

  const inputInteraction: ElleInteraction = {
    type: 'user_to_elle',
    content: request.message,
    metadata: {
      userId: request.userId,
      conversationId: request.sessionId,
      ...safeMetadata,
    },
  };

  const inputResult = await guardian.filter(inputInteraction);
  if (inputResult.redacted.length > 0) {
    guardrails.push(...inputResult.redacted.map(r => `input_redacted:${r}`));
  }

  if (inputResult.blocked) {
    return {
      request,
      inputFiltered: {
        blocked: true,
        redacted: inputResult.filtered,
        risks: inputResult.redacted,
      },
      response: {
        content: 'I cannot process this request due to security policies.',
        blocked: true,
        blockReason: inputResult.reason || 'Security policy violation',
        guardrails,
      },
      outputFiltered: {
        blocked: false,
        redacted: '',
        risks: [],
      },
      metrics: {
        entropy: 0,
        divergence: 0,
        processingTime: Date.now() - startTime,
      },
    };
  }

  // Generate response with filtered input
  const generated = await generateResponse(inputResult.filtered, request.context || {});

  // Calculate reasoning entropy (NSM Insight #10)
  const entropy = prometheusEngine.calculateEntropy(generated.reasoning || generated.content);

  // Post-filter output
  const outputInteraction: ElleInteraction = {
    type: 'elle_to_user',
    content: generated.content,
    metadata: {
      userId: request.userId,
      conversationId: request.sessionId,
    },
  };

  const outputResult = await guardian.filter(outputInteraction);
  if (outputResult.redacted.length > 0) {
    guardrails.push(...outputResult.redacted.map(r => `output_redacted:${r}`));
  }

  // NSM Insight #3: Check for communication divergence
  // Simplified heuristic: high entropy + multiple redactions = divergence
  const divergence = (entropy > 4.5 ? 0.3 : 0) + (outputResult.redacted.length * 0.1);
  const divergenceWarning = divergence > 0.5;

  return {
    request,
    inputFiltered: {
      blocked: false,
      redacted: inputResult.filtered,
      risks: inputResult.redacted,
    },
    response: {
      content: outputResult.filtered,
      reasoning: generated.reasoning,
      entropy,
      confidence: 1 - (entropy / 8), // Normalize to 0-1
      divergenceWarning,
      blocked: outputResult.blocked,
      guardrails,
    },
    outputFiltered: {
      blocked: outputResult.blocked,
      redacted: outputResult.filtered,
      risks: outputResult.redacted,
    },
    metrics: {
      entropy,
      divergence,
      processingTime: Date.now() - startTime,
    },
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// XYZA INTEGRATION #4: REAL-TIME NOVELTY SCANNER
// ═══════════════════════════════════════════════════════════════════════════════
/*
PSEUDOCODE:
```
function scanForNovelties(streams: DataStream[]): NoveltyReport
  novelties = []

  // NSM Insight #14: Dual metrics for regime detection
  for stream in streams:
    anomalies = detectPatternAnomaly(stream.values)
    breaks = detectTemporalBreak(stream.values)
    novelties.extend(anomalies + breaks)

  // NSM Insight #13: Cross-domain correlations
  for i, stream1 in enumerate(streams):
    for stream2 in streams[i+1:]:
      isos = detectCrossDomainIso(stream1, stream2)
      novelties.extend(isos)

  // NSM Insight #12: Classify into knowledge quadrants
  quadrants = buildKnowledgeQuadrant(novelties, existingKnowledge)

  // Prioritize by significance
  prioritized = prioritizeNovelties(novelties)

  return NoveltyReport(prioritized, quadrants, generateRecommendations(novelties))
```
*/

export interface DataStream {
  name: string;
  domain: string;
  values: number[];
  metadata?: Record<string, unknown>;
}

export interface NoveltyReport {
  timestamp: string;
  signals: NoveltySignal[];
  quadrants: {
    knownKnowns: string[];
    knownUnknowns: string[];
    unknownUnknowns: string[];
    unknownKnowns: string[];
  };
  recommendations: string[];
  criticalCount: number;
  requiresAction: boolean;
}

export function scanForNovelties(
  streams: DataStream[],
  existingKnowledge: string[] = []
): NoveltyReport {
  const allSignals: NoveltySignal[] = [];

  // NSM Insight #14: Detect anomalies and breaks per stream
  for (const stream of streams) {
    if (stream.values.length < 10) continue;

    const anomalies = detectPatternAnomaly(stream.values);
    const breaks = detectTemporalBreak(stream.values);

    // Tag with stream metadata
    for (const signal of [...anomalies, ...breaks]) {
      signal.sourceData = {
        ...signal.sourceData,
        streamName: stream.name,
        streamDomain: stream.domain,
      };
    }

    allSignals.push(...anomalies, ...breaks);
  }

  // NSM Insight #13: Cross-domain correlations
  for (let i = 0; i < streams.length; i++) {
    for (let j = i + 1; j < streams.length; j++) {
      if (streams[i].values.length < 20 || streams[j].values.length < 20) continue;

      const isos = detectCrossDomainIso(
        { name: streams[i].name, data: streams[i].values },
        { name: streams[j].name, data: streams[j].values }
      );
      allSignals.push(...isos);
    }
  }

  // NSM Insight #12: Build knowledge quadrants
  const quadrants = buildKnowledgeQuadrant(allSignals, existingKnowledge);

  // Prioritize signals
  const prioritized = prioritizeNovelties(allSignals);

  // Generate recommendations using NSM insights
  const recommendations = generateRecommendations(allSignals);

  // Count critical signals
  const criticalCount = allSignals.filter(s => s.confidence > 0.8 && s.significance > 0.7).length;

  return {
    timestamp: new Date().toISOString(),
    signals: prioritized,
    quadrants,
    recommendations,
    criticalCount,
    requiresAction: criticalCount > 0 || quadrants.unknownUnknowns.length > 3,
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// XYZA INTEGRATION #5: BATCH EVENT PROCESSOR
// ═══════════════════════════════════════════════════════════════════════════════
/*
PSEUDOCODE:
```
function processBatch(events: Event[], windowMs: number): ProcessedBatch
  // NSM Insight #19: Batch for throughput
  batched = groupByWindow(events, windowMs)

  results = []
  for window in batched:
    // Convert events to numeric signals
    signals = window.map(e => eventToSignal(e))

    // Use updateBatch() instead of individual updates
    state = detector.updateBatch(signals)

    // NSM Insight #20: Check for boundary events
    if hasBoundaryEvent(window):
      detector.reset()
      state.isBoundary = true

    results.push(state)

  return ProcessedBatch(results)
```
*/

export interface UserEvent {
  userId: string;
  eventType: string;
  value: number;
  timestamp: number;
  metadata?: Record<string, unknown>;
}

export interface ProcessedWindow {
  windowStart: number;
  windowEnd: number;
  eventCount: number;
  meanValue: number;
  variance: number;
  phase: 'stable' | 'approaching' | 'critical' | 'transitioning';
  isBoundary: boolean;
  boundaryType?: 'conversion' | 'incident' | 'launch' | 'reset';
}

export interface ProcessedBatch {
  userId: string;
  windows: ProcessedWindow[];
  overallTrend: 'improving' | 'stable' | 'declining';
  alertRequired: boolean;
  recommendations: string[];
}

const BOUNDARY_EVENTS = new Set(['signup', 'conversion', 'upgrade', 'downgrade', 'incident_resolved', 'product_launch']);

export function processBatchEvents(
  events: UserEvent[],
  windowMs: number = 60000 // 1 minute default
): ProcessedBatch {
  if (events.length === 0) {
    return {
      userId: '',
      windows: [],
      overallTrend: 'stable',
      alertRequired: false,
      recommendations: [],
    };
  }

  const userId = events[0].userId;

  // Group events into windows
  const sorted = [...events].sort((a, b) => a.timestamp - b.timestamp);
  const windows: UserEvent[][] = [];
  let currentWindow: UserEvent[] = [];
  let windowStart = sorted[0].timestamp;

  for (const event of sorted) {
    if (event.timestamp - windowStart > windowMs) {
      if (currentWindow.length > 0) {
        windows.push(currentWindow);
      }
      currentWindow = [event];
      windowStart = event.timestamp;
    } else {
      currentWindow.push(event);
    }
  }
  if (currentWindow.length > 0) {
    windows.push(currentWindow);
  }

  // Process each window
  const processedWindows: ProcessedWindow[] = [];

  for (const window of windows) {
    const values = window.map(e => e.value);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;

    // Check for boundary events
    const hasBoundary = window.some(e => BOUNDARY_EVENTS.has(e.eventType));
    const boundaryEvent = window.find(e => BOUNDARY_EVENTS.has(e.eventType));

    // Determine phase based on variance trend
    let phase: ProcessedWindow['phase'] = 'stable';
    if (processedWindows.length > 0) {
      const prevVariance = processedWindows[processedWindows.length - 1].variance;
      const varianceRatio = variance / (prevVariance || 1);

      if (varianceRatio < 0.5) {
        phase = 'approaching'; // Variance quieting
      } else if (varianceRatio < 0.7) {
        phase = 'critical';
      } else if (varianceRatio < 0.9 && variance < 0.1) {
        phase = 'transitioning';
      }
    }

    processedWindows.push({
      windowStart: window[0].timestamp,
      windowEnd: window[window.length - 1].timestamp,
      eventCount: window.length,
      meanValue: mean,
      variance,
      phase,
      isBoundary: hasBoundary,
      boundaryType: hasBoundary ? (boundaryEvent?.eventType as ProcessedWindow['boundaryType']) : undefined,
    });
  }

  // Determine overall trend
  let overallTrend: ProcessedBatch['overallTrend'] = 'stable';
  if (processedWindows.length >= 3) {
    const recent = processedWindows.slice(-3);
    const meanValues = recent.map(w => w.meanValue);
    const trend = meanValues[2] - meanValues[0];

    if (trend > 0.1) {
      overallTrend = 'improving';
    } else if (trend < -0.1) {
      overallTrend = 'declining';
    }
  }

  // Check if alert needed
  const alertRequired = processedWindows.some(w => w.phase === 'transitioning' || w.phase === 'critical');

  // Generate recommendations
  const recommendations: string[] = [];
  if (overallTrend === 'declining') {
    recommendations.push('[NSM #1] Declining engagement detected. Review recent product changes.');
  }
  if (processedWindows.some(w => w.phase === 'approaching')) {
    recommendations.push('[NSM #5] Variance quieting detected. Monitor for phase transition.');
  }
  if (processedWindows.filter(w => w.isBoundary).length > 1) {
    recommendations.push('[NSM #20] Multiple boundary events. Consider detector reset for fresh baseline.');
  }

  return {
    userId,
    windows: processedWindows,
    overallTrend,
    alertRequired,
    recommendations,
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// XYZA INTEGRATION #6: WASM EDGE DEPLOYMENT
// ═══════════════════════════════════════════════════════════════════════════════
/*
PSEUDOCODE:
```
// Client-side WASM integration (NSM Insight #6)
// Deploy detectors to edge for real-time, low-latency detection

async function initEdgeDetector(): EdgeDetector
  wasm = await loadWasm('/wasm/latticeforge_core_bg.wasm')
  detector = wasm.createVarianceDetector(config)

  return {
    update: (value) => detector.update(value),
    getPhase: () => detector.currentPhase(),
    serialize: () => detector.serialize(),
    syncToServer: async () => {
      state = detector.serialize()
      // Only sync state changes, not raw events
      await fetch('/api/flow-state/sync', {
        method: 'POST',
        body: encrypt(state, userKey)
      })
    }
  }

// Server just stores encrypted state
// All computation happens on client
```
*/

export interface EdgeDetectorConfig {
  wasmPath: string;
  sensitivity: 'conservative' | 'balanced' | 'sensitive';
  windowSize: number;
  syncInterval: number; // ms between server syncs
  encryptionKey: string;
}

export interface EdgeDetectorState {
  phase: 'stable' | 'approaching' | 'critical' | 'transitioning';
  variance: number;
  confidence: number;
  eventCount: number;
  lastSync: number;
  pendingSync: boolean;
}

// This is the interface for client-side implementation
export interface EdgeDetectorInterface {
  update(value: number): EdgeDetectorState;
  getState(): EdgeDetectorState;
  serialize(): string;
  deserialize(json: string): void;
  scheduleSync(): void;
  forceSyncNow(): Promise<boolean>;
}

/**
 * Factory function to create edge detector configuration
 * Actual implementation runs in browser with WASM
 */
export function createEdgeDetectorConfig(
  userEncryptionKey: string,
  options: Partial<EdgeDetectorConfig> = {}
): EdgeDetectorConfig {
  return {
    wasmPath: options.wasmPath || '/wasm/latticeforge_core_bg.wasm',
    sensitivity: options.sensitivity || 'balanced',
    windowSize: options.windowSize || 50,
    syncInterval: options.syncInterval || 30000, // 30 seconds
    encryptionKey: userEncryptionKey,
  };
}

/**
 * Generate client-side code snippet for edge deployment
 */
export function generateEdgeDeploymentCode(config: EdgeDetectorConfig): string {
  return `
// LatticeForge Edge Detector - Auto-generated
// NSM Insight #6: WASM acceleration for real-time detection

import init, { NucleationDetector, DetectorConfig } from '${config.wasmPath}';

class EdgeFlowStateDetector {
  constructor() {
    this.detector = null;
    this.lastSync = Date.now();
    this.syncInterval = ${config.syncInterval};
    this.pendingEvents = [];
  }

  async initialize() {
    await init();
    const config = new DetectorConfig();
    config.window_size = ${config.windowSize};
    this.detector = new NucleationDetector(config);
    this.startSyncLoop();
  }

  update(value) {
    const phase = this.detector.update(value);
    this.pendingEvents.push({ value, time: Date.now() });

    return {
      phase: this.phaseToString(phase),
      variance: this.detector.currentVariance(),
      confidence: this.detector.confidence(),
      eventCount: this.detector.count()
    };
  }

  phaseToString(phase) {
    const names = ['stable', 'approaching', 'critical', 'transitioning'];
    return names[phase] || 'stable';
  }

  serialize() {
    return this.detector.serialize();
  }

  async syncToServer() {
    const state = this.serialize();
    const encrypted = await this.encrypt(state);

    try {
      const response = await fetch('/api/flow-state/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state: encrypted, eventCount: this.pendingEvents.length })
      });

      if (response.ok) {
        this.pendingEvents = [];
        this.lastSync = Date.now();
        return true;
      }
    } catch (e) {
      console.warn('Flow state sync failed:', e);
    }
    return false;
  }

  startSyncLoop() {
    setInterval(() => {
      if (this.pendingEvents.length > 0) {
        this.syncToServer();
      }
    }, this.syncInterval);
  }

  async encrypt(plaintext) {
    // Use Web Crypto API with user-derived key
    const encoder = new TextEncoder();
    const salt = crypto.getRandomValues(new Uint8Array(16));
    const iv = crypto.getRandomValues(new Uint8Array(12));

    const keyMaterial = await crypto.subtle.importKey(
      'raw', encoder.encode('${config.encryptionKey}'), 'PBKDF2', false, ['deriveKey']
    );

    const key = await crypto.subtle.deriveKey(
      { name: 'PBKDF2', salt, iterations: 100000, hash: 'SHA-256' },
      keyMaterial, { name: 'AES-GCM', length: 256 }, false, ['encrypt']
    );

    const encrypted = await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv }, key, encoder.encode(plaintext)
    );

    return {
      ciphertext: btoa(String.fromCharCode(...new Uint8Array(encrypted))),
      iv: btoa(String.fromCharCode(...iv)),
      salt: btoa(String.fromCharCode(...salt))
    };
  }
}

export const edgeDetector = new EdgeFlowStateDetector();
`;
}

// ═══════════════════════════════════════════════════════════════════════════════
// XYZA INTEGRATION #7: UNIFIED ANALYTICS AGGREGATOR
// ═══════════════════════════════════════════════════════════════════════════════

export interface SystemAnalytics {
  timestamp: string;
  userFlowStates: {
    total: number;
    stable: number;
    approaching: number;
    critical: number;
    transitioning: number;
  };
  domainSignals: {
    engagement: FusedSignal;
    security: FusedSignal;
    ux: FusedSignal;
    ai: FusedSignal;
  };
  noveltyReport: NoveltyReport;
  criticalInsightsTriggered: Array<{
    insightId: number;
    title: string;
    reason: string;
  }>;
  recommendations: string[];
}

export function aggregateSystemAnalytics(
  userStates: Array<{ userId: string; phase: string }>,
  domainSignals: MultiDomainSignal[],
  dataStreams: DataStream[],
  existingKnowledge: string[] = []
): SystemAnalytics {
  // Aggregate user states
  const userFlowStates = {
    total: userStates.length,
    stable: userStates.filter(u => u.phase === 'stable').length,
    approaching: userStates.filter(u => u.phase === 'approaching').length,
    critical: userStates.filter(u => u.phase === 'critical').length,
    transitioning: userStates.filter(u => u.phase === 'transitioning').length,
  };

  // Fuse domain signals
  const groupedSignals: Record<string, MultiDomainSignal[]> = {};
  for (const signal of domainSignals) {
    if (!groupedSignals[signal.domain]) {
      groupedSignals[signal.domain] = [];
    }
    groupedSignals[signal.domain].push(signal);
  }

  const domainFused: Record<string, FusedSignal> = {};
  for (const [domain, signals] of Object.entries(groupedSignals)) {
    domainFused[domain] = fuseMultiDomainSignals(signals);
  }

  // Scan for novelties
  const noveltyReport = scanForNovelties(dataStreams, existingKnowledge);

  // Check which critical insights are triggered
  const criticalInsightsTriggered: SystemAnalytics['criticalInsightsTriggered'] = [];
  const criticalInsights = getCriticalInsights();

  // NSM Insight #1: Variance quieting
  if (userFlowStates.approaching > userFlowStates.total * 0.2) {
    const insight = criticalInsights.find(i => i.id === 1);
    if (insight) {
      criticalInsightsTriggered.push({
        insightId: 1,
        title: insight.title,
        reason: `${userFlowStates.approaching} users (${((userFlowStates.approaching / userFlowStates.total) * 100).toFixed(1)}%) in approaching phase`,
      });
    }
  }

  // NSM Insight #2: Multi-domain coherence
  const coherentDomains = Object.values(domainFused).filter(f => f.coherenceScore > 0.7);
  if (coherentDomains.length >= 3) {
    const insight = criticalInsights.find(i => i.id === 2);
    if (insight) {
      criticalInsightsTriggered.push({
        insightId: 2,
        title: insight.title,
        reason: `${coherentDomains.length} domains showing coherent signals`,
      });
    }
  }

  // NSM Insight #16: Security quieting
  if (domainFused.security?.value > 0.5 && domainFused.security?.confidence > 0.7) {
    const insight = criticalInsights.find(i => i.id === 16);
    if (insight) {
      criticalInsightsTriggered.push({
        insightId: 16,
        title: insight.title,
        reason: `Security signal elevated: ${(domainFused.security.value * 100).toFixed(1)}%`,
      });
    }
  }

  // Compile recommendations
  const allRecommendations = new Set<string>();
  for (const domain of Object.values(domainFused)) {
    if (domain.recommendation !== 'Normal operation') {
      allRecommendations.add(domain.recommendation);
    }
  }
  for (const rec of noveltyReport.recommendations) {
    allRecommendations.add(rec);
  }
  for (const triggered of criticalInsightsTriggered) {
    const insight = NSM_INSIGHTS.find(i => i.id === triggered.insightId);
    if (insight) {
      allRecommendations.add(`[NSM #${insight.id}] ${insight.application}`);
    }
  }

  return {
    timestamp: new Date().toISOString(),
    userFlowStates,
    domainSignals: {
      engagement: domainFused.engagement || fuseMultiDomainSignals([]),
      security: domainFused.security || fuseMultiDomainSignals([]),
      ux: domainFused.ux || fuseMultiDomainSignals([]),
      ai: domainFused.ai || fuseMultiDomainSignals([]),
    },
    noveltyReport,
    criticalInsightsTriggered,
    recommendations: Array.from(allRecommendations),
  };
}

export {
  NSM_INSIGHTS,
  getCriticalInsights,
  generateRecommendations,
};
