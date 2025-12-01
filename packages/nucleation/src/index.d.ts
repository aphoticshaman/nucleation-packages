/**
 * Nucleation - Early Warning Systems for Phase Transitions
 */

// Re-exports from domain packages
export { RegimeDetector, detectRegimeShift } from 'regime-shift';
export { ThreatDetector, ThreatCorrelator, assessThreat } from 'threat-pulse';
export { ChurnDetector, CohortMonitor, assessChurnRisk } from 'churn-harbinger';
export { TeamHealthMonitor, IntegrationMonitor } from 'org-canary';
export { default as SupplyMonitor } from 'supply-sentinel';
export { SensorMonitor } from 'sensor-shift';
export { CrowdMonitor } from 'crowd-phase';
export { PatientMonitor } from 'patient-drift';
export { MatchMonitor } from 'match-pulse';
export { TransitionDetector } from 'market-canary';

/** Standard alert levels */
export const LEVELS: {
  GREEN: 0;
  YELLOW: 1;
  ORANGE: 2;
  RED: 3;
};

/** Supported domains */
export type Domain =
  | 'finance'
  | 'security'
  | 'saas'
  | 'churn'
  | 'hr'
  | 'org'
  | 'supply'
  | 'iot'
  | 'sensor'
  | 'social'
  | 'community'
  | 'health'
  | 'patient'
  | 'gaming'
  | 'esports'
  | 'general';

/** Sensitivity presets */
export type Sensitivity = 'conservative' | 'balanced' | 'sensitive';

/** Configuration options */
export interface MonitorOptions {
  sensitivity?: Sensitivity;
  windowSize?: number;
  threshold?: number;
}

/** Normalized state output */
export interface NormalizedState {
  /** Human-readable level string */
  level: string;
  /** Numeric level: 0=green, 1=yellow, 2=orange, 3=red */
  levelNumeric: 0 | 1 | 2 | 3;
  /** Whether a phase transition is occurring */
  transitioning: boolean;
  /** Detection confidence (0-1) */
  confidence: number | null;
  /** Current variance value */
  variance: number | null;
  /** Timestamp of this update */
  timestamp: number;
  /** Original detector state */
  raw: Record<string, any>;
}

/** Event types */
export type MonitorEvent = 'warning' | 'critical' | 'transition' | 'update';

/** Event callback */
export type EventCallback = (state: NormalizedState) => void;

/** Monitored detector with event emitter interface */
export interface MonitoredDetector {
  /** Register event listener */
  on(event: MonitorEvent, callback: EventCallback): this;
  /** Remove event listener */
  off(event: MonitorEvent, callback: EventCallback): this;
  /** Process new observation */
  update(value: number): NormalizedState;
  /** Batch update */
  updateBatch(values: number[]): NormalizedState;
  /** Get current state without new data */
  current(): NormalizedState;
  /** Reset detector state */
  reset(): void;
  /** Serialize for persistence */
  serialize(): string;
  /** Pipe output to destination */
  pipe<T>(destination: T): T;
  /** Filter updates */
  filter(predicate: (state: NormalizedState) => boolean): MonitoredDetector;
}

/**
 * Create a monitored detector for a domain
 *
 * @example
 * const detector = await monitor('finance');
 * detector.on('warning', state => console.log('Alert:', state));
 * detector.update(100.5);
 */
export function monitor(domain: Domain, options?: MonitorOptions): Promise<MonitoredDetector>;

/**
 * Create multiple monitors at once
 *
 * @example
 * const monitors = await createMonitors({
 *   finance: { sensitivity: 'balanced' },
 *   security: { sensitivity: 'sensitive' },
 * });
 */
export function createMonitors(
  config: Record<string, MonitorOptions>
): Promise<Record<string, MonitoredDetector>>;

/** Webhook processor configuration */
export interface WebhookProcessorConfig extends MonitorOptions {
  domain: Domain;
  port?: number;
  extract?: (data: any) => number;
  onWarning?: EventCallback;
  onCritical?: EventCallback;
  onAlert?: EventCallback;
}

/** Webhook processor */
export interface WebhookProcessor {
  start(): Promise<import('http').Server>;
}

/**
 * Create a webhook processor
 *
 * @example
 * const processor = createWebhookProcessor({
 *   domain: 'finance',
 *   port: 8080,
 *   onAlert: state => sendSlack(state)
 * });
 * processor.start();
 */
export function createWebhookProcessor(config: WebhookProcessorConfig): WebhookProcessor;

/** Prometheus exporter options */
export interface PrometheusExporterOptions {
  prefix?: string;
  labels?: Record<string, string>;
}

/** Prometheus exporter */
export interface PrometheusExporter {
  metrics(): string;
}

/**
 * Create a Prometheus metrics exporter
 *
 * @example
 * const exporter = createPrometheusExporter(detector, {
 *   prefix: 'myapp_',
 *   labels: { service: 'trading' }
 * });
 * app.get('/metrics', (req, res) => res.send(exporter.metrics()));
 */
export function createPrometheusExporter(
  detector: MonitoredDetector,
  options?: PrometheusExporterOptions
): PrometheusExporter;

export default monitor;
