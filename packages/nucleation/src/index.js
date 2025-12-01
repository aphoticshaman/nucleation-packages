/**
 * Nucleation - Early Warning Systems for Phase Transitions
 *
 * Detect the calm before the storm.
 *
 * @example Zero-config quickstart
 * ```javascript
 * import { monitor } from 'nucleation';
 *
 * const detector = await monitor('finance');
 * detector.on('warning', state => console.log('Transition approaching:', state));
 *
 * for (const price of priceStream) {
 *   detector.update(price);
 * }
 * ```
 *
 * @example Full control
 * ```javascript
 * import { RegimeDetector } from 'nucleation';
 *
 * const detector = new RegimeDetector({
 *   sensitivity: 'sensitive',
 *   windowSize: 20
 * });
 * await detector.init();
 *
 * const state = detector.update(100.5);
 * if (state.transitioning) alert(state);
 * ```
 */

// Re-export all domain detectors
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

// Domain mappings for convenience API
const DOMAIN_MAP = {
  finance: async () => (await import('regime-shift')).RegimeDetector,
  security: async () => (await import('threat-pulse')).ThreatDetector,
  saas: async () => (await import('churn-harbinger')).ChurnDetector,
  churn: async () => (await import('churn-harbinger')).ChurnDetector,
  hr: async () => (await import('org-canary')).TeamHealthMonitor,
  org: async () => (await import('org-canary')).TeamHealthMonitor,
  supply: async () => (await import('supply-sentinel')).default,
  iot: async () => (await import('sensor-shift')).SensorMonitor,
  sensor: async () => (await import('sensor-shift')).SensorMonitor,
  social: async () => (await import('crowd-phase')).CrowdMonitor,
  community: async () => (await import('crowd-phase')).CrowdMonitor,
  health: async () => (await import('patient-drift')).PatientMonitor,
  patient: async () => (await import('patient-drift')).PatientMonitor,
  gaming: async () => (await import('match-pulse')).MatchMonitor,
  esports: async () => (await import('match-pulse')).MatchMonitor,
  general: async () => (await import('market-canary')).TransitionDetector,
};

/**
 * Standard output state shape across all detectors
 */
export const LEVELS = {
  GREEN: 0,
  YELLOW: 1,
  ORANGE: 2,
  RED: 3,
};

/**
 * Wrap a detector with event emitter interface
 */
class MonitoredDetector {
  #detector;
  #listeners = { warning: [], critical: [], transition: [], update: [] };
  #lastLevel = 0;

  constructor(detector) {
    this.#detector = detector;
  }

  /**
   * Register event listener
   * @param {'warning'|'critical'|'transition'|'update'} event
   * @param {function} callback
   */
  on(event, callback) {
    if (this.#listeners[event]) {
      this.#listeners[event].push(callback);
    }
    return this;
  }

  /**
   * Remove event listener
   */
  off(event, callback) {
    if (this.#listeners[event]) {
      this.#listeners[event] = this.#listeners[event].filter((cb) => cb !== callback);
    }
    return this;
  }

  /**
   * Process new observation
   */
  update(value) {
    const state = this.#detector.update(value);
    const normalized = this.#normalizeState(state);

    // Emit events
    this.#emit('update', normalized);

    if (normalized.levelNumeric >= LEVELS.YELLOW && this.#lastLevel < LEVELS.YELLOW) {
      this.#emit('warning', normalized);
    }

    if (normalized.levelNumeric >= LEVELS.RED && this.#lastLevel < LEVELS.RED) {
      this.#emit('critical', normalized);
    }

    if (normalized.transitioning && !this.#lastTransitioning) {
      this.#emit('transition', normalized);
    }

    this.#lastLevel = normalized.levelNumeric;
    this.#lastTransitioning = normalized.transitioning;

    return normalized;
  }

  #lastTransitioning = false;

  /**
   * Batch update
   */
  updateBatch(values) {
    let lastState;
    for (const value of values) {
      lastState = this.update(value);
    }
    return lastState;
  }

  /**
   * Get current state without new data
   */
  current() {
    return this.#normalizeState(this.#detector.current());
  }

  /**
   * Reset detector state
   */
  reset() {
    this.#detector.reset();
    this.#lastLevel = 0;
    this.#lastTransitioning = false;
  }

  /**
   * Serialize for persistence (Lambda, Edge, etc.)
   */
  serialize() {
    return this.#detector.serialize();
  }

  /**
   * Pipe output to another function or stream
   */
  pipe(destination) {
    this.on('update', (state) => {
      if (typeof destination === 'function') {
        destination(state);
      } else if (destination.write) {
        destination.write(state);
      } else if (destination.update) {
        destination.update(state);
      }
    });
    return destination;
  }

  /**
   * Filter updates
   */
  filter(predicate) {
    const filtered = new FilteredMonitor(predicate);
    this.pipe(filtered);
    return filtered;
  }

  #emit(event, data) {
    for (const callback of this.#listeners[event] || []) {
      try {
        callback(data);
      } catch (err) {
        console.error(`Nucleation: Error in ${event} listener:`, err);
      }
    }
  }

  #normalizeState(state) {
    // Map various detector outputs to standard shape
    const level =
      state.regime ||
      state.level ||
      state.phase ||
      state.health ||
      state.riskLevel ||
      state.tensionLevel ||
      state.alertLevel ||
      state.tiltLevel ||
      'unknown';

    const levelNumeric = this.#levelToNumeric(level, state);

    return {
      level,
      levelNumeric,
      transitioning:
        state.isShifting ||
        state.transitioning ||
        state.failing ||
        state.atRisk ||
        state.critical ||
        state.volatile ||
        state.churning ||
        state.tilted ||
        false,
      confidence: state.confidence ?? null,
      variance: state.variance ?? null,
      timestamp: Date.now(),
      raw: state,
    };
  }

  #levelToNumeric(level, state) {
    // Handle boolean flags
    if (
      state.isShifting ||
      state.critical ||
      state.failing ||
      state.churning ||
      state.volatile ||
      state.tilted
    ) {
      return LEVELS.RED;
    }
    if (
      state.isWarning ||
      state.elevated ||
      state.atRisk ||
      state.stressed ||
      state.heated ||
      state.frustrated
    ) {
      return LEVELS.ORANGE;
    }
    if (
      state.warming ||
      state.cooling ||
      state.strained ||
      state.tense ||
      state.watch ||
      state.degrading
    ) {
      return LEVELS.YELLOW;
    }

    // Handle string levels
    const lowerLevel = String(level).toLowerCase();
    if (
      [
        'critical',
        'red',
        'shifting',
        'failing',
        'churning',
        'volatile',
        'tilted',
        'disrupted',
        'toxic',
      ].includes(lowerLevel)
    ) {
      return LEVELS.RED;
    }
    if (['warning', 'orange', 'at-risk', 'atrisk', 'stressed', 'heated'].includes(lowerLevel)) {
      return LEVELS.ORANGE;
    }
    if (
      [
        'yellow',
        'warming',
        'cooling',
        'strained',
        'tense',
        'watch',
        'elevated',
        'approaching',
        'degrading',
        'frustrated',
      ].includes(lowerLevel)
    ) {
      return LEVELS.YELLOW;
    }

    return LEVELS.GREEN;
  }
}

/**
 * Filtered monitor for chaining
 */
class FilteredMonitor extends MonitoredDetector {
  #predicate;
  #listeners = { warning: [], critical: [], transition: [], update: [] };

  constructor(predicate) {
    super({ update: () => ({}), current: () => ({}), reset: () => {}, serialize: () => '{}' });
    this.#predicate = predicate;
  }

  write(state) {
    if (this.#predicate(state)) {
      for (const cb of this.#listeners.update) cb(state);
    }
  }

  on(event, callback) {
    if (this.#listeners[event]) {
      this.#listeners[event].push(callback);
    }
    return this;
  }
}

/**
 * Zero-config monitor factory
 *
 * @param {string} domain - One of: finance, security, saas, hr, supply, iot, social, health, gaming, general
 * @param {object} options - Optional configuration
 * @returns {Promise<MonitoredDetector>}
 *
 * @example
 * const detector = await monitor('finance');
 * detector.on('warning', console.log);
 * detector.update(100.5);
 */
export async function monitor(domain, options = {}) {
  const domainKey = domain.toLowerCase();
  const DetectorClass = await DOMAIN_MAP[domainKey]?.();

  if (!DetectorClass) {
    const available = Object.keys(DOMAIN_MAP).join(', ');
    throw new Error(`Unknown domain: "${domain}". Available: ${available}`);
  }

  const detector = new DetectorClass({
    sensitivity: options.sensitivity || 'balanced',
    windowSize: options.windowSize,
    threshold: options.threshold,
  });

  await detector.init();

  return new MonitoredDetector(detector);
}

/**
 * Create multiple monitors at once
 *
 * @example
 * const monitors = await createMonitors({
 *   finance: { sensitivity: 'balanced' },
 *   security: { sensitivity: 'sensitive' },
 * });
 *
 * monitors.finance.on('warning', handleFinanceWarning);
 * monitors.security.on('critical', handleSecurityCritical);
 */
export async function createMonitors(config) {
  const monitors = {};

  for (const [domain, options] of Object.entries(config)) {
    monitors[domain] = await monitor(domain, options);
  }

  return monitors;
}

/**
 * Webhook processor - receives data via HTTP, emits alerts
 *
 * @example
 * const processor = createWebhookProcessor({
 *   domain: 'finance',
 *   port: 8080,
 *   onAlert: async (state) => {
 *     await fetch('https://hooks.slack.com/...', {
 *       method: 'POST',
 *       body: JSON.stringify({ text: `Alert: ${state.level}` })
 *     });
 *   }
 * });
 * processor.start();
 */
export function createWebhookProcessor(config) {
  return {
    async start() {
      const { createServer } = await import('http');
      const detector = await monitor(config.domain, config);

      if (config.onWarning) detector.on('warning', config.onWarning);
      if (config.onCritical) detector.on('critical', config.onCritical);
      if (config.onAlert) {
        detector.on('warning', config.onAlert);
        detector.on('critical', config.onAlert);
      }

      const server = createServer(async (req, res) => {
        if (req.method === 'POST') {
          let body = '';
          for await (const chunk of req) body += chunk;

          try {
            const data = JSON.parse(body);
            const value = config.extract ? config.extract(data) : data.value;
            const state = detector.update(value);

            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(state));
          } catch (err) {
            res.writeHead(400);
            res.end(JSON.stringify({ error: err.message }));
          }
        } else if (req.method === 'GET') {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(detector.current()));
        } else {
          res.writeHead(405);
          res.end();
        }
      });

      const port = config.port || 8080;
      server.listen(port);
      console.log(`Nucleation webhook processor listening on :${port}`);

      return server;
    },
  };
}

/**
 * Prometheus metrics exporter
 */
export function createPrometheusExporter(detector, options = {}) {
  const prefix = options.prefix || 'nucleation_';
  const labels = options.labels || {};
  const labelStr = Object.entries(labels)
    .map(([k, v]) => `${k}="${v}"`)
    .join(',');
  const labelPart = labelStr ? `{${labelStr}}` : '';

  return {
    metrics() {
      const state = detector.current();
      return [
        `# HELP ${prefix}level Current alert level (0=green, 1=yellow, 2=orange, 3=red)`,
        `# TYPE ${prefix}level gauge`,
        `${prefix}level${labelPart} ${state.levelNumeric}`,
        `# HELP ${prefix}variance Current variance value`,
        `# TYPE ${prefix}variance gauge`,
        `${prefix}variance${labelPart} ${state.variance ?? 0}`,
        `# HELP ${prefix}confidence Detection confidence (0-1)`,
        `# TYPE ${prefix}confidence gauge`,
        `${prefix}confidence${labelPart} ${state.confidence ?? 0}`,
        `# HELP ${prefix}transitioning Whether a phase transition is occurring`,
        `# TYPE ${prefix}transitioning gauge`,
        `${prefix}transitioning${labelPart} ${state.transitioning ? 1 : 0}`,
      ].join('\n');
    },
  };
}

// Default export for simple usage
export default monitor;
