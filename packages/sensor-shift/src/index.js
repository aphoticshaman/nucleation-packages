import { createRequire } from 'module';
import { readFile } from 'fs/promises';

let nucleationModule = null;
let initialized = false;

export async function initialize() {
  if (!initialized) {
    nucleationModule = await import('nucleation-wasm');
    if (typeof window === 'undefined') {
      const require = createRequire(import.meta.url);
      const wasmPath = require.resolve('nucleation-wasm/nucleation_bg.wasm');
      const wasmBytes = await readFile(wasmPath);
      await nucleationModule.default(wasmBytes);
    } else {
      await nucleationModule.default();
    }
    initialized = true;
  }
}

export const HealthLevel = {
  NORMAL: 'normal',
  DEGRADING: 'degrading',
  WARNING: 'warning',
  FAILING: 'failing',
};

function phaseToLevel(phase) {
  const Phase = nucleationModule.Phase;
  switch (phase) {
    case Phase.Stable:
      return HealthLevel.NORMAL;
    case Phase.Approaching:
      return HealthLevel.DEGRADING;
    case Phase.Critical:
      return HealthLevel.WARNING;
    case Phase.Transitioning:
      return HealthLevel.FAILING;
    default:
      return HealthLevel.NORMAL;
  }
}

export class SensorMonitor {
  #detector = null;
  #config = null;
  #initialized = false;

  constructor(config = {}) {
    this.#config = {
      sensitivity: config.sensitivity || 'balanced',
      windowSize: config.windowSize || 50,
      threshold: config.threshold,
    };
  }

  async init() {
    await initialize();
    const { DetectorConfig, NucleationDetector } = nucleationModule;
    let detectorConfig;
    switch (this.#config.sensitivity) {
      case 'conservative':
        detectorConfig = DetectorConfig.conservative();
        break;
      case 'sensitive':
        detectorConfig = DetectorConfig.sensitive();
        break;
      default:
        detectorConfig = new DetectorConfig();
    }
    detectorConfig.window_size = this.#config.windowSize;
    if (this.#config.threshold) detectorConfig.threshold = this.#config.threshold;
    this.#detector = new NucleationDetector(detectorConfig);
    this.#initialized = true;
  }

  #ensureInit() {
    if (!this.#initialized) throw new Error('SensorMonitor not initialized. Call init() first.');
  }

  update(value) {
    this.#ensureInit();
    const phase = this.#detector.update(value);
    const level = phaseToLevel(phase);
    const Phase = nucleationModule.Phase;
    return {
      level,
      failing: phase === Phase.Critical || phase === Phase.Transitioning,
      elevated: phase !== Phase.Stable,
      confidence: this.#detector.confidence(),
      variance: this.#detector.currentVariance(),
      trend: this.#detector.inflectionMagnitude(),
      dataPoints: this.#detector.count(),
    };
  }

  updateBatch(values) {
    this.#ensureInit();
    const arr = values instanceof Float64Array ? values : new Float64Array(values);
    const phase = this.#detector.update_batch(arr);
    const level = phaseToLevel(phase);
    const Phase = nucleationModule.Phase;
    return {
      level,
      failing: phase === Phase.Critical || phase === Phase.Transitioning,
      elevated: phase !== Phase.Stable,
      confidence: this.#detector.confidence(),
      variance: this.#detector.currentVariance(),
      trend: this.#detector.inflectionMagnitude(),
      dataPoints: this.#detector.count(),
    };
  }

  current() {
    this.#ensureInit();
    const phase = this.#detector.currentPhase();
    const level = phaseToLevel(phase);
    const Phase = nucleationModule.Phase;
    return {
      level,
      failing: phase === Phase.Critical || phase === Phase.Transitioning,
      elevated: phase !== Phase.Stable,
      confidence: this.#detector.confidence(),
      variance: this.#detector.currentVariance(),
      trend: this.#detector.inflectionMagnitude(),
      dataPoints: this.#detector.count(),
    };
  }

  reset() {
    this.#ensureInit();
    this.#detector.reset();
  }
  serialize() {
    this.#ensureInit();
    return this.#detector.serialize();
  }

  static async deserialize(json) {
    await initialize();
    const { NucleationDetector } = nucleationModule;
    const instance = new SensorMonitor();
    instance.#detector = NucleationDetector.deserialize(json);
    instance.#initialized = true;
    return instance;
  }
}

export async function assess(values, config = {}) {
  const detector = new SensorMonitor(config);
  await detector.init();
  const state = detector.updateBatch(values);
  return {
    failing: state.failing,
    elevated: state.elevated,
    level: state.level,
    confidence: state.confidence,
  };
}

export default SensorMonitor;
