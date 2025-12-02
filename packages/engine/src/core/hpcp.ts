/**
 * HPCP - Harmonic Pitch Class Profile Detection
 *
 * Adapted from music information retrieval for financial/social signal analysis.
 * Detects harmonic structure and phase changes in time series.
 *
 * KEY INSIGHT: Financial and social signals exhibit harmonic patterns
 * similar to music - detecting these "chords" reveals hidden structure.
 *
 * © 2025 Crystalline Labs LLC - Trade Secret
 */

import { FFT } from './fft';

export interface HPCPProfile {
  /** 12-bin chroma profile (like musical pitch classes) */
  chroma: number[];
  /** Dominant "pitch class" (strongest periodic component) */
  dominant: number;
  /** Harmonic strength (how periodic is the signal) */
  harmonicity: number;
  /** Phase angle of dominant component */
  phase: number;
}

export interface PhaseChangeResult {
  /** Was a phase change detected? */
  detected: boolean;
  /** Confidence in detection (0-1) */
  confidence: number;
  /** Position in signal where change occurred */
  position: number;
  /** Magnitude of the change */
  magnitude: number;
  /** Type of phase change */
  type: 'abrupt' | 'gradual' | 'oscillatory';
  /** Pre-change profile */
  before: HPCPProfile;
  /** Post-change profile */
  after: HPCPProfile;
}

/**
 * HPCP Analysis for Signal Intelligence
 *
 * SECRET SAUCE:
 * - Adapted chroma features for financial signals
 * - Proprietary binning based on market cycles
 * - Phase-locked loop inspired tracking
 * - Multi-resolution analysis
 */
export class HPCP {
  // Proprietary frequency bins (based on market/social cycle analysis)
  // These correspond to common periodicities in financial data
  private static readonly CYCLE_BINS = [
    2, // Ultra-short (2-period cycles)
    3, // Short-term
    5, // Weekly-ish
    8, // Fibonacci
    13, // Bi-weekly
    21, // Monthly
    34, // Fibonacci
    55, // Quarterly-ish
    89, // Fibonacci
    144, // Semi-annual
    233, // Fibonacci
    365, // Annual
  ];

  private fft: FFT;
  private windowSize: number;
  private history: HPCPProfile[] = [];

  constructor(windowSize = 256) {
    this.windowSize = windowSize;
    this.fft = new FFT(windowSize);
  }

  /**
   * Compute HPCP profile for a signal segment
   */
  analyze(signal: number[]): HPCPProfile {
    // Pad or truncate to window size
    const padded = new Float64Array(this.windowSize);
    for (let i = 0; i < this.windowSize; i++) {
      padded[i] = i < signal.length ? signal[i] : 0;
    }

    // Apply window function
    const windowed = this.applyWindow(padded);

    // Compute FFT
    const spectrum = this.fft.forward(Array.from(windowed));
    const magnitude = this.fft.magnitude(spectrum);
    const phase = this.fft.phase(spectrum);

    // Map to chroma bins (12 bins like musical pitch classes)
    const chroma = this.computeChroma(magnitude);

    // Find dominant component
    const dominantIdx = chroma.indexOf(Math.max(...chroma));

    // Calculate harmonicity (ratio of harmonic to total energy)
    const harmonicity = this.calculateHarmonicity(magnitude, dominantIdx);

    // Get phase of dominant component
    const dominantFreqIdx = this.chromaToFrequencyIdx(dominantIdx);
    const dominantPhase = phase[dominantFreqIdx] ?? 0;

    const profile: HPCPProfile = {
      chroma,
      dominant: dominantIdx,
      harmonicity,
      phase: dominantPhase,
    };

    this.history.push(profile);
    if (this.history.length > 100) {
      this.history.shift();
    }

    return profile;
  }

  /**
   * Detect phase change in signal
   */
  detectPhaseChange(signal: number[], sensitivity = 0.5): PhaseChangeResult {
    const halfLen = Math.floor(signal.length / 2);

    // Analyze before and after segments
    const before = this.analyze(signal.slice(0, halfLen));
    const after = this.analyze(signal.slice(halfLen));

    // Measure change in chroma profile
    const chromaDistance = this.euclideanDistance(before.chroma, after.chroma);

    // Measure phase shift
    const phaseShift = Math.abs(after.phase - before.phase);
    const normalizedPhaseShift = phaseShift / Math.PI;

    // Measure harmonicity change
    const harmonicityChange = Math.abs(after.harmonicity - before.harmonicity);

    // Combined change magnitude
    const magnitude = chromaDistance * 0.4 + normalizedPhaseShift * 0.3 + harmonicityChange * 0.3;

    // Determine if significant
    const threshold = 0.3 * (1 - sensitivity);
    const detected = magnitude > threshold;

    // Classify type of change
    const type = this.classifyChangeType(before, after, signal);

    // Find precise position of change
    const position = detected ? this.findChangePosition(signal) : halfLen;

    // Calculate confidence
    const confidence = Math.min(1, magnitude / (threshold * 2));

    return {
      detected,
      confidence,
      position,
      magnitude,
      type,
      before,
      after,
    };
  }

  /**
   * Continuous phase tracking (phase-locked loop inspired)
   */
  trackPhase(signals: number[][]): Array<{
    index: number;
    phase: number;
    frequency: number;
    locked: boolean;
  }> {
    const results: Array<{
      index: number;
      phase: number;
      frequency: number;
      locked: boolean;
    }> = [];

    let lastPhase = 0;
    let lastFreq = 0;
    let lockCount = 0;

    for (let i = 0; i < signals.length; i++) {
      const profile = this.analyze(signals[i]);

      // Estimate instantaneous frequency from phase derivative
      this.unwrapPhase(profile.phase - lastPhase);
      const estFreq = HPCP.CYCLE_BINS[profile.dominant] ?? 1;

      // Check if "locked" (stable frequency)
      const freqError = Math.abs(estFreq - lastFreq) / (lastFreq || 1);
      if (freqError < 0.1) {
        lockCount++;
      } else {
        lockCount = 0;
      }

      results.push({
        index: i,
        phase: profile.phase,
        frequency: estFreq,
        locked: lockCount > 3,
      });

      lastPhase = profile.phase;
      lastFreq = estFreq;
    }

    return results;
  }

  /**
   * Multi-resolution HPCP analysis
   */
  multiResolution(signal: number[]): Map<number, HPCPProfile> {
    const results = new Map<number, HPCPProfile>();
    const resolutions = [32, 64, 128, 256, 512];

    for (const res of resolutions) {
      if (signal.length >= res) {
        const analyzer = new HPCP(res);
        results.set(res, analyzer.analyze(signal.slice(-res)));
      }
    }

    return results;
  }

  /**
   * Compute chroma from magnitude spectrum
   */
  private computeChroma(magnitude: number[]): number[] {
    const chroma = new Float64Array(12);
    const n = magnitude.length;

    // Map frequency bins to chroma bins
    for (let i = 1; i < n / 2; i++) {
      const freq = i; // Normalized frequency
      const chromaBin = this.frequencyToChroma(freq);
      chroma[chromaBin] += magnitude[i] * magnitude[i]; // Use power
    }

    // Normalize
    const maxVal = Math.max(...chroma);
    if (maxVal > 0) {
      for (let i = 0; i < 12; i++) {
        chroma[i] /= maxVal;
      }
    }

    return Array.from(chroma);
  }

  /**
   * Map frequency to chroma bin (0-11)
   */
  private frequencyToChroma(freq: number): number {
    // Map to nearest cycle bin, then to chroma
    let nearestBin = 0;
    let minDist = Infinity;

    for (let i = 0; i < HPCP.CYCLE_BINS.length; i++) {
      const dist = Math.abs(freq - HPCP.CYCLE_BINS[i]);
      if (dist < minDist) {
        minDist = dist;
        nearestBin = i;
      }
    }

    return nearestBin;
  }

  /**
   * Map chroma bin back to frequency index
   */
  private chromaToFrequencyIdx(chromaBin: number): number {
    const cycle = HPCP.CYCLE_BINS[chromaBin] ?? 1;
    return Math.round(this.windowSize / cycle);
  }

  /**
   * Calculate harmonicity measure
   */
  private calculateHarmonicity(magnitude: number[], dominantBin: number): number {
    const cycle = HPCP.CYCLE_BINS[dominantBin] ?? 1;
    const fundamentalIdx = Math.round(this.windowSize / cycle);

    // Sum energy at harmonics
    let harmonicEnergy = 0;
    let totalEnergy = 0;

    for (let i = 1; i < magnitude.length / 2; i++) {
      const energy = magnitude[i] * magnitude[i];
      totalEnergy += energy;

      // Check if this is a harmonic of the fundamental
      const ratio = i / fundamentalIdx;
      if (Math.abs(ratio - Math.round(ratio)) < 0.1) {
        harmonicEnergy += energy;
      }
    }

    return totalEnergy > 0 ? harmonicEnergy / totalEnergy : 0;
  }

  /**
   * Apply Hann window
   */
  private applyWindow(signal: Float64Array): Float64Array {
    const windowed = new Float64Array(signal.length);
    for (let i = 0; i < signal.length; i++) {
      const window = 0.5 * (1 - Math.cos((2 * Math.PI * i) / signal.length));
      windowed[i] = signal[i] * window;
    }
    return windowed;
  }

  /**
   * Euclidean distance between vectors
   */
  private euclideanDistance(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      sum += (a[i] - b[i]) ** 2;
    }
    return Math.sqrt(sum);
  }

  /**
   * Classify type of phase change
   */
  private classifyChangeType(
    before: HPCPProfile,
    after: HPCPProfile,
    signal: number[]
  ): 'abrupt' | 'gradual' | 'oscillatory' {
    // Check for oscillatory pattern
    if (
      Math.abs(before.harmonicity - after.harmonicity) < 0.2 &&
      before.dominant === after.dominant
    ) {
      return 'oscillatory';
    }

    // Check transition sharpness
    const midProfile = this.analyze(signal.slice(signal.length / 4, (3 * signal.length) / 4));
    const midDist = this.euclideanDistance(before.chroma, midProfile.chroma);
    const fullDist = this.euclideanDistance(before.chroma, after.chroma);

    if (midDist < fullDist * 0.3) {
      return 'abrupt';
    }

    return 'gradual';
  }

  /**
   * Find precise position of change using CUSUM
   */
  private findChangePosition(signal: number[]): number {
    // CUSUM (Cumulative Sum) change detection
    const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
    const cusum: number[] = [];
    let cumulative = 0;

    for (const val of signal) {
      cumulative += val - mean;
      cusum.push(cumulative);
    }

    // Find position of maximum absolute CUSUM
    let maxIdx = 0;
    let maxVal = 0;

    for (let i = 0; i < cusum.length; i++) {
      if (Math.abs(cusum[i]) > maxVal) {
        maxVal = Math.abs(cusum[i]);
        maxIdx = i;
      }
    }

    return maxIdx;
  }

  /**
   * Unwrap phase to handle discontinuities
   */
  private unwrapPhase(phase: number): number {
    while (phase > Math.PI) phase -= 2 * Math.PI;
    while (phase < -Math.PI) phase += 2 * Math.PI;
    return phase;
  }
}

/**
 * Convenient function for phase change detection
 */
export function detectPhaseChange(signal: number[], sensitivity = 0.5): PhaseChangeResult {
  const hpcp = new HPCP(Math.min(256, signal.length));
  return hpcp.detectPhaseChange(signal, sensitivity);
}
