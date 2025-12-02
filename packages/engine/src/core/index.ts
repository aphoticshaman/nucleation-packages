/**
 * LatticeForge Core Signal Processing
 *
 * Native TypeScript implementations of signal processing algorithms.
 * These provide the mathematical foundation without external dependencies.
 *
 * Includes:
 * - FFT (Fast Fourier Transform)
 * - HPCP (Harmonic Pitch Class Profile) detection
 * - Correlation functions
 * - Statistical utilities
 * - Time series analysis
 *
 * © 2025 Crystalline Labs LLC - Trade Secret
 */

export { FFT, fft, ifft } from './fft';
export { HPCP, detectPhaseChange } from './hpcp';
export { Statistics, correlation, variance, zscore } from './statistics';
export { TimeSeries, movingAverage, exponentialSmooth, differentiate } from './timeseries';
export { Wavelets, waveletTransform } from './wavelets';
