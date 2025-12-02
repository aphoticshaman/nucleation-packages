/**
 * Core Signal Processing Tests
 *
 * Tests for FFT, HPCP, Statistics, and TimeSeries modules.
 */

import { describe, it, expect } from 'vitest';
import { FFT } from '../core/fft';
import { HPCP, detectPhaseChange } from '../core/hpcp';
import { Statistics } from '../core/statistics';
import { TimeSeries } from '../core/timeseries';
import { Wavelets } from '../core/wavelets';

describe('FFT', () => {
  it('should compute FFT of a simple signal', () => {
    const signal = [1, 0, -1, 0, 1, 0, -1, 0];
    const fftObj = new FFT(8);
    const spectrum = fftObj.forward(signal);

    expect(spectrum).toHaveLength(8);
    // DC component should be near 0
    expect(Math.abs(spectrum[0].re)).toBeLessThan(0.01);
  });

  it('should detect dominant frequency', () => {
    // Generate a pure sinusoid at frequency 2
    const n = 64;
    const signal = Array(n)
      .fill(0)
      .map((_, i) => Math.sin((2 * Math.PI * 2 * i) / n));

    const fftObj = new FFT(n);
    const spectrum = fftObj.forward(signal);
    const dominant = fftObj.dominantFrequencies(spectrum, n, 1);

    expect(dominant).toHaveLength(1);
    expect(dominant[0].frequency).toBeCloseTo(2, 0);
  });

  it('should correctly invert FFT', () => {
    const original = [1, 2, 3, 4, 5, 6, 7, 8];
    const fftObj = new FFT(8);

    const spectrum = fftObj.forward(original);
    const reconstructed = fftObj.inverse(spectrum);

    for (let i = 0; i < original.length; i++) {
      expect(reconstructed[i]).toBeCloseTo(original[i], 5);
    }
  });
});

describe('HPCP', () => {
  it('should analyze signal harmonics', () => {
    const hpcp = new HPCP(64);
    const signal = Array(64)
      .fill(0)
      .map(
        (_, i) => Math.sin((2 * Math.PI * 4 * i) / 64) + 0.5 * Math.sin((2 * Math.PI * 8 * i) / 64)
      );

    const profile = hpcp.analyze(signal);

    expect(profile.chroma).toHaveLength(12);
    expect(profile.harmonicity).toBeGreaterThan(0);
    expect(profile.harmonicity).toBeLessThanOrEqual(1);
  });

  it('should detect phase change', () => {
    // Create signal with regime change
    const before = Array(32)
      .fill(0)
      .map((_, i) => Math.sin((2 * Math.PI * 2 * i) / 32));
    const after = Array(32)
      .fill(0)
      .map((_, i) => Math.sin((2 * Math.PI * 8 * i) / 32));
    const signal = [...before, ...after];

    const result = detectPhaseChange(signal, 0.3);

    // Should return valid structure
    expect(result).toBeDefined();
    expect(typeof result.detected).toBe('boolean');
    expect(typeof result.position).toBe('number');
    expect(result.position).toBeGreaterThanOrEqual(0);
    expect(result.position).toBeLessThan(signal.length);
  });
});

describe('Statistics', () => {
  it('should calculate mean correctly', () => {
    expect(Statistics.mean([1, 2, 3, 4, 5])).toBe(3);
    expect(Statistics.mean([10, 20, 30])).toBeCloseTo(20);
  });

  it('should calculate variance correctly', () => {
    const data = [2, 4, 4, 4, 5, 5, 7, 9];
    expect(Statistics.variance(data)).toBeCloseTo(4.57, 1);
  });

  it('should calculate correlation correctly', () => {
    const x = [1, 2, 3, 4, 5];
    const y = [2, 4, 6, 8, 10]; // Perfect correlation

    expect(Statistics.correlation(x, y)).toBeCloseTo(1, 5);
  });

  it('should calculate negative correlation', () => {
    const x = [1, 2, 3, 4, 5];
    const y = [10, 8, 6, 4, 2]; // Perfect negative correlation

    expect(Statistics.correlation(x, y)).toBeCloseTo(-1, 5);
  });

  it('should calculate z-scores correctly', () => {
    const data = [10, 20, 30, 40, 50];
    const zscores = Statistics.zscore(data);

    expect(zscores[2]).toBeCloseTo(0, 5); // Middle value has z=0
    expect(zscores[0]).toBeLessThan(0); // Lower values negative
    expect(zscores[4]).toBeGreaterThan(0); // Higher values positive
  });

  it('should calculate Sharpe ratio', () => {
    const returns = [0.1, 0.05, -0.02, 0.08, 0.03];
    const sharpe = Statistics.sharpeRatio(returns, 0.02);

    expect(sharpe).toBeGreaterThan(0);
  });

  it('should calculate max drawdown', () => {
    const values = [100, 110, 105, 95, 102, 98];
    const maxDD = Statistics.maxDrawdown(values);

    // Max drawdown from 110 to 95 = 13.6%
    expect(maxDD).toBeCloseTo(0.136, 2);
  });
});

describe('TimeSeries', () => {
  it('should calculate moving average', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const sma = TimeSeries.sma(data, 3);

    expect(sma).toHaveLength(8);
    expect(sma[0]).toBe(2); // (1+2+3)/3
    expect(sma[1]).toBe(3); // (2+3+4)/3
  });

  it('should calculate exponential moving average', () => {
    const data = [10, 11, 12, 13, 14, 15];
    const ema = TimeSeries.ema(data, 0.5);

    expect(ema).toHaveLength(6);
    expect(ema[0]).toBe(10); // First value unchanged
    expect(ema[ema.length - 1]).toBeGreaterThan(ema[0]);
  });

  it('should calculate returns', () => {
    const prices = [100, 110, 105, 115];
    const returns = TimeSeries.pctReturns(prices);

    expect(returns).toHaveLength(3);
    expect(returns[0]).toBeCloseTo(0.1, 5); // 10% gain
    expect(returns[1]).toBeCloseTo(-0.0455, 2); // ~4.5% loss
  });

  it('should detrend data', () => {
    // Linear trend: y = 2x + noise
    const data = Array(100)
      .fill(0)
      .map((_, i) => 2 * i + Math.random() * 0.1);
    const detrended = TimeSeries.detrend(data);

    // Detrended should be near zero
    const mean = Statistics.mean(detrended);
    expect(Math.abs(mean)).toBeLessThan(1);
  });

  it('should calculate RSI', () => {
    const prices = Array(50)
      .fill(100)
      .map((_, i) => 100 + i + Math.random() * 5);
    const rsi = TimeSeries.rsi(prices, 14);

    expect(rsi.length).toBeGreaterThan(0);
    // RSI should be between 0 and 100
    for (const value of rsi) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(100);
    }
  });
});

describe('Wavelets', () => {
  it('should perform Haar transform', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8];
    const result = Wavelets.haar(data);

    expect(result.approximation.length).toBeGreaterThan(0);
    expect(result.details.length).toBeGreaterThan(0);
  });

  it('should reconstruct signal from Haar coefficients', () => {
    const original = [1, 2, 3, 4, 5, 6, 7, 8];
    const coeffs = Wavelets.haar(original);
    const reconstructed = Wavelets.inverseHaar(coeffs);

    for (let i = 0; i < original.length; i++) {
      expect(reconstructed[i]).toBeCloseTo(original[i], 5);
    }
  });

  it('should denoise a signal', () => {
    // Create signal with noise
    const clean = Array(64)
      .fill(0)
      .map((_, i) => Math.sin((2 * Math.PI * i) / 16));
    const noisy = clean.map((v) => v + (Math.random() - 0.5) * 0.5);

    const denoised = Wavelets.denoise(noisy);

    // Should return same length
    expect(denoised).toHaveLength(64);
    // Should not throw
    expect(denoised.every((v) => typeof v === 'number')).toBe(true);
  });
});
