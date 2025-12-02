/**
 * Time Series Analysis Utilities
 *
 * Core time series operations for signal processing.
 *
 * © 2025 Crystalline Labs LLC
 */

import { Statistics } from './statistics';

/**
 * Time Series analysis class
 */
export class TimeSeries {
  /**
   * Simple moving average
   */
  static sma(data: number[], window: number): number[] {
    return Statistics.rolling(data, window, Statistics.mean);
  }

  /**
   * Exponential moving average
   */
  static ema(data: number[], alpha: number): number[] {
    if (data.length === 0) return [];

    const result: number[] = [data[0]];
    for (let i = 1; i < data.length; i++) {
      result.push(alpha * data[i] + (1 - alpha) * result[i - 1]);
    }
    return result;
  }

  /**
   * Double exponential smoothing (Holt)
   */
  static doubleExponential(
    data: number[],
    alpha: number,
    beta: number
  ): { level: number[]; trend: number[]; forecast: number[] } {
    if (data.length < 2) {
      return { level: data, trend: [0], forecast: data };
    }

    const level: number[] = [data[0]];
    const trend: number[] = [data[1] - data[0]];
    const forecast: number[] = [data[0]];

    for (let i = 1; i < data.length; i++) {
      level.push(alpha * data[i] + (1 - alpha) * (level[i - 1] + trend[i - 1]));
      trend.push(beta * (level[i] - level[i - 1]) + (1 - beta) * trend[i - 1]);
      forecast.push(level[i] + trend[i]);
    }

    return { level, trend, forecast };
  }

  /**
   * First difference
   */
  static diff(data: number[], periods = 1): number[] {
    if (data.length <= periods) return [];
    return data.slice(periods).map((v, i) => v - data[i]);
  }

  /**
   * Log returns
   */
  static logReturns(data: number[]): number[] {
    if (data.length < 2) return [];
    return data.slice(1).map((v, i) => Math.log(v / data[i]));
  }

  /**
   * Percentage returns
   */
  static pctReturns(data: number[]): number[] {
    if (data.length < 2) return [];
    return data.slice(1).map((v, i) => (v - data[i]) / data[i]);
  }

  /**
   * Cumulative sum
   */
  static cumsum(data: number[]): number[] {
    const result: number[] = [];
    let sum = 0;
    for (const v of data) {
      sum += v;
      result.push(sum);
    }
    return result;
  }

  /**
   * Cumulative product
   */
  static cumprod(data: number[]): number[] {
    const result: number[] = [];
    let prod = 1;
    for (const v of data) {
      prod *= v;
      result.push(prod);
    }
    return result;
  }

  /**
   * Detrend (remove linear trend)
   */
  static detrend(data: number[]): number[] {
    if (data.length < 2) return data;

    const n = data.length;
    let sumX = 0,
      sumY = 0,
      sumXY = 0,
      sumX2 = 0;

    for (let i = 0; i < n; i++) {
      sumX += i;
      sumY += data[i];
      sumXY += i * data[i];
      sumX2 += i * i;
    }

    const denom = n * sumX2 - sumX * sumX;
    if (Math.abs(denom) < 1e-10) return data;

    const slope = (n * sumXY - sumX * sumY) / denom;
    const intercept = (sumY - slope * sumX) / n;

    return data.map((v, i) => v - (intercept + slope * i));
  }

  /**
   * Seasonal decomposition (additive)
   */
  static decompose(
    data: number[],
    period: number
  ): { trend: number[]; seasonal: number[]; residual: number[] } {
    // Moving average for trend
    const trend = TimeSeries.sma(data, period);

    // Pad trend to match data length
    const paddedTrend = Array(Math.floor(period / 2))
      .fill(trend[0])
      .concat(trend)
      .concat(Array(Math.ceil(period / 2) - 1).fill(trend[trend.length - 1]));

    // Detrended series
    const detrended = data.map((v, i) => v - paddedTrend[i]);

    // Average seasonal component
    const seasonal: number[] = [];
    for (let i = 0; i < period; i++) {
      const seasonValues: number[] = [];
      for (let j = i; j < detrended.length; j += period) {
        seasonValues.push(detrended[j]);
      }
      seasonal.push(Statistics.mean(seasonValues));
    }

    // Expand seasonal to full length
    const fullSeasonal = data.map((_, i) => seasonal[i % period]);

    // Residual
    const residual = data.map((v, i) => v - paddedTrend[i] - fullSeasonal[i]);

    return { trend: paddedTrend, seasonal: fullSeasonal, residual };
  }

  /**
   * Bollinger Bands
   */
  static bollingerBands(
    data: number[],
    window = 20,
    numStd = 2
  ): { middle: number[]; upper: number[]; lower: number[] } {
    const middle = TimeSeries.sma(data, window);
    const std = Statistics.rolling(data, window, (chunk) => Statistics.std(chunk));

    const upper = middle.map((m, i) => m + numStd * std[i]);
    const lower = middle.map((m, i) => m - numStd * std[i]);

    return { middle, upper, lower };
  }

  /**
   * RSI (Relative Strength Index)
   */
  static rsi(data: number[], period = 14): number[] {
    const changes = TimeSeries.diff(data);
    const gains = changes.map((c) => (c > 0 ? c : 0));
    const losses = changes.map((c) => (c < 0 ? -c : 0));

    const avgGain = TimeSeries.ema(gains, 1 / period);
    const avgLoss = TimeSeries.ema(losses, 1 / period);

    return avgGain.map((g, i) => {
      const l = avgLoss[i];
      if (l === 0) return 100;
      return 100 - 100 / (1 + g / l);
    });
  }

  /**
   * MACD (Moving Average Convergence Divergence)
   */
  static macd(
    data: number[],
    fastPeriod = 12,
    slowPeriod = 26,
    signalPeriod = 9
  ): { macd: number[]; signal: number[]; histogram: number[] } {
    const fastEma = TimeSeries.ema(data, 2 / (fastPeriod + 1));
    const slowEma = TimeSeries.ema(data, 2 / (slowPeriod + 1));

    const macdLine = fastEma.map((f, i) => f - slowEma[i]);
    const signalLine = TimeSeries.ema(macdLine, 2 / (signalPeriod + 1));
    const histogram = macdLine.map((m, i) => m - signalLine[i]);

    return { macd: macdLine, signal: signalLine, histogram };
  }

  /**
   * Normalize to [0, 1] range
   */
  static normalize(data: number[]): number[] {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    return data.map((v) => (v - min) / range);
  }

  /**
   * Standardize (z-score)
   */
  static standardize(data: number[]): number[] {
    return Statistics.zscore(data);
  }

  /**
   * Interpolate missing values
   */
  static interpolate(data: (number | null)[]): number[] {
    const result = [...data] as number[];

    for (let i = 0; i < result.length; i++) {
      if (result[i] === null || isNaN(result[i])) {
        // Find nearest valid values
        let left = i - 1;
        let right = i + 1;

        while (left >= 0 && (result[left] === null || isNaN(result[left]))) left--;
        while (right < result.length && (result[right] === null || isNaN(result[right]))) right++;

        if (left >= 0 && right < result.length) {
          // Linear interpolation
          const span = right - left;
          const step = (result[right] - result[left]) / span;
          result[i] = result[left] + step * (i - left);
        } else if (left >= 0) {
          result[i] = result[left];
        } else if (right < result.length) {
          result[i] = result[right];
        } else {
          result[i] = 0;
        }
      }
    }

    return result;
  }

  /**
   * Resample to different frequency
   */
  static resample(
    data: number[],
    sourceFreq: number,
    targetFreq: number,
    method: 'mean' | 'sum' | 'first' | 'last' = 'mean'
  ): number[] {
    if (targetFreq >= sourceFreq) {
      // Upsampling - simple interpolation
      const ratio = targetFreq / sourceFreq;
      const result: number[] = [];

      for (let i = 0; i < data.length - 1; i++) {
        for (let j = 0; j < ratio; j++) {
          const t = j / ratio;
          result.push(data[i] * (1 - t) + data[i + 1] * t);
        }
      }
      result.push(data[data.length - 1]);

      return result;
    } else {
      // Downsampling - aggregate
      const ratio = Math.round(sourceFreq / targetFreq);
      const result: number[] = [];

      for (let i = 0; i < data.length; i += ratio) {
        const chunk = data.slice(i, i + ratio);
        switch (method) {
          case 'mean':
            result.push(Statistics.mean(chunk));
            break;
          case 'sum':
            result.push(chunk.reduce((a, b) => a + b, 0));
            break;
          case 'first':
            result.push(chunk[0]);
            break;
          case 'last':
            result.push(chunk[chunk.length - 1]);
            break;
        }
      }

      return result;
    }
  }

  /**
   * Detect outliers using IQR method
   */
  static detectOutliers(data: number[], threshold = 1.5): { indices: number[]; values: number[] } {
    const q1 = Statistics.percentile(data, 25);
    const q3 = Statistics.percentile(data, 75);
    const iqr = q3 - q1;

    const lower = q1 - threshold * iqr;
    const upper = q3 + threshold * iqr;

    const indices: number[] = [];
    const values: number[] = [];

    for (let i = 0; i < data.length; i++) {
      if (data[i] < lower || data[i] > upper) {
        indices.push(i);
        values.push(data[i]);
      }
    }

    return { indices, values };
  }

  /**
   * Granger causality test (simplified)
   * Returns F-statistic (higher = more likely X causes Y)
   */
  static grangerCausality(x: number[], y: number[], lag = 1): number {
    const n = Math.min(x.length, y.length) - lag;
    if (n < lag + 3) return 0;

    // Restricted model: Y ~ Y_lag
    const yLag = y.slice(0, n);
    const yActual = y.slice(lag, lag + n);

    // Unrestricted model: Y ~ Y_lag + X_lag
    const xLag = x.slice(0, n);

    // Calculate SSR for both models (simplified regression)
    const ssrRestricted = this.calculateSSR(yActual, yLag);
    const ssrUnrestricted = this.calculateSSRMultiple(yActual, yLag, xLag);

    // F-statistic
    const numRestrictions = 1;
    const df = n - 2 * lag - 1;

    if (df <= 0 || ssrUnrestricted === 0) return 0;

    return (ssrRestricted - ssrUnrestricted) / numRestrictions / (ssrUnrestricted / df);
  }

  private static calculateSSR(y: number[], x: number[]): number {
    const n = y.length;
    const meanX = Statistics.mean(x);
    const meanY = Statistics.mean(y);

    let ssXY = 0,
      ssXX = 0;
    for (let i = 0; i < n; i++) {
      ssXY += (x[i] - meanX) * (y[i] - meanY);
      ssXX += (x[i] - meanX) ** 2;
    }

    const beta = ssXX !== 0 ? ssXY / ssXX : 0;
    const alpha = meanY - beta * meanX;

    let ssr = 0;
    for (let i = 0; i < n; i++) {
      const predicted = alpha + beta * x[i];
      ssr += (y[i] - predicted) ** 2;
    }

    return ssr;
  }

  private static calculateSSRMultiple(y: number[], x1: number[], x2: number[]): number {
    // Simplified multiple regression (normal equations would be better)
    const n = y.length;
    const meanY = Statistics.mean(y);
    const meanX1 = Statistics.mean(x1);
    const meanX2 = Statistics.mean(x2);

    // Use correlation-based coefficients (simplified)
    const r1 = Statistics.correlation(x1, y);
    const r2 = Statistics.correlation(x2, y);
    const r12 = Statistics.correlation(x1, x2);

    const stdY = Statistics.std(y);
    const stdX1 = Statistics.std(x1);
    const stdX2 = Statistics.std(x2);

    const denom = 1 - r12 * r12;
    if (denom === 0) return this.calculateSSR(y, x1);

    const beta1 = ((r1 - r2 * r12) / denom) * (stdY / stdX1);
    const beta2 = ((r2 - r1 * r12) / denom) * (stdY / stdX2);
    const alpha = meanY - beta1 * meanX1 - beta2 * meanX2;

    let ssr = 0;
    for (let i = 0; i < n; i++) {
      const predicted = alpha + beta1 * x1[i] + beta2 * x2[i];
      ssr += (y[i] - predicted) ** 2;
    }

    return ssr;
  }
}

// Convenience exports
export const movingAverage = TimeSeries.sma;
export const exponentialSmooth = TimeSeries.ema;
export const differentiate = TimeSeries.diff;
