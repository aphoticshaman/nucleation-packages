/**
 * Statistical Utilities
 *
 * Core statistical functions for signal analysis.
 * Optimized for financial/social time series.
 *
 * © 2025 Crystalline Labs LLC
 */

/**
 * Statistics class with comprehensive analysis methods
 */
export class Statistics {
  /**
   * Calculate mean
   */
  static mean(data: number[]): number {
    if (data.length === 0) return 0;
    return data.reduce((a, b) => a + b, 0) / data.length;
  }

  /**
   * Calculate variance
   */
  static variance(data: number[], sample = true): number {
    if (data.length < 2) return 0;
    const mean = Statistics.mean(data);
    const sumSq = data.reduce((sum, x) => sum + (x - mean) ** 2, 0);
    return sumSq / (sample ? data.length - 1 : data.length);
  }

  /**
   * Calculate standard deviation
   */
  static std(data: number[], sample = true): number {
    return Math.sqrt(Statistics.variance(data, sample));
  }

  /**
   * Calculate median
   */
  static median(data: number[]): number {
    if (data.length === 0) return 0;
    const sorted = [...data].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  /**
   * Calculate percentile
   */
  static percentile(data: number[], p: number): number {
    if (data.length === 0) return 0;
    const sorted = [...data].sort((a, b) => a - b);
    const idx = Math.ceil((p / 100) * sorted.length) - 1;
    return sorted[Math.max(0, Math.min(idx, sorted.length - 1))];
  }

  /**
   * Calculate skewness
   */
  static skewness(data: number[]): number {
    if (data.length < 3) return 0;
    const mean = Statistics.mean(data);
    const std = Statistics.std(data);
    if (std === 0) return 0;

    const n = data.length;
    const m3 = data.reduce((sum, x) => sum + ((x - mean) / std) ** 3, 0) / n;
    return (m3 * Math.sqrt(n * (n - 1))) / (n - 2);
  }

  /**
   * Calculate kurtosis (excess)
   */
  static kurtosis(data: number[]): number {
    if (data.length < 4) return 0;
    const mean = Statistics.mean(data);
    const std = Statistics.std(data);
    if (std === 0) return 0;

    const n = data.length;
    const m4 = data.reduce((sum, x) => sum + ((x - mean) / std) ** 4, 0) / n;
    return m4 - 3; // Excess kurtosis
  }

  /**
   * Calculate Pearson correlation
   */
  static correlation(x: number[], y: number[]): number {
    const n = Math.min(x.length, y.length);
    if (n < 2) return 0;

    const meanX = Statistics.mean(x.slice(0, n));
    const meanY = Statistics.mean(y.slice(0, n));

    let num = 0,
      denX = 0,
      denY = 0;
    for (let i = 0; i < n; i++) {
      const dx = x[i] - meanX;
      const dy = y[i] - meanY;
      num += dx * dy;
      denX += dx * dx;
      denY += dy * dy;
    }

    const den = Math.sqrt(denX * denY);
    return den === 0 ? 0 : num / den;
  }

  /**
   * Calculate Spearman rank correlation
   */
  static spearman(x: number[], y: number[]): number {
    const n = Math.min(x.length, y.length);
    if (n < 2) return 0;

    // Convert to ranks
    const rankX = Statistics.toRanks(x.slice(0, n));
    const rankY = Statistics.toRanks(y.slice(0, n));

    return Statistics.correlation(rankX, rankY);
  }

  /**
   * Convert values to ranks
   */
  static toRanks(data: number[]): number[] {
    const indexed = data.map((v, i) => ({ v, i }));
    indexed.sort((a, b) => a.v - b.v);

    const ranks = new Array(data.length);
    for (let i = 0; i < indexed.length; i++) {
      ranks[indexed[i].i] = i + 1;
    }

    return ranks;
  }

  /**
   * Calculate covariance
   */
  static covariance(x: number[], y: number[]): number {
    const n = Math.min(x.length, y.length);
    if (n < 2) return 0;

    const meanX = Statistics.mean(x.slice(0, n));
    const meanY = Statistics.mean(y.slice(0, n));

    let cov = 0;
    for (let i = 0; i < n; i++) {
      cov += (x[i] - meanX) * (y[i] - meanY);
    }

    return cov / (n - 1);
  }

  /**
   * Calculate z-scores
   */
  static zscore(data: number[]): number[] {
    const mean = Statistics.mean(data);
    const std = Statistics.std(data);
    if (std === 0) return data.map(() => 0);
    return data.map((x) => (x - mean) / std);
  }

  /**
   * Calculate rolling statistics
   */
  static rolling(data: number[], window: number, fn: (chunk: number[]) => number): number[] {
    const result: number[] = [];
    for (let i = window - 1; i < data.length; i++) {
      const chunk = data.slice(i - window + 1, i + 1);
      result.push(fn(chunk));
    }
    return result;
  }

  /**
   * Calculate autocorrelation at given lag
   */
  static autocorrelation(data: number[], lag: number): number {
    if (lag >= data.length) return 0;
    return Statistics.correlation(data.slice(0, -lag), data.slice(lag));
  }

  /**
   * Calculate all autocorrelations up to maxLag
   */
  static acf(data: number[], maxLag: number): number[] {
    const result: number[] = [1]; // lag 0 is always 1
    for (let lag = 1; lag <= maxLag; lag++) {
      result.push(Statistics.autocorrelation(data, lag));
    }
    return result;
  }

  /**
   * Partial autocorrelation function
   */
  static pacf(data: number[], maxLag: number): number[] {
    const acfValues = Statistics.acf(data, maxLag);
    const pacfValues: number[] = [1];

    for (let k = 1; k <= maxLag; k++) {
      // Levinson-Durbin recursion
      let phi = acfValues[k];
      for (let j = 1; j < k; j++) {
        phi -= pacfValues[j] * acfValues[k - j];
      }

      let denom = 1;
      for (let j = 1; j < k; j++) {
        denom -= pacfValues[j] * acfValues[j];
      }

      pacfValues.push(denom !== 0 ? phi / denom : 0);
    }

    return pacfValues;
  }

  /**
   * Augmented Dickey-Fuller test statistic
   * Tests for stationarity (negative = stationary)
   */
  static adfStatistic(data: number[], maxLag = 1): number {
    if (data.length < maxLag + 3) return 0;

    // First difference
    const diff = data.slice(1).map((v, i) => v - data[i]);

    // Lag level
    const lagLevel = data.slice(0, -1);

    // Regression: diff = alpha + beta * lagLevel + error
    // ADF stat = beta / std_error(beta)

    const n = diff.length;
    const meanX = Statistics.mean(lagLevel.slice(0, n));
    const meanY = Statistics.mean(diff);

    let ssXY = 0,
      ssXX = 0;
    for (let i = 0; i < n; i++) {
      ssXY += (lagLevel[i] - meanX) * (diff[i] - meanY);
      ssXX += (lagLevel[i] - meanX) ** 2;
    }

    if (ssXX === 0) return 0;

    const beta = ssXY / ssXX;
    const alpha = meanY - beta * meanX;

    // Calculate residuals and standard error
    let ssE = 0;
    for (let i = 0; i < n; i++) {
      const predicted = alpha + beta * lagLevel[i];
      ssE += (diff[i] - predicted) ** 2;
    }

    const mse = ssE / (n - 2);
    const seBeta = Math.sqrt(mse / ssXX);

    return seBeta !== 0 ? beta / seBeta : 0;
  }

  /**
   * Ljung-Box test for autocorrelation
   * Returns Q statistic (high = significant autocorrelation)
   */
  static ljungBox(data: number[], maxLag = 10): number {
    const n = data.length;
    const acfValues = Statistics.acf(data, maxLag);

    let q = 0;
    for (let k = 1; k <= maxLag; k++) {
      q += acfValues[k] ** 2 / (n - k);
    }

    return n * (n + 2) * q;
  }

  /**
   * Information ratio
   */
  static informationRatio(returns: number[], benchmark: number[]): number {
    const excess = returns.map((r, i) => r - (benchmark[i] ?? 0));
    const mean = Statistics.mean(excess);
    const std = Statistics.std(excess);
    return std !== 0 ? mean / std : 0;
  }

  /**
   * Sharpe ratio
   */
  static sharpeRatio(returns: number[], riskFreeRate = 0): number {
    const excessReturns = returns.map((r) => r - riskFreeRate);
    const mean = Statistics.mean(excessReturns);
    const std = Statistics.std(excessReturns);
    return std !== 0 ? mean / std : 0;
  }

  /**
   * Maximum drawdown
   */
  static maxDrawdown(values: number[]): number {
    let maxValue = values[0] ?? 0;
    let maxDD = 0;

    for (const value of values) {
      maxValue = Math.max(maxValue, value);
      const dd = (maxValue - value) / maxValue;
      maxDD = Math.max(maxDD, dd);
    }

    return maxDD;
  }

  /**
   * Value at Risk (parametric)
   */
  static valueAtRisk(returns: number[], confidence = 0.95): number {
    const mean = Statistics.mean(returns);
    const std = Statistics.std(returns);

    // Normal distribution quantile (approximate)
    const zScores: Record<number, number> = {
      0.9: 1.282,
      0.95: 1.645,
      0.99: 2.326,
    };

    const z = zScores[confidence] ?? 1.645;
    return -(mean - z * std);
  }

  /**
   * Conditional VaR (Expected Shortfall)
   */
  static expectedShortfall(returns: number[], confidence = 0.95): number {
    const sorted = [...returns].sort((a, b) => a - b);
    const cutoff = Math.floor(sorted.length * (1 - confidence));
    const tail = sorted.slice(0, cutoff);
    return tail.length > 0 ? -Statistics.mean(tail) : 0;
  }
}

// Convenience exports
export const correlation = Statistics.correlation;
export const variance = Statistics.variance;
export const zscore = Statistics.zscore;
