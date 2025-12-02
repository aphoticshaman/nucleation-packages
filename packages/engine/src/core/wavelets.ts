/**
 * Wavelet Transform Implementation
 *
 * Multi-resolution analysis using wavelets.
 * Haar and Daubechies wavelets for signal decomposition.
 *
 * © 2025 Crystalline Labs LLC
 */

export interface WaveletCoefficients {
  approximation: number[];
  details: number[][];
  levels: number;
}

/**
 * Wavelet analysis class
 */
export class Wavelets {
  /**
   * Haar wavelet transform
   */
  static haar(data: number[], levels?: number): WaveletCoefficients {
    const maxLevels = Math.floor(Math.log2(data.length));
    levels = Math.min(levels ?? maxLevels, maxLevels);

    let current = [...data];
    const details: number[][] = [];

    for (let l = 0; l < levels; l++) {
      const n = current.length;
      const approx: number[] = [];
      const detail: number[] = [];

      for (let i = 0; i < n - 1; i += 2) {
        approx.push((current[i] + current[i + 1]) / Math.SQRT2);
        detail.push((current[i] - current[i + 1]) / Math.SQRT2);
      }

      // Handle odd length
      if (n % 2 === 1) {
        approx.push(current[n - 1] / Math.SQRT2);
      }

      details.push(detail);
      current = approx;
    }

    return {
      approximation: current,
      details,
      levels,
    };
  }

  /**
   * Inverse Haar wavelet transform
   */
  static inverseHaar(coefficients: WaveletCoefficients): number[] {
    let current = [...coefficients.approximation];

    for (let l = coefficients.levels - 1; l >= 0; l--) {
      const detail = coefficients.details[l];
      const reconstructed: number[] = [];

      for (let i = 0; i < detail.length; i++) {
        reconstructed.push((current[i] + detail[i]) / Math.SQRT2);
        reconstructed.push((current[i] - detail[i]) / Math.SQRT2);
      }

      // Handle odd length
      if (current.length > detail.length) {
        reconstructed.push(current[current.length - 1] * Math.SQRT2);
      }

      current = reconstructed;
    }

    return current;
  }

  /**
   * Daubechies-4 wavelet transform
   */
  static db4(data: number[], levels?: number): WaveletCoefficients {
    // DB4 filter coefficients
    const h0 = (1 + Math.sqrt(3)) / (4 * Math.SQRT2);
    const h1 = (3 + Math.sqrt(3)) / (4 * Math.SQRT2);
    const h2 = (3 - Math.sqrt(3)) / (4 * Math.SQRT2);
    const h3 = (1 - Math.sqrt(3)) / (4 * Math.SQRT2);

    const lowPass = [h0, h1, h2, h3];
    const highPass = [h3, -h2, h1, -h0];

    const maxLevels = Math.floor(Math.log2(data.length / 4));
    levels = Math.min(levels ?? maxLevels, maxLevels);

    let current = [...data];
    const details: number[][] = [];

    for (let l = 0; l < levels; l++) {
      const n = current.length;
      const approx: number[] = [];
      const detail: number[] = [];

      for (let i = 0; i < n; i += 2) {
        let low = 0,
          high = 0;
        for (let j = 0; j < 4; j++) {
          const idx = (i + j) % n;
          low += lowPass[j] * current[idx];
          high += highPass[j] * current[idx];
        }
        approx.push(low);
        detail.push(high);
      }

      details.push(detail);
      current = approx;
    }

    return {
      approximation: current,
      details,
      levels,
    };
  }

  /**
   * Wavelet denoising using soft thresholding
   */
  static denoise(data: number[], threshold?: number, wavelet: 'haar' | 'db4' = 'haar'): number[] {
    // Transform
    const coeffs = wavelet === 'haar' ? Wavelets.haar(data) : Wavelets.db4(data);

    // Estimate threshold if not provided (universal threshold)
    if (threshold === undefined) {
      const allDetails = coeffs.details.flat();
      const median = Wavelets.median(allDetails.map(Math.abs));
      const sigma = median / 0.6745; // Robust noise estimate
      threshold = sigma * Math.sqrt(2 * Math.log(data.length));
    }

    // Soft thresholding on detail coefficients
    for (let l = 0; l < coeffs.levels; l++) {
      coeffs.details[l] = coeffs.details[l].map((d) => {
        if (Math.abs(d) <= threshold!) return 0;
        return d > 0 ? d - threshold! : d + threshold!;
      });
    }

    // Inverse transform
    return wavelet === 'haar' ? Wavelets.inverseHaar(coeffs) : Wavelets.inverseDb4(coeffs);
  }

  /**
   * Inverse DB4 transform
   */
  private static inverseDb4(coefficients: WaveletCoefficients): number[] {
    // Reconstruction filters (time-reversed synthesis filters)
    const h0 = (1 + Math.sqrt(3)) / (4 * Math.SQRT2);
    const h1 = (3 + Math.sqrt(3)) / (4 * Math.SQRT2);
    const h2 = (3 - Math.sqrt(3)) / (4 * Math.SQRT2);
    const h3 = (1 - Math.sqrt(3)) / (4 * Math.SQRT2);

    const lowRecon = [h2, h1, h0, h3];
    const highRecon = [-h0, h1, -h2, h3];

    let current = [...coefficients.approximation];

    for (let l = coefficients.levels - 1; l >= 0; l--) {
      const detail = coefficients.details[l];
      const n = current.length;
      const reconstructed = new Array(n * 2).fill(0);

      // Upsampling and filtering
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < 4; j++) {
          const idx = (2 * i + j) % (n * 2);
          reconstructed[idx] += lowRecon[j] * current[i] + highRecon[j] * detail[i];
        }
      }

      current = reconstructed;
    }

    return current;
  }

  /**
   * Multi-resolution energy analysis
   */
  static energyByLevel(coefficients: WaveletCoefficients): number[] {
    const energies: number[] = [];

    // Approximation energy
    energies.push(coefficients.approximation.reduce((sum, c) => sum + c * c, 0));

    // Detail energies at each level
    for (const detail of coefficients.details) {
      energies.push(detail.reduce((sum, c) => sum + c * c, 0));
    }

    return energies;
  }

  /**
   * Detect dominant scales
   */
  static dominantScales(
    data: number[],
    topN = 3
  ): Array<{ level: number; energy: number; period: number }> {
    const coeffs = Wavelets.haar(data);
    const energies = Wavelets.energyByLevel(coeffs);
    const totalEnergy = energies.reduce((a, b) => a + b, 0);

    const results = energies.map((energy, level) => ({
      level,
      energy: totalEnergy > 0 ? energy / totalEnergy : 0,
      period: Math.pow(2, level), // Approximate period at this scale
    }));

    return results.sort((a, b) => b.energy - a.energy).slice(0, topN);
  }

  /**
   * Wavelet coherence between two signals
   */
  static coherence(x: number[], y: number[]): number[][] {
    const minLen = Math.min(x.length, y.length);
    const coeffsX = Wavelets.haar(x.slice(0, minLen));
    const coeffsY = Wavelets.haar(y.slice(0, minLen));

    const coherence: number[][] = [];

    for (let l = 0; l < coeffsX.levels; l++) {
      const detailX = coeffsX.details[l];
      const detailY = coeffsY.details[l];
      const levelCoherence: number[] = [];

      for (let i = 0; i < Math.min(detailX.length, detailY.length); i++) {
        const cross = detailX[i] * detailY[i];
        const autoX = detailX[i] * detailX[i];
        const autoY = detailY[i] * detailY[i];
        const denom = Math.sqrt(autoX * autoY);
        levelCoherence.push(denom > 0 ? cross / denom : 0);
      }

      coherence.push(levelCoherence);
    }

    return coherence;
  }

  private static median(data: number[]): number {
    const sorted = [...data].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }
}

// Convenience export
export const waveletTransform = Wavelets.haar;
