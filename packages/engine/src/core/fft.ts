/**
 * Fast Fourier Transform Implementation
 *
 * Cooley-Tukey radix-2 FFT algorithm.
 * Optimized TypeScript implementation for signal analysis.
 *
 * © 2025 Crystalline Labs LLC
 */

export interface Complex {
  re: number;
  im: number;
}

/**
 * FFT class with optimized operations
 */
export class FFT {
  private size: number;
  private cosTable: Float64Array;
  private sinTable: Float64Array;
  private reverseTable: Uint32Array;

  constructor(size: number) {
    // Size must be power of 2
    if (size & (size - 1)) {
      size = Math.pow(2, Math.ceil(Math.log2(size)));
    }
    this.size = size;

    // Precompute twiddle factors
    this.cosTable = new Float64Array(size / 2);
    this.sinTable = new Float64Array(size / 2);

    for (let i = 0; i < size / 2; i++) {
      this.cosTable[i] = Math.cos((2 * Math.PI * i) / size);
      this.sinTable[i] = Math.sin((2 * Math.PI * i) / size);
    }

    // Precompute bit-reversal table
    this.reverseTable = new Uint32Array(size);
    const bits = Math.log2(size);

    for (let i = 0; i < size; i++) {
      let reversed = 0;
      for (let j = 0; j < bits; j++) {
        reversed = (reversed << 1) | ((i >> j) & 1);
      }
      this.reverseTable[i] = reversed;
    }
  }

  /**
   * Perform forward FFT
   * @param real Real part of input
   * @param imag Imaginary part of input (optional, defaults to zeros)
   * @returns Complex spectrum
   */
  forward(real: number[], imag?: number[]): Complex[] {
    const n = this.size;
    const re = new Float64Array(n);
    const im = new Float64Array(n);

    // Pad or truncate input
    for (let i = 0; i < n; i++) {
      re[i] = i < real.length ? real[i] : 0;
      im[i] = imag && i < imag.length ? imag[i] : 0;
    }

    // Bit-reversal permutation
    for (let i = 0; i < n; i++) {
      const j = this.reverseTable[i];
      if (j > i) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }

    // Cooley-Tukey iterative FFT
    for (let size = 2; size <= n; size *= 2) {
      const halfSize = size / 2;
      const tableStep = n / size;

      for (let i = 0; i < n; i += size) {
        for (let j = 0; j < halfSize; j++) {
          const k = j * tableStep;
          const cos = this.cosTable[k];
          const sin = this.sinTable[k];

          const idx1 = i + j;
          const idx2 = i + j + halfSize;

          const tRe = re[idx2] * cos + im[idx2] * sin;
          const tIm = im[idx2] * cos - re[idx2] * sin;

          re[idx2] = re[idx1] - tRe;
          im[idx2] = im[idx1] - tIm;
          re[idx1] += tRe;
          im[idx1] += tIm;
        }
      }
    }

    // Convert to Complex array
    const result: Complex[] = [];
    for (let i = 0; i < n; i++) {
      result.push({ re: re[i], im: im[i] });
    }

    return result;
  }

  /**
   * Perform inverse FFT
   */
  inverse(spectrum: Complex[]): number[] {
    const n = this.size;

    // Conjugate the spectrum
    const re = spectrum.map((c) => c.re);
    const im = spectrum.map((c) => -c.im);

    // Forward FFT on conjugated input
    const result = this.forward(re, im);

    // Conjugate and scale
    return result.map((c) => c.re / n);
  }

  /**
   * Get magnitude spectrum
   */
  magnitude(spectrum: Complex[]): number[] {
    return spectrum.map((c) => Math.sqrt(c.re * c.re + c.im * c.im));
  }

  /**
   * Get phase spectrum
   */
  phase(spectrum: Complex[]): number[] {
    return spectrum.map((c) => Math.atan2(c.im, c.re));
  }

  /**
   * Get power spectrum (magnitude squared)
   */
  power(spectrum: Complex[]): number[] {
    return spectrum.map((c) => c.re * c.re + c.im * c.im);
  }

  /**
   * Find dominant frequencies
   */
  dominantFrequencies(
    spectrum: Complex[],
    sampleRate: number,
    count = 5
  ): Array<{ frequency: number; magnitude: number }> {
    const mag = this.magnitude(spectrum);
    const n = spectrum.length;

    // Only look at positive frequencies (first half)
    const results: Array<{ frequency: number; magnitude: number; index: number }> = [];

    for (let i = 1; i < n / 2; i++) {
      results.push({
        frequency: (i * sampleRate) / n,
        magnitude: mag[i],
        index: i,
      });
    }

    // Sort by magnitude and return top frequencies
    return results
      .sort((a, b) => b.magnitude - a.magnitude)
      .slice(0, count)
      .map(({ frequency, magnitude }) => ({ frequency, magnitude }));
  }
}

/**
 * Simple FFT function for one-off transforms
 */
export function fft(signal: number[]): Complex[] {
  const fftObj = new FFT(signal.length);
  return fftObj.forward(signal);
}

/**
 * Simple inverse FFT function
 */
export function ifft(spectrum: Complex[]): number[] {
  const fftObj = new FFT(spectrum.length);
  return fftObj.inverse(spectrum);
}

/**
 * Compute power spectral density
 */
export function psd(signal: number[], windowSize = 256): number[] {
  const fftObj = new FFT(windowSize);
  const numSegments = Math.floor(signal.length / windowSize);
  const avgPower = new Float64Array(windowSize);

  for (let seg = 0; seg < numSegments; seg++) {
    const segment = signal.slice(seg * windowSize, (seg + 1) * windowSize);

    // Apply Hann window
    const windowed = segment.map(
      (v, i) => v * 0.5 * (1 - Math.cos((2 * Math.PI * i) / windowSize))
    );

    const spectrum = fftObj.forward(windowed);
    const power = fftObj.power(spectrum);

    for (let i = 0; i < windowSize; i++) {
      avgPower[i] += power[i];
    }
  }

  // Average and return
  return Array.from(avgPower).map((p) => p / (numSegments || 1));
}

/**
 * Cross-spectral density between two signals
 */
export function csd(signal1: number[], signal2: number[], windowSize = 256): Complex[] {
  const fftObj = new FFT(windowSize);
  const numSegments = Math.min(
    Math.floor(signal1.length / windowSize),
    Math.floor(signal2.length / windowSize)
  );

  const avgCsd: Complex[] = Array(windowSize)
    .fill(null)
    .map(() => ({ re: 0, im: 0 }));

  for (let seg = 0; seg < numSegments; seg++) {
    const seg1 = signal1.slice(seg * windowSize, (seg + 1) * windowSize);
    const seg2 = signal2.slice(seg * windowSize, (seg + 1) * windowSize);

    // Apply Hann window
    const win1 = seg1.map((v, i) => v * 0.5 * (1 - Math.cos((2 * Math.PI * i) / windowSize)));
    const win2 = seg2.map((v, i) => v * 0.5 * (1 - Math.cos((2 * Math.PI * i) / windowSize)));

    const spec1 = fftObj.forward(win1);
    const spec2 = fftObj.forward(win2);

    // Cross-spectrum: X1 * conj(X2)
    for (let i = 0; i < windowSize; i++) {
      avgCsd[i].re += spec1[i].re * spec2[i].re + spec1[i].im * spec2[i].im;
      avgCsd[i].im += spec1[i].im * spec2[i].re - spec1[i].re * spec2[i].im;
    }
  }

  // Average
  for (let i = 0; i < windowSize; i++) {
    avgCsd[i].re /= numSegments || 1;
    avgCsd[i].im /= numSegments || 1;
  }

  return avgCsd;
}

/**
 * Coherence between two signals (0-1)
 */
export function coherence(signal1: number[], signal2: number[], windowSize = 256): number[] {
  const psd1 = psd(signal1, windowSize);
  const psd2 = psd(signal2, windowSize);
  const crossSpec = csd(signal1, signal2, windowSize);

  return crossSpec.map((c, i) => {
    const crossPower = c.re * c.re + c.im * c.im;
    const denom = psd1[i] * psd2[i];
    return denom > 0 ? crossPower / denom : 0;
  });
}
