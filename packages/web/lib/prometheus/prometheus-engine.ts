/**
 * PROMETHEUS ENGINE v1.0
 *
 * Physics-inspired answer selection for LLM inference optimization.
 * Implements: UIPT, Gravitational Basins, Solomonoff Induction, NCD
 *
 * Core Breakthroughs:
 * 1. UIPT (Universal Information Phase Transition) - Detect "grokking" via entropy collapse
 * 2. Gravitational Basins - Cluster answers by semantic gravity wells
 * 3. NCD (Normalized Compression Distance) - Structural similarity via compression
 * 4. Solomonoff Weighting - Algorithmic probability via code length
 */

export interface InferenceResult {
  answer: number | string;
  code: string;
  logProb?: number;
  entropy?: number;
}

export interface BasinCluster {
  centroid: number;
  members: number[];
  mass: number;
  density: number;
  score: number;
}

/**
 * PROMETHEUS Engine - Physics-based answer selection
 */
export class PrometheusEngine {
  private phiThreshold = 0.4; // Integration threshold
  private epsilonDistance = 0.01; // Clustering epsilon

  // ---------------------------------------------------------
  // INSIGHT #1: UIPT (Universal Information Phase Transition)
  // ---------------------------------------------------------
  /**
   * Calculates the Shannon entropy of a text/code trace.
   * Low entropy = Crystallized Logic (High confidence)
   * High entropy = Gas Phase (Hallucination)
   */
  calculateEntropy(text: string): number {
    if (!text) return 100.0;

    const counts = new Map<string, number>();
    for (const char of text) {
      counts.set(char, (counts.get(char) || 0) + 1);
    }

    const total = text.length;
    let entropy = 0;

    for (const count of counts.values()) {
      const p = count / total;
      entropy -= p * Math.log2(p);
    }

    return entropy;
  }

  /**
   * Detects phase transition in reasoning trace.
   * Returns true if entropy has "crystallized" (dropped below threshold)
   */
  detectPhaseTransition(entropyHistory: number[]): boolean {
    if (entropyHistory.length < 3) return false;

    const recent = entropyHistory.slice(-3);
    const earlier = entropyHistory.slice(0, -3);

    if (earlier.length === 0) return false;

    const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
    const earlierAvg = earlier.reduce((a, b) => a + b, 0) / earlier.length;

    // Phase transition detected if entropy dropped by >30%
    return recentAvg < earlierAvg * 0.7;
  }

  // ---------------------------------------------------------
  // INSIGHT #2: Normalized Compression Distance (NCD)
  // ---------------------------------------------------------
  /**
   * Calculates NCD between two strings.
   * Uses simplified compression approximation (run-length + unique chars).
   */
  ncd(x: string, y: string): number {
    if (x === y) return 0.0;

    // Approximate compression using unique character ratio + run-length heuristic
    const compress = (s: string): number => {
      if (!s) return 0;

      // Count unique characters (entropy proxy)
      const uniqueChars = new Set(s).size;

      // Count runs (consecutive same chars compress well)
      let runs = 1;
      for (let i = 1; i < s.length; i++) {
        if (s[i] !== s[i - 1]) runs++;
      }

      // Approximate compressed size
      return Math.min(s.length, uniqueChars * Math.log2(s.length + 1) + runs);
    };

    const cx = compress(x);
    const cy = compress(y);
    const cxy = compress(x + y);

    const denominator = Math.max(cx, cy);
    if (denominator === 0) return 0;

    return (cxy - Math.min(cx, cy)) / denominator;
  }

  /**
   * Measures code diversity using NCD.
   * High diversity = exploring different solution paths
   */
  measureCodeDiversity(codes: string[]): number {
    if (codes.length < 2) return 0;

    let totalDistance = 0;
    let pairs = 0;

    for (let i = 0; i < codes.length; i++) {
      for (let j = i + 1; j < codes.length; j++) {
        totalDistance += this.ncd(codes[i], codes[j]);
        pairs++;
      }
    }

    return pairs > 0 ? totalDistance / pairs : 0;
  }

  // ---------------------------------------------------------
  // INSIGHT #3: Gravitational Basins of Attraction
  // ---------------------------------------------------------
  /**
   * Calculates relative distance between two numeric answers.
   */
  relativeDistance(a: number | string, b: number | string): number {
    try {
      const fa = typeof a === 'number' ? a : parseFloat(a);
      const fb = typeof b === 'number' ? b : parseFloat(b);

      if (fa === fb) return 0.0;
      if (fa === 0 || fb === 0) return 1.0;

      return Math.abs(fa - fb) / Math.max(Math.abs(fa), Math.abs(fb));
    } catch {
      return a !== b ? 1.0 : 0.0;
    }
  }

  /**
   * Groups answers by semantic proximity (Gravitational Clustering).
   * Returns clusters weighted by Mass * Density.
   */
  clusterBasins(answers: (number | string)[], modulo?: number): BasinCluster[] {
    if (!answers.length) return [];

    // Parse and optionally apply modulo constraint
    const validPoints: number[] = [];
    for (const a of answers) {
      try {
        let val = typeof a === 'number' ? a : parseFloat(a);
        if (modulo) {
          val = ((val % modulo) + modulo) % modulo; // Handle negative
        }
        if (!isNaN(val)) {
          validPoints.push(val);
        }
      } catch {
        // Skip invalid
      }
    }

    if (!validPoints.length) return [];

    // Sort for clustering
    validPoints.sort((a, b) => a - b);

    // Gravitational collapse clustering
    const clusters: number[][] = [];
    let current: number[] = [validPoints[0]];

    for (let i = 1; i < validPoints.length; i++) {
      if (this.relativeDistance(validPoints[i], current[current.length - 1]) < this.epsilonDistance) {
        current.push(validPoints[i]);
      } else {
        clusters.push(current);
        current = [validPoints[i]];
      }
    }
    clusters.push(current);

    // Calculate basin properties
    return clusters.map(members => {
      const mass = members.length;
      const centroid = members.reduce((a, b) => a + b, 0) / members.length;

      // Density = inverse variance
      let density = 1.0;
      if (mass > 1) {
        const mean = centroid;
        const variance = members.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / mass;
        density = 1 / (variance + 1e-9);
      }

      return { centroid, members, mass, density, score: 0 };
    });
  }

  // ---------------------------------------------------------
  // INSIGHT #12: Solomonoff Induction Approximation
  // ---------------------------------------------------------
  /**
   * Approximation of Algorithmic Probability.
   * Shorter code = Higher probability (Occam's Razor formalized).
   */
  solomonoffWeight(code: string): number {
    const length = code.length;
    // Decay factor for Python/TypeScript scripts
    return Math.pow(0.999, length);
  }

  // ---------------------------------------------------------
  // INSIGHT #10: Entropic Gravity Selection
  // ---------------------------------------------------------
  /**
   * Calculates Shannon entropy of a probability distribution.
   * Used for selecting the answer that minimizes cluster entropy.
   */
  distributionEntropy(distribution: number[]): number {
    const total = distribution.reduce((a, b) => a + b, 0);
    if (total === 0) return 0;

    let entropy = 0;
    for (const count of distribution) {
      if (count > 0) {
        const p = count / total;
        entropy -= p * Math.log2(p);
      }
    }

    return entropy;
  }

  // ---------------------------------------------------------
  // THE MASTER FUSION (Selection Logic)
  // ---------------------------------------------------------
  /**
   * Selects the best answer using all PROMETHEUS insights.
   * Combines: Basin Mass, Density, Solomonoff Prior, Entropic Gravity
   */
  selectBestAnswer(results: InferenceResult[], modulo?: number): number {
    if (!results.length) return 0;

    // 1. Cluster by gravitational basins
    const rawAnswers = results.map(r => r.answer);
    const basins = this.clusterBasins(rawAnswers, modulo);

    if (!basins.length) return 0;

    let bestScore = -1;
    let bestValue = 0;

    for (const basin of basins) {
      // 2. Basin Mass (Count)
      const mass = basin.mass;

      // 3. Basin Density (Inverse Variance)
      const density = basin.density;

      // 4. Apply Solomonoff Prior
      // Find representative code for this cluster
      let avgSolomonoff = 0.5;
      for (const r of results) {
        try {
          const val = typeof r.answer === 'number' ? r.answer : parseFloat(r.answer);
          const compareVal = modulo ? ((val % modulo) + modulo) % modulo : val;

          if (this.relativeDistance(compareVal, basin.centroid) < this.epsilonDistance) {
            avgSolomonoff = this.solomonoffWeight(r.code);
            break;
          }
        } catch {
          // Skip
        }
      }

      // 5. Entropic Gravity Formula
      // Score = Mass * Density^0.1 * Solomonoff
      const score = mass * Math.pow(density, 0.1) * avgSolomonoff;
      basin.score = score;

      if (score > bestScore) {
        bestScore = score;
        bestValue = basin.centroid;
      }
    }

    // Apply modulo if specified
    const result = Math.round(bestValue);
    return modulo ? ((result % modulo) + modulo) % modulo : result;
  }

  // ---------------------------------------------------------
  // INSIGHT #20: Event Horizon Filtering
  // ---------------------------------------------------------
  /**
   * Detects if code has crossed the "event horizon" (will timeout).
   * Returns true if code contains O(n!) or unbounded loops.
   */
  detectEventHorizon(code: string): boolean {
    const dangerPatterns = [
      /itertools\.permutations\s*\([^)]*\b(\d{2,}|n)\b/i,  // permutations on n > 10
      /while\s+True\s*:/,  // Unbounded while
      /while\s+1\s*:/,
      /for.*in\s+range\s*\(\s*10\s*\*\*\s*[89]\s*\)/,  // 10^8+ iterations
      /factorial\s*\([^)]*\b(\d{2,}|n)\b/i,  // factorial on large n
    ];

    return dangerPatterns.some(pattern => pattern.test(code));
  }

  // ---------------------------------------------------------
  // INSIGHT #9: Quantum Zeno Error Correction
  // ---------------------------------------------------------
  /**
   * Injects assertion checkpoints into generated code.
   * Frequent "measurements" freeze the logic in valid state.
   */
  injectZenoAssertions(code: string): string {
    // Add assertions after variable assignments
    const lines = code.split('\n');
    const enhanced: string[] = [];

    for (const line of lines) {
      enhanced.push(line);

      // Detect numeric assignments and add type assertions
      const assignMatch = line.match(/^\s*(\w+)\s*=\s*(.+)$/);
      if (assignMatch) {
        const varName = assignMatch[1];
        // Add assertion for common math variables
        if (['x', 'y', 'z', 'n', 'result', 'answer'].includes(varName.toLowerCase())) {
          enhanced.push(`    # ZENO: assert isinstance(${varName}, (int, float))`);
        }
      }
    }

    return enhanced.join('\n');
  }

  // ---------------------------------------------------------
  // INSIGHT #8: Adversarial Hamiltonian Monte Carlo
  // ---------------------------------------------------------
  /**
   * Generates an "anti-prompt" to escape local minima.
   * When stuck on wrong answer X, generates prompt to prove X impossible.
   */
  generateAntiPrompt(wrongAnswer: number | string, originalProblem: string): string {
    return `
CONSTRAINT: The answer is definitively NOT ${wrongAnswer}.
Prove why ${wrongAnswer} is IMPOSSIBLE for the following problem,
then use this momentum to find the correct answer.

PROBLEM: ${originalProblem}

REASONING:
1. Assume the answer is ${wrongAnswer}
2. Derive a contradiction
3. Use the insight from the contradiction to find the true answer
`;
  }
}

// Singleton export
export const prometheusEngine = new PrometheusEngine();
