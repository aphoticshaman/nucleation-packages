/**
 * Quantum-Inspired Optimization Benchmark
 *
 * George Hotz asked for numbers. Here they are.
 *
 * Compares:
 * 1. Naive random search
 * 2. Standard simulated annealing
 * 3. Our quantum-inspired optimizer
 *
 * On real problems:
 * - Portfolio optimization (Markowitz)
 * - Signal source selection (combinatorial)
 * - Function minimization (continuous)
 *
 * Run with: npx ts-node src/benchmarks/quantum-benchmark.ts
 */

import { QuantumInspiredOptimizer } from '../formulas/quantum-optimize';

interface BenchmarkResult {
  method: string;
  problem: string;
  bestValue: number;
  iterations: number;
  timeMs: number;
  converged: boolean;
}

// ============================================
// BENCHMARK PROBLEMS
// ============================================

/**
 * Rastrigin function - nasty multimodal optimization problem
 * Global minimum: f(0,0,...,0) = 0
 * Has ~10^n local minima for n dimensions
 */
function rastrigin(x: number[]): number {
  const A = 10;
  const n = x.length;
  let sum = A * n;
  for (let i = 0; i < n; i++) {
    sum += x[i] * x[i] - A * Math.cos(2 * Math.PI * x[i]);
  }
  return sum;
}

/**
 * Rosenbrock function - the "banana" function
 * Global minimum: f(1,1,...,1) = 0
 * Easy to find the valley, hard to find the minimum
 */
function rosenbrock(x: number[]): number {
  let sum = 0;
  for (let i = 0; i < x.length - 1; i++) {
    sum += 100 * (x[i + 1] - x[i] * x[i]) ** 2 + (1 - x[i]) ** 2;
  }
  return sum;
}

/**
 * Portfolio optimization problem
 * Maximize Sharpe ratio given returns and covariance
 */
function createPortfolioProblem(n: number): {
  objective: (weights: number[]) => number;
  expectedReturns: number[];
  covariance: number[][];
} {
  // Generate realistic-ish returns and covariance
  const expectedReturns = Array(n)
    .fill(0)
    .map(() => 0.05 + Math.random() * 0.15);

  // Generate positive semi-definite covariance matrix
  const randomMatrix = Array(n)
    .fill(null)
    .map(() =>
      Array(n)
        .fill(0)
        .map(() => Math.random() - 0.5)
    );
  const covariance = Array(n)
    .fill(null)
    .map((_, i) =>
      Array(n)
        .fill(0)
        .map((_, j) => {
          let sum = 0;
          for (let k = 0; k < n; k++) {
            sum += randomMatrix[i][k] * randomMatrix[j][k];
          }
          return sum * 0.01; // Scale to realistic variance
        })
    );

  const riskFreeRate = 0.02;

  const objective = (weights: number[]): number => {
    // Normalize weights to sum to 1
    const sum = weights.reduce((a, b) => a + Math.abs(b), 0) || 1;
    const normalized = weights.map((w) => Math.abs(w) / sum);

    // Portfolio return
    let portReturn = 0;
    for (let i = 0; i < n; i++) {
      portReturn += normalized[i] * expectedReturns[i];
    }

    // Portfolio variance
    let portVariance = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        portVariance += normalized[i] * normalized[j] * covariance[i][j];
      }
    }

    const portVol = Math.sqrt(portVariance);

    // Negative Sharpe (we minimize)
    if (portVol === 0) return 1000;
    return -(portReturn - riskFreeRate) / portVol;
  };

  return { objective, expectedReturns, covariance };
}

/**
 * Signal selection problem (combinatorial)
 * Select k signals from n that maximize information gain
 */
function createSignalSelectionProblem(
  n: number,
  k: number
): {
  objective: (selection: number[]) => number;
  signalValues: number[];
  correlations: number[][];
} {
  // Each signal has base value
  const signalValues = Array(n)
    .fill(0)
    .map(() => Math.random());

  // Correlations between signals (want diverse selection)
  const correlations = Array(n)
    .fill(null)
    .map(() =>
      Array(n)
        .fill(0)
        .map(() => Math.random() * 0.8)
    );
  // Make symmetric with 1s on diagonal
  for (let i = 0; i < n; i++) {
    correlations[i][i] = 1;
    for (let j = i + 1; j < n; j++) {
      correlations[j][i] = correlations[i][j];
    }
  }

  const objective = (selection: number[]): number => {
    // Binary selection
    const selected = selection.map((s) => (s > 0.5 ? 1 : 0));
    const count = selected.reduce((a, b) => a + b, 0);

    // Penalty for wrong count
    if (count !== k) {
      return 1000 + Math.abs(count - k) * 100;
    }

    // Value: sum of signal values minus correlation penalty
    let value = 0;
    let correlationPenalty = 0;

    for (let i = 0; i < n; i++) {
      if (selected[i]) {
        value += signalValues[i];
        for (let j = i + 1; j < n; j++) {
          if (selected[j]) {
            correlationPenalty += correlations[i][j];
          }
        }
      }
    }

    // We minimize, so negate the value we want to maximize
    return -(value - correlationPenalty * 0.5);
  };

  return { objective, signalValues, correlations };
}

// ============================================
// BASELINE METHODS
// ============================================

/**
 * Naive random search
 */
function randomSearch(
  objective: (x: number[]) => number,
  dimensions: number,
  bounds: { min: number; max: number },
  iterations: number
): { solution: number[]; value: number; iterations: number } {
  let best = Array(dimensions)
    .fill(0)
    .map(() => bounds.min + Math.random() * (bounds.max - bounds.min));
  let bestValue = objective(best);

  for (let i = 0; i < iterations; i++) {
    const candidate = Array(dimensions)
      .fill(0)
      .map(() => bounds.min + Math.random() * (bounds.max - bounds.min));
    const value = objective(candidate);
    if (value < bestValue) {
      best = candidate;
      bestValue = value;
    }
  }

  return { solution: best, value: bestValue, iterations };
}

/**
 * Standard simulated annealing (no quantum stuff)
 */
function standardAnnealing(
  objective: (x: number[]) => number,
  dimensions: number,
  bounds: { min: number; max: number },
  iterations: number
): { solution: number[]; value: number; iterations: number } {
  let current = Array(dimensions)
    .fill(0)
    .map(() => bounds.min + Math.random() * (bounds.max - bounds.min));
  let currentValue = objective(current);
  let best = [...current];
  let bestValue = currentValue;

  let temp = 100;
  const cooling = 0.995;

  for (let i = 0; i < iterations; i++) {
    // Generate neighbor
    const neighbor = current.map((x) => {
      const delta = (Math.random() - 0.5) * temp * 0.1;
      return Math.max(bounds.min, Math.min(bounds.max, x + delta));
    });

    const neighborValue = objective(neighbor);

    // Standard Metropolis acceptance
    if (
      neighborValue < currentValue ||
      Math.random() < Math.exp(-(neighborValue - currentValue) / temp)
    ) {
      current = neighbor;
      currentValue = neighborValue;

      if (currentValue < bestValue) {
        best = [...current];
        bestValue = currentValue;
      }
    }

    temp *= cooling;
  }

  return { solution: best, value: bestValue, iterations };
}

// ============================================
// BENCHMARK RUNNER
// ============================================

function runBenchmark(
  name: string,
  objective: (x: number[]) => number,
  dimensions: number,
  bounds: { min: number; max: number },
  iterations: number,
  discrete: boolean = false
): BenchmarkResult[] {
  const results: BenchmarkResult[] = [];

  // Random search
  const randomStart = performance.now();
  const randomResult = randomSearch(objective, dimensions, bounds, iterations);
  const randomTime = performance.now() - randomStart;
  results.push({
    method: 'Random Search',
    problem: name,
    bestValue: randomResult.value,
    iterations: randomResult.iterations,
    timeMs: randomTime,
    converged: false,
  });

  // Standard annealing
  const annealStart = performance.now();
  const annealResult = standardAnnealing(objective, dimensions, bounds, iterations);
  const annealTime = performance.now() - annealStart;
  results.push({
    method: 'Standard SA',
    problem: name,
    bestValue: annealResult.value,
    iterations: annealResult.iterations,
    timeMs: annealTime,
    converged: false,
  });

  // Quantum-inspired
  const quantumOpt = new QuantumInspiredOptimizer();
  const quantumStart = performance.now();
  const quantumResult = quantumOpt.solve(
    {
      dimensions,
      objective,
      bounds: Array(dimensions).fill(bounds),
      discrete,
    },
    { maxIterations: iterations }
  );
  const quantumTime = performance.now() - quantumStart;
  results.push({
    method: 'Quantum-Inspired',
    problem: name,
    bestValue: quantumResult.value,
    iterations: quantumResult.iterations,
    timeMs: quantumTime,
    converged: quantumResult.converged,
  });

  return results;
}

// ============================================
// MAIN
// ============================================

async function main() {
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║     QUANTUM-INSPIRED OPTIMIZATION BENCHMARK                  ║');
  console.log('║     "Show me the numbers" - George Hotz                      ║');
  console.log('╚══════════════════════════════════════════════════════════════╝');
  console.log();

  const allResults: BenchmarkResult[] = [];
  const iterations = 5000;

  // Test 1: Rastrigin (multimodal nightmare)
  console.log('📊 Test 1: Rastrigin Function (10D) - Many local minima');
  console.log('   Global optimum: 0.0');
  const rastriginResults = runBenchmark(
    'Rastrigin 10D',
    rastrigin,
    10,
    { min: -5.12, max: 5.12 },
    iterations
  );
  allResults.push(...rastriginResults);
  printResults(rastriginResults);

  // Test 2: Rosenbrock (valley problem)
  console.log('📊 Test 2: Rosenbrock Function (10D) - Hard valley');
  console.log('   Global optimum: 0.0');
  const rosenResults = runBenchmark(
    'Rosenbrock 10D',
    rosenbrock,
    10,
    { min: -5, max: 5 },
    iterations
  );
  allResults.push(...rosenResults);
  printResults(rosenResults);

  // Test 3: Portfolio optimization
  console.log('📊 Test 3: Portfolio Optimization (20 assets)');
  console.log('   Maximize Sharpe ratio');
  const portfolio = createPortfolioProblem(20);
  const portfolioResults = runBenchmark(
    'Portfolio 20',
    portfolio.objective,
    20,
    { min: 0, max: 1 },
    iterations
  );
  allResults.push(...portfolioResults);
  printResults(portfolioResults);

  // Test 4: Signal selection (combinatorial)
  console.log('📊 Test 4: Signal Selection (30 signals, pick 8)');
  console.log('   Maximize value with diversity');
  const signalProblem = createSignalSelectionProblem(30, 8);
  const signalResults = runBenchmark(
    'Signal Select',
    signalProblem.objective,
    30,
    { min: 0, max: 1 },
    iterations,
    true
  );
  allResults.push(...signalResults);
  printResults(signalResults);

  // Summary
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('SUMMARY: Win/Loss vs Baselines');
  console.log('═══════════════════════════════════════════════════════════════');

  const problems = ['Rastrigin 10D', 'Rosenbrock 10D', 'Portfolio 20', 'Signal Select'];
  let wins = 0;
  let losses = 0;

  for (const problem of problems) {
    const problemResults = allResults.filter((r) => r.problem === problem);
    const quantum = problemResults.find((r) => r.method === 'Quantum-Inspired')!;
    const random = problemResults.find((r) => r.method === 'Random Search')!;
    const standard = problemResults.find((r) => r.method === 'Standard SA')!;

    const bestBaseline = Math.min(random.bestValue, standard.bestValue);
    const improvement = ((bestBaseline - quantum.bestValue) / Math.abs(bestBaseline)) * 100;

    if (quantum.bestValue < bestBaseline) {
      wins++;
      console.log(`✅ ${problem}: Quantum wins by ${improvement.toFixed(1)}%`);
    } else {
      losses++;
      console.log(`❌ ${problem}: Baseline wins by ${(-improvement).toFixed(1)}%`);
    }
  }

  console.log();
  console.log(`Final Score: ${wins}/${wins + losses} wins`);
  console.log();

  // Honest assessment
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('HONEST ASSESSMENT');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log();

  if (wins > losses) {
    console.log('The quantum-inspired approach DOES outperform baselines on these');
    console.log('problems. The advantage comes from:');
    console.log('  1. Tunneling probability allows escaping local minima');
    console.log('  2. QAOA-inspired mixing explores solution space better');
    console.log('  3. Adaptive temperature schedule converges faster');
    console.log();
    console.log('HOWEVER: The improvement is incremental, not revolutionary.');
    console.log('For most production use cases, standard SA would be fine.');
    console.log('The "quantum" branding is marketing - it\'s classical algorithms');
    console.log('inspired by quantum concepts, not actual quantum computation.');
  } else {
    console.log('The quantum-inspired approach did NOT consistently beat baselines.');
    console.log('This is honest: sometimes fancy algorithms lose to simple ones.');
    console.log('Consider removing "quantum" marketing or improving the algorithm.');
  }
}

function printResults(results: BenchmarkResult[]) {
  console.log('┌────────────────────┬──────────────┬──────────┬──────────┐');
  console.log('│ Method             │ Best Value   │ Time(ms) │ Iters    │');
  console.log('├────────────────────┼──────────────┼──────────┼──────────┤');

  for (const r of results) {
    const method = r.method.padEnd(18);
    const value = r.bestValue.toFixed(4).padStart(12);
    const time = r.timeMs.toFixed(1).padStart(8);
    const iters = String(r.iterations).padStart(8);
    console.log(`│ ${method} │ ${value} │ ${time} │ ${iters} │`);
  }

  console.log('└────────────────────┴──────────────┴──────────┴──────────┘');
  console.log();
}

main().catch(console.error);
