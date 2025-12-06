/**
 * Landau-Ginzburg Phase Transition Engine
 *
 * THEORETICAL BASIS:
 * Free energy functional: F[φ] = ∫ [½(∇φ)² + V(φ)] dx
 * Mexican hat potential: V(φ) = -½aφ² + ¼bφ⁴
 *
 * For a < 0: Single minimum at φ = 0 (symmetric phase)
 * For a > 0: Double minima at φ = ±√(a/b) (broken symmetry)
 *
 * Phase transition occurs at a = 0 (critical point)
 *
 * APPLICATION: Geopolitical regime transitions
 * - φ represents order parameter (stability index)
 * - a represents external stress (conflict pressure)
 * - Basin depth represents regime resilience
 */

export interface PhaseState {
  orderParameter: number;     // φ: -1 to 1 (normalized)
  velocity: number;           // dφ/dt
  potential: number;          // V(φ)
  basinId: 'left' | 'center' | 'right';
  stability: number;          // 0-1 (distance from critical point)
}

export interface LandauGinzburgConfig {
  a: number;                  // Linear coefficient (stress parameter)
  b: number;                  // Quartic coefficient (always positive)
  damping: number;            // γ: Friction coefficient
  noise: number;              // σ: Stochastic noise amplitude
  dt: number;                 // Time step
}

const DEFAULT_CONFIG: LandauGinzburgConfig = {
  a: 1.0,
  b: 1.0,
  damping: 0.5,
  noise: 0.1,
  dt: 0.01,
};

/**
 * Compute Landau-Ginzburg potential V(φ) = -½aφ² + ¼bφ⁴
 */
export function potential(phi: number, config: Partial<LandauGinzburgConfig> = {}): number {
  const { a, b } = { ...DEFAULT_CONFIG, ...config };
  return -0.5 * a * phi * phi + 0.25 * b * Math.pow(phi, 4);
}

/**
 * Compute derivative dV/dφ = -aφ + bφ³
 */
export function potentialDerivative(phi: number, config: Partial<LandauGinzburgConfig> = {}): number {
  const { a, b } = { ...DEFAULT_CONFIG, ...config };
  return -a * phi + b * Math.pow(phi, 3);
}

/**
 * Find equilibrium points (minima) of the potential
 */
export function findEquilibria(config: Partial<LandauGinzburgConfig> = {}): number[] {
  const { a, b } = { ...DEFAULT_CONFIG, ...config };

  if (a <= 0) {
    // Single minimum at origin (symmetric phase)
    return [0];
  }

  // Two minima at ±√(a/b) plus unstable maximum at 0
  const phiMin = Math.sqrt(a / b);
  return [-phiMin, 0, phiMin];
}

/**
 * Compute basin of attraction for a given state
 */
export function determineBasin(
  phi: number,
  config: Partial<LandauGinzburgConfig> = {}
): 'left' | 'center' | 'right' {
  const equilibria = findEquilibria(config);

  if (equilibria.length === 1) {
    return 'center';
  }

  const leftMin = equilibria[0];
  const rightMin = equilibria[2];

  // Barrier is at φ = 0
  if (phi < 0) {
    return 'left';
  } else if (phi > 0) {
    return 'right';
  }
  return 'center';
}

/**
 * Compute stability (inverse proximity to saddle point)
 */
export function computeStability(
  phi: number,
  config: Partial<LandauGinzburgConfig> = {}
): number {
  const { a, b } = { ...DEFAULT_CONFIG, ...config };

  if (a <= 0) {
    // In symmetric phase, stability is based on curvature at origin
    return Math.min(1, Math.abs(a) / 2);
  }

  const phiMin = Math.sqrt(a / b);
  const basinDepth = potential(0, config) - potential(phiMin, config);

  // Distance from saddle point (barrier)
  const distanceFromSaddle = Math.abs(phi);

  // Normalize: further from saddle = more stable
  const normalized = Math.min(1, distanceFromSaddle / phiMin);

  // Weight by basin depth
  return normalized * Math.min(1, basinDepth);
}

/**
 * Langevin dynamics: overdamped motion in potential landscape
 * dφ/dt = -γ⁻¹ dV/dφ + σξ(t)
 *
 * Where ξ(t) is Gaussian white noise
 */
export function langevinStep(
  state: PhaseState,
  config: Partial<LandauGinzburgConfig> = {}
): PhaseState {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const { damping, noise, dt } = cfg;

  // Deterministic force
  const force = -potentialDerivative(state.orderParameter, cfg);

  // Stochastic kick (Box-Muller transform for Gaussian)
  const u1 = Math.random();
  const u2 = Math.random();
  const gaussian = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  const stochastic = noise * gaussian * Math.sqrt(dt);

  // Update velocity (underdamped dynamics)
  const newVelocity = state.velocity * (1 - damping * dt) + force * dt + stochastic;

  // Update position
  const newPhi = state.orderParameter + newVelocity * dt;

  // Clamp to prevent runaway
  const clampedPhi = Math.max(-2, Math.min(2, newPhi));

  return {
    orderParameter: clampedPhi,
    velocity: newVelocity,
    potential: potential(clampedPhi, cfg),
    basinId: determineBasin(clampedPhi, cfg),
    stability: computeStability(clampedPhi, cfg),
  };
}

/**
 * Simulate trajectory over time
 */
export function simulateTrajectory(
  initialState: PhaseState,
  steps: number,
  config: Partial<LandauGinzburgConfig> = {}
): PhaseState[] {
  const trajectory: PhaseState[] = [initialState];
  let currentState = initialState;

  for (let i = 0; i < steps; i++) {
    currentState = langevinStep(currentState, config);
    trajectory.push(currentState);
  }

  return trajectory;
}

/**
 * Detect phase transition by monitoring basin changes
 */
export interface TransitionEvent {
  timestamp: number;
  fromBasin: 'left' | 'center' | 'right';
  toBasin: 'left' | 'center' | 'right';
  orderParameter: number;
  transitionProbability: number;
}

export function detectTransitions(trajectory: PhaseState[]): TransitionEvent[] {
  const transitions: TransitionEvent[] = [];

  for (let i = 1; i < trajectory.length; i++) {
    if (trajectory[i].basinId !== trajectory[i - 1].basinId) {
      transitions.push({
        timestamp: i,
        fromBasin: trajectory[i - 1].basinId,
        toBasin: trajectory[i].basinId,
        orderParameter: trajectory[i].orderParameter,
        transitionProbability: 1 - trajectory[i - 1].stability,
      });
    }
  }

  return transitions;
}

/**
 * Compute Kramers escape rate (transition probability per unit time)
 *
 * r = (ω_0 * ω_b / 2πγ) * exp(-ΔV / kT)
 *
 * Simplified version assuming thermal equilibrium
 */
export function kramersEscapeRate(
  config: Partial<LandauGinzburgConfig> = {},
  temperature: number = 1.0
): number {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const { a, b, damping } = cfg;

  if (a <= 0) {
    return 0; // No barrier to escape
  }

  const phiMin = Math.sqrt(a / b);
  const barrierHeight = potential(0, cfg) - potential(phiMin, cfg);

  // Curvatures at minimum and saddle
  const omegaMin = Math.sqrt(2 * a); // ω² = d²V/dφ² at minimum
  const omegaSaddle = Math.sqrt(a);   // at saddle (simpler model)

  // Kramers formula
  const prefactor = (omegaMin * omegaSaddle) / (2 * Math.PI * damping);
  const boltzmann = Math.exp(-barrierHeight / temperature);

  return prefactor * boltzmann;
}

/**
 * Map external stress parameters to Landau coefficient 'a'
 *
 * Stress factors that increase transition probability:
 * - Economic instability
 * - Military tension
 * - Political polarization
 */
export function stressToLandauCoefficient(
  economicStress: number,    // 0-1
  militaryTension: number,   // 0-1
  politicalPolarization: number  // 0-1
): number {
  // Weighted combination
  const combinedStress =
    0.35 * economicStress +
    0.40 * militaryTension +
    0.25 * politicalPolarization;

  // Map to 'a' coefficient
  // Low stress → a < 0 (single stable basin)
  // High stress → a > 0 (bistable, ready to flip)
  return 2 * combinedStress - 1; // Range: -1 to +1
}

/**
 * Generate potential curve points for visualization
 */
export function generatePotentialCurve(
  config: Partial<LandauGinzburgConfig> = {},
  numPoints: number = 100
): Array<{ phi: number; V: number }> {
  const points: Array<{ phi: number; V: number }> = [];

  for (let i = 0; i < numPoints; i++) {
    const phi = -2 + (4 * i) / (numPoints - 1);
    points.push({
      phi,
      V: potential(phi, config),
    });
  }

  return points;
}

/**
 * Create initial phase state from nation risk metrics
 */
export function createPhaseState(
  stabilityIndex: number,     // 0-1 (higher = more stable)
  currentRegime: 'stable' | 'volatile' | 'crisis'
): PhaseState {
  // Map regime to order parameter
  let orderParameter: number;
  switch (currentRegime) {
    case 'stable':
      orderParameter = 0.8; // Right basin (positive order)
      break;
    case 'volatile':
      orderParameter = 0.1; // Near saddle
      break;
    case 'crisis':
      orderParameter = -0.8; // Left basin (negative order)
      break;
  }

  return {
    orderParameter,
    velocity: 0,
    potential: potential(orderParameter, { a: 1 - stabilityIndex }),
    basinId: determineBasin(orderParameter),
    stability: stabilityIndex,
  };
}
