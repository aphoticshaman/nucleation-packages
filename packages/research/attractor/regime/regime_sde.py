"""
regime_sde.py

Regime-switching Stochastic Differential Equation (RS-SDE) for
Great Attractor macro-dynamics.

This module implements:
- Continuous-time regime switching via Markov jump process
- Potential-driven dynamics within each regime
- Coupling between particle state and regime transitions

The regime process r(t) ∈ {0, 1, ..., n_regimes-1} evolves according to
a continuous-time Markov chain with state-dependent transition rates.

The particle dynamics follow:
    dx = -∇U_r(x) dt + σ dW

where U_r is the potential for regime r.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Any


Array = NDArray[np.float64]


@dataclass
class RegimeSDEConfig:
    """Configuration for regime-switching SDE."""

    n_regimes: int = 3
    beta: float = 1.0           # inverse temperature (controls switching)
    dt: float = 0.01            # time step
    diffusion: float = 0.1      # noise strength
    switching_scale: float = 0.5  # base switching rate


class RegimeSDE:
    """
    Regime-switching SDE simulator.

    Implements coupled dynamics:
    1. Regime r(t): Markov jump process with state-dependent rates
    2. Position x(t): Gradient descent + noise in current regime's potential

    Parameters
    ----------
    config : RegimeSDEConfig
        Configuration parameters.
    potential_fns : list of callables
        U_r(x) -> float for each regime r.
    switching_rate_fn : callable, optional
        λ(r, r', x) -> float transition rate from r to r' at state x.
        If None, uses default Arrhenius-like rates.
    """

    def __init__(
        self,
        config: RegimeSDEConfig,
        potential_fns: List[Callable[[Array], float]],
        switching_rate_fn: Optional[Callable[[int, int, Array], float]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.cfg = config
        self.potential_fns = potential_fns
        self.rng = rng or np.random.default_rng()

        if len(potential_fns) != config.n_regimes:
            raise ValueError(
                f"Expected {config.n_regimes} potential functions, got {len(potential_fns)}"
            )

        # Default switching rate: Arrhenius-like
        if switching_rate_fn is None:
            self.switching_rate_fn = self._default_switching_rate
        else:
            self.switching_rate_fn = switching_rate_fn

        # State
        self.regime: int = 0
        self.time: float = 0.0
        self.x: Array = np.zeros(2)

    def _default_switching_rate(self, r_from: int, r_to: int, x: Array) -> float:
        """
        Default Arrhenius-like switching rate.

        λ(r → r') = λ_0 * exp(-β * ΔU)

        where ΔU = U_{r'}(x) - U_r(x) (energy barrier).
        """
        if r_from == r_to:
            return 0.0

        U_from = self.potential_fns[r_from](x)
        U_to = self.potential_fns[r_to](x)
        delta_U = U_to - U_from

        # Only penalize uphill transitions
        barrier = max(0.0, delta_U)
        rate = self.cfg.switching_scale * np.exp(-self.cfg.beta * barrier)

        return rate

    def _compute_gradient(self, x: Array, regime: int, eps: float = 1e-5) -> Array:
        """Numerical gradient of potential U_r at x."""
        U = self.potential_fns[regime]
        grad = np.zeros(2)

        for i in range(2):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (U(x_plus) - U(x_minus)) / (2 * eps)

        return grad

    def _sample_regime_transition(self, x: Array) -> int:
        """
        Sample next regime using Gillespie-like algorithm.

        Returns the new regime (may be same as current).
        """
        r = self.regime
        n = self.cfg.n_regimes
        dt = self.cfg.dt

        # Compute transition rates to all other regimes
        rates = np.array([
            self.switching_rate_fn(r, r_prime, x)
            for r_prime in range(n)
        ])
        rates[r] = 0.0  # no self-transition

        total_rate = rates.sum()

        if total_rate < 1e-12:
            return r

        # Probability of any transition in dt (Poisson approximation)
        p_transition = 1.0 - np.exp(-total_rate * dt)

        if self.rng.random() < p_transition:
            # Transition occurred - sample destination
            probs = rates / total_rate
            return int(self.rng.choice(n, p=probs))
        else:
            return r

    def step(self, x_external: Optional[Array] = None) -> Dict[str, Any]:
        """
        Advance by one time step.

        Parameters
        ----------
        x_external : array, optional
            External position (e.g., swarm centroid) to use instead of internal x.
            Useful for coupling to external dynamics.

        Returns
        -------
        dict
            Contains 'time', 'regime', 'x', 'switched'.
        """
        # Use external position if provided
        if x_external is not None:
            self.x = np.asarray(x_external, dtype=float)

        x = self.x
        r = self.regime
        dt = self.cfg.dt
        sigma = self.cfg.diffusion

        # 1. Regime transition
        r_new = self._sample_regime_transition(x)
        switched = r_new != r
        self.regime = r_new

        # 2. Position dynamics (Euler-Maruyama)
        grad = self._compute_gradient(x, self.regime)
        noise = self.rng.standard_normal(2)
        self.x = x - grad * dt + sigma * np.sqrt(dt) * noise

        # 3. Update time
        self.time += dt

        return {
            "time": self.time,
            "regime": self.regime,
            "x": self.x.copy(),
            "switched": switched,
        }

    def run(self, steps: int, x0: Optional[Array] = None, r0: int = 0) -> Dict[str, Any]:
        """
        Run simulation for multiple steps.

        Parameters
        ----------
        steps : int
            Number of time steps.
        x0 : array, optional
            Initial position.
        r0 : int
            Initial regime.

        Returns
        -------
        dict
            Contains 'times', 'regimes', 'positions', 'switch_times'.
        """
        if x0 is not None:
            self.x = np.asarray(x0, dtype=float)
        self.regime = r0
        self.time = 0.0

        times = []
        regimes = []
        positions = []
        switch_times = []

        for _ in range(steps):
            result = self.step()
            times.append(result["time"])
            regimes.append(result["regime"])
            positions.append(result["x"])
            if result["switched"]:
                switch_times.append(result["time"])

        return {
            "times": np.array(times),
            "regimes": np.array(regimes),
            "positions": np.array(positions),
            "switch_times": np.array(switch_times),
        }

    def switching_rates(self, x: Array, r: Optional[int] = None) -> Array:
        """
        Get instantaneous switching rates from current/specified regime.

        Parameters
        ----------
        x : array
            Current position.
        r : int, optional
            Regime to compute rates from (default: current regime).

        Returns
        -------
        array of shape (n_regimes,)
            Transition rates λ(r → r') for all r'.
        """
        if r is None:
            r = self.regime

        return np.array([
            self.switching_rate_fn(r, r_prime, x)
            for r_prime in range(self.cfg.n_regimes)
        ])

    def get_q_matrix_local(self, x: Array) -> Array:
        """
        Build local Q-matrix at position x.

        Returns
        -------
        Q : array of shape (n_regimes, n_regimes)
            Generator matrix with q_ij = λ(i → j), q_ii = -Σ_{j≠i} q_ij.
        """
        n = self.cfg.n_regimes
        Q = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    Q[i, j] = self.switching_rate_fn(i, j, x)

        # Set diagonal
        Q[np.diag_indices(n)] = -Q.sum(axis=1)

        return Q
