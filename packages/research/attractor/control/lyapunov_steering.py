"""
lyapunov_steering.py

Control Lyapunov Function (CLF) based steering for Great Attractor dynamics.

Implements:
- CLF construction for basin convergence guarantees
- QP-based control synthesis (CLF-QP)
- Barrier functions for safety constraints
- Integration with OGSE roadmap execution

A Control Lyapunov Function V(x) satisfies:
    inf_u [∇V(x)·f(x,u)] < -α(V(x))

guaranteeing exponential convergence to the target attractor basin.

References:
- Ames et al. "Control Barrier Functions" (2019)
- Sontag "A Lyapunov-Like Characterization of Asymptotic Controllability"
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict, Any
from enum import Enum

Array = NDArray[np.float64]


class ControlMode(Enum):
    """Control synthesis mode."""
    CLF_QP = "clf_qp"           # CLF-based quadratic program
    CLF_CBF = "clf_cbf"         # CLF + Control Barrier Function
    MPC = "mpc"                 # Model Predictive Control
    GRADIENT = "gradient"       # Simple gradient descent


@dataclass
class CLFConfig:
    """Configuration for CLF-based steering."""

    # CLF parameters
    alpha: float = 1.0          # Decay rate class-K function coefficient
    gamma: float = 0.1          # CLF relaxation parameter

    # Control constraints
    u_max: float = 10.0         # Maximum control magnitude

    # Barrier function parameters
    use_barrier: bool = True
    barrier_alpha: float = 1.0  # Barrier decay rate

    # Optimization
    dt: float = 0.01
    horizon: int = 10           # MPC horizon

    # Target
    target_pos: Array = field(default_factory=lambda: np.zeros(2))
    target_precision: float = 0.1  # Convergence tolerance


@dataclass
class ControlState:
    """State of the CLF controller."""
    position: Array
    velocity: Array
    lyapunov_value: float
    barrier_value: float
    control_input: Array
    converged: bool


class ControlLyapunovSteering:
    """
    Control Lyapunov Function based steering for Great Attractor.

    Provides formal guarantees on basin convergence through
    CLF-based control synthesis.

    Example
    -------
    >>> clf = ControlLyapunovSteering(CLFConfig(target_pos=np.array([5, 5])))
    >>> state = clf.compute_control(current_pos, current_vel)
    >>> # Apply state.control_input to system
    """

    def __init__(
        self,
        config: CLFConfig,
        dynamics_fn: Optional[Callable[[Array, Array], Array]] = None,
        barrier_fn: Optional[Callable[[Array], float]] = None,
    ):
        """
        Initialize CLF steering controller.

        Parameters
        ----------
        config : CLFConfig
            Controller configuration.
        dynamics_fn : callable, optional
            f(x, u) -> dx/dt system dynamics.
        barrier_fn : callable, optional
            h(x) -> float barrier function (h(x) >= 0 is safe).
        """
        self.cfg = config
        self.dynamics_fn = dynamics_fn or self._default_dynamics
        self.barrier_fn = barrier_fn or self._default_barrier

        # History for analysis
        self.history: List[ControlState] = []

    def _default_dynamics(self, x: Array, u: Array) -> Array:
        """Default single-integrator dynamics: dx/dt = u."""
        return u

    def _default_barrier(self, x: Array) -> float:
        """Default barrier: stay within bounds [-10, 10]^2."""
        return 10.0 - np.max(np.abs(x))

    def lyapunov_function(self, x: Array) -> float:
        """
        Quadratic CLF: V(x) = 0.5 * ||x - target||^2.

        More sophisticated CLFs can use:
        - Sum-of-squares polynomials
        - Neural network approximations
        - Composites with basin geometry
        """
        diff = x - self.cfg.target_pos
        return 0.5 * float(np.sum(diff ** 2))

    def lyapunov_gradient(self, x: Array) -> Array:
        """Gradient ∇V(x)."""
        return x - self.cfg.target_pos

    def barrier_function(self, x: Array) -> float:
        """Control Barrier Function h(x)."""
        return self.barrier_fn(x)

    def barrier_gradient(self, x: Array, eps: float = 1e-5) -> Array:
        """Numerical gradient of barrier function."""
        grad = np.zeros(2)
        for i in range(2):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (self.barrier_fn(x_plus) - self.barrier_fn(x_minus)) / (2 * eps)
        return grad

    def _class_k_function(self, s: float) -> float:
        """Class-K function α(s) = α * s for CLF constraint."""
        return self.cfg.alpha * s

    def compute_control_clf_qp(self, x: Array, v: Array) -> Array:
        """
        Compute control via CLF-QP.

        Solves:
            min_u  ||u||^2
            s.t.   ∇V(x)·f(x,u) <= -α(V(x)) + γ  (CLF constraint)
                   ||u|| <= u_max                  (control constraint)

        For single-integrator dynamics f(x,u) = u, this simplifies to:
            ∇V(x)·u <= -α(V(x)) + γ
        """
        grad_V = self.lyapunov_gradient(x)
        V = self.lyapunov_function(x)

        # Target decay
        target_decay = -self._class_k_function(V) + self.cfg.gamma

        # For simple dynamics, optimal u is along -∇V direction
        # with magnitude to satisfy CLF constraint
        grad_norm = np.linalg.norm(grad_V)

        if grad_norm < 1e-10:
            return np.zeros(2)

        # Direction: opposite to gradient
        direction = -grad_V / grad_norm

        # Magnitude: satisfy ∇V·u <= target_decay
        # ∇V · (-∇V/||∇V|| * mag) = -mag * ||∇V|| <= target_decay
        # mag >= -target_decay / ||∇V||

        if target_decay < 0:
            magnitude = min(-target_decay / grad_norm, self.cfg.u_max)
        else:
            # Already satisfies constraint, use minimal control
            magnitude = min(0.1 * grad_norm, self.cfg.u_max)

        u = magnitude * direction

        return u

    def compute_control_clf_cbf(self, x: Array, v: Array) -> Array:
        """
        Compute control with CLF + CBF constraints.

        Adds safety constraint:
            ∇h(x)·f(x,u) >= -α_b(h(x))  (CBF constraint)
        """
        # Start with CLF control
        u_clf = self.compute_control_clf_qp(x, v)

        if not self.cfg.use_barrier:
            return u_clf

        # Check barrier constraint
        h = self.barrier_function(x)
        grad_h = self.barrier_gradient(x)

        # Required: ∇h·u >= -α_b * h
        # For u_clf, compute ∇h·u_clf
        dot_product = np.dot(grad_h, u_clf)
        required = -self.cfg.barrier_alpha * h

        if dot_product >= required:
            # CLF control already satisfies barrier
            return u_clf

        # Project u_clf to satisfy barrier constraint
        # Find u = u_clf + λ * grad_h such that ∇h·u = required
        grad_h_norm_sq = np.dot(grad_h, grad_h)

        if grad_h_norm_sq < 1e-10:
            return u_clf

        lam = (required - dot_product) / grad_h_norm_sq
        u = u_clf + lam * grad_h

        # Clip to control bounds
        u_norm = np.linalg.norm(u)
        if u_norm > self.cfg.u_max:
            u = u / u_norm * self.cfg.u_max

        return u

    def compute_control(
        self,
        x: Array,
        v: Optional[Array] = None,
        mode: ControlMode = ControlMode.CLF_CBF,
    ) -> ControlState:
        """
        Compute control input for current state.

        Parameters
        ----------
        x : array
            Current position.
        v : array, optional
            Current velocity (for second-order systems).
        mode : ControlMode
            Control synthesis method.

        Returns
        -------
        ControlState
            Contains control input and diagnostic values.
        """
        if v is None:
            v = np.zeros(2)

        # Compute CLF and barrier values
        V = self.lyapunov_function(x)
        h = self.barrier_function(x)

        # Compute control based on mode
        if mode == ControlMode.CLF_QP:
            u = self.compute_control_clf_qp(x, v)
        elif mode == ControlMode.CLF_CBF:
            u = self.compute_control_clf_cbf(x, v)
        elif mode == ControlMode.GRADIENT:
            u = -self.lyapunov_gradient(x)
            u_norm = np.linalg.norm(u)
            if u_norm > self.cfg.u_max:
                u = u / u_norm * self.cfg.u_max
        else:
            u = np.zeros(2)

        # Check convergence
        converged = V < self.cfg.target_precision ** 2 / 2

        state = ControlState(
            position=x.copy(),
            velocity=v.copy(),
            lyapunov_value=V,
            barrier_value=h,
            control_input=u.copy(),
            converged=converged,
        )

        self.history.append(state)

        return state

    def simulate_controlled_trajectory(
        self,
        x0: Array,
        n_steps: int,
        mode: ControlMode = ControlMode.CLF_CBF,
    ) -> Dict[str, Any]:
        """
        Simulate controlled trajectory from initial state.

        Returns
        -------
        dict
            Contains 'positions', 'controls', 'lyapunov', 'barrier', 'times'.
        """
        x = x0.copy()
        v = np.zeros(2)
        dt = self.cfg.dt

        positions = [x.copy()]
        controls = []
        lyapunov_values = [self.lyapunov_function(x)]
        barrier_values = [self.barrier_function(x)]
        times = [0.0]

        for i in range(n_steps):
            state = self.compute_control(x, v, mode)

            if state.converged:
                break

            # Apply control (simple Euler integration)
            dx = self.dynamics_fn(x, state.control_input)
            x = x + dx * dt

            positions.append(x.copy())
            controls.append(state.control_input.copy())
            lyapunov_values.append(state.lyapunov_value)
            barrier_values.append(state.barrier_value)
            times.append((i + 1) * dt)

        return {
            "positions": np.array(positions),
            "controls": np.array(controls) if controls else np.array([]).reshape(0, 2),
            "lyapunov": np.array(lyapunov_values),
            "barrier": np.array(barrier_values),
            "times": np.array(times),
            "converged": state.converged,
        }

    def verify_clf_decrease(self, trajectory: Dict[str, Any]) -> bool:
        """Verify that CLF decreased along trajectory."""
        V = trajectory["lyapunov"]
        dV = np.diff(V)

        # Allow small violations due to discretization
        return np.all(dV <= 0.01)

    def estimate_convergence_time(self, x0: Array) -> float:
        """
        Estimate time to converge from x0.

        For exponential decay with rate α:
            V(t) = V(0) * e^{-αt}

        Time to reach V < ε:
            t = (1/α) * ln(V(0)/ε)
        """
        V0 = self.lyapunov_function(x0)
        eps = self.cfg.target_precision ** 2 / 2

        if V0 <= eps:
            return 0.0

        return (1.0 / self.cfg.alpha) * np.log(V0 / eps)


def integrate_clf_with_ogse(
    clf_controller: ControlLyapunovSteering,
    current_pos: Array,
    roadmap_waypoints: List[Array],
    dt: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    Execute OGSE roadmap with CLF guarantees at each waypoint.

    Parameters
    ----------
    clf_controller : ControlLyapunovSteering
        The CLF controller.
    current_pos : array
        Starting position.
    roadmap_waypoints : list of arrays
        Sequence of target positions from OGSE.
    dt : float
        Time step.

    Returns
    -------
    list of dict
        Trajectory segments for each waypoint.
    """
    segments = []
    x = current_pos.copy()

    for waypoint in roadmap_waypoints:
        # Update target
        clf_controller.cfg.target_pos = waypoint
        clf_controller.history.clear()

        # Simulate to waypoint
        traj = clf_controller.simulate_controlled_trajectory(
            x, n_steps=1000
        )

        segments.append(traj)

        # Update position for next segment
        if len(traj["positions"]) > 0:
            x = traj["positions"][-1]

    return segments
