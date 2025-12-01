"""
Outcome Gradient Steering Engine (OGSE).

This module provides a framework for:
1) Steering a high-dimensional system state toward a desired outcome
   via control inputs (interventions).
2) Generating a ROADMAP: a segmented, human-digestible sequence of
   intervention phases that approximate an optimal trajectory.
3) Integrating Great Attractor dynamics for enhanced steering.

Core ideas:
- State space is represented as a vector z ∈ R^d.
- Dynamics: z_{t+1} = f(z_t, u_t, t).
- Potential/cost landscape: V(z, t) encodes "distance" to goal.
- OGSE searches over control inputs {u_t} to minimize total cost.
- The resulting control trajectory is segmented into PHASES.

Integrates:
- Great Attractor dynamics (intentionality gradients, causal mass)
- Manifold geometry (curvature, basin structure)
- Quantum-info layer (awareness functional, channel capacity)
- Causal emergence (Ψ, Δ measures)

Domain-agnostic: usable for wealth trajectories, organizational roadmaps,
policy influence, sustainability transitions, etc.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Dict, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import norm


# ---------------------------
# Core type definitions
# ---------------------------

State = NDArray[np.float64]      # shape (d,)
Control = NDArray[np.float64]    # shape (m,)


class SteeringMode(Enum):
    """Steering strategy modes."""
    GRADIENT_DESCENT = "gradient_descent"
    GEODESIC = "geodesic"
    OPTIMAL_CONTROL = "optimal_control"
    BASIN_HOPPING = "basin_hopping"
    ATTRACTOR_AMPLIFICATION = "attractor_amplification"
    GREAT_ATTRACTOR = "great_attractor"


@dataclass
class SystemModel:
    """
    General system model z_{t+1} = f(z_t, u_t, t).
    """
    state_dim: int
    control_dim: int
    dynamics: Callable[[State, Control, int], State]


@dataclass
class PotentialModel:
    """
    Potential/cost landscape V(z, t) with optional control cost.
    """
    state_potential: Callable[[State, int], float]
    control_cost: Callable[[Control, int], float]


@dataclass
class Constraint:
    """Soft constraint evaluated as a penalty term."""
    penalty: Callable[[State, Control, int], float]
    weight: float
    name: str = "constraint"


@dataclass
class OGSEConfig:
    """Configuration for OGSE optimization and roadmap generation."""
    horizon: int = 50
    max_iters: int = 200
    step_size: float = 0.05
    control_init_scale: float = 0.1
    control_clip: float = 1.0
    phase_change_angle_deg: float = 25.0
    min_phase_length: int = 3
    random_seed: int = 42
    finite_diff_eps: float = 1e-4


@dataclass
class SteeringObjective:
    """
    Defines the target outcome for steering.

    Can be:
    - Target state (point attractor)
    - Target distribution (probabilistic outcome)
    - Target basin (region of state space)
    - Constraint satisfaction (avoid certain regions)
    """
    target_state: Optional[NDArray[np.float64]] = None
    target_distribution: Optional[Callable[[NDArray[np.float64]], float]] = None
    target_basin_center: Optional[NDArray[np.float64]] = None
    target_basin_radius: float = 1.0
    avoid_regions: List[Tuple[NDArray[np.float64], float]] = field(default_factory=list)

    state_weight: float = 1.0
    uncertainty_weight: float = 0.1
    cost_weight: float = 0.01
    horizon: int = 10

    def evaluate(self, state: NDArray[np.float64]) -> float:
        """Evaluate how well a state satisfies the objective. Higher = better."""
        score = 0.0

        if self.target_state is not None:
            distance = np.linalg.norm(state - self.target_state)
            score += self.state_weight * np.exp(-distance)

        if self.target_distribution is not None:
            score += self.state_weight * self.target_distribution(state)

        if self.target_basin_center is not None:
            distance = np.linalg.norm(state - self.target_basin_center)
            if distance < self.target_basin_radius:
                score += self.state_weight
            else:
                score += self.state_weight * np.exp(-(distance - self.target_basin_radius))

        for center, radius in self.avoid_regions:
            distance = np.linalg.norm(state - center)
            if distance < radius:
                score -= 10.0 * (radius - distance) / radius

        return score


@dataclass
class SteeringAction:
    """Represents a steering action/intervention."""
    action_vector: NDArray[np.float64]
    expected_state: NDArray[np.float64]
    confidence: float
    cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.action_vector))


@dataclass
class Phase:
    """A high-level roadmap phase representing a coherent direction of control."""
    start_step: int
    end_step: int
    avg_control: Control
    description: str = ""
    semantic_label: str = ""  # e.g., "capital_raise", "product_launch"
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> int:
        return self.end_step - self.start_step + 1


@dataclass
class Roadmap:
    """OGSE roadmap: sequence of phases + raw trajectory data."""
    phases: List[Phase]
    states: List[State]
    controls: List[Control]
    total_cost: float
    success_probability: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SteeringResult:
    """Complete result from steering computation."""
    actions: List[SteeringAction]
    predicted_trajectory: NDArray[np.float64]
    objective_scores: NDArray[np.float64]
    success_probability: float
    total_cost: float
    mode_used: SteeringMode
    roadmap: Optional[Roadmap] = None


# ---------------------------
# Utility functions
# ---------------------------

def compute_outcome_gradient(
    state: NDArray[np.float64],
    objective: SteeringObjective,
    dynamics: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    epsilon: float = 0.01
) -> NDArray[np.float64]:
    """Compute gradient of objective with respect to current state."""
    d = len(state)
    gradient = np.zeros(d)
    f_0 = objective.evaluate(state)

    for i in range(d):
        state_plus = state.copy()
        state_plus[i] += epsilon
        f_plus = objective.evaluate(state_plus)
        gradient[i] = (f_plus - f_0) / epsilon

    return gradient


def project_to_action_space(
    gradient: NDArray[np.float64],
    action_bounds: Tuple[NDArray[np.float64], NDArray[np.float64]],
    action_dim: int
) -> NDArray[np.float64]:
    """Project state-space gradient to action space."""
    if len(gradient) >= action_dim:
        action = gradient[:action_dim]
    else:
        action = np.zeros(action_dim)
        action[:len(gradient)] = gradient

    lower, upper = action_bounds
    action = np.clip(action, lower, upper)
    return action


def evaluate_steering_effectiveness(
    actions: List[SteeringAction],
    actual_trajectory: NDArray[np.float64],
    objective: SteeringObjective
) -> Dict[str, float]:
    """Evaluate how effective the steering was."""
    initial_score = objective.evaluate(actual_trajectory[0])
    final_score = objective.evaluate(actual_trajectory[-1])
    improvement = final_score - initial_score
    total_cost = sum(a.cost for a in actions)
    efficiency = improvement / max(total_cost, 0.001)

    predicted = np.array([a.expected_state for a in actions])
    actual = actual_trajectory[1:len(predicted)+1]

    if len(predicted) > 0 and len(actual) > 0:
        min_len = min(len(predicted), len(actual))
        prediction_error = np.mean(np.linalg.norm(predicted[:min_len] - actual[:min_len], axis=1))
    else:
        prediction_error = 0.0

    return {
        'final_objective': float(final_score),
        'improvement': float(improvement),
        'efficiency': float(efficiency),
        'prediction_error': float(prediction_error),
        'total_cost': float(total_cost)
    }


# ---------------------------
# Core OGSE Engine
# ---------------------------

class OutcomeGradientSteeringEngine:
    """
    Outcome Gradient Steering Engine (OGSE).

    Given a system model (dynamics), a potential model (state + control costs),
    and optional constraints, this engine optimizes a sequence of controls
    to steer the initial state towards desirable outcomes, then segments
    the control sequence into roadmap phases.

    Integrates Great Attractor dynamics for enhanced steering.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        dynamics: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
        action_bounds: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]] = None,
        config: Optional[OGSEConfig] = None,
        constraints: Optional[List[Constraint]] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dynamics = dynamics
        self.config = config or OGSEConfig()
        self.constraints = constraints or []

        if action_bounds is None:
            self.action_bounds = (
                -np.ones(action_dim),
                np.ones(action_dim)
            )
        else:
            self.action_bounds = action_bounds

        self.history: List[Dict] = []
        self.transition_jacobian: Optional[NDArray[np.float64]] = None
        self.cost_function: Callable[[NDArray[np.float64]], float] = lambda a: float(np.linalg.norm(a)**2)

        # Great Attractor parameters
        self.intentionality_gradient: Optional[NDArray[np.float64]] = None
        self.causal_mass: float = 1.0

        np.random.seed(self.config.random_seed)

    def set_great_attractor_params(
        self,
        intentionality_gradient: NDArray[np.float64],
        causal_mass: float
    ):
        """Set Great Attractor parameters for enhanced steering."""
        self.intentionality_gradient = intentionality_gradient
        self.causal_mass = causal_mass

    # -------------------------------
    # Simulation and cost evaluation
    # -------------------------------

    def simulate(
        self,
        z0: State,
        controls: NDArray[np.float64],
        potential: Optional[PotentialModel] = None
    ) -> Tuple[List[State], float]:
        """Simulate trajectory and compute total cost."""
        T = len(controls)
        states: List[State] = [z0.copy()]
        total_cost = 0.0
        z = z0.copy()

        for t in range(T):
            u = controls[t]

            if potential is not None:
                V = potential.state_potential(z, t)
                C = potential.control_cost(u, t)
            else:
                V = 0.0
                C = self.cost_function(u)

            penalty_sum = 0.0
            for c in self.constraints:
                penalty_sum += c.weight * c.penalty(z, u, t)

            total_cost += V + C + penalty_sum

            # Dynamics wrapper (handle both signatures)
            try:
                z = self.dynamics(z, u, t)
            except TypeError:
                z = self.dynamics(z, u)

            states.append(z.copy())

        return states, float(total_cost)

    # -------------------------------
    # Gradient-based control optimization
    # -------------------------------

    def _numerical_grad_controls(
        self,
        z0: State,
        controls: NDArray[np.float64],
        potential: Optional[PotentialModel] = None
    ) -> NDArray[np.float64]:
        """Compute gradient of total cost w.r.t. controls via finite differences."""
        eps = self.config.finite_diff_eps
        T, m = controls.shape
        grad = np.zeros_like(controls)

        _, base_cost = self.simulate(z0, controls, potential)

        for t in range(T):
            for j in range(m):
                perturbed = controls.copy()
                perturbed[t, j] += eps
                _, cost_eps = self.simulate(z0, perturbed, potential)
                grad[t, j] = (cost_eps - base_cost) / eps

        return grad

    def optimize_controls(
        self,
        z0: State,
        potential: Optional[PotentialModel] = None,
        u_init: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[NDArray[np.float64], List[State], float]:
        """Optimize controls via gradient descent."""
        T = self.config.horizon
        m = self.action_dim

        if u_init is None:
            u = self.config.control_init_scale * np.random.randn(T, m)
        else:
            u = u_init.copy()

        best_u = u.copy()
        best_states, best_cost = self.simulate(z0, best_u, potential)

        for it in range(self.config.max_iters):
            grad = self._numerical_grad_controls(z0, u, potential)

            # Include Great Attractor influence if available
            if self.intentionality_gradient is not None:
                intent_influence = self.causal_mass * self.intentionality_gradient[:m]
                for t in range(T):
                    grad[t] -= 0.1 * intent_influence  # Attract toward intention

            u -= self.config.step_size * grad

            # Clip control L2 norm per step
            norms = np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
            scale = np.minimum(1.0, self.config.control_clip / norms)
            u *= scale

            states, cost = self.simulate(z0, u, potential)

            if cost < best_cost:
                best_cost = cost
                best_u = u.copy()
                best_states = states

        return best_u, best_states, best_cost

    # -------------------------------
    # Roadmap / phase segmentation
    # -------------------------------

    def _segment_phases(
        self,
        controls: NDArray[np.float64],
    ) -> List[Tuple[int, int]]:
        """Segment control trajectory into phases where direction is stable."""
        T, m = controls.shape
        if T == 0:
            return []

        eps = 1e-12
        norms = np.linalg.norm(controls, axis=1, keepdims=True) + eps
        directions = controls / norms

        # Find first non-zero control for reference
        ref_idx = 0
        while ref_idx < T and np.linalg.norm(controls[ref_idx]) < eps:
            ref_idx += 1
        if ref_idx == T:
            return [(0, T - 1)]

        ref_dir = directions[ref_idx]
        angles_deg = np.zeros(T)

        for t in range(T):
            dot = float(np.clip(np.dot(ref_dir, directions[t]), -1.0, 1.0))
            angles_deg[t] = np.degrees(np.arccos(dot))

        threshold = self.config.phase_change_angle_deg
        min_len = self.config.min_phase_length

        phases: List[Tuple[int, int]] = []
        phase_start = 0

        for t in range(1, T):
            if np.linalg.norm(controls[t]) > eps:
                ref_dir = 0.8 * ref_dir + 0.2 * directions[t]
                ref_dir /= np.linalg.norm(ref_dir) + eps

            if angles_deg[t] > threshold and (t - phase_start) >= min_len:
                phases.append((phase_start, t - 1))
                phase_start = t

        if T - phase_start >= 1:
            phases.append((phase_start, T - 1))

        # Merge short trailing phases
        merged: List[Tuple[int, int]] = []
        for start, end in phases:
            if not merged:
                merged.append((start, end))
            else:
                prev_start, prev_end = merged[-1]
                if (end - start + 1) < min_len:
                    merged[-1] = (prev_start, end)
                else:
                    merged.append((start, end))

        return merged

    def _describe_phase(
        self,
        avg_control: Control,
        phase_idx: int,
        total_phases: int,
    ) -> str:
        """Generate human-readable description of a phase."""
        norm_val = float(np.linalg.norm(avg_control))
        if norm_val < 1e-6:
            magnitude = "negligible"
        elif norm_val < 0.25 * self.config.control_clip:
            magnitude = "small"
        elif norm_val < 0.75 * self.config.control_clip:
            magnitude = "moderate"
        else:
            magnitude = "strong"

        return (
            f"Phase {phase_idx + 1}/{total_phases}: "
            f"{magnitude} coherent intervention (||u|| ≈ {norm_val:.3f})"
        )

    def _infer_semantic_label(
        self,
        avg_control: Control,
        phase_idx: int,
        states_in_phase: List[State]
    ) -> str:
        """
        Infer semantic label for phase. Override for domain-specific labels.

        Default: map to quadrant-based labels.
        """
        m = len(avg_control)
        if m >= 2:
            if avg_control[0] > 0 and avg_control[1] > 0:
                return "expansion"
            elif avg_control[0] > 0 and avg_control[1] < 0:
                return "consolidation"
            elif avg_control[0] < 0 and avg_control[1] > 0:
                return "restructuring"
            else:
                return "optimization"
        elif m == 1:
            return "increase" if avg_control[0] > 0 else "decrease"
        return "intervention"

    def generate_roadmap(
        self,
        z0: State,
        potential: Optional[PotentialModel] = None,
        objective: Optional[SteeringObjective] = None,
        u_init: Optional[NDArray[np.float64]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Roadmap:
        """
        Full OGSE pipeline: optimize controls, simulate trajectory,
        segment into phases, and assemble a roadmap.
        """
        controls_opt, states_opt, total_cost = self.optimize_controls(
            z0, potential=potential, u_init=u_init
        )

        phases_idx = self._segment_phases(controls_opt)
        phases: List[Phase] = []

        for idx, (start, end) in enumerate(phases_idx):
            avg_control = controls_opt[start:end + 1].mean(axis=0)
            desc = self._describe_phase(avg_control, idx, len(phases_idx))
            states_in_phase = states_opt[start:end + 2]
            semantic = self._infer_semantic_label(avg_control, idx, states_in_phase)

            phases.append(Phase(
                start_step=start,
                end_step=end,
                avg_control=avg_control,
                description=desc,
                semantic_label=semantic,
            ))

        # Estimate success probability
        if objective is not None:
            final_score = objective.evaluate(states_opt[-1])
            if objective.target_state is not None:
                dist = np.linalg.norm(states_opt[-1] - objective.target_state)
                success_prob = float(np.exp(-dist))
            else:
                success_prob = min(1.0, final_score)
        else:
            success_prob = 0.5

        roadmap_meta = meta.copy() if meta else {}
        roadmap_meta['horizon'] = self.config.horizon
        roadmap_meta['max_iters'] = self.config.max_iters
        roadmap_meta['great_attractor_enabled'] = self.intentionality_gradient is not None

        return Roadmap(
            phases=phases,
            states=states_opt,
            controls=list(controls_opt),
            total_cost=total_cost,
            success_probability=success_prob,
            meta=roadmap_meta,
        )

    # -------------------------------
    # Legacy steering methods
    # -------------------------------

    def compute_steering(
        self,
        current_state: NDArray[np.float64],
        objective: SteeringObjective,
        mode: SteeringMode = SteeringMode.GRADIENT_DESCENT,
        horizon: Optional[int] = None
    ) -> SteeringResult:
        """Compute optimal steering actions to achieve objective."""
        horizon = horizon or objective.horizon

        if mode == SteeringMode.GREAT_ATTRACTOR:
            return self._steer_great_attractor(current_state, objective, horizon)
        elif mode == SteeringMode.OPTIMAL_CONTROL:
            return self._steer_optimal_control(current_state, objective, horizon)
        elif mode == SteeringMode.BASIN_HOPPING:
            return self._steer_basin_hopping(current_state, objective, horizon)
        else:
            return self._steer_gradient_descent(current_state, objective, horizon)

    def _steer_gradient_descent(
        self,
        state: NDArray[np.float64],
        objective: SteeringObjective,
        horizon: int
    ) -> SteeringResult:
        """Simple gradient descent steering."""
        actions = []
        trajectory = [state.copy()]
        scores = [objective.evaluate(state)]
        current = state.copy()

        for t in range(horizon):
            gradient = compute_outcome_gradient(current, objective, self.dynamics)
            action_vec = project_to_action_space(gradient, self.action_bounds, self.action_dim)
            action_vec = action_vec * 0.1

            try:
                next_state = self.dynamics(current, action_vec, t)
            except TypeError:
                next_state = self.dynamics(current, action_vec)

            action = SteeringAction(
                action_vector=action_vec,
                expected_state=next_state,
                confidence=0.8,
                cost=self.cost_function(action_vec)
            )
            actions.append(action)

            current = next_state
            trajectory.append(current.copy())
            scores.append(objective.evaluate(current))

        final_score = scores[-1]
        if objective.target_state is not None:
            dist_to_target = np.linalg.norm(current - objective.target_state)
            success_prob = np.exp(-dist_to_target)
        else:
            success_prob = min(1.0, final_score)

        return SteeringResult(
            actions=actions,
            predicted_trajectory=np.array(trajectory),
            objective_scores=np.array(scores),
            success_probability=float(success_prob),
            total_cost=sum(a.cost for a in actions),
            mode_used=SteeringMode.GRADIENT_DESCENT
        )

    def _steer_great_attractor(
        self,
        state: NDArray[np.float64],
        objective: SteeringObjective,
        horizon: int
    ) -> SteeringResult:
        """
        Steering that leverages Great Attractor dynamics.

        Instead of fighting possibility-space curvature, align with it.
        """
        actions = []
        trajectory = [state.copy()]
        scores = [objective.evaluate(state)]
        current = state.copy()

        for t in range(horizon):
            obj_gradient = compute_outcome_gradient(current, objective, self.dynamics)

            if self.intentionality_gradient is not None:
                combined_gradient = obj_gradient + self.causal_mass * self.intentionality_gradient
                alignment = np.dot(obj_gradient, self.intentionality_gradient)
                alignment /= (np.linalg.norm(obj_gradient) * np.linalg.norm(self.intentionality_gradient) + 1e-6)
                scale = 0.5 * (1 + alignment)
            else:
                combined_gradient = obj_gradient
                alignment = 0.0
                scale = 0.5

            action_vec = project_to_action_space(combined_gradient, self.action_bounds, self.action_dim)
            action_vec = action_vec * scale * 0.2

            try:
                next_state = self.dynamics(current, action_vec, t)
            except TypeError:
                next_state = self.dynamics(current, action_vec)

            action = SteeringAction(
                action_vector=action_vec,
                expected_state=next_state,
                confidence=0.7 + 0.3 * abs(alignment),
                cost=self.cost_function(action_vec) * (1 - 0.5 * alignment),
                metadata={'alignment': float(alignment), 'causal_mass': self.causal_mass}
            )
            actions.append(action)

            current = next_state
            trajectory.append(current.copy())
            scores.append(objective.evaluate(current))

        return SteeringResult(
            actions=actions,
            predicted_trajectory=np.array(trajectory),
            objective_scores=np.array(scores),
            success_probability=float(np.exp(-np.linalg.norm(current - objective.target_state)) if objective.target_state is not None else 0.5),
            total_cost=sum(a.cost for a in actions),
            mode_used=SteeringMode.GREAT_ATTRACTOR
        )

    def _steer_optimal_control(
        self,
        state: NDArray[np.float64],
        objective: SteeringObjective,
        horizon: int
    ) -> SteeringResult:
        """MPC-style optimization over action sequence."""
        n_params = horizon * self.action_dim

        def objective_fn(params):
            actions_arr = params.reshape(horizon, self.action_dim)
            total_cost = 0.0
            current = state.copy()

            for t in range(horizon):
                action = actions_arr[t]
                total_cost += objective.cost_weight * self.cost_function(action)

                try:
                    current = self.dynamics(current, action, t)
                except TypeError:
                    current = self.dynamics(current, action)

                total_cost -= objective.evaluate(current)

            return total_cost

        lower = np.tile(self.action_bounds[0], horizon)
        upper = np.tile(self.action_bounds[1], horizon)
        bounds = list(zip(lower, upper))

        result = differential_evolution(objective_fn, bounds, maxiter=100, seed=42, workers=1)
        optimal_actions = result.x.reshape(horizon, self.action_dim)

        actions = []
        trajectory = [state.copy()]
        scores = [objective.evaluate(state)]
        current = state.copy()

        for t in range(horizon):
            action_vec = optimal_actions[t]
            try:
                next_state = self.dynamics(current, action_vec, t)
            except TypeError:
                next_state = self.dynamics(current, action_vec)

            action = SteeringAction(
                action_vector=action_vec,
                expected_state=next_state,
                confidence=0.9,
                cost=self.cost_function(action_vec)
            )
            actions.append(action)

            current = next_state
            trajectory.append(current.copy())
            scores.append(objective.evaluate(current))

        return SteeringResult(
            actions=actions,
            predicted_trajectory=np.array(trajectory),
            objective_scores=np.array(scores),
            success_probability=float(np.exp(-result.fun)),
            total_cost=sum(a.cost for a in actions),
            mode_used=SteeringMode.OPTIMAL_CONTROL
        )

    def _steer_basin_hopping(
        self,
        state: NDArray[np.float64],
        objective: SteeringObjective,
        horizon: int
    ) -> SteeringResult:
        """Escape current basin and hop to target basin."""
        actions = []
        trajectory = [state.copy()]
        scores = [objective.evaluate(state)]
        current = state.copy()

        escape_steps = max(1, horizon // 3)

        for t in range(escape_steps):
            gradient = compute_outcome_gradient(current, objective, self.dynamics)

            if objective.target_basin_center is not None:
                escape_dir = objective.target_basin_center - current
                escape_dir = escape_dir / max(np.linalg.norm(escape_dir), 1e-6)
            else:
                escape_dir = -gradient / max(np.linalg.norm(gradient), 1e-6)

            action_vec = project_to_action_space(escape_dir * 2.0, self.action_bounds, self.action_dim)

            try:
                next_state = self.dynamics(current, action_vec, t)
            except TypeError:
                next_state = self.dynamics(current, action_vec)

            action = SteeringAction(
                action_vector=action_vec,
                expected_state=next_state,
                confidence=0.6,
                cost=self.cost_function(action_vec) * 2
            )
            actions.append(action)

            current = next_state
            trajectory.append(current.copy())
            scores.append(objective.evaluate(current))

        settle_result = self._steer_gradient_descent(current, objective, horizon - escape_steps)
        actions.extend(settle_result.actions)
        trajectory.extend(settle_result.predicted_trajectory[1:].tolist())
        scores.extend(settle_result.objective_scores[1:].tolist())

        return SteeringResult(
            actions=actions,
            predicted_trajectory=np.array(trajectory),
            objective_scores=np.array(scores),
            success_probability=settle_result.success_probability * 0.7,
            total_cost=sum(a.cost for a in actions),
            mode_used=SteeringMode.BASIN_HOPPING
        )


# ---------------------------
# Convenience factory
# ---------------------------

def create_simple_dynamics(
    state_dim: int,
    action_dim: int,
    noise_scale: float = 0.01
) -> Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    """Create a simple linear dynamics model for testing."""
    np.random.seed(42)
    A = np.eye(state_dim) * 0.95 + np.random.randn(state_dim, state_dim) * 0.05
    B = np.random.randn(state_dim, action_dim) * 0.3

    def dynamics(state: NDArray[np.float64], action: NDArray[np.float64]) -> NDArray[np.float64]:
        noise = np.random.randn(state_dim) * noise_scale
        return A @ state + B @ action + noise

    return dynamics


# Alias for compatibility
OGSEngine = OutcomeGradientSteeringEngine
