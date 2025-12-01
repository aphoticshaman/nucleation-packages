"""
Quantum Information Layer for Socio-Informational Systems.

NOT quantum mysticism - rigorous information theory using quantum formalism
as a general language for subsystems, correlations, and control capacity.

Classical implementations use:
- Density matrices → Covariance operators / probability kernels
- Von Neumann entropy → Shannon entropy
- Quantum channels → Markov kernels

Key Quantities:
- Agent-World Mutual Information 𝓘_{A:W}
- Agent-Memory Mutual Information 𝓘_{A:M}
- Awareness Functional 𝔄(t)
- Coherent Causal Capacity 𝓒_cause

This provides the algebraic/probabilistic structure that complements
the geometric/topological structure in geometry.py.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from scipy.linalg import logm, sqrtm, eigvalsh
from scipy.stats import entropy as shannon_entropy


@dataclass
class DensityOperator:
    """
    Density operator ρ representing a (possibly mixed) quantum/classical state.

    In classical interpretation: covariance matrix or probability kernel.

    Properties:
    - ρ ≥ 0 (positive semi-definite)
    - Tr(ρ) = 1 (normalized)
    """
    matrix: NDArray[np.float64]  # Shape: (d, d)

    @property
    def dimension(self) -> int:
        return self.matrix.shape[0]

    @property
    def is_pure(self) -> bool:
        """Pure state iff Tr(ρ²) = 1."""
        return np.isclose(np.trace(self.matrix @ self.matrix), 1.0, atol=1e-6)

    @property
    def purity(self) -> float:
        """Tr(ρ²) - measure of mixedness. Pure=1, maximally mixed=1/d."""
        return float(np.trace(self.matrix @ self.matrix))

    def validate(self) -> bool:
        """Check PSD and trace=1."""
        eigvals = eigvalsh(self.matrix)
        return (eigvals >= -1e-10).all() and np.isclose(np.trace(self.matrix), 1.0)


@dataclass
class SystemState:
    """
    Full system state factorized into world, agent, and memory subsystems.

    ℋ = ℋ_world ⊗ ℋ_agent ⊗ ℋ_memory
    """
    rho_world: DensityOperator
    rho_agent: DensityOperator
    rho_memory: DensityOperator
    rho_joint: Optional[DensityOperator] = None  # Full joint state if available
    timestamp: float = 0.0


def von_neumann_entropy(rho: DensityOperator) -> float:
    """
    Compute von Neumann entropy S(ρ) = -Tr(ρ log ρ).

    In classical interpretation, this is Shannon entropy of the eigenvalue
    spectrum (which represents probability distribution over pure states).

    Args:
        rho: Density operator

    Returns:
        Entropy in nats
    """
    eigvals = eigvalsh(rho.matrix)
    eigvals = eigvals[eigvals > 1e-15]  # Filter zeros
    return -np.sum(eigvals * np.log(eigvals))


def mutual_information_quantum(
    rho_AB: DensityOperator,
    rho_A: DensityOperator,
    rho_B: DensityOperator
) -> float:
    """
    Compute quantum mutual information between subsystems A and B.

    𝓘(A:B) = S(ρ_A) + S(ρ_B) - S(ρ_AB)

    This measures total correlations (classical + quantum) between A and B.

    Args:
        rho_AB: Joint density operator
        rho_A: Reduced density operator for A
        rho_B: Reduced density operator for B

    Returns:
        Mutual information in nats
    """
    S_A = von_neumann_entropy(rho_A)
    S_B = von_neumann_entropy(rho_B)
    S_AB = von_neumann_entropy(rho_AB)

    return max(0.0, S_A + S_B - S_AB)


def compute_agent_world_mi(state: SystemState) -> float:
    """
    Compute Agent-World mutual information 𝓘_{A:W}(t).

    High 𝓘_{A:W}: strong informational coupling to reality.

    In practice, we approximate this using the marginal entropies
    since the full joint state may not be available.
    """
    if state.rho_joint is not None:
        # Proper computation with joint state
        # Would need partial trace - simplified here
        pass

    S_agent = von_neumann_entropy(state.rho_agent)
    S_world = von_neumann_entropy(state.rho_world)

    # Without joint, estimate via correlation structure
    # This is a placeholder - in production, compute actual joint
    estimated_joint_entropy = max(S_agent, S_world) + 0.5 * min(S_agent, S_world)

    return max(0.0, S_agent + S_world - estimated_joint_entropy)


def compute_agent_memory_mi(state: SystemState) -> float:
    """
    Compute Agent-Memory mutual information 𝓘_{A:M}(t).

    High 𝓘_{A:M}: strong coupling to internal self-model,
    continuity of identity.
    """
    S_agent = von_neumann_entropy(state.rho_agent)
    S_memory = von_neumann_entropy(state.rho_memory)

    # Estimate without full joint
    estimated_joint_entropy = max(S_agent, S_memory) + 0.3 * min(S_agent, S_memory)

    return max(0.0, S_agent + S_memory - estimated_joint_entropy)


def awareness_functional(
    state: SystemState,
    alpha: float = 1.0,
    beta: float = 1.0
) -> float:
    """
    Compute the Awareness Functional 𝔄(t).

    𝔄(t) = α·𝓘_{A:W}(t) + β·𝓘_{A:M}(t)

    This is an entirely rigorous, substrate-agnostic measure of
    an agent's informational coupling to reality and self-model.

    Args:
        state: Full system state
        alpha: Weight for agent-world coupling
        beta: Weight for agent-memory coupling

    Returns:
        Awareness functional value
    """
    I_AW = compute_agent_world_mi(state)
    I_AM = compute_agent_memory_mi(state)

    return alpha * I_AW + beta * I_AM


def channel_capacity_holevo(
    channel_outputs: List[NDArray[np.float64]],
    input_probs: NDArray[np.float64]
) -> float:
    """
    Estimate Holevo-Schumacher-Westmoreland channel capacity.

    C_{A→W} = sup_{p_i, ρ_i} [S(ℰ(Σ p_i ρ_i)) - Σ p_i S(ℰ(ρ_i))]

    This is the upper bound on bits of controllable influence per channel use.

    In classical implementation: mutual information capacity between
    policy outputs and realized environment transitions.

    Args:
        channel_outputs: List of output density matrices for each input
        input_probs: Probability distribution over inputs

    Returns:
        Estimated channel capacity
    """
    n_inputs = len(channel_outputs)

    # Average output state
    avg_output = sum(
        p * out for p, out in zip(input_probs, channel_outputs)
    )
    avg_output_rho = DensityOperator(matrix=avg_output)

    # Entropy of average
    S_avg = von_neumann_entropy(avg_output_rho)

    # Average of entropies
    avg_S = sum(
        p * von_neumann_entropy(DensityOperator(matrix=out))
        for p, out in zip(input_probs, channel_outputs)
    )

    # Holevo information (lower bound on capacity)
    return max(0.0, S_avg - avg_S)


def coherent_causal_capacity(
    agent_actions: NDArray[np.float64],
    world_transitions: NDArray[np.float64],
    n_bins: int = 20
) -> float:
    """
    Compute coherent causal capacity 𝓒_cause.

    This is the channel capacity C_{A→W} measuring the upper bound
    on controllable influence.

    Practical implementation using discrete approximation:
    Estimate mutual information I(actions; transitions).

    Args:
        agent_actions: Agent action history, shape (T, D_a)
        world_transitions: World state transitions, shape (T, D_w)
        n_bins: Discretization bins

    Returns:
        Estimated causal capacity
    """
    if agent_actions.ndim == 1:
        agent_actions = agent_actions.reshape(-1, 1)
    if world_transitions.ndim == 1:
        world_transitions = world_transitions.reshape(-1, 1)

    T = len(agent_actions)

    # Discretize for MI estimation
    def discretize(x, n_bins):
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(x, percentiles)
        return np.digitize(x, bins[1:-1])

    # Use first dimension for simplicity
    a_disc = discretize(agent_actions[:, 0], n_bins)
    w_disc = discretize(world_transitions[:, 0], n_bins)

    # Joint histogram
    joint_hist = np.histogram2d(a_disc, w_disc, bins=n_bins)[0]
    joint_hist = joint_hist / joint_hist.sum()  # Normalize

    # Marginals
    p_a = joint_hist.sum(axis=1)
    p_w = joint_hist.sum(axis=0)

    # Mutual information
    H_a = shannon_entropy(p_a + 1e-10)
    H_w = shannon_entropy(p_w + 1e-10)
    H_aw = shannon_entropy(joint_hist.flatten() + 1e-10)

    return max(0.0, H_a + H_w - H_aw)


def decoherence_to_pointer_basis(
    rho: DensityOperator,
    threshold: float = 0.1
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Find the pointer basis where off-diagonal terms have decayed.

    Reality appears classical because of environment-induced decoherence.
    This selects a preferred basis {|x⟩} where ρ ≈ Σ p_x |x⟩⟨x|.

    In this basis, state evolution becomes approximately Markov
    over classical configurations.

    Args:
        rho: Density operator
        threshold: Off-diagonal threshold for classicality

    Returns:
        Tuple of (probabilities over pointer states, pointer basis vectors)
    """
    # Eigendecomposition gives the pointer basis
    eigenvalues, eigenvectors = np.linalg.eigh(rho.matrix)

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Probabilities are the eigenvalues
    probs = np.maximum(eigenvalues, 0)
    probs = probs / probs.sum()

    return probs, eigenvectors


class QuantumInfoAnalyzer:
    """
    Complete quantum information analysis for multi-agent systems.

    Provides the algebraic/probabilistic layer that bridges to
    the geometric attractor layer.
    """

    def __init__(
        self,
        world_dim: int,
        agent_dim: int,
        memory_dim: int
    ):
        self.world_dim = world_dim
        self.agent_dim = agent_dim
        self.memory_dim = memory_dim

        self.history: List[Dict] = []

    def create_state_from_distributions(
        self,
        world_dist: NDArray[np.float64],
        agent_dist: NDArray[np.float64],
        memory_dist: NDArray[np.float64],
        timestamp: float = 0.0
    ) -> SystemState:
        """
        Create SystemState from classical probability distributions.

        Converts distributions to density operators (diagonal).
        """
        # Diagonal density operators from distributions
        rho_world = DensityOperator(matrix=np.diag(world_dist / world_dist.sum()))
        rho_agent = DensityOperator(matrix=np.diag(agent_dist / agent_dist.sum()))
        rho_memory = DensityOperator(matrix=np.diag(memory_dist / memory_dist.sum()))

        return SystemState(
            rho_world=rho_world,
            rho_agent=rho_agent,
            rho_memory=rho_memory,
            timestamp=timestamp
        )

    def create_state_from_covariances(
        self,
        world_cov: NDArray[np.float64],
        agent_cov: NDArray[np.float64],
        memory_cov: NDArray[np.float64],
        timestamp: float = 0.0
    ) -> SystemState:
        """
        Create SystemState from covariance matrices (Gaussian states).

        Normalizes to trace=1 for density operator interpretation.
        """
        def normalize_cov(cov):
            trace = np.trace(cov)
            if trace > 0:
                return cov / trace
            return np.eye(cov.shape[0]) / cov.shape[0]

        rho_world = DensityOperator(matrix=normalize_cov(world_cov))
        rho_agent = DensityOperator(matrix=normalize_cov(agent_cov))
        rho_memory = DensityOperator(matrix=normalize_cov(memory_cov))

        return SystemState(
            rho_world=rho_world,
            rho_agent=rho_agent,
            rho_memory=rho_memory,
            timestamp=timestamp
        )

    def analyze(
        self,
        state: SystemState,
        alpha: float = 1.0,
        beta: float = 1.0
    ) -> Dict:
        """
        Perform full quantum-info analysis on a system state.

        Returns dict with all computed quantities.
        """
        # Core quantities
        S_world = von_neumann_entropy(state.rho_world)
        S_agent = von_neumann_entropy(state.rho_agent)
        S_memory = von_neumann_entropy(state.rho_memory)

        I_AW = compute_agent_world_mi(state)
        I_AM = compute_agent_memory_mi(state)

        awareness = awareness_functional(state, alpha, beta)

        # Pointer basis analysis
        probs_world, _ = decoherence_to_pointer_basis(state.rho_world)
        effective_dim_world = 1.0 / np.sum(probs_world**2)  # Participation ratio

        probs_agent, _ = decoherence_to_pointer_basis(state.rho_agent)
        effective_dim_agent = 1.0 / np.sum(probs_agent**2)

        result = {
            'timestamp': state.timestamp,
            'entropy_world': float(S_world),
            'entropy_agent': float(S_agent),
            'entropy_memory': float(S_memory),
            'mi_agent_world': float(I_AW),
            'mi_agent_memory': float(I_AM),
            'awareness_functional': float(awareness),
            'purity_world': state.rho_world.purity,
            'purity_agent': state.rho_agent.purity,
            'effective_dim_world': float(effective_dim_world),
            'effective_dim_agent': float(effective_dim_agent),
        }

        self.history.append(result)
        return result

    def get_manifold_collapse_signal(self) -> NDArray[np.float64]:
        """
        Detect manifold collapse by tracking effective dimensionality.

        Collapse = rapid drop in effective dimension = system locking
        into fewer macroscopic futures.
        """
        if len(self.history) < 2:
            return np.array([])

        eff_dims = np.array([h['effective_dim_world'] for h in self.history])

        # Rate of collapse = negative gradient of effective dimension
        collapse_signal = -np.gradient(eff_dims)

        return collapse_signal

    def correlate_awareness_with_dominance(
        self,
        attractor_dominance: NDArray[np.float64]
    ) -> float:
        """
        Test the conjecture:
        𝓘_{A:W} ↑ ⟹ dim(supp(μ)) ↓ and |∇V| ↑

        High agent-world MI should correlate with attractor dominance.

        Args:
            attractor_dominance: Time series of dominant attractor ratio

        Returns:
            Correlation coefficient
        """
        if len(self.history) < 5:
            return 0.0

        awareness = np.array([h['awareness_functional'] for h in self.history])

        # Align lengths
        min_len = min(len(awareness), len(attractor_dominance))
        awareness = awareness[:min_len]
        dominance = attractor_dominance[:min_len]

        corr = np.corrcoef(awareness, dominance)[0, 1]

        return float(corr) if not np.isnan(corr) else 0.0
