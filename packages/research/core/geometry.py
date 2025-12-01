"""
Manifold Geometry Layer for Socio-Informational Systems.

Complements the quantum-info layer with geometric/topological structure:
- Riemannian metric on state manifold
- Free-energy landscape and gradient flows
- Attractor basin topology and boundaries
- Manifold collapse detection
- Curvature analysis for phase transitions

Key insight: The "curvature" around Great Attractors creates geodesic
deviation - nearby trajectories converge toward high-intentionality agents.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Dict, Callable
from dataclasses import dataclass, field
from scipy.linalg import eigh, svd, norm
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RBFInterpolator


@dataclass
class RiemannianMetric:
    """
    Riemannian metric tensor g_{ij} on state manifold.

    The metric defines distances and angles on the curved space
    of possible system states.

    In information geometry: this is the Fisher information metric.
    """
    tensor: NDArray[np.float64]  # Shape: (d, d)

    @property
    def dimension(self) -> int:
        return self.tensor.shape[0]

    @property
    def determinant(self) -> float:
        """Volume element √det(g)."""
        return float(np.sqrt(np.abs(np.linalg.det(self.tensor))))

    def inner_product(
        self,
        v1: NDArray[np.float64],
        v2: NDArray[np.float64]
    ) -> float:
        """Compute ⟨v1, v2⟩_g = v1ᵀ g v2."""
        return float(v1 @ self.tensor @ v2)

    def norm(self, v: NDArray[np.float64]) -> float:
        """Compute ||v||_g = √⟨v,v⟩_g."""
        return np.sqrt(max(0, self.inner_product(v, v)))

    def geodesic_distance(
        self,
        x1: NDArray[np.float64],
        x2: NDArray[np.float64]
    ) -> float:
        """
        Approximate geodesic distance for small displacements.

        For infinitesimal displacement: ds² = dx^i g_{ij} dx^j
        """
        dx = x2 - x1
        return self.norm(dx)


@dataclass
class AttractorBasinGeometry:
    """
    Geometric characterization of an attractor basin.

    Includes boundary detection, volume, and curvature properties.
    """
    center: NDArray[np.float64]  # Basin center (attractor location)
    boundary_points: NDArray[np.float64]  # Shape: (n_points, d)
    hessian: NDArray[np.float64]  # Hessian at center (curvature)
    depth: float  # Free energy depth
    volume: float  # Basin volume estimate

    @property
    def dimension(self) -> int:
        return len(self.center)

    @property
    def mean_curvature(self) -> float:
        """Mean curvature = Tr(Hessian) / d."""
        return float(np.trace(self.hessian) / self.dimension)

    @property
    def gaussian_curvature(self) -> float:
        """Gaussian curvature = det(Hessian)."""
        return float(np.linalg.det(self.hessian))

    @property
    def principal_curvatures(self) -> NDArray[np.float64]:
        """Eigenvalues of Hessian = principal curvatures."""
        return np.linalg.eigvalsh(self.hessian)


@dataclass
class FreeEnergyLandscape:
    """
    Free energy landscape F(x) on state manifold.

    Dynamics follow gradient descent: dx/dt = -∇F(x)

    Attractors are local minima of F.
    Transition states are saddle points.
    Basin boundaries are separatrices.
    """
    grid_points: NDArray[np.float64]  # Shape: (N, d)
    values: NDArray[np.float64]  # Shape: (N,)
    gradient: Optional[NDArray[np.float64]] = None  # Shape: (N, d)
    hessian_at_points: Optional[NDArray[np.float64]] = None  # Shape: (N, d, d)

    def interpolate(self, x: NDArray[np.float64]) -> float:
        """Interpolate F(x) at arbitrary point."""
        rbf = RBFInterpolator(self.grid_points, self.values)
        return float(rbf(x.reshape(1, -1))[0])

    def find_minima(self, threshold: float = 0.1) -> List[NDArray[np.float64]]:
        """Find local minima (attractors)."""
        minima = []
        min_val = np.min(self.values)

        # Simple approach: find points below threshold above minimum
        candidates = self.grid_points[self.values < min_val + threshold]

        # Cluster nearby candidates
        if len(candidates) > 0:
            from scipy.cluster.hierarchy import fcluster, linkage
            if len(candidates) > 1:
                Z = linkage(candidates, method='ward')
                clusters = fcluster(Z, t=threshold, criterion='distance')
                for c in np.unique(clusters):
                    cluster_points = candidates[clusters == c]
                    cluster_values = self.values[self.values < min_val + threshold][clusters == c]
                    minima.append(cluster_points[np.argmin(cluster_values)])
            else:
                minima.append(candidates[0])

        return minima


def compute_fisher_metric(
    log_likelihood_grad: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    theta: NDArray[np.float64],
    n_samples: int = 1000,
    epsilon: float = 1e-4
) -> RiemannianMetric:
    """
    Compute Fisher information metric at parameter θ.

    g_{ij}(θ) = E[∂_i log p(x|θ) ∂_j log p(x|θ)]

    This is the natural Riemannian metric on statistical manifolds.

    Args:
        log_likelihood_grad: Function returning ∇_θ log p(x|θ)
        theta: Parameter point
        n_samples: Monte Carlo samples for expectation
        epsilon: Finite difference step

    Returns:
        Fisher information metric at θ
    """
    d = len(theta)

    # Estimate via finite differences and covariance
    grads = []
    for _ in range(n_samples):
        # Sample perturbation
        noise = np.random.randn(d) * epsilon
        grad = log_likelihood_grad(theta + noise)
        grads.append(grad)

    grads = np.array(grads)

    # Fisher metric = covariance of score function
    fisher = np.cov(grads.T)

    # Ensure positive definiteness
    eigvals, eigvecs = eigh(fisher)
    eigvals = np.maximum(eigvals, 1e-6)
    fisher = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return RiemannianMetric(tensor=fisher)


def compute_ricci_curvature_scalar(
    metric: RiemannianMetric,
    metric_at_neighbors: List[RiemannianMetric],
    epsilon: float = 0.1
) -> float:
    """
    Estimate Ricci scalar curvature from metric variations.

    R = g^{ij} R_{ij} where R_{ij} is Ricci tensor.

    High positive curvature: converging geodesics (attractor).
    Negative curvature: diverging geodesics (repeller).

    This is a finite-difference approximation.
    """
    d = metric.dimension
    g = metric.tensor
    g_inv = np.linalg.inv(g)

    # Estimate Christoffel symbols and curvature from neighbors
    # Simplified: use volume deficit method

    # Volume of ball in curved space vs flat
    # For small balls: V_curved ≈ V_flat * (1 - R/(6(d+2)) * r²)

    det_g = np.linalg.det(g)
    neighbor_dets = [np.linalg.det(m.tensor) for m in metric_at_neighbors]

    # Laplacian of √det(g) gives Ricci scalar
    laplacian_sqrt_det = (np.mean(neighbor_dets) - det_g) / epsilon**2

    # R ≈ -2 * Δ(√det g) / √det g  (simplified formula)
    ricci_scalar = -2 * laplacian_sqrt_det / max(np.sqrt(det_g), 1e-10)

    return float(ricci_scalar)


def compute_geodesic(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    metric_field: Callable[[NDArray[np.float64]], RiemannianMetric],
    n_steps: int = 100
) -> NDArray[np.float64]:
    """
    Compute geodesic path between two points on Riemannian manifold.

    Uses shooting method with metric-aware interpolation.

    Args:
        start: Starting point
        end: Ending point
        metric_field: Function returning metric at each point
        n_steps: Number of discretization steps

    Returns:
        Geodesic path, shape (n_steps, d)
    """
    d = len(start)

    # Initialize with straight line
    t = np.linspace(0, 1, n_steps)
    path = np.outer(1 - t, start) + np.outer(t, end)

    # Iterative refinement (relaxation method)
    for iteration in range(50):
        new_path = path.copy()

        for i in range(1, n_steps - 1):
            # Get metric at current point
            g = metric_field(path[i]).tensor
            g_inv = np.linalg.inv(g)

            # Geodesic equation: minimize length
            # Approximate: weighted average of neighbors
            weights = np.array([
                metric_field(path[i-1]).norm(path[i] - path[i-1]),
                metric_field(path[i+1]).norm(path[i+1] - path[i])
            ])
            weights = weights / weights.sum()

            new_path[i] = weights[0] * path[i-1] + weights[1] * path[i+1]

        # Check convergence
        if np.max(np.abs(new_path - path)) < 1e-6:
            break

        path = new_path

    return path


def detect_basin_boundaries(
    landscape: FreeEnergyLandscape,
    attractor_centers: List[NDArray[np.float64]],
    n_boundary_points: int = 100
) -> List[NDArray[np.float64]]:
    """
    Detect boundaries between attractor basins.

    Basin boundaries are the "ridges" of the free energy landscape -
    separatrices where gradient flow changes direction.

    Uses watershed-like algorithm on gradient field.

    Returns:
        List of boundary point arrays for each basin pair
    """
    n_attractors = len(attractor_centers)
    if n_attractors < 2:
        return []

    # Assign each grid point to nearest attractor by gradient flow
    basin_assignments = np.zeros(len(landscape.grid_points), dtype=int)

    for i, point in enumerate(landscape.grid_points):
        # Find which attractor this point flows to
        distances = [norm(point - a) for a in attractor_centers]
        basin_assignments[i] = np.argmin(distances)

    # Find boundary points (where neighboring assignments differ)
    boundaries = []

    for i in range(n_attractors):
        for j in range(i + 1, n_attractors):
            # Points assigned to basin i with neighbors in basin j
            boundary_candidates = []

            # Use distance-based neighbor detection
            distances = cdist(landscape.grid_points, landscape.grid_points)
            neighbor_threshold = np.percentile(distances[distances > 0], 10)

            for k, point in enumerate(landscape.grid_points):
                if basin_assignments[k] == i:
                    neighbors = np.where(distances[k] < neighbor_threshold)[0]
                    if any(basin_assignments[n] == j for n in neighbors):
                        boundary_candidates.append(point)

            if boundary_candidates:
                boundaries.append(np.array(boundary_candidates))

    return boundaries


def compute_manifold_collapse_rate(
    trajectory: NDArray[np.float64],
    metric_field: Callable[[NDArray[np.float64]], RiemannianMetric]
) -> NDArray[np.float64]:
    """
    Compute rate of manifold collapse along trajectory.

    Collapse = rate at which effective dimensionality decreases,
    measured by volume contraction.

    Args:
        trajectory: State trajectory, shape (T, d)
        metric_field: Function returning metric at each point

    Returns:
        Collapse rate at each time step
    """
    T = len(trajectory)
    collapse_rates = np.zeros(T - 1)

    for t in range(T - 1):
        g_t = metric_field(trajectory[t])
        g_t1 = metric_field(trajectory[t + 1])

        # Volume element ratio
        vol_ratio = g_t1.determinant / max(g_t.determinant, 1e-10)

        # Collapse rate = log of volume contraction
        collapse_rates[t] = -np.log(max(vol_ratio, 1e-10))

    return collapse_rates


def compute_basin_curvature_tensor(
    free_energy: Callable[[NDArray[np.float64]], float],
    center: NDArray[np.float64],
    epsilon: float = 0.01
) -> NDArray[np.float64]:
    """
    Compute Hessian (curvature tensor) of free energy at basin center.

    H_{ij} = ∂²F/∂x_i∂x_j

    Eigenvalues give principal curvatures (stability directions).
    Large positive eigenvalues = strong attractor in that direction.
    """
    d = len(center)
    hessian = np.zeros((d, d))

    f_center = free_energy(center)

    for i in range(d):
        for j in range(i, d):
            # Mixed partial derivative via finite differences
            e_i = np.zeros(d)
            e_i[i] = epsilon
            e_j = np.zeros(d)
            e_j[j] = epsilon

            f_pp = free_energy(center + e_i + e_j)
            f_pm = free_energy(center + e_i - e_j)
            f_mp = free_energy(center - e_i + e_j)
            f_mm = free_energy(center - e_i - e_j)

            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
            hessian[j, i] = hessian[i, j]

    return hessian


def estimate_basin_volume(
    center: NDArray[np.float64],
    boundary_points: NDArray[np.float64],
    metric: RiemannianMetric
) -> float:
    """
    Estimate basin volume using convex hull and metric.

    V = ∫_basin √det(g) dx

    Approximated using boundary extent and metric determinant.
    """
    if len(boundary_points) < 2:
        return 0.0

    # Estimate extent in each direction
    extents = boundary_points - center

    # Use metric-corrected volume
    # V ≈ √det(g) * Π_i (extent_i)
    extent_norms = [metric.norm(e) for e in extents]
    mean_extent = np.mean(extent_norms)

    d = len(center)
    volume = metric.determinant * (mean_extent ** d)

    return float(volume)


class ManifoldGeometryAnalyzer:
    """
    Complete geometric analysis of state-space manifold.

    Bridges quantum-info layer (algebraic) with attractor dynamics (topological).
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.trajectory_history: List[NDArray[np.float64]] = []
        self.metric_cache: Dict[str, RiemannianMetric] = {}
        self.basins: List[AttractorBasinGeometry] = []

    def add_trajectory(self, trajectory: NDArray[np.float64]):
        """Record trajectory for analysis."""
        self.trajectory_history.append(trajectory)

    def build_free_energy_landscape(
        self,
        n_grid: int = 50,
        bounds: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]] = None
    ) -> FreeEnergyLandscape:
        """
        Construct free energy landscape from trajectory density.

        F(x) = -log p(x) where p(x) is empirical density.
        """
        if not self.trajectory_history:
            raise ValueError("No trajectories recorded")

        # Combine all trajectories
        all_points = np.vstack(self.trajectory_history)

        # Determine bounds
        if bounds is None:
            min_vals = all_points.min(axis=0) - 1
            max_vals = all_points.max(axis=0) + 1
        else:
            min_vals, max_vals = bounds

        # Create grid
        grids = [np.linspace(min_vals[i], max_vals[i], n_grid)
                 for i in range(self.dimension)]
        mesh = np.meshgrid(*grids, indexing='ij')
        grid_points = np.stack([m.flatten() for m in mesh], axis=1)

        # Estimate density via KDE
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(all_points.T)
            densities = kde(grid_points.T)
        except:
            # Fallback: histogram-based estimate
            densities = np.ones(len(grid_points))
            for i, p in enumerate(grid_points):
                dist = np.min(cdist([p], all_points)[0])
                densities[i] = np.exp(-dist)

        # Free energy = -log(density)
        densities = np.maximum(densities, 1e-10)
        free_energies = -np.log(densities)

        # Normalize
        free_energies -= free_energies.min()

        return FreeEnergyLandscape(
            grid_points=grid_points,
            values=free_energies
        )

    def identify_attractors(
        self,
        landscape: FreeEnergyLandscape,
        depth_threshold: float = 0.5
    ) -> List[AttractorBasinGeometry]:
        """
        Identify attractor basins from free energy landscape.
        """
        minima = landscape.find_minima(threshold=depth_threshold)

        self.basins = []

        for center in minima:
            # Compute Hessian at minimum
            def f(x):
                return landscape.interpolate(x)

            hessian = compute_basin_curvature_tensor(f, center)

            # Find boundary points (approximate)
            # Points where gradient points away from center
            distances = cdist([center], landscape.grid_points)[0]
            nearby = landscape.grid_points[distances < np.percentile(distances, 30)]

            # Boundary = points where energy starts increasing
            nearby_values = np.array([landscape.interpolate(p) for p in nearby])
            center_value = landscape.interpolate(center)

            boundary_mask = nearby_values > center_value + depth_threshold
            boundary_points = nearby[boundary_mask] if boundary_mask.any() else nearby

            # Estimate volume
            metric = RiemannianMetric(tensor=np.eye(self.dimension))
            volume = estimate_basin_volume(center, boundary_points, metric)

            basin = AttractorBasinGeometry(
                center=center,
                boundary_points=boundary_points,
                hessian=hessian,
                depth=float(center_value),
                volume=volume
            )

            self.basins.append(basin)

        return self.basins

    def compute_collapse_trajectory(
        self,
        trajectory: NDArray[np.float64]
    ) -> Dict[str, NDArray[np.float64]]:
        """
        Analyze manifold collapse along a trajectory.

        Returns dictionary with:
        - effective_dims: Effective dimensionality over time
        - collapse_rates: Rate of dimension reduction
        - basin_proximities: Proximity to identified basins
        """
        T = len(trajectory)

        # Compute local covariance as proxy for metric
        window_size = min(20, T // 5)

        effective_dims = np.zeros(T)
        collapse_rates = np.zeros(T - 1)

        for t in range(window_size, T - window_size):
            # Local covariance
            local_traj = trajectory[t - window_size:t + window_size]
            cov = np.cov(local_traj.T)

            # Effective dimension via participation ratio
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.maximum(eigvals, 1e-10)
            eigvals = eigvals / eigvals.sum()
            effective_dims[t] = 1.0 / np.sum(eigvals**2)

        # Fill edges
        effective_dims[:window_size] = effective_dims[window_size]
        effective_dims[-window_size:] = effective_dims[-window_size - 1]

        # Collapse rate = negative gradient of effective dimension
        collapse_rates = -np.gradient(effective_dims)[:-1]

        # Basin proximities
        basin_proximities = np.zeros((T, max(len(self.basins), 1)))
        for i, basin in enumerate(self.basins):
            for t, point in enumerate(trajectory):
                dist = norm(point - basin.center)
                basin_proximities[t, i] = np.exp(-dist / max(basin.volume ** (1/self.dimension), 0.1))

        return {
            'effective_dims': effective_dims,
            'collapse_rates': collapse_rates,
            'basin_proximities': basin_proximities
        }

    def detect_phase_transition_geometry(
        self,
        landscape: FreeEnergyLandscape
    ) -> List[Dict]:
        """
        Detect geometric signatures of phase transitions.

        Transitions occur at saddle points where Hessian has mixed signature.
        """
        transitions = []

        # Find saddle points (local maxima in some directions, minima in others)
        for i, point in enumerate(landscape.grid_points):
            if i % 100 != 0:  # Subsample for efficiency
                continue

            def f(x):
                return landscape.interpolate(x)

            try:
                hessian = compute_basin_curvature_tensor(f, point, epsilon=0.05)
                eigvals = np.linalg.eigvalsh(hessian)

                # Saddle point: mixed positive and negative eigenvalues
                n_positive = np.sum(eigvals > 0.1)
                n_negative = np.sum(eigvals < -0.1)

                if n_positive > 0 and n_negative > 0:
                    transitions.append({
                        'location': point,
                        'hessian': hessian,
                        'eigenvalues': eigvals,
                        'morse_index': n_negative,  # Number of unstable directions
                        'free_energy': landscape.values[i]
                    })
            except:
                continue

        return transitions

    def compute_geodesic_deviation(
        self,
        trajectories: List[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """
        Compute geodesic deviation between nearby trajectories.

        Measures how fast trajectories converge (attractor) or diverge (chaos).

        Negative deviation = convergence toward Great Attractor.
        """
        if len(trajectories) < 2:
            return np.array([])

        # Pairwise distances over time
        T = min(len(t) for t in trajectories)
        n_traj = len(trajectories)

        deviations = np.zeros(T)

        for t in range(T):
            points_at_t = np.array([traj[t] for traj in trajectories])
            pairwise_dist = cdist(points_at_t, points_at_t)
            deviations[t] = np.mean(pairwise_dist[np.triu_indices(n_traj, k=1)])

        # Deviation rate (Lyapunov-like)
        deviation_rate = np.gradient(deviations)

        return deviation_rate


def great_attractor_curvature(
    intentionality_gradient: NDArray[np.float64],
    causal_mass: float,
    distance: float
) -> float:
    """
    Compute curvature of possibility-space around a Great Attractor.

    Analogous to Schwarzschild curvature in GR:
    κ ∝ M / r²

    But here M is causal mass and r is distance in possibility space.

    High curvature = strong geodesic convergence = high influence.

    Args:
        intentionality_gradient: ∇𝓘 at the agent
        causal_mass: Agent's causal mass M_agent
        distance: Distance in possibility space

    Returns:
        Curvature magnitude
    """
    grad_magnitude = norm(intentionality_gradient)

    # Curvature scales with causal mass and gradient, inversely with distance²
    curvature = (causal_mass * grad_magnitude) / max(distance**2, 0.01)

    return float(curvature)


def possibility_space_metric(
    state: NDArray[np.float64],
    uncertainty: NDArray[np.float64],
    intentionality: float
) -> RiemannianMetric:
    """
    Construct metric tensor on possibility space.

    The metric encodes:
    1. Epistemic uncertainty (Fisher information)
    2. Intentionality (curvature around agent's goals)

    g_{ij} = δ_{ij}/σ_i² + I * n_i n_j

    where n is the intentionality direction.
    """
    d = len(state)

    # Base: Fisher-like metric from uncertainty
    fisher = np.diag(1.0 / np.maximum(uncertainty**2, 1e-6))

    # Add intentionality contribution
    # Direction toward agent's goal creates additional curvature
    intent_direction = state / max(norm(state), 1e-6)
    intent_tensor = intentionality * np.outer(intent_direction, intent_direction)

    metric = fisher + intent_tensor

    # Ensure positive definiteness
    eigvals, eigvecs = eigh(metric)
    eigvals = np.maximum(eigvals, 1e-6)
    metric = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return RiemannianMetric(tensor=metric)
