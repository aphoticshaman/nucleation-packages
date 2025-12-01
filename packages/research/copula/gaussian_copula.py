"""
Gaussian Copula for Tail Dependence Modeling.

Based on expired S&P patent US8112340B2 - methods for computing
default correlations using Gaussian copula framework.

Key concepts:
- Copulas separate marginal distributions from dependence structure
- Gaussian copula: C(u,v) = Φ₂(Φ⁻¹(u), Φ⁻¹(v); ρ)
- Tail dependence measures extremal co-movements
- Used for credit risk, systemic risk, multi-asset modeling

The Gaussian copula has asymptotic tail independence (λ = 0),
but empirical tail dependence can be measured and compared
to other copula families (Clayton, Gumbel, t-copula).

Applications in LatticeForge:
- Model correlation structure between nations/entities
- Detect regime changes in correlation structure
- Measure tail risk during crises
- Inform diversification strategies
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Dict, Callable
from dataclasses import dataclass
from scipy import stats
from scipy.stats import norm, multivariate_normal, kendalltau, spearmanr
from scipy.optimize import minimize, brentq
from scipy.linalg import cholesky, solve_triangular


@dataclass
class CopulaConfig:
    """Configuration for copula estimation and simulation."""
    n_simulations: int = 10000
    tail_threshold: float = 0.05  # Quantile for tail dependence
    correlation_method: str = "kendall"  # "kendall", "spearman", "pearson"
    random_seed: Optional[int] = 42


def compute_kendall_tau(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """
    Compute Kendall's tau rank correlation.

    τ = (concordant - discordant) / (n choose 2)

    Robust to outliers and invariant under monotonic transformations.
    Directly related to copula parameter for many families.
    """
    tau, _ = kendalltau(x, y)
    return float(tau)


def compute_spearman_rho(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """
    Compute Spearman's rank correlation.

    ρ_s = Pearson correlation of ranks

    Also copula-invariant and robust.
    """
    rho, _ = spearmanr(x, y)
    return float(rho)


def kendall_to_gaussian_rho(tau: float) -> float:
    """
    Convert Kendall's tau to Gaussian copula correlation.

    For Gaussian copula: τ = (2/π) arcsin(ρ)
    Therefore: ρ = sin(π τ / 2)
    """
    return float(np.sin(np.pi * tau / 2))


def spearman_to_gaussian_rho(rho_s: float) -> float:
    """
    Convert Spearman's rho to Gaussian copula correlation.

    For Gaussian copula: ρ_s = (6/π) arcsin(ρ/2)
    Therefore: ρ = 2 sin(π ρ_s / 6)
    """
    return float(2 * np.sin(np.pi * rho_s / 6))


def fit_gaussian_copula(
    data: NDArray[np.float64],
    method: str = "kendall"
) -> NDArray[np.float64]:
    """
    Fit Gaussian copula correlation matrix to multivariate data.

    Steps:
    1. Compute pairwise rank correlations (Kendall or Spearman)
    2. Convert to Gaussian copula correlations
    3. Ensure positive definiteness

    Args:
        data: Shape (n_samples, n_variables)
        method: "kendall" or "spearman"

    Returns:
        Correlation matrix for Gaussian copula
    """
    n_vars = data.shape[1]
    corr_matrix = np.eye(n_vars)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if method == "kendall":
                tau = compute_kendall_tau(data[:, i], data[:, j])
                rho = kendall_to_gaussian_rho(tau)
            else:  # spearman
                rho_s = compute_spearman_rho(data[:, i], data[:, j])
                rho = spearman_to_gaussian_rho(rho_s)

            # Clip to valid range
            rho = np.clip(rho, -0.999, 0.999)
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    # Ensure positive definiteness via nearest correlation matrix
    corr_matrix = _nearest_positive_definite(corr_matrix)

    return corr_matrix


def _nearest_positive_definite(A: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Find nearest positive definite matrix to A.

    Uses Higham's algorithm.
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = V.T @ np.diag(s) @ V
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    # Check positive definiteness
    try:
        np.linalg.cholesky(A3)
        return A3
    except np.linalg.LinAlgError:
        # Add small positive diagonal
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while True:
            try:
                np.linalg.cholesky(A3 + k * spacing * I)
                return A3 + k * spacing * I
            except np.linalg.LinAlgError:
                k *= 2
                if k > 1e10:
                    return np.eye(A.shape[0])


def simulate_from_copula(
    corr_matrix: NDArray[np.float64],
    n_samples: int,
    marginals: Optional[List[Callable[[NDArray[np.float64]], NDArray[np.float64]]]] = None,
    seed: Optional[int] = None
) -> NDArray[np.float64]:
    """
    Simulate samples from Gaussian copula with given correlation.

    Args:
        corr_matrix: Gaussian copula correlation matrix
        n_samples: Number of samples to generate
        marginals: Optional list of inverse CDF functions for marginals
        seed: Random seed

    Returns:
        Samples, shape (n_samples, n_variables)
    """
    if seed is not None:
        np.random.seed(seed)

    n_vars = corr_matrix.shape[0]

    # Generate multivariate normal
    mvn = multivariate_normal(mean=np.zeros(n_vars), cov=corr_matrix)
    z = mvn.rvs(size=n_samples)

    if n_samples == 1:
        z = z.reshape(1, -1)

    # Transform to uniform via Gaussian CDF
    u = norm.cdf(z)

    # Apply marginal inverse CDFs if provided
    if marginals is not None:
        for j, inv_cdf in enumerate(marginals):
            u[:, j] = inv_cdf(u[:, j])

    return u


def compute_tail_dependence(
    data: NDArray[np.float64],
    threshold: float = 0.05,
    which: str = "lower"
) -> NDArray[np.float64]:
    """
    Compute empirical tail dependence coefficients.

    Lower tail dependence:
    λ_L = lim_{u→0} P(U ≤ u | V ≤ u)

    Upper tail dependence:
    λ_U = lim_{u→1} P(U > u | V > u)

    For Gaussian copula, theoretical λ = 0 (asymptotic independence),
    but empirical measures capture finite-sample tail behavior.

    Args:
        data: Shape (n_samples, n_variables)
        threshold: Quantile threshold (e.g., 0.05 for 5th percentile)
        which: "lower", "upper", or "both"

    Returns:
        Matrix of tail dependence coefficients
    """
    n_vars = data.shape[1]
    n_samples = data.shape[0]

    # Convert to ranks / pseudo-observations
    ranks = np.zeros_like(data)
    for j in range(n_vars):
        ranks[:, j] = stats.rankdata(data[:, j]) / (n_samples + 1)

    tail_dep = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                tail_dep[i, j] = 1.0
                continue

            if which in ["lower", "both"]:
                # Lower tail: P(U_i < q | U_j < q)
                mask_j = ranks[:, j] < threshold
                if mask_j.sum() > 0:
                    lower_dep = (ranks[:, i][mask_j] < threshold).mean()
                else:
                    lower_dep = 0.0

            if which in ["upper", "both"]:
                # Upper tail: P(U_i > 1-q | U_j > 1-q)
                mask_j = ranks[:, j] > (1 - threshold)
                if mask_j.sum() > 0:
                    upper_dep = (ranks[:, i][mask_j] > (1 - threshold)).mean()
                else:
                    upper_dep = 0.0

            if which == "lower":
                tail_dep[i, j] = lower_dep
            elif which == "upper":
                tail_dep[i, j] = upper_dep
            else:  # both - average
                tail_dep[i, j] = (lower_dep + upper_dep) / 2

    return tail_dep


class GaussianCopula:
    """
    Gaussian Copula model for dependence modeling.

    Implements the S&P patent methodology for:
    - Fitting copula to data
    - Simulating correlated samples
    - Computing tail dependence
    - Detecting correlation regime changes
    """

    def __init__(
        self,
        n_variables: int,
        config: Optional[CopulaConfig] = None
    ):
        self.n_variables = n_variables
        self.config = config or CopulaConfig()

        self.correlation_matrix: Optional[NDArray[np.float64]] = None
        self.cholesky_factor: Optional[NDArray[np.float64]] = None
        self.marginals: List[Callable] = []

        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

    def fit(self, data: NDArray[np.float64]) -> 'GaussianCopula':
        """
        Fit Gaussian copula to data.

        Args:
            data: Shape (n_samples, n_variables)

        Returns:
            self
        """
        if data.shape[1] != self.n_variables:
            raise ValueError(f"Expected {self.n_variables} variables, got {data.shape[1]}")

        self.correlation_matrix = fit_gaussian_copula(
            data,
            method=self.config.correlation_method
        )

        # Compute Cholesky factor for efficient simulation
        self.cholesky_factor = cholesky(self.correlation_matrix, lower=True)

        return self

    def fit_marginals(
        self,
        data: NDArray[np.float64],
        distributions: Optional[List[str]] = None
    ) -> 'GaussianCopula':
        """
        Fit marginal distributions to each variable.

        Args:
            data: Shape (n_samples, n_variables)
            distributions: List of distribution names or None for empirical

        Returns:
            self
        """
        self.marginals = []

        for j in range(self.n_variables):
            x = data[:, j]

            if distributions is None or distributions[j] == "empirical":
                # Use empirical inverse CDF
                sorted_x = np.sort(x)
                n = len(x)

                def make_inv_cdf(sorted_vals, n):
                    def inv_cdf(u):
                        idx = np.clip((u * n).astype(int), 0, n - 1)
                        return sorted_vals[idx]
                    return inv_cdf

                self.marginals.append(make_inv_cdf(sorted_x, n))
            else:
                # Fit parametric distribution
                dist_name = distributions[j]
                if dist_name == "normal":
                    mu, sigma = norm.fit(x)
                    self.marginals.append(lambda u, m=mu, s=sigma: norm.ppf(u, m, s))
                elif dist_name == "t":
                    df, loc, scale = stats.t.fit(x)
                    self.marginals.append(lambda u, d=df, l=loc, s=scale: stats.t.ppf(u, d, l, s))
                else:
                    # Fallback to empirical
                    sorted_x = np.sort(x)
                    n = len(x)
                    self.marginals.append(lambda u, s=sorted_x, n=n: s[np.clip((u * n).astype(int), 0, n - 1)])

        return self

    def simulate(self, n_samples: int) -> NDArray[np.float64]:
        """
        Simulate samples from the fitted copula.

        Args:
            n_samples: Number of samples

        Returns:
            Samples, shape (n_samples, n_variables)
        """
        if self.correlation_matrix is None:
            raise ValueError("Copula not fitted. Call fit() first.")

        return simulate_from_copula(
            self.correlation_matrix,
            n_samples,
            marginals=self.marginals if self.marginals else None,
            seed=None  # Use current random state
        )

    def compute_tail_dependence(
        self,
        data: NDArray[np.float64],
        which: str = "both"
    ) -> NDArray[np.float64]:
        """Compute empirical tail dependence from data."""
        return compute_tail_dependence(
            data,
            threshold=self.config.tail_threshold,
            which=which
        )

    def log_likelihood(self, data: NDArray[np.float64]) -> float:
        """
        Compute log-likelihood of data under fitted copula.

        Args:
            data: Shape (n_samples, n_variables)

        Returns:
            Log-likelihood
        """
        if self.correlation_matrix is None:
            raise ValueError("Copula not fitted.")

        n_samples = data.shape[0]

        # Convert to pseudo-observations (uniform)
        u = np.zeros_like(data)
        for j in range(self.n_variables):
            u[:, j] = stats.rankdata(data[:, j]) / (n_samples + 1)

        # Transform to normal
        z = norm.ppf(u)

        # Copula density: c(u) = |Σ|^{-1/2} exp(-z'(Σ^{-1} - I)z / 2)
        Sigma_inv = np.linalg.inv(self.correlation_matrix)
        log_det = np.linalg.slogdet(self.correlation_matrix)[1]

        log_lik = 0.0
        for i in range(n_samples):
            zi = z[i]
            quad_form = zi @ (Sigma_inv - np.eye(self.n_variables)) @ zi
            log_lik += -0.5 * log_det - 0.5 * quad_form

        return float(log_lik)


class TailDependenceAnalyzer:
    """
    Analyzer for tail dependence and correlation regime changes.

    Key applications:
    - Detect crisis periods (when tail dependence spikes)
    - Compare correlation structure across regimes
    - Identify systemically important entities
    """

    def __init__(self, config: Optional[CopulaConfig] = None):
        self.config = config or CopulaConfig()
        self.history: List[Dict] = []

    def rolling_tail_dependence(
        self,
        data: NDArray[np.float64],
        window: int = 60,
        which: str = "lower"
    ) -> NDArray[np.float64]:
        """
        Compute rolling window tail dependence.

        Args:
            data: Shape (T, n_variables)
            window: Rolling window size
            which: "lower", "upper", or "both"

        Returns:
            Time series of average tail dependence, shape (T - window + 1,)
        """
        T, n_vars = data.shape
        n_windows = T - window + 1

        tail_deps = np.zeros(n_windows)

        for t in range(n_windows):
            window_data = data[t:t + window]
            td_matrix = compute_tail_dependence(
                window_data,
                threshold=self.config.tail_threshold,
                which=which
            )
            # Average off-diagonal elements
            mask = ~np.eye(n_vars, dtype=bool)
            tail_deps[t] = td_matrix[mask].mean()

        return tail_deps

    def rolling_correlation(
        self,
        data: NDArray[np.float64],
        window: int = 60
    ) -> NDArray[np.float64]:
        """
        Compute rolling window average correlation.

        Args:
            data: Shape (T, n_variables)
            window: Rolling window size

        Returns:
            Time series of average correlation, shape (T - window + 1,)
        """
        T, n_vars = data.shape
        n_windows = T - window + 1

        correlations = np.zeros(n_windows)

        for t in range(n_windows):
            window_data = data[t:t + window]
            corr_matrix = fit_gaussian_copula(window_data, method="kendall")
            mask = ~np.eye(n_vars, dtype=bool)
            correlations[t] = corr_matrix[mask].mean()

        return correlations

    def detect_regime_changes(
        self,
        tail_dependence_series: NDArray[np.float64],
        threshold_percentile: float = 95
    ) -> List[int]:
        """
        Detect regime changes based on tail dependence spikes.

        Args:
            tail_dependence_series: Time series of tail dependence
            threshold_percentile: Percentile threshold for spike detection

        Returns:
            List of indices where regime changes occur
        """
        threshold = np.percentile(tail_dependence_series, threshold_percentile)
        above_threshold = tail_dependence_series > threshold

        # Find transitions from below to above threshold
        regime_changes = []
        for t in range(1, len(above_threshold)):
            if above_threshold[t] and not above_threshold[t - 1]:
                regime_changes.append(t)

        return regime_changes

    def compute_crisis_metrics(
        self,
        data: NDArray[np.float64],
        crisis_periods: List[Tuple[int, int]]
    ) -> Dict[str, float]:
        """
        Compute correlation and tail dependence metrics during crisis periods.

        Args:
            data: Shape (T, n_variables)
            crisis_periods: List of (start, end) index pairs

        Returns:
            Dictionary of crisis metrics
        """
        n_vars = data.shape[1]

        crisis_tail_deps = []
        crisis_corrs = []
        normal_tail_deps = []
        normal_corrs = []

        # Create crisis mask
        T = data.shape[0]
        crisis_mask = np.zeros(T, dtype=bool)
        for start, end in crisis_periods:
            crisis_mask[start:end] = True

        # Crisis periods
        if crisis_mask.sum() > 10:
            crisis_data = data[crisis_mask]
            td = compute_tail_dependence(crisis_data, self.config.tail_threshold, "lower")
            corr = fit_gaussian_copula(crisis_data, "kendall")
            mask = ~np.eye(n_vars, dtype=bool)
            crisis_tail_deps.append(td[mask].mean())
            crisis_corrs.append(corr[mask].mean())

        # Normal periods
        normal_mask = ~crisis_mask
        if normal_mask.sum() > 10:
            normal_data = data[normal_mask]
            td = compute_tail_dependence(normal_data, self.config.tail_threshold, "lower")
            corr = fit_gaussian_copula(normal_data, "kendall")
            mask = ~np.eye(n_vars, dtype=bool)
            normal_tail_deps.append(td[mask].mean())
            normal_corrs.append(corr[mask].mean())

        return {
            'crisis_tail_dependence': float(np.mean(crisis_tail_deps)) if crisis_tail_deps else 0.0,
            'normal_tail_dependence': float(np.mean(normal_tail_deps)) if normal_tail_deps else 0.0,
            'tail_dependence_ratio': (
                float(np.mean(crisis_tail_deps) / max(np.mean(normal_tail_deps), 0.01))
                if crisis_tail_deps and normal_tail_deps else 1.0
            ),
            'crisis_correlation': float(np.mean(crisis_corrs)) if crisis_corrs else 0.0,
            'normal_correlation': float(np.mean(normal_corrs)) if normal_corrs else 0.0,
            'correlation_ratio': (
                float(np.mean(crisis_corrs) / max(np.mean(normal_corrs), 0.01))
                if crisis_corrs and normal_corrs else 1.0
            ),
        }

    def identify_systemic_entities(
        self,
        data: NDArray[np.float64],
        entity_names: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Identify systemically important entities based on tail dependence.

        An entity is systemic if its tail dependence with others is high.

        Args:
            data: Shape (T, n_variables)
            entity_names: Optional names for each variable

        Returns:
            List of (name, score) sorted by systemic importance
        """
        n_vars = data.shape[1]

        if entity_names is None:
            entity_names = [f"Entity_{i}" for i in range(n_vars)]

        # Compute tail dependence matrix
        td_matrix = compute_tail_dependence(data, self.config.tail_threshold, "lower")

        # Systemic score = average tail dependence with all others
        scores = []
        for i in range(n_vars):
            other_deps = [td_matrix[i, j] for j in range(n_vars) if j != i]
            avg_td = np.mean(other_deps)
            scores.append((entity_names[i], float(avg_td)))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores


def compare_copula_families(
    data: NDArray[np.float64],
    families: List[str] = ["gaussian", "t", "clayton", "gumbel"]
) -> Dict[str, float]:
    """
    Compare different copula families using AIC/BIC.

    Note: Full implementation would require fitting each family.
    This is a placeholder showing the structure.

    Args:
        data: Bivariate data, shape (n_samples, 2)
        families: List of copula family names

    Returns:
        Dictionary of family -> log-likelihood
    """
    if data.shape[1] != 2:
        raise ValueError("Copula comparison currently only supports bivariate data")

    results = {}

    # Gaussian copula
    gc = GaussianCopula(n_variables=2)
    gc.fit(data)
    results["gaussian"] = gc.log_likelihood(data)

    # Other families would be implemented similarly
    # For now, return Gaussian result only
    return results
