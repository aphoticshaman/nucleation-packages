"""
Tests for persistence diagram computation and visualization.
"""

import numpy as np
import pytest

from attractor.viz.plot_persistence import (
    PersistencePlotConfig,
    PersistencePlotter,
    persistent_entropy,
    compute_distance_matrix,
)


class TestPersistentEntropy:
    """Tests for persistent_entropy function."""

    def test_empty_diagram_zero_entropy(self):
        """Empty persistence diagram should have zero entropy."""
        empty = np.array([]).reshape(0, 2)
        assert persistent_entropy(empty) == 0.0

    def test_single_feature_zero_entropy(self):
        """Single feature should have zero entropy."""
        single = np.array([[0.0, 1.0]])
        # Single feature means p=1, so entropy = -1*log(1) = 0
        assert persistent_entropy(single) == pytest.approx(0.0, abs=1e-10)

    def test_equal_lifetimes_max_entropy(self):
        """Equal lifetimes should give maximum entropy."""
        # 4 features with equal lifetime
        n = 4
        equal_lifetimes = np.array([[0.0, 1.0]] * n)
        entropy = persistent_entropy(equal_lifetimes)

        # Max entropy for n features is log(n)
        max_entropy = np.log(n)
        assert entropy == pytest.approx(max_entropy, rel=0.01)

    def test_unequal_lifetimes_lower_entropy(self):
        """Unequal lifetimes should have lower entropy than equal."""
        # One dominant feature, others small
        unequal = np.array([
            [0.0, 10.0],  # Long-lived
            [0.0, 0.1],   # Short-lived
            [0.0, 0.1],
            [0.0, 0.1],
        ])
        equal = np.array([[0.0, 2.5]] * 4)

        entropy_unequal = persistent_entropy(unequal)
        entropy_equal = persistent_entropy(equal)

        assert entropy_unequal < entropy_equal

    def test_positive_for_multiple_features(self):
        """Entropy should be positive for multiple features."""
        pairs = np.array([
            [0.0, 0.5],
            [0.1, 0.8],
            [0.2, 0.6],
        ])
        assert persistent_entropy(pairs) > 0


class TestComputeDistanceMatrix:
    """Tests for compute_distance_matrix function."""

    def test_square_output(self):
        """Output should be square."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        D = compute_distance_matrix(points)
        assert D.shape == (3, 3)

    def test_diagonal_zero(self):
        """Diagonal should be zero (distance to self)."""
        points = np.array([[0, 0], [1, 1], [2, 2]])
        D = compute_distance_matrix(points)
        assert np.allclose(np.diag(D), 0.0)

    def test_symmetric(self):
        """Distance matrix should be symmetric."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        D = compute_distance_matrix(points)
        assert np.allclose(D, D.T)

    def test_known_distances(self):
        """Test with known geometric distances."""
        points = np.array([
            [0, 0],
            [3, 0],  # distance 3 from origin
            [0, 4],  # distance 4 from origin
        ])
        D = compute_distance_matrix(points)

        assert D[0, 1] == pytest.approx(3.0)
        assert D[0, 2] == pytest.approx(4.0)
        assert D[1, 2] == pytest.approx(5.0)  # 3-4-5 triangle


class TestPersistencePlotConfig:
    """Tests for PersistencePlotConfig dataclass."""

    def test_default_values(self):
        """Default config should have expected values."""
        cfg = PersistencePlotConfig()
        assert cfg.figsize == (10, 5)
        assert cfg.max_dim == 1
        assert cfg.max_edge_length == 2.0


class TestPersistencePlotter:
    """Tests for PersistencePlotter class."""

    @pytest.fixture
    def plotter(self):
        """Create a PersistencePlotter instance."""
        return PersistencePlotter(PersistencePlotConfig())

    def test_compute_pd_returns_tuple(self, plotter, sample_positions):
        """compute_pd should return (H0, H1) tuple."""
        H0, H1 = plotter.compute_pd(sample_positions)
        assert isinstance(H0, np.ndarray)
        assert isinstance(H1, np.ndarray)

    def test_compute_pd_shape(self, plotter, sample_positions):
        """Persistence diagrams should have shape (n, 2)."""
        H0, H1 = plotter.compute_pd(sample_positions)

        if len(H0) > 0:
            assert H0.shape[1] == 2
        if len(H1) > 0:
            assert H1.shape[1] == 2

    def test_compute_pd_birth_before_death(self, plotter, sample_positions):
        """Birth should always be before death."""
        H0, H1 = plotter.compute_pd(sample_positions)

        for H in [H0, H1]:
            if len(H) > 0:
                assert np.all(H[:, 0] <= H[:, 1])

    def test_compute_pd_two_clusters(self, plotter, rng):
        """Two well-separated clusters should have one long-lived H0 feature."""
        # Two widely separated clusters
        cluster1 = rng.normal(loc=[-10, 0], scale=0.1, size=(50, 2))
        cluster2 = rng.normal(loc=[10, 0], scale=0.1, size=(50, 2))
        points = np.vstack([cluster1, cluster2])

        H0, _ = plotter.compute_pd(points, max_edge_length=25.0)

        # Should have at least one long-lived H0 feature
        if len(H0) > 0:
            lifetimes = H0[:, 1] - H0[:, 0]
            assert np.max(lifetimes) > 5.0  # Long gap between clusters

    def test_compute_pd_single_cluster(self, plotter, rng):
        """Single tight cluster should have mostly short-lived H0."""
        cluster = rng.normal(loc=[0, 0], scale=0.1, size=(100, 2))

        H0, _ = plotter.compute_pd(cluster, max_edge_length=1.0)

        # All H0 features should have short lifetimes
        if len(H0) > 0:
            lifetimes = H0[:, 1] - H0[:, 0]
            assert np.max(lifetimes) < 0.5  # Tight cluster

    def test_compute_pd_ring_has_h1(self, plotter):
        """Ring/circle should have H1 feature (loop)."""
        # Create points on a circle
        n = 50
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        points = np.column_stack([np.cos(theta), np.sin(theta)])

        _, H1 = plotter.compute_pd(points, max_edge_length=2.0)

        # Should detect at least one loop
        assert len(H1) > 0

    def test_empty_input(self, plotter):
        """Empty point cloud should return empty diagrams."""
        empty = np.array([]).reshape(0, 2)
        H0, H1 = plotter.compute_pd(empty)

        assert len(H0) == 0
        assert len(H1) == 0

    def test_single_point(self, plotter):
        """Single point should have minimal persistence."""
        single = np.array([[0.0, 0.0]])
        H0, H1 = plotter.compute_pd(single)

        assert len(H1) == 0  # No loops possible
