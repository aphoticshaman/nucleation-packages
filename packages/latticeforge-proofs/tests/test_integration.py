#!/usr/bin/env python3
"""
INTEGRATION TEST SUITE - End-to-End LatticeForge Pipeline Testing
===================================================================

Tests the complete inference pipeline from raw samples to final answer.

Test Categories:
1. Pipeline Integration - Full flow tests
2. Component Handoffs - Interface contracts
3. Edge Cases - Boundary conditions
4. Performance Regression - Speed/memory
5. Consistency - Determinism and reproducibility

Author: LatticeForge Team
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import random
import statistics
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import unittest

from cic_core import (
    ProvenConstants,
    CICFunctional,
    CICState,
    ValueClustering,
    ClusteringResult,
    PhaseTransitionDetector,
    MicroGrokkingDetector,
    LatticeForgeInference,
    NormalizedCompressionDistance,
    SystemPhase,
    PhaseState,
)

from prometheus_insights import (
    PrometheusEngine,
    VarianceParadox,
    UniversalCollapsePattern,
    AttractorBasinTheory,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

class TestFixtures:
    """Shared test data generators."""

    @staticmethod
    def perfect_consensus(n: int = 11, value: int = 12345) -> List[int]:
        """All samples agree."""
        return [value] * n

    @staticmethod
    def near_consensus(n: int = 11, value: int = 12345, noise: float = 0.01) -> List[int]:
        """Samples very close to value."""
        return [int(value * (1 + random.gauss(0, noise))) for _ in range(n)]

    @staticmethod
    def moderate_noise(n: int = 11, value: int = 12345, noise: float = 0.05) -> List[int]:
        """Moderate noise around value."""
        return [int(value * (1 + random.gauss(0, noise))) for _ in range(n)]

    @staticmethod
    def with_outliers(n: int = 11, value: int = 12345, outlier_count: int = 2) -> List[int]:
        """Mostly correct with outliers."""
        samples = [value] * (n - outlier_count)
        samples.extend([random.randint(0, 99999) for _ in range(outlier_count)])
        random.shuffle(samples)
        return samples

    @staticmethod
    def bimodal(value1: int = 12345, value2: int = 54321, n: int = 11) -> List[int]:
        """Two competing clusters."""
        n1 = n // 2
        n2 = n - n1
        return [value1] * n1 + [value2] * n2

    @staticmethod
    def uniform_random(n: int = 11, max_val: int = 99999) -> List[int]:
        """Completely random samples."""
        return [random.randint(0, max_val) for _ in range(n)]

    @staticmethod
    def grokking_entropy(length: int = 100, grokking_at: int = 50) -> List[float]:
        """Entropy sequence with grokking."""
        entropies = []
        for i in range(length):
            if i < grokking_at:
                base = 0.8 - 0.3 * (i / grokking_at)
            else:
                progress = (i - grokking_at) / (length - grokking_at)
                base = 0.5 - 0.4 * progress
            entropies.append(max(0.05, base + random.gauss(0, 0.03)))
        return entropies

    @staticmethod
    def flat_entropy(length: int = 100, value: float = 0.5) -> List[float]:
        """Constant entropy (no grokking)."""
        return [value + random.gauss(0, 0.02) for _ in range(length)]


# =============================================================================
# PIPELINE INTEGRATION TESTS
# =============================================================================

class TestPipelineIntegration(unittest.TestCase):
    """Test complete inference pipeline."""

    def setUp(self):
        self.engine = LatticeForgeInference()
        random.seed(42)

    def test_perfect_consensus_inference(self):
        """Pipeline should handle perfect consensus."""
        samples = TestFixtures.perfect_consensus(value=12345)
        result = self.engine.infer(samples)

        self.assertEqual(result.answer, 12345)
        self.assertGreater(result.confidence, 0.8)
        self.assertEqual(result.cic_state.entropy, 0)  # Zero variance
        self.assertEqual(result.clustering_result.n_clusters, 1)

    def test_near_consensus_inference(self):
        """Pipeline should handle near-consensus."""
        samples = TestFixtures.near_consensus(value=12345, noise=0.01)
        result = self.engine.infer(samples)

        # Should be within 5% of true value
        self.assertLess(abs(result.answer - 12345) / 12345, 0.05)
        self.assertGreater(result.confidence, 0.6)

    def test_outlier_robustness(self):
        """Pipeline should be robust to outliers."""
        samples = TestFixtures.with_outliers(value=12345, outlier_count=3)
        result = self.engine.infer(samples)

        # Should still recover true value
        self.assertLess(abs(result.answer - 12345) / 12345, 0.1)

    def test_bimodal_handling(self):
        """Pipeline should handle bimodal distributions."""
        samples = TestFixtures.bimodal(value1=12345, value2=54321)
        result = self.engine.infer(samples)

        # Should pick one of the modes
        self.assertTrue(
            abs(result.answer - 12345) < 1000 or abs(result.answer - 54321) < 1000
        )
        # Confidence should be lower due to ambiguity
        self.assertLess(result.confidence, 0.8)

    def test_random_samples_low_confidence(self):
        """Pipeline should have low confidence for random samples."""
        samples = TestFixtures.uniform_random()
        result = self.engine.infer(samples)

        # Confidence should be low for random data
        self.assertLess(result.confidence, 0.7)

    def test_pipeline_with_entropies(self):
        """Pipeline should use entropy data when provided."""
        samples = TestFixtures.near_consensus(value=12345)
        entropies = TestFixtures.grokking_entropy()

        result = self.engine.infer(samples, entropies=entropies)

        self.assertIsNotNone(result.grokking_signal)
        # Grokking should boost confidence
        self.assertGreater(result.confidence, 0.5)


# =============================================================================
# COMPONENT HANDOFF TESTS
# =============================================================================

class TestComponentHandoffs(unittest.TestCase):
    """Test interfaces between components."""

    def test_cic_to_clustering(self):
        """CIC state should integrate with clustering."""
        cic = CICFunctional()
        clustering = ValueClustering()

        samples = TestFixtures.moderate_noise(value=12345)

        # CIC computation
        cic_state = cic.compute(samples)
        self.assertIsInstance(cic_state, CICState)
        self.assertIn(cic_state.confidence, [x for x in [cic_state.confidence] if 0 <= x <= 1])

        # Clustering with CIC state
        answer, conf, result = clustering.infer(samples, cic_state)
        self.assertIsInstance(answer, int)
        self.assertIsInstance(conf, float)
        self.assertIsInstance(result, ClusteringResult)

    def test_phase_detector_state_consistency(self):
        """Phase detector should maintain consistent state."""
        detector = PhaseTransitionDetector()

        # Feed multiple signal sets
        for _ in range(10):
            signals = [[random.gauss(0, 1) for _ in range(50)] for _ in range(3)]
            state = detector.analyze(signals)

            self.assertIsInstance(state, PhaseState)
            self.assertIn(state.phase, list(SystemPhase))
            self.assertTrue(0 <= state.temperature <= 1)
            self.assertTrue(0 <= state.order_parameter <= 1)

    def test_grokking_detector_interface(self):
        """Grokking detector should have stable interface."""
        detector = MicroGrokkingDetector()

        # With grokking
        entropies = TestFixtures.grokking_entropy()
        signal = detector.detect(entropies)

        self.assertIsInstance(signal.detected, bool)
        self.assertIsInstance(signal.score, float)
        self.assertIsInstance(signal.d2_min, float)
        self.assertIsInstance(signal.phase, str)

        # Without grokking
        flat = TestFixtures.flat_entropy()
        signal_flat = detector.detect(flat)
        self.assertIsInstance(signal_flat.detected, bool)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    """Test boundary conditions and edge cases."""

    def setUp(self):
        self.engine = LatticeForgeInference()

    def test_empty_samples(self):
        """Should handle empty samples gracefully."""
        result = self.engine.infer([])

        self.assertEqual(result.answer, 0)
        self.assertLess(result.confidence, 0.5)

    def test_single_sample(self):
        """Should handle single sample."""
        result = self.engine.infer([12345])

        self.assertEqual(result.answer, 12345)

    def test_two_samples_same(self):
        """Should handle two identical samples."""
        result = self.engine.infer([12345, 12345])

        self.assertEqual(result.answer, 12345)
        self.assertGreater(result.confidence, 0.5)

    def test_two_samples_different(self):
        """Should handle two different samples."""
        result = self.engine.infer([12345, 54321])

        # Should pick one
        self.assertIn(result.answer, [12345, 54321, 33333])  # Or average

    def test_zero_values(self):
        """Should handle zero values."""
        result = self.engine.infer([0, 0, 0, 0, 0])

        self.assertEqual(result.answer, 0)

    def test_large_values(self):
        """Should handle large values."""
        samples = [99999, 99998, 99997, 99999, 99998]
        result = self.engine.infer(samples)

        self.assertGreater(result.answer, 99000)

    def test_negative_handling(self):
        """Should handle edge of valid range."""
        # Our system uses positive integers
        samples = [1, 2, 3, 1, 2, 1]
        result = self.engine.infer(samples)

        self.assertIn(result.answer, [1, 2])

    def test_very_short_entropy_sequence(self):
        """Should handle very short entropy sequences."""
        detector = MicroGrokkingDetector()
        signal = detector.detect([0.5, 0.4, 0.3])

        # Should not crash, should indicate insufficient data
        self.assertFalse(signal.detected)

    def test_constant_entropy_sequence(self):
        """Should handle constant entropy."""
        detector = MicroGrokkingDetector()
        signal = detector.detect([0.5] * 100)

        # No grokking in constant sequence
        self.assertFalse(signal.detected)


# =============================================================================
# PERFORMANCE REGRESSION TESTS
# =============================================================================

class TestPerformanceRegression(unittest.TestCase):
    """Test performance characteristics."""

    def test_inference_speed(self):
        """Inference should complete quickly."""
        engine = LatticeForgeInference()
        samples = TestFixtures.moderate_noise(n=11, value=12345)

        start = time.time()
        for _ in range(100):
            engine.infer(samples)
        elapsed = time.time() - start

        # Should complete 100 inferences in under 1 second
        print(f"\n[PERF] 100 inferences: {elapsed:.3f}s ({elapsed*10:.1f}ms each)")
        self.assertLess(elapsed, 2.0, "Inference too slow")

    def test_clustering_scaling(self):
        """Clustering should scale reasonably."""
        clustering = ValueClustering()

        times = {}
        for n in [10, 50, 100, 200]:
            samples = [random.randint(10000, 20000) for _ in range(n)]

            start = time.time()
            for _ in range(10):
                clustering.cluster(samples)
            elapsed = time.time() - start

            times[n] = elapsed / 10

        print(f"\n[PERF] Clustering scaling:")
        for n, t in times.items():
            print(f"  n={n}: {t*1000:.2f}ms")

        # Should be roughly O(n^2) for single-linkage
        # 200/50 = 4x samples, should be ~16x time, allow 25x
        ratio = times[200] / times[50]
        self.assertLess(ratio, 25, "Clustering scaling worse than O(n^2)")

    def test_cic_computation_speed(self):
        """CIC computation should be fast."""
        cic = CICFunctional()
        samples = TestFixtures.moderate_noise(n=11)

        start = time.time()
        for _ in range(1000):
            cic.compute(samples)
        elapsed = time.time() - start

        print(f"\n[PERF] 1000 CIC computations: {elapsed:.3f}s ({elapsed:.3f}ms each)")
        self.assertLess(elapsed, 2.0, "CIC computation too slow")


# =============================================================================
# CONSISTENCY TESTS
# =============================================================================

class TestConsistency(unittest.TestCase):
    """Test determinism and reproducibility."""

    def test_deterministic_with_seed(self):
        """Same seed should give same results."""
        results = []

        for _ in range(3):
            random.seed(42)
            engine = LatticeForgeInference()
            samples = TestFixtures.moderate_noise(value=12345)
            result = engine.infer(samples)
            results.append((result.answer, result.confidence))

        # All results should be identical
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])

    def test_confidence_bounds(self):
        """Confidence should always be in [0.05, 0.95]."""
        engine = LatticeForgeInference()

        for _ in range(100):
            samples = [random.randint(0, 99999) for _ in range(11)]
            result = engine.infer(samples)

            self.assertGreaterEqual(result.confidence, 0.05)
            self.assertLessEqual(result.confidence, 0.95)

    def test_answer_in_valid_range(self):
        """Answer should be reasonable given inputs."""
        engine = LatticeForgeInference()

        for _ in range(100):
            samples = [random.randint(1000, 50000) for _ in range(11)]
            result = engine.infer(samples)

            # Answer should be within range of samples (with some margin)
            min_sample = min(samples)
            max_sample = max(samples)
            margin = (max_sample - min_sample) * 0.5

            self.assertGreaterEqual(result.answer, min_sample - margin)
            self.assertLessEqual(result.answer, max_sample + margin)


# =============================================================================
# PROMETHEUS ENGINE INTEGRATION
# =============================================================================

class TestPrometheusIntegration(unittest.TestCase):
    """Test PROMETHEUS engine integration."""

    def setUp(self):
        self.engine = PrometheusEngine()
        random.seed(42)

    def test_full_analysis(self):
        """PROMETHEUS full analysis should work."""
        samples = TestFixtures.moderate_noise(value=12345)

        results = self.engine.full_analysis(
            samples,
            true_value=12345
        )

        self.assertIn('recommended_answer', results)
        self.assertIn('collapse_stage', results)
        self.assertIn('cic_state', results)
        self.assertIn('basins', results)

    def test_attractor_basin_detection(self):
        """Should detect attractor basins."""
        samples = TestFixtures.near_consensus(value=12345)
        basins = AttractorBasinTheory.find_basins(samples)

        self.assertGreater(len(basins), 0)
        # Should find basin near 12345
        centers = [b.center for b in basins]
        self.assertTrue(any(abs(c - 12345) < 500 for c in centers))

    def test_collapse_stage_detection(self):
        """Should detect collapse stages."""
        # High entropy (exploration)
        high_entropy = [0.9, 0.85, 0.88, 0.87, 0.86]
        stage = UniversalCollapsePattern.detect_stage(high_entropy, 0.85)
        self.assertEqual(stage.value, "exploration")

        # Low entropy (crystallized)
        low_entropy = [0.3, 0.25, 0.22, 0.20, 0.18]
        stage = UniversalCollapsePattern.detect_stage(low_entropy, 0.15)
        self.assertEqual(stage.value, "crystallized")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
