"""
PROMETHEUS Research Module

Physics-inspired reasoning engines for LLM math problem solving.

Modules:
- prometheus_kaggle: Kaggle-ready engine with UIPT + DeltaK + toroidal voting
- prometheus_v6_full: Full heterogeneous reasoning orchestrator for production

Core Components:
- UIPT (Universal Information Phase Transition): Detect reasoning crystallization
- DeltaK: Bias estimator for sampling divergence
- MDL Meta-Scheduler: Adaptive temperature/top_p/top_k control
- Toroidal Clustering: S1 clustering for mod-N answers
- Entropic Gravity: Mass * Density^0.15 * Solomonoff selection

Usage:
    from packages.research.prometheus import (
        PrometheusKaggleEngine,
        SimpleSampler,
        build_submission
    )

    sampler = SimpleSampler(model_name="gpt2", device="cuda")
    engine = PrometheusKaggleEngine(sampler=sampler, candidates=64)
    result = engine.solve_one("Compute (n^2 + 7) mod 1000 for n = 42")
"""

from .prometheus_kaggle import (
    # Core Classes
    UIPTEntropyWindow,
    DeltaKScheduler,
    MDLMetaScheduler,
    SimpleSampler,
    PrometheusKaggleEngine,

    # Utilities
    canonicalize_answer,
    sympy_to_float_safe,
    toroidal_distance,
    cluster_basins_mod1000,
    entropic_gravity_select,
    build_submission,
    run_ablation,
)

from .prometheus_v6_full import (
    # Production Classes
    UIPTEntropyBatch,
    PrometheusOrchestrator,
    TrajectoryStore,

    # Utilities
    emit_dashboard_event,
    test_harness_simple,
)

__all__ = [
    # Kaggle Engine
    'UIPTEntropyWindow',
    'DeltaKScheduler',
    'MDLMetaScheduler',
    'SimpleSampler',
    'PrometheusKaggleEngine',
    'build_submission',
    'run_ablation',

    # Production Engine
    'UIPTEntropyBatch',
    'PrometheusOrchestrator',
    'TrajectoryStore',

    # Utilities
    'canonicalize_answer',
    'sympy_to_float_safe',
    'toroidal_distance',
    'cluster_basins_mod1000',
    'entropic_gravity_select',
    'emit_dashboard_event',
    'test_harness_simple',
]
