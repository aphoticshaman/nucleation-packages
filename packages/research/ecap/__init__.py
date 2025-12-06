"""
ECAP - Entangled Co-Adaptive Protocol
======================================

A real-time bi-directional adaptation system for human-AI co-evolution.

Core Components:
- CognitiveState: Human cognitive feature representation
- AIState: AI model state representation
- ImmuneMemory: T-cell inspired pattern memory
- EntanglementMonitor: Quantum-inspired correlation measurement
- HomeostasisController: PID cognitive load regulation
- AdaptationKernel: Optimal adaptation parameter computation
- ECAPEngine: Main coordination engine

Example:
    from ecap import ECAPEngine, CognitiveState, AIState

    engine = ECAPEngine()

    human = CognitiveState(
        feature_vector=np.random.randn(128),
        attention_pattern=np.ones(10) / 10,
        cognitive_load=0.5,
        entropy=1.0
    )

    ai = AIState(
        params_hash="model_v1",
        embedding_matrix=np.random.randn(256, 128)
    )

    adaptations = engine.process_interaction(human, ai)
    print(f"Correlation: {adaptations['correlation'][0]:.3f}")
"""

from .ecap_protocol import (
    CognitiveState,
    AIState,
    InteractionPattern,
    ImmuneMemory,
    EntanglementMonitor,
    HomeostasisController,
    AdaptationKernel,
    ECAPEngine,
    simulate_ecap_session,
)

__all__ = [
    'CognitiveState',
    'AIState',
    'InteractionPattern',
    'ImmuneMemory',
    'EntanglementMonitor',
    'HomeostasisController',
    'AdaptationKernel',
    'ECAPEngine',
    'simulate_ecap_session',
]

__version__ = '1.0.0'
