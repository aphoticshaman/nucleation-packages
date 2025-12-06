"""
ECAP - Entangled Co-Adaptive Protocol
======================================

A real-time bi-directional adaptation system where human and AI agents develop
quantum-correlated feature spaces through immune-inspired pattern recognition
and homeostasis-maintained interaction dynamics.

Core Innovation:
- Treats human-AI interaction as a thermodynamic system
- Uses immune system T-cell analogy for pattern recognition
- Quantum fidelity as semantic alignment metric
- PID homeostasis for cognitive load management

The Key Equation:
    dΨ/dt = η(C_target - C(H,A)) - λ||∇_H Load||²

Translation: "Learn from the human without changing them."

Author: P.R.O.M.E.T.H.E.U.S. Protocol
License: Apache 2.0
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, field
from scipy.special import softmax
from collections import deque
import hashlib
import json


@dataclass
class CognitiveState:
    """
    Representation of human cognitive state.

    In practice, these are inferred from proxies:
    - feature_vector: Derived from interaction patterns
    - attention_pattern: Eye tracking, mouse hover, time on elements
    - cognitive_load: Response latency, error rate, backtrack frequency
    - entropy: Diversity of actions, exploration vs exploitation
    """
    feature_vector: np.ndarray  # Extracted features [n_features]
    attention_pattern: np.ndarray  # What human is focusing on [n_elements]
    cognitive_load: float  # 0-1 scale (0=relaxed, 1=overwhelmed)
    entropy: float  # Information content of current thought stream
    timestamp: float = 0.0  # Unix timestamp

    def to_dict(self) -> dict:
        return {
            'feature_vector': self.feature_vector.tolist(),
            'attention_pattern': self.attention_pattern.tolist(),
            'cognitive_load': self.cognitive_load,
            'entropy': self.entropy,
            'timestamp': self.timestamp
        }


@dataclass
class AIState:
    """Representation of AI model state"""
    params_hash: str  # Hash of current parameters (for change detection)
    embedding_matrix: np.ndarray  # Current semantic space [vocab x embed_dim]
    adaptation_rate: float = 1.0  # Current learning rate multiplier
    generation: int = 0  # Adaptation generation counter

    def to_dict(self) -> dict:
        return {
            'params_hash': self.params_hash,
            'embedding_mean': float(self.embedding_matrix.mean()),
            'embedding_std': float(self.embedding_matrix.std()),
            'adaptation_rate': self.adaptation_rate,
            'generation': self.generation
        }


@dataclass
class InteractionPattern:
    """A stored pattern from immune memory"""
    pattern_id: str
    human_features: np.ndarray
    ai_embedding_summary: np.ndarray
    reward: float
    timestamp: float
    access_count: int = 0  # For LRU-like prioritization


class ImmuneMemory:
    """
    T-cell inspired memory of productive collaboration states.

    Concepts:
    - "Self" patterns: Productive interactions to preserve
    - "Non-self" patterns: Unproductive states to avoid
    - Negative selection: Remove patterns that lead to poor outcomes
    - Clonal expansion: Successful patterns get reinforced
    """

    def __init__(self,
                 capacity: int = 1000,
                 self_threshold: float = 0.7,
                 decay_rate: float = 0.995):
        self.capacity = capacity
        self.self_threshold = self_threshold
        self.decay_rate = decay_rate

        self.memory: deque = deque(maxlen=capacity)
        self.self_patterns: Dict[str, float] = {}  # pattern_id -> confidence
        self.non_self_patterns: set = set()

        # Statistics
        self.stats = {
            'total_interactions': 0,
            'self_classifications': 0,
            'non_self_classifications': 0,
            'pattern_retrievals': 0
        }

    def add_interaction(self,
                       human_state: CognitiveState,
                       ai_state: AIState,
                       reward: float,
                       timestamp: float = 0.0) -> str:
        """
        Store interaction pattern with reward feedback.

        Args:
            human_state: Current human cognitive state
            ai_state: Current AI model state
            reward: Interaction quality score [0, 1]
            timestamp: When interaction occurred

        Returns:
            pattern_id: Unique identifier for this pattern
        """
        pattern_id = self._create_pattern_id(human_state, ai_state)
        self.stats['total_interactions'] += 1

        # Create pattern
        pattern = InteractionPattern(
            pattern_id=pattern_id,
            human_features=human_state.feature_vector.copy(),
            ai_embedding_summary=ai_state.embedding_matrix.mean(axis=0) if ai_state.embedding_matrix.ndim > 1 else ai_state.embedding_matrix.copy(),
            reward=reward,
            timestamp=timestamp
        )

        # Classify as self or non-self
        if reward >= self.self_threshold:
            # Productive interaction - add to self patterns
            if pattern_id in self.self_patterns:
                # Clonal expansion: reinforce existing pattern
                self.self_patterns[pattern_id] = min(
                    1.0,
                    self.self_patterns[pattern_id] + 0.1 * reward
                )
            else:
                self.self_patterns[pattern_id] = reward
            self.stats['self_classifications'] += 1

            # Remove from non-self if present
            self.non_self_patterns.discard(pattern_id)
        else:
            # Unproductive - mark as non-self (immune rejection)
            self.non_self_patterns.add(pattern_id)
            self.stats['non_self_classifications'] += 1

            # Decay self confidence if pattern was previously self
            if pattern_id in self.self_patterns:
                self.self_patterns[pattern_id] *= 0.5
                if self.self_patterns[pattern_id] < 0.1:
                    del self.self_patterns[pattern_id]

        # Store in memory
        self.memory.append(pattern)

        return pattern_id

    def is_self_pattern(self,
                       human_state: CognitiveState,
                       ai_state: AIState) -> Tuple[bool, float]:
        """
        Check if current state matches known productive patterns.

        Returns:
            (is_self, confidence): Boolean and confidence score
        """
        pattern_id = self._create_pattern_id(human_state, ai_state)
        self.stats['pattern_retrievals'] += 1

        if pattern_id in self.non_self_patterns:
            return False, 0.0

        if pattern_id in self.self_patterns:
            return True, self.self_patterns[pattern_id]

        # Fuzzy matching: find similar patterns
        similarity, best_match = self._find_similar_pattern(human_state, ai_state)
        if similarity > 0.8 and best_match in self.self_patterns:
            return True, similarity * self.self_patterns[best_match]

        return False, 0.5  # Unknown pattern - neutral

    def get_pattern_features(self, pattern_id: str) -> Optional[np.ndarray]:
        """Retrieve stored features for a pattern"""
        for pattern in self.memory:
            if pattern.pattern_id == pattern_id:
                pattern.access_count += 1
                return np.concatenate([
                    pattern.human_features,
                    pattern.ai_embedding_summary
                ])
        return None

    def decay_memories(self):
        """Apply temporal decay to pattern confidences"""
        for pattern_id in list(self.self_patterns.keys()):
            self.self_patterns[pattern_id] *= self.decay_rate
            if self.self_patterns[pattern_id] < 0.01:
                del self.self_patterns[pattern_id]

    def _create_pattern_id(self,
                          human: CognitiveState,
                          ai: AIState) -> str:
        """Create unique fingerprint for interaction pattern"""
        # Quantize features for stable hashing
        h_quantized = np.round(human.feature_vector * 10).astype(int)
        a_summary = ai.embedding_matrix.mean() if ai.embedding_matrix.ndim > 1 else ai.embedding_matrix.mean()

        combined = np.concatenate([
            h_quantized.flatten()[:50],  # First 50 human features
            [int(human.cognitive_load * 10)],
            [int(a_summary * 1000)]
        ])

        return hashlib.sha256(combined.tobytes()).hexdigest()[:16]

    def _find_similar_pattern(self,
                             human: CognitiveState,
                             ai: AIState) -> Tuple[float, str]:
        """Find most similar stored pattern"""
        best_similarity = 0.0
        best_id = ""

        for pattern in self.memory:
            # Cosine similarity on human features
            h_sim = np.dot(human.feature_vector, pattern.human_features)
            h_sim /= (np.linalg.norm(human.feature_vector) * np.linalg.norm(pattern.human_features) + 1e-8)

            if h_sim > best_similarity:
                best_similarity = h_sim
                best_id = pattern.pattern_id

        return float(best_similarity), best_id

    def get_stats(self) -> dict:
        return {
            **self.stats,
            'self_pattern_count': len(self.self_patterns),
            'non_self_pattern_count': len(self.non_self_patterns),
            'memory_size': len(self.memory)
        }


class EntanglementMonitor:
    """
    Quantum-inspired correlation measurement between human and AI states.

    Key Concept:
    - Treat human features and AI embeddings as quantum states
    - Measure "fidelity" (overlap) between normalized state vectors
    - Higher fidelity = better semantic alignment
    - This is more principled than raw cosine similarity
    """

    def __init__(self,
                 feature_dim: int = 128,
                 history_length: int = 50):
        self.feature_dim = feature_dim
        self.history_length = history_length

        # Rolling history for temporal correlation
        self.human_history: deque = deque(maxlen=history_length)
        self.ai_history: deque = deque(maxlen=history_length)
        self.correlation_history: deque = deque(maxlen=history_length)

    def compute_correlation(self,
                          human_features: np.ndarray,
                          ai_embeddings: np.ndarray) -> float:
        """
        Compute entanglement correlation C(H, A).

        Uses quantum-inspired fidelity measure:
        C = |⟨ψ_H|ψ_A⟩|² where states are normalized

        Args:
            human_features: Human cognitive feature vector
            ai_embeddings: AI embedding matrix or vector

        Returns:
            Correlation score in [0, 1]
        """
        # Flatten AI embeddings if matrix
        if ai_embeddings.ndim > 1:
            ai_vec = ai_embeddings.mean(axis=0)
        else:
            ai_vec = ai_embeddings

        # Project to same dimension if needed
        if len(human_features) != len(ai_vec):
            # Use random projection for dimension matching
            min_dim = min(len(human_features), len(ai_vec))
            human_features = human_features[:min_dim]
            ai_vec = ai_vec[:min_dim]

        # Normalize to quantum states (unit vectors)
        psi_h = self._normalize_state(human_features)
        psi_a = self._normalize_state(ai_vec)

        # Compute fidelity (squared overlap)
        fidelity = np.abs(np.dot(psi_h.conj(), psi_a))**2

        # Apply correction for high-dimensional spaces
        # (random vectors in high-D have ~0 overlap)
        dim_correction = 1 - (1 / np.sqrt(len(psi_h) + 1))

        correlation = float(fidelity * dim_correction)

        # Store in history
        self.human_history.append(human_features.copy())
        self.ai_history.append(ai_vec.copy())
        self.correlation_history.append(correlation)

        return correlation

    def compute_temporal_correlation(self, lag: int = 1) -> float:
        """
        Compute time-lagged correlation.

        This detects if human state changes predict AI state changes
        (or vice versa) with some delay.
        """
        if len(self.human_history) < lag + 2:
            return 0.5  # Not enough data

        # Get lagged pairs
        h_current = list(self.human_history)[-1]
        a_lagged = list(self.ai_history)[-(lag + 1)]

        return self.compute_correlation(h_current, a_lagged)

    def get_correlation_trend(self) -> float:
        """
        Return slope of correlation over recent history.
        Positive = improving alignment, Negative = drifting apart
        """
        if len(self.correlation_history) < 5:
            return 0.0

        recent = list(self.correlation_history)[-10:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]

        return float(slope)

    def _normalize_state(self, vec: np.ndarray) -> np.ndarray:
        """Normalize vector to unit norm (quantum state preparation)"""
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return np.zeros_like(vec)
        return vec / norm

    def get_stats(self) -> dict:
        if len(self.correlation_history) == 0:
            return {'avg_correlation': 0.0, 'trend': 0.0}

        return {
            'avg_correlation': float(np.mean(self.correlation_history)),
            'std_correlation': float(np.std(self.correlation_history)),
            'trend': self.get_correlation_trend(),
            'history_length': len(self.correlation_history)
        }


class HomeostasisController:
    """
    PID-like controller for maintaining cognitive balance.

    Goal: Maximize correlation while keeping cognitive load bounded.

    The control law:
        adaptation_rate = f(correlation_error, load_error)

    Where:
        - correlation_error = target_correlation - current_correlation
        - load_error = current_load - target_load (positive = overloaded)
    """

    def __init__(self,
                 target_correlation: float = 0.85,
                 target_load: float = 0.6,
                 Kp: float = 0.8,
                 Ki: float = 0.2,
                 Kd: float = 0.1,
                 adaptation_bounds: Tuple[float, float] = (0.01, 1.0)):
        self.target_correlation = target_correlation
        self.target_load = target_load

        # PID coefficients
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.adaptation_bounds = adaptation_bounds

        # Controller state
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_load = 0.5
        self.adaptation_rate = 1.0

        # History for analysis
        self.control_history: List[dict] = []

    def update(self,
               current_correlation: float,
               current_load: float,
               delta_time: float = 1.0) -> float:
        """
        Update adaptation rate based on system state.

        The key insight: We want to INCREASE adaptation when correlation
        is LOW but load is also LOW (room to push). We want to DECREASE
        adaptation when load is HIGH (protect the human).

        Returns:
            New adaptation rate in [adaptation_bounds]
        """
        # Compute errors
        correlation_error = self.target_correlation - current_correlation  # Positive = need more
        load_error = current_load - self.target_load  # Positive = overloaded

        # Load rate of change (for derivative term)
        load_derivative = (current_load - self.prev_load) / max(delta_time, 0.001)
        self.prev_load = current_load

        # Composite error: balance correlation and load
        # Prioritize avoiding overload (0.7 weight) over maximizing correlation (0.3)
        error = (0.3 * correlation_error) - (0.7 * load_error)

        # PID computation
        P = self.Kp * error
        self.integral_error = np.clip(
            self.integral_error + error * delta_time,
            -2.0, 2.0  # Anti-windup
        )
        I = self.Ki * self.integral_error
        D = self.Kd * (error - self.prev_error) / max(delta_time, 0.001)

        self.prev_error = error

        # Add load derivative term (anticipatory control)
        # If load is increasing rapidly, preemptively slow down
        anticipation = -0.5 * np.tanh(load_derivative * 2)

        # Compute control signal
        control = P + I + D + anticipation

        # Update adaptation rate
        # Positive control → increase rate, Negative → decrease
        rate_change = 0.1 * np.tanh(control)
        new_rate = self.adaptation_rate * (1.0 + rate_change)

        # Clamp to bounds
        self.adaptation_rate = np.clip(
            new_rate,
            self.adaptation_bounds[0],
            self.adaptation_bounds[1]
        )

        # Log for analysis
        self.control_history.append({
            'correlation': current_correlation,
            'load': current_load,
            'error': error,
            'control': control,
            'adaptation_rate': self.adaptation_rate
        })

        return self.adaptation_rate

    def get_stats(self) -> dict:
        if len(self.control_history) == 0:
            return {}

        recent = self.control_history[-20:]
        return {
            'current_rate': self.adaptation_rate,
            'avg_correlation': np.mean([h['correlation'] for h in recent]),
            'avg_load': np.mean([h['load'] for h in recent]),
            'integral_error': self.integral_error
        }

    def reset(self):
        """Reset controller state"""
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_load = 0.5
        self.adaptation_rate = 1.0


class AdaptationKernel:
    """
    Computes optimal adaptation parameters from human-AI state pair.

    This is a simplified version using linear transformations.
    Production version would use a neural network.

    Outputs:
    - learning_rate: How fast to update AI
    - gradient_scale: Per-parameter learning rate scaling
    - attention_mask: Which parts of AI to prioritize updating
    """

    def __init__(self,
                 human_dim: int = 128,
                 ai_dim: int = 256,
                 output_dim: int = 128):
        self.human_dim = human_dim
        self.ai_dim = ai_dim
        self.output_dim = output_dim

        # Linear projections (would be neural network in production)
        np.random.seed(42)  # Reproducibility
        self.W_encode = np.random.randn(human_dim + ai_dim, 256) * 0.1
        self.W_lr = np.random.randn(256, 1) * 0.1
        self.W_scale = np.random.randn(256, output_dim) * 0.1
        self.W_mask = np.random.randn(256, output_dim) * 0.1

        # Immune regulation weights
        self.W_immune = np.random.randn(256 + 64, 128) * 0.1

    def compute_adaptations(self,
                           human_state: CognitiveState,
                           ai_state: AIState,
                           pattern_features: Optional[np.ndarray] = None,
                           immune_confidence: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Compute adaptation parameters.

        Args:
            human_state: Current human cognitive state
            ai_state: Current AI model state
            pattern_features: Optional features from immune memory
            immune_confidence: How confident we are this is a "self" pattern

        Returns:
            Dict with:
            - learning_rate: Scalar learning rate [0, 1]
            - gradient_scale: Per-dimension scaling [output_dim]
            - attention_mask: Which dimensions to focus on [output_dim]
        """
        # Prepare inputs
        h_vec = human_state.feature_vector
        if ai_state.embedding_matrix.ndim > 1:
            a_vec = ai_state.embedding_matrix.mean(axis=0)
        else:
            a_vec = ai_state.embedding_matrix

        # Pad/truncate to expected dimensions
        h_vec = self._pad_or_truncate(h_vec, self.human_dim)
        a_vec = self._pad_or_truncate(a_vec, self.ai_dim)

        # Concatenate and encode
        combined = np.concatenate([h_vec, a_vec])
        encoded = np.tanh(combined @ self.W_encode)

        # Compute outputs
        learning_rate = float(self._sigmoid(encoded @ self.W_lr))
        gradient_scale = self._sigmoid(encoded @ self.W_scale)
        attention_mask = self._sigmoid(encoded @ self.W_mask)

        # Apply immune regulation if pattern features available
        if pattern_features is not None:
            pattern_features = self._pad_or_truncate(pattern_features, 64)
            immune_input = np.concatenate([encoded, pattern_features])
            regulation = self._sigmoid(immune_input @ self.W_immune).mean()
        else:
            # Use immune confidence directly
            regulation = 0.5 + 0.5 * immune_confidence

        # Scale outputs by regulation factor
        learning_rate *= regulation
        gradient_scale *= regulation

        return {
            'learning_rate': np.array([learning_rate]),
            'gradient_scale': gradient_scale,
            'attention_mask': attention_mask,
            'regulation': np.array([regulation])
        }

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

    def _pad_or_truncate(self, vec: np.ndarray, target_len: int) -> np.ndarray:
        if len(vec) >= target_len:
            return vec[:target_len]
        return np.pad(vec, (0, target_len - len(vec)))


class ECAPEngine:
    """
    Main ECAP Engine coordinating all components.

    The Entangled Co-Adaptive Protocol:
    1. Sample human cognitive state
    2. Check immune memory for pattern recognition
    3. Compute entanglement correlation
    4. Update homeostasis controller
    5. Generate adaptation parameters
    6. Store interaction in memory

    The core invariant:
        dΨ/dt = η(C_target - C(H,A)) - λ||∇_H Load||²

    "Increase mutual information while minimizing cognitive load gradient"
    """

    def __init__(self,
                 human_dim: int = 128,
                 ai_dim: int = 256,
                 pattern_dim: int = 64,
                 target_correlation: float = 0.85,
                 target_load: float = 0.6):
        self.human_dim = human_dim
        self.ai_dim = ai_dim

        # Initialize components
        self.immune_memory = ImmuneMemory(capacity=5000)
        self.entanglement_monitor = EntanglementMonitor(feature_dim=human_dim)
        self.adaptation_kernel = AdaptationKernel(
            human_dim=human_dim,
            ai_dim=ai_dim,
            output_dim=ai_dim
        )
        self.homeostasis = HomeostasisController(
            target_correlation=target_correlation,
            target_load=target_load
        )

        # Engine state
        self.generation = 0
        self.cumulative_reward = 0.0
        self.interaction_count = 0

        # Performance metrics
        self.metrics = {
            'avg_correlation': 0.0,
            'avg_load': 0.0,
            'avg_reward': 0.0,
            'adaptation_count': 0,
            'immune_hits': 0,
            'self_pattern_ratio': 0.0
        }

    def process_interaction(self,
                          human_state: CognitiveState,
                          ai_state: AIState,
                          external_reward: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Process a single human-AI interaction.

        Args:
            human_state: Current human cognitive state
            ai_state: Current AI model state
            external_reward: Optional external reward signal [0, 1]

        Returns:
            Adaptation parameters for AI update:
            - learning_rate: Scalar
            - gradient_scale: Per-parameter scaling
            - attention_mask: Focus areas
            - correlation: Current H-A correlation
            - adaptation_rate: From homeostasis controller
            - is_self_pattern: Whether pattern matches immune memory
        """
        self.interaction_count += 1

        # Step 1: Check immune memory
        is_self, immune_confidence = self.immune_memory.is_self_pattern(
            human_state, ai_state
        )
        if is_self:
            self.metrics['immune_hits'] += 1

        # Get pattern features if available
        pattern_id = self.immune_memory._create_pattern_id(human_state, ai_state)
        pattern_features = self.immune_memory.get_pattern_features(pattern_id)

        # Step 2: Compute entanglement correlation
        correlation = self.entanglement_monitor.compute_correlation(
            human_state.feature_vector,
            ai_state.embedding_matrix
        )

        # Step 3: Update homeostasis controller
        adaptation_rate = self.homeostasis.update(
            current_correlation=correlation,
            current_load=human_state.cognitive_load
        )

        # Step 4: Compute adaptation parameters
        adaptations = self.adaptation_kernel.compute_adaptations(
            human_state,
            ai_state,
            pattern_features,
            immune_confidence
        )

        # Scale by homeostasis adaptation rate
        adaptations['learning_rate'] *= adaptation_rate
        adaptations['gradient_scale'] *= adaptation_rate

        # Step 5: Compute reward and store in memory
        if external_reward is not None:
            reward = external_reward
        else:
            # Intrinsic reward: correlation * (1 - load) * (1 - load_increase)
            load_penalty = human_state.cognitive_load
            reward = correlation * (1 - load_penalty)

        self.immune_memory.add_interaction(
            human_state, ai_state, reward, human_state.timestamp
        )

        # Update metrics (exponential moving average)
        alpha = 0.1
        self.metrics['avg_correlation'] = (
            (1 - alpha) * self.metrics['avg_correlation'] + alpha * correlation
        )
        self.metrics['avg_load'] = (
            (1 - alpha) * self.metrics['avg_load'] + alpha * human_state.cognitive_load
        )
        self.metrics['avg_reward'] = (
            (1 - alpha) * self.metrics['avg_reward'] + alpha * reward
        )
        self.metrics['adaptation_count'] += 1

        immune_stats = self.immune_memory.get_stats()
        if immune_stats['total_interactions'] > 0:
            self.metrics['self_pattern_ratio'] = (
                immune_stats['self_classifications'] /
                immune_stats['total_interactions']
            )

        self.cumulative_reward += reward
        self.generation += 1

        # Return adaptation parameters
        return {
            **adaptations,
            'correlation': np.array([correlation]),
            'adaptation_rate': np.array([adaptation_rate]),
            'reward': np.array([reward]),
            'is_self_pattern': np.array([float(is_self)]),
            'immune_confidence': np.array([immune_confidence])
        }

    def decay_memories(self):
        """Apply periodic memory decay"""
        self.immune_memory.decay_memories()

    def get_metrics(self) -> dict:
        """Return current performance metrics"""
        return {
            **self.metrics,
            'generation': self.generation,
            'cumulative_reward': self.cumulative_reward,
            'interaction_count': self.interaction_count,
            'entanglement': self.entanglement_monitor.get_stats(),
            'homeostasis': self.homeostasis.get_stats(),
            'immune': self.immune_memory.get_stats()
        }

    def reset(self):
        """Reset engine state (keeps learned patterns)"""
        self.generation = 0
        self.cumulative_reward = 0.0
        self.interaction_count = 0
        self.homeostasis.reset()


def simulate_ecap_session(n_steps: int = 100, verbose: bool = True):
    """
    Simulate a human-AI co-adaptation session.

    This demonstrates the ECAP protocol with synthetic data.
    In production, CognitiveState would come from actual user telemetry.
    """

    if verbose:
        print("=" * 60)
        print("ECAP Protocol Simulation")
        print("=" * 60)

    # Initialize engine
    engine = ECAPEngine(
        human_dim=128,
        ai_dim=256,
        target_correlation=0.8,
        target_load=0.5
    )

    results = []

    for i in range(n_steps):
        # Generate synthetic human state with temporal structure
        # Simulates a human who starts fresh, gets tired, then recovers
        fatigue_cycle = np.sin(i * np.pi / 30)  # 60-step fatigue cycle

        human_state = CognitiveState(
            feature_vector=np.random.randn(128) * 0.3 + np.sin(np.arange(128) * i / 50) * 0.5,
            attention_pattern=softmax(np.random.randn(10) + np.arange(10) * 0.1),
            cognitive_load=np.clip(0.3 + 0.3 * fatigue_cycle + np.random.randn() * 0.1, 0, 1),
            entropy=np.random.uniform(0.5, 2.0),
            timestamp=float(i)
        )

        # Generate AI state (slowly adapting)
        ai_drift = np.sin(np.arange(256) * i / 100) * 0.1
        ai_state = AIState(
            params_hash=f"model_gen_{i}",
            embedding_matrix=np.random.randn(256, 128) * 0.1 + ai_drift.reshape(-1, 1),
            adaptation_rate=1.0,
            generation=i
        )

        # Process interaction
        adaptations = engine.process_interaction(human_state, ai_state)

        results.append({
            'step': i,
            'correlation': float(adaptations['correlation'][0]),
            'adaptation_rate': float(adaptations['adaptation_rate'][0]),
            'reward': float(adaptations['reward'][0]),
            'cognitive_load': human_state.cognitive_load,
            'is_self': bool(adaptations['is_self_pattern'][0] > 0.5)
        })

        # Periodic decay
        if i % 20 == 0:
            engine.decay_memories()

        # Logging
        if verbose and i % 10 == 0:
            metrics = engine.get_metrics()
            print(f"Step {i:3d}: "
                  f"Corr={adaptations['correlation'][0]:.3f}, "
                  f"Load={human_state.cognitive_load:.2f}, "
                  f"Rate={adaptations['adaptation_rate'][0]:.3f}, "
                  f"Self={metrics['self_pattern_ratio']:.2f}")

    if verbose:
        print("\n" + "=" * 60)
        print("Simulation Complete")
        print("=" * 60)

        final_metrics = engine.get_metrics()
        print(f"\nFinal Metrics:")
        print(f"  Average Correlation: {final_metrics['avg_correlation']:.3f}")
        print(f"  Average Load: {final_metrics['avg_load']:.3f}")
        print(f"  Average Reward: {final_metrics['avg_reward']:.3f}")
        print(f"  Immune Hits: {final_metrics['immune_hits']}")
        print(f"  Self Pattern Ratio: {final_metrics['self_pattern_ratio']:.3f}")
        print(f"  Cumulative Reward: {final_metrics['cumulative_reward']:.2f}")

    return engine, results


if __name__ == "__main__":
    engine, results = simulate_ecap_session(n_steps=100, verbose=True)
