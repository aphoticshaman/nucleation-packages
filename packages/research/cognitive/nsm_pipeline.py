"""
NSM Pipeline: Neural State Machine with Multiple Architectural Options.

This module implements a comprehensive Neural State Machine pipeline with
2x architectural variants for each core component plus a hybrid mode.

Architecture Options:

STATE ENCODING:
  Option A - VariationalStateEncoder: VAE-based latent state with KL regularization
  Option B - ContrastiveStateEncoder: InfoNCE-based discriminative encoding
  Hybrid   - DualPathEncoder: Combines generative + discriminative objectives

TRANSITION MODELING:
  Option A - GRUTransition: Recurrent gated transition with forget dynamics
  Option B - TransformerTransition: Self-attention based state evolution
  Hybrid   - AttentiveGRU: GRU with cross-attention to observation history

OBSERVATION DECODING:
  Option A - ProbabilisticDecoder: Gaussian observation model with learned variance
  Option B - FlowDecoder: Normalizing flow for complex observation distributions
  Hybrid   - MixtureDecoder: Gaussian mixture with attention-weighted components

INFERENCE:
  Option A - FilteringSMC: Sequential Monte Carlo (particle filtering)
  Option B - AmortizedVI: Amortized variational inference with encoder network
  Hybrid   - AdaptiveInference: Switches based on observation complexity

Applications:
- Threat actor behavior modeling (discrete state transitions)
- Market regime detection (continuous latent dynamics)
- Geopolitical escalation tracking (mixed discrete-continuous)
- Cognitive state monitoring (integration with NSMPhaseHead)

References:
- Deep State Space Models (Krishnan et al., 2017)
- Structured VAEs (Johnson et al., 2016)
- Filtering Variational Objectives (Maddison et al., 2017)
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Callable, Literal
from enum import Enum, auto
from abc import ABC, abstractmethod
import warnings

# Try importing PyTorch for production use
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal, Categorical, MixtureSameFamily
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Provide stub for type hints
    class nn:  # type: ignore
        class Module:
            pass


# =============================================================================
# CONFIGURATION
# =============================================================================

class StateEncoderType(Enum):
    """State encoder architecture options."""
    VARIATIONAL = auto()    # VAE-based
    CONTRASTIVE = auto()    # InfoNCE-based
    HYBRID = auto()         # Dual path


class TransitionType(Enum):
    """Transition model architecture options."""
    GRU = auto()            # Recurrent gated
    TRANSFORMER = auto()    # Self-attention
    HYBRID = auto()         # Attentive GRU


class DecoderType(Enum):
    """Observation decoder architecture options."""
    PROBABILISTIC = auto()  # Gaussian
    FLOW = auto()           # Normalizing flow
    HYBRID = auto()         # Mixture model


class InferenceType(Enum):
    """Inference algorithm options."""
    SMC = auto()            # Particle filtering
    AMORTIZED_VI = auto()   # Variational inference
    HYBRID = auto()         # Adaptive switching


@dataclass
class NSMPipelineConfig:
    """Configuration for Neural State Machine pipeline."""
    # Dimensions
    observation_dim: int = 256
    state_dim: int = 64
    hidden_dim: int = 128
    n_states: int = 8           # For discrete state models

    # Architecture choices
    encoder_type: StateEncoderType = StateEncoderType.HYBRID
    transition_type: TransitionType = TransitionType.HYBRID
    decoder_type: DecoderType = DecoderType.HYBRID
    inference_type: InferenceType = InferenceType.HYBRID

    # Training parameters
    kl_weight: float = 1.0
    contrastive_temperature: float = 0.07
    n_particles: int = 100       # For SMC

    # Regularization
    dropout: float = 0.1
    state_prior_std: float = 1.0

    # Hybrid parameters
    encoder_generative_weight: float = 0.5
    transition_attention_heads: int = 4
    decoder_n_components: int = 4

    # Flow parameters (for flow decoder)
    n_flow_layers: int = 4

    # Inference switching thresholds
    smc_complexity_threshold: float = 0.7


@dataclass
class NSMPipelineOutput:
    """Output from NSM pipeline forward pass."""
    # Latent states
    state_mean: NDArray[np.float64]          # [T, state_dim]
    state_std: NDArray[np.float64]           # [T, state_dim]
    state_samples: NDArray[np.float64]       # [T, n_samples, state_dim]

    # Discrete states (if applicable)
    discrete_probs: Optional[NDArray[np.float64]] = None  # [T, n_states]
    discrete_states: Optional[NDArray[np.int64]] = None   # [T]

    # Observation predictions
    obs_mean: Optional[NDArray[np.float64]] = None
    obs_std: Optional[NDArray[np.float64]] = None

    # Loss components
    reconstruction_loss: float = 0.0
    kl_loss: float = 0.0
    contrastive_loss: float = 0.0

    # Diagnostics
    effective_sample_size: Optional[float] = None  # For SMC
    evidence_lower_bound: Optional[float] = None


# =============================================================================
# NUMPY IMPLEMENTATIONS (For environments without PyTorch)
# =============================================================================

class NumpyModule(ABC):
    """Base class for numpy-based modules."""

    @abstractmethod
    def forward(self, x: NDArray) -> NDArray:
        pass

    def __call__(self, x: NDArray) -> NDArray:
        return self.forward(x)


# -----------------------------------------------------------------------------
# STATE ENCODERS
# -----------------------------------------------------------------------------

class VariationalStateEncoderNP(NumpyModule):
    """
    Option A: Variational autoencoder-based state encoding.

    Uses amortized inference to map observations to latent state
    distributions q(z|x) with diagonal Gaussian posterior.
    """

    def __init__(
        self,
        observation_dim: int,
        state_dim: int,
        hidden_dim: int = 128,
        prior_std: float = 1.0
    ):
        self.observation_dim = observation_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.prior_std = prior_std

        # Initialize weights
        np.random.seed(42)
        scale = 0.01

        self.W1 = np.random.randn(observation_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b2 = np.zeros(hidden_dim)
        self.W_mean = np.random.randn(hidden_dim, state_dim) * scale
        self.b_mean = np.zeros(state_dim)
        self.W_logvar = np.random.randn(hidden_dim, state_dim) * scale
        self.b_logvar = np.zeros(state_dim)

    def forward(self, x: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Encode observation to latent distribution.

        Args:
            x: [batch, observation_dim] observations

        Returns:
            Tuple of (mean, log_variance) each [batch, state_dim]
        """
        # Two-layer MLP encoder
        h = np.tanh(x @ self.W1 + self.b1)
        h = np.tanh(h @ self.W2 + self.b2)

        mean = h @ self.W_mean + self.b_mean
        logvar = h @ self.W_logvar + self.b_logvar

        # Clamp log variance for stability
        logvar = np.clip(logvar, -10, 2)

        return mean, logvar

    def sample(self, mean: NDArray, logvar: NDArray, n_samples: int = 1) -> NDArray:
        """Sample from the posterior using reparameterization trick."""
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(n_samples, *mean.shape)
        return mean + std * eps

    def kl_divergence(self, mean: NDArray, logvar: NDArray) -> float:
        """KL divergence from posterior to prior."""
        # KL(q(z|x) || p(z)) for Gaussian prior
        kl = -0.5 * np.sum(
            1 + logvar - np.log(self.prior_std**2) -
            (mean**2 + np.exp(logvar)) / self.prior_std**2
        )
        return float(kl) / mean.shape[0]


class ContrastiveStateEncoderNP(NumpyModule):
    """
    Option B: Contrastive learning-based state encoding.

    Uses InfoNCE objective to learn discriminative state representations
    that capture temporal structure through positive/negative pairs.
    """

    def __init__(
        self,
        observation_dim: int,
        state_dim: int,
        hidden_dim: int = 128,
        temperature: float = 0.07
    ):
        self.observation_dim = observation_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        np.random.seed(43)
        scale = 0.01

        # Projection network
        self.W1 = np.random.randn(observation_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b2 = np.zeros(hidden_dim)
        self.W_proj = np.random.randn(hidden_dim, state_dim) * scale
        self.b_proj = np.zeros(state_dim)

    def forward(self, x: NDArray) -> NDArray:
        """
        Encode observation to normalized embedding.

        Args:
            x: [batch, observation_dim] observations

        Returns:
            [batch, state_dim] L2-normalized embeddings
        """
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        h = np.maximum(0, h @ self.W2 + self.b2)
        z = h @ self.W_proj + self.b_proj

        # L2 normalize
        norm = np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8
        return z / norm

    def contrastive_loss(
        self,
        z_anchor: NDArray,
        z_positive: NDArray,
        z_negatives: NDArray
    ) -> float:
        """
        Compute InfoNCE loss.

        Args:
            z_anchor: [batch, state_dim] anchor embeddings
            z_positive: [batch, state_dim] positive embeddings
            z_negatives: [n_neg, state_dim] negative embeddings

        Returns:
            InfoNCE loss value
        """
        # Positive similarity
        pos_sim = np.sum(z_anchor * z_positive, axis=-1) / self.temperature

        # Negative similarities
        neg_sim = z_anchor @ z_negatives.T / self.temperature

        # Log-sum-exp for denominator
        all_sim = np.concatenate([pos_sim[:, None], neg_sim], axis=1)
        log_sum_exp = np.log(np.sum(np.exp(all_sim - all_sim.max(axis=1, keepdims=True)), axis=1))
        log_sum_exp += all_sim.max(axis=1)

        # InfoNCE loss
        loss = -pos_sim + log_sum_exp
        return float(np.mean(loss))


class DualPathEncoderNP(NumpyModule):
    """
    Hybrid: Combines variational and contrastive objectives.

    Uses both generative (VAE) and discriminative (contrastive) paths
    with learnable weighting for complementary state representations.
    """

    def __init__(
        self,
        observation_dim: int,
        state_dim: int,
        hidden_dim: int = 128,
        generative_weight: float = 0.5,
        temperature: float = 0.07,
        prior_std: float = 1.0
    ):
        self.generative_weight = generative_weight
        self.discriminative_weight = 1.0 - generative_weight

        # Both encoder paths
        self.variational = VariationalStateEncoderNP(
            observation_dim, state_dim, hidden_dim, prior_std
        )
        self.contrastive = ContrastiveStateEncoderNP(
            observation_dim, state_dim, hidden_dim, temperature
        )

        # Fusion layer
        np.random.seed(44)
        self.W_fuse = np.random.randn(state_dim * 2, state_dim) * 0.01
        self.b_fuse = np.zeros(state_dim)

    def forward(self, x: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Dual-path encoding with fusion.

        Returns:
            Tuple of (fused_state, vae_mean, vae_logvar)
        """
        # Generative path
        vae_mean, vae_logvar = self.variational.forward(x)
        vae_sample = self.variational.sample(vae_mean, vae_logvar, n_samples=1)[0]

        # Discriminative path
        contrastive_z = self.contrastive.forward(x)

        # Fuse representations
        concat = np.concatenate([vae_sample, contrastive_z], axis=-1)
        fused = np.tanh(concat @ self.W_fuse + self.b_fuse)

        return fused, vae_mean, vae_logvar


# -----------------------------------------------------------------------------
# TRANSITION MODELS
# -----------------------------------------------------------------------------

class GRUTransitionNP(NumpyModule):
    """
    Option A: GRU-based state transition.

    Recurrent gated transition with explicit forget dynamics
    for modeling state persistence and change.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        np.random.seed(45)
        scale = 0.01

        # GRU parameters
        concat_dim = state_dim * 2  # state + input

        # Update gate
        self.W_z = np.random.randn(concat_dim, state_dim) * scale
        self.b_z = np.zeros(state_dim)

        # Reset gate
        self.W_r = np.random.randn(concat_dim, state_dim) * scale
        self.b_r = np.zeros(state_dim)

        # Candidate state
        self.W_h = np.random.randn(concat_dim, state_dim) * scale
        self.b_h = np.zeros(state_dim)

    def forward(
        self,
        state: NDArray,
        input_: Optional[NDArray] = None
    ) -> NDArray:
        """
        Transition from current state to next.

        Args:
            state: [batch, state_dim] current state
            input_: [batch, state_dim] optional control input

        Returns:
            [batch, state_dim] next state
        """
        if input_ is None:
            input_ = np.zeros_like(state)

        concat = np.concatenate([state, input_], axis=-1)

        # Gates
        z = self._sigmoid(concat @ self.W_z + self.b_z)
        r = self._sigmoid(concat @ self.W_r + self.b_r)

        # Candidate with reset
        reset_state = r * state
        concat_reset = np.concatenate([reset_state, input_], axis=-1)
        h_candidate = np.tanh(concat_reset @ self.W_h + self.b_h)

        # Update
        next_state = (1 - z) * state + z * h_candidate

        return next_state

    @staticmethod
    def _sigmoid(x: NDArray) -> NDArray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class TransformerTransitionNP(NumpyModule):
    """
    Option B: Transformer-based state transition.

    Self-attention over state history for modeling long-range
    dependencies in state evolution.
    """

    def __init__(
        self,
        state_dim: int,
        n_heads: int = 4,
        context_length: int = 16
    ):
        self.state_dim = state_dim
        self.n_heads = n_heads
        self.head_dim = state_dim // n_heads
        self.context_length = context_length

        np.random.seed(46)
        scale = 0.01

        # Multi-head attention parameters
        self.W_q = np.random.randn(state_dim, state_dim) * scale
        self.W_k = np.random.randn(state_dim, state_dim) * scale
        self.W_v = np.random.randn(state_dim, state_dim) * scale
        self.W_o = np.random.randn(state_dim, state_dim) * scale

        # FFN
        self.W_ff1 = np.random.randn(state_dim, state_dim * 4) * scale
        self.b_ff1 = np.zeros(state_dim * 4)
        self.W_ff2 = np.random.randn(state_dim * 4, state_dim) * scale
        self.b_ff2 = np.zeros(state_dim)

        # Layer norms (simplified)
        self.ln1_gamma = np.ones(state_dim)
        self.ln1_beta = np.zeros(state_dim)
        self.ln2_gamma = np.ones(state_dim)
        self.ln2_beta = np.zeros(state_dim)

        # State history buffer
        self.history: List[NDArray] = []

    def forward(
        self,
        state: NDArray,
        use_history: bool = True
    ) -> NDArray:
        """
        Transition using self-attention over history.

        Args:
            state: [batch, state_dim] current state
            use_history: whether to attend to state history

        Returns:
            [batch, state_dim] next state
        """
        batch_size = state.shape[0]

        if use_history and len(self.history) > 0:
            # Build context from history
            context = np.stack(self.history[-self.context_length:] + [state], axis=1)
        else:
            context = state[:, None, :]  # [batch, 1, state_dim]

        # Self-attention
        attended = self._multi_head_attention(context)[:, -1, :]

        # Residual + norm
        normed = self._layer_norm(state + attended, self.ln1_gamma, self.ln1_beta)

        # FFN
        ff = np.maximum(0, normed @ self.W_ff1 + self.b_ff1)
        ff = ff @ self.W_ff2 + self.b_ff2

        # Residual + norm
        next_state = self._layer_norm(normed + ff, self.ln2_gamma, self.ln2_beta)

        # Update history
        if use_history:
            self.history.append(state.copy())
            if len(self.history) > self.context_length:
                self.history.pop(0)

        return next_state

    def _multi_head_attention(self, x: NDArray) -> NDArray:
        """Multi-head self-attention."""
        batch, seq, _ = x.shape

        # Project to Q, K, V
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape for multi-head
        Q = Q.reshape(batch, seq, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention scores
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)

        # Causal mask
        mask = np.triu(np.ones((seq, seq)) * -1e9, k=1)
        scores = scores + mask

        # Softmax
        attn = self._softmax(scores)

        # Apply to values
        out = attn @ V

        # Reshape and project
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, -1)
        return out @ self.W_o

    @staticmethod
    def _softmax(x: NDArray) -> NDArray:
        exp_x = np.exp(x - x.max(axis=-1, keepdims=True))
        return exp_x / (exp_x.sum(axis=-1, keepdims=True) + 1e-8)

    @staticmethod
    def _layer_norm(x: NDArray, gamma: NDArray, beta: NDArray, eps: float = 1e-5) -> NDArray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    def reset_history(self):
        """Clear state history buffer."""
        self.history = []


class AttentiveGRUTransitionNP(NumpyModule):
    """
    Hybrid: GRU with cross-attention to observation history.

    Combines recurrent dynamics with attention mechanism for
    selective incorporation of historical observations.
    """

    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        n_heads: int = 4,
        context_length: int = 16
    ):
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.n_heads = n_heads
        self.head_dim = state_dim // n_heads
        self.context_length = context_length

        # GRU core
        self.gru = GRUTransitionNP(state_dim)

        np.random.seed(47)
        scale = 0.01

        # Cross-attention (state queries observation history)
        self.W_q = np.random.randn(state_dim, state_dim) * scale
        self.W_k = np.random.randn(observation_dim, state_dim) * scale
        self.W_v = np.random.randn(observation_dim, state_dim) * scale
        self.W_o = np.random.randn(state_dim, state_dim) * scale

        # Gate for attention influence
        self.W_gate = np.random.randn(state_dim * 2, state_dim) * scale
        self.b_gate = np.zeros(state_dim)

        # Observation history
        self.obs_history: List[NDArray] = []

    def forward(
        self,
        state: NDArray,
        observation: NDArray
    ) -> NDArray:
        """
        Transition with attention to observation history.

        Args:
            state: [batch, state_dim] current state
            observation: [batch, observation_dim] current observation

        Returns:
            [batch, state_dim] next state
        """
        # GRU transition
        gru_next = self.gru.forward(state)

        # Build observation context
        self.obs_history.append(observation.copy())
        if len(self.obs_history) > self.context_length:
            self.obs_history.pop(0)

        if len(self.obs_history) < 2:
            return gru_next

        # Cross-attention
        obs_context = np.stack(self.obs_history, axis=1)  # [batch, T, obs_dim]
        attended = self._cross_attention(state, obs_context)

        # Gated fusion
        concat = np.concatenate([gru_next, attended], axis=-1)
        gate = self._sigmoid(concat @ self.W_gate + self.b_gate)

        next_state = gate * gru_next + (1 - gate) * attended

        return next_state

    def _cross_attention(self, query: NDArray, context: NDArray) -> NDArray:
        """Cross-attention from state to observation history."""
        batch, seq, _ = context.shape

        Q = query @ self.W_q  # [batch, state_dim]
        K = context @ self.W_k  # [batch, seq, state_dim]
        V = context @ self.W_v

        # Attention scores
        scores = (Q[:, None, :] @ K.transpose(0, 2, 1)) / np.sqrt(self.state_dim)
        attn = self._softmax(scores)  # [batch, 1, seq]

        out = (attn @ V)[:, 0, :]  # [batch, state_dim]
        return out @ self.W_o

    @staticmethod
    def _sigmoid(x: NDArray) -> NDArray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def _softmax(x: NDArray) -> NDArray:
        exp_x = np.exp(x - x.max(axis=-1, keepdims=True))
        return exp_x / (exp_x.sum(axis=-1, keepdims=True) + 1e-8)

    def reset_history(self):
        """Clear observation history."""
        self.obs_history = []


# -----------------------------------------------------------------------------
# OBSERVATION DECODERS
# -----------------------------------------------------------------------------

class ProbabilisticDecoderNP(NumpyModule):
    """
    Option A: Gaussian observation model.

    Decodes latent states to observation distributions with
    learned mean and variance.
    """

    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        hidden_dim: int = 128,
        min_std: float = 0.01
    ):
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.min_std = min_std

        np.random.seed(48)
        scale = 0.01

        self.W1 = np.random.randn(state_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b2 = np.zeros(hidden_dim)
        self.W_mean = np.random.randn(hidden_dim, observation_dim) * scale
        self.b_mean = np.zeros(observation_dim)
        self.W_logvar = np.random.randn(hidden_dim, observation_dim) * scale
        self.b_logvar = np.zeros(observation_dim) - 1  # Start with small variance

    def forward(self, state: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Decode state to observation distribution.

        Args:
            state: [batch, state_dim] latent states

        Returns:
            Tuple of (mean, std) each [batch, observation_dim]
        """
        h = np.tanh(state @ self.W1 + self.b1)
        h = np.tanh(h @ self.W2 + self.b2)

        mean = h @ self.W_mean + self.b_mean
        logvar = h @ self.W_logvar + self.b_logvar
        std = np.exp(0.5 * np.clip(logvar, -10, 2)) + self.min_std

        return mean, std

    def log_prob(
        self,
        state: NDArray,
        observation: NDArray
    ) -> float:
        """Compute log probability of observation given state."""
        mean, std = self.forward(state)
        log_p = -0.5 * (
            np.log(2 * np.pi) +
            2 * np.log(std) +
            ((observation - mean) / std) ** 2
        )
        return float(np.sum(log_p))

    def sample(self, state: NDArray) -> NDArray:
        """Sample observation from decoded distribution."""
        mean, std = self.forward(state)
        return mean + std * np.random.randn(*mean.shape)


class FlowDecoderNP(NumpyModule):
    """
    Option B: Normalizing flow-based decoder.

    Uses invertible transformations for complex observation
    distributions that can't be captured by Gaussians.
    """

    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        n_layers: int = 4,
        hidden_dim: int = 128
    ):
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.n_layers = n_layers

        np.random.seed(49)
        scale = 0.01

        # Affine coupling layer parameters
        self.layers = []
        split_dim = observation_dim // 2

        for i in range(n_layers):
            layer = {
                'W1': np.random.randn(split_dim + state_dim, hidden_dim) * scale,
                'b1': np.zeros(hidden_dim),
                'W2': np.random.randn(hidden_dim, hidden_dim) * scale,
                'b2': np.zeros(hidden_dim),
                'W_scale': np.random.randn(hidden_dim, observation_dim - split_dim) * scale,
                'b_scale': np.zeros(observation_dim - split_dim),
                'W_shift': np.random.randn(hidden_dim, observation_dim - split_dim) * scale,
                'b_shift': np.zeros(observation_dim - split_dim),
            }
            self.layers.append(layer)

    def forward(
        self,
        state: NDArray,
        z: NDArray
    ) -> Tuple[NDArray, float]:
        """
        Transform base distribution through flow.

        Args:
            state: [batch, state_dim] conditioning state
            z: [batch, observation_dim] samples from base distribution

        Returns:
            Tuple of (transformed samples, log determinant)
        """
        x = z.copy()
        log_det = 0.0
        split = self.observation_dim // 2

        for i, layer in enumerate(self.layers):
            # Alternate which half is transformed
            if i % 2 == 0:
                x1, x2 = x[:, :split], x[:, split:]
            else:
                x2, x1 = x[:, :split], x[:, split:]

            # Compute scale and shift from untransformed half + state
            h = np.concatenate([x1, state], axis=-1)
            h = np.tanh(h @ layer['W1'] + layer['b1'])
            h = np.tanh(h @ layer['W2'] + layer['b2'])

            log_scale = np.tanh(h @ layer['W_scale'] + layer['b_scale'])
            shift = h @ layer['W_shift'] + layer['b_shift']

            # Affine transform
            x2_new = x2 * np.exp(log_scale) + shift
            log_det += np.sum(log_scale)

            # Recombine
            if i % 2 == 0:
                x = np.concatenate([x1, x2_new], axis=-1)
            else:
                x = np.concatenate([x2_new, x1], axis=-1)

        return x, log_det / z.shape[0]

    def inverse(
        self,
        state: NDArray,
        x: NDArray
    ) -> Tuple[NDArray, float]:
        """Inverse flow transformation."""
        z = x.copy()
        log_det = 0.0
        split = self.observation_dim // 2

        # Apply layers in reverse
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]

            if i % 2 == 0:
                z1, z2 = z[:, :split], z[:, split:]
            else:
                z2, z1 = z[:, :split], z[:, split:]

            # Compute scale and shift
            h = np.concatenate([z1, state], axis=-1)
            h = np.tanh(h @ layer['W1'] + layer['b1'])
            h = np.tanh(h @ layer['W2'] + layer['b2'])

            log_scale = np.tanh(h @ layer['W_scale'] + layer['b_scale'])
            shift = h @ layer['W_shift'] + layer['b_shift']

            # Inverse affine transform
            z2_new = (z2 - shift) * np.exp(-log_scale)
            log_det -= np.sum(log_scale)

            if i % 2 == 0:
                z = np.concatenate([z1, z2_new], axis=-1)
            else:
                z = np.concatenate([z2_new, z1], axis=-1)

        return z, log_det / x.shape[0]

    def log_prob(self, state: NDArray, observation: NDArray) -> float:
        """Compute log probability under the flow model."""
        z, log_det = self.inverse(state, observation)
        # Log prob under standard normal base distribution
        log_p_z = -0.5 * (np.log(2 * np.pi) + z ** 2).sum()
        return float(log_p_z / observation.shape[0] + log_det)

    def sample(self, state: NDArray, n_samples: int = 1) -> NDArray:
        """Sample from the flow model."""
        batch_size = state.shape[0]
        z = np.random.randn(batch_size, self.observation_dim)
        x, _ = self.forward(state, z)
        return x


class MixtureDecoderNP(NumpyModule):
    """
    Hybrid: Gaussian mixture with attention-weighted components.

    Combines multiple Gaussian components with learned mixture
    weights conditioned on the latent state.
    """

    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        n_components: int = 4,
        hidden_dim: int = 128
    ):
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.n_components = n_components

        np.random.seed(50)
        scale = 0.01

        # Mixture weight predictor
        self.W_mix = np.random.randn(state_dim, n_components) * scale
        self.b_mix = np.zeros(n_components)

        # Component-specific parameters
        self.component_encoders = []
        for _ in range(n_components):
            enc = {
                'W1': np.random.randn(state_dim, hidden_dim) * scale,
                'b1': np.zeros(hidden_dim),
                'W_mean': np.random.randn(hidden_dim, observation_dim) * scale,
                'b_mean': np.zeros(observation_dim),
                'W_logvar': np.random.randn(hidden_dim, observation_dim) * scale,
                'b_logvar': np.zeros(observation_dim) - 1,
            }
            self.component_encoders.append(enc)

    def forward(
        self,
        state: NDArray
    ) -> Tuple[NDArray, List[Tuple[NDArray, NDArray]]]:
        """
        Decode state to mixture distribution.

        Args:
            state: [batch, state_dim]

        Returns:
            Tuple of (mixture_weights, list of (mean, std) per component)
        """
        # Mixture weights
        logits = state @ self.W_mix + self.b_mix
        weights = self._softmax(logits)

        # Component distributions
        components = []
        for enc in self.component_encoders:
            h = np.tanh(state @ enc['W1'] + enc['b1'])
            mean = h @ enc['W_mean'] + enc['b_mean']
            logvar = np.clip(h @ enc['W_logvar'] + enc['b_logvar'], -10, 2)
            std = np.exp(0.5 * logvar) + 0.01
            components.append((mean, std))

        return weights, components

    def log_prob(self, state: NDArray, observation: NDArray) -> float:
        """Log probability under mixture model."""
        weights, components = self.forward(state)

        # Log prob for each component
        log_probs = []
        for mean, std in components:
            log_p = -0.5 * (
                np.log(2 * np.pi) +
                2 * np.log(std) +
                ((observation - mean) / std) ** 2
            )
            log_probs.append(np.sum(log_p, axis=-1))

        log_probs = np.stack(log_probs, axis=-1)  # [batch, n_components]

        # Mixture log probability
        log_mixture = np.log(weights + 1e-8) + log_probs
        log_mixture_sum = np.log(np.sum(np.exp(log_mixture - log_mixture.max(axis=-1, keepdims=True)), axis=-1))
        log_mixture_sum += log_mixture.max(axis=-1)

        return float(np.mean(log_mixture_sum))

    def sample(self, state: NDArray) -> NDArray:
        """Sample from mixture distribution."""
        weights, components = self.forward(state)
        batch_size = state.shape[0]

        # Sample component indices
        cumsum = np.cumsum(weights, axis=-1)
        u = np.random.rand(batch_size, 1)
        component_idx = (u < cumsum).argmax(axis=-1)

        # Sample from selected components
        samples = np.zeros((batch_size, self.observation_dim))
        for i in range(batch_size):
            mean, std = components[component_idx[i]]
            samples[i] = mean[i] + std[i] * np.random.randn(self.observation_dim)

        return samples

    @staticmethod
    def _softmax(x: NDArray) -> NDArray:
        exp_x = np.exp(x - x.max(axis=-1, keepdims=True))
        return exp_x / (exp_x.sum(axis=-1, keepdims=True) + 1e-8)


# -----------------------------------------------------------------------------
# INFERENCE ALGORITHMS
# -----------------------------------------------------------------------------

class FilteringSMCNP:
    """
    Option A: Sequential Monte Carlo (Particle Filtering).

    Maintains particle approximation of posterior state distribution
    with resampling to avoid degeneracy.
    """

    def __init__(
        self,
        n_particles: int = 100,
        resampling_threshold: float = 0.5
    ):
        self.n_particles = n_particles
        self.resampling_threshold = resampling_threshold
        self.particles: Optional[NDArray] = None
        self.weights: Optional[NDArray] = None

    def initialize(self, state_dim: int, batch_size: int = 1):
        """Initialize particle population from prior."""
        self.particles = np.random.randn(batch_size, self.n_particles, state_dim)
        self.weights = np.ones((batch_size, self.n_particles)) / self.n_particles

    def update(
        self,
        observation: NDArray,
        transition_fn: Callable[[NDArray], NDArray],
        observation_log_prob_fn: Callable[[NDArray, NDArray], NDArray]
    ) -> Tuple[NDArray, NDArray, float]:
        """
        SMC update step.

        Args:
            observation: [batch, obs_dim] current observation
            transition_fn: state -> next_state
            observation_log_prob_fn: (state, obs) -> log_prob

        Returns:
            Tuple of (state_mean, state_std, effective_sample_size)
        """
        batch_size = observation.shape[0]
        state_dim = self.particles.shape[-1]

        # Propagate particles through transition
        particles_flat = self.particles.reshape(-1, state_dim)
        propagated = transition_fn(particles_flat)
        propagated = propagated.reshape(batch_size, self.n_particles, state_dim)

        # Weight update
        log_weights = np.log(self.weights + 1e-300)
        for b in range(batch_size):
            for p in range(self.n_particles):
                log_w = observation_log_prob_fn(
                    propagated[b, p:p+1],
                    observation[b:b+1]
                )
                log_weights[b, p] += log_w

        # Normalize weights
        log_weights -= log_weights.max(axis=-1, keepdims=True)
        self.weights = np.exp(log_weights)
        self.weights /= self.weights.sum(axis=-1, keepdims=True)

        # Compute ESS and resample if needed
        ess = 1.0 / np.sum(self.weights ** 2, axis=-1)
        mean_ess = float(np.mean(ess))

        if mean_ess < self.resampling_threshold * self.n_particles:
            self._resample()

        # Update particles
        self.particles = propagated

        # Compute state statistics
        state_mean = np.sum(self.weights[:, :, None] * self.particles, axis=1)
        state_var = np.sum(
            self.weights[:, :, None] * (self.particles - state_mean[:, None, :]) ** 2,
            axis=1
        )
        state_std = np.sqrt(state_var + 1e-8)

        return state_mean, state_std, mean_ess

    def _resample(self):
        """Systematic resampling."""
        batch_size, n_particles, state_dim = self.particles.shape

        for b in range(batch_size):
            cumsum = np.cumsum(self.weights[b])
            positions = (np.random.rand() + np.arange(n_particles)) / n_particles

            indices = np.searchsorted(cumsum, positions)
            indices = np.clip(indices, 0, n_particles - 1)

            self.particles[b] = self.particles[b, indices]
            self.weights[b] = np.ones(n_particles) / n_particles


class AmortizedVINP:
    """
    Option B: Amortized Variational Inference.

    Uses learned encoder network for fast approximate inference
    without iterative optimization at test time.
    """

    def __init__(
        self,
        observation_dim: int,
        state_dim: int,
        hidden_dim: int = 128
    ):
        self.encoder = VariationalStateEncoderNP(
            observation_dim, state_dim, hidden_dim
        )
        self.history: List[NDArray] = []
        self.max_history = 10

        np.random.seed(51)
        scale = 0.01

        # Temporal encoder for conditioning on history
        self.W_hist = np.random.randn(state_dim * self.max_history, hidden_dim) * scale
        self.b_hist = np.zeros(hidden_dim)
        self.W_fuse = np.random.randn(hidden_dim + state_dim, state_dim) * scale
        self.b_fuse = np.zeros(state_dim)

    def infer(
        self,
        observation: NDArray,
        use_history: bool = True
    ) -> Tuple[NDArray, NDArray]:
        """
        Amortized inference for current observation.

        Args:
            observation: [batch, obs_dim]
            use_history: whether to condition on state history

        Returns:
            Tuple of (state_mean, state_std)
        """
        mean, logvar = self.encoder.forward(observation)
        std = np.exp(0.5 * logvar)

        if use_history and len(self.history) > 0:
            # Incorporate history
            hist_features = self._encode_history()
            concat = np.concatenate([hist_features, mean], axis=-1)
            mean = mean + np.tanh(concat @ self.W_fuse + self.b_fuse) * 0.1

        # Update history
        self.history.append(mean.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return mean, std

    def _encode_history(self) -> NDArray:
        """Encode state history into fixed-size feature."""
        # Pad history to max length
        state_dim = self.history[0].shape[-1]
        batch_size = self.history[0].shape[0]

        padded = np.zeros((batch_size, self.max_history * state_dim))
        for i, h in enumerate(self.history):
            padded[:, i*state_dim:(i+1)*state_dim] = h

        return np.tanh(padded @ self.W_hist + self.b_hist)

    def reset_history(self):
        """Clear inference history."""
        self.history = []


class AdaptiveInferenceNP:
    """
    Hybrid: Adaptive switching between SMC and amortized VI.

    Uses complexity metrics to choose appropriate inference
    algorithm for each observation.
    """

    def __init__(
        self,
        observation_dim: int,
        state_dim: int,
        n_particles: int = 100,
        complexity_threshold: float = 0.7
    ):
        self.smc = FilteringSMCNP(n_particles)
        self.vi = AmortizedVINP(observation_dim, state_dim)
        self.complexity_threshold = complexity_threshold

        self.state_dim = state_dim
        self._use_smc = False

    def initialize(self, batch_size: int = 1):
        """Initialize both inference engines."""
        self.smc.initialize(self.state_dim, batch_size)
        self.vi.reset_history()
        self._use_smc = False

    def infer(
        self,
        observation: NDArray,
        transition_fn: Callable,
        obs_log_prob_fn: Callable
    ) -> Tuple[NDArray, NDArray, Dict]:
        """
        Adaptive inference with automatic algorithm selection.

        Args:
            observation: [batch, obs_dim]
            transition_fn: state transition function
            obs_log_prob_fn: observation log probability function

        Returns:
            Tuple of (state_mean, state_std, diagnostics)
        """
        # Estimate observation complexity
        complexity = self._estimate_complexity(observation)

        if complexity > self.complexity_threshold and not self._use_smc:
            # Switch to SMC for complex observations
            self._use_smc = True
            batch_size = observation.shape[0]
            self.smc.initialize(self.state_dim, batch_size)

        if self._use_smc:
            mean, std, ess = self.smc.update(
                observation, transition_fn, obs_log_prob_fn
            )
            diagnostics = {
                'method': 'smc',
                'effective_sample_size': ess,
                'complexity': complexity
            }
        else:
            mean, std = self.vi.infer(observation)
            diagnostics = {
                'method': 'amortized_vi',
                'complexity': complexity
            }

        # Consider switching back to VI if complexity drops
        if self._use_smc and complexity < self.complexity_threshold * 0.5:
            self._use_smc = False

        return mean, std, diagnostics

    def _estimate_complexity(self, observation: NDArray) -> float:
        """
        Estimate observation complexity for algorithm selection.

        Uses:
        - Variance: high variance suggests multi-modal posterior
        - Gradient magnitude: large changes suggest non-stationarity
        """
        var_score = np.var(observation)

        # Gradient score from history
        if len(self.vi.history) > 0:
            prev_obs = self.vi.history[-1]
            grad_score = np.mean(np.abs(observation[:observation.shape[0]] - prev_obs[:observation.shape[0]]))
        else:
            grad_score = 0.0

        # Combine scores (normalized heuristically)
        complexity = 0.5 * np.tanh(var_score) + 0.5 * np.tanh(grad_score)

        return float(complexity)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class NSMPipeline:
    """
    Complete Neural State Machine pipeline with configurable architecture.

    Combines encoder, transition, decoder, and inference modules
    according to configuration for end-to-end state estimation.
    """

    def __init__(self, config: Optional[NSMPipelineConfig] = None):
        self.config = config or NSMPipelineConfig()

        # Build encoder
        self.encoder = self._build_encoder()

        # Build transition model
        self.transition = self._build_transition()

        # Build decoder
        self.decoder = self._build_decoder()

        # Build inference engine
        self.inference = self._build_inference()

    def _build_encoder(self):
        """Build state encoder based on config."""
        if self.config.encoder_type == StateEncoderType.VARIATIONAL:
            return VariationalStateEncoderNP(
                self.config.observation_dim,
                self.config.state_dim,
                self.config.hidden_dim
            )
        elif self.config.encoder_type == StateEncoderType.CONTRASTIVE:
            return ContrastiveStateEncoderNP(
                self.config.observation_dim,
                self.config.state_dim,
                self.config.hidden_dim,
                self.config.contrastive_temperature
            )
        else:  # HYBRID
            return DualPathEncoderNP(
                self.config.observation_dim,
                self.config.state_dim,
                self.config.hidden_dim,
                self.config.encoder_generative_weight,
                self.config.contrastive_temperature
            )

    def _build_transition(self):
        """Build transition model based on config."""
        if self.config.transition_type == TransitionType.GRU:
            return GRUTransitionNP(
                self.config.state_dim,
                self.config.hidden_dim
            )
        elif self.config.transition_type == TransitionType.TRANSFORMER:
            return TransformerTransitionNP(
                self.config.state_dim,
                self.config.transition_attention_heads
            )
        else:  # HYBRID
            return AttentiveGRUTransitionNP(
                self.config.state_dim,
                self.config.observation_dim,
                self.config.transition_attention_heads
            )

    def _build_decoder(self):
        """Build observation decoder based on config."""
        if self.config.decoder_type == DecoderType.PROBABILISTIC:
            return ProbabilisticDecoderNP(
                self.config.state_dim,
                self.config.observation_dim,
                self.config.hidden_dim
            )
        elif self.config.decoder_type == DecoderType.FLOW:
            return FlowDecoderNP(
                self.config.state_dim,
                self.config.observation_dim,
                self.config.n_flow_layers,
                self.config.hidden_dim
            )
        else:  # HYBRID
            return MixtureDecoderNP(
                self.config.state_dim,
                self.config.observation_dim,
                self.config.decoder_n_components,
                self.config.hidden_dim
            )

    def _build_inference(self):
        """Build inference engine based on config."""
        if self.config.inference_type == InferenceType.SMC:
            return FilteringSMCNP(self.config.n_particles)
        elif self.config.inference_type == InferenceType.AMORTIZED_VI:
            return AmortizedVINP(
                self.config.observation_dim,
                self.config.state_dim,
                self.config.hidden_dim
            )
        else:  # HYBRID
            return AdaptiveInferenceNP(
                self.config.observation_dim,
                self.config.state_dim,
                self.config.n_particles,
                self.config.smc_complexity_threshold
            )

    def process_sequence(
        self,
        observations: NDArray,
        return_samples: bool = False,
        n_samples: int = 10
    ) -> NSMPipelineOutput:
        """
        Process observation sequence through full pipeline.

        Args:
            observations: [T, obs_dim] or [T, batch, obs_dim] observation sequence
            return_samples: whether to return state samples
            n_samples: number of samples per timestep

        Returns:
            NSMPipelineOutput with state estimates and diagnostics
        """
        # Handle dimensions
        if observations.ndim == 2:
            observations = observations[:, None, :]  # Add batch dim

        T, batch_size, obs_dim = observations.shape

        # Initialize storage
        state_means = np.zeros((T, batch_size, self.config.state_dim))
        state_stds = np.zeros((T, batch_size, self.config.state_dim))

        if return_samples:
            state_samples = np.zeros((T, batch_size, n_samples, self.config.state_dim))
        else:
            state_samples = np.zeros((0,))

        # Initialize inference
        if hasattr(self.inference, 'initialize'):
            self.inference.initialize(batch_size)

        # Reset transition history if applicable
        if hasattr(self.transition, 'reset_history'):
            self.transition.reset_history()

        total_recon_loss = 0.0
        total_kl_loss = 0.0
        ess_values = []

        prev_state = None

        for t in range(T):
            obs_t = observations[t]  # [batch, obs_dim]

            # Inference step
            if isinstance(self.inference, AdaptiveInferenceNP):
                mean, std, diag = self.inference.infer(
                    obs_t,
                    lambda s: self.transition.forward(s, obs_t) if hasattr(self.transition, 'obs_history') else self.transition.forward(s),
                    lambda s, o: self.decoder.log_prob(s, o)
                )
                if 'effective_sample_size' in diag:
                    ess_values.append(diag['effective_sample_size'])
            elif isinstance(self.inference, FilteringSMCNP):
                mean, std, ess = self.inference.update(
                    obs_t,
                    lambda s: self.transition.forward(s, obs_t) if hasattr(self.transition, 'obs_history') else self.transition.forward(s),
                    lambda s, o: self.decoder.log_prob(s, o)
                )
                ess_values.append(ess)
            else:  # AmortizedVI
                mean, std = self.inference.infer(obs_t)

            state_means[t] = mean
            state_stds[t] = std

            if return_samples:
                # Sample from posterior
                for s in range(n_samples):
                    state_samples[t, :, s, :] = mean + std * np.random.randn(*mean.shape)

            # Compute losses
            recon_loss = -self.decoder.log_prob(mean, obs_t)
            total_recon_loss += recon_loss

            # KL loss (if using variational encoder)
            if hasattr(self.encoder, 'kl_divergence'):
                if isinstance(self.encoder, DualPathEncoderNP):
                    _, vae_mean, vae_logvar = self.encoder.forward(obs_t)
                    kl = self.encoder.variational.kl_divergence(vae_mean, vae_logvar)
                else:
                    enc_mean, enc_logvar = self.encoder.forward(obs_t)
                    kl = self.encoder.kl_divergence(enc_mean, enc_logvar)
                total_kl_loss += kl

            prev_state = mean

        # Compute mean ESS if available
        mean_ess = float(np.mean(ess_values)) if ess_values else None

        # Squeeze batch dimension if it was added
        if observations.shape[1] == 1:
            state_means = state_means[:, 0, :]
            state_stds = state_stds[:, 0, :]
            if return_samples:
                state_samples = state_samples[:, 0, :, :]

        return NSMPipelineOutput(
            state_mean=state_means,
            state_std=state_stds,
            state_samples=state_samples,
            reconstruction_loss=total_recon_loss / T,
            kl_loss=total_kl_loss / T,
            effective_sample_size=mean_ess,
            evidence_lower_bound=-(total_recon_loss + self.config.kl_weight * total_kl_loss) / T
        )

    def reset(self):
        """Reset all stateful components."""
        if hasattr(self.transition, 'reset_history'):
            self.transition.reset_history()
        if hasattr(self.inference, 'initialize'):
            self.inference.initialize(1)
        if hasattr(self.inference, 'reset_history'):
            self.inference.reset_history()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_threat_actor_nsm() -> NSMPipeline:
    """
    Create NSM configured for threat actor behavior modeling.

    Uses discrete-continuous hybrid for modeling:
    - Discrete: Attack phase (recon, initial access, persistence, etc.)
    - Continuous: Operational tempo, TTPs similarity
    """
    config = NSMPipelineConfig(
        observation_dim=256,    # Encoded attack indicators
        state_dim=32,           # Compact behavior representation
        n_states=8,             # Attack phases
        encoder_type=StateEncoderType.HYBRID,
        transition_type=TransitionType.HYBRID,  # Attention for long campaigns
        decoder_type=DecoderType.HYBRID,        # Mixture for diverse TTPs
        inference_type=InferenceType.HYBRID,
    )
    return NSMPipeline(config)


def create_market_regime_nsm() -> NSMPipeline:
    """
    Create NSM configured for market regime detection.

    Continuous latent dynamics with flow decoder for
    complex return distributions.
    """
    config = NSMPipelineConfig(
        observation_dim=128,    # Market features
        state_dim=16,           # Regime embedding
        encoder_type=StateEncoderType.VARIATIONAL,
        transition_type=TransitionType.GRU,     # Smooth regime transitions
        decoder_type=DecoderType.FLOW,          # Fat tails, skewness
        inference_type=InferenceType.AMORTIZED_VI,
    )
    return NSMPipeline(config)


def create_escalation_tracker_nsm() -> NSMPipeline:
    """
    Create NSM configured for geopolitical escalation tracking.

    Transformer transition for long-range dependencies,
    SMC for multi-modal scenario tracking.
    """
    config = NSMPipelineConfig(
        observation_dim=512,    # Rich event features
        state_dim=64,           # Escalation state
        encoder_type=StateEncoderType.CONTRASTIVE,  # Discriminative events
        transition_type=TransitionType.TRANSFORMER,  # Long-range context
        decoder_type=DecoderType.PROBABILISTIC,
        inference_type=InferenceType.SMC,
        n_particles=200,        # More particles for scenario tracking
    )
    return NSMPipeline(config)


def create_cognitive_monitor_nsm(nsm_phase_head=None) -> NSMPipeline:
    """
    Create NSM for cognitive state monitoring.

    Integrates with NSMPhaseHead for phase-based
    cognitive state tracking.
    """
    config = NSMPipelineConfig(
        observation_dim=768,    # Transformer hidden dim
        state_dim=64,           # Match NSMPhaseHead oscillators
        encoder_type=StateEncoderType.HYBRID,
        transition_type=TransitionType.HYBRID,
        decoder_type=DecoderType.HYBRID,
        inference_type=InferenceType.HYBRID,
    )
    pipeline = NSMPipeline(config)

    # Optionally attach phase head for Kuramoto analysis
    if nsm_phase_head is not None:
        pipeline.phase_head = nsm_phase_head

    return pipeline


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def compare_encoder_options(
    observations: NDArray,
    state_dim: int = 64
) -> Dict[str, Dict]:
    """
    Compare the three encoder architectures on given observations.

    Returns metrics for each: variational, contrastive, hybrid.
    """
    obs_dim = observations.shape[-1]
    results = {}

    # Variational encoder
    vae = VariationalStateEncoderNP(obs_dim, state_dim)
    mean, logvar = vae.forward(observations)
    kl = vae.kl_divergence(mean, logvar)
    results['variational'] = {
        'state_mean': mean,
        'state_std': np.exp(0.5 * logvar),
        'kl_divergence': kl,
        'description': 'Generative, smooth latent space, KL regularized'
    }

    # Contrastive encoder
    contrast = ContrastiveStateEncoderNP(obs_dim, state_dim)
    z = contrast.forward(observations)
    results['contrastive'] = {
        'state_embedding': z,
        'embedding_norm': np.linalg.norm(z, axis=-1).mean(),
        'description': 'Discriminative, normalized embeddings, temporal structure'
    }

    # Hybrid encoder
    hybrid = DualPathEncoderNP(obs_dim, state_dim)
    fused, vae_mean, vae_logvar = hybrid.forward(observations)
    results['hybrid'] = {
        'fused_state': fused,
        'vae_component_mean': vae_mean,
        'description': 'Combined generative + discriminative, best of both'
    }

    return results


def compare_transition_options(
    state_sequence: NDArray,
    observations: Optional[NDArray] = None
) -> Dict[str, Dict]:
    """
    Compare transition model architectures.
    """
    state_dim = state_sequence.shape[-1]
    T = state_sequence.shape[0]
    results = {}

    # GRU transition
    gru = GRUTransitionNP(state_dim)
    gru_predictions = []
    for t in range(T - 1):
        pred = gru.forward(state_sequence[t:t+1])
        gru_predictions.append(pred[0])
    gru_predictions = np.stack(gru_predictions)
    gru_error = np.mean((gru_predictions - state_sequence[1:]) ** 2)
    results['gru'] = {
        'predictions': gru_predictions,
        'mse': gru_error,
        'description': 'Recurrent, explicit forget gate, efficient'
    }

    # Transformer transition
    transformer = TransformerTransitionNP(state_dim)
    trans_predictions = []
    for t in range(T - 1):
        pred = transformer.forward(state_sequence[t:t+1])
        trans_predictions.append(pred[0])
    trans_predictions = np.stack(trans_predictions)
    trans_error = np.mean((trans_predictions - state_sequence[1:]) ** 2)
    results['transformer'] = {
        'predictions': trans_predictions,
        'mse': trans_error,
        'description': 'Self-attention, long-range dependencies, context-aware'
    }

    # Hybrid (if observations provided)
    if observations is not None:
        hybrid = AttentiveGRUTransitionNP(state_dim, observations.shape[-1])
        hybrid_predictions = []
        for t in range(T - 1):
            pred = hybrid.forward(state_sequence[t:t+1], observations[t:t+1])
            hybrid_predictions.append(pred[0])
        hybrid_predictions = np.stack(hybrid_predictions)
        hybrid_error = np.mean((hybrid_predictions - state_sequence[1:]) ** 2)
        results['hybrid'] = {
            'predictions': hybrid_predictions,
            'mse': hybrid_error,
            'description': 'GRU + cross-attention, observation-conditioned'
        }

    return results


def compare_decoder_options(
    states: NDArray,
    observations: NDArray
) -> Dict[str, Dict]:
    """
    Compare decoder architectures on reconstruction quality.
    """
    state_dim = states.shape[-1]
    obs_dim = observations.shape[-1]
    results = {}

    # Probabilistic decoder
    prob = ProbabilisticDecoderNP(state_dim, obs_dim)
    prob_log_prob = prob.log_prob(states, observations)
    results['probabilistic'] = {
        'log_prob': prob_log_prob,
        'description': 'Gaussian, learned variance, fast'
    }

    # Flow decoder
    flow = FlowDecoderNP(state_dim, obs_dim)
    flow_log_prob = flow.log_prob(states, observations)
    results['flow'] = {
        'log_prob': flow_log_prob,
        'description': 'Normalizing flow, complex distributions, invertible'
    }

    # Mixture decoder
    mixture = MixtureDecoderNP(state_dim, obs_dim)
    mixture_log_prob = mixture.log_prob(states, observations)
    results['mixture'] = {
        'log_prob': mixture_log_prob,
        'description': 'Gaussian mixture, multi-modal, attention-weighted'
    }

    return results
