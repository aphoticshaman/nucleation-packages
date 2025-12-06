"""
Micro-Grokking Detection from Token Entropy.

Implements Insight #1: Per-problem "micro-grokking" is detectable from entropy alone.

The second derivative of token entropy over time is an order parameter for
"micro-grokking" - the moment the model switches from exploration to
circuit-like exploitation.

Key claim: For a single problem instance, problems the solver gets RIGHT
should show a sharp negative kink in entropy (local second-derivative spike);
failures should stay flat/high or oscillatory.

Applications:
- Real-time success predictor BEFORE final answer
- Early stopping on "already grokked" runs
- Allocate more samples when entropy never collapses

Based on:
- Grokking simulations where effective dimension predicts phase transitions
- Token-level entropy traces in insight inference
- arXiv grokking papers on delayed generalization
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


@dataclass
class GrokSignal:
    """Detected micro-grokking signal."""
    token_idx: int          # Token position of grok
    entropy_before: float   # Mean entropy before grok
    entropy_after: float    # Mean entropy after grok
    d2_entropy: float       # Second derivative at grok point
    confidence: float       # Confidence in detection [0, 1]
    grok_type: str          # 'sharp', 'gradual', 'oscillatory', 'none'


@dataclass
class MicroGrokConfig:
    """Configuration for micro-grokking detection."""
    window_size: int = 5           # Window for derivative estimation
    d2_threshold: float = -0.5     # Threshold for sharp entropy drop
    min_tokens: int = 10           # Minimum tokens to analyze
    smooth_sigma: float = 1.0      # Gaussian smoothing for entropy
    entropy_drop_ratio: float = 0.3  # Minimum drop ratio to qualify


def compute_token_entropy(
    logits: NDArray[np.float64],
    temperature: float = 1.0
) -> NDArray[np.float64]:
    """
    Compute per-token entropy from logits.

    H(t) = -sum_v p(v|context_t) * log p(v|context_t)

    Args:
        logits: [seq_len, vocab_size] raw logits
        temperature: Softmax temperature

    Returns:
        [seq_len] entropy values in nats
    """
    # Temperature-scaled softmax
    scaled = logits / temperature
    max_logits = np.max(scaled, axis=-1, keepdims=True)
    exp_logits = np.exp(scaled - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Entropy: -sum(p * log(p))
    log_probs = np.log(probs + 1e-10)
    entropy = -np.sum(probs * log_probs, axis=-1)

    return entropy


def compute_entropy_derivatives(
    entropy: NDArray[np.float64],
    config: MicroGrokConfig = MicroGrokConfig()
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute first and second derivatives of entropy trace.

    Uses Savitzky-Golay filter for smooth differentiation.

    Args:
        entropy: [seq_len] entropy values
        config: Detection configuration

    Returns:
        (d1_entropy, d2_entropy) - first and second derivatives
    """
    seq_len = len(entropy)

    if seq_len < config.window_size:
        return np.zeros_like(entropy), np.zeros_like(entropy)

    # Smooth entropy trace
    entropy_smooth = gaussian_filter1d(entropy, config.smooth_sigma)

    # First derivative via Savgol (polynomial fit + differentiate)
    window = min(config.window_size, seq_len - 1)
    if window % 2 == 0:
        window -= 1
    window = max(3, window)

    d1 = savgol_filter(entropy_smooth, window, polyorder=2, deriv=1)
    d2 = savgol_filter(entropy_smooth, window, polyorder=2, deriv=2)

    return d1, d2


def detect_micro_grokking(
    entropy: NDArray[np.float64],
    config: MicroGrokConfig = MicroGrokConfig()
) -> Optional[GrokSignal]:
    """
    Detect micro-grokking from entropy trace.

    The KEY insight: a sharp negative second derivative indicates
    the model has "clicked" - switching from exploration to exploitation.

    Args:
        entropy: [seq_len] per-token entropy values
        config: Detection configuration

    Returns:
        GrokSignal if detected, None otherwise
    """
    seq_len = len(entropy)

    if seq_len < config.min_tokens:
        return None

    # Compute derivatives
    d1, d2 = compute_entropy_derivatives(entropy, config)

    # Find strongest negative d2 (entropy acceleration downward)
    # This is the "grok point"
    min_d2_idx = np.argmin(d2)
    min_d2_val = d2[min_d2_idx]

    # Check if it qualifies as a grok
    if min_d2_val > config.d2_threshold:
        # No sharp drop detected
        return GrokSignal(
            token_idx=-1,
            entropy_before=float(np.mean(entropy)),
            entropy_after=float(np.mean(entropy)),
            d2_entropy=float(min_d2_val),
            confidence=0.0,
            grok_type='none'
        )

    # Compute before/after entropy
    before_window = max(0, min_d2_idx - config.window_size)
    after_window = min(seq_len, min_d2_idx + config.window_size)

    entropy_before = np.mean(entropy[before_window:min_d2_idx]) if min_d2_idx > 0 else entropy[0]
    entropy_after = np.mean(entropy[min_d2_idx:after_window])

    # Check entropy drop ratio
    drop_ratio = (entropy_before - entropy_after) / (entropy_before + 1e-10)

    # Classify grok type
    if min_d2_val < -1.0 and drop_ratio > 0.4:
        grok_type = 'sharp'
        confidence = min(1.0, abs(min_d2_val) * drop_ratio)
    elif drop_ratio > config.entropy_drop_ratio:
        grok_type = 'gradual'
        confidence = drop_ratio
    elif np.std(d2) > 0.5:
        grok_type = 'oscillatory'
        confidence = 0.2
    else:
        grok_type = 'none'
        confidence = 0.0

    return GrokSignal(
        token_idx=int(min_d2_idx),
        entropy_before=float(entropy_before),
        entropy_after=float(entropy_after),
        d2_entropy=float(min_d2_val),
        confidence=float(confidence),
        grok_type=grok_type
    )


def predict_success_from_entropy(
    entropy_trace: NDArray[np.float64],
    partial_token_idx: int,
    config: MicroGrokConfig = MicroGrokConfig()
) -> Tuple[float, str]:
    """
    Predict success probability from partial entropy trace.

    This is the PRACTICAL application: predict if the model will
    succeed BEFORE seeing the final answer.

    Args:
        entropy_trace: Entropy up to current token
        partial_token_idx: How far into generation we are
        config: Detection configuration

    Returns:
        (success_probability, reasoning)
    """
    if len(entropy_trace) < config.min_tokens:
        return 0.5, "insufficient_tokens"

    grok = detect_micro_grokking(entropy_trace, config)

    if grok is None or grok.grok_type == 'none':
        # No grok detected yet - check if entropy is still high
        recent_entropy = np.mean(entropy_trace[-5:])
        initial_entropy = np.mean(entropy_trace[:5])

        if recent_entropy > initial_entropy * 0.9:
            return 0.3, "entropy_not_collapsing"
        else:
            return 0.5, "gradual_progress"

    if grok.grok_type == 'sharp':
        return min(0.95, 0.6 + grok.confidence * 0.4), "sharp_grok_detected"
    elif grok.grok_type == 'gradual':
        return min(0.8, 0.5 + grok.confidence * 0.3), "gradual_grok_detected"
    elif grok.grok_type == 'oscillatory':
        return 0.4, "oscillatory_entropy"
    else:
        return 0.5, "unknown_pattern"


class MicroGrokMonitor:
    """
    Real-time monitor for micro-grokking during inference.

    Use this to:
    1. Stop early on already-grokked runs (save compute)
    2. Allocate more samples when entropy never collapses
    3. Provide real-time success predictions
    """

    def __init__(self, config: MicroGrokConfig = MicroGrokConfig()):
        self.config = config
        self.entropy_history: List[float] = []
        self.grok_detected: bool = False
        self.grok_signal: Optional[GrokSignal] = None

    def reset(self):
        """Reset for new problem."""
        self.entropy_history = []
        self.grok_detected = False
        self.grok_signal = None

    def update(
        self,
        token_logits: NDArray[np.float64],
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """
        Update monitor with new token.

        Args:
            token_logits: [vocab_size] logits for this token
            temperature: Sampling temperature

        Returns:
            Dict with current metrics
        """
        # Compute entropy for this token
        scaled = token_logits / temperature
        max_logit = np.max(scaled)
        exp_logits = np.exp(scaled - max_logit)
        probs = exp_logits / np.sum(exp_logits)
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        self.entropy_history.append(entropy)

        # Check for grok every few tokens
        if len(self.entropy_history) >= self.config.min_tokens:
            if not self.grok_detected:
                trace = np.array(self.entropy_history)
                grok = detect_micro_grokking(trace, self.config)

                if grok and grok.grok_type in ('sharp', 'gradual'):
                    self.grok_detected = True
                    self.grok_signal = grok

        # Predict success
        if len(self.entropy_history) >= self.config.min_tokens:
            trace = np.array(self.entropy_history)
            success_prob, reason = predict_success_from_entropy(trace, len(trace), self.config)
        else:
            success_prob, reason = 0.5, "warming_up"

        return {
            'current_entropy': entropy,
            'mean_entropy': float(np.mean(self.entropy_history)),
            'entropy_trend': float(self.entropy_history[-1] - self.entropy_history[0]) if len(self.entropy_history) > 1 else 0.0,
            'grok_detected': float(self.grok_detected),
            'success_probability': success_prob,
            'n_tokens': len(self.entropy_history)
        }

    def should_stop_early(self, threshold: float = 0.9) -> bool:
        """
        Check if we should stop generation early.

        Returns True if we've detected a strong grok and success is likely.
        """
        if not self.grok_detected or self.grok_signal is None:
            return False

        return (
            self.grok_signal.grok_type == 'sharp' and
            self.grok_signal.confidence > threshold
        )

    def should_allocate_more_samples(self, threshold: float = 0.4) -> bool:
        """
        Check if we should allocate more samples.

        Returns True if entropy isn't collapsing (model is confused).
        """
        if len(self.entropy_history) < self.config.min_tokens * 2:
            return False

        trace = np.array(self.entropy_history)
        recent = trace[-10:]
        initial = trace[:10]

        # Entropy not dropping significantly
        return np.mean(recent) > np.mean(initial) * 0.8


def compute_insight_potential(
    entropy_trace: NDArray[np.float64],
    log_probs: NDArray[np.float64],
    lambda_weight: float = 0.1
) -> NDArray[np.float64]:
    """
    Compute free-energy "insight potential" (Insight #2).

    Φ = entropy - λ·log_p(trajectory)

    Big downward jumps in Φ signal productive compression.
    Flat or upward trajectories signal wasted compute.

    Args:
        entropy_trace: [seq_len] per-token entropy
        log_probs: [seq_len] log probabilities of chosen tokens
        lambda_weight: Weight for log-prob term

    Returns:
        [seq_len] insight potential values
    """
    # Cumulative log probability (trajectory log-prob)
    cumulative_log_prob = np.cumsum(log_probs)

    # Insight potential
    phi = entropy_trace - lambda_weight * cumulative_log_prob

    return phi


def detect_insight_events(
    phi: NDArray[np.float64],
    threshold: float = 0.5
) -> List[Tuple[int, float]]:
    """
    Detect insight events from potential trace.

    Args:
        phi: [seq_len] insight potential values
        threshold: Minimum drop to count as insight

    Returns:
        List of (token_idx, delta_phi) for detected insights
    """
    d_phi = np.diff(phi)

    events = []
    for i, delta in enumerate(d_phi):
        if delta < -threshold:
            events.append((i + 1, float(delta)))

    return events
