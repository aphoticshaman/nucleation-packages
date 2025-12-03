"""
NSM: Neural State Model Phase Head.

Implements phase extraction from transformer hidden states for cognitive
state monitoring. The NSM maps high-dimensional neural activations to
interpretable phase representations suitable for Kuramoto analysis.

Architecture:
    Hidden States → Phase Projection → Phase Normalization → Oscillator Phases
    [batch, seq, d_model] → [batch, seq, n_oscillators] → [batch, seq, n_oscillators]

The phase head can be:
1. Trained end-to-end with the main model
2. Post-hoc fitted to existing hidden states
3. Used as a diagnostic probe without training

Key Components:
- Phase projection layer: d_model → n_oscillators
- Phase normalization: Ensures phases ∈ [0, 2π)
- Temporal smoothing: Optional low-pass for stable phases
- Hilbert transform: For analytic signal phase extraction

Applications:
- Real-time cognitive monitoring during inference
- Flow state prediction from attention patterns
- Coherence tracking across layers
- Layer-wise phase alignment analysis

References:
- Transformer attention patterns as oscillator coupling
- Hilbert transform for instantaneous phase
- Multi-scale phase analysis
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
from scipy.signal import hilbert
import warnings


@dataclass
class NSMConfig:
    """Configuration for Neural State Model phase head."""
    d_model: int = 768           # Hidden dimension
    n_oscillators: int = 64      # Number of phase oscillators
    n_layers: int = 12           # Number of transformer layers
    temporal_smooth: bool = True # Apply temporal smoothing
    smooth_window: int = 5       # Smoothing window size
    use_hilbert: bool = True     # Use Hilbert transform for phase


@dataclass
class NSMOutput:
    """Output from NSM phase extraction."""
    phases: NDArray[np.float64]           # [seq, n_oscillators] phase values
    amplitudes: NDArray[np.float64]       # [seq, n_oscillators] amplitude values
    order_parameter: float                 # Kuramoto R
    mean_phase: float                      # Kuramoto Ψ
    layer_coherence: Optional[NDArray[np.float64]] = None  # Per-layer coherence


class NSMPhaseHead:
    """
    Neural State Model phase extraction head.

    Extracts phase representations from transformer hidden states
    for cognitive state analysis.
    """

    def __init__(self, config: Optional[NSMConfig] = None):
        """
        Initialize NSM phase head.

        Args:
            config: NSM configuration
        """
        self.config = config or NSMConfig()

        # Initialize projection matrix (random orthogonal)
        np.random.seed(42)
        random_matrix = np.random.randn(self.config.d_model, self.config.n_oscillators)
        self.projection, _ = np.linalg.qr(random_matrix)

        # Temporal smoothing kernel
        if self.config.temporal_smooth:
            self.smooth_kernel = np.ones(self.config.smooth_window) / self.config.smooth_window

    def extract_phases(
        self,
        hidden_states: NDArray[np.float64],
        layer_idx: Optional[int] = None
    ) -> NSMOutput:
        """
        Extract phase representation from hidden states.

        Args:
            hidden_states: [seq_len, d_model] or [batch, seq_len, d_model]
            layer_idx: Optional layer index for per-layer analysis

        Returns:
            NSMOutput with phases, amplitudes, and coherence metrics
        """
        # Handle batch dimension
        if hidden_states.ndim == 3:
            # Process batch dimension - take mean for now
            hidden_states = np.mean(hidden_states, axis=0)

        seq_len = hidden_states.shape[0]

        # Project to oscillator space
        projected = hidden_states @ self.projection  # [seq, n_oscillators]

        if self.config.use_hilbert:
            # Use Hilbert transform for analytic signal
            phases, amplitudes = self._hilbert_phase(projected)
        else:
            # Direct phase extraction via atan2
            phases, amplitudes = self._direct_phase(projected)

        # Temporal smoothing
        if self.config.temporal_smooth and seq_len > self.config.smooth_window:
            phases = self._smooth_phases(phases)

        # Compute order parameter from final sequence position
        R, Psi = self._compute_order_parameter(phases[-1])

        return NSMOutput(
            phases=phases,
            amplitudes=amplitudes,
            order_parameter=float(R),
            mean_phase=float(Psi),
            layer_coherence=None  # Set when processing multiple layers
        )

    def extract_layer_phases(
        self,
        all_hidden_states: List[NDArray[np.float64]]
    ) -> Tuple[List[NSMOutput], NDArray[np.float64]]:
        """
        Extract phases from all layers.

        Args:
            all_hidden_states: List of hidden states per layer

        Returns:
            Tuple of (per-layer outputs, layer coherence matrix)
        """
        outputs = []
        layer_order_params = []

        for layer_idx, hidden in enumerate(all_hidden_states):
            output = self.extract_phases(hidden, layer_idx)
            outputs.append(output)
            layer_order_params.append(output.order_parameter)

        # Compute layer coherence matrix
        n_layers = len(outputs)
        coherence_matrix = np.zeros((n_layers, n_layers))

        for i in range(n_layers):
            for j in range(n_layers):
                # Phase coherence between layers
                coherence_matrix[i, j] = self._compute_phase_coherence(
                    outputs[i].phases[-1],
                    outputs[j].phases[-1]
                )

        return outputs, coherence_matrix

    def _hilbert_phase(
        self,
        signal: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Extract phase using Hilbert transform.

        Args:
            signal: [seq, n_oscillators] real-valued signal

        Returns:
            Tuple of (phases, amplitudes)
        """
        seq_len, n_osc = signal.shape
        phases = np.zeros_like(signal)
        amplitudes = np.zeros_like(signal)

        for i in range(n_osc):
            # Hilbert transform for analytic signal
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                analytic = hilbert(signal[:, i])

            # Phase from analytic signal
            phases[:, i] = np.angle(analytic)
            # Amplitude (envelope)
            amplitudes[:, i] = np.abs(analytic)

        # Normalize phases to [0, 2π)
        phases = np.mod(phases, 2 * np.pi)

        return phases, amplitudes

    def _direct_phase(
        self,
        signal: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Extract phase using direct method (pairs of components).

        Args:
            signal: [seq, n_oscillators] signal

        Returns:
            Tuple of (phases, amplitudes)
        """
        seq_len, n_osc = signal.shape

        # Treat pairs of oscillators as complex numbers
        n_pairs = n_osc // 2
        phases = np.zeros((seq_len, n_pairs))
        amplitudes = np.zeros((seq_len, n_pairs))

        for i in range(n_pairs):
            real = signal[:, 2 * i]
            imag = signal[:, 2 * i + 1]

            phases[:, i] = np.arctan2(imag, real)
            amplitudes[:, i] = np.sqrt(real ** 2 + imag ** 2)

        # Normalize to [0, 2π)
        phases = np.mod(phases, 2 * np.pi)

        return phases, amplitudes

    def _smooth_phases(
        self,
        phases: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Apply circular smoothing to phases.

        Uses circular mean to properly handle phase wraparound.

        Args:
            phases: [seq, n_oscillators] phase values

        Returns:
            Smoothed phases
        """
        seq_len, n_osc = phases.shape
        smoothed = np.zeros_like(phases)
        half_window = self.config.smooth_window // 2

        for t in range(seq_len):
            start = max(0, t - half_window)
            end = min(seq_len, t + half_window + 1)

            # Circular mean for each oscillator
            for i in range(n_osc):
                window_phases = phases[start:end, i]
                # Circular mean using complex exponential
                mean_complex = np.mean(np.exp(1j * window_phases))
                smoothed[t, i] = np.angle(mean_complex)

        return np.mod(smoothed, 2 * np.pi)

    def _compute_order_parameter(
        self,
        phases: NDArray[np.float64]
    ) -> Tuple[float, float]:
        """
        Compute Kuramoto order parameter from phases.

        Args:
            phases: [n_oscillators] phase values

        Returns:
            Tuple of (R, Psi)
        """
        # Complex order parameter
        z = np.mean(np.exp(1j * phases))

        R = np.abs(z)
        Psi = np.angle(z)

        return float(R), float(Psi)

    def _compute_phase_coherence(
        self,
        phases1: NDArray[np.float64],
        phases2: NDArray[np.float64]
    ) -> float:
        """
        Compute phase coherence between two phase arrays.

        Args:
            phases1, phases2: Phase arrays to compare

        Returns:
            Coherence value [0, 1]
        """
        # Use phase locking value (PLV)
        phase_diff = phases1 - phases2
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))

        return float(plv)

    def fit_projection(
        self,
        hidden_states: NDArray[np.float64],
        target_phases: Optional[NDArray[np.float64]] = None,
        method: str = "pca"
    ) -> None:
        """
        Fit the projection matrix to hidden states.

        Args:
            hidden_states: Training hidden states [n_samples, d_model]
            target_phases: Optional target phases for supervised fitting
            method: "pca", "ica", or "supervised"
        """
        if method == "pca":
            # PCA-based projection
            self._fit_pca(hidden_states)
        elif method == "ica":
            # ICA-based projection for independent components
            self._fit_ica(hidden_states)
        elif method == "supervised" and target_phases is not None:
            # Supervised fitting to target phases
            self._fit_supervised(hidden_states, target_phases)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _fit_pca(self, hidden_states: NDArray[np.float64]) -> None:
        """Fit projection using PCA."""
        # Center the data
        mean = np.mean(hidden_states, axis=0)
        centered = hidden_states - mean

        # SVD for PCA
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # Take top n_oscillators components
        self.projection = Vt[:self.config.n_oscillators].T

    def _fit_ica(self, hidden_states: NDArray[np.float64]) -> None:
        """Fit projection using FastICA-like approach."""
        # Simple whitening + rotation
        # (Full ICA would require iterative optimization)

        # Whitening via PCA
        mean = np.mean(hidden_states, axis=0)
        centered = hidden_states - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Whitening matrix
        D = np.diag(1.0 / np.sqrt(eigenvalues[idx] + 1e-10))
        whitening = eigenvectors @ D

        # Use top components for projection
        self.projection = whitening[:, :self.config.n_oscillators]

    def _fit_supervised(
        self,
        hidden_states: NDArray[np.float64],
        target_phases: NDArray[np.float64]
    ) -> None:
        """Fit projection to target phases using least squares."""
        # Convert target phases to complex representation
        target_complex = np.exp(1j * target_phases)
        target_2d = np.column_stack([
            np.real(target_complex).flatten(),
            np.imag(target_complex).flatten()
        ])

        # Least squares fit
        projection, _, _, _ = np.linalg.lstsq(
            hidden_states,
            target_2d,
            rcond=None
        )

        self.projection = projection


def extract_attention_phases(
    attention_weights: NDArray[np.float64],
    n_oscillators: int = 64
) -> NDArray[np.float64]:
    """
    Extract phases from attention weight patterns.

    Attention patterns can be interpreted as coupling strengths
    in a Kuramoto-like model of information flow.

    Args:
        attention_weights: [n_heads, seq, seq] attention weights
        n_oscillators: Number of phase oscillators

    Returns:
        [seq, n_oscillators] phase values
    """
    n_heads, seq_len, _ = attention_weights.shape

    # Each head contributes to oscillator phases
    # Group heads into oscillators
    heads_per_osc = max(1, n_heads // n_oscillators)

    phases = np.zeros((seq_len, n_oscillators))

    for osc_idx in range(n_oscillators):
        head_start = osc_idx * heads_per_osc
        head_end = min(head_start + heads_per_osc, n_heads)

        if head_start >= n_heads:
            break

        # Average attention over assigned heads
        head_attn = np.mean(attention_weights[head_start:head_end], axis=0)

        # Phase from attention gradient
        # High attention to recent tokens → low phase
        # Distributed attention → higher phase
        for t in range(seq_len):
            # Entropy of attention distribution as phase
            attn_row = head_attn[t] + 1e-10
            attn_row = attn_row / attn_row.sum()
            entropy = -np.sum(attn_row * np.log(attn_row))

            # Map entropy to phase [0, 2π)
            max_entropy = np.log(seq_len)
            phases[t, osc_idx] = 2 * np.pi * entropy / max_entropy

    return phases


def compute_layer_phase_alignment(
    layer_outputs: List[NSMOutput]
) -> Dict[str, float]:
    """
    Compute phase alignment metrics across layers.

    Args:
        layer_outputs: NSMOutput from each layer

    Returns:
        Dictionary of alignment metrics
    """
    n_layers = len(layer_outputs)

    if n_layers < 2:
        return {"status": "insufficient_layers"}

    # Collect order parameters
    order_params = np.array([o.order_parameter for o in layer_outputs])

    # Layer-to-layer phase coherence
    coherences = []
    for i in range(n_layers - 1):
        phases_i = layer_outputs[i].phases[-1]
        phases_j = layer_outputs[i + 1].phases[-1]

        # PLV between adjacent layers
        phase_diff = phases_i[:len(phases_j)] - phases_j[:len(phases_i)]
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        coherences.append(plv)

    return {
        "mean_order_param": float(np.mean(order_params)),
        "std_order_param": float(np.std(order_params)),
        "order_param_trend": float(order_params[-1] - order_params[0]),
        "mean_layer_coherence": float(np.mean(coherences)),
        "min_layer_coherence": float(np.min(coherences)),
        "max_layer_coherence": float(np.max(coherences)),
        "bottleneck_layer": int(np.argmin(coherences)),  # Layer with lowest coherence
        "peak_layer": int(np.argmax(order_params)),  # Layer with highest R
    }


def create_nsm_probe(
    d_model: int,
    n_oscillators: int = 64,
    pretrained_projection: Optional[NDArray[np.float64]] = None
) -> NSMPhaseHead:
    """
    Create an NSM probe for cognitive monitoring.

    Args:
        d_model: Model hidden dimension
        n_oscillators: Number of phase oscillators
        pretrained_projection: Optional pretrained projection matrix

    Returns:
        Configured NSMPhaseHead
    """
    config = NSMConfig(
        d_model=d_model,
        n_oscillators=n_oscillators,
        temporal_smooth=True,
        use_hilbert=True
    )

    nsm = NSMPhaseHead(config)

    if pretrained_projection is not None:
        nsm.projection = pretrained_projection

    return nsm
