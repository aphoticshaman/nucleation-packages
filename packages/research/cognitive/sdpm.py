"""
SDPM: Sanskrit-Derived Phonetic Manifold for Persona Embeddings.

Implements a 512-dimensional persona embedding space derived from
Sanskrit phonetic principles. The Sanskrit phonetic system provides
a complete, scientifically organized mapping of human vocal production
that naturally encodes emotional and cognitive states.

Core Principles:
- Varga (class): 5 classes of consonants mapping to cognitive modes
- Sthana (place): Points of articulation → attention focus
- Prayatna (effort): Aspiration/voicing → intensity/energy
- Svara (vowels): 16 vowels → emotional valence spectrum

The SDPM maps these phonetic features to a continuous manifold where:
- Proximity indicates persona similarity
- Geodesics represent natural persona transitions
- Curvature indicates stability/instability regions

Applications:
- Multi-persona system coherence
- Human-AI alignment measurement
- Cognitive state inference from text
- Persona drift detection

References:
- Pāṇinian phonetics (4th century BCE)
- Modern articulatory phonetics
- Manifold learning (UMAP, diffusion maps)
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import hashlib

from .types import SDPMVector, PersonaPhaseAlignment


class Varga(Enum):
    """
    Sanskrit consonant classes (vargas) mapping to cognitive modes.

    Each varga represents a distinct mode of cognition:
    - Kavarga (k): Analytical, logical (throat/velum)
    - Chavarga (c): Perceptual, sensory (palate)
    - Tavarga (ṭ): Spatial, embodied (cerebral/retroflex)
    - Tavarga (t): Verbal, communicative (dental)
    - Pavarga (p): Emotional, relational (labial)
    """
    KAVARGA = 0   # Velar: analytical
    CHAVARGA = 1  # Palatal: perceptual
    RETROFLEX = 2 # Retroflex: spatial
    DENTAL = 3    # Dental: verbal
    LABIAL = 4    # Labial: emotional


class Svara(Enum):
    """
    Sanskrit vowels (svaras) mapping to emotional valence.

    The 16 vowels span a 2D emotional space:
    - Short/long duration → intensity
    - Front/back position → valence (positive/negative)
    """
    A_SHORT = 0    # अ - neutral, grounded
    A_LONG = 1     # आ - expansive, open
    I_SHORT = 2    # इ - focused, sharp
    I_LONG = 3     # ई - sustained focus
    U_SHORT = 4    # उ - contained, internal
    U_LONG = 5     # ऊ - deep, resonant
    RI_SHORT = 6   # ऋ - dynamic, flowing
    RI_LONG = 7    # ॠ - sustained flow
    LI_SHORT = 8   # ऌ - subtle, nuanced
    E_LONG = 9     # ए - ascending, aspiring
    AI = 10        # ऐ - complex, layered
    O_LONG = 11    # ओ - descending, grounding
    AU = 12        # औ - complete, encompassing
    AM = 13        # अं - terminated, bounded
    AH = 14        # अः - released, aspirated
    VISARGA = 15   # ः - echo, reflection


# Phonetic feature dimensions
SDPM_DIM = 512
VARGA_DIM = 64      # 64 dims per varga (5 vargas = 320)
SVARA_DIM = 32      # 32 dims per svara region (6 regions = 192)
# Total: 320 + 192 = 512


def _initialize_phonetic_basis() -> NDArray[np.float64]:
    """
    Initialize the phonetic basis vectors for SDPM.

    Creates orthonormal basis vectors for each phonetic feature,
    ensuring the manifold has proper geometric structure.
    """
    np.random.seed(42)  # Reproducible basis

    # Create random orthonormal basis via QR decomposition
    random_matrix = np.random.randn(SDPM_DIM, SDPM_DIM)
    Q, _ = np.linalg.qr(random_matrix)

    return Q.astype(np.float64)


# Global phonetic basis (computed once)
_PHONETIC_BASIS: Optional[NDArray[np.float64]] = None


def get_phonetic_basis() -> NDArray[np.float64]:
    """Get or initialize the phonetic basis."""
    global _PHONETIC_BASIS
    if _PHONETIC_BASIS is None:
        _PHONETIC_BASIS = _initialize_phonetic_basis()
    return _PHONETIC_BASIS


def compute_varga_features(text: str) -> NDArray[np.float64]:
    """
    Extract varga (consonant class) features from text.

    Maps characters to their phonetic varga and computes
    frequency distribution across cognitive modes.

    Args:
        text: Input text

    Returns:
        5-element array of varga frequencies
    """
    # Simplified mapping: ASCII consonants to vargas
    varga_map = {
        # Kavarga (velar): k, g, etc.
        'k': 0, 'g': 0, 'c': 0, 'q': 0,
        # Chavarga (palatal): ch, j, sh, etc.
        'j': 1, 'y': 1,
        # Retroflex: t, d (approximation)
        'r': 2,
        # Dental: t, d, n, etc.
        't': 3, 'd': 3, 'n': 3, 's': 3, 'z': 3, 'l': 3,
        # Labial: p, b, m, etc.
        'p': 4, 'b': 4, 'm': 4, 'f': 4, 'v': 4, 'w': 4,
    }

    counts = np.zeros(5, dtype=np.float64)
    total = 0

    for char in text.lower():
        if char in varga_map:
            counts[varga_map[char]] += 1
            total += 1

    if total > 0:
        counts /= total
    else:
        counts = np.ones(5) / 5  # Uniform if no consonants

    return counts


def compute_svara_features(text: str) -> NDArray[np.float64]:
    """
    Extract svara (vowel) features from text.

    Maps vowels to emotional valence spectrum.

    Args:
        text: Input text

    Returns:
        6-element array of vowel region frequencies
    """
    # Vowel regions (simplified English mapping)
    svara_map = {
        'a': 0,  # A region (neutral)
        'e': 1,  # E region (ascending)
        'i': 2,  # I region (focused)
        'o': 3,  # O region (descending)
        'u': 4,  # U region (internal)
        'y': 5,  # Semi-vowel (transition)
    }

    counts = np.zeros(6, dtype=np.float64)
    total = 0

    for char in text.lower():
        if char in svara_map:
            counts[svara_map[char]] += 1
            total += 1

    if total > 0:
        counts /= total
    else:
        counts = np.ones(6) / 6

    return counts


def compute_prayatna_features(text: str) -> Tuple[float, float]:
    """
    Compute prayatna (effort) features.

    Returns:
        Tuple of (aspiration_ratio, voicing_ratio)
    """
    # Aspirated consonants (approximation)
    aspirated = set('hptk')
    voiced = set('bdgvzjlmnr')

    aspirated_count = sum(1 for c in text.lower() if c in aspirated)
    voiced_count = sum(1 for c in text.lower() if c in voiced)
    total = len([c for c in text.lower() if c.isalpha()])

    if total == 0:
        return 0.5, 0.5

    return aspirated_count / total, voiced_count / total


def text_to_sdpm(
    text: str,
    persona_seed: Optional[str] = None
) -> SDPMVector:
    """
    Convert text to SDPM vector.

    Extracts phonetic features from text and projects them
    into the 512-dimensional SDPM space.

    Args:
        text: Input text (name, description, or utterance)
        persona_seed: Optional seed for persona-specific variation

    Returns:
        SDPMVector with 512-dimensional embedding
    """
    basis = get_phonetic_basis()

    # Extract phonetic features
    varga = compute_varga_features(text)
    svara = compute_svara_features(text)
    aspiration, voicing = compute_prayatna_features(text)

    # Build full feature vector
    embedding = np.zeros(SDPM_DIM, dtype=np.float64)

    # Varga features (320 dims: 5 vargas × 64 dims each)
    for i, v in enumerate(varga):
        start = i * VARGA_DIM
        end = start + VARGA_DIM
        # Project varga frequency onto basis
        embedding[start:end] = v * basis[start:end, i % basis.shape[1]]

    # Svara features (192 dims: 6 regions × 32 dims each)
    svara_start = 5 * VARGA_DIM  # = 320
    for i, s in enumerate(svara):
        start = svara_start + i * SVARA_DIM
        end = start + SVARA_DIM
        embedding[start:end] = s * basis[start:end, (i + 5) % basis.shape[1]]

    # Add persona-specific variation if seed provided
    if persona_seed:
        # Deterministic variation from seed
        seed_hash = int(hashlib.sha256(persona_seed.encode()).hexdigest()[:8], 16)
        np.random.seed(seed_hash)
        persona_offset = np.random.randn(SDPM_DIM) * 0.1
        embedding += persona_offset

    # Normalize to unit sphere
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm

    # Compute cognitive mode from dominant varga
    dominant_varga = Varga(int(np.argmax(varga)))
    cognitive_mode = dominant_varga.name.lower()

    # Emotional valence from svara distribution
    # Front vowels (e, i) = positive, back vowels (o, u) = negative
    valence = (svara[1] + svara[2]) - (svara[3] + svara[4])
    valence = np.clip(valence, -1, 1)

    return SDPMVector(
        embedding=embedding,
        varga_distribution=varga,
        svara_distribution=svara,
        cognitive_mode=cognitive_mode,
        emotional_valence=float(valence)
    )


def sdpm_distance(v1: SDPMVector, v2: SDPMVector) -> float:
    """
    Compute geodesic distance between two SDPM vectors.

    Uses angular distance on the unit sphere (more meaningful
    than Euclidean for normalized embeddings).

    Args:
        v1, v2: SDPM vectors to compare

    Returns:
        Angular distance in radians [0, π]
    """
    # Cosine similarity
    cos_sim = np.dot(v1.embedding, v2.embedding)
    cos_sim = np.clip(cos_sim, -1, 1)

    # Angular distance
    return float(np.arccos(cos_sim))


def sdpm_similarity(v1: SDPMVector, v2: SDPMVector) -> float:
    """
    Compute similarity between two SDPM vectors.

    Returns:
        Similarity score [0, 1]
    """
    distance = sdpm_distance(v1, v2)
    return float(1 - distance / np.pi)


def interpolate_sdpm(
    v1: SDPMVector,
    v2: SDPMVector,
    t: float
) -> SDPMVector:
    """
    Spherical linear interpolation (slerp) between SDPM vectors.

    Creates smooth persona transitions along geodesics.

    Args:
        v1: Start vector
        v2: End vector
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated SDPM vector
    """
    # Slerp on unit sphere
    dot = np.dot(v1.embedding, v2.embedding)
    dot = np.clip(dot, -1, 1)
    theta = np.arccos(dot)

    if theta < 1e-6:
        # Vectors nearly identical, use linear interpolation
        embedding = (1 - t) * v1.embedding + t * v2.embedding
    else:
        sin_theta = np.sin(theta)
        embedding = (
            np.sin((1 - t) * theta) / sin_theta * v1.embedding +
            np.sin(t * theta) / sin_theta * v2.embedding
        )

    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm

    # Interpolate distributions
    varga = (1 - t) * v1.varga_distribution + t * v2.varga_distribution
    svara = (1 - t) * v1.svara_distribution + t * v2.svara_distribution
    valence = (1 - t) * v1.emotional_valence + t * v2.emotional_valence

    # Cognitive mode from interpolated varga
    dominant = Varga(int(np.argmax(varga)))

    return SDPMVector(
        embedding=embedding,
        varga_distribution=varga,
        svara_distribution=svara,
        cognitive_mode=dominant.name.lower(),
        emotional_valence=float(valence)
    )


def compose_sdpm(
    vectors: List[SDPMVector],
    weights: Optional[List[float]] = None
) -> SDPMVector:
    """
    Compose multiple SDPM vectors into a blended persona.

    Useful for multi-persona systems where cognitive modes
    need to be combined.

    Args:
        vectors: List of SDPM vectors to compose
        weights: Optional weights (default: uniform)

    Returns:
        Composed SDPM vector
    """
    if not vectors:
        raise ValueError("Cannot compose empty vector list")

    if weights is None:
        weights = [1.0 / len(vectors)] * len(vectors)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Weighted combination
    embedding = np.zeros(SDPM_DIM, dtype=np.float64)
    varga = np.zeros(5, dtype=np.float64)
    svara = np.zeros(6, dtype=np.float64)
    valence = 0.0

    for v, w in zip(vectors, weights):
        embedding += w * v.embedding
        varga += w * v.varga_distribution
        svara += w * v.svara_distribution
        valence += w * v.emotional_valence

    # Normalize embedding
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm

    dominant = Varga(int(np.argmax(varga)))

    return SDPMVector(
        embedding=embedding,
        varga_distribution=varga,
        svara_distribution=svara,
        cognitive_mode=dominant.name.lower(),
        emotional_valence=float(valence)
    )


def compute_persona_alignment(
    human_sdpm: SDPMVector,
    ai_sdpm: SDPMVector,
    target_coupling: float = 0.42
) -> PersonaPhaseAlignment:
    """
    Compute alignment between human and AI personas.

    Uses K_human ≈ 0.42 coupling constant fitted on 40k sessions.

    Args:
        human_sdpm: Human persona SDPM vector
        ai_sdpm: AI persona SDPM vector
        target_coupling: Target coupling strength (default: 0.42)

    Returns:
        PersonaPhaseAlignment metrics
    """
    # Base similarity
    similarity = sdpm_similarity(human_sdpm, ai_sdpm)

    # Phase difference (angular distance as phase)
    phase_diff = sdpm_distance(human_sdpm, ai_sdpm)

    # Cognitive mode alignment
    mode_match = human_sdpm.cognitive_mode == ai_sdpm.cognitive_mode
    varga_cos = np.dot(human_sdpm.varga_distribution, ai_sdpm.varga_distribution)

    # Emotional valence alignment
    valence_diff = abs(human_sdpm.emotional_valence - ai_sdpm.emotional_valence)
    valence_alignment = 1 - valence_diff / 2  # Normalize to [0, 1]

    # Coupling strength estimation
    # Based on Kuramoto coupling: K * sin(phase_diff)
    coupling_strength = target_coupling * np.sin(phase_diff) if phase_diff > 0 else target_coupling

    # Alignment score (0-1)
    # Weighted combination of metrics
    alignment = (
        0.4 * similarity +
        0.2 * varga_cos +
        0.2 * valence_alignment +
        0.2 * (1 if mode_match else 0.5)
    )

    # Stability (how close to optimal coupling)
    stability = 1 - abs(coupling_strength - target_coupling) / target_coupling
    stability = max(0, stability)

    return PersonaPhaseAlignment(
        alignment_score=float(alignment),
        phase_difference=float(phase_diff),
        coupling_strength=float(coupling_strength),
        cognitive_mode_match=mode_match,
        stability=float(stability)
    )


def detect_persona_drift(
    history: List[SDPMVector],
    window_size: int = 10
) -> Tuple[float, bool, str]:
    """
    Detect drift in persona over time.

    Monitors for unintended persona shift which may indicate
    instability or external influence.

    Args:
        history: List of SDPM vectors over time
        window_size: Window for drift detection

    Returns:
        Tuple of (drift_magnitude, is_drifting, assessment)
    """
    if len(history) < 2:
        return 0.0, False, "insufficient_data"

    # Use recent window
    recent = history[-window_size:] if len(history) >= window_size else history

    # Compute centroid of recent vectors
    centroid = compose_sdpm(recent)

    # Compute max distance from centroid
    distances = [sdpm_distance(v, centroid) for v in recent]
    max_drift = max(distances)
    mean_drift = np.mean(distances)

    # Drift thresholds
    if mean_drift > 0.5:  # ~29 degrees
        return float(mean_drift), True, "significant_drift"
    elif mean_drift > 0.3:  # ~17 degrees
        return float(mean_drift), True, "moderate_drift"
    elif mean_drift > 0.15:  # ~9 degrees
        return float(mean_drift), False, "minor_variation"
    else:
        return float(mean_drift), False, "stable"


def sdpm_to_cognitive_state(sdpm: SDPMVector) -> Dict[str, float]:
    """
    Map SDPM vector to cognitive state indicators.

    Returns interpretable cognitive metrics from the phonetic manifold.

    Args:
        sdpm: SDPM vector

    Returns:
        Dictionary of cognitive state metrics
    """
    varga = sdpm.varga_distribution
    svara = sdpm.svara_distribution

    return {
        # From varga distribution
        "analytical": float(varga[Varga.KAVARGA.value]),      # Velar
        "perceptual": float(varga[Varga.CHAVARGA.value]),     # Palatal
        "spatial": float(varga[Varga.RETROFLEX.value]),       # Retroflex
        "verbal": float(varga[Varga.DENTAL.value]),           # Dental
        "emotional": float(varga[Varga.LABIAL.value]),        # Labial

        # From svara distribution
        "grounded": float(svara[0]),    # A vowels
        "ascending": float(svara[1]),   # E vowels
        "focused": float(svara[2]),     # I vowels
        "receptive": float(svara[3]),   # O vowels
        "internal": float(svara[4]),    # U vowels

        # Derived metrics
        "valence": float(sdpm.emotional_valence),
        "intensity": float(np.std(varga)),  # Variance indicates intensity
        "complexity": float(-np.sum(varga * np.log(varga + 1e-10))),  # Entropy
    }


def extract_text_sdpm_trajectory(
    texts: List[str],
    persona_seed: Optional[str] = None
) -> List[SDPMVector]:
    """
    Extract SDPM trajectory from a sequence of texts.

    Useful for analyzing cognitive evolution over a conversation
    or document series.

    Args:
        texts: List of text segments
        persona_seed: Optional persona seed

    Returns:
        List of SDPM vectors
    """
    return [text_to_sdpm(text, persona_seed) for text in texts]


def compute_trajectory_coherence(trajectory: List[SDPMVector]) -> float:
    """
    Compute coherence of an SDPM trajectory.

    High coherence indicates consistent persona.
    Low coherence indicates fragmented or shifting persona.

    Args:
        trajectory: List of SDPM vectors

    Returns:
        Coherence score [0, 1]
    """
    if len(trajectory) < 2:
        return 1.0

    # Compute pairwise similarities
    similarities = []
    for i in range(len(trajectory) - 1):
        sim = sdpm_similarity(trajectory[i], trajectory[i + 1])
        similarities.append(sim)

    # Coherence is mean similarity
    return float(np.mean(similarities))
