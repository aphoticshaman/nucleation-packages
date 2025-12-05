"""
Phase-Coded Ensemble Voting.

Implements Insight #7: Phase-coded ensembles unify majority vote, debate, and self-consistency.

IMPORTANT: Phase assignment must be based on ANSWER AGREEMENT, not text similarity.
- Same answer → same phase (constructive interference)
- Different answers → orthogonal phases (no systematic interference)
- Explicit contradictions → opposite phases (destructive interference)

Using TF-IDF/embedding similarity for phase is WRONG because it measures
text similarity, not logical opposition. Two phrasings of "answer is 42"
would get different phases, while "yes" and "no" might get the same phase.

The CORRECT approach:
1. Group by extracted answer (simple, deployable)
2. Use explicit contradiction detection (hard, needs LLM judge)
3. Use debate-style prompts that assign stance explicitly

For most use cases, just use entropy_voting.py instead - it's simpler
and captures the real value (confidence-weighted voting) without the
broken interference math.

Based on:
- Kuramoto oscillator phase synchronization theory
- Fractal Cascade phase interference formalization
- Dempster-Shafer belief combination (extended to complex domain)
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
import re


class SampleStance(Enum):
    """Stance of a sample relative to a candidate answer."""
    SUPPORT = 0          # Phase = 0 (constructive)
    REFUTE = 1           # Phase = π (destructive)
    ALTERNATIVE = 2      # Phase = ±π/2 (orthogonal)
    IMPROVE = 3          # Phase = π/4 (supportive modification)
    NEUTRAL = 4          # Phase = random (no alignment)


@dataclass
class PhaseSample:
    """A sample with phase-coded stance."""
    content: str                    # The generated text
    answer: str                     # Extracted answer
    confidence: float               # Model confidence [0, 1]
    stance: SampleStance            # Stance relative to primary
    phase: float                    # Complex phase in radians
    amplitude: float                # Complex amplitude (confidence-weighted)
    metadata: Optional[Dict] = None


@dataclass
class PhaseEnsembleConfig:
    """Configuration for phase-coded ensemble."""
    support_phase: float = 0.0           # Phase for support
    refute_phase: float = np.pi          # Phase for refute
    alternative_phase: float = np.pi / 2  # Phase for alternatives
    improve_phase: float = np.pi / 4      # Phase for improvements
    min_confidence: float = 0.1           # Minimum confidence to include
    interference_threshold: float = 0.3   # Threshold for destructive interference


def stance_to_phase(
    stance: SampleStance,
    config: PhaseEnsembleConfig = PhaseEnsembleConfig()
) -> float:
    """
    Map stance to complex phase.

    Args:
        stance: Sample stance
        config: Ensemble configuration

    Returns:
        Phase in radians
    """
    phase_map = {
        SampleStance.SUPPORT: config.support_phase,
        SampleStance.REFUTE: config.refute_phase,
        SampleStance.ALTERNATIVE: config.alternative_phase,
        SampleStance.IMPROVE: config.improve_phase,
        SampleStance.NEUTRAL: np.random.uniform(0, 2 * np.pi)
    }
    return phase_map[stance]


def classify_stance(
    sample: str,
    reference: str,
    classifier: Optional[Callable[[str, str], SampleStance]] = None
) -> SampleStance:
    """
    Classify the stance of a sample relative to a reference answer.

    Default implementation uses simple heuristics. Override with LLM classifier
    for production use.

    Args:
        sample: Generated sample text
        reference: Reference answer to compare against
        classifier: Optional custom classifier function

    Returns:
        Classified stance
    """
    if classifier is not None:
        return classifier(sample, reference)

    # Simple heuristic classification
    sample_lower = sample.lower()

    # Check for refutation markers
    refute_markers = ['however', 'but', 'incorrect', 'wrong', 'disagree', 'actually', 'not quite']
    for marker in refute_markers:
        if marker in sample_lower:
            return SampleStance.REFUTE

    # Check for improvement markers
    improve_markers = ['also', 'additionally', 'moreover', 'building on', 'extending']
    for marker in improve_markers:
        if marker in sample_lower:
            return SampleStance.IMPROVE

    # Check for alternative markers
    alt_markers = ['alternatively', 'another approach', 'different way', 'instead']
    for marker in alt_markers:
        if marker in sample_lower:
            return SampleStance.ALTERNATIVE

    # Check for explicit agreement
    support_markers = ['agree', 'correct', 'right', 'yes', 'exactly', 'confirms']
    for marker in support_markers:
        if marker in sample_lower:
            return SampleStance.SUPPORT

    # Default: check semantic similarity (simplified)
    # In production, use embedding similarity
    return SampleStance.SUPPORT


def extract_answer(
    sample: str,
    pattern: str = r'(?:answer|result|solution)[\s:]*([^\n]+)'
) -> str:
    """
    Extract answer from sample text.

    Args:
        sample: Generated text
        pattern: Regex pattern for answer extraction

    Returns:
        Extracted answer or full sample if no match
    """
    match = re.search(pattern, sample, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try boxed answer format (common in math)
    boxed = re.search(r'\\boxed\{([^}]+)\}', sample)
    if boxed:
        return boxed.group(1).strip()

    # Return last line as answer
    lines = sample.strip().split('\n')
    return lines[-1].strip() if lines else sample


def create_phase_sample(
    content: str,
    reference_answer: Optional[str] = None,
    confidence: float = 1.0,
    stance_classifier: Optional[Callable] = None,
    config: PhaseEnsembleConfig = PhaseEnsembleConfig()
) -> PhaseSample:
    """
    Create a phase-coded sample from raw generation.

    Args:
        content: Generated text
        reference_answer: Reference answer for stance classification
        confidence: Model confidence (from log-probs or explicit)
        stance_classifier: Optional custom stance classifier
        config: Ensemble configuration

    Returns:
        PhaseSample with computed phase and amplitude
    """
    answer = extract_answer(content)

    if reference_answer is not None:
        stance = classify_stance(content, reference_answer, stance_classifier)
    else:
        stance = SampleStance.SUPPORT

    phase = stance_to_phase(stance, config)
    amplitude = max(config.min_confidence, confidence)

    return PhaseSample(
        content=content,
        answer=answer,
        confidence=confidence,
        stance=stance,
        phase=phase,
        amplitude=amplitude
    )


def phase_interference_vote(
    samples: List[PhaseSample],
    config: PhaseEnsembleConfig = PhaseEnsembleConfig()
) -> Tuple[str, float, Dict[str, complex]]:
    """
    Perform phase-interference voting across samples.

    Sum amplitudes per candidate answer as complex numbers;
    select by |sum|² (interference pattern).

    Constructive interference (aligned phases) → stronger signal
    Destructive interference (opposed phases) → weaker signal

    Args:
        samples: List of phase-coded samples
        config: Ensemble configuration

    Returns:
        (winning_answer, confidence, all_amplitudes_dict)
    """
    # Group samples by answer
    answer_amplitudes: Dict[str, complex] = {}

    for sample in samples:
        # Convert to complex amplitude
        z = sample.amplitude * np.exp(1j * sample.phase)

        if sample.answer in answer_amplitudes:
            answer_amplitudes[sample.answer] += z
        else:
            answer_amplitudes[sample.answer] = z

    # Select by |sum|² (intensity)
    intensities = {ans: abs(z) ** 2 for ans, z in answer_amplitudes.items()}

    if not intensities:
        return "", 0.0, {}

    winner = max(intensities, key=intensities.get)
    total_intensity = sum(intensities.values())
    confidence = intensities[winner] / total_intensity if total_intensity > 0 else 0.0

    return winner, confidence, answer_amplitudes


def detect_destructive_interference(
    answer_amplitudes: Dict[str, complex],
    config: PhaseEnsembleConfig = PhaseEnsembleConfig()
) -> List[str]:
    """
    Detect answers suffering from destructive interference.

    These are answers that have high individual support but cancel out
    due to refutations.

    Args:
        answer_amplitudes: Complex amplitudes per answer
        config: Ensemble configuration

    Returns:
        List of answers with significant destructive interference
    """
    destructive = []

    for answer, z in answer_amplitudes.items():
        # Compare |sum| to sum of |components|
        # If |sum| << sum(|components|), there's destructive interference
        intensity = abs(z)

        # We'd need individual components to compute this properly
        # For now, check if intensity is very low despite being in the dict
        if intensity < config.interference_threshold:
            destructive.append(answer)

    return destructive


class PhaseEnsemble:
    """
    Phase-coded ensemble aggregator.

    Usage:
        ensemble = PhaseEnsemble()

        # Generate samples with different prompts
        for prompt_type in ['propose', 'critique', 'improve']:
            response = generate(problem, prompt_type)
            ensemble.add_sample(response, prompt_type=prompt_type)

        # Get phase-interference result
        answer, confidence = ensemble.vote()
    """

    def __init__(self, config: PhaseEnsembleConfig = PhaseEnsembleConfig()):
        self.config = config
        self.samples: List[PhaseSample] = []
        self.reference_answer: Optional[str] = None

    def reset(self):
        """Reset ensemble for new problem."""
        self.samples = []
        self.reference_answer = None

    def add_sample(
        self,
        content: str,
        confidence: float = 1.0,
        prompt_type: str = 'propose',
        stance: Optional[SampleStance] = None
    ):
        """
        Add a sample to the ensemble.

        Args:
            content: Generated text
            confidence: Model confidence
            prompt_type: Type of prompt used ('propose', 'critique', 'improve')
            stance: Optional explicit stance override
        """
        # Map prompt type to default stance
        prompt_stance_map = {
            'propose': SampleStance.SUPPORT,
            'critique': SampleStance.REFUTE,
            'improve': SampleStance.IMPROVE,
            'alternative': SampleStance.ALTERNATIVE,
            'verify': SampleStance.SUPPORT
        }

        if stance is None:
            stance = prompt_stance_map.get(prompt_type, SampleStance.SUPPORT)

        # If we have a reference, classify against it
        if self.reference_answer is not None and stance == SampleStance.SUPPORT:
            stance = classify_stance(content, self.reference_answer)

        phase = stance_to_phase(stance, self.config)

        sample = PhaseSample(
            content=content,
            answer=extract_answer(content),
            confidence=confidence,
            stance=stance,
            phase=phase,
            amplitude=max(self.config.min_confidence, confidence),
            metadata={'prompt_type': prompt_type}
        )

        # Set first proposal as reference
        if self.reference_answer is None and prompt_type == 'propose':
            self.reference_answer = sample.answer

        self.samples.append(sample)

    def vote(self) -> Tuple[str, float]:
        """
        Perform phase-interference voting.

        Returns:
            (winning_answer, confidence)
        """
        winner, confidence, _ = phase_interference_vote(self.samples, self.config)
        return winner, confidence

    def get_interference_analysis(self) -> Dict:
        """
        Get detailed interference analysis.

        Returns:
            Dictionary with interference metrics per answer
        """
        _, _, amplitudes = phase_interference_vote(self.samples, self.config)

        analysis = {}
        for answer, z in amplitudes.items():
            # Count samples supporting this answer
            supporting = [s for s in self.samples if s.answer == answer]
            n_samples = len(supporting)

            # Phases of supporting samples
            phases = [s.phase for s in supporting]

            # Phase coherence (Kuramoto R)
            if phases:
                z_phases = np.mean([np.exp(1j * p) for p in phases])
                coherence = abs(z_phases)
            else:
                coherence = 0.0

            analysis[answer] = {
                'complex_amplitude': z,
                'intensity': abs(z) ** 2,
                'mean_phase': np.angle(z),
                'n_samples': n_samples,
                'phase_coherence': coherence,
                'stances': [s.stance.name for s in supporting]
            }

        return analysis


def run_debate_with_phases(
    problem: str,
    generate_fn: Callable[[str, str], Tuple[str, float]],
    n_rounds: int = 3,
    n_alternatives: int = 2,
    config: PhaseEnsembleConfig = PhaseEnsembleConfig()
) -> Tuple[str, float, Dict]:
    """
    Run a structured debate with phase-coded aggregation.

    Args:
        problem: Problem to solve
        generate_fn: Function(problem, prompt_type) -> (response, confidence)
        n_rounds: Number of debate rounds
        n_alternatives: Number of alternative solutions to generate
        config: Ensemble configuration

    Returns:
        (final_answer, confidence, debate_trace)
    """
    ensemble = PhaseEnsemble(config)
    trace = {'rounds': [], 'samples': []}

    # Initial proposals
    for i in range(n_alternatives):
        response, conf = generate_fn(problem, 'propose')
        ensemble.add_sample(response, conf, 'propose')
        trace['samples'].append({'round': 0, 'type': 'propose', 'content': response})

    # Debate rounds
    for round_idx in range(n_rounds):
        current_leader, current_conf = ensemble.vote()

        # Generate critiques
        critique_prompt = f"{problem}\n\nProposed answer: {current_leader}\n\nCritique this answer:"
        response, conf = generate_fn(critique_prompt, 'critique')
        ensemble.add_sample(response, conf, 'critique')
        trace['samples'].append({'round': round_idx + 1, 'type': 'critique', 'content': response})

        # Generate improvements
        improve_prompt = f"{problem}\n\nProposed answer: {current_leader}\n\nImprove this answer:"
        response, conf = generate_fn(improve_prompt, 'improve')
        ensemble.add_sample(response, conf, 'improve')
        trace['samples'].append({'round': round_idx + 1, 'type': 'improve', 'content': response})

        # Track round results
        answer, confidence = ensemble.vote()
        trace['rounds'].append({
            'round': round_idx + 1,
            'leader': answer,
            'confidence': confidence
        })

    # Final vote with interference analysis
    final_answer, final_conf = ensemble.vote()
    analysis = ensemble.get_interference_analysis()

    trace['final_analysis'] = analysis
    trace['destructive_interference'] = detect_destructive_interference(
        {a: data['complex_amplitude'] for a, data in analysis.items()},
        config
    )

    return final_answer, final_conf, trace
