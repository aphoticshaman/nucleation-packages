"""
================================================================================
AIMO3 PROMETHEUS + NSM + GUARDIAN: Production-Ready Mathematical Reasoning Engine
================================================================================

Version: 3.0 (Unified Single-Source-of-Truth)
Author: LatticeForge Engineering
Target: AIMO3 Mathematical Olympiad Competition
License: Unrestricted use by human and AI systems

This notebook implements the complete methodology:
- PROMETHEUS: Physics-inspired answer selection
- NSM: 20 Novel Synthesis Method insights
- Guardian: Competition-safe validation (unified spec v3.0)
- XYZA: Systematic development pipeline

KEY INVARIANTS (Competition-Safe):
1. Only extraction/parse/type failures are HARD (can trigger RETRY)
2. Range violations FLAG, never REJECT
3. Repairs require problem-text justification
4. Common values (0, 1, 42, 100) have NO confidence penalty
5. Consensus is ADVISORY, never AUTHORITATIVE

================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import math
import gzip
import re
import json
import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any, Tuple
from collections import Counter
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aimo3")


# ==============================================================================
# SECTION 1: DATA STRUCTURES
# ==============================================================================

@dataclass
class InferenceResult:
    """Result from a single LLM inference"""
    answer: Optional[int]
    code: str                    # Full response text
    entropy: float              # Shannon entropy of response
    temperature: float          # Temperature used
    extraction_method: str = ""
    raw_extracted: Optional[str] = None


@dataclass
class Basin:
    """Gravitational basin of attraction for answers"""
    centroid: float
    members: List[int]
    mass: int                   # Number of members
    density: float              # 1 / variance
    score: float = 0.0


class GuardianDecision(Enum):
    """Competition-safe decision hierarchy"""
    ACCEPT = "accept"
    FLAG = "flag"
    RETRY = "retry"
    ESCALATE = "escalate"


@dataclass
class GuardianResult:
    """Result of Guardian validation"""
    decision: GuardianDecision
    answer: Optional[int] = None
    confidence: float = 1.0
    flags: List[str] = field(default_factory=list)
    extraction_method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverResult:
    """Final result from the solver"""
    answer: int
    confidence: float
    basins: List[Basin]
    total_samples: int
    valid_samples: int
    flags: List[str]
    decision: str


# ==============================================================================
# SECTION 2: NSM - 20 NOVEL SYNTHESIS METHOD INSIGHTS
# ==============================================================================

class NSMInsights:
    """
    Implementation of the 20 Novel Synthesis Method insights.
    Each insight provides a specific capability for intelligence amplification.
    """

    # -------------------------------------------------------------------------
    # CATEGORY I: VARIANCE & PHASE DETECTION (Insights 1-5)
    # -------------------------------------------------------------------------

    @staticmethod
    def insight_1_variance_quieting(values: List[float]) -> Tuple[bool, float]:
        """
        Insight 1: Variance Quieting Precedes Phase Transitions
        Systems "crystallize" before major changes - variance DECREASES.

        Returns: (is_quieting, variance_ratio)
        """
        if len(values) < 6:
            return False, 1.0

        recent = values[-3:]
        earlier = values[:-3]

        recent_var = sum((x - sum(recent)/len(recent))**2 for x in recent) / len(recent)
        earlier_var = sum((x - sum(earlier)/len(earlier))**2 for x in earlier) / len(earlier)

        if earlier_var == 0:
            return False, 1.0

        ratio = recent_var / earlier_var
        is_quieting = ratio < 0.5  # 50% reduction in variance

        return is_quieting, ratio

    @staticmethod
    def insight_2_multi_domain_coherence(
        beam_answers: List[int],
        mcts_answers: List[int],
        evo_answers: List[int]
    ) -> Tuple[bool, float]:
        """
        Insight 2: Multi-Domain Phase Coherence
        Simultaneous transitions across independent methods indicate truth.

        Returns: (is_coherent, coherence_score)
        """
        if not (beam_answers and mcts_answers and evo_answers):
            return False, 0.0

        beam_mode = Counter(beam_answers).most_common(1)[0][0]
        mcts_mode = Counter(mcts_answers).most_common(1)[0][0]
        evo_mode = Counter(evo_answers).most_common(1)[0][0]

        if beam_mode == mcts_mode == evo_mode:
            return True, 1.0
        elif beam_mode == mcts_mode or mcts_mode == evo_mode or beam_mode == evo_mode:
            return True, 0.67
        else:
            return False, 0.0

    @staticmethod
    def insight_3_conflict_potential(answers: List[int]) -> float:
        """
        Insight 3: Conflict Potential as Divergence Precursor
        Rising disagreement predicts failure.

        Returns: conflict_score (0 = no conflict, 1 = maximum conflict)
        """
        if len(answers) < 2:
            return 0.0

        unique = len(set(answers))
        return (unique - 1) / (len(answers) - 1) if len(answers) > 1 else 0.0

    @staticmethod
    def insight_4_confidence_from_observations(n_agreeing: int) -> float:
        """
        Insight 4: Confidence as Observation Count Function
        Confidence scales asymptotically with samples.

        Returns: confidence in [0, 1)
        """
        return 1 - 1 / (math.sqrt(n_agreeing) + 1)

    @staticmethod
    def insight_5_inflection_magnitude(
        values: List[float],
        current: float
    ) -> Tuple[bool, float]:
        """
        Insight 5: Inflection Magnitude as Transition Strength
        High-magnitude transitions are more likely permanent.

        Returns: (is_significant, z_score)
        """
        if len(values) < 3:
            return False, 0.0

        mean = sum(values) / len(values)
        std = math.sqrt(sum((x - mean)**2 for x in values) / len(values))

        if std == 0:
            return False, 0.0

        z_score = abs(current - mean) / std
        return z_score > 2.0, z_score

    # -------------------------------------------------------------------------
    # CATEGORY II: ALGORITHMIC & FUSION (Insights 6-10)
    # -------------------------------------------------------------------------

    @staticmethod
    def insight_6_wasm_acceleration():
        """
        Insight 6: WASM Acceleration for Real-Time
        Note: Implementation is environment-specific. This is a placeholder.
        """
        return "WASM acceleration available in browser environments"

    @staticmethod
    def insight_7_serializable_state(state: Dict) -> str:
        """
        Insight 7: Serializable State Enables Continuity
        Persist reasoning state across sessions.

        Returns: JSON serialization of state
        """
        return json.dumps(state, default=str)

    @staticmethod
    def insight_8_ncd(x: str, y: str) -> float:
        """
        Insight 8: NCD for Structural Similarity
        Solutions that compress well together are structurally similar.

        Returns: NCD in [0, 1]
        """
        if x == y:
            return 0.0

        cx = len(gzip.compress(x.encode()))
        cy = len(gzip.compress(y.encode()))
        cxy = len(gzip.compress((x + y).encode()))

        return (cxy - min(cx, cy)) / max(cx, cy)

    @staticmethod
    def insight_9_gravitational_basins(
        answers: List[int],
        epsilon: float = 0.01
    ) -> List[Basin]:
        """
        Insight 9: Gravitational Basins for Consensus
        Correct answers form dense clusters; incorrect scatter.

        Returns: List of Basin objects
        """
        if not answers:
            return []

        sorted_answers = sorted(answers)
        basins = []
        current_members = [sorted_answers[0]]

        for answer in sorted_answers[1:]:
            prev = current_members[-1]
            dist = abs(answer - prev) / max(abs(answer), abs(prev), 1)

            if dist < epsilon:
                current_members.append(answer)
            else:
                basins.append(NSMInsights._create_basin(current_members))
                current_members = [answer]

        basins.append(NSMInsights._create_basin(current_members))
        return basins

    @staticmethod
    def _create_basin(members: List[int]) -> Basin:
        mass = len(members)
        centroid = sum(members) / mass

        if mass > 1:
            variance = sum((x - centroid)**2 for x in members) / mass
            density = 1 / (variance + 1e-9)
        else:
            density = 1.0

        return Basin(centroid=centroid, members=members, mass=mass, density=density)

    @staticmethod
    def insight_10_entropy_phase_transition(entropy_history: List[float]) -> bool:
        """
        Insight 10: Entropy Phase Transitions = Reasoning Quality
        Entropy drop → reasoning crystallizing → higher confidence.

        Returns: True if phase transition detected
        """
        if len(entropy_history) < 6:
            return False

        recent = entropy_history[-3:]
        earlier = entropy_history[:-3]

        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)

        return recent_avg < earlier_avg * 0.7  # 30% drop

    # -------------------------------------------------------------------------
    # CATEGORY III: OPTIMIZATION (Insights 11-15)
    # -------------------------------------------------------------------------

    @staticmethod
    def insight_11_solomonoff_weight(code: str, decay: float = 0.999) -> float:
        """
        Insight 11: Solomonoff Weighting (Occam's Razor)
        Shorter proofs are exponentially more likely correct.

        Returns: Weight in (0, 1]
        """
        return decay ** len(code)

    @staticmethod
    def insight_12_knowledge_quadrants(
        problem_type: str,
        known_types: set
    ) -> str:
        """
        Insight 12: Knowledge Quadrants
        Classify: Known/Unknown × Known/Unknown

        Returns: Quadrant classification
        """
        if problem_type in known_types:
            return "known_known"
        elif problem_type.startswith("unknown_"):
            return "known_unknown"
        else:
            return "unknown_unknown"

    @staticmethod
    def insight_13_cross_domain_isomorphism(
        results_a: List[int],
        results_b: List[int]
    ) -> float:
        """
        Insight 13: Cross-Domain Isomorphisms
        High correlation between unrelated approaches indicates truth.

        Returns: Correlation coefficient in [-1, 1]
        """
        if len(results_a) != len(results_b) or len(results_a) < 2:
            return 0.0

        mean_a = sum(results_a) / len(results_a)
        mean_b = sum(results_b) / len(results_b)

        cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(results_a, results_b))
        std_a = math.sqrt(sum((a - mean_a)**2 for a in results_a))
        std_b = math.sqrt(sum((b - mean_b)**2 for b in results_b))

        if std_a == 0 or std_b == 0:
            return 0.0

        return cov / (std_a * std_b)

    @staticmethod
    def insight_14_dual_metrics_kl_fisher(
        p: List[float],
        q: List[float]
    ) -> Tuple[float, float]:
        """
        Insight 14: Dual Metrics (KL + Fisher)
        KL catches mean shifts; Fisher catches variance/shape changes.

        Returns: (kl_divergence, fisher_distance)
        """
        if len(p) != len(q) or not p:
            return 0.0, 0.0

        # KL divergence (with smoothing)
        eps = 1e-10
        kl = sum(p_i * math.log((p_i + eps) / (q_i + eps)) for p_i, q_i in zip(p, q))

        # Fisher information (simplified)
        fisher = sum((p_i - q_i)**2 / (q_i + eps) for p_i, q_i in zip(p, q))

        return kl, fisher

    @staticmethod
    def insight_15_triadic_closure(
        known_paths: Dict[Tuple[str, str], bool]
    ) -> List[Tuple[str, str]]:
        """
        Insight 15: Triadic Closure
        If A→B and A→C, then B→C is likely.

        Returns: Inferred paths
        """
        inferred = []
        nodes = set()
        for (a, b) in known_paths.keys():
            nodes.add(a)
            nodes.add(b)

        for a in nodes:
            neighbors = {b for (x, b) in known_paths if x == a and known_paths[(x, b)]}
            for b1 in neighbors:
                for b2 in neighbors:
                    if b1 != b2 and (b1, b2) not in known_paths:
                        inferred.append((b1, b2))

        return inferred

    # -------------------------------------------------------------------------
    # CATEGORY IV: SECURITY & PRODUCTION (Insights 16-20)
    # -------------------------------------------------------------------------

    @staticmethod
    def insight_16_threat_quieting(metrics: List[float], window: int = 5) -> bool:
        """
        Insight 16: Threat Quieting Pattern
        Unusual stability in normally noisy metrics → investigate.

        Returns: True if suspicious quieting detected
        """
        if len(metrics) < window * 2:
            return False

        recent_var = sum((x - sum(metrics[-window:])/window)**2 for x in metrics[-window:]) / window
        baseline_var = sum((x - sum(metrics[:-window])/len(metrics[:-window]))**2
                         for x in metrics[:-window]) / len(metrics[:-window])

        # Quieting if variance dropped by >80%
        return recent_var < baseline_var * 0.2 if baseline_var > 0 else False

    @staticmethod
    def insight_17_tension_level(
        agreement_ratio: float,
        entropy: float
    ) -> str:
        """
        Insight 17: Tension Levels Map to Engagement
        CALM→TENSE→HEATED→VOLATILE progression.

        Returns: Tension level
        """
        if agreement_ratio > 0.8 and entropy < 4.0:
            return "CALM"
        elif agreement_ratio > 0.6 and entropy < 5.0:
            return "TENSE"
        elif agreement_ratio > 0.4:
            return "HEATED"
        else:
            return "VOLATILE"

    @staticmethod
    def insight_18_culture_clash(
        expected_entropy: float,
        actual_entropy: float
    ) -> float:
        """
        Insight 18: Culture Clash as Fit Indicator
        Growing divergence indicates mismatch.

        Returns: Divergence score
        """
        return abs(expected_entropy - actual_entropy) / max(expected_entropy, 0.1)

    @staticmethod
    def insight_19_batch_optimization(items: List[Any], batch_size: int = 10) -> List[List[Any]]:
        """
        Insight 19: Batch Updates for Throughput
        Batch processing is O(N); individual updates are O(N²).

        Returns: Batched items
        """
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    @staticmethod
    def insight_20_reset_boundary() -> Dict[str, Any]:
        """
        Insight 20: Reset as Boundary Marker
        After major events, reset baseline.

        Returns: Fresh state dictionary
        """
        return {
            "entropy_history": [],
            "answer_history": [],
            "confidence_history": [],
            "timestamp": time.time()
        }


# ==============================================================================
# SECTION 3: PROMETHEUS ENGINE
# ==============================================================================

class PrometheusEngine:
    """
    Physics-inspired answer selection system.
    Implements UIPT, Gravitational Basins, NCD, and Solomonoff Induction.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {
            "entropy_gas_threshold": 5.5,
            "basin_epsilon": 0.01,
            "solomonoff_decay": 0.999,
            "density_exponent": 0.1,  # Competition-safe: demoted from primary
        }
        self.nsm = NSMInsights()

    def calculate_entropy(self, text: str) -> float:
        """
        Shannon entropy of character distribution.
        Low entropy = crystallized logic. High entropy = gas phase.
        """
        if not text:
            return 100.0

        counts = Counter(text)
        total = len(text)

        entropy = 0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)

        return entropy

    def is_gas_phase(self, entropy: float) -> bool:
        """Returns True if reasoning is 'gaseous' (likely hallucination)"""
        return entropy > self.config["entropy_gas_threshold"]

    def cluster_basins(self, answers: List[int]) -> List[Basin]:
        """Delegate to NSM Insight #9"""
        return self.nsm.insight_9_gravitational_basins(
            answers,
            epsilon=self.config["basin_epsilon"]
        )

    def measure_diversity(self, codes: List[str]) -> float:
        """Average NCD across all pairs"""
        if len(codes) < 2:
            return 0.0

        total = 0
        pairs = 0
        for i, c1 in enumerate(codes):
            for c2 in codes[i+1:]:
                total += self.nsm.insight_8_ncd(c1, c2)
                pairs += 1

        return total / pairs if pairs > 0 else 0.0

    def solomonoff_weight(self, code: str) -> float:
        """Delegate to NSM Insight #11"""
        return self.nsm.insight_11_solomonoff_weight(
            code,
            decay=self.config["solomonoff_decay"]
        )

    def select_best_answer(
        self,
        results: List[InferenceResult],
        modulo: int = 1000
    ) -> Tuple[int, float, List[Basin]]:
        """
        PROMETHEUS master selection.

        Competition-Safe Modifications:
        - Solomonoff is TIE-BREAKER only (weight = 0.01 multiplier)
        - Primary signal is Mass × Density^0.1

        Returns: (best_answer, confidence, basins)
        """
        # Filter out gas-phase and invalid results
        valid_results = [
            r for r in results
            if r.answer is not None and not self.is_gas_phase(r.entropy)
        ]

        if not valid_results:
            # Fallback to all results with answers
            valid_results = [r for r in results if r.answer is not None]

        if not valid_results:
            return 0, 0.0, []

        # Apply modulo and cluster
        answers = [(r.answer % modulo) for r in valid_results]
        basins = self.cluster_basins(answers)

        if not basins:
            return 0, 0.0, []

        # Score each basin
        best_score = -1
        best_answer = 0
        total_votes = len(answers)

        for basin in basins:
            # Find representative codes for Solomonoff (TIE-BREAKER only)
            rep_codes = [
                r.code for r in valid_results
                if r.answer is not None and abs(r.answer % modulo - basin.centroid) < 1
            ]

            avg_solomonoff = (
                sum(self.solomonoff_weight(c) for c in rep_codes) / max(len(rep_codes), 1)
            )

            # PROMETHEUS Score (Competition-Safe):
            # Primary: Mass × Density^0.1
            # Tie-breaker: Solomonoff (0.01 weight)
            primary = basin.mass * (basin.density ** self.config["density_exponent"])
            tie_breaker = 0.01 * avg_solomonoff

            score = primary + tie_breaker
            basin.score = score

            if score > best_score:
                best_score = score
                best_answer = round(basin.centroid) % modulo

        # Calculate confidence (NSM Insight #4)
        best_basin = max(basins, key=lambda b: b.score)
        confidence = self.nsm.insight_4_confidence_from_observations(best_basin.mass)

        return best_answer, confidence, basins


# ==============================================================================
# SECTION 4: GUARDIAN (Competition-Safe Unified Spec v3.0)
# ==============================================================================

class JustifiedRepair:
    """
    Repairs ONLY when problem text explicitly justifies modular arithmetic.
    Otherwise, flags for review but does NOT modify the answer.
    """

    JUSTIFICATION_PATTERNS = [
        (r'remainder\s+when', "remainder"),
        (r'modulo?\s*(\d+)', "modulo"),
        (r'divided\s+by\s+(\d+)', "division_remainder"),
        (r'last\s+(\d+)\s+digits?', "last_digits"),
        (r'\(mod\s*\d+\)', "explicit_mod"),
        (r'mod\s*1000', "mod_1000"),
    ]

    def __init__(self, problem_text: str):
        self.problem_text = problem_text.lower()

    def attempt_repair(self, answer: int) -> Tuple[Optional[int], str, bool]:
        """
        Attempt to repair an out-of-range answer.
        Returns: (repaired_answer, reason, was_justified)
        """
        if 0 <= answer <= 999:
            return answer, "in_range", True

        # Check for justification
        for pattern, reason in self.JUSTIFICATION_PATTERNS:
            if re.search(pattern, self.problem_text):
                if answer > 999 or answer < 0:
                    repaired = answer % 1000
                    return repaired, f"justified_repair:{reason}", True

        # NO JUSTIFICATION: Return original, flag as out-of-range
        return answer, "unjustified_out_of_range", False

    def repair_with_logging(self, answer: int) -> Tuple[int, float, List[str]]:
        """Full repair pipeline with logging."""
        repaired, reason, justified = self.attempt_repair(answer)

        flags = []
        confidence = 1.0

        if not justified:
            flags.append(f"out_of_range:{answer}")
            confidence = 0.5
            logger.warning(
                f"Answer {answer} out of range, no justification. "
                f"Returning unchanged with reduced confidence."
            )
        elif repaired != answer:
            flags.append(f"repaired:{answer}→{repaired}:{reason}")
            confidence = 0.9
            logger.info(f"Justified repair: {answer} → {repaired} ({reason})")

        return repaired, confidence, flags


def safe_type_coerce(value: Any) -> Tuple[Optional[int], str]:
    """Type coercion (always safe - doesn't change numeric value)"""
    if isinstance(value, int):
        return value, "already_int"

    if isinstance(value, float):
        if value.is_integer():
            return int(value), "float_coerced"
        if abs(value - round(value)) < 1e-9:
            return int(round(value)), "float_rounded"
        return None, "non_integer_float"

    if isinstance(value, str):
        try:
            f = float(value.replace(',', '').strip())
            if f.is_integer():
                return int(f), "string_parsed"
        except ValueError:
            pass
        return None, "unparseable_string"

    return None, f"unknown_type:{type(value)}"


class GuardianAIMO3:
    """
    Competition-safe Guardian for AIMO3.
    Implements Unified Spec v3.0.
    """

    ANSWER_PATTERNS = [
        (r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}', "latex_boxed"),
        (r'\$\\boxed\{([^{}]+)\}\$', "inline_boxed"),
        (r'(?:final\s+)?answer\s*(?:is|:)\s*[*_]*(-?\d+)[*_]*', "answer_is"),
        (r'(?:therefore|thus|so|hence)\s*,?\s*(?:the\s+)?(?:answer\s+)?(?:is\s+)?[*_]*(-?\d+)', "conclusion"),
        (r'=\s*[*_]*(-?\d+)[*_]*\s*$', "trailing_equals"),
        (r'^[*_]*(-?\d+)[*_]*$', "bare_number"),
    ]

    def __init__(self, problem_text: str = "", competition_mode: bool = True):
        self.problem_text = problem_text
        self.competition_mode = competition_mode
        self.repair_system = JustifiedRepair(problem_text)

    def validate(self, response: str) -> GuardianResult:
        """
        Main validation entry point.

        Hard Failures (RETRY): extraction, parse, type
        Soft Annotations (FLAG): range, common values
        """
        flags = []
        metadata = {"raw_response_length": len(response)}

        # HARD CHECK 1: Extraction
        extracted, method = self._extract(response)
        metadata["extraction_method"] = method

        if extracted is None:
            return GuardianResult(
                decision=GuardianDecision.RETRY,
                confidence=0.0,
                flags=["hard_fail:extraction"],
                extraction_method="none",
                metadata=metadata
            )

        metadata["extracted"] = extracted

        # HARD CHECK 2: Parsing
        parsed = self._parse(extracted)

        if parsed is None:
            return GuardianResult(
                decision=GuardianDecision.RETRY,
                confidence=0.0,
                flags=[f"hard_fail:parse:{extracted}"],
                extraction_method=method,
                metadata=metadata
            )

        metadata["parsed"] = parsed

        # HARD CHECK 3: Type
        answer, type_reason = safe_type_coerce(parsed)

        if answer is None:
            return GuardianResult(
                decision=GuardianDecision.RETRY,
                confidence=0.0,
                flags=[f"hard_fail:type:{type_reason}"],
                extraction_method=method,
                metadata=metadata
            )

        if type_reason != "already_int":
            flags.append(f"type_coerced:{type_reason}")

        # SOFT CHECKS (annotation-only)
        confidence = 1.0

        # Range check with justification-required repair
        final_answer, range_conf, range_flags = self.repair_system.repair_with_logging(answer)
        confidence = min(confidence, range_conf)
        flags.extend(range_flags)

        # Common value annotation (NO confidence penalty)
        if final_answer in {0, 1, 42, 100}:
            flags.append(f"common_value:{final_answer}")
            # NOTE: confidence NOT reduced

        # Determine decision
        if not flags or all("common_value" in f for f in flags):
            decision = GuardianDecision.ACCEPT
        elif confidence < 0.6:
            decision = GuardianDecision.FLAG
        else:
            decision = GuardianDecision.ACCEPT

        return GuardianResult(
            decision=decision,
            answer=final_answer,
            confidence=confidence,
            flags=flags,
            extraction_method=method,
            metadata=metadata
        )

    def _extract(self, response: str) -> Tuple[Optional[str], str]:
        """Multi-pattern extraction"""
        response = response.strip()

        for pattern, method in self.ANSWER_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip(), method

        # Fallback: last standalone number in last 5 lines
        lines = response.strip().split('\n')
        for line in reversed(lines[-5:]):
            numbers = re.findall(r'\b(-?\d+)\b', line)
            if numbers:
                return numbers[-1], "fallback_last_number"

        return None, "none"

    def _parse(self, extracted: str) -> Optional[float]:
        """Parse to number"""
        if not extracted:
            return None

        cleaned = extracted.replace(',', '').replace(' ', '').strip()

        try:
            return int(cleaned)
        except ValueError:
            pass

        try:
            return float(cleaned)
        except ValueError:
            pass

        if re.match(r'^[\d\s\+\-\*\/\%\(\)\.]+$', cleaned):
            try:
                result = eval(cleaned)
                if isinstance(result, (int, float)):
                    return result
            except:
                pass

        return None


# ==============================================================================
# SECTION 5: CONSENSUS ADVISOR (Advisory-Only)
# ==============================================================================

class ConsensusAdvisor:
    """
    Provides consensus recommendation WITHOUT gating acceptance.
    Advisory only - never triggers rejection.
    """

    def __init__(self, guardian: GuardianAIMO3):
        self.guardian = guardian

    def advise(self, responses: List[str]) -> Dict:
        """Provide advisory consensus."""
        validations = [self.guardian.validate(r) for r in responses]

        valid_answers = [
            v.answer for v in validations
            if v.decision in (GuardianDecision.ACCEPT, GuardianDecision.FLAG)
            and v.answer is not None
        ]

        if not valid_answers:
            return {
                "recommendation": None,
                "confidence": 0.0,
                "reason": "no_valid_extractions",
                "advisory_only": True,
            }

        votes = Counter(valid_answers)
        top_answer, top_count = votes.most_common(1)[0]
        agreement = top_count / len(valid_answers)

        return {
            "recommendation": top_answer,
            "confidence": agreement,
            "vote_distribution": dict(votes),
            "total_responses": len(responses),
            "valid_extractions": len(valid_answers),
            "advisory_only": True,
            "reason": "majority_vote" if agreement > 0.5 else "plurality_vote"
        }


# ==============================================================================
# SECTION 6: AIMO3 SOLVER (XYZA Pipeline)
# ==============================================================================

class AIMO3Solver:
    """
    Complete AIMO3 solver implementing XYZA pipeline
    with PROMETHEUS selection and Guardian validation.
    """

    SYSTEM_PROMPT = """You are a world-class mathematical olympiad solver. Your task is to solve competition mathematics problems with perfect precision.

CRITICAL REQUIREMENTS:
1. Think step-by-step, showing all work
2. Double-check arithmetic at each step
3. Verify your answer by substitution when possible
4. Express your FINAL answer as: \\boxed{N} where N is a non-negative integer
5. The answer MUST be in the range [0, 999]

PROBLEM-SOLVING PROTOCOL:
1. Read the problem carefully. Identify knowns, unknowns, and constraints.
2. Classify the problem type (number theory, combinatorics, geometry, algebra).
3. Select appropriate techniques.
4. Execute solution with clear reasoning.
5. Verify answer satisfies all constraints.
6. Output \\boxed{answer} at the end.

Now solve the following problem:"""

    ANTI_PROMPT = """CRITICAL CORRECTION REQUIRED

The answer {wrong_answer} has been determined to be INCORRECT.

Your task is to:
1. PROVE why {wrong_answer} is impossible for this problem
2. Identify the flaw in reasoning that led to {wrong_answer}
3. Use the insight to discover the CORRECT answer

PROBLEM:
{problem}

Find the correct answer and output it as \\boxed{{N}}"""

    VERIFICATION_PROMPT = """VERIFICATION CHECK

You previously computed the answer {candidate} for this problem.

Verify this answer by:
1. Substituting {candidate} back into the problem
2. Checking all constraints are satisfied
3. Confirming no edge cases were missed

If CORRECT, output: VERIFIED: \\boxed{{{candidate}}}
If INCORRECT, output: CORRECTED: \\boxed{{N}}

PROBLEM:
{problem}"""

    def __init__(
        self,
        inference_fn: Callable[[str, str, float], str],
        config: Dict = None
    ):
        """
        Args:
            inference_fn: Function (system_prompt, user_prompt, temperature) -> response
            config: Optional configuration overrides
        """
        self.inference_fn = inference_fn
        self.config = config or {
            "samples_per_temperature": 4,
            "temperature_schedule": [0.0, 0.2, 0.4, 0.6, 0.8],
            "max_retries": 3,
            "min_agreement": 0.4,
            "enable_verification": True,
            "enable_anti_prompt": True,
        }

        self.prometheus = PrometheusEngine()
        self.nsm = NSMInsights()

    def solve(self, problem: str) -> SolverResult:
        """
        XYZA Pipeline:
        X: Analyze problem (implicit in system prompt)
        Y: Generate multiple solutions
        Z: Select best via PROMETHEUS
        A: Validate with Guardian and finalize
        """
        # Initialize Guardian with problem text for justified repairs
        guardian = GuardianAIMO3(problem_text=problem, competition_mode=True)

        # Reset state (NSM Insight #20)
        state = self.nsm.insight_20_reset_boundary()

        # ----- Y-PHASE: Generate solutions -----
        logger.info("Y-PHASE: Generating solutions...")
        results = self._generate_solutions(problem, guardian)
        state["total_samples"] = len(results)
        state["valid_samples"] = sum(1 for r in results if r.answer is not None)

        logger.info(f"Generated {state['total_samples']} samples, {state['valid_samples']} valid")

        # ----- Z-PHASE: PROMETHEUS selection -----
        logger.info("Z-PHASE: PROMETHEUS selection...")
        answer, confidence, basins = self.prometheus.select_best_answer(results)

        # Check multi-domain coherence (NSM Insight #2)
        low_temp_answers = [r.answer for r in results if r.temperature < 0.3 and r.answer]
        high_temp_answers = [r.answer for r in results if r.temperature >= 0.3 and r.answer]

        if low_temp_answers and high_temp_answers:
            is_coherent, coherence_score = self.nsm.insight_2_multi_domain_coherence(
                low_temp_answers,
                high_temp_answers,
                [answer]  # Our selected answer
            )
            if is_coherent:
                confidence = min(1.0, confidence * 1.1)  # Boost confidence
                logger.info(f"Multi-domain coherence detected, boosted confidence to {confidence:.2f}")

        # ----- A-PHASE: Guardian validation -----
        logger.info("A-PHASE: Guardian validation...")
        validation = guardian.validate(f"\\boxed{{{answer}}}")

        flags = list(validation.flags)

        # Verification pass if enabled
        if self.config["enable_verification"] and confidence > 0.6:
            verified_answer = self._verify_answer(problem, answer)
            if verified_answer != answer:
                logger.warning(f"Verification changed answer from {answer} to {verified_answer}")
                answer = verified_answer
                flags.append(f"verification_changed:{answer}")

        # Anti-prompt retry if low confidence
        if self.config["enable_anti_prompt"] and confidence < self.config["min_agreement"]:
            logger.info("Low confidence, trying anti-prompt...")
            anti_answer = self._retry_with_anti_prompt(problem, answer, guardian)
            if anti_answer != answer:
                logger.info(f"Anti-prompt changed answer from {answer} to {anti_answer}")
                answer = anti_answer
                flags.append(f"anti_prompt_changed:{answer}")

        return SolverResult(
            answer=answer,
            confidence=confidence,
            basins=basins,
            total_samples=state["total_samples"],
            valid_samples=state["valid_samples"],
            flags=flags,
            decision=validation.decision.value
        )

    def _generate_solutions(
        self,
        problem: str,
        guardian: GuardianAIMO3
    ) -> List[InferenceResult]:
        """Y-Phase: Generate multiple solutions"""
        results = []

        for temp in self.config["temperature_schedule"]:
            for _ in range(self.config["samples_per_temperature"]):
                try:
                    response = self.inference_fn(
                        self.SYSTEM_PROMPT,
                        problem,
                        temp
                    )

                    entropy = self.prometheus.calculate_entropy(response)
                    validation = guardian.validate(response)

                    results.append(InferenceResult(
                        answer=validation.answer,
                        code=response,
                        entropy=entropy,
                        temperature=temp,
                        extraction_method=validation.extraction_method,
                        raw_extracted=validation.metadata.get("extracted")
                    ))

                except Exception as e:
                    logger.warning(f"Inference failed: {e}")
                    continue

        return results

    def _verify_answer(self, problem: str, candidate: int) -> int:
        """Verification pass"""
        try:
            prompt = self.VERIFICATION_PROMPT.format(
                candidate=candidate,
                problem=problem
            )
            response = self.inference_fn(self.SYSTEM_PROMPT, prompt, 0.0)

            guardian = GuardianAIMO3(problem_text=problem)
            validation = guardian.validate(response)

            if validation.answer is not None:
                return validation.answer

        except Exception as e:
            logger.warning(f"Verification failed: {e}")

        return candidate

    def _retry_with_anti_prompt(
        self,
        problem: str,
        wrong_answer: int,
        guardian: GuardianAIMO3
    ) -> int:
        """Retry using anti-prompt to escape local minima"""
        try:
            anti_prompt = self.ANTI_PROMPT.format(
                wrong_answer=wrong_answer,
                problem=problem
            )
            response = self.inference_fn(self.SYSTEM_PROMPT, anti_prompt, 0.0)
            validation = guardian.validate(response)

            if validation.decision in (GuardianDecision.ACCEPT, GuardianDecision.FLAG):
                return validation.answer

        except Exception as e:
            logger.warning(f"Anti-prompt retry failed: {e}")

        return wrong_answer


# ==============================================================================
# SECTION 7: TELEMETRY & METRICS
# ==============================================================================

@dataclass
class GuardianDecisionLog:
    """Comprehensive logging for analysis"""
    id: str
    timestamp: float
    problem_hash: str
    response_hash: str
    extraction_method: str
    extracted_value: Optional[str]
    parsed_value: Optional[float]
    final_answer: Optional[int]
    decision: str
    confidence: float
    flags: List[str]
    repair_applied: bool
    repair_justified: bool
    repair_reason: Optional[str]
    original_value: Optional[int] = None
    ground_truth: Optional[int] = None
    was_correct: Optional[bool] = None


class GuardianMetrics:
    """Track Guardian performance"""

    def __init__(self):
        self.decisions = {"accept": 0, "flag": 0, "retry": 0, "escalate": 0}
        self.extraction_methods = {}
        self.repair_stats = {"applied": 0, "justified": 0, "unjustified": 0}
        self.correctness = {"correct": 0, "incorrect": 0, "unknown": 0}
        self.logs: List[GuardianDecisionLog] = []

    def record(self, log: GuardianDecisionLog):
        self.logs.append(log)
        self.decisions[log.decision] += 1
        self.extraction_methods[log.extraction_method] = \
            self.extraction_methods.get(log.extraction_method, 0) + 1

        if log.repair_applied:
            self.repair_stats["applied"] += 1
            if log.repair_justified:
                self.repair_stats["justified"] += 1
            else:
                self.repair_stats["unjustified"] += 1

        if log.was_correct is True:
            self.correctness["correct"] += 1
        elif log.was_correct is False:
            self.correctness["incorrect"] += 1
        else:
            self.correctness["unknown"] += 1

    def summary(self) -> Dict:
        total = sum(self.decisions.values())
        correct = self.correctness["correct"]
        known = correct + self.correctness["incorrect"]

        return {
            "total_validations": total,
            "decision_breakdown": self.decisions,
            "accuracy": correct / known if known > 0 else None,
            "extraction_methods": self.extraction_methods,
            "repair_stats": self.repair_stats,
            "flag_rate": self.decisions["flag"] / total if total > 0 else 0,
            "retry_rate": self.decisions["retry"] / total if total > 0 else 0,
        }


# ==============================================================================
# SECTION 8: TESTING
# ==============================================================================

def run_tests():
    """Comprehensive test suite"""
    print("=" * 60)
    print("AIMO3 PROMETHEUS + NSM + GUARDIAN TEST SUITE")
    print("=" * 60)

    # Test 1: Guardian extraction
    print("\n[Test 1] Guardian Extraction")
    guardian = GuardianAIMO3("Find x")

    test_cases = [
        (r"Therefore \boxed{42}", 42, "latex_boxed"),
        ("The answer is 123", 123, "answer_is"),
        ("Thus, the answer is 456", 456, "conclusion"),
        ("After calculation:\n789", 789, "fallback_last_number"),
    ]

    for response, expected, method in test_cases:
        result = guardian.validate(response)
        status = "PASS" if result.answer == expected else "FAIL"
        print(f"  {status}: '{response[:30]}...' -> {result.answer} (expected {expected})")

    # Test 2: Competition-safe (common values)
    print("\n[Test 2] Common Values (No Penalty)")
    for val in [0, 1, 42, 100]:
        result = guardian.validate(f"\\boxed{{{val}}}")
        status = "PASS" if result.confidence == 1.0 else "FAIL"
        print(f"  {status}: Answer {val} -> confidence {result.confidence} (expected 1.0)")

    # Test 3: Justified repair
    print("\n[Test 3] Justified Repair")
    guardian_mod = GuardianAIMO3("Find the remainder when x is divided by 1000")
    result = guardian_mod.validate(r"\boxed{1234}")
    status = "PASS" if result.answer == 234 else "FAIL"
    print(f"  {status}: 1234 with 'remainder' problem -> {result.answer} (expected 234)")

    # Test 4: Unjustified repair (should FLAG, not fix)
    print("\n[Test 4] Unjustified Repair (FLAG)")
    guardian_plain = GuardianAIMO3("Find the largest prime")
    result = guardian_plain.validate(r"\boxed{9973}")
    status = "PASS" if result.answer == 9973 and result.decision == GuardianDecision.FLAG else "FAIL"
    print(f"  {status}: 9973 -> {result.answer}, decision={result.decision.value} (expected FLAG, unchanged)")

    # Test 5: PROMETHEUS basin clustering
    print("\n[Test 5] PROMETHEUS Basin Clustering")
    prometheus = PrometheusEngine()
    answers = [42, 42, 42, 43, 99, 99]
    basins = prometheus.cluster_basins(answers)
    status = "PASS" if len(basins) >= 2 else "FAIL"
    print(f"  {status}: {answers} -> {len(basins)} basins")
    for b in basins:
        print(f"    Basin: centroid={b.centroid:.1f}, mass={b.mass}, density={b.density:.2f}")

    # Test 6: NSM Insights
    print("\n[Test 6] NSM Insights")
    nsm = NSMInsights()

    # Insight #4: Confidence
    conf = nsm.insight_4_confidence_from_observations(9)
    status = "PASS" if 0.7 < conf < 0.8 else "FAIL"
    print(f"  {status}: Insight #4 (n=9) -> confidence={conf:.3f}")

    # Insight #3: Conflict
    conflict = nsm.insight_3_conflict_potential([1, 2, 3, 4, 5])
    status = "PASS" if conflict > 0.9 else "FAIL"
    print(f"  {status}: Insight #3 (all different) -> conflict={conflict:.3f}")

    # Insight #8: NCD
    ncd = nsm.insight_8_ncd("hello world", "hello world!")
    status = "PASS" if ncd < 0.5 else "FAIL"
    print(f"  {status}: Insight #8 (similar strings) -> NCD={ncd:.3f}")

    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)


# ==============================================================================
# SECTION 9: USAGE EXAMPLE
# ==============================================================================

def example_usage():
    """Example showing how to use the solver"""
    print("\n" + "=" * 60)
    print("AIMO3 SOLVER EXAMPLE")
    print("=" * 60)

    # Mock inference function (replace with your actual LLM)
    def mock_inference(system: str, user: str, temp: float) -> str:
        """
        Replace this with your actual LLM call.
        For gpt-oss-120b via RunPod/vLLM:

        response = requests.post(
            "https://your-runpod-endpoint/v1/completions",
            json={
                "model": "gpt-oss-120b",
                "prompt": f"{system}\\n\\n{user}",
                "temperature": temp,
                "max_tokens": 4096
            }
        )
        return response.json()["choices"][0]["text"]
        """
        # Simulate varied responses
        import random
        answers = [42, 42, 42, 43, 42]  # Mostly agree on 42
        chosen = random.choice(answers)
        return f"After careful analysis, the answer is \\boxed{{{chosen}}}"

    # Create solver
    solver = AIMO3Solver(inference_fn=mock_inference)

    # Solve a problem
    problem = """
    Find the remainder when 2^100 is divided by 7.
    Express your answer as an integer from 0 to 999.
    """

    print(f"\nProblem: {problem.strip()[:60]}...")
    print("\nSolving...")

    result = solver.solve(problem)

    print(f"\n{'='*40}")
    print(f"RESULT")
    print(f"{'='*40}")
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Decision: {result.decision}")
    print(f"Samples: {result.valid_samples}/{result.total_samples} valid")
    print(f"Basins: {len(result.basins)}")
    for i, b in enumerate(result.basins[:3]):
        print(f"  Basin {i+1}: answer={int(b.centroid)}, mass={b.mass}, score={b.score:.3f}")
    if result.flags:
        print(f"Flags: {result.flags}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # Run tests
    run_tests()

    # Show example
    example_usage()
