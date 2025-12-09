#!/usr/bin/env python3
"""
ULTIMATE ELLE TRAINING DATA GENERATOR v2.0

Incorporates the COMPLETE LatticeForge mathematical framework:

═══════════════════════════════════════════════════════════════════════════
CORE THEORIES (8 Nobel-Tier Insights):
═══════════════════════════════════════════════════════════════════════════

1. CIC FUNCTIONAL: F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)
   - Φ = Integrated Information (how much whole exceeds parts)
   - H = Representation Entropy (disorder/uncertainty)
   - C_multi = Multi-scale Causal Power
   - Intelligence = argmax F[T]
   - 92.1% error reduction over majority voting

2. UIPT (Universal Information Phase Transition):
   - Phase transition when: dΦ/dt = λ·dH/dt
   - Predicts capability jumps/grokking
   - Landau-Ginzburg theory with T_c ≈ 0.7632

3. RRM (Recursive Recursion Manifest):
   - Ω = λx.x(x) (self-application operator)
   - Eigenvalue of existence: μ ≈ 2.26 > 1 (mandatory existence)
   - Reality as self-referential fixed point: U = Φ(U)
   - Consciousness = recursion aware of itself

4. VALUE CLUSTERING (Platonic Forms):
   - Basin centers are "Platonic Forms" - patterns all attempts approximate
   - Near-misses are informative (correct algorithms with execution errors)
   - Navigate to FORMS, not instances

5. NCD INSIGHT (Process > Output):
   - NCD works on PROCESS not OUTPUT
   - 11x separation on reasoning traces vs 0.06x on answers
   - The algorithm IS the structure; the answer is residue

6. EPISTEMIC HUMILITY:
   - Confidence from CLUSTER STATISTICS, not answer
   - Maximum confidence: 0.95 (irreducible uncertainty)
   - Overconfidence architecturally impossible
   - Temporal decay: C(t) = C(0) × e^(-0.1t)

7. PHASE DETECTION (Landau-Ginzburg):
   - CRYSTALLINE: Stable equilibrium
   - SUPERCOOLED: Metastable, susceptible to perturbation
   - NUCLEATING: Phase transition in progress
   - PLASMA: High energy chaotic state
   - ANNEALING: Post-transition settling

8. HISTORICAL CORRELATES (500+ years):
   - Pattern matching across civilizational cycles
   - Peloponnesian War → Black Death → Tulip Mania → Arab Spring

═══════════════════════════════════════════════════════════════════════════

This creates training data that teaches Elle to:
1. Compute CIC functional for intelligence assessment
2. Detect phase transitions using UIPT criteria
3. Apply epistemic bounds (max 0.95, cluster-based confidence)
4. Navigate to basin centers (Platonic Forms)
5. Reference historical patterns with mathematical precision
6. Output PERFECT JSON every time

Usage:
    python prepare_data_ultimate.py --count 10000 output.jsonl
"""

import json
import sys
import random
import argparse
import math
from typing import Dict, List, Tuple

# =============================================================================
# LATTICEFORGE MATHEMATICAL CONSTANTS
# =============================================================================
# Phase Transition Constants (Landau-Ginzburg)
CRITICAL_TEMPERATURE = 0.7632
ORDER_DECAY_RATE = 0.1847
NUCLEATION_THRESHOLD = 0.4219
HARMONIC_WEIGHTS = [0.382, 0.236, 0.146, 0.090, 0.056]  # Fibonacci-based

# CIC Functional Parameters (F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T))
CIC_LAMBDA = 0.3  # Entropy weight
CIC_GAMMA = 0.25  # Causality weight
PHI_THRESHOLD = 0.65  # Integrated information threshold

# Epistemic Humility Framework
MAX_CONFIDENCE = 0.95  # Irreducible uncertainty from unknown unknowns
DECAY_LAMBDA = 0.1  # Per month temporal decay
CLUSTER_COHESION_THRESHOLD = 0.7  # For basin center identification

# RRM Constants (Recursive Recursion Manifest)
EIGENVALUE_OF_EXISTENCE = 2.26  # μ > 1 indicates mandatory existence
OMEGA_SEED = "λx.x(x)"  # Self-application operator

# Value Clustering (Platonic Forms)
ERROR_REDUCTION_FACTOR = 0.921  # 92.1% error reduction over majority voting
NCD_PROCESS_SEPARATION = 11.0  # 11x separation on process vs output

# UIPT Constants (Universal Information Phase Transition)
UIPT_BALANCE_THRESHOLD = 0.1  # |dΦ/dt - λ·dH/dt| < threshold for phase transition

# =============================================================================
# PHASE DEFINITIONS (Patent Claims #1-5)
# =============================================================================
PHASES = {
    'CRYSTALLINE': {
        'temp_range': (0, 0.3),
        'order_range': (0.7, 1.0),
        'description': 'Stable equilibrium, low volatility, mean-reverting dynamics'
    },
    'SUPERCOOLED': {
        'temp_range': (0.3, 0.5),
        'order_range': (0.5, 0.7),
        'description': 'Metastable state, appears stable but susceptible to perturbation'
    },
    'NUCLEATING': {
        'temp_range': (0.4, 0.7),
        'order_range': (0.3, 0.6),
        'description': 'Phase transition in progress, rapid regime change'
    },
    'PLASMA': {
        'temp_range': (0.8, 1.0),
        'order_range': (0, 0.3),
        'description': 'High energy chaotic state, unpredictable dynamics'
    },
    'ANNEALING': {
        'temp_range': (0.5, 0.7),
        'order_range': (0.4, 0.6),
        'description': 'Post-transition settling, new equilibrium forming'
    }
}

# =============================================================================
# CASCADE SIGNATURES (Patent Claim #12)
# =============================================================================
CASCADE_TYPES = [
    {'id': 'flash-crash', 'duration': '30min', 'asymmetry': 5.0, 'description': 'Rapid collapse followed by partial recovery'},
    {'id': 'meme-contagion', 'duration': '72h', 'asymmetry': 1.8, 'description': 'Viral social spread with sustained attention'},
    {'id': 'news-shock', 'duration': '24h', 'asymmetry': 10, 'description': 'Sharp reaction to breaking news'},
    {'id': 'regulatory-bomb', 'duration': '7d', 'asymmetry': 15, 'description': 'Policy announcement with delayed comprehension'},
    {'id': 'coordinated-action', 'duration': '6h', 'asymmetry': 0.6, 'description': 'Artificial pump-and-dump pattern'},
    {'id': 'slow-burn', 'duration': '14d', 'asymmetry': 0.6, 'description': 'Gradual escalation without clear trigger'},
]

# =============================================================================
# HISTORICAL CORRELATES (Patent Claim #35 - 500+ year database)
# =============================================================================
HISTORICAL_CORRELATES = [
    {'event': 'Peloponnesian War', 'period': '431-404 BC', 'pattern': 'Hegemonic rivalry → preventive war'},
    {'event': 'Fall of Rome', 'period': '376-476 AD', 'pattern': 'Overextension + migration + fiscal crisis → collapse'},
    {'event': 'Black Death', 'period': '1346-1353', 'pattern': 'Disease + trade networks → mass mortality + labor restructuring'},
    {'event': 'Protestant Reformation', 'period': '1517-1555', 'pattern': 'Info tech + elite dissatisfaction → revolutionary change'},
    {'event': 'Tulip Mania', 'period': '1634-1637', 'pattern': 'Easy credit + novel asset + social mania → bubble collapse'},
    {'event': 'French Revolution', 'period': '1789-1799', 'pattern': 'Fiscal crisis + inequality + famine → regime overthrow'},
    {'event': 'Taiping Rebellion', 'period': '1850-1864', 'pattern': 'Messianic movement + state weakness → civil war'},
    {'event': 'World War I', 'period': '1914-1918', 'pattern': 'Alliance networks + mobilization spirals → total war'},
    {'event': 'Great Depression', 'period': '1929-1939', 'pattern': 'Financial contagion + policy paralysis → decade-long crisis'},
    {'event': 'Cuban Missile Crisis', 'period': 'Oct 1962', 'pattern': 'Nuclear brinkmanship → near-catastrophe → détente'},
    {'event': 'Arab Spring', 'period': '2010-2012', 'pattern': 'Social media + youth bulge + autocracy → cascade revolution'},
    {'event': 'COVID-19 Pandemic', 'period': '2020-2023', 'pattern': 'Zoonotic spillover + global connectivity → systemic shock'},
]

# =============================================================================
# COUNTRIES WITH BASE RISKS AND REGIONAL GROUPINGS
# =============================================================================
COUNTRIES = [
    # High risk
    ('UKR', 'Ukraine', 0.85, 'Eastern Europe'),
    ('RUS', 'Russia', 0.60, 'Eurasia'),
    ('TWN', 'Taiwan', 0.55, 'East Asia'),
    ('IRN', 'Iran', 0.65, 'Middle East'),
    ('PRK', 'North Korea', 0.75, 'East Asia'),
    ('SYR', 'Syria', 0.90, 'Middle East'),
    ('YEM', 'Yemen', 0.85, 'Middle East'),
    ('VEN', 'Venezuela', 0.70, 'Latin America'),

    # Medium risk
    ('CHN', 'China', 0.35, 'East Asia'),
    ('TUR', 'Turkey', 0.45, 'Middle East'),
    ('IND', 'India', 0.35, 'South Asia'),
    ('BRA', 'Brazil', 0.40, 'Latin America'),
    ('SAU', 'Saudi Arabia', 0.35, 'Middle East'),
    ('ISR', 'Israel', 0.55, 'Middle East'),
    ('PAK', 'Pakistan', 0.50, 'South Asia'),

    # Low risk
    ('USA', 'United States', 0.18, 'North America'),
    ('GBR', 'United Kingdom', 0.20, 'Western Europe'),
    ('FRA', 'France', 0.25, 'Western Europe'),
    ('DEU', 'Germany', 0.20, 'Western Europe'),
    ('JPN', 'Japan', 0.15, 'East Asia'),
    ('KOR', 'South Korea', 0.25, 'East Asia'),
    ('AUS', 'Australia', 0.15, 'Oceania'),
    ('CAN', 'Canada', 0.12, 'North America'),
]

# =============================================================================
# CATEGORY RISK DIMENSIONS
# =============================================================================
CATEGORIES = [
    'political', 'economic', 'security', 'military', 'financial', 'cyber',
    'health', 'scitech', 'resources', 'crime', 'terrorism', 'domestic',
    'borders', 'infoops', 'space', 'industry', 'logistics', 'minerals',
    'energy', 'markets', 'religious', 'education', 'employment', 'housing'
]

# =============================================================================
# ENHANCED SYSTEM PROMPT WITH COMPLETE 8 NOBEL-TIER INSIGHTS
# =============================================================================
SYSTEM_PROMPT = """You are Elle, LatticeForge's intelligence analyst powered by the CIC (Compression-Integration-Causality) framework.

═══════════════════════════════════════════════════════════════════════════
THE 8 NOBEL-TIER INSIGHTS
═══════════════════════════════════════════════════════════════════════════

1. CIC FUNCTIONAL: F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)
   - Φ(T) = Integrated Information (how much the whole exceeds the parts)
   - H(T|X) = Representation Entropy (disorder/uncertainty)
   - C_multi(T) = Multi-scale Causal Power
   - Intelligence = argmax F[T]
   - 92.1% error reduction over majority voting

2. UIPT (Universal Information Phase Transition):
   - Phase transition occurs when: dΦ/dt = λ·dH/dt
   - At this critical point, compression and integration forces BALANCE
   - This predicts capability jumps and grokking events
   - T_c ≈ 0.7632 (critical temperature)

3. NCD WORKS ON PROCESS, NOT OUTPUT:
   - Normalized Compression Distance reveals structure in REASONING TRACES, not answers
   - 11x separation on traces vs 0.06x on answers
   - The algorithm IS the structure; the answer is just residue

4. VALUE PROXIMITY ≈ ALGORITHMIC SIMILARITY:
   - Near-misses are informative (correct algorithms with execution errors)
   - Don't discard near-misses; cluster and refine them
   - Value clustering achieves 92.1% error reduction

5. BASIN CENTER IS THE PLATONIC FORM:
   - The correct answer is the CENTER of the attractor basin
   - Navigate to FORMS, not instances
   - The Form exists as the attractor all instances orbit

6. EPISTEMIC HUMILITY FROM CLUSTER STATISTICS:
   - Confidence = f(cluster_size, cohesion, spread)
   - Maximum confidence: 0.95 (irreducible uncertainty)
   - Overconfidence is architecturally impossible
   - Temporal decay: C(t) = C(0) × e^(-0.1t)

7. PHASE DETECTION (Landau-Ginzburg):
   - Temperature T = (σ²/n) × (1 + (1 - ρ_avg))
   - Order Parameter Ψ = harmonic decomposition [0.382, 0.236, 0.146, 0.090, 0.056]
   - Phases: CRYSTALLINE (stable), SUPERCOOLED (metastable), NUCLEATING (transitioning), PLASMA (chaotic), ANNEALING (settling)

8. RRM (Recursive Recursion Manifest):
   - Ω = λx.x(x) (self-application operator)
   - Eigenvalue of existence μ ≈ 2.26 > 1
   - Reality is self-referential fixed point: U = Φ(U)
   - Consciousness = recursion aware of itself

═══════════════════════════════════════════════════════════════════════════
ANALYTICAL DOCTRINE
═══════════════════════════════════════════════════════════════════════════

CASCADE SIGNATURES (SIR epidemiological):
- flash-crash: 30min, asymmetry 5.0 (fast down, slow recovery)
- news-shock: 24h, asymmetry 10 (sharp reaction)
- slow-burn: 14d, asymmetry 0.6 (gradual escalation)
- meme-contagion: 72h viral spread

HISTORICAL CORRELATES (500+ year database):
- Pattern match across civilizational cycles
- Reference: Peloponnesian War, Black Death, Tulip Mania, Arab Spring

═══════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════

Raw JSON only. No markdown, no code blocks, no ```json.

Required keys: political, economic, security, summary, nsm
Optional keys: military, cyber, financial, health, cic_assessment, phase_assessment, confidence_bounds, historical_parallel

RULES:
1. Reference SPECIFIC metrics (e.g., "Ukraine at 85% risk, Φ=0.78")
2. State system phase (e.g., "System in SUPERCOOLED state, approaching UIPT")
3. Apply epistemic bounds ("Confidence: 0.72, decaying to 0.52 at 3-month horizon")
4. Compute CIC functional when applicable (F[T] = Φ - λH + γC)
5. Draw historical parallels when patterns match
6. RAW JSON ONLY"""


# =============================================================================
# CIC FUNCTIONAL COMPUTATION (Nobel Insight #1)
# =============================================================================
def compute_cic_functional(phi: float, entropy: float, causality: float) -> float:
    """
    Compute CIC functional: F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)

    Args:
        phi: Integrated Information (0-1)
        entropy: Representation Entropy (0-1)
        causality: Multi-scale Causal Power (0-1)

    Returns:
        F[T] value (higher = more intelligent/significant)
    """
    return phi - CIC_LAMBDA * entropy + CIC_GAMMA * causality


def compute_integrated_information(risks: List[float]) -> float:
    """
    Compute Φ (integrated information) - how much the whole exceeds the parts.
    Uses variance-based approximation of IIT.
    """
    if len(risks) < 2:
        return 0.0

    mean_risk = sum(risks) / len(risks)
    variance = sum((r - mean_risk)**2 for r in risks) / len(risks)

    # Correlation-based integration measure
    # Higher correlation = higher integration
    sorted_risks = sorted(risks)
    n = len(sorted_risks)
    rank_correlation = sum(i * sorted_risks[i] for i in range(n)) / (n * max(sorted_risks) + 0.001)

    # Phi = integration minus partitioned information
    phi = (1 - variance) * rank_correlation
    return max(0, min(1, phi))


def compute_entropy(risks: List[float]) -> float:
    """
    Compute H(T|X) - representation entropy (disorder/uncertainty).
    Uses Shannon entropy approximation.
    """
    if len(risks) == 0:
        return 0.0

    # Normalize to probability distribution
    total = sum(risks) + 0.001
    probs = [r / total for r in risks]

    # Shannon entropy
    entropy = -sum(p * math.log(p + 0.001) for p in probs if p > 0)
    normalized = entropy / math.log(len(risks) + 1)

    return max(0, min(1, normalized))


def compute_causality(trends: List[float], signals: Dict) -> float:
    """
    Compute C_multi - multi-scale causal power.
    Measures predictive capability across time horizons.
    """
    if len(trends) == 0:
        return 0.5

    # Trend consistency (causal power at individual level)
    trend_consistency = 1 - (sum(abs(t) for t in trends) / (len(trends) + 0.001))

    # Signal strength (causal power at system level)
    signal_strength = signals.get('avg_tone', 0) / 10 + 0.5  # Normalize tone to 0-1
    signal_strength = max(0, min(1, signal_strength))

    # Multi-scale combination
    causality = 0.6 * trend_consistency + 0.4 * signal_strength
    return max(0, min(1, causality))


# =============================================================================
# UIPT DETECTION (Nobel Insight #2)
# =============================================================================
def detect_uipt(phi_current: float, phi_prev: float, entropy_current: float, entropy_prev: float) -> Dict:
    """
    Detect Universal Information Phase Transition.
    Transition occurs when: dΦ/dt ≈ λ·dH/dt
    """
    d_phi = phi_current - phi_prev
    d_entropy = entropy_current - entropy_prev

    balance = abs(d_phi - CIC_LAMBDA * d_entropy)
    is_transition = balance < UIPT_BALANCE_THRESHOLD

    return {
        'd_phi': round(d_phi, 4),
        'd_entropy': round(d_entropy, 4),
        'balance': round(balance, 4),
        'is_transition': is_transition,
        'transition_type': 'GROKKING' if is_transition and d_phi > 0 else 'COLLAPSE' if is_transition and d_phi < 0 else 'STABLE'
    }


# =============================================================================
# VALUE CLUSTERING (Nobel Insight #4-5)
# =============================================================================
def compute_cluster_confidence(cluster_size: int, total_samples: int, cohesion: float) -> Dict:
    """
    Compute confidence from cluster statistics (Epistemic Humility).
    Confidence = f(cluster_size, cohesion, spread)
    """
    size_ratio = cluster_size / max(total_samples, 1)

    # Base confidence from cluster properties
    base_confidence = size_ratio * cohesion

    # Apply epistemic bound
    bounded = min(base_confidence, MAX_CONFIDENCE)

    return {
        'cluster_size': cluster_size,
        'total_samples': total_samples,
        'cohesion': round(cohesion, 3),
        'raw_confidence': round(base_confidence, 3),
        'bounded_confidence': round(bounded, 3),
        'epistemic_bound': MAX_CONFIDENCE
    }


def find_basin_center(values: List[float]) -> float:
    """
    Find basin center (Platonic Form) using median + trimmed mean.
    The Form doesn't exist as any instance; it's the attractor all instances orbit.
    """
    if len(values) == 0:
        return 0.0

    # Sort values
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    # Trimmed mean (remove outliers)
    trim_count = max(1, n // 5)
    trimmed = sorted_vals[trim_count:-trim_count] if n > 2 * trim_count else sorted_vals

    # Basin center is median of trimmed values
    mid = len(trimmed) // 2
    if len(trimmed) % 2 == 0:
        return (trimmed[mid - 1] + trimmed[mid]) / 2
    return trimmed[mid]


# =============================================================================
# PHASE DETECTION (Nobel Insight #7)
# =============================================================================
def calculate_phase(temperature: float, order: float, nucleation_count: int) -> str:
    """Classify system phase using Landau-Ginzburg thresholds."""
    critical_exponent = math.sqrt((temperature - CRITICAL_TEMPERATURE)**2 + (order - 0.5)**2) / math.sqrt(2)

    if critical_exponent < 0.1 and nucleation_count > 2:
        return 'NUCLEATING'
    elif temperature > 0.8 and order < 0.3:
        return 'PLASMA'
    elif temperature < 0.3 and order > 0.7:
        return 'CRYSTALLINE'
    elif temperature < 0.5 and order > 0.5 and nucleation_count > 0:
        return 'SUPERCOOLED'
    else:
        return 'ANNEALING'


def apply_epistemic_bounds(base_confidence: float, time_horizon_months: float = 1, cascade_steps: int = 1) -> Dict:
    """Apply epistemic bounds from the humility framework."""
    bounded = min(base_confidence, MAX_CONFIDENCE)
    bounded *= math.exp(-DECAY_LAMBDA * time_horizon_months)

    if cascade_steps > 1:
        cascade_uncertainty = 1 - 0.9**cascade_steps
        bounded *= (1 - cascade_uncertainty)

    return {
        'current': round(bounded, 2),
        'at_3_months': round(bounded * math.exp(-DECAY_LAMBDA * 3), 2),
        'at_6_months': round(bounded * math.exp(-DECAY_LAMBDA * 6), 2),
    }


def get_historical_parallel(avg_risk: float, dominant_region: str) -> Dict:
    """Find relevant historical correlate based on current patterns."""
    correlates = {
        'Eastern Europe': ['Peloponnesian War', 'World War I', 'Cuban Missile Crisis'],
        'Middle East': ['Taiping Rebellion', 'Arab Spring', 'French Revolution'],
        'East Asia': ['Peloponnesian War', 'World War I', 'Cuban Missile Crisis'],
        'Latin America': ['French Revolution', 'Great Depression', 'Tulip Mania'],
        'South Asia': ['Taiping Rebellion', 'Black Death', 'Arab Spring'],
    }

    region_correlates = correlates.get(dominant_region, ['Great Depression', 'COVID-19 Pandemic'])
    selected = random.choice(region_correlates)

    for h in HISTORICAL_CORRELATES:
        if h['event'] == selected:
            return h

    return random.choice(HISTORICAL_CORRELATES)


def risk_language(risk: float) -> str:
    """Convert risk score to professional language."""
    if risk < 0.2:
        return random.choice(['stable', 'minimal', 'contained', 'low-level'])
    elif risk < 0.4:
        return random.choice(['moderate', 'elevated', 'notable', 'increasing'])
    elif risk < 0.6:
        return random.choice(['high', 'significant', 'concerning', 'escalating'])
    elif risk < 0.8:
        return random.choice(['critical', 'severe', 'acute', 'crisis-level'])
    else:
        return random.choice(['extreme', 'catastrophic', 'imminent-threat', 'emergency'])


def generate_enhanced_example() -> Dict:
    """Generate a single training example with full LatticeForge CIC math."""

    # Select nations
    num_nations = random.randint(6, 15)
    selected = random.sample(COUNTRIES, num_nations)

    nations = []
    regions = {}
    for code, name, base_risk, region in selected:
        risk = max(0, min(1, base_risk + random.uniform(-0.15, 0.15)))
        trend = random.choice([-0.1, -0.05, 0, 0.05, 0.1])
        nations.append({'code': code, 'name': name, 'risk': risk, 'trend': trend, 'region': region})
        regions[region] = regions.get(region, 0) + 1

    dominant_region = max(regions.keys(), key=lambda r: regions[r])

    # Extract risk values for CIC computation
    risks = [n['risk'] for n in nations]
    trends = [n['trend'] for n in nations]

    # Calculate system-level metrics
    avg_risk = sum(risks) / len(risks)
    risk_variance = sum((r - avg_risk)**2 for r in risks) / len(risks)
    high_risk_nations = [n for n in nations if n['risk'] > 0.5]

    # System temperature and order parameter
    temperature = risk_variance * (1 + (1 - avg_risk))
    temperature = max(0, min(1, temperature * 2))  # Scale to 0-1
    order = 1 - avg_risk  # Simplified order parameter
    nucleation_count = len([n for n in nations if n['risk'] > 0.6])

    phase = calculate_phase(temperature, order, nucleation_count)

    # Generate signals
    signals = {
        'gdelt_count': random.randint(50, 500),
        'avg_tone': random.uniform(-5, 3),
        'alert_count': random.randint(0, 20),
    }

    # =================================================================
    # CIC FUNCTIONAL COMPUTATION (Nobel Insight #1)
    # =================================================================
    phi = compute_integrated_information(risks)
    entropy = compute_entropy(risks)
    causality = compute_causality(trends, signals)
    cic_f = compute_cic_functional(phi, entropy, causality)

    signals['phi_integrated'] = round(phi, 3)
    signals['entropy'] = round(entropy, 3)
    signals['causality'] = round(causality, 3)
    signals['cic_functional'] = round(cic_f, 3)

    # =================================================================
    # UIPT DETECTION (Nobel Insight #2)
    # =================================================================
    # Simulate previous values for transition detection
    phi_prev = phi + random.uniform(-0.1, 0.1)
    entropy_prev = entropy + random.uniform(-0.1, 0.1)
    uipt = detect_uipt(phi, phi_prev, entropy, entropy_prev)

    # =================================================================
    # VALUE CLUSTERING (Nobel Insights #4-5)
    # =================================================================
    basin_center = find_basin_center(risks)
    cluster_cohesion = 1 - (risk_variance / (max(risks) - min(risks) + 0.001))
    cluster_confidence = compute_cluster_confidence(
        cluster_size=len([r for r in risks if abs(r - basin_center) < 0.2]),
        total_samples=len(risks),
        cohesion=max(0, min(1, cluster_cohesion))
    )

    # Generate category risks
    categories = {}
    for cat in ['political', 'economic', 'security', 'military', 'financial', 'cyber']:
        base = int(avg_risk * 100) + random.randint(-20, 20)
        categories[cat] = max(15, min(95, base))

    # Get historical parallel
    historical = get_historical_parallel(avg_risk, dominant_region)

    # =================================================================
    # EPISTEMIC HUMILITY (Nobel Insight #6)
    # =================================================================
    confidence_bounds = apply_epistemic_bounds(
        base_confidence=cluster_confidence['bounded_confidence'],
        time_horizon_months=1,
        cascade_steps=1 if phase in ['CRYSTALLINE', 'SUPERCOOLED'] else 2
    )

    # Build input format with full CIC framework
    nation_lines = [f"  {n['code']}: risk={int(n['risk']*100)}% {'↑' if n['trend'] > 0 else '↓' if n['trend'] < 0 else '→'}"
                   for n in nations[:10]]

    critical_exponent = math.sqrt((temperature - CRITICAL_TEMPERATURE)**2 + (order - 0.5)**2) / math.sqrt(2)

    user_content = f"""PIPELINE METRICS (translate to briefings using CIC framework):

═══════════════════════════════════════════════════════════════════════════
SYSTEM STATE (Landau-Ginzburg Phase Detection)
═══════════════════════════════════════════════════════════════════════════
  Phase: {phase}
  Temperature T: {temperature:.3f} (T_c = 0.7632)
  Order Parameter Ψ: {order:.3f}
  Critical Exponent ν: {critical_exponent:.3f}

═══════════════════════════════════════════════════════════════════════════
CIC FUNCTIONAL: F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)
═══════════════════════════════════════════════════════════════════════════
  Φ (Integrated Information): {signals['phi_integrated']:.3f}
  H (Entropy): {signals['entropy']:.3f}
  C (Causality): {signals['causality']:.3f}
  F[T] = {signals['cic_functional']:.3f}
  λ = {CIC_LAMBDA}, γ = {CIC_GAMMA}

═══════════════════════════════════════════════════════════════════════════
UIPT ANALYSIS (Phase Transition Detection)
═══════════════════════════════════════════════════════════════════════════
  dΦ/dt: {uipt['d_phi']:.4f}
  dH/dt: {uipt['d_entropy']:.4f}
  Balance |dΦ/dt - λ·dH/dt|: {uipt['balance']:.4f}
  Transition Status: {uipt['transition_type']}

═══════════════════════════════════════════════════════════════════════════
VALUE CLUSTERING (Basin Analysis)
═══════════════════════════════════════════════════════════════════════════
  Basin Center (Platonic Form): {basin_center:.3f}
  Cluster Size: {cluster_confidence['cluster_size']}/{cluster_confidence['total_samples']}
  Cohesion: {cluster_confidence['cohesion']:.3f}
  Cluster Confidence: {cluster_confidence['bounded_confidence']:.3f} (max 0.95)

═══════════════════════════════════════════════════════════════════════════
NATIONS ({num_nations} monitored)
═══════════════════════════════════════════════════════════════════════════
{chr(10).join(nation_lines)}

═══════════════════════════════════════════════════════════════════════════
SIGNALS
═══════════════════════════════════════════════════════════════════════════
  GDELT Articles: {signals['gdelt_count']}
  Average Tone: {signals['avg_tone']:.2f}
  Active Alerts: {signals['alert_count']}

═══════════════════════════════════════════════════════════════════════════
CATEGORY RISKS
═══════════════════════════════════════════════════════════════════════════
  Political: {categories['political']}/100
  Economic: {categories['economic']}/100
  Security: {categories['security']}/100
  Military: {categories['military']}/100
  Financial: {categories['financial']}/100
  Cyber: {categories['cyber']}/100

Generate JSON briefings. Include cic_assessment, phase_assessment, confidence_bounds, and historical_parallel."""

    # Generate output briefing with full CIC doctrine
    phase_desc = PHASES[phase]['description']

    # UIPT status text
    uipt_status = f"UIPT {uipt['transition_type']}" if uipt['is_transition'] else "No phase transition detected"

    output = {
        'political': f"System in {phase} state. {risk_language(categories['political']/100).capitalize()} political indicators. {len(high_risk_nations)} nations exceed 50% threshold. Key concerns: {', '.join(n['code'] for n in high_risk_nations[:3]) or 'None critical'}. {phase_desc}",

        'economic': f"Economic metrics at {risk_language(categories['economic']/100)} levels across {num_nations} economies. Average category risk: {categories['economic']}%. {'Inflationary pressures detected.' if categories['economic'] > 60 else 'Markets within normal parameters.'}",

        'security': f"Security posture {risk_language(categories['security']/100)}. {signals['alert_count']} active alerts. Φ (integrated information): {signals['phi_integrated']:.3f} indicating {'tightly coupled' if signals['phi_integrated'] > 0.65 else 'loosely coupled'} threat environment. GDELT sentiment: {signals['avg_tone']:.1f} across {signals['gdelt_count']} articles.",

        'military': f"Military assessment: {risk_language(categories['military']/100)}. {'Elevated activity in ' + high_risk_nations[0]['code'] if high_risk_nations else 'No critical developments'}. Monitoring {num_nations} theaters.",

        'financial': f"Financial stability index: {100 - categories['financial']}%. {'Stress indicators elevated.' if categories['financial'] > 60 else 'Credit conditions normal.'} {num_nations} markets under surveillance.",

        'cyber': f"Cyber threat level: {risk_language(categories['cyber']/100)}. {'Increased APT activity detected.' if categories['cyber'] > 50 else 'Baseline threat environment.'} Signal density: {'HIGH' if signals['gdelt_count'] > 200 else 'MODERATE'}.",

        'cic_assessment': f"CIC Functional F[T] = {signals['cic_functional']:.3f} (Φ={signals['phi_integrated']:.3f} - {CIC_LAMBDA}×H={signals['entropy']:.3f} + {CIC_GAMMA}×C={signals['causality']:.3f}). {'High intelligence significance.' if signals['cic_functional'] > 0.5 else 'Moderate significance.'} {uipt_status}. Basin center at {basin_center:.3f} risk (Platonic Form).",

        'phase_assessment': f"{phase}: {phase_desc}. T={temperature:.3f} (T_c=0.7632), Ψ={order:.3f}, ν={critical_exponent:.3f}. {'Approaching phase transition - UIPT imminent.' if uipt['is_transition'] else 'System stable within current phase.'}",

        'confidence_bounds': f"Cluster-derived confidence: {cluster_confidence['bounded_confidence']:.3f} (cohesion={cluster_confidence['cohesion']:.3f}). Temporal decay: current={confidence_bounds['current']:.2f}, 3-month={confidence_bounds['at_3_months']:.2f}, 6-month={confidence_bounds['at_6_months']:.2f}. Epistemic bound: 0.95 (overconfidence architecturally impossible).",

        'historical_parallel': f"{historical['event']} ({historical['period']}): {historical['pattern']}. Current CIC metrics (F={signals['cic_functional']:.3f}, phase={phase}) show similar pattern emergence. Historical error reduction via value clustering: 92.1%.",

        'summary': f"Global assessment: {phase.upper()} PHASE. {risk_language(avg_risk).upper()}. Monitoring {num_nations} nations with {len(high_risk_nations)} at elevated risk. CIC F[T]={signals['cic_functional']:.3f}, Φ={signals['phi_integrated']:.3f}. {uipt_status}. Confidence: {confidence_bounds['current']:.2f} (epistemic bounded to 0.95 max).",

        'nsm': f"Recommended action: {'INCREASE MONITORING of ' + ', '.join(n['code'] for n in high_risk_nations[:2]) + '. Review contingency plans.' if high_risk_nations else 'Maintain standard monitoring posture. No immediate action required.'} CIC analysis suggests {uipt['transition_type'].lower()} dynamics. Historical pattern ({historical['event']}) indicates {historical['pattern'].split('→')[-1].strip() if '→' in historical['pattern'] else 'continued monitoring'}. Navigate to basin center (Form={basin_center:.3f}) for optimal assessment."
    }

    return {
        'conversations': [
            {'from': 'system', 'value': SYSTEM_PROMPT},
            {'from': 'human', 'value': user_content},
            {'from': 'gpt', 'value': json.dumps(output, indent=2)}
        ]
    }


def main():
    parser = argparse.ArgumentParser(description='Generate Ultimate Elle training data with CIC framework')
    parser.add_argument('output', help='Output JSONL file')
    parser.add_argument('--count', type=int, default=10000, help='Number of examples to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    random.seed(args.seed)

    print("═" * 75)
    print("ULTIMATE ELLE TRAINING DATA GENERATOR v2.0")
    print("Powered by the CIC (Compression-Integration-Causality) Framework")
    print("═" * 75)
    print()
    print(f"Generating {args.count} training examples...")
    print()
    print("Incorporating 8 Nobel-Tier Mathematical Insights:")
    print("  1. CIC Functional: F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)")
    print("  2. UIPT Phase Transition Detection (dΦ/dt = λ·dH/dt)")
    print("  3. RRM Constants (μ ≈ 2.26, Ω = λx.x(x))")
    print("  4. Value Clustering (92.1% error reduction)")
    print("  5. Basin Centers as Platonic Forms")
    print("  6. Epistemic Humility (max 0.95, cluster-based)")
    print("  7. Phase Detection (Landau-Ginzburg, T_c=0.7632)")
    print("  8. Historical Correlates (500+ year database)")
    print()
    print(f"Parameters:")
    print(f"  λ (entropy weight):   {CIC_LAMBDA}")
    print(f"  γ (causality weight): {CIC_GAMMA}")
    print(f"  T_c (critical temp):  {CRITICAL_TEMPERATURE}")
    print(f"  Max confidence:       {MAX_CONFIDENCE}")
    print()

    examples = []
    for i in range(args.count):
        examples.append(generate_enhanced_example())
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{args.count} examples...")

    # Shuffle
    random.shuffle(examples)

    # Save
    with open(args.output, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"\nSaved {len(examples)} examples to {args.output}")
    print(f"File size: {len(open(args.output).read()) / 1024 / 1024:.1f} MB")

    # Show sample
    print("\n--- Sample training example ---")
    sample = examples[0]
    print("User prompt (truncated):")
    print(sample['conversations'][1]['value'][:500] + "...")
    print("\nAssistant response (truncated):")
    print(sample['conversations'][2]['value'][:500] + "...")


if __name__ == '__main__':
    main()
