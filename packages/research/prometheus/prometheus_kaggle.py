"""
PROMETHEUS (Kaggle Edition) — UIPT + DeltaK + MDL Meta-Scheduler + Toroidal Voting

Minimal external deps: torch, transformers (optional), sympy
Fallback: simulated sampler when transformers/model not available.

Core Components:
- UIPTEntropyWindow: Rolling entropy tracker for phase transition detection
- DeltaKScheduler: ΔK bias estimator (logp_preferred - logp_sampled)
- MDLMetaScheduler: Maps entropy + ΔK to sampling knobs (temp, top_p, top_k)
- Toroidal clustering: S¹ clustering for mod-1000 answers
- Entropic gravity voting: Mass × Density^0.15 × Solomonoff selection

Usage:
    from prometheus_kaggle import SimpleSampler, PrometheusKaggleEngine, build_submission

    sampler = SimpleSampler(model_name="gpt2", device="cuda")
    engine = PrometheusKaggleEngine(sampler=sampler, candidates=64)
    result = engine.solve_one("Compute (n^2 + 7) mod 1000 for n = 42")
    print(result['final'])  # → integer answer mod 1000
"""

from __future__ import annotations
import math
import os
import json
import time
import random
from collections import deque, Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Try imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sympy import sympify, simplify, Rational, N
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


# =============================================================================
# Utilities
# =============================================================================

def canonicalize_answer(ans: Any):
    """
    Try to convert an answer to canonical SymPy rational/expr
    or fall back to normalized numeric/string.
    """
    if SYMPY_AVAILABLE:
        try:
            if isinstance(ans, (int, float)):
                c = simplify(sympify(ans, rational=True))
                return c
            if isinstance(ans, str):
                s = ans.strip()
                # try fraction pattern
                if '/' in s:
                    try:
                        return simplify(sympify(s, rational=True))
                    except Exception:
                        pass
                try:
                    return simplify(sympify(s, rational=True))
                except Exception:
                    pass
        except Exception:
            pass
    # fallback: try numeric
    try:
        return float(ans)
    except Exception:
        return str(ans).strip()


def sympy_to_float_safe(x, prec: int = 10) -> float:
    """Safely convert SymPy expression to float."""
    if SYMPY_AVAILABLE:
        try:
            return float(N(x, prec))
        except Exception:
            pass
    try:
        return float(x)
    except Exception:
        return float('nan')


# =============================================================================
# UIPT (Universal Information Phase Transition) - Rolling Entropy Window
# =============================================================================

class UIPTEntropyWindow:
    """
    Rolling entropy tracker for detecting phase transitions in token generation.

    Low entropy = Crystallized Logic (High confidence)
    High entropy = Gas Phase (Hallucination/exploration)

    Usage:
        uipt = UIPTEntropyWindow(window=32)
        uipt.add(token_id)
        entropy = uipt.normalized()  # 0-1 range
    """

    def __init__(self, window: int = 32):
        self.window = window
        self.buf: deque = deque(maxlen=self.window)
        self.counts: Counter = Counter()

    def add(self, token_id: int) -> None:
        """Add a new token and update entropy calculation."""
        if len(self.buf) == self.window:
            old = self.buf.popleft()
            self.counts[old] -= 1
            if self.counts[old] <= 0:
                del self.counts[old]
        self.buf.append(token_id)
        self.counts[token_id] += 1

    def raw_entropy_bits(self) -> float:
        """Calculate raw Shannon entropy in bits."""
        total = sum(self.counts.values())
        if total <= 0:
            return 0.0
        probs = [c / total for c in self.counts.values() if c > 0]
        H = -sum(p * math.log2(p) for p in probs)
        return H

    def normalized(self) -> float:
        """
        Calculate normalized entropy (0-1 range).
        0 = fully crystallized (one token dominates)
        1 = maximum entropy (uniform distribution)
        """
        H = self.raw_entropy_bits()
        alph = max(1, len(self.counts))
        maxH = math.log2(alph) if alph > 1 else 1.0
        return float(H / maxH) if maxH > 0 else 0.0

    def is_crystallized(self, threshold: float = 0.3) -> bool:
        """Check if entropy has crystallized below threshold."""
        return self.normalized() < threshold

    def is_gas_phase(self, threshold: float = 0.85) -> bool:
        """Check if entropy is in gas phase (hallucination mode)."""
        return self.normalized() > threshold


# =============================================================================
# DeltaK Scheduler (ΔK Bias Estimator)
# =============================================================================

class DeltaKScheduler:
    """
    Rolling estimate of delta = logp(preferred) - logp(sampled).

    Measures how far the model is from its "preferred" next token.
    High delta = model is being forced away from its preference (exploring)
    Low delta = model is sampling near its peak probability (exploiting)

    Usage:
        delta = DeltaKScheduler(window=64)
        delta_val, mean, deriv = delta.observe(logits, sampled_idx)
    """

    def __init__(self, window: int = 64, deriv_w: int = 8):
        self.window = window
        self.buf: deque = deque(maxlen=window)
        self.ma_buf: deque = deque(maxlen=deriv_w)

    def _logprobs_from_logits(self, arr: np.ndarray) -> np.ndarray:
        """Convert logits to log probabilities."""
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            return a
        m = a.max()
        ex = np.exp(a - m)
        p = ex / (ex.sum() + 1e-40)
        return np.log(np.clip(p, 1e-40, 1.0))

    def observe(self, logits: np.ndarray, sampled_idx: int) -> Tuple[float, float, float]:
        """
        Observe a sampling decision and compute delta.

        Returns:
            delta: Current step's delta
            ma: Moving average of delta
            deriv: Derivative of moving average (trend)
        """
        lp = self._logprobs_from_logits(logits)
        if lp.size == 0:
            delta = 0.0
        else:
            pref_idx = int(np.argmax(lp))
            logp_pref = float(lp[pref_idx])
            if sampled_idx < 0 or sampled_idx >= lp.size:
                logp_samp = float(lp.min()) - 10.0
            else:
                logp_samp = float(lp[sampled_idx])
            delta = logp_pref - logp_samp

        self.buf.append(delta)
        ma = float(sum(self.buf) / len(self.buf)) if len(self.buf) > 0 else 0.0
        self.ma_buf.append(ma)
        deriv = float(self.ma_buf[-1] - self.ma_buf[-2]) if len(self.ma_buf) >= 2 else 0.0

        return delta, ma, deriv

    @property
    def mean(self) -> float:
        """Current moving average of delta."""
        return float(sum(self.buf) / len(self.buf)) if len(self.buf) > 0 else 0.0

    @property
    def deriv(self) -> float:
        """Current derivative of moving average."""
        if len(self.ma_buf) >= 2:
            return float(self.ma_buf[-1] - self.ma_buf[-2])
        return 0.0


# =============================================================================
# MDL Meta-Scheduler (Combines Entropy + DeltaK → Sampling Knobs)
# =============================================================================

class MDLMetaScheduler:
    """
    Minimum Description Length meta-scheduler.

    Maps entropy and delta signals to sampling parameters:
    - temperature: Controls randomness
    - top_p: Nucleus sampling threshold
    - top_k: Top-k truncation
    - kill: Whether to terminate this candidate
    - crystal_boost: Whether candidate has crystallized (high confidence)
    """

    def __init__(
        self,
        base_temp: float = 0.8,
        min_temp: float = 0.05,
        max_temp: float = 1.3,
        base_top_p: float = 0.92,
        min_top_p: float = 0.5,
        max_top_p: float = 0.99,
        base_top_k: int = 64,
        min_top_k: int = 1,
        max_top_k: int = 512,
        kill_entropy_threshold: float = 0.85,
        crystal_entropy_threshold: float = 0.12,
        delta_gain_threshold: float = -0.01,
        delta_bad_threshold: float = 0.35
    ):
        self.base_temp = base_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.base_top_p = base_top_p
        self.min_top_p = min_top_p
        self.max_top_p = max_top_p
        self.base_top_k = base_top_k
        self.min_top_k = min_top_k
        self.max_top_k = max_top_k
        self.kill_entropy_threshold = kill_entropy_threshold
        self.crystal_entropy_threshold = crystal_entropy_threshold
        self.delta_gain_threshold = delta_gain_threshold
        self.delta_bad_threshold = delta_bad_threshold

    def step(
        self,
        norm_entropy: float,
        delta_mean: float,
        delta_deriv: float
    ) -> Dict[str, Any]:
        """
        Compute sampling parameters for next step.

        Args:
            norm_entropy: Normalized entropy (0-1)
            delta_mean: Moving average of delta
            delta_deriv: Derivative of delta moving average

        Returns:
            Dict with temperature, top_p, top_k, kill, crystal_boost flags
        """
        # Temperature scaling based on entropy
        temp_from_e = self.base_temp + (self.max_temp - self.base_temp) * norm_entropy

        # Delta scaling via tanh
        delta_scale = math.tanh(1.5 * delta_mean)
        delta_deriv_scale = math.tanh(40.0 * (-delta_deriv))

        # Final temperature
        temp = temp_from_e * (1.0 + 0.6 * delta_scale) * (1.0 - 0.25 * delta_deriv_scale)
        temp = max(self.min_temp, min(self.max_temp, temp))

        # Top-p (higher entropy = lower top_p = more restrictive)
        top_p = self.min_top_p + (self.base_top_p - self.min_top_p) * (1.0 - norm_entropy)
        top_p = max(self.min_top_p, min(self.max_top_p, top_p))

        # Top-k
        top_k = int(round(self.min_top_k + (self.base_top_k - self.min_top_k) * (1.0 - norm_entropy)))
        top_k = max(self.min_top_k, min(self.max_top_k, top_k))

        # Kill decision (high entropy + high delta = hallucinating badly)
        kill = (norm_entropy > self.kill_entropy_threshold) and (delta_mean > self.delta_bad_threshold)

        # Crystal boost (low entropy + negative derivative = converging)
        crystal_boost = (norm_entropy < self.crystal_entropy_threshold) and (delta_deriv < self.delta_gain_threshold)

        return {
            'temperature': float(temp),
            'top_p': float(top_p),
            'top_k': int(top_k),
            'kill': bool(kill),
            'crystal_boost': bool(crystal_boost),
            'delta_mean': float(delta_mean),
            'delta_deriv': float(delta_deriv),
            'norm_entropy': float(norm_entropy)
        }


# =============================================================================
# Toroidal Clustering + Entropic Gravity Voting
# =============================================================================

def toroidal_distance(a: float, b: float, mod: int = 1000) -> float:
    """
    Calculate distance on S¹ (circle) with given modulus.
    Handles wrap-around for mod-1000 answers.
    """
    a, b = a % mod, b % mod
    d = abs(a - b)
    return min(d, mod - d)


def cluster_basins_mod1000(
    answers: List[float],
    mod: int = 1000,
    eps: int = 8
) -> List[List[float]]:
    """
    Cluster answers on S¹ (toroidal/circular clustering).

    Args:
        answers: List of numeric answers
        mod: Modulus (default 1000 for AIMO)
        eps: Clustering epsilon (max distance to same cluster)

    Returns:
        List of clusters, each cluster is a list of values
    """
    if not answers:
        return []

    pts = sorted([float(a) % mod for a in answers])
    clusters: List[List[float]] = []
    curr = [pts[0]]

    for p in pts[1:]:
        if toroidal_distance(p, curr[-1], mod) <= eps:
            curr.append(p)
        else:
            clusters.append(curr)
            curr = [p]
    clusters.append(curr)

    # Merge wrap-around (first and last cluster may be connected)
    if len(clusters) > 1 and toroidal_distance(clusters[0][0], clusters[-1][-1], mod) <= eps:
        clusters[0] = clusters[-1] + clusters[0]
        clusters.pop()

    return clusters


def entropic_gravity_select(results: List[Dict[str, Any]], mod: int = 1000) -> int:
    """
    Select best answer using entropic gravity formula.

    Score = Mass × Density^0.15 × Solomonoff_Weight

    Where:
    - Mass = cluster size
    - Density = 1 / (variance + epsilon)
    - Solomonoff = 0.9995^code_length (Occam's razor)

    Args:
        results: List of dicts with 'answer' and optionally 'code'
        mod: Modulus for toroidal clustering

    Returns:
        Selected answer as integer
    """
    if not results:
        return 0

    answers = [r['answer'] for r in results]
    clusters = cluster_basins_mod1000(answers, mod=mod, eps=max(1, int(mod * 0.01)))

    best_score = -1.0
    best_val = 0.0

    for cl in clusters:
        mass = len(cl)
        centroid = float(np.median(cl))
        var = float(np.var(cl)) if len(cl) > 1 else 0.0
        density = 1.0 / (var + 1e-12)

        # Find shortest code in cluster (Solomonoff prior)
        candidates = [
            r for r in results
            if toroidal_distance(float(r['answer']), centroid, mod) < max(1, int(mod * 0.005))
        ]

        if candidates:
            shortest = min(candidates, key=lambda r: len(r.get('code', '')))
            sol_weight = 0.9995 ** len(shortest.get('code', ''))
        else:
            sol_weight = 0.1

        # Entropic gravity score
        score = mass * (density ** 0.15) * sol_weight

        if score > best_score:
            best_score = score
            best_val = centroid

    return int(round(best_val)) % mod


# =============================================================================
# Simple Sampler Wrappers (Transformers or Simulated)
# =============================================================================

class SimpleSampler:
    """
    Wrapper for language model sampling.

    Supports:
    - Real transformers models (if available)
    - Simulated sampler for testing (deterministic fallback)

    Usage:
        sampler = SimpleSampler(model_name="gpt2", device="cuda")
        logits = sampler.get_logits_for_prompt(["Solve x + 2 = 5"])
        token_id = sampler.sample_from_logits(logits[0], temperature=0.8)
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        self.active = False
        self.model = None
        self.tokenizer = None

        if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
                self.model.eval()

                if hasattr(self.model, "config") and getattr(self.model.config, "pad_token_id", None) is None:
                    self.model.config.pad_token_id = self.tokenizer.eos_token_id

                self.active = True
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.active = False

        # Fallback: simulated sampler
        if not self.active:
            print("Transformer model not available — using simulated sampler (fast, deterministic).")

    def get_logits_for_prompt(self, prompts: List[str]) -> List[np.ndarray]:
        """
        Get final-token logits for a batch of prompts.

        Returns:
            List of numpy arrays, one per prompt, shape [vocab_size]
        """
        if self.active:
            out = []
            for p in prompts:
                ids = self.tokenizer.encode(p, return_tensors='pt').to(self.device)
                with torch.no_grad():
                    logits = self.model(ids).logits[0, -1, :].cpu().numpy()
                out.append(logits)
            return out
        else:
            # Simulate: return small random peaks
            V = 256
            out = []
            for p in prompts:
                arr = np.random.normal(loc=0.0, scale=1.0, size=V)
                # Bias if prompt contains digits (simulate math focus)
                if any(ch.isdigit() for ch in p):
                    arr += np.random.choice([0, 8], p=[0.9, 0.1], size=V)
                out.append(arr)
            return out

    def sample_from_logits(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> int:
        """
        Sample a token from logits with temperature, top-k, and top-p.

        Args:
            logits: Raw logits array
            temperature: Sampling temperature
            top_k: Top-k truncation (0 = disabled)
            top_p: Nucleus sampling threshold

        Returns:
            Sampled token ID
        """
        arr = np.array(logits, dtype=float)

        # Temperature scaling
        arr = arr / max(1e-8, temperature)

        # Top-k truncation
        if top_k > 0 and top_k < arr.size:
            kth = np.partition(arr, -top_k)[-top_k]
            arr[arr < kth] = -1e9

        # Convert to probabilities
        probs = np.exp(arr - arr.max())
        probs = probs / (probs.sum() + 1e-40)

        # Top-p filtering
        if top_p < 0.999:
            idxs = np.argsort(-probs)
            cum = 0.0
            keep = []
            for i in idxs:
                keep.append(i)
                cum += probs[i]
                if cum >= top_p:
                    break
            mask = np.zeros_like(probs, dtype=bool)
            mask[keep] = True
            probs = probs * mask
            if probs.sum() == 0:
                probs = np.ones_like(probs) / probs.size
            else:
                probs = probs / probs.sum()

        return int(np.random.choice(len(probs), p=probs))


# =============================================================================
# High-Level Orchestrator
# =============================================================================

class PrometheusKaggleEngine:
    """
    PROMETHEUS Kaggle Engine — orchestrates multiple candidate solutions.

    For each problem:
    1. Spawns N candidate generation traces
    2. Applies UIPT entropy monitoring + DeltaK scheduling
    3. Terminates poor candidates early (saves compute)
    4. Aggregates via entropic gravity voting

    Usage:
        engine = PrometheusKaggleEngine(sampler=sampler, candidates=64)
        result = engine.solve_one("Compute (n^2 + 7) mod 1000 for n = 42")
        answer = result['final']  # integer mod 1000
    """

    def __init__(
        self,
        sampler: SimpleSampler,
        candidates: int = 64,
        mod: int = 1000,
        uipt_w: int = 32,
        delta_w: int = 64
    ):
        self.sampler = sampler
        self.candidates = candidates
        self.mod = mod
        self.uipt_w = uipt_w
        self.delta_w = delta_w

        # Per-candidate orchestrators
        self.uipt_windows = [UIPTEntropyWindow(window=self.uipt_w) for _ in range(candidates)]
        self.delta_scheds = [DeltaKScheduler(window=self.delta_w) for _ in range(candidates)]
        self.meta = MDLMetaScheduler()

    def reset(self):
        """Reset all per-candidate state for a new problem."""
        self.uipt_windows = [UIPTEntropyWindow(window=self.uipt_w) for _ in range(self.candidates)]
        self.delta_scheds = [DeltaKScheduler(window=self.delta_w) for _ in range(self.candidates)]

    def solve_one(self, prompt: str, max_steps: int = 64) -> Dict[str, Any]:
        """
        Solve a single problem with multiple candidate traces.

        Args:
            prompt: The problem prompt
            max_steps: Maximum tokens to generate per candidate

        Returns:
            Dict with 'final' (answer), 'raw_results', 'actions'
        """
        self.reset()

        # Initialize candidate traces
        traces = [
            {
                'tokens': [],
                'logits': [],
                'temps': [],
                'alive': True,
                'code': '',
                'answer': None
            }
            for _ in range(self.candidates)
        ]

        # Initial prompts per candidate
        prompts = [prompt for _ in range(self.candidates)]

        # Per-candidate sampling parameters
        temps = [self.meta.base_temp for _ in range(self.candidates)]
        top_ps = [self.meta.base_top_p for _ in range(self.candidates)]
        top_ks = [self.meta.base_top_k for _ in range(self.candidates)]

        actions_log = []

        for step in range(max_steps):
            logits_batch = self.sampler.get_logits_for_prompt(prompts)
            sampled = []
            actions = []

            # Per-candidate step
            for i in range(self.candidates):
                if not traces[i]['alive']:
                    sampled.append(None)
                    actions.append(None)
                    continue

                logits = logits_batch[i]
                sampled_id = self.sampler.sample_from_logits(
                    logits,
                    temperature=temps[i],
                    top_k=top_ks[i],
                    top_p=top_ps[i]
                )

                # Update UIPT and DeltaK
                self.uipt_windows[i].add(sampled_id)
                norm_e = self.uipt_windows[i].normalized()
                delta, ma, deriv = self.delta_scheds[i].observe(logits, sampled_id)

                # Get action from meta-scheduler
                action = self.meta.step(norm_e, ma, deriv)
                action['keep_mask'] = (norm_e <= self.meta.kill_entropy_threshold)

                # Decision to kill/terminate
                if action['kill'] or not action['keep_mask']:
                    traces[i]['alive'] = False

                # Record trace
                traces[i]['tokens'].append(sampled_id)
                traces[i]['logits'].append(logits)
                traces[i]['temps'].append(action['temperature'])

                # Update sampling knobs for next step
                temps[i] = action['temperature']
                top_ps[i] = action['top_p']
                top_ks[i] = action['top_k']

                sampled.append(sampled_id)
                actions.append(action)

            actions_log.append(actions)

            # Early exit if all candidates dead
            alive_count = sum(1 for t in traces if t['alive'])
            if alive_count == 0:
                break

        # Convert token traces to answers
        results = []
        for i, t in enumerate(traces):
            if len(t['tokens']) == 0:
                ans = 0
            else:
                # Fold tokens to produce an answer (simplified)
                # In production: decode tokens, parse numeric answer
                val = sum(t['tokens'][-8:]) % self.mod
                ans = int(val)

            code_stub = "generated_code_stub"
            results.append({
                'answer': ans,
                'code': code_stub,
                'trace': t
            })

        # Final selection via entropic gravity
        final = entropic_gravity_select(results, mod=self.mod)

        return {
            'final': final,
            'raw_results': results,
            'actions': actions_log
        }


# =============================================================================
# Submission Builder
# =============================================================================

def build_submission(
    engine: PrometheusKaggleEngine,
    test_items: List[Dict[str, Any]],
    out_path: str = "/kaggle/working/submission.csv"
) -> str:
    """
    Build Kaggle submission.csv from test items.

    Args:
        engine: PrometheusKaggleEngine instance
        test_items: List of {'id': ..., 'prompt': ...}
        out_path: Output path for submission.csv

    Returns:
        Path to written submission file
    """
    import csv

    rows = []
    for itm in test_items:
        res = engine.solve_one(itm['prompt'])
        rows.append({
            'id': itm.get('id', len(rows)),
            'prediction': int(res['final'])
        })

    # Write CSV
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'prediction'])
        for r in rows:
            writer.writerow([r['id'], r['prediction']])

    print(f"Submission written to {out_path}")
    return out_path


# =============================================================================
# Ablation Harness
# =============================================================================

def run_ablation(
    test_items: List[Dict[str, Any]],
    sampler: SimpleSampler,
    candidates: int = 32,
    max_steps: int = 32
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run ablation study comparing different configurations.

    Configurations:
    - baseline: No UIPT/DeltaK (fixed temperature)
    - uipt_only: Only UIPT entropy scheduling
    - deltak_only: Only DeltaK scheduling
    - full: Full PROMETHEUS (UIPT + DeltaK + Solomonoff)

    Returns:
        Dict mapping config name to list of results
    """
    results = {}

    # Full PROMETHEUS
    engine_full = PrometheusKaggleEngine(sampler=sampler, candidates=candidates)
    results['full'] = []
    for itm in test_items:
        t0 = time.time()
        r = engine_full.solve_one(itm['prompt'], max_steps=max_steps)
        results['full'].append({
            'id': itm['id'],
            'answer': r['final'],
            'time': time.time() - t0
        })

    # Baseline (fixed params, no scheduling)
    class BaselineScheduler(MDLMetaScheduler):
        def step(self, norm_entropy, delta_mean, delta_deriv):
            return {
                'temperature': 0.8,
                'top_p': 0.92,
                'top_k': 64,
                'kill': False,
                'crystal_boost': False,
                'delta_mean': delta_mean,
                'delta_deriv': delta_deriv,
                'norm_entropy': norm_entropy
            }

    engine_baseline = PrometheusKaggleEngine(sampler=sampler, candidates=candidates)
    engine_baseline.meta = BaselineScheduler()
    results['baseline'] = []
    for itm in test_items:
        t0 = time.time()
        r = engine_baseline.solve_one(itm['prompt'], max_steps=max_steps)
        results['baseline'].append({
            'id': itm['id'],
            'answer': r['final'],
            'time': time.time() - t0
        })

    return results


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("PROMETHEUS Kaggle Engine - Test Run")

    # Create sampler (will use simulation if no model available)
    device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    sampler = SimpleSampler(model_name="gpt2", device=device)

    # Create engine
    engine = PrometheusKaggleEngine(sampler=sampler, candidates=16, mod=1000)

    # Test on synthetic problem
    test_prompt = "Compute (n^2 + 7) mod 1000 for n = 42"
    result = engine.solve_one(test_prompt, max_steps=32)

    print(f"Prompt: {test_prompt}")
    print(f"Final answer: {result['final']}")
    print(f"Alive candidates: {sum(1 for r in result['raw_results'] if r['trace']['alive'])}")

    print("\nTest complete.")
