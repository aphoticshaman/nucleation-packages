#!/usr/bin/env python3
"""
TROPICAL ATTENTION THEOREM - QWEN FORMALIZATION
================================================
Tailored for Qwen-Math: Assumes algebraic maturity, 
tests deeper structural understanding.

The proof: Attention mechanisms in the T→∞ limit become
tropical polynomial evaluators, implying transformers
compute piecewise-linear functions whose complexity
is bounded by tropical geometry.
"""

import requests
import json
import time
import os

URL = "http://localhost:8000"
RESULTS_FILE = "/workspace/qwen_tropical_results.json"

PROOF_STEPS = [
    # Abstract algebra foundation
    {
        "id": "Q_SEMIRING",
        "prompt": """
The tropical semiring (T, ⊕, ⊗) has:
- T = R ∪ {-∞}
- a ⊕ b = max(a, b)
- a ⊗ b = a + b
- Identity for ⊕: -∞
- Identity for ⊗: 0

Verify: In this semiring, the "tropical determinant" of a 2×2 matrix 
[[a,b],[c,d]] is max(a⊗d, b⊗c) = max(a+d, b+c).

For matrix [[3,1],[2,5]], tropical det = max(3+5, 1+2) = ?
""",
        "answer": 8,
    },
    
    {
        "id": "Q_POLYNOMIAL",
        "prompt": """
A tropical polynomial p(x) = ⊕_i (c_i ⊗ x^{⊗i}) where x^{⊗i} = i·x.

p(x) = max_i(c_i + i·x)

This is the maximum of linear functions → piecewise linear.

For p(x) = 5 ⊕ (2⊗x) ⊕ ((-1)⊗x⊗x) = max(5, 2+x, -1+2x):
At what value of x do all three terms equal? Solve:
5 = 2+x → x=3
5 = -1+2x → x=3  
2+x = -1+2x → x=3

All meet at x=3. Value at x=3: max(5, 5, 5) = ?
""",
        "answer": 5,
    },
    
    {
        "id": "Q_SOFTMAX_LIMIT",
        "prompt": """
Key limit: For temperature T and scores s = [s_1, ..., s_n]:

T · log(Σ_i exp(s_i/T)) → max_i(s_i) as T → ∞

This is the log-sum-exp → max correspondence.

Proof sketch: Let m = max(s). Then:
T·log(Σ exp(s_i/T)) = T·log(exp(m/T) · Σ exp((s_i-m)/T))
                    = m + T·log(Σ exp((s_i-m)/T))

As T→∞, each (s_i-m)/T → 0⁻, so exp((s_i-m)/T) → 1 for s_i=m, →0 for s_i<m.

For s = [3, 7, 4, 7, 2], max = 7. How many indices achieve max?
""",
        "answer": 2,
    },
    
    {
        "id": "Q_ATTENTION_TROPICAL",
        "prompt": """
Standard attention: Attn(Q,K,V) = softmax(QK^T/√d) · V

In the T→∞ limit with temperature T·√d:
- softmax_T(QK^T) → one-hot on argmax
- Weighted sum → selection of single V row

But log-space attention:
log(Σ_i softmax(s)_i · exp(v_i/T)) ≈ max_i(s_i + v_i)/T

This IS tropical matrix multiplication: (S ⊕ V)_ij = max_k(S_ik + V_kj)

For S = [[1,3],[4,2]] and V = [[5,1],[2,6]]:
Tropical product [0,0] entry: max(1+5, 3+2) = max(6,5) = ?
""",
        "answer": 6,
    },
    
    {
        "id": "Q_EXPRESSIVITY",
        "prompt": """
A tropical rational function is a ratio of tropical polynomials.
Tropical polynomials are piecewise-linear with convex domains.
Tropical rationals are piecewise-linear with polyhedral domains.

The NUMBER of linear regions of a tropical polynomial in R^n of degree d
is bounded by the number of lattice points in the Newton polytope.

For a "generic" degree-d polynomial in n variables:
Upper bound ≈ C(n+d, d)

For n=3 variables, degree d=4:
C(3+4, 4) = C(7,4) = 7!/(4!·3!) = ?
""",
        "answer": 35,
    },
    
    {
        "id": "Q_TRANSFORMER_DEPTH",
        "prompt": """
MAIN THEOREM CONNECTION:

A transformer with L layers, each with attention (tropical max in limit),
followed by MLP (piecewise linear), computes:

f = MLP_L ∘ Attn_L ∘ ... ∘ MLP_1 ∘ Attn_1

Each layer can square the tropical degree (composition).
Starting degree 1, after L layers: degree 2^L.

For 6 layers to achieve degree ≥ 100:
2^6 = 64 < 100
2^7 = 128 ≥ 100

Minimum L such that 2^L ≥ 100?
""",
        "answer": 7,
    },
    
    {
        "id": "Q_ARC_BOUND",
        "prompt": """
ARC IMPLICATION:

An ARC task requires recognizing a pattern that needs k "nested" operations.
This corresponds to tropical degree 2^k.

If a task needs: detect symmetry (1) → find axis (2) → reflect (3) → recolor (4)
That's k=4 nested operations.

Tropical degree needed: 2^4 = ?
""",
        "answer": 16,
    },
    
    {
        "id": "Q_VERIFICATION",
        "prompt": """
Final verification of the theorem:

lim_{T→∞} Transformer_T(x) is piecewise-linear with:
- Number of pieces bounded by tropical geometry
- Each piece computable by linear algebra
- Composition depth = tropical degree exponent

The theorem implies: infinite-temperature transformers are 
computationally equivalent to tropical circuits.

Tropical circuits with n inputs, depth d, width w compute
functions with at most w^d linear regions.

For n=10, d=3, w=8: max regions = 8^3 = ?
""",
        "answer": 512,
    },
]

def solve(prompt, temp=0.3):
    try:
        r = requests.post(f"{URL}/solve",
            json={"problem": prompt, "temperature": temp, "mode": "reasoning", "max_tokens": 4096},
            timeout=300)
        return r.json().get("answer"), r.json().get("raw_output", "")[:500]
    except Exception as e:
        return None, str(e)

def main():
    print("="*70)
    print("QWEN TROPICAL ATTENTION - ABSTRACT FORMALIZATION")
    print("="*70)
    
    results = []
    correct = 0
    
    for i, step in enumerate(PROOF_STEPS):
        print(f"\n[{i+1}/{len(PROOF_STEPS)}] {step['id']}")
        
        start = time.time()
        pred, raw = solve(step["prompt"])
        elapsed = time.time() - start
        
        is_correct = pred == step["answer"]
        if is_correct: correct += 1
        status = "✓" if is_correct else "✗"
        
        print(f"  Pred: {pred} | Correct: {step['answer']} {status} ({elapsed:.1f}s)")
        
        results.append({
            "id": step["id"], "predicted": pred, 
            "correct": step["answer"], "is_correct": is_correct,
            "time": elapsed
        })
    
    print(f"\n{'='*70}")
    print(f"RESULT: {correct}/{len(PROOF_STEPS)} = {100*correct/len(PROOF_STEPS):.1f}%")
    print("="*70)
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
