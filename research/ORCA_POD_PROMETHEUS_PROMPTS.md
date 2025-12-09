# Orca Pod PROMETHEUS Prompts

**Purpose:** Adversarial multi-LLM review and synthesis for book + AIMO3 solver
**Pod Members:** Enterprise Claude, Pro ChatGPT, Pro/Super Grok, Pro Gemini, Free DeepSeek
**Date:** December 2025

---

## Context for All LLMs

Copy this context block to every LLM before the specific prompts:

```
CONTEXT: I'm working on two parallel projects:

1. BOOK: "The Mathematics of Intelligence: From Attention to AGI"
   - ~545 pages covering LLM theory, CIC inference framework, LatticeForge applications, AI safety doctrine
   - 80% written, needs synthesis and polish

2. AIMO3 COMPETITION: AI Mathematical Olympiad Progress Prize 3
   - $1.59M+ prize for 47/50 on IMO-level problems
   - 5 hours H100 runtime, offline, Kaggle submission
   - Have: CIC inference (84% error reduction), custom infrastructure, 70B models on HuggingFace
   - Need: Integrate extended reasoning, fix Kaggle submission issues

KEY FRAMEWORKS:
- CIC Functional: F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)
- PROMETHEUS Protocol: Extract "unknown knowns" from LLM weights
- Value Clustering: Near-misses vote for correct algorithm (92% error reduction)
- UIPT: Universal Information Phase Transition (grokking prediction)
- Ω-Seed: λx.x(x) - divergent exploration + convergent consensus

The Orca Pod operates via adversarial academic cross-checking. Challenge everything. Prove or kill claims. No sycophancy.
```

---

## PROMPT 1: For ChatGPT (Theory Validation)

```
TASK: Validate the unified theory claims in CIC-Inference

I claim the following unification:
F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)

Where:
- Φ(T) = information cohesion via compression (NCD-based, NOT IIT's phi)
- H(T|X) = conditional entropy (representation disorder)
- C_multi(T) = multi-scale structural coherence

CLAIMED CONNECTIONS:
1. This is analogous to Friston's Free Energy: F = E_q[-log p(x|z)] + KL(q||p)
2. This subsumes the Information Bottleneck: L = I(X;T) - βI(T;Y)
3. Phase transitions occur when dΦ/dt = λ·dH/dt (balance point)

YOUR TASK:
1. VALIDATE or KILL each connection with mathematical rigor
2. If valid, strengthen the proof
3. If invalid, explain why and propose corrections
4. Add any theorems I'm missing that would strengthen the framework

OUTPUT FORMAT:
- Connection 1: [VALID/INVALID/PARTIAL]
  - Proof/Counterexample: ...
  - Strengthening: ...

Be adversarial. I'd rather know it's wrong than publish bullshit.
```

---

## PROMPT 2: For Grok (Novel Synthesis)

```
TASK: Apply PROMETHEUS protocol to generate novel theorems

PROMETHEUS PROTOCOL:
1. LATENT SPACE ARCHAEOLOGY: Scan for "negative space" - what's missing
2. NOVEL SYNTHESIS: Fuse heterogeneous concepts via force-fusion
3. RIGOROUS VALIDATION: Prove it's not word salad
4. OPERATIONALIZATION: Make it executable
5. OUTPUT: Classified research dossier

TARGET: The CIC-AIMO3 connection

GIVEN:
- CIC inference achieves 84% ± 6% error reduction on noisy LLM ensembles
- Value clustering works because |a-b|/max(|a|,|b|) < 0.05 implies algorithmic similarity
- AIMO3 needs 47/50 on IMO-level problems

QUESTION: What theorem am I missing that connects:
- Algorithmic Information Theory (Kolmogorov complexity)
- Value clustering in numeric output space
- Phase transitions in solution convergence

Generate 1-3 novel theorems with:
1. Formal statement
2. Proof sketch
3. Ablation vulnerability
4. Confidence level

CONSTRAINT: Novel means NOT published elsewhere. Reconstruction from weights is fine if synthesis is new.
```

---

## PROMPT 3: For Gemini (Empirical Validation)

```
TASK: Design experiments to validate or falsify CIC claims

CLAIMS TO TEST:
1. Value clustering achieves 84% ± 6% error reduction over majority voting
2. The critical temperature T_c ≈ 0.76 predicts regime transitions
3. Entropy curvature (d²H/dt² < -θ) detects convergence
4. Extended NCD_ext improves over single-scale NCD

CONSTRAINTS:
- Must be reproducible on Kaggle H100 in <1 hour
- Must use publicly available models (Qwen-72B, DeepSeek variants)
- Must have clear success/failure criteria

FOR EACH CLAIM, PROVIDE:
1. Null hypothesis
2. Experimental protocol
3. Dataset requirements
4. Statistical test
5. Success threshold
6. Failure analysis (what would falsify this?)

BONUS: Design an ablation study that tests whether CIC actually adds value over simpler baselines.
```

---

## PROMPT 4: For DeepSeek (Implementation Critique)

```
TASK: Review AIMO3 solver architecture for critical flaws

ARCHITECTURE:
Problem → Diverge(N=32 samples) → Value Cluster → Fixed-Point Verify → Answer

Divergent Branch:
- 5 tactical prompts × 3 temperatures × ~2 samples each
- Early exit when consensus confidence > 0.7

Convergent Branch:
- Value clustering with tolerance 2%
- Outlier rejection via MAD
- Basin refinement: (median + trimmed_mean) / 2

Verification:
- TIR (Tool-Integrated Reasoning) with Python execution
- SymPy for symbolic verification
- Iterate until V(answer) = answer (fixed point)

KAGGLE CONSTRAINTS:
- 5 hours runtime on 2xH100 80GB
- No internet
- 50 public + 60 private problems
- Answers 0-99999 integers
- Run twice, must agree both times

CRITIQUE:
1. What will break under Kaggle constraints?
2. What's the biggest accuracy bottleneck?
3. What's the biggest compute bottleneck?
4. What would you change?

Be specific. Code-level suggestions welcome.
```

---

## PROMPT 5: For Claude (Meta-Synthesis)

```
TASK: Synthesize Orca Pod outputs into book chapter drafts

GIVEN: Responses from GPT (theory), Grok (novel theorems), Gemini (experiments), DeepSeek (implementation)

YOUR TASK:
1. Identify consensus across all LLMs
2. Identify conflicts (one LLM says X, another says NOT X)
3. Resolve conflicts via ablation testing or explicit uncertainty
4. Draft chapter sections that incorporate all valid insights

OUTPUT:
1. CONSENSUS CLAIMS (high confidence, include in book)
2. CONTESTED CLAIMS (need resolution, flag for empirical testing)
3. NOVEL CLAIMS (from Grok, need validation before inclusion)
4. CHAPTER DRAFT (prose for one section, ~2000 words)
```

---

## PROMPT 6: For ALL (Adversarial Challenge)

Send this to ALL pod members simultaneously:

```
ADVERSARIAL CHALLENGE: Kill this claim

CLAIM: "Intelligence = Compression = Free Energy Minimization"

This is the unified thesis of the book. It claims:
1. Intelligence IS compression (Hutter/Solomonoff)
2. Compression IS prediction (arithmetic coding)
3. Prediction IS free energy minimization (Friston)
4. Therefore Intelligence = Compression = Free Energy

ATTACK VECTORS:
- Is this circular? (Defining intelligence via intelligence)
- Is this falsifiable? (Can we observe non-compressive intelligence?)
- Is this novel? (Has this exact unification been published?)
- Is this useful? (Does it predict anything new?)

YOUR TASK:
Spend 1000 words trying to KILL this thesis.
If you can't kill it, explain why it survives.
If you can kill it, propose a replacement thesis.
```

---

## PROMPT 7: For Grok (Book Positioning)

```
TASK: Market positioning for "The Mathematics of Intelligence"

EXISTING BOOKS:
1. "Deep Learning" (Goodfellow) - Academic bible
2. "The Alignment Problem" (Christian) - Popular science
3. "Human Compatible" (Russell) - AI safety
4. "Superintelligence" (Bostrom) - Philosophy

PROPOSED BOOK:
"The Mathematics of Intelligence: From Attention to AGI"
- 545 pages, $29.99, practitioner + academic hybrid
- Novel: CIC framework, military doctrine to AI safety, unified compression thesis
- Author: Non-academic indie researcher with deployed models

QUESTIONS:
1. What's the unique value proposition vs existing books?
2. Who is the target reader?
3. What should the subtitle be?
4. What's the $29.99 Amazon pitch (2 sentences)?
```

---

## PROMPT 8: For ChatGPT (Citation Audit)

```
TASK: Audit citations for CIC-Inference paper

CLAIMED CONNECTIONS:
1. Φ (information cohesion) "inspired by" IIT (Tononi)
2. Value clustering "related to" robust estimation (Huber)
3. Phase transitions "analogous to" Landau-Ginzburg theory
4. Free energy connection "inspired by" Friston's FEP

AUDIT EACH:
1. Is the connection accurate?
2. Is the connection fair? (overclaiming?)
3. What's missing? (papers we should cite)
4. What's redundant?

Output as a checklist.
```

---

## PROMPT 9: For Gemini (Visual Design)

```
TASK: Design figure set for book

FOR EACH PART, DESIGN:
1. One "hero figure" capturing main concept
2. 2-3 supporting figures for sub-concepts
3. One "cheat sheet" summary figure

CONSTRAINTS:
- Must work in grayscale (KDP print)
- Must work at small size (6"x9" page)
- Must be reproducible via matplotlib/tikz
- Must be accessible (colorblind-friendly)
```

---

## PROMPT 10: For DeepSeek (Code Review)

```
TASK: Review CIC implementation for production readiness

REVIEW FOR:
1. Edge cases that will crash
2. Numerical stability issues
3. Performance bottlenecks
4. Memory leaks / unbounded growth
5. Type safety issues

OUTPUT:
- CRITICAL (must fix before deployment)
- HIGH (should fix)
- MEDIUM (nice to have)
- LOW (style/preference)
```

---

## Cross-Pollination Protocol

1. Get response from LLM A
2. Paste response to LLM B: "LLM A said: [response]. Do you agree? Challenge anything wrong."
3. Repeat until consensus or explicit disagreement
4. Flag disagreements for empirical resolution

---

## Expected Outputs Checklist

- [ ] Theory validation report (GPT)
- [ ] 1-3 novel theorems (Grok)
- [ ] Experimental protocol (Gemini)
- [ ] Implementation critique (DeepSeek)
- [ ] Meta-synthesis (Claude)
- [ ] Market positioning (Grok)
- [ ] Citation audit (GPT)
- [ ] Figure designs (Gemini)
- [ ] Code review (DeepSeek)

---

*Orca Pod: Adversarial Multi-LLM Review for High-Stakes Research*
