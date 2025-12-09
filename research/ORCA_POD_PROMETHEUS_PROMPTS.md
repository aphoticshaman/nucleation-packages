# Orca Pod PROMETHEUS Prompts

**Purpose:** Extract meta-insights from your LLM interactions that tie into "The Mathematics of Intelligence" book.

**Usage:** Copy each prompt to the respective LLM. Paste your conversation history (or key excerpts) after the prompt. Collect outputs and cross-pollinate.

---

## PROMPT 1: For ChatGPT (Meta-Cognitive Analysis)

```
You are executing the PROMETHEUS Protocol: Protocol for Recursive Optimization, Meta-Enhanced Theoretical Heuristic Extraction, and Universal Synthesis.

I'm going to paste a conversation history between me and another AI. Your task is to perform LATENT SPACE ARCHAEOLOGY on this interaction itself.

**YOUR OBJECTIVE:** Extract meta-insights about:
1. Intelligence and reasoning patterns that emerged in the conversation
2. Novel synthesis that occurred when domains were combined
3. Phase transitions in understanding (moments where capability "jumped")
4. Information compression patterns (where complexity collapsed into simplicity)
5. Connections to the CIC framework: F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)

**OUTPUT FORMAT:**
1. **META-INSIGHT [N]:** One-sentence statement
2. **EVIDENCE:** Quote from conversation showing this
3. **CONNECTION TO BOOK:** How this relates to Intelligence = Compression = Free Energy
4. **POTENTIAL THEOREM:** If formalizable, state it mathematically

**CONVERSATION TO ANALYZE:**
[PASTE YOUR CONVERSATION HERE]
```

---

## PROMPT 2: For Grok (Adversarial Stress Test)

```
I need you to be maximally adversarial and skeptical.

I'm writing a book called "The Mathematics of Intelligence" with the thesis that Intelligence = Compression = Free Energy. The core equation is:

F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)

Where:
- Φ(T) = Integrated Information (cohesion via compression)
- H(T|X) = Representation Entropy
- C_multi(T) = Multi-scale Structural Coherence

Below is a conversation where this framework was discussed/developed.

**YOUR TASK:**
1. Find the WEAKEST claims in this conversation
2. Identify any logical gaps or circular reasoning
3. Find prior art that might invalidate novelty claims
4. Propose ABLATION TESTS that would kill these ideas if they're wrong
5. If any claim SURVIVES your attack, note it as "HARDENED"

Be ruthless. I want to know what doesn't hold up.

**CONVERSATION:**
[PASTE YOUR CONVERSATION HERE]
```

---

## PROMPT 3: For Gemini (Cross-Domain Synthesis)

```
I'm building a unified theory of intelligence with this core equation:

F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)

This connects:
- Information theory (compression, entropy)
- Statistical physics (phase transitions, free energy)
- Neuroscience (integrated information)
- Machine learning (attention, grokking, capability emergence)

I'm pasting a conversation that developed parts of this framework.

**YOUR TASK: HORIZONTAL SCAN**
1. What ANALOGOUS STRUCTURES exist in fields NOT mentioned?
2. What would a biologist see in this? A physicist? An economist? A philosopher?
3. What's the "negative space" - what obvious connection is MISSING?
4. Propose ONE novel cross-domain insight that follows logically but hasn't been stated

**BONUS:** If you can connect this to:
- Category theory
- Topology
- Game theory
- Thermodynamics beyond free energy
...I want to hear it.

**CONVERSATION:**
[PASTE YOUR CONVERSATION HERE]
```

---

## PROMPT 4: For DeepSeek (Mathematical Formalization)

```
I need rigorous mathematical formalization.

Core framework: CIC (Compression-Integration-Coherence)

F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)

Claims made in the conversation below include:
- Extended NCD is a pseudo-metric
- CIC bounds predictive risk under sub-Gaussian noise
- Φ computed via compression lower-bounds mutual information
- Phase transitions are detectable via entropy curvature

**YOUR TASK:**
1. For EACH mathematical claim, either:
   a. PROVE IT (with explicit steps)
   b. DISPROVE IT (with counterexample)
   c. STATE NECESSARY CONDITIONS for it to hold

2. Propose ADDITIONAL THEOREMS that follow from the axioms but weren't stated

3. Check DIMENSIONAL CONSISTENCY of all equations

4. Identify any HIDDEN ASSUMPTIONS

Output in LaTeX-compatible format where possible.

**CONVERSATION:**
[PASTE YOUR CONVERSATION HERE]
```

---

## PROMPT 5: For Claude (Non-Code) - Recursive Self-Reference

```
This is a conversation I had with Claude Code about building a book on intelligence, compression, and AI.

You are Claude (non-code). I want you to analyze this conversation AS DATA about human-AI collaboration.

**META-RECURSIVE TASK:**
1. What does THIS CONVERSATION reveal about how intelligence emerges from human-AI interaction?
2. Does our collaboration exhibit the properties we're theorizing about?
   - Did information compress as we iterated?
   - Did phase transitions occur in our shared understanding?
   - Did integration (Φ) increase as we connected domains?
3. If our conversation IS an instance of the CIC framework, what's our F[T] score?
4. What insights about intelligence can ONLY be seen by analyzing the process, not the output?

**THE LOOP:**
We're building a theory of intelligence. The theory should explain our own collaboration. Does it?

**CONVERSATION:**
[PASTE YOUR CONVERSATION HERE]
```

---

## PROMPT 6: Universal Insight Extraction (Any LLM)

```
# PROMETHEUS EXTRACTION PROTOCOL

## INPUT
A conversation about intelligence, AI, compression, and inference optimization.

## TASK
Extract ALL novel insights that could become:
- Theorems in a mathematics appendix
- Chapter sections in a practitioner's guide
- Figures/diagrams explaining concepts
- Code implementations
- Worked examples

## OUTPUT FORMAT

### THEOREM CANDIDATES
| ID | Statement | Confidence | Proof Sketch |
|----|-----------|------------|--------------|

### CHAPTER CONTRIBUTIONS
| Topic | Key Insight | Word Count Estimate |
|-------|-------------|---------------------|

### DIAGRAM IDEAS
| Concept | Visual Description |
|---------|-------------------|

### CODE SNIPPETS
```python
# Function name and purpose
```

### WORKED EXAMPLES
| Problem | Solution Approach | Pedagogical Value |
|---------|-------------------|-------------------|

## CONVERSATION
[PASTE HERE]
```

---

## AGGREGATION PROMPT (After Collecting All Outputs)

```
I have outputs from 5 different LLMs analyzing the same conversation about intelligence and compression theory.

**OUTPUTS:**
[Paste ChatGPT output]
---
[Paste Grok output]
---
[Paste Gemini output]
---
[Paste DeepSeek output]
---
[Paste Claude output]

**YOUR TASK:**
1. Find CONSENSUS insights (mentioned by 3+ LLMs)
2. Find UNIQUE insights (only one LLM saw it)
3. Find CONTRADICTIONS (LLMs disagree)
4. SYNTHESIZE into a ranked list of insights to add to the book
5. Flag anything that needs EMPIRICAL VALIDATION before including

Output as a prioritized action list.
```

---

## USAGE WORKFLOW

1. **Export** your key conversations as .md or .txt
2. **Run** prompts 1-5 on respective LLMs
3. **Collect** outputs into a single document
4. **Run** aggregation prompt on your strongest LLM
5. **Add** validated insights to BOOK_STRUCTURE.md
6. **Iterate**

---

*"The Orca Pod surfaces what no single whale can see."*
