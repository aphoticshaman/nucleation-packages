# Chapter 19: Quantum-Inspired Classical Optimization

Quantum computers promise exponential speedups for certain problems. But quantum hardware is finicky, expensive, and limited in scale.

This chapter takes the *ideas* from quantum algorithms and implements them on classical hardware. No qubits required. Just better optimization through quantum-inspired techniques.

---

## The Quantum Advantage (Without Quantum Hardware)

Quantum algorithms succeed because they explore solution spaces differently than classical algorithms.

**Classical search:** Check solutions one by one, or follow gradients
**Quantum search:** Superposition allows exploring many paths simultaneously

We can't achieve true quantum parallelism classically. But we can borrow the *strategies* that make quantum algorithms effective:

- **Tunneling:** Escaping local optima by "tunneling through" barriers
- **Mixing:** Combining solutions in ways that preserve good features
- **Amplitude amplification:** Boosting probability of finding good solutions

These strategies, implemented classically, often outperform standard optimization.

---

## Tunneling-Based Acceptance

### The Local Optima Problem

Standard optimization (gradient descent, simulated annealing) gets stuck in local optima. The algorithm finds a good-ish solution and can't escape because all neighbors are worse.

**Simulated annealing** addresses this by occasionally accepting worse solutions:
```
P(accept worse) = exp(-ΔE / T)
```

This works but has problems:
- Acceptance probability depends on energy difference
- Large barriers require long equilibration
- Temperature schedule requires careful tuning

### Quantum Tunneling Intuition

In quantum mechanics, particles can "tunnel through" energy barriers. A particle trapped in a well can appear on the other side without having enough energy to climb over.

The probability depends on barrier width and height, but thin barriers are nearly transparent.

### Classical Implementation

We implement tunneling-like behavior by accepting moves based on barrier *width* rather than just *height*:

```
ALGORITHM: Tunneling-Based Acceptance
INPUT: Current state x, candidate state x', temperature T

1. Compute direct energy difference: ΔE_direct = E(x') - E(x)

2. Estimate barrier width: Sample random paths from x to x'
   width ≈ min path length through states with E > max(E(x), E(x'))

3. Compute tunneling factor:
   tunnel = exp(-width × barrier_height / T)

4. Acceptance probability:
   P(accept) = min(1, exp(-ΔE_direct/T) + tunnel)
```

The key insight: narrow barriers (even if tall) get higher acceptance than wide barriers (even if short).

### When This Helps

Tunneling acceptance excels when:
- Solution landscape has many local optima
- Optima are separated by thin barriers
- Standard annealing schedule is too slow

Examples:
- Protein folding energy landscapes
- Neural network loss surfaces
- Combinatorial optimization with many similar solutions

---

## QAOA-Inspired Mixing Operators

### The QAOA Framework

The Quantum Approximate Optimization Algorithm (QAOA) alternates between:
- **Cost Hamiltonian:** Encodes objective function
- **Mixer Hamiltonian:** Spreads amplitude across solutions

The mixer doesn't just make random changes—it systematically explores the solution space in ways that preserve structure.

### Classical Mixing

We implement QAOA-style mixing by combining solutions intelligently:

**Uniform mixing (crossover):**
```
mix(x, y)[i] = x[i] if random() < 0.5 else y[i]
```

**Weighted mixing:**
```
mix(x, y)[i] = x[i] if E(x) < E(y) else y[i]
```

**Structure-preserving mixing:**
```
mix(x, y) = solution that preserves features common to x and y
            while varying features where they differ
```

### The Mixing Schedule

Like QAOA's alternating layers, we alternate:

1. **Exploitation phase:** Local search to improve current solutions
2. **Mixing phase:** Combine solutions to explore new regions

The ratio of exploitation to mixing varies with optimization progress:
- Early: More mixing (exploration)
- Late: More exploitation (refinement)

### Population-Based Mixing

With multiple candidate solutions:

```
ALGORITHM: Population Mixing
INPUT: Population {x₁, ..., xₖ}, mixer type, exploitation steps

For each generation:
   1. Exploitation: Apply local_search to each xᵢ
   2. Selection: Keep top 50% by fitness
   3. Mixing: Generate new candidates by mixing survivors
      - Pair survivors randomly
      - Apply structure-preserving mix
      - Add mixed candidates to population
   4. Repeat
```

---

## Grover-Inspired Amplitude Amplification

### Grover's Algorithm Intuition

Grover's search finds a marked item in an unstructured database using O(√N) queries instead of O(N).

The key insight: Grover doesn't just search—it *amplifies* the probability of finding the target. Each iteration increases the amplitude of the correct answer while decreasing others.

### Classical Amplitude Amplification

We can't directly amplify probabilities classically. But we can implement analogous boosting:

**Fitness-proportional sampling:**
Instead of uniform random sampling, sample proportionally to fitness:
```
P(select xᵢ) ∝ fitness(xᵢ)^α
```

Where α controls amplification strength.

**Iterative refinement:**
```
ALGORITHM: Amplitude-Inspired Search
INPUT: Sample generator, fitness function, iterations k

1. Generate initial samples S₀
2. For i = 1 to k:
   a. Compute fitness for all samples
   b. Weight samples by fitness^α
   c. Resample with replacement (weighted)
   d. Perturb samples slightly
3. Return best sample
```

This iteratively concentrates samples around high-fitness regions.

### The √N Intuition

Grover achieves √N speedup because each iteration provides quadratic amplification.

Classically, fitness-proportional selection provides linear concentration. But with multiple iterations:
- First iteration: Concentrate 2x on good regions
- Second iteration: Concentrate 2x again (4x total)
- After k iterations: 2^k concentration

This is exponential in iterations, potentially faster than brute force for hard problems.

---

## Hybrid Classical-Quantum Strategy

Even without quantum hardware, we can structure algorithms to be "quantum-ready":

### The VQE Pattern

Variational Quantum Eigensolver (VQE) uses:
- Classical optimization of parameters
- Quantum evaluation of objective

Classically, we implement the pattern:
- Optimize parameters with gradient-based methods
- Evaluate objective with simulation or sampling

### The QAOA Pattern

- Parameterized alternating layers
- Classical optimization of layer parameters

Classically:
- Alternate exploitation and mixing phases
- Optimize the schedule with meta-learning

### When Quantum Arrives

Code structured this way can be upgraded:
- Replace classical simulation with quantum evaluation
- Keep classical optimization loop
- Gain quantum speedup without rewriting everything

---

## Application: Portfolio Optimization

Portfolio optimization is NP-hard in general. Quantum-inspired techniques help.

### The Problem

Choose asset weights w to maximize risk-adjusted return:
```
maximize: μᵀw - λ × wᵀΣw
subject to: Σwᵢ = 1, wᵢ ≥ 0
```

Where μ is expected returns and Σ is covariance matrix.

### Standard Approaches

**Quadratic programming:** Works for convex case
**Genetic algorithms:** Handle constraints poorly
**Simulated annealing:** Slow convergence

### Quantum-Inspired Approach

**Tunneling acceptance:** Escape local optima in constrained space
**Mixing operators:** Combine portfolios preserving good features
**Amplitude amplification:** Concentrate search on high Sharpe ratio regions

```
ALGORITHM: Quantum-Inspired Portfolio Optimization

1. Initialize population of random valid portfolios

2. For each generation:
   a. Local improvement (exploit):
      - Gradient step on each portfolio
      - Project back to constraint set

   b. Tunneling moves (escape):
      - Propose random rebalancing
      - Accept via tunneling criteria

   c. Mixing (explore):
      - Pair portfolios by similarity
      - Generate children preserving shared positions

   d. Amplification (concentrate):
      - Weight by Sharpe ratio
      - Resample population

3. Return best portfolio found
```

### Results

On standard portfolio optimization benchmarks:
- 15-25% improvement over simulated annealing
- Faster convergence than genetic algorithms
- More robust to local optima

---

## Application: Combinatorial Search

Many practical problems are combinatorial: scheduling, routing, assignment.

### Traveling Salesman

Find shortest tour visiting all cities.

**Classical:** 2-opt, simulated annealing, branch-and-bound
**Quantum-inspired:** Tunneling between tours, mixing preserving subtours

```
Mixing operator for TSP:
- Find longest common subsequence of cities
- Preserve this subsequence in children
- Fill remaining positions with unexplored orderings
```

### Graph Partitioning

Divide graph into equal parts minimizing cut edges.

**Classical:** Spectral methods, Kernighan-Lin
**Quantum-inspired:** Tunneling across partition boundaries, mixing preserving clusters

### Satisfiability

Find variable assignment satisfying boolean formula.

**Classical:** DPLL, WalkSAT
**Quantum-inspired:** Tunneling between assignments, mixing preserving satisfied clauses

---

## Implementation Guidelines

### When to Use Quantum-Inspired Methods

**Good candidates:**
- Rugged fitness landscapes with many local optima
- Problems where solution structure matters (not just random search)
- Moderate problem sizes (100-10,000 variables)

**Poor candidates:**
- Convex problems (standard methods work fine)
- Very small problems (brute force is fast enough)
- Problems without exploitable structure

### Parameter Tuning

**Tunneling temperature:** Start high, anneal down
**Mixing ratio:** Typically 0.3-0.5 of population
**Amplification exponent:** α ≈ 2-4 (higher = more aggressive)

### Computational Cost

Quantum-inspired methods have overhead:
- Barrier estimation requires sampling
- Mixing requires population management
- Amplification requires fitness evaluation

Worth it when:
- Standard methods converge slowly
- Local optima are a significant problem
- Solution quality justifies computation

---

## Summary

Quantum-inspired optimization borrows strategies from quantum algorithms:

**Tunneling acceptance:**
- Escape local optima via barrier-width-based acceptance
- Thin barriers are more permeable than tall ones

**QAOA-inspired mixing:**
- Alternate exploitation and exploration phases
- Structure-preserving combination of solutions

**Grover-inspired amplification:**
- Fitness-proportional sampling
- Iterative concentration on good regions

**Applications:**
- Portfolio optimization: Better risk-adjusted returns
- Combinatorial search: Faster convergence to optima
- Scheduling/routing: Escape from poor local solutions

No quantum hardware required—just better classical algorithms inspired by quantum principles.

The next chapter covers multi-signal fusion: combining information from diverse sources with attention-weighted integration.
