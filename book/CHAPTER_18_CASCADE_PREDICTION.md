# Chapter 18: Cascade Prediction via Epidemiological Models

Disturbances spread. A rumor propagates through social networks. A financial shock cascades through interconnected markets. A failure ripples through dependent systems.

Understanding how things spread—and predicting how far they'll spread—is critical for everything from viral marketing to systemic risk management.

This chapter adapts epidemiological models, originally developed to track disease spread, to predict cascades in information, finance, and infrastructure systems.

---

## Why Epidemiology?

Epidemiologists have spent a century modeling how diseases spread through populations. They've developed sophisticated frameworks for:

- Predicting outbreak size
- Identifying superspreaders
- Evaluating intervention strategies
- Forecasting peak timing

These same dynamics apply to any phenomenon that spreads through connections:
- **Viruses** spread through physical contact
- **Ideas** spread through communication
- **Failures** spread through dependencies
- **Financial stress** spreads through counterparty relationships

The mathematics is identical. Only the interpretation changes.

---

## The SIR Framework

The foundational model is SIR: Susceptible-Infected-Recovered.

### The Compartments

Every node in the network occupies one of three states:

**Susceptible (S):** Has not yet been affected, but could be
- Person who hasn't caught the disease
- User who hasn't seen the viral content
- System component still functioning

**Infected (I):** Currently affected and can spread to others
- Person with active disease
- User actively sharing content
- Failed component causing dependent failures

**Recovered (R):** No longer affected and can't be reinfected
- Person with immunity
- User who's moved on from the content
- Repaired component

### The Dynamics

The model tracks flow between compartments:

**S → I:** Infection/adoption/failure rate
```
dS/dt = -β × S × I / N
```
Where β is the transmission rate and N is total population.

**I → R:** Recovery/burnout/repair rate
```
dI/dt = β × S × I / N - γ × I
```
Where γ is the recovery rate.

**Conservation:**
```
dR/dt = γ × I
S + I + R = N
```

### The Basic Reproduction Number

The critical parameter is R₀:

**R₀ = β / γ**

- **R₀ > 1:** Epidemic grows (each infected node infects more than one other)
- **R₀ < 1:** Epidemic dies out (each infected node infects fewer than one other)
- **R₀ = 1:** Critical threshold

For cascade prediction, estimating R₀ tells you whether the cascade will grow or fade.

---

## Extending to SEIR

Real cascades often have latency—a delay between exposure and infectiousness.

### The Exposed Compartment

**SEIR** adds Exposed (E):

**Exposed (E):** Has been affected but not yet spreading
- Person in incubation period
- User who saw content but hasn't shared yet
- Component under stress but not yet failed

### The Dynamics

```
dS/dt = -β × S × I / N
dE/dt = β × S × I / N - σ × E
dI/dt = σ × E - γ × I
dR/dt = γ × I
```

Where σ is the rate of progression from exposed to infectious.

### Why This Matters

The exposed period creates lag between:
- Initial spread and peak infection
- Intervention and visible effect
- Leading indicators and cascade completion

For prediction, E tells you what's in the pipeline.

---

## Network-Aware Models

SIR and SEIR assume uniform mixing—anyone can infect anyone. Real systems have network structure.

### Network SIR

Instead of differential equations, simulate:

**For each timestep:**
1. Each infected node attempts to infect each susceptible neighbor with probability β·dt
2. Each infected node recovers with probability γ·dt
3. Update compartments

### Network Effects

**Degree distribution:** Highly connected nodes get infected early, spread to many
**Clustering:** Dense local connections speed initial spread but limit total reach
**Community structure:** Cascades can be contained within communities
**Bridge nodes:** Control spread between communities

### Effective R₀

On networks, R₀ depends on structure:

```
R₀_network ≈ β/γ × ⟨k²⟩/⟨k⟩
```

Where ⟨k⟩ is mean degree and ⟨k²⟩ is mean squared degree.

High-variance degree distributions (scale-free networks) have much higher effective R₀.

---

## Cascade Magnitude Prediction

Given early observations, predict final cascade size.

### The Prediction Problem

At time t_early, we observe:
- Number currently infected: I(t_early)
- Number recovered: R(t_early)
- Growth rate: dI/dt at t_early

Predict:
- Total eventual infections: R(∞)
- Peak infection time: t_peak
- Peak infection count: I(t_peak)

### Early-Stage Prediction

During exponential growth phase:

```
I(t) ≈ I(0) × exp(r × t)
```

Where r = β - γ is the growth rate.

Estimate r from early observations:
```
r ≈ log(I(t₂)/I(t₁)) / (t₂ - t₁)
```

### Final Size Prediction

For SIR with known R₀:

```
R(∞)/N = 1 - exp(-R₀ × R(∞)/N)
```

This implicit equation determines final size as function of R₀.

For R₀ = 2: ~80% eventually infected
For R₀ = 1.5: ~58% eventually infected
For R₀ = 1.2: ~36% eventually infected

### Peak Timing Prediction

Peak occurs when dI/dt = 0:

```
t_peak ≈ (1/r) × log(S(0)/(N - S(0) + I(0)))
```

Depends on initial conditions and growth rate.

---

## Applications to Information Spread

### Viral Content Modeling

**Susceptible:** Users who haven't seen content
**Infected:** Users actively sharing content
**Recovered:** Users who've seen it and moved on

**Parameters:**
- β: Sharing probability per impression
- γ: Attention decay rate

**Prediction use cases:**
- Will this content go viral? (R₀ > 1?)
- How many will see it? (R(∞))
- When will it peak? (t_peak)

### Misinformation Spread

Add compartments for fact-checking:

**Susceptible:** Uninformed
**Exposed:** Saw misinformation, not yet decided
**Misinformed:** Believing and spreading misinformation
**Corrected:** Exposed to correction, no longer spreading

Model competition between misinformation spread (β_mis) and correction spread (β_cor).

**Prediction:** Under what conditions does correction outpace misinformation?

---

## Applications to Financial Contagion

### Credit Contagion

**Susceptible:** Healthy institutions
**Exposed:** Institutions with stressed counterparties
**Defaulted:** Failed institutions spreading stress to counterparties
**Resolved:** Institutions that have been restructured/bailed out

**Parameters:**
- β: Counterparty loss rate (how much stress transmits per connection)
- σ: Time from stress to default
- γ: Resolution/bailout rate

**Prediction:** If Bank A defaults, what's the cascade magnitude?

### Market Contagion

**Susceptible:** Assets not yet affected
**Correlated:** Assets showing stress correlation
**Crashed:** Assets that have experienced significant drops
**Stabilized:** Assets that have found new equilibrium

Model spread of volatility through correlation structure.

**Prediction:** Given initial shock to asset class A, which other assets will be affected?

---

## Applications to Infrastructure Cascades

### Power Grid Failures

**Susceptible:** Lines operating within capacity
**Overloaded:** Lines above normal but below failure threshold
**Failed:** Lines that have tripped
**Restored:** Lines returned to service

**Network structure:** Power flow determines which lines receive diverted load when others fail.

**Prediction:** If Line X fails, does cascade self-limit or cause blackout?

### Network Cascade Failures

**Susceptible:** Nodes at normal load
**Stressed:** Nodes receiving redirected traffic
**Failed:** Nodes that have crashed/become unresponsive
**Recovered:** Nodes restored to service

**Parameters depend on:**
- Load distribution before failure
- Routing behavior after failure
- Node capacity margins

---

## Intervention Modeling

Cascades can be controlled. Epidemiological models evaluate interventions.

### Vaccination (Immunization)

Remove nodes from susceptible pool before cascade:
- In diseases: Vaccination
- In information: Pre-bunking, media literacy
- In finance: Capital buffers
- In infrastructure: Redundancy

**Effect:** Reduces effective S, lowers R₀

**Targeting:** Immunizing high-degree nodes has disproportionate effect (degree-weighted immunization)

### Quarantine (Isolation)

Remove infected nodes from spreading:
- In diseases: Isolation
- In information: Content removal
- In finance: Trading halts
- In infrastructure: Load shedding

**Effect:** Reduces effective I, breaks transmission chains

**Timing:** Earlier intervention is exponentially more effective

### Treatment (Faster Recovery)

Increase recovery rate:
- In diseases: Treatment
- In information: Correction campaigns
- In finance: Liquidity injection
- In infrastructure: Fast repair crews

**Effect:** Increases γ, reduces R₀, shortens peak

### Contact Reduction

Reduce transmission probability:
- In diseases: Social distancing
- In information: Algorithm changes
- In finance: Margin requirements
- In infrastructure: Load limits

**Effect:** Reduces β, may bring R₀ below 1

---

## Fitting Models to Data

### Parameter Estimation

Given observed cascade data, estimate parameters:

**Maximum likelihood:**
Find β, γ that maximize probability of observed trajectory

**Bayesian inference:**
Place priors on parameters, update with observations, get posterior distributions with uncertainty

**ABC (Approximate Bayesian Computation):**
Simulate many parameter combinations, accept those producing trajectories similar to observed

### Early Warning Indicators

Before cascade is obvious, look for:

**Increasing R₀:** Growth rate accelerating
```
r(t) = d/dt log(I(t))
```
If r is increasing, cascade is accelerating.

**Decreasing doubling time:**
```
t_double = log(2) / r
```
Faster doubling = stronger cascade.

**Community escape:** Initial containment failing
Monitor spread across network community boundaries.

---

## Practical Implementation

### Simulation Framework

```
ALGORITHM: Network Cascade Simulation
INPUT: Network G, parameters (β, γ, σ), initial infected I₀
OUTPUT: Trajectory over time, final size

1. Initialize compartments:
   - I ← I₀ (initial infected)
   - S ← V \ I₀ (all others susceptible)
   - E ← ∅, R ← ∅

2. For each timestep t:
   a. Exposure: For each (u,v) edge with u ∈ I, v ∈ S:
      - With probability β·dt: move v to E
   b. Infection: For each u ∈ E:
      - With probability σ·dt: move u to I
   c. Recovery: For each u ∈ I:
      - With probability γ·dt: move u to R
   d. Record state

3. Continue until I = ∅ or max time reached

4. Return trajectory and final |R|
```

### Ensemble Prediction

Single simulation has high variance. For prediction:

1. Run many simulations with same parameters
2. Aggregate outcomes for distribution of final sizes
3. Report median, confidence intervals
4. Use CIC framework to identify most likely outcomes

### Real-Time Updating

As cascade progresses:
1. Update parameter estimates with new observations
2. Re-run predictions with updated parameters
3. Report confidence that accounts for uncertainty in both parameters and stochastic outcome

---

## Summary

Epidemiological models predict cascade dynamics:

**SIR framework:** Susceptible → Infected → Recovered with R₀ determining growth
**SEIR extension:** Adds latency through Exposed compartment
**Network models:** Account for connection structure affecting spread

**Key predictions:**
- Final cascade size: R(∞) determined by R₀
- Peak timing: When dI/dt = 0
- Growth rate: r = β - γ in early phase

**Applications:**
- Information spread: Viral content, misinformation
- Financial contagion: Credit chains, market correlation
- Infrastructure cascades: Power grid, network failures

**Interventions:**
- Immunization: Remove susceptibles
- Quarantine: Remove infecteds
- Treatment: Increase recovery
- Contact reduction: Decrease transmission

The next chapter applies quantum-inspired optimization techniques to classical computing: tunneling-based acceptance, mixing operators, and accelerated search.
