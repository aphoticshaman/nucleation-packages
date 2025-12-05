"""
Focus-Gating Abstraction for Tool/Strategy Routing.

Implements Insight #9: Focus-gating is the unifying abstraction for prompt
engineering, tool routing, and neurosymbolic interfaces.

The core abstraction:
    activate edge/tool/strategy iff <focus(x), e> > τ

where e can be:
- An edge in a search graph (Fractal Cascade)
- A tool (calculator, search, prover)
- A symbolic module in a neurosymbolic system

"Prompt engineering" = manipulating focus vector
"Tool routing" = choosing thresholds and embeddings
"Neurosymbolic" = gating between neural and symbolic edges

This subsumes:
- Hand-authored tool triggers
- "if mentions primes, call number-theory" rules
- Arbitrary wormhole heuristics

Key insight: A single learned gating network can replace all these.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
from enum import Enum


class GateType(Enum):
    """Types of gates in the focus-gating system."""
    TOOL = "tool"           # External tool (calculator, search, prover)
    SYMBOLIC = "symbolic"   # Symbolic reasoning module
    NEURAL = "neural"       # Neural continuation
    SEARCH = "search"       # Search graph edge
    PROMPT = "prompt"       # Prompt template/strategy


@dataclass
class Gate:
    """A gatable resource in the inference system."""
    name: str
    gate_type: GateType
    embedding: NDArray[np.float64]  # d-dimensional gate embedding
    threshold: float = 0.5          # Activation threshold
    cost: float = 1.0               # Compute cost (for budgeting)
    priority: int = 0               # Priority when multiple gates activate

    # Callable that executes this gate
    executor: Optional[Callable[[str, Any], str]] = None

    # Gate-specific metadata
    metadata: Dict = field(default_factory=dict)


@dataclass
class FocusVector:
    """Focus vector derived from input/context."""
    vector: NDArray[np.float64]     # d-dimensional focus
    context: str                     # Source context
    attention_weights: Optional[NDArray] = None  # If from attention


@dataclass
class GatingDecision:
    """Result of focus-gating decision."""
    activated_gates: List[Gate]     # Gates that passed threshold
    scores: Dict[str, float]        # All gate scores
    focus: FocusVector              # The focus used
    budget_remaining: float         # Remaining compute budget


class FocusGatingNetwork:
    """
    Unified focus-gating network for tool/strategy routing.

    This replaces:
    1. Hand-coded tool triggers
    2. Regex-based routing
    3. Separate prompt engineering
    4. Ad-hoc neurosymbolic interfaces

    With a single learnable gating mechanism.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        default_threshold: float = 0.5
    ):
        self.embedding_dim = embedding_dim
        self.default_threshold = default_threshold
        self.gates: Dict[str, Gate] = {}

        # Focus projection (can be learned)
        np.random.seed(42)
        self.focus_projection = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.focus_projection = self.focus_projection @ self.focus_projection.T  # Make symmetric

    def register_gate(
        self,
        name: str,
        gate_type: GateType,
        embedding: Optional[NDArray[np.float64]] = None,
        threshold: Optional[float] = None,
        cost: float = 1.0,
        executor: Optional[Callable] = None,
        keywords: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Register a new gate in the system.

        Args:
            name: Unique gate identifier
            gate_type: Type of gate
            embedding: Gate embedding (auto-generated from keywords if None)
            threshold: Activation threshold (uses default if None)
            cost: Compute cost
            executor: Function to execute when gate activates
            keywords: Keywords for auto-embedding generation
            metadata: Additional gate metadata
        """
        if embedding is None:
            if keywords:
                embedding = self._keywords_to_embedding(keywords)
            else:
                embedding = np.random.randn(self.embedding_dim)
                embedding = embedding / np.linalg.norm(embedding)

        gate = Gate(
            name=name,
            gate_type=gate_type,
            embedding=embedding,
            threshold=threshold or self.default_threshold,
            cost=cost,
            executor=executor,
            metadata=metadata or {}
        )
        self.gates[name] = gate

    def _keywords_to_embedding(self, keywords: List[str]) -> NDArray[np.float64]:
        """
        Convert keywords to embedding via simple hash-based method.

        In production, use actual word embeddings or learned projections.
        """
        embedding = np.zeros(self.embedding_dim)
        for kw in keywords:
            # Hash-based pseudo-random projection
            np.random.seed(hash(kw) % (2**31))
            direction = np.random.randn(self.embedding_dim)
            embedding += direction / np.linalg.norm(direction)

        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def compute_focus(
        self,
        context: str,
        attention_weights: Optional[NDArray] = None,
        context_embedding: Optional[NDArray] = None
    ) -> FocusVector:
        """
        Compute focus vector from context.

        In production, this would use:
        - LLM hidden states
        - Attention patterns
        - Learned focus projection

        Args:
            context: Text context
            attention_weights: Optional attention weights from transformer
            context_embedding: Pre-computed context embedding

        Returns:
            FocusVector
        """
        if context_embedding is not None:
            # Use provided embedding
            raw_focus = context_embedding
        else:
            # Simple keyword-based focus (placeholder for learned model)
            raw_focus = self._keywords_to_embedding(context.lower().split())

        # Apply learned projection
        if len(raw_focus) == self.embedding_dim:
            focus = self.focus_projection @ raw_focus
            focus = focus / (np.linalg.norm(focus) + 1e-10)
        else:
            focus = raw_focus

        return FocusVector(
            vector=focus,
            context=context,
            attention_weights=attention_weights
        )

    def gate_scores(self, focus: FocusVector) -> Dict[str, float]:
        """
        Compute activation scores for all gates.

        Args:
            focus: Focus vector

        Returns:
            Dict mapping gate names to scores
        """
        scores = {}
        for name, gate in self.gates.items():
            # Dot product (cosine similarity if normalized)
            score = float(np.dot(focus.vector, gate.embedding))
            scores[name] = score
        return scores

    def decide(
        self,
        focus: FocusVector,
        budget: float = float('inf'),
        max_gates: int = 5
    ) -> GatingDecision:
        """
        Make gating decision based on focus vector.

        Args:
            focus: Focus vector from context
            budget: Maximum total cost to spend
            max_gates: Maximum number of gates to activate

        Returns:
            GatingDecision with activated gates
        """
        scores = self.gate_scores(focus)

        # Filter by threshold
        candidates = [
            (name, score, self.gates[name])
            for name, score in scores.items()
            if score > self.gates[name].threshold
        ]

        # Sort by score * priority
        candidates.sort(key=lambda x: x[1] * (1 + x[2].priority * 0.1), reverse=True)

        # Select within budget
        activated = []
        total_cost = 0.0

        for name, score, gate in candidates:
            if len(activated) >= max_gates:
                break
            if total_cost + gate.cost > budget:
                continue

            activated.append(gate)
            total_cost += gate.cost

        return GatingDecision(
            activated_gates=activated,
            scores=scores,
            focus=focus,
            budget_remaining=budget - total_cost
        )

    def route(
        self,
        context: str,
        input_data: Any = None,
        budget: float = float('inf')
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        High-level routing: context -> activated gates -> results.

        Args:
            context: Input context
            input_data: Additional input for executors
            budget: Compute budget

        Returns:
            (list of gate names, dict of results)
        """
        focus = self.compute_focus(context)
        decision = self.decide(focus, budget)

        results = {}
        for gate in decision.activated_gates:
            if gate.executor is not None:
                try:
                    results[gate.name] = gate.executor(context, input_data)
                except Exception as e:
                    results[gate.name] = f"ERROR: {e}"

        return [g.name for g in decision.activated_gates], results


# ==============================================================================
# PRE-BUILT GATES FOR COMMON USE CASES
# ==============================================================================

def create_math_gates(network: FocusGatingNetwork):
    """Register common math/reasoning tool gates."""

    network.register_gate(
        name="calculator",
        gate_type=GateType.TOOL,
        keywords=["calculate", "compute", "sum", "product", "add", "multiply", "divide", "subtract", "arithmetic"],
        threshold=0.4,
        cost=0.1,
        metadata={"description": "Basic arithmetic operations"}
    )

    network.register_gate(
        name="number_theory",
        gate_type=GateType.SYMBOLIC,
        keywords=["prime", "factor", "divisor", "modulo", "gcd", "lcm", "coprime", "euler", "fermat"],
        threshold=0.5,
        cost=0.5,
        metadata={"description": "Number theory reasoning"}
    )

    network.register_gate(
        name="symbolic_solver",
        gate_type=GateType.SYMBOLIC,
        keywords=["equation", "solve", "algebra", "polynomial", "root", "quadratic", "variable", "unknown"],
        threshold=0.5,
        cost=1.0,
        metadata={"description": "Symbolic equation solving"}
    )

    network.register_gate(
        name="geometry",
        gate_type=GateType.SYMBOLIC,
        keywords=["triangle", "circle", "angle", "area", "perimeter", "coordinate", "distance", "geometric"],
        threshold=0.5,
        cost=0.5,
        metadata={"description": "Geometric reasoning"}
    )

    network.register_gate(
        name="combinatorics",
        gate_type=GateType.SYMBOLIC,
        keywords=["permutation", "combination", "counting", "ways", "arrange", "choose", "binomial"],
        threshold=0.5,
        cost=0.5,
        metadata={"description": "Combinatorial reasoning"}
    )


def create_search_gates(network: FocusGatingNetwork):
    """Register search/retrieval gates."""

    network.register_gate(
        name="web_search",
        gate_type=GateType.TOOL,
        keywords=["search", "find", "look up", "current", "latest", "news", "today"],
        threshold=0.6,
        cost=2.0,
        metadata={"description": "Web search for current information"}
    )

    network.register_gate(
        name="knowledge_base",
        gate_type=GateType.TOOL,
        keywords=["definition", "what is", "explain", "describe", "fact", "information"],
        threshold=0.4,
        cost=0.5,
        metadata={"description": "Knowledge base lookup"}
    )

    network.register_gate(
        name="code_search",
        gate_type=GateType.TOOL,
        keywords=["code", "function", "class", "implementation", "source", "repository"],
        threshold=0.5,
        cost=1.0,
        metadata={"description": "Code/documentation search"}
    )


def create_reasoning_gates(network: FocusGatingNetwork):
    """Register meta-reasoning strategy gates."""

    network.register_gate(
        name="chain_of_thought",
        gate_type=GateType.PROMPT,
        keywords=["step", "reason", "think", "explain", "show", "work"],
        threshold=0.3,
        cost=0.2,
        priority=1,
        metadata={"description": "Enable chain-of-thought reasoning"}
    )

    network.register_gate(
        name="self_critique",
        gate_type=GateType.PROMPT,
        keywords=["check", "verify", "correct", "error", "mistake", "sure"],
        threshold=0.4,
        cost=0.5,
        metadata={"description": "Self-critique and verification"}
    )

    network.register_gate(
        name="decomposition",
        gate_type=GateType.PROMPT,
        keywords=["complex", "multiple", "parts", "break down", "subproblem", "first"],
        threshold=0.4,
        cost=0.3,
        metadata={"description": "Problem decomposition strategy"}
    )

    network.register_gate(
        name="analogy",
        gate_type=GateType.PROMPT,
        keywords=["similar", "like", "analogy", "example", "case", "instance"],
        threshold=0.4,
        cost=0.2,
        metadata={"description": "Analogical reasoning"}
    )


# ==============================================================================
# INTEGRATION WITH FRACTAL CASCADE
# ==============================================================================

class FractalFocusRouter:
    """
    Integration of focus-gating with Fractal Cascade search.

    Each node in the cascade has an associated focus vector.
    Gate activation determines which edges to explore.
    """

    def __init__(
        self,
        network: FocusGatingNetwork,
        coherence_decay: float = 0.9,
        max_depth: int = 10
    ):
        self.network = network
        self.coherence_decay = coherence_decay
        self.max_depth = max_depth

    def expand_node(
        self,
        state: str,
        depth: int,
        parent_coherence: float = 1.0
    ) -> List[Tuple[Gate, float]]:
        """
        Expand a search node by selecting gates to activate.

        Args:
            state: Current reasoning state
            depth: Current depth in search tree
            parent_coherence: Coherence from parent node

        Returns:
            List of (gate, score) tuples for expansion
        """
        # Decay coherence with depth
        current_coherence = parent_coherence * (self.coherence_decay ** depth)

        # Compute focus from state
        focus = self.network.compute_focus(state)

        # Get gate scores, weighted by coherence
        scores = self.network.gate_scores(focus)

        # Select gates above threshold, weighted by coherence
        expansions = []
        for name, score in scores.items():
            gate = self.network.gates[name]
            effective_score = score * current_coherence

            if effective_score > gate.threshold:
                expansions.append((gate, effective_score))

        # Sort by score
        expansions.sort(key=lambda x: x[1], reverse=True)

        return expansions

    def cascade_search(
        self,
        initial_state: str,
        max_expansions: int = 100
    ) -> List[Dict]:
        """
        Run focus-gated cascade search.

        Args:
            initial_state: Starting problem state
            max_expansions: Maximum total expansions

        Returns:
            List of search traces
        """
        traces = []
        frontier = [(initial_state, 0, 1.0, [])]  # (state, depth, coherence, path)
        expansions = 0

        while frontier and expansions < max_expansions:
            state, depth, coherence, path = frontier.pop(0)

            if depth >= self.max_depth:
                traces.append({
                    'path': path,
                    'final_state': state,
                    'final_coherence': coherence,
                    'depth': depth
                })
                continue

            # Get valid expansions
            candidates = self.expand_node(state, depth, coherence)

            if not candidates:
                traces.append({
                    'path': path,
                    'final_state': state,
                    'final_coherence': coherence,
                    'depth': depth
                })
                continue

            # Expand top candidates
            for gate, score in candidates[:3]:  # Limit branching
                new_path = path + [(gate.name, score)]

                if gate.executor:
                    try:
                        new_state = gate.executor(state, None)
                        frontier.append((new_state, depth + 1, score, new_path))
                        expansions += 1
                    except Exception:
                        pass
                else:
                    # No executor, just log the gate would activate
                    traces.append({
                        'path': new_path,
                        'final_state': state,
                        'final_coherence': score,
                        'depth': depth,
                        'pending_gate': gate.name
                    })

        return traces


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def create_default_router() -> FocusGatingNetwork:
    """Create a focus-gating network with common gates."""
    network = FocusGatingNetwork(embedding_dim=128)

    create_math_gates(network)
    create_search_gates(network)
    create_reasoning_gates(network)

    return network


def create_aimo_router() -> FocusGatingNetwork:
    """Create router optimized for AIMO-style math problems."""
    network = FocusGatingNetwork(embedding_dim=128, default_threshold=0.4)

    create_math_gates(network)
    create_reasoning_gates(network)

    # AIMO-specific gates
    network.register_gate(
        name="modular_arithmetic",
        gate_type=GateType.SYMBOLIC,
        keywords=["mod", "modulo", "remainder", "congruent", "residue"],
        threshold=0.45,
        cost=0.5
    )

    network.register_gate(
        name="functional_equations",
        gate_type=GateType.SYMBOLIC,
        keywords=["f(x)", "function", "satisfies", "for all", "find all functions"],
        threshold=0.5,
        cost=1.0
    )

    network.register_gate(
        name="inequality",
        gate_type=GateType.SYMBOLIC,
        keywords=["inequality", "prove", "show that", "greater", "less", "bound"],
        threshold=0.5,
        cost=0.5
    )

    return network
