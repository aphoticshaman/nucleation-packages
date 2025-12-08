import math
import random
from dataclasses import dataclass, field
from typing import Any, List, Optional, Callable

@dataclass
class Node:
    state: Any
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    action: Optional[Any] = None

def uct_score(parent_visits: int, child: Node, c: float = 1.4) -> float:
    if child.visits == 0:
        return float("inf")
    exploit = child.value / child.visits
    explore = c * math.sqrt(math.log(parent_visits + 1) / child.visits)
    return exploit + explore

def select(node: Node, c: float = 1.4) -> Node:
    current = node
    while current.children:
        current = max(current.children, key=lambda ch: uct_score(current.visits, ch, c))
    return current

def expand(node: Node, actions: List[Any], transition_fn: Callable[[Any, Any], Any]):
    if node.children:
        return
    for a in actions:
        ns = transition_fn(node.state, a)
        child = Node(state=ns, parent=node, action=a)
        node.children.append(child)

def backpropagate(node: Node, reward: float):
    current = node
    while current is not None:
        current.visits += 1
        current.value += reward
        current = current.parent

def run_mcts(root_state: Any,
             actions_fn: Callable[[Any], List[Any]],
             transition_fn: Callable[[Any, Any], Any],
             rollout_fn: Callable[[Any], float],
             n_iters: int = 64,
             c: float = 1.4) -> Node:
    root = Node(state=root_state)
    for _ in range(n_iters):
        leaf = select(root, c)
        actions = actions_fn(leaf.state)
        if actions:
            expand(leaf, actions, transition_fn)
            leaf = random.choice(leaf.children)
        reward = rollout_fn(leaf.state)
        backpropagate(leaf, reward)
    return max(root.children, key=lambda ch: ch.value / (ch.visits or 1)) if root.children else root
