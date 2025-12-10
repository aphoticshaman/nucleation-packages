"""
RYANSTREAM SPECULATIVE DECODING
===============================

Extensions for speculative decoding and tree attention.

Features:
- Draft model speculation (small model proposes, big model verifies)
- Tree attention for parallel hypothesis exploration
- Self-speculation (same model, different layers)
- Adaptive speculation depth (backs off when rejection rate high)

Target: 2-4x speedup on math proofs (lots of deterministic tokens).

Author: Ryan J Cardwell (Archer Phoenix)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import math
import time


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SpeculativeCandidate:
    """A speculative token sequence candidate."""
    tokens: List[int]
    log_probs: List[float]
    cumulative_log_prob: float = 0.0
    accepted: int = 0  # How many verified
    
    def __post_init__(self):
        if self.log_probs:
            self.cumulative_log_prob = sum(self.log_probs)


@dataclass 
class TreeNode:
    """Node in speculation tree."""
    token: int
    log_prob: float
    depth: int
    children: List['TreeNode'] = field(default_factory=list)
    parent: Optional['TreeNode'] = None
    verified: bool = False
    
    def get_path(self) -> List[int]:
        """Get token path from root to this node."""
        path = []
        node = self
        while node is not None:
            path.append(node.token)
            node = node.parent
        return list(reversed(path))


@dataclass
class SpeculationStats:
    """Track speculation performance."""
    total_drafted: int = 0
    total_accepted: int = 0
    total_verifications: int = 0
    rejection_streak: int = 0
    
    @property
    def acceptance_rate(self) -> float:
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted
    
    @property
    def speedup_estimate(self) -> float:
        """Estimate speedup from speculation."""
        if self.total_verifications == 0:
            return 1.0
        avg_accepted = self.total_accepted / self.total_verifications
        # Speedup ≈ (1 + avg_accepted) / 1 since we verify in batches
        return 1 + avg_accepted


# =============================================================================
# DRAFT MODEL SPECULATION
# =============================================================================

class DraftModelSpeculator:
    """
    Speculative decoding with draft model.
    
    Small draft model (e.g., 1B) proposes K tokens.
    Large target model verifies in one forward pass.
    Accept prefix that matches, resample from target where diverged.
    
    Math proofs have high acceptance (deterministic steps).
    """
    
    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        tokenizer: Any,
        max_speculation_length: int = 8,
        temperature: float = 0.0,
        adaptive: bool = True,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.max_speculation_length = max_speculation_length
        self.temperature = temperature
        self.adaptive = adaptive
        
        # Adaptive speculation depth
        self.current_depth = max_speculation_length
        self.min_depth = 2
        self.stats = SpeculationStats()
    
    @torch.no_grad()
    def speculate(self, input_ids: torch.Tensor) -> Tuple[List[int], int]:
        """
        Generate tokens speculatively.
        
        Args:
            input_ids: [batch=1, seq_len] current context
        
        Returns:
            (accepted_tokens, num_target_calls)
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        assert batch_size == 1, "Speculation requires batch size 1"
        
        # Phase 1: Draft model generates K candidates
        draft_tokens = []
        draft_probs = []
        
        current_ids = input_ids.clone()
        for _ in range(self.current_depth):
            outputs = self.draft_model(current_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            next_logits = logits[0, -1, :]
            
            if self.temperature > 0:
                probs = F.softmax(next_logits / self.temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = next_logits.argmax().item()
            
            draft_tokens.append(next_token)
            draft_probs.append(F.softmax(next_logits, dim=-1)[next_token].item())
            
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[next_token]], device=device)
            ], dim=1)
        
        self.stats.total_drafted += len(draft_tokens)
        
        # Phase 2: Target model verifies all K+1 positions in one pass
        # Input: original + draft tokens
        verify_ids = torch.cat([
            input_ids,
            torch.tensor([draft_tokens], device=device)
        ], dim=1)
        
        outputs = self.target_model(verify_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        self.stats.total_verifications += 1
        
        # Phase 3: Accept/reject
        accepted_tokens = []
        
        for i, (draft_token, draft_prob) in enumerate(zip(draft_tokens, draft_probs)):
            # Target's distribution at this position
            target_logits = logits[0, input_ids.shape[1] + i - 1, :]
            target_probs = F.softmax(target_logits, dim=-1)
            target_prob = target_probs[draft_token].item()
            
            # Acceptance criterion: accept if target agrees
            if self.temperature == 0:
                # Greedy: accept if same as argmax
                if target_logits.argmax().item() == draft_token:
                    accepted_tokens.append(draft_token)
                else:
                    # Resample from target
                    accepted_tokens.append(target_logits.argmax().item())
                    break
            else:
                # Stochastic: rejection sampling
                # Accept with prob min(1, target_prob / draft_prob)
                accept_prob = min(1.0, target_prob / (draft_prob + 1e-10))
                if torch.rand(1).item() < accept_prob:
                    accepted_tokens.append(draft_token)
                else:
                    # Resample from adjusted distribution
                    adjusted = F.relu(target_probs - draft_probs[i] * torch.ones_like(target_probs))
                    adjusted = adjusted / (adjusted.sum() + 1e-10)
                    resampled = torch.multinomial(adjusted, 1).item()
                    accepted_tokens.append(resampled)
                    break
        
        self.stats.total_accepted += len(accepted_tokens)
        
        # Adaptive depth adjustment
        if self.adaptive:
            self._adjust_depth(len(accepted_tokens))
        
        return accepted_tokens, 1  # 1 target call
    
    def _adjust_depth(self, num_accepted: int):
        """Adjust speculation depth based on acceptance."""
        acceptance_rate = num_accepted / self.current_depth
        
        if acceptance_rate > 0.8:
            # High acceptance → try deeper
            self.current_depth = min(self.max_speculation_length, self.current_depth + 1)
            self.stats.rejection_streak = 0
        elif acceptance_rate < 0.3:
            # Low acceptance → back off
            self.stats.rejection_streak += 1
            if self.stats.rejection_streak >= 3:
                self.current_depth = max(self.min_depth, self.current_depth - 1)
                self.stats.rejection_streak = 0


# =============================================================================
# SELF-SPECULATION
# =============================================================================

class SelfSpeculator:
    """
    Self-speculation using early exit.
    
    Use early layers to draft, full model to verify.
    No separate draft model needed!
    
    Works with models that have intermediate outputs.
    """
    
    def __init__(
        self,
        model: nn.Module,
        draft_exit_layer: int = 8,  # Exit after this layer for drafting
        max_speculation_length: int = 6,
        temperature: float = 0.0,
    ):
        self.model = model
        self.draft_exit_layer = draft_exit_layer
        self.max_speculation_length = max_speculation_length
        self.temperature = temperature
        
        # Build early exit head if needed
        self.exit_head = self._build_exit_head()
        self.stats = SpeculationStats()
    
    def _build_exit_head(self) -> Optional[nn.Module]:
        """Build classification head for early exit."""
        # Try to find model dimensions
        if hasattr(self.model, 'config'):
            hidden_size = getattr(self.model.config, 'hidden_size', 4096)
            vocab_size = getattr(self.model.config, 'vocab_size', 32000)
        else:
            hidden_size = 4096
            vocab_size = 32000
        
        # Simple linear head
        return nn.Linear(hidden_size, vocab_size).to(
            next(self.model.parameters()).device
        )
    
    def _get_early_exit_logits(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Get logits from early layers."""
        # This requires model modification to expose intermediate states
        # Here's a generic approach for Llama-style models
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Llama-style
            hidden = self.model.model.embed_tokens(input_ids)
            
            for i, layer in enumerate(self.model.model.layers[:self.draft_exit_layer]):
                hidden = layer(hidden)[0]
            
            if self.exit_head is not None:
                return self.exit_head(hidden)
        
        # Fallback: just use full model
        outputs = self.model(input_ids)
        return outputs.logits if hasattr(outputs, 'logits') else outputs
    
    @torch.no_grad()
    def speculate(self, input_ids: torch.Tensor) -> Tuple[List[int], int]:
        """Generate tokens using self-speculation."""
        device = input_ids.device
        
        # Draft with early exit
        draft_tokens = []
        current_ids = input_ids.clone()
        
        for _ in range(self.max_speculation_length):
            logits = self._get_early_exit_logits(current_ids)
            next_logits = logits[0, -1, :]
            
            if self.temperature > 0:
                probs = F.softmax(next_logits / self.temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = next_logits.argmax().item()
            
            draft_tokens.append(next_token)
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[next_token]], device=device)
            ], dim=1)
        
        self.stats.total_drafted += len(draft_tokens)
        
        # Verify with full model
        verify_ids = torch.cat([
            input_ids,
            torch.tensor([draft_tokens], device=device)
        ], dim=1)
        
        outputs = self.model(verify_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        self.stats.total_verifications += 1
        
        # Accept matching prefix
        accepted = []
        for i, draft_token in enumerate(draft_tokens):
            target_token = logits[0, input_ids.shape[1] + i - 1, :].argmax().item()
            if target_token == draft_token:
                accepted.append(draft_token)
            else:
                accepted.append(target_token)
                break
        
        self.stats.total_accepted += len(accepted)
        return accepted, 1


# =============================================================================
# TREE ATTENTION
# =============================================================================

class TreeAttention:
    """
    Tree attention for parallel hypothesis exploration.
    
    Instead of generating one sequence, explore a tree of possibilities.
    Prune unlikely branches, expand promising ones.
    
    Great for math where multiple solution paths exist.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_depth: int = 8,
        beam_width: int = 4,
        prune_threshold: float = 0.01,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.prune_threshold = prune_threshold
    
    @torch.no_grad()
    def generate_tree(
        self,
        input_ids: torch.Tensor,
        num_tokens: int = 50,
    ) -> List[List[int]]:
        """
        Generate a tree of token sequences.
        
        Returns top paths through the tree.
        """
        device = input_ids.device
        
        # Initialize root
        root = TreeNode(token=-1, log_prob=0.0, depth=0)
        leaves = [root]
        
        for depth in range(min(num_tokens, self.max_depth)):
            new_leaves = []
            
            for leaf in leaves:
                # Get path to this leaf
                path = leaf.get_path()[1:]  # Skip dummy root token
                
                # Build input
                if path:
                    current_ids = torch.cat([
                        input_ids,
                        torch.tensor([path], device=device)
                    ], dim=1)
                else:
                    current_ids = input_ids
                
                # Get next token distribution
                outputs = self.model(current_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                next_logits = logits[0, -1, :]
                probs = F.softmax(next_logits, dim=-1)
                log_probs = F.log_softmax(next_logits, dim=-1)
                
                # Get top-k candidates
                top_probs, top_tokens = probs.topk(self.beam_width)
                
                for prob, token in zip(top_probs, top_tokens):
                    if prob.item() < self.prune_threshold:
                        continue
                    
                    child = TreeNode(
                        token=token.item(),
                        log_prob=log_probs[token].item(),
                        depth=depth + 1,
                        parent=leaf,
                    )
                    leaf.children.append(child)
                    new_leaves.append(child)
            
            # Prune to beam width
            if len(new_leaves) > self.beam_width:
                # Keep highest cumulative probability
                new_leaves.sort(
                    key=lambda n: self._cumulative_log_prob(n),
                    reverse=True
                )
                new_leaves = new_leaves[:self.beam_width]
            
            leaves = new_leaves
            
            if not leaves:
                break
        
        # Extract best paths
        paths = []
        for leaf in leaves:
            path = leaf.get_path()[1:]  # Skip dummy root
            if path:
                paths.append(path)
        
        # Sort by cumulative probability
        paths.sort(key=lambda p: self._path_score(input_ids, p, device), reverse=True)
        
        return paths
    
    def _cumulative_log_prob(self, node: TreeNode) -> float:
        """Get cumulative log probability from root to node."""
        total = 0.0
        current = node
        while current.parent is not None:
            total += current.log_prob
            current = current.parent
        return total
    
    def _path_score(
        self,
        prefix: torch.Tensor,
        path: List[int],
        device: str,
    ) -> float:
        """Score a complete path."""
        if not path:
            return float('-inf')
        
        # Simple: sum of log probs
        full_ids = torch.cat([
            prefix,
            torch.tensor([path], device=device)
        ], dim=1)
        
        outputs = self.model(full_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        score = 0.0
        for i, token in enumerate(path):
            log_probs = F.log_softmax(logits[0, prefix.shape[1] + i - 1, :], dim=-1)
            score += log_probs[token].item()
        
        return score


# =============================================================================
# INTEGRATED SPECULATIVE ENGINE
# =============================================================================

class SpeculativeEngine:
    """
    Unified speculative generation engine.
    
    Combines:
    - Draft model speculation
    - Self-speculation
    - Tree exploration
    
    Automatically selects best method based on context.
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        tokenizer: Any,
        draft_model: Optional[nn.Module] = None,
        mode: str = 'auto',  # 'draft', 'self', 'tree', 'auto'
        max_speculation_length: int = 8,
    ):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.mode = mode
        
        # Initialize speculators
        self.draft_speculator = None
        if draft_model is not None:
            self.draft_speculator = DraftModelSpeculator(
                draft_model=draft_model,
                target_model=target_model,
                tokenizer=tokenizer,
                max_speculation_length=max_speculation_length,
            )
        
        self.self_speculator = SelfSpeculator(
            model=target_model,
            max_speculation_length=max_speculation_length,
        )
        
        self.tree_attention = TreeAttention(
            model=target_model,
            tokenizer=tokenizer,
        )
        
        # Performance tracking
        self.tokens_generated = 0
        self.target_calls = 0
        self.start_time = time.time()
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        use_tree: bool = False,
    ) -> torch.Tensor:
        """
        Generate tokens speculatively.
        
        Returns full sequence including input.
        """
        device = input_ids.device
        current_ids = input_ids.clone()
        
        while current_ids.shape[1] - input_ids.shape[1] < max_new_tokens:
            if use_tree:
                # Tree exploration
                paths = self.tree_attention.generate_tree(current_ids, num_tokens=8)
                if paths:
                    best_path = paths[0]
                    current_ids = torch.cat([
                        current_ids,
                        torch.tensor([best_path], device=device)
                    ], dim=1)
                    self.tokens_generated += len(best_path)
                    self.target_calls += len(best_path)  # Tree does individual calls
                else:
                    break
            else:
                # Speculation
                speculator = self._select_speculator()
                accepted, calls = speculator.speculate(current_ids)
                
                if not accepted:
                    break
                
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor([accepted], device=device)
                ], dim=1)
                
                self.tokens_generated += len(accepted)
                self.target_calls += calls
                
                # Check for EOS
                if self.tokenizer.eos_token_id in accepted:
                    break
        
        return current_ids
    
    def _select_speculator(self):
        """Select best speculator based on mode and performance."""
        if self.mode == 'draft' and self.draft_speculator:
            return self.draft_speculator
        elif self.mode == 'self':
            return self.self_speculator
        elif self.mode == 'auto':
            # Pick based on recent performance
            if self.draft_speculator:
                draft_rate = self.draft_speculator.stats.acceptance_rate
                self_rate = self.self_speculator.stats.acceptance_rate
                
                if draft_rate > self_rate:
                    return self.draft_speculator
            return self.self_speculator
        else:
            return self.self_speculator
    
    def get_stats(self) -> Dict[str, float]:
        """Get generation statistics."""
        elapsed = time.time() - self.start_time
        
        return {
            'tokens_generated': self.tokens_generated,
            'target_calls': self.target_calls,
            'tokens_per_call': self.tokens_generated / max(1, self.target_calls),
            'tokens_per_second': self.tokens_generated / max(0.001, elapsed),
            'speedup_vs_autoregressive': self.tokens_generated / max(1, self.target_calls),
            'draft_acceptance_rate': self.draft_speculator.stats.acceptance_rate if self.draft_speculator else 0,
            'self_acceptance_rate': self.self_speculator.stats.acceptance_rate,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SpeculativeCandidate',
    'TreeNode',
    'SpeculationStats',
    'DraftModelSpeculator',
    'SelfSpeculator',
    'TreeAttention',
    'SpeculativeEngine',
]


if __name__ == "__main__":
    print("RyanStream Speculative Decoding")
    print("================================")
    print("Components:")
    print("  - DraftModelSpeculator: Small model drafts, big model verifies")
    print("  - SelfSpeculator: Early exit drafting, no extra model needed")
    print("  - TreeAttention: Parallel hypothesis exploration")
    print("  - SpeculativeEngine: Unified interface with auto-selection")
