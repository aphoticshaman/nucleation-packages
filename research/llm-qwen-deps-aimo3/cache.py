"""
PROMPTCACHE 1.0
===============

Prefix caching for math problem solving.

AIMO problems share common prefixes:
- "Let's solve this step by step"
- Common math preambles
- Similar problem structures

PromptCache:
1. Hashes prompt prefixes
2. Stores KV cache for reuse
3. Massive latency reduction on similar problems

Target: 50%+ latency reduction on batched math solving.

Author: Ryan J Cardwell (Archer Phoenix)
"""

import torch
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import time
import threading


# =============================================================================
# CACHE ENTRY
# =============================================================================

@dataclass
class CacheEntry:
    """Single cache entry with KV tensors."""
    prefix_hash: str
    prefix_length: int
    prefix_tokens: List[int]
    
    # KV cache: List[Tuple[key, value]] per layer
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    
    # Size tracking
    size_bytes: int = 0
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


# =============================================================================
# HASH FUNCTIONS
# =============================================================================

class PrefixHasher:
    """
    Hashes token prefixes for cache lookup.
    
    Uses rolling hash for efficient prefix matching.
    """
    
    def __init__(self, chunk_size: int = 64):
        self.chunk_size = chunk_size
    
    def hash_tokens(self, tokens: List[int]) -> str:
        """Hash a token sequence."""
        # Use SHA256 for collision resistance
        token_bytes = bytes(str(tokens), 'utf-8')
        return hashlib.sha256(token_bytes).hexdigest()[:16]
    
    def hash_prefix(self, tokens: List[int], length: int) -> str:
        """Hash a prefix of specific length."""
        return self.hash_tokens(tokens[:length])
    
    def find_longest_cached_prefix(
        self,
        tokens: List[int],
        cache_hashes: Dict[str, int],
    ) -> Tuple[Optional[str], int]:
        """
        Find longest prefix that exists in cache.
        
        Args:
            tokens: Full token sequence
            cache_hashes: Dict of hash -> prefix_length
        
        Returns:
            (hash, length) of longest cached prefix, or (None, 0)
        """
        # Binary search for longest match
        best_hash = None
        best_length = 0
        
        # Check at chunk boundaries
        for length in range(self.chunk_size, len(tokens) + 1, self.chunk_size):
            h = self.hash_prefix(tokens, length)
            if h in cache_hashes:
                best_hash = h
                best_length = length
        
        # Also check full length if not at boundary
        if len(tokens) % self.chunk_size != 0:
            h = self.hash_tokens(tokens)
            if h in cache_hashes:
                best_hash = h
                best_length = len(tokens)
        
        return best_hash, best_length


# =============================================================================
# MAIN CACHE
# =============================================================================

class PromptCache:
    """
    Main prompt cache for KV reuse.
    
    Usage:
        cache = PromptCache(max_size_gb=4)
        
        # Check for cached prefix
        entry = cache.get(prompt_tokens)
        if entry:
            # Use cached KV, only compute new tokens
            kv_cache = entry.kv_cache
            start_pos = entry.prefix_length
        else:
            # Compute full sequence
            kv_cache = None
            start_pos = 0
        
        # After generation, cache the result
        cache.put(prompt_tokens, kv_cache)
    """
    
    def __init__(
        self,
        max_size_gb: float = 4.0,
        max_entries: int = 1000,
        min_prefix_length: int = 64,
        eviction_policy: str = 'lru',  # 'lru', 'lfu', 'fifo'
    ):
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.max_entries = max_entries
        self.min_prefix_length = min_prefix_length
        self.eviction_policy = eviction_policy
        
        # Cache storage
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hash_to_length: Dict[str, int] = {}
        
        # Hasher
        self.hasher = PrefixHasher()
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.current_size = 0
        
        # Thread safety
        self.lock = threading.RLock()
    
    def get(self, tokens: List[int]) -> Optional[CacheEntry]:
        """
        Look up cached KV for token prefix.
        
        Returns CacheEntry if found, None otherwise.
        """
        with self.lock:
            # Find longest cached prefix
            prefix_hash, prefix_length = self.hasher.find_longest_cached_prefix(
                tokens, self.hash_to_length
            )
            
            if prefix_hash is None or prefix_length < self.min_prefix_length:
                self.misses += 1
                return None
            
            if prefix_hash not in self.entries:
                self.misses += 1
                return None
            
            entry = self.entries[prefix_hash]
            
            # Verify tokens match (hash collision check)
            if tokens[:prefix_length] != entry.prefix_tokens:
                self.misses += 1
                return None
            
            # Update access
            entry.update_access()
            
            # Move to end for LRU
            if self.eviction_policy == 'lru':
                self.entries.move_to_end(prefix_hash)
            
            self.hits += 1
            return entry
    
    def put(
        self,
        tokens: List[int],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        prefix_length: Optional[int] = None,
    ) -> bool:
        """
        Store KV cache for token prefix.
        
        Args:
            tokens: Full token sequence
            kv_cache: KV tensors per layer
            prefix_length: Length of prefix to cache (default: full sequence)
        
        Returns:
            True if cached successfully
        """
        if prefix_length is None:
            prefix_length = len(tokens)
        
        if prefix_length < self.min_prefix_length:
            return False
        
        with self.lock:
            # Compute hash
            prefix_tokens = tokens[:prefix_length]
            prefix_hash = self.hasher.hash_tokens(prefix_tokens)
            
            # Already cached?
            if prefix_hash in self.entries:
                return True
            
            # Compute size
            size_bytes = sum(
                k.numel() * k.element_size() + v.numel() * v.element_size()
                for k, v in kv_cache
            )
            
            # Evict if necessary
            while (
                self.current_size + size_bytes > self.max_size_bytes or
                len(self.entries) >= self.max_entries
            ):
                if not self._evict_one():
                    return False  # Can't evict anything
            
            # Clone KV tensors (they might be views)
            kv_clone = [(k.clone(), v.clone()) for k, v in kv_cache]
            
            # Create entry
            entry = CacheEntry(
                prefix_hash=prefix_hash,
                prefix_length=prefix_length,
                prefix_tokens=prefix_tokens,
                kv_cache=kv_clone,
                size_bytes=size_bytes,
            )
            
            # Store
            self.entries[prefix_hash] = entry
            self.hash_to_length[prefix_hash] = prefix_length
            self.current_size += size_bytes
            
            return True
    
    def _evict_one(self) -> bool:
        """Evict one entry based on policy."""
        if not self.entries:
            return False
        
        if self.eviction_policy == 'lru':
            # Remove oldest (first in OrderedDict)
            key = next(iter(self.entries))
        
        elif self.eviction_policy == 'lfu':
            # Remove least frequently used
            key = min(self.entries.keys(), key=lambda k: self.entries[k].access_count)
        
        else:  # fifo
            key = next(iter(self.entries))
        
        entry = self.entries.pop(key)
        del self.hash_to_length[key]
        self.current_size -= entry.size_bytes
        self.evictions += 1
        
        # Free memory
        for k, v in entry.kv_cache:
            del k, v
        
        return True
    
    def clear(self):
        """Clear entire cache."""
        with self.lock:
            self.entries.clear()
            self.hash_to_length.clear()
            self.current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(1, total_requests)
            
            return {
                'entries': len(self.entries),
                'size_mb': self.current_size / 1024 / 1024,
                'max_size_mb': self.max_size_bytes / 1024 / 1024,
                'utilization': self.current_size / self.max_size_bytes,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions,
            }


# =============================================================================
# COMMON PREFIX DETECTOR
# =============================================================================

class CommonPrefixDetector:
    """
    Detects common prefixes across problems for proactive caching.
    
    Analyzes problem batches to find shared prefixes.
    """
    
    def __init__(self, min_frequency: int = 2, min_length: int = 32):
        self.min_frequency = min_frequency
        self.min_length = min_length
        self.prefix_counts: Dict[Tuple[int, ...], int] = {}
    
    def add_sequence(self, tokens: List[int]):
        """Add a sequence for prefix analysis."""
        # Count all prefixes
        for length in range(self.min_length, len(tokens) + 1, 16):
            prefix = tuple(tokens[:length])
            self.prefix_counts[prefix] = self.prefix_counts.get(prefix, 0) + 1
    
    def get_common_prefixes(self) -> List[Tuple[List[int], int]]:
        """Get prefixes that appear frequently."""
        common = [
            (list(prefix), count)
            for prefix, count in self.prefix_counts.items()
            if count >= self.min_frequency
        ]
        # Sort by length * frequency (longer and more frequent = better)
        common.sort(key=lambda x: len(x[0]) * x[1], reverse=True)
        return common
    
    def suggest_cache_prefixes(self, top_k: int = 10) -> List[List[int]]:
        """Suggest prefixes to proactively cache."""
        common = self.get_common_prefixes()
        
        # Filter out redundant (shorter prefixes of longer ones)
        result = []
        for prefix, count in common[:top_k * 2]:
            is_redundant = any(
                len(existing) > len(prefix) and
                existing[:len(prefix)] == prefix
                for existing in result
            )
            if not is_redundant:
                result.append(prefix)
            if len(result) >= top_k:
                break
        
        return result


# =============================================================================
# MATH PROMPT TEMPLATES
# =============================================================================

class MathPromptTemplates:
    """
    Pre-defined math prompt templates for caching.
    
    Common problem-solving prefixes that benefit from caching.
    """
    
    STEP_BY_STEP = "Let's solve this step by step.\n\n"
    
    PROOF_START = "Proof:\n"
    
    INDUCTION_BASE = "Base case: Let n = 1.\n"
    
    INDUCTION_STEP = "Inductive step: Assume P(k) holds for some k ≥ 1.\n"
    
    CONTRADICTION = "Proof by contradiction:\nAssume the opposite.\n"
    
    NUMBER_THEORY_SETUP = "Let's analyze the divisibility properties.\n"
    
    COMBINATORICS_SETUP = "Let's count the number of ways.\n"
    
    ALGEBRA_SETUP = "Let's define our variables and set up equations.\n"
    
    GEOMETRY_SETUP = "Consider the geometric configuration.\n"
    
    @classmethod
    def get_all_templates(cls) -> List[str]:
        """Get all template strings."""
        return [
            cls.STEP_BY_STEP,
            cls.PROOF_START,
            cls.INDUCTION_BASE,
            cls.INDUCTION_STEP,
            cls.CONTRADICTION,
            cls.NUMBER_THEORY_SETUP,
            cls.COMBINATORICS_SETUP,
            cls.ALGEBRA_SETUP,
            cls.GEOMETRY_SETUP,
        ]
    
    @classmethod
    def tokenize_templates(cls, tokenizer: Any) -> Dict[str, List[int]]:
        """Tokenize all templates."""
        result = {}
        for name in dir(cls):
            if name.isupper() and not name.startswith('_'):
                text = getattr(cls, name)
                if isinstance(text, str):
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    result[name] = tokens
        return result


# =============================================================================
# CACHE-AWARE GENERATION
# =============================================================================

class CachedGenerator:
    """
    Generation wrapper with automatic cache usage.
    
    Usage:
        generator = CachedGenerator(model, tokenizer, cache)
        output = generator.generate(prompt, max_new_tokens=256)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        cache: PromptCache,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate with cache lookup."""
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        tokens = input_ids[0].tolist()
        
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Check cache
        cache_entry = self.cache.get(tokens)
        
        if cache_entry is not None:
            # Cache hit! Use cached KV
            print(f"[PromptCache] Hit! Reusing {cache_entry.prefix_length} tokens")
            
            kv_cache = cache_entry.kv_cache
            start_pos = cache_entry.prefix_length
            
            # Only process new tokens
            new_input_ids = input_ids[:, start_pos:]
            
            # Generate with cached KV
            output_ids = self._generate_with_cache(
                new_input_ids,
                kv_cache,
                max_new_tokens,
                temperature,
            )
            
            # Prepend cached prefix
            full_output = torch.cat([input_ids[:, :start_pos], output_ids], dim=1)
        else:
            # Cache miss - full generation
            print(f"[PromptCache] Miss - full generation")
            
            full_output, kv_cache = self._generate_full(
                input_ids,
                max_new_tokens,
                temperature,
            )
            
            # Cache the prompt KV
            self.cache.put(tokens, kv_cache, prefix_length=len(tokens))
        
        # Decode
        output_text = self.tokenizer.decode(full_output[0], skip_special_tokens=True)
        return output_text
    
    def _generate_with_cache(
        self,
        input_ids: torch.Tensor,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        max_new_tokens: int,
        temperature: float,
    ) -> torch.Tensor:
        """Generate using cached KV as starting point."""
        # Move cache to device
        device = input_ids.device
        kv_cache = [(k.to(device), v.to(device)) for k, v in kv_cache]
        
        current_ids = input_ids
        
        for _ in range(max_new_tokens):
            # Forward with KV cache
            outputs = self.model(
                current_ids,
                past_key_values=kv_cache,
                use_cache=True,
            )
            
            logits = outputs.logits[:, -1, :]
            kv_cache = outputs.past_key_values
            
            # Sample
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            current_ids = next_token
            
            # Check EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Return all generated tokens
        return torch.cat([input_ids, current_ids], dim=1)
    
    def _generate_full(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Full generation from scratch."""
        current_ids = input_ids
        kv_cache = None
        
        for _ in range(max_new_tokens):
            outputs = self.model(
                current_ids,
                past_key_values=kv_cache,
                use_cache=True,
            )
            
            logits = outputs.logits[:, -1, :]
            kv_cache = outputs.past_key_values
            
            # Sample
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Extract just the prompt's KV (for caching)
        prompt_length = input_ids.shape[1]
        prompt_kv = []
        for k, v in kv_cache:
            # k, v shape: [batch, heads, seq, dim]
            prompt_kv.append((
                k[:, :, :prompt_length, :].cpu(),
                v[:, :, :prompt_length, :].cpu(),
            ))
        
        return current_ids, prompt_kv
    
    def warmup_templates(self):
        """Pre-cache common templates."""
        templates = MathPromptTemplates.tokenize_templates(self.tokenizer)
        
        for name, tokens in templates.items():
            # Run forward pass to get KV
            input_ids = torch.tensor([tokens]).to(next(self.model.parameters()).device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, use_cache=True)
                kv_cache = outputs.past_key_values
            
            # Cache it
            prompt_kv = [(k.cpu(), v.cpu()) for k, v in kv_cache]
            self.cache.put(tokens, prompt_kv)
            
            print(f"[PromptCache] Warmed up: {name} ({len(tokens)} tokens)")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main cache
    'PromptCache',
    'CacheEntry',
    
    # Utilities
    'PrefixHasher',
    'CommonPrefixDetector',
    'MathPromptTemplates',
    
    # Generation
    'CachedGenerator',
]


if __name__ == "__main__":
    print("PromptCache 1.0")
    print("===============")
    print()
    print("Prefix caching for math problems.")
    print()
    print("Features:")
    print("  - Hash-based prefix matching")
    print("  - LRU/LFU/FIFO eviction")
    print("  - Common prefix detection")
    print("  - Pre-defined math templates")
    print("  - Cache-aware generation")
    print()
    print("Usage:")
    print("  cache = PromptCache(max_size_gb=4)")
    print("  generator = CachedGenerator(model, tokenizer, cache)")
    print("  generator.warmup_templates()  # Pre-cache common prefixes")
    print("  output = generator.generate(prompt)")
