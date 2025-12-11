"""
SMARTCHECKPOINT 1.0
===================

Selective gradient checkpointing.

Standard checkpointing is all-or-nothing.
SmartCheckpoint profiles layers and checkpoints only where it matters.

Features:
1. Profile memory per layer
2. Checkpoint high-memory layers only
3. Dynamic based on batch size
4. Recompute cost estimation
5. Optimal checkpoint placement

Target: Better memory/compute tradeoff than blanket checkpointing.

Author: Ryan J Cardwell (Archer Phoenix)
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from typing import Dict, List, Optional, Tuple, Set, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time
import math


# =============================================================================
# MEMORY PROFILER
# =============================================================================

@dataclass
class LayerProfile:
    """Memory and compute profile for a layer."""
    name: str
    
    # Memory (bytes)
    param_memory: int = 0
    activation_memory: int = 0
    gradient_memory: int = 0
    peak_memory: int = 0
    
    # Compute
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    flops: int = 0
    
    # Derived
    memory_score: float = 0.0  # Higher = more benefit from checkpointing
    compute_score: float = 0.0  # Higher = more cost to recompute
    
    @property
    def checkpoint_benefit(self) -> float:
        """Net benefit of checkpointing this layer."""
        # Benefit = memory saved - compute cost
        # We want high memory savings and low recompute cost
        if self.compute_score == 0:
            return self.memory_score
        return self.memory_score / (1 + self.compute_score)


class MemoryProfiler:
    """
    Profiles memory usage per layer.
    
    Runs sample forward/backward to measure actual memory.
    """
    
    def __init__(self):
        self.profiles: Dict[str, LayerProfile] = {}
        self.hooks = []
    
    def profile(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        sample_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, LayerProfile]:
        """Profile all layers in the model."""
        self.profiles.clear()
        
        # Register hooks
        for name, module in model.named_modules():
            if self._should_profile(module):
                self._register_hooks(name, module)
        
        # Run forward
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
        
        with torch.enable_grad():
            output = model(sample_input)
            
            # Run backward if we have a target
            if sample_target is not None:
                loss = nn.functional.cross_entropy(
                    output.view(-1, output.size(-1)),
                    sample_target.view(-1),
                )
                loss.backward()
        
        end_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()
        
        # Clean up hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        # Calculate scores
        self._calculate_scores()
        
        return self.profiles
    
    def _should_profile(self, module: nn.Module) -> bool:
        """Decide if a module should be profiled."""
        # Profile attention, MLP, and layer norms
        name = module.__class__.__name__.lower()
        return any(k in name for k in ['attention', 'mlp', 'linear', 'conv', 'norm'])
    
    def _register_hooks(self, name: str, module: nn.Module):
        """Register forward and backward hooks."""
        profile = LayerProfile(name=name)
        
        # Parameter memory
        profile.param_memory = sum(
            p.numel() * p.element_size() for p in module.parameters()
        )
        
        def forward_hook(mod, inp, out):
            # Measure activation memory
            if isinstance(out, torch.Tensor):
                profile.activation_memory = out.numel() * out.element_size()
            elif isinstance(out, tuple):
                profile.activation_memory = sum(
                    o.numel() * o.element_size() for o in out if isinstance(o, torch.Tensor)
                )
        
        def backward_hook(mod, grad_in, grad_out):
            # Measure gradient memory
            total = 0
            for g in grad_out:
                if isinstance(g, torch.Tensor):
                    total += g.numel() * g.element_size()
            profile.gradient_memory = total
        
        fh = module.register_forward_hook(forward_hook)
        bh = module.register_full_backward_hook(backward_hook)
        
        self.hooks.extend([fh, bh])
        self.profiles[name] = profile
    
    def _calculate_scores(self):
        """Calculate memory and compute scores."""
        if not self.profiles:
            return
        
        # Normalize scores
        max_mem = max(p.activation_memory for p in self.profiles.values()) or 1
        max_param = max(p.param_memory for p in self.profiles.values()) or 1
        
        for profile in self.profiles.values():
            # Memory score: how much we save by checkpointing
            profile.memory_score = profile.activation_memory / max_mem
            
            # Compute score: how expensive to recompute (proxy: param count)
            profile.compute_score = profile.param_memory / max_param


# =============================================================================
# CHECKPOINT POLICY
# =============================================================================

class CheckpointPolicy:
    """
    Determines which layers to checkpoint.
    """
    
    def __init__(
        self,
        mode: str = 'auto',  # 'auto', 'aggressive', 'conservative', 'none'
        memory_threshold: float = 0.7,  # Checkpoint if activation > 70% of max
        benefit_threshold: float = 0.5,  # Checkpoint if benefit > 0.5
        max_checkpoints: Optional[int] = None,
    ):
        self.mode = mode
        self.memory_threshold = memory_threshold
        self.benefit_threshold = benefit_threshold
        self.max_checkpoints = max_checkpoints
    
    def select_layers(
        self,
        profiles: Dict[str, LayerProfile],
    ) -> Set[str]:
        """Select layers to checkpoint."""
        if self.mode == 'none':
            return set()
        
        if self.mode == 'aggressive':
            # Checkpoint everything
            return set(profiles.keys())
        
        # Sort by benefit
        sorted_profiles = sorted(
            profiles.values(),
            key=lambda p: p.checkpoint_benefit,
            reverse=True,
        )
        
        selected = set()
        
        for profile in sorted_profiles:
            if self.mode == 'conservative':
                # Only checkpoint very high memory layers
                if profile.memory_score >= 0.9:
                    selected.add(profile.name)
            else:  # auto
                # Checkpoint based on benefit threshold
                if profile.checkpoint_benefit >= self.benefit_threshold:
                    selected.add(profile.name)
                elif profile.memory_score >= self.memory_threshold:
                    selected.add(profile.name)
            
            # Respect max checkpoints
            if self.max_checkpoints and len(selected) >= self.max_checkpoints:
                break
        
        return selected


# =============================================================================
# CHECKPOINTED WRAPPER
# =============================================================================

class CheckpointedModule(nn.Module):
    """
    Wrapper that applies gradient checkpointing to a module.
    """
    
    def __init__(self, module: nn.Module, use_reentrant: bool = False):
        super().__init__()
        self.module = module
        self.use_reentrant = use_reentrant
    
    def forward(self, *args, **kwargs):
        # Convert kwargs to args for checkpoint compatibility
        if kwargs:
            # Checkpoint doesn't handle kwargs well
            # We need to be careful here
            return checkpoint(
                self._forward_with_kwargs,
                *args,
                kwargs,
                use_reentrant=self.use_reentrant,
            )
        else:
            return checkpoint(
                self.module,
                *args,
                use_reentrant=self.use_reentrant,
            )
    
    def _forward_with_kwargs(self, *args):
        # Last arg is kwargs dict
        kwargs = args[-1]
        args = args[:-1]
        return self.module(*args, **kwargs)


class CheckpointedSequential(nn.Module):
    """
    Sequential module with checkpointing at specified layers.
    """
    
    def __init__(
        self,
        modules: List[nn.Module],
        checkpoint_indices: Set[int],
        use_reentrant: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(modules)
        self.checkpoint_indices = checkpoint_indices
        self.use_reentrant = use_reentrant
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if i in self.checkpoint_indices:
                x = checkpoint(layer, x, use_reentrant=self.use_reentrant)
            else:
                x = layer(x)
        return x


# =============================================================================
# MAIN SMART CHECKPOINT
# =============================================================================

class SmartCheckpoint:
    """
    Smart gradient checkpointing for transformers.
    
    Usage:
        smart_ckpt = SmartCheckpoint()
        model = smart_ckpt.apply(model, sample_input)
        
        # Or profile first then apply
        profiles = smart_ckpt.profile(model, sample_input)
        layers_to_checkpoint = smart_ckpt.recommend(profiles)
        model = smart_ckpt.apply_to_layers(model, layers_to_checkpoint)
    """
    
    def __init__(
        self,
        mode: str = 'auto',
        memory_threshold: float = 0.7,
        benefit_threshold: float = 0.5,
        use_reentrant: bool = False,
    ):
        self.profiler = MemoryProfiler()
        self.policy = CheckpointPolicy(
            mode=mode,
            memory_threshold=memory_threshold,
            benefit_threshold=benefit_threshold,
        )
        self.use_reentrant = use_reentrant
        
        # Track what we've checkpointed
        self.checkpointed_layers: Set[str] = set()
    
    def profile(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        sample_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, LayerProfile]:
        """Profile model memory usage."""
        return self.profiler.profile(model, sample_input, sample_target)
    
    def recommend(
        self,
        profiles: Dict[str, LayerProfile],
    ) -> Set[str]:
        """Recommend layers to checkpoint."""
        return self.policy.select_layers(profiles)
    
    def apply(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        sample_target: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """
        Profile and apply checkpointing automatically.
        """
        # Profile
        profiles = self.profile(model, sample_input, sample_target)
        
        # Get recommendations
        to_checkpoint = self.recommend(profiles)
        
        print(f"[SmartCheckpoint] Checkpointing {len(to_checkpoint)} layers:")
        for name in sorted(to_checkpoint):
            profile = profiles[name]
            print(f"  - {name}: benefit={profile.checkpoint_benefit:.2f}")
        
        # Apply
        return self.apply_to_layers(model, to_checkpoint)
    
    def apply_to_layers(
        self,
        model: nn.Module,
        layer_names: Set[str],
    ) -> nn.Module:
        """Apply checkpointing to specific layers."""
        for name, module in model.named_modules():
            if name in layer_names:
                self._wrap_layer(model, name, module)
                self.checkpointed_layers.add(name)
        
        return model
    
    def _wrap_layer(self, model: nn.Module, name: str, module: nn.Module):
        """Wrap a layer with checkpointing."""
        wrapped = CheckpointedModule(module, use_reentrant=self.use_reentrant)
        
        # Set the wrapped module
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], wrapped)
    
    def apply_to_transformer_layers(
        self,
        model: nn.Module,
        layer_pattern: str = 'layers',
        checkpoint_every: int = 1,
    ) -> nn.Module:
        """
        Apply checkpointing to transformer layers.
        
        Simple pattern-based approach for common architectures.
        """
        layers_module = None
        layers_name = None
        
        # Find the layers module
        for name, module in model.named_modules():
            if layer_pattern in name and isinstance(module, nn.ModuleList):
                layers_module = module
                layers_name = name
                break
        
        if layers_module is None:
            print(f"[SmartCheckpoint] Warning: Could not find '{layer_pattern}'")
            return model
        
        # Checkpoint every N layers
        checkpoint_indices = set(
            i for i in range(len(layers_module)) if i % checkpoint_every == 0
        )
        
        print(f"[SmartCheckpoint] Checkpointing {len(checkpoint_indices)}/{len(layers_module)} layers")
        
        # Wrap individual layers
        for i in checkpoint_indices:
            wrapped = CheckpointedModule(
                layers_module[i],
                use_reentrant=self.use_reentrant,
            )
            layers_module[i] = wrapped
            self.checkpointed_layers.add(f"{layers_name}.{i}")
        
        return model
    
    def estimate_savings(
        self,
        profiles: Dict[str, LayerProfile],
        checkpointed: Set[str],
    ) -> Dict[str, float]:
        """Estimate memory savings from checkpointing."""
        total_activation_memory = sum(p.activation_memory for p in profiles.values())
        saved_memory = sum(
            profiles[name].activation_memory
            for name in checkpointed
            if name in profiles
        )
        
        # Recompute cost (approximate)
        recompute_cost = sum(
            profiles[name].param_memory  # Proxy for compute
            for name in checkpointed
            if name in profiles
        )
        
        return {
            'total_activation_mb': total_activation_memory / 1024 / 1024,
            'saved_activation_mb': saved_memory / 1024 / 1024,
            'savings_percent': saved_memory / max(1, total_activation_memory) * 100,
            'recompute_cost_proxy': recompute_cost / 1024 / 1024,
            'checkpointed_layers': len(checkpointed),
        }


# =============================================================================
# DYNAMIC CHECKPOINTING
# =============================================================================

class DynamicCheckpoint:
    """
    Dynamically adjusts checkpointing based on batch size.
    
    Larger batches need more memory, so we checkpoint more layers.
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_batch_size: int = 1,
        target_memory_gb: float = 40.0,
    ):
        self.model = model
        self.base_batch_size = base_batch_size
        self.target_memory_gb = target_memory_gb
        
        # Profile at base batch size
        self.smart_ckpt = SmartCheckpoint(mode='auto')
        self.base_profiles: Optional[Dict[str, LayerProfile]] = None
    
    def calibrate(self, sample_input: torch.Tensor):
        """Calibrate with sample input at base batch size."""
        assert sample_input.shape[0] == self.base_batch_size
        self.base_profiles = self.smart_ckpt.profile(self.model, sample_input)
    
    def configure_for_batch(self, batch_size: int) -> Set[str]:
        """Configure checkpointing for a specific batch size."""
        if self.base_profiles is None:
            raise RuntimeError("Call calibrate() first")
        
        # Scale memory by batch size
        scale = batch_size / self.base_batch_size
        
        # Estimate total memory at this batch size
        total_activation = sum(
            p.activation_memory * scale
            for p in self.base_profiles.values()
        )
        total_param = sum(p.param_memory for p in self.base_profiles.values())
        total_grad = sum(p.gradient_memory for p in self.base_profiles.values())
        
        total_memory_gb = (total_activation + total_param + total_grad) / 1e9
        
        if total_memory_gb <= self.target_memory_gb:
            # Fits without checkpointing
            return set()
        
        # Need to checkpoint - be more aggressive with larger batches
        memory_ratio = total_memory_gb / self.target_memory_gb
        
        if memory_ratio > 2.0:
            # Very aggressive
            threshold = 0.2
        elif memory_ratio > 1.5:
            # Aggressive
            threshold = 0.4
        else:
            # Normal
            threshold = 0.6
        
        policy = CheckpointPolicy(
            mode='auto',
            benefit_threshold=threshold,
        )
        
        return policy.select_layers(self.base_profiles)
    
    def apply_for_batch(self, batch_size: int) -> nn.Module:
        """Apply appropriate checkpointing for batch size."""
        layers = self.configure_for_batch(batch_size)
        
        print(f"[DynamicCheckpoint] Batch size {batch_size}: "
              f"checkpointing {len(layers)} layers")
        
        return self.smart_ckpt.apply_to_layers(self.model, layers)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def apply_smart_checkpointing(
    model: nn.Module,
    sample_input: torch.Tensor,
    mode: str = 'auto',
) -> nn.Module:
    """
    Quick function to apply smart checkpointing.
    
    Usage:
        model = apply_smart_checkpointing(model, sample_input)
    """
    smart_ckpt = SmartCheckpoint(mode=mode)
    return smart_ckpt.apply(model, sample_input)


def checkpoint_transformer_layers(
    model: nn.Module,
    checkpoint_every: int = 2,
    layer_pattern: str = 'layers',
) -> nn.Module:
    """
    Simple transformer checkpointing.
    
    Checkpoints every N layers.
    
    Usage:
        model = checkpoint_transformer_layers(model, checkpoint_every=2)
    """
    smart_ckpt = SmartCheckpoint()
    return smart_ckpt.apply_to_transformer_layers(
        model,
        layer_pattern=layer_pattern,
        checkpoint_every=checkpoint_every,
    )


def estimate_checkpoint_memory(
    model: nn.Module,
    sample_input: torch.Tensor,
    checkpoint_ratio: float = 0.5,  # Checkpoint 50% of layers
) -> Dict[str, float]:
    """
    Estimate memory with checkpointing.
    
    Returns estimates for different checkpoint ratios.
    """
    profiler = MemoryProfiler()
    profiles = profiler.profile(model, sample_input)
    
    # Sort by memory
    sorted_profiles = sorted(
        profiles.values(),
        key=lambda p: p.activation_memory,
        reverse=True,
    )
    
    # Checkpoint top X% by memory
    n_checkpoint = int(len(sorted_profiles) * checkpoint_ratio)
    to_checkpoint = {p.name for p in sorted_profiles[:n_checkpoint]}
    
    smart_ckpt = SmartCheckpoint()
    return smart_ckpt.estimate_savings(profiles, to_checkpoint)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Profiling
    'LayerProfile',
    'MemoryProfiler',
    
    # Policy
    'CheckpointPolicy',
    
    # Wrappers
    'CheckpointedModule',
    'CheckpointedSequential',
    
    # Main
    'SmartCheckpoint',
    'DynamicCheckpoint',
    
    # Convenience
    'apply_smart_checkpointing',
    'checkpoint_transformer_layers',
    'estimate_checkpoint_memory',
]


if __name__ == "__main__":
    print("SmartCheckpoint 1.0")
    print("===================")
    print()
    print("Selective gradient checkpointing.")
    print()
    print("Usage:")
    print("  from ryanstream import SmartCheckpoint")
    print()
    print("  # Automatic profiling and application")
    print("  smart_ckpt = SmartCheckpoint(mode='auto')")
    print("  model = smart_ckpt.apply(model, sample_input)")
    print()
    print("  # Or simple transformer checkpointing")
    print("  from ryanstream import checkpoint_transformer_layers")
    print("  model = checkpoint_transformer_layers(model, checkpoint_every=2)")
    print()
    print("Modes:")
    print("  'auto' - Profile and checkpoint high-memory layers")
    print("  'aggressive' - Checkpoint all layers")
    print("  'conservative' - Only checkpoint very high memory layers")
    print("  'none' - Disable checkpointing")
