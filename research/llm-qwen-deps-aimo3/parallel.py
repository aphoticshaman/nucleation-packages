"""
RYANPARALLEL 1.0
================

Simplified distributed training for 72B models.

DeepSpeed/FSDP are over-engineered for our needs.
We just want tensor parallel on 2-8 GPUs.

RyanParallel:
1. Auto-shards model across GPUs
2. Tensor parallel (not pipeline - simpler)
3. Works out of the box
4. No PhD required

Target: Train 72B on 4x A100 without config hell.

Author: Ryan J Cardwell (Archer Phoenix)
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum, auto
import os
import math


# =============================================================================
# CONFIGURATION
# =============================================================================

class ParallelMode(Enum):
    """Parallelism modes."""
    TENSOR = auto()      # Split tensors across GPUs
    DATA = auto()         # Standard data parallel
    HYBRID = auto()       # Tensor + Data parallel


@dataclass
class ParallelConfig:
    """Configuration for parallel training."""
    # World setup
    world_size: int = 1
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    
    # Sharding
    shard_attention: bool = True
    shard_ffn: bool = True
    shard_embeddings: bool = True
    
    # Communication
    async_allreduce: bool = True
    bucket_size_mb: float = 25.0
    
    # Memory
    activation_checkpointing: bool = False
    offload_optimizer: bool = False
    
    @property
    def mode(self) -> ParallelMode:
        if self.tensor_parallel_size > 1 and self.data_parallel_size > 1:
            return ParallelMode.HYBRID
        elif self.tensor_parallel_size > 1:
            return ParallelMode.TENSOR
        else:
            return ParallelMode.DATA


# =============================================================================
# TENSOR PARALLEL LAYERS
# =============================================================================

class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column-wise parallelism.
    
    Splits output features across GPUs.
    Each GPU computes a slice of the output.
    
    Y = XA where A is [in, out/world_size] per GPU
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        world_size: int = 1,
        rank: int = 0,
        gather_output: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.gather_output = gather_output
        
        # Each GPU gets out_features / world_size columns
        assert out_features % world_size == 0
        self.out_features_per_rank = out_features // world_size
        
        # Local weight
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_rank, in_features)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features_per_rank))
        else:
            self.register_parameter('bias', None)
        
        # Initialize
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local matmul: [batch, seq, in] @ [in, out/world] -> [batch, seq, out/world]
        output = torch.matmul(x, self.weight.t())
        
        if self.bias is not None:
            output = output + self.bias
        
        # Optionally gather across GPUs
        if self.gather_output and self.world_size > 1:
            output = self._all_gather(output)
        
        return output
    
    def _all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensor from all ranks."""
        if not dist.is_initialized():
            return tensor
        
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=-1)


class RowParallelLinear(nn.Module):
    """
    Linear layer with row-wise parallelism.
    
    Splits input features across GPUs.
    Each GPU computes partial output, then all-reduce.
    
    Y = XA where A is [in/world_size, out] per GPU
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        world_size: int = 1,
        rank: int = 0,
        input_is_parallel: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.input_is_parallel = input_is_parallel
        
        # Each GPU gets in_features / world_size rows
        assert in_features % world_size == 0
        self.in_features_per_rank = in_features // world_size
        
        # Local weight
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_rank)
        )
        
        # Bias only on rank 0 (added after all-reduce)
        if bias and rank == 0:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is already parallel, use as-is
        # Otherwise, scatter input across ranks
        if not self.input_is_parallel and self.world_size > 1:
            x = self._scatter(x)
        
        # Local matmul
        output = torch.matmul(x, self.weight.t())
        
        # All-reduce to sum partial results
        if self.world_size > 1:
            output = self._all_reduce(output)
        
        # Add bias (only exists on rank 0, but after all-reduce all have same output)
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def _scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        """Scatter input to this rank's slice."""
        if not dist.is_initialized():
            return tensor
        
        # Get this rank's slice
        start = self.rank * self.in_features_per_rank
        end = start + self.in_features_per_rank
        return tensor[..., start:end].contiguous()
    
    def _all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce tensor across ranks."""
        if not dist.is_initialized():
            return tensor
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor


class ParallelEmbedding(nn.Module):
    """
    Embedding with vocabulary parallelism.
    
    Splits vocabulary across GPUs.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.world_size = world_size
        self.rank = rank
        
        # Split vocabulary
        assert num_embeddings % world_size == 0
        self.num_embeddings_per_rank = num_embeddings // world_size
        self.vocab_start = rank * self.num_embeddings_per_rank
        self.vocab_end = self.vocab_start + self.num_embeddings_per_rank
        
        # Local embedding table
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_rank, embedding_dim)
        )
        
        nn.init.normal_(self.weight)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Mask tokens not in this rank's vocabulary
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        
        # Local lookup
        local_ids = input_ids - self.vocab_start
        local_ids = torch.clamp(local_ids, 0, self.num_embeddings_per_rank - 1)
        
        embeddings = torch.embedding(self.weight, local_ids)
        
        # Zero out tokens not in our vocab
        embeddings = embeddings * mask.unsqueeze(-1).float()
        
        # All-reduce to combine
        if self.world_size > 1:
            dist.all_reduce(embeddings, op=dist.ReduceOp.SUM)
        
        return embeddings


# =============================================================================
# ATTENTION PARALLELISM
# =============================================================================

class ParallelAttention(nn.Module):
    """
    Attention with tensor parallelism.
    
    Splits heads across GPUs.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.world_size = world_size
        self.rank = rank
        
        assert num_heads % world_size == 0
        self.num_heads_per_rank = num_heads // world_size
        self.head_dim = hidden_size // num_heads
        
        # Q, K, V projections (column parallel - split heads)
        self.q_proj = ColumnParallelLinear(
            hidden_size, hidden_size,
            world_size=world_size, rank=rank,
            gather_output=False,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size, hidden_size,
            world_size=world_size, rank=rank,
            gather_output=False,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size, hidden_size,
            world_size=world_size, rank=rank,
            gather_output=False,
        )
        
        # Output projection (row parallel - combine heads)
        self.o_proj = RowParallelLinear(
            hidden_size, hidden_size,
            world_size=world_size, rank=rank,
            input_is_parallel=True,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V (each GPU gets subset of heads)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to [batch, heads_per_rank, seq, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads_per_rank, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads_per_rank, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads_per_rank, self.head_dim).transpose(1, 2)
        
        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        # Output projection (row parallel - will all-reduce)
        output = self.o_proj(attn_output)
        
        return output


class ParallelMLP(nn.Module):
    """
    MLP with tensor parallelism.
    
    Splits hidden dimension across GPUs.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        world_size: int = 1,
        rank: int = 0,
        activation: str = 'silu',
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Gate and up projection (column parallel)
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size,
            world_size=world_size, rank=rank,
            gather_output=False,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size,
            world_size=world_size, rank=rank,
            gather_output=False,
        )
        
        # Down projection (row parallel)
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size,
            world_size=world_size, rank=rank,
            input_is_parallel=True,
        )
        
        # Activation
        if activation == 'silu':
            self.act = nn.SiLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: act(gate) * up
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)
        
        # Down projection (will all-reduce)
        return self.down_proj(gate * up)


# =============================================================================
# MODEL PARALLELIZER
# =============================================================================

class RyanParallel:
    """
    Main parallelization wrapper.
    
    Usage:
        parallel = RyanParallel(config)
        parallel.init_distributed()
        model = parallel.parallelize(model)
        optimizer = parallel.wrap_optimizer(optimizer)
    """
    
    def __init__(self, config: ParallelConfig = None):
        self.config = config or ParallelConfig()
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.device = torch.device('cuda:0')
        
        # Process groups
        self.tensor_parallel_group = None
        self.data_parallel_group = None
    
    def init_distributed(
        self,
        backend: str = 'nccl',
        init_method: str = 'env://',
    ):
        """Initialize distributed training."""
        if not dist.is_initialized():
            # Get environment variables
            self.rank = int(os.environ.get('RANK', 0))
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            if self.world_size > 1:
                dist.init_process_group(
                    backend=backend,
                    init_method=init_method,
                    world_size=self.world_size,
                    rank=self.rank,
                )
        else:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = self.rank % torch.cuda.device_count()
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
        
        # Update config
        self.config.world_size = self.world_size
        if self.config.tensor_parallel_size == 1:
            self.config.tensor_parallel_size = self.world_size
        
        # Create process groups
        self._create_process_groups()
        
        print(f"[RyanParallel] Rank {self.rank}/{self.world_size} on {self.device}")
    
    def _create_process_groups(self):
        """Create tensor and data parallel groups."""
        tp_size = self.config.tensor_parallel_size
        dp_size = self.config.data_parallel_size
        
        if self.world_size == 1:
            return
        
        # Tensor parallel groups (GPUs that share model shards)
        # Example: world_size=8, tp_size=4 -> groups: [0,1,2,3], [4,5,6,7]
        for i in range(0, self.world_size, tp_size):
            ranks = list(range(i, i + tp_size))
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.tensor_parallel_group = group
        
        # Data parallel groups (GPUs with same model shard)
        # Example: world_size=8, tp_size=4, dp_size=2 -> groups: [0,4], [1,5], [2,6], [3,7]
        if dp_size > 1:
            for i in range(tp_size):
                ranks = list(range(i, self.world_size, tp_size))
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self.data_parallel_group = group
    
    def parallelize(self, model: nn.Module) -> nn.Module:
        """
        Parallelize a model.
        
        Replaces layers with tensor-parallel versions.
        """
        model = model.to(self.device)
        
        if self.world_size == 1:
            return model
        
        tp_size = self.config.tensor_parallel_size
        tp_rank = self.rank % tp_size
        
        # Replace layers
        if self.config.shard_attention:
            model = self._shard_attention(model, tp_size, tp_rank)
        
        if self.config.shard_ffn:
            model = self._shard_ffn(model, tp_size, tp_rank)
        
        if self.config.shard_embeddings:
            model = self._shard_embeddings(model, tp_size, tp_rank)
        
        # Wrap with DDP for data parallelism
        if self.config.data_parallel_size > 1 and self.data_parallel_group is not None:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                process_group=self.data_parallel_group,
                bucket_cap_mb=self.config.bucket_size_mb,
            )
        
        return model
    
    def _shard_attention(
        self,
        model: nn.Module,
        tp_size: int,
        tp_rank: int,
    ) -> nn.Module:
        """Replace attention layers with parallel versions."""
        for name, module in model.named_modules():
            # Find attention layers
            if 'attention' in name.lower() or 'attn' in name.lower():
                if hasattr(module, 'q_proj') and hasattr(module, 'num_heads'):
                    # Create parallel attention
                    hidden_size = module.q_proj.in_features
                    num_heads = module.num_heads
                    
                    parallel_attn = ParallelAttention(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        world_size=tp_size,
                        rank=tp_rank,
                    )
                    
                    # Copy weights (sliced)
                    self._copy_attention_weights(module, parallel_attn, tp_size, tp_rank)
                    
                    # Replace
                    self._set_module(model, name, parallel_attn)
        
        return model
    
    def _shard_ffn(
        self,
        model: nn.Module,
        tp_size: int,
        tp_rank: int,
    ) -> nn.Module:
        """Replace FFN/MLP layers with parallel versions."""
        for name, module in model.named_modules():
            if 'mlp' in name.lower() or 'ffn' in name.lower():
                if hasattr(module, 'gate_proj') and hasattr(module, 'up_proj'):
                    hidden_size = module.gate_proj.in_features
                    intermediate_size = module.gate_proj.out_features
                    
                    parallel_mlp = ParallelMLP(
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        world_size=tp_size,
                        rank=tp_rank,
                    )
                    
                    # Copy weights
                    self._copy_mlp_weights(module, parallel_mlp, tp_size, tp_rank)
                    
                    # Replace
                    self._set_module(model, name, parallel_mlp)
        
        return model
    
    def _shard_embeddings(
        self,
        model: nn.Module,
        tp_size: int,
        tp_rank: int,
    ) -> nn.Module:
        """Replace embedding layers with parallel versions."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                if module.num_embeddings % tp_size == 0:
                    parallel_emb = ParallelEmbedding(
                        num_embeddings=module.num_embeddings,
                        embedding_dim=module.embedding_dim,
                        world_size=tp_size,
                        rank=tp_rank,
                    )
                    
                    # Copy weights (sliced)
                    start = tp_rank * parallel_emb.num_embeddings_per_rank
                    end = start + parallel_emb.num_embeddings_per_rank
                    parallel_emb.weight.data = module.weight.data[start:end].clone()
                    
                    # Replace
                    self._set_module(model, name, parallel_emb)
        
        return model
    
    def _copy_attention_weights(self, src, dst, tp_size, tp_rank):
        """Copy and shard attention weights."""
        hidden_per_rank = dst.q_proj.out_features_per_rank
        start = tp_rank * hidden_per_rank
        end = start + hidden_per_rank
        
        # Copy Q, K, V projections (column parallel - slice output)
        dst.q_proj.weight.data = src.q_proj.weight.data[start:end].clone()
        dst.k_proj.weight.data = src.k_proj.weight.data[start:end].clone()
        dst.v_proj.weight.data = src.v_proj.weight.data[start:end].clone()
        
        if src.q_proj.bias is not None:
            dst.q_proj.bias.data = src.q_proj.bias.data[start:end].clone()
            dst.k_proj.bias.data = src.k_proj.bias.data[start:end].clone()
            dst.v_proj.bias.data = src.v_proj.bias.data[start:end].clone()
        
        # Copy O projection (row parallel - slice input)
        dst.o_proj.weight.data = src.o_proj.weight.data[:, start:end].clone()
        if tp_rank == 0 and src.o_proj.bias is not None:
            dst.o_proj.bias.data = src.o_proj.bias.data.clone()
    
    def _copy_mlp_weights(self, src, dst, tp_size, tp_rank):
        """Copy and shard MLP weights."""
        inter_per_rank = dst.gate_proj.out_features_per_rank
        start = tp_rank * inter_per_rank
        end = start + inter_per_rank
        
        # Gate and up (column parallel - slice output)
        dst.gate_proj.weight.data = src.gate_proj.weight.data[start:end].clone()
        dst.up_proj.weight.data = src.up_proj.weight.data[start:end].clone()
        
        if src.gate_proj.bias is not None:
            dst.gate_proj.bias.data = src.gate_proj.bias.data[start:end].clone()
            dst.up_proj.bias.data = src.up_proj.bias.data[start:end].clone()
        
        # Down (row parallel - slice input)
        dst.down_proj.weight.data = src.down_proj.weight.data[:, start:end].clone()
        if tp_rank == 0 and src.down_proj.bias is not None:
            dst.down_proj.bias.data = src.down_proj.bias.data.clone()
    
    def _set_module(self, model: nn.Module, name: str, new_module: nn.Module):
        """Set a nested module by name."""
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
    
    def wrap_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer:
        """Wrap optimizer for distributed training."""
        # For now, just return as-is
        # Could add ZeRO-style optimizer sharding later
        return optimizer
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        path: str,
        **kwargs,
    ):
        """Save distributed checkpoint."""
        if self.rank == 0:
            # Gather model state
            if isinstance(model, DDP):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            checkpoint = {
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                **kwargs,
            }
            
            torch.save(checkpoint, path)
            print(f"[RyanParallel] Saved checkpoint to {path}")
        
        # Sync
        if dist.is_initialized():
            dist.barrier()
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        path: str,
    ) -> Dict[str, Any]:
        """Load distributed checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        print(f"[RyanParallel] Loaded checkpoint from {path}")
        return checkpoint


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def setup_distributed(
    tensor_parallel_size: int = None,
    backend: str = 'nccl',
) -> RyanParallel:
    """
    Quick setup for distributed training.
    
    Usage:
        parallel = setup_distributed(tensor_parallel_size=4)
        model = parallel.parallelize(model)
    """
    world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    
    if tensor_parallel_size is None:
        tensor_parallel_size = world_size
    
    config = ParallelConfig(
        world_size=world_size,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=world_size // tensor_parallel_size,
    )
    
    parallel = RyanParallel(config)
    parallel.init_distributed(backend=backend)
    
    return parallel


def estimate_memory(
    model_params_billions: float,
    tensor_parallel_size: int,
    batch_size: int = 1,
    seq_length: int = 2048,
    dtype_bytes: int = 2,  # FP16/BF16
) -> Dict[str, float]:
    """
    Estimate memory usage per GPU.
    
    Returns dict with memory estimates in GB.
    """
    params_per_gpu = model_params_billions * 1e9 / tensor_parallel_size
    
    # Model weights
    weights_gb = params_per_gpu * dtype_bytes / 1e9
    
    # Optimizer states (AdamW: 2x for momentum + variance)
    optimizer_gb = weights_gb * 2
    
    # Gradients
    gradients_gb = weights_gb
    
    # Activations (rough estimate)
    hidden_size = int(math.sqrt(params_per_gpu / 100))  # Very rough
    activations_gb = batch_size * seq_length * hidden_size * dtype_bytes / 1e9
    
    total_gb = weights_gb + optimizer_gb + gradients_gb + activations_gb
    
    return {
        'weights_gb': weights_gb,
        'optimizer_gb': optimizer_gb,
        'gradients_gb': gradients_gb,
        'activations_gb': activations_gb,
        'total_gb': total_gb,
        'recommended_vram_gb': total_gb * 1.2,  # 20% buffer
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Config
    'ParallelMode',
    'ParallelConfig',
    
    # Layers
    'ColumnParallelLinear',
    'RowParallelLinear',
    'ParallelEmbedding',
    'ParallelAttention',
    'ParallelMLP',
    
    # Main
    'RyanParallel',
    'setup_distributed',
    'estimate_memory',
]


if __name__ == "__main__":
    print("RyanParallel 1.0")
    print("================")
    print()
    print("Simplified tensor parallel for 72B models.")
    print()
    print("Usage:")
    print("  # Launch with torchrun:")
    print("  # torchrun --nproc_per_node=4 train.py")
    print()
    print("  from ryanstream import setup_distributed")
    print("  parallel = setup_distributed(tensor_parallel_size=4)")
    print("  model = parallel.parallelize(model)")
    print()
    
    # Memory estimate
    print("Memory estimate for Qwen-72B on 4x A100:")
    mem = estimate_memory(72, tensor_parallel_size=4, batch_size=1)
    for k, v in mem.items():
        print(f"  {k}: {v:.1f} GB")
