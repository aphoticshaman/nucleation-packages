"""
RYANSTREAM 1.0
==============

Full-stack inference engine for 72B+ math models.

Components:
- RyanStream Scheduler: Lookahead + KV eviction + auto precision
- Ryan-Pipeline: GPU-direct data loading
- Ryan-Monitor: Local dashboard + email alerts + cloud burst
- Ryan-Format: 1/10th size model serialization
- Ryan-Kernels: Fused CUDA kernels (20% FLOP savings)

Usage:
    from ryanstream import RyanStreamEngine, patch_vllm_scheduler
    
    # Option 1: Wrap existing vLLM
    from vllm import LLM
    llm = LLM(model="...")
    scheduler = patch_vllm_scheduler(llm.llm_engine)
    
    # Option 2: Standalone engine
    engine = RyanStreamEngine(model=model, tokenizer=tokenizer)
    seq_id = engine.add_request("Prove that...")
    while True:
        results = engine.step()
        if not results:
            break
    
    # Full stack
    from ryanstream import (
        create_math_dataloader,  # GPU-direct loading
        RyanMonitor,             # Dashboard + alerts
        RyanFormat,              # Compact serialization
        optimize_model,          # Fused kernels
    )

Author: Ryan J Cardwell (Archer Phoenix)
"""

__version__ = "1.0.0"
__author__ = "Ryan J Cardwell"

# Core scheduler
from .scheduler import (
    RyanStreamScheduler,
    RyanStreamEngine,
    LookaheadPredictor,
    KVCacheManager,
    AutoPrecisionManager,
    patch_vllm_scheduler,
    benchmark_stall_rate,
    SequenceState,
    SequenceStatus,
    PrecisionMode,
)

# Data pipeline
from .pipeline import (
    GPUBufferPool,
    GPUShuffle,
    MMapMathDataset,
    AsyncPrefetchLoader,
    MathTokenizer,
    create_math_dataloader,
    benchmark_loader,
)

# Monitoring
from .monitor import (
    RyanMonitor,
    SpikeDetector,
    EmailAlerter,
    GPUWatchdog,
    CloudBurstManager,
    MetricsDB,
)

# Serialization
from .format import (
    RyanFormat,
    StreamingLoader,
    HuffmanCodec,
    WeightQuantizer,
    DeltaEncoder,
)

# Kernels
from .kernels import (
    FusedAttention,
    FusedFFN,
    replace_attention_layers,
    replace_ffn_layers,
    optimize_model,
    benchmark_kernels,
    HAS_TRITON,
)

# Speculative Decoding
from .speculative import (
    DraftModelSpeculator,
    SelfSpeculator,
    TreeAttention,
    SpeculativeEngine,
    SpeculationStats,
)

# Training-Inference Bridge
from .bridge import (
    RyanConfig,
    CheckpointConverter,
    RyanPipeline,
    DistillationPipeline,
)

# ProofSampler - Constraint-aware decoding
from .sampler import (
    ProofSampler,
    create_proof_sampler,
    ProofState,
    BracketTracker,
    EquationTracker,
    BracketConstraint,
    ProofBeamSearch,
    SymbolicVerifier,
    extract_answer,
)

# RyanQuant - Math-optimized quantization
from .quant import (
    QuantScheme,
    QuantConfig,
    NF4Quantizer,
    OutlierAwareQuantizer,
    DynamicQuantizer,
    QuantizedLinear,
    TokenAwareEmbedding,
    RyanQuantizer,
    generate_math_calibration_data,
)

# PromptCache - Prefix caching
from .cache import (
    PromptCache,
    CacheEntry,
    PrefixHasher,
    CommonPrefixDetector,
    MathPromptTemplates,
    CachedGenerator,
)

# MathTokenizer - Math-native tokenization
from .tokenizer import (
    MathTokenizer,
    MathLexer,
    MathVocab,
    NumberEncoder,
    Token,
    TokenType,
)

# RyanParallel - Simplified tensor parallel
from .parallel import (
    ParallelMode,
    ParallelConfig,
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
    ParallelAttention,
    ParallelMLP,
    RyanParallel,
    setup_distributed,
    estimate_memory,
)

# SmartCheckpoint - Selective gradient checkpointing
from .checkpoint import (
    LayerProfile,
    MemoryProfiler,
    CheckpointPolicy,
    CheckpointedModule,
    SmartCheckpoint,
    DynamicCheckpoint,
    apply_smart_checkpointing,
    checkpoint_transformer_layers,
    estimate_checkpoint_memory,
)

__all__ = [
    # Version
    '__version__',
    
    # Scheduler
    'RyanStreamScheduler',
    'RyanStreamEngine',
    'LookaheadPredictor',
    'KVCacheManager',
    'AutoPrecisionManager',
    'patch_vllm_scheduler',
    'benchmark_stall_rate',
    
    # Pipeline
    'create_math_dataloader',
    'AsyncPrefetchLoader',
    'MMapMathDataset',
    'MathTokenizer',
    'benchmark_loader',
    
    # Monitor
    'RyanMonitor',
    'EmailAlerter',
    'GPUWatchdog',
    'CloudBurstManager',
    
    # Format
    'RyanFormat',
    'StreamingLoader',
    
    # Kernels
    'FusedAttention',
    'FusedFFN',
    'optimize_model',
    'benchmark_kernels',
    'HAS_TRITON',
    
    # Speculative
    'DraftModelSpeculator',
    'SelfSpeculator',
    'TreeAttention',
    'SpeculativeEngine',
    
    # Bridge
    'RyanConfig',
    'CheckpointConverter',
    'RyanPipeline',
    'DistillationPipeline',
    
    # ProofSampler
    'ProofSampler',
    'create_proof_sampler',
    'ProofState',
    'BracketTracker',
    'SymbolicVerifier',
    'extract_answer',
    
    # RyanQuant
    'QuantScheme',
    'QuantConfig',
    'RyanQuantizer',
    'QuantizedLinear',
    'NF4Quantizer',
    
    # PromptCache
    'PromptCache',
    'CachedGenerator',
    'MathPromptTemplates',
    
    # MathTokenizer
    'MathTokenizer',
    'MathLexer',
    'MathVocab',
    'NumberEncoder',
    'TokenType',
    
    # RyanParallel
    'ParallelMode',
    'ParallelConfig',
    'RyanParallel',
    'setup_distributed',
    'estimate_memory',
    'ColumnParallelLinear',
    'RowParallelLinear',
    
    # SmartCheckpoint
    'SmartCheckpoint',
    'DynamicCheckpoint',
    'MemoryProfiler',
    'LayerProfile',
    'apply_smart_checkpointing',
    'checkpoint_transformer_layers',
]
