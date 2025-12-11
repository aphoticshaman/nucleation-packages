"""
RYANQUANT 1.0
=============

Math-optimized quantization.

bitsandbytes is generic. We can beat it by:
1. Per-layer precision (attention needs more than FFN)
2. Token-aware quantization (numbers need more precision)
3. Dynamic precision based on activation patterns
4. Math-specific calibration data
5. Outlier-aware quantization

Target: Better accuracy than bitsandbytes at same memory.

Author: Ryan J Cardwell (Archer Phoenix)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import math


# =============================================================================
# QUANTIZATION SCHEMES
# =============================================================================

class QuantScheme(Enum):
    """Quantization schemes."""
    INT8 = auto()       # Standard 8-bit
    INT4 = auto()       # Standard 4-bit
    NF4 = auto()        # Normal float 4-bit (bitsandbytes style)
    FP8 = auto()        # 8-bit float
    MIXED = auto()      # Mixed precision per-layer
    DYNAMIC = auto()    # Dynamic based on activations


@dataclass
class QuantConfig:
    """Configuration for quantization."""
    scheme: QuantScheme = QuantScheme.NF4
    
    # Per-layer settings
    attention_bits: int = 8  # Attention needs more precision
    ffn_bits: int = 4        # FFN can be more aggressive
    embedding_bits: int = 8  # Keep embeddings higher
    lm_head_bits: int = 8    # Output layer important
    
    # Token-aware
    number_token_boost: bool = True  # Higher precision for number tokens
    operator_token_boost: bool = True  # Higher precision for math operators
    
    # Calibration
    use_calibration: bool = True
    calibration_samples: int = 128
    
    # Outlier handling
    outlier_threshold: float = 6.0  # Standard deviations
    outlier_bits: int = 16  # Keep outliers in higher precision
    
    # Block size for quantization
    block_size: int = 64


# =============================================================================
# NF4 QUANTIZATION (Our Implementation)
# =============================================================================

class NF4Quantizer:
    """
    Normal Float 4-bit quantization.
    
    NF4 uses a lookup table optimized for normally-distributed weights.
    Better than uniform INT4 for neural network weights.
    """
    
    # NF4 quantization levels (optimized for normal distribution)
    # Correct NF4 levels
    NF4_LEVELS = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0,
    ])
    
    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        block_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to NF4.
        
        Returns:
            (quantized_indices, scales, zero_points)
        """
        original_shape = tensor.shape
        tensor = tensor.flatten()
        
        # Pad to block size
        pad_size = (block_size - tensor.numel() % block_size) % block_size
        if pad_size:
            tensor = F.pad(tensor, (0, pad_size))
        
        # Reshape to blocks
        tensor = tensor.view(-1, block_size)
        
        # Compute per-block scale (absmax)
        scales = tensor.abs().max(dim=1, keepdim=True)[0]
        scales = torch.clamp(scales, min=1e-8)
        
        # Normalize to [-1, 1]
        normalized = tensor / scales
        
        # Quantize to NF4 indices
        levels = cls.NF4_LEVELS.to(tensor.device)
        # Find nearest level
        distances = (normalized.unsqueeze(-1) - levels.unsqueeze(0).unsqueeze(0)).abs()
        indices = distances.argmin(dim=-1).to(torch.uint8)
        
        return indices, scales.squeeze(-1), original_shape
    
    @classmethod
    def dequantize(
        cls,
        indices: torch.Tensor,
        scales: torch.Tensor,
        original_shape: torch.Size,
        block_size: int = 64,
    ) -> torch.Tensor:
        """Dequantize NF4 back to float."""
        levels = cls.NF4_LEVELS.to(indices.device)
        
        # Map indices to levels
        values = levels[indices.long()]
        
        # Scale back
        values = values * scales.unsqueeze(-1)
        
        # Flatten and trim to original size
        values = values.flatten()
        numel = 1
        for s in original_shape:
            numel *= s
        values = values[:numel]
        
        return values.view(original_shape)


# =============================================================================
# OUTLIER-AWARE QUANTIZATION
# =============================================================================

class OutlierAwareQuantizer:
    """
    Handles outliers separately to preserve accuracy.
    
    Transformer weights often have outliers that break quantization.
    We detect and store outliers in higher precision.
    """
    
    def __init__(
        self,
        threshold: float = 6.0,  # Standard deviations
        outlier_bits: int = 16,
        main_bits: int = 4,
    ):
        self.threshold = threshold
        self.outlier_bits = outlier_bits
        self.main_bits = main_bits
    
    def quantize(
        self,
        tensor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize with outlier handling.
        
        Returns dict with:
            - main_quantized: Low-bit main values
            - outlier_mask: Where outliers are
            - outlier_values: Full precision outliers
            - scale, zero_point: For main quantization
        """
        # Detect outliers
        mean = tensor.mean()
        std = tensor.std()
        outlier_mask = (tensor - mean).abs() > self.threshold * std
        
        # Store outliers in higher precision
        outlier_values = tensor[outlier_mask].clone()
        
        # Quantize non-outliers
        non_outlier = tensor.clone()
        non_outlier[outlier_mask] = mean  # Replace outliers with mean
        
        # Quantize main values
        if self.main_bits == 4:
            indices, scales, shape = NF4Quantizer.quantize(non_outlier)
            main_quantized = indices
        else:
            # INT8 fallback
            scale = non_outlier.abs().max() / 127
            main_quantized = (non_outlier / scale).round().clamp(-128, 127).to(torch.int8)
            scales = scale
            shape = tensor.shape
        
        return {
            'main_quantized': main_quantized,
            'outlier_mask': outlier_mask,
            'outlier_values': outlier_values,
            'scales': scales,
            'shape': shape,
            'outlier_indices': outlier_mask.nonzero().squeeze(-1),
        }
    
    def dequantize(self, quant_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Dequantize with outliers restored."""
        # Dequantize main
        if quant_dict['main_quantized'].dtype == torch.uint8:
            # NF4
            tensor = NF4Quantizer.dequantize(
                quant_dict['main_quantized'],
                quant_dict['scales'],
                quant_dict['shape'],
            )
        else:
            # INT8
            tensor = quant_dict['main_quantized'].float() * quant_dict['scales']
            tensor = tensor.view(quant_dict['shape'])
        
        # Restore outliers (if any)
        if quant_dict['outlier_values'].numel() > 0:
            flat = tensor.flatten()
            indices = quant_dict['outlier_indices']
            if indices.dim() == 1:
                flat[indices] = quant_dict['outlier_values']
            tensor = flat.view(quant_dict['shape'])
        
        return tensor


# =============================================================================
# DYNAMIC QUANTIZATION
# =============================================================================

class DynamicQuantizer:
    """
    Dynamic quantization based on activation patterns.
    
    Adjusts precision on-the-fly based on:
    - Input complexity
    - Layer importance (measured by gradient magnitude)
    - Token type (numbers vs text)
    """
    
    def __init__(
        self,
        min_bits: int = 4,
        max_bits: int = 16,
        complexity_threshold: float = 0.5,
    ):
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.complexity_threshold = complexity_threshold
        
        # Track layer importance
        self.layer_importance: Dict[str, float] = {}
    
    def compute_complexity(self, activations: torch.Tensor) -> float:
        """
        Compute activation complexity score.
        
        High complexity → need more bits.
        """
        # Entropy-based complexity
        flat = activations.flatten()
        
        # Discretize for histogram
        bins = 256
        hist = torch.histc(flat, bins=bins)
        hist = hist / hist.sum()
        
        # Compute entropy
        entropy = -(hist * torch.log2(hist + 1e-10)).sum()
        max_entropy = math.log2(bins)
        
        complexity = entropy / max_entropy
        return complexity.item()
    
    def select_bits(
        self,
        layer_name: str,
        activations: torch.Tensor,
    ) -> int:
        """Select number of bits based on complexity."""
        complexity = self.compute_complexity(activations)
        importance = self.layer_importance.get(layer_name, 0.5)
        
        # Combined score
        score = 0.7 * complexity + 0.3 * importance
        
        # Map to bits
        bits = int(self.min_bits + score * (self.max_bits - self.min_bits))
        bits = max(self.min_bits, min(self.max_bits, bits))
        
        # Round to supported bit widths
        if bits <= 4:
            return 4
        elif bits <= 8:
            return 8
        else:
            return 16
    
    def update_importance(self, layer_name: str, gradient_norm: float):
        """Update layer importance based on gradients."""
        # EMA update
        alpha = 0.1
        current = self.layer_importance.get(layer_name, 0.5)
        normalized = min(1.0, gradient_norm)  # Assume normalized
        self.layer_importance[layer_name] = alpha * normalized + (1 - alpha) * current


# =============================================================================
# QUANTIZED LINEAR LAYER
# =============================================================================

class QuantizedLinear(nn.Module):
    """
    Quantized linear layer with math-optimized settings.
    
    Features:
    - Configurable bit width
    - Outlier handling
    - Optional dynamic precision
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 4,
        use_outlier_handling: bool = True,
        outlier_threshold: float = 6.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.use_outlier_handling = use_outlier_handling
        
        # Quantization
        if use_outlier_handling:
            self.quantizer = OutlierAwareQuantizer(
                threshold=outlier_threshold,
                main_bits=bits,
            )
        else:
            self.quantizer = None
        
        # Will be populated during quantization
        self.weight_quantized: Optional[Dict] = None
        self.weight_scale: Optional[torch.Tensor] = None
        
        # Bias (keep in full precision)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # For initialization
        self._weight_placeholder = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self._weight_placeholder)
    
    def quantize_weights(self, weight: torch.Tensor):
        """Quantize weights."""
        if self.quantizer:
            self.weight_quantized = self.quantizer.quantize(weight)
        else:
            indices, scales, shape = NF4Quantizer.quantize(weight)
            self.weight_quantized = {
                'main_quantized': indices,
                'scales': scales,
                'shape': shape,
            }
    
    def get_weight(self) -> torch.Tensor:
        """Dequantize weights for computation."""
        if self.weight_quantized is None:
            return self._weight_placeholder
        
        if self.quantizer:
            return self.quantizer.dequantize(self.weight_quantized)
        else:
            return NF4Quantizer.dequantize(
                self.weight_quantized['main_quantized'],
                self.weight_quantized['scales'],
                self.weight_quantized['shape'],
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.get_weight()
        return F.linear(x, weight, self.bias)
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        use_outlier_handling: bool = True,
    ) -> 'QuantizedLinear':
        """Convert a regular Linear layer to QuantizedLinear."""
        quant_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            bits=bits,
            use_outlier_handling=use_outlier_handling,
        )
        
        # Quantize weights
        quant_linear.quantize_weights(linear.weight.data)
        
        # Copy bias
        if linear.bias is not None:
            quant_linear.bias.data = linear.bias.data.clone()
        
        return quant_linear


# =============================================================================
# TOKEN-AWARE QUANTIZATION
# =============================================================================

class TokenAwareEmbedding(nn.Module):
    """
    Embedding layer with token-type-aware precision.
    
    Number and operator tokens get higher precision.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        number_token_ids: List[int] = None,
        operator_token_ids: List[int] = None,
        main_bits: int = 4,
        special_bits: int = 8,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.main_bits = main_bits
        self.special_bits = special_bits
        
        # Token sets
        self.number_tokens = set(number_token_ids or [])
        self.operator_tokens = set(operator_token_ids or [])
        self.special_tokens = self.number_tokens | self.operator_tokens
        
        # Separate storage for special vs regular tokens
        self.register_buffer('main_quantized', None)
        self.register_buffer('main_scales', None)
        self.register_buffer('special_embeddings', None)
        self.register_buffer('special_indices', None)
    
    def quantize(self, embeddings: torch.Tensor):
        """Quantize embedding table."""
        # Identify special token rows
        special_mask = torch.zeros(self.num_embeddings, dtype=torch.bool)
        for idx in self.special_tokens:
            if idx < self.num_embeddings:
                special_mask[idx] = True
        
        # Store special embeddings in higher precision (FP16)
        self.special_indices = special_mask.nonzero().squeeze(-1)
        self.special_embeddings = embeddings[special_mask].half()
        
        # Quantize rest
        regular_embeddings = embeddings.clone()
        regular_embeddings[special_mask] = 0  # Zero out special
        
        indices, scales, shape = NF4Quantizer.quantize(regular_embeddings)
        self.main_quantized = indices
        self.main_scales = scales
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings with mixed precision."""
        # Dequantize main embeddings
        main_emb = NF4Quantizer.dequantize(
            self.main_quantized,
            self.main_scales,
            (self.num_embeddings, self.embedding_dim),
        )
        
        # Restore special embeddings
        if self.special_embeddings is not None:
            main_emb[self.special_indices] = self.special_embeddings.float()
        
        # Look up
        return F.embedding(input_ids, main_emb)


# =============================================================================
# MODEL QUANTIZER
# =============================================================================

class RyanQuantizer:
    """
    Main quantizer for entire models.
    
    Usage:
        quantizer = RyanQuantizer(config)
        quantized_model = quantizer.quantize(model)
    """
    
    def __init__(self, config: QuantConfig = None):
        self.config = config or QuantConfig()
        
        # Layer type detection patterns
        self.attention_patterns = ['attn', 'attention', 'self_attn', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
        self.ffn_patterns = ['mlp', 'ffn', 'feed_forward', 'up_proj', 'down_proj', 'gate_proj']
        self.embedding_patterns = ['embed', 'wte', 'wpe']
        self.lm_head_patterns = ['lm_head', 'output', 'classifier']
    
    def _get_layer_bits(self, name: str) -> int:
        """Determine bits for a layer based on its name."""
        name_lower = name.lower()
        
        if any(p in name_lower for p in self.embedding_patterns):
            return self.config.embedding_bits
        
        if any(p in name_lower for p in self.lm_head_patterns):
            return self.config.lm_head_bits
        
        if any(p in name_lower for p in self.attention_patterns):
            return self.config.attention_bits
        
        if any(p in name_lower for p in self.ffn_patterns):
            return self.config.ffn_bits
        
        # Default to FFN bits (more aggressive)
        return self.config.ffn_bits
    
    def quantize(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """
        Quantize entire model.
        
        Args:
            model: Model to quantize
            calibration_data: Optional data for calibration
        
        Returns:
            Quantized model
        """
        # Calibrate if requested
        if self.config.use_calibration and calibration_data is not None:
            self._calibrate(model, calibration_data)
        
        # Quantize layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                bits = self._get_layer_bits(name)
                
                # Replace with quantized version
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                parent = model
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                
                quant_linear = QuantizedLinear.from_linear(
                    module,
                    bits=bits,
                    use_outlier_handling=(self.config.outlier_threshold > 0),
                )
                
                setattr(parent, child_name, quant_linear)
                print(f"[RyanQuant] {name}: {bits}-bit")
        
        return model
    
    def _calibrate(self, model: nn.Module, data: torch.Tensor):
        """Run calibration pass to collect statistics."""
        model.eval()
        
        hooks = []
        activation_stats = {}
        
        def make_hook(name):
            def hook(module, input, output):
                if name not in activation_stats:
                    activation_stats[name] = {
                        'min': float('inf'),
                        'max': float('-inf'),
                        'mean': 0.0,
                        'count': 0,
                    }
                
                stats = activation_stats[name]
                flat = output.flatten().float()
                stats['min'] = min(stats['min'], flat.min().item())
                stats['max'] = max(stats['max'], flat.max().item())
                stats['mean'] += flat.mean().item()
                stats['count'] += 1
            
            return hook
        
        # Register hooks
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(make_hook(name))
                hooks.append(h)
        
        # Run calibration
        with torch.no_grad():
            for i in range(min(self.config.calibration_samples, data.shape[0])):
                sample = data[i:i+1]
                model(sample)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        # Store calibration results
        self.calibration_stats = activation_stats
    
    def get_memory_savings(self, model: nn.Module) -> Dict[str, float]:
        """Calculate memory savings from quantization."""
        original_bytes = 0
        quantized_bytes = 0
        
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLinear):
                # Original would be FP16
                original = module.in_features * module.out_features * 2
                
                # Quantized
                quant = module.in_features * module.out_features * module.bits / 8
                
                original_bytes += original
                quantized_bytes += quant
        
        return {
            'original_mb': original_bytes / 1024 / 1024,
            'quantized_mb': quantized_bytes / 1024 / 1024,
            'compression_ratio': original_bytes / max(1, quantized_bytes),
            'savings_percent': (1 - quantized_bytes / max(1, original_bytes)) * 100,
        }


# =============================================================================
# MATH-SPECIFIC CALIBRATION DATA
# =============================================================================

def generate_math_calibration_data(
    tokenizer: Any,
    num_samples: int = 128,
    max_length: int = 256,
) -> torch.Tensor:
    """
    Generate calibration data with math-specific content.
    
    Includes:
    - Equations
    - Numbers
    - Proofs
    - Common math patterns
    """
    math_templates = [
        "Let x = {n1}. Then x + {n2} = {n3}.",
        "Prove that {n1} + {n2} = {n3}.",
        "If a = {n1} and b = {n2}, then a * b = {n4}.",
        "Consider the equation x^2 + {n1}x + {n2} = 0.",
        "The sum of {n1} and {n2} is {n3}.",
        "Given that n = {n1}, show that n^2 = {n5}.",
        "Suppose x > {n1}. Then x + 1 > {n2}.",
        "By induction, assume P(k) for k = {n1}.",
        "Therefore, {n1} ≡ {n2} (mod {n6}).",
        "The GCD of {n1} and {n2} is {n7}.",
    ]
    
    import random
    samples = []
    
    for _ in range(num_samples):
        template = random.choice(math_templates)
        n1 = random.randint(1, 100)
        n2 = random.randint(1, 100)
        n3 = n1 + n2
        n4 = n1 * n2
        n5 = n1 * n1
        n6 = random.randint(2, 20)
        n7 = math.gcd(n1, n2)
        
        text = template.format(n1=n1, n2=n2, n3=n3, n4=n4, n5=n5, n6=n6, n7=n7)
        samples.append(text)
    
    # Tokenize
    encoded = tokenizer(
        samples,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    
    return encoded['input_ids']


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Config
    'QuantScheme',
    'QuantConfig',
    
    # Core quantizers
    'NF4Quantizer',
    'OutlierAwareQuantizer',
    'DynamicQuantizer',
    
    # Layers
    'QuantizedLinear',
    'TokenAwareEmbedding',
    
    # Main interface
    'RyanQuantizer',
    
    # Utils
    'generate_math_calibration_data',
]


if __name__ == "__main__":
    print("RyanQuant 1.0")
    print("=============")
    print()
    print("Math-optimized quantization.")
    print()
    print("Features:")
    print("  - Per-layer precision (attention 8-bit, FFN 4-bit)")
    print("  - Outlier-aware quantization")
    print("  - Token-type-aware embeddings")
    print("  - Dynamic precision based on complexity")
    print("  - Math-specific calibration")
    print()
    print("Usage:")
    print("  config = QuantConfig(attention_bits=8, ffn_bits=4)")
    print("  quantizer = RyanQuantizer(config)")
    print("  model = quantizer.quantize(model)")
