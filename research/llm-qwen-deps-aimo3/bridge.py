"""
RYAN-BRIDGE 1.0
===============

Wires Ryan-Optimizer (training) to RyanStream (inference).

The Complete Pipeline:
1. Train with Ryan-Optimizer (custom scheduler, EchoKill, etc.)
2. Save with RyanFormat (10x compression)
3. Deploy with RyanStream (speculative decoding, auto-precision)
4. Monitor with RyanMonitor (alerts, cloud burst)

This module provides:
- Unified config that flows from training to inference
- Checkpoint conversion (optimizer states → inference-ready)
- Warm-start inference from training checkpoints
- Continuous training from inference state
- Distillation pipeline (big model → draft model)

Author: Ryan J Cardwell (Archer Phoenix)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import time
import copy


# =============================================================================
# UNIFIED CONFIG
# =============================================================================

@dataclass
class RyanConfig:
    """
    Unified configuration for training and inference.
    
    Flows through the entire pipeline:
    Training → Checkpoint → Inference → Monitoring
    """
    
    # Model
    model_name: str = "custom"
    model_type: str = "transformer"
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    vocab_size: int = 32000
    max_seq_len: int = 4096
    
    # Training (Ryan-Optimizer)
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    use_ghost_filter: bool = True
    use_echo_kill: bool = True
    use_spectral_gelu: bool = True
    use_mind_hive: bool = True
    use_low_noise: bool = True
    use_regret_guard: bool = True
    
    # Inference (RyanStream)
    max_batch_size: int = 32
    lookahead_tokens: int = 10
    kv_eviction_threshold: float = 3.0
    auto_precision: bool = True
    vram_high_threshold: float = 0.80
    vram_low_threshold: float = 0.60
    
    # Speculative
    use_speculation: bool = True
    speculation_mode: str = "self"  # "draft", "self", "tree", "auto"
    max_speculation_length: int = 8
    draft_model_path: Optional[str] = None
    
    # Serialization
    use_quantization: bool = True
    use_huffman: bool = True
    use_delta: bool = True
    
    # Monitoring
    enable_monitoring: bool = True
    alert_email: Optional[str] = None
    loss_spike_threshold: float = 3.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'RyanConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'RyanConfig':
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# CHECKPOINT CONVERTER
# =============================================================================

class CheckpointConverter:
    """
    Converts between training and inference checkpoints.
    
    Training checkpoint:
    - Model weights
    - Optimizer state (Adam moments, etc.)
    - Scheduler state (GhostFilter history)
    - EchoKill memory
    - Training metadata
    
    Inference checkpoint:
    - Model weights (possibly quantized)
    - KV cache config
    - Speculation config
    - No optimizer cruft
    """
    
    @staticmethod
    def training_to_inference(
        training_checkpoint: Dict,
        config: RyanConfig,
        quantize: bool = True,
    ) -> Dict:
        """
        Convert training checkpoint to inference-ready format.
        
        Strips optimizer state, quantizes if requested.
        """
        inference_ckpt = {
            'model_state_dict': {},
            'config': config.to_dict(),
            'created_at': time.time(),
            'source': 'training',
        }
        
        # Extract model weights
        if 'model_state_dict' in training_checkpoint:
            model_state = training_checkpoint['model_state_dict']
        elif 'state_dict' in training_checkpoint:
            model_state = training_checkpoint['state_dict']
        else:
            # Assume top-level is model state
            model_state = {
                k: v for k, v in training_checkpoint.items()
                if isinstance(v, torch.Tensor)
            }
        
        # Process weights
        for name, tensor in model_state.items():
            # Skip optimizer-related keys
            if any(x in name for x in ['optimizer', 'scheduler', 'echo_kill', 'ghost_filter']):
                continue
            
            inference_ckpt['model_state_dict'][name] = tensor
        
        # Extract useful training metadata
        if 'echo_kill_memory' in training_checkpoint:
            # Save EchoKill patterns for potential warm-start
            inference_ckpt['echo_kill_memory'] = training_checkpoint['echo_kill_memory']
        
        if 'training_steps' in training_checkpoint:
            inference_ckpt['training_steps'] = training_checkpoint['training_steps']
        
        return inference_ckpt
    
    @staticmethod
    def inference_to_training(
        inference_checkpoint: Dict,
        config: RyanConfig,
        optimizer_class: type = None,
    ) -> Dict:
        """
        Convert inference checkpoint back to training format.
        
        Useful for continuing training from deployed model.
        """
        training_ckpt = {
            'model_state_dict': inference_checkpoint['model_state_dict'],
            'config': config.to_dict(),
            'created_at': time.time(),
            'source': 'inference',
        }
        
        # Initialize fresh optimizer state (can't recover moments)
        # Training loop will need to warm up
        training_ckpt['requires_warmup'] = True
        
        # Restore EchoKill memory if available
        if 'echo_kill_memory' in inference_checkpoint:
            training_ckpt['echo_kill_memory'] = inference_checkpoint['echo_kill_memory']
        
        return training_ckpt


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class RyanPipeline:
    """
    Unified training-to-inference pipeline.
    
    Usage:
        pipeline = RyanPipeline(config)
        
        # Training phase
        pipeline.setup_training(model)
        for batch in dataloader:
            loss = pipeline.train_step(batch)
        pipeline.save_checkpoint("model.ryan")
        
        # Inference phase
        pipeline.setup_inference()
        output = pipeline.generate("Prove that...")
    """
    
    def __init__(self, config: RyanConfig):
        self.config = config
        
        # Components (lazy initialized)
        self.model: Optional[nn.Module] = None
        self.optimizer = None
        self.scheduler = None
        self.inference_engine = None
        self.monitor = None
        
        # State
        self.mode: str = 'init'  # 'init', 'training', 'inference'
        self.step: int = 0
    
    def setup_training(
        self,
        model: nn.Module,
        dataloader: Any = None,
    ):
        """Initialize training components."""
        self.model = model
        self.mode = 'training'
        
        # Try to import Ryan-Optimizer
        try:
            import sys
            sys.path.insert(0, '/home/claude')
            from ryan_optimizer import (
                create_ryan_optimizer_and_scheduler,
                replace_dropout_with_echokill,
                replace_gelu_with_spectralgelu,
            )
            
            # Apply layer replacements
            if self.config.use_echo_kill:
                self.model = replace_dropout_with_echokill(self.model)
            
            if self.config.use_spectral_gelu:
                self.model = replace_gelu_with_spectralgelu(self.model)
            
            # Create optimizer and scheduler
            self.optimizer, self.scheduler = create_ryan_optimizer_and_scheduler(
                self.model,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            
            print("[RyanPipeline] Training setup with Ryan-Optimizer")
            
        except ImportError:
            # Fallback to standard AdamW
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=10000,
            )
            print("[RyanPipeline] Training setup with standard AdamW (Ryan-Optimizer not found)")
        
        # Setup monitoring
        if self.config.enable_monitoring:
            try:
                from .monitor import RyanMonitor
                
                self.monitor = RyanMonitor(
                    spike_threshold=self.config.loss_spike_threshold,
                    email_config={'to_addrs': [self.config.alert_email]} if self.config.alert_email else {},
                )
                self.monitor.start()
            except ImportError:
                pass
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        loss_fn: Callable = None,
    ) -> float:
        """Execute one training step."""
        assert self.mode == 'training', "Must call setup_training first"
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward
        outputs = self.model(**batch)
        
        if loss_fn:
            loss = loss_fn(outputs, batch)
        elif hasattr(outputs, 'loss'):
            loss = outputs.loss
        else:
            raise ValueError("No loss found. Provide loss_fn or model with .loss attribute")
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Scheduler update (for GhostFilter)
        if hasattr(self.scheduler, 'register_gradient_norm'):
            self.scheduler.register_gradient_norm(grad_norm.item())
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        self.step += 1
        
        # Monitor
        if self.monitor:
            self.monitor.log('loss', loss.item(), step=self.step)
            self.monitor.log('lr', self.scheduler.get_last_lr()[0], step=self.step)
            self.monitor.log('grad_norm', grad_norm.item(), step=self.step)
        
        return loss.item()
    
    def save_checkpoint(self, path: str, inference_ready: bool = True):
        """Save checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'step': self.step,
        }
        
        # Extract EchoKill memory if present
        for name, module in self.model.named_modules():
            if hasattr(module, 'long_term_memory'):
                checkpoint['echo_kill_memory'] = {
                    name: module.long_term_memory.clone()
                }
        
        if inference_ready:
            # Save in RyanFormat
            try:
                from .format import RyanFormat
                
                inference_ckpt = CheckpointConverter.training_to_inference(
                    checkpoint, self.config
                )
                
                # Create temporary model with inference state
                temp_model = copy.deepcopy(self.model)
                temp_model.load_state_dict(inference_ckpt['model_state_dict'])
                
                RyanFormat.save(
                    temp_model,
                    path,
                    config=inference_ckpt['config'],
                    use_quantization=self.config.use_quantization,
                    use_huffman=self.config.use_huffman,
                )
                
                del temp_model
                print(f"[RyanPipeline] Saved inference-ready checkpoint: {path}")
                
            except ImportError:
                torch.save(checkpoint, path)
                print(f"[RyanPipeline] Saved training checkpoint: {path}")
        else:
            torch.save(checkpoint, path)
            print(f"[RyanPipeline] Saved training checkpoint: {path}")
    
    def setup_inference(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
    ):
        """Initialize inference components."""
        self.mode = 'inference'
        
        # Load model
        if checkpoint_path:
            try:
                from .format import RyanFormat
                state_dict, config = RyanFormat.load(checkpoint_path)
                
                if model:
                    model.load_state_dict(state_dict)
                    self.model = model
                else:
                    # Need model architecture
                    raise ValueError("Must provide model for RyanFormat loading")
                
                self.config = RyanConfig.from_dict(config)
                
            except ImportError:
                checkpoint = torch.load(checkpoint_path)
                if model:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    self.model = model
                self.config = RyanConfig.from_dict(checkpoint.get('config', {}))
        elif model:
            self.model = model
        
        # Setup inference engine
        try:
            from .scheduler import RyanStreamEngine
            from .speculative import SpeculativeEngine
            
            self.inference_engine = RyanStreamEngine(
                model=self.model,
                max_batch_size=self.config.max_batch_size,
                lookahead_tokens=self.config.lookahead_tokens,
            )
            
            if self.config.use_speculation:
                # Load draft model if specified
                draft_model = None
                if self.config.draft_model_path:
                    # Would need to load draft model here
                    pass
                
                self.speculative_engine = SpeculativeEngine(
                    target_model=self.model,
                    tokenizer=None,  # Set by user
                    draft_model=draft_model,
                    mode=self.config.speculation_mode,
                    max_speculation_length=self.config.max_speculation_length,
                )
            
            print("[RyanPipeline] Inference setup complete")
            
        except ImportError:
            print("[RyanPipeline] RyanStream not available, using basic inference")
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        use_speculation: bool = None,
    ) -> torch.Tensor:
        """Generate tokens."""
        assert self.mode == 'inference', "Must call setup_inference first"
        
        use_spec = use_speculation if use_speculation is not None else self.config.use_speculation
        
        if use_spec and hasattr(self, 'speculative_engine'):
            return self.speculative_engine.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
            )
        else:
            # Basic autoregressive
            return self._basic_generate(input_ids, max_new_tokens)
    
    def _basic_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """Basic autoregressive generation."""
        self.model.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                next_token = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# =============================================================================
# DISTILLATION PIPELINE
# =============================================================================

class DistillationPipeline:
    """
    Distill large model into small draft model for speculation.
    
    Usage:
        distiller = DistillationPipeline(teacher=big_model, student=small_model)
        distiller.distill(dataloader, num_steps=10000)
        distiller.save_student("draft_model.ryan")
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5,  # Balance between soft and hard targets
    ):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=5e-5)
    
    def distill_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """One distillation step."""
        self.student.train()
        
        input_ids = batch['input_ids']
        labels = batch.get('labels', input_ids[:, 1:])
        
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids)
            teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs
        
        # Student forward
        student_outputs = self.student(input_ids)
        student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs
        
        # Soft targets (KL divergence)
        soft_targets = torch.nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
        soft_preds = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=-1)
        soft_loss = torch.nn.functional.kl_div(
            soft_preds, soft_targets, reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard targets (cross entropy)
        hard_loss = torch.nn.functional.cross_entropy(
            student_logits[:, :-1].reshape(-1, student_logits.size(-1)),
            labels.reshape(-1),
        )
        
        # Combined loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def distill(
        self,
        dataloader,
        num_steps: int = 10000,
        log_interval: int = 100,
    ):
        """Run full distillation."""
        step = 0
        
        while step < num_steps:
            for batch in dataloader:
                loss = self.distill_step(batch)
                step += 1
                
                if step % log_interval == 0:
                    print(f"[Distillation] Step {step}/{num_steps}, Loss: {loss:.4f}")
                
                if step >= num_steps:
                    break
        
        print(f"[Distillation] Complete. Final loss: {loss:.4f}")
    
    def save_student(self, path: str):
        """Save distilled student model."""
        try:
            from .format import RyanFormat
            RyanFormat.save(self.student, path, config={'distilled': True})
        except ImportError:
            torch.save(self.student.state_dict(), path)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'RyanConfig',
    'CheckpointConverter',
    'RyanPipeline',
    'DistillationPipeline',
]


if __name__ == "__main__":
    print("Ryan-Bridge 1.0")
    print("===============")
    print("Training → Inference Pipeline:")
    print("  1. RyanConfig: Unified configuration")
    print("  2. CheckpointConverter: Training ↔ Inference conversion")
    print("  3. RyanPipeline: End-to-end orchestration")
    print("  4. DistillationPipeline: Big → Draft model")
