"""
LFBM Training Script for RunPod

Usage:
    python train.py --data training_data.jsonl --epochs 10 --batch_size 8

Hardware requirements:
    - Single H200: 141GB VRAM (massive overkill for 150M model)
    - Can also run on: A100, A10G, or even RTX 4090

Training time estimate:
    - 5000 examples, 10 epochs: ~30 minutes on H200
    - Cost: ~$1.25 at $2.50/hr
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.architecture import LFBM, LFBMConfig


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_epochs: int = 10
    batch_size: int = 8
    gradient_accumulation: int = 4
    max_grad_norm: float = 1.0
    save_every: int = 1000
    eval_every: int = 500
    fp16: bool = True  # Mixed precision


class BriefingDataset(Dataset):
    """Dataset for LFBM training"""

    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 1024):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Nation code to index mapping
        self.nation_to_idx = {}
        self.next_nation_idx = 0

        # Load data
        with open(data_path, 'r') as f:
            for line in f:
                ex = json.loads(line)
                self.examples.append(ex)

        print(f"Loaded {len(self.examples)} training examples")

    def _get_nation_idx(self, code: str) -> int:
        if code not in self.nation_to_idx:
            self.nation_to_idx[code] = self.next_nation_idx
            self.next_nation_idx += 1
        return self.nation_to_idx[code]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]

        # Process nations
        nations = ex.get('input_nations', [])[:20]  # Max 20 nations
        nation_codes = torch.zeros(20, dtype=torch.long)
        nation_risks = torch.zeros(20)
        nation_trends = torch.zeros(20)

        for i, nation in enumerate(nations):
            nation_codes[i] = self._get_nation_idx(nation.get('code', 'UNK'))
            nation_risks[i] = nation.get('risk', 0.5)
            nation_trends[i] = nation.get('trend', 0.0) + 0.5  # Normalize to [0, 1]

        # Process signals
        signals = ex.get('input_signals', {})
        signal_values = torch.tensor([
            signals.get('gdelt_count', 50) / 200,  # Normalize
            (signals.get('avg_tone', 0) + 10) / 20,  # Normalize [-10, 10] -> [0, 1]
            signals.get('alert_count', 5) / 50,
            0.5,  # Padding
        ], dtype=torch.float32)

        # Process categories
        categories = ex.get('input_categories', {})
        category_risks = torch.tensor([
            categories.get('political', 50) / 100,
            categories.get('economic', 50) / 100,
            categories.get('security', 50) / 100,
            categories.get('military', 50) / 100,
            categories.get('financial', 50) / 100,
            categories.get('cyber', 50) / 100,
            categories.get('health', 50) / 100,
            categories.get('scitech', 50) / 100,
            categories.get('energy', 50) / 100,
            categories.get('domestic', 50) / 100,
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # Padding to 26
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5,
        ], dtype=torch.float32)[:26]

        # Tokenize output
        output = ex.get('output_briefings', {})
        output_text = json.dumps(output, indent=None)
        tokens = self.tokenizer.encode(output_text, max_length=self.max_seq_len)

        target_tokens = torch.tensor(tokens, dtype=torch.long)

        # Pad to max length
        if len(target_tokens) < self.max_seq_len:
            padding = torch.zeros(self.max_seq_len - len(target_tokens), dtype=torch.long)
            target_tokens = torch.cat([target_tokens, padding])
        else:
            target_tokens = target_tokens[:self.max_seq_len]

        return {
            'nation_codes': nation_codes,
            'nation_risks': nation_risks,
            'nation_trends': nation_trends,
            'signal_values': signal_values,
            'category_risks': category_risks,
            'target_tokens': target_tokens,
            'quality_score': ex.get('quality_score', 1.0),
        }


class SimpleTokenizer:
    """
    Simple BPE-like tokenizer for intel domain.
    In production, would train a proper tokenizer on the corpus.
    """

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0

        # Build basic vocabulary
        self.token_to_id = {
            '<pad>': 0,
            '<bos>': 1,
            '<eos>': 2,
            '<unk>': 3,
        }

        # Add common intel terms
        intel_terms = [
            'risk', 'indicators', 'show', 'elevated', 'stable', 'moderate',
            'critical', 'monitoring', 'nations', 'metrics', 'assessment',
            'political', 'economic', 'security', 'military', 'financial',
            'cyber', 'energy', 'health', 'domestic', 'regional', 'global',
            'trend', 'increasing', 'decreasing', 'volatile', 'concerning',
            'threat', 'stability', 'conflict', 'tension', 'escalation',
            '{', '}', ':', '"', ',', '[', ']',  # JSON chars
        ]

        for term in intel_terms:
            if term not in self.token_to_id:
                self.token_to_id[term] = len(self.token_to_id)

        # Add character-level fallback
        for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?-_/()%':
            if c not in self.token_to_id:
                self.token_to_id[c] = len(self.token_to_id)

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, text: str, max_length: int = 1024) -> List[int]:
        """Encode text to token ids"""
        tokens = [self.bos_token_id]

        # Simple character-level with term matching
        i = 0
        while i < len(text) and len(tokens) < max_length - 1:
            # Try to match known terms
            matched = False
            for term in sorted(self.token_to_id.keys(), key=len, reverse=True):
                if len(term) > 1 and text[i:i+len(term)].lower() == term.lower():
                    tokens.append(self.token_to_id[term])
                    i += len(term)
                    matched = True
                    break

            if not matched:
                char = text[i]
                tokens.append(self.token_to_id.get(char, 3))  # 3 = <unk>
                i += 1

        tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids to text"""
        chars = []
        for tid in token_ids:
            if tid in [self.bos_token_id, self.eos_token_id, self.pad_token_id]:
                continue
            chars.append(self.id_to_token.get(tid, '?'))
        return ''.join(chars)


def train(
    model: LFBM,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: TrainingConfig,
    output_dir: str,
    device: torch.device,
):
    """Training loop"""

    model = model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    total_steps = len(train_loader) * config.max_epochs // config.gradient_accumulation
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.fp16 and device.type == 'cuda' else None

    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            nation_codes = batch['nation_codes'].to(device)
            nation_risks = batch['nation_risks'].to(device)
            nation_trends = batch['nation_trends'].to(device)
            signal_values = batch['signal_values'].to(device)
            category_risks = batch['category_risks'].to(device)
            target_tokens = batch['target_tokens'].to(device)
            quality_scores = batch['quality_score'].to(device)

            # Forward pass
            with torch.cuda.amp.autocast(enabled=config.fp16 and device.type == 'cuda'):
                logits = model(
                    nation_codes, nation_risks, nation_trends,
                    signal_values, category_risks, target_tokens[:, :-1]
                )

                # Compute loss with quality weighting
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_tokens[:, 1:].reshape(-1),
                    ignore_index=0,  # Ignore padding
                    reduction='none'
                )
                loss = loss.view(target_tokens.size(0), -1).mean(dim=1)
                loss = (loss * quality_scores).mean()
                loss = loss / config.gradient_accumulation

            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * config.gradient_accumulation
            num_batches += 1

            # Gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % 100 == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"Epoch {epoch+1}, Step {global_step}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

                # Save checkpoint
                if global_step % config.save_every == 0:
                    save_checkpoint(model, optimizer, scheduler, global_step, output_dir)

        # End of epoch
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")

        # Validation
        if val_loader:
            val_loss = evaluate(model, val_loader, device, config.fp16)
            print(f"Validation loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, global_step, output_dir, best=True)

    # Final save
    save_checkpoint(model, optimizer, scheduler, global_step, output_dir, final=True)
    print("Training complete!")


def evaluate(model: LFBM, val_loader: DataLoader, device: torch.device, fp16: bool) -> float:
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            nation_codes = batch['nation_codes'].to(device)
            nation_risks = batch['nation_risks'].to(device)
            nation_trends = batch['nation_trends'].to(device)
            signal_values = batch['signal_values'].to(device)
            category_risks = batch['category_risks'].to(device)
            target_tokens = batch['target_tokens'].to(device)

            with torch.cuda.amp.autocast(enabled=fp16 and device.type == 'cuda'):
                logits = model(
                    nation_codes, nation_risks, nation_trends,
                    signal_values, category_risks, target_tokens[:, :-1]
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_tokens[:, 1:].reshape(-1),
                    ignore_index=0
                )

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def save_checkpoint(model, optimizer, scheduler, step, output_dir, best=False, final=False):
    """Save model checkpoint"""
    os.makedirs(output_dir, exist_ok=True)

    if final:
        path = os.path.join(output_dir, 'lfbm_final.pt')
    elif best:
        path = os.path.join(output_dir, 'lfbm_best.pt')
    else:
        path = os.path.join(output_dir, f'lfbm_step_{step}.pt')

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'config': model.config,
    }, path)
    print(f"Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description='Train LFBM')
    parser.add_argument('--data', type=str, required=True, help='Training data JSONL')
    parser.add_argument('--val_data', type=str, help='Validation data JSONL')
    parser.add_argument('--output', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fp16', action='store_true', default=True)
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Tokenizer
    tokenizer = SimpleTokenizer()

    # Dataset
    train_dataset = BriefingDataset(args.data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = None
    if args.val_data:
        val_dataset = BriefingDataset(args.val_data, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    config = LFBMConfig()
    model = LFBM(config)

    # Training config
    train_config = TrainingConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=args.fp16,
    )

    # Train
    train(model, train_loader, val_loader, train_config, args.output, device)


if __name__ == '__main__':
    main()
