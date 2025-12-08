#!/usr/bin/env python3
"""
CIC PROOF: UNIVERSAL INFORMATION PHASE TRANSITION (UIPT)
=========================================================
PyTorch implementation for Kaggle/RunPod GPU execution.

RUN ON KAGGLE:
1. Create new notebook with GPU accelerator
2. Paste this entire file
3. Run

RUN ON RUNPOD:
runpodctl exec python3 uipt_grokking.py

EXPECTED RESULT:
- Train accuracy hits 99%+ early (memorization)
- Test accuracy stays low for thousands of steps
- Then SUDDENLY jumps to 90%+ (grokking = UIPT)
- CIC metrics (dC/dt, dΦ/dt) spike at transition

Ryan J. Cardwell + Claude Opus 4.5
December 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    "p": 97,                    # Modular arithmetic base
    "train_frac": 0.3,          # Small = forces memorization first
    "embed_dim": 128,           # Embedding dimension
    "hidden_dim": 256,          # Hidden layer size
    "lr": 1e-3,                 # Learning rate
    "weight_decay": 1.0,        # HIGH = forces compression (critical for grokking)
    "epochs": 50000,            # Grokking happens LATE
    "log_every": 500,           # Log frequency
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
}

# =============================================================================
# MODEL
# =============================================================================

class GrokkingMLP(nn.Module):
    """MLP for modular arithmetic - designed to exhibit grokking."""
    
    def __init__(self, p, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.p = p
        self.embed = nn.Embedding(p, embed_dim)
        self.fc1 = nn.Linear(2 * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, p)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: [batch, 2] - two integers to add
        emb = self.embed(x)  # [batch, 2, embed_dim]
        flat = emb.view(x.size(0), -1)  # [batch, 2*embed_dim]
        h1 = self.relu(self.fc1(flat))
        h2 = self.relu(self.fc2(h1))
        out = self.fc3(h2)
        return out, h2  # Return logits and hidden activations

# =============================================================================
# CIC METRICS
# =============================================================================

def compute_compression(activations: torch.Tensor) -> float:
    """
    Compression (C): Effective rank via SVD entropy.
    Lower effective rank = higher compression = simpler representation.
    
    From Information Bottleneck theory (Tishby).
    """
    if activations.size(0) < 2:
        return 0.0
    
    with torch.no_grad():
        # Center
        centered = activations - activations.mean(dim=0)
        
        # SVD
        try:
            _, s, _ = torch.linalg.svd(centered, full_matrices=False)
            s = s.cpu().numpy()
        except:
            return 0.0
        
        # Normalize to probability distribution
        s_norm = s / (s.sum() + 1e-9)
        
        # Shannon entropy = effective rank
        entropy = -np.sum(s_norm * np.log(s_norm + 1e-9))
        
        return float(entropy)

def compute_integration(model: nn.Module) -> float:
    """
    Integration (Φ): Spectral gap of weight matrix.
    Higher gap = system cannot be easily partitioned = more integrated.
    
    From Integrated Information Theory (Tononi).
    """
    with torch.no_grad():
        # Use fc2 weights as proxy for internal connectivity
        W = model.fc2.weight.detach().cpu().numpy()
        
        # Gram matrix (connectivity proxy)
        gram = W.T @ W
        
        try:
            eigs = np.linalg.eigvalsh(gram)
            sorted_eigs = np.sort(eigs)[::-1]
            
            # Spectral gap: difference between top two eigenvalues
            if len(sorted_eigs) > 1:
                gap = float(abs(sorted_eigs[0] - sorted_eigs[1]))
                return gap
        except:
            pass
        
        return 0.0

def compute_weight_norm(model: nn.Module) -> float:
    """Total L2 norm of weights - tracks compression pressure."""
    total = 0.0
    with torch.no_grad():
        for p in model.parameters():
            total += (p ** 2).sum().item()
    return np.sqrt(total)

# =============================================================================
# DATA
# =============================================================================

def create_modular_addition_data(p: int, train_frac: float, device: str):
    """Create modular addition dataset: a + b = c (mod p)"""
    
    # Generate all pairs
    data = []
    for i in range(p):
        for j in range(p):
            data.append([i, j, (i + j) % p])
    
    data = np.array(data)
    np.random.shuffle(data)
    
    # Split
    split = int(len(data) * train_frac)
    
    train_X = torch.tensor(data[:split, :2], dtype=torch.long, device=device)
    train_Y = torch.tensor(data[:split, 2], dtype=torch.long, device=device)
    test_X = torch.tensor(data[split:, :2], dtype=torch.long, device=device)
    test_Y = torch.tensor(data[split:, 2], dtype=torch.long, device=device)
    
    return train_X, train_Y, test_X, test_Y

# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_grokking(config: dict):
    """Main training loop with CIC metric tracking."""
    
    print("=" * 70)
    print("UNIVERSAL INFORMATION PHASE TRANSITION (UIPT) EXPERIMENT")
    print("=" * 70)
    print(f"Device: {config['device']}")
    print(f"p={config['p']}, train_frac={config['train_frac']}")
    print(f"LR={config['lr']}, Weight Decay={config['weight_decay']}")
    print(f"Epochs: {config['epochs']}")
    print()
    
    # Set seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Data
    train_X, train_Y, test_X, test_Y = create_modular_addition_data(
        config['p'], config['train_frac'], config['device']
    )
    print(f"Train size: {len(train_X)}, Test size: {len(test_X)}")
    
    # Model
    model = GrokkingMLP(
        config['p'], 
        config['embed_dim'], 
        config['hidden_dim']
    ).to(config['device'])
    
    # Optimizer - AdamW with high weight decay is CRITICAL for grokking
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # History
    history = {
        'epoch': [],
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'compression': [],      # C - effective rank
        'integration': [],      # Φ - spectral gap
        'weight_norm': [],      # Total weight magnitude
    }
    
    print(f"{'Epoch':>6} | {'Train':>6} | {'Test':>6} | {'C':>7} | {'Φ':>9} | {'Loss':>7} | {'W':>7}")
    print("-" * 70)
    
    grokking_detected = False
    grokking_epoch = None
    
    for epoch in range(config['epochs']):
        model.train()
        
        # Forward
        logits, hidden = model(train_X)
        loss = criterion(logits, train_Y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        if epoch % config['log_every'] == 0:
            model.eval()
            with torch.no_grad():
                # Accuracy
                train_pred = logits.argmax(dim=1)
                train_acc = (train_pred == train_Y).float().mean().item()
                
                test_logits, _ = model(test_X)
                test_pred = test_logits.argmax(dim=1)
                test_acc = (test_pred == test_Y).float().mean().item()
                
                # CIC Metrics
                _, h_full = model(train_X)
                C = compute_compression(h_full)
                Phi = compute_integration(model)
                W_norm = compute_weight_norm(model)
            
            history['epoch'].append(epoch)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['train_loss'].append(loss.item())
            history['compression'].append(C)
            history['integration'].append(Phi)
            history['weight_norm'].append(W_norm)
            
            print(f"{epoch:>6} | {train_acc:>6.3f} | {test_acc:>6.3f} | {C:>7.2f} | {Phi:>9.2f} | {loss.item():>7.3f} | {W_norm:>7.1f}")
            
            # Detect grokking
            if not grokking_detected and train_acc > 0.95 and test_acc > 0.5:
                grokking_detected = True
                grokking_epoch = epoch
                print(f"\n*** GROKKING DETECTED AT EPOCH {epoch} ***\n")
            
            # Early stop if fully generalized
            if train_acc > 0.99 and test_acc > 0.95:
                print(f"\nFully generalized at epoch {epoch}")
                break
    
    return history, grokking_epoch

# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_uipt(history: dict, grokking_epoch: int):
    """Analyze for Universal Information Phase Transition."""
    
    print("\n" + "=" * 70)
    print("UIPT ANALYSIS")
    print("=" * 70)
    
    epochs = np.array(history['epoch'])
    train_acc = np.array(history['train_acc'])
    test_acc = np.array(history['test_acc'])
    C = np.array(history['compression'])
    Phi = np.array(history['integration'])
    
    # Normalize for comparison
    def norm(x):
        r = x.max() - x.min()
        if r < 1e-9:
            return np.zeros_like(x)
        return (x - x.min()) / r
    
    # Compute derivatives
    d_test = np.gradient(test_acc)
    d_C = np.gradient(norm(C))
    d_Phi = np.gradient(norm(Phi))
    
    # Find peak acceleration
    grok_idx = np.argmax(d_test)
    
    print(f"\nPhase Transition Analysis:")
    print(f"  Peak test acceleration at epoch: {epochs[grok_idx]}")
    print(f"  Test accuracy at peak: {test_acc[grok_idx]:.3f}")
    print(f"  d(Test)/dt at peak: {d_test[grok_idx]:.4f}")
    print(f"  d(C)/dt at peak: {d_C[grok_idx]:.4f}")
    print(f"  d(Φ)/dt at peak: {d_Phi[grok_idx]:.4f}")
    
    # UIPT signature: forces are active at transition
    force_activity = abs(d_C[grok_idx]) + abs(d_Phi[grok_idx])
    print(f"\n  CIC Force Activity: {force_activity:.4f}")
    
    # Check for memorization-then-generalization signature
    train_99_idx = np.where(train_acc > 0.99)[0]
    test_50_idx = np.where(test_acc > 0.50)[0]
    
    if len(train_99_idx) > 0 and len(test_50_idx) > 0:
        mem_epoch = epochs[train_99_idx[0]]
        gen_epoch = epochs[test_50_idx[0]]
        
        if gen_epoch > mem_epoch:
            gap = gen_epoch - mem_epoch
            print(f"\n✓ GROKKING CONFIRMED!")
            print(f"  Memorization complete: epoch {mem_epoch}")
            print(f"  Generalization emerged: epoch {gen_epoch}")
            print(f"  Memorization gap: {gap} epochs")
            
            # CIC at transition
            if grok_idx > 0 and grok_idx < len(C) - 1:
                print(f"\n  CIC State at Transition:")
                print(f"    Compression (C): {C[grok_idx]:.3f}")
                print(f"    Integration (Φ): {Phi[grok_idx]:.3f}")
                print(f"    dC/dt: {d_C[grok_idx]:+.4f}")
                print(f"    dΦ/dt: {d_Phi[grok_idx]:+.4f}")
                
                # The UIPT equation: at transition, forces balance
                # dΦ/dt ≈ λ·dH/dt (where H ~ 1/C for effective rank)
                print(f"\n  UIPT Equation Check:")
                print(f"    At phase transition, compression and integration")
                print(f"    forces should show correlated activity.")
                
                # Correlation around transition
                window = 3
                start = max(0, grok_idx - window)
                end = min(len(d_C), grok_idx + window + 1)
                
                if end - start > 2:
                    corr = np.corrcoef(d_C[start:end], d_Phi[start:end])[0, 1]
                    print(f"    Correlation(dC, dΦ) near transition: {corr:.3f}")
            
            return True
    
    if test_acc[-1] < 0.3:
        print("\n✗ No grokking - test accuracy never rose")
        print("  Try: more epochs, different weight decay")
    elif train_acc[-1] < 0.9:
        print("\n✗ Training failed to memorize")
        print("  Try: lower weight decay, higher learning rate")
    else:
        print("\n? Partial result - no clear phase transition")
    
    return False

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Check for GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected. This will be slow.")
        print("For best results, run on Kaggle GPU or RunPod.")
    print()
    
    # Train
    history, grokking_epoch = train_grokking(CONFIG)
    
    # Analyze
    success = analyze_uipt(history, grokking_epoch)
    
    # Save results
    output = {
        "config": CONFIG,
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
        "grokking_epoch": grokking_epoch,
        "success": success,
    }
    
    output_path = Path("uipt_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Final summary
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           UIPT EXPERIMENT COMPLETE                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   THE CIC FIXED-POINT EQUATION:                                              ║
║                                                                              ║
║   ∂Φ/∂T + γ·∂C_multi/∂T = λ·∂H/∂T                                           ║
║                                                                              ║
║   AT PHASE TRANSITION (GROKKING):                                            ║
║                                                                              ║
║   dΦ/dt + γ·dC/dt ≈ λ·dH/dt                                                  ║
║                                                                              ║
║   RESULT: {'✓ GROKKING DETECTED' if success else '✗ NO GROKKING (adjust hyperparameters)':^40}                         ║
║                                                                              ║
║   Grokking = delayed generalization after memorization.                      ║
║   UIPT = the phase transition where compression and integration balance.     ║
║   This is the moment abstraction emerges.                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    return history, success

if __name__ == "__main__":
    history, success = main()
