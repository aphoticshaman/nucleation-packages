"""
Contextual Multi-Fold Compression (CMFC)
=========================================

NI-047: Query-Determined Decompression Pathway Selection

Core insight: Store information in multi-stable "folded" representations.
The query context at retrieval time—not stored metadata—determines which
unfolding pathway is executed. Same bits → different outputs based on context.

Analogous to protein folding thermodynamics:
- Multiple stable conformations exist in the energy landscape
- Environmental conditions (pH, temperature, binding partners) select the fold
- Here: query context plays the role of environment, selecting the decode path

Key differentiators from prior art:
- VAEs: Single decode pathway. CMFC has N stable states per representation.
- RAG: Query selects documents. CMFC selects pathways WITHIN a single doc.
- Compression (gzip): Deterministic decode. CMFC decode is query-conditioned.
- AlphaFold: Predicts structure. CMFC uses folding as STORAGE mechanism.

Applications:
- Database compression with user-context-adapted retrieval
- LLM prompt caching (fold prompts, unfold per downstream query)
- ARC Prize: Fold training examples into latent, unfold per test task
- Personalization: Same content, different decompression per user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import math


class PathwaySelectionMode(Enum):
    """How to select the unfolding pathway from energy landscape."""
    SOFT = "soft"          # Weighted combination via softmax
    HARD = "hard"          # Argmin selection (discrete)
    GUMBEL = "gumbel"      # Gumbel-softmax for differentiable discrete
    TOPK = "topk"          # Top-k pathway ensemble
    ANNEALED = "annealed"  # Temperature-scheduled (hot→cold)


@dataclass
class FoldedRepresentation:
    """Multi-stable folded representation with metadata."""
    states: torch.Tensor          # [batch, n_states, d_model]
    n_stable_states: int
    fold_energy: torch.Tensor     # [batch] - energy of the folding
    source_length: int            # Original sequence length

    def to(self, device):
        return FoldedRepresentation(
            states=self.states.to(device),
            n_stable_states=self.n_stable_states,
            fold_energy=self.fold_energy.to(device),
            source_length=self.source_length
        )


@dataclass
class UnfoldResult:
    """Result of query-conditioned unfolding."""
    output: torch.Tensor              # Decoded output
    pathway_weights: torch.Tensor     # Which pathways were selected
    pathway_energies: torch.Tensor    # Energy of each pathway for this query
    selected_state: torch.Tensor      # The intermediate unfolded state
    confidence: float                 # How decisive the pathway selection was


class EnergyFunction(nn.Module):
    """
    Maps (folded_representation, query_context) → pathway energies.

    Lower energy = more favorable pathway for this query.
    This is the learned "thermodynamic" function that determines
    which stable state the query "prefers."
    """

    def __init__(
        self,
        d_model: int = 512,
        n_stable_states: int = 8,
        n_heads: int = 8,
        use_cross_attention: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_stable_states = n_stable_states
        self.use_cross_attention = use_cross_attention

        if use_cross_attention:
            # Query attends to each stable state to compute compatibility
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                batch_first=True
            )
            self.energy_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1)
            )
        else:
            # Simpler: project query, dot product with each state
            self.query_proj = nn.Linear(d_model, d_model)
            self.state_proj = nn.Linear(d_model, d_model)
            self.energy_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        folded: torch.Tensor,      # [batch, n_states, d_model]
        query: torch.Tensor        # [batch, d_model] or [batch, seq, d_model]
    ) -> torch.Tensor:             # [batch, n_states] - energies

        batch_size = folded.size(0)

        # Ensure query is 2D for pooling
        if query.dim() == 3:
            query = query.mean(dim=1)  # Pool over sequence

        if self.use_cross_attention:
            # Query (as single token) attends to all stable states
            query_expanded = query.unsqueeze(1)  # [batch, 1, d_model]

            # Cross-attention: query attends to states
            attended, _ = self.cross_attention(
                query_expanded, folded, folded
            )  # [batch, 1, d_model]

            # But we want per-state energies, so we do it differently:
            # Compute energy for each state given the query
            energies = []
            for i in range(self.n_stable_states):
                state_i = folded[:, i:i+1, :]  # [batch, 1, d_model]
                # Concatenate query and state, compute energy
                combined = query_expanded + state_i  # Simple combination
                energy_i = self.energy_head(combined).squeeze(-1).squeeze(-1)
                energies.append(energy_i)

            energies = torch.stack(energies, dim=1)  # [batch, n_states]
        else:
            # Dot-product energy
            q = self.query_proj(query)  # [batch, d_model]
            s = self.state_proj(folded)  # [batch, n_states, d_model]

            # Energy = negative similarity (lower = better match)
            energies = -torch.einsum('bd,bnd->bn', q, s) * self.energy_scale

        return energies


class CMFCEncoder(nn.Module):
    """
    Folds input data into multi-stable latent representation.

    The encoder creates N stable states from the input. These states
    are NOT meant to be identical—each represents a different valid
    "reading" or "interpretation" of the input that can be selected
    at decode time based on query context.

    Think of it as: the input contains multiple latent aspects, and
    the encoder separates these into distinct stable configurations.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_stable_states: int = 8,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.d_model = d_model
        self.n_stable_states = n_stable_states

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # State projector: creates multiple stable configurations
        # Each state is a different "reading" of the encoded input
        self.state_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )
            for _ in range(n_stable_states)
        ])

        # Stability regularizer: encourages states to be distinct
        self.stability_temp = nn.Parameter(torch.tensor(1.0))

        # Energy estimator for the folding (how "good" is this fold)
        self.fold_energy_head = nn.Sequential(
            nn.Linear(d_model * n_stable_states, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> FoldedRepresentation:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            FoldedRepresentation containing multi-stable states
        """
        batch_size, seq_len, _ = x.shape

        # Encode input
        encoded = self.encoder(x, src_key_padding_mask=mask)  # [batch, seq, d_model]

        # Pool to fixed-size representation
        if mask is not None:
            # Masked mean pooling
            mask_expanded = (~mask).unsqueeze(-1).float()
            pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)  # [batch, d_model]

        # Project to multiple stable states
        states = []
        for projector in self.state_projectors:
            state = projector(pooled)  # [batch, d_model]
            states.append(state)

        folded_states = torch.stack(states, dim=1)  # [batch, n_states, d_model]

        # Compute fold energy (how stable is this folded representation)
        flat_states = folded_states.view(batch_size, -1)
        fold_energy = self.fold_energy_head(flat_states).squeeze(-1)

        return FoldedRepresentation(
            states=folded_states,
            n_stable_states=self.n_stable_states,
            fold_energy=fold_energy,
            source_length=seq_len
        )

    def state_diversity_loss(self, folded: FoldedRepresentation) -> torch.Tensor:
        """
        Regularization loss encouraging stable states to be distinct.
        If states are too similar, the multi-fold compression is wasted.
        """
        states = folded.states  # [batch, n_states, d_model]

        # Compute pairwise cosine similarity between states
        states_norm = F.normalize(states, dim=-1)
        similarity = torch.bmm(states_norm, states_norm.transpose(1, 2))

        # We want off-diagonal elements to be low (distinct states)
        # Create mask for off-diagonal
        n = self.n_stable_states
        off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=states.device)

        off_diag_sim = similarity[:, off_diag_mask].view(states.size(0), -1)

        # Loss: penalize high similarity between different states
        diversity_loss = off_diag_sim.pow(2).mean()

        return diversity_loss


class CMFCDecoder(nn.Module):
    """
    Unfolds representation based on query context.

    The decoder receives:
    1. A folded representation with N stable states
    2. A query context that determines which pathway to unfold along

    The energy function computes compatibility between the query and
    each stable state. The minimum-energy pathway is selected (or
    a soft combination is used).
    """

    def __init__(
        self,
        d_model: int = 512,
        n_stable_states: int = 8,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 2048,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        selection_mode: PathwaySelectionMode = PathwaySelectionMode.SOFT
    ):
        super().__init__()
        self.d_model = d_model
        self.n_stable_states = n_stable_states
        self.selection_mode = selection_mode
        self.output_dim = output_dim or d_model

        # Energy function for pathway selection
        self.energy_fn = EnergyFunction(
            d_model=d_model,
            n_stable_states=n_stable_states,
            n_heads=n_heads
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, self.output_dim)

        # Temperature for Gumbel-softmax
        self.gumbel_temp = nn.Parameter(torch.tensor(1.0))

        # For annealed selection
        self._anneal_step = 0

    def select_pathway(
        self,
        energies: torch.Tensor,
        folded_states: torch.Tensor,
        mode: Optional[PathwaySelectionMode] = None,
        topk: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select unfolding pathway based on energy landscape.

        Args:
            energies: [batch, n_states] - energy of each pathway
            folded_states: [batch, n_states, d_model]
            mode: Selection mode (defaults to self.selection_mode)
            topk: Number of pathways for TOPK mode

        Returns:
            selected_state: [batch, d_model]
            weights: [batch, n_states] - pathway weights used
        """
        mode = mode or self.selection_mode

        if mode == PathwaySelectionMode.SOFT:
            # Soft selection: weighted combination
            weights = F.softmax(-energies, dim=-1)  # Lower energy = higher weight
            selected = torch.einsum('bs,bsd->bd', weights, folded_states)

        elif mode == PathwaySelectionMode.HARD:
            # Hard selection: argmin
            indices = energies.argmin(dim=-1)  # [batch]
            weights = F.one_hot(indices, num_classes=self.n_stable_states).float()
            selected = folded_states[torch.arange(folded_states.size(0)), indices]

        elif mode == PathwaySelectionMode.GUMBEL:
            # Gumbel-softmax for differentiable discrete selection
            logits = -energies  # Convert energy to logits
            weights = F.gumbel_softmax(logits, tau=self.gumbel_temp, hard=True)
            selected = torch.einsum('bs,bsd->bd', weights, folded_states)

        elif mode == PathwaySelectionMode.TOPK:
            # Top-k pathway ensemble
            _, top_indices = energies.topk(topk, dim=-1, largest=False)
            # Get weights for top-k only
            top_energies = energies.gather(1, top_indices)
            top_weights = F.softmax(-top_energies, dim=-1)
            # Scatter back to full weight tensor
            weights = torch.zeros_like(energies)
            weights.scatter_(1, top_indices, top_weights)
            selected = torch.einsum('bs,bsd->bd', weights, folded_states)

        elif mode == PathwaySelectionMode.ANNEALED:
            # Temperature annealing: starts soft, becomes hard
            temp = max(0.1, 2.0 - self._anneal_step * 0.01)
            weights = F.softmax(-energies / temp, dim=-1)
            selected = torch.einsum('bs,bsd->bd', weights, folded_states)
            self._anneal_step += 1

        else:
            raise ValueError(f"Unknown selection mode: {mode}")

        return selected, weights

    def forward(
        self,
        folded: FoldedRepresentation,
        query_context: torch.Tensor,
        output_length: Optional[int] = None,
        selection_mode: Optional[PathwaySelectionMode] = None
    ) -> UnfoldResult:
        """
        Unfold representation conditioned on query context.

        Args:
            folded: Multi-stable folded representation
            query_context: [batch, d_model] or [batch, seq, d_model]
            output_length: Desired output sequence length
            selection_mode: Override default selection mode

        Returns:
            UnfoldResult with decoded output and pathway information
        """
        folded_states = folded.states  # [batch, n_states, d_model]
        batch_size = folded_states.size(0)

        # Compute energy landscape for this query
        energies = self.energy_fn(folded_states, query_context)

        # Select pathway
        selected_state, weights = self.select_pathway(
            energies, folded_states, selection_mode
        )

        # Compute confidence (how decisive was the selection)
        weight_entropy = -(weights * (weights + 1e-10).log()).sum(dim=-1)
        max_entropy = math.log(self.n_stable_states)
        confidence = 1.0 - (weight_entropy.mean().item() / max_entropy)

        # Decode: expand selected state to sequence
        output_len = output_length or folded.source_length

        # Use selected state as memory for decoder
        memory = selected_state.unsqueeze(1)  # [batch, 1, d_model]

        # Auto-regressive or parallel decoding
        # For simplicity, use parallel with positional queries
        pos_queries = self._make_position_queries(batch_size, output_len, selected_state.device)

        decoded = self.decoder(pos_queries, memory)
        output = self.output_proj(decoded)

        return UnfoldResult(
            output=output,
            pathway_weights=weights,
            pathway_energies=energies,
            selected_state=selected_state,
            confidence=confidence
        )

    def _make_position_queries(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create positional query embeddings for parallel decoding."""
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Sinusoidal positional encoding
        d_model = self.d_model
        pe = torch.zeros(seq_len, d_model, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) *
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0).expand(batch_size, -1, -1)


class HierarchicalCMFC(nn.Module):
    """
    Hierarchical multi-fold compression with cascaded unfolding.

    Implements "not just once" insight: multiple sequential unfold
    operations, each query-conditioned, producing hierarchical
    decompression. Each level can have different query context.

    Level 0: Coarse structure (e.g., topic selection)
    Level 1: Medium structure (e.g., subtopic/approach)
    Level 2: Fine structure (e.g., specific details)
    """

    def __init__(
        self,
        d_model: int = 512,
        n_levels: int = 3,
        states_per_level: List[int] = [4, 8, 16],
        n_layers_per_level: int = 2,
        n_heads: int = 8
    ):
        super().__init__()
        self.n_levels = n_levels
        self.states_per_level = states_per_level

        assert len(states_per_level) == n_levels

        # Encoder for initial folding
        self.initial_encoder = CMFCEncoder(
            d_model=d_model,
            n_stable_states=states_per_level[0],
            n_layers=4,
            n_heads=n_heads
        )

        # Decoder for each level
        self.level_decoders = nn.ModuleList()
        for level in range(n_levels):
            is_last = (level == n_levels - 1)
            decoder = CMFCDecoder(
                d_model=d_model,
                n_stable_states=states_per_level[level],
                n_layers=n_layers_per_level,
                n_heads=n_heads,
                output_dim=d_model if not is_last else d_model,
                selection_mode=PathwaySelectionMode.SOFT
            )
            self.level_decoders.append(decoder)

        # Re-folders for intermediate levels (fold output for next level)
        self.level_refolders = nn.ModuleList()
        for level in range(n_levels - 1):
            refolder = nn.Sequential(
                nn.Linear(d_model, d_model * states_per_level[level + 1]),
                nn.GELU(),
            )
            self.level_refolders.append(refolder)

    def forward(
        self,
        x: torch.Tensor,
        query_contexts: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Hierarchical fold and cascaded unfold.

        Args:
            x: Input [batch, seq, d_model]
            query_contexts: List of query tensors, one per level
            mask: Optional padding mask

        Returns:
            Dict with outputs from each level and final output
        """
        assert len(query_contexts) == self.n_levels

        results = {
            'level_outputs': [],
            'level_weights': [],
            'level_confidences': []
        }

        # Initial fold
        folded = self.initial_encoder(x, mask)

        for level in range(self.n_levels):
            # Unfold at this level
            unfold_result = self.level_decoders[level](
                folded,
                query_contexts[level]
            )

            results['level_outputs'].append(unfold_result.output)
            results['level_weights'].append(unfold_result.pathway_weights)
            results['level_confidences'].append(unfold_result.confidence)

            # Re-fold for next level (if not last)
            if level < self.n_levels - 1:
                # Take the selected state and create new stable states
                refold_proj = self.level_refolders[level](unfold_result.selected_state)
                new_states = refold_proj.view(
                    -1, self.states_per_level[level + 1], self.d_model
                )
                folded = FoldedRepresentation(
                    states=new_states,
                    n_stable_states=self.states_per_level[level + 1],
                    fold_energy=folded.fold_energy,  # Carry forward
                    source_length=folded.source_length
                )

        results['final_output'] = results['level_outputs'][-1]

        return results


class CMFCForARC(nn.Module):
    """
    CMFC adapted for ARC Prize challenge.

    Strategy: Fold all training examples into a shared latent space.
    At test time, use the test input as query context to unfold
    the relevant transformation pattern.

    The hypothesis: ARC tasks share abstract transformation primitives
    that can be "folded" into stable states. The test input selects
    which primitive (or combination) to apply.
    """

    def __init__(
        self,
        grid_size: int = 30,
        n_colors: int = 10,
        d_model: int = 256,
        n_stable_states: int = 32,  # More states for diverse transforms
        n_layers: int = 6
    ):
        super().__init__()
        self.grid_size = grid_size
        self.n_colors = n_colors
        self.d_model = d_model

        # Grid encoder: (batch, H, W, colors) -> (batch, seq, d_model)
        self.grid_encoder = nn.Sequential(
            nn.Embedding(n_colors, d_model // 4),
            nn.Flatten(start_dim=2),  # Flatten spatial and embedding
            nn.Linear(grid_size * grid_size * (d_model // 4), d_model)
        )

        # Encoder for folding training examples
        self.example_encoder = CMFCEncoder(
            d_model=d_model,
            n_stable_states=n_stable_states,
            n_layers=n_layers
        )

        # Decoder for unfolding based on test input
        self.transform_decoder = CMFCDecoder(
            d_model=d_model,
            n_stable_states=n_stable_states,
            n_layers=n_layers,
            output_dim=grid_size * grid_size * n_colors,
            selection_mode=PathwaySelectionMode.SOFT
        )

        # Aggregator for multiple training examples
        self.example_aggregator = nn.MultiheadAttention(
            embed_dim=d_model * n_stable_states,
            num_heads=8,
            batch_first=True
        )

    def encode_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """Encode a grid to d_model representation."""
        # grid: [batch, H, W] with values 0-9
        embedded = self.grid_encoder[0](grid)  # [batch, H, W, d_model//4]
        flat = embedded.view(grid.size(0), -1)  # [batch, H*W*(d_model//4)]
        return self.grid_encoder[2](flat)  # [batch, d_model]

    def fold_training_examples(
        self,
        input_grids: List[torch.Tensor],
        output_grids: List[torch.Tensor]
    ) -> FoldedRepresentation:
        """
        Fold all training input-output pairs into multi-stable representation.

        The key insight: we encode the TRANSFORMATION (input→output),
        not just the grids themselves.
        """
        example_encodings = []

        for inp, out in zip(input_grids, output_grids):
            inp_enc = self.encode_grid(inp)
            out_enc = self.encode_grid(out)
            # Encode the transformation as the difference/combination
            transform_enc = torch.cat([inp_enc, out_enc, out_enc - inp_enc], dim=-1)
            example_encodings.append(transform_enc)

        # Stack examples as sequence
        examples_seq = torch.stack(example_encodings, dim=1)  # [batch, n_examples, d*3]

        # Project to d_model
        examples_proj = nn.Linear(self.d_model * 3, self.d_model).to(examples_seq.device)(examples_seq)

        # Fold into multi-stable representation
        folded = self.example_encoder(examples_proj)

        return folded

    def solve(
        self,
        training_inputs: List[torch.Tensor],
        training_outputs: List[torch.Tensor],
        test_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Solve ARC task: use training examples to inform test output.

        1. Fold training examples into multi-stable transform space
        2. Use test input as query to select relevant transform
        3. Apply transform to produce output
        """
        # Fold training examples
        folded_transforms = self.fold_training_examples(
            training_inputs, training_outputs
        )

        # Encode test input as query context
        test_query = self.encode_grid(test_input)

        # Unfold: select and apply transform
        result = self.transform_decoder(folded_transforms, test_query)

        # Reshape output to grid
        output_logits = result.output.view(
            -1, self.grid_size, self.grid_size, self.n_colors
        )

        # Return argmax prediction
        return output_logits.argmax(dim=-1)


# Training utilities

def contrastive_cmfc_loss(
    encoder: CMFCEncoder,
    decoder: CMFCDecoder,
    data: torch.Tensor,
    queries: torch.Tensor,
    targets: torch.Tensor,
    negative_queries: Optional[torch.Tensor] = None,
    margin: float = 0.5
) -> torch.Tensor:
    """
    Contrastive training loss for CMFC.

    Trains the energy function to:
    - Assign low energy to correct (query, target) pairs
    - Assign high energy to incorrect pairs

    This teaches the model which pathway to select for each query type.
    """
    # Fold input
    folded = encoder(data)

    # Compute positive energies
    pos_result = decoder(folded, queries)

    # Reconstruction loss
    recon_loss = F.mse_loss(pos_result.output, targets)

    # Contrastive loss on pathway energies
    if negative_queries is not None:
        neg_result = decoder(folded, negative_queries)

        # Positive should have lower energy than negative
        pos_energies = pos_result.pathway_energies.min(dim=-1).values
        neg_energies = neg_result.pathway_energies.min(dim=-1).values

        contrastive = F.relu(margin + pos_energies - neg_energies).mean()
    else:
        contrastive = torch.tensor(0.0)

    # Diversity regularization
    diversity_loss = encoder.state_diversity_loss(folded)

    return recon_loss + 0.1 * contrastive + 0.01 * diversity_loss


if __name__ == "__main__":
    # Demo: Basic CMFC encode-decode cycle

    d_model = 256
    n_states = 8
    batch_size = 4
    seq_len = 32

    # Create encoder and decoder
    encoder = CMFCEncoder(d_model=d_model, n_stable_states=n_states)
    decoder = CMFCDecoder(d_model=d_model, n_stable_states=n_states)

    # Sample input
    x = torch.randn(batch_size, seq_len, d_model)

    # Fold
    folded = encoder(x)
    print(f"Folded representation: {folded.states.shape}")
    print(f"  {n_states} stable states, each {d_model}-dimensional")
    print(f"  Fold energy: {folded.fold_energy}")

    # Different queries lead to different unfolding
    query_a = torch.randn(batch_size, d_model)
    query_b = torch.randn(batch_size, d_model)

    result_a = decoder(folded, query_a)
    result_b = decoder(folded, query_b)

    print(f"\nQuery A pathway weights: {result_a.pathway_weights[0].tolist()}")
    print(f"Query A confidence: {result_a.confidence:.3f}")

    print(f"\nQuery B pathway weights: {result_b.pathway_weights[0].tolist()}")
    print(f"Query B confidence: {result_b.confidence:.3f}")

    # Outputs differ despite same folded representation
    output_diff = (result_a.output - result_b.output).abs().mean()
    print(f"\nOutput difference: {output_diff:.4f}")
    print("(Same storage, different outputs based on query context)")
