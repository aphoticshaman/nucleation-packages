"""
LatticeForge Briefing Model (LFBM)
Purpose-built encoder-decoder for metrics → intelligence prose

Design principles:
1. Structured input encoder (not text-based)
2. Domain-specific vocabulary (8K tokens)
3. Constrained decoding for JSON output
4. Tiny footprint (~150M params) for edge deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class LFBMConfig:
    """Model configuration"""
    # Encoder
    num_nations: int = 250  # ISO country codes + regions
    nation_embed_dim: int = 128
    metric_embed_dim: int = 64
    signal_embed_dim: int = 64
    encoder_hidden: int = 512
    encoder_layers: int = 4
    encoder_heads: int = 8

    # Decoder
    vocab_size: int = 8192  # Small domain-specific vocab
    decoder_hidden: int = 768
    decoder_layers: int = 12
    decoder_heads: int = 12
    max_seq_len: int = 2048

    # Training
    dropout: float = 0.1

    @property
    def total_params(self) -> int:
        """Estimate total parameters"""
        encoder = (
            self.num_nations * self.nation_embed_dim +  # Nation embeddings
            100 * self.metric_embed_dim +  # Metric type embeddings
            self.encoder_hidden * self.encoder_hidden * self.encoder_layers * 4  # Transformer
        )
        decoder = (
            self.vocab_size * self.decoder_hidden +  # Token embeddings
            self.max_seq_len * self.decoder_hidden +  # Position embeddings
            self.decoder_hidden * self.decoder_hidden * self.decoder_layers * 4  # Transformer
        )
        return encoder + decoder


class MetricEncoder(nn.Module):
    """
    Encodes structured metrics into dense representations.

    Input format:
    {
        "nations": [{"code": "USA", "risk": 0.3, "trend": 1}, ...],
        "signals": {"gdelt_count": 45, "avg_tone": -2.3, ...},
        "categories": {"political": 72, "economic": 45, ...}
    }
    """

    def __init__(self, config: LFBMConfig):
        super().__init__()
        self.config = config

        # Nation embeddings (learned geopolitical representations)
        self.nation_embed = nn.Embedding(config.num_nations, config.nation_embed_dim)

        # Metric type embeddings
        self.metric_types = nn.Embedding(100, config.metric_embed_dim)

        # Value encoder (continuous → discrete buckets → embedding)
        self.value_buckets = 64
        self.value_embed = nn.Embedding(self.value_buckets, config.metric_embed_dim)

        # Projection to encoder hidden dim
        self.nation_proj = nn.Linear(config.nation_embed_dim + config.metric_embed_dim * 2, config.encoder_hidden)
        self.signal_proj = nn.Linear(config.signal_embed_dim * 4, config.encoder_hidden)
        self.category_proj = nn.Linear(config.metric_embed_dim * 26, config.encoder_hidden)  # 26 categories

        # Cross-attention between nation/signal/category representations
        self.cross_attn = nn.MultiheadAttention(
            config.encoder_hidden,
            config.encoder_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_hidden,
            nhead=config.encoder_heads,
            dim_feedforward=config.encoder_hidden * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)

        # Output projection to decoder hidden
        self.out_proj = nn.Linear(config.encoder_hidden, config.decoder_hidden)

    def _bucketize(self, values: torch.Tensor) -> torch.Tensor:
        """Convert continuous values to bucket indices"""
        # Assume values in [0, 1] range, map to buckets
        clamped = torch.clamp(values, 0, 1)
        buckets = (clamped * (self.value_buckets - 1)).long()
        return buckets

    def forward(
        self,
        nation_codes: torch.Tensor,      # [batch, num_nations]
        nation_risks: torch.Tensor,       # [batch, num_nations]
        nation_trends: torch.Tensor,      # [batch, num_nations]
        signal_values: torch.Tensor,      # [batch, num_signals]
        category_risks: torch.Tensor,     # [batch, 26]
    ) -> torch.Tensor:
        batch_size = nation_codes.size(0)

        # Encode nations
        nation_emb = self.nation_embed(nation_codes)  # [batch, nations, 128]
        risk_emb = self.value_embed(self._bucketize(nation_risks))  # [batch, nations, 64]
        trend_emb = self.value_embed(self._bucketize(nation_trends))
        nation_repr = self.nation_proj(torch.cat([nation_emb, risk_emb, trend_emb], dim=-1))  # [batch, nations, 512]

        # Encode signals (aggregate into fixed representation)
        signal_buckets = self._bucketize(signal_values)
        signal_emb = self.value_embed(signal_buckets)  # [batch, signals, 64]
        signal_repr = self.signal_proj(signal_emb.view(batch_size, -1)).unsqueeze(1)  # [batch, 1, 512]

        # Encode category risks
        cat_buckets = self._bucketize(category_risks)
        cat_emb = self.value_embed(cat_buckets)  # [batch, 26, 64]
        cat_repr = self.category_proj(cat_emb.view(batch_size, -1)).unsqueeze(1)  # [batch, 1, 512]

        # Combine all representations
        combined = torch.cat([nation_repr, signal_repr, cat_repr], dim=1)  # [batch, nations+2, 512]

        # Self-attention
        encoded = self.encoder(combined)

        # Project to decoder hidden
        return self.out_proj(encoded)


class ProseDecoder(nn.Module):
    """
    Generates intelligence prose from encoded metrics.
    Uses constrained decoding for valid JSON output.
    """

    def __init__(self, config: LFBMConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.decoder_hidden)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.decoder_hidden)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_hidden,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_hidden * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)

        # Output head
        self.out_proj = nn.Linear(config.decoder_hidden, config.vocab_size)

        # Causal mask cache
        self.register_buffer("causal_mask", None)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask"""
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            self.causal_mask = mask
        return self.causal_mask[:seq_len, :seq_len]

    def forward(
        self,
        encoder_output: torch.Tensor,  # [batch, enc_seq, hidden]
        target_tokens: torch.Tensor,    # [batch, dec_seq]
    ) -> torch.Tensor:
        batch_size, seq_len = target_tokens.shape
        device = target_tokens.device

        # Embed tokens
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embed(target_tokens) + self.pos_embed(positions)

        # Causal mask
        causal_mask = self._get_causal_mask(seq_len, device)

        # Decode
        decoded = self.decoder(
            token_emb,
            encoder_output,
            tgt_mask=causal_mask
        )

        # Project to vocabulary
        logits = self.out_proj(decoded)
        return logits

    @torch.no_grad()
    def generate(
        self,
        encoder_output: torch.Tensor,
        start_token: int,
        max_length: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[int]:
        """Autoregressive generation with nucleus sampling"""
        device = encoder_output.device
        batch_size = encoder_output.size(0)

        # Start with BOS token
        tokens = torch.tensor([[start_token]], device=device).expand(batch_size, -1)

        for _ in range(max_length):
            # Get logits for next token
            logits = self.forward(encoder_output, tokens)[:, -1, :]  # [batch, vocab]

            # Temperature scaling
            logits = logits / temperature

            # Nucleus (top-p) sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

            # Stop at EOS (assume token 2)
            if next_token.item() == 2:
                break

        return tokens[0].tolist()


class LFBM(nn.Module):
    """
    LatticeForge Briefing Model
    End-to-end metrics → prose generation
    """

    def __init__(self, config: Optional[LFBMConfig] = None):
        super().__init__()
        self.config = config or LFBMConfig()

        self.encoder = MetricEncoder(self.config)
        self.decoder = ProseDecoder(self.config)

        # Initialize weights
        self.apply(self._init_weights)

        print(f"LFBM initialized with ~{self.count_parameters()/1e6:.1f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        nation_codes: torch.Tensor,
        nation_risks: torch.Tensor,
        nation_trends: torch.Tensor,
        signal_values: torch.Tensor,
        category_risks: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Training forward pass"""
        encoder_output = self.encoder(
            nation_codes, nation_risks, nation_trends,
            signal_values, category_risks
        )
        logits = self.decoder(encoder_output, target_tokens)
        return logits

    @torch.no_grad()
    def generate_briefing(
        self,
        nation_codes: torch.Tensor,
        nation_risks: torch.Tensor,
        nation_trends: torch.Tensor,
        signal_values: torch.Tensor,
        category_risks: torch.Tensor,
        tokenizer,  # Will be our custom tokenizer
        max_length: int = 1024,
    ) -> str:
        """Generate a complete briefing from metrics"""
        encoder_output = self.encoder(
            nation_codes, nation_risks, nation_trends,
            signal_values, category_risks
        )

        tokens = self.decoder.generate(
            encoder_output,
            start_token=tokenizer.bos_token_id,
            max_length=max_length,
        )

        return tokenizer.decode(tokens)


# Quick test
if __name__ == "__main__":
    config = LFBMConfig()
    model = LFBM(config)

    # Dummy input
    batch_size = 2
    num_nations = 20

    nation_codes = torch.randint(0, 250, (batch_size, num_nations))
    nation_risks = torch.rand(batch_size, num_nations)
    nation_trends = torch.rand(batch_size, num_nations)
    signal_values = torch.rand(batch_size, 16)  # 16 signal types
    category_risks = torch.rand(batch_size, 26)  # 26 categories
    target_tokens = torch.randint(0, 8192, (batch_size, 100))

    logits = model(
        nation_codes, nation_risks, nation_trends,
        signal_values, category_risks, target_tokens
    )

    print(f"Output shape: {logits.shape}")  # [2, 100, 8192]
    print(f"Total params: {model.count_parameters():,}")
