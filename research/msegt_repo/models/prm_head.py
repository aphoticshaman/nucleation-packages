import torch
from torch import nn

class PRMHead(nn.Module):
    """
    Simple process reward model head:
    input: (B, D) embeddings
    output: logits (B,) interpreted via sigmoid as p(correct | step, context)
    """
    def __init__(self, hidden_dim: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        logits = self.net(embeddings)  # (B, 1)
        return logits.squeeze(-1)

def prm_probs(prm_head: PRMHead, embeddings: torch.Tensor) -> torch.Tensor:
    logits = prm_head(embeddings)
    return torch.sigmoid(logits)
