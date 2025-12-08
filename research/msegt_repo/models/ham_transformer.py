import math
import torch
from torch import nn

class HamiltonianAttention(nn.Module):
    """
    Prototype Hamiltonian (energy-stable) attention layer.
    This is not wired into Eagle by default; it is provided for
    experimentation in custom backbones.
    """
    def __init__(self, dim: int, epsilon: float = 0.03):
        super().__init__()
        self.epsilon = epsilon
        self.scale = 1.0 / math.sqrt(dim)
        self.H_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        A = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        attn_out = A @ V
        S = torch.cat([Q, V], dim=-1)
        H_input = attn_out
        H = self.H_net(H_input).sum()
        grads = torch.autograd.grad(H, [Q, V], create_graph=True, retain_graph=True)
        dH_dQ, dH_dV = grads
        dS = torch.cat([dH_dV, -dH_dQ], dim=-1)
        S_new = S + self.epsilon * dS
        _, V_new = torch.split(S_new, Q.size(-1), dim=-1)
        return V_new + attn_out
