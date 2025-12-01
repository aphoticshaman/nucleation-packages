"""Regime detection: MS-VAR with hierarchical Bayesian group LASSO (IBM)."""

from .msvar import (
    MSVARConfig,
    embed_var,
    MarkovSwitchingVAR,
)

__all__ = [
    "MSVARConfig",
    "embed_var",
    "MarkovSwitchingVAR",
]
