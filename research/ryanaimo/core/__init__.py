"""Core CIC primitives and configuration."""

from ryanaimo.core.cic import (
    ncd,
    phi_integrated_information,
    representation_entropy,
    causal_power_multiscale,
    compute_cic_functional,
    CICState,
)
from ryanaimo.core.config import RyanAIMOConfig

__all__ = [
    "ncd",
    "phi_integrated_information",
    "representation_entropy",
    "causal_power_multiscale",
    "compute_cic_functional",
    "CICState",
    "RyanAIMOConfig",
]
