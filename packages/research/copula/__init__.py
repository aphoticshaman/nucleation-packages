"""Copula models for tail dependence and correlation modeling (S&P patent)."""

from .gaussian_copula import (
    GaussianCopula,
    TailDependenceAnalyzer,
    CopulaConfig,
    compute_kendall_tau,
    compute_spearman_rho,
    fit_gaussian_copula,
    simulate_from_copula,
    compute_tail_dependence,
)

__all__ = [
    "GaussianCopula",
    "TailDependenceAnalyzer",
    "CopulaConfig",
    "compute_kendall_tau",
    "compute_spearman_rho",
    "fit_gaussian_copula",
    "simulate_from_copula",
    "compute_tail_dependence",
]
