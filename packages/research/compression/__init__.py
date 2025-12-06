"""
Compression submodule for research package.

Implements novel compression techniques including:
- CMFC: Contextual Multi-Fold Compression
"""

from .cmfc import (
    CMFCEncoder,
    CMFCDecoder,
    HierarchicalCMFC,
    CMFCForARC,
    FoldedRepresentation,
    UnfoldResult,
    PathwaySelectionMode,
    EnergyFunction,
    contrastive_cmfc_loss,
)

__all__ = [
    "CMFCEncoder",
    "CMFCDecoder",
    "HierarchicalCMFC",
    "CMFCForARC",
    "FoldedRepresentation",
    "UnfoldResult",
    "PathwaySelectionMode",
    "EnergyFunction",
    "contrastive_cmfc_loss",
]
