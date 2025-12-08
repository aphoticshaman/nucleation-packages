"""Answer selection via value clustering.

Tries to import Rust implementation for 10-50x speedup.
Falls back to pure Python if Rust module not available.
"""

# Try Rust implementation first (10-50x faster)
try:
    from ryanaimo_clustering import (
        value_cluster_fast as value_clustering,
        basin_refinement_fast as basin_refinement,
        select_answer_fast as select_answer,
        Cluster,
    )
    RUST_AVAILABLE = True
    print("[RYANAIMO] Using Rust clustering (10-50x faster)")
except ImportError:
    # Fall back to pure Python
    from ryanaimo.selection.clustering import (
        value_clustering,
        basin_refinement,
        select_answer,
        Cluster,
    )
    RUST_AVAILABLE = False

# Always available from Python module
from ryanaimo.selection.clustering import relative_distance

__all__ = [
    "value_clustering",
    "basin_refinement",
    "select_answer",
    "relative_distance",
    "Cluster",
    "RUST_AVAILABLE",
]
