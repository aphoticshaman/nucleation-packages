"""
Universal data ingestion and export for LatticeForge.

Ingest from anywhere. Export to anything.
All deterministic. All auditable. No LLM.
"""

from .universal_signal import (
    SignalType,
    ConfidenceLevel,
    SourceTier,
    GeoReference,
    EntityReference,
    Provenance,
    UniversalSignal,
    SignalBatch,
    # Adapters
    from_gdelt_article,
    from_fred_observation,
    from_world_bank,
    from_latticeforge_alert,
)

from .source_registry import (
    SourceInfo,
    SOURCE_REGISTRY,
    get_free_sources,
    get_sources_by_category,
)

from .export_adapters import (
    to_csv,
    to_json,
    to_ndjson,
    to_stix_bundle,
    to_stix_json,
    to_cef,
    to_leef,
    to_webhook_payload,
)

__all__ = [
    # Core types
    "SignalType",
    "ConfidenceLevel",
    "SourceTier",
    "GeoReference",
    "EntityReference",
    "Provenance",
    "UniversalSignal",
    "SignalBatch",
    # Ingest adapters
    "from_gdelt_article",
    "from_fred_observation",
    "from_world_bank",
    "from_latticeforge_alert",
    # Source registry
    "SourceInfo",
    "SOURCE_REGISTRY",
    "get_free_sources",
    "get_sources_by_category",
    # Export adapters
    "to_csv",
    "to_json",
    "to_ndjson",
    "to_stix_bundle",
    "to_stix_json",
    "to_cef",
    "to_leef",
    "to_webhook_payload",
]
