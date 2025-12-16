"""
Data source adapters for LatticeForge.

Each adapter converts raw API responses to UniversalSignal format.
"""

from .conflict import (
    # ACLED
    ACLEDConfig,
    fetch_acled_events,
    from_acled_event,
    ingest_acled,
    # UCDP
    fetch_ucdp_events,
    from_ucdp_event,
    ingest_ucdp,
    # ReliefWeb
    fetch_reliefweb_reports,
    from_reliefweb_report,
    ingest_reliefweb,
    # Combined
    ingest_all_conflict_sources,
)

__all__ = [
    # ACLED
    "ACLEDConfig",
    "fetch_acled_events",
    "from_acled_event",
    "ingest_acled",
    # UCDP
    "fetch_ucdp_events",
    "from_ucdp_event",
    "ingest_ucdp",
    # ReliefWeb
    "fetch_reliefweb_reports",
    "from_reliefweb_report",
    "ingest_reliefweb",
    # Combined
    "ingest_all_conflict_sources",
]
