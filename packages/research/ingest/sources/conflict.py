"""
Conflict Data Source Adapters.

Free conflict/humanitarian data sources:
- ACLED (Armed Conflict Location & Event Data)
- UCDP (Uppsala Conflict Data Program)
- ReliefWeb (UN OCHA humanitarian info)

All adapters produce UniversalSignal format.
"""

import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
import urllib.request
import urllib.parse
import urllib.error

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from universal_signal import (
    UniversalSignal,
    SignalType,
    ConfidenceLevel,
    SourceTier,
    GeoReference,
    EntityReference,
    Provenance,
)


# ============================================================
# ACLED - Armed Conflict Location & Event Data
# https://acleddata.com/
# Free for research/non-commercial use
# ============================================================

@dataclass
class ACLEDConfig:
    """ACLED API configuration."""
    api_key: str
    email: str
    base_url: str = "https://api.acleddata.com/acled/read"


def fetch_acled_events(
    config: ACLEDConfig,
    country: Optional[str] = None,
    iso: Optional[int] = None,
    event_date_start: Optional[str] = None,
    event_date_end: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """
    Fetch conflict events from ACLED API.

    Args:
        config: ACLED API credentials
        country: Country name filter
        iso: ISO country code filter
        event_date_start: Start date (YYYY-MM-DD)
        event_date_end: End date (YYYY-MM-DD)
        event_type: Event type filter (Battles, Violence against civilians, etc.)
        limit: Maximum events to return

    Returns:
        List of raw ACLED event dictionaries
    """
    params = {
        "key": config.api_key,
        "email": config.email,
        "limit": limit,
    }

    if country:
        params["country"] = country
    if iso:
        params["iso"] = iso
    if event_date_start:
        params["event_date"] = event_date_start
        params["event_date_where"] = ">="
    if event_date_end:
        params["event_date"] = event_date_end
        params["event_date_where"] = "<="
    if event_type:
        params["event_type"] = event_type

    url = f"{config.base_url}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())
            return data.get("data", [])
    except urllib.error.URLError as e:
        print(f"ACLED API error: {e}")
        return []


def from_acled_event(event: Dict[str, Any]) -> UniversalSignal:
    """Convert ACLED event to UniversalSignal."""

    # Parse location
    lat = float(event.get("latitude", 0)) if event.get("latitude") else None
    lon = float(event.get("longitude", 0)) if event.get("longitude") else None

    geo = GeoReference(
        country_code=event.get("iso3", "")[:2] if event.get("iso3") else None,
        country_name=event.get("country"),
        region=event.get("admin1"),
        city=event.get("location"),
        lat=lat,
        lon=lon,
        precision="city" if event.get("location") else "region",
    )

    # Parse actors as entities
    entities = []
    if event.get("actor1"):
        entities.append(EntityReference(
            entity_type="organization",
            name=event["actor1"],
            id=event.get("inter1"),
        ))
    if event.get("actor2"):
        entities.append(EntityReference(
            entity_type="organization",
            name=event["actor2"],
            id=event.get("inter2"),
        ))

    # Fatalities as value
    fatalities = int(event.get("fatalities", 0))

    # Confidence based on source quality
    # ACLED uses multiple sources and verification
    confidence = ConfidenceLevel.HIGH if event.get("source_scale") else ConfidenceLevel.MEDIUM

    # Event type categorization
    event_type = event.get("event_type", "")
    sub_event_type = event.get("sub_event_type", "")
    categories = [event_type, sub_event_type] if sub_event_type else [event_type]
    categories = [c.lower().replace(" ", "_") for c in categories if c]

    return UniversalSignal(
        id=f"acled:{event.get('event_id_cnty', event.get('data_id', ''))}",
        signal_type=SignalType.EVENT,
        timestamp=f"{event.get('event_date', '')}T00:00:00Z",
        geo=geo,
        title=f"{event_type}: {event.get('location', 'Unknown')}",
        content=event.get("notes", ""),
        value=float(fatalities) if fatalities > 0 else None,
        value_type="fatalities",
        confidence=confidence,
        confidence_score=0.85 if confidence == ConfidenceLevel.HIGH else 0.65,
        entities=entities,
        categories=["conflict"] + categories,
        tags=[event.get("event_type", ""), event.get("disorder_type", "")],
        provenance=Provenance(
            source_id="acled",
            source_name="Armed Conflict Location & Event Data",
            source_tier=SourceTier.INSTITUTIONAL,
            source_url=f"https://acleddata.com/data-export-tool/",
            attribution="Data from ACLED (www.acleddata.com)",
            license="Free for research/non-commercial use",
        ),
        enrichments={
            "event_type": event_type,
            "sub_event_type": sub_event_type,
            "disorder_type": event.get("disorder_type"),
            "interaction": event.get("interaction"),
            "source": event.get("source"),
        },
        raw=event,
    )


def ingest_acled(
    config: ACLEDConfig,
    countries: Optional[List[str]] = None,
    days_back: int = 7,
    limit_per_country: int = 100,
) -> Generator[UniversalSignal, None, None]:
    """
    Ingest recent ACLED events as UniversalSignals.

    Args:
        config: ACLED API credentials
        countries: List of country names to fetch (None = all)
        days_back: How many days of history
        limit_per_country: Max events per country

    Yields:
        UniversalSignal for each event
    """
    from datetime import timedelta

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    if countries:
        for country in countries:
            events = fetch_acled_events(
                config,
                country=country,
                event_date_start=start_str,
                event_date_end=end_str,
                limit=limit_per_country,
            )
            for event in events:
                yield from_acled_event(event)
    else:
        events = fetch_acled_events(
            config,
            event_date_start=start_str,
            event_date_end=end_str,
            limit=limit_per_country * 10,
        )
        for event in events:
            yield from_acled_event(event)


# ============================================================
# UCDP - Uppsala Conflict Data Program
# https://ucdp.uu.se/
# Fully open, academic conflict database
# ============================================================

UCDP_API_BASE = "https://ucdpapi.pcr.uu.se/api"


def fetch_ucdp_events(
    year: Optional[int] = None,
    country_id: Optional[int] = None,
    page: int = 1,
    page_size: int = 100,
) -> Dict[str, Any]:
    """
    Fetch events from UCDP Georeferenced Event Dataset (GED).

    Args:
        year: Filter by year
        country_id: UCDP country ID
        page: Page number
        page_size: Results per page

    Returns:
        API response with events
    """
    params = {
        "pagesize": page_size,
        "page": page,
    }

    if year:
        params["Year"] = year
    if country_id:
        params["Country"] = country_id

    url = f"{UCDP_API_BASE}/gedevents/23.1?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode())
    except urllib.error.URLError as e:
        print(f"UCDP API error: {e}")
        return {"Result": []}


def from_ucdp_event(event: Dict[str, Any]) -> UniversalSignal:
    """Convert UCDP GED event to UniversalSignal."""

    geo = GeoReference(
        country_code=event.get("country_id"),
        country_name=event.get("country"),
        region=event.get("adm_1"),
        city=event.get("adm_2"),
        lat=float(event.get("latitude", 0)) if event.get("latitude") else None,
        lon=float(event.get("longitude", 0)) if event.get("longitude") else None,
        precision="city" if event.get("adm_2") else "region",
    )

    # Actors
    entities = []
    if event.get("side_a"):
        entities.append(EntityReference(
            entity_type="organization",
            name=event["side_a"],
        ))
    if event.get("side_b"):
        entities.append(EntityReference(
            entity_type="organization",
            name=event["side_b"],
        ))

    # Deaths
    deaths_a = int(event.get("deaths_a", 0))
    deaths_b = int(event.get("deaths_b", 0))
    deaths_civilians = int(event.get("deaths_civilians", 0))
    deaths_unknown = int(event.get("deaths_unknown", 0))
    total_deaths = int(event.get("best", deaths_a + deaths_b + deaths_civilians + deaths_unknown))

    # Conflict type
    type_of_violence = event.get("type_of_violence", 0)
    violence_types = {
        1: "state-based",
        2: "non-state",
        3: "one-sided",
    }
    conflict_type = violence_types.get(type_of_violence, "unknown")

    return UniversalSignal(
        id=f"ucdp:{event.get('id', '')}",
        signal_type=SignalType.EVENT,
        timestamp=f"{event.get('date_start', '')}T00:00:00Z" if event.get("date_start") else datetime.utcnow().isoformat() + "Z",
        geo=geo,
        title=f"UCDP Event: {event.get('country', 'Unknown')} ({conflict_type})",
        content=event.get("source_article", ""),
        value=float(total_deaths) if total_deaths > 0 else None,
        value_type="fatalities",
        confidence=ConfidenceLevel.HIGH,  # UCDP is rigorously verified
        confidence_score=0.9,
        entities=entities,
        categories=["conflict", conflict_type],
        tags=[event.get("dyad_name", "")],
        provenance=Provenance(
            source_id="ucdp",
            source_name="Uppsala Conflict Data Program",
            source_tier=SourceTier.INSTITUTIONAL,
            source_url="https://ucdp.uu.se/",
            attribution="UCDP Georeferenced Event Dataset (GED) Global version 23.1",
            license="Open Data",
        ),
        enrichments={
            "type_of_violence": conflict_type,
            "dyad_name": event.get("dyad_name"),
            "conflict_name": event.get("conflict_name"),
            "deaths_breakdown": {
                "side_a": deaths_a,
                "side_b": deaths_b,
                "civilians": deaths_civilians,
                "unknown": deaths_unknown,
            },
        },
        raw=event,
    )


def ingest_ucdp(
    year: Optional[int] = None,
    countries: Optional[List[int]] = None,
    max_pages: int = 5,
) -> Generator[UniversalSignal, None, None]:
    """
    Ingest UCDP events as UniversalSignals.

    Args:
        year: Filter by year (None = latest)
        countries: List of UCDP country IDs
        max_pages: Maximum pages to fetch

    Yields:
        UniversalSignal for each event
    """
    if year is None:
        year = datetime.utcnow().year - 1  # UCDP data has ~1 year lag

    if countries:
        for country_id in countries:
            for page in range(1, max_pages + 1):
                response = fetch_ucdp_events(year=year, country_id=country_id, page=page)
                events = response.get("Result", [])
                if not events:
                    break
                for event in events:
                    yield from_ucdp_event(event)
    else:
        for page in range(1, max_pages + 1):
            response = fetch_ucdp_events(year=year, page=page)
            events = response.get("Result", [])
            if not events:
                break
            for event in events:
                yield from_ucdp_event(event)


# ============================================================
# ReliefWeb - UN OCHA Humanitarian Information
# https://reliefweb.int/
# Fully open, no auth required
# ============================================================

RELIEFWEB_API_BASE = "https://api.reliefweb.int/v1"


def fetch_reliefweb_reports(
    query: Optional[str] = None,
    country: Optional[str] = None,
    disaster_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Fetch reports from ReliefWeb API.

    Args:
        query: Full-text search query
        country: Country filter (ISO3 or name)
        disaster_type: Disaster type filter
        limit: Results per page
        offset: Pagination offset

    Returns:
        API response with reports
    """
    payload = {
        "limit": limit,
        "offset": offset,
        "fields": {
            "include": [
                "id", "title", "body", "date.created", "date.original",
                "country.name", "country.iso3", "source.name",
                "disaster.name", "disaster_type.name", "theme.name",
                "primary_country.location",
            ]
        },
        "sort": ["date.created:desc"],
    }

    filters = []
    if query:
        filters.append({"field": "title", "value": query})
    if country:
        filters.append({"field": "country.iso3", "value": country.upper()})
    if disaster_type:
        filters.append({"field": "disaster_type.name", "value": disaster_type})

    if filters:
        payload["filter"] = {"conditions": filters}

    url = f"{RELIEFWEB_API_BASE}/reports"

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode())
    except urllib.error.URLError as e:
        print(f"ReliefWeb API error: {e}")
        return {"data": []}


def from_reliefweb_report(report: Dict[str, Any]) -> UniversalSignal:
    """Convert ReliefWeb report to UniversalSignal."""
    fields = report.get("fields", {})

    # Get country info
    countries = fields.get("country", [])
    primary_country = countries[0] if countries else {}

    # Location from primary country
    location = fields.get("primary_country", {}).get("location", {})

    geo = GeoReference(
        country_code=primary_country.get("iso3", "")[:2] if primary_country.get("iso3") else None,
        country_name=primary_country.get("name"),
        lat=location.get("lat"),
        lon=location.get("lon"),
        precision="country",
    )

    # Sources as entities
    entities = []
    for source in fields.get("source", []):
        entities.append(EntityReference(
            entity_type="organization",
            name=source.get("name", "Unknown"),
        ))

    # Categories from themes and disaster types
    categories = ["humanitarian"]
    for theme in fields.get("theme", []):
        categories.append(theme.get("name", "").lower().replace(" ", "_"))
    for dtype in fields.get("disaster_type", []):
        categories.append(dtype.get("name", "").lower().replace(" ", "_"))

    # Disasters as tags
    tags = [d.get("name", "") for d in fields.get("disaster", [])]

    # Timestamp
    timestamp = fields.get("date", {}).get("original") or fields.get("date", {}).get("created")
    if timestamp:
        timestamp = timestamp.replace(" ", "T") + "Z" if "T" not in timestamp else timestamp

    return UniversalSignal(
        id=f"reliefweb:{report.get('id', '')}",
        signal_type=SignalType.NEWS,
        timestamp=timestamp or datetime.utcnow().isoformat() + "Z",
        geo=geo,
        title=fields.get("title", ""),
        content=fields.get("body", "")[:2000],  # Truncate long bodies
        confidence=ConfidenceLevel.HIGH,  # UN OCHA sources
        confidence_score=0.9,
        entities=entities,
        categories=categories[:5],
        tags=tags[:5],
        provenance=Provenance(
            source_id="reliefweb",
            source_name="ReliefWeb (UN OCHA)",
            source_tier=SourceTier.INSTITUTIONAL,
            source_url=f"https://reliefweb.int/node/{report.get('id', '')}",
            attribution="Data from ReliefWeb (OCHA)",
            license="Open Data",
        ),
        enrichments={
            "disaster_types": [d.get("name") for d in fields.get("disaster_type", [])],
            "themes": [t.get("name") for t in fields.get("theme", [])],
            "disasters": [d.get("name") for d in fields.get("disaster", [])],
        },
        raw=report,
    )


def ingest_reliefweb(
    query: Optional[str] = None,
    countries: Optional[List[str]] = None,
    disaster_types: Optional[List[str]] = None,
    limit: int = 100,
) -> Generator[UniversalSignal, None, None]:
    """
    Ingest ReliefWeb reports as UniversalSignals.

    Args:
        query: Search query
        countries: List of ISO3 country codes
        disaster_types: List of disaster type names
        limit: Maximum reports to fetch

    Yields:
        UniversalSignal for each report
    """
    if countries:
        for country in countries:
            response = fetch_reliefweb_reports(
                query=query,
                country=country,
                limit=min(limit, 50),
            )
            for report in response.get("data", []):
                yield from_reliefweb_report(report)
    elif disaster_types:
        for dtype in disaster_types:
            response = fetch_reliefweb_reports(
                query=query,
                disaster_type=dtype,
                limit=min(limit, 50),
            )
            for report in response.get("data", []):
                yield from_reliefweb_report(report)
    else:
        response = fetch_reliefweb_reports(query=query, limit=limit)
        for report in response.get("data", []):
            yield from_reliefweb_report(report)


# ============================================================
# Convenience: Fetch all conflict data
# ============================================================

def ingest_all_conflict_sources(
    acled_config: Optional[ACLEDConfig] = None,
    countries: Optional[List[str]] = None,
    days_back: int = 7,
) -> Generator[UniversalSignal, None, None]:
    """
    Ingest from all available conflict sources.

    Args:
        acled_config: ACLED credentials (if available)
        countries: Country filter
        days_back: Days of history to fetch

    Yields:
        UniversalSignal from each source
    """
    # ACLED (requires API key)
    if acled_config:
        print("Ingesting from ACLED...")
        try:
            for signal in ingest_acled(acled_config, countries=countries, days_back=days_back):
                yield signal
        except Exception as e:
            print(f"ACLED error: {e}")

    # UCDP (open, no auth)
    print("Ingesting from UCDP...")
    try:
        for signal in ingest_ucdp():
            yield signal
    except Exception as e:
        print(f"UCDP error: {e}")

    # ReliefWeb (open, no auth)
    print("Ingesting from ReliefWeb...")
    try:
        # Convert country names to ISO3 if needed
        iso3_countries = None
        if countries:
            # Simple mapping - in production would use a proper lookup
            iso3_countries = countries  # Assume already ISO3

        for signal in ingest_reliefweb(countries=iso3_countries):
            yield signal
    except Exception as e:
        print(f"ReliefWeb error: {e}")


# ============================================================
# CLI for testing
# ============================================================

if __name__ == "__main__":
    import sys

    print("Testing conflict data sources...\n")

    # Test UCDP (no auth required)
    print("=" * 50)
    print("UCDP - Uppsala Conflict Data Program")
    print("=" * 50)
    count = 0
    for signal in ingest_ucdp(max_pages=1):
        print(f"  {signal.id}: {signal.title}")
        print(f"    Location: {signal.geo.country_name if signal.geo else 'Unknown'}")
        print(f"    Fatalities: {signal.value}")
        print()
        count += 1
        if count >= 3:
            break
    print(f"Total UCDP events: {count}\n")

    # Test ReliefWeb (no auth required)
    print("=" * 50)
    print("ReliefWeb - UN OCHA Humanitarian Info")
    print("=" * 50)
    count = 0
    for signal in ingest_reliefweb(limit=5):
        print(f"  {signal.id}: {signal.title[:60]}...")
        print(f"    Country: {signal.geo.country_name if signal.geo else 'Unknown'}")
        print(f"    Categories: {', '.join(signal.categories[:3])}")
        print()
        count += 1
        if count >= 3:
            break
    print(f"Total ReliefWeb reports: {count}\n")

    print("Note: ACLED requires API key. Register at https://acleddata.com/register/")
