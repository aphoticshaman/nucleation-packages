"""
Source Registry - Catalog of all available data sources.

Categorized by:
- Cost (free/paid)
- Category (news, economic, conflict, social, etc.)
- Rate limits
- Authentication requirements

This is the "menu" of data sources LatticeForge can ingest.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class SourceCategory(Enum):
    """Categories of data sources."""
    NEWS = "news"
    EVENTS = "events"
    ECONOMIC = "economic"
    CONFLICT = "conflict"
    SOCIAL = "social"
    GOVERNMENT = "government"
    FINANCIAL = "financial"
    THREAT_INTEL = "threat_intel"
    WEATHER = "weather"
    ACADEMIC = "academic"
    CUSTOM = "custom"


class AuthType(Enum):
    """Authentication types."""
    NONE = "none"
    API_KEY = "api_key"
    OAUTH = "oauth"
    BEARER = "bearer"
    BASIC = "basic"


@dataclass
class RateLimit:
    """Rate limit specification."""
    requests_per_minute: Optional[int] = None
    requests_per_day: Optional[int] = None
    requests_per_month: Optional[int] = None


@dataclass
class SourceInfo:
    """Complete information about a data source."""
    id: str
    name: str
    description: str
    category: SourceCategory
    url: str

    # Cost
    is_free: bool
    free_tier_limits: Optional[str] = None  # Description of free tier
    paid_plans: Optional[str] = None

    # Authentication
    auth_type: AuthType = AuthType.NONE
    auth_url: Optional[str] = None  # Where to get credentials

    # Rate limits (free tier)
    rate_limit: Optional[RateLimit] = None

    # Data characteristics
    update_frequency: str = "varies"  # e.g., "15 minutes", "daily"
    historical_depth: Optional[str] = None  # e.g., "2 years", "since 2013"
    coverage: str = "global"  # Geographic coverage

    # Integration status in LatticeForge
    implemented: bool = False
    implementation_file: Optional[str] = None

    # Notes
    notes: Optional[str] = None


# ============================================================
# THE REGISTRY
# ============================================================

SOURCE_REGISTRY: Dict[str, SourceInfo] = {}


def _register(source: SourceInfo):
    """Register a source in the registry."""
    SOURCE_REGISTRY[source.id] = source


# ============================================================
# FREE SOURCES (Priority for bootstrap)
# ============================================================

_register(SourceInfo(
    id="gdelt",
    name="GDELT Project",
    description="Global Database of Events, Language, and Tone. Monitors worldwide news media in 100+ languages.",
    category=SourceCategory.NEWS,
    url="https://www.gdeltproject.org/",
    is_free=True,
    rate_limit=RateLimit(requests_per_minute=60),
    update_frequency="15 minutes",
    historical_depth="Since 2015 (v2)",
    coverage="Global, 100+ languages",
    implemented=True,
    implementation_file="packages/social-pulse/src/sources/gdelt.ts",
    notes="Best free source for real-time global news events. No auth required."
))

_register(SourceInfo(
    id="fred",
    name="FRED (Federal Reserve Economic Data)",
    description="800,000+ economic time series from the St. Louis Fed.",
    category=SourceCategory.ECONOMIC,
    url="https://fred.stlouisfed.org/",
    is_free=True,
    auth_type=AuthType.API_KEY,
    auth_url="https://fred.stlouisfed.org/docs/api/api_key.html",
    rate_limit=RateLimit(requests_per_minute=120),
    update_frequency="Varies by series",
    historical_depth="Decades for most series",
    coverage="Primarily US, some international",
    implemented=True,
    implementation_file="packages/sources/official/fred/src/index.ts",
    notes="Gold standard for US economic data. Free API key required."
))

_register(SourceInfo(
    id="world_bank",
    name="World Bank Open Data",
    description="Development indicators for 217 countries.",
    category=SourceCategory.ECONOMIC,
    url="https://data.worldbank.org/",
    is_free=True,
    update_frequency="Annual/Quarterly",
    historical_depth="Since 1960 for many indicators",
    coverage="Global (217 countries)",
    implemented=True,
    implementation_file="packages/social-pulse/src/sources/economic-indicators.ts",
    notes="Best free source for international economic comparisons."
))

_register(SourceInfo(
    id="acled",
    name="ACLED (Armed Conflict Location & Event Data)",
    description="Real-time conflict event data globally.",
    category=SourceCategory.CONFLICT,
    url="https://acleddata.com/",
    is_free=True,
    free_tier_limits="Free for research/non-commercial use",
    auth_type=AuthType.API_KEY,
    auth_url="https://acleddata.com/register/",
    update_frequency="Weekly",
    historical_depth="Since 1997 (varies by region)",
    coverage="Global",
    implemented=False,
    notes="Critical for geopolitical risk. Register for free research access."
))

_register(SourceInfo(
    id="ucdp",
    name="UCDP (Uppsala Conflict Data Program)",
    description="Academic conflict dataset from Uppsala University.",
    category=SourceCategory.CONFLICT,
    url="https://ucdp.uu.se/",
    is_free=True,
    update_frequency="Annual (with mid-year updates)",
    historical_depth="Since 1946",
    coverage="Global",
    implemented=False,
    notes="Most cited conflict database in academia. Fully open."
))

_register(SourceInfo(
    id="cdc",
    name="CDC Open Data",
    description="US health and disease surveillance data.",
    category=SourceCategory.GOVERNMENT,
    url="https://data.cdc.gov/",
    is_free=True,
    update_frequency="Varies by dataset",
    coverage="United States",
    implemented=True,
    implementation_file="packages/sources/official/cdc/src/index.ts",
))

_register(SourceInfo(
    id="sec_edgar",
    name="SEC EDGAR",
    description="US corporate filings (10-K, 8-K, etc.)",
    category=SourceCategory.FINANCIAL,
    url="https://www.sec.gov/edgar.shtml",
    is_free=True,
    rate_limit=RateLimit(requests_per_minute=10),
    update_frequency="Real-time filings",
    coverage="US public companies",
    implemented=True,
    implementation_file="packages/sources/official/sec-edgar/src/index.ts",
))

_register(SourceInfo(
    id="arxiv",
    name="arXiv",
    description="Open access academic preprints.",
    category=SourceCategory.ACADEMIC,
    url="https://arxiv.org/",
    is_free=True,
    update_frequency="Daily",
    coverage="Global academic research",
    implemented=True,
    implementation_file="packages/social-pulse/src/sources/arxiv.ts",
    notes="Useful for tracking emerging tech/science trends."
))

_register(SourceInfo(
    id="reliefweb",
    name="ReliefWeb",
    description="UN OCHA humanitarian information service.",
    category=SourceCategory.EVENTS,
    url="https://reliefweb.int/",
    is_free=True,
    update_frequency="Real-time",
    coverage="Global humanitarian crises",
    implemented=False,
    notes="Key source for disaster and humanitarian events."
))

_register(SourceInfo(
    id="eonet",
    name="NASA EONET",
    description="Earth Observatory Natural Event Tracker.",
    category=SourceCategory.WEATHER,
    url="https://eonet.gsfc.nasa.gov/",
    is_free=True,
    update_frequency="Near real-time",
    coverage="Global natural events",
    implemented=False,
    notes="Wildfires, storms, volcanic activity, etc."
))

_register(SourceInfo(
    id="bluesky",
    name="Bluesky",
    description="Decentralized social network.",
    category=SourceCategory.SOCIAL,
    url="https://bsky.app/",
    is_free=True,
    auth_type=AuthType.BEARER,
    update_frequency="Real-time",
    coverage="Social media posts",
    implemented=True,
    implementation_file="packages/social-pulse/src/sources/bluesky.ts",
))

_register(SourceInfo(
    id="telegram",
    name="Telegram",
    description="Public Telegram channels.",
    category=SourceCategory.SOCIAL,
    url="https://telegram.org/",
    is_free=True,
    auth_type=AuthType.API_KEY,
    update_frequency="Real-time",
    coverage="Public channels only",
    implemented=True,
    implementation_file="packages/social-pulse/src/sources/telegram.ts",
    notes="Critical for OSINT in conflict zones."
))


# ============================================================
# PAID SOURCES (Future expansion)
# ============================================================

_register(SourceInfo(
    id="newsapi",
    name="NewsAPI",
    description="Aggregated news from 80,000+ sources.",
    category=SourceCategory.NEWS,
    url="https://newsapi.org/",
    is_free=False,
    free_tier_limits="100 requests/day, 1 month old articles only",
    paid_plans="$449/mo for 250K requests",
    auth_type=AuthType.API_KEY,
    auth_url="https://newsapi.org/register",
    update_frequency="Real-time",
    coverage="Global",
    implemented=False,
    notes="Good but GDELT is better for free tier."
))

_register(SourceInfo(
    id="event_registry",
    name="Event Registry",
    description="News and event extraction from global media.",
    category=SourceCategory.NEWS,
    url="https://eventregistry.org/",
    is_free=False,
    free_tier_limits="2,000 tokens/day",
    paid_plans="Starting $99/mo",
    auth_type=AuthType.API_KEY,
    update_frequency="Real-time",
    coverage="Global, 100+ languages",
    implemented=False,
))

_register(SourceInfo(
    id="polygon",
    name="Polygon.io",
    description="Real-time and historical market data.",
    category=SourceCategory.FINANCIAL,
    url="https://polygon.io/",
    is_free=False,
    free_tier_limits="5 API calls/min, delayed data",
    paid_plans="$29/mo for real-time",
    auth_type=AuthType.API_KEY,
    update_frequency="Real-time",
    coverage="US stocks, crypto, forex",
    implemented=False,
))

_register(SourceInfo(
    id="alpha_vantage",
    name="Alpha Vantage",
    description="Free stock/crypto/forex data API.",
    category=SourceCategory.FINANCIAL,
    url="https://www.alphavantage.co/",
    is_free=True,
    free_tier_limits="5 API calls/min, 500/day",
    auth_type=AuthType.API_KEY,
    auth_url="https://www.alphavantage.co/support/#api-key",
    update_frequency="Real-time (with limits)",
    coverage="Global markets",
    implemented=False,
    notes="Good free option for market data."
))


# ============================================================
# THREAT INTEL SOURCES (Specialized)
# ============================================================

_register(SourceInfo(
    id="otx",
    name="AlienVault OTX",
    description="Open Threat Exchange - crowd-sourced threat intel.",
    category=SourceCategory.THREAT_INTEL,
    url="https://otx.alienvault.com/",
    is_free=True,
    auth_type=AuthType.API_KEY,
    update_frequency="Real-time",
    coverage="Global cyber threats",
    implemented=False,
    notes="Useful for cyber-physical convergence analysis."
))

_register(SourceInfo(
    id="abuse_ch",
    name="abuse.ch",
    description="Swiss non-profit tracking malware and botnets.",
    category=SourceCategory.THREAT_INTEL,
    url="https://abuse.ch/",
    is_free=True,
    update_frequency="Real-time",
    coverage="Global malware/botnet tracking",
    implemented=False,
))


# ============================================================
# COMPETITOR/INTEGRATION TARGETS
# ============================================================

_register(SourceInfo(
    id="recorded_future",
    name="Recorded Future",
    description="Threat intelligence platform (potential acquirer).",
    category=SourceCategory.THREAT_INTEL,
    url="https://www.recordedfuture.com/",
    is_free=False,
    paid_plans="Enterprise pricing",
    auth_type=AuthType.API_KEY,
    implemented=False,
    notes="INTEGRATION TARGET: Can consume their API, they could consume ours."
))

_register(SourceInfo(
    id="dataminr",
    name="Dataminr",
    description="Real-time event detection (competitor).",
    category=SourceCategory.EVENTS,
    url="https://www.dataminr.com/",
    is_free=False,
    paid_plans="Enterprise pricing",
    implemented=False,
    notes="COMPETITOR: Our phase detection could complement their event detection."
))

_register(SourceInfo(
    id="palantir_foundry",
    name="Palantir Foundry",
    description="Data integration platform.",
    category=SourceCategory.CUSTOM,
    url="https://www.palantir.com/platforms/foundry/",
    is_free=False,
    paid_plans="$141K+/seat",
    implemented=False,
    notes="INTEGRATION TARGET: LatticeForge as a data source for Foundry."
))


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_free_sources() -> List[SourceInfo]:
    """Get all free data sources."""
    return [s for s in SOURCE_REGISTRY.values() if s.is_free]


def get_implemented_sources() -> List[SourceInfo]:
    """Get all implemented data sources."""
    return [s for s in SOURCE_REGISTRY.values() if s.implemented]


def get_sources_by_category(category: SourceCategory) -> List[SourceInfo]:
    """Get sources by category."""
    return [s for s in SOURCE_REGISTRY.values() if s.category == category]


def get_sources_requiring_auth() -> List[SourceInfo]:
    """Get sources that require authentication."""
    return [s for s in SOURCE_REGISTRY.values() if s.auth_type != AuthType.NONE]


def print_source_summary():
    """Print summary of available sources."""
    print("=" * 60)
    print("LATTICEFORGE DATA SOURCE REGISTRY")
    print("=" * 60)

    free_count = len(get_free_sources())
    impl_count = len(get_implemented_sources())
    total_count = len(SOURCE_REGISTRY)

    print(f"\nTotal Sources: {total_count}")
    print(f"Free Sources: {free_count}")
    print(f"Implemented: {impl_count}")

    print("\n--- BY CATEGORY ---")
    for cat in SourceCategory:
        sources = get_sources_by_category(cat)
        if sources:
            print(f"\n{cat.value.upper()} ({len(sources)}):")
            for s in sources:
                status = "✓" if s.implemented else "○"
                cost = "FREE" if s.is_free else "PAID"
                print(f"  {status} [{cost}] {s.name}")
