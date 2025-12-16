"""
Universal Signal Format for LatticeForge.

All data sources normalize to this format.
All exports can serialize from this format.

This is the data contract that makes LatticeForge
interoperable with any system.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class SignalType(Enum):
    """Types of signals LatticeForge processes."""
    # Events
    EVENT = "event"              # Discrete occurrence
    NEWS = "news"                # News article
    SOCIAL = "social"            # Social media post

    # Metrics
    METRIC = "metric"            # Numeric measurement
    INDICATOR = "indicator"      # Economic/political indicator
    INDEX = "index"              # Composite index value

    # Analysis
    ALERT = "alert"              # Generated alert
    PHASE = "phase"              # Phase transition signal
    CAUSAL = "causal"            # Causal relationship

    # External
    THREAT = "threat"            # Threat intelligence
    VULNERABILITY = "vulnerability"


class ConfidenceLevel(Enum):
    """Confidence/reliability levels."""
    UNKNOWN = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CONFIRMED = 4


class SourceTier(Enum):
    """Data source reliability tiers."""
    OFFICIAL = "official"        # Government, central banks
    INSTITUTIONAL = "institutional"  # Research orgs, NGOs
    COMMERCIAL = "commercial"    # Commercial data providers
    OPEN = "open"                # Open source (GDELT, etc.)
    CROWDSOURCED = "crowdsourced"  # Social, user-generated
    DERIVED = "derived"          # LatticeForge-computed


@dataclass
class GeoReference:
    """Geographic reference for a signal."""
    country_code: Optional[str] = None  # ISO 3166-1 alpha-2
    country_name: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    precision: str = "country"  # country, region, city, exact

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class EntityReference:
    """Reference to an entity (person, org, etc.)."""
    entity_type: str  # person, organization, location, event
    name: str
    id: Optional[str] = None  # External ID if known
    aliases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Provenance:
    """
    Full provenance tracking for auditability.

    This is critical for defense/intel customers who need
    to trace every data point back to its source.
    """
    source_id: str                    # Unique source identifier
    source_name: str                  # Human-readable name
    source_tier: SourceTier           # Reliability tier
    source_url: Optional[str] = None  # Original URL if applicable
    fetched_at: Optional[str] = None  # ISO timestamp of fetch

    # Attribution
    attribution: Optional[str] = None  # Required attribution text
    license: Optional[str] = None      # License type

    # Processing chain
    transformations: List[str] = field(default_factory=list)

    # Integrity
    original_hash: Optional[str] = None  # Hash of original data

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['source_tier'] = self.source_tier.value
        return d


@dataclass
class UniversalSignal:
    """
    Universal signal format for all LatticeForge data.

    Every piece of data - whether from GDELT, FRED, Palantir,
    or computed internally - normalizes to this format.

    This enables:
    1. Unified fusion across heterogeneous sources
    2. Consistent export to any destination
    3. Full auditability and provenance tracking
    4. Deterministic processing (no LLM interpretation)
    """

    # Identity
    id: str                           # Unique signal ID
    signal_type: SignalType           # Type of signal

    # Temporal
    timestamp: str                    # ISO 8601 timestamp
    valid_from: Optional[str] = None  # Validity window start
    valid_until: Optional[str] = None # Validity window end

    # Spatial
    geo: Optional[GeoReference] = None

    # Content
    title: Optional[str] = None       # Brief title/headline
    content: Optional[str] = None     # Full content/description

    # Numeric value (for metrics/indicators)
    value: Optional[float] = None
    value_type: Optional[str] = None  # What the value represents
    unit: Optional[str] = None        # Unit of measurement

    # Confidence
    confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    confidence_score: float = 0.0     # 0.0 to 1.0

    # Relationships
    entities: List[EntityReference] = field(default_factory=list)
    related_signals: List[str] = field(default_factory=list)  # IDs

    # Classification
    categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # Provenance
    provenance: Optional[Provenance] = None

    # LatticeForge-computed enrichments
    enrichments: Dict[str, Any] = field(default_factory=dict)

    # Raw data (for debugging/audit)
    raw: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate deterministic ID from content."""
        content = f"{self.signal_type.value}:{self.timestamp}:{self.title or ''}:{self.value or ''}"
        if self.provenance:
            content += f":{self.provenance.source_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            'id': self.id,
            'signal_type': self.signal_type.value,
            'timestamp': self.timestamp,
        }

        if self.valid_from:
            d['valid_from'] = self.valid_from
        if self.valid_until:
            d['valid_until'] = self.valid_until
        if self.geo:
            d['geo'] = self.geo.to_dict()
        if self.title:
            d['title'] = self.title
        if self.content:
            d['content'] = self.content
        if self.value is not None:
            d['value'] = self.value
            if self.value_type:
                d['value_type'] = self.value_type
            if self.unit:
                d['unit'] = self.unit

        d['confidence'] = self.confidence.name.lower()
        d['confidence_score'] = self.confidence_score

        if self.entities:
            d['entities'] = [e.to_dict() for e in self.entities]
        if self.related_signals:
            d['related_signals'] = self.related_signals
        if self.categories:
            d['categories'] = self.categories
        if self.tags:
            d['tags'] = self.tags
        if self.provenance:
            d['provenance'] = self.provenance.to_dict()
        if self.enrichments:
            d['enrichments'] = self.enrichments

        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UniversalSignal':
        """Deserialize from dictionary."""
        # Handle enums
        data['signal_type'] = SignalType(data['signal_type'])

        if 'confidence' in data:
            conf_str = data['confidence'].upper()
            data['confidence'] = ConfidenceLevel[conf_str]

        # Handle nested objects
        if 'geo' in data and data['geo']:
            data['geo'] = GeoReference(**data['geo'])

        if 'provenance' in data and data['provenance']:
            prov_data = data['provenance']
            prov_data['source_tier'] = SourceTier(prov_data['source_tier'])
            data['provenance'] = Provenance(**prov_data)

        if 'entities' in data:
            data['entities'] = [EntityReference(**e) for e in data['entities']]

        return cls(**data)


# ============================================================
# Signal Batch for bulk operations
# ============================================================

@dataclass
class SignalBatch:
    """Batch of signals for bulk processing/export."""
    signals: List[UniversalSignal]
    batch_id: str = field(default_factory=lambda: datetime.utcnow().strftime('%Y%m%d%H%M%S'))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'batch_id': self.batch_id,
            'created_at': self.created_at,
            'source': self.source,
            'count': len(self.signals),
            'signals': [s.to_dict() for s in self.signals],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_ndjson(self) -> str:
        """Newline-delimited JSON for streaming."""
        return '\n'.join(s.to_json(indent=None) for s in self.signals)


# ============================================================
# Adapters for common formats
# ============================================================

def from_gdelt_article(article: Dict[str, Any]) -> UniversalSignal:
    """Convert GDELT article to UniversalSignal."""
    geo = None
    if 'sourcecountry' in article:
        geo = GeoReference(
            country_code=article.get('sourcecountry', '')[:2].upper(),
            precision='country'
        )

    # GDELT tone is -100 to +100, normalize to confidence
    tone = article.get('tone', 0)
    confidence_score = abs(tone) / 100  # Higher magnitude = more confident

    return UniversalSignal(
        id=f"gdelt:{hashlib.md5(article.get('url', '').encode()).hexdigest()[:12]}",
        signal_type=SignalType.NEWS,
        timestamp=_parse_gdelt_date(article.get('seendate', '')),
        geo=geo,
        title=article.get('title'),
        content=article.get('title'),  # GDELT only gives title
        value=tone,
        value_type='tone',
        confidence=ConfidenceLevel.MEDIUM,
        confidence_score=confidence_score,
        categories=article.get('themes', [])[:5],
        provenance=Provenance(
            source_id='gdelt',
            source_name='GDELT Project',
            source_tier=SourceTier.OPEN,
            source_url=article.get('url'),
            attribution='Data from GDELT Project',
            license='Open'
        ),
        raw=article
    )


def from_fred_observation(
    series_id: str,
    observation: Dict[str, Any],
    series_meta: Optional[Dict[str, Any]] = None
) -> UniversalSignal:
    """Convert FRED observation to UniversalSignal."""
    value = observation.get('value', '.')

    try:
        numeric_value = float(value)
    except (ValueError, TypeError):
        numeric_value = None

    return UniversalSignal(
        id=f"fred:{series_id}:{observation.get('date', '')}",
        signal_type=SignalType.INDICATOR,
        timestamp=observation.get('date', '') + 'T00:00:00Z',
        geo=GeoReference(country_code='US', precision='country'),
        title=f"{series_id}: {value}",
        value=numeric_value,
        value_type=series_id,
        unit=series_meta.get('units', '') if series_meta else '',
        confidence=ConfidenceLevel.CONFIRMED,
        confidence_score=1.0,
        categories=['economic', 'indicator'],
        provenance=Provenance(
            source_id='fred',
            source_name='Federal Reserve Economic Data',
            source_tier=SourceTier.OFFICIAL,
            attribution='Source: FRED, Federal Reserve Bank of St. Louis',
            license='Public Domain'
        ),
        raw=observation
    )


def from_world_bank(
    indicator_code: str,
    observation: Dict[str, Any]
) -> UniversalSignal:
    """Convert World Bank data to UniversalSignal."""
    return UniversalSignal(
        id=f"wb:{indicator_code}:{observation.get('country', {}).get('id', '')}:{observation.get('date', '')}",
        signal_type=SignalType.INDICATOR,
        timestamp=f"{observation.get('date', '')}-01-01T00:00:00Z",
        geo=GeoReference(
            country_code=observation.get('country', {}).get('id', ''),
            country_name=observation.get('country', {}).get('value', ''),
            precision='country'
        ),
        title=f"{indicator_code}: {observation.get('value')}",
        value=observation.get('value'),
        value_type=indicator_code,
        confidence=ConfidenceLevel.HIGH,
        confidence_score=0.9,
        categories=['economic', 'world_bank'],
        provenance=Provenance(
            source_id='world_bank',
            source_name='World Bank Open Data',
            source_tier=SourceTier.INSTITUTIONAL,
            attribution='World Bank Open Data',
            license='CC BY 4.0'
        ),
        raw=observation
    )


def from_latticeforge_alert(
    region: str,
    phase_data: Dict[str, Any],
    analysis_timestamp: str
) -> UniversalSignal:
    """Convert LatticeForge phase analysis to UniversalSignal."""
    return UniversalSignal(
        id=f"lf:alert:{region}:{analysis_timestamp}",
        signal_type=SignalType.ALERT,
        timestamp=analysis_timestamp,
        geo=GeoReference(country_code=region, precision='country'),
        title=f"Phase Transition Alert: {region}",
        value=phase_data.get('transition_probability'),
        value_type='phase_transition_probability',
        confidence=ConfidenceLevel.HIGH if phase_data.get('confidence', 0) > 0.7 else ConfidenceLevel.MEDIUM,
        confidence_score=phase_data.get('confidence', 0.5),
        categories=['phase_transition', 'alert'],
        tags=[phase_data.get('current_phase', 'unknown')],
        enrichments={
            'regime_probabilities': phase_data.get('regime', {}),
            'causal_drivers': phase_data.get('causal_drivers', []),
            'alert_level': phase_data.get('alert_level', 'unknown'),
        },
        provenance=Provenance(
            source_id='latticeforge',
            source_name='LatticeForge Analysis',
            source_tier=SourceTier.DERIVED,
            transformations=['phase_detection', 'dempster_shafer_fusion', 'value_clustering']
        )
    )


# ============================================================
# Helpers
# ============================================================

def _parse_gdelt_date(date_str: str) -> str:
    """Parse GDELT date format to ISO 8601."""
    if not date_str or len(date_str) < 8:
        return datetime.utcnow().isoformat() + 'Z'

    try:
        year = date_str[0:4]
        month = date_str[4:6]
        day = date_str[6:8]
        hour = date_str[8:10] if len(date_str) >= 10 else '00'
        minute = date_str[10:12] if len(date_str) >= 12 else '00'
        second = date_str[12:14] if len(date_str) >= 14 else '00'
        return f"{year}-{month}-{day}T{hour}:{minute}:{second}Z"
    except Exception:
        return datetime.utcnow().isoformat() + 'Z'
