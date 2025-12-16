"""
Export Adapters for LatticeForge.

Convert UniversalSignal to industry-standard formats for integration with:
- SIEM systems (Splunk, ArcSight, QRadar) via CEF/LEEF
- Threat intel platforms (Recorded Future, ThreatConnect) via STIX/TAXII
- General systems via JSON, CSV, NDJSON
- Webhooks for real-time push

References:
- STIX 2.1: https://docs.oasis-open.org/cti/stix/v2.1/stix-v2.1.html
- CEF: https://www.microfocus.com/documentation/arcsight/arcsight-smartconnectors-8.3/cef-implementation-standard/
- TAXII 2.1: https://docs.oasis-open.org/cti/taxii/v2.1/taxii-v2.1.html
"""

import json
import csv
import io
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import asdict

from .universal_signal import (
    UniversalSignal,
    SignalBatch,
    SignalType,
    ConfidenceLevel,
)


# ============================================================
# BASIC FORMATS (CSV, JSON, NDJSON)
# ============================================================

def to_json(
    signals: Union[UniversalSignal, List[UniversalSignal], SignalBatch],
    indent: int = 2,
    include_raw: bool = False
) -> str:
    """
    Export to JSON format.

    Args:
        signals: Single signal, list, or batch
        indent: JSON indentation (None for compact)
        include_raw: Whether to include raw source data

    Returns:
        JSON string
    """
    if isinstance(signals, SignalBatch):
        data = signals.to_dict()
    elif isinstance(signals, list):
        data = {
            'count': len(signals),
            'exported_at': datetime.utcnow().isoformat() + 'Z',
            'signals': [s.to_dict() for s in signals]
        }
    else:
        data = signals.to_dict()

    if not include_raw and 'signals' in data:
        for s in data['signals']:
            s.pop('raw', None)

    return json.dumps(data, indent=indent, default=str)


def to_ndjson(
    signals: Union[List[UniversalSignal], SignalBatch],
    include_raw: bool = False
) -> str:
    """
    Export to Newline-Delimited JSON (NDJSON) for streaming.

    This format is ideal for:
    - Kafka ingestion
    - Elasticsearch bulk API
    - Log aggregators
    - Large file processing

    Args:
        signals: List of signals or batch
        include_raw: Whether to include raw source data

    Returns:
        NDJSON string (one JSON object per line)
    """
    if isinstance(signals, SignalBatch):
        signal_list = signals.signals
    else:
        signal_list = signals

    lines = []
    for signal in signal_list:
        data = signal.to_dict()
        if not include_raw:
            data.pop('raw', None)
        lines.append(json.dumps(data, default=str))

    return '\n'.join(lines)


def to_csv(
    signals: Union[List[UniversalSignal], SignalBatch],
    columns: Optional[List[str]] = None
) -> str:
    """
    Export to CSV format for analyst workflows.

    Args:
        signals: List of signals or batch
        columns: Specific columns to include (default: common fields)

    Returns:
        CSV string
    """
    if isinstance(signals, SignalBatch):
        signal_list = signals.signals
    else:
        signal_list = signals

    if not signal_list:
        return ""

    # Default columns for analyst export
    if columns is None:
        columns = [
            'id', 'signal_type', 'timestamp', 'title',
            'country_code', 'value', 'value_type',
            'confidence', 'confidence_score',
            'categories', 'source_name'
        ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()

    for signal in signal_list:
        row = {
            'id': signal.id,
            'signal_type': signal.signal_type.value,
            'timestamp': signal.timestamp,
            'title': signal.title or '',
            'country_code': signal.geo.country_code if signal.geo else '',
            'value': signal.value if signal.value is not None else '',
            'value_type': signal.value_type or '',
            'confidence': signal.confidence.name.lower(),
            'confidence_score': signal.confidence_score,
            'categories': '|'.join(signal.categories) if signal.categories else '',
            'source_name': signal.provenance.source_name if signal.provenance else '',
        }
        writer.writerow(row)

    return output.getvalue()


# ============================================================
# STIX 2.1 (Threat Intelligence Standard)
# ============================================================

def to_stix_bundle(
    signals: Union[List[UniversalSignal], SignalBatch],
    bundle_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Export to STIX 2.1 Bundle format.

    STIX (Structured Threat Information Expression) is the standard
    for sharing cyber threat intelligence. Supported by:
    - Recorded Future
    - ThreatConnect
    - Microsoft Sentinel
    - Anomali
    - MISP

    Reference: https://docs.oasis-open.org/cti/stix/v2.1/stix-v2.1.html

    Args:
        signals: Signals to export
        bundle_id: Optional bundle identifier

    Returns:
        STIX 2.1 Bundle as dictionary
    """
    if isinstance(signals, SignalBatch):
        signal_list = signals.signals
        bundle_id = bundle_id or f"bundle--{signals.batch_id}"
    else:
        signal_list = signals
        bundle_id = bundle_id or f"bundle--{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    stix_objects = []

    for signal in signal_list:
        stix_obj = _signal_to_stix_object(signal)
        if stix_obj:
            stix_objects.append(stix_obj)

    return {
        "type": "bundle",
        "id": bundle_id,
        "objects": stix_objects
    }


def _signal_to_stix_object(signal: UniversalSignal) -> Optional[Dict[str, Any]]:
    """Convert a single signal to appropriate STIX object."""

    # Generate deterministic STIX ID
    stix_id_hash = hashlib.sha256(signal.id.encode()).hexdigest()[:8]

    # Map signal types to STIX types
    if signal.signal_type in [SignalType.THREAT, SignalType.ALERT]:
        return _to_stix_indicator(signal, stix_id_hash)
    elif signal.signal_type == SignalType.EVENT:
        return _to_stix_observed_data(signal, stix_id_hash)
    elif signal.signal_type in [SignalType.NEWS, SignalType.SOCIAL]:
        return _to_stix_report(signal, stix_id_hash)
    elif signal.signal_type == SignalType.CAUSAL:
        return _to_stix_relationship(signal, stix_id_hash)
    else:
        # Default to note for other types
        return _to_stix_note(signal, stix_id_hash)


def _to_stix_indicator(signal: UniversalSignal, id_suffix: str) -> Dict[str, Any]:
    """Convert alert/threat signal to STIX Indicator."""
    return {
        "type": "indicator",
        "spec_version": "2.1",
        "id": f"indicator--{id_suffix}",
        "created": signal.timestamp,
        "modified": signal.timestamp,
        "name": signal.title or f"LatticeForge Alert: {signal.id}",
        "description": signal.content or "",
        "indicator_types": ["anomalous-activity"],
        "pattern": f"[x-latticeforge:signal_id = '{signal.id}']",
        "pattern_type": "stix",
        "valid_from": signal.valid_from or signal.timestamp,
        "confidence": int(signal.confidence_score * 100),
        "labels": signal.categories or [],
        "external_references": [
            {
                "source_name": "LatticeForge",
                "external_id": signal.id,
            }
        ],
        "x_latticeforge_enrichments": signal.enrichments,
    }


def _to_stix_observed_data(signal: UniversalSignal, id_suffix: str) -> Dict[str, Any]:
    """Convert event signal to STIX Observed Data."""
    return {
        "type": "observed-data",
        "spec_version": "2.1",
        "id": f"observed-data--{id_suffix}",
        "created": signal.timestamp,
        "modified": signal.timestamp,
        "first_observed": signal.timestamp,
        "last_observed": signal.timestamp,
        "number_observed": 1,
        "object_refs": [],  # Would reference actual observables
        "x_latticeforge_signal": {
            "id": signal.id,
            "type": signal.signal_type.value,
            "value": signal.value,
            "confidence": signal.confidence_score,
        }
    }


def _to_stix_report(signal: UniversalSignal, id_suffix: str) -> Dict[str, Any]:
    """Convert news/social signal to STIX Report."""
    return {
        "type": "report",
        "spec_version": "2.1",
        "id": f"report--{id_suffix}",
        "created": signal.timestamp,
        "modified": signal.timestamp,
        "name": signal.title or f"Signal: {signal.id}",
        "description": signal.content or "",
        "published": signal.timestamp,
        "report_types": ["threat-report"],
        "labels": signal.categories or [],
        "confidence": int(signal.confidence_score * 100),
        "external_references": [
            {
                "source_name": signal.provenance.source_name if signal.provenance else "LatticeForge",
                "url": signal.provenance.source_url if signal.provenance else None,
            }
        ] if signal.provenance else [],
    }


def _to_stix_relationship(signal: UniversalSignal, id_suffix: str) -> Dict[str, Any]:
    """Convert causal signal to STIX Relationship."""
    return {
        "type": "relationship",
        "spec_version": "2.1",
        "id": f"relationship--{id_suffix}",
        "created": signal.timestamp,
        "modified": signal.timestamp,
        "relationship_type": "related-to",
        "source_ref": signal.related_signals[0] if signal.related_signals else f"unknown--{id_suffix}",
        "target_ref": signal.related_signals[1] if len(signal.related_signals) > 1 else f"unknown--{id_suffix}",
        "confidence": int(signal.confidence_score * 100),
        "x_latticeforge_causal_weight": signal.value,
    }


def _to_stix_note(signal: UniversalSignal, id_suffix: str) -> Dict[str, Any]:
    """Convert generic signal to STIX Note."""
    return {
        "type": "note",
        "spec_version": "2.1",
        "id": f"note--{id_suffix}",
        "created": signal.timestamp,
        "modified": signal.timestamp,
        "content": signal.content or signal.title or f"Signal {signal.id}",
        "abstract": signal.title,
        "labels": signal.categories or [],
        "x_latticeforge_signal_type": signal.signal_type.value,
        "x_latticeforge_value": signal.value,
    }


def to_stix_json(
    signals: Union[List[UniversalSignal], SignalBatch],
    indent: int = 2
) -> str:
    """Export to STIX 2.1 JSON string."""
    bundle = to_stix_bundle(signals)
    return json.dumps(bundle, indent=indent, default=str)


# ============================================================
# CEF (Common Event Format) for SIEM
# ============================================================

def to_cef(
    signals: Union[List[UniversalSignal], SignalBatch],
    device_vendor: str = "LatticeForge",
    device_product: str = "PhaseDetector",
    device_version: str = "1.0"
) -> str:
    """
    Export to CEF (Common Event Format) for SIEM integration.

    CEF is supported by:
    - ArcSight (creator of CEF)
    - Splunk
    - QRadar
    - LogRhythm
    - McAfee ESM

    Format:
    CEF:Version|Device Vendor|Device Product|Device Version|Event Class ID|Name|Severity|Extension

    Reference: https://www.microfocus.com/documentation/arcsight/arcsight-smartconnectors-8.3/cef-implementation-standard/

    Args:
        signals: Signals to export
        device_vendor: Vendor name
        device_product: Product name
        device_version: Product version

    Returns:
        CEF formatted string (one line per event)
    """
    if isinstance(signals, SignalBatch):
        signal_list = signals.signals
    else:
        signal_list = signals

    lines = []

    for signal in signal_list:
        cef_line = _signal_to_cef(
            signal,
            device_vendor,
            device_product,
            device_version
        )
        lines.append(cef_line)

    return '\n'.join(lines)


def _signal_to_cef(
    signal: UniversalSignal,
    device_vendor: str,
    device_product: str,
    device_version: str
) -> str:
    """Convert single signal to CEF line."""

    # Map confidence to severity (0-10)
    severity = int(signal.confidence_score * 10)

    # Event class ID based on signal type
    event_class_id = f"LF-{signal.signal_type.value.upper()}"

    # Name/signature
    name = (signal.title or signal.signal_type.value)[:63]  # CEF max 63 chars

    # Build extension key-value pairs
    extensions = []

    # Standard CEF extensions
    extensions.append(f"rt={_cef_timestamp(signal.timestamp)}")
    extensions.append(f"msg={_cef_escape(signal.content or signal.title or '')[:1023]}")

    if signal.geo:
        if signal.geo.country_code:
            extensions.append(f"dvc={signal.geo.country_code}")
        if signal.geo.lat and signal.geo.lon:
            extensions.append(f"dlat={signal.geo.lat}")
            extensions.append(f"dlong={signal.geo.lon}")

    if signal.value is not None:
        extensions.append(f"cn1={signal.value}")
        extensions.append(f"cn1Label={signal.value_type or 'value'}")

    extensions.append(f"cs1={signal.id}")
    extensions.append(f"cs1Label=signal_id")

    if signal.provenance:
        extensions.append(f"cs2={signal.provenance.source_name}")
        extensions.append(f"cs2Label=source")

    if signal.categories:
        extensions.append(f"cat={','.join(signal.categories[:5])}")

    # Build CEF line
    # CEF:Version|Device Vendor|Device Product|Device Version|Event Class ID|Name|Severity|Extension
    header = f"CEF:0|{_cef_escape(device_vendor)}|{_cef_escape(device_product)}|{device_version}|{event_class_id}|{_cef_escape(name)}|{severity}"
    extension = ' '.join(extensions)

    return f"{header}|{extension}"


def _cef_escape(s: str) -> str:
    """Escape special characters for CEF."""
    if not s:
        return ""
    return s.replace('\\', '\\\\').replace('|', '\\|').replace('=', '\\=').replace('\n', ' ')


def _cef_timestamp(iso_timestamp: str) -> str:
    """Convert ISO timestamp to CEF format (milliseconds since epoch)."""
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
        return str(int(dt.timestamp() * 1000))
    except Exception:
        return str(int(datetime.utcnow().timestamp() * 1000))


# ============================================================
# LEEF (Log Event Extended Format) for QRadar
# ============================================================

def to_leef(
    signals: Union[List[UniversalSignal], SignalBatch],
    vendor: str = "LatticeForge",
    product: str = "PhaseDetector",
    version: str = "1.0"
) -> str:
    """
    Export to LEEF (Log Event Extended Format) for IBM QRadar.

    LEEF is QRadar's native format, similar to CEF but with different syntax.

    Format:
    LEEF:Version|Vendor|Product|Version|EventID|Key1=Value1<tab>Key2=Value2

    Args:
        signals: Signals to export
        vendor: Vendor name
        product: Product name
        version: Product version

    Returns:
        LEEF formatted string
    """
    if isinstance(signals, SignalBatch):
        signal_list = signals.signals
    else:
        signal_list = signals

    lines = []

    for signal in signal_list:
        event_id = f"LF_{signal.signal_type.value}"

        attrs = []
        attrs.append(f"devTime={signal.timestamp}")
        attrs.append(f"severity={int(signal.confidence_score * 10)}")

        if signal.title:
            attrs.append(f"name={signal.title}")
        if signal.content:
            attrs.append(f"msg={signal.content[:500]}")
        if signal.geo and signal.geo.country_code:
            attrs.append(f"dstGeoCountryCode={signal.geo.country_code}")
        if signal.value is not None:
            attrs.append(f"customNumber1={signal.value}")

        attrs.append(f"externalId={signal.id}")

        # LEEF uses tab as separator
        header = f"LEEF:2.0|{vendor}|{product}|{version}|{event_id}|"
        body = '\t'.join(attrs)

        lines.append(header + body)

    return '\n'.join(lines)


# ============================================================
# Webhook Payload Builder
# ============================================================

def to_webhook_payload(
    signals: Union[UniversalSignal, List[UniversalSignal], SignalBatch],
    webhook_type: str = "generic",
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Build webhook payload for pushing to external systems.

    Supports multiple webhook formats:
    - generic: Standard JSON
    - slack: Slack webhook format
    - teams: Microsoft Teams webhook format
    - pagerduty: PagerDuty Events API format

    Args:
        signals: Signals to include
        webhook_type: Target webhook format
        include_metadata: Include batch metadata

    Returns:
        Dictionary ready for JSON serialization and POST
    """
    if isinstance(signals, SignalBatch):
        signal_list = signals.signals
    elif isinstance(signals, list):
        signal_list = signals
    else:
        signal_list = [signals]

    if webhook_type == "slack":
        return _to_slack_webhook(signal_list)
    elif webhook_type == "teams":
        return _to_teams_webhook(signal_list)
    elif webhook_type == "pagerduty":
        return _to_pagerduty_webhook(signal_list)
    else:
        # Generic webhook
        payload = {
            "source": "latticeforge",
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "count": len(signal_list),
            "signals": [s.to_dict() for s in signal_list]
        }
        return payload


def _to_slack_webhook(signals: List[UniversalSignal]) -> Dict[str, Any]:
    """Format for Slack incoming webhook."""
    blocks = []

    for signal in signals[:10]:  # Slack limits blocks
        severity_emoji = "🔴" if signal.confidence_score > 0.7 else "🟡" if signal.confidence_score > 0.4 else "🟢"

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{severity_emoji} *{signal.title or signal.signal_type.value}*\n{signal.content or ''}"[:3000]
            }
        })

        if signal.geo or signal.value:
            fields = []
            if signal.geo and signal.geo.country_code:
                fields.append({"type": "mrkdwn", "text": f"*Region:* {signal.geo.country_code}"})
            if signal.value is not None:
                fields.append({"type": "mrkdwn", "text": f"*Value:* {signal.value}"})
            if fields:
                blocks.append({"type": "section", "fields": fields})

    return {"blocks": blocks}


def _to_teams_webhook(signals: List[UniversalSignal]) -> Dict[str, Any]:
    """Format for Microsoft Teams webhook."""
    facts = []

    for signal in signals[:5]:
        facts.append({
            "name": signal.title or signal.signal_type.value,
            "value": f"Confidence: {signal.confidence_score:.0%}"
        })

    return {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": "FF0000" if any(s.confidence_score > 0.7 for s in signals) else "FFFF00",
        "summary": f"LatticeForge Alert: {len(signals)} signals",
        "sections": [{
            "activityTitle": "LatticeForge Phase Detection",
            "facts": facts,
            "markdown": True
        }]
    }


def _to_pagerduty_webhook(signals: List[UniversalSignal]) -> Dict[str, Any]:
    """Format for PagerDuty Events API v2."""
    # Use highest severity signal
    top_signal = max(signals, key=lambda s: s.confidence_score)

    severity = "critical" if top_signal.confidence_score > 0.7 else "warning" if top_signal.confidence_score > 0.4 else "info"

    return {
        "routing_key": "",  # User must fill in
        "event_action": "trigger",
        "dedup_key": top_signal.id,
        "payload": {
            "summary": top_signal.title or f"LatticeForge {top_signal.signal_type.value} Alert",
            "severity": severity,
            "source": "latticeforge",
            "timestamp": top_signal.timestamp,
            "custom_details": {
                "signal_id": top_signal.id,
                "confidence": top_signal.confidence_score,
                "region": top_signal.geo.country_code if top_signal.geo else None,
                "value": top_signal.value,
                "categories": top_signal.categories,
            }
        }
    }
