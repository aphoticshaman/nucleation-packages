#!/usr/bin/env python3
"""
Convert raw GDELT BigQuery export to alpaca training format.
Maps EventCodes to human-readable descriptions and generates analysis prompts.
"""

import json
import sys
from datetime import datetime

# GDELT CAMEO Event Codes (top-level)
EVENT_CODES = {
    "01": "Make public statement",
    "02": "Appeal",
    "03": "Express intent to cooperate",
    "04": "Consult",
    "05": "Engage in diplomatic cooperation",
    "06": "Engage in material cooperation",
    "07": "Provide aid",
    "08": "Yield",
    "09": "Investigate",
    "10": "Demand",
    "11": "Disapprove",
    "12": "Reject",
    "13": "Threaten",
    "14": "Protest",
    "15": "Exhibit military posture",
    "16": "Reduce relations",
    "17": "Coerce",
    "18": "Assault",
    "19": "Fight",
    "20": "Engage in unconventional mass violence",
}

def get_event_description(code: str) -> str:
    """Map GDELT event code to description."""
    if not code:
        return "Unknown event"
    prefix = code[:2] if len(code) >= 2 else code
    return EVENT_CODES.get(prefix, f"Event type {code}")

def goldstein_to_risk(goldstein: float) -> str:
    """Convert Goldstein scale to risk assessment."""
    if goldstein <= -7:
        return "CRITICAL - Severe conflict/violence"
    elif goldstein <= -4:
        return "HIGH - Significant tensions/hostility"
    elif goldstein <= -1:
        return "ELEVATED - Moderate tensions"
    elif goldstein <= 1:
        return "NEUTRAL - Routine interactions"
    elif goldstein <= 4:
        return "LOW - Cooperative signals"
    else:
        return "POSITIVE - Strong cooperation"

def format_date(sqldate: str) -> str:
    """Format YYYYMMDD to readable date."""
    try:
        dt = datetime.strptime(sqldate, "%Y%m%d")
        return dt.strftime("%B %d, %Y")
    except:
        return sqldate

def convert_event(event: dict) -> dict:
    """Convert single GDELT event to alpaca format."""

    date = format_date(event.get("SQLDATE", ""))
    actor1 = event.get("Actor1Name") or "Unknown actor"
    actor2 = event.get("Actor2Name") or "unspecified target"
    event_code = event.get("EventCode", "")
    event_desc = get_event_description(event_code)
    goldstein = float(event.get("GoldsteinScale", 0))
    mentions = int(event.get("NumMentions", 0))
    tone = float(event.get("AvgTone", 0))
    country1 = event.get("Actor1CountryCode") or "UNK"
    country2 = event.get("Actor2CountryCode") or ""
    source = event.get("SOURCEURL", "")

    # Build the input description
    input_text = f"{date}: {actor1}"
    if actor2 and actor2 != "unspecified target":
        input_text += f" → {actor2}"
    input_text += f". Event: {event_desc} (Code {event_code}). "
    input_text += f"Goldstein Scale: {goldstein}. "
    input_text += f"Media mentions: {mentions}. Average tone: {tone:.2f}."
    if country2:
        input_text += f" Countries involved: {country1}, {country2}."

    # Build analysis output
    risk_level = goldstein_to_risk(goldstein)

    output_parts = [
        f"RISK ASSESSMENT: {risk_level}",
        "",
        f"1) EVENT CLASSIFICATION: {event_desc} between {actor1} and {actor2}. "
        f"Goldstein score of {goldstein} indicates {'hostile/conflictual' if goldstein < 0 else 'cooperative/neutral'} interaction.",
        "",
        f"2) MEDIA INTENSITY: {mentions} mentions with average tone {tone:.2f}. "
        f"{'High media attention amplifies potential for escalation.' if mentions > 500 else 'Moderate coverage suggests contained event.'} "
        f"{'Negative tone indicates critical/concerned coverage.' if tone < -3 else 'Tone suggests measured reporting.'}",
        "",
        f"3) GEOPOLITICAL CONTEXT: {country1}"
    ]

    if country2:
        output_parts[-1] += f"-{country2} relations. "
        if goldstein <= -5:
            output_parts[-1] += "Severe strain on bilateral relations. Monitor for escalation, sanctions, or military posturing."
        elif goldstein <= 0:
            output_parts[-1] += "Tensions present but within historical norms. Watch for pattern changes."
        else:
            output_parts[-1] += "Cooperative signal. May indicate diplomatic progress or alliance strengthening."

    output_parts.extend([
        "",
        f"4) CASCADE POTENTIAL: {'HIGH' if goldstein <= -7 and mentions > 500 else 'MODERATE' if goldstein <= -4 else 'LOW'}. "
        f"{'Mass violence events historically trigger refugee flows, market volatility, and international response.' if event_code in ['19', '20'] else ''}"
        f"{'Military posturing may escalate or serve as deterrent signaling.' if event_code in ['15', '17'] else ''}"
        f"{'Protest activity indicates domestic instability - monitor for regime response.' if event_code == '14' else ''}"
    ])

    output_parts.extend([
        "",
        f"5) RECOMMENDED MONITORING: Track {actor1} official statements, {country1} military movements, "
        f"{'and ' + country2 + ' response' if country2 else 'regional ally reactions'}. "
        f"Set alerts for follow-up events within 72-hour window."
    ])

    output_text = "\n".join(output_parts)

    return {
        "instruction": "Analyze the geopolitical risk signals in the following GDELT event data",
        "input": input_text,
        "output": output_text
    }

def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "training_data_from_bigquery.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "training_data_bigquery_alpaca.json"

    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Converting {len(data)} events...")
    converted = []

    for event in data:
        try:
            converted.append(convert_event(event))
        except Exception as e:
            print(f"Skipping event due to error: {e}")
            continue

    print(f"Writing {len(converted)} examples to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(converted, f, indent=2)

    print(f"Done! {len(converted)} training examples created.")
    print(f"\nSample output:")
    print(json.dumps(converted[0], indent=2)[:1000] + "...")

if __name__ == "__main__":
    main()
