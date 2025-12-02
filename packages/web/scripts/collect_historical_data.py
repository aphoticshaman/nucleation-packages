#!/usr/bin/env python3
"""
Historical Data Collector for LatticeAI Fine-Tuning
Pulls GDELT events + World Bank indicators → Training data

Run: python collect_historical_data.py
Output: training_data_alpaca.json (ready for fine-tuning)

No API keys needed - all free public data.
"""

import json
import requests
import csv
import io
from datetime import datetime, timedelta
from collections import defaultdict
import time
import sys

# ============================================
# GDELT Data Collection
# ============================================

GDELT_EVENTS_URL = "http://data.gdeltproject.org/events/{date}.export.CSV.zip"
GDELT_GKG_URL = "http://data.gdeltproject.org/gkg/{date}.gkg.csv.zip"

# CAMEO event codes for geopolitical events
CAMEO_CODES = {
    "14": "PROTEST",
    "15": "FORCE_POSTURE",
    "17": "COERCE",
    "18": "ASSAULT",
    "19": "FIGHT",
    "20": "MASS_VIOLENCE",
    "10": "DEMAND",
    "11": "DISAPPROVE",
    "12": "REJECT",
    "13": "THREATEN",
}

def fetch_gdelt_mentions(query: str, days_back: int = 30, max_records: int = 250) -> list:
    """Fetch recent articles from GDELT DOC API (no auth needed)"""

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": str(max_records),
        "format": "json",
        "timespan": f"{days_back}d",
        "sort": "hybridrel"
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("articles", [])
    except Exception as e:
        print(f"  GDELT error for '{query}': {e}")

    return []


def collect_gdelt_data(countries: list, themes: list, days_back: int = 90) -> list:
    """Collect GDELT articles for countries and themes"""

    print(f"\n📡 Collecting GDELT data ({days_back} days back)...")

    all_articles = []

    # Country-specific queries
    for country in countries:
        print(f"  Fetching: {country}...")
        articles = fetch_gdelt_mentions(f"sourcecountry:{country}", days_back, 100)
        for a in articles:
            a["query_country"] = country
        all_articles.extend(articles)
        time.sleep(1)  # Rate limit

    # Theme queries
    for theme in themes:
        print(f"  Fetching theme: {theme}...")
        articles = fetch_gdelt_mentions(f"theme:{theme}", days_back, 100)
        for a in articles:
            a["query_theme"] = theme
        all_articles.extend(articles)
        time.sleep(1)

    print(f"  ✅ Collected {len(all_articles)} GDELT articles")
    return all_articles


# ============================================
# World Bank Data Collection
# ============================================

WB_API = "https://api.worldbank.org/v2"

WB_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "FP.CPI.TOTL.ZG": "inflation",
    "SL.UEM.TOTL.ZS": "unemployment",
    "GC.DOD.TOTL.GD.ZS": "debt_to_gdp",
    "BN.CAB.XOKA.GD.ZS": "current_account",
}


def fetch_worldbank_indicator(indicator: str, countries: str = "all", years: str = "2015:2024") -> list:
    """Fetch World Bank indicator data"""

    url = f"{WB_API}/country/{countries}/indicator/{indicator}"
    params = {
        "format": "json",
        "per_page": 1000,
        "date": years,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if len(data) > 1 and data[1]:
                return data[1]
    except Exception as e:
        print(f"  WB error for {indicator}: {e}")

    return []


def collect_worldbank_data(countries: list) -> dict:
    """Collect World Bank economic data"""

    print(f"\n📊 Collecting World Bank data...")

    country_data = defaultdict(lambda: defaultdict(dict))
    countries_str = ";".join(countries)

    for indicator_id, indicator_name in WB_INDICATORS.items():
        print(f"  Fetching: {indicator_name}...")
        data = fetch_worldbank_indicator(indicator_id, countries_str)

        for record in data:
            if record.get("value") is not None:
                country = record["country"]["id"]
                year = record["date"]
                country_data[country][year][indicator_name] = record["value"]

        time.sleep(0.5)

    print(f"  ✅ Collected data for {len(country_data)} countries")
    return dict(country_data)


# ============================================
# Crisis/Conflict Historical Events
# ============================================

HISTORICAL_CRISES = [
    # Format: (year, country, event, outcome, risk_level)
    (2022, "RUS", "Invasion of Ukraine begins", "Ongoing war, sanctions, energy crisis", "CRITICAL"),
    (2022, "UKR", "Russian invasion", "Territorial losses, humanitarian crisis", "CRITICAL"),
    (2023, "ISR", "October 7 Hamas attack", "Gaza war, regional escalation", "CRITICAL"),
    (2023, "SYR", "Continued civil war", "Assad regime weakening, Turkish intervention", "HIGH"),
    (2024, "SYR", "Assad regime collapses", "Power vacuum, HTS takeover", "CRITICAL"),
    (2024, "BGD", "Sheikh Hasina ousted", "Political transition, instability", "HIGH"),
    (2024, "KOR", "Martial law declared/reversed", "Political crisis, constitutional tensions", "HIGH"),
    (2024, "VEN", "Disputed election", "Maduro claims victory, protests", "HIGH"),
    (2024, "FRA", "Government collapses", "No-confidence vote, political chaos", "ELEVATED"),
    (2023, "NER", "Military coup", "Junta takes power, ECOWAS tensions", "HIGH"),
    (2023, "GAB", "Military coup", "Bongo dynasty ends", "HIGH"),
    (2023, "SDN", "Civil war erupts", "RSF vs SAF, humanitarian disaster", "CRITICAL"),
    (2022, "LKA", "Economic collapse", "President flees, IMF bailout", "CRITICAL"),
    (2022, "PAK", "Imran Khan ousted", "Political instability, economic crisis", "HIGH"),
    (2021, "AFG", "Taliban takeover", "US withdrawal, regime change", "CRITICAL"),
    (2021, "MMR", "Military coup", "Democracy ends, civil war begins", "CRITICAL"),
    (2020, "BLR", "Election protests", "Lukashenko crackdown, sanctions", "HIGH"),
    (2020, "LBN", "Beirut explosion", "Government collapse, economic crisis", "CRITICAL"),
    (2019, "HKG", "Democracy protests", "China crackdown, autonomy eroded", "HIGH"),
    (2019, "ECU", "Fuel subsidy protests", "Government reversal, instability", "ELEVATED"),
    (2019, "CHL", "Social uprising", "Constitutional reform process", "ELEVATED"),
    (2018, "VEN", "Hyperinflation crisis", "Mass migration, humanitarian emergency", "CRITICAL"),
    (2014, "UKR", "Crimea annexation", "War in Donbas begins", "CRITICAL"),
    (2011, "SYR", "Civil war begins", "Assad vs rebels, ISIS emerges", "CRITICAL"),
    (2011, "LBY", "Gaddafi overthrown", "State collapse, ongoing civil war", "CRITICAL"),
    (2011, "EGY", "Arab Spring revolution", "Mubarak ousted, military rule returns", "HIGH"),
    (2011, "TUN", "Jasmine Revolution", "Democratic transition", "ELEVATED"),
    (2011, "YEM", "Civil war begins", "Houthi takeover, Saudi intervention", "CRITICAL"),
]


# ============================================
# Convert to Training Format
# ============================================

def gdelt_to_training(articles: list) -> list:
    """Convert GDELT articles to training examples"""

    examples = []

    for article in articles:
        tone = article.get("tone", 0)
        country = article.get("sourcecountry", article.get("query_country", "Unknown"))
        title = article.get("title", "")

        if not title:
            continue

        # Classify risk from tone
        if tone < -5:
            risk = "HIGH"
            analysis = "Strongly negative media coverage indicates elevated tensions."
        elif tone < -2:
            risk = "ELEVATED"
            analysis = "Moderately negative coverage suggests developing concerns."
        elif tone < 2:
            risk = "MODERATE"
            analysis = "Neutral coverage indicates stable conditions."
        else:
            risk = "LOW"
            analysis = "Positive coverage suggests improving conditions."

        example = {
            "instruction": "Analyze this news headline and assess geopolitical risk level for the mentioned country or region.",
            "input": f"Headline: {title}\nSource Country: {country}\nMedia Tone Score: {tone:.1f}",
            "output": f"Risk Assessment: {risk}\n\nAnalysis: {analysis} Media tone of {tone:.1f} {'indicates concerning developments' if tone < -2 else 'suggests stability'}. Recommend {'close monitoring' if risk in ['HIGH', 'CRITICAL'] else 'standard tracking'}."
        }
        examples.append(example)

    return examples


def economic_to_training(country_data: dict) -> list:
    """Convert World Bank economic data to training examples"""

    examples = []

    for country, years in country_data.items():
        # Get most recent year with data
        sorted_years = sorted(years.keys(), reverse=True)
        if not sorted_years:
            continue

        recent = years[sorted_years[0]]

        # Calculate risk factors
        risk_factors = []
        risk_score = 0

        if recent.get("inflation", 0) > 10:
            risk_factors.append(f"High inflation ({recent['inflation']:.1f}%)")
            risk_score += 2
        if recent.get("unemployment", 0) > 15:
            risk_factors.append(f"High unemployment ({recent['unemployment']:.1f}%)")
            risk_score += 2
        if recent.get("debt_to_gdp", 0) > 100:
            risk_factors.append(f"High debt ({recent['debt_to_gdp']:.1f}% of GDP)")
            risk_score += 1
        if recent.get("gdp_growth", 0) < 0:
            risk_factors.append(f"Negative growth ({recent['gdp_growth']:.1f}%)")
            risk_score += 2
        if recent.get("current_account", 0) < -5:
            risk_factors.append(f"Large current account deficit ({recent['current_account']:.1f}%)")
            risk_score += 1

        if risk_score >= 4:
            risk = "HIGH"
        elif risk_score >= 2:
            risk = "ELEVATED"
        elif risk_score >= 1:
            risk = "MODERATE"
        else:
            risk = "LOW"

        indicators_str = "\n".join([f"- {k}: {v:.1f}" for k, v in recent.items() if v is not None])

        example = {
            "instruction": "Analyze these economic indicators and assess economic stability risk.",
            "input": f"Country: {country}\nYear: {sorted_years[0]}\nIndicators:\n{indicators_str}",
            "output": f"Economic Risk: {risk}\n\nKey Factors: {'; '.join(risk_factors) if risk_factors else 'No major risk factors identified'}\n\nAssessment: {'Economic stress indicators suggest potential for instability.' if risk in ['HIGH', 'ELEVATED'] else 'Economic fundamentals appear stable.'}"
        }
        examples.append(example)

    return examples


def crises_to_training(crises: list) -> list:
    """Convert historical crises to training examples"""

    examples = []

    for year, country, event, outcome, risk in crises:
        example = {
            "instruction": "Analyze this historical geopolitical event and explain the risk assessment.",
            "input": f"Year: {year}\nCountry: {country}\nEvent: {event}",
            "output": f"Risk Level: {risk}\n\nOutcome: {outcome}\n\nAnalysis: This event represents a {'critical threat to regime stability' if risk == 'CRITICAL' else 'significant political disruption' if risk == 'HIGH' else 'notable but contained development'}. Historical pattern suggests {'high probability of cascading effects' if risk in ['CRITICAL', 'HIGH'] else 'limited regional impact'}."
        }
        examples.append(example)

    return examples


def generate_synthetic_scenarios() -> list:
    """Generate synthetic training scenarios"""

    scenarios = []

    # Template-based generation
    templates = [
        {
            "input_template": "Country: {country}\nBasin Strength: {basin:.2f}\nTransition Risk: {risk:.2f}\nRecent Events: {events}",
            "countries": ["USA", "CHN", "RUS", "DEU", "BRA", "IND", "NGA", "EGY", "IRN", "TUR", "MEX", "ARG"],
        }
    ]

    import random
    random.seed(42)

    events_pool = [
        "Political protests in capital",
        "Currency devaluation announced",
        "Military exercises near border",
        "Trade negotiations stalled",
        "Opposition leader arrested",
        "Central bank raises rates",
        "Diplomatic incident reported",
        "Economic sanctions imposed",
        "Coalition government formed",
        "Constitutional amendment proposed",
    ]

    for country in templates[0]["countries"]:
        for _ in range(5):  # 5 scenarios per country
            basin = random.uniform(0.2, 0.9)
            risk = random.uniform(0.1, 0.8)
            events = random.choice(events_pool)

            if risk > 0.7:
                level = "CRITICAL"
                rec = "Immediate escalation protocols recommended."
            elif risk > 0.5:
                level = "HIGH"
                rec = "Enhanced monitoring required."
            elif risk > 0.3:
                level = "ELEVATED"
                rec = "Increased attention warranted."
            else:
                level = "LOW"
                rec = "Standard monitoring sufficient."

            scenario = {
                "instruction": "Assess the geopolitical risk for this country based on the provided indicators.",
                "input": f"Country: {country}\nBasin Strength: {basin:.2f}\nTransition Risk: {risk:.2f}\nRecent Events: {events}",
                "output": f"Risk Level: {level}\n\nBasin Analysis: {'Weak stability foundation' if basin < 0.4 else 'Moderate resilience' if basin < 0.7 else 'Strong institutional stability'}.\n\nTransition Probability: {risk*100:.0f}%\n\nRecommendation: {rec}"
            }
            scenarios.append(scenario)

    return scenarios


# ============================================
# Main Collection Pipeline
# ============================================

def main():
    print("=" * 60)
    print("🌍 LatticeAI Historical Data Collector")
    print("=" * 60)

    # Countries to focus on
    focus_countries = [
        "US", "CN", "RU", "DE", "GB", "FR", "JP", "IN", "BR", "AU",
        "KR", "SA", "IR", "IL", "TR", "MX", "ID", "NG", "ZA", "EG",
        "UA", "SY", "VE", "PK", "BD", "MM", "HT", "LB", "SD", "YE",
        "AF", "LY", "SO", "CD", "CF", "ET", "KE", "CO", "AR", "CL",
    ]

    # Themes for GDELT
    themes = [
        "MILITARY", "PROTEST", "TERROR", "ECON_BANKRUPTCY",
        "CRISISLEX_CRISISLEXREC", "LEADER", "ARREST", "COUP",
    ]

    all_training_data = []

    # 1. Collect GDELT articles
    gdelt_articles = collect_gdelt_data(focus_countries[:20], themes, days_back=60)
    gdelt_examples = gdelt_to_training(gdelt_articles)
    all_training_data.extend(gdelt_examples)
    print(f"  → {len(gdelt_examples)} examples from GDELT")

    # 2. Collect World Bank data
    wb_data = collect_worldbank_data(focus_countries)
    wb_examples = economic_to_training(wb_data)
    all_training_data.extend(wb_examples)
    print(f"  → {len(wb_examples)} examples from World Bank")

    # 3. Add historical crises
    crisis_examples = crises_to_training(HISTORICAL_CRISES)
    all_training_data.extend(crisis_examples)
    print(f"  → {len(crisis_examples)} examples from historical crises")

    # 4. Generate synthetic scenarios
    synthetic = generate_synthetic_scenarios()
    all_training_data.extend(synthetic)
    print(f"  → {len(synthetic)} synthetic examples")

    # Deduplicate by input
    seen = set()
    unique_data = []
    for ex in all_training_data:
        key = ex["input"][:100]
        if key not in seen:
            seen.add(key)
            unique_data.append(ex)

    print(f"\n✅ Total unique examples: {len(unique_data)}")

    # Save outputs
    output_file = "training_data_alpaca.json"
    with open(output_file, "w") as f:
        json.dump(unique_data, f, indent=2)
    print(f"📁 Saved to: {output_file}")

    # Also save ChatML format
    chatml_data = [
        {
            "messages": [
                {"role": "system", "content": ex["instruction"]},
                {"role": "user", "content": ex["input"]},
                {"role": "assistant", "content": ex["output"]},
            ]
        }
        for ex in unique_data
    ]
    with open("training_data_chatml.json", "w") as f:
        json.dump(chatml_data, f, indent=2)
    print(f"📁 Saved to: training_data_chatml.json")

    print("\n" + "=" * 60)
    print("🎯 Ready for fine-tuning!")
    print(f"   Examples: {len(unique_data)}")
    print(f"   Run: python train.py training_data_alpaca.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
