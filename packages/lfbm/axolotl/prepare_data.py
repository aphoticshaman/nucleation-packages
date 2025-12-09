#!/usr/bin/env python3
"""
Convert LatticeForge training data to Axolotl ShareGPT format.

Input: JSONL with our custom format
Output: JSONL in ShareGPT/ChatML format for Axolotl

Usage:
    python prepare_data.py input.jsonl output.jsonl
"""

import json
import sys
from typing import Dict, List
import random


def build_system_prompt() -> str:
    """System prompt that teaches the model its role"""
    return """You are a prose translation engine for an intelligence pipeline.

Your job: Convert numerical metrics into professional intelligence briefings.

Input format:
- Nation risk data (country code, risk score 0-1, trend)
- Signal data (GDELT article counts, sentiment tones)
- Category risk levels (political, economic, security, etc.)

Output format: JSON with briefings for each category.

Rules:
1. Reference the SPECIFIC metrics provided
2. Use professional intelligence analyst voice
3. Output valid JSON only
4. Do not fabricate events - describe what the NUMBERS indicate"""


def convert_example(example: Dict) -> Dict:
    """Convert one example to ShareGPT format"""

    # Build the user message (metrics input)
    nations = example.get('input_nations', [])
    signals = example.get('input_signals', {})
    categories = example.get('input_categories', {})

    # Format nations
    nation_lines = []
    for n in nations[:10]:  # Limit to top 10
        code = n.get('code', 'UNK')
        risk = n.get('risk', 0.5)
        trend = n.get('trend', 0)
        trend_str = '↑' if trend > 0 else '↓' if trend < 0 else '→'
        nation_lines.append(f"  {code}: risk={risk:.0%} {trend_str}")

    # Format signals
    signal_lines = []
    for k, v in signals.items():
        if isinstance(v, (int, float)):
            signal_lines.append(f"  {k}: {v}")

    # Format categories
    cat_lines = []
    for k, v in categories.items():
        if isinstance(v, (int, float)):
            cat_lines.append(f"  {k}: {v}/100")

    user_content = f"""PIPELINE METRICS (translate to briefings):

NATIONS:
{chr(10).join(nation_lines) if nation_lines else '  No nation data'}

SIGNALS:
{chr(10).join(signal_lines) if signal_lines else '  No signal data'}

CATEGORY RISKS:
{chr(10).join(cat_lines) if cat_lines else '  No category data'}

Generate JSON briefings for each category."""

    # Build the assistant response (expected output)
    output = example.get('output_briefings', {})
    assistant_content = json.dumps(output, indent=2)

    # ShareGPT format
    return {
        "conversations": [
            {"from": "system", "value": build_system_prompt()},
            {"from": "human", "value": user_content},
            {"from": "gpt", "value": assistant_content}
        ]
    }


def generate_synthetic_examples(num_examples: int = 1000) -> List[Dict]:
    """Generate synthetic training examples"""

    countries = [
        ('USA', 'United States', 0.15), ('CHN', 'China', 0.30),
        ('RUS', 'Russia', 0.55), ('UKR', 'Ukraine', 0.85),
        ('ISR', 'Israel', 0.60), ('IRN', 'Iran', 0.65),
        ('GBR', 'United Kingdom', 0.20), ('FRA', 'France', 0.25),
        ('DEU', 'Germany', 0.20), ('JPN', 'Japan', 0.15),
        ('IND', 'India', 0.35), ('BRA', 'Brazil', 0.40),
        ('SAU', 'Saudi Arabia', 0.30), ('TUR', 'Turkey', 0.45),
        ('TWN', 'Taiwan', 0.55), ('KOR', 'South Korea', 0.25),
        ('PRK', 'North Korea', 0.75), ('VEN', 'Venezuela', 0.70),
        ('SYR', 'Syria', 0.90), ('YEM', 'Yemen', 0.85),
    ]

    risk_words = {
        (0, 0.3): ['stable', 'low', 'minimal', 'contained'],
        (0.3, 0.5): ['moderate', 'elevated', 'notable', 'increasing'],
        (0.5, 0.7): ['high', 'significant', 'concerning', 'escalating'],
        (0.7, 1.0): ['critical', 'severe', 'extreme', 'crisis-level'],
    }

    def get_risk_word(risk: float) -> str:
        for (lo, hi), words in risk_words.items():
            if lo <= risk < hi:
                return random.choice(words)
        return 'elevated'

    examples = []

    for _ in range(num_examples):
        # Random nation selection
        num_nations = random.randint(5, 12)
        selected = random.sample(countries, num_nations)

        # Add some noise to base risks
        nations = []
        for code, name, base_risk in selected:
            risk = max(0, min(1, base_risk + random.uniform(-0.15, 0.15)))
            trend = random.choice([-0.1, -0.05, 0, 0.05, 0.1])
            nations.append({'code': code, 'name': name, 'risk': risk, 'trend': trend})

        # Random signals
        signals = {
            'gdelt_count': random.randint(20, 300),
            'avg_tone': random.uniform(-8, 3),
            'alert_count': random.randint(0, 25),
        }

        # Random categories
        categories = {
            'political': random.randint(25, 90),
            'economic': random.randint(25, 85),
            'security': random.randint(30, 90),
            'military': random.randint(20, 85),
            'financial': random.randint(25, 80),
            'cyber': random.randint(20, 75),
        }

        # Generate realistic output
        avg_risk = sum(n['risk'] for n in nations) / len(nations)
        high_risk_nations = [n for n in nations if n['risk'] > 0.5]
        risk_level = get_risk_word(avg_risk)

        # Build output briefings
        output = {
            'political': f"Risk indicators show {risk_level} political stability. {len(high_risk_nations)} nations exceed 50% threshold. Key concerns: {', '.join(n['code'] for n in high_risk_nations[:3]) or 'None critical'}.",

            'economic': f"Economic metrics at {get_risk_word(categories['economic']/100)} levels across {num_nations} monitored economies. Average category risk: {categories['economic']}%. {'Inflationary pressures detected.' if categories['economic'] > 60 else 'Markets within normal parameters.'}",

            'security': f"Security posture {risk_level}. {signals['alert_count']} active monitoring alerts. GDELT sentiment averaging {signals['avg_tone']:.1f} across {signals['gdelt_count']} articles analyzed.",

            'military': f"Military situation assessment: {get_risk_word(categories['military']/100)}. {'Elevated activity in ' + high_risk_nations[0]['code'] if high_risk_nations else 'No critical developments'}. Monitoring {num_nations} theaters.",

            'financial': f"Financial stability index: {100 - categories['financial']}%. {'Stress indicators elevated.' if categories['financial'] > 60 else 'Credit conditions normal.'} {num_nations} markets under surveillance.",

            'cyber': f"Cyber threat level: {get_risk_word(categories['cyber']/100)}. {'Increased activity detected.' if categories['cyber'] > 50 else 'Baseline threat environment.'} Monitoring {signals['gdelt_count']} open source signals.",

            'summary': f"Global assessment: {risk_level.upper()}. Monitoring {num_nations} nations with {len(high_risk_nations)} at elevated risk. Signal density: {'HIGH' if signals['gdelt_count'] > 150 else 'MODERATE' if signals['gdelt_count'] > 75 else 'LOW'}.",

            'nsm': f"Recommended action: {'Increase monitoring of ' + ', '.join(n['code'] for n in high_risk_nations[:2]) if high_risk_nations else 'Maintain standard monitoring posture'}. {'Review contingency plans.' if avg_risk > 0.5 else 'No immediate action required.'}"
        }

        examples.append({
            'input_nations': nations,
            'input_signals': signals,
            'input_categories': categories,
            'output_briefings': output,
        })

    return examples


def main():
    if len(sys.argv) < 2:
        print("Usage: python prepare_data.py [input.jsonl] output.jsonl")
        print("       If no input file, generates synthetic data")
        sys.exit(1)

    output_file = sys.argv[-1]

    # Load or generate examples
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        print(f"Loading from {input_file}...")
        examples = []
        with open(input_file, 'r') as f:
            for line in f:
                examples.append(json.loads(line))
        print(f"Loaded {len(examples)} examples")
    else:
        print("Generating synthetic training data...")
        examples = generate_synthetic_examples(3000)
        print(f"Generated {len(examples)} synthetic examples")

    # Convert to ShareGPT format
    print("Converting to Axolotl format...")
    converted = [convert_example(ex) for ex in examples]

    # Shuffle
    random.shuffle(converted)

    # Save
    with open(output_file, 'w') as f:
        for ex in converted:
            f.write(json.dumps(ex) + '\n')

    print(f"Saved {len(converted)} examples to {output_file}")

    # Show sample
    print("\n--- Sample training example ---")
    sample = converted[0]
    print("System:", sample['conversations'][0]['value'][:200] + "...")
    print("\nUser:", sample['conversations'][1]['value'][:300] + "...")
    print("\nAssistant:", sample['conversations'][2]['value'][:300] + "...")


if __name__ == '__main__':
    main()
