"""
Extract training data from Supabase for LFBM fine-tuning.

Sources:
1. learning_events table - historical LLM interactions
2. nations table - risk metrics
3. country_signals table - economic indicators

Output format:
{
    "input": {
        "nations": [{"code": "USA", "risk": 0.3, "trend": 0.1}, ...],
        "signals": {"gdelt_count": 45, "avg_tone": -2.3},
        "categories": {"political": 72, "economic": 45, ...}
    },
    "output": {
        "political": "Risk indicators show...",
        "economic": "Economic metrics at...",
        ...
    }
}
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import asyncio

# For Supabase connection
try:
    from supabase import create_client, Client
except ImportError:
    print("pip install supabase")
    create_client = None


@dataclass
class TrainingExample:
    """Single training example for LFBM"""
    input_nations: List[Dict]
    input_signals: Dict[str, float]
    input_categories: Dict[str, float]
    output_briefings: Dict[str, str]
    timestamp: str
    source: str  # 'claude', 'human', 'synthetic'
    quality_score: Optional[float] = None


class TrainingDataExtractor:
    """Extract and format training data from Supabase"""

    def __init__(self, supabase_url: str, supabase_key: str):
        if create_client is None:
            raise ImportError("supabase not installed")
        self.client: Client = create_client(supabase_url, supabase_key)
        self.examples: List[TrainingExample] = []

    async def extract_learning_events(self, days_back: int = 90) -> List[TrainingExample]:
        """Extract from learning_events table (Claude interactions)"""
        cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

        # Get successful LLM interactions
        result = self.client.table('learning_events').select('*').eq(
            'type', 'llm_interaction'
        ).gte('timestamp', cutoff).execute()

        examples = []
        for row in result.data or []:
            try:
                data = row.get('data', {})
                metadata = row.get('metadata', {})

                # Extract input context
                input_context = metadata.get('input_context', {})
                if not input_context:
                    continue

                # Extract output briefings
                output = metadata.get('output', {})
                if not output or not isinstance(output, dict):
                    continue

                example = TrainingExample(
                    input_nations=input_context.get('nations', []),
                    input_signals=input_context.get('signals', {}),
                    input_categories=input_context.get('categories', {}),
                    output_briefings=output,
                    timestamp=row.get('timestamp', ''),
                    source='claude',
                    quality_score=metadata.get('user_rating'),
                )
                examples.append(example)

            except Exception as e:
                print(f"Error parsing row: {e}")
                continue

        return examples

    async def extract_nation_snapshots(self) -> Dict[str, Dict]:
        """Get current nation state vectors for context"""
        result = self.client.table('nations').select(
            'code, name, basin_strength, transition_risk, regime'
        ).execute()

        nations = {}
        for row in result.data or []:
            code = row.get('code')
            if code:
                nations[code] = {
                    'name': row.get('name', ''),
                    'basin_strength': row.get('basin_strength', 0),
                    'transition_risk': row.get('transition_risk', 0),
                    'regime': row.get('regime', 0),
                }
        return nations

    async def generate_synthetic_examples(
        self,
        nation_data: Dict[str, Dict],
        num_examples: int = 1000
    ) -> List[TrainingExample]:
        """
        Generate synthetic training examples using templates.
        This bootstraps the model before we have enough real data.
        """
        import random

        templates = {
            'political': [
                "Risk indicators show {risk_level} political stability across {region}. {nation} leads concerns with {risk_pct}% transition risk.",
                "Political metrics at {risk_level} levels for {num_nations} monitored nations. Key watchpoints: {nation_list}.",
                "Assessment based on current readings shows {trend} political trajectory. {detail}",
            ],
            'economic': [
                "Economic indicators from {num_signals} recent data points suggest {outlook}. Average inflation: {inflation}%.",
                "Market conditions {trend}. GDP indicators: {detail}. {num_nations} economies under monitoring.",
                "Financial stability metrics at {risk_level}. {nation} showing {specific_concern}.",
            ],
            'security': [
                "Security posture {status} across {region}. {num_alerts} active monitoring alerts.",
                "Threat assessment: {risk_level}. Key concerns in {nation_list}. Overall stability: {stability_pct}%.",
                "Defense indicators {trend}. Military activity elevated in {nation}. {detail}",
            ],
            'summary': [
                "Global risk assessment: {risk_level}. Monitoring {num_nations} nations with {high_risk_count} at elevated risk.",
                "{region} stability index at {stability_pct}%. {key_concern}",
                "Current assessment: {status}. {num_alerts} active situations requiring attention.",
            ],
        }

        risk_levels = ['stable', 'moderate', 'elevated', 'high', 'critical']
        trends = ['improving', 'stable', 'deteriorating', 'volatile']
        statuses = ['stable', 'monitoring', 'elevated', 'concerning', 'critical']

        examples = []
        nation_codes = list(nation_data.keys())

        for _ in range(num_examples):
            # Random nation selection
            num_nations = random.randint(5, 20)
            selected_codes = random.sample(nation_codes, min(num_nations, len(nation_codes)))

            # Build input
            nations_input = []
            for code in selected_codes:
                nation = nation_data.get(code, {})
                nations_input.append({
                    'code': code,
                    'name': nation.get('name', code),
                    'risk': nation.get('transition_risk', random.random()),
                    'trend': random.choice([-0.1, 0, 0.1]),
                })

            # Random signals
            signals = {
                'gdelt_count': random.randint(10, 200),
                'avg_tone': random.uniform(-5, 2),
                'alert_count': random.randint(0, 20),
                'signal_density': random.choice(['low', 'moderate', 'high']),
            }

            # Random category risks
            categories = {
                'political': random.randint(20, 90),
                'economic': random.randint(20, 90),
                'security': random.randint(20, 90),
                'military': random.randint(20, 90),
                'financial': random.randint(20, 90),
                'cyber': random.randint(20, 90),
            }

            # Generate output using templates
            avg_risk = sum(categories.values()) / len(categories)
            risk_level = risk_levels[min(int(avg_risk / 20), 4)]
            high_risk_nations = [n for n in nations_input if n['risk'] > 0.5]

            output = {}
            for category, template_list in templates.items():
                template = random.choice(template_list)
                try:
                    output[category] = template.format(
                        risk_level=risk_level,
                        region=random.choice(['Global', 'EMEA', 'APAC', 'Americas']),
                        nation=random.choice(selected_codes) if selected_codes else 'Unknown',
                        nation_list=', '.join(selected_codes[:3]),
                        risk_pct=random.randint(50, 95),
                        num_nations=num_nations,
                        num_signals=signals['gdelt_count'],
                        trend=random.choice(trends),
                        outlook=random.choice(['stable', 'uncertain', 'improving']),
                        inflation=round(random.uniform(2, 15), 1),
                        detail=f"{random.choice(['Monitoring', 'Tracking', 'Analyzing'])} key indicators.",
                        status=random.choice(statuses),
                        stability_pct=random.randint(40, 95),
                        num_alerts=signals['alert_count'],
                        high_risk_count=len(high_risk_nations),
                        specific_concern=random.choice(['inflation pressure', 'currency volatility', 'debt concerns']),
                        key_concern=f"{len(high_risk_nations)} nations at elevated risk" if high_risk_nations else "No critical concerns",
                    )
                except KeyError:
                    output[category] = f"{category.title()} metrics at {risk_level} level."

            example = TrainingExample(
                input_nations=nations_input,
                input_signals=signals,
                input_categories=categories,
                output_briefings=output,
                timestamp=datetime.utcnow().isoformat(),
                source='synthetic',
                quality_score=0.5,  # Lower weight for synthetic
            )
            examples.append(example)

        return examples

    def save_dataset(self, examples: List[TrainingExample], output_path: str):
        """Save training data as JSONL"""
        with open(output_path, 'w') as f:
            for ex in examples:
                f.write(json.dumps(asdict(ex)) + '\n')
        print(f"Saved {len(examples)} examples to {output_path}")

    def load_dataset(self, input_path: str) -> List[TrainingExample]:
        """Load training data from JSONL"""
        examples = []
        with open(input_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                examples.append(TrainingExample(**data))
        return examples


async def main():
    """Extract and prepare training data"""
    # Get credentials from environment
    supabase_url = os.environ.get('NEXT_PUBLIC_SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')

    if not supabase_url or not supabase_key:
        print("Missing Supabase credentials, generating synthetic data only")

        # Generate synthetic examples for bootstrapping
        extractor = None
        synthetic_nations = {
            'USA': {'name': 'United States', 'transition_risk': 0.15, 'regime': 1},
            'CHN': {'name': 'China', 'transition_risk': 0.25, 'regime': 4},
            'RUS': {'name': 'Russia', 'transition_risk': 0.45, 'regime': 4},
            'UKR': {'name': 'Ukraine', 'transition_risk': 0.85, 'regime': 2},
            'ISR': {'name': 'Israel', 'transition_risk': 0.55, 'regime': 1},
            'IRN': {'name': 'Iran', 'transition_risk': 0.65, 'regime': 4},
            'GBR': {'name': 'United Kingdom', 'transition_risk': 0.20, 'regime': 1},
            'FRA': {'name': 'France', 'transition_risk': 0.30, 'regime': 1},
            'DEU': {'name': 'Germany', 'transition_risk': 0.25, 'regime': 1},
            'JPN': {'name': 'Japan', 'transition_risk': 0.15, 'regime': 1},
            'IND': {'name': 'India', 'transition_risk': 0.35, 'regime': 1},
            'BRA': {'name': 'Brazil', 'transition_risk': 0.40, 'regime': 1},
            'SAU': {'name': 'Saudi Arabia', 'transition_risk': 0.30, 'regime': 3},
            'TUR': {'name': 'Turkey', 'transition_risk': 0.45, 'regime': 2},
            'POL': {'name': 'Poland', 'transition_risk': 0.25, 'regime': 1},
            'TWN': {'name': 'Taiwan', 'transition_risk': 0.60, 'regime': 1},
            'KOR': {'name': 'South Korea', 'transition_risk': 0.25, 'regime': 1},
            'PRK': {'name': 'North Korea', 'transition_risk': 0.70, 'regime': 4},
            'VEN': {'name': 'Venezuela', 'transition_risk': 0.75, 'regime': 4},
            'SYR': {'name': 'Syria', 'transition_risk': 0.90, 'regime': 4},
        }

        # Create dummy extractor for synthetic generation
        class DummyExtractor:
            async def generate_synthetic_examples(self, nations, num):
                extractor = TrainingDataExtractor.__new__(TrainingDataExtractor)
                return await TrainingDataExtractor.generate_synthetic_examples(extractor, nations, num)

            def save_dataset(self, examples, path):
                with open(path, 'w') as f:
                    for ex in examples:
                        f.write(json.dumps(asdict(ex)) + '\n')
                print(f"Saved {len(examples)} examples to {path}")

        dummy = DummyExtractor()
        synthetic = await dummy.generate_synthetic_examples(synthetic_nations, 5000)
        dummy.save_dataset(synthetic, 'training_data_synthetic.jsonl')
        return

    # Full extraction with Supabase
    extractor = TrainingDataExtractor(supabase_url, supabase_key)

    print("Extracting learning events...")
    claude_examples = await extractor.extract_learning_events(days_back=90)
    print(f"Found {len(claude_examples)} Claude interaction examples")

    print("Extracting nation data...")
    nations = await extractor.extract_nation_snapshots()
    print(f"Found {len(nations)} nations")

    print("Generating synthetic examples...")
    synthetic = await extractor.generate_synthetic_examples(nations, num_examples=2000)
    print(f"Generated {len(synthetic)} synthetic examples")

    # Combine and save
    all_examples = claude_examples + synthetic
    extractor.save_dataset(all_examples, 'training_data.jsonl')


if __name__ == "__main__":
    asyncio.run(main())
