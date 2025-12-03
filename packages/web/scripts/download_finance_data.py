#!/usr/bin/env python3
"""
Download and merge finance training datasets into Alpaca format.
Target: 5000+ examples per topic area.

Datasets:
1. Josephgflowers/Finance-Instruct-500k - 500k instruction pairs
2. gbharti/finance-alpaca - Finance-specific Alpaca
3. FinLang/investopedia-embedding-dataset - Definitions

Output: finance_training_data.json in Alpaca format
"""

import json
import hashlib
from datasets import load_dataset
from collections import defaultdict

def deduplicate(examples):
    """Remove duplicates based on instruction hash."""
    seen = set()
    unique = []
    for ex in examples:
        key = hashlib.sha256(ex.get('instruction', '')[:100].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique

def convert_to_alpaca(instruction, input_text, output_text):
    """Convert to standard Alpaca format."""
    return {
        "instruction": instruction.strip(),
        "input": (input_text or "").strip(),
        "output": output_text.strip()
    }

def categorize_by_topic(examples):
    """Categorize examples by financial topic."""
    topics = defaultdict(list)

    keywords = {
        "stocks": ["stock", "equity", "share", "dividend", "ipo", "nasdaq", "nyse", "s&p", "dow"],
        "bonds": ["bond", "treasury", "yield", "coupon", "maturity", "fixed income", "municipal"],
        "derivatives": ["option", "future", "swap", "derivative", "call", "put", "strike", "expir"],
        "funds": ["etf", "mutual fund", "index fund", "hedge fund", "fund manager", "nav", "expense ratio"],
        "commodities": ["commodity", "gold", "silver", "oil", "crude", "wheat", "corn", "futures"],
        "trading": ["trading", "broker", "margin", "leverage", "short sell", "day trad", "arbitrage"],
        "banking": ["bank", "credit", "loan", "mortgage", "interest rate", "fed", "central bank"],
        "crypto": ["crypto", "bitcoin", "ethereum", "blockchain", "defi", "token", "wallet"],
        "economics": ["gdp", "inflation", "recession", "monetary", "fiscal", "economic", "employment"],
        "regulations": ["sec", "finra", "regulation", "compliance", "dodd-frank", "fiduciary"],
        "analysis": ["valuation", "p/e ratio", "fundamental", "technical", "chart", "indicator"],
        "risk": ["risk", "volatility", "beta", "sharpe", "var", "hedg", "diversif"],
    }

    for ex in examples:
        text = (ex.get('instruction', '') + ' ' + ex.get('output', '')).lower()
        categorized = False
        for topic, kws in keywords.items():
            if any(kw in text for kw in kws):
                topics[topic].append(ex)
                categorized = True
                break
        if not categorized:
            topics["general_finance"].append(ex)

    return dict(topics)

def main():
    all_examples = []

    print("=" * 60)
    print("FINANCE TRAINING DATA DOWNLOADER")
    print("=" * 60)

    # Dataset 1: Finance-Instruct-500k
    print("\n[1/3] Downloading Finance-Instruct-500k...")
    try:
        ds1 = load_dataset("Josephgflowers/Finance-Instruct-500k", split="train")
        print(f"  Loaded {len(ds1)} examples")

        for row in ds1:
            # Handle various column names
            instruction = row.get('instruction') or row.get('question') or row.get('input') or ''
            input_text = row.get('input') or row.get('context') or ''
            output = row.get('output') or row.get('response') or row.get('answer') or ''

            if instruction and output:
                all_examples.append(convert_to_alpaca(instruction, input_text, output))
    except Exception as e:
        print(f"  Error loading Finance-Instruct-500k: {e}")

    # Dataset 2: finance-alpaca
    print("\n[2/3] Downloading finance-alpaca...")
    try:
        ds2 = load_dataset("gbharti/finance-alpaca", split="train")
        print(f"  Loaded {len(ds2)} examples")

        for row in ds2:
            instruction = row.get('instruction', '')
            input_text = row.get('input', '')
            output = row.get('output', '')

            if instruction and output:
                all_examples.append(convert_to_alpaca(instruction, input_text, output))
    except Exception as e:
        print(f"  Error loading finance-alpaca: {e}")

    # Dataset 3: Investopedia embeddings
    print("\n[3/3] Downloading investopedia-embedding-dataset...")
    try:
        ds3 = load_dataset("FinLang/investopedia-embedding-dataset", split="train")
        print(f"  Loaded {len(ds3)} examples")

        for row in ds3:
            # Convert Q&A format to instruction format
            question = row.get('question') or row.get('query') or ''
            answer = row.get('answer') or row.get('response') or row.get('text') or ''
            topic = row.get('topic', '')

            if question and answer:
                instruction = f"Explain the following financial concept: {question}" if topic else question
                all_examples.append(convert_to_alpaca(instruction, topic, answer))
    except Exception as e:
        print(f"  Error loading investopedia: {e}")

    print(f"\n{'=' * 60}")
    print(f"Total raw examples: {len(all_examples)}")

    # Deduplicate
    print("Deduplicating...")
    all_examples = deduplicate(all_examples)
    print(f"After deduplication: {len(all_examples)}")

    # Filter quality
    print("Filtering for quality...")
    quality_examples = [
        ex for ex in all_examples
        if len(ex['instruction']) > 10
        and len(ex['output']) > 20
        and len(ex['output']) < 4000  # Not too long
    ]
    print(f"After quality filter: {len(quality_examples)}")

    # Categorize by topic
    print("\nCategorizing by topic...")
    topics = categorize_by_topic(quality_examples)

    print("\n" + "=" * 60)
    print("TOPIC BREAKDOWN:")
    print("=" * 60)
    for topic, examples in sorted(topics.items(), key=lambda x: -len(x[1])):
        status = "✓" if len(examples) >= 1000 else "⚠"
        print(f"  {status} {topic}: {len(examples)} examples")

    # Save full dataset
    output_path = "finance_training_data.json"
    with open(output_path, 'w') as f:
        json.dump(quality_examples, f, indent=2)
    print(f"\n✓ Saved {len(quality_examples)} examples to {output_path}")

    # Also save topic-specific files for targeted training
    print("\nSaving topic-specific files...")
    for topic, examples in topics.items():
        if len(examples) >= 100:  # Only save if meaningful
            topic_path = f"finance_{topic}.json"
            with open(topic_path, 'w') as f:
                json.dump(examples, f, indent=2)
            print(f"  ✓ {topic_path}: {len(examples)} examples")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)

    return quality_examples

if __name__ == "__main__":
    main()
