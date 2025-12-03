#!/usr/bin/env python3
"""
Generate market intelligence and economic indicator training data.
Focus: Fed policy, economic indicators, earnings, M&A, market structure
"""
import json
from typing import List, Dict

def generate_examples() -> List[Dict]:
    examples = []

    # ==========================================================================
    # 1. CENTRAL BANK & MONETARY POLICY
    # ==========================================================================
    central_banks = {
        "fed": {
            "name": "Federal Reserve",
            "officials": ["Jerome Powell", "Christopher Waller", "Michelle Bowman", "Philip Jefferson", "Lisa Cook"],
            "tools": ["federal funds rate", "quantitative easing", "reverse repo", "discount window", "forward guidance"],
        },
        "ecb": {
            "name": "European Central Bank",
            "officials": ["Christine Lagarde", "Luis de Guindos", "Philip Lane", "Isabel Schnabel"],
            "tools": ["deposit rate", "PEPP", "TLTROs", "asset purchases"],
        },
        "boj": {
            "name": "Bank of Japan",
            "officials": ["Kazuo Ueda", "Shinichi Uchida", "Ryozo Himino"],
            "tools": ["yield curve control", "negative rates", "ETF purchases"],
        },
        "pboc": {
            "name": "People's Bank of China",
            "officials": ["Pan Gongsheng", "Yi Gang"],
            "tools": ["MLF rate", "LPR", "RRR cuts", "currency intervention"],
        },
    }

    cb_questions = [
        "What is the {bank} likely to do at its next meeting?",
        "How will {tool} changes affect {market}?",
        "What did {official}'s recent speech signal about policy?",
        "How are {bank} balance sheet changes affecting liquidity?",
        "What is the transmission mechanism of {bank}'s {tool}?",
        "How does {bank} policy divergence affect currency markets?",
        "What are the risks of {bank} policy error in the current environment?",
        "How should investors position for {bank} pivot?",
    ]

    markets = ["equities", "bonds", "credit", "currencies", "commodities", "real estate"]

    for cb_key, cb_data in central_banks.items():
        for q_template in cb_questions:
            for tool in cb_data["tools"][:2]:
                for market in markets[:3]:
                    for official in cb_data["officials"][:2]:
                        q = q_template.format(
                            bank=cb_data["name"],
                            tool=tool,
                            market=market,
                            official=official
                        )
                        examples.append({
                            "instruction": q,
                            "input": "",
                            "output": f"[Central bank analysis: policy trajectory, market implications, historical precedents, and positioning recommendations.]"
                        })

    # ==========================================================================
    # 2. ECONOMIC INDICATORS
    # ==========================================================================
    indicators = {
        "employment": {
            "metrics": ["NFP", "unemployment rate", "JOLTS", "jobless claims", "labor force participation", "wage growth"],
            "questions": [
                "What does the latest {metric} tell us about the labor market?",
                "How will {metric} affect Fed policy expectations?",
                "What sectors are driving {metric} changes?",
                "How does {metric} compare to historical trends?",
            ]
        },
        "inflation": {
            "metrics": ["CPI", "PCE", "core inflation", "PPI", "import prices", "shelter costs"],
            "questions": [
                "What is driving the latest {metric} reading?",
                "How sticky is {metric} likely to be?",
                "What components of {metric} are most concerning?",
                "When will {metric} return to target?",
            ]
        },
        "growth": {
            "metrics": ["GDP", "ISM Manufacturing", "ISM Services", "retail sales", "industrial production", "housing starts"],
            "questions": [
                "What does {metric} signal about recession risk?",
                "How sustainable is current {metric} growth?",
                "What leading indicators are diverging from {metric}?",
                "How does {metric} compare across major economies?",
            ]
        },
        "financial": {
            "metrics": ["yield curve", "credit spreads", "VIX", "financial conditions index", "bank lending standards"],
            "questions": [
                "What is {metric} signaling about risk appetite?",
                "How does current {metric} compare to past recessions?",
                "What is causing {metric} movement?",
                "How should portfolios be adjusted based on {metric}?",
            ]
        },
    }

    for category, data in indicators.items():
        for metric in data["metrics"]:
            for q_template in data["questions"]:
                q = q_template.format(metric=metric)
                examples.append({
                    "instruction": q,
                    "input": "",
                    "output": f"[Economic indicator analysis: latest reading, trend analysis, sector breakdown, policy implications, and market impact assessment.]"
                })

    # ==========================================================================
    # 3. EARNINGS & CORPORATE ANALYSIS
    # ==========================================================================
    sectors = ["Technology", "Healthcare", "Financials", "Energy", "Consumer", "Industrials", "Materials", "Utilities", "Real Estate", "Communications"]

    companies = {
        "mega_tech": ["Apple", "Microsoft", "Alphabet", "Amazon", "Meta", "NVIDIA", "Tesla"],
        "financials": ["JPMorgan", "Goldman Sachs", "Morgan Stanley", "Bank of America", "Citigroup", "BlackRock"],
        "healthcare": ["UnitedHealth", "Johnson & Johnson", "Pfizer", "Eli Lilly", "Merck", "AbbVie"],
        "industrials": ["Caterpillar", "Boeing", "Honeywell", "Union Pacific", "Deere", "3M"],
        "energy": ["ExxonMobil", "Chevron", "ConocoPhillips", "Schlumberger", "EOG Resources"],
    }

    earnings_questions = [
        "What are the key themes from {company}'s latest earnings?",
        "How does {company}'s guidance compare to consensus expectations?",
        "What is {company}'s competitive position in {sector}?",
        "What risks are emerging for {company}?",
        "How is {company} positioned for AI/digital transformation?",
        "What are the margin trends for {company}?",
        "How is {company} managing its capital allocation?",
        "What is the bull/bear case for {company}?",
    ]

    for sector_name, cos in companies.items():
        for company in cos:
            for q_template in earnings_questions:
                q = q_template.format(company=company, sector=sector_name)
                examples.append({
                    "instruction": q,
                    "input": "",
                    "output": f"[Corporate analysis: financial performance, competitive dynamics, growth outlook, valuation, and investment thesis.]"
                })

    # Sector-level questions
    sector_questions = [
        "What are the key themes for {sector} sector earnings?",
        "How is the {sector} sector positioned for the current macro environment?",
        "What are the relative value opportunities in {sector}?",
        "What regulatory changes are affecting {sector}?",
        "How is AI disrupting the {sector} sector?",
        "What are the best and worst positioned companies in {sector}?",
    ]

    for sector in sectors:
        for q in sector_questions:
            examples.append({
                "instruction": q.format(sector=sector),
                "input": "",
                "output": f"[Sector analysis: earnings trends, macro sensitivity, competitive dynamics, regulatory landscape, and investment recommendations.]"
            })

    # ==========================================================================
    # 4. M&A AND CORPORATE EVENTS
    # ==========================================================================
    ma_questions = [
        "What are the strategic rationales for M&A in {sector}?",
        "What deal multiples are currently achievable in {sector}?",
        "What regulatory hurdles face {sector} M&A?",
        "What are the likely M&A targets in {sector}?",
        "How should investors play {sector} consolidation?",
        "What private equity trends are affecting {sector}?",
        "What SPACs and de-SPACs are notable in {sector}?",
        "What activist investors are active in {sector}?",
    ]

    for sector in sectors:
        for q in ma_questions:
            examples.append({
                "instruction": q.format(sector=sector),
                "input": "",
                "output": "[M&A/corporate event analysis: deal rationale, valuation, regulatory probability, timeline, and trading strategies.]"
            })

    # ==========================================================================
    # 5. MARKET STRUCTURE & FLOWS
    # ==========================================================================
    flow_questions = [
        "What are the institutional positioning trends in equities?",
        "How are retail flows affecting market dynamics?",
        "What is the state of options market structure?",
        "How is algorithmic trading affecting market quality?",
        "What are the risks from passive/index investing growth?",
        "How are ETF flows affecting underlying markets?",
        "What is the state of market liquidity?",
        "How are market maker dynamics changing?",
        "What are the risks of gamma hedging on market moves?",
        "How is securities lending affecting short interest?",
        "What are the implications of T+1 settlement?",
        "How are crypto ETFs affecting traditional markets?",
        "What are the effects of 0DTE options proliferation?",
        "How is dark pool trading affecting price discovery?",
        "What are the trends in corporate buyback activity?",
        "How are pension fund flows affecting bond markets?",
        "What is the state of basis trades in Treasuries?",
        "How are currency hedging costs affecting international flows?",
        "What are the risks from concentrated positions?",
        "How is retail options activity affecting index volatility?",
    ]

    for q in flow_questions:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Market structure analysis: flow dynamics, positioning data, liquidity metrics, and market impact assessment.]"
        })

    # ==========================================================================
    # 6. ASSET ALLOCATION & PORTFOLIO STRATEGY
    # ==========================================================================
    portfolio_questions = [
        "How should portfolios be positioned for late cycle?",
        "What is the optimal bond allocation in current environment?",
        "How should investors think about cash allocation?",
        "What role should alternatives play in portfolios?",
        "How should international vs domestic exposure be weighted?",
        "What factor exposures are attractive currently?",
        "How should duration risk be managed?",
        "What hedging strategies are most effective currently?",
        "How should credit risk be positioned?",
        "What are the best diversifiers in current correlations?",
        "How should real assets be allocated?",
        "What tail risk hedges are most attractive?",
        "How should portfolios prepare for recession?",
        "What is the role of commodities in portfolios?",
        "How should emerging markets be weighted?",
        "What private market allocations make sense?",
        "How should rebalancing frequency be adjusted?",
        "What currency hedging strategy is optimal?",
        "How should ESG considerations affect allocation?",
        "What is the outlook for 60/40 portfolios?",
    ]

    for q in portfolio_questions:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Portfolio strategy analysis: macro regime, asset class outlook, risk management, implementation considerations, and specific recommendations.]"
        })

    # ==========================================================================
    # 7. TRADING & EXECUTION
    # ==========================================================================
    trading_questions = [
        "What is the optimal execution strategy for large equity orders?",
        "How should algorithmic trading be deployed in volatile markets?",
        "What are the risks of VWAP vs TWAP execution?",
        "How should block trades be structured?",
        "What dark pool strategies are most effective?",
        "How should FX execution be optimized?",
        "What are the considerations for bond market execution?",
        "How should options execution minimize slippage?",
        "What are the risks of market-on-close orders?",
        "How should execution be timed around economic releases?",
    ]

    for q in trading_questions:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Execution analysis: market microstructure considerations, algorithm selection, timing strategies, and cost minimization techniques.]"
        })

    return examples


def main():
    print("=" * 70)
    print("MARKET INTELLIGENCE TRAINING DATA GENERATOR")
    print("=" * 70)

    examples = generate_examples()
    print(f"\nGenerated {len(examples)} training examples")

    # Categorize
    categories = {
        "central_bank": 0,
        "indicators": 0,
        "earnings": 0,
        "ma_events": 0,
        "market_structure": 0,
        "portfolio": 0,
        "trading": 0,
        "other": 0,
    }

    for ex in examples:
        text = ex["instruction"].lower()
        if any(x in text for x in ["fed", "reserve", "ecb", "boj", "pboc", "central bank", "policy"]):
            categories["central_bank"] += 1
        elif any(x in text for x in ["cpi", "gdp", "nfp", "inflation", "unemployment", "indicator", "pce", "ism"]):
            categories["indicators"] += 1
        elif any(x in text for x in ["earnings", "company", "guidance", "margin", "competitive"]):
            categories["earnings"] += 1
        elif any(x in text for x in ["m&a", "deal", "activist", "spac", "consolidation"]):
            categories["ma_events"] += 1
        elif any(x in text for x in ["flow", "positioning", "liquidity", "market maker", "etf flow", "options"]):
            categories["market_structure"] += 1
        elif any(x in text for x in ["portfolio", "allocation", "hedge", "duration", "diversif"]):
            categories["portfolio"] += 1
        elif any(x in text for x in ["execution", "vwap", "algorithm", "block", "slippage"]):
            categories["trading"] += 1
        else:
            categories["other"] += 1

    print("\n" + "=" * 70)
    print("CATEGORY BREAKDOWN:")
    print("=" * 70)
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} examples")

    # Save
    output_path = "market_intel_training.json"
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"\n Saved to {output_path}")

    return examples


if __name__ == "__main__":
    main()
