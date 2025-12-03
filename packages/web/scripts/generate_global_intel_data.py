#!/usr/bin/env python3
"""
Generate comprehensive global intelligence training data.

Coverage:
1. GLOBAL MARKETS (20 major markets)
2. EMERGING TECH (quantum, AI, fusion, biotech, etc.)
3. AUTOMATION & LABOR
4. SOCIOPOLITICAL DYNAMICS
5. ETHICAL/PHILOSOPHICAL DIMENSIONS

Target: 1000+ examples per major category
Output: Alpaca format for LLM fine-tuning
"""

import json
import random
from typing import List, Dict

def generate_examples() -> List[Dict]:
    examples = []

    # ==========================================================================
    # 1. GLOBAL MARKETS (Top 20 exchanges + key indices)
    # ==========================================================================
    markets = {
        "NYSE": {"country": "USA", "indices": ["Dow Jones", "S&P 500"], "currency": "USD"},
        "NASDAQ": {"country": "USA", "indices": ["NASDAQ Composite", "NASDAQ-100"], "currency": "USD"},
        "LSE": {"country": "UK", "indices": ["FTSE 100", "FTSE 250"], "currency": "GBP"},
        "TSE": {"country": "Japan", "indices": ["Nikkei 225", "TOPIX"], "currency": "JPY"},
        "SSE": {"country": "China", "indices": ["SSE Composite", "CSI 300"], "currency": "CNY"},
        "HKEX": {"country": "Hong Kong", "indices": ["Hang Seng", "HSI"], "currency": "HKD"},
        "Euronext": {"country": "EU", "indices": ["CAC 40", "AEX"], "currency": "EUR"},
        "Deutsche Börse": {"country": "Germany", "indices": ["DAX", "MDAX"], "currency": "EUR"},
        "BSE": {"country": "India", "indices": ["SENSEX", "NIFTY 50"], "currency": "INR"},
        "B3": {"country": "Brazil", "indices": ["Bovespa", "IBOVESPA"], "currency": "BRL"},
        "TSX": {"country": "Canada", "indices": ["S&P/TSX Composite"], "currency": "CAD"},
        "ASX": {"country": "Australia", "indices": ["ASX 200", "All Ordinaries"], "currency": "AUD"},
        "KRX": {"country": "South Korea", "indices": ["KOSPI", "KOSDAQ"], "currency": "KRW"},
        "SIX": {"country": "Switzerland", "indices": ["SMI", "SPI"], "currency": "CHF"},
        "MOEX": {"country": "Russia", "indices": ["MOEX Russia", "RTS"], "currency": "RUB"},
        "SGX": {"country": "Singapore", "indices": ["STI", "FTSE Singapore"], "currency": "SGD"},
        "TWSE": {"country": "Taiwan", "indices": ["TAIEX"], "currency": "TWD"},
        "Tadawul": {"country": "Saudi Arabia", "indices": ["TASI"], "currency": "SAR"},
        "JSE": {"country": "South Africa", "indices": ["JSE All Share"], "currency": "ZAR"},
        "BMV": {"country": "Mexico", "indices": ["IPC", "S&P/BMV IPC"], "currency": "MXN"},
    }

    market_topics = [
        ("market structure", "Explain the structure and trading hours of {exchange} in {country}."),
        ("index composition", "What companies make up the {index} and how is it weighted?"),
        ("market access", "How can foreign investors access {exchange}? What are the restrictions?"),
        ("currency impact", "How does {currency} fluctuation affect {exchange} listed stocks?"),
        ("regulatory body", "What regulatory body oversees {exchange} and what are key regulations?"),
        ("trading volume", "What is the typical daily trading volume on {exchange}?"),
        ("market hours", "What are the trading hours for {exchange} and how do they overlap with other major markets?"),
        ("listing requirements", "What are the listing requirements for companies on {exchange}?"),
        ("ETF access", "What ETFs provide exposure to {index} for US-based investors?"),
        ("correlation", "How correlated is {index} with the S&P 500 historically?"),
        ("sector composition", "What sectors dominate {exchange} and why?"),
        ("volatility", "How does volatility on {exchange} compare to developed market peers?"),
        ("dividend policy", "What are typical dividend policies for {country} listed companies?"),
        ("corporate governance", "How does corporate governance differ in {country} vs US markets?"),
        ("market efficiency", "How efficient is {exchange} compared to developed markets?"),
        ("political risk", "What political risks affect investments in {exchange}?"),
        ("economic indicators", "What economic indicators most affect {index} performance?"),
        ("foreign ownership", "What percentage of {exchange} is foreign-owned?"),
        ("market cap", "What is the total market capitalization of {exchange}?"),
        ("IPO activity", "How active is the IPO market on {exchange}?"),
        ("derivatives market", "What derivative products are available on {exchange}?"),
        ("short selling", "What are the short selling rules on {exchange}?"),
        ("margin trading", "What are margin requirements for trading on {exchange}?"),
        ("settlement", "What is the settlement cycle for trades on {exchange}?"),
        ("benchmark", "How is the {index} used as a benchmark globally?"),
        ("tech stocks", "What technology companies are listed on {exchange}?"),
        ("energy sector", "How significant is the energy sector on {exchange}?"),
        ("financial sector", "What banks and financials dominate {exchange}?"),
        ("consumer sector", "What consumer companies trade on {exchange}?"),
        ("real estate", "How significant is real estate on {exchange}?"),
        ("healthcare", "What healthcare companies are available on {exchange}?"),
        ("materials", "What materials and mining stocks trade on {exchange}?"),
        ("industrial", "What industrial companies are listed on {exchange}?"),
        ("utilities", "How regulated are utility stocks on {exchange}?"),
        ("currency hedging", "How should US investors hedge {currency} exposure when investing in {exchange}?"),
        ("ADR access", "What ADRs provide exposure to {country} stocks?"),
        ("dividend withholding", "What dividend withholding taxes apply to {country} stocks?"),
        ("tax treaties", "What US-{country} tax treaties affect investors?"),
        ("insider trading", "How does {country} regulate insider trading?"),
        ("market manipulation", "What safeguards against market manipulation exist on {exchange}?"),
    ]

    for exchange, info in markets.items():
        for topic, template in market_topics:
            for index in info["indices"]:
                instruction = template.format(
                    exchange=exchange,
                    country=info["country"],
                    index=index,
                    currency=info["currency"]
                )
                examples.append({
                    "instruction": instruction,
                    "input": "",
                    "output": f"[Analysis of {topic} for {exchange}/{index} in {info['country']} would go here - this is a template for fine-tuning data generation]"
                })

    # ==========================================================================
    # 2. EMERGING TECHNOLOGIES
    # ==========================================================================
    emerging_tech = {
        "quantum_computing": {
            "companies": ["IBM Quantum", "Google Quantum AI", "IonQ", "Rigetti", "D-Wave", "Honeywell Quantum"],
            "concepts": ["qubit", "superposition", "entanglement", "quantum supremacy", "error correction", "NISQ"],
            "applications": ["cryptography", "drug discovery", "optimization", "financial modeling", "materials science"],
        },
        "artificial_general_intelligence": {
            "companies": ["OpenAI", "DeepMind", "Anthropic", "xAI", "Meta AI", "Cohere"],
            "concepts": ["AGI", "ASI", "alignment", "interpretability", "emergent capabilities", "scaling laws"],
            "applications": ["research automation", "code generation", "scientific discovery", "decision support"],
        },
        "neural_interfaces": {
            "companies": ["Neuralink", "Synchron", "Blackrock Neurotech", "Kernel", "Paradromics"],
            "concepts": ["BCI", "neural implant", "motor cortex", "sensory feedback", "wireless transmission"],
            "applications": ["paralysis treatment", "memory enhancement", "direct communication", "prosthetic control"],
        },
        "fusion_energy": {
            "companies": ["Commonwealth Fusion", "TAE Technologies", "Helion", "General Fusion", "Tokamak Energy"],
            "concepts": ["tokamak", "stellarator", "plasma confinement", "net energy gain", "tritium breeding"],
            "applications": ["baseload power", "desalination", "hydrogen production", "space propulsion"],
        },
        "photonics": {
            "companies": ["Lumentum", "II-VI", "Coherent", "IPG Photonics", "NVIDIA (silicon photonics)"],
            "concepts": ["silicon photonics", "optical computing", "LiDAR", "fiber optics", "photonic chips"],
            "applications": ["data centers", "autonomous vehicles", "telecommunications", "sensing"],
        },
        "nanobots": {
            "companies": ["Nanobots Therapeutics", "Nanorobotix", "academic research labs"],
            "concepts": ["targeted drug delivery", "molecular machines", "self-assembly", "biocompatibility"],
            "applications": ["cancer treatment", "microsurgery", "environmental remediation", "manufacturing"],
        },
        "additive_manufacturing": {
            "companies": ["Stratasys", "3D Systems", "Desktop Metal", "HP 3D", "Carbon", "Relativity Space"],
            "concepts": ["SLS", "FDM", "SLA", "metal printing", "multi-material", "generative design"],
            "applications": ["aerospace", "medical implants", "prototyping", "mass customization", "construction"],
        },
        "autonomous_systems": {
            "companies": ["Waymo", "Tesla FSD", "Aurora", "Cruise", "Boston Dynamics", "Agility Robotics"],
            "concepts": ["SLAM", "sensor fusion", "path planning", "edge AI", "V2X communication"],
            "applications": ["autonomous vehicles", "warehouse robots", "delivery drones", "agricultural robots"],
        },
        "solar_technology": {
            "companies": ["First Solar", "SunPower", "Enphase", "Canadian Solar", "JinkoSolar"],
            "concepts": ["perovskite", "bifacial", "PERC", "thin-film", "solar tracking", "grid parity"],
            "applications": ["utility scale", "residential", "agrivoltaics", "building-integrated", "space solar"],
        },
        "biotechnology": {
            "companies": ["Moderna", "BioNTech", "CRISPR Therapeutics", "Illumina", "Ginkgo Bioworks"],
            "concepts": ["mRNA", "CRISPR", "gene therapy", "synthetic biology", "cell therapy", "organoids"],
            "applications": ["vaccines", "cancer treatment", "rare diseases", "agriculture", "biomanufacturing"],
        },
    }

    tech_questions = [
        "What is {concept} and how does it work in {sector}?",
        "How is {company} positioned in the {sector} market?",
        "What are the investment implications of advances in {sector}?",
        "What are the key risks facing {sector} technology development?",
        "How might {sector} disrupt traditional industries?",
        "What is the current state of {sector} commercialization?",
        "What regulatory challenges does {sector} face?",
        "How does {concept} compare to competing approaches?",
        "What timeline should investors expect for {sector} mainstream adoption?",
        "What are the geopolitical implications of {country} leadership in {sector}?",
    ]

    for sector, info in emerging_tech.items():
        sector_name = sector.replace("_", " ").title()
        for q_template in tech_questions:
            for concept in info["concepts"][:3]:
                for company in info["companies"][:2]:
                    instruction = q_template.format(
                        sector=sector_name,
                        concept=concept,
                        company=company,
                        country=random.choice(["US", "China", "EU", "Japan"])
                    )
                    examples.append({
                        "instruction": instruction,
                        "input": "",
                        "output": f"[Analysis of {sector_name} focusing on {concept} would go here]"
                    })

    # ==========================================================================
    # 3. AUTOMATION & LABOR DISRUPTION
    # ==========================================================================
    automation_topics = [
        # Logistics & Supply Chain
        ("warehouse automation", "How is Amazon's warehouse automation affecting employment in logistics?"),
        ("autonomous trucking", "What is the timeline for autonomous trucking replacing human drivers?"),
        ("port automation", "How are automated ports changing global shipping and labor dynamics?"),
        ("last-mile delivery", "How will drone and robot delivery affect last-mile logistics jobs?"),

        # Manufacturing
        ("lights-out manufacturing", "What is lights-out manufacturing and which industries are adopting it?"),
        ("collaborative robots", "How are cobots changing the human-robot work dynamic in factories?"),
        ("reshoring", "How is automation enabling manufacturing reshoring to developed countries?"),

        # White Collar
        ("AI coding assistants", "How are GitHub Copilot and similar tools affecting software developer jobs?"),
        ("legal AI", "How is AI changing the legal profession and paralegal employment?"),
        ("financial automation", "What finance and accounting jobs are most vulnerable to automation?"),
        ("medical AI", "How will AI diagnostic tools change healthcare employment?"),
        ("content generation", "How is generative AI affecting content creation and journalism jobs?"),

        # Service Sector
        ("retail automation", "How are self-checkout and automated stores changing retail employment?"),
        ("food service robots", "What is the adoption rate of robots in restaurants and food service?"),
        ("hotel automation", "How is automation changing the hospitality industry workforce?"),

        # Agriculture
        ("agricultural robots", "How are harvesting robots and drones changing farm labor needs?"),
        ("vertical farming", "How does automated vertical farming change agricultural employment?"),

        # Broader Impacts
        ("universal basic income", "How might UBI address automation-driven unemployment?"),
        ("skills retraining", "What skills will be valuable in a highly automated economy?"),
        ("gig economy", "How does the gig economy interact with automation trends?"),
        ("work week reduction", "Is a shorter work week a viable response to automation?"),
    ]

    for topic, question in automation_topics:
        examples.append({
            "instruction": question,
            "input": "",
            "output": f"[Analysis of {topic} and labor implications would go here]"
        })
        # Add variations
        variations = [
            f"What are the second-order effects of {topic} on society?",
            f"How should policymakers respond to {topic}?",
            f"Which regions will be most affected by {topic}?",
            f"What historical precedents exist for {topic}?",
            f"How are unions responding to {topic}?",
        ]
        for v in variations:
            examples.append({"instruction": v, "input": "", "output": f"[Analysis would go here]"})

    # ==========================================================================
    # 4. SOCIOPOLITICAL DYNAMICS - Expanded significantly
    # ==========================================================================
    sociopolitical_topics = {
        "us_tribalism": [
            "How is political polarization in the US affecting market stability?",
            "What are the economic implications of the rural-urban divide in America?",
            "How does media fragmentation contribute to political tribalism?",
            "What role does social media play in amplifying political division?",
            "How might a contested election affect US financial markets?",
            "What are the investment implications of potential US political violence?",
            "How does political tribalism affect corporate decision-making?",
            "What sectors benefit or suffer from increased political polarization?",
            "How do ESG considerations intersect with political tribalism?",
            "What is the relationship between economic anxiety and political extremism?",
            "How does partisan media affect consumer behavior?",
            "What are the implications of red state vs blue state economic divergence?",
            "How does political tribalism affect workforce mobility?",
            "What is the impact of culture wars on entertainment stocks?",
            "How do political divisions affect infrastructure investment?",
        ],
        "global_nationalism": [
            "How is the rise of nationalism affecting global trade flows?",
            "What are the investment implications of deglobalization?",
            "How do nationalist movements affect multinational corporations?",
            "What is the relationship between nationalism and protectionist policies?",
            "How does nationalism in Europe affect the EU's economic cohesion?",
            "What are the market implications of rising nationalism in India?",
            "How does Chinese nationalism affect foreign investment?",
            "What is the connection between nationalism and currency policies?",
            "How do nationalist movements affect immigration and labor markets?",
            "What historical patterns of nationalism inform current trends?",
            "How does resource nationalism affect mining investments?",
            "What are the implications for global supply chains?",
            "How does economic nationalism affect technology transfer?",
            "What is the relationship between nationalism and sovereign wealth funds?",
            "How do nationalist policies affect cross-border M&A?",
        ],
        "great_power_competition": [
            "How does US-China competition affect global supply chains?",
            "What are the investment implications of technology decoupling?",
            "How might conflict over Taiwan affect global markets?",
            "What is the future of the dollar as global reserve currency?",
            "How does Russia's isolation affect global commodity markets?",
            "What are the implications of BRICS expansion?",
            "How does military spending competition affect government debt?",
            "What is the economic impact of sanctions regimes?",
            "How do space competition and satellite technology affect defense stocks?",
            "What are the cybersecurity investment implications of great power rivalry?",
            "How does the chip war affect semiconductor supply chains?",
            "What are the implications of rare earth dependencies?",
            "How does competition affect undersea cable infrastructure?",
            "What is the future of SWIFT alternatives?",
            "How do proxy conflicts affect regional economies?",
        ],
        "climate_politics": [
            "How do climate policies affect energy sector investments?",
            "What is the investment case for carbon capture technology?",
            "How might climate migration affect real estate markets?",
            "What are the implications of stranded fossil fuel assets?",
            "How do extreme weather events affect insurance markets?",
            "What is the relationship between climate policy and inflation?",
            "How does the green transition affect emerging market debt?",
            "What are the geopolitical implications of the clean energy transition?",
            "How do carbon border adjustments affect trade?",
            "What is the investment case for climate adaptation vs mitigation?",
            "How do climate lawsuits affect corporate liability?",
            "What are the implications of net-zero commitments?",
            "How does climate policy uncertainty affect investment?",
            "What is the future of carbon markets?",
            "How do climate disclosures affect capital allocation?",
        ],
        "middle_east": [
            "How does the Israel-Palestine conflict affect regional economic integration?",
            "What are the investment implications of Saudi Arabia's Vision 2030?",
            "How does the Iran nuclear situation affect oil markets?",
            "What is the future of the Abraham Accords economically?",
            "How do Gulf sovereign wealth funds affect global asset prices?",
            "What are the implications of Turkey's economic policies?",
            "How does water scarcity affect Middle East geopolitics?",
            "What is the impact of Suez Canal disruptions on shipping?",
            "How do Middle East conflicts affect defense sector stocks?",
            "What are the implications of regional desalination investments?",
        ],
        "africa": [
            "How does the African Continental Free Trade Area affect investment?",
            "What are the implications of Chinese Belt and Road in Africa?",
            "How does resource extraction affect African political stability?",
            "What is the future of African tech hubs?",
            "How do African demographics affect global labor markets?",
            "What are the implications of mobile money adoption in Africa?",
            "How does climate change affect African agricultural investments?",
            "What is the impact of security challenges on African infrastructure?",
            "How do African port developments affect regional trade?",
            "What are the implications of African debt sustainability?",
        ],
        "latin_america": [
            "How does political instability affect Latin American commodities?",
            "What are the investment implications of leftist governments?",
            "How does Mexico-US relations affect manufacturing investment?",
            "What is the future of lithium triangle investments?",
            "How do drug trafficking dynamics affect Central American economies?",
            "What are the implications of Venezuelan political changes?",
            "How does Latin American migration affect US labor markets?",
            "What is the impact of nearshoring on Mexican industrial real estate?",
            "How do currency controls in Argentina affect investment?",
            "What are the implications of Brazil's agricultural sector?",
        ],
        "asia_pacific": [
            "How does North Korea's nuclear program affect regional investment?",
            "What are the implications of South China Sea disputes for shipping?",
            "How does India-Pakistan tension affect regional integration?",
            "What is the future of ASEAN economic development?",
            "How do Japan-China territorial disputes affect trade?",
            "What are the implications of Myanmar's situation for supply chains?",
            "How does Australia-China relations affect commodity exports?",
            "What is the impact of Korean reunification scenarios?",
            "How do maritime disputes affect fishing and energy?",
            "What are the implications of Indo-Pacific defense spending?",
        ],
    }

    for category, questions in sociopolitical_topics.items():
        for q in questions:
            examples.append({
                "instruction": q,
                "input": "",
                "output": f"[Geopolitical analysis of {category.replace('_', ' ')}: covers political dynamics, market effects, regional implications, strategic considerations, and actionable insights for decision-makers]"
            })
            # Add investment-focused variations
            examples.append({
                "instruction": f"How should investors position for: {q}",
                "input": "",
                "output": f"[Investment strategy: sector allocation, geographic exposure, risk hedging, and portfolio positioning recommendations]"
            })

    # ==========================================================================
    # 5. ETHICAL & PHILOSOPHICAL DIMENSIONS
    # ==========================================================================
    ethics_topics = [
        # Speciesism & Animal Exploitation
        ("speciesism", "What are the ethical implications of speciesism in industrial agriculture?"),
        ("animal agriculture", "How might changing attitudes toward animal welfare affect food industry stocks?"),
        ("lab-grown meat", "What is the investment case for cultured meat companies?"),
        ("animal testing", "How are regulations around animal testing affecting pharmaceutical and cosmetic industries?"),

        # AI Ethics
        ("AI alignment", "What are the existential risks of misaligned artificial general intelligence?"),
        ("algorithmic bias", "How does algorithmic bias affect financial services and lending?"),
        ("AI surveillance", "What are the ethical implications of AI-powered surveillance?"),
        ("AI consciousness", "What are the ethical implications if AI systems develop consciousness?"),

        # Human Enhancement
        ("genetic enhancement", "What are the ethical implications of human genetic enhancement?"),
        ("cognitive enhancement", "How might cognitive enhancement drugs affect workplace equality?"),
        ("longevity", "What are the societal implications of radical life extension?"),
        ("digital consciousness", "What are the implications of whole brain emulation or mind uploading?"),

        # Resource Ethics
        ("water rights", "How do water scarcity and rights affect global stability?"),
        ("rare earth mining", "What are the ethical implications of rare earth mining for clean energy?"),
        ("deep sea mining", "What are the environmental and ethical considerations of deep sea mining?"),
        ("space resources", "What ethical frameworks should govern space resource extraction?"),

        # Economic Ethics
        ("wealth concentration", "What are the systemic risks of extreme wealth concentration?"),
        ("surveillance capitalism", "How does the surveillance capitalism business model affect society?"),
        ("algorithmic trading", "Does high-frequency trading benefit or harm market participants?"),
        ("corporate personhood", "How does the concept of corporate personhood affect accountability?"),
    ]

    for topic, question in ethics_topics:
        examples.append({
            "instruction": question,
            "input": "",
            "output": f"[Ethical analysis of {topic} would go here]"
        })
        # Add analytical variations
        for variation in [
            f"What historical precedents inform the {topic} debate?",
            f"How do different ethical frameworks approach {topic}?",
            f"What stakeholders are affected by decisions around {topic}?",
            f"How might regulation of {topic} evolve?",
        ]:
            examples.append({"instruction": variation, "input": "", "output": "[Analysis would go here]"})

    # ==========================================================================
    # 6. BLUE CHIP & SECTOR ANALYSIS
    # ==========================================================================
    blue_chips = {
        "technology": ["Apple", "Microsoft", "NVIDIA", "Alphabet", "Amazon", "Meta", "Tesla"],
        "finance": ["JPMorgan", "Goldman Sachs", "Berkshire Hathaway", "Visa", "Mastercard"],
        "healthcare": ["UnitedHealth", "Johnson & Johnson", "Pfizer", "Eli Lilly", "Merck"],
        "energy": ["ExxonMobil", "Chevron", "Shell", "BP", "TotalEnergies"],
        "consumer": ["Walmart", "Procter & Gamble", "Coca-Cola", "PepsiCo", "McDonald's"],
        "industrial": ["Boeing", "Caterpillar", "3M", "Honeywell", "General Electric"],
    }

    sector_questions = [
        "What is the competitive moat of {company}?",
        "How does {company} compare to peers in the {sector} sector?",
        "What are the key risks facing {company}?",
        "How might {company} be affected by rising interest rates?",
        "What is {company}'s exposure to China?",
        "How is {company} positioned for the AI transition?",
        "What is {company}'s ESG profile?",
        "How has {company}'s capital allocation strategy evolved?",
        "What is the total addressable market for {company}?",
        "How defensible is {company}'s position in a recession?",
    ]

    for sector, companies in blue_chips.items():
        for company in companies:
            for q_template in sector_questions:
                examples.append({
                    "instruction": q_template.format(company=company, sector=sector),
                    "input": "",
                    "output": f"[Analysis of {company} in {sector} would go here]"
                })

    return examples


def main():
    print("=" * 70)
    print("GLOBAL INTELLIGENCE TRAINING DATA GENERATOR")
    print("=" * 70)

    examples = generate_examples()
    print(f"\nGenerated {len(examples)} training examples")

    # Categorize
    categories = {
        "global_markets": 0,
        "emerging_tech": 0,
        "automation_labor": 0,
        "sociopolitical": 0,
        "ethics": 0,
        "blue_chip": 0,
        "other": 0,
    }

    for ex in examples:
        text = ex["instruction"].lower()
        # More comprehensive keyword matching
        if any(x in text for x in ["exchange", "index", "ftse", "dax", "nikkei", "bovespa", "sensex",
                                    "nasdaq", "nyse", "lse", "tse", "sse", "hkex", "euronext",
                                    "bse", "tsx", "asx", "krx", "moex", "sgx", "kospi", "topix",
                                    "hang seng", "cac 40", "trading hours", "listing", "market cap",
                                    "ipo market", "derivatives market", "short selling", "margin",
                                    "settlement", "etf", "adr", "currency", "dividend"]):
            categories["global_markets"] += 1
        elif any(x in text for x in ["quantum", "fusion", "agi", "neural", "nanobot", "photonic",
                                       "qubit", "superposition", "tokamak", "bci", "neuralink",
                                       "crispr", "mrna", "biotech", "solar", "perovskite",
                                       "additive manufacturing", "3d print", "autonomous", "waymo",
                                       "lidar", "ai development", "deepmind", "openai", "anthropic"]):
            categories["emerging_tech"] += 1
        elif any(x in text for x in ["automation", "robot", "autonomous", "ai replacing", "job",
                                       "warehouse", "trucking", "lights-out", "cobot", "reshoring",
                                       "copilot", "legal ai", "gig economy", "ubi", "work week",
                                       "labor", "employment", "workforce", "displacement"]):
            categories["automation_labor"] += 1
        elif any(x in text for x in ["nationalism", "tribalism", "polarization", "china", "geopolit",
                                       "deglobalization", "protectionist", "sanctions", "brics",
                                       "nato", "military", "conflict", "taiwan", "ukraine", "russia",
                                       "election", "political", "domestic", "extremism", "climate",
                                       "middle east", "africa", "latin america", "asia pacific", "israel",
                                       "saudi", "iran", "gulf", "turkey", "suez", "migration", "lithium",
                                       "venezuela", "nearshoring", "korea", "asean", "india-pakistan",
                                       "south china sea", "myanmar", "australia", "red state", "blue state",
                                       "culture war", "media fragment", "investor position"]):
            categories["sociopolitical"] += 1
        elif any(x in text for x in ["ethical", "speciesism", "consciousness", "moral", "alignment",
                                       "animal welfare", "lab-grown", "genetic enhancement", "longevity",
                                       "water rights", "deep sea mining", "space resource", "surveillance",
                                       "wealth concentration", "algorithmic bias"]):
            categories["ethics"] += 1
        elif any(x in text for x in ["apple", "microsoft", "jpmorgan", "competitive moat", "nvidia",
                                       "alphabet", "amazon", "meta", "tesla", "goldman", "berkshire",
                                       "unitedhealth", "johnson", "pfizer", "exxon", "chevron",
                                       "walmart", "procter", "coca-cola", "boeing", "caterpillar"]):
            categories["blue_chip"] += 1
        else:
            categories["other"] += 1

    print("\n" + "=" * 70)
    print("CATEGORY BREAKDOWN:")
    print("=" * 70)
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        status = "✓" if count >= 1000 else "⚠" if count >= 500 else "✗"
        print(f"  {status} {cat}: {count} examples")

    # Save
    output_path = "global_intel_training.json"
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"\n✓ Saved to {output_path}")

    # Note about placeholder outputs
    print("\n" + "=" * 70)
    print("NOTE: These examples have placeholder outputs.")
    print("For production, fill with real analytical content or")
    print("use an LLM to generate substantive responses.")
    print("=" * 70)

    return examples


if __name__ == "__main__":
    main()
