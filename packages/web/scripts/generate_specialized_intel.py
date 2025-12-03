#!/usr/bin/env python3
"""
Generate specialized intelligence training data.
Focus areas: Cyber, Defense, Crypto/DeFi, Supply Chain, Energy Security
"""
import json
import random
from typing import List, Dict

def generate_examples() -> List[Dict]:
    examples = []

    # ==========================================================================
    # 1. CYBER THREAT INTELLIGENCE
    # ==========================================================================
    cyber_topics = {
        "threat_actors": {
            "apt_groups": ["APT28", "APT29", "Lazarus Group", "Equation Group", "Fancy Bear", "Cozy Bear", "Sandworm", "Turla", "Kimsuky", "APT41"],
            "nation_states": ["Russia", "China", "North Korea", "Iran", "Israel"],
            "questions": [
                "What are the TTPs (tactics, techniques, procedures) of {actor}?",
                "How has {actor}'s targeting evolved over the past year?",
                "What sectors does {actor} primarily target?",
                "What malware families are associated with {actor}?",
                "How does {actor} conduct initial access operations?",
                "What are the geopolitical motivations behind {actor}'s operations?",
                "How can organizations defend against {actor}?",
                "What attribution evidence links {actor} to {nation}?",
            ]
        },
        "attack_vectors": [
            ("ransomware", "How is ransomware affecting critical infrastructure sectors?"),
            ("supply_chain", "What are the implications of software supply chain attacks?"),
            ("zero_day", "How should organizations prioritize zero-day vulnerability response?"),
            ("phishing", "How are AI-generated phishing attacks evolving?"),
            ("credential_stuffing", "What industries are most vulnerable to credential attacks?"),
            ("iot_botnets", "How are IoT botnets being weaponized for DDoS attacks?"),
            ("deepfakes", "How are deepfakes being used in business email compromise?"),
            ("api_attacks", "What are the emerging threats to API security?"),
        ],
        "critical_infra": [
            "How vulnerable is the US power grid to cyber attacks?",
            "What cyber threats face water treatment facilities?",
            "How are hospitals being targeted by ransomware groups?",
            "What are the cyber risks to financial market infrastructure?",
            "How secure is air traffic control from nation-state attacks?",
            "What cyber threats face nuclear facilities?",
            "How are ports and shipping vulnerable to cyber attacks?",
            "What are the cyber risks to telecommunications infrastructure?",
            "How secure are election systems from foreign interference?",
            "What cyber threats face the defense industrial base?",
        ],
    }

    # Generate cyber threat actor questions
    for actor in cyber_topics["threat_actors"]["apt_groups"]:
        nation = random.choice(cyber_topics["threat_actors"]["nation_states"])
        for q_template in cyber_topics["threat_actors"]["questions"]:
            q = q_template.format(actor=actor, nation=nation)
            examples.append({
                "instruction": q,
                "input": "",
                "output": f"[Cyber threat intelligence analysis of {actor}: covers known TTPs, targeting patterns, malware arsenal, attribution confidence, and defensive recommendations. Assessment based on open-source intelligence and industry reports.]"
            })

    # Attack vector questions
    for topic, question in cyber_topics["attack_vectors"]:
        examples.append({
            "instruction": question,
            "input": "",
            "output": f"[Analysis of {topic} threat landscape: current trends, notable incidents, sector-specific risks, and mitigation strategies.]"
        })
        # Add variations
        examples.append({
            "instruction": f"What is the financial impact of {topic} attacks on enterprises?",
            "input": "",
            "output": f"[Financial impact analysis covering direct costs, recovery expenses, reputational damage, and insurance implications.]"
        })

    # Critical infrastructure
    for q in cyber_topics["critical_infra"]:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Critical infrastructure vulnerability assessment: threat actors, attack surfaces, historical incidents, regulatory requirements, and resilience recommendations.]"
        })

    # ==========================================================================
    # 2. DEFENSE & MILITARY INTELLIGENCE
    # ==========================================================================
    defense_topics = {
        "platforms": {
            "air": ["F-35", "F-22", "B-21", "KC-46", "MQ-9", "NGAD", "F-15EX", "E-7 Wedgetail"],
            "naval": ["Ford-class carrier", "Virginia-class submarine", "Columbia-class SSBN", "Constellation-class frigate", "DDG-51 destroyer"],
            "land": ["Abrams tank", "Bradley IFV", "HIMARS", "Patriot", "THAAD", "Stryker"],
            "space": ["GPS III", "SBIRS", "X-37B", "Starlink military", "space-based radar"],
        },
        "programs": [
            ("JADC2", "How will Joint All-Domain Command and Control change warfare?"),
            ("hypersonics", "What is the state of US hypersonic weapons development?"),
            ("directed_energy", "How are directed energy weapons progressing?"),
            ("autonomous", "What role will autonomous systems play in future conflicts?"),
            ("cyber_warfare", "How is cyber integrated into military operations?"),
            ("electronic_warfare", "How is EW capability evolving for peer conflicts?"),
            ("space_warfare", "How are nations preparing for space-based conflict?"),
            ("undersea_warfare", "What are the trends in undersea warfare technology?"),
        ],
        "geopolitical": [
            "How would a Taiwan Strait conflict unfold?",
            "What are NATO's eastern flank vulnerabilities?",
            "How is the Russia-Ukraine war changing military doctrine?",
            "What lessons from Ukraine apply to a Pacific conflict?",
            "How is the Middle East military balance shifting?",
            "What are the implications of AUKUS for Indo-Pacific security?",
            "How would a Korean Peninsula conflict escalate?",
            "What are the military implications of Arctic competition?",
            "How is the Red Sea crisis affecting naval operations?",
            "What are the defense implications of BRICS military cooperation?",
        ],
    }

    # Platform questions
    for category, platforms in defense_topics["platforms"].items():
        for platform in platforms:
            examples.append({
                "instruction": f"What are the capabilities and limitations of the {platform}?",
                "input": "",
                "output": f"[Defense platform analysis: operational capabilities, production status, unit costs, deployment status, and comparison to adversary equivalents.]"
            })
            examples.append({
                "instruction": f"How does the {platform} fit into US military strategy?",
                "input": "",
                "output": f"[Strategic analysis of {platform} role in joint operations, deterrence value, and force structure planning.]"
            })

    # Program questions
    for topic, question in defense_topics["programs"]:
        examples.append({
            "instruction": question,
            "input": "",
            "output": f"[Defense program analysis: development status, funding trajectory, technical challenges, timeline, and strategic implications.]"
        })

    # Geopolitical military questions
    for q in defense_topics["geopolitical"]:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Military-strategic analysis: force postures, escalation dynamics, alliance considerations, and scenario outcomes.]"
        })

    # ==========================================================================
    # 3. CRYPTO & DEFI INTELLIGENCE
    # ==========================================================================
    crypto_topics = {
        "protocols": ["Bitcoin", "Ethereum", "Solana", "Cardano", "Polkadot", "Avalanche", "Cosmos", "Near"],
        "defi": ["Uniswap", "Aave", "Compound", "MakerDAO", "Curve", "Lido", "Rocket Pool", "Convex"],
        "layer2": ["Arbitrum", "Optimism", "zkSync", "Starknet", "Polygon", "Base", "Linea"],
        "stablecoins": ["USDT", "USDC", "DAI", "FRAX", "LUSD", "GHO"],
        "themes": [
            "How do institutional investors approach crypto allocation?",
            "What are the systemic risks in DeFi lending protocols?",
            "How might CBDC development affect private stablecoins?",
            "What are the implications of Ethereum's transition to proof-of-stake?",
            "How do MEV dynamics affect DeFi users?",
            "What are the regulatory risks facing crypto exchanges?",
            "How might a Bitcoin ETF affect market structure?",
            "What are the security risks in cross-chain bridges?",
            "How is institutional custody evolving for digital assets?",
            "What are the implications of crypto for sanctions evasion?",
            "How might tokenization disrupt traditional finance?",
            "What are the environmental concerns with proof-of-work?",
            "How do privacy coins interact with AML regulations?",
            "What are the risks of concentrated staking in PoS networks?",
            "How might AI integration change DeFi protocols?",
        ],
    }

    for protocol in crypto_topics["protocols"]:
        examples.append({
            "instruction": f"What is the investment thesis for {protocol}?",
            "input": "",
            "output": f"[Crypto asset analysis: technology fundamentals, network metrics, competitive positioning, risk factors, and valuation considerations.]"
        })

    for defi_protocol in crypto_topics["defi"]:
        examples.append({
            "instruction": f"What are the risks and opportunities in {defi_protocol}?",
            "input": "",
            "output": f"[DeFi protocol analysis: TVL trends, smart contract risks, tokenomics, governance dynamics, and competitive moat.]"
        })

    for q in crypto_topics["themes"]:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Crypto market intelligence: current state, trend analysis, regulatory landscape, and forward-looking assessment.]"
        })

    # ==========================================================================
    # 4. SUPPLY CHAIN INTELLIGENCE
    # ==========================================================================
    supply_chain = {
        "commodities": {
            "semiconductors": ["TSMC", "Samsung", "Intel", "GlobalFoundries", "SMIC"],
            "rare_earths": ["neodymium", "dysprosium", "terbium", "europium", "yttrium"],
            "energy": ["lithium", "cobalt", "nickel", "copper", "graphite"],
            "food": ["wheat", "corn", "soybeans", "fertilizers", "palm oil"],
        },
        "chokepoints": [
            "How vulnerable is global trade to Strait of Malacca disruption?",
            "What are the risks of Suez Canal blockage?",
            "How would Panama Canal drought affect shipping?",
            "What are the alternatives to South China Sea shipping routes?",
            "How vulnerable are undersea cables to sabotage?",
            "What are the risks to Black Sea grain exports?",
            "How might Taiwan Strait tensions affect chip supply?",
            "What are the vulnerabilities in Arctic shipping routes?",
        ],
        "resilience": [
            "How are companies reshoring semiconductor production?",
            "What is the state of rare earth processing outside China?",
            "How is the EV battery supply chain diversifying?",
            "What are the trends in nearshoring manufacturing?",
            "How are companies building supply chain visibility?",
            "What role does 3D printing play in supply chain resilience?",
            "How are companies managing multi-tier supplier risk?",
            "What are the costs and benefits of just-in-case inventory?",
        ],
    }

    for category, items in supply_chain["commodities"].items():
        for item in items:
            examples.append({
                "instruction": f"What are the supply chain risks for {item}?",
                "input": "",
                "output": f"[Supply chain analysis of {item}: geographic concentration, key suppliers, demand drivers, substitution options, and risk mitigation strategies.]"
            })

    for q in supply_chain["chokepoints"]:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Maritime/logistics chokepoint analysis: traffic volumes, alternative routes, historical disruptions, insurance implications, and geopolitical factors.]"
        })

    for q in supply_chain["resilience"]:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Supply chain resilience assessment: current initiatives, cost-benefit analysis, timeline considerations, and strategic recommendations.]"
        })

    # ==========================================================================
    # 5. ENERGY SECURITY INTELLIGENCE
    # ==========================================================================
    energy_topics = {
        "oil_gas": [
            "How is OPEC+ production policy affecting oil markets?",
            "What are the implications of US shale production trends?",
            "How is Russia rerouting oil exports under sanctions?",
            "What is the outlook for LNG supply and demand?",
            "How are petrochemical feedstock markets evolving?",
            "What are the implications of Venezuela production recovery?",
            "How might Iran sanctions relief affect oil markets?",
            "What is the state of strategic petroleum reserves globally?",
            "How are refining margins affecting fuel prices?",
            "What are the risks to offshore oil infrastructure?",
        ],
        "clean_energy": [
            "What is the outlook for solar panel manufacturing capacity?",
            "How is offshore wind development progressing globally?",
            "What are the bottlenecks in grid-scale battery deployment?",
            "How is green hydrogen production scaling?",
            "What is the state of nuclear new build projects?",
            "How are grid interconnection queues affecting renewables?",
            "What are the rare earth needs for clean energy transition?",
            "How is energy storage technology evolving?",
            "What are the grid stability challenges with high renewables?",
            "How are transmission constraints limiting clean energy?",
        ],
        "security": [
            "How vulnerable is the US electric grid to physical attack?",
            "What are the cybersecurity risks to oil and gas pipelines?",
            "How might climate change affect energy infrastructure?",
            "What are the energy security implications of EV adoption?",
            "How does energy independence affect national security?",
            "What are the risks of concentrated battery manufacturing?",
            "How might geopolitics affect critical mineral supply for energy?",
            "What are the insurance implications of climate-related energy disruptions?",
        ],
    }

    for category, questions in energy_topics.items():
        for q in questions:
            examples.append({
                "instruction": q,
                "input": "",
                "output": f"[Energy sector analysis: market dynamics, supply-demand fundamentals, geopolitical factors, regulatory landscape, and investment implications.]"
            })

    # ==========================================================================
    # 6. SCENARIO ANALYSIS & WAR GAMING
    # ==========================================================================
    scenarios = [
        # Conflict scenarios
        "What would be the economic impact of a 72-hour Taiwan blockade?",
        "How would markets react to a limited nuclear exchange?",
        "What are the cascade effects of a major cyberattack on US banks?",
        "How would a Strait of Hormuz closure affect global oil prices?",
        "What if China invades Taiwan in the next 5 years?",
        "How would a Russia-NATO direct conflict unfold?",
        "What are the implications of North Korea collapse?",
        "How would a major earthquake affecting Tokyo impact global markets?",

        # Economic scenarios
        "What if the US dollar loses reserve currency status?",
        "How would a Chinese real estate collapse affect global markets?",
        "What if inflation becomes entrenched above 5%?",
        "How would a major sovereign debt crisis unfold?",
        "What if AI displaces 30% of white-collar jobs in 10 years?",
        "How would a global pandemic worse than COVID affect markets?",
        "What if there is a major failure in the derivatives market?",
        "How would discovery of room-temperature superconductors affect energy markets?",

        # Technology scenarios
        "What if AGI is achieved within 5 years?",
        "How would quantum computing breaking encryption affect finance?",
        "What if fusion power becomes commercially viable?",
        "How would a major social media platform collapse affect markets?",
        "What if bioweapons become easily accessible through AI?",
        "How would discovery of extraterrestrial life affect markets?",
    ]

    for scenario in scenarios:
        examples.append({
            "instruction": scenario,
            "input": "",
            "output": "[Scenario analysis: probability assessment, timeline considerations, first-order effects, second-order cascades, sector impacts, and hedging strategies.]"
        })
        examples.append({
            "instruction": f"How should investors prepare for: {scenario}",
            "input": "",
            "output": "[Investment strategy under scenario: asset allocation, sector positioning, hedging instruments, and portfolio stress-testing considerations.]"
        })

    return examples


def main():
    print("=" * 70)
    print("SPECIALIZED INTELLIGENCE TRAINING DATA GENERATOR")
    print("=" * 70)

    examples = generate_examples()
    print(f"\nGenerated {len(examples)} training examples")

    # Categorize
    categories = {
        "cyber": 0,
        "defense": 0,
        "crypto": 0,
        "supply_chain": 0,
        "energy": 0,
        "scenarios": 0,
        "other": 0,
    }

    for ex in examples:
        text = ex["instruction"].lower()
        if any(x in text for x in ["apt", "ransomware", "cyber", "malware", "phishing", "zero-day", "ttp", "threat actor"]):
            categories["cyber"] += 1
        elif any(x in text for x in ["f-35", "military", "defense", "nato", "taiwan", "carrier", "submarine", "hypersonic", "warfare"]):
            categories["defense"] += 1
        elif any(x in text for x in ["bitcoin", "ethereum", "defi", "crypto", "stablecoin", "blockchain", "token"]):
            categories["crypto"] += 1
        elif any(x in text for x in ["supply chain", "semiconductor", "rare earth", "chokepoint", "shipping", "reshoring"]):
            categories["supply_chain"] += 1
        elif any(x in text for x in ["oil", "gas", "energy", "solar", "wind", "nuclear", "grid", "lng", "opec"]):
            categories["energy"] += 1
        elif any(x in text for x in ["what if", "scenario", "how would", "collapse", "blockade", "crisis"]):
            categories["scenarios"] += 1
        else:
            categories["other"] += 1

    print("\n" + "=" * 70)
    print("CATEGORY BREAKDOWN:")
    print("=" * 70)
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} examples")

    # Save
    output_path = "specialized_intel_training.json"
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"\n✓ Saved to {output_path}")

    return examples


if __name__ == "__main__":
    main()
