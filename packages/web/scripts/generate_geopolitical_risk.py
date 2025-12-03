#!/usr/bin/env python3
"""
Generate geopolitical risk and scenario analysis training data.
Focus: Country risk, regional conflicts, sanctions, trade wars, political risk
"""
import json
from typing import List, Dict

def generate_examples() -> List[Dict]:
    examples = []

    # ==========================================================================
    # 1. COUNTRY RISK PROFILES
    # ==========================================================================
    countries = {
        "great_powers": ["United States", "China", "Russia", "European Union"],
        "regional_powers": ["India", "Japan", "South Korea", "Australia", "Brazil", "Saudi Arabia", "Turkey", "Iran", "Israel", "Pakistan"],
        "emerging": ["Indonesia", "Vietnam", "Mexico", "Poland", "South Africa", "Nigeria", "Egypt", "Thailand", "Philippines", "Colombia"],
        "hotspots": ["Taiwan", "Ukraine", "North Korea", "Venezuela", "Myanmar", "Syria", "Yemen", "Libya", "Afghanistan", "Haiti"],
    }

    country_questions = [
        "What is the political stability outlook for {country}?",
        "What are the key investment risks in {country}?",
        "How might leadership change affect {country}'s policy direction?",
        "What is the economic reform trajectory for {country}?",
        "What are the social and demographic pressures facing {country}?",
        "How does {country} fit into global supply chains?",
        "What are the currency and capital control risks in {country}?",
        "What sanctions or trade restrictions affect {country}?",
        "What is {country}'s relationship with major powers?",
        "What are the military/security risks involving {country}?",
    ]

    for category, country_list in countries.items():
        for country in country_list:
            for q_template in country_questions:
                q = q_template.format(country=country)
                examples.append({
                    "instruction": q,
                    "input": "",
                    "output": f"[Country risk analysis: political assessment, economic indicators, external relations, scenario mapping, and investment implications.]"
                })

    # ==========================================================================
    # 2. BILATERAL RELATIONSHIPS
    # ==========================================================================
    key_relationships = [
        ("US", "China"), ("US", "Russia"), ("US", "EU"), ("US", "Saudi Arabia"),
        ("China", "Russia"), ("China", "Taiwan"), ("China", "India"), ("China", "Japan"),
        ("Russia", "EU"), ("Russia", "Ukraine"), ("Russia", "NATO"),
        ("India", "Pakistan"), ("Israel", "Iran"), ("Saudi Arabia", "Iran"),
        ("Japan", "South Korea"), ("North Korea", "South Korea"),
    ]

    relationship_questions = [
        "How is the {a}-{b} relationship evolving?",
        "What are the key friction points between {a} and {b}?",
        "What are the economic interdependencies between {a} and {b}?",
        "How might {a}-{b} tensions affect global markets?",
        "What is the risk of military conflict between {a} and {b}?",
        "How are {a}-{b} relations affecting regional stability?",
        "What diplomatic initiatives could improve {a}-{b} relations?",
        "How do {a}-{b} dynamics affect third countries?",
    ]

    for a, b in key_relationships:
        for q_template in relationship_questions:
            q = q_template.format(a=a, b=b)
            examples.append({
                "instruction": q,
                "input": "",
                "output": f"[Bilateral analysis: historical context, current flashpoints, stakeholder interests, scenario probabilities, and market implications.]"
            })

    # ==========================================================================
    # 3. REGIONAL FLASHPOINTS
    # ==========================================================================
    flashpoints = {
        "taiwan_strait": [
            "What is the current military balance in the Taiwan Strait?",
            "How might a Taiwan blockade scenario unfold?",
            "What would trigger a Chinese military action against Taiwan?",
            "How would the US respond to Taiwan contingencies?",
            "What are the semiconductor supply chain implications of Taiwan conflict?",
            "How are TSMC and Taiwan preparing for conflict scenarios?",
            "What is Japan's likely role in a Taiwan contingency?",
            "How would a Taiwan crisis affect global shipping?",
        ],
        "eastern_europe": [
            "What is Russia's strategic calculus in Ukraine?",
            "How sustainable is Ukraine's military position?",
            "What would Russian escalation in Ukraine look like?",
            "How are NATO members responding to Russian threats?",
            "What are the energy security implications of the Ukraine conflict?",
            "How might the war in Ukraine end?",
            "What is the risk of conflict spreading to NATO territory?",
            "How are Baltic states preparing for potential Russian aggression?",
        ],
        "middle_east": [
            "What is the risk of Iran-Israel direct conflict?",
            "How is the Saudi-Iran rivalry evolving?",
            "What are the implications of Abraham Accords for regional stability?",
            "How might Iran's nuclear program affect regional dynamics?",
            "What is the outlook for Yemen conflict resolution?",
            "How are Houthi attacks affecting Red Sea shipping?",
            "What is the trajectory of US-Iran relations?",
            "How might regime change in Iran occur and what would be the implications?",
        ],
        "south_asia": [
            "What is the risk of India-Pakistan escalation?",
            "How is the China-India border situation evolving?",
            "What are the implications of Taliban governance in Afghanistan?",
            "How might Sri Lanka's crisis affect regional stability?",
            "What is the military balance in the Kashmir region?",
            "How are India-China border tensions affecting investment flows?",
        ],
        "east_asia": [
            "What is North Korea's strategic calculus?",
            "How might a North Korean collapse scenario unfold?",
            "What is the risk of North Korean missile or nuclear tests?",
            "How are Japan-South Korea relations affecting regional security?",
            "What are China's intentions in the South China Sea?",
            "How might ASEAN respond to Chinese territorial claims?",
        ],
    }

    for region, questions in flashpoints.items():
        for q in questions:
            examples.append({
                "instruction": q,
                "input": "",
                "output": f"[Regional security analysis: force disposition, escalation pathways, diplomatic options, probability assessment, and economic impact.]"
            })

    # ==========================================================================
    # 4. SANCTIONS AND ECONOMIC WARFARE
    # ==========================================================================
    sanctions_questions = [
        "How effective have sanctions been against Russia?",
        "What are the loopholes in Russia sanctions enforcement?",
        "How is China positioning itself regarding sanctions compliance?",
        "What sanctions risks exist for companies operating in China?",
        "How might secondary sanctions affect European companies?",
        "What is the state of Iran sanctions and JCPOA?",
        "How are cryptocurrencies used for sanctions evasion?",
        "What are the implications of SWIFT disconnection?",
        "How effective are export controls on advanced semiconductors?",
        "What is the risk of sanctions escalation with China?",
        "How are sanctions affecting global commodity flows?",
        "What humanitarian exemptions exist in sanctions regimes?",
        "How are sanctions affecting global dollar dominance?",
        "What is the effectiveness of asset freezes and seizures?",
        "How might tariff escalation affect global trade?",
        "What are the WTO implications of current trade measures?",
        "How are companies navigating sanctions compliance?",
        "What is the legal risk from sanctions violations?",
    ]

    for q in sanctions_questions:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Sanctions/economic warfare analysis: legal framework, enforcement mechanisms, evasion methods, effectiveness assessment, and commercial implications.]"
        })

    # ==========================================================================
    # 5. POLITICAL RISK EVENTS
    # ==========================================================================
    political_events = {
        "elections": [
            "How might US elections affect foreign policy?",
            "What are the market implications of European elections?",
            "How might Indian elections affect economic policy?",
            "What is the outlook for Mexican political stability?",
            "How might Brazilian elections affect Mercosur?",
            "What are the implications of Turkish political dynamics?",
            "How might South African elections affect ANC policy?",
            "What is the outlook for UK political stability?",
        ],
        "regime_change": [
            "What are the scenarios for regime change in Iran?",
            "How stable is the Putin regime in Russia?",
            "What would Xi Jinping succession look like?",
            "What are the risks of coup in Saudi Arabia?",
            "How might North Korean leadership transition occur?",
            "What is the political stability outlook for Egypt?",
            "How might military intervention affect Myanmar's trajectory?",
        ],
        "populism": [
            "How is populism affecting European integration?",
            "What is the impact of nationalist movements on global trade?",
            "How might anti-globalization sentiment affect supply chains?",
            "What are the implications of immigration policy changes?",
            "How is technological nationalism affecting semiconductors?",
        ],
    }

    for category, questions in political_events.items():
        for q in questions:
            examples.append({
                "instruction": q,
                "input": "",
                "output": "[Political risk analysis: key actors, scenario mapping, probability assessment, policy implications, and investment impact.]"
            })

    # ==========================================================================
    # 6. GRAY ZONE AND HYBRID THREATS
    # ==========================================================================
    gray_zone = [
        "How is Russia conducting information warfare against the West?",
        "What are the hybrid warfare tactics used in Eastern Europe?",
        "How is China conducting influence operations globally?",
        "What cyber operations have been attributed to state actors?",
        "How are states using economic coercion as a tool of statecraft?",
        "What are the risks of critical infrastructure sabotage?",
        "How are undersea cables being targeted for espionage?",
        "What election interference risks exist from foreign actors?",
        "How are private military companies being used in conflicts?",
        "What space-based threats exist from adversarial nations?",
        "How is AI being weaponized for disinformation?",
        "What are the biosecurity risks from state programs?",
    ]

    for q in gray_zone:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Gray zone threat analysis: attribution, tactics, targets, response options, and risk mitigation strategies.]"
        })

    # ==========================================================================
    # 7. STRATEGIC RESOURCES AND CHOKEPOINTS
    # ==========================================================================
    resources = [
        "How vulnerable is global oil supply to Strait of Hormuz closure?",
        "What are the risks to rare earth supply from Chinese dominance?",
        "How might lithium supply constraints affect EV transition?",
        "What are the food security implications of Black Sea disruption?",
        "How vulnerable is global shipping to Suez Canal blockage?",
        "What are the strategic implications of Arctic resource competition?",
        "How might water scarcity drive regional conflicts?",
        "What are the risks to global semiconductor supply concentration?",
        "How vulnerable is fertilizer supply to geopolitical disruption?",
        "What are the strategic implications of cobalt supply chain risks?",
        "How might energy transition reshape geopolitical alignments?",
        "What are the risks of pharmaceutical supply chain concentration?",
    ]

    for q in resources:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Strategic resource analysis: supply concentration, chokepoint vulnerabilities, demand dynamics, diversification options, and investment implications.]"
        })

    # ==========================================================================
    # 8. SCENARIO PLANNING
    # ==========================================================================
    scenarios = [
        # Conflict escalation
        "What would be the global economic impact of Taiwan invasion?",
        "How would markets react to Russia using tactical nuclear weapons?",
        "What would happen if China blockades Taiwan?",
        "How would a major Middle East war affect oil prices?",
        "What if NATO and Russia engage in direct conflict?",
        "How would India-Pakistan nuclear exchange affect global markets?",
        # Economic scenarios
        "What would US-China full economic decoupling look like?",
        "How would Chinese real estate collapse affect global markets?",
        "What if the dollar loses reserve currency status?",
        "How would a European energy crisis unfold?",
        "What if BRICS launches a competing currency system?",
        "How would Japanese yen crisis affect global finance?",
        # Political scenarios
        "What if populist governments take power across Europe?",
        "How would US withdrawal from NATO affect global security?",
        "What if there is regime change in Saudi Arabia?",
        "How would Chinese Communist Party collapse affect markets?",
        "What if there is civil conflict in a major economy?",
        # Technology scenarios
        "What if quantum computing breaks current encryption?",
        "How would AI-enabled cyberattack affect critical infrastructure?",
        "What if space-based conflict disables satellite systems?",
        "How would controlled nuclear fusion affect geopolitics?",
    ]

    for scenario in scenarios:
        examples.append({
            "instruction": scenario,
            "input": "",
            "output": "[Scenario analysis: probability assessment, trigger events, cascade effects, sector impacts, and portfolio hedging strategies.]"
        })
        # Add preparation version
        examples.append({
            "instruction": f"How should investors prepare for: {scenario}",
            "input": "",
            "output": "[Risk preparation strategy: early warning indicators, hedging instruments, asset allocation adjustments, and contingency planning.]"
        })

    return examples


def main():
    print("=" * 70)
    print("GEOPOLITICAL RISK TRAINING DATA GENERATOR")
    print("=" * 70)

    examples = generate_examples()
    print(f"\nGenerated {len(examples)} training examples")

    # Categorize
    categories = {
        "country_risk": 0,
        "bilateral": 0,
        "flashpoints": 0,
        "sanctions": 0,
        "political": 0,
        "gray_zone": 0,
        "resources": 0,
        "scenarios": 0,
        "other": 0,
    }

    for ex in examples:
        text = ex["instruction"].lower()
        if any(x in text for x in ["stability", "investment risk", "leadership", "reform", "demographic"]):
            categories["country_risk"] += 1
        elif any(x in text for x in ["relationship", "friction", "interdepend", "tensions"]):
            categories["bilateral"] += 1
        elif any(x in text for x in ["taiwan", "ukraine", "iran", "korea", "kashmir", "south china sea"]):
            categories["flashpoints"] += 1
        elif any(x in text for x in ["sanction", "swift", "tariff", "export control"]):
            categories["sanctions"] += 1
        elif any(x in text for x in ["election", "regime", "populis", "nationalist"]):
            categories["political"] += 1
        elif any(x in text for x in ["hybrid", "cyber", "disinformation", "influence", "sabotage"]):
            categories["gray_zone"] += 1
        elif any(x in text for x in ["supply", "chokepoint", "rare earth", "strategic resource"]):
            categories["resources"] += 1
        elif any(x in text for x in ["what if", "what would", "how would", "scenario", "prepare for"]):
            categories["scenarios"] += 1
        else:
            categories["other"] += 1

    print("\n" + "=" * 70)
    print("CATEGORY BREAKDOWN:")
    print("=" * 70)
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} examples")

    # Save
    output_path = "geopolitical_risk_training.json"
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"\n Saved to {output_path}")

    return examples


if __name__ == "__main__":
    main()
