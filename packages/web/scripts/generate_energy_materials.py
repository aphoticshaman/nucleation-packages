#!/usr/bin/env python3
"""
Generate training data on energy systems, critical materials, rare earths,
nuclear programs, and strategic resource intelligence.
"""
import json
from typing import List, Dict

def generate_examples() -> List[Dict]:
    examples = []

    # ==========================================================================
    # 1. ENERGY SOURCE INTELLIGENCE
    # ==========================================================================
    energy_sources = {
        "oil": {
            "aspects": ["production", "reserves", "refining", "grades", "pricing", "transport"],
            "questions": [
                "What is the global {aspect} outlook for crude oil?",
                "How do geopolitical factors affect oil {aspect}?",
                "What technological changes are affecting oil {aspect}?",
                "How is the energy transition impacting oil {aspect}?",
                "What are the strategic implications of oil {aspect} concentration?",
            ]
        },
        "natural_gas": {
            "aspects": ["LNG", "pipelines", "storage", "pricing", "liquefaction", "regasification"],
            "questions": [
                "How is the global {aspect} infrastructure evolving?",
                "What geopolitical risks affect natural gas {aspect}?",
                "How does {aspect} capacity affect energy security?",
                "What are the investment trends in gas {aspect}?",
                "How does {aspect} affect regional gas pricing?",
            ]
        },
        "coal": {
            "aspects": ["thermal coal", "metallurgical coal", "reserves", "exports", "phase-out"],
            "questions": [
                "What is the outlook for {aspect} markets?",
                "How is {aspect} affected by decarbonization policies?",
                "Which countries dominate {aspect} supply?",
                "What are the logistics challenges for {aspect}?",
                "How does {aspect} affect steelmaking economics?",
            ]
        },
        "nuclear": {
            "aspects": ["uranium", "enrichment", "fuel fabrication", "reactor fleet", "waste storage"],
            "questions": [
                "What is the global supply chain for nuclear {aspect}?",
                "How do geopolitics affect nuclear {aspect}?",
                "What are the security implications of {aspect}?",
                "How is {aspect} capacity distributed globally?",
                "What are the chokepoints in nuclear {aspect}?",
            ]
        },
        "renewables": {
            "aspects": ["solar PV", "wind turbines", "hydropower", "geothermal", "biomass"],
            "questions": [
                "What is the manufacturing supply chain for {aspect}?",
                "How does {aspect} deployment vary by region?",
                "What are the material constraints for {aspect}?",
                "How is {aspect} affecting grid stability?",
                "What are the cost trajectories for {aspect}?",
            ]
        },
    }

    for source, data in energy_sources.items():
        for aspect in data["aspects"]:
            for q_template in data["questions"]:
                q = q_template.format(aspect=aspect)
                examples.append({
                    "instruction": q,
                    "input": "",
                    "output": f"[Energy sector analysis: supply-demand dynamics, geographic distribution, infrastructure constraints, price drivers, and strategic implications.]"
                })

    # ==========================================================================
    # 2. FUEL TYPE SPECIFIC INTELLIGENCE
    # ==========================================================================
    fuel_questions = {
        "gasoline": [
            "How do refining margins affect gasoline prices?",
            "What seasonal factors affect gasoline demand?",
            "How is EV adoption affecting gasoline demand projections?",
            "What are the regional variations in gasoline specifications?",
            "How do biofuel mandates affect gasoline markets?",
        ],
        "diesel": [
            "Why is diesel crucial for freight and agriculture?",
            "How do diesel shortages cascade through supply chains?",
            "What is the outlook for diesel demand with trucking electrification?",
            "How does the diesel-gasoline spread affect refinery economics?",
            "What are the marine fuel transitions affecting diesel demand?",
        ],
        "jet_fuel": [
            "How is aviation fuel demand recovering post-COVID?",
            "What are the prospects for sustainable aviation fuel?",
            "How does jet fuel pricing affect airline economics?",
            "What are the refinery yield challenges for jet fuel?",
            "How might hydrogen affect aviation fuel demand?",
        ],
        "lpg_propane": [
            "What drives LPG/propane demand in developing markets?",
            "How does propane affect petrochemical feedstock economics?",
            "What are the seasonal dynamics in propane markets?",
            "How does US propane exports affect global pricing?",
            "What role does LPG play in energy access?",
        ],
        "heating_oil": [
            "How is heating oil demand affected by climate and efficiency?",
            "What are the regional dependencies on heating oil?",
            "How do heat pump transitions affect heating oil demand?",
            "What are the storage and logistics for heating oil?",
            "How do cold snaps cascade through heating oil markets?",
        ],
        "bunker_fuel": [
            "How did IMO 2020 regulations transform bunker fuel markets?",
            "What are the scrubber vs. low-sulfur fuel economics?",
            "How is LNG bunkering infrastructure developing?",
            "What are the methanol and ammonia prospects for shipping?",
            "How do bunker fuel prices affect global trade costs?",
        ],
    }

    for fuel, questions in fuel_questions.items():
        for q in questions:
            examples.append({
                "instruction": q,
                "input": "",
                "output": "[Fuel market analysis: demand drivers, supply dynamics, substitution effects, regional variations, and price implications.]"
            })

    # ==========================================================================
    # 3. RARE EARTH ELEMENTS
    # ==========================================================================
    rare_earths = {
        "light_rees": {
            "elements": ["lanthanum", "cerium", "praseodymium", "neodymium", "samarium"],
            "applications": ["magnets", "catalysts", "glass polishing", "batteries", "metallurgy"],
        },
        "heavy_rees": {
            "elements": ["europium", "gadolinium", "terbium", "dysprosium", "holmium", "erbium", "ytterbium", "lutetium", "yttrium"],
            "applications": ["phosphors", "lasers", "nuclear", "magnets", "electronics"],
        },
    }

    ree_questions = [
        "What is the supply chain concentration for {element}?",
        "How is {element} used in {application}?",
        "What are the substitution options for {element}?",
        "How does {element} pricing affect downstream industries?",
        "What are the environmental impacts of {element} extraction?",
        "What recycling potential exists for {element}?",
        "How is {element} supply affected by Chinese export policies?",
        "What new mining projects could affect {element} supply?",
        "How strategic is {element} for defense applications?",
        "What processing bottlenecks exist for {element}?",
    ]

    for category, data in rare_earths.items():
        for element in data["elements"]:
            for application in data["applications"][:2]:
                for q_template in ree_questions[:5]:
                    q = q_template.format(element=element, application=application)
                    examples.append({
                        "instruction": q,
                        "input": "",
                        "output": f"[Rare earth analysis: supply concentration, processing locations, demand drivers, strategic reserves, and supply security measures.]"
                    })

    # ==========================================================================
    # 4. CRITICAL MINERALS & HEAVY METALS
    # ==========================================================================
    critical_minerals = {
        "battery_metals": {
            "minerals": ["lithium", "cobalt", "nickel", "manganese", "graphite"],
            "focus": "EV and energy storage applications",
        },
        "industrial_metals": {
            "minerals": ["copper", "aluminum", "zinc", "lead", "tin"],
            "focus": "infrastructure and manufacturing",
        },
        "specialty_metals": {
            "minerals": ["tungsten", "molybdenum", "vanadium", "titanium", "chromium"],
            "focus": "aerospace, defense, and specialty steel",
        },
        "precious_metals": {
            "minerals": ["gold", "silver", "platinum", "palladium", "rhodium"],
            "focus": "monetary, industrial, and catalytic applications",
        },
        "heavy_metals": {
            "minerals": ["uranium", "thorium", "beryllium", "tantalum", "niobium"],
            "focus": "nuclear, electronics, and aerospace",
        },
    }

    mineral_questions = [
        "What is the global production and reserve distribution for {mineral}?",
        "How is {mineral} demand affected by the energy transition?",
        "What are the processing and refining bottlenecks for {mineral}?",
        "How do {mineral} supply disruptions cascade through industries?",
        "What are the recycling rates and potential for {mineral}?",
        "How is {mineral} pricing affected by speculation and hoarding?",
        "What are the environmental and social issues with {mineral} extraction?",
        "How does {mineral} fit into national strategic stockpiles?",
        "What new extraction technologies could affect {mineral} supply?",
        "How do trade policies affect {mineral} flows?",
    ]

    for category, data in critical_minerals.items():
        for mineral in data["minerals"]:
            for q_template in mineral_questions:
                q = q_template.format(mineral=mineral)
                examples.append({
                    "instruction": q,
                    "input": "",
                    "output": f"[Critical mineral analysis: production geography, demand drivers, supply chain risks, strategic importance, and security implications.]"
                })

    # ==========================================================================
    # 5. EXOTIC ALLOYS & ADVANCED MATERIALS
    # ==========================================================================
    alloys_questions = [
        # Superalloys
        "What are the supply chain constraints for nickel superalloys?",
        "How do cobalt-based superalloys affect jet engine performance?",
        "What rare elements are critical for single-crystal turbine blades?",
        "How does superalloy scrap recycling affect supply security?",
        "What nations dominate superalloy production capacity?",

        # Titanium alloys
        "How does titanium supply chain affect aerospace production?",
        "What are the Russian titanium dependency risks?",
        "How does titanium sponge production concentrate?",
        "What are the emerging titanium alloy applications?",
        "How does titanium pricing affect defense procurement?",

        # Steel alloys
        "How do rare earth additions improve steel properties?",
        "What drives demand for high-strength low-alloy steels?",
        "How does stainless steel demand vary by grade?",
        "What are the supply chains for tool steel production?",
        "How do manganese and vanadium affect steel markets?",

        # Specialized alloys
        "What are the applications for shape-memory alloys?",
        "How do high-entropy alloys affect materials science?",
        "What are the supply constraints for beryllium-copper alloys?",
        "How do tungsten carbide supply chains operate?",
        "What zirconium alloys are critical for nuclear applications?",

        # Electronic materials
        "How does gallium supply affect semiconductor production?",
        "What germanium applications face supply constraints?",
        "How do indium markets affect display manufacturing?",
        "What are the supply chains for silicon carbide production?",
        "How does rhenium scarcity affect superalloy costs?",
    ]

    for q in alloys_questions:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Advanced materials analysis: production processes, supply concentration, application requirements, substitution potential, and strategic implications.]"
        })

    # ==========================================================================
    # 6. NUCLEAR PROGRAMS BY COUNTRY (IAEA PERSPECTIVE)
    # ==========================================================================
    nuclear_countries = {
        "weapons_states": ["United States", "Russia", "China", "France", "United Kingdom", "India", "Pakistan", "Israel", "North Korea"],
        "major_civil": ["Japan", "South Korea", "Canada", "Germany", "Ukraine", "Sweden", "Belgium", "Switzerland", "Finland"],
        "expanding": ["UAE", "Saudi Arabia", "Turkey", "Egypt", "Bangladesh", "Poland", "Czech Republic"],
        "legacy": ["Kazakhstan", "Belarus", "Armenia", "Bulgaria", "Romania", "Slovakia", "Slovenia"],
    }

    nuclear_country_questions = [
        "What is {country}'s nuclear power capacity and expansion plans?",
        "How does {country}'s nuclear program affect regional security?",
        "What is {country}'s uranium supply and enrichment status?",
        "How does {country} manage nuclear waste and spent fuel?",
        "What IAEA safeguards apply to {country}'s nuclear facilities?",
        "What is {country}'s nuclear technology export policy?",
        "How does {country}'s nuclear program affect energy independence?",
        "What reactor types does {country} operate or plan?",
        "How does {country} approach nuclear safety post-Fukushima?",
        "What is {country}'s position on nuclear non-proliferation?",
    ]

    for category, countries in nuclear_countries.items():
        for country in countries:
            for q_template in nuclear_country_questions:
                q = q_template.format(country=country)
                examples.append({
                    "instruction": q,
                    "input": "",
                    "output": f"[Nuclear program assessment: capacity, technology, fuel cycle status, safety record, proliferation risks, and geopolitical implications.]"
                })

    # ==========================================================================
    # 7. FUSION & ADVANCED ENERGY
    # ==========================================================================
    fusion_questions = [
        # Fuel and materials
        "What are the deuterium-tritium fuel cycle challenges for fusion?",
        "How is tritium breeding achieved in fusion reactor designs?",
        "What lithium resources are needed for fusion blankets?",
        "How does helium-3 availability affect fusion prospects?",
        "What are the deuterium extraction economics from seawater?",

        # Technology status
        "What is the status of ITER construction and timeline?",
        "How do tokamak and stellarator approaches compare?",
        "What are the prospects for inertial confinement fusion?",
        "How are private fusion ventures progressing?",
        "What high-temperature superconductors enable compact fusion?",

        # Materials challenges
        "What materials can withstand fusion plasma conditions?",
        "How do neutron damage limits affect fusion reactor lifetimes?",
        "What are the first-wall material challenges for fusion?",
        "How does tritium retention affect fusion safety?",
        "What are the remote handling requirements for fusion maintenance?",

        # Economics and timeline
        "What is the realistic timeline for commercial fusion power?",
        "How might fusion affect geopolitical energy dynamics?",
        "What are the capital cost projections for fusion plants?",
        "How does fusion compare to fission for baseload power?",
        "What grid integration challenges would fusion face?",
    ]

    for q in fusion_questions:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Fusion energy analysis: technology status, materials requirements, timeline projections, investment landscape, and strategic implications.]"
        })

    # ==========================================================================
    # 8. ELECTRIC POWER SYSTEMS
    # ==========================================================================
    electric_questions = [
        # Grid infrastructure
        "How is grid modernization progressing in major economies?",
        "What are the transmission constraints limiting renewable integration?",
        "How do grid interconnections affect energy security?",
        "What are the cybersecurity vulnerabilities in power grids?",
        "How does distributed generation affect grid stability?",

        # Storage and flexibility
        "What is the state of grid-scale battery deployment?",
        "How do pumped hydro resources compare by region?",
        "What role can demand response play in grid balancing?",
        "How are virtual power plants evolving?",
        "What are the economics of long-duration energy storage?",

        # Market structures
        "How do capacity markets affect generation investment?",
        "What are the effects of negative electricity prices?",
        "How do ancillary services markets value flexibility?",
        "What are the cross-border electricity trading dynamics?",
        "How do retail electricity markets affect consumer behavior?",

        # Reliability and resilience
        "How do extreme weather events affect grid reliability?",
        "What are the wildfire risks to transmission infrastructure?",
        "How vulnerable are transformer supplies to disruption?",
        "What are the electromagnetic pulse risks to power grids?",
        "How does aging infrastructure affect reliability?",
    ]

    for q in electric_questions:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Electric power analysis: infrastructure status, market dynamics, reliability metrics, investment needs, and policy implications.]"
        })

    # ==========================================================================
    # 9. ENERGY GEOPOLITICS
    # ==========================================================================
    energy_geopolitics = [
        # Producer dynamics
        "How is OPEC+ production policy affecting global oil markets?",
        "What are Russia's energy leverage tools in Europe?",
        "How does US shale production affect global supply elasticity?",
        "What is the role of NOCs in global energy markets?",
        "How do Persian Gulf producers compete for Asian market share?",

        # Consumer security
        "How is Europe diversifying away from Russian gas?",
        "What are China's energy security vulnerabilities?",
        "How does Japan manage energy import dependencies?",
        "What are India's energy security strategies?",
        "How do developing nations manage energy access vs. cost?",

        # Transit and chokepoints
        "How vulnerable is global oil trade to Strait of Hormuz closure?",
        "What are the risks to Suez Canal energy transit?",
        "How do pipeline politics affect Central Asian energy flows?",
        "What are the maritime security risks in the Malacca Strait?",
        "How do sanctions affect energy transit routes?",

        # Energy transition geopolitics
        "How will the energy transition reshape petrostates?",
        "What are the geopolitics of critical mineral supply chains?",
        "How might hydrogen trade create new energy dependencies?",
        "What are the technology transfer dynamics in clean energy?",
        "How does carbon border adjustment affect trade relations?",
    ]

    for q in energy_geopolitics:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Energy geopolitics analysis: supply-demand dynamics, transit routes, leverage mechanisms, and strategic realignment implications.]"
        })

    # ==========================================================================
    # 10. URANIUM & NUCLEAR FUEL CYCLE
    # ==========================================================================
    uranium_questions = [
        # Mining and production
        "What is the global uranium production by country?",
        "How do in-situ leaching vs. conventional mining compare?",
        "What are the major uranium deposits and their grades?",
        "How do uranium spot vs. contract markets function?",
        "What is the state of uranium inventory overhang?",

        # Conversion and enrichment
        "How is uranium hexafluoride conversion capacity distributed?",
        "What enrichment capacity exists by country and technology?",
        "How do centrifuge vs. diffusion enrichment economics compare?",
        "What is Russia's share of global enrichment services?",
        "How do HALEU requirements affect enrichment demand?",

        # Fuel fabrication
        "What fuel fabrication capacity exists for different reactor types?",
        "How does MOX fuel production affect plutonium management?",
        "What are the supply chains for zirconium cladding?",
        "How do fuel assembly designs vary by reactor vendor?",
        "What is the state of accident-tolerant fuel development?",

        # Back-end
        "How do reprocessing policies vary by country?",
        "What is the state of geological repository development?",
        "How is spent fuel storage capacity managed?",
        "What are the decommissioning cost estimates for reactor fleets?",
        "How do fast reactor programs affect waste management?",
    ]

    for q in uranium_questions:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Nuclear fuel cycle analysis: production capacity, processing bottlenecks, geographic concentration, and supply security implications.]"
        })

    return examples


def main():
    print("=" * 70)
    print("ENERGY & CRITICAL MATERIALS TRAINING DATA GENERATOR")
    print("=" * 70)

    examples = generate_examples()
    print(f"\nGenerated {len(examples)} training examples")

    # Categorize
    categories = {
        "energy_sources": 0,
        "fuels": 0,
        "rare_earths": 0,
        "critical_minerals": 0,
        "alloys": 0,
        "nuclear_countries": 0,
        "fusion": 0,
        "electric_grid": 0,
        "geopolitics": 0,
        "uranium": 0,
        "other": 0,
    }

    for ex in examples:
        text = ex["instruction"].lower()
        if any(x in text for x in ["oil", "natural gas", "coal", "lng", "pipeline", "refin"]):
            categories["energy_sources"] += 1
        elif any(x in text for x in ["gasoline", "diesel", "jet fuel", "propane", "lpg", "bunker", "heating oil"]):
            categories["fuels"] += 1
        elif any(x in text for x in ["rare earth", "neodymium", "dysprosium", "lanthanum", "cerium", "terbium"]):
            categories["rare_earths"] += 1
        elif any(x in text for x in ["lithium", "cobalt", "copper", "tungsten", "platinum", "tantalum"]):
            categories["critical_minerals"] += 1
        elif any(x in text for x in ["alloy", "superalloy", "titanium", "steel"]):
            categories["alloys"] += 1
        elif any(x in text for x in ["iaea", "nuclear power", "reactor", "safeguard", "proliferation"]):
            categories["nuclear_countries"] += 1
        elif any(x in text for x in ["fusion", "iter", "tokamak", "tritium", "deuterium"]):
            categories["fusion"] += 1
        elif any(x in text for x in ["grid", "transmission", "storage", "electricity", "reliability"]):
            categories["electric_grid"] += 1
        elif any(x in text for x in ["opec", "strait", "transit", "geopolitic", "petro"]):
            categories["geopolitics"] += 1
        elif any(x in text for x in ["uranium", "enrichment", "fuel fabrication", "reprocessing", "spent fuel"]):
            categories["uranium"] += 1
        else:
            categories["other"] += 1

    print("\n" + "=" * 70)
    print("CATEGORY BREAKDOWN:")
    print("=" * 70)
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} examples")

    # Save
    output_path = "energy_materials_training.json"
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"\nSaved to {output_path}")

    return examples


if __name__ == "__main__":
    main()
