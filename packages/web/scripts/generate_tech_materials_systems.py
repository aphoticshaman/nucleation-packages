#!/usr/bin/env python3
"""
Generate training data on natural resources, advanced materials, and the
branching technological dependencies from raw materials to quantum systems.
"""
import json
from typing import List, Dict

def generate_examples() -> List[Dict]:
    examples = []

    # ==========================================================================
    # 1. HYDROCARBON RESOURCES
    # ==========================================================================
    hydrocarbons = {
        "conventional_oil": [
            "What are the global conventional oil reserves by region?",
            "How does conventional oil extraction affect water resources?",
            "What is the EROI trajectory for conventional oil deposits?",
            "How does oil field depletion cascade through regional economies?",
            "What determines the economic viability of oil field development?",
        ],
        "shale_oil": [
            "How did shale oil transform US energy independence?",
            "What are the water requirements for hydraulic fracturing?",
            "How does shale decline rate affect production economics?",
            "What are the cascading infrastructure needs for shale development?",
            "How does shale oil quality differ from conventional crude?",
            "What induced seismicity risks accompany shale extraction?",
        ],
        "oil_sands": [
            "What are the energy and water inputs for oil sands extraction?",
            "How do tailings ponds cascade through watershed ecosystems?",
            "What is the carbon intensity of oil sands production?",
            "How does oil sands economics respond to price changes?",
        ],
        "natural_gas": [
            "How does associated gas production link to oil markets?",
            "What infrastructure cascades from natural gas field development?",
            "How does gas flaring affect climate and local communities?",
            "What are the methane leakage rates across gas supply chains?",
        ],
        "lng": [
            "How does LNG liquefaction cascade through energy markets?",
            "What are the geopolitical implications of LNG trade growth?",
            "How does LNG enable gas markets to globalize?",
            "What infrastructure investments does LNG require?",
        ],
    }

    for category, questions in hydrocarbons.items():
        for q in questions:
            examples.append({
                "instruction": q,
                "input": "",
                "output": f"[Hydrocarbon resource analysis: extraction methods, environmental cascades, economic dependencies, and infrastructure requirements.]"
            })

    # ==========================================================================
    # 2. BATTERY & ENERGY STORAGE MATERIALS
    # ==========================================================================
    battery_materials = {
        "lithium": [
            "What are the lithium brine vs. hard rock extraction trade-offs?",
            "How does lithium concentration in the 'lithium triangle' create dependencies?",
            "What water conflicts emerge from lithium extraction?",
            "How does lithium demand cascade through automotive supply chains?",
            "What direct lithium extraction technologies are emerging?",
            "How does lithium recycling economics affect virgin material demand?",
        ],
        "cobalt": [
            "How does DRC cobalt concentration create supply chain risks?",
            "What human rights issues cascade through cobalt supply chains?",
            "How do cobalt-free battery chemistries affect supply dynamics?",
            "What artisanal mining practices affect cobalt supply?",
        ],
        "nickel": [
            "How does nickel grade affect battery performance?",
            "What are the environmental cascades from nickel laterite processing?",
            "How does Indonesian nickel processing capacity affect markets?",
            "What energy inputs does nickel refining require?",
        ],
        "graphite": [
            "How does synthetic vs. natural graphite affect battery production?",
            "What is China's share of graphite processing and why?",
            "How does graphite anode quality affect battery performance?",
            "What are the scaling challenges for graphite production?",
        ],
        "manganese": [
            "What role does manganese play in battery chemistry evolution?",
            "How does manganese supply chain differ from other battery metals?",
            "What are the South African manganese dependencies?",
        ],
    }

    for material, questions in battery_materials.items():
        for q in questions:
            examples.append({
                "instruction": q,
                "input": "",
                "output": f"[Battery material analysis: extraction geography, processing bottlenecks, supply chain ethics, and technology evolution implications.]"
            })

    # ==========================================================================
    # 3. ELECTRONICS & SEMICONDUCTOR MATERIALS
    # ==========================================================================
    electronics_materials = {
        "silicon": [
            "How does polysilicon production cascade through solar and chip industries?",
            "What purity levels are required for different silicon applications?",
            "How does silicon wafer production concentrate geographically?",
            "What energy inputs does semiconductor-grade silicon require?",
        ],
        "gallium": [
            "How does gallium scarcity affect compound semiconductor production?",
            "What role does gallium play in 5G and radar systems?",
            "How does gallium arsenide enable high-frequency electronics?",
            "What is China's control over gallium supply?",
        ],
        "germanium": [
            "How does germanium enable fiber optic and infrared systems?",
            "What are the recycling pathways for germanium?",
            "How does germanium supply concentrate and why?",
        ],
        "indium": [
            "How does indium affect display and touchscreen production?",
            "What are the indium supply constraints for ITO production?",
            "How might indium shortages affect electronics manufacturing?",
        ],
        "tantalum": [
            "How does tantalum conflict sourcing affect electronics supply chains?",
            "What role does tantalum play in capacitor production?",
            "How do conflict mineral regulations affect tantalum sourcing?",
        ],
        "neon_gases": [
            "How did Ukraine conflict expose neon gas dependencies?",
            "What role do noble gases play in semiconductor lithography?",
            "How concentrated is specialty gas production?",
        ],
    }

    for material, questions in electronics_materials.items():
        for q in questions:
            examples.append({
                "instruction": q,
                "input": "",
                "output": f"[Electronics material analysis: purity requirements, geographic concentration, supply chain vulnerabilities, and downstream technology dependencies.]"
            })

    # ==========================================================================
    # 4. CELL PHONE MATERIAL CASCADE
    # ==========================================================================
    phone_materials = [
        "What materials are in a smartphone and where do they come from?",
        "How many countries contribute materials to a single smartphone?",
        "What rare earth elements are essential for smartphone functionality?",
        "How does smartphone production cascade through global mining?",
        "What is the material footprint of annual smartphone production?",
        "How do smartphone display materials concentrate supply chains?",
        "What battery materials limit smartphone production capacity?",
        "How do smartphone magnets depend on rare earth supply?",
        "What copper requirements flow from smartphone production?",
        "How does smartphone gold content affect artisanal mining?",
        "What tungsten supply serves smartphone vibration motors?",
        "How does smartphone glass production cascade through silica supply?",
        "What palladium requirements flow from smartphone electronics?",
        "How does smartphone demand cascade through aluminum supply?",
        "What are the environmental cascades of smartphone mineral extraction?",
        "How does smartphone e-waste affect material recovery systems?",
        "What conflict minerals enter smartphone supply chains?",
        "How do smartphone camera modules depend on rare elements?",
    ]

    for q in phone_materials:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Smartphone material cascade: component-level material mapping, geographic sourcing, supply chain ethics, and environmental footprint analysis.]"
        })

    # ==========================================================================
    # 5. AEROSPACE & ROCKET MATERIALS
    # ==========================================================================
    aerospace = {
        "structural": [
            "What aluminum alloys are critical for aerospace structures?",
            "How does carbon fiber cascade through aerospace production?",
            "What titanium supply chains serve aerospace manufacturing?",
            "How do aerospace steel requirements differ from commercial grades?",
        ],
        "propulsion": [
            "What materials enable high-temperature turbine operation?",
            "How do rhenium and hafnium affect jet engine performance?",
            "What rocket nozzle materials withstand extreme conditions?",
            "How do single-crystal superalloys cascade through engine production?",
        ],
        "rocket_specific": [
            "What materials are used in solid rocket motor casings?",
            "How does hypergolic propellant chemistry affect material selection?",
            "What ablative materials protect reentry vehicles?",
            "How do cryogenic fuel systems affect material requirements?",
            "What composite materials enable rocket fairing production?",
            "How do pyrophoric materials serve as hypergolic igniters?",
            "What beryllium applications exist in aerospace systems?",
        ],
        "pyrophorics": [
            "What are the applications of pyrophoric materials in aerospace?",
            "How do pyrophoric alloys enable specialized ignition systems?",
            "What safety considerations cascade from pyrophoric material use?",
            "How are pyrophoric materials manufactured and handled?",
            "What role do pyrophoric liners play in armor and munitions?",
        ],
    }

    for category, questions in aerospace.items():
        for q in questions:
            examples.append({
                "instruction": q,
                "input": "",
                "output": f"[Aerospace materials analysis: performance requirements, supply chain concentration, manufacturing complexity, and technology dependencies.]"
            })

    # ==========================================================================
    # 6. SATELLITE SYSTEMS
    # ==========================================================================
    satellite = [
        # Materials
        "What materials enable satellite solar panel production?",
        "How do satellite electronics depend on radiation-hardened components?",
        "What thermal management materials serve satellite systems?",
        "How do satellite propulsion systems depend on specialty materials?",
        "What optical materials enable satellite imaging systems?",

        # Manufacturing cascade
        "How does satellite production cascade through aerospace supply chains?",
        "What cleanroom and testing infrastructure does satellite manufacturing require?",
        "How does small satellite proliferation affect material demand?",
        "What ground segment infrastructure cascades from satellite deployment?",

        # Systems integration
        "How do satellite constellations cascade through launch infrastructure?",
        "What spectrum allocation cascades from satellite proliferation?",
        "How does satellite imagery cascade through agricultural and defense applications?",
        "What connectivity cascades from low-earth orbit satellite networks?",
        "How do GPS and navigation satellites cascade through economic systems?",
        "What weather satellite dependencies exist across industries?",
    ]

    for q in satellite:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Satellite systems analysis: material requirements, manufacturing cascades, ground infrastructure, and downstream application dependencies.]"
        })

    # ==========================================================================
    # 7. FUSION REACTOR MATERIALS
    # ==========================================================================
    fusion = [
        # Plasma-facing materials
        "What materials can survive direct plasma contact in fusion reactors?",
        "How does tungsten performance affect fusion reactor viability?",
        "What are the beryllium requirements for fusion first-walls?",
        "How do plasma-material interactions limit reactor lifetime?",

        # Structural materials
        "What reduced-activation ferritic-martensitic steels serve fusion?",
        "How does neutron damage cascade through fusion structural materials?",
        "What silicon carbide composites are being developed for fusion?",
        "How do helium bubble formation affect fusion material lifetime?",

        # Tritium breeding
        "What lithium blanket materials enable tritium breeding?",
        "How does lithium-6 enrichment cascade through fusion fuel supply?",
        "What neutron multiplier materials enhance tritium production?",
        "How does tritium retention in materials affect reactor safety?",

        # Magnets
        "What superconducting materials enable fusion magnetic confinement?",
        "How does REBCO tape production cascade through fusion development?",
        "What cooling systems does superconducting magnet operation require?",
        "How do high-temperature superconductors change fusion economics?",

        # Supply chain
        "How does fusion material demand compare to fission?",
        "What rare materials could constrain fusion deployment at scale?",
        "How might fusion success cascade through energy material markets?",
    ]

    for q in fusion:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Fusion materials analysis: extreme environment requirements, material development status, supply chain implications, and technology readiness cascades.]"
        })

    # ==========================================================================
    # 8. PARTICLE COLLIDER SYSTEMS
    # ==========================================================================
    collider = [
        # Magnets and acceleration
        "What superconducting materials enable particle accelerator magnets?",
        "How does niobium-titanium production cascade through physics infrastructure?",
        "What niobium-tin requirements exist for next-generation colliders?",
        "How do RF cavity materials affect accelerator performance?",

        # Detection systems
        "What silicon detector technologies serve particle physics?",
        "How do calorimeter materials affect energy measurement precision?",
        "What scintillator materials enable particle detection?",
        "How does radiation damage cascade through detector lifetime?",

        # Infrastructure
        "What cryogenic systems support particle accelerator operation?",
        "How does collider construction cascade through civil engineering?",
        "What power requirements cascade from large accelerator operation?",
        "How do computing requirements cascade from collider data production?",

        # Knowledge cascade
        "How does particle physics cascade through technology development?",
        "What technologies originated from accelerator science?",
        "How does fundamental physics cascade through applied research?",
    ]

    for q in collider:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Particle physics infrastructure analysis: material requirements, engineering challenges, resource needs, and technology spillover effects.]"
        })

    # ==========================================================================
    # 9. QUANTUM SYSTEMS
    # ==========================================================================
    quantum = {
        "materials": [
            "What materials enable superconducting qubit fabrication?",
            "How do diamond nitrogen-vacancy centers cascade through quantum sensing?",
            "What ion trap materials serve quantum computing?",
            "How do topological materials cascade through quantum research?",
            "What photonic materials enable quantum communication?",
            "How does isotopic purity affect quantum material performance?",
        ],
        "infrastructure": [
            "What dilution refrigeration systems support quantum computing?",
            "How do vibration isolation requirements cascade through quantum labs?",
            "What electromagnetic shielding serves quantum systems?",
            "How does cleanroom infrastructure cascade through quantum fabrication?",
        ],
        "supply_chain": [
            "What rare materials could constrain quantum computer scaling?",
            "How does helium-3 scarcity affect quantum cooling systems?",
            "What specialized manufacturing serves quantum component production?",
            "How do quantum systems depend on semiconductor supply chains?",
        ],
        "network_cascade": [
            "How might quantum networks cascade through cryptographic infrastructure?",
            "What fiber optic requirements serve quantum communication?",
            "How do quantum repeaters cascade through network architecture?",
            "What satellite systems might enable quantum key distribution?",
        ],
    }

    for category, questions in quantum.items():
        for q in questions:
            examples.append({
                "instruction": q,
                "input": "",
                "output": f"[Quantum systems analysis: material requirements, infrastructure dependencies, fabrication challenges, and technology cascade implications.]"
            })

    # ==========================================================================
    # 10. PHOTONIC SYSTEMS
    # ==========================================================================
    photonics = [
        # Materials
        "What III-V semiconductors enable photonic device production?",
        "How does indium phosphide cascade through optical communications?",
        "What lithium niobate applications serve photonics?",
        "How do silicon photonics cascade through data center infrastructure?",
        "What gain materials enable laser production?",

        # Manufacturing
        "How does photonic integrated circuit fabrication cascade through fabs?",
        "What epitaxial growth requirements serve photonic manufacturing?",
        "How does optical fiber production cascade through silica supply?",
        "What packaging requirements constrain photonic device deployment?",

        # Applications cascade
        "How do photonic systems cascade through telecommunications?",
        "What LiDAR dependencies exist across autonomous systems?",
        "How does photonic computing cascade through AI infrastructure?",
        "What biosensing applications cascade from photonic advances?",
        "How do photonic systems cascade through quantum computing?",
    ]

    for q in photonics:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Photonics analysis: material platforms, manufacturing requirements, integration challenges, and application cascade effects.]"
        })

    # ==========================================================================
    # 11. COMPUTING INFRASTRUCTURE
    # ==========================================================================
    computing = {
        "chips": [
            "How does leading-edge chip production concentrate geographically?",
            "What materials cascade through advanced node fabrication?",
            "How do EUV lithography requirements constrain chip production?",
            "What gas and chemical supply chains serve semiconductor fabs?",
            "How does chip packaging cascade through computing performance?",
        ],
        "servers": [
            "What materials cascade through server production?",
            "How do server power supplies depend on rare materials?",
            "What cooling requirements cascade from server density?",
            "How does memory production cascade through computing infrastructure?",
        ],
        "datacenters": [
            "What infrastructure cascades from hyperscale datacenter construction?",
            "How do datacenter power requirements cascade through grids?",
            "What cooling water dependencies exist for datacenters?",
            "How does datacenter fiber connectivity cascade through networks?",
            "What battery and UPS systems serve datacenter reliability?",
        ],
        "network": [
            "How does network switch production cascade through semiconductor supply?",
            "What fiber optic infrastructure cascades from bandwidth growth?",
            "How do undersea cables cascade through global connectivity?",
            "What materials serve cellular tower and radio production?",
        ],
    }

    for category, questions in computing.items():
        for q in questions:
            examples.append({
                "instruction": q,
                "input": "",
                "output": f"[Computing infrastructure analysis: material dependencies, manufacturing concentration, energy requirements, and system interdependencies.]"
            })

    # ==========================================================================
    # 12. DATABASE & STORAGE ARCHITECTURE
    # ==========================================================================
    storage = [
        # Physical storage
        "What materials cascade through hard drive production?",
        "How do rare earth magnets serve storage device operation?",
        "What flash memory materials constrain SSD production?",
        "How does tape storage cascade through archival infrastructure?",

        # Architecture cascade
        "How does distributed storage cascade through datacenter design?",
        "What network requirements cascade from storage architecture choices?",
        "How do consistency requirements cascade through database design?",
        "What caching hierarchies cascade through storage performance?",

        # Scaling implications
        "How does data growth cascade through storage infrastructure?",
        "What energy requirements cascade from storage scaling?",
        "How do durability requirements cascade through storage technology choice?",
        "What materials would constrain global storage capacity growth?",
    ]

    for q in storage:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Storage systems analysis: material requirements, architectural dependencies, scaling challenges, and infrastructure cascade effects.]"
        })

    # ==========================================================================
    # 13. BRANCHING EVOLUTION OF IDEAS
    # ==========================================================================
    idea_evolution = [
        # Historical branches
        "How did vacuum tube technology branch into solid-state electronics?",
        "What branching occurred from early electromagnetic research?",
        "How did materials science branch from chemistry and physics?",
        "What technological branches emerged from laser invention?",
        "How did networking concepts branch through communications systems?",

        # Convergence points
        "How do materials, physics, and engineering converge in quantum systems?",
        "What convergences enabled smartphone technology?",
        "How do AI and materials science converge in discovery?",
        "What field convergences are enabling fusion progress?",

        # Ecosystem dynamics
        "How do research ecosystems enable technological flourishing?",
        "What conditions allow idea cross-pollination between fields?",
        "How do funding structures cascade through research directions?",
        "What role does serendipity play in technological branching?",
        "How do publication and communication cascade through knowledge growth?",

        # Future branches
        "What technological branches might quantum computing enable?",
        "How might fusion success branch through energy applications?",
        "What branches might emerge from room-temperature superconductivity?",
        "How might biocomputing branch from current research?",
        "What technological branches does space colonization require?",
    ]

    for q in idea_evolution:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Idea evolution analysis: historical branching patterns, convergence dynamics, ecosystem conditions, and future trajectory possibilities.]"
        })

    # ==========================================================================
    # 14. META-PATTERNS IN TECHNOLOGICAL CASCADE
    # ==========================================================================
    meta_patterns = [
        "What patterns recur across technological revolutions?",
        "How do materials constraints shape technological possibility space?",
        "What determines whether a technology cascades globally or remains niche?",
        "How do supply chain concentrations create systemic fragility?",
        "What role does energy availability play in technological trajectory?",
        "How do feedback loops between technologies accelerate progress?",
        "What determines the branching factor in technological evolution?",
        "How do economic incentives shape which branches flourish?",
        "What role does war and competition play in technological cascade?",
        "How do regulatory environments affect technological branching?",
        "What enables technological ecosystems to self-sustain and grow?",
        "How do path dependencies lock in technological trajectories?",
        "What allows technological paradigm shifts to occur?",
        "How do materials discoveries cascade through multiple industries?",
        "What meta-stability patterns exist in technological systems?",
    ]

    for q in meta_patterns:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Meta-pattern analysis: recurring dynamics, structural constraints, evolutionary mechanisms, and systemic properties of technological development.]"
        })

    return examples


def main():
    print("=" * 70)
    print("TECH MATERIALS & SYSTEMS CASCADE TRAINING DATA GENERATOR")
    print("=" * 70)

    examples = generate_examples()
    print(f"\nGenerated {len(examples)} training examples")

    categories = {
        "hydrocarbons": 0,
        "battery": 0,
        "electronics": 0,
        "phone": 0,
        "aerospace": 0,
        "satellite": 0,
        "fusion": 0,
        "collider": 0,
        "quantum": 0,
        "photonics": 0,
        "computing": 0,
        "storage": 0,
        "ideas": 0,
        "meta": 0,
        "other": 0,
    }

    for ex in examples:
        text = ex["instruction"].lower()
        if any(x in text for x in ["oil", "gas", "shale", "lng", "hydrocarbon"]):
            categories["hydrocarbons"] += 1
        elif any(x in text for x in ["lithium", "cobalt", "battery", "nickel", "graphite"]):
            categories["battery"] += 1
        elif any(x in text for x in ["silicon", "gallium", "semiconductor", "indium", "tantalum"]):
            categories["electronics"] += 1
        elif any(x in text for x in ["smartphone", "phone", "display", "mobile"]):
            categories["phone"] += 1
        elif any(x in text for x in ["aerospace", "rocket", "turbine", "pyrophoric", "ablative"]):
            categories["aerospace"] += 1
        elif any(x in text for x in ["satellite", "orbit", "constellation", "gps"]):
            categories["satellite"] += 1
        elif any(x in text for x in ["fusion", "tritium", "plasma", "blanket"]):
            categories["fusion"] += 1
        elif any(x in text for x in ["collider", "accelerator", "particle", "detector"]):
            categories["collider"] += 1
        elif any(x in text for x in ["quantum", "qubit", "superconducting"]):
            categories["quantum"] += 1
        elif any(x in text for x in ["photonic", "lidar", "optical", "laser"]):
            categories["photonics"] += 1
        elif any(x in text for x in ["chip", "server", "datacenter", "network"]):
            categories["computing"] += 1
        elif any(x in text for x in ["storage", "memory", "database", "ssd", "drive"]):
            categories["storage"] += 1
        elif any(x in text for x in ["branch", "evolut", "ecosystem", "converge"]):
            categories["ideas"] += 1
        elif any(x in text for x in ["pattern", "trajectory", "paradigm", "meta"]):
            categories["meta"] += 1
        else:
            categories["other"] += 1

    print("\n" + "=" * 70)
    print("CATEGORY BREAKDOWN:")
    print("=" * 70)
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} examples")

    output_path = "tech_materials_systems_training.json"
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"\nSaved to {output_path}")

    return examples


if __name__ == "__main__":
    main()
