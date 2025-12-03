#!/usr/bin/env python3
"""Telecommunications infrastructure and space domain awareness training data."""
import json

def generate_examples():
    examples = []

    # TELECOM INFRASTRUCTURE
    telecom = [
        "What is the global structure of telecommunications infrastructure?",
        "How do undersea cables route global internet traffic?",
        "What are the chokepoints in global communications infrastructure?",
        "How does 5G deployment vary by region and operator?",
        "What is the state of spectrum allocation globally?",
        "How do telecom operators structure their networks?",
        "What is the role of internet exchange points?",
        "How does content delivery network infrastructure work?",
        "What is the state of fiber optic deployment?",
        "How do mobile network architectures evolve?",
        "What is the role of tower companies in telecom infrastructure?",
        "How do data center locations affect network architecture?",
        "What is the state of rural connectivity infrastructure?",
        "How do peering arrangements affect internet routing?",
        "What is the physical infrastructure behind cloud services?",
    ]
    for q in telecom:
        examples.append({"instruction": q, "input": "", "output": "[Telecom infrastructure analysis: network topology, capacity constraints, geographic distribution, and vulnerability points.]"})

    # TELECOM GEOPOLITICS
    telecom_geo = [
        "How does the Huawei controversy affect telecom infrastructure?",
        "What are the implications of telecom supply chain restrictions?",
        "How do countries approach telecom equipment security?",
        "What is the role of national telecom champions?",
        "How do subsea cable routes reflect geopolitical alignments?",
        "What are the implications of internet fragmentation?",
        "How do data sovereignty requirements affect network architecture?",
        "What is the role of telecom in Belt and Road Initiative?",
        "How do sanctions affect telecom equipment supply?",
        "What is the state of Open RAN adoption and its implications?",
        "How do spectrum auctions reflect national priorities?",
        "What is the approach to critical infrastructure protection in telecom?",
        "How do content restrictions affect internet architecture?",
        "What is the role of telecom in surveillance capabilities?",
        "How do different countries approach internet governance?",
    ]
    for q in telecom_geo:
        examples.append({"instruction": q, "input": "", "output": "[Telecom geopolitics analysis: supply chain dynamics, security concerns, governance approaches, and strategic implications.]"})

    # SATELLITE SYSTEMS
    satellite = [
        "What is the structure of the global satellite industry?",
        "How do different satellite orbits serve different purposes?",
        "What is the state of LEO constellation deployment?",
        "How do satellite ground networks function?",
        "What is the economics of satellite manufacturing?",
        "How do launch providers compete and cooperate?",
        "What is the state of satellite broadband economics?",
        "How do satellite imaging markets function?",
        "What is the role of government vs commercial satellites?",
        "How do spectrum rights work for satellite operators?",
        "What is the role of satellite navigation systems?",
        "How do weather and earth observation satellites function?",
        "What is the state of satellite servicing and debris removal?",
        "How do satellites enable global communications?",
        "What is the trajectory of satellite technology evolution?",
    ]
    for q in satellite:
        examples.append({"instruction": q, "input": "", "output": "[Satellite systems analysis: orbital mechanics, market structure, technology evolution, and application domains.]"})

    # SPACE DOMAIN AWARENESS
    space_awareness = [
        "What is space domain awareness and why does it matter?",
        "How do countries track objects in space?",
        "What is the state of space debris and conjunction risk?",
        "How do space situational awareness capabilities differ by nation?",
        "What is the role of commercial space tracking?",
        "How does space weather affect satellite operations?",
        "What are the implications of anti-satellite weapons tests?",
        "How do rendezvous and proximity operations get monitored?",
        "What is the state of space traffic management?",
        "How do nations protect critical space assets?",
        "What is the attribution challenge in space?",
        "How does dual-use technology complicate space security?",
        "What is the role of space in military operations?",
        "How do space treaties and norms evolve?",
        "What is the trajectory of space militarization?",
    ]
    for q in space_awareness:
        examples.append({"instruction": q, "input": "", "output": "[Space domain awareness: tracking capabilities, threat assessment, debris management, and security implications.]"})

    # SPACE ECONOMY
    space_econ = [
        "What is the size and structure of the space economy?",
        "How do commercial launch economics work?",
        "What is the business model for satellite operators?",
        "How does space tourism market develop?",
        "What is the role of government contracts in space industry?",
        "How do space manufacturing prospects develop?",
        "What is the economics of asteroid and lunar resources?",
        "How do space station commercial opportunities evolve?",
        "What is the role of venture capital in space?",
        "How do space SPAC valuations compare to reality?",
        "What is the competitive dynamics among launch providers?",
        "How do small satellite markets affect industry structure?",
        "What is the role of vertical integration in space?",
        "How do insurance and risk affect space economics?",
        "What is the trajectory of space industry consolidation?",
    ]
    for q in space_econ:
        examples.append({"instruction": q, "input": "", "output": "[Space economy analysis: market structure, business models, investment dynamics, and growth trajectories.]"})

    # CONNECTIVITY AND ACCESS
    connectivity = [
        "What is the global state of internet access?",
        "How does the digital divide manifest geographically?",
        "What role does satellite play in bridging connectivity gaps?",
        "How do mobile networks extend access in developing regions?",
        "What is the economics of last-mile connectivity?",
        "How do community networks expand access?",
        "What is the role of public investment in connectivity?",
        "How does connectivity affect economic development?",
        "What are the barriers to universal connectivity?",
        "How do affordability and access interact?",
        "What is the state of connectivity in conflict zones?",
        "How does connectivity affect education and health access?",
        "What role does regulation play in connectivity expansion?",
        "How do connectivity metrics differ from meaningful access?",
        "What is the trajectory for closing the digital divide?",
    ]
    for q in connectivity:
        examples.append({"instruction": q, "input": "", "output": "[Connectivity analysis: access patterns, economic factors, infrastructure requirements, and development implications.]"})

    # EMERGING TECHNOLOGIES
    emerging = [
        "What is the state of 6G research and development?",
        "How might quantum communications affect telecom?",
        "What is the potential of terahertz communications?",
        "How do mesh and decentralized networks evolve?",
        "What is the role of AI in network management?",
        "How might non-terrestrial networks integrate with terrestrial?",
        "What is the trajectory of edge computing in networks?",
        "How do software-defined networks affect infrastructure?",
        "What is the role of network slicing for 5G/6G?",
        "How might blockchain affect telecom infrastructure?",
        "What is the potential of free-space optical communications?",
        "How do neuromorphic computing approaches affect networks?",
        "What is the state of holographic communications research?",
        "How might brain-computer interfaces affect communications?",
        "What convergences are reshaping telecommunications?",
    ]
    for q in emerging:
        examples.append({"instruction": q, "input": "", "output": "[Emerging telecom technology: research status, deployment trajectory, infrastructure implications, and market disruption potential.]"})

    return examples

if __name__ == "__main__":
    examples = generate_examples()
    print(f"Generated {len(examples)} telecom/space examples")
    with open("telecom_space_training.json", "w") as f:
        json.dump(examples, f, indent=2)
    print("Saved to telecom_space_training.json")
