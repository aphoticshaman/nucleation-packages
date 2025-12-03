#!/usr/bin/env python3
"""
Generate training data on historical cascades, industrial revolutions,
and humanity's trajectory - focusing on higher-order effects and systemic ripples.
"""
import json
from typing import List, Dict

def generate_examples() -> List[Dict]:
    examples = []

    # ==========================================================================
    # 1. INDUSTRIAL REVOLUTIONS & TECHNOLOGICAL CASCADES
    # ==========================================================================
    industrial_revolutions = {
        "first": {
            "period": "1760-1840",
            "innovations": ["steam engine", "spinning jenny", "power loom", "iron smelting", "coal mining", "canals"],
            "questions": [
                "How did the {innovation} trigger cascading changes across British society?",
                "What second-order effects did the {innovation} have on urbanization?",
                "How did the {innovation} reshape labor markets and class structures?",
                "What environmental consequences flowed from widespread {innovation} adoption?",
                "How did the {innovation} affect child labor and family structures?",
                "What geopolitical advantages did early {innovation} adoption confer?",
            ]
        },
        "second": {
            "period": "1870-1914",
            "innovations": ["electricity", "internal combustion engine", "telephone", "steel production", "chemical synthesis", "assembly line"],
            "questions": [
                "How did {innovation} enable entirely new industries and business models?",
                "What cascading urban transformations resulted from {innovation}?",
                "How did {innovation} change warfare and international relations?",
                "What labor movements emerged in response to {innovation}?",
                "How did {innovation} accelerate globalization patterns?",
                "What unintended environmental consequences followed {innovation}?",
            ]
        },
        "third": {
            "period": "1950-2000",
            "innovations": ["transistor", "mainframe computer", "internet", "nuclear power", "satellites", "containerization"],
            "questions": [
                "How did {innovation} create winner-take-all market dynamics?",
                "What societal restructuring followed mass {innovation} adoption?",
                "How did {innovation} change the nature of work and employment?",
                "What geopolitical realignments resulted from {innovation}?",
                "How did {innovation} affect information flows and power structures?",
                "What path dependencies were created by early {innovation} choices?",
            ]
        },
        "fourth": {
            "period": "2010-present",
            "innovations": ["AI/machine learning", "smartphones", "social media", "cloud computing", "CRISPR", "blockchain"],
            "questions": [
                "How is {innovation} creating exponential rather than linear change?",
                "What jobs and industries will {innovation} eliminate or create?",
                "How does {innovation} concentrate or distribute power?",
                "What cascading mental health effects follow from {innovation}?",
                "How might {innovation} reshape democratic institutions?",
                "What existential risks does {innovation} introduce or mitigate?",
            ]
        },
    }

    for rev_name, rev_data in industrial_revolutions.items():
        for innovation in rev_data["innovations"]:
            for q_template in rev_data["questions"]:
                q = q_template.format(innovation=innovation)
                examples.append({
                    "instruction": q,
                    "input": "",
                    "output": f"[Historical cascade analysis: initial conditions, first-order effects, second/third-order ripples, feedback loops, and parallels to current transitions.]"
                })

    # ==========================================================================
    # 2. GREAT DISRUPTIONS & THEIR RIPPLES
    # ==========================================================================
    great_disruptions = {
        "black_death": {
            "event": "Black Death (1347-1351)",
            "questions": [
                "How did the Black Death reshape European labor markets for centuries?",
                "What religious and philosophical shifts cascaded from the plague?",
                "How did depopulation accelerate technological innovation?",
                "What political power realignments followed the Black Death?",
                "How did the plague affect the trajectory of feudalism?",
                "What second-order effects on trade routes emerged from the pandemic?",
            ]
        },
        "columbian_exchange": {
            "event": "Columbian Exchange (1492+)",
            "questions": [
                "How did New World crops cascade through global population dynamics?",
                "What epidemiological catastrophes rippled through the Americas?",
                "How did silver flows reshape global trade and finance?",
                "What agricultural transformations spread across Eurasia and Africa?",
                "How did the exchange alter planetary ecosystems permanently?",
                "What labor system innovations emerged from the exchange?",
            ]
        },
        "world_wars": {
            "event": "World Wars (1914-1945)",
            "questions": [
                "How did WWI cascade into the conditions for WWII?",
                "What technological accelerations resulted from wartime pressures?",
                "How did the wars reshape gender roles and labor markets?",
                "What institutional innovations emerged from total war mobilization?",
                "How did the wars redraw global power hierarchies?",
                "What demographic cascades followed from wartime losses?",
            ]
        },
        "oil_shocks": {
            "event": "Oil Shocks (1973, 1979)",
            "questions": [
                "How did oil price spikes cascade through industrial economies?",
                "What energy policy innovations emerged from the shocks?",
                "How did the shocks reshape automotive and manufacturing?",
                "What geopolitical realignments followed from oil dependency?",
                "How did stagflation change economic policy orthodoxy?",
                "What long-term effects on urban planning emerged from oil shocks?",
            ]
        },
        "internet_revolution": {
            "event": "Internet Revolution (1990s-2000s)",
            "questions": [
                "How did the internet cascade through retail and commerce?",
                "What knowledge work transformations rippled from connectivity?",
                "How did the internet reshape political organization and protest?",
                "What privacy and surveillance dynamics emerged?",
                "How did network effects create new monopoly structures?",
                "What media and journalism disruptions cascaded from the internet?",
            ]
        },
        "covid_pandemic": {
            "event": "COVID-19 Pandemic (2020+)",
            "questions": [
                "How did COVID accelerate pre-existing technological trends?",
                "What labor market restructuring cascaded from the pandemic?",
                "How did supply chain disruptions ripple through the economy?",
                "What political and social polarization effects emerged?",
                "How did the pandemic reshape urban vs. remote work dynamics?",
                "What healthcare system vulnerabilities were exposed?",
            ]
        },
    }

    for event_key, event_data in great_disruptions.items():
        for q in event_data["questions"]:
            examples.append({
                "instruction": q,
                "input": "",
                "output": f"[Historical disruption analysis: triggering event, immediate responses, cascading consequences across decades, counterfactuals, and lessons for future shocks.]"
            })

    # ==========================================================================
    # 3. SUPPLY CHAIN & LOGISTICS EVOLUTION
    # ==========================================================================
    logistics_questions = [
        # Historical evolution
        "How did the Silk Road create cascading cultural and technological exchanges?",
        "What second-order effects followed from the invention of the shipping container?",
        "How did railroad expansion reshape continental power dynamics?",
        "What cascading effects did the Panama Canal have on global trade?",
        "How did the telegraph transform commodity markets and arbitrage?",
        "What supply chain innovations emerged from military logistics?",
        "How did refrigeration cascade through food systems and demographics?",
        "What effects did standardized pallets have on warehouse efficiency?",
        "How did barcode scanning transform inventory management?",
        "What cascading effects followed from just-in-time manufacturing adoption?",

        # Modern complexity
        "How does a chip shortage cascade through dozens of unrelated industries?",
        "What ripple effects follow from a single port closure?",
        "How do rare earth supply constraints cascade through clean energy?",
        "What happens when a critical Tier-3 supplier fails?",
        "How do shipping rate spikes cascade through consumer prices?",
        "What systemic risks emerge from supply chain concentration?",
        "How does a cyberattack on logistics cascade through the economy?",
        "What feedback loops exist between inventory levels and demand?",
        "How do labor disputes at chokepoints cascade globally?",
        "What cascading effects follow from fuel price spikes on logistics?",

        # Future trajectories
        "How might autonomous shipping cascade through port labor markets?",
        "What effects will 3D printing have on global manufacturing geography?",
        "How might drone delivery reshape last-mile logistics?",
        "What cascading changes follow from real-time supply chain visibility?",
        "How will climate change force supply chain redesign?",
        "What happens when nearshoring trends accelerate?",
    ]

    for q in logistics_questions:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Supply chain cascade analysis: initial disruption, propagation pathways, amplification mechanisms, affected sectors, and systemic vulnerabilities.]"
        })

    # ==========================================================================
    # 4. MANUFACTURING & INDUSTRIAL TRANSFORMATIONS
    # ==========================================================================
    manufacturing_questions = [
        # Historical shifts
        "How did interchangeable parts cascade through military and civilian manufacturing?",
        "What ripple effects followed from Ford's assembly line innovation?",
        "How did Toyota's lean manufacturing reshape global industry?",
        "What cascading changes followed from offshore manufacturing?",
        "How did quality management revolutions spread across industries?",
        "What effects did CAD/CAM have on product development cycles?",
        "How did robotics cascade through automotive employment?",
        "What ripple effects followed from Six Sigma adoption?",
        "How did modular design principles reshape product architectures?",
        "What cascading effects emerged from manufacturing ERP systems?",

        # Sectoral cascades
        "How did semiconductor manufacturing advances cascade through all electronics?",
        "What ripple effects follow from battery chemistry breakthroughs?",
        "How do steel production innovations cascade through construction?",
        "What effects did synthetic fiber development have on textiles and agriculture?",
        "How did precision machining advances cascade through aerospace?",
        "What ripple effects follow from pharmaceutical manufacturing innovations?",
        "How did food processing industrialization cascade through agriculture?",
        "What effects did plastic manufacturing have on packaging and commerce?",

        # Geographic shifts
        "How did China's manufacturing rise cascade through global trade patterns?",
        "What ripple effects followed from the decline of Rust Belt manufacturing?",
        "How did special economic zones cascade through development models?",
        "What effects did maquiladora growth have on US-Mexico dynamics?",
        "How did German Mittelstand model cascade through supply networks?",
        "What ripple effects follow from manufacturing automation on labor?",
    ]

    for q in manufacturing_questions:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Manufacturing cascade analysis: innovation diffusion, productivity ripples, labor market effects, geographic shifts, and competitive dynamics.]"
        })

    # ==========================================================================
    # 5. COMMERCE & TRADE PATTERN EVOLUTION
    # ==========================================================================
    commerce_questions = [
        # Historical trade cascades
        "How did spice trade routes cascade through European expansion?",
        "What ripple effects followed from the tea trade with China?",
        "How did cotton trade cascade through slavery and industrialization?",
        "What effects did the fur trade have on North American development?",
        "How did opium trade cascade through China-Britain relations?",
        "What ripple effects followed from the Atlantic slave trade?",
        "How did rubber extraction cascade through colonial systems?",
        "What effects did whaling industry have on lighting and lubrication?",

        # Modern commerce
        "How did e-commerce cascade through retail real estate?",
        "What ripple effects followed from Amazon's marketplace model?",
        "How did subscription models cascade through media and software?",
        "What effects did platform economics have on labor and pricing?",
        "How did fintech cascade through traditional banking?",
        "What ripple effects follow from Buy Now Pay Later adoption?",
        "How did social commerce cascade through marketing and discovery?",
        "What effects did algorithmic pricing have on market dynamics?",

        # Trade policy cascades
        "How do tariffs cascade through supply chains and prices?",
        "What ripple effects follow from trade agreement changes?",
        "How did Bretton Woods collapse cascade through currency markets?",
        "What effects did WTO accession have on developing economies?",
        "How do sanctions cascade through third-party trade relationships?",
        "What ripple effects follow from currency manipulation?",
    ]

    for q in commerce_questions:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Commerce cascade analysis: trade flow changes, market structure evolution, consumer behavior shifts, and regulatory responses.]"
        })

    # ==========================================================================
    # 6. ENERGY TRANSITIONS & ENVIRONMENTAL CASCADES
    # ==========================================================================
    energy_environment_questions = [
        # Energy transition cascades
        "How did the wood-to-coal transition cascade through early industry?",
        "What ripple effects followed from electrification of factories?",
        "How did oil discovery cascade through transportation and geopolitics?",
        "What effects did rural electrification have on agricultural productivity?",
        "How did natural gas adoption cascade through heating and power?",
        "What ripple effects follow from solar cost declines?",
        "How might fusion breakthrough cascade through energy systems?",
        "What effects did nuclear power have on grid planning?",

        # Environmental cascades
        "How did CFCs cascade through atmospheric chemistry before detection?",
        "What ripple effects followed from DDT accumulation in food chains?",
        "How did agricultural runoff cascade through aquatic ecosystems?",
        "What effects did deforestation have on regional climate patterns?",
        "How do microplastics cascade through marine food webs?",
        "What ripple effects follow from coral reef collapse?",
        "How did lead in gasoline cascade through human health?",
        "What effects did the Green Revolution have on soil and water?",

        # Climate cascades
        "How does arctic ice loss cascade through global weather patterns?",
        "What ripple effects follow from permafrost thawing?",
        "How do ocean temperature changes cascade through fisheries?",
        "What effects does sea level rise have on coastal infrastructure?",
        "How do changing precipitation patterns cascade through agriculture?",
        "What ripple effects follow from increased extreme weather frequency?",
        "How does climate migration cascade through receiving regions?",
        "What effects do heat waves have on labor productivity and health?",
    ]

    for q in energy_environment_questions:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Energy/environmental cascade analysis: physical mechanisms, feedback loops, tipping points, socioeconomic ripples, and intervention options.]"
        })

    # ==========================================================================
    # 7. DEMOGRAPHIC & SOCIAL CASCADES
    # ==========================================================================
    demographic_questions = [
        # Population dynamics
        "How did the demographic transition cascade through economic development?",
        "What ripple effects follow from aging population structures?",
        "How did urbanization cascade through family structures and fertility?",
        "What effects did the baby boom have on housing and education?",
        "How does declining fertility cascade through pension systems?",
        "What ripple effects follow from mass migration waves?",
        "How did public health advances cascade through population growth?",
        "What effects did contraception access have on female labor participation?",

        # Social structure shifts
        "How did universal education cascade through labor markets?",
        "What ripple effects followed from women entering the workforce?",
        "How did automobile ownership cascade through suburban development?",
        "What effects did television have on political and consumer culture?",
        "How did air conditioning cascade through southern US development?",
        "What ripple effects follow from smartphone ubiquity?",
        "How did credit card adoption cascade through consumer behavior?",
        "What effects did two-income households have on housing prices?",

        # Inequality dynamics
        "How does educational inequality cascade through generations?",
        "What ripple effects follow from housing wealth concentration?",
        "How did deindustrialization cascade through regional inequality?",
        "What effects does healthcare cost growth have on economic mobility?",
        "How do skill-biased technological changes cascade through wages?",
        "What ripple effects follow from geographic sorting by education?",
    ]

    for q in demographic_questions:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Demographic cascade analysis: population dynamics, social structure evolution, intergenerational effects, and policy implications.]"
        })

    # ==========================================================================
    # 8. FINANCIAL SYSTEM CASCADES
    # ==========================================================================
    financial_questions = [
        # Historical financial cascades
        "How did the South Sea Bubble cascade through British finance and politics?",
        "What ripple effects followed from the 1929 crash and Great Depression?",
        "How did Bretton Woods creation cascade through global finance?",
        "What effects did Nixon closing the gold window have?",
        "How did the 1997 Asian financial crisis cascade regionally?",
        "What ripple effects followed from the 2008 global financial crisis?",
        "How did quantitative easing cascade through asset markets?",
        "What effects did negative interest rates have on banking models?",

        # Systemic risk dynamics
        "How do bank failures cascade through counterparty networks?",
        "What ripple effects follow from sovereign debt crises?",
        "How does currency crisis cascade through import-dependent economies?",
        "What effects do margin calls have on market-wide liquidity?",
        "How do rating downgrades cascade through bond markets?",
        "What ripple effects follow from repo market dysfunction?",
        "How does shadow banking stress cascade to regulated banks?",
        "What effects do algorithmic trading failures have on market stability?",

        # Innovation cascades
        "How did credit scoring cascade through lending access?",
        "What ripple effects followed from securitization innovation?",
        "How did derivatives cascade through risk distribution?",
        "What effects did electronic trading have on market structure?",
        "How does cryptocurrency cascade through monetary policy?",
        "What ripple effects follow from central bank digital currencies?",
    ]

    for q in financial_questions:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Financial cascade analysis: initial shock, transmission mechanisms, amplification dynamics, contagion pathways, and policy responses.]"
        })

    # ==========================================================================
    # 9. HUMANITY'S TRAJECTORY - BIG PICTURE
    # ==========================================================================
    trajectory_questions = [
        # Long-arc patterns
        "What are the recurring patterns in humanity's technological revolutions?",
        "How do innovation waves cascade through social and political structures?",
        "What determines whether a disruption leads to progress or collapse?",
        "How do path dependencies from early choices constrain future options?",
        "What role does energy availability play in civilizational trajectories?",
        "How do information revolutions cascade through power structures?",
        "What patterns emerge in the rise and fall of great powers?",
        "How do demographic transitions cascade through economic development?",

        # Inflection points
        "What made the Scientific Revolution cascade into sustained progress?",
        "How did the Agricultural Revolution cascade into civilization?",
        "What allowed the Industrial Revolution to escape the Malthusian trap?",
        "How might AI cascade into a new phase of human development?",
        "What determines whether technological change is liberating or oppressive?",
        "How do revolutions in transportation cascade through political geography?",
        "What patterns emerge in how societies adapt to rapid change?",
        "How do crises cascade into institutional innovation or decay?",

        # Future trajectories
        "What cascade dynamics might lead to civilizational collapse?",
        "How might space colonization cascade through human development?",
        "What effects might radical life extension have on society?",
        "How might superintelligent AI cascade through all human systems?",
        "What determines whether climate change leads to cooperation or conflict?",
        "How might biotechnology cascade through human evolution?",
        "What cascade dynamics could lead to post-scarcity economics?",
        "How might global governance evolve in response to existential risks?",
    ]

    for q in trajectory_questions:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Civilizational trajectory analysis: historical patterns, causal mechanisms, feedback dynamics, counterfactuals, and scenario implications.]"
        })

    # ==========================================================================
    # 10. SPECIFIC HISTORICAL CASCADES - DEEP DIVES
    # ==========================================================================
    deep_dives = [
        # Printing press cascade
        "How did Gutenberg's press cascade through religious authority?",
        "What effects did printing have on scientific knowledge accumulation?",
        "How did cheap books cascade through literacy and education?",
        "What ripple effects followed from vernacular Bible printing?",
        "How did printing cascade through legal and administrative systems?",

        # Steam engine cascade
        "How did steam power cascade from mines to factories to ships to trains?",
        "What effects did steam have on the concentration of manufacturing?",
        "How did steam-powered transport cascade through agricultural markets?",
        "What ripple effects followed from steam-enabled colonial expansion?",
        "How did steam cascade through military and naval power?",

        # Electrification cascade
        "How did electric lighting cascade through work hours and productivity?",
        "What effects did electric motors have on factory layout and efficiency?",
        "How did household electrification cascade through domestic labor?",
        "What ripple effects followed from electric communication systems?",
        "How did electrification cascade through entertainment and culture?",

        # Computer/Internet cascade
        "How did mainframe computing cascade through corporate organization?",
        "What effects did personal computers have on knowledge work?",
        "How did email cascade through business communication?",
        "What ripple effects followed from web browser adoption?",
        "How did mobile internet cascade through daily life and commerce?",
        "What effects did social media have on information and politics?",
        "How did cloud computing cascade through software business models?",
        "What ripple effects follow from AI assistant adoption?",
    ]

    for q in deep_dives:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Deep historical cascade analysis: specific innovation pathway, adoption dynamics, sectoral propagation, unintended consequences, and lasting legacies.]"
        })

    return examples


def main():
    print("=" * 70)
    print("HISTORICAL CASCADE & HUMANITY TRAJECTORY DATA GENERATOR")
    print("=" * 70)

    examples = generate_examples()
    print(f"\nGenerated {len(examples)} training examples")

    # Categorize
    categories = {
        "industrial_revolutions": 0,
        "great_disruptions": 0,
        "supply_chain": 0,
        "manufacturing": 0,
        "commerce": 0,
        "energy_environment": 0,
        "demographic": 0,
        "financial": 0,
        "trajectory": 0,
        "deep_dives": 0,
        "other": 0,
    }

    for ex in examples:
        text = ex["instruction"].lower()
        if any(x in text for x in ["spinning", "assembly line", "transistor", "ai/machine", "smartphone", "crispr"]):
            categories["industrial_revolutions"] += 1
        elif any(x in text for x in ["black death", "columbian", "world war", "oil shock", "covid", "pandemic"]):
            categories["great_disruptions"] += 1
        elif any(x in text for x in ["supply chain", "logistics", "shipping", "container", "port", "inventory"]):
            categories["supply_chain"] += 1
        elif any(x in text for x in ["manufacturing", "assembly", "factory", "lean", "robotics", "cad"]):
            categories["manufacturing"] += 1
        elif any(x in text for x in ["commerce", "trade", "e-commerce", "retail", "tariff", "merchant"]):
            categories["commerce"] += 1
        elif any(x in text for x in ["energy", "coal", "oil", "solar", "climate", "environment", "emission"]):
            categories["energy_environment"] += 1
        elif any(x in text for x in ["demographic", "population", "aging", "fertility", "migration", "urban"]):
            categories["demographic"] += 1
        elif any(x in text for x in ["financial", "bank", "crisis", "credit", "currency", "debt"]):
            categories["financial"] += 1
        elif any(x in text for x in ["trajectory", "civilization", "humanity", "long-arc", "existential"]):
            categories["trajectory"] += 1
        elif any(x in text for x in ["gutenberg", "steam", "electric", "computer", "internet", "printing"]):
            categories["deep_dives"] += 1
        else:
            categories["other"] += 1

    print("\n" + "=" * 70)
    print("CATEGORY BREAKDOWN:")
    print("=" * 70)
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} examples")

    # Save
    output_path = "cascade_history_training.json"
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"\nSaved to {output_path}")

    return examples


if __name__ == "__main__":
    main()
