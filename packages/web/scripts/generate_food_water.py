#!/usr/bin/env python3
"""Food security, agriculture systems, and water/hydropolitics training data."""
import json

def generate_examples():
    examples = []

    # FOOD SECURITY
    food_security = [
        "What are the key indicators of food security?",
        "How do food price spikes affect political stability?",
        "What are the main drivers of global food insecurity?",
        "How does climate change threaten food production?",
        "What role do strategic grain reserves play in food security?",
        "How do export bans cascade through food markets?",
        "What is the relationship between energy prices and food prices?",
        "How does food insecurity contribute to migration?",
        "What role does food aid play in geopolitics?",
        "How do sanctions affect food security in target countries?",
        "What is the global distribution of food surplus and deficit?",
        "How do supply chain disruptions affect food availability?",
        "What is the role of speculation in food price volatility?",
        "How does food insecurity affect conflict and instability?",
        "What are the regional food security hotspots?",
    ]
    for q in food_security:
        examples.append({"instruction": q, "input": "", "output": "[Food security analysis: availability factors, access constraints, price dynamics, and stability implications.]"})

    # AGRICULTURAL SYSTEMS
    agriculture = [
        "How is global crop production distributed geographically?",
        "What are the major agricultural trade flows?",
        "How do fertilizer supply chains affect food production?",
        "What is the role of seeds and genetics in agriculture?",
        "How is precision agriculture transforming farming?",
        "What are the economics of different farming systems?",
        "How do agricultural subsidies distort global markets?",
        "What is the state of soil health globally?",
        "How does irrigation water availability affect agriculture?",
        "What are the implications of farming consolidation?",
        "How do labor conditions affect agricultural production?",
        "What is the role of commodity traders in food systems?",
        "How does storage and logistics affect food availability?",
        "What are the environmental impacts of industrial agriculture?",
        "How is organic and regenerative agriculture evolving?",
    ]
    for q in agriculture:
        examples.append({"instruction": q, "input": "", "output": "[Agriculture systems analysis: production geography, input dependencies, trade flows, and sustainability considerations.]"})

    # KEY COMMODITIES
    commodities = [
        "What drives wheat price volatility?",
        "How does rice production and trade flow globally?",
        "What is the state of corn/maize markets?",
        "How does soybean production affect land use?",
        "What are the dynamics of palm oil production and trade?",
        "How does coffee production respond to climate change?",
        "What is the state of global meat and protein markets?",
        "How do dairy markets function internationally?",
        "What are the dynamics of sugar markets?",
        "How does cocoa production concentrate and why?",
        "What is the state of fisheries and aquaculture?",
        "How do fruit and vegetable supply chains work?",
        "What are the dynamics of edible oil markets?",
        "How does feed grain demand affect food markets?",
        "What is the relationship between biofuels and food?",
    ]
    for q in commodities:
        examples.append({"instruction": q, "input": "", "output": "[Commodity analysis: production dynamics, trade patterns, price drivers, and substitution effects.]"})

    # WATER RESOURCES
    water = [
        "What is the global distribution of freshwater resources?",
        "How does groundwater depletion affect long-term water security?",
        "What are the major transboundary water disputes?",
        "How do dams affect downstream water availability?",
        "What is the state of aquifer depletion globally?",
        "How does water pricing affect usage and allocation?",
        "What is the water-energy nexus?",
        "How does urbanization affect water resources?",
        "What is the state of water infrastructure investment?",
        "How does water scarcity affect industrial production?",
        "What is the role of desalination in water security?",
        "How does water pollution affect usable supply?",
        "What are the implications of glacier melt for water supply?",
        "How do water markets and trading function?",
        "What is the virtual water trade and why does it matter?",
    ]
    for q in water:
        examples.append({"instruction": q, "input": "", "output": "[Water resources analysis: supply distribution, depletion trends, infrastructure status, and allocation challenges.]"})

    # HYDROPOLITICS
    hydropolitics = [
        "How does water scarcity contribute to conflict?",
        "What are the major river basins with geopolitical tensions?",
        "How does the Nile water dispute affect regional dynamics?",
        "What are the implications of Chinese dam building on the Mekong?",
        "How does India-Pakistan water sharing affect relations?",
        "What is the state of water cooperation in Central Asia?",
        "How do water disputes affect Middle East geopolitics?",
        "What role does water play in the India-China relationship?",
        "How does Turkey's GAP project affect downstream countries?",
        "What are the dynamics of Colorado River water allocation?",
        "How does water stress affect sub-Saharan African stability?",
        "What is the role of international water law?",
        "How do water treaties hold up under stress?",
        "What are the risks of water conflict escalation?",
        "How can water cooperation build peace?",
    ]
    for q in hydropolitics:
        examples.append({"instruction": q, "input": "", "output": "[Hydropolitics analysis: basin dynamics, treaty frameworks, conflict potential, and cooperation opportunities.]"})

    # AGRICULTURAL INPUTS
    inputs = [
        "How concentrated is the fertilizer industry?",
        "What are the ammonia production and trade dynamics?",
        "How does potash supply chain concentration affect agriculture?",
        "What is the phosphate rock supply situation?",
        "How do pesticide and herbicide markets function?",
        "What is the state of agricultural machinery supply?",
        "How do seed markets and IP affect farmers?",
        "What is the role of agricultural credit in production?",
        "How does diesel availability affect farming operations?",
        "What are the dynamics of animal feed markets?",
        "How does veterinary pharmaceutical supply affect livestock?",
        "What is the state of agricultural technology adoption?",
        "How do input price spikes affect farmer economics?",
        "What is the relationship between sanctions and agricultural inputs?",
        "How does the natural gas price affect fertilizer production?",
    ]
    for q in inputs:
        examples.append({"instruction": q, "input": "", "output": "[Agricultural inputs analysis: supply concentration, price transmission, availability constraints, and farmer economics.]"})

    # FOOD SYSTEM SHOCKS
    shocks = [
        "How did the 2008 food price crisis unfold?",
        "What was the impact of the Ukraine war on food systems?",
        "How do droughts cascade through food systems?",
        "What happens when major exporters impose food export bans?",
        "How do disease outbreaks affect livestock and food supply?",
        "What is the impact of port closures on food trade?",
        "How do currency crises affect food affordability?",
        "What happens when fertilizer supply is disrupted?",
        "How do fuel shortages affect food production and distribution?",
        "What is the impact of extreme heat on crop yields?",
        "How do floods affect food production and infrastructure?",
        "What happens when cold storage and logistics fail?",
        "How do labor shortages affect food systems?",
        "What is the impact of containerization disruptions on food?",
        "How do multiple shocks compound in food systems?",
    ]
    for q in shocks:
        examples.append({"instruction": q, "input": "", "output": "[Food shock analysis: trigger events, transmission mechanisms, vulnerability factors, and cascading effects.]"})

    # FUTURE TRAJECTORIES
    future = [
        "How will climate change affect global food production geography?",
        "What is the potential of vertical farming and controlled environment agriculture?",
        "How might alternative proteins affect food systems?",
        "What is the future of food trade under deglobalization?",
        "How will water scarcity reshape agriculture?",
        "What is the potential of precision fermentation?",
        "How might gene editing transform crop development?",
        "What is the trajectory of agricultural automation?",
        "How will changing diets affect food demand?",
        "What is the future of smallholder farming?",
        "How might carbon pricing affect agriculture?",
        "What is the potential of ocean farming?",
        "How will soil degradation be addressed?",
        "What is the future of food system resilience?",
        "How might food systems achieve sustainability?",
    ]
    for q in future:
        examples.append({"instruction": q, "input": "", "output": "[Future food systems analysis: technology potential, climate adaptation, sustainability pathways, and resilience strategies.]"})

    return examples

if __name__ == "__main__":
    examples = generate_examples()
    print(f"Generated {len(examples)} food/water examples")
    with open("food_water_training.json", "w") as f:
        json.dump(examples, f, indent=2)
    print("Saved to food_water_training.json")
