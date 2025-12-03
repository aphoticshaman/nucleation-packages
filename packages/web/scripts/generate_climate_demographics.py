#!/usr/bin/env python3
"""Climate modeling, tipping points, demographics, and migration training data."""
import json

def generate_examples():
    examples = []

    # CLIMATE TIPPING POINTS
    tipping = [
        "What are the major climate tipping points and their triggers?",
        "How might Amazon rainforest dieback unfold?",
        "What are the implications of Arctic sea ice loss?",
        "How does permafrost thaw create feedback loops?",
        "What is the risk of ice sheet collapse in Greenland and Antarctica?",
        "How might Atlantic circulation (AMOC) weakening affect climate?",
        "What are the coral reef tipping point dynamics?",
        "How do monsoon system changes affect billions of people?",
        "What is the boreal forest die-off risk?",
        "How do tipping point cascades and interactions work?",
        "What early warning signals exist for tipping points?",
        "How do tipping points affect climate model projections?",
        "What is the relationship between warming levels and tipping points?",
        "How do tipping points affect the carbon budget?",
        "What are the economic implications of tipping point scenarios?",
    ]
    for q in tipping:
        examples.append({"instruction": q, "input": "", "output": "[Climate tipping point analysis: trigger thresholds, cascade dynamics, warning indicators, and impact projections.]"})

    # PHYSICAL CLIMATE
    physical = [
        "How do global temperature projections vary by scenario?",
        "What are the regional patterns of climate change?",
        "How is precipitation changing globally?",
        "What is the trajectory of sea level rise?",
        "How are extreme weather events changing in frequency and intensity?",
        "What is the state of ocean acidification?",
        "How are heat waves and extreme heat evolving?",
        "What are the patterns of drought intensification?",
        "How are tropical cyclones changing?",
        "What is the trajectory of flooding risk?",
        "How is wildfire risk evolving globally?",
        "What are the patterns of seasonal shift?",
        "How is snow and ice coverage changing?",
        "What are the implications for ecosystem shifts?",
        "How do climate models project regional differences?",
    ]
    for q in physical:
        examples.append({"instruction": q, "input": "", "output": "[Physical climate analysis: observed trends, model projections, regional variation, and uncertainty quantification.]"})

    # CLIMATE ECONOMICS
    climate_econ = [
        "How do economic models estimate climate damages?",
        "What is the social cost of carbon and how is it calculated?",
        "How does climate change affect GDP projections?",
        "What are the distributional effects of climate change?",
        "How do climate damages compound over time?",
        "What is the economics of climate adaptation?",
        "How do stranded asset risks affect energy investments?",
        "What are the macroeconomic effects of climate policy?",
        "How does climate change affect insurance and risk pricing?",
        "What are the labor productivity effects of heat?",
        "How does climate change affect agricultural economics?",
        "What are the infrastructure cost implications of climate?",
        "How do climate scenarios affect financial risk assessment?",
        "What is the economics of climate migration?",
        "How do climate damages affect sovereign creditworthiness?",
    ]
    for q in climate_econ:
        examples.append({"instruction": q, "input": "", "output": "[Climate economics analysis: damage functions, cost estimates, distributional effects, and financial risk implications.]"})

    # DEMOGRAPHIC TRANSITIONS
    demographics = [
        "What is the global demographic transition and where are different countries?",
        "How does aging population affect economic growth?",
        "What are the implications of declining fertility rates?",
        "How does the youth bulge affect stability in some regions?",
        "What are the dependency ratio trends by region?",
        "How do demographic windows of opportunity work?",
        "What is the demographic outlook for major economies?",
        "How does education affect demographic transition?",
        "What are the implications of population decline scenarios?",
        "How do demographic changes affect labor markets?",
        "What is the relationship between urbanization and demographics?",
        "How do demographic trends affect pension systems?",
        "What are the healthcare demand implications of aging?",
        "How do demographic changes affect housing markets?",
        "What is the relationship between demographics and political power?",
    ]
    for q in demographics:
        examples.append({"instruction": q, "input": "", "output": "[Demographic analysis: transition stage, trajectory projections, economic implications, and policy considerations.]"})

    # MIGRATION PATTERNS
    migration = [
        "What are the major global migration corridors?",
        "How do economic factors drive migration decisions?",
        "What role does conflict play in forced displacement?",
        "How does climate change drive migration?",
        "What are the demographics of migration flows?",
        "How do remittances affect origin country economics?",
        "What is the economic impact of migration on receiving countries?",
        "How do migration policies affect flow patterns?",
        "What are the labor market effects of immigration?",
        "How does brain drain affect developing countries?",
        "What are the integration challenges for migrants?",
        "How do refugee flows differ from economic migration?",
        "What is the relationship between migration and demographics?",
        "How do diaspora networks affect migration patterns?",
        "What are the projections for climate migration?",
    ]
    for q in migration:
        examples.append({"instruction": q, "input": "", "output": "[Migration analysis: driver factors, flow patterns, economic effects, and policy implications for origin and destination.]"})

    # URBANIZATION
    urban = [
        "What is the trajectory of global urbanization?",
        "How do megacities affect resource consumption?",
        "What are the infrastructure challenges of rapid urbanization?",
        "How does urbanization affect climate vulnerability?",
        "What is the informal settlement trajectory in developing countries?",
        "How does urbanization affect food systems?",
        "What are the mobility challenges in growing cities?",
        "How does urbanization affect water resources?",
        "What is the relationship between urbanization and emissions?",
        "How do urban heat islands affect livability?",
        "What are the governance challenges of megacities?",
        "How does urbanization affect public health?",
        "What is the future of secondary cities?",
        "How do smart city technologies affect urban development?",
        "What are the resilience challenges for coastal cities?",
    ]
    for q in urban:
        examples.append({"instruction": q, "input": "", "output": "[Urbanization analysis: growth trajectories, infrastructure needs, climate vulnerability, and governance challenges.]"})

    # CLIMATE MIGRATION NEXUS
    climate_migration = [
        "How does sea level rise create climate refugees?",
        "What is the impact of desertification on population movement?",
        "How do changing agricultural zones affect rural migration?",
        "What are the small island state climate migration scenarios?",
        "How does water scarcity drive climate migration?",
        "What is the relationship between extreme weather and displacement?",
        "How do climate migrants differ from other migrants legally?",
        "What are the internal vs. cross-border climate migration dynamics?",
        "How might climate migration affect political stability?",
        "What are the adaptation options vs. migration tradeoffs?",
        "How do climate projections inform migration forecasting?",
        "What is the timeline for major climate-driven displacements?",
        "How can receiving areas prepare for climate migration?",
        "What role does planned relocation play?",
        "How might climate migration reshape global population distribution?",
    ]
    for q in climate_migration:
        examples.append({"instruction": q, "input": "", "output": "[Climate-migration nexus: driver mechanisms, displacement scenarios, timeline projections, and policy responses.]"})

    # POPULATION HEALTH
    health = [
        "How does climate change affect disease patterns?",
        "What are the heat-related mortality projections?",
        "How does air quality affect population health?",
        "What is the relationship between climate and infectious disease?",
        "How do demographic transitions affect health systems?",
        "What are the mental health implications of climate change?",
        "How does aging affect health system capacity?",
        "What is the relationship between migration and health?",
        "How do climate and demographic trends affect nutrition?",
        "What are the vector-borne disease range expansion patterns?",
        "How does urbanization affect health outcomes?",
        "What are the occupational health effects of climate change?",
        "How do climate and demographic changes affect pandemic risk?",
        "What is the trajectory of antibiotic resistance?",
        "How do health systems adapt to climate and demographic change?",
    ]
    for q in health:
        examples.append({"instruction": q, "input": "", "output": "[Population health analysis: climate-health pathways, demographic drivers, system capacity, and adaptation requirements.]"})

    return examples

if __name__ == "__main__":
    examples = generate_examples()
    print(f"Generated {len(examples)} climate/demographics examples")
    with open("climate_demographics_training.json", "w") as f:
        json.dump(examples, f, indent=2)
    print("Saved to climate_demographics_training.json")
