#!/usr/bin/env python3
"""Insurance, actuarial risk, and real estate as economic indicator training data."""
import json

def generate_examples():
    examples = []

    # INSURANCE RISK ASSESSMENT
    insurance_risk = [
        "How do insurers assess and price climate-related risks?",
        "What is the reinsurance market and how does it function?",
        "How do catastrophe models inform insurance pricing?",
        "What role does insurance play in economic stability?",
        "How do underwriting cycles affect insurance availability?",
        "What is the state of cyber insurance and its pricing challenges?",
        "How do insurers handle correlated risks?",
        "What is the role of insurance in supply chain risk?",
        "How does political risk insurance work?",
        "What are the implications of uninsurability for markets?",
        "How do loss reserves and their adequacy get assessed?",
        "What is the role of alternative risk transfer?",
        "How do insurance-linked securities function?",
        "What are the implications of insurance withdrawal from markets?",
        "How does insurance regulation affect risk capacity?",
    ]
    for q in insurance_risk:
        examples.append({"instruction": q, "input": "", "output": "[Insurance risk analysis: pricing methodology, capacity constraints, market dynamics, and systemic implications.]"})

    # ACTUARIAL METHODS
    actuarial = [
        "How do actuaries model long-tail liability risks?",
        "What is the role of mortality tables in insurance pricing?",
        "How do actuaries handle parameter uncertainty?",
        "What is the actuarial approach to climate risk?",
        "How do reserve adequacy studies work?",
        "What is the role of stochastic modeling in actuarial work?",
        "How do actuaries model pandemic risks?",
        "What is the approach to modeling cyber risk?",
        "How do life and non-life actuarial methods differ?",
        "What is the role of credibility theory?",
        "How do actuaries model extreme events?",
        "What is the approach to modeling emerging risks?",
        "How do regulatory capital models work?",
        "What is the role of scenario testing in actuarial work?",
        "How do actuaries handle model risk?",
    ]
    for q in actuarial:
        examples.append({"instruction": q, "input": "", "output": "[Actuarial analysis: modeling methodology, uncertainty quantification, regulatory requirements, and emerging risk approaches.]"})

    # INSURANCE MARKET STRUCTURE
    insurance_market = [
        "How is the global insurance market structured?",
        "What role do Lloyd's of London syndicates play?",
        "How do mutual and stock insurers differ?",
        "What is the role of captive insurance?",
        "How do insurance brokers affect market dynamics?",
        "What is the role of run-off and legacy markets?",
        "How do Bermuda and other offshore markets function?",
        "What is the state of insurtech and its impact?",
        "How do parametric insurance products work?",
        "What is the role of microinsurance in development?",
        "How do trade credit insurance markets function?",
        "What is the structure of the life insurance market?",
        "How do annuity markets and pension buyouts work?",
        "What is the role of specialty insurance markets?",
        "How do terrorism risk insurance schemes work?",
    ]
    for q in insurance_market:
        examples.append({"instruction": q, "input": "", "output": "[Insurance market analysis: structure, participant roles, product innovation, and market access dynamics.]"})

    # REAL ESTATE AS INDICATOR
    re_indicator = [
        "How does commercial real estate signal economic conditions?",
        "What do office vacancy rates indicate about the economy?",
        "How do housing starts predict economic activity?",
        "What do mortgage application trends signal?",
        "How does real estate capital flow indicate investor sentiment?",
        "What do REIT performance patterns indicate?",
        "How does construction activity signal economic cycles?",
        "What do rental rate trends indicate about affordability?",
        "How do home price indices predict economic conditions?",
        "What do building permit trends signal?",
        "How does industrial real estate demand indicate trade patterns?",
        "What do retail vacancy rates signal about consumption?",
        "How do hotel occupancy rates indicate travel and business activity?",
        "What do multifamily starts indicate about housing demand?",
        "How does land price appreciation signal market conditions?",
    ]
    for q in re_indicator:
        examples.append({"instruction": q, "input": "", "output": "[Real estate indicator analysis: signal interpretation, leading vs lagging properties, and economic cycle relationships.]"})

    # REAL ESTATE MARKETS
    re_markets = [
        "How do real estate cycles differ from business cycles?",
        "What drives commercial real estate cap rate compression?",
        "How do interest rates affect real estate valuations?",
        "What is the role of institutional investors in real estate?",
        "How do CMBS markets affect real estate finance?",
        "What is the state of global real estate capital flows?",
        "How does real estate development respond to demand signals?",
        "What is the role of leverage in real estate returns?",
        "How do real estate markets vary by property type?",
        "What is the role of real estate in inflation hedging?",
        "How do zoning and land use affect market dynamics?",
        "What is the impact of work-from-home on office markets?",
        "How do demographic shifts affect housing demand?",
        "What is the role of foreign investment in real estate?",
        "How do real estate taxes affect investment decisions?",
    ]
    for q in re_markets:
        examples.append({"instruction": q, "input": "", "output": "[Real estate market analysis: cycle dynamics, capital flows, valuation drivers, and property type differentiation.]"})

    # CLIMATE AND REAL ESTATE
    climate_re = [
        "How does flood risk affect property values?",
        "What is the impact of wildfire risk on real estate?",
        "How are coastal property markets adapting to sea level rise?",
        "What is the state of climate risk disclosure in real estate?",
        "How does heat stress affect property usability?",
        "What are the implications of insurance retreat for property values?",
        "How do green building certifications affect values?",
        "What is the role of energy efficiency in real estate valuation?",
        "How are mortgage markets pricing climate risk?",
        "What is the trajectory of stranded asset risk in real estate?",
        "How do physical climate risks vary by geography?",
        "What is the role of resilience investments in property value?",
        "How are real estate investors incorporating climate scenarios?",
        "What is the regulatory trajectory for climate disclosure in real estate?",
        "How might climate migration affect regional property markets?",
    ]
    for q in climate_re:
        examples.append({"instruction": q, "input": "", "output": "[Climate-real estate nexus: risk assessment, valuation implications, adaptation strategies, and regulatory trajectories.]"})

    # RISK PRICING
    risk_pricing = [
        "How do markets price tail risks?",
        "What is the role of risk premiums in asset pricing?",
        "How do credit spreads reflect risk perceptions?",
        "What is the term premium and how does it behave?",
        "How does volatility pricing reflect market expectations?",
        "What role do risk-free rates play in valuation?",
        "How do equity risk premiums vary over time?",
        "What is the liquidity premium and how is it measured?",
        "How do currency risk premiums function?",
        "What is the role of inflation risk premiums?",
        "How do commodity risk premiums behave?",
        "What is the relationship between risk and return empirically?",
        "How do behavioral factors affect risk pricing?",
        "What role does regulation play in risk pricing?",
        "How has risk pricing evolved with market structure changes?",
    ]
    for q in risk_pricing:
        examples.append({"instruction": q, "input": "", "output": "[Risk pricing analysis: premium dynamics, measurement methods, market structure effects, and empirical patterns.]"})

    return examples

if __name__ == "__main__":
    examples = generate_examples()
    print(f"Generated {len(examples)} insurance/real estate examples")
    with open("insurance_realestate_training.json", "w") as f:
        json.dump(examples, f, indent=2)
    print("Saved to insurance_realestate_training.json")
