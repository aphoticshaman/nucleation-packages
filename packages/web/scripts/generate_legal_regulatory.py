#!/usr/bin/env python3
"""Legal frameworks, regulatory systems, and compliance training data."""
import json

def generate_examples():
    examples = []

    # JURISDICTIONAL FRAMEWORKS
    jurisdiction = [
        "How do common law and civil law systems differ for business?",
        "What are the key regulatory differences between US and EU?",
        "How does China's legal system affect foreign businesses?",
        "What is the role of offshore jurisdictions in global finance?",
        "How do tax treaty networks affect corporate structuring?",
        "What are the implications of Brexit for regulatory alignment?",
        "How do US extraterritorial laws affect foreign companies?",
        "What is the role of the FCPA in international business?",
        "How do different countries approach data protection?",
        "What are the key differences in employment law by jurisdiction?",
        "How do intellectual property regimes differ internationally?",
        "What is the role of bilateral investment treaties?",
        "How do arbitration venues differ in their approaches?",
        "What are the enforcement mechanisms for international judgments?",
        "How do anti-corruption frameworks differ by region?",
    ]
    for q in jurisdiction:
        examples.append({"instruction": q, "input": "", "output": "[Jurisdictional analysis: legal system characteristics, regulatory requirements, enforcement mechanisms, and compliance considerations.]"})

    # FINANCIAL REGULATION
    fin_reg = [
        "How do banking regulations differ across major jurisdictions?",
        "What is the Basel framework and how is it implemented?",
        "How do securities regulations vary internationally?",
        "What are the key AML/KYC requirements by region?",
        "How do sanctions compliance requirements work?",
        "What is the regulatory framework for cryptocurrency?",
        "How do investment fund regulations differ globally?",
        "What are the key insurance regulatory frameworks?",
        "How do capital requirements affect bank behavior?",
        "What is the role of macroprudential regulation?",
        "How do cross-border financial regulations interact?",
        "What are the implications of regulatory fragmentation?",
        "How do stress testing requirements differ by jurisdiction?",
        "What is the regulatory approach to fintech?",
        "How do climate disclosure requirements affect finance?",
    ]
    for q in fin_reg:
        examples.append({"instruction": q, "input": "", "output": "[Financial regulation analysis: framework requirements, implementation variation, compliance obligations, and cross-border considerations.]"})

    # TRADE AND INVESTMENT
    trade = [
        "How do WTO rules affect international trade?",
        "What are the key provisions of major free trade agreements?",
        "How do rules of origin work in trade agreements?",
        "What is the role of trade remedies and how are they used?",
        "How do investment screening mechanisms work?",
        "What are the implications of CFIUS for foreign investment?",
        "How do export control regimes function?",
        "What is the EU's foreign subsidy regulation?",
        "How do local content requirements affect market access?",
        "What are the implications of trade agreement withdrawal?",
        "How do bilateral sanctions interact with trade rules?",
        "What is the role of trade finance regulation?",
        "How do forced technology transfer concerns shape policy?",
        "What are the key provisions of supply chain due diligence laws?",
        "How do carbon border adjustments work legally?",
    ]
    for q in trade:
        examples.append({"instruction": q, "input": "", "output": "[Trade regulation analysis: rule frameworks, compliance requirements, market access implications, and policy trend trajectories.]"})

    # COMPETITION/ANTITRUST
    competition = [
        "How do antitrust frameworks differ between US and EU?",
        "What triggers merger review in major jurisdictions?",
        "How are market definitions determined in antitrust?",
        "What is the approach to platform regulation and competition?",
        "How do antitrust remedies differ by jurisdiction?",
        "What is the role of vertical restraints analysis?",
        "How do bundling and tying practices get scrutinized?",
        "What is the approach to killer acquisitions?",
        "How do pricing practices trigger antitrust concerns?",
        "What is the role of competitor cooperation rules?",
        "How do state aid rules affect competition in the EU?",
        "What is the approach to essential facilities and access?",
        "How do antitrust authorities cooperate internationally?",
        "What are the trends in antitrust enforcement?",
        "How do labor market effects enter antitrust analysis?",
    ]
    for q in competition:
        examples.append({"instruction": q, "input": "", "output": "[Competition law analysis: framework differences, trigger thresholds, remedy approaches, and enforcement trends.]"})

    # DATA AND PRIVACY
    privacy = [
        "How does GDPR affect global business operations?",
        "What are the key differences between GDPR and US privacy laws?",
        "How do data localization requirements affect technology?",
        "What are the implications of cross-border data transfer rules?",
        "How do consent requirements differ by jurisdiction?",
        "What are the key provisions of China's PIPL?",
        "How do data breach notification requirements differ?",
        "What is the approach to AI and automated decision-making regulation?",
        "How do children's privacy rules differ internationally?",
        "What are the implications of biometric data regulations?",
        "How do employee privacy rules vary by jurisdiction?",
        "What is the regulatory approach to health data?",
        "How do privacy frameworks affect marketing practices?",
        "What are the enforcement mechanisms for privacy violations?",
        "How is privacy intersecting with national security?",
    ]
    for q in privacy:
        examples.append({"instruction": q, "input": "", "output": "[Privacy regulation analysis: framework requirements, jurisdictional variation, compliance obligations, and enforcement mechanisms.]"})

    # ENVIRONMENTAL REGULATION
    environmental = [
        "How do carbon pricing mechanisms differ globally?",
        "What are the key provisions of environmental due diligence laws?",
        "How do emissions reporting requirements vary?",
        "What is the regulatory approach to ESG disclosure?",
        "How do water quality regulations differ by jurisdiction?",
        "What are the key provisions of circular economy regulation?",
        "How do chemical regulations like REACH affect products?",
        "What is the approach to plastics regulation?",
        "How do biodiversity regulations affect business?",
        "What are the implications of CBAM for international trade?",
        "How do environmental liability regimes differ?",
        "What is the regulatory approach to green finance?",
        "How do environmental permits and approvals work?",
        "What are the trends in climate litigation?",
        "How do net-zero commitments translate into regulation?",
    ]
    for q in environmental:
        examples.append({"instruction": q, "input": "", "output": "[Environmental regulation analysis: mechanism design, disclosure requirements, compliance pathways, and enforcement trends.]"})

    # LABOR AND EMPLOYMENT
    labor = [
        "How do employment termination rules differ internationally?",
        "What are the key collective bargaining frameworks?",
        "How do minimum wage regulations vary globally?",
        "What is the approach to working time regulation?",
        "How do gig economy regulations differ by jurisdiction?",
        "What are the key provisions of supply chain labor due diligence?",
        "How do discrimination and equality laws vary?",
        "What is the approach to worker classification?",
        "How do occupational safety regulations differ?",
        "What are the implications of works council requirements?",
        "How do immigration work permit systems function?",
        "What is the approach to executive compensation regulation?",
        "How do benefit and pension regulations vary?",
        "What are the trends in labor law reform?",
        "How do remote work regulations affect international employment?",
    ]
    for q in labor:
        examples.append({"instruction": q, "input": "", "output": "[Labor regulation analysis: framework requirements, cross-border considerations, compliance obligations, and reform trends.]"})

    # SECTOR-SPECIFIC
    sector = [
        "How do telecom regulations affect market entry?",
        "What are the key provisions of defense procurement rules?",
        "How do pharmaceutical regulations differ internationally?",
        "What is the regulatory approach to autonomous vehicles?",
        "How do aviation regulations affect international operations?",
        "What are the key provisions of maritime regulation?",
        "How do food safety regulations differ by jurisdiction?",
        "What is the approach to medical device regulation?",
        "How do nuclear regulations affect civilian applications?",
        "What are the key provisions of space activity regulation?",
        "How do gambling and gaming regulations differ?",
        "What is the approach to alcohol and tobacco regulation?",
        "How do professional licensing requirements vary?",
        "What are the trends in AI-specific regulation?",
        "How do social media content regulations differ?",
    ]
    for q in sector:
        examples.append({"instruction": q, "input": "", "output": "[Sector regulation analysis: framework requirements, market access implications, compliance obligations, and regulatory trajectory.]"})

    return examples

if __name__ == "__main__":
    examples = generate_examples()
    print(f"Generated {len(examples)} legal/regulatory examples")
    with open("legal_regulatory_training.json", "w") as f:
        json.dump(examples, f, indent=2)
    print("Saved to legal_regulatory_training.json")
