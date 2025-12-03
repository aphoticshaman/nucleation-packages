#!/usr/bin/env python3
"""
Generate training data with REAL historical facts, dates, figures, and context.
This is grounded in actual events, not just question templates.
"""
import json

def generate_examples():
    examples = []

    # ==========================================================================
    # SPACE INDUSTRY MILESTONES
    # ==========================================================================
    space_facts = [
        # SpaceX launches
        {"q": "How many orbital launches did SpaceX complete in 2024?", "a": "SpaceX completed 134 orbital launches in 2024, up nearly 40% from 96 launches in 2023. This made 2024 a record-breaking year, with global launches reaching 259 total—more than double 2019-2020 levels."},
        {"q": "How many Starlink satellites are in orbit as of 2024?", "a": "SpaceX put its 6,000th Starlink satellite in orbit on May 17, 2024, reaching the halfway point of its planned 12,000 satellite constellation. Starlink now comprises 65% of all active satellites globally."},
        {"q": "What is Starlink's revenue and subscriber count?", "a": "Starlink's revenue jumped from $1.4 billion in 2022 to a projected $6.6 billion in 2024. Subscriber count grew from 1 million (Dec 2022) to 4 million (Sep 2024). 2024 marked the first year of profitability with $72.7 million net profit after a $30.7 million loss in 2023."},
        {"q": "What milestones did SpaceX achieve with Starship in 2024?", "a": "SpaceX flew four near-orbital Starship tests from Starbase in 2024, including the historic Flight 5 in October where the Super Heavy booster was caught by the launch tower's 'chopsticks' for the first time. This demonstrated rapid reusability potential."},
        {"q": "What reuse records has the Falcon 9 achieved?", "a": "On May 17, 2024, SpaceX set a new record by launching and landing a Falcon 9 first stage for the 21st time—for a rocket originally designed to last 10 launches. This dramatically reduced launch costs."},
        {"q": "What was significant about the Polaris Dawn mission?", "a": "The Polaris Dawn mission in 2024 featured the first-ever private spacewalk, marking a major milestone in commercial space exploration and private astronaut activity beyond the ISS."},

        # Other space developments
        {"q": "What rockets retired and debuted in 2024?", "a": "2024 saw the retirement of ULA's Delta IV rocket and the original Arianespace Vega. New rockets debuting included Ariane 6, ULA's Vulcan, and China's Chang Zheng 12."},
    ]
    for f in space_facts:
        examples.append({"instruction": f["q"], "input": "", "output": f["a"]})

    # ==========================================================================
    # BIOTECH/PHARMA BREAKTHROUGHS
    # ==========================================================================
    biotech_facts = [
        {"q": "What made December 8, 2023 historic for gene therapy?", "a": "December 8, 2023 was called 'the biggest day in the history of gene therapy' when the FDA approved two gene therapies for sickle cell disease: Vertex Pharmaceuticals' Casgevy (the first-ever CRISPR-based therapy approved by FDA) and Bluebird Bio's Lyfgenia. Jennifer Doudna noted: 'Going from the lab to an approved CRISPR therapy in just 11 years is a truly remarkable achievement.'"},
        {"q": "How many FDA drug approvals were there in 2023?", "a": "2023 saw a record-breaking 54 new drugs and 7 cellular and gene therapy products approved by the FDA, recovering from the 2022 dip back to 2020-2021 levels. This included five gene therapies and the first CRISPR-Cas9-edited therapy."},
        {"q": "What notable gene therapy approvals occurred in 2024?", "a": "In 2024, the FDA approved: Casgevy's second indication for beta thalassemia; Lenmeldy (Orchard Therapeutics) for metachromatic leukodystrophy; Kebilidi (PTC Therapeutics) for AADC deficiency—the first gene therapy administered directly to the brain; Amtagvi (Iovance) as the first T cell therapy for solid tumors; and Beqvez (Pfizer) as a second gene therapy option for hemophilia B."},
        {"q": "How many FDA-approved gene therapies exist as of 2024?", "a": "As of 2024, there are 17 FDA-approved gene therapies on the market, treating conditions from pediatric leukemia to spinal muscular atrophy, according to Peter Marks, director of the FDA's CBER."},
        {"q": "What is the status of in vivo CRISPR gene editing?", "a": "Intellia started enrolling patients in the Phase III MAGNITUDE trial for nexiguran ziclumeran—the first in vivo CRISPR gene-editing therapy cleared to enter late-stage clinical development in the US. This represents CRISPR moving from ex vivo (cells edited outside body) to direct in-body editing."},
        {"q": "How much is invested in gene therapy research?", "a": "In 2023, there were 1,894 ongoing clinical trials in gene therapy globally, with $11.7 billion in total investment. However, venture funding into biopharma dipped below pandemic levels, forcing companies to be more selective about programs."},
    ]
    for f in biotech_facts:
        examples.append({"instruction": f["q"], "input": "", "output": f["a"]})

    # ==========================================================================
    # FOOD CRISIS & UKRAINE GRAIN
    # ==========================================================================
    food_facts = [
        {"q": "How did Russia's invasion affect Ukraine's grain exports?", "a": "In March, April, and May 2022, after the Russian invasion, Ukraine's exports dropped by over 90%. Before the war, around 90% of Ukraine's agricultural exports were transported by sea. Russia's military blockade of Black Sea ports brought exports to a virtual standstill."},
        {"q": "How important is Ukraine to global food supply?", "a": "Ukraine is the world's largest exporter of sunflower oil (50% of world exports), third largest of barley (18%), fourth largest of maize (16%), and fifth largest of wheat (12%). It possesses some of the most fertile land on Earth."},
        {"q": "What happened to food prices after Russia's invasion?", "a": "Wheat prices were 58% higher in March 2022 than March 2021. The FAO Food Price Index increased by 14.3% in 2022 vs 2021—the highest since records began in 1990. Some wheat contracts hit all-time highs."},
        {"q": "What was the Black Sea Grain Initiative?", "a": "Between July 2022 and July 2023, an agreement between the UN, Turkey, and Russia allowed exports via a safe maritime corridor. Over 1,000 ships exported nearly 33 million tonnes of grain from three Ukrainian ports (Chornomorsk, Odesa, Pivdennyi). The UN estimated this reduced food prices by more than 23%."},
        {"q": "What happened when Russia ended the grain deal?", "a": "On July 17, 2023, Russia announced it would end the Black Sea Grain Initiative. Wheat prices rose about 3% to $6.81/bushel (still half the 2022 peak). Russia then bombed Odesa port, destroying over 60,000 tons of grain."},
        {"q": "What are the long-term food security implications?", "a": "The FAO's 2023 report predicts nearly 600 million people will be chronically undernourished in 2030—about 23 million more than if there had been no war. 36 of the 55 countries already experiencing a food crisis depend on exports from Ukraine and Russia. The World Food Programme purchased 80% of its wheat from Ukraine."},
    ]
    for f in food_facts:
        examples.append({"instruction": f["q"], "input": "", "output": f["a"]})

    # ==========================================================================
    # RARE EARTHS & CRITICAL MINERALS
    # ==========================================================================
    minerals_facts = [
        {"q": "What is China's dominance in rare earths and battery materials?", "a": "For rare earths used in magnets (neodymium, praseodymium, dysprosium, terbium), China accounts for ~60% of global mining output. China dominates battery supply chain midstream with 80%+ shares; for precursor cathode and LFP cathode materials, China maintains 95%+ near-monopoly."},
        {"q": "What export controls has China imposed on critical minerals?", "a": "In December 2024, China restricted exports of gallium, germanium, and antimony—key semiconductor materials—to the United States. On October 9, 2025, new controls on rare earths, lithium batteries, and diamond technologies were announced, effective November 8, 2025—the most comprehensive since 2010 WTO-challenged quotas."},
        {"q": "What happened to lithium prices after 2022?", "a": "Lithium prices, which had surged eightfold during 2021-22, fell by over 80% since 2023, returning to pre-pandemic levels. Graphite, cobalt, and nickel prices also dropped 10-20% in 2024 as post-shortage capacity came online."},
        {"q": "Why is diversifying rare earth supply difficult?", "a": "China dominates not because it holds the largest resource base, but because it invested in processing, refining, and usage technologies. New restrictions on processing equipment risk constraining emerging projects. Western miners can extract but lack processing capacity."},
        {"q": "What is the Western response to critical mineral dependence?", "a": "The US Inflation Reduction Act and Defense Production Act earmarked billions for rare earth separation and magnet production. The EU Critical Raw Materials Act mandates 15% domestic processing by 2030. The UK Critical Minerals Strategy supports recycling and refining projects."},
    ]
    for f in minerals_facts:
        examples.append({"instruction": f["q"], "input": "", "output": f["a"]})

    # ==========================================================================
    # CLIMATE TIPPING POINTS
    # ==========================================================================
    climate_facts = [
        {"q": "What is the current state of the AMOC?", "a": "The Atlantic Meridional Overturning Circulation (AMOC) has declined 15% since 1950 and is in its weakest state in over 1,000 years. Research in Science Advances (2024) indicates AMOC is on route to tipping. Collapse forecasts range from 2025-2095, with central estimates around 2050."},
        {"q": "What would AMOC collapse mean for Europe?", "a": "Under AMOC collapse, European climate would see yearly temperature drops exceeding 1°C per decade over northwestern Europe. Several European cities could experience 5-15°C temperature drops. This would also worsen Amazon drought and accelerate Antarctic ice loss through feedback loops."},
        {"q": "What is the status of Amazon rainforest tipping point?", "a": "Research found up to half the Amazon faces combined stress from heat, drought, and deforestation that could push it to a tipping point by 2050. Over the last 10 years, the Amazon experienced three 'once-in-a-hundred-year' droughts. The rainforest has shifted from a carbon sink to a net source of greenhouse gases."},
        {"q": "How fast is the Arctic warming?", "a": "The Arctic is warming 3-4 times faster than the rest of the world, adding almost 1mm to global sea levels yearly. In 2023, scientists indicated we may have already crossed the West Arctic ice sheet tipping point. The 1.5°C threshold could trigger irreversible Greenland ice sheet melting, now the main factor in sea level rise."},
        {"q": "Has Earth breached the 1.5°C threshold?", "a": "Yes, in 2024 the planet temporarily breached the 1.5°C warming threshold—the goal 194 countries agreed to try not to cross under the Paris Agreement. This increases proximity to multiple tipping point triggers."},
    ]
    for f in climate_facts:
        examples.append({"instruction": f["q"], "input": "", "output": f["a"]})

    # ==========================================================================
    # SEMICONDUCTOR SHORTAGE
    # ==========================================================================
    chip_facts = [
        {"q": "What caused the 2020-2023 global chip shortage?", "a": "COVID-19 disruptions combined with a 13% increase in global PC demand due to stay-at-home economy. Taiwan's worst drought in 50 years hit TSMC (which used 63,000+ tons of water daily). A 2022 fire at ASML's Berlin plant affected EUV lithography equipment. Neon prices rose sixfold between Dec 2021 and March 2022 due to pandemic and Ukraine war."},
        {"q": "How many industries were affected by the chip shortage?", "a": "The shortage affected more than 169 industries, causing major price increases, long queues, and reselling for automobiles, graphics cards, video game consoles, computers, household appliances, and other consumer electronics."},
        {"q": "What investments did TSMC make in response?", "a": "In April 2021, TSMC announced $100 billion over three years to increase capacity. They announced a $12 billion Arizona fab in 2020 (later tripled to $40 billion), with first production pushed to 2025, second factory by 2027-2028. In November 2021, TSMC partnered with Sony for a $7 billion Japan fab."},
        {"q": "What were Intel's expansion plans?", "a": "Intel announced plans to build two fabs in Arizona and upgrade one in New Mexico (totaling $23.5 billion), with operations starting 2023. Intel broke ground on the Arizona plant in September 2021 and committed $20 billion to two new US factories."},
        {"q": "When did the chip shortage end?", "a": "The global supply stabilized near the end of 2023 after three difficult years. By July 2023, manufacturers had ramped up production and customers adjusted to predictable supply. The industry shifted from shortage in 2021-2022 to capacity underutilization in some segments. TSMC estimated semiconductor market decline of 4% for 2023 before recovery."},
    ]
    for f in chip_facts:
        examples.append({"instruction": f["q"], "input": "", "output": f["a"]})

    # ==========================================================================
    # MIGRATION & REFUGEES
    # ==========================================================================
    migration_facts = [
        {"q": "How many people are forcibly displaced globally as of 2024?", "a": "At the end of 2024, an estimated 123.2 million people worldwide were forcibly displaced—an increase of 7 million (6%) from 2023's 117.3 million. This continues 12 consecutive years of increases. By April 2025, UNHCR estimates a slight 1% decrease to 122.1 million—the first decrease in over a decade."},
        {"q": "What is the world's largest displacement crisis?", "a": "The war in Sudan is the world's largest displacement crisis. At end of 2024, 14.3 million Sudanese people remained displaced—3.5 million more than 12 months prior, representing nearly one in three of the national population. Sudan also had 11.6 million internally displaced—the largest internal displacement ever recorded."},
        {"q": "Which countries produce the most refugees?", "a": "73% of refugees under UNHCR's mandate come from just five countries: Afghanistan (6.4 million), Syria (6.4 million), Venezuela (6.1 million), Ukraine (6.0 million), and Sudan (1.5 million)."},
        {"q": "Which countries host the most refugees?", "a": "Top host countries: Iran (3.8 million), Turkey (3.3 million), Colombia (2.9 million), Germany (2.6 million), and Pakistan (2 million). Despite global displacement nearly doubling over the past decade, the rate of increase slowed in second half 2024."},
        {"q": "What positive developments occurred in 2024?", "a": "Solutions for refugees and IDPs all increased in 2024. Refugee returns reached the highest in more than two decades (1.6 million). There were 42.7 million refugees globally at end of 2024—a 1% decrease from 2023. However, humanitarian programmes remain substantially underfunded with UNHCR projecting $8.5 billion shortfall."},
    ]
    for f in migration_facts:
        examples.append({"instruction": f["q"], "input": "", "output": f["a"]})

    # ==========================================================================
    # AI INDUSTRY & FUNDING
    # ==========================================================================
    ai_facts = [
        {"q": "How much funding went to AI companies in 2024?", "a": "Close to a third of all global venture funding in 2024 went to AI-related companies—over $100 billion, up 80% YoY from $55.6 billion in 2023. Generative AI funding specifically reached a new record, up 92% from 2023's $29.1 billion across 691 deals."},
        {"q": "What is Microsoft's investment in OpenAI?", "a": "Microsoft invested $10 billion in OpenAI in January 2023, adding to $1 billion in 2019 and $2 billion quietly in 2021. Microsoft has exclusive license on GPT-4 and all other OpenAI models. OpenAI was valued at $157 billion in 2024 after raising a $6.6 billion round in Q4."},
        {"q": "What is Anthropic's funding and valuation?", "a": "Anthropic raised $300 million from Google (10% stake) in February 2023, then $450 million in May 2023—the largest AI round that year after Microsoft/OpenAI. Google added up to $2 billion in October 2023. Amazon invested $4 billion in Q4 2024. Anthropic reached $18.4 billion valuation in 2024."},
        {"q": "Who founded Anthropic and why?", "a": "Anthropic was founded in 2021 by Dario Amodei (OpenAI's former VP of Research) and his sister Daniela Amodei (OpenAI's VP of Safety and Policy), along with other OpenAI research alumni. They left due to differences in OpenAI's approach to AI safety and ethics."},
        {"q": "What were the biggest AI deals in 2024?", "a": "Databricks was valued at $62 billion in the year's largest venture deal—a $10 billion round. xAI doubled its valuation to $50 billion in six months. Inflection AI raised $1.3 billion led by Microsoft, Reid Hoffman, Bill Gates, Eric Schmidt, and NVIDIA. Overall startup funding reached $314 billion globally, up 3% from 2023."},
    ]
    for f in ai_facts:
        examples.append({"instruction": f["q"], "input": "", "output": f["a"]})

    # ==========================================================================
    # NUCLEAR & SMR DEVELOPMENTS
    # ==========================================================================
    nuclear_facts = [
        {"q": "What was the first SMR design certified by the US NRC?", "a": "NuScale Power's small modular reactor became the first SMR design certified by the NRC and just the seventh reactor design ever cleared for use in the United States. The rule took effect February 21, 2023, allowing utilities to reference NuScale's design when applying for combined licenses."},
        {"q": "What operational SMRs exist worldwide?", "a": "Only two SMR plants currently operate: Russia's Akademik Lomonosov (a floating power plant with two 35 MWe reactors supplying Pevek since 2020), and China's HTR-PM (200 MWe from two high-temperature gas-cooled reactors, commercial operation December 2023). China's was the world's first commercial 210 MW SMR."},
        {"q": "What is the status of SMR construction in the West?", "a": "As of January 2024, no commercial SMRs are under construction in Western nations. About 22 GW of SMR projects are underway globally (+65% since 2021), but none in advanced development. The US leads in announced SMR projects with nearly 4 GW. TerraPower's Natrium broke ground in June 2024 in Wyoming, targeting 2030 operation."},
        {"q": "What is the tech industry investing in nuclear?", "a": "In 2024, Amazon committed over $500 million to SMR development with Energy Northwest for up to four SMR units (960 MW) in Washington state, and invested in X-energy. Google partnered with Kairos Power for 500 MW of SMR clean power by 2030 for data centers."},
        {"q": "What government funding supports SMRs?", "a": "The US DOE has spent over $1.2 billion on SMR and announced additional funding up to $5.5 billion for the next decade. DOE provided more than $600 million since 2014 to support NuScale's VOYGR SMR plant design, licensing, and siting."},
    ]
    for f in nuclear_facts:
        examples.append({"instruction": f["q"], "input": "", "output": f["a"]})

    # ==========================================================================
    # CYBERSECURITY INCIDENTS
    # ==========================================================================
    cyber_facts = [
        {"q": "What is the scale of ransomware attacks globally?", "a": "Ransomware affected 59% of organizations in 2024 (Sophos). Globally, 72.7% of all organizations fell prey to ransomware in 2023. Over 5,263 attacks were recorded in 2024—the highest ever since NCC began tracking in 2021."},
        {"q": "How much do ransomware attackers receive in payments?", "a": "Ransomware actors received $1.1 billion in 2023—a 140% increase from $457 million in 2022. The average ransom payment rose from $400,000 in 2023 to $2 million in 2024—a 500% increase. The FBI reported $12.3 billion lost to 2023 cyberattack incidents in the US."},
        {"q": "What was the largest data breach in 2024?", "a": "The National Public Data breach was the second largest in history, stealing 2.9 billion records from people in the US, UK, and Canada—including full names, addresses, Social Security numbers, dates of birth, and phone numbers. The Change Healthcare attack was the biggest healthcare data breach to date."},
        {"q": "What was notable about the MOVEit attacks?", "a": "The most noteworthy ransomware incident of 2023 was the Clop ransomware group's barrage of MOVEit Transfer attacks, hitting multiple US government agencies, BBC, British Airways, HR provider Zellis, and Nova Scotia's government. Analysts estimated over 600 breaches from MOVEit alone."},
        {"q": "What industries are most targeted by ransomware?", "a": "Healthcare faced the highest ransomware rate in 2024 (Forrester), with 630+ incidents worldwide in 2023. Healthcare breach costs reached $10.93 million per incident—double the financial sector. Manufacturing saw 638 attacks in 2023 (most targeted). Government/civic sectors saw a 229% increase to 293 verified victims in 2024."},
        {"q": "What is the recovery rate from ransomware?", "a": "Only 35% of organizations fully recovered from ransomware within one week in 2024, down from 47% in 2023. While 97% of victims who paid in 2024 regained access to data, only 59% recovered all data, highlighting unreliable decryptors."},
    ]
    for f in cyber_facts:
        examples.append({"instruction": f["q"], "input": "", "output": f["a"]})

    return examples


def main():
    print("=" * 70)
    print("FACTUAL CONTEXT TRAINING DATA GENERATOR")
    print("=" * 70)

    examples = generate_examples()
    print(f"\nGenerated {len(examples)} factual examples with real data")

    # Categorize
    categories = {}
    for ex in examples:
        text = ex["instruction"].lower()
        if "spacex" in text or "starlink" in text or "starship" in text or "falcon" in text or "launch" in text:
            cat = "space"
        elif "gene" in text or "crispr" in text or "fda" in text or "therapy" in text or "drug" in text:
            cat = "biotech"
        elif "grain" in text or "wheat" in text or "ukraine" in text or "food" in text:
            cat = "food_crisis"
        elif "rare earth" in text or "lithium" in text or "china" in text and "export" in text:
            cat = "minerals"
        elif "amoc" in text or "arctic" in text or "amazon" in text or "tipping" in text or "climate" in text:
            cat = "climate"
        elif "chip" in text or "semiconductor" in text or "tsmc" in text or "intel" in text:
            cat = "chips"
        elif "refugee" in text or "displaced" in text or "migration" in text or "unhcr" in text:
            cat = "migration"
        elif "ai" in text or "openai" in text or "anthropic" in text or "funding" in text:
            cat = "ai_industry"
        elif "nuclear" in text or "smr" in text or "reactor" in text:
            cat = "nuclear"
        elif "ransomware" in text or "cyber" in text or "breach" in text:
            cat = "cybersecurity"
        else:
            cat = "other"
        categories[cat] = categories.get(cat, 0) + 1

    print("\n" + "=" * 70)
    print("CATEGORY BREAKDOWN:")
    print("=" * 70)
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} examples")

    output_path = "factual_context_training.json"
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"\nSaved to {output_path}")

    return examples


if __name__ == "__main__":
    main()
