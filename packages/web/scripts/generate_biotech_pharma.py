#!/usr/bin/env python3
"""Biotech/pharma pipeline, drug development, and healthcare supply chain training data."""
import json

def generate_examples():
    examples = []

    # DRUG DEVELOPMENT PIPELINE
    pipeline = [
        "What are the stages of drug development from discovery to approval?",
        "How long does typical drug development take and what affects timelines?",
        "What is the attrition rate at each phase of clinical trials?",
        "How do Phase 1, 2, and 3 trials differ in objectives and design?",
        "What determines whether a drug candidate advances to the next phase?",
        "How do breakthrough therapy and fast track designations affect development?",
        "What is the role of preclinical studies in drug development?",
        "How do biomarkers affect drug development strategies?",
        "What is adaptive trial design and when is it used?",
        "How do companion diagnostics affect drug development?",
        "What is the cost breakdown of drug development by phase?",
        "How do platform technologies affect drug development efficiency?",
        "What role do CROs play in drug development?",
        "How has COVID affected drug development timelines and approaches?",
        "What is the role of real-world evidence in drug development?",
    ]
    for q in pipeline:
        examples.append({"instruction": q, "input": "", "output": "[Drug development analysis: phase requirements, success probabilities, timeline factors, and regulatory pathway considerations.]"})

    # THERAPEUTIC AREAS
    therapeutics = [
        "What is the current state of oncology drug development?",
        "How is immunotherapy transforming cancer treatment pipelines?",
        "What are the challenges in CNS drug development?",
        "How is gene therapy pipeline evolving?",
        "What is the state of mRNA technology beyond vaccines?",
        "How are cell therapies like CAR-T progressing?",
        "What challenges exist in Alzheimer's drug development?",
        "How is the obesity drug market evolving?",
        "What is the state of antibiotic development and market failure?",
        "How are rare disease drugs developed and priced?",
        "What is the pipeline for autoimmune disease treatments?",
        "How is the diabetes treatment landscape changing?",
        "What are the prospects for aging-related drug development?",
        "How is the cardiovascular drug pipeline evolving?",
        "What is the state of infectious disease drug development?",
    ]
    for q in therapeutics:
        examples.append({"instruction": q, "input": "", "output": "[Therapeutic area analysis: pipeline status, scientific challenges, commercial dynamics, and development trends.]"})

    # REGULATORY LANDSCAPE
    regulatory = [
        "How do FDA and EMA approval processes differ?",
        "What is the 505(b)(2) pathway and when is it used?",
        "How do accelerated approval pathways work?",
        "What is the biosimilar approval process?",
        "How do orphan drug designations affect development incentives?",
        "What is the REMS program and when is it required?",
        "How do post-marketing requirements affect drug companies?",
        "What is the role of advisory committees in drug approval?",
        "How do complete response letters affect company strategy?",
        "What is the FDA's approach to digital therapeutics?",
        "How do pediatric exclusivity requirements affect development?",
        "What is the regulatory pathway for combination products?",
        "How is China's drug regulatory framework evolving?",
        "What are the implications of FDA user fee negotiations?",
        "How do manufacturing inspections affect drug approval timelines?",
    ]
    for q in regulatory:
        examples.append({"instruction": q, "input": "", "output": "[Regulatory analysis: approval pathways, designation benefits, post-marketing requirements, and cross-jurisdiction considerations.]"})

    # PHARMA SUPPLY CHAIN
    supply = [
        "What are the key vulnerabilities in pharmaceutical supply chains?",
        "How concentrated is API manufacturing geographically?",
        "What role does China play in pharmaceutical supply chains?",
        "How do drug shortages occur and propagate?",
        "What is the state of pharmaceutical manufacturing in the US?",
        "How do cold chain requirements affect vaccine distribution?",
        "What are the challenges in scaling up new modality manufacturing?",
        "How does contract manufacturing affect supply chain resilience?",
        "What inventory strategies do pharmaceutical companies use?",
        "How do quality issues cascade through pharmaceutical supply chains?",
        "What is the role of APIs vs. finished dosage form manufacturing?",
        "How do trade policies affect pharmaceutical supply chains?",
        "What are the supply chain considerations for cell and gene therapies?",
        "How do environmental regulations affect pharmaceutical manufacturing?",
        "What is the state of pharmaceutical serialization and track-and-trace?",
    ]
    for q in supply:
        examples.append({"instruction": q, "input": "", "output": "[Pharma supply chain analysis: geographic concentration, vulnerability points, shortage dynamics, and resilience strategies.]"})

    # BIOTECH BUSINESS MODELS
    business = [
        "How do biotech funding rounds and valuations work?",
        "What is the role of venture capital in biotech?",
        "How do licensing deals structure risk and reward?",
        "What triggers M&A activity in biotech?",
        "How do platform companies differ from pipeline companies?",
        "What is the IPO window dynamic in biotech?",
        "How do biotech valuations respond to clinical trial results?",
        "What is the role of partnerships in biotech development?",
        "How do spinouts from pharma companies work?",
        "What determines biotech cash runway requirements?",
        "How do royalty monetization deals work?",
        "What is the role of SPACs in biotech?",
        "How do biotech companies structure option deals?",
        "What are the economics of biotech platform licensing?",
        "How do activist investors affect biotech strategy?",
    ]
    for q in business:
        examples.append({"instruction": q, "input": "", "output": "[Biotech business analysis: funding dynamics, deal structures, valuation drivers, and strategic partnership considerations.]"})

    # IP AND EXCLUSIVITY
    ip = [
        "How do pharmaceutical patents protect market exclusivity?",
        "What is the Hatch-Waxman framework for generics?",
        "How do patent thickets and evergreening work?",
        "What triggers patent challenges from generic companies?",
        "How do biologics exclusivity periods differ from small molecules?",
        "What is the role of trade secrets in pharma IP strategy?",
        "How do pay-for-delay settlements affect competition?",
        "What is the IPR challenge process for pharma patents?",
        "How do freedom-to-operate analyses work in biotech?",
        "What is the state of CRISPR IP landscape?",
        "How do data exclusivity regulations protect originator drugs?",
        "What are the implications of patent term extensions?",
        "How do compulsory licensing provisions affect IP strategy?",
        "What role do patents play in mRNA technology access?",
        "How do international patent strategies differ by region?",
    ]
    for q in ip:
        examples.append({"instruction": q, "input": "", "output": "[Pharma IP analysis: exclusivity mechanisms, patent challenges, litigation dynamics, and international protection strategies.]"})

    # PRICING AND ACCESS
    pricing = [
        "How are drug prices determined in the US vs. other countries?",
        "What is the role of PBMs in drug pricing?",
        "How do reference pricing systems work internationally?",
        "What are the implications of the Inflation Reduction Act for pharma?",
        "How do value-based contracts work for drugs?",
        "What role does ICER play in drug pricing debates?",
        "How do 340B discounts affect pharmaceutical economics?",
        "What is the role of specialty pharmacies in drug distribution?",
        "How do rebates and discounts flow through the drug channel?",
        "What are the economics of gene therapy pricing?",
        "How does outcomes-based pricing work for high-cost drugs?",
        "What is the role of patient assistance programs?",
        "How do international price comparisons affect US policy?",
        "What are the implications of reference product selection?",
        "How do biosimilar pricing dynamics differ from generics?",
    ]
    for q in pricing:
        examples.append({"instruction": q, "input": "", "output": "[Drug pricing analysis: pricing mechanisms, channel economics, policy implications, and access considerations by market.]"})

    # EMERGING TECHNOLOGIES
    emerging = [
        "How is AI affecting drug discovery and development?",
        "What is the state of protein structure prediction in drug design?",
        "How are organoids and organs-on-chips changing preclinical testing?",
        "What is the potential of targeted protein degradation?",
        "How is base editing different from CRISPR-Cas9?",
        "What are the prospects for in vivo gene editing?",
        "How is single-cell sequencing affecting drug development?",
        "What is the state of drug delivery technology innovation?",
        "How are digital biomarkers affecting clinical trials?",
        "What is the potential of epigenetic therapies?",
        "How is microbiome-based therapy development progressing?",
        "What are the prospects for radiopharmaceuticals?",
        "How is liquid biopsy technology affecting oncology development?",
        "What is the state of RNAi therapeutics after early setbacks?",
        "How are antibody-drug conjugates evolving?",
    ]
    for q in emerging:
        examples.append({"instruction": q, "input": "", "output": "[Emerging technology analysis: scientific basis, development status, commercial potential, and timeline considerations.]"})

    return examples

if __name__ == "__main__":
    examples = generate_examples()
    print(f"Generated {len(examples)} biotech/pharma examples")
    with open("biotech_pharma_training.json", "w") as f:
        json.dump(examples, f, indent=2)
    print("Saved to biotech_pharma_training.json")
