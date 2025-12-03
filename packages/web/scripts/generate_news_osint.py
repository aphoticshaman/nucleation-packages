#!/usr/bin/env python3
"""News analysis, OSINT, and source evaluation training data."""
import json

def generate_examples():
    examples = []

    # SOURCE RELIABILITY
    reliability = [
        "How do you assess the reliability of a news source?",
        "What distinguishes primary from secondary sources?",
        "How do you evaluate source access and expertise?",
        "What is the difference between reliability and credibility?",
        "How do you assess source track record?",
        "What indicators suggest a source may be compromised?",
        "How do you handle anonymous sources in analysis?",
        "What is source triangulation and how is it done?",
        "How do you assess the independence of sources?",
        "What role does source motivation play in evaluation?",
        "How do you weight conflicting sources?",
        "What is circular reporting and how do you detect it?",
        "How do you assess technical source capabilities?",
        "What is the role of access in source quality?",
        "How do you build and maintain source evaluation criteria?",
    ]
    for q in reliability:
        examples.append({"instruction": q, "input": "", "output": "[Source reliability analysis: access assessment, track record evaluation, independence verification, and triangulation methods.]"})

    # BIAS DETECTION
    bias = [
        "How do you detect political bias in news coverage?",
        "What is selection bias in media and how do you identify it?",
        "How do you recognize framing effects in reporting?",
        "What is omission bias and how do you detect it?",
        "How do you identify loaded language in articles?",
        "What is false balance and when does it occur?",
        "How do you assess ideological lean of publications?",
        "What is confirmation bias in news consumption?",
        "How do you detect sponsored content masquerading as news?",
        "What role does ownership play in media bias?",
        "How do you identify editorial vs. news content?",
        "What is headline bias and how do you account for it?",
        "How do you detect sensationalism in reporting?",
        "What is access journalism and how does it affect coverage?",
        "How do you build a balanced information diet?",
    ]
    for q in bias:
        examples.append({"instruction": q, "input": "", "output": "[Bias detection: political lean identification, framing analysis, omission detection, and balanced information consumption strategies.]"})

    # PROPAGANDA & DISINFORMATION
    disinfo = [
        "How do you identify state-sponsored disinformation?",
        "What are the characteristics of propaganda?",
        "How do you detect coordinated inauthentic behavior?",
        "What is astroturfing and how do you recognize it?",
        "How do you identify bot networks on social media?",
        "What is a disinformation campaign's typical structure?",
        "How do you track narrative laundering?",
        "What role do amplification networks play?",
        "How do you identify fabricated content?",
        "What is the firehose of falsehood technique?",
        "How do you detect deepfakes and manipulated media?",
        "What are the indicators of foreign influence operations?",
        "How do you assess the intent behind disinformation?",
        "What role do useful idiots play in disinformation?",
        "How do you counter disinformation effectively?",
    ]
    for q in disinfo:
        examples.append({"instruction": q, "input": "", "output": "[Disinformation analysis: campaign identification, network detection, content verification, and counter-messaging strategies.]"})

    # SOCIAL MEDIA INTELLIGENCE
    socmint = [
        "How do you extract signal from social media noise?",
        "What metadata is valuable in social media analysis?",
        "How do you verify user-generated content?",
        "What is geolocation analysis of social media?",
        "How do you track trending narratives?",
        "What is sentiment analysis and how reliable is it?",
        "How do you identify influencers in topic networks?",
        "What is network analysis in social media intelligence?",
        "How do you detect breaking events from social media?",
        "What is the role of image analysis in SOCMINT?",
        "How do you handle the volume of social media data?",
        "What are the ethics of social media intelligence?",
        "How do you track narrative evolution over time?",
        "What is community detection in social networks?",
        "How do you assess the representativeness of social media data?",
    ]
    for q in socmint:
        examples.append({"instruction": q, "input": "", "output": "[Social media intelligence: signal extraction, content verification, network analysis, and narrative tracking methods.]"})

    # OSINT TECHNIQUES
    osint = [
        "What is OSINT and what are its primary sources?",
        "How do you conduct effective open source research?",
        "What is reverse image search and how is it used?",
        "How do you use satellite imagery for analysis?",
        "What is domain and IP research?",
        "How do you research corporate structures and ownership?",
        "What is document metadata analysis?",
        "How do you use web archives for research?",
        "What is flight tracking and how is it used in OSINT?",
        "How do you research vessel movements?",
        "What is radio frequency OSINT?",
        "How do you use public records in research?",
        "What is facial recognition in OSINT?",
        "How do you maintain operational security in OSINT?",
        "What are the legal and ethical limits of OSINT?",
    ]
    for q in osint:
        examples.append({"instruction": q, "input": "", "output": "[OSINT techniques: source utilization, verification methods, technical tools, and operational security considerations.]"})

    # NARRATIVE TRACKING
    narrative = [
        "How do you identify emerging narratives?",
        "What is narrative velocity and how do you measure it?",
        "How do you track narrative spread across platforms?",
        "What is the lifecycle of a narrative?",
        "How do you identify narrative entrepreneurs?",
        "What role do echo chambers play in narrative spread?",
        "How do you detect narrative pivots?",
        "What is counter-narrative analysis?",
        "How do you measure narrative resonance?",
        "What is the role of events in narrative evolution?",
        "How do you identify narrative frames?",
        "What is the relationship between narratives and behavior?",
        "How do you forecast narrative trajectory?",
        "What makes narratives sticky or viral?",
        "How do you analyze competing narratives?",
    ]
    for q in narrative:
        examples.append({"instruction": q, "input": "", "output": "[Narrative analysis: emergence detection, velocity tracking, spread patterns, and resonance measurement.]"})

    # VERIFICATION METHODS
    verify = [
        "How do you verify breaking news claims?",
        "What is the CRAAP test for source evaluation?",
        "How do you verify the date of content?",
        "What is video verification and how is it done?",
        "How do you verify eyewitness accounts?",
        "What is chain of custody in content verification?",
        "How do you identify manipulated images?",
        "What is chronolocation in verification?",
        "How do you verify document authenticity?",
        "What is shadow analysis for verification?",
        "How do you verify claims about protests or crowds?",
        "What is the role of metadata in verification?",
        "How do you verify during information vacuums?",
        "What is provenance analysis?",
        "How do you handle unverifiable but important claims?",
    ]
    for q in verify:
        examples.append({"instruction": q, "input": "", "output": "[Verification methods: content analysis, metadata examination, chronolocation, and authenticity assessment techniques.]"})

    # INFORMATION TRIAGE
    triage = [
        "How do you prioritize information during crises?",
        "What is the difference between urgent and important information?",
        "How do you manage information overload?",
        "What is the minimum viable information for a decision?",
        "How do you identify actionable intelligence?",
        "What is the signal-to-noise ratio in news monitoring?",
        "How do you filter for relevance?",
        "What is the role of alerts in information management?",
        "How do you avoid alert fatigue?",
        "What is the optimal information monitoring cadence?",
        "How do you handle information gaps?",
        "What is the role of synthesis in triage?",
        "How do you escalate critical information?",
        "What is the 80/20 rule in information gathering?",
        "How do you know when you have enough information?",
    ]
    for q in triage:
        examples.append({"instruction": q, "input": "", "output": "[Information triage: prioritization methods, overload management, relevance filtering, and escalation protocols.]"})

    return examples

if __name__ == "__main__":
    examples = generate_examples()
    print(f"Generated {len(examples)} news/OSINT examples")
    with open("news_osint_training.json", "w") as f:
        json.dump(examples, f, indent=2)
    print("Saved to news_osint_training.json")
