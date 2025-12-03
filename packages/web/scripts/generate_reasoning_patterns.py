#!/usr/bin/env python3
"""Reasoning patterns, analytical methods, and cognitive frameworks training data."""
import json

def generate_examples():
    examples = []

    # COUNTERFACTUAL ANALYSIS
    counterfactual = [
        "How do you construct useful counterfactual scenarios?",
        "What would have happened if the Soviet Union hadn't collapsed?",
        "How do counterfactuals help identify causal mechanisms?",
        "What makes a counterfactual plausible vs. implausible?",
        "How do you avoid hindsight bias in counterfactual analysis?",
        "What role do counterfactuals play in policy evaluation?",
        "How do you identify the critical decision points in history?",
        "What would have happened if different technology emerged first?",
        "How do counterfactuals help stress test assumptions?",
        "What is the nearest possible world in counterfactual reasoning?",
        "How do you evaluate the probability of counterfactual outcomes?",
        "What role do structural factors play in counterfactual analysis?",
        "How do counterfactuals inform contingency planning?",
        "What are the limits of counterfactual reasoning?",
        "How do you use counterfactuals for scenario planning?",
    ]
    for q in counterfactual:
        examples.append({"instruction": q, "input": "", "output": "[Counterfactual analysis: scenario construction, causal mechanism identification, plausibility assessment, and policy implications.]"})

    # RED TEAM / BLUE TEAM
    redteam = [
        "What is red team analysis and when should it be used?",
        "How do you structure an effective red team exercise?",
        "What cognitive biases does red teaming help overcome?",
        "How do you think like an adversary?",
        "What is devil's advocacy and how is it different from red teaming?",
        "How do you challenge consensus views effectively?",
        "What makes a good red team member?",
        "How do you red team your own analysis?",
        "What is Team A/Team B analysis?",
        "How do you integrate red team findings into decisions?",
        "What is pre-mortem analysis?",
        "How do you anticipate adversary responses to your actions?",
        "What is competitive hypothesis testing?",
        "How do you avoid groupthink in analytical teams?",
        "What is the murder board technique?",
    ]
    for q in redteam:
        examples.append({"instruction": q, "input": "", "output": "[Red team analysis: adversarial thinking, bias mitigation, structured challenge, and decision improvement techniques.]"})

    # GAME THEORY
    gametheory = [
        "How does game theory apply to geopolitical analysis?",
        "What is a Nash equilibrium and why does it matter?",
        "How do you model adversarial decision-making?",
        "What is the prisoner's dilemma and what does it teach?",
        "How do repeated games differ from one-shot games?",
        "What is the role of credible commitments in strategy?",
        "How do you analyze deterrence using game theory?",
        "What is a coordination game and when does it apply?",
        "How do you model incomplete information in games?",
        "What is signaling and how does it affect strategic interaction?",
        "How do you analyze chicken games in international relations?",
        "What is the security dilemma in game theoretic terms?",
        "How do you model escalation dynamics?",
        "What is the role of reputation in repeated games?",
        "How do you identify dominant strategies?",
    ]
    for q in gametheory:
        examples.append({"instruction": q, "input": "", "output": "[Game theory analysis: strategic interaction modeling, equilibrium concepts, signaling, and application to adversarial situations.]"})

    # COGNITIVE BIASES
    biases = [
        "What is confirmation bias and how do you mitigate it?",
        "How does anchoring affect analytical judgment?",
        "What is availability bias and how does it distort risk perception?",
        "How do you recognize and counter groupthink?",
        "What is the hindsight bias and why is it problematic?",
        "How does framing affect decision-making?",
        "What is the sunk cost fallacy in strategic decisions?",
        "How do you mitigate overconfidence in estimates?",
        "What is the conjunction fallacy?",
        "How does the affect heuristic influence judgment?",
        "What is base rate neglect and how do you avoid it?",
        "How do you recognize motivated reasoning?",
        "What is the planning fallacy and how do you counter it?",
        "How do you structure analysis to reduce bias?",
        "What is the curse of knowledge in communication?",
    ]
    for q in biases:
        examples.append({"instruction": q, "input": "", "output": "[Cognitive bias analysis: recognition, impact on judgment, structured mitigation techniques, and debiasing strategies.]"})

    # DECISION UNDER UNCERTAINTY
    decision = [
        "How do you make decisions with incomplete information?",
        "What is expected value and when is it appropriate?",
        "How do you handle decisions with fat-tailed outcomes?",
        "What is the difference between risk and uncertainty?",
        "How do you value optionality in decisions?",
        "What is regret minimization as a decision framework?",
        "How do you handle irreversible decisions?",
        "What is satisficing vs. optimizing?",
        "How do you structure decisions with multiple objectives?",
        "What is the value of information in decision-making?",
        "How do you handle decisions with unknown probabilities?",
        "What is robust decision-making?",
        "How do you prioritize decisions under time pressure?",
        "What is the difference between decisions and outcomes?",
        "How do you evaluate decisions after the fact?",
    ]
    for q in decision:
        examples.append({"instruction": q, "input": "", "output": "[Decision analysis: uncertainty handling, multi-objective evaluation, robustness, and decision quality assessment.]"})

    # SCENARIO PLANNING
    scenario = [
        "What is scenario planning and how does it differ from forecasting?",
        "How do you identify critical uncertainties for scenarios?",
        "What makes a good scenario set?",
        "How do you develop internally consistent scenarios?",
        "What is the 2x2 matrix approach to scenario planning?",
        "How do you use scenarios for strategy development?",
        "What is the difference between scenarios and projections?",
        "How do you identify early warning indicators for scenarios?",
        "What is wind-tunneling in scenario planning?",
        "How do you avoid scenario planning becoming wishful thinking?",
        "What is the role of wild cards in scenario planning?",
        "How do you update scenarios as events unfold?",
        "What is the Shell scenario planning methodology?",
        "How do you communicate scenarios to stakeholders?",
        "What is the role of narratives in scenario planning?",
    ]
    for q in scenario:
        examples.append({"instruction": q, "input": "", "output": "[Scenario planning: uncertainty identification, scenario development, strategy testing, and early warning systems.]"})

    # STRUCTURED ANALYTIC TECHNIQUES
    sat = [
        "What is Analysis of Competing Hypotheses (ACH)?",
        "How do you use a key assumptions check?",
        "What is a weighted ranking matrix?",
        "How do you conduct a SWOT analysis effectively?",
        "What is the Delphi method for expert judgment?",
        "How do you use argument mapping?",
        "What is a causal loop diagram?",
        "How do you conduct a stakeholder analysis?",
        "What is a morphological analysis?",
        "How do you use a decision tree for analysis?",
        "What is a quality of information check?",
        "How do you conduct an indicators analysis?",
        "What is a cone of plausibility?",
        "How do you use timeline analysis?",
        "What is network analysis in intelligence?",
    ]
    for q in sat:
        examples.append({"instruction": q, "input": "", "output": "[Structured analytic techniques: methodology, application, output interpretation, and integration into analysis.]"})

    # SYSTEMS THINKING
    systems = [
        "What is systems thinking and why does it matter?",
        "How do you identify feedback loops in complex systems?",
        "What is emergence and how do you analyze it?",
        "How do you map system boundaries?",
        "What is the difference between complicated and complex systems?",
        "How do you identify leverage points in a system?",
        "What are system archetypes and how do you use them?",
        "How do you model delays in system dynamics?",
        "What is requisite variety in systems?",
        "How do you handle nonlinear dynamics in analysis?",
        "What is the role of stocks and flows in system models?",
        "How do you identify unintended consequences?",
        "What is path dependence in complex systems?",
        "How do you analyze system resilience?",
        "What is the difference between efficiency and resilience?",
    ]
    for q in systems:
        examples.append({"instruction": q, "input": "", "output": "[Systems thinking: feedback loops, emergence, leverage points, dynamics modeling, and unintended consequence analysis.]"})

    return examples

if __name__ == "__main__":
    examples = generate_examples()
    print(f"Generated {len(examples)} reasoning pattern examples")
    with open("reasoning_patterns_training.json", "w") as f:
        json.dump(examples, f, indent=2)
    print("Saved to reasoning_patterns_training.json")
