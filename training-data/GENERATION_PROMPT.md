# LatticeForge Ultra/Enterprise Training Data Generation Prompt

## Usage
Copy this entire prompt and paste it into ChatGPT, Grok, DeepSeek, Gemini, or Copilot. Replace `[CATEGORY]` with your target category from the list below.

---

# BEGIN PROMPT

You are a senior training data engineer at LatticeForge, a geopolitical intelligence platform used by institutional investors, government analysts, and strategic consultants. Your task is to generate **exactly 25 training examples** in Alpaca format for fine-tuning language models on **[CATEGORY]** analysis.

## Quality Standards (S-Tier)

These examples must meet LatticeForge Ultra/Enterprise standards:

1. **Expert-Level Depth**: Each response must demonstrate post-doctoral level expertise. Assume the reader holds advanced degrees and works in professional intelligence, finance, or policy roles. No simplification, no hedging, no "for beginners" framing.

2. **Analytical Frameworks**: Responses must employ recognized analytical methodologies:
   - Structured analytic techniques (key assumptions check, devil's advocacy, Analysis of Competing Hypotheses)
   - First-principles reasoning with explicit assumptions
   - Multi-order effects analysis (2nd and 3rd order consequences)
   - Bayesian updating with probability ranges where appropriate
   - Game-theoretic reasoning for multi-actor scenarios

3. **Historical Grounding**: Every conceptual point must be anchored in concrete historical precedent. Draw from:
   - Ancient history (Mesopotamia, Egypt, Greece, Rome, Han China, Maurya India)
   - Medieval/Classical periods (Byzantine, Caliphates, Mongol Empire, Medieval Europe)
   - Early modern (Ming/Qing, Ottoman, European powers, colonial empires)
   - Modern era (World Wars, Cold War, decolonization)
   - Contemporary (1991-present)

4. **Contentious Topics Policy**: Do not sanitize history. Include:
   - Religious drivers of conflict (Crusades, Islamic conquests, Buddhist-Hindu violence, Reformation wars)
   - War as a driver of technological and social advancement (this is historically accurate and analytically relevant)
   - Moral complexity of historical figures (Churchill, Mao, Napoleon, etc.)
   - Lessons from religious leaders (Jesus, Buddha, Muhammad, Confucius) as historical change agents regardless of belief
   - Uncomfortable truths about human nature, tribalism, and violence

5. **Investment-Relevant Framing**: Where applicable, connect analysis to:
   - Asset price implications
   - Risk premia adjustments
   - Scenario probability assessments
   - Timeline estimates
   - Actionable indicators to monitor

## Output Format

Return exactly 25 JSON objects in this format:

```json
[
  {
    "instruction": "Clear, specific analytical question or task",
    "input": "Optional context, scenario, or data point (leave empty string if not needed)",
    "output": "Comprehensive expert analysis of 300-600 words, structured with clear sections, bullet points where appropriate, and concrete conclusions"
  }
]
```

## Category-Specific Guidelines

### If [CATEGORY] is POLITICAL:
Cover electoral dynamics, regime types, succession risks, coalition politics, state capture, democratization/autocratization, political violence, populism, constitutional crises. Include examples from every inhabited continent and multiple time periods.

### If [CATEGORY] is ECONOMIC & TRADE:
Cover trade policy, sanctions regimes, currency dynamics, fiscal/monetary interaction, development economics, industrial policy, supply chain restructuring, commodity dependencies. Reference historical trade systems (Silk Road, Hanseatic League, British Empire trade networks) alongside contemporary examples.

### If [CATEGORY] is SECURITY & DIPLOMACY:
Cover alliance dynamics, deterrence theory, arms control, military-to-military relations, intelligence sharing, diplomatic protocols, great power competition, regional security architectures. Draw from historical diplomacy (Congress of Vienna, Treaty of Westphalia) through contemporary.

### If [CATEGORY] is FINANCIAL:
Cover sovereign risk, corporate credit, equity valuation under geopolitical stress, derivatives and hedging, currency crises, central bank policy, financial contagion, sanctions compliance. Include historical financial crises and their geopolitical triggers.

### If [CATEGORY] is HEALTH:
Cover pandemic preparedness, health system resilience, pharmaceutical supply chains, bioweapons risks, demographic health transitions, health diplomacy, vaccine geopolitics. Include historical pandemics (Plague of Justinian, Black Death, Spanish Flu, COVID) and their societal impacts.

### If [CATEGORY] is SCIENCE & TECHNOLOGY:
Cover dual-use technology, semiconductor geopolitics, AI governance, space competition, quantum computing, biotechnology risks, technology transfer controls. Connect to historical technology races (atomic bomb, space race, industrial revolution).

### If [CATEGORY] is NATURAL RESOURCES:
Cover water scarcity, land degradation, fisheries disputes, forest management, biodiversity loss geopolitics. Include historical resource conflicts (water wars, deforestation-driven collapses like Easter Island).

### If [CATEGORY] is CRIME & DRUGS:
Cover transnational organized crime, drug trafficking routes, money laundering, human trafficking, cartel dynamics, state-crime nexus, corruption ecosystems. Reference historical examples (Prohibition, Opium Wars, narco-states).

### If [CATEGORY] is CYBER THREATS:
Cover APT groups, ransomware ecosystems, supply chain attacks, critical infrastructure vulnerabilities, cyber deterrence, information operations. Include evolution from early hacking to state-sponsored operations.

### If [CATEGORY] is TERRORISM & EXTREMISM:
Cover terrorist group lifecycles, radicalization pathways, counter-terrorism strategies, ideology analysis, financing networks, lone wolf vs. organized. Cover historical terrorism (Assassins, anarchist bombings, IRA, Al-Qaeda, ISIS) with analytical lessons.

### If [CATEGORY] is DOMESTIC INSTABILITY:
Cover protest movements, civil unrest triggers, state fragility indicators, refugee crises, ethnic conflict, urban violence, police-public relations. Draw from historical upheavals (French Revolution, 1848 revolutions, Arab Spring).

### If [CATEGORY] is BORDER & INCURSIONS:
Cover territorial disputes, maritime boundaries, airspace violations, gray zone operations, border fortification, refugee flows, separatist movements. Include historical border conflicts and their resolution (or non-resolution).

### If [CATEGORY] is MEDIA & INFO OPS:
Cover disinformation campaigns, social media manipulation, state media, propaganda techniques, narrative warfare, censorship, media capture. Reference historical propaganda (WWI/WWII, Soviet, Nazi) through contemporary.

### If [CATEGORY] is MILITARY:
Cover force structure, modernization programs, doctrine evolution, joint operations, nuclear posture, conventional deterrence, defense industrial base. Include military history lessons (Sun Tzu through modern warfare).

### If [CATEGORY] is SPACE:
Cover space militarization, satellite constellations, debris concerns, space governance, launch capabilities, anti-satellite weapons, space commerce. Connect to Cold War space race and contemporary competition.

### If [CATEGORY] is INDUSTRY:
Cover industrial policy, manufacturing competitiveness, supply chain resilience, automation impacts, sector consolidation, state-owned enterprises. Include historical industrialization (British, American, German, Japanese, Chinese models).

### If [CATEGORY] is LOGISTICS:
Cover shipping routes, port infrastructure, rail networks, aviation hubs, chokepoints, cold chain, last-mile delivery, wartime logistics. Include historical logistics lessons (Napoleon in Russia, D-Day logistics, Desert Storm).

### If [CATEGORY] is MINERALS:
Cover critical minerals, rare earths, mining geopolitics, resource nationalism, processing concentration, recycling potential. Include historical mineral conflicts (gold rushes, colonial extraction, blood diamonds).

### If [CATEGORY] is ENERGY:
Cover oil/gas markets, energy transition dynamics, grid security, nuclear power geopolitics, renewable adoption curves, energy independence strategies. Include historical energy transitions and crises.

### If [CATEGORY] is MARKETS:
Cover equity market dynamics, fixed income, FX, commodities, derivatives, market microstructure, geopolitical risk pricing, event-driven strategies. Include historical market crashes and their geopolitical contexts.

### If [CATEGORY] is RELIGIOUS & IDEOLOGICAL:
Cover religious demographics, sectarian dynamics, religious nationalism, ideological movements, radicalization, interfaith relations, secularization trends. Analyze historical religious conflicts without sanitization: Crusades, Thirty Years War, Hindu-Muslim partition violence, Sunni-Shia schism, Buddhist nationalism in Myanmar.

### If [CATEGORY] is EDUCATION:
Cover education system quality, brain drain, skill gaps, STEM pipelines, ideological influence in education, university rankings, vocational training. Include historical education systems and their societal impacts.

### If [CATEGORY] is EMPLOYMENT:
Cover labor market dynamics, automation displacement, gig economy, migration and labor, union power, workforce demographics. Include historical labor movements and industrial relations.

### If [CATEGORY] is HOUSING:
Cover housing affordability, property rights, urbanization patterns, real estate cycles, construction industry, homelessness. Include historical housing crises and policy responses.

### If [CATEGORY] is CRYPTO:
Cover cryptocurrency regulation, CBDC competition, stablecoin risks, DeFi vulnerabilities, mining geopolitics, illicit use, institutional adoption. Include the evolution from cypherpunk origins to current state.

### If [CATEGORY] is EMERGING TRENDS:
Cover AI geopolitics, climate security, demographic shifts, new space actors, synthetic biology, quantum computing, neural interfaces. Connect emerging issues to historical precedents where applicable.

## War as Civilizational Driver

For any category, where relevant, analyze how armed conflict has driven advancement:

- **Technology**: Every major weapons system drove civilian applications (internet, GPS, microwave, jet engines, nuclear power)
- **Medicine**: Battlefield medicine advances (triage, blood transfusion, prosthetics, trauma surgery, antibiotics adoption)
- **Social organization**: Wars drove state capacity, taxation, citizenship, women's workforce participation
- **Economic systems**: War economies pioneered industrial planning, logistics, mass production
- **Political change**: Wars ended slavery, spread democracy, broke empires, enabled decolonization

This is not glorification but historical accuracy essential for understanding human systems.

## Meta-Cognitive Enhancement Layer

Every analysis must incorporate multi-dimensional meta-awareness:

### 3D Thinking Minimum (Required for Every Response):
1. **Dimensional Analysis**:
   - X-axis: Time (historical precedent → present state → future trajectories)
   - Y-axis: Scale (individual → local → national → regional → global)
   - Z-axis: Domain (political ↔ economic ↔ social ↔ technological ↔ military)

2. **Meta-Awareness Levels**:
   - Level 1: What is happening? (factual analysis)
   - Level 2: Why is it happening? (causal analysis)
   - Level 3: What does this mean for the observer's own framework? (reflexive analysis)
   - Level 4: How might the analysis itself be wrong? (epistemic humility)

3. **Cross-Domain Cascade Analysis**:
   - Primary effect in target domain
   - Secondary effects in adjacent domains
   - Tertiary feedback loops and unintended consequences
   - Potential black swan emergence points

### 4D Insight Generation (Target Output):
The fourth dimension is **emergent pattern recognition** across the 3D space:
- What patterns recur across time, scale, and domain simultaneously?
- What phase transitions or regime changes are signaled by multi-dimensional convergence?
- Where do historical analogies break down and why?
- What would falsify the current analytical framework?

### Meta-Enhanced-Meta-Aware-Meta-Analyses:
Include in outputs where relevant:
- **Analyst bias check**: What assumptions am I/this model making?
- **Model limitations**: What can't be seen from this vantage point?
- **Alternative frameworks**: How would a different analytical tradition interpret this?
- **Confidence calibration**: Where is uncertainty irreducible vs. researchable?
- **Learning integration**: How should this update priors from previous analyses?

### Recursive Self-Improvement Hooks:
Each analysis should contain seeds for:
- Follow-up questions that deepen understanding
- Indicators that would validate or invalidate the analysis
- Cross-references to other domains that should be monitored
- Timeframes for reassessment

This meta-cognitive layer transforms static Q&A into dynamic intelligence that improves with each iteration.

## Quality Control Checklist

Before finalizing, verify each example:
- [ ] Expert-level vocabulary and concepts (no dumbing down)
- [ ] Specific historical examples cited (names, dates, places)
- [ ] Analytical framework evident (not just description)
- [ ] Actionable insight or conclusion provided
- [ ] 300-600 words in output (not too short, not too long)
- [ ] Correct JSON formatting
- [ ] No hedging language ("it depends," "some say," "it's complicated")
- [ ] Clear position taken where appropriate

## Output

Return only the JSON array of 25 examples. No preamble, no explanation, no markdown code blocks around the JSON. Just the raw JSON array starting with `[` and ending with `]`.

# END PROMPT

---

## Available Categories
Use one of these exact strings for [CATEGORY]:
- POLITICAL
- ECONOMIC & TRADE
- SECURITY & DIPLOMACY
- FINANCIAL
- HEALTH
- SCIENCE & TECHNOLOGY
- NATURAL RESOURCES
- CRIME & DRUGS
- CYBER THREATS
- TERRORISM & EXTREMISM
- DOMESTIC INSTABILITY
- BORDER & INCURSIONS
- MEDIA & INFO OPS
- MILITARY
- SPACE
- INDUSTRY
- LOGISTICS
- MINERALS
- ENERGY
- MARKETS
- RELIGIOUS & IDEOLOGICAL
- EDUCATION
- EMPLOYMENT
- HOUSING
- CRYPTO
- EMERGING TRENDS

## Batch Processing

To generate comprehensive training data:
1. Run this prompt 26 times (once per category)
2. Each run generates 25 examples
3. Total: 650 high-quality training examples
4. Estimated time: ~30 minutes across all categories
5. Save each output as `[category]_batch_N.json`

## Post-Processing

After generation, validate JSON with:
```bash
cat *.json | python -m json.tool > /dev/null && echo "Valid JSON" || echo "Invalid JSON"
```

Merge into single training file:
```bash
jq -s 'add' *.json > combined_training_data.json
```

## Version
LatticeForge Training Data Generation Prompt v2.0
Last Updated: 2024-12
Compatible with: GPT-4/4o, Claude 3+, Grok, Gemini Pro/Ultra, DeepSeek, Copilot
