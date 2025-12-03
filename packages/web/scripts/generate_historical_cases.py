#!/usr/bin/env python3
"""Historical case studies and pattern recognition training data."""
import json

def generate_examples():
    examples = []

    # MARKET CRASHES
    crashes = [
        ("1929 crash", "What caused the 1929 stock market crash?", "How did the 1929 crash cascade into the Great Depression?", "What warning signs preceded the 1929 crash?", "How did margin trading contribute to the 1929 crash?", "What policy responses followed the 1929 crash?"),
        ("1987 Black Monday", "What caused Black Monday in 1987?", "How did portfolio insurance contribute to the 1987 crash?", "What role did program trading play in 1987?", "How did markets recover after Black Monday?", "What circuit breakers emerged from the 1987 crash?"),
        ("1997 Asian crisis", "What triggered the 1997 Asian financial crisis?", "How did currency pegs contribute to the 1997 crisis?", "What was the contagion pattern in the Asian crisis?", "How did the IMF response affect the 1997 crisis?", "What lessons emerged from the 1997 Asian crisis?"),
        ("2000 dotcom", "What caused the dotcom bubble and crash?", "How did valuation metrics fail during the dotcom bubble?", "What survived the dotcom crash and why?", "How did the dotcom crash affect venture capital?", "What patterns from dotcom recur in tech bubbles?"),
        ("2008 financial crisis", "What caused the 2008 global financial crisis?", "How did subprime mortgages cascade through the financial system?", "What role did CDOs and credit derivatives play in 2008?", "How did Lehman Brothers' failure trigger contagion?", "What regulatory changes followed the 2008 crisis?"),
        ("2010 Flash Crash", "What caused the 2010 Flash Crash?", "How did algorithmic trading contribute to the Flash Crash?", "What circuit breakers emerged from the Flash Crash?", "How do flash crashes inform market structure design?", "What vulnerabilities did the Flash Crash reveal?"),
        ("2020 COVID crash", "How did COVID-19 trigger the March 2020 crash?", "What was unprecedented about the 2020 market recovery?", "How did Fed intervention affect the 2020 crash?", "What sectors diverged during the COVID crash?", "What lessons emerged from the 2020 market volatility?"),
    ]
    for crash in crashes:
        name = crash[0]
        for q in crash[1:]:
            examples.append({"instruction": q, "input": "", "output": f"[{name} analysis: causal factors, cascade dynamics, policy responses, and lessons for future crisis recognition.]"})

    # WARS AS ECONOMIC EVENTS
    wars = [
        ("WWI economics", "How did WWI reshape global economic leadership?", "What was the economic cost of WWI and how was it financed?", "How did war debts cascade through the interwar period?", "What industries emerged or transformed from WWI?", "How did WWI affect global trade patterns?"),
        ("WWII economics", "How did WWII transform the US economy?", "What was the economic mobilization model of WWII?", "How did WWII set up the postwar economic order?", "What technologies from WWII transformed civilian economy?", "How did WWII affect global currency arrangements?"),
        ("Korean War", "How did the Korean War affect US defense spending?", "What industries benefited from Korean War mobilization?", "How did the Korean War affect inflation and prices?", "What was the economic impact of Korean War on Japan?", "How did Korean War shape Cold War economics?"),
        ("Vietnam War", "How did Vietnam War spending affect the US economy?", "What was the guns and butter problem during Vietnam?", "How did Vietnam War contribute to 1970s inflation?", "What was the economic relationship between Vietnam and the draft?", "How did Vietnam War affect the dollar and gold standard?"),
        ("Gulf War", "What was the economic impact of the Gulf War on oil markets?", "How did the Gulf War affect defense industry structure?", "What was the cost of the Gulf War and who paid?", "How did the Gulf War affect regional economies?", "What economic factors contributed to the Gulf War?"),
        ("Iraq/Afghanistan", "What was the full economic cost of Iraq and Afghanistan wars?", "How did post-9/11 wars affect defense spending patterns?", "What opportunity costs came from the post-9/11 wars?", "How did the wars affect veteran economics and healthcare?", "What industries benefited from the post-9/11 wars?"),
        ("Ukraine war", "How has the Ukraine war affected global energy markets?", "What is the economic impact of Russia sanctions?", "How has the Ukraine war affected food security and prices?", "What defense industry changes result from Ukraine?", "How has the Ukraine war affected European energy security?"),
    ]
    for war in wars:
        name = war[0]
        for q in war[1:]:
            examples.append({"instruction": q, "input": "", "output": f"[{name} analysis: economic mobilization, industry transformation, fiscal impact, and long-term structural effects.]"})

    # PANDEMIC PATTERNS
    pandemics = [
        ("1918 flu", "What was the economic impact of the 1918 influenza pandemic?", "How did the 1918 pandemic affect labor markets?", "What industries were most affected by the 1918 flu?", "How did cities that locked down earlier recover economically?", "What long-term changes followed the 1918 pandemic?"),
        ("SARS 2003", "What was the economic impact of SARS in 2003?", "How did SARS affect Asian tourism and travel?", "What public health changes followed SARS?", "How did SARS affect supply chains temporarily?", "What lessons from SARS were applied to COVID?"),
        ("H1N1 2009", "What was the economic impact of H1N1 in 2009?", "How did H1N1 affect pandemic preparedness?", "What vaccine development lessons came from H1N1?", "How did H1N1 response differ from COVID?", "What economic sectors were most affected by H1N1?"),
        ("COVID-19", "How did COVID-19 accelerate existing economic trends?", "What was the K-shaped recovery from COVID?", "How did COVID affect commercial real estate?", "What labor market changes from COVID are permanent?", "How did COVID affect wealth inequality?"),
    ]
    for pandemic in pandemics:
        name = pandemic[0]
        for q in pandemic[1:]:
            examples.append({"instruction": q, "input": "", "output": f"[{name} analysis: economic disruption patterns, sector impacts, policy responses, and lasting structural changes.]"})

    # TECHNOLOGY S-CURVES
    tech = [
        ("Railroad", "How did railroads transform the 19th century economy?", "What was the railroad bubble and how did it end?", "How did railroads affect land values and development?", "What industries were created by railroads?", "How did railroad financing innovate capital markets?"),
        ("Electricity", "How did electrification transform manufacturing?", "What was the productivity paradox with early electrification?", "How long did it take for electricity to show economic gains?", "What industries were created by electrification?", "How did electricity change urban development?"),
        ("Automobile", "How did automobiles transform the 20th century economy?", "What industries were created by the automobile?", "How did automobiles affect urban planning and suburbs?", "What was the economic impact of the interstate highway system?", "How did automobiles affect retail and logistics?"),
        ("Television", "How did television transform advertising and marketing?", "What industries were disrupted by television?", "How did television affect political campaigns?", "What was the economic model of broadcast television?", "How did cable disrupt broadcast economics?"),
        ("Personal computer", "How did personal computers transform productivity?", "What was the Solow productivity paradox with computers?", "How did PCs transform office work?", "What industries were created by personal computers?", "How did PCs affect the geography of work?"),
        ("Internet", "How did the internet transform commerce?", "What industries were disrupted by the internet?", "How did the internet affect information economics?", "What business models emerged from the internet?", "How did the internet affect globalization?"),
        ("Smartphone", "How did smartphones transform mobile commerce?", "What industries were disrupted by smartphones?", "How did smartphones affect attention and advertising?", "What was the app economy's impact?", "How did smartphones affect developing economies?"),
        ("AI/ML", "How is AI transforming knowledge work?", "What industries will be most disrupted by AI?", "How does the AI S-curve compare to previous technologies?", "What is the timeline for AI economic impact?", "How might AI affect employment differently than previous technologies?"),
    ]
    for technology in tech:
        name = technology[0]
        for q in technology[1:]:
            examples.append({"instruction": q, "input": "", "output": f"[{name} analysis: adoption curve, industry creation/disruption, productivity effects, and economic transformation patterns.]"})

    # CURRENCY CRISES
    currency = [
        ("Bretton Woods collapse", "What caused the collapse of Bretton Woods?", "How did the Nixon shock affect global currency markets?", "What replaced the Bretton Woods system?", "How did floating exchange rates affect trade?", "What lessons from Bretton Woods apply today?"),
        ("Latin American debt 1980s", "What caused the Latin American debt crisis?", "How did petrodollar recycling contribute to the crisis?", "What was the Brady Plan and how did it work?", "How did the crisis affect development economics?", "What structural adjustment followed the crisis?"),
        ("ERM crisis 1992", "What caused the 1992 ERM crisis?", "How did Soros break the Bank of England?", "What was the role of German reunification in the crisis?", "How did the ERM crisis affect European integration?", "What lessons from the ERM informed the Euro design?"),
        ("Turkish lira crises", "What causes recurring Turkish lira crises?", "How do unorthodox monetary policies affect the lira?", "What is the relationship between politics and Turkish currency?", "How do Turkish crises affect regional markets?", "What structural factors make Turkey vulnerable?"),
        ("Argentine defaults", "What caused Argentina's repeated sovereign defaults?", "How did the currency board contribute to Argentina's 2001 crisis?", "What role does IMF policy play in Argentine crises?", "How does Argentine default history affect its borrowing costs?", "What patterns repeat in Argentine currency crises?"),
    ]
    for crisis in currency:
        name = crisis[0]
        for q in crisis[1:]:
            examples.append({"instruction": q, "input": "", "output": f"[{name} analysis: trigger factors, policy failures, contagion dynamics, and lessons for currency stability.]"})

    # EMPIRE RISE AND FALL
    empire = [
        "What patterns repeat in the rise and fall of economic powers?",
        "How did the Dutch Republic lose economic leadership?",
        "What factors enabled British economic dominance in the 19th century?",
        "How did the US become the dominant economic power?",
        "What are the indicators of relative decline in great powers?",
        "How does the Kindleberger trap explain hegemonic stability?",
        "What role does debt play in imperial decline?",
        "How do reserve currency advantages erode over time?",
        "What is the Thucydides trap in economic terms?",
        "How do demographic patterns affect great power trajectories?",
        "What role does innovation play in great power competition?",
        "How do financial centers shift between powers?",
        "What is the relationship between military and economic power?",
        "How do trading systems reflect and reinforce hegemony?",
        "What patterns mark the transition between hegemonic powers?",
    ]
    for q in empire:
        examples.append({"instruction": q, "input": "", "output": "[Great power analysis: rise and decline patterns, hegemonic transitions, structural factors, and historical precedents.]"})

    # REGULATORY RESPONSES
    regulatory = [
        "How did the Great Depression lead to financial regulation?",
        "What was the regulatory response to the 2008 crisis?",
        "How do crises typically lead to regulatory change?",
        "What is the regulatory cycle and how does it work?",
        "How does deregulation contribute to crisis conditions?",
        "What role does regulatory arbitrage play in crises?",
        "How do international regulatory frameworks evolve?",
        "What is regulatory capture and how does it affect crisis response?",
        "How do crises affect the balance between regulation and innovation?",
        "What patterns emerge in post-crisis regulatory overreach?",
        "How does technology outpace regulatory frameworks?",
        "What role does political economy play in regulatory design?",
        "How do different countries respond to the same crisis?",
        "What makes regulatory reforms durable vs. temporary?",
        "How do crises create windows for regulatory change?",
    ]
    for q in regulatory:
        examples.append({"instruction": q, "input": "", "output": "[Regulatory analysis: crisis-response patterns, reform durability, political economy factors, and international coordination.]"})

    return examples

if __name__ == "__main__":
    examples = generate_examples()
    print(f"Generated {len(examples)} historical case study examples")
    with open("historical_cases_training.json", "w") as f:
        json.dump(examples, f, indent=2)
    print("Saved to historical_cases_training.json")
