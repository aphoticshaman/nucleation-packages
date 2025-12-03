#!/usr/bin/env python3
"""
Generate training data on freedom, coercion, labor choice, and the spectrum
of voluntary vs. forced participation in economic and social systems.
"""
import json
from typing import List, Dict

def generate_examples() -> List[Dict]:
    examples = []

    # ==========================================================================
    # 1. ECONOMIC COERCION & LABOR "CHOICE"
    # ==========================================================================
    labor_coercion = [
        # Sweatshops and manufacturing
        "How do sweatshop workers 'choose' between starvation wages and no wages?",
        "What economic conditions make sweatshop labor the 'best available option'?",
        "How does global supply chain pressure create coercive labor conditions?",
        "What is the difference between a sweatshop and a living wage job in practice?",
        "How do export processing zones create captive labor markets?",
        "What happens to communities when the only employer is an exploitative factory?",
        "How does debt bondage trap workers in exploitative labor arrangements?",
        "What role do labor recruiters play in coercive employment systems?",

        # Agricultural labor
        "How are migrant farmworkers trapped in cycles of exploitative labor?",
        "What makes seasonal agricultural work coercive despite being 'voluntary'?",
        "How do visa systems create power imbalances between workers and employers?",
        "What options do undocumented workers have when facing labor exploitation?",
        "How does the piece-rate system in agriculture mask coercive conditions?",

        # Gig economy and modern precarity
        "How does algorithmic management create new forms of labor coercion?",
        "What freedom do gig workers actually have in their 'flexible' arrangements?",
        "How does classification as independent contractors strip worker protections?",
        "What happens when platform workers have no alternative income sources?",
        "How do rating systems create coercive compliance in gig work?",

        # General economic coercion
        "What is the difference between choosing a job and being forced into one?",
        "How does lack of social safety nets convert all labor into coerced labor?",
        "What role does healthcare tied to employment play in labor coercion?",
        "How does student debt constrain career choices and create coercion?",
        "What options exist when every available job pays below living wage?",
    ]

    for q in labor_coercion:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Labor coercion analysis: structural constraints, alternative options, power dynamics, and the spectrum between voluntary choice and forced participation.]"
        })

    # ==========================================================================
    # 2. HUMAN TRAFFICKING & MODERN SLAVERY
    # ==========================================================================
    trafficking = [
        # Mechanisms
        "How does human trafficking operate through seemingly legal channels?",
        "What role does debt bondage play in modern trafficking systems?",
        "How are victims recruited into trafficking through false promises?",
        "What makes someone vulnerable to human trafficking?",
        "How do traffickers maintain control over victims?",
        "What is the relationship between migration and trafficking vulnerability?",
        "How do legal visa programs sometimes enable trafficking?",
        "What role does document confiscation play in trafficking control?",

        # Sectors
        "How does labor trafficking operate in domestic work?",
        "What are the trafficking patterns in construction and agriculture?",
        "How does trafficking intersect with the fishing industry?",
        "What trafficking risks exist in hospitality and service sectors?",
        "How does trafficking manifest in manufacturing supply chains?",

        # Geographic patterns
        "What are the major trafficking routes and source/destination patterns?",
        "How does internal trafficking differ from cross-border trafficking?",
        "What role do conflict zones play in trafficking vulnerability?",
        "How do economic disparities between regions enable trafficking?",

        # Responses
        "Why do trafficking victims often not self-identify or seek help?",
        "What barriers prevent trafficking victims from escaping?",
        "How effective are anti-trafficking laws and enforcement?",
        "What support systems exist for trafficking survivors?",
    ]

    for q in trafficking:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Human trafficking analysis: mechanisms of control, vulnerability factors, systemic enablers, and the spectrum of coercion that traps victims.]"
        })

    # ==========================================================================
    # 3. ILLICIT ECONOMIES & SURVIVAL
    # ==========================================================================
    illicit_economies = [
        # Drug trade
        "What economic conditions push people into drug trafficking?",
        "How does the drug trade become the only employer in some communities?",
        "What 'choices' do young people have in cartel-controlled territories?",
        "How does the drug economy cascade through legitimate local economies?",
        "What happens to people who try to leave the drug trade?",
        "How do drug organizations recruit and retain workers?",
        "What is the mortality and incarceration rate for drug trade workers?",
        "How does prohibition create the economic conditions for coercive participation?",

        # Sex work spectrum
        "What is the spectrum from voluntary sex work to trafficking?",
        "How do economic conditions push people into survival sex work?",
        "What distinguishes coerced sex work from chosen sex work?",
        "How do criminalization policies affect sex worker safety and choice?",
        "What exit options exist for those wanting to leave sex work?",
        "How does the Nordic model affect sex worker agency and safety?",
        "What role does addiction play in coerced sex work?",
        "How do pimps and managers create debt bondage in sex work?",

        # Crime as survival
        "When does theft become a survival strategy rather than a choice?",
        "How do communities with no legitimate economy produce 'criminals'?",
        "What options exist for formerly incarcerated people in the labor market?",
        "How does the school-to-prison pipeline constrain life choices?",
        "What is the relationship between poverty and property crime?",
        "How do three-strikes laws affect the risk calculus of crime?",
    ]

    for q in illicit_economies:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Illicit economy analysis: structural drivers, constrained choices, survival strategies, and the continuum between voluntary participation and coerced involvement.]"
        })

    # ==========================================================================
    # 4. ARMED GROUPS & MILITARIZATION
    # ==========================================================================
    armed_groups = [
        # Gangs
        "What drives young people to join gangs in underserved communities?",
        "How do gangs provide social services that the state doesn't?",
        "What protection calculus makes gang membership 'rational'?",
        "How difficult is it to leave gang life once involved?",
        "What happens to communities when gangs are the primary authority?",
        "How do gangs recruit and retain members through coercion vs. incentives?",

        # Cartels
        "How do cartels become the de facto government in some regions?",
        "What choices do local businesses have in cartel-controlled areas?",
        "How do cartels recruit skilled professionals like accountants and lawyers?",
        "What happens to people who refuse cartel employment or cooperation?",
        "How do cartels provide economic stability that states fail to provide?",
        "What is the 'plata o plomo' dynamic and how does it constrain choice?",

        # Militias and paramilitaries
        "What conditions lead people to join militia movements?",
        "How do political militias recruit and radicalize members?",
        "What economic incentives exist for joining armed groups?",
        "How do paramilitaries operate in weak-state environments?",
        "What role does identity and belonging play in armed group recruitment?",

        # Military recruitment
        "How does the 'economic draft' shape military recruitment?",
        "What options do young people in poor communities have besides military service?",
        "How do military recruiters target economically disadvantaged areas?",
        "What is the relationship between college costs and military enlistment?",
        "How do signing bonuses and benefits packages influence military 'choice'?",
        "What happens to veterans who struggle to transition to civilian employment?",
        "How does the military provide stability that civilian life doesn't offer?",
        "What is the difference between patriotic service and economic necessity?",
    ]

    for q in armed_groups:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Armed group analysis: recruitment dynamics, structural drivers, constrained alternatives, and the spectrum between voluntary affiliation and coerced membership.]"
        })

    # ==========================================================================
    # 5. STATE INSTITUTIONS & COERCION
    # ==========================================================================
    state_institutions = [
        # Police and law enforcement
        "What economic conditions drive people to become police officers?",
        "How does policing in corrupt systems differ from idealized versions?",
        "What happens to officers who refuse to participate in corruption?",
        "How do quotas and performance metrics distort policing behavior?",
        "What is the spectrum of police work from community service to oppression?",
        "How do officers rationalize participation in unjust enforcement?",
        "What options do police have when ordered to do unethical things?",

        # Corrupt institutions
        "How do people navigate employment in corrupt bureaucracies?",
        "What happens to whistleblowers in corrupt institutional environments?",
        "How does institutional corruption become normalized over time?",
        "What choices do civil servants have in autocratic systems?",
        "How do people maintain integrity while working in compromised institutions?",

        # Prisons and detention
        "How does prison labor blur the line between work and slavery?",
        "What 'choices' do incarcerated people have regarding labor?",
        "How do private prisons create incentives for mass incarceration?",
        "What are the conditions in immigration detention facilities?",
        "How does the 13th Amendment exception enable prison labor exploitation?",
    ]

    for q in state_institutions:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[State institution analysis: employment dynamics, ethical compromises, systemic pressures, and individual agency within institutional constraints.]"
        })

    # ==========================================================================
    # 6. LEVELS OF FREEDOM - THEORETICAL FRAMEWORK
    # ==========================================================================
    freedom_theory = [
        # Conceptual distinctions
        "What is the difference between formal freedom and substantive freedom?",
        "How does negative liberty differ from positive liberty?",
        "What is the relationship between freedom and capability?",
        "How do background conditions affect the meaning of 'choice'?",
        "What is the difference between freedom from and freedom to?",
        "How does structural constraint differ from direct coercion?",
        "What makes a choice genuinely voluntary versus nominally voluntary?",
        "How do philosophers distinguish between freedom and autonomy?",

        # True freedom
        "What would true economic freedom actually look like?",
        "How much material security is required for genuine choice?",
        "What social conditions enable authentic self-determination?",
        "How does wealth inequality affect the distribution of freedom?",
        "What is the relationship between freedom and power?",

        # Inherent vs. granted freedom
        "What freedoms are considered inherent human rights?",
        "How do legal frameworks create or restrict freedom?",
        "What is the difference between natural rights and civil rights?",
        "How do constitutions attempt to protect freedom?",
        "What happens when inherent rights conflict with granted rights?",

        # Implied and restricted freedom
        "What freedoms are implied but not guaranteed in democratic societies?",
        "How do social norms restrict formally available freedoms?",
        "What is the relationship between cultural expectations and freedom?",
        "How do economic systems create implicit restrictions on choice?",
        "What freedoms do people assume they have but actually don't?",
    ]

    for q in freedom_theory:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Freedom theory analysis: conceptual distinctions, structural conditions, capability requirements, and the gap between formal and substantive liberty.]"
        })

    # ==========================================================================
    # 7. PERCEIVED VS. ACTUAL FREEDOM
    # ==========================================================================
    perceived_vs_actual = [
        # Consumer choice illusion
        "How does consumer choice create an illusion of freedom?",
        "What is the difference between choosing products and choosing life paths?",
        "How do marketing and advertising shape perceived freedom?",
        "What freedoms do Americans believe they have that don't exist?",
        "How does media consumption affect perceptions of freedom?",

        # Political freedom illusion
        "How does voting between limited options relate to political freedom?",
        "What is the relationship between democracy and actual freedom?",
        "How do two-party systems constrain political choice?",
        "What freedoms do citizens have between elections?",
        "How does corporate influence on politics affect citizen freedom?",

        # Social freedom illusion
        "How do social expectations constrain nominally free choices?",
        "What is the relationship between conformity and perceived freedom?",
        "How does social media create illusions of self-expression and freedom?",
        "What are people actually free to do versus what they believe they can do?",
        "How do cultural scripts constrain life choices?",

        # Measurement and comparison
        "How do freedom indices measure freedom and what do they miss?",
        "What is the relationship between economic development and freedom?",
        "How does the US compare to other developed nations on substantive freedom?",
        "What freedoms do Americans have that others don't, and vice versa?",
        "How do different societies conceptualize and prioritize different freedoms?",
    ]

    for q in perceived_vs_actual:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Perceived vs actual freedom analysis: illusion mechanisms, measurement challenges, comparative perspectives, and the gap between belief and reality.]"
        })

    # ==========================================================================
    # 8. GEOGRAPHIC & SOCIOECONOMIC VARIATION
    # ==========================================================================
    geographic_variation = [
        # Urban vs. rural
        "How does freedom differ between urban and rural environments?",
        "What economic options exist in rural areas versus cities?",
        "How does transportation access affect freedom in different locations?",
        "What freedoms do rural communities have that urban areas lack?",
        "How does internet access affect rural vs. urban freedom?",

        # Regional US variation
        "How does freedom differ between coastal cities and heartland towns?",
        "What economic mobility exists in different US regions?",
        "How do state-level policies create different freedom profiles?",
        "What freedoms exist in Alaska bush communities versus suburban LA?",
        "How does cost of living constrain freedom in different regions?",

        # Suburban reality
        "What freedoms does suburban living actually provide?",
        "How does car dependency affect suburban freedom?",
        "What economic constraints exist in supposedly comfortable suburbs?",
        "How do HOAs and zoning restrict suburban freedom?",
        "What is the relationship between suburban isolation and perceived freedom?",

        # Extreme environments
        "What freedoms exist for subsistence communities in remote Alaska?",
        "How does survival necessity constrain choice in frontier environments?",
        "What trade-offs exist between self-sufficiency and social services?",
        "How do indigenous communities balance traditional freedom with modern constraints?",
        "What happens when you're free from society but constrained by nature?",

        # Global comparison
        "How does freedom differ between developed and developing nations?",
        "What freedoms do Scandinavian citizens have that Americans lack?",
        "How do different healthcare systems affect life freedom?",
        "What is the relationship between social democracy and individual freedom?",
        "How does authoritarian stability compare to democratic precarity?",
    ]

    for q in geographic_variation:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Geographic freedom analysis: environmental constraints, economic opportunities, policy variation, and how location shapes the practical meaning of freedom.]"
        })

    # ==========================================================================
    # 9. SYSTEMIC CONSTRAINTS ON CHOICE
    # ==========================================================================
    systemic_constraints = [
        # Healthcare
        "How does the US healthcare system constrain employment freedom?",
        "What happens to entrepreneurship when health insurance is employer-tied?",
        "How does chronic illness affect freedom and choice?",
        "What is 'job lock' and how does it affect labor freedom?",
        "How do medical bankruptcies affect life choices?",

        # Housing
        "How does housing cost affect freedom to choose location?",
        "What happens when housing consumes most of income?",
        "How does homeownership both enable and constrain freedom?",
        "What freedoms do renters lack compared to owners?",
        "How does homelessness represent the ultimate constraint on choice?",

        # Education
        "How does educational debt constrain post-graduation choices?",
        "What is the relationship between education access and freedom?",
        "How do educational credentials become barriers to opportunity?",
        "What happens to people who can't access higher education?",
        "How does educational tracking constrain future choices?",

        # Family and care
        "How do childcare costs constrain parental employment choices?",
        "What freedoms do primary caregivers sacrifice?",
        "How does elder care responsibility affect career freedom?",
        "What choices do single parents actually have?",
        "How does lack of parental leave affect reproductive freedom?",

        # Time and attention
        "How does time poverty constrain freedom?",
        "What happens when people have no time for political participation?",
        "How does overwork affect freedom to live fully?",
        "What is the relationship between leisure and freedom?",
        "How does attention economy affect cognitive freedom?",
    ]

    for q in systemic_constraints:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Systemic constraint analysis: institutional barriers, structural limitations, and how systems designed for other purposes end up restricting individual freedom.]"
        })

    # ==========================================================================
    # 10. ESCAPE AND ALTERNATIVE PATHS
    # ==========================================================================
    alternatives = [
        # Individual escape routes
        "What options exist for escaping poverty traps?",
        "How do people break out of constrained circumstances?",
        "What role does luck play in escaping structural constraints?",
        "How do networks and social capital enable escape from constraint?",
        "What individual strategies work for expanding personal freedom?",

        # Collective alternatives
        "How do labor unions expand worker freedom and choice?",
        "What role do cooperatives play in expanding economic freedom?",
        "How do mutual aid networks expand freedom for participants?",
        "What collective movements have successfully expanded freedom?",
        "How does organizing change the balance of power and choice?",

        # Policy alternatives
        "How would universal basic income affect freedom and choice?",
        "What is the relationship between universal healthcare and freedom?",
        "How does free higher education expand life choices?",
        "What housing policies would expand freedom?",
        "How do different retirement systems affect late-life freedom?",

        # Radical alternatives
        "What would a society with genuine freedom of labor look like?",
        "How do intentional communities attempt to create alternative freedom?",
        "What can be learned from experiments in expanded freedom?",
        "How do different economic systems distribute freedom differently?",
        "What structural changes would be required for genuine freedom?",
    ]

    for q in alternatives:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Alternative paths analysis: escape mechanisms, collective strategies, policy options, and structural changes that could expand genuine freedom.]"
        })

    # ==========================================================================
    # 11. PHILOSOPHICAL & ETHICAL DIMENSIONS
    # ==========================================================================
    philosophical = [
        # Responsibility and blame
        "How should we think about responsibility when choices are constrained?",
        "What does it mean to 'choose' under coercive conditions?",
        "How do we assign moral blame in systems of constrained choice?",
        "What is the relationship between freedom and responsibility?",
        "How should society respond to 'crimes of necessity'?",

        # Dignity and worth
        "What is the relationship between freedom and human dignity?",
        "How does constrained choice affect human flourishing?",
        "What minimum freedom is required for a dignified life?",
        "How do we value freedom relative to security and stability?",
        "What trade-offs between freedoms are acceptable?",

        # Justice
        "How does unequal distribution of freedom relate to justice?",
        "What obligations do the free have toward the constrained?",
        "How should societies prioritize which freedoms to protect?",
        "What is the relationship between equality and freedom?",
        "How do libertarian and egalitarian conceptions of freedom differ?",

        # Future
        "How might automation affect future freedom of labor?",
        "What new constraints and freedoms will technology create?",
        "How might climate change affect freedom and choice?",
        "What would post-scarcity mean for freedom?",
        "How should we think about freedom for future generations?",
    ]

    for q in philosophical:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Philosophical analysis: ethical frameworks, responsibility attribution, justice considerations, and normative questions about freedom and constraint.]"
        })

    return examples


def main():
    print("=" * 70)
    print("FREEDOM & COERCION TRAINING DATA GENERATOR")
    print("=" * 70)

    examples = generate_examples()
    print(f"\nGenerated {len(examples)} training examples")

    # Categorize
    categories = {
        "labor_coercion": 0,
        "trafficking": 0,
        "illicit_economy": 0,
        "armed_groups": 0,
        "state_institutions": 0,
        "freedom_theory": 0,
        "perceived_vs_actual": 0,
        "geographic": 0,
        "systemic": 0,
        "alternatives": 0,
        "philosophical": 0,
        "other": 0,
    }

    for ex in examples:
        text = ex["instruction"].lower()
        if any(x in text for x in ["sweatshop", "wage", "gig", "platform", "employer", "worker"]):
            categories["labor_coercion"] += 1
        elif any(x in text for x in ["trafficking", "bondage", "victim", "recruit"]):
            categories["trafficking"] += 1
        elif any(x in text for x in ["drug", "sex work", "cartel", "theft", "crime", "illicit"]):
            categories["illicit_economy"] += 1
        elif any(x in text for x in ["gang", "militia", "military", "armed", "paramilitary"]):
            categories["armed_groups"] += 1
        elif any(x in text for x in ["police", "prison", "corrupt", "institution", "civil servant"]):
            categories["state_institutions"] += 1
        elif any(x in text for x in ["liberty", "autonomy", "voluntary", "negative", "positive", "rights"]):
            categories["freedom_theory"] += 1
        elif any(x in text for x in ["perceive", "illusion", "believe", "consumer choice"]):
            categories["perceived_vs_actual"] += 1
        elif any(x in text for x in ["urban", "rural", "suburban", "alaska", "region", "coast"]):
            categories["geographic"] += 1
        elif any(x in text for x in ["healthcare", "housing", "education", "childcare", "debt"]):
            categories["systemic"] += 1
        elif any(x in text for x in ["escape", "alternative", "union", "ubi", "cooperative", "policy"]):
            categories["alternatives"] += 1
        elif any(x in text for x in ["responsibility", "dignity", "justice", "ethical", "moral"]):
            categories["philosophical"] += 1
        else:
            categories["other"] += 1

    print("\n" + "=" * 70)
    print("CATEGORY BREAKDOWN:")
    print("=" * 70)
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} examples")

    # Save
    output_path = "freedom_coercion_training.json"
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"\nSaved to {output_path}")

    return examples


if __name__ == "__main__":
    main()
