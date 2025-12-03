#!/usr/bin/env python3
"""
Generate training data on meta-linguistic compression, token efficiency,
user-driven exploration, and information architecture that maximizes
choice presentation while minimizing computational footprint.
"""
import json
from typing import List, Dict

def generate_examples() -> List[Dict]:
    examples = []

    # ==========================================================================
    # 1. META-LANGUAGE & SEMANTIC COMPRESSION
    # ==========================================================================
    meta_language = [
        # Compression principles
        "How can semantic density reduce token usage without losing meaning?",
        "What linguistic patterns compress complex ideas into minimal tokens?",
        "How do domain-specific vocabularies enable compression?",
        "What role does shared context play in communication efficiency?",
        "How can presupposition reduce explicit statement requirements?",
        "What makes some words carry more information than others?",
        "How do technical jargons achieve compression through specialization?",
        "What is the relationship between abstraction level and token efficiency?",

        # Meta-linguistic awareness
        "How does meta-linguistic awareness improve prompt engineering?",
        "What patterns allow language to reference itself efficiently?",
        "How can deixis reduce explicit context specification?",
        "What role does ellipsis play in efficient communication?",
        "How do anaphoric references compress repeated concepts?",
        "What linguistic shortcuts exist for common reasoning patterns?",
        "How can metaphor compress complex explanations?",
        "What makes some phrasings more token-efficient than others?",

        # Structured compression
        "How do structured formats compress compared to prose?",
        "What compression gains come from consistent formatting?",
        "How can hierarchical organization reduce redundancy?",
        "What role do schemas play in efficient data representation?",
        "How does typing/categorization enable compression?",
        "What compression emerges from predictable structure?",
    ]

    for q in meta_language:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Meta-linguistic analysis: compression mechanisms, information density, contextual efficiency, and practical token reduction strategies.]"
        })

    # ==========================================================================
    # 2. META-GRAMMAR & SYNTACTIC EFFICIENCY
    # ==========================================================================
    meta_grammar = [
        # Syntactic compression
        "How do different grammatical structures vary in token efficiency?",
        "What syntactic patterns maximize meaning per token?",
        "How can nominalization compress clauses into phrases?",
        "What role does word order play in compression efficiency?",
        "How do compound structures reduce syntactic overhead?",
        "What grammatical shortcuts exist for common patterns?",

        # Structural patterns
        "How do lists compress compared to sentences?",
        "What makes bullet points more efficient than paragraphs?",
        "How can tables compress relational information?",
        "What compression gains come from consistent grammar?",
        "How do abbreviated forms reduce token usage?",
        "What syntactic redundancy can be eliminated?",

        # Query efficiency
        "How should questions be structured for minimal tokens?",
        "What makes some prompts more efficient than others?",
        "How can compound questions reduce total token usage?",
        "What role does question presupposition play in efficiency?",
        "How can implicit context reduce explicit question length?",
    ]

    for q in meta_grammar:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Meta-grammar analysis: syntactic efficiency patterns, structural compression, and grammatical strategies for token reduction.]"
        })

    # ==========================================================================
    # 3. META-ETYMOLOGY & SEMANTIC DEPTH
    # ==========================================================================
    meta_etymology = [
        # Root power
        "How do etymological roots pack meaning across derived words?",
        "What compression power exists in Greek/Latin technical roots?",
        "How does knowledge of word origins enable denser communication?",
        "What semantic depth do compound technical terms carry?",
        "How do neologisms compress novel concepts?",

        # Semantic networks
        "How do word families enable implicit reference?",
        "What compression emerges from shared etymological roots?",
        "How does morphological awareness enable meaning inference?",
        "What role do prefixes/suffixes play in semantic compression?",
        "How can etymology guide efficient term selection?",

        # Domain vocabulary
        "How do technical vocabularies achieve semantic density?",
        "What compression exists in standardized terminology?",
        "How do acronyms and initialisms compress references?",
        "What efficiency gains come from shared professional vocabulary?",
        "How does jargon trade accessibility for compression?",
    ]

    for q in meta_etymology:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Meta-etymology analysis: root-level compression, semantic networks, vocabulary efficiency, and term selection strategies.]"
        })

    # ==========================================================================
    # 4. META-PHONETICS & COGNITIVE LOAD
    # ==========================================================================
    meta_phonetics = [
        # Processing efficiency
        "How does phonetic complexity affect cognitive processing?",
        "What makes some terms easier to process than others?",
        "How do mnemonic patterns aid information retention?",
        "What role does rhythm play in information processing?",
        "How can phonetic patterns signal semantic relationships?",

        # Interface implications
        "How should terms be chosen for user interface efficiency?",
        "What makes labels scannable and memorable?",
        "How do naming conventions affect user navigation speed?",
        "What phonetic properties make good category names?",
        "How can terminology design reduce cognitive load?",
    ]

    for q in meta_phonetics:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Meta-phonetics analysis: cognitive processing efficiency, naming conventions, and interface terminology optimization.]"
        })

    # ==========================================================================
    # 5. TOKEN ECONOMY & SPENDING STRATEGY
    # ==========================================================================
    token_economy = [
        # Cost optimization
        "How should token budgets be allocated across system components?",
        "What is the token cost of different prompt strategies?",
        "How can caching reduce redundant token spending?",
        "What trade-offs exist between token usage and response quality?",
        "How should token spending vary by user tier or query type?",
        "What compression techniques reduce API costs most?",

        # Prompt engineering
        "How do few-shot examples affect token efficiency?",
        "What is the token cost of different context strategies?",
        "How can system prompts be optimized for token efficiency?",
        "What role does prompt structure play in response efficiency?",
        "How do conversation patterns affect cumulative token usage?",
        "What strategies minimize tokens while maintaining quality?",

        # Response efficiency
        "How can response format requirements reduce token usage?",
        "What output structures are most token-efficient?",
        "How does requested verbosity affect token spending?",
        "What compression can be applied to model outputs?",
        "How should response length targets be set for efficiency?",

        # System design
        "How should systems be designed for token efficiency?",
        "What architectural patterns minimize token usage?",
        "How can multi-turn conversations be optimized for tokens?",
        "What preprocessing reduces token requirements?",
        "How does context window management affect token costs?",
    ]

    for q in token_economy:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Token economy analysis: cost optimization strategies, prompt efficiency patterns, and system design for minimal token spending.]"
        })

    # ==========================================================================
    # 6. USER-DRIVEN EXPLORATION PARADIGM
    # ==========================================================================
    user_driven = [
        # Blank canvas philosophy
        "Why start with a blank map rather than pre-populated views?",
        "How does user-driven selection improve engagement?",
        "What is the value of progressive disclosure over data dumps?",
        "How do empty states invite exploration?",
        "What makes 'pull' interfaces more efficient than 'push'?",
        "How does user agency affect information retention?",

        # Choice architecture
        "How should choices be structured for efficient selection?",
        "What makes dropdown menus efficient for exploration?",
        "How do checkboxes enable combinatorial exploration?",
        "What role do filters play in user-driven discovery?",
        "How should category hierarchies be designed for navigation?",
        "What makes faceted search effective for exploration?",

        # Cognitive efficiency
        "How does user selection reduce cognitive overload?",
        "What is the optimal number of choices to present?",
        "How can defaults guide without constraining?",
        "What role does progressive refinement play?",
        "How should complexity be revealed incrementally?",
        "What makes some navigation patterns more intuitive?",
    ]

    for q in user_driven:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[User-driven design analysis: blank canvas philosophy, choice architecture, progressive disclosure, and cognitive efficiency patterns.]"
        })

    # ==========================================================================
    # 7. CONNECTION & TENSION VISUALIZATION
    # ==========================================================================
    visualization = [
        # Relationship display
        "How should connections between entities be visualized?",
        "What visual encodings work for relationship strength?",
        "How can tensions be distinguished from alignments visually?",
        "What makes network graphs effective for showing connections?",
        "How should bidirectional relationships be displayed?",
        "What visual patterns indicate causal vs. correlational links?",

        # User-controlled views
        "How can users filter which connections to display?",
        "What controls should users have over visualization density?",
        "How should layers of connection types be toggled?",
        "What role do highlights play in focused exploration?",
        "How can zoom levels manage visual complexity?",
        "What interactions enable connection discovery?",

        # Tension mapping
        "How should competing forces be visualized?",
        "What visual metaphors work for tension representation?",
        "How can opposing relationships be distinguished?",
        "What color encodings communicate tension vs. harmony?",
        "How should conflicting data points be indicated?",
        "What visualizations show balance/imbalance dynamics?",
    ]

    for q in visualization:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Visualization analysis: relationship encoding, tension representation, user-controlled views, and interaction patterns for connection discovery.]"
        })

    # ==========================================================================
    # 8. DROPDOWN & CHECKBOX SEMANTICS
    # ==========================================================================
    ui_components = [
        # Selection patterns
        "When should dropdowns vs. checkboxes vs. radio buttons be used?",
        "How do selection UI patterns affect user mental models?",
        "What makes multi-select interfaces efficient?",
        "How should hierarchical selections be structured?",
        "What role do smart defaults play in selection efficiency?",
        "How can typeahead/autocomplete reduce selection friction?",

        # Combinatorial exploration
        "How do checkbox combinations enable exploratory analysis?",
        "What visualization updates should follow checkbox changes?",
        "How should AND vs. OR logic be communicated in multi-select?",
        "What feedback indicates the effect of selection changes?",
        "How can selection state be communicated clearly?",
        "What undo/reset patterns support exploration?",

        # Information architecture
        "How should filter categories be organized?",
        "What grouping strategies work for large option sets?",
        "How can related options be clustered effectively?",
        "What progressive disclosure patterns work for complex filters?",
        "How should mutual exclusivity be communicated?",
        "What search within options patterns work?",
    ]

    for q in ui_components:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[UI component analysis: selection pattern semantics, combinatorial exploration, and information architecture for efficient filtering.]"
        })

    # ==========================================================================
    # 9. DATA PRESENTATION STRATEGY
    # ==========================================================================
    data_presentation = [
        # Minimalist philosophy
        "Why present choices before presenting data?",
        "How does minimal initial state reduce cognitive load?",
        "What makes on-demand data more valuable than pre-loaded?",
        "How can whitespace communicate available possibility?",
        "What is the value of showing structure before content?",
        "How do sparse initial views invite engagement?",

        # Progressive revelation
        "How should data complexity be revealed progressively?",
        "What triggers should reveal additional detail?",
        "How can drill-down patterns manage information depth?",
        "What role do expandable sections play?",
        "How should summary vs. detail be balanced?",
        "What accordion patterns work for dense information?",

        # Context-sensitive display
        "How should data display adapt to user selection?",
        "What relevance filtering should follow user choices?",
        "How can related data be surfaced based on context?",
        "What dynamic highlighting aids comprehension?",
        "How should non-selected data be de-emphasized?",
        "What fade/dim patterns communicate relevance?",
    ]

    for q in data_presentation:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Data presentation analysis: minimalist philosophy, progressive revelation, and context-sensitive display strategies.]"
        })

    # ==========================================================================
    # 10. ECOSYSTEM COMPRESSION
    # ==========================================================================
    ecosystem_compression = [
        # Conceptual compression
        "How can complex ecosystems be represented compactly?",
        "What abstraction layers enable ecosystem compression?",
        "How do type hierarchies compress entity representation?",
        "What role do archetypes play in ecosystem models?",
        "How can relationship patterns be parameterized?",
        "What compression emerges from recognizing isomorphisms?",

        # Data model efficiency
        "How should ecosystem data models minimize redundancy?",
        "What normalization strategies apply to knowledge graphs?",
        "How can inheritance reduce entity description size?",
        "What compression comes from shared property schemas?",
        "How do ontologies enable efficient ecosystem encoding?",
        "What graph compression techniques apply?",

        # Runtime efficiency
        "How can ecosystem queries be optimized?",
        "What indexing strategies enable efficient exploration?",
        "How should caching be structured for ecosystems?",
        "What lazy loading patterns apply to complex ecosystems?",
        "How can incremental computation reduce redundant work?",
        "What materialized views optimize ecosystem queries?",
    ]

    for q in ecosystem_compression:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Ecosystem compression analysis: conceptual abstraction, data model efficiency, and runtime optimization strategies.]"
        })

    # ==========================================================================
    # 11. META-META-ANALYTICAL THINKING
    # ==========================================================================
    meta_meta = [
        # Recursive reflection
        "What does it mean to think about thinking about analysis?",
        "How does meta-level awareness improve analytical quality?",
        "What patterns emerge from analyzing analytical methods?",
        "How can meta-cognition be systematized?",
        "What role does recursive self-improvement play?",
        "How do meta-frameworks subsume lower frameworks?",

        # Analytical compression
        "How can analytical patterns be compressed into reusable schemas?",
        "What meta-patterns recur across different analytical domains?",
        "How do analytical primitives combine into complex methods?",
        "What role do analytical templates play in efficiency?",
        "How can analytical workflows be parameterized?",
        "What compression emerges from recognizing analytical isomorphisms?",

        # Enhanced meta
        "What does 'enhanced' meta-analysis add beyond basic meta?",
        "How can meta-analytical tools be recursively improved?",
        "What feedback loops enhance meta-analytical capability?",
        "How does meta-meta thinking enable new analytical modes?",
        "What emerges from applying analysis to itself repeatedly?",
        "How can self-referential systems remain coherent?",
    ]

    for q in meta_meta:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Meta-meta analysis: recursive reflection patterns, analytical compression, enhanced meta-cognition, and self-referential coherence strategies.]"
        })

    # ==========================================================================
    # 12. PROJECT FOOTPRINT REDUCTION
    # ==========================================================================
    footprint = [
        # Computational efficiency
        "How can LLM-powered projects minimize token footprint?",
        "What architectural patterns reduce API costs?",
        "How should caching be designed for LLM applications?",
        "What preprocessing reduces required inference?",
        "How can local models reduce cloud token spending?",
        "What hybrid architectures optimize cost vs. capability?",

        # Storage efficiency
        "How should training data be compressed for storage?",
        "What deduplication strategies apply to text datasets?",
        "How can embeddings reduce storage requirements?",
        "What quantization applies to model artifacts?",
        "How should versioning be managed for efficiency?",
        "What garbage collection applies to ML projects?",

        # Development efficiency
        "How can development workflows minimize wasted tokens?",
        "What testing strategies reduce token usage during development?",
        "How should prompts be version-controlled efficiently?",
        "What evaluation strategies optimize token spending?",
        "How can CI/CD be designed for token efficiency?",
        "What monitoring identifies token waste?",
    ]

    for q in footprint:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Footprint reduction analysis: computational efficiency, storage optimization, and development workflow strategies for minimal resource usage.]"
        })

    # ==========================================================================
    # 13. CHOICE MAXIMIZATION PATTERNS
    # ==========================================================================
    choice_max = [
        # Presentation strategies
        "How can maximum choice be presented in minimum space?",
        "What visual hierarchies enable rapid choice scanning?",
        "How do categorization schemes multiply apparent choice?",
        "What interaction patterns reveal hidden choices?",
        "How can shortcuts expose power-user capabilities?",
        "What progressive disclosure maximizes accessible choices?",

        # Combinatorial design
        "How do independent filters multiply choice combinations?",
        "What orthogonal dimensions enable maximum exploration?",
        "How can composable components maximize choice space?",
        "What parameterization enables fine-grained control?",
        "How do templates with variables maximize flexibility?",
        "What role do wildcards play in choice expansion?",

        # User empowerment
        "How can users be empowered to create their own views?",
        "What customization enables user-specific choice sets?",
        "How can saved preferences reduce repeated selection?",
        "What sharing mechanisms enable choice set transfer?",
        "How do presets balance efficiency with choice?",
        "What role does undo play in enabling experimentation?",
    ]

    for q in choice_max:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Choice maximization analysis: presentation strategies, combinatorial design, and user empowerment patterns for expanded exploration capability.]"
        })

    # ==========================================================================
    # 14. BLANK MAP PHILOSOPHY
    # ==========================================================================
    blank_map = [
        # Initial state design
        "Why is a blank map more inviting than a pre-filled one?",
        "How does emptiness communicate possibility?",
        "What minimal scaffolding guides without constraining?",
        "How should the first interaction be designed?",
        "What makes blank states feel like opportunities?",
        "How do tooltips guide without overwhelming?",

        # User journey
        "How should the first selection change the view?",
        "What feedback confirms user agency?",
        "How does progressive filling create ownership?",
        "What milestones mark exploration progress?",
        "How should complexity ramp as users explore?",
        "What resets enable fresh exploration?",

        # Map metaphor
        "How does the map metaphor aid spatial reasoning?",
        "What zoom levels serve different exploration modes?",
        "How should territories and boundaries be indicated?",
        "What role do landmarks play in orientation?",
        "How can users mark their own territories?",
        "What legends enable map interpretation?",
    ]

    for q in blank_map:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Blank map philosophy: initial state design, user journey orchestration, and spatial metaphors for exploration-based interfaces.]"
        })

    # ==========================================================================
    # 15. FEATURE HIGHLIGHTING SYSTEMS
    # ==========================================================================
    highlighting = [
        # Visual emphasis
        "How should selected features be highlighted?",
        "What visual weight indicates importance?",
        "How can color encode feature categories?",
        "What animation draws attention effectively?",
        "How should related features be visually linked?",
        "What contrast levels communicate hierarchy?",

        # Interactive highlighting
        "How should hover states reveal connections?",
        "What click interactions toggle feature visibility?",
        "How can brushing and linking coordinate views?",
        "What keyboard shortcuts enable rapid highlighting?",
        "How should multi-select highlighting work?",
        "What reset patterns clear highlighting?",

        # Semantic highlighting
        "How can highlighting encode semantic relationships?",
        "What visual encodings show tension vs. alignment?",
        "How should causal chains be highlighted?",
        "What highlighting patterns show temporal relationships?",
        "How can uncertainty be visually communicated?",
        "What highlighting indicates confidence levels?",
    ]

    for q in highlighting:
        examples.append({
            "instruction": q,
            "input": "",
            "output": "[Feature highlighting analysis: visual emphasis patterns, interactive mechanics, and semantic encoding strategies for connection revelation.]"
        })

    return examples


def main():
    print("=" * 70)
    print("META-COMPRESSION & USER-DRIVEN EXPLORATION TRAINING DATA")
    print("=" * 70)

    examples = generate_examples()
    print(f"\nGenerated {len(examples)} training examples")

    categories = {
        "meta_language": 0,
        "meta_grammar": 0,
        "token_economy": 0,
        "user_driven": 0,
        "visualization": 0,
        "ui_components": 0,
        "data_presentation": 0,
        "ecosystem": 0,
        "meta_meta": 0,
        "footprint": 0,
        "choice": 0,
        "blank_map": 0,
        "highlighting": 0,
        "other": 0,
    }

    for ex in examples:
        text = ex["instruction"].lower()
        if any(x in text for x in ["semantic", "linguistic", "language", "presupposition"]):
            categories["meta_language"] += 1
        elif any(x in text for x in ["grammar", "syntactic", "nominalization", "sentence"]):
            categories["meta_grammar"] += 1
        elif any(x in text for x in ["token", "api cost", "prompt", "cache"]):
            categories["token_economy"] += 1
        elif any(x in text for x in ["user-driven", "blank", "canvas", "progressive disclosure"]):
            categories["user_driven"] += 1
        elif any(x in text for x in ["visual", "graph", "network", "encoding"]):
            categories["visualization"] += 1
        elif any(x in text for x in ["dropdown", "checkbox", "select", "filter"]):
            categories["ui_components"] += 1
        elif any(x in text for x in ["present", "reveal", "display", "whitespace"]):
            categories["data_presentation"] += 1
        elif any(x in text for x in ["ecosystem", "ontolog", "knowledge graph"]):
            categories["ecosystem"] += 1
        elif any(x in text for x in ["meta-meta", "recursive", "self-referential"]):
            categories["meta_meta"] += 1
        elif any(x in text for x in ["footprint", "storage", "compress"]):
            categories["footprint"] += 1
        elif any(x in text for x in ["choice", "option", "combination"]):
            categories["choice"] += 1
        elif any(x in text for x in ["map", "territory", "zoom", "landmark"]):
            categories["blank_map"] += 1
        elif any(x in text for x in ["highlight", "emphasis", "color", "hover"]):
            categories["highlighting"] += 1
        else:
            categories["other"] += 1

    print("\n" + "=" * 70)
    print("CATEGORY BREAKDOWN:")
    print("=" * 70)
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} examples")

    output_path = "meta_compression_training.json"
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"\nSaved to {output_path}")

    return examples


if __name__ == "__main__":
    main()
