/**
 * LatticeForge Knowledge Base
 *
 * Built-in documentation, lexicon, and methodology reference.
 * Written in plain English for humans, not robots.
 *
 * Design philosophy:
 * - Every term is explained like you're explaining to a smart friend
 * - Examples > abstract definitions
 * - "Why" matters as much as "what"
 * - If a 5-year-old can't understand the summary, rewrite it
 */

// ============================================
// Types
// ============================================

export interface KnowledgeArticle {
  id: string;
  slug: string;
  title: string;
  summary: string; // One sentence, plain English
  category: KnowledgeCategory;
  tags: string[];
  content: ArticleSection[];
  relatedArticles: string[];
  externalLinks?: { label: string; url: string }[];
  lastUpdated: string;
  readTimeMinutes: number;
}

export interface ArticleSection {
  heading?: string;
  content: string; // Markdown supported
  type: 'text' | 'example' | 'warning' | 'tip' | 'definition';
}

export type KnowledgeCategory =
  | 'getting-started'
  | 'concepts'
  | 'analysis'
  | 'visualization'
  | 'filtering'
  | 'shortcuts'
  | 'advanced'
  | 'glossary';

export interface LexiconEntry {
  term: string;
  definition: string;
  etymology?: string; // Where the term comes from
  context: string; // When/why you'd encounter this
  example?: string;
  relatedTerms: string[];
  category: 'core' | 'analysis' | 'visualization' | 'technical' | 'tradecraft';
}

// ============================================
// Lexicon - Our Language, Explained
// ============================================

export const LEXICON: LexiconEntry[] = [
  // === Core Concepts ===
  {
    term: 'Intel',
    definition: 'Analyzed information about entities, events, or situations. Not raw data - intel has been processed, verified (to some degree), and contextualized.',
    etymology: 'Short for "intelligence" in the sense of gathered information, not cognitive ability. From Latin "intelligere" (to understand).',
    context: 'The main currency of LatticeForge. Everything you see is intel - reports, assessments, entity profiles.',
    example: 'A news report says "Protests in Country X." Intel says "Protests in Country X likely to escalate due to economic conditions, historical patterns, and regime response."',
    relatedTerms: ['OSINT', 'Raw Data', 'Assessment'],
    category: 'core',
  },
  {
    term: 'Entity',
    definition: 'Any "thing" that acts or can be acted upon. Countries, organizations, people, military units, facilities, even abstract concepts like alliances.',
    etymology: 'From Latin "ens" (being). In our context, anything with enough significance to track.',
    context: 'Entities are the nouns of intelligence. Events happen TO or BECAUSE OF entities.',
    example: 'Russia (country), Wagner Group (organization), Vladimir Putin (person), Kaliningrad Naval Base (facility).',
    relatedTerms: ['Actor', 'Target', 'Subject'],
    category: 'core',
  },
  {
    term: 'Relation',
    definition: 'A connection between two entities. Can be positive (alliance), negative (hostility), neutral (trade), or complex (all of the above).',
    etymology: 'From Latin "relatio" (a bringing back, a report).',
    context: 'Relations are the verbs connecting entities. They change over time and have strength/intensity.',
    example: 'US-UK (strong alliance), Russia-Ukraine (hostile), China-Russia (complex - economic partners, strategic alignment, historical distrust).',
    relatedTerms: ['Alliance', 'Hostility', 'Dependency'],
    category: 'core',
  },
  {
    term: 'Assessment',
    definition: 'Our analytical judgment about a situation. Includes confidence level (how sure we are) and sourcing (why we think this).',
    etymology: 'From Latin "assidere" (to sit beside as a judge).',
    context: 'Assessments are opinions, not facts. They represent our best understanding given available information. Always note confidence level.',
    example: 'Assessment: Conflict likely to escalate within 30 days (HIGH confidence, based on troop movements, diplomatic breakdown, historical patterns).',
    relatedTerms: ['Confidence', 'Sourcing', 'Estimate'],
    category: 'core',
  },
  {
    term: 'Indicator',
    definition: 'An observable sign that something might happen or be happening. Early warning signal.',
    etymology: 'From Latin "indicare" (to point out).',
    context: 'We watch indicators to anticipate events. Multiple indicators pointing the same direction = higher confidence.',
    example: 'Indicators of imminent military action: troop buildups, evacuation of civilians, diplomatic expulsions, communication blackouts.',
    relatedTerms: ['Warning', 'Signal', 'Precursor'],
    category: 'core',
  },

  // === Analysis Terms ===
  {
    term: 'OSINT',
    definition: 'Open Source Intelligence. Information gathered from publicly available sources - news, social media, satellite imagery, academic papers, government reports.',
    etymology: 'Military/intelligence acronym. "Open" as opposed to classified/secret sources.',
    context: 'Most of what we use is OSINT. It\'s not "lesser" intelligence - often the best insights come from connecting public dots.',
    example: 'Ship tracking data (AIS), flight radar, social media posts from conflict zones, company filings, press releases.',
    relatedTerms: ['Intel', 'Sourcing', 'Collection'],
    category: 'analysis',
  },
  {
    term: 'Confidence Level',
    definition: 'How sure we are about an assessment. LOW (could easily be wrong), MODERATE (reasonable basis), HIGH (strong evidence).',
    etymology: 'From Latin "confidere" (to trust fully).',
    context: 'Always stated with assessments. Low confidence doesn\'t mean worthless - sometimes a low-confidence heads-up is better than nothing.',
    example: 'HIGH: Based on multiple independent sources, historical pattern matching, and direct observation. LOW: Single source, limited corroboration, unusual claim.',
    relatedTerms: ['Assessment', 'Uncertainty', 'Probability'],
    category: 'analysis',
  },
  {
    term: 'Temporal Analysis',
    definition: 'Looking at how things change over time. Past patterns, current state, projected futures.',
    etymology: 'From Latin "temporalis" (of time).',
    context: 'The 4th dimension of our analysis. Everything exists in time - relationships evolve, threats rise and fall, patterns repeat.',
    example: 'US-China relations in 2000 vs 2010 vs 2020 vs projected 2030. Very different pictures.',
    relatedTerms: ['Projection', 'Historical', 'Trend'],
    category: 'analysis',
  },
  {
    term: 'Projection',
    definition: 'Our estimate of future states based on current trajectories, historical patterns, and analytical models.',
    etymology: 'From Latin "proicere" (to throw forward).',
    context: 'Projections are NOT predictions. They\'re "if current trends continue" scenarios. The future is inherently uncertain.',
    example: 'Projection: At current rate of resource depletion and population growth, Country X faces critical water shortage by 2030.',
    relatedTerms: ['Forecast', 'Scenario', 'Trajectory'],
    category: 'analysis',
  },

  // === Visualization Terms ===
  {
    term: 'Threat Matrix',
    definition: 'Visual representation of threats plotted by probability (X-axis) and severity (Y-axis). Size indicates impact.',
    etymology: 'Matrix from Latin "mater" (mother, source). Threat matrix = mother of all threat displays.',
    context: 'Quick visual prioritization tool. Top-right quadrant = high probability, high severity = immediate concern.',
    example: 'A pandemic risk might be low probability but high severity (upper left). Daily protests might be high probability but low severity (lower right).',
    relatedTerms: ['Risk Assessment', 'Prioritization', 'Impact'],
    category: 'visualization',
  },
  {
    term: 'Relation Graph',
    definition: 'Network visualization showing how entities connect. Nodes = entities, edges = relationships. Color and thickness encode relationship type/strength.',
    etymology: 'Graph in the mathematical sense (nodes and edges), not the charting sense.',
    context: 'Reveals patterns invisible in text: clusters, bridges, isolated actors, unexpected connections.',
    example: 'Two countries with no direct relationship might both connect strongly to a third - potential mediation opportunity or point of leverage.',
    relatedTerms: ['Network', 'Entity', 'Relation'],
    category: 'visualization',
  },
  {
    term: '3D Tree Navigator',
    definition: 'Our spatial-temporal visualization. Intel items positioned in 3D space with time as vertical axis. Related items cluster together.',
    etymology: 'Inspired by actual trees - branching structure, roots (historical) to canopy (future).',
    context: 'Navigate intel intuitively. See how events connect across time and topic. Zoom, rotate, click to explore.',
    example: 'Looking at a conflict: root events (historical grievances) at bottom, current situation at center, projected scenarios at top.',
    relatedTerms: ['Temporal Analysis', 'Visualization', 'Navigation'],
    category: 'visualization',
  },

  // === Technical/System Terms ===
  {
    term: 'XYZA Metrics',
    definition: 'Our analysis confidence framework. X=Coherence (internal consistency), Y=Complexity (nuance captured), Z=Reflection (analytical depth), A=Attunement (contextual relevance).',
    etymology: 'Extends XYZ (spatial coordinates) with A (attunement) for a 4th dimension.',
    context: 'Shows how confident the system is in its analysis. High scores = more reliable. Low scores = take with grain of salt.',
    example: 'A rushed analysis might have low Z (reflection). An analysis of unfamiliar territory might have low A (attunement).',
    relatedTerms: ['Confidence', 'Cognitive Metrics', 'Flow State'],
    category: 'technical',
  },
  {
    term: 'Flow State',
    definition: 'System performance indicator based on Kuramoto synchronization. High flow = system operating optimally. Low flow = potential issues.',
    etymology: 'From psychology (Csikszentmihalyi\'s flow) + physics (Kuramoto model of coupled oscillators).',
    context: 'Background metric. If flow is low, analysis might be less reliable. Usually self-corrects.',
    example: 'During high-load periods or with conflicting data, flow might dip. Green = good, yellow = watch, red = degraded.',
    relatedTerms: ['XYZA', 'System Status', 'Reliability'],
    category: 'technical',
  },
  {
    term: 'Widget',
    definition: 'A self-contained component on your dashboard. Each widget shows specific information or provides specific functionality.',
    etymology: 'Origin disputed. Possibly from "gadget" or fictional "widget" in economics examples.',
    context: 'Build your dashboard by adding, removing, and arranging widgets. Each widget is independently configurable.',
    example: 'Threat Matrix widget, News Feed widget, Relation Graph widget, Notes widget.',
    relatedTerms: ['Dashboard', 'Configuration', 'Component'],
    category: 'technical',
  },

  // === Tradecraft Terms ===
  {
    term: 'AO',
    definition: 'Area of Operations. The geographic or conceptual space you\'re focused on.',
    etymology: 'Military abbreviation.',
    context: 'Define your AO to filter and focus analysis. Can be a country, region, or topic area.',
    example: 'A Middle East analyst\'s AO might include the Levant, Gulf states, and North Africa.',
    relatedTerms: ['Region', 'Scope', 'Focus'],
    category: 'tradecraft',
  },
  {
    term: 'BLUF',
    definition: 'Bottom Line Up Front. Lead with the conclusion, then provide supporting details.',
    etymology: 'Military communication practice.',
    context: 'How our executive summaries are structured. Busy readers get the point immediately, can read details if needed.',
    example: 'BLUF: Conflict likely to escalate. [Then details follow]',
    relatedTerms: ['Executive Summary', 'Assessment', 'Brief'],
    category: 'tradecraft',
  },
  {
    term: 'Collection',
    definition: 'Gathering raw information from sources. First step before analysis.',
    etymology: 'From Latin "colligere" (to gather together).',
    context: 'We collect from OSINT sources. More collection = more data for analysis, but also more noise to filter.',
    example: 'Collecting news articles, social media posts, government statements, satellite imagery about a developing situation.',
    relatedTerms: ['OSINT', 'Sources', 'Raw Data'],
    category: 'tradecraft',
  },
  {
    term: 'Sanitized',
    definition: 'Information with sensitive details removed or obscured to protect sources or methods.',
    etymology: 'From "sanitary" - cleaned.',
    context: 'Some information is sanitized before presentation. You might see "[source]" instead of specific source identification.',
    example: 'Original: "CIA operative in Moscow reports..." Sanitized: "A credible source reports..."',
    relatedTerms: ['Sourcing', 'Classification', 'Redaction'],
    category: 'tradecraft',
  },
];

// ============================================
// Knowledge Articles
// ============================================

export const KNOWLEDGE_ARTICLES: KnowledgeArticle[] = [
  // === Getting Started ===
  {
    id: 'gs-what-is-latticeforge',
    slug: 'what-is-latticeforge',
    title: 'What is LatticeForge?',
    summary: 'LatticeForge helps you understand what\'s happening in the world and why it matters.',
    category: 'getting-started',
    tags: ['introduction', 'overview'],
    content: [
      {
        type: 'text',
        content: `
LatticeForge is an intelligence analysis platform. In plain English: we help you make sense of complex global situations.

**Think of it like this:** The world generates enormous amounts of information every day. News articles, government statements, economic data, social media posts, satellite imagery. Most of it is noise. LatticeForge helps you find the signal.

We collect information from public sources (OSINT), analyze it using advanced algorithms and human expertise, and present it in ways that help you understand:

- **What's happening** - Current events and situations
- **Why it's happening** - Historical context and causal factors
- **Who's involved** - Entities, relationships, and motivations
- **What might happen next** - Projections and scenarios
        `,
      },
      {
        type: 'tip',
        heading: 'Who uses this?',
        content: 'Journalists, researchers, business intelligence teams, policy analysts, government agencies, and anyone who needs to understand complex situations beyond what headlines tell you.',
      },
    ],
    relatedArticles: ['gs-first-steps', 'gs-dashboard-basics'],
    lastUpdated: '2024-01-15',
    readTimeMinutes: 3,
  },
  {
    id: 'gs-first-steps',
    slug: 'first-steps',
    title: 'Your First 5 Minutes',
    summary: 'Quick guide to get oriented and start exploring.',
    category: 'getting-started',
    tags: ['tutorial', 'basics', 'onboarding'],
    content: [
      {
        type: 'text',
        content: `
## Start Here

1. **Check your Daily Brief** - The Executive Summary at the top shows today's most important developments. Read this first.

2. **Explore the Map** - Click regions to see what's happening there. Red markers = active situations.

3. **Try a Search** - Press \`/\` and type anything: a country name, a topic, a person. See what comes up.

4. **Click Something** - Everything is clickable. News items, entities, chart elements. Click to learn more.

5. **Come Back Tomorrow** - Intelligence analysis is about patterns over time. Check back daily to build understanding.
        `,
      },
      {
        type: 'tip',
        heading: 'Keyboard Shortcut',
        content: 'Press `?` anytime to see all available keyboard shortcuts.',
      },
      {
        type: 'warning',
        heading: 'Don\'t get overwhelmed',
        content: 'You don\'t need to understand everything immediately. Start with one region or topic you care about. Expand from there.',
      },
    ],
    relatedArticles: ['gs-what-is-latticeforge', 'gs-dashboard-basics', 'shortcuts-basics'],
    lastUpdated: '2024-01-15',
    readTimeMinutes: 2,
  },

  // === Analysis Methodology ===
  {
    id: 'analysis-methodology',
    slug: 'how-we-analyze',
    title: 'How We Analyze Information',
    summary: 'Our methodology for turning raw data into useful intelligence.',
    category: 'analysis',
    tags: ['methodology', 'process', 'transparency'],
    content: [
      {
        type: 'text',
        content: `
## The Intelligence Cycle

We follow a structured process to ensure quality and consistency:

### 1. Collection
We gather information from thousands of public sources: news outlets, government publications, academic research, social media, satellite imagery, economic data, and more.

**Key principle:** More sources = better corroboration, but also more noise. Quality matters more than quantity.

### 2. Processing
Raw information gets cleaned, translated (if needed), and structured. We extract entities (people, places, organizations), events, and relationships.

**Key principle:** Garbage in, garbage out. We invest heavily in data quality.

### 3. Analysis
This is where the magic happens. We combine algorithmic pattern recognition with analytical frameworks to understand:
- What does this information mean?
- How does it connect to other information?
- What's the significance?
- How confident are we?

**Key principle:** Always state confidence levels. Uncertainty is information too.

### 4. Dissemination
Analyzed intelligence is presented through dashboards, reports, alerts, and visualizations designed for your specific needs.

**Key principle:** The best analysis is useless if it doesn't reach the right person at the right time.

### 5. Feedback
Your interactions help us improve. What did you find useful? What was noise? This feeds back into our collection and analysis priorities.
        `,
      },
      {
        type: 'definition',
        heading: 'Why transparency matters',
        content: 'We tell you HOW we reached conclusions, not just WHAT we concluded. You can evaluate our reasoning and adjust your trust accordingly.',
      },
    ],
    relatedArticles: ['analysis-confidence', 'analysis-sourcing', 'analysis-bias'],
    lastUpdated: '2024-01-15',
    readTimeMinutes: 5,
  },
  {
    id: 'analysis-confidence',
    slug: 'understanding-confidence',
    title: 'Understanding Confidence Levels',
    summary: 'What HIGH, MODERATE, and LOW confidence actually mean.',
    category: 'analysis',
    tags: ['confidence', 'uncertainty', 'methodology'],
    content: [
      {
        type: 'text',
        content: `
## Confidence Is Not Binary

Intelligence isn't about certainty - it's about reducing uncertainty. Our confidence levels tell you how much uncertainty remains.

### HIGH Confidence
**What it means:** Strong analytical basis. Multiple independent sources corroborate. Fits established patterns. Low likelihood of being significantly wrong.

**NOT what it means:** 100% certain, guaranteed to be correct, no possibility of error.

**When you see HIGH:** Reasonable to act on this assessment. Still hedge for unlikely alternatives.

### MODERATE Confidence
**What it means:** Reasonable analytical basis. Some corroboration. Plausible interpretation but alternatives exist.

**NOT what it means:** Mediocre analysis, take it or leave it.

**When you see MODERATE:** Consider this the working hypothesis. Watch for confirming or disconfirming indicators.

### LOW Confidence
**What it means:** Limited information, weak corroboration, or unusual claim. Best current understanding but could easily change with new information.

**NOT what it means:** Probably wrong, ignore this.

**When you see LOW:** Valuable as an early indicator or hypothesis. Don't plan around it, but don't dismiss it either.
        `,
      },
      {
        type: 'example',
        heading: 'Real-world example',
        content: `
**Scenario:** Satellite imagery shows military equipment moving toward a border.

- **HIGH confidence:** "Military equipment is moving toward the border" (we can literally see it)
- **MODERATE confidence:** "This likely indicates preparation for exercises or potential military action" (fits patterns, but alternatives exist)
- **LOW confidence:** "This suggests invasion is imminent within 48 hours" (possible, but many unknowns about intent and timing)
        `,
      },
    ],
    relatedArticles: ['analysis-methodology', 'analysis-sourcing', 'concept-uncertainty'],
    lastUpdated: '2024-01-15',
    readTimeMinutes: 4,
  },
  {
    id: 'analysis-sourcing',
    slug: 'sourcing-and-verification',
    title: 'Sourcing and Verification',
    summary: 'How we evaluate and attribute information sources.',
    category: 'analysis',
    tags: ['sources', 'verification', 'osint', 'methodology'],
    content: [
      {
        type: 'text',
        content: `
## Not All Sources Are Equal

A claim from Reuters is different from an anonymous Twitter account. We evaluate sources on several dimensions:

### Source Evaluation Criteria

**Reliability:** Has this source been accurate in the past? Do they have access to the information they claim?

**Independence:** Is this source connected to parties who benefit from this narrative? Are they repeating someone else or providing original information?

**Corroboration:** Do other independent sources report the same thing? Does it fit with other verified information?

**Plausibility:** Does this make sense given what we know? Extraordinary claims require extraordinary evidence.

### Attribution

When we cite sources, we try to be specific:
- "According to Reuters..." (named, reliable source)
- "Multiple credible sources report..." (corroborated, sources protected)
- "Unverified reports suggest..." (information available but not confirmed)
- "A single source claims..." (minimal corroboration, use caution)
        `,
      },
      {
        type: 'warning',
        heading: 'On social media',
        content: 'Social media can be valuable OSINT but requires extra scrutiny. Viral doesn\'t mean verified. We cross-reference social media claims against other sources before treating them as reliable.',
      },
    ],
    relatedArticles: ['analysis-confidence', 'analysis-bias', 'glossary-osint'],
    lastUpdated: '2024-01-15',
    readTimeMinutes: 4,
  },
  {
    id: 'analysis-bias',
    slug: 'analytical-biases',
    title: 'Recognizing Analytical Biases',
    summary: 'Common mental traps and how we try to avoid them.',
    category: 'analysis',
    tags: ['bias', 'methodology', 'critical-thinking'],
    content: [
      {
        type: 'text',
        content: `
## We're All Human

Even the best analysts fall into cognitive traps. Awareness is the first defense.

### Confirmation Bias
**The trap:** Seeking information that confirms what we already believe, ignoring contradictory evidence.

**Our defense:** Actively seek disconfirming evidence. Devil's advocate reviews. Red team exercises.

### Availability Bias
**The trap:** Overweighting recent or memorable events. "This is just like last time" when it's actually different.

**Our defense:** Systematic historical analysis. Base rate comparisons. Don't just remember - measure.

### Mirror Imaging
**The trap:** Assuming others think like we do. "They wouldn't do that because we wouldn't."

**Our defense:** Cultural expertise. Alternative analysis. Challenge assumptions about rationality.

### Anchoring
**The trap:** Over-relying on first information received. Hard to update when new info contradicts initial understanding.

**Our defense:** Explicit confidence calibration. Bayesian updating. Regular reassessment.

### Groupthink
**The trap:** Conforming to team consensus. Dissent feels risky.

**Our defense:** Structured disagreement. Anonymous input options. Reward contrarian views that prove valuable.
        `,
      },
      {
        type: 'tip',
        heading: 'You have biases too',
        content: 'Be aware of your own biases when consuming analysis. Do you want this assessment to be true? That\'s a signal to scrutinize it more carefully.',
      },
    ],
    relatedArticles: ['analysis-methodology', 'analysis-confidence', 'concept-critical-thinking'],
    lastUpdated: '2024-01-15',
    readTimeMinutes: 5,
  },

  // === Visualization Guides ===
  {
    id: 'viz-threat-matrix',
    slug: 'reading-threat-matrix',
    title: 'Reading the Threat Matrix',
    summary: 'How to interpret our threat visualization.',
    category: 'visualization',
    tags: ['threat-matrix', 'visualization', 'risk'],
    content: [
      {
        type: 'text',
        content: `
## The Threat Matrix at a Glance

The threat matrix is a prioritization tool. It helps you see which threats deserve immediate attention vs. ongoing monitoring vs. background awareness.

### The Axes

**X-Axis (Probability):** Left = unlikely to happen. Right = likely to happen.

**Y-Axis (Severity):** Bottom = low impact if it occurs. Top = high impact if it occurs.

**Circle Size:** Represents potential scale of impact (affected population, economic cost, geographic scope).

### The Quadrants

**Top-Right (High probability, High severity):** Immediate concerns. These should be actively managed and monitored.

**Top-Left (Low probability, High severity):** Black swan territory. Unlikely but catastrophic if they occur. Worth contingency planning.

**Bottom-Right (High probability, Low severity):** Likely annoyances. Monitor but don't overreact.

**Bottom-Left (Low probability, Low severity):** Background noise. Periodic check-ins sufficient.
        `,
      },
      {
        type: 'example',
        heading: 'Example interpretation',
        content: `
A threat in the top-right moving toward the center might indicate:
- Situation is de-escalating (good news)
- Our assessment of probability or severity has changed based on new information
- Time-based factors have shifted (e.g., an election passed without incident)
        `,
      },
    ],
    relatedArticles: ['viz-relation-graph', 'viz-3d-tree', 'analysis-confidence'],
    lastUpdated: '2024-01-15',
    readTimeMinutes: 4,
  },
];

// ============================================
// Search and Retrieval
// ============================================

/**
 * Search knowledge base
 */
export function searchKnowledge(query: string): {
  articles: KnowledgeArticle[];
  lexicon: LexiconEntry[];
} {
  const q = query.toLowerCase();

  const articles = KNOWLEDGE_ARTICLES.filter(
    (article) =>
      article.title.toLowerCase().includes(q) ||
      article.summary.toLowerCase().includes(q) ||
      article.tags.some((tag) => tag.toLowerCase().includes(q)) ||
      article.content.some((section) => section.content.toLowerCase().includes(q))
  );

  const lexicon = LEXICON.filter(
    (entry) =>
      entry.term.toLowerCase().includes(q) ||
      entry.definition.toLowerCase().includes(q) ||
      entry.relatedTerms.some((term) => term.toLowerCase().includes(q))
  );

  return { articles, lexicon };
}

/**
 * Get article by slug
 */
export function getArticle(slug: string): KnowledgeArticle | undefined {
  return KNOWLEDGE_ARTICLES.find((a) => a.slug === slug);
}

/**
 * Get lexicon entry
 */
export function getLexiconEntry(term: string): LexiconEntry | undefined {
  return LEXICON.find((e) => e.term.toLowerCase() === term.toLowerCase());
}

/**
 * Get articles by category
 */
export function getArticlesByCategory(category: KnowledgeCategory): KnowledgeArticle[] {
  return KNOWLEDGE_ARTICLES.filter((a) => a.category === category);
}

/**
 * Get related articles
 */
export function getRelatedArticles(articleId: string): KnowledgeArticle[] {
  const article = KNOWLEDGE_ARTICLES.find((a) => a.id === articleId);
  if (!article) return [];

  return article.relatedArticles
    .map((id) => KNOWLEDGE_ARTICLES.find((a) => a.id === id))
    .filter((a): a is KnowledgeArticle => a !== undefined);
}

/**
 * Get all categories with counts
 */
export function getCategories(): { category: KnowledgeCategory; count: number; label: string }[] {
  const labels: Record<KnowledgeCategory, string> = {
    'getting-started': 'Getting Started',
    concepts: 'Core Concepts',
    analysis: 'Analysis Methods',
    visualization: 'Visualizations',
    filtering: 'Filtering & Search',
    shortcuts: 'Keyboard Shortcuts',
    advanced: 'Advanced Features',
    glossary: 'Glossary',
  };

  const counts: Record<string, number> = {};
  KNOWLEDGE_ARTICLES.forEach((a) => {
    counts[a.category] = (counts[a.category] || 0) + 1;
  });

  return Object.entries(labels).map(([category, label]) => ({
    category: category as KnowledgeCategory,
    count: counts[category] || 0,
    label,
  }));
}

/**
 * Get lexicon by category
 */
export function getLexiconByCategory(category: LexiconEntry['category']): LexiconEntry[] {
  return LEXICON.filter((e) => e.category === category);
}
