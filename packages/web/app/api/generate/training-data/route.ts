import { NextResponse } from 'next/server';
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import Anthropic from '@anthropic-ai/sdk';

// Lazy initialization to avoid build-time errors
let supabase: SupabaseClient | null = null;
let anthropic: Anthropic | null = null;

function getSupabase() {
  if (!supabase) {
    supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );
  }
  return supabase;
}

function getAnthropic() {
  if (!anthropic) {
    anthropic = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY!,
    });
  }
  return anthropic;
}

// RSS feeds by domain - comprehensive multi-sector coverage
const RSS_SOURCES: Record<string, { url: string; domain: string }[]> = {
  // Geopolitical & World Affairs
  geopolitical: [
    { url: 'https://feeds.reuters.com/Reuters/worldNews', domain: 'geopolitical' },
    { url: 'https://feeds.npr.org/1004/rss.xml', domain: 'geopolitical' },
    { url: 'https://rss.nytimes.com/services/xml/rss/nyt/World.xml', domain: 'geopolitical' },
  ],
  // Financial & Economic
  financial: [
    { url: 'https://feeds.reuters.com/reuters/businessNews', domain: 'financial' },
    { url: 'https://feeds.bloomberg.com/markets/news.rss', domain: 'financial' },
    { url: 'https://www.cnbc.com/id/100003114/device/rss/rss.html', domain: 'financial' },
  ],
  // Cybersecurity
  cyber: [
    { url: 'https://www.cisa.gov/news.xml', domain: 'cyber' },
    { url: 'https://krebsonsecurity.com/feed/', domain: 'cyber' },
    { url: 'https://feeds.feedburner.com/TheHackersNews', domain: 'cyber' },
    { url: 'https://www.darkreading.com/rss.xml', domain: 'cyber' },
  ],
  // Defense & Military
  defense: [
    { url: 'https://www.defensenews.com/arc/outboundfeeds/rss/?outputType=xml', domain: 'defense' },
    { url: 'https://breakingdefense.com/feed/', domain: 'defense' },
    { url: 'https://www.janes.com/feeds/news', domain: 'defense' },
  ],
  // Energy & Petrochemical
  energy: [
    { url: 'https://www.eia.gov/rss/todayinenergy.xml', domain: 'energy' },
    { url: 'https://oilprice.com/rss/main', domain: 'energy' },
    { url: 'https://www.rigzone.com/news/rss/rigzone_latest.aspx', domain: 'energy' },
  ],
  // Healthcare & Pharma
  health: [
    { url: 'https://tools.cdc.gov/api/v2/resources/media/rss', domain: 'health' },
    { url: 'https://www.who.int/rss-feeds/news-english.xml', domain: 'health' },
    { url: 'https://www.fiercepharma.com/rss/xml', domain: 'pharma' },
    { url: 'https://www.fiercebiotech.com/rss/xml', domain: 'biotech' },
  ],
  // Biotech & Cancer Research
  biotech: [
    { url: 'https://www.nature.com/nbt.rss', domain: 'biotech' },
    { url: 'https://www.cancer.gov/news-events/cancer-currents-blog/rss', domain: 'cancer_research' },
    { url: 'https://www.statnews.com/feed/', domain: 'biotech' },
  ],
  // Technology & AI
  tech: [
    { url: 'https://feeds.arstechnica.com/arstechnica/technology-lab', domain: 'tech' },
    { url: 'https://techcrunch.com/feed/', domain: 'tech' },
    { url: 'https://www.wired.com/feed/rss', domain: 'tech' },
    { url: 'https://news.ycombinator.com/rss', domain: 'tech' },
  ],
  // Quantum & Fusion
  quantum_fusion: [
    { url: 'https://thequantuminsider.com/feed/', domain: 'quantum' },
    { url: 'https://www.nextbigfuture.com/feed', domain: 'fusion' },
    { url: 'https://physicsworld.com/feed/', domain: 'quantum' },
  ],
  // Space & Aerospace
  space: [
    { url: 'https://spacenews.com/feed/', domain: 'space' },
    { url: 'https://www.nasa.gov/rss/dyn/breaking_news.rss', domain: 'space' },
    { url: 'https://feeds.arstechnica.com/arstechnica/science', domain: 'space' },
  ],
  // Telecom
  telecom: [
    { url: 'https://www.fiercewireless.com/rss/xml', domain: 'telecom' },
    { url: 'https://www.lightreading.com/rss.xml', domain: 'telecom' },
  ],
  // Automotive & EV
  automotive: [
    { url: 'https://www.autonews.com/rss.xml', domain: 'automotive' },
    { url: 'https://electrek.co/feed/', domain: 'automotive' },
    { url: 'https://insideevs.com/rss/news/all/', domain: 'automotive' },
  ],
  // Manufacturing & Industrial
  manufacturing: [
    { url: 'https://www.industryweek.com/rss.xml', domain: 'manufacturing' },
    { url: 'https://www.supplychaindive.com/feeds/news/', domain: 'supply_chain' },
  ],
  // Climate & Environment
  climate: [
    { url: 'https://www.noaa.gov/rss.xml', domain: 'climate' },
    { url: 'https://www.epa.gov/rss/epa-news.xml', domain: 'climate' },
    { url: 'https://climate.nasa.gov/rss/news', domain: 'climate' },
  ],
  // Employment & Labor
  employment: [
    { url: 'https://www.bls.gov/feed/bls_latest.rss', domain: 'employment' },
    { url: 'https://www.shrm.org/rss/pages/rss.aspx', domain: 'employment' },
  ],
  // Agriculture & Food Security
  agriculture: [
    { url: 'https://www.usda.gov/rss/latest-releases.xml', domain: 'agriculture' },
    { url: 'https://www.fao.org/news/rss-feed/en/', domain: 'agriculture' },
  ],
  // Neurotech & Brain-Computer Interface
  neurotech: [
    { url: 'https://www.neurotechreports.com/rss.xml', domain: 'neurotech' },
    { url: 'https://www.medgadget.com/feed', domain: 'neurotech' },
    { url: 'https://spectrum.ieee.org/feeds/topic/biomedical', domain: 'neurotech' },
  ],
  // Semiconductors & Chips
  semiconductors: [
    { url: 'https://www.eetimes.com/feed/', domain: 'semiconductors' },
    { url: 'https://semiengineering.com/feed/', domain: 'semiconductors' },
    { url: 'https://www.tomshardware.com/feeds/all', domain: 'semiconductors' },
  ],
  // AI & Machine Learning
  ai: [
    { url: 'https://venturebeat.com/category/ai/feed/', domain: 'ai' },
    { url: 'https://www.marktechpost.com/feed/', domain: 'ai' },
    { url: 'https://syncedreview.com/feed/', domain: 'ai' },
  ],
  // Robotics & Automation
  robotics: [
    { url: 'https://www.therobotreport.com/feed/', domain: 'robotics' },
    { url: 'https://spectrum.ieee.org/feeds/topic/robotics', domain: 'robotics' },
  ],
  // Materials Science & Nanotech
  materials: [
    { url: 'https://www.materialstoday.com/rss/', domain: 'materials' },
    { url: 'https://nanotechweb.org/cws/rss', domain: 'nanotech' },
  ],
  // Cryptocurrency & Blockchain
  crypto: [
    { url: 'https://cointelegraph.com/rss', domain: 'crypto' },
    { url: 'https://www.coindesk.com/arc/outboundfeeds/rss/', domain: 'crypto' },
  ],
};

interface NewsItem {
  title: string;
  description: string;
  link: string;
  pubDate: string;
  domain: string;
  source_type: string;
}

// Domain-specific analyst agents with specialized expertise
const DOMAIN_ANALYSTS: Record<string, {
  name: string;
  expertise: string;
  riskFactors: string[];
  cascadeTargets: string[];
}> = {
  // Geopolitical & Defense
  geopolitical: {
    name: 'Geopolitical Intelligence Analyst',
    expertise: 'international relations, territorial disputes, diplomatic tensions, sanctions regimes, alliance dynamics',
    riskFactors: ['military escalation', 'diplomatic breakdown', 'sanctions', 'regime change', 'territorial disputes'],
    cascadeTargets: ['financial markets', 'energy prices', 'supply chains', 'defense spending', 'refugee flows'],
  },
  defense: {
    name: 'Defense & Military Analyst',
    expertise: 'military capabilities, defense procurement, force postures, arms transfers, military exercises',
    riskFactors: ['arms race', 'force projection', 'nuclear posture', 'cyber warfare', 'space militarization'],
    cascadeTargets: ['geopolitical tensions', 'defense stocks', 'rare earth demand', 'tech transfer restrictions'],
  },

  // Economic & Financial
  financial: {
    name: 'Financial Markets Analyst',
    expertise: 'equity markets, fixed income, forex, derivatives, central bank policy, credit markets',
    riskFactors: ['liquidity crisis', 'credit spreads', 'yield curve', 'currency volatility', 'margin calls'],
    cascadeTargets: ['corporate solvency', 'pension funds', 'emerging markets', 'real estate', 'consumer credit'],
  },
  crypto: {
    name: 'Digital Assets & DeFi Analyst',
    expertise: 'cryptocurrencies, blockchain protocols, DeFi, stablecoins, regulatory frameworks',
    riskFactors: ['exchange insolvency', 'smart contract exploits', 'regulatory crackdown', 'stablecoin depegging'],
    cascadeTargets: ['traditional finance', 'tech stocks', 'payment systems', 'remittance markets'],
  },

  // Technology & Cyber
  cyber: {
    name: 'Cybersecurity Threat Analyst',
    expertise: 'APT groups, ransomware, zero-days, critical infrastructure, nation-state actors',
    riskFactors: ['ransomware surge', 'supply chain attacks', 'critical infrastructure compromise', 'data breaches'],
    cascadeTargets: ['insurance markets', 'healthcare operations', 'financial systems', 'industrial control'],
  },
  tech: {
    name: 'Technology & Innovation Analyst',
    expertise: 'semiconductor supply, cloud infrastructure, platform dynamics, tech regulation, AI development',
    riskFactors: ['chip shortage', 'platform instability', 'tech regulation', 'talent migration', 'IP disputes'],
    cascadeTargets: ['automotive production', 'consumer electronics', 'defense systems', 'AI capabilities'],
  },
  ai: {
    name: 'AI & Machine Learning Analyst',
    expertise: 'foundation models, AI governance, compute infrastructure, AI safety, autonomous systems',
    riskFactors: ['capability jumps', 'alignment failures', 'compute bottlenecks', 'regulatory fragmentation'],
    cascadeTargets: ['labor markets', 'content authenticity', 'cybersecurity', 'scientific research'],
  },
  semiconductors: {
    name: 'Semiconductor Industry Analyst',
    expertise: 'chip fabrication, EDA tools, packaging, foundry capacity, export controls',
    riskFactors: ['fab disruption', 'export restrictions', 'equipment bottlenecks', 'talent shortage'],
    cascadeTargets: ['automotive', 'consumer electronics', 'defense', 'AI compute', 'telecommunications'],
  },
  quantum: {
    name: 'Quantum & Emerging Tech Analyst',
    expertise: 'quantum computing, quantum sensing, post-quantum cryptography, fusion energy',
    riskFactors: ['cryptographic vulnerability', 'quantum supremacy claims', 'talent concentration'],
    cascadeTargets: ['cybersecurity', 'drug discovery', 'financial modeling', 'materials science'],
  },

  // Energy & Resources
  energy: {
    name: 'Energy Markets Analyst',
    expertise: 'oil & gas markets, renewables, grid infrastructure, energy security, OPEC dynamics',
    riskFactors: ['supply disruption', 'refinery outages', 'pipeline incidents', 'OPEC decisions', 'grid instability'],
    cascadeTargets: ['transportation costs', 'manufacturing', 'inflation', 'petrochemicals', 'food prices'],
  },
  climate: {
    name: 'Climate & Environmental Risk Analyst',
    expertise: 'extreme weather, climate policy, carbon markets, physical climate risks, transition risks',
    riskFactors: ['extreme weather', 'stranded assets', 'regulatory shifts', 'water stress', 'biodiversity loss'],
    cascadeTargets: ['insurance', 'agriculture', 'real estate', 'supply chains', 'migration patterns'],
  },

  // Healthcare & Biotech
  health: {
    name: 'Healthcare & Epidemiology Analyst',
    expertise: 'disease surveillance, health systems, pandemic preparedness, drug supply chains',
    riskFactors: ['outbreak emergence', 'health system strain', 'drug shortages', 'antimicrobial resistance'],
    cascadeTargets: ['labor markets', 'travel industry', 'pharmaceutical stocks', 'global trade'],
  },
  pharma: {
    name: 'Pharmaceutical Industry Analyst',
    expertise: 'drug development, clinical trials, FDA/EMA approvals, pricing dynamics, patent cliffs',
    riskFactors: ['trial failures', 'regulatory rejection', 'pricing pressure', 'patent expiry', 'supply issues'],
    cascadeTargets: ['biotech valuations', 'healthcare costs', 'patient outcomes', 'insurance premiums'],
  },
  biotech: {
    name: 'Biotechnology & Life Sciences Analyst',
    expertise: 'gene therapy, CRISPR, synthetic biology, diagnostics, biomanufacturing',
    riskFactors: ['safety signals', 'regulatory hurdles', 'manufacturing failures', 'IP litigation'],
    cascadeTargets: ['pharma partnerships', 'agricultural biotech', 'industrial biotech', 'healthcare costs'],
  },
  cancer_research: {
    name: 'Oncology Research Analyst',
    expertise: 'cancer therapeutics, immunotherapy, precision medicine, clinical outcomes',
    riskFactors: ['trial failures', 'resistance mechanisms', 'pricing access', 'competitive dynamics'],
    cascadeTargets: ['biotech valuations', 'healthcare budgets', 'patient access', 'research priorities'],
  },
  neurotech: {
    name: 'Neurotechnology Analyst',
    expertise: 'brain-computer interfaces, neurostimulation, neuroimaging, cognitive enhancement',
    riskFactors: ['safety concerns', 'regulatory uncertainty', 'ethical debates', 'data privacy'],
    cascadeTargets: ['disability services', 'mental health treatment', 'human augmentation debate'],
  },

  // Industrial & Manufacturing
  manufacturing: {
    name: 'Industrial & Manufacturing Analyst',
    expertise: 'factory automation, industrial IoT, lean manufacturing, reshoring trends',
    riskFactors: ['supply disruption', 'labor disputes', 'equipment failures', 'quality issues'],
    cascadeTargets: ['consumer goods availability', 'employment', 'trade balances', 'inflation'],
  },
  supply_chain: {
    name: 'Supply Chain & Logistics Analyst',
    expertise: 'global logistics, port operations, freight markets, inventory management',
    riskFactors: ['port congestion', 'shipping delays', 'container shortages', 'customs disruption'],
    cascadeTargets: ['retail inventory', 'manufacturing', 'commodity prices', 'inflation'],
  },
  automotive: {
    name: 'Automotive & EV Analyst',
    expertise: 'vehicle production, EV transition, battery technology, autonomous driving',
    riskFactors: ['chip shortage', 'battery supply', 'charging infrastructure', 'recall events'],
    cascadeTargets: ['lithium/cobalt prices', 'grid demand', 'oil demand', 'urban planning'],
  },
  robotics: {
    name: 'Robotics & Automation Analyst',
    expertise: 'industrial robots, warehouse automation, humanoid robotics, autonomous systems',
    riskFactors: ['labor displacement', 'safety incidents', 'capability limitations', 'cost barriers'],
    cascadeTargets: ['manufacturing employment', 'logistics efficiency', 'labor markets'],
  },
  materials: {
    name: 'Advanced Materials Analyst',
    expertise: 'nanomaterials, composites, rare earths, critical minerals, battery materials',
    riskFactors: ['supply concentration', 'processing bottlenecks', 'environmental regulations'],
    cascadeTargets: ['electronics manufacturing', 'defense systems', 'clean energy', 'construction'],
  },

  // Space & Telecom
  space: {
    name: 'Space & Aerospace Analyst',
    expertise: 'launch services, satellite constellations, space debris, lunar/Mars programs',
    riskFactors: ['launch failures', 'orbital congestion', 'space weather', 'ASAT capabilities'],
    cascadeTargets: ['telecommunications', 'GPS services', 'weather forecasting', 'defense ISR'],
  },
  telecom: {
    name: 'Telecommunications Analyst',
    expertise: '5G/6G networks, spectrum policy, undersea cables, satellite internet',
    riskFactors: ['network outages', 'spectrum conflicts', 'cable damage', 'equipment bans'],
    cascadeTargets: ['financial transactions', 'emergency services', 'cloud services', 'IoT'],
  },

  // Agriculture & Employment
  agriculture: {
    name: 'Agriculture & Food Security Analyst',
    expertise: 'crop production, commodity markets, food supply chains, agricultural policy',
    riskFactors: ['crop failures', 'export bans', 'fertilizer shortages', 'pest outbreaks'],
    cascadeTargets: ['food prices', 'social stability', 'biofuel markets', 'livestock feed'],
  },
  employment: {
    name: 'Labor Markets & Employment Analyst',
    expertise: 'labor statistics, wage dynamics, workforce participation, automation impact',
    riskFactors: ['mass layoffs', 'skills mismatch', 'wage-price spiral', 'strike actions'],
    cascadeTargets: ['consumer spending', 'housing markets', 'social programs', 'political sentiment'],
  },
};

// Get analyst for a domain (with fallback)
function getAnalyst(domain: string) {
  return DOMAIN_ANALYSTS[domain] || {
    name: 'Multi-Domain Intelligence Analyst',
    expertise: 'cross-domain risk assessment, systemic analysis, emerging threats',
    riskFactors: ['systemic risk', 'cascading failures', 'black swan events'],
    cascadeTargets: ['multiple sectors', 'global markets', 'social stability'],
  };
}

// ============================================================================
// HISTORIAN AGENT - Searches 500+ year old events for correlates
// ============================================================================

const HISTORIAN_AGENT = {
  name: 'Historical Intelligence Analyst',
  expertise: 'ancient history, medieval conflicts, early modern state formation, historical cycles, long-term pattern recognition',
  minimumYearsAgo: 500,
  instruction: 'Drawing on events from 500+ years ago, identify historical parallels and recurring patterns',
};

// Curated historical events database (500+ years ago only)
const HISTORICAL_EVENTS: {
  id: string;
  name: string;
  period: string;
  yearsAgo: number;
  keywords: string[];
  causalPattern: string;
  outcome: string;
  domains: string[];
}[] = [
  // Ancient & Classical
  {
    id: 'peloponnesian-war',
    name: 'Peloponnesian War',
    period: '431-404 BC',
    yearsAgo: 2455,
    keywords: ['hegemonic rivalry', 'alliance systems', 'democracy vs oligarchy', 'naval power', 'plague'],
    causalPattern: 'Rising power threatens established power → preventive war (Thucydides Trap)',
    outcome: 'Mutual exhaustion, third-party gains (Persia, later Macedon)',
    domains: ['geopolitical', 'defense', 'health'],
  },
  {
    id: 'fall-of-rome',
    name: 'Fall of Western Roman Empire',
    period: '376-476 AD',
    yearsAgo: 1549,
    keywords: ['imperial overstretch', 'border pressure', 'economic decline', 'currency debasement', 'migration'],
    causalPattern: 'Overextension + migration pressure + fiscal crisis → systemic collapse',
    outcome: 'Fragmentation, institutional memory loss, centuries of instability',
    domains: ['geopolitical', 'financial', 'employment', 'defense'],
  },
  {
    id: 'byzantine-arab-wars',
    name: 'Byzantine-Arab Wars',
    period: '629-1180 AD',
    yearsAgo: 1395,
    keywords: ['religious conflict', 'territorial expansion', 'naval warfare', 'siege warfare'],
    causalPattern: 'Ideological expansion + weakened opponent → territorial transformation',
    outcome: 'Permanent territorial realignment, cultural exchange despite conflict',
    domains: ['geopolitical', 'defense'],
  },
  {
    id: 'mongol-invasions',
    name: 'Mongol Invasions',
    period: '1206-1368',
    yearsAgo: 818,
    keywords: ['rapid conquest', 'steppe warfare', 'psychological warfare', 'disease spread', 'trade routes'],
    causalPattern: 'Unified nomadic power + military innovation → rapid expansion',
    outcome: 'Eurasian trade integration (Pax Mongolica), population collapse, institutional destruction',
    domains: ['geopolitical', 'defense', 'supply_chain', 'health'],
  },
  {
    id: 'black-death',
    name: 'Black Death Pandemic',
    period: '1346-1353',
    yearsAgo: 678,
    keywords: ['pandemic', 'trade routes', 'social upheaval', 'labor shortage', 'religious crisis'],
    causalPattern: 'Disease + trade networks + urban density → mass mortality → social transformation',
    outcome: 'Labor power shift, wage gains, religious questioning, eventual Renaissance',
    domains: ['health', 'employment', 'financial', 'supply_chain'],
  },
  {
    id: 'ottoman-rise',
    name: 'Ottoman Rise and Fall of Constantinople',
    period: '1299-1453',
    yearsAgo: 571,
    keywords: ['siege warfare', 'gunpowder', 'religious conflict', 'trade route control', 'military innovation'],
    causalPattern: 'Military innovation + declining opponent → regime change',
    outcome: 'Trade route disruption, Renaissance acceleration, religious polarization',
    domains: ['geopolitical', 'defense', 'tech', 'supply_chain'],
  },
  {
    id: 'hundred-years-war',
    name: 'Hundred Years War',
    period: '1337-1453',
    yearsAgo: 687,
    keywords: ['dynastic conflict', 'longbow', 'nationalism', 'mercenaries', 'economic devastation'],
    causalPattern: 'Succession dispute + technological advantage → prolonged attritional conflict',
    outcome: 'French national identity, English pivot to sea power, end of feudal warfare',
    domains: ['geopolitical', 'defense', 'financial'],
  },
  {
    id: 'ming-treasure-voyages',
    name: 'Ming Treasure Voyages & Withdrawal',
    period: '1405-1433',
    yearsAgo: 620,
    keywords: ['naval power', 'isolationism', 'bureaucratic politics', 'strategic choice'],
    causalPattern: 'Capability + choice not to use → strategic withdrawal',
    outcome: 'Power vacuum in Indian Ocean, European maritime opportunity',
    domains: ['geopolitical', 'space', 'defense'],
  },
  {
    id: 'italian-wars',
    name: 'Italian Wars',
    period: '1494-1559',
    yearsAgo: 531,
    keywords: ['balance of power', 'mercenaries', 'foreign intervention', 'city-state rivalry'],
    causalPattern: 'Power vacuum + external intervention → prolonged proxy conflict',
    outcome: 'Habsburg dominance, end of Italian independence, diplomatic innovation (permanent embassies)',
    domains: ['geopolitical', 'defense', 'financial'],
  },
  {
    id: 'reformation',
    name: 'Protestant Reformation',
    period: '1517-1555',
    yearsAgo: 508,
    keywords: ['religious schism', 'printing press', 'political fragmentation', 'ideological warfare', 'information revolution'],
    causalPattern: 'Information technology + elite dissatisfaction + popular grievance → revolutionary transformation',
    outcome: 'Permanent religious split, wars of religion, state-church reconfiguration',
    domains: ['geopolitical', 'tech', 'ai'],
  },
  {
    id: 'spanish-conquest',
    name: 'Spanish Conquest of Americas',
    period: '1492-1572',
    yearsAgo: 533,
    keywords: ['technological asymmetry', 'disease', 'alliance exploitation', 'resource extraction'],
    causalPattern: 'Technology gap + disease + local divisions → rapid conquest',
    outcome: 'Demographic collapse, silver inflation (Price Revolution), global trade transformation',
    domains: ['geopolitical', 'health', 'financial', 'materials'],
  },
  // Economic & Financial Historical Events
  {
    id: 'florentine-banking',
    name: 'Florentine Banking Crisis (Bardi/Peruzzi)',
    period: '1343-1346',
    yearsAgo: 681,
    keywords: ['sovereign default', 'banking collapse', 'credit crisis', 'Edward III'],
    causalPattern: 'Concentrated sovereign lending + default → systemic banking collapse',
    outcome: 'Medici rise, more diversified banking practices',
    domains: ['financial', 'geopolitical'],
  },
  {
    id: 'debasement-roman',
    name: 'Roman Currency Debasement',
    period: '200-300 AD',
    yearsAgo: 1800,
    keywords: ['inflation', 'currency debasement', 'fiscal crisis', 'silver content'],
    causalPattern: 'Military spending + revenue shortfall → currency debasement → inflation spiral',
    outcome: 'Economic instability, barter economy emergence, Diocletian price controls',
    domains: ['financial', 'defense', 'crypto'],
  },
  // Technological/Scientific
  {
    id: 'printing-revolution',
    name: 'Gutenberg Printing Revolution',
    period: '1440-1500',
    yearsAgo: 585,
    keywords: ['information democratization', 'literacy', 'censorship attempts', 'knowledge diffusion'],
    causalPattern: 'Information technology breakthrough → accelerated social change + resistance',
    outcome: 'Reformation, Scientific Revolution, modern public sphere',
    domains: ['tech', 'ai', 'geopolitical'],
  },
  // Climate & Agriculture
  {
    id: 'little-ice-age-start',
    name: 'Little Ice Age Onset',
    period: '1300-1400',
    yearsAgo: 700,
    keywords: ['climate change', 'crop failure', 'famine', 'social unrest'],
    causalPattern: 'Climate shift → agricultural stress → social/political instability',
    outcome: 'Great Famine 1315-1317, weakened populations before Black Death',
    domains: ['climate', 'agriculture', 'health', 'geopolitical'],
  },
  {
    id: 'mayan-collapse',
    name: 'Classic Maya Collapse',
    period: '800-1000 AD',
    yearsAgo: 1200,
    keywords: ['drought', 'deforestation', 'warfare', 'elite competition', 'infrastructure abandonment'],
    causalPattern: 'Environmental degradation + resource competition → cascading urban collapse',
    outcome: 'Abandonment of major cities, population dispersal, knowledge loss',
    domains: ['climate', 'agriculture', 'geopolitical'],
  },
  // More Ancient Events
  {
    id: 'bronze-age-collapse',
    name: 'Bronze Age Collapse',
    period: '1200-1150 BC',
    yearsAgo: 3200,
    keywords: ['systems collapse', 'sea peoples', 'drought', 'trade disruption', 'palace economies'],
    causalPattern: 'Multiple simultaneous stressors + interdependent systems → cascading civilizational failure',
    outcome: 'Near-total collapse of Eastern Mediterranean civilizations, Dark Age, eventual Iron Age',
    domains: ['geopolitical', 'supply_chain', 'climate', 'defense'],
  },
  {
    id: 'persian-wars',
    name: 'Persian Wars',
    period: '499-449 BC',
    yearsAgo: 2500,
    keywords: ['asymmetric warfare', 'naval innovation', 'coalition warfare', 'imperial overreach'],
    causalPattern: 'Empire attacks smaller states → unified resistance + tactical innovation → repulsion',
    outcome: 'Greek independence, Athenian naval supremacy, eventual Delian League imperialism',
    domains: ['geopolitical', 'defense', 'space'],
  },
  {
    id: 'alexander-conquests',
    name: 'Alexander the Great Conquests',
    period: '336-323 BC',
    yearsAgo: 2350,
    keywords: ['blitzkrieg', 'combined arms', 'decapitation strikes', 'cultural fusion'],
    causalPattern: 'Military genius + decrepit opponent + speed → continental conquest',
    outcome: 'Hellenistic kingdoms, Greek-Persian fusion, knowledge preservation/transmission',
    domains: ['geopolitical', 'defense'],
  },
  {
    id: 'punic-wars',
    name: 'Punic Wars',
    period: '264-146 BC',
    yearsAgo: 2300,
    keywords: ['total war', 'naval supremacy', 'economic warfare', 'scorched earth'],
    causalPattern: 'Commercial rivalry → existential conflict → complete destruction of loser',
    outcome: 'Roman Mediterranean hegemony, destruction of Carthage, expansion of slavery',
    domains: ['geopolitical', 'defense', 'financial', 'supply_chain'],
  },
  {
    id: 'roman-republic-fall',
    name: 'Fall of Roman Republic',
    period: '133-27 BC',
    yearsAgo: 2100,
    keywords: ['political violence', 'wealth inequality', 'military loyalty', 'constitutional crisis'],
    causalPattern: 'Inequality + armed factions + constitutional breakdown → autocracy',
    outcome: 'End of republic, Principate, eventual stability under Augustus',
    domains: ['geopolitical', 'financial', 'employment'],
  },
  {
    id: 'han-dynasty-fall',
    name: 'Fall of Han Dynasty',
    period: '184-220 AD',
    yearsAgo: 1800,
    keywords: ['peasant rebellion', 'warlordism', 'eunuch politics', 'land concentration'],
    causalPattern: 'Agrarian crisis + elite capture + military fragmentation → dynastic collapse',
    outcome: 'Three Kingdoms period, 400 years of division, eventual Sui reunification',
    domains: ['geopolitical', 'agriculture', 'employment'],
  },
  {
    id: 'antonine-plague',
    name: 'Antonine Plague',
    period: '165-180 AD',
    yearsAgo: 1860,
    keywords: ['pandemic', 'military spread', 'population decline', 'economic contraction'],
    causalPattern: 'Disease spread via military/trade networks → demographic shock → economic decline',
    outcome: 'Estimated 5-10 million dead, Roman military weakening, crisis of third century precursor',
    domains: ['health', 'defense', 'financial', 'employment'],
  },
  {
    id: 'justinian-plague',
    name: 'Plague of Justinian',
    period: '541-549 AD',
    yearsAgo: 1480,
    keywords: ['bubonic plague', 'trade routes', 'empire decline', 'labor shortage'],
    causalPattern: 'First bubonic pandemic → massive mortality → imperial overstretch failure',
    outcome: 'End of Justinian reconquest ambitions, Byzantine retrenchment, Arab expansion opportunity',
    domains: ['health', 'geopolitical', 'employment', 'defense'],
  },
  {
    id: 'islamic-conquests',
    name: 'Islamic Conquests',
    period: '632-750 AD',
    yearsAgo: 1400,
    keywords: ['rapid expansion', 'religious motivation', 'cavalry warfare', 'empire collapse'],
    causalPattern: 'Unified ideology + weakened empires + mobile warfare → territorial transformation',
    outcome: 'Umayyad/Abbasid Caliphates, preservation of Greek knowledge, trade network expansion',
    domains: ['geopolitical', 'defense'],
  },
  {
    id: 'viking-age',
    name: 'Viking Age',
    period: '793-1066 AD',
    yearsAgo: 1200,
    keywords: ['raiding', 'naval technology', 'trade routes', 'state formation'],
    causalPattern: 'Demographic pressure + naval innovation + weak targets → expansion/settlement',
    outcome: 'Norman states, Rus formation, North Atlantic exploration, eventual Christianization',
    domains: ['geopolitical', 'defense', 'supply_chain'],
  },
  {
    id: 'song-dynasty-economy',
    name: 'Song Dynasty Economic Revolution',
    period: '960-1279 AD',
    yearsAgo: 1000,
    keywords: ['paper money', 'iron production', 'urbanization', 'proto-industrialization'],
    causalPattern: 'Technological innovation + market integration → economic takeoff',
    outcome: 'First paper money, 100M+ population, eventual Mongol conquest despite wealth',
    domains: ['financial', 'tech', 'manufacturing', 'crypto'],
  },
  {
    id: 'crusades',
    name: 'Crusades',
    period: '1095-1291 AD',
    yearsAgo: 930,
    keywords: ['religious warfare', 'colonialism', 'military orders', 'trade expansion'],
    causalPattern: 'Religious mobilization + demographic pressure → foreign intervention cycle',
    outcome: 'Failed permanent conquest, Italian trade dominance, technology/knowledge transfer',
    domains: ['geopolitical', 'defense', 'supply_chain'],
  },
  {
    id: 'magna-carta',
    name: 'Magna Carta',
    period: '1215 AD',
    yearsAgo: 809,
    keywords: ['constitutional limits', 'noble rebellion', 'rule of law', 'taxation consent'],
    causalPattern: 'Royal overreach + organized opposition → constitutional constraint',
    outcome: 'Foundation of limited government tradition, eventual parliamentary sovereignty',
    domains: ['geopolitical', 'financial'],
  },
  {
    id: 'fourth-crusade',
    name: 'Fourth Crusade Sack of Constantinople',
    period: '1204 AD',
    yearsAgo: 821,
    keywords: ['betrayal', 'commercial interests', 'civilizational damage', 'debt default'],
    causalPattern: 'Debt obligation + commercial opportunism → strategic catastrophe',
    outcome: 'Byzantine permanent weakening, Ottoman eventual conquest, East-West schism deepened',
    domains: ['geopolitical', 'financial', 'defense'],
  },
  {
    id: 'great-famine-1315',
    name: 'Great Famine of 1315-1317',
    period: '1315-1317 AD',
    yearsAgo: 710,
    keywords: ['crop failure', 'climate', 'population collapse', 'social unrest'],
    causalPattern: 'Climate anomaly + population peak → famine → societal stress',
    outcome: 'Estimated 10-25% mortality, end of medieval population growth, Black Death susceptibility',
    domains: ['climate', 'agriculture', 'health'],
  },
  {
    id: 'avignon-papacy',
    name: 'Avignon Papacy & Western Schism',
    period: '1309-1417 AD',
    yearsAgo: 715,
    keywords: ['institutional crisis', 'legitimacy split', 'state capture', 'reform demands'],
    causalPattern: 'Institutional capture + legitimacy crisis → authority fragmentation',
    outcome: 'Permanent damage to papal authority, conciliarism, eventual Reformation',
    domains: ['geopolitical'],
  },
  {
    id: 'tamerlane',
    name: 'Tamerlane Conquests',
    period: '1370-1405 AD',
    yearsAgo: 655,
    keywords: ['terror warfare', 'pyramid skulls', 'cavalry', 'trade disruption'],
    causalPattern: 'Military genius + extreme violence → short-term empire',
    outcome: 'Massive destruction (17M dead est.), Timurid renaissance, Delhi Sultanate weakened',
    domains: ['geopolitical', 'defense'],
  },
  {
    id: 'zheng-he',
    name: 'Zheng He Voyages',
    period: '1405-1433 AD',
    yearsAgo: 620,
    keywords: ['treasure fleets', 'soft power', 'maritime exploration', 'strategic withdrawal'],
    causalPattern: 'State capacity demonstration + policy reversal → strategic opportunity cost',
    outcome: 'Chinese withdrawal, eventual European maritime dominance',
    domains: ['geopolitical', 'defense', 'space'],
  },
  {
    id: 'wars-roses',
    name: 'Wars of the Roses',
    period: '1455-1487 AD',
    yearsAgo: 570,
    keywords: ['dynastic war', 'noble factionalism', 'weak king', 'military aristocracy'],
    causalPattern: 'Succession dispute + noble power + weak center → civil war',
    outcome: 'Tudor dynasty, noble power broken, centralized state',
    domains: ['geopolitical', 'defense'],
  },
  {
    id: 'columbus-exchange',
    name: 'Columbian Exchange',
    period: '1492-1600 AD',
    yearsAgo: 533,
    keywords: ['biological exchange', 'disease', 'crops', 'silver', 'globalization'],
    causalPattern: 'Continental connection → biological/economic transformation',
    outcome: 'Indigenous population collapse (90%+), global trade, European enrichment',
    domains: ['health', 'agriculture', 'financial', 'supply_chain'],
  },
  {
    id: 'machiavelli-era',
    name: 'Machiavelli Era Italian Politics',
    period: '1494-1527 AD',
    yearsAgo: 530,
    keywords: ['realpolitik', 'balance of power', 'foreign invasion', 'political science'],
    causalPattern: 'Multi-polar competition + external intervention → endless conflict',
    outcome: 'Birth of modern political theory, Habsburg-Valois rivalry',
    domains: ['geopolitical', 'defense'],
  },
  {
    id: 'price-revolution',
    name: 'Price Revolution',
    period: '1500-1650 AD',
    yearsAgo: 525,
    keywords: ['inflation', 'silver imports', 'monetary expansion', 'social dislocation'],
    causalPattern: 'Monetary expansion (American silver) → sustained inflation → social transformation',
    outcome: '400-600% price increase, Spanish decline, Northern European rise',
    domains: ['financial', 'crypto', 'employment'],
  },
  {
    id: 'ottoman-peak',
    name: 'Ottoman Empire Peak',
    period: '1520-1566 AD',
    yearsAgo: 505,
    keywords: ['Suleiman', 'naval power', 'siege warfare', 'imperial overstretch'],
    causalPattern: 'Military excellence + institutional strength → expansion → limits',
    outcome: 'Vienna siege failure 1529, Mediterranean contested, eventual stagnation',
    domains: ['geopolitical', 'defense'],
  },
  // More Financial/Economic
  {
    id: 'fugger-banking',
    name: 'Fugger Banking Empire',
    period: '1470-1560 AD',
    yearsAgo: 555,
    keywords: ['merchant banking', 'sovereign lending', 'political influence', 'monopolies'],
    causalPattern: 'Financial innovation + state dependency → political power',
    outcome: 'Habsburg financing, eventual default exposure, diversification',
    domains: ['financial', 'geopolitical', 'materials'],
  },
  {
    id: 'spanish-bankruptcies',
    name: 'Spanish Imperial Bankruptcies',
    period: '1557-1627 AD',
    yearsAgo: 468,
    keywords: ['sovereign default', 'military overstretch', 'silver dependence', 'inflation'],
    causalPattern: 'Military spending > revenue → serial default → creditor losses',
    outcome: '6+ bankruptcies, Genoese banker displacement, Spanish decline',
    domains: ['financial', 'defense', 'geopolitical'],
  },
  // Technology/Science
  {
    id: 'compass-adoption',
    name: 'Magnetic Compass Adoption',
    period: '1100-1300 AD',
    yearsAgo: 900,
    keywords: ['navigation', 'maritime trade', 'Chinese origin', 'technology diffusion'],
    causalPattern: 'Technology transfer + adoption → capability expansion',
    outcome: 'All-weather navigation, European exploration enabled',
    domains: ['tech', 'space', 'supply_chain'],
  },
  {
    id: 'gunpowder-revolution',
    name: 'Gunpowder Revolution',
    period: '1300-1500 AD',
    yearsAgo: 700,
    keywords: ['military revolution', 'castle obsolescence', 'state centralization', 'Chinese origin'],
    causalPattern: 'Weapon technology shift → military organization change → political change',
    outcome: 'Centralized states, noble power decline, modern warfare origins',
    domains: ['defense', 'tech', 'geopolitical'],
  },
  {
    id: 'arabic-science',
    name: 'Islamic Golden Age Translation Movement',
    period: '750-1100 AD',
    yearsAgo: 1250,
    keywords: ['knowledge preservation', 'translation', 'mathematics', 'medicine'],
    causalPattern: 'Patronage + multicultural empire → knowledge synthesis',
    outcome: 'Greek knowledge preserved, algebra/algorithms invented, European Renaissance enabled',
    domains: ['tech', 'ai', 'health'],
  },
  // Disease/Health
  {
    id: 'leprosy-peak',
    name: 'Medieval Leprosy Epidemic',
    period: '1000-1350 AD',
    yearsAgo: 1000,
    keywords: ['disease', 'social exclusion', 'institutional response', 'quarantine'],
    causalPattern: 'Endemic disease → institutional response → social marginalization',
    outcome: 'Leper colonies, quarantine precedent, eventual decline before Black Death',
    domains: ['health'],
  },
  {
    id: 'ergot-poisoning',
    name: 'Ergotism (St. Anthony\'s Fire)',
    period: '857-1300 AD',
    yearsAgo: 1150,
    keywords: ['food contamination', 'mass psychosis', 'rye bread', 'religious interpretation'],
    causalPattern: 'Grain contamination + poverty → mass illness + social disruption',
    outcome: 'Religious hospital orders, grain storage improvements',
    domains: ['health', 'agriculture'],
  },
  // Expansion/Exploration
  {
    id: 'polynesian-expansion',
    name: 'Polynesian Pacific Expansion',
    period: '1000 BC - 1200 AD',
    yearsAgo: 2500,
    keywords: ['navigation', 'colonization', 'resource limits', 'cultural adaptation'],
    causalPattern: 'Navigation mastery + demographic pressure → island colonization',
    outcome: 'Settlement of Pacific including Hawaii, NZ, Easter Island',
    domains: ['space', 'climate', 'agriculture'],
  },
  {
    id: 'vinland',
    name: 'Norse Vinland Settlement',
    period: '1000-1020 AD',
    yearsAgo: 1025,
    keywords: ['exploration', 'failed colonization', 'indigenous resistance', 'climate'],
    causalPattern: 'Exploration + settlement attempt + resistance → abandonment',
    outcome: 'First European Americas contact, no permanent settlement, forgotten until modern',
    domains: ['geopolitical', 'climate'],
  },
  // More Medieval/Institutional
  {
    id: 'hanseatic-league',
    name: 'Hanseatic League',
    period: '1200-1450 AD',
    yearsAgo: 825,
    keywords: ['trade confederation', 'merchant power', 'standardization', 'collective security'],
    causalPattern: 'Commercial cooperation + mutual defense → regional trade dominance',
    outcome: 'Northern European trade control, urban autonomy, eventual nation-state displacement',
    domains: ['financial', 'supply_chain', 'geopolitical'],
  },
  {
    id: 'venetian-system',
    name: 'Venetian Commercial Republic',
    period: '697-1500 AD',
    yearsAgo: 1300,
    keywords: ['maritime republic', 'trade monopoly', 'espionage', 'financial innovation'],
    causalPattern: 'Geographic advantage + institutional innovation → commercial dominance',
    outcome: 'Mediterranean trade control, banking innovation, eventual Portuguese disruption',
    domains: ['financial', 'supply_chain', 'geopolitical', 'cyber'],
  },
  {
    id: 'university-formation',
    name: 'Medieval University Formation',
    period: '1088-1300 AD',
    yearsAgo: 935,
    keywords: ['education', 'knowledge institutions', 'scholasticism', 'autonomy'],
    causalPattern: 'Knowledge demand + institutional innovation → new social institution',
    outcome: 'Bologna, Paris, Oxford - template for higher education, intellectual class',
    domains: ['tech', 'ai'],
  },
  {
    id: 'inquisition',
    name: 'Medieval Inquisition',
    period: '1184-1500 AD',
    yearsAgo: 840,
    keywords: ['heresy', 'surveillance', 'information control', 'confession'],
    causalPattern: 'Ideological threat perception → institutional surveillance response',
    outcome: 'Precedent for systematic persecution, bureaucratic surveillance',
    domains: ['cyber', 'geopolitical'],
  },
  // More Climate/Agriculture
  {
    id: 'medieval-warm-period',
    name: 'Medieval Warm Period',
    period: '900-1300 AD',
    yearsAgo: 1100,
    keywords: ['climate optimum', 'agricultural expansion', 'population growth', 'Greenland'],
    causalPattern: 'Climate warming → agricultural expansion → population/economic growth',
    outcome: 'Viking Greenland, European population peak, cathedral building',
    domains: ['climate', 'agriculture', 'geopolitical'],
  },
  {
    id: 'three-field-system',
    name: 'Three-Field System Adoption',
    period: '800-1200 AD',
    yearsAgo: 1200,
    keywords: ['agricultural revolution', 'productivity', 'population support'],
    causalPattern: 'Agricultural innovation → productivity increase → population capacity',
    outcome: 'European population growth, urbanization enabled',
    domains: ['agriculture', 'employment'],
  },
  // Plague/Epidemic Variations
  {
    id: 'dancing-plague',
    name: 'Dancing Plague of 1518',
    period: '1518 AD',
    yearsAgo: 507,
    keywords: ['mass hysteria', 'stress response', 'social contagion', 'famine context'],
    causalPattern: 'Extreme stress + social contagion → mass psychogenic illness',
    outcome: 'Hundreds affected, deaths, example of stress-induced social phenomena',
    domains: ['health'],
  },
  {
    id: 'sweating-sickness',
    name: 'English Sweating Sickness',
    period: '1485-1551 AD',
    yearsAgo: 540,
    keywords: ['unknown pathogen', 'rapid mortality', 'class differential', 'disappearance'],
    causalPattern: 'Novel pathogen → high mortality → unexplained disappearance',
    outcome: 'Multiple outbreaks then vanished, Arthur Tudor death affected succession',
    domains: ['health', 'geopolitical'],
  },
];

// Find historical correlates for modern events
function findHistoricalCorrelates(
  modernKeywords: string[],
  modernDomain: string,
  limit: number = 3
): typeof HISTORICAL_EVENTS {
  const normalizedKeywords = modernKeywords.map(k => k.toLowerCase());

  // Score each historical event by keyword match and domain relevance
  const scored = HISTORICAL_EVENTS.map(event => {
    let score = 0;

    // Keyword matching
    for (const keyword of event.keywords) {
      for (const modern of normalizedKeywords) {
        if (keyword.includes(modern) || modern.includes(keyword)) {
          score += 2;
        }
      }
    }

    // Domain matching
    if (event.domains.includes(modernDomain)) {
      score += 3;
    }

    // Bonus for causal pattern relevance
    for (const modern of normalizedKeywords) {
      if (event.causalPattern.toLowerCase().includes(modern)) {
        score += 1;
      }
    }

    return { event, score };
  });

  // Return top matches
  return scored
    .filter(s => s.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map(s => s.event);
}

// Generate historical correlate training example
async function generateHistorianExample(
  modernNews: NewsItem[],
  historicalEvents: typeof HISTORICAL_EVENTS
): Promise<{
  instruction: string;
  input: string;
  output: string;
  confidence: number;
} | null> {
  if (historicalEvents.length === 0) return null;

  const modernSummary = modernNews.slice(0, 5).map(n =>
    `- ${n.domain.toUpperCase()}: ${n.title}`
  ).join('\n');

  const historicalContext = historicalEvents.map(e =>
    `- ${e.name} (${e.period}): ${e.causalPattern}`
  ).join('\n');

  const prompt = `You are the ${HISTORIAN_AGENT.name} with expertise in ${HISTORIAN_AGENT.expertise}.

CURRENT SIGNALS (Modern Events):
${modernSummary}

HISTORICAL CORRELATES (500+ years ago):
${historicalContext}

Generate a training example that teaches an AI to recognize historical patterns in modern events.

The example should:
1. Describe a SYNTHESIZED modern situation combining themes from the current signals
2. Provide EXPERT HISTORICAL ANALYSIS that:
   - Identifies the most relevant historical parallel from 500+ years ago
   - Explains the causal mechanism that recurs across centuries
   - Notes KEY DIVERGENCES where modern context differs
   - Calculates "rhyme probability" - how likely the pattern repeats
   - Recommends what historical outcome to watch for

Respond in this exact JSON format:
{
  "input": "Modern situation synthesis with specific actors, dates, and metrics...",
  "output": "HISTORICAL PATTERN ANALYSIS\\n\\n1) PRIMARY CORRELATE: [Historical event] (${historicalEvents[0].period})\\n\\n2) CAUSAL MECHANISM: ...\\n\\n3) KEY DIVERGENCES: ...\\n\\n4) RHYME PROBABILITY: 0.XX - ...\\n\\n5) WATCH FOR: [historical outcome that may recur]",
  "confidence": 0.75
}

Only output the JSON.`;

  try {
    const response = await getAnthropic().messages.create({
      model: 'claude-3-haiku-20240307',
      max_tokens: 1200,
      messages: [{ role: 'user', content: prompt }],
    });

    const text = response.content[0].type === 'text' ? response.content[0].text : '';
    const parsed = JSON.parse(text);

    return {
      instruction: `As a ${HISTORIAN_AGENT.name}, identify historical patterns (500+ years old) that illuminate modern ${modernNews[0]?.domain || 'geopolitical'} developments`,
      input: parsed.input,
      output: parsed.output,
      confidence: parsed.confidence || 0.72,
    };
  } catch (e) {
    console.error('Historian agent generation failed:', e);
    return null;
  }
}

async function fetchGDELT(): Promise<NewsItem[]> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);

    const url = 'https://api.gdeltproject.org/api/v2/doc/doc?query=conflict OR crisis OR war OR sanctions&mode=artlist&maxrecords=20&format=json';
    const res = await fetch(url, {
      next: { revalidate: 0 },
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (!res.ok) {
      console.log(`GDELT returned ${res.status}`);
      return [];
    }

    const data = await res.json();
    const articles = data.articles || [];

    console.log(`GDELT: fetched ${articles.length} articles`);

    return articles.slice(0, 15).map((a: { title: string; seendate: string; url: string }) => ({
      title: a.title,
      description: a.title, // GDELT doesn't give description
      link: a.url,
      pubDate: a.seendate,
      domain: 'geopolitical',
      source_type: 'gdelt',
    }));
  } catch (e) {
    const error = e as Error;
    if (error.name === 'AbortError') {
      console.error('GDELT fetch timeout');
    } else {
      console.error('GDELT fetch failed:', error.message);
    }
    return [];
  }
}

async function fetchRSS(feedUrl: string, domain: string): Promise<NewsItem[]> {
  try {
    // Add timeout to prevent hanging
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);

    const res = await fetch(feedUrl, {
      next: { revalidate: 0 },
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (!res.ok) {
      console.log(`RSS ${feedUrl} returned ${res.status}`);
      return [];
    }

    const text = await res.text();
    // Simple XML parsing for RSS - handle both item and entry (Atom)
    const items: NewsItem[] = [];
    const itemMatches = text.match(/<item>([\s\S]*?)<\/item>/g) ||
                        text.match(/<entry>([\s\S]*?)<\/entry>/g) || [];

    for (const item of itemMatches.slice(0, 10)) { // Increased from 5 to 10
      // Handle multiple CDATA and plain text formats
      const titleMatch = item.match(/<title><!\[CDATA\[([\s\S]*?)\]\]><\/title>/) ||
                        item.match(/<title[^>]*>([\s\S]*?)<\/title>/);
      const title = titleMatch?.[1] || '';

      const descMatch = item.match(/<description><!\[CDATA\[([\s\S]*?)\]\]><\/description>/) ||
                       item.match(/<description[^>]*>([\s\S]*?)<\/description>/) ||
                       item.match(/<summary[^>]*>([\s\S]*?)<\/summary>/) ||
                       item.match(/<content[^>]*>([\s\S]*?)<\/content>/);
      const description = descMatch?.[1] || '';

      const linkMatch = item.match(/<link[^>]*href="([^"]+)"/) ||
                       item.match(/<link>([^<]+)<\/link>/);
      const link = linkMatch?.[1] || '';

      const dateMatch = item.match(/<pubDate>([\s\S]*?)<\/pubDate>/) ||
                       item.match(/<published>([\s\S]*?)<\/published>/) ||
                       item.match(/<updated>([\s\S]*?)<\/updated>/);
      const pubDate = dateMatch?.[1] || new Date().toISOString();

      if (title && title.length > 10) {
        items.push({
          title: title.replace(/<[^>]*>/g, '').trim(),
          description: description.replace(/<[^>]*>/g, '').trim().slice(0, 500),
          link,
          pubDate,
          domain,
          source_type: 'rss',
        });
      }
    }

    console.log(`RSS ${domain}: fetched ${items.length} items from ${feedUrl}`);
    return items;
  } catch (e) {
    const error = e as Error;
    if (error.name === 'AbortError') {
      console.error(`RSS timeout for ${feedUrl}`);
    } else {
      console.error(`RSS fetch failed for ${feedUrl}: ${error.message}`);
    }
    return [];
  }
}

async function generateTrainingExample(news: NewsItem): Promise<{
  instruction: string;
  input: string;
  output: string;
  confidence: number;
} | null> {
  const analyst = getAnalyst(news.domain);

  const prompt = `You are a ${analyst.name} with deep expertise in ${analyst.expertise}.

Convert this news into a training example for an AI risk analysis system.

NEWS:
Title: ${news.title}
Description: ${news.description}
Date: ${news.pubDate}
Domain: ${news.domain}

Your domain-specific risk factors to consider: ${analyst.riskFactors.join(', ')}
Potential cascade effects to: ${analyst.cascadeTargets.join(', ')}

Generate a training example with:
1. A detailed INPUT that describes the situation with specific details, metrics, actors, and context relevant to ${news.domain}
2. An expert OUTPUT analysis as a ${analyst.name} covering:
   - Risk assessment (CRITICAL/HIGH/ELEVATED/LOW) with domain-specific reasoning
   - Key indicators and signals specific to ${news.domain}
   - Cascade potential to: ${analyst.cascadeTargets.slice(0, 3).join(', ')}
   - Historical parallels from your domain expertise
   - Recommended monitoring actions

Respond in this exact JSON format:
{
  "input": "Detailed situation description with dates, actors, metrics...",
  "output": "RISK ASSESSMENT: [LEVEL]\\n\\n1) KEY INDICATORS: ...\\n\\n2) CASCADE POTENTIAL: ...\\n\\n3) HISTORICAL PARALLELS: ...\\n\\n4) MONITORING: ...",
  "confidence": 0.85
}

Only output the JSON, nothing else.`;

  try {
    const response = await getAnthropic().messages.create({
      model: 'claude-3-haiku-20240307',
      max_tokens: 1024,
      messages: [{ role: 'user', content: prompt }],
    });

    const text = response.content[0].type === 'text' ? response.content[0].text : '';
    const parsed = JSON.parse(text);

    return {
      instruction: `As a ${analyst.name}, analyze the ${news.domain} risk signals in the following situation`,
      input: parsed.input,
      output: parsed.output,
      confidence: parsed.confidence || 0.8,
    };
  } catch (e) {
    console.error(`LLM generation failed for ${news.domain}:`, e);
    return null;
  }
}

export async function GET(request: Request) {
  const authHeader = request.headers.get('authorization');
  const cronSecret = process.env.CRON_SECRET;

  // Allow cron or authenticated requests
  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    // Check if it's a Vercel cron request
    const isVercelCron = request.headers.get('x-vercel-cron') === '1';
    if (!isVercelCron) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
  }

  const results = {
    fetched: 0,
    generated: 0,
    stored: 0,
    errors: [] as string[],
    domains: {} as Record<string, number>,
    feed_stats: {} as Record<string, { success: number; failed: number }>,
  };

  // Fetch from all sources IN PARALLEL
  const allNews: NewsItem[] = [];

  // Build list of all feed fetches
  const allSources: { url: string; domain: string }[] = [];
  for (const [category, sources] of Object.entries(RSS_SOURCES)) {
    for (const source of sources) {
      allSources.push(source);
    }
    results.feed_stats[category] = { success: 0, failed: 0 };
  }

  console.log(`Fetching from ${allSources.length} RSS feeds in parallel...`);

  // Fetch all RSS feeds in parallel (with GDELT)
  const fetchPromises = [
    fetchGDELT(),
    ...allSources.map(source => fetchRSS(source.url, source.domain)),
  ];

  const fetchResults = await Promise.allSettled(fetchPromises);

  // Process GDELT result
  if (fetchResults[0].status === 'fulfilled') {
    allNews.push(...fetchResults[0].value);
  }

  // Process RSS results
  for (let i = 1; i < fetchResults.length; i++) {
    const result = fetchResults[i];
    const source = allSources[i - 1];
    const category = Object.entries(RSS_SOURCES).find(([, sources]) =>
      sources.some(s => s.url === source.url)
    )?.[0] || 'unknown';

    if (result.status === 'fulfilled' && result.value.length > 0) {
      allNews.push(...result.value);
      results.feed_stats[category].success++;
    } else {
      results.feed_stats[category].failed++;
      if (result.status === 'rejected') {
        results.errors.push(`Feed ${source.url}: ${result.reason}`);
      }
    }
  }

  console.log(`Total news items fetched: ${allNews.length}`);

  results.fetched = allNews.length;

  // Deduplicate by title before processing
  const seenTitles = new Set<string>();
  const uniqueNews = allNews.filter(news => {
    const key = news.title.toLowerCase().slice(0, 50);
    if (seenTitles.has(key)) return false;
    seenTitles.add(key);
    return true;
  });

  console.log(`Unique news items after dedup: ${uniqueNews.length}`);

  // Group news by domain
  const newsByDomain: Record<string, NewsItem[]> = {};
  for (const news of uniqueNews) {
    if (!newsByDomain[news.domain]) {
      newsByDomain[news.domain] = [];
    }
    newsByDomain[news.domain].push(news);
  }

  const domains = Object.keys(newsByDomain);
  console.log(`Processing ${domains.length} domains in parallel...`);

  // Process each domain as an independent "analyst agent" - ALL IN PARALLEL
  const ITEMS_PER_DOMAIN = 4; // 3-4 items per domain max

  const domainPromises = domains.map(async (domain) => {
    const domainNews = newsByDomain[domain].slice(0, ITEMS_PER_DOMAIN);
    const domainResults: { news: NewsItem; example: Awaited<ReturnType<typeof generateTrainingExample>> }[] = [];

    // Process this domain's items (can do 2-3 in parallel within domain)
    const itemPromises = domainNews.map(async (news) => {
      try {
        const example = await generateTrainingExample(news);
        return { news, example };
      } catch (e) {
        console.error(`Domain ${domain} item error:`, e);
        return { news, example: null };
      }
    });

    const itemResults = await Promise.allSettled(itemPromises);
    for (const result of itemResults) {
      if (result.status === 'fulfilled' && result.value.example) {
        domainResults.push(result.value as { news: NewsItem; example: NonNullable<Awaited<ReturnType<typeof generateTrainingExample>>> });
      }
    }

    return { domain, results: domainResults };
  });

  // Wait for ALL domain agents to complete
  const allDomainResults = await Promise.allSettled(domainPromises);

  // Process results from all domains
  for (const domainResult of allDomainResults) {
    if (domainResult.status !== 'fulfilled') continue;

    const { domain, results: domainItems } = domainResult.value;

    for (const { news, example } of domainItems) {
      if (!example) continue;

      results.generated++;

      // Store in Supabase
      try {
        const { error } = await getSupabase().from('training_examples').insert({
          instruction: example.instruction,
          input: example.input,
          output: example.output,
          domain: news.domain,
          source_type: news.source_type,
          source_url: news.link,
          source_date: new Date(news.pubDate).toISOString(),
          confidence: example.confidence,
        });

        if (error) {
          if (error.code === '23505') {
            // Duplicate - already have this one
            continue;
          }
          results.errors.push(`Insert error (${domain}): ${error.message}`);
        } else {
          results.stored++;
          results.domains[news.domain] = (results.domains[news.domain] || 0) + 1;
        }
      } catch (e) {
        results.errors.push(`DB error (${domain}): ${e}`);
      }
    }
  }

  // ========================================================================
  // HISTORIAN AGENT - Run in parallel with domain agents
  // ========================================================================
  console.log('Running Historian Agent to find historical correlates...');

  // Extract keywords from all modern news for historical matching
  const allKeywords: string[] = [];
  for (const news of uniqueNews.slice(0, 20)) {
    // Extract keywords from titles
    const words = news.title.toLowerCase()
      .replace(/[^a-z\s]/g, '')
      .split(/\s+/)
      .filter(w => w.length > 4);
    allKeywords.push(...words);
  }

  // Find historical correlates
  const historicalCorrelates = findHistoricalCorrelates(
    [...new Set(allKeywords)],
    uniqueNews[0]?.domain || 'geopolitical',
    5
  );

  // Generate historian training examples (2-3 per run)
  const historianPromises = [];
  const domainsWithNews = Object.entries(newsByDomain)
    .filter(([, items]) => items.length > 0)
    .slice(0, 3);

  for (const [domain, domainNews] of domainsWithNews) {
    const domainCorrelates = findHistoricalCorrelates(
      domainNews.slice(0, 3).flatMap(n =>
        n.title.toLowerCase().replace(/[^a-z\s]/g, '').split(/\s+/).filter(w => w.length > 4)
      ),
      domain,
      3
    );

    if (domainCorrelates.length > 0) {
      historianPromises.push(
        generateHistorianExample(domainNews.slice(0, 5), domainCorrelates)
          .then(example => ({ domain, example }))
          .catch(e => {
            console.error(`Historian agent error for ${domain}:`, e);
            return { domain, example: null };
          })
      );
    }
  }

  // Wait for historian examples
  const historianResults = await Promise.allSettled(historianPromises);
  let historianGenerated = 0;
  let historianStored = 0;

  for (const result of historianResults) {
    if (result.status !== 'fulfilled' || !result.value.example) continue;

    const { domain, example } = result.value;
    historianGenerated++;

    try {
      const { error } = await getSupabase().from('training_examples').insert({
        instruction: example.instruction,
        input: example.input,
        output: example.output,
        domain: `historical_${domain}`,
        source_type: 'historian_agent',
        source_url: null,
        source_date: new Date().toISOString(),
        confidence: example.confidence,
      });

      if (error && error.code !== '23505') {
        results.errors.push(`Historian insert error: ${error.message}`);
      } else if (!error) {
        historianStored++;
        results.domains[`historical_${domain}`] = (results.domains[`historical_${domain}`] || 0) + 1;
      }
    } catch (e) {
      results.errors.push(`Historian DB error: ${e}`);
    }
  }

  console.log(`Historian Agent: generated ${historianGenerated}, stored ${historianStored}`);

  // Get total count
  const { count } = await getSupabase()
    .from('training_examples')
    .select('*', { count: 'exact', head: true });

  return NextResponse.json({
    success: true,
    ...results,
    unique_after_dedup: uniqueNews.length,
    domains_processed: domains.length,
    items_per_domain: ITEMS_PER_DOMAIN,
    historian_agent: {
      historical_events_db: HISTORICAL_EVENTS.length,
      correlates_found: historicalCorrelates.length,
      examples_generated: historianGenerated,
      examples_stored: historianStored,
    },
    total_examples: count,
    timestamp: new Date().toISOString(),
  });
}

// Export endpoint to download training data
export async function POST(request: Request) {
  const { format = 'alpaca', mark_exported = false } = await request.json();

  const { data, error } = await getSupabase()
    .from('training_examples')
    .select('id, instruction, input, output')
    .eq('exported', false)
    .gte('confidence', 0.7)
    .order('created_at', { ascending: false });

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  if (mark_exported && data.length > 0) {
    const ids = data.map(d => d.id);
    await getSupabase()
      .from('training_examples')
      .update({ exported: true })
      .in('id', ids);
  }

  if (format === 'alpaca') {
    const alpaca = data.map(d => ({
      instruction: d.instruction,
      input: d.input,
      output: d.output,
    }));
    return NextResponse.json(alpaca);
  }

  // ChatML format
  const chatml = data.map(d => ({
    messages: [
      { role: 'system', content: 'You are a geopolitical risk analyst.' },
      { role: 'user', content: `${d.instruction}\n\n${d.input}` },
      { role: 'assistant', content: d.output },
    ],
  }));

  return NextResponse.json(chatml);
}
