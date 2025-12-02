/**
 * SIGNAL SOURCES CONFIGURATION
 *
 * Real-time data ingestion from multiple sources.
 * Based on Grok research on OSINT and news APIs.
 *
 * Sources ranked by value for geopolitical intelligence:
 * 1. GDELT (events, mentions, tone) - FREE
 * 2. NewsAPI (80K+ sources) - $449/mo enterprise
 * 3. NewsCatcher (entity resolution) - usage-based
 * 4. World News API (historical + real-time) - $79/mo
 * 5. GNews (170+ languages) - $79/mo
 * 6. X/Twitter (burst detection) - $100/mo basic
 */

export interface SignalSource {
  id: string;
  name: string;
  type: 'news' | 'social' | 'economic' | 'government' | 'market';
  endpoint: string;
  apiKeyEnv: string;
  rateLimit: {
    requests: number;
    window: 'minute' | 'hour' | 'day';
  };
  priority: 1 | 2 | 3; // 1 = critical, 2 = important, 3 = supplementary
  latency: 'realtime' | 'hourly' | 'daily';
  coverage: string[];
  costPerMonth: number;
  enabled: boolean;
}

export const SIGNAL_SOURCES: SignalSource[] = [
  // FREE TIER - Always enabled
  {
    id: 'gdelt',
    name: 'GDELT Project',
    type: 'news',
    endpoint: 'https://api.gdeltproject.org/api/v2/doc/doc',
    apiKeyEnv: '', // No key needed
    rateLimit: { requests: 60, window: 'minute' },
    priority: 1,
    latency: 'realtime',
    coverage: ['global', 'events', 'tone', 'themes'],
    costPerMonth: 0,
    enabled: true,
  },
  {
    id: 'world_bank',
    name: 'World Bank API',
    type: 'economic',
    endpoint: 'https://api.worldbank.org/v2',
    apiKeyEnv: '',
    rateLimit: { requests: 100, window: 'minute' },
    priority: 1,
    latency: 'daily',
    coverage: ['global', 'economic', 'development'],
    costPerMonth: 0,
    enabled: true,
  },
  {
    id: 'fred',
    name: 'Federal Reserve Economic Data',
    type: 'economic',
    endpoint: 'https://api.stlouisfed.org/fred',
    apiKeyEnv: 'FRED_API_KEY',
    rateLimit: { requests: 120, window: 'minute' },
    priority: 1,
    latency: 'daily',
    coverage: ['usa', 'economic', 'monetary'],
    costPerMonth: 0,
    enabled: true,
  },

  // PAID TIER - Enable as budget allows
  {
    id: 'newscatcher',
    name: 'NewsCatcher API',
    type: 'news',
    endpoint: 'https://api.newscatcherapi.com/v2',
    apiKeyEnv: 'NEWSCATCHER_API_KEY',
    rateLimit: { requests: 21, window: 'hour' },
    priority: 1,
    latency: 'realtime',
    coverage: ['global', 'entity_resolution', 'risk_signals'],
    costPerMonth: 99, // Starter plan
    enabled: false,
  },
  {
    id: 'world_news_api',
    name: 'World News API',
    type: 'news',
    endpoint: 'https://api.worldnewsapi.com',
    apiKeyEnv: 'WORLD_NEWS_API_KEY',
    rateLimit: { requests: 1000, window: 'day' },
    priority: 2,
    latency: 'realtime',
    coverage: ['global', 'historical', 'multilingual'],
    costPerMonth: 79,
    enabled: false,
  },
  {
    id: 'gnews',
    name: 'GNews API',
    type: 'news',
    endpoint: 'https://gnews.io/api/v4',
    apiKeyEnv: 'GNEWS_API_KEY',
    rateLimit: { requests: 100, window: 'day' },
    priority: 2,
    latency: 'realtime',
    coverage: ['global', '170_languages', 'topics'],
    costPerMonth: 79,
    enabled: false,
  },
  {
    id: 'newsapi',
    name: 'NewsAPI',
    type: 'news',
    endpoint: 'https://newsapi.org/v2',
    apiKeyEnv: 'NEWSAPI_KEY',
    rateLimit: { requests: 500, window: 'day' },
    priority: 2,
    latency: 'hourly',
    coverage: ['global', '80k_sources', 'headlines'],
    costPerMonth: 449, // Business plan
    enabled: false,
  },

  // SOCIAL MEDIA
  {
    id: 'twitter_x',
    name: 'X/Twitter API',
    type: 'social',
    endpoint: 'https://api.twitter.com/2',
    apiKeyEnv: 'TWITTER_BEARER_TOKEN',
    rateLimit: { requests: 300, window: 'minute' },
    priority: 1,
    latency: 'realtime',
    coverage: ['global', 'social', 'burst_detection', 'sentiment'],
    costPerMonth: 100, // Basic plan
    enabled: false,
  },
  {
    id: 'bluesky',
    name: 'Bluesky Firehose',
    type: 'social',
    endpoint: 'wss://bsky.network/xrpc/com.atproto.sync.subscribeRepos',
    apiKeyEnv: '', // Public firehose
    rateLimit: { requests: 1000, window: 'minute' },
    priority: 3,
    latency: 'realtime',
    coverage: ['social', 'decentralized', 'tech_focused'],
    costPerMonth: 0,
    enabled: false,
  },

  // GOVERNMENT/OFFICIAL
  {
    id: 'usgs_earthquakes',
    name: 'USGS Earthquake API',
    type: 'government',
    endpoint: 'https://earthquake.usgs.gov/fdsnws/event/1',
    apiKeyEnv: '',
    rateLimit: { requests: 100, window: 'minute' },
    priority: 2,
    latency: 'realtime',
    coverage: ['global', 'natural_disasters', 'seismic'],
    costPerMonth: 0,
    enabled: true,
  },
  {
    id: 'cia_factbook',
    name: 'CIA World Factbook',
    type: 'government',
    endpoint: 'https://raw.githubusercontent.com/factbook/factbook.json/master',
    apiKeyEnv: '',
    rateLimit: { requests: 60, window: 'minute' },
    priority: 2,
    latency: 'daily',
    coverage: ['global', 'country_profiles', 'demographics'],
    costPerMonth: 0,
    enabled: true,
  },

  // MARKET DATA
  {
    id: 'coingecko',
    name: 'CoinGecko API',
    type: 'market',
    endpoint: 'https://api.coingecko.com/api/v3',
    apiKeyEnv: '',
    rateLimit: { requests: 50, window: 'minute' },
    priority: 2,
    latency: 'realtime',
    coverage: ['crypto', 'prices', 'market_cap'],
    costPerMonth: 0,
    enabled: true,
  },
  {
    id: 'fear_greed',
    name: 'Fear & Greed Index',
    type: 'market',
    endpoint: 'https://api.alternative.me/fng',
    apiKeyEnv: '',
    rateLimit: { requests: 100, window: 'day' },
    priority: 3,
    latency: 'daily',
    coverage: ['crypto', 'sentiment'],
    costPerMonth: 0,
    enabled: true,
  },
];

/**
 * Get enabled sources
 */
export function getEnabledSources(): SignalSource[] {
  return SIGNAL_SOURCES.filter((s) => s.enabled);
}

/**
 * Get sources by type
 */
export function getSourcesByType(type: SignalSource['type']): SignalSource[] {
  return SIGNAL_SOURCES.filter((s) => s.type === type && s.enabled);
}

/**
 * Get sources by priority
 */
export function getCriticalSources(): SignalSource[] {
  return SIGNAL_SOURCES.filter((s) => s.priority === 1 && s.enabled);
}

/**
 * Calculate monthly cost for enabled sources
 */
export function calculateMonthlyCost(): number {
  return getEnabledSources().reduce((sum, s) => sum + s.costPerMonth, 0);
}

/**
 * Signal processing keywords for X/Twitter burst detection
 * Based on Grok research: bigrams outperform unigrams for event detection
 */
export const TWITTER_BIGRAMS = {
  conflict: [
    'North Korea missile',
    'Russia Ukraine',
    'Israel Gaza',
    'China Taiwan',
    'Iran nuclear',
    'military strike',
    'troops deployed',
    'ceasefire broken',
    'sanctions imposed',
    'embassy evacuated',
  ],
  economic: [
    'interest rate',
    'Fed raises',
    'market crash',
    'bank failure',
    'currency devaluation',
    'trade war',
    'supply chain',
    'inflation surge',
    'recession fears',
    'debt default',
  ],
  political: [
    'coup attempt',
    'election fraud',
    'martial law',
    'protest violence',
    'leader assassinated',
    'government collapse',
    'emergency declared',
    'constitutional crisis',
    'mass arrests',
    'parliament dissolved',
  ],
  natural: [
    'earthquake magnitude',
    'hurricane category',
    'tsunami warning',
    'volcanic eruption',
    'flood emergency',
    'wildfire spreading',
    'drought crisis',
    'famine declared',
    'pandemic outbreak',
    'nuclear accident',
  ],
  cyber: [
    'data breach',
    'ransomware attack',
    'infrastructure hack',
    'election interference',
    'DDoS attack',
    'zero-day exploit',
    'state-sponsored',
    'critical systems',
    'power grid',
    'financial system',
  ],
};

/**
 * Entity types to extract from news
 * For NewsCatcher API integration
 */
export const ENTITY_TYPES = [
  'PERSON', // Leaders, officials
  'ORG', // Organizations, companies
  'GPE', // Countries, cities
  'EVENT', // Named events
  'NORP', // Nationalities, religious groups
  'FAC', // Facilities, bases
  'PRODUCT', // Weapons, tech
  'MONEY', // Financial amounts
  'PERCENT', // Statistics
  'DATE', // Temporal references
];

/**
 * Risk signal keywords for news filtering
 */
export const RISK_SIGNALS = {
  immediate: [
    'breaking',
    'urgent',
    'emergency',
    'attack',
    'explosion',
    'shooting',
    'crash',
    'collapse',
    'outbreak',
    'invasion',
  ],
  escalating: [
    'tensions rise',
    'threatens',
    'warns',
    'ultimatum',
    'deadline',
    'mobilizes',
    'deploys',
    'sanctions',
    'embargo',
    'blockade',
  ],
  systemic: [
    'crisis',
    'recession',
    'default',
    'contagion',
    'cascade',
    'systemic',
    'collapse',
    'failure',
    'meltdown',
    'spiral',
  ],
};
