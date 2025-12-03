/**
 * ANALYTICS & OBSERVABILITY CONFIGURATION
 *
 * Recommended stack for federal-grade intelligence platform:
 *
 * INCLUDED WITH VERCEL (FREE):
 * - Vercel Analytics - Web vitals, page views
 * - Vercel Speed Insights - Performance monitoring
 *
 * RECOMMENDED ADDITIONS:
 * - PostHog - Product analytics, feature flags, session replay (self-host option)
 * - Sentry - Error tracking, performance monitoring
 * - Upstash Redis - Rate limiting, caching (already billed through Vercel)
 *
 * FOR FEDERAL COMPLIANCE:
 * - All analytics data stays in US regions
 * - PII scrubbing enabled by default
 * - Audit trail integration
 *
 * DATA SCIENCE / ML:
 * - Modal (your stack) - Model inference
 * - HuggingFace - Model hosting
 * - Supabase pgvector - Vector embeddings
 */

// ============================================
// ANALYTICS PROVIDERS
// ============================================

export interface AnalyticsProvider {
  id: string;
  name: string;
  category: 'product' | 'error' | 'performance' | 'user' | 'infra' | 'ml';
  pricing: 'free' | 'freemium' | 'paid' | 'enterprise';
  selfHostable: boolean;
  federalCompliant: boolean;
  description: string;
  useCases: string[];
  vercelIntegration: boolean;
  recommendation: 'essential' | 'recommended' | 'optional' | 'skip';
  monthlyEstimate?: string;
  setupDifficulty: 'trivial' | 'easy' | 'moderate' | 'complex';
}

export const ANALYTICS_PROVIDERS: AnalyticsProvider[] = [
  // ============================================
  // ALREADY IN YOUR STACK
  // ============================================
  {
    id: 'vercel-analytics',
    name: 'Vercel Analytics',
    category: 'performance',
    pricing: 'free',
    selfHostable: false,
    federalCompliant: true,
    description: 'Built-in web vitals and page view analytics',
    useCases: ['Core Web Vitals', 'Page views', 'Traffic sources'],
    vercelIntegration: true,
    recommendation: 'essential',
    monthlyEstimate: 'Free with Vercel Pro',
    setupDifficulty: 'trivial',
  },
  {
    id: 'vercel-speed-insights',
    name: 'Vercel Speed Insights',
    category: 'performance',
    pricing: 'free',
    selfHostable: false,
    federalCompliant: true,
    description: 'Real user performance monitoring',
    useCases: ['LCP', 'FID', 'CLS', 'TTFB', 'INP'],
    vercelIntegration: true,
    recommendation: 'essential',
    monthlyEstimate: 'Free with Vercel Pro',
    setupDifficulty: 'trivial',
  },
  {
    id: 'supabase',
    name: 'Supabase',
    category: 'infra',
    pricing: 'freemium',
    selfHostable: true,
    federalCompliant: true,
    description: 'Your database + auth + realtime + edge functions',
    useCases: ['Database', 'Auth', 'Realtime subscriptions', 'Vector search'],
    vercelIntegration: true,
    recommendation: 'essential',
    monthlyEstimate: '$25/mo Pro tier',
    setupDifficulty: 'easy',
  },

  // ============================================
  // ESSENTIAL ADDITIONS
  // ============================================
  {
    id: 'sentry',
    name: 'Sentry',
    category: 'error',
    pricing: 'freemium',
    selfHostable: true,
    federalCompliant: true,
    description: 'Error tracking, performance monitoring, session replay',
    useCases: [
      'Exception tracking',
      'Stack traces',
      'Release tracking',
      'Performance spans',
      'User feedback',
    ],
    vercelIntegration: true,
    recommendation: 'essential',
    monthlyEstimate: 'Free tier: 5K errors/mo, Team: $26/mo',
    setupDifficulty: 'easy',
  },
  {
    id: 'posthog',
    name: 'PostHog',
    category: 'product',
    pricing: 'freemium',
    selfHostable: true,
    federalCompliant: true,
    description: 'Product analytics, feature flags, A/B testing, session replay',
    useCases: [
      'User funnels',
      'Feature flags',
      'Session recordings',
      'Heatmaps',
      'A/B tests',
      'User cohorts',
    ],
    vercelIntegration: true,
    recommendation: 'essential',
    monthlyEstimate: 'Free: 1M events/mo, then $0.00031/event',
    setupDifficulty: 'easy',
  },

  // ============================================
  // ALREADY BILLED (EVALUATE IF NEEDED)
  // ============================================
  {
    id: 'upstash',
    name: 'Upstash',
    category: 'infra',
    pricing: 'freemium',
    selfHostable: false,
    federalCompliant: true,
    description: 'Serverless Redis & Kafka. NOT for art assets.',
    useCases: [
      'Rate limiting',
      'Session caching',
      'API response caching',
      'Real-time queues',
      'Pub/sub messaging',
    ],
    vercelIntegration: true,
    recommendation: 'recommended',
    monthlyEstimate: 'Pay per request, typically $5-20/mo',
    setupDifficulty: 'easy',
  },

  // ============================================
  // RECOMMENDED FOR SCALE
  // ============================================
  {
    id: 'axiom',
    name: 'Axiom',
    category: 'infra',
    pricing: 'freemium',
    selfHostable: false,
    federalCompliant: true,
    description: 'Log management and observability for Vercel',
    useCases: [
      'Centralized logging',
      'Log search',
      'Dashboards',
      'Alerts',
      'Vercel log drain',
    ],
    vercelIntegration: true,
    recommendation: 'recommended',
    monthlyEstimate: 'Free: 500GB ingest/mo',
    setupDifficulty: 'trivial',
  },
  {
    id: 'checkly',
    name: 'Checkly',
    category: 'performance',
    pricing: 'freemium',
    selfHostable: false,
    federalCompliant: true,
    description: 'Synthetic monitoring and API testing',
    useCases: [
      'Uptime monitoring',
      'API health checks',
      'Browser checks',
      'Alerting',
    ],
    vercelIntegration: true,
    recommendation: 'recommended',
    monthlyEstimate: 'Free: 50 checks, Starter: $30/mo',
    setupDifficulty: 'easy',
  },

  // ============================================
  // ML / DATA SCIENCE
  // ============================================
  {
    id: 'modal',
    name: 'Modal',
    category: 'ml',
    pricing: 'freemium',
    selfHostable: false,
    federalCompliant: true,
    description: 'Serverless ML inference and batch jobs',
    useCases: [
      'Model inference',
      'Batch processing',
      'GPU workloads',
      'Scheduled jobs',
    ],
    vercelIntegration: false,
    recommendation: 'essential',
    monthlyEstimate: '$30 free credits, then usage-based',
    setupDifficulty: 'moderate',
  },
  {
    id: 'huggingface',
    name: 'HuggingFace',
    category: 'ml',
    pricing: 'freemium',
    selfHostable: true,
    federalCompliant: true,
    description: 'Model hub, inference endpoints, datasets',
    useCases: [
      'Model hosting',
      'Inference API',
      'Fine-tuning',
      'Dataset management',
    ],
    vercelIntegration: false,
    recommendation: 'essential',
    monthlyEstimate: 'Free for public models, Inference: $0.06/hr+',
    setupDifficulty: 'moderate',
  },

  // ============================================
  // OPTIONAL / SKIP FOR NOW
  // ============================================
  {
    id: 'mixpanel',
    name: 'Mixpanel',
    category: 'product',
    pricing: 'freemium',
    selfHostable: false,
    federalCompliant: true,
    description: 'Product analytics (PostHog is better value)',
    useCases: ['Event tracking', 'Funnels', 'Retention'],
    vercelIntegration: true,
    recommendation: 'skip',
    monthlyEstimate: 'Free: 20M events/mo',
    setupDifficulty: 'easy',
  },
  {
    id: 'amplitude',
    name: 'Amplitude',
    category: 'product',
    pricing: 'freemium',
    selfHostable: false,
    federalCompliant: true,
    description: 'Enterprise product analytics (overkill for now)',
    useCases: ['Behavioral analytics', 'Cohorts', 'Predictions'],
    vercelIntegration: true,
    recommendation: 'skip',
    monthlyEstimate: 'Free: 10M events/mo',
    setupDifficulty: 'easy',
  },
  {
    id: 'segment',
    name: 'Segment',
    category: 'infra',
    pricing: 'paid',
    selfHostable: false,
    federalCompliant: true,
    description: 'Customer data platform (expensive, not needed yet)',
    useCases: ['Data routing', 'Identity resolution', 'Warehouse sync'],
    vercelIntegration: true,
    recommendation: 'skip',
    monthlyEstimate: 'Starts at $120/mo',
    setupDifficulty: 'moderate',
  },
  {
    id: 'datadog',
    name: 'Datadog',
    category: 'infra',
    pricing: 'enterprise',
    selfHostable: false,
    federalCompliant: true,
    description: 'Enterprise APM (expensive, use Sentry + Axiom instead)',
    useCases: ['APM', 'Logs', 'Metrics', 'Traces'],
    vercelIntegration: true,
    recommendation: 'skip',
    monthlyEstimate: '$15/host/mo minimum',
    setupDifficulty: 'complex',
  },
];

// ============================================
// RECOMMENDED STACK SUMMARY
// ============================================

export const RECOMMENDED_STACK = {
  // Tier 1: Already have or free with Vercel
  included: [
    'Vercel Analytics (free)',
    'Vercel Speed Insights (free)',
    'Supabase (your DB)',
  ],

  // Tier 2: Add these first
  essential: [
    'Sentry - Error tracking ($0-26/mo)',
    'PostHog - Product analytics ($0-50/mo)',
    'Modal - ML inference (already setting up)',
  ],

  // Tier 3: Nice to have
  recommended: [
    'Upstash Redis - Rate limiting (you have this, keep it)',
    'Axiom - Log management (free tier is generous)',
    'Checkly - Uptime monitoring ($0-30/mo)',
  ],

  // Skip these for now
  skip: [
    'Mixpanel (PostHog covers this)',
    'Amplitude (PostHog covers this)',
    'Segment (not needed at your scale)',
    'Datadog (expensive, Sentry + Axiom covers this)',
  ],

  // Monthly estimate
  estimatedMonthlyCost: '$50-150/mo for full observability stack',
};

// ============================================
// DATA SCIENCE CONNECTORS
// ============================================

export interface DataConnector {
  id: string;
  name: string;
  type: 'database' | 'api' | 'stream' | 'file' | 'ml';
  description: string;
  useCases: string[];
  supabaseCompatible: boolean;
  setup: string;
}

export const DATA_CONNECTORS: DataConnector[] = [
  // Vector / Embeddings
  {
    id: 'pgvector',
    name: 'pgvector (Supabase)',
    type: 'database',
    description: 'Vector similarity search in PostgreSQL',
    useCases: [
      'Semantic search',
      'Document similarity',
      'Recommendation engines',
      'RAG applications',
    ],
    supabaseCompatible: true,
    setup: 'Enable pgvector extension in Supabase dashboard',
  },
  {
    id: 'supabase-realtime',
    name: 'Supabase Realtime',
    type: 'stream',
    description: 'Real-time database subscriptions',
    useCases: [
      'Live dashboards',
      'Collaborative features',
      'Alert notifications',
    ],
    supabaseCompatible: true,
    setup: 'Built into Supabase client',
  },

  // External APIs (already in your financialSources.ts)
  {
    id: 'financial-apis',
    name: 'Financial Data APIs',
    type: 'api',
    description: 'Market data from multiple sources',
    useCases: [
      'Stock prices',
      'Economic indicators',
      'Crypto prices',
      'Forex rates',
    ],
    supabaseCompatible: true,
    setup: 'Already configured in lib/signals/financialSources.ts',
  },

  // Geospatial
  {
    id: 'postgis',
    name: 'PostGIS (Supabase)',
    type: 'database',
    description: 'Geospatial queries in PostgreSQL',
    useCases: [
      'Location-based search',
      'Conflict zone mapping',
      'Distance calculations',
    ],
    supabaseCompatible: true,
    setup: 'Enable PostGIS extension in Supabase',
  },

  // Time Series
  {
    id: 'timescaledb',
    name: 'TimescaleDB',
    type: 'database',
    description: 'Time-series extension for PostgreSQL',
    useCases: [
      'Historical trends',
      'Time-bucketed aggregations',
      'Continuous aggregates',
    ],
    supabaseCompatible: false,
    setup: 'Would require separate TimescaleDB instance',
  },
];

// ============================================
// CUSTOMER ANALYTICS EVENTS
// ============================================

export const ANALYTICS_EVENTS = {
  // Onboarding
  onboarding: {
    started: 'onboarding_started',
    step_completed: 'onboarding_step_completed',
    completed: 'onboarding_completed',
    abandoned: 'onboarding_abandoned',
  },

  // Authentication
  auth: {
    signup: 'user_signup',
    login: 'user_login',
    logout: 'user_logout',
    password_reset: 'password_reset_requested',
  },

  // Product Usage
  dashboard: {
    created: 'dashboard_created',
    viewed: 'dashboard_viewed',
    customized: 'dashboard_customized',
    shared: 'dashboard_shared',
    deleted: 'dashboard_deleted',
  },

  briefing: {
    generated: 'briefing_generated',
    viewed: 'briefing_viewed',
    exported: 'briefing_exported',
    shared: 'briefing_shared',
  },

  // Data Pipeline
  pipeline: {
    source_enabled: 'data_source_enabled',
    source_disabled: 'data_source_disabled',
    refresh_triggered: 'data_refresh_triggered',
    quality_warning: 'data_quality_warning',
  },

  // Payments
  payment: {
    checkout_started: 'checkout_started',
    checkout_completed: 'checkout_completed',
    subscription_upgraded: 'subscription_upgraded',
    subscription_downgraded: 'subscription_downgraded',
    subscription_cancelled: 'subscription_cancelled',
  },

  // Feature Usage
  feature: {
    package_builder_opened: 'package_builder_opened',
    package_exported: 'package_exported',
    api_key_generated: 'api_key_generated',
    webhook_created: 'webhook_created',
  },

  // Engagement
  engagement: {
    session_start: 'session_start',
    session_end: 'session_end',
    feature_discovered: 'feature_discovered',
    help_requested: 'help_requested',
    feedback_submitted: 'feedback_submitted',
  },
};

// ============================================
// USER SEGMENTATION
// ============================================

export const USER_SEGMENTS = {
  // By usage pattern
  usage: {
    power_user: {
      criteria: 'sessions > 10/week AND dashboards > 3',
      value: 'high',
    },
    regular: {
      criteria: 'sessions 3-10/week',
      value: 'medium',
    },
    casual: {
      criteria: 'sessions < 3/week',
      value: 'low',
    },
    dormant: {
      criteria: 'no sessions in 14+ days',
      value: 'at_risk',
    },
  },

  // By tier
  tier: {
    explorer: { features: ['basic'], conversion_target: 'analyst' },
    analyst: { features: ['dashboards', 'exports'], conversion_target: 'strategist' },
    strategist: { features: ['team', 'api'], conversion_target: 'architect' },
    architect: { features: ['unlimited', 'custom'], conversion_target: 'enterprise' },
  },

  // By sector (for federal sales)
  sector: {
    government: {
      indicators: ['*.gov email', 'DoD', 'DHS', 'IC'],
      priority: 'high',
    },
    defense_contractor: {
      indicators: ['Lockheed', 'Raytheon', 'Northrop', 'BAE'],
      priority: 'high',
    },
    financial: {
      indicators: ['hedge fund', 'asset management', 'bank'],
      priority: 'high',
    },
    enterprise: {
      indicators: ['Fortune 500', '> 1000 employees'],
      priority: 'medium',
    },
  },
};

// ============================================
// IMPLEMENTATION PRIORITY
// ============================================

export const IMPLEMENTATION_PRIORITY = [
  {
    phase: 1,
    name: 'Foundation',
    timeframe: 'This week',
    tasks: [
      'Enable Vercel Analytics (1 line of code)',
      'Add Sentry for error tracking',
      'Keep Upstash for rate limiting',
    ],
  },
  {
    phase: 2,
    name: 'Product Analytics',
    timeframe: 'Next sprint',
    tasks: [
      'Add PostHog for user analytics',
      'Implement event tracking for key actions',
      'Set up basic funnels (signup → activation → conversion)',
    ],
  },
  {
    phase: 3,
    name: 'Observability',
    timeframe: 'Following sprint',
    tasks: [
      'Add Axiom for log aggregation',
      'Set up Checkly for uptime monitoring',
      'Create alerting rules for critical paths',
    ],
  },
  {
    phase: 4,
    name: 'ML Pipeline',
    timeframe: 'Ongoing',
    tasks: [
      'Configure Modal workspace',
      'Deploy HuggingFace model',
      'Enable pgvector for embeddings',
    ],
  },
];
