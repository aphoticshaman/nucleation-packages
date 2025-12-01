/**
 * Core types for social-pulse sentiment aggregation
 */

/**
 * A social media post from any platform
 */
export interface SocialPost {
  /** Unique identifier */
  id: string;
  /** Source platform */
  platform: Platform;
  /** Post content/text */
  content: string;
  /** ISO timestamp */
  timestamp: string;
  /** Author information */
  author: AuthorInfo;
  /** Engagement metrics */
  engagement: EngagementMetrics;
  /** Detected language (ISO 639-1 code) */
  language?: string;
  /** Geolocation data if available */
  geo?: GeoInfo;
  /** Bot probability score 0-1 */
  botScore?: number;
  /** Sentiment score -1 to 1 */
  sentimentScore?: number;
  /** Raw platform-specific data */
  raw?: unknown;
}

/**
 * Supported platforms
 */
export type Platform =
  | 'bluesky'
  | 'telegram'
  | 'x'
  | 'tiktok'
  | 'vk'
  | 'weibo'
  | 'line'
  | 'kakao'
  | 'zalo'
  | 'sharechat'
  | 'gdelt'
  | 'dark_web'
  | 'custom';

/**
 * Author information
 */
export interface AuthorInfo {
  /** Platform-specific user ID */
  id: string;
  /** Display name or handle */
  name: string;
  /** Account creation date if available */
  createdAt?: string;
  /** Follower count if available */
  followers?: number;
  /** Following count if available */
  following?: number;
  /** Post count if available */
  postCount?: number;
  /** Self-reported location string */
  location?: string;
  /** Bio/description */
  bio?: string;
  /** Verified status */
  verified?: boolean;
}

/**
 * Engagement metrics
 */
export interface EngagementMetrics {
  likes?: number;
  reposts?: number;
  replies?: number;
  views?: number;
  shares?: number;
}

/**
 * Geolocation information
 */
export interface GeoInfo {
  /** ISO 3166-1 alpha-2 country code */
  countryCode?: string;
  /** Country name */
  country?: string;
  /** Region/state/province */
  region?: string;
  /** City */
  city?: string;
  /** Latitude */
  lat?: number;
  /** Longitude */
  lon?: number;
  /** Source of geo data */
  source: 'gps' | 'ip' | 'profile' | 'inferred' | 'language';
  /** Confidence 0-1 */
  confidence: number;
}

/**
 * Data source configuration
 */
export interface SourceConfig {
  /** Platform identifier */
  platform: Platform;
  /** API credentials if required */
  credentials?: {
    apiKey?: string;
    apiSecret?: string;
    accessToken?: string;
    bearerToken?: string;
  };
  /** Rate limit (requests per minute) */
  rateLimit?: number;
  /** Custom API endpoint */
  endpoint?: string;
  /** Proxy configuration */
  proxy?: {
    host: string;
    port: number;
    auth?: { username: string; password: string };
  };
}

/**
 * Search/filter parameters
 */
export interface SearchParams {
  /** Keywords to search for */
  keywords?: string[];
  /** Hashtags */
  hashtags?: string[];
  /** Filter by country codes */
  countries?: string[];
  /** Filter by languages */
  languages?: string[];
  /** Minimum engagement threshold */
  minEngagement?: number;
  /** Maximum bot score (filter out likely bots) */
  maxBotScore?: number;
  /** Time range start */
  since?: Date;
  /** Time range end */
  until?: Date;
  /** Maximum results to return */
  limit?: number;
}

/**
 * Aggregated sentiment for a region/topic
 */
export interface SentimentAggregate {
  /** Region or topic identifier */
  id: string;
  /** ISO 3166-1 alpha-2 country code if geographic */
  countryCode?: string;
  /** Topic/keyword if topic-based */
  topic?: string;
  /** Time window start */
  windowStart: string;
  /** Time window end */
  windowEnd: string;
  /** Number of posts analyzed */
  postCount: number;
  /** Unique authors */
  authorCount: number;
  /** Average sentiment -1 to 1 */
  avgSentiment: number;
  /** Sentiment standard deviation */
  sentimentStdDev: number;
  /** Percentage of negative posts */
  negativeRatio: number;
  /** Percentage of posts filtered as bots */
  botFilteredRatio: number;
  /** Top keywords/hashtags */
  topKeywords: Array<{ word: string; count: number }>;
  /** Platform breakdown */
  platformBreakdown: Partial<Record<Platform, number>>;
  /** Variance (for phase transition detection) */
  variance: number;
  /** Previous period's variance for comparison */
  previousVariance?: number;
}

/**
 * Revolution/upheaval detection state
 */
export type UpheavalLevel = 'calm' | 'stirring' | 'unrest' | 'volatile';

export interface UpheavalState {
  level: UpheavalLevel;
  levelNumeric: number;
  variance: number;
  mean: number;
  dataPoints: number;
  lastUpdate: string;
  /** Countries with elevated signals */
  hotspots: Array<{
    countryCode: string;
    level: UpheavalLevel;
    variance: number;
    topKeywords: string[];
  }>;
}

/**
 * Data source interface - implement this to add new platforms
 */
export interface DataSource {
  /** Platform identifier */
  readonly platform: Platform;

  /** Initialize the source (authenticate, etc.) */
  init(): Promise<void>;

  /** Fetch posts matching search params */
  fetch(params: SearchParams): Promise<SocialPost[]>;

  /** Stream posts in real-time (if supported) */
  stream?(
    params: SearchParams,
    callback: (post: SocialPost) => void
  ): Promise<{ stop: () => void }>;

  /** Check if source is ready */
  isReady(): boolean;

  /** Get rate limit status */
  getRateLimitStatus?(): { remaining: number; resetAt: Date };
}

/**
 * Post filter interface - implement to add filters
 */
export interface PostFilter {
  /** Filter name */
  readonly name: string;

  /** Process a post, returning modified post or null to filter out */
  process(post: SocialPost): Promise<SocialPost | null>;
}
