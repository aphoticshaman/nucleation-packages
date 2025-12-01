/**
 * Geolocation Inference
 *
 * Estimates geographic location from:
 * - User-provided profile location
 * - Language detection
 * - Timezone indicators
 * - Content mentions
 */

import type { PostFilter, SocialPost, GeoInfo } from '../types.js';

/**
 * Geolocation config
 */
export interface GeolocationConfig {
  /** Minimum confidence to attach geo */
  minConfidence?: number;
  /** Filter by country codes */
  allowedCountries?: string[];
}

/**
 * Location keyword to country mapping
 */
const LOCATION_KEYWORDS: Record<string, string[]> = {
  US: [
    'usa',
    'united states',
    'america',
    'new york',
    'los angeles',
    'chicago',
    'texas',
    'california',
    'florida',
    'washington dc',
  ],
  GB: [
    'uk',
    'united kingdom',
    'england',
    'london',
    'manchester',
    'birmingham',
    'scotland',
    'wales',
    'british',
  ],
  CA: ['canada', 'toronto', 'vancouver', 'montreal', 'ottawa', 'canadian'],
  AU: ['australia', 'sydney', 'melbourne', 'brisbane', 'perth', 'aussie', 'australian'],
  DE: ['germany', 'deutschland', 'berlin', 'munich', 'frankfurt', 'hamburg', 'german'],
  FR: ['france', 'paris', 'lyon', 'marseille', 'french'],
  ES: ['spain', 'españa', 'madrid', 'barcelona', 'spanish'],
  IT: ['italy', 'italia', 'rome', 'milan', 'italian'],
  BR: ['brazil', 'brasil', 'são paulo', 'rio', 'brazilian'],
  MX: ['mexico', 'méxico', 'mexican', 'ciudad de mexico'],
  JP: ['japan', '日本', 'tokyo', '東京', 'osaka', 'japanese'],
  KR: ['korea', '한국', 'seoul', '서울', 'korean', 'south korea'],
  CN: ['china', '中国', 'beijing', '北京', 'shanghai', '上海', 'chinese'],
  IN: ['india', 'mumbai', 'delhi', 'bangalore', 'indian'],
  RU: ['russia', 'россия', 'moscow', 'москва', 'russian', 'russian federation'],
  UA: ['ukraine', 'україна', 'kyiv', 'київ', 'ukrainian'],
  PL: ['poland', 'polska', 'warsaw', 'polish'],
  NL: ['netherlands', 'holland', 'amsterdam', 'dutch'],
  SE: ['sweden', 'sverige', 'stockholm', 'swedish'],
  NO: ['norway', 'norge', 'oslo', 'norwegian'],
  FI: ['finland', 'suomi', 'helsinki', 'finnish'],
  DK: ['denmark', 'danmark', 'copenhagen', 'danish'],
  TR: ['turkey', 'türkiye', 'istanbul', 'ankara', 'turkish'],
  SA: ['saudi arabia', 'saudi', 'riyadh', 'jeddah'],
  AE: ['uae', 'emirates', 'dubai', 'abu dhabi'],
  IL: ['israel', 'tel aviv', 'jerusalem', 'israeli'],
  IR: ['iran', 'tehran', 'iranian', 'persia'],
  PK: ['pakistan', 'karachi', 'lahore', 'pakistani'],
  ID: ['indonesia', 'jakarta', 'indonesian'],
  TH: ['thailand', 'bangkok', 'thai'],
  VN: ['vietnam', 'hanoi', 'vietnamese'],
  PH: ['philippines', 'manila', 'filipino'],
  MY: ['malaysia', 'kuala lumpur', 'malaysian'],
  SG: ['singapore', 'singaporean'],
  NG: ['nigeria', 'lagos', 'nigerian'],
  ZA: ['south africa', 'johannesburg', 'cape town'],
  EG: ['egypt', 'cairo', 'egyptian'],
  AR: ['argentina', 'buenos aires', 'argentine'],
  CO: ['colombia', 'bogota', 'colombian'],
  CL: ['chile', 'santiago', 'chilean'],
  PE: ['peru', 'lima', 'peruvian'],
};

/**
 * Language to likely country mapping
 */
const LANGUAGE_TO_COUNTRY: Record<string, string[]> = {
  en: ['US', 'GB', 'CA', 'AU'],
  es: ['ES', 'MX', 'AR', 'CO'],
  fr: ['FR', 'CA'],
  de: ['DE', 'AT', 'CH'],
  pt: ['BR', 'PT'],
  ru: ['RU'],
  ar: ['SA', 'AE', 'EG'],
  zh: ['CN', 'TW'],
  ja: ['JP'],
  ko: ['KR'],
  hi: ['IN'],
  tr: ['TR'],
  uk: ['UA'],
  fa: ['IR'],
  vi: ['VN'],
  th: ['TH'],
  id: ['ID'],
};

export class GeolocationFilter implements PostFilter {
  readonly name = 'geolocation';
  private config: Required<GeolocationConfig>;
  private allowedSet: Set<string>;

  constructor(config: GeolocationConfig = {}) {
    this.config = {
      minConfidence: config.minConfidence ?? 0.3,
      allowedCountries: config.allowedCountries ?? [],
    };
    this.allowedSet = new Set(this.config.allowedCountries);
  }

  /**
   * Process a post and infer geolocation
   */
  async process(post: SocialPost): Promise<SocialPost | null> {
    const geo = this.infer(post);

    // Don't override existing high-confidence geo
    if (post.geo && post.geo.confidence > (geo?.confidence ?? 0)) {
      if (this.allowedSet.size > 0 && !this.allowedSet.has(post.geo.countryCode ?? '')) {
        return null;
      }
      return post;
    }

    const processedPost: SocialPost = {
      ...post,
    };

    // Only set geo if it has a value
    const newGeo = geo ?? post.geo;
    if (newGeo) {
      processedPost.geo = newGeo;
    }

    // Filter by country if restrictions set
    if (this.allowedSet.size > 0) {
      const countryCode = processedPost.geo?.countryCode;
      if (!countryCode || !this.allowedSet.has(countryCode)) {
        return null;
      }
    }

    return processedPost;
  }

  /**
   * Infer geolocation from post data
   */
  infer(post: SocialPost): GeoInfo | null {
    const candidates: Array<{
      countryCode: string;
      confidence: number;
      source: GeoInfo['source'];
    }> = [];

    // 1. Check author profile location
    if (post.author.location) {
      const fromProfile = this.parseLocation(post.author.location);
      if (fromProfile) {
        candidates.push({
          countryCode: fromProfile,
          confidence: 0.7,
          source: 'profile',
        });
      }
    }

    // 2. Check language
    if (post.language) {
      const countriesForLang = LANGUAGE_TO_COUNTRY[post.language];
      if (countriesForLang?.length) {
        // First country in list is most likely
        candidates.push({
          countryCode: countriesForLang[0]!,
          confidence: 0.4,
          source: 'language',
        });
      }
    }

    // 3. Check content for location mentions
    const contentLocation = this.extractLocationFromContent(post.content);
    if (contentLocation) {
      candidates.push({
        countryCode: contentLocation,
        confidence: 0.5,
        source: 'inferred',
      });
    }

    // 4. Check author bio
    if (post.author.bio) {
      const fromBio = this.parseLocation(post.author.bio);
      if (fromBio) {
        candidates.push({
          countryCode: fromBio,
          confidence: 0.6,
          source: 'profile',
        });
      }
    }

    if (candidates.length === 0) {
      return null;
    }

    // If multiple candidates agree, boost confidence
    const countryVotes = new Map<string, number>();
    for (const c of candidates) {
      countryVotes.set(c.countryCode, (countryVotes.get(c.countryCode) ?? 0) + c.confidence);
    }

    // Get highest voted country
    const sorted = [...countryVotes.entries()].sort((a, b) => b[1] - a[1]);
    const topCountry = sorted[0]![0];

    // Find best candidate for this country
    const bestCandidate = candidates
      .filter((c) => c.countryCode === topCountry)
      .sort((a, b) => b.confidence - a.confidence)[0];

    if (!bestCandidate) {
      return null;
    }

    // Boost confidence if multiple signals agree
    const agreementBoost = sorted[0]![1] > 1 ? 0.1 : 0;
    const finalConfidence = Math.min(1, bestCandidate.confidence + agreementBoost);

    if (finalConfidence < this.config.minConfidence) {
      return null;
    }

    return {
      countryCode: topCountry,
      source: bestCandidate.source,
      confidence: finalConfidence,
    };
  }

  /**
   * Parse location string to country code
   */
  private parseLocation(location: string): string | null {
    const lower = location.toLowerCase();

    for (const [country, keywords] of Object.entries(LOCATION_KEYWORDS)) {
      for (const keyword of keywords) {
        if (lower.includes(keyword)) {
          return country;
        }
      }
    }

    // Check for ISO country codes directly
    const codeMatch = /\b([A-Z]{2})\b/.exec(location);
    if (codeMatch && codeMatch[1] && Object.keys(LOCATION_KEYWORDS).includes(codeMatch[1])) {
      return codeMatch[1];
    }

    return null;
  }

  /**
   * Extract location from content mentions
   */
  private extractLocationFromContent(content: string): string | null {
    const lower = content.toLowerCase();

    // Look for explicit location mentions
    const patterns = [
      /(?:from|in|at|based in|located in|living in)\s+([a-z\s]+)/gi,
      /📍\s*([a-z\s]+)/gi,
    ];

    for (const pattern of patterns) {
      const matches = content.matchAll(pattern);
      for (const match of matches) {
        const location = match[1]?.trim();
        if (location) {
          const country = this.parseLocation(location);
          if (country) {
            return country;
          }
        }
      }
    }

    // Check for country/city mentions in content
    for (const [country, keywords] of Object.entries(LOCATION_KEYWORDS)) {
      for (const keyword of keywords) {
        // Only match standalone words (not substrings)
        const regex = new RegExp(`\\b${keyword}\\b`, 'i');
        if (regex.test(lower)) {
          return country;
        }
      }
    }

    return null;
  }

  /**
   * Get all supported country codes
   */
  getSupportedCountries(): string[] {
    return Object.keys(LOCATION_KEYWORDS);
  }
}
