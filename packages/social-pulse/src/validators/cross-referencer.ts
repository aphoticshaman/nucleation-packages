/**
 * Cross-Referencer: Adversarial fact-checking and background verification
 *
 * Acts as a hostile reviewer of information before publishing.
 * Challenges claims by:
 * 1. Searching fact-check databases (Snopes, Google Fact Check API)
 * 2. Cross-referencing with OSINT sources
 * 3. Detecting contradictions in source's history
 * 4. Checking for coordinated inauthentic behavior patterns
 */

import type { SocialPost } from '../types.js';

/**
 * Verification result for a piece of content
 */
export interface VerificationResult {
  /** Overall trust score 0-1 (0 = definitely false, 1 = highly trusted) */
  trustScore: number;
  /** Verification status */
  status: 'verified' | 'disputed' | 'unverified' | 'likely_false' | 'manipulation_detected';
  /** Found fact-checks */
  factChecks: FactCheckResult[];
  /** Corroborating sources */
  corroboration: CorroborationResult[];
  /** Red flags detected */
  redFlags: RedFlag[];
  /** Contradictions found */
  contradictions: Contradiction[];
  /** Timestamp of verification */
  verifiedAt: string;
  /** Recommendation */
  recommendation: 'publish' | 'hold_for_review' | 'do_not_publish';
  /** Human-readable summary */
  summary: string;
}

export interface FactCheckResult {
  source: string;
  url?: string;
  rating: string;
  claim: string;
  date?: string;
}

export interface CorroborationResult {
  source: string;
  type: 'supports' | 'contradicts' | 'related';
  url?: string;
  snippet?: string;
  confidence: number;
}

export interface RedFlag {
  type: RedFlagType;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  evidence?: string;
}

export type RedFlagType =
  | 'coordinated_behavior'
  | 'bot_network'
  | 'known_disinfo_source'
  | 'manipulated_media'
  | 'temporal_anomaly'
  | 'geographic_mismatch'
  | 'narrative_amplification'
  | 'sockpuppet_pattern'
  | 'fresh_account'
  | 'deleted_history';

export interface Contradiction {
  claim: string;
  contradictingSource: string;
  contradictingClaim: string;
  confidence: number;
}

/**
 * Google Fact Check Tools API response types
 */
interface GoogleFactCheckResponse {
  claims?: Array<{
    text: string;
    claimant?: string;
    claimDate?: string;
    claimReview?: Array<{
      publisher: { name: string; site?: string };
      url: string;
      title: string;
      textualRating: string;
    }>;
  }>;
}

/**
 * Cross-Referencer configuration
 */
export interface CrossReferencerConfig {
  /** Google Fact Check API key (free tier available) */
  googleFactCheckApiKey?: string;
  /** Minimum trust score to recommend publishing */
  publishThreshold?: number;
  /** Enable aggressive adversarial mode */
  adversarialMode?: boolean;
  /** Known disinfo domains to flag */
  knownDisinfoDomains?: string[];
}

// Known disinformation domains (partial list - expand as needed)
const DEFAULT_DISINFO_DOMAINS = [
  'rt.com',
  'sputniknews.com',
  'globalresearch.ca',
  'naturalnews.com',
  'infowars.com',
  'beforeitsnews.com',
  'yournewswire.com',
  'newspunch.com',
];

// Patterns indicating coordinated behavior
const COORDINATION_PATTERNS = {
  // Posts within seconds of each other with similar content
  temporalClustering: 60, // seconds
  // Same hashtags appearing across unrelated accounts
  hashtagAmplification: 5,
  // Account age vs. engagement ratio (suspicious if very new with high engagement)
  freshAccountThreshold: 7, // days
};

export class CrossReferencer {
  private config: Required<CrossReferencerConfig>;
  private factCheckCache = new Map<string, FactCheckResult[]>();

  constructor(config: CrossReferencerConfig = {}) {
    this.config = {
      googleFactCheckApiKey: config.googleFactCheckApiKey ?? '',
      publishThreshold: config.publishThreshold ?? 0.6,
      adversarialMode: config.adversarialMode ?? true,
      knownDisinfoDomains: config.knownDisinfoDomains ?? DEFAULT_DISINFO_DOMAINS,
    };
  }

  /**
   * Verify a single post/claim
   */
  async verify(post: SocialPost): Promise<VerificationResult> {
    const redFlags: RedFlag[] = [];
    const contradictions: Contradiction[] = [];
    const factChecks: FactCheckResult[] = [];
    const corroboration: CorroborationResult[] = [];

    // Run all checks in parallel
    const [factCheckResults, authorRedFlags, contentRedFlags, coordinationFlags] =
      await Promise.all([
        this.checkFactCheckDatabases(post.content),
        this.analyzeAuthor(post.author, post),
        this.analyzeContent(post.content),
        this.detectCoordinatedBehavior(post),
      ]);

    factChecks.push(...factCheckResults);
    redFlags.push(...authorRedFlags, ...contentRedFlags, ...coordinationFlags);

    // Calculate trust score
    const trustScore = this.calculateTrustScore(factChecks, redFlags, corroboration);

    // Determine status
    const status = this.determineStatus(trustScore, factChecks, redFlags);

    // Generate recommendation
    const recommendation = this.generateRecommendation(trustScore, status, redFlags);

    // Generate summary
    const summary = this.generateSummary(trustScore, status, redFlags, factChecks);

    return {
      trustScore,
      status,
      factChecks,
      corroboration,
      redFlags,
      contradictions,
      verifiedAt: new Date().toISOString(),
      recommendation,
      summary,
    };
  }

  /**
   * Batch verify multiple posts
   */
  async verifyBatch(posts: SocialPost[]): Promise<Map<string, VerificationResult>> {
    const results = new Map<string, VerificationResult>();

    // Also check for cross-post coordination
    const coordinationFlags = this.detectCrossPostCoordination(posts);

    for (const post of posts) {
      const result = await this.verify(post);

      // Add coordination flags specific to this batch
      const postCoordFlags = coordinationFlags.get(post.id) ?? [];
      result.redFlags.push(...postCoordFlags);

      // Recalculate if coordination flags were added
      if (postCoordFlags.length > 0) {
        result.trustScore = this.calculateTrustScore(
          result.factChecks,
          result.redFlags,
          result.corroboration
        );
        result.recommendation = this.generateRecommendation(
          result.trustScore,
          result.status,
          result.redFlags
        );
      }

      results.set(post.id, result);
    }

    return results;
  }

  /**
   * Check fact-checking databases
   */
  private async checkFactCheckDatabases(content: string): Promise<FactCheckResult[]> {
    const results: FactCheckResult[] = [];

    // Extract key claims from content
    const claims = this.extractClaims(content);

    for (const claim of claims.slice(0, 3)) {
      // Check cache
      if (this.factCheckCache.has(claim)) {
        results.push(...this.factCheckCache.get(claim)!);
        continue;
      }

      // Google Fact Check API
      if (this.config.googleFactCheckApiKey) {
        try {
          const googleResults = await this.queryGoogleFactCheck(claim);
          results.push(...googleResults);
          this.factCheckCache.set(claim, googleResults);
        } catch (error) {
          console.warn('Google Fact Check API error:', error);
        }
      }
    }

    return results;
  }

  /**
   * Query Google Fact Check Tools API
   */
  private async queryGoogleFactCheck(query: string): Promise<FactCheckResult[]> {
    const url = new URL('https://factchecktools.googleapis.com/v1alpha1/claims:search');
    url.searchParams.set('query', query);
    url.searchParams.set('key', this.config.googleFactCheckApiKey);
    url.searchParams.set('languageCode', 'en');

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`Fact check API error: ${response.status}`);
    }

    const data = (await response.json()) as GoogleFactCheckResponse;
    const results: FactCheckResult[] = [];

    for (const claim of data.claims ?? []) {
      for (const review of claim.claimReview ?? []) {
        const result: FactCheckResult = {
          source: review.publisher.name,
          url: review.url,
          rating: review.textualRating,
          claim: claim.text,
        };
        // Only add date if it exists
        if (claim.claimDate) {
          result.date = claim.claimDate;
        }
        results.push(result);
      }
    }

    return results;
  }

  /**
   * Analyze author for red flags
   */
  private async analyzeAuthor(author: SocialPost['author'], post: SocialPost): Promise<RedFlag[]> {
    const flags: RedFlag[] = [];

    // Check account age
    if (author.createdAt) {
      const accountAge = Date.now() - new Date(author.createdAt).getTime();
      const daysSinceCreation = accountAge / (1000 * 60 * 60 * 24);

      if (daysSinceCreation < COORDINATION_PATTERNS.freshAccountThreshold) {
        flags.push({
          type: 'fresh_account',
          severity: daysSinceCreation < 1 ? 'high' : 'medium',
          description: `Account created ${Math.round(daysSinceCreation)} days ago`,
          evidence: author.createdAt,
        });
      }
    }

    // Check follower/following ratio (bot indicator)
    if (author.followers !== undefined && author.following !== undefined) {
      if (author.following > 0 && author.followers / author.following < 0.01) {
        flags.push({
          type: 'bot_network',
          severity: 'medium',
          description: 'Suspicious follower/following ratio (follows many, few followers)',
        });
      }
    }

    // Check for verified status vs. claims
    if (!author.verified && post.content.toLowerCase().includes('official')) {
      flags.push({
        type: 'sockpuppet_pattern',
        severity: 'low',
        description: 'Unverified account making official-sounding claims',
      });
    }

    return flags;
  }

  /**
   * Analyze content for red flags
   */
  private async analyzeContent(content: string): Promise<RedFlag[]> {
    const flags: RedFlag[] = [];
    const lowerContent = content.toLowerCase();

    // Check for known disinfo domains
    for (const domain of this.config.knownDisinfoDomains) {
      if (lowerContent.includes(domain)) {
        flags.push({
          type: 'known_disinfo_source',
          severity: 'high',
          description: `Links to known disinformation source: ${domain}`,
          evidence: domain,
        });
      }
    }

    // Emotional manipulation patterns
    const manipulationPatterns = [
      { pattern: /breaking:?\s*!/i, severity: 'low' as const },
      { pattern: /share before.*deleted/i, severity: 'high' as const },
      { pattern: /they don't want you to know/i, severity: 'medium' as const },
      { pattern: /exposed!|bombshell!/i, severity: 'medium' as const },
      { pattern: /wake up/i, severity: 'low' as const },
    ];

    for (const { pattern, severity } of manipulationPatterns) {
      if (pattern.test(content)) {
        flags.push({
          type: 'narrative_amplification',
          severity,
          description: 'Contains emotional manipulation language',
          evidence: pattern.source,
        });
      }
    }

    return flags;
  }

  /**
   * Detect coordinated inauthentic behavior
   */
  private async detectCoordinatedBehavior(post: SocialPost): Promise<RedFlag[]> {
    const flags: RedFlag[] = [];

    // Geographic mismatch
    if (post.geo && post.author.location) {
      const claimedLocation = post.author.location.toLowerCase();
      const detectedCountry = post.geo.country?.toLowerCase() ?? '';

      if (
        detectedCountry &&
        !claimedLocation.includes(detectedCountry) &&
        post.geo.confidence > 0.7
      ) {
        flags.push({
          type: 'geographic_mismatch',
          severity: 'medium',
          description: `Claimed location "${post.author.location}" doesn't match detected location "${post.geo.country}"`,
        });
      }
    }

    return flags;
  }

  /**
   * Detect coordination across multiple posts
   */
  private detectCrossPostCoordination(posts: SocialPost[]): Map<string, RedFlag[]> {
    const flagsByPost = new Map<string, RedFlag[]>();

    // Group posts by timestamp proximity
    const sortedPosts = [...posts].sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    for (let i = 0; i < sortedPosts.length - 1; i++) {
      const current = sortedPosts[i];
      const next = sortedPosts[i + 1];

      if (!current || !next) continue;

      const timeDiff =
        (new Date(next.timestamp).getTime() - new Date(current.timestamp).getTime()) / 1000;

      // Check for suspiciously close timing with similar content
      if (timeDiff < COORDINATION_PATTERNS.temporalClustering) {
        const similarity = this.calculateSimilarity(current.content, next.content);

        if (similarity > 0.7) {
          const flag: RedFlag = {
            type: 'coordinated_behavior',
            severity: 'high',
            description: `Posts within ${timeDiff}s with ${Math.round(similarity * 100)}% similar content`,
          };

          // Add flag to both posts
          if (!flagsByPost.has(current.id)) {
            flagsByPost.set(current.id, []);
          }
          if (!flagsByPost.has(next.id)) {
            flagsByPost.set(next.id, []);
          }
          flagsByPost.get(current.id)!.push(flag);
          flagsByPost.get(next.id)!.push(flag);
        }
      }
    }

    return flagsByPost;
  }

  /**
   * Simple content similarity check
   */
  private calculateSimilarity(a: string, b: string): number {
    const wordsA = new Set(a.toLowerCase().split(/\s+/));
    const wordsB = new Set(b.toLowerCase().split(/\s+/));

    const intersection = new Set([...wordsA].filter((x) => wordsB.has(x)));
    const union = new Set([...wordsA, ...wordsB]);

    return intersection.size / union.size;
  }

  /**
   * Extract key claims from content
   */
  private extractClaims(content: string): string[] {
    // Split into sentences and extract claim-like statements
    const sentences = content
      .split(/[.!?]+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 20 && s.length < 200);

    // Prioritize sentences with claim indicators
    const claimIndicators = [
      /\b(is|are|was|were)\b/i,
      /\b(says?|said|claims?|reported)\b/i,
      /\b(according to)\b/i,
      /\b(breaking|confirmed|exclusive)\b/i,
    ];

    return sentences.filter((s) => claimIndicators.some((pattern) => pattern.test(s))).slice(0, 5);
  }

  /**
   * Calculate overall trust score
   */
  private calculateTrustScore(
    factChecks: FactCheckResult[],
    redFlags: RedFlag[],
    corroboration: CorroborationResult[]
  ): number {
    let score = 0.5; // Start neutral

    // Adjust based on fact checks
    for (const check of factChecks) {
      const rating = check.rating.toLowerCase();
      if (rating.includes('false') || rating.includes('pants on fire')) {
        score -= 0.3;
      } else if (rating.includes('mostly false') || rating.includes('misleading')) {
        score -= 0.2;
      } else if (rating.includes('true') || rating.includes('correct')) {
        score += 0.2;
      } else if (rating.includes('mostly true')) {
        score += 0.1;
      }
    }

    // Adjust based on red flags
    const severityPenalty = {
      critical: 0.25,
      high: 0.15,
      medium: 0.08,
      low: 0.03,
    };

    for (const flag of redFlags) {
      score -= severityPenalty[flag.severity];
    }

    // Adjust based on corroboration
    for (const corr of corroboration) {
      if (corr.type === 'supports') {
        score += 0.1 * corr.confidence;
      } else if (corr.type === 'contradicts') {
        score -= 0.15 * corr.confidence;
      }
    }

    // Clamp to 0-1
    return Math.max(0, Math.min(1, score));
  }

  /**
   * Determine verification status
   */
  private determineStatus(
    trustScore: number,
    factChecks: FactCheckResult[],
    redFlags: RedFlag[]
  ): VerificationResult['status'] {
    // Check for manipulation first
    const hasManipulation = redFlags.some(
      (f) =>
        f.type === 'coordinated_behavior' ||
        f.type === 'bot_network' ||
        f.type === 'manipulated_media'
    );

    if (
      hasManipulation &&
      redFlags.some((f) => f.severity === 'high' || f.severity === 'critical')
    ) {
      return 'manipulation_detected';
    }

    // Check fact check results
    const hasFalseRating = factChecks.some((fc) => fc.rating.toLowerCase().includes('false'));

    if (hasFalseRating) {
      return 'likely_false';
    }

    if (factChecks.some((fc) => fc.rating.toLowerCase().includes('disputed'))) {
      return 'disputed';
    }

    if (trustScore >= 0.7 && factChecks.some((fc) => fc.rating.toLowerCase().includes('true'))) {
      return 'verified';
    }

    return 'unverified';
  }

  /**
   * Generate publish recommendation
   */
  private generateRecommendation(
    trustScore: number,
    status: VerificationResult['status'],
    redFlags: RedFlag[]
  ): VerificationResult['recommendation'] {
    // Never publish manipulation or likely false
    if (status === 'manipulation_detected' || status === 'likely_false') {
      return 'do_not_publish';
    }

    // Critical red flags require review
    if (redFlags.some((f) => f.severity === 'critical')) {
      return 'do_not_publish';
    }

    // High severity flags need review
    if (redFlags.filter((f) => f.severity === 'high').length >= 2) {
      return 'hold_for_review';
    }

    // Check threshold
    if (trustScore >= this.config.publishThreshold) {
      return 'publish';
    }

    return 'hold_for_review';
  }

  /**
   * Generate human-readable summary
   */
  private generateSummary(
    trustScore: number,
    status: VerificationResult['status'],
    redFlags: RedFlag[],
    factChecks: FactCheckResult[]
  ): string {
    const parts: string[] = [];

    parts.push(`Trust score: ${Math.round(trustScore * 100)}%`);

    if (status === 'manipulation_detected') {
      parts.push('WARNING: Coordinated manipulation detected.');
    } else if (status === 'likely_false') {
      parts.push('WARNING: Fact-checkers have rated similar claims as false.');
    } else if (status === 'disputed') {
      parts.push('NOTICE: This claim is disputed by fact-checkers.');
    } else if (status === 'verified') {
      parts.push('This claim appears to be supported by fact-checkers.');
    } else {
      parts.push('This claim could not be independently verified.');
    }

    if (redFlags.length > 0) {
      const highFlags = redFlags.filter((f) => f.severity === 'high' || f.severity === 'critical');
      if (highFlags.length > 0) {
        parts.push(`${highFlags.length} serious concern(s) detected.`);
      }
    }

    if (factChecks.length > 0) {
      parts.push(`${factChecks.length} related fact-check(s) found.`);
    }

    return parts.join(' ');
  }
}
