/**
 * Bot Detection Filter
 *
 * Heuristic-based bot detection using account patterns,
 * posting behavior, and content analysis.
 */

import type { PostFilter, SocialPost } from '../types.js';

/**
 * Bot detection configuration
 */
export interface BotFilterConfig {
  /** Score threshold above which post is filtered (0-1) */
  threshold?: number;
  /** Weight adjustments for each signal */
  weights?: Partial<Record<BotSignal, number>>;
}

/**
 * Bot detection signals
 */
export type BotSignal =
  | 'fresh_account'
  | 'high_frequency'
  | 'follower_ratio'
  | 'default_avatar'
  | 'suspicious_name'
  | 'repetitive_content'
  | 'timing_pattern'
  | 'link_heavy'
  | 'engagement_anomaly'
  | 'coordinated_hashtags';

/**
 * Default weights for each signal
 */
const DEFAULT_WEIGHTS: Record<BotSignal, number> = {
  fresh_account: 0.15,
  high_frequency: 0.12,
  follower_ratio: 0.1,
  default_avatar: 0.08,
  suspicious_name: 0.1,
  repetitive_content: 0.15,
  timing_pattern: 0.1,
  link_heavy: 0.08,
  engagement_anomaly: 0.07,
  coordinated_hashtags: 0.05,
};

/**
 * Patterns indicating bot-like usernames
 */
const SUSPICIOUS_NAME_PATTERNS = [
  /^[a-z]+\d{4,}$/i, // name1234567
  /^[a-z]{2,3}\d{6,}$/i, // ab123456
  /^\d+[a-z]+\d+$/i, // 123name456
  /_\d{5,}$/, // anything_12345
  /^user\d+$/i, // user123
  /^[a-z]{20,}$/i, // very long random letters
];

/**
 * Content patterns indicating automated posting
 */
const AUTOMATED_CONTENT_PATTERNS = [
  /follow\s*(me|back|for follow)/i,
  /check\s*(out|my)\s*(bio|link|profile)/i,
  /DM\s*(me|for)\s*(promo|collab)/i,
  /\d+\s*followers?\s*(away|to go)/i,
  /free\s*(crypto|money|giveaway)/i,
];

export class BotFilter implements PostFilter {
  readonly name = 'bot-filter';
  private config: Required<BotFilterConfig>;
  private weights: Record<BotSignal, number>;

  /** Track posting frequency per author */
  private postingHistory = new Map<string, number[]>();
  /** Track content hashes for repetition detection */
  private contentHashes = new Map<string, Set<string>>();

  constructor(config: BotFilterConfig = {}) {
    this.config = {
      threshold: config.threshold ?? 0.6,
      weights: config.weights ?? {},
    };

    this.weights = { ...DEFAULT_WEIGHTS, ...this.config.weights };
  }

  /**
   * Process a post and return it with bot score, or null if filtered
   */
  async process(post: SocialPost): Promise<SocialPost | null> {
    const signals = this.detectSignals(post);
    const botScore = this.calculateScore(signals);

    // Record for frequency analysis
    this.recordPost(post);

    // Attach bot score to post
    const processedPost: SocialPost = {
      ...post,
      botScore,
    };

    // Filter if above threshold
    if (botScore >= this.config.threshold) {
      return null;
    }

    return processedPost;
  }

  /**
   * Get bot score without filtering
   */
  score(post: SocialPost): { score: number; signals: BotSignal[] } {
    const signals = this.detectSignals(post);
    const score = this.calculateScore(signals);
    return { score, signals };
  }

  /**
   * Detect all bot signals for a post
   */
  private detectSignals(post: SocialPost): BotSignal[] {
    const signals: BotSignal[] = [];

    // 1. Fresh account check
    if (post.author.createdAt) {
      const accountAge = Date.now() - new Date(post.author.createdAt).getTime();
      const daysSinceCreation = accountAge / (1000 * 60 * 60 * 24);
      if (daysSinceCreation < 7) {
        signals.push('fresh_account');
      }
    }

    // 2. Follower/following ratio
    const { followers, following } = post.author;
    if (followers !== undefined && following !== undefined && following > 0) {
      const ratio = followers / following;
      // Very low ratio (follows many, few followers) is suspicious
      if (ratio < 0.1 && following > 100) {
        signals.push('follower_ratio');
      }
    }

    // 3. Suspicious username patterns
    const username = post.author.name.toLowerCase();
    if (SUSPICIOUS_NAME_PATTERNS.some((pattern) => pattern.test(username))) {
      signals.push('suspicious_name');
    }

    // 4. High posting frequency
    const authorHistory = this.postingHistory.get(post.author.id);
    if (authorHistory && authorHistory.length >= 10) {
      const recentPosts = authorHistory.slice(-10);
      const timeSpan = recentPosts[recentPosts.length - 1]! - recentPosts[0]!;
      const postsPerHour = (10 / timeSpan) * 3600000;
      if (postsPerHour > 20) {
        signals.push('high_frequency');
      }
    }

    // 5. Repetitive content
    const contentHash = this.hashContent(post.content);
    const authorHashes = this.contentHashes.get(post.author.id);
    if (authorHashes?.has(contentHash)) {
      signals.push('repetitive_content');
    }

    // 6. Link-heavy content
    const linkCount = (post.content.match(/https?:\/\//g) ?? []).length;
    const wordCount = post.content.split(/\s+/).length;
    if (linkCount > 2 || (linkCount > 0 && linkCount / wordCount > 0.2)) {
      signals.push('link_heavy');
    }

    // 7. Automated content patterns
    if (AUTOMATED_CONTENT_PATTERNS.some((pattern) => pattern.test(post.content))) {
      signals.push('repetitive_content');
    }

    // 8. Engagement anomaly (very low engagement for large follower count)
    const totalEngagement =
      (post.engagement.likes ?? 0) +
      (post.engagement.reposts ?? 0) +
      (post.engagement.replies ?? 0);

    if (followers && followers > 10000 && totalEngagement < 5) {
      signals.push('engagement_anomaly');
    }

    // 9. Timing pattern (posting at exact intervals)
    if (authorHistory && authorHistory.length >= 5) {
      const intervals: number[] = [];
      for (let i = 1; i < authorHistory.length; i++) {
        intervals.push(authorHistory[i]! - authorHistory[i - 1]!);
      }
      const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
      const variance =
        intervals.reduce((sum, i) => sum + Math.pow(i - avgInterval, 2), 0) / intervals.length;
      const stdDev = Math.sqrt(variance);

      // Very low variance = suspicious regularity
      if (stdDev < avgInterval * 0.1 && avgInterval < 300000) {
        signals.push('timing_pattern');
      }
    }

    return signals;
  }

  /**
   * Calculate weighted bot score
   */
  private calculateScore(signals: BotSignal[]): number {
    let score = 0;
    for (const signal of signals) {
      score += this.weights[signal];
    }
    return Math.min(1, score);
  }

  /**
   * Record post for frequency analysis
   */
  private recordPost(post: SocialPost): void {
    const timestamp = new Date(post.timestamp).getTime();

    // Update posting history
    if (!this.postingHistory.has(post.author.id)) {
      this.postingHistory.set(post.author.id, []);
    }
    const history = this.postingHistory.get(post.author.id)!;
    history.push(timestamp);
    // Keep last 100 posts per author
    if (history.length > 100) {
      history.shift();
    }

    // Update content hashes
    if (!this.contentHashes.has(post.author.id)) {
      this.contentHashes.set(post.author.id, new Set());
    }
    const hashes = this.contentHashes.get(post.author.id)!;
    hashes.add(this.hashContent(post.content));
    // Limit hash set size
    if (hashes.size > 1000) {
      const arr = [...hashes];
      arr.splice(0, 500);
      this.contentHashes.set(post.author.id, new Set(arr));
    }
  }

  /**
   * Simple content hash for repetition detection
   */
  private hashContent(content: string): string {
    // Normalize content
    const normalized = content
      .toLowerCase()
      .replace(/https?:\/\/\S+/g, '[link]')
      .replace(/@\S+/g, '[mention]')
      .replace(/\s+/g, ' ')
      .trim();

    // Simple hash
    let hash = 0;
    for (let i = 0; i < normalized.length; i++) {
      const char = normalized.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return hash.toString(36);
  }

  /**
   * Clear tracking data (for memory management)
   */
  clearHistory(): void {
    this.postingHistory.clear();
    this.contentHashes.clear();
  }
}
