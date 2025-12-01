/**
 * RateLimiter
 *
 * Respects API rate limits to stay compliant with ToS.
 * Implements token bucket algorithm with per-source tracking.
 */

export interface RateLimitConfig {
  requestsPerSecond: number;
  burstSize?: number;
}

interface TokenBucket {
  tokens: number;
  lastRefill: number;
  config: RateLimitConfig;
}

export class RateLimiter {
  private buckets: Map<string, TokenBucket> = new Map();
  private globalConfig: RateLimitConfig;

  constructor(globalConfig: RateLimitConfig = { requestsPerSecond: 1, burstSize: 5 }) {
    this.globalConfig = globalConfig;
  }

  /**
   * Configure rate limit for a specific source
   */
  configureSource(sourceId: string, config: RateLimitConfig): void {
    this.buckets.set(sourceId, {
      tokens: config.burstSize ?? config.requestsPerSecond,
      lastRefill: Date.now(),
      config,
    });
  }

  /**
   * Check if a request can proceed
   */
  canProceed(sourceId: string): boolean {
    const bucket = this.getOrCreateBucket(sourceId);
    this.refillBucket(bucket);
    return bucket.tokens >= 1;
  }

  /**
   * Consume a token (call after successful request)
   */
  consume(sourceId: string): boolean {
    const bucket = this.getOrCreateBucket(sourceId);
    this.refillBucket(bucket);

    if (bucket.tokens >= 1) {
      bucket.tokens -= 1;
      return true;
    }

    return false;
  }

  /**
   * Get wait time until next request is allowed (in ms)
   */
  getWaitTime(sourceId: string): number {
    const bucket = this.getOrCreateBucket(sourceId);
    this.refillBucket(bucket);

    if (bucket.tokens >= 1) return 0;

    const tokensNeeded = 1 - bucket.tokens;
    const msPerToken = 1000 / bucket.config.requestsPerSecond;
    return Math.ceil(tokensNeeded * msPerToken);
  }

  /**
   * Wait until request can proceed
   */
  async waitForSlot(sourceId: string): Promise<void> {
    const waitTime = this.getWaitTime(sourceId);
    if (waitTime > 0) {
      await new Promise((resolve) => setTimeout(resolve, waitTime));
    }
    this.consume(sourceId);
  }

  /**
   * Get current token count for a source
   */
  getTokens(sourceId: string): number {
    const bucket = this.getOrCreateBucket(sourceId);
    this.refillBucket(bucket);
    return bucket.tokens;
  }

  /**
   * Reset rate limiter for a source
   */
  reset(sourceId: string): void {
    this.buckets.delete(sourceId);
  }

  /**
   * Reset all rate limiters
   */
  resetAll(): void {
    this.buckets.clear();
  }

  private getOrCreateBucket(sourceId: string): TokenBucket {
    let bucket = this.buckets.get(sourceId);

    if (!bucket) {
      bucket = {
        tokens: this.globalConfig.burstSize ?? this.globalConfig.requestsPerSecond,
        lastRefill: Date.now(),
        config: this.globalConfig,
      };
      this.buckets.set(sourceId, bucket);
    }

    return bucket;
  }

  private refillBucket(bucket: TokenBucket): void {
    const now = Date.now();
    const elapsed = now - bucket.lastRefill;
    const tokensToAdd = (elapsed / 1000) * bucket.config.requestsPerSecond;
    const maxTokens = bucket.config.burstSize ?? bucket.config.requestsPerSecond;

    bucket.tokens = Math.min(maxTokens, bucket.tokens + tokensToAdd);
    bucket.lastRefill = now;
  }
}
