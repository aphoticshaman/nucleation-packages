/**
 * UPSTASH CONFIGURATION
 *
 * What Upstash IS: Serverless Redis & Kafka
 * What Upstash is NOT: Art asset storage (use Vercel Blob or S3 for that)
 *
 * YOUR USE CASES:
 * 1. Rate limiting API endpoints
 * 2. Session caching (reduce Supabase queries)
 * 3. API response caching (financial data, expensive queries)
 * 4. Real-time features (pub/sub for live updates)
 * 5. Job queues (background processing)
 *
 * BILLING: Pay-per-request through Vercel
 * - ~$0.20 per 100K requests
 * - Typical usage: $5-20/month for your scale
 */

import { Ratelimit } from '@upstash/ratelimit';
import { Redis } from '@upstash/redis';

// ============================================
// REDIS CLIENT SETUP
// ============================================

// Initialize from environment variables
// Set these in Vercel: UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN
const redis = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL!,
  token: process.env.UPSTASH_REDIS_REST_TOKEN!,
});

// ============================================
// RATE LIMITING
// ============================================

/**
 * Rate limiter configurations by endpoint type
 */
export const rateLimiters = {
  // Authentication - strict limits
  auth: new Ratelimit({
    redis,
    limiter: Ratelimit.slidingWindow(5, '15 m'), // 5 requests per 15 minutes
    analytics: true,
    prefix: 'ratelimit:auth',
  }),

  // Login attempts - very strict
  login: new Ratelimit({
    redis,
    limiter: Ratelimit.slidingWindow(5, '15 m'),
    analytics: true,
    prefix: 'ratelimit:login',
  }),

  // Password reset - strict
  passwordReset: new Ratelimit({
    redis,
    limiter: Ratelimit.slidingWindow(3, '1 h'), // 3 per hour
    analytics: true,
    prefix: 'ratelimit:password-reset',
  }),

  // API by tier
  api: {
    explorer: new Ratelimit({
      redis,
      limiter: Ratelimit.slidingWindow(100, '1 h'),
      analytics: true,
      prefix: 'ratelimit:api:explorer',
    }),
    analyst: new Ratelimit({
      redis,
      limiter: Ratelimit.slidingWindow(1000, '1 h'),
      analytics: true,
      prefix: 'ratelimit:api:analyst',
    }),
    strategist: new Ratelimit({
      redis,
      limiter: Ratelimit.slidingWindow(5000, '1 h'),
      analytics: true,
      prefix: 'ratelimit:api:strategist',
    }),
    architect: new Ratelimit({
      redis,
      limiter: Ratelimit.slidingWindow(20000, '1 h'),
      analytics: true,
      prefix: 'ratelimit:api:architect',
    }),
  },

  // General API - fallback
  general: new Ratelimit({
    redis,
    limiter: Ratelimit.slidingWindow(60, '1 m'), // 60 per minute
    analytics: true,
    prefix: 'ratelimit:general',
  }),

  // Exports - expensive operations
  export: new Ratelimit({
    redis,
    limiter: Ratelimit.slidingWindow(10, '1 h'), // 10 per hour
    analytics: true,
    prefix: 'ratelimit:export',
  }),

  // AI/LLM calls - expensive
  ai: new Ratelimit({
    redis,
    limiter: Ratelimit.slidingWindow(20, '1 h'), // 20 per hour
    analytics: true,
    prefix: 'ratelimit:ai',
  }),
};

/**
 * Check rate limit for a given identifier
 */
export async function checkRateLimit(
  limiterType: keyof typeof rateLimiters | 'api',
  identifier: string,
  tier?: 'explorer' | 'analyst' | 'strategist' | 'architect'
): Promise<{
  success: boolean;
  limit: number;
  remaining: number;
  reset: number;
}> {
  let limiter: Ratelimit;

  if (limiterType === 'api' && tier) {
    limiter = rateLimiters.api[tier];
  } else if (limiterType in rateLimiters && limiterType !== 'api') {
    limiter = rateLimiters[limiterType as keyof Omit<typeof rateLimiters, 'api'>];
  } else {
    limiter = rateLimiters.general;
  }

  const result = await limiter.limit(identifier);

  return {
    success: result.success,
    limit: result.limit,
    remaining: result.remaining,
    reset: result.reset,
  };
}

// ============================================
// CACHING
// ============================================

/**
 * Cache configuration by data type
 */
export const CACHE_TTL = {
  // User data - short TTL, user expects real-time
  userSession: 60 * 5, // 5 minutes
  userProfile: 60 * 15, // 15 minutes

  // Dashboard data - medium TTL
  dashboardConfig: 60 * 10, // 10 minutes
  dashboardData: 60 * 5, // 5 minutes

  // Financial data - varies by source
  stockPrice: 60 * 1, // 1 minute (real-time-ish)
  economicIndicator: 60 * 60, // 1 hour (updates less frequently)
  historicalData: 60 * 60 * 24, // 24 hours (doesn't change)

  // Intelligence data - medium TTL
  briefing: 60 * 30, // 30 minutes
  riskScore: 60 * 15, // 15 minutes

  // Static/computed data - long TTL
  geoData: 60 * 60 * 24, // 24 hours
  aggregations: 60 * 60, // 1 hour
};

/**
 * Get cached value or fetch from source
 */
export async function getCached<T>(
  key: string,
  fetcher: () => Promise<T>,
  ttl: number = CACHE_TTL.dashboardData
): Promise<T> {
  // Try to get from cache
  const cached = await redis.get<T>(key);
  if (cached !== null) {
    return cached;
  }

  // Fetch from source
  const value = await fetcher();

  // Store in cache
  await redis.set(key, value, { ex: ttl });

  return value;
}

/**
 * Invalidate cache by key or pattern
 */
export async function invalidateCache(keyOrPattern: string): Promise<void> {
  if (keyOrPattern.includes('*')) {
    // Pattern-based invalidation
    const keys = await redis.keys(keyOrPattern);
    if (keys.length > 0) {
      await redis.del(...keys);
    }
  } else {
    // Single key invalidation
    await redis.del(keyOrPattern);
  }
}

/**
 * Cache key generators
 */
export const cacheKeys = {
  userSession: (userId: string) => `session:${userId}`,
  userProfile: (userId: string) => `profile:${userId}`,
  dashboard: (dashboardId: string) => `dashboard:${dashboardId}`,
  dashboardData: (dashboardId: string, dataType: string) =>
    `dashboard:${dashboardId}:data:${dataType}`,
  financialData: (source: string, symbol: string) =>
    `financial:${source}:${symbol}`,
  briefing: (briefingId: string) => `briefing:${briefingId}`,
  riskScore: (countryCode: string) => `risk:${countryCode}`,
  geoData: (type: string) => `geo:${type}`,
};

// ============================================
// SESSION MANAGEMENT
// ============================================

/**
 * Store session data
 */
export async function setSession(
  sessionId: string,
  data: Record<string, unknown>,
  ttl: number = 60 * 60 * 24 * 7 // 7 days
): Promise<void> {
  await redis.set(`session:${sessionId}`, data, { ex: ttl });
}

/**
 * Get session data
 */
export async function getSession(
  sessionId: string
): Promise<Record<string, unknown> | null> {
  return redis.get(`session:${sessionId}`);
}

/**
 * Delete session
 */
export async function deleteSession(sessionId: string): Promise<void> {
  await redis.del(`session:${sessionId}`);
}

/**
 * Extend session TTL
 */
export async function extendSession(
  sessionId: string,
  ttl: number = 60 * 60 * 24 * 7
): Promise<void> {
  await redis.expire(`session:${sessionId}`, ttl);
}

// ============================================
// REAL-TIME FEATURES
// ============================================

/**
 * Publish message to channel (for real-time updates)
 */
export async function publishUpdate(
  channel: string,
  message: Record<string, unknown>
): Promise<void> {
  // Note: Upstash Redis supports pub/sub but serverless
  // For real-time, consider Supabase Realtime or Ably
  await redis.publish(channel, JSON.stringify(message));
}

/**
 * Track online users (for presence)
 */
export async function trackPresence(
  userId: string,
  status: 'online' | 'away' | 'offline'
): Promise<void> {
  const key = `presence:${userId}`;
  if (status === 'offline') {
    await redis.del(key);
  } else {
    await redis.set(key, status, { ex: 300 }); // 5 minute expiry
  }
}

/**
 * Get online users count
 */
export async function getOnlineCount(): Promise<number> {
  const keys = await redis.keys('presence:*');
  return keys.length;
}

// ============================================
// JOB QUEUE (Simple implementation)
// ============================================

/**
 * Add job to queue
 */
export async function enqueueJob(
  queue: string,
  job: Record<string, unknown>
): Promise<string> {
  const jobId = `job:${Date.now()}:${Math.random().toString(36).slice(2)}`;
  await redis.lpush(`queue:${queue}`, JSON.stringify({ id: jobId, ...job }));
  return jobId;
}

/**
 * Process next job from queue
 */
export async function dequeueJob(
  queue: string
): Promise<Record<string, unknown> | null> {
  const job = await redis.rpop(`queue:${queue}`);
  if (job) {
    return JSON.parse(job);
  }
  return null;
}

/**
 * Get queue length
 */
export async function getQueueLength(queue: string): Promise<number> {
  return redis.llen(`queue:${queue}`);
}

// ============================================
// ANALYTICS HELPERS
// ============================================

/**
 * Increment counter (for analytics)
 */
export async function incrementCounter(
  key: string,
  amount: number = 1
): Promise<number> {
  return redis.incrby(`counter:${key}`, amount);
}

/**
 * Track event (simple analytics)
 */
export async function trackEvent(
  event: string,
  properties: Record<string, unknown>
): Promise<void> {
  const timestamp = Date.now();
  await redis.zadd(`events:${event}`, {
    score: timestamp,
    member: JSON.stringify({ ...properties, timestamp }),
  });

  // Keep only last 10000 events per type
  await redis.zremrangebyrank(`events:${event}`, 0, -10001);
}

/**
 * Get events in time range
 */
export async function getEvents(
  event: string,
  startTime: number,
  endTime: number = Date.now()
): Promise<Record<string, unknown>[]> {
  const results = await redis.zrange(`events:${event}`, startTime, endTime, {
    byScore: true,
  });
  return results.map((r) => JSON.parse(r as string));
}

// ============================================
// HEALTH CHECK
// ============================================

export async function checkRedisHealth(): Promise<{
  healthy: boolean;
  latencyMs: number;
  error?: string;
}> {
  const start = Date.now();
  try {
    await redis.ping();
    return {
      healthy: true,
      latencyMs: Date.now() - start,
    };
  } catch (error) {
    return {
      healthy: false,
      latencyMs: Date.now() - start,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

// ============================================
// EXPORT
// ============================================

export { redis };
