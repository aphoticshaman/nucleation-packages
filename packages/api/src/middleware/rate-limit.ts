/**
 * Rate Limiting Middleware
 *
 * Tier-based rate limiting with sliding window.
 * Protects the API from abuse while allowing legitimate high-volume use.
 */

import type { Context, Next } from 'hono';
import type { AuthContext, RateLimitConfig } from '../types.js';

interface RateLimitState {
  requests: number[];
  dailyRequests: number;
  dailyReset: number;
}

// In-memory rate limit state (use Redis in production)
const rateLimitState = new Map<string, RateLimitState>();

/**
 * Rate limiting middleware
 */
export function rateLimit() {
  return async (c: Context, next: Next) => {
    const auth = c.get('auth') as AuthContext | undefined;

    if (!auth) {
      // Unauthenticated requests get very strict limits
      return applyAnonymousRateLimit(c, next);
    }

    const clientId = auth.client.id;
    const config = auth.client.rateLimit;
    const now = Date.now();

    // Get or create rate limit state
    let state = rateLimitState.get(clientId);
    if (!state) {
      state = {
        requests: [],
        dailyRequests: 0,
        dailyReset: getNextMidnight(),
      };
      rateLimitState.set(clientId, state);
    }

    // Reset daily counter if needed
    if (now >= state.dailyReset) {
      state.dailyRequests = 0;
      state.dailyReset = getNextMidnight();
    }

    // Clean old requests from sliding window (last minute)
    const windowStart = now - 60000;
    state.requests = state.requests.filter((t) => t > windowStart);

    // Check rate limits
    const minuteRequests = state.requests.length;
    const dailyRequests = state.dailyRequests;

    // Check burst limit (requests in last second)
    const lastSecond = now - 1000;
    const burstRequests = state.requests.filter((t) => t > lastSecond).length;

    if (burstRequests >= config.burstLimit) {
      return rateLimitExceeded(c, 'BURST_LIMIT', config, state, 1);
    }

    if (minuteRequests >= config.requestsPerMinute) {
      const resetIn = Math.ceil((state.requests[0] + 60000 - now) / 1000);
      return rateLimitExceeded(c, 'MINUTE_LIMIT', config, state, resetIn);
    }

    if (dailyRequests >= config.requestsPerDay) {
      const resetIn = Math.ceil((state.dailyReset - now) / 1000);
      return rateLimitExceeded(c, 'DAILY_LIMIT', config, state, resetIn);
    }

    // Record this request
    state.requests.push(now);
    state.dailyRequests++;

    // Set rate limit headers
    const remaining = config.requestsPerMinute - state.requests.length;
    const reset = Math.ceil((state.requests[0] + 60000 - now) / 1000);

    c.header('X-RateLimit-Limit', String(config.requestsPerMinute));
    c.header('X-RateLimit-Remaining', String(Math.max(0, remaining)));
    c.header('X-RateLimit-Reset', String(Math.max(0, reset)));
    c.header('X-RateLimit-Daily-Limit', String(config.requestsPerDay));
    c.header('X-RateLimit-Daily-Remaining', String(config.requestsPerDay - state.dailyRequests));

    // Store rate limit info for response meta
    c.set('rateLimit', {
      remaining: Math.max(0, remaining),
      reset: Math.max(0, reset),
    });

    await next();
  };
}

/**
 * Anonymous rate limiting (very strict)
 */
async function applyAnonymousRateLimit(c: Context, next: Next) {
  const ip = getIp(c);
  const now = Date.now();

  let state = rateLimitState.get(`anon:${ip}`);
  if (!state) {
    state = {
      requests: [],
      dailyRequests: 0,
      dailyReset: getNextMidnight(),
    };
    rateLimitState.set(`anon:${ip}`, state);
  }

  // Clean sliding window
  const windowStart = now - 60000;
  state.requests = state.requests.filter((t) => t > windowStart);

  // Anonymous: 5 requests per minute, 50 per day
  if (state.requests.length >= 5) {
    c.header('Retry-After', '60');
    return c.json(
      {
        success: false,
        error: {
          code: 'RATE_LIMIT_EXCEEDED',
          message: 'Anonymous rate limit exceeded. Authenticate for higher limits.',
        },
      },
      429
    );
  }

  state.requests.push(now);
  state.dailyRequests++;

  await next();
}

/**
 * Return rate limit exceeded response
 */
function rateLimitExceeded(
  c: Context,
  code: string,
  config: RateLimitConfig,
  state: RateLimitState,
  resetIn: number
) {
  c.header('Retry-After', String(resetIn));
  c.header('X-RateLimit-Limit', String(config.requestsPerMinute));
  c.header('X-RateLimit-Remaining', '0');
  c.header('X-RateLimit-Reset', String(resetIn));

  const messages: Record<string, string> = {
    BURST_LIMIT: 'Too many requests in quick succession. Slow down.',
    MINUTE_LIMIT: `Rate limit exceeded. ${config.requestsPerMinute} requests per minute allowed.`,
    DAILY_LIMIT: `Daily limit exceeded. ${config.requestsPerDay} requests per day allowed.`,
  };

  return c.json(
    {
      success: false,
      error: {
        code: 'RATE_LIMIT_EXCEEDED',
        message: messages[code] ?? 'Rate limit exceeded',
        details: {
          type: code,
          limit: code === 'DAILY_LIMIT' ? config.requestsPerDay : config.requestsPerMinute,
          resetIn,
        },
      },
    },
    429
  );
}

function getNextMidnight(): number {
  const now = new Date();
  const midnight = new Date(now);
  midnight.setUTCHours(24, 0, 0, 0);
  return midnight.getTime();
}

function getIp(c: Context): string {
  return c.req.header('X-Forwarded-For')?.split(',')[0].trim() ??
         c.req.header('X-Real-IP') ??
         '0.0.0.0';
}

/**
 * Clear rate limit state for a client (admin use)
 */
export function clearRateLimit(clientId: string): void {
  rateLimitState.delete(clientId);
}

/**
 * Get rate limit state for a client (admin use)
 */
export function getRateLimitState(clientId: string): RateLimitState | undefined {
  return rateLimitState.get(clientId);
}
