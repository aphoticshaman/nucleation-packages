/**
 * Security Middleware
 *
 * Fort Knox-grade security for enterprise API.
 * Multiple layers of protection.
 */

import type { Context, Next } from 'hono';
import { createHash, randomBytes, timingSafeEqual } from 'node:crypto';

/**
 * Security headers middleware
 * Sets comprehensive security headers
 */
export function securityHeaders() {
  return async (c: Context, next: Next) => {
    // Generate unique request ID
    const requestId = generateRequestId();
    c.set('requestId', requestId);

    await next();

    // Security headers
    c.header('X-Request-ID', requestId);
    c.header('X-Content-Type-Options', 'nosniff');
    c.header('X-Frame-Options', 'DENY');
    c.header('X-XSS-Protection', '1; mode=block');
    c.header('Strict-Transport-Security', 'max-age=31536000; includeSubDomains; preload');
    c.header('Content-Security-Policy', "default-src 'none'; frame-ancestors 'none'");
    c.header('Cache-Control', 'no-store, no-cache, must-revalidate, private');
    c.header('Pragma', 'no-cache');
    c.header('X-Permitted-Cross-Domain-Policies', 'none');
    c.header('Referrer-Policy', 'no-referrer');
    c.header(
      'Permissions-Policy',
      'accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()'
    );

    // Remove server identification
    c.header('Server', 'LatticeForge');
    c.header('X-Powered-By', '');
  };
}

/**
 * CORS middleware - locked down
 * Only allows specific origins in production
 */
export function corsMiddleware(allowedOrigins: string[] = []) {
  return async (c: Context, next: Next) => {
    const origin = c.req.header('Origin');

    // In production, strictly validate origin
    if (allowedOrigins.length > 0) {
      if (!origin || !allowedOrigins.includes(origin)) {
        return c.json(
          {
            success: false,
            error: {
              code: 'CORS_REJECTED',
              message: 'Origin not allowed',
            },
          },
          403
        );
      }
      c.header('Access-Control-Allow-Origin', origin);
    } else {
      // Development mode - still restrictive
      c.header('Access-Control-Allow-Origin', origin ?? '*');
    }

    c.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    c.header(
      'Access-Control-Allow-Headers',
      'Authorization, Content-Type, X-Request-ID, X-API-Key'
    );
    c.header('Access-Control-Max-Age', '86400');
    c.header('Access-Control-Allow-Credentials', 'true');

    if (c.req.method === 'OPTIONS') {
      return c.text('', 204);
    }

    await next();
  };
}

/**
 * Request validation middleware
 * Validates request structure and content
 */
export function validateRequest() {
  return async (c: Context, next: Next) => {
    // Check content type for POST/PUT/PATCH
    if (['POST', 'PUT', 'PATCH'].includes(c.req.method)) {
      const contentType = c.req.header('Content-Type');
      if (!contentType?.includes('application/json')) {
        return c.json(
          {
            success: false,
            error: {
              code: 'INVALID_CONTENT_TYPE',
              message: 'Content-Type must be application/json',
            },
          },
          415
        );
      }
    }

    // Check request size (max 1MB)
    const contentLength = c.req.header('Content-Length');
    if (contentLength && parseInt(contentLength, 10) > 1048576) {
      return c.json(
        {
          success: false,
          error: {
            code: 'PAYLOAD_TOO_LARGE',
            message: 'Request body exceeds 1MB limit',
          },
        },
        413
      );
    }

    await next();
  };
}

/**
 * IP allowlist middleware
 * For government/enterprise clients with IP restrictions
 */
export function ipAllowlist(allowedIps: string[]) {
  return async (c: Context, next: Next) => {
    if (allowedIps.length === 0) {
      await next();
      return;
    }

    const clientIp = getClientIp(c);

    if (!allowedIps.includes(clientIp)) {
      return c.json(
        {
          success: false,
          error: {
            code: 'IP_BLOCKED',
            message: 'Access denied from this IP address',
          },
        },
        403
      );
    }

    await next();
  };
}

/**
 * Request logging middleware
 * Logs all requests for audit trail
 */
export function requestLogger(logFn: (entry: RequestLogEntry) => void) {
  return async (c: Context, next: Next) => {
    const start = performance.now();

    await next();

    const duration = performance.now() - start;
    const requestId = c.get('requestId') ?? 'unknown';

    const entry: RequestLogEntry = {
      requestId,
      timestamp: new Date().toISOString(),
      method: c.req.method,
      path: c.req.path,
      status: c.res.status,
      duration: Math.round(duration * 100) / 100,
      ip: getClientIp(c),
      userAgent: c.req.header('User-Agent') ?? 'unknown',
    };

    logFn(entry);
  };
}

export interface RequestLogEntry {
  requestId: string;
  timestamp: string;
  method: string;
  path: string;
  status: number;
  duration: number;
  ip: string;
  userAgent: string;
  clientId?: string;
}

/**
 * Timing-safe API key comparison
 * Prevents timing attacks
 */
export function secureCompare(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;
  }

  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);

  return timingSafeEqual(bufA, bufB);
}

/**
 * Hash API key for storage
 */
export function hashApiKey(key: string): string {
  return createHash('sha256').update(key).digest('hex');
}

/**
 * Generate secure API key
 */
export function generateApiKey(): string {
  const prefix = 'lf_live_';
  const random = randomBytes(32).toString('base64url');
  return `${prefix}${random}`;
}

/**
 * Generate request ID
 */
export function generateRequestId(): string {
  return `req_${Date.now()}_${randomBytes(8).toString('hex')}`;
}

/**
 * Get client IP from request
 */
export function getClientIp(c: Context): string {
  // Check common proxy headers
  const forwarded = c.req.header('X-Forwarded-For');
  if (forwarded) {
    return forwarded.split(',')[0].trim();
  }

  const realIp = c.req.header('X-Real-IP');
  if (realIp) {
    return realIp;
  }

  // Fallback to connection IP (may not exist in all environments)
  return '0.0.0.0';
}

/**
 * Sanitize error for response
 * Never leak internal details
 */
export function sanitizeError(error: unknown): { code: string; message: string } {
  if (error instanceof Error) {
    // Only return generic messages in production
    if (process.env.NODE_ENV === 'production') {
      return {
        code: 'INTERNAL_ERROR',
        message: 'An internal error occurred',
      };
    }

    return {
      code: 'INTERNAL_ERROR',
      message: error.message,
    };
  }

  return {
    code: 'UNKNOWN_ERROR',
    message: 'An unknown error occurred',
  };
}
