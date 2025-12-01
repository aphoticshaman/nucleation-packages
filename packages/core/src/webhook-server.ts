/**
 * Secure webhook server for receiving detector data.
 *
 * This module provides a secure HTTP server for receiving data
 * from external sources with proper authentication and rate limiting.
 */

import { createServer, type IncomingMessage, type ServerResponse, type Server } from 'node:http';
import { createHmac, timingSafeEqual } from 'node:crypto';
import { NucleationError } from './validation.js';

/**
 * Authentication configuration
 */
export interface WebhookAuth {
  /** Authentication type */
  type: 'bearer' | 'hmac' | 'none';
  /** Secret token or key */
  secret?: string;
  /** Header name for HMAC signature (default: 'x-signature') */
  signatureHeader?: string;
}

/**
 * Rate limiting configuration
 */
export interface RateLimitConfig {
  /** Maximum requests per window */
  maxRequests: number;
  /** Window duration in milliseconds */
  windowMs: number;
}

/**
 * Webhook server configuration
 */
export interface WebhookServerConfig {
  /** Port to listen on */
  port?: number;
  /** Host to bind to (default: '127.0.0.1' for security) */
  host?: string;
  /** Authentication configuration */
  auth?: WebhookAuth;
  /** Rate limiting configuration */
  rateLimit?: RateLimitConfig;
  /** Maximum request body size in bytes (default: 1MB) */
  maxBodySize?: number;
  /** Request timeout in milliseconds (default: 30000) */
  timeout?: number;
  /** Custom data extractor function */
  extract?: (data: unknown) => number;
  /** Called on every update */
  onUpdate?: (state: unknown) => void;
  /** Called on warning level */
  onWarning?: (state: unknown) => void;
  /** Called on critical level */
  onCritical?: (state: unknown) => void;
}

/**
 * Rate limiter using sliding window
 */
class RateLimiter {
  private requests = new Map<string, number[]>();
  private readonly maxRequests: number;
  private readonly windowMs: number;

  constructor(config: RateLimitConfig) {
    this.maxRequests = config.maxRequests;
    this.windowMs = config.windowMs;
  }

  /**
   * Check if a request is allowed
   *
   * @param clientId - Client identifier (IP address)
   * @returns Whether the request is allowed
   */
  isAllowed(clientId: string): boolean {
    const now = Date.now();
    const windowStart = now - this.windowMs;

    // Get existing timestamps or create new array
    let timestamps = this.requests.get(clientId) ?? [];

    // Filter to only include requests within the window
    timestamps = timestamps.filter((ts) => ts > windowStart);

    if (timestamps.length >= this.maxRequests) {
      return false;
    }

    timestamps.push(now);
    this.requests.set(clientId, timestamps);

    return true;
  }

  /**
   * Clear old entries to prevent memory leaks
   */
  cleanup(): void {
    const now = Date.now();
    const windowStart = now - this.windowMs;

    for (const [clientId, timestamps] of this.requests.entries()) {
      const valid = timestamps.filter((ts) => ts > windowStart);
      if (valid.length === 0) {
        this.requests.delete(clientId);
      } else {
        this.requests.set(clientId, valid);
      }
    }
  }
}

/**
 * Verify HMAC signature
 */
function verifyHmacSignature(payload: string, signature: string, secret: string): boolean {
  const expectedSignature = createHmac('sha256', secret).update(payload).digest('hex');

  // Use timing-safe comparison to prevent timing attacks
  try {
    return timingSafeEqual(Buffer.from(signature), Buffer.from(expectedSignature));
  } catch {
    return false;
  }
}

/**
 * Get client IP address from request
 */
function getClientIp(req: IncomingMessage): string {
  // Check X-Forwarded-For header (for reverse proxies)
  const forwarded = req.headers['x-forwarded-for'];
  if (typeof forwarded === 'string') {
    return forwarded.split(',')[0]?.trim() ?? 'unknown';
  }
  return req.socket.remoteAddress ?? 'unknown';
}

/**
 * Create a secure webhook server for detector data ingestion.
 *
 * @example
 * ```typescript
 * import { createSecureWebhookServer } from '@nucleation/core';
 * import { RegimeDetector } from 'regime-shift';
 *
 * const detector = new RegimeDetector();
 * await detector.init();
 *
 * const server = createSecureWebhookServer(detector, {
 *   port: 8080,
 *   auth: {
 *     type: 'bearer',
 *     secret: process.env.WEBHOOK_SECRET,
 *   },
 *   rateLimit: {
 *     maxRequests: 100,
 *     windowMs: 60000, // 1 minute
 *   },
 *   onWarning: (state) => console.log('Warning:', state),
 * });
 *
 * await server.start();
 * ```
 */
export function createSecureWebhookServer<
  TDetector extends { update: (value: number) => unknown; current: () => unknown },
>(
  detector: TDetector,
  config: WebhookServerConfig = {}
): {
  start: () => Promise<Server>;
  stop: () => Promise<void>;
} {
  const port = config.port ?? 8080;
  const host = config.host ?? '127.0.0.1'; // Default to localhost for security
  const maxBodySize = config.maxBodySize ?? 1024 * 1024; // 1MB
  const timeout = config.timeout ?? 30000;

  const auth = config.auth ?? { type: 'none' };
  const rateLimiter = config.rateLimit ? new RateLimiter(config.rateLimit) : null;

  let server: Server | null = null;
  let cleanupInterval: NodeJS.Timeout | null = null;

  /**
   * Authenticate a request
   */
  function authenticate(req: IncomingMessage, body: string): boolean {
    if (auth.type === 'none') {
      return true;
    }

    if (auth.type === 'bearer') {
      const authHeader = req.headers.authorization;
      if (!authHeader?.startsWith('Bearer ')) {
        return false;
      }
      const token = authHeader.slice(7);
      // Use timing-safe comparison
      try {
        return timingSafeEqual(Buffer.from(token), Buffer.from(auth.secret ?? ''));
      } catch {
        return false;
      }
    }

    if (auth.type === 'hmac') {
      const signatureHeader = auth.signatureHeader ?? 'x-signature';
      const signature = req.headers[signatureHeader.toLowerCase()];
      if (typeof signature !== 'string') {
        return false;
      }
      return verifyHmacSignature(body, signature, auth.secret ?? '');
    }

    return false;
  }

  /**
   * Send JSON response
   */
  function sendJson(res: ServerResponse, statusCode: number, data: unknown): void {
    res.writeHead(statusCode, {
      'Content-Type': 'application/json',
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY',
    });
    res.end(JSON.stringify(data));
  }

  /**
   * Handle incoming request
   */
  async function handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
    // Set timeout
    res.setTimeout(timeout, () => {
      sendJson(res, 408, { error: 'Request timeout' });
    });

    // Check rate limit
    if (rateLimiter) {
      const clientIp = getClientIp(req);
      if (!rateLimiter.isAllowed(clientIp)) {
        sendJson(res, 429, { error: 'Too many requests' });
        return;
      }
    }

    // Handle different methods
    if (req.method === 'GET' && req.url === '/health') {
      sendJson(res, 200, { status: 'ok' });
      return;
    }

    if (req.method === 'GET') {
      const state = detector.current();
      sendJson(res, 200, state);
      return;
    }

    if (req.method !== 'POST') {
      sendJson(res, 405, { error: 'Method not allowed' });
      return;
    }

    // Read body with size limit
    let body = '';
    let bodySize = 0;

    try {
      for await (const chunk of req) {
        bodySize += (chunk as Buffer).length;
        if (bodySize > maxBodySize) {
          sendJson(res, 413, { error: 'Request body too large' });
          return;
        }
        body += chunk;
      }
    } catch {
      sendJson(res, 400, { error: 'Failed to read request body' });
      return;
    }

    // Authenticate
    if (!authenticate(req, body)) {
      sendJson(res, 401, { error: 'Unauthorized' });
      return;
    }

    // Parse and process
    try {
      const data: unknown = JSON.parse(body);

      // Extract value
      let value: number;
      if (config.extract) {
        value = config.extract(data);
      } else if (
        typeof data === 'object' &&
        data !== null &&
        'value' in data &&
        typeof (data as { value: unknown }).value === 'number'
      ) {
        value = (data as { value: number }).value;
      } else {
        sendJson(res, 400, { error: 'Missing or invalid "value" field' });
        return;
      }

      // Validate value
      if (!Number.isFinite(value)) {
        sendJson(res, 400, { error: 'Value must be a finite number' });
        return;
      }

      // Update detector
      const state = detector.update(value);

      // Call callbacks
      config.onUpdate?.(state);

      // Check for warning/critical levels
      const stateObj = state as { levelNumeric?: number };
      if (typeof stateObj.levelNumeric === 'number') {
        if (stateObj.levelNumeric >= 2) {
          config.onWarning?.(state);
        }
        if (stateObj.levelNumeric >= 3) {
          config.onCritical?.(state);
        }
      }

      sendJson(res, 200, state);
    } catch (error) {
      if (error instanceof SyntaxError) {
        sendJson(res, 400, { error: 'Invalid JSON' });
      } else {
        sendJson(res, 500, {
          error: 'Internal server error',
          message: error instanceof Error ? error.message : 'Unknown error',
        });
      }
    }
  }

  return {
    async start(): Promise<Server> {
      if (server) {
        throw new NucleationError('Server already started', 'ALREADY_STARTED');
      }

      // Warn if no auth configured
      if (auth.type === 'none') {
        console.warn(
          'WARNING: Webhook server started without authentication. ' +
            'This is insecure for production use.'
        );
      }

      // Warn if binding to all interfaces
      if (host === '0.0.0.0') {
        console.warn(
          'WARNING: Webhook server bound to all interfaces. ' +
            'Consider using a reverse proxy with TLS.'
        );
      }

      return new Promise((resolve, reject) => {
        server = createServer((req, res) => {
          handleRequest(req, res).catch((error) => {
            console.error('Webhook server error:', error);
            if (!res.headersSent) {
              sendJson(res, 500, { error: 'Internal server error' });
            }
          });
        });

        server.on('error', reject);

        server.listen(port, host, () => {
          console.log(`Nucleation webhook server listening on ${host}:${port}`);

          // Start cleanup interval for rate limiter
          if (rateLimiter) {
            cleanupInterval = setInterval(() => rateLimiter.cleanup(), 60000);
          }

          resolve(server!);
        });
      });
    },

    async stop(): Promise<void> {
      if (cleanupInterval) {
        clearInterval(cleanupInterval);
        cleanupInterval = null;
      }

      if (!server) {
        return;
      }

      return new Promise((resolve, reject) => {
        server!.close((error) => {
          server = null;
          if (error) {
            reject(error);
          } else {
            resolve();
          }
        });
      });
    },
  };
}
