/**
 * LatticeForge Security Hardening Layer
 *
 * This module provides comprehensive security controls:
 * 1. Request validation and sanitization
 * 2. Rate limiting with abuse detection
 * 3. API call protection (users cannot force calls)
 * 4. Input validation and injection prevention
 * 5. Encryption for sensitive data
 * 6. Audit logging
 *
 * Threat model: APT-level attackers targeting:
 * - API abuse (cost attacks via Anthropic calls)
 * - Injection attacks (XSS, SQL, prompt injection)
 * - Data exfiltration
 * - Unauthorized access
 */

import { headers } from 'next/headers';
import crypto from 'crypto';

// ============================================
// Request Validation & Sanitization
// ============================================

/**
 * Validate and sanitize user input
 */
export function sanitizeInput(input: unknown): string {
  if (input === null || input === undefined) return '';
  if (typeof input !== 'string') return String(input);

  // Remove potentially dangerous characters
  return input
    // Remove null bytes
    .replace(/\0/g, '')
    // Remove control characters except newlines and tabs
    .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '')
    // Limit length
    .slice(0, 10000)
    // Normalize Unicode
    .normalize('NFC');
}

/**
 * Validate preset is in allowed list
 */
export function validatePreset(preset: string): boolean {
  const VALID_PRESETS = ['global', 'nato', 'brics', 'conflict'];
  return VALID_PRESETS.includes(preset);
}

/**
 * Validate region code
 */
export function validateRegion(region: string): boolean {
  // ISO 3166-1 alpha-2 or alpha-3 codes, or "Global"
  return /^[A-Z]{2,3}$/.test(region) || region === 'Global';
}

/**
 * Check for potential injection patterns
 */
export function detectInjection(input: string): { safe: boolean; type?: string } {
  const patterns = [
    { regex: /<script/i, type: 'xss' },
    { regex: /javascript:/i, type: 'xss' },
    { regex: /on\w+\s*=/i, type: 'xss' },
    { regex: /['";]\s*(?:OR|AND|UNION|SELECT|INSERT|UPDATE|DELETE|DROP)/i, type: 'sql' },
    { regex: /\$\{.*\}/i, type: 'template' },
    { regex: /\{\{.*\}\}/i, type: 'template' },
    { regex: /<!--[\s\S]*?-->/, type: 'html_comment' },
    { regex: /\beval\s*\(/i, type: 'code_exec' },
    { regex: /\bFunction\s*\(/i, type: 'code_exec' },
    // Prompt injection patterns
    { regex: /ignore.*(?:previous|above|prior).*instructions/i, type: 'prompt_injection' },
    { regex: /disregard.*(?:system|initial).*prompt/i, type: 'prompt_injection' },
    { regex: /you are now/i, type: 'prompt_injection' },
    { regex: /new instructions:/i, type: 'prompt_injection' },
  ];

  for (const { regex, type } of patterns) {
    if (regex.test(input)) {
      return { safe: false, type };
    }
  }

  return { safe: true };
}

// ============================================
// Rate Limiting with Abuse Detection
// ============================================

interface RateLimitEntry {
  count: number;
  firstRequest: number;
  lastRequest: number;
  anomalyScore: number;
}

const rateLimitStore = new Map<string, RateLimitEntry>();

interface RateLimitConfig {
  windowMs: number;
  maxRequests: number;
  burstThreshold: number;
  burstWindowMs: number;
}

const DEFAULT_RATE_LIMIT: RateLimitConfig = {
  windowMs: 60 * 1000, // 1 minute
  maxRequests: 30,
  burstThreshold: 10, // 10 requests in burst window = anomaly
  burstWindowMs: 5 * 1000, // 5 seconds
};

/**
 * Check rate limit and detect anomalous patterns
 */
export function checkRateLimit(
  clientId: string,
  config: Partial<RateLimitConfig> = {}
): { allowed: boolean; remaining: number; anomalyScore: number; resetAt: Date } {
  const cfg = { ...DEFAULT_RATE_LIMIT, ...config };
  const now = Date.now();

  let entry = rateLimitStore.get(clientId);

  if (!entry || now - entry.firstRequest > cfg.windowMs) {
    // New window
    entry = {
      count: 0,
      firstRequest: now,
      lastRequest: now,
      anomalyScore: 0,
    };
  }

  entry.count++;
  entry.lastRequest = now;

  // Detect burst (anomaly indicator)
  const timeSinceFirst = now - entry.firstRequest;
  if (timeSinceFirst < cfg.burstWindowMs && entry.count > cfg.burstThreshold) {
    entry.anomalyScore = Math.min(1, entry.anomalyScore + 0.3);
  }

  // Detect regular timing (bot indicator)
  // TODO: Track request intervals and flag uniform timing

  rateLimitStore.set(clientId, entry);

  const remaining = Math.max(0, cfg.maxRequests - entry.count);
  const resetAt = new Date(entry.firstRequest + cfg.windowMs);

  return {
    allowed: entry.count <= cfg.maxRequests,
    remaining,
    anomalyScore: entry.anomalyScore,
    resetAt,
  };
}

/**
 * Get client identifier from request
 */
export async function getClientId(request: Request): Promise<string> {
  const headersList = await headers();

  // Try various headers for client identification
  const forwarded = headersList.get('x-forwarded-for');
  const real = headersList.get('x-real-ip');
  const vercelIp = headersList.get('x-vercel-ip');
  const cfConnecting = headersList.get('cf-connecting-ip');

  const ip = forwarded?.split(',')[0] || real || vercelIp || cfConnecting || 'unknown';

  // Include user agent for fingerprinting
  const ua = headersList.get('user-agent') || '';
  const uaHash = crypto.createHash('sha256').update(ua).digest('hex').slice(0, 8);

  return `${ip}:${uaHash}`;
}

// ============================================
// API Call Protection
// ============================================

/**
 * Verify request is from internal/cron source
 */
export async function isPrivilegedRequest(request: Request): Promise<boolean> {
  const headersList = await headers();

  // Check cron header
  const isCronWarm = headersList.get('x-cron-warm') === '1';
  const isVercelCron = headersList.get('x-vercel-cron') === '1';

  // Check internal service secret
  const internalSecret = process.env.INTERNAL_SERVICE_SECRET;
  const providedSecret = headersList.get('x-internal-service');
  const isInternalService = internalSecret && providedSecret === internalSecret;

  return isCronWarm || isVercelCron || !!isInternalService;
}

/**
 * Verify HMAC signature on request
 */
export function verifyHmacSignature(
  payload: string,
  signature: string,
  secret: string
): boolean {
  if (!secret || !signature) return false;

  try {
    const expectedSig = crypto
      .createHmac('sha256', secret)
      .update(payload)
      .digest('hex');

    // Timing-safe comparison
    return crypto.timingSafeEqual(
      Buffer.from(signature),
      Buffer.from(expectedSig)
    );
  } catch {
    return false;
  }
}

/**
 * Generate HMAC signature for request
 */
export function generateHmacSignature(payload: string, secret: string): string {
  return crypto
    .createHmac('sha256', secret)
    .update(payload)
    .digest('hex');
}

// ============================================
// Encryption for Sensitive Data
// ============================================

const ALGORITHM = 'aes-256-gcm';
const IV_LENGTH = 16;
const AUTH_TAG_LENGTH = 16;

/**
 * Encrypt sensitive data
 */
export function encryptData(plaintext: string, key: string): string {
  const keyBuffer = crypto.scryptSync(key, 'lattice-salt', 32);
  const iv = crypto.randomBytes(IV_LENGTH);

  const cipher = crypto.createCipheriv(ALGORITHM, keyBuffer, iv);
  cipher.setAAD(Buffer.from('latticeforge-v1'));

  let encrypted = cipher.update(plaintext, 'utf8', 'base64');
  encrypted += cipher.final('base64');

  const authTag = cipher.getAuthTag();

  // Format: iv:authTag:encrypted
  return `${iv.toString('base64')}:${authTag.toString('base64')}:${encrypted}`;
}

/**
 * Decrypt sensitive data
 */
export function decryptData(ciphertext: string, key: string): string | null {
  try {
    const [ivStr, authTagStr, encrypted] = ciphertext.split(':');
    if (!ivStr || !authTagStr || !encrypted) return null;

    const keyBuffer = crypto.scryptSync(key, 'lattice-salt', 32);
    const iv = Buffer.from(ivStr, 'base64');
    const authTag = Buffer.from(authTagStr, 'base64');

    const decipher = crypto.createDecipheriv(ALGORITHM, keyBuffer, iv);
    decipher.setAuthTag(authTag);
    decipher.setAAD(Buffer.from('latticeforge-v1'));

    let decrypted = decipher.update(encrypted, 'base64', 'utf8');
    decrypted += decipher.final('utf8');

    return decrypted;
  } catch {
    return null;
  }
}

/**
 * Hash data for storage (one-way)
 */
export function hashData(data: string): string {
  const salt = process.env.HASH_SALT || 'lattice-default-salt';
  return crypto
    .createHash('sha256')
    .update(data + salt)
    .digest('hex');
}

/**
 * Generate secure random token
 */
export function generateSecureToken(length: number = 32): string {
  return crypto.randomBytes(length).toString('hex');
}

// ============================================
// Audit Logging
// ============================================

interface AuditEvent {
  timestamp: string;
  eventType: string;
  clientId: string;
  action: string;
  resource?: string;
  success: boolean;
  metadata?: Record<string, unknown>;
  severity: 'info' | 'warning' | 'error' | 'critical';
}

const auditBuffer: AuditEvent[] = [];
const AUDIT_BUFFER_SIZE = 100;

/**
 * Log security audit event
 */
export function logAuditEvent(event: Omit<AuditEvent, 'timestamp'>): void {
  const auditEvent: AuditEvent = {
    ...event,
    timestamp: new Date().toISOString(),
  };

  auditBuffer.push(auditEvent);

  // Keep buffer size limited
  while (auditBuffer.length > AUDIT_BUFFER_SIZE) {
    auditBuffer.shift();
  }

  // Log to console for debugging (in production, send to logging service)
  if (event.severity === 'critical' || event.severity === 'error') {
    console.error('[SECURITY AUDIT]', JSON.stringify(auditEvent));
  } else if (process.env.NODE_ENV !== 'production') {
    console.log('[AUDIT]', JSON.stringify(auditEvent));
  }
}

/**
 * Get recent audit events
 */
export function getRecentAuditEvents(count: number = 50): AuditEvent[] {
  return auditBuffer.slice(-count);
}

// ============================================
// Request Guard Middleware
// ============================================

export interface GuardResult {
  allowed: boolean;
  error?: string;
  statusCode?: number;
  clientId: string;
  isPrivileged: boolean;
  rateLimitInfo: {
    remaining: number;
    resetAt: Date;
  };
}

/**
 * Comprehensive request guard
 */
export async function guardRequest(
  request: Request,
  options: {
    requireAuth?: boolean;
    allowPublic?: boolean;
    rateLimit?: Partial<RateLimitConfig>;
    validateBody?: (body: unknown) => boolean;
  } = {}
): Promise<GuardResult> {
  const clientId = await getClientId(request);
  const isPrivileged = await isPrivilegedRequest(request);

  // Rate limiting
  const rateLimitResult = checkRateLimit(clientId, options.rateLimit);

  if (!rateLimitResult.allowed) {
    logAuditEvent({
      eventType: 'rate_limit_exceeded',
      clientId,
      action: 'request',
      success: false,
      severity: 'warning',
      metadata: { anomalyScore: rateLimitResult.anomalyScore },
    });

    return {
      allowed: false,
      error: 'Rate limit exceeded',
      statusCode: 429,
      clientId,
      isPrivileged,
      rateLimitInfo: {
        remaining: 0,
        resetAt: rateLimitResult.resetAt,
      },
    };
  }

  // High anomaly score warning
  if (rateLimitResult.anomalyScore > 0.7) {
    logAuditEvent({
      eventType: 'high_anomaly_score',
      clientId,
      action: 'request',
      success: true,
      severity: 'warning',
      metadata: { anomalyScore: rateLimitResult.anomalyScore },
    });
  }

  // Authentication check (if required)
  if (options.requireAuth && !isPrivileged) {
    logAuditEvent({
      eventType: 'unauthorized_access',
      clientId,
      action: 'request',
      success: false,
      severity: 'warning',
    });

    return {
      allowed: false,
      error: 'Unauthorized',
      statusCode: 401,
      clientId,
      isPrivileged,
      rateLimitInfo: {
        remaining: rateLimitResult.remaining,
        resetAt: rateLimitResult.resetAt,
      },
    };
  }

  // Body validation (if provided)
  if (options.validateBody) {
    try {
      const body = await request.clone().json();
      if (!options.validateBody(body)) {
        logAuditEvent({
          eventType: 'invalid_body',
          clientId,
          action: 'request',
          success: false,
          severity: 'warning',
        });

        return {
          allowed: false,
          error: 'Invalid request body',
          statusCode: 400,
          clientId,
          isPrivileged,
          rateLimitInfo: {
            remaining: rateLimitResult.remaining,
            resetAt: rateLimitResult.resetAt,
          },
        };
      }
    } catch {
      // JSON parse error or other issue
    }
  }

  return {
    allowed: true,
    clientId,
    isPrivileged,
    rateLimitInfo: {
      remaining: rateLimitResult.remaining,
      resetAt: rateLimitResult.resetAt,
    },
  };
}

// ============================================
// Content Security Headers
// ============================================

export const SECURITY_HEADERS = {
  'Content-Security-Policy': [
    "default-src 'self'",
    "script-src 'self' 'unsafe-eval' 'unsafe-inline' https://vercel.live",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob: https:",
    "font-src 'self' data:",
    "connect-src 'self' https://*.supabase.co https://*.vercel-insights.com wss://*.supabase.co",
    "frame-ancestors 'none'",
    "base-uri 'self'",
    "form-action 'self'",
  ].join('; '),
  'X-Frame-Options': 'DENY',
  'X-Content-Type-Options': 'nosniff',
  'X-XSS-Protection': '1; mode=block',
  'Referrer-Policy': 'strict-origin-when-cross-origin',
  'Permissions-Policy': 'camera=(), microphone=(), geolocation=(), interest-cohort=()',
  'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
};

/**
 * Apply security headers to response
 */
export function applySecurityHeaders(response: Response): Response {
  const newHeaders = new Headers(response.headers);

  for (const [key, value] of Object.entries(SECURITY_HEADERS)) {
    newHeaders.set(key, value);
  }

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: newHeaders,
  });
}
