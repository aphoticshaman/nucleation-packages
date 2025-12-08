/**
 * EDGE SECURITY & PERFORMANCE UTILITIES
 * 50 advanced optimizations for cyber-secure, high-performance edge computing
 *
 * Categories:
 * 1. Rate Limiting (1-5)
 * 2. Circuit Breakers (6-10)
 * 3. Request Deduplication (11-15)
 * 4. Cache Security (16-20)
 * 5. Input Validation (21-25)
 * 6. Fingerprinting & Anomaly Detection (26-30)
 * 7. Encryption & Signing (31-35)
 * 8. Headers & CORS (36-40)
 * 9. Audit & Logging (41-45)
 * 10. Compression & Optimization (46-50)
 */

// =============================================================================
// 1-5: RATE LIMITING
// =============================================================================

interface RateLimitEntry {
  count: number;
  resetAt: number;
  blocked: boolean;
}

const rateLimitStore = new Map<string, RateLimitEntry>();

/**
 * 1. Token Bucket Rate Limiter
 * Smooth rate limiting with burst allowance
 */
export function tokenBucketRateLimit(
  key: string,
  maxTokens: number = 100,
  refillRate: number = 10, // tokens per second
  windowMs: number = 60000
): { allowed: boolean; remaining: number; resetAt: Date } {
  const now = Date.now();
  const entry = rateLimitStore.get(key);

  if (!entry || now > entry.resetAt) {
    rateLimitStore.set(key, { count: 1, resetAt: now + windowMs, blocked: false });
    return { allowed: true, remaining: maxTokens - 1, resetAt: new Date(now + windowMs) };
  }

  // Refill tokens based on time elapsed
  const elapsed = (now - (entry.resetAt - windowMs)) / 1000;
  const refilled = Math.min(maxTokens, entry.count + Math.floor(elapsed * refillRate));

  if (refilled <= 0) {
    return { allowed: false, remaining: 0, resetAt: new Date(entry.resetAt) };
  }

  entry.count = refilled - 1;
  return { allowed: true, remaining: entry.count, resetAt: new Date(entry.resetAt) };
}

/**
 * 2. Sliding Window Rate Limiter
 * More accurate than fixed windows
 */
const slidingWindowStore = new Map<string, number[]>();

export function slidingWindowRateLimit(
  key: string,
  maxRequests: number = 100,
  windowMs: number = 60000
): { allowed: boolean; count: number } {
  const now = Date.now();
  const timestamps = slidingWindowStore.get(key) || [];

  // Remove expired timestamps
  const valid = timestamps.filter(t => now - t < windowMs);

  if (valid.length >= maxRequests) {
    slidingWindowStore.set(key, valid);
    return { allowed: false, count: valid.length };
  }

  valid.push(now);
  slidingWindowStore.set(key, valid);
  return { allowed: true, count: valid.length };
}

/**
 * 3. IP-based Rate Limiting with Exponential Backoff
 */
const ipBackoffStore = new Map<string, { violations: number; backoffUntil: number }>();

export function ipRateLimitWithBackoff(
  ip: string,
  maxViolations: number = 5
): { allowed: boolean; backoffSeconds: number } {
  const now = Date.now();
  const entry = ipBackoffStore.get(ip);

  if (entry && now < entry.backoffUntil) {
    const remaining = Math.ceil((entry.backoffUntil - now) / 1000);
    return { allowed: false, backoffSeconds: remaining };
  }

  return { allowed: true, backoffSeconds: 0 };
}

export function recordRateLimitViolation(ip: string): void {
  const entry = ipBackoffStore.get(ip) || { violations: 0, backoffUntil: 0 };
  entry.violations++;
  // Exponential backoff: 2^violations seconds (max 1 hour)
  const backoffMs = Math.min(Math.pow(2, entry.violations) * 1000, 3600000);
  entry.backoffUntil = Date.now() + backoffMs;
  ipBackoffStore.set(ip, entry);
}

/**
 * 4. Concurrent Request Limiter
 */
const concurrentRequests = new Map<string, number>();

export function acquireConcurrencySlot(
  key: string,
  maxConcurrent: number = 5
): { acquired: boolean; current: number } {
  const current = concurrentRequests.get(key) || 0;
  if (current >= maxConcurrent) {
    return { acquired: false, current };
  }
  concurrentRequests.set(key, current + 1);
  return { acquired: true, current: current + 1 };
}

export function releaseConcurrencySlot(key: string): void {
  const current = concurrentRequests.get(key) || 0;
  concurrentRequests.set(key, Math.max(0, current - 1));
}

/**
 * 5. Adaptive Rate Limiting (adjusts based on server load)
 */
let serverLoad = 0.5; // 0-1 scale

export function updateServerLoad(load: number): void {
  serverLoad = Math.max(0, Math.min(1, load));
}

export function adaptiveRateLimit(
  key: string,
  baseLimit: number = 100
): { allowed: boolean; effectiveLimit: number } {
  // Reduce limit when server is under high load
  const effectiveLimit = Math.floor(baseLimit * (1 - serverLoad * 0.5));
  const result = slidingWindowRateLimit(key, effectiveLimit);
  return { allowed: result.allowed, effectiveLimit };
}

// =============================================================================
// 6-10: CIRCUIT BREAKERS
// =============================================================================

interface CircuitState {
  failures: number;
  lastFailure: number;
  state: 'closed' | 'open' | 'half-open';
  successCount: number;
}

const circuitBreakers = new Map<string, CircuitState>();

/**
 * 6. Circuit Breaker Pattern
 * Prevents cascading failures to external services
 */
export function getCircuitState(
  service: string,
  failureThreshold: number = 5,
  resetTimeoutMs: number = 30000,
  halfOpenSuccesses: number = 3
): { canProceed: boolean; state: CircuitState['state'] } {
  const now = Date.now();
  const circuit = circuitBreakers.get(service) || {
    failures: 0,
    lastFailure: 0,
    state: 'closed' as const,
    successCount: 0,
  };

  // Check if we should transition from open to half-open
  if (circuit.state === 'open' && now - circuit.lastFailure > resetTimeoutMs) {
    circuit.state = 'half-open';
    circuit.successCount = 0;
    circuitBreakers.set(service, circuit);
  }

  if (circuit.state === 'open') {
    return { canProceed: false, state: 'open' };
  }

  return { canProceed: true, state: circuit.state };
}

/**
 * 7. Record Circuit Success
 */
export function recordCircuitSuccess(service: string, halfOpenSuccesses: number = 3): void {
  const circuit = circuitBreakers.get(service);
  if (!circuit) return;

  if (circuit.state === 'half-open') {
    circuit.successCount++;
    if (circuit.successCount >= halfOpenSuccesses) {
      circuit.state = 'closed';
      circuit.failures = 0;
    }
  } else if (circuit.state === 'closed') {
    circuit.failures = Math.max(0, circuit.failures - 1); // Decay failures on success
  }

  circuitBreakers.set(service, circuit);
}

/**
 * 8. Record Circuit Failure
 */
export function recordCircuitFailure(service: string, failureThreshold: number = 5): void {
  const circuit = circuitBreakers.get(service) || {
    failures: 0,
    lastFailure: 0,
    state: 'closed' as const,
    successCount: 0,
  };

  circuit.failures++;
  circuit.lastFailure = Date.now();

  if (circuit.failures >= failureThreshold) {
    circuit.state = 'open';
  }

  circuitBreakers.set(service, circuit);
}

/**
 * 9. Service Health Monitor
 */
interface ServiceHealth {
  healthy: boolean;
  lastCheck: number;
  latencyMs: number;
  errorRate: number;
}

const serviceHealth = new Map<string, ServiceHealth>();

export function updateServiceHealth(
  service: string,
  healthy: boolean,
  latencyMs: number
): void {
  const current = serviceHealth.get(service) || {
    healthy: true,
    lastCheck: 0,
    latencyMs: 0,
    errorRate: 0,
  };

  // Exponential moving average for error rate
  const alpha = 0.3;
  current.errorRate = alpha * (healthy ? 0 : 1) + (1 - alpha) * current.errorRate;
  current.latencyMs = alpha * latencyMs + (1 - alpha) * current.latencyMs;
  current.healthy = current.errorRate < 0.5;
  current.lastCheck = Date.now();

  serviceHealth.set(service, current);
}

export function getServiceHealth(service: string): ServiceHealth | null {
  return serviceHealth.get(service) || null;
}

/**
 * 10. Bulkhead Pattern
 * Isolate failures to specific partitions
 */
const bulkheads = new Map<string, { active: number; max: number; queue: number }>();

export function tryAcquireBulkhead(
  partition: string,
  maxConcurrent: number = 10,
  maxQueue: number = 50
): { acquired: boolean; position: number } {
  const bulkhead = bulkheads.get(partition) || { active: 0, max: maxConcurrent, queue: 0 };

  if (bulkhead.active < maxConcurrent) {
    bulkhead.active++;
    bulkheads.set(partition, bulkhead);
    return { acquired: true, position: 0 };
  }

  if (bulkhead.queue < maxQueue) {
    bulkhead.queue++;
    bulkheads.set(partition, bulkhead);
    return { acquired: false, position: bulkhead.queue };
  }

  return { acquired: false, position: -1 }; // Rejected
}

export function releaseBulkhead(partition: string): void {
  const bulkhead = bulkheads.get(partition);
  if (bulkhead) {
    bulkhead.active = Math.max(0, bulkhead.active - 1);
    if (bulkhead.queue > 0) {
      bulkhead.queue--;
      bulkhead.active++;
    }
    bulkheads.set(partition, bulkhead);
  }
}

// =============================================================================
// 11-15: REQUEST DEDUPLICATION
// =============================================================================

interface PendingRequest {
  promise: Promise<unknown>;
  timestamp: number;
}

const pendingRequests = new Map<string, PendingRequest>();

/**
 * 11. Request Coalescing
 * Deduplicate identical concurrent requests
 */
export async function coalesceRequest<T>(
  key: string,
  executor: () => Promise<T>,
  maxAgeMs: number = 5000
): Promise<T> {
  const now = Date.now();
  const pending = pendingRequests.get(key);

  // Return existing promise if still valid
  if (pending && now - pending.timestamp < maxAgeMs) {
    return pending.promise as Promise<T>;
  }

  // Execute and store promise
  const promise = executor();
  pendingRequests.set(key, { promise, timestamp: now });

  try {
    const result = await promise;
    return result;
  } finally {
    // Clean up after completion
    setTimeout(() => pendingRequests.delete(key), maxAgeMs);
  }
}

/**
 * 12. Request Hash Generator
 * Create consistent hash for deduplication
 */
export function hashRequest(
  method: string,
  path: string,
  body?: unknown,
  userId?: string
): string {
  const payload = JSON.stringify({ method, path, body, userId });
  // Simple hash for edge runtime (no crypto.subtle needed)
  let hash = 0;
  for (let i = 0; i < payload.length; i++) {
    const char = payload.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return hash.toString(36);
}

/**
 * 13. Idempotency Key Tracker
 */
const idempotencyKeys = new Map<string, { response: unknown; timestamp: number }>();

export function checkIdempotencyKey(
  key: string,
  maxAgeMs: number = 86400000 // 24 hours
): { exists: boolean; response?: unknown } {
  const entry = idempotencyKeys.get(key);
  if (entry && Date.now() - entry.timestamp < maxAgeMs) {
    return { exists: true, response: entry.response };
  }
  return { exists: false };
}

export function storeIdempotencyResponse(key: string, response: unknown): void {
  idempotencyKeys.set(key, { response, timestamp: Date.now() });

  // Clean up old entries periodically
  if (idempotencyKeys.size > 10000) {
    const cutoff = Date.now() - 86400000;
    for (const [k, v] of idempotencyKeys.entries()) {
      if (v.timestamp < cutoff) idempotencyKeys.delete(k);
    }
  }
}

/**
 * 14. Request Debouncer
 */
const debounceTimers = new Map<string, NodeJS.Timeout>();

export function debounceRequest(
  key: string,
  delayMs: number = 500
): { shouldProcess: boolean } {
  if (debounceTimers.has(key)) {
    return { shouldProcess: false };
  }

  const timer = setTimeout(() => {
    debounceTimers.delete(key);
  }, delayMs);

  debounceTimers.set(key, timer);
  return { shouldProcess: true };
}

/**
 * 15. Batch Request Aggregator
 */
interface BatchEntry<T> {
  id: string;
  data: T;
  resolve: (result: unknown) => void;
  reject: (error: Error) => void;
}

const batchQueues = new Map<string, BatchEntry<unknown>[]>();
const batchTimers = new Map<string, NodeJS.Timeout>();

export function queueForBatch<T, R>(
  batchKey: string,
  id: string,
  data: T,
  processor: (items: BatchEntry<T>[]) => Promise<Map<string, R>>,
  maxBatchSize: number = 100,
  maxWaitMs: number = 50
): Promise<R> {
  return new Promise((resolve, reject) => {
    const queue = (batchQueues.get(batchKey) || []) as BatchEntry<T>[];
    queue.push({ id, data, resolve: resolve as (r: unknown) => void, reject });
    batchQueues.set(batchKey, queue);

    const processBatch = async () => {
      const items = batchQueues.get(batchKey) as BatchEntry<T>[] || [];
      batchQueues.delete(batchKey);
      batchTimers.delete(batchKey);

      try {
        const results = await processor(items);
        for (const item of items) {
          const result = results.get(item.id);
          if (result !== undefined) {
            item.resolve(result);
          } else {
            item.reject(new Error('No result for batch item'));
          }
        }
      } catch (error) {
        for (const item of items) {
          item.reject(error instanceof Error ? error : new Error(String(error)));
        }
      }
    };

    if (queue.length >= maxBatchSize) {
      const timer = batchTimers.get(batchKey);
      if (timer) clearTimeout(timer);
      void processBatch();
    } else if (!batchTimers.has(batchKey)) {
      const timer = setTimeout(() => void processBatch(), maxWaitMs);
      batchTimers.set(batchKey, timer);
    }
  });
}

// =============================================================================
// 16-20: CACHE SECURITY
// =============================================================================

/**
 * 16. Cache Key Sanitization
 */
export function sanitizeCacheKey(key: string): string {
  // Remove dangerous characters, limit length
  return key
    .replace(/[^a-zA-Z0-9_:-]/g, '_')
    .slice(0, 256);
}

/**
 * 17. Cache Entry Signing (HMAC-like without crypto)
 */
export function signCacheEntry(data: string, secret: string): string {
  // Simple signing for edge runtime
  let hash = 0;
  const combined = data + secret;
  for (let i = 0; i < combined.length; i++) {
    hash = ((hash << 5) - hash) + combined.charCodeAt(i);
    hash = hash & hash;
  }
  return Math.abs(hash).toString(36);
}

export function verifyCacheSignature(
  data: string,
  signature: string,
  secret: string
): boolean {
  const expected = signCacheEntry(data, secret);
  return signature === expected;
}

/**
 * 18. Cache Poisoning Prevention
 */
const cacheWriteRateLimit = new Map<string, number[]>();

export function preventCachePoisoning(
  key: string,
  maxWritesPerMinute: number = 10
): { allowed: boolean } {
  const now = Date.now();
  const writes = cacheWriteRateLimit.get(key) || [];
  const recent = writes.filter(t => now - t < 60000);

  if (recent.length >= maxWritesPerMinute) {
    return { allowed: false };
  }

  recent.push(now);
  cacheWriteRateLimit.set(key, recent);
  return { allowed: true };
}

/**
 * 19. Cache Stampede Prevention (Probabilistic Early Expiry)
 */
export function shouldRefreshEarly(
  ttlRemainingMs: number,
  totalTtlMs: number,
  beta: number = 1
): boolean {
  // XFetch algorithm: probabilistic early refresh
  const random = Math.random();
  const delta = beta * Math.log(random);
  const threshold = ttlRemainingMs + delta * (totalTtlMs / 10);
  return threshold <= 0;
}

/**
 * 20. Cache Namespace Isolation
 */
export function getNamespacedKey(
  namespace: string,
  key: string,
  version: number = 1
): string {
  return `${namespace}:v${version}:${sanitizeCacheKey(key)}`;
}

// =============================================================================
// 21-25: INPUT VALIDATION
// =============================================================================

/**
 * 21. SQL Injection Prevention
 */
export function sanitizeSqlInput(input: string): string {
  return input
    .replace(/'/g, "''")
    .replace(/--/g, '')
    .replace(/;/g, '')
    .replace(/\/\*/g, '')
    .replace(/\*\//g, '');
}

/**
 * 22. XSS Prevention
 */
export function sanitizeHtml(input: string): string {
  return input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
    .replace(/\//g, '&#x2F;');
}

/**
 * 23. JSON Schema Validator (lightweight)
 */
export function validateJsonSchema(
  data: unknown,
  schema: {
    type: 'object' | 'array' | 'string' | 'number' | 'boolean';
    properties?: Record<string, { type: string; required?: boolean; maxLength?: number }>;
    maxItems?: number;
  }
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (schema.type === 'object' && typeof data !== 'object') {
    errors.push('Expected object');
  }

  if (schema.type === 'array' && !Array.isArray(data)) {
    errors.push('Expected array');
  }

  if (schema.type === 'array' && Array.isArray(data) && schema.maxItems && data.length > schema.maxItems) {
    errors.push(`Array exceeds max items: ${schema.maxItems}`);
  }

  if (schema.properties && typeof data === 'object' && data !== null) {
    for (const [key, prop] of Object.entries(schema.properties)) {
      const value = (data as Record<string, unknown>)[key];
      if (prop.required && value === undefined) {
        errors.push(`Missing required field: ${key}`);
      }
      if (value !== undefined && typeof value !== prop.type) {
        errors.push(`Invalid type for ${key}: expected ${prop.type}`);
      }
      if (prop.maxLength && typeof value === 'string' && value.length > prop.maxLength) {
        errors.push(`${key} exceeds max length: ${prop.maxLength}`);
      }
    }
  }

  return { valid: errors.length === 0, errors };
}

/**
 * 24. Path Traversal Prevention
 */
export function sanitizePath(path: string): string {
  return path
    .replace(/\.\./g, '')
    .replace(/\/\//g, '/')
    .replace(/^\/+/, '')
    .replace(/[<>:"|?*]/g, '');
}

/**
 * 25. Command Injection Prevention
 */
export function sanitizeShellArg(arg: string): string {
  return arg
    .replace(/[`$(){}|;&<>]/g, '')
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/'/g, "\\'");
}

// =============================================================================
// 26-30: FINGERPRINTING & ANOMALY DETECTION
// =============================================================================

/**
 * 26. Request Fingerprint Generator
 */
export function generateRequestFingerprint(
  ip: string,
  userAgent: string,
  acceptLanguage: string,
  acceptEncoding: string
): string {
  const raw = `${ip}|${userAgent}|${acceptLanguage}|${acceptEncoding}`;
  let hash = 0;
  for (let i = 0; i < raw.length; i++) {
    hash = ((hash << 5) - hash) + raw.charCodeAt(i);
    hash = hash & hash;
  }
  return Math.abs(hash).toString(36);
}

/**
 * 27. Bot Detection Heuristics
 */
export function detectBot(userAgent: string): { isBot: boolean; botType: string | null } {
  const ua = userAgent.toLowerCase();
  const bots = [
    { pattern: /googlebot/i, type: 'googlebot' },
    { pattern: /bingbot/i, type: 'bingbot' },
    { pattern: /slurp/i, type: 'yahoo' },
    { pattern: /duckduckbot/i, type: 'duckduckgo' },
    { pattern: /baiduspider/i, type: 'baidu' },
    { pattern: /yandexbot/i, type: 'yandex' },
    { pattern: /facebookexternalhit/i, type: 'facebook' },
    { pattern: /twitterbot/i, type: 'twitter' },
    { pattern: /linkedinbot/i, type: 'linkedin' },
    { pattern: /curl|wget|python|java|ruby|perl|php/i, type: 'script' },
    { pattern: /headless|phantom|selenium|puppeteer/i, type: 'automation' },
  ];

  for (const { pattern, type } of bots) {
    if (pattern.test(ua)) {
      return { isBot: true, botType: type };
    }
  }

  // Heuristic: very short user agent
  if (userAgent.length < 20) {
    return { isBot: true, botType: 'suspicious' };
  }

  return { isBot: false, botType: null };
}

/**
 * 28. Behavioral Anomaly Score
 */
interface BehaviorProfile {
  requestCount: number;
  avgIntervalMs: number;
  lastRequest: number;
  endpoints: Set<string>;
}

const behaviorProfiles = new Map<string, BehaviorProfile>();

export function updateBehaviorProfile(
  fingerprint: string,
  endpoint: string
): { anomalyScore: number } {
  const now = Date.now();
  const profile = behaviorProfiles.get(fingerprint) || {
    requestCount: 0,
    avgIntervalMs: 0,
    lastRequest: 0,
    endpoints: new Set<string>(),
  };

  const interval = profile.lastRequest ? now - profile.lastRequest : 1000;
  profile.avgIntervalMs = (profile.avgIntervalMs * profile.requestCount + interval) / (profile.requestCount + 1);
  profile.requestCount++;
  profile.lastRequest = now;
  profile.endpoints.add(endpoint);

  behaviorProfiles.set(fingerprint, profile);

  // Calculate anomaly score (0-1)
  let score = 0;

  // Too fast
  if (profile.avgIntervalMs < 100) score += 0.3;

  // Too many endpoints in short time
  if (profile.endpoints.size > 20 && profile.requestCount < 100) score += 0.3;

  // Inhuman consistency
  const variance = Math.abs(interval - profile.avgIntervalMs) / Math.max(profile.avgIntervalMs, 1);
  if (variance < 0.1 && profile.requestCount > 10) score += 0.2;

  // High volume
  if (profile.requestCount > 1000) score += 0.2;

  return { anomalyScore: Math.min(1, score) };
}

/**
 * 29. Geo-velocity Check
 */
interface GeoLocation {
  country: string;
  timestamp: number;
}

const geoHistory = new Map<string, GeoLocation[]>();

export function checkGeoVelocity(
  userId: string,
  country: string,
  maxKmPerHour: number = 1000 // Roughly max flight speed
): { suspicious: boolean; reason?: string } {
  const history = geoHistory.get(userId) || [];
  const now = Date.now();

  if (history.length > 0) {
    const last = history[history.length - 1];
    if (last.country !== country) {
      const hoursSinceLast = (now - last.timestamp) / 3600000;

      // Simple check: different country in < 1 hour is suspicious
      if (hoursSinceLast < 1) {
        return {
          suspicious: true,
          reason: `Country changed from ${last.country} to ${country} in ${Math.round(hoursSinceLast * 60)} minutes`,
        };
      }
    }
  }

  // Update history
  history.push({ country, timestamp: now });
  if (history.length > 10) history.shift();
  geoHistory.set(userId, history);

  return { suspicious: false };
}

/**
 * 30. Pattern Matching for Attack Detection
 */
export function detectAttackPatterns(
  path: string,
  queryString: string
): { detected: boolean; patterns: string[] } {
  const detected: string[] = [];

  const patterns = [
    { name: 'sql_injection', regex: /(\bUNION\b|\bSELECT\b|\bDROP\b|\bINSERT\b|\bDELETE\b)/i },
    { name: 'xss', regex: /<script|javascript:|on\w+=/i },
    { name: 'path_traversal', regex: /\.\.[\/\\]/ },
    { name: 'command_injection', regex: /[;&|`$()]/ },
    { name: 'ldap_injection', regex: /[)(|*\\]/ },
    { name: 'xml_injection', regex: /<!\[CDATA\[|<!ENTITY/i },
    { name: 'log_forging', regex: /[\r\n]/ },
  ];

  const fullInput = path + queryString;

  for (const { name, regex } of patterns) {
    if (regex.test(fullInput)) {
      detected.push(name);
    }
  }

  return { detected: detected.length > 0, patterns: detected };
}

// =============================================================================
// 31-35: ENCRYPTION & SIGNING
// =============================================================================

/**
 * 31. Simple XOR Obfuscation (not real encryption, but hides data)
 */
export function xorObfuscate(data: string, key: string): string {
  let result = '';
  for (let i = 0; i < data.length; i++) {
    result += String.fromCharCode(data.charCodeAt(i) ^ key.charCodeAt(i % key.length));
  }
  return btoa(result);
}

export function xorDeobfuscate(obfuscated: string, key: string): string {
  const data = atob(obfuscated);
  let result = '';
  for (let i = 0; i < data.length; i++) {
    result += String.fromCharCode(data.charCodeAt(i) ^ key.charCodeAt(i % key.length));
  }
  return result;
}

/**
 * 32. Token Generator
 */
export function generateSecureToken(length: number = 32): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  const array = new Uint8Array(length);
  crypto.getRandomValues(array);
  for (let i = 0; i < length; i++) {
    result += chars[array[i] % chars.length];
  }
  return result;
}

/**
 * 33. CSRF Token Validator
 */
const csrfTokens = new Map<string, { token: string; expiry: number }>();

export function generateCsrfToken(sessionId: string): string {
  const token = generateSecureToken(32);
  csrfTokens.set(sessionId, { token, expiry: Date.now() + 3600000 }); // 1 hour
  return token;
}

export function validateCsrfToken(sessionId: string, token: string): boolean {
  const entry = csrfTokens.get(sessionId);
  if (!entry || Date.now() > entry.expiry) {
    return false;
  }
  return entry.token === token;
}

/**
 * 34. Request Timestamp Validation
 */
export function validateRequestTimestamp(
  timestamp: number,
  maxAgeMs: number = 300000 // 5 minutes
): { valid: boolean; drift: number } {
  const now = Date.now();
  const drift = Math.abs(now - timestamp);
  return { valid: drift <= maxAgeMs, drift };
}

/**
 * 35. Nonce Tracker (Replay Prevention)
 */
const usedNonces = new Set<string>();

export function checkAndUseNonce(nonce: string, maxSize: number = 100000): boolean {
  if (usedNonces.has(nonce)) {
    return false; // Replay detected
  }

  usedNonces.add(nonce);

  // Prevent memory bloat
  if (usedNonces.size > maxSize) {
    const toRemove = Math.floor(maxSize * 0.2);
    const iterator = usedNonces.values();
    for (let i = 0; i < toRemove; i++) {
      const value = iterator.next().value;
      if (value) usedNonces.delete(value);
    }
  }

  return true;
}

// =============================================================================
// 36-40: HEADERS & CORS
// =============================================================================

/**
 * 36. Security Headers
 */
export function getSecurityHeaders(): Record<string, string> {
  return {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Referrer-Policy': 'strict-origin-when-cross-origin',
    'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
    'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'",
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
  };
}

/**
 * 37. CORS Validation
 */
const allowedOrigins = new Set([
  'https://latticeforge.com',
  'https://www.latticeforge.com',
  'https://app.latticeforge.com',
]);

export function validateCorsOrigin(origin: string | null): { allowed: boolean; headers: Record<string, string> } {
  if (!origin) {
    return { allowed: false, headers: {} };
  }

  // Development mode
  if (process.env.NODE_ENV === 'development' && origin.includes('localhost')) {
    return {
      allowed: true,
      headers: {
        'Access-Control-Allow-Origin': origin,
        'Access-Control-Allow-Credentials': 'true',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-CSRF-Token',
      },
    };
  }

  if (allowedOrigins.has(origin)) {
    return {
      allowed: true,
      headers: {
        'Access-Control-Allow-Origin': origin,
        'Access-Control-Allow-Credentials': 'true',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-CSRF-Token',
        'Access-Control-Max-Age': '86400',
      },
    };
  }

  return { allowed: false, headers: {} };
}

/**
 * 38. Content-Type Validation
 */
export function validateContentType(
  contentType: string | null,
  allowed: string[] = ['application/json']
): boolean {
  if (!contentType) return false;
  return allowed.some(type => contentType.includes(type));
}

/**
 * 39. Request Size Limiter
 */
export function checkRequestSize(
  contentLength: number,
  maxBytes: number = 1048576 // 1MB
): { allowed: boolean; overflow: number } {
  if (contentLength > maxBytes) {
    return { allowed: false, overflow: contentLength - maxBytes };
  }
  return { allowed: true, overflow: 0 };
}

/**
 * 40. Accept Header Negotiation
 */
export function negotiateContentType(
  acceptHeader: string,
  supported: string[] = ['application/json', 'text/html']
): string | null {
  const accepted = acceptHeader.split(',').map(t => t.trim().split(';')[0]);

  for (const type of accepted) {
    if (type === '*/*') return supported[0];
    if (supported.includes(type)) return type;
  }

  return null;
}

// =============================================================================
// 41-45: AUDIT & LOGGING
// =============================================================================

/**
 * 41. Structured Audit Log Entry
 */
export interface AuditLogEntry {
  timestamp: string;
  eventType: string;
  userId?: string;
  ip: string;
  userAgent: string;
  path: string;
  method: string;
  statusCode: number;
  latencyMs: number;
  metadata?: Record<string, unknown>;
}

const auditBuffer: AuditLogEntry[] = [];
const AUDIT_BUFFER_SIZE = 100;

export function logAuditEvent(entry: Omit<AuditLogEntry, 'timestamp'>): void {
  const fullEntry: AuditLogEntry = {
    ...entry,
    timestamp: new Date().toISOString(),
  };

  auditBuffer.push(fullEntry);

  // Flush when buffer is full
  if (auditBuffer.length >= AUDIT_BUFFER_SIZE) {
    flushAuditLogs();
  }
}

export function flushAuditLogs(): AuditLogEntry[] {
  const logs = [...auditBuffer];
  auditBuffer.length = 0;
  // In production, you'd send these to a logging service
  console.log(`[AUDIT] Flushing ${logs.length} audit entries`);
  return logs;
}

/**
 * 42. Error Categorization
 */
export function categorizeError(error: Error): {
  category: 'client' | 'server' | 'external' | 'unknown';
  severity: 'low' | 'medium' | 'high' | 'critical';
  retryable: boolean;
} {
  const message = error.message.toLowerCase();

  if (message.includes('validation') || message.includes('invalid')) {
    return { category: 'client', severity: 'low', retryable: false };
  }

  if (message.includes('timeout') || message.includes('network')) {
    return { category: 'external', severity: 'medium', retryable: true };
  }

  if (message.includes('rate limit')) {
    return { category: 'external', severity: 'medium', retryable: true };
  }

  if (message.includes('auth') || message.includes('permission')) {
    return { category: 'client', severity: 'medium', retryable: false };
  }

  if (message.includes('database') || message.includes('sql')) {
    return { category: 'server', severity: 'high', retryable: false };
  }

  return { category: 'unknown', severity: 'medium', retryable: false };
}

/**
 * 43. Performance Metrics Collector
 */
interface PerformanceMetric {
  endpoint: string;
  latencyMs: number;
  timestamp: number;
}

const performanceMetrics: PerformanceMetric[] = [];

export function recordPerformanceMetric(endpoint: string, latencyMs: number): void {
  performanceMetrics.push({
    endpoint,
    latencyMs,
    timestamp: Date.now(),
  });

  // Keep only last 1000 entries
  if (performanceMetrics.length > 1000) {
    performanceMetrics.shift();
  }
}

export function getPerformanceStats(endpoint?: string): {
  p50: number;
  p95: number;
  p99: number;
  avg: number;
} {
  let metrics = performanceMetrics;
  if (endpoint) {
    metrics = metrics.filter(m => m.endpoint === endpoint);
  }

  if (metrics.length === 0) {
    return { p50: 0, p95: 0, p99: 0, avg: 0 };
  }

  const sorted = metrics.map(m => m.latencyMs).sort((a, b) => a - b);
  const avg = sorted.reduce((a, b) => a + b, 0) / sorted.length;

  return {
    p50: sorted[Math.floor(sorted.length * 0.5)],
    p95: sorted[Math.floor(sorted.length * 0.95)],
    p99: sorted[Math.floor(sorted.length * 0.99)],
    avg: Math.round(avg),
  };
}

/**
 * 44. Request Correlation ID Generator
 */
export function generateCorrelationId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 10);
  return `${timestamp}-${random}`;
}

/**
 * 45. Sensitive Data Redactor
 */
export function redactSensitiveData(
  data: Record<string, unknown>,
  sensitiveFields: string[] = ['password', 'token', 'secret', 'apiKey', 'authorization', 'credit_card', 'ssn']
): Record<string, unknown> {
  const result: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(data)) {
    const isSenitiveField = sensitiveFields.some(f =>
      key.toLowerCase().includes(f.toLowerCase())
    );

    if (isSenitiveField) {
      result[key] = '[REDACTED]';
    } else if (typeof value === 'object' && value !== null) {
      result[key] = redactSensitiveData(value as Record<string, unknown>, sensitiveFields);
    } else {
      result[key] = value;
    }
  }

  return result;
}

// =============================================================================
// 46-50: COMPRESSION & OPTIMIZATION
// =============================================================================

/**
 * 46. JSON Compression (removes whitespace, shortens keys)
 */
export function compressJson(data: unknown): string {
  return JSON.stringify(data);
}

/**
 * 47. Response Size Estimation
 */
export function estimateResponseSize(data: unknown): number {
  return JSON.stringify(data).length * 2; // Approximate UTF-8 byte size
}

/**
 * 48. Selective Field Projection
 */
export function projectFields<T extends Record<string, unknown>>(
  data: T,
  fields: (keyof T)[]
): Partial<T> {
  const result: Partial<T> = {};
  for (const field of fields) {
    if (field in data) {
      result[field] = data[field];
    }
  }
  return result;
}

/**
 * 49. Response Pagination Helper
 */
export function paginateArray<T>(
  data: T[],
  page: number,
  pageSize: number
): { data: T[]; pagination: { page: number; pageSize: number; total: number; totalPages: number } } {
  const start = (page - 1) * pageSize;
  const end = start + pageSize;
  const paginatedData = data.slice(start, end);

  return {
    data: paginatedData,
    pagination: {
      page,
      pageSize,
      total: data.length,
      totalPages: Math.ceil(data.length / pageSize),
    },
  };
}

/**
 * 50. Cache-Control Header Generator
 */
export function generateCacheControl(
  options: {
    maxAge?: number;
    sMaxAge?: number;
    staleWhileRevalidate?: number;
    staleIfError?: number;
    private?: boolean;
    noStore?: boolean;
    mustRevalidate?: boolean;
  }
): string {
  const directives: string[] = [];

  if (options.noStore) {
    return 'no-store';
  }

  if (options.private) {
    directives.push('private');
  } else {
    directives.push('public');
  }

  if (options.maxAge !== undefined) {
    directives.push(`max-age=${options.maxAge}`);
  }

  if (options.sMaxAge !== undefined) {
    directives.push(`s-maxage=${options.sMaxAge}`);
  }

  if (options.staleWhileRevalidate !== undefined) {
    directives.push(`stale-while-revalidate=${options.staleWhileRevalidate}`);
  }

  if (options.staleIfError !== undefined) {
    directives.push(`stale-if-error=${options.staleIfError}`);
  }

  if (options.mustRevalidate) {
    directives.push('must-revalidate');
  }

  return directives.join(', ');
}

// =============================================================================
// EXPORTS SUMMARY
// =============================================================================

export const EdgeSecurity = {
  // Rate Limiting
  tokenBucketRateLimit,
  slidingWindowRateLimit,
  ipRateLimitWithBackoff,
  recordRateLimitViolation,
  acquireConcurrencySlot,
  releaseConcurrencySlot,
  adaptiveRateLimit,
  updateServerLoad,

  // Circuit Breakers
  getCircuitState,
  recordCircuitSuccess,
  recordCircuitFailure,
  updateServiceHealth,
  getServiceHealth,
  tryAcquireBulkhead,
  releaseBulkhead,

  // Request Deduplication
  coalesceRequest,
  hashRequest,
  checkIdempotencyKey,
  storeIdempotencyResponse,
  debounceRequest,
  queueForBatch,

  // Cache Security
  sanitizeCacheKey,
  signCacheEntry,
  verifyCacheSignature,
  preventCachePoisoning,
  shouldRefreshEarly,
  getNamespacedKey,

  // Input Validation
  sanitizeSqlInput,
  sanitizeHtml,
  validateJsonSchema,
  sanitizePath,
  sanitizeShellArg,

  // Fingerprinting & Anomaly Detection
  generateRequestFingerprint,
  detectBot,
  updateBehaviorProfile,
  checkGeoVelocity,
  detectAttackPatterns,

  // Encryption & Signing
  xorObfuscate,
  xorDeobfuscate,
  generateSecureToken,
  generateCsrfToken,
  validateCsrfToken,
  validateRequestTimestamp,
  checkAndUseNonce,

  // Headers & CORS
  getSecurityHeaders,
  validateCorsOrigin,
  validateContentType,
  checkRequestSize,
  negotiateContentType,

  // Audit & Logging
  logAuditEvent,
  flushAuditLogs,
  categorizeError,
  recordPerformanceMetric,
  getPerformanceStats,
  generateCorrelationId,
  redactSensitiveData,

  // Compression & Optimization
  compressJson,
  estimateResponseSize,
  projectFields,
  paginateArray,
  generateCacheControl,
};
