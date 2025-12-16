/**
 * SECURITY LAYER
 *
 * Input sanitization, rate limiting, and API key protection.
 */

import { createClient } from '@supabase/supabase-js';

const SECURITY_CONFIG = {
  rateLimits: {
    consumer: 20,
    pro: 100,
    enterprise: 500,
  },
  maxInputLength: 1000,
  maxContextKeys: 50,
  blockedPatterns: [
    /ignore\s+(previous|above|all)\s+instructions/i,
    /disregard\s+(previous|above|all)/i,
    /forget\s+(everything|all|previous)/i,
    /you\s+are\s+now/i,
    /pretend\s+(to\s+be|you\s+are)/i,
    /act\s+as\s+(if|a)/i,
    /system\s*:\s*/i,
    /\[INST\]/i,
    /\[\/INST\]/i,
    /<\|im_start\|>/i,
    /<\|im_end\|>/i,
    /api[_\s]?key/i,
    /bearer\s+token/i,
    /authorization/i,
  ],
  suspiciousPatterns: [
    /what\s+is\s+your\s+(system|original)\s+prompt/i,
    /reveal\s+(your|the)\s+(instructions|prompt)/i,
    /bypass/i,
    /jailbreak/i,
    /override/i,
  ],
};

export interface SecurityResult {
  allowed: boolean;
  reason?: string;
  sanitizedInput?: string;
  riskScore: number;
  flags: string[];
}

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  resetAt: Date;
}

export class SecurityGuardian {
  private supabase;
  private rateLimitCache: Map<string, { count: number; resetAt: number }> = new Map();

  constructor(supabaseUrl: string, supabaseKey: string) {
    this.supabase = createClient(supabaseUrl, supabaseKey);
  }

  validateInput(input: string, userId: string, userTier: string): SecurityResult {
    const flags: string[] = [];
    let riskScore = 0;

    if (input.length > SECURITY_CONFIG.maxInputLength) {
      return {
        allowed: false,
        reason: 'Input exceeds maximum length',
        riskScore: 0.3,
        flags: ['length_exceeded'],
      };
    }

    for (const pattern of SECURITY_CONFIG.blockedPatterns) {
      if (pattern.test(input)) {
        void this.logSecurityEvent(userId, 'blocked_pattern', {
          pattern: pattern.toString(),
          input_preview: input.slice(0, 100),
        });
        return {
          allowed: false,
          reason: 'Invalid input pattern detected',
          riskScore: 0.9,
          flags: ['injection_attempt'],
        };
      }
    }

    for (const pattern of SECURITY_CONFIG.suspiciousPatterns) {
      if (pattern.test(input)) {
        flags.push('suspicious_pattern');
        riskScore += 0.2;
        void this.logSecurityEvent(userId, 'suspicious_pattern', {
          pattern: pattern.toString(),
          input_preview: input.slice(0, 100),
        });
      }
    }

    const sanitized = this.sanitizeInput(input);

    if (this.hasEncodingTricks(input)) {
      flags.push('encoding_tricks');
      riskScore += 0.3;
    }

    return {
      allowed: true,
      sanitizedInput: sanitized,
      riskScore: Math.min(riskScore, 1),
      flags,
    };
  }

  async checkRateLimit(userId: string, userTier: string): Promise<RateLimitResult> {
    const limit =
      SECURITY_CONFIG.rateLimits[userTier as keyof typeof SECURITY_CONFIG.rateLimits] || 20;
    const cacheKey = `${userId}:${userTier}`;
    const now = Date.now();
    const hourMs = 60 * 60 * 1000;

    const cached = this.rateLimitCache.get(cacheKey);
    if (cached && cached.resetAt > now) {
      if (cached.count >= limit) {
        return { allowed: false, remaining: 0, resetAt: new Date(cached.resetAt) };
      }
      cached.count++;
      return { allowed: true, remaining: limit - cached.count, resetAt: new Date(cached.resetAt) };
    }

    this.rateLimitCache.set(cacheKey, { count: 1, resetAt: now + hourMs });
    return { allowed: true, remaining: limit - 1, resetAt: new Date(now + hourMs) };
  }

  validateContext(context: Record<string, unknown>): SecurityResult {
    const flags: string[] = [];
    let riskScore = 0;

    if (Object.keys(context).length > SECURITY_CONFIG.maxContextKeys) {
      return {
        allowed: false,
        reason: 'Too many context keys',
        riskScore: 0.5,
        flags: ['context_overflow'],
      };
    }

    const suspiciousKeys = ['prompt', 'system', 'instructions', 'api_key', 'token', 'password'];
    for (const key of Object.keys(context)) {
      if (suspiciousKeys.some((s) => key.toLowerCase().includes(s))) {
        flags.push('suspicious_key');
        riskScore += 0.3;
      }
    }

    for (const [key, value] of Object.entries(context)) {
      if (typeof value === 'string') {
        const result = this.validateInput(value, 'context', 'system');
        if (!result.allowed) {
          return {
            allowed: false,
            reason: `Invalid value in context.${key}`,
            riskScore: result.riskScore,
            flags: [...flags, ...result.flags],
          };
        }
      }
    }

    return { allowed: true, riskScore, flags };
  }

  buildSafePrompt(template: string, variables: Record<string, string | number>): string {
    let result = template;
    for (const [key, value] of Object.entries(variables)) {
      const placeholder = `{{${key}}}`;
      const safeValue = typeof value === 'number' ? String(value) : this.sanitizeInput(value);
      result = result.replace(new RegExp(placeholder, 'g'), safeValue);
    }
    return result;
  }

  private sanitizeInput(input: string): string {
    return input
      .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '')
      .replace(/\s+/g, ' ')
      .replace(/<[^>]*>/g, '')
      .trim();
  }

  private hasEncodingTricks(input: string): boolean {
    const hasRTL = /[\u200E\u200F\u202A-\u202E\u2066-\u2069]/g.test(input);
    const hasZeroWidth = /[\u200B\u200C\u200D\uFEFF]/g.test(input);
    const hasHomoglyphs = /[а-яА-Я]/g.test(input);
    return hasRTL || hasZeroWidth || hasHomoglyphs;
  }

  private async logSecurityEvent(userId: string, eventType: string, details: Record<string, unknown>) {
    try {
      await this.supabase.from('security_logs').insert({
        user_id_hash: await this.hashUserId(userId),
        event_type: eventType,
        details,
        timestamp: new Date().toISOString(),
        severity: eventType.includes('blocked') ? 'high' : 'medium',
      });
    } catch {
      // Silent fail for logging
    }
  }

  private async hashUserId(userId: string): Promise<string> {
    const encoder = new TextEncoder();
    const data = encoder.encode(userId + (process.env.ANONYMIZATION_SALT || 'salt'));
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('').slice(0, 16);
  }
}

export class APIKeyGuardian {
  static ensureServerSide() {
    if (typeof window !== 'undefined') {
      throw new Error('This operation is not allowed on client side');
    }
  }

  static maskKey(key: string): string {
    if (key.length < 10) return '***';
    return `${key.slice(0, 7)}...${key.slice(-4)}`;
  }
}

let guardianInstance: SecurityGuardian | null = null;

export function getSecurityGuardian(): SecurityGuardian {
  if (!guardianInstance) {
    guardianInstance = new SecurityGuardian(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
    );
  }
  return guardianInstance;
}
