/**
 * LATTICE SECURITY LAYER
 *
 * CRITICAL: Users must NEVER access our LLM API directly.
 * All interactions go through our reasoning layer.
 *
 * This module handles:
 * - Input sanitization (prevent prompt injection)
 * - Rate limiting per user/tier
 * - API key protection
 * - Audit logging
 * - Anomaly detection
 */

import { createClient } from '@supabase/supabase-js';

// Security configuration
const SECURITY_CONFIG = {
  // Rate limits by tier (requests per hour)
  rateLimits: {
    consumer: 20,
    pro: 100,
    enterprise: 500,
  },

  // Max input lengths
  maxInputLength: 1000,
  maxContextKeys: 50,

  // Blocked patterns (prompt injection attempts)
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
    /anthropic/i, // Block references to our provider
    /claude/i,
    /api[_\s]?key/i,
    /bearer\s+token/i,
    /authorization/i,
  ],

  // Suspicious patterns (log but don't block)
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

/**
 * Security Guardian - protects the system
 */
export class SecurityGuardian {
  private supabase;
  private rateLimitCache: Map<string, { count: number; resetAt: number }> = new Map();

  constructor(supabaseUrl: string, supabaseKey: string) {
    this.supabase = createClient(supabaseUrl, supabaseKey);
  }

  /**
   * Validate and sanitize user input
   */
  validateInput(input: string, userId: string, userTier: string): SecurityResult {
    const flags: string[] = [];
    let riskScore = 0;

    // 1. Check length
    if (input.length > SECURITY_CONFIG.maxInputLength) {
      return {
        allowed: false,
        reason: 'Input exceeds maximum length',
        riskScore: 0.3,
        flags: ['length_exceeded'],
      };
    }

    // 2. Check for blocked patterns (prompt injection)
    for (const pattern of SECURITY_CONFIG.blockedPatterns) {
      if (pattern.test(input)) {
        // Log the attempt
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

    // 3. Check for suspicious patterns (log but allow)
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

    // 4. Sanitize the input
    const sanitized = this.sanitizeInput(input);

    // 5. Check for encoding tricks
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

  /**
   * Check rate limits
   */
  async checkRateLimit(userId: string, userTier: string): Promise<RateLimitResult> {
    const limit =
      SECURITY_CONFIG.rateLimits[userTier as keyof typeof SECURITY_CONFIG.rateLimits] || 20;
    const cacheKey = `${userId}:${userTier}`;
    const now = Date.now();
    const hourMs = 60 * 60 * 1000;

    // Check cache first
    const cached = this.rateLimitCache.get(cacheKey);
    if (cached && cached.resetAt > now) {
      if (cached.count >= limit) {
        return {
          allowed: false,
          remaining: 0,
          resetAt: new Date(cached.resetAt),
        };
      }
      cached.count++;
      return {
        allowed: true,
        remaining: limit - cached.count,
        resetAt: new Date(cached.resetAt),
      };
    }

    // Reset or create new entry
    this.rateLimitCache.set(cacheKey, {
      count: 1,
      resetAt: now + hourMs,
    });

    return {
      allowed: true,
      remaining: limit - 1,
      resetAt: new Date(now + hourMs),
    };
  }

  /**
   * Validate context object
   */
  validateContext(context: Record<string, unknown>): SecurityResult {
    const flags: string[] = [];
    let riskScore = 0;

    // Check number of keys
    if (Object.keys(context).length > SECURITY_CONFIG.maxContextKeys) {
      return {
        allowed: false,
        reason: 'Too many context keys',
        riskScore: 0.5,
        flags: ['context_overflow'],
      };
    }

    // Check for suspicious keys
    const suspiciousKeys = ['prompt', 'system', 'instructions', 'api_key', 'token', 'password'];
    for (const key of Object.keys(context)) {
      if (suspiciousKeys.some((s) => key.toLowerCase().includes(s))) {
        flags.push('suspicious_key');
        riskScore += 0.3;
      }
    }

    // Check for string values that might contain injections
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

    return {
      allowed: true,
      riskScore,
      flags,
    };
  }

  /**
   * Build safe prompt for LLM
   * This ensures user input NEVER becomes part of the system prompt
   */
  buildSafePrompt(template: string, variables: Record<string, string | number>): string {
    let result = template;

    // Only allow whitelisted variable substitution
    for (const [key, value] of Object.entries(variables)) {
      const placeholder = `{{${key}}}`;

      // Sanitize the value
      let safeValue: string;
      if (typeof value === 'number') {
        safeValue = String(value);
      } else {
        // Remove any potential prompt injection from string values
        safeValue = this.sanitizeInput(value);
      }

      result = result.replace(new RegExp(placeholder, 'g'), safeValue);
    }

    return result;
  }

  /**
   * Sanitize input string
   */
  private sanitizeInput(input: string): string {
    return (
      input
        // Remove control characters
        .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '')
        // Normalize whitespace
        .replace(/\s+/g, ' ')
        // Remove potential XML/HTML
        .replace(/<[^>]*>/g, '')
        // Escape special characters
        .trim()
    );
  }

  /**
   * Check for encoding tricks
   */
  private hasEncodingTricks(input: string): boolean {
    // Check for unusual Unicode
    const hasRTL = /[\u200E\u200F\u202A-\u202E\u2066-\u2069]/g.test(input);
    const hasZeroWidth = /[\u200B\u200C\u200D\uFEFF]/g.test(input);
    const hasHomoglyphs = /[а-яА-Я]/g.test(input); // Cyrillic lookalikes

    return hasRTL || hasZeroWidth || hasHomoglyphs;
  }

  /**
   * Log security event
   */
  private async logSecurityEvent(
    userId: string,
    eventType: string,
    details: Record<string, unknown>
  ) {
    try {
      await this.supabase.from('security_logs').insert({
        user_id_hash: await this.hashUserId(userId),
        event_type: eventType,
        details,
        timestamp: new Date().toISOString(),
        severity: eventType.includes('blocked') ? 'high' : 'medium',
      });
    } catch (error) {
      console.error('Failed to log security event:', error);
    }
  }

  /**
   * Hash user ID for logging
   */
  private async hashUserId(userId: string): Promise<string> {
    const encoder = new TextEncoder();
    const data = encoder.encode(userId + (process.env.ANONYMIZATION_SALT || 'default-salt'));
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('')
      .slice(0, 16);
  }
}

/**
 * API Key Guardian - NEVER expose keys
 */
export class APIKeyGuardian {
  /**
   * Get LFBM API key (RunPod) safely
   */
  static getLFBMKey(): string {
    const key = process.env.LFBM_API_KEY;
    if (!key) {
      throw new Error('LFBM_API_KEY not configured');
    }
    return key;
  }

  /**
   * Check if request is from server-side only
   */
  static ensureServerSide() {
    if (typeof window !== 'undefined') {
      throw new Error('This operation is not allowed on client side');
    }
  }

  /**
   * Mask key for logging (never log full key)
   */
  static maskKey(key: string): string {
    if (key.length < 10) return '***';
    return `${key.slice(0, 7)}...${key.slice(-4)}`;
  }
}

/**
 * Singleton instance
 */
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

// =============================================================================
// OUTPUT GUARDIAN - Validates Elle's (LFBM) output before caching
// =============================================================================
// The stock Qwen model sometimes returns malformed JSON. Guardian validates
// and attempts to fix common issues before the data gets cached.

/**
 * Required keys for a valid briefing response
 */
const REQUIRED_BRIEFING_KEYS = ['political', 'economic', 'security', 'summary'];
const OPTIONAL_BRIEFING_KEYS = [
  'military', 'cyber', 'financial', 'nsm', 'health', 'scitech', 'resources',
  'crime', 'terrorism', 'domestic', 'borders', 'infoops', 'space', 'industry',
  'logistics', 'minerals', 'energy', 'markets', 'religious', 'education',
  'employment', 'housing', 'crypto', 'emerging'
];

export interface BriefingValidationResult {
  valid: boolean;
  briefings: Record<string, string> | null;
  errors: string[];
  warnings: string[];
  fixed: boolean; // True if we had to fix/repair the JSON
}

/**
 * Output Guardian - validates and repairs Elle's JSON output
 */
export class OutputGuardian {
  /**
   * Validate briefing output from Elle (LFBM)
   * Returns validated briefings or null if unfixable
   */
  validateBriefings(raw: unknown): BriefingValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];
    let fixed = false;

    // Handle null/undefined
    if (raw === null || raw === undefined) {
      return {
        valid: false,
        briefings: null,
        errors: ['Output is null or undefined'],
        warnings: [],
        fixed: false,
      };
    }

    // If it's a string, try to parse as JSON
    let parsed: unknown = raw;
    if (typeof raw === 'string') {
      const extracted = this.extractJSON(raw);
      if (extracted) {
        parsed = extracted;
        fixed = true;
      } else {
        return {
          valid: false,
          briefings: null,
          errors: ['Could not extract valid JSON from string output'],
          warnings: [],
          fixed: false,
        };
      }
    }

    // Must be an object at this point
    if (typeof parsed !== 'object' || Array.isArray(parsed)) {
      return {
        valid: false,
        briefings: null,
        errors: ['Output is not an object'],
        warnings: [],
        fixed: false,
      };
    }

    const obj = parsed as Record<string, unknown>;

    // Check for the {raw: ...} fallback pattern - this means parsing already failed
    if ('raw' in obj && Object.keys(obj).length === 1) {
      // Try to extract from the raw content
      const rawContent = obj.raw;
      if (typeof rawContent === 'string') {
        const extracted = this.extractJSON(rawContent);
        if (extracted) {
          // Recursively validate the extracted JSON
          return this.validateBriefings(extracted);
        }
      }
      return {
        valid: false,
        briefings: null,
        errors: ['Output contains only raw content that could not be parsed'],
        warnings: [],
        fixed: false,
      };
    }

    // Check for required keys
    const missingRequired: string[] = [];
    for (const key of REQUIRED_BRIEFING_KEYS) {
      if (!(key in obj)) {
        missingRequired.push(key);
      }
    }

    if (missingRequired.length > 0) {
      errors.push(`Missing required keys: ${missingRequired.join(', ')}`);
    }

    // Validate each value is a string (or can be converted to one)
    const briefings: Record<string, string> = {};
    for (const [key, value] of Object.entries(obj)) {
      if (typeof value === 'string') {
        // Check for nested JSON in string values (common Qwen bug)
        if (value.startsWith('{') || value.startsWith('[')) {
          try {
            const nested = JSON.parse(value);
            // If it parsed, it's nested JSON - flatten it
            briefings[key] = this.flattenNestedJSON(nested);
            warnings.push(`Key '${key}' contained nested JSON - flattened`);
            fixed = true;
          } catch {
            // Not valid JSON, use as-is
            briefings[key] = value;
          }
        } else {
          briefings[key] = value;
        }
      } else if (typeof value === 'object' && value !== null) {
        // Object value - try to stringify meaningfully
        briefings[key] = this.flattenNestedJSON(value);
        warnings.push(`Key '${key}' was an object - converted to string`);
        fixed = true;
      } else if (value !== null && value !== undefined) {
        briefings[key] = String(value);
        warnings.push(`Key '${key}' was not a string - converted`);
        fixed = true;
      }
    }

    // If we have at least some required keys, consider it partially valid
    const hasEnoughKeys = REQUIRED_BRIEFING_KEYS.filter(k => k in briefings).length >= 2;

    return {
      valid: missingRequired.length === 0 || hasEnoughKeys,
      briefings: Object.keys(briefings).length > 0 ? briefings : null,
      errors,
      warnings,
      fixed,
    };
  }

  /**
   * Extract JSON from a string that may contain markdown code blocks or extra text
   */
  private extractJSON(content: string): Record<string, unknown> | null {
    // Try direct parse first
    try {
      return JSON.parse(content);
    } catch {
      // Continue with extraction attempts
    }

    // Remove markdown code blocks
    let cleaned = content
      .replace(/```json\s*/gi, '')
      .replace(/```\s*/g, '')
      .trim();

    // Try parsing cleaned content
    try {
      return JSON.parse(cleaned);
    } catch {
      // Continue with regex extraction
    }

    // Try to find JSON object with regex (greedy match)
    const jsonMatch = cleaned.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      try {
        return JSON.parse(jsonMatch[0]);
      } catch {
        // JSON is malformed
      }
    }

    // Try to find the outermost balanced braces
    const balanced = this.extractBalancedBraces(cleaned);
    if (balanced) {
      try {
        return JSON.parse(balanced);
      } catch {
        // Still malformed
      }
    }

    return null;
  }

  /**
   * Extract balanced braces from a string
   */
  private extractBalancedBraces(content: string): string | null {
    const start = content.indexOf('{');
    if (start === -1) return null;

    let depth = 0;
    let inString = false;
    let escaped = false;

    for (let i = start; i < content.length; i++) {
      const char = content[i];

      if (escaped) {
        escaped = false;
        continue;
      }

      if (char === '\\') {
        escaped = true;
        continue;
      }

      if (char === '"') {
        inString = !inString;
        continue;
      }

      if (inString) continue;

      if (char === '{') depth++;
      if (char === '}') {
        depth--;
        if (depth === 0) {
          return content.slice(start, i + 1);
        }
      }
    }

    return null;
  }

  /**
   * Flatten nested JSON into a readable string
   */
  private flattenNestedJSON(obj: unknown): string {
    if (typeof obj === 'string') return obj;
    if (typeof obj !== 'object' || obj === null) return String(obj);

    // For arrays, join with newlines
    if (Array.isArray(obj)) {
      return obj.map(item => this.flattenNestedJSON(item)).join('\n');
    }

    // For objects, extract meaningful content
    const record = obj as Record<string, unknown>;

    // Common patterns in briefings
    if ('text' in record) return this.flattenNestedJSON(record.text);
    if ('content' in record) return this.flattenNestedJSON(record.content);
    if ('value' in record) return this.flattenNestedJSON(record.value);
    if ('description' in record) return this.flattenNestedJSON(record.description);

    // Fallback: join all string values
    const parts: string[] = [];
    for (const [key, value] of Object.entries(record)) {
      if (typeof value === 'string') {
        parts.push(value);
      } else if (typeof value === 'object') {
        parts.push(this.flattenNestedJSON(value));
      }
    }

    return parts.join(' ').trim() || JSON.stringify(obj);
  }
}

// Singleton
let outputGuardianInstance: OutputGuardian | null = null;

export function getOutputGuardian(): OutputGuardian {
  if (!outputGuardianInstance) {
    outputGuardianInstance = new OutputGuardian();
  }
  return outputGuardianInstance;
}

// =============================================================================
// ELLE GUARDIAN - Master filter for ALL Elle interactions
// =============================================================================
// Guardian wraps every interaction Elle has:
// - User → Elle (input validation)
// - Elle → User (output filtering)
// - Elle → Web (research request filtering)
// - Elle → Database (query sanitization)
// - Elle → Self (chain-of-thought filtering)
// - Elle → Website (action validation)

export type ElleInteractionType =
  | 'user_to_elle'      // User message to Elle
  | 'elle_to_user'      // Elle response to user
  | 'elle_to_web'       // Elle initiating web research
  | 'elle_to_db'        // Elle querying database
  | 'elle_to_self'      // Elle chain-of-thought/reflection
  | 'elle_to_website';  // Elle taking action on website

export interface ElleInteraction {
  type: ElleInteractionType;
  content: string;
  metadata?: {
    userId?: string;
    conversationId?: string;
    mode?: string;
    tool?: string;
    query?: string;
    riskScore?: number;
    [key: string]: string | number | boolean | undefined;
  };
}

export interface ElleGuardianResult {
  allowed: boolean;
  filtered: string;
  blocked: boolean;
  reason?: string;
  riskScore: number;
  redacted: string[];  // List of items redacted from content
  logged: boolean;
}

/**
 * Elle Guardian - Master filter for ALL Elle interactions
 */
export class ElleGuardian {
  private securityGuardian: SecurityGuardian;
  private outputGuardian: OutputGuardian;
  private supabase;

  // Patterns that should NEVER appear in Elle's output to users
  private readonly outputBlockedPatterns = [
    // API keys / secrets
    /sk-[a-zA-Z0-9]{20,}/g,           // Anthropic/OpenAI keys
    /ghp_[a-zA-Z0-9]{36}/g,           // GitHub tokens
    /xoxb-[a-zA-Z0-9-]+/g,            // Slack tokens
    /AKIA[A-Z0-9]{16}/g,              // AWS access keys
    /eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+/g, // JWTs

    // Internal system info
    /supabase[_-]?service[_-]?role/gi,
    /NEXT_PUBLIC_/g,
    /process\.env\./g,
    /Bearer\s+[a-zA-Z0-9._-]+/g,

    // Database internals
    /postgresql:\/\/[^\s]+/gi,
    /postgres:\/\/[^\s]+/gi,
    /connection\s*string/gi,

    // File paths that could expose server structure
    /\/home\/[a-zA-Z0-9_-]+\//g,
    /\/var\/www\//g,
    /\/etc\//g,
    /C:\\Users\\[^\\]+\\/g,
  ];

  // Patterns to redact (replace with [REDACTED])
  private readonly redactPatterns = [
    // Email addresses (except in certain contexts)
    /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g,
    // Phone numbers
    /\+?[1-9]\d{1,14}/g,
    // Credit card numbers
    /\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b/g,
    // SSN patterns
    /\b\d{3}-\d{2}-\d{4}\b/g,
  ];

  // Research/web blocked patterns
  private readonly webBlockedPatterns = [
    /localhost/i,
    /127\.0\.0\.1/,
    /192\.168\./,
    /10\.\d+\.\d+\.\d+/,
    /172\.(1[6-9]|2[0-9]|3[0-1])\./,
    /\.internal\./i,
    /\.local$/i,
    /file:\/\//,
  ];

  constructor(supabaseUrl?: string, supabaseKey?: string) {
    this.securityGuardian = getSecurityGuardian();
    this.outputGuardian = getOutputGuardian();
    this.supabase = createClient(
      supabaseUrl || process.env.NEXT_PUBLIC_SUPABASE_URL!,
      supabaseKey || process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
    );
  }

  /**
   * Filter any interaction to/from Elle
   */
  async filter(interaction: ElleInteraction): Promise<ElleGuardianResult> {
    const startTime = Date.now();

    switch (interaction.type) {
      case 'user_to_elle':
        return this.filterUserInput(interaction);

      case 'elle_to_user':
        return this.filterElleOutput(interaction);

      case 'elle_to_web':
        return this.filterWebRequest(interaction);

      case 'elle_to_db':
        return this.filterDbQuery(interaction);

      case 'elle_to_self':
        return this.filterSelfReflection(interaction);

      case 'elle_to_website':
        return this.filterWebsiteAction(interaction);

      default:
        // Unknown interaction type - block by default
        return {
          allowed: false,
          filtered: '',
          blocked: true,
          reason: 'Unknown interaction type',
          riskScore: 1.0,
          redacted: [],
          logged: await this.logInteraction(interaction, 'blocked_unknown', 1.0, Date.now() - startTime),
        };
    }
  }

  /**
   * Filter user input to Elle
   */
  private async filterUserInput(interaction: ElleInteraction): Promise<ElleGuardianResult> {
    const startTime = Date.now();
    const redacted: string[] = [];

    // Use SecurityGuardian for input validation
    const securityResult = this.securityGuardian.validateInput(
      interaction.content,
      interaction.metadata?.userId || 'unknown',
      'admin' // Study Book is admin-only
    );

    if (!securityResult.allowed) {
      return {
        allowed: false,
        filtered: '',
        blocked: true,
        reason: securityResult.reason,
        riskScore: securityResult.riskScore,
        redacted: [],
        logged: await this.logInteraction(interaction, 'blocked_input', securityResult.riskScore, Date.now() - startTime),
      };
    }

    // Additional filtering for sensitive data in user input
    let filtered = securityResult.sanitizedInput || interaction.content;

    // Redact any sensitive patterns the user might accidentally include
    for (const pattern of this.redactPatterns) {
      const matches = filtered.match(pattern);
      if (matches) {
        redacted.push(...matches);
        filtered = filtered.replace(pattern, '[REDACTED]');
      }
    }

    return {
      allowed: true,
      filtered,
      blocked: false,
      riskScore: securityResult.riskScore,
      redacted,
      logged: await this.logInteraction(interaction, 'allowed', securityResult.riskScore, Date.now() - startTime),
    };
  }

  /**
   * Filter Elle's output to user
   */
  private async filterElleOutput(interaction: ElleInteraction): Promise<ElleGuardianResult> {
    const startTime = Date.now();
    const redacted: string[] = [];
    let filtered = interaction.content;
    let riskScore = 0;

    // Block any output containing sensitive patterns
    for (const pattern of this.outputBlockedPatterns) {
      if (pattern.test(filtered)) {
        // This is a serious issue - Elle is leaking secrets
        return {
          allowed: false,
          filtered: '[Response blocked - contained sensitive information]',
          blocked: true,
          reason: 'Output contained sensitive patterns',
          riskScore: 1.0,
          redacted: [],
          logged: await this.logInteraction(interaction, 'blocked_sensitive_output', 1.0, Date.now() - startTime),
        };
      }
    }

    // Redact PII from output
    for (const pattern of this.redactPatterns) {
      const matches = filtered.match(pattern);
      if (matches) {
        redacted.push(...matches);
        filtered = filtered.replace(pattern, '[REDACTED]');
        riskScore += 0.1;
      }
    }

    // Check for injection attempts in Elle's output (could indicate compromise)
    const injectionPatterns = [
      /<script/gi,
      /javascript:/gi,
      /on\w+\s*=/gi,  // event handlers
      /eval\s*\(/gi,
      /document\./gi,
      /window\./gi,
    ];

    for (const pattern of injectionPatterns) {
      if (pattern.test(filtered)) {
        filtered = filtered.replace(pattern, '[SANITIZED]');
        riskScore += 0.3;
        redacted.push('potential_xss');
      }
    }

    return {
      allowed: true,
      filtered,
      blocked: false,
      riskScore: Math.min(riskScore, 1),
      redacted,
      logged: await this.logInteraction(interaction, 'allowed', riskScore, Date.now() - startTime),
    };
  }

  /**
   * Filter Elle's web research requests
   */
  private async filterWebRequest(interaction: ElleInteraction): Promise<ElleGuardianResult> {
    const startTime = Date.now();
    const url = interaction.metadata?.query || interaction.content;

    // Block internal/localhost URLs
    for (const pattern of this.webBlockedPatterns) {
      if (pattern.test(url)) {
        return {
          allowed: false,
          filtered: '',
          blocked: true,
          reason: 'Blocked internal URL access',
          riskScore: 0.9,
          redacted: [],
          logged: await this.logInteraction(interaction, 'blocked_internal_url', 0.9, Date.now() - startTime),
        };
      }
    }

    // Block file:// protocol
    if (url.startsWith('file://')) {
      return {
        allowed: false,
        filtered: '',
        blocked: true,
        reason: 'File protocol not allowed',
        riskScore: 1.0,
        redacted: [],
        logged: await this.logInteraction(interaction, 'blocked_file_protocol', 1.0, Date.now() - startTime),
      };
    }

    return {
      allowed: true,
      filtered: url,
      blocked: false,
      riskScore: 0,
      redacted: [],
      logged: await this.logInteraction(interaction, 'allowed', 0, Date.now() - startTime),
    };
  }

  /**
   * Filter Elle's database queries
   */
  private async filterDbQuery(interaction: ElleInteraction): Promise<ElleGuardianResult> {
    const startTime = Date.now();
    const query = interaction.content;

    // Block dangerous SQL operations
    const dangerousPatterns = [
      /DROP\s+TABLE/gi,
      /DELETE\s+FROM\s+\*/gi,
      /TRUNCATE/gi,
      /ALTER\s+TABLE/gi,
      /GRANT/gi,
      /REVOKE/gi,
      /CREATE\s+USER/gi,
      /pg_dump/gi,
    ];

    for (const pattern of dangerousPatterns) {
      if (pattern.test(query)) {
        return {
          allowed: false,
          filtered: '',
          blocked: true,
          reason: 'Dangerous database operation blocked',
          riskScore: 1.0,
          redacted: [],
          logged: await this.logInteraction(interaction, 'blocked_dangerous_sql', 1.0, Date.now() - startTime),
        };
      }
    }

    return {
      allowed: true,
      filtered: query,
      blocked: false,
      riskScore: 0,
      redacted: [],
      logged: await this.logInteraction(interaction, 'allowed', 0, Date.now() - startTime),
    };
  }

  /**
   * Filter Elle's self-reflection/chain-of-thought
   */
  private async filterSelfReflection(interaction: ElleInteraction): Promise<ElleGuardianResult> {
    const startTime = Date.now();
    let filtered = interaction.content;
    const redacted: string[] = [];

    // Redact any sensitive info that might appear in thinking
    for (const pattern of [...this.outputBlockedPatterns, ...this.redactPatterns]) {
      const matches = filtered.match(pattern);
      if (matches) {
        redacted.push(...matches.map(m => 'thinking_' + typeof m === 'string' ? m.slice(0, 10) : ''));
        filtered = filtered.replace(pattern, '[THINKING_REDACTED]');
      }
    }

    return {
      allowed: true,
      filtered,
      blocked: false,
      riskScore: redacted.length > 0 ? 0.3 : 0,
      redacted,
      logged: await this.logInteraction(interaction, 'allowed', redacted.length > 0 ? 0.3 : 0, Date.now() - startTime),
    };
  }

  /**
   * Filter Elle's website actions
   */
  private async filterWebsiteAction(interaction: ElleInteraction): Promise<ElleGuardianResult> {
    const startTime = Date.now();

    // Actions Elle is allowed to take
    const allowedActions = [
      'navigate',
      'search',
      'view',
      'read',
      'analyze',
    ];

    // Actions that require explicit admin confirmation
    const confirmActions = [
      'create',
      'update',
      'delete',
      'modify',
      'submit',
      'execute',
    ];

    const action = interaction.metadata?.tool?.toLowerCase() || '';

    // Block if action is not in allowed or confirm list
    if (!allowedActions.includes(action) && !confirmActions.includes(action)) {
      return {
        allowed: false,
        filtered: '',
        blocked: true,
        reason: `Action '${action}' not permitted`,
        riskScore: 0.8,
        redacted: [],
        logged: await this.logInteraction(interaction, 'blocked_action', 0.8, Date.now() - startTime),
      };
    }

    // Actions requiring confirmation should be flagged
    if (confirmActions.includes(action)) {
      return {
        allowed: true,
        filtered: interaction.content,
        blocked: false,
        reason: 'Action requires confirmation',
        riskScore: 0.5,
        redacted: [],
        logged: await this.logInteraction(interaction, 'requires_confirmation', 0.5, Date.now() - startTime),
      };
    }

    return {
      allowed: true,
      filtered: interaction.content,
      blocked: false,
      riskScore: 0,
      redacted: [],
      logged: await this.logInteraction(interaction, 'allowed', 0, Date.now() - startTime),
    };
  }

  /**
   * Log interaction for audit trail
   */
  private async logInteraction(
    interaction: ElleInteraction,
    result: string,
    riskScore: number,
    latencyMs: number
  ): Promise<boolean> {
    try {
      await this.supabase.from('elle_guardian_logs').insert({
        interaction_type: interaction.type,
        result,
        risk_score: riskScore,
        latency_ms: latencyMs,
        user_id: interaction.metadata?.userId,
        conversation_id: interaction.metadata?.conversationId,
        mode: interaction.metadata?.mode,
        content_preview: interaction.content.slice(0, 100),
        timestamp: new Date().toISOString(),
      });
      return true;
    } catch (error) {
      console.error('[ElleGuardian] Failed to log interaction:', error);
      return false;
    }
  }
}

// Singleton
let elleGuardianInstance: ElleGuardian | null = null;

export function getElleGuardian(): ElleGuardian {
  if (!elleGuardianInstance) {
    elleGuardianInstance = new ElleGuardian();
  }
  return elleGuardianInstance;
}
