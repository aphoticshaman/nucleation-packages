// LatticeForge Security Module
// Centralized security: auth, feature gating, input sanitization
// NO BYPASS POSSIBLE - all checks happen server-side

import { createClient, SupabaseClient } from 'https://esm.sh/@supabase/supabase-js@2';

// ============================================
// TYPES
// ============================================

export interface AuthContext {
  client_id: string;
  client_tier: 'free' | 'pro' | 'enterprise';
  key_id: string;
  user_id?: string;
  is_valid: boolean;
}

export interface TierLimits {
  max_signal_tokens: number;
  max_fusion_tokens: number;
  max_analysis_tokens: number;
  max_total_tokens: number;
  max_api_keys: number;
}

export type Feature =
  | 'brief.quick'
  | 'brief.us_deep_dive'
  | 'brief.intel'
  | 'data.worldbank'
  | 'data.factbook'
  | 'data.fred'
  | 'data.realtime'
  | 'analysis.fusion'
  | 'analysis.anomaly'
  | 'analysis.predictive';

// ============================================
// FEATURE GATES - HARDCODED, NOT CONFIGURABLE
// ============================================

const FEATURE_ACCESS: Record<string, Feature[]> = {
  free: ['brief.quick', 'data.worldbank', 'data.factbook', 'analysis.fusion', 'analysis.anomaly'],
  pro: [
    'brief.quick',
    'brief.us_deep_dive',
    'data.worldbank',
    'data.factbook',
    'data.fred',
    'analysis.fusion',
    'analysis.anomaly',
  ],
  enterprise: [
    'brief.quick',
    'brief.us_deep_dive',
    'brief.intel',
    'data.worldbank',
    'data.factbook',
    'data.fred',
    'data.realtime',
    'analysis.fusion',
    'analysis.anomaly',
    'analysis.predictive',
  ],
} as const;

// Rate limits per tier (per hour)
const RATE_LIMITS: Record<string, Record<string, number>> = {
  free: {
    'brief.quick': 5,
    'analysis.fusion': 10,
  },
  pro: {
    'brief.quick': 100,
    'brief.us_deep_dive': 4,
    'analysis.fusion': 100,
  },
  enterprise: {
    'brief.quick': 1000,
    'brief.us_deep_dive': 100,
    'brief.intel': 10,
    'analysis.fusion': 1000,
  },
};

// ============================================
// AUTHENTICATION
// ============================================

export async function authenticate(
  request: Request,
  supabase: SupabaseClient
): Promise<AuthContext | null> {
  const authHeader = request.headers.get('Authorization');

  if (!authHeader?.startsWith('Bearer ')) {
    return null;
  }

  const token = authHeader.slice(7); // Remove 'Bearer '

  // Validate API key via database function (server-side only)
  const { data, error } = await supabase.rpc('validate_api_key', { p_key: token });

  if (error || !data?.length) {
    return null;
  }

  const auth = data[0];

  // Verify tier is valid
  if (!['free', 'pro', 'enterprise'].includes(auth.client_tier)) {
    return null;
  }

  return {
    client_id: auth.client_id,
    client_tier: auth.client_tier as AuthContext['client_tier'],
    key_id: auth.key_id,
    is_valid: true,
  };
}

// ============================================
// FEATURE GATING (NOT BYPASSABLE)
// ============================================

export function hasFeatureAccess(tier: string, feature: Feature): boolean {
  const allowed = FEATURE_ACCESS[tier];
  if (!allowed) return false;
  return allowed.includes(feature);
}

export function requireFeature(
  auth: AuthContext,
  feature: Feature
): { allowed: true } | { allowed: false; error: string; status: number } {
  if (!auth.is_valid) {
    return { allowed: false, error: 'Invalid authentication', status: 401 };
  }

  if (!hasFeatureAccess(auth.client_tier, feature)) {
    return {
      allowed: false,
      error: `Feature '${feature}' requires ${getRequiredTier(feature)} tier or higher`,
      status: 403,
    };
  }

  return { allowed: true };
}

function getRequiredTier(feature: Feature): string {
  if (FEATURE_ACCESS.free.includes(feature)) return 'free';
  if (FEATURE_ACCESS.pro.includes(feature)) return 'pro';
  return 'enterprise';
}

// ============================================
// RATE LIMITING
// ============================================

export async function checkRateLimit(
  supabase: SupabaseClient,
  auth: AuthContext,
  feature: Feature
): Promise<{ allowed: boolean; remaining?: number; reset?: Date }> {
  const limit = RATE_LIMITS[auth.client_tier]?.[feature];

  if (!limit) {
    // No rate limit for this feature/tier combination
    return { allowed: true };
  }

  const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();

  // Count recent usage
  const { count } = await supabase
    .from('usage_records')
    .select('*', { count: 'exact', head: true })
    .eq('client_id', auth.client_id)
    .eq('operation', feature)
    .gte('created_at', oneHourAgo);

  const used = count || 0;
  const remaining = limit - used;

  if (remaining <= 0) {
    return {
      allowed: false,
      remaining: 0,
      reset: new Date(Date.now() + 60 * 60 * 1000),
    };
  }

  return { allowed: true, remaining };
}

// ============================================
// INPUT SANITIZATION (ANTI-INJECTION)
// ============================================

// Patterns that indicate prompt injection attempts
const INJECTION_PATTERNS = [
  /ignore\s+(previous|above|all)\s+instructions/i,
  /disregard\s+(previous|above|all)/i,
  /forget\s+(everything|previous|above)/i,
  /you\s+are\s+now\s+a/i,
  /pretend\s+(you|to\s+be)/i,
  /act\s+as\s+(if|a)/i,
  /new\s+instructions?:/i,
  /system\s*prompt/i,
  /\[SYSTEM\]/i,
  /\[INST\]/i,
  /<\|im_start\|>/i,
  /<\|endoftext\|>/i,
  /###\s*(instruction|system)/i,
  /override\s+(previous|system)/i,
  /jailbreak/i,
  /DAN\s+mode/i,
  /bypass\s+(safety|filter|restriction)/i,
];

// Characters that could be used for injection
const DANGEROUS_CHARS = /[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g;

export interface SanitizeResult {
  safe: boolean;
  sanitized: string;
  blocked_reason?: string;
}

export function sanitizeUserInput(input: string, maxLength = 10000): SanitizeResult {
  if (!input || typeof input !== 'string') {
    return { safe: true, sanitized: '' };
  }

  // Remove dangerous control characters
  let sanitized = input.replace(DANGEROUS_CHARS, '');

  // Truncate to max length
  if (sanitized.length > maxLength) {
    sanitized = sanitized.slice(0, maxLength);
  }

  // Check for injection patterns
  for (const pattern of INJECTION_PATTERNS) {
    if (pattern.test(sanitized)) {
      return {
        safe: false,
        sanitized: '',
        blocked_reason: 'Input contains disallowed patterns',
      };
    }
  }

  // Escape any remaining special sequences
  sanitized = sanitized
    .replace(/```/g, '`​`​`') // Zero-width space in code blocks
    .replace(/###/g, '#​#​#')
    .replace(/<\|/g, '< |')
    .replace(/\|>/g, '| >');

  return { safe: true, sanitized };
}

export function sanitizeObject(
  obj: Record<string, unknown>,
  maxDepth = 3
): Record<string, unknown> {
  if (maxDepth <= 0) return {};

  const result: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(obj)) {
    // Sanitize key
    const sanitizedKey = key.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 100);

    if (typeof value === 'string') {
      const { safe, sanitized } = sanitizeUserInput(value);
      if (safe) {
        result[sanitizedKey] = sanitized;
      }
    } else if (typeof value === 'number' || typeof value === 'boolean') {
      result[sanitizedKey] = value;
    } else if (Array.isArray(value)) {
      result[sanitizedKey] = value
        .slice(0, 100)
        .map((v) => (typeof v === 'string' ? sanitizeUserInput(v).sanitized : v));
    } else if (value && typeof value === 'object') {
      result[sanitizedKey] = sanitizeObject(value as Record<string, unknown>, maxDepth - 1);
    }
  }

  return result;
}

// ============================================
// USAGE RECORDING
// ============================================

export async function recordUsage(
  supabase: SupabaseClient,
  auth: AuthContext,
  operation: string,
  tokens: { signal?: number; fusion?: number; analysis?: number } = {}
): Promise<void> {
  await supabase.rpc('record_usage', {
    p_client_id: auth.client_id,
    p_api_key_id: auth.key_id,
    p_operation: operation,
    p_signal_tokens: tokens.signal || 0,
    p_fusion_tokens: tokens.fusion || 0,
    p_analysis_tokens: tokens.analysis || 0,
  });
}

// ============================================
// HELPER: CREATE SECURE SUPABASE CLIENT
// ============================================

export function createSecureClient(): SupabaseClient {
  const url = Deno.env.get('SUPABASE_URL');
  const key = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');

  if (!url || !key) {
    throw new Error('Missing Supabase configuration');
  }

  return createClient(url, key);
}

// ============================================
// HELPER: CORS HEADERS
// ============================================

export const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
};

export function jsonResponse(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  });
}
