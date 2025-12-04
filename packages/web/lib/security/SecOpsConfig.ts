/**
 * SECURITY OPERATIONS CONFIGURATION
 *
 * DevSecOps-ready security controls aligned with:
 * - SANS CIS Critical Security Controls
 * - NIST Cybersecurity Framework
 * - OWASP Security Guidelines
 *
 * This file defines security policies, not implementations.
 * Actual enforcement happens in middleware, API routes, and CI/CD.
 */

// ============================================
// SECURITY HEADERS (OWASP Recommendations)
// ============================================

export const SECURITY_HEADERS = {
  // Prevent clickjacking
  'X-Frame-Options': 'DENY',

  // Prevent MIME sniffing
  'X-Content-Type-Options': 'nosniff',

  // Enable XSS filtering
  'X-XSS-Protection': '1; mode=block',

  // Referrer policy
  'Referrer-Policy': 'strict-origin-when-cross-origin',

  // Permissions policy (disable unnecessary browser features)
  'Permissions-Policy':
    'accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()',

  // Content Security Policy
  'Content-Security-Policy': [
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://js.stripe.com",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: https: blob:",
    "font-src 'self' data:",
    "connect-src 'self' https://*.supabase.co https://api.stripe.com wss://*.supabase.co",
    "frame-src https://js.stripe.com https://hooks.stripe.com",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
  ].join('; '),

  // HSTS (only in production)
  'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
};

// ============================================
// RATE LIMITING CONFIGURATION
// ============================================

export const RATE_LIMITS = {
  // Authentication endpoints
  auth: {
    login: { requests: 5, window: '15m', blockDuration: '30m' },
    passwordReset: { requests: 3, window: '1h', blockDuration: '1h' },
    mfaVerify: { requests: 5, window: '5m', blockDuration: '15m' },
    apiKeyCreate: { requests: 3, window: '1h', blockDuration: '1h' },
  },

  // API endpoints by tier
  api: {
    explorer: { requests: 100, window: '1h' },
    analyst: { requests: 1000, window: '1h' },
    strategist: { requests: 5000, window: '1h' },
    architect: { requests: 20000, window: '1h' },
  },

  // Special endpoints
  export: {
    pdf: { requests: 10, window: '1h' },
    bulk: { requests: 3, window: '1h' },
  },

  // Admin endpoints
  admin: {
    userModification: { requests: 50, window: '1h' },
    configChange: { requests: 10, window: '1h' },
  },
};

// ============================================
// INPUT VALIDATION RULES
// ============================================

export const VALIDATION_RULES = {
  // User input limits
  maxInputLength: {
    username: 50,
    email: 254, // RFC 5321
    password: 128,
    searchQuery: 500,
    dashboardName: 100,
    description: 2000,
    note: 10000,
    apiKey: 64,
  },

  // Patterns
  patterns: {
    email: /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/,
    username: /^[a-zA-Z0-9_-]{3,50}$/,
    uuid: /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,
    apiKey: /^lf_[a-zA-Z0-9]{32}$/,
  },

  // Forbidden patterns (potential injection)
  forbidden: {
    sqlInjection: /('|--|;|\/\*|\*\/|xp_|exec\s|execute\s|union\s|select\s|insert\s|update\s|delete\s|drop\s|alter\s)/i,
    xss: /(<script|javascript:|on\w+\s*=|<iframe|<object|<embed)/i,
    pathTraversal: /(\.\.|\/etc\/|\/proc\/|c:\\)/i,
    shellInjection: /(\||;|`|\$\(|&&|\|\|)/,
  },
};

// ============================================
// SESSION CONFIGURATION
// ============================================

export const SESSION_CONFIG = {
  // Session lifetime
  maxAge: 7 * 24 * 60 * 60, // 7 days
  renewalThreshold: 24 * 60 * 60, // Renew if < 1 day left

  // Idle timeout
  idleTimeout: 30 * 60, // 30 minutes

  // Concurrent sessions
  maxConcurrentSessions: {
    explorer: 2,
    analyst: 3,
    strategist: 5,
    architect: 10,
  },

  // Session binding
  bindToIP: false, // Too aggressive for mobile users
  bindToUserAgent: true,

  // Cookie settings
  cookie: {
    name: 'lf_session',
    httpOnly: true,
    secure: true, // HTTPS only in production
    sameSite: 'lax' as const,
    domain: '.latticeforge.ai', // Cross-subdomain
  },
};

// ============================================
// ACCESS CONTROL MATRIX
// ============================================

export const ACCESS_CONTROL = {
  // Resource permissions by role
  permissions: {
    explorer: {
      dashboards: { read: true, write: false, delete: false, share: false },
      briefings: { read: true, write: false, delete: false, share: false },
      exports: { read: true, write: false, delete: false, share: false },
      api: { read: true, write: false, delete: false, share: false },
      admin: { read: false, write: false, delete: false, share: false },
    },
    analyst: {
      dashboards: { read: true, write: true, delete: true, share: false },
      briefings: { read: true, write: true, delete: false, share: false },
      exports: { read: true, write: true, delete: false, share: false },
      api: { read: true, write: true, delete: false, share: false },
      admin: { read: false, write: false, delete: false, share: false },
    },
    strategist: {
      dashboards: { read: true, write: true, delete: true, share: true },
      briefings: { read: true, write: true, delete: true, share: true },
      exports: { read: true, write: true, delete: true, share: true },
      api: { read: true, write: true, delete: true, share: false },
      admin: { read: false, write: false, delete: false, share: false },
    },
    architect: {
      dashboards: { read: true, write: true, delete: true, share: true },
      briefings: { read: true, write: true, delete: true, share: true },
      exports: { read: true, write: true, delete: true, share: true },
      api: { read: true, write: true, delete: true, share: true },
      admin: { read: true, write: true, delete: false, share: false },
    },
    admin: {
      dashboards: { read: true, write: true, delete: true, share: true },
      briefings: { read: true, write: true, delete: true, share: true },
      exports: { read: true, write: true, delete: true, share: true },
      api: { read: true, write: true, delete: true, share: true },
      admin: { read: true, write: true, delete: true, share: true },
    },
  },

  // IP allowlist for admin functions (if needed)
  adminIPAllowlist: [] as string[], // Empty = allow all

  // Feature flags by tier
  features: {
    explorer: ['basic_dashboard', 'basic_briefings'],
    analyst: ['custom_dashboards', 'exports', 'alerts', 'api_read'],
    strategist: ['team_sharing', 'webhooks', 'advanced_analytics', 'api_write'],
    architect: ['custom_integrations', 'white_label', 'priority_support', 'unlimited'],
  },
};

// ============================================
// DATA CLASSIFICATION
// ============================================

export const DATA_CLASSIFICATION = {
  // Data sensitivity levels
  levels: {
    public: {
      description: 'Publicly available data',
      encryption: 'at_rest',
      retention: '1y',
      logging: 'standard',
    },
    internal: {
      description: 'Internal business data',
      encryption: 'at_rest_and_transit',
      retention: '3y',
      logging: 'detailed',
    },
    confidential: {
      description: 'Sensitive business data',
      encryption: 'at_rest_and_transit',
      retention: '7y',
      logging: 'full',
      accessLog: true,
    },
    restricted: {
      description: 'Highly sensitive data',
      encryption: 'at_rest_and_transit',
      retention: '10y',
      logging: 'full',
      accessLog: true,
      approvalRequired: true,
    },
  },

  // Data type classifications
  types: {
    user_email: 'confidential',
    user_name: 'internal',
    user_password_hash: 'restricted',
    payment_info: 'restricted', // Note: Stripe handles actual card data
    api_keys: 'restricted',
    session_tokens: 'restricted',
    audit_logs: 'confidential',
    intel_briefings: 'confidential',
    user_dashboards: 'internal',
    system_logs: 'internal',
    public_data: 'public',
  },
};

// ============================================
// INCIDENT RESPONSE
// ============================================

export const INCIDENT_RESPONSE = {
  // Severity levels
  severityLevels: {
    critical: {
      description: 'Active breach, data exfiltration, system compromise',
      responseTime: '15m',
      escalation: ['security_team', 'cto', 'legal'],
      actions: ['isolate', 'preserve_evidence', 'notify_affected'],
    },
    high: {
      description: 'Attempted breach, vulnerability exploited',
      responseTime: '1h',
      escalation: ['security_team', 'engineering_lead'],
      actions: ['investigate', 'patch', 'monitor'],
    },
    medium: {
      description: 'Suspicious activity, policy violation',
      responseTime: '4h',
      escalation: ['security_team'],
      actions: ['investigate', 'document'],
    },
    low: {
      description: 'Minor security event, informational',
      responseTime: '24h',
      escalation: [],
      actions: ['document', 'review'],
    },
  },

  // Automatic triggers
  triggers: {
    bruteForceThreshold: 10, // Failed logins before alert
    dataExportThreshold: 50, // Exports per hour before alert
    apiErrorThreshold: 100, // API errors per hour before alert
    suspiciousIPThreshold: 5, // Blocked requests before alert
  },

  // Contact information (redacted in source)
  contacts: {
    securityTeam: 'security@latticeforge.ai',
    legal: 'legal@latticeforge.ai',
    dataProtection: 'dpo@latticeforge.ai',
  },
};

// ============================================
// SECURE DEVELOPMENT LIFECYCLE
// ============================================

export const SDL_REQUIREMENTS = {
  // Pre-commit checks (via Husky)
  preCommit: [
    'lint',
    'type-check',
    'secrets-scan', // gitleaks or similar
    'dependency-check',
  ],

  // CI/CD security checks
  cicd: [
    'sast', // Semgrep
    'dependency-audit', // npm audit
    'container-scan', // if using Docker
    'license-check',
    'test-coverage',
  ],

  // Required for merge to main
  mergeRequirements: [
    'code-review',
    'security-review', // for sensitive changes
    'test-pass',
    'sast-pass',
    'no-high-vulnerabilities',
  ],

  // Deployment checks
  deployment: [
    'environment-parity',
    'rollback-ready',
    'monitoring-enabled',
    'alerts-configured',
  ],
};

// ============================================
// MONITORING & ALERTING
// ============================================

export const MONITORING_CONFIG = {
  // Metrics to track
  metrics: [
    'request_latency_p50',
    'request_latency_p99',
    'error_rate',
    'auth_failure_rate',
    'api_usage_by_tier',
    'database_connections',
    'cache_hit_rate',
    'external_api_latency',
  ],

  // Alert thresholds
  alerts: {
    errorRateThreshold: 0.05, // 5%
    latencyP99Threshold: 2000, // 2s
    authFailureRateThreshold: 0.1, // 10%
    databaseConnectionThreshold: 80, // 80% of pool
    cacheHitRateThreshold: 0.7, // 70% minimum
  },

  // Log retention
  logRetention: {
    application: '30d',
    security: '1y',
    audit: '7y',
    access: '90d',
  },
};

// ============================================
// EXPORT HELPERS
// ============================================

/**
 * Get security headers for Next.js config
 */
export function getSecurityHeaders(): { key: string; value: string }[] {
  return Object.entries(SECURITY_HEADERS).map(([key, value]) => ({
    key,
    value,
  }));
}

/**
 * Check if action is allowed for role
 */
export function isAllowed(
  role: keyof typeof ACCESS_CONTROL.permissions,
  resource: string,
  action: 'read' | 'write' | 'delete' | 'share'
): boolean {
  const permissions = ACCESS_CONTROL.permissions[role];
  if (!permissions) return false;

  const resourcePerms = permissions[resource as keyof typeof permissions];
  if (!resourcePerms) return false;

  return resourcePerms[action] ?? false;
}

/**
 * Check if input passes validation
 */
export function validateInput(
  input: string,
  type: keyof typeof VALIDATION_RULES.maxInputLength
): { valid: boolean; error?: string } {
  // Check length
  const maxLength = VALIDATION_RULES.maxInputLength[type];
  if (input.length > maxLength) {
    return { valid: false, error: `Input exceeds maximum length of ${maxLength}` };
  }

  // Check forbidden patterns
  for (const [name, pattern] of Object.entries(VALIDATION_RULES.forbidden)) {
    if (pattern.test(input)) {
      return { valid: false, error: `Input contains forbidden pattern: ${name}` };
    }
  }

  return { valid: true };
}
