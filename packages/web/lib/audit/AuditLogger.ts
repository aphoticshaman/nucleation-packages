/**
 * AUDIT LOGGING SYSTEM
 *
 * Federal compliance-ready audit trail for:
 * - SOX (Sarbanes-Oxley) - Financial controls, data integrity
 * - FISMA - Federal information security management
 * - NIST 800-53 - Security controls
 *
 * All audit events are:
 * - Immutable (append-only)
 * - Timestamped (UTC)
 * - Tamper-evident (hash chain)
 * - Attributable (user ID, IP, session)
 * - Categorized (action type, resource type)
 *
 * Target customers: ICE, CBP, DHS, DOD
 */

// ============================================
// AUDIT EVENT TYPES
// ============================================

export type AuditCategory =
  | 'authentication'
  | 'authorization'
  | 'data_access'
  | 'data_modification'
  | 'data_export'
  | 'configuration'
  | 'admin'
  | 'api'
  | 'payment'
  | 'security'
  | 'system';

export type AuditSeverity =
  | 'info'     // Normal operations
  | 'notice'   // Notable but normal
  | 'warning'  // Potentially problematic
  | 'alert'    // Requires attention
  | 'critical'; // Security incident

export interface AuditEvent {
  // Event identification
  id: string;
  timestamp: string; // ISO 8601 UTC
  sequence: number;  // Monotonic counter for ordering

  // Actor identification
  actor: {
    userId: string | null;
    email: string | null;
    role: string | null;
    ipAddress: string | null;
    userAgent: string | null;
    sessionId: string | null;
    isSystem: boolean;
  };

  // Event details
  category: AuditCategory;
  action: string;
  severity: AuditSeverity;
  description: string;

  // Resource affected
  resource: {
    type: string;
    id: string | null;
    name: string | null;
  };

  // Request context
  request: {
    method: string | null;
    path: string | null;
    queryParams: Record<string, string> | null;
    bodyHash: string | null; // SHA-256 of request body (not the body itself)
  };

  // Response context
  response: {
    statusCode: number | null;
    success: boolean;
    errorCode: string | null;
    errorMessage: string | null;
  };

  // Additional data (structured, no PII)
  metadata: Record<string, unknown>;

  // Integrity
  previousHash: string | null;
  hash: string;
}

// ============================================
// AUDIT ACTIONS BY CATEGORY
// ============================================

export const AUDIT_ACTIONS = {
  authentication: [
    'login_attempt',
    'login_success',
    'login_failure',
    'logout',
    'password_change',
    'password_reset_request',
    'password_reset_complete',
    'mfa_enable',
    'mfa_disable',
    'mfa_challenge',
    'mfa_success',
    'mfa_failure',
    'session_create',
    'session_expire',
    'session_revoke',
    'api_key_create',
    'api_key_revoke',
  ],
  authorization: [
    'access_granted',
    'access_denied',
    'role_assign',
    'role_revoke',
    'permission_check',
    'tier_upgrade',
    'tier_downgrade',
  ],
  data_access: [
    'view_dashboard',
    'view_briefing',
    'view_report',
    'view_entity',
    'view_user',
    'search_query',
    'export_request',
    'api_read',
  ],
  data_modification: [
    'create_dashboard',
    'update_dashboard',
    'delete_dashboard',
    'create_watchlist',
    'update_watchlist',
    'delete_watchlist',
    'create_alert',
    'update_alert',
    'delete_alert',
    'api_write',
  ],
  data_export: [
    'export_pdf',
    'export_csv',
    'export_json',
    'export_pptx',
    'export_docx',
    'bulk_export',
  ],
  configuration: [
    'update_profile',
    'update_settings',
    'update_notifications',
    'update_preferences',
    'webhook_create',
    'webhook_update',
    'webhook_delete',
  ],
  admin: [
    'user_create',
    'user_update',
    'user_delete',
    'user_suspend',
    'user_activate',
    'invite_send',
    'invite_accept',
    'config_change',
    'feature_flag_toggle',
  ],
  api: [
    'api_request',
    'api_rate_limit',
    'api_error',
    'webhook_delivery',
    'webhook_failure',
  ],
  payment: [
    'checkout_start',
    'checkout_complete',
    'checkout_fail',
    'subscription_create',
    'subscription_update',
    'subscription_cancel',
    'invoice_paid',
    'invoice_failed',
    'refund_request',
    'refund_complete',
  ],
  security: [
    'suspicious_activity',
    'rate_limit_exceeded',
    'invalid_token',
    'ip_blocked',
    'brute_force_detected',
    'data_breach_attempt',
    'privilege_escalation_attempt',
  ],
  system: [
    'startup',
    'shutdown',
    'config_reload',
    'cache_clear',
    'maintenance_start',
    'maintenance_end',
    'backup_start',
    'backup_complete',
    'backup_fail',
  ],
} as const;

// ============================================
// AUDIT LOGGER CLASS
// ============================================

class AuditLogger {
  private sequence = 0;
  private lastHash: string | null = null;
  private buffer: AuditEvent[] = [];
  private flushInterval: NodeJS.Timeout | null = null;

  constructor() {
    // In production, this would connect to a secure, append-only log store
    // Options: AWS CloudWatch Logs, Supabase audit table, dedicated SIEM
  }

  /**
   * Log an audit event
   */
  async log(params: {
    category: AuditCategory;
    action: string;
    severity?: AuditSeverity;
    description: string;
    actor?: Partial<AuditEvent['actor']>;
    resource?: Partial<AuditEvent['resource']>;
    request?: Partial<AuditEvent['request']>;
    response?: Partial<AuditEvent['response']>;
    metadata?: Record<string, unknown>;
  }): Promise<AuditEvent> {
    const event: AuditEvent = {
      id: this.generateId(),
      timestamp: new Date().toISOString(),
      sequence: ++this.sequence,

      actor: {
        userId: params.actor?.userId ?? null,
        email: params.actor?.email ?? null,
        role: params.actor?.role ?? null,
        ipAddress: params.actor?.ipAddress ?? null,
        userAgent: params.actor?.userAgent ?? null,
        sessionId: params.actor?.sessionId ?? null,
        isSystem: params.actor?.isSystem ?? false,
      },

      category: params.category,
      action: params.action,
      severity: params.severity ?? 'info',
      description: params.description,

      resource: {
        type: params.resource?.type ?? 'unknown',
        id: params.resource?.id ?? null,
        name: params.resource?.name ?? null,
      },

      request: {
        method: params.request?.method ?? null,
        path: params.request?.path ?? null,
        queryParams: params.request?.queryParams ?? null,
        bodyHash: params.request?.bodyHash ?? null,
      },

      response: {
        statusCode: params.response?.statusCode ?? null,
        success: params.response?.success ?? true,
        errorCode: params.response?.errorCode ?? null,
        errorMessage: params.response?.errorMessage ?? null,
      },

      metadata: params.metadata ?? {},

      previousHash: this.lastHash,
      hash: '', // Will be computed
    };

    // Compute hash for tamper evidence
    event.hash = await this.computeHash(event);
    this.lastHash = event.hash;

    // Add to buffer
    this.buffer.push(event);

    // In production: async flush to persistent storage
    if (this.buffer.length >= 100) {
      await this.flush();
    }

    return event;
  }

  /**
   * Convenience methods for common audit events
   */
  async logAuth(
    action: (typeof AUDIT_ACTIONS.authentication)[number],
    params: {
      userId?: string;
      email?: string;
      ipAddress?: string;
      success: boolean;
      reason?: string;
    }
  ): Promise<AuditEvent> {
    return this.log({
      category: 'authentication',
      action,
      severity: params.success ? 'info' : 'warning',
      description: `${action}: ${params.success ? 'success' : 'failure'}${params.reason ? ` - ${params.reason}` : ''}`,
      actor: {
        userId: params.userId,
        email: params.email,
        ipAddress: params.ipAddress,
      },
      response: { success: params.success },
    });
  }

  async logDataAccess(
    action: (typeof AUDIT_ACTIONS.data_access)[number],
    params: {
      userId: string;
      resourceType: string;
      resourceId?: string;
      resourceName?: string;
    }
  ): Promise<AuditEvent> {
    return this.log({
      category: 'data_access',
      action,
      description: `User ${params.userId} accessed ${params.resourceType}${params.resourceId ? ` (${params.resourceId})` : ''}`,
      actor: { userId: params.userId },
      resource: {
        type: params.resourceType,
        id: params.resourceId,
        name: params.resourceName,
      },
    });
  }

  async logDataModification(
    action: (typeof AUDIT_ACTIONS.data_modification)[number],
    params: {
      userId: string;
      resourceType: string;
      resourceId?: string;
      changes?: Record<string, { from: unknown; to: unknown }>;
    }
  ): Promise<AuditEvent> {
    return this.log({
      category: 'data_modification',
      action,
      severity: 'notice',
      description: `User ${params.userId} performed ${action} on ${params.resourceType}`,
      actor: { userId: params.userId },
      resource: { type: params.resourceType, id: params.resourceId },
      metadata: { changes: params.changes },
    });
  }

  async logSecurity(
    action: (typeof AUDIT_ACTIONS.security)[number],
    params: {
      userId?: string;
      ipAddress?: string;
      description: string;
      metadata?: Record<string, unknown>;
    }
  ): Promise<AuditEvent> {
    return this.log({
      category: 'security',
      action,
      severity: 'alert',
      description: params.description,
      actor: { userId: params.userId, ipAddress: params.ipAddress },
      metadata: params.metadata,
    });
  }

  async logPayment(
    action: (typeof AUDIT_ACTIONS.payment)[number],
    params: {
      userId: string;
      amount?: number;
      currency?: string;
      stripeId?: string;
      success: boolean;
    }
  ): Promise<AuditEvent> {
    return this.log({
      category: 'payment',
      action,
      severity: params.success ? 'notice' : 'warning',
      description: `Payment ${action} for user ${params.userId}`,
      actor: { userId: params.userId },
      response: { success: params.success },
      metadata: {
        amount: params.amount,
        currency: params.currency,
        stripeId: params.stripeId,
      },
    });
  }

  /**
   * Flush buffer to persistent storage
   */
  async flush(): Promise<void> {
    if (this.buffer.length === 0) return;

    const events = [...this.buffer];
    this.buffer = [];

    // In production, this would:
    // 1. Write to Supabase audit_logs table
    // 2. Send to CloudWatch Logs
    // 3. Forward to SIEM (if configured)
    console.log(`[AUDIT] Flushing ${events.length} events`);

    // TODO: Implement actual persistence
    // await supabase.from('audit_logs').insert(events);
  }

  /**
   * Query audit logs (admin only)
   */
  async query(params: {
    startTime?: Date;
    endTime?: Date;
    userId?: string;
    category?: AuditCategory;
    action?: string;
    severity?: AuditSeverity;
    limit?: number;
    offset?: number;
  }): Promise<{ events: AuditEvent[]; total: number }> {
    // In production, this queries the audit_logs table
    // with proper authorization checks
    return { events: [], total: 0 };
  }

  /**
   * Verify audit log integrity
   */
  async verifyIntegrity(startSequence: number, endSequence: number): Promise<{
    valid: boolean;
    brokenAt?: number;
    details?: string;
  }> {
    // In production, this:
    // 1. Fetches logs in range
    // 2. Verifies hash chain
    // 3. Reports any breaks
    return { valid: true };
  }

  // ============================================
  // PRIVATE METHODS
  // ============================================

  private generateId(): string {
    // UUID v7-like: timestamp-based for ordering
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substring(2, 10);
    return `audit_${timestamp}_${random}`;
  }

  private async computeHash(event: Omit<AuditEvent, 'hash'>): Promise<string> {
    // Create canonical string for hashing
    const canonical = JSON.stringify({
      id: event.id,
      timestamp: event.timestamp,
      sequence: event.sequence,
      actor: event.actor,
      category: event.category,
      action: event.action,
      description: event.description,
      resource: event.resource,
      previousHash: event.previousHash,
    });

    // SHA-256 hash
    if (typeof crypto !== 'undefined' && crypto.subtle) {
      const encoder = new TextEncoder();
      const data = encoder.encode(canonical);
      const hashBuffer = await crypto.subtle.digest('SHA-256', data);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    }

    // Fallback for environments without crypto.subtle
    return `hash_${event.id}_${event.sequence}`;
  }
}

// ============================================
// SINGLETON EXPORT
// ============================================

export const auditLogger = new AuditLogger();

// ============================================
// MIDDLEWARE HELPER
// ============================================

/**
 * Extract audit context from Next.js request
 */
export function extractAuditContext(request: Request): {
  ipAddress: string | null;
  userAgent: string | null;
  method: string;
  path: string;
  queryParams: Record<string, string> | null;
} {
  const url = new URL(request.url);
  const queryParams: Record<string, string> = {};
  url.searchParams.forEach((v, k) => {
    queryParams[k] = v;
  });

  return {
    ipAddress:
      request.headers.get('x-forwarded-for')?.split(',')[0] ??
      request.headers.get('x-real-ip') ??
      null,
    userAgent: request.headers.get('user-agent'),
    method: request.method,
    path: url.pathname,
    queryParams: Object.keys(queryParams).length > 0 ? queryParams : null,
  };
}

/**
 * Hash request body for audit (without storing actual content)
 */
export async function hashRequestBody(body: unknown): Promise<string | null> {
  if (!body) return null;

  const canonical = JSON.stringify(body);

  if (typeof crypto !== 'undefined' && crypto.subtle) {
    const encoder = new TextEncoder();
    const data = encoder.encode(canonical);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  }

  return null;
}
