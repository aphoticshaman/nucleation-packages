/**
 * AuditLog
 *
 * Immutable audit trail for compliance.
 * Records all data access and transformations.
 */

import { randomBytes, createHash } from 'node:crypto';

export interface AuditEntry {
  id: string;
  timestamp: string;
  action: AuditAction;
  actor: string;
  resource: string;
  details: Record<string, unknown>;
  hash: string;
  previousHash: string;
}

export type AuditAction =
  | 'fetch'
  | 'transform'
  | 'filter'
  | 'export'
  | 'access'
  | 'delete'
  | 'configure'
  | 'error';

export interface AuditLogConfig {
  maxEntries?: number;
  retentionDays?: number;
  actor?: string;
}

export class AuditLog {
  private entries: AuditEntry[] = [];
  private config: Required<AuditLogConfig>;
  private lastHash: string;

  constructor(config: AuditLogConfig = {}) {
    this.config = {
      maxEntries: config.maxEntries ?? 10000,
      retentionDays: config.retentionDays ?? 90,
      actor: config.actor ?? 'system',
    };
    this.lastHash = this.generateGenesisHash();
  }

  /**
   * Record an audit entry
   */
  record(
    action: AuditAction,
    resource: string,
    details: Record<string, unknown> = {},
    actor?: string
  ): AuditEntry {
    const entry: AuditEntry = {
      id: this.generateId(),
      timestamp: new Date().toISOString(),
      action,
      actor: actor ?? this.config.actor,
      resource,
      details,
      previousHash: this.lastHash,
      hash: '', // Will be set below
    };

    // Calculate hash for integrity
    entry.hash = this.calculateHash(entry);
    this.lastHash = entry.hash;

    // Add to log
    this.entries.push(entry);

    // Enforce max entries
    if (this.entries.length > this.config.maxEntries) {
      this.entries.shift();
    }

    return entry;
  }

  /**
   * Get all entries
   */
  getAll(): AuditEntry[] {
    return [...this.entries];
  }

  /**
   * Get entries by action type
   */
  getByAction(action: AuditAction): AuditEntry[] {
    return this.entries.filter((e) => e.action === action);
  }

  /**
   * Get entries by resource
   */
  getByResource(resource: string): AuditEntry[] {
    return this.entries.filter((e) => e.resource === resource);
  }

  /**
   * Get entries within time range
   */
  getByTimeRange(start: Date, end: Date): AuditEntry[] {
    return this.entries.filter((e) => {
      const ts = new Date(e.timestamp);
      return ts >= start && ts <= end;
    });
  }

  /**
   * Get entries for last N days
   */
  getLastDays(days: number): AuditEntry[] {
    const start = new Date();
    start.setDate(start.getDate() - days);
    return this.getByTimeRange(start, new Date());
  }

  /**
   * Verify integrity of audit log
   */
  verifyIntegrity(): { valid: boolean; brokenAt?: number } {
    if (this.entries.length === 0) return { valid: true };

    for (let i = 0; i < this.entries.length; i++) {
      const entry = this.entries[i];
      const calculatedHash = this.calculateHash(entry);

      if (calculatedHash !== entry.hash) {
        return { valid: false, brokenAt: i };
      }

      if (i > 0) {
        const previousEntry = this.entries[i - 1];
        if (entry.previousHash !== previousEntry.hash) {
          return { valid: false, brokenAt: i };
        }
      }
    }

    return { valid: true };
  }

  /**
   * Export audit log as JSON
   */
  exportJSON(): string {
    return JSON.stringify(
      {
        exportedAt: new Date().toISOString(),
        entryCount: this.entries.length,
        integrityCheck: this.verifyIntegrity(),
        entries: this.entries,
      },
      null,
      2
    );
  }

  /**
   * Export audit log as CSV
   */
  exportCSV(): string {
    const headers = ['id', 'timestamp', 'action', 'actor', 'resource', 'hash'];
    const rows = this.entries.map((e) => [
      e.id,
      e.timestamp,
      e.action,
      e.actor,
      e.resource,
      e.hash,
    ]);
    return [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
  }

  /**
   * Get entry count
   */
  getCount(): number {
    return this.entries.length;
  }

  /**
   * Prune old entries based on retention policy
   */
  prune(): number {
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - this.config.retentionDays);

    const before = this.entries.length;
    this.entries = this.entries.filter((e) => new Date(e.timestamp) >= cutoff);
    return before - this.entries.length;
  }

  private generateId(): string {
    const bytes = randomBytes(8);
    return `audit_${Date.now()}_${bytes.toString('hex')}`;
  }

  private generateGenesisHash(): string {
    return createHash('sha256').update('genesis').digest('hex').slice(0, 16);
  }

  private calculateHash(entry: AuditEntry): string {
    const data = JSON.stringify({
      id: entry.id,
      timestamp: entry.timestamp,
      action: entry.action,
      actor: entry.actor,
      resource: entry.resource,
      details: entry.details,
      previousHash: entry.previousHash,
    });

    return createHash('sha256').update(data).digest('hex').slice(0, 32);
  }
}
