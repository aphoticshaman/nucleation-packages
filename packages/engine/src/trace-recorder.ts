/**
 * DataTraceRecorder
 *
 * Full audit trail for open-box visibility.
 * Records all operations for compliance and debugging.
 */

import { randomBytes } from 'node:crypto';
import type { TraceEntry, DataTrace, DataProvenance } from './types.js';

export class DataTraceRecorder {
  private sessionId: string;
  private startTime: string;
  private entries: TraceEntry[] = [];
  private enabled = false;

  constructor() {
    this.sessionId = this.generateId();
    this.startTime = new Date().toISOString();
  }

  /**
   * Enable/disable tracing
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (enabled && this.entries.length === 0) {
      this.sessionId = this.generateId();
      this.startTime = new Date().toISOString();
    }
  }

  /**
   * Check if tracing is enabled
   */
  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Record a trace entry
   */
  record(
    type: TraceEntry['type'],
    input: unknown,
    output: unknown,
    duration_ms: number,
    source?: string,
    metadata?: Record<string, unknown>,
    provenance?: DataProvenance
  ): void {
    if (!this.enabled) return;

    const entry: TraceEntry = {
      id: this.generateId(),
      timestamp: new Date().toISOString(),
      type,
      input: this.sanitize(input),
      output: this.sanitize(output),
      duration_ms,
    };

    if (source) entry.source = source;
    if (metadata) entry.metadata = metadata;
    if (provenance) entry.provenance = provenance;

    this.entries.push(entry);
  }

  /**
   * Export complete trace
   */
  export(): DataTrace {
    const byType: Record<string, number> = {};
    const sourcesUsed = new Set<string>();
    const filtersApplied = new Set<string>();
    let totalDuration = 0;

    for (const entry of this.entries) {
      byType[entry.type] = (byType[entry.type] ?? 0) + 1;
      totalDuration += entry.duration_ms;

      if (entry.source) {
        if (entry.type === 'fetch') sourcesUsed.add(entry.source);
        if (entry.type === 'filter') filtersApplied.add(entry.source);
      }
    }

    return {
      sessionId: this.sessionId,
      startTime: this.startTime,
      endTime: new Date().toISOString(),
      entries: this.entries,
      summary: {
        totalEntries: this.entries.length,
        byType,
        totalDuration_ms: totalDuration,
        sourcesUsed: [...sourcesUsed],
        filtersApplied: [...filtersApplied],
      },
    };
  }

  /**
   * Export as JSON string
   */
  toJSON(): string {
    return JSON.stringify(this.export(), null, 2);
  }

  /**
   * Export as CSV (entries only)
   */
  toCSV(): string {
    const headers = ['id', 'timestamp', 'type', 'source', 'duration_ms', 'tier'];
    const rows = this.entries.map((e) => [
      e.id,
      e.timestamp,
      e.type,
      e.source ?? '',
      String(e.duration_ms),
      e.provenance?.sourceTier ?? '',
    ]);
    return [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
  }

  /**
   * Get entries by type
   */
  getEntriesByType(type: TraceEntry['type']): TraceEntry[] {
    return this.entries.filter((e) => e.type === type);
  }

  /**
   * Get entries by source tier
   */
  getEntriesByTier(tier: DataProvenance['sourceTier']): TraceEntry[] {
    return this.entries.filter((e) => e.provenance?.sourceTier === tier);
  }

  /**
   * Clear trace history
   */
  clear(): void {
    this.entries = [];
    this.sessionId = this.generateId();
    this.startTime = new Date().toISOString();
  }

  private generateId(): string {
    const bytes = randomBytes(8);
    const hex = bytes.toString('hex');
    return `trace_${Date.now()}_${hex}`;
  }

  private sanitize(data: unknown): unknown {
    const str = JSON.stringify(data);
    if (str.length > 10000) {
      return { _truncated: true, _length: str.length, _preview: str.slice(0, 500) };
    }
    return data;
  }
}
