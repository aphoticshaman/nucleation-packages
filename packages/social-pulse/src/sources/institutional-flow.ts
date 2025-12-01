/**
 * Institutional Flow Tracker
 *
 * Tracks granular stock movements from institutional sources.
 * Correlates timing patterns for early signals.
 *
 * Data sources:
 * - SEC EDGAR Form 4 (insider transactions)
 * - 13F filings (institutional holdings)
 * - Unusual options activity
 * - Public disclosure feeds
 */

import type { DataSource, SearchParams, SocialPost } from '../types.js';

// SEC EDGAR API base
const SEC_EDGAR_BASE = 'https://data.sec.gov';

/**
 * Tracked transaction record
 */
interface InstitutionalTransaction {
  id: string;
  filer: string;
  filerRole?: string;
  ticker: string;
  companyName: string;
  transactionType: 'buy' | 'sell' | 'exercise' | 'gift';
  shares: number;
  pricePerShare?: number;
  totalValue?: number;
  transactionDate: string;
  filingDate: string;
  sector?: string;
  /** Internal classification */
  sourceCategory: 'sec_form4' | 'sec_13f' | 'disclosure' | 'options';
}

/**
 * Sector aggregation for pattern detection
 */
interface SectorFlow {
  sector: string;
  netFlow: number; // positive = buying, negative = selling
  transactionCount: number;
  topBuyers: string[];
  topSellers: string[];
  periodStart: string;
  periodEnd: string;
}

/**
 * Signal when unusual patterns detected
 */
interface FlowSignal {
  type: 'sector_rotation' | 'concentrated_selling' | 'unusual_timing' | 'cluster_activity';
  severity: 'low' | 'medium' | 'high';
  sectors: string[];
  description: string;
  timestamp: string;
  /** Correlated events if any */
  correlatedEvents?: string[];
}

// Sector mappings for pattern analysis
const SECTOR_KEYWORDS: Record<string, string[]> = {
  defense: ['raytheon', 'lockheed', 'northrop', 'boeing', 'general dynamics', 'l3harris', 'bae'],
  energy: ['exxon', 'chevron', 'conocophillips', 'schlumberger', 'halliburton', 'occidental'],
  pharma: ['pfizer', 'moderna', 'johnson', 'merck', 'abbvie', 'bristol', 'eli lilly'],
  tech: ['apple', 'microsoft', 'google', 'amazon', 'meta', 'nvidia', 'tesla'],
  finance: ['jpmorgan', 'goldman', 'morgan stanley', 'bank of america', 'citigroup', 'wells fargo'],
  industrial: ['caterpillar', 'deere', '3m', 'honeywell', 'ge'],
};

export class InstitutionalFlowSource implements DataSource {
  readonly platform = 'custom' as const;
  private ready = false;
  private transactionCache: InstitutionalTransaction[] = [];
  private lastFetch: Date | null = null;

  async init(): Promise<void> {
    this.ready = true;
  }

  isReady(): boolean {
    return this.ready;
  }

  /**
   * Fetch recent transactions
   */
  async fetch(params: SearchParams): Promise<SocialPost[]> {
    if (!this.ready) {
      throw new Error('Institutional flow source not initialized');
    }

    // Fetch fresh data if cache is stale (>1 hour)
    if (!this.lastFetch || Date.now() - this.lastFetch.getTime() > 3600000) {
      await this.refreshCache();
    }

    // Filter and transform to SocialPost format
    const filtered = this.filterTransactions(this.transactionCache, params);
    return filtered.map((tx) => this.transformToPost(tx));
  }

  /**
   * Analyze sector flow patterns
   * Returns signals when unusual patterns detected
   */
  async analyzeSectorFlow(days: number = 30): Promise<{
    flows: SectorFlow[];
    signals: FlowSignal[];
  }> {
    await this.refreshCache();

    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - days);

    const recentTx = this.transactionCache.filter((tx) => new Date(tx.transactionDate) >= cutoff);

    // Calculate sector flows
    const flows = this.calculateSectorFlows(recentTx);

    // Detect signals
    const signals = this.detectSignals(flows, recentTx);

    return { flows, signals };
  }

  /**
   * Get concentrated activity for specific sectors
   */
  async getSectorActivity(sectors: string[]): Promise<InstitutionalTransaction[]> {
    await this.refreshCache();

    return this.transactionCache.filter((tx) => {
      const txSector = tx.sector?.toLowerCase();
      return sectors.some((s) => txSector?.includes(s.toLowerCase()));
    });
  }

  /**
   * Detect timing clusters (multiple filers acting in short window)
   */
  async detectTimingClusters(
    windowHours: number = 48
  ): Promise<Array<{ transactions: InstitutionalTransaction[]; significance: number }>> {
    await this.refreshCache();

    const clusters: Array<{ transactions: InstitutionalTransaction[]; significance: number }> = [];
    const windowMs = windowHours * 60 * 60 * 1000;

    // Group by company
    const byCompany = new Map<string, InstitutionalTransaction[]>();
    for (const tx of this.transactionCache) {
      const key = tx.ticker.toUpperCase();
      if (!byCompany.has(key)) {
        byCompany.set(key, []);
      }
      byCompany.get(key)!.push(tx);
    }

    // Find clusters within time window
    for (const transactions of byCompany.values()) {
      if (transactions.length < 3) continue;

      // Sort by date
      const sorted = transactions.sort(
        (a, b) => new Date(a.transactionDate).getTime() - new Date(b.transactionDate).getTime()
      );

      // Sliding window
      for (let i = 0; i < sorted.length; i++) {
        const windowStart = new Date(sorted[i]!.transactionDate).getTime();
        const windowEnd = windowStart + windowMs;

        const inWindow = sorted.filter((tx) => {
          const txTime = new Date(tx.transactionDate).getTime();
          return txTime >= windowStart && txTime <= windowEnd;
        });

        // Multiple distinct filers in window = cluster
        const distinctFilers = new Set(inWindow.map((tx) => tx.filer));
        if (distinctFilers.size >= 3) {
          // Calculate significance based on total value and filer count
          const totalValue = inWindow.reduce((sum, tx) => sum + (tx.totalValue ?? 0), 0);
          const significance = Math.min(1, (distinctFilers.size / 5) * (totalValue / 10000000));

          clusters.push({
            transactions: inWindow,
            significance,
          });
        }
      }
    }

    // Deduplicate overlapping clusters
    return this.deduplicateClusters(clusters);
  }

  /**
   * Refresh transaction cache from sources
   */
  private async refreshCache(): Promise<void> {
    const transactions: InstitutionalTransaction[] = [];

    // Fetch SEC Form 4 data
    try {
      const form4Data = await this.fetchSecForm4();
      transactions.push(...form4Data);
    } catch (error) {
      console.warn('Failed to fetch SEC Form 4 data:', error);
    }

    this.transactionCache = transactions;
    this.lastFetch = new Date();
    this.classifySectors();
  }

  /**
   * Fetch SEC Form 4 filings
   */
  private async fetchSecForm4(): Promise<InstitutionalTransaction[]> {
    // SEC EDGAR provides RSS feed of recent filings
    const feedUrl = `${SEC_EDGAR_BASE}/cgi-bin/browse-edgar?action=getcurrent&type=4&company=&dateb=&owner=include&count=100&output=atom`;

    try {
      const response = await fetch(feedUrl, {
        headers: {
          'User-Agent': 'NucleationResearch/1.0 (research@example.com)',
        },
      });

      if (!response.ok) {
        throw new Error(`SEC API error: ${response.status}`);
      }

      const xml = await response.text();
      return this.parseSecAtomFeed(xml);
    } catch {
      // Return empty if SEC is unavailable
      return [];
    }
  }

  /**
   * Parse SEC EDGAR Atom feed
   */
  private parseSecAtomFeed(xml: string): InstitutionalTransaction[] {
    const transactions: InstitutionalTransaction[] = [];
    const entryRegex = /<entry>([\s\S]*?)<\/entry>/g;

    let match;
    while ((match = entryRegex.exec(xml)) !== null) {
      const entry = match[1];
      if (!entry) continue;

      const getId = (tag: string): string => {
        const m = new RegExp(`<${tag}[^>]*>([\\s\\S]*?)</${tag}>`).exec(entry);
        return m && m[1] ? m[1].trim() : '';
      };

      const title = getId('title');
      const updated = getId('updated');
      const linkMatch = /<link[^>]*href="([^"]+)"/.exec(entry);
      const link = (linkMatch && linkMatch[1]) || '';

      // Parse title for transaction info
      // Format: "4 - Company Name (0001234567) (Filer Name)"
      const titleMatch = /4\s*-\s*(.+?)\s*\((\d+)\)\s*\((.+?)\)/.exec(title);
      const companyName = titleMatch?.[1];
      const filer = titleMatch?.[3];
      if (companyName && filer) {
        transactions.push({
          id: link,
          filer,
          companyName,
          ticker: '', // Would need to resolve from CIK
          transactionType: 'buy', // Would need to parse actual filing
          shares: 0,
          transactionDate: updated,
          filingDate: updated,
          sourceCategory: 'sec_form4',
        });
      }
    }

    return transactions;
  }

  /**
   * Classify transactions by sector
   */
  private classifySectors(): void {
    for (const tx of this.transactionCache) {
      const nameLower = tx.companyName.toLowerCase();

      for (const [sector, keywords] of Object.entries(SECTOR_KEYWORDS)) {
        if (keywords.some((kw) => nameLower.includes(kw))) {
          tx.sector = sector;
          break;
        }
      }
    }
  }

  /**
   * Calculate net flow by sector
   */
  private calculateSectorFlows(transactions: InstitutionalTransaction[]): SectorFlow[] {
    const sectorData = new Map<
      string,
      {
        netFlow: number;
        count: number;
        buyers: Map<string, number>;
        sellers: Map<string, number>;
      }
    >();

    for (const tx of transactions) {
      const sector = tx.sector ?? 'other';
      if (!sectorData.has(sector)) {
        sectorData.set(sector, {
          netFlow: 0,
          count: 0,
          buyers: new Map(),
          sellers: new Map(),
        });
      }

      const data = sectorData.get(sector)!;
      const value = tx.totalValue ?? tx.shares * (tx.pricePerShare ?? 0);

      data.count++;
      if (tx.transactionType === 'buy') {
        data.netFlow += value;
        data.buyers.set(tx.filer, (data.buyers.get(tx.filer) ?? 0) + value);
      } else if (tx.transactionType === 'sell') {
        data.netFlow -= value;
        data.sellers.set(tx.filer, (data.sellers.get(tx.filer) ?? 0) + value);
      }
    }

    const flows: SectorFlow[] = [];
    const now = new Date().toISOString();

    for (const [sector, data] of sectorData) {
      const topBuyers = [...data.buyers.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([name]) => name);

      const topSellers = [...data.sellers.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([name]) => name);

      flows.push({
        sector,
        netFlow: data.netFlow,
        transactionCount: data.count,
        topBuyers,
        topSellers,
        periodStart: now, // Would calculate from actual data
        periodEnd: now,
      });
    }

    return flows.sort((a, b) => Math.abs(b.netFlow) - Math.abs(a.netFlow));
  }

  /**
   * Detect unusual patterns
   */
  private detectSignals(
    flows: SectorFlow[],
    _transactions: InstitutionalTransaction[]
  ): FlowSignal[] {
    const signals: FlowSignal[] = [];

    // Check for sector rotation (one sector heavily sold, another heavily bought)
    const selling = flows.filter((f) => f.netFlow < -1000000);
    const buying = flows.filter((f) => f.netFlow > 1000000);

    if (selling.length > 0 && buying.length > 0) {
      signals.push({
        type: 'sector_rotation',
        severity: 'medium',
        sectors: [...selling.map((f) => f.sector), ...buying.map((f) => f.sector)],
        description: `Rotation detected: selling ${selling.map((f) => f.sector).join(', ')}, buying ${buying.map((f) => f.sector).join(', ')}`,
        timestamp: new Date().toISOString(),
      });
    }

    // Check for concentrated selling in sensitive sectors
    const sensitiveSectors = ['defense', 'energy', 'pharma'];
    for (const sector of sensitiveSectors) {
      const flow = flows.find((f) => f.sector === sector);
      if (flow && flow.netFlow < -5000000) {
        signals.push({
          type: 'concentrated_selling',
          severity: 'high',
          sectors: [sector],
          description: `Heavy institutional selling in ${sector} sector`,
          timestamp: new Date().toISOString(),
        });
      }
    }

    return signals;
  }

  /**
   * Filter transactions by search params
   */
  private filterTransactions(
    transactions: InstitutionalTransaction[],
    params: SearchParams
  ): InstitutionalTransaction[] {
    return transactions.filter((tx) => {
      if (params.keywords?.length) {
        const searchStr = `${tx.companyName} ${tx.filer} ${tx.ticker}`.toLowerCase();
        if (!params.keywords.some((kw) => searchStr.includes(kw.toLowerCase()))) {
          return false;
        }
      }

      if (params.since && new Date(tx.transactionDate) < params.since) {
        return false;
      }

      if (params.until && new Date(tx.transactionDate) > params.until) {
        return false;
      }

      return true;
    });
  }

  /**
   * Transform transaction to standard post format
   */
  private transformToPost(tx: InstitutionalTransaction): SocialPost {
    const direction = tx.transactionType === 'buy' ? 'acquired' : 'disposed of';
    const content = `${tx.filer} ${direction} ${tx.shares.toLocaleString()} shares of ${tx.companyName} (${tx.ticker})`;

    return {
      id: tx.id,
      platform: 'custom',
      content,
      timestamp: tx.filingDate,
      author: {
        id: tx.filer,
        name: tx.filer,
      },
      engagement: {},
      raw: tx,
    };
  }

  /**
   * Remove overlapping clusters
   */
  private deduplicateClusters(
    clusters: Array<{ transactions: InstitutionalTransaction[]; significance: number }>
  ): Array<{ transactions: InstitutionalTransaction[]; significance: number }> {
    // Sort by significance descending
    const sorted = clusters.sort((a, b) => b.significance - a.significance);
    const seen = new Set<string>();
    const result: typeof clusters = [];

    for (const cluster of sorted) {
      const key = cluster.transactions
        .map((t) => t.id)
        .sort()
        .join(',');
      if (!seen.has(key)) {
        seen.add(key);
        result.push(cluster);
      }
    }

    return result;
  }
}
