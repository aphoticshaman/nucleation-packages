/**
 * SocialPulseDetector
 *
 * Phase transition detector for social sentiment.
 * Aggregates data from multiple sources, applies filters,
 * and detects variance changes that precede upheavals.
 *
 * Use cases:
 * - Social unrest / revolution precursors
 * - Earnings sentiment (Q1/Q2/Q3/Q4 reports)
 * - Geopolitical shifts
 * - Market sentiment regime changes
 */

import type {
  DataSource,
  SearchParams,
  SocialPost,
  PostFilter,
  SentimentAggregate,
  UpheavalState,
  UpheavalLevel,
  Platform,
} from './types.js';

// Import nucleation-wasm for phase transition detection
import { NucleationDetector, Shepherd, Phase, type DetectorConfig } from 'nucleation-wasm';

/**
 * SocialPulseDetector configuration
 */
export interface SocialPulseConfig {
  /** Data sources to use */
  sources?: DataSource[];
  /** Post filters to apply */
  filters?: PostFilter[];
  /** Window size for variance calculation */
  windowSize?: number;
  /** Sensitivity: low, medium, high */
  sensitivity?: 'low' | 'medium' | 'high';
  /** Aggregate by region */
  aggregateByRegion?: boolean;
  /** Custom WASM detector config */
  detectorConfig?: DetectorConfig;
}

/**
 * Earnings calendar entry
 */
interface EarningsEntry {
  ticker: string;
  companyName: string;
  reportDate: Date;
  quarter: 'Q1' | 'Q2' | 'Q3' | 'Q4';
  fiscalYear: number;
}

/**
 * Earnings sentiment tracking
 */
interface EarningsSentiment {
  ticker: string;
  daysUntilReport: number;
  sentimentTrend: number[]; // Last 30 days
  currentSentiment: number;
  sentimentVariance: number;
  phase: Phase;
  prediction: 'beat' | 'miss' | 'inline' | 'uncertain';
  confidence: number;
}

/**
 * Sensitivity presets
 */
const SENSITIVITY_PRESETS: Record<string, DetectorConfig> = {
  low: { windowSize: 50, threshold: 2.5, minVarianceRatio: 1.8 },
  medium: { windowSize: 30, threshold: 2.0, minVarianceRatio: 1.5 },
  high: { windowSize: 20, threshold: 1.5, minVarianceRatio: 1.3 },
};

export class SocialPulseDetector {
  private sources: DataSource[] = [];
  private filters: PostFilter[] = [];
  private config: SocialPulseConfig;

  // Phase transition detectors by region/topic
  private detectors = new Map<string, NucleationDetector>();
  private shepherd: Shepherd | null = null;

  // Sentiment history for variance tracking
  private sentimentHistory = new Map<string, number[]>();

  // Earnings tracking
  private earningsCalendar: EarningsEntry[] = [];
  private earningsSentiment = new Map<string, EarningsSentiment>();

  // Ready state
  private initialized = false;

  constructor(config: SocialPulseConfig = {}) {
    this.config = {
      windowSize: config.windowSize ?? 30,
      sensitivity: config.sensitivity ?? 'medium',
      aggregateByRegion: config.aggregateByRegion ?? true,
      ...config,
    };

    if (config.sources) {
      this.sources = config.sources;
    }
    if (config.filters) {
      this.filters = config.filters;
    }
  }

  /**
   * Initialize detector and all sources
   */
  async init(): Promise<void> {
    // Initialize all sources
    await Promise.all(this.sources.map((s) => s.init()));

    // Initialize shepherd for correlation
    this.shepherd = new Shepherd();

    this.initialized = true;
  }

  /**
   * Add a data source
   */
  addSource(source: DataSource): void {
    this.sources.push(source);
  }

  /**
   * Add a post filter
   */
  addFilter(filter: PostFilter): void {
    this.filters.push(filter);
  }

  /**
   * Fetch and process posts from all sources
   */
  async fetch(params: SearchParams = {}): Promise<SocialPost[]> {
    this.ensureInitialized();

    const allPosts: SocialPost[] = [];

    // Fetch from all sources in parallel
    const results = await Promise.allSettled(this.sources.map((source) => source.fetch(params)));

    for (const result of results) {
      if (result.status === 'fulfilled') {
        allPosts.push(...result.value);
      }
    }

    // Apply filters
    const filteredPosts: SocialPost[] = [];
    for (const post of allPosts) {
      let current: SocialPost | null = post;

      for (const filter of this.filters) {
        if (!current) break;
        current = await filter.process(current);
      }

      if (current) {
        filteredPosts.push(current);
      }
    }

    return filteredPosts;
  }

  /**
   * Update detector with new posts and return current state
   */
  async update(params: SearchParams = {}): Promise<{
    state: UpheavalState;
    aggregates: SentimentAggregate[];
    posts: SocialPost[];
  }> {
    const posts = await this.fetch(params);

    // Calculate sentiment for each post if not already done
    for (const post of posts) {
      if (post.sentimentScore === undefined) {
        post.sentimentScore = this.calculateSentiment(post.content);
      }
    }

    // Aggregate by region
    const aggregates = this.aggregatePosts(posts);

    // Update detectors with aggregated variance
    for (const agg of aggregates) {
      this.updateDetector(agg.id, agg.avgSentiment, agg.variance);
    }

    // Calculate global state
    const state = this.calculateGlobalState(aggregates);

    // Update shepherd for correlation
    if (this.shepherd) {
      for (const agg of aggregates) {
        this.shepherd.addDetector(agg.id, this.getOrCreateDetector(agg.id));
      }
    }

    return { state, aggregates, posts };
  }

  /**
   * Get current upheaval state
   */
  current(): UpheavalState {
    const hotspots = this.getHotspots();
    const globalVariance = this.calculateGlobalVariance();

    return {
      level: this.varianceToLevel(globalVariance),
      levelNumeric: this.varianceToNumeric(globalVariance),
      variance: globalVariance,
      mean: this.calculateGlobalMean(),
      dataPoints: this.getTotalDataPoints(),
      lastUpdate: new Date().toISOString(),
      hotspots,
    };
  }

  /**
   * Track earnings sentiment for a ticker
   */
  trackEarnings(ticker: string, reportDate: Date, quarter: 'Q1' | 'Q2' | 'Q3' | 'Q4'): void {
    this.earningsCalendar.push({
      ticker: ticker.toUpperCase(),
      companyName: ticker,
      reportDate,
      quarter,
      fiscalYear: reportDate.getFullYear(),
    });

    // Initialize sentiment tracking
    this.earningsSentiment.set(ticker.toUpperCase(), {
      ticker: ticker.toUpperCase(),
      daysUntilReport: Math.ceil((reportDate.getTime() - Date.now()) / (1000 * 60 * 60 * 24)),
      sentimentTrend: [],
      currentSentiment: 0,
      sentimentVariance: 0,
      phase: Phase.Calm,
      prediction: 'uncertain',
      confidence: 0,
    });
  }

  /**
   * Update earnings sentiment with new data
   */
  async updateEarningsSentiment(ticker: string): Promise<EarningsSentiment | null> {
    const entry = this.earningsSentiment.get(ticker.toUpperCase());
    if (!entry) return null;

    // Fetch posts about this ticker
    const posts = await this.fetch({
      keywords: [ticker, `$${ticker}`],
      limit: 100,
    });

    if (posts.length === 0) return entry;

    // Calculate average sentiment
    const sentiments = posts
      .map((p) => p.sentimentScore)
      .filter((s): s is number => s !== undefined);

    if (sentiments.length === 0) return entry;

    const avgSentiment = sentiments.reduce((a, b) => a + b, 0) / sentiments.length;
    const variance = this.calculateVariance(sentiments);

    // Update trend
    entry.sentimentTrend.push(avgSentiment);
    if (entry.sentimentTrend.length > 30) {
      entry.sentimentTrend.shift();
    }

    entry.currentSentiment = avgSentiment;
    entry.sentimentVariance = variance;
    entry.daysUntilReport = Math.ceil(
      (this.earningsCalendar.find((e) => e.ticker === ticker)?.reportDate.getTime() ??
        Date.now() - Date.now()) /
        (1000 * 60 * 60 * 24)
    );

    // Update phase from detector
    const detector = this.getOrCreateDetector(`earnings:${ticker}`);
    detector.update(avgSentiment);
    entry.phase = detector.current().phase;

    // Simple prediction based on sentiment and variance
    entry.prediction = this.predictEarnings(entry);
    entry.confidence = this.calculatePredictionConfidence(entry);

    return entry;
  }

  /**
   * Get all earnings being tracked
   */
  getTrackedEarnings(): EarningsSentiment[] {
    return [...this.earningsSentiment.values()];
  }

  /**
   * Get detectors correlated with a specific region/topic
   */
  getCorrelatedSignals(id: string): Array<{ id: string; correlation: number }> {
    if (!this.shepherd) return [];

    const correlated = this.shepherd.getCorrelations(id);
    return correlated
      .filter((c) => c.correlation > 0.5)
      .sort((a, b) => b.correlation - a.correlation);
  }

  // ============ Private Methods ============

  private ensureInitialized(): void {
    if (!this.initialized) {
      throw new Error('SocialPulseDetector not initialized. Call init() first.');
    }
  }

  private getOrCreateDetector(id: string): NucleationDetector {
    if (!this.detectors.has(id)) {
      const config =
        this.config.detectorConfig ?? SENSITIVITY_PRESETS[this.config.sensitivity ?? 'medium'];
      this.detectors.set(id, new NucleationDetector(config));
    }
    return this.detectors.get(id)!;
  }

  private updateDetector(id: string, sentiment: number, variance: number): void {
    const detector = this.getOrCreateDetector(id);

    // Feed variance as the signal (variance spikes indicate phase transitions)
    detector.update(variance);

    // Track sentiment history
    if (!this.sentimentHistory.has(id)) {
      this.sentimentHistory.set(id, []);
    }
    const history = this.sentimentHistory.get(id)!;
    history.push(sentiment);
    if (history.length > 100) {
      history.shift();
    }
  }

  private aggregatePosts(posts: SocialPost[]): SentimentAggregate[] {
    const byRegion = new Map<string, SocialPost[]>();

    for (const post of posts) {
      const region = post.geo?.countryCode ?? 'global';
      if (!byRegion.has(region)) {
        byRegion.set(region, []);
      }
      byRegion.get(region)!.push(post);
    }

    const aggregates: SentimentAggregate[] = [];

    for (const [region, regionPosts] of byRegion) {
      const sentiments = regionPosts
        .map((p) => p.sentimentScore)
        .filter((s): s is number => s !== undefined);

      if (sentiments.length === 0) continue;

      const avgSentiment = sentiments.reduce((a, b) => a + b, 0) / sentiments.length;
      const variance = this.calculateVariance(sentiments);
      const negative = sentiments.filter((s) => s < 0).length;
      const botFiltered = posts.length - regionPosts.length;

      // Extract top keywords
      const keywords = this.extractKeywords(regionPosts);

      // Platform breakdown
      const platformCounts: Partial<Record<Platform, number>> = {};
      for (const post of regionPosts) {
        platformCounts[post.platform] = (platformCounts[post.platform] ?? 0) + 1;
      }

      const history = this.sentimentHistory.get(region);
      const previousVariance =
        history && history.length >= 2 ? this.calculateVariance(history.slice(-10, -1)) : undefined;

      aggregates.push({
        id: region,
        countryCode: region !== 'global' ? region : undefined,
        windowStart: new Date(Date.now() - 3600000).toISOString(),
        windowEnd: new Date().toISOString(),
        postCount: regionPosts.length,
        authorCount: new Set(regionPosts.map((p) => p.author.id)).size,
        avgSentiment,
        sentimentStdDev: Math.sqrt(variance),
        negativeRatio: negative / sentiments.length,
        botFilteredRatio: botFiltered / posts.length,
        topKeywords: keywords.slice(0, 10),
        platformBreakdown: platformCounts,
        variance,
        previousVariance,
      });
    }

    return aggregates;
  }

  private calculateGlobalState(aggregates: SentimentAggregate[]): UpheavalState {
    const hotspots: UpheavalState['hotspots'] = [];

    for (const agg of aggregates) {
      const level = this.varianceToLevel(agg.variance);
      if (level !== 'calm') {
        hotspots.push({
          countryCode: agg.countryCode ?? agg.id,
          level,
          variance: agg.variance,
          topKeywords: agg.topKeywords.slice(0, 5).map((k) => k.word),
        });
      }
    }

    // Sort hotspots by variance descending
    hotspots.sort((a, b) => b.variance - a.variance);

    const globalVariance = this.calculateGlobalVariance();

    return {
      level: this.varianceToLevel(globalVariance),
      levelNumeric: this.varianceToNumeric(globalVariance),
      variance: globalVariance,
      mean: this.calculateGlobalMean(),
      dataPoints: this.getTotalDataPoints(),
      lastUpdate: new Date().toISOString(),
      hotspots,
    };
  }

  private calculateGlobalVariance(): number {
    let totalVariance = 0;
    let count = 0;

    for (const detector of this.detectors.values()) {
      const state = detector.current();
      totalVariance += state.variance;
      count++;
    }

    return count > 0 ? totalVariance / count : 0;
  }

  private calculateGlobalMean(): number {
    let total = 0;
    let count = 0;

    for (const history of this.sentimentHistory.values()) {
      if (history.length > 0) {
        total += history.reduce((a, b) => a + b, 0) / history.length;
        count++;
      }
    }

    return count > 0 ? total / count : 0;
  }

  private getTotalDataPoints(): number {
    let total = 0;
    for (const history of this.sentimentHistory.values()) {
      total += history.length;
    }
    return total;
  }

  private getHotspots(): UpheavalState['hotspots'] {
    const hotspots: UpheavalState['hotspots'] = [];

    for (const [id, detector] of this.detectors) {
      const state = detector.current();
      const level = this.phaseToLevel(state.phase);

      if (level !== 'calm') {
        const keywords = this.extractKeywordsFromHistory(id);
        hotspots.push({
          countryCode: id,
          level,
          variance: state.variance,
          topKeywords: keywords,
        });
      }
    }

    return hotspots.sort((a, b) => b.variance - a.variance);
  }

  private calculateVariance(values: number[]): number {
    if (values.length < 2) return 0;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map((v) => Math.pow(v - mean, 2));
    return squaredDiffs.reduce((a, b) => a + b, 0) / (values.length - 1);
  }

  private varianceToLevel(variance: number): UpheavalLevel {
    const threshold = SENSITIVITY_PRESETS[this.config.sensitivity ?? 'medium'].threshold;
    if (variance < threshold * 0.5) return 'calm';
    if (variance < threshold) return 'stirring';
    if (variance < threshold * 1.5) return 'unrest';
    return 'volatile';
  }

  private varianceToNumeric(variance: number): number {
    const level = this.varianceToLevel(variance);
    const mapping: Record<UpheavalLevel, number> = {
      calm: 0,
      stirring: 1,
      unrest: 2,
      volatile: 3,
    };
    return mapping[level];
  }

  private phaseToLevel(phase: Phase): UpheavalLevel {
    switch (phase) {
      case Phase.Calm:
        return 'calm';
      case Phase.PreTransition:
        return 'stirring';
      case Phase.Transition:
        return 'unrest';
      case Phase.PostTransition:
        return 'volatile';
      default:
        return 'calm';
    }
  }

  /**
   * Simple sentiment calculation
   * In production, use a proper NLP model
   */
  private calculateSentiment(text: string): number {
    const lower = text.toLowerCase();

    // Simple word lists (would use ML in production)
    const positiveWords = [
      'good',
      'great',
      'excellent',
      'amazing',
      'wonderful',
      'fantastic',
      'love',
      'happy',
      'joy',
      'success',
      'win',
      'best',
      'awesome',
      'perfect',
      'beautiful',
      'hope',
      'bullish',
      'growth',
      'profit',
      'gain',
      'rally',
      'surge',
      'boom',
    ];

    const negativeWords = [
      'bad',
      'terrible',
      'awful',
      'horrible',
      'hate',
      'sad',
      'angry',
      'fear',
      'fail',
      'loss',
      'worst',
      'poor',
      'ugly',
      'disaster',
      'crisis',
      'crash',
      'bearish',
      'decline',
      'drop',
      'plunge',
      'collapse',
      'recession',
      'war',
      'protest',
      'riot',
      'unrest',
      'violence',
      'conflict',
      'death',
      'kill',
    ];

    let score = 0;
    const words = lower.split(/\s+/);

    for (const word of words) {
      if (positiveWords.includes(word)) score += 0.1;
      if (negativeWords.includes(word)) score -= 0.1;
    }

    // Clamp to -1 to 1
    return Math.max(-1, Math.min(1, score));
  }

  private extractKeywords(posts: SocialPost[]): Array<{ word: string; count: number }> {
    const wordCounts = new Map<string, number>();
    const stopwords = new Set([
      'the',
      'a',
      'an',
      'is',
      'are',
      'was',
      'were',
      'be',
      'been',
      'being',
      'have',
      'has',
      'had',
      'do',
      'does',
      'did',
      'will',
      'would',
      'could',
      'should',
      'may',
      'might',
      'must',
      'and',
      'or',
      'but',
      'if',
      'then',
      'than',
      'so',
      'as',
      'of',
      'at',
      'by',
      'for',
      'with',
      'about',
      'to',
      'from',
      'in',
      'on',
      'it',
      'its',
      'this',
      'that',
      'these',
      'those',
    ]);

    for (const post of posts) {
      const words = post.content
        .toLowerCase()
        .replace(/[^\w\s#@]/g, '')
        .split(/\s+/)
        .filter((w) => w.length > 2 && !stopwords.has(w));

      for (const word of words) {
        wordCounts.set(word, (wordCounts.get(word) ?? 0) + 1);
      }
    }

    return [...wordCounts.entries()]
      .map(([word, count]) => ({ word, count }))
      .sort((a, b) => b.count - a.count);
  }

  private extractKeywordsFromHistory(_id: string): string[] {
    // Would track keywords per region in production
    return [];
  }

  private predictEarnings(sentiment: EarningsSentiment): 'beat' | 'miss' | 'inline' | 'uncertain' {
    const trend = sentiment.sentimentTrend;
    if (trend.length < 5) return 'uncertain';

    // Calculate trend direction
    const recentAvg = trend.slice(-5).reduce((a, b) => a + b, 0) / 5;
    const olderAvg =
      trend.slice(-10, -5).reduce((a, b) => a + b, 0) / Math.min(5, trend.slice(-10, -5).length);

    const trendDirection = recentAvg - olderAvg;

    // High variance near report = uncertainty
    if (sentiment.phase === Phase.Transition || sentiment.phase === Phase.PreTransition) {
      return 'uncertain';
    }

    // Strong positive trend
    if (trendDirection > 0.2 && sentiment.currentSentiment > 0.3) {
      return 'beat';
    }

    // Strong negative trend
    if (trendDirection < -0.2 && sentiment.currentSentiment < -0.3) {
      return 'miss';
    }

    // Stable sentiment near zero
    if (Math.abs(sentiment.currentSentiment) < 0.2 && Math.abs(trendDirection) < 0.1) {
      return 'inline';
    }

    return 'uncertain';
  }

  private calculatePredictionConfidence(sentiment: EarningsSentiment): number {
    // Base confidence on data quality
    let confidence = 0.3;

    // More data = higher confidence
    if (sentiment.sentimentTrend.length >= 20) confidence += 0.2;
    else if (sentiment.sentimentTrend.length >= 10) confidence += 0.1;

    // Consistent trend = higher confidence
    const variance = this.calculateVariance(sentiment.sentimentTrend);
    if (variance < 0.1) confidence += 0.2;
    else if (variance < 0.2) confidence += 0.1;

    // Far from report = lower confidence
    if (sentiment.daysUntilReport > 14) confidence -= 0.1;

    // Uncertain prediction = lower confidence
    if (sentiment.prediction === 'uncertain') confidence -= 0.2;

    return Math.max(0, Math.min(1, confidence));
  }
}
