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

/**
 * Phase states for detection
 */
type Phase = 'Stable' | 'Approaching' | 'Critical' | 'Transitioning';

/**
 * Internal detector state
 */
interface InternalDetectorState {
  phase: Phase;
  variance: number;
  mean: number;
  observations: number;
}

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
  sentimentTrend: number[];
  currentSentiment: number;
  sentimentVariance: number;
  phase: Phase;
  prediction: 'beat' | 'miss' | 'inline' | 'uncertain';
  confidence: number;
}

/**
 * Sensitivity presets (variance thresholds)
 */
const SENSITIVITY_THRESHOLDS = {
  low: { threshold: 2.5, windowSize: 50 },
  medium: { threshold: 2.0, windowSize: 30 },
  high: { threshold: 1.5, windowSize: 20 },
};

/**
 * Simple variance-based detector
 */
class SimpleVarianceDetector {
  private values: number[] = [];
  private windowSize: number;
  private threshold: number;

  constructor(windowSize = 30, threshold = 2.0) {
    this.windowSize = windowSize;
    this.threshold = threshold;
  }

  update(value: number): InternalDetectorState {
    this.values.push(value);
    if (this.values.length > this.windowSize) {
      this.values.shift();
    }
    return this.current();
  }

  current(): InternalDetectorState {
    if (this.values.length < 2) {
      return {
        phase: 'Stable',
        variance: 0,
        mean: this.values[0] ?? 0,
        observations: this.values.length,
      };
    }

    const mean = this.values.reduce((a, b) => a + b, 0) / this.values.length;
    const variance =
      this.values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (this.values.length - 1);

    const phase = this.varianceToPhase(variance);

    return {
      phase,
      variance,
      mean,
      observations: this.values.length,
    };
  }

  reset(): void {
    this.values = [];
  }

  private varianceToPhase(variance: number): Phase {
    if (variance < this.threshold * 0.5) return 'Stable';
    if (variance < this.threshold) return 'Approaching';
    if (variance < this.threshold * 1.5) return 'Critical';
    return 'Transitioning';
  }
}

export class SocialPulseDetector {
  private sources: DataSource[] = [];
  private filters: PostFilter[] = [];
  private config: SocialPulseConfig;

  // Phase transition detectors by region/topic
  private detectors = new Map<string, SimpleVarianceDetector>();

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
    await Promise.all(this.sources.map((s) => s.init()));
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

    const results = await Promise.allSettled(this.sources.map((source) => source.fetch(params)));

    for (const result of results) {
      if (result.status === 'fulfilled') {
        allPosts.push(...result.value);
      }
    }

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

    for (const post of posts) {
      if (post.sentimentScore === undefined) {
        post.sentimentScore = this.calculateSentiment(post.content);
      }
    }

    const aggregates = this.aggregatePosts(posts);

    for (const agg of aggregates) {
      this.updateDetector(agg.id, agg.avgSentiment, agg.variance);
    }

    const state = this.calculateGlobalState(aggregates);

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

    this.earningsSentiment.set(ticker.toUpperCase(), {
      ticker: ticker.toUpperCase(),
      daysUntilReport: Math.ceil((reportDate.getTime() - Date.now()) / (1000 * 60 * 60 * 24)),
      sentimentTrend: [],
      currentSentiment: 0,
      sentimentVariance: 0,
      phase: 'Stable',
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

    const posts = await this.fetch({
      keywords: [ticker, `$${ticker}`],
      limit: 100,
    });

    if (posts.length === 0) return entry;

    const sentiments = posts
      .map((p) => p.sentimentScore)
      .filter((s): s is number => s !== undefined);

    if (sentiments.length === 0) return entry;

    const avgSentiment = sentiments.reduce((a, b) => a + b, 0) / sentiments.length;
    const variance = this.calculateVariance(sentiments);

    entry.sentimentTrend.push(avgSentiment);
    if (entry.sentimentTrend.length > 30) {
      entry.sentimentTrend.shift();
    }

    entry.currentSentiment = avgSentiment;
    entry.sentimentVariance = variance;

    const calendarEntry = this.earningsCalendar.find((e) => e.ticker === ticker.toUpperCase());
    entry.daysUntilReport = calendarEntry
      ? Math.ceil((calendarEntry.reportDate.getTime() - Date.now()) / (1000 * 60 * 60 * 24))
      : 0;

    const detector = this.getOrCreateDetector(`earnings:${ticker}`);
    detector.update(avgSentiment);
    entry.phase = detector.current().phase;

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

  // ============ Private Methods ============

  private ensureInitialized(): void {
    if (!this.initialized) {
      throw new Error('SocialPulseDetector not initialized. Call init() first.');
    }
  }

  private getOrCreateDetector(id: string): SimpleVarianceDetector {
    let detector = this.detectors.get(id);
    if (!detector) {
      const preset = SENSITIVITY_THRESHOLDS[this.config.sensitivity ?? 'medium'];
      detector = new SimpleVarianceDetector(preset.windowSize, preset.threshold);
      this.detectors.set(id, detector);
    }
    return detector;
  }

  private updateDetector(id: string, sentiment: number, variance: number): void {
    const detector = this.getOrCreateDetector(id);
    detector.update(variance);

    let history = this.sentimentHistory.get(id);
    if (!history) {
      history = [];
      this.sentimentHistory.set(id, history);
    }
    history.push(sentiment);
    if (history.length > 100) {
      history.shift();
    }
  }

  private aggregatePosts(posts: SocialPost[]): SentimentAggregate[] {
    const byRegion = new Map<string, SocialPost[]>();

    for (const post of posts) {
      const region = post.geo?.countryCode ?? 'global';
      let regionPosts = byRegion.get(region);
      if (!regionPosts) {
        regionPosts = [];
        byRegion.set(region, regionPosts);
      }
      regionPosts.push(post);
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

      const keywords = this.extractKeywords(regionPosts);

      const platformCounts: Partial<Record<Platform, number>> = {};
      for (const post of regionPosts) {
        platformCounts[post.platform] = (platformCounts[post.platform] ?? 0) + 1;
      }

      const history = this.sentimentHistory.get(region);
      const previousVariance =
        history && history.length >= 2 ? this.calculateVariance(history.slice(-10, -1)) : undefined;

      const aggregate: SentimentAggregate = {
        id: region,
        windowStart: new Date(Date.now() - 3600000).toISOString(),
        windowEnd: new Date().toISOString(),
        postCount: regionPosts.length,
        authorCount: new Set(regionPosts.map((p) => p.author.id)).size,
        avgSentiment,
        sentimentStdDev: Math.sqrt(variance),
        negativeRatio: negative / sentiments.length,
        botFilteredRatio: botFiltered / Math.max(posts.length, 1),
        topKeywords: keywords.slice(0, 10),
        platformBreakdown: platformCounts,
        variance,
      };

      // Only add optional properties if they have values
      if (region !== 'global') {
        aggregate.countryCode = region;
      }
      if (previousVariance !== undefined) {
        aggregate.previousVariance = previousVariance;
      }

      aggregates.push(aggregate);
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
        hotspots.push({
          countryCode: id,
          level,
          variance: state.variance,
          topKeywords: [],
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
    const threshold = SENSITIVITY_THRESHOLDS[this.config.sensitivity ?? 'medium'].threshold;
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
      case 'Stable':
        return 'calm';
      case 'Approaching':
        return 'stirring';
      case 'Critical':
        return 'unrest';
      case 'Transitioning':
        return 'volatile';
      default:
        return 'calm';
    }
  }

  /**
   * Simple sentiment calculation
   */
  private calculateSentiment(text: string): number {
    const lower = text.toLowerCase();

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

    return Math.max(-1, Math.min(1, score));
  }

  private extractKeywords(posts: SocialPost[]): { word: string; count: number }[] {
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

  private predictEarnings(sentiment: EarningsSentiment): 'beat' | 'miss' | 'inline' | 'uncertain' {
    const trend = sentiment.sentimentTrend;
    if (trend.length < 5) return 'uncertain';

    const recentAvg = trend.slice(-5).reduce((a, b) => a + b, 0) / 5;
    const olderSlice = trend.slice(-10, -5);
    const olderAvg =
      olderSlice.length > 0 ? olderSlice.reduce((a, b) => a + b, 0) / olderSlice.length : recentAvg;

    const trendDirection = recentAvg - olderAvg;

    if (sentiment.phase === 'Transitioning' || sentiment.phase === 'Approaching') {
      return 'uncertain';
    }

    if (trendDirection > 0.2 && sentiment.currentSentiment > 0.3) {
      return 'beat';
    }

    if (trendDirection < -0.2 && sentiment.currentSentiment < -0.3) {
      return 'miss';
    }

    if (Math.abs(sentiment.currentSentiment) < 0.2 && Math.abs(trendDirection) < 0.1) {
      return 'inline';
    }

    return 'uncertain';
  }

  private calculatePredictionConfidence(sentiment: EarningsSentiment): number {
    let confidence = 0.3;

    if (sentiment.sentimentTrend.length >= 20) confidence += 0.2;
    else if (sentiment.sentimentTrend.length >= 10) confidence += 0.1;

    const variance = this.calculateVariance(sentiment.sentimentTrend);
    if (variance < 0.1) confidence += 0.2;
    else if (variance < 0.2) confidence += 0.1;

    if (sentiment.daysUntilReport > 14) confidence -= 0.1;

    if (sentiment.prediction === 'uncertain') confidence -= 0.2;

    return Math.max(0, Math.min(1, confidence));
  }
}
