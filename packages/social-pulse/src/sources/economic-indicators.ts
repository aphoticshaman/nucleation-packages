/**
 * Economic Indicators Data Source
 *
 * Aggregates public macroeconomic and social indicators:
 * - Employment/unemployment (BLS, OECD)
 * - Housing (FRED)
 * - Education metrics (NCES, UNESCO)
 * - Population/census (UN, Census Bureau)
 * - Freedom indices (Freedom House, V-Dem)
 *
 * All sources are public and free.
 */

import type { DataSource, SearchParams, SocialPost } from '../types.js';

// FRED (Federal Reserve Economic Data) - free, no auth required
const FRED_BASE = 'https://api.stlouisfed.org/fred';

// World Bank Open Data API - free, no auth required
const WORLD_BANK_BASE = 'https://api.worldbank.org/v2';

/**
 * Economic indicator types
 */
export type IndicatorType =
  | 'unemployment'
  | 'employment'
  | 'housing_starts'
  | 'housing_prices'
  | 'gdp'
  | 'inflation'
  | 'interest_rate'
  | 'education_enrollment'
  | 'education_completion'
  | 'population'
  | 'population_growth'
  | 'freedom_index'
  | 'democracy_index'
  | 'press_freedom';

/**
 * Indicator observation
 */
export interface IndicatorObservation {
  indicator: IndicatorType;
  countryCode: string;
  value: number;
  date: string;
  source: string;
  unit?: string;
}

/**
 * Freedom index data point
 */
export interface FreedomDataPoint {
  countryCode: string;
  year: number;
  score: number; // 0-100 typically
  category: 'free' | 'partly_free' | 'not_free';
  civilLiberties?: number;
  politicalRights?: number;
  trend: 'improving' | 'stable' | 'declining';
  source: string;
}

/**
 * Trajectory/vector analysis
 */
export interface TrajectoryAnalysis {
  countryCode: string;
  indicator: IndicatorType;
  currentValue: number;
  historicalValues: Array<{ date: string; value: number }>;
  velocity: number; // Rate of change
  acceleration: number; // Change in rate of change
  predictedValue: number; // Next period prediction
  predictedTrend: 'up' | 'down' | 'stable';
  confidence: number;
}

/**
 * FRED series IDs for common indicators
 */
const FRED_SERIES: Record<string, IndicatorType> = {
  UNRATE: 'unemployment',
  PAYEMS: 'employment',
  HOUST: 'housing_starts',
  CSUSHPINSA: 'housing_prices',
  GDP: 'gdp',
  CPIAUCSL: 'inflation',
  FEDFUNDS: 'interest_rate',
};

/**
 * World Bank indicator codes
 */
const WORLD_BANK_INDICATORS: Record<string, IndicatorType> = {
  'SL.UEM.TOTL.ZS': 'unemployment',
  'NY.GDP.MKTP.CD': 'gdp',
  'SP.POP.TOTL': 'population',
  'SP.POP.GROW': 'population_growth',
  'SE.PRM.ENRR': 'education_enrollment',
  'SE.TER.CUAT.BA.ZS': 'education_completion',
  'FP.CPI.TOTL.ZG': 'inflation',
};

export class EconomicIndicatorsSource implements DataSource {
  readonly platform = 'custom' as const;
  private ready = false;
  private fredApiKey?: string;
  private cache = new Map<string, { data: IndicatorObservation[]; timestamp: number }>();
  private cacheMaxAge = 3600000; // 1 hour

  // Freedom index historical data (embedded for offline use)
  private freedomData: FreedomDataPoint[] = [];

  constructor(config: { fredApiKey?: string } = {}) {
    if (config.fredApiKey) {
      this.fredApiKey = config.fredApiKey;
    }
  }

  async init(): Promise<void> {
    // Load embedded freedom index data
    this.loadFreedomData();
    this.ready = true;
  }

  isReady(): boolean {
    return this.ready;
  }

  /**
   * Fetch economic indicators as posts (for unified processing)
   */
  async fetch(params: SearchParams): Promise<SocialPost[]> {
    if (!this.ready) {
      throw new Error('Economic indicators source not initialized');
    }

    const observations: IndicatorObservation[] = [];

    // Determine which indicators to fetch based on keywords
    const indicators = this.parseIndicators(params.keywords ?? []);
    const countries = params.countries ?? ['US'];

    for (const indicator of indicators) {
      for (const country of countries) {
        try {
          const data = await this.fetchIndicator(indicator, country);
          observations.push(...data);
        } catch (error) {
          console.warn(`Failed to fetch ${indicator} for ${country}:`, error);
        }
      }
    }

    // Convert to SocialPost format for unified processing
    return observations.map((obs) => this.transformToPost(obs));
  }

  /**
   * Fetch specific indicator for a country
   */
  async fetchIndicator(
    indicator: IndicatorType,
    countryCode: string = 'US'
  ): Promise<IndicatorObservation[]> {
    const cacheKey = `${indicator}:${countryCode}`;
    const cached = this.cache.get(cacheKey);

    if (cached && Date.now() - cached.timestamp < this.cacheMaxAge) {
      return cached.data;
    }

    let observations: IndicatorObservation[] = [];

    // Try FRED for US data
    if (countryCode === 'US' && this.fredApiKey) {
      const fredEntry = Object.entries(FRED_SERIES).find(([_, type]) => type === indicator);
      const fredSeries = fredEntry?.[0];
      if (fredSeries) {
        observations = await this.fetchFred(fredSeries, indicator);
      }
    }

    // Fall back to World Bank for international data
    if (observations.length === 0) {
      const wbEntry = Object.entries(WORLD_BANK_INDICATORS).find(([_, type]) => type === indicator);
      const wbIndicator = wbEntry?.[0];
      if (wbIndicator) {
        observations = await this.fetchWorldBank(wbIndicator, countryCode, indicator);
      }
    }

    // Cache results
    this.cache.set(cacheKey, { data: observations, timestamp: Date.now() });

    return observations;
  }

  /**
   * Get freedom index trajectory for a country
   */
  getFreedomTrajectory(countryCode: string): TrajectoryAnalysis | null {
    const countryData = this.freedomData
      .filter((d) => d.countryCode === countryCode)
      .sort((a, b) => a.year - b.year);

    if (countryData.length < 3) return null;

    const values = countryData.map((d) => ({ date: `${d.year}-01-01`, value: d.score }));
    const current = values[values.length - 1]!.value;

    // Calculate velocity (rate of change)
    const recentChange =
      values.length >= 2 ? values[values.length - 1]!.value - values[values.length - 2]!.value : 0;

    // Calculate acceleration (change in rate of change)
    const previousChange =
      values.length >= 3 ? values[values.length - 2]!.value - values[values.length - 3]!.value : 0;
    const acceleration = recentChange - previousChange;

    // Simple linear prediction
    const predictedValue = current + recentChange;

    return {
      countryCode,
      indicator: 'freedom_index',
      currentValue: current,
      historicalValues: values,
      velocity: recentChange,
      acceleration,
      predictedValue: Math.max(0, Math.min(100, predictedValue)),
      predictedTrend: recentChange > 1 ? 'up' : recentChange < -1 ? 'down' : 'stable',
      confidence: Math.min(0.8, 0.3 + countryData.length * 0.05),
    };
  }

  /**
   * Get all countries with declining freedom
   */
  getDecliningFreedom(years: number = 5): FreedomDataPoint[] {
    const cutoffYear = new Date().getFullYear() - years;

    return this.freedomData
      .filter((d) => d.year >= cutoffYear && d.trend === 'declining')
      .sort((a, b) => a.score - b.score);
  }

  /**
   * Calculate trajectory for any indicator
   */
  async calculateTrajectory(
    indicator: IndicatorType,
    countryCode: string = 'US'
  ): Promise<TrajectoryAnalysis | null> {
    const observations = await this.fetchIndicator(indicator, countryCode);

    if (observations.length < 3) return null;

    const sorted = observations.sort(
      (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
    );

    const values = sorted.map((o) => ({ date: o.date, value: o.value }));
    const current = values[values.length - 1]!.value;

    // Calculate velocity
    const recentChange =
      values.length >= 2 ? values[values.length - 1]!.value - values[values.length - 2]!.value : 0;

    // Calculate acceleration
    const previousChange =
      values.length >= 3 ? values[values.length - 2]!.value - values[values.length - 3]!.value : 0;
    const acceleration = recentChange - previousChange;

    // Predict using simple momentum
    const predictedValue = current + recentChange + acceleration * 0.5;

    return {
      countryCode,
      indicator,
      currentValue: current,
      historicalValues: values.slice(-20),
      velocity: recentChange,
      acceleration,
      predictedValue,
      predictedTrend: recentChange > 0 ? 'up' : recentChange < 0 ? 'down' : 'stable',
      confidence: Math.min(0.7, 0.2 + values.length * 0.02),
    };
  }

  // ============ Private Methods ============

  private async fetchFred(
    seriesId: string,
    indicatorType: IndicatorType
  ): Promise<IndicatorObservation[]> {
    const url = new URL(`${FRED_BASE}/series/observations`);
    url.searchParams.set('series_id', seriesId);
    url.searchParams.set('api_key', this.fredApiKey!);
    url.searchParams.set('file_type', 'json');
    url.searchParams.set('sort_order', 'desc');
    url.searchParams.set('limit', '100');

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`FRED API error: ${response.status}`);
    }

    const data = (await response.json()) as {
      observations?: Array<{ date: string; value: string }>;
    };

    return (data.observations ?? []).map((obs) => ({
      indicator: indicatorType,
      countryCode: 'US',
      value: parseFloat(obs.value) || 0,
      date: obs.date,
      source: 'FRED',
    }));
  }

  private async fetchWorldBank(
    indicatorCode: string,
    countryCode: string,
    indicatorType: IndicatorType
  ): Promise<IndicatorObservation[]> {
    const url = `${WORLD_BANK_BASE}/country/${countryCode}/indicator/${indicatorCode}?format=json&per_page=100`;

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`World Bank API error: ${response.status}`);
    }

    const data = (await response.json()) as unknown;

    // World Bank returns [metadata, data] array
    if (!Array.isArray(data) || data.length < 2) {
      return [];
    }

    const dataArray = data[1] as Array<{
      date: string;
      value: number | null;
      country: { id: string };
    }>;

    return (dataArray ?? [])
      .filter((d) => d.value !== null)
      .map((d) => ({
        indicator: indicatorType,
        countryCode: d.country.id,
        value: d.value!,
        date: `${d.date}-01-01`,
        source: 'World Bank',
      }));
  }

  private parseIndicators(keywords: string[]): IndicatorType[] {
    const indicators: IndicatorType[] = [];
    const keywordLower = keywords.map((k) => k.toLowerCase());

    const mappings: Record<string, IndicatorType[]> = {
      employment: ['employment', 'unemployment'],
      unemployment: ['unemployment'],
      housing: ['housing_starts', 'housing_prices'],
      education: ['education_enrollment', 'education_completion'],
      population: ['population', 'population_growth'],
      census: ['population', 'population_growth'],
      freedom: ['freedom_index'],
      democracy: ['democracy_index'],
      gdp: ['gdp'],
      inflation: ['inflation'],
      economic: ['gdp', 'inflation', 'unemployment'],
    };

    for (const keyword of keywordLower) {
      for (const [key, types] of Object.entries(mappings)) {
        if (keyword.includes(key)) {
          indicators.push(...types);
        }
      }
    }

    // Default to core indicators if none specified
    if (indicators.length === 0) {
      indicators.push('unemployment', 'gdp', 'inflation');
    }

    return [...new Set(indicators)];
  }

  private transformToPost(obs: IndicatorObservation): SocialPost {
    const content = `${obs.indicator.replace(/_/g, ' ')}: ${obs.value}${obs.unit ?? ''} (${obs.source})`;

    return {
      id: `${obs.indicator}:${obs.countryCode}:${obs.date}`,
      platform: 'custom',
      content,
      timestamp: obs.date,
      author: {
        id: obs.source,
        name: obs.source,
      },
      engagement: {},
      geo: {
        countryCode: obs.countryCode,
        source: 'inferred',
        confidence: 1,
      },
      // Use normalized value as sentiment proxy
      sentimentScore: this.normalizeIndicator(obs),
      raw: obs,
    };
  }

  /**
   * Normalize indicator to -1 to 1 scale for sentiment comparison
   */
  private normalizeIndicator(obs: IndicatorObservation): number {
    // Different normalization based on indicator type
    switch (obs.indicator) {
      case 'unemployment':
        // Higher unemployment = more negative
        return Math.max(-1, Math.min(1, -(obs.value - 5) / 10));

      case 'gdp':
        // GDP growth - assume value is % change
        return Math.max(-1, Math.min(1, obs.value / 10));

      case 'inflation': {
        // Moderate inflation (2%) is neutral, extremes are negative
        const deviation = Math.abs(obs.value - 2);
        return Math.max(-1, 1 - deviation / 5);
      }

      case 'freedom_index':
        // 0-100 scale to -1 to 1
        return (obs.value - 50) / 50;

      case 'population_growth':
        // Moderate growth is neutral
        return Math.max(-1, Math.min(1, obs.value / 3));

      default:
        return 0;
    }
  }

  /**
   * Load embedded freedom index data
   * In production, would fetch from Freedom House API
   */
  private loadFreedomData(): void {
    // Sample data - would be populated from API in production
    const sampleCountries = [
      { code: 'US', scores: [89, 89, 86, 83, 83] },
      { code: 'GB', scores: [94, 93, 93, 93, 91] },
      { code: 'DE', scores: [94, 94, 94, 94, 94] },
      { code: 'FR', scores: [90, 90, 89, 89, 89] },
      { code: 'RU', scores: [22, 20, 19, 16, 13] },
      { code: 'CN', scores: [14, 11, 10, 9, 9] },
      { code: 'IN', scores: [71, 67, 67, 66, 66] },
      { code: 'BR', scores: [75, 74, 73, 72, 72] },
      { code: 'TR', scores: [53, 32, 32, 32, 32] },
      { code: 'HU', scores: [76, 70, 69, 66, 66] },
      { code: 'PL', scores: [93, 84, 82, 81, 80] },
      { code: 'UA', scores: [60, 62, 60, 61, 50] },
      { code: 'VE', scores: [35, 26, 19, 14, 14] },
      { code: 'PH', scores: [65, 59, 56, 55, 54] },
      { code: 'MX', scores: [65, 62, 60, 60, 58] },
    ];

    const currentYear = new Date().getFullYear();

    for (const country of sampleCountries) {
      for (let i = 0; i < country.scores.length; i++) {
        const year = currentYear - country.scores.length + i + 1;
        const score = country.scores[i]!;
        const prevScore = i > 0 ? country.scores[i - 1]! : score;

        let trend: FreedomDataPoint['trend'] = 'stable';
        if (score - prevScore > 2) trend = 'improving';
        else if (score - prevScore < -2) trend = 'declining';

        let category: FreedomDataPoint['category'] = 'partly_free';
        if (score >= 70) category = 'free';
        else if (score < 40) category = 'not_free';

        this.freedomData.push({
          countryCode: country.code,
          year,
          score,
          category,
          trend,
          source: 'Freedom House',
        });
      }
    }
  }
}
