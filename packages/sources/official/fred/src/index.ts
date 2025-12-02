/**
 * FRED Data Source
 *
 * Federal Reserve Economic Data from St. Louis Fed.
 * Requires free API key from https://fred.stlouisfed.org/docs/api/api_key.html
 *
 * Rate limit: 120 requests per minute
 * Attribution: Required - "Source: FRED, Federal Reserve Bank of St. Louis"
 */

export interface FredObservation {
  date: string;
  value: string; // Can be "." for missing data
}

export interface FredSeries {
  id: string;
  title: string;
  observation_start: string;
  observation_end: string;
  frequency: string;
  frequency_short: string;
  units: string;
  units_short: string;
  seasonal_adjustment: string;
  seasonal_adjustment_short: string;
  last_updated: string;
  popularity: number;
  notes: string;
}

export interface FredConfig {
  apiKey: string;
  baseUrl?: string;
}

// Common economic indicators
export const POPULAR_SERIES = {
  // GDP & Growth
  GDP: 'GDP', // Gross Domestic Product
  GDPC1: 'GDPC1', // Real GDP
  A191RL1Q225SBEA: 'A191RL1Q225SBEA', // Real GDP Growth Rate

  // Employment
  UNRATE: 'UNRATE', // Unemployment Rate
  PAYEMS: 'PAYEMS', // Total Nonfarm Payrolls
  ICSA: 'ICSA', // Initial Jobless Claims

  // Inflation
  CPIAUCSL: 'CPIAUCSL', // Consumer Price Index
  CPILFESL: 'CPILFESL', // Core CPI
  PCEPI: 'PCEPI', // PCE Price Index

  // Interest Rates
  FEDFUNDS: 'FEDFUNDS', // Federal Funds Rate
  DGS10: 'DGS10', // 10-Year Treasury
  DGS2: 'DGS2', // 2-Year Treasury
  T10Y2Y: 'T10Y2Y', // 10Y-2Y Spread (yield curve)

  // Money Supply
  M2SL: 'M2SL', // M2 Money Supply
  WALCL: 'WALCL', // Fed Balance Sheet

  // Housing
  HOUST: 'HOUST', // Housing Starts
  CSUSHPINSA: 'CSUSHPINSA', // Case-Shiller Home Price Index

  // Consumer
  UMCSENT: 'UMCSENT', // Consumer Sentiment
  RSXFS: 'RSXFS', // Retail Sales

  // Financial
  SP500: 'SP500', // S&P 500
  VIXCLS: 'VIXCLS', // VIX
  DTWEXBGS: 'DTWEXBGS', // Trade Weighted Dollar Index
} as const;

const BASE_URL = 'https://api.stlouisfed.org/fred';

export class FredSource {
  private apiKey: string;
  private baseUrl: string;
  private requestCount = 0;
  private lastMinuteStart = Date.now();
  private requestsThisMinute = 0;

  constructor(config: FredConfig) {
    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl ?? BASE_URL;
  }

  /**
   * Get series metadata
   */
  async getSeries(seriesId: string): Promise<FredSeries> {
    await this.respectRateLimit();

    const url = new URL(`${this.baseUrl}/series`);
    url.searchParams.set('series_id', seriesId);
    url.searchParams.set('api_key', this.apiKey);
    url.searchParams.set('file_type', 'json');

    const response = await fetch(url.toString());

    if (!response.ok) {
      throw new Error(`FRED API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json() as { seriess: FredSeries[] };
    return data.seriess[0];
  }

  /**
   * Get observations for a series
   */
  async getObservations(
    seriesId: string,
    options: {
      startDate?: string;
      endDate?: string;
      limit?: number;
      sortOrder?: 'asc' | 'desc';
    } = {}
  ): Promise<FredObservation[]> {
    await this.respectRateLimit();

    const url = new URL(`${this.baseUrl}/series/observations`);
    url.searchParams.set('series_id', seriesId);
    url.searchParams.set('api_key', this.apiKey);
    url.searchParams.set('file_type', 'json');

    if (options.startDate) {
      url.searchParams.set('observation_start', options.startDate);
    }
    if (options.endDate) {
      url.searchParams.set('observation_end', options.endDate);
    }
    if (options.limit) {
      url.searchParams.set('limit', String(options.limit));
    }
    if (options.sortOrder) {
      url.searchParams.set('sort_order', options.sortOrder);
    }

    const response = await fetch(url.toString());

    if (!response.ok) {
      throw new Error(`FRED API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json() as { observations: FredObservation[] };
    return data.observations;
  }

  /**
   * Get latest value for a series
   */
  async getLatest(seriesId: string): Promise<{ date: string; value: number } | null> {
    const observations = await this.getObservations(seriesId, {
      sortOrder: 'desc',
      limit: 1,
    });

    if (observations.length === 0) return null;

    const obs = observations[0];
    const value = parseFloat(obs.value);

    if (isNaN(value)) return null;

    return { date: obs.date, value };
  }

  /**
   * Convert observations to numeric signal
   */
  toSignal(observations: FredObservation[]): number[] {
    return observations
      .map((o) => parseFloat(o.value))
      .filter((v) => !isNaN(v));
  }

  /**
   * Get multiple series at once
   */
  async getMultipleSeries(
    seriesIds: string[],
    options: {
      startDate?: string;
      endDate?: string;
    } = {}
  ): Promise<Map<string, FredObservation[]>> {
    const results = new Map<string, FredObservation[]>();

    // Fetch in parallel but respect rate limits
    const chunks = this.chunkArray(seriesIds, 5);

    for (const chunk of chunks) {
      const promises = chunk.map(async (id) => {
        const obs = await this.getObservations(id, options);
        return { id, obs };
      });

      const chunkResults = await Promise.all(promises);
      for (const { id, obs } of chunkResults) {
        results.set(id, obs);
      }
    }

    return results;
  }

  /**
   * Get yield curve data (2Y, 10Y, spread)
   */
  async getYieldCurve(
    options: { startDate?: string; endDate?: string } = {}
  ): Promise<{
    dates: string[];
    twoYear: number[];
    tenYear: number[];
    spread: number[];
  }> {
    const [twoYearObs, tenYearObs] = await Promise.all([
      this.getObservations(POPULAR_SERIES.DGS2, options),
      this.getObservations(POPULAR_SERIES.DGS10, options),
    ]);

    // Align by date
    const twoYearMap = new Map(twoYearObs.map((o) => [o.date, parseFloat(o.value)]));
    const tenYearMap = new Map(tenYearObs.map((o) => [o.date, parseFloat(o.value)]));

    const dates: string[] = [];
    const twoYear: number[] = [];
    const tenYear: number[] = [];
    const spread: number[] = [];

    for (const date of twoYearMap.keys()) {
      const two = twoYearMap.get(date);
      const ten = tenYearMap.get(date);

      if (two !== undefined && ten !== undefined && !isNaN(two) && !isNaN(ten)) {
        dates.push(date);
        twoYear.push(two);
        tenYear.push(ten);
        spread.push(ten - two);
      }
    }

    return { dates, twoYear, tenYear, spread };
  }

  /**
   * Get source metadata for compliance
   */
  getSourceMetadata() {
    return {
      sourceId: 'fred',
      name: 'Federal Reserve Economic Data',
      tier: 'official' as const,
      url: 'https://fred.stlouisfed.org',
      license: 'Public Domain',
      attribution: 'Source: FRED, Federal Reserve Bank of St. Louis',
      rateLimit: { requestsPerMinute: 120 },
    };
  }

  private async respectRateLimit(): Promise<void> {
    const now = Date.now();

    // Reset counter every minute
    if (now - this.lastMinuteStart >= 60000) {
      this.lastMinuteStart = now;
      this.requestsThisMinute = 0;
    }

    // If at limit, wait until next minute
    if (this.requestsThisMinute >= 120) {
      const waitTime = 60000 - (now - this.lastMinuteStart);
      await new Promise((resolve) => setTimeout(resolve, waitTime));
      this.lastMinuteStart = Date.now();
      this.requestsThisMinute = 0;
    }

    this.requestsThisMinute++;
    this.requestCount++;
  }

  private chunkArray<T>(array: T[], size: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  }
}

export default FredSource;
