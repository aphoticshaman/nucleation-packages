/**
 * CDC Disease Surveillance Data Source
 *
 * Integrates with CDC's public health surveillance APIs:
 * - WONDER (Wide-ranging Online Data for Epidemiologic Research)
 * - FluView (Influenza surveillance)
 * - NNDSS (National Notifiable Diseases Surveillance System)
 * - COVID Data Tracker
 * - NHSN (National Healthcare Safety Network)
 *
 * All data is PUBLIC DOMAIN (US Government work)
 * Rate limits are generous but we respect them
 *
 * © 2025 Crystalline Labs LLC
 */

import { RateLimiter, AttributionTracker } from '@latticeforge/compliance';

// CDC API endpoints
const CDC_ENDPOINTS = {
  // COVID Data Tracker API
  covid: 'https://data.cdc.gov/resource/9mfq-cb36.json',
  // FluView API
  fluview: 'https://gis.cdc.gov/grasp/fluview/FluViewPhase6ServicesApi/GetFluData',
  // NNDSS Weekly Tables
  nndss: 'https://data.cdc.gov/resource/r8kw-7aab.json',
  // Vaccination data
  vaccinations: 'https://data.cdc.gov/resource/rh2h-3yt2.json',
  // NHSN Healthcare data
  nhsn: 'https://data.cdc.gov/resource/g62h-syeh.json',
  // Wastewater surveillance
  wastewater: 'https://data.cdc.gov/resource/2ew6-ywp6.json',
  // Mortality data
  mortality: 'https://data.cdc.gov/resource/r8kw-7aab.json',
} as const;

export interface DiseaseDataPoint {
  date: Date;
  region: string;
  disease: string;
  cases: number;
  deaths?: number;
  hospitalizations?: number;
  positivityRate?: number;
  metadata: Record<string, unknown>;
}

export interface SurveillanceAlert {
  level: 'low' | 'moderate' | 'high' | 'very-high' | 'critical';
  disease: string;
  region: string;
  metric: string;
  currentValue: number;
  threshold: number;
  trend: 'increasing' | 'stable' | 'decreasing';
  weekOverWeekChange: number;
}

export interface HealthSignal {
  timestamp: Date;
  category: 'respiratory' | 'gi' | 'neurological' | 'other';
  severity: number; // 0-1 normalized
  geographic: {
    national: number;
    regional: Map<string, number>;
  };
  momentum: number; // Rate of change
  seasonalAdjusted: number;
}

export interface CDCSourceConfig {
  /** App token for higher rate limits (optional) */
  appToken?: string;
  /** Regions to track */
  regions?: string[];
  /** Diseases to monitor */
  diseases?: string[];
  /** Enable wastewater early warning */
  wastewaterEnabled?: boolean;
}

/**
 * CDC Disease Surveillance Source
 *
 * Provides real-time public health intelligence from CDC surveillance systems.
 * This is PUBLIC DOMAIN data from the US Government.
 *
 * Use cases:
 * - Healthcare sector signal generation
 * - Pandemic early warning
 * - Regional health risk assessment
 * - Seasonal adjustment for economic models
 */
export class CDCSource {
  private rateLimiter: RateLimiter;
  private attribution: AttributionTracker;
  private config: CDCSourceConfig;
  private cache: Map<string, { data: unknown; timestamp: number }> = new Map();
  private cacheLifetimeMs = 3600000; // 1 hour cache

  // CDC HHS regions
  static readonly REGIONS = [
    'Region 1', // CT, ME, MA, NH, RI, VT
    'Region 2', // NJ, NY, PR, VI
    'Region 3', // DE, DC, MD, PA, VA, WV
    'Region 4', // AL, FL, GA, KY, MS, NC, SC, TN
    'Region 5', // IL, IN, MI, MN, OH, WI
    'Region 6', // AR, LA, NM, OK, TX
    'Region 7', // IA, KS, MO, NE
    'Region 8', // CO, MT, ND, SD, UT, WY
    'Region 9', // AZ, CA, HI, NV, AS, GU
    'Region 10', // AK, ID, OR, WA
  ] as const;

  // Monitored diseases
  static readonly DISEASES = [
    'COVID-19',
    'Influenza',
    'RSV',
    'Norovirus',
    'Salmonella',
    'E. coli',
    'Measles',
    'Pertussis',
    'Hepatitis A',
    'Legionellosis',
  ] as const;

  constructor(config: CDCSourceConfig = {}) {
    this.config = {
      regions: [...CDCSource.REGIONS],
      diseases: [...CDCSource.DISEASES],
      wastewaterEnabled: true,
      ...config,
    };

    // CDC allows 1000 requests/hour without token, 10000 with token
    this.rateLimiter = new RateLimiter();
    this.rateLimiter.configure('cdc', {
      requestsPerSecond: config.appToken ? 3 : 0.25, // Conservative
      burstSize: config.appToken ? 10 : 2,
    });

    this.attribution = new AttributionTracker();
    this.attribution.registerSource({
      id: 'cdc',
      name: 'Centers for Disease Control and Prevention',
      type: 'official',
      license: 'Public Domain',
      attribution: 'Source: CDC, https://www.cdc.gov',
      dataUrl: 'https://data.cdc.gov',
      updateFrequency: 'weekly',
    });
  }

  /**
   * Get COVID-19 surveillance data
   */
  async getCovidData(options: {
    state?: string;
    startDate?: Date;
    limit?: number;
  } = {}): Promise<DiseaseDataPoint[]> {
    await this.rateLimiter.waitForSlot('cdc');

    const params = new URLSearchParams();
    if (options.state) params.set('state', options.state);
    if (options.startDate) params.set('$where', `date > '${options.startDate.toISOString().split('T')[0]}'`);
    params.set('$limit', String(options.limit ?? 1000));
    params.set('$order', 'date DESC');

    const url = `${CDC_ENDPOINTS.covid}?${params}`;
    const data = await this.fetchWithCache<Array<{
      date: string;
      state: string;
      new_cases?: string;
      new_deaths?: string;
      [key: string]: unknown;
    }>>(url);

    this.attribution.recordUsage('cdc', data.length);

    return data.map((row) => ({
      date: new Date(row.date),
      region: row.state,
      disease: 'COVID-19',
      cases: parseInt(row.new_cases ?? '0', 10),
      deaths: parseInt(row.new_deaths ?? '0', 10),
      metadata: row,
    }));
  }

  /**
   * Get influenza surveillance data from FluView
   */
  async getFluData(options: {
    season?: string;
    region?: string;
  } = {}): Promise<DiseaseDataPoint[]> {
    await this.rateLimiter.waitForSlot('cdc');

    // FluView uses a different API format
    const params = new URLSearchParams();
    if (options.season) params.set('SeasonsDT', options.season);
    if (options.region) params.set('RegionsDT', options.region);

    // Note: FluView API is more complex - this is simplified
    const url = `${CDC_ENDPOINTS.fluview}?${params}`;

    try {
      const data = await this.fetchWithCache<Array<{
        WEEK: string;
        YEAR: string;
        REGION: string;
        ILITOTAL?: string;
        'NUM OF PROVIDERS'?: string;
        [key: string]: unknown;
      }>>(url);

      this.attribution.recordUsage('cdc', data.length);

      return data.map((row) => ({
        date: this.weekToDate(parseInt(row.YEAR), parseInt(row.WEEK)),
        region: row.REGION,
        disease: 'Influenza',
        cases: parseInt(row.ILITOTAL ?? '0', 10),
        metadata: row,
      }));
    } catch {
      // FluView API can be finicky - return empty on error
      return [];
    }
  }

  /**
   * Get wastewater surveillance data (early warning indicator)
   */
  async getWastewaterData(options: {
    state?: string;
    limit?: number;
  } = {}): Promise<Array<{
    date: Date;
    state: string;
    county?: string;
    pathogen: string;
    concentration: number;
    percentChange: number;
    trend: 'increasing' | 'stable' | 'decreasing';
  }>> {
    if (!this.config.wastewaterEnabled) return [];

    await this.rateLimiter.waitForSlot('cdc');

    const params = new URLSearchParams();
    if (options.state) params.set('state', options.state);
    params.set('$limit', String(options.limit ?? 500));
    params.set('$order', 'date_end DESC');

    const url = `${CDC_ENDPOINTS.wastewater}?${params}`;

    try {
      const data = await this.fetchWithCache<Array<{
        date_end: string;
        state: string;
        county?: string;
        ptc_15d?: string;
        [key: string]: unknown;
      }>>(url);

      this.attribution.recordUsage('cdc', data.length);

      return data.map((row) => {
        const percentChange = parseFloat(row.ptc_15d ?? '0');
        return {
          date: new Date(row.date_end),
          state: row.state,
          county: row.county,
          pathogen: 'SARS-CoV-2',
          concentration: 0, // Would need normalization
          percentChange,
          trend: percentChange > 10 ? 'increasing' : percentChange < -10 ? 'decreasing' : 'stable',
        };
      });
    } catch {
      return [];
    }
  }

  /**
   * Get NHSN hospital capacity data
   */
  async getHospitalData(options: {
    state?: string;
    limit?: number;
  } = {}): Promise<Array<{
    date: Date;
    state: string;
    hospitalizations: number;
    icuOccupancy: number;
    bedUtilization: number;
  }>> {
    await this.rateLimiter.waitForSlot('cdc');

    const params = new URLSearchParams();
    if (options.state) params.set('state', options.state);
    params.set('$limit', String(options.limit ?? 500));
    params.set('$order', 'collection_week DESC');

    const url = `${CDC_ENDPOINTS.nhsn}?${params}`;

    try {
      const data = await this.fetchWithCache<Array<{
        collection_week: string;
        state: string;
        total_patients_hospitalized_confirmed_influenza_and_covid?: string;
        inpatient_bed_covid_utilization?: string;
        [key: string]: unknown;
      }>>(url);

      this.attribution.recordUsage('cdc', data.length);

      return data.map((row) => ({
        date: new Date(row.collection_week),
        state: row.state,
        hospitalizations: parseInt(
          row.total_patients_hospitalized_confirmed_influenza_and_covid ?? '0',
          10
        ),
        icuOccupancy: 0, // Would parse from data
        bedUtilization: parseFloat(row.inpatient_bed_covid_utilization ?? '0'),
      }));
    } catch {
      return [];
    }
  }

  /**
   * Generate composite health signal from multiple CDC data sources
   */
  async generateHealthSignal(): Promise<HealthSignal> {
    // Fetch data from multiple sources in parallel
    const [covidData, wastewaterData, hospitalData] = await Promise.all([
      this.getCovidData({ limit: 100 }),
      this.getWastewaterData({ limit: 100 }),
      this.getHospitalData({ limit: 100 }),
    ]);

    // Calculate national severity
    const recentCases = covidData.slice(0, 7);
    const avgCases = recentCases.reduce((sum, d) => sum + d.cases, 0) / (recentCases.length || 1);

    // Normalize to 0-1 (assuming 100k cases/day is high)
    const caseSeverity = Math.min(1, avgCases / 100000);

    // Wastewater trend (leading indicator)
    const wastewaterIncreasing = wastewaterData.filter((d) => d.trend === 'increasing').length;
    const wastewaterSeverity = wastewaterData.length > 0
      ? wastewaterIncreasing / wastewaterData.length
      : 0;

    // Hospital utilization
    const avgUtilization = hospitalData.reduce((sum, d) => sum + d.bedUtilization, 0) /
      (hospitalData.length || 1);
    const hospitalSeverity = avgUtilization;

    // Composite severity (weighted)
    const severity = caseSeverity * 0.3 + wastewaterSeverity * 0.4 + hospitalSeverity * 0.3;

    // Calculate momentum (week-over-week change)
    const thisWeekCases = covidData.slice(0, 7).reduce((sum, d) => sum + d.cases, 0);
    const lastWeekCases = covidData.slice(7, 14).reduce((sum, d) => sum + d.cases, 0);
    const momentum = lastWeekCases > 0 ? (thisWeekCases - lastWeekCases) / lastWeekCases : 0;

    // Regional breakdown
    const regionalMap = new Map<string, number>();
    for (const region of CDCSource.REGIONS) {
      const regionData = covidData.filter((d) =>
        this.stateToRegion(d.region) === region
      );
      const regionSeverity = regionData.reduce((sum, d) => sum + d.cases, 0) /
        (regionData.length || 1) / 10000;
      regionalMap.set(region, Math.min(1, regionSeverity));
    }

    return {
      timestamp: new Date(),
      category: 'respiratory',
      severity,
      geographic: {
        national: severity,
        regional: regionalMap,
      },
      momentum,
      seasonalAdjusted: this.seasonalAdjust(severity),
    };
  }

  /**
   * Generate alerts based on thresholds
   */
  async checkAlerts(thresholds?: {
    casesHigh?: number;
    hospitalizationHigh?: number;
    wastewaterIncreaseHigh?: number;
  }): Promise<SurveillanceAlert[]> {
    const alerts: SurveillanceAlert[] = [];
    const defaults = {
      casesHigh: 50000,
      hospitalizationHigh: 0.15,
      wastewaterIncreaseHigh: 50,
      ...thresholds,
    };

    const healthSignal = await this.generateHealthSignal();

    // National case alert
    if (healthSignal.severity > 0.7) {
      alerts.push({
        level: healthSignal.severity > 0.85 ? 'critical' : 'very-high',
        disease: 'COVID-19',
        region: 'National',
        metric: 'composite-severity',
        currentValue: healthSignal.severity,
        threshold: 0.7,
        trend: healthSignal.momentum > 0.1 ? 'increasing' : healthSignal.momentum < -0.1 ? 'decreasing' : 'stable',
        weekOverWeekChange: healthSignal.momentum * 100,
      });
    }

    // Regional alerts
    for (const [region, severity] of healthSignal.geographic.regional) {
      if (severity > 0.6) {
        alerts.push({
          level: severity > 0.8 ? 'very-high' : 'high',
          disease: 'COVID-19',
          region,
          metric: 'regional-severity',
          currentValue: severity,
          threshold: 0.6,
          trend: 'stable', // Would need historical data
          weekOverWeekChange: 0,
        });
      }
    }

    return alerts;
  }

  /**
   * Convert disease data to normalized signal array for fusion
   */
  toSignal(data: DiseaseDataPoint[], normalize = true): number[] {
    const sorted = [...data].sort((a, b) => a.date.getTime() - b.date.getTime());
    const values = sorted.map((d) => d.cases);

    if (!normalize || values.length === 0) return values;

    // Normalize to 0-1
    const max = Math.max(...values);
    const min = Math.min(...values);
    const range = max - min || 1;

    return values.map((v) => (v - min) / range);
  }

  /**
   * Get required attribution text
   */
  getAttribution(): string {
    return this.attribution.getAttributionText(['cdc']);
  }

  // Helper methods
  private async fetchWithCache<T>(url: string): Promise<T> {
    const cached = this.cache.get(url);
    if (cached && Date.now() - cached.timestamp < this.cacheLifetimeMs) {
      return cached.data as T;
    }

    const headers: Record<string, string> = {
      'Accept': 'application/json',
    };

    if (this.config.appToken) {
      headers['X-App-Token'] = this.config.appToken;
    }

    const response = await fetch(url, { headers });

    if (!response.ok) {
      throw new Error(`CDC API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    this.cache.set(url, { data, timestamp: Date.now() });

    return data as T;
  }

  private weekToDate(year: number, week: number): Date {
    const jan1 = new Date(year, 0, 1);
    const days = (week - 1) * 7;
    return new Date(jan1.getTime() + days * 24 * 60 * 60 * 1000);
  }

  private stateToRegion(state: string): string {
    // Simplified mapping - would need full lookup table
    const regionMap: Record<string, string> = {
      'CA': 'Region 9',
      'TX': 'Region 6',
      'FL': 'Region 4',
      'NY': 'Region 2',
      // ... would include all states
    };
    return regionMap[state] ?? 'Region 1';
  }

  private seasonalAdjust(value: number): number {
    // Simple seasonal adjustment based on time of year
    // Respiratory diseases peak in winter
    const month = new Date().getMonth();
    const seasonalFactor = [1.3, 1.2, 1.0, 0.9, 0.8, 0.7, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2][month];
    return value / seasonalFactor;
  }
}

// Export types
export type { CDCSourceConfig };
