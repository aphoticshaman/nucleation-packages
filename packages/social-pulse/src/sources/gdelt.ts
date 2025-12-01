/**
 * GDELT (Global Database of Events, Language, and Tone)
 *
 * Free, open database tracking worldwide news and events.
 * Updates every 15 minutes with global media coverage.
 * Excellent for geopolitical sentiment and unrest detection.
 */

import type { DataSource, SearchParams, SocialPost, GeoInfo } from '../types.js';

const GDELT_DOC_API = 'https://api.gdeltproject.org/api/v2/doc/doc';
const GDELT_GEO_API = 'https://api.gdeltproject.org/api/v2/geo/geo';

interface GdeltArticle {
  url: string;
  title: string;
  seendate: string;
  socialimage?: string;
  domain: string;
  language: string;
  sourcecountry: string;
  tone: number;
  themes?: string[];
  locations?: Array<{
    type: string;
    fullname: string;
    countrycode: string;
    lat?: number;
    long?: number;
  }>;
}

interface GdeltDocResponse {
  articles?: GdeltArticle[];
}

export class GdeltSource implements DataSource {
  readonly platform = 'gdelt' as const;
  private ready = false;

  async init(): Promise<void> {
    // GDELT is always available, no auth needed
    this.ready = true;
  }

  isReady(): boolean {
    return this.ready;
  }

  async fetch(params: SearchParams): Promise<SocialPost[]> {
    if (!this.ready) {
      throw new Error('GDELT source not initialized');
    }

    const query = this.buildQuery(params);
    const url = new URL(GDELT_DOC_API);
    url.searchParams.set('query', query);
    url.searchParams.set('mode', 'artlist');
    url.searchParams.set('maxrecords', String(params.limit ?? 250));
    url.searchParams.set('format', 'json');
    url.searchParams.set('sort', 'datedesc');

    // Time range
    if (params.since) {
      url.searchParams.set('startdatetime', this.formatGdeltDate(params.since));
    }
    if (params.until) {
      url.searchParams.set('enddatetime', this.formatGdeltDate(params.until));
    }

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`GDELT query failed: ${response.status}`);
    }

    const data = (await response.json()) as GdeltDocResponse;
    return (data.articles ?? []).map((article) => this.transformArticle(article));
  }

  /**
   * Get geographic heatmap of activity for keywords
   */
  async getGeoHeatmap(keywords: string[], timespan: string = '24h'): Promise<Map<string, number>> {
    const url = new URL(GDELT_GEO_API);
    url.searchParams.set('query', keywords.join(' OR '));
    url.searchParams.set('mode', 'pointdata');
    url.searchParams.set('format', 'json');
    url.searchParams.set('timespan', timespan);

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`GDELT geo query failed: ${response.status}`);
    }

    const data = (await response.json()) as {
      features?: Array<{
        properties?: { countrycode?: string; count?: number };
      }>;
    };
    const countryActivity = new Map<string, number>();

    for (const feature of data.features ?? []) {
      const country = feature.properties?.countrycode;
      if (country) {
        countryActivity.set(
          country,
          (countryActivity.get(country) ?? 0) + (feature.properties?.count ?? 1)
        );
      }
    }

    return countryActivity;
  }

  /**
   * Get tone/sentiment trend for keywords over time
   */
  async getToneTrend(
    keywords: string[],
    timespan: string = '7d'
  ): Promise<Array<{ date: string; tone: number; volume: number }>> {
    const url = new URL(GDELT_DOC_API);
    url.searchParams.set('query', keywords.join(' OR '));
    url.searchParams.set('mode', 'timelinetone');
    url.searchParams.set('format', 'json');
    url.searchParams.set('timespan', timespan);

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`GDELT tone query failed: ${response.status}`);
    }

    const data = (await response.json()) as {
      timeline?: Array<{ date: string; value: number; count: number }>;
    };
    return (data.timeline ?? []).map((point) => ({
      date: point.date,
      tone: point.value,
      volume: point.count,
    }));
  }

  private buildQuery(params: SearchParams): string {
    const parts: string[] = [];

    if (params.keywords?.length) {
      parts.push(`(${params.keywords.join(' OR ')})`);
    }

    if (params.countries?.length) {
      parts.push(`sourcecountry:${params.countries.join(' OR sourcecountry:')}`);
    }

    if (params.languages?.length) {
      parts.push(`sourcelang:${params.languages.join(' OR sourcelang:')}`);
    }

    // Default to news sources if no specific query
    return parts.length > 0 ? parts.join(' ') : 'protest OR unrest OR revolution';
  }

  private formatGdeltDate(date: Date): string {
    return date.toISOString().replace(/[-:T]/g, '').slice(0, 14);
  }

  private transformArticle(article: GdeltArticle): SocialPost {
    // Extract primary location
    let geo: GeoInfo | undefined;
    const primaryLocation = article.locations?.[0];
    if (primaryLocation) {
      geo = {
        countryCode: primaryLocation.countrycode,
        country: primaryLocation.fullname,
        source: 'inferred',
        confidence: 0.7,
      };
      // Only add lat/lon if they exist
      if (primaryLocation.lat !== undefined) geo.lat = primaryLocation.lat;
      if (primaryLocation.long !== undefined) geo.lon = primaryLocation.long;
    } else if (article.sourcecountry) {
      geo = {
        countryCode: article.sourcecountry.slice(0, 2).toUpperCase(),
        source: 'inferred',
        confidence: 0.5,
      };
    }

    // Convert GDELT tone (-100 to +100) to our scale (-1 to +1)
    const sentimentScore = article.tone / 100;

    const socialPost: SocialPost = {
      id: article.url,
      platform: 'gdelt',
      content: article.title,
      timestamp: this.parseGdeltDate(article.seendate),
      author: {
        id: article.domain,
        name: article.domain,
      },
      engagement: {},
      language: article.language,
      sentimentScore,
      raw: article,
    };

    // Only add geo if it exists
    if (geo) {
      socialPost.geo = geo;
    }

    return socialPost;
  }

  private parseGdeltDate(dateStr: string): string {
    // GDELT format: YYYYMMDDHHMMSS
    const year = dateStr.slice(0, 4);
    const month = dateStr.slice(4, 6);
    const day = dateStr.slice(6, 8);
    const hour = dateStr.slice(8, 10);
    const min = dateStr.slice(10, 12);
    const sec = dateStr.slice(12, 14);
    return `${year}-${month}-${day}T${hour}:${min}:${sec}Z`;
  }
}
