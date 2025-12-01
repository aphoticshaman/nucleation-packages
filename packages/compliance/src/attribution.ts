/**
 * Attribution
 *
 * Tracks and generates required attributions for data sources.
 * Many APIs require attribution in specific formats.
 */

export interface SourceAttribution {
  sourceId: string;
  name: string;
  url?: string;
  license: string;
  requiredText?: string;
  logoUrl?: string;
  linkRequired: boolean;
}

export interface AttributionOutput {
  text: string;
  html: string;
  markdown: string;
  json: SourceAttribution[];
}

export class AttributionTracker {
  private attributions: Map<string, SourceAttribution> = new Map();

  /**
   * Register a source attribution requirement
   */
  register(attribution: SourceAttribution): void {
    this.attributions.set(attribution.sourceId, attribution);
  }

  /**
   * Register multiple attributions
   */
  registerBulk(attributions: SourceAttribution[]): void {
    for (const attr of attributions) {
      this.register(attr);
    }
  }

  /**
   * Get attribution for a source
   */
  get(sourceId: string): SourceAttribution | undefined {
    return this.attributions.get(sourceId);
  }

  /**
   * Check if source requires attribution
   */
  requiresAttribution(sourceId: string): boolean {
    const attr = this.attributions.get(sourceId);
    return attr !== undefined && (attr.requiredText !== undefined || attr.linkRequired);
  }

  /**
   * Generate attribution output for used sources
   */
  generate(usedSourceIds: string[]): AttributionOutput {
    const sources = usedSourceIds
      .map((id) => this.attributions.get(id))
      .filter((a): a is SourceAttribution => a !== undefined);

    if (sources.length === 0) {
      return {
        text: '',
        html: '',
        markdown: '',
        json: [],
      };
    }

    // Text format
    const textParts = sources.map((s) => {
      if (s.requiredText) return s.requiredText;
      return `Data from ${s.name}${s.url ? ` (${s.url})` : ''} - ${s.license}`;
    });
    const text = `Data sources: ${textParts.join('; ')}`;

    // HTML format
    const htmlParts = sources.map((s) => {
      const link = s.url ? `<a href="${s.url}" target="_blank">${s.name}</a>` : s.name;
      if (s.requiredText) return s.requiredText.replace(s.name, link);
      return `${link} (${s.license})`;
    });
    const html = `<p class="attribution">Data sources: ${htmlParts.join(', ')}</p>`;

    // Markdown format
    const mdParts = sources.map((s) => {
      const link = s.url ? `[${s.name}](${s.url})` : s.name;
      if (s.requiredText) return s.requiredText.replace(s.name, link);
      return `${link} (${s.license})`;
    });
    const markdown = `*Data sources: ${mdParts.join(', ')}*`;

    return {
      text,
      html,
      markdown,
      json: sources,
    };
  }

  /**
   * Get all registered attributions
   */
  getAll(): SourceAttribution[] {
    return [...this.attributions.values()];
  }

  /**
   * Clear all attributions
   */
  clear(): void {
    this.attributions.clear();
  }
}

/**
 * Pre-configured attributions for common sources
 */
export const COMMON_ATTRIBUTIONS: SourceAttribution[] = [
  {
    sourceId: 'sec-edgar',
    name: 'SEC EDGAR',
    url: 'https://www.sec.gov/edgar',
    license: 'Public Domain',
    linkRequired: false,
  },
  {
    sourceId: 'fred',
    name: 'Federal Reserve Economic Data',
    url: 'https://fred.stlouisfed.org',
    license: 'Public Domain',
    requiredText: 'Source: FRED, Federal Reserve Bank of St. Louis',
    linkRequired: true,
  },
  {
    sourceId: 'guardian',
    name: 'The Guardian',
    url: 'https://www.theguardian.com',
    license: 'CC BY-SA 3.0',
    requiredText: 'Powered by Guardian Open Platform',
    linkRequired: true,
  },
  {
    sourceId: 'newsapi',
    name: 'NewsAPI',
    url: 'https://newsapi.org',
    license: 'NewsAPI Terms',
    requiredText: 'Powered by NewsAPI.org',
    linkRequired: true,
  },
  {
    sourceId: 'gdelt',
    name: 'GDELT Project',
    url: 'https://www.gdeltproject.org',
    license: 'Creative Commons',
    requiredText: 'Data provided by the GDELT Project',
    linkRequired: true,
  },
  {
    sourceId: 'alpha-vantage',
    name: 'Alpha Vantage',
    url: 'https://www.alphavantage.co',
    license: 'Alpha Vantage Terms',
    requiredText: 'Data provided by Alpha Vantage',
    linkRequired: true,
  },
  {
    sourceId: 'reddit',
    name: 'Reddit',
    url: 'https://www.reddit.com',
    license: 'Reddit API Terms',
    linkRequired: false,
  },
  {
    sourceId: 'bluesky',
    name: 'Bluesky',
    url: 'https://bsky.app',
    license: 'AT Protocol',
    linkRequired: false,
  },
];
