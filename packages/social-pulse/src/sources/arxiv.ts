/**
 * arXiv preprint data source
 *
 * Free API for scientific preprints.
 * Excellent for detecting breakthrough research before mainstream awareness.
 */

import type { DataSource, SearchParams, SocialPost } from '../types.js';

const ARXIV_API = 'https://export.arxiv.org/api/query';

interface ArxivEntry {
  id: string;
  title: string;
  summary: string;
  published: string;
  updated: string;
  authors: Array<{ name: string; affiliation?: string }>;
  categories: string[];
  links: Array<{ href: string; type?: string }>;
}

export class ArxivSource implements DataSource {
  readonly platform = 'custom' as const;
  private ready = false;

  async init(): Promise<void> {
    this.ready = true;
  }

  isReady(): boolean {
    return this.ready;
  }

  async fetch(params: SearchParams): Promise<SocialPost[]> {
    if (!this.ready) {
      throw new Error('arXiv source not initialized');
    }

    const query = this.buildQuery(params);
    const url = new URL(ARXIV_API);
    url.searchParams.set('search_query', query);
    url.searchParams.set('start', '0');
    url.searchParams.set('max_results', String(params.limit ?? 100));
    url.searchParams.set('sortBy', 'submittedDate');
    url.searchParams.set('sortOrder', 'descending');

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`arXiv query failed: ${response.status}`);
    }

    const xml = await response.text();
    const entries = this.parseAtomFeed(xml);
    return entries.map((entry) => this.transformEntry(entry));
  }

  /**
   * Get trending categories by submission volume
   */
  async getTrendingCategories(days = 7): Promise<Map<string, number>> {
    const categories = new Map<string, number>();
    const since = new Date();
    since.setDate(since.getDate() - days);

    // Sample major categories
    const majorCategories = [
      'cs.AI',
      'cs.LG',
      'cs.CL',
      'cs.CV', // AI/ML
      'physics',
      'math',
      'q-bio',
      'q-fin',
      'cond-mat',
      'hep-th',
      'quant-ph',
    ];

    for (const cat of majorCategories) {
      try {
        const url = new URL(ARXIV_API);
        url.searchParams.set('search_query', `cat:${cat}`);
        url.searchParams.set('max_results', '1');

        const response = await fetch(url.toString());
        const xml = await response.text();

        // Extract total results from opensearch:totalResults
        const match = /<opensearch:totalResults[^>]*>(\d+)</.exec(xml);
        if (match && match[1]) {
          categories.set(cat, parseInt(match[1], 10));
        }
      } catch {
        // Skip failed categories
      }
    }

    return categories;
  }

  /**
   * Find undercited papers that might be quiet breakthroughs
   * (papers with high update frequency but low external citations)
   */
  async findQuietBreakthroughs(category: string, days = 30): Promise<SocialPost[]> {
    const since = new Date();
    since.setDate(since.getDate() - days);

    const posts = await this.fetch({
      keywords: [`cat:${category}`],
      since,
      limit: 200,
    });

    // Sort by update frequency (papers being revised often = active research)
    return posts.filter((post) => {
      const entry = post.raw as ArxivEntry;
      const published = new Date(entry.published);
      const updated = new Date(entry.updated);
      const daysSincePublished = (Date.now() - published.getTime()) / (1000 * 60 * 60 * 24);
      const daysSinceUpdated = (Date.now() - updated.getTime()) / (1000 * 60 * 60 * 24);

      // Papers updated recently but published a while ago
      return daysSinceUpdated < 7 && daysSincePublished > 14;
    });
  }

  private buildQuery(params: SearchParams): string {
    const parts: string[] = [];

    if (params.keywords?.length) {
      // Support category syntax like "cat:cs.AI"
      const catKeywords = params.keywords.filter((k) => k.startsWith('cat:'));
      const textKeywords = params.keywords.filter((k) => !k.startsWith('cat:'));

      if (catKeywords.length) {
        parts.push(catKeywords.join(' OR '));
      }
      if (textKeywords.length) {
        parts.push(`all:(${textKeywords.join(' OR ')})`);
      }
    }

    return parts.join(' AND ') || 'all:*';
  }

  private parseAtomFeed(xml: string): ArxivEntry[] {
    const entries: ArxivEntry[] = [];
    const entryRegex = /<entry>([\s\S]*?)<\/entry>/g;
    let match;

    while ((match = entryRegex.exec(xml)) !== null) {
      if (!match[1]) continue;
      const entryXml = match[1];

      const getId = (tag: string): string => {
        const m = new RegExp(`<${tag}[^>]*>([\\s\\S]*?)</${tag}>`).exec(entryXml);
        return m && m[1] ? m[1].trim() : '';
      };

      const id = getId('id');
      const title = getId('title').replace(/\s+/g, ' ');
      const summary = getId('summary').replace(/\s+/g, ' ');
      const published = getId('published');
      const updated = getId('updated');

      // Parse authors
      const authors: Array<{ name: string; affiliation?: string }> = [];
      const authorRegex = /<author>[\s\S]*?<name>([^<]+)<\/name>[\s\S]*?<\/author>/g;
      let authorMatch;
      while ((authorMatch = authorRegex.exec(entryXml)) !== null) {
        const authorName = authorMatch[1];
        if (authorName) {
          authors.push({ name: authorName.trim() });
        }
      }

      // Parse categories
      const categories: string[] = [];
      const catRegex = /<category[^>]*term="([^"]+)"/g;
      let catMatch;
      while ((catMatch = catRegex.exec(entryXml)) !== null) {
        const catTerm = catMatch[1];
        if (catTerm) {
          categories.push(catTerm);
        }
      }

      // Parse links
      const links: { href: string; type?: string }[] = [];
      const linkRegex = /<link\s+[^>]*href="([^"]+)"[^>]*>/g;
      let linkMatch;
      while ((linkMatch = linkRegex.exec(entryXml)) !== null) {
        const href = linkMatch[1];
        if (href) {
          links.push({ href });
        }
      }

      entries.push({ id, title, summary, published, updated, authors, categories, links });
    }

    return entries;
  }

  private transformEntry(entry: ArxivEntry): SocialPost {
    // Combine title and abstract
    const content = `${entry.title}\n\n${entry.summary}`;

    // Primary author
    const primaryAuthor = entry.authors[0];

    return {
      id: entry.id,
      platform: 'custom',
      content,
      timestamp: entry.published,
      author: {
        id: primaryAuthor?.name ?? 'unknown',
        name: primaryAuthor?.name ?? 'unknown',
        bio: entry.authors.map((a) => a.name).join(', '),
      },
      engagement: {
        // arXiv doesn't have engagement metrics
      },
      language: 'en', // arXiv is predominantly English
      raw: entry,
    };
  }
}
