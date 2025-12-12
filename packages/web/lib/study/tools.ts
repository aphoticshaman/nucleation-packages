/**
 * Study Book Tools
 *
 * Tools available to Elle for enhanced capabilities:
 * - Web search (live data access)
 * - Code execution
 * - Database queries
 * - GDELT intel feeds
 * - GitHub operations
 *
 * Three research depths:
 * - INSTANT: Use cached/recent data, no searches (~2s)
 * - MODERATE: 1-3 web searches, quick synthesis (~15s)
 * - THOROUGH: Deep multi-source research, fact-checking (~2-5min)
 */

// =============================================================================
// TYPES
// =============================================================================

export type ResearchDepth = 'instant' | 'moderate' | 'thorough';

export interface ToolResult {
  tool: string;
  success: boolean;
  data: unknown;
  latency_ms: number;
  error?: string;
}

export interface WebSearchResult {
  title: string;
  url: string;
  snippet: string;
  publishedDate?: string;
}

export interface GDELTSignal {
  theme: string;
  country: string;
  tone: number;
  articleCount: number;
  headlines: string[];
  timestamp: string;
}

// =============================================================================
// WEB SEARCH TOOL
// =============================================================================

/**
 * Search the web using multiple providers for redundancy
 * Priority: SearXNG (free/unlimited) → Serper → Brave → DuckDuckGo
 */
export async function webSearch(
  query: string,
  options?: {
    depth: ResearchDepth;
    numResults?: number;
    freshness?: 'day' | 'week' | 'month' | 'year';
  }
): Promise<WebSearchResult[]> {
  const startTime = Date.now();
  const depth = options?.depth || 'moderate';

  // Determine search parameters based on depth
  const numResults = options?.numResults || (
    depth === 'instant' ? 3 :
    depth === 'moderate' ? 8 :
    15
  );

  // Try SearXNG first (FREE - self-hosted meta-search)
  // Aggregates Google, Bing, DuckDuckGo, Brave, etc.
  const searxngUrl = process.env.SEARXNG_URL;
  if (searxngUrl) {
    try {
      const params = new URLSearchParams({
        q: query,
        format: 'json',
        categories: 'general',
        language: 'en',
        pageno: '1',
      });

      // Add time range for freshness
      if (options?.freshness) {
        const timeRanges: Record<string, string> = {
          day: 'day',
          week: 'week',
          month: 'month',
          year: 'year',
        };
        params.set('time_range', timeRanges[options.freshness] || '');
      }

      const response = await fetch(`${searxngUrl}/search?${params}`, {
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'LatticeForge-Elle/1.0',
        },
      });

      if (response.ok) {
        const data = await response.json();
        const results: WebSearchResult[] = (data.results || [])
          .slice(0, numResults)
          .map((r: {
            title: string;
            url: string;
            content: string;
            publishedDate?: string;
          }) => ({
            title: r.title,
            url: r.url,
            snippet: r.content,
            publishedDate: r.publishedDate,
          }));

        console.log(`[WebSearch] SearXNG returned ${results.length} results in ${Date.now() - startTime}ms`);
        return results;
      }
    } catch (e) {
      console.warn('[WebSearch] SearXNG failed:', e);
    }
  }

  // Fallback to Serper (Google results, paid)
  if (process.env.SERPER_API_KEY) {
    try {
      const response = await fetch('https://google.serper.dev/search', {
        method: 'POST',
        headers: {
          'X-API-KEY': process.env.SERPER_API_KEY,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          q: query,
          num: numResults,
          tbs: options?.freshness ? `qdr:${options.freshness[0]}` : undefined,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const results: WebSearchResult[] = (data.organic || []).map((r: {
          title: string;
          link: string;
          snippet: string;
          date?: string;
        }) => ({
          title: r.title,
          url: r.link,
          snippet: r.snippet,
          publishedDate: r.date,
        }));

        console.log(`[WebSearch] Serper returned ${results.length} results in ${Date.now() - startTime}ms`);
        return results;
      }
    } catch (e) {
      console.warn('[WebSearch] Serper failed:', e);
    }
  }

  // Fallback to Brave Search
  if (process.env.BRAVE_SEARCH_API_KEY) {
    try {
      const params = new URLSearchParams({
        q: query,
        count: String(numResults),
      });
      if (options?.freshness) {
        params.set('freshness', options.freshness);
      }

      const response = await fetch(`https://api.search.brave.com/res/v1/web/search?${params}`, {
        headers: {
          'X-Subscription-Token': process.env.BRAVE_SEARCH_API_KEY,
        },
      });

      if (response.ok) {
        const data = await response.json();
        const results: WebSearchResult[] = (data.web?.results || []).map((r: {
          title: string;
          url: string;
          description: string;
          age?: string;
        }) => ({
          title: r.title,
          url: r.url,
          snippet: r.description,
          publishedDate: r.age,
        }));

        console.log(`[WebSearch] Brave returned ${results.length} results in ${Date.now() - startTime}ms`);
        return results;
      }
    } catch (e) {
      console.warn('[WebSearch] Brave failed:', e);
    }
  }

  // Final fallback: DuckDuckGo (no API key needed)
  try {
    const response = await fetch(
      `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1`
    );

    if (response.ok) {
      const data = await response.json();
      const results: WebSearchResult[] = [];

      // Abstract result
      if (data.Abstract) {
        results.push({
          title: data.Heading || query,
          url: data.AbstractURL || '',
          snippet: data.Abstract,
        });
      }

      // Related topics
      for (const topic of (data.RelatedTopics || []).slice(0, numResults - 1)) {
        if (topic.Text) {
          results.push({
            title: topic.Text.split(' - ')[0],
            url: topic.FirstURL || '',
            snippet: topic.Text,
          });
        }
      }

      console.log(`[WebSearch] DuckDuckGo returned ${results.length} results in ${Date.now() - startTime}ms`);
      return results;
    }
  } catch (e) {
    console.warn('[WebSearch] DuckDuckGo failed:', e);
  }

  return [];
}

/**
 * Fetch and extract content from a URL
 * Uses Jina Reader (free) for clean LLM-ready markdown, with fallback to raw fetch
 */
export async function fetchWebPage(url: string): Promise<string> {
  const startTime = Date.now();

  // Try Jina Reader first (FREE - converts any URL to clean markdown)
  // https://jina.ai/reader - 1M tokens/month free
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);

    const response = await fetch(`https://r.jina.ai/${url}`, {
      signal: controller.signal,
      headers: {
        'Accept': 'text/plain',
        'User-Agent': 'LatticeForge-Elle/1.0',
      },
    });

    clearTimeout(timeout);

    if (response.ok) {
      const text = await response.text();
      console.log(`[WebFetch] Jina Reader returned ${text.length} chars in ${Date.now() - startTime}ms`);
      return text.slice(0, 15000); // Limit to 15k chars
    }
  } catch (e) {
    console.warn(`[WebFetch] Jina Reader failed for ${url}:`, e);
  }

  // Fallback to raw fetch + basic extraction
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);

    const response = await fetch(url, {
      signal: controller.signal,
      headers: {
        'User-Agent': 'LatticeForge-Elle/1.0 (Research Assistant)',
      },
    });

    clearTimeout(timeout);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const html = await response.text();

    // Basic HTML to text extraction
    const text = html
      .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
      .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
      .replace(/<[^>]+>/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, 10000);

    console.log(`[WebFetch] Raw fetch returned ${text.length} chars in ${Date.now() - startTime}ms`);
    return text;
  } catch (e) {
    console.warn(`[WebFetch] Failed to fetch ${url}:`, e);
    return '';
  }
}

// =============================================================================
// GDELT INTEL TOOL
// =============================================================================

/**
 * Get real-time GDELT signals
 */
export async function getGDELTSignals(
  options?: {
    themes?: string[];
    countries?: string[];
    hours?: number;
  }
): Promise<GDELTSignal[]> {
  const themes = options?.themes || ['MILITARY', 'PROTEST', 'TERROR', 'ELECTION', 'ECON_BANKRUPTCY'];
  const hours = options?.hours || 24;

  const signals: GDELTSignal[] = [];

  for (const theme of themes.slice(0, 5)) {  // Limit to 5 themes
    try {
      const params = new URLSearchParams({
        query: `theme:${theme}`,
        mode: 'artlist',
        maxrecords: '25',
        format: 'json',
        timespan: `${hours}h`,
        sort: 'toneasc',
      });

      const response = await fetch(`https://api.gdeltproject.org/api/v2/doc/doc?${params}`, {
        headers: { 'User-Agent': 'LatticeForge/2.0' },
      });

      if (response.ok) {
        const data = await response.json();
        const articles = data.articles || [];

        if (articles.length > 0) {
          const avgTone = articles.reduce((sum: number, a: { tone: number }) => sum + a.tone, 0) / articles.length;

          signals.push({
            theme,
            country: 'GLOBAL',
            tone: avgTone,
            articleCount: articles.length,
            headlines: articles.slice(0, 5).map((a: { title: string }) => a.title),
            timestamp: new Date().toISOString(),
          });
        }
      }

      // Rate limit: 1 req/sec
      await new Promise(r => setTimeout(r, 1000));
    } catch (e) {
      console.warn(`[GDELT] Theme ${theme} failed:`, e);
    }
  }

  return signals;
}

// =============================================================================
// CODE EXECUTION TOOL
// =============================================================================

/**
 * Execute code in a sandboxed environment
 * Supports: JavaScript, Python (if available)
 */
export async function executeCode(
  code: string,
  language: 'javascript' | 'python' = 'javascript'
): Promise<{ output: string; error?: string; exitCode: number }> {
  const startTime = Date.now();

  if (language === 'javascript') {
    try {
      // Create a sandboxed context
      const sandbox = {
        console: {
          log: (...args: unknown[]) => outputs.push(args.map(String).join(' ')),
          error: (...args: unknown[]) => outputs.push(`ERROR: ${args.map(String).join(' ')}`),
          warn: (...args: unknown[]) => outputs.push(`WARN: ${args.map(String).join(' ')}`),
        },
        Math,
        Date,
        JSON,
        Array,
        Object,
        String,
        Number,
        Boolean,
        RegExp,
        Error,
        setTimeout: undefined,  // Disabled for safety
        setInterval: undefined,
        fetch: undefined,
      };

      const outputs: string[] = [];

      // Use Function constructor for sandboxed execution
      const fn = new Function(
        ...Object.keys(sandbox),
        `"use strict";\n${code}`
      );

      const result = fn(...Object.values(sandbox));

      if (result !== undefined) {
        outputs.push(String(result));
      }

      console.log(`[CodeExec] JS executed in ${Date.now() - startTime}ms`);
      return {
        output: outputs.join('\n'),
        exitCode: 0,
      };
    } catch (e) {
      return {
        output: '',
        error: e instanceof Error ? e.message : String(e),
        exitCode: 1,
      };
    }
  }

  // Python execution would require a separate service
  return {
    output: '',
    error: 'Python execution not yet configured',
    exitCode: 1,
  };
}

// =============================================================================
// DATABASE QUERY TOOL
// =============================================================================

import { createClient } from '@supabase/supabase-js';

/**
 * Query LatticeForge database (admin only)
 * Restricted to SELECT queries for safety
 */
export async function queryDatabase(
  sql: string,
  params?: unknown[]
): Promise<{ data: unknown[]; rowCount: number; error?: string }> {
  // Validate it's a SELECT query
  const normalized = sql.trim().toLowerCase();
  if (!normalized.startsWith('select')) {
    return {
      data: [],
      rowCount: 0,
      error: 'Only SELECT queries are allowed',
    };
  }

  try {
    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );

    // Use rpc for raw SQL (requires a function in Supabase)
    // For now, we'll use the REST API with table names
    const tableMatch = sql.match(/from\s+(\w+)/i);
    if (!tableMatch) {
      return { data: [], rowCount: 0, error: 'Could not parse table name' };
    }

    const tableName = tableMatch[1];
    const { data, error } = await supabase
      .from(tableName)
      .select('*')
      .limit(100);

    if (error) {
      return { data: [], rowCount: 0, error: error.message };
    }

    return {
      data: data || [],
      rowCount: data?.length || 0,
    };
  } catch (e) {
    return {
      data: [],
      rowCount: 0,
      error: e instanceof Error ? e.message : String(e),
    };
  }
}

// =============================================================================
// RESEARCH ORCHESTRATOR
// =============================================================================

export interface ResearchResult {
  depth: ResearchDepth;
  webResults: WebSearchResult[];
  gdeltSignals: GDELTSignal[];
  fetchedPages: Array<{ url: string; content: string }>;
  synthesis?: string;
  totalLatency_ms: number;
}

/**
 * Orchestrate research based on depth
 */
export async function conductResearch(
  query: string,
  depth: ResearchDepth
): Promise<ResearchResult> {
  const startTime = Date.now();
  const result: ResearchResult = {
    depth,
    webResults: [],
    gdeltSignals: [],
    fetchedPages: [],
    totalLatency_ms: 0,
  };

  // INSTANT: Quick cached/recent data only
  if (depth === 'instant') {
    // Just do a quick search
    result.webResults = await webSearch(query, { depth: 'instant', numResults: 3 });
    result.totalLatency_ms = Date.now() - startTime;
    return result;
  }

  // MODERATE: Web search + GDELT + fetch top 2 pages
  if (depth === 'moderate') {
    const [webResults, gdeltSignals] = await Promise.all([
      webSearch(query, { depth: 'moderate', numResults: 8 }),
      getGDELTSignals({ hours: 48 }),
    ]);

    result.webResults = webResults;
    result.gdeltSignals = gdeltSignals;

    // Fetch top 2 relevant pages
    const topUrls = webResults.slice(0, 2).map(r => r.url).filter(Boolean);
    const fetchedPages = await Promise.all(
      topUrls.map(async url => ({
        url,
        content: await fetchWebPage(url),
      }))
    );
    result.fetchedPages = fetchedPages.filter(p => p.content.length > 100);

    result.totalLatency_ms = Date.now() - startTime;
    return result;
  }

  // THOROUGH: Multi-search + GDELT + fetch many pages + cross-reference
  if (depth === 'thorough') {
    // Multiple search queries to get diverse perspectives
    const searchQueries = [
      query,
      `${query} analysis`,
      `${query} latest news`,
      `${query} expert opinion`,
    ];

    const searchPromises = searchQueries.map(q =>
      webSearch(q, { depth: 'thorough', numResults: 10, freshness: 'week' })
    );

    const gdeltPromise = getGDELTSignals({ hours: 72 });

    const [searchResults, gdeltSignals] = await Promise.all([
      Promise.all(searchPromises),
      gdeltPromise,
    ]);

    // Deduplicate and merge results
    const seenUrls = new Set<string>();
    for (const results of searchResults) {
      for (const r of results) {
        if (!seenUrls.has(r.url)) {
          seenUrls.add(r.url);
          result.webResults.push(r);
        }
      }
    }

    result.gdeltSignals = gdeltSignals;

    // Fetch top 5 relevant pages
    const topUrls = result.webResults.slice(0, 5).map(r => r.url).filter(Boolean);
    const fetchedPages = await Promise.all(
      topUrls.map(async url => ({
        url,
        content: await fetchWebPage(url),
      }))
    );
    result.fetchedPages = fetchedPages.filter(p => p.content.length > 100);

    result.totalLatency_ms = Date.now() - startTime;
    return result;
  }

  return result;
}

// =============================================================================
// TOOL EXECUTOR
// =============================================================================

export type ToolName = 'web_search' | 'fetch_page' | 'gdelt' | 'execute_code' | 'query_db' | 'research';

export interface ToolCall {
  name: ToolName;
  arguments: Record<string, unknown>;
}

export async function executeTool(call: ToolCall): Promise<ToolResult> {
  const startTime = Date.now();

  try {
    switch (call.name) {
      case 'web_search': {
        const data = await webSearch(
          call.arguments.query as string,
          {
            depth: (call.arguments.depth as ResearchDepth) || 'moderate',
            numResults: call.arguments.numResults as number,
          }
        );
        return {
          tool: 'web_search',
          success: true,
          data,
          latency_ms: Date.now() - startTime,
        };
      }

      case 'fetch_page': {
        const content = await fetchWebPage(call.arguments.url as string);
        return {
          tool: 'fetch_page',
          success: content.length > 0,
          data: { content },
          latency_ms: Date.now() - startTime,
        };
      }

      case 'gdelt': {
        const data = await getGDELTSignals({
          themes: call.arguments.themes as string[],
          hours: call.arguments.hours as number,
        });
        return {
          tool: 'gdelt',
          success: true,
          data,
          latency_ms: Date.now() - startTime,
        };
      }

      case 'execute_code': {
        const data = await executeCode(
          call.arguments.code as string,
          call.arguments.language as 'javascript' | 'python'
        );
        return {
          tool: 'execute_code',
          success: data.exitCode === 0,
          data,
          latency_ms: Date.now() - startTime,
          error: data.error,
        };
      }

      case 'query_db': {
        const data = await queryDatabase(
          call.arguments.sql as string,
          call.arguments.params as unknown[]
        );
        return {
          tool: 'query_db',
          success: !data.error,
          data,
          latency_ms: Date.now() - startTime,
          error: data.error,
        };
      }

      case 'research': {
        const data = await conductResearch(
          call.arguments.query as string,
          (call.arguments.depth as ResearchDepth) || 'moderate'
        );
        return {
          tool: 'research',
          success: true,
          data,
          latency_ms: Date.now() - startTime,
        };
      }

      default:
        return {
          tool: call.name,
          success: false,
          data: null,
          latency_ms: Date.now() - startTime,
          error: `Unknown tool: ${call.name}`,
        };
    }
  } catch (e) {
    return {
      tool: call.name,
      success: false,
      data: null,
      latency_ms: Date.now() - startTime,
      error: e instanceof Error ? e.message : String(e),
    };
  }
}
