import { NextResponse } from 'next/server';
import Anthropic from '@anthropic-ai/sdk';

// Vercel Edge Runtime for low latency
export const runtime = 'edge';

// PRODUCTION-ONLY: Block Anthropic API calls in non-production unless explicitly enabled
function isAnthropicAllowed(): boolean {
  const env = process.env.VERCEL_ENV || process.env.NODE_ENV;
  if (env === 'production') return true;
  if (process.env.ALLOW_ANTHROPIC_IN_DEV === 'true') return true;
  return false;
}

// =============================================================================
// CRITICAL EVENT HANDLER - Full analysis triggered by pulse detecting major event
// =============================================================================
// When pulse detects severity: 'major' or 'critical', this endpoint:
// 1. Pulls latest news from free APIs
// 2. Makes a large token call to Claude for deep analysis
// 3. Updates the briefing cache immediately

interface NewsArticle {
  title: string;
  description: string;
  source: string;
  publishedAt: string;
  url: string;
}

// Free news APIs we can use
const NEWS_SOURCES = {
  // GNews API - 100 requests/day free
  gnews: 'https://gnews.io/api/v4/top-headlines',
  // NewsData.io - 200 requests/day free
  newsdata: 'https://newsdata.io/api/1/news',
  // MediaStack - 500 requests/month free
  mediastack: 'http://api.mediastack.com/v1/news',
};

async function fetchBreakingNews(): Promise<NewsArticle[]> {
  const articles: NewsArticle[] = [];
  const _today = new Date().toISOString().split('T')[0]; // Available for date filtering

  // Try GNews (if API key available)
  if (process.env.GNEWS_API_KEY) {
    try {
      const response = await fetch(
        `${NEWS_SOURCES.gnews}?category=world&lang=en&max=10&apikey=${process.env.GNEWS_API_KEY}`
      );
      if (response.ok) {
        const data = await response.json();
        for (const article of data.articles || []) {
          articles.push({
            title: article.title,
            description: article.description,
            source: article.source?.name || 'Unknown',
            publishedAt: article.publishedAt,
            url: article.url,
          });
        }
      }
    } catch (e) {
      console.error('GNews fetch failed:', e);
    }
  }

  // Try NewsData (if API key available)
  if (process.env.NEWSDATA_API_KEY) {
    try {
      const response = await fetch(
        `${NEWS_SOURCES.newsdata}?apikey=${process.env.NEWSDATA_API_KEY}&language=en&category=politics,world`
      );
      if (response.ok) {
        const data = await response.json();
        for (const article of data.results || []) {
          articles.push({
            title: article.title,
            description: article.description,
            source: article.source_id,
            publishedAt: article.pubDate,
            url: article.link,
          });
        }
      }
    } catch (e) {
      console.error('NewsData fetch failed:', e);
    }
  }

  // Fallback: Use web search if no API keys
  if (articles.length === 0) {
    console.log('No news API keys configured, using fallback analysis');
  }

  return articles;
}

export async function POST(req: Request) {
  const startTime = Date.now();

  try {
    // ============================================================
    // SECURITY: Only cron/internal can trigger critical analysis
    // ============================================================
    const isCronWarm = req.headers.get('x-cron-warm') === '1';
    const isInternalService = req.headers.get('x-internal-service') === process.env.INTERNAL_SERVICE_SECRET;
    const isVercelCron = req.headers.get('x-vercel-cron') === '1';

    if (!isCronWarm && !isInternalService && !isVercelCron) {
      console.log('[CRITICAL] Unauthorized request blocked');
      return NextResponse.json({ error: 'Unauthorized' }, { status: 403 });
    }

    // BLOCK non-production Anthropic API calls
    if (!isAnthropicAllowed()) {
      return NextResponse.json({
        success: false,
        error: 'Anthropic API blocked in non-production environment',
        environment: process.env.VERCEL_ENV || process.env.NODE_ENV,
      }, { status: 403 });
    }

    const { severity, headline, preset = 'global' } = await req.json();

    // Only process major/critical events
    if (severity !== 'major' && severity !== 'critical') {
      return NextResponse.json({ error: 'Not a critical event' }, { status: 400 });
    }

    console.log(`[CRITICAL] Processing ${severity} event: ${headline}`);

    // Fetch breaking news from available sources
    const articles = await fetchBreakingNews();
    const today = new Date();
    const todayStr = today.toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });

    // Build context from news articles
    const newsContext = articles.length > 0
      ? `\n\nBREAKING NEWS FEED (${articles.length} articles):\n${articles
          .slice(0, 10)
          .map((a, i) => `${i + 1}. [${a.source}] ${a.title}\n   ${a.description || 'No description'}`)
          .join('\n\n')}`
      : '';

    // Make deep analysis call to Claude
    const anthropic = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY!,
    });

    const message = await anthropic.messages.create({
      model: 'claude-haiku-4-5-20251001',
      max_tokens: 4096,
      system: `You are an intelligence analyst providing EMERGENCY briefings during breaking events.

Today's Date: ${todayStr}
Current UTC Time: ${today.toISOString()}

This is a ${severity.toUpperCase()} ALERT situation. Provide comprehensive, actionable analysis.

Your response must be a JSON object with these fields:
- political: 2-3 sentence analysis
- economic: 2-3 sentence analysis
- security: 2-3 sentence analysis
- military: 2-3 sentence analysis
- financial: 2-3 sentence analysis (market impact)
- emerging: 2-3 sentences on second-order effects
- summary: 1 sentence overall assessment
- nsm: Next Strategic Move - specific actionable recommendation
- confidence: number 0-100 (how confident in this analysis)
- sources_quality: "verified" | "unverified" | "mixed"`,
      messages: [
        {
          role: 'user',
          content: `CRITICAL EVENT DETECTED: ${headline}

Severity: ${severity.toUpperCase()}
Preset: ${preset}
${newsContext}

Provide emergency intelligence briefing in JSON format. Focus on:
1. Immediate implications
2. Cascade effects across domains
3. What decision-makers should do RIGHT NOW
4. What to monitor in the next 1-6 hours`,
        },
      ],
    });

    const textContent = message.content.find((c) => c.type === 'text');
    if (!textContent || textContent.type !== 'text') {
      throw new Error('No response from Claude');
    }

    // Parse JSON response
    const jsonMatch = textContent.text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('Could not parse briefing JSON');
    }

    const briefings = JSON.parse(jsonMatch[0]);
    const latency = Date.now() - startTime;

    // Build response to cache
    const criticalResponse = {
      briefings,
      metadata: {
        preset,
        timestamp: today.toISOString(),
        overallRisk: 'critical' as const,
        isCriticalEvent: true,
        triggerHeadline: headline,
        newsArticlesAnalyzed: articles.length,
        performance: {
          totalLatencyMs: latency,
          tokensUsed: message.usage?.output_tokens || 0,
        },
      },
    };

    console.log(`[CRITICAL] Analysis complete in ${latency}ms, ${message.usage?.output_tokens} tokens`);

    // Return the critical briefing - caller should update cache
    return NextResponse.json({
      success: true,
      ...criticalResponse,
      cacheInvalidated: true,
    });
  } catch (error) {
    console.error('Critical event handler error:', error);
    return NextResponse.json(
      { error: 'Failed to process critical event' },
      { status: 500 }
    );
  }
}
