import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import Anthropic from '@anthropic-ai/sdk';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY!,
});

// RSS feeds by domain
const RSS_SOURCES: Record<string, { url: string; domain: string }[]> = {
  geopolitical: [
    { url: 'https://feeds.reuters.com/Reuters/worldNews', domain: 'geopolitical' },
    { url: 'https://feeds.npr.org/1004/rss.xml', domain: 'geopolitical' }, // NPR World
  ],
  economic: [
    { url: 'https://feeds.reuters.com/reuters/businessNews', domain: 'economic' },
  ],
  cyber: [
    { url: 'https://www.cisa.gov/news.xml', domain: 'cyber' },
  ],
  health: [
    { url: 'https://tools.cdc.gov/api/v2/resources/media/rss', domain: 'health' },
  ],
  climate: [
    { url: 'https://www.noaa.gov/rss.xml', domain: 'climate' },
  ],
};

// GDELT event codes that indicate significant events
const SIGNIFICANT_EVENT_CODES = ['14', '15', '17', '18', '19', '20']; // Protest, Military, Coerce, Assault, Fight, Mass Violence

interface NewsItem {
  title: string;
  description: string;
  link: string;
  pubDate: string;
  domain: string;
  source_type: string;
}

interface GDELTEvent {
  Actor1Name: string;
  Actor2Name: string | null;
  EventCode: string;
  GoldsteinScale: string;
  NumMentions: string;
  AvgTone: string;
  Actor1CountryCode: string;
  Actor2CountryCode: string | null;
  SOURCEURL: string;
  SQLDATE: string;
}

async function fetchGDELT(): Promise<NewsItem[]> {
  try {
    const url = 'https://api.gdeltproject.org/api/v2/doc/doc?query=conflict OR crisis OR war OR sanctions&mode=artlist&maxrecords=20&format=json';
    const res = await fetch(url, { next: { revalidate: 0 } });

    if (!res.ok) return [];

    const data = await res.json();
    const articles = data.articles || [];

    return articles.slice(0, 10).map((a: { title: string; seendate: string; url: string }) => ({
      title: a.title,
      description: a.title, // GDELT doesn't give description
      link: a.url,
      pubDate: a.seendate,
      domain: 'geopolitical',
      source_type: 'gdelt',
    }));
  } catch {
    console.error('GDELT fetch failed');
    return [];
  }
}

async function fetchRSS(feedUrl: string, domain: string): Promise<NewsItem[]> {
  try {
    const res = await fetch(feedUrl, { next: { revalidate: 0 } });
    if (!res.ok) return [];

    const text = await res.text();
    // Simple XML parsing for RSS
    const items: NewsItem[] = [];
    const itemMatches = text.match(/<item>([\s\S]*?)<\/item>/g) || [];

    for (const item of itemMatches.slice(0, 5)) {
      const title = item.match(/<title><!\[CDATA\[(.*?)\]\]><\/title>|<title>(.*?)<\/title>/)?.[1] || item.match(/<title>(.*?)<\/title>/)?.[1] || '';
      const description = item.match(/<description><!\[CDATA\[(.*?)\]\]><\/description>|<description>(.*?)<\/description>/)?.[1] || '';
      const link = item.match(/<link>(.*?)<\/link>/)?.[1] || '';
      const pubDate = item.match(/<pubDate>(.*?)<\/pubDate>/)?.[1] || new Date().toISOString();

      if (title) {
        items.push({
          title: title.replace(/<[^>]*>/g, ''),
          description: description.replace(/<[^>]*>/g, '').slice(0, 500),
          link,
          pubDate,
          domain,
          source_type: 'rss',
        });
      }
    }

    return items;
  } catch {
    console.error(`RSS fetch failed for ${feedUrl}`);
    return [];
  }
}

async function generateTrainingExample(news: NewsItem): Promise<{
  instruction: string;
  input: string;
  output: string;
  confidence: number;
} | null> {
  const prompt = `You are a geopolitical intelligence analyst. Convert this news into a training example for an AI risk analysis system.

NEWS:
Title: ${news.title}
Description: ${news.description}
Date: ${news.pubDate}
Domain: ${news.domain}

Generate a training example with:
1. A detailed INPUT that describes the situation with specific details, metrics, and context
2. An expert OUTPUT analysis covering:
   - Risk assessment (CRITICAL/HIGH/ELEVATED/LOW)
   - Key indicators and signals
   - Cascade potential to other domains (economic, military, social, etc.)
   - Historical parallels
   - Recommended monitoring actions

Respond in this exact JSON format:
{
  "input": "Detailed situation description with dates, actors, metrics...",
  "output": "RISK ASSESSMENT: [LEVEL]\\n\\n1) KEY INDICATORS: ...\\n\\n2) CASCADE POTENTIAL: ...\\n\\n3) HISTORICAL PARALLELS: ...\\n\\n4) MONITORING: ...",
  "confidence": 0.85
}

Only output the JSON, nothing else.`;

  try {
    const response = await anthropic.messages.create({
      model: 'claude-3-haiku-20240307',
      max_tokens: 1024,
      messages: [{ role: 'user', content: prompt }],
    });

    const text = response.content[0].type === 'text' ? response.content[0].text : '';
    const parsed = JSON.parse(text);

    return {
      instruction: `Analyze the ${news.domain} risk signals in the following situation`,
      input: parsed.input,
      output: parsed.output,
      confidence: parsed.confidence || 0.8,
    };
  } catch (e) {
    console.error('LLM generation failed:', e);
    return null;
  }
}

export async function GET(request: Request) {
  const authHeader = request.headers.get('authorization');
  const cronSecret = process.env.CRON_SECRET;

  // Allow cron or authenticated requests
  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    // Check if it's a Vercel cron request
    const isVercelCron = request.headers.get('x-vercel-cron') === '1';
    if (!isVercelCron) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
  }

  const results = {
    fetched: 0,
    generated: 0,
    stored: 0,
    errors: [] as string[],
    domains: {} as Record<string, number>,
  };

  // Fetch from all sources
  const allNews: NewsItem[] = [];

  // GDELT
  const gdeltNews = await fetchGDELT();
  allNews.push(...gdeltNews);

  // RSS feeds
  for (const [, sources] of Object.entries(RSS_SOURCES)) {
    for (const source of sources) {
      const news = await fetchRSS(source.url, source.domain);
      allNews.push(...news);
    }
  }

  results.fetched = allNews.length;

  // Generate training examples
  for (const news of allNews) {
    try {
      const example = await generateTrainingExample(news);
      if (!example) continue;

      results.generated++;

      // Store in Supabase
      const { error } = await supabase.from('training_examples').insert({
        instruction: example.instruction,
        input: example.input,
        output: example.output,
        domain: news.domain,
        source_type: news.source_type,
        source_url: news.link,
        source_date: new Date(news.pubDate).toISOString(),
        confidence: example.confidence,
      });

      if (error) {
        if (error.code === '23505') {
          // Duplicate - already have this one
          continue;
        }
        results.errors.push(`Insert error: ${error.message}`);
      } else {
        results.stored++;
        results.domains[news.domain] = (results.domains[news.domain] || 0) + 1;
      }
    } catch (e) {
      results.errors.push(`Processing error: ${e}`);
    }
  }

  // Get total count
  const { count } = await supabase
    .from('training_examples')
    .select('*', { count: 'exact', head: true });

  return NextResponse.json({
    success: true,
    ...results,
    total_examples: count,
    timestamp: new Date().toISOString(),
  });
}

// Export endpoint to download training data
export async function POST(request: Request) {
  const { format = 'alpaca', mark_exported = false } = await request.json();

  const { data, error } = await supabase
    .from('training_examples')
    .select('id, instruction, input, output')
    .eq('exported', false)
    .gte('confidence', 0.7)
    .order('created_at', { ascending: false });

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  if (mark_exported && data.length > 0) {
    const ids = data.map(d => d.id);
    await supabase
      .from('training_examples')
      .update({ exported: true })
      .in('id', ids);
  }

  if (format === 'alpaca') {
    const alpaca = data.map(d => ({
      instruction: d.instruction,
      input: d.input,
      output: d.output,
    }));
    return NextResponse.json(alpaca);
  }

  // ChatML format
  const chatml = data.map(d => ({
    messages: [
      { role: 'system', content: 'You are a geopolitical risk analyst.' },
      { role: 'user', content: `${d.instruction}\n\n${d.input}` },
      { role: 'assistant', content: d.output },
    ],
  }));

  return NextResponse.json(chatml);
}
