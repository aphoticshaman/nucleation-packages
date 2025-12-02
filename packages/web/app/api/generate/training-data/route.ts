import { NextResponse } from 'next/server';
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import Anthropic from '@anthropic-ai/sdk';

// Lazy initialization to avoid build-time errors
let supabase: SupabaseClient | null = null;
let anthropic: Anthropic | null = null;

function getSupabase() {
  if (!supabase) {
    supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );
  }
  return supabase;
}

function getAnthropic() {
  if (!anthropic) {
    anthropic = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY!,
    });
  }
  return anthropic;
}

// RSS feeds by domain - comprehensive multi-sector coverage
const RSS_SOURCES: Record<string, { url: string; domain: string }[]> = {
  // Geopolitical & World Affairs
  geopolitical: [
    { url: 'https://feeds.reuters.com/Reuters/worldNews', domain: 'geopolitical' },
    { url: 'https://feeds.npr.org/1004/rss.xml', domain: 'geopolitical' },
    { url: 'https://rss.nytimes.com/services/xml/rss/nyt/World.xml', domain: 'geopolitical' },
  ],
  // Financial & Economic
  financial: [
    { url: 'https://feeds.reuters.com/reuters/businessNews', domain: 'financial' },
    { url: 'https://feeds.bloomberg.com/markets/news.rss', domain: 'financial' },
    { url: 'https://www.cnbc.com/id/100003114/device/rss/rss.html', domain: 'financial' },
  ],
  // Cybersecurity
  cyber: [
    { url: 'https://www.cisa.gov/news.xml', domain: 'cyber' },
    { url: 'https://krebsonsecurity.com/feed/', domain: 'cyber' },
    { url: 'https://feeds.feedburner.com/TheHackersNews', domain: 'cyber' },
    { url: 'https://www.darkreading.com/rss.xml', domain: 'cyber' },
  ],
  // Defense & Military
  defense: [
    { url: 'https://www.defensenews.com/arc/outboundfeeds/rss/?outputType=xml', domain: 'defense' },
    { url: 'https://breakingdefense.com/feed/', domain: 'defense' },
    { url: 'https://www.janes.com/feeds/news', domain: 'defense' },
  ],
  // Energy & Petrochemical
  energy: [
    { url: 'https://www.eia.gov/rss/todayinenergy.xml', domain: 'energy' },
    { url: 'https://oilprice.com/rss/main', domain: 'energy' },
    { url: 'https://www.rigzone.com/news/rss/rigzone_latest.aspx', domain: 'energy' },
  ],
  // Healthcare & Pharma
  health: [
    { url: 'https://tools.cdc.gov/api/v2/resources/media/rss', domain: 'health' },
    { url: 'https://www.who.int/rss-feeds/news-english.xml', domain: 'health' },
    { url: 'https://www.fiercepharma.com/rss/xml', domain: 'pharma' },
    { url: 'https://www.fiercebiotech.com/rss/xml', domain: 'biotech' },
  ],
  // Biotech & Cancer Research
  biotech: [
    { url: 'https://www.nature.com/nbt.rss', domain: 'biotech' },
    { url: 'https://www.cancer.gov/news-events/cancer-currents-blog/rss', domain: 'cancer_research' },
    { url: 'https://www.statnews.com/feed/', domain: 'biotech' },
  ],
  // Technology & AI
  tech: [
    { url: 'https://feeds.arstechnica.com/arstechnica/technology-lab', domain: 'tech' },
    { url: 'https://techcrunch.com/feed/', domain: 'tech' },
    { url: 'https://www.wired.com/feed/rss', domain: 'tech' },
    { url: 'https://news.ycombinator.com/rss', domain: 'tech' },
  ],
  // Quantum & Fusion
  quantum_fusion: [
    { url: 'https://thequantuminsider.com/feed/', domain: 'quantum' },
    { url: 'https://www.nextbigfuture.com/feed', domain: 'fusion' },
    { url: 'https://physicsworld.com/feed/', domain: 'quantum' },
  ],
  // Space & Aerospace
  space: [
    { url: 'https://spacenews.com/feed/', domain: 'space' },
    { url: 'https://www.nasa.gov/rss/dyn/breaking_news.rss', domain: 'space' },
    { url: 'https://feeds.arstechnica.com/arstechnica/science', domain: 'space' },
  ],
  // Telecom
  telecom: [
    { url: 'https://www.fiercewireless.com/rss/xml', domain: 'telecom' },
    { url: 'https://www.lightreading.com/rss.xml', domain: 'telecom' },
  ],
  // Automotive & EV
  automotive: [
    { url: 'https://www.autonews.com/rss.xml', domain: 'automotive' },
    { url: 'https://electrek.co/feed/', domain: 'automotive' },
    { url: 'https://insideevs.com/rss/news/all/', domain: 'automotive' },
  ],
  // Manufacturing & Industrial
  manufacturing: [
    { url: 'https://www.industryweek.com/rss.xml', domain: 'manufacturing' },
    { url: 'https://www.supplychaindive.com/feeds/news/', domain: 'supply_chain' },
  ],
  // Climate & Environment
  climate: [
    { url: 'https://www.noaa.gov/rss.xml', domain: 'climate' },
    { url: 'https://www.epa.gov/rss/epa-news.xml', domain: 'climate' },
    { url: 'https://climate.nasa.gov/rss/news', domain: 'climate' },
  ],
  // Employment & Labor
  employment: [
    { url: 'https://www.bls.gov/feed/bls_latest.rss', domain: 'employment' },
    { url: 'https://www.shrm.org/rss/pages/rss.aspx', domain: 'employment' },
  ],
  // Agriculture & Food Security
  agriculture: [
    { url: 'https://www.usda.gov/rss/latest-releases.xml', domain: 'agriculture' },
    { url: 'https://www.fao.org/news/rss-feed/en/', domain: 'agriculture' },
  ],
  // Neurotech & Brain-Computer Interface
  neurotech: [
    { url: 'https://www.neurotechreports.com/rss.xml', domain: 'neurotech' },
    { url: 'https://www.medgadget.com/feed', domain: 'neurotech' },
    { url: 'https://spectrum.ieee.org/feeds/topic/biomedical', domain: 'neurotech' },
  ],
  // Semiconductors & Chips
  semiconductors: [
    { url: 'https://www.eetimes.com/feed/', domain: 'semiconductors' },
    { url: 'https://semiengineering.com/feed/', domain: 'semiconductors' },
    { url: 'https://www.tomshardware.com/feeds/all', domain: 'semiconductors' },
  ],
  // AI & Machine Learning
  ai: [
    { url: 'https://venturebeat.com/category/ai/feed/', domain: 'ai' },
    { url: 'https://www.marktechpost.com/feed/', domain: 'ai' },
    { url: 'https://syncedreview.com/feed/', domain: 'ai' },
  ],
  // Robotics & Automation
  robotics: [
    { url: 'https://www.therobotreport.com/feed/', domain: 'robotics' },
    { url: 'https://spectrum.ieee.org/feeds/topic/robotics', domain: 'robotics' },
  ],
  // Materials Science & Nanotech
  materials: [
    { url: 'https://www.materialstoday.com/rss/', domain: 'materials' },
    { url: 'https://nanotechweb.org/cws/rss', domain: 'nanotech' },
  ],
  // Cryptocurrency & Blockchain
  crypto: [
    { url: 'https://cointelegraph.com/rss', domain: 'crypto' },
    { url: 'https://www.coindesk.com/arc/outboundfeeds/rss/', domain: 'crypto' },
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
    const response = await getAnthropic().messages.create({
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
      const { error } = await getSupabase().from('training_examples').insert({
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
  const { count } = await getSupabase()
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

  const { data, error } = await getSupabase()
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
    await getSupabase()
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
