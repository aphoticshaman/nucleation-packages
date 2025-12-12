/**
 * Elle Keep-Warm Cron
 *
 * Prevents cold starts by pinging Elle every 5 minutes.
 * RunPod serverless workers stay warm for ~5 mins after last request.
 *
 * Cost: ~$0.001 per ping × 288 pings/day = ~$0.29/day
 * Savings: Eliminates 2-5 minute cold start delays
 */

import { NextResponse } from 'next/server';

export const runtime = 'edge';

const WARMUP_PROMPT = 'Respond with only: "warm"';

export async function GET(request: Request) {
  const startTime = Date.now();

  // Verify cron secret
  const authHeader = request.headers.get('authorization');
  const cronSecret = process.env.CRON_SECRET;

  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    const isVercelCron = request.headers.get('x-vercel-cron') === '1';
    if (!isVercelCron) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
  }

  const results = {
    timestamp: new Date().toISOString(),
    elle: { status: 'skipped', latency_ms: 0 },
    workhorse: { status: 'skipped', latency_ms: 0 },
  };

  // Warm Elle (main model)
  const elleEndpoint = process.env.LFBM_ENDPOINT;
  if (elleEndpoint) {
    const elleStart = Date.now();
    try {
      const syncEndpoint = elleEndpoint.replace(/\/run$/, '/runsync');

      const response = await fetch(syncEndpoint, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${process.env.LFBM_API_KEY || process.env.RUNPOD_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input: {
            messages: [{ role: 'user', content: WARMUP_PROMPT }],
            max_tokens: 5,
            temperature: 0,
          },
        }),
      });

      if (response.ok) {
        results.elle = {
          status: 'warm',
          latency_ms: Date.now() - elleStart,
        };
      } else {
        results.elle = {
          status: `error_${response.status}`,
          latency_ms: Date.now() - elleStart,
        };
      }
    } catch (e) {
      results.elle = {
        status: `failed: ${e instanceof Error ? e.message : 'unknown'}`,
        latency_ms: Date.now() - elleStart,
      };
    }
  }

  // Warm Workhorse (if configured)
  const workhorseEndpoint = process.env.WORKHORSE_ENDPOINT;
  if (workhorseEndpoint && workhorseEndpoint !== elleEndpoint) {
    const workhorseStart = Date.now();
    try {
      const syncEndpoint = workhorseEndpoint.replace(/\/run$/, '/runsync');

      const response = await fetch(syncEndpoint, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${process.env.WORKHORSE_API_KEY || process.env.RUNPOD_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input: {
            messages: [{ role: 'user', content: WARMUP_PROMPT }],
            max_tokens: 5,
            temperature: 0,
          },
        }),
      });

      if (response.ok) {
        results.workhorse = {
          status: 'warm',
          latency_ms: Date.now() - workhorseStart,
        };
      } else {
        results.workhorse = {
          status: `error_${response.status}`,
          latency_ms: Date.now() - workhorseStart,
        };
      }
    } catch (e) {
      results.workhorse = {
        status: `failed: ${e instanceof Error ? e.message : 'unknown'}`,
        latency_ms: Date.now() - workhorseStart,
      };
    }
  }

  console.log(`[Elle Warmup] Elle: ${results.elle.status} (${results.elle.latency_ms}ms), Workhorse: ${results.workhorse.status} (${results.workhorse.latency_ms}ms)`);

  return NextResponse.json({
    success: true,
    ...results,
    totalLatency_ms: Date.now() - startTime,
  });
}
