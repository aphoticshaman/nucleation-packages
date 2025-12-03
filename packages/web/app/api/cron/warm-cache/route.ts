import { NextResponse } from 'next/server';

/**
 * CRON: Cache Warming Endpoint
 *
 * Call this endpoint periodically (every 5-10 min) to pre-warm the intel briefing cache.
 * This ensures users get instant responses from cached data instead of waiting for
 * fresh Anthropic API calls.
 *
 * Vercel cron config (vercel.json):
 * {
 *   "crons": [{
 *     "path": "/api/cron/warm-cache",
 *     "schedule": "0,10,20,30,40,50 (every-10-min)"
 *   }]
 * }
 *
 * Or call externally: curl -X POST https://yoursite.com/api/cron/warm-cache
 *                          -H "Authorization: Bearer YOUR_CRON_SECRET"
 */

export const runtime = 'edge';

// All presets to warm
const PRESETS = ['global', 'nato', 'brics', 'conflict'];

export async function POST(req: Request) {
  // Verify cron secret (prevent unauthorized warming)
  const authHeader = req.headers.get('authorization');
  const cronSecret = process.env.CRON_SECRET;

  // Allow Vercel internal cron (no auth header) or external with secret
  const isVercelCron = req.headers.get('x-vercel-cron') === '1';
  const hasValidSecret = cronSecret && authHeader === `Bearer ${cronSecret}`;

  if (!isVercelCron && !hasValidSecret) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const results: Record<string, { success: boolean; cached?: boolean; error?: string }> = {};
  const baseUrl = process.env.VERCEL_URL
    ? `https://${process.env.VERCEL_URL}`
    : process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';

  console.log(`[CRON] Warming cache for ${PRESETS.length} presets...`);

  // Warm each preset in parallel
  const warmPromises = PRESETS.map(async (preset) => {
    try {
      const response = await fetch(`${baseUrl}/api/intel-briefing`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Use service account cookie or pass special header to bypass rate limits
          'X-Cron-Warm': '1',
        },
        body: JSON.stringify({
          preset,
          forceRefresh: false, // Use cache if available, only generate if missing
        }),
      });

      if (response.ok) {
        const data = await response.json();
        results[preset] = {
          success: true,
          cached: data.metadata?.cached || false
        };
        console.log(`[CRON] Warmed ${preset}: ${data.metadata?.cached ? 'from cache' : 'fresh'}`);
      } else {
        results[preset] = {
          success: false,
          error: `HTTP ${response.status}`
        };
        console.error(`[CRON] Failed to warm ${preset}: ${response.status}`);
      }
    } catch (error) {
      results[preset] = {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
      console.error(`[CRON] Error warming ${preset}:`, error);
    }
  });

  await Promise.all(warmPromises);

  const successCount = Object.values(results).filter(r => r.success).length;

  return NextResponse.json({
    message: `Cache warming complete: ${successCount}/${PRESETS.length} succeeded`,
    results,
    timestamp: new Date().toISOString(),
  });
}

// Also support GET for Vercel cron (which uses GET by default)
export async function GET(req: Request) {
  // For GET, verify cron header
  const isVercelCron = req.headers.get('x-vercel-cron') === '1';

  if (!isVercelCron) {
    return NextResponse.json({
      error: 'Use POST for external calls, or ensure this is called from Vercel Cron'
    }, { status: 405 });
  }

  // Delegate to POST handler
  return POST(req);
}
