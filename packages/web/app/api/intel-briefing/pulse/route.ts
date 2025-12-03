import { NextResponse } from 'next/server';
import Anthropic from '@anthropic-ai/sdk';

// Vercel Edge Runtime for low latency
export const runtime = 'edge';

// =============================================================================
// PULSE CHECK - Ultra-cheap breaking news detector
// =============================================================================
// Runs every 30-60 seconds, uses ~50 tokens to check for earth-shattering events
// If something major is detected, invalidates the main briefing cache

// Pulse check interval: 60 seconds
const PULSE_INTERVAL_MS = 60 * 1000;

interface PulseResult {
  breaking: boolean;
  severity: 'none' | 'significant' | 'major' | 'critical';
  headline?: string;
  category?: string;
  timestamp: string;
}

// Simple in-memory pulse cache
let lastPulseCheck: { result: PulseResult; timestamp: number } | null = null;

// Reference to main briefing cache invalidation (imported from parent route)
// In production, this would use a shared cache like Redis or Vercel KV
const pulseAlerts: PulseResult[] = [];

// Flag to prevent multiple simultaneous critical analyses
let criticalAnalysisInProgress = false;

// Trigger deep analysis for critical events
async function triggerCriticalAnalysis(pulse: PulseResult): Promise<void> {
  // Prevent duplicate analyses
  if (criticalAnalysisInProgress) {
    console.log('[PULSE] Critical analysis already in progress, skipping');
    return;
  }

  criticalAnalysisInProgress = true;

  try {
    console.log(`[PULSE] Triggering critical analysis for: ${pulse.headline}`);

    // Call the critical event handler
    // In edge runtime, we make an internal fetch to the critical endpoint
    const baseUrl = process.env.VERCEL_URL
      ? `https://${process.env.VERCEL_URL}`
      : process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';

    const response = await fetch(`${baseUrl}/api/intel-briefing/critical`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        severity: pulse.severity,
        headline: pulse.headline,
        preset: 'global', // Analyze all presets on critical
      }),
    });

    if (response.ok) {
      const result = await response.json();
      console.log(`[PULSE] Critical analysis complete, cache updated`);

      // Store the critical briefing for immediate access
      if (result.briefings) {
        pulseAlerts.push({
          ...pulse,
          criticalBriefing: result.briefings,
        } as PulseResult);
      }
    } else {
      console.error(`[PULSE] Critical analysis failed: ${response.status}`);
    }
  } catch (error) {
    console.error('[PULSE] Critical analysis error:', error);
  } finally {
    criticalAnalysisInProgress = false;
  }
}

export async function GET() {
  const now = Date.now();

  // Return cached pulse if checked recently
  if (lastPulseCheck && (now - lastPulseCheck.timestamp) < PULSE_INTERVAL_MS) {
    return NextResponse.json({
      ...lastPulseCheck.result,
      cached: true,
      checkAgeSeconds: Math.round((now - lastPulseCheck.timestamp) / 1000),
    });
  }

  try {
    const anthropic = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY!,
    });

    // Ultra-minimal prompt - just checking for breaking news
    // Uses Haiku for speed and cost (fraction of a cent)
    const message = await anthropic.messages.create({
      model: 'claude-3-5-haiku-20241022',
      max_tokens: 100,
      system: `You check for breaking world events. Respond ONLY with JSON:
{"breaking":true/false,"severity":"none|significant|major|critical","headline":"brief if breaking"}
Major = war outbreak, terror attack, market crash, natural disaster, leader death
Be conservative - false positives waste resources.`,
      messages: [
        {
          role: 'user',
          content: `Current time: ${new Date().toISOString()}. Any earth-shattering global events in the last hour that would affect geopolitical intelligence analysis?`,
        },
      ],
    });

    // Parse response
    const textContent = message.content.find((c) => c.type === 'text');
    if (!textContent || textContent.type !== 'text') {
      throw new Error('No response');
    }

    let pulseResult: PulseResult;
    try {
      const parsed = JSON.parse(textContent.text);
      pulseResult = {
        breaking: Boolean(parsed.breaking),
        severity: parsed.severity || 'none',
        headline: parsed.headline,
        category: parsed.category,
        timestamp: new Date().toISOString(),
      };
    } catch {
      // If parsing fails, assume no breaking news
      pulseResult = {
        breaking: false,
        severity: 'none',
        timestamp: new Date().toISOString(),
      };
    }

    // Cache the result
    lastPulseCheck = {
      result: pulseResult,
      timestamp: now,
    };

    // If major/critical event detected, trigger immediate deep analysis
    if (pulseResult.breaking && (pulseResult.severity === 'major' || pulseResult.severity === 'critical')) {
      pulseAlerts.push(pulseResult);
      console.log(`[PULSE ALERT] Breaking event detected: ${pulseResult.headline}`);

      // Trigger critical event handler to do deep analysis with news APIs
      // This runs async - don't await to keep pulse response fast
      void triggerCriticalAnalysis(pulseResult);
    }

    return NextResponse.json({
      ...pulseResult,
      cached: false,
      tokensUsed: message.usage?.output_tokens || 0,
    });
  } catch (error) {
    console.error('Pulse check error:', error);

    // On error, return last known state or default
    if (lastPulseCheck) {
      return NextResponse.json({
        ...lastPulseCheck.result,
        cached: true,
        error: 'Pulse check failed, returning last known state',
      });
    }

    return NextResponse.json({
      breaking: false,
      severity: 'none',
      timestamp: new Date().toISOString(),
      error: 'Pulse check unavailable',
    });
  }
}

// Get recent alerts (for the main briefing to check)
export async function POST() {
  // Return any alerts from the last 10 minutes
  const tenMinutesAgo = Date.now() - (10 * 60 * 1000);
  const recentAlerts = pulseAlerts.filter(
    (a) => new Date(a.timestamp).getTime() > tenMinutesAgo
  );

  return NextResponse.json({
    alerts: recentAlerts,
    lastCheck: lastPulseCheck?.timestamp
      ? new Date(lastPulseCheck.timestamp).toISOString()
      : null,
  });
}
