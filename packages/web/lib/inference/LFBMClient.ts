/**
 * LFBM Client - Drop-in replacement for Anthropic calls
 *
 * Routes briefing generation to your self-hosted LFBM model
 * instead of Anthropic API.
 *
 * Setup:
 * 1. Deploy LFBM on RunPod (see packages/lfbm/inference/server.py)
 * 2. Set LFBM_ENDPOINT in Vercel env vars
 * 3. Set LFBM_API_KEY (optional, for auth)
 *
 * Cost comparison:
 * - Anthropic Haiku: ~$0.25-0.75 per briefing
 * - LFBM on RunPod: ~$0.001 per briefing
 */

export interface LFBMNationInput {
  code: string;
  name?: string;
  risk: number;
  trend: number;
}

export interface LFBMRequest {
  nations: LFBMNationInput[];
  signals: Record<string, number>;
  categories: Record<string, number>;
  max_tokens?: number;
  temperature?: number;
}

export interface LFBMResponse {
  briefings: Record<string, string>;
  latency_ms: number;
  tokens_generated: number;
  model: string;
}

export class LFBMClient {
  private endpoint: string;
  private apiKey?: string;

  constructor(endpoint?: string, apiKey?: string) {
    this.endpoint = endpoint || process.env.LFBM_ENDPOINT || '';
    this.apiKey = apiKey || process.env.LFBM_API_KEY;
  }

  isConfigured(): boolean {
    return !!this.endpoint;
  }

  async generateBriefing(request: LFBMRequest): Promise<LFBMResponse> {
    if (!this.endpoint) {
      throw new Error('LFBM_ENDPOINT not configured');
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const response = await fetch(`${this.endpoint}/v1/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        nations: request.nations,
        signals: request.signals,
        categories: request.categories,
        max_tokens: request.max_tokens || 1024,
        temperature: request.temperature || 0.7,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`LFBM error: ${response.status} ${error}`);
    }

    return response.json();
  }

  /**
   * Convert from the format used in intel-briefing route
   */
  async generateFromMetrics(
    nationData: Array<{ code: string; name: string; basin_strength?: number; transition_risk?: number; regime?: number }>,
    gdeltSignals: Record<string, number>,
    categoryRisks: Record<string, number>
  ): Promise<Record<string, string>> {
    const request: LFBMRequest = {
      nations: nationData.map(n => ({
        code: n.code,
        name: n.name,
        risk: n.transition_risk || 0,
        trend: (n.basin_strength || 0) > 0.5 ? 0.1 : -0.1,
      })),
      signals: {
        gdelt_count: gdeltSignals.count || 0,
        avg_tone: gdeltSignals.avg_tone || 0,
        alert_count: gdeltSignals.alerts || 0,
      },
      categories: categoryRisks,
    };

    const response = await this.generateBriefing(request);
    return response.briefings;
  }
}

// Singleton instance
let _client: LFBMClient | null = null;

export function getLFBMClient(): LFBMClient {
  if (!_client) {
    _client = new LFBMClient();
  }
  return _client;
}

/**
 * Check if we should use LFBM instead of Anthropic
 */
export function shouldUseLFBM(): boolean {
  // Use LFBM if configured and explicitly enabled
  const endpoint = process.env.LFBM_ENDPOINT;
  const preferLFBM = process.env.PREFER_LFBM === 'true';

  if (!endpoint) return false;
  if (preferLFBM) return true;

  // Fall back to LFBM if Anthropic is disabled
  const anthropicDisabled = process.env.DISABLE_LLM === 'true';
  return anthropicDisabled;
}

/**
 * Example integration with existing intel-briefing route:
 *
 * // In intel-briefing/route.ts:
 * import { getLFBMClient, shouldUseLFBM } from '@/lib/inference/LFBMClient';
 *
 * // Instead of Anthropic call:
 * if (shouldUseLFBM()) {
 *   const lfbm = getLFBMClient();
 *   const briefings = await lfbm.generateFromMetrics(nationData, gdeltSignals, categoryRisks);
 *   return NextResponse.json({ briefings, metadata: { source: 'lfbm' } });
 * }
 *
 * // Otherwise continue with Anthropic...
 */
