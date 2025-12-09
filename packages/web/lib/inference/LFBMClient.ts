/**
 * LFBM Client - Drop-in replacement for Anthropic calls
 *
 * Routes briefing generation to your self-hosted vLLM endpoint
 * running a fine-tuned Qwen2.5-3B-Instruct model.
 *
 * Setup:
 * 1. Fine-tune model on RunPod Axolotl (see packages/lfbm/axolotl/)
 * 2. Deploy with vLLM serverless on RunPod
 * 3. Set LFBM_ENDPOINT in Vercel env vars
 * 4. Set LFBM_API_KEY to your RunPod API key
 *
 * RunPod Endpoints (LatticeForge):
 * - Axolotl Training: https://api.runpod.ai/v2/w5w7c0mseycf23/run
 * - vLLM Inference:   https://api.runpod.ai/v2/p2rvk115ebb42j/run
 *
 * Cost comparison:
 * - Anthropic Haiku: ~$0.25-0.75 per briefing
 * - LFBM on RunPod vLLM: ~$0.001 per briefing
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

// System prompt matching training data format
const SYSTEM_PROMPT = `You are a prose translation engine for an intelligence pipeline.

Your job: Convert numerical metrics into professional intelligence briefings.

Input format:
- Nation risk data (country code, risk score 0-1, trend)
- Signal data (GDELT article counts, sentiment tones)
- Category risk levels (political, economic, security, etc.)

Output format: JSON with briefings for each category.

Rules:
1. Reference the SPECIFIC metrics provided
2. Use professional intelligence analyst voice
3. Output valid JSON only
4. Do not fabricate events - describe what the NUMBERS indicate`;

export class LFBMClient {
  private endpoint: string;
  private apiKey?: string;
  private model: string;

  constructor(endpoint?: string, apiKey?: string) {
    this.endpoint = endpoint || process.env.LFBM_ENDPOINT || '';
    this.apiKey = apiKey || process.env.LFBM_API_KEY;
    this.model = process.env.LFBM_MODEL || 'Qwen/Qwen2.5-3B-Instruct';
  }

  isConfigured(): boolean {
    return !!this.endpoint;
  }

  /**
   * Format input data into the prompt format matching training data
   */
  private formatInput(request: LFBMRequest): string {
    const nationLines = request.nations.slice(0, 10).map((n) => {
      const trendStr = n.trend > 0 ? '↑' : n.trend < 0 ? '↓' : '→';
      const riskPct = Math.round(n.risk * 100);
      return `  ${n.code}: risk=${riskPct}% ${trendStr}`;
    });

    const signalLines = Object.entries(request.signals).map(
      ([k, v]) => `  ${k}: ${typeof v === 'number' ? v.toFixed(1) : v}`
    );

    const catLines = Object.entries(request.categories).map(
      ([k, v]) => `  ${k}: ${v}/100`
    );

    return `PIPELINE METRICS (translate to briefings):

NATIONS:
${nationLines.length > 0 ? nationLines.join('\n') : '  No nation data'}

SIGNALS:
${signalLines.length > 0 ? signalLines.join('\n') : '  No signal data'}

CATEGORY RISKS:
${catLines.length > 0 ? catLines.join('\n') : '  No category data'}

Generate JSON briefings for each category.`;
  }

  /**
   * Call RunPod serverless vLLM endpoint
   * RunPod uses /runsync for synchronous calls with { input: { ... } } wrapper
   */
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

    const userMessage = this.formatInput(request);
    const startTime = Date.now();

    // Detect if this is a RunPod serverless endpoint
    const isRunPod = this.endpoint.includes('api.runpod.ai');

    if (isRunPod) {
      // RunPod serverless format - use /runsync for synchronous response
      // Replace /run with /runsync if present
      const syncEndpoint = this.endpoint.replace(/\/run$/, '/runsync');

      const response = await fetch(syncEndpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          input: {
            messages: [
              { role: 'system', content: SYSTEM_PROMPT },
              { role: 'user', content: userMessage },
            ],
            max_tokens: request.max_tokens || 1024,
            temperature: request.temperature || 0.7,
            model: this.model,
          },
        }),
      });

      if (!response.ok) {
        const error = await response.text();
        throw new Error(`LFBM RunPod error: ${response.status} ${error}`);
      }

      const data = await response.json();
      const latencyMs = Date.now() - startTime;

      // RunPod response formats vary by worker type:
      // - vLLM: { output: [{ choices: [{ tokens: [...] }] }] }
      // - OpenAI compat: { output: { choices: [{ message: { content } }] } }
      // - Raw text: { output: "string" }
      let content: string;
      if (typeof data.output === 'string') {
        content = data.output;
      } else if (Array.isArray(data.output) && data.output[0]?.choices?.[0]?.tokens) {
        // vLLM format: output is array, tokens is array of strings
        content = data.output[0].choices[0].tokens.join('');
      } else if (data.output?.choices?.[0]?.message?.content) {
        // OpenAI compat format
        content = data.output.choices[0].message.content;
      } else if (data.output?.text) {
        content = data.output.text;
      } else {
        content = JSON.stringify(data.output || {});
      }

      let briefings: Record<string, string>;
      try {
        // Try to extract JSON from response
        const jsonMatch = content.match(/\{[\s\S]*\}/);
        briefings = jsonMatch ? JSON.parse(jsonMatch[0]) : { raw: content };
      } catch {
        briefings = { raw: content };
      }

      return {
        briefings,
        latency_ms: latencyMs,
        tokens_generated: data.output?.usage?.completion_tokens || 0,
        model: this.model,
      };
    }

    // Direct vLLM server (non-RunPod) - use OpenAI-compatible endpoint
    const response = await fetch(
      `${this.endpoint}/openai/v1/chat/completions`,
      {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model: this.model,
          messages: [
            { role: 'system', content: SYSTEM_PROMPT },
            { role: 'user', content: userMessage },
          ],
          max_tokens: request.max_tokens || 1024,
          temperature: request.temperature || 0.7,
        }),
      }
    );

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`LFBM error: ${response.status} ${error}`);
    }

    const data = await response.json();
    const latencyMs = Date.now() - startTime;

    // Parse the response
    const content = data.choices?.[0]?.message?.content || '{}';
    let briefings: Record<string, string>;

    try {
      briefings = JSON.parse(content);
    } catch {
      // If not valid JSON, wrap the content as a single briefing
      briefings = { raw: content };
    }

    return {
      briefings,
      latency_ms: latencyMs,
      tokens_generated: data.usage?.completion_tokens || 0,
      model: data.model || this.model,
    };
  }

  /**
   * Convert from the format used in intel-briefing route
   */
  async generateFromMetrics(
    nationData: Array<{
      code: string;
      name: string;
      basin_strength?: number;
      transition_risk?: number;
      regime?: number;
    }>,
    gdeltSignals: Record<string, number>,
    categoryRisks: Record<string, number>
  ): Promise<Record<string, string>> {
    const request: LFBMRequest = {
      nations: nationData.map((n) => ({
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

  /**
   * Health check for the vLLM endpoint
   */
  async healthCheck(): Promise<{
    healthy: boolean;
    latency_ms: number;
    error?: string;
  }> {
    const startTime = Date.now();

    try {
      const response = await fetch(`${this.endpoint}/health`, {
        method: 'GET',
        headers: this.apiKey
          ? { Authorization: `Bearer ${this.apiKey}` }
          : undefined,
      });

      return {
        healthy: response.ok,
        latency_ms: Date.now() - startTime,
      };
    } catch (error) {
      return {
        healthy: false,
        latency_ms: Date.now() - startTime,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
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
