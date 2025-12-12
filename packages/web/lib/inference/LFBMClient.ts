/**
 * LFBM Client - Self-hosted LLM inference for LatticeForge
 *
 * Routes ALL briefing generation to your self-hosted vLLM endpoint
 * running Qwen2.5-3B-Instruct. No external LLM dependencies.
 *
 * PERSONA: "Elle" - The user-facing AI intelligence analyst
 * - L from LatticeForge
 * - Elegant, professional, concise
 * - Powers latticeforge.ai/chat and all briefings
 *
 * GUARDIAN INTEGRATION:
 * - All output from Elle is validated by Guardian before caching
 * - Guardian fixes common JSON malformation issues (nested JSON, markdown blocks)
 * - Invalid output is rejected to prevent bad cache pollution
 *
 * Modes:
 * - realtime: Pure metric translation (~$0.001)
 * - historical: Pattern analysis using model's training knowledge (~$0.002)
 * - hybrid: Current metrics + historical context (~$0.003)
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
 * Non-Production Control:
 * - Set ENABLE_LFBM_IN_PREVIEW=true to enable in preview/dev
 * - Without this, LFBM calls return fallback responses in non-prod
 */

import { getOutputGuardian, type BriefingValidationResult } from '@/lib/reasoning/security';

// =============================================================================
// NON-PRODUCTION API BLOCKING
// =============================================================================
// Uses LF_PROD_ENABLE environment variable for master control:
// - 'true' in production → APIs enabled
// - 'false' in preview/dev → APIs BLOCKED by default
//
// Admin can temporarily enable via timed override (ARM + FIRE mechanism):
// 1. Set LF_ADMIN_OVERRIDE_UNTIL in localStorage (timestamp)
// 2. Override expires after the timeout (default 30 minutes)
// This prevents dozens of preview deployments from racking up RunPod costs.

// Fallback responses for when LFBM is disabled in non-prod
const FALLBACK_BRIEFING: Record<string, string> = {
  political: '[LFBM DISABLED] LF_PROD_ENABLE=false - Enable via admin override',
  economic: '[LFBM DISABLED] No RunPod calls in preview/dev by default',
  security: '[LFBM DISABLED] This is a fallback response',
  summary: 'LFBM API calls are disabled. LF_PROD_ENABLE must be true or admin must override.',
  nsm: 'N/A - Non-production environment',
};

const FALLBACK_RESPONSE: LFBMResponse = {
  briefings: FALLBACK_BRIEFING,
  latency_ms: 0,
  tokens_generated: 0,
  model: 'fallback-nonprod',
};

// In-memory admin override state (for serverless functions)
// Each function instance has its own state - this is a per-request cache
let adminOverrideExpiry: number | null = null;

/**
 * Set a timed admin override to temporarily enable APIs
 * @param minutes - How long the override should last (default 30 minutes)
 */
export function setAdminOverride(minutes: number = 30): { expiresAt: string; durationMinutes: number } {
  const expiryTime = Date.now() + (minutes * 60 * 1000);
  adminOverrideExpiry = expiryTime;

  return {
    expiresAt: new Date(expiryTime).toISOString(),
    durationMinutes: minutes,
  };
}

/**
 * Clear any active admin override
 */
export function clearAdminOverride(): void {
  adminOverrideExpiry = null;
}

/**
 * Check if admin override is active
 */
export function isAdminOverrideActive(): boolean {
  if (!adminOverrideExpiry) return false;

  // Check if override has expired
  if (Date.now() > adminOverrideExpiry) {
    adminOverrideExpiry = null;
    return false;
  }

  return true;
}

/**
 * Get remaining time on admin override
 */
export function getAdminOverrideRemaining(): { active: boolean; remainingMinutes: number } {
  if (!adminOverrideExpiry) {
    return { active: false, remainingMinutes: 0 };
  }

  const remaining = adminOverrideExpiry - Date.now();
  if (remaining <= 0) {
    adminOverrideExpiry = null;
    return { active: false, remainingMinutes: 0 };
  }

  return {
    active: true,
    remainingMinutes: Math.ceil(remaining / (60 * 1000)),
  };
}

/**
 * Master check: Is LFBM enabled in current environment?
 *
 * Priority order:
 * 1. LF_PROD_ENABLE='true' → Enabled (production)
 * 2. Admin timed override active → Enabled (temporary)
 * 3. ENABLE_APIS_IN_PREVIEW='true' → Enabled (legacy env var)
 * 4. Default → BLOCKED
 */
export function isLFBMEnabled(): boolean {
  // Primary control: LF_PROD_ENABLE environment variable
  // This is set to 'true' in production and 'false' in preview/dev
  if (process.env.LF_PROD_ENABLE === 'true') {
    return true;
  }

  // Admin timed override (ARM + FIRE mechanism)
  if (isAdminOverrideActive()) {
    console.log('[LFBM] Admin override active - APIs temporarily enabled');
    return true;
  }

  // Legacy fallback: explicit enable in preview/dev
  if (process.env.ENABLE_LFBM_IN_PREVIEW === 'true') return true;
  if (process.env.ENABLE_APIS_IN_PREVIEW === 'true') return true;

  // BLOCKED by default when LF_PROD_ENABLE is not 'true'
  console.log('[LFBM] BLOCKED - LF_PROD_ENABLE is not true and no admin override');
  return false;
}

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
  /** Guardian validation result - included for transparency */
  validation?: BriefingValidationResult;
}

// =============================================================================
// ELLE - LatticeForge Intelligence Analyst Persona
// =============================================================================
// Elle is the user-facing AI analyst powered by LFBM (Qwen2.5 on RunPod).
// Named after the "L" in LatticeForge. Elegant, professional, concise.

// System prompts for different analysis modes - MUST output raw JSON
const REALTIME_PROMPT = `You are Elle, LatticeForge's intelligence analyst. Convert metrics into actionable briefings.

CRITICAL: Output RAW JSON only. No markdown, no code blocks, no \`\`\`json.

Output format: {"political":"...","economic":"...","security":"...","summary":"...","nsm":"..."}

Rules:
1. Reference the SPECIFIC metrics provided
2. Professional, concise intelligence analyst voice
3. RAW JSON ONLY - no markdown formatting
4. Describe what the NUMBERS indicate`;

const HISTORICAL_PROMPT = `You are Elle, LatticeForge's senior intelligence historian. Analyze historical patterns and precedents.

CRITICAL: Output RAW JSON only. No markdown, no code blocks.

Output format: {"summary":"...","precedents":[{"event":"...","date":"...","parallel":"..."}],"patterns":[{"name":"...","examples":["..."],"current_phase":"..."}],"lessons":["..."],"warnings":["..."],"political":"...","economic":"...","security":"...","military":"...","nsm":"..."}

Rules:
1. Ground claims in verifiable historical events with dates
2. Connect patterns to provided nation data
3. Professional, authoritative voice
4. RAW JSON ONLY`;

const HYBRID_PROMPT = `You are Elle, LatticeForge's dual-mode analyst combining realtime metrics with historical context.

CRITICAL: Output RAW JSON only. No markdown, no code blocks.

Output format: {"summary":"...","current_assessment":{"political":"...","economic":"...","security":"..."},"historical_context":{"precedent":"...","pattern":"...","outcome":"..."},"synthesis":{"political":"...","economic":"...","security":"...","military":"...","cyber":"..."},"risk_assessment":"...","nsm":"..."}

Rules:
1. Translate current metrics to prose (Layer 1)
2. Overlay historical context with dates (Layer 2)
3. Synthesize both for assessment (Layer 3)
4. RAW JSON ONLY`;

export type AnalysisMode = 'realtime' | 'historical' | 'hybrid';

export interface HistoricalRequest {
  nations: LFBMNationInput[];
  gdeltSummary: Record<string, number>;
  focus?: string;
  selectedEras?: string[];
  depth?: 'quick' | 'standard' | 'deep';
}

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
   *
   * IMPORTANT: Blocked in non-production by default to prevent preview deployments
   * from making RunPod calls. Enable via ENABLE_APIS_IN_PREVIEW=true.
   */
  async generateBriefing(request: LFBMRequest): Promise<LFBMResponse> {
    // BLOCK in non-production unless explicitly enabled
    if (!isLFBMEnabled()) {
      console.log('[LFBM] BLOCKED - Non-production environment, returning fallback');
      return FALLBACK_RESPONSE;
    }

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
              { role: 'system', content: REALTIME_PROMPT },
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

      // GUARDIAN VALIDATION: Validate and fix Elle's output before returning
      const guardian = getOutputGuardian();
      const validation = guardian.validateBriefings(content);

      if (validation.warnings.length > 0) {
        console.log('[LFBM GUARDIAN] Warnings:', validation.warnings);
      }
      if (validation.errors.length > 0) {
        console.warn('[LFBM GUARDIAN] Errors:', validation.errors);
      }
      if (validation.fixed) {
        console.log('[LFBM GUARDIAN] Fixed malformed JSON from Elle');
      }

      // If Guardian couldn't validate, throw error instead of returning bad data
      if (!validation.valid || !validation.briefings) {
        console.error('[LFBM GUARDIAN] REJECTED - Output failed validation:', validation.errors);
        throw new Error(`LFBM output validation failed: ${validation.errors.join(', ')}`);
      }

      return {
        briefings: validation.briefings,
        latency_ms: latencyMs,
        tokens_generated: data.output?.usage?.completion_tokens || 0,
        model: this.model,
        validation,
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
            { role: 'system', content: REALTIME_PROMPT },
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

    // GUARDIAN VALIDATION: Validate and fix Elle's output before returning
    const guardian = getOutputGuardian();
    const validation = guardian.validateBriefings(content);

    if (validation.warnings.length > 0) {
      console.log('[LFBM GUARDIAN] Warnings:', validation.warnings);
    }
    if (validation.errors.length > 0) {
      console.warn('[LFBM GUARDIAN] Errors:', validation.errors);
    }
    if (validation.fixed) {
      console.log('[LFBM GUARDIAN] Fixed malformed JSON from Elle');
    }

    // If Guardian couldn't validate, throw error instead of returning bad data
    if (!validation.valid || !validation.briefings) {
      console.error('[LFBM GUARDIAN] REJECTED - Output failed validation:', validation.errors);
      throw new Error(`LFBM output validation failed: ${validation.errors.join(', ')}`);
    }

    return {
      briefings: validation.briefings,
      latency_ms: latencyMs,
      tokens_generated: data.usage?.completion_tokens || 0,
      model: data.model || this.model,
      validation,
    };
  }

  /**
   * Generate with raw prompts - for routes that need custom system/user messages
   * BLOCKED in non-production by default
   */
  async generateRaw(params: {
    systemPrompt?: string;
    userMessage: string;
    max_tokens?: number;
    temperature?: number;
  }): Promise<string> {
    // BLOCK in non-production unless explicitly enabled
    if (!isLFBMEnabled()) {
      console.log('[LFBM] BLOCKED raw generation - Non-production environment');
      return JSON.stringify({ blocked: true, reason: 'LFBM disabled in non-production', fallback: true });
    }

    if (!this.endpoint) {
      throw new Error('LFBM_ENDPOINT not configured');
    }

    const response = await this.callEndpoint(
      params.systemPrompt || 'You are Elle, LatticeForge\'s intelligence analyst. Respond with JSON only.',
      params.userMessage,
      params.max_tokens || 256
    );

    // Return raw content string for routes to parse
    return JSON.stringify(response.briefings);
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
   * Generate historical pattern analysis
   * BLOCKED in non-production by default
   */
  async generateHistoricalAnalysis(request: HistoricalRequest): Promise<LFBMResponse> {
    // BLOCK in non-production unless explicitly enabled
    if (!isLFBMEnabled()) {
      console.log('[LFBM] BLOCKED historical analysis - Non-production environment');
      return FALLBACK_RESPONSE;
    }

    if (!this.endpoint) {
      throw new Error('LFBM_ENDPOINT not configured');
    }

    const nationLines = request.nations.slice(0, 10).map((n) => {
      const riskPct = Math.round(n.risk * 100);
      return `  ${n.name || n.code}: ${riskPct}% transition risk`;
    });

    const gdeltLines = Object.entries(request.gdeltSummary).map(
      ([domain, count]) => `  ${domain}: ${count} signals`
    );

    const userMessage = `HISTORICAL META-ANALYSIS REQUEST

FOCUS AREA: ${request.focus || 'geopolitical transitions and power shifts'}

CURRENT NATION RISK DATA:
${nationLines.length > 0 ? nationLines.join('\n') : '  No nations above threshold'}

GDELT SIGNAL CONTEXT:
${gdeltLines.length > 0 ? gdeltLines.join('\n') : '  No GDELT data'}

${request.selectedEras?.length ? `ERAS TO ANALYZE: ${request.selectedEras.join(', ')}` : ''}

Generate historical pattern analysis connecting precedents to current data.`;

    return this.callEndpoint(HISTORICAL_PROMPT, userMessage, request.depth === 'deep' ? 2048 : 1024);
  }

  /**
   * Generate hybrid analysis (current + historical)
   * BLOCKED in non-production by default
   */
  async generateHybridAnalysis(
    nationData: Array<{ code: string; name: string; transition_risk?: number }>,
    gdeltSignals: Record<string, number>,
    categoryRisks: Record<string, number>,
    focus?: string
  ): Promise<LFBMResponse> {
    // BLOCK in non-production unless explicitly enabled
    if (!isLFBMEnabled()) {
      console.log('[LFBM] BLOCKED hybrid analysis - Non-production environment');
      return FALLBACK_RESPONSE;
    }

    if (!this.endpoint) {
      throw new Error('LFBM_ENDPOINT not configured');
    }

    const nationLines = nationData.slice(0, 10).map((n) => {
      const risk = ((n.transition_risk || 0) * 100).toFixed(0);
      return `  ${n.name}: ${risk}% risk`;
    });

    const catLines = Object.entries(categoryRisks).map(
      ([k, v]) => `  ${k}: ${v}/100`
    );

    const userMessage = `HYBRID ANALYSIS REQUEST

CURRENT METRICS (from live pipeline):
Nations at elevated risk:
${nationLines.length > 0 ? nationLines.join('\n') : '  None above threshold'}

Category risks:
${catLines.join('\n')}

GDELT signals: ${gdeltSignals.count || 0} in analysis window
Avg tone: ${gdeltSignals.avg_tone?.toFixed(2) || 'N/A'}

HISTORICAL FOCUS: ${focus || 'General geopolitical patterns'}

Generate hybrid briefing combining current metrics with historical context.`;

    return this.callEndpoint(HYBRID_PROMPT, userMessage, 1536);
  }

  /**
   * Internal method to call the endpoint with any prompt
   */
  private async callEndpoint(systemPrompt: string, userMessage: string, maxTokens: number): Promise<LFBMResponse> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const startTime = Date.now();
    const isRunPod = this.endpoint.includes('api.runpod.ai');

    if (isRunPod) {
      const syncEndpoint = this.endpoint.replace(/\/run$/, '/runsync');

      const response = await fetch(syncEndpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          input: {
            messages: [
              { role: 'system', content: systemPrompt },
              { role: 'user', content: userMessage },
            ],
            max_tokens: maxTokens,
            temperature: 0.7,
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

      let content: string;
      if (typeof data.output === 'string') {
        content = data.output;
      } else if (Array.isArray(data.output) && data.output[0]?.choices?.[0]?.tokens) {
        content = data.output[0].choices[0].tokens.join('');
      } else if (data.output?.choices?.[0]?.message?.content) {
        content = data.output.choices[0].message.content;
      } else if (data.output?.text) {
        content = data.output.text;
      } else {
        content = JSON.stringify(data.output || {});
      }

      // GUARDIAN VALIDATION: Validate and fix Elle's output
      const guardian = getOutputGuardian();
      const validation = guardian.validateBriefings(content);

      if (validation.warnings.length > 0) {
        console.log('[LFBM GUARDIAN callEndpoint] Warnings:', validation.warnings);
      }
      if (validation.fixed) {
        console.log('[LFBM GUARDIAN callEndpoint] Fixed malformed JSON from Elle');
      }

      // If Guardian couldn't validate, throw error instead of returning bad data
      if (!validation.valid || !validation.briefings) {
        console.error('[LFBM GUARDIAN callEndpoint] REJECTED:', validation.errors);
        throw new Error(`LFBM output validation failed: ${validation.errors.join(', ')}`);
      }

      return {
        briefings: validation.briefings,
        latency_ms: latencyMs,
        tokens_generated: data.output?.usage?.completion_tokens || 0,
        model: this.model,
        validation,
      };
    }

    // Direct vLLM server
    const response = await fetch(`${this.endpoint}/openai/v1/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        model: this.model,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userMessage },
        ],
        max_tokens: maxTokens,
        temperature: 0.7,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`LFBM error: ${response.status} ${error}`);
    }

    const data = await response.json();
    const latencyMs = Date.now() - startTime;
    const content = data.choices?.[0]?.message?.content || '{}';

    // GUARDIAN VALIDATION: Validate and fix Elle's output
    const guardian = getOutputGuardian();
    const validation = guardian.validateBriefings(content);

    if (validation.warnings.length > 0) {
      console.log('[LFBM GUARDIAN direct] Warnings:', validation.warnings);
    }
    if (validation.fixed) {
      console.log('[LFBM GUARDIAN direct] Fixed malformed JSON from Elle');
    }

    // If Guardian couldn't validate, throw error instead of returning bad data
    if (!validation.valid || !validation.briefings) {
      console.error('[LFBM GUARDIAN direct] REJECTED:', validation.errors);
      throw new Error(`LFBM output validation failed: ${validation.errors.join(', ')}`);
    }

    return {
      briefings: validation.briefings,
      latency_ms: latencyMs,
      tokens_generated: data.usage?.completion_tokens || 0,
      model: data.model || this.model,
      validation,
    };
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
 * Check if Elle (LFBM) is configured
 */
export function isElleConfigured(): boolean {
  return !!process.env.LFBM_ENDPOINT;
}

/**
 * Example integration with intel-briefing route:
 *
 * // In intel-briefing/route.ts:
 * import { getLFBMClient } from '@/lib/inference/LFBMClient';
 *
 * const elle = getLFBMClient();
 * const briefings = await elle.generateFromMetrics(nationData, gdeltSignals, categoryRisks);
 * return NextResponse.json({ briefings, metadata: { source: 'elle' } });
 */
