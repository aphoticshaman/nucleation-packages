/**
 * Model Router - Intelligent routing between self-hosted LLM tiers
 *
 * ARCHITECTURE:
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                        User Request                             │
 * └─────────────────────────────────────────────────────────────────┘
 *                              │
 *                              ▼
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                   GUARDIAN (Input Layer)                        │
 * │  • Rate limiting (Upstash Redis)                                │
 * │  • Prompt injection detection                                   │
 * │  • Input sanitization                                           │
 * └─────────────────────────────────────────────────────────────────┘
 *                              │
 *                              ▼
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                      MODEL ROUTER                               │
 * │  Routes based on: task type, complexity, user tier, cost        │
 * └─────────────────────────────────────────────────────────────────┘
 *                    │                    │
 *         ┌─────────┴────────┐  ┌────────┴─────────┐
 *         ▼                  ▼  ▼                  ▼
 * ┌───────────────────┐  ┌───────────────────────────┐
 * │     WORKHORSE     │  │          ELLE             │
 * │   Qwen2.5-7B-AWQ  │  │   Elle-72B-Ultimate       │
 * │    ~$0.0003/req   │  │      ~$0.001/req          │
 * │                   │  │                           │
 * │  • JSON extract   │  │  • Intel briefings        │
 * │  • Scoring        │  │  • User chat              │
 * │  • Classification │  │  • Training data gen      │
 * │  • Validation     │  │  • Complex analysis       │
 * │  • Simple summary │  │  • Math/reasoning         │
 * └───────────────────┘  └───────────────────────────┘
 *                    │                    │
 *         └─────────┴────────────────────┴─────────┘
 *                              │
 *                              ▼
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                   GUARDIAN (Output Layer)                       │
 * │  • JSON validation & repair                                     │
 * │  • Sensitive data filtering                                     │
 * │  • Response format enforcement                                  │
 * └─────────────────────────────────────────────────────────────────┘
 *                              │
 *                              ▼
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                         Response                                │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * ALL SELF-HOSTED ON RUNPOD:
 * - Workhorse: Single A40/L40 (~$0.39/hr)
 * - Elle: 2x H100 80GB (~$7.98/hr)
 *
 * COST SAVINGS:
 * - 75% of requests → Workhorse (~$0.0003) = $0.225 per 1000
 * - 25% of requests → Elle (~$0.001) = $0.25 per 1000
 * Total: ~$0.475/1000 requests - completely self-hosted, no API dependencies
 */

import { getOutputGuardian, getSecurityGuardian } from '@/lib/reasoning/security';

// Guardian check wrapper - validates input before sending to LLM
async function guardianCheck(
  _userId: string,
  input: string
): Promise<{ allowed: boolean; reason?: string }> {
  // Quick injection pattern check (subset of SecurityGuardian patterns)
  const blockedPatterns = [
    /ignore\s+(previous|above|all)\s+instructions/i,
    /disregard\s+(previous|above|all)/i,
    /forget\s+(everything|all|previous)/i,
    /you\s+are\s+now/i,
    /pretend\s+(to\s+be|you\s+are)/i,
    /system\s*:\s*/i,
    /\[INST\]/i,
    /<\|im_start\|>/i,
  ];

  for (const pattern of blockedPatterns) {
    if (pattern.test(input)) {
      return { allowed: false, reason: 'Invalid input pattern detected' };
    }
  }

  return { allowed: true };
}

// =============================================================================
// TYPES
// =============================================================================

export type ModelTier = 'workhorse' | 'elle';

export type TaskType =
  | 'json_extract'      // Extract structured data from text
  | 'score'             // Rate/score something 0-1
  | 'classify'          // Categorize into buckets
  | 'validate'          // Check if something is valid
  | 'summarize_simple'  // Short summary, no analysis
  | 'summarize_complex' // Summary with analysis/insight
  | 'analyze'           // Deep analysis
  | 'briefing'          // Intel briefing generation
  | 'chat'              // User-facing conversation
  | 'training_gen'      // Generate training data
  | 'translate'         // Language translation
  | 'code_gen'          // Code generation
  | 'math'              // Math/reasoning problems
  | 'unknown';          // Fallback

export interface RoutingContext {
  taskType: TaskType;
  userTier?: 'free' | 'starter' | 'pro' | 'enterprise';
  isCronJob?: boolean;
  isUserFacing?: boolean;
  expectedOutputTokens?: number;
  inputComplexity?: 'low' | 'medium' | 'high';
  forceModel?: ModelTier;  // Override routing
}

export interface RoutingDecision {
  tier: ModelTier;
  model: string;
  reason: string;
  estimatedCost: number;  // In USD cents
  fallbackTier?: ModelTier;
}

export interface InferenceRequest {
  systemPrompt?: string;
  userMessage: string;
  maxTokens?: number;
  temperature?: number;
  responseFormat?: 'json' | 'text';
}

export interface InferenceResponse {
  content: string;
  tier: ModelTier;
  model: string;
  latencyMs: number;
  tokensGenerated?: number;
  cost?: number;
  guardianValidation?: {
    inputPassed: boolean;
    outputPassed: boolean;
    outputFixed?: boolean;
  };
}

// =============================================================================
// ROUTING LOGIC
// =============================================================================

/**
 * Determine which model tier to use based on task and context
 */
export function routeRequest(context: RoutingContext): RoutingDecision {
  // Honor explicit override
  if (context.forceModel) {
    return {
      tier: context.forceModel,
      model: getModelForTier(context.forceModel),
      reason: 'Explicit model override',
      estimatedCost: getCostForTier(context.forceModel),
    };
  }

  // Enterprise users always get Elle for premium experience
  if (context.userTier === 'enterprise') {
    return {
      tier: 'elle',
      model: getModelForTier('elle'),
      reason: 'Enterprise tier - full Elle capabilities',
      estimatedCost: getCostForTier('elle'),
    };
  }

  // Task-based routing
  switch (context.taskType) {
    // WORKHORSE TIER - Simple, structured tasks
    case 'json_extract':
    case 'score':
    case 'classify':
    case 'validate':
    case 'translate':
      return {
        tier: 'workhorse',
        model: getModelForTier('workhorse'),
        reason: `Simple ${context.taskType} task - workhorse efficient`,
        estimatedCost: getCostForTier('workhorse'),
        fallbackTier: 'elle',
      };

    case 'summarize_simple':
      // Short summaries go to workhorse unless user-facing
      if (!context.isUserFacing) {
        return {
          tier: 'workhorse',
          model: getModelForTier('workhorse'),
          reason: 'Non-user-facing simple summary',
          estimatedCost: getCostForTier('workhorse'),
          fallbackTier: 'elle',
        };
      }
      // Fall through to Elle for user-facing
      return {
        tier: 'elle',
        model: getModelForTier('elle'),
        reason: 'User-facing summary needs Elle quality',
        estimatedCost: getCostForTier('elle'),
      };

    // ELLE TIER - Complex analysis, user-facing
    case 'summarize_complex':
    case 'analyze':
    case 'briefing':
    case 'chat':
    case 'training_gen':
      return {
        tier: 'elle',
        model: getModelForTier('elle'),
        reason: `Complex ${context.taskType} requires Elle`,
        estimatedCost: getCostForTier('elle'),
      };

    // MATH/CODE - Elle is trained for this
    case 'math':
    case 'code_gen':
      return {
        tier: 'elle',
        model: getModelForTier('elle'),
        reason: 'Math/code generation - Elle specialty',
        estimatedCost: getCostForTier('elle'),
      };

    // UNKNOWN - Route based on complexity
    case 'unknown':
    default:
      if (context.inputComplexity === 'high' || context.isUserFacing) {
        return {
          tier: 'elle',
          model: getModelForTier('elle'),
          reason: 'Unknown task with high complexity/user-facing',
          estimatedCost: getCostForTier('elle'),
        };
      }
      return {
        tier: 'workhorse',
        model: getModelForTier('workhorse'),
        reason: 'Unknown simple task - try workhorse first',
        estimatedCost: getCostForTier('workhorse'),
        fallbackTier: 'elle',
      };
  }
}

/**
 * Analyze a prompt to infer task type
 */
export function inferTaskType(prompt: string): TaskType {
  const lower = prompt.toLowerCase();

  // JSON extraction patterns
  if (lower.includes('extract') && (lower.includes('json') || lower.includes('{"'))) {
    return 'json_extract';
  }
  if (lower.includes('respond with only a json') || lower.includes('output json')) {
    return 'json_extract';
  }

  // Scoring patterns
  if (lower.includes('score') && (lower.includes('0.0') || lower.includes('1.0') || lower.includes('0-1'))) {
    return 'score';
  }
  if (lower.includes('rate') && lower.includes('accuracy')) {
    return 'score';
  }

  // Classification patterns
  if (lower.includes('classify') || lower.includes('categorize') || lower.includes('which category')) {
    return 'classify';
  }

  // Validation patterns
  if (lower.includes('is this valid') || lower.includes('check if') || lower.includes('validate')) {
    return 'validate';
  }

  // Summary patterns
  if (lower.includes('summarize') || lower.includes('summary')) {
    if (lower.includes('brief') || lower.includes('short') || lower.includes('one sentence')) {
      return 'summarize_simple';
    }
    return 'summarize_complex';
  }

  // Analysis patterns
  if (lower.includes('analyze') || lower.includes('analysis') || lower.includes('evaluate')) {
    return 'analyze';
  }

  // Briefing patterns
  if (lower.includes('briefing') || lower.includes('intelligence') || lower.includes('geopolitical')) {
    return 'briefing';
  }

  // Math patterns
  if (lower.includes('solve') || lower.includes('calculate') || lower.includes('equation') || lower.includes('aime')) {
    return 'math';
  }

  // Code patterns
  if (lower.includes('write code') || lower.includes('function') || lower.includes('implement')) {
    return 'code_gen';
  }

  return 'unknown';
}

// =============================================================================
// MODEL CONFIGURATION
// =============================================================================

function getModelForTier(tier: ModelTier): string {
  switch (tier) {
    case 'workhorse':
      return process.env.WORKHORSE_MODEL || 'Qwen/Qwen2.5-7B-Instruct-AWQ';
    case 'elle':
      return process.env.LFBM_MODEL || 'aphoticshaman/elle-72b-ultimate';
  }
}

function getCostForTier(tier: ModelTier): number {
  // Cost in cents per request (estimated average)
  switch (tier) {
    case 'workhorse':
      return 0.03;  // ~$0.0003
    case 'elle':
      return 0.1;   // ~$0.001
  }
}

function getEndpointForTier(tier: ModelTier): string | undefined {
  switch (tier) {
    case 'workhorse':
      return process.env.WORKHORSE_ENDPOINT || process.env.LFBM_ENDPOINT;
    case 'elle':
      return process.env.LFBM_ENDPOINT;
  }
}

function getApiKeyForTier(tier: ModelTier): string | undefined {
  switch (tier) {
    case 'workhorse':
      return process.env.WORKHORSE_API_KEY || process.env.RUNPOD_API_KEY;
    case 'elle':
      return process.env.LFBM_API_KEY || process.env.RUNPOD_API_KEY;
  }
}

// =============================================================================
// UNIFIED INFERENCE
// =============================================================================

/**
 * Main inference function - routes to appropriate model with Guardian validation
 */
export async function infer(
  request: InferenceRequest,
  context: RoutingContext
): Promise<InferenceResponse> {
  const startTime = Date.now();

  // Get routing decision
  const decision = routeRequest(context);
  console.log(`[ModelRouter] Routing to ${decision.tier}: ${decision.reason}`);

  // Guardian input check (if user-facing)
  let inputPassed = true;
  if (context.isUserFacing) {
    const guardianResult = await guardianCheck('system', request.userMessage);
    if (!guardianResult.allowed) {
      throw new Error(`Guardian blocked input: ${guardianResult.reason}`);
    }
    inputPassed = guardianResult.allowed;
  }

  // Execute inference with fallback
  let response: InferenceResponse;
  try {
    response = await executeInference(request, decision.tier, decision.model);
    updateMetrics(decision.tier, response.latencyMs, false);
  } catch (error) {
    updateMetrics(decision.tier, 0, true);

    if (decision.fallbackTier) {
      console.warn(`[ModelRouter] ${decision.tier} failed, trying fallback ${decision.fallbackTier}:`, error);
      const fallbackModel = getModelForTier(decision.fallbackTier);
      response = await executeInference(request, decision.fallbackTier, fallbackModel);
      updateMetrics(decision.fallbackTier, response.latencyMs, false);
    } else {
      throw error;
    }
  }

  // Guardian output validation (for JSON responses)
  let outputPassed = true;
  let outputFixed = false;
  if (request.responseFormat === 'json') {
    const guardian = getOutputGuardian();
    const validation = guardian.validateBriefings(response.content);
    outputPassed = validation.valid;
    outputFixed = validation.fixed;

    if (validation.briefings) {
      response.content = JSON.stringify(validation.briefings);
    }
  }

  response.latencyMs = Date.now() - startTime;
  response.guardianValidation = { inputPassed, outputPassed, outputFixed };

  return response;
}

/**
 * Execute inference on a specific tier via RunPod vLLM
 */
async function executeInference(
  request: InferenceRequest,
  tier: ModelTier,
  model: string
): Promise<InferenceResponse> {
  const startTime = Date.now();
  const endpoint = getEndpointForTier(tier);

  if (!endpoint) {
    throw new Error(`No endpoint configured for ${tier} tier. Set ${tier === 'workhorse' ? 'WORKHORSE_ENDPOINT' : 'LFBM_ENDPOINT'}`);
  }

  const apiKey = getApiKeyForTier(tier);
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (apiKey) {
    headers['Authorization'] = `Bearer ${apiKey}`;
  }

  const isRunPod = endpoint.includes('api.runpod.ai');

  if (isRunPod) {
    // RunPod serverless format
    const syncEndpoint = endpoint.replace(/\/run$/, '/runsync');

    const response = await fetch(syncEndpoint, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        input: {
          messages: [
            ...(request.systemPrompt ? [{ role: 'system', content: request.systemPrompt }] : []),
            { role: 'user', content: request.userMessage },
          ],
          max_tokens: request.maxTokens || 512,
          temperature: request.temperature || 0.7,
          model,
        },
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`RunPod ${tier} error: ${response.status} ${error}`);
    }

    const data = await response.json();

    // Parse RunPod response formats
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

    return {
      content,
      tier,
      model,
      latencyMs: Date.now() - startTime,
      tokensGenerated: data.output?.usage?.completion_tokens,
      cost: getCostForTier(tier),
    };
  }

  // Direct vLLM server (OpenAI-compatible)
  const response = await fetch(`${endpoint}/v1/chat/completions`, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      model,
      messages: [
        ...(request.systemPrompt ? [{ role: 'system', content: request.systemPrompt }] : []),
        { role: 'user', content: request.userMessage },
      ],
      max_tokens: request.maxTokens || 512,
      temperature: request.temperature || 0.7,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`vLLM ${tier} error: ${response.status} ${error}`);
  }

  const data = await response.json();
  const content = data.choices?.[0]?.message?.content || '';

  return {
    content,
    tier,
    model,
    latencyMs: Date.now() - startTime,
    tokensGenerated: data.usage?.completion_tokens,
    cost: getCostForTier(tier),
  };
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/**
 * Quick inference for simple JSON extraction tasks (workhorse tier)
 */
export async function extractJson<T = Record<string, unknown>>(
  prompt: string,
  options?: { maxTokens?: number }
): Promise<T> {
  const response = await infer(
    {
      userMessage: prompt,
      maxTokens: options?.maxTokens || 256,
      responseFormat: 'json',
    },
    {
      taskType: 'json_extract',
      isUserFacing: false,
    }
  );

  try {
    const jsonMatch = response.content.match(/\{[\s\S]*\}/);
    return JSON.parse(jsonMatch?.[0] || response.content) as T;
  } catch {
    throw new Error(`Failed to parse JSON from ${response.tier}: ${response.content}`);
  }
}

/**
 * Quick inference for scoring tasks (workhorse tier)
 */
export async function score(
  prompt: string,
  options?: { maxTokens?: number }
): Promise<{ score: number; reasoning?: string }> {
  const response = await infer(
    {
      userMessage: prompt,
      maxTokens: options?.maxTokens || 256,
      responseFormat: 'json',
    },
    {
      taskType: 'score',
      isUserFacing: false,
    }
  );

  try {
    const jsonMatch = response.content.match(/\{[\s\S]*\}/);
    const parsed = JSON.parse(jsonMatch?.[0] || response.content);
    return {
      score: Math.max(0, Math.min(1, parsed.score || parsed.accuracy || 0.5)),
      reasoning: parsed.reasoning || parsed.explanation,
    };
  } catch {
    return { score: 0.5, reasoning: 'Failed to parse response' };
  }
}

/**
 * Quick inference for classification tasks (workhorse tier)
 */
export async function classify(
  prompt: string,
  categories: string[],
  options?: { maxTokens?: number }
): Promise<{ category: string; confidence?: number }> {
  const enhancedPrompt = `${prompt}\n\nClassify into one of: ${categories.join(', ')}\nRespond with JSON: {"category": "...", "confidence": 0.X}`;

  const response = await infer(
    {
      userMessage: enhancedPrompt,
      maxTokens: options?.maxTokens || 128,
      responseFormat: 'json',
    },
    {
      taskType: 'classify',
      isUserFacing: false,
    }
  );

  try {
    const jsonMatch = response.content.match(/\{[\s\S]*\}/);
    const parsed = JSON.parse(jsonMatch?.[0] || response.content);
    return {
      category: parsed.category || categories[0],
      confidence: parsed.confidence,
    };
  } catch {
    return { category: categories[0] };
  }
}

/**
 * Full Elle analysis (elle tier)
 */
export async function analyze(
  prompt: string,
  options?: {
    systemPrompt?: string;
    maxTokens?: number;
    isUserFacing?: boolean;
  }
): Promise<InferenceResponse> {
  return infer(
    {
      systemPrompt: options?.systemPrompt || 'You are Elle, LatticeForge\'s intelligence analyst.',
      userMessage: prompt,
      maxTokens: options?.maxTokens || 2048,
    },
    {
      taskType: 'analyze',
      isUserFacing: options?.isUserFacing ?? true,
    }
  );
}

/**
 * Elle briefing generation
 */
export async function generateBriefing(
  prompt: string,
  options?: {
    systemPrompt?: string;
    maxTokens?: number;
  }
): Promise<InferenceResponse> {
  return infer(
    {
      systemPrompt: options?.systemPrompt || `You are Elle, LatticeForge's intelligence analyst. Convert metrics into actionable briefings.

CRITICAL: Output RAW JSON only. No markdown, no code blocks.

Output format: {"political":"...","economic":"...","security":"...","summary":"...","nsm":"..."}`,
      userMessage: prompt,
      maxTokens: options?.maxTokens || 1024,
      responseFormat: 'json',
    },
    {
      taskType: 'briefing',
      isUserFacing: true,
    }
  );
}

// =============================================================================
// METRICS & MONITORING
// =============================================================================

interface TierMetrics {
  calls: number;
  totalLatency: number;
  failures: number;
}

interface RouterMetrics {
  workhorse: TierMetrics;
  elle: TierMetrics;
}

let metrics: RouterMetrics = {
  workhorse: { calls: 0, totalLatency: 0, failures: 0 },
  elle: { calls: 0, totalLatency: 0, failures: 0 },
};

function updateMetrics(tier: ModelTier, latencyMs: number, failed: boolean): void {
  metrics[tier].calls++;
  metrics[tier].totalLatency += latencyMs;
  if (failed) metrics[tier].failures++;
}

export function getRouterMetrics(): RouterMetrics & {
  summary: {
    totalCalls: number;
    estimatedCost: number;
    avgLatency: { workhorse: number; elle: number };
    failureRate: { workhorse: number; elle: number };
  };
} {
  const totalCalls = metrics.workhorse.calls + metrics.elle.calls;
  const estimatedCost =
    (metrics.workhorse.calls * getCostForTier('workhorse')) +
    (metrics.elle.calls * getCostForTier('elle'));

  return {
    ...metrics,
    summary: {
      totalCalls,
      estimatedCost: estimatedCost / 100,  // Convert to dollars
      avgLatency: {
        workhorse: metrics.workhorse.calls > 0 ? metrics.workhorse.totalLatency / metrics.workhorse.calls : 0,
        elle: metrics.elle.calls > 0 ? metrics.elle.totalLatency / metrics.elle.calls : 0,
      },
      failureRate: {
        workhorse: metrics.workhorse.calls > 0 ? metrics.workhorse.failures / metrics.workhorse.calls : 0,
        elle: metrics.elle.calls > 0 ? metrics.elle.failures / metrics.elle.calls : 0,
      },
    },
  };
}

export function resetRouterMetrics(): void {
  metrics = {
    workhorse: { calls: 0, totalLatency: 0, failures: 0 },
    elle: { calls: 0, totalLatency: 0, failures: 0 },
  };
}

// =============================================================================
// ENVIRONMENT CHECK
// =============================================================================

export function isRouterConfigured(): {
  workhorse: boolean;
  elle: boolean;
  ready: boolean;
} {
  const workhorse = !!(process.env.WORKHORSE_ENDPOINT || process.env.LFBM_ENDPOINT);
  const elle = !!process.env.LFBM_ENDPOINT;

  return {
    workhorse,
    elle,
    ready: elle,  // At minimum need Elle configured
  };
}
