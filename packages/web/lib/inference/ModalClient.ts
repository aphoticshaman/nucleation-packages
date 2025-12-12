/**
 * MODAL CLIENT - DEPRECATED
 *
 * This client was designed for Modal.com Phi-2 deployment.
 * LatticeForge now uses RunPod Serverless with Elle-72B-Ultimate.
 *
 * For new code, use:
 * - ModelRouter for automatic tier selection (workhorse vs elle)
 * - LFBMClient for direct Elle-72B access
 *
 * @deprecated Use ModelRouter or LFBMClient instead
 */

import { infer, type RoutingContext } from './ModelRouter';

// Modal endpoint URL - DEPRECATED, kept for backwards compatibility
const MODAL_ENDPOINT = process.env.MODAL_INFERENCE_URL;

interface ModalGenerateRequest {
  action: 'generate';
  prompt: string;
  max_new_tokens?: number;
  temperature?: number;
  system_prompt?: string;
}

interface ModalAnalyzeRequest {
  action: 'analyze_news';
  news_items: Array<{
    title: string;
    description?: string;
    link?: string;
    pubDate?: string;
    domain: string;
  }>;
  domain: string;
}

interface ModalBriefingRequest {
  action: 'generate_briefing';
  preset: 'global' | 'nato' | 'brics' | 'conflict';
  metrics: Record<string, { riskLevel: number; trend: string }>;
  alerts: Array<{
    category: string;
    severity: string;
    summary: string;
  }>;
}

type ModalRequest = ModalGenerateRequest | ModalAnalyzeRequest | ModalBriefingRequest;

interface ModalResponse {
  text?: string;
  analysis?: string;
  briefings?: Record<string, string>;
  tokens_generated?: number;
  latency_ms: number;
  model: string;
  error?: string;
}

/**
 * Call Modal inference endpoint (DEPRECATED)
 * Routes to ModelRouter instead
 */
async function callModal(request: ModalRequest): Promise<ModalResponse> {
  // If Modal endpoint is configured, use it for backwards compatibility
  if (MODAL_ENDPOINT) {
    const response = await fetch(MODAL_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Modal request failed: ${response.status}`);
    }

    return response.json();
  }

  // Otherwise route through ModelRouter
  if (request.action === 'generate') {
    const result = await infer(
      {
        userMessage: request.prompt,
        systemPrompt: request.system_prompt,
        maxTokens: request.max_new_tokens || 512,
        temperature: request.temperature || 0.7,
      },
      {
        taskType: 'unknown',
        isUserFacing: false,
      }
    );

    return {
      text: result.content,
      latency_ms: result.latencyMs,
      model: result.model,
    };
  }

  throw new Error('Unsupported Modal action - migrate to ModelRouter');
}

// ============================================
// HIGH-LEVEL API (now routes to ModelRouter)
// ============================================

/**
 * @deprecated Use ModelRouter.infer() instead
 */
export async function generateWithPhi2(
  prompt: string,
  options?: {
    maxTokens?: number;
    temperature?: number;
    systemPrompt?: string;
  }
): Promise<{ text: string; latencyMs: number }> {
  const result = await callModal({
    action: 'generate',
    prompt,
    max_new_tokens: options?.maxTokens ?? 512,
    temperature: options?.temperature ?? 0.7,
    system_prompt: options?.systemPrompt,
  });

  return {
    text: result.text || '',
    latencyMs: result.latency_ms,
  };
}

/**
 * @deprecated Use ModelRouter with taskType: 'analyze' instead
 */
export async function analyzeNewsWithPhi2(
  newsItems: Array<{
    title: string;
    description?: string;
    domain: string;
  }>,
  domain: string
): Promise<{
  llmAnalysis: string;
  latencyMs: number;
}> {
  const prompt = `Analyze these news items from ${domain}:\n\n${newsItems.map(n => `- ${n.title}: ${n.description || ''}`).join('\n')}`;

  const result = await infer(
    {
      userMessage: prompt,
      maxTokens: 512,
    },
    {
      taskType: 'analyze',
      isUserFacing: false,
    }
  );

  return {
    llmAnalysis: result.content,
    latencyMs: result.latencyMs,
  };
}

/**
 * @deprecated Use ModelRouter.generateBriefing() instead
 */
export async function generateBriefingWithPhi2(
  preset: 'global' | 'nato' | 'brics' | 'conflict',
  metrics: Record<string, { riskLevel: number; trend: string }>,
  alerts: Array<{ category: string; severity: string; summary: string }>
): Promise<{
  llmBriefings: Record<string, string>;
  latencyMs: number;
}> {
  const prompt = `Generate intel briefing for ${preset} preset. Metrics: ${JSON.stringify(metrics)}. Alerts: ${JSON.stringify(alerts)}`;

  const result = await infer(
    {
      userMessage: prompt,
      maxTokens: 1024,
      responseFormat: 'json',
    },
    {
      taskType: 'briefing',
      isUserFacing: true,
    }
  );

  // Parse JSON response
  let briefings: Record<string, string> = {};
  try {
    const jsonMatch = result.content.match(/\{[\s\S]*\}/);
    briefings = JSON.parse(jsonMatch?.[0] || '{}');
  } catch {
    briefings = { summary: result.content };
  }

  return {
    llmBriefings: briefings,
    latencyMs: result.latencyMs,
  };
}

// ============================================
// ROUTING (redirects to ModelRouter)
// ============================================

/**
 * @deprecated Use ModelRouter.routeRequest() instead
 */
export function shouldUsePhi2(context: {
  userTier?: 'consumer' | 'pro' | 'enterprise';
  taskType: 'briefing' | 'training' | 'analysis' | 'complex';
  isCronJob?: boolean;
}): boolean {
  // Map to ModelRouter tiers
  // Enterprise -> elle
  // Complex -> elle
  // Everything else -> workhorse (which is now our "phi2" equivalent)
  if (context.userTier === 'enterprise') return false;
  if (context.taskType === 'complex') return false;
  return true;
}

/**
 * @deprecated Use ModelRouter.infer() instead
 */
export async function inferDeprecated(
  prompt: string,
  context: {
    userTier?: 'consumer' | 'pro' | 'enterprise';
    taskType: 'briefing' | 'training' | 'analysis' | 'complex';
    isCronJob?: boolean;
    maxTokens?: number;
    temperature?: number;
    systemPrompt?: string;
  }
): Promise<{
  text: string;
  model: 'workhorse' | 'elle';
  latencyMs: number;
}> {
  // Map old context to new RoutingContext
  const routingContext: RoutingContext = {
    taskType: context.taskType === 'complex' ? 'analyze' :
              context.taskType === 'briefing' ? 'briefing' :
              context.taskType === 'training' ? 'training_gen' : 'unknown',
    userTier: context.userTier === 'consumer' ? 'free' :
              context.userTier === 'pro' ? 'pro' :
              context.userTier === 'enterprise' ? 'enterprise' : 'free',
    isCronJob: context.isCronJob,
    isUserFacing: context.taskType === 'briefing',
  };

  const result = await infer(
    {
      userMessage: prompt,
      systemPrompt: context.systemPrompt,
      maxTokens: context.maxTokens || 512,
      temperature: context.temperature || 0.7,
    },
    routingContext
  );

  return {
    text: result.content,
    model: result.tier,
    latencyMs: result.latencyMs,
  };
}

// ============================================
// COST TRACKING (updated for RunPod)
// ============================================

interface InferenceCost {
  workhorse: number;  // cents - Qwen-7B on RunPod
  elle: number;       // cents - Elle-72B on RunPod
}

const COST_PER_INFERENCE: InferenceCost = {
  workhorse: 0.03,  // ~$0.0003
  elle: 0.1,        // ~$0.001
};

/**
 * Estimate monthly cost based on inference volume
 */
export function estimateMonthlyCost(
  inferencesPerDay: number,
  workhorsePercentage: number = 75
): {
  workhorseCost: number;
  elleCost: number;
  totalCost: number;
  savingsVsExternal: number;
} {
  const monthlyInferences = inferencesPerDay * 30;
  const workhorseInferences = monthlyInferences * (workhorsePercentage / 100);
  const elleInferences = monthlyInferences * ((100 - workhorsePercentage) / 100);

  const workhorseCost = (workhorseInferences * COST_PER_INFERENCE.workhorse) / 100;
  const elleCost = (elleInferences * COST_PER_INFERENCE.elle) / 100;
  const totalCost = workhorseCost + elleCost;

  // External cost estimate (using external APIs)
  const externalCostPerInference = 1.0; // ~$0.01 per inference average
  const externalCost = (monthlyInferences * externalCostPerInference) / 100;
  const savingsVsExternal = externalCost - totalCost;

  return {
    workhorseCost: Math.round(workhorseCost * 100) / 100,
    elleCost: Math.round(elleCost * 100) / 100,
    totalCost: Math.round(totalCost * 100) / 100,
    savingsVsExternal: Math.round(savingsVsExternal * 100) / 100,
  };
}

// Example: 1000 inferences/day, 75% workhorse, 25% elle
// workhorseCost: $0.68, elleCost: $0.75, total: $1.43
// vs external APIs: ~$30/mo → savings: $28.57/mo
