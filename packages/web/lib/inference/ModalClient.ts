/**
 * INFERENCE CLIENT - Routes all inference to Elle (LFBM/Qwen)
 *
 * Elle is the ONLY inference system used by LatticeForge.
 * Powered by self-hosted Qwen2.5-3B-Instruct on RunPod.
 * No external LLM API dependencies.
 */

import { getLFBMClient } from './LFBMClient';

/**
 * Unified inference function - all requests go to Elle (LFBM)
 */
export async function infer(
  prompt: string,
  context: {
    userTier?: 'consumer' | 'pro' | 'enterprise';
    taskType?: 'briefing' | 'training' | 'analysis' | 'complex';
    isCronJob?: boolean;
    maxTokens?: number;
    temperature?: number;
    systemPrompt?: string;
  }
): Promise<{
  text: string;
  model: 'elle';
  latencyMs: number;
}> {
  const lfbm = getLFBMClient();

  if (!lfbm.isConfigured()) {
    throw new Error('LFBM_ENDPOINT not configured. Elle requires RunPod endpoint.');
  }

  const startTime = Date.now();
  const response = await lfbm.generateRaw({
    systemPrompt: context.systemPrompt,
    userMessage: prompt,
    max_tokens: context.maxTokens,
    temperature: context.temperature,
  });

  return {
    text: response,
    model: 'elle',
    latencyMs: Date.now() - startTime,
  };
}

/**
 * Generate text with Elle
 */
export async function generateWithElle(
  prompt: string,
  options?: {
    maxTokens?: number;
    temperature?: number;
    systemPrompt?: string;
  }
): Promise<{ text: string; latencyMs: number }> {
  const result = await infer(prompt, {
    maxTokens: options?.maxTokens ?? 512,
    temperature: options?.temperature ?? 0.7,
    systemPrompt: options?.systemPrompt,
  });

  return {
    text: result.text,
    latencyMs: result.latencyMs,
  };
}

/**
 * Analyze news items with Elle
 */
export async function analyzeNewsWithElle(
  newsItems: Array<{
    title: string;
    description?: string;
    domain: string;
  }>,
  domain: string
): Promise<{
  analysis: string;
  latencyMs: number;
}> {
  const prompt = `Analyze these ${domain} news items:\n\n${newsItems.map(n => `- ${n.title}${n.description ? `: ${n.description}` : ''}`).join('\n')}`;

  const result = await infer(prompt, {
    taskType: 'analysis',
    systemPrompt: 'You are Elle, LatticeForge\'s intelligence analyst. Provide concise analysis.',
  });

  return {
    analysis: result.text,
    latencyMs: result.latencyMs,
  };
}

/**
 * Generate briefing with Elle
 */
export async function generateBriefingWithElle(
  preset: 'global' | 'nato' | 'brics' | 'conflict',
  metrics: Record<string, { riskLevel: number; trend: string }>,
  alerts: Array<{ category: string; severity: string; summary: string }>
): Promise<{
  briefings: Record<string, string>;
  latencyMs: number;
}> {
  const lfbm = getLFBMClient();

  const response = await lfbm.generateBriefing({
    nations: [],
    signals: Object.fromEntries(
      Object.entries(metrics).map(([k, v]) => [k, v.riskLevel])
    ),
    categories: Object.fromEntries(
      alerts.map(a => [a.category, a.severity === 'high' ? 80 : a.severity === 'medium' ? 50 : 20])
    ),
  });

  return {
    briefings: response.briefings,
    latencyMs: response.latency_ms,
  };
}

// ============================================
// COST TRACKING
// ============================================

/**
 * Estimate monthly cost - Elle only (self-hosted Qwen on RunPod)
 */
export function estimateMonthlyCost(inferencesPerDay: number): {
  elleCost: number;
  totalCost: number;
} {
  const monthlyInferences = inferencesPerDay * 30;
  // ~$0.002 per inference on RunPod serverless
  const elleCost = (monthlyInferences * 0.2) / 100;

  return {
    elleCost: Math.round(elleCost * 100) / 100,
    totalCost: Math.round(elleCost * 100) / 100,
  };
}
