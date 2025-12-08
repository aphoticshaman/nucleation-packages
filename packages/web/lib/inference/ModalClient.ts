/**
 * MODAL CLIENT - Call your fine-tuned Phi-2 model
 *
 * Architecture:
 * 1. Raw Data (APIs, RSS, GDELT) → Ingested automatically
 * 2. LLM Processing (Phi-2) → Generates ONE interpretation
 * 3. Your Expert Analysis → Added on top (the value-add)
 * 4. Customer Deliverable → Big picture synthesis
 *
 * Phi-2 replaces Anthropic for routine processing:
 * - Cost: ~$0.001 vs $0.01+ per inference
 * - Runs every 10 minutes via cron
 * - Output is ONE of THREE inputs (raw, LLM, expert)
 */

// Modal endpoint URL - set in Vercel env vars
const MODAL_ENDPOINT = process.env.MODAL_INFERENCE_URL;

// PRODUCTION-ONLY: Block Anthropic API calls in non-production unless explicitly enabled
function isAnthropicAllowed(): boolean {
  const env = process.env.VERCEL_ENV || process.env.NODE_ENV;
  if (env === 'production') return true;
  if (process.env.ALLOW_ANTHROPIC_IN_DEV === 'true') return true;
  return false;
}

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
 * Call Modal inference endpoint
 */
async function callModal(request: ModalRequest): Promise<ModalResponse> {
  if (!MODAL_ENDPOINT) {
    throw new Error('MODAL_INFERENCE_URL not configured');
  }

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

// ============================================
// HIGH-LEVEL API
// ============================================

/**
 * Generate text with Phi-2 (general purpose)
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
 * Analyze news items with Phi-2 (for training data generation)
 *
 * Output is the LLM's interpretation - ONE of your three inputs
 */
export async function analyzeNewsWithPhi2(
  newsItems: Array<{
    title: string;
    description?: string;
    domain: string;
  }>,
  domain: string
): Promise<{
  llmAnalysis: string;  // LLM's take (one input)
  latencyMs: number;
}> {
  const result = await callModal({
    action: 'analyze_news',
    news_items: newsItems,
    domain,
  });

  return {
    llmAnalysis: result.analysis || '',
    latencyMs: result.latency_ms,
  };
}

/**
 * Generate briefing with Phi-2 (for intel-briefing endpoint)
 *
 * Output is the LLM's narrative - you add expert layer on top
 */
export async function generateBriefingWithPhi2(
  preset: 'global' | 'nato' | 'brics' | 'conflict',
  metrics: Record<string, { riskLevel: number; trend: string }>,
  alerts: Array<{ category: string; severity: string; summary: string }>
): Promise<{
  llmBriefings: Record<string, string>;  // LLM's briefings (one input)
  latencyMs: number;
}> {
  const result = await callModal({
    action: 'generate_briefing',
    preset,
    metrics,
    alerts,
  });

  return {
    llmBriefings: result.briefings || {},
    latencyMs: result.latency_ms,
  };
}

// ============================================
// FALLBACK TO ANTHROPIC
// ============================================

/**
 * Determine which model to use
 *
 * Use Phi-2 for:
 * - Routine 10-minute summaries
 * - Training data generation
 * - Standard briefings
 *
 * Use Anthropic for:
 * - Enterprise customers (tier === 'enterprise')
 * - Complex multi-step reasoning
 * - Novel/unprecedented situations
 * - When Phi-2 fails or is unavailable
 */
export function shouldUsePhi2(context: {
  userTier?: 'consumer' | 'pro' | 'enterprise';
  taskType: 'briefing' | 'training' | 'analysis' | 'complex';
  isCronJob?: boolean;
}): boolean {
  // Always use Phi-2 for cron jobs (cost savings)
  if (context.isCronJob) {
    return true;
  }

  // Enterprise always gets Anthropic (premium experience)
  if (context.userTier === 'enterprise') {
    return false;
  }

  // Complex tasks go to Anthropic
  if (context.taskType === 'complex') {
    return false;
  }

  // Everything else uses Phi-2
  return true;
}

/**
 * Unified inference function - routes to correct model
 */
export async function infer(
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
  model: 'phi2' | 'anthropic';
  latencyMs: number;
}> {
  const usePhi2 = shouldUsePhi2(context);

  if (usePhi2 && MODAL_ENDPOINT) {
    try {
      const result = await generateWithPhi2(prompt, {
        maxTokens: context.maxTokens,
        temperature: context.temperature,
        systemPrompt: context.systemPrompt,
      });

      return {
        text: result.text,
        model: 'phi2',
        latencyMs: result.latencyMs,
      };
    } catch (error) {
      console.error('Phi-2 failed, falling back to Anthropic:', error);
      // Fall through to Anthropic
    }
  }

  // Fallback to Anthropic - BLOCKED in non-production
  if (!isAnthropicAllowed()) {
    throw new Error(`Anthropic API blocked in non-production (${process.env.VERCEL_ENV || process.env.NODE_ENV}). Set ALLOW_ANTHROPIC_IN_DEV=true to enable.`);
  }

  const Anthropic = (await import('@anthropic-ai/sdk')).default;
  const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY!,
  });

  const startTime = Date.now();

  const message = await anthropic.messages.create({
    model: 'claude-haiku-4-5-20251001', // Haiku 4.5 - fastest, best value
    max_tokens: context.maxTokens || 512,
    system: context.systemPrompt || 'You are a helpful assistant.',
    messages: [{ role: 'user', content: prompt }],
  });

  const text = message.content[0].type === 'text' ? message.content[0].text : '';

  return {
    text,
    model: 'anthropic',
    latencyMs: Date.now() - startTime,
  };
}

// ============================================
// COST TRACKING
// ============================================

interface InferenceCost {
  phi2: number;    // cents
  anthropic: number; // cents
}

const COST_PER_INFERENCE: InferenceCost = {
  phi2: 0.1,      // ~$0.001 (Modal T4 GPU time)
  anthropic: 1.0, // ~$0.01 (Haiku) to $0.10 (Sonnet)
};

/**
 * Estimate monthly cost based on inference volume
 */
export function estimateMonthlyCost(
  inferencesPerDay: number,
  phi2Percentage: number = 90
): {
  phi2Cost: number;
  anthropicCost: number;
  totalCost: number;
  savingsVsAllAnthropic: number;
} {
  const monthlyInferences = inferencesPerDay * 30;
  const phi2Inferences = monthlyInferences * (phi2Percentage / 100);
  const anthropicInferences = monthlyInferences * ((100 - phi2Percentage) / 100);

  const phi2Cost = (phi2Inferences * COST_PER_INFERENCE.phi2) / 100;
  const anthropicCost = (anthropicInferences * COST_PER_INFERENCE.anthropic) / 100;
  const totalCost = phi2Cost + anthropicCost;

  const allAnthropicCost = (monthlyInferences * COST_PER_INFERENCE.anthropic) / 100;
  const savingsVsAllAnthropic = allAnthropicCost - totalCost;

  return {
    phi2Cost: Math.round(phi2Cost * 100) / 100,
    anthropicCost: Math.round(anthropicCost * 100) / 100,
    totalCost: Math.round(totalCost * 100) / 100,
    savingsVsAllAnthropic: Math.round(savingsVsAllAnthropic * 100) / 100,
  };
}

// Example: 1000 inferences/day, 90% Phi-2
// phi2Cost: $2.70, anthropicCost: $3.00, total: $5.70
// vs all Anthropic: $30/mo → savings: $24.30/mo
