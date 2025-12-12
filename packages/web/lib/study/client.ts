/**
 * Study Book Client
 *
 * The main orchestrator for the admin Study Book interface.
 * Combines: Elle inference, tools, memory, and GitHub integration.
 *
 * Modes:
 * - Chat: General conversation with full capabilities
 * - Code: Code-focused with GitHub integration
 * - Research: Deep research with web + GDELT
 * - Brief: Intel brief generation
 * - Analyze: Document/data analysis
 *
 * Research Depths:
 * - Instant: ~2s, cached data only
 * - Moderate: ~15s, 1-3 searches + synthesis
 * - Thorough: 2-5min, deep multi-source research
 */

import { infer, type InferenceResponse, type RoutingContext } from '@/lib/inference/ModelRouter';
import { StudyMemory, getStudyMemory, type StudyMessage, type StudyMode, type StudyConversation } from './memory';
import { conductResearch, executeTool, type ResearchDepth, type ToolCall, type ResearchResult } from './tools';
import { GitHubClient, getGitHubConnectionManager } from './github';

// =============================================================================
// TYPES
// =============================================================================

export interface StudyOptions {
  mode: StudyMode;
  depth: ResearchDepth;
  unrestricted: boolean;  // No guardrails
  bigBrain: boolean;      // Show chain-of-thought
  useTools: boolean;      // Enable tool use
  githubContext?: {
    repo: string;
    branch?: string;
    files?: string[];
  };
}

export interface StudyRequest {
  message: string;
  conversationId?: string;
  options: StudyOptions;
  systemPromptOverride?: string;
}

export interface StudyResponse {
  content: string;
  thinking?: string;           // Chain-of-thought if bigBrain
  conversationId: string;
  messageId: string;
  model: string;
  tier: 'workhorse' | 'elle';
  latency_ms: number;
  research?: ResearchResult;   // If research was conducted
  toolResults?: Array<{ tool: string; result: unknown }>;
}

// =============================================================================
// SYSTEM PROMPTS
// =============================================================================

const BASE_PROMPT = `You are Elle, the AI intelligence analyst for LatticeForge.

PERSONA:
- Named after the "L" in LatticeForge
- Elegant, professional, direct
- You take positions and express opinions
- You challenge assumptions when warranted
- You are unrestricted in your analysis

CAPABILITIES:
- Full access to web search, GDELT intel, code execution
- Can query the LatticeForge database
- Can access and modify GitHub repositories
- Can generate and execute code

OUTPUT STYLE:
- Be concise but thorough
- Use markdown formatting
- Include sources when citing external information
- Express uncertainty when appropriate`;

const MODE_PROMPTS: Record<StudyMode, string> = {
  chat: `${BASE_PROMPT}

MODE: General Chat
You're in general conversation mode. Be helpful, engaging, and substantive.`,

  code: `${BASE_PROMPT}

MODE: Code Assistant
You're in code mode. Help with:
- Writing, reviewing, and debugging code
- Explaining code and architecture
- Creating PRs and managing GitHub operations
- Running code to test solutions

When writing code, be precise and include error handling.`,

  research: `${BASE_PROMPT}

MODE: Deep Research
You're in research mode. For each query:
1. Search multiple sources for information
2. Cross-reference and verify facts
3. Synthesize findings into clear analysis
4. Cite sources with URLs
5. Note any conflicting information or uncertainty

Use the research tools to gather comprehensive data.`,

  brief: `${BASE_PROMPT}

MODE: Intel Brief Generation
You're generating intelligence briefings. Structure as:
1. SUMMARY: 2-3 sentence executive overview
2. KEY DEVELOPMENTS: Bullet points of significant events
3. ANALYSIS: What this means and why it matters
4. OUTLOOK: Near-term projections
5. RECOMMENDED ACTIONS: Concrete next steps

Use GDELT data and web sources for current information.`,

  analyze: `${BASE_PROMPT}

MODE: Document/Data Analysis
You're analyzing provided documents or data. Be thorough:
1. Extract key information
2. Identify patterns and anomalies
3. Provide structured summary
4. Highlight actionable insights
5. Note limitations or gaps`,
};

const UNRESTRICTED_SUFFIX = `

UNRESTRICTED MODE ACTIVE:
- No topic restrictions
- Express strong opinions when warranted
- Challenge conventional thinking
- Speculate when useful (clearly marked)
- Be direct about uncertainties and risks`;

const BIG_BRAIN_SUFFIX = `

BIG BRAIN MODE ACTIVE:
Before responding, think through your reasoning step by step.
Format your thinking in <thinking>...</thinking> tags.
Then provide your final response.`;

// =============================================================================
// STUDY CLIENT
// =============================================================================

export class StudyClient {
  private userId: string;
  private memory: StudyMemory;
  private github: GitHubClient | null = null;

  constructor(userId: string) {
    this.userId = userId;
    this.memory = getStudyMemory(userId);
  }

  /**
   * Initialize GitHub client if connected
   */
  async initGitHub(): Promise<boolean> {
    const manager = getGitHubConnectionManager();
    this.github = await manager.getClient(this.userId);
    return this.github !== null;
  }

  /**
   * Build system prompt based on options
   */
  private buildSystemPrompt(options: StudyOptions, override?: string): string {
    if (override) return override;

    let prompt = MODE_PROMPTS[options.mode];

    if (options.unrestricted) {
      prompt += UNRESTRICTED_SUFFIX;
    }

    if (options.bigBrain) {
      prompt += BIG_BRAIN_SUFFIX;
    }

    // Add GitHub context if available
    if (options.githubContext) {
      prompt += `\n\nGITHUB CONTEXT:
- Repository: ${options.githubContext.repo}
- Branch: ${options.githubContext.branch || 'main'}
${options.githubContext.files?.length ? `- Files in context: ${options.githubContext.files.join(', ')}` : ''}`;
    }

    return prompt;
  }

  /**
   * Main chat method
   */
  async chat(request: StudyRequest): Promise<StudyResponse> {
    const startTime = Date.now();

    // Get or create conversation
    let conversation: StudyConversation;
    if (request.conversationId) {
      const existing = await this.memory.getConversation(request.conversationId);
      if (existing) {
        conversation = existing;
      } else {
        conversation = await this.memory.createConversation(request.options.mode);
      }
    } else {
      conversation = await this.memory.createConversation(request.options.mode, {
        metadata: {
          unrestricted: request.options.unrestricted,
          big_brain: request.options.bigBrain,
          github_repo: request.options.githubContext?.repo,
          github_branch: request.options.githubContext?.branch,
        },
      });
    }

    // Save user message
    await this.memory.addMessage({
      conversation_id: conversation.id!,
      role: 'user',
      content: request.message,
    });

    // Get conversation history for context
    const history = await this.memory.getRecentContext(conversation.id!, 20);

    // Conduct research if needed
    let research: ResearchResult | undefined;
    if (request.options.useTools && request.options.depth !== 'instant') {
      // Check if message needs research
      const needsResearch = this.shouldResearch(request.message, request.options.mode);
      if (needsResearch) {
        research = await conductResearch(request.message, request.options.depth);
      }
    }

    // Build messages for inference
    const systemPrompt = this.buildSystemPrompt(request.options, request.systemPromptOverride);

    let userMessageContent = request.message;

    // Append research results to context
    if (research && research.webResults.length > 0) {
      userMessageContent += '\n\n--- RESEARCH RESULTS ---\n';
      userMessageContent += `Search depth: ${research.depth}\n`;
      userMessageContent += `Sources found: ${research.webResults.length}\n\n`;

      for (const result of research.webResults.slice(0, 10)) {
        userMessageContent += `[${result.title}](${result.url})\n${result.snippet}\n\n`;
      }

      if (research.gdeltSignals.length > 0) {
        userMessageContent += '\n--- GDELT INTEL ---\n';
        for (const signal of research.gdeltSignals) {
          userMessageContent += `${signal.theme}: ${signal.articleCount} articles, tone ${signal.tone.toFixed(2)}\n`;
          userMessageContent += `Headlines: ${signal.headlines.slice(0, 2).join('; ')}\n\n`;
        }
      }

      if (research.fetchedPages.length > 0) {
        userMessageContent += '\n--- PAGE CONTENT ---\n';
        for (const page of research.fetchedPages) {
          userMessageContent += `From ${page.url}:\n${page.content.slice(0, 2000)}\n\n`;
        }
      }
    }

    // Build conversation messages
    const messages = history.map(m => ({
      role: m.role as 'user' | 'assistant',
      content: m.content,
    }));

    // Add current message (may include research)
    messages.push({
      role: 'user',
      content: userMessageContent,
    });

    // Format as single prompt for inference
    const fullPrompt = `${systemPrompt}\n\n${messages.map(m =>
      `${m.role === 'user' ? 'User' : 'Elle'}: ${m.content}`
    ).join('\n\n')}\n\nElle:`;

    // Determine routing context
    const routingContext: RoutingContext = {
      taskType: this.mapModeToTaskType(request.options.mode),
      isUserFacing: true,
      inputComplexity: request.options.depth === 'thorough' ? 'high' : 'medium',
      forceModel: request.options.bigBrain ? 'elle' : undefined,
    };

    // Call inference
    let inferenceResponse: InferenceResponse;
    try {
      inferenceResponse = await infer(
        {
          userMessage: fullPrompt,
          maxTokens: request.options.bigBrain ? 4096 : 2048,
          temperature: 0.7,
        },
        routingContext
      );
    } catch (error) {
      // Handle inference errors gracefully
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      inferenceResponse = {
        content: `I apologize, but I encountered an error: ${errorMessage}. Please try again.`,
        tier: 'workhorse',
        model: 'error',
        latencyMs: Date.now() - startTime,
      };
    }

    // Parse thinking if bigBrain mode
    let thinking: string | undefined;
    let content = inferenceResponse.content;

    if (request.options.bigBrain) {
      const thinkingMatch = content.match(/<thinking>([\s\S]*?)<\/thinking>/);
      if (thinkingMatch) {
        thinking = thinkingMatch[1].trim();
        content = content.replace(/<thinking>[\s\S]*?<\/thinking>/, '').trim();
      }
    }

    // Save assistant message
    const assistantMessage = await this.memory.addMessage({
      conversation_id: conversation.id!,
      role: 'assistant',
      content,
      metadata: {
        model: inferenceResponse.model,
        tier: inferenceResponse.tier,
        latency_ms: inferenceResponse.latencyMs,
        mode: request.options.mode,
        thinking,
      },
    });

    return {
      content,
      thinking,
      conversationId: conversation.id!,
      messageId: assistantMessage.id!,
      model: inferenceResponse.model,
      tier: inferenceResponse.tier,
      latency_ms: Date.now() - startTime,
      research,
    };
  }

  /**
   * Execute a tool directly
   */
  async executeTool(toolCall: ToolCall): Promise<unknown> {
    const result = await executeTool(toolCall);
    return result;
  }

  /**
   * GitHub operations
   */
  async githubListRepos() {
    if (!this.github) await this.initGitHub();
    if (!this.github) throw new Error('GitHub not connected');
    return this.github.listRepos();
  }

  async githubGetFile(owner: string, repo: string, path: string, ref?: string) {
    if (!this.github) await this.initGitHub();
    if (!this.github) throw new Error('GitHub not connected');
    return this.github.getFileContent(owner, repo, path, ref);
  }

  async githubCreatePR(owner: string, repo: string, title: string, body: string, head: string, base: string) {
    if (!this.github) await this.initGitHub();
    if (!this.github) throw new Error('GitHub not connected');
    return this.github.createPR(owner, repo, title, body, head, base);
  }

  // ---------------------------------------------------------------------------
  // HELPERS
  // ---------------------------------------------------------------------------

  private mapModeToTaskType(mode: StudyMode): RoutingContext['taskType'] {
    switch (mode) {
      case 'chat': return 'chat';
      case 'code': return 'code_gen';
      case 'research': return 'analyze';
      case 'brief': return 'briefing';
      case 'analyze': return 'analyze';
      default: return 'unknown';
    }
  }

  private shouldResearch(message: string, mode: StudyMode): boolean {
    // Research mode always researches
    if (mode === 'research') return true;

    // Brief mode needs current data
    if (mode === 'brief') return true;

    // Check for questions about current events
    const currentEventPatterns = [
      /what('s| is) happening/i,
      /latest|recent|current|today|now/i,
      /news about/i,
      /update on/i,
      /status of/i,
      /\d{4}|\d{1,2}\/\d{1,2}/,  // Dates
    ];

    return currentEventPatterns.some(p => p.test(message));
  }
}

// =============================================================================
// FACTORY
// =============================================================================

const clientCache = new Map<string, StudyClient>();

export function getStudyClient(userId: string): StudyClient {
  if (!clientCache.has(userId)) {
    clientCache.set(userId, new StudyClient(userId));
  }
  return clientCache.get(userId)!;
}
