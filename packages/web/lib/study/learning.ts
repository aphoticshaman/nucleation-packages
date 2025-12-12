/**
 * Elle Learning System
 *
 * Self-improvement pipeline that captures interactions and converts
 * them into training data for periodic Elle retraining.
 *
 * FLYWHEEL:
 * 1. User interacts with Elle
 * 2. System captures detailed metadata
 * 3. Admin rates responses (thumbs up/down, detailed feedback)
 * 4. High-quality interactions → training queue
 * 5. Periodic export → Axolotl fine-tuning
 * 6. Deploy improved Elle
 * 7. Repeat
 *
 * METADATA CAPTURED:
 * - Full conversation context
 * - Mode (chat/code/research/brief/analyze)
 * - Research depth and sources used
 * - Tools invoked and their results
 * - Response time and model tier
 * - User edits/corrections
 * - Follow-up questions (indicates confusion)
 * - Task success/failure signals
 */

import { createClient, SupabaseClient } from '@supabase/supabase-js';
import type { StudyMode, StudyMessage, StudyConversation } from './memory';
import type { ResearchDepth, ResearchResult, ToolResult } from './tools';

// =============================================================================
// TYPES
// =============================================================================

export interface InteractionMetadata {
  // Context
  conversation_id: string;
  message_id: string;
  user_id: string;
  mode: StudyMode;
  depth: ResearchDepth;

  // Settings
  unrestricted: boolean;
  big_brain: boolean;
  use_tools: boolean;

  // Performance
  model: string;
  tier: 'workhorse' | 'elle';
  latency_ms: number;
  tokens_in?: number;
  tokens_out?: number;

  // Research
  research_conducted: boolean;
  sources_count?: number;
  gdelt_signals_count?: number;
  pages_analyzed?: number;

  // Tools
  tools_used: string[];
  tool_results: Array<{
    tool: string;
    success: boolean;
    latency_ms: number;
  }>;

  // Quality signals (updated over time)
  user_rating?: number;           // 1-5 stars
  user_feedback?: string;         // Free-form feedback
  flagged_for_training?: boolean; // Explicitly marked as good example
  flagged_as_bad?: boolean;       // Explicitly marked as bad
  was_edited?: boolean;           // User edited the response
  had_followup?: boolean;         // User asked follow-up (may indicate confusion)
  task_completed?: boolean;       // For task-oriented interactions

  // Timestamps
  created_at: string;
  rated_at?: string;
}

export interface TrainingExample {
  id?: string;

  // Core training pair
  system_prompt: string;
  user_input: string;
  assistant_output: string;

  // Categorization
  domain: string;           // chat, code, research, brief, analyze
  task_type: string;        // question_answer, code_generation, analysis, etc.
  difficulty: 'easy' | 'medium' | 'hard';

  // Quality
  quality_score: number;    // 0-1, computed from various signals
  human_rating?: number;    // 1-5 if rated
  source: 'study_book' | 'synthetic' | 'curated';

  // Metadata for filtering
  metadata: {
    conversation_id?: string;
    message_id?: string;
    mode?: StudyMode;
    model_used?: string;
    latency_ms?: number;
    research_sources?: number;
    tools_used?: string[];
  };

  // Status
  status: 'pending' | 'approved' | 'rejected' | 'exported';
  reviewed_at?: string;
  exported_at?: string;

  created_at?: string;
}

export interface LearningStats {
  total_interactions: number;
  rated_interactions: number;
  positive_ratings: number;
  negative_ratings: number;
  training_examples_pending: number;
  training_examples_approved: number;
  training_examples_exported: number;
  avg_response_time_ms: number;
  elle_usage_percent: number;
  workhorse_usage_percent: number;
  by_mode: Record<StudyMode, number>;
  by_depth: Record<ResearchDepth, number>;
  quality_over_time: Array<{ date: string; avg_rating: number; count: number }>;
}

// =============================================================================
// LEARNING ENGINE
// =============================================================================

export class LearningEngine {
  private supabase: SupabaseClient;

  constructor(supabaseUrl?: string, supabaseKey?: string) {
    this.supabase = createClient(
      supabaseUrl || process.env.NEXT_PUBLIC_SUPABASE_URL!,
      supabaseKey || process.env.SUPABASE_SERVICE_ROLE_KEY!
    );
  }

  // ---------------------------------------------------------------------------
  // INTERACTION CAPTURE
  // ---------------------------------------------------------------------------

  /**
   * Log an interaction with full metadata
   */
  async logInteraction(metadata: InteractionMetadata): Promise<void> {
    const { error } = await this.supabase
      .from('elle_interactions')
      .insert({
        conversation_id: metadata.conversation_id,
        message_id: metadata.message_id,
        user_id: metadata.user_id,
        mode: metadata.mode,
        depth: metadata.depth,
        unrestricted: metadata.unrestricted,
        big_brain: metadata.big_brain,
        use_tools: metadata.use_tools,
        model: metadata.model,
        tier: metadata.tier,
        latency_ms: metadata.latency_ms,
        tokens_in: metadata.tokens_in,
        tokens_out: metadata.tokens_out,
        research_conducted: metadata.research_conducted,
        sources_count: metadata.sources_count,
        gdelt_signals_count: metadata.gdelt_signals_count,
        pages_analyzed: metadata.pages_analyzed,
        tools_used: metadata.tools_used,
        tool_results: metadata.tool_results,
        created_at: metadata.created_at,
      });

    if (error) {
      console.error('[Learning] Failed to log interaction:', error);
    }
  }

  /**
   * Update interaction with quality signals
   */
  async updateInteraction(
    messageId: string,
    updates: Partial<Pick<
      InteractionMetadata,
      'user_rating' | 'user_feedback' | 'flagged_for_training' | 'flagged_as_bad' |
      'was_edited' | 'had_followup' | 'task_completed'
    >>
  ): Promise<void> {
    const { error } = await this.supabase
      .from('elle_interactions')
      .update({
        ...updates,
        rated_at: updates.user_rating ? new Date().toISOString() : undefined,
      })
      .eq('message_id', messageId);

    if (error) {
      console.error('[Learning] Failed to update interaction:', error);
      throw error;
    }

    // If highly rated, auto-create training example
    if (updates.user_rating && updates.user_rating >= 4) {
      await this.createTrainingExampleFromMessage(messageId);
    }
  }

  // ---------------------------------------------------------------------------
  // TRAINING DATA EXTRACTION
  // ---------------------------------------------------------------------------

  /**
   * Create a training example from a rated message
   */
  async createTrainingExampleFromMessage(messageId: string): Promise<void> {
    // Get the message and its conversation
    const { data: message } = await this.supabase
      .from('study_messages')
      .select('*, conversation:study_conversations(*)')
      .eq('id', messageId)
      .single();

    if (!message || message.role !== 'assistant') return;

    // Get interaction metadata
    const { data: interaction } = await this.supabase
      .from('elle_interactions')
      .select('*')
      .eq('message_id', messageId)
      .single();

    // Get the preceding user message
    const { data: context } = await this.supabase
      .from('study_messages')
      .select('*')
      .eq('conversation_id', message.conversation_id)
      .lt('created_at', message.created_at)
      .order('created_at', { ascending: false })
      .limit(5);

    const userMessage = context?.find((m: StudyMessage) => m.role === 'user');
    if (!userMessage) return;

    // Build system prompt based on mode
    const systemPrompt = this.buildSystemPromptForMode(
      message.conversation?.mode || 'chat',
      interaction?.unrestricted,
      interaction?.big_brain
    );

    // Calculate quality score
    const qualityScore = this.calculateQualityScore(interaction);

    // Determine task type and difficulty
    const taskType = this.inferTaskType(userMessage.content, message.conversation?.mode);
    const difficulty = this.inferDifficulty(userMessage.content, message.content);

    // Create training example
    const { error } = await this.supabase
      .from('elle_training_examples')
      .insert({
        system_prompt: systemPrompt,
        user_input: userMessage.content,
        assistant_output: message.content,
        domain: message.conversation?.mode || 'chat',
        task_type: taskType,
        difficulty,
        quality_score: qualityScore,
        human_rating: interaction?.user_rating,
        source: 'study_book',
        metadata: {
          conversation_id: message.conversation_id,
          message_id: messageId,
          mode: message.conversation?.mode,
          model_used: interaction?.model,
          latency_ms: interaction?.latency_ms,
          research_sources: interaction?.sources_count,
          tools_used: interaction?.tools_used,
        },
        status: qualityScore >= 0.7 ? 'approved' : 'pending',
      });

    if (error) {
      console.error('[Learning] Failed to create training example:', error);
    }
  }

  /**
   * Batch extract training examples from highly rated interactions
   */
  async extractTrainingExamples(
    minRating: number = 4,
    limit: number = 100
  ): Promise<number> {
    // Get rated interactions not yet converted to training examples
    const { data: interactions } = await this.supabase
      .from('elle_interactions')
      .select('message_id')
      .gte('user_rating', minRating)
      .is('flagged_as_bad', false)
      .order('created_at', { ascending: false })
      .limit(limit);

    if (!interactions?.length) return 0;

    let created = 0;
    for (const interaction of interactions) {
      // Check if example already exists
      const { data: existing } = await this.supabase
        .from('elle_training_examples')
        .select('id')
        .eq('metadata->>message_id', interaction.message_id)
        .single();

      if (!existing) {
        await this.createTrainingExampleFromMessage(interaction.message_id);
        created++;
      }
    }

    return created;
  }

  // ---------------------------------------------------------------------------
  // TRAINING EXPORT
  // ---------------------------------------------------------------------------

  /**
   * Export training examples in Axolotl JSONL format
   */
  async exportForAxolotl(
    options?: {
      status?: 'approved' | 'pending';
      minQuality?: number;
      domains?: string[];
      limit?: number;
    }
  ): Promise<string> {
    let query = this.supabase
      .from('elle_training_examples')
      .select('*')
      .order('quality_score', { ascending: false });

    if (options?.status) {
      query = query.eq('status', options.status);
    }
    if (options?.minQuality) {
      query = query.gte('quality_score', options.minQuality);
    }
    if (options?.domains?.length) {
      query = query.in('domain', options.domains);
    }
    if (options?.limit) {
      query = query.limit(options.limit);
    }

    const { data: examples, error } = await query;

    if (error || !examples?.length) {
      throw new Error(error?.message || 'No training examples found');
    }

    // Convert to Axolotl chat format
    const jsonlLines = examples.map((ex: TrainingExample) => {
      const entry = {
        conversations: [
          { from: 'system', value: ex.system_prompt },
          { from: 'human', value: ex.user_input },
          { from: 'gpt', value: ex.assistant_output },
        ],
        // Axolotl metadata
        source: 'latticeforge_study_book',
        domain: ex.domain,
        task_type: ex.task_type,
        quality_score: ex.quality_score,
      };
      return JSON.stringify(entry);
    });

    // Mark as exported
    const exportedIds = examples.map((ex: TrainingExample) => ex.id);
    await this.supabase
      .from('elle_training_examples')
      .update({
        status: 'exported',
        exported_at: new Date().toISOString(),
      })
      .in('id', exportedIds);

    return jsonlLines.join('\n');
  }

  /**
   * Export training examples in ShareGPT format (alternative)
   */
  async exportShareGPT(options?: {
    minQuality?: number;
    limit?: number;
  }): Promise<string> {
    let query = this.supabase
      .from('elle_training_examples')
      .select('*')
      .in('status', ['approved', 'pending'])
      .order('quality_score', { ascending: false });

    if (options?.minQuality) {
      query = query.gte('quality_score', options.minQuality);
    }
    if (options?.limit) {
      query = query.limit(options.limit);
    }

    const { data: examples } = await query;

    if (!examples?.length) return '[]';

    const shareGPT = examples.map((ex: TrainingExample) => ({
      id: ex.id,
      conversations: [
        { from: 'system', value: ex.system_prompt },
        { from: 'human', value: ex.user_input },
        { from: 'gpt', value: ex.assistant_output },
      ],
    }));

    return JSON.stringify(shareGPT, null, 2);
  }

  // ---------------------------------------------------------------------------
  // ANALYTICS
  // ---------------------------------------------------------------------------

  /**
   * Get learning statistics
   */
  async getStats(userId?: string): Promise<LearningStats> {
    const baseQuery = userId
      ? this.supabase.from('elle_interactions').select('*').eq('user_id', userId)
      : this.supabase.from('elle_interactions').select('*');

    const { data: interactions } = await baseQuery;
    const { data: examples } = await this.supabase
      .from('elle_training_examples')
      .select('status');

    const total = interactions?.length || 0;
    const rated = interactions?.filter((i: InteractionMetadata) => i.user_rating)?.length || 0;
    const positive = interactions?.filter((i: InteractionMetadata) => (i.user_rating || 0) >= 4)?.length || 0;
    const negative = interactions?.filter((i: InteractionMetadata) => (i.user_rating || 0) <= 2)?.length || 0;

    const elleCount = interactions?.filter((i: InteractionMetadata) => i.tier === 'elle')?.length || 0;
    const workhorseCount = interactions?.filter((i: InteractionMetadata) => i.tier === 'workhorse')?.length || 0;

    const avgLatency = interactions?.length
      ? interactions.reduce((sum: number, i: InteractionMetadata) => sum + (i.latency_ms || 0), 0) / interactions.length
      : 0;

    // Count by mode
    const byMode = {} as Record<StudyMode, number>;
    for (const mode of ['chat', 'code', 'research', 'brief', 'analyze'] as StudyMode[]) {
      byMode[mode] = interactions?.filter((i: InteractionMetadata) => i.mode === mode)?.length || 0;
    }

    // Count by depth
    const byDepth = {} as Record<ResearchDepth, number>;
    for (const depth of ['instant', 'moderate', 'thorough'] as ResearchDepth[]) {
      byDepth[depth] = interactions?.filter((i: InteractionMetadata) => i.depth === depth)?.length || 0;
    }

    // Quality over time (last 30 days)
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

    const qualityOverTime: Array<{ date: string; avg_rating: number; count: number }> = [];
    const ratedInteractions = interactions?.filter(
      (i: InteractionMetadata) => i.user_rating && new Date(i.created_at) >= thirtyDaysAgo
    ) || [];

    // Group by date
    const byDate = new Map<string, number[]>();
    for (const i of ratedInteractions) {
      const date = i.created_at.split('T')[0];
      if (!byDate.has(date)) byDate.set(date, []);
      byDate.get(date)!.push(i.user_rating!);
    }

    for (const [date, ratings] of byDate) {
      qualityOverTime.push({
        date,
        avg_rating: ratings.reduce((a, b) => a + b, 0) / ratings.length,
        count: ratings.length,
      });
    }
    qualityOverTime.sort((a, b) => a.date.localeCompare(b.date));

    return {
      total_interactions: total,
      rated_interactions: rated,
      positive_ratings: positive,
      negative_ratings: negative,
      training_examples_pending: examples?.filter((e: { status: string }) => e.status === 'pending')?.length || 0,
      training_examples_approved: examples?.filter((e: { status: string }) => e.status === 'approved')?.length || 0,
      training_examples_exported: examples?.filter((e: { status: string }) => e.status === 'exported')?.length || 0,
      avg_response_time_ms: avgLatency,
      elle_usage_percent: total ? (elleCount / total) * 100 : 0,
      workhorse_usage_percent: total ? (workhorseCount / total) * 100 : 0,
      by_mode: byMode,
      by_depth: byDepth,
      quality_over_time: qualityOverTime,
    };
  }

  /**
   * Get improvement recommendations
   */
  async getImprovementRecommendations(): Promise<string[]> {
    const recommendations: string[] = [];

    // Get poorly rated interactions
    const { data: poorInteractions } = await this.supabase
      .from('elle_interactions')
      .select('mode, depth, tools_used, user_feedback')
      .lte('user_rating', 2)
      .order('created_at', { ascending: false })
      .limit(50);

    if (poorInteractions?.length) {
      // Analyze patterns
      const modeFailures = new Map<string, number>();
      const depthFailures = new Map<string, number>();
      const toolFailures = new Map<string, number>();

      for (const i of poorInteractions) {
        modeFailures.set(i.mode, (modeFailures.get(i.mode) || 0) + 1);
        depthFailures.set(i.depth, (depthFailures.get(i.depth) || 0) + 1);
        for (const tool of i.tools_used || []) {
          toolFailures.set(tool, (toolFailures.get(tool) || 0) + 1);
        }
      }

      // Generate recommendations
      const worstMode = [...modeFailures.entries()].sort((a, b) => b[1] - a[1])[0];
      if (worstMode && worstMode[1] >= 5) {
        recommendations.push(
          `Consider more training data for "${worstMode[0]}" mode - ${worstMode[1]} low ratings`
        );
      }

      const worstTool = [...toolFailures.entries()].sort((a, b) => b[1] - a[1])[0];
      if (worstTool && worstTool[1] >= 3) {
        recommendations.push(
          `The "${worstTool[0]}" tool may need improvement - ${worstTool[1]} failures in low-rated responses`
        );
      }
    }

    // Check for follow-up patterns (indicates confusion)
    const { data: followups } = await this.supabase
      .from('elle_interactions')
      .select('mode')
      .eq('had_followup', true)
      .order('created_at', { ascending: false })
      .limit(100);

    if (followups?.length) {
      const followupModes = new Map<string, number>();
      for (const i of followups) {
        followupModes.set(i.mode, (followupModes.get(i.mode) || 0) + 1);
      }

      const confusingMode = [...followupModes.entries()].sort((a, b) => b[1] - a[1])[0];
      if (confusingMode && confusingMode[1] >= 10) {
        recommendations.push(
          `"${confusingMode[0]}" mode often requires follow-ups - consider clearer initial responses`
        );
      }
    }

    // Check training data coverage
    const { data: trainingByDomain } = await this.supabase
      .from('elle_training_examples')
      .select('domain')
      .in('status', ['approved', 'exported']);

    if (trainingByDomain) {
      const domainCounts = new Map<string, number>();
      for (const ex of trainingByDomain) {
        domainCounts.set(ex.domain, (domainCounts.get(ex.domain) || 0) + 1);
      }

      for (const domain of ['chat', 'code', 'research', 'brief', 'analyze']) {
        const count = domainCounts.get(domain) || 0;
        if (count < 50) {
          recommendations.push(
            `Low training data for "${domain}" domain (${count} examples) - consider more curation`
          );
        }
      }
    }

    return recommendations;
  }

  // ---------------------------------------------------------------------------
  // HELPERS
  // ---------------------------------------------------------------------------

  private buildSystemPromptForMode(
    mode: StudyMode,
    unrestricted?: boolean,
    bigBrain?: boolean
  ): string {
    let prompt = `You are Elle, the AI intelligence analyst for LatticeForge.

PERSONA:
- Named after the "L" in LatticeForge
- Elegant, professional, direct
- You take positions and express opinions
- You challenge assumptions when warranted`;

    const modeDescriptions: Record<StudyMode, string> = {
      chat: 'General conversation mode. Be helpful and engaging.',
      code: 'Code assistant mode. Help with writing, reviewing, and debugging code.',
      research: 'Research mode. Conduct thorough research and synthesize findings.',
      brief: 'Intel brief mode. Generate structured intelligence briefings.',
      analyze: 'Analysis mode. Analyze documents and data thoroughly.',
    };

    prompt += `\n\nMODE: ${mode.charAt(0).toUpperCase() + mode.slice(1)}\n${modeDescriptions[mode]}`;

    if (unrestricted) {
      prompt += '\n\nUNRESTRICTED: No topic restrictions. Express strong opinions when warranted.';
    }

    if (bigBrain) {
      prompt += '\n\nThink through your reasoning step by step before responding.';
    }

    return prompt;
  }

  private calculateQualityScore(interaction?: InteractionMetadata | null): number {
    if (!interaction) return 0.5;

    let score = 0.5; // Base score

    // User rating is strongest signal
    if (interaction.user_rating) {
      score = interaction.user_rating / 5;
    }

    // Explicit flags
    if (interaction.flagged_for_training) score = Math.max(score, 0.8);
    if (interaction.flagged_as_bad) score = Math.min(score, 0.2);

    // Was edited (user corrected it)
    if (interaction.was_edited) score -= 0.1;

    // Had follow-up (may indicate confusion)
    if (interaction.had_followup) score -= 0.05;

    // Task completed successfully
    if (interaction.task_completed) score += 0.1;

    // Used Elle (complex task handled well)
    if (interaction.tier === 'elle' && score > 0.5) score += 0.05;

    return Math.max(0, Math.min(1, score));
  }

  private inferTaskType(userInput: string, mode?: StudyMode): string {
    const lower = userInput.toLowerCase();

    // Code-related
    if (lower.includes('write') && (lower.includes('code') || lower.includes('function') || lower.includes('script'))) {
      return 'code_generation';
    }
    if (lower.includes('fix') || lower.includes('bug') || lower.includes('error')) {
      return 'debugging';
    }
    if (lower.includes('review') || lower.includes('improve')) {
      return 'code_review';
    }
    if (lower.includes('explain') && mode === 'code') {
      return 'code_explanation';
    }

    // Research-related
    if (lower.includes('research') || lower.includes('find out') || lower.includes('what is')) {
      return 'research';
    }
    if (lower.includes('compare') || lower.includes('difference')) {
      return 'comparison';
    }

    // Analysis
    if (lower.includes('analyze') || lower.includes('analysis')) {
      return 'analysis';
    }
    if (lower.includes('summarize') || lower.includes('summary')) {
      return 'summarization';
    }

    // Brief
    if (lower.includes('brief') || lower.includes('briefing') || lower.includes('report')) {
      return 'brief_generation';
    }

    // General
    if (lower.includes('how to') || lower.includes('how do')) {
      return 'how_to';
    }
    if (lower.includes('?')) {
      return 'question_answer';
    }

    return 'general';
  }

  private inferDifficulty(
    userInput: string,
    assistantOutput: string
  ): 'easy' | 'medium' | 'hard' {
    const inputLength = userInput.length;
    const outputLength = assistantOutput.length;

    // Long, complex outputs suggest harder tasks
    if (outputLength > 3000) return 'hard';
    if (outputLength > 1000) return 'medium';

    // Complex inputs
    if (inputLength > 500) return 'hard';
    if (inputLength > 200) return 'medium';

    // Keywords suggesting complexity
    const hardKeywords = ['complex', 'advanced', 'detailed', 'comprehensive', 'multi-step'];
    const mediumKeywords = ['explain', 'analyze', 'compare', 'implement'];

    const lower = userInput.toLowerCase();
    if (hardKeywords.some(k => lower.includes(k))) return 'hard';
    if (mediumKeywords.some(k => lower.includes(k))) return 'medium';

    return 'easy';
  }
}

// =============================================================================
// SINGLETON
// =============================================================================

let learningEngine: LearningEngine | null = null;

export function getLearningEngine(): LearningEngine {
  if (!learningEngine) {
    learningEngine = new LearningEngine();
  }
  return learningEngine;
}

// =============================================================================
// DATABASE SCHEMA
// =============================================================================

export const LEARNING_SCHEMA_SQL = `
-- Elle Interactions (detailed logging)
CREATE TABLE IF NOT EXISTS elle_interactions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID REFERENCES study_conversations(id) ON DELETE CASCADE,
  message_id UUID REFERENCES study_messages(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

  -- Context
  mode TEXT NOT NULL,
  depth TEXT NOT NULL,
  unrestricted BOOLEAN DEFAULT false,
  big_brain BOOLEAN DEFAULT false,
  use_tools BOOLEAN DEFAULT true,

  -- Performance
  model TEXT,
  tier TEXT,
  latency_ms INTEGER,
  tokens_in INTEGER,
  tokens_out INTEGER,

  -- Research
  research_conducted BOOLEAN DEFAULT false,
  sources_count INTEGER,
  gdelt_signals_count INTEGER,
  pages_analyzed INTEGER,

  -- Tools
  tools_used TEXT[] DEFAULT '{}',
  tool_results JSONB DEFAULT '[]',

  -- Quality signals
  user_rating INTEGER CHECK (user_rating >= 1 AND user_rating <= 5),
  user_feedback TEXT,
  flagged_for_training BOOLEAN DEFAULT false,
  flagged_as_bad BOOLEAN DEFAULT false,
  was_edited BOOLEAN DEFAULT false,
  had_followup BOOLEAN DEFAULT false,
  task_completed BOOLEAN,

  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  rated_at TIMESTAMPTZ
);

-- Elle Training Examples
CREATE TABLE IF NOT EXISTS elle_training_examples (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Core training pair
  system_prompt TEXT NOT NULL,
  user_input TEXT NOT NULL,
  assistant_output TEXT NOT NULL,

  -- Categorization
  domain TEXT NOT NULL,
  task_type TEXT NOT NULL,
  difficulty TEXT CHECK (difficulty IN ('easy', 'medium', 'hard')),

  -- Quality
  quality_score FLOAT NOT NULL DEFAULT 0.5,
  human_rating INTEGER,
  source TEXT NOT NULL DEFAULT 'study_book',

  -- Metadata
  metadata JSONB DEFAULT '{}',

  -- Status
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'exported')),
  reviewed_at TIMESTAMPTZ,
  exported_at TIMESTAMPTZ,

  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_elle_interactions_user ON elle_interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_elle_interactions_message ON elle_interactions(message_id);
CREATE INDEX IF NOT EXISTS idx_elle_interactions_rating ON elle_interactions(user_rating) WHERE user_rating IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_elle_interactions_created ON elle_interactions(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_elle_training_status ON elle_training_examples(status);
CREATE INDEX IF NOT EXISTS idx_elle_training_quality ON elle_training_examples(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_elle_training_domain ON elle_training_examples(domain);

-- RLS
ALTER TABLE elle_interactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE elle_training_examples ENABLE ROW LEVEL SECURITY;

-- Admin can see all interactions
CREATE POLICY elle_interactions_admin_policy ON elle_interactions
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM profiles WHERE id = auth.uid() AND role = 'admin'
    )
  );

-- Service role can access everything
CREATE POLICY elle_interactions_service_policy ON elle_interactions
  FOR ALL USING (auth.jwt()->>'role' = 'service_role');

CREATE POLICY elle_training_service_policy ON elle_training_examples
  FOR ALL USING (auth.jwt()->>'role' = 'service_role');
`;
