/**
 * Study Book Memory System
 *
 * Persistent conversation memory for the admin Study Book interface.
 * Unlike commercial AI assistants with weak/no memory, we store EVERYTHING.
 *
 * Features:
 * - Full conversation history in Supabase
 * - Semantic search over past conversations
 * - Learning queue for retraining Elle
 * - Cross-session context retrieval
 */

import { createClient, SupabaseClient } from '@supabase/supabase-js';

// =============================================================================
// TYPES
// =============================================================================

export interface StudyMessage {
  id?: string;
  conversation_id: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  metadata?: {
    model?: string;
    tier?: 'workhorse' | 'elle';
    latency_ms?: number;
    tokens?: number;
    mode?: StudyMode;
    tools_used?: string[];
    files_referenced?: string[];
    thinking?: string;  // Chain-of-thought if Big Brain mode
    rating?: number;    // User rating for learning
    flagged_for_training?: boolean;
  };
  created_at?: string;
}

export interface StudyConversation {
  id?: string;
  user_id: string;
  title?: string;
  mode: StudyMode;
  metadata?: {
    github_repo?: string;
    github_branch?: string;
    files_context?: string[];
    custom_system_prompt?: string;
    unrestricted?: boolean;
    big_brain?: boolean;
  };
  message_count?: number;
  last_message_at?: string;
  created_at?: string;
  updated_at?: string;
}

export type StudyMode =
  | 'chat'      // General conversation
  | 'code'      // Code assistant mode
  | 'research'  // Deep research with GDELT + web
  | 'brief'     // Intel brief generation
  | 'analyze';  // Document/data analysis

// =============================================================================
// MEMORY CLIENT
// =============================================================================

export class StudyMemory {
  private supabase: SupabaseClient;
  private userId: string;

  constructor(userId: string, supabaseUrl?: string, supabaseKey?: string) {
    this.userId = userId;
    this.supabase = createClient(
      supabaseUrl || process.env.NEXT_PUBLIC_SUPABASE_URL!,
      supabaseKey || process.env.SUPABASE_SERVICE_ROLE_KEY!
    );
  }

  // ---------------------------------------------------------------------------
  // CONVERSATIONS
  // ---------------------------------------------------------------------------

  /**
   * Create a new conversation
   */
  async createConversation(
    mode: StudyMode,
    options?: {
      title?: string;
      metadata?: StudyConversation['metadata'];
    }
  ): Promise<StudyConversation> {
    const { data, error } = await this.supabase
      .from('study_conversations')
      .insert({
        user_id: this.userId,
        mode,
        title: options?.title || `${mode} - ${new Date().toLocaleDateString()}`,
        metadata: options?.metadata || {},
      })
      .select()
      .single();

    if (error) throw new Error(`Failed to create conversation: ${error.message}`);
    return data;
  }

  /**
   * Get conversation by ID
   */
  async getConversation(conversationId: string): Promise<StudyConversation | null> {
    const { data, error } = await this.supabase
      .from('study_conversations')
      .select('*')
      .eq('id', conversationId)
      .eq('user_id', this.userId)
      .single();

    if (error) return null;
    return data;
  }

  /**
   * List recent conversations
   */
  async listConversations(
    options?: {
      mode?: StudyMode;
      limit?: number;
      offset?: number;
    }
  ): Promise<StudyConversation[]> {
    let query = this.supabase
      .from('study_conversations')
      .select('*')
      .eq('user_id', this.userId)
      .order('updated_at', { ascending: false });

    if (options?.mode) {
      query = query.eq('mode', options.mode);
    }
    if (options?.limit) {
      query = query.limit(options.limit);
    }
    if (options?.offset) {
      query = query.range(options.offset, options.offset + (options.limit || 20) - 1);
    }

    const { data, error } = await query;
    if (error) throw new Error(`Failed to list conversations: ${error.message}`);
    return data || [];
  }

  /**
   * Update conversation metadata
   */
  async updateConversation(
    conversationId: string,
    updates: Partial<Pick<StudyConversation, 'title' | 'metadata'>>
  ): Promise<void> {
    const { error } = await this.supabase
      .from('study_conversations')
      .update({
        ...updates,
        updated_at: new Date().toISOString(),
      })
      .eq('id', conversationId)
      .eq('user_id', this.userId);

    if (error) throw new Error(`Failed to update conversation: ${error.message}`);
  }

  /**
   * Delete a conversation and all its messages
   */
  async deleteConversation(conversationId: string): Promise<void> {
    // Messages will be cascade deleted
    const { error } = await this.supabase
      .from('study_conversations')
      .delete()
      .eq('id', conversationId)
      .eq('user_id', this.userId);

    if (error) throw new Error(`Failed to delete conversation: ${error.message}`);
  }

  // ---------------------------------------------------------------------------
  // MESSAGES
  // ---------------------------------------------------------------------------

  /**
   * Add a message to a conversation
   */
  async addMessage(message: StudyMessage): Promise<StudyMessage> {
    const { data, error } = await this.supabase
      .from('study_messages')
      .insert({
        conversation_id: message.conversation_id,
        role: message.role,
        content: message.content,
        metadata: message.metadata || {},
      })
      .select()
      .single();

    if (error) throw new Error(`Failed to add message: ${error.message}`);

    // Update conversation timestamp
    await this.supabase
      .from('study_conversations')
      .update({
        updated_at: new Date().toISOString(),
        last_message_at: new Date().toISOString(),
      })
      .eq('id', message.conversation_id);

    return data;
  }

  /**
   * Get messages for a conversation
   */
  async getMessages(
    conversationId: string,
    options?: {
      limit?: number;
      before?: string;  // Get messages before this timestamp
    }
  ): Promise<StudyMessage[]> {
    let query = this.supabase
      .from('study_messages')
      .select('*')
      .eq('conversation_id', conversationId)
      .order('created_at', { ascending: true });

    if (options?.before) {
      query = query.lt('created_at', options.before);
    }
    if (options?.limit) {
      query = query.limit(options.limit);
    }

    const { data, error } = await query;
    if (error) throw new Error(`Failed to get messages: ${error.message}`);
    return data || [];
  }

  /**
   * Get recent messages for context (most recent N messages)
   */
  async getRecentContext(
    conversationId: string,
    limit: number = 20
  ): Promise<StudyMessage[]> {
    const { data, error } = await this.supabase
      .from('study_messages')
      .select('*')
      .eq('conversation_id', conversationId)
      .order('created_at', { ascending: false })
      .limit(limit);

    if (error) throw new Error(`Failed to get context: ${error.message}`);
    return (data || []).reverse();  // Return in chronological order
  }

  /**
   * Rate a message (for learning/improvement)
   */
  async rateMessage(
    messageId: string,
    rating: number,
    flagForTraining: boolean = false
  ): Promise<void> {
    const { data: message, error: fetchError } = await this.supabase
      .from('study_messages')
      .select('metadata')
      .eq('id', messageId)
      .single();

    if (fetchError) throw new Error(`Message not found: ${fetchError.message}`);

    const { error } = await this.supabase
      .from('study_messages')
      .update({
        metadata: {
          ...message.metadata,
          rating,
          flagged_for_training: flagForTraining,
        },
      })
      .eq('id', messageId);

    if (error) throw new Error(`Failed to rate message: ${error.message}`);

    // If flagged for training, add to learning queue
    if (flagForTraining && rating >= 4) {
      await this.addToLearningQueue(messageId);
    }
  }

  // ---------------------------------------------------------------------------
  // LEARNING QUEUE (for Elle retraining)
  // ---------------------------------------------------------------------------

  /**
   * Add a high-quality response to the learning queue
   */
  private async addToLearningQueue(messageId: string): Promise<void> {
    // Get the message and its context
    const { data: message } = await this.supabase
      .from('study_messages')
      .select('*, conversation:study_conversations(*)')
      .eq('id', messageId)
      .single();

    if (!message || message.role !== 'assistant') return;

    // Get the preceding user message
    const { data: messages } = await this.supabase
      .from('study_messages')
      .select('*')
      .eq('conversation_id', message.conversation_id)
      .lt('created_at', message.created_at)
      .order('created_at', { ascending: false })
      .limit(1);

    const userMessage = messages?.[0];
    if (!userMessage || userMessage.role !== 'user') return;

    // Add to training examples
    await this.supabase
      .from('training_examples')
      .insert({
        input: userMessage.content,
        output: message.content,
        domain: message.conversation?.mode || 'general',
        source: 'study_book_rated',
        metadata: {
          message_id: messageId,
          conversation_id: message.conversation_id,
          rating: message.metadata?.rating,
          mode: message.metadata?.mode,
        },
      });
  }

  /**
   * Get training examples from rated messages
   */
  async getTrainingExamples(
    minRating: number = 4,
    limit: number = 100
  ): Promise<Array<{ input: string; output: string; domain: string }>> {
    const { data, error } = await this.supabase
      .from('training_examples')
      .select('input, output, domain')
      .eq('source', 'study_book_rated')
      .gte('metadata->>rating', minRating)
      .order('created_at', { ascending: false })
      .limit(limit);

    if (error) throw new Error(`Failed to get training examples: ${error.message}`);
    return data || [];
  }

  // ---------------------------------------------------------------------------
  // SEMANTIC SEARCH (across all conversations)
  // ---------------------------------------------------------------------------

  /**
   * Search past conversations for relevant context
   * Uses PostgreSQL full-text search (can upgrade to pgvector later)
   */
  async searchMemory(
    query: string,
    options?: {
      mode?: StudyMode;
      limit?: number;
    }
  ): Promise<Array<StudyMessage & { conversation: StudyConversation }>> {
    // Basic text search - can upgrade to vector search later
    const searchQuery = query.split(' ').join(' & ');

    let dbQuery = this.supabase
      .from('study_messages')
      .select('*, conversation:study_conversations!inner(*)')
      .eq('study_conversations.user_id', this.userId)
      .textSearch('content', searchQuery)
      .order('created_at', { ascending: false })
      .limit(options?.limit || 10);

    if (options?.mode) {
      dbQuery = dbQuery.eq('study_conversations.mode', options.mode);
    }

    const { data, error } = await dbQuery;
    if (error) {
      // Fallback to LIKE search if FTS fails
      const { data: fallbackData } = await this.supabase
        .from('study_messages')
        .select('*, conversation:study_conversations!inner(*)')
        .eq('study_conversations.user_id', this.userId)
        .ilike('content', `%${query}%`)
        .order('created_at', { ascending: false })
        .limit(options?.limit || 10);

      return fallbackData || [];
    }

    return data || [];
  }
}

// =============================================================================
// SINGLETON
// =============================================================================

let memoryInstance: StudyMemory | null = null;

export function getStudyMemory(userId: string): StudyMemory {
  if (!memoryInstance || memoryInstance['userId'] !== userId) {
    memoryInstance = new StudyMemory(userId);
  }
  return memoryInstance;
}

// =============================================================================
// DATABASE SCHEMA (run this in Supabase SQL editor)
// =============================================================================

export const STUDY_SCHEMA_SQL = `
-- Study Book Conversations
CREATE TABLE IF NOT EXISTS study_conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  title TEXT,
  mode TEXT NOT NULL DEFAULT 'chat',
  metadata JSONB DEFAULT '{}',
  message_count INTEGER DEFAULT 0,
  last_message_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Study Book Messages
CREATE TABLE IF NOT EXISTS study_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES study_conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
  content TEXT NOT NULL,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_study_conversations_user ON study_conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_study_conversations_updated ON study_conversations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_study_messages_conversation ON study_messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_study_messages_created ON study_messages(created_at);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_study_messages_content_fts ON study_messages
  USING gin(to_tsvector('english', content));

-- RLS Policies
ALTER TABLE study_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE study_messages ENABLE ROW LEVEL SECURITY;

-- Users can only see their own conversations
CREATE POLICY study_conversations_user_policy ON study_conversations
  FOR ALL USING (auth.uid() = user_id);

-- Users can only see messages in their conversations
CREATE POLICY study_messages_user_policy ON study_messages
  FOR ALL USING (
    conversation_id IN (
      SELECT id FROM study_conversations WHERE user_id = auth.uid()
    )
  );

-- Admin bypass for service role
CREATE POLICY study_conversations_service_policy ON study_conversations
  FOR ALL USING (auth.jwt()->>'role' = 'service_role');

CREATE POLICY study_messages_service_policy ON study_messages
  FOR ALL USING (auth.jwt()->>'role' = 'service_role');

-- Update message count trigger
CREATE OR REPLACE FUNCTION update_conversation_message_count()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    UPDATE study_conversations
    SET message_count = message_count + 1
    WHERE id = NEW.conversation_id;
  ELSIF TG_OP = 'DELETE' THEN
    UPDATE study_conversations
    SET message_count = message_count - 1
    WHERE id = OLD.conversation_id;
  END IF;
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS study_messages_count_trigger ON study_messages;
CREATE TRIGGER study_messages_count_trigger
  AFTER INSERT OR DELETE ON study_messages
  FOR EACH ROW EXECUTE FUNCTION update_conversation_message_count();
`;
