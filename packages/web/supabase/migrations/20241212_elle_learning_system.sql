-- Elle Learning System Migration
--
-- Creates tables for capturing Elle interactions and training data
-- for continuous improvement through fine-tuning.

-- =============================================================================
-- DEPENDENCIES: study_conversations, study_messages tables must exist first
-- =============================================================================

-- Check if study tables exist, create if not
CREATE TABLE IF NOT EXISTS study_conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  mode TEXT NOT NULL DEFAULT 'chat',
  title TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS study_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES study_conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
  content TEXT NOT NULL,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- ELLE INTERACTIONS TABLE
-- Captures detailed metadata for each Elle response
-- =============================================================================

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

  -- Quality signals (updated over time via feedback)
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

-- =============================================================================
-- ELLE TRAINING EXAMPLES TABLE
-- Stores curated training pairs for fine-tuning
-- =============================================================================

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

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Interactions indexes
CREATE INDEX IF NOT EXISTS idx_elle_interactions_user ON elle_interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_elle_interactions_conversation ON elle_interactions(conversation_id);
CREATE INDEX IF NOT EXISTS idx_elle_interactions_message ON elle_interactions(message_id);
CREATE INDEX IF NOT EXISTS idx_elle_interactions_rating ON elle_interactions(user_rating) WHERE user_rating IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_elle_interactions_created ON elle_interactions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_elle_interactions_mode ON elle_interactions(mode);
CREATE INDEX IF NOT EXISTS idx_elle_interactions_tier ON elle_interactions(tier);

-- Training examples indexes
CREATE INDEX IF NOT EXISTS idx_elle_training_status ON elle_training_examples(status);
CREATE INDEX IF NOT EXISTS idx_elle_training_quality ON elle_training_examples(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_elle_training_domain ON elle_training_examples(domain);
CREATE INDEX IF NOT EXISTS idx_elle_training_source ON elle_training_examples(source);

-- Study tables indexes (if they don't exist)
CREATE INDEX IF NOT EXISTS idx_study_conversations_user ON study_conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_study_conversations_mode ON study_conversations(mode);
CREATE INDEX IF NOT EXISTS idx_study_messages_conversation ON study_messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_study_messages_created ON study_messages(created_at DESC);

-- =============================================================================
-- ROW LEVEL SECURITY
-- =============================================================================

ALTER TABLE study_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE study_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE elle_interactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE elle_training_examples ENABLE ROW LEVEL SECURITY;

-- Study conversations: users see their own, admins see all
DROP POLICY IF EXISTS study_conversations_user_policy ON study_conversations;
CREATE POLICY study_conversations_user_policy ON study_conversations
  FOR ALL USING (
    user_id = auth.uid() OR
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND role = 'admin')
  );

-- Study messages: users see their own conversations, admins see all
DROP POLICY IF EXISTS study_messages_user_policy ON study_messages;
CREATE POLICY study_messages_user_policy ON study_messages
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM study_conversations sc
      WHERE sc.id = study_messages.conversation_id
      AND (sc.user_id = auth.uid() OR EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND role = 'admin'))
    )
  );

-- Elle interactions: admins only
DROP POLICY IF EXISTS elle_interactions_admin_policy ON elle_interactions;
CREATE POLICY elle_interactions_admin_policy ON elle_interactions
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM profiles WHERE id = auth.uid() AND role = 'admin'
    )
  );

-- Service role can access everything
DROP POLICY IF EXISTS elle_interactions_service_policy ON elle_interactions;
CREATE POLICY elle_interactions_service_policy ON elle_interactions
  FOR ALL USING (auth.jwt()->>'role' = 'service_role');

DROP POLICY IF EXISTS elle_training_service_policy ON elle_training_examples;
CREATE POLICY elle_training_service_policy ON elle_training_examples
  FOR ALL USING (auth.jwt()->>'role' = 'service_role');

-- Admins can manage training examples
DROP POLICY IF EXISTS elle_training_admin_policy ON elle_training_examples;
CREATE POLICY elle_training_admin_policy ON elle_training_examples
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM profiles WHERE id = auth.uid() AND role = 'admin'
    )
  );

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Auto-update timestamps
CREATE OR REPLACE FUNCTION update_study_conversation_timestamp()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  UPDATE study_conversations
  SET updated_at = NOW()
  WHERE id = NEW.conversation_id;
  RETURN NEW;
END;
$$;

-- Trigger to update conversation timestamp on new message
DROP TRIGGER IF EXISTS study_message_update_conversation ON study_messages;
CREATE TRIGGER study_message_update_conversation
  AFTER INSERT ON study_messages
  FOR EACH ROW
  EXECUTE FUNCTION update_study_conversation_timestamp();

-- Function to get training data stats
CREATE OR REPLACE FUNCTION get_elle_training_stats()
RETURNS TABLE (
  total_interactions BIGINT,
  rated_interactions BIGINT,
  positive_ratings BIGINT,
  training_pending BIGINT,
  training_approved BIGINT,
  training_exported BIGINT
)
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT
    (SELECT COUNT(*) FROM elle_interactions) as total_interactions,
    (SELECT COUNT(*) FROM elle_interactions WHERE user_rating IS NOT NULL) as rated_interactions,
    (SELECT COUNT(*) FROM elle_interactions WHERE user_rating >= 4) as positive_ratings,
    (SELECT COUNT(*) FROM elle_training_examples WHERE status = 'pending') as training_pending,
    (SELECT COUNT(*) FROM elle_training_examples WHERE status = 'approved') as training_approved,
    (SELECT COUNT(*) FROM elle_training_examples WHERE status = 'exported') as training_exported;
$$;

-- Grant execute to authenticated users (RLS handles access)
GRANT EXECUTE ON FUNCTION get_elle_training_stats() TO authenticated;
