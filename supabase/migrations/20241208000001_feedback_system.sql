-- Feedback System for Bug Reports and Ideas/Suggestions
-- Migration: 20241208000001_feedback_system

-- Create feedback table
CREATE TABLE IF NOT EXISTS public.feedback (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,

  -- Feedback classification
  type TEXT NOT NULL CHECK (type IN ('bug', 'idea', 'question', 'other')),
  title TEXT NOT NULL,
  description TEXT NOT NULL,

  -- Status workflow: unread → acknowledged → in_progress → resolved/wont_fix/duplicate
  status TEXT DEFAULT 'unread' CHECK (status IN (
    'unread',        -- New, not yet seen
    'acknowledged',  -- Support has seen it
    'in_progress',   -- Being worked on
    'resolved',      -- Fixed/implemented
    'wont_fix',      -- Decided not to address
    'duplicate'      -- Already reported
  )),

  -- Priority for triage
  priority TEXT DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high', 'critical')),

  -- Assignment
  assigned_to UUID REFERENCES public.profiles(id) ON DELETE SET NULL,

  -- Context capture (auto-filled by frontend)
  page_url TEXT,
  user_agent TEXT,
  screenshot_url TEXT,  -- Optional screenshot in Supabase Storage

  -- Admin/support notes (internal)
  admin_notes TEXT,
  resolution_notes TEXT,

  -- Metadata for extensibility
  metadata JSONB DEFAULT '{}',

  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  resolved_at TIMESTAMPTZ
);

-- Create indexes for common queries
CREATE INDEX idx_feedback_status ON public.feedback(status);
CREATE INDEX idx_feedback_type ON public.feedback(type);
CREATE INDEX idx_feedback_priority ON public.feedback(priority);
CREATE INDEX idx_feedback_user_id ON public.feedback(user_id);
CREATE INDEX idx_feedback_assigned_to ON public.feedback(assigned_to);
CREATE INDEX idx_feedback_created_at ON public.feedback(created_at DESC);

-- Enable RLS
ALTER TABLE public.feedback ENABLE ROW LEVEL SECURITY;

-- Policy: Users can insert their own feedback
CREATE POLICY "Users can submit feedback"
  ON public.feedback FOR INSERT
  WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

-- Policy: Users can view their own feedback
CREATE POLICY "Users can view own feedback"
  ON public.feedback FOR SELECT
  USING (auth.uid() = user_id);

-- Policy: Admins and support can view all feedback
CREATE POLICY "Admins can view all feedback"
  ON public.feedback FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.profiles
      WHERE id = auth.uid() AND role IN ('admin', 'support')
    )
  );

-- Policy: Admins and support can update any feedback
CREATE POLICY "Admins can update feedback"
  ON public.feedback FOR UPDATE
  USING (
    EXISTS (
      SELECT 1 FROM public.profiles
      WHERE id = auth.uid() AND role IN ('admin', 'support')
    )
  );

-- Policy: Only admins can delete feedback
CREATE POLICY "Admins can delete feedback"
  ON public.feedback FOR DELETE
  USING (
    EXISTS (
      SELECT 1 FROM public.profiles
      WHERE id = auth.uid() AND role = 'admin'
    )
  );

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_feedback_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  -- Set resolved_at when status changes to resolved
  IF NEW.status = 'resolved' AND OLD.status != 'resolved' THEN
    NEW.resolved_at = now();
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = public;

CREATE TRIGGER feedback_updated_at
  BEFORE UPDATE ON public.feedback
  FOR EACH ROW
  EXECUTE FUNCTION update_feedback_timestamp();

-- Grant permissions
GRANT SELECT, INSERT ON public.feedback TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.feedback TO service_role;

-- Create view for feedback stats (admin dashboard)
CREATE OR REPLACE VIEW public.feedback_stats AS
SELECT
  COUNT(*) FILTER (WHERE status = 'unread') AS unread_count,
  COUNT(*) FILTER (WHERE status = 'acknowledged') AS acknowledged_count,
  COUNT(*) FILTER (WHERE status = 'in_progress') AS in_progress_count,
  COUNT(*) FILTER (WHERE status = 'resolved') AS resolved_count,
  COUNT(*) FILTER (WHERE type = 'bug') AS bug_count,
  COUNT(*) FILTER (WHERE type = 'idea') AS idea_count,
  COUNT(*) FILTER (WHERE priority = 'critical') AS critical_count,
  COUNT(*) FILTER (WHERE priority = 'high') AS high_priority_count,
  COUNT(*) AS total_count,
  MAX(created_at) AS latest_feedback_at
FROM public.feedback;

-- Grant view access to admins
GRANT SELECT ON public.feedback_stats TO authenticated;

COMMENT ON TABLE public.feedback IS 'User feedback including bug reports, feature ideas, and questions';
COMMENT ON COLUMN public.feedback.type IS 'Type of feedback: bug, idea, question, or other';
COMMENT ON COLUMN public.feedback.status IS 'Workflow status from unread through resolution';
COMMENT ON COLUMN public.feedback.metadata IS 'Extensible JSON for additional context like browser info, feature flags, etc.';
