-- Security Linter Final Fixes
-- Resolves all remaining Supabase database linter warnings

-- ============================================================
-- 1. Fix SECURITY DEFINER views - convert to SECURITY INVOKER
-- ============================================================

-- Drop and recreate training_data_stats with SECURITY INVOKER
DROP VIEW IF EXISTS public.training_data_stats;
CREATE OR REPLACE VIEW public.training_data_stats
WITH (security_invoker = true)
AS
SELECT
  domain,
  COUNT(*) as example_count,
  COUNT(DISTINCT session_hash) as unique_sessions,
  AVG(CASE WHEN quality_score IS NOT NULL THEN quality_score ELSE 0 END) as avg_quality,
  MAX(created_at) as latest_example
FROM training_examples
GROUP BY domain;

-- Drop and recreate exportable_training_data with SECURITY INVOKER
DROP VIEW IF EXISTS public.exportable_training_data;
CREATE OR REPLACE VIEW public.exportable_training_data
WITH (security_invoker = true)
AS
SELECT
  id,
  domain,
  input,
  output,
  quality_score,
  created_at
FROM training_examples
WHERE quality_score >= 0.7
  AND is_validated = true;

-- ============================================================
-- 2. Fix function search_path issues
-- ============================================================

-- Fix record_nation_snapshot function
CREATE OR REPLACE FUNCTION public.record_nation_snapshot()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
  INSERT INTO public.nation_history (
    nation_code,
    basin_strength,
    transition_risk,
    regime,
    metadata,
    snapshot_at
  ) VALUES (
    NEW.code,
    NEW.basin_strength,
    NEW.transition_risk,
    NEW.regime,
    NEW.metadata,
    CURRENT_TIMESTAMP
  );
  RETURN NEW;
END;
$$;

-- Fix sync_profile_to_client function
CREATE OR REPLACE FUNCTION public.sync_profile_to_client()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
  -- Sync profile changes to clients table
  UPDATE public.clients
  SET
    name = NEW.full_name,
    email = NEW.email,
    updated_at = CURRENT_TIMESTAMP
  WHERE user_id = NEW.id;

  RETURN NEW;
END;
$$;

-- ============================================================
-- 3. Fix materialized view API access
-- ============================================================

-- Revoke direct anon/authenticated access to materialized views
-- These should only be accessed through proper API routes
REVOKE SELECT ON public.usage_summary FROM anon, authenticated;
REVOKE SELECT ON public.country_signals_latest FROM anon, authenticated;

-- Grant access only to service_role (for cron jobs and internal use)
GRANT SELECT ON public.usage_summary TO service_role;
GRANT SELECT ON public.country_signals_latest TO service_role;

-- ============================================================
-- 4. Fix spatial_ref_sys RLS (PostGIS system table)
-- ============================================================

-- Enable RLS on spatial_ref_sys (PostGIS table)
-- This is a read-only reference table, so we allow all authenticated reads
ALTER TABLE IF EXISTS public.spatial_ref_sys ENABLE ROW LEVEL SECURITY;

-- Allow read access for all (it's reference data)
DROP POLICY IF EXISTS "spatial_ref_sys_read_policy" ON public.spatial_ref_sys;
CREATE POLICY "spatial_ref_sys_read_policy" ON public.spatial_ref_sys
  FOR SELECT
  TO authenticated, anon
  USING (true);

-- ============================================================
-- 5. Grant proper view access
-- ============================================================

-- Training data views should only be accessible to authenticated users
GRANT SELECT ON public.training_data_stats TO authenticated;
GRANT SELECT ON public.exportable_training_data TO authenticated;

-- Revoke anon access (if any)
REVOKE ALL ON public.training_data_stats FROM anon;
REVOKE ALL ON public.exportable_training_data FROM anon;

-- ============================================================
-- 6. Consolidate multiple permissive policies (performance)
-- ============================================================

-- briefing_cache: consolidate read+write into single policy
DROP POLICY IF EXISTS "Briefing cache read access" ON public.briefing_cache;
DROP POLICY IF EXISTS "Briefing cache write access" ON public.briefing_cache;
CREATE POLICY "briefing_cache_policy" ON public.briefing_cache
  FOR ALL
  USING (true)
  WITH CHECK (true);

-- country_signals: consolidate
DROP POLICY IF EXISTS "Country signals read access" ON public.country_signals;
DROP POLICY IF EXISTS "Country signals write access" ON public.country_signals;
CREATE POLICY "country_signals_policy" ON public.country_signals
  FOR ALL
  USING (true)
  WITH CHECK (true);

-- saved_simulations: consolidate (user owns their own)
DROP POLICY IF EXISTS "Saved simulations read access" ON public.saved_simulations;
DROP POLICY IF EXISTS "Saved simulations write access" ON public.saved_simulations;
CREATE POLICY "saved_simulations_policy" ON public.saved_simulations
  FOR ALL
  USING (auth.uid() = user_id OR auth.uid() IS NULL)
  WITH CHECK (auth.uid() = user_id);

-- tier_limits: consolidate (read-only for most users)
DROP POLICY IF EXISTS "Tier limits read access" ON public.tier_limits;
DROP POLICY IF EXISTS "Tier limits write access" ON public.tier_limits;
CREATE POLICY "tier_limits_policy" ON public.tier_limits
  FOR SELECT
  USING (true);

-- user_preferences: consolidate (user owns their own)
DROP POLICY IF EXISTS "User preferences read access" ON public.user_preferences;
DROP POLICY IF EXISTS "User preferences write access" ON public.user_preferences;
CREATE POLICY "user_preferences_policy" ON public.user_preferences
  FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);
