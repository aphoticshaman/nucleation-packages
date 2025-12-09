-- ============================================================================
-- RLS Cleanup - Fix remaining auth.role() and consolidate duplicate policies
-- ============================================================================
-- This migration fixes:
-- 1. auth_rls_initplan: Change auth.role() to (select auth.role())
-- 2. multiple_permissive_policies: Consolidate into single policies
-- ============================================================================

-- ============================================================================
-- PART 1: Fix auth.role() → (select auth.role()) in service role policies
-- ============================================================================

-- alert_send_log: Consolidate user + service role policies
DROP POLICY IF EXISTS "Users can manage own alert logs" ON public.alert_send_log;
DROP POLICY IF EXISTS "Service role full access to alert log" ON public.alert_send_log;

CREATE POLICY "Alert log access" ON public.alert_send_log
    FOR ALL USING (
        (select auth.uid()) = user_id
        OR (select auth.role()) = 'service_role'
    );

-- tier_limits: Consolidate anyone read + service role manage
DROP POLICY IF EXISTS "Anyone can view tier limits" ON public.tier_limits;
DROP POLICY IF EXISTS "Service role can manage tier_limits" ON public.tier_limits;

CREATE POLICY "Tier limits read access" ON public.tier_limits
    FOR SELECT USING (true);

CREATE POLICY "Tier limits write access" ON public.tier_limits
    FOR ALL USING ((select auth.role()) = 'service_role');

-- country_signals: Consolidate anyone read + service role manage
DROP POLICY IF EXISTS "Anyone can view country signals" ON public.country_signals;
DROP POLICY IF EXISTS "Service role can manage country signals" ON public.country_signals;

CREATE POLICY "Country signals read access" ON public.country_signals
    FOR SELECT USING (true);

CREATE POLICY "Country signals write access" ON public.country_signals
    FOR ALL USING ((select auth.role()) = 'service_role');

-- email_export_preferences: Consolidate user + service role
DROP POLICY IF EXISTS "Users can manage own email preferences" ON public.email_export_preferences;
DROP POLICY IF EXISTS "Service role full access to email preferences" ON public.email_export_preferences;

CREATE POLICY "Email preferences access" ON public.email_export_preferences
    FOR ALL USING (
        (select auth.uid()) = user_id
        OR (select auth.role()) = 'service_role'
    );

-- briefing_cache: Consolidate anyone read + service role manage
DROP POLICY IF EXISTS "Anyone can read briefing cache" ON public.briefing_cache;
DROP POLICY IF EXISTS "Service role can manage cache" ON public.briefing_cache;

CREATE POLICY "Briefing cache read access" ON public.briefing_cache
    FOR SELECT USING (true);

CREATE POLICY "Briefing cache write access" ON public.briefing_cache
    FOR ALL USING ((select auth.role()) = 'service_role');

-- briefs: Fix service role policy
DROP POLICY IF EXISTS "Authenticated users can view briefs" ON public.briefs;
DROP POLICY IF EXISTS "Service role can insert briefs" ON public.briefs;

CREATE POLICY "Briefs read access" ON public.briefs
    FOR SELECT USING ((select auth.uid()) IS NOT NULL);

CREATE POLICY "Briefs write access" ON public.briefs
    FOR INSERT WITH CHECK ((select auth.role()) = 'service_role');

-- reasoning_traces: Fix service role policy
DROP POLICY IF EXISTS "Service role full access" ON public.reasoning_traces;

CREATE POLICY "Reasoning traces access" ON public.reasoning_traces
    FOR ALL USING ((select auth.role()) = 'service_role');

-- prediction_outcomes: Fix service role policy
DROP POLICY IF EXISTS "Service role full access" ON public.prediction_outcomes;

CREATE POLICY "Prediction outcomes access" ON public.prediction_outcomes
    FOR ALL USING ((select auth.role()) = 'service_role');

-- historical_cases: Fix service role + admin policy
DROP POLICY IF EXISTS "Service role and admins can manage historical cases" ON public.historical_cases;

CREATE POLICY "Historical cases access" ON public.historical_cases
    FOR ALL USING (
        (select auth.role()) = 'service_role'
        OR EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
    );

-- rate_limits: Fix service role policy
DROP POLICY IF EXISTS "Service role full access" ON public.rate_limits;

CREATE POLICY "Rate limits access" ON public.rate_limits
    FOR ALL USING ((select auth.role()) = 'service_role');

-- training_batches: Fix service role policy
DROP POLICY IF EXISTS "Service role full access" ON public.training_batches;

CREATE POLICY "Training batches access" ON public.training_batches
    FOR ALL USING ((select auth.role()) = 'service_role');

-- training_examples: Fix admin + service role policy
DROP POLICY IF EXISTS "Admin full access to training_examples" ON public.training_examples;

CREATE POLICY "Training examples access" ON public.training_examples
    FOR ALL USING (
        (select auth.role()) = 'service_role'
        OR EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
    );

-- nation_changes: Fix admin + service role policy
DROP POLICY IF EXISTS "Admin full access to nation_changes" ON public.nation_changes;

CREATE POLICY "Nation changes access" ON public.nation_changes
    FOR ALL USING (
        (select auth.role()) = 'service_role'
        OR EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
    );

-- training_backups: Fix admin + service role policy
DROP POLICY IF EXISTS "Admin full access to training_backups" ON public.training_backups;

CREATE POLICY "Training backups access" ON public.training_backups
    FOR ALL USING (
        (select auth.role()) = 'service_role'
        OR EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
    );

-- training_quarantine: Fix admin + service role policy
DROP POLICY IF EXISTS "Admin full access to training_quarantine" ON public.training_quarantine;

CREATE POLICY "Training quarantine access" ON public.training_quarantine
    FOR ALL USING (
        (select auth.role()) = 'service_role'
        OR EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
    );

-- learning_events: Fix service role policy
DROP POLICY IF EXISTS "Service role full access" ON public.learning_events;

CREATE POLICY "Learning events access" ON public.learning_events
    FOR ALL USING ((select auth.role()) = 'service_role');

-- security_logs: Fix service role + admin policy
DROP POLICY IF EXISTS "Service role and admins can access security logs" ON public.security_logs;

CREATE POLICY "Security logs access" ON public.security_logs
    FOR ALL USING (
        (select auth.role()) = 'service_role'
        OR EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
    );

-- ============================================================================
-- PART 2: Consolidate remaining duplicate SELECT policies
-- ============================================================================

-- saved_simulations: Consolidate public + owner SELECT policies
DROP POLICY IF EXISTS "Users can CRUD own simulations" ON public.saved_simulations;
DROP POLICY IF EXISTS "Public simulations are viewable" ON public.saved_simulations;

CREATE POLICY "Saved simulations read access" ON public.saved_simulations
    FOR SELECT USING (
        (select auth.uid()) = user_id
        OR is_public = true
    );

CREATE POLICY "Saved simulations write access" ON public.saved_simulations
    FOR ALL USING ((select auth.uid()) = user_id);

-- user_preferences: Consolidate user + admin SELECT policies
DROP POLICY IF EXISTS "Users can manage own preferences" ON public.user_preferences;
DROP POLICY IF EXISTS "Admins can view all preferences" ON public.user_preferences;

CREATE POLICY "User preferences read access" ON public.user_preferences
    FOR SELECT USING (
        (select auth.uid()) = user_id
        OR EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
    );

CREATE POLICY "User preferences write access" ON public.user_preferences
    FOR ALL USING ((select auth.uid()) = user_id);

-- user_sessions: Consolidate user + admin SELECT policies
DROP POLICY IF EXISTS "Users can view own sessions" ON public.user_sessions;
DROP POLICY IF EXISTS "Admins can view all sessions" ON public.user_sessions;

CREATE POLICY "User sessions access" ON public.user_sessions
    FOR SELECT USING (
        (select auth.uid()) = user_id
        OR EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
    );
