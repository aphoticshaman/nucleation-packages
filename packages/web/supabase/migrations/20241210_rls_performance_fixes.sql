-- ============================================================================
-- RLS Performance Fixes
-- ============================================================================
-- Fixes two categories of performance issues:
-- 1. auth_rls_initplan: Wrap auth.uid() in (select auth.uid()) for single evaluation
-- 2. multiple_permissive_policies: Consolidate duplicate policies
--
-- Run this AFTER backing up your database.
-- ============================================================================

-- ============================================================================
-- PART 1: Fix auth.uid() calls to use (select auth.uid())
-- ============================================================================
-- The fix pattern: Change `auth.uid()` to `(select auth.uid())`
-- This ensures the function is evaluated once per query, not per row.

-- profiles table
DROP POLICY IF EXISTS "Users can view own profile" ON public.profiles;
DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;
DROP POLICY IF EXISTS "Users can insert own profile" ON public.profiles;

CREATE POLICY "Users can view own profile" ON public.profiles
    FOR SELECT USING ((select auth.uid()) = id);

CREATE POLICY "Users can update own profile" ON public.profiles
    FOR UPDATE USING ((select auth.uid()) = id);

CREATE POLICY "Users can insert own profile" ON public.profiles
    FOR INSERT WITH CHECK ((select auth.uid()) = id);

-- clients table (uses `uuid` column, not `user_id`)
DROP POLICY IF EXISTS "Users can view own client" ON public.clients;
DROP POLICY IF EXISTS "Users can update own client" ON public.clients;

CREATE POLICY "Users can view own client" ON public.clients
    FOR SELECT USING ((select auth.uid()) = uuid);

CREATE POLICY "Users can update own client" ON public.clients
    FOR UPDATE USING ((select auth.uid()) = uuid);

-- feedback table (consolidate admin + user policies)
DROP POLICY IF EXISTS "Users can submit feedback" ON public.feedback;
DROP POLICY IF EXISTS "Users can view own feedback" ON public.feedback;
DROP POLICY IF EXISTS "Admins can view all feedback" ON public.feedback;
DROP POLICY IF EXISTS "Admins can update feedback" ON public.feedback;
DROP POLICY IF EXISTS "Admins can delete feedback" ON public.feedback;

CREATE POLICY "Users can submit feedback" ON public.feedback
    FOR INSERT WITH CHECK ((select auth.uid()) IS NOT NULL);

CREATE POLICY "Users and admins can view feedback" ON public.feedback
    FOR SELECT USING (
        (select auth.uid()) = user_id
        OR EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
    );

CREATE POLICY "Admins can update feedback" ON public.feedback
    FOR UPDATE USING (
        EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
    );

CREATE POLICY "Admins can delete feedback" ON public.feedback
    FOR DELETE USING (
        EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
    );

-- usage_records table
DROP POLICY IF EXISTS "Users can view own usage" ON public.usage_records;

CREATE POLICY "Users can view own usage" ON public.usage_records
    FOR SELECT USING ((select auth.uid()) = user_id);

-- alerts table
DROP POLICY IF EXISTS "Users can view own alerts" ON public.alerts;
DROP POLICY IF EXISTS "Users can update own alerts" ON public.alerts;

CREATE POLICY "Users can view own alerts" ON public.alerts
    FOR SELECT USING ((select auth.uid()) = user_id);

CREATE POLICY "Users can update own alerts" ON public.alerts
    FOR UPDATE USING ((select auth.uid()) = user_id);

-- alert_send_log table (consolidate duplicate policies)
DROP POLICY IF EXISTS "Users can view own alert logs" ON public.alert_send_log;
DROP POLICY IF EXISTS "Users can view own alert log" ON public.alert_send_log;
DROP POLICY IF EXISTS "Users can create own alert logs" ON public.alert_send_log;
DROP POLICY IF EXISTS "Service role full access to alert log" ON public.alert_send_log;

CREATE POLICY "Users can manage own alert logs" ON public.alert_send_log
    FOR ALL USING ((select auth.uid()) = user_id);

CREATE POLICY "Service role full access to alert log" ON public.alert_send_log
    FOR ALL USING (auth.role() = 'service_role');

-- webhooks table
DROP POLICY IF EXISTS "Users can manage own webhooks" ON public.webhooks;

CREATE POLICY "Users can manage own webhooks" ON public.webhooks
    FOR ALL USING ((select auth.uid()) = user_id);

-- tier_limits table (consolidate)
DROP POLICY IF EXISTS "Service role can manage tier_limits" ON public.tier_limits;
DROP POLICY IF EXISTS "Anyone can view tier limits" ON public.tier_limits;

CREATE POLICY "Anyone can view tier limits" ON public.tier_limits
    FOR SELECT USING (true);

CREATE POLICY "Service role can manage tier_limits" ON public.tier_limits
    FOR ALL USING (auth.role() = 'service_role');

-- signals_cache table (consolidate duplicate SELECT policies)
DROP POLICY IF EXISTS "Authenticated users can view signals cache" ON public.signals_cache;
DROP POLICY IF EXISTS "Authenticated users can read signals_cache" ON public.signals_cache;

CREATE POLICY "Authenticated users can view signals cache" ON public.signals_cache
    FOR SELECT USING ((select auth.uid()) IS NOT NULL);

-- country_signals table (consolidate)
DROP POLICY IF EXISTS "Service role can manage country signals" ON public.country_signals;
DROP POLICY IF EXISTS "Anyone can view country signals" ON public.country_signals;

CREATE POLICY "Anyone can view country signals" ON public.country_signals
    FOR SELECT USING (true);

CREATE POLICY "Service role can manage country signals" ON public.country_signals
    FOR ALL USING (auth.role() = 'service_role');

-- briefs table
DROP POLICY IF EXISTS "Authenticated users can view briefs" ON public.briefs;
DROP POLICY IF EXISTS "Service role can insert briefs" ON public.briefs;

CREATE POLICY "Authenticated users can view briefs" ON public.briefs
    FOR SELECT USING ((select auth.uid()) IS NOT NULL);

CREATE POLICY "Service role can insert briefs" ON public.briefs
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

-- email_export_log table
DROP POLICY IF EXISTS "Users can view own email exports" ON public.email_export_log;
DROP POLICY IF EXISTS "Users can create own email exports" ON public.email_export_log;

CREATE POLICY "Users can manage own email exports" ON public.email_export_log
    FOR ALL USING ((select auth.uid()) = user_id);

-- simulations table
DROP POLICY IF EXISTS "Users can view their own simulations" ON public.simulations;
DROP POLICY IF EXISTS "Users can create simulations" ON public.simulations;
DROP POLICY IF EXISTS "Users can update their own simulations" ON public.simulations;

CREATE POLICY "Users can manage own simulations" ON public.simulations
    FOR ALL USING ((select auth.uid()) = user_id);

-- api_keys table (consolidate admin + org member policies)
DROP POLICY IF EXISTS "Org members can delete their API keys" ON public.api_keys;
DROP POLICY IF EXISTS "Admins can manage all API keys" ON public.api_keys;
DROP POLICY IF EXISTS "Org members can view their API keys" ON public.api_keys;
DROP POLICY IF EXISTS "Org members can create API keys" ON public.api_keys;

CREATE POLICY "Users can manage org API keys" ON public.api_keys
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM public.profiles
            WHERE id = (select auth.uid())
            AND (role = 'admin' OR organization_id = api_keys.organization_id)
        )
    );

-- organizations table (consolidate)
DROP POLICY IF EXISTS "Org members can view their org" ON public.organizations;
DROP POLICY IF EXISTS "Admins can view all orgs" ON public.organizations;
DROP POLICY IF EXISTS "Users can view own organization" ON public.organizations;
DROP POLICY IF EXISTS "Users can update own organization" ON public.organizations;

CREATE POLICY "Users can view organizations" ON public.organizations
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.profiles
            WHERE id = (select auth.uid())
            AND (role = 'admin' OR organization_id = organizations.id)
        )
    );

CREATE POLICY "Users can update own organization" ON public.organizations
    FOR UPDATE USING (
        EXISTS (
            SELECT 1 FROM public.profiles
            WHERE id = (select auth.uid())
            AND organization_id = organizations.id
        )
    );

-- saved_simulations table (consolidate)
DROP POLICY IF EXISTS "Users can CRUD own simulations" ON public.saved_simulations;
DROP POLICY IF EXISTS "Public simulations are viewable" ON public.saved_simulations;

CREATE POLICY "Users can CRUD own simulations" ON public.saved_simulations
    FOR ALL USING ((select auth.uid()) = user_id);

CREATE POLICY "Public simulations are viewable" ON public.saved_simulations
    FOR SELECT USING (is_public = true);

-- trial_invites table
DROP POLICY IF EXISTS "Admins can manage trial invites" ON public.trial_invites;

CREATE POLICY "Admins can manage trial invites" ON public.trial_invites
    FOR ALL USING (
        EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
    );

-- user_preferences table (consolidate)
DROP POLICY IF EXISTS "Users can view own preferences" ON public.user_preferences;
DROP POLICY IF EXISTS "Users can insert own preferences" ON public.user_preferences;
DROP POLICY IF EXISTS "Users can delete own preferences" ON public.user_preferences;
DROP POLICY IF EXISTS "Users can update own preferences" ON public.user_preferences;
DROP POLICY IF EXISTS "Admins can view all preferences" ON public.user_preferences;

CREATE POLICY "Users can manage own preferences" ON public.user_preferences
    FOR ALL USING ((select auth.uid()) = user_id);

CREATE POLICY "Admins can view all preferences" ON public.user_preferences
    FOR SELECT USING (
        EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
    );

-- email_export_preferences table (consolidate heavily duplicated policies)
DROP POLICY IF EXISTS "Users can view own email preferences" ON public.email_export_preferences;
DROP POLICY IF EXISTS "Users can update own email preferences" ON public.email_export_preferences;
DROP POLICY IF EXISTS "Users can insert own email preferences" ON public.email_export_preferences;
DROP POLICY IF EXISTS "Service role full access to email preferences" ON public.email_export_preferences;
DROP POLICY IF EXISTS "Users can manage own email prefs" ON public.email_export_preferences;

CREATE POLICY "Users can manage own email preferences" ON public.email_export_preferences
    FOR ALL USING ((select auth.uid()) = user_id);

CREATE POLICY "Service role full access to email preferences" ON public.email_export_preferences
    FOR ALL USING (auth.role() = 'service_role');

-- user_sessions table (consolidate)
DROP POLICY IF EXISTS "Users can view own sessions" ON public.user_sessions;
DROP POLICY IF EXISTS "Admins can view all sessions" ON public.user_sessions;

CREATE POLICY "Users can view own sessions" ON public.user_sessions
    FOR SELECT USING ((select auth.uid()) = user_id);

CREATE POLICY "Admins can view all sessions" ON public.user_sessions
    FOR SELECT USING (
        EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
    );

-- briefing_cache table (consolidate)
DROP POLICY IF EXISTS "Service role can manage cache" ON public.briefing_cache;
DROP POLICY IF EXISTS "Anyone can read briefing cache" ON public.briefing_cache;

CREATE POLICY "Anyone can read briefing cache" ON public.briefing_cache
    FOR SELECT USING (true);

CREATE POLICY "Service role can manage cache" ON public.briefing_cache
    FOR ALL USING (auth.role() = 'service_role');

-- training_examples table
DROP POLICY IF EXISTS "Admin full access to training_examples" ON public.training_examples;

CREATE POLICY "Admin full access to training_examples" ON public.training_examples
    FOR ALL USING (
        EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
        OR auth.role() = 'service_role'
    );

-- nation_changes table
DROP POLICY IF EXISTS "Admin full access to nation_changes" ON public.nation_changes;

CREATE POLICY "Admin full access to nation_changes" ON public.nation_changes
    FOR ALL USING (
        EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
        OR auth.role() = 'service_role'
    );

-- training_backups table
DROP POLICY IF EXISTS "Admin full access to training_backups" ON public.training_backups;

CREATE POLICY "Admin full access to training_backups" ON public.training_backups
    FOR ALL USING (
        EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
        OR auth.role() = 'service_role'
    );

-- training_quarantine table
DROP POLICY IF EXISTS "Admin full access to training_quarantine" ON public.training_quarantine;

CREATE POLICY "Admin full access to training_quarantine" ON public.training_quarantine
    FOR ALL USING (
        EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
        OR auth.role() = 'service_role'
    );

-- learning_events table
DROP POLICY IF EXISTS "Service role full access" ON public.learning_events;

CREATE POLICY "Service role full access" ON public.learning_events
    FOR ALL USING (auth.role() = 'service_role');

-- security_logs table (consolidate)
DROP POLICY IF EXISTS "Service role full access" ON public.security_logs;
DROP POLICY IF EXISTS "Admins can read security logs" ON public.security_logs;

CREATE POLICY "Service role and admins can access security logs" ON public.security_logs
    FOR ALL USING (
        auth.role() = 'service_role'
        OR EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
    );

-- reasoning_traces table
DROP POLICY IF EXISTS "Service role full access" ON public.reasoning_traces;

CREATE POLICY "Service role full access" ON public.reasoning_traces
    FOR ALL USING (auth.role() = 'service_role');

-- prediction_outcomes table
DROP POLICY IF EXISTS "Service role full access" ON public.prediction_outcomes;

CREATE POLICY "Service role full access" ON public.prediction_outcomes
    FOR ALL USING (auth.role() = 'service_role');

-- historical_cases table (consolidate)
DROP POLICY IF EXISTS "Service role full access" ON public.historical_cases;
DROP POLICY IF EXISTS "Admins can manage historical cases" ON public.historical_cases;

CREATE POLICY "Service role and admins can manage historical cases" ON public.historical_cases
    FOR ALL USING (
        auth.role() = 'service_role'
        OR EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
    );

-- rate_limits table
DROP POLICY IF EXISTS "Service role full access" ON public.rate_limits;

CREATE POLICY "Service role full access" ON public.rate_limits
    FOR ALL USING (auth.role() = 'service_role');

-- training_batches table
DROP POLICY IF EXISTS "Service role full access" ON public.training_batches;

CREATE POLICY "Service role full access" ON public.training_batches
    FOR ALL USING (auth.role() = 'service_role');

-- ============================================================================
-- VERIFICATION
-- ============================================================================
-- Run these queries to verify the fixes:
--
-- Check for remaining initplan issues:
-- SELECT * FROM pg_policies WHERE definition LIKE '%auth.uid()%' AND definition NOT LIKE '%(select auth.uid())%';
--
-- Check for multiple permissive policies:
-- SELECT schemaname, tablename, policyname, roles, cmd
-- FROM pg_policies
-- WHERE schemaname = 'public'
-- ORDER BY tablename, cmd, roles;
-- ============================================================================
