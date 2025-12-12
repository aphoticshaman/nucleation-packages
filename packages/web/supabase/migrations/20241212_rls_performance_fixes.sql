-- RLS Performance Fixes
--
-- 1. Wrap auth.uid() in (select ...) to avoid per-row re-evaluation
-- 2. Consolidate multiple permissive policies into single policies

-- =============================================================================
-- INSIGHT_REPORTS - Consolidate and optimize policies
-- =============================================================================

DROP POLICY IF EXISTS "Admins can view all insights" ON insight_reports;
DROP POLICY IF EXISTS "Admins can modify insights" ON insight_reports;
DROP POLICY IF EXISTS "Service role full access" ON insight_reports;

-- Single combined policy for all access
CREATE POLICY "insight_reports_access" ON insight_reports
    FOR ALL USING (
        -- Service role has full access
        (select auth.jwt()->>'role') = 'service_role'
        OR
        -- Admins have full access
        EXISTS (
            SELECT 1 FROM profiles
            WHERE profiles.id = (select auth.uid())
            AND profiles.role = 'admin'
        )
    );

-- =============================================================================
-- STUDY_CONVERSATIONS - Optimize auth calls
-- =============================================================================

DROP POLICY IF EXISTS study_conversations_user_policy ON study_conversations;

CREATE POLICY study_conversations_user_policy ON study_conversations
    FOR ALL USING (
        user_id = (select auth.uid())
        OR EXISTS (
            SELECT 1 FROM profiles
            WHERE id = (select auth.uid())
            AND role = 'admin'
        )
    );

-- =============================================================================
-- STUDY_MESSAGES - Optimize auth calls
-- =============================================================================

DROP POLICY IF EXISTS study_messages_user_policy ON study_messages;

CREATE POLICY study_messages_user_policy ON study_messages
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM study_conversations sc
            WHERE sc.id = study_messages.conversation_id
            AND (
                sc.user_id = (select auth.uid())
                OR EXISTS (
                    SELECT 1 FROM profiles
                    WHERE id = (select auth.uid())
                    AND role = 'admin'
                )
            )
        )
    );

-- =============================================================================
-- ELLE_INTERACTIONS - Consolidate and optimize policies
-- =============================================================================

DROP POLICY IF EXISTS elle_interactions_admin_policy ON elle_interactions;
DROP POLICY IF EXISTS elle_interactions_service_policy ON elle_interactions;

-- Single combined policy
CREATE POLICY elle_interactions_access ON elle_interactions
    FOR ALL USING (
        -- Service role has full access
        (select auth.jwt()->>'role') = 'service_role'
        OR
        -- Admins have full access
        EXISTS (
            SELECT 1 FROM profiles
            WHERE profiles.id = (select auth.uid())
            AND profiles.role = 'admin'
        )
    );

-- =============================================================================
-- ELLE_TRAINING_EXAMPLES - Consolidate and optimize policies
-- =============================================================================

DROP POLICY IF EXISTS elle_training_service_policy ON elle_training_examples;
DROP POLICY IF EXISTS elle_training_admin_policy ON elle_training_examples;

-- Single combined policy
CREATE POLICY elle_training_examples_access ON elle_training_examples
    FOR ALL USING (
        -- Service role has full access
        (select auth.jwt()->>'role') = 'service_role'
        OR
        -- Admins have full access
        EXISTS (
            SELECT 1 FROM profiles
            WHERE profiles.id = (select auth.uid())
            AND profiles.role = 'admin'
        )
    );

-- =============================================================================
-- SAVED_SIMULATIONS - Optimize auth calls
-- =============================================================================

DROP POLICY IF EXISTS saved_simulations_policy ON saved_simulations;

CREATE POLICY saved_simulations_policy ON saved_simulations
    FOR ALL USING (
        user_id = (select auth.uid())
    );

-- =============================================================================
-- USER_PREFERENCES - Optimize auth calls
-- =============================================================================

DROP POLICY IF EXISTS user_preferences_policy ON user_preferences;

CREATE POLICY user_preferences_policy ON user_preferences
    FOR ALL USING (
        user_id = (select auth.uid())
    );

-- =============================================================================
-- TRAINING_ITEMS - Optimize auth calls
-- =============================================================================

DROP POLICY IF EXISTS training_items_admin ON training_items;

CREATE POLICY training_items_admin ON training_items
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE profiles.id = (select auth.uid())
            AND profiles.role = 'admin'
        )
    );

-- =============================================================================
-- TRAINING_EXPORT_BATCHES - Optimize auth calls
-- =============================================================================

DROP POLICY IF EXISTS training_batches_admin ON training_export_batches;

CREATE POLICY training_batches_admin ON training_export_batches
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE profiles.id = (select auth.uid())
            AND profiles.role = 'admin'
        )
    );

-- =============================================================================
-- TRAINING_ITEM_EXPORTS - Optimize auth calls
-- =============================================================================

DROP POLICY IF EXISTS training_item_exports_admin ON training_item_exports;

CREATE POLICY training_item_exports_admin ON training_item_exports
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE profiles.id = (select auth.uid())
            AND profiles.role = 'admin'
        )
    );

-- =============================================================================
-- TRAINING_AUDIT_LOG - Optimize auth calls
-- =============================================================================

DROP POLICY IF EXISTS training_audit_insert ON training_audit_log;
DROP POLICY IF EXISTS training_audit_select ON training_audit_log;

-- Combined policy for select/insert
CREATE POLICY training_audit_access ON training_audit_log
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE profiles.id = (select auth.uid())
            AND profiles.role = 'admin'
        )
    );
