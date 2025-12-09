-- ============================================================================
-- Foreign Key Indexes + Primary Key Fixes
-- ============================================================================
-- Addresses Supabase linter INFO warnings:
-- 1. unindexed_foreign_keys: Add indexes for FK columns used in JOINs
-- 2. no_primary_key: Add PK to tier_limits table
-- ============================================================================

-- ============================================================================
-- PART 1: Add indexes for unindexed foreign keys
-- ============================================================================

-- api_keys
CREATE INDEX IF NOT EXISTS idx_api_keys_created_by ON public.api_keys(created_by);
CREATE INDEX IF NOT EXISTS idx_api_keys_organization_id ON public.api_keys(organization_id);

-- esteem_relations
CREATE INDEX IF NOT EXISTS idx_esteem_relations_target_id ON public.esteem_relations(target_id);

-- influence_edges
CREATE INDEX IF NOT EXISTS idx_influence_edges_target_id ON public.influence_edges(target_id);

-- nation_changes
CREATE INDEX IF NOT EXISTS idx_nation_changes_created_by ON public.nation_changes(created_by);
CREATE INDEX IF NOT EXISTS idx_nation_changes_verified_by ON public.nation_changes(verified_by);

-- profiles
CREATE INDEX IF NOT EXISTS idx_profiles_organization_id ON public.profiles(organization_id);

-- saved_simulations
CREATE INDEX IF NOT EXISTS idx_saved_simulations_user_id ON public.saved_simulations(user_id);

-- training_backups
CREATE INDEX IF NOT EXISTS idx_training_backups_restored_by ON public.training_backups(restored_by);

-- training_quarantine
CREATE INDEX IF NOT EXISTS idx_training_quarantine_quarantined_by ON public.training_quarantine(quarantined_by);
CREATE INDEX IF NOT EXISTS idx_training_quarantine_resolved_by ON public.training_quarantine(resolved_by);

-- trial_invites
CREATE INDEX IF NOT EXISTS idx_trial_invites_accepted_by ON public.trial_invites(accepted_by);
CREATE INDEX IF NOT EXISTS idx_trial_invites_invited_by ON public.trial_invites(invited_by);

-- ============================================================================
-- PART 2: Add primary key to tier_limits
-- ============================================================================
-- tier_limits likely uses (tier) or (tier, limit_type) as natural key
-- Adding a synthetic PK is safest

-- First check if PK already exists, if not add one
DO $$
BEGIN
    -- Check if tier_limits has a primary key
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE table_name = 'tier_limits'
        AND table_schema = 'public'
        AND constraint_type = 'PRIMARY KEY'
    ) THEN
        -- Try to add id column if it doesn't exist
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'tier_limits'
            AND table_schema = 'public'
            AND column_name = 'id'
        ) THEN
            ALTER TABLE public.tier_limits ADD COLUMN id UUID DEFAULT uuid_generate_v4();
        END IF;

        -- Add primary key constraint
        ALTER TABLE public.tier_limits ADD PRIMARY KEY (id);
    END IF;
END $$;
