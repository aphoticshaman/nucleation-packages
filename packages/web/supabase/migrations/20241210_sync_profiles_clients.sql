-- ============================================================================
-- Sync Profiles and Clients Tables
-- ============================================================================
-- Problem: profiles and clients tables are mismatched - no auto-sync
-- Solution:
--   1. Backfill missing clients for existing profiles
--   2. Add trigger to auto-create client when profile is created
--   3. Add trigger to cascade profile updates to client
-- ============================================================================

-- ============================================================================
-- PART 1: Backfill - Create clients for profiles that don't have one
-- ============================================================================

INSERT INTO public.clients (user_id, name, email, tier, created_at)
SELECT
    p.id as user_id,
    COALESCE(p.full_name, split_part(p.email, '@', 1)) as name,
    p.email,
    'free' as tier,
    p.created_at
FROM public.profiles p
LEFT JOIN public.clients c ON c.user_id = p.id
WHERE c.id IS NULL
ON CONFLICT (email) DO NOTHING;

-- ============================================================================
-- PART 2: Trigger to auto-create client when profile is created
-- ============================================================================

CREATE OR REPLACE FUNCTION sync_profile_to_client()
RETURNS TRIGGER AS $$
BEGIN
    -- On INSERT: Create corresponding client
    IF TG_OP = 'INSERT' THEN
        INSERT INTO public.clients (user_id, name, email, tier, created_at)
        VALUES (
            NEW.id,
            COALESCE(NEW.full_name, split_part(NEW.email, '@', 1)),
            NEW.email,
            'free',
            NEW.created_at
        )
        ON CONFLICT (email) DO UPDATE SET
            user_id = EXCLUDED.user_id,
            name = EXCLUDED.name;
        RETURN NEW;
    END IF;

    -- On UPDATE: Sync changes to client
    IF TG_OP = 'UPDATE' THEN
        UPDATE public.clients
        SET
            name = COALESCE(NEW.full_name, split_part(NEW.email, '@', 1)),
            email = NEW.email,
            updated_at = NOW()
        WHERE user_id = NEW.id;
        RETURN NEW;
    END IF;

    -- On DELETE: Client cascades via FK (if configured) or delete here
    IF TG_OP = 'DELETE' THEN
        DELETE FROM public.clients WHERE user_id = OLD.id;
        RETURN OLD;
    END IF;

    RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Drop existing trigger if any
DROP TRIGGER IF EXISTS sync_profile_to_client_trigger ON public.profiles;

-- Create trigger for all operations
CREATE TRIGGER sync_profile_to_client_trigger
    AFTER INSERT OR UPDATE OR DELETE ON public.profiles
    FOR EACH ROW EXECUTE FUNCTION sync_profile_to_client();

-- ============================================================================
-- PART 3: Cleanup orphaned clients (clients without profiles)
-- ============================================================================
-- Optional: Remove clients that have no corresponding profile
-- Uncomment if you want to clean up orphans:

-- DELETE FROM public.clients c
-- WHERE NOT EXISTS (
--     SELECT 1 FROM public.profiles p WHERE p.id = c.user_id
-- );

-- ============================================================================
-- Verification queries (run manually to check results)
-- ============================================================================
--
-- -- Check all profiles have clients:
-- SELECT p.id, p.email, c.id as client_id
-- FROM profiles p
-- LEFT JOIN clients c ON c.user_id = p.id;
--
-- -- Check for orphaned clients:
-- SELECT c.id, c.email
-- FROM clients c
-- LEFT JOIN profiles p ON p.id = c.user_id
-- WHERE p.id IS NULL;
