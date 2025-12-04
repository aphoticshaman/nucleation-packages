-- Fix RLS policies for profiles table
-- This ensures users can read their own profile

-- First, check if RLS is enabled (it should be)
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if any (to recreate clean)
DROP POLICY IF EXISTS "Users can view own profile" ON public.profiles;
DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;
DROP POLICY IF EXISTS "Users can insert own profile" ON public.profiles;
DROP POLICY IF EXISTS "Admins can view all profiles" ON public.profiles;
DROP POLICY IF EXISTS "Service role bypasses RLS" ON public.profiles;

-- Policy: Users can read their own profile
CREATE POLICY "Users can view own profile"
ON public.profiles
FOR SELECT
USING (auth.uid() = id);

-- Policy: Users can update their own profile
CREATE POLICY "Users can update own profile"
ON public.profiles
FOR UPDATE
USING (auth.uid() = id)
WITH CHECK (auth.uid() = id);

-- Policy: Users can insert their own profile (for new signups)
CREATE POLICY "Users can insert own profile"
ON public.profiles
FOR INSERT
WITH CHECK (auth.uid() = id);

-- Policy: Admins can view all profiles (for admin dashboard)
-- This uses a subquery to check if the current user is an admin
CREATE POLICY "Admins can view all profiles"
ON public.profiles
FOR SELECT
USING (
  EXISTS (
    SELECT 1 FROM public.profiles
    WHERE id = auth.uid() AND role = 'admin'
  )
);

-- Policy: Admins can update all profiles
CREATE POLICY "Admins can update all profiles"
ON public.profiles
FOR UPDATE
USING (
  EXISTS (
    SELECT 1 FROM public.profiles
    WHERE id = auth.uid() AND role = 'admin'
  )
);

-- Fix user_activity table (from linter warning)
DROP POLICY IF EXISTS "Users can insert own activity" ON public.user_activity;
DROP POLICY IF EXISTS "Users can view own activity" ON public.user_activity;

CREATE POLICY "Users can insert own activity"
ON public.user_activity
FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view own activity"
ON public.user_activity
FOR SELECT
USING (auth.uid() = user_id);

-- Fix api_usage table (from linter warning)
DROP POLICY IF EXISTS "Users can view own api usage" ON public.api_usage;

CREATE POLICY "Users can view own api usage"
ON public.api_usage
FOR SELECT
USING (auth.uid() = user_id);

-- Fix simulation_snapshots table (from linter warning)
DROP POLICY IF EXISTS "Users can manage own snapshots" ON public.simulation_snapshots;

CREATE POLICY "Users can manage own snapshots"
ON public.simulation_snapshots
FOR ALL
USING (auth.uid() = user_id)
WITH CHECK (auth.uid() = user_id);

-- Grant necessary permissions
GRANT SELECT, UPDATE ON public.profiles TO authenticated;
GRANT INSERT ON public.profiles TO authenticated;
GRANT SELECT, INSERT ON public.user_activity TO authenticated;
GRANT SELECT ON public.api_usage TO authenticated;
GRANT ALL ON public.simulation_snapshots TO authenticated;
