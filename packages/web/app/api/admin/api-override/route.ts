import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { createClient } from '@supabase/supabase-js';
import {
  setAdminOverride,
  clearAdminOverride,
  getAdminOverrideRemaining,
  isLFBMEnabled,
} from '@/lib/inference/LFBMClient';

// =============================================================================
// ADMIN API OVERRIDE - ARM & FIRE mechanism
// =============================================================================
// Allows admins to temporarily enable RunPod/LFBM APIs in non-production
// environments with a timed expiry.
//
// POST /api/admin/api-override
// Body: { action: 'arm' | 'fire' | 'disarm', minutes?: number }
//
// ARM: Prime the system for override (returns confirmation)
// FIRE: Actually enable the override with timer
// DISARM: Clear any active override

// Verify admin auth
async function verifyAdminAuth(): Promise<{ isAdmin: boolean; userId?: string; error?: string }> {
  try {
    const cookieStore = await cookies();
    const authClient = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          getAll() {
            return cookieStore.getAll();
          },
          setAll() {
            // Read-only
          },
        },
      }
    );

    const { data: { user }, error: authError } = await authClient.auth.getUser();
    if (authError || !user) {
      return { isAdmin: false, error: 'Authentication required' };
    }

    // Check admin role using service client (bypasses RLS)
    const db = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );

    const { data: profile, error: profileError } = await db
      .from('profiles')
      .select('role')
      .eq('id', user.id)
      .single();

    if (profileError || !profile) {
      return { isAdmin: false, userId: user.id, error: 'Profile not found' };
    }

    return { isAdmin: profile.role === 'admin', userId: user.id };
  } catch {
    return { isAdmin: false, error: 'Auth check failed' };
  }
}

// In-memory ARM state (per serverless instance)
// This is intentionally ephemeral - forces admin to ARM before FIRE
let isArmed = false;
let armedAt: number | null = null;
const ARM_TIMEOUT_MS = 60 * 1000; // 60 seconds to FIRE after arming

export async function POST(request: Request) {
  try {
    // SECURITY: Admin only
    const auth = await verifyAdminAuth();
    if (!auth.isAdmin) {
      return NextResponse.json(
        { error: auth.error || 'Admin access required' },
        { status: 403 }
      );
    }

    const body = await request.json();
    const action = body.action as string;
    const minutes = body.minutes as number | undefined;

  switch (action) {
    case 'arm': {
      // ARM the system - admin is preparing to enable
      isArmed = true;
      armedAt = Date.now();

      console.log(`[API-OVERRIDE] System ARMED by admin ${auth.userId}`);

      return NextResponse.json({
        success: true,
        action: 'arm',
        status: 'ARMED',
        message: 'System armed. You have 60 seconds to FIRE to enable APIs.',
        armExpiresIn: ARM_TIMEOUT_MS / 1000,
        currentState: {
          lfbmEnabled: isLFBMEnabled(),
          override: getAdminOverrideRemaining(),
          environment: process.env.LF_PROD_ENABLE === 'true' ? 'production' : 'non-production',
        },
      });
    }

    case 'fire': {
      // FIRE - actually enable the override (must be armed first)
      if (!isArmed || !armedAt) {
        return NextResponse.json({
          success: false,
          error: 'System not armed. ARM first, then FIRE within 60 seconds.',
        }, { status: 400 });
      }

      // Check if ARM has expired
      if (Date.now() - armedAt > ARM_TIMEOUT_MS) {
        isArmed = false;
        armedAt = null;
        return NextResponse.json({
          success: false,
          error: 'ARM expired. Please ARM again, then FIRE within 60 seconds.',
        }, { status: 400 });
      }

      // FIRE! Enable the override
      const overrideMinutes = minutes || 30; // Default 30 minutes
      const result = setAdminOverride(overrideMinutes);

      // Reset ARM state
      isArmed = false;
      armedAt = null;

      console.log(`[API-OVERRIDE] APIs ENABLED for ${overrideMinutes} minutes by admin ${auth.userId}`);

      return NextResponse.json({
        success: true,
        action: 'fire',
        status: 'ENABLED',
        message: `APIs enabled for ${overrideMinutes} minutes`,
        override: {
          expiresAt: result.expiresAt,
          durationMinutes: result.durationMinutes,
        },
        currentState: {
          lfbmEnabled: true,
          override: getAdminOverrideRemaining(),
          environment: process.env.LF_PROD_ENABLE === 'true' ? 'production' : 'non-production',
        },
      });
    }

    case 'disarm': {
      // DISARM - clear any active override
      isArmed = false;
      armedAt = null;
      clearAdminOverride();

      console.log(`[API-OVERRIDE] Override CLEARED by admin ${auth.userId}`);

      return NextResponse.json({
        success: true,
        action: 'disarm',
        status: 'DISABLED',
        message: 'API override cleared. APIs will use default behavior.',
        currentState: {
          lfbmEnabled: isLFBMEnabled(),
          override: getAdminOverrideRemaining(),
          environment: process.env.LF_PROD_ENABLE === 'true' ? 'production' : 'non-production',
        },
      });
    }

    case 'status': {
      // Just return current status
      const armExpired = armedAt && (Date.now() - armedAt > ARM_TIMEOUT_MS);
      if (armExpired) {
        isArmed = false;
        armedAt = null;
      }

      return NextResponse.json({
        success: true,
        action: 'status',
        armed: isArmed,
        armedSecondsRemaining: isArmed && armedAt
          ? Math.max(0, Math.round((ARM_TIMEOUT_MS - (Date.now() - armedAt)) / 1000))
          : 0,
        currentState: {
          lfbmEnabled: isLFBMEnabled(),
          override: getAdminOverrideRemaining(),
          environment: process.env.LF_PROD_ENABLE === 'true' ? 'production' : 'non-production',
          lfProdEnable: process.env.LF_PROD_ENABLE,
        },
      });
    }

    default:
      return NextResponse.json({
        error: 'Invalid action. Use: arm, fire, disarm, or status',
      }, { status: 400 });
  }
  } catch (error) {
    console.error('[API-OVERRIDE] POST error:', error);
    return NextResponse.json({
      error: 'Internal server error',
      details: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}

// GET endpoint for status check
export async function GET(_request: Request) {
  try {
    // SECURITY: Admin only for status check
    const auth = await verifyAdminAuth();
    if (!auth.isAdmin) {
      return NextResponse.json(
        { error: auth.error || 'Admin access required' },
        { status: 403 }
      );
    }

    const armExpired = armedAt && (Date.now() - armedAt > ARM_TIMEOUT_MS);
    if (armExpired) {
      isArmed = false;
      armedAt = null;
    }

    return NextResponse.json({
      armed: isArmed,
      armedSecondsRemaining: isArmed && armedAt
        ? Math.max(0, Math.round((ARM_TIMEOUT_MS - (Date.now() - armedAt)) / 1000))
        : 0,
      currentState: {
        lfbmEnabled: isLFBMEnabled(),
        override: getAdminOverrideRemaining(),
        environment: process.env.LF_PROD_ENABLE === 'true' ? 'production' : 'non-production',
        lfProdEnable: process.env.LF_PROD_ENABLE,
      },
    });
  } catch (error) {
    console.error('[API-OVERRIDE] GET error:', error);
    return NextResponse.json({
      error: 'Internal server error',
      details: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}
