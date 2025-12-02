import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

// Validate an invite token (public endpoint)
export async function GET(req: Request, { params }: { params: Promise<{ token: string }> }) {
  try {
    const { token } = await params;

    if (!token) {
      return NextResponse.json({ valid: false, error: 'Token required' }, { status: 400 });
    }

    const supabaseAdmin = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );

    // Find the invite
    const { data: invite, error } = await supabaseAdmin
      .from('trial_invites')
      .select('*')
      .eq('token', token)
      .single();

    if (error || !invite) {
      return NextResponse.json({ valid: false, error: 'Invalid invite token' });
    }

    // Check if already used
    if (invite.status === 'accepted') {
      return NextResponse.json({ valid: false, error: 'Invite has already been used' });
    }

    // Check if expired
    if (new Date(invite.expires_at) < new Date()) {
      await supabaseAdmin.from('trial_invites').update({ status: 'expired' }).eq('token', token);

      return NextResponse.json({ valid: false, error: 'Invite has expired' });
    }

    // Valid invite
    return NextResponse.json({
      valid: true,
      email: invite.email,
      trialDays: invite.trial_days,
    });
  } catch (error) {
    console.error('Invite validation error:', error);
    return NextResponse.json({ valid: false, error: 'Validation failed' }, { status: 500 });
  }
}
