import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { Resend } from 'resend';

export const runtime = 'edge';
export const maxDuration = 60; // 60 second timeout for batch processing

// Verify this is a legitimate cron call
function verifyCronAuth(request: Request): boolean {
  const authHeader = request.headers.get('authorization');
  if (authHeader === `Bearer ${process.env.CRON_SECRET}`) {
    return true;
  }
  // Vercel cron jobs also set this header
  const vercelCron = request.headers.get('x-vercel-cron');
  if (vercelCron === '1') {
    return true;
  }
  return false;
}

interface UserAlertPrefs {
  user_id: string;
  email: string;
  full_name: string | null;
  frequency: 'daily' | 'weekly' | 'on_demand';
  preferred_time: string | null;
  include_global: boolean;
  include_watchlist: boolean;
  format: 'summary' | 'detailed';
  plan: string;
}

export async function GET(request: Request) {
  const startTime = Date.now();

  // Security check
  if (!verifyCronAuth(request)) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  // Initialize clients
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const resendKey = process.env.RESEND_API_KEY;

  if (!supabaseUrl || !supabaseServiceKey) {
    console.error('[DAILY-ALERTS] Missing Supabase credentials');
    return NextResponse.json({ error: 'Server misconfigured' }, { status: 500 });
  }

  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  // Check if Resend is configured
  const resend = resendKey ? new Resend(resendKey) : null;
  if (!resend) {
    console.warn('[DAILY-ALERTS] Resend not configured - emails will be logged only');
  }

  try {
    // Get current hour (UTC) to match preferred_time
    const currentHour = new Date().getUTCHours();
    const currentDay = new Date().getUTCDay(); // 0 = Sunday

    // Fetch users who want daily alerts at this hour
    // Also include users with no preference (default to 8am UTC)
    const { data: dailyUsers, error: dailyError } = await supabase
      .from('email_export_preferences')
      .select(`
        user_id,
        frequency,
        preferred_time,
        include_global,
        include_watchlist,
        format
      `)
      .eq('frequency', 'daily')
      .eq('enabled', true);

    if (dailyError) {
      console.error('[DAILY-ALERTS] Error fetching preferences:', dailyError);
      throw dailyError;
    }

    // Fetch weekly users (only on Sunday)
    let weeklyUsers: typeof dailyUsers = [];
    if (currentDay === 0) {
      const { data: weekly, error: weeklyError } = await supabase
        .from('email_export_preferences')
        .select(`
          user_id,
          frequency,
          preferred_time,
          include_global,
          include_watchlist,
          format
        `)
        .eq('frequency', 'weekly')
        .eq('enabled', true);

      if (!weeklyError && weekly) {
        weeklyUsers = weekly;
      }
    }

    const allUsers = [...(dailyUsers || []), ...weeklyUsers];

    // Filter by preferred time (default 8am UTC if not set)
    const usersToNotify = allUsers.filter((user) => {
      const preferredHour = user.preferred_time
        ? parseInt(user.preferred_time.split(':')[0], 10)
        : 8;
      return preferredHour === currentHour;
    });

    if (usersToNotify.length === 0) {
      console.log(`[DAILY-ALERTS] No users to notify at hour ${currentHour} UTC`);
      return NextResponse.json({
        success: true,
        usersProcessed: 0,
        message: `No users scheduled for hour ${currentHour} UTC`,
      });
    }

    // Get user emails from profiles
    const userIds = usersToNotify.map((u) => u.user_id);
    const { data: profiles } = await supabase
      .from('profiles')
      .select('id, email, full_name, plan')
      .in('id', userIds);

    const profileMap = new Map(profiles?.map((p) => [p.id, p]) || []);

    // Fetch the latest briefing data
    const { data: latestBriefing } = await supabase
      .from('briefing_cache')
      .select('data, created_at')
      .eq('preset', 'global')
      .order('created_at', { ascending: false })
      .limit(1)
      .single();

    // Send emails
    let sent = 0;
    let failed = 0;
    const errors: string[] = [];

    for (const userPref of usersToNotify) {
      const profile = profileMap.get(userPref.user_id);
      if (!profile?.email) {
        console.warn(`[DAILY-ALERTS] No email for user ${userPref.user_id}`);
        continue;
      }

      // Skip free users (they don't get email alerts)
      if (profile.plan === 'free') {
        continue;
      }

      try {
        const emailContent = buildAlertEmail(
          profile.full_name || 'Intelligence Analyst',
          userPref.format,
          latestBriefing?.data,
          userPref.frequency
        );

        if (resend) {
          await resend.emails.send({
            from: 'LatticeForge Alerts <alerts@latticeforge.ai>',
            to: profile.email,
            subject: emailContent.subject,
            html: emailContent.html,
          });
          sent++;
        } else {
          // Log-only mode
          console.log(`[DAILY-ALERTS] Would send to ${profile.email}:`, emailContent.subject);
          sent++;
        }

        // Log the send
        await supabase.from('alert_send_log').insert({
          user_id: userPref.user_id,
          alert_type: userPref.frequency === 'weekly' ? 'weekly_digest' : 'daily_briefing',
          sent_at: new Date().toISOString(),
        });
      } catch (emailErr) {
        console.error(`[DAILY-ALERTS] Failed to send to ${profile.email}:`, emailErr);
        failed++;
        errors.push(`${profile.email}: ${emailErr}`);
      }
    }

    const duration = Date.now() - startTime;
    console.log(`[DAILY-ALERTS] Completed: ${sent} sent, ${failed} failed in ${duration}ms`);

    return NextResponse.json({
      success: true,
      usersProcessed: usersToNotify.length,
      sent,
      failed,
      errors: errors.length > 0 ? errors : undefined,
      durationMs: duration,
    });
  } catch (error) {
    console.error('[DAILY-ALERTS] Fatal error:', error);
    return NextResponse.json(
      { error: 'Failed to process daily alerts', details: String(error) },
      { status: 500 }
    );
  }
}

function buildAlertEmail(
  userName: string,
  format: 'summary' | 'detailed',
  briefingData: Record<string, unknown> | null,
  frequency: 'daily' | 'weekly'
): { subject: string; html: string } {
  const dateStr = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  const subjectPrefix = frequency === 'weekly' ? 'Weekly Digest' : 'Daily Briefing';
  const subject = `${subjectPrefix}: Global Intelligence Update`;

  // Extract key metrics from briefing if available
  const briefings = (briefingData as { briefings?: Record<string, string> })?.briefings || {};
  const nsm = briefings['nsm'] || 'Monitor developing situations and maintain standard operational awareness.';
  const summary = briefings['summary'] || 'Multiple regions showing mixed risk indicators.';

  const detailedSection =
    format === 'detailed'
      ? `
      <div style="margin-top: 24px; padding: 20px; background: rgba(59,130,246,0.1); border-radius: 12px; border-left: 4px solid #3b82f6;">
        <h3 style="margin: 0 0 12px 0; color: #60a5fa; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Key Sectors</h3>
        ${Object.entries(briefings)
          .filter(([k]) => !['nsm', 'summary'].includes(k))
          .slice(0, 5)
          .map(
            ([category, text]) => `
          <div style="margin-bottom: 12px;">
            <strong style="color: #e2e8f0; text-transform: capitalize;">${category}:</strong>
            <span style="color: #94a3b8;">${String(text).slice(0, 200)}${String(text).length > 200 ? '...' : ''}</span>
          </div>
        `
          )
          .join('')}
      </div>
    `
      : '';

  const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0a0a0f;">
  <table width="100%" cellspacing="0" cellpadding="0" style="background-color: #0a0a0f;">
    <tr>
      <td align="center" style="padding: 40px 20px;">
        <table width="100%" style="max-width: 600px;">
          <!-- Header -->
          <tr>
            <td style="padding-bottom: 24px; border-bottom: 1px solid rgba(255,255,255,0.1);">
              <img src="https://latticeforge.ai/images/brand/logo-full-white.png" alt="LatticeForge" width="160" style="display: block;">
            </td>
          </tr>

          <!-- Greeting -->
          <tr>
            <td style="padding: 32px 0 16px 0;">
              <p style="margin: 0; color: #94a3b8; font-size: 14px;">Good morning, ${userName}</p>
              <h1 style="margin: 8px 0 0 0; color: #ffffff; font-size: 24px; font-weight: 600;">
                ${frequency === 'weekly' ? 'Your Weekly Intelligence Digest' : 'Your Daily Intelligence Briefing'}
              </h1>
              <p style="margin: 8px 0 0 0; color: #64748b; font-size: 13px;">${dateStr}</p>
            </td>
          </tr>

          <!-- Summary Card -->
          <tr>
            <td style="padding: 24px; background: linear-gradient(135deg, rgba(18,18,26,0.95), rgba(10,10,15,0.95)); border-radius: 16px; border: 1px solid rgba(255,255,255,0.08);">
              <h2 style="margin: 0 0 16px 0; color: #f8fafc; font-size: 18px;">Executive Summary</h2>
              <p style="margin: 0; color: #cbd5e1; font-size: 15px; line-height: 1.6;">${summary}</p>

              ${detailedSection}

              <!-- NSM -->
              <div style="margin-top: 24px; padding: 16px; background: rgba(251,146,60,0.1); border-radius: 8px; border-left: 4px solid #f97316;">
                <h3 style="margin: 0 0 8px 0; color: #fb923c; font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">Next Strategic Move</h3>
                <p style="margin: 0; color: #fcd34d; font-size: 14px; line-height: 1.5;">${nsm}</p>
              </div>
            </td>
          </tr>

          <!-- CTA -->
          <tr>
            <td align="center" style="padding: 32px 0;">
              <a href="https://latticeforge.ai/app/briefings" style="display: inline-block; padding: 16px 40px; background: linear-gradient(135deg, #f97316, #fb923c, #fbbf24); color: #1a1a2e; font-size: 16px; font-weight: 700; text-decoration: none; border-radius: 12px; box-shadow: 0 4px 20px rgba(251,146,60,0.4);">
                View Full Briefing →
              </a>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="padding-top: 24px; border-top: 1px solid rgba(255,255,255,0.1);">
              <p style="margin: 0 0 8px 0; color: #64748b; font-size: 12px; text-align: center;">
                You're receiving this because you subscribed to ${frequency} alerts on LatticeForge.
              </p>
              <p style="margin: 0; color: #475569; font-size: 12px; text-align: center;">
                <a href="https://latticeforge.ai/app/settings/notifications" style="color: #60a5fa; text-decoration: none;">Manage preferences</a>
                &nbsp;•&nbsp;
                <a href="https://latticeforge.ai/unsubscribe?type=${frequency}" style="color: #60a5fa; text-decoration: none;">Unsubscribe</a>
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>
`;

  return { subject, html };
}
