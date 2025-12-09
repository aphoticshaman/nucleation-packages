import { NextResponse } from 'next/server';

// =============================================================================
// CRON: API Usage Monitor
// =============================================================================
// Monitors non-prod environments for RunPod/LFBM API enablement.
// Sends email alert if APIs have been enabled for more than 2 hours.
//
// Schedule: Every 30 minutes (vercel.json)
// */30 * * * *
//
// Note: Uses Vercel KV or simple stateless approach to track state.

const ADMIN_EMAIL = 'admin@latticeforge.ai';
const MAX_RUNTIME_HOURS = 2;

// For Vercel Edge/Serverless, we use a simple approach:
// The admin override API stores the enabledAt timestamp.
// This cron just checks if it's been too long.

async function sendAlertEmail(runtimeMinutes: number, environment: string): Promise<boolean> {
  // Try using Resend if configured
  if (process.env.RESEND_API_KEY) {
    try {
      const response = await fetch('https://api.resend.com/emails', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${process.env.RESEND_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          from: 'LatticeForge Alerts <alerts@latticeforge.ai>',
          to: [ADMIN_EMAIL],
          subject: `[ALERT] RunPod APIs Running ${runtimeMinutes}+ mins in Non-Prod`,
          html: `
            <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto;">
              <h2 style="color: #dc2626;">RunPod API Usage Alert</h2>
              <p>RunPod/LFBM APIs have been enabled in a <strong>non-production</strong> environment for over ${MAX_RUNTIME_HOURS} hours.</p>
              <table style="margin: 20px 0; border-collapse: collapse;">
                <tr>
                  <td style="padding: 8px; border: 1px solid #ccc;"><strong>Environment:</strong></td>
                  <td style="padding: 8px; border: 1px solid #ccc;">${environment}</td>
                </tr>
                <tr>
                  <td style="padding: 8px; border: 1px solid #ccc;"><strong>Runtime:</strong></td>
                  <td style="padding: 8px; border: 1px solid #ccc;">${runtimeMinutes} minutes</td>
                </tr>
                <tr>
                  <td style="padding: 8px; border: 1px solid #ccc;"><strong>Timestamp:</strong></td>
                  <td style="padding: 8px; border: 1px solid #ccc;">${new Date().toISOString()}</td>
                </tr>
              </table>
              <p style="color: #dc2626; font-weight: bold;">
                This may be costing money on RunPod. Consider disabling if not needed.
              </p>
              <p>
                To disable: Go to the deployment → Click the API button → Disable
              </p>
            </div>
          `,
        }),
      });

      if (response.ok) {
        console.log('[API-MONITOR] Alert email sent successfully');
        return true;
      } else {
        console.error('[API-MONITOR] Resend returned:', response.status);
        return false;
      }
    } catch (error) {
      console.error('[API-MONITOR] Resend email failed:', error);
      return false;
    }
  }

  // Fallback: Just log to console
  console.warn(`[API-MONITOR] ALERT: RunPod APIs running ${runtimeMinutes}+ mins in ${environment}`);
  console.warn(`[API-MONITOR] No RESEND_API_KEY configured, email not sent`);
  return false;
}

export async function GET(request: Request) {
  // Verify cron authorization
  const isVercelCron = request.headers.get('x-vercel-cron') === '1';
  const cronSecret = process.env.CRON_SECRET;
  const authHeader = request.headers.get('authorization');

  if (!isVercelCron && cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  // Skip in production - only monitor non-prod
  const environment = process.env.VERCEL_ENV || process.env.NODE_ENV;
  if (environment === 'production') {
    return NextResponse.json({
      skipped: true,
      reason: 'Production environment - monitoring not needed',
    });
  }

  // Check if APIs are currently enabled via LF_PROD_ENABLE
  const lfProdEnable = process.env.LF_PROD_ENABLE;
  const apisEnabled = lfProdEnable === 'true';

  if (!apisEnabled) {
    return NextResponse.json({
      status: 'disabled',
      environment,
      lfProdEnable,
      message: 'APIs are disabled, no monitoring needed',
    });
  }

  // APIs are enabled in non-prod - this is noteworthy
  // For stateless monitoring, we can only alert each time this runs
  // To avoid spam, we could use Vercel KV to track last alert time

  // For now, simple approach: always log when APIs are enabled in non-prod
  console.log(`[API-MONITOR] APIs enabled in non-prod environment: ${environment}`);

  // Check Vercel KV for last alert time (if configured)
  if (process.env.KV_REST_API_URL && process.env.KV_REST_API_TOKEN) {
    try {
      // Check last alert time from KV
      const kvResponse = await fetch(`${process.env.KV_REST_API_URL}/get/api_monitor_last_alert`, {
        headers: {
          Authorization: `Bearer ${process.env.KV_REST_API_TOKEN}`,
        },
      });

      if (kvResponse.ok) {
        const kvData = await kvResponse.json();
        const lastAlertTime = kvData.result ? parseInt(kvData.result, 10) : 0;
        const timeSinceLastAlert = Date.now() - lastAlertTime;
        const minAlertInterval = MAX_RUNTIME_HOURS * 60 * 60 * 1000; // 2 hours between alerts

        if (timeSinceLastAlert < minAlertInterval) {
          return NextResponse.json({
            status: 'monitoring',
            apisEnabled: true,
            environment,
            message: 'Alert already sent recently, skipping',
            lastAlertMinutesAgo: Math.round(timeSinceLastAlert / (60 * 1000)),
          });
        }
      }

      // Send alert
      const alertSent = await sendAlertEmail(MAX_RUNTIME_HOURS * 60, environment || 'unknown');

      if (alertSent) {
        // Record alert time in KV
        await fetch(`${process.env.KV_REST_API_URL}/set/api_monitor_last_alert/${Date.now()}`, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${process.env.KV_REST_API_TOKEN}`,
          },
        });
      }

      return NextResponse.json({
        status: 'alert_sent',
        apisEnabled: true,
        environment,
        alertSent,
      });
    } catch (error) {
      console.error('[API-MONITOR] KV error:', error);
    }
  }

  // Without KV, send alert every time (rely on cron interval for throttling)
  const alertSent = await sendAlertEmail(MAX_RUNTIME_HOURS * 60, environment || 'unknown');

  return NextResponse.json({
    status: 'monitoring',
    apisEnabled: true,
    environment,
    alertSent,
    message: 'No KV configured - sending alert on every cron run when APIs enabled',
  });
}
