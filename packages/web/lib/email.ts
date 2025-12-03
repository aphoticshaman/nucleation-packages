/**
 * EMAIL NOTIFICATION SERVICE
 *
 * Handles transactional emails for billing and account notifications.
 * Uses Resend for delivery with fallback logging.
 */

export interface EmailOptions {
  to: string;
  subject: string;
  html: string;
  text?: string;
}

export interface EmailResult {
  success: boolean;
  messageId?: string;
  error?: string;
}

/**
 * Send an email using Resend API
 * Falls back to logging if API key not configured
 */
export async function sendEmail(options: EmailOptions): Promise<EmailResult> {
  const apiKey = process.env.RESEND_API_KEY;
  const fromAddress = process.env.EMAIL_FROM || 'LatticeForge <notifications@latticeforge.ai>';

  // Log-only mode if no API key
  if (!apiKey) {
    console.log('[Email] Would send:', {
      to: options.to,
      subject: options.subject,
      preview: options.text?.slice(0, 100) || options.html.slice(0, 100),
    });
    return { success: true, messageId: 'log-only' };
  }

  try {
    const response = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: fromAddress,
        to: options.to,
        subject: options.subject,
        html: options.html,
        text: options.text,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('[Email] Failed to send:', error);
      return { success: false, error };
    }

    const data = await response.json();
    return { success: true, messageId: data.id };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    console.error('[Email] Send error:', message);
    return { success: false, error: message };
  }
}

/**
 * Email templates for billing events
 */
export const emailTemplates = {
  paymentFailed: (orgName: string, amount?: number) => ({
    subject: `Payment Failed - ${orgName}`,
    html: `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8">
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }
            .container { max-width: 600px; margin: 0 auto; padding: 20px; }
            .header { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); color: white; padding: 30px; border-radius: 8px 8px 0 0; }
            .content { background: #f8fafc; padding: 30px; border-radius: 0 0 8px 8px; }
            .alert { background: #fef2f2; border-left: 4px solid #ef4444; padding: 15px; margin: 20px 0; }
            .button { display: inline-block; background: #3b82f6; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; margin-top: 20px; }
            .footer { text-align: center; color: #64748b; font-size: 12px; margin-top: 30px; }
          </style>
        </head>
        <body>
          <div class="container">
            <div class="header">
              <h1 style="margin: 0;">LatticeForge</h1>
              <p style="margin: 10px 0 0; opacity: 0.9;">Payment Notification</p>
            </div>
            <div class="content">
              <div class="alert">
                <strong>Payment Failed</strong>
                <p>We were unable to process the payment for ${orgName}${amount ? ` ($${(amount / 100).toFixed(2)})` : ''}.</p>
              </div>
              <p>Please update your payment method to continue your subscription without interruption.</p>
              <p>If you believe this is an error or need assistance, please contact our support team.</p>
              <a href="https://latticeforge.ai/settings/billing" class="button">Update Payment Method</a>
            </div>
            <div class="footer">
              <p>LatticeForge - Geopolitical Intelligence Platform</p>
              <p>This is an automated message. Please do not reply directly.</p>
            </div>
          </div>
        </body>
      </html>
    `,
    text: `Payment Failed - ${orgName}

We were unable to process your payment${amount ? ` ($${(amount / 100).toFixed(2)})` : ''}.

Please update your payment method at: https://latticeforge.ai/settings/billing

If you need assistance, please contact our support team.

- LatticeForge Team`,
  }),

  subscriptionCanceled: (orgName: string) => ({
    subject: `Subscription Canceled - ${orgName}`,
    html: `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8">
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }
            .container { max-width: 600px; margin: 0 auto; padding: 20px; }
            .header { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); color: white; padding: 30px; border-radius: 8px 8px 0 0; }
            .content { background: #f8fafc; padding: 30px; border-radius: 0 0 8px 8px; }
            .info { background: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin: 20px 0; }
            .button { display: inline-block; background: #3b82f6; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; margin-top: 20px; }
            .footer { text-align: center; color: #64748b; font-size: 12px; margin-top: 30px; }
          </style>
        </head>
        <body>
          <div class="container">
            <div class="header">
              <h1 style="margin: 0;">LatticeForge</h1>
              <p style="margin: 10px 0 0; opacity: 0.9;">Subscription Update</p>
            </div>
            <div class="content">
              <div class="info">
                <strong>Subscription Canceled</strong>
                <p>Your subscription for ${orgName} has been canceled.</p>
              </div>
              <p>Your account has been downgraded to the free tier. You can still access basic features.</p>
              <p>We'd love to have you back! If you'd like to resubscribe, you can do so at any time.</p>
              <a href="https://latticeforge.ai/pricing" class="button">View Plans</a>
            </div>
            <div class="footer">
              <p>LatticeForge - Geopolitical Intelligence Platform</p>
              <p>This is an automated message. Please do not reply directly.</p>
            </div>
          </div>
        </body>
      </html>
    `,
    text: `Subscription Canceled - ${orgName}

Your subscription has been canceled and your account has been downgraded to the free tier.

You can still access basic features, and we'd love to have you back!

Resubscribe at: https://latticeforge.ai/pricing

- LatticeForge Team`,
  }),

  paymentSucceeded: (orgName: string, amount: number, planName: string) => ({
    subject: `Payment Received - ${orgName}`,
    html: `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8">
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }
            .container { max-width: 600px; margin: 0 auto; padding: 20px; }
            .header { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); color: white; padding: 30px; border-radius: 8px 8px 0 0; }
            .content { background: #f8fafc; padding: 30px; border-radius: 0 0 8px 8px; }
            .success { background: #f0fdf4; border-left: 4px solid #22c55e; padding: 15px; margin: 20px 0; }
            .details { background: white; padding: 15px; border-radius: 6px; margin: 20px 0; }
            .footer { text-align: center; color: #64748b; font-size: 12px; margin-top: 30px; }
          </style>
        </head>
        <body>
          <div class="container">
            <div class="header">
              <h1 style="margin: 0;">LatticeForge</h1>
              <p style="margin: 10px 0 0; opacity: 0.9;">Payment Receipt</p>
            </div>
            <div class="content">
              <div class="success">
                <strong>Payment Successful</strong>
                <p>Thank you! Your payment has been processed.</p>
              </div>
              <div class="details">
                <p><strong>Organization:</strong> ${orgName}</p>
                <p><strong>Plan:</strong> ${planName}</p>
                <p><strong>Amount:</strong> $${(amount / 100).toFixed(2)}</p>
              </div>
              <p>Your API usage limits have been reset for the new billing period.</p>
            </div>
            <div class="footer">
              <p>LatticeForge - Geopolitical Intelligence Platform</p>
              <p>This is an automated message. Please do not reply directly.</p>
            </div>
          </div>
        </body>
      </html>
    `,
    text: `Payment Received - ${orgName}

Thank you! Your payment has been processed.

Organization: ${orgName}
Plan: ${planName}
Amount: $${(amount / 100).toFixed(2)}

Your API usage limits have been reset for the new billing period.

- LatticeForge Team`,
  }),
};

/**
 * Get admin email for an organization
 */
export async function getOrgAdminEmail(
  supabase: { from: (table: string) => { select: (cols: string) => { eq: (col: string, val: string) => { single: () => Promise<{ data: { email?: string } | null }> } } } },
  orgId: string
): Promise<string | null> {
  const { data } = await supabase
    .from('profiles')
    .select('email')
    .eq('organization_id', orgId)
    .single();

  return data?.email || null;
}
