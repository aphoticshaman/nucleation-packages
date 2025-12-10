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

  welcomeSubscription: (userName: string, planName: string, trialDays?: number) => ({
    subject: trialDays
      ? `Welcome to LatticeForge - Your ${trialDays}-Day Trial Starts Now!`
      : `Welcome to LatticeForge ${planName}!`,
    html: `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8">
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }
            .container { max-width: 600px; margin: 0 auto; padding: 20px; }
            .header { background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); color: white; padding: 30px; border-radius: 8px 8px 0 0; text-align: center; }
            .content { background: #f8fafc; padding: 30px; border-radius: 0 0 8px 8px; }
            .success { background: #f0fdf4; border-left: 4px solid #22c55e; padding: 15px; margin: 20px 0; }
            .feature { display: flex; align-items: center; margin: 10px 0; }
            .feature-icon { background: #dbeafe; padding: 8px; border-radius: 6px; margin-right: 12px; }
            .button { display: inline-block; background: linear-gradient(135deg, #f97316, #fb923c); color: white; padding: 14px 28px; text-decoration: none; border-radius: 8px; margin-top: 20px; font-weight: 600; }
            .trial-note { background: #fef3c7; border: 1px solid #fcd34d; padding: 15px; border-radius: 8px; margin: 20px 0; }
            .footer { text-align: center; color: #64748b; font-size: 12px; margin-top: 30px; }
          </style>
        </head>
        <body>
          <div class="container">
            <div class="header">
              <h1 style="margin: 0; font-size: 28px;">Welcome to LatticeForge!</h1>
              <p style="margin: 10px 0 0; opacity: 0.9;">${trialDays ? `Your ${trialDays}-day Pro trial is active` : `${planName} subscription activated`}</p>
            </div>
            <div class="content">
              <p>Hi ${userName || 'there'},</p>
              <div class="success">
                <strong>You're all set!</strong>
                <p>Your ${trialDays ? 'trial' : planName + ' subscription'} is now active. Start exploring:</p>
              </div>
              ${trialDays ? `
              <div class="trial-note">
                <strong>Trial Details</strong>
                <p style="margin: 5px 0 0;">Your card won't be charged for ${trialDays} days. We'll send you a reminder before your trial ends.</p>
              </div>
              ` : ''}
              <p><strong>What you can do now:</strong></p>
              <div class="feature"><span class="feature-icon">🌍</span> Monitor 195 countries in real-time</div>
              <div class="feature"><span class="feature-icon">⚡</span> Get instant alerts on breaking events</div>
              <div class="feature"><span class="feature-icon">📊</span> Access predictive risk analytics</div>
              <div class="feature"><span class="feature-icon">📧</span> Receive daily briefings in your inbox</div>
              <center>
                <a href="https://latticeforge.ai/app" class="button">Open Dashboard</a>
              </center>
            </div>
            <div class="footer">
              <p>LatticeForge - Know What Happens Next</p>
              <p>Questions? Reply to this email or reach us at support@latticeforge.ai</p>
            </div>
          </div>
        </body>
      </html>
    `,
    text: `Welcome to LatticeForge!

Hi ${userName || 'there'},

Your ${trialDays ? trialDays + '-day Pro trial' : planName + ' subscription'} is now active.

${trialDays ? `Your card won't be charged for ${trialDays} days. We'll send you a reminder before your trial ends.` : ''}

What you can do now:
- Monitor 195 countries in real-time
- Get instant alerts on breaking events
- Access predictive risk analytics
- Receive daily briefings in your inbox

Open your dashboard: https://latticeforge.ai/app

Questions? Reply to this email or reach us at support@latticeforge.ai

- LatticeForge Team`,
  }),

  trialEndingReminder: (userName: string, daysLeft: number, planName: string, monthlyPrice: number) => ({
    subject: daysLeft === 0
      ? `Your LatticeForge trial ends today`
      : `Your LatticeForge trial ends in ${daysLeft} day${daysLeft > 1 ? 's' : ''}`,
    html: `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8">
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }
            .container { max-width: 600px; margin: 0 auto; padding: 20px; }
            .header { background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); color: white; padding: 30px; border-radius: 8px 8px 0 0; }
            .content { background: #f8fafc; padding: 30px; border-radius: 0 0 8px 8px; }
            .alert { background: ${daysLeft === 0 ? '#fef2f2' : '#fef3c7'}; border-left: 4px solid ${daysLeft === 0 ? '#ef4444' : '#f59e0b'}; padding: 15px; margin: 20px 0; }
            .details { background: white; padding: 15px; border-radius: 6px; margin: 20px 0; border: 1px solid #e2e8f0; }
            .button { display: inline-block; background: linear-gradient(135deg, #f97316, #fb923c); color: white; padding: 14px 28px; text-decoration: none; border-radius: 8px; margin-top: 10px; font-weight: 600; }
            .button-secondary { display: inline-block; background: transparent; color: #64748b; padding: 14px 28px; text-decoration: none; border-radius: 8px; margin-top: 10px; border: 1px solid #e2e8f0; }
            .footer { text-align: center; color: #64748b; font-size: 12px; margin-top: 30px; }
          </style>
        </head>
        <body>
          <div class="container">
            <div class="header">
              <h1 style="margin: 0;">LatticeForge</h1>
              <p style="margin: 10px 0 0; opacity: 0.9;">Trial Reminder</p>
            </div>
            <div class="content">
              <p>Hi ${userName || 'there'},</p>
              <div class="alert">
                <strong>${daysLeft === 0 ? 'Your trial ends today!' : `${daysLeft} day${daysLeft > 1 ? 's' : ''} left in your trial`}</strong>
                <p>${daysLeft === 0
                  ? 'Your card will be charged today to continue your Pro subscription.'
                  : 'Your Pro trial is coming to an end soon.'}</p>
              </div>
              <div class="details">
                <p><strong>What happens next:</strong></p>
                <p>Your card will be charged <strong>$${monthlyPrice}/month</strong> for ${planName} when the trial ends.</p>
                <p>Don't want to continue? You can cancel anytime from your billing settings.</p>
              </div>
              <center>
                <a href="https://latticeforge.ai/app" class="button">Continue with Pro</a>
                <br/>
                <a href="https://latticeforge.ai/settings/billing" class="button-secondary">Manage Subscription</a>
              </center>
            </div>
            <div class="footer">
              <p>LatticeForge - Know What Happens Next</p>
              <p>Questions? Reply to this email or reach us at support@latticeforge.ai</p>
            </div>
          </div>
        </body>
      </html>
    `,
    text: `Your LatticeForge trial ${daysLeft === 0 ? 'ends today' : `ends in ${daysLeft} day${daysLeft > 1 ? 's' : ''}`}

Hi ${userName || 'there'},

${daysLeft === 0
  ? 'Your trial ends today! Your card will be charged to continue your Pro subscription.'
  : `You have ${daysLeft} day${daysLeft > 1 ? 's' : ''} left in your Pro trial.`}

What happens next:
Your card will be charged $${monthlyPrice}/month for ${planName} when the trial ends.

Don't want to continue? Cancel anytime: https://latticeforge.ai/settings/billing

Continue with Pro: https://latticeforge.ai/app

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

  dailyBriefing: (userName: string, briefingData: {
    date: string;
    globalRiskLevel: 'low' | 'moderate' | 'elevated' | 'high' | 'critical';
    topAlerts: Array<{ country: string; summary: string; severity: 'low' | 'moderate' | 'high' }>;
    watchlistUpdates: Array<{ country: string; change: string }>;
    keyInsights: string[];
  }) => ({
    subject: `Daily Intel Briefing - ${briefingData.date}`,
    html: `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8">
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #e2e8f0; margin: 0; padding: 0; background: #0a0a0f; }
            .container { max-width: 600px; margin: 0 auto; padding: 20px; }
            .header { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); color: white; padding: 30px; border-radius: 12px 12px 0 0; text-align: center; }
            .logo { width: 48px; height: 48px; border-radius: 12px; margin-bottom: 16px; }
            .content { background: #12121a; padding: 30px; border-radius: 0 0 12px 12px; border: 1px solid rgba(255,255,255,0.08); border-top: none; }
            .risk-badge { display: inline-block; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
            .risk-low { background: rgba(34,197,94,0.2); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
            .risk-moderate { background: rgba(59,130,246,0.2); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
            .risk-elevated { background: rgba(251,191,36,0.2); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
            .risk-high { background: rgba(249,115,22,0.2); color: #fb923c; border: 1px solid rgba(249,115,22,0.3); }
            .risk-critical { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
            .section { margin: 24px 0; padding: 20px; background: rgba(255,255,255,0.03); border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); }
            .section-title { margin: 0 0 16px 0; color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
            .alert-item { padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.06); }
            .alert-item:last-child { border-bottom: none; padding-bottom: 0; }
            .alert-country { font-weight: 600; color: #ffffff; }
            .alert-summary { color: #94a3b8; font-size: 14px; margin-top: 4px; }
            .severity-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }
            .severity-low { background: #4ade80; }
            .severity-moderate { background: #fbbf24; }
            .severity-high { background: #f87171; }
            .insight-item { padding: 8px 0; color: #cbd5e1; font-size: 14px; }
            .button { display: inline-block; background: linear-gradient(135deg, #f97316, #fb923c); color: white; padding: 14px 28px; text-decoration: none; border-radius: 8px; margin-top: 20px; font-weight: 600; }
            .footer { text-align: center; color: #64748b; font-size: 12px; margin-top: 30px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.06); }
          </style>
        </head>
        <body>
          <div class="container">
            <div class="header">
              <img src="https://latticeforge.ai/images/brand/app-icon.png" alt="LatticeForge" class="logo">
              <h1 style="margin: 0; font-size: 20px; font-weight: 600;">Daily Intelligence Briefing</h1>
              <p style="margin: 8px 0 0; opacity: 0.7; font-size: 14px;">${briefingData.date}</p>
            </div>
            <div class="content">
              <p style="margin: 0 0 20px; color: #94a3b8;">Good morning${userName ? `, ${userName}` : ''}. Here's your intelligence summary:</p>

              <div style="text-align: center; margin-bottom: 24px;">
                <p style="margin: 0 0 8px; color: #64748b; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Global Risk Level</p>
                <span class="risk-badge risk-${briefingData.globalRiskLevel}">${briefingData.globalRiskLevel}</span>
              </div>

              ${briefingData.topAlerts.length > 0 ? `
              <div class="section">
                <p class="section-title">Top Alerts</p>
                ${briefingData.topAlerts.map(alert => `
                  <div class="alert-item">
                    <span class="severity-dot severity-${alert.severity}"></span>
                    <span class="alert-country">${alert.country}</span>
                    <p class="alert-summary">${alert.summary}</p>
                  </div>
                `).join('')}
              </div>
              ` : ''}

              ${briefingData.watchlistUpdates.length > 0 ? `
              <div class="section">
                <p class="section-title">Watchlist Updates</p>
                ${briefingData.watchlistUpdates.map(update => `
                  <div class="alert-item">
                    <span class="alert-country">${update.country}</span>
                    <p class="alert-summary">${update.change}</p>
                  </div>
                `).join('')}
              </div>
              ` : ''}

              ${briefingData.keyInsights.length > 0 ? `
              <div class="section">
                <p class="section-title">Key Insights</p>
                ${briefingData.keyInsights.map(insight => `
                  <div class="insight-item">• ${insight}</div>
                `).join('')}
              </div>
              ` : ''}

              <center>
                <a href="https://latticeforge.ai/app" class="button">Open Dashboard</a>
              </center>
            </div>
            <div class="footer">
              <p style="margin: 0;">LatticeForge - Know What Happens Next</p>
              <p style="margin: 8px 0 0;">
                <a href="https://latticeforge.ai/settings/notifications" style="color: #60a5fa; text-decoration: none;">Manage preferences</a>
                &nbsp;•&nbsp;
                <a href="https://latticeforge.ai/unsubscribe" style="color: #60a5fa; text-decoration: none;">Unsubscribe</a>
              </p>
            </div>
          </div>
        </body>
      </html>
    `,
    text: `Daily Intelligence Briefing - ${briefingData.date}

Good morning${userName ? `, ${userName}` : ''}. Here's your intelligence summary:

Global Risk Level: ${briefingData.globalRiskLevel.toUpperCase()}

${briefingData.topAlerts.length > 0 ? `TOP ALERTS:
${briefingData.topAlerts.map(a => `[${a.severity.toUpperCase()}] ${a.country}: ${a.summary}`).join('\n')}
` : ''}
${briefingData.watchlistUpdates.length > 0 ? `WATCHLIST UPDATES:
${briefingData.watchlistUpdates.map(u => `${u.country}: ${u.change}`).join('\n')}
` : ''}
${briefingData.keyInsights.length > 0 ? `KEY INSIGHTS:
${briefingData.keyInsights.map(i => `• ${i}`).join('\n')}
` : ''}
View full dashboard: https://latticeforge.ai/app

Manage preferences: https://latticeforge.ai/settings/notifications

- LatticeForge Team`,
  }),

  alertNotification: (alertData: {
    country: string;
    countryCode: string;
    alertType: 'risk_spike' | 'cascade_warning' | 'breaking_event' | 'threshold_breach';
    severity: 'low' | 'moderate' | 'high' | 'critical';
    headline: string;
    details: string;
    riskScore?: number;
    previousScore?: number;
    timestamp: string;
    sources?: string[];
  }) => ({
    subject: `[${alertData.severity.toUpperCase()}] ${alertData.country} Alert: ${alertData.headline}`,
    html: `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8">
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #e2e8f0; margin: 0; padding: 0; background: #0a0a0f; }
            .container { max-width: 600px; margin: 0 auto; padding: 20px; }
            .header { padding: 24px; border-radius: 12px 12px 0 0; text-align: center; }
            .header-critical { background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%); }
            .header-high { background: linear-gradient(135deg, #7c2d12 0%, #9a3412 100%); }
            .header-moderate { background: linear-gradient(135deg, #78350f 0%, #92400e 100%); }
            .header-low { background: linear-gradient(135deg, #1e3a5f 0%, #1e40af 100%); }
            .logo { width: 40px; height: 40px; border-radius: 10px; margin-bottom: 12px; }
            .severity-badge { display: inline-block; padding: 4px 12px; border-radius: 4px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
            .severity-critical { background: rgba(239,68,68,0.3); color: #fca5a5; }
            .severity-high { background: rgba(249,115,22,0.3); color: #fdba74; }
            .severity-moderate { background: rgba(251,191,36,0.3); color: #fde047; }
            .severity-low { background: rgba(59,130,246,0.3); color: #93c5fd; }
            .content { background: #12121a; padding: 30px; border-radius: 0 0 12px 12px; border: 1px solid rgba(255,255,255,0.08); border-top: none; }
            .country-flag { font-size: 32px; margin-right: 8px; }
            .headline { font-size: 20px; font-weight: 600; color: #ffffff; margin: 0; }
            .details { color: #94a3b8; font-size: 15px; margin: 16px 0 0; line-height: 1.7; }
            .metrics { display: flex; gap: 16px; margin: 24px 0; }
            .metric { flex: 1; padding: 16px; background: rgba(255,255,255,0.03); border-radius: 8px; text-align: center; }
            .metric-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
            .metric-value { font-size: 24px; font-weight: 700; margin-top: 4px; }
            .metric-change { font-size: 12px; margin-top: 4px; }
            .change-up { color: #f87171; }
            .change-down { color: #4ade80; }
            .sources { margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.06); }
            .sources-title { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
            .source-item { font-size: 13px; color: #94a3b8; }
            .button { display: inline-block; background: linear-gradient(135deg, #f97316, #fb923c); color: white; padding: 14px 28px; text-decoration: none; border-radius: 8px; margin-top: 24px; font-weight: 600; }
            .footer { text-align: center; color: #64748b; font-size: 12px; margin-top: 24px; }
          </style>
        </head>
        <body>
          <div class="container">
            <div class="header header-${alertData.severity}">
              <img src="https://latticeforge.ai/images/brand/app-icon.png" alt="LatticeForge" class="logo">
              <span class="severity-badge severity-${alertData.severity}">${alertData.severity} priority</span>
              <p style="margin: 0; color: rgba(255,255,255,0.7); font-size: 13px;">${alertData.timestamp}</p>
            </div>
            <div class="content">
              <h1 class="headline">${alertData.country} - ${alertData.headline}</h1>
              <p class="details">${alertData.details}</p>

              ${alertData.riskScore !== undefined ? `
              <table width="100%" cellpadding="0" cellspacing="0" style="margin: 24px 0;">
                <tr>
                  <td style="padding: 16px; background: rgba(255,255,255,0.03); border-radius: 8px; text-align: center; width: 50%;">
                    <p style="margin: 0; font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">Current Risk</p>
                    <p style="margin: 4px 0 0; font-size: 28px; font-weight: 700; color: ${alertData.riskScore > 0.7 ? '#f87171' : alertData.riskScore > 0.4 ? '#fbbf24' : '#4ade80'};">${(alertData.riskScore * 100).toFixed(0)}%</p>
                  </td>
                  ${alertData.previousScore !== undefined ? `
                  <td style="width: 16px;"></td>
                  <td style="padding: 16px; background: rgba(255,255,255,0.03); border-radius: 8px; text-align: center; width: 50%;">
                    <p style="margin: 0; font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">Change</p>
                    <p style="margin: 4px 0 0; font-size: 28px; font-weight: 700; color: ${alertData.riskScore > alertData.previousScore ? '#f87171' : '#4ade80'};">
                      ${alertData.riskScore > alertData.previousScore ? '+' : ''}${((alertData.riskScore - alertData.previousScore) * 100).toFixed(0)}%
                    </p>
                  </td>
                  ` : ''}
                </tr>
              </table>
              ` : ''}

              ${alertData.sources && alertData.sources.length > 0 ? `
              <div class="sources">
                <p class="sources-title">Intelligence Sources</p>
                ${alertData.sources.map(s => `<p class="source-item">• ${s}</p>`).join('')}
              </div>
              ` : ''}

              <center>
                <a href="https://latticeforge.ai/app/country/${alertData.countryCode}" class="button">View ${alertData.country} Details</a>
              </center>
            </div>
            <div class="footer">
              <p style="margin: 0;">This alert was triggered by your LatticeForge monitoring settings.</p>
              <p style="margin: 8px 0 0;">
                <a href="https://latticeforge.ai/settings/alerts" style="color: #60a5fa; text-decoration: none;">Manage alerts</a>
              </p>
            </div>
          </div>
        </body>
      </html>
    `,
    text: `[${alertData.severity.toUpperCase()}] ${alertData.country} Alert

${alertData.headline}

${alertData.details}

${alertData.riskScore !== undefined ? `Risk Score: ${(alertData.riskScore * 100).toFixed(0)}%${alertData.previousScore !== undefined ? ` (${alertData.riskScore > alertData.previousScore ? '+' : ''}${((alertData.riskScore - alertData.previousScore) * 100).toFixed(0)}% change)` : ''}` : ''}

${alertData.sources && alertData.sources.length > 0 ? `Sources:
${alertData.sources.map(s => `• ${s}`).join('\n')}
` : ''}
View details: https://latticeforge.ai/app/country/${alertData.countryCode}

Manage alerts: https://latticeforge.ai/settings/alerts

- LatticeForge Team`,
  }),

  weeklyDigest: (userName: string, digestData: {
    weekRange: string;
    topMovers: Array<{ country: string; change: number; direction: 'up' | 'down' }>;
    significantEvents: Array<{ date: string; country: string; event: string }>;
    regionSummary: Record<string, { avgRisk: number; trend: 'stable' | 'improving' | 'deteriorating' }>;
    aiInsight: string;
  }) => ({
    subject: `Weekly Intelligence Digest - ${digestData.weekRange}`,
    html: `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8">
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #e2e8f0; margin: 0; padding: 0; background: #0a0a0f; }
            .container { max-width: 600px; margin: 0 auto; padding: 20px; }
            .header { background: linear-gradient(135deg, #1e40af 0%, #7c3aed 100%); color: white; padding: 30px; border-radius: 12px 12px 0 0; text-align: center; }
            .logo { width: 48px; height: 48px; border-radius: 12px; margin-bottom: 16px; }
            .content { background: #12121a; padding: 30px; border-radius: 0 0 12px 12px; border: 1px solid rgba(255,255,255,0.08); border-top: none; }
            .section { margin: 24px 0; padding: 20px; background: rgba(255,255,255,0.03); border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); }
            .section-title { margin: 0 0 16px 0; color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
            .mover-item { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.06); }
            .mover-item:last-child { border-bottom: none; }
            .mover-country { font-weight: 500; color: #ffffff; }
            .mover-change { font-weight: 600; }
            .change-up { color: #f87171; }
            .change-down { color: #4ade80; }
            .event-item { padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.06); }
            .event-date { font-size: 12px; color: #64748b; }
            .event-country { font-weight: 500; color: #60a5fa; }
            .event-text { color: #94a3b8; font-size: 14px; margin-top: 4px; }
            .insight-box { background: linear-gradient(135deg, rgba(124,58,237,0.1) 0%, rgba(59,130,246,0.1) 100%); border: 1px solid rgba(124,58,237,0.2); border-radius: 12px; padding: 20px; margin-top: 20px; }
            .insight-label { font-size: 11px; color: #a78bfa; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
            .insight-text { color: #e2e8f0; font-size: 15px; font-style: italic; }
            .button { display: inline-block; background: linear-gradient(135deg, #f97316, #fb923c); color: white; padding: 14px 28px; text-decoration: none; border-radius: 8px; margin-top: 20px; font-weight: 600; }
            .footer { text-align: center; color: #64748b; font-size: 12px; margin-top: 30px; }
          </style>
        </head>
        <body>
          <div class="container">
            <div class="header">
              <img src="https://latticeforge.ai/images/brand/app-icon.png" alt="LatticeForge" class="logo">
              <h1 style="margin: 0; font-size: 20px; font-weight: 600;">Weekly Intelligence Digest</h1>
              <p style="margin: 8px 0 0; opacity: 0.8; font-size: 14px;">${digestData.weekRange}</p>
            </div>
            <div class="content">
              <p style="margin: 0 0 20px; color: #94a3b8;">Hi${userName ? ` ${userName}` : ''}, here's your weekly summary of global developments:</p>

              ${digestData.topMovers.length > 0 ? `
              <div class="section">
                <p class="section-title">Biggest Risk Changes</p>
                ${digestData.topMovers.map(m => `
                  <div class="mover-item">
                    <span class="mover-country">${m.country}</span>
                    <span class="mover-change ${m.direction === 'up' ? 'change-up' : 'change-down'}">
                      ${m.direction === 'up' ? '&#9650;' : '&#9660;'} ${Math.abs(m.change).toFixed(1)}%
                    </span>
                  </div>
                `).join('')}
              </div>
              ` : ''}

              ${digestData.significantEvents.length > 0 ? `
              <div class="section">
                <p class="section-title">Significant Events</p>
                ${digestData.significantEvents.map(e => `
                  <div class="event-item">
                    <span class="event-date">${e.date}</span> • <span class="event-country">${e.country}</span>
                    <p class="event-text">${e.event}</p>
                  </div>
                `).join('')}
              </div>
              ` : ''}

              <div class="insight-box">
                <p class="insight-label">AI Analysis</p>
                <p class="insight-text">"${digestData.aiInsight}"</p>
              </div>

              <center>
                <a href="https://latticeforge.ai/app" class="button">View Full Analysis</a>
              </center>
            </div>
            <div class="footer">
              <p style="margin: 0;">LatticeForge - Know What Happens Next</p>
              <p style="margin: 8px 0 0;">
                <a href="https://latticeforge.ai/settings/notifications" style="color: #60a5fa; text-decoration: none;">Manage preferences</a>
                &nbsp;•&nbsp;
                <a href="https://latticeforge.ai/unsubscribe" style="color: #60a5fa; text-decoration: none;">Unsubscribe</a>
              </p>
            </div>
          </div>
        </body>
      </html>
    `,
    text: `Weekly Intelligence Digest - ${digestData.weekRange}

Hi${userName ? ` ${userName}` : ''}, here's your weekly summary:

${digestData.topMovers.length > 0 ? `BIGGEST RISK CHANGES:
${digestData.topMovers.map(m => `${m.country}: ${m.direction === 'up' ? '+' : '-'}${Math.abs(m.change).toFixed(1)}%`).join('\n')}
` : ''}
${digestData.significantEvents.length > 0 ? `SIGNIFICANT EVENTS:
${digestData.significantEvents.map(e => `${e.date} - ${e.country}: ${e.event}`).join('\n')}
` : ''}
AI ANALYSIS:
"${digestData.aiInsight}"

View full analysis: https://latticeforge.ai/app

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
