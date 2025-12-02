// LatticeForge Email - Resend Integration
// Transactional emails: alerts, briefs, invoices, welcome
// Deploy: supabase functions deploy email

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts';
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

const RESEND_API = 'https://api.resend.com/emails';
const FROM_EMAIL = 'LatticeForge <alerts@latticeforge.io>';

interface EmailRequest {
  to: string;
  template: 'welcome' | 'alert' | 'brief' | 'invoice' | 'password_reset';
  data: Record<string, unknown>;
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  const resendKey = Deno.env.get('RESEND_API_KEY');
  if (!resendKey) {
    return jsonResponse({ error: 'Resend not configured' }, 500);
  }

  const supabase = createClient(
    Deno.env.get('SUPABASE_URL') ?? '',
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
  );

  try {
    const body: EmailRequest = await req.json();
    const { to, template, data } = body;

    if (!to || !template) {
      return jsonResponse({ error: 'Missing to or template' }, 400);
    }

    const email = buildEmail(template, data);

    const response = await fetch(RESEND_API, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${resendKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: FROM_EMAIL,
        to: [to],
        subject: email.subject,
        html: email.html,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('Resend error:', error);
      return jsonResponse({ error: 'Failed to send email' }, 500);
    }

    const result = await response.json();

    return jsonResponse({
      success: true,
      id: result.id,
    });
  } catch (error) {
    console.error('Email error:', error);
    return jsonResponse({ error: error.message }, 500);
  }
});

function buildEmail(
  template: string,
  data: Record<string, unknown>
): { subject: string; html: string } {
  switch (template) {
    case 'welcome':
      return {
        subject: 'Welcome to LatticeForge',
        html: `
          <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 600px; margin: 0 auto;">
            <h1 style="color: #1a1a1a;">Welcome to LatticeForge</h1>
            <p>Hi ${data.name || 'there'},</p>
            <p>Your account is ready. Here's what you can do:</p>
            <ul>
              <li><strong>Signal Fusion</strong> - Combine data from 193 countries</li>
              <li><strong>AI Briefs</strong> - Get executive summaries powered by Claude</li>
              <li><strong>Anomaly Detection</strong> - Spot market regime changes early</li>
            </ul>
            <p>
              <a href="https://latticeforge.vercel.app/dashboard"
                 style="background: #0066ff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;">
                Go to Dashboard
              </a>
            </p>
            <p style="color: #666; font-size: 14px; margin-top: 32px;">
              Questions? Reply to this email or check our docs.
            </p>
          </div>
        `,
      };

    case 'alert':
      return {
        subject: `🚨 ${data.severity === 'critical' ? 'CRITICAL' : 'Alert'}: ${data.title}`,
        html: `
          <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: ${data.severity === 'critical' ? '#dc2626' : '#f59e0b'}; color: white; padding: 16px; border-radius: 8px 8px 0 0;">
              <h2 style="margin: 0;">${data.severity === 'critical' ? '🚨 Critical Alert' : '⚠️ Alert'}</h2>
            </div>
            <div style="border: 1px solid #e5e5e5; border-top: none; padding: 24px; border-radius: 0 0 8px 8px;">
              <h3 style="margin-top: 0;">${data.title}</h3>
              <p>${data.message}</p>
              ${
                data.indicator
                  ? `
                <table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
                  <tr style="background: #f9f9f9;">
                    <td style="padding: 8px; border: 1px solid #e5e5e5;"><strong>Indicator</strong></td>
                    <td style="padding: 8px; border: 1px solid #e5e5e5;">${data.indicator}</td>
                  </tr>
                  <tr>
                    <td style="padding: 8px; border: 1px solid #e5e5e5;"><strong>Value</strong></td>
                    <td style="padding: 8px; border: 1px solid #e5e5e5;">${data.value}</td>
                  </tr>
                  <tr style="background: #f9f9f9;">
                    <td style="padding: 8px; border: 1px solid #e5e5e5;"><strong>Threshold</strong></td>
                    <td style="padding: 8px; border: 1px solid #e5e5e5;">${data.threshold}</td>
                  </tr>
                </table>
              `
                  : ''
              }
              <p>
                <a href="https://latticeforge.vercel.app/alerts"
                   style="background: #0066ff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;">
                  View Details
                </a>
              </p>
              <p style="color: #666; font-size: 12px; margin-top: 24px;">
                Detected at ${new Date().toISOString()}
              </p>
            </div>
          </div>
        `,
      };

    case 'brief':
      return {
        subject: `📊 Daily Brief: ${data.summary || 'Market Update'}`,
        html: `
          <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 600px; margin: 0 auto;">
            <h1 style="color: #1a1a1a;">📊 Executive Brief</h1>
            <p style="color: #666; font-size: 14px;">${new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</p>

            <div style="background: #f9f9f9; padding: 16px; border-radius: 8px; margin: 16px 0;">
              <h3 style="margin-top: 0;">Summary</h3>
              <p>${data.summary}</p>
            </div>

            ${
              data.content
                ? `
              <div style="white-space: pre-wrap; line-height: 1.6;">
                ${data.content}
              </div>
            `
                : ''
            }

            <p>
              <a href="https://latticeforge.vercel.app/briefs"
                 style="background: #0066ff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;">
                View Full Brief
              </a>
            </p>
          </div>
        `,
      };

    case 'invoice':
      return {
        subject: `Invoice from LatticeForge - ${data.amount}`,
        html: `
          <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 600px; margin: 0 auto;">
            <h1 style="color: #1a1a1a;">Invoice</h1>
            <p>Hi ${data.name},</p>
            <p>Thank you for your payment.</p>

            <table style="width: 100%; border-collapse: collapse; margin: 24px 0;">
              <tr style="background: #f9f9f9;">
                <td style="padding: 12px; border: 1px solid #e5e5e5;"><strong>Plan</strong></td>
                <td style="padding: 12px; border: 1px solid #e5e5e5;">${data.plan}</td>
              </tr>
              <tr>
                <td style="padding: 12px; border: 1px solid #e5e5e5;"><strong>Amount</strong></td>
                <td style="padding: 12px; border: 1px solid #e5e5e5;">${data.amount}</td>
              </tr>
              <tr style="background: #f9f9f9;">
                <td style="padding: 12px; border: 1px solid #e5e5e5;"><strong>Date</strong></td>
                <td style="padding: 12px; border: 1px solid #e5e5e5;">${data.date || new Date().toLocaleDateString()}</td>
              </tr>
            </table>

            <p>
              <a href="${data.invoice_url || 'https://latticeforge.vercel.app/billing'}"
                 style="background: #0066ff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;">
                View Invoice
              </a>
            </p>
          </div>
        `,
      };

    case 'password_reset':
      return {
        subject: 'Reset your LatticeForge password',
        html: `
          <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 600px; margin: 0 auto;">
            <h1 style="color: #1a1a1a;">Reset Password</h1>
            <p>We received a request to reset your password.</p>
            <p>
              <a href="${data.reset_url}"
                 style="background: #0066ff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;">
                Reset Password
              </a>
            </p>
            <p style="color: #666; font-size: 14px; margin-top: 24px;">
              If you didn't request this, ignore this email. The link expires in 1 hour.
            </p>
          </div>
        `,
      };

    default:
      return {
        subject: 'LatticeForge Notification',
        html: `<p>${JSON.stringify(data)}</p>`,
      };
  }
}

function jsonResponse(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  });
}
