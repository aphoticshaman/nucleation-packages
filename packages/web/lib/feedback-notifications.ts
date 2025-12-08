/**
 * Feedback Email Notifications
 *
 * Sends automated emails via Resend for:
 * - New ticket creation (to admin team)
 * - Status changes (to submitter)
 * - Assignment notifications (to assignee)
 */

// Admin emails to notify on new tickets
const ADMIN_EMAILS = [
  'support@latticeforge.ai', // Support team inbox (via ImprovMX)
];

// Status labels for human-readable emails
const STATUS_LABELS: Record<string, string> = {
  unread: 'New',
  acknowledged: 'Acknowledged',
  in_progress: 'In Progress',
  resolved: 'Resolved',
  wont_fix: "Won't Fix",
  duplicate: 'Duplicate',
};

// Priority colors for emails
const PRIORITY_COLORS: Record<string, string> = {
  critical: '#DC2626',
  high: '#F59E0B',
  normal: '#3B82F6',
  low: '#6B7280',
};

// Type labels
const TYPE_LABELS: Record<string, string> = {
  bug: '🐛 Bug Report',
  idea: '💡 Feature Idea',
  question: '❓ Question',
  other: '📝 Feedback',
};

interface FeedbackData {
  id: string;
  type: string;
  title: string;
  description: string;
  priority: string;
  status: string;
  pageUrl?: string;
  userEmail?: string;
  userName?: string;
}

/**
 * Send email via Resend API
 */
async function sendEmail(options: {
  to: string[];
  subject: string;
  html: string;
  replyTo?: string;
}): Promise<boolean> {
  const resendKey = process.env.RESEND_API_KEY;
  if (!resendKey) {
    console.error('[FEEDBACK EMAIL] RESEND_API_KEY not configured');
    return false;
  }

  try {
    const response = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${resendKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: 'LatticeForge Feedback <feedback@latticeforge.ai>',
        to: options.to,
        subject: options.subject,
        html: options.html,
        reply_to: options.replyTo,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('[FEEDBACK EMAIL] Resend error:', error);
      return false;
    }

    console.log(`[FEEDBACK EMAIL] Sent to ${options.to.join(', ')}`);
    return true;
  } catch (error) {
    console.error('[FEEDBACK EMAIL] Failed to send:', error);
    return false;
  }
}

/**
 * Generate HTML for new ticket notification (to admins)
 */
function generateNewTicketHtml(feedback: FeedbackData): string {
  const priorityColor = PRIORITY_COLORS[feedback.priority] || PRIORITY_COLORS.normal;
  const typeLabel = TYPE_LABELS[feedback.type] || feedback.type;

  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0f172a; color: #e2e8f0; margin: 0; padding: 20px;">
  <div style="max-width: 600px; margin: 0 auto; background-color: #1e293b; border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1);">
    <!-- Header -->
    <div style="background: linear-gradient(135deg, ${priorityColor}22, ${priorityColor}11); padding: 24px; border-bottom: 1px solid rgba(255,255,255,0.1);">
      <div style="display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 24px;">${typeLabel.split(' ')[0]}</span>
        <div>
          <h1 style="margin: 0; font-size: 18px; color: #f1f5f9;">New ${typeLabel.split(' ').slice(1).join(' ')}</h1>
          <p style="margin: 4px 0 0; font-size: 12px; color: #94a3b8;">Ticket #${feedback.id.slice(0, 8)}</p>
        </div>
      </div>
    </div>

    <!-- Priority Badge -->
    <div style="padding: 16px 24px 0;">
      <span style="display: inline-block; padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600; text-transform: uppercase; background-color: ${priorityColor}22; color: ${priorityColor}; border: 1px solid ${priorityColor}44;">
        ${feedback.priority} priority
      </span>
    </div>

    <!-- Content -->
    <div style="padding: 24px;">
      <h2 style="margin: 0 0 12px; font-size: 16px; color: #f1f5f9;">${feedback.title}</h2>
      <p style="margin: 0 0 20px; font-size: 14px; color: #cbd5e1; line-height: 1.6; white-space: pre-wrap;">${feedback.description}</p>

      <!-- Metadata -->
      <div style="background-color: #0f172a; border-radius: 8px; padding: 16px; font-size: 13px;">
        <table style="width: 100%; border-collapse: collapse;">
          <tr>
            <td style="color: #64748b; padding: 4px 0;">Submitted by:</td>
            <td style="color: #e2e8f0; text-align: right;">${feedback.userName || feedback.userEmail || 'Anonymous'}</td>
          </tr>
          ${feedback.userEmail ? `
          <tr>
            <td style="color: #64748b; padding: 4px 0;">Email:</td>
            <td style="color: #e2e8f0; text-align: right;"><a href="mailto:${feedback.userEmail}" style="color: #38bdf8;">${feedback.userEmail}</a></td>
          </tr>
          ` : ''}
          ${feedback.pageUrl ? `
          <tr>
            <td style="color: #64748b; padding: 4px 0;">Page:</td>
            <td style="color: #e2e8f0; text-align: right; word-break: break-all;"><a href="${feedback.pageUrl}" style="color: #38bdf8;">${feedback.pageUrl}</a></td>
          </tr>
          ` : ''}
        </table>
      </div>
    </div>

    <!-- Action Button -->
    <div style="padding: 0 24px 24px;">
      <a href="https://latticeforge.ai/admin/feedback/${feedback.id}" style="display: block; text-align: center; padding: 12px 24px; background-color: #0ea5e9; color: white; text-decoration: none; border-radius: 8px; font-weight: 600;">
        View in Dashboard →
      </a>
    </div>

    <!-- Footer -->
    <div style="background-color: #0f172a; padding: 16px 24px; text-align: center; font-size: 12px; color: #64748b;">
      LatticeForge Feedback System
    </div>
  </div>
</body>
</html>
  `;
}

/**
 * Generate HTML for status update notification (to submitter)
 */
function generateStatusUpdateHtml(feedback: FeedbackData, oldStatus: string): string {
  const newStatusLabel = STATUS_LABELS[feedback.status] || feedback.status;
  const oldStatusLabel = STATUS_LABELS[oldStatus] || oldStatus;

  const isResolved = ['resolved', 'wont_fix', 'duplicate'].includes(feedback.status);
  const statusColor = isResolved ? '#10B981' : '#3B82F6';

  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0f172a; color: #e2e8f0; margin: 0; padding: 20px;">
  <div style="max-width: 600px; margin: 0 auto; background-color: #1e293b; border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1);">
    <!-- Header -->
    <div style="background: linear-gradient(135deg, ${statusColor}22, ${statusColor}11); padding: 24px; border-bottom: 1px solid rgba(255,255,255,0.1);">
      <h1 style="margin: 0; font-size: 18px; color: #f1f5f9;">Your Feedback Status Updated</h1>
      <p style="margin: 8px 0 0; font-size: 14px; color: #94a3b8;">Ticket #${feedback.id.slice(0, 8)}</p>
    </div>

    <!-- Status Change -->
    <div style="padding: 24px;">
      <div style="text-align: center; margin-bottom: 24px;">
        <span style="display: inline-block; padding: 6px 16px; border-radius: 9999px; font-size: 13px; background-color: #374151; color: #9ca3af; text-decoration: line-through;">
          ${oldStatusLabel}
        </span>
        <span style="display: inline-block; padding: 0 12px; color: #64748b;">→</span>
        <span style="display: inline-block; padding: 6px 16px; border-radius: 9999px; font-size: 13px; font-weight: 600; background-color: ${statusColor}22; color: ${statusColor}; border: 1px solid ${statusColor}44;">
          ${newStatusLabel}
        </span>
      </div>

      <!-- Original Feedback -->
      <div style="background-color: #0f172a; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
        <p style="margin: 0 0 8px; font-size: 12px; color: #64748b; text-transform: uppercase;">Your Feedback:</p>
        <h3 style="margin: 0 0 8px; font-size: 15px; color: #f1f5f9;">${feedback.title}</h3>
        <p style="margin: 0; font-size: 13px; color: #94a3b8; line-height: 1.5;">${feedback.description.slice(0, 200)}${feedback.description.length > 200 ? '...' : ''}</p>
      </div>

      ${isResolved ? `
      <div style="background-color: ${statusColor}11; border: 1px solid ${statusColor}33; border-radius: 8px; padding: 16px; text-align: center;">
        <p style="margin: 0; color: ${statusColor}; font-size: 14px;">
          ${feedback.status === 'resolved' ? '✓ Your feedback has been addressed!' : feedback.status === 'duplicate' ? 'This was marked as a duplicate of an existing ticket.' : 'This will not be implemented at this time.'}
        </p>
      </div>
      ` : `
      <p style="margin: 0; text-align: center; color: #94a3b8; font-size: 13px;">
        Our team is ${feedback.status === 'acknowledged' ? 'reviewing' : 'actively working on'} your feedback.
      </p>
      `}
    </div>

    <!-- Footer -->
    <div style="background-color: #0f172a; padding: 16px 24px; text-align: center; font-size: 12px; color: #64748b;">
      Thank you for helping us improve LatticeForge!
    </div>
  </div>
</body>
</html>
  `;
}

/**
 * Notify admins of new feedback ticket
 */
export async function notifyNewTicket(feedback: FeedbackData): Promise<void> {
  const typeLabel = TYPE_LABELS[feedback.type] || feedback.type;
  const priorityTag = feedback.priority === 'critical' ? '🚨 CRITICAL: ' :
                      feedback.priority === 'high' ? '⚠️ HIGH: ' : '';

  await sendEmail({
    to: ADMIN_EMAILS,
    subject: `${priorityTag}[${typeLabel}] ${feedback.title}`,
    html: generateNewTicketHtml(feedback),
    replyTo: feedback.userEmail,
  });
}

/**
 * Notify submitter of status change
 */
export async function notifyStatusChange(
  feedback: FeedbackData,
  oldStatus: string
): Promise<void> {
  // Only notify if we have the user's email
  if (!feedback.userEmail) {
    console.log('[FEEDBACK EMAIL] No user email, skipping status notification');
    return;
  }

  // Skip notification if status hasn't really changed
  if (oldStatus === feedback.status) {
    return;
  }

  const newStatusLabel = STATUS_LABELS[feedback.status] || feedback.status;

  await sendEmail({
    to: [feedback.userEmail],
    subject: `Your feedback is now "${newStatusLabel}" - ${feedback.title}`,
    html: generateStatusUpdateHtml(feedback, oldStatus),
  });
}

/**
 * Notify assignee when assigned to a ticket
 */
export async function notifyAssignment(
  feedback: FeedbackData,
  assigneeEmail: string
): Promise<void> {
  const typeLabel = TYPE_LABELS[feedback.type] || feedback.type;

  await sendEmail({
    to: [assigneeEmail],
    subject: `You've been assigned: [${typeLabel}] ${feedback.title}`,
    html: generateNewTicketHtml(feedback), // Reuse new ticket template
  });
}
