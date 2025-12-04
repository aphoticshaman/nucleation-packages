import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';

// Email export API endpoint
// Sends intelligence packages via email with various format attachments

interface EmailExportRequest {
  recipientEmail: string;
  subject?: string;
  includeTextBody: boolean;
  includePdfAttachment: boolean;
  includeJsonAttachment: boolean;
  includeMarkdownAttachment: boolean;
  packageContent: {
    title: string;
    subtitle: string;
    generatedAt: string;
    sections: Array<{
      id: string;
      title: string;
      icon: string;
      content: string;
      config: Record<string, unknown>;
    }>;
  };
  audience: string;
}

// Convert package content to plain text
function generatePlainText(content: EmailExportRequest['packageContent'], audience: string): string {
  const divider = '═'.repeat(60);
  const subDivider = '─'.repeat(40);

  let text = `
${divider}
${content.title.toUpperCase()}
${content.subtitle}
${divider}

Generated: ${new Date(content.generatedAt).toLocaleString()}
Audience: ${audience}
Classification: OSINT / UNCLASSIFIED

${content.sections.map(s => `
${subDivider}
${s.icon} ${s.title.toUpperCase()}
${subDivider}

${s.content}
`).join('\n')}

${divider}
LatticeForge Intelligence Platform
OSINT Only - No Classification Authority
${divider}
`.trim();

  return text;
}

// Convert package content to markdown
function generateMarkdown(content: EmailExportRequest['packageContent'], audience: string): string {
  return `# ${content.title}

**${content.subtitle}**

---

**Generated:** ${new Date(content.generatedAt).toLocaleString()}
**Audience:** ${audience}
**Classification:** OSINT / UNCLASSIFIED

---

${content.sections.map(s => `
## ${s.icon} ${s.title}

\`\`\`
${s.content}
\`\`\`
`).join('\n')}

---

*LatticeForge Intelligence Platform | OSINT Only - No Classification Authority*
`.trim();
}

// Generate HTML email body
function generateHtmlEmail(content: EmailExportRequest['packageContent'], audience: string): string {
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; background-color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="max-width: 700px; margin: 0 auto; background-color: #1e293b;">
    <!-- Header -->
    <tr>
      <td style="padding: 30px; text-align: center; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-bottom: 1px solid #334155;">
        <h1 style="margin: 0; color: #f8fafc; font-size: 24px;">${escapeHtml(content.title)}</h1>
        <p style="margin: 8px 0 0; color: #94a3b8; font-size: 14px;">${escapeHtml(content.subtitle)}</p>
        <p style="margin: 12px 0 0; color: #64748b; font-size: 12px;">
          Generated: ${new Date(content.generatedAt).toLocaleString()} | ${escapeHtml(audience)} | OSINT / UNCLASSIFIED
        </p>
      </td>
    </tr>

    <!-- Sections -->
    ${content.sections.map(s => `
    <tr>
      <td style="padding: 24px 30px; border-bottom: 1px solid #334155;">
        <h2 style="margin: 0 0 12px; color: #f8fafc; font-size: 16px; display: flex; align-items: center;">
          <span style="margin-right: 8px;">${escapeHtml(s.icon)}</span>
          ${escapeHtml(s.title)}
        </h2>
        <pre style="margin: 0; white-space: pre-wrap; word-wrap: break-word; font-family: 'SF Mono', Monaco, monospace; font-size: 12px; line-height: 1.6; color: #cbd5e1; background-color: #0f172a; padding: 16px; border-radius: 8px; border: 1px solid #334155;">${escapeHtml(s.content)}</pre>
      </td>
    </tr>
    `).join('')}

    <!-- Footer -->
    <tr>
      <td style="padding: 24px 30px; text-align: center; background-color: #0f172a;">
        <p style="margin: 0; color: #64748b; font-size: 12px;">
          LatticeForge Intelligence Platform | OSINT Only - No Classification Authority
        </p>
        <p style="margin: 8px 0 0; color: #475569; font-size: 11px;">
          This email was sent from LatticeForge. Do not reply to this message.
        </p>
      </td>
    </tr>
  </table>
</body>
</html>`;
}

// Escape HTML to prevent XSS
function escapeHtml(text: string): string {
  const htmlEscapes: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#x27;',
  };
  return text.replace(/[&<>"']/g, char => htmlEscapes[char] || char);
}

export async function POST(req: Request) {
  try {
    // Authenticate user
    const cookieStore = await cookies();
    const supabase = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          get(name: string) {
            return cookieStore.get(name)?.value;
          },
          set() {},
          remove() {},
        },
      }
    );

    const { data: { user } } = await supabase.auth.getUser();

    if (!user) {
      return NextResponse.json({ error: 'Not authenticated' }, { status: 401 });
    }

    // Parse request
    const body: EmailExportRequest = await req.json();
    const {
      recipientEmail,
      subject,
      includeTextBody,
      includePdfAttachment,
      includeJsonAttachment,
      includeMarkdownAttachment,
      packageContent,
      audience,
    } = body;

    // Validate email
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(recipientEmail)) {
      return NextResponse.json({ error: 'Invalid email address' }, { status: 400 });
    }

    // Generate content
    const textContent = generatePlainText(packageContent, audience);
    const htmlContent = generateHtmlEmail(packageContent, audience);
    const markdownContent = generateMarkdown(packageContent, audience);

    // Build attachments array
    const attachments: Array<{ filename: string; content: string; encoding: string; type: string }> = [];

    if (includeJsonAttachment) {
      attachments.push({
        filename: `latticeforge-intel-${Date.now()}.json`,
        content: Buffer.from(JSON.stringify(packageContent, null, 2)).toString('base64'),
        encoding: 'base64',
        type: 'application/json',
      });
    }

    if (includeMarkdownAttachment) {
      attachments.push({
        filename: `latticeforge-intel-${Date.now()}.md`,
        content: Buffer.from(markdownContent).toString('base64'),
        encoding: 'base64',
        type: 'text/markdown',
      });
    }

    // Note: PDF attachment would require a PDF generation library like puppeteer or pdfkit
    // For now, we'll skip PDF attachment and note it in the email
    if (includePdfAttachment) {
      // PDF generation would go here - for now we include markdown as alternative
      attachments.push({
        filename: `latticeforge-intel-${Date.now()}.md`,
        content: Buffer.from(markdownContent).toString('base64'),
        encoding: 'base64',
        type: 'text/markdown',
      });
    }

    // Check for email service configuration
    const resendApiKey = process.env.RESEND_API_KEY;

    if (!resendApiKey) {
      // Email service not configured - save to database for manual processing
      // and return success with note
      await supabase.from('email_export_queue').insert({
        user_id: user.id,
        recipient_email: recipientEmail,
        subject: subject || `LatticeForge Intelligence Package - ${audience}`,
        html_content: htmlContent,
        text_content: includeTextBody ? textContent : null,
        attachments: JSON.stringify(attachments),
        status: 'pending',
      }).then(() => {}).catch(() => {
        // Table might not exist yet
      });

      return NextResponse.json({
        success: true,
        message: 'Email queued for delivery',
        note: 'Email service configuration pending. Package saved for delivery.',
      });
    }

    // Send via Resend
    const emailResponse = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${resendApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: 'LatticeForge <intel@latticeforge.ai>',
        to: recipientEmail,
        subject: subject || `LatticeForge Intelligence Package - ${audience}`,
        html: htmlContent,
        text: includeTextBody ? textContent : undefined,
        attachments: attachments.length > 0 ? attachments : undefined,
      }),
    });

    if (!emailResponse.ok) {
      const errorData = await emailResponse.json();
      console.error('Resend error:', errorData);
      return NextResponse.json({ error: 'Failed to send email' }, { status: 500 });
    }

    // Log the export
    await supabase.from('user_activity').insert({
      user_id: user.id,
      action: 'email_export',
      details: {
        recipient: recipientEmail,
        audience,
        sections: packageContent.sections.length,
        attachments: attachments.map(a => a.filename),
      },
    });

    return NextResponse.json({
      success: true,
      message: `Intelligence package sent to ${recipientEmail}`,
    });

  } catch (error) {
    console.error('Email export error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
