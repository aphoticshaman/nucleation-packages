# LatticeForge Email Templates

Branded email templates for authentication and notification flows.

## Logo Reference

**App Icon:** `aphoticshaman_App_icon_for_LatticeForge_intelligence_platform_fd831ec4-b981-45c0-b01e-5a64eaf2f52c_2`

Host at: `https://latticeforge.ai/images/brand/app-icon.png`

## Setup Instructions

### Supabase Auth Templates

1. Go to your Supabase Dashboard: https://supabase.com/dashboard
2. Select your project
3. Navigate to **Authentication** → **Email Templates**

#### Templates to Configure:

| Template Type | File | Subject Line |
|--------------|------|--------------|
| Confirm Signup | `confirm-signup.html` | Confirm your LatticeForge account |
| Reset Password | `reset-password.html` | Reset your LatticeForge password |
| Magic Link | `magic-link.html` | Sign in to LatticeForge |
| Invite User | `invite-user.html` | You're Invited to LatticeForge |
| Change Email | `change-email.html` | Confirm your new email address |

#### For Each Template:

1. Click on the template type (e.g., "Confirm signup")
2. Edit the **Subject** field
3. Copy the contents of the corresponding `.html` file
4. Paste into the **Message** (HTML) editor
5. Click **Save**

### Resend Templates (Programmatic)

Located in `lib/email.ts` - sent via Resend API:

| Template | Use Case | Trigger |
|----------|----------|---------|
| `paymentFailed` | Billing failure notification | Stripe webhook |
| `subscriptionCanceled` | Subscription canceled | Stripe webhook |
| `welcomeSubscription` | Welcome + trial start | Stripe webhook |
| `trialEndingReminder` | Trial expiry warning | Cron job |
| `paymentSucceeded` | Payment receipt | Stripe webhook |
| `dailyBriefing` | Morning intel digest | `/api/cron/daily-alerts` |
| `alertNotification` | Real-time risk alerts | Alert system |
| `weeklyDigest` | Weekly summary | Cron job (Sunday) |

## Template Variables

### Supabase (Go template syntax):

- `{{ .ConfirmationURL }}` - The action URL (confirm, reset, magic link)
- `{{ .Email }}` - User's email address
- `{{ .Token }}` - The raw token (rarely used)
- `{{ .TokenHash }}` - Hashed token (rarely used)
- `{{ .SiteURL }}` - Your site URL from project settings

### Resend (TypeScript):

Templates are functions that accept data objects. See `lib/email.ts` for type definitions.

## Image Hosting

Templates reference images from `https://latticeforge.ai/images/...`

Required images to host:
- `/images/brand/app-icon.png` (64x64, rounded corners) - **Main app icon**
- `/images/brand/logo-full-white.png` (180px wide) - Full logo
- `/images/icons/shield-check.png` (32x32) - Confirmation icon
- `/images/icons/key.png` (32x32) - Password reset icon
- `/images/icons/magic-wand.png` (32x32) - Magic link icon

**Alternative:** The newer templates use HTML entities/emoji as fallback if images aren't hosted.

## Testing

### Supabase templates:
1. Create a test account with a new email
2. Use "Forgot password" to test reset flow
3. Check both inbox and spam folder

### Resend templates:
```typescript
import { sendEmail, emailTemplates } from '@/lib/email';

// Test daily briefing
const briefing = emailTemplates.dailyBriefing('Test User', {
  date: 'December 10, 2025',
  globalRiskLevel: 'moderate',
  topAlerts: [{ country: 'Ukraine', summary: 'Conflict escalation', severity: 'high' }],
  watchlistUpdates: [],
  keyInsights: ['Regional tensions elevated'],
});

await sendEmail({ to: 'test@example.com', ...briefing });
```

## Customization

- **Colors:** LatticeForge brand palette
  - Primary Blue: `#2563eb` → `#3b82f6`
  - Orange/Amber: `#f97316` → `#fbbf24`
  - Success Green: `#10b981` → `#4ade80`
  - Dark BG: `#0a0a0f`, `#12121a`
- **Logo:** Update the logo URL in all templates
- **Footer links:** Update Twitter handle and support email
