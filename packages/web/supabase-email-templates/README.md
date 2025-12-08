# Supabase Email Templates

Branded email templates for LatticeForge authentication flows.

## Setup Instructions

1. Go to your Supabase Dashboard: https://supabase.com/dashboard
2. Select your project
3. Navigate to **Authentication** → **Email Templates**

### Templates to Configure:

| Template Type | File | Subject Line |
|--------------|------|--------------|
| Confirm Signup | `confirm-signup.html` | Confirm your LatticeForge account |
| Reset Password | `reset-password.html` | Reset your LatticeForge password |
| Magic Link | `magic-link.html` | Sign in to LatticeForge |

### For Each Template:

1. Click on the template type (e.g., "Confirm signup")
2. Edit the **Subject** field
3. Copy the contents of the corresponding `.html` file
4. Paste into the **Message** (HTML) editor
5. Click **Save**

## Template Variables

Supabase uses Go template syntax. These variables are available:

- `{{ .ConfirmationURL }}` - The action URL (confirm, reset, magic link)
- `{{ .Email }}` - User's email address
- `{{ .Token }}` - The raw token (rarely used)
- `{{ .TokenHash }}` - Hashed token (rarely used)
- `{{ .SiteURL }}` - Your site URL from project settings

## Image Hosting

The templates reference images from `https://latticeforge.ai/images/...`

Required images to host:
- `/images/brand/logo-full-white.png` (180px wide)
- `/images/icons/shield-check.png` (32x32)
- `/images/icons/key.png` (32x32)
- `/images/icons/magic-wand.png` (32x32)

**Alternative:** Replace image URLs with emoji or inline SVG if you don't want to host images.

## Testing

After saving templates:
1. Create a test account with a new email
2. Use "Forgot password" to test reset flow
3. Check both inbox and spam folder

## Customization

- **Colors:** Edit the gradient values (`#2563eb`, `#f97316`, `#10b981`)
- **Logo:** Update the logo URL and width
- **Footer links:** Update Twitter handle and support email
