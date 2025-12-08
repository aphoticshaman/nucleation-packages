# LatticeForge Setup Checklist

Complete configuration guide for Supabase, Stripe, Vercel, and Resend.

---

## 1. SUPABASE SETUP

### Database Migration
Run this SQL in **Supabase Dashboard → SQL Editor**:

```sql
-- Copy the entire contents of:
-- packages/web/supabase/migrations/20241208_fix_trial_and_alerts.sql
```

This adds:
- ✅ 14-day trial period (extended from 7)
- ✅ Email alert tracking columns on profiles
- ✅ email_export_preferences table
- ✅ alert_send_log table
- ✅ briefing_cache table
- ✅ Auto-create email prefs trigger
- ✅ RLS policies

### Email Templates (Auth)
Go to **Authentication → Email Templates** and paste templates from:
- `packages/web/supabase-email-templates/confirm-signup.html`
- `packages/web/supabase-email-templates/reset-password.html`
- `packages/web/supabase-email-templates/magic-link.html`

Set subjects:
- Confirm signup: `Confirm your LatticeForge account`
- Reset password: `Reset your LatticeForge password`
- Magic link: `Sign in to LatticeForge`

### Environment Variables
In **Project Settings → API**:
- Copy `URL` → `NEXT_PUBLIC_SUPABASE_URL`
- Copy `anon public` key → `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- Copy `service_role secret` key → `SUPABASE_SERVICE_ROLE_KEY`

---

## 2. STRIPE SETUP

### Create Products & Prices
In **Stripe Dashboard → Products**:

1. **Pro Plan** - $79/month
   - Create product: "LatticeForge Pro"
   - Add price: $79.00 USD, recurring monthly
   - Copy Price ID → `STRIPE_PRICE_PRO`

2. **Team Plan** - $59/seat/month
   - Create product: "LatticeForge Team"
   - Add price: $59.00 USD, recurring monthly, per unit
   - Copy Price ID → `STRIPE_PRICE_TEAM`

3. **Enterprise Plan** - Custom pricing
   - Create product: "LatticeForge Enterprise"
   - Add price: $0.00 USD (placeholder, quotes handled manually)
   - Copy Price ID → `STRIPE_PRICE_ENTERPRISE`

### Trial Settings
For Pro plan, enable trial:
- Go to Pro price → Edit
- Enable "Free trial" → 14 days
- ✅ "Require payment method upfront"

### Webhook Setup
In **Developers → Webhooks**:

1. Add endpoint: `https://latticeforge.ai/api/webhooks/stripe`
2. Select events:
   - `checkout.session.completed`
   - `customer.subscription.created`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
   - `invoice.payment_succeeded`
   - `invoice.payment_failed`
3. Copy Signing secret → `STRIPE_WEBHOOK_SECRET`

### API Keys
In **Developers → API keys**:
- Copy Secret key → `STRIPE_SECRET_KEY`
- Copy Publishable key → `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY`

---

## 3. VERCEL SETUP

### Environment Variables
In **Project Settings → Environment Variables**, add:

```
# Supabase
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_ROLE_KEY=eyJ...

# Stripe
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRICE_PRO=price_...
STRIPE_PRICE_TEAM=price_...
STRIPE_PRICE_ENTERPRISE=price_...
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_live_...

# Resend (for email)
RESEND_API_KEY=re_...

# Anthropic (for LLM briefings)
ANTHROPIC_API_KEY=sk-ant-...

# Redis/Upstash (for caching)
UPSTASH_REDIS_REST_URL=https://xxx.upstash.io
UPSTASH_REDIS_REST_TOKEN=...

# Internal service auth
INTERNAL_SERVICE_SECRET=<generate-random-32-char-string>
CRON_SECRET=<generate-random-32-char-string>
```

### Cron Jobs
Vercel reads `vercel.json` automatically. These crons are configured:

| Job | Schedule | Purpose |
|-----|----------|---------|
| `/api/cron/daily-alerts` | Every hour | Send daily/weekly email digests |
| `/api/cron/warm-cache` | Every 5 min | Keep briefings fresh |
| `/api/cron/rolling-country-update` | Every 30 min | Update nation intel |
| `/api/ingest/gdelt` | Every 4 hours | Fetch news/events |
| `/api/ingest/worldbank` | Weekly (Sun 6am) | Economic indicators |
| `/api/ingest/usgs` | Every 6 hours | Earthquake data |
| `/api/ingest/sentiment` | Every 2 hours | Market sentiment |
| `/api/compute/nation-risk` | Every 4 hours | Recalculate risk scores |
| `/api/analyze/cascades` | Daily midnight | Cascade analysis |

**Note:** Vercel Pro plan required for cron jobs.

---

## 4. RESEND SETUP

### Get API Key
1. Go to https://resend.com/api-keys
2. Create API key with "Full access"
3. Copy → `RESEND_API_KEY`

### Domain Setup
1. Go to **Domains** → Add domain
2. Add `latticeforge.ai`
3. Add DNS records (TXT, CNAME) to your DNS provider
4. Verify domain

### Sending Address
Once verified, emails will send from:
- `alerts@latticeforge.ai` - Daily/weekly digests
- `noreply@latticeforge.ai` - Transactional emails

---

## 5. QUICK VERIFICATION TESTS

### Test Trial System
```bash
# Create test user via Supabase SQL
SELECT * FROM profiles WHERE email = 'test@example.com';
-- Check: plan = 'trial', trial_ends_at = 14 days from now
```

### Test Email Cron (manual trigger)
```bash
curl -X GET "https://latticeforge.ai/api/cron/daily-alerts" \
  -H "Authorization: Bearer YOUR_CRON_SECRET"
```

### Test Stripe Webhook
```bash
stripe listen --forward-to localhost:3000/api/webhooks/stripe
# In another terminal:
stripe trigger checkout.session.completed
```

### Test Briefing API
```bash
curl -X POST "https://latticeforge.ai/api/intel-briefing" \
  -H "Content-Type: application/json" \
  -d '{"preset": "global"}'
# Should return template-engine response (no LLM cost)
```

---

## 6. MONITORING

### Check Cron Logs
- Vercel Dashboard → Deployments → Functions → Logs
- Filter by `/api/cron/daily-alerts`

### Check Email Sends
- Resend Dashboard → Emails
- Look for `alerts@latticeforge.ai` sends

### Database Health
```sql
-- Check active trials
SELECT COUNT(*) as active_trials FROM profiles WHERE plan = 'trial' AND trial_ends_at > NOW();

-- Check email prefs
SELECT frequency, COUNT(*) FROM email_export_preferences WHERE enabled = TRUE GROUP BY frequency;

-- Check alert sends today
SELECT COUNT(*) FROM alert_send_log WHERE sent_at > NOW() - INTERVAL '24 hours';
```

---

## Summary of What You Need to Do

### Supabase (5 min)
1. [ ] Run SQL migration in SQL Editor
2. [ ] Paste email templates in Auth → Email Templates

### Stripe (10 min)
1. [ ] Create 3 products (Pro, Team, Enterprise)
2. [ ] Add prices with correct IDs
3. [ ] Enable 14-day trial on Pro
4. [ ] Create webhook endpoint
5. [ ] Copy API keys and webhook secret

### Vercel (5 min)
1. [ ] Add all environment variables
2. [ ] Deploy (crons auto-configured from vercel.json)

### Resend (5 min)
1. [ ] Create account at resend.com
2. [ ] Get API key
3. [ ] Add and verify domain
