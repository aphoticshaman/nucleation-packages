# Authentication Troubleshooting Guide
## Fixing Google OAuth and Custom Domain Issues

---

## Current Issues

### Issue 1: Google Login Cycles Twice, Returns to Login Page
**Symptom:** User clicks "Continue with Google", selects account, clicks continue, then gets sent back to login page.

### Issue 2: Auth Shows Raw Supabase URL Instead of Custom Domain
**Symptom:** Auth popups/redirects show `sylqhqewqtggjqgwdsjf.supabase.co` instead of `auth.latticeforge.ai`.

---

## Root Cause Analysis

Both issues are **configuration problems** in external dashboards, not code issues. The code in `packages/web` is correct.

### Google OAuth Loop Causes

1. **Mismatched Redirect URIs** - Google's authorized redirect URIs don't match what Supabase sends
2. **Missing Site URL** - Supabase doesn't know the production domain
3. **Callback URL Mismatch** - The OAuth callback path isn't whitelisted

### Custom Domain Issue Cause

Supabase custom domains require specific configuration in the Supabase dashboard and DNS setup. Simply creating a CNAME for `auth.latticeforge.ai` isn't sufficient.

---

## Fix: Google OAuth Configuration

### Step 1: Supabase Dashboard Configuration

1. Go to: **Supabase Dashboard → Project → Authentication → URL Configuration**

2. Set these values:
   ```
   Site URL: https://latticeforge.ai

   Redirect URLs (add all of these):
   - https://latticeforge.ai/**
   - https://latticeforge.ai/auth/callback
   - https://www.latticeforge.ai/**
   - https://www.latticeforge.ai/auth/callback
   - http://localhost:3000/**  (for development)
   - http://localhost:3000/auth/callback
   ```

### Step 2: Google Cloud Console Configuration

1. Go to: **Google Cloud Console → APIs & Services → Credentials**

2. Select your OAuth 2.0 Client ID

3. Under **Authorized JavaScript origins**, add:
   ```
   https://latticeforge.ai
   https://www.latticeforge.ai
   http://localhost:3000
   ```

4. Under **Authorized redirect URIs**, add:
   ```
   https://sylqhqewqtggjqgwdsjf.supabase.co/auth/v1/callback
   https://latticeforge.ai/auth/callback
   https://www.latticeforge.ai/auth/callback
   http://localhost:3000/auth/callback
   ```

   **Note:** You MUST include the Supabase callback URL (`sylqhqewqtggjqgwdsjf.supabase.co/auth/v1/callback`) because that's where Google redirects first, then Supabase redirects to your app.

### Step 3: Verify Supabase Google Provider Settings

1. Go to: **Supabase Dashboard → Authentication → Providers → Google**

2. Ensure:
   - Provider is **enabled**
   - Client ID matches Google Cloud Console
   - Client Secret matches Google Cloud Console

---

## Fix: Custom Auth Domain

### Understanding the Flow

```
User clicks "Login with Google"
    ↓
Your app redirects to Supabase
    ↓
Supabase redirects to Google
    ↓
Google authenticates, redirects to Supabase callback
    ↓
Supabase creates session, redirects to your app
```

The "Supabase" step in the middle shows the raw Supabase URL by default.

### Option A: Supabase Custom Domain (Recommended)

Supabase Pro plan includes custom domain support.

1. Go to: **Supabase Dashboard → Settings → Custom Domains**

2. Add your custom domain: `auth.latticeforge.ai`

3. Supabase will provide DNS records to add:
   ```
   Type: CNAME
   Name: auth
   Value: <provided-by-supabase>.supabase.co
   ```

4. Verify domain in Supabase dashboard

5. Update your `.env`:
   ```
   NEXT_PUBLIC_SUPABASE_URL=https://auth.latticeforge.ai
   ```

**Note:** Custom domains require Supabase Pro plan ($25/month) or higher.

### Option B: If on Free Plan

On the free plan, you cannot change the Supabase URL shown during OAuth. The workaround is:

1. Accept that users will see `*.supabase.co` during OAuth
2. Focus on ensuring the rest of the experience uses your domain
3. Upgrade to Pro when budget allows

---

## Debugging Steps

### Test 1: Check OAuth Flow in Browser DevTools

1. Open DevTools → Network tab
2. Click "Continue with Google"
3. Watch the redirect chain:
   ```
   latticeforge.ai/login →
   sylqhqewqtggjqgwdsjf.supabase.co/auth/v1/authorize →
   accounts.google.com/... →
   sylqhqewqtggjqgwdsjf.supabase.co/auth/v1/callback →
   latticeforge.ai/auth/callback
   ```

4. If the chain breaks, check which redirect fails

### Test 2: Check for Errors in Supabase Logs

1. Go to: **Supabase Dashboard → Logs → Auth**
2. Look for errors during the login attempt
3. Common errors:
   - `redirect_uri_mismatch` - Google Console redirect URIs wrong
   - `invalid_request` - Missing parameters
   - `access_denied` - User denied consent

### Test 3: Check Browser Console

After the OAuth loop, check the browser console for:
- JavaScript errors
- Failed network requests
- Session storage issues

---

## Code Fixes Applied (December 2024)

### Fix 1: Added `/auth/callback` to Middleware Matcher
The middleware wasn't processing the OAuth callback route, so session cookies weren't being set properly on the server side.

**File:** `packages/web/middleware.ts`
```typescript
export const config = {
  matcher: [
    // ... other routes
    // Auth callback - CRITICAL: Must be included for session cookies to be set
    '/auth/callback',
  ],
};
```

### Fix 2: Excluded Auth Routes from COOP/COEP Headers
The `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers were being applied globally, which can break OAuth flows.

**File:** `packages/web/next.config.js`
```javascript
async headers() {
  return [
    {
      // Apply COOP/COEP only to app routes that need WASM, not auth routes
      source: '/(app|dashboard|admin)/:path*',
      // ...
    },
  ];
},
```

### Fix 3: Improved Auth Callback Handler
Added PKCE code exchange support and better error handling.

**File:** `packages/web/app/auth/callback/page.tsx`
- Added explicit code exchange via `exchangeCodeForSession()`
- Added OAuth error parameter handling
- Added visible error state display
- Added try-catch for unexpected errors

---

## Code Verification

The current code is correct. Here's what it does:

**Login Page (`app/(auth)/login/page.tsx`):**
```typescript
const handleOAuthLogin = async (provider: 'google' | 'github') => {
  const { error } = await supabase.auth.signInWithOAuth({
    provider,
    options: {
      redirectTo: `${window.location.origin}/auth/callback?redirect=${redirect}`,
    },
  });
  // ...
};
```

**Auth Callback (`app/auth/callback/page.tsx`):**
```typescript
useEffect(() => {
  const handleCallback = async () => {
    const { data: { session }, error } = await supabase.auth.getSession();

    if (session) {
      router.push(redirect);
    } else {
      // Listen for auth state change...
    }
  };
  // ...
}, []);
```

This is the standard Supabase OAuth flow. No code changes needed.

---

## Quick Checklist

- [ ] Supabase Site URL set to `https://latticeforge.ai`
- [ ] Supabase Redirect URLs include production and callback paths
- [ ] Google Cloud authorized origins include your domain
- [ ] Google Cloud redirect URIs include **Supabase callback URL**
- [ ] Google provider enabled in Supabase with correct credentials
- [ ] (Optional) Custom domain configured in Supabase Pro

---

## Contact Points

- **Supabase Support:** support@supabase.io
- **Google Cloud Console:** https://console.cloud.google.com
- **Supabase Dashboard:** https://app.supabase.com

---

*Document Version 1.0 | Authentication Troubleshooting*
*Last Updated: December 2024*
