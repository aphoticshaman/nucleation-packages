# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability within nucleation packages, please follow these steps:

### Do NOT

- Open a public GitHub issue
- Post about it on social media
- Disclose it publicly before we've had a chance to fix it

### Do

1. **Email us directly** at: aphotic.noise@gmail.com

2. **Include the following information:**
   - Type of vulnerability (e.g., XSS, injection, authentication bypass)
   - Full path to the vulnerable file(s)
   - Step-by-step instructions to reproduce
   - Proof of concept code (if applicable)
   - Impact assessment

3. **Expect a response within 48 hours** acknowledging receipt

4. **Work with us** to understand and resolve the issue

### What to Expect

- **Acknowledgment:** Within 48 hours
- **Initial Assessment:** Within 7 days
- **Resolution Timeline:** Depends on severity
  - Critical: 24-72 hours
  - High: 7 days
  - Medium: 30 days
  - Low: 90 days

### Recognition

We believe in recognizing security researchers who help us:

- Credit in our CHANGELOG (unless you prefer anonymity)
- Credit in our security advisories
- We're working on a bug bounty program

## Security Best Practices for Users

### Webhook Server

If using `createWebhookProcessor()`, ensure you:

1. **Never expose it directly to the internet** without authentication
2. **Use a reverse proxy** (nginx, Cloudflare) with rate limiting
3. **Add authentication** via the `auth` config option
4. **Use HTTPS** in production

```javascript
// UNSAFE - Don't do this
createWebhookProcessor({ domain: 'finance', port: 8080 }).start();

// SAFER - Use with reverse proxy + auth
createWebhookProcessor({
  domain: 'finance',
  port: 8080,
  auth: {
    type: 'bearer',
    token: process.env.WEBHOOK_SECRET
  }
}).start();
```

### Environment Variables

- Never commit `.env` files
- Use secrets management (AWS Secrets Manager, Vault, etc.)
- Rotate credentials regularly

### Dependencies

- Run `npm audit` regularly
- Keep dependencies updated
- Review Dependabot PRs promptly

## Security Features

### Built-in Protections

- Input validation on all public APIs
- No use of `eval()` or `Function()` constructor
- No shell command execution
- Type-safe interfaces (TypeScript)
- Immutable state patterns

### WASM Isolation

The core algorithm runs in WebAssembly, providing:
- Memory isolation from JavaScript
- No access to filesystem or network
- Sandboxed execution environment

## Audit History

| Date | Auditor | Scope | Result |
|------|---------|-------|--------|
| TBD  | TBD     | Full  | Pending |

We plan to conduct professional security audits as the project matures.
