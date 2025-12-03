# LatticeForge Security Posture

## Threat Model

When this platform scales, expect:
- **Nation-state APTs** - Persistent, well-resourced attackers
- **Hacktivists** - DDoS, defacement attempts
- **Competitors** - Scraping, model theft attempts
- **Insider threats** - Credential compromise

## Architecture Principles

### 1. Zero Trust Everything

```
NEVER trust:
- User input (always sanitize)
- Internal services (always authenticate)
- Network location (always encrypt)
- Previous authentication (always re-verify)
```

### 2. Ephemeral Infrastructure

```
Inference servers:
- RunPod serverless = containers destroyed after use
- No persistent storage on inference nodes
- Model pulled fresh each cold start
- Nothing to steal, nothing to persist

Benefits:
- APT can't establish persistence
- Each request is isolated
- Compromise is temporary
```

### 3. Defense in Depth

```
Layer 1: Cloudflare (DDoS, WAF, bot detection)
Layer 2: Vercel Edge (geo-blocking, rate limiting)
Layer 3: API Gateway (auth, request signing)
Layer 4: Inference (input validation, output filtering)
Layer 5: Model (private, access-controlled)
Layer 6: Data (encrypted, row-level security)
```

## Critical Controls

### Authentication & Authorization

```typescript
// All requests must have:
1. Valid JWT from Supabase Auth
2. HMAC signature (prevents tampering)
3. Timestamp + nonce (prevents replay)
4. User in allowed tier (rate limits)
```

### Request Signing (Client → Server)

```python
# Every request signed with:
signature = HMAC-SHA256(
    key=SIGNING_KEY,
    message=f"{timestamp}:{nonce}:{json.dumps(payload)}"
)

# Server verifies:
1. Timestamp within 5 minutes
2. Nonce not seen before
3. Signature matches
```

### Secrets Management

```
NEVER in code:
- API keys
- Signing keys
- Database credentials

ALWAYS:
- Environment variables
- Rotated regularly
- Different per environment
- Audit access logs
```

### Rate Limiting (Per User)

```
Free tier:    10 requests/minute,   100/hour
Pro tier:     60 requests/minute,   500/hour
Enterprise:   Custom limits

Burst detection:
- >20 requests in 10 seconds = flag
- >50 requests in 10 seconds = block
```

## Monitoring & Detection

### Anomaly Indicators

```python
HIGH_RISK_PATTERNS = [
    "Requests from new IP at 3am local time",
    "Sudden spike in API usage",
    "Requests for unusual endpoints",
    "Malformed payloads",
    "Known attack signatures in input",
    "Attempts to access other users' data",
]
```

### Audit Logging

Every request logged with:
- Timestamp
- User ID (hashed)
- Request hash
- Response hash
- Client IP (hashed)
- Anomaly score

Logs are:
- Append-only
- Integrity-chained (each entry hashes previous)
- Retained 90 days minimum
- Exported to cold storage

### Canary Tokens

Plant fake credentials/data that trigger alerts:
- Fake API keys in decoy locations
- Honeypot endpoints
- Trigger = immediate incident response

## Incident Response

### Severity Levels

```
SEV 1 - Critical
- Active data breach
- Model compromise
- Auth bypass
Response: Immediate (minutes)

SEV 2 - High
- Detected intrusion attempt
- Anomaly spike
- Canary triggered
Response: Within 1 hour

SEV 3 - Medium
- Failed auth attempts
- Rate limit violations
- Suspicious patterns
Response: Within 24 hours
```

### Response Playbook

```
1. CONTAIN
   - Revoke compromised credentials
   - Block suspicious IPs
   - Disable affected endpoints

2. INVESTIGATE
   - Pull audit logs
   - Identify attack vector
   - Assess data exposure

3. REMEDIATE
   - Patch vulnerability
   - Rotate all secrets
   - Update WAF rules

4. NOTIFY
   - Affected users (if data exposed)
   - Legal (if required)
   - Document incident
```

## Deployment Checklist

### Before Going Public

- [ ] Cloudflare in front of everything
- [ ] All secrets in environment variables
- [ ] Request signing enabled
- [ ] Rate limiting configured
- [ ] Audit logging enabled
- [ ] Anomaly detection active
- [ ] Canary tokens planted
- [ ] Incident response plan documented
- [ ] Security contact published
- [ ] Bug bounty program (optional but recommended)

### Regular Tasks

- [ ] Weekly: Review anomaly alerts
- [ ] Monthly: Rotate API keys
- [ ] Monthly: Review access logs
- [ ] Quarterly: Penetration test
- [ ] Quarterly: Dependency audit

## Service-Specific Hardening

### Vercel

```
vercel.json:
{
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "X-Frame-Options", "value": "DENY" },
        { "key": "X-Content-Type-Options", "value": "nosniff" },
        { "key": "Strict-Transport-Security", "value": "max-age=31536000" },
        { "key": "Content-Security-Policy", "value": "default-src 'self'" }
      ]
    }
  ]
}
```

### Supabase

```sql
-- Row Level Security on ALL tables
ALTER TABLE intel_reports ENABLE ROW LEVEL SECURITY;

-- Users can only see their own data
CREATE POLICY "Users see own data" ON intel_reports
  FOR SELECT USING (auth.uid() = user_id);

-- Audit table (append-only)
CREATE TABLE audit_log (
  id uuid PRIMARY KEY,
  timestamp timestamptz DEFAULT now(),
  user_id uuid,
  action text,
  resource text,
  details jsonb
);

-- No updates or deletes on audit
REVOKE UPDATE, DELETE ON audit_log FROM authenticated;
```

### RunPod

```python
# Environment variables only
SIGNING_KEY=...
HF_TOKEN=...  # Read-only token
CANARY_TOKEN=...

# No persistent storage
# No SSH access in production
# Logs shipped to external service
```

### HuggingFace

```
- Private repository
- 2FA enabled
- Read-only tokens for inference
- Write tokens only for CI/CD (separate account)
- Access logs monitored
```

## Cost of Breach

If compromised:
- User trust = destroyed
- Legal liability = significant
- Reputation = years to rebuild
- Competitive advantage = lost

Investment in security = insurance policy.

## Resources

- OWASP Top 10: https://owasp.org/Top10/
- NIST Cybersecurity Framework: https://www.nist.gov/cyberframework
- CIS Controls: https://www.cisecurity.org/controls
