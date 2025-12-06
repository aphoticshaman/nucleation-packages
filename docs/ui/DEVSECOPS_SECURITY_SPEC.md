# LatticeForge DevSecOps Security Specification

## Document Purpose

This specification provides comprehensive security guidance for LatticeForge. It covers threat modeling, authentication architecture, data protection, input validation, secure development practices, incident response, and compliance requirements. Security engineers and developers alike should use this as the authoritative reference for building and maintaining secure systems.

---

## 1. Security Architecture Overview

### 1.1 Security Principles

**Defense in Depth**
No single security control is sufficient. Multiple layers of security ensure that if one fails, others remain. Every component assumes breach of adjacent components.

**Principle of Least Privilege**
Every process, user, and service operates with the minimum privileges necessary. Elevated access is temporary, audited, and justified.

**Secure by Default**
New features default to the most restrictive settings. Users must explicitly enable less secure configurations.

**Zero Trust**
No implicit trust based on network location or prior authentication. Every request is verified, every session is validated.

**Fail Securely**
When errors occur, systems fail to a secure state. Errors never expose sensitive data or bypass security controls.

### 1.2 Security Boundaries

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INTERNET (Untrusted)                          │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                              [WAF / CDN]
                                     │
┌────────────────────────────────────┴────────────────────────────────────┐
│                        DMZ (Semi-Trusted)                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  Load Balancer  │  │  API Gateway    │  │  Rate Limiter   │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────┴────────────────────────────────────┐
│                     APPLICATION TIER (Internal)                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  Web Servers    │  │  API Servers    │  │  Worker Nodes   │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────┴────────────────────────────────────┐
│                       DATA TIER (Restricted)                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  PostgreSQL     │  │  Redis          │  │  Object Storage │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Trust Boundaries

| Boundary | Trust Level | Controls |
|----------|-------------|----------|
| Internet → DMZ | Untrusted | WAF, DDoS protection, TLS termination |
| DMZ → Application | Semi-trusted | Authentication, rate limiting, input validation |
| Application → Data | Trusted | Authorization, encryption, audit logging |
| Application → AI Services | Trusted | API keys, request signing, output sanitization |
| Cross-tenant | Isolated | Tenant ID verification, RLS, namespace separation |

---

## 2. Threat Model

### 2.1 Threat Actors

| Actor | Capability | Motivation | Likelihood |
|-------|------------|------------|------------|
| Script Kiddies | Low | Mischief, reputation | High |
| Credential Stuffers | Medium | Account takeover | High |
| Competitors | Medium | Data theft, disruption | Medium |
| Nation-State | High | Espionage, research theft | Low |
| Malicious Insider | High | Financial, revenge | Low |
| Researchers | Medium | Bug bounty, reputation | Medium |

### 2.2 Asset Classification

**Critical Assets:**
| Asset | Description | Impact if Compromised |
|-------|-------------|----------------------|
| User credentials | Passwords, tokens, API keys | Full account takeover |
| Research data | Sources, syntheses, insights | Competitive harm, IP theft |
| AI model outputs | Generated content | Misinformation, liability |
| Billing data | Payment methods, invoices | Financial fraud |

**Sensitive Assets:**
| Asset | Description | Impact if Compromised |
|-------|-------------|----------------------|
| User PII | Names, emails, affiliations | Privacy violation, GDPR fines |
| Usage analytics | Feature usage, search queries | Competitive intelligence |
| System configurations | Infrastructure secrets | Lateral movement |

### 2.3 STRIDE Analysis

**Spoofing Identity:**
- Threat: Attacker impersonates legitimate user
- Controls: MFA, session binding, device fingerprinting
- Residual Risk: SIM swapping for SMS-based MFA

**Tampering:**
- Threat: Attacker modifies data in transit or at rest
- Controls: TLS, signed requests, database integrity checks
- Residual Risk: Compromised application server

**Repudiation:**
- Threat: User denies actions they performed
- Controls: Comprehensive audit logging, tamper-evident logs
- Residual Risk: Log retention limits

**Information Disclosure:**
- Threat: Unauthorized access to sensitive data
- Controls: Encryption, access controls, data masking
- Residual Risk: Memory-based attacks, side channels

**Denial of Service:**
- Threat: Service unavailability
- Controls: Rate limiting, auto-scaling, redundancy
- Residual Risk: Application-layer DoS, volumetric attacks

**Elevation of Privilege:**
- Threat: User gains unauthorized permissions
- Controls: RBAC, input validation, sandboxing
- Residual Risk: Zero-day vulnerabilities

### 2.4 Attack Tree Summary

```
Root: Unauthorized Access to Research Data
├── 1. Credential Theft
│   ├── 1.1 Phishing (MITIGATED: MFA, security training)
│   ├── 1.2 Credential stuffing (MITIGATED: rate limiting, breach detection)
│   └── 1.3 Session hijacking (MITIGATED: secure cookies, session binding)
├── 2. Application Vulnerability
│   ├── 2.1 SQL Injection (MITIGATED: parameterized queries, ORM)
│   ├── 2.2 XSS (MITIGATED: CSP, output encoding, Rust/WASM)
│   ├── 2.3 IDOR (MITIGATED: authorization checks, RLS)
│   └── 2.4 SSRF (MITIGATED: URL allowlisting, network segmentation)
├── 3. Infrastructure Compromise
│   ├── 3.1 Misconfiguration (MITIGATED: IaC, security scanning)
│   ├── 3.2 Unpatched systems (MITIGATED: automated patching, SBOMs)
│   └── 3.3 Supply chain attack (MITIGATED: dependency scanning, lockfiles)
└── 4. Insider Threat
    ├── 4.1 Malicious admin (MITIGATED: audit logging, least privilege)
    └── 4.2 Compromised developer (MITIGATED: code review, signed commits)
```

---

## 3. Authentication System

### 3.1 Authentication Methods

**Primary: Email/Password with MFA**
```
┌──────────────────────────────────────────────────────────────────┐
│                     Authentication Flow                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. User submits email/password                                 │
│                    ↓                                             │
│  2. Server validates credentials (argon2id hash)                │
│                    ↓                                             │
│  3. If MFA enabled: challenge user for TOTP/WebAuthn            │
│                    ↓                                             │
│  4. Generate JWT access token (15min) + refresh token (7days)   │
│                    ↓                                             │
│  5. Set secure, httpOnly cookies                                │
│                    ↓                                             │
│  6. Log authentication event with context                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**OAuth 2.0 / OIDC:**
- Supported providers: Google, GitHub, ORCID
- PKCE flow required (no implicit grants)
- State parameter for CSRF protection
- Nonce validation for ID tokens

**API Keys:**
- For service-to-service and programmatic access
- Scoped permissions (read-only, full access)
- Rotatable without user intervention
- Rate limited independently

### 3.2 Password Requirements

```rust
struct PasswordPolicy {
    min_length: 12,
    max_length: 128,
    require_uppercase: false,  // NIST 800-63B: no composition rules
    require_lowercase: false,
    require_digit: false,
    require_special: false,
    check_breached: true,      // Check against HaveIBeenPwned
    check_common: true,        // Block common passwords
    check_context: true,       // Block email/name in password
}
```

**Password Storage:**
- Algorithm: Argon2id
- Memory: 64MB
- Iterations: 3
- Parallelism: 4
- Salt: 16 bytes random per password

### 3.3 Session Management

**JWT Structure:**
```json
{
  "header": {
    "alg": "RS256",
    "typ": "JWT",
    "kid": "key-2024-03"
  },
  "payload": {
    "sub": "user_abc123",
    "iss": "https://api.latticeforge.io",
    "aud": "latticeforge",
    "iat": 1699900000,
    "exp": 1699900900,
    "jti": "session_xyz789",
    "org_id": "org_123",
    "role": "member",
    "permissions": ["stream:read", "stream:write"]
  }
}
```

**Token Lifetimes:**
| Token Type | Lifetime | Storage |
|------------|----------|---------|
| Access Token | 15 minutes | Memory/httpOnly cookie |
| Refresh Token | 7 days | httpOnly cookie + DB reference |
| API Key | Until revoked | Client-stored securely |

**Session Binding:**
- Bind refresh tokens to user agent hash
- Bind to IP range (optional, configurable)
- Device fingerprint for anomaly detection

### 3.4 Multi-Factor Authentication

**Supported Methods:**
1. TOTP (Authenticator apps) - Default
2. WebAuthn/Passkeys - Recommended
3. SMS OTP - Discouraged, enterprise only
4. Recovery codes - 10 single-use codes

**TOTP Implementation:**
```rust
struct TotpConfig {
    algorithm: "SHA1",
    digits: 6,
    period: 30,
    issuer: "LatticeForge",
    // Allow ±1 time step for clock skew
    allowed_drift: 1,
}
```

**WebAuthn Implementation:**
- Resident keys preferred
- User verification required
- Attestation: None (privacy-preserving)
- Challenge: 32 bytes random

### 3.5 Account Recovery

**Forgot Password Flow:**
1. User requests reset via email
2. Generate time-limited token (1 hour)
3. Send email with unique link
4. Token is single-use, invalidated on use or password change
5. Require re-authentication for sensitive actions post-reset

**Account Lockout:**
- 5 failed attempts → 15 minute lockout
- 10 failed attempts → 1 hour lockout + email notification
- 20 failed attempts → Account locked, manual review required
- Lockout applies per-account, not per-IP (prevents lockout attacks)

---

## 4. Authorization System

### 4.1 Role-Based Access Control

**System Roles:**
```rust
enum SystemRole {
    SuperAdmin,    // Platform operations (internal only)
    Support,       // Customer support access
}

enum OrganizationRole {
    Owner,         // Full control, billing
    Admin,         // Member management, settings
    Member,        // Standard access
    Viewer,        // Read-only access
    Guest,         // Limited shared access
}
```

**Permission Matrix:**
| Permission | Owner | Admin | Member | Viewer | Guest |
|------------|-------|-------|--------|--------|-------|
| org.manage | ✓ | | | | |
| org.billing | ✓ | | | | |
| members.invite | ✓ | ✓ | | | |
| members.remove | ✓ | ✓ | | | |
| stream.create | ✓ | ✓ | ✓ | | |
| stream.read (own) | ✓ | ✓ | ✓ | | |
| stream.read (shared) | ✓ | ✓ | ✓ | ✓ | ✓ |
| stream.write (own) | ✓ | ✓ | ✓ | | |
| stream.delete (own) | ✓ | ✓ | ✓ | | |
| stream.share | ✓ | ✓ | ✓ | | |

### 4.2 Resource-Level Authorization

**Ownership Model:**
```sql
-- Every resource has an owner and optional sharing
CREATE TABLE streams (
    id UUID PRIMARY KEY,
    owner_id UUID REFERENCES users(id),
    workspace_id UUID REFERENCES workspaces(id),
    visibility visibility_enum DEFAULT 'private'
);

CREATE TABLE stream_shares (
    stream_id UUID REFERENCES streams(id),
    shared_with_id UUID,  -- user_id or team_id
    shared_with_type share_target_enum,
    permission permission_enum,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Row-Level Security:**
```sql
-- Users can only see their own streams or shared streams
CREATE POLICY stream_access ON streams
    FOR SELECT
    USING (
        owner_id = auth.uid()
        OR workspace_id IN (SELECT workspace_id FROM workspace_members WHERE user_id = auth.uid())
        OR id IN (SELECT stream_id FROM stream_shares WHERE shared_with_id = auth.uid())
        OR visibility = 'public'
    );

-- Users can only modify their own streams
CREATE POLICY stream_modify ON streams
    FOR UPDATE
    USING (owner_id = auth.uid())
    WITH CHECK (owner_id = auth.uid());
```

### 4.3 API Authorization

**Request Authorization Flow:**
```rust
async fn authorize_request(
    req: &Request,
    required_permission: Permission,
) -> Result<AuthContext, AuthError> {
    // 1. Extract and validate token
    let token = extract_token(req)?;
    let claims = validate_jwt(token)?;

    // 2. Check token not revoked
    if is_token_revoked(&claims.jti).await? {
        return Err(AuthError::TokenRevoked);
    }

    // 3. Load user permissions
    let permissions = load_user_permissions(&claims.sub).await?;

    // 4. Check required permission
    if !permissions.contains(&required_permission) {
        audit_log(AuditEvent::PermissionDenied {
            user_id: claims.sub,
            resource: req.path(),
            required: required_permission,
        });
        return Err(AuthError::Forbidden);
    }

    // 5. Return authorized context
    Ok(AuthContext {
        user_id: claims.sub,
        org_id: claims.org_id,
        role: claims.role,
        permissions,
    })
}
```

### 4.4 IDOR Prevention

**Direct Object Reference Checks:**
```rust
// BAD: Trust user-provided ID
async fn get_stream(stream_id: &str) -> Result<Stream> {
    db.get_stream(stream_id).await
}

// GOOD: Verify ownership/access
async fn get_stream(auth: &AuthContext, stream_id: &str) -> Result<Stream> {
    let stream = db.get_stream(stream_id).await?;

    // Verify access
    if !can_access_stream(auth, &stream) {
        return Err(Error::NotFound);  // Don't reveal existence
    }

    Ok(stream)
}

fn can_access_stream(auth: &AuthContext, stream: &Stream) -> bool {
    stream.owner_id == auth.user_id
        || stream.workspace_id == auth.org_id
        || stream.shares.iter().any(|s| s.user_id == auth.user_id)
        || stream.visibility == Visibility::Public
}
```

---

## 5. Input Validation and Sanitization

### 5.1 Validation Strategy

**Defense Layers:**
1. Schema validation (type, format)
2. Business rule validation (ranges, relationships)
3. Security validation (injection prevention)
4. Contextual validation (authorization)

### 5.2 Schema Validation

**Request Validation:**
```rust
#[derive(Deserialize, Validate)]
struct CreateStreamRequest {
    #[validate(length(min = 1, max = 255))]
    name: String,

    #[validate(length(max = 2000))]
    description: Option<String>,

    #[validate(custom = "validate_visibility")]
    visibility: Visibility,
}

fn validate_visibility(visibility: &Visibility) -> Result<(), ValidationError> {
    match visibility {
        Visibility::Public if !feature_enabled("public_streams") => {
            Err(ValidationError::new("public_streams_disabled"))
        }
        _ => Ok(())
    }
}
```

### 5.3 SQL Injection Prevention

**Parameterized Queries Only:**
```rust
// BAD: String interpolation
let query = format!("SELECT * FROM users WHERE email = '{}'", email);

// GOOD: Parameterized query
let user = sqlx::query_as!(
    User,
    "SELECT * FROM users WHERE email = $1",
    email
).fetch_optional(&pool).await?;

// GOOD: Query builder with type safety
let streams = Stream::find()
    .filter(stream::Column::OwnerId.eq(user_id))
    .filter(stream::Column::Name.contains(&search_term))
    .all(&db)
    .await?;
```

### 5.4 XSS Prevention

**Content Security Policy:**
```rust
fn security_headers() -> impl IntoResponse {
    [
        ("Content-Security-Policy",
         "default-src 'self'; \
          script-src 'self' 'wasm-unsafe-eval'; \
          style-src 'self' 'unsafe-inline'; \
          img-src 'self' data: https:; \
          connect-src 'self' wss://api.latticeforge.io; \
          frame-ancestors 'none'; \
          base-uri 'self'; \
          form-action 'self';"),
        ("X-Content-Type-Options", "nosniff"),
        ("X-Frame-Options", "DENY"),
        ("X-XSS-Protection", "0"),  // Disabled, CSP handles it
        ("Referrer-Policy", "strict-origin-when-cross-origin"),
    ]
}
```

**Output Encoding:**
```rust
// In Leptos (Rust frontend), strings are escaped by default
view! {
    <p>{user_provided_content}</p>  // Automatically escaped
}

// For intentional HTML (rare), use explicit unsafe
view! {
    <div inner_html={sanitize_html(content)}></div>
}

fn sanitize_html(content: &str) -> String {
    ammonia::clean(content)  // Whitelist-based HTML sanitizer
}
```

### 5.5 URL and File Handling

**URL Validation:**
```rust
fn validate_source_url(url: &str) -> Result<Url, ValidationError> {
    let parsed = Url::parse(url)?;

    // Scheme whitelist
    if !["http", "https"].contains(&parsed.scheme()) {
        return Err(ValidationError::new("invalid_scheme"));
    }

    // Block internal IPs (SSRF prevention)
    if let Some(host) = parsed.host_str() {
        let ip = resolve_host(host)?;
        if is_internal_ip(&ip) {
            return Err(ValidationError::new("internal_ip_blocked"));
        }
    }

    // Block known problematic domains
    if is_blocked_domain(parsed.host_str().unwrap_or("")) {
        return Err(ValidationError::new("blocked_domain"));
    }

    Ok(parsed)
}

fn is_internal_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(ipv4) => {
            ipv4.is_private()
                || ipv4.is_loopback()
                || ipv4.is_link_local()
                || ipv4.octets()[0] == 10  // 10.0.0.0/8
                || (ipv4.octets()[0] == 172 && (16..=31).contains(&ipv4.octets()[1]))
                || (ipv4.octets()[0] == 192 && ipv4.octets()[1] == 168)
        }
        IpAddr::V6(ipv6) => ipv6.is_loopback(),
    }
}
```

**File Upload Validation:**
```rust
async fn validate_upload(upload: &Upload) -> Result<(), ValidationError> {
    // Size limit
    if upload.size > MAX_FILE_SIZE {
        return Err(ValidationError::new("file_too_large"));
    }

    // MIME type validation (from magic bytes, not extension)
    let mime = tree_magic_mini::from_u8(&upload.bytes);
    let allowed = ["application/pdf", "text/plain", "application/epub+zip"];
    if !allowed.contains(&mime) {
        return Err(ValidationError::new("invalid_file_type"));
    }

    // For PDFs, validate structure
    if mime == "application/pdf" {
        validate_pdf_structure(&upload.bytes)?;
    }

    // Scan for malware (if enabled)
    if config.malware_scan_enabled {
        malware_scan(&upload.bytes).await?;
    }

    Ok(())
}
```

---

## 6. Data Protection

### 6.1 Encryption Standards

**Data in Transit:**
- TLS 1.3 required (1.2 minimum with secure ciphers)
- HSTS with 1-year max-age, includeSubDomains, preload
- Certificate transparency required
- Mutual TLS for service-to-service (internal)

**Cipher Suites:**
```
TLS_AES_256_GCM_SHA384
TLS_CHACHA20_POLY1305_SHA256
TLS_AES_128_GCM_SHA256
```

**Data at Rest:**
| Data Type | Encryption | Key Management |
|-----------|------------|----------------|
| Database | AES-256-GCM (TDE) | Managed KMS |
| Object Storage | AES-256-GCM (SSE) | Per-object keys |
| Backups | AES-256-GCM | Separate backup keys |
| Logs | AES-256-GCM | Rotate monthly |

### 6.2 Key Management

**Key Hierarchy:**
```
Root Key (HSM-protected)
    │
    ├── Master Key (Database encryption)
    │       └── Data Encryption Keys (per-table)
    │
    ├── Master Key (Object storage)
    │       └── Object Keys (per-object)
    │
    └── Application Keys
            ├── JWT Signing Keys (rotate quarterly)
            ├── API Key Encryption Keys
            └── Webhook Signing Keys
```

**Key Rotation:**
| Key Type | Rotation Period | Transition |
|----------|-----------------|------------|
| JWT Signing | 90 days | Dual-active for 30 days |
| Database DEK | 365 days | Re-encrypt in background |
| API Keys | User-controlled | Instant invalidation |
| TLS Certificates | 90 days | Automated via ACME |

### 6.3 Secrets Management

**Environment Variables:**
```bash
# NEVER hardcode secrets
# BAD:
DATABASE_URL=postgres://user:password123@host/db

# GOOD: Use secret references
DATABASE_URL=${DATABASE_URL}  # Injected from Vault/Secret Manager
```

**Secret Storage:**
- HashiCorp Vault or cloud-native secret manager
- Secrets never logged
- Secrets rotated on compromise
- Access audited

**Application Secret Handling:**
```rust
// Secrets use zero-on-drop wrappers
use secrecy::{ExposeSecret, SecretString};

struct DatabaseConfig {
    host: String,
    port: u16,
    username: String,
    password: SecretString,  // Auto-zeroed on drop
}

// Never log secrets
impl Debug for DatabaseConfig {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("DatabaseConfig")
            .field("host", &self.host)
            .field("password", &"[REDACTED]")
            .finish()
    }
}
```

### 6.4 PII Handling

**Data Classification:**
| Field | Classification | Retention | Access |
|-------|---------------|-----------|--------|
| Email | PII | Account lifetime | User, Admin |
| Name | PII | Account lifetime | User, Admin |
| IP Address | PII | 90 days | Security team |
| Search queries | Sensitive | 30 days | Analytics team |
| Research content | User data | User-controlled | User, Shared |

**Data Minimization:**
- Collect only necessary data
- Anonymize analytics where possible
- Provide data export and deletion
- Document data flows

---

## 7. Secure Development Practices

### 7.1 Code Review Requirements

**Security-Sensitive Changes:**
- Authentication/authorization changes: 2 reviewers
- Cryptographic code: Security team review
- External integrations: Security checklist
- Database schema changes: DBA + Security review

**Review Checklist:**
```markdown
## Security Review Checklist

### Authentication/Authorization
- [ ] No hardcoded credentials
- [ ] Authorization checks on all endpoints
- [ ] Sensitive actions require re-authentication
- [ ] Session handling is correct

### Input Validation
- [ ] All inputs validated at trust boundary
- [ ] SQL queries use parameterization
- [ ] File uploads validated (type, size, content)
- [ ] URLs validated (scheme, host, SSRF protection)

### Output Handling
- [ ] User content properly escaped
- [ ] Error messages don't leak sensitive info
- [ ] Logs don't contain secrets or PII
- [ ] API responses match expected schema

### Cryptography
- [ ] Using approved algorithms
- [ ] Keys properly managed
- [ ] Random numbers from secure source
- [ ] No custom crypto implementations

### Data Protection
- [ ] Sensitive data encrypted at rest
- [ ] PII minimized
- [ ] Data retention policies followed
- [ ] Backup encryption verified
```

### 7.2 Dependency Management

**Supply Chain Security:**
```toml
# Cargo.toml - Use exact versions for security-critical deps
[dependencies]
ring = "=0.17.7"  # Cryptography - exact version
rustls = "=0.22.2"  # TLS - exact version

# Use cargo-audit for vulnerability scanning
# cargo audit

# Use cargo-deny for license and advisory checking
# cargo deny check
```

**Automated Scanning:**
```yaml
# GitHub Actions workflow
security-scan:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4

    - name: Dependency audit
      run: cargo audit

    - name: License check
      run: cargo deny check licenses

    - name: SAST scan
      uses: github/codeql-action/analyze@v3

    - name: Container scan
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'latticeforge/api:${{ github.sha }}'
```

### 7.3 Secure Defaults

**Configuration:**
```rust
impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            // Restrictive defaults
            cors_allowed_origins: vec![],  // No CORS by default
            max_request_size: 10 * 1024 * 1024,  // 10MB
            rate_limit_requests_per_minute: 60,
            session_timeout_minutes: 15,
            require_mfa: false,  // Encouraged but not required

            // Security features enabled
            csrf_protection: true,
            hsts_enabled: true,
            content_security_policy: true,
            audit_logging: true,
        }
    }
}
```

### 7.4 Error Handling

**Secure Error Responses:**
```rust
// Internal error type with full context
#[derive(Debug)]
enum InternalError {
    DatabaseError { query: String, error: sqlx::Error },
    AuthenticationFailed { user_id: Option<String>, reason: String },
    // ...
}

// External error type - safe to expose
#[derive(Serialize)]
struct ApiError {
    code: String,
    message: String,
    // NO: stack traces, query details, internal state
}

impl From<InternalError> for ApiError {
    fn from(err: InternalError) -> Self {
        // Log full internal error
        error!("Internal error: {:?}", err);

        // Return sanitized external error
        match err {
            InternalError::DatabaseError { .. } => ApiError {
                code: "internal_error".into(),
                message: "An internal error occurred".into(),
            },
            InternalError::AuthenticationFailed { .. } => ApiError {
                code: "unauthorized".into(),
                message: "Authentication failed".into(),
            },
            // ...
        }
    }
}
```

---

## 8. Infrastructure Security

### 8.1 Network Security

**Network Segmentation:**
```
┌─────────────────────────────────────────────────────────────────┐
│ VPC: 10.0.0.0/16                                                │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Public Subnet: 10.0.1.0/24                              │   │
│  │   - Load Balancers                                       │   │
│  │   - NAT Gateways                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Private Subnet: 10.0.10.0/24                            │   │
│  │   - Application Servers                                  │   │
│  │   - Worker Nodes                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Data Subnet: 10.0.20.0/24                               │   │
│  │   - Database (no internet access)                        │   │
│  │   - Redis (no internet access)                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Security Groups:**
| Group | Inbound | Outbound |
|-------|---------|----------|
| load-balancer | 443/tcp from 0.0.0.0/0 | 8080/tcp to app |
| app-server | 8080/tcp from LB | 5432/tcp to DB, 6379/tcp to Redis |
| database | 5432/tcp from app | None |
| redis | 6379/tcp from app | None |

### 8.2 Container Security

**Image Security:**
```dockerfile
# Use minimal base image
FROM gcr.io/distroless/cc-debian12:latest

# Don't run as root
USER nonroot:nonroot

# No shell, no package manager
# COPY only necessary artifacts
COPY --from=builder /app/target/release/api /app/api

# Read-only filesystem where possible
# (configured in orchestration layer)

ENTRYPOINT ["/app/api"]
```

**Runtime Security:**
```yaml
# Kubernetes security context
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 65534
    fsGroup: 65534
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: api
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop:
            - ALL
```

### 8.3 Database Security

**PostgreSQL Hardening:**
```sql
-- Enforce SSL connections
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET ssl_min_protocol_version = 'TLSv1.3';

-- Disable dangerous functions
REVOKE EXECUTE ON FUNCTION pg_read_file(text) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION pg_read_binary_file(text) FROM PUBLIC;

-- Limit connection sources
-- (in pg_hba.conf)
hostssl all app_user 10.0.10.0/24 scram-sha-256
host all all 0.0.0.0/0 reject
```

**RLS Enforcement:**
```sql
-- Enable RLS on all user-data tables
ALTER TABLE streams ENABLE ROW LEVEL SECURITY;
ALTER TABLE sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE insights ENABLE ROW LEVEL SECURITY;

-- Force RLS even for table owners
ALTER TABLE streams FORCE ROW LEVEL SECURITY;
```

---

## 9. Logging and Monitoring

### 9.1 Security Logging

**Required Log Events:**
| Event | Log Level | Retention |
|-------|-----------|-----------|
| Authentication success | INFO | 90 days |
| Authentication failure | WARN | 1 year |
| Authorization denial | WARN | 1 year |
| Password change | INFO | 1 year |
| MFA enrollment/removal | INFO | 1 year |
| API key creation/revocation | INFO | 1 year |
| Admin actions | INFO | 2 years |
| Security exceptions | ERROR | 2 years |

**Structured Log Format:**
```json
{
  "timestamp": "2024-03-15T10:30:00.123Z",
  "level": "WARN",
  "event_type": "authentication.failed",
  "request_id": "req_abc123",
  "user_id": null,
  "attempted_email": "user@example.com",
  "ip_address": "203.0.113.42",
  "user_agent": "Mozilla/5.0...",
  "failure_reason": "invalid_password",
  "attempt_count": 3,
  "geo": {
    "country": "US",
    "region": "CA"
  }
}
```

### 9.2 Security Monitoring

**Detection Rules:**
```yaml
# Brute force detection
- name: brute_force_login
  condition: |
    count(auth.failed where ip = $ip) > 10
    within 5 minutes
  action: block_ip, alert

# Credential stuffing detection
- name: credential_stuffing
  condition: |
    count(distinct auth.failed.email where ip = $ip) > 20
    within 10 minutes
  action: block_ip, alert, investigate

# Privilege escalation attempt
- name: privilege_escalation
  condition: |
    authz.denied where
      required_permission in ['admin', 'owner']
      and user.role in ['member', 'viewer']
  action: alert, lock_account_pending_review

# Anomalous data access
- name: data_exfiltration
  condition: |
    count(api.requests where
      user = $user
      and endpoint matches '/v1/sources/*/export'
    ) > 50 within 1 hour
  action: alert, rate_limit
```

### 9.3 Audit Trail

**Audit Log Schema:**
```sql
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    actor_id UUID,  -- User or service account
    actor_type TEXT NOT NULL,  -- 'user', 'api_key', 'system'
    action TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT,
    resource_name TEXT,
    old_values JSONB,  -- For updates/deletes
    new_values JSONB,  -- For creates/updates
    ip_address INET,
    user_agent TEXT,
    request_id TEXT,
    success BOOLEAN NOT NULL,
    failure_reason TEXT,
    metadata JSONB
);

-- Append-only (no updates or deletes)
CREATE POLICY audit_append_only ON audit_log
    FOR INSERT
    WITH CHECK (true);

REVOKE UPDATE, DELETE ON audit_log FROM app_user;
```

---

## 10. Incident Response

### 10.1 Incident Classification

| Severity | Description | Response Time | Examples |
|----------|-------------|---------------|----------|
| P1 - Critical | Active breach, data exposure | 15 minutes | Data exfiltration, auth bypass |
| P2 - High | Significant vulnerability | 4 hours | RCE, SQL injection discovered |
| P3 - Medium | Potential vulnerability | 24 hours | XSS, information disclosure |
| P4 - Low | Minor security issue | 7 days | Missing security header |

### 10.2 Response Procedures

**P1 Incident Playbook:**
```markdown
## Critical Security Incident

### Immediate Actions (0-15 minutes)
1. [ ] Assemble incident response team
2. [ ] Assess scope of breach
3. [ ] Contain: Revoke compromised credentials
4. [ ] Preserve: Snapshot affected systems
5. [ ] Communicate: Notify executive team

### Short-term Actions (15 min - 4 hours)
6. [ ] Identify attack vector
7. [ ] Block attacker access
8. [ ] Begin forensic analysis
9. [ ] Draft user notification (if required)
10. [ ] Engage legal/compliance if PII involved

### Recovery (4-24 hours)
11. [ ] Patch vulnerability
12. [ ] Rotate potentially compromised secrets
13. [ ] Restore from clean backup if needed
14. [ ] Validate system integrity
15. [ ] Re-enable services

### Post-Incident (24-72 hours)
16. [ ] Complete forensic report
17. [ ] Conduct post-mortem
18. [ ] Update runbooks
19. [ ] Notify affected users
20. [ ] File regulatory reports if required
```

### 10.3 Security Contacts

**Internal:**
- Security Team: security@latticeforge.io
- On-call Security: PagerDuty rotation
- CISO: [Name], [phone]

**External:**
- Legal Counsel: [Firm], [contact]
- Forensic Partner: [Company], [contact]
- Insurance: [Provider], [policy number]
- Bug Bounty: https://hackerone.com/latticeforge

### 10.4 Communication Templates

**User Notification (Data Breach):**
```
Subject: Important Security Notice from LatticeForge

Dear [User],

We are writing to inform you of a security incident that may have affected your account.

What Happened:
[Brief, factual description]

What Information Was Involved:
[List specific data types]

What We Are Doing:
[Remediation steps]

What You Can Do:
[Recommended actions]

For More Information:
[Contact details, FAQ link]

We sincerely apologize for this incident and any inconvenience it may cause.

[Signature]
```

---

## 11. Compliance Requirements

### 11.1 Regulatory Framework

**GDPR Compliance:**
- Lawful basis documented for all processing
- Privacy policy clear and accessible
- Data subject rights implemented (access, delete, export)
- DPA agreements with processors
- 72-hour breach notification capability
- Data Protection Impact Assessment for AI features

**SOC 2 Type II:**
- Security: Logical and physical access controls
- Availability: System monitoring, disaster recovery
- Confidentiality: Data encryption, access restrictions
- Processing Integrity: Quality assurance, error handling
- Privacy: PII handling procedures

### 11.2 Security Controls Mapping

| Control | SOC 2 | GDPR | Implementation |
|---------|-------|------|----------------|
| Access Control | CC6.1 | Art. 32 | RBAC, MFA |
| Encryption | CC6.7 | Art. 32 | TLS, AES-256 |
| Audit Logging | CC7.2 | Art. 30 | Structured logs |
| Incident Response | CC7.3 | Art. 33 | Playbooks |
| Vendor Management | CC9.2 | Art. 28 | DPA agreements |
| Change Management | CC8.1 | - | PR reviews |
| Vulnerability Management | CC7.1 | Art. 32 | Scanning, patching |

### 11.3 Evidence Collection

**Automated Evidence:**
- Access control configurations (exported weekly)
- Encryption status reports (daily)
- Vulnerability scan results (weekly)
- Penetration test reports (annual)
- Audit log samples (on demand)

**Manual Evidence:**
- Security training records
- Policy acknowledgments
- Incident response documentation
- Change management records

---

## 12. Security Testing

### 12.1 Testing Schedule

| Test Type | Frequency | Scope | Performed By |
|-----------|-----------|-------|--------------|
| Automated SAST | Every commit | All code | CI/CD |
| Dependency scan | Daily | All dependencies | CI/CD |
| Container scan | Every build | All images | CI/CD |
| DAST | Weekly | Production | Security team |
| Penetration test | Quarterly | Full application | Third party |
| Red team exercise | Annually | Full stack | External firm |

### 12.2 Penetration Testing Scope

**In Scope:**
- Authentication and session management
- Authorization and access control
- Input validation and injection attacks
- API security
- Business logic flaws
- Cryptographic implementation
- Configuration security

**Out of Scope:**
- Social engineering (separate engagement)
- Physical security
- Denial of service (requires coordination)
- Third-party services (covered by their programs)

### 12.3 Bug Bounty Program

**Scope:**
- *.latticeforge.io
- LatticeForge mobile apps
- Open source components

**Rewards:**
| Severity | Range |
|----------|-------|
| Critical (RCE, auth bypass) | $5,000 - $15,000 |
| High (SQLi, XSS stored, IDOR) | $1,000 - $5,000 |
| Medium (CSRF, info disclosure) | $250 - $1,000 |
| Low (best practices) | $50 - $250 |

**Rules:**
- No automated scanning without coordination
- No denial of service
- No accessing other users' data
- Responsible disclosure required

---

## 13. Security Checklist

### 13.1 Pre-Launch Security Review

```markdown
## Security Readiness Checklist

### Authentication
- [ ] Password hashing uses Argon2id
- [ ] MFA available and encouraged
- [ ] Session management secure
- [ ] OAuth implementation reviewed
- [ ] Account recovery secure

### Authorization
- [ ] RBAC implemented correctly
- [ ] RLS enabled on all user tables
- [ ] IDOR testing completed
- [ ] Admin functions protected

### Data Protection
- [ ] TLS 1.3 configured
- [ ] Data encrypted at rest
- [ ] Secrets in secure storage
- [ ] PII handling documented
- [ ] Backup encryption verified

### Infrastructure
- [ ] Network segmentation complete
- [ ] Security groups least privilege
- [ ] Container hardening done
- [ ] Database hardening done
- [ ] Logging configured

### Compliance
- [ ] Privacy policy published
- [ ] Terms of service published
- [ ] DPA agreements signed
- [ ] Data processing documented

### Testing
- [ ] Penetration test completed
- [ ] Vulnerabilities remediated
- [ ] Security scanning automated
```

### 13.2 Ongoing Security Tasks

**Daily:**
- Review security alerts
- Check failed authentication reports
- Monitor rate limiting triggers

**Weekly:**
- Review access logs for anomalies
- Check dependency vulnerability reports
- Update threat intelligence feeds

**Monthly:**
- Rotate short-lived credentials
- Review access permissions
- Update security runbooks
- Security team sync

**Quarterly:**
- Penetration testing
- Key rotation review
- Incident response drill
- Compliance evidence collection

**Annually:**
- Comprehensive security audit
- Red team engagement
- Policy review and update
- Security training refresh

---

*Security is not a feature—it's a foundation. Every team member shares responsibility for maintaining the security posture described in this document. When in doubt, ask the security team.*
