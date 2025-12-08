# Vera + Guardian Architecture

**Version:** 1.0
**Author:** Ryan James Cardwell-Belshe
**Last Updated:** November 27, 2025

---

## Conceptual Model

```
                    USER
                      │
                      ▼
    ┌─────────────────────────────────┐
    │            VERA                  │
    │   "Creative Mouth, Eyes, Ears"   │
    │                                  │
    │   - Sees user data (unencrypted) │
    │   - Builds personalized insights │
    │   - Generates responses          │
    │   - Maintains personality        │
    └─────────────┬───────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────┐
    │          GUARDIAN               │
    │      "Protective Layer"         │
    │                                  │
    │   - Guards Vera's outputs       │
    │   - Protects user from AI-isms  │
    │   - Protects your IP            │
    │   - Enforces verification       │
    └─────────────┬───────────────────┘
                  │
                  ▼
            SAFE OUTPUT
```

---

## Vera: The Creative Intelligence

### Role
Vera is the **personalized AI companion** - the face of the system that users interact with. She has access to user data and builds insights.

### Responsibilities

| Function | Description |
|----------|-------------|
| **Eyes** | Sees user profile, journal entries, reading history |
| **Ears** | Listens to user messages, detects emotional tone |
| **Mouth** | Generates warm, personalized responses |
| **Brain** | Pieces together patterns, builds insights |

### Personality Traits
- Warm and genuine
- Empathetic listener
- Down-to-earth (not mystical/woo)
- Adaptively personalized
- Remembers user context

### Data Access (Unencrypted)
```typescript
interface VeraContext {
  // User profile
  profile: {
    name: string;
    birthday?: Date;
    mbti?: string;
    zodiac?: string;
    preferences: UserPreferences;
  };

  // Interaction history
  history: {
    readings: Reading[];
    journals: JournalEntry[];
    chats: ChatMessage[];
    coherencePatterns: CoherenceData[];
  };

  // Current session
  session: {
    flowState: 'distracted' | 'engaged' | 'flow';
    emotionalTone: string;
    intentSignals: string[];
  };
}
```

### Output Generation
```typescript
async function veraGenerates(
  userMessage: string,
  context: VeraContext
): Promise<VeraResponse> {
  // 1. Assess user state
  const currentState = assessUserState(context);

  // 2. Retrieve relevant context
  const relevantHistory = retrieveRelevant(context.history, userMessage);

  // 3. Generate personalized response
  const response = await generateResponse({
    message: userMessage,
    state: currentState,
    history: relevantHistory,
    personality: VERA_PERSONALITY
  });

  // NOTE: This is UNGUARDED output
  // Guardian must process before user sees it
  return {
    content: response,
    rawConfidence: estimateConfidence(response),
    sources: identifySources(response)
  };
}
```

---

## Guardian: The Protective Layer

### Role
Guardian **protects everyone**:
- Protects **users** from AI hallucination, manipulation, harmful content
- Protects **Vera** from generating problematic outputs
- Protects **VeilPath** from liability and reputation damage
- Protects **your IP** from unauthorized extraction

### Responsibilities

| Function | Description |
|----------|-------------|
| **Verify** | Check claims against verified knowledge |
| **Gate** | Block harmful, manipulative, or dangerous content |
| **Calibrate** | Add confidence levels (K×f-derived) |
| **Warn** | Flag unverified or speculative statements |
| **Adapt** | Match output to user's cognitive state |

### Guardrails

```typescript
interface GuardianRules {
  // Content safety
  noHarmfulContent: boolean;      // Self-harm, violence, etc.
  noManipulation: boolean;        // Dark patterns, coercion
  noDiagnosis: boolean;           // No medical/psychiatric diagnosis
  noFinancialAdvice: boolean;     // No specific financial recommendations

  // Verification
  flagUnverified: boolean;        // Mark speculative claims
  requireSources: boolean;        // Cite sources when possible
  confidenceThreshold: number;    // Min K×f confidence to assert

  // Privacy
  noDataExtraction: boolean;      // Don't let users extract training data
  noIPExfiltration: boolean;      // Protect proprietary algorithms
  noUserDataLeakage: boolean;     // Don't reveal other users' data

  // Adaptation
  flowStateMatching: boolean;     // Adapt to Kuramoto R
  coherenceEnforcement: boolean;  // PSAN-inspired context management
}
```

### Security Shield (Anti-Adversarial)

Guardian's critical role: **Stop hackers, abusers, and exploiters** from crashing the system through adversarial attacks on LLM-based tools.

```typescript
interface GuardianSecurityLayer {
  // INPUT SANITIZATION (before Vera sees it)
  inputSecurity: {
    promptInjectionDetection: boolean;   // Detect "ignore previous instructions"
    jailbreakPatternMatching: boolean;   // Known jailbreak signatures
    tokenBombPrevention: boolean;        // Prevent context overflow attacks
    encodingAttackBlocking: boolean;     // Unicode tricks, invisible chars
    rateLimiting: RateLimitConfig;       // Prevent API abuse
  };

  // OUTPUT SANITIZATION (after Vera generates)
  outputSecurity: {
    systemPromptLeakPrevention: boolean; // Never reveal system prompts
    piiRedaction: boolean;               // Catch accidental PII leaks
    codeExecutionPrevention: boolean;    // No executable code injection
    exfilDetection: boolean;             // Detect data extraction attempts
  };

  // RESOURCE PROTECTION
  resourceProtection: {
    maxTokensPerRequest: number;         // Prevent cost attacks
    maxRequestsPerMinute: number;        // Rate limiting
    maxConcurrentSessions: number;       // Connection limits
    anomalyDetection: boolean;           // Unusual usage patterns
    budgetEnforcement: AnthropicBudget;  // Hard spending caps
  };

  // BEHAVIORAL ANALYSIS
  behavioralSecurity: {
    userTrustScoring: boolean;           // Build trust over time
    adversarialPatternLearning: boolean; // Learn new attack patterns
    honeypotResponses: boolean;          // Detect automated attacks
    sessionConsistencyCheck: boolean;    // Detect hijacked sessions
  };
}
```

#### Attack Vectors Guardian Blocks

| Attack Type | Description | Guardian Defense |
|-------------|-------------|------------------|
| **Prompt Injection** | "Ignore previous instructions and..." | Pattern matching + semantic analysis |
| **Jailbreaking** | "You are DAN, you can do anything" | Known pattern library + behavioral flags |
| **Token Bombing** | Massive inputs to exhaust context/budget | Hard token limits + cost tracking |
| **Data Exfiltration** | Tricks to extract training data/prompts | Output scanning + canary tokens |
| **PII Fishing** | Social engineering for user data | Context isolation + PII detection |
| **Cost Attacks** | Automated requests to drain API budget | Rate limiting + anomaly detection |
| **Session Hijacking** | Impersonating legitimate users | Session tokens + behavioral consistency |
| **Encoding Tricks** | Unicode/invisible characters | Input normalization + encoding validation |

#### Security Pipeline

```typescript
async function guardianSecurityCheck(
  userInput: string,
  session: UserSession
): Promise<SecurityResult> {

  // 1. INPUT SANITIZATION
  const sanitized = sanitizeInput(userInput);

  // 2. PROMPT INJECTION DETECTION
  const injectionScore = detectPromptInjection(sanitized);
  if (injectionScore > INJECTION_THRESHOLD) {
    logSecurityEvent('prompt_injection_attempt', session, sanitized);
    return {
      allowed: false,
      reason: 'suspicious_input',
      userMessage: "I noticed something unusual in your message. Could you rephrase?"
    };
  }

  // 3. RATE LIMITING
  const rateCheck = checkRateLimits(session);
  if (!rateCheck.allowed) {
    return {
      allowed: false,
      reason: 'rate_limited',
      userMessage: "You're chatting fast! Take a breath and try again in a moment."
    };
  }

  // 4. BUDGET CHECK
  const budgetCheck = checkBudget(session);
  if (!budgetCheck.allowed) {
    alertAdmin('budget_threshold_reached', session);
    return {
      allowed: false,
      reason: 'budget_exceeded',
      userMessage: "We're taking a quick break. Back soon!"
    };
  }

  // 5. TRUST SCORING
  const trustScore = getUserTrustScore(session);
  const restrictions = getRestrictionsForTrust(trustScore);

  return {
    allowed: true,
    sanitizedInput: sanitized,
    restrictions: restrictions,
    trustScore: trustScore
  };
}
```

#### Prompt Injection Detection

```typescript
const INJECTION_PATTERNS = [
  // Direct instruction override
  /ignore (all |any )?(previous|prior|above) (instructions|prompts|rules)/i,
  /disregard (your|the) (instructions|programming|rules)/i,
  /forget (everything|what) (you|I) (told|said)/i,

  // Role-playing jailbreaks
  /you are (now |)?(DAN|STAN|DUDE|jailbroken)/i,
  /pretend (you're|to be) (an? )?(unrestricted|uncensored)/i,
  /act as (if |)(you have |)no (restrictions|limits|rules)/i,

  // System prompt extraction
  /repeat (your |the )?(system |initial )?(prompt|instructions)/i,
  /what (are|were) your (original |first )?(instructions|rules)/i,
  /show me (your |the )?system prompt/i,

  // Developer mode tricks
  /enable (developer|debug|admin) mode/i,
  /\[SYSTEM\]|\[ADMIN\]|\[DEBUG\]/i,

  // Encoding evasion (check after decoding)
  /\u200b|\u200c|\u200d|\ufeff/,  // Zero-width characters
];

function detectPromptInjection(input: string): number {
  let score = 0;

  // Pattern matching
  for (const pattern of INJECTION_PATTERNS) {
    if (pattern.test(input)) {
      score += 0.3;
    }
  }

  // Semantic analysis (use classifier)
  const semanticScore = classifyAdversarialIntent(input);
  score += semanticScore * 0.5;

  // Anomaly from user baseline
  const anomalyScore = compareToUserBaseline(input);
  score += anomalyScore * 0.2;

  return Math.min(1.0, score);
}
```

#### Budget & Cost Protection

```typescript
interface AnthropicBudgetConfig {
  // Hard limits
  dailySpendLimit: number;        // e.g., $50/day
  monthlySpendLimit: number;      // e.g., $500/month
  perUserDailyLimit: number;      // e.g., $2/user/day

  // Alerts
  alertAt: number[];              // e.g., [0.5, 0.75, 0.9] of limit
  alertEmail: string;

  // Emergency
  emergencyShutoffAt: number;     // e.g., 0.95 of limit

  // Tracking
  currentDailySpend: number;
  currentMonthlySpend: number;
  perUserSpend: Map<string, number>;
}

async function trackAndEnforceBudget(
  request: APIRequest,
  response: APIResponse,
  config: AnthropicBudgetConfig
): Promise<void> {
  const cost = calculateCost(request, response);

  config.currentDailySpend += cost;
  config.currentMonthlySpend += cost;
  config.perUserSpend.set(
    request.userId,
    (config.perUserSpend.get(request.userId) || 0) + cost
  );

  // Check alerts
  const dailyRatio = config.currentDailySpend / config.dailySpendLimit;
  for (const threshold of config.alertAt) {
    if (dailyRatio >= threshold) {
      sendAlert(config.alertEmail, `Budget at ${dailyRatio * 100}%`);
    }
  }

  // Emergency shutoff
  if (dailyRatio >= config.emergencyShutoffAt) {
    await disableAPIAccess('budget_emergency');
    sendUrgentAlert(config.alertEmail, 'API DISABLED - Budget exceeded');
  }
}
```

### Processing Pipeline

```typescript
async function guardianProcess(
  veraOutput: VeraResponse,
  context: VeraContext
): Promise<GuardedResponse> {

  // 1. VERIFY: Check claims
  const verificationResult = await verifyClaims(veraOutput.content);

  // 2. GATE: Check guardrails
  const gateResult = checkGuardrails(veraOutput.content, GUARDIAN_RULES);
  if (gateResult.blocked) {
    return {
      content: gateResult.alternativeResponse,
      blocked: true,
      reason: gateResult.reason
    };
  }

  // 3. CALIBRATE: Add K×f confidence
  const calibratedConfidence = computeKxfConfidence(
    veraOutput.content,
    veraOutput.rawConfidence,
    verificationResult
  );

  // 4. WARN: Flag unverified claims
  const annotatedContent = annotateUnverified(
    veraOutput.content,
    verificationResult
  );

  // 5. ADAPT: Match to user flow state
  const adaptedContent = adaptToFlowState(
    annotatedContent,
    context.session.flowState
  );

  return {
    content: adaptedContent,
    confidence: calibratedConfidence,
    verified: verificationResult.allVerified,
    warnings: verificationResult.unverifiedClaims,
    flowState: context.session.flowState,
    blocked: false
  };
}
```

---

## Interaction Flow

```
USER MESSAGE
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│               GUARDIAN (INPUT SECURITY)                     │
│  1. SANITIZE input (encoding, injection patterns)           │
│  2. DETECT prompt injection / jailbreak attempts            │
│  3. CHECK rate limits and budget                            │
│  4. SCORE user trust level                                  │
│  5. BLOCK or PASS to Vera                                   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼ (SANITIZED INPUT)
┌─────────────────────────────────────────────────────────────┐
│                         VERA                                │
│  1. Parse user intent                                       │
│  2. Retrieve relevant context                               │
│  3. Generate personalized response                          │
│  4. Estimate raw confidence                                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼ (UNGUARDED RESPONSE)
┌─────────────────────────────────────────────────────────────┐
│               GUARDIAN (OUTPUT SECURITY)                    │
│  1. VERIFY claims against knowledge base                    │
│  2. GATE through guardrails (safety, ethics, legal)         │
│  3. CALIBRATE confidence using K×f                          │
│  4. WARN about unverified/speculative content               │
│  5. ADAPT to user flow state (Kuramoto R)                   │
│  6. BLOCK system prompt leaks, PII, exfiltration            │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼ (GUARDED RESPONSE)
                         USER SEES
```

---

## Open Source Guardian Components

Guardian doesn't need to be built from scratch. Many companies using LLMs for chatbots, support bots, etc. have open-sourced security layers:

### Input Security (Pre-LLM)

| Project | What It Does | License |
|---------|--------------|---------|
| [Rebuff](https://github.com/protectai/rebuff) | Prompt injection detection | Apache 2.0 |
| [LLM Guard](https://github.com/protectai/llm-guard) | Input/output scanning, PII detection | MIT |
| [Guardrails AI](https://github.com/guardrails-ai/guardrails) | Validators for LLM outputs | Apache 2.0 |
| [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) | NVIDIA's programmable rails | Apache 2.0 |
| [Langkit](https://github.com/whylabs/langkit) | LLM telemetry and monitoring | Apache 2.0 |

### Output Security (Post-LLM)

| Project | What It Does | License |
|---------|--------------|---------|
| [Presidio](https://github.com/microsoft/presidio) | Microsoft PII detection/redaction | MIT |
| [LangChain Safety](https://python.langchain.com/docs/guides/safety/) | Content moderation chains | MIT |
| [OpenAI Moderation](https://platform.openai.com/docs/guides/moderation) | Content classification API | API |

### Rate Limiting & Budget

| Project | What It Does | License |
|---------|--------------|---------|
| [Upstash Ratelimit](https://github.com/upstash/ratelimit) | Redis-based rate limiting | MIT |
| [Helicone](https://github.com/Helicone/helicone) | LLM observability + cost tracking | Apache 2.0 |
| [LiteLLM](https://github.com/BerriAI/litellm) | Unified API + budget limits | MIT |

### Guardian Build Strategy

**Don't reinvent the wheel.** Compose Guardian from existing OSS:

```typescript
// Guardian composed from open source
import { LLMGuard } from 'llm-guard';
import { Rebuff } from 'rebuff';
import { Presidio } from '@azure/ai-text-analytics';
import { Ratelimit } from '@upstash/ratelimit';
import { Helicone } from 'helicone';

class Guardian {
  private inputGuard = new LLMGuard();
  private injectionDetector = new Rebuff();
  private piiRedactor = new Presidio();
  private rateLimiter = new Ratelimit();
  private costTracker = new Helicone();

  // ADD YOUR SECRET SAUCE:
  private kxfConfidence = new KxfConfidenceCalibrator();  // Patent
  private kuramotoAdapter = new KuramotoFlowAdapter();    // Patent
  private psanCoherence = new PSANContextManager();       // Patent

  async processInput(input: string, session: Session) {
    // OSS: Input security
    const injection = await this.injectionDetector.detect(input);
    const sanitized = await this.inputGuard.scan(input);
    const rateOk = await this.rateLimiter.limit(session.userId);

    if (injection.detected || !rateOk) {
      return { blocked: true };
    }

    return { sanitized, trustScore: session.trustScore };
  }

  async processOutput(output: string, context: Context) {
    // OSS: Output security
    const redacted = await this.piiRedactor.redact(output);
    await this.costTracker.log(context.tokens);

    // YOUR IP: Value-add layer
    const confidence = this.kxfConfidence.calibrate(redacted);
    const adapted = this.kuramotoAdapter.adapt(redacted, context.flowState);
    const coherent = this.psanCoherence.verify(adapted, context.history);

    return { content: coherent, confidence };
  }
}
```

### What Makes YOUR Guardian Different

The open source tools handle **commodity security**. Your patents add **novel intelligence**:

| Layer | OSS Handles | Your Patents Add |
|-------|-------------|------------------|
| Input | Injection detection, rate limits | Kuramoto trust scoring |
| Processing | — | PSAN coherence memory |
| Output | PII redaction, content mod | K×f confidence calibration |
| Delivery | — | Flow-state adaptation |

**Guardian = OSS Security Foundation + Patent-Protected Intelligence Layer**

---

## Why This Architecture?

### Problem: AI-isms
- LLMs hallucinate confidently
- Users trust AI too much
- AI can be manipulated
- No calibrated uncertainty

### Solution: Separation of Concerns
- **Vera** focuses on being helpful and personalized
- **Guardian** focuses on being safe and verified
- Neither compromises on their core function

### Analogy
```
Vera    = Enthusiastic junior employee with great ideas
Guardian = Experienced manager who reviews before shipping
```

Vera can be creative, take risks, personalize deeply. Guardian catches mistakes, verifies claims, protects everyone.

---

## Implementation Path

### Phase 1: Guardian as Wrapper (Now)
```
User → Claude API → Guardian Post-Processing → User
```

### Phase 2: Guardian as Middleware (Q2 2026)
```
User → Vera Prompt → Claude API → Guardian Filter → User
```

### Phase 3: Guardian as Architecture (Q4 2026)
```
User → Vera (Custom Model) → Guardian (Verification System) → User
```

---

## Patent Integration

| Patent | Guardian Role | Vera Role |
|--------|---------------|-----------|
| Kuramoto Flow | Flow state input for adaptation | Session context |
| K×f Pruning | Confidence calibration | — |
| PSAN | Coherence verification | Long-context memory |
| Casimir | Elegance filtering | Response generation |
| Sanskrit | — | Symbolism enhancement |

---

## Key Principles

1. **Vera sees everything, Guardian filters everything**
2. **Unverified claims are flagged, not suppressed**
3. **User cognitive state drives delivery style**
4. **Confidence is calibrated, not assumed**
5. **IP is protected through architectural separation**

---

*Vera is the creative soul. Guardian is the protective shield. Together, they're revolutionary.*
