# Lunatiq Subscription Tiers - Detailed Specification

## Tier Overview

| Feature | FREE | BASIC ($4.99/mo) | PREMIUM ($9.99/mo) |
|---------|------|------------------|---------------------|
| **Price** | $0 | $4.99/mo or $39.99/yr | $9.99/mo or $79.99/yr |
| **Savings** | - | 33% annual | 33% annual |

---

## Reading Limits

| Limit | FREE | BASIC | PREMIUM |
|-------|------|-------|---------|
| Readings/day | 2 | 15 | Unlimited |
| Max cards/spread | 3 | 10 (all spreads) | 10 (all spreads) |
| Available spreads | single, 3-card | All 9 | All 9 |
| Reading types | 2 (general, daily) | All 8 | All 8 |

---

## Oracle Chat Limits

| Limit | FREE | BASIC | PREMIUM |
|-------|------|-------|---------|
| Oracle chat access | ❌ NO | ✅ Yes | ✅ Yes |
| Chats/day | 0 | 5 | 50 |
| Messages/chat | 0 | 20 | 100 |
| Tokens/message | 0 | 500 | 2,000 |
| Chat history | ❌ | 7 days | Forever |
| Personality selection | ❌ | 3 personalities | All 7+ |

---

## Cloud LLM Token Budget

| Limit | FREE | BASIC | PREMIUM |
|-------|------|-------|---------|
| Daily token budget | 1,000 | 25,000 | 150,000 |
| Per-reading tokens | 500 | 2,000 | 5,000 |
| Per-chat-message tokens | 0 | 500 | 2,000 |
| Synthesis tokens | 0 | 1,500 | 4,000 |
| Rollover unused | ❌ | ❌ | ✅ (up to 50k) |

---

## Features

| Feature | FREE | BASIC | PREMIUM |
|---------|------|-------|---------|
| **Core** |
| Card interpretations | Basic (template) | AI-powered | AI-powered + deep |
| Synthesis | ❌ | Basic | Full + meta-analysis |
| Quantum RNG | ✅ | ✅ | ✅ |
| **AI Insights** |
| Pattern detection | ❌ | Basic | Advanced |
| Theme tracking | ❌ | ✅ | ✅ + predictions |
| Archetype analysis | ❌ | ❌ | ✅ |
| **History & Data** |
| Reading history | ❌ | 30 days | Forever |
| Save readings | ❌ | ✅ | ✅ |
| Export readings | ❌ | ✅ | ✅ + bulk |
| Share readings | ❌ | ✅ | ✅ |
| Stats & analytics | ❌ | Basic | Full dashboard |
| **Achievements** |
| Earn achievements | ❌ | ✅ | ✅ |
| View achievements | ❌ | ✅ | ✅ |
| Achievement badges | ❌ | ❌ | ✅ |
| Leaderboards | ❌ | ❌ | ✅ |
| **Personalization** |
| MBTI integration | ❌ | ✅ | ✅ |
| Astrology integration | Basic (sun only) | Full | Full + transits |
| Custom spreads | ❌ | ❌ | ✅ |
| **3D Navigator** |
| Access | ❌ | Basic (view only) | Full (interactive) |
| Personal space | ❌ | Limited | Unlimited |
| Object customization | ❌ | ❌ | ✅ |
| **Support** |
| Priority support | ❌ | ❌ | ✅ |
| Early features | ❌ | ❌ | ✅ |

---

## Upgrade Triggers

### Free → Basic
- Hit daily reading limit
- Try to access Oracle chat
- Try to view achievements
- Try to save/export reading
- Try to see stats
- Try to access 3D navigator

### Basic → Premium
- Hit chat limit
- Hit token budget
- Try to access deep insights
- Try to access full 3D
- Try to customize oracle personality (beyond 3)
- Try to access archetype analysis
- Try to create custom spread

---

## Token Economics

### Cost Assumptions (Claude API)
- Input: $3/1M tokens
- Output: $15/1M tokens
- Average reading: 500 input + 1000 output = $0.0165
- Average chat message: 200 input + 500 output = $0.0081

### Per-User Cost Estimates

| Tier | Daily Usage | Monthly Cost | Revenue | Margin |
|------|-------------|--------------|---------|--------|
| FREE | 2 readings | $1.00 | $0 | -$1.00 |
| BASIC | 10 readings + 5 chats | $4.50 | $4.99 | $0.49 |
| PREMIUM | 20 readings + 20 chats | $9.00 | $9.99 | $0.99 |

**Note:** Margins are thin! Need to:
1. Use cheaper models for basic tasks
2. Cache common interpretations
3. Aggressive rate limiting for free tier
4. Optimize prompts for token efficiency

### Mitigation Strategies
1. **Caching**: Cache card meanings, common spreads
2. **Tiered models**: Use GPT-3.5/Haiku for free, Claude/GPT-4 for premium
3. **Prompt optimization**: Minimize tokens while maintaining quality
4. **Abuse detection**: Block users gaming the system

---

## Implementation Priority

### Phase 1: Core Subscription
1. [ ] Subscription IAP products (monthly/annual for each tier)
2. [ ] Feature gate restructure for 3 tiers
3. [ ] Token tracking system
4. [ ] Basic cloud API proxy

### Phase 2: Limits Enforcement
1. [ ] Reading limit enforcement
2. [ ] Chat limit enforcement
3. [ ] Token budget tracking
4. [ ] Upgrade prompts at limits

### Phase 3: Feature Rollout
1. [ ] AI insights (basic → advanced)
2. [ ] Stats dashboard
3. [ ] Achievement system with viewing
4. [ ] 3D navigator (basic → full)

### Phase 4: Optimization
1. [ ] Caching layer
2. [ ] Model routing (cheap → expensive)
3. [ ] Abuse detection
4. [ ] Analytics for usage patterns

---

## Revenue Projections

### Conservative (1,000 DAU)
- 70% free = 700 users × -$1 = -$700/mo
- 20% basic = 200 users × $4.99 = $998/mo
- 10% premium = 100 users × $9.99 = $999/mo
- **Net: $1,297/mo**

### Target (10,000 DAU)
- 60% free = 6,000 users × -$1 = -$6,000/mo
- 25% basic = 2,500 users × $4.99 = $12,475/mo
- 15% premium = 1,500 users × $9.99 = $14,985/mo
- **Net: $21,460/mo**

### Scale (100,000 DAU)
- 55% free = 55,000 users × -$1 = -$55,000/mo
- 30% basic = 30,000 users × $4.99 = $149,700/mo
- 15% premium = 15,000 users × $9.99 = $149,850/mo
- **Net: $244,550/mo = $2.9M/year**

---

## Path to $1M

**Annual revenue target: $1,000,000**
**Monthly revenue target: $83,333**

**Required subscribers:**
- If 100% premium: 8,342 subscribers
- If 100% basic: 16,700 subscribers
- If 70% basic / 30% premium: ~12,000 subscribers

**Required DAU (at 40% conversion):**
- ~30,000 DAU

**Marketing budget needed:**
- CAC assumption: $5-10 per install
- To get 30,000 DAU: ~$150,000-300,000 marketing spend
- OR organic growth via virality + ASO

---

---

## Anti-Abuse & Rate Limiting

### Hard Limits (Cannot Exceed)

| Limit | FREE | BASIC | PREMIUM |
|-------|------|-------|---------|
| Readings/day | 2 | 15 | 100 |
| Chats/day | 0 | 5 | 50 |
| Tokens/day | 1,000 | 25,000 | 150,000 |
| Tokens/minute | 200 | 1,000 | 5,000 |
| Requests/minute | 5 | 20 | 60 |

### Abuse Detection Triggers

1. **Velocity abuse**: >3x normal request rate → temporary slowdown
2. **Token stuffing**: Max input tokens exceeded → reject request
3. **Prompt injection**: Detected jailbreak attempts → flag account
4. **Multi-accounting**: Same device ID with multiple accounts → flag
5. **Refund abuse**: >2 refunds → ban from subscription

### Enforcement Actions

| Level | Trigger | Action |
|-------|---------|--------|
| Warning | First offense | In-app warning |
| Throttle | Repeated velocity abuse | 10x slower responses for 1 hour |
| Timeout | Continued abuse | 24-hour lockout |
| Suspend | Egregious abuse | Account suspended, manual review |
| Ban | Confirmed malicious | Permanent ban, device blacklist |

### Server-Side Validation

```javascript
// Cloudflare Worker middleware
async function validateRequest(request, user) {
  // 1. Check rate limits
  const rateLimit = await getRateLimit(user.id, user.tier);
  if (rateLimit.exceeded) {
    return { error: 'RATE_LIMIT_EXCEEDED', retryAfter: rateLimit.retryAfter };
  }

  // 2. Check token budget
  const budget = await getTokenBudget(user.id, user.tier);
  if (budget.remaining < request.estimatedTokens) {
    return { error: 'TOKEN_BUDGET_EXCEEDED', remaining: budget.remaining };
  }

  // 3. Check for abuse patterns
  const abuseScore = await checkAbusePatterns(user.id, request);
  if (abuseScore > ABUSE_THRESHOLD) {
    await flagAccount(user.id, abuseScore);
    return { error: 'SUSPICIOUS_ACTIVITY' };
  }

  // 4. Validate input
  if (request.input.length > MAX_INPUT_TOKENS[user.tier]) {
    return { error: 'INPUT_TOO_LONG' };
  }

  return { valid: true };
}
```

### Prompt Security

1. **System prompt hardening**: Never expose system prompt, reject requests asking for it
2. **Output filtering**: Strip any leaked system content
3. **Jailbreak detection**: Pattern matching on known jailbreak attempts
4. **Context isolation**: Each request is isolated, no cross-user contamination

### Monitoring & Alerts

- Real-time dashboard for token usage per tier
- Alerts when any user hits 80% of daily limit
- Alerts when abuse detection triggers
- Weekly report on cost per user segment
- Anomaly detection for unusual patterns

### Graceful Degradation

When user approaches limits:
1. **80% of limit**: Show warning "You're approaching your daily limit"
2. **100% of limit**: Block with upgrade prompt
3. **If rate limited**: Show "Please wait X seconds" with countdown

---

## Notes

- Free tier MUST be aggressive to drive conversions
- Basic tier is the volume play
- Premium tier is the margin play
- Token costs are the main variable expense
- Need to watch margins closely and adjust limits if needed
- NEVER let individual users bankrupt us
- Always have kill switch for runaway costs
