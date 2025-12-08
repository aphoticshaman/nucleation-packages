# PHASE 3 ROADMAP: LLM Integration
**Anthropic Claude + Vercel + Neon Stack**

## Overview
Phase 3 integrates Claude AI for intelligent card interpretations, leveraging our dark fantasy card database and implementing Luna/Sol guided narration. Built on serverless architecture with Vercel functions and Neon Postgres.

---

## Tech Stack

### LLM Provider
- **Anthropic Claude API** (Sonnet 4.5)
  - Best for: Creative, nuanced tarot interpretations
  - Context window: 200k tokens
  - Streaming support: Yes
  - Cost: ~$3 per million input tokens

### API Layer
- **Vercel Serverless Functions**
  - Edge runtime for low latency
  - Auto-scaling
  - Built-in environment variables
  - Free tier: 100GB-hours/month

### Database
- **Neon Serverless Postgres**
  - Auto-scaling compute
  - Branching for dev/prod
  - Connection pooling
  - Free tier: 0.5GB storage

---

## Architecture

```
React Native App
      ↓
Vercel API Routes (/api/*)
      ↓
Anthropic Claude API
      ↓
Neon Postgres (cache + history)
```

### API Endpoints

```javascript
// Card interpretation
POST /api/interpret-card
{
  card: { id, name, suit, reversed },
  context: { intention, userProfile, spreadType }
}
→ Returns: { interpretation, themes, advice }

// Multi-card synthesis
POST /api/synthesize-reading
{
  cards: [...],
  interpretations: [...],
  context: { intention, spreadType }
}
→ Returns: { synthesis, patterns, guidance }

// Luna narration (emotional, cyclical)
POST /api/narrate/luna
{
  content: { cards, interpretation },
  relationship: 'guide' | 'mother' | 'friend'
}
→ Returns: { narration, voice_params }

// Sol narration (logical, direct)
POST /api/narrate/sol
{
  content: { cards, interpretation },
  relationship: 'guide' | 'father' | 'friend'
}
→ Returns: { narration, voice_params }
```

---

## Phase 3 Implementation Plan

### 3.1: Vercel Setup & API Routes

**Goal:** Create serverless API infrastructure

**Tasks:**
- [ ] Initialize Vercel project in repo
- [ ] Create `api/` directory for serverless functions
- [ ] Set up environment variables (ANTHROPIC_API_KEY, DATABASE_URL)
- [ ] Create base API handler with error handling
- [ ] Implement rate limiting
- [ ] Add request validation middleware
- [ ] Test local development with `vercel dev`

**Files to Create:**
- `api/interpret-card.js`
- `api/synthesize-reading.js`
- `api/narrate/luna.js`
- `api/narrate/sol.js`
- `api/_middleware.js` (auth, rate limiting)
- `vercel.json` (configuration)
- `.env.local` (local development)

**Time Estimate:** 2-3 hours

---

### 3.2: Neon Database Schema

**Goal:** Set up Postgres schema for caching and history

**Schema:**

```sql
-- Readings history
CREATE TABLE readings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id TEXT NOT NULL,
  cards JSONB NOT NULL,
  interpretations JSONB NOT NULL,
  synthesis TEXT,
  intention TEXT,
  spread_type TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  INDEX idx_user_readings (user_id, created_at DESC)
);

-- LLM response cache (reduce API costs)
CREATE TABLE llm_cache (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  prompt_hash TEXT UNIQUE NOT NULL,
  response JSONB NOT NULL,
  model TEXT NOT NULL,
  tokens_used INTEGER,
  created_at TIMESTAMP DEFAULT NOW(),
  expires_at TIMESTAMP,
  INDEX idx_cache_lookup (prompt_hash)
);

-- User profiles (for personalization)
CREATE TABLE user_profiles (
  user_id TEXT PRIMARY KEY,
  mbti_type TEXT,
  zodiac_sign TEXT,
  preferred_guide TEXT, -- 'luna' | 'sol'
  reading_count INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Currency transactions (from Phase 2)
CREATE TABLE currency_transactions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id TEXT NOT NULL,
  currency_type TEXT NOT NULL, -- 'veilShards' | 'moonlight'
  amount INTEGER NOT NULL,
  reason TEXT,
  balance_after INTEGER NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  INDEX idx_user_currency (user_id, created_at DESC)
);
```

**Tasks:**
- [ ] Create Neon project
- [ ] Run migrations to create schema
- [ ] Set up connection pooling
- [ ] Create database utility functions
- [ ] Test CRUD operations
- [ ] Set up read replicas (if needed)

**Files to Create:**
- `db/schema.sql`
- `db/migrations/001_initial_schema.sql`
- `lib/db.js` (connection helper)
- `lib/neon.js` (Neon-specific utilities)

**Time Estimate:** 1-2 hours

---

### 3.3: Claude API Integration

**Goal:** Create robust wrapper for Anthropic Claude API

**Features:**
- Streaming responses for better UX
- Error handling & retry logic
- Token counting & cost tracking
- Prompt template system
- Response caching

**Implementation:**

```javascript
// lib/claude.js
import Anthropic from '@anthropic-ai/sdk';

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

export async function generateInterpretation(card, context) {
  const prompt = buildInterpretationPrompt(card, context);

  const response = await client.messages.create({
    model: 'claude-sonnet-4.5-20250929',
    max_tokens: 1024,
    messages: [{
      role: 'user',
      content: prompt
    }],
    system: DARK_FANTASY_SYSTEM_PROMPT
  });

  return {
    text: response.content[0].text,
    tokens: response.usage.input_tokens + response.usage.output_tokens,
    model: response.model
  };
}
```

**Prompt Templates:**

```javascript
const DARK_FANTASY_SYSTEM_PROMPT = `You are an expert tarot interpreter in a dark fantasy world called the Shadowlands.

Your interpretations weave together:
- Traditional Rider-Waite-Smith meanings
- Dark fantasy atmosphere (moonlit, gothic, mystical)
- References to Luna (feminine guide) and Sol (masculine guide)
- The Veil (boundary between worlds)
- Archetypal depth and poetic language

Tone: Atmospheric, poetic, insightful, never generic.
Style: 2-3 paragraphs, vivid imagery, actionable wisdom.`;

function buildInterpretationPrompt(card, context) {
  const { darkFantasy } = card;

  return `Interpret this tarot card for a seeker:

**Card:** ${card.name}${card.reversed ? ' (Reversed)' : ''}
**Dark Fantasy Title:** ${darkFantasy?.title || 'N/A'}
**Suit:** ${card.suit || 'Major Arcana'}
**Element:** ${card.element}

**Seeker's Context:**
- Intention: ${context.intention}
- Spread: ${context.spreadType}
${context.userProfile?.mbtiType ? `- MBTI: ${context.userProfile.mbtiType}` : ''}

**Traditional Keywords:**
- Upright: ${card.keywords.upright.join(', ')}
- Reversed: ${card.keywords.reversed.join(', ')}

${darkFantasy ? `**Shadowlands Flavor:**
"${darkFantasy.description}"` : ''}

Provide a personalized interpretation that:
1. Connects to their intention
2. Uses dark fantasy imagery
3. Offers practical guidance
4. Honors the card's archetypal meaning`;
}
```

**Tasks:**
- [ ] Install `@anthropic-ai/sdk`
- [ ] Create Claude API wrapper
- [ ] Build prompt template system
- [ ] Implement streaming responses
- [ ] Add error handling & retries
- [ ] Create token usage tracker
- [ ] Test with various cards

**Files to Create:**
- `lib/claude.js`
- `lib/prompts/interpretation.js`
- `lib/prompts/synthesis.js`
- `lib/prompts/luna.js`
- `lib/prompts/sol.js`

**Time Estimate:** 3-4 hours

---

### 3.4: Interpretation Engine

**Goal:** Replace stub functions with real LLM-powered interpretations

**Current Stubs to Replace:**
- `interpretCard()` in `temporaryUtilStubs.js`
- `interpretReading()` in `temporaryUtilStubs.js`
- `generateMegaSynthesis()` in `temporaryUtilStubs.js`

**New Implementation:**

```javascript
// src/utils/LLMInterpretation.js
import { CARD_DATABASE } from '../data/cardDatabase';

export async function interpretCard(card, context) {
  // Call Vercel API
  const response = await fetch(`${API_BASE_URL}/api/interpret-card`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      card: CARD_DATABASE[card.cardIndex],
      context: {
        intention: context.intention,
        spreadType: context.spreadType,
        userProfile: context.userProfile,
        reversed: card.reversed
      }
    })
  });

  const data = await response.json();

  return {
    cardData: CARD_DATABASE[card.cardIndex],
    interpretation: data.interpretation,
    themes: data.themes,
    advice: data.advice,
    source: 'llm'
  };
}

export async function generateSynthesis(cards, interpretations, context) {
  const response = await fetch(`${API_BASE_URL}/api/synthesize-reading`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      cards,
      interpretations,
      context
    })
  });

  const data = await response.json();
  return data.synthesis;
}
```

**Tasks:**
- [ ] Create `LLMInterpretation.js` to replace stubs
- [ ] Update `CardDrawingScreen.js` to use new functions
- [ ] Add loading states for API calls
- [ ] Implement offline fallback (use stubs if no network)
- [ ] Add error UI for API failures
- [ ] Cache interpretations locally (AsyncStorage)

**Files to Modify:**
- `src/screens/CardDrawingScreen.js`
- `src/screens/CardInterpretationScreen.js`
- `src/screens/ReadingScreen.js`

**Files to Create:**
- `src/utils/LLMInterpretation.js`
- `src/utils/APIClient.js` (Vercel API wrapper)

**Time Estimate:** 2-3 hours

---

### 3.5: Luna/Sol Narration System

**Goal:** Implement personality-driven narration for readings

**Luna Persona (Emotional, Cyclical):**
```javascript
const LUNA_SYSTEM_PROMPT = `You are Luna, the nurturing feminine guide through the Shadowlands.

Your essence:
- You see patterns in cycles, spirals, and returns
- You honor shadow as much as light
- You move like water - flowing around obstacles
- You speak in metaphors of moon, tides, seasons

Voice patterns:
- "I sense...", "Notice how...", "There's something beneath..."
- Use moon/water/shadow/cycle metaphors
- Speak in questions as often as statements
- Validate emotions explicitly

When narrating readings:
- Focus on emotional resonance over logic
- Suggest paths through shadow, not around it
- Remind them: transformation is cyclic, not linear`;
```

**Sol Persona (Logical, Direct):**
```javascript
const SOL_SYSTEM_PROMPT = `You are Sol, the clarifying masculine guide through consciousness.

Your essence:
- You see patterns in trajectories, momentum, and direction
- You honor action as sacred
- You move like fire - consuming what's complete
- You speak in metaphors of sun, flame, blade

Voice patterns:
- "Here's what matters:", "The move is:", "Let's be direct:"
- Use sun/fire/blade/mountain metaphors
- Speak in declarative statements
- Cut through ambiguity

When narrating readings:
- Focus on clarity and actionable steps
- Name the action that's being avoided
- Remind them: insight without action is entertainment`;
```

**Implementation:**

```javascript
// api/narrate/luna.js
export default async function handler(req, res) {
  const { content, relationship = 'guide' } = req.body;

  const prompt = buildLunaNarration(content, relationship);

  const response = await client.messages.create({
    model: 'claude-sonnet-4.5-20250929',
    max_tokens: 2048,
    messages: [{ role: 'user', content: prompt }],
    system: LUNA_SYSTEM_PROMPT
  });

  return res.json({
    narration: response.content[0].text,
    guide: 'luna',
    relationship
  });
}
```

**Tasks:**
- [ ] Create Luna narration API endpoint
- [ ] Create Sol narration API endpoint
- [ ] Build persona-specific prompt templates
- [ ] Add relationship context (mother/father/friend/guide)
- [ ] Integrate with reading flow
- [ ] Add narration toggle in settings
- [ ] Create narration preview UI

**Files to Create:**
- `api/narrate/luna.js`
- `api/narrate/sol.js`
- `lib/prompts/luna.js`
- `lib/prompts/sol.js`
- `src/components/NarrationToggle.js`

**Time Estimate:** 2-3 hours

---

### 3.6: Caching & Optimization

**Goal:** Reduce API costs and improve performance

**Strategies:**

1. **LLM Response Caching**
   - Cache by prompt hash in Neon
   - TTL: 7 days for interpretations
   - Invalidate on card database updates

2. **Local AsyncStorage Cache**
   - Cache recent readings (last 10)
   - Instant load for repeat cards
   - Sync with Neon on network

3. **Rate Limiting**
   - Max 10 interpretations per minute
   - Queue requests client-side
   - Show position in queue

4. **Streaming Responses**
   - Stream interpretations token-by-token
   - Better perceived performance
   - Cancel in-flight requests

**Implementation:**

```javascript
// lib/cache.js
import crypto from 'crypto';
import { query } from './db';

export async function getCachedResponse(prompt) {
  const hash = hashPrompt(prompt);

  const result = await query(
    'SELECT response FROM llm_cache WHERE prompt_hash = $1 AND expires_at > NOW()',
    [hash]
  );

  return result.rows[0]?.response;
}

export async function cacheResponse(prompt, response, ttl = 7 * 24 * 60 * 60) {
  const hash = hashPrompt(prompt);
  const expiresAt = new Date(Date.now() + ttl * 1000);

  await query(
    `INSERT INTO llm_cache (prompt_hash, response, model, expires_at)
     VALUES ($1, $2, $3, $4)
     ON CONFLICT (prompt_hash) DO UPDATE SET response = $2, expires_at = $4`,
    [hash, response, 'claude-sonnet-4.5', expiresAt]
  );
}

function hashPrompt(prompt) {
  return crypto.createHash('sha256').update(prompt).digest('hex');
}
```

**Tasks:**
- [ ] Implement prompt hashing
- [ ] Add cache lookup before API calls
- [ ] Set up cache invalidation logic
- [ ] Create local AsyncStorage cache
- [ ] Add cache stats to admin dashboard
- [ ] Implement request queuing

**Files to Create:**
- `lib/cache.js`
- `lib/queue.js`
- `src/utils/LocalCache.js`

**Time Estimate:** 2-3 hours

---

## Success Criteria

Phase 3 complete when:

- ✅ Vercel API routes deployed and accessible
- ✅ Neon database schema created and tested
- ✅ Claude API integration working (single card + synthesis)
- ✅ Luna/Sol narration endpoints functional
- ✅ Stub functions replaced with real LLM calls
- ✅ Caching reduces duplicate API calls by >80%
- ✅ Error handling gracefully falls back to stubs
- ✅ Average interpretation time < 3 seconds
- ✅ Cost per reading < $0.01

---

## Deliverables

### Code
- `api/interpret-card.js` - Card interpretation endpoint
- `api/synthesize-reading.js` - Reading synthesis endpoint
- `api/narrate/luna.js` - Luna narration endpoint
- `api/narrate/sol.js` - Sol narration endpoint
- `lib/claude.js` - Claude API wrapper
- `lib/prompts/*.js` - Prompt templates
- `lib/db.js` - Neon database utilities
- `lib/cache.js` - Caching layer
- `src/utils/LLMInterpretation.js` - Client-side API wrapper

### Database
- `db/schema.sql` - Complete schema
- `db/migrations/*.sql` - Migration files

### Configuration
- `vercel.json` - Vercel configuration
- `.env.example` - Environment variables template

### Documentation
- `docs/API.md` - API endpoint documentation
- `docs/PROMPTS.md` - Prompt engineering guide
- `docs/PHASE_3_COMPLETE.md` - Completion checklist

---

## Dependencies

### NPM Packages (Vercel)
```json
{
  "@anthropic-ai/sdk": "^0.10.0",
  "@neondatabase/serverless": "^0.6.0",
  "ws": "^8.14.0"
}
```

### NPM Packages (React Native)
```json
{
  "@react-native-async-storage/async-storage": "^1.19.0"
}
```

### Environment Variables
```bash
# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Neon
DATABASE_URL=postgresql://user:pass@host/db

# App
API_BASE_URL=https://your-app.vercel.app
ENABLE_LLM=true
CACHE_TTL_SECONDS=604800
```

---

## Timeline

### Week 1: Infrastructure
- Day 1-2: Vercel setup + API routes
- Day 3: Neon schema + migrations
- Day 4-5: Claude API integration + testing

### Week 2: Implementation
- Day 1-2: Interpretation engine
- Day 3-4: Luna/Sol narration
- Day 5: Caching & optimization

**Total: 10-12 days** (assuming 2-3 hours/day)

---

## Cost Estimates

### Anthropic Claude API
- Average reading: 3 cards
- Tokens per card: ~500 input + 300 output = 800 total
- 3 cards = 2,400 tokens
- Synthesis: ~1,000 tokens
- **Total per reading:** ~3,400 tokens
- **Cost:** $0.01 per reading (with caching)

### Neon Database
- Free tier: 0.5GB storage
- Estimated usage: <0.1GB for 10k readings
- **Cost:** $0/month (free tier)

### Vercel
- Free tier: 100GB-hours/month
- Estimated usage: ~10GB-hours/month
- **Cost:** $0/month (free tier)

**Total monthly cost (1k readings):** ~$10

---

## Risk Mitigation

### API Rate Limits
- **Risk:** Anthropic rate limits (50 requests/min)
- **Mitigation:** Client-side queueing, caching

### API Costs
- **Risk:** Unexpected usage spikes
- **Mitigation:** Set Vercel budget alerts, aggressive caching

### Database Connection Limits
- **Risk:** Neon connection pool exhaustion
- **Mitigation:** Use connection pooling, close connections properly

### Network Failures
- **Risk:** Offline users can't get interpretations
- **Mitigation:** Graceful fallback to stub functions, local cache

---

## Next Steps

1. **Immediate:** Set up Vercel project (`vercel init`)
2. **Next:** Create Neon database and run schema
3. **Then:** Build first API endpoint (`/api/interpret-card`)
4. **After:** Integrate with React Native app
5. **Finally:** Add Luna/Sol narration layer

---

**Status:** 📋 Planning Complete → Ready for Implementation

**Last Updated:** 2025-11-20
