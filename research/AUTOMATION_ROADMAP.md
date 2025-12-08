# VeilPath Automation Roadmap

**Lead:** Claude (with Ryan oversight)
**Stack:** Vercel + Supabase + React Native
**Goal:** Push to prod for bugs only, every 1-2 weeks max

---

## The Vision

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VEILPATH AUTONOMOUS OPERATION                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   DAILY (Automated)                                                          │
│   ├── Shop rotation (new items featured)                                     │
│   ├── Quest refresh (daily challenges)                                       │
│   ├── Featured content cycling                                               │
│   └── Analytics digest                                                       │
│                                                                              │
│   WEEKLY (Automated)                                                         │
│   ├── New cosmetics drop from asset queue                                    │
│   ├── Achievement unlocks based on calendar                                  │
│   ├── Weekly quest chain progression                                         │
│   └── Performance report                                                     │
│                                                                              │
│   MONTHLY/SEASONAL (Pre-scheduled)                                           │
│   ├── Holiday events (Christmas, Halloween, etc.)                            │
│   ├── Limited edition cosmetics                                              │
│   ├── Contest/community events                                               │
│   └── Major feature unlocks                                                  │
│                                                                              │
│   HUMAN INTERVENTION (Bugs only)                                             │
│   └── Push to prod every 1-2 weeks for fixes                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Foundation (This Sprint)

### 1.1 Guardian Security Layer
**Priority:** CRITICAL (before any public API)

```typescript
// src/services/guardian/index.ts
export class Guardian {
  // OSS Foundation
  private rateLimiter: UpstashRatelimit;
  private inputScanner: LLMGuard;

  // Patent IP Layer
  private kxfConfidence: KxfCalibrator;
  private kuramotoAdapter: FlowStateAdapter;

  async processRequest(input: string, session: Session): Promise<GuardedRequest> {
    // Rate limit check
    const { success } = await this.rateLimiter.limit(session.userId);
    if (!success) return { blocked: true, reason: 'rate_limited' };

    // Input sanitization
    const sanitized = await this.inputScanner.scan(input);
    if (sanitized.flagged) return { blocked: true, reason: sanitized.reason };

    // Trust scoring
    const trust = await this.calculateTrust(session);

    return { blocked: false, input: sanitized.text, trust };
  }
}
```

### 1.2 Multi-Provider LLM Orchestrator
**Purpose:** Never hit rate limits, maximize free tiers

```typescript
// src/services/hive/orchestrator.ts
interface LLMProvider {
  id: string;
  endpoint: string;
  freeLimit: { requests: number; tokens: number; resetPeriod: 'day' | 'month' };
  currentUsage: { requests: number; tokens: number };
}

export class HiveOrchestrator {
  private providers: LLMProvider[] = [
    { id: 'anthropic', endpoint: '/api/llm/anthropic', freeLimit: { requests: 1000, tokens: 100000, resetPeriod: 'month' }},
    { id: 'google', endpoint: '/api/llm/google', freeLimit: { requests: 60, tokens: 1000000, resetPeriod: 'day' }},
    { id: 'groq', endpoint: '/api/llm/groq', freeLimit: { requests: 30, tokens: 500000, resetPeriod: 'day' }},
  ];

  async selectProvider(task: TaskType): Promise<LLMProvider> {
    // Find provider with most remaining quota
    const available = this.providers
      .filter(p => p.currentUsage.requests < p.freeLimit.requests)
      .sort((a, b) => this.getAvailability(b) - this.getAvailability(a));

    return available[0] || this.fallbackProvider;
  }
}
```

### 1.3 Vercel Edge Functions
**Location:** `/api/` directory

```
api/
├── llm/
│   ├── anthropic.ts      # Claude API proxy
│   ├── google.ts         # Gemini API proxy
│   └── groq.ts           # Groq API proxy
├── vera/
│   ├── chat.ts           # Main Vera endpoint
│   └── interpret.ts      # Card interpretation
├── shop/
│   ├── inventory.ts      # Get current shop items
│   ├── purchase.ts       # Handle purchases
│   └── rotate.ts         # Cron: daily rotation
├── quests/
│   ├── daily.ts          # Get daily quests
│   ├── complete.ts       # Mark quest complete
│   └── refresh.ts        # Cron: daily refresh
└── cron/
    ├── daily-rotation.ts # 12:00 UTC daily
    ├── weekly-drop.ts    # Monday 00:00 UTC
    └── event-check.ts    # Hourly event triggers
```

---

## Phase 2: Content Pipeline (Week 2)

### 2.1 Supabase Schema

```sql
-- Shop System
CREATE TABLE shop_items (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  cosmetic_id TEXT NOT NULL REFERENCES cosmetics(id),
  price_coins INTEGER NOT NULL,
  price_gems INTEGER,
  featured BOOLEAN DEFAULT false,
  available_from TIMESTAMPTZ,
  available_until TIMESTAMPTZ,
  rotation_pool TEXT DEFAULT 'standard', -- 'standard', 'rare', 'event', 'holiday'
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Cosmetics Library (pre-loaded with all assets)
CREATE TABLE cosmetics (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  type TEXT NOT NULL, -- 'card_back', 'title', 'border', 'effect', 'avatar'
  rarity TEXT NOT NULL, -- 'common', 'uncommon', 'rare', 'epic', 'legendary'
  asset_path TEXT NOT NULL,
  preview_path TEXT,
  unlock_method TEXT DEFAULT 'purchase', -- 'purchase', 'achievement', 'event', 'quest'
  base_price INTEGER NOT NULL,
  release_date DATE,
  event_id TEXT REFERENCES events(id),
  metadata JSONB DEFAULT '{}'
);

-- Daily Quests
CREATE TABLE quests (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  quest_type TEXT NOT NULL, -- 'daily', 'weekly', 'achievement', 'event'
  title TEXT NOT NULL,
  description TEXT NOT NULL,
  requirements JSONB NOT NULL, -- { "action": "complete_reading", "count": 3 }
  rewards JSONB NOT NULL, -- { "coins": 50, "xp": 100, "cosmetic_id": null }
  difficulty TEXT DEFAULT 'normal',
  rotation_pool TEXT DEFAULT 'standard',
  active BOOLEAN DEFAULT true
);

-- Daily Quest Assignments
CREATE TABLE daily_quest_assignments (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES users(id),
  quest_id UUID REFERENCES quests(id),
  assigned_date DATE NOT NULL,
  completed BOOLEAN DEFAULT false,
  progress JSONB DEFAULT '{}',
  completed_at TIMESTAMPTZ,
  UNIQUE(user_id, quest_id, assigned_date)
);

-- Events Calendar
CREATE TABLE events (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  type TEXT NOT NULL, -- 'holiday', 'seasonal', 'contest', 'flash_sale'
  start_date TIMESTAMPTZ NOT NULL,
  end_date TIMESTAMPTZ NOT NULL,
  config JSONB NOT NULL, -- Event-specific configuration
  cosmetics TEXT[], -- Array of cosmetic IDs available during event
  quests TEXT[], -- Array of quest IDs active during event
  banner_asset TEXT,
  active BOOLEAN DEFAULT true
);

-- Asset Queue (for CI/CD pipeline)
CREATE TABLE asset_queue (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  asset_type TEXT NOT NULL,
  name TEXT NOT NULL,
  prompt TEXT NOT NULL, -- AI generation prompt
  status TEXT DEFAULT 'pending', -- 'pending', 'generating', 'review', 'approved', 'rejected'
  generated_path TEXT,
  reviewed_by TEXT,
  reviewed_at TIMESTAMPTZ,
  scheduled_release DATE,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 2.2 Daily Rotation Cron

```typescript
// api/cron/daily-rotation.ts
import { createClient } from '@supabase/supabase-js';

export const config = {
  runtime: 'edge',
  // Vercel Cron: runs at 12:00 UTC daily
  cron: '0 12 * * *'
};

export default async function handler() {
  const supabase = createClient(
    process.env.SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_KEY!
  );

  // 1. Rotate shop items
  await rotateShopItems(supabase);

  // 2. Refresh daily quests
  await refreshDailyQuests(supabase);

  // 3. Check for event triggers
  await checkEventTriggers(supabase);

  // 4. Release scheduled cosmetics
  await releaseScheduledCosmetics(supabase);

  return new Response(JSON.stringify({ success: true, timestamp: new Date() }));
}

async function rotateShopItems(supabase: SupabaseClient) {
  // Clear yesterday's featured
  await supabase
    .from('shop_items')
    .update({ featured: false })
    .eq('featured', true);

  // Select new featured items (weighted by rarity)
  const { data: pool } = await supabase
    .from('cosmetics')
    .select('id, rarity')
    .eq('unlock_method', 'purchase')
    .lte('release_date', new Date().toISOString());

  const featured = selectWeightedRandom(pool, 6, {
    common: 1,
    uncommon: 2,
    rare: 3,
    epic: 5,
    legendary: 8
  });

  // Insert/update shop items
  for (const item of featured) {
    await supabase.from('shop_items').upsert({
      cosmetic_id: item.id,
      price_coins: calculatePrice(item),
      featured: true,
      available_from: new Date(),
      available_until: addDays(new Date(), 1)
    });
  }
}

async function refreshDailyQuests(supabase: SupabaseClient) {
  // Get all users who need new quests
  const { data: users } = await supabase
    .from('users')
    .select('id');

  // Get quest pool
  const { data: questPool } = await supabase
    .from('quests')
    .select('*')
    .eq('quest_type', 'daily')
    .eq('active', true);

  // Assign 3 random quests per user
  for (const user of users) {
    const assigned = selectRandom(questPool, 3);

    for (const quest of assigned) {
      await supabase.from('daily_quest_assignments').insert({
        user_id: user.id,
        quest_id: quest.id,
        assigned_date: new Date().toISOString().split('T')[0]
      });
    }
  }
}
```

### 2.3 Event Calendar System

```typescript
// src/services/events/calendar.ts
export const EVENT_CALENDAR_2025_2026 = [
  // 2025 Holidays
  { id: 'winter_solstice_2025', name: 'Winter Solstice', type: 'holiday', start: '2025-12-21', end: '2025-12-23' },
  { id: 'christmas_2025', name: 'Mystic Christmas', type: 'holiday', start: '2025-12-24', end: '2025-12-26' },
  { id: 'new_year_2026', name: 'New Year Ritual', type: 'holiday', start: '2025-12-31', end: '2026-01-02' },

  // 2026 Events
  { id: 'lunar_new_year_2026', name: 'Lunar New Year', type: 'holiday', start: '2026-01-29', end: '2026-02-02' },
  { id: 'valentines_2026', name: 'Love & Connection', type: 'holiday', start: '2026-02-12', end: '2026-02-15' },
  { id: 'spring_equinox_2026', name: 'Spring Awakening', type: 'seasonal', start: '2026-03-19', end: '2026-03-22' },
  { id: 'april_fools_2026', name: 'The Fool\'s Journey', type: 'holiday', start: '2026-04-01', end: '2026-04-02' },
  { id: 'beltane_2026', name: 'Beltane Fire', type: 'holiday', start: '2026-04-30', end: '2026-05-02' },
  { id: 'summer_solstice_2026', name: 'Summer Solstice', type: 'seasonal', start: '2026-06-20', end: '2026-06-22' },
  { id: 'lammas_2026', name: 'First Harvest', type: 'holiday', start: '2026-08-01', end: '2026-08-02' },
  { id: 'fall_equinox_2026', name: 'Autumn Balance', type: 'seasonal', start: '2026-09-22', end: '2026-09-24' },
  { id: 'halloween_2026', name: 'Samhain', type: 'holiday', start: '2026-10-29', end: '2026-11-01' },
  { id: 'thanksgiving_2026', name: 'Gratitude Ritual', type: 'holiday', start: '2026-11-26', end: '2026-11-27' },

  // Flash Sales (monthly)
  { id: 'flash_jan_2026', name: 'New Beginnings Sale', type: 'flash_sale', start: '2026-01-15', end: '2026-01-16' },
  // ... more flash sales
];

// Pre-generate cosmetics for each event
export const EVENT_COSMETICS = {
  christmas_2025: [
    { id: 'cb_winter_frost', type: 'card_back', name: 'Winter Frost', rarity: 'rare' },
    { id: 'title_yuletide_seer', type: 'title', name: 'Yuletide Seer', rarity: 'epic' },
    { id: 'border_holly', type: 'border', name: 'Holly Wreath', rarity: 'uncommon' },
  ],
  halloween_2026: [
    { id: 'cb_haunted', type: 'card_back', name: 'Haunted Deck', rarity: 'legendary' },
    { id: 'title_spirit_walker', type: 'title', name: 'Spirit Walker', rarity: 'epic' },
    { id: 'effect_ghostly', type: 'effect', name: 'Ghostly Aura', rarity: 'rare' },
  ],
  // ... more event cosmetics
};
```

---

## Phase 3: CI/CD Art Pipeline (Week 3-4)

### 3.1 Procedural Cosmetic Generation

```typescript
// scripts/generate-cosmetics.ts
import Anthropic from '@anthropic-ai/sdk';
import { createClient } from '@supabase/supabase-js';
import Replicate from 'replicate';

interface CosmeticSpec {
  type: 'card_back' | 'border' | 'effect' | 'avatar';
  theme: string;
  rarity: 'common' | 'uncommon' | 'rare' | 'epic' | 'legendary';
  event?: string;
}

class CosmeticGenerator {
  private claude: Anthropic;
  private replicate: Replicate;
  private supabase: SupabaseClient;

  /**
   * Generate AI prompt for cosmetic asset
   */
  async generatePrompt(spec: CosmeticSpec): Promise<string> {
    const response = await this.claude.messages.create({
      model: 'claude-3-haiku-20240307',
      max_tokens: 500,
      messages: [{
        role: 'user',
        content: `Generate a Midjourney/DALL-E prompt for a VeilPath tarot app cosmetic:

Type: ${spec.type}
Theme: ${spec.theme}
Rarity: ${spec.rarity} (${this.getRarityDescription(spec.rarity)})
${spec.event ? `Event: ${spec.event}` : ''}

Style requirements:
- Dark fantasy aesthetic with ethereal glow
- Colors: Deep purples, golds, cosmic blues
- Must work as a mobile app asset (clean edges, readable at small sizes)
- ${spec.type === 'card_back' ? '750x1125px tarot card back' : ''}
- ${spec.type === 'border' ? 'Transparent PNG border frame' : ''}
- ${spec.type === 'effect' ? 'Animated particle effect (describe keyframes)' : ''}

Output ONLY the prompt, no explanation.`
      }]
    });

    return response.content[0].text;
  }

  /**
   * Generate image using Replicate (SDXL/Midjourney alternative)
   */
  async generateImage(prompt: string, spec: CosmeticSpec): Promise<string> {
    const output = await this.replicate.run(
      'stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b',
      {
        input: {
          prompt: prompt,
          negative_prompt: 'blurry, low quality, text, watermark, signature',
          width: spec.type === 'card_back' ? 750 : 512,
          height: spec.type === 'card_back' ? 1125 : 512,
          num_outputs: 1,
          scheduler: 'K_EULER',
          num_inference_steps: 50,
          guidance_scale: 7.5,
        }
      }
    );

    return output[0]; // URL to generated image
  }

  /**
   * Queue cosmetic for generation
   */
  async queueCosmetic(spec: CosmeticSpec): Promise<void> {
    const prompt = await this.generatePrompt(spec);

    await this.supabase.from('asset_queue').insert({
      asset_type: spec.type,
      name: this.generateName(spec),
      prompt: prompt,
      status: 'pending',
      scheduled_release: this.calculateReleaseDate(spec),
      metadata: spec
    });
  }

  /**
   * Process pending assets (called by cron)
   */
  async processPendingAssets(): Promise<void> {
    const { data: pending } = await this.supabase
      .from('asset_queue')
      .select('*')
      .eq('status', 'pending')
      .limit(5); // Process 5 at a time

    for (const asset of pending) {
      try {
        // Update status
        await this.supabase
          .from('asset_queue')
          .update({ status: 'generating' })
          .eq('id', asset.id);

        // Generate image
        const imageUrl = await this.generateImage(asset.prompt, asset.metadata);

        // Download and upload to storage
        const storagePath = await this.uploadToStorage(imageUrl, asset);

        // Update with result
        await this.supabase
          .from('asset_queue')
          .update({
            status: 'review',
            generated_path: storagePath
          })
          .eq('id', asset.id);

      } catch (error) {
        await this.supabase
          .from('asset_queue')
          .update({
            status: 'pending',
            metadata: { ...asset.metadata, error: error.message }
          })
          .eq('id', asset.id);
      }
    }
  }
}
```

### 3.2 Asset Pipeline Cron Jobs

```typescript
// api/cron/asset-pipeline.ts
export const config = {
  runtime: 'edge',
  cron: '0 */4 * * *' // Every 4 hours
};

export default async function handler() {
  const generator = new CosmeticGenerator();

  // 1. Process pending generations
  await generator.processPendingAssets();

  // 2. Check for upcoming events needing assets
  await checkUpcomingEventAssets();

  // 3. Ensure minimum asset buffer (always have 30 days of content ready)
  await ensureAssetBuffer();

  return new Response(JSON.stringify({ success: true }));
}

async function ensureAssetBuffer() {
  const supabase = createClient(/* ... */);

  // Count approved assets not yet released
  const { count } = await supabase
    .from('asset_queue')
    .select('*', { count: 'exact' })
    .eq('status', 'approved')
    .gte('scheduled_release', new Date().toISOString());

  // If buffer is low, queue more generations
  if (count < 30) {
    const generator = new CosmeticGenerator();
    const needed = 30 - count;

    // Generate mix of types and rarities
    for (let i = 0; i < needed; i++) {
      await generator.queueCosmetic({
        type: randomChoice(['card_back', 'border', 'title', 'effect']),
        theme: randomChoice(THEME_POOL),
        rarity: weightedChoice({
          common: 40,
          uncommon: 30,
          rare: 20,
          epic: 8,
          legendary: 2
        })
      });
    }
  }
}
```

### 3.3 Admin Review Interface (Optional - can approve via Supabase dashboard)

```sql
-- Quick approval queries for Supabase dashboard

-- View pending reviews
SELECT id, name, asset_type, prompt, generated_path, created_at
FROM asset_queue
WHERE status = 'review'
ORDER BY created_at ASC;

-- Approve asset
UPDATE asset_queue
SET status = 'approved', reviewed_by = 'ryan', reviewed_at = NOW()
WHERE id = '<asset_id>';

-- Reject asset (will regenerate)
UPDATE asset_queue
SET status = 'pending', metadata = metadata || '{"regenerate": true}'
WHERE id = '<asset_id>';

-- Release approved asset to cosmetics table
INSERT INTO cosmetics (id, name, type, rarity, asset_path, release_date, metadata)
SELECT
  'cosmetic_' || id,
  name,
  asset_type,
  (metadata->>'rarity')::text,
  generated_path,
  scheduled_release,
  metadata
FROM asset_queue
WHERE id = '<asset_id>' AND status = 'approved';
```

---

## Phase 4: Testing & Deployment

### 4.1 Automated Testing

```yaml
# .github/workflows/test.yml
name: Test & Deploy

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test

      - name: Type check
        run: npm run typecheck

      - name: Lint
        run: npm run lint

  deploy-preview:
    needs: test
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Vercel Preview
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}

  deploy-production:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Vercel Production
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          vercel-args: '--prod'
```

### 4.2 Weekly Deploy Schedule

```
MONDAY:    Review generated assets, approve/reject
TUESDAY:   Merge approved PRs to develop
WEDNESDAY: QA on staging
THURSDAY:  Deploy to production (if needed)
FRIDAY:    Monitor, hotfix only if critical
WEEKEND:   System runs autonomously
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Manual deploys | ≤2/month | GitHub Actions |
| Asset buffer | ≥30 days | Supabase query |
| Daily rotation uptime | 99.9% | Vercel cron logs |
| Shop engagement | +20% vs manual | Analytics |
| User retention | +15% | Cohort analysis |

---

## Next Steps (Immediate)

1. [ ] Create Supabase tables (schema above)
2. [ ] Implement Guardian security layer
3. [ ] Build daily rotation cron
4. [ ] Set up Vercel cron jobs
5. [ ] Create first batch of cosmetic generation prompts
6. [ ] Test end-to-end daily rotation
7. [ ] Deploy to staging
8. [ ] Seed cosmetics database with existing assets
9. [ ] Configure event calendar
10. [ ] CI/CD art pipeline v1

---

*Autonomous VeilPath: You build once, it runs forever.*
