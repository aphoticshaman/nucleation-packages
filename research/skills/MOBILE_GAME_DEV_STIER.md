# MOBILE_GAME_DEV_STIER.skill.md

## S-Tier Mobile Game Development: React Native + Serverless Architecture

**Version**: 1.0
**Domain**: Mobile Game Development, React Native, Serverless Backend, Gamification, Monetization
**Prerequisites**: React/React Native, Node.js, SQL, API design, Game design fundamentals
**Output**: Production-ready mobile game with retention mechanics and monetization

---

## 1. EXECUTIVE SUMMARY

This skill file codifies S-tier mobile game development for React Native apps with serverless backends. It covers the complete lifecycle from broken code to production-ready, monetized game with high retention.

**Core Principle**: Work backwards from player value, fix what's broken before adding features, optimize for retention before scale.

**The S-Tier Stack**:
```
React Native (Expo) → Vercel Serverless API → Neon PostgreSQL → LLM Integration (Claude)
RevenueCat (monetization) → Firebase (cloud sync) → EAS Build (deployment)
```

**Anti-Patterns to Avoid**:
- ❌ Adding features before fixing bugs
- ❌ Optimizing before measuring
- ❌ Building RPG systems without retention data
- ❌ Premature abstraction (isometric villages, complex game worlds)
- ❌ Asset bloat (12MB video backgrounds)
- ❌ Subscription-only monetization (need microtransactions too)

---

## 2. BACKWARDS PLANNING: THE ONLY WAY TO BUILD

### 2.1 The Failure Mode

Most devs work forwards:
```
❌ "Let me add this cool feature" → "Now this other thing" → "Wait, nothing works"
```

S-Tier devs work backwards:
```
✅ End Goal → What's needed before that → What's needed before THAT → Start here
```

### 2.2 The Backwards Planning Protocol

**Step 1: Define the End State**

Be brutally specific. Not "working app" but:
```
END STATE:
- App launches without crashes on Android 13-15
- Users complete a tarot reading in < 2 minutes
- 40% of users return next day (D1 retention)
- 10% convert to paying users within 7 days
- Average session: 8-12 minutes
- App store rating > 4.2 stars
- API costs < $0.05 per user per month
```

**Step 2: Work Backwards in Measurable Milestones**

```
END: App in production with 1000 DAU
  ↑
BEFORE: App passes beta testing (50 users, 7 days)
  ↑
BEFORE: All core flows work end-to-end
  ↑
BEFORE: Critical bugs fixed (app doesn't crash)
  ↑
BEFORE: Audit complete (know what's broken)
  ↑
BEFORE: Framework established (know HOW to fix)
  ↑
START: Write this skill file
```

**Step 3: Verify Each Milestone Has Clear Exit Criteria**

```
❌ "Fix bugs" (vague)
✅ "App launches, navigates to main menu, starts reading without crash" (testable)

❌ "Add gamification" (scope creep)
✅ "Daily login gives 50 currency, user sees balance increment" (minimal viable)

❌ "Optimize performance" (premature)
✅ "Reading screen renders in < 500ms on Pixel 6" (measurable)
```

### 2.3 The Backwards Planning Template

```markdown
## PROJECT: [Name]

### End Goal (The North Star)
- [Specific, measurable outcome]
- [Business metric: revenue, retention, MAU]
- [User experience metric: time to value, NPS]
- [Technical metric: crash rate, API cost]

### Milestone Chain (Work Backwards)
1. [Final state - deployed, monitored, profitable]
2. [Beta tested - 50 users validated]
3. [Feature complete - all flows work]
4. [Core flows work - happy path only]
5. [Bugs fixed - app doesn't crash]
6. [Audit done - know what's broken]
7. [Framework ready - know how to proceed]

### Current State
- [Brutal honesty about what works/doesn't]
- [Specific blockers]
- [What you DON'T know yet]

### Next Action (Start Here)
- [The ONE thing to do next]
- [Exit criteria for this action]
- [How to verify it worked]
```

---

## 3. BUG FIXING: FIX WHAT'S BROKEN BEFORE BUILDING NEW

### 3.1 The Bug Triage Protocol

**Priority 1: App-Breaking Bugs**
```
Symptoms:
- App won't start
- Immediate crash on launch
- Can't navigate to main screen
- Build fails

Fix Order:
1. Missing dependencies (npm install fails)
2. Import errors (module not found)
3. Navigation config broken (screen doesn't exist)
4. Critical API failures (no network = crash)

Time Box: Fix within 1 hour or revert last working state
```

**Priority 2: Core Flow Blockers**
```
Symptoms:
- Can start app but can't complete main user action
- Reading flow breaks midway
- Payment flow doesn't work
- Data doesn't save

Fix Order:
1. API endpoint errors (500s, 404s)
2. State management bugs (undefined data)
3. Database schema mismatches
4. Race conditions in async flows

Time Box: Fix within 4 hours or redesign flow
```

**Priority 3: UX Degradation**
```
Symptoms:
- Feature works but slowly
- UI jank or lag
- Poor error messages
- Confusing user flow

Fix Order:
1. Performance bottlenecks
2. Error handling gaps
3. Loading states missing
4. Unclear CTAs

Time Box: Fix within 1 day or defer to backlog
```

**Priority 4: Nice-to-Haves**
```
Symptoms:
- Feature works fine but could be better
- Edge cases not handled
- Missing polish

Action: Backlog these, fix after P1-P3 clear
```

### 3.2 The Debugging Methodology

**Step 1: Reproduce Reliably**
```javascript
// DON'T: "It crashes sometimes"
// DO: "It crashes when clicking 'Start Reading' after fresh install"

Test Script:
1. Fresh install (uninstall first)
2. Open app
3. Skip onboarding
4. Tap "Start Reading"
5. Expected: Reading screen
6. Actual: White screen crash
```

**Step 2: Isolate the Failure Point**
```javascript
// Binary search for the bug
console.log('[DEBUG] Step 1: Navigation triggered');
console.log('[DEBUG] Step 2: Route params:', params);
console.log('[DEBUG] Step 3: Screen mounted');
console.log('[DEBUG] Step 4: Data fetched:', data);
console.log('[DEBUG] Step 5: Render complete');

// One of these won't log - that's your bug location
```

**Step 3: Root Cause Analysis (5 Whys)**
```
Bug: App crashes on Start Reading

Why? Screen tries to access undefined card data
Why? API call failed
Why? Database query returned empty
Why? User progression row doesn't exist
Why? User registration didn't create progression row

ROOT CAUSE: Registration flow incomplete
FIX: Add progression row creation to registration endpoint
```

**Step 4: Fix + Verify + Prevent**
```javascript
// Fix
export async function registerUser(userId, deviceId) {
  await sql`INSERT INTO users (user_id, device_id) VALUES (${userId}, ${deviceId})`;

  // ADD THIS: Create progression row
  await sql`INSERT INTO user_progression (user_id) VALUES (${userId})`;

  return { success: true };
}

// Verify
// Test: Fresh registration → Start reading → Should work

// Prevent
// Add database constraint: user_progression.user_id REFERENCES users(user_id) ON DELETE CASCADE
// Add test: "New user can start reading immediately after registration"
```

### 3.3 Common React Native Bugs & Fixes

**Bug: "Element type is invalid" Error**
```javascript
// WRONG
import { Button } from './components/Button'; // Named import
export default Button; // But exported as default

// RIGHT
import Button from './components/Button'; // Default import

// OR
import { Button } from './components/Button'; // Named export
export { Button };
```

**Bug: "Can't find variable: expo"**
```bash
# Fix: Missing Expo dependency
npm install expo
# Or rebuild
npm install && npx expo start -c
```

**Bug: Navigation "The action 'NAVIGATE' with payload {name: 'Screen'} was not handled"**
```javascript
// WRONG: Screen not registered
<Stack.Navigator>
  <Stack.Screen name="Home" component={HomeScreen} />
  {/* CardReading not registered but trying to navigate to it */}
</Stack.Navigator>

navigation.navigate('CardReading'); // CRASHES

// RIGHT: Register all screens you navigate to
<Stack.Navigator>
  <Stack.Screen name="Home" component={HomeScreen} />
  <Stack.Screen name="CardReading" component={CardReadingScreen} />
</Stack.Navigator>
```

**Bug: AsyncStorage quota exceeded**
```javascript
// DON'T: Store entire reading history locally
await AsyncStorage.setItem('readings', JSON.stringify(all5000Readings));

// DO: Store last 20 readings, rest in cloud
const recentReadings = allReadings.slice(0, 20);
await AsyncStorage.setItem('readings', JSON.stringify(recentReadings));
```

**Bug: API call works in dev but fails in production**
```javascript
// WRONG: Using localhost in production build
const API_URL = 'http://localhost:3000/api'; // Works on dev machine only

// RIGHT: Use environment variables
const API_URL = process.env.EXPO_PUBLIC_API_URL || 'https://your-api.vercel.app/api';
```

---

## 4. PERFORMANCE OPTIMIZATION: MODERN PHONES ONLY

### 4.1 Target Devices (2024-2025 Mid to High Tier)

**Android**:
- Pixel 6/7/8 (Tensor chips)
- Samsung Galaxy S22/S23/S24 (Snapdragon 8 Gen 1+)
- OnePlus 10/11/12
- Min specs: 6GB RAM, 90Hz+ screen, Snapdragon 7 Gen 1+

**iOS**:
- iPhone 12 and newer (A14 Bionic+)
- iPhone SE 3rd gen (A15 Bionic)
- Min specs: A14 chip, 4GB RAM

**DON'T optimize for**:
- Budget phones (< $300)
- Phones > 3 years old
- Anything with < 4GB RAM

### 4.2 Performance Budgets

**App Launch**:
```
Target: 1.5 seconds from tap to interactive
- Splash screen: 300ms
- JS bundle load: 800ms
- Initial render: 400ms
```

**Screen Transitions**:
```
Target: < 300ms (60fps minimum)
- Use native animations (LayoutAnimation, Animated)
- Avoid re-renders during transition
- Pre-load next screen data
```

**API Calls**:
```
Target: p95 < 1 second
- Card interpretation: 800ms
- Reading synthesis: 1.5s (acceptable, complex)
- User data fetch: 200ms
- Cached responses: < 100ms
```

**Memory Usage**:
```
Target: < 200MB average
- Images: Lazy load, compress
- Videos: Avoid or stream
- Large datasets: Paginate
```

### 4.3 Optimization Techniques

**Image Optimization**:
```javascript
// WRONG: Load all 78 cards at once
const cards = [
  require('../assets/cards/card_00.png'),
  require('../assets/cards/card_01.png'),
  // ... 76 more
];

// RIGHT: Lazy load on demand
<Image
  source={{ uri: `https://cdn.yourapp.com/cards/card_${cardId}.png` }}
  style={styles.card}
  resizeMode="contain"
  // Only load when visible
  loadingIndicatorSource={require('../assets/card_back.png')}
/>
```

**List Optimization**:
```javascript
// WRONG: FlatList with all 500 readings
<FlatList data={allReadings} renderItem={renderReading} />

// RIGHT: Virtualized with pagination
<FlatList
  data={readings}
  renderItem={renderReading}
  initialNumToRender={10}
  maxToRenderPerBatch={10}
  windowSize={5}
  removeClippedSubviews={true}
  onEndReached={loadMoreReadings}
  onEndReachedThreshold={0.5}
/>
```

**Animation Performance**:
```javascript
// WRONG: setState in animation loop
Animated.timing(fadeAnim, {
  toValue: 1,
  duration: 1000,
  useNativeDriver: false // JS thread, janky
}).start();

// RIGHT: Use native driver
Animated.timing(fadeAnim, {
  toValue: 1,
  duration: 1000,
  useNativeDriver: true // Native thread, smooth
}).start();
```

**Bundle Size Optimization**:
```bash
# Analyze bundle
npx react-native-bundle-visualizer

# Remove unused dependencies
npm prune

# Use Hermes engine (Android)
# In app.json:
{
  "expo": {
    "android": {
      "enableHermesEngine": true
    }
  }
}
```

---

## 5. GAMIFICATION: RETENTION THROUGH PROGRESSION

### 5.1 The Retention Ladder

```
Day 1: HOOK
- First reading feels magical
- Immediate value (interpretation in < 30s)
- Clear next action ("Journal your thoughts")

Day 2-7: HABIT
- Daily quest: "Do a reading"
- Streak tracking: "2 day streak!"
- Small reward: +50 Moonlight currency

Week 2-4: INVESTMENT
- Unlock new spread types (3-card → Celtic Cross)
- Level up (Level 5: "Apprentice Reader")
- Collection progress: "15/78 cards discovered"

Month 2+: MASTERY
- Weekly challenges (journal reflections)
- Rare card unlocks (holographic variants)
- Social proof (share readings)
```

### 5.2 Currency System Design

**Soft Currency (Moonlight)**:
```
Earn:
- Daily login: +50
- Complete reading: +100
- Journal entry: +200
- Weekly challenge: +500
- Watch ad: +25

Spend:
- Unlock spread: 1000
- Extra reading (free tier): 500
- Card cosmetic: 2000
- Name change: 100

Design: Generous early, scarce later
- Day 1: Earn 500, unlock first spread (feels generous)
- Day 7: Earn 300/day, spreads cost 2000 (need 7 days OR pay)
```

**Hard Currency (Veil Shards)**:
```
Earn:
- Purchase only: $0.99 = 100 shards
- Rare: Weekly challenge bonus: +10
- Very rare: Achievement: +5

Spend:
- Skip timers: 50 shards
- Exclusive cosmetics: 500 shards
- Battle pass: 1000 shards ($9.99)

Design: ONLY monetization, never required
- Can enjoy full game without spending shards
- Shards = convenience + cosmetics only
```

### 5.3 Progression System Design

**XP & Levels**:
```javascript
// Level formula: Exponential with soft cap at 50
const xpForLevel = (level) => {
  if (level <= 10) return level * 100; // Early levels fast
  if (level <= 30) return level * 200; // Mid levels slower
  return level * 500; // Late levels grind (for hardcore users)
};

// XP rewards
const XP_REWARDS = {
  reading: 50,
  journal: 100,
  weeklyChallenge: 500,
  achievement: 200,
  dailyQuest: 75
};

// Level benefits (MUST be meaningful)
const LEVEL_UNLOCKS = {
  5: { unlock: 'three_card_spread', bonus: 'Daily Moonlight +25' },
  10: { unlock: 'celtic_cross', bonus: 'Free reading recharge -1 hour' },
  15: { unlock: 'custom_spread', bonus: 'Card discovery rate +10%' },
  20: { unlock: 'advanced_insights', bonus: 'Premium LLM access (Sonnet)' }
};
```

**Streaks (The Retention Hook)**:
```javascript
// Streak system: Forgiving but rewarding
const STREAK_CONFIG = {
  gracePeriod: 36, // 36 hours to maintain streak (not 24 - too harsh)

  rewards: {
    3: { moonlight: 100, message: "3 day streak! Keep it up!" },
    7: { moonlight: 300, veilShards: 5, message: "Week streak! Have some shards!" },
    14: { moonlight: 500, cosmetic: 'streak_card_back', message: "2 weeks! Here's exclusive card back" },
    30: { moonlight: 1000, veilShards: 50, badge: 'dedicated', message: "30 days! You're dedicated!" }
  },

  recovery: {
    cost: 50, // 50 Veil Shards to recover broken streak
    maxStreak: 100 // Cap at 100 to prevent infinite grinding
  }
};

// Implementation: Server-authoritative
async function checkStreak(userId) {
  const progression = await getUserProgression(userId);
  const now = new Date();
  const lastReading = new Date(progression.last_reading_date);
  const hoursSince = (now - lastReading) / 1000 / 60 / 60;

  if (hoursSince < 36) {
    // Streak continues
    await incrementStreak(userId);
  } else {
    // Streak broken (offer recovery)
    return { streakBroken: true, canRecover: true, cost: 50 };
  }
}
```

### 5.4 Achievement System

**Categories**:
```javascript
const ACHIEVEMENTS = {
  // Discovery
  first_reading: { name: "First Steps", reward: { moonlight: 100, xp: 50 } },
  all_major_arcana: { name: "Major Mysteries", reward: { veilShards: 10, xp: 500 } },
  all_78_cards: { name: "Complete Deck", reward: { veilShards: 50, cosmetic: 'master_card_back' } },

  // Engagement
  7_day_streak: { name: "Dedicated Seeker", reward: { moonlight: 300, xp: 200 } },
  100_readings: { name: "Experienced Reader", reward: { veilShards: 20, xp: 1000 } },

  // Social
  share_reading: { name: "Spread the Word", reward: { moonlight: 50 } },

  // Monetization
  first_purchase: { name: "Supporter", reward: { veilShards: 50, badge: 'supporter' } }
};
```

---

## 6. MONETIZATION: MICROTRANSACTIONS + SUBSCRIPTION

### 6.1 The Monetization Philosophy

**Core Principle**: Premium users should feel smart, not necessary.

```
Free Users:
- 2 readings/day (enough to build habit)
- Template interpretations (still valuable)
- Single card + 3-card spreads (covers 80% of use cases)
- Ads (non-intrusive, skippable)

Premium Subscription ($9.99/mo):
- Unlimited readings
- AI interpretations (Claude)
- All spread types
- No ads
- Cloud sync
- Priority support

Microtransactions:
- Extra readings (one-time): $0.99 for 5
- Unlock spread permanently: $1.99 each
- Cosmetics (card backs, themes): $0.99-$4.99
- Currency packs: $0.99 (100 shards) to $49.99 (7500 shards + bonus)
```

### 6.2 The Monetization Funnel

```
100 Free Users
  ↓
80 complete tutorial (20% drop)
  ↓
40 return Day 2 (50% D1 retention)
  ↓
20 return Day 7 (50% D7 retention)
  ↓
10 see paywall prompt (50% engagement)
  ↓
2-3 make purchase (20-30% conversion)
```

**Conversion Tactics**:
```javascript
// DON'T: Aggressive paywalls
if (!isPremium) {
  return <Paywall blocking={true} />; // User leaves
}

// DO: Value-first, gentle upsells
if (!isPremium && readingsToday >= 2) {
  return (
    <View>
      <Text>You've used your free readings today!</Text>
      <Button title="Get 5 more for $0.99" />
      <Button title="Go Premium (unlimited)" />
      <Button title="Come back tomorrow" onPress={showEncouragement} />
    </View>
  );
}
```

### 6.3 Microtransaction Psychology

**The Currency Obfuscation**:
```
DON'T show: "Unlock for $1.99"
DO show: "Unlock for 200 Veil Shards"

Why?
- Reduces pain of payment
- Users pre-buy shards ($4.99 pack), spend over time
- Leftover shards encourage more purchases
```

**The Anchoring Effect**:
```
Currency Packs:
- 100 shards: $0.99 (1¢ per shard) - "Bad deal"
- 500 shards: $4.99 (0.99¢ per shard) - Most popular ⭐
- 1200 shards: $9.99 (0.83¢ per shard) - "Best value"
- 2750 shards: $19.99 (0.72¢ per shard + 250 bonus)

Psychology: Middle option feels "right", expensive option makes it seem reasonable
```

**The Time Gate**:
```javascript
// Free users: 24h cooldown between free readings
// Premium: No cooldown
// Microtransaction: Skip cooldown for 50 shards ($0.50)

const canReadNow = async (userId) => {
  const lastReading = await getLastReadingTime(userId);
  const hoursSince = (Date.now() - lastReading) / 1000 / 60 / 60;

  if (isPremium(userId)) return { allowed: true };

  if (hoursSince < 24) {
    return {
      allowed: false,
      options: [
        { action: 'wait', timeLeft: formatTime(24 - hoursSince) },
        { action: 'skip', cost: 50, currency: 'shards' },
        { action: 'premium', price: '$9.99/mo', benefit: 'Unlimited forever' }
      ]
    };
  }

  return { allowed: true };
};
```

### 6.4 Subscription vs Microtransaction Balance

**The 70/30 Rule**:
```
Revenue Target:
- 70% from subscriptions (recurring, predictable)
- 30% from microtransactions (impulse, whales)

User Segmentation:
- 85% free (ad revenue, virality, data)
- 10% microtransaction (dolphins, $5-20/mo)
- 5% subscription (whales, $10-50/mo if they also buy cosmetics)
```

**The Upgrade Path**:
```
Day 1: Free user, discovers app
Day 3: Hits 2-reading limit, buys 5 extra readings for $0.99 (first purchase!)
Day 7: Buys another pack, realizes subscription is better value
Day 10: Converts to $9.99/mo subscription
Month 2: Still subscribed, buys exclusive card back for $2.99 (happy customer)
```

---

## 7. ASSET MANAGEMENT: PERFORMANCE VS BEAUTY

### 7.1 The Asset Budget

```
Total App Size Target: < 100MB
- Code bundle: 10MB
- Essential assets: 40MB
- Optional assets: 50MB (download on demand)

Per-Screen Budget: < 10MB loaded in memory
```

### 7.2 Asset Audit & Cleanup Protocol

**Step 1: Find Bloat**:
```bash
# Find files > 1MB
find assets/ -type f -size +1M -exec ls -lh {} \; | sort -k5 -h

# Find unused assets (compare with grep)
find assets/ -name "*.png" | while read file; do
  basename=$(basename "$file")
  if ! grep -r "$basename" src/; then
    echo "UNUSED: $file"
  fi
done
```

**Step 2: Compress**:
```bash
# PNG compression (lossless)
for file in assets/**/*.png; do
  pngquant --quality 80-95 --ext .png --force "$file"
done

# JPG compression
for file in assets/**/*.jpg; do
  jpegoptim --max=85 --strip-all "$file"
done

# Remove video files > 5MB (use streaming or remove)
find assets/ -name "*.mp4" -size +5M -delete
```

**Step 3: Organize**:
```
assets/
├── cards/                  # 78 PNGs, 50-80KB each = 5MB total
├── art/
│   ├── curated/           # ONLY best assets, hand-picked
│   │   ├── backgrounds/   # 5-10 backgrounds MAX
│   │   ├── buttons/       # Reusable UI elements
│   │   └── effects/       # Particles, animations
│   └── DELETED/           # Everything else GONE
├── sounds/                # < 1MB total (short SFX only)
└── icon.png               # Required
```

### 7.3 The Curated Assets Philosophy

**WRONG: Kitchen Sink**
```
assets/art/village/village_day_variant_1.png (2MB)
assets/art/village/village_day_variant_2.png (2MB)
assets/art/village/village_night_variant_1.png (2MB)
... 20 more variants

Total: 40MB for one feature user barely notices
```

**RIGHT: Intentional**
```
assets/art/curated/bg_main.png (400KB, compressed)
// ONE background, reused across app with tint variations

styles/backgrounds.js:
  daytime: { tint: 'rgba(255, 255, 255, 0.9)' }
  night: { tint: 'rgba(50, 50, 100, 0.8)' }

Total: 400KB for entire app theming
```

---

## 8. DATABASE MIGRATION: ZERO-DOWNTIME UPDATES

### 8.1 Migration Philosophy

**Core Principle**: Never break production. Always have rollback.

```
DON'T: "Let me just ALTER TABLE in production"
DO: Write migration, test locally, run on production with rollback script
```

### 8.2 Migration Script Template

```sql
-- migrations/002_add_progression_system.sql
-- Description: Add user progression tables for gamification
-- Author: Claude
-- Date: 2025-11-20
-- Rollback: migrations/002_rollback.sql

BEGIN;

-- Create new tables (safe - doesn't affect existing)
CREATE TABLE IF NOT EXISTS user_progression (
  user_id TEXT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
  level INTEGER NOT NULL DEFAULT 1,
  xp INTEGER NOT NULL DEFAULT 0,
  moonlight INTEGER NOT NULL DEFAULT 0 CHECK (moonlight >= 0),
  veil_shards INTEGER NOT NULL DEFAULT 0 CHECK (veil_shards >= 0),
  current_streak INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes (safe)
CREATE INDEX IF NOT EXISTS idx_progression_level ON user_progression(level DESC);
CREATE INDEX IF NOT EXISTS idx_progression_streak ON user_progression(current_streak DESC);

-- Backfill existing users (safe - one-time operation)
INSERT INTO user_progression (user_id)
SELECT user_id FROM users
WHERE NOT EXISTS (
  SELECT 1 FROM user_progression WHERE user_progression.user_id = users.user_id
);

-- Add new column to existing table (safe - has default)
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_premium BOOLEAN DEFAULT false;

COMMIT;

-- Verify
SELECT COUNT(*) FROM user_progression; -- Should match user count
SELECT * FROM user_progression LIMIT 5; -- Spot check
```

**Rollback Script**:
```sql
-- migrations/002_rollback.sql
BEGIN;

DROP TABLE IF EXISTS user_progression CASCADE;
ALTER TABLE users DROP COLUMN IF EXISTS is_premium;

COMMIT;
```

### 8.3 Running Migrations on Neon

**Step 1: Test Locally**
```bash
# Start local Postgres (Docker)
docker run --name postgres-test -e POSTGRES_PASSWORD=test -p 5432:5432 -d postgres

# Run migration
psql postgresql://postgres:test@localhost:5432/postgres -f migrations/002_add_progression_system.sql

# Verify
psql postgresql://postgres:test@localhost:5432/postgres -c "SELECT * FROM user_progression LIMIT 5;"

# Test rollback
psql postgresql://postgres:test@localhost:5432/postgres -f migrations/002_rollback.sql
```

**Step 2: Backup Production**
```bash
# Neon has automatic backups, but create snapshot before big migrations
# In Neon dashboard: Backups → Create snapshot → "Before progression migration"
```

**Step 3: Run on Production**
```bash
# Set production DB URL (from Vercel env vars)
export DATABASE_URL="postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require"

# Dry run (read-only check)
psql $DATABASE_URL -c "SELECT COUNT(*) FROM users;"

# Run migration
psql $DATABASE_URL -f migrations/002_add_progression_system.sql

# Verify
psql $DATABASE_URL -c "SELECT COUNT(*) FROM user_progression;"
psql $DATABASE_URL -c "SELECT * FROM user_progression ORDER BY created_at DESC LIMIT 10;"
```

**Step 4: Monitor**
```bash
# Watch for errors in Vercel logs
vercel logs --follow

# Test API endpoints that use new tables
curl https://your-api.vercel.app/api/user/progression?userId=test123

# If anything breaks: ROLLBACK IMMEDIATELY
psql $DATABASE_URL -f migrations/002_rollback.sql
```

---

## 9. TESTING: BREAK IT BEFORE USERS DO

### 9.1 The Testing Pyramid

```
       /\
      /E2E\         (5% of tests - slow, brittle, critical paths only)
     /______\
    /  API   \      (25% of tests - medium speed, test business logic)
   /__________\
  / Component  \    (70% of tests - fast, test UI components)
 /______________\
```

### 9.2 Component Testing (Jest + React Native Testing Library)

```javascript
// __tests__/CardReadingScreen.test.js
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import CardReadingScreen from '../screens/CardReadingScreen';

describe('CardReadingScreen', () => {
  it('should show card shuffle animation on mount', async () => {
    const { getByTestId } = render(<CardReadingScreen />);

    const shuffleAnimation = getByTestId('card-shuffle');
    expect(shuffleAnimation).toBeTruthy();

    await waitFor(() => {
      expect(getByTestId('card-spread')).toBeTruthy();
    }, { timeout: 3000 }); // Animation duration
  });

  it('should navigate to summary after all cards interpreted', async () => {
    const mockNavigation = { navigate: jest.fn() };
    const { getByText } = render(
      <CardReadingScreen navigation={mockNavigation} />
    );

    // Simulate interpreting all 3 cards
    fireEvent.press(getByText('Continue')); // Card 1
    await waitFor(() => expect(getByText('Continue')).toBeTruthy());

    fireEvent.press(getByText('Continue')); // Card 2
    await waitFor(() => expect(getByText('Continue')).toBeTruthy());

    fireEvent.press(getByText('Continue')); // Card 3
    await waitFor(() => {
      expect(mockNavigation.navigate).toHaveBeenCalledWith('ReadingSummary', expect.any(Object));
    });
  });
});
```

### 9.3 API Testing (Automated)

```javascript
// __tests__/api/interpret-card.test.js
describe('POST /api/interpret-card', () => {
  it('should return interpretation for valid card', async () => {
    const response = await fetch('http://localhost:3000/api/interpret-card', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        card: { id: 0, name: 'The Fool', reversed: false },
        intention: 'Test question'
      })
    });

    const data = await response.json();
    expect(response.status).toBe(200);
    expect(data.success).toBe(true);
    expect(data.data.interpretation).toBeTruthy();
    expect(data.meta.tokens).toBeGreaterThan(0);
  });

  it('should return 400 for missing card data', async () => {
    const response = await fetch('http://localhost:3000/api/interpret-card', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ intention: 'Test' }) // Missing card
    });

    expect(response.status).toBe(400);
  });
});
```

### 9.4 Manual Testing Checklist

```markdown
## Pre-Production Testing Checklist

### Fresh Install Flow
- [ ] Uninstall app completely
- [ ] Install from build
- [ ] App launches without crash
- [ ] Onboarding displays
- [ ] Skip/complete onboarding works
- [ ] Reaches main menu

### Core User Flow (Happy Path)
- [ ] Tap "Start Reading"
- [ ] Select spread type
- [ ] Enter intention/question
- [ ] Cards shuffle and reveal
- [ ] Interpretation loads (< 5 seconds)
- [ ] Can navigate to next card
- [ ] Final synthesis generates
- [ ] Summary screen displays
- [ ] Can save reading

### Gamification
- [ ] XP increases after reading
- [ ] Moonlight currency awarded
- [ ] Level up notification shows
- [ ] Daily quest completes
- [ ] Streak increments

### Monetization
- [ ] Free user hits 2-reading limit
- [ ] Paywall displays correctly
- [ ] Can purchase extra readings
- [ ] Can view subscription options
- [ ] Premium features locked for free users

### Offline Mode
- [ ] App works without internet (cached data)
- [ ] Graceful error for API calls
- [ ] Syncs when connection restored

### Edge Cases
- [ ] Rapid tapping doesn't crash
- [ ] Back button at any point doesn't crash
- [ ] App suspension/resume works
- [ ] Low memory doesn't crash
- [ ] Airplane mode toggle works
```

---

## 10. PRODUCTION DEPLOYMENT: SHIP WITH CONFIDENCE

### 10.1 Pre-Deployment Checklist

```markdown
## Code Ready?
- [ ] All P1/P2 bugs fixed
- [ ] Core flows tested end-to-end
- [ ] No console.errors in production build
- [ ] API keys in env vars (not hardcoded)
- [ ] Database migrations tested on staging
- [ ] Rollback scripts ready

## Assets Ready?
- [ ] Images compressed (< 200KB each)
- [ ] No unused assets in bundle
- [ ] Total app size < 100MB
- [ ] Icons and splash screens correct

## Config Ready?
- [ ] app.json version bumped
- [ ] package.json version matches
- [ ] Build number incremented (iOS/Android)
- [ ] Environment variables set in Vercel

## Monitoring Ready?
- [ ] Sentry/error tracking configured
- [ ] Analytics events firing
- [ ] API health checks working
- [ ] Database connection pooling tested
```

### 10.2 The Deployment Flow

```bash
# 1. Final commit
git status
git add .
git commit -m "Release v1.2.0: Add progression system, fix crash bugs"

# 2. Tag release
git tag -a v1.2.0 -m "Release 1.2.0"

# 3. Push to production branch
git push origin main --tags

# 4. Vercel auto-deploys API (30-40 seconds)
# Watch: https://vercel.com/your-project/deployments

# 5. Build mobile app
eas build --platform android --profile production
eas build --platform ios --profile production

# 6. Test build before submission
eas build:run --platform android --latest

# 7. Submit to stores
eas submit --platform android --latest
eas submit --platform ios --latest

# 8. Monitor rollout
# - Check Sentry for crash spikes
# - Watch API logs for 500 errors
# - Monitor user feedback

# 9. If anything breaks: ROLLBACK
git revert HEAD
git push origin main
# Vercel auto-deploys previous version
```

### 10.3 Post-Deployment Monitoring

```bash
# Check API health
curl https://your-api.vercel.app/api/health

# Watch logs
vercel logs --follow

# Check database
psql $DATABASE_URL -c "SELECT COUNT(*) FROM users WHERE created_at > NOW() - INTERVAL '1 hour';"

# Monitor error rate (Sentry, Datadog, etc.)
# Target: < 1% error rate
# Alert if: > 5% error rate or any P1 errors
```

---

## 11. COMMON PITFALLS & HOW TO AVOID

### 11.1 The "One More Feature" Trap

```
❌ "App almost works, but let me add this cool RPG village first"
Result: Nothing works, RPG village half-baked, user confused

✅ "App works for core flow. Ship it. Add features in v1.1 based on user feedback"
Result: Users have working app, next feature is data-driven
```

### 11.2 The "Perfect Code" Trap

```
❌ "Let me refactor this into a proper architecture before shipping"
Result: Months of refactoring, no user feedback, wrong architecture anyway

✅ "Code works but messy. Ship it. Refactor AFTER validating with users"
Result: Know what users actually use, refactor the right things
```

### 11.3 The "Premature Optimization" Trap

```
❌ "Let me optimize this to run on 2018 phones"
Result: Weeks of optimization, targeting users who won't pay

✅ "Works great on 2024 phones. Ship it. Optimize if low-end users complain"
Result: Target paying users first, expand later
```

### 11.4 The "Asset Bloat" Trap

```
❌ "Let me add 50 background variants so users have choice"
Result: 100MB app, nobody downloads

✅ "3 beautiful backgrounds. Users can unlock more later as DLC"
Result: 20MB app, monetization opportunity
```

---

## 12. THE S-TIER DEVELOPER MINDSET

### 12.1 Core Principles

1. **Work Backwards**: Start from user value, plan backwards to current state
2. **Fix Before Feature**: Broken app with 10 features < Working app with 3 features
3. **Measure Everything**: Guesses are free, data is truth
4. **Ship Small**: v1.0 should be embarrassingly minimal but WORKS
5. **Iterate Fast**: Weekly releases > Monthly perfection
6. **Delete Aggressively**: Code you don't ship can't break
7. **Respect User's Time**: Every second counts, every tap matters
8. **Monetize Ethically**: Premium should feel smart, not required

### 12.2 The Daily Workflow

```
Morning:
- Review yesterday's crash reports (fix P1s immediately)
- Check analytics (retention, engagement, conversion)
- Prioritize: Fix > Feature > Polish

Afternoon:
- Code in 2-hour blocks (deep work)
- Test after every feature (don't accumulate bugs)
- Commit frequently ("Fix card flip animation" not "Various fixes")

Evening:
- Deploy to staging (test in production-like environment)
- Write next day's backwards plan
- Update skill files with lessons learned
```

### 12.3 When to Ask for Help

```
DON'T ask:
- "How do I do X?" (Google first, docs second, ask third)
- "Why doesn't this work?" (Debug first, include full error + context)

DO ask:
- "Tried X, Y, Z. All failed because [reasons]. What am I missing?"
- "Debugging shows [specific data]. Expected [X], got [Y]. Theory: [Z]?"
- "Researched A and B. Trade-offs: [list]. Recommendation?"
```

---

## 13. APPENDIX: QUICK REFERENCE

### 13.1 Common Commands

```bash
# Development
npm install                          # Install dependencies
npx expo start                       # Start dev server
npx expo start --clear              # Clear cache and start

# Building
eas build --platform android --profile preview    # Android preview
eas build --platform ios --profile preview        # iOS preview
eas build --platform all --profile production     # Production builds

# Database
psql $DATABASE_URL                   # Connect to database
psql $DATABASE_URL -f migration.sql  # Run migration
psql $DATABASE_URL -c "SELECT COUNT(*) FROM users;" # Quick query

# Git
git status -sb                       # Short status
git log --oneline -10               # Recent commits
git diff --stat                      # Files changed
git commit -m "msg" && git push     # Commit and push

# Debugging
adb logcat | grep -i "error"        # Android logs
npx react-devtools                  # React DevTools
```

### 13.2 Performance Targets

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| App Launch | < 1.5s | > 3s |
| Screen Transition | < 300ms | > 500ms |
| API Response (p95) | < 1s | > 2s |
| Memory Usage | < 200MB | > 400MB |
| App Size | < 100MB | > 150MB |
| Crash-Free Rate | > 99.5% | < 98% |

### 13.3 Monetization Benchmarks

| Metric | Good | Great | Exceptional |
|--------|------|-------|-------------|
| D1 Retention | 40% | 50% | 60%+ |
| D7 Retention | 20% | 30% | 40%+ |
| D30 Retention | 10% | 15% | 25%+ |
| Conversion Rate | 2% | 5% | 10%+ |
| ARPU (month 1) | $0.50 | $1.00 | $2.00+ |
| LTV (90 day) | $5 | $10 | $20+ |

---

## FINAL WORDS

S-Tier development is not about being the smartest or fastest. It's about:
- **Planning backwards** from user value
- **Shipping working software** over perfect code
- **Measuring obsessively** and iterating fast
- **Respecting users** with performance and ethics

The code you ship teaches you more than the code you write.

Ship early. Ship often. Ship with confidence.

---

**Version**: 1.0
**Last Updated**: 2025-11-20
**Maintainer**: Claude (S-Tier Mobile Game Dev Agent)
