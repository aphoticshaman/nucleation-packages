# Cosmetic Rewards & Premium Feature Gates

## Philosophy

**Free tier**: Hook them, make them love the app, show them what's possible
**Premium tier**: "I NEED that feature, take my $4.99"

The goal is to create moments where users think:
> "This app is amazing free, but if I just had [X], it would be perfect..."

---

## Part 1: Cosmetic Reward System

### Acquisition Methods

| Method | What They Get | Psychology |
|--------|---------------|------------|
| **Default** | 3 basic card backs, 2 animations | "This is nice" |
| **Streaks** | Rare card backs, badges | "I can't break my streak!" |
| **Achievements** | Epic animations, titles | "I earned this" |
| **Holidays** | Limited-time exclusives | "I'll never get this again!" |
| **Level-ups** | Unlock slots, new options | "I'm progressing" |
| **Shop (Moonlight)** | Common/Rare cosmetics | "I can grind for this" |
| **Shop (Veil Shards)** | Epic/Legendary cosmetics | "Worth spending real money" |
| **Subscription** | Exclusive monthly cosmetics | "Only subscribers have this" |

---

### Card Backs - Full Catalog

#### FREE (Default)
Everyone gets these immediately:

| ID | Name | Rarity |
|----|------|--------|
| `celestial_default` | Celestial | Common |
| `cosmic_void` | Cosmic Void | Common |
| `ethereal_mist` | Ethereal Mist | Common |

#### EARNED (Streaks & Achievements)

| ID | Name | Rarity | How to Unlock |
|----|------|--------|---------------|
| `ember_glow` | Ember Glow | Rare | 7-day streak |
| `moonlit_path` | Moonlit Path | Rare | 14-day streak |
| `golden_dawn` | Golden Dawn | Rare | 30-day streak |
| `phoenix_rise` | Phoenix Rising | Epic | 50-day streak |
| `eternal_flame` | Eternal Flame | Epic | 100-day streak |
| `journeyman` | Journeyman's Mark | Rare | 10 journal entries |
| `chronicler` | Chronicler's Tome | Epic | 50 journal entries |
| `scribe_eternal` | Scribe Eternal | Legendary | 100 journal entries |
| `first_steps` | First Steps | Rare | Complete first reading |
| `oracle_touched` | Oracle Touched | Epic | 25 Oracle conversations |
| `mindful_master` | Mindful Master | Epic | 50 mindfulness sessions |
| `cbt_champion` | CBT Champion | Rare | Identify 25 distortions |
| `dbt_disciple` | DBT Disciple | Rare | Practice 10 DBT skills |

#### EARNED (Holidays - LIMITED TIME)

| ID | Name | Rarity | When Available |
|----|------|--------|----------------|
| `halloween_spirits` | Spirit Veil | Epic | Oct 15-31 |
| `winter_solstice` | Winter Solstice | Epic | Dec 15-Jan 5 |
| `lunar_new_year` | Lunar Blessing | Epic | Lunar New Year week |
| `spring_equinox` | Vernal Awakening | Rare | Mar 19-25 |
| `summer_solstice` | Solar Radiance | Rare | Jun 19-25 |
| `autumn_equinox` | Harvest Moon | Rare | Sep 19-25 |
| `valentines` | Lovers' Embrace | Rare | Feb 7-14 |
| `friday_13th` | Dark Omen | Epic | Every Friday 13th |
| `full_moon` | Full Moon Rising | Rare | Every full moon (24h) |
| `eclipse` | Eclipse | Legendary | Solar/lunar eclipses only |

#### PURCHASABLE (Veil Shards)

| ID | Name | Rarity | Price | Notes |
|----|------|--------|-------|-------|
| `blood_moon` | Blood Moon | Rare | 300 💎 | |
| `astral_dream` | Astral Dream | Rare | 400 💎 | |
| `void_walker` | Void Walker | Epic | 750 💎 | |
| `shadow_realm` | Shadow Realm | Epic | 750 💎 | |
| `ancient_oracle` | Ancient Oracle | Legendary | 1,500 💎 | |
| `celestial_throne` | Celestial Throne | Legendary | 2,000 💎 | |
| `primordial_chaos` | Primordial Chaos | Legendary | 2,500 💎 | Most expensive |

#### SUBSCRIBER EXCLUSIVE

| ID | Name | Rarity | Tier Required |
|----|------|--------|---------------|
| `seeker_sigil` | Seeker's Sigil | Rare | Seeker ($4.99) |
| `adept_aura` | Adept's Aura | Epic | Adept ($9.99) |
| `mystic_mantle` | Mystic's Mantle | Legendary | Mystic ($14.99) |
| `monthly_[month]_[year]` | Monthly Exclusive | Epic | Any paid tier |

**Monthly exclusives** are only available during that month. Miss it = gone forever.

---

### Flip Animations

#### FREE
| ID | Name | Description |
|----|------|-------------|
| `classic` | Classic | Simple flip |
| `fade` | Fade | Crossfade reveal |

#### EARNED
| ID | Name | Unlock |
|----|------|--------|
| `owlTurn` | Owl Turn | Level 20 |
| `sleepyYawn` | Sleepy Yawn | Level 15 |
| `butterflyMetamorphosis` | Butterfly | 100 readings |
| `strikeVictory` | Victory Strike | 50 readings |

#### PURCHASABLE (200-1,000 💎)
| ID | Name | Price |
|----|------|-------|
| `catPounce` | Cat Pounce | 300 💎 |
| `bunnyHop` | Bunny Hop | 300 💎 |
| `dogShake` | Dog Shake | 300 💎 |
| `frogLeap` | Frog Leap | 400 💎 |
| `fishSplash` | Fish Splash | 400 💎 |
| `scaredJelly` | Scared Jelly | 500 💎 |
| `peekaboo` | Peekaboo | 500 💎 |
| `excitedJump` | Excited Jump | 600 💎 |
| `tipsyStumble` | Tipsy Stumble | 750 💎 |
| `dizzyTwirl` | Dizzy Twirl | 750 💎 |

---

### Themes (Full App Reskin)

#### FREE
| ID | Name |
|----|------|
| `default` | Cosmic Purple (default) |

#### PURCHASABLE
| ID | Name | Price | Description |
|----|------|-------|-------------|
| `midnight_ritual` | Midnight Ritual | 1,500 💎 | Deep blacks, candle glow |
| `aurora_borealis` | Aurora Borealis | 1,500 💎 | Northern lights colors |
| `blood_moon_theme` | Blood Moon | 2,000 💎 | Dark red, crimson accents |
| `celestial_garden` | Celestial Garden | 2,000 💎 | Soft greens, moonflowers |
| `golden_age` | Golden Age | 2,500 💎 | Rich golds, warm tones |
| `void_embrace` | Void Embrace | 3,000 💎 | Pure black, silver accents |

#### SUBSCRIBER EXCLUSIVE
| ID | Name | Tier |
|----|------|------|
| `seeker_theme` | Seeker's Sanctuary | Seeker+ |
| `adept_theme` | Adept's Chamber | Adept+ |
| `mystic_theme` | Mystic's Domain | Mystic only |

---

### Profile Titles

Displayed under username. Status symbols.

#### EARNED
| Title | Unlock |
|-------|--------|
| Seeker | Default |
| Apprentice | Level 5 |
| Practitioner | Level 10 |
| Adept | Level 15 |
| Master | Level 20 |
| Sage | Level 30 |
| Enlightened | Level 40 |
| Eternal One | Level 50 |
| Streak Keeper | 30-day streak |
| Devoted Scribe | 50 journal entries |
| Oracle's Chosen | 100 Oracle chats |
| Early Adopter | Joined before [date] |
| Founding Member | First 1,000 users |

#### PURCHASABLE
| Title | Price |
|-------|-------|
| Mystic Wanderer | 500 💎 |
| Shadow Walker | 500 💎 |
| Starborn | 750 💎 |
| Veil Keeper | 1,000 💎 |
| Cosmic Entity | 2,000 💎 |

---

## Part 2: Premium Feature Gates

### The "I NEED This" Moments

These are features that free users can **see** and **taste** but not fully access.

---

### Gate 1: Reading Limits

**Free**: 3 readings per day
**Premium**: Unlimited

**The Hook**:
```
┌─────────────────────────────────────────────┐
│                                             │
│    You've used 3 of 3 daily readings        │
│                                             │
│    ┌─────────────────────────────────────┐  │
│    │  🔮 Want unlimited readings?        │  │
│    │                                     │  │
│    │  Upgrade to Seeker for just         │  │
│    │  $4.99/month                        │  │
│    │                                     │  │
│    │  [Unlock Unlimited →]               │  │
│    └─────────────────────────────────────┘  │
│                                             │
│    Your readings reset in: 14h 23m          │
│                                             │
│    [Watch Ad for 1 More] ← Rewarded ad      │
│                                             │
└─────────────────────────────────────────────┘
```

---

### Gate 2: Celtic Cross Spread

**Free**: Preview only (can see the spread, can't use it)
**Premium**: Full access

**The Hook**:
```
┌─────────────────────────────────────────────┐
│                                             │
│    CELTIC CROSS SPREAD                      │
│    The most comprehensive reading           │
│                                             │
│    ┌─────────────────────────────────────┐  │
│    │                                     │  │
│    │    [Blurred Celtic Cross Image]     │  │
│    │                                     │  │
│    │         🔒 PREMIUM FEATURE          │  │
│    │                                     │  │
│    └─────────────────────────────────────┘  │
│                                             │
│    10 cards reveal past, present, future,   │
│    hopes, fears, and ultimate outcome.      │
│                                             │
│    Available with Adept ($9.99/mo) or       │
│    use 200 💎 for single reading            │
│                                             │
│    [Upgrade to Adept] [Use 200 💎]          │
│                                             │
└─────────────────────────────────────────────┘
```

---

### Gate 3: Oracle AI (Limited vs Unlimited)

**Free**: 5 messages per day
**Premium (Adept+)**: Unlimited

**The Hook**:
```
┌─────────────────────────────────────────────┐
│                                             │
│  You: What does The Tower mean for my       │
│       relationship?                         │
│                                             │
│  Oracle: The Tower appearing in matters     │
│          of the heart suggests...           │
│          [Message truncated]                │
│                                             │
│  ─────────────────────────────────────────  │
│                                             │
│    💬 4 of 5 daily messages remaining       │
│                                             │
│    Unlock unlimited Oracle access with      │
│    Adept ($9.99/mo)                         │
│                                             │
│    [Continue Conversation →] (Premium)      │
│    [Watch Ad for 2 More Messages]           │
│                                             │
└─────────────────────────────────────────────┘
```

---

### Gate 4: Voice Narration

**Free**: Not available (button is visible but locked)
**Premium (Mystic only)**: Full feature

**The Hook**:
```
┌─────────────────────────────────────────────┐
│                                             │
│    THE MAGICIAN                             │
│                                             │
│    [Card Image]                             │
│                                             │
│    Interpretation:                          │
│    Through focused will, you shape...       │
│                                             │
│    ─────────────────────────────────────    │
│                                             │
│    🔊 Listen to Narration                   │
│                                             │
│    ┌─────────────────────────────────────┐  │
│    │  🔒 Voice narration is a Mystic     │  │
│    │     exclusive feature               │  │
│    │                                     │  │
│    │  Hear your readings spoken aloud    │  │
│    │  in a soothing, mystical voice.     │  │
│    │                                     │  │
│    │  [🎧 Preview Voice] [Upgrade →]     │  │
│    └─────────────────────────────────────┘  │
│                                             │
└─────────────────────────────────────────────┘
```

The **preview** lets them hear 10 seconds. Just enough to want more.

---

### Gate 5: Deep Interpretation Layers

**Free**: Basic interpretation
**Premium**: Full 4-layer interpretation

**The Hook**:
```
┌─────────────────────────────────────────────┐
│                                             │
│    THE MAGICIAN - INTERPRETATION            │
│                                             │
│    ✓ Surface Meaning                        │
│    "The Magician represents your power      │
│     to manifest your desires..."            │
│                                             │
│    🔒 Shadow Work (Premium)                 │
│    ┌─────────────────────────────────────┐  │
│    │ Unlock deeper psychological         │  │
│    │ insights with Seeker+               │  │
│    └─────────────────────────────────────┘  │
│                                             │
│    🔒 Archetypal Patterns (Premium)         │
│    ┌─────────────────────────────────────┐  │
│    │ Connect to Jungian archetypes       │  │
│    │ with Adept+                         │  │
│    └─────────────────────────────────────┘  │
│                                             │
│    🔒 Personalized Guidance (Premium)       │
│    ┌─────────────────────────────────────┐  │
│    │ AI-tailored advice based on your    │  │
│    │ history with Mystic                 │  │
│    └─────────────────────────────────────┘  │
│                                             │
│    [Unlock All Layers - Upgrade →]          │
│                                             │
└─────────────────────────────────────────────┘
```

They can **see** there's more. They just can't have it.

---

### Gate 6: Cloud Sync & Export

**Free**: Local only, no export
**Premium**: Cloud sync + beautiful PDF/image exports

**The Hook**:
```
┌─────────────────────────────────────────────┐
│                                             │
│    READING COMPLETE                         │
│                                             │
│    ✓ Saved to this device                   │
│                                             │
│    ─────────────────────────────────────    │
│                                             │
│    [Share Reading] 🔒                       │
│                                             │
│    ┌─────────────────────────────────────┐  │
│    │  Create beautiful shareable images  │  │
│    │  of your readings                   │  │
│    │                                     │  │
│    │  [Preview] ← shows blurred example  │  │
│    │                                     │  │
│    │  Available with Seeker ($4.99/mo)   │  │
│    └─────────────────────────────────────┘  │
│                                             │
│    [Sync to Cloud] 🔒                       │
│                                             │
│    ┌─────────────────────────────────────┐  │
│    │  Access your readings on any device │  │
│    │  Never lose your history            │  │
│    │                                     │  │
│    │  Available with Seeker ($4.99/mo)   │  │
│    └─────────────────────────────────────┘  │
│                                             │
└─────────────────────────────────────────────┘
```

---

### Gate 7: Journal Limits

**Free**: 10 entries total (ever)
**Premium**: Unlimited

**The Hook**:
```
┌─────────────────────────────────────────────┐
│                                             │
│    📔 YOUR JOURNAL                          │
│                                             │
│    Entry 10 of 10                           │
│                                             │
│    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                             │
│    You've reached the free journal limit.   │
│                                             │
│    Your existing entries are safe, but      │
│    you'll need to upgrade to add more.      │
│                                             │
│    ┌─────────────────────────────────────┐  │
│    │  Unlock unlimited journaling with   │  │
│    │  Seeker for $4.99/month             │  │
│    │                                     │  │
│    │  [Upgrade Now →]                    │  │
│    └─────────────────────────────────────┘  │
│                                             │
│    Or delete old entries to make room       │
│    [Manage Entries]                         │
│                                             │
└─────────────────────────────────────────────┘
```

---

### Gate 8: Ad-Free Experience

**Free**: Sees rewarded ads, occasional native ads
**Premium (Any tier)**: Completely ad-free

This one sells itself. No special UI needed.

---

### Gate 9: CBT/DBT Full Access

**Free**: Preview of 3 tools each
**Premium (Adept+)**: Full library

**The Hook**:
```
┌─────────────────────────────────────────────┐
│                                             │
│    🧠 CBT TOOLS                             │
│                                             │
│    ✓ Thought Record (Free)                  │
│    ✓ Cognitive Distortion Quiz (Free)       │
│    ✓ Simple Reframing (Free)                │
│                                             │
│    🔒 Advanced Tools (Adept+)               │
│    ┌─────────────────────────────────────┐  │
│    │ • Behavioral Activation Planner     │  │
│    │ • Core Belief Worksheet             │  │
│    │ • Worry Time Protocol               │  │
│    │ • ABCDE Technique                   │  │
│    │ • Decatastrophizing                 │  │
│    │ • + 12 more tools                   │  │
│    │                                     │  │
│    │ [Unlock All - $9.99/mo]             │  │
│    └─────────────────────────────────────┘  │
│                                             │
└─────────────────────────────────────────────┘
```

---

## Part 3: Feature Gate Matrix

| Feature | Free | Seeker $4.99 | Adept $9.99 | Mystic $14.99 |
|---------|------|--------------|-------------|---------------|
| Daily readings | 3/day | Unlimited | Unlimited | Unlimited |
| Journal entries | 10 total | Unlimited | Unlimited | Unlimited |
| Oracle messages | 5/day | 20/day | Unlimited | Unlimited |
| Single card spread | ✅ | ✅ | ✅ | ✅ |
| Three card spread | ✅ | ✅ | ✅ | ✅ |
| Celtic Cross | ❌ | ❌ | ✅ | ✅ |
| Basic interpretation | ✅ | ✅ | ✅ | ✅ |
| Shadow work layer | ❌ | ✅ | ✅ | ✅ |
| Archetypal layer | ❌ | ❌ | ✅ | ✅ |
| Personalized layer | ❌ | ❌ | ❌ | ✅ |
| Voice narration | ❌ | ❌ | ❌ | ✅ |
| CBT tools | 3 tools | 3 tools | All tools | All tools |
| DBT tools | 3 tools | 3 tools | All tools | All tools |
| Mindfulness | 5 sessions | 10 sessions | All sessions | All sessions |
| Cloud sync | ❌ | ✅ | ✅ | ✅ |
| Export/share | ❌ | ✅ | ✅ | ✅ |
| Ad-free | ❌ | ✅ | ✅ | ✅ |
| Exclusive cosmetics | ❌ | Monthly | Monthly | Monthly + Exclusive |
| Priority support | ❌ | ❌ | ✅ | ✅ |

---

## Part 4: Conversion Triggers

### When to Show Upgrade Prompts

| Trigger | What Happens |
|---------|--------------|
| Hit daily reading limit | Show upgrade modal |
| Try to access Celtic Cross | Show preview + upgrade |
| Hit Oracle message limit | Offer ad OR upgrade |
| Hit journal entry 10 | Show upgrade modal |
| Complete 7-day streak | "Celebrate! Also, subscribers get exclusive streak rewards..." |
| View locked cosmetic | "Get this with Seeker!" |
| Try to export reading | Show blurred preview + upgrade |
| Try voice narration | Play 10s preview + upgrade |
| After 5th reading ever | "Loving VeilPath? Here's 50% off first month..." |

### Soft vs Hard Gates

**Soft gates** (can bypass with ads/currency):
- Daily reading limit (+1 with rewarded ad)
- Oracle messages (+2 with rewarded ad)
- Single Celtic Cross reading (200 💎)

**Hard gates** (subscription only):
- Unlimited everything
- Voice narration
- Cloud sync
- Ad-free
- Exclusive cosmetics
- Deep interpretation layers

---

## Part 5: Implementation Priority

### Phase 1 (Launch)
1. ✅ Reading limits (3/day free)
2. ✅ Journal limits (10 total free)
3. ✅ Basic upgrade modals
4. ✅ Cosmetic unlock system

### Phase 2 (Month 2)
1. Oracle message limits
2. Celtic Cross gate
3. Interpretation layers
4. Rewarded ad bypasses

### Phase 3 (Month 3)
1. Voice narration (Mystic exclusive)
2. Export/share (Seeker+)
3. Cloud sync
4. Holiday cosmetics system

### Phase 4 (Month 4+)
1. CBT/DBT full gating
2. Personalized interpretation
3. Monthly exclusive cosmetics
4. Streak cosmetic rewards

---

## Summary

**Free tier**: Generous enough to love, limited enough to want more
**Seeker ($4.99)**: Removes annoyances (limits, ads)
**Adept ($9.99)**: Adds depth (Celtic Cross, full therapy tools, unlimited Oracle)
**Mystic ($14.99)**: Exclusive luxury (voice, personalized AI, exclusive cosmetics)

The goal: Make users think "This is worth $4.99" multiple times per session.
