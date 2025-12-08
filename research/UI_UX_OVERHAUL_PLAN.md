# VeilPath UI/UX Overhaul Plan

## Executive Summary

**Problem**: Current app has 9 bottom tabs, janky alignment, no first-time guidance, and users are confused.

**Solution**: Reduce to 4 tabs, add first-time tour, fix layouts, integrate ads from day 1.

---

## 1. Current State (Problems)

### Navigation Overload
```
Current: Home | Reading | Journal | Mindfulness | Oracle | Quests | Shop | Locker | Profile
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         9 TABS - Most users can't even see them all without scrolling
```

**Industry Standard**: 4-5 tabs maximum
- Instagram: 5 tabs
- Headspace: 5 tabs
- Calm: 5 tabs
- Spotify: 5 tabs

### Layout Issues
1. "Welcome back, Seeker" left-aligned despite `textAlign: 'center'`
2. Cards don't align to viewport center
3. Inconsistent padding/margins between sections
4. Mobile vs web layouts diverge significantly

### Missing First-Time Experience
- Users dropped directly into home screen
- No explanation of what anything does
- Card of the Day is confusing without context
- No guided path through core features

---

## 2. Proposed Navigation Structure

### Option A: 4 Tabs (Recommended)

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│              [Main Content Area]                    │
│                                                     │
├─────────────┬─────────────┬─────────────┬──────────┤
│    Home     │   Practice  │   Journal   │    Me    │
│     🏠      │     🔮      │     📔      │    👤    │
└─────────────┴─────────────┴─────────────┴──────────┘
```

| Tab | Contains | Notes |
|-----|----------|-------|
| **Home** | Daily Oracle, Stats, Quick Actions, News | First thing users see |
| **Practice** | All readings, Mindfulness, CBT/DBT, Oracle Chat | Unified "doing" space |
| **Journal** | Journal entries, Mood tracking, Prompts | Writing/reflection space |
| **Me** | Profile, Shop, Locker, Quests, Settings, Achievements | Everything personal |

### Option B: 3 Tabs + FAB

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│              [Main Content Area]                    │
│                                                     │
│                              ┌─────┐                │
│                              │ 🔮 │ ← Floating      │
│                              └─────┘   Action       │
├───────────────┬───────────────┬─────────────────────┤
│     Home      │    Journal    │        Me           │
│      🏠       │      📔       │        👤           │
└───────────────┴───────────────┴─────────────────────┘
```

FAB opens radial menu with: New Reading, Oracle, Quick Journal

### Option C: Sidebar/Drawer (Desktop-first)

For web, consider a persistent left sidebar:

```
┌────────────────────────────────────────────────────────────┐
│ VEILPATH                                                    │
├──────────────┬─────────────────────────────────────────────┤
│              │                                             │
│ 🏠 Home      │          [Main Content Area]                │
│ 🔮 Reading   │                                             │
│ 📔 Journal   │                                             │
│ 🧘 Mindful   │                                             │
│ 💬 Oracle    │                                             │
│ ─────────    │                                             │
│ 🎯 Quests    │                                             │
│ 🛒 Shop      │                                             │
│ 📦 Locker    │                                             │
│ ─────────    │                                             │
│ ⚙️ Settings  │                                             │
│ 👤 Profile   │                                             │
│              │                                             │
└──────────────┴─────────────────────────────────────────────┘
```

---

## 3. Recommended: Option A with Progressive Disclosure

### Tab Structure

**Tab 1: Home**
- Daily Oracle card (primary focus)
- Quick stats (streak, readings, entries)
- Quick action buttons (New Reading, Journal, Oracle)
- First-time tour entry point

**Tab 2: Practice** (Hub for all activities)
- Horizontal category pills: Readings | Mindfulness | Therapy
- Readings sub-section:
  - Single Card
  - Three-Card Spread
  - Celtic Cross
  - Oracle Chat
- Mindfulness sub-section:
  - Breathing exercises
  - Guided meditations
  - Body scan
- Therapy sub-section:
  - CBT Distortion Identifier
  - DBT Skills Library
  - Mood Check-in

**Tab 3: Journal**
- Entry list (reverse chronological)
- Mood trend graph
- Journal prompts
- Search/filter
- + New Entry FAB

**Tab 4: Me**
- Profile header (avatar, name, level, XP bar)
- Quick links grid:
  - 🎯 Quests
  - 🛒 Shop
  - 📦 Locker/Collection
  - 🏆 Achievements
  - 📊 Statistics
  - ⚙️ Settings

---

## 4. First-Time Tour System

### Tour Sequence (5-7 steps)

**Step 1: Welcome**
```
┌─────────────────────────────────────────┐
│           ✧ Welcome to VeilPath ✧        │
│                                          │
│   Your personal sanctuary for tarot,     │
│   mindfulness, and self-discovery.       │
│                                          │
│   Let me show you around...              │
│                                          │
│            [Begin Journey →]             │
└─────────────────────────────────────────┘
```

**Step 2: Daily Oracle (highlight Home tab)**
```
┌─────────────────────────────────────────┐
│        ← Spotlight on Daily Card         │
│  ╔═══════════════════════════════╗      │
│  ║   Your Daily Oracle awaits    ║      │
│  ║   Tap to reveal your card     ║      │
│  ╚═══════════════════════════════╝      │
│                                          │
│ Each day, a new card guides your path.   │
│                                          │
│            [Next →]                      │
└─────────────────────────────────────────┘
```

**Step 3: Practice Tab (highlight Practice tab)**
```
Explore different reading types, from quick
single-card pulls to deep Celtic Cross spreads.

Plus: Oracle AI chat, mindfulness exercises,
and therapy tools when you need them.
```

**Step 4: Journal Tab (highlight Journal tab)**
```
Capture your thoughts, track your mood,
and reflect on your readings.

Your entries stay private unless you choose
to sync to the cloud.
```

**Step 5: Me Tab (highlight Me tab)**
```
Your profile, achievements, collection, and
the Shop live here.

Customize your experience with card backs,
animations, and themes.
```

**Step 6: First Bonus (Shop incentive)**
```
┌─────────────────────────────────────────┐
│       💎 A Gift to Start Your Journey    │
│                                          │
│   Here's 1,000 Veil Shards to explore    │
│   the Mystical Shop!                     │
│                                          │
│         [Claim & Visit Shop →]           │
└─────────────────────────────────────────┘
```

**Step 7: Begin**
```
┌─────────────────────────────────────────┐
│          You're Ready, Seeker            │
│                                          │
│   Start by revealing your Daily Oracle   │
│   or dive into a full reading.           │
│                                          │
│          [Begin Your Journey]            │
└─────────────────────────────────────────┘
```

### Tour Technical Implementation

```javascript
// Tour state in userStore
onboarding: {
  hasSeenTour: false,
  tourStep: 0,
  tourCompletedAt: null,
}

// Tour overlay component
<TourOverlay
  steps={TOUR_STEPS}
  currentStep={tourStep}
  onNext={() => setTourStep(tourStep + 1)}
  onSkip={() => completeTour()}
  onComplete={() => {
    completeTour();
    triggerFirstShopVisitBonus();
  }}
/>
```

---

## 5. Layout Fixes

### Fix 1: Center "Welcome back, Seeker"

**Before:**
```javascript
headerTop: {
  flexDirection: 'row',
  justifyContent: 'space-between',  // ❌ Forces left-align
  ...
}
```

**After:**
```javascript
headerTop: {
  alignItems: 'center',
  marginBottom: 16,
}
greeting: {
  fontSize: 24,
  fontWeight: '300',
  color: COSMIC.moonGlow,
  textAlign: 'center',  // ✅ Now works
  width: '100%',        // ✅ Full width
}
```

### Fix 2: Consistent Container

Create a reusable `ScreenContainer` component:

```javascript
const ScreenContainer = ({ children, centered = false }) => (
  <View style={[
    styles.container,
    centered && styles.centered
  ]}>
    <View style={styles.content}>
      {children}
    </View>
  </View>
);

// Styles
container: {
  flex: 1,
  alignItems: 'center',
},
content: {
  width: '100%',
  maxWidth: 600,  // Prevents ultra-wide on desktop
  paddingHorizontal: 16,
}
```

### Fix 3: Card Grid System

```javascript
// Consistent spacing
const GRID = {
  gutter: 16,
  cardPadding: 20,
  sectionGap: 24,
};

// Card component with consistent sizing
<Card style={{
  padding: GRID.cardPadding,
  marginBottom: GRID.sectionGap,
  width: '100%',
}}/>
```

---

## 6. Ad Integration Points

### Built Into New Layout

| Location | Ad Type | User Experience |
|----------|---------|-----------------|
| **Reading Complete** | Rewarded Video | "Watch 30s for bonus insight" |
| **Oracle Cooldown** | Rewarded Video | "Watch to unlock 1 free chat" |
| **Journal List** | Native (every 5 entries) | Blends with entry cards |
| **Me Tab Footer** | Banner | Below the fold, unobtrusive |
| **Practice Tab** | Interstitial (rare) | Only after 5th reading of day |

### Ad-Free Experience (Subscribers)

All subscribers get:
- No banner ads
- No interstitials
- Rewarded ads still available (optional bonus)

---

## 7. Implementation Priority

### Phase 1: Navigation Restructure (Week 1)
1. Create new 4-tab navigator
2. Merge screens into new structure
3. Update all navigation calls

### Phase 2: Tour System (Week 1-2)
1. Build TourOverlay component
2. Define tour steps
3. Connect to bonus system
4. Add skip/complete logic

### Phase 3: Layout Fixes (Week 2)
1. Fix alignment issues
2. Create ScreenContainer
3. Standardize card/grid system
4. Test on mobile/tablet/desktop

### Phase 4: Ad Infrastructure (Week 2-3)
1. Set up Google AdMob (or web alternative)
2. Create ad slot components
3. Implement rewarded video flow
4. A/B test placements

---

## 8. Metrics to Track

| Metric | Target | Why |
|--------|--------|-----|
| Tour completion rate | >80% | Users understand the app |
| Day 1 retention | >40% | First impression works |
| Tab engagement distribution | Even | All features discovered |
| Ad click-through rate | 1-3% | Ads aren't annoying |
| Rewarded ad completion | >50% | Users find value |

---

## Next Steps

1. **Get your approval** on navigation structure (Option A/B/C?)
2. **Finalize tour content** - Want to review the copy?
3. **Choose ad network** - AdMob? Unity Ads? Something web-native?
4. **Start implementation** - I can begin with the tour component
