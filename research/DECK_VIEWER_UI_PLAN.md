# Deck Viewer UI Plan
## Three-Tier Rarity System Integration

**Goal:** Transform deck viewer into Hearthstone-quality collection screen with rarity tiers, unlocking, and progression visualization.

---

## 📱 CURRENT STATE vs. DESIRED STATE

### Current (Assumptions):
- Simple grid of all 78 cards
- No rarity differentiation
- No unlock progression
- Static display

### Desired:
- Three-tier rarity system (common/rare/artifact)
- Visual differentiation by tier
- Unlock/collection progress tracking
- Animated card reveals
- Filter by arcana/suit/rarity/locked status
- Sort by various criteria
- Card detail view with unlock requirements

---

## 🎨 UI LAYOUT DESIGN

### **Main Collection Screen**

```
┌─────────────────────────────────────┐
│  COLLECTION (58/78 seen)            │← Header
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░ 74%         │← Progress bar
├─────────────────────────────────────┤
│                                      │
│  [All] [Major] [Minor] [Locked]     │← Filters
│  Sort: [Recently Unlocked ▼]        │← Sorting
│                                      │
├─────────────────────────────────────┤
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐          │
│  │ 🌟│ │ 💎│ │🔒 │ │ ⭐│          │← Card grid
│  │   │ │   │ │   │ │   │          │  (with tier badges)
│  └───┘ └───┘ └───┘ └───┘          │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐          │
│  │ 💎│ │ 🌟│ │ 💎│ │🔒 │          │
│  │   │ │   │ │   │ │   │          │
│  └───┘ └───┘ └───┘ └───┘          │
│         ... (scrollable)             │
│                                      │
├─────────────────────────────────────┤
│  Rarity Stats:                       │← Footer stats
│  Common: 78/78  Rare: 45/78         │
│  Artifact: 12/22 (Major Arcana only)│
└─────────────────────────────────────┘

Legend:
🔒 = Locked/unseen
🌟 = Common (always available)
💎 = Rare (unlocked)
⭐ = Artifact (unlocked, animated)
```

---

## 🎴 CARD DISPLAY STATES

### 1. **Locked/Unseen Card**
**Visual:**
- Show card back with lock icon overlay
- Greyscale or dark silhouette
- Slight opacity (70%)
- No glow effect
- "?" or card name teaser

**On Tap:**
- Modal shows unlock requirements
- "Draw this card in a reading to unlock"
- Progress toward unlock (e.g., "Seen in 2/5 readings")

---

### 2. **Common Tier (Unlocked)**
**Visual:**
- Full-color RWS card art
- Subtle white border (1-2px)
- No glow effect
- Small white badge in corner

**On Tap:**
- Full card detail view
- Show times drawn
- First drawn date
- Interpretation snippet
- "Upgrade to Rare" button/progress

---

### 3. **Rare Tier (Unlocked)**
**Visual:**
- Midjourney art (static PNG)
- Glowing cyan border (2-3px)
- Soft pulsing glow animation
- Cyan gem badge in corner
- Slightly larger than common cards in grid

**On Tap:**
- Full card detail with Midjourney art
- Unlock story/achievement
- Times drawn at rare tier
- "Upgrade to Artifact" button/progress (Major Arcana only)

---

### 4. **Artifact Tier (Unlocked - Major Arcana Only)**
**Visual:**
- Animated video loop
- Golden glowing border (3-4px)
- Radiant glow animation (purple/gold)
- Golden crown badge
- Noticeably larger/featured in grid
- Particle effects around card

**On Tap:**
- Full-screen animated card view
- Epic unlock animation replay
- Special achievement badge
- Rarity stats
- Share button (flex to friends)

---

## 📊 COLLECTION PROGRESS VISUALIZATION

### **Header Stats:**

```
┌────────────────────────────────────┐
│ COLLECTION PROGRESS                │
│                                    │
│ Cards Seen: 58/78 (74%)            │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░              │
│                                    │
│ Rarity Breakdown:                  │
│ ● Common: 78/78 (100%) - All Free │
│ ● Rare: 45/78 (58%)               │
│ ● Artifact: 12/22 (55%) Major Only│
│                                    │
│ Next Milestone: 60 cards for       │
│ "Mystic Collector" achievement     │
└────────────────────────────────────┘
```

---

## 🔍 FILTERING & SORTING

### **Filter Options:**

1. **All Cards** - Show everything
2. **Major Arcana** - 22 cards (can have artifacts)
3. **Minor Arcana** - 56 cards (wands/cups/swords/pentacles)
4. **By Suit** - Wands, Cups, Swords, Pentacles
5. **Locked Only** - Cards not yet unlocked
6. **Rare+ Only** - Only rare and artifact tiers
7. **Artifact Only** - Just the 22 animated Major Arcana

### **Sort Options:**

1. **Card Number** (0-77, default)
2. **Recently Unlocked** (newest first)
3. **Most Drawn** (popularity)
4. **Never Drawn** (help complete collection)
5. **Rarity (High→Low)** (Artifact, Rare, Common)
6. **Alphabetical** (card name)

---

## 💡 CARD DETAIL VIEW

When user taps a card:

```
┌────────────────────────────────────┐
│         [X Close]                  │← Close button
│                                    │
│      ┌─────────────────┐          │
│      │                 │          │
│      │   CARD IMAGE    │          │← Large card display
│      │  (animated if   │          │  (artifact tier)
│      │   artifact)     │          │
│      │                 │          │
│      └─────────────────┘          │
│                                    │
│  THE FOOL                          │← Card name
│  Major Arcana #0                   │← Arcana/number
│                                    │
│  ┌──────────────────┐             │
│  │ ⭐ ARTIFACT TIER │             │← Current tier badge
│  └──────────────────┘             │
│                                    │
│  Unlocked: Nov 15, 2025            │← Unlock date
│  Times Drawn: 47                   │← Usage stats
│  Last Drawn: 2 days ago            │
│                                    │
│  "New beginnings, spontaneity..."  │← Snippet
│                                    │
│  [View Full Interpretation]        │← Button
│                                    │
│  ─── Unlock Progress ───           │
│  ✓ Common (Always Available)      │
│  ✓ Rare (Unlocked via Level 5)    │← How unlocked
│  ✓ Artifact (Completed Major      │
│     Arcana Achievement)            │
│                                    │
│  [Share Card] [Set as Favorite]   │← Actions
└────────────────────────────────────┘
```

---

## 🎯 UNLOCK MECHANICS DISPLAY

### **How Users Unlock Tiers:**

**Common Tier:**
- ✓ Always unlocked for all 78 cards
- Display: "Available to everyone (Public Domain RWS)"

**Rare Tier:**
- Unlock via gameplay:
  - Draw card X times (e.g., 5 draws)
  - Reach player level threshold (e.g., Level 5)
  - Complete readings (e.g., 10 total readings)
  - Purchase with Moonlight currency (500 ML)
  - Purchase with Veil Shards (50 shards)
- Display: Progress bar + requirement

**Artifact Tier (Major Arcana only):**
- Unlock via achievements:
  - Complete all Major Arcana in rare tier
  - Reach Level 20
  - Complete specific challenges
  - Purchase with Veil Shards (200 shards)
  - Subscription perk (all artifacts unlocked)
- Display: Achievement badge + requirement

---

## 🎨 VISUAL DESIGN SPECS

### **Card Grid Layout:**

**Mobile (Portrait):**
- 3 columns
- Card size: ~30% screen width
- Gap: 8px
- Scrollable vertical grid

**Tablet/iPad (Portrait):**
- 4-5 columns
- Card size: ~20% screen width
- Gap: 12px

**Landscape:**
- 6-8 columns
- Smaller cards, more visible

### **Card Animations:**

**Entry Animation (when screen loads):**
- Cards fade in sequentially
- Slight scale-up spring animation
- Stagger delay: 50ms between cards

**Unlock Animation (when new tier unlocked):**
- Card glows intensely
- Particle burst effect
- Scale up → wiggle → scale back
- Sound effect (mystical chime)

**Hover/Focus (on tap/long-press):**
- Card scales up slightly (1.05x)
- Glow intensifies
- Haptic feedback (light vibration)

---

## 🔊 SOUND DESIGN

**UI Sounds:**
- **Card Tap:** Soft card flip sound
- **Filter Select:** Subtle click
- **Rare Unlock:** Magical chime (medium pitch)
- **Artifact Unlock:** Epic orchestral hit
- **Scroll:** Soft whoosh (optional)

**Ambience:**
- Subtle mystical background hum
- Occasional wind chimes (very quiet)
- Crackle of distant fire (atmospheric)

---

## 📱 RESPONSIVE BEHAVIOR

### **Small Phones (iPhone SE, small Android):**
- 3-column grid
- Smaller card previews
- Simplified badges
- Essential filters only

### **Standard Phones (iPhone 14, Pixel):**
- 3-4 column grid (adjustable)
- Full feature set
- Smooth animations

### **Tablets/iPad:**
- 5-6 column grid
- Larger card previews
- Side-by-side card detail view (iPad landscape)
- Enhanced particle effects

---

## 🚀 IMPLEMENTATION CHECKLIST

### Phase 1: Basic Structure
- [ ] Create DeckViewerScreen component
- [ ] Implement card grid layout (responsive)
- [ ] Add filter buttons (All, Major, Minor)
- [ ] Add sort dropdown
- [ ] Integrate CardAssetLoader utility

### Phase 2: Card States
- [ ] Implement locked card display
- [ ] Implement common tier display
- [ ] Implement rare tier display (with glow)
- [ ] Implement artifact tier display (with video)
- [ ] Add tier badges to cards

### Phase 3: Interactions
- [ ] Card tap → detail view modal
- [ ] Filter functionality
- [ ] Sort functionality
- [ ] Unlock progress display
- [ ] Share card feature

### Phase 4: Animations & Polish
- [ ] Entry animations (staggered fade-in)
- [ ] Unlock celebration animations
- [ ] Glow/pulse animations for rare/artifact
- [ ] Particle effects
- [ ] Sound effects integration

### Phase 5: Performance
- [ ] Lazy loading for large collections
- [ ] Image caching
- [ ] Video optimization (artifact tier)
- [ ] Scroll performance tuning
- [ ] Memory management

---

## 💾 DATA STRUCTURE

### UserCollection (AsyncStorage/Database):

```javascript
{
  "userCollection": {
    "0": { // Card ID (The Fool)
      "unlockedTiers": ["common", "rare", "artifact"],
      "timesDrawn": 47,
      "firstDrawnAt": "2025-10-01T14:30:00Z",
      "lastDrawnAt": "2025-11-19T10:15:00Z",
      "lastTierDrawn": "artifact",
      "rareUnlockedAt": "2025-10-15T16:20:00Z",
      "rareUnlockMethod": "level_threshold",
      "artifactUnlockedAt": "2025-11-01T09:00:00Z",
      "artifactUnlockMethod": "achievement_major_arcana",
      "isFavorite": true
    },
    "1": { // Card ID (The Magician)
      "unlockedTiers": ["common", "rare"],
      "timesDrawn": 23,
      // ...
    }
  },
  "collectionStats": {
    "cardsSeen": 58,
    "cardsNeverDrawn": 20,
    "totalDraws": 342,
    "rareUnlocks": 45,
    "artifactUnlocks": 12,
    "lastUpdated": "2025-11-21T05:30:00Z"
  }
}
```

---

## 🎁 MONETIZATION INTEGRATION

### Free-to-Play Path:
- Common tier: Always available
- Rare tier: Unlock via gameplay (slow but achievable)
- Artifact tier: Achievement-gated (premium feel but earnable)

### Paid Shortcuts:
- "Unlock Rare Tier" button: 50 Veil Shards per card
- "Unlock All Rares for Suit" bundle: 500 Veil Shards
- "Unlock Artifact Tier" button: 200 Veil Shards (Major Arcana)
- "Unlock All Artifacts" pack: 3000 Veil Shards (whale bait)

### Subscription Benefits:
- All artifact tiers unlocked instantly
- Rare tier unlock requirements halved
- Exclusive animated card backs
- Priority unlock notifications

---

## 📊 SUCCESS METRICS

**Engagement:**
- Time spent in deck viewer
- Cards tapped per session
- Filter usage frequency

**Monetization:**
- Rare tier purchase rate
- Artifact tier purchase rate
- Subscription conversion from viewing artifacts

**Progression:**
- Average unlock speed
- Drop-off points (where users stop unlocking)
- Rarity distribution across user base

---

## 🎯 FINAL NOTES

**This deck viewer should feel like:**
- Opening a mystical collection vault
- Hearthstone's card collection quality
- Progress you can see and show off
- Worth grinding/paying for

**Key differentiators from competitors:**
- Three-tier rarity system (unique to Lunatiq)
- Animated artifact tier (Major Arcana only = special)
- Beautiful Dark Carnival Fae aesthetic
- Progression that respects free users but rewards payers

---

**Ready for implementation once UI assets are generated!**
