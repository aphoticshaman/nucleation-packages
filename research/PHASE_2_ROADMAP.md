# PHASE 2 ROADMAP: Core Data & Economy
**Dark Fantasy Tarot Foundation**

## Overview
Phase 2 establishes the fundamental data structures and economy system for VeilPath Tarot's dark fantasy vision. This phase creates the foundation upon which all gameplay, aesthetics, and monetization will be built.

---

## Phase 2 Objectives
1. ✅ **Complete 78-Card Tarot Database** - Canonical data for all cards
2. ✅ **Placeholder Card Artwork** - Temporary visuals until Midjourney art is ready
3. ✅ **Veil Shard Economy** - Currency system for dark fantasy RPG mechanics

---

## 2.1: Complete Tarot Card Database

### Goal
Create a comprehensive, canonical database of all 78 tarot cards with:
- Traditional meanings (upright & reversed)
- Dark fantasy flavor adaptations
- Suit associations (Wands/Fire, Cups/Water, Swords/Air, Pentacles/Earth)
- Astrological/elemental correspondences
- Keywords for interpretation engine

### Structure
```javascript
// src/data/cardDatabase.js
export const CARD_DATABASE = {
  0: { // The Fool
    name: "The Fool",
    arcana: "Major",
    number: 0,
    element: "Air",
    astrology: "Uranus",
    suit: null,

    keywords: {
      upright: ["new beginnings", "innocence", "spontaneity", "leap of faith"],
      reversed: ["recklessness", "naivety", "foolishness", "fear of unknown"]
    },

    meanings: {
      upright: "The Fool represents new beginnings, innocence, and the courage to step into the unknown...",
      reversed: "Reversed, The Fool warns against reckless decisions and naive assumptions..."
    },

    darkFantasyFlavor: {
      theme: "The Wanderer at the Veil's Edge",
      description: "A hooded figure stands at the precipice of a moonlit chasm, one foot suspended in air...",
      questConnection: "First steps into the Shadowlands",
      npcReference: "The Fool archetype mirrors Luna's guidance for new seekers"
    }
  },
  // ... 77 more cards
};
```

### Implementation Tasks
- [x] Research traditional tarot meanings (RWS system)
- [ ] Create JSON schema for card data structure
- [ ] Write Major Arcana (0-21): 22 cards
- [ ] Write Minor Arcana Wands (1-14): 14 cards
- [ ] Write Minor Arcana Cups (1-14): 14 cards
- [ ] Write Minor Arcana Swords (1-14): 14 cards
- [ ] Write Minor Arcana Pentacles (1-14): 14 cards
- [ ] Add dark fantasy flavor text to all cards
- [ ] Validate database completeness (78 total)
- [ ] Create database validation script

### Tools Needed
- **Card Database Generator** (Python script)
  - Input: CSV with traditional meanings
  - Output: JavaScript object with full structure
  - Features: Validation, dark fantasy flavor injection

### Time Estimate
**6-8 hours**
- Major Arcana: 2-3 hours
- Minor Arcana (4 suits): 3-4 hours
- Dark fantasy adaptation: 1-2 hours

---

## 2.2: Placeholder Card Artwork

### Goal
Generate simple, atmospheric placeholder images for all 78 cards until professional Midjourney art is commissioned in Phase 5.

### Requirements
- **Consistent style** across all cards
- **Suit-coded colors** (Wands/red, Cups/blue, Swords/cyan, Pentacles/green, Major/violet)
- **Card name & number** clearly visible
- **Dark fantasy aesthetic** (moonlit, gothic, mystical)
- **1080x1920 portrait aspect** (standard tarot ratio)

### Approaches (Pick One)
**Option A: ASCII Art Placeholders**
```
╔══════════════════════════════╗
║                              ║
║         THE FOOL             ║
║            0                 ║
║                              ║
║          🌙                  ║
║        /  |  \               ║
║       *   |   *              ║
║           |                  ║
║                              ║
║    NEW BEGINNINGS            ║
║                              ║
╚══════════════════════════════╝
```

**Option B: Python-Generated Gradient Cards** ⭐ RECOMMENDED
- Background: Gradient (suit-specific colors)
- Card name: Elegant serif font
- Symbolic icon: Simple SVG shapes
- Border: Ornate frame
- Moon phase indicator: Top corner

**Option C: Figma Template Batch Export**
- Design one template in Figma
- Auto-populate 78 variants with text variables
- Export all at once

### Implementation Tasks
- [ ] Choose placeholder approach
- [ ] Create card art generator script (Python/Pillow)
- [ ] Generate Major Arcana (22 cards)
- [ ] Generate Minor Arcana Wands (14 cards)
- [ ] Generate Minor Arcana Cups (14 cards)
- [ ] Generate Minor Arcana Swords (14 cards)
- [ ] Generate Minor Arcana Pentacles (14 cards)
- [ ] Save to `assets/cards/` directory
- [ ] Update card loading system to use new images
- [ ] Test card display in app

### Tools Needed
- **Card Art Generator** (Python/Pillow)
  ```python
  # tools/generate_card_placeholders.py
  # Generates 78 cards with suit-coded gradients
  ```

### Time Estimate
**3-4 hours**
- Generator script: 1-2 hours
- Batch generation: 30 minutes
- Testing & integration: 1 hour

---

## 2.3: Veil Shard Currency System

### Goal
Implement the foundational economy system for VeilPath Tarot's RPG mechanics:
- **Veil Shards** (premium currency, purchased with real money)
- **Moonlight** (soft currency, earned through gameplay)
- Earning, spending, and balancing mechanics

### Currency Design

#### Veil Shards (💎 Hard Currency)
- **Earned via:** In-app purchases, rare achievements, daily login streaks
- **Spent on:**
  - Unlock premium spreads (Celtic Cross, Horseshoe, etc.)
  - Cosmetic card backs
  - Voice narration (Luna/Sol)
  - NPC consultations (special guided readings)
- **Pricing:** $0.99 = 100 shards, $4.99 = 600 shards, $9.99 = 1,400 shards
- **Non-refillable:** Cannot be earned easily (premium scarcity)

#### Moonlight (🌙 Soft Currency)
- **Earned via:**
  - Complete readings (10 Moonlight)
  - Daily login (25 Moonlight)
  - Achievements (50-200 Moonlight)
  - Quest completion (100-500 Moonlight)
- **Spent on:**
  - Basic spreads (3-card, 5-card)
  - Card unlocks (deck viewer)
  - Minor cosmetics
- **Refillable:** Abundant, encourages engagement

### Implementation Tasks
- [ ] Create `CurrencyManager` utility
- [ ] Add currency storage (AsyncStorage)
- [ ] Implement earn currency functions
- [ ] Implement spend currency functions
- [ ] Add currency display UI component
- [ ] Create currency transaction log
- [ ] Add IAP integration hooks (Phase 7)
- [ ] Balance economy (cost/earn rates)
- [ ] Add currency animations (sparkle on earn)

### Data Structure
```javascript
// User currency state
{
  veilShards: 250,
  moonlight: 1840,
  lifetime: {
    veilShardsEarned: 250,
    veilShardsSpent: 0,
    moonlightEarned: 2340,
    moonlightSpent: 500
  },
  transactions: [
    { type: 'earn', currency: 'moonlight', amount: 25, reason: 'daily_login', timestamp: 1700000000 },
    // ...
  ]
}
```

### Tools Needed
- **Economy Balancing Spreadsheet** (Google Sheets)
  - Model earn/spend rates
  - Calculate time-to-unlock for features
  - Ensure F2P progression isn't too slow

### Time Estimate
**4-5 hours**
- CurrencyManager: 2 hours
- UI components: 1-2 hours
- Testing & balancing: 1-2 hours

---

## Success Criteria

### Phase 2 Complete When:
- ✅ All 78 cards exist in `cardDatabase.js` with complete data
- ✅ All 78 placeholder images exist in `assets/cards/`
- ✅ Cards display correctly in app (DeckViewer, readings)
- ✅ Currency system functional (earn, spend, persist)
- ✅ Currency UI displays on main menu
- ✅ No console errors when drawing cards
- ✅ Database passes validation script

---

## Phase 2 Deliverables

### Code Files
- `src/data/cardDatabase.js` - Complete 78-card database
- `src/utils/CurrencyManager.js` - Economy system
- `src/components/CurrencyDisplay.js` - UI for currency
- `assets/cards/*.png` - 78 placeholder card images
- `tools/generate_card_placeholders.py` - Card art generator
- `tools/validate_card_database.js` - Database validator

### Documentation
- `docs/CARD_DATABASE_SCHEMA.md` - Database structure reference
- `docs/ECONOMY_DESIGN.md` - Currency system design doc
- `docs/PHASE_2_TESTING_GUIDE.md` - Testing checklist

---

## Dependencies

### Before Phase 2 Starts
- ✅ Phase 1 complete (cleanup done)
- ✅ Import errors fixed
- ✅ App runs without crashes

### Blocks Phase 3
Phase 2 must be complete before Phase 3 (LLM integration) because:
- Interpretation engine needs card database
- Economy gates access to premium features
- Placeholder art needed for testing flow

---

## Timeline

### Week 1 (Phase 2.1)
- Day 1-2: Card database research & structure
- Day 3-4: Major Arcana + Wands/Cups
- Day 5: Swords/Pentacles + validation

### Week 2 (Phase 2.2 + 2.3)
- Day 1-2: Card art generator + batch creation
- Day 3-4: Currency system implementation
- Day 5: Testing, balancing, documentation

**Total: 10-12 days** (assuming 2-3 hours/day)

---

## Next Steps

1. **Immediate:** Create card database schema
2. **Next:** Build card database generator tool
3. **Then:** Generate Major Arcana (proof of concept)
4. **After:** Complete Minor Arcana (batch work)
5. **Finally:** Placeholder art + currency system

---

## Notes

- **Dark Fantasy Flavor:** Every card should reference the Shadowlands, Luna/Sol, or the Veil
- **Future-Proofing:** Database structure should support future expansions (custom decks, card variants)
- **Localization Ready:** Structure supports i18n for future translations
- **Performance:** Keep database small (<500KB total) for fast loading

---

**Status:** 📋 Planning Complete → Ready for Implementation

**Last Updated:** 2025-11-20
