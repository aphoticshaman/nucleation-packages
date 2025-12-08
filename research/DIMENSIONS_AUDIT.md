# Dimensions Usage Audit - Complete Inventory

**Date:** 2025-11-20
**Purpose:** Catalog all files using `useSafeDimensions` and `useWindowDimensions` for Skia migration
**Status:** PHASE 0 - Ready for migration when dependencies installed

---

## Summary

**Total Files:** 18 (including docs, test files, and implementation files)
**Implementation Files Requiring Migration:** 14
**Utility/Infrastructure Files:** 2
**Documentation/Test Files:** 2

---

## Current State: useSafeDimensions.js

**File:** `src/utils/useSafeDimensions.js`

**Current Implementation:**
```javascript
export function useSafeDimensions() {
  // TODO: Hermes is broken - just return static dimensions until we can debug properly
  // const dimensions = useWindowDimensions();

  // TEMPORARY: Return iPhone standard dimensions
  // This will work but won't be responsive to different screen sizes
  return { width: 375, height: 812 };
}
```

**Status:** ⏸️ TEMPORARY - Returns static dimensions
**Issue:** Not responsive to screen size changes
**Migration Plan:** Replace with `useSkiaDimensions()` from `SkiaDimensionsContext`

---

## Files by Priority

### 🔴 CRITICAL PRIORITY - Visual Effects (Hermes Error Source)

These files caused the original Hermes property access errors and must be migrated first.

#### 1. src/components/AmbientEffects.js

**Lines:** 14, 216
**Usage Pattern:**
```javascript
// FloatingParticle component (line 14)
const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = useSafeDimensions();

// LightBeams component (line 216)
const { width, height } = useSafeDimensions();
```

**What it does:**
- FloatingParticle: Creates floating particle effects on MainMenuScreen
- LightBeams: Creates animated light beam effects

**Why critical:**
- Imported by MainMenuAmbience
- Identified as root cause of Hermes errors via binary search

**Migration Strategy:**
1. Create `src/components/skia/FloatingParticleSkia.js` with Skia Circle rendering
2. Create `src/components/skia/LightBeamsSkia.js` with Skia Path/LinearGradient
3. Update AmbientEffects.js to export both old and new versions
4. Update MainMenuScreen to use Skia versions
5. Test on iPhone

**Estimated Effort:** 3 hours

---

#### 2. src/components/InteractiveBackground.js

**Lines:** Usage of useSafeDimensions for hotspot positioning

**Usage Pattern:**
```javascript
const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = useSafeDimensions();

const hotspots = [
  { id: 'moon', x: SCREEN_WIDTH * 0.15, y: SCREEN_HEIGHT * 0.12, size: 60 },
  // ... more hotspots with percentage-based positioning
];
```

**What it does:**
- Interactive hotspots (moon, stars, crystal) on MainMenuScreen
- Ripple effects on touch
- Percentage-based positioning for responsive layout

**Why critical:**
- Imported by MainMenuScreen
- Part of the Hermes error chain

**Migration Strategy:**
1. Create `src/components/skia/InteractiveBackgroundSkia.js`
2. Use Skia Canvas with gesture handling
3. Render hotspots as Skia Circles
4. Animate ripples with Reanimated shared values

**Estimated Effort:** 2 hours

---

### 🟡 HIGH PRIORITY - Main Screens

These screens are user-facing and should be responsive.

#### 3. screens/MainMenuScreen.js

**Lines:** 10, 18
**Usage Pattern:**
```javascript
import { useSafeDimensions } from '../src/utils/useSafeDimensions';

const { width, height } = useSafeDimensions();
```

**What it does:**
- App home screen
- Displays title, buttons, ambient effects
- Entry point after loading

**Dependencies:**
- Imports AmbientEffects (MainMenuAmbience)
- Imports InteractiveBackground

**Migration Strategy:**
1. Update import to `useSkiaDimensions` from context
2. Ensure SkiaDimensionsProvider wraps app in App.js
3. Test after AmbientEffects and InteractiveBackground migrated

**Estimated Effort:** 30 minutes

---

#### 4. screens/LoadingScreenNew.js

**Lines:** 9, 16
**Usage Pattern:**
```javascript
import { useSafeDimensions } from '../src/utils/useSafeDimensions';

const { width, height } = useSafeDimensions();
```

**What it does:**
- App loading/splash screen
- Logo animation
- Progress indicator
- Navigation to MainMenuScreen

**Migration Strategy:**
1. Update import to `useSkiaDimensions`
2. Simple substitution - no complex effects

**Estimated Effort:** 15 minutes

---

#### 5. screens/CardReadingScreen.js

**Lines:** 17, 19, 22, 34
**Usage Pattern:**
```javascript
import { useSafeDimensions } from '../src/utils/useSafeDimensions';

const { width } = useSafeDimensions();
```

**What it does:**
- Card reading flow orchestration
- Card selection, interpretation, reflection
- Uses width for layout calculations

**Migration Strategy:**
1. Update import to `useSkiaDimensions`
2. Test card reading flow

**Estimated Effort:** 20 minutes

---

#### 6. screens/ProfileScreen.js

**Lines:** 13, 15, 18, 21
**Usage Pattern:**
```javascript
import { useSafeDimensions } from '../src/utils/useSafeDimensions';

const { width } = useSafeDimensions();
```

**What it does:**
- User profile stats display
- Total readings, favorite suit, etc.

**Migration Strategy:**
1. Update import to `useSkiaDimensions`
2. Simple substitution

**Estimated Effort:** 15 minutes

---

#### 7. screens/ReadingSummaryScreen.js

**Lines:** 16, 18, 21, 32
**Usage Pattern:**
```javascript
import { useSafeDimensions } from '../src/utils/useSafeDimensions';

const { width, height } = useSafeDimensions();
const CARD_WIDTH = width * 0.35;
const CARD_HEIGHT = CARD_WIDTH * 1.5;
```

**What it does:**
- Displays reading results
- Card layout and visualization
- Summary text

**Migration Strategy:**
1. Update import to `useSkiaDimensions`
2. Test card sizing calculations

**Estimated Effort:** 20 minutes

---

### 🟡 HIGH PRIORITY - Animation Components

These components handle card animations and visual effects.

#### 8. src/components/CardShuffleAnimation.js

**Usage:** useSafeDimensions for card positioning during shuffle

**What it does:**
- Animates cards shuffling before draw
- Random positions and rotations

**Migration Strategy:**
1. Create `src/components/skia/CardShuffleAnimationSkia.js`
2. Use Skia Image rendering with Reanimated transforms
3. Replace React Animated with Reanimated worklets

**Estimated Effort:** 2 hours

---

#### 9. src/components/TarotCardFlip.js

**Usage:** useSafeDimensions for card dimensions

**What it does:**
- 3D flip animation for revealing cards
- Front/back card images

**Migration Strategy:**
1. Create `src/components/skia/TarotCardFlipSkia.js`
2. Simulate 3D flip with 2D Skia transforms (scale + rotation)
3. Swap images at 90° rotation point

**Estimated Effort:** 2 hours

---

#### 10. src/components/CardSelectionSpread.js

**Usage:** useSafeDimensions for card spread layout

**What it does:**
- Displays cards in various spread patterns
- Celtic Cross, Three Card, etc.

**Migration Strategy:**
1. Update import to `useSkiaDimensions`
2. May need Skia version for better performance with many cards

**Estimated Effort:** 1 hour

---

### 🟢 MEDIUM PRIORITY - Feature Screens

#### 11. src/screens/CardDrawingScreen.js

**Lines:** 6, 7
**Usage Pattern:**
```javascript
import { useSafeDimensions } from '../src/utils/useSafeDimensions';
```

**What it does:**
- Handles card drawing mechanics
- Visual effects during draw

**Migration Strategy:**
1. Update import to `useSkiaDimensions`

**Estimated Effort:** 15 minutes

---

#### 12. src/screens/DeckViewerScreen.js

**Lines:** 15, 18, 32
**Usage Pattern:**
```javascript
import { useSafeDimensions } from '../src/utils/useSafeDimensions';

const { width: SCREEN_WIDTH } = useSafeDimensions();
const CARD_WIDTH = (SCREEN_WIDTH - 60) / 3; // 3 cards per row
```

**What it does:**
- Displays card collection
- Grid layout of all cards
- Shows locked/unlocked cards

**Migration Strategy:**
1. Update import to `useSkiaDimensions`
2. Test grid layout responsiveness

**Estimated Effort:** 20 minutes

---

#### 13. src/screens/OracleChatScreen.js

**Usage:** useSafeDimensions for chat UI layout

**What it does:**
- AI oracle chat interface
- Message list and input

**Migration Strategy:**
1. Update import to `useSkiaDimensions`

**Estimated Effort:** 15 minutes

---

### 🟢 MEDIUM PRIORITY - Card Components

#### 14. components/TarotCard.js (OLD VERSION)

**Lines:** 2, 3, 16, 18
**Usage Pattern:**
```javascript
import { useSafeDimensions } from '../src/utils/useSafeDimensions';

const dimensions = useSafeDimensions();
const screenWidth = (dimensions && typeof dimensions.width === 'number') ? dimensions.width : 375;
```

**What it does:**
- Old tarot card rendering component
- May be deprecated

**Status:** ⚠️ VERIFY IF STILL IN USE

**Migration Strategy:**
1. Check if any files import this version
2. If unused, DELETE during cleanup phase
3. If used, update to useSkiaDimensions OR migrate callers to new version

**Estimated Effort:** 30 minutes (investigation + decision)

---

#### 15. src/components/TarotCard.js (NEW VERSION)

**Usage:** Likely uses useSafeDimensions for rarity-based card system

**What it does:**
- New tarot card component with rarity system
- Visual effects for different rarity tiers
- Probably 1000+ lines

**Migration Strategy:**
1. Update import to `useSkiaDimensions`
2. May need Skia version for rarity visual effects (glow, particles, etc.)

**Estimated Effort:** 2 hours (large file, complex effects)

---

### ✅ INFRASTRUCTURE - Already Complete or No Action Needed

#### 16. src/contexts/SkiaDimensionsContext.js

**Lines:** 4, 7, 9, 42, 51, 53, 56
**Status:** ✅ INFRASTRUCTURE - Uses useWindowDimensions internally (SAFE)

**What it does:**
- Context provider for dimensions
- Wraps useWindowDimensions in React component context
- Safe because it's component-level, not module-level

**Action:** None - this is the solution, not the problem

---

#### 17. src/utils/useSafeDimensions.js

**Status:** 🔄 TO BE REPLACED

**Action:** After all migrations complete, replace with deprecation wrapper:
```javascript
export function useSafeDimensions() {
  console.warn('useSafeDimensions is deprecated. Use useSkiaDimensions from SkiaDimensionsContext.');
  return useSkiaDimensions();
}
```

Then eventually DELETE this file entirely.

---

#### 18. screens/SkiaTestHarnessScreen.js

**Lines:** 103
**Status:** ✅ TEST INFRASTRUCTURE

**What it does:**
- Test harness for migration validation
- Already uses useSkiaDimensions correctly

**Action:** None - this tests the solution

---

#### 19. docs/SKIA_MIGRATION_PLAN.md

**Status:** ✅ DOCUMENTATION

**Action:** None - documentation only

---

## Migration Checklist

### PHASE 1: Setup (BLOCKED - Network Issue)

- [ ] Install @shopify/react-native-skia
- [ ] Install react-native-reanimated@3
- [ ] Configure babel.config.js with Reanimated plugin
- [ ] Wrap App.js with SkiaDimensionsProvider

### PHASE 2: Critical Visual Effects

- [ ] Migrate AmbientEffects.FloatingParticle to Skia
- [ ] Migrate AmbientEffects.LightBeams to Skia
- [ ] Migrate InteractiveBackground to Skia
- [ ] Test MainMenuScreen with Skia effects
- [ ] Verify no Hermes errors on iPhone

### PHASE 3: Main Screens (Simple Import Updates)

- [ ] Update LoadingScreenNew.js → useSkiaDimensions
- [ ] Update MainMenuScreen.js → useSkiaDimensions
- [ ] Update ProfileScreen.js → useSkiaDimensions
- [ ] Update CardReadingScreen.js → useSkiaDimensions
- [ ] Update ReadingSummaryScreen.js → useSkiaDimensions

### PHASE 4: Animation Components

- [ ] Migrate CardShuffleAnimation to Skia
- [ ] Migrate TarotCardFlip to Skia
- [ ] Update CardSelectionSpread → useSkiaDimensions

### PHASE 5: Feature Screens (Simple Import Updates)

- [ ] Update CardDrawingScreen.js → useSkiaDimensions
- [ ] Update DeckViewerScreen.js → useSkiaDimensions
- [ ] Update OracleChatScreen.js → useSkiaDimensions

### PHASE 6: Card Components

- [ ] Investigate components/TarotCard.js (old) - DELETE if unused
- [ ] Update src/components/TarotCard.js (new) → useSkiaDimensions
- [ ] Consider Skia version for rarity visual effects

### PHASE 7: Cleanup

- [ ] Replace useSafeDimensions.js with deprecation wrapper
- [ ] Run comprehensive grep to verify all migrations
- [ ] Test on iPhone (multiple screen sizes)
- [ ] Test on Android
- [ ] Remove deprecation wrapper
- [ ] DELETE useSafeDimensions.js
- [ ] DELETE components/TarotCard.js (old) if confirmed unused

### PHASE 8: Validation

- [ ] Run SkiaTestHarnessScreen tests
- [ ] Navigate through all screens
- [ ] Test all animations
- [ ] Verify responsive behavior (rotate device)
- [ ] Performance profiling (60 FPS target)
- [ ] No Hermes errors in console

---

## Migration Patterns

### Pattern 1: Simple Import Update (Non-Animation Components)

**Before:**
```javascript
import { useSafeDimensions } from '../src/utils/useSafeDimensions';

function MyComponent() {
  const { width, height } = useSafeDimensions();
  // ... use width and height
}
```

**After:**
```javascript
import { useSkiaDimensions } from '../src/contexts/SkiaDimensionsContext';

function MyComponent() {
  const { width, height } = useSkiaDimensions();
  // ... use width and height
}
```

**Files Using This Pattern:**
- LoadingScreenNew.js
- MainMenuScreen.js (after AmbientEffects migrated)
- ProfileScreen.js
- CardReadingScreen.js
- ReadingSummaryScreen.js
- CardDrawingScreen.js
- DeckViewerScreen.js
- OracleChatScreen.js

---

### Pattern 2: Full Skia Migration (Visual Effect Components)

**Before:**
```javascript
import { View, Animated } from 'react-native';
import { useSafeDimensions } from '../utils/useSafeDimensions';

function FloatingParticle() {
  const { width, height } = useSafeDimensions();
  const translateY = useRef(new Animated.Value(0)).current;

  // Animated.timing...

  return (
    <Animated.View style={{ transform: [{ translateY }] }}>
      <View style={{ width: 5, height: 5, backgroundColor: 'cyan' }} />
    </Animated.View>
  );
}
```

**After:**
```javascript
import { Circle, useValue } from '@shopify/react-native-skia';
import { useSharedValue, withTiming } from 'react-native-reanimated';
import { useSkiaDimensions } from '../../contexts/SkiaDimensionsContext';

function FloatingParticleSkia() {
  const { width, height } = useSkiaDimensions();
  const translateY = useSharedValue(0);

  // withTiming...

  return (
    <Circle
      cx={100}
      cy={translateY}
      r={2.5}
      color="cyan"
    />
  );
}
```

**Files Using This Pattern:**
- AmbientEffects.js (FloatingParticle, LightBeams)
- InteractiveBackground.js
- CardShuffleAnimation.js
- TarotCardFlip.js
- Possibly: TarotCard.js (new) for rarity effects

---

### Pattern 3: Hybrid Approach (Skia + React Native)

For components that need both canvas effects and React Native UI:

```javascript
import { View } from 'react-native';
import { Canvas, Circle } from '@shopify/react-native-skia';
import { useSkiaDimensions } from '../contexts/SkiaDimensionsContext';

function HybridComponent() {
  const { width, height } = useSkiaDimensions();

  return (
    <View style={{ flex: 1 }}>
      {/* Skia canvas for visual effects */}
      <Canvas style={{ position: 'absolute', width, height }}>
        <Circle cx={width / 2} cy={height / 2} r={50} color="cyan" />
      </Canvas>

      {/* React Native UI on top */}
      <View style={{ padding: 20 }}>
        {/* buttons, text, etc. */}
      </View>
    </View>
  );
}
```

**Files Likely Using This Pattern:**
- MainMenuScreen.js (Skia effects + React Native buttons)
- InteractiveBackground.js (Skia canvas + gesture handling)

---

## Risk Assessment

### Low Risk (Simple Import Update)

These files only need import path changes. No logic changes.

- LoadingScreenNew.js ✅
- ProfileScreen.js ✅
- CardDrawingScreen.js ✅
- DeckViewerScreen.js ✅
- OracleChatScreen.js ✅

**Estimated Time:** 1.5 hours total

---

### Medium Risk (Logic + Visual Effects)

These files need testing but follow established patterns.

- MainMenuScreen.js ⚠️
- CardReadingScreen.js ⚠️
- ReadingSummaryScreen.js ⚠️
- CardSelectionSpread.js ⚠️

**Estimated Time:** 2 hours total

---

### High Risk (Complex Migration)

These require significant refactoring with Skia.

- AmbientEffects.js (FloatingParticle, LightBeams) 🔴
- InteractiveBackground.js 🔴
- CardShuffleAnimation.js 🔴
- TarotCardFlip.js 🔴
- TarotCard.js (new version) 🔴

**Estimated Time:** 11 hours total

---

## Total Effort Estimate

**Setup:** 1 hour
**Low Risk Migrations:** 1.5 hours
**Medium Risk Migrations:** 2 hours
**High Risk Migrations:** 11 hours
**Testing & Validation:** 4 hours
**Documentation & Cleanup:** 2 hours

**Grand Total:** ~21.5 hours

---

## Next Steps (When Network Restored)

1. ✅ Run `npm install @shopify/react-native-skia react-native-reanimated@3`
2. ✅ Add SkiaDimensionsProvider to App.js
3. ✅ Start with AmbientEffects.FloatingParticle (highest risk, highest impact)
4. ✅ Test immediately after each migration
5. ✅ Commit frequently
6. ✅ Follow the checklist above

---

## Questions for User

1. **Priority:** Should we prioritize getting MainMenuScreen working first (high-risk migrations), or knock out all the low-risk migrations first for quick wins?

2. **TarotCard.js:** The old version in `components/TarotCard.js` appears to be a duplicate. Should we delete it during cleanup, or is it still used somewhere?

3. **Rarity Effects:** The new TarotCard.js (1000+ lines) likely has fancy rarity visual effects. Should these be migrated to Skia, or are simple dimension updates sufficient?

4. **Testing:** Do you have access to multiple iOS devices (different screen sizes) for testing? Or should we prioritize iPhone 14 Pro (6.1") dimensions?

---

**Document Status:** Complete audit - ready for migration
**Last Updated:** 2025-11-20
**Branch:** `claude/catalog-and-sync-repos-01TrrmUDEmT3znrryD57LozM`
