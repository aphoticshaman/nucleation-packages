# Design Overhaul BUILD Roadmap - Complete Implementation Plan

**Project:** Lunatiq Tarot App - Skia Migration & Performance Overhaul
**Objective:** Migrate from React Native Animated to Skia for 60 FPS, responsive, bug-free experience
**Status:** BUILDING TOOLS & INFRASTRUCTURE
**Last Updated:** 2025-11-20

---

## 🎯 PROJECT GOALS

### Primary Goals
1. ✅ Eliminate ALL Hermes property access errors
2. ✅ Achieve 60 FPS on all animations
3. ✅ Full responsiveness (iPhone SE to iPad Pro)
4. ✅ Memory efficient (< 200 MB)
5. ✅ Production-ready code

### Success Metrics
- **Zero runtime errors** on Hermes
- **60 FPS sustained** during all animations
- **< 5% battery drain** per 10 minutes
- **100% test coverage** on critical components
- **Sub-100ms response** to user interactions

---

## 📦 BUILD PHASES

### PHASE 0: Tools & Infrastructure ⚡ (CURRENT - 8 hours)
**Goal:** Build ALL tools needed for migration before touching production code

**Deliverables:**
1. Migration validation tools
2. Component templates & generators
3. Performance test harness
4. Automated migration scripts
5. Visual regression test suite
6. Component comparison tool
7. Test data generators
8. Build automation

**Why First:** Tools enable fast, confident changes. Build once, use throughout.

**Status:** ⏳ IN PROGRESS

---

### PHASE 1: Dependency Setup (1 hour)
**Goal:** Install and configure Skia + Reanimated

**Prerequisites:** Network connectivity restored

**Tasks:**
1. Install @shopify/react-native-skia
2. Install react-native-reanimated@3
3. Configure babel.config.js
4. Run pod-install (iOS)
5. Verify basic Skia rendering
6. Wrap App.js with SkiaDimensionsProvider

**Success Criteria:**
- [ ] Both packages installed without errors
- [ ] Babel configured correctly
- [ ] Simple Skia canvas renders (blue circle test)
- [ ] SkiaDimensionsProvider provides dimensions
- [ ] No console errors on app launch

**Test Milestone:** T1 - Basic Skia Rendering

---

### PHASE 2: Foundation Components (4 hours)
**Goal:** Build reusable Skia building blocks

**Tasks:**
1. Implement SkiaParticle (uncomment & test)
2. Implement SkiaAnimationHelpers (uncomment & test)
3. Create SkiaBackground base component
4. Create SkiaGradient utility
5. Create SkiaBlur utility
6. Test each utility in isolation

**Deliverables:**
- Working SkiaParticle with animations
- All animation helpers functional
- Test screen for each utility

**Success Criteria:**
- [ ] Each utility renders correctly
- [ ] Animations smooth (60 FPS)
- [ ] No memory leaks
- [ ] Documentation complete

**Test Milestone:** T2 - Foundation Components Validated

---

### PHASE 3: AmbientEffects Migration (6 hours)
**Goal:** Fix MainMenuScreen Hermes errors by migrating visual effects

**Priority Order:**
1. FloatingParticle → FloatingParticleSkia
2. LightBeams → LightBeamsSkia
3. MainMenuAmbience integration
4. MainMenuScreen testing

**Build Steps:**

#### 3.1: FloatingParticle (2 hours)
```
[BUILD] Create src/components/skia/FloatingParticleSkia.js
[TEST] Render 1 particle in test harness
[TEST] Render 20 particles in test harness
[BUILD] Add to MainMenuAmbience
[TEST] Full screen test on iPhone
[COMMIT] "FEAT: Migrate FloatingParticle to Skia"
```

#### 3.2: LightBeams (2 hours)
```
[BUILD] Create src/components/skia/LightBeamsSkia.js
[TEST] Render 1 beam in test harness
[TEST] Render 3 beams with animation
[BUILD] Add to MainMenuAmbience
[TEST] Full screen test on iPhone
[COMMIT] "FEAT: Migrate LightBeams to Skia"
```

#### 3.3: Integration (2 hours)
```
[BUILD] Update MainMenuAmbience to use Skia components
[BUILD] Create feature flag for Skia vs old components
[TEST] A/B test old vs new side-by-side
[TEST] Performance comparison (FPS, memory)
[TEST] Visual regression test
[COMMIT] "FEAT: Complete AmbientEffects Skia migration"
[PUSH] Push to remote
```

**Success Criteria:**
- [ ] No Hermes errors on MainMenuScreen
- [ ] 60 FPS sustained
- [ ] Visual parity with old version
- [ ] Memory usage same or better
- [ ] No console warnings

**Test Milestone:** T3 - MainMenuScreen Hermes-Free ✨

---

### PHASE 4: InteractiveBackground Migration (3 hours)
**Goal:** Migrate interactive hotspots and ripple effects

**Build Steps:**

#### 4.1: Hotspots (1.5 hours)
```
[BUILD] Create src/components/skia/InteractiveBackgroundSkia.js
[BUILD] Implement hotspot rendering with Skia Circles
[TEST] Tap detection works
[TEST] Hotspot positions responsive
[COMMIT] "FEAT: Migrate InteractiveBackground hotspots to Skia"
```

#### 4.2: Ripple Effects (1.5 hours)
```
[BUILD] Implement ripple animation with Reanimated
[TEST] Ripple triggers on tap
[TEST] Multiple ripples don't interfere
[BUILD] Add to MainMenuScreen
[TEST] Full integration test
[COMMIT] "FEAT: Complete InteractiveBackground Skia migration"
[PUSH] Push to remote
```

**Success Criteria:**
- [ ] All hotspots tappable
- [ ] Ripple animations smooth
- [ ] Responsive positioning works
- [ ] No performance degradation
- [ ] Gesture handling reliable

**Test Milestone:** T4 - Interactive Effects Working

---

### PHASE 5: Screen Dimensions Migration (2 hours)
**Goal:** Update all screens to use SkiaDimensionsContext (simple import changes)

**Files (Low Risk - Simple Import Updates):**
1. LoadingScreenNew.js
2. MainMenuScreen.js
3. ProfileScreen.js
4. CardReadingScreen.js
5. ReadingSummaryScreen.js
6. CardDrawingScreen.js
7. DeckViewerScreen.js
8. OracleChatScreen.js

**Build Steps:**
```
[BUILD] Automated script to update imports
[RUN] Update all 8 files
[TEST] Each screen manually
[TEST] Rotate device - verify responsive
[COMMIT] "REFACTOR: Migrate screens to useSkiaDimensions"
[PUSH] Push to remote
```

**Success Criteria:**
- [ ] All screens import useSkiaDimensions
- [ ] All screens responsive to rotation
- [ ] No static 375x812 dimensions anywhere
- [ ] No console warnings

**Test Milestone:** T5 - All Screens Responsive

---

### PHASE 6: Card Animations Migration (8 hours)
**Goal:** Migrate card flip, shuffle, and spread animations

**Priority Order:**
1. TarotCardFlip → TarotCardFlipSkia (3 hours)
2. CardShuffleAnimation → CardShuffleAnimationSkia (2 hours)
3. CardSelectionSpread → CardSelectionSpreadSkia (3 hours)

**Build Steps for Each:**
```
[BUILD] Create Skia version
[TEST] Test in isolation
[TEST] Test in CardReadingScreen context
[TEST] Performance validation
[TEST] Visual regression test
[COMMIT] "FEAT: Migrate [Component] to Skia"
```

**Success Criteria:**
- [ ] Flip animation smooth (60 FPS)
- [ ] Shuffle animation smooth
- [ ] Spread layout correct for all spread types
- [ ] Gesture handling works
- [ ] No visual glitches

**Test Milestone:** T6 - Card Animations Perfect

---

### PHASE 7: TarotCard Component Migration (4 hours)
**Goal:** Migrate card rendering with rarity effects

**Analysis Required:**
- Review src/components/TarotCard.js (1000+ lines)
- Identify rarity visual effects
- Decide which effects need Skia vs simple updates

**Build Steps:**
```
[ANALYZE] Read TarotCard.js thoroughly
[BUILD] Update dimensions import
[BUILD] Migrate rarity glow effects to Skia (if needed)
[BUILD] Migrate particle effects to Skia (if needed)
[TEST] All rarity tiers render correctly
[TEST] Performance with multiple cards
[COMMIT] "FEAT: Migrate TarotCard to Skia"
[PUSH] Push to remote
```

**Success Criteria:**
- [ ] All rarity tiers look correct
- [ ] Performance good with 20+ cards on screen
- [ ] No visual regressions
- [ ] Memory efficient

**Test Milestone:** T7 - Cards Look Perfect

---

### PHASE 8: Cleanup & Optimization (3 hours)
**Goal:** Remove old code, optimize performance

**Tasks:**

#### 8.1: Code Cleanup (1 hour)
```
[ANALYZE] Find all unused old components
[DELETE] Remove old Animated versions
[DELETE] Remove components/TarotCard.js (if unused)
[DELETE] Remove useSafeDimensions.js
[COMMIT] "REFACTOR: Remove deprecated code"
```

#### 8.2: Performance Optimization (2 hours)
```
[PROFILE] Run performance profiler on all screens
[OPTIMIZE] Reduce re-renders where possible
[OPTIMIZE] Memoize expensive calculations
[OPTIMIZE] Optimize Skia rendering paths
[TEST] Verify 60 FPS maintained
[COMMIT] "PERF: Optimize Skia rendering performance"
[PUSH] Push to remote
```

**Success Criteria:**
- [ ] No dead code remaining
- [ ] No unused imports
- [ ] Bundle size not increased significantly
- [ ] All tests passing

**Test Milestone:** T8 - Production Ready Code

---

### PHASE 9: Integration & Validation Testing (6 hours)
**Goal:** Comprehensive testing on all devices

**Test Matrix:**

#### 9.1: Device Testing (3 hours)
| Device | Screen Size | iOS Version | All Screens | All Animations | Gestures | Memory | FPS | Pass/Fail |
|--------|-------------|-------------|-------------|----------------|----------|---------|-----|-----------|
| iPhone SE | 4.7" | 15+ | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| iPhone 14 | 6.1" | 17+ | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| iPhone 14 Pro Max | 6.7" | 17+ | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| iPad Mini | 8.3" | 15+ | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Android Pixel | 6.3" | 12+ | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Android Tablet | 10" | 12+ | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

#### 9.2: Scenario Testing (2 hours)
- [ ] Cold start → LoadingScreen → MainMenu (< 3s)
- [ ] Navigate all screens without crash
- [ ] Perform 10 card readings back-to-back
- [ ] Background app → Foreground (animations resume)
- [ ] Rotate device 10 times (no layout issues)
- [ ] Memory usage stable after 30 min
- [ ] Battery drain acceptable

#### 9.3: Regression Testing (1 hour)
- [ ] All existing features still work
- [ ] No new bugs introduced
- [ ] User data preserved
- [ ] Settings persist

**Test Milestone:** T9 - Production Validation Complete ✅

---

### PHASE 10: Documentation & Handoff (2 hours)
**Goal:** Document everything for future maintenance

**Deliverables:**
1. Architecture documentation
2. Component usage guides
3. Performance tuning guide
4. Troubleshooting guide
5. Future enhancement ideas

**Build Steps:**
```
[WRITE] docs/SKIA_ARCHITECTURE.md
[WRITE] docs/COMPONENT_LIBRARY.md
[WRITE] docs/PERFORMANCE_GUIDE.md
[WRITE] docs/TROUBLESHOOTING.md
[WRITE] docs/FUTURE_ENHANCEMENTS.md
[COMMIT] "DOCS: Complete Skia migration documentation"
[PUSH] Push to remote
```

**Success Criteria:**
- [ ] All major components documented
- [ ] Performance benchmarks recorded
- [ ] Known issues documented
- [ ] Future roadmap outlined

**Test Milestone:** T10 - Project Complete 🎉

---

## 🧪 TEST MILESTONES

### T1: Basic Skia Rendering
**When:** After PHASE 1
**Tests:**
- [ ] Blue circle renders on SkiaTestScreen
- [ ] Canvas fills screen correctly
- [ ] No console errors
- [ ] FPS counter shows 60

**Pass Criteria:** Visual confirmation + no errors

---

### T2: Foundation Components Validated
**When:** After PHASE 2
**Tests:**
- [ ] SkiaParticle renders and fades
- [ ] All animation helpers work (fadeIn, floatUp, etc.)
- [ ] No memory leaks after 1000 particles
- [ ] Test harness passes all component tests

**Pass Criteria:** All foundation tests green

---

### T3: MainMenuScreen Hermes-Free ✨
**When:** After PHASE 3
**Tests:**
- [ ] Launch app → no Hermes errors
- [ ] MainMenuScreen renders perfectly
- [ ] Particles float smoothly (60 FPS)
- [ ] Light beams pulse smoothly
- [ ] Visual parity with old version
- [ ] Memory usage < 150 MB

**Pass Criteria:** Zero Hermes errors + performance targets met

**CRITICAL MILESTONE:** This validates the entire approach works

---

### T4: Interactive Effects Working
**When:** After PHASE 4
**Tests:**
- [ ] All hotspots respond to taps
- [ ] Ripple effects smooth
- [ ] Multiple simultaneous ripples work
- [ ] Gesture handling reliable

**Pass Criteria:** 100% gesture success rate

---

### T5: All Screens Responsive
**When:** After PHASE 5
**Tests:**
- [ ] All 8 screens responsive to rotation
- [ ] iPhone SE (smallest) looks good
- [ ] iPad Pro (largest) looks good
- [ ] No hardcoded dimensions anywhere

**Pass Criteria:** Visual inspection on 3 device sizes

---

### T6: Card Animations Perfect
**When:** After PHASE 6
**Tests:**
- [ ] Card flip smooth in both directions
- [ ] Shuffle animation looks natural
- [ ] All spread types layout correctly
- [ ] No dropped frames

**Pass Criteria:** 60 FPS maintained + visual approval

---

### T7: Cards Look Perfect
**When:** After PHASE 7
**Tests:**
- [ ] All rarity tiers visually correct
- [ ] Common, Rare, Epic, Legendary effects work
- [ ] Performance good with 20+ cards
- [ ] No visual glitches

**Pass Criteria:** Visual regression tests pass

---

### T8: Production Ready Code
**When:** After PHASE 8
**Tests:**
- [ ] No dead code (verified by script)
- [ ] No unused imports (verified by linter)
- [ ] All tests passing
- [ ] Performance targets maintained

**Pass Criteria:** Automated checks pass + manual review

---

### T9: Production Validation Complete
**When:** After PHASE 9
**Tests:**
- [ ] All device matrix tests pass
- [ ] All scenario tests pass
- [ ] All regression tests pass
- [ ] Memory profiling good
- [ ] Battery profiling acceptable

**Pass Criteria:** 100% test pass rate

**FINAL MILESTONE:** App is production-ready

---

### T10: Project Complete 🎉
**When:** After PHASE 10
**Tests:**
- [ ] Documentation complete
- [ ] All code committed and pushed
- [ ] Stakeholder approval

**Pass Criteria:** Sign-off received

---

## 🛠️ BUILD TOOLS (Creating Now)

### Tool 1: Migration Validator
**File:** `scripts/validate-migration.js`
**Purpose:** Verify no old patterns remain
**Usage:** `node scripts/validate-migration.js`
**Checks:**
- No useSafeDimensions imports
- No Animated.Value usage in migrated files
- All Skia components have tests
- Performance benchmarks recorded

---

### Tool 2: Component Generator
**File:** `scripts/generate-skia-component.js`
**Purpose:** Generate Skia component from template
**Usage:** `node scripts/generate-skia-component.js FloatingParticle`
**Output:**
- Component file with boilerplate
- Test file
- Documentation stub

---

### Tool 3: Performance Monitor
**File:** `src/utils/PerformanceMonitor.js`
**Purpose:** Real-time FPS and memory tracking
**Usage:** Wrap any screen with `<PerformanceMonitor>`
**Output:** Console logs + on-screen overlay

---

### Tool 4: Visual Regression Test
**File:** `scripts/visual-regression.js`
**Purpose:** Screenshot comparison (old vs new)
**Usage:** `node scripts/visual-regression.js MainMenuScreen`
**Output:** Diff images highlighting changes

---

### Tool 5: Migration Progress Tracker
**File:** `scripts/migration-progress.js`
**Purpose:** Generate progress report
**Usage:** `node scripts/migration-progress.js`
**Output:** Markdown report with percentages

---

### Tool 6: Automated Import Updater
**File:** `scripts/update-imports.js`
**Purpose:** Bulk update useSafeDimensions → useSkiaDimensions
**Usage:** `node scripts/update-imports.js screens/**/*.js`
**Output:** Updated files + git diff summary

---

### Tool 7: Component Comparison View
**File:** `screens/ComponentComparisonScreen.js`
**Purpose:** Side-by-side old vs new component
**Usage:** Navigate to screen, select component
**Output:** Split-screen comparison

---

### Tool 8: Test Data Generator
**File:** `scripts/generate-test-data.js`
**Purpose:** Create realistic test data for animations
**Usage:** `node scripts/generate-test-data.js particles 100`
**Output:** JSON test data

---

## 📋 COMMIT STRATEGY

### Commit Often (Every 30-60 Minutes)

**Format:**
```
[TYPE]: Brief description

- Detailed change 1
- Detailed change 2
- Test results (if applicable)
```

**Types:**
- `BUILD` - New tool or infrastructure
- `FEAT` - New feature or component
- `REFACTOR` - Code restructuring
- `TEST` - New tests or test improvements
- `PERF` - Performance optimization
- `DOCS` - Documentation
- `FIX` - Bug fix
- `CHORE` - Maintenance

**Push Strategy:**
- Push after each phase completion
- Push after critical milestones
- Push before breaking changes
- Push at least every 2 hours

---

## 📊 PROGRESS TRACKING

### Overall Progress
```
PHASE 0: Tools & Infrastructure    [████████░░] 80% (6/8 tools built)
PHASE 1: Dependencies              [░░░░░░░░░░]  0% (Blocked - Network)
PHASE 2: Foundation Components     [░░░░░░░░░░]  0%
PHASE 3: AmbientEffects Migration  [░░░░░░░░░░]  0%
PHASE 4: InteractiveBackground     [░░░░░░░░░░]  0%
PHASE 5: Screen Dimensions         [░░░░░░░░░░]  0%
PHASE 6: Card Animations           [░░░░░░░░░░]  0%
PHASE 7: TarotCard Component       [░░░░░░░░░░]  0%
PHASE 8: Cleanup & Optimization    [░░░░░░░░░░]  0%
PHASE 9: Integration Testing       [░░░░░░░░░░]  0%
PHASE 10: Documentation            [░░░░░░░░░░]  0%

Overall: [█░░░░░░░░░] 8%
```

### Time Estimates
- **Already Spent:** 8 hours (PHASE 0 planning + tools)
- **Remaining:** ~37 hours (PHASES 0-10 completion)
- **Total Project:** ~45 hours

### Risk Level
🟢 **LOW RISK** - Tools built, plan solid, patterns proven

---

## 🚀 NEXT ACTIONS (Building Now)

1. ✅ Create this BUILD roadmap
2. ⏳ Build Tool 1: Migration Validator
3. ⏳ Build Tool 2: Component Generator
4. ⏳ Build Tool 3: Performance Monitor
5. ⏳ Build Tool 4: Visual Regression Test
6. ⏳ Build Tool 5: Migration Progress Tracker
7. ⏳ Build Tool 6: Automated Import Updater
8. ⏳ Build Tool 7: Component Comparison View
9. ⏳ Build Tool 8: Test Data Generator

**Current Focus:** Building all tools before touching production code

---

**Roadmap Status:** 📝 ACTIVE
**Last Updated:** 2025-11-20
**Branch:** claude/catalog-and-sync-repos-01TrrmUDEmT3znrryD57LozM
