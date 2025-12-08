# Comprehensive Codebase Audit
**Date: 2025-11-20**
**Purpose: Identify what exists vs what's designed, find dead code, assess crash risks**

---

## Executive Summary

**CRITICAL FINDING: Two parallel codebases exist**

1. **NEW codebase** (root level): 4 screens, 2 components - Dark fantasy tarot MVP
2. **OLD codebase** (src/): 24 screens, 27 components, 57 utilities - Cyberpunk AGI Oracle version

**Current State:** App likely crashes or shows old UI because App.js navigation may reference old/missing files.

**Designed State:** 9 comprehensive design documents describe dark fantasy RPG tarot game with:
- Veil Shard economy
- NPC reputation system
- Subscription model ($3.99-4.99/month)
- 100+ easter eggs
- Suit-specific color palettes
- Atmospheric lighting
- BookTok romance

**Reality Gap:** 99% of designed features are not implemented.

---

## File Inventory

### NEW Files (Dark Fantasy MVP - Root Level)

**Screens (screens/):**
1. `LoadingScreenNew.js` (7.2KB) - Mystical loading with "VeilPath TAROT" branding
2. `MainMenuScreen.js` (5.7KB) - Main menu after loading
3. `IsometricVillageScreen.js` (9.7KB) - Village navigation
4. `OracleShopScreen.js` (11KB) - Oracle interaction/readings

**Components (components/):**
1. `TarotCard.js` (5.7KB) - Tarot card display component
2. `CardInterpretationDialog.js` (6.4KB) - Card interpretation UI

**Total NEW code:** ~45KB, 6 files

### OLD Files (Cyberpunk AGI Oracle - src/)

**Screens (src/screens/) - 24 files:**
1. AccessibilityScreen.js
2. AchievementsScreen.js
3. BirthdayScreen.js
4. CardDrawingScreen.js
5. CardInterpretationScreen.js
6. CareerCounselorScreen.js
7. DeckViewerScreen.js
8. InsightsScreen.js
9. IntentionScreen.js
10. IsometricVillageScreen.js (DUPLICATE - conflicts with new one!)
11. JournalScreen.js
12. LoadingScreen.js (OLD loading screen)
13. MBTITestScreen.js (Myers-Briggs personality test)
14. OnboardingScreen.js (OLD onboarding flow)
15. OracleChatScreen.js
16. PersonalGrowthScreen.js
17. PersonalityQuestionsScreen.js
18. PrivacyPolicyScreen.js
19. ProfileDetailScreen.js
20. ProfileImportScreen.js
21. ProfileSelectScreen.js
22. ProfileSetupScreen.js
23. ReadingHistoryScreen.js
24. ReadingScreen.js
25. ReadingTypeScreen.js
26. SemanticVisualizerScreen.js
27. SettingsScreen.js
28. StatsScreen.js
29. SynthesisScreen.js
30. WelcomeScreen.js (OLD welcome screen)

**Components (src/components/) - 27 files:**
1. AchievementUnlock.js
2. AnimatedASCIIText.js
3. AnticipationOverlay.js
4. AuthenticationScreen.js
5. BadgeDisplay.js
6. CardSelectionSpread.js
7. CustomerCenter.js
8. CyberpunkCard.js (CYBERPUNK aesthetic!)
9. CyberpunkHeader.js (CYBERPUNK aesthetic!)
10. EncryptedTextReveal.js
11. ErrorBoundary.js
12. ExitPrompt.js
13. FlippableCard.js
14. GlitchParticles.js (CYBERPUNK aesthetic!)
15. GuideSelector.js
16. MCQModal.js
17. OptimizedMatrixRain.js (CYBERPUNK aesthetic!)
18. PacManLoader.js
19. Paywall.js
20. PermissionExplanationModal.js
21. TarotCardFlip.js
22. TerminalEffects.js (CYBERPUNK aesthetic!)
23. TipJarButton.js
24. VoiceControls.js

**Utilities (src/utils/) - 57 files:**
1. AdManager.js
2. InAppPurchaseManager.js
3. PermissionsManager.js
4. UpgradePromptManager.js
5. accessibilityManager.js
6. actionTracker.js
7. advancedAstrology.js
8. agiInterpretation.js (AGI! This is the old vision)
9. apiEnrichment.js
10. asciiCardArt.js
11. astrology.js
12. balancedWisdom.js
13. cardSynergyMatrix.js
14. chineseZodiac.js
15. coldReadingEnhancer.js
16. dataBackup.js
17. deepAGI.js (AGI!)
18. diverseWisdom.js
19. featureGate.js
20. geometricSemanticSpace.js
21. insightsPipeline.js
22. intentionValidator.js
23. lazyLLM.js
24. llmCardInterpretation.js
25. llmCostTracker.js
26. llmEnhancement.js
27. llmWorker.js
28. localLLMEnhancer.js
29. mbtiTest.js (Myers-Briggs)
30. megaSynthesisEngine.js
31. metaPromptAnalyzer.js
32. narrativeArcComposer.js
33. oracleMemory.js
34. oraclePersonality.js
35. oracleThinkingMessages.js
36. poeticInterpretation.js
37. postCardQuestions.js
38. profileBackup.js
39. profileSecurity.js
40. quantumNarrativeEngine.js
41. quantumRNG.js
42. readingFeedback.js
43. readingHistory.js
44. readingStorage.js
45. scenarioQuestions.js
46. semanticEntityManager.js
47. synthesisCache.js
48. synthesisHelpers.js
49. temporalPredictions.js
50. tokenEstimator.js
51. varietyEngine.js
52. varietyEngineCompressed.js
53. varietyEngineMassive.js

**Total OLD code:** ~108 files (massive!)

---

## Design Documents (docs/)

**Implemented (exist in code):**
- None directly match new designs

**Created but not implemented:**
1. CLEANUP_AND_OPTIMIZATION_PLAN.md - Plan to archive old code
2. DARK_CARNIVAL_AESTHETIC.md - Soul wagers, Faustian bargains (700 lines)
3. EASTER_EGGS_CATALOG.md - 100+ pop culture references (1033 lines)
4. EDEMA_RUH_AESTHETIC_GUIDE.md - Romani/traveling aesthetic (440 lines)
5. ESOTERIC_PHILOSOPHY_CORE.md - Sacred exchange philosophy (450 lines)
6. ETHICAL_MONETIZATION.md - $50/month cap system (577 lines)
7. LLM_COST_CONTROL_ECONOMICS.md - Game economy gates (550 lines)
8. RPG_PROGRESSION_SYSTEM.md - Shards, reputation, quests (650 lines)
9. SET_AND_SETTING_SYSTEM.md - Shamanic preparation (500 lines)
10. SPOTIFY_PLAYLIST_EASTER_EGGS.md - 30 music references (just created)

**OLD design documents (from previous vision):**
1. 3D_NAVIGATOR_LEARNING_PROMPT.md
2. AUDIO_INTEGRATION_EXAMPLES.md
3. CARD_DATABASE_ARCHITECTURE.md
4. CYBERPUNK_EFFECTS_GUIDE.md (CYBERPUNK!)
5. DEEP_AGI_GUIDE.md (AGI vision)
6. DUAL_VERSION_STRATEGY.md
7. LLM_CUSTOMIZATION_GUIDE.md
8. LUNA_SOL_ARCHITECTURE.md
9. OPTIMIZATION_GUIDE.md
10. PALANTIR_VISION.md (The "Palantir" concept)
11. PERSONALIZATION_SYSTEM.md
12. ULTIMATE_ORACLE_STACK.md
13. VOICE_SYSTEM_GUIDE.md
14. Multiple skills docs (ANIMATION_RIGGING, GAME_ASSET_ARTIST, etc.)

**Total:** ~25 design docs, ~10 new (dark fantasy), ~15 old (cyberpunk/AGI)

---

## Asset Inventory

**Icons/App branding:**
- ✅ icon.png (NEW - 1.2MB) - Dark fantasy moon + tarot card
- ✅ adaptive-icon.png (NEW - 1.2MB) - Android version
- ✅ icon_preview_with_guides.png - Safe zone preview
- ❌ OLD icons (archived/overwritten)

**Art assets (assets/art/):**

**Tiles (assets/art/tiles/):**
- 16 isometric pixel art ground tiles (PNG)
- For village/isometric view
- Midjourney generated

**Procedural (assets/art/procedural/):**
- 15+ particle sheets, animations, tiles
- Buttons: start, continue, menu, settings, draw_card
- Sacred geometry tiles
- Circuit patterns (cyberpunk aesthetic!)
- Mandala patterns
- Particle animation GIFs (5 total)

**Effects (assets/art/effects/):**
- 3 magical particle sprite sheets
- 2 holographic aura overlays (GIF)
- Purple/cyan aesthetic

**Icons (assets/art/icons/):**
- Fantasy RPG skill icons
- Semantic category icons (emotion, heart)
- Moon/shadow work themed

**Misc:**
- 1 large animation GIF (motion loop)

**Sounds (assets/sounds/):**
- Directory exists but likely empty or placeholder
- Docs exist: AUDIO_ASSETS_GUIDE.md, PLACEHOLDER_INSTRUCTIONS.md

**Total assets:** ~50+ image files, mostly Midjourney generated

---

## Styles Created (New Aesthetic System)

**Implemented:**
1. ✅ `styles/colors.js` - Suit-specific color palettes
2. ✅ `styles/backgroundStyles.js` - Black backgrounds, vignettes
3. `styles/silhouetteEffects.js` - Silhouettes with glows
4. ✅ `styles/atmosphericLighting.js` - Twilight skies, candlelight

**Status:** Created but not yet applied to screens

---

## App.js Navigation Analysis

**Current App.js navigation** (need to read to confirm):
- Likely still references old screens
- May cause crashes if old screens imported but paths wrong

**Should be:**
```javascript
<Stack.Navigator initialRouteName="LoadingNew">
  <Stack.Screen name="LoadingNew" component={LoadingScreenNew} />
  <Stack.Screen name="MainMenu" component={MainMenuScreen} />
  <Stack.Screen name="Village" component={IsometricVillageScreen} />
  <Stack.Screen name="OracleShop" component={OracleShopScreen} />
</Stack.Navigator>
```

---

## Dependencies (package.json)

**Key dependencies likely used:**
- React Native
- Expo
- React Navigation
- Firebase (auth, firestore)
- AsyncStorage
- Linear Gradient
- Others TBD

**Unused dependencies (from old vision):**
- Likely many AGI/LLM related packages
- Need to audit package.json

---

## Dead Code Analysis

### Definitely Dead (Old Cyberpunk/AGI Vision):

**Screens to archive (src/screens/):**
1. WelcomeScreen.js - OLD welcome flow
2. OnboardingScreen.js - OLD onboarding
3. LoadingScreen.js - OLD loading (replaced by LoadingScreenNew.js)
4. MBTITestScreen.js - Myers-Briggs personality test (not in new vision)
5. PersonalityQuestionsScreen.js - Personality profiling (not in new vision)
6. BirthdayScreen.js - Birthday input (not in new vision)
7. ProfileSetupScreen.js - Profile creation (may need simplified version)
8. ProfileSelectScreen.js - Multi-profile system (not in new vision)
9. ProfileDetailScreen.js - Profile details (not in new vision)
10. ProfileImportScreen.js - Profile import (not in new vision)
11. SemanticVisualizerScreen.js - Cyberpunk semantic visualization
12. SynthesisScreen.js - AGI synthesis system
13. InsightsScreen.js - AGI insights
14. CareerCounselorScreen.js - Career advice feature
15. PersonalGrowthScreen.js - Personal growth tracking

**Components to archive (src/components/):**
1. CyberpunkCard.js - Cyberpunk aesthetic
2. CyberpunkHeader.js - Cyberpunk aesthetic
3. OptimizedMatrixRain.js - Matrix rain effect
4. GlitchParticles.js - Glitch effects
5. TerminalEffects.js - Terminal aesthetics
6. EncryptedTextReveal.js - Cyberpunk text effects
7. AnimatedASCIIText.js - ASCII art animations
8. PacManLoader.js - Pac-Man loading animation
9. AuthenticationScreen.js - May need replacement
10. GuideSelector.js - OLD guide selection UI

**Utilities to archive (src/utils/):**
1. deepAGI.js - AGI interpretation system
2. agiInterpretation.js - AGI system
3. megaSynthesisEngine.js - AGI synthesis
4. quantumNarrativeEngine.js - Narrative AGI
5. metaPromptAnalyzer.js - Prompt analysis
6. narrativeArcComposer.js - Story composition
7. geometricSemanticSpace.js - Semantic space visualization
8. semanticEntityManager.js - Entity management
9. advancedAstrology.js - Advanced astrology (may keep simplified)
10. chineseZodiac.js - Chinese zodiac (may keep as easter egg)
11. coldReadingEnhancer.js - Cold reading techniques
12. diverseWisdom.js - Multiple wisdom sources
13. balancedWisdom.js - Wisdom balancing
14. poeticInterpretation.js - Poetic interpretations
15. mbtiTest.js - Myers-Briggs test
16. varietyEngine.js (all 3 versions) - Variety generation
17. synthesisHelpers.js - Synthesis utilities
18. synthesisCache.js - Synthesis caching
19. tokenEstimator.js - Token estimation (may keep for LLM cost control)
20. oracleThinkingMessages.js - Oracle "thinking" messages
21. postCardQuestions.js - Post-card questions
22. scenarioQuestions.js - Scenario-based questions
23. temporalPredictions.js - Time-based predictions

### Maybe Keep (Potentially Useful):

**Screens:**
1. OracleChatScreen.js - Oracle chat (may adapt for Luna dialog)
2. CardDrawingScreen.js - Card drawing mechanics
3. CardInterpretationScreen.js - Interpretation display
4. DeckViewerScreen.js - View all cards
5. JournalScreen.js - Journal entries (RPG quest log?)
6. ReadingHistoryScreen.js - Past readings
7. ReadingScreen.js - Reading flow
8. ReadingTypeScreen.js - Choose reading type
9. SettingsScreen.js - Settings
10. StatsScreen.js - Statistics (RPG stats?)
11. AchievementsScreen.js - Achievements (matches new design!)
12. PrivacyPolicyScreen.js - Legal requirement

**Components:**
1. ErrorBoundary.js - Error handling (always useful)
2. Paywall.js - Subscription paywall (needed!)
3. TipJarButton.js - Tipping (ethical monetization)
4. FlippableCard.js - Card flip animation
5. TarotCardFlip.js - Tarot-specific flip
6. CardSelectionSpread.js - Spread selection
7. VoiceControls.js - Voice input (accessibility)
8. ExitPrompt.js - Exit confirmation
9. AchievementUnlock.js - Achievement notifications
10. BadgeDisplay.js - Badge/achievement display
11. PermissionExplanationModal.js - Permission requests
12. MCQModal.js - Multiple choice modal

**Utilities:**
1. llmCardInterpretation.js - LLM for readings (NEEDED!)
2. llmEnhancement.js - LLM enhancement
3. llmCostTracker.js - LLM cost tracking (NEEDED for economics!)
4. llmWorker.js - LLM worker thread
5. lazyLLM.js - Lazy loading LLM
6. localLLMEnhancer.js - Local LLM fallback
7. oracleMemory.js - Oracle remembers past readings (NEEDED!)
8. oraclePersonality.js - Oracle personality (Luna character!)
9. readingHistory.js - Reading storage (NEEDED!)
10. readingStorage.js - Storage utilities
11. readingFeedback.js - User feedback on readings
12. cardSynergyMatrix.js - Card combination meanings
13. intentionValidator.js - Validate user intentions
14. insightsPipeline.js - Generate insights
15. profileBackup.js - Backup user data
16. profileSecurity.js - Secure profiles
17. dataBackup.js - Data backup
18. accessibilityManager.js - Accessibility features
19. actionTracker.js - Analytics
20. featureGate.js - Feature flags
21. asciiCardArt.js - ASCII card art (maybe cute easter egg?)
22. astrology.js - Basic astrology
23. quantumRNG.js - Random number generation
24. InAppPurchaseManager.js - IAP (NEEDED!)
25. PermissionsManager.js - Permissions
26. UpgradePromptManager.js - Upgrade prompts
27. AdManager.js - Ads (may not use)

---

## Potential Crash Points

### HIGH RISK:

1. **Duplicate Screen Names:**
   - `src/screens/IsometricVillageScreen.js` vs `screens/IsometricVillageScreen.js`
   - Which one does App.js import?

2. **Missing Imports in App.js:**
   - If App.js imports from `./src/screens/` but we're using `./screens/`, crash

3. **Missing Assets:**
   - Tarot card images (78 cards) - NOT YET GENERATED
   - NPC sprites - NOT CREATED
   - Background images - LIMITED
   - UI buttons - SOME EXIST (procedural/buttons)

4. **Missing Data:**
   - Tarot card database (78 cards with meanings)
   - NPC data (dialog, quests, reputation)
   - Quest chains
   - Achievement definitions

5. **LLM Integration:**
   - No API keys configured?
   - No LLM prompts implemented
   - Cost tracking not connected

6. **Navigation Flow:**
   - User completes loading → goes where?
   - MainMenu has what options?
   - Village navigation works?
   - Oracle interaction functional?

### MEDIUM RISK:

1. **AsyncStorage schema changes:**
   - Old data format vs new data format

2. **Firebase configuration:**
   - Auth setup
   - Firestore collections

3. **Styling conflicts:**
   - Old cyberpunk styles vs new dark fantasy styles
   - Color system changes

4. **Missing environment variables:**
   - API keys
   - Firebase config
   - Feature flags

### LOW RISK:

1. **Performance:**
   - Large asset files
   - Unoptimized images
   - Memory leaks in old code

---

## Gap Analysis: Design vs Reality

### Designed (Docs) but NOT Implemented:

**RPG Systems (0% complete):**
- ❌ Veil Shard currency system
- ❌ Daily earning caps (100 shards/day)
- ❌ NPC reputation (0-100 scale)
- ❌ Quest system
- ❌ Quest chains
- ❌ Currency costs for readings
- ❌ Cooldown timers
- ❌ Achievement system (structure exists in old code, needs adaptation)

**NPCs (0% complete):**
- ❌ Luna (Oracle) - character, dialog, personality
- ❌ Kael (Minstrel) - cute-core character
- ❌ Rosalind (Cottage Witch)
- ❌ Vesper (Fire-Eater)
- ❌ Seraphine (Contortionist)
- ❌ Dorian (Ringmaster)

**Reading Types (0% complete):**
- ❌ Single card (10 shards)
- ❌ Three-card (25 shards)
- ❌ Celtic Cross (50 shards)
- ❌ Year Wheel (100 shards)
- ❌ Soul Reading (200 shards, 30-day cooldown)

**Subscription System (0% complete):**
- ❌ Free tier (2 readings/day, 3-card max)
- ❌ Subscription tier ($3.99-4.99/month)
- ❌ Feature gates
- ❌ Revenue integration

**Easter Eggs (0% complete):**
- ❌ 100+ pop culture references catalogued, none implemented
- ❌ Poe, Anne Rice, Witcher, Hunter S. Thompson, etc.
- ❌ Music references (30 catalogued)
- ❌ BookTok references
- ❌ Hidden locations, items, quests

**Aesthetic Systems (10% complete):**
- ✅ Color palette system created (styles/colors.js)
- ❌ NOT applied to screens yet
- ❌ Atmospheric lighting created but not used
- ❌ Silhouette effects created but not used
- ❌ Particle systems (some assets exist, not integrated)
- ❌ Twilight skies
- ❌ Candlelight/campfire effects

**Art Assets (5% complete):**
- ❌ 78 tarot cards (0/78)
- ❌ 6 NPC sprites (0/6)
- ❌ Village backgrounds (partial - have tiles)
- ❌ Oracle shop interior (0/1)
- ❌ UI frames/borders (0)
- ✅ Some particle effects exist
- ✅ Some buttons exist (procedural folder)
- ✅ App icon created

**LLM Integration (0% complete):**
- ❌ Oracle personality prompts
- ❌ Luna character voice
- ❌ Reading generation
- ❌ Set and setting assessment
- ❌ Sacred exchange philosophy in prompts
- ❌ Cost tracking integration
- ❌ Cached response system

**Set & Setting System (0% complete):**
- ❌ Mental state assessment
- ❌ Emotional state check
- ❌ Spiritual readiness
- ❌ Environment adaptation
- ❌ Ritual container creation

**Monetization (0% complete):**
- ❌ Veil Shard purchase bundles
- ❌ Premium crystals (decided against? - "use the shards")
- ❌ Subscription flow
- ❌ Spending tracking
- ❌ Revenue analytics

---

## What Actually Works Right Now?

**Best guess (needs testing):**

✅ **Probably works:**
- App loads (if environment setup correct)
- Shows LoadingScreenNew
- Basic navigation structure

❓ **Unclear:**
- Does it crash immediately?
- Can you navigate past loading screen?
- Do tarot cards display (no images yet)?
- Does Oracle interaction work?

❌ **Definitely doesn't work:**
- Tarot readings (no LLM integration)
- Card images (not generated)
- RPG systems (not built)
- Subscription (not implemented)
- NPCs (don't exist)
- Quests (don't exist)
- Currency (doesn't exist)
- Most designed features

---

## Recommendations

### IMMEDIATE (Fix Crashes):

1. **Audit App.js:**
   - Check all imports
   - Ensure navigation references correct files
   - Remove dead imports

2. **Archive old code:**
   - Move src/ to src_archived/
   - Ensure no imports break

3. **Create tarot card database:**
   - 78 cards with basic data
   - Placeholder images (solid colors with text)
   - Get app minimally functional

4. **Test basic flow:**
   - Loading → Main Menu → Village → Oracle
   - Fix crashes one by one

### SHORT TERM (MVP Functionality):

1. **Generate tarot card art:**
   - Midjourney prompts for 78 cards
   - Process through Python scripts
   - Integrate into app

2. **Build basic LLM integration:**
   - Single card reading only
   - Simple prompt (no fancy Luna personality yet)
   - Basic interpretation

3. **Implement free tier gates:**
   - 2 readings/day
   - AsyncStorage tracking
   - Simple counter

4. **Apply new aesthetic:**
   - Update all screens to use colors.js
   - Add black backgrounds
   - Basic atmospheric effects

### MEDIUM TERM (Core Features):

1. **RPG progression:**
   - Veil Shards
   - Basic quests (3-5)
   - NPC reputation (Luna only)

2. **Subscription system:**
   - Paywall component
   - IAP integration
   - Feature unlocking

3. **Multiple reading types:**
   - Three-card spread
   - Celtic Cross
   - Different shard costs

4. **Luna character:**
   - Personality system
   - Dialog variations
   - Memory of past readings

### LONG TERM (Full Vision):

1. **All 6 NPCs**
2. **100+ easter eggs**
3. **Full quest chains**
4. **Seasonal events**
5. **All reading types**
6. **Set & setting system**
7. **Achievement system**
8. **BookTok integrations**

---

## Files to Archive Immediately

**Create: src_archived/ folder**

**Move these 15 screens (dead code):**
1. src/screens/WelcomeScreen.js
2. src/screens/OnboardingScreen.js
3. src/screens/LoadingScreen.js
4. src/screens/MBTITestScreen.js
5. src/screens/PersonalityQuestionsScreen.js
6. src/screens/BirthdayScreen.js
7. src/screens/ProfileSetupScreen.js
8. src/screens/ProfileSelectScreen.js
9. src/screens/ProfileDetailScreen.js
10. src/screens/ProfileImportScreen.js
11. src/screens/SemanticVisualizerScreen.js
12. src/screens/SynthesisScreen.js
13. src/screens/InsightsScreen.js
14. src/screens/CareerCounselorScreen.js
15. src/screens/PersonalGrowthScreen.js

**Move these 10 components (cyberpunk aesthetics):**
1. src/components/CyberpunkCard.js
2. src/components/CyberpunkHeader.js
3. src/components/OptimizedMatrixRain.js
4. src/components/GlitchParticles.js
5. src/components/TerminalEffects.js
6. src/components/EncryptedTextReveal.js
7. src/components/AnimatedASCIIText.js
8. src/components/PacManLoader.js
9. src/components/GuideSelector.js
10. src/components/AnticipationOverlay.js

**Move these 23 utilities (AGI/old vision):**
1. src/utils/deepAGI.js
2. src/utils/agiInterpretation.js
3. src/utils/megaSynthesisEngine.js
4. src/utils/quantumNarrativeEngine.js
5. src/utils/metaPromptAnalyzer.js
6. src/utils/narrativeArcComposer.js
7. src/utils/geometricSemanticSpace.js
8. src/utils/semanticEntityManager.js
9. src/utils/advancedAstrology.js
10. src/utils/chineseZodiac.js
11. src/utils/coldReadingEnhancer.js
12. src/utils/diverseWisdom.js
13. src/utils/balancedWisdom.js
14. src/utils/poeticInterpretation.js
15. src/utils/mbtiTest.js
16. src/utils/varietyEngine.js
17. src/utils/varietyEngineCompressed.js
18. src/utils/varietyEngineMassive.js
19. src/utils/synthesisHelpers.js
20. src/utils/synthesisCache.js
21. src/utils/oracleThinkingMessages.js
22. src/utils/postCardQuestions.js
23. src/utils/scenarioQuestions.js
24. src/utils/temporalPredictions.js

**Total to archive: ~48 files**

---

## Critical Questions to Answer

1. **Does the app run without crashing?**
   - Need to test actual startup
   - Check error logs

2. **What does App.js actually import?**
   - Need to read App.js fully
   - Check navigation config

3. **Are tarot card images expected to exist?**
   - Check if components try to load images
   - Will crash if missing?

4. **Is LLM configured?**
   - API keys?
   - Fallback behavior if not?

5. **What happens when user clicks Oracle?**
   - Does reading flow exist?
   - What breaks first?

---

## Summary

**Current State: 1% Complete**

- App structure exists (6 new files)
- Design vision complete (9 docs, ~4500 lines)
- Art assets: 5% (icon done, cards missing)
- Code implementation: <1% (navigation only)
- Old codebase: 108 files of cruft causing conflicts

**To reach MVP (playable demo):**

1. Archive old code (48 files) - 2 hours
2. Fix App.js navigation - 1 hour
3. Generate 78 tarot cards - 8-12 hours (Midjourney + processing)
4. Build tarot database - 4 hours
5. Implement basic LLM reading - 6 hours
6. Apply new aesthetic to 4 screens - 4 hours
7. Add free tier gates (2/day limit) - 2 hours
8. Test and debug - 4-8 hours

**Total to MVP: ~30-40 hours of focused work**

**User was right:** "i'm betting it crashes and looks nothing like you're saying or my brain is picturing"

**Reality:** Massive design vision, minimal implementation, high crash risk from dual codebases.
