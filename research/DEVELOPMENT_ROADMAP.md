# VeilPath Development Roadmap & Technical Plan

**Mental WELLNESS Platform - Production Implementation Guide**

**Date**: November 22, 2025
**Target Launch**: Q2 2026 (6 months)
**Methodology**: Backwards planning from launch + XYZA execution framework

---

## Executive Summary

VeilPath is a **premium mental wellness platform** disguised as a tarot app, integrating evidence-based CBT, DBT, and mindfulness practices with engaging gamification. This roadmap transforms 3,700+ lines of therapeutic content data into a production-ready React Native application.

**Key Differentiators**:
- ✅ Custom dark fantasy 78-card tarot deck (517 assets)
- ✅ Evidence-based therapy integration (CBT/DBT/Mindfulness)
- ✅ 5-level gamification (XP, achievements, skill trees, prestige)
- ✅ 625+ therapeutic journal prompts
- ✅ 28 guided mindfulness practices
- ✅ Premium positioning ($9.99/month vs. $2.99 competitors)

**Success Metrics**:
- Apple App Store approval (addressing prior rejection)
- 10,000 downloads in first 3 months
- 15% free-to-paid conversion
- 4.5+ star rating
- 70% 30-day retention

---

## Table of Contents

1. [Technical Stack](#technical-stack)
2. [Architecture](#architecture)
3. [Sprint Plan (12 Sprints × 2 Weeks)](#sprint-plan)
4. [Phase 1: Foundation (Sprints 1-3)](#phase-1-foundation)
5. [Phase 2: Core Features (Sprints 4-6)](#phase-2-core-features)
6. [Phase 3: Gamification (Sprints 7-9)](#phase-3-gamification)
7. [Phase 4: Polish & Launch (Sprints 10-12)](#phase-4-polish--launch)
8. [Testing Strategy](#testing-strategy)
9. [Performance Benchmarks](#performance-benchmarks)
10. [App Store Submission](#app-store-submission)
11. [Post-Launch](#post-launch)

---

## Technical Stack

### Frontend
- **React Native**: 0.81.5 (Expo SDK 54)
- **React**: 19.1.0 (after Hermes compatibility fix)
- **Navigation**: React Navigation v7 (with Dimensions patch)
- **State Management**: Zustand (lightweight, performant)
- **Styling**: Styled Components + React Native StyleSheet
- **Animations**: React Native Reanimated 3 + Lottie

### Backend (Serverless)
- **Database**: Supabase (PostgreSQL + real-time + auth)
- **AI**: Anthropic Claude API (optional journal analysis)
- **Storage**: Supabase Storage (cloud backup, encrypted)
- **Analytics**: PostHog (privacy-first analytics)

### Local Storage
- **Primary**: AsyncStorage (user data, settings)
- **Encrypted**: expo-secure-store (sensitive data)
- **File System**: expo-file-system (assets, cache)

### Development Tools
- **TypeScript**: Full type safety
- **ESLint**: Code quality
- **Prettier**: Code formatting
- **Jest**: Unit testing
- **Detox**: E2E testing
- **Sentry**: Error tracking

### CI/CD
- **GitHub Actions**: Automated testing + builds
- **EAS Build**: Expo Application Services
- **EAS Submit**: Automated App Store submission
- **Fastlane**: iOS/Android deployment automation

---

## Architecture

### App Structure
```
src/
├── screens/               # 20+ screens
│   ├── onboarding/
│   ├── readings/
│   ├── journal/
│   ├── mindfulness/
│   ├── profile/
│   └── gamification/
├── components/            # Reusable UI components
│   ├── cards/            # Card rendering, animations
│   ├── buttons/          # Custom buttons with assets
│   ├── modals/           # Dialog boxes, overlays
│   ├── gamification/     # XP bars, achievements, skill trees
│   └── mindfulness/      # Breathing animations, timers
├── services/             # Business logic
│   ├── cardEngine.ts     # Card drawing, shuffling, spreads
│   ├── journalService.ts # Journal CRUD, analysis
│   ├── gamificationService.ts # XP, achievements, progression
│   ├── therapyService.ts # CBT/DBT/Mindfulness content delivery
│   └── aiService.ts      # Claude API integration (opt-in)
├── stores/               # Zustand state management
│   ├── userStore.ts      # User profile, settings, preferences
│   ├── progressStore.ts  # XP, level, achievements, streak
│   ├── readingStore.ts   # Current reading, history
│   └── journalStore.ts   # Journal entries, tags
├── data/                 # Static content (JSON)
│   ├── therapy/          # CBT, DBT, mindfulness, prompts
│   ├── gamification/     # Achievements, skill trees, XP
│   └── cards/            # Card meanings, interpretations
├── assets/               # 517 image/audio files
├── hooks/                # Custom React hooks
├── utils/                # Helper functions
├── types/                # TypeScript types
└── navigation/           # Navigation config
```

### Data Architecture

**Local-First** (Privacy-first design):
```
User Device (Primary)
├── AsyncStorage (unencrypted)
│   ├── User preferences
│   ├── App settings
│   ├── Reading history
│   └── Journal entries (encrypted at rest)
├── SecureStore (encrypted)
│   ├── User API keys (if custom Claude key)
│   └── Sensitive preferences
└── FileSystem
    ├── Cached assets
    └── Exported data (JSON backups)

Optional Cloud Sync (Opt-in)
└── Supabase
    ├── Encrypted journal backup
    ├── Cross-device sync
    └── Premium features (AI analysis)
```

**Privacy Architecture**:
- Default: Everything local, encrypted
- Opt-in: Cloud backup (end-to-end encrypted)
- Opt-in: AI analysis (ephemeral, not stored)
- No telemetry without consent
- Full data export anytime
- One-click data deletion

---

## Sprint Plan (12 Sprints × 2 Weeks)

### Phase 1: Foundation (Weeks 1-6)
| Sprint | Focus | Deliverables |
|--------|-------|-------------|
| 1 | Project Setup + Navigation | App shell, navigation, screens structure |
| 2 | Data Layer + Card Engine | JSON loading, card drawing logic, spread algorithms |
| 3 | UI Component Library | Reusable components, theming, asset integration |

### Phase 2: Core Features (Weeks 7-12)
| Sprint | Focus | Deliverables |
|--------|-------|-------------|
| 4 | Reading Experience | Card selection, spread layouts, flip animations |
| 5 | Journal System | Entry creation, prompts, CBT integration |
| 6 | Mindfulness Practices | 28 guided practices, timers, tracking |

### Phase 3: Gamification (Weeks 13-18)
| Sprint | Focus | Deliverables |
|--------|-------|-------------|
| 7 | XP & Progression | Level system, XP calculations, unlocks |
| 8 | Achievements | Badge system, notifications, gallery |
| 9 | Skill Trees | 3 branches, node unlocking, visual tree |

### Phase 4: Polish & Launch (Weeks 19-24)
| Sprint | Focus | Deliverables |
|--------|-------|-------------|
| 10 | UI/UX Polish | Animations, transitions, accessibility |
| 11 | Testing & Performance | E2E tests, performance optimization, bug fixes |
| 12 | App Store Submission | Screenshots, metadata, TestFlight beta, submission |

---

## Phase 1: Foundation (Sprints 1-3)

### Sprint 1: Project Setup + Navigation

**Goal**: Production-ready project structure with navigation

**Tasks**:
1. Initialize Expo SDK 54 project with TypeScript
2. Configure React Navigation v7
   - Apply Dimensions patch (pre-init-dimensions.js)
   - Stack navigator for main flow
   - Tab navigator for home screens
   - Modal navigator for overlays
3. Set up folder structure (screens, components, services, etc.)
4. Configure Zustand stores (user, progress, reading, journal)
5. Implement dark theme system (purple/cosmic palette)
6. Create placeholder screens (20+ screens)
7. Test navigation flow end-to-end

**Deliverables**:
- ✅ App launches without errors
- ✅ Navigation between all screens works
- ✅ Dark theme applied globally
- ✅ Zustand stores accessible from components

**Acceptance Criteria**:
- Build time < 30 seconds
- Hot reload works reliably
- No navigation warnings/errors
- Theme switching works (for future light mode)

---

### Sprint 2: Data Layer + Card Engine

**Goal**: Load therapeutic content, implement card drawing logic

**Tasks**:
1. Create data loading service
   - Load JSON files from `/data/therapy/` and `/data/gamification/`
   - Parse and validate data structures
   - Cache in memory for performance
2. Implement card engine service
   - Fisher-Yates shuffle algorithm
   - Draw single card (daily pull)
   - Draw 3-card spread (Past/Present/Future)
   - Draw Celtic Cross (10-card)
   - Card position meanings
   - Reversed card logic (optional feature)
3. Implement reading service
   - Save reading to AsyncStorage
   - Reading history management
   - Reading analysis (connect cards to therapeutic themes)
4. Create card meaning lookup service
   - Get card interpretation
   - Get CBT/DBT/Mindfulness connections
   - Get journal prompts for card
5. Unit tests for all services (Jest)

**Deliverables**:
- ✅ All 3,700+ lines of JSON data load successfully
- ✅ Card drawing produces valid results
- ✅ Readings save to local storage
- ✅ 90%+ test coverage for services

**Acceptance Criteria**:
- Data loads in < 100ms (first time) / < 10ms (cached)
- Shuffles are truly random (Chi-square test)
- No duplicate cards in single reading
- Readings persist across app restarts

---

### Sprint 3: UI Component Library

**Goal**: Build reusable components with asset integration

**Tasks**:
1. Create card components
   - CardImage (renders 1080x1920 card asset)
   - CardFlip (animated flip from back to front)
   - CardStack (pile of cards)
   - CardSpread (layout cards in spread pattern)
2. Create button components
   - PrimaryButton (gold/purple themed)
   - IconButton (using curated button assets)
   - TabBarButton (navigation tabs)
3. Create modal components
   - Dialog box (using frame assets)
   - Achievement notification
   - Level-up celebration
4. Create progress components
   - XPBar (animated fill)
   - StreakFlame (animated flame icon)
   - AchievementBadge (rarity glow effects)
5. Create mindfulness components
   - BreathingCircle (expanding/contracting animation)
   - Timer (countdown with sound)
   - ProgressRing (circular progress)
6. Implement asset loading system
   - Lazy load card images
   - Preload critical UI assets
   - Cache strategy for 517 assets
7. Create Storybook for component testing (optional but recommended)

**Deliverables**:
- ✅ 30+ reusable components
- ✅ All components themed with assets
- ✅ Smooth animations (60fps)
- ✅ Asset loading optimized

**Acceptance Criteria**:
- Component render time < 16ms (60fps)
- Assets load progressively (no blocking)
- Memory usage < 200MB with all assets cached
- Components work on both iOS and Android

---

## Phase 2: Core Features (Sprints 4-6)

### Sprint 4: Reading Experience

**Goal**: Complete tarot reading flow with animations

**Tasks**:
1. Implement reading type selection screen
   - Single Card (daily pull)
   - 3-Card Spread (Past/Present/Future)
   - Celtic Cross
   - Other spreads (5-card, relationship, etc.)
2. Implement card selection flow
   - Deck animation (fan out, shuffle)
   - User taps to draw card
   - Card flip animation (back → front)
   - Reveal animation (glow, particles)
3. Implement card meaning screen
   - Display card image
   - Show card meaning (upright/reversed)
   - Show position meaning (if spread)
   - Show CBT/DBT/Mindfulness connections
   - Offer journal prompts
4. Implement reading summary screen
   - Show all drawn cards
   - Overall reading interpretation
   - Save to history
   - Share option (screenshot, no personal data)
5. Implement reading history screen
   - List past readings
   - Filter by date, spread type
   - Re-view past readings
   - Delete readings

**Deliverables**:
- ✅ Complete reading flow (select → draw → interpret → save)
- ✅ Smooth animations throughout
- ✅ Reading history functional

**Acceptance Criteria**:
- Reading flow takes < 2 minutes
- Animations are smooth (60fps)
- Readings saved to storage
- No frame drops during card flip

---

### Sprint 5: Journal System

**Goal**: Comprehensive journaling with therapeutic integration

**Tasks**:
1. Implement journal entry screen
   - Text input (multiline, markdown support)
   - Prompt selector (625+ prompts)
   - Emotion selector (for DBT diary card)
   - Distortion tagging (CBT integration)
   - DBT skill tagging (which skills used)
2. Implement prompt system
   - Smart prompt suggestions based on:
     - Card drawn
     - User mood
     - Recent themes
     - Skill tree progress
     - Time of day
   - Random deep-dive prompts
3. Implement CBT thought record
   - Situation → Thoughts → Feelings → Alternative → New Feelings
   - Distortion identification hints
   - Evidence gathering prompts
4. Implement DBT diary card
   - Daily emotion ratings (0-10)
   - Urge tracking
   - Skills used + effectiveness
   - Notes
5. Implement journal history
   - List entries by date
   - Search by keyword, emotion, distortion
   - Filter by tags
   - Export as JSON/PDF
6. Implement AI journal analysis (OPTIONAL, OPT-IN)
   - Send entry to Claude API
   - Receive insights (distortions, patterns, compassionate reflection)
   - Ephemeral (not stored on server)
   - Requires user API key OR premium subscription

**Deliverables**:
- ✅ Full journal system with prompts
- ✅ CBT/DBT integration
- ✅ Optional AI analysis

**Acceptance Criteria**:
- Journal entries save locally
- Prompts are contextually relevant
- Search returns results in < 100ms
- AI analysis (if enabled) works without storing data

---

### Sprint 6: Mindfulness Practices

**Goal**: 28 guided mindfulness exercises with tracking

**Tasks**:
1. Implement mindfulness library screen
   - List all 28 practices
   - Categorize (breathing, body scan, meditation, movement, daily)
   - Show which practices completed
   - Track favorites
2. Implement breathing exercises
   - Box breathing (4-4-4-4) with visual guide
   - 4-7-8 breathing with timer
   - Coherent breathing (5-5)
   - Alternate nostril (instructions + timer)
   - Physiological sigh (instructions)
3. Implement body scan practices
   - Progressive Muscle Relaxation (15min guided audio)
   - Body scan meditation (20min guided audio)
   - Quick body check-in (1min)
4. Implement meditation practices
   - Loving-kindness (metta) with script
   - Breath awareness with timer
   - Open awareness (choiceless awareness)
   - Safe place visualization with script
5. Implement movement practices
   - Walking meditation (instructions + timer)
   - Mindful stretching (instructions)
   - Shaking practice (TRE-inspired)
6. Implement daily activities
   - Mindful eating
   - Mindful dishwashing (Thich Nhat Hanh)
   - Morning intentions
   - Gratitude practice
   - Tech-free first hour (reminder)
   - Sacred pause (instructions)
7. Implement practice tracking
   - Log completion
   - Track duration
   - Rate effectiveness (1-5)
   - Add notes
   - Award XP

**Deliverables**:
- ✅ All 28 practices implemented
- ✅ Audio guides (or TTS)
- ✅ Practice tracking with XP rewards

**Acceptance Criteria**:
- Timers are accurate (no drift)
- Audio syncs with visual guidance
- Practice data persists
- XP awarded correctly

---

## Phase 3: Gamification (Sprints 7-9)

### Sprint 7: XP & Progression

**Goal**: Implement 50-level progression system

**Tasks**:
1. Implement XP calculation service
   - XP for each action (reading, journal, mindfulness, etc.)
   - Multipliers (first of day, quality, etc.)
   - Streak bonuses
2. Implement level progression
   - Calculate level from XP (formula: 100 * level^1.8)
   - Level up detection
   - Level up rewards (unlock features, cosmetics)
3. Implement progression UI
   - XP bar (animated fill on gain)
   - Level indicator
   - Next level progress (e.g., "340/724 XP to Level 5")
   - Level up animation (celebration!)
4. Implement milestone unlocks
   - Level 5: Advanced readings
   - Level 10: AI journal analysis
   - Level 15: Second skill tree branch
   - Level 20: Custom spreads
   - Level 25: Oracle mode (community)
   - Level 30: Full unlock + prestige option
5. Implement streak system
   - Daily login detection
   - Consecutive days tracking
   - Streak bonuses (3, 7, 14, 30, 100, 365 days)
   - Gentle streak reset (no shame)

**Deliverables**:
- ✅ XP system functional
- ✅ 50 levels with titles
- ✅ Milestone unlocks work
- ✅ Streak tracking accurate

**Acceptance Criteria**:
- XP calculations are correct
- Level up triggers reliably
- Unlocks activate at correct levels
- Streak persists across days

---

### Sprint 8: Achievements

**Goal**: Implement 30+ achievements with rarity system

**Tasks**:
1. Implement achievement detection service
   - Listen for events (first reading, 100 journals, etc.)
   - Check completion criteria
   - Award achievement
   - Notify user
2. Implement achievement UI
   - Achievement gallery (grid of badges)
   - Rarity visual (common, uncommon, rare, epic, legendary)
   - Glow effects based on rarity
   - Progress toward locked achievements
   - Achievement detail modal
3. Implement achievement notifications
   - Toast notification on unlock
   - Full-screen celebration for epic/legendary
   - Sound effect (optional)
4. Implement achievement categories
   - Foundations (first steps)
   - Mastery (skill-based)
   - Dedication (streaks, consistency)
   - Discovery (try everything)
   - Helping (community)
   - Challenges (difficult tasks)
5. Create secret achievements
   - Hidden until unlocked
   - Easter eggs (e.g., "Night Owl" for 2am journaling)

**Deliverables**:
- ✅ 30+ achievements functional
- ✅ Achievement gallery beautiful
- ✅ Notifications work

**Acceptance Criteria**:
- Achievements unlock correctly
- No duplicate unlocks
- Gallery shows correct rarity
- Secret achievements stay hidden

---

### Sprint 9: Skill Trees

**Goal**: Implement 3-branch skill tree system

**Tasks**:
1. Implement skill tree data service
   - Load skill tree JSON
   - Track unlocked nodes
   - Calculate skill point availability
   - Validate unlock requirements
2. Implement skill tree UI
   - Visual tree layout (CBT, DBT, Mindfulness branches)
   - Node states (locked, unlocked, active)
   - Unlock animations
   - Connection lines between tiers
3. Implement node unlocking
   - Spend skill points
   - Activate node effect
   - Visual feedback
   - Undo/respec option (monthly or premium)
4. Implement node effects
   - CBT nodes: Unlock features (distortion hints, thought records, etc.)
   - DBT nodes: Unlock tools (TIPP guide, DEAR MAN script, etc.)
   - Mindfulness nodes: Unlock practices (breathing guides, meditations, etc.)
5. Implement skill point sources
   - Level up: 1 point
   - Complete achievement: 0-5 points (by rarity)
   - Milestone streaks: 1 point

**Deliverables**:
- ✅ 3 skill trees functional
- ✅ 30+ nodes total
- ✅ Unlocking works correctly
- ✅ Effects activate

**Acceptance Criteria**:
- Trees render correctly
- Unlock logic prevents invalid states
- Effects actually grant features
- Respec option works

---

## Phase 4: Polish & Launch (Sprints 10-12)

### Sprint 10: UI/UX Polish

**Goal**: S-tier visual polish and accessibility

**Tasks**:
1. Animation polish
   - Card flip: Smooth 3D rotation
   - Particle effects: Subtle, performant
   - Level up: Celebratory but not obnoxious
   - Achievement unlock: Satisfying reveal
   - Page transitions: Smooth, directional
2. Micro-interactions
   - Button press feedback (haptic + visual)
   - Swipe gestures (dismiss modals)
   - Pull-to-refresh (reading history)
   - Drag-to-reorder (favorites)
3. Loading states
   - Skeleton screens (not spinners)
   - Progressive image loading
   - Smooth error states
4. Accessibility
   - Screen reader support (all content)
   - Color contrast (WCAG AA minimum)
   - Font size scaling
   - Reduced motion option
   - Haptic feedback toggle
5. Onboarding
   - Welcome flow (3-5 screens)
   - Tutorial overlays (first use)
   - Gentle guidance (not overwhelming)
6. Empty states
   - No readings yet → Draw first card!
   - No journal entries → Start writing!
   - No achievements → Here's how to unlock!

**Deliverables**:
- ✅ All animations smooth (60fps)
- ✅ Accessibility score > 90%
- ✅ Onboarding complete
- ✅ Empty states informative

**Acceptance Criteria**:
- No jank or frame drops
- VoiceOver/TalkBack work correctly
- Reduced motion mode works
- New users understand app flow

---

### Sprint 11: Testing & Performance

**Goal**: Production-ready quality

**Tasks**:
1. Unit testing (Jest)
   - Services: 90%+ coverage
   - Components: 80%+ coverage
   - Critical paths: 100% coverage
2. Integration testing
   - Reading flow end-to-end
   - Journal flow end-to-end
   - Mindfulness flow end-to-end
   - Gamification calculations
3. E2E testing (Detox)
   - Happy path user journey
   - Error cases (no network, etc.)
   - Edge cases (empty data, etc.)
4. Performance testing
   - App launch time < 2 seconds
   - Screen transition < 100ms
   - Asset loading optimized
   - Memory usage < 200MB
   - Bundle size < 50MB
5. Device testing
   - iOS 14+ (iPhone SE to iPhone 15 Pro Max)
   - Android 10+ (mid-range to flagship)
   - Tablet layouts (iPad, Android tablets)
6. Bug bash
   - Internal team testing
   - Fix all critical bugs
   - Triage medium/low bugs
7. Performance optimization
   - Code splitting
   - Asset compression
   - Memoization
   - Lazy loading

**Deliverables**:
- ✅ Test suite passing
- ✅ Performance benchmarks met
- ✅ Critical bugs fixed
- ✅ App store-ready build

**Acceptance Criteria**:
- 0 critical bugs
- 0 crashes in testing
- Performance targets met
- Works on target devices

---

### Sprint 12: App Store Submission

**Goal**: Launch on iOS and Android

**Tasks**:
1. App Store assets
   - App icon (1024x1024 for iOS, adaptive for Android)
   - Screenshots (6.7", 5.5" for iOS; multiple for Android)
   - App preview video (15-30 seconds)
2. App Store metadata
   - App name: VeilPath - Mental Wellness Tarot
   - Subtitle: Evidence-Based Therapy + Tarot
   - Description (4000 chars, optimized for ASO)
   - Keywords (100 chars, competitive analysis)
   - Category: Health & Fitness > Mental Health
   - Age rating: 17+ (mental health content)
3. Privacy policy & ToS hosting
   - Host PRIVACY_POLICY.md and TERMS_OF_SERVICE.md
   - Create app website (landing page)
   - Link from app metadata
4. TestFlight beta
   - Upload build to TestFlight
   - Invite 20-50 beta testers
   - Collect feedback
   - Iterate on critical feedback
5. App Store submission (iOS)
   - Submit for review
   - Respond to any questions
   - Address rejection reasons from prior version:
     - Spam: "No, unique custom deck + therapy integration"
     - Lacking features: "625+ prompts, 5-level gamification, 28 mindfulness practices"
     - Saturated market: "Mental WELLNESS focus, not generic tarot"
6. Google Play submission (Android)
   - Create store listing
   - Upload APK/AAB
   - Submit for review
7. Launch monitoring
   - Sentry for crash reporting
   - Analytics for user behavior
   - Support email ready

**Deliverables**:
- ✅ App Store submission complete
- ✅ Google Play submission complete
- ✅ Beta tested
- ✅ Monitoring in place

**Acceptance Criteria**:
- App approved by Apple
- App approved by Google
- No crashes in first 48 hours
- Support requests answered within 24 hours

---

## Testing Strategy

### Unit Testing (Jest)
**Coverage target**: 85%+ overall

**Priority areas**:
- Card engine logic (shuffle, draw, spreads)
- XP calculations and level progression
- Achievement detection
- Data loading and parsing
- Store mutations

### Integration Testing
**Focus**: Service interactions

**Test scenarios**:
- Reading flow: Draw card → Save → Load from history
- Journal flow: Create entry → Tag distortion → Search
- Gamification flow: Earn XP → Level up → Unlock achievement

### E2E Testing (Detox)
**Platforms**: iOS simulator, Android emulator

**Critical user journeys**:
1. New user onboarding → First reading → First journal
2. Returning user → Daily card pull → Earn XP → Level up
3. Mindfulness practice → Track completion → Earn achievement
4. Skill tree → Unlock node → Use new feature
5. Error handling → No network → Graceful degradation

### Manual QA
**Device matrix**:
- iPhone SE (small screen, old iOS)
- iPhone 14 Pro (notch)
- iPhone 15 Pro Max (large screen)
- iPad Pro (tablet layout)
- Samsung Galaxy S21 (Android flagship)
- Pixel 6 (pure Android)
- OnePlus 9 (Android mid-range)

**Test cases**:
- All features work on all devices
- Layouts adapt to screen sizes
- Performance acceptable on mid-range devices
- Dark theme consistent

### Beta Testing
**Testers**: 30-50 people

**Profiles**:
- 10 tech-savvy early adopters
- 10 therapy/mental health professionals
- 10 tarot enthusiasts
- 10 general users (non-technical)

**Feedback channels**:
- TestFlight feedback
- Discord channel
- Google Form survey

---

## Performance Benchmarks

### App Performance
| Metric | Target | Measurement |
|--------|--------|-------------|
| Cold start (app launch) | < 2 seconds | Time to interactive |
| Hot start (background return) | < 500ms | Time to interactive |
| Screen transition | < 100ms | Navigation delay |
| Card flip animation | 60fps | Frame rate monitor |
| Asset loading | < 100ms | Image appear time |
| Memory usage (idle) | < 150MB | Xcode instruments |
| Memory usage (peak) | < 250MB | During heavy use |
| Bundle size (iOS) | < 50MB | Archive size |
| Bundle size (Android) | < 50MB | APK/AAB size |

### Data Performance
| Operation | Target | Notes |
|-----------|--------|-------|
| Load JSON data | < 100ms | First launch |
| Load JSON data (cached) | < 10ms | Subsequent |
| Save journal entry | < 50ms | AsyncStorage write |
| Load reading history | < 100ms | 100+ readings |
| Search journal entries | < 100ms | Full-text search |
| Calculate XP/level | < 5ms | After each action |

### Network Performance (Optional Features)
| Operation | Target | Notes |
|-----------|--------|-------|
| AI journal analysis | < 3 seconds | Claude API call |
| Cloud backup sync | < 5 seconds | Supabase sync |
| Image upload | < 2 seconds | Profile picture |

---

## App Store Submission

### Apple App Store

**Category**: Health & Fitness > Mental Health
**Age Rating**: 17+
**Price**: Free (with in-app subscription)

**Addressing Prior Rejection**:

1. **"Spam"**
   - **Counter**: Custom 78-card deck (not stock Rider-Waite)
   - **Counter**: 3,700+ lines of therapeutic content (not generic meanings)
   - **Counter**: Original UI/UX with 517 curated assets

2. **"Lacking features to set it apart"**
   - **Counter**: 5-level gamification (30+ achievements, skill trees, prestige)
   - **Counter**: 625+ therapeutic journal prompts
   - **Counter**: 28 guided mindfulness practices
   - **Counter**: Evidence-based CBT/DBT/Mindfulness integration
   - **Counter**: Optional AI-powered journal analysis

3. **"Brought nothing new to the table"**
   - **Counter**: Mental WELLNESS platform (not generic tarot)
   - **Counter**: Therapeutic journaling with distortion identification
   - **Counter**: Skill tree progression teaching real therapy skills
   - **Counter**: Tarot as archetypal entry point for self-reflection (novel positioning)

**App Store Description** (excerpt):
> **VeilPath: Therapy Disguised as Tarot. Growth Gamified.**
>
> Struggling with anxiety, depression, or just feeling stuck? VeilPath combines the wisdom of tarot with evidence-based therapy (CBT, DBT, mindfulness) and engaging gamification to support your mental wellness journey.
>
> **Why VeilPath is Different:**
> - 🎴 Custom dark fantasy tarot deck (78 cards)
> - 🧠 CBT integration: Identify cognitive distortions, challenge thoughts
> - 🧘 DBT skills: Emotion regulation, distress tolerance, mindfulness
> - 📓 625+ therapeutic journal prompts
> - 🎯 30+ achievements, skill trees, XP progression
> - 🌬️ 28 guided mindfulness practices
> - 🤖 Optional AI journal analysis (privacy-first, ephemeral)
>
> **NOT a replacement for therapy.** VeilPath is a self-help tool that complements professional mental health care.

**Keywords**: tarot, mental wellness, CBT, DBT, mindfulness, therapy, journaling, anxiety, depression, self-help, meditation

**Screenshots**:
1. **Tarot Reading in Progress** - Show cosmic background, card spread
2. **Journal with Prompts** - Highlight therapeutic integration
3. **Achievement Unlocked** - Show gamification (epic badge with glow)
4. **Skill Tree** - Show CBT/DBT/Mindfulness branches
5. **Mindfulness Practice** - Show breathing circle animation

**App Preview Video** (30 seconds):
- 0-5s: Card flip animation + cosmic background
- 5-10s: Journal entry with prompt suggestion
- 10-15s: Achievement unlocked celebration
- 15-20s: Skill tree node unlock
- 20-25s: Mindfulness breathing exercise
- 25-30s: Logo + tagline: "Therapy Disguised as Tarot. Growth Gamified."

---

### Google Play Store

**Category**: Health & Fitness
**Content Rating**: Teen (13+) with mental health disclaimer
**Price**: Free (with in-app subscription)

**Similar positioning** to Apple, but:
- More permissive review process
- Faster approval (typically 1-3 days vs. 1-2 weeks)
- Can update faster if needed

---

## Post-Launch

### Week 1-2 (Stabilization)
- Monitor Sentry for crashes (target: < 0.1% crash rate)
- Respond to user reviews (target: < 24 hours)
- Hot-fix critical bugs
- Monitor analytics (retention, engagement)

### Month 1 (Iteration)
- Analyze user behavior (which features used most)
- A/B test onboarding flow
- Iterate on UI based on feedback
- Add requested features to roadmap

### Month 2-3 (Growth)
- Implement referral program
- Add community features (Oracle mode for level 25+ users)
- Seasonal content (Winter Solstice season)
- Marketing push (ProductHunt, Reddit, social media)

### Month 4-6 (Expansion)
- Add new card spread types
- Add new mindfulness practices
- Build web app (React with same codebase)
- Explore partnerships (therapists, wellness centers)

### Long-term (6-12 months)
- AI improvements (better analysis, personalization)
- Wearable integration (Apple Watch, Fitbit for mindfulness tracking)
- Therapist dashboard (optional supervised mode)
- Internationalization (Spanish, French, German)
- Android Wear app
- Desktop app (Electron)

---

## Risk Mitigation

### Technical Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| React 19 + Hermes incompatibility | Low | High | Already patched (pre-init-dimensions.js) |
| Performance issues on mid-range devices | Medium | High | Optimize early, test on target devices |
| Asset loading slow | Medium | Medium | Lazy loading, caching, compression |
| App Store rejection | Medium | High | Address prior rejection reasons explicitly |

### Business Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Low conversion (free → paid) | Medium | High | Optimize paywall, offer trial, demonstrate value |
| High churn | Medium | High | Gamification, streaks, community to drive retention |
| Competition | Medium | Medium | Unique positioning (therapy + tarot), premium quality |
| Legal (mental health claims) | Low | High | Clear disclaimers, lawyer review of ToS/Privacy |

### Operational Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Solo developer burnout | Medium | High | Scope wisely, take breaks, hire help if revenue supports |
| Support overwhelm | Low | Medium | Clear in-app help, FAQ, automated responses |
| Server costs exceed revenue | Low | Medium | Serverless architecture, usage-based pricing |

---

## Success Metrics (KPIs)

### Acquisition
- **Downloads**: 10,000 in first 3 months
- **Install source**: 60% organic (ASO), 40% marketing
- **Cost per install (if paid ads)**: < $2

### Activation
- **Onboarding completion**: > 70%
- **First reading within 24h**: > 80%
- **First journal entry within 7 days**: > 50%

### Engagement
- **DAU/MAU ratio**: > 25% (daily/monthly active users)
- **Session length**: > 10 minutes average
- **Sessions per week**: > 3

### Retention
- **Day 1**: > 60%
- **Day 7**: > 40%
- **Day 30**: > 20%

### Revenue
- **Free → Paid conversion**: > 15%
- **Average revenue per user (ARPU)**: > $1/month
- **Churn**: < 5% monthly
- **Lifetime value (LTV)**: > $100

### Satisfaction
- **App Store rating**: > 4.5 stars
- **Net Promoter Score**: > 50
- **Support ticket resolution**: < 24 hours

---

## Conclusion

This roadmap transforms VeilPath from concept to production-ready mental wellness platform in 6 months (12 sprints). The backwards planning approach ensures we ship something Apple will approve while delivering genuine value to users struggling with mental health.

**Key Success Factors**:
1. ✅ **Differentiation**: Not a generic tarot app - therapy integration sets us apart
2. ✅ **Quality**: S-tier UI/UX polish, smooth animations, premium feel
3. ✅ **Evidence**: CBT, DBT, Mindfulness content is research-backed
4. ✅ **Ethics**: Mental wellness without manipulation, privacy-first
5. ✅ **Gamification**: Motivates therapeutic work without dark patterns
6. ✅ **Scalability**: Serverless architecture, local-first data

**The Vision**: VeilPath becomes the **premier mental wellness platform** that uses tarot as an accessible, engaging entry point for evidence-based self-help. Users think they're getting a tarot app. They discover a comprehensive therapy toolkit that actually works.

---

**Next Steps**:
1. Review and approve this roadmap
2. Set up development environment (Sprint 1, Week 1)
3. Begin implementation
4. Launch in Q2 2026

Let's build something that changes lives. 🌟
