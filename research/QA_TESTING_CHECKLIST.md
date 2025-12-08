# QA Testing Checklist

**App**: VeilPath - Mental Wellness Tarot App
**Version**: 1.0.0
**Last Updated**: 2025-11-22
**Test Environments**: iOS, Android, Web

---

## Testing Overview

This comprehensive checklist covers functional testing, UI/UX validation, performance testing, and edge case verification for all features in VeilPath.

**Testing Methodology**:
- ✅ = Pass
- ❌ = Fail
- ⚠️ = Partial/Needs Review
- ⏭️ = Skipped/Not Applicable

---

## 1. Onboarding & First-Time Experience

### 1.1 Welcome Flow
- [ ] App launches successfully on first install
- [ ] Welcome screen displays correctly
- [ ] Welcome screen displays all 4 slides
- [ ] User can swipe through slides
- [ ] "Get Started" button navigates to main app
- [ ] Skipping onboarding works correctly
- [ ] Onboarding doesn't show on second launch

### 1.2 First-Time Hints
- [ ] First reading shows tutorial hints
- [ ] First journal entry shows tutorial hints
- [ ] Achievements intro appears after first achievement
- [ ] Skill tree intro appears on first visit

---

## 2. Tarot Reading System

### 2.1 Card Selection & Drawing
- [ ] Single Card reading loads correctly
- [ ] Three Card spread loads correctly
- [ ] Celtic Cross spread loads correctly
- [ ] Cards shuffle animation plays smoothly
- [ ] Cards draw randomly (no repeats in same spread)
- [ ] Card flip animation is smooth (60fps)
- [ ] Reversed cards display upside down correctly

### 2.2 Reading Interface
- [ ] Question input field works
- [ ] Placeholder text displays
- [ ] Question saves with reading
- [ ] Card images load correctly
- [ ] Card names display correctly
- [ ] Position labels are accurate
- [ ] Reversed status shows correctly
- [ ] Interpretation text displays

### 2.3 AI Interpretation (Claude)
- [ ] "Get AI Interpretation" button appears
- [ ] Loading indicator shows during API call
- [ ] AI interpretation displays correctly
- [ ] Error handling for API failures
- [ ] Timeout handling (10s)
- [ ] Retry functionality works
- [ ] Interpretation saves with reading

### 2.4 Save & History
- [ ] Save reading button works
- [ ] Notes field saves correctly
- [ ] Reading appears in history immediately
- [ ] Reading history sorts by date (newest first)
- [ ] Favorite toggle works
- [ ] Favorite readings filter correctly
- [ ] Delete reading works
- [ ] Delete confirmation modal appears

---

## 3. Journal System

### 3.1 Create Entry
- [ ] New entry button opens editor
- [ ] Title field works (max 100 chars)
- [ ] Content field works (unlimited)
- [ ] Mood selector displays all moods
- [ ] Selected mood highlights correctly
- [ ] Tags input works
- [ ] Tag chips display correctly
- [ ] Private toggle works
- [ ] Word count updates live
- [ ] Save button enables when content exists
- [ ] Entry saves successfully

### 3.2 CBT Features
- [ ] "Identify Distortions" button works
- [ ] Distortion list displays
- [ ] User can select multiple distortions
- [ ] Selected distortions save
- [ ] Reframe field works
- [ ] CBT data saves with entry
- [ ] Distortion counter increments in stats

### 3.3 DBT Features
- [ ] DBT skill selector displays
- [ ] All 4 modules available (Mindfulness, Distress Tolerance, Emotion Regulation, Interpersonal Effectiveness)
- [ ] Skill notes field works
- [ ] DBT data saves with entry
- [ ] DBT counter increments in stats

### 3.4 View & Edit Entries
- [ ] Entry list displays all entries
- [ ] Entries sort by date (newest first)
- [ ] Tap entry to view details
- [ ] Edit button opens editor
- [ ] Edits save correctly
- [ ] Entry metadata updates (date modified)
- [ ] Delete entry works
- [ ] Delete confirmation appears

### 3.5 Filtering & Search
- [ ] Filter by mood works
- [ ] Filter by tags works
- [ ] Multiple filters work together
- [ ] Search by keyword works
- [ ] Search includes title and content
- [ ] Clear filters works

---

## 4. Mindfulness & Therapy Tools

### 4.1 Breathing Exercises
- [ ] Exercise list displays all 4 exercises
- [ ] Box Breathing (4-4-4-4) works
- [ ] 4-7-8 Breathing works
- [ ] Equal Breathing works
- [ ] Alternate Nostril visualization works
- [ ] Animation is smooth
- [ ] Audio cues play (if enabled)
- [ ] Haptic feedback works (if enabled)
- [ ] Timer counts correctly
- [ ] Session completes and saves

### 4.2 Guided Meditations
- [ ] Meditation list displays
- [ ] Play/pause works
- [ ] Audio plays correctly
- [ ] Progress bar updates
- [ ] Seek/scrub works
- [ ] Session time tracks correctly
- [ ] Completion triggers stats update

### 4.3 Affirmations
- [ ] Daily affirmation displays
- [ ] Affirmation changes daily
- [ ] Share affirmation works
- [ ] Favorite affirmations save
- [ ] Category filter works

---

## 5. Gamification System

### 5.1 XP & Leveling
- [ ] XP awards correctly for actions
- [ ] XP multipliers apply (first of day, CBT, DBT)
- [ ] Level-up animation plays
- [ ] Level-up awards skill point
- [ ] Title changes at milestones
- [ ] Progress bar shows correctly
- [ ] XP to next level calculates correctly

### 5.2 Achievements
- [ ] Achievement list displays
- [ ] Locked achievements show requirements
- [ ] Unlocked achievements show timestamp
- [ ] Achievement unlock animation plays
- [ ] Achievement notification appears
- [ ] Progress bar shows for progressive achievements
- [ ] Secret achievements hide until unlocked

### 5.3 Skill Trees
- [ ] All 3 branches display (CBT, DBT, Mindfulness)
- [ ] Branch progress shows correctly
- [ ] Skill tree detail view opens
- [ ] Nodes show locked/available/unlocked states
- [ ] Dependency lines display correctly
- [ ] Unlock button enables when criteria met
- [ ] Unlock deducts skill points
- [ ] Unlock confirmation appears
- [ ] Skill effects apply (if implemented)

---

## 6. Profile & Statistics

### 6.1 Profile Display
- [ ] User name displays
- [ ] Level and XP display correctly
- [ ] Current title displays
- [ ] Profile avatar/icon shows
- [ ] Edit profile button works
- [ ] Profile saves correctly

### 6.2 Statistics Overview
- [ ] Total readings count accurate
- [ ] Total journal entries accurate
- [ ] Total mindfulness minutes accurate
- [ ] Current streak displays
- [ ] Longest streak displays
- [ ] Days active calculates correctly

### 6.3 Detailed Statistics
- [ ] Reading insights accurate
- [ ] Most drawn cards chart correct
- [ ] Journal mood breakdown correct
- [ ] Mood trend chart displays
- [ ] Mindfulness progress shows
- [ ] CBT/DBT stats accurate
- [ ] Milestones list populates

### 6.4 Progression Display
- [ ] Level card shows correct level
- [ ] XP bar fills correctly
- [ ] Next level XP shows
- [ ] Achievements count accurate
- [ ] Skill points count accurate

---

## 7. Settings & Preferences

### 7.1 General Settings
- [ ] Theme toggle works (if implemented)
- [ ] Notification toggle works
- [ ] Sound toggle works
- [ ] Haptics toggle works
- [ ] Settings persist after app restart

### 7.2 Data Management
- [ ] Export data button works
- [ ] Export generates valid JSON
- [ ] Export includes all user data
- [ ] Export excludes private entries
- [ ] Journal markdown export works
- [ ] Reading markdown export works
- [ ] Share functionality works
- [ ] Import data works (if implemented)

### 7.3 Privacy & Security
- [ ] Private entries excluded from export
- [ ] Biometric lock works (if enabled)
- [ ] Privacy policy link works
- [ ] Terms of service link works

### 7.4 About & Support
- [ ] App version displays
- [ ] Credits display correctly
- [ ] Help/FAQ opens
- [ ] Contact support works
- [ ] Rate app link works

---

## 8. Data Persistence & Sync

### 8.1 Local Storage
- [ ] User data persists after app close
- [ ] Readings persist
- [ ] Journal entries persist
- [ ] Achievements persist
- [ ] Skill tree state persists
- [ ] Settings persist
- [ ] No data loss on crash

### 8.2 State Management
- [ ] Zustand stores initialize correctly
- [ ] State updates propagate immediately
- [ ] No race conditions observed
- [ ] AsyncStorage errors handle gracefully

---

## 9. Performance Testing

### 9.1 Load Times
- [ ] App launches in <3s (cold start)
- [ ] App resumes in <1s (warm start)
- [ ] Home screen loads <100ms
- [ ] Reading screen loads <100ms
- [ ] Journal screen loads <100ms
- [ ] Statistics screen loads <500ms

### 9.2 Animation Performance
- [ ] Card flip animation 60fps
- [ ] Page transitions smooth
- [ ] Scroll performance smooth
- [ ] List rendering optimized
- [ ] No dropped frames during interactions

### 9.3 Memory Usage
- [ ] Baseline memory <50MB
- [ ] Peak memory <150MB
- [ ] No memory leaks after 30min use
- [ ] App doesn't crash under heavy use

---

## 10. Error Handling

### 10.1 Network Errors
- [ ] API timeout handled gracefully
- [ ] Offline mode works
- [ ] Network error messages clear
- [ ] Retry mechanism works
- [ ] Cached data displays when offline

### 10.2 Input Validation
- [ ] Empty fields validated
- [ ] Max length enforced
- [ ] Invalid characters handled
- [ ] Special characters work in journal

### 10.3 Error Boundaries
- [ ] App doesn't crash on JS errors
- [ ] Error boundary shows fallback UI
- [ ] "Try Again" button works
- [ ] Error details show in dev mode

---

## 11. Cross-Platform Testing

### 11.1 iOS Specific
- [ ] Safe area insets correct
- [ ] Notch/Dynamic Island handled
- [ ] Keyboard dismissal works
- [ ] Haptics work on supported devices
- [ ] Face ID/Touch ID works
- [ ] Share sheet works
- [ ] File export works

### 11.2 Android Specific
- [ ] Back button navigation works
- [ ] Hardware back button correct behavior
- [ ] Status bar theming correct
- [ ] Keyboard handling correct
- [ ] Share menu works
- [ ] File permissions handled
- [ ] Download/export to storage works

### 11.3 Web Specific
- [ ] Responsive design works
- [ ] Desktop layout appropriate
- [ ] Mobile web layout works
- [ ] File download works
- [ ] Keyboard shortcuts work (if implemented)

---

## 12. Accessibility

### 12.1 Screen Reader Support
- [ ] All interactive elements labeled
- [ ] Navigation announcements work
- [ ] Button labels descriptive
- [ ] Form inputs labeled
- [ ] Images have alt text

### 12.2 Visual Accessibility
- [ ] Text contrast meets WCAG AA
- [ ] Font sizes readable
- [ ] Touch targets ≥44x44pt
- [ ] Color not sole indicator
- [ ] Focus indicators visible

### 12.3 Motor Accessibility
- [ ] One-handed use possible
- [ ] No time-limited actions
- [ ] Gestures have alternatives
- [ ] Scrollable areas have momentum

---

## 13. Edge Cases & Stress Testing

### 13.1 Data Limits
- [ ] 1000+ readings perform well
- [ ] 1000+ journal entries perform well
- [ ] Very long journal entries (10,000+ words)
- [ ] Many tags on single entry (50+)
- [ ] Empty states display correctly

### 13.2 Unusual Inputs
- [ ] Emoji in journal titles
- [ ] Special characters in questions
- [ ] Very long card interpretation
- [ ] Rapid button tapping
- [ ] Quick navigation between screens

### 13.3 State Edge Cases
- [ ] Level 100+ progression works
- [ ] All achievements unlocked
- [ ] All skill nodes unlocked
- [ ] 0 skill points available
- [ ] Maximum streak (365+ days)

---

## 14. Security & Privacy

### 14.1 Data Privacy
- [ ] No analytics without consent
- [ ] Private entries truly private
- [ ] No PII leaked in logs
- [ ] Secure storage for sensitive data
- [ ] Export respects privacy settings

### 14.2 API Security
- [ ] API keys not exposed in app
- [ ] HTTPS only for network requests
- [ ] Request rate limiting
- [ ] Timeout protections

---

## 15. Pre-Launch Checklist

### 15.1 Final Checks
- [ ] All features working on all platforms
- [ ] No console errors in production
- [ ] No "TODO" or debug code
- [ ] Analytics properly configured
- [ ] Crash reporting configured
- [ ] Privacy policy finalized
- [ ] Terms of service finalized

### 15.2 Store Assets
- [ ] App icon all sizes
- [ ] Screenshots all sizes
- [ ] App Store description
- [ ] Google Play description
- [ ] Keywords optimized
- [ ] Age rating appropriate
- [ ] Category correct

### 15.3 Legal & Compliance
- [ ] Privacy policy accessible
- [ ] Terms accessible
- [ ] COPPA compliance (if applicable)
- [ ] GDPR compliance (if applicable)
- [ ] Data deletion process

---

## Bug Reporting Template

When reporting bugs, include:

**Bug ID**: [Unique identifier]
**Severity**: [Critical/High/Medium/Low]
**Priority**: [P0/P1/P2/P3]

**Environment**:
- Device: [iPhone 14, Pixel 7, etc.]
- OS Version: [iOS 17.2, Android 14, etc.]
- App Version: [1.0.0]

**Steps to Reproduce**:
1. Step one
2. Step two
3. Step three

**Expected Behavior**: [What should happen]

**Actual Behavior**: [What actually happens]

**Screenshots**: [Attach if applicable]

**Logs**: [Attach console logs if available]

**Frequency**: [Always/Sometimes/Rare]

**Workaround**: [If known]

---

## Test Cycle Tracking

### Alpha Testing (Internal)
- **Date**: [TBD]
- **Testers**: [Team members]
- **Focus**: Core functionality, critical bugs
- **Pass Rate**: [TBD]

### Beta Testing (External)
- **Date**: [TBD]
- **Testers**: [50-100 users]
- **Focus**: Real-world usage, UX feedback
- **Pass Rate**: [TBD]

### Release Candidate
- **Date**: [TBD]
- **Testers**: [All stakeholders]
- **Focus**: Final validation, edge cases
- **Pass Rate**: [Must be 100%]

---

**Testing Coordinator**: [Name]
**Last Test Cycle**: [Date]
**Next Test Cycle**: [Date]
**Issues Found**: [Count]
**Issues Resolved**: [Count]
**Release Ready**: [Yes/No]
