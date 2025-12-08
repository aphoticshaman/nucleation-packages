# INDIE GAME STUDIO PIPELINE

**S-Tier Game Development from Concept to Ship**

## Philosophy

**Ship Fast, Iterate Faster**

This isn't AAA. We're indie. That means:
- ✅ MVP first, polish later
- ✅ Vertical slice > horizontal features
- ✅ Playable beats perfect
- ✅ Ship weekly, improve daily
- ✅ Kill features aggressively
- ✅ Automate everything

---

## PHASE 0: PRE-PRODUCTION (Week 1)

### Game Design Document (GDD)

**Don't write a 50-page GDD. Write THIS:**

```markdown
# Lunatiq - Tarot RPG

## Elevator Pitch (30 seconds)
Farmville meets Diablo meets mystical tarot. Wander an isometric village, get AI-powered tarot readings, chat with Oracle, unlock your shadow.

## Core Loop (What players do every session)
1. Wander village (2-5 min)
2. Enter building (tarot shop, Oracle temple, etc.)
3. Get reading / chat / journal (5-15 min)
4. Gain XP, unlock achievements
5. Repeat

## Unique Selling Points (Why this, not competitors?)
1. AI-powered personalized readings (not canned responses)
2. 3D semantic space visualizer (see your psyche mapped)
3. Progression system (RuneScape for self-improvement)

## Target Audience
- Primary: 25-40 year old women interested in tarot, self-improvement
- Secondary: Jung/psychology nerds, indie game fans

## Monetization
- Free: Basic readings, limited Oracle chats
- Premium ($9.99/month): Unlimited AI, cloud sync, exclusive features

## Success Metrics
- D1 retention > 40%
- D7 retention > 20%
- Conversion to premium > 5%
- Session length > 10 minutes

## Development Timeline
- Week 1: Core loop playable
- Week 2-3: Polish, juice, content
- Week 4: TestFlight beta
- Week 5-6: Iterate based on feedback
- Week 7: Ship v1.0
```

**That's it. No more. Anything else goes in separate docs.**

### Technical Design Doc

```markdown
# Technical Architecture

## Stack
- React Native (Expo)
- Firebase (auth, Firestore, cloud sync)
- Vercel (API, Claude integration)
- AsyncStorage (local data)

## Key Systems
1. Isometric village (tap-to-move, building interactions)
2. Cloud API integration (Claude Haiku/Sonnet)
3. Data sync (local → cloud, conflict resolution)
4. Anti-cheat (HMAC signatures, rate limiting)
5. Animation system (sprite sheets, particles)

## Performance Budget
- App size: < 50 MB
- Cold start: < 3 seconds
- Frame rate: 30+ FPS
- API latency: < 2 seconds

## Third-Party Services
- Anthropic Claude API (AI readings)
- Firebase (backend)
- RevenueCat (subscriptions)
- Sentry (error tracking - add later)
```

### Asset List

**Spreadsheet or Notion board:**

| Asset Type | Count | Status | Owner | Notes |
|---|---|---|---|---|
| Buildings | 5 | Done | MJ + Claude | Need normalization |
| Tiles | 10 | Done | MJ + Claude | - |
| Character sprites | 1 | Done | MJ | Need idle anim |
| UI panels | 5 | In progress | MJ | - |
| Icons | 20 | To do | MJ | - |
| Tarot cards | 78 | To do | MJ | Low priority |
| Sound effects | 20 | Not started | - | Week 3 |
| Music | 3 tracks | Not started | - | Week 3 |

---

## PHASE 1: VERTICAL SLICE (Week 1-2)

### Goal: Playable Core Loop

**Definition:** One complete path through the game, polished enough to be fun.

**For Lunatiq:**
1. Character spawns in village
2. Walk to tarot shop
3. Tap to enter
4. Get ONE 3-card reading (real AI)
5. See results
6. XP gained, level up
7. Exit to village

**That's it. Don't build anything else yet.**

### Development Order

**Day 1-2: Core Mechanics**
- [ ] Isometric village renders
- [ ] Character spawns and can move
- [ ] Camera follows character
- [ ] Buildings are clickable

**Day 3-4: First Feature**
- [ ] Tarot shop interaction
- [ ] Reading flow (intention → cards → interpretation)
- [ ] AI integration working

**Day 5-6: Progression**
- [ ] XP system
- [ ] Level up feedback
- [ ] Stats screen

**Day 7: Polish**
- [ ] Smooth animations
- [ ] Sound effects
- [ ] Visual feedback (particles, glows)
- [ ] Playtestable build

**SHIP IT** (even if just to yourself)

---

## PHASE 2: HORIZONTAL EXPANSION (Week 3-4)

Now that core loop works, add breadth.

### Feature Priority Matrix

| Feature | Impact | Effort | Priority |
|---|---|---|---|
| Oracle chat | High | Medium | P0 (ship blocker) |
| Journal | Medium | Low | P1 (nice to have) |
| Career counselor | Medium | Medium | P2 (post-launch) |
| 3D semantic viz | High | High | P2 (post-launch) |
| Achievements | Low | Low | P1 (retention booster) |
| Cloud sync | High | High | P0 (premium feature) |

**P0 = Must ship**
**P1 = Ship if time**
**P2 = Post-launch content**

### Week 3 Tasks
- [ ] Implement all P0 features
- [ ] 80% of P1 features
- [ ] Ignore P2 (seriously, cut them)

### Week 4 Tasks
- [ ] Bug fixing
- [ ] Performance optimization
- [ ] Beta testing with 10-20 users
- [ ] Iterate based on feedback

---

## PHASE 3: ART PIPELINE

### Concept → Final Asset Workflow

```
1. CONCEPT (10 min)
   - Quick sketch or Pinterest reference
   - Color palette decision
   - Art direction note ("Diablo 3 meets Hades")

2. GENERATE (Midjourney, 20 min)
   - Use prompts from MIDJOURNEY_ASSET_PROMPTS.md
   - Generate 4 variations
   - Pick best, iterate with --sref

3. POST-PROCESS (Claude, 5 min)
   python tools/image_tools.py normalize input.png output.png
   python tools/image_tools.py remove-bg output.png final.png
   python tools/image_tools.py optimize final.png final_optimized.png

4. INTEGRATE (Code, 10 min)
   - Move to assets/art/{category}/
   - Import in component
   - Test in-game

5. ITERATE (Based on testing)
   - Too dark? Adjust brightness
   - Wrong style? Re-generate with --sref
   - Doesn't fit? Try different concept

Total time per asset: 45 minutes
```

### Batch Processing (For Similar Assets)

```bash
# Process all buildings at once
python tools/image_tools.py batch \
  assets/art/buildings_raw/ \
  assets/art/buildings/ \
  normalize --reference best_building.png

# Then remove backgrounds
python tools/image_tools.py batch \
  assets/art/buildings/ \
  assets/art/buildings_final/ \
  remove_bg

# Then optimize
python tools/image_tools.py batch \
  assets/art/buildings_final/ \
  assets/art/buildings_optimized/ \
  optimize
```

Saves hours vs manual GIMP work.

---

## PHASE 4: ANIMATION PIPELINE

### Character Animation Workflow

```
1. IDLE ANIMATION
   - Generate in MJ: "8 frame idle animation sprite sheet"
   - Extract frames:
     python tools/animation_tools.py extract idle_sheet.png frames/ --width 256 --height 256 --columns 8 --rows 1
   - Create manifest:
     python tools/animation_tools.py manifest frames/ character_idle --fps 12 --loop
   - Generate component:
     python tools/animation_tools.py generate-code character_idle_manifest.json src/components/CharacterIdleAnim.js
   - Use in game:
     <CharacterIdleAnim playing={true} style={{width: 128, height: 128}} />

2. WALK ANIMATION
   - Same process, 8 frames, 18 FPS

3. PARTICLES/EFFECTS
   - Generate particle sprite sheet (16 variations)
   - Use ParticleSystem component
   - Spawn randomly in background
```

### Preview Before Integrating

```bash
# Create GIF preview to test timing
python tools/animation_tools.py flipbook \
  frames/ \
  preview.gif \
  --duration 1000
```

Open `preview.gif` → looks good? Integrate. Looks bad? Adjust FPS or regenerate.

---

## PHASE 5: SOUND & MUSIC

### Sound Effects Priority

**P0 (Must have):**
- Button tap/click
- Card flip
- Level up/achievement
- Error/failure

**P1 (Nice to have):**
- Footsteps
- Door open/close
- Ambient village sounds
- Magic sparkle

**P2 (Post-launch):**
- Character voice grunts
- Oracle voice lines
- Full Foley

### Where to Get Sounds

**Free:**
- Freesound.org (CC0 sounds)
- Zapsplat.com (free tier)
- Kenney.nl (game assets including audio)

**Paid:**
- Epidemic Sound ($15/month, unlimited)
- Artlist ($15/month for SFX)

### Integration

```javascript
import { Audio } from 'expo-av';

// Load sound
const [sound, setSound] = useState();

async function playSound() {
  const { sound } = await Audio.Sound.createAsync(
    require('./assets/sounds/button_click.mp3')
  );
  setSound(sound);
  await sound.playAsync();
}

// Clean up
useEffect(() => {
  return sound
    ? () => { sound.unloadAsync(); }
    : undefined;
}, [sound]);
```

---

## PHASE 6: POLISH & JUICE

### What is "Juice"?

**Game feel** - the tiny details that make it satisfying.

**Examples:**
- Button scales down when pressed
- Particles spawn on tap
- Screen shake on level up
- Cards glow when hovered
- Smooth easing on all animations

### Juice Checklist

**Buttons:**
- [ ] Scale down on press (0.95x)
- [ ] Haptic feedback (Haptics.impactAsync())
- [ ] Sound effect
- [ ] Subtle glow/highlight

**Transitions:**
- [ ] Fade in/out (not instant)
- [ ] Slide animations (not teleport)
- [ ] Easing curves (not linear)

**Feedback:**
- [ ] Visual confirmation (checkmark, particle burst)
- [ ] Audio confirmation (chime, ding)
- [ ] Haptic confirmation (vibration)

**Progression:**
- [ ] XP bar fills smoothly (animated)
- [ ] Level up: screen shake + particles + sound
- [ ] Achievements: badge flies in from top

### Implementation

```javascript
import { Animated } from 'react-native';
import * as Haptics from 'expo-haptics';

function JuicyButton({ onPress, children }) {
  const scale = useRef(new Animated.Value(1)).current;

  const handlePressIn = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    Animated.spring(scale, {
      toValue: 0.95,
      useNativeDriver: true
    }).start();
  };

  const handlePressOut = () => {
    Animated.spring(scale, {
      toValue: 1,
      friction: 3,
      useNativeDriver: true
    }).start();
  };

  return (
    <Animated.View style={{ transform: [{ scale }] }}>
      <TouchableOpacity
        onPressIn={handlePressIn}
        onPressOut={handlePressOut}
        onPress={onPress}
      >
        {children}
      </TouchableOpacity>
    </Animated.View>
  );
}
```

---

## PHASE 7: PERFORMANCE OPTIMIZATION

### Measurement First

**Don't optimize blindly. Measure:**

```javascript
import { PerformanceObserver, performance } from 'perf_hooks';

// Measure render time
const start = performance.now();
// ... render ...
const end = performance.now();
console.log(`Render took ${end - start}ms`);

// Target: < 16ms (60 FPS) or < 33ms (30 FPS)
```

### Common Bottlenecks

**1. Too many re-renders**
```javascript
// Bad: Re-renders every frame
function BadComponent() {
  const [particles, setParticles] = useState(generateParticles());

  useEffect(() => {
    setInterval(() => {
      setParticles(prev => updateParticles(prev)); // Re-render!
    }, 16);
  }, []);
}

// Good: Update outside React, only re-render when needed
function GoodComponent() {
  const particlesRef = useRef(generateParticles());
  const [, forceUpdate] = useReducer(x => x + 1, 0);

  useEffect(() => {
    setInterval(() => {
      particlesRef.current = updateParticles(particlesRef.current);
      forceUpdate(); // Force re-render once per second, not every frame
    }, 1000);
  }, []);
}
```

**2. Large images**
```bash
# Before: 2MB PNG
# After: 200KB PNG (10x smaller)
python tools/image_tools.py optimize input.png output.png
```

**3. Too many animations**
```javascript
// Limit particles
const MAX_PARTICLES = 30; // Not 100

// Pause off-screen animations
if (!isVisible) {
  return null; // Don't render at all
}
```

### Performance Budget

- **Cold start:** < 3 seconds
- **Frame rate:** 30+ FPS (60 FPS ideal)
- **Memory:** < 150 MB
- **Battery:** < 5% per hour of active use

---

## PHASE 8: TESTING & QA

### Automated Testing

**Don't test everything. Test critical paths:**

```javascript
// Test: Can user get a reading?
describe('Tarot Reading Flow', () => {
  it('should complete a 3-card reading', async () => {
    const { getByText, getByTestId } = render(<App />);

    // Tap tarot shop
    fireEvent.press(getByTestId('tarot-shop'));

    // Enter intention
    fireEvent.changeText(getByTestId('intention-input'), 'Will I find love?');
    fireEvent.press(getByText('Begin Reading'));

    // Draw 3 cards
    for (let i = 0; i < 3; i++) {
      fireEvent.press(getByTestId(`card-${i}`));
    }

    // Should see interpretation
    await waitFor(() => {
      expect(getByText(/interpretation/i)).toBeTruthy();
    });
  });
});
```

### Manual Testing Checklist

**Before every build:**

- [ ] App launches without crash
- [ ] Can complete core loop (village → reading → results)
- [ ] Cloud sync works (if premium)
- [ ] No visual glitches
- [ ] Sounds play correctly
- [ ] Performance acceptable (no lag)

**Device Testing:**
- [ ] iPhone SE (low-end iOS)
- [ ] iPhone 14 (high-end iOS)
- [ ] Pixel 4a (low-end Android)
- [ ] Samsung S21 (high-end Android)

### Beta Testing

**Week 4-5: Closed Beta (10-20 users)**

1. **Recruit:** Friends, family, Twitter followers
2. **Distribute:** TestFlight (iOS), Internal Testing (Android)
3. **Collect feedback:** Google Form, Discord channel
4. **Iterate:** Fix bugs, adjust based on feedback
5. **Metrics:** Track retention, session length, crashes

**Key Questions:**
- Is the core loop fun?
- Do users understand how to play?
- Any game-breaking bugs?
- Would they pay for premium?

---

## PHASE 9: LAUNCH PREPARATION

### Pre-Launch Checklist

**App Store Listing:**
- [ ] Icon (1024x1024)
- [ ] Screenshots (6.5", 6.7" iPhone)
- [ ] App preview video (optional but recommended)
- [ ] Description (compelling, keywords)
- [ ] Privacy policy URL
- [ ] Support email

**Marketing:**
- [ ] Landing page (simple, 1-pager)
- [ ] Twitter account
- [ ] Reddit posts (r/tarot, r/incremental_games, r/IndieGaming)
- [ ] Press kit (screenshots, description, contact)
- [ ] Email to indie game YouTubers/streamers

**Analytics:**
- [ ] Firebase Analytics configured
- [ ] Key events tracked (reading completed, premium purchased, etc.)
- [ ] Crash reporting (Firebase Crashlytics)

**Monetization:**
- [ ] RevenueCat configured
- [ ] Subscription products created (Apple, Google)
- [ ] Paywall screen implemented
- [ ] Restore purchases flow working

### Launch Day

1. **Submit to stores** (Apple: 1-3 days review, Google: instant)
2. **Post on socials** (Twitter, Reddit, Discord)
3. **Email beta testers** ("We launched! Leave a review!")
4. **Monitor analytics** (crashes, retention, conversion)
5. **Respond to reviews** (especially negative ones)

---

## PHASE 10: POST-LAUNCH

### Week 1 After Launch

**Focus: Stability**

- Fix critical bugs ASAP (same day)
- Monitor crash rate (target: < 0.1%)
- Watch retention (D1, D7, D30)
- Respond to all reviews

### Week 2-4 After Launch

**Focus: Retention & Monetization**

- Improve onboarding (if D1 retention < 40%)
- Optimize paywall (if conversion < 5%)
- Add content (new buildings, features)
- A/B test (pricing, UI, messaging)

### Ongoing

**Weekly Release Cycle:**

**Monday:** Plan week (pick 1-2 features or fixes)
**Tuesday-Thursday:** Build
**Friday:** Test, polish
**Saturday:** Submit to stores
**Sunday:** Rest (or start next week's planning)

**Monthly Roadmap:**

- Month 1: Stability, core loop polish
- Month 2: Content (new buildings, features)
- Month 3: Social features (leaderboards, sharing)
- Month 4: Major feature (3D semantic visualizer)

---

## TOOLS & AUTOMATION

### Development Tools

**Essential:**
- VS Code (editor)
- GitHub (version control)
- Expo (React Native development)
- Figma (UI mockups - optional)

**Nice to have:**
- Notion (task management)
- Discord (beta tester communication)
- Google Analytics (web landing page)

### Automation Scripts

**All the tools Claude built for you:**

```bash
# Process all new Midjourney assets
./scripts/process_new_assets.sh

# Run full test suite
npm test

# Build and deploy
npm run build:ios
npm run build:android

# Deploy API
vercel deploy --prod
```

### CI/CD (Future)

- GitHub Actions: Auto-run tests on PR
- Fastlane: Automate app store uploads
- Sentry: Auto-report crashes

---

## METRICS THAT MATTER

### User Acquisition
- **Installs per day** (track growth)
- **Install source** (where did they come from?)
- **Cost per install** (if running ads)

### Engagement
- **D1 retention** (came back next day?)
- **D7 retention** (came back after a week?)
- **D30 retention** (still active after a month?)
- **Session length** (how long do they play?)
- **Sessions per day** (how often do they open app?)

### Monetization
- **Conversion rate** (free → premium %)
- **ARPU** (Average Revenue Per User)
- **LTV** (Lifetime Value - how much a user is worth)
- **Churn rate** (% who cancel subscription)

### Quality
- **Crash rate** (% of sessions that crash)
- **Load time** (how fast does app start?)
- **Frame rate** (30+ FPS?)

### Targets (Realistic for Indie)

| Metric | Good | Great | Exceptional |
|---|---|---|---|
| D1 Retention | 30% | 40% | 50%+ |
| D7 Retention | 15% | 20% | 30%+ |
| Conversion | 2% | 5% | 10%+ |
| Session Length | 5 min | 10 min | 20 min+ |
| Crash Rate | < 1% | < 0.5% | < 0.1% |

---

## COMMON MISTAKES TO AVOID

### Scope Creep
❌ "Let's add multiplayer!"
✅ Ship core loop first, add features later

### Perfectionism
❌ "This button animation isn't perfect yet"
✅ Ship 80% perfect, iterate to 95% later

### Feature Bloat
❌ 50 mediocre features
✅ 5 polished, fun features

### Ignoring Feedback
❌ "Users just don't get it"
✅ "Users are confused, let's improve onboarding"

### Premature Optimization
❌ Optimize before measuring
✅ Profile, find bottleneck, then optimize

### No Marketing
❌ "Build it and they will come"
✅ Market from day 1 (Twitter, Reddit, landing page)

---

## QUICK START CHECKLIST

**Day 1:**
- [ ] Create GDD (1 page, elevator pitch + core loop)
- [ ] Set up repo (GitHub, clone, npm install)
- [ ] Create asset list (spreadsheet)
- [ ] Generate first batch of assets (Midjourney)

**Day 2-7:**
- [ ] Build vertical slice (core loop playable)
- [ ] Process assets (Claude tools)
- [ ] Integrate art and animations
- [ ] Playtest yourself

**Day 8-14:**
- [ ] Add horizontal features (P0 only)
- [ ] Polish and juice
- [ ] Beta test with 10 users
- [ ] Iterate based on feedback

**Day 15-21:**
- [ ] Bug fixes
- [ ] Performance optimization
- [ ] App store prep (screenshots, description)
- [ ] Marketing prep (landing page, socials)

**Day 22+:**
- [ ] Submit to stores
- [ ] Launch!
- [ ] Monitor, iterate, improve

---

**You're not making AAA. You're making indie gold. Ship fast, iterate faster, listen to users, and HAVE FUN.**
