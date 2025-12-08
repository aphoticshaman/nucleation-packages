# Cleanup and Optimization Plan

## Priority 1: Remove Dead Code (Immediate)

### Screens to Archive/Remove
```
src/screens/WelcomeScreen.js          → Old onboarding flow
src/screens/OnboardingScreen.js       → Old onboarding flow
src/screens/ProfileSetupScreen.js     → Old profile creation
src/screens/PersonalityQuestionsScreen.js → Old MBTI flow
src/screens/MBTITestScreen.js         → Old MBTI flow
src/screens/BirthdayScreen.js         → Old astrology flow
src/screens/LoadingScreen.js          → Replaced by LoadingScreenNew
src/screens/ReadingTypeScreen.js      → Conflicting with game flow
src/screens/IntentionScreen.js        → Conflicting with game flow
src/screens/CardDrawingScreen.js      → Conflicting with game flow
```

**Action:** Move to `/archive/old_screens/` and remove from App.js

### Keep But Redesign
```
src/screens/SettingsScreen.js         → Redesign with dark fantasy aesthetic
src/screens/ReadingHistoryScreen.js   → Rename to JournalScreen, redesign
src/screens/AchievementsScreen.js     → Redesign with game aesthetic
src/screens/DeckViewerScreen.js       → Redesign as "Arcana Codex"
src/screens/StatsScreen.js            → Integrate into profile screen
src/screens/OracleChatScreen.js       → Keep (LLM chat with Oracle)
```

## Priority 2: Color Palette Overhaul

### Old Colors (Remove)
```javascript
const OLD_PALETTE = {
  primary: '#8a2be2',   // Bright purple
  accent: '#00ffff',    // Cyan
  glow: '#8a2be2',      // Bright purple
};
```

### New Colors (Implement)
```javascript
const DARK_FANTASY_PALETTE = {
  // Base colors
  deepPurple: '#2a1a4a',      // Primary backgrounds
  richGold: '#d4af37',        // Accent highlights
  voidBlack: '#0a0a0f',       // Pure black backgrounds
  shadowPurple: '#1a0d2e',    // Darker purple

  // Accent colors
  mysticGold: '#f4e4b8',      // Light gold (text)
  amberGlow: '#ffb347',       // Orange-gold (particles)
  royalPurple: '#4a2b6a',     // Medium purple (borders)

  // Effects
  glitterGold: '#ffe55c',     // Sparkle highlights
  faeMist: 'rgba(212, 175, 55, 0.3)',  // Gold mist overlay
  shadowVeil: 'rgba(26, 13, 46, 0.7)', // Purple shadow
};
```

### Files to Update
- `screens/LoadingScreenNew.js` - Replace cyan with gold
- `screens/MainMenuScreen.js` - Replace cyan with gold
- `screens/IsometricVillageScreen.js` - Update dialog colors
- `screens/OracleShopScreen.js` - Update UI colors
- `components/TarotCard.js` - Gold accents instead of cyan
- `components/CardInterpretationDialog.js` - Gold text, purple gradients
- `styles/colors.js` (create) - Centralized color constants

## Priority 3: Button System

### Create Image-Based Button Component
```javascript
// components/ImageButton.js
export default function ImageButton({
  baseImage,      // 'assets/buttons/begin_journey'
  onPress,
  disabled = false,
  style
}) {
  const [pressed, setPressed] = useState(false);

  const getImageSource = () => {
    if (disabled) return require(`${baseImage}_disabled.png`);
    if (pressed) return require(`${baseImage}_pressed.png`);
    return require(`${baseImage}_normal.png`);
  };

  return (
    <TouchableOpacity
      onPressIn={() => setPressed(true)}
      onPressOut={() => setPressed(false)}
      onPress={onPress}
      disabled={disabled}
      activeOpacity={1}
    >
      <Image source={getImageSource()} style={style} />
    </TouchableOpacity>
  );
}
```

### Button Assets Needed (Midjourney)
```
assets/buttons/
├── begin_journey_normal.png    (gold ornate button)
├── begin_journey_pressed.png   (darker, indented)
├── begin_journey_disabled.png  (grayed out)
├── continue_normal.png
├── continue_pressed.png
├── settings_normal.png
├── settings_pressed.png
├── journal_normal.png
├── journal_pressed.png
├── draw_card_normal.png
├── draw_card_pressed.png
└── ask_oracle_normal.png
```

**Midjourney Prompt Template:**
```
ornate fantasy game button, dark purple wood frame with gold trim,
engraved text "[BUTTON TEXT]", art nouveau style, mystical runes in
corners, slight emboss effect, game UI asset, transparent background,
dark fantasy aesthetic, 400x120px --ar 10:3 --v 6
```

## Priority 4: Particle Effects System

### Particle Sprites Needed
```
assets/particles/
├── sparkle_00.png - sparkle_15.png    (16-frame sparkle animation)
├── glow_pulse.gif                      (pulsing gold orb)
├── fae_trail_00.png - fae_trail_07.png (8-frame trailing mist)
├── glitter_bubble.gif                  (floating shimmer bubble)
├── mist_overlay.gif                    (ambient mist layer)
└── phosphor_dust.gif                   (falling glitter particles)
```

### Particle System Component
```javascript
// components/ParticleEmitter.js
export default function ParticleEmitter({
  particleType = 'sparkle',
  emissionRate = 5,        // particles per second
  maxParticles = 50,
  bounds = screenBounds,
  enabled = true
}) {
  // Particle pool with recycling
  // Velocity, gravity, fade-out
  // Culling for off-screen particles
}
```

### Generate with Python
```bash
python tools/generative_art.py sparkle \
  --frames 16 \
  --size 64 \
  --colors "#d4af37,#ffe55c,#f4e4b8" \
  --glow-intensity 0.8

python tools/generative_art.py fae-trail \
  --frames 8 \
  --size 128 \
  --colors "#d4af37,rgba(212,175,55,0.3)" \
  --trail-length 200
```

## Priority 5: Menu System Redesign

### New Menu Structure
```
Game Flow:
LoadingNew → MainMenu → Village → OracleShop → OracleChat

Main Menu Options:
├── Begin Journey (→ Village)
├── Journal
│   └── JournalScreen (redesigned ReadingHistory)
├── Profile
│   └── ProfileScreen (stats + achievements combined)
├── Arcana Codex
│   └── DeckViewerScreen (redesigned)
└── Settings
    └── SettingsScreen (redesigned)
```

### Menu Screens to Create
```javascript
// screens/JournalScreen.js (redesign ReadingHistoryScreen)
- Dark fantasy book interface
- Gold-trimmed pages
- Saved readings with card thumbnails
- Filter by date, card, spread type

// screens/ProfileScreen.js (new, combines Stats + Achievements)
- Character portrait placeholder
- Reading statistics
- Unlocked achievements grid
- Personality insights from readings

// screens/ArcanCodexScreen.js (redesign DeckViewerScreen)
- Browse all 78 cards
- Tap to see full interpretation
- Track which cards drawn
- "Mastery" progress per card
```

## Priority 6: Hardware Detection & Optimization

### Device Capability Detection
```javascript
// utils/deviceCapabilities.js
import { Platform, Dimensions } from 'react-native';
import * as Device from 'expo-device';

export async function detectCapabilities() {
  const { width, height } = Dimensions.get('window');
  const totalPixels = width * height;

  // Estimate device tier
  let tier = 'high';

  if (Platform.OS === 'web') {
    tier = 'high'; // Assume desktop has good performance
  } else if (totalPixels < 1280 * 720) {
    tier = 'low';  // Old phones
  } else if (totalPixels < 1920 * 1080) {
    tier = 'medium';
  }

  // Get device info
  const deviceYear = await Device.deviceYearClass; // Expo provides this
  if (deviceYear && deviceYear < 2018) tier = 'low';
  if (deviceYear && deviceYear < 2020 && tier === 'high') tier = 'medium';

  return {
    tier,
    screenWidth: width,
    screenHeight: height,
    platform: Platform.OS,
    isLowEnd: tier === 'low',
    isMediumEnd: tier === 'medium',
    isHighEnd: tier === 'high',
  };
}

export function getOptimalSettings(capabilities) {
  const settings = {
    maxParticles: capabilities.isHighEnd ? 100 : capabilities.isMediumEnd ? 50 : 20,
    particleQuality: capabilities.isHighEnd ? 'high' : 'medium',
    enableGlow: !capabilities.isLowEnd,
    enableBlur: capabilities.isHighEnd,
    cardImageQuality: capabilities.isHighEnd ? 'full' : 'compressed',
    animationFrameRate: capabilities.isLowEnd ? 30 : 60,
    useNativeDriver: true, // Always true for performance
  };

  return settings;
}
```

### Asset Quality Tiers
```
assets/
├── cards/
│   ├── full/      (2048x3072, ~300KB each, for high-end devices)
│   ├── medium/    (1024x1536, ~100KB each, for medium devices)
│   └── low/       (512x768, ~30KB each, for low-end devices)
├── backgrounds/
│   ├── full/      (1920x1080)
│   ├── medium/    (1280x720)
│   └── low/       (854x480)
└── particles/
    ├── full/      (PNG with alpha, high quality)
    └── low/       (Reduced color depth, smaller)
```

### Sprite Atlas for Cards
```javascript
// Instead of loading 78 individual images:
import { Asset } from 'expo-asset';

// Create sprite atlas (one large image with all 78 cards)
const CARD_ATLAS = require('./assets/cards/atlas_4k.png');
const CARD_POSITIONS = {
  'the-fool': { x: 0, y: 0, width: 200, height: 300 },
  'the-magician': { x: 200, y: 0, width: 200, height: 300 },
  // ... all 78 cards
};

// Extract card from atlas at runtime (much faster than 78 separate loads)
function getCardFromAtlas(cardId) {
  const pos = CARD_POSITIONS[cardId];
  return {
    uri: CARD_ATLAS,
    crop: { x: pos.x, y: pos.y, width: pos.width, height: pos.height }
  };
}
```

### Lazy Loading
```javascript
// Only load assets when needed
import { Asset } from 'expo-asset';

class AssetManager {
  constructor() {
    this.loaded = new Set();
  }

  async preloadCritical() {
    // Load only what's needed for loading screen + main menu
    await Asset.loadAsync([
      require('./assets/backgrounds/loading_bg.png'),
      require('./assets/ui/logo.png'),
      require('./assets/buttons/begin_journey_normal.png'),
    ]);
  }

  async loadVillageAssets() {
    // Load village assets when entering village
    if (this.loaded.has('village')) return;

    await Asset.loadAsync([
      require('./assets/backgrounds/village_isometric.png'),
      require('./assets/npcs/merchant.png'),
      // ...
    ]);

    this.loaded.add('village');
  }

  async loadOracleAssets() {
    // Load oracle + card assets when entering oracle shop
    if (this.loaded.has('oracle')) return;

    await Asset.loadAsync([
      require('./assets/backgrounds/oracle_shop.png'),
      require('./assets/cards/atlas_medium.png'),
      // ...
    ]);

    this.loaded.add('oracle');
  }
}

export default new AssetManager();
```

### Frame Rate Control
```javascript
// utils/performanceMonitor.js
import { InteractionManager } from 'react-native';

class PerformanceMonitor {
  constructor() {
    this.frameTime = 16.67; // 60fps target
    this.lastFrameTime = Date.now();
    this.isLagging = false;
  }

  startFrame() {
    const now = Date.now();
    const delta = now - this.lastFrameTime;

    // If frame took >33ms (under 30fps), we're lagging
    if (delta > 33) {
      this.isLagging = true;
      this.reduceQuality();
    } else {
      this.isLagging = false;
    }

    this.lastFrameTime = now;
  }

  reduceQuality() {
    // Dynamically reduce particle count, disable glow, etc.
    EventEmitter.emit('reduce-quality');
  }
}

export default new PerformanceMonitor();
```

## Priority 7: Implementation Order

### Week 1: Cleanup & Color Update
- [ ] Move old screens to `/archive/`
- [ ] Update App.js to remove archived screens
- [ ] Create `styles/colors.js` with new palette
- [ ] Update all existing screens to use new colors
- [ ] Test navigation flow (Loading → MainMenu → Village → Oracle)

### Week 2: Button System & Particles
- [ ] Create `ImageButton` component
- [ ] Generate button assets (Midjourney + Python)
- [ ] Update all screens to use `ImageButton`
- [ ] Generate particle sprites (sparkles, glitter, fae trails)
- [ ] Create `ParticleEmitter` component
- [ ] Add particles to LoadingNew, MainMenu, OracleShop

### Week 3: Menu Redesign
- [ ] Redesign JournalScreen (dark fantasy book)
- [ ] Create ProfileScreen (character sheet aesthetic)
- [ ] Redesign ArcanCodexScreen (mystical tome)
- [ ] Redesign SettingsScreen (ornate panel)
- [ ] Update navigation to new menu structure

### Week 4: Performance Optimization
- [ ] Implement device capability detection
- [ ] Create asset quality tiers (low/medium/full)
- [ ] Build sprite atlas for tarot cards
- [ ] Implement lazy loading with AssetManager
- [ ] Add performance monitoring
- [ ] Test on low-end devices

## File Size Targets

### REVISED: Quality First Approach

**Philosophy:** Prioritize visual fidelity over file size. Optimize AFTER art is complete.

### Realistic App Size (Quality-Focused)
```
Total app size: ~600-800MB (acceptable for modern devices)
├── Tarot cards (78 × 2-4MB):   200-300 MB (full resolution, high quality)
├── Village assets:              80-120 MB (isometric scenes, NPCs, buildings)
├── Backgrounds (10-15 × 5MB):   50-75 MB (4K mystical scenes)
├── Particle effects:            40-60 MB (high-quality animated sprites)
├── Button/UI assets:            30-50 MB (ornate frames, borders, icons)
├── Audio (music + SFX):         50-80 MB (ambient, card shuffle, mystical)
├── Game engine + dependencies:  70-100 MB (React Native, Expo, libraries)
```

**Strategy:**
1. Build with FULL QUALITY assets (2K-4K images, high-res particles)
2. Implement lazy loading (download village assets when entering village)
3. Cloud-based asset delivery (optional DLC: extra spreads, alt art)
4. File size optimization comes LATER (after MVP is visually stunning)

**Target:** 500-800MB full app, runs smoothly on 2020+ devices with 4GB+ RAM

## Performance Metrics

### Target Frame Rates
- High-end devices: 60fps constant
- Medium devices: 60fps normal, 45fps with heavy particles
- Low-end devices: 30fps locked (disable expensive effects)

### Load Times
- Loading screen → Main menu: <2 seconds
- Main menu → Village: <1 second (assets preloaded)
- Village → Oracle: <1 second (lazy load cards)
- Card draw animation: <500ms (feels instant)

### Memory Usage
- High-end: <200MB RAM
- Medium: <150MB RAM
- Low-end: <100MB RAM (aggressive asset unloading)
