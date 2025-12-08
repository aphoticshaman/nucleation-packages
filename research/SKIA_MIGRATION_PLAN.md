# Skia Migration Plan for HungryOrca (Lunatiq Tarot App)

## Executive Summary

**Problem:** Hermes JavaScript engine throws "Property doesn't exist" errors when accessing object properties during module initialization, specifically with `useWindowDimensions()` destructuring.

**Solution:** Migrate visual effects and animations from React Native Animated API to React Native Skia for high-performance canvas-based rendering that bypasses Hermes property access issues.

**Status:** Network connectivity issue blocking dependency installation. Documentation and architecture prepared for implementation when network restored.

---

## Current State Analysis

### Working (Temporary Solution)
- **File:** `src/utils/useSafeDimensions.js`
- **Approach:** Returns static dimensions `{ width: 375, height: 812 }`
- **Limitation:** Not responsive to different screen sizes
- **Purpose:** Prevents Hermes errors but not production-ready

### Root Cause of Hermes Issues

1. **Module-level execution:** Hermes evaluates code during module load before React runtime ready
2. **Property access timing:** Accessing properties on potentially undefined objects fails
3. **Destructuring failures:** `const { width } = obj` fails if obj lacks properties at module load time
4. **StyleSheet.create limitations:** Cannot reference component-level dynamic values

### Files Currently Using useSafeDimensions (Need Migration)

#### High Priority - Visual Effects (MainMenuScreen dependencies)
- `src/components/AmbientEffects.js` (FloatingParticle, LightBeams)
- `src/components/InteractiveBackground.js` (hotspots, ripples)

#### Medium Priority - Screens
- `screens/LoadingScreenNew.js`
- `screens/MainMenuScreen.js`
- `screens/ProfileScreen.js`
- `screens/CardReadingScreen.js`
- `screens/ReadingSummaryScreen.js`

#### Medium Priority - Animation Components
- `src/components/CardShuffleAnimation.js`
- `src/components/TarotCardFlip.js`
- `src/components/CardSelectionSpread.js`

#### Medium Priority - Feature Screens
- `src/screens/CardDrawingScreen.js`
- `src/screens/DeckViewerScreen.js`
- `src/screens/OracleChatScreen.js`

#### Low Priority - Card Components
- `components/TarotCard.js` (old version - may be deprecated)
- `src/components/TarotCard.js` (new version with rarity system)

---

## Target Architecture: React Native + Skia

### Why Skia?

1. **Canvas-based rendering** - Bypasses React component property access issues
2. **High performance** - Hardware-accelerated 2D graphics (60+ FPS)
3. **Native integration** - Built on Skia (Google's graphics library used in Chrome, Android)
4. **React Native compatible** - Drop-in replacement for existing animations
5. **Dimension handling** - Canvas sizing doesn't use destructured hooks

### Technology Stack

```
@shopify/react-native-skia ^1.0.0      - Core Skia rendering
react-native-reanimated@3              - Animation worklet engine
```

### Architecture Layers

```
┌─────────────────────────────────────────┐
│   React Native Screens & Navigation    │  ← Existing (keep as-is)
├─────────────────────────────────────────┤
│   Skia Canvas Components                │  ← NEW: Visual effects layer
│   - AmbientEffects (particles, beams)  │
│   - InteractiveBackground (hotspots)    │
│   - Card animations (flip, shuffle)     │
├─────────────────────────────────────────┤
│   Skia Utilities & Context             │  ← NEW: Infrastructure
│   - SkiaDimensionsProvider             │
│   - SkiaParticle base class            │
│   - Animation helpers                   │
├─────────────────────────────────────────┤
│   React Native Core (Hermes engine)    │  ← Existing
└─────────────────────────────────────────┘
```

---

## Migration Phases

### PHASE 0: Documentation & Preparation (CURRENT - No network needed) ✅

**Goal:** Prepare architecture and code structure without external dependencies

**Tasks:**
- [x] Create this migration plan document
- [ ] Create SkiaDimensionsProvider context stub
- [ ] Create test harness structure
- [ ] Analyze and document all useWindowDimensions usage
- [ ] Create migration utility helpers

**Deliverables:**
- `docs/SKIA_MIGRATION_PLAN.md` (this file)
- `src/contexts/SkiaDimensionsContext.js` (stub)
- `src/utils/skia/` directory structure
- `screens/SkiaTestScreen.js` (test harness)
- `docs/DIMENSIONS_AUDIT.md`

---

### PHASE 1: Install & Configure Dependencies (BLOCKED - Network)

**Goal:** Install Skia and Reanimated libraries

**Dependencies:**
```bash
npm install @shopify/react-native-skia
npm install react-native-reanimated@3
npx pod-install  # iOS only
```

**Configuration:**

**File:** `babel.config.js`
```javascript
module.exports = function(api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: ['react-native-reanimated/plugin'],  // ADD THIS
  };
};
```

**Test:** Create basic Skia canvas test

**File:** `screens/SkiaTestScreen.js`
```javascript
import { Canvas, Circle, Fill } from '@shopify/react-native-skia';

export default function SkiaTestScreen() {
  return (
    <Canvas style={{ flex: 1 }}>
      <Fill color="#000000" />
      <Circle cx={100} cy={100} r={50} color="#00ffff" />
    </Canvas>
  );
}
```

**Success Criteria:**
- [ ] Skia installed without errors
- [ ] Reanimated installed and configured
- [ ] SkiaTestScreen renders blue circle on black background
- [ ] No Hermes property errors

---

### PHASE 2: Create Skia Utility Infrastructure

**Goal:** Build reusable utilities and context for Skia components

#### 2.1: SkiaDimensionsProvider Context

**File:** `src/contexts/SkiaDimensionsContext.js`
```javascript
import React, { createContext, useContext } from 'react';
import { useWindowDimensions } from 'react-native';

const SkiaDimensionsContext = createContext(null);

export function SkiaDimensionsProvider({ children }) {
  const { width, height } = useWindowDimensions();

  // Safe to use here - this is React component level, not module level
  return (
    <SkiaDimensionsContext.Provider value={{ width, height }}>
      {children}
    </SkiaDimensionsContext.Provider>
  );
}

export function useSkiaDimensions() {
  const context = useContext(SkiaDimensionsContext);
  if (!context) {
    throw new Error('useSkiaDimensions must be used within SkiaDimensionsProvider');
  }
  return context;
}
```

**Usage in App.js:**
```javascript
import { SkiaDimensionsProvider } from './src/contexts/SkiaDimensionsContext';

export default function App() {
  return (
    <ErrorBoundary>
      <SkiaDimensionsProvider>
        <NavigationContainer>
          {/* screens */}
        </NavigationContainer>
      </SkiaDimensionsProvider>
    </ErrorBoundary>
  );
}
```

#### 2.2: Base Skia Particle Component

**File:** `src/utils/skia/SkiaParticle.js`
```javascript
import { Circle, useValue } from '@shopify/react-native-skia';
import { useEffect } from 'react';
import { withTiming, Easing } from 'react-native-reanimated';

export function SkiaParticle({ x, y, size, color, duration }) {
  const opacity = useValue(0);

  useEffect(() => {
    opacity.value = withTiming(1, {
      duration,
      easing: Easing.inOut(Easing.ease),
    });
  }, []);

  return (
    <Circle
      cx={x}
      cy={y}
      r={size}
      color={color}
      opacity={opacity}
    />
  );
}
```

#### 2.3: Test Harness

**File:** `screens/SkiaTestHarnessScreen.js`
```javascript
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Canvas, Fill } from '@shopify/react-native-skia';
import { useSkiaDimensions } from '../src/contexts/SkiaDimensionsContext';
import { SkiaParticle } from '../src/utils/skia/SkiaParticle';

export default function SkiaTestHarnessScreen() {
  const { width, height } = useSkiaDimensions();

  return (
    <View style={styles.container}>
      <Canvas style={styles.canvas}>
        <Fill color="#000000" />

        {/* Test particles */}
        <SkiaParticle
          x={width * 0.25}
          y={height * 0.25}
          size={20}
          color="#00ffff"
          duration={1000}
        />
        <SkiaParticle
          x={width * 0.75}
          y={height * 0.75}
          size={30}
          color="#8a2be2"
          duration={1500}
        />
      </Canvas>

      <View style={styles.overlay}>
        <Text style={styles.text}>Skia Test Harness</Text>
        <Text style={styles.info}>Screen: {width}x{height}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  canvas: { flex: 1 },
  overlay: {
    position: 'absolute',
    top: 50,
    left: 20,
    right: 20,
  },
  text: {
    color: '#00ffff',
    fontSize: 24,
    fontWeight: 'bold',
  },
  info: {
    color: '#ffffff',
    fontSize: 14,
    marginTop: 10,
  },
});
```

**Success Criteria:**
- [ ] SkiaDimensionsProvider provides dimensions to all children
- [ ] SkiaParticle renders animated particles
- [ ] Test harness displays correctly with responsive dimensions
- [ ] No console errors

---

### PHASE 3: Migrate AmbientEffects.FloatingParticle

**Goal:** Convert floating particles from React Animated to Skia

**Current Implementation Analysis:**

**File:** `src/components/AmbientEffects.js` (lines 1-120)
```javascript
// CURRENT (BROKEN):
const FloatingParticle = ({ delay = 0 }) => {
  const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = useSafeDimensions();  // Static!

  const translateY = useRef(new Animated.Value(0)).current;
  const opacity = useRef(new Animated.Value(0)).current;

  // Complex animation loop...

  return (
    <Animated.View style={[styles.particle, { transform: [{ translateY }], opacity }]}>
      <View style={styles.particleDot} />
    </Animated.View>
  );
};
```

**New Skia Implementation:**

**File:** `src/components/skia/FloatingParticleSkia.js`
```javascript
import React, { useEffect } from 'react';
import { Circle, useValue } from '@shopify/react-native-skia';
import {
  useSharedValue,
  withRepeat,
  withTiming,
  Easing,
  withDelay
} from 'react-native-reanimated';
import { useSkiaDimensions } from '../../contexts/SkiaDimensionsContext';

export function FloatingParticleSkia({ delay = 0, index = 0 }) {
  const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = useSkiaDimensions();

  // Initialize particle position
  const startX = Math.random() * SCREEN_WIDTH;
  const startY = SCREEN_HEIGHT + Math.random() * 100;

  const translateY = useSharedValue(startY);
  const opacity = useSharedValue(0);

  useEffect(() => {
    // Fade in
    opacity.value = withDelay(
      delay,
      withTiming(0.6, { duration: 1000, easing: Easing.inOut(Easing.ease) })
    );

    // Float upward infinitely
    translateY.value = withDelay(
      delay,
      withRepeat(
        withTiming(-100, {
          duration: 15000 + Math.random() * 10000,
          easing: Easing.linear,
        }),
        -1,  // Infinite
        false
      )
    );
  }, [delay]);

  return (
    <Circle
      cx={startX}
      cy={translateY}
      r={2 + Math.random() * 3}  // Size variation
      color="#00ffff"
      opacity={opacity}
    />
  );
}
```

**Integration:**

**File:** `src/components/AmbientEffects.js` (update MainMenuAmbience)
```javascript
import { Canvas, Fill } from '@shopify/react-native-skia';
import { FloatingParticleSkia } from './skia/FloatingParticleSkia';

export const MainMenuAmbience = () => {
  return (
    <View style={styles.container}>
      <Canvas style={StyleSheet.absoluteFillObject}>
        <Fill color="transparent" />

        {/* Render 20 floating particles */}
        {Array.from({ length: 20 }).map((_, i) => (
          <FloatingParticleSkia
            key={i}
            index={i}
            delay={i * 200}  // Stagger appearance
          />
        ))}
      </Canvas>

      {/* Keep other effects for now */}
      <LightBeams />
    </View>
  );
};
```

**Testing:**
1. Run app on iPhone
2. Navigate to MainMenuScreen
3. Verify particles float upward smoothly
4. Verify no Hermes errors
5. Check performance (should be 60 FPS)

**Success Criteria:**
- [ ] Particles render on Skia canvas
- [ ] Smooth floating animation
- [ ] Responsive to screen size (via useSkiaDimensions)
- [ ] No property access errors
- [ ] 60 FPS performance

---

### PHASE 4: Migrate AmbientEffects.LightBeams

**Goal:** Convert light beams from React Animated to Skia Path rendering

**Current Implementation:**

**File:** `src/components/AmbientEffects.js` (lines 210-330)
```javascript
const LightBeams = () => {
  const { width, height } = useSafeDimensions();  // Static!

  // Creates 3 animated beam components
  // Uses LinearGradient and Animated.View
};
```

**New Skia Implementation:**

**File:** `src/components/skia/LightBeamsSkia.js`
```javascript
import React, { useEffect } from 'react';
import {
  Path,
  LinearGradient,
  vec,
  useValue
} from '@shopify/react-native-skia';
import {
  useSharedValue,
  withRepeat,
  withTiming,
  Easing
} from 'react-native-reanimated';
import { useSkiaDimensions } from '../../contexts/SkiaDimensionsContext';

function LightBeamSkia({ index, startX, width: beamWidth }) {
  const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = useSkiaDimensions();

  const opacity = useSharedValue(0);

  useEffect(() => {
    opacity.value = withRepeat(
      withTiming(0.3, {
        duration: 3000 + index * 500,
        easing: Easing.inOut(Easing.ease),
      }),
      -1,
      true  // Reverse
    );
  }, [index]);

  // Create trapezoid path for beam
  const path = `
    M ${startX} 0
    L ${startX + beamWidth} 0
    L ${startX + beamWidth * 1.5} ${SCREEN_HEIGHT}
    L ${startX - beamWidth * 0.5} ${SCREEN_HEIGHT}
    Z
  `;

  return (
    <Path path={path} opacity={opacity}>
      <LinearGradient
        start={vec(startX, 0)}
        end={vec(startX, SCREEN_HEIGHT)}
        colors={['rgba(138, 43, 226, 0.8)', 'rgba(138, 43, 226, 0)']}
      />
    </Path>
  );
}

export function LightBeamsSkia() {
  const { width: SCREEN_WIDTH } = useSkiaDimensions();

  const beams = [
    { startX: SCREEN_WIDTH * 0.2, width: 80 },
    { startX: SCREEN_WIDTH * 0.5, width: 100 },
    { startX: SCREEN_WIDTH * 0.75, width: 60 },
  ];

  return (
    <>
      {beams.map((beam, index) => (
        <LightBeamSkia
          key={index}
          index={index}
          startX={beam.startX}
          width={beam.width}
        />
      ))}
    </>
  );
}
```

**Integration:**

Update `MainMenuAmbience` to use both Skia components:
```javascript
export const MainMenuAmbience = () => {
  return (
    <View style={styles.container}>
      <Canvas style={StyleSheet.absoluteFillObject}>
        <Fill color="transparent" />

        {/* Light beams (background layer) */}
        <LightBeamsSkia />

        {/* Floating particles (foreground layer) */}
        {Array.from({ length: 20 }).map((_, i) => (
          <FloatingParticleSkia key={i} index={i} delay={i * 200} />
        ))}
      </Canvas>
    </View>
  );
};
```

**Success Criteria:**
- [ ] Light beams render as gradient trapezoids
- [ ] Smooth pulsing opacity animation
- [ ] Layered correctly (beams behind particles)
- [ ] Responsive to screen size
- [ ] No Hermes errors

---

### PHASE 5: Migrate InteractiveBackground

**Goal:** Convert interactive hotspots and ripple effects to Skia

**Current Implementation:**

**File:** `src/components/InteractiveBackground.js`
```javascript
const InteractiveBackground = ({ onHotspotActivate }) => {
  const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = useSafeDimensions();  // Static!

  const hotspots = [
    { id: 'moon', x: SCREEN_WIDTH * 0.15, y: SCREEN_HEIGHT * 0.12, size: 60 },
    // ... more hotspots
  ];

  // Animated ripple effects on touch
};
```

**New Skia Implementation:**

**File:** `src/components/skia/InteractiveBackgroundSkia.js`
```javascript
import React, { useState } from 'react';
import { Canvas, Circle, Group } from '@shopify/react-native-skia';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import {
  useSharedValue,
  withTiming,
  Easing
} from 'react-native-reanimated';
import { useSkiaDimensions } from '../../contexts/SkiaDimensionsContext';

function HotspotSkia({ hotspot, onActivate }) {
  const [isActive, setIsActive] = useState(false);
  const rippleScale = useSharedValue(0);
  const rippleOpacity = useSharedValue(0);

  const handlePress = () => {
    setIsActive(true);
    rippleScale.value = 0;
    rippleOpacity.value = 0.6;

    rippleScale.value = withTiming(2, { duration: 600, easing: Easing.out(Easing.ease) });
    rippleOpacity.value = withTiming(0, { duration: 600, easing: Easing.out(Easing.ease) });

    onActivate(hotspot.id);

    setTimeout(() => setIsActive(false), 600);
  };

  return (
    <Group>
      {/* Main hotspot circle */}
      <Circle
        cx={hotspot.x}
        cy={hotspot.y}
        r={hotspot.size / 2}
        color="rgba(0, 255, 255, 0.2)"
      />

      {/* Ripple effect */}
      {isActive && (
        <Circle
          cx={hotspot.x}
          cy={hotspot.y}
          r={hotspot.size * rippleScale.value}
          color="#00ffff"
          opacity={rippleOpacity.value}
          style="stroke"
          strokeWidth={2}
        />
      )}
    </Group>
  );
}

export function InteractiveBackgroundSkia({ onHotspotActivate }) {
  const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = useSkiaDimensions();

  const hotspots = [
    { id: 'moon', x: SCREEN_WIDTH * 0.15, y: SCREEN_HEIGHT * 0.12, size: 60 },
    { id: 'star1', x: SCREEN_WIDTH * 0.75, y: SCREEN_HEIGHT * 0.20, size: 40 },
    { id: 'star2', x: SCREEN_WIDTH * 0.85, y: SCREEN_HEIGHT * 0.45, size: 35 },
    { id: 'crystal', x: SCREEN_WIDTH * 0.50, y: SCREEN_HEIGHT * 0.75, size: 50 },
  ];

  const tap = Gesture.Tap().onEnd((event) => {
    const { x, y } = event;

    // Find if tap hit any hotspot
    hotspots.forEach((hotspot) => {
      const distance = Math.sqrt(
        Math.pow(x - hotspot.x, 2) + Math.pow(y - hotspot.y, 2)
      );

      if (distance < hotspot.size / 2) {
        onHotspotActivate(hotspot.id);
      }
    });
  });

  return (
    <GestureDetector gesture={tap}>
      <Canvas style={{ flex: 1 }}>
        {hotspots.map((hotspot) => (
          <HotspotSkia
            key={hotspot.id}
            hotspot={hotspot}
            onActivate={onHotspotActivate}
          />
        ))}
      </Canvas>
    </GestureDetector>
  );
}
```

**Success Criteria:**
- [ ] Hotspots render and are tappable
- [ ] Ripple effects trigger on touch
- [ ] Responsive positioning (percentage-based)
- [ ] No Hermes errors

---

### PHASE 6: Migrate Card Animations

**Goal:** Convert TarotCardFlip, CardShuffleAnimation, CardSelectionSpread to Skia

#### 6.1: TarotCardFlip

**Challenge:** Skia doesn't support 3D transforms directly. Use 2D perspective simulation.

**Strategy:**
- Use scale + skew to simulate flip
- Swap front/back image at 50% rotation
- Use opacity for smooth transition

**File:** `src/components/skia/TarotCardFlipSkia.js`
```javascript
import React, { useEffect } from 'react';
import { Group, Image, useImage } from '@shopify/react-native-skia';
import {
  useSharedValue,
  withTiming,
  Easing,
  interpolate
} from 'react-native-reanimated';

export function TarotCardFlipSkia({
  frontImage,
  backImage,
  width = 200,
  height = 300,
  flipped = false
}) {
  const rotation = useSharedValue(0);

  const frontImageObj = useImage(frontImage);
  const backImageObj = useImage(backImage);

  useEffect(() => {
    rotation.value = withTiming(flipped ? 180 : 0, {
      duration: 800,
      easing: Easing.inOut(Easing.ease),
    });
  }, [flipped]);

  const scaleX = interpolate(
    rotation.value,
    [0, 90, 180],
    [1, 0, 1]
  );

  const showFront = rotation.value < 90;

  return (
    <Group transform={[{ scaleX }]}>
      {showFront ? (
        <Image
          image={frontImageObj}
          x={0}
          y={0}
          width={width}
          height={height}
        />
      ) : (
        <Image
          image={backImageObj}
          x={0}
          y={0}
          width={width}
          height={height}
        />
      )}
    </Group>
  );
}
```

#### 6.2: CardShuffleAnimation

**Current:** Uses Animated.timing with random positions

**New:** Use Skia Images with Reanimated shared values

**File:** `src/components/skia/CardShuffleAnimationSkia.js`
```javascript
import React, { useEffect } from 'react';
import { Group, Image, useImage } from '@shopify/react-native-skia';
import {
  useSharedValue,
  withRepeat,
  withSequence,
  withTiming,
  Easing
} from 'react-native-reanimated';
import { useSkiaDimensions } from '../../contexts/SkiaDimensionsContext';

function ShufflingCard({ index, cardImage }) {
  const { width: SCREEN_WIDTH } = useSkiaDimensions();

  const x = useSharedValue(SCREEN_WIDTH / 2);
  const y = useSharedValue(100);
  const rotation = useSharedValue(0);

  const image = useImage(cardImage);

  useEffect(() => {
    // Random shuffle movement
    x.value = withRepeat(
      withSequence(
        withTiming(Math.random() * SCREEN_WIDTH, { duration: 500 }),
        withTiming(SCREEN_WIDTH / 2, { duration: 500 })
      ),
      -1,
      false
    );

    rotation.value = withRepeat(
      withSequence(
        withTiming(-15, { duration: 250 }),
        withTiming(15, { duration: 500 }),
        withTiming(0, { duration: 250 })
      ),
      -1,
      false
    );
  }, []);

  return (
    <Group transform={[{ translateX: x }, { translateY: y }, { rotate: rotation }]}>
      <Image image={image} x={-50} y={-75} width={100} height={150} />
    </Group>
  );
}

export function CardShuffleAnimationSkia({ cards }) {
  return (
    <Group>
      {cards.map((card, index) => (
        <ShufflingCard key={index} index={index} cardImage={card.image} />
      ))}
    </Group>
  );
}
```

**Success Criteria:**
- [ ] Card flip animation smooth
- [ ] Shuffle animation shows multiple cards moving
- [ ] No Hermes errors
- [ ] 60 FPS performance

---

### PHASE 7: Update useSafeDimensions

**Goal:** Replace static dimensions with proper responsive sizing via Skia context

**Current Implementation:**

**File:** `src/utils/useSafeDimensions.js`
```javascript
export function useSafeDimensions() {
  // TEMPORARY: Return static dimensions
  return { width: 375, height: 812 };
}
```

**New Implementation:**

```javascript
/**
 * useSafeDimensions - Now uses SkiaDimensionsContext
 *
 * DEPRECATED: Use useSkiaDimensions() directly instead.
 * This wrapper maintained for backward compatibility during migration.
 */

import { useSkiaDimensions } from '../contexts/SkiaDimensionsContext';

export function useSafeDimensions() {
  console.warn(
    'useSafeDimensions is deprecated. Use useSkiaDimensions from SkiaDimensionsContext instead.'
  );

  return useSkiaDimensions();
}
```

**Refactor Strategy:**

1. Update all files to import `useSkiaDimensions` instead of `useSafeDimensions`
2. Run comprehensive grep to ensure no missed instances
3. Delete `useSafeDimensions.js` after confirming all migrations complete

**Success Criteria:**
- [ ] All components responsive to actual screen size
- [ ] No console warnings about deprecated hook
- [ ] useSafeDimensions.js deleted
- [ ] No Hermes errors

---

### PHASE 8: Full Integration Testing

**Goal:** Test all screens and components on multiple devices

#### 8.1: iPhone Testing

**Devices:**
- iPhone 14 Pro (6.1" OLED)
- iPhone SE (4.7" LCD)

**Test Matrix:**

| Screen | FloatingParticle | LightBeams | InteractiveBackground | Card Animations | Pass/Fail |
|--------|-----------------|------------|---------------------|-----------------|-----------|
| LoadingScreenNew | N/A | N/A | N/A | N/A | [ ] |
| MainMenuScreen | ✓ | ✓ | ✓ | N/A | [ ] |
| ProfileScreen | N/A | N/A | N/A | N/A | [ ] |
| CardReadingScreen | N/A | N/A | N/A | ✓ | [ ] |
| ReadingSummaryScreen | N/A | N/A | N/A | ✓ | [ ] |
| CardDrawingScreen | ✓ | N/A | N/A | ✓ | [ ] |

**Test Cases:**
1. Launch app → LoadingScreenNew → MainMenuScreen
2. Verify particles floating smoothly
3. Verify light beams pulsing
4. Tap interactive hotspots → verify ripple effects
5. Navigate to CardReadingScreen → draw card → verify flip animation
6. Rotate device → verify responsive layout
7. Background app → foreground → verify animations resume

#### 8.2: Android Testing

**Devices:**
- Pixel 7 (6.3" OLED)
- Samsung Galaxy S21 (6.2" AMOLED)

**Same test matrix as iPhone**

#### 8.3: Performance Profiling

**Metrics:**
- FPS (target: 60 FPS constant)
- Memory usage (target: < 200 MB)
- Battery drain (target: < 5% per 10 min)

**Tools:**
- React Native Performance Monitor
- Flipper profiler
- Xcode Instruments (iOS)
- Android Profiler (Android)

**Success Criteria:**
- [ ] All screens render correctly on iPhone SE (smallest screen)
- [ ] All screens render correctly on iPad (largest screen)
- [ ] All animations 60 FPS
- [ ] No Hermes errors on any device
- [ ] No memory leaks
- [ ] Battery usage acceptable

---

### PHASE 9: Documentation & Cleanup

**Goal:** Document patterns and remove deprecated code

#### 9.1: Create Developer Documentation

**File:** `docs/SKIA_DEVELOPMENT_GUIDE.md`

Contents:
- How to create new Skia components
- Animation patterns with Reanimated
- Responsive design best practices
- Performance optimization tips
- Common pitfalls and solutions

#### 9.2: Remove Deprecated Code

Files to delete:
- `src/utils/useSafeDimensions.js` (replaced by useSkiaDimensions)
- `components/TarotCard.js` (old version - if confirmed unused)
- Any temporary test screens

Files to update:
- Remove old Animated API imports from migrated components
- Clean up console.log statements
- Remove commented-out code

#### 9.3: Update README

**File:** `README.md`

Add sections:
- Architecture overview (React Native + Skia)
- Visual effects system
- Animation patterns
- Performance characteristics

**Success Criteria:**
- [ ] SKIA_DEVELOPMENT_GUIDE.md complete
- [ ] All deprecated files removed
- [ ] No unused imports
- [ ] README updated
- [ ] All commits pushed to branch

---

## Rollback Plan

If Skia migration fails or causes issues:

1. **Revert to static dimensions:**
   ```bash
   git revert <commit-hash-range>
   ```

2. **Restore useSafeDimensions with static values:**
   ```javascript
   export function useSafeDimensions() {
     return { width: 375, height: 812 };
   }
   ```

3. **Document issues** in GitHub issue for future investigation

4. **Alternative approaches:**
   - Ship with static dimensions (iPhone-only app initially)
   - Pivot to native Swift/Kotlin with compose-multiplatform
   - Use React Native Web (JSC engine instead of Hermes)

---

## Network Dependency Blocker

**Current Status:** Cannot install npm packages due to DNS resolution failure (`EAI_AGAIN github.com`)

**Attempted:** 4 retries with exponential backoff (2s, 4s, 8s delays)

**Blocking Tasks:**
- PHASE 1.1: Install @shopify/react-native-skia
- PHASE 1.2: Install react-native-reanimated@3

**Workaround:**
- Complete PHASE 0 (documentation and architecture)
- Prepare stub implementations
- Wait for network connectivity restoration

**When Network Restored:**
```bash
# Run these commands
npm install @shopify/react-native-skia
npm install react-native-reanimated@3
npx pod-install  # iOS only

# Clear cache
rm -rf node_modules/.cache .expo
npx expo start --clear

# Test basic rendering
# Navigate to SkiaTestScreen
```

---

## Success Metrics

### Technical Metrics
- ✅ Zero Hermes property access errors
- ✅ 60 FPS on all screens
- ✅ Responsive to all screen sizes (iPhone SE to iPad Pro)
- ✅ < 200 MB memory usage
- ✅ All visual effects working

### Development Metrics
- ✅ All phases completed
- ✅ All tests passing
- ✅ Documentation complete
- ✅ Code committed and pushed
- ✅ Ready for production deployment

### User Experience Metrics
- ✅ App launches without errors
- ✅ Smooth, beautiful visual effects
- ✅ Responsive interactions
- ✅ No crashes or freezes

---

## Timeline Estimate

**PHASE 0:** 2 hours (Documentation) - CURRENT
**PHASE 1:** 1 hour (Dependencies) - BLOCKED
**PHASE 2:** 2 hours (Utilities)
**PHASE 3:** 3 hours (AmbientEffects)
**PHASE 4:** 2 hours (InteractiveBackground)
**PHASE 5:** 4 hours (Card animations)
**PHASE 6:** 2 hours (useSafeDimensions)
**PHASE 7:** 4 hours (Testing)
**PHASE 8:** 2 hours (Documentation)

**Total:** ~22 hours of development + testing time

---

## Questions for User (When They Return)

1. **Network Access:** Is the environment expected to have GitHub access for npm installs?
2. **Testing Devices:** Which devices should be prioritized for testing?
3. **Deployment Timeline:** When does this need to be production-ready?
4. **Feature Scope:** Are all visual effects critical, or can we prioritize certain screens?
5. **Performance Targets:** Are there specific FPS or battery usage requirements?

---

## Next Steps (When Network Restored)

1. ✅ Install Skia and Reanimated dependencies
2. ✅ Configure babel.config.js
3. ✅ Create SkiaTestScreen and verify rendering
4. ✅ Implement SkiaDimensionsProvider
5. ✅ Migrate AmbientEffects.FloatingParticle
6. ✅ Test on iPhone
7. ✅ Continue with remaining phases

---

**Document Status:** Living document - will be updated as implementation progresses
**Last Updated:** 2025-11-20 (PHASE 0 - Network blocked)
**Author:** Claude Code
**Branch:** `claude/catalog-and-sync-repos-01TrrmUDEmT3znrryD57LozM`
