# ANIMATION RIGGING FOR GAME ASSETS

**From Static Sprites to Animated Game Characters and Effects**

## Types of Game Animations

### 1. Character Animations
- **Idle**: Breathing, subtle movements (looping)
- **Walk/Run**: Movement cycle (looping)
- **Jump**: Take off → air → landing (one-shot)
- **Attack/Action**: Special moves (one-shot)
- **Emote**: Expressions, gestures (one-shot or loop)

### 2. Background Ambient Animations
- **Particles**: Floating dust, sparkles, magic
- **Environment**: Flickering candles, flowing water, swaying plants
- **Weather**: Rain, snow, fog
- **UI Effects**: Glowing buttons, pulsing icons

### 3. Effect Animations
- **Impact**: Hits, explosions, collisions
- **Projectiles**: Fireballs, arrows, magic bolts
- **Status**: Healing, poison, buffs
- **Transitions**: Fade in/out, warps, teleports

---

## PART 1: Creating Sprite Sheet Animations

### Sprite Sheet Layout

**Horizontal Strip** (Simple, Best for Web/Mobile):
```
[Frame 0][Frame 1][Frame 2][Frame 3][Frame 4][Frame 5][Frame 6][Frame 7]
```
- 2048x256px total (8 frames × 256px each)
- Easy to code (just change X offset)
- Good for simple loops

**Grid Layout** (Multiple Animations):
```
Row 0: [Idle 0][Idle 1][Idle 2][Idle 3][Idle 4][Idle 5][Idle 6][Idle 7]
Row 1: [Walk 0][Walk 1][Walk 2][Walk 3][Walk 4][Walk 5][Walk 6][Walk 7]
Row 2: [Run 0 ][Run 1 ][Run 2 ][Run 3 ][Run 4 ][Run 5 ][Run 6 ][Run 7 ]
Row 3: [Jump 0][Jump 1][Jump 2][Jump 3][Jump 4][Jump 5][Jump 6][Jump 7]
```
- 2048x1024px (8 columns × 4 rows × 256px frames)
- Multiple animations in one file
- More complex code but fewer assets

### Frame Count Guidelines

| Animation Type | Frame Count | FPS | Loop |
|---|---|---|---|
| **Idle (subtle)** | 4-6 | 10-12 | Yes |
| **Idle (active)** | 8-12 | 15-20 | Yes |
| **Walk** | 6-8 | 18-24 | Yes |
| **Run** | 8-10 | 24-30 | Yes |
| **Jump** | 6-10 | 24 | No |
| **Attack** | 8-16 | 24-30 | No |
| **Particles** | 8-16 | 30-60 | Yes |
| **Effects** | 12-24 | 30-60 | No |

---

## PART 2: Character Idle Animation

### Design Principles

**Idle = Character is alive, not frozen**

Good idle animations have:
1. **Breathing**: Chest/shoulders rise and fall
2. **Weight shift**: Slight sway side to side
3. **Micro-movements**: Blink, hair sway, cloth flutter
4. **Asymmetry**: Not perfectly mirrored (looks robotic)

### Frame Breakdown (8-frame idle loop)

```
Frame 0: Neutral pose (resting)
Frame 1: Inhale start (chest expands slightly)
Frame 2: Peak inhale (chest full)
Frame 3: Exhale start (chest contracts)
Frame 4: Neutral (back to resting)
Frame 5: Weight shift left (subtle lean)
Frame 6: Return to center
Frame 7: Weight shift right (subtle lean)
```

**Timing**: 10-12 FPS (slow, subtle breathing)

### Creating Idle Frames in GIMP

If you have ONE static character sprite:

1. **Open sprite in GIMP**
2. **Layer → Duplicate** (make 8 copies)
3. **For each frame:**
   - Select layer
   - Filters → Distorts → IWarp
   - Slightly deform (breathing motion)
   - Move tool: shift pixels 1-2px for sway
4. **Export each layer** as frame_000.png, frame_001.png, etc.
5. **Combine into sprite sheet** (see Part 5)

---

## PART 3: Background Ambient Animations

### Particle Systems

**Examples:**
- Floating dust motes
- Magic sparkles
- Fireflies
- Embers from fire
- Falling leaves

**Creating Particle Sprite Sheet:**

**Method 1: Individual Particles**
```
[Particle 1][Particle 2][Particle 3][Particle 4]...[Particle 16]
```
- Each particle is a different shape/rotation
- Code randomly picks particles and animates them
- No frame animation, just spawn/move/fade

**Method 2: Animated Particles**
```
[Sparkle Frame 0][Sparkle Frame 1][Sparkle Frame 2]...[Sparkle Frame 8]
```
- Each frame shows particle evolving (grow → shrink → fade)
- Code plays animation when particle spawns
- Looks more polished

**React Native Particle Code Example:**
```javascript
// Spawn particles randomly
const particles = [];
for (let i = 0; i < 20; i++) {
  particles.push({
    x: Math.random() * screenWidth,
    y: Math.random() * screenHeight,
    vx: (Math.random() - 0.5) * 2, // Velocity X
    vy: -Math.random() * 3,         // Velocity Y (float up)
    frame: Math.floor(Math.random() * 16)
  });
}

// Animate each particle
particles.forEach(p => {
  p.x += p.vx;
  p.y += p.vy;
  p.frame = (p.frame + 1) % 16;
});
```

### Environmental Animations

**Flickering Candle/Fire:**
```
[Bright][Medium][Dim][Medium][Bright][Very Bright][Medium]
```
- 6-8 frames, looping
- 24-30 FPS for realistic flicker
- Each frame: slightly different flame shape

**Water Flow:**
```
[Ripple 0][Ripple 1][Ripple 2][Ripple 3]
```
- 4-6 frames, seamless loop
- 12-15 FPS (slow, calming)
- Use tiling texture for rivers/lakes

**Swaying Plants:**
```
[Center][Right][Center][Left][Center]
```
- 5 frames, looping
- 8-10 FPS (gentle sway)
- Use Filters → Distorts → IWarp in GIMP

---

## PART 4: Using Claude's Animation Tools

### Extract Frames from Midjourney Sprite Sheet

Your Midjourney gave you a sprite sheet with 8 frames in a row:

```bash
# Extract frames
python tools/animation_tools.py extract \
  assets/art/ui/card_flip_sprite_sheet.png \
  assets/animations/card_flip/ \
  --width 256 \
  --height 256 \
  --columns 8 \
  --rows 1
```

This creates:
```
assets/animations/card_flip/
  frame_000.png
  frame_001.png
  frame_002.png
  ...
  frame_007.png
```

### Create Animation Manifest

```bash
# Generate manifest JSON
python tools/animation_tools.py manifest \
  assets/animations/card_flip/ \
  card_flip \
  --fps 24 \
  --loop
```

Creates `card_flip_manifest.json`:
```json
{
  "name": "card_flip",
  "fps": 24,
  "loop": false,
  "frameCount": 8,
  "frameDelay": 41,
  "frames": [...]
}
```

### Generate React Native Component

```bash
# Auto-generate animation component
python tools/animation_tools.py generate-code \
  assets/animations/card_flip/card_flip_manifest.json \
  src/components/CardFlipAnimation.js
```

Now you have a ready-to-use component:
```javascript
import CardFlipAnimation from './components/CardFlipAnimation';

<CardFlipAnimation
  playing={true}
  onComplete={() => console.log('Animation done!')}
  style={{ width: 256, height: 256 }}
/>
```

### Create Preview GIF

```bash
# Test animation as GIF
python tools/animation_tools.py flipbook \
  assets/animations/card_flip/ \
  preview_card_flip.gif \
  --duration 1000
```

---

## PART 5: Manual Sprite Sheet Creation

### In GIMP (If You Have Individual Frames)

1. **Calculate sheet size:**
   - 8 frames × 256px width = 2048px wide
   - 1 row × 256px height = 256px tall

2. **Create new image:**
   - File → New → 2048x256px
   - Transparent background

3. **Import frames as layers:**
   - File → Open as Layers
   - Select all 8 frames
   - They stack as layers

4. **Position each layer:**
   - Move tool (M)
   - Drag layer to correct position:
     - Frame 0: X=0
     - Frame 1: X=256
     - Frame 2: X=512
     - Frame 3: X=768
     - Frame 4: X=1024
     - Frame 5: X=1280
     - Frame 6: X=1536
     - Frame 7: X=1792

5. **Flatten and export:**
   - Image → Flatten Image
   - File → Export As → sprite_sheet.png

**Pro Tip**: Use grid
- View → Show Grid
- Image → Configure Grid → 256x256px
- Snap to Grid (View → Snap to Grid)

---

## PART 6: React Native Animation Code

### Method 1: Simple Frame-Based Animation

```javascript
import React, { useState, useEffect } from 'react';
import { Image, StyleSheet } from 'react-native';

export default function SpriteAnimation({ spriteSheet, frameWidth, frameHeight, frameCount, fps, loop = true }) {
  const [currentFrame, setCurrentFrame] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentFrame(prev => {
        const next = (prev + 1) % frameCount;
        if (!loop && next === 0) {
          clearInterval(interval);
        }
        return next;
      });
    }, 1000 / fps);

    return () => clearInterval(interval);
  }, []);

  return (
    <Image
      source={spriteSheet}
      style={[
        styles.sprite,
        {
          width: frameWidth,
          height: frameHeight,
          transform: [{ translateX: -currentFrame * frameWidth }]
        }
      ]}
    />
  );
}

const styles = StyleSheet.create({
  sprite: {
    overflow: 'hidden'
  }
});
```

### Method 2: Using Animated API (Smoother)

```javascript
import React, { useEffect, useRef } from 'react';
import { Animated, Image } from 'react-native';

export default function SmoothSpriteAnimation({ spriteSheet, frameCount, fps }) {
  const frameAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.loop(
      Animated.timing(frameAnim, {
        toValue: frameCount - 1,
        duration: (frameCount / fps) * 1000,
        useNativeDriver: true
      })
    ).start();
  }, []);

  const translateX = frameAnim.interpolate({
    inputRange: [0, frameCount - 1],
    outputRange: [0, -(frameCount - 1) * 256]
  });

  return (
    <Animated.Image
      source={spriteSheet}
      style={{
        width: 256,
        height: 256,
        transform: [{ translateX }]
      }}
    />
  );
}
```

### Method 3: Using Lottie (Best for Complex Animations)

```bash
npm install lottie-react-native
```

```javascript
import LottieView from 'lottie-react-native';

<LottieView
  source={require('./animations/magic_particles.json')}
  autoPlay
  loop
  style={{ width: 200, height: 200 }}
/>
```

---

## PART 7: Particle System Implementation

### Simple Particle Component

```javascript
import React, { useState, useEffect } from 'react';
import { View, Image, StyleSheet } from 'react-native';

export default function ParticleSystem({ particleImage, count = 20, speed = 1 }) {
  const [particles, setParticles] = useState([]);

  useEffect(() => {
    // Initialize particles
    const initial = Array.from({ length: count }, (_, i) => ({
      id: i,
      x: Math.random() * 300,
      y: Math.random() * 500,
      vx: (Math.random() - 0.5) * speed,
      vy: -Math.random() * speed * 2,
      opacity: Math.random() * 0.5 + 0.5,
      scale: Math.random() * 0.5 + 0.5
    }));
    setParticles(initial);

    // Animate particles
    const interval = setInterval(() => {
      setParticles(prev => prev.map(p => ({
        ...p,
        x: p.x + p.vx,
        y: p.y + p.vy,
        // Reset if off screen
        ...(p.y < -50 ? {
          y: 550,
          x: Math.random() * 300
        } : {})
      })));
    }, 16); // ~60 FPS

    return () => clearInterval(interval);
  }, []);

  return (
    <View style={styles.container}>
      {particles.map(p => (
        <Image
          key={p.id}
          source={particleImage}
          style={[
            styles.particle,
            {
              left: p.x,
              top: p.y,
              opacity: p.opacity,
              transform: [{ scale: p.scale }]
            }
          ]}
        />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    width: '100%',
    height: '100%'
  },
  particle: {
    position: 'absolute',
    width: 20,
    height: 20
  }
});
```

---

## PART 8: Performance Optimization

### DO:
1. **Reuse sprite sheets** - Don't load same asset multiple times
2. **Use shouldRasterizeIOS** - Rasterize complex animations
3. **Limit particle count** - More than 30-50 particles = lag on mobile
4. **Use native driver** - `useNativeDriver: true` in Animated
5. **Preload assets** - Load all sprites on app start
6. **Use FastImage** - `react-native-fast-image` for better performance
7. **Batch state updates** - Don't update every particle individually

### DON'T:
1. **Don't animate too many things at once** - Prioritize visible animations
2. **Don't use GIFs** - Use sprite sheets or Lottie instead (better performance)
3. **Don't use huge sprite sheets** - Keep under 2048x2048px
4. **Don't animate every frame** - 24-30 FPS is enough, not 60 FPS
5. **Don't forget to cleanup** - Clear intervals/animations on unmount

---

## PART 9: Complete Workflow Example

### Character Idle Animation (Start to Finish)

**Step 1: Generate in Midjourney**
```
isometric pixel art character sprite sheet, mystical wizard, 8 frame idle animation breathing cycle, dark robes, glowing staff, purple and cyan, transparent background, game asset --ar 8:1 --v 6
```

**Step 2: Download and Extract**
```bash
# Download: character_idle_sheet.png (2048x256, 8 frames)

# Extract frames
python tools/animation_tools.py extract \
  character_idle_sheet.png \
  assets/animations/wizard_idle/ \
  --width 256 --height 256 \
  --columns 8 --rows 1
```

**Step 3: Clean up frames (if needed)**
```bash
# Remove backgrounds
python tools/image_tools.py batch \
  assets/animations/wizard_idle/ \
  assets/animations/wizard_idle_clean/ \
  remove_bg --threshold 240

# Optimize file sizes
python tools/image_tools.py batch \
  assets/animations/wizard_idle_clean/ \
  assets/animations/wizard_idle_final/ \
  optimize
```

**Step 4: Create manifest**
```bash
python tools/animation_tools.py manifest \
  assets/animations/wizard_idle_final/ \
  wizard_idle \
  --fps 12 \
  --loop
```

**Step 5: Generate React component**
```bash
python tools/animation_tools.py generate-code \
  assets/animations/wizard_idle_final/wizard_idle_manifest.json \
  src/components/WizardIdleAnimation.js
```

**Step 6: Use in game**
```javascript
import WizardIdleAnimation from './components/WizardIdleAnimation';

<WizardIdleAnimation
  playing={true}
  style={{ width: 128, height: 128 }}
/>
```

---

## PART 10: Advanced Techniques

### Blending Animations

```javascript
// Smooth transition between idle and walk
const blendValue = useRef(new Animated.Value(0)).current;

// 0 = idle, 1 = walk
Animated.timing(blendValue, {
  toValue: isWalking ? 1 : 0,
  duration: 200,
  useNativeDriver: true
}).start();

const idleOpacity = blendValue.interpolate({
  inputRange: [0, 1],
  outputRange: [1, 0]
});

const walkOpacity = blendValue.interpolate({
  inputRange: [0, 1],
  outputRange: [0, 1]
});

<>
  <Animated.View style={{ opacity: idleOpacity }}>
    <IdleAnimation />
  </Animated.View>
  <Animated.View style={{ opacity: walkOpacity, position: 'absolute' }}>
    <WalkAnimation />
  </Animated.View>
</>
```

### Direction-Based Sprite Sheets

For 8-directional movement:

```
Row 0: [North idle frames...]
Row 1: [Northeast idle frames...]
Row 2: [East idle frames...]
Row 3: [Southeast idle frames...]
Row 4: [South idle frames...]
Row 5: [Southwest idle frames...]
Row 6: [West idle frames...]
Row 7: [Northwest idle frames...]
```

Or use sprite flipping (cheaper):
```javascript
// Only need 4 directions, flip horizontally for other 4
<Image
  style={{
    transform: [
      { scaleX: facingLeft ? -1 : 1 }
    ]
  }}
/>
```

---

## Quick Reference

### Command Cheat Sheet

```bash
# Extract sprite sheet frames
python tools/animation_tools.py extract input.png output_dir/ --width 256 --height 256 --columns 8 --rows 1

# Create manifest
python tools/animation_tools.py manifest frames_dir/ animation_name --fps 24 --loop

# Generate React Native component
python tools/animation_tools.py generate-code manifest.json output.js

# Create preview GIF
python tools/animation_tools.py flipbook frames_dir/ output.gif --duration 1000

# Detect keyframes (reduce frame count)
python tools/animation_tools.py keyframes frames_dir/ --threshold 0.1
```

### FPS Guidelines

- **Idle**: 10-15 FPS
- **Walk**: 18-24 FPS
- **Run**: 24-30 FPS
- **Attack**: 24-30 FPS
- **Particles**: 30-60 FPS
- **UI effects**: 30 FPS

### File Size Targets

- **Single frame**: < 50 KB
- **8-frame sheet**: < 400 KB
- **16-frame sheet**: < 800 KB
- **Particle sprite**: < 20 KB

---

**You now have the tools to bring your Midjourney art to life. Start simple (idle animations), then add complexity (particles, effects). Iterate!**
