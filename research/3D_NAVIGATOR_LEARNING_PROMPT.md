# 3D Symbolic Navigator - Learning Prompt for Claude

## Context

I'm building a mobile app (React Native, iOS/Android) that needs a 3D FPS-style navigation system. Users will explore a symbolic space representing their tarot readings, archetypes, and personal growth journey.

## What I Need to Learn

### 1. React Native 3D Development Fundamentals

**Core Questions:**
- What are the best libraries for 3D in React Native? (expo-three, react-three-fiber, react-native-game-engine)
- How do I set up a basic 3D scene with camera controls?
- What's the performance profile on mobile devices?
- How do I handle touch-based FPS controls (joystick + look)?

**Specific Topics:**
- Scene graph management
- Asset loading (3D models, textures)
- Lighting and shadows on mobile
- LOD (Level of Detail) for performance
- Collision detection

### 2. FPS-Style Camera Controls for Mobile

**Requirements:**
- Left thumb: virtual joystick for movement (WASD equivalent)
- Right thumb: drag to look around (mouse-look equivalent)
- Tap to interact with objects
- Smooth, non-nauseating movement

**Questions:**
- How do I implement a virtual joystick in React Native?
- What's the best camera rig setup for FPS?
- How do I prevent motion sickness on mobile?
- How do I handle portrait vs landscape orientation?

### 3. Symbolic Object System

**Object Types:**
- Floating tarot cards (from user's readings)
- Archetype monuments (Jungian archetypes activated)
- Theme clusters (recurring themes from readings)
- Oracle conversation threads (visualized as paths/trees)
- Achievement trophies
- Personal growth markers

**Questions:**
- How do I procedurally place objects based on data?
- How do I make objects interactive (tap to view details)?
- How do I animate objects (floating, glowing, rotating)?
- How do I create visual connections between related objects?

### 4. Performance Optimization

**Constraints:**
- Must run smoothly on iPhone 8+ and mid-range Android
- 60fps target
- Battery-conscious
- Memory-efficient

**Questions:**
- How do I profile 3D performance in React Native?
- What are the key bottlenecks?
- How do I implement object pooling?
- When should I use instancing?
- How do I optimize shaders for mobile?

### 5. Visual Style

**Aesthetic:**
- Mystical/cosmic theme
- Dark environment with glowing objects
- Particle effects for atmosphere
- Minimalist but evocative

**Questions:**
- How do I create a skybox/environment?
- How do I implement bloom/glow effects on mobile?
- How do I create particle systems?
- What shader techniques work well on mobile?

## Deliverables I Need

1. **Tech stack recommendation** with pros/cons
2. **Basic scene setup** code example
3. **FPS camera controller** implementation
4. **Object interaction system** implementation
5. **Performance optimization checklist**
6. **Example of mapping data to 3D positions**

## My Current Stack

- React Native (Expo managed workflow)
- TypeScript
- Targeting iOS 13+ and Android 8+
- Already using: AsyncStorage, react-navigation, various Expo modules

## Constraints

- Must work in Expo managed workflow (or have clear path to eject)
- Must be offline-capable (3D assets bundled or cached)
- Must integrate with existing app architecture
- Budget for 3D portion: ~2-3 weeks development

## Output Format

Please provide:
1. Executive summary of recommended approach
2. Step-by-step implementation guide
3. Code examples for each major component
4. Common pitfalls to avoid
5. Resources for deeper learning

---

## Example Use Case

User opens 3D Navigator. They appear in a cosmic void. Around them float:
- 5 tarot cards from their last reading (The Fool, The Tower, etc.)
- A glowing archetype symbol (The Hero) they've been working with
- A cluster of crystals representing "transformation" theme
- A path leading to their Oracle conversation history

They use left thumb joystick to walk toward The Tower card. It grows larger as they approach. They tap it - a panel opens showing the reading details, date, their reflection notes.

They look up (right thumb drag) and see achievement monuments on floating islands above - visual representation of their progress.

---

When you respond to this prompt, assume I have intermediate React Native experience but zero 3D graphics experience. Start from fundamentals but move quickly to implementation.
