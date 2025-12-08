---
name: mystical-ui-patterns
description: Engineering mystical, tarot-inspired interfaces that balance spiritual aesthetics with modern UX. Specific patterns for card reveals, cosmic effects, and sacred geometry in UI.
---

# MYSTICAL_UI_PATTERNS.skill.md

## Engineering Mystical Digital Experiences Without Losing Usability

**Version**: 1.0
**Domain**: Mystical UI/UX, Sacred Geometry, Tarot Interfaces, Spiritual Digital Design
**Prerequisites**: Understanding of symbolism, animation, WebGL basics
**Output**: Interfaces that feel magical while remaining instantly usable

---

## 1. THE MYSTICAL PARADOX

### 1.1 The Balance Formula

Mystical doesn't mean unusable:

```typescript
interface MysticalBalance {
  magicalElements: number;       // Particles, glows, animations (0-10)
  usabilityScore: number;        // Task completion rate (0-1)
  loadTime: number;              // Performance in ms
  
  // The Mystical Balance Formula
  calculateBalance(): number {
    const magic = this.magicalElements / 10;
    const usable = this.usabilityScore;
    const performance = Math.max(0, 1 - (this.loadTime / 3000));
    
    // Perfect balance when all three equal
    return (magic * usable * performance) * 100;
  }
  
  // Target: 60+ score
  // Too much magic = unusable
  // Too little magic = generic
}
```

### 1.2 The Mystical Design Language

```scss
:root {
  // Cosmic color palette
  --cosmos-deep: #0a0a1e;
  --cosmos-void: #16162e;
  --cosmos-nebula: #2e1a4a;
  --cosmos-star: #9b72cf;
  --cosmos-aurora: #00d4ff;
  --cosmos-divine: #ffd700;
  
  // Sacred ratios
  --phi: 1.618;          // Golden ratio
  --sqrt2: 1.414;        // Sacred root
  --pi: 3.14159;         // Circle constant
  
  // Mystical timing (all in ms)
  --transition-instant: 88;      // Mercury number
  --transition-quick: 333;       // Trinity
  --transition-sacred: 618;      // Fibonacci
  --transition-ritual: 1618;     // Golden ms
  --transition-cosmic: 3333;     // Master number
  
  // Glow intensities
  --glow-subtle: 0 0 20px rgba(155, 114, 207, 0.3);
  --glow-medium: 0 0 40px rgba(155, 114, 207, 0.5);
  --glow-intense: 0 0 60px rgba(155, 114, 207, 0.7);
  --glow-divine: 
    0 0 30px rgba(155, 114, 207, 0.5),
    0 0 60px rgba(0, 212, 255, 0.3),
    0 0 90px rgba(255, 215, 0, 0.2);
}
```

---

## 2. TAROT CARD INTERFACE PATTERNS

### 2.1 The Card Reveal Ritual

```scss
.tarot-card {
  --card-width: 200px;
  --card-height: calc(var(--card-width) * var(--phi));  // Golden ratio
  
  position: relative;
  width: var(--card-width);
  height: var(--card-height);
  transform-style: preserve-3d;
  transition: transform var(--transition-sacred) cubic-bezier(0.4, 0, 0.2, 1);
  cursor: pointer;
  
  // Card faces
  &__face {
    position: absolute;
    width: 100%;
    height: 100%;
    backface-visibility: hidden;
    border-radius: 16px;
    overflow: hidden;
    
    &--front {
      transform: rotateY(0);
      
      // The card artwork
      img {
        width: 100%;
        height: 100%;
        object-fit: cover;
      }
      
      // Mystical overlay
      &::after {
        content: '';
        position: absolute;
        inset: 0;
        background: 
          radial-gradient(
            ellipse at center,
            transparent 0%,
            rgba(155, 114, 207, 0.1) 50%,
            rgba(155, 114, 207, 0.2) 100%
          );
        mix-blend-mode: overlay;
        animation: card-pulse 3333ms ease-in-out infinite;
      }
    }
    
    &--back {
      transform: rotateY(180deg);
      background: 
        radial-gradient(
          circle at 30% 30%,
          var(--cosmos-nebula) 0%,
          var(--cosmos-deep) 100%
        );
      
      // Sacred geometry pattern
      &::before {
        content: '';
        position: absolute;
        inset: 10%;
        border: 1px solid var(--cosmos-star);
        border-radius: 12px;
        opacity: 0.3;
        
        // Nested golden rectangles
        box-shadow: 
          inset 0 0 0 10px var(--cosmos-void),
          inset 0 0 0 11px var(--cosmos-star),
          inset 0 0 0 25px var(--cosmos-void),
          inset 0 0 0 26px var(--cosmos-star);
      }
      
      // Center symbol
      &::after {
        content: '☽';  // Moon symbol
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 48px;
        color: var(--cosmos-divine);
        text-shadow: var(--glow-divine);
        animation: symbol-rotate 30s linear infinite;
      }
    }
  }
  
  // Hover state
  &:hover {
    transform: translateY(-10px) rotateX(5deg);
    
    .tarot-card__glow {
      opacity: 1;
    }
    
    .tarot-card__particles {
      opacity: 1;
    }
  }
  
  // Flipped state
  &.is-flipped {
    transform: rotateY(180deg);
  }
  
  // The glow effect
  &__glow {
    position: absolute;
    inset: -20px;
    background: radial-gradient(
      circle at center,
      var(--cosmos-star) 0%,
      transparent 70%
    );
    opacity: 0;
    transition: opacity var(--transition-sacred) ease;
    pointer-events: none;
    filter: blur(20px);
  }
}

@keyframes card-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

@keyframes symbol-rotate {
  0% { transform: translate(-50%, -50%) rotate(0deg); }
  100% { transform: translate(-50%, -50%) rotate(360deg); }
}
```

### 2.2 The Spread Layout

```javascript
class TarotSpread {
  constructor(type = 'celtic-cross') {
    this.spreads = {
      'three-card': [
        { x: -33, y: 0, rotation: -5, name: 'Past' },
        { x: 0, y: 0, rotation: 0, name: 'Present' },
        { x: 33, y: 0, rotation: 5, name: 'Future' }
      ],
      'celtic-cross': [
        { x: 0, y: 0, rotation: 0, name: 'Present' },
        { x: 0, y: 0, rotation: 90, name: 'Cross' },
        { x: 0, y: -30, rotation: 0, name: 'Distant Past' },
        { x: 0, y: 30, rotation: 0, name: 'Recent Past' },
        { x: -30, y: 0, rotation: 0, name: 'Possible Future' },
        { x: 30, y: 0, rotation: 0, name: 'Near Future' },
        { x: 50, y: 30, rotation: 0, name: 'Self' },
        { x: 50, y: 10, rotation: 0, name: 'External' },
        { x: 50, y: -10, rotation: 0, name: 'Hopes' },
        { x: 50, y: -30, rotation: 0, name: 'Outcome' }
      ],
      'pentagram': this.generatePentagram()
    };
    
    this.currentSpread = this.spreads[type];
  }
  
  generatePentagram() {
    const positions = [];
    const points = 5;
    const radius = 40;
    
    for (let i = 0; i < points; i++) {
      const angle = (i * 72 - 90) * (Math.PI / 180);
      positions.push({
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
        rotation: (i * 72),
        name: ['Spirit', 'Water', 'Fire', 'Earth', 'Air'][i]
      });
    }
    
    return positions;
  }
  
  animateReveal(cards) {
    cards.forEach((card, index) => {
      const position = this.currentSpread[index];
      const delay = index * 200;  // Stagger reveals
      
      // Start from deck position
      card.style.transform = `translate(0, 0) rotate(0)`;
      card.style.opacity = '0';
      
      // Animate to spread position
      setTimeout(() => {
        card.style.transition = `all ${1618}ms cubic-bezier(0.34, 1.56, 0.64, 1)`;
        card.style.transform = `
          translate(${position.x}%, ${position.y}%) 
          rotate(${position.rotation}deg)
        `;
        card.style.opacity = '1';
        
        // Add glow on arrival
        setTimeout(() => {
          this.createArrivalEffect(card);
        }, 1618);
      }, delay);
    });
  }
  
  createArrivalEffect(card) {
    // Create ripple
    const ripple = document.createElement('div');
    ripple.className = 'mystical-ripple';
    card.appendChild(ripple);
    
    // Trigger animation
    requestAnimationFrame(() => {
      ripple.style.transform = 'scale(3)';
      ripple.style.opacity = '0';
    });
    
    // Clean up
    setTimeout(() => ripple.remove(), 1000);
  }
}
```

### 2.3 Card Energy Visualization

```scss
.card-energy {
  position: relative;
  
  // Energy aura based on card type
  &[data-suit="wands"] {
    --energy-color: #ff6b35;  // Fire orange
    --energy-symbol: '🔥';
  }
  
  &[data-suit="cups"] {
    --energy-color: #4dabf7;  // Water blue
    --energy-symbol: '💧';
  }
  
  &[data-suit="swords"] {
    --energy-color: #ffd43b;  // Air yellow
    --energy-symbol: '💨';
  }
  
  &[data-suit="pentacles"] {
    --energy-color: #51cf66;  // Earth green
    --energy-symbol: '🌍';
  }
  
  &[data-suit="major"] {
    --energy-color: #9b72cf;  // Divine purple
    --energy-symbol: '✨';
  }
  
  // Animated energy field
  &::before {
    content: '';
    position: absolute;
    inset: -40px;
    background: radial-gradient(
      circle at center,
      transparent 30%,
      var(--energy-color) 50%,
      transparent 70%
    );
    opacity: 0.3;
    animation: energy-pulse 2s ease-in-out infinite;
    pointer-events: none;
    filter: blur(15px);
  }
  
  // Floating symbols
  &::after {
    content: var(--energy-symbol);
    position: absolute;
    font-size: 24px;
    animation: 
      symbol-float 3s ease-in-out infinite,
      symbol-fade 3s ease-in-out infinite;
  }
  
  // Multiple floating particles
  .energy-particles {
    position: absolute;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
    
    span {
      position: absolute;
      width: 4px;
      height: 4px;
      background: var(--energy-color);
      border-radius: 50%;
      box-shadow: 0 0 6px var(--energy-color);
      animation: particle-rise 4s linear infinite;
      
      @for $i from 1 through 20 {
        &:nth-child(#{$i}) {
          left: #{random(100)}%;
          animation-delay: #{random(4000)}ms;
          animation-duration: #{3000 + random(2000)}ms;
        }
      }
    }
  }
}

@keyframes energy-pulse {
  0%, 100% { 
    transform: scale(1);
    opacity: 0.3;
  }
  50% { 
    transform: scale(1.1);
    opacity: 0.5;
  }
}

@keyframes symbol-float {
  0%, 100% { transform: translateY(0) rotate(0); }
  50% { transform: translateY(-20px) rotate(180deg); }
}

@keyframes particle-rise {
  0% {
    bottom: -10px;
    opacity: 0;
  }
  10% {
    opacity: 1;
  }
  90% {
    opacity: 1;
  }
  100% {
    bottom: 100%;
    opacity: 0;
  }
}
```

---

## 3. COSMIC BACKGROUND SYSTEMS

### 3.1 WebGL Starfield

```javascript
class CosmicBackground {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl');
    this.stars = [];
    this.time = 0;
    
    this.init();
  }
  
  init() {
    // Generate star field
    const starCount = 1000;
    for (let i = 0; i < starCount; i++) {
      this.stars.push({
        x: (Math.random() - 0.5) * 2,
        y: (Math.random() - 0.5) * 2,
        z: Math.random(),
        size: Math.random() * 2 + 0.5,
        twinkle: Math.random() * Math.PI * 2,
        speed: Math.random() * 0.001 + 0.0005
      });
    }
    
    // Vertex shader
    const vertexShader = `
      attribute vec3 position;
      attribute float size;
      attribute float twinkle;
      
      uniform float time;
      uniform mat4 projection;
      
      varying float vTwinkle;
      
      void main() {
        vTwinkle = sin(twinkle + time) * 0.5 + 0.5;
        gl_Position = projection * vec4(position, 1.0);
        gl_PointSize = size * (1.0 + vTwinkle * 0.5);
      }
    `;
    
    // Fragment shader
    const fragmentShader = `
      precision mediump float;
      
      varying float vTwinkle;
      
      void main() {
        vec2 center = gl_PointCoord - 0.5;
        float dist = length(center);
        
        if (dist > 0.5) discard;
        
        float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
        vec3 color = mix(
          vec3(0.7, 0.8, 1.0),  // Cool white
          vec3(1.0, 0.9, 0.7),  // Warm white
          vTwinkle
        );
        
        gl_FragColor = vec4(color, alpha * vTwinkle);
      }
    `;
    
    this.setupShaders(vertexShader, fragmentShader);
    this.animate();
  }
  
  animate() {
    this.time += 0.01;
    
    // Update star positions
    this.stars.forEach(star => {
      star.z -= star.speed;
      if (star.z < 0) star.z = 1;
      
      // Parallax effect based on mouse
      const mouseX = (window.mouseX || 0) / window.innerWidth - 0.5;
      const mouseY = (window.mouseY || 0) / window.innerHeight - 0.5;
      
      star.x += mouseX * star.z * 0.0001;
      star.y += mouseY * star.z * 0.0001;
    });
    
    this.render();
    requestAnimationFrame(() => this.animate());
  }
}
```

### 3.2 CSS-Only Cosmic Effects

```scss
.cosmic-background {
  position: fixed;
  inset: 0;
  overflow: hidden;
  background: var(--cosmos-deep);
  
  // Layer 1: Deep space gradient
  &::before {
    content: '';
    position: absolute;
    inset: 0;
    background: 
      radial-gradient(
        ellipse at 20% 30%,
        rgba(155, 114, 207, 0.3) 0%,
        transparent 40%
      ),
      radial-gradient(
        ellipse at 80% 70%,
        rgba(0, 212, 255, 0.2) 0%,
        transparent 50%
      ),
      radial-gradient(
        ellipse at 50% 50%,
        rgba(255, 215, 0, 0.1) 0%,
        transparent 60%
      );
    animation: nebula-drift 60s ease-in-out infinite;
  }
  
  // Layer 2: Stars
  &::after {
    content: '';
    position: absolute;
    inset: 0;
    background-image: 
      radial-gradient(2px 2px at 20% 30%, white, transparent),
      radial-gradient(2px 2px at 40% 70%, white, transparent),
      radial-gradient(1px 1px at 50% 50%, white, transparent),
      radial-gradient(1px 1px at 80% 10%, white, transparent),
      radial-gradient(2px 2px at 90% 60%, white, transparent);
    background-size: 
      300px 300px,
      400px 400px,
      250px 250px,
      350px 350px,
      450px 450px;
    background-position: 
      0 0,
      100px 100px,
      50px 50px,
      150px 150px,
      200px 200px;
    animation: star-movement 120s linear infinite;
  }
  
  // Layer 3: Shooting stars
  .shooting-star {
    position: absolute;
    top: 0;
    width: 2px;
    height: 100px;
    background: linear-gradient(
      to bottom,
      transparent,
      rgba(255, 255, 255, 0.8),
      transparent
    );
    animation: shooting 3s ease-in-out infinite;
    
    @for $i from 1 through 5 {
      &:nth-child(#{$i}) {
        left: #{random(100)}%;
        animation-delay: #{random(5000)}ms;
        animation-duration: #{2000 + random(2000)}ms;
      }
    }
  }
}

@keyframes nebula-drift {
  0%, 100% { transform: rotate(0deg) scale(1); }
  25% { transform: rotate(1deg) scale(1.05); }
  50% { transform: rotate(-1deg) scale(1.1); }
  75% { transform: rotate(0.5deg) scale(1.05); }
}

@keyframes star-movement {
  0% { transform: translate(0, 0); }
  100% { transform: translate(-100px, -100px); }
}

@keyframes shooting {
  0% {
    transform: translateY(-100px) translateX(0) rotate(45deg);
    opacity: 0;
  }
  5% {
    opacity: 1;
  }
  95% {
    opacity: 1;
  }
  100% {
    transform: translateY(100vh) translateX(100px) rotate(45deg);
    opacity: 0;
  }
}
```

---

## 4. SACRED GEOMETRY IN UI

### 4.1 The Golden Spiral Layout

```javascript
class GoldenSpiralLayout {
  constructor(container) {
    this.container = container;
    this.phi = 1.618;
    this.items = [];
  }
  
  calculateSpiral(n) {
    const positions = [];
    
    for (let i = 0; i < n; i++) {
      // Fibonacci spiral
      const angle = i * 137.5 * (Math.PI / 180); // Golden angle
      const radius = Math.sqrt(i) * 20;
      
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius;
      const scale = 1 / (1 + i * 0.1);
      const rotation = angle * (180 / Math.PI);
      
      positions.push({ x, y, scale, rotation });
    }
    
    return positions;
  }
  
  layoutItems() {
    const positions = this.calculateSpiral(this.items.length);
    const centerX = this.container.offsetWidth / 2;
    const centerY = this.container.offsetHeight / 2;
    
    this.items.forEach((item, index) => {
      const pos = positions[index];
      
      item.style.position = 'absolute';
      item.style.left = `${centerX + pos.x}px`;
      item.style.top = `${centerY + pos.y}px`;
      item.style.transform = `
        translate(-50%, -50%)
        scale(${pos.scale})
        rotate(${pos.rotation}deg)
      `;
      item.style.opacity = pos.scale;
      item.style.zIndex = Math.floor(pos.scale * 100);
      
      // Animate in
      item.style.animation = `spiral-in ${1618}ms ${index * 100}ms both cubic-bezier(0.34, 1.56, 0.64, 1)`;
    });
  }
}
```

### 4.2 Metatron's Cube Navigation

```scss
.metatron-nav {
  position: relative;
  width: 300px;
  height: 300px;
  
  // Center node
  &__center {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 60px;
    height: 60px;
    transform: translate(-50%, -50%);
    background: var(--cosmos-nebula);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10;
    box-shadow: var(--glow-divine);
    
    &:hover {
      transform: translate(-50%, -50%) scale(1.1);
    }
  }
  
  // Outer nodes (6 points)
  &__node {
    position: absolute;
    width: 40px;
    height: 40px;
    background: var(--cosmos-void);
    border: 2px solid var(--cosmos-star);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all var(--transition-sacred) ease;
    
    @for $i from 1 through 6 {
      &:nth-child(#{$i}) {
        $angle: ($i - 1) * 60deg - 90deg;
        $x: 50% + 100px * cos($angle);
        $y: 50% + 100px * sin($angle);
        
        left: calc(#{$x} - 20px);
        top: calc(#{$y} - 20px);
      }
    }
    
    &:hover {
      background: var(--cosmos-star);
      transform: scale(1.2);
      box-shadow: var(--glow-intense);
    }
  }
  
  // Connection lines
  &__lines {
    position: absolute;
    inset: 0;
    pointer-events: none;
    
    svg {
      width: 100%;
      height: 100%;
      
      line {
        stroke: var(--cosmos-star);
        stroke-width: 1;
        opacity: 0.3;
        stroke-dasharray: 5, 5;
        animation: line-pulse 3s linear infinite;
      }
    }
  }
}

@keyframes line-pulse {
  0% { stroke-dashoffset: 0; }
  100% { stroke-dashoffset: 10; }
}
```

---

## 5. MYSTICAL INTERACTION PATTERNS

### 5.1 Crystal Ball Hover Effect

```scss
.crystal-ball {
  position: relative;
  width: 200px;
  height: 200px;
  border-radius: 50%;
  background: 
    radial-gradient(
      circle at 30% 30%,
      rgba(255, 255, 255, 0.8),
      transparent 30%
    ),
    radial-gradient(
      circle at center,
      var(--cosmos-void),
      var(--cosmos-deep)
    );
  overflow: hidden;
  cursor: pointer;
  
  // Glass refraction effect
  &::before {
    content: '';
    position: absolute;
    inset: 5px;
    border-radius: 50%;
    background: 
      radial-gradient(
        ellipse at 70% 70%,
        transparent 30%,
        rgba(155, 114, 207, 0.1) 100%
      );
    backdrop-filter: blur(5px);
  }
  
  // Mist inside
  &__mist {
    position: absolute;
    inset: 20%;
    border-radius: 50%;
    background: radial-gradient(
      circle at center,
      var(--cosmos-star),
      transparent
    );
    opacity: 0;
    filter: blur(20px);
    transform: scale(0.5);
    transition: all var(--transition-ritual) ease;
  }
  
  // Vision content
  &__vision {
    position: absolute;
    inset: 25%;
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0;
    transform: scale(0) rotate(-180deg);
    transition: all var(--transition-ritual) ease;
    
    img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      border-radius: 50%;
    }
  }
  
  // Hover state
  &:hover {
    .crystal-ball__mist {
      opacity: 0.5;
      transform: scale(1);
      animation: mist-swirl 4s ease-in-out infinite;
    }
    
    .crystal-ball__vision {
      opacity: 1;
      transform: scale(1) rotate(0);
      transition-delay: 300ms;
    }
  }
}

@keyframes mist-swirl {
  0%, 100% { transform: scale(1) rotate(0); }
  50% { transform: scale(1.2) rotate(180deg); }
}
```

### 5.2 Rune Stone Interactions

```javascript
class RuneStone {
  constructor(element) {
    this.element = element;
    this.runes = ['ᚠ', 'ᚢ', 'ᚦ', 'ᚩ', 'ᚱ', 'ᚳ', 'ᚷ', 'ᚹ', 'ᚻ'];
    this.init();
  }
  
  init() {
    // Create rune faces
    this.faces = this.createFaces();
    
    // Mouse interaction
    this.element.addEventListener('mousemove', (e) => {
      this.handleRotation(e);
    });
    
    // Click to cast
    this.element.addEventListener('click', () => {
      this.castRune();
    });
  }
  
  createFaces() {
    const faces = [];
    
    // Create 6 faces of a cube
    ['front', 'back', 'right', 'left', 'top', 'bottom'].forEach((face, i) => {
      const div = document.createElement('div');
      div.className = `rune-face rune-face--${face}`;
      div.textContent = this.runes[i];
      this.element.appendChild(div);
      faces.push(div);
    });
    
    return faces;
  }
  
  handleRotation(e) {
    const rect = this.element.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;
    
    const rotateY = (x / rect.width) * 60;
    const rotateX = -(y / rect.height) * 60;
    
    this.element.style.transform = `
      perspective(1000px)
      rotateX(${rotateX}deg)
      rotateY(${rotateY}deg)
    `;
  }
  
  castRune() {
    // Random tumble animation
    const finalX = Math.random() * 720 - 360;
    const finalY = Math.random() * 720 - 360;
    const finalZ = Math.random() * 720 - 360;
    
    this.element.style.animation = 'rune-cast 2s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
    
    setTimeout(() => {
      this.element.style.transform = `
        rotateX(${finalX}deg)
        rotateY(${finalY}deg)
        rotateZ(${finalZ}deg)
      `;
      
      // Show result
      this.showReading();
    }, 2000);
  }
  
  showReading() {
    // Glow effect on landing
    this.element.classList.add('casting-complete');
    
    // Create energy burst
    const burst = document.createElement('div');
    burst.className = 'rune-burst';
    this.element.appendChild(burst);
    
    setTimeout(() => burst.remove(), 1000);
  }
}
```

---

## 6. MYSTICAL LOADING & TRANSITIONS

### 6.1 Portal Loading Screen

```scss
.portal-loader {
  position: fixed;
  inset: 0;
  background: var(--cosmos-deep);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
  
  // The portal
  &__ring {
    width: 200px;
    height: 200px;
    border-radius: 50%;
    position: relative;
    
    // Outer ring
    &::before {
      content: '';
      position: absolute;
      inset: -20px;
      border-radius: 50%;
      border: 3px solid var(--cosmos-star);
      border-top-color: transparent;
      animation: portal-spin 2s linear infinite;
    }
    
    // Inner ring
    &::after {
      content: '';
      position: absolute;
      inset: -10px;
      border-radius: 50%;
      border: 2px solid var(--cosmos-aurora);
      border-bottom-color: transparent;
      animation: portal-spin 1.5s linear infinite reverse;
    }
  }
  
  // Center vortex
  &__vortex {
    position: absolute;
    inset: 20px;
    border-radius: 50%;
    background: 
      radial-gradient(
        circle at center,
        var(--cosmos-divine) 0%,
        var(--cosmos-star) 30%,
        var(--cosmos-deep) 100%
      );
    animation: vortex-swirl 3s ease-in-out infinite;
    overflow: hidden;
    
    &::before {
      content: '';
      position: absolute;
      inset: -50%;
      background: conic-gradient(
        from 0deg,
        transparent,
        var(--cosmos-aurora),
        transparent
      );
      animation: portal-spin 1s linear infinite;
    }
  }
  
  // Loading text
  &__text {
    position: absolute;
    bottom: 30%;
    text-align: center;
    color: var(--cosmos-star);
    font-size: 14px;
    letter-spacing: 4px;
    text-transform: uppercase;
    opacity: 0.7;
    animation: text-pulse 2s ease-in-out infinite;
  }
  
  // Fade out when complete
  &.complete {
    animation: portal-close 1s ease-in-out forwards;
  }
}

@keyframes portal-spin {
  0% { transform: rotate(0); }
  100% { transform: rotate(360deg); }
}

@keyframes vortex-swirl {
  0%, 100% { transform: scale(1); filter: blur(0); }
  50% { transform: scale(1.1); filter: blur(2px); }
}

@keyframes text-pulse {
  0%, 100% { opacity: 0.7; }
  50% { opacity: 1; }
}

@keyframes portal-close {
  0% { transform: scale(1); opacity: 1; }
  100% { transform: scale(0); opacity: 0; }
}
```

---

## 7. IMPLEMENTATION CHECKLIST

### Phase 1: Foundation (Week 1)
- [ ] Set up cosmic color system
- [ ] Implement sacred geometry grid
- [ ] Create base card component
- [ ] Add WebGL/CSS starfield
- [ ] Define mystical timing functions

### Phase 2: Core Components (Week 2)
- [ ] Tarot card flip animations
- [ ] Card spread layouts
- [ ] Energy visualization system
- [ ] Crystal ball component
- [ ] Rune stone interactions

### Phase 3: Polish (Week 3)
- [ ] Portal loading screens
- [ ] Particle effects system
- [ ] Sound design integration
- [ ] Haptic feedback patterns
- [ ] Sacred geometry navigation

### Phase 4: Optimization (Week 4)
- [ ] Performance profiling
- [ ] GPU optimization
- [ ] Mobile adaptations
- [ ] Accessibility features
- [ ] Cross-browser testing

---

## 8. BALANCING MYSTICISM WITH USABILITY

**The Golden Rules**:

1. **Every mystical element must have purpose**
   - Don't add particles just because you can
   - Each effect should enhance meaning

2. **Performance is sacred**
   - 60fps or remove the effect
   - Mobile first, enhance for desktop

3. **Accessibility is non-negotiable**
   - Screen readers must work
   - Reduce motion option required
   - Keyboard navigation complete

4. **Progressive enhancement**
   - Works without JavaScript
   - WebGL optional, CSS fallbacks
   - Core functionality always accessible

5. **Cultural respect**
   - Research symbol meanings
   - Avoid appropriation
   - Include diverse traditions

Remember: The goal is to create an experience that feels magical while remaining instantly usable. Every mystical element should enhance, not hinder, the user journey. The best mystical UI is one where users feel the magic without thinking about the interface.
