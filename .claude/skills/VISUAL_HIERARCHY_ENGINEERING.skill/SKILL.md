---
name: visual-hierarchy-engineering
description: Systematic approach to information architecture through typography, spacing, color, and visual weight. Formulas for creating scannable, intuitive interfaces that guide user attention.
---

# VISUAL_HIERARCHY_ENGINEERING.skill.md

## Engineering Information Architecture Through Systematic Visual Design

**Version**: 1.0
**Domain**: Information Architecture, Typography Systems, Visual Communication
**Prerequisites**: Understanding of gestalt principles, typography, color theory
**Output**: Interfaces where users find information in <2 seconds without conscious thought

---

## 1. THE HIERARCHY FORMULA

### 1.1 Visual Weight Calculation

Every element has measurable visual weight:

```typescript
interface VisualWeight {
  size: number;           // Font size or element dimensions (px)
  contrast: number;       // Contrast ratio with background (1-21)
  color: number;          // Saturation level (0-1)
  spacing: number;        // White space around element (px)
  position: number;       // Distance from top-left (0-1)
  
  calculateWeight(): number {
    // The Visual Weight Formula
    return (this.size * 0.4) + 
           (this.contrast * 0.3) + 
           (this.color * 0.1) + 
           (this.spacing * 0.15) + 
           ((1 - this.position) * 0.05);
  }
}

// Hierarchy must have clear weight ratios
interface HierarchyRatios {
  primary: 1.0;           // Highest weight (100%)
  secondary: 0.618;       // Golden ratio down (61.8%)
  tertiary: 0.382;        // Another golden ratio (38.2%)
  quaternary: 0.236;      // And another (23.6%)
  body: 0.146;           // Base text (14.6%)
}
```

### 1.2 The Typographic Scale

Not random sizes. Mathematical progression:

```scss
// Perfect Fourth Scale (1.333 ratio)
:root {
  --type-scale: 1.333;
  
  // Base size
  --text-base: 16px;
  
  // Scale up
  --text-lg: calc(var(--text-base) * var(--type-scale));           // 21.33px
  --text-xl: calc(var(--text-lg) * var(--type-scale));             // 28.43px
  --text-2xl: calc(var(--text-xl) * var(--type-scale));            // 37.90px
  --text-3xl: calc(var(--text-2xl) * var(--type-scale));           // 50.52px
  --text-4xl: calc(var(--text-3xl) * var(--type-scale));           // 67.34px
  
  // Scale down
  --text-sm: calc(var(--text-base) / var(--type-scale));           // 12px
  --text-xs: calc(var(--text-sm) / var(--type-scale));             // 9px
  
  // Line heights follow different ratio
  --leading-tight: 1.25;
  --leading-normal: 1.618;  // Golden ratio
  --leading-relaxed: 2;
  
  // Letter spacing decreases as size increases
  --tracking-4xl: -0.04em;
  --tracking-3xl: -0.03em;
  --tracking-2xl: -0.02em;
  --tracking-xl: -0.01em;
  --tracking-lg: 0;
  --tracking-base: 0.01em;
  --tracking-sm: 0.02em;
  --tracking-xs: 0.04em;
}
```

### 1.3 Implementation of Scale

```scss
// Typography system with automatic hierarchy
@mixin typography-level($level: 'body') {
  @if $level == 'display' {
    font-size: var(--text-4xl);
    line-height: var(--leading-tight);
    letter-spacing: var(--tracking-4xl);
    font-weight: 900;
    
    // Optical adjustments for large text
    font-feature-settings: 
      "kern" 1,  // Kerning
      "liga" 1,  // Ligatures
      "calt" 1,  // Contextual alternates
      "ss01" 1;  // Stylistic set 1
  }
  
  @else if $level == 'h1' {
    font-size: var(--text-3xl);
    line-height: var(--leading-tight);
    letter-spacing: var(--tracking-3xl);
    font-weight: 800;
    margin-top: 0;
    margin-bottom: calc(var(--text-3xl) * 0.5);
    
    // Gradient text for premium feel
    background: linear-gradient(
      135deg,
      var(--text-primary) 0%,
      var(--text-primary-light) 100%
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  
  @else if $level == 'h2' {
    font-size: var(--text-2xl);
    line-height: var(--leading-tight);
    letter-spacing: var(--tracking-2xl);
    font-weight: 700;
    margin-top: calc(var(--text-2xl) * 1);
    margin-bottom: calc(var(--text-2xl) * 0.5);
  }
  
  @else if $level == 'h3' {
    font-size: var(--text-xl);
    line-height: var(--leading-normal);
    letter-spacing: var(--tracking-xl);
    font-weight: 600;
    margin-top: calc(var(--text-xl) * 1);
    margin-bottom: calc(var(--text-xl) * 0.5);
  }
  
  @else if $level == 'body' {
    font-size: var(--text-base);
    line-height: var(--leading-normal);
    letter-spacing: var(--tracking-base);
    font-weight: 400;
    
    // Optimize for readability
    text-rendering: optimizeLegibility;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
  
  @else if $level == 'caption' {
    font-size: var(--text-sm);
    line-height: var(--leading-normal);
    letter-spacing: var(--tracking-sm);
    font-weight: 500;
    text-transform: uppercase;
    opacity: 0.7;
  }
  
  @else if $level == 'overline' {
    font-size: var(--text-xs);
    line-height: var(--leading-relaxed);
    letter-spacing: var(--tracking-xs);
    font-weight: 600;
    text-transform: uppercase;
    opacity: 0.6;
  }
}
```

---

## 2. SPATIAL HIERARCHY SYSTEM

### 2.1 The 8-Point Grid

All spacing follows 8px increments:

```scss
:root {
  --space-unit: 8px;
  
  // T-shirt sizing
  --space-xs: calc(var(--space-unit) * 0.5);   // 4px
  --space-sm: calc(var(--space-unit) * 1);     // 8px
  --space-md: calc(var(--space-unit) * 2);     // 16px
  --space-lg: calc(var(--space-unit) * 3);     // 24px
  --space-xl: calc(var(--space-unit) * 4);     // 32px
  --space-2xl: calc(var(--space-unit) * 6);    // 48px
  --space-3xl: calc(var(--space-unit) * 8);    // 64px
  --space-4xl: calc(var(--space-unit) * 12);   // 96px
  
  // Component-specific spacing
  --card-padding: var(--space-lg);
  --section-padding: var(--space-2xl);
  --page-margin: var(--space-3xl);
  
  // Dynamic spacing based on viewport
  @media (max-width: 768px) {
    --space-unit: 6px;  // Tighter on mobile
  }
  
  @media (min-width: 1920px) {
    --space-unit: 10px; // Looser on large screens
  }
}
```

### 2.2 The Proximity Principle

Related items cluster, unrelated items separate:

```scss
.content-hierarchy {
  // Heading close to its content
  h2 {
    margin-bottom: var(--space-sm);
    margin-top: var(--space-3xl);
    
    // First heading doesn't need top margin
    &:first-child {
      margin-top: 0;
    }
  }
  
  // Paragraph spacing
  p {
    margin-bottom: var(--space-md);
    
    // Last paragraph in section no bottom margin
    &:last-child {
      margin-bottom: 0;
    }
  }
  
  // List items grouped tight
  ul, ol {
    margin: var(--space-md) 0;
    padding-left: var(--space-lg);
    
    li {
      margin-bottom: var(--space-xs);
      
      &:last-child {
        margin-bottom: 0;
      }
    }
  }
  
  // Sections clearly separated
  section {
    padding: var(--section-padding) 0;
    
    & + section {
      border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
  }
}
```

### 2.3 Visual Grouping Through Cards

```scss
.card-hierarchy {
  background: var(--surface-base);
  border-radius: 16px;
  overflow: hidden;
  
  // Header establishes hierarchy
  &__header {
    padding: var(--space-lg);
    background: linear-gradient(
      180deg,
      rgba(255, 255, 255, 0.03) 0%,
      transparent 100%
    );
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    
    h3 {
      @include typography-level('h3');
      margin: 0;
    }
    
    .subtitle {
      @include typography-level('caption');
      margin-top: var(--space-xs);
      color: var(--text-secondary);
    }
  }
  
  // Content area with clear boundaries
  &__content {
    padding: var(--space-lg);
    
    // Nested cards for sub-grouping
    .sub-card {
      background: rgba(255, 255, 255, 0.02);
      border-radius: 8px;
      padding: var(--space-md);
      margin: var(--space-md) 0;
      
      // Visual recession through opacity
      opacity: 0.9;
      transition: opacity 200ms ease;
      
      &:hover {
        opacity: 1;
      }
    }
  }
  
  // Actions separated and right-aligned
  &__actions {
    padding: var(--space-md) var(--space-lg);
    background: rgba(0, 0, 0, 0.2);
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    display: flex;
    justify-content: flex-end;
    gap: var(--space-sm);
  }
}
```

---

## 3. COLOR HIERARCHY ENGINEERING

### 3.1 The Attention Color System

```scss
:root {
  // Primary attention (10% of interface)
  --color-primary: hsl(280, 100%, 65%);
  --color-primary-dark: hsl(280, 100%, 45%);
  --color-primary-light: hsl(280, 100%, 75%);
  
  // Secondary attention (20% of interface)
  --color-secondary: hsl(200, 100%, 60%);
  --color-secondary-dark: hsl(200, 100%, 40%);
  --color-secondary-light: hsl(200, 100%, 70%);
  
  // Neutral base (60% of interface)
  --color-neutral-900: hsl(240, 20%, 10%);
  --color-neutral-800: hsl(240, 20%, 15%);
  --color-neutral-700: hsl(240, 15%, 20%);
  --color-neutral-600: hsl(240, 15%, 30%);
  --color-neutral-500: hsl(240, 10%, 45%);
  --color-neutral-400: hsl(240, 10%, 60%);
  --color-neutral-300: hsl(240, 10%, 75%);
  --color-neutral-200: hsl(240, 10%, 85%);
  --color-neutral-100: hsl(240, 10%, 95%);
  
  // Semantic colors (10% of interface)
  --color-success: hsl(140, 100%, 50%);
  --color-warning: hsl(45, 100%, 50%);
  --color-error: hsl(0, 100%, 60%);
  --color-info: hsl(210, 100%, 60%);
  
  // Contrast ratios for hierarchy
  --contrast-primary: 15:1;     // Headlines
  --contrast-secondary: 10:1;   // Subheadings
  --contrast-body: 7:1;         // Body text
  --contrast-muted: 4.5:1;      // Supporting text
  --contrast-disabled: 3:1;     // Disabled state
}
```

### 3.2 Color Application Rules

```scss
// Primary color = Call to action ONLY
.cta-button {
  background: linear-gradient(135deg, 
    var(--color-primary) 0%, 
    var(--color-primary-dark) 100%
  );
  
  // Nothing else gets primary color at full saturation
}

// Secondary color = Interactive elements
.interactive {
  color: var(--color-secondary);
  
  &:hover {
    color: var(--color-secondary-light);
  }
  
  &:active {
    color: var(--color-secondary-dark);
  }
}

// Text hierarchy through opacity
.text-hierarchy {
  // Primary text - full opacity
  color: rgba(255, 255, 255, 1);
  
  // Secondary text - reduced opacity
  .secondary {
    color: rgba(255, 255, 255, 0.7);
  }
  
  // Tertiary text - further reduced
  .tertiary {
    color: rgba(255, 255, 255, 0.5);
  }
  
  // Disabled text - minimum legal contrast
  .disabled {
    color: rgba(255, 255, 255, 0.3);
  }
}
```

---

## 4. SCANNING PATTERNS & READING FLOW

### 4.1 The F-Pattern Implementation

```scss
.f-pattern-layout {
  // Top horizontal: Most important info
  .header-strip {
    display: flex;
    justify-content: space-between;
    padding: var(--space-lg);
    background: linear-gradient(
      90deg,
      rgba(255, 255, 255, 0.05) 0%,
      transparent 70%
    );
    
    // Logo/Title gets highest weight
    .logo {
      font-size: var(--text-2xl);
      font-weight: 900;
    }
    
    // Navigation gets secondary weight
    nav {
      font-size: var(--text-base);
      font-weight: 600;
    }
  }
  
  // Second horizontal: Key features
  .feature-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: var(--space-lg);
    padding: var(--space-xl) var(--space-lg);
    
    // Features decrease in visual weight left to right
    .feature {
      &:nth-child(1) { opacity: 1; font-weight: 700; }
      &:nth-child(2) { opacity: 0.9; font-weight: 600; }
      &:nth-child(3) { opacity: 0.8; font-weight: 500; }
      &:nth-child(n+4) { opacity: 0.7; font-weight: 400; }
    }
  }
  
  // Vertical scan: Left column content
  .content-area {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: var(--space-2xl);
    
    .main-content {
      // Progressive disclosure through spacing
      > * {
        margin-bottom: var(--space-lg);
        
        // Decreasing prominence
        &:nth-child(1) { font-size: var(--text-lg); }
        &:nth-child(2) { font-size: var(--text-base); }
        &:nth-child(n+3) { font-size: var(--text-sm); }
      }
    }
    
    .sidebar {
      // Lower visual weight
      opacity: 0.8;
      font-size: var(--text-sm);
    }
  }
}
```

### 4.2 The Z-Pattern for Landing Pages

```scss
.z-pattern-hero {
  display: grid;
  grid-template-columns: 1fr 1fr;
  min-height: 100vh;
  
  // Point 1: Top-left (Brand)
  .brand {
    grid-column: 1;
    grid-row: 1;
    align-self: start;
    font-size: var(--text-xl);
    font-weight: 900;
    padding: var(--space-xl);
  }
  
  // Point 2: Top-right (CTA)
  .primary-cta {
    grid-column: 2;
    grid-row: 1;
    align-self: start;
    justify-self: end;
    padding: var(--space-xl);
    
    button {
      font-size: var(--text-lg);
      padding: var(--space-md) var(--space-xl);
    }
  }
  
  // Diagonal: Hero content
  .hero-content {
    grid-column: 1 / -1;
    grid-row: 2;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: var(--space-4xl) var(--space-xl);
    
    h1 {
      font-size: var(--text-4xl);
      max-width: 20ch;
      margin-bottom: var(--space-lg);
    }
    
    p {
      font-size: var(--text-lg);
      max-width: 60ch;
      opacity: 0.8;
    }
  }
  
  // Point 3: Bottom-left (Social proof)
  .social-proof {
    grid-column: 1;
    grid-row: 3;
    align-self: end;
    padding: var(--space-xl);
    
    .testimonial {
      font-size: var(--text-sm);
      opacity: 0.7;
    }
  }
  
  // Point 4: Bottom-right (Secondary CTA)
  .secondary-cta {
    grid-column: 2;
    grid-row: 3;
    align-self: end;
    justify-self: end;
    padding: var(--space-xl);
    
    button {
      font-size: var(--text-base);
      font-weight: 600;
    }
  }
}
```

---

## 5. INFORMATION DENSITY OPTIMIZATION

### 5.1 The Progressive Disclosure System

```javascript
class ProgressiveDisclosure {
  constructor() {
    this.levels = {
      essential: 1,    // Always visible
      important: 2,    // Visible on hover/focus
      detailed: 3,     // Visible on click/tap
      expert: 4        // Hidden behind "advanced" toggle
    };
  }
  
  initializeCard(card) {
    const data = this.parseCardData(card);
    
    // Render essential information
    this.renderLevel(card, data.essential, 'always');
    
    // Add hover layer
    card.addEventListener('mouseenter', () => {
      this.renderLevel(card, data.important, 'hover');
    });
    
    // Add click expansion
    card.addEventListener('click', (e) => {
      if (!e.target.closest('.action')) {
        this.renderLevel(card, data.detailed, 'expanded');
        card.classList.toggle('expanded');
      }
    });
    
    // Expert mode toggle
    if (data.expert && this.userIsExpert()) {
      this.addExpertToggle(card, data.expert);
    }
  }
  
  renderLevel(element, data, state) {
    const container = element.querySelector(`.disclosure-${state}`);
    if (!container) return;
    
    // Animate in new information
    container.style.height = '0';
    container.innerHTML = this.formatData(data);
    
    // Trigger reflow
    container.offsetHeight;
    
    // Animate height
    container.style.height = container.scrollHeight + 'px';
    container.style.opacity = '0';
    
    requestAnimationFrame(() => {
      container.style.transition = 'all 300ms ease';
      container.style.opacity = '1';
    });
  }
}
```

### 5.2 Information Density by Context

```scss
// Mobile: Low density, high clarity
@media (max-width: 768px) {
  .info-card {
    // Larger touch targets
    min-height: 64px;
    padding: var(--space-lg);
    
    // Bigger fonts
    font-size: var(--text-base);
    line-height: var(--leading-relaxed);
    
    // More spacing
    > * + * {
      margin-top: var(--space-md);
    }
    
    // Hide secondary info
    .secondary-info {
      display: none;
    }
  }
}

// Desktop: Higher density, more info
@media (min-width: 1024px) {
  .info-card {
    // Compact layout
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: var(--space-md);
    padding: var(--space-md);
    
    // Smaller fonts acceptable
    font-size: var(--text-sm);
    line-height: var(--leading-normal);
    
    // Show all information
    .secondary-info {
      display: block;
      opacity: 0.7;
    }
  }
}

// Expert mode: Maximum density
.expert-mode {
  .info-card {
    // Ultra-compact
    padding: var(--space-sm);
    font-size: var(--text-xs);
    line-height: var(--leading-tight);
    
    // Show everything
    .tertiary-info,
    .metadata,
    .debug-info {
      display: block;
      opacity: 0.5;
    }
  }
}
```

---

## 6. VISUAL RHYTHM & REPETITION

### 6.1 Creating Rhythm Through Repetition

```scss
.rhythmic-layout {
  // Consistent card rhythm
  .card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: var(--space-lg);
    
    // Every 3rd card is featured
    .card:nth-child(3n+1) {
      grid-column: span 2;
      
      h3 {
        font-size: var(--text-2xl);
      }
    }
    
    // Creates visual rhythm: large-small-small-large-small-small
  }
  
  // Alternating content blocks
  .content-block {
    &:nth-child(odd) {
      display: grid;
      grid-template-columns: 2fr 1fr;
      
      .image {
        order: 2;
      }
    }
    
    &:nth-child(even) {
      display: grid;
      grid-template-columns: 1fr 2fr;
      
      .image {
        order: 1;
      }
    }
  }
  
  // Consistent spacing rhythm
  .section {
    padding: var(--space-4xl) 0;
    
    &:nth-child(2n) {
      background: rgba(255, 255, 255, 0.02);
    }
  }
}
```

### 6.2 Breaking Rhythm for Emphasis

```scss
.rhythm-breaker {
  // Standard rhythm
  .item {
    padding: var(--space-md);
    margin: var(--space-sm) 0;
    border-radius: 8px;
    
    // The breaker - draws immediate attention
    &.featured {
      padding: var(--space-xl);
      margin: var(--space-xl) calc(var(--space-lg) * -1);
      border-radius: 16px;
      background: linear-gradient(
        135deg,
        var(--color-primary-dark) 0%,
        var(--color-primary) 100%
      );
      transform: scale(1.05);
      box-shadow: var(--shadow-xl);
    }
  }
}
```

---

## 7. HIERARCHY ACCESSIBILITY

### 7.1 Semantic Structure

```html
<!-- Proper heading hierarchy -->
<article role="main">
  <h1>Page Title</h1> <!-- Only one per page -->
  
  <section aria-labelledby="section1">
    <h2 id="section1">Section Title</h2>
    
    <h3>Subsection Title</h3>
    <p>Content...</p>
    
    <h3>Another Subsection</h3>
    <p>More content...</p>
  </section>
  
  <section aria-labelledby="section2">
    <h2 id="section2">Another Section</h2>
    <!-- Never skip levels: h2 -> h4 is wrong -->
  </section>
</article>

<!-- Navigation hierarchy -->
<nav aria-label="Main navigation">
  <ul role="menubar">
    <li role="none">
      <a role="menuitem" href="/" aria-current="page">Home</a>
    </li>
    <li role="none">
      <a role="menuitem" href="/about">About</a>
    </li>
  </ul>
</nav>
```

### 7.2 Screen Reader Optimization

```scss
// Visual hierarchy that works for screen readers
.sr-hierarchy {
  // Visually hidden but screen reader accessible
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border-width: 0;
  }
  
  // Focus brings it back into view
  .sr-only-focusable:focus {
    position: static;
    width: auto;
    height: auto;
    padding: inherit;
    margin: inherit;
    overflow: visible;
    clip: auto;
    white-space: normal;
  }
  
  // Important content announced first
  .important {
    // Move to top for screen readers
    order: -1;
    
    // But visually position elsewhere
    @media screen {
      order: initial;
    }
  }
}
```

---

## 8. IMPLEMENTATION CHECKLIST

### Audit Phase (Day 1-2)
- [ ] Map current visual weights
- [ ] Document heading hierarchy
- [ ] Measure contrast ratios
- [ ] Identify scanning patterns
- [ ] Test with screen readers

### Design Phase (Day 3-5)
- [ ] Establish type scale
- [ ] Define spacing system
- [ ] Create color hierarchy
- [ ] Design information density levels
- [ ] Plan progressive disclosure

### Implementation Phase (Week 2)
- [ ] Implement typography system
- [ ] Apply spacing grid
- [ ] Build card components
- [ ] Add progressive disclosure
- [ ] Create rhythm patterns

### Validation Phase (Week 3)
- [ ] 5-second test (users find primary action)
- [ ] Scanning heat maps
- [ ] Screen reader testing
- [ ] Contrast validation
- [ ] Mobile hierarchy testing

---

## 9. SUCCESS METRICS

**Quantitative Measures**:
- Time to find primary action: <2 seconds
- Scanning efficiency: 80% follow intended path
- Information retention: +40% vs flat hierarchy
- Error rate: -60% on form completion
- Accessibility score: WCAG AAA compliant

**Qualitative Measures**:
- "Easy to scan" in feedback: >90%
- "Found what I needed quickly": >85%
- "Clear what to do next": >95%
- "Professional appearance": >90%

Remember: Hierarchy is not decoration. It's information architecture made visible. Every size, space, and color choice guides attention. Master this, and users navigate your interface without thinking.
