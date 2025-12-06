# LatticeForge Visual Design System Specification

## Document Purpose

This specification defines the complete visual design system for LatticeForge. It covers design tokens, typography, color system, iconography, component styling, spacing system, animation principles, and accessibility requirements. This document serves as the single source of truth for visual consistency across the platform.

---

## 1. Design Philosophy

### 1.1 Visual Principles

**Clarity Over Decoration**
Every visual element must serve a purpose. If an element doesn't improve comprehension, navigation, or emotional resonance, remove it. Research tools demand visual quietness so content can speak.

**Density Without Clutter**
Researchers work with large amounts of information. The interface should accommodate high information density while maintaining clear hierarchy and breathing room. Pack information efficiently; don't spread it thin.

**Trust Through Consistency**
Users dealing with AI-generated content need to trust the interface. Visual consistency breeds familiarity; familiarity breeds trust. Every button, every card, every interaction should feel inevitable.

**Light Interface, Deep Content**
The chrome should recede. Navigation, controls, and structural elements should be visually lightweight. User content—papers, insights, graphs—deserves visual prominence.

**Motion With Purpose**
Animation should communicate state changes, not entertain. Every transition should answer a question: "Where did that come from?" or "Where did that go?" Gratuitous animation erodes professional credibility.

### 1.2 Brand Personality

| Attribute | Expression |
|-----------|------------|
| Intelligent | Refined typography, precise spacing, subtle sophistication |
| Trustworthy | Consistent patterns, clear feedback, no surprises |
| Efficient | Compact layouts, keyboard-first, information density |
| Approachable | Warm neutrals, friendly empty states, progressive disclosure |
| Modern | Contemporary sans-serif, generous whitespace, subtle shadows |

### 1.3 Competitive Positioning

LatticeForge sits between:
- **Academic tools** (ResearchRabbit, Zotero): More polished, less utilitarian
- **Enterprise knowledge tools** (Notion, Confluence): More focused, less generic
- **AI interfaces** (ChatGPT, Perplexity): More structured, less conversational

Visual positioning: Premium professional tool, not a toy, not a legacy system.

---

## 2. Design Tokens

### 2.1 Token Architecture

Design tokens are organized in three tiers:

1. **Primitive Tokens**: Raw values (colors, sizes, fonts)
2. **Semantic Tokens**: Purpose-driven mappings (background-primary, text-muted)
3. **Component Tokens**: Component-specific values (button-padding, card-radius)

```
Primitive      →    Semantic        →    Component
gray-900            text-primary          button-text
blue-500            interactive-primary   link-color
16px                spacing-md            card-padding
```

### 2.2 Token Naming Convention

```
[category]-[property]-[variant]-[state]

Examples:
color-background-surface
color-text-primary
color-border-muted
spacing-padding-lg
radius-card-default
shadow-elevation-2
motion-duration-fast
```

### 2.3 Core Token Definitions

```css
/* ===========================================
   PRIMITIVE TOKENS
   =========================================== */

/* Colors - Gray Scale */
--gray-50: #FAFAFA;
--gray-100: #F5F5F5;
--gray-200: #E5E5E5;
--gray-300: #D4D4D4;
--gray-400: #A3A3A3;
--gray-500: #737373;
--gray-600: #525252;
--gray-700: #404040;
--gray-800: #262626;
--gray-900: #171717;
--gray-950: #0A0A0A;

/* Colors - Primary (Deep Blue) */
--blue-50: #EFF6FF;
--blue-100: #DBEAFE;
--blue-200: #BFDBFE;
--blue-300: #93C5FD;
--blue-400: #60A5FA;
--blue-500: #3B82F6;
--blue-600: #2563EB;
--blue-700: #1D4ED8;
--blue-800: #1E40AF;
--blue-900: #1E3A8A;

/* Colors - Accent (Teal for insights) */
--teal-50: #F0FDFA;
--teal-100: #CCFBF1;
--teal-200: #99F6E4;
--teal-300: #5EEAD4;
--teal-400: #2DD4BF;
--teal-500: #14B8A6;
--teal-600: #0D9488;
--teal-700: #0F766E;
--teal-800: #115E59;
--teal-900: #134E4A;

/* Colors - Semantic Status */
--red-500: #EF4444;
--red-600: #DC2626;
--amber-500: #F59E0B;
--amber-600: #D97706;
--green-500: #22C55E;
--green-600: #16A34A;

/* Typography Scale */
--font-size-xs: 0.75rem;    /* 12px */
--font-size-sm: 0.875rem;   /* 14px */
--font-size-base: 1rem;     /* 16px */
--font-size-lg: 1.125rem;   /* 18px */
--font-size-xl: 1.25rem;    /* 20px */
--font-size-2xl: 1.5rem;    /* 24px */
--font-size-3xl: 1.875rem;  /* 30px */
--font-size-4xl: 2.25rem;   /* 36px */

/* Line Heights */
--line-height-tight: 1.25;
--line-height-snug: 1.375;
--line-height-normal: 1.5;
--line-height-relaxed: 1.625;
--line-height-loose: 2;

/* Font Weights */
--font-weight-normal: 400;
--font-weight-medium: 500;
--font-weight-semibold: 600;
--font-weight-bold: 700;

/* Spacing Scale (4px base) */
--spacing-0: 0;
--spacing-1: 0.25rem;   /* 4px */
--spacing-2: 0.5rem;    /* 8px */
--spacing-3: 0.75rem;   /* 12px */
--spacing-4: 1rem;      /* 16px */
--spacing-5: 1.25rem;   /* 20px */
--spacing-6: 1.5rem;    /* 24px */
--spacing-8: 2rem;      /* 32px */
--spacing-10: 2.5rem;   /* 40px */
--spacing-12: 3rem;     /* 48px */
--spacing-16: 4rem;     /* 64px */
--spacing-20: 5rem;     /* 80px */

/* Border Radius */
--radius-none: 0;
--radius-sm: 0.125rem;  /* 2px */
--radius-default: 0.25rem;  /* 4px */
--radius-md: 0.375rem;  /* 6px */
--radius-lg: 0.5rem;    /* 8px */
--radius-xl: 0.75rem;   /* 12px */
--radius-2xl: 1rem;     /* 16px */
--radius-full: 9999px;

/* Shadows */
--shadow-xs: 0 1px 2px 0 rgb(0 0 0 / 0.05);
--shadow-sm: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
--shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
--shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
--shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);

/* Motion */
--duration-instant: 0ms;
--duration-fast: 100ms;
--duration-normal: 200ms;
--duration-slow: 300ms;
--duration-slower: 500ms;

--ease-default: cubic-bezier(0.4, 0, 0.2, 1);
--ease-in: cubic-bezier(0.4, 0, 1, 1);
--ease-out: cubic-bezier(0, 0, 0.2, 1);
--ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);
--ease-bounce: cubic-bezier(0.34, 1.56, 0.64, 1);

/* Z-Index Scale */
--z-base: 0;
--z-dropdown: 100;
--z-sticky: 200;
--z-fixed: 300;
--z-modal-backdrop: 400;
--z-modal: 500;
--z-popover: 600;
--z-tooltip: 700;
--z-toast: 800;
```

### 2.4 Semantic Token Mappings

```css
/* ===========================================
   SEMANTIC TOKENS - LIGHT MODE
   =========================================== */

/* Backgrounds */
--color-bg-app: var(--gray-50);
--color-bg-surface: #FFFFFF;
--color-bg-surface-raised: #FFFFFF;
--color-bg-surface-overlay: #FFFFFF;
--color-bg-muted: var(--gray-100);
--color-bg-subtle: var(--gray-50);
--color-bg-inverse: var(--gray-900);

/* Text */
--color-text-primary: var(--gray-900);
--color-text-secondary: var(--gray-600);
--color-text-muted: var(--gray-500);
--color-text-disabled: var(--gray-400);
--color-text-inverse: #FFFFFF;
--color-text-link: var(--blue-600);
--color-text-link-hover: var(--blue-700);

/* Borders */
--color-border-default: var(--gray-200);
--color-border-muted: var(--gray-100);
--color-border-strong: var(--gray-300);
--color-border-focus: var(--blue-500);

/* Interactive */
--color-interactive-primary: var(--blue-600);
--color-interactive-primary-hover: var(--blue-700);
--color-interactive-primary-active: var(--blue-800);
--color-interactive-secondary: var(--gray-100);
--color-interactive-secondary-hover: var(--gray-200);

/* Status */
--color-status-success: var(--green-600);
--color-status-success-bg: var(--green-50);
--color-status-warning: var(--amber-600);
--color-status-warning-bg: var(--amber-50);
--color-status-error: var(--red-600);
--color-status-error-bg: var(--red-50);
--color-status-info: var(--blue-600);
--color-status-info-bg: var(--blue-50);

/* Accent (Insights) */
--color-accent-primary: var(--teal-600);
--color-accent-primary-subtle: var(--teal-50);
--color-accent-primary-hover: var(--teal-700);

/* ===========================================
   SEMANTIC TOKENS - DARK MODE
   =========================================== */

@media (prefers-color-scheme: dark) {
  :root {
    --color-bg-app: var(--gray-950);
    --color-bg-surface: var(--gray-900);
    --color-bg-surface-raised: var(--gray-800);
    --color-bg-surface-overlay: var(--gray-800);
    --color-bg-muted: var(--gray-800);
    --color-bg-subtle: var(--gray-900);
    --color-bg-inverse: var(--gray-50);

    --color-text-primary: var(--gray-50);
    --color-text-secondary: var(--gray-300);
    --color-text-muted: var(--gray-400);
    --color-text-disabled: var(--gray-600);
    --color-text-inverse: var(--gray-900);
    --color-text-link: var(--blue-400);
    --color-text-link-hover: var(--blue-300);

    --color-border-default: var(--gray-700);
    --color-border-muted: var(--gray-800);
    --color-border-strong: var(--gray-600);

    --color-interactive-primary: var(--blue-500);
    --color-interactive-primary-hover: var(--blue-400);
    --color-interactive-secondary: var(--gray-800);
    --color-interactive-secondary-hover: var(--gray-700);
  }
}
```

---

## 3. Typography System

### 3.1 Font Stack

**Primary Font: Inter**
- Modern, highly legible sans-serif
- Excellent for UI text and long-form reading
- Strong number forms for data display
- Open source, widely supported

```css
--font-family-sans: 'Inter', -apple-system, BlinkMacSystemFont,
    'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
```

**Monospace Font: JetBrains Mono**
- For code, citations, technical identifiers
- Clear distinction between similar characters
- Coding ligatures disabled for clarity

```css
--font-family-mono: 'JetBrains Mono', 'SF Mono', 'Consolas',
    'Liberation Mono', 'Menlo', monospace;
```

### 3.2 Type Scale

| Name | Size | Weight | Line Height | Use Case |
|------|------|--------|-------------|----------|
| display-lg | 36px | 700 | 1.2 | Marketing headlines only |
| display | 30px | 700 | 1.25 | Page titles |
| heading-1 | 24px | 600 | 1.3 | Section headers |
| heading-2 | 20px | 600 | 1.35 | Subsection headers |
| heading-3 | 18px | 600 | 1.4 | Card titles |
| body-lg | 18px | 400 | 1.6 | Lead paragraphs |
| body | 16px | 400 | 1.5 | Primary body text |
| body-sm | 14px | 400 | 1.5 | Secondary text, captions |
| caption | 12px | 400 | 1.4 | Labels, metadata |
| overline | 12px | 600 | 1.3 | Section labels, uppercase |

### 3.3 Typography Components

```css
/* Headings */
.text-display-lg {
  font-size: var(--font-size-4xl);
  font-weight: var(--font-weight-bold);
  line-height: var(--line-height-tight);
  letter-spacing: -0.02em;
}

.text-display {
  font-size: var(--font-size-3xl);
  font-weight: var(--font-weight-bold);
  line-height: var(--line-height-tight);
  letter-spacing: -0.01em;
}

.text-heading-1 {
  font-size: var(--font-size-2xl);
  font-weight: var(--font-weight-semibold);
  line-height: var(--line-height-snug);
  letter-spacing: -0.01em;
}

.text-heading-2 {
  font-size: var(--font-size-xl);
  font-weight: var(--font-weight-semibold);
  line-height: var(--line-height-snug);
}

.text-heading-3 {
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
  line-height: var(--line-height-snug);
}

/* Body Text */
.text-body-lg {
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-normal);
  line-height: var(--line-height-relaxed);
}

.text-body {
  font-size: var(--font-size-base);
  font-weight: var(--font-weight-normal);
  line-height: var(--line-height-normal);
}

.text-body-sm {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-normal);
  line-height: var(--line-height-normal);
}

/* Utility */
.text-caption {
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-normal);
  line-height: var(--line-height-snug);
  color: var(--color-text-muted);
}

.text-overline {
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-semibold);
  line-height: var(--line-height-snug);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-text-secondary);
}

.text-mono {
  font-family: var(--font-family-mono);
  font-size: 0.9em; /* Slightly smaller than surrounding text */
}
```

### 3.4 Text Color Classes

```css
.text-primary { color: var(--color-text-primary); }
.text-secondary { color: var(--color-text-secondary); }
.text-muted { color: var(--color-text-muted); }
.text-disabled { color: var(--color-text-disabled); }
.text-inverse { color: var(--color-text-inverse); }
.text-link { color: var(--color-text-link); }
.text-success { color: var(--color-status-success); }
.text-warning { color: var(--color-status-warning); }
.text-error { color: var(--color-status-error); }
```

---

## 4. Color System

### 4.1 Color Usage Guidelines

**Primary Blue (#2563EB)**
- Primary actions (buttons, links)
- Selected states
- Progress indicators
- Focus rings

**Accent Teal (#0D9488)**
- Insights and discoveries
- AI-generated content indicators
- Connection highlights in graphs
- Feature callouts

**Grays**
- 50-200: Backgrounds, subtle separators
- 300-400: Borders, disabled states
- 500-600: Secondary text, icons
- 700-900: Primary text, headings

**Status Colors**
- Red: Errors, destructive actions, conflicts
- Amber: Warnings, processing states
- Green: Success, confirmations, positive metrics
- Blue: Information, tips, neutral highlights

### 4.2 Color Application Matrix

| Element | Light Mode | Dark Mode |
|---------|------------|-----------|
| App background | gray-50 | gray-950 |
| Card background | white | gray-900 |
| Primary text | gray-900 | gray-50 |
| Secondary text | gray-600 | gray-300 |
| Border | gray-200 | gray-700 |
| Primary button bg | blue-600 | blue-500 |
| Primary button text | white | white |
| Link | blue-600 | blue-400 |
| Insight indicator | teal-600 | teal-400 |
| Error | red-600 | red-400 |

### 4.3 Color Contrast Requirements

All text must meet WCAG 2.1 AA:
- Normal text: 4.5:1 contrast ratio minimum
- Large text (18px+ or 14px bold): 3:1 minimum
- UI components: 3:1 minimum

| Combination | Light Mode Ratio | Dark Mode Ratio |
|-------------|------------------|-----------------|
| Primary text on surface | 15.8:1 | 14.5:1 |
| Secondary text on surface | 5.7:1 | 7.2:1 |
| Muted text on surface | 4.6:1 | 5.1:1 |
| Primary button | 4.5:1 | 4.5:1 |
| Link on surface | 4.6:1 | 5.3:1 |

### 4.4 Data Visualization Palette

For graphs, charts, and entity type differentiation:

```css
/* Categorical palette (up to 8 categories) */
--viz-1: #3B82F6;  /* Blue - Sources */
--viz-2: #14B8A6;  /* Teal - Insights */
--viz-3: #F59E0B;  /* Amber - Entities */
--viz-4: #8B5CF6;  /* Purple - Concepts */
--viz-5: #EC4899;  /* Pink - People */
--viz-6: #10B981;  /* Green - Organizations */
--viz-7: #6366F1;  /* Indigo - Methods */
--viz-8: #F97316;  /* Orange - Misc */

/* Sequential palette (for intensity) */
--viz-seq-1: #DBEAFE;
--viz-seq-2: #BFDBFE;
--viz-seq-3: #93C5FD;
--viz-seq-4: #60A5FA;
--viz-seq-5: #3B82F6;
--viz-seq-6: #2563EB;
--viz-seq-7: #1D4ED8;

/* Diverging palette (for comparison) */
--viz-neg-3: #DC2626;
--viz-neg-2: #F87171;
--viz-neg-1: #FCA5A5;
--viz-neutral: #E5E5E5;
--viz-pos-1: #86EFAC;
--viz-pos-2: #22C55E;
--viz-pos-3: #16A34A;
```

---

## 5. Spacing System

### 5.1 Base Unit

All spacing derives from a 4px base unit. This creates visual rhythm and predictable relationships.

```
4px  → micro spacing (icon-text gaps)
8px  → compact spacing (inline elements)
12px → small spacing (list items)
16px → base spacing (default padding)
24px → medium spacing (section gaps)
32px → large spacing (card separation)
48px → extra large spacing (major sections)
```

### 5.2 Spacing Applications

**Component Internal Spacing:**
```css
/* Button padding */
--button-padding-sm: var(--spacing-2) var(--spacing-3);  /* 8px 12px */
--button-padding-md: var(--spacing-2) var(--spacing-4);  /* 8px 16px */
--button-padding-lg: var(--spacing-3) var(--spacing-6);  /* 12px 24px */

/* Card padding */
--card-padding-sm: var(--spacing-3);   /* 12px */
--card-padding-md: var(--spacing-4);   /* 16px */
--card-padding-lg: var(--spacing-6);   /* 24px */

/* Input padding */
--input-padding: var(--spacing-2) var(--spacing-3);  /* 8px 12px */

/* Modal padding */
--modal-padding: var(--spacing-6);  /* 24px */
```

**Layout Spacing:**
```css
/* Content margins */
--page-margin-x: var(--spacing-6);
--page-margin-x-lg: var(--spacing-10);

/* Section spacing */
--section-gap: var(--spacing-12);
--subsection-gap: var(--spacing-8);

/* Grid gaps */
--grid-gap-sm: var(--spacing-3);
--grid-gap-md: var(--spacing-4);
--grid-gap-lg: var(--spacing-6);
```

### 5.3 Responsive Spacing

Spacing should breathe on larger screens:

```css
/* Mobile (default) */
:root {
  --content-max-width: 100%;
  --page-padding: var(--spacing-4);
  --section-margin: var(--spacing-8);
}

/* Tablet (768px+) */
@media (min-width: 768px) {
  :root {
    --page-padding: var(--spacing-6);
    --section-margin: var(--spacing-10);
  }
}

/* Desktop (1024px+) */
@media (min-width: 1024px) {
  :root {
    --content-max-width: 1200px;
    --page-padding: var(--spacing-8);
    --section-margin: var(--spacing-12);
  }
}

/* Wide (1440px+) */
@media (min-width: 1440px) {
  :root {
    --content-max-width: 1400px;
    --page-padding: var(--spacing-10);
  }
}
```

---

## 6. Iconography

### 6.1 Icon System

**Primary Icon Set: Lucide Icons**
- Clean, minimal line icons
- Consistent 24x24 base size
- 1.5px stroke weight
- Open source, regularly updated

**Icon Sizes:**
```css
--icon-xs: 12px;   /* Inline badges, status dots */
--icon-sm: 16px;   /* Button icons, inline */
--icon-md: 20px;   /* Default size */
--icon-lg: 24px;   /* Navigation, standalone */
--icon-xl: 32px;   /* Feature highlights */
--icon-2xl: 48px;  /* Empty states */
```

### 6.2 Icon Usage Guidelines

**With Text:**
- Icon precedes text (for actions)
- Icon follows text (for external links, dropdowns)
- 8px gap between icon and text
- Icon should be vertically centered with text baseline

**Standalone:**
- Must have accessible label (aria-label or title)
- Minimum touch target: 44x44px
- Visual icon can be smaller within target

**Color:**
- Icons inherit text color by default
- Interactive icons: use `--color-text-secondary`
- Interactive icons on hover: use `--color-text-primary`
- Decorative icons: use `--color-text-muted`

### 6.3 Custom Icons

For LatticeForge-specific concepts:

| Concept | Icon Description |
|---------|------------------|
| Research Stream | Flowing lines converging |
| Insight | Lightbulb with nodes |
| Entity | Circle with tag |
| Synthesis | Document with connecting lines |
| Graph Node | Circle with edges |
| Confidence | Stacked dots (like signal strength) |

These should be designed to match Lucide's style:
- 24x24 viewBox
- 1.5px stroke
- Round line caps and joins
- No fills (line icons only)

---

## 7. Component Specifications

### 7.1 Buttons

**Variants:**
```
┌─────────────────────────────────────────────────────────────┐
│ Primary    │ Secondary   │ Ghost       │ Destructive       │
│ [Filled]   │ [Outlined]  │ [Text only] │ [Red filled]      │
└─────────────────────────────────────────────────────────────┘
```

**Sizes:**
| Size | Height | Padding | Font Size | Icon Size |
|------|--------|---------|-----------|-----------|
| sm | 32px | 8px 12px | 14px | 16px |
| md | 40px | 8px 16px | 14px | 20px |
| lg | 48px | 12px 24px | 16px | 20px |

**States:**
```css
/* Primary Button */
.btn-primary {
  background: var(--color-interactive-primary);
  color: var(--color-text-inverse);
  border: none;
  border-radius: var(--radius-md);
  font-weight: var(--font-weight-medium);
  transition: background var(--duration-fast) var(--ease-default);
}

.btn-primary:hover {
  background: var(--color-interactive-primary-hover);
}

.btn-primary:active {
  background: var(--color-interactive-primary-active);
}

.btn-primary:focus-visible {
  outline: 2px solid var(--color-border-focus);
  outline-offset: 2px;
}

.btn-primary:disabled {
  background: var(--gray-300);
  color: var(--gray-500);
  cursor: not-allowed;
}
```

### 7.2 Cards

**Base Card:**
```css
.card {
  background: var(--color-bg-surface);
  border: 1px solid var(--color-border-default);
  border-radius: var(--radius-lg);
  padding: var(--spacing-4);
  transition: box-shadow var(--duration-fast) var(--ease-default);
}

.card:hover {
  box-shadow: var(--shadow-md);
}

.card-interactive:hover {
  border-color: var(--color-border-strong);
  cursor: pointer;
}

.card-selected {
  border-color: var(--color-interactive-primary);
  box-shadow: 0 0 0 1px var(--color-interactive-primary);
}
```

**Card Anatomy:**
```
┌────────────────────────────────────────────┐
│ [Thumbnail] │ Header Row             │ [⋮] │
│             │ Metadata line          │     │
│             │ Badges/tags            │     │
├────────────────────────────────────────────┤
│ Description or preview content             │
│ (optional, may be truncated)               │
├────────────────────────────────────────────┤
│ [Action 1]  [Action 2]      Timestamp      │
└────────────────────────────────────────────┘
```

### 7.3 Form Inputs

**Text Input:**
```css
.input {
  height: 40px;
  padding: var(--spacing-2) var(--spacing-3);
  font-size: var(--font-size-sm);
  color: var(--color-text-primary);
  background: var(--color-bg-surface);
  border: 1px solid var(--color-border-default);
  border-radius: var(--radius-md);
  transition: border-color var(--duration-fast) var(--ease-default),
              box-shadow var(--duration-fast) var(--ease-default);
}

.input:hover {
  border-color: var(--color-border-strong);
}

.input:focus {
  border-color: var(--color-border-focus);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
  outline: none;
}

.input::placeholder {
  color: var(--color-text-muted);
}

.input:disabled {
  background: var(--color-bg-muted);
  color: var(--color-text-disabled);
  cursor: not-allowed;
}

.input-error {
  border-color: var(--color-status-error);
}

.input-error:focus {
  box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.15);
}
```

**Input with Label:**
```
┌─ Label ──────────────────────────────────┐
│                                          │
│  [Input field                         ]  │
│                                          │
│  Helper text or error message            │
└──────────────────────────────────────────┘
```

### 7.4 Modals and Dialogs

**Modal Sizes:**
| Size | Max Width | Use Case |
|------|-----------|----------|
| sm | 400px | Confirmations, simple forms |
| md | 560px | Standard forms, settings |
| lg | 720px | Complex content, previews |
| xl | 900px | Multi-step wizards |
| full | 100% - 48px | Full-page overlays |

**Modal Structure:**
```css
.modal-backdrop {
  background: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(4px);
}

.modal {
  background: var(--color-bg-surface);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-xl);
  max-height: calc(100vh - 48px);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.modal-header {
  padding: var(--spacing-6);
  border-bottom: 1px solid var(--color-border-muted);
}

.modal-body {
  padding: var(--spacing-6);
  overflow-y: auto;
}

.modal-footer {
  padding: var(--spacing-4) var(--spacing-6);
  border-top: 1px solid var(--color-border-muted);
  display: flex;
  justify-content: flex-end;
  gap: var(--spacing-3);
}
```

### 7.5 Navigation

**Sidebar Navigation:**
```css
.nav-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-3);
  padding: var(--spacing-2) var(--spacing-3);
  border-radius: var(--radius-md);
  color: var(--color-text-secondary);
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-medium);
  transition: all var(--duration-fast) var(--ease-default);
}

.nav-item:hover {
  background: var(--color-bg-muted);
  color: var(--color-text-primary);
}

.nav-item-active {
  background: var(--color-interactive-primary);
  color: var(--color-text-inverse);
}

.nav-item-active:hover {
  background: var(--color-interactive-primary-hover);
  color: var(--color-text-inverse);
}
```

### 7.6 Tags and Badges

**Entity Tags:**
```css
.tag {
  display: inline-flex;
  align-items: center;
  gap: var(--spacing-1);
  padding: 2px var(--spacing-2);
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-medium);
  border-radius: var(--radius-sm);
  background: var(--color-bg-muted);
  color: var(--color-text-secondary);
}

.tag-interactive:hover {
  background: var(--color-bg-subtle);
  cursor: pointer;
}

/* Entity type variants */
.tag-person { background: #FDE68A; color: #92400E; }
.tag-organization { background: #D1FAE5; color: #065F46; }
.tag-concept { background: #DDD6FE; color: #5B21B6; }
.tag-method { background: #CFFAFE; color: #0E7490; }
```

**Status Badges:**
```css
.badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 20px;
  height: 20px;
  padding: 0 var(--spacing-2);
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-semibold);
  border-radius: var(--radius-full);
}

.badge-default { background: var(--gray-200); color: var(--gray-700); }
.badge-primary { background: var(--blue-100); color: var(--blue-700); }
.badge-success { background: var(--green-100); color: var(--green-700); }
.badge-warning { background: var(--amber-100); color: var(--amber-700); }
.badge-error { background: var(--red-100); color: var(--red-700); }
```

---

## 8. Animation and Motion

### 8.1 Motion Principles

1. **Purposeful**: Animation communicates something (state change, relationship, direction)
2. **Quick**: Most transitions under 200ms; complex sequences under 500ms
3. **Natural**: Use easing curves that feel physical, not robotic
4. **Reducible**: Honor prefers-reduced-motion

### 8.2 Standard Transitions

```css
/* Micro-interactions (hover, focus) */
--transition-micro: var(--duration-fast) var(--ease-default);

/* UI state changes (showing/hiding) */
--transition-normal: var(--duration-normal) var(--ease-default);

/* Layout changes (expansion, collapse) */
--transition-layout: var(--duration-slow) var(--ease-out);

/* Entering elements */
--transition-enter: var(--duration-normal) var(--ease-out);

/* Exiting elements */
--transition-exit: var(--duration-fast) var(--ease-in);
```

### 8.3 Common Animations

**Fade In:**
```css
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.animate-fade-in {
  animation: fadeIn var(--duration-normal) var(--ease-out);
}
```

**Slide In (from bottom):**
```css
@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.animate-slide-up {
  animation: slideUp var(--duration-normal) var(--ease-out);
}
```

**Scale In (for modals):**
```css
@keyframes scaleIn {
  from {
    opacity: 0;
    transform: scale(0.95);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

.animate-scale-in {
  animation: scaleIn var(--duration-normal) var(--ease-out);
}
```

**Skeleton Loading:**
```css
@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

.skeleton {
  background: linear-gradient(
    90deg,
    var(--gray-200) 0%,
    var(--gray-100) 50%,
    var(--gray-200) 100%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite linear;
}
```

### 8.4 Reduced Motion

```css
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
```

---

## 9. Accessibility Checklist

### 9.1 Color and Contrast

- [ ] All text meets 4.5:1 contrast (normal) or 3:1 (large)
- [ ] Interactive elements have 3:1 contrast against background
- [ ] Color is not the only indicator of meaning
- [ ] Focus states are clearly visible
- [ ] Links are distinguishable from body text (not just by color)

### 9.2 Typography

- [ ] Base font size is 16px minimum
- [ ] Line height is at least 1.5 for body text
- [ ] Text can be resized to 200% without loss of content
- [ ] No justified text (creates uneven spacing)
- [ ] Paragraph width doesn't exceed 80 characters

### 9.3 Interactive Elements

- [ ] All interactive elements have visible focus states
- [ ] Touch targets are at least 44x44px
- [ ] Hover states don't convey essential information
- [ ] Disabled elements are visually distinct
- [ ] Error states use icon/text, not just color

### 9.4 Motion

- [ ] No content flashes more than 3 times per second
- [ ] Motion respects prefers-reduced-motion
- [ ] Auto-playing content can be paused
- [ ] Animations don't last more than 5 seconds

### 9.5 Images and Icons

- [ ] All images have alt text (or aria-hidden if decorative)
- [ ] Icon buttons have accessible labels
- [ ] Complex images have long descriptions if needed
- [ ] Icons aren't the only indicator of meaning

---

## 10. Dark Mode Implementation

### 10.1 Strategy

LatticeForge supports system-preference-based dark mode with optional manual override.

**Detection:**
```javascript
// System preference
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

// User preference (stored)
const userPreference = localStorage.getItem('theme'); // 'light', 'dark', or null

// Applied theme
const theme = userPreference || (prefersDark ? 'dark' : 'light');
```

**Implementation:**
```css
:root {
  /* Light mode values */
  --color-bg-app: var(--gray-50);
  /* ... */
}

[data-theme="dark"] {
  /* Dark mode overrides */
  --color-bg-app: var(--gray-950);
  /* ... */
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    /* Dark mode for system preference when no explicit choice */
    --color-bg-app: var(--gray-950);
    /* ... */
  }
}
```

### 10.2 Dark Mode Specific Considerations

**Shadows:**
- Reduce shadow intensity in dark mode
- Consider using subtle borders instead of shadows
- Elevated surfaces can be slightly lighter, not shadowed

**Images and Graphics:**
- Avoid pure black backgrounds behind light images
- Consider providing dark mode variants for illustrations
- Apply subtle background to screenshots

**Data Visualization:**
- Reduce color saturation slightly
- Ensure chart colors remain distinguishable
- Avoid colors that vibrate against dark backgrounds

---

## 11. Responsive Design Specifications

### 11.1 Breakpoint System

```css
/* Mobile first approach */
/* Default: 0-767px (Mobile) */

@media (min-width: 768px) { /* Tablet: 768-1023px */ }
@media (min-width: 1024px) { /* Desktop: 1024-1439px */ }
@media (min-width: 1440px) { /* Wide: 1440px+ */ }
```

### 11.2 Component Responsive Behavior

| Component | Mobile | Tablet | Desktop |
|-----------|--------|--------|---------|
| Navigation | Bottom tabs | Collapsible sidebar | Fixed sidebar |
| Cards | Full width stack | 2-column grid | 3-column grid |
| Modals | Full screen | Centered, 80% width | Centered, fixed width |
| Tables | Card view | Horizontal scroll | Full table |
| Graph | Simplified | Full, bottom panel | Full, side panel |

### 11.3 Touch Adaptations

```css
/* Larger touch targets on touch devices */
@media (pointer: coarse) {
  .btn { min-height: 48px; }
  .input { min-height: 48px; }
  .nav-item { min-height: 48px; }
  .tag { min-height: 32px; }
}

/* Hover states only on hover-capable devices */
@media (hover: hover) {
  .card:hover { box-shadow: var(--shadow-md); }
  .btn:hover { background: var(--color-interactive-primary-hover); }
}
```

---

## 12. Asset Specifications

### 12.1 Logo Usage

**Primary Logo:**
- Minimum size: 32px height
- Clear space: Equal to height of "L" on all sides
- No rotation, no effects, no modifications

**Logo Variants:**
- Full color (primary)
- Monochrome (for single-color applications)
- White (for dark backgrounds)
- Icon only (for compact spaces, minimum 24px)

### 12.2 Favicon and App Icons

| Context | Size | Format |
|---------|------|--------|
| Favicon | 32x32, 16x16 | ICO, PNG |
| Apple Touch Icon | 180x180 | PNG |
| Android Chrome | 192x192, 512x512 | PNG |
| Windows Tile | 150x150, 310x310 | PNG |
| OG Image | 1200x630 | PNG |

### 12.3 Illustration Style

For empty states, onboarding, and feature highlights:

- Line-art style, consistent with icon system
- Limited color palette (primary blue, accent teal, neutrals)
- Abstract/conceptual rather than literal
- No human figures (to avoid demographic assumptions)
- Subtle, professional tone

---

## 13. Quality Assurance Checklist

### 13.1 Before Design Handoff

- [ ] All states documented (default, hover, active, focus, disabled, error)
- [ ] Responsive behavior specified
- [ ] Dark mode variants provided
- [ ] Spacing consistent with token system
- [ ] Colors from approved palette only
- [ ] Typography matches type scale
- [ ] Icons from approved set or custom specs provided
- [ ] Accessibility requirements noted

### 13.2 Design File Standards

**Figma Organization:**
```
📁 LatticeForge Design System
├── 📄 Cover
├── 📄 Foundations
│   ├── Colors
│   ├── Typography
│   ├── Spacing
│   └── Icons
├── 📄 Components
│   ├── Buttons
│   ├── Inputs
│   ├── Cards
│   └── ...
├── 📄 Patterns
│   ├── Navigation
│   ├── Data Display
│   └── Feedback
└── 📄 Pages
    ├── Dashboard
    ├── Stream Detail
    └── ...
```

**Naming Conventions:**
- Layers: `component/variant/state` (e.g., `button/primary/hover`)
- Frames: `PageName/ViewName` (e.g., `Dashboard/Mobile`)
- Components: PascalCase (e.g., `ButtonPrimary`)

---

*This design system is a living document. As the product evolves, update tokens, add components, and refine patterns while maintaining the core principles of clarity, consistency, and accessibility.*
