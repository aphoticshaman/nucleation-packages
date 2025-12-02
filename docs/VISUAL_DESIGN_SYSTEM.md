# LatticeForge Visual Design System
## A Designer's Guide to Data-Rich Intelligence Interfaces

---

## 1. THE FIVE Ws + HOW OF ART PRESENTATION

### WHO sees what?

| Tier | Persona | Visual Needs | Emotional Goal |
|------|---------|--------------|----------------|
| **Trial/Free** | Curious explorer | Clean, inviting, non-intimidating | "I can do this" |
| **Starter** | Junior analyst | Guided, tooltipped, progressive | "I'm learning" |
| **Pro** | Seasoned analyst | Dense but organized, keyboard shortcuts | "I'm efficient" |
| **Enterprise** | Data engineer | API-first, code snippets, technical | "I'm integrating" |
| **Admin** | Platform owner | Metrics, user management, overview | "I'm in control" |

### WHAT do they see?

**Trial/Free**:
- Hero map (80% of screen)
- Minimal controls (3 buttons max visible)
- Soft gradients, welcoming illustrations
- "Upgrade" as helpful suggestion, not nag

**Starter**:
- Map + sidebar (70/30 split)
- Contextual tooltips on hover
- "Learn more" links embedded
- Progress indicators ("You've run 3/10 simulations today")

**Pro**:
- Dense multi-panel layout
- Collapsible sidebars
- Keyboard shortcut hints
- Quick-access command palette (⌘K)
- Data tables alongside visualizations

**Enterprise**:
- Dashboard with API metrics
- Code snippets in dark terminals
- Webhook logs, real-time
- Team activity feed
- Integration status cards

**Admin**:
- God-view: all users, all orgs
- Revenue/usage charts
- System health monitors
- Audit logs
- User impersonation capability

### WHERE on screen?

```
┌─────────────────────────────────────────────────────┐
│ HEADER: Logo | Nav | Skill Toggle | User           │
├───────────┬─────────────────────────────┬──────────┤
│           │                             │          │
│  PRESETS  │      PRIMARY CANVAS         │ CONTROLS │
│  (left)   │      (map, chart, data)     │ (right)  │
│           │                             │          │
├───────────┴─────────────────────────────┴──────────┤
│ CONTEXT BAR: Legend | Status | Quick Stats         │
├────────────────────────────────────────────────────┤
│ FOOTER: Help Cards | Upgrade CTA | Disclaimers     │
└────────────────────────────────────────────────────┘
```

**Mobile Adaptation**:
```
┌──────────────────────┐
│ HEADER (compact)     │
├──────────────────────┤
│                      │
│   PRIMARY CANVAS     │
│   (full width)       │
│                      │
├──────────────────────┤
│ PRESET PILLS (scroll)│
├──────────────────────┤
│ [Controls FAB] ───┐  │
│                   ▼  │
│  BOTTOM SHEET        │
│  (swipe up)          │
└──────────────────────┘
```

### WHEN do elements appear?

**Progressive Disclosure Timeline**:
1. **0-2 seconds**: Core visualization loads
2. **2-5 seconds**: Secondary controls fade in
3. **On hover**: Tooltips, legends, details
4. **On click**: Modals, drill-downs, expanded views
5. **On scroll**: Lazy-loaded content, infinite scroll data
6. **On demand**: Advanced filters, export options, settings

### WHY this visual hierarchy?

| Element | Why First | Why Hidden |
|---------|-----------|------------|
| Map | Core value prop, immediate understanding | - |
| Presets | Quick context switching | Can be collapsed |
| Layer toggle | Changes visualization meaning | |
| Simulate button | Primary action | |
| Save | Secondary action | Mobile: in sheet |
| Upgrade CTA | Conversion | Not intrusive |
| Detailed stats | Expert need | Progressive reveal |
| API docs | Enterprise only | Separate page |

### HOW do we present it?

---

## 2. TIER-SPECIFIC VISUAL LANGUAGE

### Trial/Free Tier
```css
/* Welcoming, non-threatening */
--bg-primary: #0f172a;        /* Deep but not black */
--accent: #60a5fa;            /* Friendly blue */
--text-primary: #f1f5f9;
--card-bg: rgba(30,41,59,0.8); /* Soft cards */
--border-radius: 16px;         /* Rounded, friendly */
--shadow: 0 4px 24px rgba(0,0,0,0.3); /* Soft depth */
```

**Visual Characteristics**:
- Generous whitespace
- Larger touch targets (48px min)
- Illustrations over raw data
- Friendly iconography
- Guided onboarding overlays
- Success celebrations (confetti on first sim)

### Starter Tier
```css
/* Educational, supportive */
--accent: #3b82f6;            /* Professional blue */
--success: #22c55e;           /* Green for progress */
--border-radius: 12px;        /* Slightly more refined */
```

**Visual Characteristics**:
- Info icons with tooltip hints
- Progress bars and usage meters
- "Pro tip" callout boxes
- Contextual help links
- Skill level indicator
- Achievement badges

### Pro Tier
```css
/* Efficient, dense, powerful */
--accent: #6366f1;            /* Indigo - deeper */
--bg-primary: #020617;        /* Darker for contrast */
--border-radius: 8px;         /* Sharper, professional */
--transition: 100ms;          /* Snappier interactions */
```

**Visual Characteristics**:
- Compact spacing (less padding)
- Data tables with sort/filter
- Keyboard shortcut badges
- Split-pane layouts
- Real-time update indicators
- Batch action toolbars
- Command palette (⌘K)

### Enterprise Tier
```css
/* Technical, API-focused */
--accent: #8b5cf6;            /* Purple - premium */
--terminal-bg: #0d1117;       /* GitHub dark */
--code-font: 'JetBrains Mono';
--border-radius: 6px;         /* Angular, technical */
```

**Visual Characteristics**:
- Code blocks with syntax highlighting
- Terminal-style log viewers
- API endpoint cards with copy buttons
- Webhook status indicators (green dot = connected)
- Usage graphs (API calls over time)
- Team avatar stacks
- Integration logos grid

### Admin Tier
```css
/* Authoritative, overview-focused */
--accent: #f59e0b;            /* Amber - admin gold */
--danger: #ef4444;            /* Red for destructive */
--border-radius: 8px;
```

**Visual Characteristics**:
- Full-width data tables
- Org/user hierarchy trees
- Revenue charts (line + bar)
- System status dashboard
- Impersonation banner (bright warning)
- Audit log timeline
- Bulk action controls

---

## 3. DESKTOP VS MOBILE VISUAL STRATEGY

### Desktop (1024px+)
- **Layout**: Multi-column, sidebars visible
- **Density**: Medium-high, professional
- **Interactions**: Hover states, keyboard shortcuts, right-click menus
- **Data**: Full tables, complex charts, real-time streams
- **Navigation**: Persistent sidebar or top nav

### Tablet (768px - 1023px)
- **Layout**: Collapsible sidebar, responsive grid
- **Density**: Medium, touch-friendly
- **Interactions**: Tap + swipe gestures
- **Data**: Scrollable tables, simplified charts
- **Navigation**: Hamburger menu or bottom tabs

### Mobile (< 768px)
- **Layout**: Single column, stacked
- **Density**: Low, generous spacing
- **Interactions**: Thumb-zone optimized, bottom sheets
- **Data**: Card-based, progressive loading
- **Navigation**: Bottom nav or floating action button

### Critical Mobile Adaptations

| Desktop Element | Mobile Adaptation |
|-----------------|-------------------|
| Sidebar controls | Bottom sheet (swipe up) |
| Data table | Horizontal scroll or card list |
| Multi-select | Long-press + selection mode |
| Hover tooltips | Tap-and-hold or info buttons |
| Right-click menu | Long-press menu |
| Keyboard shortcuts | Gesture shortcuts |
| Dense stats grid | Swipeable card carousel |
| Complex chart | Simplified or tap-to-expand |

---

## 4. COLOR PSYCHOLOGY BY CONTEXT

### Data States
| State | Color | Hex | Usage |
|-------|-------|-----|-------|
| Stable | Blue | #3b82f6 | Low risk, secure, trusted |
| Caution | Amber | #f59e0b | Elevated risk, attention |
| Warning | Orange | #f97316 | High risk, act soon |
| Critical | Red | #ef4444 | Immediate action required |
| Success | Green | #22c55e | Positive outcome, confirmed |
| Neutral | Slate | #64748b | Inactive, disabled, muted |
| Premium | Purple | #8b5cf6 | Pro features, upgrade prompts |

### Emotional Mapping
| Emotion | How to Evoke | Where Used |
|---------|--------------|------------|
| Trust | Blue tones, clean layouts | Landing, pricing |
| Urgency | Orange/red accents, motion | Alerts, CTAs |
| Calm | Muted tones, whitespace | Reports, analysis |
| Power | Dark bg, purple accents | Enterprise features |
| Clarity | High contrast, minimal | Data visualizations |
| Excitement | Gradients, subtle animation | Onboarding, achievements |

---

## 5. TYPOGRAPHY HIERARCHY

```css
/* Display - Landing pages, hero sections */
.display-1 { font-size: 3.5rem; font-weight: 800; letter-spacing: -0.02em; }
.display-2 { font-size: 2.5rem; font-weight: 700; }

/* Headings - Dashboard sections */
.h1 { font-size: 1.875rem; font-weight: 700; }
.h2 { font-size: 1.5rem; font-weight: 600; }
.h3 { font-size: 1.25rem; font-weight: 600; }
.h4 { font-size: 1rem; font-weight: 600; }

/* Body */
.body-lg { font-size: 1.125rem; line-height: 1.75; }
.body { font-size: 1rem; line-height: 1.6; }
.body-sm { font-size: 0.875rem; line-height: 1.5; }

/* Data */
.stat-value { font-size: 2rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.stat-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }

/* Code */
.code { font-family: 'JetBrains Mono', monospace; font-size: 0.875rem; }
```

---

## 6. MOTION & ANIMATION PRINCIPLES

### Performance Budget
- **Page transitions**: 200-300ms
- **Micro-interactions**: 100-150ms
- **Loading states**: Skeleton shimmer
- **Data updates**: Fade/slide, no jarring jumps

### Animation Patterns
```css
/* Entrance */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Loading shimmer */
@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

/* Pulse for live data */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* Glow for CTAs */
@keyframes glow {
  0%, 100% { box-shadow: 0 0 20px rgba(59,130,246,0.3); }
  50% { box-shadow: 0 0 30px rgba(59,130,246,0.5); }
}
```

### When to Animate
| Action | Animation | Duration |
|--------|-----------|----------|
| Page load | Staggered fade-in | 300ms |
| Modal open | Scale + fade | 200ms |
| Tooltip show | Fade | 150ms |
| Button hover | Background shift | 100ms |
| Tab switch | Slide | 200ms |
| Data refresh | Pulse highlight | 500ms |
| Error shake | Horizontal shake | 300ms |
| Success | Check mark draw | 400ms |

---

## 7. ICONOGRAPHY SYSTEM

### Icon Style
- **Stroke weight**: 1.5px (consistent)
- **Corner radius**: 2px
- **Grid**: 24x24 base
- **Style**: Outlined (not filled) for interface
- **Filled variants**: For selected/active states

### Icon Categories
```
Navigation:    home, compass, map, layers, settings
Actions:       play, pause, save, export, share, copy
Data:          chart-bar, chart-line, table, filter, sort
Status:        check, x, alert, info, loader
Users:         user, users, building, key
Communication: mail, bell, message, phone
Files:         file, folder, download, upload
Misc:          globe, clock, calendar, search, menu
```

### Icon + Text Pairing
- Icon left of text for actions
- Icon above text for navigation (mobile)
- Icon right for external links (↗)
- Icon only with tooltip for compact UI

---

## 8. COMPONENT VISUAL PATTERNS

### Cards
```
┌─────────────────────────────────┐
│ ○ Icon   Title            Badge │  ← Header: icon, title, status
├─────────────────────────────────┤
│                                 │
│  Main content area              │  ← Body: flexible content
│  (chart, text, data)            │
│                                 │
├─────────────────────────────────┤
│ Meta info          [Action] →   │  ← Footer: metadata, CTA
└─────────────────────────────────┘
```

### Stat Blocks
```
┌──────────────────┐
│     1,247        │  ← Large value (tabular nums)
│   API Calls      │  ← Label (muted, uppercase)
│   ↑ 12% vs last  │  ← Trend (green up, red down)
└──────────────────┘
```

### Data Tables
```
┌──────┬──────────┬─────────┬────────┐
│ ☑    │ Name ↕   │ Status  │ Actions│  ← Sortable headers
├──────┼──────────┼─────────┼────────┤
│ ☐    │ Row 1    │ ● Live  │ ⋮      │  ← Selectable rows
│ ☐    │ Row 2    │ ○ Draft │ ⋮      │
│ ☐    │ Row 3    │ ● Live  │ ⋮      │
└──────┴──────────┴─────────┴────────┘
     Showing 1-3 of 127  [←][→]        ← Pagination
```

---

## 9. ART ASSET REQUIREMENTS

### Logo Variations Needed
1. **Symbol only** (32x32 to 512x512)
2. **Wordmark** (text "LatticeForge")
3. **Lockup** (symbol + wordmark horizontal)
4. **Lockup vertical** (symbol above wordmark)
5. **Favicon** (simplified for 16x16)
6. **Social avatar** (circle-safe)
7. **Dark mode version** (for light backgrounds)
8. **Animated version** (for loading states)

### Hero/Marketing Assets
| Asset | Size | Purpose |
|-------|------|---------|
| og-image.png | 1200x630 | Social shares |
| twitter-card.png | 1200x600 | Twitter |
| hero-desktop.png | 1920x1080 | Landing page |
| hero-mobile.png | 750x1334 | Mobile landing |
| feature-1.png | 800x600 | Feature sections |
| pricing-bg.png | 1920x800 | Pricing page bg |

### Dashboard Assets
| Asset | Purpose |
|-------|---------|
| map-placeholder.png | Loading state for map |
| empty-state.svg | No data illustrations |
| onboarding-1-5.svg | Walkthrough graphics |
| upgrade-prompt.svg | Upgrade CTA illustration |

### Icon Assets
| Set | Count | Sizes |
|-----|-------|-------|
| Navigation icons | 12 | 20, 24, 32 |
| Action icons | 20 | 16, 20, 24 |
| Status icons | 8 | 12, 16, 20 |
| Feature icons | 10 | 48, 64 |

---

## 10. MIDJOURNEY PROMPTS FOR MISSING ASSETS

### Logo Mark
```
Abstract geometric logo mark for "LatticeForge", combining crystalline lattice structure with forge/metalworking energy. Interconnected nodes forming a stable three-dimensional shape with subtle orange glow emanating from center suggesting transformation of raw data into intelligence. Scientific precision meets industrial strength. Clean vector style suitable for favicon scaling. Centered on pure black background. Premium tech brand aesthetic, Palantir meets Bloomberg visual language --ar 1:1 --v 6.1 --style raw
```

### Hero Background
```
Abstract dark visualization of global data networks, flowing particle streams connecting node clusters across a curved horizon suggesting earth's surface without being literal. Deep navy to slate gradient, subtle blue and purple accent glows. Ethereal, intelligent, vast scale. Cinematic depth of field. Clean areas for text overlay on left third. Premium SaaS hero aesthetic --ar 21:9 --v 6.1
```

### Empty State Illustration
```
Minimal line illustration of constellation forming into shape, gentle blue glow, suggesting data coming together. Simple, friendly, not intimidating. White lines on transparent background. Subtle animated potential. Zen, calm, patient aesthetic --ar 1:1 --v 6.1 --style raw
```

### Upgrade Prompt Graphic
```
Abstract upward-flowing energy particles, transformation visual, basic form becoming refined crystal. Aspirational but not pushy. Blue to purple gradient on dark background. Suggests unlocking potential. Suitable for small card graphic --ar 4:3 --v 6.1
```

---

## 11. VISUAL QA CHECKLIST

### Before Launch
- [ ] All interactive elements have hover/focus states
- [ ] Touch targets are 48px minimum on mobile
- [ ] Color contrast meets WCAG AA (4.5:1 for text)
- [ ] Loading states exist for all async content
- [ ] Empty states are designed, not blank
- [ ] Error states are helpful, not just red
- [ ] Responsive at 320px, 768px, 1024px, 1440px, 1920px
- [ ] Dark mode is tested (not just inverted)
- [ ] Icons are crisp at all sizes
- [ ] Typography scales properly
- [ ] Animations respect prefers-reduced-motion
- [ ] Favicon visible on both light and dark browser themes

---

*Document Version 1.0 | LatticeForge Design System*
