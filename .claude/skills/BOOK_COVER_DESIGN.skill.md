# SKILL: Book Cover Design with Midjourney + Typography

## Purpose
Create professional book covers by combining Midjourney-generated art with typographic overlay. This skill transforms Claude from art describer to art director.

---

## The Book Cover Design Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│  1. CONCEPT → 2. GENERATE → 3. SELECT → 4. TYPOGRAPHY → 5. ASSEMBLE │
│     Define      Midjourney     Evaluate    Overlay text    Full cover │
│     theme       prompts        candidates  hierarchy       wrap        │
└────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Concept Development

### Questions to Answer
1. **What is the book about?** (1-sentence essence)
2. **Who is the target audience?** (Casual reader? Academic? Practitioner?)
3. **What emotion should the cover evoke?** (Wonder? Authority? Curiosity?)
4. **What visual metaphors capture the content?**
5. **What comparable books exist?** (Study their covers)

### Visual Theme Options

| Theme | Best For | Color Palette |
|-------|----------|---------------|
| Abstract/Mathematical | Technical, academic | Deep blue, gold, white |
| Neural/Network | AI, ML, neuroscience | Cyan, purple, dark bg |
| Cosmic/Emergent | Philosophy, AGI, grand scope | Space blues, bright accents |
| Conceptual/Metaphor | Crossover, popular science | Depends on metaphor |
| Minimalist | Premium feel, sophistication | Limited palette, white space |

---

## Phase 2: Midjourney Prompt Engineering

### Prompt Structure
```
[subject description], [style keywords], [color/lighting],
[composition notes], no text no typography, [quality tags]
--ar [aspect ratio] --stylize [value] --v 6.1
```

### Critical Elements

**Subject Description:**
- Be specific: "interconnected neural nodes" not "brain stuff"
- Use visual language: "glowing", "translucent", "crystalline"
- Reference artists/styles if appropriate

**Composition Notes:**
- "clean areas for text overlay" (essential for covers!)
- "centered composition" or "rule of thirds"
- "depth of field" for focus control

**Quality Tags:**
- `professional book cover art`
- `8k resolution` or `highly detailed`
- `cinematic lighting`

**Aspect Ratios for Books:**
- eBook: `--ar 2:3` (standard)
- Paperback: `--ar 2:3` or calculate from trim
- Square (audiobook): `--ar 1:1`

**Stylize Values:**
- 100-250: More literal interpretation
- 400-600: Balanced creativity
- 750-1000: More artistic interpretation

### Anti-Patterns to Avoid
- No `--no text` (use natural language: "no text no typography")
- Avoid ambiguous subjects ("abstract thing")
- Don't over-prompt (keep under 60 words)
- Never include author names or titles in prompt

---

## Phase 3: Selection Criteria

### Technical Requirements
| Criterion | Requirement |
|-----------|-------------|
| Resolution | 1600×2400px minimum (eBook), 4000×6000px (print) |
| Composition | Clear space for title (top 1/3) and author (bottom 1/6) |
| Focus | Main subject clearly defined |
| Color | Consistent palette, not muddy |
| Artifacts | No AI artifacts, hands, text |

### Emotional Fit
- Does it match the book's tone?
- Would you pick it up based on the cover alone?
- Is it distinctive in the category?
- Does it photograph well (for thumbnails)?

### Thumbnail Test
Shrink to 100×150px. Is the cover still:
- Readable? (title must work small)
- Distinctive? (recognizable shape/color)
- Appealing? (draws the eye)

---

## Phase 4: Typography Overlay

### The Hierarchy
```
1. TITLE (largest, most prominent)
2. Subtitle (smaller, supports title)
3. Author name (establishes credibility)
4. Other elements (edition, endorsement, series)
```

### Font Pairing Rules

**For Technical/Academic:**
- Title: Bold sans-serif (Bebas Neue, Oswald, Montserrat)
- Subtitle: Light weight of same family
- Author: Same family, medium weight

**For Literary/Philosophical:**
- Title: Elegant serif (Playfair Display, Cormorant)
- Subtitle: Complementary sans-serif
- Author: Matching serif

**Never:**
- More than 2 font families
- Decorative fonts for body text
- Fonts that clash with the art

### Placement Zones
```
┌─────────────────────────────┐
│       TITLE ZONE            │  ← Top 25-35%
│       (most visible)        │
├─────────────────────────────┤
│                             │
│       ART FOCUS             │  ← Center 40-50%
│       (hero image)          │
│                             │
├─────────────────────────────┤
│    Subtitle / Tagline       │  ← Flexible
├─────────────────────────────┤
│       AUTHOR NAME           │  ← Bottom 10-15%
└─────────────────────────────┘
```

### Text Legibility Techniques
1. **Contrast band**: Darken/lighten area behind text
2. **Drop shadow**: Subtle shadow for depth
3. **Outline/glow**: For text over complex backgrounds
4. **Solid banner**: Clean area carved out for text
5. **Overlay gradient**: Fade to solid color at top/bottom

---

## Phase 5: Full Cover Assembly

### Paperback Wrap-Around

```
┌──────────┬─────────┬──────────┐
│          │         │          │
│  BACK    │ SPINE   │  FRONT   │
│  COVER   │ (text   │  COVER   │
│          │ vertical)│         │
│          │         │          │
│ [Blurb]  │         │ [Title]  │
│ [Author] │         │ [Art]    │
│ [Barcode]│         │ [Author] │
│          │         │          │
└──────────┴─────────┴──────────┘
```

### Spine Calculations
```
Spine width = Page count × Paper thickness
- White paper: 0.002252" per page
- Cream paper: 0.0025" per page

Example: 550 pages × 0.0025" = 1.375" spine
```

### Safe Zones
- **Bleed**: 0.125" outside trim (will be cut)
- **Live area**: 0.25" inside trim (keep text here)
- **Spine hinge**: 0.0625" each side of spine

---

## KDP-Specific Requirements

### eBook Cover
| Spec | Value |
|------|-------|
| Format | JPEG or TIFF |
| Color | RGB |
| Minimum size | 1000×1600px |
| Recommended | 1600×2400px |
| Max file size | 50MB |

### Paperback Cover
| Spec | Value |
|------|-------|
| Format | PDF |
| Color | CMYK |
| Resolution | 300 DPI |
| Bleed | 0.125" all edges |
| Use calculator | kdp.amazon.com/cover-calculator |

---

## Asset Library: Current Book Cover Candidates

### "The Mathematics of Intelligence"

**Primary Recommendation: Infinity Symbol**
- File: `aphoticshaman_Infinity_symbol_made_of_flowing_light_particles_786821d7-4605-4582-869c-5e29fe675337_1.png`
- Strengths: Mathematical elegance, cyan-purple gradient, centered composition, clean background
- Use: Direct as hero element

**Alternative: Cyan Neural Network**
- File: `aphoticshaman_Abstract_visualization_of_interconnected_nodes__dcc2e015-bd29-418a-956d-2f3f2270f31d_1.png`
- Strengths: Literal AI/neural imagery, professional
- Use: Full bleed background

**Alternative: Mercury Earth**
- File: `aphoticshaman_Mercury_droplet_shaped_like_earth_hovering_refl_48219cf9-6e36-409b-8ab4-ef17303b2388_0.png`
- Strengths: Unique, memorable, metaphorical
- Use: Conversation-starter cover

---

## Workflow Checklist

### Before Generating
- [ ] Define concept and emotion
- [ ] Research comparable covers
- [ ] Write 3-5 prompt variations
- [ ] Set aspect ratio for target format

### After Generating
- [ ] Generate 4 variations per prompt
- [ ] Upscale top candidates (Midjourney or Topaz)
- [ ] Test at thumbnail size
- [ ] Verify resolution meets spec

### Typography Phase
- [ ] Choose font pairing
- [ ] Place title in clear zone
- [ ] Test legibility at multiple sizes
- [ ] Add subtle text effects if needed

### Final Assembly
- [ ] Calculate spine width
- [ ] Create full wrap template
- [ ] Add barcode zone (back cover)
- [ ] Export: PDF for print, JPEG for eBook
- [ ] Order print proof before finalizing

---

## Teaching Claude to Be an Art Director

### What I Can Do
1. **Analyze existing art** for cover suitability
2. **Write Midjourney prompts** tailored to book concepts
3. **Evaluate composition** for text placement
4. **Suggest typography** pairings and hierarchy
5. **Specify technical requirements** for KDP/print
6. **Create ASCII mockups** for layout planning

### What I Need From You
1. **Run Midjourney prompts** (I can't generate images directly)
2. **Upload results** for evaluation
3. **Make final aesthetic choices** (taste is personal)
4. **Execute in design tools** (Canva, Figma, Photoshop)
5. **Order and approve proofs**

### Our Collaboration Model
```
You: "I need a cover for [book concept]"
Me: [Prompts + Typography plan + Technical specs]
You: [Generate in Midjourney, upload results]
Me: [Evaluate, select, suggest refinements]
You: [Execute in design tool]
Me: [Review, verify specs, final checks]
You: [Upload to KDP, order proof]
```

---

## Resources

- [KDP Cover Calculator](https://kdp.amazon.com/cover-calculator)
- [KDP Cover Guidelines](https://kdp.amazon.com/en_US/help/topic/G201953020)
- [Midjourney Parameter Reference](https://docs.midjourney.com/docs/parameter-list)
- [Canva Book Cover Templates](https://www.canva.com/book-covers/templates/)
- [Typewolf Font Pairing](https://www.typewolf.com/recommendations)

---

---

## Related Skills

### Core Skills Directory (`.claude/skills/`)
- **EPISTEMIC_ABLATION.md** - Kill bad cover concepts before investing in them
- **HUMAN_AI_FUSION.md** - Collaboration model for Claude + Human design workflow
- **RESEARCH_DELIVERABLES.md** - Structure for presenting cover options to stakeholders
- **ARC_PRIZE_2026.skill.md** - Energy-based hypothesis selection (applies to cover choices too)

### Extended Skills Library (`research/skills/` & `research/`)
- **CREATIVITY_AND_INSPIRATION.skill.md** - Divergent thinking for cover concepts
- **VEILPATH_VISUAL_DESIGN_SYSTEM.skill.md** - Deep dive on Midjourney asset analysis, color psychology, sacred geometry
- **COMMUNICATION.skill.md** - How to present cover concepts to publishers
- **EFFICIENCY_AND_ACCURACY.skill.md** - Fast iteration through cover options

### Design System Documentation (`docs/`)
- **VISUAL_DESIGN_SYSTEM.md** - LatticeForge's full visual language (color, typography, layout)
- **ART_ASSETS_ANALYSIS.md** - How to evaluate Midjourney outputs

### Book-Specific Resources (`book/`)
- **COVER_ART_GUIDE.md** - Technical specs for "Mathematics of Intelligence"
- **VISUAL_ASSETS_PLAN.md** - 141 figures/diagrams planned for the book
- **PREFACE.md** - Book essence for cover concept alignment

### Art Asset Catalog (`art_assets/`)
- **CATALOG.md** - 100+ Midjourney assets organized by category
- Key assets for book covers:
  - `aphoticshaman_Infinity_symbol_*` - Mathematical elegance
  - `aphoticshaman_Abstract_visualization_of_interconnected_nodes_*` - Neural networks
  - `aphoticshaman_Geometric_constellation_*` - Scientific constellation
  - `aphoticshaman_Mercury_droplet_shaped_like_earth_*` - Fluid intelligence metaphor

---

*Skill Version 1.0 | "The cover sells the book. The book keeps the reader."*
