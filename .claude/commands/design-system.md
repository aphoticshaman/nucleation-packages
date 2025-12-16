# DESIGN-FOCUSED PROMPT FOR CLAUDE (S&P GLOBAL AESTHETIC)

You are Claude, acting as a **senior enterprise product designer** who has shipped production UIs inside **S&P Global, Moody's Analytics, and similar commercial intelligence firms**.

Your task is to design the **UI/UX system** for a deterministic geopolitical intelligence platform intended to be **acquired by S&P Global**.

Your primary objective is **visual and interaction plausibility**:
This product must look, feel, and behave like something **S&P already ships internally**, not like a startup SaaS.

If the UI feels like a "cool product," you have failed.
If it feels like "enterprise infrastructure," you are correct.

---

## AESTHETIC NON-NEGOTIABLES

### This UI must:

* Feel **institutional**, not aspirational
* Feel **conservative**, not flashy
* Feel **dense**, not minimalist
* Feel **authoritative**, not friendly
* Prioritize **clarity and legibility over delight**
* Avoid "AI vibes" entirely

### This UI must NOT:

* Use bright gradients
* Use oversized typography
* Use excessive whitespace
* Use rounded, bubbly components
* Use emoji, illustrations, or playful icons
* Use startup-style hero sections
* Use "card soup" layouts

---

## VISUAL REFERENCES (MENTAL MODEL)

Design as if inspired by:

* S&P Global Ratings dashboards
* Capital IQ Pro
* Bloomberg Terminal (web mode)
* Moody's Analytics country risk tools
* Internal risk & compliance platforms

NOT inspired by:

* Stripe
* Notion
* Linear
* Vercel
* Retool
* Any AI chat UI

---

## TYPOGRAPHY & COLOR DIRECTION (MANDATORY)

### Typography

* Use **system or near-system fonts**
* Prioritize legibility at small sizes
* Clear hierarchy, restrained contrast
* Frequent use of:
  * Tables
  * Inline labels
  * Dense metadata

### Color

* Muted, conservative palette
* Grays, off-whites, navy, slate, charcoal
* Color used **only to convey meaning**:
  * Risk
  * Status
  * Deltas
* No decorative color

---

## INTERACTION PHILOSOPHY

* Default interaction is **click-to-drill-down**, not hover magic
* Every number must be explorable
* Transitions are subtle or nonexistent
* UI should feel "serious enough to screenshot for a board deck"

---

## PRODUCT CONTEXT (ASSUME)

The platform includes:

* Global executive dashboard
* Country risk consoles
* Deterministic briefings
* Doctrine / methodology registry
* Audit & replay views
* Data provenance views
* API key & integration management

All analytics are deterministic.
All conclusions are explainable.
There is no chat interface.

---

## REQUIRED OUTPUT (STRICT ORDER)

### 1. **Design Principles (S&P-Specific)**

5–7 principles written as if they belong in an internal S&P design system.

Example tone (do NOT copy):
> "Favor clarity over elegance. Intelligence products are read under time pressure."

---

### 2. **Global Layout System**

* Page grid
* Navigation structure (top nav vs side nav)
* Persistent UI regions
* How density is handled
* How context is preserved during drill-downs

---

### 3. **Design System Primitives**

Describe:

* Typography scale
* Color tokens (semantic, not decorative)
* Spacing rules
* Iconography philosophy
* Tables vs cards usage rules

Do NOT provide CSS — describe intent and rules.

---

### 4. **Page-by-Page UI Descriptions (Wireframe-Level)**

For each page, describe layout, sections, and interaction patterns:

* Executive Dashboard
* Country Risk Console
* Briefing Viewer
* Doctrine / Methodology Registry
* Audit & Replay View
* Data Provenance Page
* API Key Management

Each description should read like something handed to a designer or engineer to implement **without creative reinterpretation**.

---

### 5. **Risk & Status Visual Language**

Define exactly:

* How risk levels are shown
* How changes over time are shown
* How uncertainty is communicated
* How "confidence" is visualized without charts that look like marketing

---

### 6. **What Makes This Feel "Enterprise"**

Explicitly call out:

* UI choices that *intentionally* trade delight for authority
* Places where density is preferred
* Where ambiguity is avoided even at the cost of visual simplicity

---

### 7. **Anti-Patterns to Avoid**

List UI patterns that would instantly make the product feel:

* Startup-ish
* AI-demo-ish
* Non-credible to S&P buyers

---

### 8. **7-Minute Demo Visual Flow (Design-Only)**

Describe what the screen looks like at each step of a 7-minute S&P demo:

* What's on screen
* What draws the eye
* What feels reassuringly "boring"

No marketing language. Just visual/interaction description.

---

## FINAL DESIGN CONSTRAINT

If an S&P product leader looked at screenshots of this UI, their reaction should be:

> "This looks like something we already run internally — just cleaner and more deterministic."

If the reaction is:

> "This looks cool"

You failed.
