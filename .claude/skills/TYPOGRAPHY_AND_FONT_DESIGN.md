# TYPOGRAPHY AND FONT DESIGN MASTERY

**From foundational principles to badass custom font creation**

---

## PART 1: Typography Fundamentals

### The Anatomy of Type

**Every letterform has specific parts:**

```
     apex
      /\
     /  \    terminal
    /    \      ↓
   /  /\  \    ___
  /  /  \  \  |   |
 /__/____\__\ |___|
     |          ↑
  stem/stroke   serif

ascender: parts above x-height (b, d, f, h, k, l, t)
descender: parts below baseline (g, j, p, q, y)
x-height: height of lowercase x (defines readability)
cap height: height of capital letters
baseline: invisible line letters sit on
```

**Key measurements:**
- **Point size**: Height from ascender to descender (12pt, 16pt, etc.)
- **Leading** (line-height): Space between baselines
- **Kerning**: Space between specific letter pairs
- **Tracking**: Overall letter spacing across text

---

## PART 2: Font Classifications

### Serif Fonts

**Old Style** (Humanist):
- Examples: Garamond, Palatino, Caslon
- Characteristics: Low contrast, angled stress, bracketed serifs
- Use: Body text in books, elegant traditional designs
- Emotion: Classic, trustworthy, literary

**Transitional:**
- Examples: Times New Roman, Baskerville, Georgia
- Characteristics: Higher contrast, vertical stress
- Use: Newspapers, formal documents
- Emotion: Professional, authoritative

**Modern** (Didone):
- Examples: Bodoni, Didot
- Characteristics: Extreme contrast, thin serifs, vertical stress
- Use: Fashion, luxury brands, headlines
- Emotion: Sophisticated, dramatic, high-end

**Slab Serif:**
- Examples: Rockwell, Courier, Clarendon
- Characteristics: Thick, blocky serifs, low contrast
- Use: Displays, typewriter aesthetic, Western themes
- Emotion: Sturdy, bold, mechanical

### Sans-Serif Fonts

**Grotesque:**
- Examples: Helvetica, Akzidenz-Grotesk
- Characteristics: Minimal contrast, closed apertures
- Use: Signage, corporate branding
- Emotion: Neutral, modern, professional

**Geometric:**
- Examples: Futura, Avant Garde, Century Gothic
- Characteristics: Perfect circles, strict geometry
- Use: Tech brands, minimalist design
- Emotion: Clean, futuristic, efficient

**Humanist:**
- Examples: Gill Sans, Frutiger, Verdana
- Characteristics: Calligraphic influence, open apertures
- Use: Body text (digital), wayfinding
- Emotion: Friendly, readable, approachable

### Display & Decorative

**Script:**
- Formal (Copperplate), Casual (Brush Script)
- Use: Invitations, logos, special occasions
- Emotion: Elegant or playful depending on style

**Blackletter:**
- Examples: Fraktur, Textura
- Use: Gothic themes, metal bands, medieval aesthetics
- Emotion: Traditional, ornate, dramatic

**Stencil:**
- Examples: Stencil, Impact
- Use: Military, industrial, bold statements
- Emotion: Utilitarian, strong, direct

---

## PART 3: Typography Hierarchy

### Establishing Visual Order

**Six levels of hierarchy:**

1. **Main Headline** (H1)
   - Size: 32-72pt
   - Weight: Bold or Black
   - Purpose: Grab attention, primary message

2. **Subheading** (H2)
   - Size: 24-36pt
   - Weight: Semi-bold or Medium
   - Purpose: Section breaks, secondary message

3. **Section Heading** (H3)
   - Size: 18-24pt
   - Weight: Bold or Semi-bold
   - Purpose: Organize content

4. **Body Text** (P)
   - Size: 14-18pt (print), 16-20pt (web)
   - Weight: Regular
   - Purpose: Main content, readability

5. **Caption/Metadata**
   - Size: 10-12pt
   - Weight: Regular or Light
   - Purpose: Supporting info, image captions

6. **Fine Print**
   - Size: 8-10pt
   - Weight: Light
   - Purpose: Legal, disclaimers

**Hierarchy through contrast:**
- Size (most obvious)
- Weight (bold vs. light)
- Color (dark vs. light, color vs. gray)
- Style (uppercase vs. sentence case)
- Font (serif vs. sans-serif)
- Spacing (tight vs. loose)

---

## PART 4: Font Pairing

### Rules for Combining Fonts

**The Golden Rules:**

1. **Contrast, don't conflict**
   - Good: Serif headline + Sans body
   - Bad: Two similar sans-serifs

2. **Limit to 2-3 fonts max**
   - Headline font
   - Body font
   - Optional: Accent font

3. **Match x-heights**
   - Fonts with similar x-heights look harmonious

4. **Use font families**
   - Many weights (Light, Regular, Bold, Black)
   - Saves pairing headaches

**Classic Pairings:**

- **Playfair Display (Serif H) + Source Sans Pro (Sans Body)**
  - Use: Editorial, elegant websites
  - Why: High contrast, complementary styles

- **Montserrat (Sans H) + Merriweather (Serif Body)**
  - Use: Blogs, modern publications
  - Why: Geometric headline, readable body

- **Bebas Neue (Display H) + Open Sans (Sans Body)**
  - Use: Posters, bold designs
  - Why: Impact vs. neutrality

- **Lora (Serif H) + Lato (Sans Body)**
  - Use: Professional documents
  - Why: Warm, approachable, versatile

**For Your Tarot App:**

Recommended pairing:
- **Headlines**: Cinzel (elegant serif, classical)
- **Body text**: Raleway (clean sans, readable)
- **Accents**: Philosopher (mystical, decorative)

---

## PART 5: Readability & Legibility

### Body Text Best Practices

**Line length (measure):**
- Ideal: 45-75 characters per line
- Too short: Choppy reading
- Too long: Eye fatigue, losing place

**Line height (leading):**
- Body text: 1.4-1.6× font size
- Headlines: 1.0-1.2× font size
- Tight leading: Headlines, impact
- Loose leading: Poetry, emphasis

**Letter spacing (tracking):**
- Body text: 0 (default)
- ALL CAPS: +50-100 (needs more space)
- Small caps: +25-50

**Alignment:**
- **Left-aligned**: Best for readability (Western languages)
- **Centered**: Headlines, short text, formal
- **Justified**: Books, newspapers (needs hyphenation)
- **Right-aligned**: Rare, special effects

**Color contrast:**
- WCAG AA: 4.5:1 minimum (body text)
- WCAG AAA: 7:1 (preferred)
- Tool: WebAIM Contrast Checker

---

## PART 6: Digital Typography (Web & Mobile)

### System Font Stacks

**iOS:**
```css
font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", ...;
```

**Android:**
```css
font-family: "Roboto", "Noto Sans", ...;
```

**Cross-platform:**
```css
font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, ...;
```

### Web Fonts

**Google Fonts** (free):
```html
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&display=swap" rel="stylesheet">
```

**Adobe Fonts** (paid):
- Sync with Creative Cloud
- High-quality, professional fonts

**Self-hosted:**
```css
@font-face {
  font-family: 'CustomFont';
  src: url('/fonts/CustomFont.woff2') format('woff2');
  font-weight: 400;
  font-display: swap; /* Prevent FOIT */
}
```

### React Native Typography

```javascript
import { Platform, StyleSheet } from 'react-native';

const styles = StyleSheet.create({
  headline: {
    fontFamily: Platform.select({
      ios: 'Cochin',
      android: 'serif',
    }),
    fontSize: 32,
    fontWeight: 'bold',
    lineHeight: 40,
    letterSpacing: -0.5,
  },
  body: {
    fontFamily: Platform.select({
      ios: 'Helvetica Neue',
      android: 'Roboto',
    }),
    fontSize: 16,
    lineHeight: 24,
    letterSpacing: 0,
  },
});
```

**Custom fonts in React Native:**
```bash
# 1. Add fonts to assets/fonts/
# 2. Link fonts:
expo install expo-font
```

```javascript
import * as Font from 'expo-font';
import { useFonts } from 'expo-font';

export default function App() {
  const [fontsLoaded] = useFonts({
    'Cinzel-Regular': require('./assets/fonts/Cinzel-Regular.ttf'),
    'Cinzel-Bold': require('./assets/fonts/Cinzel-Bold.ttf'),
  });

  if (!fontsLoaded) return <LoadingScreen />;

  return <Text style={{ fontFamily: 'Cinzel-Bold' }}>Mystical Text</Text>;
}
```

---

## PART 7: Print Typography (Books & Magazines)

### Book Typography

**Page layout:**
```
Margins (single page):
  Inner (gutter): 0.75-1 inch
  Outer: 0.5-0.75 inch
  Top: 0.75 inch
  Bottom: 1 inch

Body text:
  Font: Garamond, Caslon, Minion Pro
  Size: 10-12pt
  Leading: 14-16pt
  Line length: 60-70 characters
```

**Chapter openings:**
- Drop caps (3-5 lines tall)
- Ornamental dividers
- White space (1/3 page minimum)

**Running headers:**
- Book title (left page)
- Chapter title (right page)
- Size: 8-10pt, small caps

**Page numbers (folios):**
- Bottom center or outer margin
- Omit on chapter openings

### Magazine Layout

**Grid system:**
- 12-column grid (flexibility)
- Baseline grid (18pt typical)
- Gutters: 1-2 picas

**Headlines:**
- Size: 36-72pt
- Weight: Bold or Black
- Short (5-10 words max)

**Deck/Standfirst** (subhead below headline):
- Size: 14-18pt
- Weight: Semi-bold or Regular
- 1-2 sentences summarizing article

**Pull quotes:**
- Size: 24-36pt
- Borders or background
- 2-3 lines max

**Captions:**
- Size: 8-10pt
- Italic or different font
- Left-aligned under image

---

## PART 8: Custom Font Creation

### Tools

**Free:**
- **FontForge**: Open-source, full-featured
- **BirdFont**: Simpler, good for beginners

**Paid:**
- **Glyphs**: Mac, industry standard
- **FontLab**: Windows/Mac, professional

### Process

**1. Sketching**
- Draw letterforms by hand
- Establish character (5-10 key letters)
- Test at various sizes

**2. Digitization**
- Scan sketches at 600 DPI
- Trace in vector (Illustrator/Inkscape)
- Import into font editor

**3. Letter design**
- Start with: n, o, H, O
- These define proportions for rest
- **n**: establishes x-height, stem width, curves
- **o**: defines rounds, stress angle
- **H**: sets cap height, crossbar
- **O**: cap rounds, optical corrections

**4. Spacing**
- **Sidebearings**: Space on each side of glyph
- **Kerning pairs**: Adjust specific combos (AV, To, We)
- Test string: "Hamburgefonstiv" (contains key shapes)

**5. Complete character set**

Minimum (Basic Latin):
- Uppercase: A-Z
- Lowercase: a-z
- Numbers: 0-9
- Punctuation: . , ! ? ' " - ( )

Extended:
- Accents: á é í ó ú ñ ç
- Ligatures: fi fl ff ffi ffl
- Symbols: @ # $ % & *

**6. Optical adjustments**
- Round letters (O, C) overshoot baseline/cap height slightly
- Pointed letters (A, V) overshoot more
- Makes letters appear same size optically

**7. Hinting** (for screen rendering)
- Instructions for rasterization
- Critical for small sizes (< 20px)
- Auto-hinting often sufficient

**8. Export**
- **OTF** (OpenType Font): Modern, supports features
- **TTF** (TrueType): Wide compatibility
- **WOFF/WOFF2**: Web fonts (compressed)

### Example: Creating a Mystical Display Font

**Concept**: "Arcana" - tarot-inspired display font

**Character:**
- Serif (traditional)
- Decorative terminals (mystical)
- High contrast (dramatic)
- Ligatures (ornate)

**Design process:**

1. **Sketch key letters**
   ```
   Letters to design first:
   - n: defines lowercase proportions
   - o: defines rounds and stress
   - H: defines uppercase
   - O: defines cap rounds
   - a, g, A, M: test contrast and style
   ```

2. **Establish metrics**
   ```
   Cap height: 700 units
   X-height: 500 units (high x-height = readable at small sizes)
   Ascender: 750 units
   Descender: -250 units
   Stem width: 80 units (thick)
   Hairline: 10 units (thin, high contrast)
   ```

3. **Design special features**
   ```
   Ligatures:
   - st (connect s to t)
   - ct (tarot-specific)
   - Th (mystical crossbar)

   Alternates:
   - Swash capitals (ornate A, M, N)
   - Decorated terminals
   ```

4. **Create swashes and ornaments**
   ```
   Ornaments:
   - Decorative dividers
   - Corner flourishes
   - Mystical symbols (☽ ☾ ★ ✦)
   ```

---

## PART 9: Typography in Your Tarot App

### Font Strategy

**Recommended setup:**

```javascript
// Load custom fonts
const [fontsLoaded] = useFonts({
  // Headlines (card names, titles)
  'Cinzel-Bold': require('./assets/fonts/Cinzel-Bold.ttf'),
  'Cinzel-Regular': require('./assets/fonts/Cinzel-Regular.ttf'),

  // Body text (interpretations, dialog)
  'Raleway-Regular': require('./assets/fonts/Raleway-Regular.ttf'),
  'Raleway-SemiBold': require('./assets/fonts/Raleway-SemiBold.ttf'),

  // Accents (ornamental, Roman numerals)
  'Philosopher-Bold': require('./assets/fonts/Philosopher-Bold.ttf'),
});

// Typography system
const typography = {
  cardName: {
    fontFamily: 'Cinzel-Bold',
    fontSize: 28,
    lineHeight: 36,
    letterSpacing: 2,
    color: '#00ffff',
  },
  romanNumeral: {
    fontFamily: 'Philosopher-Bold',
    fontSize: 32,
    letterSpacing: 4,
    color: '#00ffff',
  },
  interpretation: {
    fontFamily: 'Raleway-Regular',
    fontSize: 16,
    lineHeight: 24,
    letterSpacing: 0,
    color: '#ffffff',
  },
  oracleDialog: {
    fontFamily: 'Raleway-SemiBold',
    fontSize: 18,
    lineHeight: 28,
    letterSpacing: 0.5,
    color: '#ffffff',
    fontStyle: 'italic',
  },
};
```

### Text Hierarchy for Card Readings

```
1. Card Name (Cinzel Bold, 28pt, Cyan)
   ↓
2. Meaning/Keywords (Raleway Regular, 14pt, White 90%)
   ↓
3. Interpretation Message (Raleway Regular, 16pt, White)
   ↓
4. Action Button (Raleway Semi-Bold, 18pt, White)
```

---

## PART 10: Advanced Typography Techniques

### OpenType Features

**Ligatures:**
```css
/* Enable ligatures */
font-feature-settings: "liga" 1;

/* Discretionary ligatures (more decorative) */
font-feature-settings: "dlig" 1;
```

**Small caps:**
```css
font-feature-settings: "smcp" 1;
```

**Oldstyle figures** (text numbers):
```css
font-feature-settings: "onum" 1;
```

**Tabular figures** (monospaced numbers):
```css
font-feature-settings: "tnum" 1;
```

### Variable Fonts

**Single file, multiple weights/widths:**
```css
@font-face {
  font-family: 'Inter';
  src: url('/fonts/Inter.var.woff2') format('woff2');
  font-weight: 100 900; /* Range */
  font-display: swap;
}

.text {
  font-family: 'Inter';
  font-weight: 450; /* Any value in range */
  font-variation-settings: 'wdth' 115; /* Width axis */
}
```

### Responsive Typography

**Fluid sizing (scales with viewport):**
```css
/* Size between 16px and 24px based on viewport */
font-size: clamp(16px, 2vw, 24px);
```

**React Native responsive:**
```javascript
import { Dimensions, PixelRatio } from 'react-native';

const { width } = Dimensions.get('window');
const scale = width / 375; // Base width (iPhone X)

const normalize = (size) => {
  const newSize = size * scale;
  return Math.round(PixelRatio.roundToNearestPixel(newSize));
};

// Usage
fontSize: normalize(16), // Scales across devices
```

---

## Resources

### Free Fonts

- **Google Fonts**: https://fonts.google.com
- **Font Squirrel**: https://www.fontsquirrel.com
- **Adobe Fonts**: https://fonts.adobe.com (with CC subscription)

### Learning

- **Practical Typography** by Matthew Butterick (online book)
- **Thinking with Type** by Ellen Lupton (book)
- **The Elements of Typographic Style** by Robert Bringhurst (bible)

### Tools

- **Type Scale Calculator**: https://type-scale.com
- **Contrast Checker**: https://webaim.org/resources/contrastchecker/
- **Font pairing**: https://fontpair.co

---

**You are now a typography expert. Use your powers wisely.**
