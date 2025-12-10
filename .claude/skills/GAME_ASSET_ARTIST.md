# GAME ASSET ARTIST GUIDE

**From Midjourney to Professional Game Assets**

## Philosophy

**Goal:** Turn AI-generated art into production-ready game assets

**Principles:**
1. **Consistency >** Individual beauty
2. **Performance > Fidelity** (mobile games need small files)
3. **Usability > Realism** (players need to see what things are)
4. **Iteration > Perfection** (ship and improve)

---

## PART 1: Asset Requirements for React Native Games

### File Size Targets

**Total App Budget:** < 50MB (Google Play initial download limit)

**Individual Assets:**
- **Icons/Buttons:** 5-20 KB each
- **UI Panels:** 20-50 KB each
- **Sprites:** 30-100 KB each
- **Backgrounds:** 100-300 KB each
- **Sprite Sheets:** 200-500 KB each

**How to Achieve:**
1. Generate at correct size (don't upscale)
2. Use PNG compression (level 9)
3. Run through TinyPNG.com
4. Use indexed color for non-photo assets
5. Reuse assets (palette swaps, flips, rotations)

### Resolution Guide

**Mobile Screens:**
- Low-end: 720x1280px (HD)
- Mid-range: 1080x1920px (FHD)
- High-end: 1440x2560px (QHD)

**Asset Sizes:**
```
UI Icons:          128x128px   (displays at 64x64)
Buttons:           256x64px    (displays at 128x32)
Character Sprites: 256x256px   (displays at 128x128)
Buildings:         512x512px   (displays at 256x256)
Tiles:             256x256px   (displays at 128x128)
Backgrounds:       1920x1080px (displays at full screen)
```

**Why 2x?** Retina/HD displays. Always create at 2x, React Native scales down.

---

## PART 2: Sprite Sheet Creation

### What is a Sprite Sheet?

One image containing multiple animation frames or variations.

**Benefits:**
- Faster loading (1 request vs 8)
- Smaller file size (shared compression)
- Easier to manage

### Layouts

**Horizontal Strip** (Recommended for Simple Animations):
```
[Frame1][Frame2][Frame3][Frame4][Frame5][Frame6][Frame7][Frame8]
```
Size: 2048x256px (8 frames × 256px each)

**Grid** (For Complex Animations):
```
[Idle1][Idle2][Idle3][Idle4]
[Walk1][Walk2][Walk3][Walk4]
[Run1 ][Run2 ][Run3 ][Run4 ]
[Jump1][Jump2][Jump3][Jump4]
```
Size: 1024x1024px (4x4 grid × 256px each)

**Atlas** (For UI Elements):
```
Randomly packed (use TexturePacker tool)
```

### Frame Timing

**Animation Speed:**
- Idle: 10-15 FPS (slow, subtle)
- Walk: 15-20 FPS (moderate)
- Run: 20-30 FPS (fast)
- Effects: 30-60 FPS (very fast)

**Frame Count:**
- Idle: 4-6 frames (looping)
- Walk: 6-8 frames (looping)
- Attack: 8-12 frames (one-shot)
- Effects: 8-16 frames (one-shot)

---

## PART 3: Isometric Art Guidelines

### Perspective Rules

**Isometric = 2:1 ratio**
- For every 2 pixels horizontal, 1 pixel vertical
- Angle: 26.565° (arctan(0.5))
- No vanishing point (parallel projection)

**Grid Setup in GIMP:**
1. View → Show Grid
2. Image → Configure Grid
   - Spacing: 64x32px (or 128x64px for larger tiles)
   - Style: Isometric

### Tile Dimensions

**Standard Isometric Tile:**
```
   /\
  /  \
 /    \
/______\

Width: 64px (or 128px)
Height: 32px (or 64px)
```

**Building Heights:**
- 1-story: 64px tall
- 2-story: 128px tall
- 3-story: 192px tall

### Depth Sorting

**Z-Order (Back to Front):**
1. Ground tiles
2. Objects at Y=0
3. Objects at Y=1
4. Objects at Y=2
5. Characters
6. Effects/UI

**React Native:** Use `zIndex` style property

---

## PART 4: Maintaining Consistency Across Assets

### Color Palette

**Extract from your best Midjourney asset:**

1. Open in GIMP
2. Colors → Color Cube Analysis
3. Save top 16 colors
4. Create palette file
5. Apply to all assets

**Palette Example:**
```
Background: #0a0a0f (deep black-blue)
Shadow:     #1a1a2e (dark blue)
Mid-tone:   #2a2a4e (medium blue)
Highlight:  #00ffff (cyan neon)
Accent 1:   #ff00ff (magenta)
Accent 2:   #ffff00 (yellow)
```

### Lighting Direction

**Pick ONE light source direction and stick to it:**
- Top-left (classic game lighting)
- Top-right (less common)
- Direct overhead (isometric standard)

**Check every asset:**
- Shadows point same direction
- Highlights on same side
- Consistent shadow softness

### Line Weight

**Consistent outlines:**
- UI elements: 2-3px outlines
- Characters: 1-2px outlines
- Buildings: 2-4px outlines
- Tiles: 1px or no outline

**GIMP Outline:**
1. Select → By Color → Click asset
2. Select → Grow → 2px
3. Edit → Stroke Selection → 2px

### Style Consistency Checklist

Before calling an asset "done":
- [ ] Colors match palette
- [ ] Lighting from correct direction
- [ ] Line weight consistent
- [ ] Level of detail appropriate for size
- [ ] Transparency properly cut
- [ ] File size under budget
- [ ] Tested in-game

---

## PART 5: Advanced Techniques

### Palette Swaps (Create Variants Fast)

**Problem:** Need red, blue, green versions of same asset

**Solution:**
1. Colors → Map → Color Exchange
2. Select old color (e.g., red #ff0000)
3. Select new color (e.g., blue #0000ff)
4. Exchange
5. Repeat for all colors in palette
6. Export as new variant

### Normal Maps (3D-like Lighting)

**What:** Grayscale image that fakes 3D lighting

**Create in GIMP:**
1. Filters → Light and Shadow → Bump Map
2. Elevation: Your original asset (converted to grayscale)
3. Depth: 5-10
4. Export as `asset_normal.png`

### Parallax Layers

**For backgrounds with depth:**
- Background: Moves at 0.2x speed
- Midground: Moves at 0.5x speed
- Foreground: Moves at 1.0x speed

**Export 3 separate layers**

### Tiling Textures

**Problem:** Visible seams in repeating tiles

**Fix in GIMP:**
1. Filters → Map → Tile Seamless
2. Automatic (GIMP fixes seams)
3. Test: Filters → Map → Tile → Preview

---

## PART 6: Asset Pipeline

### Workflow (Midjourney → Game)

```
1. GENERATE (Midjourney)
   └─> Download 2048x2048 PNG

2. EDIT (GIMP)
   ├─> Crop watermark
   ├─> Remove background → transparency
   ├─> Color correct to match palette
   ├─> Add outline if needed
   └─> Resize to target resolution

3. OPTIMIZE (TinyPNG)
   └─> Upload PNG → download optimized version

4. ORGANIZE (File System)
   └─> Move to assets/art/{category}/

5. INTEGRATE (React Native)
   └─> Import in code, display on screen

6. TEST (Device)
   └─> Check performance, appearance, file size

7. ITERATE (Repeat)
   └─> Adjust based on testing
```

### Folder Structure

```
assets/art/
├── backgrounds/
│   ├── village_day.png
│   ├── village_night.png
│   └── temple_interior.png
├── buildings/
│   ├── tarot_shop_v1.png
│   ├── tarot_shop_v2.png
│   └── oracle_temple.png
├── characters/
│   ├── wizard_idle_sheet.png
│   ├── wizard_walk_sheet.png
│   └── wizard_portrait.png
├── tiles/
│   ├── grass_01.png
│   ├── path_01.png
│   └── water_01.png
├── ui/
│   ├── button_primary.png
│   ├── panel_dialog.png
│   └── icon_menu.png
├── effects/
│   ├── particles_magic.png
│   ├── glow_cyan.png
│   └── sparkle_sheet.png
└── icons/
    ├── skill_shadow.png
    ├── skill_intuition.png
    └── achievement_badge.png
```

---

## PART 7: Common Mistakes

### ❌ DON'T

1. **Use different art styles** - Pick one, stick to it
2. **Forget alpha channels** - Always add before exporting
3. **Make assets too detailed** - Players see them small
4. **Use white backgrounds** - Use transparency
5. **Forget to test on device** - Looks different than desktop
6. **Over-optimize** - Don't sacrifice quality for 2KB
7. **Create unique assets for everything** - Reuse with variations
8. **Mix pixel art with painted** - Choose one style
9. **Ignore performance** - Big assets = slow app
10. **Skip documentation** - Future you will forget what's what

### ✅ DO

1. **Create a style guide** - Document your palette, sizes, rules
2. **Test early and often** - See assets in-game ASAP
3. **Use version control** - Git your assets folder
4. **Keep source files** - Save GIMP .xcf before exporting PNG
5. **Batch process** - Don't manually resize 100 files
6. **Get feedback** - Show to others, iterate
7. **Study reference** - Look at Hades, Diablo 3, Stardew Valley
8. **Iterate relentlessly** - v1 will be bad, that's okay
9. **Document everything** - What each asset is for
10. **Have fun** - This is creative work!

---

## PART 8: React Native Integration

### Importing Assets

```javascript
// In your component
const tarotShop = require('../assets/art/buildings/tarot_shop_v1.png');

<Image
  source={tarotShop}
  style={{ width: 256, height: 256 }}
  resizeMode="contain"
/>
```

### Image Component Props

```javascript
<Image
  source={require('./asset.png')}
  style={{ width: 128, height: 128 }}
  resizeMode="contain"  // Options: cover, contain, stretch, repeat, center
  fadeDuration={300}    // Fade in animation
  blurRadius={5}        // Blur effect
/>
```

### Performance Tips

**Use FastImage for better performance:**
```bash
npm install react-native-fast-image
```

```javascript
import FastImage from 'react-native-fast-image';

<FastImage
  source={{ uri: 'file://path/to/asset.png' }}
  style={{ width: 200, height: 200 }}
  resizeMode={FastImage.resizeMode.contain}
/>
```

---

## PART 9: Asset Naming Conventions

### Best Practices

**Format:** `category_descriptor_variant_state.png`

**Examples:**
```
buildings/tarot_shop_01_day.png
buildings/tarot_shop_01_night.png
buildings/oracle_temple_02_idle.png

characters/wizard_idle_01.png
characters/wizard_walk_sheet.png
characters/wizard_portrait.png

tiles/grass_dark_01.png
tiles/grass_light_01.png
tiles/path_cobblestone_01.png

ui/button_primary_normal.png
ui/button_primary_pressed.png
ui/button_primary_disabled.png

icons/skill_shadow_locked.png
icons/skill_shadow_unlocked.png

effects/particles_magic_cyan.png
effects/glow_magenta_soft.png
```

### Version Control

**Don't delete old versions, rename:**
```
tarot_shop_v1.png   (original)
tarot_shop_v2.png   (revised)
tarot_shop_v3.png   (final)
tarot_shop_FINAL.png (actually final)
tarot_shop_FINAL_v2.png (no really final)
```

---

## PART 10: Learning Resources

### Study These Games (Similar to Yours)

1. **Hades** - Painted isometric, amazing effects
2. **Diablo 3** - Isometric ARPG, great lighting
3. **Stardew Valley** - Pixel isometric, charming style
4. **Bastion** - Painted isometric, beautiful colors
5. **Hearthstone** - Card game UI, polish

### Tools to Learn

**Free:**
- GIMP (image editing)
- Krita (painting, faster than GIMP)
- Inkscape (vector art)
- Aseprite (pixel art - $20 but worth it)
- Blender (3D, if you want to go nuts)

**Paid:**
- TexturePacker (sprite sheet packing)
- Photoshop (industry standard, but expensive)
- Procreate (iPad, amazing for painting)

### Tutorials

**YouTube Channels:**
- "Pixel Pete" - Pixel art fundamentals
- "Marco Bucci" - Color and lighting
- "Level Up!" - Game asset creation
- "Game Endeavor" - Isometric game dev

**Websites:**
- OpenGameArt.org - Free game assets (study these)
- Itch.io - Indie game asset packs
- /r/gamedev - Reddit community

---

## Next Steps

1. **Practice with your Midjourney assets**
   - Pick 5 assets
   - Apply techniques from this guide
   - Compare before/after

2. **Build your pipeline**
   - Document your exact workflow
   - Create templates (GIMP files, scripts)
   - Save time on future assets

3. **Test in-game**
   - See your assets in React Native
   - Iterate based on what you see
   - Performance test on device

4. **Iterate relentlessly**
   - Version 1 will suck, that's okay
   - Each iteration gets better
   - Ship early, improve fast

**Remember:** Professional game artists iterate 5-10 times per asset. Your first version won't be perfect. That's normal. Keep refining.
