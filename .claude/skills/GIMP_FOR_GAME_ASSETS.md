# GIMP FOR GAME ASSET EDITING

**Complete guide to using GIMP for editing Midjourney-generated game assets**

## Installation

**Windows:**
1. Download: https://www.gimp.org/downloads/
2. Install (default options are fine)
3. Launch GIMP

**First Launch Setup:**
- Window → Single-Window Mode (easier for beginners)
- Edit → Preferences → Interface → Theme: Dark (easier on eyes)
- Edit → Preferences → Image Windows → Default Image → Default Image Size: 2048x2048

---

## TASK 1: Crop & Resize Assets

### Remove Midjourney Watermarks

**Problem:** Midjourney adds borders/watermarks to some images

**Fix:**
1. File → Open → Select your asset
2. Tools → Selection Tools → Rectangle Select (R)
3. Draw selection around the clean area (avoid watermark)
4. Image → Crop to Selection
5. File → Export As → PNG

### Resize for React Native

**Problem:** Midjourney generates huge files (2MB+), too big for mobile

**Fix:**
1. Open image
2. Image → Scale Image
   - For UI elements: 512x512px or 1024x1024px
   - For backgrounds: 1920x1080px
   - For tiles: 256x256px
   - For sprite sheets: Keep original, we'll optimize later
3. Quality: Cubic (best quality)
4. Scale
5. File → Export As → PNG
   - Compression level: 9 (maximum)

---

## TASK 2: Fix Midjourney Inconsistencies

### Color Correction (Make Assets Match)

**Problem:** Midjourney generated similar assets with different color tones

**Fix:**
1. Open the "master" asset (the one with perfect colors)
2. Colors → Levels → Export
   - Save as "my_color_profile.txt"
3. Open the mismatched asset
4. Colors → Levels → Import
   - Load "my_color_profile.txt"
5. Adjust if needed
6. Export

### Manual Color Matching (Advanced)

1. Open both images in GIMP (master + bad one)
2. In bad image: Colors → Color Balance
3. Adjust Cyan-Red, Magenta-Green, Yellow-Blue sliders
4. Use master image as reference (Alt+Tab between them)
5. Match as closely as possible

---

## TASK 3: Create Transparent Backgrounds

**Problem:** Midjourney asset has background, but you need transparency

**Method 1: Magic Wand (Simple Backgrounds)**
1. Layer → Transparency → Add Alpha Channel
2. Tools → Selection Tools → Fuzzy Select (U)
3. Click the background color
4. Adjust Threshold if needed (higher = selects more similar colors)
5. Edit → Clear (Delete key)
6. Select → None
7. Export as PNG

**Method 2: Color to Alpha (Solid Color Backgrounds)**
1. Layer → Transparency → Add Alpha Channel
2. Colors → Color to Alpha
3. Click the color picker, select background color
4. Adjust Opacity Threshold if needed
5. OK
6. Export as PNG

**Method 3: Manual Selection (Complex Backgrounds)**
1. Layer → Transparency → Add Alpha Channel
2. Tools → Selection Tools → Free Select (F)
3. Carefully trace around your subject
4. Select → Invert
5. Edit → Clear
6. Select → None
7. Export as PNG

---

## TASK 4: Create Sprite Sheets

**Problem:** You have 8 animation frames as separate images, need them in one sprite sheet

**Steps:**
1. File → New → 2048x256px (8 frames × 256px each)
2. File → Open as Layers → Select all 8 frames
3. Layers panel: Drag each layer to position
   - Frame 1: X=0, Y=0
   - Frame 2: X=256, Y=0
   - Frame 3: X=512, Y=0
   - ... etc
4. Image → Flatten Image
5. Export as PNG

**Pro Tip:** Use Grid to align perfectly
- View → Show Grid
- Image → Configure Grid → Spacing: 256x256px

---

## TASK 5: Optimize File Size

**Problem:** Assets are 2MB each, app will be huge

**Method 1: GIMP Export Compression**
1. File → Export As → PNG
2. Compression level: 9
3. Interlacing: None
4. Save background color: Unchecked
5. Export

**Method 2: TinyPNG (Best, External Tool)**
1. Go to https://tinypng.com
2. Upload your PNG (up to 20 at once)
3. Download optimized version (usually 50-70% smaller)
4. Replace original with optimized

**Method 3: Indexed Color (For Non-Photo Assets)**
1. Image → Mode → Indexed
2. Maximum colors: 256 (reduce if needed)
3. Export as PNG
4. File size will be MUCH smaller

---

## TASK 6: Batch Processing (Process 50+ Assets)

### Install Script-Fu

1. Filters → Script-Fu → Console
2. Paste this script:

```scheme
(define (batch-resize pattern width height)
  (let* ((filelist (cadr (file-glob pattern 1))))
    (while (not (null? filelist))
           (let* ((filename (car filelist))
                  (image (car (gimp-file-load RUN-NONINTERACTIVE filename filename)))
                  (drawable (car (gimp-image-get-active-layer image))))
             (gimp-image-scale image width height)
             (gimp-file-save RUN-NONINTERACTIVE image drawable filename filename)
             (gimp-image-delete image))
           (set! filelist (cdr filelist)))))
```

3. Run: `(batch-resize "C:/path/to/images/*.png" 1024 1024)`

**Or use GIMP's built-in batch mode:**

```bash
# Resize all PNGs in folder to 512x512
gimp -i -b '(batch-resize "*.png" 512 512)' -b '(gimp-quit 0)'
```

---

## TASK 7: Fix Midjourney Artifacts

### Remove Blur/Soft Edges

1. Filters → Enhance → Sharpen (Unsharp Mask)
   - Radius: 1.0
   - Amount: 0.5
   - Threshold: 0
2. OK

### Remove Noise/Grain

1. Filters → Noise → Despeckle
   - Adaptive
   - Radius: 5
2. OK

### Fix Banding (Gradient Artifacts)

1. Filters → Noise → RGB Noise
   - Correlated: Checked
   - Amount: 0.02 (very small!)
2. This adds tiny noise to break up banding

---

## TASK 8: Create Variations

### Color Shift (Create Red/Blue/Green Variants)

1. Colors → Hue-Saturation
2. Hue: +30 (shifts to red), -30 (shifts to blue)
3. Export as new file

### Brightness Variants (Day/Night)

1. Colors → Brightness-Contrast
2. Brightness: -50 (darker), +50 (lighter)
3. Export

### Glow Effect

1. Filters → Light and Shadow → Bloom
   - Threshold: 50
   - Softness: 30
   - Glow radius: 10
2. OK

---

## TASK 9: Create Icons from Large Assets

**Problem:** You have a 2048x2048 building, need a 128x128 icon

**Steps:**
1. Open large asset
2. Select → All
3. Edit → Copy
4. File → Create → From Clipboard
5. Image → Scale Image → 128x128px
6. Filters → Enhance → Sharpen (Unsharp Mask)
7. Export

---

## TASK 10: Export for React Native

### Optimal Settings

**For ALL Assets:**
- Format: PNG
- Compression: 9 (maximum)
- Bit depth: 8-bit (not 16-bit)
- Color space: sRGB

**For Transparency:**
- Always use PNG (not JPG)
- Add Alpha Channel before exporting

**For Backgrounds (No Transparency):**
- Can use JPG if file size matters
- Quality: 90%

**Naming Convention:**
```
tarot_shop_01.png
tarot_shop_02.png
ground_tile_grass_01.png
character_idle_001.png
```

---

## Common Mistakes to Avoid

1. **Don't save as GIMP's .xcf format** - always export as PNG
2. **Don't forget Alpha Channel** - needed for transparency
3. **Don't over-compress** - use TinyPNG, not low-quality JPG
4. **Don't resize without sharpening** - images look blurry
5. **Don't use white backgrounds** - use transparency or match game background

---

## Keyboard Shortcuts (Essential)

- **R** - Rectangle Select
- **U** - Fuzzy Select (Magic Wand)
- **F** - Free Select (Lasso)
- **M** - Move tool
- **Z** - Zoom
- **Ctrl+Z** - Undo
- **Ctrl+Shift+E** - Export
- **Ctrl+L** - Levels
- **Ctrl+U** - Hue-Saturation

---

## Next Steps

After editing your assets in GIMP:
1. Optimize with TinyPNG.com
2. Test in React Native app
3. Adjust if needed
4. Document which assets go where

**For advanced asset work, see:** `GAME_ASSET_ARTIST.md`
