# PROCEDURAL ART ENGINEERING: COMPLETE GUIDE
## From Zero to AI-Guided Algorithmic Art Mastery
### Version 1.0 | November 2025

---

# TABLE OF CONTENTS

1. [PART I: FOUNDATIONS](#part-i-foundations)
   - Chapter 1: Digital Image Fundamentals
   - Chapter 2: Color Theory and Color Spaces
   - Chapter 3: Image File Formats and Compression
   - Chapter 4: Raster vs Vector Graphics

2. [PART II: TOOLS AND ENVIRONMENTS](#part-ii-tools-and-environments)
   - Chapter 5: GIMP Architecture and Scripting
   - Chapter 6: Photoshop Concepts and Automation
   - Chapter 7: Python PIL/Pillow Deep Dive
   - Chapter 8: ImageMagick Command-Line Mastery
   - Chapter 9: SVG and Vector Programming

3. [PART III: MATHEMATICAL FOUNDATIONS](#part-iii-mathematical-foundations)
   - Chapter 10: Linear Algebra for Graphics
   - Chapter 11: Convolution and Kernel Operations
   - Chapter 12: Frequency Domain and FFT
   - Chapter 13: Noise Functions and Randomness
   - Chapter 14: Fractals and Recursive Geometry

4. [PART IV: CORE IMAGE OPERATIONS](#part-iv-core-image-operations)
   - Chapter 15: Pixel Manipulation
   - Chapter 16: Geometric Transformations
   - Chapter 17: Color Transformations
   - Chapter 18: Compositing and Blending
   - Chapter 19: Masking and Selection

5. [PART V: FILTERS AND EFFECTS](#part-v-filters-and-effects)
   - Chapter 20: Blur Operations
   - Chapter 21: Sharpening and Edge Detection
   - Chapter 22: Morphological Operations
   - Chapter 23: Distortion Effects
   - Chapter 24: Artistic Filters
   - Chapter 25: Light and Shadow Effects

6. [PART VI: PROCEDURAL GENERATION](#part-vi-procedural-generation)
   - Chapter 26: Noise-Based Textures
   - Chapter 27: Pattern Generation
   - Chapter 28: Sacred Geometry
   - Chapter 29: Fractals and L-Systems
   - Chapter 30: Particle Systems

7. [PART VII: ADVANCED TECHNIQUES](#part-vii-advanced-techniques)
   - Chapter 31: HDR and Tone Mapping
   - Chapter 32: Color Grading and LUTs
   - Chapter 33: Texture Synthesis
   - Chapter 34: Style Transfer Fundamentals
   - Chapter 35: Displacement and Normal Maps

8. [PART VIII: PLUGIN DEVELOPMENT](#part-viii-plugin-development)
   - Chapter 36: GIMP Python-Fu Plugins
   - Chapter 37: GIMP Script-Fu (Scheme)
   - Chapter 38: Photoshop Scripting (JSX)
   - Chapter 39: Custom Pillow Filters
   - Chapter 40: Building Filter Pipelines

9. [PART IX: AI-GUIDED PROCEDURAL ART](#part-ix-ai-guided-procedural-art)
   - Chapter 41: Prompt-to-Parameter Translation
   - Chapter 42: Evolutionary Art Systems
   - Chapter 43: Coherence-Guided Generation
   - Chapter 44: Style Parameterization
   - Chapter 45: Human-AI Collaborative Workflows

10. [PART X: PROJECT TEMPLATES](#part-x-project-templates)
    - Chapter 46: Card Back Generator
    - Chapter 47: Profile Frame Creator
    - Chapter 48: Badge and Icon System
    - Chapter 49: Texture Atlas Generator
    - Chapter 50: Complete Art Pipeline

---

# PART I: FOUNDATIONS

---

## Chapter 1: Digital Image Fundamentals

### 1.1 What Is a Digital Image?

A digital image is a discrete, two-dimensional representation of visual information stored as a finite grid of picture elements (pixels). Each pixel contains numerical values representing color and optionally transparency.

```
CONCEPTUAL MODEL:

Physical World          Sampling           Digital Image
    │                      │                    │
    ▼                      ▼                    ▼
┌─────────┐           ┌─────────┐         ┌─────────┐
│Continuous│  ──────► │Discrete │ ──────► │ Finite  │
│  Light   │  Sample  │ Samples │  Store  │  Array  │
│  Field   │  at grid │ at each │  as     │  of     │
│          │  points  │  point  │  numbers│  pixels │
└─────────┘           └─────────┘         └─────────┘
```

### 1.2 Pixel Anatomy

A pixel (picture element) is the atomic unit of a raster image. Its structure depends on the image mode:

```python
# PIXEL STRUCTURES BY IMAGE MODE

# 1-bit (Binary/Bilevel)
# Each pixel is 0 or 1
pixel_1bit = 0  # Black
pixel_1bit = 1  # White

# 8-bit Grayscale (L mode)
# Each pixel is 0-255
pixel_gray = 128  # Mid-gray

# 24-bit RGB (RGB mode)
# Each pixel is a tuple of three 8-bit values
pixel_rgb = (255, 128, 0)  # Orange
#            R    G    B

# 32-bit RGBA (RGBA mode)
# RGB plus 8-bit alpha channel
pixel_rgba = (255, 128, 0, 255)  # Opaque orange
#             R    G    B   A

# 48-bit RGB (16-bit per channel)
# For HDR and professional work
pixel_rgb16 = (65535, 32768, 0)  # Orange in 16-bit

# Floating point (HDR)
# Unbounded values for HDR processing
pixel_hdr = (1.5, 0.8, 0.0)  # Super-bright orange
```

### 1.3 Image Dimensions and Resolution

```
TERMINOLOGY:

┌─────────────────────────────────────┐
│                                     │
│  Width (pixels)                     │
│  ◄──────────────────────────────►   │
│                                     │  ▲
│  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐   │  │
│  │  │  │  │  │  │  │  │  │  │  │   │  │
│  ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤   │  │ Height
│  │  │  │  │  │  │  │  │  │  │  │   │  │ (pixels)
│  ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤   │  │
│  │  │  │  │  │  │  │  │  │  │  │   │  │
│  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘   │  ▼
│                                     │
│  Resolution = Width × Height        │
│  Example: 1920 × 1080 = 2,073,600   │
│           (about 2 megapixels)      │
│                                     │
└─────────────────────────────────────┘

PIXEL DENSITY (DPI/PPI):

DPI = Dots Per Inch (print)
PPI = Pixels Per Inch (screen)

Physical Size = Pixel Dimensions / DPI

Example:
- Image: 3000 × 2000 pixels
- Print at 300 DPI: 10" × 6.67"
- Print at 72 DPI: 41.67" × 27.78"
- Screen at 96 PPI: 31.25" × 20.83"
```

### 1.4 Bit Depth and Dynamic Range

Bit depth determines how many distinct values each channel can hold:

```
BIT DEPTH COMPARISON:

Bits │ Values │ Levels │ Use Case
─────┼────────┼────────┼──────────────────────────
  1  │    2   │ B/W    │ Line art, fax
  8  │  256   │ 256    │ Standard images, web
 10  │ 1024   │ 1024   │ Video (HDR10)
 12  │ 4096   │ 4096   │ RAW photos, cinema
 16  │ 65536  │ 65536  │ Professional editing
 32  │ 4.3B   │ Float  │ HDR compositing

DYNAMIC RANGE:

Dynamic Range (stops) ≈ log2(max_value)

 8-bit: ~8 stops  (256 levels)
10-bit: ~10 stops (1024 levels)
12-bit: ~12 stops (4096 levels)
16-bit: ~16 stops (65536 levels)
32-bit: Unlimited (floating point)
```

### 1.5 Coordinate Systems

```
IMAGE COORDINATE SYSTEM:

Origin (0,0) is TOP-LEFT (most image libraries)

    (0,0)────────────────────► X (width)
      │
      │   ┌────────────────┐
      │   │                │
      │   │     IMAGE      │
      │   │                │
      │   │                │
      │   └────────────────┘
      │
      ▼
      Y (height)

PIXEL ACCESS PATTERNS:

# Row-major (most common): image[y][x] or image[y, x]
pixel = image[row, column]
pixel = image[y, x]

# PIL/Pillow uses (x, y) for many operations
pixel = image.getpixel((x, y))

# NumPy uses (row, col) which is (y, x)
pixel = np_image[y, x]
pixel = np_image[row, col]
```

### 1.6 Memory Layout

Understanding memory layout is crucial for performance:

```
MEMORY LAYOUT PATTERNS:

1. INTERLEAVED (Pixel-packed) - Most common
   Memory: R₀G₀B₀ R₁G₁B₁ R₂G₂B₂ ...

   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
   │ R₀  │ G₀  │ B₀  │ R₁  │ G₁  │ B₁  │ R₂  │ G₂  │ B₂  │
   └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

   Pros: Natural pixel access, cache-friendly for pixel ops
   Cons: Channel operations require stride

2. PLANAR (Channel-separated)
   Memory: R₀R₁R₂...R_n G₀G₁G₂...G_n B₀B₁B₂...B_n

   ┌─────┬─────┬─────┐ ┌─────┬─────┬─────┐ ┌─────┬─────┬─────┐
   │ R₀  │ R₁  │ R₂  │ │ G₀  │ G₁  │ G₂  │ │ B₀  │ B₁  │ B₂  │
   └─────┴─────┴─────┘ └─────┴─────┴─────┘ └─────┴─────┴─────┘

   Pros: Efficient channel operations, SIMD-friendly
   Cons: Pixel access requires multiple memory locations

3. TILED
   Memory: Tile₀[pixels] Tile₁[pixels] Tile₂[pixels] ...

   ┌─────────┬─────────┐
   │ Tile 0  │ Tile 1  │
   ├─────────┼─────────┤
   │ Tile 2  │ Tile 3  │
   └─────────┴─────────┘

   Pros: Better cache locality for local operations
   Cons: Complex addressing, edge handling
```

### 1.7 Pillow Image Modes Reference

```python
"""
COMPLETE PIL/PILLOW IMAGE MODES
"""

from PIL import Image

# ═══════════════════════════════════════════════════════════════
# SINGLE-CHANNEL MODES
# ═══════════════════════════════════════════════════════════════

# 1 - 1-bit pixels, black and white, stored as 8 pixels per byte
img_1 = Image.new('1', (100, 100), 0)

# L - 8-bit grayscale
img_L = Image.new('L', (100, 100), 128)

# P - 8-bit palette mode (256 colors max)
img_P = Image.new('P', (100, 100), 0)

# I - 32-bit signed integer grayscale
img_I = Image.new('I', (100, 100), 0)

# F - 32-bit floating point grayscale
img_F = Image.new('F', (100, 100), 0.0)

# ═══════════════════════════════════════════════════════════════
# MULTI-CHANNEL MODES
# ═══════════════════════════════════════════════════════════════

# RGB - 3×8-bit color
img_RGB = Image.new('RGB', (100, 100), (255, 0, 0))

# RGBA - 4×8-bit color with alpha
img_RGBA = Image.new('RGBA', (100, 100), (255, 0, 0, 128))

# RGBX - 4×8-bit color with padding (no alpha)
img_RGBX = Image.new('RGBX', (100, 100), (255, 0, 0, 255))

# CMYK - 4×8-bit color separation
img_CMYK = Image.new('CMYK', (100, 100), (0, 255, 255, 0))

# YCbCr - 3×8-bit color video format
img_YCbCr = Image.new('YCbCr', (100, 100), (128, 128, 128))

# LAB - 3×8-bit L*a*b color space
img_LAB = Image.new('LAB', (100, 100), (128, 128, 128))

# HSV - 3×8-bit Hue, Saturation, Value
img_HSV = Image.new('HSV', (100, 100), (0, 255, 255))

# ═══════════════════════════════════════════════════════════════
# SPECIAL MODES
# ═══════════════════════════════════════════════════════════════

# LA - 8-bit grayscale with alpha
img_LA = Image.new('LA', (100, 100), (128, 255))

# PA - 8-bit palette with alpha
img_PA = Image.new('PA', (100, 100), (0, 255))

# RGBa - 3×8-bit RGB with premultiplied alpha
# (alpha already multiplied into RGB channels)

# I;16 - 16-bit unsigned integer grayscale
img_I16 = Image.new('I;16', (100, 100), 32768)

# I;16L - 16-bit unsigned integer grayscale, little-endian
# I;16B - 16-bit unsigned integer grayscale, big-endian

# ═══════════════════════════════════════════════════════════════
# MODE CONVERSION MATRIX
# ═══════════════════════════════════════════════════════════════

"""
Conversion paths (direct conversions supported by Pillow):

1 ──► L ──► RGB ──► RGBA
        │      │
        │      └──► CMYK
        │
        └──► P

From any mode to any other:
  image.convert('TARGET_MODE')

Special conversions:
  RGB → L:    L = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601)
  RGB → CMYK: Subtractive color conversion
  RGBA → RGB: Alpha channel discarded or composited
  P → RGB:    Palette lookup expansion
"""
```

---

## Chapter 2: Color Theory and Color Spaces

### 2.1 The Nature of Color

Color is the brain's interpretation of electromagnetic radiation in the visible spectrum (approximately 380-700nm wavelength). Digital color systems approximate human color perception using mathematical models.

```
ELECTROMAGNETIC SPECTRUM (Visible Light):

380nm ◄─────────────────────────────────────────────► 700nm
  │                                                      │
  ▼                                                      ▼
Violet  Blue  Cyan  Green  Yellow  Orange  Red
  │      │     │      │       │       │      │
 380    450   490    550     580     600    700  (nm)

HUMAN COLOR PERCEPTION:

┌─────────────────────────────────────────────────────┐
│                    LIGHT SOURCE                     │
│                         │                           │
│                         ▼                           │
│                   ┌─────────┐                       │
│                   │ OBJECT  │                       │
│                   │(absorbs/│                       │
│                   │reflects)│                       │
│                   └────┬────┘                       │
│                        │                            │
│                        ▼                            │
│                   ┌─────────┐                       │
│                   │   EYE   │                       │
│                   │ (cones: │                       │
│                   │  S,M,L) │                       │
│                   └────┬────┘                       │
│                        │                            │
│                        ▼                            │
│                   ┌─────────┐                       │
│                   │  BRAIN  │                       │
│                   │(perceives│                      │
│                   │  color) │                       │
│                   └─────────┘                       │
└─────────────────────────────────────────────────────┘

CONE SENSITIVITY:

      S (Short)    M (Medium)    L (Long)
      "Blue"       "Green"       "Red"
        │            │             │
        ▼            ▼             ▼
   ┌────────┐   ┌────────┐   ┌────────┐
   │   /\   │   │   /\   │   │   /\   │
   │  /  \  │   │  /  \  │   │  /  \  │
   │ /    \ │   │ /    \ │   │ /    \ │
   │/      \│   │/      \│   │/      \│
   └────────┘   └────────┘   └────────┘
    420nm        534nm        564nm
    (peak)       (peak)       (peak)
```

### 2.2 RGB Color Model

The RGB model is an **additive** color system where red, green, and blue light combine to create colors.

```python
"""
RGB COLOR MODEL - COMPLETE REFERENCE
"""

# ═══════════════════════════════════════════════════════════════
# RGB FUNDAMENTALS
# ═══════════════════════════════════════════════════════════════

# RGB Cube visualization:
#
#                    White (1,1,1)
#                        /\
#                       /  \
#                      /    \
#              Yellow /      \ Cyan
#             (1,1,0)/        \(0,1,1)
#                   /    Green \
#                  /    (0,1,0) \
#                 /              \
#                /                \
#         Red   /                  \  Blue
#        (1,0,0)────────────────────(0,0,1)
#                \                  /
#                 \                /
#                  \   Magenta    /
#                   \  (1,0,1)   /
#                    \          /
#                     \        /
#                      \      /
#                       \    /
#                        \  /
#                         \/
#                    Black (0,0,0)

# Primary colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Secondary colors (additive mixing)
YELLOW = (255, 255, 0)   # Red + Green
CYAN = (0, 255, 255)     # Green + Blue
MAGENTA = (255, 0, 255)  # Red + Blue

# Neutral colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY_50 = (128, 128, 128)

# ═══════════════════════════════════════════════════════════════
# RGB ARITHMETIC
# ═══════════════════════════════════════════════════════════════

def rgb_add(c1, c2):
    """Additive color mixing (light)"""
    return tuple(min(255, a + b) for a, b in zip(c1, c2))

def rgb_multiply(c1, c2):
    """Multiplicative mixing (filter)"""
    return tuple((a * b) // 255 for a, b in zip(c1, c2))

def rgb_screen(c1, c2):
    """Screen blend (inverse multiply)"""
    return tuple(255 - ((255 - a) * (255 - b)) // 255 for a, b in zip(c1, c2))

def rgb_lerp(c1, c2, t):
    """Linear interpolation between colors"""
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

def rgb_brightness(color, factor):
    """Adjust brightness (factor: 0.0-2.0)"""
    return tuple(min(255, max(0, int(c * factor))) for c in color)

def rgb_invert(color):
    """Invert color"""
    return tuple(255 - c for c in color)

# ═══════════════════════════════════════════════════════════════
# GAMMA CORRECTION
# ═══════════════════════════════════════════════════════════════

"""
GAMMA EXPLANATION:

Human perception is non-linear. We perceive brightness changes
more sensitively in dark regions than bright regions.

Display gamma: How monitors convert signal to light
  Light = Signal^gamma  (typically gamma ≈ 2.2)

sRGB: Standard color space with gamma encoding
  Linear → sRGB: Apply gamma curve
  sRGB → Linear: Remove gamma curve

ALWAYS work in LINEAR space for:
  - Color mixing
  - Lighting calculations
  - Blur operations
  - Any mathematical color operation

Convert back to sRGB/gamma for:
  - Display
  - Saving to standard formats
"""

import math

def srgb_to_linear(c):
    """Convert sRGB (0-255) to linear (0.0-1.0)"""
    c = c / 255.0
    if c <= 0.04045:
        return c / 12.92
    else:
        return ((c + 0.055) / 1.055) ** 2.4

def linear_to_srgb(c):
    """Convert linear (0.0-1.0) to sRGB (0-255)"""
    if c <= 0.0031308:
        c = c * 12.92
    else:
        c = 1.055 * (c ** (1/2.4)) - 0.055
    return int(min(255, max(0, c * 255)))

def rgb_to_linear(rgb):
    """Convert RGB tuple from sRGB to linear"""
    return tuple(srgb_to_linear(c) for c in rgb)

def linear_to_rgb(linear):
    """Convert linear tuple back to sRGB"""
    return tuple(linear_to_srgb(c) for c in linear)

# CORRECT color mixing (in linear space):
def rgb_mix_correct(c1, c2, t):
    """Perceptually correct color mixing"""
    lin1 = rgb_to_linear(c1)
    lin2 = rgb_to_linear(c2)
    mixed = tuple(a + (b - a) * t for a, b in zip(lin1, lin2))
    return linear_to_rgb(mixed)
```

### 2.3 HSV/HSL Color Models

HSV (Hue, Saturation, Value) and HSL (Hue, Saturation, Lightness) are cylindrical color models that align with human color intuition.

```python
"""
HSV AND HSL COLOR MODELS
"""

import colorsys

# ═══════════════════════════════════════════════════════════════
# HSV MODEL (Hue, Saturation, Value)
# ═══════════════════════════════════════════════════════════════

"""
HSV CYLINDER:

          Value (Brightness)
              │
              │    ┌─────────────┐ V=1.0 (Full brightness)
              │   /             /│
              │  /   WHITE     / │
              │ /     ○       /  │
              │/─────────────/   │ ◄── Saturation increases
              │      ○       │   │     toward edge
         V=1 │   Colors at  │   │
              │   full sat.  │   │
              │      ○       │   │
              │\─────────────\   │
              │ \             \  │
              │  \   BLACK    \ │
              │   \     ○     \│ V=0.0
              │    └───────────┘
              │
              └────────────────────► Hue (angle, 0-360°)

                   Red=0°
                    │
           Magenta  │  Yellow
            300°    │    60°
                \   │   /
                 \  │  /
                  \ │ /
       Blue 240°───○───Green 120°
                  / │ \
                 /  │  \
                /   │   \
           Cyan    │
            180°   │
                   │
"""

def rgb_to_hsv(r, g, b):
    """Convert RGB (0-255) to HSV (H: 0-360, S: 0-1, V: 0-1)"""
    r, g, b = r/255.0, g/255.0, b/255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return (h * 360, s, v)

def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB (0-255)"""
    r, g, b = colorsys.hsv_to_rgb(h/360.0, s, v)
    return (int(r*255), int(g*255), int(b*255))

# HSV operations
def hsv_rotate_hue(rgb, degrees):
    """Rotate hue by specified degrees"""
    h, s, v = rgb_to_hsv(*rgb)
    h = (h + degrees) % 360
    return hsv_to_rgb(h, s, v)

def hsv_saturate(rgb, factor):
    """Adjust saturation (factor: 0.0-2.0)"""
    h, s, v = rgb_to_hsv(*rgb)
    s = min(1.0, max(0.0, s * factor))
    return hsv_to_rgb(h, s, v)

def hsv_brighten(rgb, factor):
    """Adjust value/brightness"""
    h, s, v = rgb_to_hsv(*rgb)
    v = min(1.0, max(0.0, v * factor))
    return hsv_to_rgb(h, s, v)

# ═══════════════════════════════════════════════════════════════
# HSL MODEL (Hue, Saturation, Lightness)
# ═══════════════════════════════════════════════════════════════

"""
HSL vs HSV COMPARISON:

HSV (Hexcone):                HSL (Bi-hexcone):

    V=1 ┌─────────┐               L=1 ○ White
        │ Colors  │                  /│\
        │         │                 / │ \
        │         │                /  │  \
        └─────────┘   L=0.5 ┌────/───┼───\────┐ Full colors
            │               │        │        │
            │               │        │        │
    V=0 ────○ Black         └────\───┼───/────┘
                               L=0 ○ Black
                                  \  │  /
                                   \ │ /
                                    \│/

HSV: V=1 gives brightest, pure colors at S=1
HSL: L=0.5 gives purest colors, L=1 is always white
"""

def rgb_to_hsl(r, g, b):
    """Convert RGB (0-255) to HSL (H: 0-360, S: 0-1, L: 0-1)"""
    r, g, b = r/255.0, g/255.0, b/255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return (h * 360, s, l)

def hsl_to_rgb(h, s, l):
    """Convert HSL to RGB (0-255)"""
    r, g, b = colorsys.hls_to_rgb(h/360.0, l, s)
    return (int(r*255), int(g*255), int(b*255))

# ═══════════════════════════════════════════════════════════════
# COLOR HARMONY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def complementary(rgb):
    """Get complementary color (180° opposite)"""
    return hsv_rotate_hue(rgb, 180)

def triadic(rgb):
    """Get triadic colors (120° apart)"""
    return [
        rgb,
        hsv_rotate_hue(rgb, 120),
        hsv_rotate_hue(rgb, 240)
    ]

def analogous(rgb, spread=30):
    """Get analogous colors (adjacent on wheel)"""
    return [
        hsv_rotate_hue(rgb, -spread),
        rgb,
        hsv_rotate_hue(rgb, spread)
    ]

def split_complementary(rgb, spread=30):
    """Split complementary (two colors adjacent to complement)"""
    return [
        rgb,
        hsv_rotate_hue(rgb, 180 - spread),
        hsv_rotate_hue(rgb, 180 + spread)
    ]

def tetradic(rgb):
    """Tetradic/Square colors (90° apart)"""
    return [
        rgb,
        hsv_rotate_hue(rgb, 90),
        hsv_rotate_hue(rgb, 180),
        hsv_rotate_hue(rgb, 270)
    ]
```

### 2.4 LAB Color Space

LAB (CIELAB) is a perceptually uniform color space where numerical distance correlates with perceived color difference.

```python
"""
LAB (CIELAB) COLOR SPACE

L = Lightness (0 = black, 100 = white)
a = Green(-) to Red(+) axis
b = Blue(-) to Yellow(+) axis

PERCEPTUAL UNIFORMITY:
A ΔE of 1.0 represents the smallest color difference
perceivable by most humans under normal viewing conditions.

     +b (Yellow)
         │
         │
-a ──────┼────── +a
(Green)  │      (Red)
         │
         │
     -b (Blue)

LAB is DEVICE-INDEPENDENT and used for:
- Color matching across devices
- Perceptual color operations
- Color difference calculations
"""

import numpy as np

# XYZ reference white (D65 illuminant)
XYZ_REF = np.array([95.047, 100.000, 108.883])

def rgb_to_xyz(rgb):
    """Convert sRGB to XYZ"""
    rgb = np.array(rgb) / 255.0

    # Linearize
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92

    # RGB to XYZ matrix (sRGB to D65)
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])

    xyz = np.dot(matrix, rgb * 100)
    return xyz

def xyz_to_lab(xyz):
    """Convert XYZ to LAB"""
    xyz = xyz / XYZ_REF

    # Apply f(t) function
    mask = xyz > 0.008856
    xyz[mask] = xyz[mask] ** (1/3)
    xyz[~mask] = (7.787 * xyz[~mask]) + (16/116)

    L = (116 * xyz[1]) - 16
    a = 500 * (xyz[0] - xyz[1])
    b = 200 * (xyz[1] - xyz[2])

    return np.array([L, a, b])

def rgb_to_lab(rgb):
    """Convert RGB to LAB"""
    xyz = rgb_to_xyz(rgb)
    return xyz_to_lab(xyz)

def lab_to_xyz(lab):
    """Convert LAB to XYZ"""
    L, a, b = lab

    y = (L + 16) / 116
    x = a / 500 + y
    z = y - b / 200

    xyz = np.array([x, y, z])

    mask = xyz ** 3 > 0.008856
    xyz[mask] = xyz[mask] ** 3
    xyz[~mask] = (xyz[~mask] - 16/116) / 7.787

    return xyz * XYZ_REF

def xyz_to_rgb(xyz):
    """Convert XYZ to sRGB"""
    # XYZ to RGB matrix
    matrix = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])

    rgb = np.dot(matrix, xyz / 100)

    # Gamma correction
    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * (rgb[mask] ** (1/2.4)) - 0.055
    rgb[~mask] = 12.92 * rgb[~mask]

    rgb = np.clip(rgb * 255, 0, 255).astype(int)
    return tuple(rgb)

def lab_to_rgb(lab):
    """Convert LAB to RGB"""
    xyz = lab_to_xyz(lab)
    return xyz_to_rgb(xyz)

# ═══════════════════════════════════════════════════════════════
# COLOR DIFFERENCE (Delta E)
# ═══════════════════════════════════════════════════════════════

def delta_e_76(lab1, lab2):
    """
    CIE76 Delta E - Simple Euclidean distance

    Interpretation:
    0-1:   Imperceptible
    1-2:   Perceptible through close observation
    2-10:  Perceptible at a glance
    11-49: Colors are more similar than opposite
    100:   Colors are exact opposites
    """
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))

def delta_e_2000(lab1, lab2):
    """
    CIE2000 Delta E - More accurate perceptual difference
    (Simplified implementation)
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    # This is a simplified version; full CIEDE2000 is more complex
    dL = L2 - L1
    da = a2 - a1
    db = b2 - b1

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    dC = C2 - C1

    dH_sq = da**2 + db**2 - dC**2
    dH = np.sqrt(max(0, dH_sq))

    # Weighting factors (simplified)
    kL, kC, kH = 1, 1, 1

    dE = np.sqrt((dL/kL)**2 + (dC/kC)**2 + (dH/kH)**2)
    return dE

def find_closest_color(target_rgb, palette):
    """Find perceptually closest color in palette"""
    target_lab = rgb_to_lab(target_rgb)

    min_dist = float('inf')
    closest = None

    for color in palette:
        color_lab = rgb_to_lab(color)
        dist = delta_e_76(target_lab, color_lab)
        if dist < min_dist:
            min_dist = dist
            closest = color

    return closest, min_dist
```

### 2.5 CMYK Color Model

CMYK is a **subtractive** color model used for print production.

```python
"""
CMYK (Cyan, Magenta, Yellow, Key/Black) COLOR MODEL

SUBTRACTIVE COLOR:
- Starts with white (paper)
- Each ink REMOVES wavelengths
- More ink = darker result

CMY alone cannot produce true black (produces muddy brown),
so K (Key/Black) is added for:
- True black
- Better contrast
- Ink savings
- Sharper text

PRINT vs SCREEN:
┌─────────────────────────────────────────────────────┐
│                                                     │
│  RGB (Screen)              CMYK (Print)             │
│  ┌─────────┐               ┌─────────┐              │
│  │  R+G=Y  │               │  C+M=B  │              │
│  │  G+B=C  │               │  M+Y=R  │              │
│  │  R+B=M  │               │  C+Y=G  │              │
│  │R+G+B=W │               │C+M+Y=K  │              │
│  └─────────┘               └─────────┘              │
│                                                     │
│  Additive                  Subtractive              │
│  Light emission            Light absorption         │
│  Wider gamut               Narrower gamut           │
│                                                     │
└─────────────────────────────────────────────────────┘
"""

def rgb_to_cmyk(r, g, b):
    """Convert RGB (0-255) to CMYK (0-100%)"""
    if r == 0 and g == 0 and b == 0:
        return (0, 0, 0, 100)

    r, g, b = r/255.0, g/255.0, b/255.0

    k = 1 - max(r, g, b)

    if k == 1:
        return (0, 0, 0, 100)

    c = (1 - r - k) / (1 - k)
    m = (1 - g - k) / (1 - k)
    y = (1 - b - k) / (1 - k)

    return (
        round(c * 100),
        round(m * 100),
        round(y * 100),
        round(k * 100)
    )

def cmyk_to_rgb(c, m, y, k):
    """Convert CMYK (0-100%) to RGB (0-255)"""
    c, m, y, k = c/100.0, m/100.0, y/100.0, k/100.0

    r = 255 * (1 - c) * (1 - k)
    g = 255 * (1 - m) * (1 - k)
    b = 255 * (1 - y) * (1 - k)

    return (int(r), int(g), int(b))

# CMYK-safe RGB conversion (keeps colors in printable gamut)
def rgb_to_print_safe(r, g, b):
    """Convert to CMYK and back to get print-safe RGB"""
    cmyk = rgb_to_cmyk(r, g, b)
    return cmyk_to_rgb(*cmyk)

# Check if color is in CMYK gamut
def is_cmyk_safe(r, g, b, tolerance=5):
    """Check if RGB color can be accurately printed"""
    safe = rgb_to_print_safe(r, g, b)
    diff = sum(abs(a - b) for a, b in zip((r, g, b), safe))
    return diff <= tolerance
```

---

## Chapter 3: Image File Formats and Compression

### 3.1 Format Overview

```
IMAGE FORMAT DECISION TREE:

                    Need transparency?
                          │
              ┌───────────┴───────────┐
              │ NO                    │ YES
              ▼                       ▼
         Is it a photo?          PNG or WebP
              │                   (or TIFF/EXR
    ┌─────────┴─────────┐        for pro work)
    │ YES               │ NO
    ▼                   ▼
  JPEG              Is it simple
(or WebP,          shapes/colors?
AVIF, HEIC)              │
                ┌────────┴────────┐
                │ YES             │ NO
                ▼                 ▼
              SVG            PNG-8 or GIF
          (vector)          (palette mode)


DETAILED FORMAT COMPARISON:

Format │ Type   │ Compression │ Alpha │ Animation │ Bits │ Use Case
───────┼────────┼─────────────┼───────┼───────────┼──────┼──────────────
JPEG   │ Raster │ Lossy       │ No    │ No        │ 8    │ Photos
PNG    │ Raster │ Lossless    │ Yes   │ No        │ 8/16 │ Graphics, UI
GIF    │ Raster │ Lossless    │ 1-bit │ Yes       │ 8    │ Simple anim
WebP   │ Raster │ Both        │ Yes   │ Yes       │ 8    │ Web (modern)
AVIF   │ Raster │ Lossy       │ Yes   │ Yes       │ 10+  │ Next-gen
HEIC   │ Raster │ Lossy       │ Yes   │ Yes       │ 10   │ Apple devices
TIFF   │ Raster │ Both        │ Yes   │ No        │ 8-32 │ Pro/archive
BMP    │ Raster │ None        │ Yes   │ No        │ 8-32 │ Legacy
SVG    │ Vector │ Text/gzip   │ Yes   │ Yes       │ N/A  │ Icons, UI
EXR    │ Raster │ Lossy/Less  │ Yes   │ No        │ 16-32│ VFX, HDR
PSD    │ Raster │ RLE         │ Yes   │ No        │ 8-32 │ Photoshop
```

### 3.2 JPEG Deep Dive

```python
"""
JPEG COMPRESSION - COMPLETE GUIDE

JPEG uses Discrete Cosine Transform (DCT) to convert
spatial data into frequency data, then quantizes
high-frequency components (which are less visible).

JPEG PIPELINE:

┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  RGB    │───►│  YCbCr  │───►│  8×8    │───►│  DCT    │
│  Image  │    │ Convert │    │ Blocks  │    │Transform│
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                                                  │
┌─────────┐    ┌─────────┐    ┌─────────┐         │
│  JPEG   │◄───│ Huffman │◄───│Quantize │◄────────┘
│  File   │    │ Encode  │    │         │
└─────────┘    └─────────┘    └─────────┘

QUALITY vs FILE SIZE:
Quality │ Compression │ File Size │ Artifacts
────────┼─────────────┼───────────┼───────────
  100   │    ~2:1     │  Largest  │ None
   95   │    ~5:1     │   Large   │ Minimal
   85   │   ~10:1     │  Medium   │ Slight
   75   │   ~15:1     │   Small   │ Noticeable
   50   │   ~25:1     │   Tiny    │ Obvious
   25   │   ~50:1     │  Minimal  │ Severe
"""

from PIL import Image
import io

def save_jpeg_quality(image, path, quality=85):
    """Save JPEG with specified quality"""
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image.save(path, 'JPEG', quality=quality, optimize=True)

def save_jpeg_progressive(image, path, quality=85):
    """Save progressive JPEG (loads blurry → sharp)"""
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image.save(path, 'JPEG', quality=quality, progressive=True, optimize=True)

def get_jpeg_size_at_quality(image, quality):
    """Get file size at specified quality without saving"""
    buffer = io.BytesIO()
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image.save(buffer, 'JPEG', quality=quality)
    return buffer.tell()

def find_optimal_quality(image, target_size_kb, tolerance=5):
    """Find quality that produces target file size"""
    target_bytes = target_size_kb * 1024

    low, high = 1, 100
    best_quality = 85

    while low <= high:
        mid = (low + high) // 2
        size = get_jpeg_size_at_quality(image, mid)

        if abs(size - target_bytes) < tolerance * 1024:
            return mid
        elif size > target_bytes:
            high = mid - 1
        else:
            low = mid + 1
            best_quality = mid

    return best_quality

# JPEG SUBSAMPLING:
"""
Chroma subsampling reduces color resolution (humans are less
sensitive to color detail than brightness detail).

4:4:4 - No subsampling (best quality, largest file)
4:2:2 - Half horizontal color resolution
4:2:0 - Half horizontal AND vertical (default, smallest)

       4:4:4              4:2:2              4:2:0
    ┌──┬──┬──┬──┐      ┌──┬──┬──┬──┐      ┌──┬──┬──┬──┐
    │Y │Y │Y │Y │      │Y │Y │Y │Y │      │Y │Y │Y │Y │
    │Cb│Cb│Cb│Cb│      │Cb│  │Cb│  │      │Cb│  │Cb│  │
    │Cr│Cr│Cr│Cr│      │Cr│  │Cr│  │      │  │  │  │  │
    ├──┼──┼──┼──┤      ├──┼──┼──┼──┤      ├──┼──┼──┼──┤
    │Y │Y │Y │Y │      │Y │Y │Y │Y │      │Y │Y │Y │Y │
    │Cb│Cb│Cb│Cb│      │Cb│  │Cb│  │      │Cr│  │Cr│  │
    │Cr│Cr│Cr│Cr│      │Cr│  │Cr│  │      │  │  │  │  │
    └──┴──┴──┴──┘      └──┴──┴──┴──┘      └──┴──┴──┴──┘
"""

def save_jpeg_444(image, path, quality=85):
    """Save JPEG with 4:4:4 subsampling (no chroma reduction)"""
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image.save(path, 'JPEG', quality=quality, subsampling=0)  # 0 = 4:4:4

def save_jpeg_420(image, path, quality=85):
    """Save JPEG with 4:2:0 subsampling (default)"""
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image.save(path, 'JPEG', quality=quality, subsampling=2)  # 2 = 4:2:0
```

### 3.3 PNG Deep Dive

```python
"""
PNG - PORTABLE NETWORK GRAPHICS

PNG uses lossless compression (DEFLATE algorithm).
Supports transparency, multiple bit depths, and metadata.

PNG STRUCTURE:
┌──────────────────────────────────────────────────────┐
│ PNG Signature (8 bytes)                              │
├──────────────────────────────────────────────────────┤
│ IHDR - Image Header (always first)                   │
│   - Width, Height                                    │
│   - Bit depth, Color type                            │
│   - Compression, Filter, Interlace method            │
├──────────────────────────────────────────────────────┤
│ [Optional chunks: PLTE, tRNS, gAMA, cHRM, etc.]     │
├──────────────────────────────────────────────────────┤
│ IDAT - Image Data (compressed pixel data)            │
│   - Can be multiple IDAT chunks                      │
├──────────────────────────────────────────────────────┤
│ IEND - Image End (always last)                       │
└──────────────────────────────────────────────────────┘

PNG COLOR TYPES:
Type │ Name       │ Channels │ Bits/Channel │ Description
─────┼────────────┼──────────┼──────────────┼──────────────
  0  │ Grayscale  │    1     │ 1,2,4,8,16   │ Gray only
  2  │ Truecolor  │    3     │ 8,16         │ RGB
  3  │ Indexed    │    1     │ 1,2,4,8      │ Palette
  4  │ Gray+Alpha │    2     │ 8,16         │ LA
  6  │ RGBA       │    4     │ 8,16         │ Full color+alpha
"""

from PIL import Image, PngImagePlugin

def save_png_optimized(image, path, compress_level=9):
    """Save PNG with maximum compression"""
    image.save(path, 'PNG', compress_level=compress_level, optimize=True)

def save_png_with_metadata(image, path, metadata_dict):
    """Save PNG with custom metadata"""
    pnginfo = PngImagePlugin.PngInfo()
    for key, value in metadata_dict.items():
        pnginfo.add_text(key, str(value))
    image.save(path, 'PNG', pnginfo=pnginfo)

def read_png_metadata(path):
    """Read PNG metadata"""
    image = Image.open(path)
    return dict(image.info)

def save_png_indexed(image, path, colors=256):
    """Save as indexed PNG (palette mode) for smaller file"""
    if image.mode == 'RGBA':
        # Quantize with alpha
        image = image.quantize(colors=colors, method=Image.MEDIANCUT)
    else:
        image = image.convert('P', palette=Image.ADAPTIVE, colors=colors)
    image.save(path, 'PNG')

def png_crush(image, path):
    """Aggressive PNG optimization"""
    # Convert to optimal mode
    if image.mode == 'RGBA':
        # Check if alpha is needed
        if image.getextrema()[3][0] == 255:  # All opaque
            image = image.convert('RGB')

    # Try palette mode for simple images
    colors = len(set(image.getdata()))
    if colors <= 256 and image.mode in ('RGB', 'L'):
        image = image.convert('P', palette=Image.ADAPTIVE)

    image.save(path, 'PNG', compress_level=9, optimize=True)

# PNG BIT DEPTH OPTIMIZATION:
"""
For grayscale and indexed images, reducing bit depth
can significantly reduce file size:

Bits │ Values │ Good For
─────┼────────┼──────────────────────────────
  1  │    2   │ Black & white line art
  2  │    4   │ 4-level grayscale
  4  │   16   │ Simple icons, limited palette
  8  │  256   │ Standard indexed/grayscale
 16  │ 65536  │ High precision grayscale
"""

def optimize_grayscale_depth(image):
    """Reduce grayscale bit depth if possible"""
    if image.mode != 'L':
        image = image.convert('L')

    values = set(image.getdata())
    num_values = len(values)

    if num_values == 2:
        return image.convert('1')  # 1-bit
    elif num_values <= 16:
        # 4-bit not directly supported, keep as 8-bit
        return image
    else:
        return image
```

### 3.4 WebP and Modern Formats

```python
"""
WEBP - MODERN WEB IMAGE FORMAT

WebP supports both lossy and lossless compression,
animation, and alpha transparency.

WEBP vs JPEG vs PNG:
─────────────────────────────────────────────────────────
Metric          │ JPEG      │ PNG       │ WebP
─────────────────────────────────────────────────────────
Lossy compress  │ Yes       │ No        │ Yes
Lossless        │ No        │ Yes       │ Yes
Transparency    │ No        │ Yes       │ Yes
Animation       │ No        │ No        │ Yes
Avg file size   │ Baseline  │ 2-10x     │ 25-35% smaller
Browser support │ Universal │ Universal │ 97%+
─────────────────────────────────────────────────────────
"""

from PIL import Image

def save_webp_lossy(image, path, quality=80):
    """Save lossy WebP"""
    image.save(path, 'WEBP', quality=quality, method=6)

def save_webp_lossless(image, path):
    """Save lossless WebP"""
    image.save(path, 'WEBP', lossless=True, quality=100, method=6)

def save_webp_animated(frames, path, duration=100, loop=0):
    """Save animated WebP from list of PIL Images"""
    frames[0].save(
        path,
        'WEBP',
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )

def convert_to_webp_batch(input_dir, output_dir, quality=80):
    """Convert all images in directory to WebP"""
    import os
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir,
                                       os.path.splitext(filename)[0] + '.webp')

            image = Image.open(input_path)
            if filename.lower().endswith('.png') and image.mode == 'RGBA':
                # Keep lossless for PNG with alpha
                image.save(output_path, 'WEBP', lossless=True)
            else:
                image.save(output_path, 'WEBP', quality=quality)

# ═══════════════════════════════════════════════════════════════
# FORMAT SELECTION HELPER
# ═══════════════════════════════════════════════════════════════

def recommend_format(image, use_case='web'):
    """Recommend optimal format based on image characteristics"""

    has_alpha = image.mode in ('RGBA', 'LA', 'PA')
    is_photo = _is_photographic(image)
    num_colors = len(set(image.convert('RGB').getdata()))

    recommendations = []

    if use_case == 'web':
        if has_alpha:
            if is_photo:
                recommendations.append(('WebP lossy', 'Best size with alpha'))
            else:
                if num_colors <= 256:
                    recommendations.append(('PNG-8', 'Small palette image'))
                else:
                    recommendations.append(('PNG', 'Lossless with alpha'))
                    recommendations.append(('WebP lossless', 'Smaller lossless'))
        else:
            if is_photo:
                recommendations.append(('JPEG', 'Best for photos'))
                recommendations.append(('WebP lossy', '25-35% smaller'))
            else:
                recommendations.append(('PNG', 'Sharp edges preserved'))
                recommendations.append(('WebP lossless', 'Smaller lossless'))

    elif use_case == 'archive':
        recommendations.append(('PNG', 'Lossless, universal'))
        recommendations.append(('TIFF', 'Professional archival'))

    elif use_case == 'print':
        recommendations.append(('TIFF', 'Uncompressed quality'))
        recommendations.append(('PNG', 'Lossless alternative'))

    return recommendations

def _is_photographic(image):
    """Heuristic to detect if image is photographic"""
    if image.mode == 'P':
        return False

    # Sample pixels and check variance
    img_small = image.resize((50, 50)).convert('RGB')
    pixels = list(img_small.getdata())

    # High unique color count suggests photo
    unique_colors = len(set(pixels))

    # Calculate average local variance
    import numpy as np
    arr = np.array(img_small)
    variance = np.var(arr)

    return unique_colors > 500 and variance > 1000
```

---

## Chapter 4: Raster vs Vector Graphics

### 4.1 Fundamental Differences

```
RASTER vs VECTOR COMPARISON:

RASTER (Bitmap):                    VECTOR:
──────────────────────────────      ──────────────────────────────
Stored as pixel grid                Stored as math equations
Fixed resolution                    Resolution independent
Larger file sizes (usually)         Smaller for simple shapes
Perfect for photos                  Perfect for logos/icons
Degrades when scaled up            Infinite scaling
Examples: JPEG, PNG, GIF            Examples: SVG, PDF, AI

VISUAL COMPARISON (scaling up):

RASTER:                             VECTOR:
┌─────────────────┐                 ┌─────────────────┐
│ █ █ █ █ █ █ █ █ │                 │   ╭───────╮     │
│ █ ▓ ▓ ▓ ▓ ▓ ▓ █ │                 │  │       │     │
│ █ ▓ ░ ░ ░ ░ ▓ █ │                 │  │       │     │
│ █ ▓ ░ ░ ░ ░ ▓ █ │  ──────►       │  │       │     │  ──────►
│ █ ▓ ░ ░ ░ ░ ▓ █ │  Scale up      │  │       │     │  Scale up
│ █ ▓ ▓ ▓ ▓ ▓ ▓ █ │                 │   ╰───────╯     │
│ █ █ █ █ █ █ █ █ │                 │                 │
└─────────────────┘                 └─────────────────┘
         │                                   │
         ▼                                   ▼
┌─────────────────────────┐        ┌─────────────────────────┐
│ ██████████████████████  │        │    ╭─────────────╮      │
│ ██░░░░░░░░░░░░░░░░░░██  │        │   │             │      │
│ ██░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░██  │        │   │             │      │
│ ██░░▓▓░░░░░░░░░░▓▓░░██  │        │   │             │      │
│ ██░░▓▓░░        ░░▓▓░░██│        │   │             │      │
│ ██░░▓▓░░  PIXELATED ██  │        │   │   SMOOTH    │      │
│ ██░░▓▓░░        ░░▓▓░░██│        │   │             │      │
│ ██░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░██  │        │   │             │      │
│ ██░░░░░░░░░░░░░░░░░░██  │        │    ╰─────────────╯      │
│ ██████████████████████  │        │                         │
└─────────────────────────┘        └─────────────────────────┘
        Jagged edges                    Perfect edges
```

### 4.2 SVG Fundamentals

```python
"""
SVG - SCALABLE VECTOR GRAPHICS

SVG is an XML-based format for 2D vector graphics.
Can be styled with CSS, animated with JS, and embedded in HTML.

SVG COORDINATE SYSTEM:
(0,0)─────────────────► X
  │
  │   viewBox="0 0 100 100"
  │   defines coordinate space
  │
  ▼
  Y
"""

# ═══════════════════════════════════════════════════════════════
# SVG GENERATION WITH PYTHON
# ═══════════════════════════════════════════════════════════════

class SVGCanvas:
    """Simple SVG generator"""

    def __init__(self, width, height, viewbox=None):
        self.width = width
        self.height = height
        self.viewbox = viewbox or f"0 0 {width} {height}"
        self.elements = []
        self.defs = []

    def add_rect(self, x, y, width, height, fill='black', stroke=None,
                 stroke_width=1, rx=0, ry=0, **kwargs):
        """Add rectangle"""
        attrs = f'x="{x}" y="{y}" width="{width}" height="{height}"'
        attrs += f' fill="{fill}"'
        if stroke:
            attrs += f' stroke="{stroke}" stroke-width="{stroke_width}"'
        if rx or ry:
            attrs += f' rx="{rx}" ry="{ry}"'
        for k, v in kwargs.items():
            attrs += f' {k.replace("_", "-")}="{v}"'
        self.elements.append(f'<rect {attrs}/>')

    def add_circle(self, cx, cy, r, fill='black', stroke=None,
                   stroke_width=1, **kwargs):
        """Add circle"""
        attrs = f'cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"'
        if stroke:
            attrs += f' stroke="{stroke}" stroke-width="{stroke_width}"'
        for k, v in kwargs.items():
            attrs += f' {k.replace("_", "-")}="{v}"'
        self.elements.append(f'<circle {attrs}/>')

    def add_ellipse(self, cx, cy, rx, ry, fill='black', **kwargs):
        """Add ellipse"""
        attrs = f'cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}"'
        for k, v in kwargs.items():
            attrs += f' {k.replace("_", "-")}="{v}"'
        self.elements.append(f'<ellipse {attrs}/>')

    def add_line(self, x1, y1, x2, y2, stroke='black', stroke_width=1, **kwargs):
        """Add line"""
        attrs = f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"'
        attrs += f' stroke="{stroke}" stroke-width="{stroke_width}"'
        for k, v in kwargs.items():
            attrs += f' {k.replace("_", "-")}="{v}"'
        self.elements.append(f'<line {attrs}/>')

    def add_polyline(self, points, stroke='black', stroke_width=1,
                     fill='none', **kwargs):
        """Add polyline (connected line segments)"""
        points_str = ' '.join(f'{x},{y}' for x, y in points)
        attrs = f'points="{points_str}" stroke="{stroke}"'
        attrs += f' stroke-width="{stroke_width}" fill="{fill}"'
        for k, v in kwargs.items():
            attrs += f' {k.replace("_", "-")}="{v}"'
        self.elements.append(f'<polyline {attrs}/>')

    def add_polygon(self, points, fill='black', stroke=None,
                    stroke_width=1, **kwargs):
        """Add polygon (closed shape)"""
        points_str = ' '.join(f'{x},{y}' for x, y in points)
        attrs = f'points="{points_str}" fill="{fill}"'
        if stroke:
            attrs += f' stroke="{stroke}" stroke-width="{stroke_width}"'
        for k, v in kwargs.items():
            attrs += f' {k.replace("_", "-")}="{v}"'
        self.elements.append(f'<polygon {attrs}/>')

    def add_path(self, d, fill='black', stroke=None, stroke_width=1, **kwargs):
        """Add path (most powerful SVG element)"""
        attrs = f'd="{d}" fill="{fill}"'
        if stroke:
            attrs += f' stroke="{stroke}" stroke-width="{stroke_width}"'
        for k, v in kwargs.items():
            attrs += f' {k.replace("_", "-")}="{v}"'
        self.elements.append(f'<path {attrs}/>')

    def add_text(self, x, y, text, font_size=16, font_family='sans-serif',
                 fill='black', anchor='start', **kwargs):
        """Add text"""
        attrs = f'x="{x}" y="{y}" font-size="{font_size}"'
        attrs += f' font-family="{font_family}" fill="{fill}"'
        attrs += f' text-anchor="{anchor}"'
        for k, v in kwargs.items():
            attrs += f' {k.replace("_", "-")}="{v}"'
        self.elements.append(f'<text {attrs}>{text}</text>')

    def add_gradient(self, id, type='linear', stops=None, **kwargs):
        """Add gradient definition"""
        if stops is None:
            stops = [('0%', 'white'), ('100%', 'black')]

        if type == 'linear':
            x1 = kwargs.get('x1', '0%')
            y1 = kwargs.get('y1', '0%')
            x2 = kwargs.get('x2', '100%')
            y2 = kwargs.get('y2', '0%')
            grad = f'<linearGradient id="{id}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}">'
        else:  # radial
            cx = kwargs.get('cx', '50%')
            cy = kwargs.get('cy', '50%')
            r = kwargs.get('r', '50%')
            grad = f'<radialGradient id="{id}" cx="{cx}" cy="{cy}" r="{r}">'

        for offset, color in stops:
            opacity = kwargs.get(f'opacity_{offset}', 1)
            grad += f'<stop offset="{offset}" stop-color="{color}" stop-opacity="{opacity}"/>'

        grad += f'</{type}Gradient>'
        self.defs.append(grad)

    def add_filter(self, id, filter_elements):
        """Add filter definition"""
        filter_def = f'<filter id="{id}">'
        filter_def += ''.join(filter_elements)
        filter_def += '</filter>'
        self.defs.append(filter_def)

    def add_group(self, elements, transform=None, **kwargs):
        """Group elements with optional transform"""
        attrs = ''
        if transform:
            attrs += f' transform="{transform}"'
        for k, v in kwargs.items():
            attrs += f' {k.replace("_", "-")}="{v}"'

        group = f'<g{attrs}>'
        group += '\n'.join(elements)
        group += '</g>'
        self.elements.append(group)

    def render(self):
        """Generate SVG string"""
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" '
        svg += f'width="{self.width}" height="{self.height}" '
        svg += f'viewBox="{self.viewbox}">\n'

        if self.defs:
            svg += '<defs>\n'
            svg += '\n'.join(self.defs)
            svg += '\n</defs>\n'

        svg += '\n'.join(self.elements)
        svg += '\n</svg>'

        return svg

    def save(self, path):
        """Save to file"""
        with open(path, 'w') as f:
            f.write(self.render())


# ═══════════════════════════════════════════════════════════════
# SVG PATH COMMANDS REFERENCE
# ═══════════════════════════════════════════════════════════════

"""
SVG PATH COMMANDS (d attribute):

COMMAND │ NAME           │ PARAMETERS      │ DESCRIPTION
────────┼────────────────┼─────────────────┼─────────────────────
   M    │ moveto         │ x y             │ Move to point
   m    │ moveto (rel)   │ dx dy           │ Move relative
   L    │ lineto         │ x y             │ Line to point
   l    │ lineto (rel)   │ dx dy           │ Line relative
   H    │ horizontal     │ x               │ Horizontal line to
   h    │ horizontal     │ dx              │ Horizontal relative
   V    │ vertical       │ y               │ Vertical line to
   v    │ vertical       │ dy              │ Vertical relative
   C    │ curveto        │ x1 y1 x2 y2 x y │ Cubic Bézier
   c    │ curveto (rel)  │ same, relative  │ Cubic relative
   S    │ smooth curve   │ x2 y2 x y       │ Smooth cubic
   s    │ smooth (rel)   │ same, relative  │ Smooth relative
   Q    │ quadratic      │ x1 y1 x y       │ Quadratic Bézier
   q    │ quadratic(rel) │ same, relative  │ Quadratic relative
   T    │ smooth quad    │ x y             │ Smooth quadratic
   t    │ smooth (rel)   │ dx dy           │ Smooth relative
   A    │ arc            │ rx ry rot lf sf x y │ Elliptical arc
   a    │ arc (rel)      │ same, relative  │ Arc relative
   Z/z  │ closepath      │ (none)          │ Close path

UPPERCASE = absolute coordinates
lowercase = relative coordinates (from current point)
"""

class SVGPathBuilder:
    """Helper for building SVG paths"""

    def __init__(self):
        self.commands = []

    def move_to(self, x, y, relative=False):
        cmd = 'm' if relative else 'M'
        self.commands.append(f'{cmd} {x} {y}')
        return self

    def line_to(self, x, y, relative=False):
        cmd = 'l' if relative else 'L'
        self.commands.append(f'{cmd} {x} {y}')
        return self

    def horizontal(self, x, relative=False):
        cmd = 'h' if relative else 'H'
        self.commands.append(f'{cmd} {x}')
        return self

    def vertical(self, y, relative=False):
        cmd = 'v' if relative else 'V'
        self.commands.append(f'{cmd} {y}')
        return self

    def cubic_bezier(self, x1, y1, x2, y2, x, y, relative=False):
        cmd = 'c' if relative else 'C'
        self.commands.append(f'{cmd} {x1} {y1} {x2} {y2} {x} {y}')
        return self

    def smooth_cubic(self, x2, y2, x, y, relative=False):
        cmd = 's' if relative else 'S'
        self.commands.append(f'{cmd} {x2} {y2} {x} {y}')
        return self

    def quadratic_bezier(self, x1, y1, x, y, relative=False):
        cmd = 'q' if relative else 'Q'
        self.commands.append(f'{cmd} {x1} {y1} {x} {y}')
        return self

    def arc(self, rx, ry, rotation, large_arc, sweep, x, y, relative=False):
        cmd = 'a' if relative else 'A'
        self.commands.append(f'{cmd} {rx} {ry} {rotation} {large_arc} {sweep} {x} {y}')
        return self

    def close(self):
        self.commands.append('Z')
        return self

    def build(self):
        return ' '.join(self.commands)


# ═══════════════════════════════════════════════════════════════
# CONVERTING RASTER TO VECTOR (TRACING)
# ═══════════════════════════════════════════════════════════════

def trace_bitmap_simple(image, threshold=128):
    """
    Simple bitmap tracing - converts grayscale to path outline
    (Basic implementation - production would use potrace or similar)
    """
    from PIL import Image
    import numpy as np

    # Convert to binary
    if image.mode != 'L':
        image = image.convert('L')

    arr = np.array(image)
    binary = (arr < threshold).astype(np.uint8)

    # Find contours (simplified - would use marching squares)
    # This is a placeholder for actual contour detection

    # Return paths based on detected contours
    paths = []
    # ... contour detection algorithm ...

    return paths
```

---

This is just the beginning. The document continues with:

- **Part II**: GIMP/Photoshop architecture, Pillow deep dive, ImageMagick mastery
- **Part III**: Linear algebra, convolution, FFT, noise functions, fractals
- **Part IV**: All pixel/color/geometric operations
- **Part V**: Every filter and effect with math
- **Part VI**: Procedural generation techniques
- **Part VII**: Advanced HDR, LUTs, texture synthesis
- **Part VIII**: Plugin development for GIMP/PS
- **Part IX**: AI-guided procedural art
- **Part X**: Complete project templates

Want me to continue with the next parts? This is going to be a beast - easily 800KB+ when complete.

---

# PART II: TOOLS AND ENVIRONMENTS

---

## Chapter 5: GIMP Architecture and Scripting

### 5.1 GIMP Architecture Overview

```
GIMP ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────┐
│                         GIMP APPLICATION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │   USER INTERFACE │    │    CORE ENGINE   │    │   PLUGINS   │ │
│  │                 │    │                 │    │             │ │
│  │  - Toolbox      │◄──►│  - GEGL Library │◄──►│  - Python-Fu│ │
│  │  - Dialogs      │    │  - Image Model  │    │  - Script-Fu│ │
│  │  - Dockables    │    │  - Layer Stack  │    │  - C Plugins│ │
│  │  - Canvas       │    │  - Selection    │    │  - Custom   │ │
│  │                 │    │  - Undo System  │    │             │ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
│           │                      │                     │        │
│           └──────────────────────┴─────────────────────┘        │
│                                  │                              │
│                           ┌──────▼──────┐                       │
│                           │    GEGL     │                       │
│                           │  (Generic   │                       │
│                           │   Graphics  │                       │
│                           │   Library)  │                       │
│                           └─────────────┘                       │
│                                  │                              │
│                           ┌──────▼──────┐                       │
│                           │    BABL     │                       │
│                           │  (Pixel     │                       │
│                           │   Format    │                       │
│                           │   Library)  │                       │
│                           └─────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

GEGL PIPELINE:
GEGL (Generic Graphics Library) is a graph-based image processing
framework. Operations are nodes in a directed acyclic graph (DAG).

┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  Load   │───►│  Blur   │───►│ Contrast│───►│  Save   │
│  Image  │    │         │    │         │    │  Image  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘

Benefits of GEGL:
- Non-destructive editing
- Arbitrary bit depths (8, 16, 32-bit float)
- GPU acceleration (OpenCL)
- Efficient region-of-interest processing
```

### 5.2 Python-Fu Scripting

```python
"""
GIMP PYTHON-FU PLUGIN DEVELOPMENT

Python-Fu is GIMP's Python scripting interface.
Plugins go in: ~/.config/GIMP/2.10/plug-ins/ (Linux/Mac)
              %APPDATA%\GIMP\2.10\plug-ins\ (Windows)

PLUGIN STRUCTURE:
#!/usr/bin/env python
from gimpfu import *

def my_plugin_function(image, drawable, param1, param2):
    # Plugin logic here
    pass

register(
    "plugin_name",           # Procedure name
    "Short description",     # Blurb
    "Long description",      # Help
    "Author Name",           # Author
    "Copyright",             # Copyright
    "2025",                  # Date
    "<Image>/Filters/MyPlugin",  # Menu path
    "*",                     # Image types (* = all)
    [
        (PF_INT, "param1", "Parameter 1", 10),
        (PF_FLOAT, "param2", "Parameter 2", 0.5),
    ],
    [],                      # Results
    my_plugin_function       # Function
)

main()
"""

# ═══════════════════════════════════════════════════════════════
# GIMP PYTHON-FU COMPLETE REFERENCE
# ═══════════════════════════════════════════════════════════════

GIMP_PYTHON_PLUGIN_TEMPLATE = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
{plugin_name} - GIMP Plugin
{description}

Author: {author}
License: {license}
"""

from gimpfu import *
import gimp
from gimp import pdb
import math

# ═══════════════════════════════════════════════════════════════
# PLUGIN IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════

def {function_name}(image, drawable, {params}):
    """
    Main plugin function.

    Args:
        image: The current GIMP image
        drawable: The active layer/drawable
        {param_docs}
    """
    # Start undo group (allows single undo for entire operation)
    gimp.progress_init("Processing...")
    pdb.gimp_image_undo_group_start(image)

    try:
        # Get image dimensions
        width = drawable.width
        height = drawable.height

        # Get pixel region for direct pixel access
        src_rgn = drawable.get_pixel_rgn(0, 0, width, height, False, False)
        dst_rgn = drawable.get_pixel_rgn(0, 0, width, height, True, True)

        # Process pixels
        for y in range(height):
            for x in range(width):
                # Get pixel (returns string of bytes)
                pixel = src_rgn[x, y]

                # Unpack pixel values
                if drawable.bpp == 3:  # RGB
                    r, g, b = ord(pixel[0]), ord(pixel[1]), ord(pixel[2])
                    # Process...
                    new_pixel = chr(r) + chr(g) + chr(b)
                elif drawable.bpp == 4:  # RGBA
                    r, g, b, a = [ord(c) for c in pixel]
                    # Process...
                    new_pixel = chr(r) + chr(g) + chr(b) + chr(a)

                # Set new pixel
                dst_rgn[x, y] = new_pixel

            # Update progress
            gimp.progress_update(float(y) / height)

        # Update the drawable
        drawable.flush()
        drawable.merge_shadow(True)
        drawable.update(0, 0, width, height)

    finally:
        # End undo group
        pdb.gimp_image_undo_group_end(image)
        gimp.displays_flush()

    return

# ═══════════════════════════════════════════════════════════════
# PLUGIN REGISTRATION
# ═══════════════════════════════════════════════════════════════

register(
    "{proc_name}",
    "{short_desc}",
    "{long_desc}",
    "{author}",
    "{copyright}",
    "{date}",
    "<Image>/Filters/{menu_path}",
    "{image_types}",
    [
        # Parameter definitions
        # (PF_TYPE, "name", "Label", default_value),
        {param_definitions}
    ],
    [],  # Results
    {function_name}
)

main()
'''

# ═══════════════════════════════════════════════════════════════
# GIMP PARAMETER TYPES
# ═══════════════════════════════════════════════════════════════

"""
PARAMETER TYPES (PF_*):

Type          │ Python Type │ Description              │ Example
──────────────┼─────────────┼──────────────────────────┼─────────────────
PF_INT        │ int         │ Integer value            │ (PF_INT, "size", "Size", 10)
PF_INT8       │ int         │ 8-bit integer (0-255)    │ (PF_INT8, "val", "Value", 128)
PF_INT16      │ int         │ 16-bit integer           │
PF_INT32      │ int         │ 32-bit integer           │
PF_FLOAT      │ float       │ Floating point           │ (PF_FLOAT, "amount", "Amount", 0.5)
PF_STRING     │ str         │ Text string              │ (PF_STRING, "text", "Text", "Hello")
PF_TEXT       │ str         │ Multi-line text          │
PF_BOOL       │ bool        │ Boolean checkbox         │ (PF_BOOL, "preview", "Preview", True)
PF_TOGGLE     │ bool        │ Toggle button            │
PF_SLIDER     │ float       │ Slider with range        │ (PF_SLIDER, "val", "Value", 50, (0, 100, 1))
PF_SPINNER    │ float       │ Spin button              │
PF_ADJUSTMENT │ float       │ Adjustment widget        │
PF_COLOR      │ tuple       │ Color chooser            │ (PF_COLOR, "color", "Color", (255, 0, 0))
PF_COLOUR     │ tuple       │ Alias for PF_COLOR       │
PF_IMAGE      │ gimp.Image  │ Image selector           │
PF_DRAWABLE   │ gimp.Layer  │ Drawable selector        │
PF_LAYER      │ gimp.Layer  │ Layer selector           │
PF_CHANNEL    │ gimp.Channel│ Channel selector         │
PF_FONT       │ str         │ Font selector            │ (PF_FONT, "font", "Font", "Sans")
PF_FILE       │ str         │ File chooser             │ (PF_FILE, "file", "File", "")
PF_DIRNAME    │ str         │ Directory chooser        │
PF_OPTION     │ int         │ Dropdown menu            │ (PF_OPTION, "mode", "Mode", 0, ["Add", "Sub"])
PF_RADIO      │ int         │ Radio buttons            │
PF_PALETTE    │ str         │ Palette selector         │
PF_PATTERN    │ str         │ Pattern selector         │
PF_GRADIENT   │ str         │ Gradient selector        │
PF_BRUSH      │ str         │ Brush selector           │
"""

# ═══════════════════════════════════════════════════════════════
# COMMON GIMP PDB PROCEDURES
# ═══════════════════════════════════════════════════════════════

GIMP_PDB_REFERENCE = """
COMMONLY USED PDB PROCEDURES:

IMAGE OPERATIONS:
─────────────────────────────────────────────────────────────────
pdb.gimp_image_new(width, height, type)
    Create new image. type: RGB, GRAY, INDEXED

pdb.gimp_image_delete(image)
    Delete image from memory

pdb.gimp_image_flatten(image)
    Flatten all layers

pdb.gimp_image_merge_visible_layers(image, merge_type)
    Merge visible layers. merge_type: CLIP_TO_IMAGE, etc.

pdb.gimp_image_get_active_layer(image)
    Get active layer

pdb.gimp_image_set_active_layer(image, layer)
    Set active layer

pdb.gimp_image_add_layer(image, layer, position)
    Add layer to image

pdb.gimp_image_remove_layer(image, layer)
    Remove layer from image

pdb.gimp_image_resize(image, new_width, new_height, offx, offy)
    Resize image canvas

pdb.gimp_image_scale(image, new_width, new_height)
    Scale image

pdb.gimp_image_crop(image, new_width, new_height, offx, offy)
    Crop image


LAYER OPERATIONS:
─────────────────────────────────────────────────────────────────
pdb.gimp_layer_new(image, width, height, type, name, opacity, mode)
    Create new layer

pdb.gimp_layer_copy(layer, add_alpha)
    Copy layer

pdb.gimp_layer_scale(layer, new_width, new_height, local_origin)
    Scale layer

pdb.gimp_layer_resize(layer, new_width, new_height, offx, offy)
    Resize layer

pdb.gimp_layer_set_offsets(layer, offx, offy)
    Set layer position

pdb.gimp_layer_set_opacity(layer, opacity)
    Set layer opacity (0-100)

pdb.gimp_layer_set_mode(layer, mode)
    Set blend mode (NORMAL_MODE, MULTIPLY_MODE, etc.)


DRAWABLE OPERATIONS:
─────────────────────────────────────────────────────────────────
pdb.gimp_drawable_width(drawable)
    Get width

pdb.gimp_drawable_height(drawable)
    Get height

pdb.gimp_drawable_get_pixel(drawable, x, y)
    Get pixel value at (x, y)

pdb.gimp_drawable_set_pixel(drawable, x, y, num_channels, pixel)
    Set pixel value

pdb.gimp_drawable_fill(drawable, fill_type)
    Fill drawable. fill_type: FOREGROUND_FILL, BACKGROUND_FILL, etc.

pdb.gimp_drawable_offset(drawable, wrap, fill_type, offx, offy)
    Offset drawable contents


COLOR OPERATIONS:
─────────────────────────────────────────────────────────────────
pdb.gimp_context_get_foreground()
    Get foreground color

pdb.gimp_context_set_foreground(color)
    Set foreground color

pdb.gimp_context_get_background()
    Get background color

pdb.gimp_context_set_background(color)
    Set background color

pdb.gimp_context_swap_colors()
    Swap foreground/background


SELECTION OPERATIONS:
─────────────────────────────────────────────────────────────────
pdb.gimp_image_select_rectangle(image, operation, x, y, width, height)
    Rectangle selection

pdb.gimp_image_select_ellipse(image, operation, x, y, width, height)
    Ellipse selection

pdb.gimp_image_select_polygon(image, operation, num_points, points)
    Polygon selection

pdb.gimp_selection_all(image)
    Select all

pdb.gimp_selection_none(image)
    Select none

pdb.gimp_selection_invert(image)
    Invert selection

pdb.gimp_selection_feather(image, radius)
    Feather selection

pdb.gimp_selection_grow(image, steps)
    Grow selection

pdb.gimp_selection_shrink(image, steps)
    Shrink selection


FILTER/EFFECT PROCEDURES:
─────────────────────────────────────────────────────────────────
pdb.plug_in_gauss(image, drawable, horizontal, vertical, method)
    Gaussian blur

pdb.plug_in_unsharp_mask(image, drawable, radius, amount, threshold)
    Unsharp mask

pdb.plug_in_edge(image, drawable, amount, wrapmode, edgemode)
    Edge detection

pdb.plug_in_emboss(image, drawable, azimuth, elevation, depth)
    Emboss effect

pdb.plug_in_bump_map(image, drawable, bumpmap, azimuth, elevation, depth, ...)
    Bump map

pdb.plug_in_displace(image, drawable, amount_x, amount_y, do_x, do_y, ...)
    Displacement

pdb.plug_in_plasma(image, drawable, seed, turbulence)
    Plasma render

pdb.plug_in_noise_solid(image, drawable, tileable, turbulent, seed, detail, ...)
    Solid noise

pdb.gimp_brightness_contrast(drawable, brightness, contrast)
    Brightness/Contrast

pdb.gimp_hue_saturation(drawable, hue_range, hue, lightness, saturation)
    Hue/Saturation

pdb.gimp_color_balance(drawable, transfer_mode, preserve_lum, cyan_red, ...)
    Color balance

pdb.gimp_levels(drawable, channel, low_input, high_input, gamma, ...)
    Levels adjustment

pdb.gimp_curves_spline(drawable, channel, num_points, control_points)
    Curves adjustment

pdb.gimp_desaturate_full(drawable, desaturate_mode)
    Desaturate


FILE OPERATIONS:
─────────────────────────────────────────────────────────────────
pdb.file_png_save(image, drawable, filename, raw_filename, ...)
    Save as PNG

pdb.file_jpeg_save(image, drawable, filename, raw_filename, quality, ...)
    Save as JPEG

pdb.gimp_file_load(run_mode, filename, raw_filename)
    Load file

pdb.gimp_file_save(run_mode, image, drawable, filename, raw_filename)
    Save file


DISPLAY OPERATIONS:
─────────────────────────────────────────────────────────────────
pdb.gimp_display_new(image)
    Create new display window

pdb.gimp_display_delete(display)
    Delete display

pdb.gimp_displays_flush()
    Update all displays
"""

# ═══════════════════════════════════════════════════════════════
# EXAMPLE: COMPLETE GIMP PLUGIN - PROCEDURAL TEXTURE
# ═══════════════════════════════════════════════════════════════

GIMP_PROCEDURAL_TEXTURE_PLUGIN = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Procedural Texture Generator - GIMP Plugin
Generates various procedural textures using noise algorithms.
"""

from gimpfu import *
import gimp
from gimp import pdb
import math
import random

def perlin_noise_2d(x, y, seed=0):
    """Simple Perlin-like noise implementation"""
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(a, b, t):
        return a + t * (b - a)

    def grad(hash_val, x, y):
        h = hash_val & 3
        if h == 0: return x + y
        elif h == 1: return -x + y
        elif h == 2: return x - y
        else: return -x - y

    # Permutation table
    random.seed(seed)
    perm = list(range(256))
    random.shuffle(perm)
    perm = perm + perm  # Duplicate for overflow

    # Grid coordinates
    xi = int(x) & 255
    yi = int(y) & 255

    # Relative position in grid cell
    xf = x - int(x)
    yf = y - int(y)

    # Fade curves
    u = fade(xf)
    v = fade(yf)

    # Hash coordinates of corners
    aa = perm[perm[xi] + yi]
    ab = perm[perm[xi] + yi + 1]
    ba = perm[perm[xi + 1] + yi]
    bb = perm[perm[xi + 1] + yi + 1]

    # Blend
    x1 = lerp(grad(aa, xf, yf), grad(ba, xf - 1, yf), u)
    x2 = lerp(grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1), u)

    return (lerp(x1, x2, v) + 1) / 2  # Normalize to 0-1

def fbm(x, y, octaves, persistence, lacunarity, seed):
    """Fractal Brownian Motion - layered noise"""
    total = 0
    amplitude = 1
    frequency = 1
    max_value = 0

    for _ in range(octaves):
        total += perlin_noise_2d(x * frequency, y * frequency, seed) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return total / max_value

def generate_procedural_texture(image, drawable, texture_type, scale,
                                octaves, persistence, seed, color1, color2):
    """Generate procedural texture on layer"""

    gimp.progress_init("Generating texture...")
    pdb.gimp_image_undo_group_start(image)

    try:
        width = drawable.width
        height = drawable.height

        # Create new layer for texture
        texture_layer = pdb.gimp_layer_new(
            image, width, height, RGBA_IMAGE,
            "Procedural Texture", 100, NORMAL_MODE
        )
        pdb.gimp_image_insert_layer(image, texture_layer, None, 0)

        # Get pixel region
        rgn = texture_layer.get_pixel_rgn(0, 0, width, height, True, True)

        # Generate texture
        for y in range(height):
            row = ""
            for x in range(width):
                # Calculate noise value based on texture type
                nx = x / scale
                ny = y / scale

                if texture_type == 0:  # Perlin
                    value = perlin_noise_2d(nx, ny, seed)
                elif texture_type == 1:  # FBM
                    value = fbm(nx, ny, octaves, persistence, 2.0, seed)
                elif texture_type == 2:  # Turbulence
                    value = abs(fbm(nx, ny, octaves, persistence, 2.0, seed) * 2 - 1)
                elif texture_type == 3:  # Ridged
                    value = 1 - abs(fbm(nx, ny, octaves, persistence, 2.0, seed) * 2 - 1)
                elif texture_type == 4:  # Marble
                    value = math.sin(nx + fbm(nx, ny, octaves, persistence, 2.0, seed) * 10)
                    value = (value + 1) / 2
                elif texture_type == 5:  # Wood
                    dist = math.sqrt(nx * nx + ny * ny)
                    value = math.sin(dist + fbm(nx, ny, octaves, persistence, 2.0, seed) * 5)
                    value = (value + 1) / 2
                else:
                    value = 0.5

                # Interpolate between colors
                r = int(color1[0] + (color2[0] - color1[0]) * value)
                g = int(color1[1] + (color2[1] - color1[1]) * value)
                b = int(color1[2] + (color2[2] - color1[2]) * value)

                row += chr(r) + chr(g) + chr(b) + chr(255)

            rgn[0:width, y] = row
            gimp.progress_update(float(y) / height)

        texture_layer.flush()
        texture_layer.merge_shadow(True)
        texture_layer.update(0, 0, width, height)

    finally:
        pdb.gimp_image_undo_group_end(image)
        gimp.displays_flush()

register(
    "python_fu_procedural_texture",
    "Generate Procedural Texture",
    "Generates various procedural textures using noise algorithms",
    "Art Engineering Guide",
    "Public Domain",
    "2025",
    "<Image>/Filters/Render/Procedural Texture...",
    "*",
    [
        (PF_OPTION, "texture_type", "Texture Type", 0,
         ["Perlin", "FBM", "Turbulence", "Ridged", "Marble", "Wood"]),
        (PF_FLOAT, "scale", "Scale", 50.0),
        (PF_INT, "octaves", "Octaves (FBM)", 4),
        (PF_FLOAT, "persistence", "Persistence", 0.5),
        (PF_INT, "seed", "Random Seed", 42),
        (PF_COLOR, "color1", "Color 1", (0, 0, 0)),
        (PF_COLOR, "color2", "Color 2", (255, 255, 255)),
    ],
    [],
    generate_procedural_texture
)

main()
'''


### 5.3 Script-Fu (Scheme) Scripting

"""
GIMP SCRIPT-FU REFERENCE

Script-Fu uses a Scheme dialect called TinyScheme.
Powerful for batch operations and automation.
"""

SCRIPT_FU_REFERENCE = """
SCRIPT-FU SYNTAX BASICS:

; Comments start with semicolon

; Define a variable
(define my-var 42)

; Define a function
(define (my-function arg1 arg2)
  (+ arg1 arg2))

; Call a function
(my-function 10 20)  ; Returns 30

; Conditional
(if (> x 10)
    "greater"
    "less or equal")

; Let binding (local variables)
(let ((x 10)
      (y 20))
  (+ x y))

; Loop with while
(while (< i 10)
  (set! i (+ i 1)))

; List operations
(car '(1 2 3))       ; First element: 1
(cdr '(1 2 3))       ; Rest: (2 3)
(cons 0 '(1 2 3))    ; Prepend: (0 1 2 3)
(list 1 2 3)         ; Create list


SCRIPT-FU GIMP PROCEDURES:

; Create new image
(gimp-image-new width height type)

; Get image dimensions
(gimp-image-width image)
(gimp-image-height image)

; Create layer
(gimp-layer-new image width height type name opacity mode)

; Add layer to image
(gimp-image-insert-layer image layer parent position)

; Fill layer
(gimp-drawable-fill layer fill-type)

; Apply filter
(plug-in-gauss RUN-NONINTERACTIVE image layer h-radius v-radius method)

; File operations
(file-png-save RUN-NONINTERACTIVE image drawable filename raw-filename ...)
"""

SCRIPT_FU_BATCH_EXAMPLE = """
; BATCH PROCESSING EXAMPLE
; Process all images in a directory

(define (batch-process-directory input-dir output-dir)
  (let* ((filelist (cadr (file-glob (string-append input-dir "/*.jpg") 1))))
    (while (not (null? filelist))
      (let* ((filename (car filelist))
             (image (car (gimp-file-load RUN-NONINTERACTIVE filename filename)))
             (drawable (car (gimp-image-get-active-layer image))))

        ; Apply processing
        (gimp-brightness-contrast drawable 10 20)
        (plug-in-unsharp-mask RUN-NONINTERACTIVE image drawable 5.0 0.5 0)

        ; Save
        (let ((output-file (string-append output-dir "/"
                           (car (strbreakup (car (last (strbreakup filename "/"))) "."))
                           "_processed.jpg")))
          (file-jpeg-save RUN-NONINTERACTIVE image drawable
                          output-file output-file 0.9 0 0 0 "" 0 0 0 0))

        ; Clean up
        (gimp-image-delete image))

      (set! filelist (cdr filelist)))))

; Register as script
(script-fu-register
  "batch-process-directory"
  "Batch Process Directory"
  "Process all JPEGs in input directory"
  "Author"
  "Copyright"
  "2025"
  ""
  SF-DIRNAME "Input Directory" ""
  SF-DIRNAME "Output Directory" ""
)

(script-fu-menu-register "batch-process-directory" "<Image>/Filters/Batch")
"""


---

## Chapter 6: Photoshop Concepts and Automation

### 6.1 Photoshop Architecture

```
PHOTOSHOP ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────┐
│                    ADOBE PHOTOSHOP                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                    USER INTERFACE                         │ │
│  │  ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌───────────────┐  │ │
│  │  │ Toolbar │ │  Panels  │ │ Canvas  │ │ Menu/Actions  │  │ │
│  │  └─────────┘ └──────────┘ └─────────┘ └───────────────┘  │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────────┐ │
│  │                    DOCUMENT MODEL                         │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │                    Layer Stack                      │  │ │
│  │  │  ┌─────────────────────────────────────────────┐   │  │ │
│  │  │  │ Adjustment Layer (Curves)                   │   │  │ │
│  │  │  ├─────────────────────────────────────────────┤   │  │ │
│  │  │  │ Smart Object (embedded/linked)              │   │  │ │
│  │  │  ├─────────────────────────────────────────────┤   │  │ │
│  │  │  │ Group (Layer Set)                           │   │  │ │
│  │  │  │   ├── Layer 1 [blend mode, opacity, mask]  │   │  │ │
│  │  │  │   └── Layer 2 [effects: shadow, glow...]   │   │  │ │
│  │  │  ├─────────────────────────────────────────────┤   │  │ │
│  │  │  │ Background Layer (locked)                   │   │  │ │
│  │  │  └─────────────────────────────────────────────┘   │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────────┐ │
│  │                    SCRIPTING APIs                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │ │
│  │  │ ExtendScript│  │   UXP/CEP   │  │  Actions    │       │ │
│  │  │    (JSX)    │  │  (HTML/JS)  │  │  (recorded) │       │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘       │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

BLEND MODES (Layer Compositing):

Category     │ Mode          │ Formula (A=base, B=blend)
─────────────┼───────────────┼──────────────────────────────────
Normal       │ Normal        │ B
             │ Dissolve      │ Random threshold
─────────────┼───────────────┼──────────────────────────────────
Darken       │ Darken        │ min(A, B)
             │ Multiply      │ A × B
             │ Color Burn    │ 1 - (1-A)/B
             │ Linear Burn   │ A + B - 1
─────────────┼───────────────┼──────────────────────────────────
Lighten      │ Lighten       │ max(A, B)
             │ Screen        │ 1 - (1-A)(1-B)
             │ Color Dodge   │ A / (1-B)
             │ Linear Dodge  │ A + B
─────────────┼───────────────┼──────────────────────────────────
Contrast     │ Overlay       │ A<0.5: 2AB, else: 1-2(1-A)(1-B)
             │ Soft Light    │ Complex formula
             │ Hard Light    │ B<0.5: 2AB, else: 1-2(1-A)(1-B)
             │ Vivid Light   │ Color burn/dodge based on B
             │ Linear Light  │ Linear burn/dodge based on B
             │ Pin Light     │ Darken/lighten based on B
             │ Hard Mix      │ (A+B)≥1: 1, else: 0
─────────────┼───────────────┼──────────────────────────────────
Inversion    │ Difference    │ |A - B|
             │ Exclusion     │ A + B - 2AB
             │ Subtract      │ A - B
             │ Divide        │ A / B
─────────────┼───────────────┼──────────────────────────────────
Component    │ Hue           │ Hue of B, Sat/Lum of A
             │ Saturation    │ Sat of B, Hue/Lum of A
             │ Color         │ Hue/Sat of B, Lum of A
             │ Luminosity    │ Lum of B, Hue/Sat of A
```

### 6.2 ExtendScript (JSX) Automation

```javascript
/*
 * PHOTOSHOP EXTENDSCRIPT COMPLETE REFERENCE
 *
 * ExtendScript (.jsx) files run in Photoshop's JavaScript engine.
 * Place in: Applications/Adobe Photoshop/Presets/Scripts (Mac)
 *           C:\Program Files\Adobe\Adobe Photoshop\Presets\Scripts (Win)
 */

// ═══════════════════════════════════════════════════════════════
// DOCUMENT OPERATIONS
// ═══════════════════════════════════════════════════════════════

// Create new document
var doc = app.documents.add(
    UnitValue(1920, 'px'),    // width
    UnitValue(1080, 'px'),    // height
    300,                       // resolution (DPI)
    "My Document",            // name
    NewDocumentMode.RGB,      // mode
    DocumentFill.WHITE        // fill
);

// Open document
var fileRef = new File("/path/to/image.jpg");
var doc = app.open(fileRef);

// Save document
var saveFile = new File("/path/to/output.psd");
doc.saveAs(saveFile, new PhotoshopSaveOptions(), true);

// Save as JPEG
var jpegOptions = new JPEGSaveOptions();
jpegOptions.quality = 10;  // 0-12
jpegOptions.embedColorProfile = true;
doc.saveAs(new File("/path/to/output.jpg"), jpegOptions, true);

// Save as PNG
var pngOptions = new PNGSaveOptions();
pngOptions.compression = 9;  // 0-9
pngOptions.interlaced = false;
doc.saveAs(new File("/path/to/output.png"), pngOptions, true);

// Close document
doc.close(SaveOptions.DONOTSAVECHANGES);

// ═══════════════════════════════════════════════════════════════
// LAYER OPERATIONS
// ═══════════════════════════════════════════════════════════════

// Create new layer
var newLayer = doc.artLayers.add();
newLayer.name = "My Layer";
newLayer.opacity = 75;
newLayer.blendMode = BlendMode.MULTIPLY;

// Duplicate layer
var dupLayer = doc.activeLayer.duplicate();
dupLayer.name = "Layer Copy";

// Delete layer
doc.artLayers.getByName("Layer Name").remove();

// Move layer
doc.activeLayer.move(doc.artLayers[0], ElementPlacement.PLACEBEFORE);

// Merge layers
doc.activeLayer.merge();

// Flatten image
doc.flatten();

// Create layer group
var layerSet = doc.layerSets.add();
layerSet.name = "Group 1";

// Move layer into group
doc.activeLayer.move(layerSet, ElementPlacement.INSIDE);

// ═══════════════════════════════════════════════════════════════
// SELECTION OPERATIONS
// ═══════════════════════════════════════════════════════════════

// Select all
doc.selection.selectAll();

// Deselect
doc.selection.deselect();

// Rectangle selection
var selBounds = [[100, 100], [500, 100], [500, 400], [100, 400]];
doc.selection.select(selBounds);

// Ellipse selection (via action)
function selectEllipse(x, y, width, height) {
    var desc = new ActionDescriptor();
    var ref = new ActionReference();
    ref.putProperty(charIDToTypeID("Chnl"), charIDToTypeID("fsel"));
    desc.putReference(charIDToTypeID("null"), ref);
    var ellipseDesc = new ActionDescriptor();
    ellipseDesc.putUnitDouble(charIDToTypeID("Top "), charIDToTypeID("#Pxl"), y);
    ellipseDesc.putUnitDouble(charIDToTypeID("Left"), charIDToTypeID("#Pxl"), x);
    ellipseDesc.putUnitDouble(charIDToTypeID("Btom"), charIDToTypeID("#Pxl"), y + height);
    ellipseDesc.putUnitDouble(charIDToTypeID("Rght"), charIDToTypeID("#Pxl"), x + width);
    desc.putObject(charIDToTypeID("T   "), charIDToTypeID("Elps"), ellipseDesc);
    executeAction(charIDToTypeID("setd"), desc, DialogModes.NO);
}

// Feather selection
doc.selection.feather(UnitValue(5, 'px'));

// Expand/contract selection
doc.selection.expand(UnitValue(10, 'px'));
doc.selection.contract(UnitValue(5, 'px'));

// Invert selection
doc.selection.invert();

// Fill selection
var fillColor = new SolidColor();
fillColor.rgb.red = 255;
fillColor.rgb.green = 0;
fillColor.rgb.blue = 0;
doc.selection.fill(fillColor, ColorBlendMode.NORMAL, 100, false);

// Stroke selection
doc.selection.stroke(fillColor, 2, StrokeLocation.INSIDE);

// ═══════════════════════════════════════════════════════════════
// COLOR ADJUSTMENTS
// ═══════════════════════════════════════════════════════════════

// Brightness/Contrast
doc.activeLayer.adjustBrightnessContrast(25, 15);

// Levels
// Parameters: inputRangeStart, inputRangeEnd, inputRangeGamma,
//             outputRangeStart, outputRangeEnd
doc.activeLayer.adjustLevels(0, 255, 1.0, 0, 255);

// Curves (via action descriptor)
function adjustCurves(points) {
    var desc = new ActionDescriptor();
    var adjDesc = new ActionDescriptor();
    var curveList = new ActionList();

    var curveDesc = new ActionDescriptor();
    var channelRef = new ActionReference();
    channelRef.putEnumerated(charIDToTypeID("Chnl"), charIDToTypeID("Chnl"), charIDToTypeID("Cmps"));
    curveDesc.putReference(charIDToTypeID("Chnl"), channelRef);

    var pointList = new ActionList();
    for (var i = 0; i < points.length; i++) {
        var pointDesc = new ActionDescriptor();
        pointDesc.putDouble(charIDToTypeID("Hrzn"), points[i][0]);
        pointDesc.putDouble(charIDToTypeID("Vrtc"), points[i][1]);
        pointList.putObject(charIDToTypeID("Pnt "), pointDesc);
    }
    curveDesc.putList(charIDToTypeID("Crv "), pointList);
    curveList.putObject(charIDToTypeID("CrvA"), curveDesc);
    adjDesc.putList(charIDToTypeID("AdjL"), curveList);
    desc.putObject(charIDToTypeID("Usng"), charIDToTypeID("Crvs"), adjDesc);

    executeAction(charIDToTypeID("Crvs"), desc, DialogModes.NO);
}

// Hue/Saturation
doc.activeLayer.adjustHueSaturation(0, 0, 25);  // hue, saturation, lightness

// Color Balance
// (requires action descriptor for full control)

// Invert
doc.activeLayer.invert();

// Desaturate
doc.activeLayer.desaturate();

// ═══════════════════════════════════════════════════════════════
// FILTERS
// ═══════════════════════════════════════════════════════════════

// Gaussian Blur
doc.activeLayer.applyGaussianBlur(5.0);  // radius in pixels

// Unsharp Mask
doc.activeLayer.applyUnSharpMask(100, 1.5, 0);  // amount%, radius, threshold

// Motion Blur
doc.activeLayer.applyMotionBlur(45, 20);  // angle, distance

// Radial Blur
doc.activeLayer.applyRadialBlur(10, RadialBlurMethod.SPIN, RadialBlurQuality.GOOD);

// Add Noise
doc.activeLayer.applyAddNoise(10, NoiseDistribution.GAUSSIAN, false);

// Median
doc.activeLayer.applyMedianNoise(3);

// High Pass
doc.activeLayer.applyHighPass(3.0);

// ═══════════════════════════════════════════════════════════════
// TRANSFORMS
// ═══════════════════════════════════════════════════════════════

// Resize canvas
doc.resizeCanvas(UnitValue(2000, 'px'), UnitValue(1500, 'px'), AnchorPosition.MIDDLECENTER);

// Resize image
doc.resizeImage(UnitValue(1920, 'px'), UnitValue(1080, 'px'), 300, ResampleMethod.BICUBIC);

// Rotate
doc.activeLayer.rotate(45, AnchorPosition.MIDDLECENTER);

// Scale
doc.activeLayer.resize(150, 150, AnchorPosition.MIDDLECENTER);

// Flip
doc.activeLayer.flipHorizontal();
doc.activeLayer.flipVertical();

// Free Transform (via action)
function freeTransform(scaleX, scaleY, rotation, translateX, translateY) {
    var desc = new ActionDescriptor();
    var ref = new ActionReference();
    ref.putEnumerated(charIDToTypeID("Lyr "), charIDToTypeID("Ordn"), charIDToTypeID("Trgt"));
    desc.putReference(charIDToTypeID("null"), ref);
    desc.putEnumerated(charIDToTypeID("FTcs"), charIDToTypeID("QCSt"), charIDToTypeID("Qcsa"));

    var transformDesc = new ActionDescriptor();
    transformDesc.putUnitDouble(charIDToTypeID("Hrzn"), charIDToTypeID("#Pxl"), translateX);
    transformDesc.putUnitDouble(charIDToTypeID("Vrtc"), charIDToTypeID("#Pxl"), translateY);
    desc.putObject(charIDToTypeID("Ofst"), charIDToTypeID("Ofst"), transformDesc);

    desc.putUnitDouble(charIDToTypeID("Wdth"), charIDToTypeID("#Prc"), scaleX);
    desc.putUnitDouble(charIDToTypeID("Hght"), charIDToTypeID("#Prc"), scaleY);
    desc.putUnitDouble(charIDToTypeID("Angl"), charIDToTypeID("#Ang"), rotation);

    executeAction(charIDToTypeID("Trnf"), desc, DialogModes.NO);
}

// ═══════════════════════════════════════════════════════════════
// BATCH PROCESSING
// ═══════════════════════════════════════════════════════════════

function batchProcess(inputFolder, outputFolder, processFunction) {
    var folder = new Folder(inputFolder);
    var files = folder.getFiles(/\.(jpg|jpeg|png|tif|tiff|psd)$/i);

    for (var i = 0; i < files.length; i++) {
        var file = files[i];

        // Open file
        var doc = app.open(file);

        // Process
        processFunction(doc);

        // Save
        var outputFile = new File(outputFolder + "/" + file.name);
        var saveOptions = new JPEGSaveOptions();
        saveOptions.quality = 10;
        doc.saveAs(outputFile, saveOptions, true);

        // Close
        doc.close(SaveOptions.DONOTSAVECHANGES);
    }
}

// Example usage:
// batchProcess("/input", "/output", function(doc) {
//     doc.activeLayer.adjustBrightnessContrast(10, 5);
//     doc.activeLayer.applyUnSharpMask(50, 1.0, 0);
// });

// ═══════════════════════════════════════════════════════════════
// SMART OBJECTS
// ═══════════════════════════════════════════════════════════════

// Convert to Smart Object
function convertToSmartObject() {
    var desc = new ActionDescriptor();
    executeAction(stringIDToTypeID("newPlacedLayer"), desc, DialogModes.NO);
}

// Edit Smart Object contents
function editSmartObjectContents() {
    var desc = new ActionDescriptor();
    executeAction(stringIDToTypeID("placedLayerEditContents"), desc, DialogModes.NO);
}

// ═══════════════════════════════════════════════════════════════
// LAYER STYLES (EFFECTS)
// ═══════════════════════════════════════════════════════════════

function addDropShadow(angle, distance, spread, size, opacity) {
    var desc = new ActionDescriptor();
    var styleDesc = new ActionDescriptor();
    var shadowDesc = new ActionDescriptor();

    shadowDesc.putBoolean(charIDToTypeID("enab"), true);
    shadowDesc.putEnumerated(charIDToTypeID("Md  "), charIDToTypeID("BlnM"), charIDToTypeID("Mltp"));

    var colorDesc = new ActionDescriptor();
    colorDesc.putDouble(charIDToTypeID("Rd  "), 0);
    colorDesc.putDouble(charIDToTypeID("Grn "), 0);
    colorDesc.putDouble(charIDToTypeID("Bl  "), 0);
    shadowDesc.putObject(charIDToTypeID("Clr "), charIDToTypeID("RGBC"), colorDesc);

    shadowDesc.putUnitDouble(charIDToTypeID("Opct"), charIDToTypeID("#Prc"), opacity);
    shadowDesc.putBoolean(charIDToTypeID("uglg"), true);
    shadowDesc.putUnitDouble(charIDToTypeID("lagl"), charIDToTypeID("#Ang"), angle);
    shadowDesc.putUnitDouble(charIDToTypeID("Dstn"), charIDToTypeID("#Pxl"), distance);
    shadowDesc.putUnitDouble(charIDToTypeID("Ckmt"), charIDToTypeID("#Prc"), spread);
    shadowDesc.putUnitDouble(charIDToTypeID("blur"), charIDToTypeID("#Pxl"), size);

    styleDesc.putObject(charIDToTypeID("DrSh"), charIDToTypeID("DrSh"), shadowDesc);
    desc.putObject(charIDToTypeID("T   "), charIDToTypeID("Lefx"), styleDesc);

    var ref = new ActionReference();
    ref.putProperty(charIDToTypeID("Prpr"), charIDToTypeID("Lefx"));
    ref.putEnumerated(charIDToTypeID("Lyr "), charIDToTypeID("Ordn"), charIDToTypeID("Trgt"));
    desc.putReference(charIDToTypeID("null"), ref);

    executeAction(charIDToTypeID("setd"), desc, DialogModes.NO);
}
```



---

## Chapter 7: Python PIL/Pillow Deep Dive

### 7.1 Pillow Architecture and Internals

```python
"""
PIL/PILLOW COMPLETE REFERENCE

Pillow is the friendly PIL (Python Imaging Library) fork.
It's the most widely used Python imaging library.

ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────┐
│                         PILLOW                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                    HIGH-LEVEL API                         │ │
│  │  Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps    │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────────┐ │
│  │                    IMAGE CORE                             │ │
│  │  - Pixel storage and access                               │ │
│  │  - Mode conversions                                       │ │
│  │  - Memory management                                      │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────────┐ │
│  │                    C EXTENSIONS                           │ │
│  │  _imaging.c, _imagingft.c, _imagingmath.c, etc.          │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────────┐ │
│  │                    EXTERNAL LIBRARIES                     │ │
│  │  libjpeg, libpng, libtiff, zlib, freetype, etc.          │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
from PIL import ImageFont, ImageColor, ImageChops, ImageStat, ImageMath
import numpy as np

# ═══════════════════════════════════════════════════════════════
# IMAGE CREATION AND LOADING
# ═══════════════════════════════════════════════════════════════

# Create new image
img = Image.new('RGB', (800, 600), color='black')
img = Image.new('RGBA', (800, 600), color=(255, 0, 0, 128))
img = Image.new('L', (800, 600), color=128)

# Load image
img = Image.open('image.jpg')
img = Image.open('image.png')

# Load from URL (with requests)
import requests
from io import BytesIO
response = requests.get('https://example.com/image.jpg')
img = Image.open(BytesIO(response.content))

# Load from bytes
img = Image.open(BytesIO(image_bytes))

# Load from numpy array
arr = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
img = Image.fromarray(arr)

# Create from raw data
img = Image.frombytes('RGB', (800, 600), raw_bytes)
img = Image.frombuffer('RGB', (800, 600), buffer_data)

# ═══════════════════════════════════════════════════════════════
# IMAGE PROPERTIES AND INFO
# ═══════════════════════════════════════════════════════════════

# Basic properties
width, height = img.size
mode = img.mode  # 'RGB', 'RGBA', 'L', etc.
format = img.format  # 'JPEG', 'PNG', etc.
info = img.info  # Metadata dictionary

# Pixel access
pixel = img.getpixel((x, y))
img.putpixel((x, y), (255, 0, 0))

# Get all pixels as flat sequence
pixels = list(img.getdata())

# Set all pixels
img.putdata(new_pixels)

# Get image bands (channels)
bands = img.split()  # Returns tuple of Images
r, g, b = img.split()

# Merge bands
img = Image.merge('RGB', (r, g, b))

# Get bounding box of non-zero regions
bbox = img.getbbox()  # (left, upper, right, lower) or None

# Get color extremes
extrema = img.getextrema()  # ((min_r, max_r), (min_g, max_g), ...)

# Get histogram
histogram = img.histogram()  # List of 256 values per channel

# Convert to numpy array
arr = np.array(img)
arr = np.asarray(img)  # Read-only view

# ═══════════════════════════════════════════════════════════════
# MODE CONVERSIONS
# ═══════════════════════════════════════════════════════════════

# Convert modes
rgb = img.convert('RGB')
rgba = img.convert('RGBA')
gray = img.convert('L')
binary = img.convert('1')
palette = img.convert('P')  # 256-color palette
cmyk = img.convert('CMYK')
lab = img.convert('LAB')
hsv = img.convert('HSV')

# Convert with custom matrix
# Matrix is 4 or 12 floats for RGB→RGB or RGB→RGBA
matrix = (
    0.393, 0.769, 0.189, 0,  # R
    0.349, 0.686, 0.168, 0,  # G
    0.272, 0.534, 0.131, 0   # B
)  # Sepia effect
sepia = img.convert('RGB', matrix)

# Dithering for palette conversion
dithered = img.convert('P', palette=Image.ADAPTIVE, colors=16)

# ═══════════════════════════════════════════════════════════════
# GEOMETRIC OPERATIONS
# ═══════════════════════════════════════════════════════════════

# Resize
resized = img.resize((400, 300))
resized = img.resize((400, 300), resample=Image.LANCZOS)
resized = img.resize((400, 300), resample=Image.BILINEAR)
resized = img.resize((400, 300), resample=Image.BICUBIC)
resized = img.resize((400, 300), resample=Image.NEAREST)

# Thumbnail (maintains aspect ratio, modifies in place)
img.thumbnail((400, 300))

# Rotate
rotated = img.rotate(45)  # Degrees counter-clockwise
rotated = img.rotate(45, expand=True)  # Expand canvas to fit
rotated = img.rotate(45, center=(100, 100))  # Custom center
rotated = img.rotate(45, fillcolor='white')  # Fill background
rotated = img.rotate(45, resample=Image.BICUBIC)

# Transpose (flip/rotate 90°)
flipped_h = img.transpose(Image.FLIP_LEFT_RIGHT)
flipped_v = img.transpose(Image.FLIP_TOP_BOTTOM)
rotated_90 = img.transpose(Image.ROTATE_90)
rotated_180 = img.transpose(Image.ROTATE_180)
rotated_270 = img.transpose(Image.ROTATE_270)
transposed = img.transpose(Image.TRANSPOSE)

# Crop
cropped = img.crop((left, upper, right, lower))

# Paste
img.paste(other_img, (x, y))
img.paste(other_img, (x, y), mask=mask_img)  # With alpha mask
img.paste(color, box)  # Fill region with color

# Transform (arbitrary affine/perspective)
# data is 6 floats for AFFINE or 8 floats for PERSPECTIVE
transformed = img.transform(
    (width, height),
    Image.AFFINE,
    data=(scale, shear, translate_x, 0, scale, translate_y)
)

transformed = img.transform(
    (width, height),
    Image.PERSPECTIVE,
    data=coefficients  # 8 coefficients
)

transformed = img.transform(
    (width, height),
    Image.QUAD,
    data=(x0, y0, x1, y1, x2, y2, x3, y3)  # Four corners
)

# ═══════════════════════════════════════════════════════════════
# IMAGEDRAW - DRAWING PRIMITIVES
# ═══════════════════════════════════════════════════════════════

from PIL import ImageDraw, ImageFont

# Create draw object
draw = ImageDraw.Draw(img)
draw = ImageDraw.Draw(img, 'RGBA')  # For alpha drawing

# Line
draw.line([(0, 0), (100, 100)], fill='white', width=2)
draw.line([0, 0, 100, 100, 200, 50], fill='red', width=1)  # Polyline

# Rectangle
draw.rectangle([10, 10, 100, 100], fill='blue', outline='white', width=2)
draw.rectangle([(10, 10), (100, 100)], fill='blue')

# Rounded rectangle
draw.rounded_rectangle([10, 10, 100, 100], radius=10, fill='green')

# Ellipse
draw.ellipse([10, 10, 100, 100], fill='yellow', outline='black', width=2)

# Arc
draw.arc([10, 10, 100, 100], start=0, end=180, fill='white', width=2)

# Chord (arc with straight line connecting endpoints)
draw.chord([10, 10, 100, 100], start=0, end=180, fill='red', outline='white')

# Pie slice
draw.pieslice([10, 10, 100, 100], start=0, end=90, fill='orange', outline='black')

# Polygon
draw.polygon([(50, 10), (90, 90), (10, 90)], fill='purple', outline='white')

# Point
draw.point([(10, 10), (20, 20), (30, 30)], fill='white')

# Text
draw.text((10, 10), "Hello", fill='white')

# Text with font
font = ImageFont.truetype('arial.ttf', size=24)
draw.text((10, 10), "Hello", font=font, fill='white')

# Multiline text
draw.multiline_text((10, 10), "Line 1\nLine 2", font=font, fill='white', spacing=4)

# Get text size
bbox = draw.textbbox((0, 0), "Hello", font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]

# Bitmap font (built-in)
font = ImageFont.load_default()

# ═══════════════════════════════════════════════════════════════
# IMAGEFILTER - BUILT-IN FILTERS
# ═══════════════════════════════════════════════════════════════

from PIL import ImageFilter

# Blur filters
blurred = img.filter(ImageFilter.BLUR)
blurred = img.filter(ImageFilter.GaussianBlur(radius=5))
blurred = img.filter(ImageFilter.BoxBlur(radius=5))

# Sharpen
sharpened = img.filter(ImageFilter.SHARPEN)
sharpened = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

# Edge detection
edges = img.filter(ImageFilter.FIND_EDGES)
edges = img.filter(ImageFilter.EDGE_ENHANCE)
edges = img.filter(ImageFilter.EDGE_ENHANCE_MORE)

# Contour
contour = img.filter(ImageFilter.CONTOUR)

# Emboss
embossed = img.filter(ImageFilter.EMBOSS)

# Smooth
smoothed = img.filter(ImageFilter.SMOOTH)
smoothed = img.filter(ImageFilter.SMOOTH_MORE)

# Detail
detailed = img.filter(ImageFilter.DETAIL)

# Median filter (noise reduction)
median = img.filter(ImageFilter.MedianFilter(size=3))

# Min/Max filters
minimum = img.filter(ImageFilter.MinFilter(size=3))
maximum = img.filter(ImageFilter.MaxFilter(size=3))

# Mode filter
mode_filtered = img.filter(ImageFilter.ModeFilter(size=3))

# Rank filter
rank = img.filter(ImageFilter.RankFilter(size=3, rank=5))

# Custom kernel
kernel = ImageFilter.Kernel(
    size=(3, 3),
    kernel=[
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0
    ],
    scale=1,
    offset=0
)
custom_filtered = img.filter(kernel)

# ═══════════════════════════════════════════════════════════════
# IMAGEENHANCE - ADJUSTMENT OPERATIONS
# ═══════════════════════════════════════════════════════════════

from PIL import ImageEnhance

# Brightness
enhancer = ImageEnhance.Brightness(img)
brightened = enhancer.enhance(1.5)  # 1.0 = original, >1 = brighter

# Contrast
enhancer = ImageEnhance.Contrast(img)
contrasted = enhancer.enhance(1.5)  # 1.0 = original, >1 = more contrast

# Color (saturation)
enhancer = ImageEnhance.Color(img)
saturated = enhancer.enhance(1.5)  # 0 = grayscale, 1.0 = original

# Sharpness
enhancer = ImageEnhance.Sharpness(img)
sharpened = enhancer.enhance(2.0)  # 0 = blurred, 1.0 = original

# ═══════════════════════════════════════════════════════════════
# IMAGEOPS - OPERATIONS AND EFFECTS
# ═══════════════════════════════════════════════════════════════

from PIL import ImageOps

# Auto-adjust
equalized = ImageOps.equalize(img)  # Histogram equalization
autocontrast = ImageOps.autocontrast(img, cutoff=0)  # Auto levels

# Color operations
grayscale = ImageOps.grayscale(img)
inverted = ImageOps.invert(img)
posterized = ImageOps.posterize(img, bits=4)  # Reduce color depth
solarized = ImageOps.solarize(img, threshold=128)  # Solarize effect

# Padding/cropping
padded = ImageOps.pad(img, (800, 600), color='black')  # Pad to fit
fitted = ImageOps.fit(img, (800, 600))  # Crop to fit
contained = ImageOps.contain(img, (800, 600))  # Scale to fit

# Mirror
mirrored = ImageOps.mirror(img)  # Horizontal flip
flipped = ImageOps.flip(img)  # Vertical flip

# Expand (add border)
bordered = ImageOps.expand(img, border=10, fill='black')
bordered = ImageOps.expand(img, border=(10, 20, 30, 40), fill='white')  # (left, top, right, bottom)

# Crop to content
cropped = ImageOps.crop(img, border=0)  # Remove border of 'border' pixels

# Colorize grayscale
colorized = ImageOps.colorize(gray_img, black='blue', white='yellow')
colorized = ImageOps.colorize(gray_img, black='blue', white='yellow', mid='green')

# Deform
def deform_func(xy):
    x, y = xy
    return (x + 10, y)  # Shift right by 10
deformed = ImageOps.deform(img, deform_func)

# Exif orientation
oriented = ImageOps.exif_transpose(img)  # Apply EXIF rotation

# ═══════════════════════════════════════════════════════════════
# IMAGECHOPS - CHANNEL OPERATIONS
# ═══════════════════════════════════════════════════════════════

from PIL import ImageChops

# Arithmetic operations
added = ImageChops.add(img1, img2)  # Clipped to 255
added = ImageChops.add(img1, img2, scale=2, offset=0)  # (img1+img2)/scale + offset
added_modulo = ImageChops.add_modulo(img1, img2)  # Wraps at 255

subtracted = ImageChops.subtract(img1, img2)
subtracted = ImageChops.subtract(img1, img2, scale=1, offset=128)
subtracted_modulo = ImageChops.subtract_modulo(img1, img2)

multiplied = ImageChops.multiply(img1, img2)  # Darken blend

# Screen blend (lighten)
screened = ImageChops.screen(img1, img2)

# Compositing
darker = ImageChops.darker(img1, img2)  # min(img1, img2)
lighter = ImageChops.lighter(img1, img2)  # max(img1, img2)

# Difference and comparison
difference = ImageChops.difference(img1, img2)  # abs(img1 - img2)
is_different = ImageChops.difference(img1, img2).getbbox() is not None

# Invert
inverted = ImageChops.invert(img)

# Blend (linear interpolation)
blended = ImageChops.blend(img1, img2, alpha=0.5)  # 0=img1, 1=img2

# Composite (with mask)
composited = ImageChops.composite(img1, img2, mask)

# Offset
offset = ImageChops.offset(img, xoffset=10, yoffset=20)

# Logical operations (for binary images)
logical_and = ImageChops.logical_and(img1, img2)
logical_or = ImageChops.logical_or(img1, img2)
logical_xor = ImageChops.logical_xor(img1, img2)

# Constant operations
constant = ImageChops.constant(img, 128)  # Create constant image

# ═══════════════════════════════════════════════════════════════
# IMAGEMATH - PIXEL-LEVEL MATH
# ═══════════════════════════════════════════════════════════════

from PIL import ImageMath

# Evaluate expression on images
# Images must be same size, result is 32-bit integer mode
result = ImageMath.eval("a + b", a=img1.convert('L'), b=img2.convert('L'))
result = result.convert('L')

# Supported operations:
# Arithmetic: +, -, *, /, %, **
# Bitwise: &, |, ^, ~, <<, >>
# Comparison: <, <=, ==, !=, >=, >
# Functions: abs, min, max, int, float

# Examples
inverted = ImageMath.eval("255 - a", a=img.convert('L'))
thresholded = ImageMath.eval("(a > 128) * 255", a=img.convert('L'))
blended = ImageMath.eval("(a * 0.5 + b * 0.5)", a=img1, b=img2)
gamma = ImageMath.eval("int(((a / 255.0) ** 2.2) * 255)", a=img.convert('L'))

# Convert result to usable format
result_img = result.convert('L')

# ═══════════════════════════════════════════════════════════════
# SAVING AND EXPORTING
# ═══════════════════════════════════════════════════════════════

# Save as JPEG
img.save('output.jpg', 'JPEG', quality=85, optimize=True, progressive=True)

# Save as PNG
img.save('output.png', 'PNG', compress_level=9, optimize=True)

# Save as WebP
img.save('output.webp', 'WEBP', quality=80, method=6)
img.save('output.webp', 'WEBP', lossless=True)

# Save as GIF
img.save('output.gif', 'GIF')

# Save animated GIF
frames[0].save(
    'animation.gif',
    save_all=True,
    append_images=frames[1:],
    duration=100,  # ms per frame
    loop=0  # 0 = infinite loop
)

# Save as TIFF
img.save('output.tiff', 'TIFF', compression='lzw')

# Save as BMP
img.save('output.bmp', 'BMP')

# Save as ICO
img.save('output.ico', 'ICO', sizes=[(16, 16), (32, 32), (48, 48)])

# Save to bytes
from io import BytesIO
buffer = BytesIO()
img.save(buffer, format='PNG')
image_bytes = buffer.getvalue()


### 7.2 Custom Filters and Effects with Pillow

```python
"""
ADVANCED PILLOW TECHNIQUES - CUSTOM FILTERS AND EFFECTS
"""

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import math

# ═══════════════════════════════════════════════════════════════
# CUSTOM CONVOLUTION KERNELS
# ═══════════════════════════════════════════════════════════════

class CustomKernel:
    """Collection of custom convolution kernels"""

    @staticmethod
    def identity():
        """No change"""
        return ImageFilter.Kernel(
            size=(3, 3),
            kernel=[0, 0, 0, 0, 1, 0, 0, 0, 0],
            scale=1
        )

    @staticmethod
    def sharpen_strong():
        """Strong sharpening"""
        return ImageFilter.Kernel(
            size=(3, 3),
            kernel=[
                -1, -1, -1,
                -1,  9, -1,
                -1, -1, -1
            ],
            scale=1
        )

    @staticmethod
    def sobel_x():
        """Sobel horizontal edge detection"""
        return ImageFilter.Kernel(
            size=(3, 3),
            kernel=[
                -1, 0, 1,
                -2, 0, 2,
                -1, 0, 1
            ],
            scale=1,
            offset=128
        )

    @staticmethod
    def sobel_y():
        """Sobel vertical edge detection"""
        return ImageFilter.Kernel(
            size=(3, 3),
            kernel=[
                -1, -2, -1,
                 0,  0,  0,
                 1,  2,  1
            ],
            scale=1,
            offset=128
        )

    @staticmethod
    def laplacian():
        """Laplacian edge detection"""
        return ImageFilter.Kernel(
            size=(3, 3),
            kernel=[
                0,  1, 0,
                1, -4, 1,
                0,  1, 0
            ],
            scale=1,
            offset=128
        )

    @staticmethod
    def emboss_custom(angle=135):
        """Emboss at custom angle"""
        rad = math.radians(angle)
        x = math.cos(rad)
        y = math.sin(rad)

        return ImageFilter.Kernel(
            size=(3, 3),
            kernel=[
                -y-x,  -y, -y+x,
                  -x,   0,    x,
                 y-x,   y,  y+x
            ],
            scale=1,
            offset=128
        )

    @staticmethod
    def motion_blur(angle=0, length=9):
        """Motion blur kernel"""
        rad = math.radians(angle)
        kernel = [0] * (length * length)
        center = length // 2

        for i in range(length):
            offset = i - center
            x = int(center + offset * math.cos(rad))
            y = int(center + offset * math.sin(rad))
            if 0 <= x < length and 0 <= y < length:
                kernel[y * length + x] = 1

        return ImageFilter.Kernel(
            size=(length, length),
            kernel=kernel,
            scale=sum(kernel) or 1
        )

    @staticmethod
    def gaussian(size=5, sigma=1.0):
        """Gaussian blur kernel"""
        kernel = []
        center = size // 2

        for y in range(size):
            for x in range(size):
                dx = x - center
                dy = y - center
                g = math.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma))
                kernel.append(g)

        total = sum(kernel)
        kernel = [k / total for k in kernel]

        return ImageFilter.Kernel(
            size=(size, size),
            kernel=kernel,
            scale=1
        )


# ═══════════════════════════════════════════════════════════════
# NUMPY-BASED IMAGE PROCESSING
# ═══════════════════════════════════════════════════════════════

class NumpyImageOps:
    """Advanced image operations using NumPy"""

    @staticmethod
    def to_array(img):
        """Convert PIL Image to numpy array"""
        return np.array(img, dtype=np.float32) / 255.0

    @staticmethod
    def from_array(arr):
        """Convert numpy array to PIL Image"""
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def gamma_correct(img, gamma):
        """Apply gamma correction"""
        arr = NumpyImageOps.to_array(img)
        arr = np.power(arr, gamma)
        return NumpyImageOps.from_array(arr)

    @staticmethod
    def levels(img, in_black=0, in_white=255, gamma=1.0, out_black=0, out_white=255):
        """Photoshop-style levels adjustment"""
        arr = np.array(img, dtype=np.float32)

        # Input levels
        arr = (arr - in_black) / (in_white - in_black)
        arr = np.clip(arr, 0, 1)

        # Gamma
        arr = np.power(arr, gamma)

        # Output levels
        arr = arr * (out_white - out_black) + out_black
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        return Image.fromarray(arr)

    @staticmethod
    def curves(img, curve_points):
        """Apply curves adjustment using interpolation"""
        from scipy import interpolate

        # Create lookup table from curve points
        points = np.array(curve_points)
        x = points[:, 0]
        y = points[:, 1]

        # Interpolate
        f = interpolate.interp1d(x, y, kind='cubic', fill_value='extrapolate')
        lut = np.clip(f(np.arange(256)), 0, 255).astype(np.uint8)

        # Apply LUT
        arr = np.array(img)
        if len(arr.shape) == 3:
            for i in range(arr.shape[2]):
                arr[:, :, i] = lut[arr[:, :, i]]
        else:
            arr = lut[arr]

        return Image.fromarray(arr)

    @staticmethod
    def color_matrix(img, matrix):
        """Apply 3x3 color transformation matrix"""
        arr = NumpyImageOps.to_array(img.convert('RGB'))

        matrix = np.array(matrix).reshape(3, 3)

        # Reshape for matrix multiplication
        flat = arr.reshape(-1, 3)
        transformed = np.dot(flat, matrix.T)
        transformed = transformed.reshape(arr.shape)

        return NumpyImageOps.from_array(transformed)

    @staticmethod
    def channel_mixer(img, r_out=(1, 0, 0), g_out=(0, 1, 0), b_out=(0, 0, 1)):
        """Channel mixer"""
        matrix = [
            r_out[0], r_out[1], r_out[2],
            g_out[0], g_out[1], g_out[2],
            b_out[0], b_out[1], b_out[2]
        ]
        return NumpyImageOps.color_matrix(img, matrix)

    @staticmethod
    def hue_rotate(img, degrees):
        """Rotate hue by specified degrees"""
        arr = np.array(img.convert('HSV'))
        arr[:, :, 0] = (arr[:, :, 0].astype(int) + int(degrees * 255 / 360)) % 256
        return Image.fromarray(arr, 'HSV').convert('RGB')

    @staticmethod
    def selective_color(img, hue_range, saturation_adjust=0, lightness_adjust=0):
        """Adjust colors in specific hue range"""
        hsv = np.array(img.convert('HSV'), dtype=np.float32)

        # Create mask for hue range
        h = hsv[:, :, 0]
        hue_low, hue_high = hue_range
        if hue_low <= hue_high:
            mask = (h >= hue_low) & (h <= hue_high)
        else:  # Range wraps around (e.g., red)
            mask = (h >= hue_low) | (h <= hue_high)

        # Adjust saturation and value in masked area
        hsv[:, :, 1] = np.where(mask,
                                np.clip(hsv[:, :, 1] + saturation_adjust, 0, 255),
                                hsv[:, :, 1])
        hsv[:, :, 2] = np.where(mask,
                                np.clip(hsv[:, :, 2] + lightness_adjust, 0, 255),
                                hsv[:, :, 2])

        return Image.fromarray(hsv.astype(np.uint8), 'HSV').convert('RGB')

    @staticmethod
    def vignette(img, strength=0.5, radius=1.0):
        """Add vignette effect"""
        arr = NumpyImageOps.to_array(img)
        h, w = arr.shape[:2]

        # Create distance map from center
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2) * radius

        # Create vignette mask
        vignette = 1 - strength * (dist / max_dist) ** 2
        vignette = np.clip(vignette, 0, 1)

        # Apply to image
        if len(arr.shape) == 3:
            vignette = vignette[:, :, np.newaxis]

        arr = arr * vignette
        return NumpyImageOps.from_array(arr)

    @staticmethod
    def chromatic_aberration(img, offset=5):
        """Simulate chromatic aberration"""
        arr = np.array(img)
        h, w = arr.shape[:2]

        result = np.zeros_like(arr)

        # Offset red channel outward
        result[:, :, 0] = np.roll(arr[:, :, 0], offset, axis=1)

        # Keep green centered
        result[:, :, 1] = arr[:, :, 1]

        # Offset blue channel inward
        result[:, :, 2] = np.roll(arr[:, :, 2], -offset, axis=1)

        return Image.fromarray(result)

    @staticmethod
    def duotone(img, dark_color, light_color):
        """Convert to duotone"""
        gray = np.array(img.convert('L'), dtype=np.float32) / 255.0

        dark = np.array(dark_color) / 255.0
        light = np.array(light_color) / 255.0

        result = np.zeros((*gray.shape, 3))
        for i in range(3):
            result[:, :, i] = dark[i] + (light[i] - dark[i]) * gray

        return NumpyImageOps.from_array(result)

    @staticmethod
    def threshold_adaptive(img, block_size=11, c=2):
        """Adaptive thresholding"""
        from scipy.ndimage import uniform_filter

        gray = np.array(img.convert('L'), dtype=np.float32)

        # Local mean
        local_mean = uniform_filter(gray, size=block_size)

        # Threshold
        binary = (gray > local_mean - c).astype(np.uint8) * 255

        return Image.fromarray(binary, 'L')


# ═══════════════════════════════════════════════════════════════
# DISPLACEMENT AND DISTORTION
# ═══════════════════════════════════════════════════════════════

class DistortionEffects:
    """Image distortion effects"""

    @staticmethod
    def displace(img, displacement_map, scale=10):
        """Displace pixels based on displacement map"""
        arr = np.array(img)
        disp = np.array(displacement_map.convert('L'), dtype=np.float32)

        # Normalize displacement to -1 to 1
        disp = (disp - 128) / 128 * scale

        h, w = arr.shape[:2]
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Calculate new coordinates
        new_x = np.clip(x + disp, 0, w - 1).astype(int)
        new_y = np.clip(y + disp, 0, h - 1).astype(int)

        result = arr[new_y, new_x]
        return Image.fromarray(result)

    @staticmethod
    def polar_coordinates(img, inverse=False):
        """Convert to/from polar coordinates"""
        arr = np.array(img)
        h, w = arr.shape[:2]
        cx, cy = w / 2, h / 2
        max_r = min(cx, cy)

        result = np.zeros_like(arr)
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        if not inverse:
            # Cartesian to polar
            dx, dy = x - cx, y - cy
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)

            src_x = ((theta + np.pi) / (2 * np.pi) * w).astype(int) % w
            src_y = (r / max_r * h).astype(int)
            src_y = np.clip(src_y, 0, h - 1)
        else:
            # Polar to Cartesian
            r = y / h * max_r
            theta = (x / w) * 2 * np.pi - np.pi

            src_x = (cx + r * np.cos(theta)).astype(int)
            src_y = (cy + r * np.sin(theta)).astype(int)
            src_x = np.clip(src_x, 0, w - 1)
            src_y = np.clip(src_y, 0, h - 1)

        result = arr[src_y, src_x]
        return Image.fromarray(result)

    @staticmethod
    def spherize(img, strength=1.0):
        """Spherize distortion"""
        arr = np.array(img)
        h, w = arr.shape[:2]
        cx, cy = w / 2, h / 2
        max_r = min(cx, cy)

        result = np.zeros_like(arr)
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        dx, dy = (x - cx) / max_r, (y - cy) / max_r
        r = np.sqrt(dx**2 + dy**2)

        # Spherize transformation
        mask = r < 1
        factor = np.where(mask,
                         np.sin(r * np.pi / 2) ** strength / (r + 0.0001),
                         1)

        src_x = (cx + dx * factor * max_r).astype(int)
        src_y = (cy + dy * factor * max_r).astype(int)
        src_x = np.clip(src_x, 0, w - 1)
        src_y = np.clip(src_y, 0, h - 1)

        result = arr[src_y, src_x]
        return Image.fromarray(result)

    @staticmethod
    def wave(img, amplitude=10, wavelength=50, direction='horizontal'):
        """Wave distortion"""
        arr = np.array(img)
        h, w = arr.shape[:2]

        result = np.zeros_like(arr)
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        if direction == 'horizontal':
            offset = amplitude * np.sin(2 * np.pi * y / wavelength)
            src_x = (x + offset).astype(int) % w
            src_y = y
        else:
            offset = amplitude * np.sin(2 * np.pi * x / wavelength)
            src_x = x
            src_y = (y + offset).astype(int) % h

        result = arr[src_y, src_x]
        return Image.fromarray(result)

    @staticmethod
    def twirl(img, strength=1.0, radius=None):
        """Twirl/swirl distortion"""
        arr = np.array(img)
        h, w = arr.shape[:2]
        cx, cy = w / 2, h / 2
        if radius is None:
            radius = min(cx, cy)

        result = np.zeros_like(arr)
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        dx, dy = x - cx, y - cy
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)

        # Twirl amount decreases with distance
        twirl_amount = strength * (1 - r / radius)
        twirl_amount = np.where(r < radius, twirl_amount, 0)

        new_theta = theta + twirl_amount

        src_x = (cx + r * np.cos(new_theta)).astype(int)
        src_y = (cy + r * np.sin(new_theta)).astype(int)
        src_x = np.clip(src_x, 0, w - 1)
        src_y = np.clip(src_y, 0, h - 1)

        result = arr[src_y, src_x]
        return Image.fromarray(result)

    @staticmethod
    def pixelate(img, block_size=10):
        """Pixelate effect"""
        w, h = img.size
        small = img.resize((w // block_size, h // block_size), Image.NEAREST)
        return small.resize((w, h), Image.NEAREST)

    @staticmethod
    def oil_paint(img, radius=3, levels=8):
        """Oil paint effect"""
        arr = np.array(img.convert('RGB'))
        gray = np.array(img.convert('L'))
        h, w = arr.shape[:2]

        # Quantize intensity
        quantized = (gray // (256 // levels)) * (256 // levels)

        result = np.zeros_like(arr)

        for y in range(radius, h - radius):
            for x in range(radius, w - radius):
                # Get local region
                region = quantized[y-radius:y+radius+1, x-radius:x+radius+1]
                colors = arr[y-radius:y+radius+1, x-radius:x+radius+1]

                # Find most common intensity
                values, counts = np.unique(region, return_counts=True)
                most_common = values[np.argmax(counts)]

                # Average colors with that intensity
                mask = region == most_common
                result[y, x] = colors[mask].mean(axis=0)

        return Image.fromarray(result.astype(np.uint8))


---

## Chapter 8: ImageMagick Command-Line Mastery

### 8.1 ImageMagick Overview

```bash
# ═══════════════════════════════════════════════════════════════
# IMAGEMAGICK COMMAND-LINE REFERENCE
# ═══════════════════════════════════════════════════════════════

# ImageMagick uses two main commands:
# - convert: Create new files (legacy, but widely used)
# - magick: Modern unified command (ImageMagick 7+)

# ─────────────────────────────────────────────────────────────────
# BASIC OPERATIONS
# ─────────────────────────────────────────────────────────────────

# Convert format
magick input.jpg output.png
convert input.png output.jpg

# Resize
magick input.jpg -resize 800x600 output.jpg
magick input.jpg -resize 50% output.jpg
magick input.jpg -resize 800x600! output.jpg  # Exact size (ignore aspect)
magick input.jpg -resize 800x600^ output.jpg  # Fill (crop excess)
magick input.jpg -resize 800x600\> output.jpg # Only shrink if larger

# Crop
magick input.jpg -crop 100x100+50+50 output.jpg  # width x height + x + y
magick input.jpg -gravity center -crop 100x100+0+0 output.jpg

# Rotate
magick input.jpg -rotate 45 output.jpg
magick input.jpg -rotate 45 -background white output.jpg

# Flip/Flop
magick input.jpg -flip output.jpg    # Vertical flip
magick input.jpg -flop output.jpg    # Horizontal flip

# ─────────────────────────────────────────────────────────────────
# COLOR ADJUSTMENTS
# ─────────────────────────────────────────────────────────────────

# Brightness/Contrast
magick input.jpg -brightness-contrast 10x20 output.jpg

# Levels
magick input.jpg -level 10%,90% output.jpg
magick input.jpg -level 10%,90%,1.5 output.jpg  # With gamma

# Gamma
magick input.jpg -gamma 1.5 output.jpg
magick input.jpg -gamma 1.5,1.0,0.8 output.jpg  # Per channel

# Normalize (auto-levels)
magick input.jpg -normalize output.jpg
magick input.jpg -auto-level output.jpg

# Equalize histogram
magick input.jpg -equalize output.jpg

# Modulate (HSL adjustment)
magick input.jpg -modulate 110,120,100 output.jpg  # brightness,saturation,hue

# Colorize
magick input.jpg -colorize 50,0,50 output.jpg

# Grayscale
magick input.jpg -colorspace Gray output.jpg
magick input.jpg -type Grayscale output.jpg

# Sepia
magick input.jpg -sepia-tone 80% output.jpg

# Negate/Invert
magick input.jpg -negate output.jpg

# Posterize
magick input.jpg -posterize 4 output.jpg

# Solarize
magick input.jpg -solarize 50% output.jpg

# ─────────────────────────────────────────────────────────────────
# FILTERS AND EFFECTS
# ─────────────────────────────────────────────────────────────────

# Blur
magick input.jpg -blur 0x5 output.jpg  # radius x sigma
magick input.jpg -gaussian-blur 0x5 output.jpg
magick input.jpg -motion-blur 0x10+45 output.jpg  # radius x sigma + angle
magick input.jpg -radial-blur 10 output.jpg

# Sharpen
magick input.jpg -sharpen 0x1 output.jpg
magick input.jpg -unsharp 0x1+1+0.05 output.jpg  # radius+sigma+amount+threshold

# Edge detection
magick input.jpg -edge 1 output.jpg
magick input.jpg -canny 0x1+10%+30% output.jpg

# Emboss
magick input.jpg -emboss 0x1 output.jpg

# Noise
magick input.jpg -noise 5 output.jpg  # Add noise
magick input.jpg +noise Gaussian output.jpg
magick input.jpg -noise 3 output.jpg  # Reduce noise

# Oil paint
magick input.jpg -paint 3 output.jpg

# Charcoal
magick input.jpg -charcoal 2 output.jpg

# Sketch
magick input.jpg -sketch 0x20+120 output.jpg

# Vignette
magick input.jpg -vignette 0x50 output.jpg

# ─────────────────────────────────────────────────────────────────
# DISTORTION
# ─────────────────────────────────────────────────────────────────

# Perspective
magick input.jpg -distort Perspective '0,0,10,10 0,100,10,90 100,0,90,10 100,100,90,90' output.jpg

# Barrel/Pincushion
magick input.jpg -distort Barrel '0.1 0.0 0.0' output.jpg

# Polar
magick input.jpg -distort Polar '0' output.jpg
magick input.jpg -distort DePolar '0' output.jpg

# Arc
magick input.jpg -distort Arc '180' output.jpg

# Wave
magick input.jpg -wave 10x100 output.jpg

# Swirl
magick input.jpg -swirl 90 output.jpg

# Implode/Explode
magick input.jpg -implode 0.5 output.jpg
magick input.jpg -implode -0.5 output.jpg  # Explode

# ─────────────────────────────────────────────────────────────────
# COMPOSITING
# ─────────────────────────────────────────────────────────────────

# Composite images
magick base.jpg overlay.png -composite output.jpg
magick base.jpg overlay.png -gravity center -composite output.jpg

# With blend modes
magick base.jpg overlay.png -compose Multiply -composite output.jpg
magick base.jpg overlay.png -compose Screen -composite output.jpg
magick base.jpg overlay.png -compose Overlay -composite output.jpg
magick base.jpg overlay.png -compose SoftLight -composite output.jpg
magick base.jpg overlay.png -compose Difference -composite output.jpg

# Blend with opacity
magick base.jpg overlay.png -compose blend -define compose:args=50 -composite output.jpg

# Dissolve
magick base.jpg overlay.png -compose dissolve -define compose:args=50% -composite output.jpg

# ─────────────────────────────────────────────────────────────────
# TEXT AND ANNOTATION
# ─────────────────────────────────────────────────────────────────

# Add text
magick input.jpg -font Arial -pointsize 48 -fill white \
    -annotate +100+100 'Hello World' output.jpg

# Text with stroke
magick input.jpg -font Arial -pointsize 48 \
    -stroke black -strokewidth 2 -fill white \
    -annotate +100+100 'Hello World' output.jpg

# Text with gravity
magick input.jpg -font Arial -pointsize 48 -fill white \
    -gravity south -annotate +0+20 'Caption' output.jpg

# Watermark
magick input.jpg \( watermark.png -resize 100x100 \) \
    -gravity southeast -geometry +10+10 -composite output.jpg

# ─────────────────────────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────────────────────────

# Process multiple files
magick mogrify -resize 800x600 *.jpg

# Convert directory
for f in *.png; do magick "$f" "${f%.png}.jpg"; done

# Batch with specific output directory
magick mogrify -path output/ -resize 50% *.jpg

# Create contact sheet
magick montage *.jpg -geometry 200x200+5+5 -tile 4x4 contact.jpg

# Create animated GIF
magick -delay 100 -loop 0 frame*.png animation.gif

# ─────────────────────────────────────────────────────────────────
# ADVANCED OPERATIONS
# ─────────────────────────────────────────────────────────────────

# Create gradient
magick -size 800x600 gradient:blue-white gradient.png
magick -size 800x600 radial-gradient:white-black radial.png

# Create noise pattern
magick -size 800x600 xc: +noise Random noise.png

# Create plasma
magick -size 800x600 plasma:red-blue plasma.png

# Create checkerboard
magick -size 800x600 pattern:checkerboard checker.png

# Apply lookup table (LUT)
magick input.jpg -hald-clut lut.png output.jpg

# Custom convolution
magick input.jpg -morphology Convolve '3x3: -1,-1,-1 -1,9,-1 -1,-1,-1' output.jpg

# Channel operations
magick input.jpg -channel R -separate red_channel.jpg
magick r.jpg g.jpg b.jpg -combine rgb.jpg

# Alpha channel
magick input.png -alpha extract alpha.jpg
magick input.jpg mask.png -alpha off -compose CopyOpacity -composite output.png
```

---

# PART III: MATHEMATICAL FOUNDATIONS

---

## Chapter 10: Linear Algebra for Graphics

### 10.1 Vectors and Matrices

```python
"""
LINEAR ALGEBRA FOR GRAPHICS - COMPLETE GUIDE

Essential mathematical foundation for image processing
and procedural art generation.
"""

import numpy as np
import math

# ═══════════════════════════════════════════════════════════════
# VECTORS
# ═══════════════════════════════════════════════════════════════

class Vec2:
    """2D Vector operations"""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        return Vec2(self.x / scalar, self.y / scalar)

    def dot(self, other):
        """Dot product"""
        return self.x * other.x + self.y * other.y

    def cross(self, other):
        """2D cross product (returns scalar)"""
        return self.x * other.y - self.y * other.x

    def length(self):
        """Magnitude/length of vector"""
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        """Unit vector"""
        l = self.length()
        if l == 0:
            return Vec2(0, 0)
        return Vec2(self.x / l, self.y / l)

    def rotate(self, angle):
        """Rotate by angle (radians)"""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vec2(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )

    def perpendicular(self):
        """Perpendicular vector (90° counter-clockwise)"""
        return Vec2(-self.y, self.x)

    def reflect(self, normal):
        """Reflect across normal"""
        d = self.dot(normal) * 2
        return Vec2(self.x - normal.x * d, self.y - normal.y * d)

    def lerp(self, other, t):
        """Linear interpolation"""
        return Vec2(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t
        )

    def angle(self):
        """Angle from positive x-axis"""
        return math.atan2(self.y, self.x)

    def distance_to(self, other):
        """Distance to another point"""
        return (self - other).length()


class Vec3:
    """3D Vector operations"""

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        """3D cross product"""
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        l = self.length()
        if l == 0:
            return Vec3(0, 0, 0)
        return Vec3(self.x / l, self.y / l, self.z / l)


# ═══════════════════════════════════════════════════════════════
# TRANSFORMATION MATRICES
# ═══════════════════════════════════════════════════════════════

class Matrix3x3:
    """3x3 Transformation matrix for 2D graphics (homogeneous coordinates)"""

    def __init__(self, data=None):
        if data is None:
            self.data = np.eye(3)
        else:
            self.data = np.array(data, dtype=np.float64).reshape(3, 3)

    @staticmethod
    def identity():
        """Identity matrix"""
        return Matrix3x3()

    @staticmethod
    def translation(tx, ty):
        """Translation matrix"""
        return Matrix3x3([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])

    @staticmethod
    def scale(sx, sy=None):
        """Scale matrix"""
        if sy is None:
            sy = sx
        return Matrix3x3([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ])

    @staticmethod
    def rotation(angle):
        """Rotation matrix (angle in radians)"""
        c, s = math.cos(angle), math.sin(angle)
        return Matrix3x3([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

    @staticmethod
    def shear(shx, shy):
        """Shear matrix"""
        return Matrix3x3([
            [1, shx, 0],
            [shy, 1, 0],
            [0, 0, 1]
        ])

    @staticmethod
    def reflection_x():
        """Reflect across x-axis"""
        return Matrix3x3([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])

    @staticmethod
    def reflection_y():
        """Reflect across y-axis"""
        return Matrix3x3([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

    @staticmethod
    def reflection_origin():
        """Reflect across origin"""
        return Matrix3x3([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])

    def __matmul__(self, other):
        """Matrix multiplication"""
        if isinstance(other, Matrix3x3):
            return Matrix3x3(np.matmul(self.data, other.data))
        elif isinstance(other, (tuple, list)) and len(other) == 2:
            # Transform point
            p = np.array([other[0], other[1], 1])
            result = np.matmul(self.data, p)
            return (result[0] / result[2], result[1] / result[2])
        else:
            raise ValueError("Invalid operand")

    def transform_point(self, x, y):
        """Transform a 2D point"""
        return self @ (x, y)

    def transform_points(self, points):
        """Transform multiple points"""
        return [self.transform_point(x, y) for x, y in points]

    def inverse(self):
        """Matrix inverse"""
        return Matrix3x3(np.linalg.inv(self.data))

    def determinant(self):
        """Matrix determinant"""
        return np.linalg.det(self.data)


# ═══════════════════════════════════════════════════════════════
# AFFINE TRANSFORMATIONS
# ═══════════════════════════════════════════════════════════════

class AffineTransform:
    """Affine transformation utilities"""

    @staticmethod
    def combine(*matrices):
        """Combine multiple transformations (apply left to right)"""
        result = Matrix3x3.identity()
        for m in matrices:
            result = result @ m
        return result

    @staticmethod
    def rotate_around(angle, cx, cy):
        """Rotate around point (cx, cy)"""
        return AffineTransform.combine(
            Matrix3x3.translation(-cx, -cy),
            Matrix3x3.rotation(angle),
            Matrix3x3.translation(cx, cy)
        )

    @staticmethod
    def scale_around(sx, sy, cx, cy):
        """Scale around point (cx, cy)"""
        return AffineTransform.combine(
            Matrix3x3.translation(-cx, -cy),
            Matrix3x3.scale(sx, sy),
            Matrix3x3.translation(cx, cy)
        )

    @staticmethod
    def from_points(src, dst):
        """
        Calculate affine transform from 3 source points to 3 destination points.

        src: [(x1,y1), (x2,y2), (x3,y3)]
        dst: [(x1',y1'), (x2',y2'), (x3',y3')]
        """
        # Build system of equations
        A = np.array([
            [src[0][0], src[0][1], 1, 0, 0, 0],
            [0, 0, 0, src[0][0], src[0][1], 1],
            [src[1][0], src[1][1], 1, 0, 0, 0],
            [0, 0, 0, src[1][0], src[1][1], 1],
            [src[2][0], src[2][1], 1, 0, 0, 0],
            [0, 0, 0, src[2][0], src[2][1], 1],
        ])

        b = np.array([dst[0][0], dst[0][1], dst[1][0], dst[1][1], dst[2][0], dst[2][1]])

        # Solve
        x = np.linalg.solve(A, b)

        return Matrix3x3([
            [x[0], x[1], x[2]],
            [x[3], x[4], x[5]],
            [0, 0, 1]
        ])


# ═══════════════════════════════════════════════════════════════
# PERSPECTIVE TRANSFORMATIONS
# ═══════════════════════════════════════════════════════════════

class PerspectiveTransform:
    """Perspective (projective) transformation utilities"""

    @staticmethod
    def from_quadrilateral(src, dst):
        """
        Calculate perspective transform from 4 source points to 4 destination points.

        src: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        dst: [(x1',y1'), (x2',y2'), (x3',y3'), (x4',y4')]
        """
        # Build matrix A for system Ax = b
        A = []
        b = []

        for (x, y), (xp, yp) in zip(src, dst):
            A.append([x, y, 1, 0, 0, 0, -xp*x, -xp*y])
            A.append([0, 0, 0, x, y, 1, -yp*x, -yp*y])
            b.extend([xp, yp])

        A = np.array(A)
        b = np.array(b)

        # Solve for transformation coefficients
        h = np.linalg.solve(A, b)

        return Matrix3x3([
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], 1]
        ])

    @staticmethod
    def transform_image(img, matrix, output_size=None):
        """Apply perspective transform to PIL Image"""
        from PIL import Image

        if output_size is None:
            output_size = img.size

        # Get inverse matrix for reverse mapping
        inv_matrix = matrix.inverse()
        coeffs = inv_matrix.data.flatten()[:8]

        return img.transform(
            output_size,
            Image.PERSPECTIVE,
            coeffs,
            Image.BICUBIC
        )


# ═══════════════════════════════════════════════════════════════
# BEZIER CURVES
# ═══════════════════════════════════════════════════════════════

class BezierCurve:
    """Bezier curve mathematics"""

    @staticmethod
    def binomial(n, k):
        """Binomial coefficient C(n, k)"""
        if k < 0 or k > n:
            return 0
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result

    @staticmethod
    def bernstein(n, i, t):
        """Bernstein polynomial B(n,i,t)"""
        return BezierCurve.binomial(n, i) * (t ** i) * ((1 - t) ** (n - i))

    @staticmethod
    def evaluate(points, t):
        """
        Evaluate Bezier curve at parameter t.
        points: list of control points [(x0,y0), (x1,y1), ...]
        t: parameter 0-1
        """
        n = len(points) - 1
        x = sum(BezierCurve.bernstein(n, i, t) * p[0] for i, p in enumerate(points))
        y = sum(BezierCurve.bernstein(n, i, t) * p[1] for i, p in enumerate(points))
        return (x, y)

    @staticmethod
    def sample(points, num_samples=100):
        """Sample curve at regular intervals"""
        return [BezierCurve.evaluate(points, t / (num_samples - 1))
                for t in range(num_samples)]

    @staticmethod
    def de_casteljau(points, t):
        """De Casteljau's algorithm (numerically stable)"""
        points = [list(p) for p in points]
        n = len(points)

        for j in range(1, n):
            for i in range(n - j):
                points[i][0] = (1 - t) * points[i][0] + t * points[i + 1][0]
                points[i][1] = (1 - t) * points[i][1] + t * points[i + 1][1]

        return tuple(points[0])

    @staticmethod
    def derivative(points, t):
        """First derivative (tangent vector) at t"""
        n = len(points) - 1
        if n == 0:
            return (0, 0)

        # Derivative control points
        deriv_points = [
            ((points[i+1][0] - points[i][0]) * n,
             (points[i+1][1] - points[i][1]) * n)
            for i in range(n)
        ]

        return BezierCurve.evaluate(deriv_points, t)

    @staticmethod
    def curvature(points, t):
        """Curvature at parameter t"""
        d1 = BezierCurve.derivative(points, t)

        # Second derivative
        n = len(points) - 1
        deriv1_points = [
            ((points[i+1][0] - points[i][0]) * n,
             (points[i+1][1] - points[i][1]) * n)
            for i in range(n)
        ]
        d2 = BezierCurve.derivative(deriv1_points, t)

        # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        denom = (d1[0]**2 + d1[1]**2) ** 1.5

        if denom == 0:
            return 0
        return abs(cross) / denom

    @staticmethod
    def split(points, t):
        """Split curve at parameter t into two curves"""
        n = len(points)
        left = []
        right = [None] * n

        # Working copy
        work = [list(p) for p in points]

        left.append(tuple(work[0]))
        right[n-1] = tuple(work[n-1])

        for j in range(1, n):
            for i in range(n - j):
                work[i] = (
                    (1 - t) * work[i][0] + t * work[i + 1][0],
                    (1 - t) * work[i][1] + t * work[i + 1][1]
                )
            left.append(tuple(work[0]))
            right[n - 1 - j] = tuple(work[n - 1 - j])

        return left, right


# ═══════════════════════════════════════════════════════════════
# INTERPOLATION
# ═══════════════════════════════════════════════════════════════

class Interpolation:
    """Interpolation functions"""

    @staticmethod
    def linear(a, b, t):
        """Linear interpolation"""
        return a + (b - a) * t

    @staticmethod
    def smoothstep(a, b, t):
        """Smooth interpolation (Hermite)"""
        t = max(0, min(1, t))
        t = t * t * (3 - 2 * t)
        return a + (b - a) * t

    @staticmethod
    def smootherstep(a, b, t):
        """Smoother interpolation (Ken Perlin)"""
        t = max(0, min(1, t))
        t = t * t * t * (t * (t * 6 - 15) + 10)
        return a + (b - a) * t

    @staticmethod
    def cosine(a, b, t):
        """Cosine interpolation"""
        t = (1 - math.cos(t * math.pi)) / 2
        return a + (b - a) * t

    @staticmethod
    def cubic(y0, y1, y2, y3, t):
        """Cubic interpolation using 4 points"""
        a0 = y3 - y2 - y0 + y1
        a1 = y0 - y1 - a0
        a2 = y2 - y0
        a3 = y1
        return a0 * t**3 + a1 * t**2 + a2 * t + a3

    @staticmethod
    def catmull_rom(p0, p1, p2, p3, t):
        """Catmull-Rom spline interpolation"""
        t2 = t * t
        t3 = t2 * t

        return 0.5 * (
            (2 * p1) +
            (-p0 + p2) * t +
            (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
            (-p0 + 3*p1 - 3*p2 + p3) * t3
        )

    @staticmethod
    def bilinear(values, x, y):
        """
        Bilinear interpolation on 2x2 grid.
        values: [[v00, v01], [v10, v11]]
        x, y: fractional coordinates 0-1
        """
        v00, v01 = values[0]
        v10, v11 = values[1]

        return (
            v00 * (1-x) * (1-y) +
            v01 * x * (1-y) +
            v10 * (1-x) * y +
            v11 * x * y
        )

    @staticmethod
    def bicubic(values, x, y):
        """
        Bicubic interpolation on 4x4 grid.
        values: 4x4 array
        x, y: fractional coordinates 0-1
        """
        # Interpolate 4 rows
        rows = [
            Interpolation.cubic(values[i][0], values[i][1],
                              values[i][2], values[i][3], x)
            for i in range(4)
        ]
        # Interpolate column
        return Interpolation.cubic(rows[0], rows[1], rows[2], rows[3], y)
```


---

## Chapter 11: Convolution and Kernel Operations

### 11.1 Understanding Convolution

Convolution is the mathematical operation at the heart of almost all image filtering. It slides a small matrix (kernel) across the image, computing weighted sums at each position.

```python
"""
CONVOLUTION AND KERNEL OPERATIONS - COMPREHENSIVE GUIDE

The mathematical foundation of image filtering, edge detection,
blurring, sharpening, and countless other operations.
"""

import numpy as np
from PIL import Image
from scipy import ndimage, signal
import math

# ═══════════════════════════════════════════════════════════════
# CONVOLUTION FUNDAMENTALS
# ═══════════════════════════════════════════════════════════════

class ConvolutionEngine:
    """Complete convolution implementation and kernel library"""
    
    @staticmethod
    def convolve_2d_manual(image: np.ndarray, kernel: np.ndarray, 
                          padding: str = 'same') -> np.ndarray:
        """
        Manual 2D convolution implementation for educational purposes.
        
        This is slower than scipy but shows exactly what's happening.
        
        Args:
            image: 2D grayscale image as numpy array
            kernel: 2D convolution kernel
            padding: 'same' (output same size), 'valid' (no padding), 'full'
        """
        img_h, img_w = image.shape
        k_h, k_w = kernel.shape
        
        # Calculate padding
        if padding == 'same':
            pad_h = k_h // 2
            pad_w = k_w // 2
            output_h = img_h
            output_w = img_w
        elif padding == 'valid':
            pad_h = pad_w = 0
            output_h = img_h - k_h + 1
            output_w = img_w - k_w + 1
        else:  # 'full'
            pad_h = k_h - 1
            pad_w = k_w - 1
            output_h = img_h + k_h - 1
            output_w = img_w + k_w - 1
        
        # Pad image
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        
        # Output array
        output = np.zeros((output_h, output_w), dtype=np.float64)
        
        # Flip kernel (convolution vs correlation)
        kernel_flipped = np.flip(np.flip(kernel, 0), 1)
        
        # Perform convolution
        for y in range(output_h):
            for x in range(output_w):
                # Extract region
                region = padded[y:y+k_h, x:x+k_w]
                # Compute weighted sum
                output[y, x] = np.sum(region * kernel_flipped)
        
        return output
    
    @staticmethod
    def convolve_color(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Apply convolution to each color channel separately.
        """
        if len(image.shape) == 2:
            return ndimage.convolve(image.astype(float), kernel)
        
        result = np.zeros_like(image, dtype=np.float64)
        for c in range(image.shape[2]):
            result[:, :, c] = ndimage.convolve(
                image[:, :, c].astype(float), kernel
            )
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def separable_convolve(image: np.ndarray, 
                          kernel_h: np.ndarray, 
                          kernel_v: np.ndarray) -> np.ndarray:
        """
        Optimized convolution for separable kernels (Gaussian, box, etc.).
        
        A 2D kernel is separable if it can be expressed as the outer product
        of two 1D kernels. This reduces O(n²) to O(2n) operations.
        
        Example: 5x5 Gaussian requires 25 multiplications per pixel normally.
        Separable: 5 + 5 = 10 multiplications per pixel.
        """
        # Horizontal pass
        temp = ndimage.convolve1d(image.astype(float), kernel_h, axis=1)
        # Vertical pass
        result = ndimage.convolve1d(temp, kernel_v, axis=0)
        return result


# ═══════════════════════════════════════════════════════════════
# KERNEL LIBRARY
# ═══════════════════════════════════════════════════════════════

class Kernels:
    """Comprehensive library of convolution kernels"""
    
    # ─────────────────────────────────────────────────────────
    # IDENTITY AND BASIC
    # ─────────────────────────────────────────────────────────
    
    IDENTITY_3x3 = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.float64)
    
    # ─────────────────────────────────────────────────────────
    # BOX BLUR (MEAN FILTER)
    # ─────────────────────────────────────────────────────────
    
    BOX_3x3 = np.ones((3, 3), dtype=np.float64) / 9
    BOX_5x5 = np.ones((5, 5), dtype=np.float64) / 25
    BOX_7x7 = np.ones((7, 7), dtype=np.float64) / 49
    
    @staticmethod
    def box_blur(size: int) -> np.ndarray:
        """Generate box blur kernel of any size"""
        return np.ones((size, size), dtype=np.float64) / (size * size)
    
    # ─────────────────────────────────────────────────────────
    # GAUSSIAN BLUR
    # ─────────────────────────────────────────────────────────
    
    GAUSSIAN_3x3 = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float64) / 16
    
    GAUSSIAN_5x5 = np.array([
        [1,  4,  7,  4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1,  4,  7,  4, 1]
    ], dtype=np.float64) / 273
    
    @staticmethod
    def gaussian(size: int, sigma: float = None) -> np.ndarray:
        """
        Generate Gaussian kernel of specified size.
        
        The Gaussian function creates a bell curve that provides
        smooth, natural-looking blur without artifacts.
        
        G(x,y) = (1/(2πσ²)) * e^(-(x²+y²)/(2σ²))
        
        Args:
            size: Kernel size (should be odd)
            sigma: Standard deviation (default: size/6)
        """
        if sigma is None:
            sigma = size / 6
        
        # Generate coordinate grids
        x = np.arange(size) - size // 2
        y = np.arange(size) - size // 2
        xx, yy = np.meshgrid(x, y)
        
        # Gaussian formula
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        
        # Normalize
        return kernel / kernel.sum()
    
    @staticmethod
    def gaussian_separable(size: int, sigma: float = None):
        """
        Generate separable 1D Gaussian kernels.
        Returns (horizontal, vertical) kernel pair.
        """
        if sigma is None:
            sigma = size / 6
        
        x = np.arange(size) - size // 2
        kernel = np.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        return kernel, kernel
    
    # ─────────────────────────────────────────────────────────
    # SHARPENING
    # ─────────────────────────────────────────────────────────
    
    # Basic sharpen
    SHARPEN_3x3 = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float64)
    
    # Strong sharpen
    SHARPEN_STRONG = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float64)
    
    # Unsharp mask style
    SHARPEN_UNSHARP = np.array([
        [ 1,  4,    6,  4,  1],
        [ 4, 16,   24, 16,  4],
        [ 6, 24, -476, 24,  6],
        [ 4, 16,   24, 16,  4],
        [ 1,  4,    6,  4,  1]
    ], dtype=np.float64) / -256
    
    @staticmethod
    def unsharp_mask(size: int = 5, sigma: float = 1.0, amount: float = 1.0):
        """
        Generate unsharp mask kernel.
        
        Unsharp masking: original + amount * (original - blurred)
        """
        gaussian = Kernels.gaussian(size, sigma)
        identity = np.zeros_like(gaussian)
        identity[size//2, size//2] = 1
        
        # original + amount * (original - blurred)
        # = (1 + amount) * identity - amount * gaussian
        return (1 + amount) * identity - amount * gaussian
    
    # ─────────────────────────────────────────────────────────
    # EDGE DETECTION - FIRST DERIVATIVE
    # ─────────────────────────────────────────────────────────
    
    # Simple gradients
    GRADIENT_X = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    GRADIENT_Y = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float64)
    
    # Prewitt operator
    PREWITT_X = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    PREWITT_Y = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float64)
    
    # Sobel operator (weighted Prewitt)
    SOBEL_X = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    SOBEL_Y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)
    
    # Scharr operator (improved Sobel)
    SCHARR_X = np.array([
        [-3,  0,  3],
        [-10, 0, 10],
        [-3,  0,  3]
    ], dtype=np.float64)
    
    SCHARR_Y = np.array([
        [-3, -10, -3],
        [ 0,   0,  0],
        [ 3,  10,  3]
    ], dtype=np.float64)
    
    # Roberts cross
    ROBERTS_X = np.array([
        [1,  0],
        [0, -1]
    ], dtype=np.float64)
    
    ROBERTS_Y = np.array([
        [ 0, 1],
        [-1, 0]
    ], dtype=np.float64)
    
    # ─────────────────────────────────────────────────────────
    # EDGE DETECTION - SECOND DERIVATIVE
    # ─────────────────────────────────────────────────────────
    
    # Laplacian (isotropic)
    LAPLACIAN_4 = np.array([
        [ 0,  1,  0],
        [ 1, -4,  1],
        [ 0,  1,  0]
    ], dtype=np.float64)
    
    LAPLACIAN_8 = np.array([
        [ 1,  1,  1],
        [ 1, -8,  1],
        [ 1,  1,  1]
    ], dtype=np.float64)
    
    # Laplacian of Gaussian (LoG) - "Mexican Hat"
    @staticmethod
    def laplacian_of_gaussian(size: int = 9, sigma: float = 1.4) -> np.ndarray:
        """
        Generate Laplacian of Gaussian kernel.
        
        Combines Gaussian smoothing with Laplacian edge detection.
        Known as "Mexican Hat" due to its shape.
        
        LoG(x,y) = -1/(πσ⁴) * [1 - (x²+y²)/(2σ²)] * e^(-(x²+y²)/(2σ²))
        """
        x = np.arange(size) - size // 2
        y = np.arange(size) - size // 2
        xx, yy = np.meshgrid(x, y)
        
        r_squared = xx**2 + yy**2
        sigma_sq = sigma**2
        
        # LoG formula
        kernel = -(1 / (np.pi * sigma_sq**2)) * \
                 (1 - r_squared / (2 * sigma_sq)) * \
                 np.exp(-r_squared / (2 * sigma_sq))
        
        # Normalize to sum to 0
        kernel = kernel - kernel.mean()
        
        return kernel
    
    # ─────────────────────────────────────────────────────────
    # EMBOSSING
    # ─────────────────────────────────────────────────────────
    
    EMBOSS_TOP_LEFT = np.array([
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2]
    ], dtype=np.float64)
    
    EMBOSS_TOP = np.array([
        [-1, -1, -1],
        [ 0,  1,  0],
        [ 1,  1,  1]
    ], dtype=np.float64)
    
    EMBOSS_TOP_RIGHT = np.array([
        [0, -1, -2],
        [1,  1, -1],
        [2,  1,  0]
    ], dtype=np.float64)
    
    EMBOSS_LEFT = np.array([
        [-1, 0, 1],
        [-2, 1, 2],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    @staticmethod
    def emboss(angle_degrees: float = 135, strength: float = 1.0) -> np.ndarray:
        """
        Generate emboss kernel at specified angle.
        
        Args:
            angle_degrees: Light direction (0=right, 90=top, 180=left, 270=bottom)
            strength: Emboss intensity
        """
        angle = math.radians(angle_degrees)
        dx = math.cos(angle)
        dy = -math.sin(angle)  # Negative because y increases downward
        
        # Create directional kernel
        kernel = np.array([
            [-dy - dx, -dy, -dy + dx],
            [-dx,       1,  dx],
            [dy - dx,   dy,  dy + dx]
        ], dtype=np.float64)
        
        return kernel * strength
    
    # ─────────────────────────────────────────────────────────
    # MOTION BLUR
    # ─────────────────────────────────────────────────────────
    
    @staticmethod
    def motion_blur(size: int, angle_degrees: float = 0) -> np.ndarray:
        """
        Generate motion blur kernel.
        
        Args:
            size: Length of motion blur
            angle_degrees: Direction of motion
        """
        kernel = np.zeros((size, size), dtype=np.float64)
        center = size // 2
        
        angle = math.radians(angle_degrees)
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        for i in range(size):
            t = i - center
            x = int(round(center + t * dx))
            y = int(round(center + t * dy))
            
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1
        
        return kernel / kernel.sum()
    
    # ─────────────────────────────────────────────────────────
    # HIGH-PASS AND BAND-PASS
    # ─────────────────────────────────────────────────────────
    
    # High-pass = Identity - Low-pass
    HIGH_PASS_3x3 = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float64)
    
    HIGH_PASS_5x5 = np.array([
        [-1, -1, -1, -1, -1],
        [-1,  1,  2,  1, -1],
        [-1,  2,  4,  2, -1],
        [-1,  1,  2,  1, -1],
        [-1, -1, -1, -1, -1]
    ], dtype=np.float64)
    
    @staticmethod
    def band_pass(low_sigma: float, high_sigma: float, size: int = 11):
        """
        Generate band-pass filter (Difference of Gaussians).
        
        Passes frequencies between two cutoffs.
        DoG approximates the Laplacian of Gaussian.
        """
        g1 = Kernels.gaussian(size, low_sigma)
        g2 = Kernels.gaussian(size, high_sigma)
        return g1 - g2
    
    # ─────────────────────────────────────────────────────────
    # MORPHOLOGICAL OPERATIONS
    # ─────────────────────────────────────────────────────────
    
    # Structuring elements for morphological operations
    MORPH_SQUARE_3x3 = np.ones((3, 3), dtype=np.uint8)
    MORPH_CROSS_3x3 = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.uint8)
    
    @staticmethod
    def morph_circle(radius: int) -> np.ndarray:
        """Generate circular structuring element"""
        size = 2 * radius + 1
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x**2 + y**2 <= radius**2
        return mask.astype(np.uint8)
    
    @staticmethod
    def morph_ellipse(width: int, height: int) -> np.ndarray:
        """Generate elliptical structuring element"""
        y, x = np.ogrid[-height:height+1, -width:width+1]
        mask = (x**2 / width**2) + (y**2 / height**2) <= 1
        return mask.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# EDGE DETECTION ALGORITHMS
# ═══════════════════════════════════════════════════════════════

class EdgeDetection:
    """Complete edge detection implementations"""
    
    @staticmethod
    def sobel(image: np.ndarray) -> tuple:
        """
        Sobel edge detection.
        
        Returns:
            gradient_magnitude: Edge strength
            gradient_direction: Edge direction in radians
        """
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)  # Convert to grayscale
        
        gx = ndimage.convolve(image.astype(float), Kernels.SOBEL_X)
        gy = ndimage.convolve(image.astype(float), Kernels.SOBEL_Y)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)
        
        return magnitude, direction
    
    @staticmethod
    def canny(image: np.ndarray, 
              low_threshold: float = 50, 
              high_threshold: float = 150,
              sigma: float = 1.4) -> np.ndarray:
        """
        Canny edge detection - the gold standard.
        
        Steps:
        1. Gaussian smoothing
        2. Gradient calculation (Sobel)
        3. Non-maximum suppression
        4. Double thresholding
        5. Edge tracking by hysteresis
        
        Args:
            image: Input grayscale image
            low_threshold: Lower threshold for hysteresis
            high_threshold: Upper threshold for hysteresis
            sigma: Gaussian smoothing sigma
        """
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        
        image = image.astype(float)
        
        # Step 1: Gaussian smoothing
        smoothed = ndimage.gaussian_filter(image, sigma)
        
        # Step 2: Gradient calculation
        gx = ndimage.convolve(smoothed, Kernels.SOBEL_X)
        gy = ndimage.convolve(smoothed, Kernels.SOBEL_Y)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)
        
        # Step 3: Non-maximum suppression
        h, w = magnitude.shape
        suppressed = np.zeros_like(magnitude)
        
        # Quantize direction to 4 angles
        direction = direction * 180 / np.pi
        direction[direction < 0] += 180
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                angle = direction[y, x]
                mag = magnitude[y, x]
                
                # Check neighbors based on gradient direction
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    # Horizontal edge - check left/right
                    n1, n2 = magnitude[y, x-1], magnitude[y, x+1]
                elif 22.5 <= angle < 67.5:
                    # Diagonal edge
                    n1, n2 = magnitude[y-1, x+1], magnitude[y+1, x-1]
                elif 67.5 <= angle < 112.5:
                    # Vertical edge - check top/bottom
                    n1, n2 = magnitude[y-1, x], magnitude[y+1, x]
                else:
                    # Other diagonal
                    n1, n2 = magnitude[y-1, x-1], magnitude[y+1, x+1]
                
                if mag >= n1 and mag >= n2:
                    suppressed[y, x] = mag
        
        # Step 4: Double thresholding
        strong = suppressed >= high_threshold
        weak = (suppressed >= low_threshold) & (suppressed < high_threshold)
        
        # Step 5: Hysteresis edge tracking
        edges = strong.copy()
        
        # Iteratively connect weak edges to strong edges
        for _ in range(10):  # Multiple passes for full connectivity
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if weak[y, x]:
                        # Check if connected to strong edge
                        if np.any(edges[y-1:y+2, x-1:x+2]):
                            edges[y, x] = True
                            weak[y, x] = False
        
        return (edges * 255).astype(np.uint8)
    
    @staticmethod
    def laplacian_edges(image: np.ndarray, 
                        threshold: float = 0) -> np.ndarray:
        """
        Laplacian edge detection with zero-crossing.
        
        The Laplacian finds edges by detecting sign changes
        in the second derivative.
        """
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        
        # Smooth first to reduce noise
        smoothed = ndimage.gaussian_filter(image.astype(float), 1.0)
        
        # Apply Laplacian
        laplacian = ndimage.convolve(smoothed, Kernels.LAPLACIAN_8)
        
        # Find zero crossings
        h, w = laplacian.shape
        edges = np.zeros((h, w), dtype=np.uint8)
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                neighbors = [
                    laplacian[y-1, x], laplacian[y+1, x],
                    laplacian[y, x-1], laplacian[y, x+1]
                ]
                
                # Check for sign change
                center = laplacian[y, x]
                for n in neighbors:
                    if (center > 0 and n < 0) or (center < 0 and n > 0):
                        if abs(center - n) > threshold:
                            edges[y, x] = 255
                            break
        
        return edges
    
    @staticmethod
    def difference_of_gaussians(image: np.ndarray,
                                sigma1: float = 1.0,
                                sigma2: float = 2.0,
                                threshold: float = 0) -> np.ndarray:
        """
        Difference of Gaussians (DoG) edge detection.
        
        Approximates the Laplacian of Gaussian but is faster
        because Gaussian blur can be computed efficiently.
        """
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        
        g1 = ndimage.gaussian_filter(image.astype(float), sigma1)
        g2 = ndimage.gaussian_filter(image.astype(float), sigma2)
        
        dog = g1 - g2
        
        if threshold > 0:
            dog = np.where(np.abs(dog) > threshold, dog, 0)
        
        # Normalize
        dog = (dog - dog.min()) / (dog.max() - dog.min()) * 255
        
        return dog.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# MORPHOLOGICAL OPERATIONS
# ═══════════════════════════════════════════════════════════════

class Morphology:
    """Morphological image operations"""
    
    @staticmethod
    def dilate(image: np.ndarray, 
               structuring_element: np.ndarray = None,
               iterations: int = 1) -> np.ndarray:
        """
        Dilation - expands bright regions.
        
        For each pixel, take the maximum value in the neighborhood
        defined by the structuring element.
        """
        if structuring_element is None:
            structuring_element = Kernels.MORPH_SQUARE_3x3
        
        result = image.copy()
        for _ in range(iterations):
            result = ndimage.grey_dilation(result, structure=structuring_element)
        
        return result
    
    @staticmethod
    def erode(image: np.ndarray,
              structuring_element: np.ndarray = None,
              iterations: int = 1) -> np.ndarray:
        """
        Erosion - shrinks bright regions.
        
        For each pixel, take the minimum value in the neighborhood.
        """
        if structuring_element is None:
            structuring_element = Kernels.MORPH_SQUARE_3x3
        
        result = image.copy()
        for _ in range(iterations):
            result = ndimage.grey_erosion(result, structure=structuring_element)
        
        return result
    
    @staticmethod
    def opening(image: np.ndarray,
                structuring_element: np.ndarray = None) -> np.ndarray:
        """
        Opening = Erosion followed by Dilation.
        
        Removes small bright spots (noise) while preserving shape.
        """
        eroded = Morphology.erode(image, structuring_element)
        return Morphology.dilate(eroded, structuring_element)
    
    @staticmethod
    def closing(image: np.ndarray,
                structuring_element: np.ndarray = None) -> np.ndarray:
        """
        Closing = Dilation followed by Erosion.
        
        Fills small dark holes while preserving shape.
        """
        dilated = Morphology.dilate(image, structuring_element)
        return Morphology.erode(dilated, structuring_element)
    
    @staticmethod
    def gradient(image: np.ndarray,
                 structuring_element: np.ndarray = None) -> np.ndarray:
        """
        Morphological gradient = Dilation - Erosion.
        
        Highlights edges/boundaries.
        """
        dilated = Morphology.dilate(image, structuring_element)
        eroded = Morphology.erode(image, structuring_element)
        return dilated - eroded
    
    @staticmethod
    def top_hat(image: np.ndarray,
                structuring_element: np.ndarray = None) -> np.ndarray:
        """
        Top-hat = Original - Opening.
        
        Extracts bright spots smaller than structuring element.
        """
        opened = Morphology.opening(image, structuring_element)
        return image.astype(float) - opened.astype(float)
    
    @staticmethod
    def black_hat(image: np.ndarray,
                  structuring_element: np.ndarray = None) -> np.ndarray:
        """
        Black-hat = Closing - Original.
        
        Extracts dark spots smaller than structuring element.
        """
        closed = Morphology.closing(image, structuring_element)
        return closed.astype(float) - image.astype(float)
    
    @staticmethod
    def skeleton(image: np.ndarray) -> np.ndarray:
        """
        Skeletonization - reduce shapes to 1-pixel thick lines.
        
        Uses hit-or-miss transform iteratively.
        """
        # Threshold to binary
        binary = image > 127 if len(image.shape) == 2 else \
                 np.mean(image, axis=2) > 127
        
        # Structuring elements for skeletonization
        elements = [
            np.array([[0, 0, 0], [-1, 1, -1], [1, 1, 1]]),
            np.array([[-1, 0, 0], [1, 1, 0], [-1, 1, -1]]),
            np.array([[1, -1, 0], [1, 1, 0], [1, -1, 0]]),
            np.array([[-1, 1, -1], [1, 1, 0], [-1, 0, 0]]),
            np.array([[1, 1, 1], [-1, 1, -1], [0, 0, 0]]),
            np.array([[-1, 1, -1], [0, 1, 1], [0, 0, -1]]),
            np.array([[0, -1, 1], [0, 1, 1], [0, -1, 1]]),
            np.array([[0, 0, -1], [0, 1, 1], [-1, 1, -1]])
        ]
        
        skeleton = binary.copy()
        changed = True
        
        while changed:
            changed = False
            for element in elements:
                eroded = ndimage.binary_erosion(skeleton, element)
                if not np.array_equal(skeleton, eroded | skeleton):
                    skeleton = eroded | skeleton
                    changed = True
        
        return (skeleton * 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# PRACTICAL FILTER COMBINATIONS
# ═══════════════════════════════════════════════════════════════

class FilterPipelines:
    """Common filter combinations for real-world effects"""
    
    @staticmethod
    def artistic_edge_glow(image: np.ndarray, 
                          glow_color: tuple = (0, 255, 255)) -> np.ndarray:
        """
        Create glowing edge effect.
        
        1. Detect edges
        2. Dilate for thickness
        3. Apply Gaussian blur for glow
        4. Colorize and composite
        """
        # Detect edges
        edges = EdgeDetection.canny(image, 30, 100)
        
        # Dilate edges
        edges = Morphology.dilate(edges, iterations=2)
        
        # Create glow
        glow = ndimage.gaussian_filter(edges.astype(float), 5)
        glow = np.clip(glow * 2, 0, 255)
        
        # Colorize
        h, w = edges.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        for c in range(3):
            result[:, :, c] = (glow * glow_color[c] / 255).astype(np.uint8)
        
        # Composite over original
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
        
        alpha = glow[:, :, np.newaxis] / 255
        output = (1 - alpha) * image + alpha * result
        
        return np.clip(output, 0, 255).astype(np.uint8)
    
    @staticmethod
    def pencil_sketch(image: np.ndarray, 
                      blur_sigma: float = 21,
                      blend_strength: float = 1.0) -> np.ndarray:
        """
        Create pencil sketch effect.
        
        1. Convert to grayscale
        2. Invert
        3. Apply heavy Gaussian blur
        4. Color dodge blend with original
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.copy()
        
        # Invert
        inverted = 255 - gray
        
        # Heavy blur
        blurred = ndimage.gaussian_filter(inverted, blur_sigma)
        
        # Color dodge blend: result = bottom / (1 - top)
        # Avoid division by zero
        blurred = np.clip(blurred, 1, 255)
        sketch = np.clip(gray / (1 - blurred / 255 + 0.001) * blend_strength, 0, 255)
        
        return sketch.astype(np.uint8)
    
    @staticmethod
    def oil_painting(image: np.ndarray, 
                     radius: int = 4, 
                     intensity_levels: int = 20) -> np.ndarray:
        """
        Create oil painting effect.
        
        For each pixel, find the most common intensity level
        in the neighborhood and use its average color.
        """
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
        
        h, w, c = image.shape
        result = np.zeros_like(image)
        
        # Compute intensity
        intensity = np.mean(image, axis=2)
        intensity_quantized = (intensity * intensity_levels / 255).astype(int)
        intensity_quantized = np.clip(intensity_quantized, 0, intensity_levels - 1)
        
        for y in range(radius, h - radius):
            for x in range(radius, w - radius):
                # Get neighborhood
                region = image[y-radius:y+radius+1, x-radius:x+radius+1]
                int_region = intensity_quantized[y-radius:y+radius+1, x-radius:x+radius+1]
                
                # Find most common intensity level
                hist = np.bincount(int_region.flatten(), minlength=intensity_levels)
                most_common = np.argmax(hist)
                
                # Get average color of that intensity level
                mask = int_region == most_common
                if np.any(mask):
                    result[y, x] = np.mean(region[mask], axis=0)
                else:
                    result[y, x] = image[y, x]
        
        return result.astype(np.uint8)
    
    @staticmethod
    def posterize(image: np.ndarray, levels: int = 4) -> np.ndarray:
        """
        Reduce image to limited number of colors per channel.
        """
        factor = 255 / (levels - 1)
        return (np.round(image / factor) * factor).astype(np.uint8)
    
    @staticmethod
    def vignette(image: np.ndarray, 
                 strength: float = 0.5,
                 radius: float = 0.8) -> np.ndarray:
        """
        Add vignette (darkened corners) effect.
        """
        if len(image.shape) == 2:
            h, w = image.shape
        else:
            h, w = image.shape[:2]
        
        # Create distance map from center
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)
        xx, yy = np.meshgrid(x, y)
        
        # Radial distance
        distance = np.sqrt(xx**2 + yy**2)
        
        # Vignette mask
        vignette = 1 - np.clip((distance - radius) / (1 - radius), 0, 1) * strength
        
        if len(image.shape) == 3:
            vignette = vignette[:, :, np.newaxis]
        
        return np.clip(image * vignette, 0, 255).astype(np.uint8)
    
    @staticmethod
    def cross_process(image: np.ndarray) -> np.ndarray:
        """
        Cross-processing effect (film development technique).
        
        Increases contrast, shifts colors toward cyan/yellow.
        """
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
        
        result = image.copy().astype(float)
        
        # Increase contrast with S-curve on each channel
        def s_curve(x, contrast=1.5):
            return 255 / (1 + np.exp(-contrast * (x / 255 - 0.5) * 10))
        
        # Red: shift shadows cyan, highlights yellow
        result[:, :, 0] = s_curve(result[:, :, 0], 1.3)
        # Green: increase contrast
        result[:, :, 1] = s_curve(result[:, :, 1], 1.5)
        # Blue: reduce in highlights, increase in shadows
        result[:, :, 2] = s_curve(result[:, :, 2], 1.2)
        
        # Shift colors
        result[:, :, 0] = np.clip(result[:, :, 0] * 0.9 + 10, 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * 1.1, 0, 255)
        
        return result.astype(np.uint8)
```

### 11.2 Custom Kernel Design

```python
"""
CUSTOM KERNEL DESIGN - Advanced Techniques
"""

class KernelDesigner:
    """Tools for designing and analyzing custom kernels"""
    
    @staticmethod
    def visualize_kernel(kernel: np.ndarray, 
                        title: str = "Kernel") -> np.ndarray:
        """
        Create a visual representation of a kernel.
        Returns image showing kernel values with color coding.
        """
        h, w = kernel.shape
        
        # Normalize for visualization
        k_min, k_max = kernel.min(), kernel.max()
        if k_max - k_min > 0:
            normalized = (kernel - k_min) / (k_max - k_min)
        else:
            normalized = kernel - k_min
        
        # Create colored visualization
        # Blue = negative, White = zero, Red = positive
        visual = np.zeros((h, w, 3), dtype=np.uint8)
        
        for y in range(h):
            for x in range(w):
                val = kernel[y, x]
                if val < 0:
                    # Blue for negative
                    intensity = int(255 * abs(val) / abs(k_min)) if k_min != 0 else 0
                    visual[y, x] = [intensity, 0, 0]  # BGR
                elif val > 0:
                    # Red for positive
                    intensity = int(255 * val / k_max) if k_max != 0 else 0
                    visual[y, x] = [0, 0, intensity]
                else:
                    # Gray for zero
                    visual[y, x] = [128, 128, 128]
        
        # Scale up for visibility
        scale = max(1, 50 // max(h, w))
        visual = np.repeat(np.repeat(visual, scale, axis=0), scale, axis=1)
        
        return visual
    
    @staticmethod
    def analyze_kernel(kernel: np.ndarray) -> dict:
        """
        Analyze kernel properties.
        """
        return {
            'shape': kernel.shape,
            'sum': kernel.sum(),
            'is_normalized': abs(kernel.sum() - 1.0) < 0.001,
            'is_zero_sum': abs(kernel.sum()) < 0.001,
            'min': kernel.min(),
            'max': kernel.max(),
            'is_symmetric': np.allclose(kernel, kernel.T),
            'is_separable': KernelDesigner.check_separable(kernel),
            'center': kernel[kernel.shape[0]//2, kernel.shape[1]//2],
            'frequency_response': 'low-pass' if kernel.sum() > 0 else 
                                 'high-pass' if kernel.sum() == 0 else 'other'
        }
    
    @staticmethod
    def check_separable(kernel: np.ndarray, tolerance: float = 0.01) -> bool:
        """
        Check if a 2D kernel is separable into 1D kernels.
        
        A kernel is separable if it can be expressed as the outer
        product of two 1D vectors: K = v * h^T
        """
        # Use SVD to check separability
        u, s, vh = np.linalg.svd(kernel)
        
        # If only one singular value is significant, kernel is separable
        if len(s) > 1:
            return s[1] / s[0] < tolerance
        return True
    
    @staticmethod
    def decompose_separable(kernel: np.ndarray) -> tuple:
        """
        Decompose separable kernel into 1D kernels.
        
        Returns:
            (horizontal, vertical) 1D kernels or None if not separable
        """
        u, s, vh = np.linalg.svd(kernel)
        
        if len(s) > 1 and s[1] / s[0] > 0.01:
            return None  # Not separable
        
        sqrt_s = np.sqrt(s[0])
        h = vh[0] * sqrt_s
        v = u[:, 0] * sqrt_s
        
        return h, v
    
    @staticmethod
    def normalize_kernel(kernel: np.ndarray, 
                        mode: str = 'sum_one') -> np.ndarray:
        """
        Normalize kernel.
        
        Modes:
            'sum_one': Make sum equal 1 (for blur)
            'sum_zero': Make sum equal 0 (for edge detection)
            'max_one': Make maximum equal 1
        """
        if mode == 'sum_one':
            s = kernel.sum()
            return kernel / s if s != 0 else kernel
        elif mode == 'sum_zero':
            return kernel - kernel.mean()
        elif mode == 'max_one':
            return kernel / kernel.max()
        else:
            return kernel
    
    @staticmethod
    def combine_kernels(k1: np.ndarray, k2: np.ndarray, 
                       method: str = 'cascade') -> np.ndarray:
        """
        Combine two kernels.
        
        Methods:
            'cascade': Convolve kernels (apply sequentially)
            'add': Sum kernels
            'average': Average kernels
        """
        if method == 'cascade':
            return signal.convolve2d(k1, k2, mode='full')
        elif method == 'add':
            # Ensure same size
            max_h = max(k1.shape[0], k2.shape[0])
            max_w = max(k1.shape[1], k2.shape[1])
            
            r1 = np.zeros((max_h, max_w))
            r2 = np.zeros((max_h, max_w))
            
            h1, w1 = k1.shape
            h2, w2 = k2.shape
            
            r1[(max_h-h1)//2:(max_h-h1)//2+h1, 
               (max_w-w1)//2:(max_w-w1)//2+w1] = k1
            r2[(max_h-h2)//2:(max_h-h2)//2+h2,
               (max_w-w2)//2:(max_w-w2)//2+w2] = k2
            
            return r1 + r2
        elif method == 'average':
            combined = KernelDesigner.combine_kernels(k1, k2, 'add')
            return combined / 2
    
    @staticmethod
    def create_directional_kernel(size: int, 
                                  angle_degrees: float,
                                  kernel_type: str = 'gradient') -> np.ndarray:
        """
        Create kernel sensitive to specific direction.
        
        Types:
            'gradient': First derivative in direction
            'second': Second derivative in direction
            'line': Detects lines at angle
        """
        angle = math.radians(angle_degrees)
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        kernel = np.zeros((size, size), dtype=np.float64)
        center = size // 2
        
        if kernel_type == 'gradient':
            for y in range(size):
                for x in range(size):
                    # Project position onto direction vector
                    px, py = x - center, y - center
                    proj = px * dx + py * dy
                    kernel[y, x] = proj
            
            # Normalize
            kernel = kernel / np.sum(np.abs(kernel))
            
        elif kernel_type == 'second':
            for y in range(size):
                for x in range(size):
                    px, py = x - center, y - center
                    proj = px * dx + py * dy
                    # Second derivative approximation
                    kernel[y, x] = proj ** 2 - (size / 4) ** 2
            
            kernel = kernel - kernel.mean()
            
        elif kernel_type == 'line':
            # Line detection at angle
            perp_dx = -dy
            perp_dy = dx
            
            for y in range(size):
                for x in range(size):
                    px, py = x - center, y - center
                    # Distance from center line
                    dist = abs(px * perp_dx + py * perp_dy)
                    kernel[y, x] = 1 if dist < 1 else -1
            
            kernel = kernel - kernel.mean()
        
        return kernel
    
    @staticmethod
    def gabor_kernel(size: int,
                     sigma: float,
                     theta: float,
                     wavelength: float,
                     psi: float = 0,
                     gamma: float = 1.0) -> np.ndarray:
        """
        Generate Gabor kernel - excellent for texture analysis.
        
        Args:
            size: Kernel size
            sigma: Standard deviation of Gaussian
            theta: Orientation in radians
            wavelength: Wavelength of sinusoidal factor
            psi: Phase offset
            gamma: Spatial aspect ratio
        """
        x = np.arange(size) - size // 2
        y = np.arange(size) - size // 2
        xx, yy = np.meshgrid(x, y)
        
        # Rotation
        x_theta = xx * np.cos(theta) + yy * np.sin(theta)
        y_theta = -xx * np.sin(theta) + yy * np.cos(theta)
        
        # Gabor formula
        gaussian = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
        sinusoid = np.cos(2 * np.pi * x_theta / wavelength + psi)
        
        return gaussian * sinusoid
    
    @staticmethod
    def gabor_filter_bank(size: int = 31,
                          num_orientations: int = 8,
                          num_scales: int = 4,
                          min_wavelength: float = 4,
                          wavelength_factor: float = 2) -> list:
        """
        Generate bank of Gabor filters for texture analysis.
        
        Returns list of kernels at different orientations and scales.
        """
        kernels = []
        
        for scale in range(num_scales):
            wavelength = min_wavelength * (wavelength_factor ** scale)
            sigma = wavelength / 2
            
            for orientation in range(num_orientations):
                theta = orientation * np.pi / num_orientations
                
                kernel = KernelDesigner.gabor_kernel(
                    size=size,
                    sigma=sigma,
                    theta=theta,
                    wavelength=wavelength
                )
                
                kernels.append({
                    'kernel': kernel,
                    'orientation': theta,
                    'wavelength': wavelength,
                    'sigma': sigma
                })
        
        return kernels
```

---

## Chapter 12: Frequency Domain and FFT

### 12.1 Fourier Transform Fundamentals

The Fourier Transform decomposes an image into its frequency components - a powerful tool for filtering, analysis, and special effects.

```python
"""
FREQUENCY DOMAIN PROCESSING - COMPREHENSIVE GUIDE

The Fourier Transform reveals the hidden frequency structure
of images, enabling powerful filtering techniques impossible
in the spatial domain.
"""

import numpy as np
from PIL import Image
from scipy import fft, ndimage
import math

# ═══════════════════════════════════════════════════════════════
# FOURIER TRANSFORM BASICS
# ═══════════════════════════════════════════════════════════════

class FourierProcessor:
    """Complete Fourier transform processing toolkit"""
    
    @staticmethod
    def forward_transform(image: np.ndarray) -> np.ndarray:
        """
        Compute 2D Discrete Fourier Transform.
        
        The DFT converts spatial domain to frequency domain.
        Each point in the result represents a specific frequency
        component's magnitude and phase.
        
        Returns complex array of shape (H, W) or (H, W, C)
        """
        if len(image.shape) == 2:
            return fft.fft2(image)
        else:
            # Process each channel
            result = np.zeros(image.shape, dtype=complex)
            for c in range(image.shape[2]):
                result[:, :, c] = fft.fft2(image[:, :, c])
            return result
    
    @staticmethod
    def inverse_transform(freq_data: np.ndarray) -> np.ndarray:
        """
        Compute Inverse 2D DFT - convert back to spatial domain.
        """
        if len(freq_data.shape) == 2:
            return np.real(fft.ifft2(freq_data))
        else:
            result = np.zeros(freq_data.shape, dtype=np.float64)
            for c in range(freq_data.shape[2]):
                result[:, :, c] = np.real(fft.ifft2(freq_data[:, :, c]))
            return result
    
    @staticmethod
    def shift_to_center(freq_data: np.ndarray) -> np.ndarray:
        """
        Shift zero-frequency component to center of spectrum.
        
        By default, FFT puts DC (zero frequency) at corners.
        Shifting to center makes visualization and filtering easier.
        """
        return fft.fftshift(freq_data)
    
    @staticmethod
    def shift_from_center(freq_data: np.ndarray) -> np.ndarray:
        """
        Inverse of shift_to_center - put DC back at corners.
        Required before inverse transform if you shifted.
        """
        return fft.ifftshift(freq_data)
    
    @staticmethod
    def get_magnitude(freq_data: np.ndarray) -> np.ndarray:
        """
        Get magnitude spectrum (absolute value).
        
        Magnitude tells us the strength of each frequency.
        """
        return np.abs(freq_data)
    
    @staticmethod
    def get_phase(freq_data: np.ndarray) -> np.ndarray:
        """
        Get phase spectrum (angle).
        
        Phase tells us the position/alignment of each frequency.
        Phase carries more structural information than magnitude!
        """
        return np.angle(freq_data)
    
    @staticmethod
    def visualize_spectrum(freq_data: np.ndarray, 
                          log_scale: bool = True) -> np.ndarray:
        """
        Create viewable image of frequency spectrum.
        
        The raw magnitude values span a huge range, so we typically
        use logarithmic scaling for visualization.
        """
        # Shift to center
        centered = FourierProcessor.shift_to_center(freq_data)
        
        # Get magnitude
        magnitude = np.abs(centered)
        
        if len(magnitude.shape) == 3:
            magnitude = np.mean(magnitude, axis=2)
        
        # Log scale for better visualization
        if log_scale:
            magnitude = np.log(1 + magnitude)
        
        # Normalize to 0-255
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
        magnitude = (magnitude * 255).astype(np.uint8)
        
        return magnitude
    
    @staticmethod
    def reconstruct_from_magnitude_phase(magnitude: np.ndarray,
                                        phase: np.ndarray) -> np.ndarray:
        """
        Reconstruct complex frequency data from magnitude and phase.
        
        F = magnitude * e^(i*phase)
        """
        return magnitude * np.exp(1j * phase)


# ═══════════════════════════════════════════════════════════════
# FREQUENCY DOMAIN FILTERING
# ═══════════════════════════════════════════════════════════════

class FrequencyFilters:
    """Filters that operate in frequency domain"""
    
    @staticmethod
    def create_distance_grid(shape: tuple) -> np.ndarray:
        """
        Create grid of distances from center.
        Used as basis for many frequency filters.
        """
        h, w = shape[:2]
        y = np.arange(h) - h // 2
        x = np.arange(w) - w // 2
        xx, yy = np.meshgrid(x, y)
        return np.sqrt(xx**2 + yy**2)
    
    # ─────────────────────────────────────────────────────────
    # IDEAL FILTERS (Sharp cutoff)
    # ─────────────────────────────────────────────────────────
    
    @staticmethod
    def ideal_lowpass(shape: tuple, cutoff: float) -> np.ndarray:
        """
        Ideal low-pass filter - passes frequencies below cutoff.
        
        Sharp cutoff causes ringing artifacts (Gibbs phenomenon)
        but useful for understanding frequency concepts.
        
        Args:
            shape: (height, width) of image
            cutoff: Cutoff frequency (radius in frequency domain)
        """
        distance = FrequencyFilters.create_distance_grid(shape)
        return (distance <= cutoff).astype(np.float64)
    
    @staticmethod
    def ideal_highpass(shape: tuple, cutoff: float) -> np.ndarray:
        """
        Ideal high-pass filter - passes frequencies above cutoff.
        
        Preserves edges and fine detail, removes smooth regions.
        """
        return 1 - FrequencyFilters.ideal_lowpass(shape, cutoff)
    
    @staticmethod
    def ideal_bandpass(shape: tuple, 
                       low_cutoff: float, 
                       high_cutoff: float) -> np.ndarray:
        """
        Ideal band-pass filter - passes frequencies in range.
        """
        distance = FrequencyFilters.create_distance_grid(shape)
        return ((distance >= low_cutoff) & (distance <= high_cutoff)).astype(np.float64)
    
    @staticmethod
    def ideal_bandreject(shape: tuple,
                         low_cutoff: float,
                         high_cutoff: float) -> np.ndarray:
        """
        Ideal band-reject (notch) filter - removes frequency range.
        """
        return 1 - FrequencyFilters.ideal_bandpass(shape, low_cutoff, high_cutoff)
    
    # ─────────────────────────────────────────────────────────
    # BUTTERWORTH FILTERS (Smooth transition)
    # ─────────────────────────────────────────────────────────
    
    @staticmethod
    def butterworth_lowpass(shape: tuple, 
                           cutoff: float, 
                           order: int = 2) -> np.ndarray:
        """
        Butterworth low-pass filter - smooth transition.
        
        Higher order = sharper transition (more like ideal filter).
        Order 2 is a good default - smooth enough to avoid ringing.
        
        H(u,v) = 1 / (1 + (D/D0)^(2n))
        
        Args:
            shape: (height, width)
            cutoff: Cutoff frequency D0
            order: Filter order n
        """
        distance = FrequencyFilters.create_distance_grid(shape)
        return 1 / (1 + (distance / cutoff) ** (2 * order))
    
    @staticmethod
    def butterworth_highpass(shape: tuple,
                            cutoff: float,
                            order: int = 2) -> np.ndarray:
        """
        Butterworth high-pass filter.
        
        H(u,v) = 1 / (1 + (D0/D)^(2n))
        """
        distance = FrequencyFilters.create_distance_grid(shape)
        # Avoid division by zero at center
        distance = np.maximum(distance, 0.001)
        return 1 / (1 + (cutoff / distance) ** (2 * order))
    
    @staticmethod
    def butterworth_bandpass(shape: tuple,
                            center_freq: float,
                            bandwidth: float,
                            order: int = 2) -> np.ndarray:
        """
        Butterworth band-pass filter.
        """
        distance = FrequencyFilters.create_distance_grid(shape)
        w = bandwidth / 2
        
        # Avoid issues at DC
        distance = np.maximum(distance, 0.001)
        
        return 1 / (1 + ((distance**2 - center_freq**2) / 
                        (distance * w)) ** (2 * order))
    
    # ─────────────────────────────────────────────────────────
    # GAUSSIAN FILTERS (Smoothest transition)
    # ─────────────────────────────────────────────────────────
    
    @staticmethod
    def gaussian_lowpass(shape: tuple, cutoff: float) -> np.ndarray:
        """
        Gaussian low-pass filter - smoothest possible transition.
        
        No ringing artifacts. The Gaussian in frequency domain
        corresponds to a Gaussian in spatial domain.
        
        H(u,v) = e^(-D²/(2*D0²))
        """
        distance = FrequencyFilters.create_distance_grid(shape)
        return np.exp(-(distance**2) / (2 * cutoff**2))
    
    @staticmethod
    def gaussian_highpass(shape: tuple, cutoff: float) -> np.ndarray:
        """
        Gaussian high-pass filter.
        """
        return 1 - FrequencyFilters.gaussian_lowpass(shape, cutoff)
    
    @staticmethod
    def gaussian_bandpass(shape: tuple,
                         center_freq: float,
                         bandwidth: float) -> np.ndarray:
        """
        Gaussian band-pass filter.
        """
        distance = FrequencyFilters.create_distance_grid(shape)
        return np.exp(-((distance - center_freq)**2) / (2 * (bandwidth/2)**2))
    
    # ─────────────────────────────────────────────────────────
    # DIRECTIONAL FILTERS
    # ─────────────────────────────────────────────────────────
    
    @staticmethod
    def directional_filter(shape: tuple,
                          angle: float,
                          width: float = 10) -> np.ndarray:
        """
        Pass frequencies along a specific direction.
        
        Useful for enhancing or detecting features at specific orientations.
        
        Args:
            shape: (height, width)
            angle: Direction in degrees (0 = horizontal)
            width: Angular width of passband in degrees
        """
        h, w = shape[:2]
        y = np.arange(h) - h // 2
        x = np.arange(w) - w // 2
        xx, yy = np.meshgrid(x, y)
        
        # Calculate angle at each point
        angles = np.degrees(np.arctan2(yy, xx))
        
        # Create angular passband (considering both directions)
        angle_diff_1 = np.abs(angles - angle)
        angle_diff_2 = np.abs(angles - (angle + 180))
        angle_diff_3 = np.abs(angles - (angle - 180))
        
        angle_diff = np.minimum(angle_diff_1, np.minimum(angle_diff_2, angle_diff_3))
        
        # Gaussian falloff
        return np.exp(-(angle_diff**2) / (2 * (width/2)**2))
    
    @staticmethod
    def wedge_filter(shape: tuple,
                    angle: float,
                    angular_width: float,
                    min_freq: float = 0,
                    max_freq: float = None) -> np.ndarray:
        """
        Wedge-shaped filter in frequency domain.
        
        Combines directional and radial filtering.
        """
        h, w = shape[:2]
        if max_freq is None:
            max_freq = min(h, w) / 2
        
        # Directional component
        directional = FrequencyFilters.directional_filter(shape, angle, angular_width)
        
        # Radial component
        distance = FrequencyFilters.create_distance_grid(shape)
        radial = ((distance >= min_freq) & (distance <= max_freq)).astype(float)
        
        return directional * radial
    
    # ─────────────────────────────────────────────────────────
    # NOTCH FILTERS
    # ─────────────────────────────────────────────────────────
    
    @staticmethod
    def notch_filter(shape: tuple,
                    points: list,
                    radius: float,
                    filter_type: str = 'butterworth',
                    order: int = 2) -> np.ndarray:
        """
        Notch filter - removes specific frequencies.
        
        Useful for removing periodic noise patterns.
        
        Args:
            shape: (height, width)
            points: List of (x, y) frequency coordinates to notch
            radius: Radius of notch
            filter_type: 'ideal', 'butterworth', or 'gaussian'
        """
        h, w = shape[:2]
        result = np.ones((h, w), dtype=np.float64)
        
        y = np.arange(h) - h // 2
        x = np.arange(w) - w // 2
        xx, yy = np.meshgrid(x, y)
        
        for px, py in points:
            # Distance from notch point
            d1 = np.sqrt((xx - px)**2 + (yy - py)**2)
            # Distance from symmetric point (notches usually come in pairs)
            d2 = np.sqrt((xx + px)**2 + (yy + py)**2)
            
            if filter_type == 'ideal':
                notch1 = (d1 > radius).astype(float)
                notch2 = (d2 > radius).astype(float)
            elif filter_type == 'butterworth':
                d1 = np.maximum(d1, 0.001)
                d2 = np.maximum(d2, 0.001)
                notch1 = 1 / (1 + (radius / d1) ** (2 * order))
                notch2 = 1 / (1 + (radius / d2) ** (2 * order))
            else:  # gaussian
                notch1 = 1 - np.exp(-(d1**2) / (2 * radius**2))
                notch2 = 1 - np.exp(-(d2**2) / (2 * radius**2))
            
            result *= notch1 * notch2
        
        return result
    
    # ─────────────────────────────────────────────────────────
    # HOMOMORPHIC FILTER
    # ─────────────────────────────────────────────────────────
    
    @staticmethod
    def homomorphic_filter(shape: tuple,
                          gamma_low: float = 0.5,
                          gamma_high: float = 2.0,
                          cutoff: float = 30,
                          c: float = 1.0) -> np.ndarray:
        """
        Homomorphic filter for illumination normalization.
        
        Based on the illumination-reflectance model:
        Image = Illumination * Reflectance
        
        - Illumination varies slowly (low frequency)
        - Reflectance varies quickly (high frequency)
        
        By attenuating low frequencies and boosting high frequencies,
        we can reduce illumination variations while enhancing detail.
        
        H(u,v) = (gamma_H - gamma_L) * (1 - e^(-c*(D²/D0²))) + gamma_L
        """
        distance = FrequencyFilters.create_distance_grid(shape)
        
        # Exponential high-pass shape
        h = (gamma_high - gamma_low) * \
            (1 - np.exp(-c * (distance**2) / (cutoff**2))) + gamma_low
        
        return h


# ═══════════════════════════════════════════════════════════════
# FREQUENCY DOMAIN PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════

class FrequencyProcessor:
    """High-level frequency domain processing"""
    
    @staticmethod
    def apply_filter(image: np.ndarray, 
                    filter_mask: np.ndarray,
                    preserve_dc: bool = True) -> np.ndarray:
        """
        Apply frequency domain filter to image.
        
        Args:
            image: Input image
            filter_mask: Filter to apply (should be same size as image)
            preserve_dc: If True, always keep DC component (average brightness)
        """
        # Forward transform
        freq = FourierProcessor.forward_transform(image.astype(float))
        
        # Shift to center for filtering
        freq_shifted = FourierProcessor.shift_to_center(freq)
        
        # Ensure filter is right shape
        if len(image.shape) == 3 and len(filter_mask.shape) == 2:
            filter_3d = filter_mask[:, :, np.newaxis]
        else:
            filter_3d = filter_mask
        
        # Preserve DC if requested
        if preserve_dc and len(filter_mask.shape) == 2:
            h, w = filter_mask.shape
            filter_mask[h//2, w//2] = 1.0
        
        # Apply filter
        filtered = freq_shifted * filter_3d
        
        # Shift back and inverse transform
        filtered_unshifted = FourierProcessor.shift_from_center(filtered)
        result = FourierProcessor.inverse_transform(filtered_unshifted)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def blur_frequency(image: np.ndarray, 
                      strength: float = 30) -> np.ndarray:
        """
        Blur image using frequency domain low-pass filter.
        
        Equivalent to Gaussian blur but can be faster for large kernels.
        """
        h, w = image.shape[:2]
        filter_mask = FrequencyFilters.gaussian_lowpass((h, w), strength)
        return FrequencyProcessor.apply_filter(image, filter_mask)
    
    @staticmethod
    def sharpen_frequency(image: np.ndarray,
                         boost: float = 1.5,
                         radius: float = 30) -> np.ndarray:
        """
        Sharpen image using high-frequency boost.
        
        Adds scaled high-frequency content back to image.
        """
        h, w = image.shape[:2]
        
        # Create high-boost filter
        highpass = FrequencyFilters.gaussian_highpass((h, w), radius)
        filter_mask = 1 + boost * highpass
        
        return FrequencyProcessor.apply_filter(image, filter_mask)
    
    @staticmethod
    def remove_periodic_noise(image: np.ndarray,
                             noise_points: list = None,
                             auto_detect: bool = True,
                             threshold: float = 0.8) -> np.ndarray:
        """
        Remove periodic noise patterns using notch filtering.
        
        Periodic noise (like screen door patterns, moiré) appears
        as bright spots in the frequency spectrum.
        
        Args:
            noise_points: Known noise frequency locations
            auto_detect: If True, automatically find noise peaks
            threshold: Detection threshold for auto-detection
        """
        h, w = image.shape[:2]
        
        if auto_detect and noise_points is None:
            # Automatically detect noise peaks
            freq = FourierProcessor.forward_transform(
                np.mean(image, axis=2) if len(image.shape) == 3 else image
            )
            freq_shifted = FourierProcessor.shift_to_center(freq)
            magnitude = np.abs(freq_shifted)
            
            # Normalize magnitude
            mag_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
            
            # Find peaks (excluding DC and immediate neighbors)
            noise_points = []
            for y in range(5, h - 5):
                for x in range(5, w - 5):
                    # Skip center region (DC)
                    if abs(y - h//2) < 5 and abs(x - w//2) < 5:
                        continue
                    
                    # Check if local maximum and above threshold
                    local_region = mag_norm[y-2:y+3, x-2:x+3]
                    if mag_norm[y, x] == local_region.max() and mag_norm[y, x] > threshold:
                        # Store relative to center
                        noise_points.append((x - w//2, y - h//2))
        
        if not noise_points:
            return image
        
        # Create notch filter
        notch = FrequencyFilters.notch_filter(
            (h, w), noise_points, radius=10, filter_type='butterworth'
        )
        
        return FrequencyProcessor.apply_filter(image, notch)
    
    @staticmethod
    def enhance_edges_frequency(image: np.ndarray,
                                enhancement: float = 1.0) -> np.ndarray:
        """
        Enhance edges using Laplacian in frequency domain.
        
        The Laplacian in frequency domain is simply -(u² + v²),
        which naturally amplifies high frequencies.
        """
        h, w = image.shape[:2]
        
        y = np.arange(h) - h // 2
        x = np.arange(w) - w // 2
        xx, yy = np.meshgrid(x, y)
        
        # Laplacian in frequency domain
        laplacian = -(xx**2 + yy**2)
        
        # Normalize
        laplacian = laplacian / np.abs(laplacian).max()
        
        # Create enhancement filter: 1 - enhancement * laplacian
        filter_mask = 1 - enhancement * laplacian
        
        return FrequencyProcessor.apply_filter(image, filter_mask)
    
    @staticmethod
    def homomorphic_enhance(image: np.ndarray,
                           gamma_low: float = 0.5,
                           gamma_high: float = 2.0) -> np.ndarray:
        """
        Apply homomorphic filtering for illumination normalization.
        
        Great for images with varying lighting conditions.
        """
        # Convert to float and take log
        img_float = image.astype(float) + 1  # Avoid log(0)
        
        if len(img_float.shape) == 3:
            img_log = np.log(img_float)
        else:
            img_log = np.log(img_float)
        
        h, w = image.shape[:2]
        
        # Create homomorphic filter
        homo_filter = FrequencyFilters.homomorphic_filter(
            (h, w), gamma_low, gamma_high
        )
        
        # Apply in frequency domain
        freq = FourierProcessor.forward_transform(img_log)
        freq_shifted = FourierProcessor.shift_to_center(freq)
        
        if len(image.shape) == 3:
            homo_filter = homo_filter[:, :, np.newaxis]
        
        filtered = freq_shifted * homo_filter
        
        filtered_unshifted = FourierProcessor.shift_from_center(filtered)
        result_log = FourierProcessor.inverse_transform(filtered_unshifted)
        
        # Exponentiate to reverse log
        result = np.exp(result_log) - 1
        
        return np.clip(result, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# FREQUENCY ANALYSIS AND ARTISTIC EFFECTS
# ═══════════════════════════════════════════════════════════════

class FrequencyArtist:
    """Creative uses of frequency domain processing"""
    
    @staticmethod
    def phase_scramble(image: np.ndarray, 
                       amount: float = 1.0) -> np.ndarray:
        """
        Scramble phase while keeping magnitude.
        
        Creates surreal, dreamlike images that preserve
        overall brightness distribution but lose structure.
        
        Fun fact: Human vision is more sensitive to phase than magnitude!
        """
        freq = FourierProcessor.forward_transform(image.astype(float))
        
        magnitude = np.abs(freq)
        phase = np.angle(freq)
        
        # Random phase
        random_phase = np.random.uniform(-np.pi, np.pi, phase.shape)
        
        # Blend original and random phase
        blended_phase = phase * (1 - amount) + random_phase * amount
        
        # Reconstruct
        new_freq = magnitude * np.exp(1j * blended_phase)
        result = FourierProcessor.inverse_transform(new_freq)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def hybrid_image(image1: np.ndarray, 
                    image2: np.ndarray,
                    cutoff: float = 30) -> np.ndarray:
        """
        Create hybrid image that changes with viewing distance.
        
        Close up: See high frequencies of image2
        Far away: See low frequencies of image1
        
        Famous examples: Einstein-Monroe, Angry-Neutral face
        """
        assert image1.shape == image2.shape
        h, w = image1.shape[:2]
        
        # Low frequencies from image1
        lowpass = FrequencyFilters.gaussian_lowpass((h, w), cutoff)
        low_freq = FrequencyProcessor.apply_filter(image1, lowpass)
        
        # High frequencies from image2
        highpass = FrequencyFilters.gaussian_highpass((h, w), cutoff)
        high_freq = FrequencyProcessor.apply_filter(image2, highpass)
        
        # Combine
        return np.clip(low_freq.astype(float) + high_freq.astype(float) - 128, 
                      0, 255).astype(np.uint8)
    
    @staticmethod
    def frequency_painting(image: np.ndarray,
                          num_bands: int = 5,
                          colors: list = None) -> np.ndarray:
        """
        Color-code different frequency bands.
        
        Creates artistic visualization showing frequency content.
        """
        if colors is None:
            # Rainbow colors for bands
            colors = [
                (255, 0, 0),    # Red - low freq
                (255, 255, 0),  # Yellow
                (0, 255, 0),    # Green
                (0, 255, 255),  # Cyan
                (0, 0, 255),    # Blue - high freq
            ]
        
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)
        
        h, w = gray.shape
        max_freq = min(h, w) / 2
        band_width = max_freq / num_bands
        
        result = np.zeros((h, w, 3), dtype=np.float64)
        
        for i in range(num_bands):
            low = i * band_width
            high = (i + 1) * band_width
            
            # Create band-pass filter
            bandpass = FrequencyFilters.gaussian_bandpass((h, w), (low + high) / 2, band_width)
            
            # Extract band
            freq = FourierProcessor.forward_transform(gray)
            freq_shifted = FourierProcessor.shift_to_center(freq)
            filtered = freq_shifted * bandpass
            band = np.abs(FourierProcessor.inverse_transform(
                FourierProcessor.shift_from_center(filtered)
            ))
            
            # Normalize band
            if band.max() > 0:
                band = band / band.max()
            
            # Add colored band to result
            color = colors[i % len(colors)]
            for c in range(3):
                result[:, :, c] += band * color[c]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def spectrum_art(image: np.ndarray,
                    rotation: float = 0,
                    scale: float = 1.0,
                    colormap: str = 'viridis') -> np.ndarray:
        """
        Transform frequency spectrum into artistic image.
        
        The spectrum itself becomes the art.
        """
        # Get spectrum
        freq = FourierProcessor.forward_transform(
            np.mean(image, axis=2) if len(image.shape) == 3 else image
        )
        centered = FourierProcessor.shift_to_center(freq)
        magnitude = np.log(1 + np.abs(centered))
        
        # Apply transformations
        if rotation != 0:
            magnitude = ndimage.rotate(magnitude, rotation, reshape=False)
        
        if scale != 1.0:
            magnitude = ndimage.zoom(magnitude, scale)
            # Crop or pad to original size
            h, w = image.shape[:2]
            mh, mw = magnitude.shape
            if mh > h:
                start = (mh - h) // 2
                magnitude = magnitude[start:start+h, :]
            if mw > w:
                start = (mw - w) // 2
                magnitude = magnitude[:, start:start+w]
        
        # Normalize
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
        
        # Apply colormap
        h, w = magnitude.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        if colormap == 'viridis':
            result[:, :, 0] = (magnitude * 68 + (1-magnitude) * 68).astype(np.uint8)
            result[:, :, 1] = (magnitude * 1 + (1-magnitude) * 191).astype(np.uint8)
            result[:, :, 2] = (magnitude * 84 + (1-magnitude) * 255).astype(np.uint8)
        elif colormap == 'plasma':
            result[:, :, 0] = (magnitude * 240 + (1-magnitude) * 13).astype(np.uint8)
            result[:, :, 1] = (magnitude * 249 + (1-magnitude) * 8).astype(np.uint8)
            result[:, :, 2] = (magnitude * 33 + (1-magnitude) * 135).astype(np.uint8)
        elif colormap == 'magma':
            result[:, :, 0] = (magnitude * 252 + (1-magnitude) * 0).astype(np.uint8)
            result[:, :, 1] = (magnitude * 253 + (1-magnitude) * 0).astype(np.uint8)
            result[:, :, 2] = (magnitude * 191 + (1-magnitude) * 4).astype(np.uint8)
        else:  # grayscale
            val = (magnitude * 255).astype(np.uint8)
            result[:, :, 0] = val
            result[:, :, 1] = val
            result[:, :, 2] = val
        
        return result
    
    @staticmethod
    def frequency_transfer(source: np.ndarray,
                          target: np.ndarray,
                          bands_to_transfer: list = ['low', 'high']) -> np.ndarray:
        """
        Transfer frequency content between images.
        
        Example: Take low frequencies from painting, high from photo
        to create painterly version of photo.
        """
        assert source.shape == target.shape
        h, w = source.shape[:2]
        
        # Get frequency data
        src_freq = FourierProcessor.forward_transform(source.astype(float))
        tgt_freq = FourierProcessor.forward_transform(target.astype(float))
        
        src_shifted = FourierProcessor.shift_to_center(src_freq)
        tgt_shifted = FourierProcessor.shift_to_center(tgt_freq)
        
        # Create result starting with target
        result_freq = tgt_shifted.copy()
        
        cutoff = min(h, w) / 4
        
        if 'low' in bands_to_transfer:
            lowpass = FrequencyFilters.gaussian_lowpass((h, w), cutoff)
            if len(source.shape) == 3:
                lowpass = lowpass[:, :, np.newaxis]
            result_freq = result_freq * (1 - lowpass) + src_shifted * lowpass
        
        if 'high' in bands_to_transfer:
            highpass = FrequencyFilters.gaussian_highpass((h, w), cutoff)
            if len(source.shape) == 3:
                highpass = highpass[:, :, np.newaxis]
            result_freq = result_freq * (1 - highpass) + src_shifted * highpass
        
        # Transform back
        result = FourierProcessor.inverse_transform(
            FourierProcessor.shift_from_center(result_freq)
        )
        
        return np.clip(result, 0, 255).astype(np.uint8)
```

### 12.2 Practical FFT Applications

```python
"""
PRACTICAL FFT APPLICATIONS

Real-world uses of frequency domain processing
"""

class FFTApplications:
    """Practical frequency domain applications"""
    
    @staticmethod
    def deconvolution(blurred: np.ndarray,
                     psf: np.ndarray,
                     noise_variance: float = 0.01) -> np.ndarray:
        """
        Wiener deconvolution - restore blurred image.
        
        If we know the Point Spread Function (blur kernel),
        we can theoretically reverse the blur in frequency domain.
        
        Wiener filter balances deconvolution with noise suppression:
        H_wiener = conj(H) / (|H|² + noise/signal)
        
        Args:
            blurred: Blurred input image
            psf: Point Spread Function (blur kernel)
            noise_variance: Estimated noise variance
        """
        if len(blurred.shape) == 3:
            blurred = np.mean(blurred, axis=2)
        
        # Pad PSF to image size
        h, w = blurred.shape
        psf_padded = np.zeros((h, w))
        ph, pw = psf.shape
        psf_padded[:ph, :pw] = psf
        # Shift so center of PSF is at origin
        psf_padded = np.roll(np.roll(psf_padded, -ph//2, axis=0), -pw//2, axis=1)
        
        # Transform
        blurred_freq = fft.fft2(blurred)
        psf_freq = fft.fft2(psf_padded)
        
        # Wiener deconvolution
        psf_conj = np.conj(psf_freq)
        psf_power = np.abs(psf_freq)**2
        
        # Regularization to avoid division by zero
        wiener = psf_conj / (psf_power + noise_variance)
        
        # Apply deconvolution
        result_freq = blurred_freq * wiener
        result = np.real(fft.ifft2(result_freq))
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def texture_synthesis_fft(texture: np.ndarray,
                             output_size: tuple) -> np.ndarray:
        """
        Simple texture synthesis using FFT.
        
        Preserves frequency statistics of original texture
        while generating new random phase.
        """
        # Resize texture to output size
        from PIL import Image
        texture_pil = Image.fromarray(texture)
        texture_resized = np.array(texture_pil.resize(output_size[::-1]))
        
        # Get magnitude spectrum
        freq = FourierProcessor.forward_transform(texture_resized.astype(float))
        magnitude = np.abs(freq)
        
        # Random phase
        random_phase = np.random.uniform(-np.pi, np.pi, freq.shape)
        
        # Reconstruct with random phase
        new_freq = magnitude * np.exp(1j * random_phase)
        result = FourierProcessor.inverse_transform(new_freq)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def image_registration_fft(image1: np.ndarray,
                              image2: np.ndarray) -> tuple:
        """
        Find translation offset between two images using phase correlation.
        
        Much faster than spatial domain correlation for large images.
        
        Returns:
            (dx, dy): Translation offset
        """
        if len(image1.shape) == 3:
            image1 = np.mean(image1, axis=2)
        if len(image2.shape) == 3:
            image2 = np.mean(image2, axis=2)
        
        # Compute cross-power spectrum
        f1 = fft.fft2(image1)
        f2 = fft.fft2(image2)
        
        cross_power = (f1 * np.conj(f2)) / np.abs(f1 * np.conj(f2))
        
        # Inverse transform gives correlation peak
        correlation = np.real(fft.ifft2(cross_power))
        
        # Find peak location
        peak = np.unravel_index(np.argmax(correlation), correlation.shape)
        
        # Convert to offset (handle wrap-around)
        h, w = image1.shape
        dy = peak[0] if peak[0] < h // 2 else peak[0] - h
        dx = peak[1] if peak[1] < w // 2 else peak[1] - w
        
        return dx, dy
    
    @staticmethod
    def frequency_watermark(image: np.ndarray,
                           watermark: str,
                           strength: float = 0.1) -> np.ndarray:
        """
        Embed invisible watermark in frequency domain.
        
        Watermark is embedded in mid-frequencies where it's
        harder to detect but survives image processing.
        """
        h, w = image.shape[:2]
        
        # Create watermark pattern
        wm_size = min(h, w) // 4
        wm_img = np.zeros((wm_size, wm_size))
        
        # Simple text watermark as binary pattern
        for i, char in enumerate(watermark):
            if i >= wm_size * wm_size:
                break
            y = i // wm_size
            x = i % wm_size
            wm_img[y, x] = ord(char) / 255.0
        
        # Resize to match mid-frequency region
        from PIL import Image as PILImage
        wm_pil = PILImage.fromarray((wm_img * 255).astype(np.uint8))
        wm_resized = np.array(wm_pil.resize((w // 4, h // 4))) / 255.0
        
        # Embed in frequency domain
        freq = FourierProcessor.forward_transform(image.astype(float))
        freq_shifted = FourierProcessor.shift_to_center(freq)
        
        # Place watermark in mid-frequency region (not DC, not edges)
        cy, cx = h // 2, w // 2
        wh, ww = wm_resized.shape
        
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                freq_shifted[cy+10:cy+10+wh, cx+10:cx+10+ww, c] += wm_resized * strength * 1000
        else:
            freq_shifted[cy+10:cy+10+wh, cx+10:cx+10+ww] += wm_resized * strength * 1000
        
        # Transform back
        result = FourierProcessor.inverse_transform(
            FourierProcessor.shift_from_center(freq_shifted)
        )
        
        return np.clip(result, 0, 255).astype(np.uint8)
```

---

## Chapter 13: Noise Functions and Randomness

### 13.1 The Mathematics of Procedural Noise

Noise functions are the foundation of procedural content generation. They provide controlled randomness that creates natural-looking patterns.

```python
"""
NOISE FUNCTIONS - COMPREHENSIVE IMPLEMENTATION

From basic random noise to advanced multi-octave Perlin and Simplex noise.
These are the building blocks of procedural art.
"""

import numpy as np
import math
from typing import Tuple, List, Optional, Callable

# ═══════════════════════════════════════════════════════════════
# BASIC NOISE TYPES
# ═══════════════════════════════════════════════════════════════

class BasicNoise:
    """Fundamental noise generation"""
    
    @staticmethod
    def white_noise(width: int, height: int, 
                   channels: int = 1,
                   seed: Optional[int] = None) -> np.ndarray:
        """
        Pure random white noise.
        
        Every pixel is completely independent.
        No spatial coherence - looks like TV static.
        
        Use cases:
        - Dithering
        - Starting point for other effects
        - Film grain base
        """
        if seed is not None:
            np.random.seed(seed)
        
        if channels == 1:
            return np.random.randint(0, 256, (height, width), dtype=np.uint8)
        else:
            return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
    
    @staticmethod
    def uniform_noise(width: int, height: int,
                     low: float = 0.0,
                     high: float = 1.0,
                     seed: Optional[int] = None) -> np.ndarray:
        """
        Uniform distribution noise (float values).
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(low, high, (height, width))
    
    @staticmethod
    def gaussian_noise(width: int, height: int,
                      mean: float = 0.5,
                      std: float = 0.2,
                      seed: Optional[int] = None) -> np.ndarray:
        """
        Gaussian (normal) distribution noise.
        
        Values cluster around mean with bell curve distribution.
        More natural looking than uniform noise.
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(mean, std, (height, width))
    
    @staticmethod
    def salt_and_pepper(width: int, height: int,
                       amount: float = 0.05,
                       seed: Optional[int] = None) -> np.ndarray:
        """
        Salt and pepper noise - random black and white pixels.
        
        Simulates dead pixels or impulse noise.
        """
        if seed is not None:
            np.random.seed(seed)
        
        result = np.full((height, width), 128, dtype=np.uint8)
        
        # Salt (white)
        num_salt = int(width * height * amount / 2)
        salt_coords = (np.random.randint(0, height, num_salt),
                      np.random.randint(0, width, num_salt))
        result[salt_coords] = 255
        
        # Pepper (black)
        num_pepper = int(width * height * amount / 2)
        pepper_coords = (np.random.randint(0, height, num_pepper),
                        np.random.randint(0, width, num_pepper))
        result[pepper_coords] = 0
        
        return result
    
    @staticmethod
    def poisson_noise(width: int, height: int,
                     lam: float = 50,
                     seed: Optional[int] = None) -> np.ndarray:
        """
        Poisson noise - simulates photon counting noise.
        
        Common in low-light photography and medical imaging.
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.poisson(lam, (height, width)).astype(np.float64)
    
    @staticmethod
    def blue_noise(width: int, height: int,
                  num_points: int = 1000,
                  seed: Optional[int] = None) -> np.ndarray:
        """
        Blue noise via Poisson disk sampling.
        
        Points are randomly placed but maintain minimum distance.
        Looks more natural than white noise, no clumping.
        
        Used for:
        - Dithering
        - Stippling effects
        - Natural-looking distributions
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Calculate minimum distance based on desired point count
        min_dist = math.sqrt((width * height) / (num_points * math.pi * 2))
        
        points = []
        active = []
        
        # Start with random point
        x, y = np.random.uniform(0, width), np.random.uniform(0, height)
        points.append((x, y))
        active.append((x, y))
        
        # Build grid for fast neighbor lookup
        cell_size = min_dist / math.sqrt(2)
        grid_w = int(math.ceil(width / cell_size))
        grid_h = int(math.ceil(height / cell_size))
        grid = [[None for _ in range(grid_w)] for _ in range(grid_h)]
        
        def grid_coords(px, py):
            return int(py / cell_size), int(px / cell_size)
        
        def is_valid(px, py):
            if px < 0 or px >= width or py < 0 or py >= height:
                return False
            
            gy, gx = grid_coords(px, py)
            
            # Check neighboring cells
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < grid_h and 0 <= nx < grid_w:
                        neighbor = grid[ny][nx]
                        if neighbor:
                            dist = math.sqrt((px - neighbor[0])**2 + (py - neighbor[1])**2)
                            if dist < min_dist:
                                return False
            return True
        
        # Add initial point to grid
        gy, gx = grid_coords(x, y)
        grid[gy][gx] = (x, y)
        
        # Generate points
        k = 30  # Candidates to try per point
        while active and len(points) < num_points:
            idx = np.random.randint(len(active))
            ax, ay = active[idx]
            
            found = False
            for _ in range(k):
                # Random point in annulus
                angle = np.random.uniform(0, 2 * math.pi)
                r = np.random.uniform(min_dist, 2 * min_dist)
                nx = ax + r * math.cos(angle)
                ny = ay + r * math.sin(angle)
                
                if is_valid(nx, ny):
                    points.append((nx, ny))
                    active.append((nx, ny))
                    gy, gx = grid_coords(nx, ny)
                    if 0 <= gy < grid_h and 0 <= gx < grid_w:
                        grid[gy][gx] = (nx, ny)
                    found = True
                    break
            
            if not found:
                active.pop(idx)
        
        # Render to image
        result = np.zeros((height, width), dtype=np.uint8)
        for px, py in points:
            ix, iy = int(px), int(py)
            if 0 <= ix < width and 0 <= iy < height:
                result[iy, ix] = 255
        
        return result


# ═══════════════════════════════════════════════════════════════
# VALUE NOISE
# ═══════════════════════════════════════════════════════════════

class ValueNoise:
    """
    Value noise - simplest coherent noise.
    
    Random values at integer coordinates, interpolated between.
    Creates smooth, cloud-like patterns.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.perm = self._generate_permutation()
    
    def _generate_permutation(self) -> np.ndarray:
        """Generate permutation table for reproducible randomness"""
        np.random.seed(self.seed)
        perm = np.arange(256, dtype=np.int32)
        np.random.shuffle(perm)
        return np.tile(perm, 2)
    
    def _hash(self, x: int, y: int) -> float:
        """Hash coordinates to pseudo-random value 0-1"""
        return self.perm[(self.perm[x & 255] + y) & 255] / 255.0
    
    def noise_2d(self, x: float, y: float) -> float:
        """
        Get noise value at coordinates.
        
        Bilinearly interpolates between grid values.
        """
        # Integer coordinates
        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Fractional parts
        fx = x - x0
        fy = y - y0
        
        # Smooth interpolation (smoothstep)
        sx = fx * fx * (3 - 2 * fx)
        sy = fy * fy * (3 - 2 * fy)
        
        # Get corner values
        v00 = self._hash(x0, y0)
        v10 = self._hash(x1, y0)
        v01 = self._hash(x0, y1)
        v11 = self._hash(x1, y1)
        
        # Bilinear interpolation
        v0 = v00 + sx * (v10 - v00)
        v1 = v01 + sx * (v11 - v01)
        
        return v0 + sy * (v1 - v0)
    
    def generate(self, width: int, height: int,
                scale: float = 0.1) -> np.ndarray:
        """Generate noise image at given resolution and scale."""
        result = np.zeros((height, width), dtype=np.float64)
        
        for y in range(height):
            for x in range(width):
                result[y, x] = self.noise_2d(x * scale, y * scale)
        
        return result


# ═══════════════════════════════════════════════════════════════
# PERLIN NOISE
# ═══════════════════════════════════════════════════════════════

class PerlinNoise:
    """
    Ken Perlin's gradient noise (improved 2002 version).
    
    Uses gradients at integer points, not values.
    This produces smoother, more natural-looking results
    than value noise.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.perm = self._generate_permutation()
        self.grad_x, self.grad_y = self._generate_gradients()
    
    def _generate_permutation(self) -> np.ndarray:
        """Generate permutation table"""
        np.random.seed(self.seed)
        perm = np.arange(256, dtype=np.int32)
        np.random.shuffle(perm)
        return np.tile(perm, 2)
    
    def _generate_gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate unit gradient vectors"""
        np.random.seed(self.seed + 1)
        angles = np.random.uniform(0, 2 * math.pi, 256)
        return np.cos(angles), np.sin(angles)
    
    def _dot_grid_gradient(self, ix: int, iy: int, x: float, y: float) -> float:
        """Compute dot product of gradient and distance vector"""
        idx = self.perm[(self.perm[ix & 255] + iy) & 255]
        
        dx = x - ix
        dy = y - iy
        
        return dx * self.grad_x[idx] + dy * self.grad_y[idx]
    
    @staticmethod
    def _smootherstep(t: float) -> float:
        """Ken Perlin's improved smoothstep (6t⁵ - 15t⁴ + 10t³)"""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def noise_2d(self, x: float, y: float) -> float:
        """
        Get Perlin noise value at coordinates.
        
        Returns value in range [-1, 1] (approximately).
        """
        # Grid cell coordinates
        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Interpolation weights
        sx = self._smootherstep(x - x0)
        sy = self._smootherstep(y - y0)
        
        # Dot products at corners
        n00 = self._dot_grid_gradient(x0, y0, x, y)
        n10 = self._dot_grid_gradient(x1, y0, x, y)
        n01 = self._dot_grid_gradient(x0, y1, x, y)
        n11 = self._dot_grid_gradient(x1, y1, x, y)
        
        # Bilinear interpolation
        nx0 = n00 + sx * (n10 - n00)
        nx1 = n01 + sx * (n11 - n01)
        
        return nx0 + sy * (nx1 - nx0)
    
    def generate(self, width: int, height: int,
                scale: float = 0.05) -> np.ndarray:
        """Generate Perlin noise image."""
        result = np.zeros((height, width), dtype=np.float64)
        
        for y in range(height):
            for x in range(width):
                result[y, x] = self.noise_2d(x * scale, y * scale)
        
        # Normalize to 0-1
        result = (result + 1) / 2
        return np.clip(result, 0, 1)


# ═══════════════════════════════════════════════════════════════
# SIMPLEX NOISE
# ═══════════════════════════════════════════════════════════════

class SimplexNoise:
    """
    Simplex noise - Ken Perlin's 2001 improvement.
    
    Advantages over classic Perlin:
    - Lower computational complexity (O(n²) vs O(2^n) in n dimensions)
    - No visible directional artifacts
    - Continuous derivatives
    - Easier to implement in higher dimensions
    
    Uses simplex (triangle in 2D, tetrahedron in 3D) instead of hypercube.
    """
    
    # Skewing factors for 2D
    F2 = 0.5 * (math.sqrt(3) - 1)  # Skew input to simplex grid
    G2 = (3 - math.sqrt(3)) / 6    # Unskew simplex to input
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.perm = self._generate_permutation()
        
        # 12 gradient directions
        self.grad3 = np.array([
            [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
            [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
            [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]
        ], dtype=np.float64)
    
    def _generate_permutation(self) -> np.ndarray:
        np.random.seed(self.seed)
        perm = np.arange(256, dtype=np.int32)
        np.random.shuffle(perm)
        return np.tile(perm, 2)
    
    def _gradient_2d(self, hash_val: int, x: float, y: float) -> float:
        """Get gradient contribution"""
        h = hash_val & 7
        u = x if h < 4 else y
        v = y if h < 4 else x
        return (u if (h & 1) == 0 else -u) + (2 * v if (h & 2) == 0 else -2 * v)
    
    def noise_2d(self, x: float, y: float) -> float:
        """
        2D Simplex noise.
        
        Returns value in range [-1, 1] (approximately).
        """
        # Skew input space to simplex cell
        s = (x + y) * self.F2
        i = int(math.floor(x + s))
        j = int(math.floor(y + s))
        
        # Unskew back to input space
        t = (i + j) * self.G2
        X0 = i - t
        Y0 = j - t
        x0 = x - X0  # Distance from cell origin
        y0 = y - Y0
        
        # Determine which simplex we're in
        if x0 > y0:
            i1, j1 = 1, 0  # Lower triangle
        else:
            i1, j1 = 0, 1  # Upper triangle
        
        # Offsets for middle and last corners
        x1 = x0 - i1 + self.G2
        y1 = y0 - j1 + self.G2
        x2 = x0 - 1.0 + 2.0 * self.G2
        y2 = y0 - 1.0 + 2.0 * self.G2
        
        # Hash coordinates
        ii = i & 255
        jj = j & 255
        
        # Calculate contributions from three corners
        n0 = n1 = n2 = 0.0
        
        # Corner 0
        t0 = 0.5 - x0*x0 - y0*y0
        if t0 > 0:
            t0 *= t0
            gi0 = self.perm[ii + self.perm[jj]] % 12
            n0 = t0 * t0 * (self.grad3[gi0, 0] * x0 + self.grad3[gi0, 1] * y0)
        
        # Corner 1
        t1 = 0.5 - x1*x1 - y1*y1
        if t1 > 0:
            t1 *= t1
            gi1 = self.perm[ii + i1 + self.perm[jj + j1]] % 12
            n1 = t1 * t1 * (self.grad3[gi1, 0] * x1 + self.grad3[gi1, 1] * y1)
        
        # Corner 2
        t2 = 0.5 - x2*x2 - y2*y2
        if t2 > 0:
            t2 *= t2
            gi2 = self.perm[ii + 1 + self.perm[jj + 1]] % 12
            n2 = t2 * t2 * (self.grad3[gi2, 0] * x2 + self.grad3[gi2, 1] * y2)
        
        # Scale to [-1, 1]
        return 70.0 * (n0 + n1 + n2)
    
    def generate(self, width: int, height: int,
                scale: float = 0.02) -> np.ndarray:
        """Generate simplex noise image."""
        result = np.zeros((height, width), dtype=np.float64)
        
        for y in range(height):
            for x in range(width):
                result[y, x] = self.noise_2d(x * scale, y * scale)
        
        return (result + 1) / 2


# ═══════════════════════════════════════════════════════════════
# WORLEY NOISE (CELLULAR/VORONOI)
# ═══════════════════════════════════════════════════════════════

class WorleyNoise:
    """
    Worley noise (cellular/Voronoi noise).
    
    Based on distance to nearest feature points.
    Creates cell-like, organic patterns.
    
    Use cases:
    - Stone/cracked textures
    - Cell/organic patterns
    - Soap bubbles
    - Water caustics
    """
    
    def __init__(self, seed: int = 42, num_points_per_cell: int = 1):
        self.seed = seed
        self.num_points = num_points_per_cell
        np.random.seed(seed)
    
    def _hash_point(self, ix: int, iy: int, n: int) -> Tuple[float, float]:
        """Get feature point position in cell"""
        np.random.seed(self.seed + ix * 374761393 + iy * 668265263 + n * 1013904223)
        return np.random.random(), np.random.random()
    
    def noise_2d(self, x: float, y: float, 
                distance_func: str = 'euclidean',
                which: int = 1) -> float:
        """
        Get Worley noise value.
        
        Args:
            x, y: Coordinates
            distance_func: 'euclidean', 'manhattan', 'chebyshev'
            which: Which nearest point (1=nearest, 2=second nearest, etc.)
        
        Returns:
            Distance to the nth nearest feature point
        """
        ix = int(math.floor(x))
        iy = int(math.floor(y))
        
        distances = []
        
        # Check 3x3 neighborhood of cells
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cell_x = ix + dx
                cell_y = iy + dy
                
                # Get feature points in this cell
                for n in range(self.num_points):
                    px, py = self._hash_point(cell_x, cell_y, n)
                    px += cell_x
                    py += cell_y
                    
                    # Calculate distance
                    if distance_func == 'euclidean':
                        d = math.sqrt((x - px)**2 + (y - py)**2)
                    elif distance_func == 'manhattan':
                        d = abs(x - px) + abs(y - py)
                    elif distance_func == 'chebyshev':
                        d = max(abs(x - px), abs(y - py))
                    else:
                        d = math.sqrt((x - px)**2 + (y - py)**2)
                    
                    distances.append(d)
        
        distances.sort()
        return distances[min(which - 1, len(distances) - 1)]
    
    def generate(self, width: int, height: int,
                scale: float = 0.05,
                distance_func: str = 'euclidean',
                which: int = 1,
                combination: str = 'F1') -> np.ndarray:
        """
        Generate Worley noise image.
        
        Combinations:
        - 'F1': Distance to nearest
        - 'F2': Distance to second nearest  
        - 'F2-F1': Difference (creates cell boundaries)
        - 'F1*F2': Product
        """
        result = np.zeros((height, width), dtype=np.float64)
        
        for y in range(height):
            for x in range(width):
                sx, sy = x * scale, y * scale
                
                if combination == 'F1':
                    result[y, x] = self.noise_2d(sx, sy, distance_func, 1)
                elif combination == 'F2':
                    result[y, x] = self.noise_2d(sx, sy, distance_func, 2)
                elif combination == 'F2-F1':
                    f1 = self.noise_2d(sx, sy, distance_func, 1)
                    f2 = self.noise_2d(sx, sy, distance_func, 2)
                    result[y, x] = f2 - f1
                elif combination == 'F1*F2':
                    f1 = self.noise_2d(sx, sy, distance_func, 1)
                    f2 = self.noise_2d(sx, sy, distance_func, 2)
                    result[y, x] = f1 * f2
        
        # Normalize
        result = (result - result.min()) / (result.max() - result.min() + 0.0001)
        return result


# ═══════════════════════════════════════════════════════════════
# FRACTAL BROWNIAN MOTION (FBM)
# ═══════════════════════════════════════════════════════════════

class FractalBrownianMotion:
    """
    Fractal Brownian Motion - layered noise.
    
    Combines multiple octaves of noise at different frequencies
    and amplitudes. Creates complex, natural-looking patterns.
    
    Parameters:
    - Octaves: Number of noise layers
    - Lacunarity: Frequency multiplier between octaves (usually 2)
    - Persistence: Amplitude multiplier between octaves (usually 0.5)
    """
    
    def __init__(self, 
                 noise_generator: Callable,
                 octaves: int = 6,
                 lacunarity: float = 2.0,
                 persistence: float = 0.5,
                 seed: int = 42):
        self.noise_func = noise_generator
        self.octaves = octaves
        self.lacunarity = lacunarity
        self.persistence = persistence
        self.seed = seed
    
    def sample(self, x: float, y: float) -> float:
        """Sample FBM at coordinates."""
        value = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        
        for _ in range(self.octaves):
            value += amplitude * self.noise_func(x * frequency, y * frequency)
            max_value += amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity
        
        return value / max_value
    
    def generate(self, width: int, height: int,
                scale: float = 0.01) -> np.ndarray:
        """Generate FBM noise image."""
        result = np.zeros((height, width), dtype=np.float64)
        
        for y in range(height):
            for x in range(width):
                result[y, x] = self.sample(x * scale, y * scale)
        
        return (result + 1) / 2


class TurbulenceNoise:
    """
    Turbulence - FBM with absolute value.
    
    Taking absolute value creates sharper ridges and
    more turbulent patterns. Good for:
    - Marble textures
    - Fire
    - Clouds
    """
    
    def __init__(self,
                 noise_generator: Callable,
                 octaves: int = 6,
                 lacunarity: float = 2.0,
                 persistence: float = 0.5):
        self.noise_func = noise_generator
        self.octaves = octaves
        self.lacunarity = lacunarity
        self.persistence = persistence
    
    def sample(self, x: float, y: float) -> float:
        """Sample turbulence at coordinates."""
        value = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        
        for _ in range(self.octaves):
            value += amplitude * abs(self.noise_func(x * frequency, y * frequency))
            max_value += amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity
        
        return value / max_value
    
    def generate(self, width: int, height: int,
                scale: float = 0.01) -> np.ndarray:
        result = np.zeros((height, width), dtype=np.float64)
        
        for y in range(height):
            for x in range(width):
                result[y, x] = self.sample(x * scale, y * scale)
        
        return result


class RidgedNoise:
    """
    Ridged multifractal noise.
    
    Creates sharp ridge-like features. Good for:
    - Mountain ridges
    - Lightning
    - Cracks
    """
    
    def __init__(self,
                 noise_generator: Callable,
                 octaves: int = 6,
                 lacunarity: float = 2.0,
                 gain: float = 0.5,
                 offset: float = 1.0):
        self.noise_func = noise_generator
        self.octaves = octaves
        self.lacunarity = lacunarity
        self.gain = gain
        self.offset = offset
    
    def sample(self, x: float, y: float) -> float:
        """Sample ridged noise at coordinates."""
        value = 0.0
        weight = 1.0
        frequency = 1.0
        
        for _ in range(self.octaves):
            # Get noise value and create ridge
            signal = self.noise_func(x * frequency, y * frequency)
            signal = self.offset - abs(signal)
            signal *= signal  # Square for sharper ridges
            
            # Weight by previous octave's value
            signal *= weight
            
            # Update weight (lower frequencies have more influence)
            weight = np.clip(signal * self.gain, 0, 1)
            
            value += signal
            frequency *= self.lacunarity
        
        return value / self.octaves
    
    def generate(self, width: int, height: int,
                scale: float = 0.01) -> np.ndarray:
        result = np.zeros((height, width), dtype=np.float64)
        
        for y in range(height):
            for x in range(width):
                result[y, x] = self.sample(x * scale, y * scale)
        
        return result


# ═══════════════════════════════════════════════════════════════
# DOMAIN WARPING
# ═══════════════════════════════════════════════════════════════

class DomainWarping:
    """
    Domain warping - distort the input coordinates.
    
    Use one noise function to distort the input to another.
    Creates complex, organic patterns.
    
    Technique by Inigo Quilez.
    """
    
    def __init__(self, base_noise: Callable, warp_noise: Callable = None):
        self.base_noise = base_noise
        self.warp_noise = warp_noise or base_noise
    
    def warp(self, x: float, y: float, 
            warp_strength: float = 0.5,
            iterations: int = 1) -> float:
        """
        Apply domain warping.
        
        More iterations = more complex patterns.
        """
        # First warping
        qx = self.warp_noise(x, y)
        qy = self.warp_noise(x + 5.2, y + 1.3)
        
        if iterations == 1:
            return self.base_noise(
                x + warp_strength * qx,
                y + warp_strength * qy
            )
        
        # Second warping
        rx = self.warp_noise(x + warp_strength * qx + 1.7, y + warp_strength * qy + 9.2)
        ry = self.warp_noise(x + warp_strength * qx + 8.3, y + warp_strength * qy + 2.8)
        
        if iterations == 2:
            return self.base_noise(
                x + warp_strength * rx,
                y + warp_strength * ry
            )
        
        # Third warping
        sx = self.warp_noise(x + warp_strength * rx + 4.1, y + warp_strength * ry + 7.6)
        sy = self.warp_noise(x + warp_strength * rx + 3.9, y + warp_strength * ry + 1.2)
        
        return self.base_noise(
            x + warp_strength * sx,
            y + warp_strength * sy
        )
    
    def generate(self, width: int, height: int,
                scale: float = 0.01,
                warp_strength: float = 4.0,
                iterations: int = 2) -> np.ndarray:
        """Generate domain-warped noise image."""
        result = np.zeros((height, width), dtype=np.float64)
        
        for y in range(height):
            for x in range(width):
                result[y, x] = self.warp(
                    x * scale, y * scale,
                    warp_strength, iterations
                )
        
        return (result + 1) / 2


# ═══════════════════════════════════════════════════════════════
# CURL NOISE
# ═══════════════════════════════════════════════════════════════

class CurlNoise:
    """
    Curl noise - divergence-free noise.
    
    The curl of a potential field is always divergence-free,
    meaning it has no sources or sinks. This creates smooth,
    flowing vector fields perfect for:
    - Fluid simulation
    - Particle systems  
    - Hair/fur movement
    """
    
    def __init__(self, potential_noise: Callable, epsilon: float = 0.0001):
        self.potential = potential_noise
        self.epsilon = epsilon
    
    def curl_2d(self, x: float, y: float) -> Tuple[float, float]:
        """
        Compute 2D curl at coordinates.
        
        In 2D, curl is computed from a scalar potential field:
        curl = (dP/dy, -dP/dx)
        
        This is always perpendicular to the gradient.
        """
        e = self.epsilon
        
        # Partial derivative w.r.t. y
        dp_dy = (self.potential(x, y + e) - self.potential(x, y - e)) / (2 * e)
        
        # Partial derivative w.r.t. x
        dp_dx = (self.potential(x + e, y) - self.potential(x - e, y)) / (2 * e)
        
        return dp_dy, -dp_dx
    
    def generate_flow_field(self, width: int, height: int,
                           scale: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate velocity field from curl noise.
        
        Returns:
            (vx, vy): Velocity component arrays
        """
        vx = np.zeros((height, width), dtype=np.float64)
        vy = np.zeros((height, width), dtype=np.float64)
        
        for y in range(height):
            for x in range(width):
                cx, cy = self.curl_2d(x * scale, y * scale)
                vx[y, x] = cx
                vy[y, x] = cy
        
        return vx, vy
    
    def visualize(self, width: int, height: int,
                 scale: float = 0.01) -> np.ndarray:
        """
        Create visual representation of curl field.
        
        Encodes direction in hue, magnitude in brightness.
        """
        vx, vy = self.generate_flow_field(width, height, scale)
        
        # Calculate magnitude and angle
        magnitude = np.sqrt(vx**2 + vy**2)
        angle = np.arctan2(vy, vx)
        
        # Normalize
        magnitude = magnitude / (magnitude.max() + 0.0001)
        
        # Convert to HSV then RGB
        h = (angle + np.pi) / (2 * np.pi)  # 0-1
        s = np.ones_like(h)
        v = magnitude
        
        # HSV to RGB
        result = np.zeros((height, width, 3), dtype=np.float64)
        
        hi = (h * 6).astype(int) % 6
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        for c, (r_val, g_val, b_val) in enumerate([
            (v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q)
        ]):
            mask = hi == c
            result[mask, 0] = r_val[mask]
            result[mask, 1] = g_val[mask]
            result[mask, 2] = b_val[mask]
        
        return (result * 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# NOISE UTILITIES
# ═══════════════════════════════════════════════════════════════

class NoiseUtilities:
    """Utility functions for working with noise"""
    
    @staticmethod
    def normalize(noise: np.ndarray, 
                 new_min: float = 0.0, 
                 new_max: float = 1.0) -> np.ndarray:
        """Normalize noise to specified range"""
        old_min, old_max = noise.min(), noise.max()
        if old_max - old_min < 0.0001:
            return np.full_like(noise, (new_min + new_max) / 2)
        return (noise - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    
    @staticmethod
    def quantize(noise: np.ndarray, levels: int = 8) -> np.ndarray:
        """Quantize noise to discrete levels"""
        normalized = NoiseUtilities.normalize(noise, 0, 1)
        return np.floor(normalized * levels) / (levels - 1)
    
    @staticmethod
    def threshold(noise: np.ndarray, 
                 threshold: float = 0.5,
                 soft: bool = False,
                 softness: float = 0.1) -> np.ndarray:
        """
        Apply threshold to noise.
        
        soft=True creates smooth transition around threshold.
        """
        if soft:
            return 1 / (1 + np.exp(-(noise - threshold) / softness))
        else:
            return (noise > threshold).astype(np.float64)
    
    @staticmethod
    def remap(noise: np.ndarray, 
             curve: Callable) -> np.ndarray:
        """
        Remap noise values through a curve function.
        
        Examples:
        - curve=lambda x: x**2 (darker)
        - curve=lambda x: np.sqrt(x) (lighter)
        - curve=lambda x: np.sin(x * np.pi) (peaks)
        """
        normalized = NoiseUtilities.normalize(noise, 0, 1)
        return curve(normalized)
    
    @staticmethod
    def combine(noise_list: List[np.ndarray],
               weights: List[float] = None,
               mode: str = 'add') -> np.ndarray:
        """
        Combine multiple noise layers.
        
        Modes:
        - 'add': Weighted sum (additive)
        - 'multiply': Product
        - 'max': Maximum of each
        - 'min': Minimum of each
        - 'average': Mean
        """
        if weights is None:
            weights = [1.0] * len(noise_list)
        
        # Normalize weights
        weights = [w / sum(weights) for w in weights]
        
        if mode == 'add':
            result = sum(n * w for n, w in zip(noise_list, weights))
        elif mode == 'multiply':
            result = noise_list[0] ** weights[0]
            for n, w in zip(noise_list[1:], weights[1:]):
                result *= n ** w
        elif mode == 'max':
            result = np.maximum.reduce(noise_list)
        elif mode == 'min':
            result = np.minimum.reduce(noise_list)
        elif mode == 'average':
            result = np.mean(noise_list, axis=0)
        else:
            result = sum(n * w for n, w in zip(noise_list, weights))
        
        return result
    
    @staticmethod
    def seamless_tile(noise_func: Callable,
                     width: int, 
                     height: int,
                     scale: float = 0.05) -> np.ndarray:
        """
        Generate seamlessly tileable noise.
        
        Maps 2D coordinates to surface of 4D torus,
        ensuring seamless wrapping in both dimensions.
        """
        result = np.zeros((height, width), dtype=np.float64)
        
        for y in range(height):
            for x in range(width):
                # Map to torus
                s = x / width
                t = y / height
                
                # Circle coordinates
                dx = math.cos(s * 2 * math.pi) / (2 * math.pi * scale)
                dy = math.sin(s * 2 * math.pi) / (2 * math.pi * scale)
                dz = math.cos(t * 2 * math.pi) / (2 * math.pi * scale)
                dw = math.sin(t * 2 * math.pi) / (2 * math.pi * scale)
                
                # Sample 4D noise (using 2D noise with combined coords)
                result[y, x] = noise_func(dx + dz, dy + dw)
        
        return (result + 1) / 2
    
    @staticmethod
    def to_image(noise: np.ndarray, 
                colormap: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
        """
        Convert noise to RGB image.
        
        colormap: List of (R, G, B) colors to interpolate between.
        """
        normalized = NoiseUtilities.normalize(noise, 0, 1)
        
        if colormap is None:
            # Grayscale
            gray = (normalized * 255).astype(np.uint8)
            return np.stack([gray, gray, gray], axis=2)
        
        # Apply colormap
        h, w = noise.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        num_colors = len(colormap)
        for y in range(h):
            for x in range(w):
                t = normalized[y, x] * (num_colors - 1)
                idx = int(t)
                frac = t - idx
                
                if idx >= num_colors - 1:
                    result[y, x] = colormap[-1]
                else:
                    c1 = np.array(colormap[idx])
                    c2 = np.array(colormap[idx + 1])
                    result[y, x] = (c1 * (1 - frac) + c2 * frac).astype(np.uint8)
        
        return result
```

### 13.2 Applying Noise for Procedural Textures

```python
"""
PROCEDURAL TEXTURE GENERATION USING NOISE

Creating realistic and artistic textures from noise functions.
"""

class ProceduralTextures:
    """Generate various textures using noise"""
    
    @staticmethod
    def wood(width: int, height: int, 
            ring_count: float = 10,
            turbulence: float = 0.1,
            seed: int = 42) -> np.ndarray:
        """
        Generate wood grain texture.
        
        Uses concentric rings disturbed by noise.
        """
        perlin = PerlinNoise(seed)
        
        result = np.zeros((height, width), dtype=np.float64)
        
        cx, cy = width / 2, height / 2
        
        for y in range(height):
            for x in range(width):
                # Distance from center
                dx = (x - cx) / width
                dy = (y - cy) / height
                dist = math.sqrt(dx*dx + dy*dy)
                
                # Add noise perturbation
                noise = perlin.noise_2d(x * 0.01, y * 0.01)
                
                # Create rings
                ring = (dist * ring_count + turbulence * noise * 10) % 1.0
                
                # Darken near ring edges
                result[y, x] = 0.3 + 0.7 * (0.5 + 0.5 * math.cos(ring * 2 * math.pi))
        
        # Apply wood colors
        colormap = [
            (101, 67, 33),   # Dark brown
            (139, 90, 43),   # Medium brown
            (160, 120, 70),  # Light brown
            (139, 90, 43),   # Medium brown
        ]
        
        return NoiseUtilities.to_image(result, colormap)
    
    @staticmethod
    def marble(width: int, height: int,
              vein_scale: float = 0.01,
              turbulence_scale: float = 0.005,
              seed: int = 42) -> np.ndarray:
        """
        Generate marble texture.
        
        Uses sine function distorted by turbulence.
        """
        perlin = PerlinNoise(seed)
        turb = TurbulenceNoise(perlin.noise_2d, octaves=6)
        
        result = np.zeros((height, width), dtype=np.float64)
        
        for y in range(height):
            for x in range(width):
                # Base sine pattern
                base = x * vein_scale
                
                # Add turbulence
                t = turb.sample(x * turbulence_scale, y * turbulence_scale)
                
                # Create veiny pattern
                marble = math.sin(base + t * 10)
                result[y, x] = (marble + 1) / 2
        
        # Marble colors
        colormap = [
            (40, 40, 45),     # Dark gray (veins)
            (180, 180, 185),  # Light gray
            (220, 220, 225),  # Almost white
            (240, 235, 230),  # Warm white
            (220, 220, 225),  # Light gray
        ]
        
        return NoiseUtilities.to_image(result, colormap)
    
    @staticmethod
    def clouds(width: int, height: int,
              coverage: float = 0.5,
              softness: float = 0.3,
              seed: int = 42) -> np.ndarray:
        """
        Generate cloud texture.
        
        Uses FBM noise with coverage control.
        """
        perlin = PerlinNoise(seed)
        fbm = FractalBrownianMotion(perlin.noise_2d, octaves=6)
        
        noise = fbm.generate(width, height, scale=0.005)
        
        # Apply coverage threshold with soft falloff
        threshold = 1 - coverage
        clouds = NoiseUtilities.threshold(noise, threshold, soft=True, softness=softness)
        
        # White clouds on blue sky
        h, w = noise.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Sky gradient
        for y in range(h):
            sky_blue = (135 - int(y / h * 50), 
                       206 - int(y / h * 30), 
                       235 - int(y / h * 20))
            result[y, :] = sky_blue
        
        # Add clouds
        cloud_alpha = (clouds * 255).astype(np.uint8)
        for c in range(3):
            result[:, :, c] = (
                result[:, :, c] * (1 - clouds) + 
                255 * clouds
            ).astype(np.uint8)
        
        return result
    
    @staticmethod
    def terrain(width: int, height: int,
               sea_level: float = 0.3,
               snow_level: float = 0.8,
               seed: int = 42) -> np.ndarray:
        """
        Generate terrain height map with coloring.
        """
        perlin = PerlinNoise(seed)
        fbm = FractalBrownianMotion(perlin.noise_2d, octaves=8, persistence=0.45)
        
        noise = fbm.generate(width, height, scale=0.003)
        
        # Color based on height
        colormap = [
            (0, 0, 100),      # Deep water
            (30, 80, 180),    # Shallow water
            (238, 214, 175),  # Beach
            (34, 139, 34),    # Grass
            (85, 107, 47),    # Forest
            (139, 137, 112),  # Mountain
            (255, 255, 255),  # Snow
        ]
        
        thresholds = [0.0, 0.25, sea_level, 0.35, 0.5, 0.7, snow_level]
        
        h, w = noise.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        for y in range(h):
            for x in range(w):
                val = noise[y, x]
                
                # Find color band
                for i in range(len(thresholds) - 1):
                    if val < thresholds[i + 1]:
                        t = (val - thresholds[i]) / (thresholds[i + 1] - thresholds[i])
                        c1 = np.array(colormap[i])
                        c2 = np.array(colormap[i + 1])
                        result[y, x] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
                        break
                else:
                    result[y, x] = colormap[-1]
        
        return result
    
    @staticmethod
    def fire(width: int, height: int,
            intensity: float = 1.0,
            speed_seed: int = 0,
            seed: int = 42) -> np.ndarray:
        """
        Generate fire/flame texture.
        
        Uses turbulence with vertical bias and heat coloring.
        """
        perlin = PerlinNoise(seed)
        
        result = np.zeros((height, width), dtype=np.float64)
        
        for y in range(height):
            for x in range(width):
                # Turbulence at different scales
                n1 = perlin.noise_2d(x * 0.02, y * 0.01 - speed_seed)
                n2 = perlin.noise_2d(x * 0.04, y * 0.02 - speed_seed * 1.5)
                n3 = perlin.noise_2d(x * 0.08, y * 0.04 - speed_seed * 2)
                
                noise = (n1 + 0.5 * n2 + 0.25 * n3) / 1.75
                
                # Vertical gradient (fire rises)
                gradient = 1 - (y / height)
                
                # Combine
                fire_val = (noise + 1) / 2 * gradient * intensity
                
                # Shape flame
                center_dist = abs(x - width/2) / (width/2)
                fire_val *= 1 - center_dist * 0.5
                
                result[y, x] = min(1, fire_val)
        
        # Fire colors
        colormap = [
            (0, 0, 0),        # Black (no fire)
            (128, 0, 0),      # Dark red
            (255, 80, 0),     # Orange
            (255, 200, 0),    # Yellow
            (255, 255, 200),  # White hot
        ]
        
        return NoiseUtilities.to_image(result, colormap)
    
    @staticmethod
    def water_caustics(width: int, height: int,
                      cell_scale: float = 0.03,
                      brightness: float = 1.5,
                      seed: int = 42) -> np.ndarray:
        """
        Generate water caustic patterns.
        
        Uses Worley noise to simulate light refraction.
        """
        worley = WorleyNoise(seed, num_points_per_cell=1)
        
        # Get F2-F1 for cell boundaries
        noise = worley.generate(width, height, cell_scale, 
                               combination='F2-F1')
        
        # Apply power for sharper caustics
        caustics = noise ** 2 * brightness
        caustics = np.clip(caustics, 0, 1)
        
        # Blue tint
        h, w = noise.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        result[:, :, 0] = (caustics * 100 + 50).astype(np.uint8)   # R
        result[:, :, 1] = (caustics * 150 + 100).astype(np.uint8)  # G
        result[:, :, 2] = (caustics * 200 + 150).astype(np.uint8)  # B
        
        return result
    
    @staticmethod
    def stone_wall(width: int, height: int,
                  cell_scale: float = 0.02,
                  grout_width: float = 0.1,
                  seed: int = 42) -> np.ndarray:
        """
        Generate stone/brick wall texture.
        
        Uses Worley noise for irregular stones with surface detail.
        """
        worley = WorleyNoise(seed, num_points_per_cell=1)
        perlin = PerlinNoise(seed + 1)
        
        # Cell boundaries (grout)
        f2_f1 = worley.generate(width, height, cell_scale, combination='F2-F1')
        
        # Grout mask
        grout = (f2_f1 < grout_width).astype(np.float64)
        
        # Stone surface variation
        surface = np.zeros((height, width), dtype=np.float64)
        for y in range(height):
            for x in range(width):
                # Multiple noise scales for surface detail
                n1 = perlin.noise_2d(x * 0.02, y * 0.02)
                n2 = perlin.noise_2d(x * 0.1, y * 0.1)
                surface[y, x] = 0.7 + 0.2 * n1 + 0.1 * n2
        
        # Combine stone and grout
        stone_mask = 1 - grout
        
        result = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Stone color
        stone_r = (surface * 160 * stone_mask).astype(np.uint8)
        stone_g = (surface * 150 * stone_mask).astype(np.uint8)
        stone_b = (surface * 130 * stone_mask).astype(np.uint8)
        
        # Grout color
        grout_color = 80
        
        result[:, :, 0] = stone_r + (grout * grout_color).astype(np.uint8)
        result[:, :, 1] = stone_g + (grout * grout_color).astype(np.uint8)
        result[:, :, 2] = stone_b + (grout * grout_color).astype(np.uint8)
        
        return result
    
    @staticmethod
    def organic_cells(width: int, height: int,
                     cell_scale: float = 0.025,
                     membrane_thickness: float = 0.08,
                     seed: int = 42) -> np.ndarray:
        """
        Generate organic cell/membrane texture.
        
        Good for biological, alien, or horror themes.
        """
        worley = WorleyNoise(seed)
        perlin = PerlinNoise(seed + 1)
        
        # Cell structure
        f1 = worley.generate(width, height, cell_scale, combination='F1')
        f2 = worley.generate(width, height, cell_scale, combination='F2')
        
        # Membrane (between cells)
        membrane = np.clip((f2 - f1 - membrane_thickness) * 5, 0, 1)
        
        # Cell interior variation
        interior = np.zeros((height, width), dtype=np.float64)
        for y in range(height):
            for x in range(width):
                interior[y, x] = 0.5 + 0.3 * perlin.noise_2d(x * 0.05, y * 0.05)
        
        # Combine
        combined = interior * membrane
        
        # Organic colors
        colormap = [
            (40, 20, 30),     # Dark interior
            (120, 60, 80),    # Cell interior
            (180, 100, 120),  # Cell highlight
            (200, 150, 160),  # Membrane edge
        ]
        
        return NoiseUtilities.to_image(combined, colormap)
```

---

## Chapter 14: Fractals and Recursive Geometry

### 14.1 Understanding Fractals

Fractals are infinitely complex patterns that are self-similar across different scales. They're nature's building blocks - found in coastlines, mountains, trees, lightning, and blood vessels.

```python
"""
FRACTALS AND RECURSIVE GEOMETRY - COMPLETE IMPLEMENTATION

From classic mathematical fractals to practical applications
in procedural art generation.
"""

import numpy as np
import math
from typing import Tuple, List, Optional, Callable
from PIL import Image, ImageDraw

# ═══════════════════════════════════════════════════════════════
# ESCAPE-TIME FRACTALS
# ═══════════════════════════════════════════════════════════════

class MandelbrotSet:
    """
    The Mandelbrot set - the most famous fractal.
    
    Defined as the set of complex numbers c for which the iteration
    z = z² + c does not escape to infinity.
    
    The boundary of this set has infinite complexity.
    """
    
    def __init__(self, 
                 max_iterations: int = 256,
                 escape_radius: float = 2.0):
        self.max_iter = max_iterations
        self.escape_radius = escape_radius
        self.escape_radius_sq = escape_radius ** 2
    
    def iterate(self, c: complex) -> Tuple[int, complex]:
        """
        Iterate z = z² + c and return escape iteration.
        
        Returns:
            (iterations, final_z): Iteration count and final z value
        """
        z = 0j
        for i in range(self.max_iter):
            if z.real * z.real + z.imag * z.imag > self.escape_radius_sq:
                return i, z
            z = z * z + c
        return self.max_iter, z
    
    def smooth_iteration(self, c: complex) -> float:
        """
        Get smooth iteration count for anti-aliased coloring.
        
        Uses the continuous potential method:
        smooth_i = i + 1 - log(log(|z|)) / log(2)
        """
        z = 0j
        for i in range(self.max_iter):
            if z.real * z.real + z.imag * z.imag > self.escape_radius_sq:
                # Smooth iteration count
                log_zn = math.log(z.real * z.real + z.imag * z.imag) / 2
                nu = math.log(log_zn / math.log(2)) / math.log(2)
                return i + 1 - nu
            z = z * z + c
        return self.max_iter
    
    def generate(self, 
                width: int, 
                height: int,
                center: Tuple[float, float] = (-0.5, 0.0),
                zoom: float = 1.0,
                smooth: bool = True) -> np.ndarray:
        """
        Generate Mandelbrot set image.
        
        Args:
            width, height: Output dimensions
            center: Center of view in complex plane
            zoom: Zoom level (higher = closer)
            smooth: Use smooth coloring
        """
        result = np.zeros((height, width), dtype=np.float64)
        
        # Calculate bounds
        aspect = width / height
        y_range = 2.0 / zoom
        x_range = y_range * aspect
        
        x_min = center[0] - x_range / 2
        y_min = center[1] - y_range / 2
        
        dx = x_range / width
        dy = y_range / height
        
        for py in range(height):
            y = y_min + py * dy
            for px in range(width):
                x = x_min + px * dx
                c = complex(x, y)
                
                if smooth:
                    result[py, px] = self.smooth_iteration(c)
                else:
                    result[py, px] = self.iterate(c)[0]
        
        return result
    
    def generate_orbit_trap(self,
                           width: int,
                           height: int,
                           trap_point: complex = 0j,
                           trap_radius: float = 0.5,
                           center: Tuple[float, float] = (-0.5, 0.0),
                           zoom: float = 1.0) -> np.ndarray:
        """
        Generate using orbit trap coloring.
        
        Colors based on minimum distance to a "trap" shape
        during iteration. Creates interesting patterns.
        """
        result = np.zeros((height, width), dtype=np.float64)
        
        aspect = width / height
        y_range = 2.0 / zoom
        x_range = y_range * aspect
        
        x_min = center[0] - x_range / 2
        y_min = center[1] - y_range / 2
        
        for py in range(height):
            y = y_min + py * (y_range / height)
            for px in range(width):
                x = x_min + px * (x_range / width)
                c = complex(x, y)
                
                z = 0j
                min_dist = float('inf')
                
                for _ in range(self.max_iter):
                    if abs(z) > self.escape_radius:
                        break
                    
                    # Distance to trap point
                    dist = abs(z - trap_point)
                    min_dist = min(min_dist, dist)
                    
                    z = z * z + c
                
                result[py, px] = min(min_dist / trap_radius, 1.0)
        
        return result


class JuliaSet:
    """
    Julia sets - each point in the Mandelbrot set corresponds
    to a unique Julia set.
    
    Uses the same iteration z = z² + c, but c is fixed and
    we vary the starting z value.
    """
    
    def __init__(self,
                 c: complex,
                 max_iterations: int = 256,
                 escape_radius: float = 2.0):
        self.c = c
        self.max_iter = max_iterations
        self.escape_radius = escape_radius
        self.escape_radius_sq = escape_radius ** 2
    
    def smooth_iteration(self, z: complex) -> float:
        """Get smooth iteration count."""
        for i in range(self.max_iter):
            if z.real * z.real + z.imag * z.imag > self.escape_radius_sq:
                log_zn = math.log(z.real * z.real + z.imag * z.imag) / 2
                nu = math.log(log_zn / math.log(2)) / math.log(2)
                return i + 1 - nu
            z = z * z + self.c
        return self.max_iter
    
    def generate(self,
                width: int,
                height: int,
                center: Tuple[float, float] = (0.0, 0.0),
                zoom: float = 1.0) -> np.ndarray:
        """Generate Julia set image."""
        result = np.zeros((height, width), dtype=np.float64)
        
        aspect = width / height
        y_range = 3.0 / zoom
        x_range = y_range * aspect
        
        x_min = center[0] - x_range / 2
        y_min = center[1] - y_range / 2
        
        dx = x_range / width
        dy = y_range / height
        
        for py in range(height):
            y = y_min + py * dy
            for px in range(width):
                x = x_min + px * dx
                z = complex(x, y)
                result[py, px] = self.smooth_iteration(z)
        
        return result
    
    @staticmethod
    def interesting_c_values() -> dict:
        """Return dictionary of visually interesting c values."""
        return {
            'dendritic': complex(-0.4, 0.6),
            'rabbit': complex(-0.123, 0.745),
            'dragon': complex(-0.8, 0.156),
            'spiral': complex(0.285, 0.01),
            'sea_horse': complex(-0.75, 0.11),
            'siegel_disk': complex(-0.391, -0.587),
            'douady_rabbit': complex(-0.123, 0.745),
            'san_marco': complex(-0.75, 0.0),
            'lightning': complex(-0.7269, 0.1889),
            'galaxy': complex(0.285, 0.013),
        }


class BurningShip:
    """
    Burning Ship fractal - variation where absolute values
    are taken before squaring.
    
    z = (|Re(z)| + i|Im(z)|)² + c
    
    Creates asymmetric, ship-like shapes.
    """
    
    def __init__(self, max_iterations: int = 256):
        self.max_iter = max_iterations
    
    def smooth_iteration(self, c: complex) -> float:
        z = 0j
        for i in range(self.max_iter):
            if abs(z) > 2:
                log_zn = math.log(abs(z))
                nu = math.log(log_zn / math.log(2)) / math.log(2)
                return i + 1 - nu
            # Take absolute values before squaring
            z = complex(abs(z.real), abs(z.imag))
            z = z * z + c
        return self.max_iter
    
    def generate(self,
                width: int,
                height: int,
                center: Tuple[float, float] = (-0.5, -0.5),
                zoom: float = 1.0) -> np.ndarray:
        result = np.zeros((height, width), dtype=np.float64)
        
        aspect = width / height
        y_range = 2.5 / zoom
        x_range = y_range * aspect
        
        x_min = center[0] - x_range / 2
        y_min = center[1] - y_range / 2
        
        for py in range(height):
            y = y_min + py * (y_range / height)
            for px in range(width):
                x = x_min + px * (x_range / width)
                c = complex(x, y)
                result[py, px] = self.smooth_iteration(c)
        
        return result


class Multibrot:
    """
    Multibrot sets - generalization of Mandelbrot with
    different exponents.
    
    z = z^d + c
    
    d=2 is standard Mandelbrot
    d=3 creates 3-fold symmetry
    d>2 creates (d-1)-fold symmetry
    """
    
    def __init__(self, 
                 power: float = 2,
                 max_iterations: int = 256):
        self.power = power
        self.max_iter = max_iterations
    
    def smooth_iteration(self, c: complex) -> float:
        z = 0j
        for i in range(self.max_iter):
            r = abs(z)
            if r > 2:
                log_zn = math.log(r)
                nu = math.log(log_zn / math.log(self.power)) / math.log(self.power)
                return i + 1 - nu
            # z^power - use polar form for non-integer powers
            if r > 0:
                theta = math.atan2(z.imag, z.real)
                new_r = r ** self.power
                new_theta = theta * self.power
                z = complex(new_r * math.cos(new_theta), 
                           new_r * math.sin(new_theta))
            z = z + c
        return self.max_iter
    
    def generate(self,
                width: int,
                height: int,
                center: Tuple[float, float] = (0.0, 0.0),
                zoom: float = 1.0) -> np.ndarray:
        result = np.zeros((height, width), dtype=np.float64)
        
        aspect = width / height
        y_range = 2.5 / zoom
        x_range = y_range * aspect
        
        x_min = center[0] - x_range / 2
        y_min = center[1] - y_range / 2
        
        for py in range(height):
            y = y_min + py * (y_range / height)
            for px in range(width):
                x = x_min + px * (x_range / width)
                c = complex(x, y)
                result[py, px] = self.smooth_iteration(c)
        
        return result


class NewtonFractal:
    """
    Newton fractals - visualization of Newton's method
    for finding polynomial roots.
    
    Colors based on which root the iteration converges to
    and how many iterations it takes.
    """
    
    def __init__(self,
                 polynomial: List[complex],
                 max_iterations: int = 50,
                 tolerance: float = 1e-6):
        """
        Args:
            polynomial: Coefficients [a_n, a_{n-1}, ..., a_1, a_0]
                       for a_n*z^n + a_{n-1}*z^{n-1} + ... + a_0
        """
        self.coeffs = polynomial
        self.degree = len(polynomial) - 1
        self.max_iter = max_iterations
        self.tolerance = tolerance
        
        # Derivative coefficients
        self.deriv = [
            c * (self.degree - i) 
            for i, c in enumerate(polynomial[:-1])
        ]
        
        # Find roots numerically
        self.roots = self._find_roots()
    
    def _eval_poly(self, z: complex, coeffs: List[complex]) -> complex:
        """Evaluate polynomial at z using Horner's method."""
        result = coeffs[0]
        for c in coeffs[1:]:
            result = result * z + c
        return result
    
    def _find_roots(self) -> List[complex]:
        """Find polynomial roots numerically."""
        # Use numpy for root finding
        roots = np.roots(self.coeffs)
        return [complex(r) for r in roots]
    
    def iterate(self, z: complex) -> Tuple[int, int]:
        """
        Apply Newton's method and return (iterations, root_index).
        """
        for i in range(self.max_iter):
            # Check convergence to each root
            for ri, root in enumerate(self.roots):
                if abs(z - root) < self.tolerance:
                    return i, ri
            
            # Newton iteration: z = z - f(z)/f'(z)
            fz = self._eval_poly(z, self.coeffs)
            fpz = self._eval_poly(z, self.deriv)
            
            if abs(fpz) < 1e-12:
                return self.max_iter, -1
            
            z = z - fz / fpz
        
        # Find nearest root
        min_dist = float('inf')
        nearest = 0
        for ri, root in enumerate(self.roots):
            dist = abs(z - root)
            if dist < min_dist:
                min_dist = dist
                nearest = ri
        
        return self.max_iter, nearest
    
    def generate(self,
                width: int,
                height: int,
                center: Tuple[float, float] = (0.0, 0.0),
                zoom: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Newton fractal.
        
        Returns:
            (root_image, iteration_image): Which root and how many iterations
        """
        roots_result = np.zeros((height, width), dtype=np.int32)
        iters_result = np.zeros((height, width), dtype=np.float64)
        
        aspect = width / height
        y_range = 4.0 / zoom
        x_range = y_range * aspect
        
        x_min = center[0] - x_range / 2
        y_min = center[1] - y_range / 2
        
        for py in range(height):
            y = y_min + py * (y_range / height)
            for px in range(width):
                x = x_min + px * (x_range / width)
                z = complex(x, y)
                iters, root = self.iterate(z)
                roots_result[py, px] = root
                iters_result[py, px] = iters
        
        return roots_result, iters_result
    
    def generate_colored(self,
                        width: int,
                        height: int,
                        center: Tuple[float, float] = (0.0, 0.0),
                        zoom: float = 1.0,
                        colors: List[Tuple[int, int, int]] = None) -> np.ndarray:
        """Generate colored Newton fractal image."""
        roots, iters = self.generate(width, height, center, zoom)
        
        if colors is None:
            # Default colors for roots
            colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
            ]
        
        # Normalize iterations for shading
        max_iter = self.max_iter
        iters_norm = iters / max_iter
        
        result = np.zeros((height, width, 3), dtype=np.uint8)
        
        for py in range(height):
            for px in range(width):
                root_idx = roots[py, px]
                if root_idx >= 0 and root_idx < len(colors):
                    color = np.array(colors[root_idx])
                    # Shade by iteration count
                    shade = 1 - iters_norm[py, px] * 0.7
                    result[py, px] = (color * shade).astype(np.uint8)
        
        return result


# ═══════════════════════════════════════════════════════════════
# ITERATED FUNCTION SYSTEMS (IFS)
# ═══════════════════════════════════════════════════════════════

class IFS:
    """
    Iterated Function System - generate fractals by repeatedly
    applying affine transformations.
    
    Famous examples: Sierpinski triangle, Barnsley fern, Koch curve.
    """
    
    def __init__(self, transforms: List[dict]):
        """
        Args:
            transforms: List of transformation dictionaries, each containing:
                - 'a', 'b', 'c', 'd': 2x2 matrix coefficients
                - 'e', 'f': Translation
                - 'prob': Probability of choosing this transform
        """
        self.transforms = transforms
        
        # Normalize probabilities
        total_prob = sum(t.get('prob', 1.0) for t in transforms)
        self.probs = [t.get('prob', 1.0) / total_prob for t in transforms]
        self.cumulative_probs = np.cumsum(self.probs)
    
    def apply_transform(self, x: float, y: float, t: dict) -> Tuple[float, float]:
        """Apply affine transformation."""
        new_x = t['a'] * x + t['b'] * y + t['e']
        new_y = t['c'] * x + t['d'] * y + t['f']
        return new_x, new_y
    
    def generate_points(self, num_points: int = 100000) -> List[Tuple[float, float]]:
        """Generate IFS fractal points using chaos game."""
        points = []
        x, y = 0.0, 0.0
        
        # Skip first iterations
        skip = 20
        
        for i in range(num_points + skip):
            # Choose transform based on probability
            r = np.random.random()
            for j, cp in enumerate(self.cumulative_probs):
                if r < cp:
                    x, y = self.apply_transform(x, y, self.transforms[j])
                    break
            
            if i >= skip:
                points.append((x, y))
        
        return points
    
    def render(self,
              width: int,
              height: int,
              num_points: int = 500000,
              color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Render IFS fractal to image."""
        points = self.generate_points(num_points)
        
        # Find bounds
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Add margin
        margin = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * margin
        x_max += x_range * margin
        y_min -= y_range * margin
        y_max += y_range * margin
        
        # Create accumulation buffer
        buffer = np.zeros((height, width), dtype=np.float64)
        
        for x, y in points:
            px = int((x - x_min) / (x_max - x_min) * (width - 1))
            py = int((y_max - y) / (y_max - y_min) * (height - 1))  # Flip y
            
            if 0 <= px < width and 0 <= py < height:
                buffer[py, px] += 1
        
        # Log scale for better visibility
        buffer = np.log(1 + buffer)
        buffer = buffer / buffer.max()
        
        # Apply color
        result = np.zeros((height, width, 3), dtype=np.uint8)
        for c in range(3):
            result[:, :, c] = (buffer * color[c]).astype(np.uint8)
        
        return result
    
    @staticmethod
    def sierpinski_triangle() -> 'IFS':
        """Create Sierpinski triangle IFS."""
        return IFS([
            {'a': 0.5, 'b': 0, 'c': 0, 'd': 0.5, 'e': 0, 'f': 0, 'prob': 1},
            {'a': 0.5, 'b': 0, 'c': 0, 'd': 0.5, 'e': 0.5, 'f': 0, 'prob': 1},
            {'a': 0.5, 'b': 0, 'c': 0, 'd': 0.5, 'e': 0.25, 'f': 0.5, 'prob': 1},
        ])
    
    @staticmethod
    def barnsley_fern() -> 'IFS':
        """Create Barnsley fern IFS."""
        return IFS([
            # Stem
            {'a': 0, 'b': 0, 'c': 0, 'd': 0.16, 'e': 0, 'f': 0, 'prob': 0.01},
            # Successively smaller leaflets
            {'a': 0.85, 'b': 0.04, 'c': -0.04, 'd': 0.85, 'e': 0, 'f': 1.6, 'prob': 0.85},
            # Largest left leaflet
            {'a': 0.2, 'b': -0.26, 'c': 0.23, 'd': 0.22, 'e': 0, 'f': 1.6, 'prob': 0.07},
            # Largest right leaflet
            {'a': -0.15, 'b': 0.28, 'c': 0.26, 'd': 0.24, 'e': 0, 'f': 0.44, 'prob': 0.07},
        ])
    
    @staticmethod
    def dragon_curve() -> 'IFS':
        """Create dragon curve IFS."""
        s = 1 / math.sqrt(2)
        return IFS([
            {'a': s, 'b': -s, 'c': s, 'd': s, 'e': 0, 'f': 0, 'prob': 0.5},
            {'a': -s, 'b': -s, 'c': s, 'd': -s, 'e': 1, 'f': 0, 'prob': 0.5},
        ])
    
    @staticmethod
    def maple_leaf() -> 'IFS':
        """Create maple leaf IFS."""
        return IFS([
            {'a': 0.14, 'b': 0.01, 'c': 0, 'd': 0.51, 'e': -0.08, 'f': -1.31, 'prob': 0.1},
            {'a': 0.43, 'b': 0.52, 'c': -0.45, 'd': 0.5, 'e': 1.49, 'f': -0.75, 'prob': 0.35},
            {'a': 0.45, 'b': -0.49, 'c': 0.47, 'd': 0.47, 'e': -1.62, 'f': -0.74, 'prob': 0.35},
            {'a': 0.49, 'b': 0, 'c': 0, 'd': 0.51, 'e': 0.02, 'f': 1.62, 'prob': 0.2},
        ])


# ═══════════════════════════════════════════════════════════════
# L-SYSTEMS (LINDENMAYER SYSTEMS)
# ═══════════════════════════════════════════════════════════════

class LSystem:
    """
    L-Systems for generating fractal curves and plant-like structures.
    
    Uses string rewriting rules and turtle graphics interpretation.
    """
    
    def __init__(self,
                 axiom: str,
                 rules: dict,
                 angle: float = 90,
                 initial_angle: float = 90):
        """
        Args:
            axiom: Starting string
            rules: Dictionary of replacement rules (char -> string)
            angle: Turn angle in degrees
            initial_angle: Starting angle (90 = up)
        """
        self.axiom = axiom
        self.rules = rules
        self.angle = math.radians(angle)
        self.initial_angle = math.radians(initial_angle)
    
    def generate_string(self, iterations: int) -> str:
        """Generate L-system string after n iterations."""
        result = self.axiom
        for _ in range(iterations):
            new_result = ""
            for char in result:
                new_result += self.rules.get(char, char)
            result = new_result
        return result
    
    def interpret_turtle(self, 
                        lstring: str, 
                        step_length: float = 10) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Interpret L-system string as turtle graphics commands.
        
        Commands:
            F: Move forward and draw
            f: Move forward without drawing
            +: Turn right by angle
            -: Turn left by angle
            [: Push position/angle to stack
            ]: Pop position/angle from stack
        
        Returns:
            List of line segments [(start, end), ...]
        """
        lines = []
        stack = []
        
        x, y = 0.0, 0.0
        angle = self.initial_angle
        
        for char in lstring:
            if char == 'F':
                # Move forward and draw
                new_x = x + step_length * math.cos(angle)
                new_y = y + step_length * math.sin(angle)
                lines.append(((x, y), (new_x, new_y)))
                x, y = new_x, new_y
            elif char == 'f':
                # Move forward without drawing
                x += step_length * math.cos(angle)
                y += step_length * math.sin(angle)
            elif char == '+':
                angle -= self.angle  # Turn right
            elif char == '-':
                angle += self.angle  # Turn left
            elif char == '[':
                stack.append((x, y, angle))
            elif char == ']':
                x, y, angle = stack.pop()
        
        return lines
    
    def render(self,
              width: int,
              height: int,
              iterations: int,
              line_color: Tuple[int, int, int] = (255, 255, 255),
              bg_color: Tuple[int, int, int] = (0, 0, 0),
              line_width: int = 1) -> np.ndarray:
        """Render L-system to image."""
        # Generate string
        lstring = self.generate_string(iterations)
        
        # Get lines
        lines = self.interpret_turtle(lstring)
        
        if not lines:
            return np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        # Find bounds
        all_points = [p for line in lines for p in line]
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Add margin
        margin = 0.1
        x_range = x_max - x_min or 1
        y_range = y_max - y_min or 1
        
        # Calculate scale to fit
        scale = min(
            width * (1 - 2*margin) / x_range,
            height * (1 - 2*margin) / y_range
        )
        
        # Create image
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw lines
        offset_x = width / 2 - (x_min + x_max) / 2 * scale
        offset_y = height / 2 - (y_min + y_max) / 2 * scale
        
        for (x1, y1), (x2, y2) in lines:
            px1 = int(x1 * scale + offset_x)
            py1 = int(height - (y1 * scale + offset_y))  # Flip y
            px2 = int(x2 * scale + offset_x)
            py2 = int(height - (y2 * scale + offset_y))
            
            draw.line([(px1, py1), (px2, py2)], fill=line_color, width=line_width)
        
        return np.array(img)
    
    @staticmethod
    def koch_curve() -> 'LSystem':
        """Koch snowflake curve."""
        return LSystem(
            axiom='F',
            rules={'F': 'F+F-F-F+F'},
            angle=90
        )
    
    @staticmethod
    def koch_snowflake() -> 'LSystem':
        """Complete Koch snowflake."""
        return LSystem(
            axiom='F--F--F',
            rules={'F': 'F+F--F+F'},
            angle=60
        )
    
    @staticmethod
    def sierpinski_triangle() -> 'LSystem':
        """Sierpinski triangle via L-system."""
        return LSystem(
            axiom='F-G-G',
            rules={'F': 'F-G+F+G-F', 'G': 'GG'},
            angle=120
        )
    
    @staticmethod
    def dragon_curve() -> 'LSystem':
        """Dragon curve."""
        return LSystem(
            axiom='FX',
            rules={'X': 'X+YF+', 'Y': '-FX-Y'},
            angle=90,
            initial_angle=0
        )
    
    @staticmethod
    def hilbert_curve() -> 'LSystem':
        """Hilbert space-filling curve."""
        return LSystem(
            axiom='A',
            rules={
                'A': '-BF+AFA+FB-',
                'B': '+AF-BFB-FA+'
            },
            angle=90
        )
    
    @staticmethod
    def plant_1() -> 'LSystem':
        """Simple plant structure."""
        return LSystem(
            axiom='X',
            rules={
                'X': 'F+[[X]-X]-F[-FX]+X',
                'F': 'FF'
            },
            angle=25,
            initial_angle=90
        )
    
    @staticmethod
    def plant_2() -> 'LSystem':
        """Bushy plant structure."""
        return LSystem(
            axiom='F',
            rules={'F': 'FF+[+F-F-F]-[-F+F+F]'},
            angle=22.5,
            initial_angle=90
        )
    
    @staticmethod
    def plant_3() -> 'LSystem':
        """Stochastic-looking plant."""
        return LSystem(
            axiom='X',
            rules={
                'X': 'F[+X][-X]FX',
                'F': 'FF'
            },
            angle=25.7,
            initial_angle=90
        )
    
    @staticmethod
    def binary_tree() -> 'LSystem':
        """Binary fractal tree."""
        return LSystem(
            axiom='0',
            rules={
                '1': '11',
                '0': '1[+0]-0'
            },
            angle=45,
            initial_angle=90
        )
    
    @staticmethod
    def penrose_tiling() -> 'LSystem':
        """Penrose P3 tiling (partial)."""
        return LSystem(
            axiom='[7]++[7]++[7]++[7]++[7]',
            rules={
                '6': '81++91----71[-81----61]++',
                '7': '+81--91[---61--71]+',
                '8': '-61++71[+++81++91]-',
                '9': '--81++++61[+91++++71]--71',
                '1': ''  # Don't draw, just turn
            },
            angle=36
        )


# ═══════════════════════════════════════════════════════════════
# FRACTAL DIMENSION AND ANALYSIS
# ═══════════════════════════════════════════════════════════════

class FractalAnalysis:
    """Tools for analyzing fractal properties"""
    
    @staticmethod
    def box_counting_dimension(image: np.ndarray,
                              min_box_size: int = 2,
                              max_box_size: int = None) -> float:
        """
        Estimate fractal dimension using box counting.
        
        Count how many boxes of size s are needed to cover the set,
        then D = -log(N)/log(s)
        """
        if len(image.shape) == 3:
            # Convert to binary
            gray = np.mean(image, axis=2)
            binary = gray > 128
        else:
            binary = image > 0
        
        h, w = binary.shape
        if max_box_size is None:
            max_box_size = min(h, w) // 2
        
        sizes = []
        counts = []
        
        box_size = min_box_size
        while box_size <= max_box_size:
            count = 0
            for y in range(0, h, box_size):
                for x in range(0, w, box_size):
                    # Check if box contains any points
                    box = binary[y:min(y+box_size, h), x:min(x+box_size, w)]
                    if np.any(box):
                        count += 1
            
            sizes.append(box_size)
            counts.append(count)
            box_size *= 2
        
        if len(sizes) < 2:
            return 0.0
        
        # Linear regression on log-log plot
        log_sizes = np.log(sizes)
        log_counts = np.log(counts)
        
        # D = -slope
        slope, _ = np.polyfit(log_sizes, log_counts, 1)
        
        return -slope
    
    @staticmethod
    def lacunarity(image: np.ndarray, box_sizes: List[int] = None) -> dict:
        """
        Calculate lacunarity - measure of "gappiness" or texture.
        
        High lacunarity = clumpy, low lacunarity = more uniform.
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
            binary = gray > 128
        else:
            binary = image > 0
        
        h, w = binary.shape
        
        if box_sizes is None:
            box_sizes = [2, 4, 8, 16, 32]
        
        results = {}
        
        for size in box_sizes:
            if size > min(h, w):
                continue
            
            box_masses = []
            
            for y in range(0, h - size + 1, size):
                for x in range(0, w - size + 1, size):
                    box = binary[y:y+size, x:x+size]
                    mass = np.sum(box)
                    box_masses.append(mass)
            
            if len(box_masses) > 0:
                mean_mass = np.mean(box_masses)
                var_mass = np.var(box_masses)
                
                if mean_mass > 0:
                    lacunarity = (var_mass / mean_mass**2) + 1
                else:
                    lacunarity = 0
                
                results[size] = lacunarity
        
        return results


# ═══════════════════════════════════════════════════════════════
# FRACTAL COLORING ALGORITHMS
# ═══════════════════════════════════════════════════════════════

class FractalColoring:
    """Various coloring algorithms for escape-time fractals"""
    
    @staticmethod
    def linear_gradient(iteration_data: np.ndarray,
                       colors: List[Tuple[int, int, int]],
                       period: float = 1.0) -> np.ndarray:
        """
        Map iterations to color gradient.
        """
        h, w = iteration_data.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        normalized = (iteration_data % period) / period
        
        num_colors = len(colors)
        
        for y in range(h):
            for x in range(w):
                t = normalized[y, x] * (num_colors - 1)
                idx = int(t)
                frac = t - idx
                
                if idx >= num_colors - 1:
                    result[y, x] = colors[-1]
                else:
                    c1 = np.array(colors[idx])
                    c2 = np.array(colors[idx + 1])
                    result[y, x] = (c1 * (1 - frac) + c2 * frac).astype(np.uint8)
        
        return result
    
    @staticmethod
    def hsv_coloring(iteration_data: np.ndarray,
                    saturation: float = 0.8,
                    value: float = 0.9,
                    hue_offset: float = 0.0,
                    hue_scale: float = 1.0) -> np.ndarray:
        """
        Map iterations to HSV color space.
        """
        h, w = iteration_data.shape
        
        # Normalize and apply scale
        max_iter = iteration_data.max()
        normalized = iteration_data / max_iter * hue_scale + hue_offset
        
        # Create HSV image
        hue = (normalized % 1.0)
        sat = np.full_like(hue, saturation)
        val = np.where(iteration_data >= max_iter, 0, value)
        
        # HSV to RGB
        hsv = np.stack([hue * 360, sat, val], axis=2)
        
        h_channel = hsv[:, :, 0] / 60
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        
        hi = (h_channel.astype(int) % 6)
        f = h_channel - hi
        p = v_channel * (1 - s_channel)
        q = v_channel * (1 - f * s_channel)
        t = v_channel * (1 - (1 - f) * s_channel)
        
        result = np.zeros((h, w, 3), dtype=np.float64)
        
        for i, (r, g, b) in enumerate([
            (v_channel, t, p),
            (q, v_channel, p),
            (p, v_channel, t),
            (p, q, v_channel),
            (t, p, v_channel),
            (v_channel, p, q)
        ]):
            mask = hi == i
            result[mask, 0] = r[mask]
            result[mask, 1] = g[mask]
            result[mask, 2] = b[mask]
        
        return (result * 255).astype(np.uint8)
    
    @staticmethod
    def histogram_coloring(iteration_data: np.ndarray,
                          colors: List[Tuple[int, int, int]]) -> np.ndarray:
        """
        Histogram equalization for more uniform color distribution.
        
        This ensures all colors are equally represented in the final image.
        """
        h, w = iteration_data.shape
        
        # Get unique iteration values and their counts
        flat = iteration_data.flatten()
        unique, counts = np.unique(flat, return_counts=True)
        
        # Build cumulative histogram
        cumsum = np.cumsum(counts)
        total = cumsum[-1]
        
        # Map iterations to equalized values
        mapping = {}
        for i, (val, cs) in enumerate(zip(unique, cumsum)):
            mapping[val] = cs / total
        
        # Apply mapping
        equalized = np.zeros((h, w), dtype=np.float64)
        for y in range(h):
            for x in range(w):
                equalized[y, x] = mapping[iteration_data[y, x]]
        
        # Map to colors
        return FractalColoring.linear_gradient(equalized, colors, period=1.0)
    
    @staticmethod
    def distance_estimation(width: int,
                           height: int,
                           fractal_func: Callable,
                           center: Tuple[float, float] = (-0.5, 0.0),
                           zoom: float = 1.0,
                           max_iter: int = 256) -> np.ndarray:
        """
        Distance estimation method for sharp edges.
        
        Computes approximate distance to fractal boundary.
        """
        result = np.zeros((height, width), dtype=np.float64)
        
        aspect = width / height
        y_range = 2.0 / zoom
        x_range = y_range * aspect
        
        x_min = center[0] - x_range / 2
        y_min = center[1] - y_range / 2
        
        for py in range(height):
            y = y_min + py * (y_range / height)
            for px in range(width):
                x = x_min + px * (x_range / width)
                c = complex(x, y)
                
                # Iterate with derivative
                z = 0j
                dz = 0j  # Derivative
                
                for _ in range(max_iter):
                    if abs(z) > 1e6:
                        break
                    dz = 2 * z * dz + 1
                    z = z * z + c
                
                # Distance estimation
                if abs(z) > 2:
                    dist = 0.5 * math.log(abs(z)) * abs(z) / abs(dz)
                    result[py, px] = dist
                else:
                    result[py, px] = 0
        
        return result
```

---

# PART IV: CORE IMAGE OPERATIONS

---

## Chapter 15: Color Manipulation

### 15.1 Color Space Conversions

```python
"""
COLOR MANIPULATION - COMPREHENSIVE GUIDE

Complete coverage of color space conversions, adjustments,
and artistic color effects.
"""

import numpy as np
import math
from typing import Tuple, Union, Optional

# ═══════════════════════════════════════════════════════════════
# COLOR SPACE CONVERSIONS
# ═══════════════════════════════════════════════════════════════

class ColorSpaceConversion:
    """Complete color space conversion utilities"""
    
    # ─────────────────────────────────────────────────────────
    # RGB ↔ HSV
    # ─────────────────────────────────────────────────────────
    
    @staticmethod
    def rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
        """
        Convert RGB to HSV.
        
        RGB: 0-255 or 0-1
        HSV: H=0-360°, S=0-1, V=0-1
        """
        # Normalize to 0-1 if needed
        if r > 1 or g > 1 or b > 1:
            r, g, b = r/255, g/255, b/255
        
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        diff = max_c - min_c
        
        # Value
        v = max_c
        
        # Saturation
        s = 0 if max_c == 0 else diff / max_c
        
        # Hue
        if diff == 0:
            h = 0
        elif max_c == r:
            h = 60 * (((g - b) / diff) % 6)
        elif max_c == g:
            h = 60 * ((b - r) / diff + 2)
        else:
            h = 60 * ((r - g) / diff + 4)
        
        return h, s, v
    
    @staticmethod
    def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
        """
        Convert HSV to RGB.
        
        HSV: H=0-360°, S=0-1, V=0-1
        RGB: 0-1 (multiply by 255 for 8-bit)
        """
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return r + m, g + m, b + m
    
    # ─────────────────────────────────────────────────────────
    # RGB ↔ HSL
    # ─────────────────────────────────────────────────────────
    
    @staticmethod
    def rgb_to_hsl(r: float, g: float, b: float) -> Tuple[float, float, float]:
        """
        Convert RGB to HSL.
        
        HSL: H=0-360°, S=0-1, L=0-1
        
        HSL is often more intuitive than HSV:
        - L=0 is always black
        - L=1 is always white
        - L=0.5 gives pure colors
        """
        if r > 1 or g > 1 or b > 1:
            r, g, b = r/255, g/255, b/255
        
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        diff = max_c - min_c
        
        # Lightness
        l = (max_c + min_c) / 2
        
        # Saturation
        if diff == 0:
            s = 0
        else:
            s = diff / (1 - abs(2 * l - 1))
        
        # Hue (same as HSV)
        if diff == 0:
            h = 0
        elif max_c == r:
            h = 60 * (((g - b) / diff) % 6)
        elif max_c == g:
            h = 60 * ((b - r) / diff + 2)
        else:
            h = 60 * ((r - g) / diff + 4)
        
        return h, s, l
    
    @staticmethod
    def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
        """Convert HSL to RGB."""
        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l - c / 2
        
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return r + m, g + m, b + m
    
    # ─────────────────────────────────────────────────────────
    # RGB ↔ LAB (CIELAB)
    # ─────────────────────────────────────────────────────────
    
    @staticmethod
    def rgb_to_xyz(r: float, g: float, b: float) -> Tuple[float, float, float]:
        """
        Convert RGB to XYZ (intermediate step for LAB).
        
        Uses sRGB to XYZ transformation matrix.
        """
        if r > 1 or g > 1 or b > 1:
            r, g, b = r/255, g/255, b/255
        
        # Linearize (remove gamma)
        def linearize(c):
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        
        r_lin = linearize(r)
        g_lin = linearize(g)
        b_lin = linearize(b)
        
        # sRGB to XYZ matrix (D65 illuminant)
        x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
        y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
        z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041
        
        return x * 100, y * 100, z * 100
    
    @staticmethod
    def xyz_to_rgb(x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Convert XYZ to RGB."""
        x, y, z = x / 100, y / 100, z / 100
        
        # XYZ to sRGB matrix
        r_lin = x *  3.2404542 + y * -1.5371385 + z * -0.4985314
        g_lin = x * -0.9692660 + y *  1.8760108 + z *  0.0415560
        b_lin = x *  0.0556434 + y * -0.2040259 + z *  1.0572252
        
        # Apply gamma
        def gamma(c):
            return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1/2.4)) - 0.055
        
        r = np.clip(gamma(r_lin), 0, 1)
        g = np.clip(gamma(g_lin), 0, 1)
        b = np.clip(gamma(b_lin), 0, 1)
        
        return r, g, b
    
    @staticmethod
    def xyz_to_lab(x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Convert XYZ to LAB.
        
        LAB is perceptually uniform - equal distances represent
        equal perceived color differences.
        
        L: Lightness (0-100)
        a: Green(-) to Red(+) axis
        b: Blue(-) to Yellow(+) axis
        """
        # D65 reference white
        xn, yn, zn = 95.047, 100.0, 108.883
        
        def f(t):
            delta = 6/29
            return t ** (1/3) if t > delta ** 3 else t / (3 * delta ** 2) + 4/29
        
        l = 116 * f(y / yn) - 16
        a = 500 * (f(x / xn) - f(y / yn))
        b = 200 * (f(y / yn) - f(z / zn))
        
        return l, a, b
    
    @staticmethod
    def lab_to_xyz(l: float, a: float, b: float) -> Tuple[float, float, float]:
        """Convert LAB to XYZ."""
        xn, yn, zn = 95.047, 100.0, 108.883
        
        def f_inv(t):
            delta = 6/29
            return t ** 3 if t > delta else 3 * delta ** 2 * (t - 4/29)
        
        l_adj = (l + 16) / 116
        
        x = xn * f_inv(l_adj + a / 500)
        y = yn * f_inv(l_adj)
        z = zn * f_inv(l_adj - b / 200)
        
        return x, y, z
    
    @staticmethod
    def rgb_to_lab(r: float, g: float, b: float) -> Tuple[float, float, float]:
        """Direct RGB to LAB conversion."""
        x, y, z = ColorSpaceConversion.rgb_to_xyz(r, g, b)
        return ColorSpaceConversion.xyz_to_lab(x, y, z)
    
    @staticmethod
    def lab_to_rgb(l: float, a: float, b: float) -> Tuple[float, float, float]:
        """Direct LAB to RGB conversion."""
        x, y, z = ColorSpaceConversion.lab_to_xyz(l, a, b)
        return ColorSpaceConversion.xyz_to_rgb(x, y, z)
    
    # ─────────────────────────────────────────────────────────
    # RGB ↔ LCH (Cylindrical LAB)
    # ─────────────────────────────────────────────────────────
    
    @staticmethod
    def lab_to_lch(l: float, a: float, b: float) -> Tuple[float, float, float]:
        """
        Convert LAB to LCH (Lightness, Chroma, Hue).
        
        LCH is LAB in cylindrical coordinates:
        - L: Lightness (same as LAB)
        - C: Chroma (saturation) = sqrt(a² + b²)
        - H: Hue angle = atan2(b, a)
        """
        c = math.sqrt(a ** 2 + b ** 2)
        h = math.degrees(math.atan2(b, a))
        if h < 0:
            h += 360
        return l, c, h
    
    @staticmethod
    def lch_to_lab(l: float, c: float, h: float) -> Tuple[float, float, float]:
        """Convert LCH to LAB."""
        h_rad = math.radians(h)
        a = c * math.cos(h_rad)
        b = c * math.sin(h_rad)
        return l, a, b
    
    # ─────────────────────────────────────────────────────────
    # RGB ↔ YUV/YCbCr
    # ─────────────────────────────────────────────────────────
    
    @staticmethod
    def rgb_to_yuv(r: float, g: float, b: float) -> Tuple[float, float, float]:
        """
        Convert RGB to YUV.
        
        Y: Luminance
        U, V: Chrominance (color difference)
        
        Used in video encoding and broadcast.
        """
        if r > 1 or g > 1 or b > 1:
            r, g, b = r/255, g/255, b/255
        
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v = 0.615 * r - 0.51499 * g - 0.10001 * b
        
        return y, u, v
    
    @staticmethod
    def yuv_to_rgb(y: float, u: float, v: float) -> Tuple[float, float, float]:
        """Convert YUV to RGB."""
        r = y + 1.13983 * v
        g = y - 0.39465 * u - 0.58060 * v
        b = y + 2.03211 * u
        
        return (np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1))
    
    # ─────────────────────────────────────────────────────────
    # CMYK
    # ─────────────────────────────────────────────────────────
    
    @staticmethod
    def rgb_to_cmyk(r: float, g: float, b: float) -> Tuple[float, float, float, float]:
        """
        Convert RGB to CMYK.
        
        CMYK: Cyan, Magenta, Yellow, Key (Black)
        Subtractive color model for printing.
        """
        if r > 1 or g > 1 or b > 1:
            r, g, b = r/255, g/255, b/255
        
        k = 1 - max(r, g, b)
        
        if k == 1:
            return 0, 0, 0, 1
        
        c = (1 - r - k) / (1 - k)
        m = (1 - g - k) / (1 - k)
        y = (1 - b - k) / (1 - k)
        
        return c, m, y, k
    
    @staticmethod
    def cmyk_to_rgb(c: float, m: float, y: float, k: float) -> Tuple[float, float, float]:
        """Convert CMYK to RGB."""
        r = (1 - c) * (1 - k)
        g = (1 - m) * (1 - k)
        b = (1 - y) * (1 - k)
        
        return r, g, b


# ═══════════════════════════════════════════════════════════════
# COLOR ADJUSTMENTS
# ═══════════════════════════════════════════════════════════════

class ColorAdjustments:
    """Image color adjustment operations"""
    
    @staticmethod
    def brightness(image: np.ndarray, amount: float) -> np.ndarray:
        """
        Adjust brightness.
        
        amount: -1 to 1 (negative darkens, positive lightens)
        """
        adjusted = image.astype(float) + amount * 255
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    @staticmethod
    def contrast(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust contrast.
        
        factor: 0 = gray, 1 = unchanged, >1 = more contrast
        """
        mean = 128  # Mid-gray
        adjusted = (image.astype(float) - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    @staticmethod
    def saturation(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust color saturation.
        
        factor: 0 = grayscale, 1 = unchanged, >1 = more saturated
        """
        if len(image.shape) != 3:
            return image
        
        # Convert to HSV
        result = np.zeros_like(image, dtype=np.float64)
        
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                r, g, b = image[y, x] / 255.0
                h, s, v = ColorSpaceConversion.rgb_to_hsv(r, g, b)
                
                # Adjust saturation
                s = np.clip(s * factor, 0, 1)
                
                # Convert back
                r, g, b = ColorSpaceConversion.hsv_to_rgb(h, s, v)
                result[y, x] = [r * 255, g * 255, b * 255]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def hue_shift(image: np.ndarray, degrees: float) -> np.ndarray:
        """
        Shift hue by specified degrees.
        
        degrees: -180 to 180 (or any value, wraps around)
        """
        if len(image.shape) != 3:
            return image
        
        result = np.zeros_like(image, dtype=np.float64)
        
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                r, g, b = image[y, x] / 255.0
                h, s, v = ColorSpaceConversion.rgb_to_hsv(r, g, b)
                
                # Shift hue
                h = (h + degrees) % 360
                
                r, g, b = ColorSpaceConversion.hsv_to_rgb(h, s, v)
                result[y, x] = [r * 255, g * 255, b * 255]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def levels(image: np.ndarray,
              input_black: int = 0,
              input_white: int = 255,
              gamma: float = 1.0,
              output_black: int = 0,
              output_white: int = 255) -> np.ndarray:
        """
        Photoshop-style levels adjustment.
        
        Maps input range to output range with optional gamma correction.
        """
        # Normalize to 0-1
        normalized = (image.astype(float) - input_black) / (input_white - input_black)
        normalized = np.clip(normalized, 0, 1)
        
        # Apply gamma
        normalized = normalized ** (1 / gamma)
        
        # Map to output range
        result = normalized * (output_white - output_black) + output_black
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def curves(image: np.ndarray,
              control_points: list) -> np.ndarray:
        """
        Apply curves adjustment using control points.
        
        control_points: List of (input, output) tuples
                       e.g., [(0, 0), (64, 48), (192, 224), (255, 255)]
        """
        # Sort by input value
        points = sorted(control_points)
        
        # Create lookup table using spline interpolation
        lut = np.zeros(256, dtype=np.uint8)
        
        for i in range(256):
            # Find surrounding control points
            lower_idx = 0
            for j, (inp, out) in enumerate(points):
                if inp <= i:
                    lower_idx = j
                else:
                    break
            
            if lower_idx >= len(points) - 1:
                lut[i] = points[-1][1]
            else:
                # Linear interpolation between points
                x0, y0 = points[lower_idx]
                x1, y1 = points[lower_idx + 1]
                
                if x1 == x0:
                    lut[i] = y0
                else:
                    t = (i - x0) / (x1 - x0)
                    lut[i] = int(y0 + t * (y1 - y0))
        
        return lut[image]
    
    @staticmethod
    def color_balance(image: np.ndarray,
                     shadows: Tuple[float, float, float] = (0, 0, 0),
                     midtones: Tuple[float, float, float] = (0, 0, 0),
                     highlights: Tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
        """
        Photoshop-style color balance.
        
        Adjust color channels separately in shadows, midtones, and highlights.
        Values typically -100 to 100.
        """
        if len(image.shape) != 3:
            return image
        
        result = image.astype(float)
        
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(3):
                    v = result[y, x, c] / 255
                    
                    # Weight by tonal range
                    shadow_weight = 1 - v
                    highlight_weight = v
                    midtone_weight = 1 - abs(2 * v - 1)
                    
                    # Apply adjustments
                    adjustment = (
                        shadows[c] / 100 * shadow_weight +
                        midtones[c] / 100 * midtone_weight +
                        highlights[c] / 100 * highlight_weight
                    ) * 128
                    
                    result[y, x, c] = v * 255 + adjustment
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def vibrance(image: np.ndarray, amount: float) -> np.ndarray:
        """
        Smart saturation - increases saturation more for less-saturated colors.
        
        Unlike regular saturation, doesn't over-saturate already vivid colors.
        """
        if len(image.shape) != 3:
            return image
        
        result = np.zeros_like(image, dtype=np.float64)
        
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                r, g, b = image[y, x] / 255.0
                h, s, v = ColorSpaceConversion.rgb_to_hsv(r, g, b)
                
                # More adjustment for less saturated colors
                adjust = amount * (1 - s)
                s = np.clip(s + adjust, 0, 1)
                
                r, g, b = ColorSpaceConversion.hsv_to_rgb(h, s, v)
                result[y, x] = [r * 255, g * 255, b * 255]
        
        return np.clip(result, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# COLOR EFFECTS
# ═══════════════════════════════════════════════════════════════

class ColorEffects:
    """Artistic color effects"""
    
    @staticmethod
    def grayscale(image: np.ndarray, method: str = 'luminosity') -> np.ndarray:
        """
        Convert to grayscale.
        
        Methods:
        - 'average': Simple average of RGB
        - 'luminosity': Weighted by human perception
        - 'lightness': Average of max and min RGB
        - 'desaturate': Set saturation to 0
        """
        if len(image.shape) != 3:
            return image
        
        if method == 'average':
            gray = np.mean(image, axis=2)
        elif method == 'luminosity':
            gray = (0.299 * image[:, :, 0] + 
                   0.587 * image[:, :, 1] + 
                   0.114 * image[:, :, 2])
        elif method == 'lightness':
            gray = (np.max(image, axis=2) + np.min(image, axis=2)) / 2
        elif method == 'desaturate':
            h, w, c = image.shape
            gray = np.zeros((h, w), dtype=np.float64)
            for y in range(h):
                for x in range(w):
                    r, g, b = image[y, x] / 255.0
                    h, s, l = ColorSpaceConversion.rgb_to_hsl(r, g, b)
                    gray[y, x] = l * 255
        else:
            gray = np.mean(image, axis=2)
        
        return gray.astype(np.uint8)
    
    @staticmethod
    def sepia(image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """
        Apply sepia tone effect.
        """
        if len(image.shape) != 3:
            image = np.stack([image] * 3, axis=2)
        
        # Sepia transform matrix
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        # Apply transform
        result = np.zeros_like(image, dtype=np.float64)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                rgb = image[y, x] / 255.0
                sepia_rgb = sepia_matrix @ rgb
                
                # Blend with original
                result[y, x] = (
                    rgb * (1 - intensity) + sepia_rgb * intensity
                ) * 255
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def duotone(image: np.ndarray,
               shadow_color: Tuple[int, int, int],
               highlight_color: Tuple[int, int, int]) -> np.ndarray:
        """
        Create duotone effect - two colors based on brightness.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = ColorEffects.grayscale(image, 'luminosity')
        else:
            gray = image
        
        # Interpolate between colors
        h, w = gray.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        shadow = np.array(shadow_color)
        highlight = np.array(highlight_color)
        
        for y in range(h):
            for x in range(w):
                t = gray[y, x] / 255.0
                result[y, x] = (shadow * (1 - t) + highlight * t).astype(np.uint8)
        
        return result
    
    @staticmethod
    def tritone(image: np.ndarray,
               shadow_color: Tuple[int, int, int],
               midtone_color: Tuple[int, int, int],
               highlight_color: Tuple[int, int, int]) -> np.ndarray:
        """
        Three-color toning effect.
        """
        if len(image.shape) == 3:
            gray = ColorEffects.grayscale(image, 'luminosity')
        else:
            gray = image
        
        h, w = gray.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        shadow = np.array(shadow_color)
        mid = np.array(midtone_color)
        highlight = np.array(highlight_color)
        
        for y in range(h):
            for x in range(w):
                t = gray[y, x] / 255.0
                
                if t < 0.5:
                    # Shadow to midtone
                    t2 = t * 2
                    result[y, x] = (shadow * (1 - t2) + mid * t2).astype(np.uint8)
                else:
                    # Midtone to highlight
                    t2 = (t - 0.5) * 2
                    result[y, x] = (mid * (1 - t2) + highlight * t2).astype(np.uint8)
        
        return result
    
    @staticmethod
    def color_lookup(image: np.ndarray, 
                    lut: np.ndarray) -> np.ndarray:
        """
        Apply 3D color lookup table (LUT).
        
        lut: 3D array of shape (size, size, size, 3)
             or 2D HALD format
        
        LUTs are used for color grading in film/video.
        """
        h, w, c = image.shape
        result = np.zeros_like(image)
        
        if len(lut.shape) == 4:
            # 3D LUT
            lut_size = lut.shape[0]
            
            for y in range(h):
                for x in range(w):
                    r, g, b = image[y, x] / 255.0
                    
                    # Find LUT indices
                    ri = r * (lut_size - 1)
                    gi = g * (lut_size - 1)
                    bi = b * (lut_size - 1)
                    
                    # Trilinear interpolation
                    r0, g0, b0 = int(ri), int(gi), int(bi)
                    r1 = min(r0 + 1, lut_size - 1)
                    g1 = min(g0 + 1, lut_size - 1)
                    b1 = min(b0 + 1, lut_size - 1)
                    
                    rf, gf, bf = ri - r0, gi - g0, bi - b0
                    
                    # Interpolate
                    c000 = lut[r0, g0, b0]
                    c001 = lut[r0, g0, b1]
                    c010 = lut[r0, g1, b0]
                    c011 = lut[r0, g1, b1]
                    c100 = lut[r1, g0, b0]
                    c101 = lut[r1, g0, b1]
                    c110 = lut[r1, g1, b0]
                    c111 = lut[r1, g1, b1]
                    
                    c00 = c000 * (1 - bf) + c001 * bf
                    c01 = c010 * (1 - bf) + c011 * bf
                    c10 = c100 * (1 - bf) + c101 * bf
                    c11 = c110 * (1 - bf) + c111 * bf
                    
                    c0 = c00 * (1 - gf) + c01 * gf
                    c1 = c10 * (1 - gf) + c11 * gf
                    
                    result[y, x] = (c0 * (1 - rf) + c1 * rf).astype(np.uint8)
        
        return result
    
    @staticmethod
    def split_toning(image: np.ndarray,
                    shadow_hue: float,
                    shadow_saturation: float,
                    highlight_hue: float,
                    highlight_saturation: float,
                    balance: float = 0.5) -> np.ndarray:
        """
        Split toning - different color tints for shadows and highlights.
        
        Hue: 0-360 degrees
        Saturation: 0-1
        Balance: 0 = favor shadows, 1 = favor highlights
        """
        if len(image.shape) != 3:
            image = np.stack([image] * 3, axis=2)
        
        result = np.zeros_like(image, dtype=np.float64)
        
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                r, g, b = image[y, x] / 255.0
                h, s, l = ColorSpaceConversion.rgb_to_hsl(r, g, b)
                
                # Determine shadow/highlight blend based on lightness
                if l < balance:
                    # Shadow region
                    blend = 1 - l / balance
                    tint_h = shadow_hue
                    tint_s = shadow_saturation * blend
                elif l > balance:
                    # Highlight region
                    blend = (l - balance) / (1 - balance)
                    tint_h = highlight_hue
                    tint_s = highlight_saturation * blend
                else:
                    tint_h = h
                    tint_s = 0
                
                # Blend with original color
                new_s = s + tint_s * (1 - s)
                new_h = h if s > 0.01 else tint_h
                if tint_s > 0:
                    # Weighted average of hues
                    h_weight = s / (s + tint_s)
                    # Handle hue wrapping
                    h_diff = tint_h - h
                    if h_diff > 180:
                        h_diff -= 360
                    elif h_diff < -180:
                        h_diff += 360
                    new_h = (h + h_diff * (1 - h_weight)) % 360
                
                r, g, b = ColorSpaceConversion.hsl_to_rgb(new_h, new_s, l)
                result[y, x] = [r * 255, g * 255, b * 255]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def posterize(image: np.ndarray, levels: int = 4) -> np.ndarray:
        """
        Reduce colors to specified number of levels per channel.
        """
        factor = 255 / (levels - 1)
        return (np.round(image / factor) * factor).astype(np.uint8)
    
    @staticmethod
    def threshold(image: np.ndarray, 
                 level: int = 128,
                 method: str = 'binary') -> np.ndarray:
        """
        Apply threshold to convert to black and white.
        
        Methods:
        - 'binary': Simple threshold
        - 'otsu': Automatic optimal threshold
        - 'adaptive': Local adaptive threshold
        """
        if len(image.shape) == 3:
            gray = ColorEffects.grayscale(image, 'luminosity')
        else:
            gray = image
        
        if method == 'binary':
            return ((gray > level) * 255).astype(np.uint8)
        
        elif method == 'otsu':
            # Otsu's automatic thresholding
            hist, _ = np.histogram(gray.flatten(), 256, [0, 256])
            hist = hist.astype(float) / hist.sum()
            
            best_threshold = 0
            best_variance = 0
            
            cumsum = np.cumsum(hist)
            cumsum_weight = np.cumsum(hist * np.arange(256))
            
            for t in range(1, 255):
                w0 = cumsum[t]
                w1 = 1 - w0
                
                if w0 == 0 or w1 == 0:
                    continue
                
                mean0 = cumsum_weight[t] / w0
                mean1 = (cumsum_weight[255] - cumsum_weight[t]) / w1
                
                variance = w0 * w1 * (mean0 - mean1) ** 2
                
                if variance > best_variance:
                    best_variance = variance
                    best_threshold = t
            
            return ((gray > best_threshold) * 255).astype(np.uint8)
        
        elif method == 'adaptive':
            # Adaptive threshold using local mean
            from scipy import ndimage
            
            kernel_size = 15
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
            local_mean = ndimage.convolve(gray.astype(float), kernel)
            
            return ((gray > local_mean - 2) * 255).astype(np.uint8)
        
        return ((gray > level) * 255).astype(np.uint8)
    
    @staticmethod
    def invert(image: np.ndarray) -> np.ndarray:
        """Invert colors (negative)."""
        return 255 - image
    
    @staticmethod
    def solarize(image: np.ndarray, threshold: int = 128) -> np.ndarray:
        """
        Solarize effect - invert values above threshold.
        
        Creates surreal, partially-inverted images.
        """
        result = image.copy()
        mask = image > threshold
        result[mask] = 255 - result[mask]
        return result
```

---

## Chapter 16: Geometric Transformations

### 16.1 Fundamental Transforms

```python
"""
GEOMETRIC TRANSFORMATIONS - COMPREHENSIVE GUIDE

Complete coverage of 2D and 3D geometric transformations
for image manipulation and procedural art.
"""

import numpy as np
import math
from typing import Tuple, Optional, Callable
from PIL import Image

# ═══════════════════════════════════════════════════════════════
# AFFINE TRANSFORMATIONS
# ═══════════════════════════════════════════════════════════════

class AffineTransform:
    """
    Affine transformation using 3x3 homogeneous matrix.
    
    Preserves:
    - Parallelism
    - Ratios of distances along a line
    
    Can represent:
    - Translation
    - Rotation
    - Scaling
    - Shearing
    - Any combination of the above
    """
    
    def __init__(self, matrix: np.ndarray = None):
        if matrix is None:
            self.matrix = np.eye(3)
        else:
            self.matrix = matrix.copy()
    
    @staticmethod
    def identity() -> 'AffineTransform':
        """Create identity transform."""
        return AffineTransform(np.eye(3))
    
    @staticmethod
    def translation(tx: float, ty: float) -> 'AffineTransform':
        """
        Create translation transform.
        
        [1  0  tx]   [x]   [x + tx]
        [0  1  ty] * [y] = [y + ty]
        [0  0  1 ]   [1]   [1     ]
        """
        return AffineTransform(np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ], dtype=np.float64))
    
    @staticmethod
    def rotation(angle_degrees: float, 
                center: Tuple[float, float] = (0, 0)) -> 'AffineTransform':
        """
        Create rotation transform.
        
        Rotates counter-clockwise around center point.
        """
        theta = math.radians(angle_degrees)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        cx, cy = center
        
        # Translate to origin, rotate, translate back
        return (
            AffineTransform.translation(cx, cy)
            .compose(AffineTransform(np.array([
                [cos_t, -sin_t, 0],
                [sin_t, cos_t, 0],
                [0, 0, 1]
            ])))
            .compose(AffineTransform.translation(-cx, -cy))
        )
    
    @staticmethod
    def scale(sx: float, sy: float = None,
             center: Tuple[float, float] = (0, 0)) -> 'AffineTransform':
        """
        Create scaling transform.
        
        sx, sy: Scale factors (sy defaults to sx for uniform)
        """
        if sy is None:
            sy = sx
        cx, cy = center
        
        return (
            AffineTransform.translation(cx, cy)
            .compose(AffineTransform(np.array([
                [sx, 0, 0],
                [0, sy, 0],
                [0, 0, 1]
            ])))
            .compose(AffineTransform.translation(-cx, -cy))
        )
    
    @staticmethod
    def shear(shx: float = 0, shy: float = 0) -> 'AffineTransform':
        """
        Create shearing transform.
        
        shx: Horizontal shear (x += shx * y)
        shy: Vertical shear (y += shy * x)
        """
        return AffineTransform(np.array([
            [1, shx, 0],
            [shy, 1, 0],
            [0, 0, 1]
        ], dtype=np.float64))
    
    @staticmethod
    def reflection(axis: str = 'x', 
                  point: float = 0) -> 'AffineTransform':
        """
        Create reflection transform.
        
        axis: 'x' (flip vertical), 'y' (flip horizontal), 
              'origin' (flip both), 'line' (reflect across y=x)
        """
        if axis == 'x':
            return AffineTransform(np.array([
                [1, 0, 0],
                [0, -1, 2 * point],
                [0, 0, 1]
            ], dtype=np.float64))
        elif axis == 'y':
            return AffineTransform(np.array([
                [-1, 0, 2 * point],
                [0, 1, 0],
                [0, 0, 1]
            ], dtype=np.float64))
        elif axis == 'origin':
            return AffineTransform(np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ], dtype=np.float64))
        elif axis == 'line':  # y = x
            return AffineTransform(np.array([
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1]
            ], dtype=np.float64))
    
    def compose(self, other: 'AffineTransform') -> 'AffineTransform':
        """
        Compose with another transform (this * other).
        
        The resulting transform applies 'other' first, then 'this'.
        """
        return AffineTransform(self.matrix @ other.matrix)
    
    def inverse(self) -> 'AffineTransform':
        """Get inverse transform."""
        return AffineTransform(np.linalg.inv(self.matrix))
    
    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """Transform a single point."""
        p = np.array([x, y, 1])
        result = self.matrix @ p
        return result[0], result[1]
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform array of points (N x 2)."""
        ones = np.ones((points.shape[0], 1))
        homogeneous = np.hstack([points, ones])
        transformed = (self.matrix @ homogeneous.T).T
        return transformed[:, :2]
    
    def apply_to_image(self, image: np.ndarray,
                      output_size: Tuple[int, int] = None,
                      interpolation: str = 'bilinear',
                      fill_value: int = 0) -> np.ndarray:
        """
        Apply transform to image.
        
        Uses inverse mapping for proper sampling.
        """
        h, w = image.shape[:2]
        if output_size is None:
            output_size = (w, h)
        out_w, out_h = output_size
        
        if len(image.shape) == 3:
            result = np.full((out_h, out_w, image.shape[2]), fill_value, dtype=np.uint8)
        else:
            result = np.full((out_h, out_w), fill_value, dtype=np.uint8)
        
        # Inverse transform for backward mapping
        inv_matrix = np.linalg.inv(self.matrix)
        
        for out_y in range(out_h):
            for out_x in range(out_w):
                # Find source coordinates
                src = inv_matrix @ np.array([out_x, out_y, 1])
                src_x, src_y = src[0], src[1]
                
                if interpolation == 'nearest':
                    ix, iy = int(round(src_x)), int(round(src_y))
                    if 0 <= ix < w and 0 <= iy < h:
                        result[out_y, out_x] = image[iy, ix]
                
                elif interpolation == 'bilinear':
                    if 0 <= src_x < w - 1 and 0 <= src_y < h - 1:
                        x0, y0 = int(src_x), int(src_y)
                        x1, y1 = x0 + 1, y0 + 1
                        fx, fy = src_x - x0, src_y - y0
                        
                        if len(image.shape) == 3:
                            for c in range(image.shape[2]):
                                result[out_y, out_x, c] = (
                                    image[y0, x0, c] * (1-fx) * (1-fy) +
                                    image[y0, x1, c] * fx * (1-fy) +
                                    image[y1, x0, c] * (1-fx) * fy +
                                    image[y1, x1, c] * fx * fy
                                )
                        else:
                            result[out_y, out_x] = (
                                image[y0, x0] * (1-fx) * (1-fy) +
                                image[y0, x1] * fx * (1-fy) +
                                image[y1, x0] * (1-fx) * fy +
                                image[y1, x1] * fx * fy
                            )
        
        return result


# ═══════════════════════════════════════════════════════════════
# PERSPECTIVE TRANSFORMATIONS
# ═══════════════════════════════════════════════════════════════

class PerspectiveTransform:
    """
    Perspective (projective) transformation using 3x3 matrix.
    
    More general than affine - can represent any perspective view.
    Does NOT preserve parallelism.
    """
    
    def __init__(self, matrix: np.ndarray = None):
        if matrix is None:
            self.matrix = np.eye(3)
        else:
            self.matrix = matrix.copy()
    
    @staticmethod
    def from_four_points(src: list, dst: list) -> 'PerspectiveTransform':
        """
        Compute perspective transform from 4 corresponding point pairs.
        
        src, dst: Lists of 4 (x, y) tuples representing corners.
        """
        # Build system of equations
        A = []
        b = []
        
        for (sx, sy), (dx, dy) in zip(src, dst):
            A.append([sx, sy, 1, 0, 0, 0, -dx*sx, -dx*sy])
            A.append([0, 0, 0, sx, sy, 1, -dy*sx, -dy*sy])
            b.append(dx)
            b.append(dy)
        
        A = np.array(A)
        b = np.array(b)
        
        # Solve for transform parameters
        h = np.linalg.lstsq(A, b, rcond=None)[0]
        
        matrix = np.array([
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], 1]
        ])
        
        return PerspectiveTransform(matrix)
    
    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """Transform point using homogeneous division."""
        p = np.array([x, y, 1])
        result = self.matrix @ p
        return result[0] / result[2], result[1] / result[2]
    
    def inverse(self) -> 'PerspectiveTransform':
        """Get inverse transform."""
        return PerspectiveTransform(np.linalg.inv(self.matrix))
    
    def apply_to_image(self, image: np.ndarray,
                      output_size: Tuple[int, int] = None,
                      interpolation: str = 'bilinear') -> np.ndarray:
        """Apply perspective transform to image."""
        h, w = image.shape[:2]
        if output_size is None:
            output_size = (w, h)
        out_w, out_h = output_size
        
        if len(image.shape) == 3:
            result = np.zeros((out_h, out_w, image.shape[2]), dtype=np.uint8)
        else:
            result = np.zeros((out_h, out_w), dtype=np.uint8)
        
        inv_matrix = np.linalg.inv(self.matrix)
        
        for out_y in range(out_h):
            for out_x in range(out_w):
                src = inv_matrix @ np.array([out_x, out_y, 1])
                src_x, src_y = src[0] / src[2], src[1] / src[2]
                
                if 0 <= src_x < w - 1 and 0 <= src_y < h - 1:
                    x0, y0 = int(src_x), int(src_y)
                    fx, fy = src_x - x0, src_y - y0
                    
                    if len(image.shape) == 3:
                        for c in range(image.shape[2]):
                            result[out_y, out_x, c] = (
                                image[y0, x0, c] * (1-fx) * (1-fy) +
                                image[y0, x0+1, c] * fx * (1-fy) +
                                image[y0+1, x0, c] * (1-fx) * fy +
                                image[y0+1, x0+1, c] * fx * fy
                            )
                    else:
                        result[out_y, out_x] = (
                            image[y0, x0] * (1-fx) * (1-fy) +
                            image[y0, x0+1] * fx * (1-fy) +
                            image[y0+1, x0] * (1-fx) * fy +
                            image[y0+1, x0+1] * fx * fy
                        )
        
        return result


# ═══════════════════════════════════════════════════════════════
# DISTORTION EFFECTS
# ═══════════════════════════════════════════════════════════════

class DistortionEffects:
    """Non-linear geometric distortion effects"""
    
    @staticmethod
    def apply_distortion(image: np.ndarray,
                        map_func: Callable,
                        output_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Apply arbitrary distortion using mapping function.
        
        map_func(x, y, w, h) -> (src_x, src_y)
        """
        h, w = image.shape[:2]
        if output_size is None:
            out_w, out_h = w, h
        else:
            out_w, out_h = output_size
        
        if len(image.shape) == 3:
            result = np.zeros((out_h, out_w, image.shape[2]), dtype=np.uint8)
        else:
            result = np.zeros((out_h, out_w), dtype=np.uint8)
        
        for out_y in range(out_h):
            for out_x in range(out_w):
                src_x, src_y = map_func(out_x, out_y, out_w, out_h)
                
                if 0 <= src_x < w - 1 and 0 <= src_y < h - 1:
                    x0, y0 = int(src_x), int(src_y)
                    fx, fy = src_x - x0, src_y - y0
                    
                    if len(image.shape) == 3:
                        for c in range(image.shape[2]):
                            result[out_y, out_x, c] = (
                                image[y0, x0, c] * (1-fx) * (1-fy) +
                                image[y0, x0+1, c] * fx * (1-fy) +
                                image[y0+1, x0, c] * (1-fx) * fy +
                                image[y0+1, x0+1, c] * fx * fy
                            )
                    else:
                        result[out_y, out_x] = (
                            image[y0, x0] * (1-fx) * (1-fy) +
                            image[y0, x0+1] * fx * (1-fy) +
                            image[y0+1, x0] * (1-fx) * fy +
                            image[y0+1, x0+1] * fx * fy
                        )
        
        return result
    
    @staticmethod
    def barrel_distortion(image: np.ndarray,
                         k1: float = 0.1,
                         k2: float = 0.0) -> np.ndarray:
        """
        Barrel/pincushion distortion.
        
        k1 > 0: Barrel (edges curve outward)
        k1 < 0: Pincushion (edges curve inward)
        k2: Higher-order correction
        """
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2
        max_r = math.sqrt(cx**2 + cy**2)
        
        def map_func(x, y, out_w, out_h):
            # Normalize to [-1, 1]
            nx = (x - cx) / max_r
            ny = (y - cy) / max_r
            
            r = math.sqrt(nx**2 + ny**2)
            
            # Radial distortion
            factor = 1 + k1 * r**2 + k2 * r**4
            
            src_x = cx + nx * max_r * factor
            src_y = cy + ny * max_r * factor
            
            return src_x, src_y
        
        return DistortionEffects.apply_distortion(image, map_func)
    
    @staticmethod
    def fisheye(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Fisheye lens distortion.
        """
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2
        max_r = min(cx, cy)
        
        def map_func(x, y, out_w, out_h):
            dx = x - cx
            dy = y - cy
            r = math.sqrt(dx**2 + dy**2)
            
            if r == 0:
                return x, y
            
            # Fisheye mapping
            theta = math.atan2(r, max_r * strength)
            r_new = max_r * theta / (math.pi / 2)
            
            factor = r_new / r if r > 0 else 1
            
            src_x = cx + dx * factor
            src_y = cy + dy * factor
            
            return src_x, src_y
        
        return DistortionEffects.apply_distortion(image, map_func)
    
    @staticmethod
    def swirl(image: np.ndarray,
             strength: float = 1.0,
             radius: float = None) -> np.ndarray:
        """
        Swirl/twirl distortion.
        
        Rotates pixels based on distance from center.
        """
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2
        
        if radius is None:
            radius = min(w, h) / 2
        
        def map_func(x, y, out_w, out_h):
            dx = x - cx
            dy = y - cy
            r = math.sqrt(dx**2 + dy**2)
            
            if r > radius or r == 0:
                return x, y
            
            # Rotation angle decreases with distance
            angle = strength * (1 - r / radius) * math.pi
            
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            
            src_x = cx + dx * cos_a - dy * sin_a
            src_y = cy + dx * sin_a + dy * cos_a
            
            return src_x, src_y
        
        return DistortionEffects.apply_distortion(image, map_func)
    
    @staticmethod
    def spherize(image: np.ndarray,
                strength: float = 1.0) -> np.ndarray:
        """
        Spherical distortion - wrap image around a sphere.
        """
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2
        radius = min(cx, cy)
        
        def map_func(x, y, out_w, out_h):
            dx = (x - cx) / radius
            dy = (y - cy) / radius
            r = math.sqrt(dx**2 + dy**2)
            
            if r > 1:
                return x, y
            
            # Spherical mapping
            z = math.sqrt(1 - r**2) if r < 1 else 0
            
            # Project back
            factor = (1 - strength) + strength * (1 - z)
            
            src_x = cx + dx * radius * factor
            src_y = cy + dy * radius * factor
            
            return src_x, src_y
        
        return DistortionEffects.apply_distortion(image, map_func)
    
    @staticmethod
    def pinch(image: np.ndarray,
             strength: float = 0.5,
             radius: float = None) -> np.ndarray:
        """
        Pinch effect - pull pixels toward or away from center.
        
        strength > 0: Pull toward center
        strength < 0: Push away from center
        """
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2
        
        if radius is None:
            radius = min(w, h) / 2
        
        def map_func(x, y, out_w, out_h):
            dx = x - cx
            dy = y - cy
            r = math.sqrt(dx**2 + dy**2)
            
            if r > radius or r == 0:
                return x, y
            
            # Pinch factor
            t = r / radius
            factor = (1 - strength) + strength * t
            
            src_x = cx + dx / factor
            src_y = cy + dy / factor
            
            return src_x, src_y
        
        return DistortionEffects.apply_distortion(image, map_func)
    
    @staticmethod
    def wave(image: np.ndarray,
            amplitude: float = 10,
            wavelength: float = 50,
            direction: str = 'horizontal') -> np.ndarray:
        """
        Sine wave distortion.
        
        direction: 'horizontal', 'vertical', or 'both'
        """
        def map_func(x, y, out_w, out_h):
            if direction == 'horizontal':
                offset_x = amplitude * math.sin(2 * math.pi * y / wavelength)
                offset_y = 0
            elif direction == 'vertical':
                offset_x = 0
                offset_y = amplitude * math.sin(2 * math.pi * x / wavelength)
            else:  # both
                offset_x = amplitude * math.sin(2 * math.pi * y / wavelength)
                offset_y = amplitude * math.sin(2 * math.pi * x / wavelength)
            
            return x - offset_x, y - offset_y
        
        return DistortionEffects.apply_distortion(image, map_func)
    
    @staticmethod
    def ripple(image: np.ndarray,
              amplitude: float = 5,
              wavelength: float = 30,
              center: Tuple[float, float] = None) -> np.ndarray:
        """
        Circular ripple distortion (water drop effect).
        """
        h, w = image.shape[:2]
        if center is None:
            cx, cy = w / 2, h / 2
        else:
            cx, cy = center
        
        def map_func(x, y, out_w, out_h):
            dx = x - cx
            dy = y - cy
            r = math.sqrt(dx**2 + dy**2)
            
            if r == 0:
                return x, y
            
            offset = amplitude * math.sin(2 * math.pi * r / wavelength)
            
            factor = (r + offset) / r
            
            src_x = cx + dx * factor
            src_y = cy + dy * factor
            
            return src_x, src_y
        
        return DistortionEffects.apply_distortion(image, map_func)
    
    @staticmethod
    def polar_to_cartesian(image: np.ndarray) -> np.ndarray:
        """
        Convert polar coordinates to Cartesian (unwrap circle).
        """
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2
        max_r = min(cx, cy)
        
        out_w = int(2 * math.pi * max_r)
        out_h = int(max_r)
        
        def map_func(x, y, out_w, out_h):
            # x -> angle, y -> radius
            angle = x / out_w * 2 * math.pi
            radius = y / out_h * max_r
            
            src_x = cx + radius * math.cos(angle)
            src_y = cy + radius * math.sin(angle)
            
            return src_x, src_y
        
        return DistortionEffects.apply_distortion(image, map_func, (out_w, out_h))
    
    @staticmethod
    def cartesian_to_polar(image: np.ndarray) -> np.ndarray:
        """
        Convert Cartesian to polar coordinates (wrap into circle).
        """
        h, w = image.shape[:2]
        
        out_size = max(w, h)
        cx, cy = out_size / 2, out_size / 2
        max_r = out_size / 2
        
        def map_func(x, y, out_w, out_h):
            dx = x - cx
            dy = y - cy
            r = math.sqrt(dx**2 + dy**2)
            angle = math.atan2(dy, dx)
            
            if angle < 0:
                angle += 2 * math.pi
            
            # radius -> y, angle -> x
            src_x = angle / (2 * math.pi) * w
            src_y = r / max_r * h
            
            return src_x, src_y
        
        return DistortionEffects.apply_distortion(image, map_func, (out_size, out_size))
    
    @staticmethod
    def kaleidoscope(image: np.ndarray,
                    segments: int = 6,
                    angle_offset: float = 0) -> np.ndarray:
        """
        Kaleidoscope effect - mirror segments around center.
        """
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2
        segment_angle = 2 * math.pi / segments
        
        def map_func(x, y, out_w, out_h):
            dx = x - cx
            dy = y - cy
            r = math.sqrt(dx**2 + dy**2)
            angle = math.atan2(dy, dx) + angle_offset
            
            # Fold angle into first segment
            angle = angle % segment_angle
            
            # Mirror every other segment
            if int((math.atan2(dy, dx) + angle_offset) / segment_angle) % 2 == 1:
                angle = segment_angle - angle
            
            src_x = cx + r * math.cos(angle)
            src_y = cy + r * math.sin(angle)
            
            return src_x, src_y
        
        return DistortionEffects.apply_distortion(image, map_func)
    
    @staticmethod
    def glass_tiles(image: np.ndarray,
                   tile_size: int = 20,
                   displacement: float = 5) -> np.ndarray:
        """
        Glass block/tile distortion effect.
        """
        def map_func(x, y, out_w, out_h):
            # Position within tile
            tx = (x % tile_size) / tile_size - 0.5
            ty = (y % tile_size) / tile_size - 0.5
            
            # Distort based on position in tile
            dist_x = displacement * math.sin(tx * math.pi)
            dist_y = displacement * math.sin(ty * math.pi)
            
            return x + dist_x, y + dist_y
        
        return DistortionEffects.apply_distortion(image, map_func)


# ═══════════════════════════════════════════════════════════════
# CROP AND RESIZE OPERATIONS
# ═══════════════════════════════════════════════════════════════

class ImageResizing:
    """Image resizing with various algorithms"""
    
    @staticmethod
    def nearest_neighbor(image: np.ndarray,
                        new_width: int,
                        new_height: int) -> np.ndarray:
        """
        Nearest neighbor interpolation.
        
        Fast but produces blocky results.
        Good for: Pixel art, preserving sharp edges.
        """
        h, w = image.shape[:2]
        
        if len(image.shape) == 3:
            result = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)
        else:
            result = np.zeros((new_height, new_width), dtype=np.uint8)
        
        x_ratio = w / new_width
        y_ratio = h / new_height
        
        for y in range(new_height):
            for x in range(new_width):
                src_x = int(x * x_ratio)
                src_y = int(y * y_ratio)
                result[y, x] = image[src_y, src_x]
        
        return result
    
    @staticmethod
    def bilinear(image: np.ndarray,
                new_width: int,
                new_height: int) -> np.ndarray:
        """
        Bilinear interpolation.
        
        Smooth results, moderate quality.
        Good balance of speed and quality.
        """
        h, w = image.shape[:2]
        
        if len(image.shape) == 3:
            result = np.zeros((new_height, new_width, image.shape[2]), dtype=np.float64)
            channels = image.shape[2]
        else:
            result = np.zeros((new_height, new_width), dtype=np.float64)
            channels = 1
        
        x_ratio = (w - 1) / (new_width - 1) if new_width > 1 else 0
        y_ratio = (h - 1) / (new_height - 1) if new_height > 1 else 0
        
        for y in range(new_height):
            for x in range(new_width):
                src_x = x * x_ratio
                src_y = y * y_ratio
                
                x0 = int(src_x)
                y0 = int(src_y)
                x1 = min(x0 + 1, w - 1)
                y1 = min(y0 + 1, h - 1)
                
                fx = src_x - x0
                fy = src_y - y0
                
                if channels > 1:
                    for c in range(channels):
                        result[y, x, c] = (
                            image[y0, x0, c] * (1-fx) * (1-fy) +
                            image[y0, x1, c] * fx * (1-fy) +
                            image[y1, x0, c] * (1-fx) * fy +
                            image[y1, x1, c] * fx * fy
                        )
                else:
                    result[y, x] = (
                        image[y0, x0] * (1-fx) * (1-fy) +
                        image[y0, x1] * fx * (1-fy) +
                        image[y1, x0] * (1-fx) * fy +
                        image[y1, x1] * fx * fy
                    )
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def bicubic(image: np.ndarray,
               new_width: int,
               new_height: int) -> np.ndarray:
        """
        Bicubic interpolation.
        
        High quality, smooth gradients.
        Best for photographic images.
        """
        def cubic_kernel(t):
            """Mitchell-Netravali cubic filter"""
            a = -0.5
            t = abs(t)
            if t <= 1:
                return (a + 2) * t**3 - (a + 3) * t**2 + 1
            elif t < 2:
                return a * t**3 - 5*a * t**2 + 8*a * t - 4*a
            return 0
        
        h, w = image.shape[:2]
        
        if len(image.shape) == 3:
            result = np.zeros((new_height, new_width, image.shape[2]), dtype=np.float64)
            channels = image.shape[2]
        else:
            result = np.zeros((new_height, new_width), dtype=np.float64)
            channels = 1
        
        x_ratio = w / new_width
        y_ratio = h / new_height
        
        for y in range(new_height):
            for x in range(new_width):
                src_x = x * x_ratio
                src_y = y * y_ratio
                
                x0 = int(src_x)
                y0 = int(src_y)
                
                value = np.zeros(channels if channels > 1 else 1)
                weight_sum = 0
                
                for j in range(-1, 3):
                    for i in range(-1, 3):
                        xi = x0 + i
                        yi = y0 + j
                        
                        if 0 <= xi < w and 0 <= yi < h:
                            wx = cubic_kernel(src_x - xi)
                            wy = cubic_kernel(src_y - yi)
                            weight = wx * wy
                            
                            if channels > 1:
                                value += weight * image[yi, xi]
                            else:
                                value[0] += weight * image[yi, xi]
                            
                            weight_sum += weight
                
                if weight_sum > 0:
                    value /= weight_sum
                
                if channels > 1:
                    result[y, x] = value
                else:
                    result[y, x] = value[0]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def lanczos(image: np.ndarray,
               new_width: int,
               new_height: int,
               a: int = 3) -> np.ndarray:
        """
        Lanczos resampling.
        
        Highest quality, best for downscaling.
        Preserves detail while avoiding aliasing.
        
        a: Kernel size (2 or 3 typical)
        """
        def sinc(x):
            if x == 0:
                return 1
            return math.sin(math.pi * x) / (math.pi * x)
        
        def lanczos_kernel(x, a):
            if abs(x) >= a:
                return 0
            return sinc(x) * sinc(x / a)
        
        h, w = image.shape[:2]
        
        if len(image.shape) == 3:
            result = np.zeros((new_height, new_width, image.shape[2]), dtype=np.float64)
            channels = image.shape[2]
        else:
            result = np.zeros((new_height, new_width), dtype=np.float64)
            channels = 1
        
        x_ratio = w / new_width
        y_ratio = h / new_height
        
        for y in range(new_height):
            for x in range(new_width):
                src_x = (x + 0.5) * x_ratio - 0.5
                src_y = (y + 0.5) * y_ratio - 0.5
                
                x0 = int(math.floor(src_x))
                y0 = int(math.floor(src_y))
                
                value = np.zeros(channels if channels > 1 else 1)
                weight_sum = 0
                
                for j in range(-a + 1, a + 1):
                    for i in range(-a + 1, a + 1):
                        xi = x0 + i
                        yi = y0 + j
                        
                        if 0 <= xi < w and 0 <= yi < h:
                            wx = lanczos_kernel(src_x - xi, a)
                            wy = lanczos_kernel(src_y - yi, a)
                            weight = wx * wy
                            
                            if channels > 1:
                                value += weight * image[yi, xi]
                            else:
                                value[0] += weight * image[yi, xi]
                            
                            weight_sum += weight
                
                if weight_sum > 0:
                    value /= weight_sum
                
                if channels > 1:
                    result[y, x] = value
                else:
                    result[y, x] = value[0]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def seam_carving(image: np.ndarray,
                    new_width: int,
                    new_height: int = None) -> np.ndarray:
        """
        Content-aware resize using seam carving.
        
        Removes low-energy seams to resize while preserving
        important image features.
        """
        if new_height is None:
            new_height = image.shape[0]
        
        current = image.copy()
        
        # Remove vertical seams
        while current.shape[1] > new_width:
            current = ImageResizing._remove_vertical_seam(current)
        
        # Remove horizontal seams
        while current.shape[0] > new_height:
            current = ImageResizing._remove_horizontal_seam(current)
        
        return current
    
    @staticmethod
    def _calculate_energy(image: np.ndarray) -> np.ndarray:
        """Calculate energy map using gradient magnitude."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)
        
        # Sobel gradients
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        
        return np.sqrt(gx**2 + gy**2)
    
    @staticmethod
    def _remove_vertical_seam(image: np.ndarray) -> np.ndarray:
        """Remove lowest energy vertical seam."""
        h, w = image.shape[:2]
        energy = ImageResizing._calculate_energy(image)
        
        # Dynamic programming to find minimum seam
        M = energy.copy()
        backtrack = np.zeros((h, w), dtype=int)
        
        for y in range(1, h):
            for x in range(w):
                if x == 0:
                    candidates = [M[y-1, x], M[y-1, x+1]]
                    offsets = [0, 1]
                elif x == w - 1:
                    candidates = [M[y-1, x-1], M[y-1, x]]
                    offsets = [-1, 0]
                else:
                    candidates = [M[y-1, x-1], M[y-1, x], M[y-1, x+1]]
                    offsets = [-1, 0, 1]
                
                min_idx = np.argmin(candidates)
                M[y, x] = energy[y, x] + candidates[min_idx]
                backtrack[y, x] = offsets[min_idx]
        
        # Find seam
        seam = np.zeros(h, dtype=int)
        seam[-1] = np.argmin(M[-1])
        
        for y in range(h - 2, -1, -1):
            seam[y] = seam[y + 1] + backtrack[y + 1, seam[y + 1]]
        
        # Remove seam
        if len(image.shape) == 3:
            result = np.zeros((h, w - 1, image.shape[2]), dtype=image.dtype)
            for y in range(h):
                result[y, :seam[y]] = image[y, :seam[y]]
                result[y, seam[y]:] = image[y, seam[y] + 1:]
        else:
            result = np.zeros((h, w - 1), dtype=image.dtype)
            for y in range(h):
                result[y, :seam[y]] = image[y, :seam[y]]
                result[y, seam[y]:] = image[y, seam[y] + 1:]
        
        return result
    
    @staticmethod
    def _remove_horizontal_seam(image: np.ndarray) -> np.ndarray:
        """Remove lowest energy horizontal seam."""
        # Transpose, remove vertical seam, transpose back
        if len(image.shape) == 3:
            transposed = np.transpose(image, (1, 0, 2))
            result = ImageResizing._remove_vertical_seam(transposed)
            return np.transpose(result, (1, 0, 2))
        else:
            transposed = image.T
            result = ImageResizing._remove_vertical_seam(transposed)
            return result.T
```

---

# PART V: FILTERS AND EFFECTS

---

## Chapter 17: Blur and Sharpening Effects

### 17.1 Advanced Blur Techniques

```python
"""
BLUR AND SHARPENING EFFECTS - COMPREHENSIVE GUIDE

From basic blur to advanced depth-of-field and focus effects.
"""

import numpy as np
import math
from scipy import ndimage, signal
from typing import Tuple, Optional, List

# ═══════════════════════════════════════════════════════════════
# BLUR EFFECTS
# ═══════════════════════════════════════════════════════════════

class BlurEffects:
    """Complete blur effect library"""
    
    @staticmethod
    def box_blur(image: np.ndarray, radius: int) -> np.ndarray:
        """
        Simple box blur (mean filter).
        
        Fast but produces blocky artifacts.
        """
        size = 2 * radius + 1
        kernel = np.ones((size, size)) / (size * size)
        
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=np.float64)
            for c in range(image.shape[2]):
                result[:, :, c] = ndimage.convolve(
                    image[:, :, c].astype(float), kernel
                )
            return np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = ndimage.convolve(image.astype(float), kernel)
            return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def gaussian_blur(image: np.ndarray, 
                     sigma: float,
                     radius: int = None) -> np.ndarray:
        """
        Gaussian blur - smooth, natural-looking blur.
        
        Most common blur for photo editing.
        """
        if radius is None:
            radius = int(3 * sigma)
        
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=np.float64)
            for c in range(image.shape[2]):
                result[:, :, c] = ndimage.gaussian_filter(
                    image[:, :, c].astype(float), sigma
                )
            return np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = ndimage.gaussian_filter(image.astype(float), sigma)
            return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def motion_blur(image: np.ndarray,
                   length: int,
                   angle: float = 0) -> np.ndarray:
        """
        Motion blur - simulates camera or object motion.
        
        length: Blur distance in pixels
        angle: Motion direction in degrees
        """
        # Create motion blur kernel
        kernel = np.zeros((length, length))
        center = length // 2
        
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))
        
        for i in range(length):
            t = i - center
            x = int(center + t * cos_a)
            y = int(center + t * sin_a)
            
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1
        
        kernel /= kernel.sum()
        
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=np.float64)
            for c in range(image.shape[2]):
                result[:, :, c] = ndimage.convolve(
                    image[:, :, c].astype(float), kernel
                )
            return np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = ndimage.convolve(image.astype(float), kernel)
            return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def radial_blur(image: np.ndarray,
                   strength: float = 0.1,
                   center: Tuple[float, float] = None) -> np.ndarray:
        """
        Radial/zoom blur - blur emanating from center.
        
        Simulates zooming camera or forward motion.
        """
        h, w = image.shape[:2]
        if center is None:
            cx, cy = w / 2, h / 2
        else:
            cx, cy = center
        
        # Number of samples for blur
        num_samples = max(1, int(20 * strength))
        
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=np.float64)
        else:
            result = np.zeros_like(image, dtype=np.float64)
        
        for sample in range(num_samples):
            # Scale factor for this sample
            scale = 1 + (sample / num_samples) * strength * 0.2
            
            # Apply zoom transform
            transform = AffineTransform.scale(scale, scale, (cx, cy))
            
            # Simple nearest neighbor for speed
            for y in range(h):
                for x in range(w):
                    src_x, src_y = transform.inverse().transform_point(x, y)
                    
                    ix = int(src_x)
                    iy = int(src_y)
                    
                    if 0 <= ix < w and 0 <= iy < h:
                        result[y, x] += image[iy, ix]
        
        result /= num_samples
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def spin_blur(image: np.ndarray,
                 angle: float,
                 center: Tuple[float, float] = None) -> np.ndarray:
        """
        Spin/rotational blur - blur in circular direction.
        
        Simulates rotating camera or spinning subject.
        """
        h, w = image.shape[:2]
        if center is None:
            cx, cy = w / 2, h / 2
        else:
            cx, cy = center
        
        num_samples = max(1, int(abs(angle) / 2))
        
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=np.float64)
        else:
            result = np.zeros_like(image, dtype=np.float64)
        
        for sample in range(num_samples):
            # Rotation angle for this sample
            rot = -angle / 2 + (sample / num_samples) * angle
            
            # Create rotation transform
            transform = AffineTransform.rotation(rot, (cx, cy))
            
            for y in range(h):
                for x in range(w):
                    src_x, src_y = transform.inverse().transform_point(x, y)
                    
                    ix = int(src_x)
                    iy = int(src_y)
                    
                    if 0 <= ix < w and 0 <= iy < h:
                        result[y, x] += image[iy, ix]
        
        result /= num_samples
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def surface_blur(image: np.ndarray,
                    radius: int = 5,
                    threshold: float = 30) -> np.ndarray:
        """
        Surface blur - blur while preserving edges.
        
        Only blurs pixels that are similar to the center pixel.
        Great for skin smoothing while preserving features.
        """
        h, w = image.shape[:2]
        
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=np.float64)
            
            for y in range(h):
                for x in range(w):
                    center_color = image[y, x].astype(float)
                    
                    total = np.zeros(3)
                    weight_sum = 0
                    
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            ny, nx = y + dy, x + dx
                            
                            if 0 <= ny < h and 0 <= nx < w:
                                neighbor = image[ny, nx].astype(float)
                                
                                # Color difference
                                diff = np.sqrt(np.sum((neighbor - center_color)**2))
                                
                                # Weight based on color similarity
                                if diff < threshold:
                                    weight = 1 - diff / threshold
                                    total += neighbor * weight
                                    weight_sum += weight
                    
                    if weight_sum > 0:
                        result[y, x] = total / weight_sum
                    else:
                        result[y, x] = center_color
        else:
            result = np.zeros_like(image, dtype=np.float64)
            
            for y in range(h):
                for x in range(w):
                    center_val = float(image[y, x])
                    
                    total = 0
                    weight_sum = 0
                    
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            ny, nx = y + dy, x + dx
                            
                            if 0 <= ny < h and 0 <= nx < w:
                                neighbor = float(image[ny, nx])
                                diff = abs(neighbor - center_val)
                                
                                if diff < threshold:
                                    weight = 1 - diff / threshold
                                    total += neighbor * weight
                                    weight_sum += weight
                    
                    if weight_sum > 0:
                        result[y, x] = total / weight_sum
                    else:
                        result[y, x] = center_val
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def bilateral_filter(image: np.ndarray,
                        sigma_spatial: float = 3,
                        sigma_range: float = 30) -> np.ndarray:
        """
        Bilateral filter - edge-preserving smoothing.
        
        Considers both spatial distance and intensity difference.
        Excellent for noise reduction while preserving edges.
        """
        h, w = image.shape[:2]
        radius = int(3 * sigma_spatial)
        
        # Precompute spatial Gaussian
        y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
        spatial_weights = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma_spatial**2))
        
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=np.float64)
            
            for c in range(image.shape[2]):
                channel = image[:, :, c].astype(float)
                
                for y in range(h):
                    for x in range(w):
                        y_min = max(0, y - radius)
                        y_max = min(h, y + radius + 1)
                        x_min = max(0, x - radius)
                        x_max = min(w, x + radius + 1)
                        
                        # Extract region
                        region = channel[y_min:y_max, x_min:x_max]
                        
                        # Adjust spatial weights to region size
                        sw_y_start = radius - (y - y_min)
                        sw_y_end = radius + (y_max - y)
                        sw_x_start = radius - (x - x_min)
                        sw_x_end = radius + (x_max - x)
                        
                        sw = spatial_weights[sw_y_start:sw_y_end, sw_x_start:sw_x_end]
                        
                        # Range weights
                        intensity_diff = region - channel[y, x]
                        range_weights = np.exp(-intensity_diff**2 / (2 * sigma_range**2))
                        
                        # Combined weights
                        weights = sw * range_weights
                        
                        result[y, x, c] = np.sum(region * weights) / np.sum(weights)
        else:
            result = np.zeros_like(image, dtype=np.float64)
            channel = image.astype(float)
            
            for y in range(h):
                for x in range(w):
                    y_min = max(0, y - radius)
                    y_max = min(h, y + radius + 1)
                    x_min = max(0, x - radius)
                    x_max = min(w, x + radius + 1)
                    
                    region = channel[y_min:y_max, x_min:x_max]
                    
                    sw_y_start = radius - (y - y_min)
                    sw_y_end = radius + (y_max - y)
                    sw_x_start = radius - (x - x_min)
                    sw_x_end = radius + (x_max - x)
                    
                    sw = spatial_weights[sw_y_start:sw_y_end, sw_x_start:sw_x_end]
                    
                    intensity_diff = region - channel[y, x]
                    range_weights = np.exp(-intensity_diff**2 / (2 * sigma_range**2))
                    
                    weights = sw * range_weights
                    result[y, x] = np.sum(region * weights) / np.sum(weights)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def lens_blur(image: np.ndarray,
                 radius: int = 10,
                 blade_count: int = 6,
                 rotation: float = 0) -> np.ndarray:
        """
        Lens/bokeh blur - simulates camera lens blur.
        
        Creates characteristic bokeh shapes (circular or polygonal).
        """
        # Create bokeh kernel shape
        size = 2 * radius + 1
        kernel = np.zeros((size, size))
        center = radius
        
        for y in range(size):
            for x in range(size):
                dx = x - center
                dy = y - center
                
                if blade_count == 0:
                    # Circular bokeh
                    if dx**2 + dy**2 <= radius**2:
                        kernel[y, x] = 1
                else:
                    # Polygonal bokeh
                    r = math.sqrt(dx**2 + dy**2)
                    if r == 0:
                        kernel[y, x] = 1
                        continue
                    
                    angle = math.atan2(dy, dx) + math.radians(rotation)
                    
                    # Check if inside polygon
                    segment_angle = 2 * math.pi / blade_count
                    local_angle = angle % segment_angle
                    
                    # Distance to polygon edge
                    cos_half = math.cos(segment_angle / 2)
                    polygon_r = radius * cos_half / math.cos(local_angle - segment_angle / 2)
                    
                    if r <= polygon_r:
                        kernel[y, x] = 1
        
        # Normalize
        kernel /= kernel.sum()
        
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=np.float64)
            for c in range(image.shape[2]):
                result[:, :, c] = ndimage.convolve(
                    image[:, :, c].astype(float), kernel
                )
            return np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = ndimage.convolve(image.astype(float), kernel)
            return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def tilt_shift(image: np.ndarray,
                  focus_position: float = 0.5,
                  focus_width: float = 0.2,
                  max_blur: float = 10,
                  horizontal: bool = False) -> np.ndarray:
        """
        Tilt-shift blur - miniature/toy camera effect.
        
        Creates a band of focus with blur above and below.
        """
        h, w = image.shape[:2]
        
        # Create blur map
        blur_map = np.zeros((h, w))
        
        if horizontal:
            for x in range(w):
                dist = abs(x / w - focus_position)
                blur_amount = max(0, (dist - focus_width / 2) / (0.5 - focus_width / 2))
                blur_map[:, x] = blur_amount * max_blur
        else:
            for y in range(h):
                dist = abs(y / h - focus_position)
                blur_amount = max(0, (dist - focus_width / 2) / (0.5 - focus_width / 2))
                blur_map[y, :] = blur_amount * max_blur
        
        # Apply variable blur
        return BlurEffects.variable_blur(image, blur_map)
    
    @staticmethod
    def variable_blur(image: np.ndarray,
                     blur_map: np.ndarray) -> np.ndarray:
        """
        Apply spatially-varying blur based on blur map.
        
        blur_map: 2D array of blur radii for each pixel.
        """
        h, w = image.shape[:2]
        
        # Quantize blur levels for efficiency
        unique_blurs = np.unique(blur_map.astype(int))
        unique_blurs = unique_blurs[unique_blurs > 0]
        
        # Pre-compute blurred versions
        blurred_versions = {}
        for blur_level in unique_blurs:
            if blur_level > 0:
                blurred_versions[blur_level] = BlurEffects.gaussian_blur(
                    image, blur_level / 3
                )
        
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=np.float64)
        else:
            result = np.zeros_like(image, dtype=np.float64)
        
        for y in range(h):
            for x in range(w):
                blur_level = int(blur_map[y, x])
                
                if blur_level == 0:
                    result[y, x] = image[y, x]
                else:
                    # Find closest pre-computed blur
                    closest = min(unique_blurs, key=lambda b: abs(b - blur_level))
                    result[y, x] = blurred_versions[closest][y, x]
        
        return np.clip(result, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# SHARPENING EFFECTS
# ═══════════════════════════════════════════════════════════════

class SharpeningEffects:
    """Complete sharpening effect library"""
    
    @staticmethod
    def simple_sharpen(image: np.ndarray, 
                      amount: float = 1.0) -> np.ndarray:
        """
        Simple convolution-based sharpening.
        """
        kernel = np.array([
            [0, -1, 0],
            [-1, 4 + 1/amount, -1],
            [0, -1, 0]
        ]) * amount / (4 * amount + 1)
        
        # Add identity to keep original brightness
        kernel[1, 1] += 1 - amount / (4 * amount + 1)
        
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=np.float64)
            for c in range(image.shape[2]):
                result[:, :, c] = ndimage.convolve(
                    image[:, :, c].astype(float), kernel
                )
            return np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = ndimage.convolve(image.astype(float), kernel)
            return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def unsharp_mask(image: np.ndarray,
                    radius: float = 1.0,
                    amount: float = 1.0,
                    threshold: int = 0) -> np.ndarray:
        """
        Unsharp mask - the standard image sharpening technique.
        
        1. Blur the image
        2. Subtract blur from original (high-frequency detail)
        3. Add scaled detail back to original
        
        threshold: Only sharpen pixels where difference exceeds this
        """
        # Create blurred version
        blurred = BlurEffects.gaussian_blur(image, radius)
        
        # Calculate detail (high-pass)
        detail = image.astype(float) - blurred.astype(float)
        
        # Apply threshold
        if threshold > 0:
            mask = np.abs(detail) > threshold
            detail = detail * mask
        
        # Add scaled detail
        result = image.astype(float) + amount * detail
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def high_pass_sharpen(image: np.ndarray,
                         radius: float = 3.0,
                         amount: float = 1.0,
                         mode: str = 'overlay') -> np.ndarray:
        """
        High-pass sharpening.
        
        Applies high-pass filter then blends with original.
        
        modes: 'overlay', 'soft_light', 'hard_light', 'linear'
        """
        # High-pass = original - low-pass
        blurred = BlurEffects.gaussian_blur(image, radius)
        high_pass = image.astype(float) - blurred.astype(float) + 128
        high_pass = np.clip(high_pass, 0, 255)
        
        # Blend with original
        result = image.astype(float)
        hp = high_pass.astype(float) / 255
        orig = image.astype(float) / 255
        
        if mode == 'overlay':
            # Overlay blend
            mask = orig < 0.5
            blended = np.where(
                mask,
                2 * orig * hp,
                1 - 2 * (1 - orig) * (1 - hp)
            )
        elif mode == 'soft_light':
            # Soft light blend
            blended = (1 - 2*hp) * orig**2 + 2*hp*orig
        elif mode == 'hard_light':
            # Hard light blend
            mask = hp < 0.5
            blended = np.where(
                mask,
                2 * orig * hp,
                1 - 2 * (1 - orig) * (1 - hp)
            )
        else:  # linear
            blended = orig + (hp - 0.5) * amount
        
        # Mix with original based on amount
        result = orig * (1 - amount) + blended * amount
        
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    
    @staticmethod
    def smart_sharpen(image: np.ndarray,
                     amount: float = 1.0,
                     radius: float = 1.0,
                     reduce_noise: float = 0.0,
                     more_accurate: bool = True) -> np.ndarray:
        """
        Smart sharpen - advanced sharpening with noise control.
        
        Attempts to sharpen detail while minimizing noise amplification.
        """
        # Detect edges
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)
        
        # Sobel edge detection
        gx = ndimage.sobel(gray, axis=1)
        gy = ndimage.sobel(gray, axis=0)
        edges = np.sqrt(gx**2 + gy**2)
        
        # Normalize edge map
        edges = edges / edges.max() if edges.max() > 0 else edges
        
        # Create adaptive sharpening map
        # More sharpening on edges, less on flat areas (noise)
        sharpen_map = edges
        
        if reduce_noise > 0:
            # Suppress sharpening in low-contrast areas
            local_std = ndimage.generic_filter(gray, np.std, size=5)
            noise_mask = local_std < (reduce_noise * 30)
            sharpen_map[noise_mask] *= (1 - reduce_noise)
        
        # Apply unsharp mask with adaptive amount
        blurred = BlurEffects.gaussian_blur(image, radius)
        detail = image.astype(float) - blurred.astype(float)
        
        if len(image.shape) == 3:
            sharpen_map = sharpen_map[:, :, np.newaxis]
        
        result = image.astype(float) + amount * detail * sharpen_map
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def clarity(image: np.ndarray, amount: float = 0.5) -> np.ndarray:
        """
        Clarity effect - midtone contrast enhancement.
        
        Enhances local contrast without affecting highlights/shadows.
        Similar to Lightroom's Clarity slider.
        """
        # Apply local contrast enhancement
        # Using a large-radius unsharp mask on luminance
        
        if len(image.shape) == 3:
            # Convert to LAB for luminance-only processing
            lab = np.zeros_like(image, dtype=np.float64)
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    r, g, b = image[y, x] / 255.0
                    l, a, bb = ColorSpaceConversion.rgb_to_lab(r, g, b)
                    lab[y, x] = [l, a, bb]
            
            # Blur luminance
            blurred_l = ndimage.gaussian_filter(lab[:, :, 0], 30)
            
            # Enhance midtone contrast
            detail = lab[:, :, 0] - blurred_l
            
            # Reduce effect on highlights and shadows
            mask = 1 - np.abs(lab[:, :, 0] - 50) / 50
            mask = np.clip(mask, 0, 1)
            
            lab[:, :, 0] += detail * mask * amount * 30
            lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 100)
            
            # Convert back to RGB
            result = np.zeros_like(image)
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    l, a, bb = lab[y, x]
                    r, g, b = ColorSpaceConversion.lab_to_rgb(l, a, bb)
                    result[y, x] = [
                        np.clip(r * 255, 0, 255),
                        np.clip(g * 255, 0, 255),
                        np.clip(b * 255, 0, 255)
                    ]
            
            return result.astype(np.uint8)
        else:
            blurred = ndimage.gaussian_filter(image.astype(float), 30)
            detail = image.astype(float) - blurred
            result = image.astype(float) + detail * amount
            return np.clip(result, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# DEPTH OF FIELD EFFECTS
# ═══════════════════════════════════════════════════════════════

class DepthOfFieldEffects:
    """Simulate camera depth of field"""
    
    @staticmethod
    def depth_blur(image: np.ndarray,
                  depth_map: np.ndarray,
                  focus_depth: float = 0.5,
                  aperture: float = 2.8,
                  focal_length: float = 50) -> np.ndarray:
        """
        Apply depth-based blur using depth map.
        
        depth_map: 2D array with depth values (0=near, 1=far)
        focus_depth: Depth value that's in focus
        aperture: f-stop (lower = more blur)
        """
        # Calculate circle of confusion for each depth
        # CoC = |depth - focus| * aperture_factor
        aperture_factor = 50 / aperture  # Approximate
        
        coc_map = np.abs(depth_map - focus_depth) * aperture_factor
        
        # Convert CoC to blur radius
        max_blur = 20
        blur_map = np.clip(coc_map * max_blur, 0, max_blur)
        
        return BlurEffects.variable_blur(image, blur_map)
    
    @staticmethod
    def portrait_mode(image: np.ndarray,
                     subject_mask: np.ndarray = None,
                     blur_strength: float = 10) -> np.ndarray:
        """
        Portrait mode effect - blur background while keeping subject sharp.
        
        If subject_mask is None, uses simple center-weighted focus.
        """
        h, w = image.shape[:2]
        
        if subject_mask is None:
            # Create simple oval subject mask
            y, x = np.ogrid[:h, :w]
            cy, cx = h / 2, w / 2
            
            # Oval mask
            mask = ((x - cx)**2 / (w/3)**2 + (y - cy)**2 / (h/2.5)**2) < 1
            subject_mask = mask.astype(float)
            
            # Smooth mask edges
            subject_mask = ndimage.gaussian_filter(subject_mask, 20)
        
        # Blur the entire image
        blurred = BlurEffects.lens_blur(image, int(blur_strength), blade_count=6)
        
        # Composite based on mask
        if len(image.shape) == 3:
            mask_3d = subject_mask[:, :, np.newaxis]
            result = image * mask_3d + blurred * (1 - mask_3d)
        else:
            result = image * subject_mask + blurred * (1 - subject_mask)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def focus_stack(images: List[np.ndarray],
                   method: str = 'laplacian') -> np.ndarray:
        """
        Focus stacking - combine multiple images with different focus.
        
        Creates all-in-focus image from stack of differently focused shots.
        
        methods: 'laplacian', 'gradient', 'variance'
        """
        if len(images) == 0:
            return None
        
        h, w = images[0].shape[:2]
        num_images = len(images)
        
        # Calculate focus measure for each image
        focus_measures = []
        
        for img in images:
            if len(img.shape) == 3:
                gray = np.mean(img, axis=2)
            else:
                gray = img.astype(float)
            
            if method == 'laplacian':
                # Laplacian variance (higher = more in focus)
                lap = ndimage.laplace(gray)
                focus = np.abs(lap)
            elif method == 'gradient':
                # Gradient magnitude
                gx = ndimage.sobel(gray, axis=1)
                gy = ndimage.sobel(gray, axis=0)
                focus = np.sqrt(gx**2 + gy**2)
            else:  # variance
                # Local variance
                focus = ndimage.generic_filter(gray, np.var, size=9)
            
            # Smooth focus measure
            focus = ndimage.gaussian_filter(focus, 3)
            focus_measures.append(focus)
        
        # Find which image is sharpest at each pixel
        focus_stack = np.stack(focus_measures)
        best_focus = np.argmax(focus_stack, axis=0)
        
        # Compose result
        if len(images[0].shape) == 3:
            result = np.zeros_like(images[0], dtype=np.float64)
            for c in range(images[0].shape[2]):
                for i in range(num_images):
                    mask = best_focus == i
                    result[:, :, c][mask] = images[i][:, :, c][mask]
        else:
            result = np.zeros_like(images[0], dtype=np.float64)
            for i in range(num_images):
                mask = best_focus == i
                result[mask] = images[i][mask]
        
        return result.astype(np.uint8)
```

---

## Chapter 18: Stylization Effects

### 18.1 Artistic Filter Implementations

```python
"""
STYLIZATION EFFECTS - ARTISTIC FILTERS

Transform images into various artistic styles.
"""

import numpy as np
from scipy import ndimage
import math

class StylizationEffects:
    """Artistic stylization filters"""
    
    @staticmethod
    def cartoon(image: np.ndarray,
               num_colors: int = 8,
               edge_strength: float = 1.0,
               blur_d: int = 7) -> np.ndarray:
        """
        Cartoon/cel-shading effect.
        
        1. Reduce colors (quantize)
        2. Detect and overlay edges
        """
        # Bilateral filter for smoothing while preserving edges
        smoothed = BlurEffects.bilateral_filter(image, blur_d, 50)
        
        # Quantize colors
        posterized = ColorEffects.posterize(smoothed, num_colors)
        
        # Edge detection
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)
        
        # Use adaptive threshold for edges
        blurred = ndimage.gaussian_filter(gray, 3)
        edges = gray - blurred
        
        # Threshold edges
        edge_mask = (np.abs(edges) > 5).astype(float)
        edge_mask = ndimage.maximum_filter(edge_mask, size=2)
        
        # Darken edges
        if len(image.shape) == 3:
            result = posterized.astype(float)
            for c in range(3):
                result[:, :, c] *= (1 - edge_mask * edge_strength)
        else:
            result = posterized.astype(float) * (1 - edge_mask * edge_strength)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def watercolor(image: np.ndarray,
                  smoothing: float = 5,
                  edge_darkening: float = 0.5) -> np.ndarray:
        """
        Watercolor painting effect.
        
        Uses median filter for painterly smoothing and edge darkening.
        """
        h, w = image.shape[:2]
        
        # Multiple median filter passes for watercolor look
        result = image.copy()
        
        for _ in range(3):
            if len(result.shape) == 3:
                for c in range(result.shape[2]):
                    result[:, :, c] = ndimage.median_filter(
                        result[:, :, c], size=int(smoothing)
                    )
            else:
                result = ndimage.median_filter(result, size=int(smoothing))
        
        # Edge detection for darkening
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)
        
        edges = np.abs(ndimage.laplace(gray))
        edges = ndimage.gaussian_filter(edges, 1)
        edges = np.clip(edges / 20, 0, 1) * edge_darkening
        
        # Darken edges
        if len(result.shape) == 3:
            for c in range(3):
                result[:, :, c] = result[:, :, c].astype(float) * (1 - edges)
        else:
            result = result.astype(float) * (1 - edges)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def oil_painting(image: np.ndarray,
                    radius: int = 4,
                    levels: int = 20) -> np.ndarray:
        """
        Oil painting effect.
        
        For each pixel, find most common intensity in neighborhood
        and use its average color.
        """
        h, w = image.shape[:2]
        
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
        
        # Compute intensity
        intensity = np.mean(image, axis=2)
        intensity_quantized = (intensity * levels / 255).astype(int)
        intensity_quantized = np.clip(intensity_quantized, 0, levels - 1)
        
        result = np.zeros_like(image)
        
        for y in range(h):
            for x in range(w):
                y_min = max(0, y - radius)
                y_max = min(h, y + radius + 1)
                x_min = max(0, x - radius)
                x_max = min(w, x + radius + 1)
                
                region = image[y_min:y_max, x_min:x_max]
                int_region = intensity_quantized[y_min:y_max, x_min:x_max]
                
                # Find most common intensity
                hist = np.bincount(int_region.flatten(), minlength=levels)
                most_common = np.argmax(hist)
                
                # Average color of that intensity
                mask = int_region == most_common
                if np.any(mask):
                    result[y, x] = np.mean(region[mask], axis=0)
                else:
                    result[y, x] = image[y, x]
        
        return result.astype(np.uint8)
    
    @staticmethod
    def pointillism(image: np.ndarray,
                   dot_size: int = 5,
                   density: float = 0.8) -> np.ndarray:
        """
        Pointillism effect - image made of colored dots.
        """
        h, w = image.shape[:2]
        
        # Create white background
        if len(image.shape) == 3:
            result = np.full((h, w, image.shape[2]), 255, dtype=np.uint8)
        else:
            result = np.full((h, w), 255, dtype=np.uint8)
        
        # Generate dot positions
        num_dots = int(h * w * density / (dot_size ** 2))
        
        for _ in range(num_dots):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            
            # Get color from image
            color = image[y, x]
            
            # Draw dot
            for dy in range(-dot_size, dot_size + 1):
                for dx in range(-dot_size, dot_size + 1):
                    if dx**2 + dy**2 <= dot_size**2:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            result[ny, nx] = color
        
        return result
    
    @staticmethod
    def halftone(image: np.ndarray,
                dot_size: int = 4,
                angle: float = 45) -> np.ndarray:
        """
        Halftone effect - newspaper print style.
        
        Converts continuous tones to variable-sized dots.
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)
        
        h, w = gray.shape
        result = np.full((h, w), 255, dtype=np.uint8)
        
        # Grid spacing
        spacing = dot_size * 2
        
        # Rotated grid
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))
        
        for gy in range(-spacing, h + spacing, spacing):
            for gx in range(-spacing, w + spacing, spacing):
                # Rotate grid point
                rx = int(gx * cos_a - gy * sin_a)
                ry = int(gx * sin_a + gy * cos_a)
                
                if 0 <= rx < w and 0 <= ry < h:
                    # Sample intensity
                    intensity = 255 - gray[ry, rx]  # Invert for dark = big dots
                    
                    # Calculate dot radius based on intensity
                    radius = int(intensity / 255 * dot_size)
                    
                    # Draw dot
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            if dx**2 + dy**2 <= radius**2:
                                ny, nx = ry + dy, rx + dx
                                if 0 <= ny < h and 0 <= nx < w:
                                    result[ny, nx] = 0
        
        return result
    
    @staticmethod
    def pixel_art(image: np.ndarray,
                 pixel_size: int = 8,
                 num_colors: int = 16) -> np.ndarray:
        """
        Pixel art effect.
        
        Downsamples and reduces colors for retro game look.
        """
        h, w = image.shape[:2]
        
        # Downsample
        new_h = h // pixel_size
        new_w = w // pixel_size
        
        downsampled = ImageResizing.nearest_neighbor(image, new_w, new_h)
        
        # Reduce colors
        posterized = ColorEffects.posterize(downsampled, num_colors)
        
        # Upsample with nearest neighbor
        result = ImageResizing.nearest_neighbor(posterized, w, h)
        
        return result
    
    @staticmethod
    def emboss(image: np.ndarray,
              angle: float = 135,
              strength: float = 1.0,
              colorize: bool = False) -> np.ndarray:
        """
        Emboss effect - raised relief appearance.
        """
        # Create emboss kernel
        theta = math.radians(angle)
        dx = math.cos(theta)
        dy = -math.sin(theta)
        
        kernel = np.array([
            [-dy - dx, -dy, -dy + dx],
            [-dx, 0, dx],
            [dy - dx, dy, dy + dx]
        ]) * strength
        
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)
        
        embossed = ndimage.convolve(gray, kernel)
        embossed = embossed + 128  # Center at mid-gray
        embossed = np.clip(embossed, 0, 255)
        
        if colorize and len(image.shape) == 3:
            # Blend with original colors
            result = np.zeros_like(image, dtype=np.float64)
            for c in range(3):
                result[:, :, c] = image[:, :, c] * (embossed / 255)
            return np.clip(result, 0, 255).astype(np.uint8)
        else:
            return embossed.astype(np.uint8)
    
    @staticmethod
    def glitch(image: np.ndarray,
              intensity: float = 0.1,
              seed: int = None) -> np.ndarray:
        """
        Digital glitch effect.
        
        Creates RGB channel separation, scan lines, and block displacement.
        """
        if seed is not None:
            np.random.seed(seed)
        
        h, w = image.shape[:2]
        result = image.copy()
        
        if len(image.shape) == 3:
            # Channel shift
            shift_amount = int(w * intensity * 0.05)
            
            result[:, :, 0] = np.roll(result[:, :, 0], shift_amount, axis=1)
            result[:, :, 2] = np.roll(result[:, :, 2], -shift_amount, axis=1)
            
            # Random block displacement
            num_blocks = int(10 * intensity)
            for _ in range(num_blocks):
                block_h = np.random.randint(5, 30)
                block_w = np.random.randint(50, w // 2)
                
                y = np.random.randint(0, h - block_h)
                x = np.random.randint(0, w - block_w)
                
                offset = np.random.randint(-block_w // 2, block_w // 2)
                
                block = result[y:y+block_h, x:x+block_w].copy()
                
                new_x = max(0, min(w - block_w, x + offset))
                result[y:y+block_h, new_x:new_x+block_w] = block
            
            # Scan lines
            for y in range(0, h, 4):
                result[y, :] = (result[y, :].astype(float) * 0.7).astype(np.uint8)
        
        return result
    
    @staticmethod
    def vhs(image: np.ndarray,
           tracking_error: float = 0.1,
           color_bleed: float = 0.3) -> np.ndarray:
        """
        VHS tape effect.
        
        Simulates analog video artifacts.
        """
        h, w = image.shape[:2]
        result = image.copy().astype(float)
        
        if len(image.shape) == 3:
            # Horizontal blur (color bleed)
            blur_kernel = np.ones((1, int(5 + color_bleed * 10))) / (5 + color_bleed * 10)
            
            for c in range(3):
                result[:, :, c] = ndimage.convolve(result[:, :, c], blur_kernel)
            
            # Slight channel misalignment
            result[:, :, 0] = np.roll(result[:, :, 0], 1, axis=1)
            result[:, :, 2] = np.roll(result[:, :, 2], -1, axis=1)
            
            # Tracking errors (horizontal displacement)
            num_errors = int(5 * tracking_error)
            for _ in range(num_errors):
                y = np.random.randint(0, h)
                error_height = np.random.randint(1, 10)
                offset = np.random.randint(-20, 20)
                
                y_end = min(h, y + error_height)
                result[y:y_end] = np.roll(result[y:y_end], offset, axis=1)
            
            # Reduce saturation
            for y in range(h):
                for x in range(w):
                    r, g, b = result[y, x] / 255
                    h_val, s, v = ColorSpaceConversion.rgb_to_hsv(r, g, b)
                    s *= 0.8
                    r, g, b = ColorSpaceConversion.hsv_to_rgb(h_val, s, v)
                    result[y, x] = [r * 255, g * 255, b * 255]
            
            # Add scan lines
            for y in range(0, h, 2):
                result[y] = result[y] * 0.9
            
            # Add noise
            noise = np.random.normal(0, 5, result.shape)
            result += noise
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def neon_glow(image: np.ndarray,
                 glow_color: Tuple[int, int, int] = (0, 255, 255),
                 threshold: int = 200,
                 glow_radius: float = 10) -> np.ndarray:
        """
        Neon glow effect on bright areas.
        """
        if len(image.shape) == 3:
            brightness = np.max(image, axis=2)
        else:
            brightness = image
        
        # Create glow mask
        mask = (brightness > threshold).astype(float)
        
        # Blur mask for glow
        glow = ndimage.gaussian_filter(mask, glow_radius)
        
        # Apply glow color
        h, w = image.shape[:2]
        
        if len(image.shape) == 3:
            result = image.astype(float)
            for c in range(3):
                result[:, :, c] += glow * glow_color[c]
        else:
            result = image.astype(float) + glow * np.mean(glow_color)
        
        return np.clip(result, 0, 255).astype(np.uint8)
```

---

# PART VI: PROCEDURAL GENERATION

---

## Chapter 19: Pattern Generation

### 19.1 Geometric Patterns

```python
"""
PATTERN GENERATION - COMPREHENSIVE GUIDE

Creating repeatable, tileable patterns for procedural art.
"""

import numpy as np
import math
from typing import Tuple, List, Callable, Optional

# ═══════════════════════════════════════════════════════════════
# GEOMETRIC PATTERNS
# ═══════════════════════════════════════════════════════════════

class GeometricPatterns:
    """Generate geometric and tiled patterns"""
    
    @staticmethod
    def checkerboard(width: int, height: int,
                    square_size: int = 32,
                    color1: Tuple[int, int, int] = (255, 255, 255),
                    color2: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        Classic checkerboard pattern.
        """
        result = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                if ((x // square_size) + (y // square_size)) % 2 == 0:
                    result[y, x] = color1
                else:
                    result[y, x] = color2
        
        return result
    
    @staticmethod
    def stripes(width: int, height: int,
               stripe_width: int = 20,
               angle: float = 0,
               colors: List[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Striped pattern at any angle.
        """
        if colors is None:
            colors = [(255, 255, 255), (0, 0, 0)]
        
        result = np.zeros((height, width, 3), dtype=np.uint8)
        
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))
        
        period = stripe_width * len(colors)
        
        for y in range(height):
            for x in range(width):
                # Project onto stripe direction
                pos = x * cos_a + y * sin_a
                
                # Determine which stripe
                stripe_index = int(pos / stripe_width) % len(colors)
                result[y, x] = colors[stripe_index]
        
        return result
    
    @staticmethod
    def polka_dots(width: int, height: int,
                  dot_radius: int = 10,
                  spacing: int = 30,
                  dot_color: Tuple[int, int, int] = (255, 0, 0),
                  bg_color: Tuple[int, int, int] = (255, 255, 255),
                  offset_rows: bool = True) -> np.ndarray:
        """
        Polka dot pattern with optional row offset.
        """
        result = np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        for gy in range(0, height + spacing, spacing):
            offset = (spacing // 2) if (offset_rows and (gy // spacing) % 2 == 1) else 0
            
            for gx in range(offset, width + spacing, spacing):
                # Draw dot
                for dy in range(-dot_radius, dot_radius + 1):
                    for dx in range(-dot_radius, dot_radius + 1):
                        if dx**2 + dy**2 <= dot_radius**2:
                            y, x = gy + dy, gx + dx
                            if 0 <= y < height and 0 <= x < width:
                                result[y, x] = dot_color
        
        return result
    
    @staticmethod
    def hexagonal_grid(width: int, height: int,
                      hex_size: int = 30,
                      line_color: Tuple[int, int, int] = (0, 0, 0),
                      fill_colors: List[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Hexagonal grid pattern.
        """
        if fill_colors is None:
            fill_colors = [(255, 255, 255)]
        
        result = np.full((height, width, 3), fill_colors[0], dtype=np.uint8)
        
        # Hexagon dimensions
        hex_h = hex_size * 2
        hex_w = int(hex_size * math.sqrt(3))
        
        def point_in_hexagon(px, py, cx, cy, size):
            """Check if point is inside hexagon centered at (cx, cy)."""
            dx = abs(px - cx)
            dy = abs(py - cy)
            
            if dx > hex_w / 2 or dy > hex_h / 2:
                return False
            
            return dy <= hex_h / 2 - (dx / (hex_w / 2)) * (hex_size / 2)
        
        # Generate hexagon centers
        color_index = 0
        for row in range(height // int(hex_h * 0.75) + 2):
            for col in range(width // hex_w + 2):
                cx = col * hex_w + (hex_w / 2 if row % 2 == 1 else 0)
                cy = row * (hex_h * 0.75)
                
                # Fill hexagon
                fill_color = fill_colors[color_index % len(fill_colors)]
                color_index += 1
                
                for y in range(max(0, int(cy - hex_h/2)), min(height, int(cy + hex_h/2))):
                    for x in range(max(0, int(cx - hex_w/2)), min(width, int(cx + hex_w/2))):
                        if point_in_hexagon(x, y, cx, cy, hex_size):
                            result[y, x] = fill_color
        
        return result
    
    @staticmethod
    def circles_grid(width: int, height: int,
                    circle_radius: int = 20,
                    spacing: int = 50,
                    line_width: int = 2,
                    line_color: Tuple[int, int, int] = (0, 0, 0),
                    bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        Grid of circle outlines.
        """
        result = np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        for cy in range(spacing // 2, height, spacing):
            for cx in range(spacing // 2, width, spacing):
                # Draw circle outline
                for y in range(max(0, cy - circle_radius - line_width),
                              min(height, cy + circle_radius + line_width)):
                    for x in range(max(0, cx - circle_radius - line_width),
                                  min(width, cx + circle_radius + line_width)):
                        dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                        
                        if abs(dist - circle_radius) < line_width / 2:
                            result[y, x] = line_color
        
        return result
    
    @staticmethod
    def diamond_pattern(width: int, height: int,
                       diamond_size: int = 40,
                       colors: List[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Diamond/argyle pattern.
        """
        if colors is None:
            colors = [(200, 200, 200), (100, 100, 100)]
        
        result = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                # Transform to diamond coordinates
                dx = (x + y) % (diamond_size * 2)
                dy = (x - y + height) % (diamond_size * 2)
                
                color_idx = 0
                if dx >= diamond_size:
                    color_idx ^= 1
                if dy >= diamond_size:
                    color_idx ^= 1
                
                result[y, x] = colors[color_idx]
        
        return result
    
    @staticmethod
    def triangle_pattern(width: int, height: int,
                        triangle_size: int = 50,
                        colors: List[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Tessellating triangle pattern.
        """
        if colors is None:
            colors = [(255, 200, 200), (200, 200, 255)]
        
        result = np.zeros((height, width, 3), dtype=np.uint8)
        
        tri_h = int(triangle_size * math.sqrt(3) / 2)
        
        for y in range(height):
            for x in range(width):
                row = y // tri_h
                
                # Position within row
                local_y = y % tri_h
                local_x = x % triangle_size
                
                # Offset every other row
                if row % 2 == 1:
                    local_x = (local_x + triangle_size // 2) % triangle_size
                
                # Determine if point is in upper or lower triangle
                # Using barycentric-like check
                in_upper = local_y < local_x * tri_h / (triangle_size / 2)
                if local_x > triangle_size / 2:
                    in_upper = local_y < (triangle_size - local_x) * tri_h / (triangle_size / 2)
                
                color_idx = (row + (1 if in_upper else 0)) % len(colors)
                result[y, x] = colors[color_idx]
        
        return result
    
    @staticmethod
    def spirograph(width: int, height: int,
                  R: float = 100,  # Fixed circle radius
                  r: float = 40,   # Moving circle radius
                  d: float = 80,   # Distance from moving circle center
                  line_color: Tuple[int, int, int] = (0, 0, 255),
                  bg_color: Tuple[int, int, int] = (255, 255, 255),
                  line_width: int = 2) -> np.ndarray:
        """
        Spirograph (hypotrochoid/epitrochoid) pattern.
        """
        from PIL import Image, ImageDraw
        
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        cx, cy = width / 2, height / 2
        
        # Calculate number of rotations needed
        from math import gcd
        num_rotations = int(r / gcd(int(R), int(r)))
        
        points = []
        for t in np.linspace(0, num_rotations * 2 * math.pi, 10000):
            # Hypotrochoid equations
            x = (R - r) * math.cos(t) + d * math.cos((R - r) / r * t)
            y = (R - r) * math.sin(t) - d * math.sin((R - r) / r * t)
            
            points.append((cx + x, cy + y))
        
        # Draw lines
        for i in range(len(points) - 1):
            draw.line([points[i], points[i+1]], fill=line_color, width=line_width)
        
        return np.array(img)
    
    @staticmethod
    def islamic_pattern(width: int, height: int,
                       tile_size: int = 100,
                       colors: List[Tuple[int, int, int]] = None,
                       pattern_type: str = 'star') -> np.ndarray:
        """
        Islamic geometric pattern.
        
        Types: 'star', 'octagon', 'interlace'
        """
        if colors is None:
            colors = [(255, 255, 255), (100, 150, 200), (50, 80, 120)]
        
        from PIL import Image, ImageDraw
        
        img = Image.new('RGB', (width, height), colors[0])
        draw = ImageDraw.Draw(img)
        
        if pattern_type == 'star':
            # 8-pointed star pattern
            for ty in range(-1, height // tile_size + 2):
                for tx in range(-1, width // tile_size + 2):
                    cx = tx * tile_size + tile_size // 2
                    cy = ty * tile_size + tile_size // 2
                    
                    # Draw 8-pointed star
                    points = []
                    for i in range(16):
                        angle = i * math.pi / 8
                        r = tile_size * 0.4 if i % 2 == 0 else tile_size * 0.2
                        x = cx + r * math.cos(angle)
                        y = cy + r * math.sin(angle)
                        points.append((x, y))
                    
                    draw.polygon(points, fill=colors[1], outline=colors[2])
        
        elif pattern_type == 'octagon':
            # Octagon and square pattern
            oct_size = tile_size * 0.7
            
            for ty in range(-1, height // tile_size + 2):
                for tx in range(-1, width // tile_size + 2):
                    cx = tx * tile_size + tile_size // 2
                    cy = ty * tile_size + tile_size // 2
                    
                    # Draw octagon
                    points = []
                    for i in range(8):
                        angle = i * math.pi / 4 + math.pi / 8
                        x = cx + oct_size / 2 * math.cos(angle)
                        y = cy + oct_size / 2 * math.sin(angle)
                        points.append((x, y))
                    
                    draw.polygon(points, fill=colors[1], outline=colors[2])
        
        return np.array(img)


# ═══════════════════════════════════════════════════════════════
# ORGANIC PATTERNS
# ═══════════════════════════════════════════════════════════════

class OrganicPatterns:
    """Generate organic and natural patterns"""
    
    @staticmethod
    def voronoi_pattern(width: int, height: int,
                       num_points: int = 50,
                       colors: List[Tuple[int, int, int]] = None,
                       draw_edges: bool = True,
                       edge_color: Tuple[int, int, int] = (0, 0, 0),
                       seed: int = None) -> np.ndarray:
        """
        Voronoi diagram pattern.
        """
        if seed is not None:
            np.random.seed(seed)
        
        if colors is None:
            colors = [
                (255, 200, 200), (200, 255, 200), (200, 200, 255),
                (255, 255, 200), (255, 200, 255), (200, 255, 255)
            ]
        
        # Generate random points
        points = np.random.rand(num_points, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        
        result = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Assign each pixel to nearest point
        closest = np.zeros((height, width), dtype=int)
        
        for y in range(height):
            for x in range(width):
                min_dist = float('inf')
                min_idx = 0
                
                for i, (px, py) in enumerate(points):
                    dist = (x - px)**2 + (y - py)**2
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = i
                
                closest[y, x] = min_idx
                result[y, x] = colors[min_idx % len(colors)]
        
        if draw_edges:
            # Find edges (where nearest point changes)
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    current = closest[y, x]
                    
                    # Check neighbors
                    if (closest[y-1, x] != current or
                        closest[y+1, x] != current or
                        closest[y, x-1] != current or
                        closest[y, x+1] != current):
                        result[y, x] = edge_color
        
        return result
    
    @staticmethod
    def reaction_diffusion(width: int, height: int,
                          iterations: int = 5000,
                          feed_rate: float = 0.055,
                          kill_rate: float = 0.062,
                          diffusion_a: float = 1.0,
                          diffusion_b: float = 0.5) -> np.ndarray:
        """
        Gray-Scott reaction-diffusion pattern.
        
        Creates organic, coral-like patterns through chemical simulation.
        """
        # Initialize grids
        a = np.ones((height, width))
        b = np.zeros((height, width))
        
        # Seed with some B chemical
        center_h, center_w = height // 2, width // 2
        seed_size = min(width, height) // 10
        
        b[center_h - seed_size:center_h + seed_size,
          center_w - seed_size:center_w + seed_size] = 1.0
        
        # Add some random seeds
        for _ in range(10):
            rx = np.random.randint(0, width - 20)
            ry = np.random.randint(0, height - 20)
            b[ry:ry+20, rx:rx+20] = 1.0
        
        # Laplacian kernel
        laplacian = np.array([
            [0.05, 0.2, 0.05],
            [0.2, -1, 0.2],
            [0.05, 0.2, 0.05]
        ])
        
        dt = 1.0
        
        for _ in range(iterations):
            # Calculate Laplacians
            lap_a = ndimage.convolve(a, laplacian, mode='wrap')
            lap_b = ndimage.convolve(b, laplacian, mode='wrap')
            
            # Reaction
            reaction = a * b * b
            
            # Update
            a += (diffusion_a * lap_a - reaction + feed_rate * (1 - a)) * dt
            b += (diffusion_b * lap_b + reaction - (kill_rate + feed_rate) * b) * dt
            
            # Clamp
            a = np.clip(a, 0, 1)
            b = np.clip(b, 0, 1)
        
        # Convert to grayscale image
        result = ((1 - b) * 255).astype(np.uint8)
        
        return result
    
    @staticmethod
    def lichen_growth(width: int, height: int,
                     iterations: int = 1000,
                     num_seeds: int = 5,
                     branch_prob: float = 0.3,
                     seed: int = None) -> np.ndarray:
        """
        Lichen/coral growth pattern using DLA (Diffusion Limited Aggregation).
        """
        if seed is not None:
            np.random.seed(seed)
        
        result = np.zeros((height, width), dtype=np.uint8)
        
        # Place seed points
        for _ in range(num_seeds):
            x = np.random.randint(width // 4, 3 * width // 4)
            y = np.random.randint(height // 4, 3 * height // 4)
            result[y, x] = 255
        
        for _ in range(iterations):
            # Release a random walker from edge
            side = np.random.randint(4)
            if side == 0:  # Top
                x, y = np.random.randint(width), 0
            elif side == 1:  # Bottom
                x, y = np.random.randint(width), height - 1
            elif side == 2:  # Left
                x, y = 0, np.random.randint(height)
            else:  # Right
                x, y = width - 1, np.random.randint(height)
            
            # Random walk until hitting structure or leaving
            max_steps = width * height
            for _ in range(max_steps):
                # Check if adjacent to structure
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if result[ny, nx] > 0:
                            # Stick to structure
                            result[y, x] = 255
                            
                            # Maybe branch
                            if np.random.random() < branch_prob:
                                for _ in range(np.random.randint(1, 4)):
                                    bx = x + np.random.randint(-1, 2)
                                    by = y + np.random.randint(-1, 2)
                                    if 0 <= bx < width and 0 <= by < height:
                                        result[by, bx] = 255
                            break
                else:
                    # Random step
                    dx = np.random.randint(-1, 2)
                    dy = np.random.randint(-1, 2)
                    x = (x + dx) % width
                    y = (y + dy) % height
                    continue
                break
        
        return result
    
    @staticmethod
    def cracked_earth(width: int, height: int,
                     num_cracks: int = 20,
                     crack_width: int = 2,
                     colors: List[Tuple[int, int, int]] = None,
                     seed: int = None) -> np.ndarray:
        """
        Cracked earth/mud pattern.
        """
        if seed is not None:
            np.random.seed(seed)
        
        if colors is None:
            colors = [(180, 140, 100), (120, 90, 60)]  # Earth tones
        
        # Start with Voronoi for cell structure
        result = OrganicPatterns.voronoi_pattern(
            width, height, num_cracks,
            colors=[colors[0]], draw_edges=False, seed=seed
        )
        
        # Add cracks along Voronoi edges
        # Generate points again
        np.random.seed(seed)
        points = np.random.rand(num_cracks, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        
        # For each pixel, check if it's near an edge
        for y in range(height):
            for x in range(width):
                # Find two nearest points
                dists = []
                for i, (px, py) in enumerate(points):
                    dist = math.sqrt((x - px)**2 + (y - py)**2)
                    dists.append((dist, i))
                
                dists.sort()
                
                # If nearly equidistant from two points, it's on an edge
                if len(dists) >= 2:
                    diff = abs(dists[0][0] - dists[1][0])
                    if diff < crack_width:
                        result[y, x] = colors[1]
        
        # Add texture variation
        noise = PerlinNoise(seed or 42)
        for y in range(height):
            for x in range(width):
                n = noise.noise_2d(x * 0.05, y * 0.05)
                factor = 0.8 + 0.2 * n
                result[y, x] = np.clip(result[y, x] * factor, 0, 255).astype(np.uint8)
        
        return result
    
    @staticmethod
    def scales_pattern(width: int, height: int,
                      scale_size: int = 30,
                      colors: List[Tuple[int, int, int]] = None,
                      variation: float = 0.2) -> np.ndarray:
        """
        Fish/reptile scales pattern.
        """
        if colors is None:
            colors = [(100, 150, 100), (80, 120, 80), (60, 90, 60)]
        
        result = np.zeros((height, width, 3), dtype=np.uint8)
        
        row_height = int(scale_size * 0.7)
        
        for row in range(height // row_height + 2):
            offset = (scale_size // 2) if row % 2 == 1 else 0
            
            for col in range(-1, width // scale_size + 2):
                cx = col * scale_size + offset
                cy = row * row_height
                
                # Color variation
                color_idx = (row + col) % len(colors)
                color = np.array(colors[color_idx])
                
                # Add random variation
                if variation > 0:
                    var = np.random.uniform(1 - variation, 1 + variation, 3)
                    color = np.clip(color * var, 0, 255)
                
                # Draw scale (arc shape)
                for y in range(max(0, cy - scale_size), min(height, cy + scale_size)):
                    for x in range(max(0, cx - scale_size // 2), 
                                  min(width, cx + scale_size // 2)):
                        # Scale shape
                        dx = (x - cx) / (scale_size // 2)
                        dy = (y - cy) / scale_size
                        
                        # Overlapping arc
                        if dy >= 0 and dx**2 + dy**2 <= 1:
                            # Shading based on position
                            shade = 1 - dy * 0.3 - abs(dx) * 0.2
                            result[y, x] = np.clip(color * shade, 0, 255).astype(np.uint8)
        
        return result


# ═══════════════════════════════════════════════════════════════
# PROCEDURAL TEXTURE SYNTHESIS
# ═══════════════════════════════════════════════════════════════

class TextureSynthesis:
    """Advanced texture synthesis algorithms"""
    
    @staticmethod
    def quilting(source: np.ndarray,
                output_size: Tuple[int, int],
                patch_size: int = 32,
                overlap: int = 6,
                seed: int = None) -> np.ndarray:
        """
        Image quilting for texture synthesis.
        
        Creates new texture by stitching together patches from source.
        """
        if seed is not None:
            np.random.seed(seed)
        
        src_h, src_w = source.shape[:2]
        out_h, out_w = output_size
        
        if len(source.shape) == 3:
            result = np.zeros((out_h, out_w, source.shape[2]), dtype=np.uint8)
        else:
            result = np.zeros((out_h, out_w), dtype=np.uint8)
        
        step = patch_size - overlap
        
        for row in range(0, out_h, step):
            for col in range(0, out_w, step):
                # Find best matching patch
                best_patch = None
                best_error = float('inf')
                
                # Sample random patches
                num_candidates = 50
                for _ in range(num_candidates):
                    px = np.random.randint(0, src_w - patch_size)
                    py = np.random.randint(0, src_h - patch_size)
                    
                    patch = source[py:py+patch_size, px:px+patch_size]
                    
                    # Calculate error in overlap region
                    error = 0
                    
                    # Left overlap
                    if col > 0:
                        left_overlap_result = result[row:row+patch_size, col:col+overlap]
                        left_overlap_patch = patch[:, :overlap]
                        
                        h_overlap = min(patch_size, out_h - row)
                        error += np.sum((
                            left_overlap_result[:h_overlap].astype(float) - 
                            left_overlap_patch[:h_overlap].astype(float)
                        )**2)
                    
                    # Top overlap
                    if row > 0:
                        top_overlap_result = result[row:row+overlap, col:col+patch_size]
                        top_overlap_patch = patch[:overlap, :]
                        
                        w_overlap = min(patch_size, out_w - col)
                        error += np.sum((
                            top_overlap_result[:, :w_overlap].astype(float) - 
                            top_overlap_patch[:, :w_overlap].astype(float)
                        )**2)
                    
                    if error < best_error:
                        best_error = error
                        best_patch = patch.copy()
                
                if best_patch is None:
                    continue
                
                # Place patch (with minimum error boundary cut)
                h_to_place = min(patch_size, out_h - row)
                w_to_place = min(patch_size, out_w - col)
                
                result[row:row+h_to_place, col:col+w_to_place] = \
                    best_patch[:h_to_place, :w_to_place]
        
        return result
    
    @staticmethod
    def style_transfer_simple(content: np.ndarray,
                             style: np.ndarray,
                             style_weight: float = 0.5) -> np.ndarray:
        """
        Simple histogram-based style transfer.
        
        Matches color statistics of style to content.
        """
        # Convert to LAB color space
        content_lab = np.zeros_like(content, dtype=np.float64)
        style_lab = np.zeros_like(style, dtype=np.float64)
        
        for y in range(content.shape[0]):
            for x in range(content.shape[1]):
                r, g, b = content[y, x] / 255.0
                l, a, bb = ColorSpaceConversion.rgb_to_lab(r, g, b)
                content_lab[y, x] = [l, a, bb]
        
        for y in range(style.shape[0]):
            for x in range(style.shape[1]):
                r, g, b = style[y, x] / 255.0
                l, a, bb = ColorSpaceConversion.rgb_to_lab(r, g, b)
                style_lab[y, x] = [l, a, bb]
        
        # Calculate statistics
        for c in range(3):
            content_mean = np.mean(content_lab[:, :, c])
            content_std = np.std(content_lab[:, :, c])
            
            style_mean = np.mean(style_lab[:, :, c])
            style_std = np.std(style_lab[:, :, c])
            
            # Transfer statistics
            if content_std > 0:
                content_lab[:, :, c] = (
                    (content_lab[:, :, c] - content_mean) * 
                    (style_std / content_std) * style_weight +
                    content_mean * (1 - style_weight) +
                    style_mean * style_weight
                )
        
        # Convert back to RGB
        result = np.zeros_like(content)
        
        for y in range(content.shape[0]):
            for x in range(content.shape[1]):
                l, a, bb = content_lab[y, x]
                r, g, b = ColorSpaceConversion.lab_to_rgb(l, a, bb)
                result[y, x] = [
                    np.clip(r * 255, 0, 255),
                    np.clip(g * 255, 0, 255),
                    np.clip(b * 255, 0, 255)
                ]
        
        return result.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# PARAMETRIC SHAPES
# ═══════════════════════════════════════════════════════════════

class ParametricShapes:
    """Generate shapes using parametric equations"""
    
    @staticmethod
    def superellipse(width: int, height: int,
                    a: float = None,
                    b: float = None,
                    n: float = 2.5,
                    fill_color: Tuple[int, int, int] = (255, 255, 255),
                    bg_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        Superellipse (Lamé curve).
        
        |x/a|^n + |y/b|^n = 1
        
        n < 2: Star-like
        n = 2: Ellipse
        n > 2: Squircle
        """
        if a is None:
            a = width * 0.4
        if b is None:
            b = height * 0.4
        
        result = np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        cx, cy = width / 2, height / 2
        
        for y in range(height):
            for x in range(width):
                dx = abs(x - cx) / a
                dy = abs(y - cy) / b
                
                if dx**n + dy**n <= 1:
                    result[y, x] = fill_color
        
        return result
    
    @staticmethod
    def rose_curve(width: int, height: int,
                  k: float = 5,
                  radius: float = None,
                  line_color: Tuple[int, int, int] = (255, 0, 100),
                  bg_color: Tuple[int, int, int] = (0, 0, 0),
                  line_width: int = 2) -> np.ndarray:
        """
        Rose curve (rhodonea curve).
        
        r = cos(k * theta)
        
        k integer: k or 2k petals depending on odd/even
        """
        if radius is None:
            radius = min(width, height) * 0.4
        
        from PIL import Image, ImageDraw
        
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        cx, cy = width / 2, height / 2
        
        points = []
        for theta in np.linspace(0, 2 * math.pi * (2 if k % 1 == 0 else 1), 1000):
            r = radius * math.cos(k * theta)
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            points.append((x, y))
        
        for i in range(len(points) - 1):
            draw.line([points[i], points[i+1]], fill=line_color, width=line_width)
        
        return np.array(img)
    
    @staticmethod
    def lissajous(width: int, height: int,
                 a: int = 3,
                 b: int = 2,
                 delta: float = math.pi / 2,
                 radius_x: float = None,
                 radius_y: float = None,
                 line_color: Tuple[int, int, int] = (0, 255, 255),
                 bg_color: Tuple[int, int, int] = (0, 0, 0),
                 line_width: int = 2) -> np.ndarray:
        """
        Lissajous curve.
        
        x = A * sin(a*t + delta)
        y = B * sin(b*t)
        """
        if radius_x is None:
            radius_x = width * 0.4
        if radius_y is None:
            radius_y = height * 0.4
        
        from PIL import Image, ImageDraw
        
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        cx, cy = width / 2, height / 2
        
        points = []
        for t in np.linspace(0, 2 * math.pi, 1000):
            x = cx + radius_x * math.sin(a * t + delta)
            y = cy + radius_y * math.sin(b * t)
            points.append((x, y))
        
        for i in range(len(points) - 1):
            draw.line([points[i], points[i+1]], fill=line_color, width=line_width)
        
        return np.array(img)
    
    @staticmethod
    def butterfly_curve(width: int, height: int,
                       scale: float = None,
                       line_color: Tuple[int, int, int] = (255, 100, 0),
                       bg_color: Tuple[int, int, int] = (0, 0, 0),
                       line_width: int = 2) -> np.ndarray:
        """
        Butterfly curve (transcendental curve).
        """
        if scale is None:
            scale = min(width, height) * 0.08
        
        from PIL import Image, ImageDraw
        
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        cx, cy = width / 2, height / 2
        
        points = []
        for t in np.linspace(0, 12 * math.pi, 5000):
            r = math.exp(math.sin(t)) - 2 * math.cos(4 * t) + math.sin((2*t - math.pi) / 24)**5
            x = cx + r * math.sin(t) * scale
            y = cy - r * math.cos(t) * scale
            points.append((x, y))
        
        for i in range(len(points) - 1):
            draw.line([points[i], points[i+1]], fill=line_color, width=line_width)
        
        return np.array(img)
```

---

# PART VII: ADVANCED TECHNIQUES

---

## Chapter 20: Compositing and Blending

### 20.1 Blend Modes

```python
"""
COMPOSITING AND BLENDING - COMPREHENSIVE GUIDE

Layer blending modes, alpha compositing, and advanced compositing techniques.
"""

import numpy as np
import math
from typing import Tuple, Optional, Callable

# ═══════════════════════════════════════════════════════════════
# BLEND MODES
# ═══════════════════════════════════════════════════════════════

class BlendModes:
    """
    Complete implementation of Photoshop-style blend modes.
    
    All functions take normalized values (0-1) for base and blend layers.
    """
    
    # ─────────────────────────────────────────────────────────
    # NORMAL MODES
    # ─────────────────────────────────────────────────────────
    
    @staticmethod
    def normal(base: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        """Normal blend - simple opacity mixing."""
        return base * (1 - opacity) + blend * opacity
    
    @staticmethod
    def dissolve(base: np.ndarray, blend: np.ndarray, opacity: float = 1.0,
                seed: int = None) -> np.ndarray:
        """Dissolve - random pixels from base or blend based on opacity."""
        if seed is not None:
            np.random.seed(seed)
        
        mask = np.random.random(base.shape[:2]) < opacity
        if len(base.shape) == 3:
            mask = mask[:, :, np.newaxis]
        
        return np.where(mask, blend, base)
    
    # ─────────────────────────────────────────────────────────
    # DARKEN MODES
    # ─────────────────────────────────────────────────────────
    
    @staticmethod
    def darken(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Darken - keeps darker of two pixels."""
        return np.minimum(base, blend)
    
    @staticmethod
    def multiply(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """
        Multiply - darkens image.
        
        Result is always darker than input (except with white).
        Used for: shadows, color mixing, darkening.
        """
        return base * blend
    
    @staticmethod
    def color_burn(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """
        Color Burn - darkens base to reflect blend.
        
        Increases contrast. Black blend gives black.
        """
        result = np.where(
            blend == 0,
            0,
            1 - np.minimum(1, (1 - base) / (blend + 1e-10))
        )
        return np.clip(result, 0, 1)
    
    @staticmethod
    def linear_burn(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Linear Burn - darker than multiply."""
        return np.clip(base + blend - 1, 0, 1)
    
    @staticmethod
    def darker_color(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Darker Color - compares total RGB, keeps darker."""
        if len(base.shape) == 3:
            base_luminosity = np.sum(base, axis=2, keepdims=True)
            blend_luminosity = np.sum(blend, axis=2, keepdims=True)
            return np.where(base_luminosity < blend_luminosity, base, blend)
        else:
            return np.minimum(base, blend)
    
    # ─────────────────────────────────────────────────────────
    # LIGHTEN MODES
    # ─────────────────────────────────────────────────────────
    
    @staticmethod
    def lighten(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Lighten - keeps lighter of two pixels."""
        return np.maximum(base, blend)
    
    @staticmethod
    def screen(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """
        Screen - lightens image.
        
        Opposite of multiply. Result is always lighter (except with black).
        Used for: highlights, glow effects, lightening.
        """
        return 1 - (1 - base) * (1 - blend)
    
    @staticmethod
    def color_dodge(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """
        Color Dodge - brightens base to reflect blend.
        
        Decreases contrast. White blend gives white.
        """
        result = np.where(
            blend >= 1,
            1,
            np.minimum(1, base / (1 - blend + 1e-10))
        )
        return np.clip(result, 0, 1)
    
    @staticmethod
    def linear_dodge(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Linear Dodge (Add) - brighter than screen."""
        return np.clip(base + blend, 0, 1)
    
    @staticmethod
    def lighter_color(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Lighter Color - compares total RGB, keeps lighter."""
        if len(base.shape) == 3:
            base_luminosity = np.sum(base, axis=2, keepdims=True)
            blend_luminosity = np.sum(blend, axis=2, keepdims=True)
            return np.where(base_luminosity > blend_luminosity, base, blend)
        else:
            return np.maximum(base, blend)
    
    # ─────────────────────────────────────────────────────────
    # CONTRAST MODES
    # ─────────────────────────────────────────────────────────
    
    @staticmethod
    def overlay(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """
        Overlay - combines multiply and screen.
        
        Darkens darks, lightens lights. 50% gray is neutral.
        Used for: contrast, texture overlays, color grading.
        """
        return np.where(
            base < 0.5,
            2 * base * blend,
            1 - 2 * (1 - base) * (1 - blend)
        )
    
    @staticmethod
    def soft_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """
        Soft Light - softer version of overlay.
        
        More subtle contrast adjustment.
        """
        return np.where(
            blend < 0.5,
            base - (1 - 2 * blend) * base * (1 - base),
            base + (2 * blend - 1) * (
                np.where(base < 0.25,
                        ((16 * base - 12) * base + 4) * base,
                        np.sqrt(base)) - base
            )
        )
    
    @staticmethod
    def hard_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """
        Hard Light - overlay with base and blend swapped.
        
        Stronger contrast effect.
        """
        return np.where(
            blend < 0.5,
            2 * base * blend,
            1 - 2 * (1 - base) * (1 - blend)
        )
    
    @staticmethod
    def vivid_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """
        Vivid Light - combines color burn and color dodge.
        """
        return np.where(
            blend < 0.5,
            BlendModes.color_burn(base, 2 * blend),
            BlendModes.color_dodge(base, 2 * (blend - 0.5))
        )
    
    @staticmethod
    def linear_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """
        Linear Light - combines linear burn and linear dodge.
        """
        return np.clip(base + 2 * blend - 1, 0, 1)
    
    @staticmethod
    def pin_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """
        Pin Light - combines darken and lighten.
        """
        return np.where(
            blend < 0.5,
            np.minimum(base, 2 * blend),
            np.maximum(base, 2 * (blend - 0.5))
        )
    
    @staticmethod
    def hard_mix(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """
        Hard Mix - posterizes to black or white.
        """
        return np.where(base + blend < 1, 0, 1).astype(float)
    
    # ─────────────────────────────────────────────────────────
    # INVERSION MODES
    # ─────────────────────────────────────────────────────────
    
    @staticmethod
    def difference(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """
        Difference - absolute difference between layers.
        
        Black is neutral. Same colors give black.
        """
        return np.abs(base - blend)
    
    @staticmethod
    def exclusion(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """
        Exclusion - similar to difference but lower contrast.
        """
        return base + blend - 2 * base * blend
    
    @staticmethod
    def subtract(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Subtract - subtracts blend from base."""
        return np.clip(base - blend, 0, 1)
    
    @staticmethod
    def divide(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Divide - divides base by blend."""
        return np.clip(base / (blend + 1e-10), 0, 1)
    
    # ─────────────────────────────────────────────────────────
    # COMPONENT MODES (HSL)
    # ─────────────────────────────────────────────────────────
    
    @staticmethod
    def hue(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """
        Hue - takes hue from blend, saturation and luminosity from base.
        """
        if len(base.shape) != 3:
            return base
        
        result = np.zeros_like(base)
        
        for y in range(base.shape[0]):
            for x in range(base.shape[1]):
                # Get base HSL
                bh, bs, bl = ColorSpaceConversion.rgb_to_hsl(*base[y, x])
                # Get blend hue
                blend_h, _, _ = ColorSpaceConversion.rgb_to_hsl(*blend[y, x])
                
                # Combine
                r, g, b = ColorSpaceConversion.hsl_to_rgb(blend_h, bs, bl)
                result[y, x] = [r, g, b]
        
        return result
    
    @staticmethod
    def saturation(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """
        Saturation - takes saturation from blend, hue and luminosity from base.
        """
        if len(base.shape) != 3:
            return base
        
        result = np.zeros_like(base)
        
        for y in range(base.shape[0]):
            for x in range(base.shape[1]):
                bh, bs, bl = ColorSpaceConversion.rgb_to_hsl(*base[y, x])
                _, blend_s, _ = ColorSpaceConversion.rgb_to_hsl(*blend[y, x])
                
                r, g, b = ColorSpaceConversion.hsl_to_rgb(bh, blend_s, bl)
                result[y, x] = [r, g, b]
        
        return result
    
    @staticmethod
    def color(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """
        Color - takes hue and saturation from blend, luminosity from base.
        """
        if len(base.shape) != 3:
            return base
        
        result = np.zeros_like(base)
        
        for y in range(base.shape[0]):
            for x in range(base.shape[1]):
                _, _, bl = ColorSpaceConversion.rgb_to_hsl(*base[y, x])
                blend_h, blend_s, _ = ColorSpaceConversion.rgb_to_hsl(*blend[y, x])
                
                r, g, b = ColorSpaceConversion.hsl_to_rgb(blend_h, blend_s, bl)
                result[y, x] = [r, g, b]
        
        return result
    
    @staticmethod
    def luminosity(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """
        Luminosity - takes luminosity from blend, hue and saturation from base.
        """
        if len(base.shape) != 3:
            return base
        
        result = np.zeros_like(base)
        
        for y in range(base.shape[0]):
            for x in range(base.shape[1]):
                bh, bs, _ = ColorSpaceConversion.rgb_to_hsl(*base[y, x])
                _, _, blend_l = ColorSpaceConversion.rgb_to_hsl(*blend[y, x])
                
                r, g, b = ColorSpaceConversion.hsl_to_rgb(bh, bs, blend_l)
                result[y, x] = [r, g, b]
        
        return result


# ═══════════════════════════════════════════════════════════════
# ALPHA COMPOSITING
# ═══════════════════════════════════════════════════════════════

class AlphaCompositing:
    """Porter-Duff compositing operations"""
    
    @staticmethod
    def over(src: np.ndarray, src_alpha: np.ndarray,
            dst: np.ndarray, dst_alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Source Over - standard alpha compositing.
        
        Places source over destination.
        """
        # Output alpha
        out_alpha = src_alpha + dst_alpha * (1 - src_alpha)
        
        # Output color (with alpha premultiplication)
        out_color = np.zeros_like(src)
        
        mask = out_alpha > 0
        if len(src.shape) == 3:
            for c in range(src.shape[2]):
                out_color[:, :, c] = np.where(
                    mask,
                    (src[:, :, c] * src_alpha + 
                     dst[:, :, c] * dst_alpha * (1 - src_alpha)) / (out_alpha + 1e-10),
                    0
                )
        else:
            out_color = np.where(
                mask,
                (src * src_alpha + dst * dst_alpha * (1 - src_alpha)) / (out_alpha + 1e-10),
                0
            )
        
        return out_color, out_alpha
    
    @staticmethod
    def under(src: np.ndarray, src_alpha: np.ndarray,
             dst: np.ndarray, dst_alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Source Under - places source behind destination."""
        return AlphaCompositing.over(dst, dst_alpha, src, src_alpha)
    
    @staticmethod
    def inside(src: np.ndarray, src_alpha: np.ndarray,
              dst: np.ndarray, dst_alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Source In - source where destination exists."""
        out_alpha = src_alpha * dst_alpha
        out_color = src
        return out_color, out_alpha
    
    @staticmethod
    def outside(src: np.ndarray, src_alpha: np.ndarray,
               dst: np.ndarray, dst_alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Source Out - source where destination doesn't exist."""
        out_alpha = src_alpha * (1 - dst_alpha)
        out_color = src
        return out_color, out_alpha
    
    @staticmethod
    def atop(src: np.ndarray, src_alpha: np.ndarray,
            dst: np.ndarray, dst_alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Source Atop - source over destination, clipped to destination."""
        out_alpha = dst_alpha
        
        out_color = np.zeros_like(src)
        if len(src.shape) == 3:
            for c in range(src.shape[2]):
                out_color[:, :, c] = src[:, :, c] * src_alpha + \
                                    dst[:, :, c] * (1 - src_alpha)
        else:
            out_color = src * src_alpha + dst * (1 - src_alpha)
        
        return out_color, out_alpha
    
    @staticmethod
    def xor(src: np.ndarray, src_alpha: np.ndarray,
           dst: np.ndarray, dst_alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """XOR - source or destination but not both."""
        out_alpha = src_alpha + dst_alpha - 2 * src_alpha * dst_alpha
        
        out_color = np.zeros_like(src)
        mask = out_alpha > 0
        
        if len(src.shape) == 3:
            for c in range(src.shape[2]):
                out_color[:, :, c] = np.where(
                    mask,
                    (src[:, :, c] * src_alpha * (1 - dst_alpha) +
                     dst[:, :, c] * dst_alpha * (1 - src_alpha)) / (out_alpha + 1e-10),
                    0
                )
        else:
            out_color = np.where(
                mask,
                (src * src_alpha * (1 - dst_alpha) +
                 dst * dst_alpha * (1 - src_alpha)) / (out_alpha + 1e-10),
                0
            )
        
        return out_color, out_alpha


# ═══════════════════════════════════════════════════════════════
# LAYER SYSTEM
# ═══════════════════════════════════════════════════════════════

class Layer:
    """Single compositing layer"""
    
    def __init__(self, image: np.ndarray, 
                alpha: np.ndarray = None,
                blend_mode: str = 'normal',
                opacity: float = 1.0,
                name: str = 'Layer'):
        self.image = image.astype(float) / 255 if image.max() > 1 else image.astype(float)
        
        if alpha is None:
            self.alpha = np.ones(image.shape[:2])
        else:
            self.alpha = alpha.astype(float) / 255 if alpha.max() > 1 else alpha.astype(float)
        
        self.blend_mode = blend_mode
        self.opacity = opacity
        self.name = name
        self.visible = True
        self.mask = None
    
    def set_mask(self, mask: np.ndarray):
        """Set layer mask (grayscale, white = visible)."""
        self.mask = mask.astype(float) / 255 if mask.max() > 1 else mask.astype(float)
    
    def get_effective_alpha(self) -> np.ndarray:
        """Get alpha with mask applied."""
        effective = self.alpha * self.opacity
        if self.mask is not None:
            effective = effective * self.mask
        return effective


class LayerStack:
    """Stack of compositing layers"""
    
    BLEND_FUNCTIONS = {
        'normal': BlendModes.normal,
        'multiply': BlendModes.multiply,
        'screen': BlendModes.screen,
        'overlay': BlendModes.overlay,
        'soft_light': BlendModes.soft_light,
        'hard_light': BlendModes.hard_light,
        'color_dodge': BlendModes.color_dodge,
        'color_burn': BlendModes.color_burn,
        'darken': BlendModes.darken,
        'lighten': BlendModes.lighten,
        'difference': BlendModes.difference,
        'exclusion': BlendModes.exclusion,
        'hue': BlendModes.hue,
        'saturation': BlendModes.saturation,
        'color': BlendModes.color,
        'luminosity': BlendModes.luminosity,
    }
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.layers = []
        self.background_color = (1, 1, 1)  # White
    
    def add_layer(self, layer: Layer, index: int = -1):
        """Add layer at index (-1 = top)."""
        if index == -1:
            self.layers.append(layer)
        else:
            self.layers.insert(index, layer)
    
    def remove_layer(self, index: int):
        """Remove layer at index."""
        if 0 <= index < len(self.layers):
            self.layers.pop(index)
    
    def move_layer(self, from_idx: int, to_idx: int):
        """Move layer from one position to another."""
        if 0 <= from_idx < len(self.layers) and 0 <= to_idx < len(self.layers):
            layer = self.layers.pop(from_idx)
            self.layers.insert(to_idx, layer)
    
    def composite(self) -> np.ndarray:
        """Composite all visible layers and return result."""
        # Start with background
        result = np.full((self.height, self.width, 3), self.background_color)
        result_alpha = np.ones((self.height, self.width))
        
        for layer in self.layers:
            if not layer.visible:
                continue
            
            # Get blend function
            blend_func = self.BLEND_FUNCTIONS.get(layer.blend_mode, BlendModes.normal)
            
            # Apply blend
            layer_alpha = layer.get_effective_alpha()
            
            # Blend colors
            if layer.blend_mode == 'normal':
                blended = blend_func(result, layer.image, layer.opacity)
            else:
                blended = blend_func(result, layer.image)
            
            # Composite with alpha
            result, result_alpha = AlphaCompositing.over(
                blended, layer_alpha,
                result, result_alpha
            )
        
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    
    def flatten(self) -> np.ndarray:
        """Flatten all layers into single image."""
        return self.composite()


# ═══════════════════════════════════════════════════════════════
# MASKING TECHNIQUES
# ═══════════════════════════════════════════════════════════════

class MaskingTechniques:
    """Advanced masking and selection techniques"""
    
    @staticmethod
    def color_range_mask(image: np.ndarray,
                        target_color: Tuple[int, int, int],
                        tolerance: float = 30,
                        fuzziness: float = 20) -> np.ndarray:
        """
        Create mask based on color similarity.
        
        Similar to Photoshop's Color Range selection.
        """
        if len(image.shape) != 3:
            return np.ones(image.shape)
        
        # Calculate color distance
        target = np.array(target_color)
        distance = np.sqrt(np.sum((image.astype(float) - target)**2, axis=2))
        
        # Create soft mask
        mask = 1 - np.clip((distance - tolerance) / fuzziness, 0, 1)
        
        return mask
    
    @staticmethod
    def luminosity_mask(image: np.ndarray,
                       range_type: str = 'highlights',
                       threshold: float = 0.5,
                       feather: float = 0.2) -> np.ndarray:
        """
        Create luminosity mask for targeted adjustments.
        
        range_type: 'highlights', 'midtones', 'shadows'
        """
        if len(image.shape) == 3:
            luminosity = np.mean(image, axis=2) / 255
        else:
            luminosity = image / 255
        
        if range_type == 'highlights':
            # Mask for bright areas
            mask = np.clip((luminosity - threshold) / feather, 0, 1)
        
        elif range_type == 'shadows':
            # Mask for dark areas
            mask = np.clip((threshold - luminosity) / feather, 0, 1)
        
        elif range_type == 'midtones':
            # Mask for middle tones
            mid = 0.5
            dist = np.abs(luminosity - mid)
            mask = np.clip(1 - (dist - 0.25) / feather, 0, 1)
        
        else:
            mask = np.ones_like(luminosity)
        
        return mask
    
    @staticmethod
    def edge_mask(image: np.ndarray,
                 threshold: float = 0.1,
                 blur: float = 1.0) -> np.ndarray:
        """
        Create mask highlighting edges.
        
        Useful for selective sharpening or edge effects.
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)
        
        # Sobel edge detection
        gx = ndimage.sobel(gray, axis=1)
        gy = ndimage.sobel(gray, axis=0)
        edges = np.sqrt(gx**2 + gy**2)
        
        # Normalize
        edges = edges / (edges.max() + 1e-10)
        
        # Threshold
        mask = np.clip((edges - threshold) / (1 - threshold), 0, 1)
        
        # Blur
        if blur > 0:
            mask = ndimage.gaussian_filter(mask, blur)
        
        return mask
    
    @staticmethod
    def depth_mask_from_blur(image: np.ndarray,
                            kernel_size: int = 15) -> np.ndarray:
        """
        Estimate depth mask from image blur (simple approach).
        
        Areas with less detail are assumed to be further away.
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)
        
        # Calculate local variance (measure of sharpness)
        local_mean = ndimage.uniform_filter(gray, size=kernel_size)
        local_sqr_mean = ndimage.uniform_filter(gray**2, size=kernel_size)
        local_var = local_sqr_mean - local_mean**2
        
        # High variance = sharp = near
        # Low variance = blurry = far
        sharpness = np.sqrt(np.maximum(local_var, 0))
        
        # Normalize and invert (far = 1, near = 0)
        depth = 1 - sharpness / (sharpness.max() + 1e-10)
        
        # Smooth the depth map
        depth = ndimage.gaussian_filter(depth, 10)
        
        return depth
    
    @staticmethod
    def gradient_mask(width: int, height: int,
                     start_point: Tuple[float, float],
                     end_point: Tuple[float, float],
                     gradient_type: str = 'linear') -> np.ndarray:
        """
        Create gradient mask.
        
        gradient_type: 'linear', 'radial', 'angular', 'reflected'
        """
        mask = np.zeros((height, width))
        
        x1, y1 = start_point
        x2, y2 = end_point
        
        if gradient_type == 'linear':
            # Linear gradient from start to end
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx**2 + dy**2)
            
            if length == 0:
                return np.zeros((height, width))
            
            for y in range(height):
                for x in range(width):
                    # Project point onto gradient line
                    t = ((x - x1) * dx + (y - y1) * dy) / (length**2)
                    mask[y, x] = np.clip(t, 0, 1)
        
        elif gradient_type == 'radial':
            # Radial gradient from center
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            radius = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2
            
            for y in range(height):
                for x in range(width):
                    dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                    mask[y, x] = np.clip(dist / radius, 0, 1)
        
        elif gradient_type == 'angular':
            # Angular/conical gradient
            cx, cy = x1, y1
            
            for y in range(height):
                for x in range(width):
                    angle = math.atan2(y - cy, x - cx)
                    mask[y, x] = (angle + math.pi) / (2 * math.pi)
        
        elif gradient_type == 'reflected':
            # Reflected linear gradient
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx**2 + dy**2)
            
            if length == 0:
                return np.zeros((height, width))
            
            for y in range(height):
                for x in range(width):
                    t = ((x - x1) * dx + (y - y1) * dy) / (length**2)
                    t = abs(2 * t - 1) if t < 1 else 1
                    mask[y, x] = 1 - np.clip(t, 0, 1)
        
        return mask
```

---

## Chapter 21: Image Analysis and Processing Pipelines

### 21.1 Histogram Analysis

```python
"""
IMAGE ANALYSIS AND PROCESSING PIPELINES

Tools for analyzing images and building processing workflows.
"""

import numpy as np
from typing import Tuple, List, Dict, Callable, Optional

# ═══════════════════════════════════════════════════════════════
# HISTOGRAM ANALYSIS
# ═══════════════════════════════════════════════════════════════

class HistogramAnalysis:
    """Image histogram analysis tools"""
    
    @staticmethod
    def compute_histogram(image: np.ndarray, 
                         bins: int = 256,
                         channel: int = None) -> np.ndarray:
        """
        Compute histogram of image.
        
        channel: None = all/gray, 0=R, 1=G, 2=B
        """
        if len(image.shape) == 3 and channel is None:
            # Convert to grayscale
            data = np.mean(image, axis=2)
        elif len(image.shape) == 3:
            data = image[:, :, channel]
        else:
            data = image
        
        hist, _ = np.histogram(data.flatten(), bins=bins, range=(0, 256))
        return hist
    
    @staticmethod
    def compute_cumulative_histogram(image: np.ndarray,
                                     channel: int = None) -> np.ndarray:
        """Compute cumulative histogram."""
        hist = HistogramAnalysis.compute_histogram(image, channel=channel)
        return np.cumsum(hist)
    
    @staticmethod
    def histogram_statistics(image: np.ndarray) -> Dict:
        """
        Compute statistical measures from histogram.
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)
        
        flat = gray.flatten()
        
        return {
            'mean': np.mean(flat),
            'median': np.median(flat),
            'std': np.std(flat),
            'min': np.min(flat),
            'max': np.max(flat),
            'dynamic_range': np.max(flat) - np.min(flat),
            'contrast': np.std(flat) / (np.mean(flat) + 1e-10),
            'percentile_1': np.percentile(flat, 1),
            'percentile_99': np.percentile(flat, 99),
        }
    
    @staticmethod
    def histogram_equalization(image: np.ndarray,
                              clip_limit: float = None) -> np.ndarray:
        """
        Apply histogram equalization for contrast enhancement.
        
        clip_limit: If specified, uses CLAHE (Contrast Limited AHE)
        """
        if len(image.shape) == 3:
            # Convert to LAB and equalize L channel
            result = np.zeros_like(image)
            
            # Simple per-channel approach
            for c in range(3):
                result[:, :, c] = HistogramAnalysis._equalize_channel(
                    image[:, :, c], clip_limit
                )
            
            return result
        else:
            return HistogramAnalysis._equalize_channel(image, clip_limit)
    
    @staticmethod
    def _equalize_channel(channel: np.ndarray,
                         clip_limit: float = None) -> np.ndarray:
        """Equalize single channel."""
        hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
        
        if clip_limit is not None:
            # Clip histogram for CLAHE
            clip_threshold = clip_limit * np.mean(hist)
            excess = np.sum(np.maximum(hist - clip_threshold, 0))
            hist = np.minimum(hist, clip_threshold)
            hist = hist + excess / 256
        
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]
        
        return np.interp(channel.flatten(), bins[:-1], cdf_normalized).reshape(channel.shape).astype(np.uint8)
    
    @staticmethod
    def adaptive_histogram_equalization(image: np.ndarray,
                                       tile_size: int = 64,
                                       clip_limit: float = 2.0) -> np.ndarray:
        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        Applies equalization to tiles for local contrast enhancement.
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image
        
        h, w = gray.shape
        result = np.zeros_like(gray, dtype=np.float64)
        
        # Process tiles
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # Get tile
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                tile = gray[y:y_end, x:x_end]
                
                # Equalize tile
                equalized = HistogramAnalysis._equalize_channel(tile, clip_limit)
                
                result[y:y_end, x:x_end] = equalized
        
        return result.astype(np.uint8)
    
    @staticmethod
    def match_histogram(source: np.ndarray,
                       reference: np.ndarray) -> np.ndarray:
        """
        Match histogram of source to reference image.
        
        Useful for color/tone matching between images.
        """
        if len(source.shape) == 3 and len(reference.shape) == 3:
            result = np.zeros_like(source)
            
            for c in range(3):
                result[:, :, c] = HistogramAnalysis._match_channel(
                    source[:, :, c], reference[:, :, c]
                )
            
            return result
        else:
            return HistogramAnalysis._match_channel(source, reference)
    
    @staticmethod
    def _match_channel(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Match histogram of single channel."""
        # Compute CDFs
        src_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])
        
        src_cdf = src_hist.cumsum()
        ref_cdf = ref_hist.cumsum()
        
        # Normalize
        src_cdf = src_cdf / src_cdf[-1]
        ref_cdf = ref_cdf / ref_cdf[-1]
        
        # Create mapping
        lookup = np.zeros(256)
        for i in range(256):
            j = 255
            while j > 0 and ref_cdf[j] > src_cdf[i]:
                j -= 1
            lookup[i] = j
        
        return lookup[source].astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# IMAGE QUALITY METRICS
# ═══════════════════════════════════════════════════════════════

class ImageQualityMetrics:
    """Image quality assessment"""
    
    @staticmethod
    def mse(image1: np.ndarray, image2: np.ndarray) -> float:
        """Mean Squared Error between two images."""
        return np.mean((image1.astype(float) - image2.astype(float))**2)
    
    @staticmethod
    def psnr(image1: np.ndarray, image2: np.ndarray, 
            max_val: float = 255) -> float:
        """
        Peak Signal-to-Noise Ratio.
        
        Higher = better quality.
        Typical values: 30-50 dB for lossy compression.
        """
        mse_val = ImageQualityMetrics.mse(image1, image2)
        if mse_val == 0:
            return float('inf')
        return 10 * np.log10(max_val**2 / mse_val)
    
    @staticmethod
    def ssim(image1: np.ndarray, image2: np.ndarray,
            window_size: int = 11,
            k1: float = 0.01,
            k2: float = 0.03) -> float:
        """
        Structural Similarity Index.
        
        More perceptually relevant than MSE/PSNR.
        Range: -1 to 1 (1 = identical)
        """
        if len(image1.shape) == 3:
            image1 = np.mean(image1, axis=2)
        if len(image2.shape) == 3:
            image2 = np.mean(image2, axis=2)
        
        image1 = image1.astype(float)
        image2 = image2.astype(float)
        
        L = 255  # Dynamic range
        c1 = (k1 * L)**2
        c2 = (k2 * L)**2
        
        # Compute means
        mu1 = ndimage.uniform_filter(image1, window_size)
        mu2 = ndimage.uniform_filter(image2, window_size)
        
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        
        # Compute variances and covariance
        sigma1_sq = ndimage.uniform_filter(image1**2, window_size) - mu1_sq
        sigma2_sq = ndimage.uniform_filter(image2**2, window_size) - mu2_sq
        sigma12 = ndimage.uniform_filter(image1 * image2, window_size) - mu1_mu2
        
        # SSIM formula
        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        
        ssim_map = numerator / denominator
        
        return np.mean(ssim_map)
    
    @staticmethod
    def sharpness(image: np.ndarray) -> float:
        """
        Estimate image sharpness using Laplacian variance.
        
        Higher = sharper image.
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)
        
        laplacian = ndimage.laplace(gray)
        return np.var(laplacian)
    
    @staticmethod
    def noise_estimate(image: np.ndarray) -> float:
        """
        Estimate noise level using median absolute deviation.
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)
        
        # High-pass filter to isolate noise
        kernel = np.array([
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1]
        ])
        
        filtered = ndimage.convolve(gray, kernel)
        
        # Median absolute deviation
        mad = np.median(np.abs(filtered - np.median(filtered)))
        
        # Convert to standard deviation estimate
        sigma = mad / 0.6745
        
        return sigma
    
    @staticmethod
    def exposure_quality(image: np.ndarray) -> Dict:
        """
        Assess exposure quality.
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)
        
        mean_brightness = np.mean(gray)
        
        # Count clipped pixels
        underexposed = np.sum(gray < 10) / gray.size
        overexposed = np.sum(gray > 245) / gray.size
        
        return {
            'mean_brightness': mean_brightness,
            'is_dark': mean_brightness < 85,
            'is_bright': mean_brightness > 170,
            'underexposed_ratio': underexposed,
            'overexposed_ratio': overexposed,
            'well_exposed': underexposed < 0.01 and overexposed < 0.01,
            'exposure_rating': 'good' if 85 < mean_brightness < 170 else 
                             ('dark' if mean_brightness < 85 else 'bright')
        }


# ═══════════════════════════════════════════════════════════════
# PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════

class ProcessingPipeline:
    """Build and execute image processing pipelines"""
    
    def __init__(self, name: str = 'Pipeline'):
        self.name = name
        self.steps = []
        self.history = []
    
    def add_step(self, name: str, 
                func: Callable,
                params: Dict = None,
                enabled: bool = True):
        """Add processing step to pipeline."""
        self.steps.append({
            'name': name,
            'function': func,
            'params': params or {},
            'enabled': enabled
        })
    
    def remove_step(self, index: int):
        """Remove step by index."""
        if 0 <= index < len(self.steps):
            self.steps.pop(index)
    
    def enable_step(self, index: int, enabled: bool = True):
        """Enable or disable a step."""
        if 0 <= index < len(self.steps):
            self.steps[index]['enabled'] = enabled
    
    def execute(self, image: np.ndarray, 
               record_history: bool = False) -> np.ndarray:
        """
        Execute pipeline on image.
        
        record_history: If True, saves intermediate results.
        """
        result = image.copy()
        
        if record_history:
            self.history = [{'name': 'Original', 'image': image.copy()}]
        
        for step in self.steps:
            if not step['enabled']:
                continue
            
            try:
                result = step['function'](result, **step['params'])
                
                if record_history:
                    self.history.append({
                        'name': step['name'],
                        'image': result.copy()
                    })
            
            except Exception as e:
                print(f"Error in step '{step['name']}': {e}")
                continue
        
        return result
    
    def preview_step(self, image: np.ndarray, step_index: int) -> np.ndarray:
        """Execute pipeline up to and including specified step."""
        result = image.copy()
        
        for i, step in enumerate(self.steps):
            if i > step_index:
                break
            
            if not step['enabled']:
                continue
            
            result = step['function'](result, **step['params'])
        
        return result
    
    def to_dict(self) -> Dict:
        """Serialize pipeline to dictionary (for saving)."""
        return {
            'name': self.name,
            'steps': [
                {
                    'name': s['name'],
                    'function_name': s['function'].__name__,
                    'params': s['params'],
                    'enabled': s['enabled']
                }
                for s in self.steps
            ]
        }
    
    @staticmethod
    def create_standard_pipeline() -> 'ProcessingPipeline':
        """Create a standard photo processing pipeline."""
        pipeline = ProcessingPipeline('Standard Photo Processing')
        
        # Typical photo processing steps
        pipeline.add_step(
            'White Balance',
            lambda img: img,  # Placeholder
            enabled=False
        )
        
        pipeline.add_step(
            'Exposure',
            ColorAdjustments.levels,
            {'input_black': 0, 'input_white': 255, 'gamma': 1.0}
        )
        
        pipeline.add_step(
            'Contrast',
            ColorAdjustments.contrast,
            {'factor': 1.1}
        )
        
        pipeline.add_step(
            'Saturation',
            ColorAdjustments.saturation,
            {'factor': 1.1}
        )
        
        pipeline.add_step(
            'Sharpen',
            SharpeningEffects.unsharp_mask,
            {'radius': 1.0, 'amount': 0.5}
        )
        
        return pipeline
```

---

# PART VIII: PLUGIN DEVELOPMENT

---

## Chapter 22: GIMP Plugin Development

### 22.1 Python-Fu Plugin Architecture

```python
"""
GIMP PYTHON-FU PLUGIN DEVELOPMENT - COMPREHENSIVE GUIDE

Create custom GIMP plugins using Python-Fu (GIMP 2.x) and 
the new Python 3 API (GIMP 3.x).
"""

# ═══════════════════════════════════════════════════════════════
# GIMP 2.x PYTHON-FU PLUGIN TEMPLATE
# ═══════════════════════════════════════════════════════════════

"""
GIMP 2.x Plugin Template

Save this file in your GIMP plug-ins folder:
- Linux: ~/.gimp-2.10/plug-ins/
- Windows: C:\\Users\\<username>\\AppData\\Roaming\\GIMP\\2.10\\plug-ins\\
- macOS: ~/Library/Application Support/GIMP/2.10/plug-ins/

Make sure the file is executable (Linux/macOS: chmod +x filename.py)
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# GIMP 2.x imports
from gimpfu import *
import gimp
from gimp import pdb
import math

def my_plugin_function(image, drawable, param1, param2):
    """
    Main plugin function.
    
    Parameters:
        image: The current GIMP image
        drawable: The active layer/drawable
        param1, param2: Custom parameters from the dialog
    """
    # Start an undo group - all operations can be undone as one step
    gimp.progress_init("Processing image...")
    pdb.gimp_image_undo_group_start(image)
    
    try:
        # Get image dimensions
        width = drawable.width
        height = drawable.height
        
        # Get pixel region for reading/writing
        src_rgn = drawable.get_pixel_rgn(0, 0, width, height, False, False)
        dst_rgn = drawable.get_pixel_rgn(0, 0, width, height, True, True)
        
        # Get pixel data as string (GIMP 2.x style)
        src_pixels = src_rgn[0:width, 0:height]
        
        # Determine bytes per pixel
        bpp = drawable.bpp
        has_alpha = drawable.has_alpha
        
        # Process pixels
        result = process_pixels(src_pixels, width, height, bpp, param1, param2)
        
        # Write result back
        dst_rgn[0:width, 0:height] = result
        
        # Merge shadow and update drawable
        drawable.flush()
        drawable.merge_shadow(True)
        drawable.update(0, 0, width, height)
        
        # Update display
        gimp.displays_flush()
        
    except Exception as e:
        gimp.message("Error: " + str(e))
    
    finally:
        # End undo group
        pdb.gimp_image_undo_group_end(image)
        gimp.progress_end()


def process_pixels(pixels, width, height, bpp, param1, param2):
    """
    Process pixel data.
    
    pixels: String of pixel data
    width, height: Image dimensions
    bpp: Bytes per pixel (3=RGB, 4=RGBA)
    """
    # Convert to list for modification
    pixel_list = list(pixels)
    
    # Process each pixel
    for y in range(height):
        for x in range(width):
            idx = (y * width + x) * bpp
            
            # Get pixel values
            r = ord(pixel_list[idx])
            g = ord(pixel_list[idx + 1])
            b = ord(pixel_list[idx + 2])
            
            # Apply effect (example: adjust brightness)
            r = min(255, int(r * param1))
            g = min(255, int(g * param1))
            b = min(255, int(b * param1))
            
            # Write back
            pixel_list[idx] = chr(r)
            pixel_list[idx + 1] = chr(g)
            pixel_list[idx + 2] = chr(b)
        
        # Update progress
        gimp.progress_update(float(y) / height)
    
    return ''.join(pixel_list)


# Register the plugin
register(
    "python_fu_my_plugin",           # Unique procedure name
    "My Custom Plugin",               # Plugin description
    "A detailed description of what this plugin does",  # Help text
    "Your Name",                      # Author
    "Your Name",                      # Copyright
    "2024",                           # Year
    "<Image>/Filters/My Plugin...",  # Menu location
    "RGB*, GRAY*",                    # Supported image modes
    [
        (PF_FLOAT, "param1", "Brightness multiplier", 1.0),
        (PF_INT, "param2", "Another parameter", 50),
    ],                                # Parameters
    [],                               # Return values
    my_plugin_function                # Function to call
)

main()


# ═══════════════════════════════════════════════════════════════
# GIMP 3.x (GEGL-based) PLUGIN TEMPLATE  
# ═══════════════════════════════════════════════════════════════

"""
GIMP 3.x Plugin Template

The new GIMP 3.x uses a different Python API based on GObject Introspection.
"""

#!/usr/bin/env python3

import gi
gi.require_version('Gimp', '3.0')
gi.require_version('GimpUi', '3.0')
gi.require_version('Gegl', '0.4')

from gi.repository import Gimp, GimpUi, GObject, GLib, Gio, Gegl

class MyPlugin(Gimp.PlugIn):
    """GIMP 3.x Plugin Class"""
    
    # GObject properties for plugin parameters
    __gproperties__ = {
        "brightness": (float,
                      "Brightness",
                      "Brightness multiplier",
                      0.0, 5.0, 1.0,
                      GObject.ParamFlags.READWRITE),
    }
    
    def __init__(self):
        Gimp.PlugIn.__init__(self)
        self._brightness = 1.0
    
    def do_query_procedures(self):
        """Return list of procedure names this plugin provides"""
        return ["python-fu-my-plugin-3"]
    
    def do_set_i18n(self, name):
        """Set internationalization"""
        return False, None, None
    
    def do_create_procedure(self, name):
        """Create and configure the procedure"""
        procedure = Gimp.ImageProcedure.new(
            self,
            name,
            Gimp.PDBProcType.PLUGIN,
            self.run,
            None
        )
        
        procedure.set_image_types("RGB*, GRAY*")
        procedure.set_sensitivity_mask(
            Gimp.ProcedureSensitivityMask.DRAWABLE
        )
        
        procedure.set_menu_label("My GIMP 3 Plugin")
        procedure.add_menu_path("<Image>/Filters/My Plugins")
        
        procedure.set_documentation(
            "My Custom Plugin",
            "A detailed description of the plugin",
            name
        )
        procedure.set_attribution("Your Name", "Copyright", "2024")
        
        # Add parameters
        procedure.add_double_argument(
            "brightness",
            "Brightness",
            "Brightness multiplier",
            0.0, 5.0, 1.0,
            GObject.ParamFlags.READWRITE
        )
        
        return procedure
    
    def run(self, procedure, run_mode, image, drawables, config, data):
        """Main plugin execution"""
        if run_mode == Gimp.RunMode.INTERACTIVE:
            GimpUi.init("my-plugin")
            dialog = GimpUi.ProcedureDialog.new(procedure, config, "My Plugin")
            dialog.fill(None)
            
            if not dialog.run():
                dialog.destroy()
                return procedure.new_return_values(
                    Gimp.PDBStatusType.CANCEL,
                    GLib.Error()
                )
            dialog.destroy()
        
        brightness = config.get_property("brightness")
        
        # Process each drawable
        for drawable in drawables:
            self.process_drawable(image, drawable, brightness)
        
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())
    
    def process_drawable(self, image, drawable, brightness):
        """Process a single drawable"""
        # Start undo group
        image.undo_group_start()
        
        try:
            # Get GEGL buffer
            buffer = drawable.get_buffer()
            shadow = drawable.get_shadow_buffer()
            
            # Get extent
            extent = buffer.get_extent()
            
            # Use GEGL for processing
            gegl = Gegl.Node()
            
            # Create processing graph
            src = gegl.create_child("gegl:buffer-source")
            src.set_property("buffer", buffer)
            
            brightness_node = gegl.create_child("gegl:brightness-contrast")
            brightness_node.set_property("brightness", brightness - 1.0)
            
            sink = gegl.create_child("gegl:write-buffer")
            sink.set_property("buffer", shadow)
            
            # Connect nodes
            src.connect_to("output", brightness_node, "input")
            brightness_node.connect_to("output", sink, "input")
            
            # Process
            sink.process()
            
            # Flush and merge
            shadow.flush()
            drawable.merge_shadow(True)
            drawable.update(extent.x, extent.y, extent.width, extent.height)
            
        finally:
            image.undo_group_end()


# Register plugin
Gimp.main(MyPlugin.__gtype__, sys.argv)


# ═══════════════════════════════════════════════════════════════
# ADVANCED GIMP PLUGIN TECHNIQUES
# ═══════════════════════════════════════════════════════════════

class GimpPluginAdvanced:
    """Advanced GIMP plugin development patterns"""
    
    @staticmethod
    def create_new_layer(image, name="New Layer", width=None, height=None,
                        opacity=100, mode=NORMAL_MODE):
        """
        Create a new layer in the image.
        
        GIMP 2.x example.
        """
        if width is None:
            width = image.width
        if height is None:
            height = image.height
        
        # Create layer
        layer = gimp.Layer(image, name, width, height, 
                          RGBA_IMAGE, opacity, mode)
        
        # Add to image
        image.add_layer(layer, 0)  # 0 = top position
        
        return layer
    
    @staticmethod
    def apply_convolution_filter(drawable, kernel, divisor=None, offset=0):
        """
        Apply convolution filter using GIMP's built-in function.
        """
        if divisor is None:
            divisor = sum(sum(row) for row in kernel)
            if divisor == 0:
                divisor = 1
        
        # Flatten kernel for GIMP
        flat_kernel = []
        for row in kernel:
            flat_kernel.extend(row)
        
        kernel_size = len(kernel)
        
        pdb.plug_in_convmatrix(
            gimp.image_list()[0],
            drawable,
            kernel_size * kernel_size,  # matrix size
            flat_kernel,
            False,  # alpha-weighting
            divisor,
            offset,
            5,  # bmode
            True,  # preserve alpha
        )
    
    @staticmethod
    def get_selection_bounds(image):
        """
        Get bounds of current selection, or full image if no selection.
        """
        non_empty, x1, y1, x2, y2 = pdb.gimp_selection_bounds(image)
        
        if non_empty:
            return (x1, y1, x2 - x1, y2 - y1)
        else:
            return (0, 0, image.width, image.height)
    
    @staticmethod
    def iterate_tiles(drawable, x, y, width, height, shadow=True):
        """
        Generator for iterating over tiles efficiently.
        
        GIMP processes large images in tiles for memory efficiency.
        """
        rgn = drawable.get_pixel_rgn(x, y, width, height, shadow, shadow)
        
        # Tile iteration
        for i in range(rgn.nrows):
            for j in range(rgn.ncols):
                # Get tile bounds
                tx = x + j * rgn.tilewidth
                ty = y + i * rgn.tileheight
                tw = min(rgn.tilewidth, x + width - tx)
                th = min(rgn.tileheight, y + height - ty)
                
                # Get tile data
                tile_data = rgn[tx:tx+tw, ty:ty+th]
                
                yield tx, ty, tw, th, tile_data
    
    @staticmethod
    def create_preview_dialog(title, preview_func):
        """
        Create a dialog with live preview functionality.
        
        GIMP 2.x example using gimp-python.
        """
        dialog = gimp.Dialog(
            title,
            None,
            0,
            gtk.MESSAGE_OTHER,
            (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
             gtk.STOCK_OK, gtk.RESPONSE_OK)
        )
        
        # Preview checkbox
        preview_check = gtk.CheckButton("Preview")
        preview_check.set_active(True)
        
        # Parameter controls...
        
        def on_preview_toggle(widget):
            if widget.get_active():
                preview_func()
        
        preview_check.connect("toggled", on_preview_toggle)
        
        return dialog


# ═══════════════════════════════════════════════════════════════
# SCRIPT-FU (SCHEME) EXAMPLES
# ═══════════════════════════════════════════════════════════════

"""
GIMP Script-Fu Examples

Script-Fu uses Scheme (a Lisp dialect). These scripts go in:
- Linux: ~/.gimp-2.10/scripts/
- Windows: C:\\Users\\<username>\\AppData\\Roaming\\GIMP\\2.10\\scripts\\
"""

SCRIPT_FU_EXAMPLE = """
; Simple Script-Fu example
; Save as my-script.scm in your GIMP scripts folder

(define (script-fu-my-effect image drawable brightness)
  (gimp-image-undo-group-start image)
  
  ; Convert brightness to GIMP range
  (let* (
      (brightness-adjust (* (- brightness 1.0) 127))
    )
    
    ; Apply brightness-contrast
    (gimp-brightness-contrast drawable brightness-adjust 0)
  )
  
  (gimp-image-undo-group-end image)
  (gimp-displays-flush)
)

(script-fu-register
  "script-fu-my-effect"                     ; Function name
  "My Effect"                                ; Menu label
  "Adjusts brightness of the image"          ; Description
  "Your Name"                                ; Author
  "Copyright"                                ; Copyright
  "2024"                                     ; Date
  "RGB* GRAY*"                               ; Image type
  SF-IMAGE    "Image"      0                 ; Parameters
  SF-DRAWABLE "Drawable"   0
  SF-ADJUSTMENT "Brightness" '(1.0 0.0 5.0 0.1 0.5 1 0)
)

(script-fu-menu-register "script-fu-my-effect" "<Image>/Filters/My Scripts")
"""

# ═══════════════════════════════════════════════════════════════
# BATCH PROCESSING PLUGIN
# ═══════════════════════════════════════════════════════════════

BATCH_PROCESSING_TEMPLATE = """
#!/usr/bin/env python
# GIMP Batch Processing Plugin

from gimpfu import *
import os
import glob

def batch_process_folder(input_folder, output_folder, operation, file_format):
    '''
    Batch process all images in a folder.
    '''
    # Get list of files
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.bmp']
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(input_folder, pattern)))
    
    if not files:
        gimp.message("No images found in input folder")
        return
    
    # Create output folder if needed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    total = len(files)
    
    for i, filepath in enumerate(files):
        gimp.progress_init("Processing %s (%d/%d)" % 
                          (os.path.basename(filepath), i+1, total))
        
        try:
            # Load image
            image = pdb.gimp_file_load(filepath, filepath)
            drawable = pdb.gimp_image_get_active_layer(image)
            
            # Apply operation
            if operation == 0:  # Resize
                pdb.gimp_image_scale(image, 800, 600)
            elif operation == 1:  # Sharpen
                pdb.plug_in_sharpen(image, drawable, 50)
            elif operation == 2:  # Auto levels
                pdb.gimp_levels_stretch(drawable)
            elif operation == 3:  # Grayscale
                pdb.gimp_image_convert_grayscale(image)
            
            # Flatten if needed
            pdb.gimp_image_flatten(image)
            drawable = pdb.gimp_image_get_active_layer(image)
            
            # Save
            basename = os.path.splitext(os.path.basename(filepath))[0]
            
            if file_format == 0:  # JPEG
                output_path = os.path.join(output_folder, basename + '.jpg')
                pdb.file_jpeg_save(image, drawable, output_path, output_path,
                                  0.9, 0, 0, 0, "", 0, 0, 0, 0)
            elif file_format == 1:  # PNG
                output_path = os.path.join(output_folder, basename + '.png')
                pdb.file_png_save(image, drawable, output_path, output_path,
                                 0, 9, 0, 0, 0, 0, 0)
            
            # Close image
            pdb.gimp_image_delete(image)
            
        except Exception as e:
            gimp.message("Error processing %s: %s" % (filepath, str(e)))
        
        gimp.progress_update(float(i + 1) / total)
    
    gimp.message("Batch processing complete! Processed %d images." % total)

register(
    "python_fu_batch_processor",
    "Batch Process Images",
    "Process multiple images with the same operation",
    "Your Name",
    "Your Name",
    "2024",
    "<Toolbox>/Batch/Batch Processor...",
    "",
    [
        (PF_DIRNAME, "input_folder", "Input Folder", ""),
        (PF_DIRNAME, "output_folder", "Output Folder", ""),
        (PF_OPTION, "operation", "Operation", 0, 
         ["Resize to 800x600", "Sharpen", "Auto Levels", "Convert to Grayscale"]),
        (PF_OPTION, "file_format", "Output Format", 0, ["JPEG", "PNG"]),
    ],
    [],
    batch_process_folder
)

main()
"""
```

---

## Chapter 23: Photoshop Scripting

### 23.1 ExtendScript (JSX) Development

```javascript
/*
 * PHOTOSHOP EXTENDSCRIPT DEVELOPMENT - COMPREHENSIVE GUIDE
 *
 * ExtendScript is Adobe's JavaScript implementation for automation.
 * Scripts are saved as .jsx files and run from:
 * - File > Scripts > Browse
 * - Placing in Presets/Scripts folder
 */

// ═══════════════════════════════════════════════════════════════
// BASIC SCRIPT TEMPLATE
// ═══════════════════════════════════════════════════════════════

// #target photoshop
// The above directive tells ExtendScript Toolkit which application to target

// Enable double-clicking from file system
#target photoshop

// Set measurement units
app.preferences.rulerUnits = Units.PIXELS;
app.preferences.typeUnits = TypeUnits.PIXELS;

function main() {
    // Check if document is open
    if (app.documents.length === 0) {
        alert("Please open a document first.");
        return;
    }
    
    var doc = app.activeDocument;
    var layer = doc.activeLayer;
    
    // Start history state for undo
    var savedState = doc.activeHistoryState;
    
    try {
        // Your processing code here
        processImage(doc, layer);
        
    } catch (e) {
        alert("Error: " + e.message);
        // Restore original state on error
        doc.activeHistoryState = savedState;
    }
}

function processImage(doc, layer) {
    // Example: Adjust levels
    var levels = layer.adjustLevels(
        [0, 255],      // Input shadows, highlights
        [0, 255],      // Output shadows, highlights
        1.0,           // Gamma
        [0, 255],      // Input channels
        [0, 255]       // Output channels
    );
}

// Run the script
main();


// ═══════════════════════════════════════════════════════════════
// DIALOG UI CREATION
// ═══════════════════════════════════════════════════════════════

function showDialog() {
    /*
     * Create a ScriptUI dialog with controls.
     */
    var dialog = new Window('dialog', 'My Plugin');
    
    // Panel for grouping controls
    var mainPanel = dialog.add('panel', undefined, 'Settings');
    mainPanel.alignChildren = ['fill', 'top'];
    
    // Slider control
    var sliderGroup = mainPanel.add('group');
    sliderGroup.add('statictext', undefined, 'Brightness:');
    var brightnessSlider = sliderGroup.add('slider', undefined, 0, -100, 100);
    var brightnessValue = sliderGroup.add('statictext', undefined, '0');
    brightnessValue.characters = 5;
    
    brightnessSlider.onChanging = function() {
        brightnessValue.text = Math.round(this.value);
    };
    
    // Dropdown control
    var dropdownGroup = mainPanel.add('group');
    dropdownGroup.add('statictext', undefined, 'Mode:');
    var modeDropdown = dropdownGroup.add('dropdownlist', undefined, 
        ['Normal', 'Multiply', 'Screen', 'Overlay']);
    modeDropdown.selection = 0;
    
    // Checkbox
    var preserveAlpha = mainPanel.add('checkbox', undefined, 'Preserve Alpha');
    preserveAlpha.value = true;
    
    // Radio buttons
    var radioPanel = mainPanel.add('panel', undefined, 'Quality');
    var qualityLow = radioPanel.add('radiobutton', undefined, 'Low');
    var qualityMed = radioPanel.add('radiobutton', undefined, 'Medium');
    var qualityHigh = radioPanel.add('radiobutton', undefined, 'High');
    qualityMed.value = true;
    
    // Text input
    var textGroup = mainPanel.add('group');
    textGroup.add('statictext', undefined, 'Output Name:');
    var outputName = textGroup.add('edittext', undefined, 'output');
    outputName.characters = 20;
    
    // Buttons
    var buttonGroup = dialog.add('group');
    buttonGroup.add('button', undefined, 'OK', {name: 'ok'});
    buttonGroup.add('button', undefined, 'Cancel', {name: 'cancel'});
    
    // Preview button
    var previewBtn = buttonGroup.add('button', undefined, 'Preview');
    previewBtn.onClick = function() {
        // Apply preview
        applyEffect(brightnessSlider.value);
    };
    
    // Show dialog
    if (dialog.show() === 1) {
        // OK pressed - return values
        return {
            brightness: brightnessSlider.value,
            mode: modeDropdown.selection.text,
            preserveAlpha: preserveAlpha.value,
            quality: qualityLow.value ? 'low' : (qualityMed.value ? 'med' : 'high'),
            outputName: outputName.text
        };
    }
    
    return null;
}


// ═══════════════════════════════════════════════════════════════
// LAYER OPERATIONS
// ═══════════════════════════════════════════════════════════════

function layerOperations() {
    var doc = app.activeDocument;
    
    // Create new layer
    var newLayer = doc.artLayers.add();
    newLayer.name = "New Layer";
    newLayer.blendMode = BlendMode.NORMAL;
    newLayer.opacity = 100;
    
    // Duplicate layer
    var duplicate = doc.activeLayer.duplicate();
    
    // Move layer
    newLayer.move(doc.layers[0], ElementPlacement.PLACEBEFORE);
    
    // Merge layers
    // doc.mergeVisibleLayers();
    
    // Create layer group
    var group = doc.layerSets.add();
    group.name = "My Group";
    
    // Add layer to group
    newLayer.move(group, ElementPlacement.INSIDE);
    
    // Flatten image
    // doc.flatten();
    
    // Get all layers recursively
    function getAllLayers(parent) {
        var layers = [];
        
        for (var i = 0; i < parent.layers.length; i++) {
            var layer = parent.layers[i];
            
            if (layer.typename === "LayerSet") {
                layers = layers.concat(getAllLayers(layer));
            } else {
                layers.push(layer);
            }
        }
        
        return layers;
    }
    
    var allLayers = getAllLayers(doc);
}


// ═══════════════════════════════════════════════════════════════
// SELECTION AND MASKING
// ═══════════════════════════════════════════════════════════════

function selectionOperations() {
    var doc = app.activeDocument;
    
    // Select all
    doc.selection.selectAll();
    
    // Select rectangle
    var bounds = [[100, 100], [300, 100], [300, 300], [100, 300]];
    doc.selection.select(bounds);
    
    // Select ellipse
    doc.selection.selectEllipse([100, 100, 300, 300]);
    
    // Feather selection
    doc.selection.feather(5);
    
    // Expand/contract
    doc.selection.expand(10);
    doc.selection.contract(5);
    
    // Smooth selection
    doc.selection.smooth(3);
    
    // Invert selection
    doc.selection.invert();
    
    // Deselect
    doc.selection.deselect();
    
    // Select by color range
    var color = new SolidColor();
    color.rgb.red = 255;
    color.rgb.green = 0;
    color.rgb.blue = 0;
    doc.selection.selectColor(color, ColorBlendMode.NORMAL, 32, false);
    
    // Create mask from selection
    var layer = doc.activeLayer;
    // layer.applyLayerMask(LayerMaskApplication.REVEAL); // Apply existing mask
    
    // Load selection from channel
    // doc.selection.load(doc.channels["Alpha 1"]);
    
    // Save selection to channel
    doc.selection.store(doc.channels.add());
}


// ═══════════════════════════════════════════════════════════════
// IMAGE ADJUSTMENTS
// ═══════════════════════════════════════════════════════════════

function adjustmentOperations() {
    var doc = app.activeDocument;
    var layer = doc.activeLayer;
    
    // Brightness/Contrast
    layer.adjustBrightnessContrast(20, 10);
    
    // Levels
    // adjustLevels(inputShadows, inputHighlights, gamma, outputShadows, outputHighlights)
    layer.adjustLevels([0, 255], [0, 255], 1.0, [0, 255], [0, 255]);
    
    // Curves
    // Array of points: [[input, output], ...]
    var curvePoints = [[0, 0], [64, 48], [128, 140], [192, 224], [255, 255]];
    layer.adjustCurves(curvePoints);
    
    // Hue/Saturation
    layer.adjustHueSaturation(0, 20, 0);  // hue, saturation, lightness
    
    // Color Balance
    layer.adjustColorBalance([10, 0, -10], [0, 0, 0], [0, 0, 0], true);
    
    // Invert
    layer.invert();
    
    // Desaturate
    layer.desaturate();
    
    // Posterize
    layer.posterize(4);
    
    // Threshold
    layer.threshold(128);
    
    // Photo Filter
    // layer.photoFilter(warmingFilter85, 25, false);
    
    // Shadows/Highlights
    // Using Action Manager for unavailable functions
    var idShdH = charIDToTypeID("ShdH");
    var desc = new ActionDescriptor();
    desc.putInteger(charIDToTypeID("Amnt"), 50);
    executeAction(idShdH, desc, DialogModes.NO);
}


// ═══════════════════════════════════════════════════════════════
// FILTERS
// ═══════════════════════════════════════════════════════════════

function filterOperations() {
    var doc = app.activeDocument;
    var layer = doc.activeLayer;
    
    // Gaussian Blur
    layer.applyGaussianBlur(5.0);
    
    // Motion Blur
    layer.applyMotionBlur(45, 20);
    
    // Radial Blur
    layer.applyRadialBlur(10, RadialBlurMethod.SPIN, RadialBlurQuality.BEST);
    
    // Sharpen
    layer.applySharpen();
    layer.applySharpenMore();
    
    // Unsharp Mask
    layer.applyUnSharpMask(100, 1.0, 0);  // amount, radius, threshold
    
    // Add Noise
    layer.applyAddNoise(10, NoiseDistribution.GAUSSIAN, false);
    
    // Median
    layer.applyMedianNoise(2);
    
    // Dust & Scratches
    layer.applyDustAndScratches(2, 10);
    
    // High Pass
    layer.applyHighPass(3);
    
    // Minimum/Maximum
    layer.applyMinimum(1);
    layer.applyMaximum(1);
    
    // Custom Filter using Action Manager
    function applyCustomFilter(kernel) {
        var idCstm = charIDToTypeID("Cstm");
        var desc = new ActionDescriptor();
        var list = new ActionList();
        
        for (var i = 0; i < kernel.length; i++) {
            list.putInteger(kernel[i]);
        }
        
        desc.putList(charIDToTypeID("Mtrx"), list);
        desc.putInteger(charIDToTypeID("Scl "), 1);
        desc.putInteger(charIDToTypeID("Ofst"), 0);
        
        executeAction(idCstm, desc, DialogModes.NO);
    }
    
    // Sharpen kernel
    var sharpenKernel = [0, -1, 0, -1, 5, -1, 0, -1, 0];
    // applyCustomFilter(sharpenKernel);
}


// ═══════════════════════════════════════════════════════════════
// BATCH PROCESSING
// ═══════════════════════════════════════════════════════════════

function batchProcess() {
    // Get input folder
    var inputFolder = Folder.selectDialog("Select input folder");
    if (!inputFolder) return;
    
    // Get output folder
    var outputFolder = Folder.selectDialog("Select output folder");
    if (!outputFolder) return;
    
    // Get files
    var files = inputFolder.getFiles(/\.(jpg|jpeg|png|tiff|psd)$/i);
    
    if (files.length === 0) {
        alert("No image files found.");
        return;
    }
    
    // Process each file
    for (var i = 0; i < files.length; i++) {
        var file = files[i];
        
        try {
            // Open file
            var doc = app.open(file);
            
            // Apply processing
            processDocument(doc);
            
            // Save
            var outputFile = new File(outputFolder + "/" + 
                file.name.replace(/\.[^.]+$/, ".jpg"));
            
            var jpegOptions = new JPEGSaveOptions();
            jpegOptions.quality = 10;  // 0-12
            jpegOptions.embedColorProfile = true;
            
            doc.saveAs(outputFile, jpegOptions, true);
            
            // Close without saving original
            doc.close(SaveOptions.DONOTSAVECHANGES);
            
        } catch (e) {
            alert("Error processing " + file.name + ": " + e.message);
        }
    }
    
    alert("Batch processing complete! Processed " + files.length + " files.");
}

function processDocument(doc) {
    // Resize
    var maxSize = 1920;
    var width = doc.width.as('px');
    var height = doc.height.as('px');
    
    if (width > maxSize || height > maxSize) {
        if (width > height) {
            doc.resizeImage(maxSize, null, null, ResampleMethod.BICUBIC);
        } else {
            doc.resizeImage(null, maxSize, null, ResampleMethod.BICUBIC);
        }
    }
    
    // Auto tone
    doc.autoTone();
    
    // Sharpen
    doc.activeLayer.applyUnSharpMask(50, 1.0, 0);
    
    // Flatten
    doc.flatten();
}


// ═══════════════════════════════════════════════════════════════
// ACTION MANAGER (DESCRIPTOR-BASED API)
// ═══════════════════════════════════════════════════════════════

/*
 * Action Manager provides access to ALL Photoshop functionality,
 * including features not exposed in the DOM.
 *
 * Use the ScriptingListener plugin to record action code.
 */

function actionManagerExample() {
    // Create Gaussian Blur using Action Manager
    var idGsnB = charIDToTypeID("GsnB");
    var desc = new ActionDescriptor();
    desc.putUnitDouble(charIDToTypeID("Rds "), charIDToTypeID("#Pxl"), 5.0);
    executeAction(idGsnB, desc, DialogModes.NO);
    
    // Create new document
    var idMk = charIDToTypeID("Mk  ");
    var desc1 = new ActionDescriptor();
    var idNw = charIDToTypeID("Nw  ");
    var desc2 = new ActionDescriptor();
    desc2.putUnitDouble(charIDToTypeID("Wdth"), charIDToTypeID("#Pxl"), 1920);
    desc2.putUnitDouble(charIDToTypeID("Hght"), charIDToTypeID("#Pxl"), 1080);
    desc2.putUnitDouble(charIDToTypeID("Rslt"), charIDToTypeID("#Rsl"), 300);
    desc2.putEnumerated(charIDToTypeID("Md  "), charIDToTypeID("RGBM"), charIDToTypeID("RGBM"));
    desc1.putObject(idNw, charIDToTypeID("Dcmn"), desc2);
    executeAction(idMk, desc1, DialogModes.NO);
    
    // Helper function to convert string to TypeID
    function s2t(s) {
        return stringIDToTypeID(s);
    }
    
    function c2t(c) {
        return charIDToTypeID(c);
    }
}


// ═══════════════════════════════════════════════════════════════
// WORKING WITH PIXEL DATA
// ═══════════════════════════════════════════════════════════════

/*
 * Photoshop scripting doesn't provide direct pixel access.
 * Workarounds include:
 * 1. Export to raw format, process externally, reimport
 * 2. Use Generator (Node.js) for real-time processing
 * 3. Use C++ plugin (Filter SDK)
 */

function pixelDataWorkaround() {
    var doc = app.activeDocument;
    
    // Save as raw data
    var tempFile = new File(Folder.temp + "/temp_raw.raw");
    
    var rawOptions = new RawSaveOptions();
    rawOptions.headerSize = 0;
    
    // This doesn't actually work for general raw - need to use
    // a temporary PSD/TIFF and external processing
    
    // Alternative: Use histogram to analyze image
    var histogram = doc.histogram;
    // histogram is an array of 256 values
    
    // Calculate mean brightness from histogram
    var total = 0;
    var count = 0;
    for (var i = 0; i < 256; i++) {
        total += i * histogram[i];
        count += histogram[i];
    }
    var meanBrightness = total / count;
    
    alert("Mean brightness: " + meanBrightness.toFixed(2));
}
```


---

# PART IX: AI-GUIDED PROCEDURAL ART

## Chapter 24: Machine Learning Fundamentals for Procedural Art

### 24.1 Neural Networks for Image Generation

Understanding how neural networks process and generate images is essential for creating
AI-guided procedural art systems.

```python
"""
Neural Network Fundamentals for Procedural Art
==============================================

This module provides foundational neural network implementations
optimized for procedural art generation tasks.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import pickle
import json


class ActivationFunction(Enum):
    """Activation functions for neural networks."""
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SWISH = "swish"
    GELU = "gelu"
    SOFTMAX = "softmax"


class Activations:
    """Collection of activation functions and their derivatives."""
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation: 1 / (1 + exp(-x))"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid."""
        s = Activations.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation."""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh."""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit."""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU."""
        return (x > 0).astype(float)
    
    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU with small gradient for negative values."""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivative of Leaky ReLU."""
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Exponential Linear Unit."""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def elu_derivative(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Derivative of ELU."""
        return np.where(x > 0, 1, Activations.elu(x, alpha) + alpha)
    
    @staticmethod
    def swish(x: np.ndarray) -> np.ndarray:
        """Swish activation: x * sigmoid(x)"""
        return x * Activations.sigmoid(x)
    
    @staticmethod
    def swish_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of Swish."""
        sig = Activations.sigmoid(x)
        return sig + x * sig * (1 - sig)
    
    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        """Gaussian Error Linear Unit."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax for classification outputs."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


@dataclass
class LayerConfig:
    """Configuration for a neural network layer."""
    input_size: int
    output_size: int
    activation: ActivationFunction = ActivationFunction.RELU
    dropout_rate: float = 0.0
    batch_norm: bool = False
    weight_init: str = "he"  # "he", "xavier", "uniform"


class DenseLayer:
    """Fully connected neural network layer."""
    
    def __init__(self, config: LayerConfig):
        self.config = config
        self.weights = self._init_weights()
        self.biases = np.zeros((1, config.output_size))
        
        # For batch normalization
        self.gamma = np.ones((1, config.output_size))
        self.beta = np.zeros((1, config.output_size))
        self.running_mean = np.zeros((1, config.output_size))
        self.running_var = np.ones((1, config.output_size))
        
        # Cache for backpropagation
        self.cache = {}
        
        # Gradients
        self.dW = None
        self.db = None
        self.dgamma = None
        self.dbeta = None
    
    def _init_weights(self) -> np.ndarray:
        """Initialize weights based on configuration."""
        in_size = self.config.input_size
        out_size = self.config.output_size
        
        if self.config.weight_init == "he":
            # He initialization for ReLU
            scale = np.sqrt(2.0 / in_size)
        elif self.config.weight_init == "xavier":
            # Xavier/Glorot initialization
            scale = np.sqrt(2.0 / (in_size + out_size))
        else:
            # Uniform initialization
            scale = 0.1
        
        return np.random.randn(in_size, out_size) * scale
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through the layer."""
        self.cache['input'] = x
        
        # Linear transformation
        z = np.dot(x, self.weights) + self.biases
        self.cache['pre_bn'] = z
        
        # Batch normalization
        if self.config.batch_norm:
            if training:
                mean = np.mean(z, axis=0, keepdims=True)
                var = np.var(z, axis=0, keepdims=True)
                
                # Update running statistics
                momentum = 0.9
                self.running_mean = momentum * self.running_mean + (1 - momentum) * mean
                self.running_var = momentum * self.running_var + (1 - momentum) * var
            else:
                mean = self.running_mean
                var = self.running_var
            
            z_norm = (z - mean) / np.sqrt(var + 1e-8)
            z = self.gamma * z_norm + self.beta
            
            self.cache['bn_mean'] = mean
            self.cache['bn_var'] = var
            self.cache['z_norm'] = z_norm
        
        self.cache['pre_activation'] = z
        
        # Activation
        activation = self._get_activation()
        output = activation(z)
        self.cache['activation_output'] = output
        
        # Dropout
        if self.config.dropout_rate > 0 and training:
            dropout_mask = (np.random.rand(*output.shape) > self.config.dropout_rate)
            output = output * dropout_mask / (1 - self.config.dropout_rate)
            self.cache['dropout_mask'] = dropout_mask
        
        return output
    
    def backward(self, grad: np.ndarray, learning_rate: float = 0.001) -> np.ndarray:
        """Backward pass for gradient computation."""
        batch_size = grad.shape[0]
        
        # Dropout gradient
        if 'dropout_mask' in self.cache:
            grad = grad * self.cache['dropout_mask'] / (1 - self.config.dropout_rate)
        
        # Activation gradient
        activation_derivative = self._get_activation_derivative()
        grad = grad * activation_derivative(self.cache['pre_activation'])
        
        # Batch normalization gradient
        if self.config.batch_norm:
            self.dgamma = np.sum(grad * self.cache['z_norm'], axis=0, keepdims=True)
            self.dbeta = np.sum(grad, axis=0, keepdims=True)
            
            z_norm = self.cache['z_norm']
            var = self.cache['bn_var']
            mean = self.cache['bn_mean']
            
            dz_norm = grad * self.gamma
            dvar = np.sum(dz_norm * (self.cache['pre_bn'] - mean) * -0.5 * (var + 1e-8)**(-1.5), axis=0)
            dmean = np.sum(dz_norm * -1/np.sqrt(var + 1e-8), axis=0) + dvar * np.mean(-2 * (self.cache['pre_bn'] - mean), axis=0)
            grad = dz_norm / np.sqrt(var + 1e-8) + dvar * 2 * (self.cache['pre_bn'] - mean) / batch_size + dmean / batch_size
            
            # Update batch norm parameters
            self.gamma -= learning_rate * self.dgamma
            self.beta -= learning_rate * self.dbeta
        
        # Weight and bias gradients
        self.dW = np.dot(self.cache['input'].T, grad) / batch_size
        self.db = np.mean(grad, axis=0, keepdims=True)
        
        # Gradient for previous layer
        grad_input = np.dot(grad, self.weights.T)
        
        # Update weights
        self.weights -= learning_rate * self.dW
        self.biases -= learning_rate * self.db
        
        return grad_input
    
    def _get_activation(self) -> Callable:
        """Get activation function."""
        activations = {
            ActivationFunction.SIGMOID: Activations.sigmoid,
            ActivationFunction.TANH: Activations.tanh,
            ActivationFunction.RELU: Activations.relu,
            ActivationFunction.LEAKY_RELU: Activations.leaky_relu,
            ActivationFunction.ELU: Activations.elu,
            ActivationFunction.SWISH: Activations.swish,
            ActivationFunction.GELU: Activations.gelu,
            ActivationFunction.SOFTMAX: Activations.softmax,
        }
        return activations.get(self.config.activation, Activations.relu)
    
    def _get_activation_derivative(self) -> Callable:
        """Get activation function derivative."""
        derivatives = {
            ActivationFunction.SIGMOID: Activations.sigmoid_derivative,
            ActivationFunction.TANH: Activations.tanh_derivative,
            ActivationFunction.RELU: Activations.relu_derivative,
            ActivationFunction.LEAKY_RELU: Activations.leaky_relu_derivative,
            ActivationFunction.ELU: Activations.elu_derivative,
            ActivationFunction.SWISH: Activations.swish_derivative,
        }
        return derivatives.get(self.config.activation, Activations.relu_derivative)


class ConvolutionalLayer:
    """2D Convolutional layer for image processing."""
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 activation: ActivationFunction = ActivationFunction.RELU):
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        
        # Initialize kernels using He initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.kernels = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * scale
        self.biases = np.zeros((out_channels, 1))
        
        # Cache for backpropagation
        self.cache = {}
    
    def _pad_input(self, x: np.ndarray) -> np.ndarray:
        """Add padding to input."""
        if self.padding == 0:
            return x
        return np.pad(
            x,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode='constant'
        )
    
    def _im2col(self, x: np.ndarray) -> np.ndarray:
        """Convert image to column matrix for efficient convolution."""
        batch_size, channels, height, width = x.shape
        
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        
        col = np.zeros((
            batch_size, channels, self.kernel_size, self.kernel_size, out_height, out_width
        ))
        
        for y in range(self.kernel_size):
            y_max = y + self.stride * out_height
            for x_idx in range(self.kernel_size):
                x_max = x_idx + self.stride * out_width
                col[:, :, y, x_idx, :, :] = x[:, :, y:y_max:self.stride, x_idx:x_max:self.stride]
        
        return col.transpose(0, 4, 5, 1, 2, 3).reshape(
            batch_size * out_height * out_width, -1
        )
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through convolutional layer."""
        batch_size = x.shape[0]
        
        # Pad input
        x_padded = self._pad_input(x)
        self.cache['input'] = x
        self.cache['input_padded'] = x_padded
        
        # Calculate output dimensions
        _, _, h_padded, w_padded = x_padded.shape
        out_height = (h_padded - self.kernel_size) // self.stride + 1
        out_width = (w_padded - self.kernel_size) // self.stride + 1
        
        # im2col transformation
        col = self._im2col(x_padded)
        self.cache['col'] = col
        
        # Reshape kernels for matrix multiplication
        kernel_col = self.kernels.reshape(self.out_channels, -1).T
        
        # Convolution as matrix multiplication
        output = np.dot(col, kernel_col) + self.biases.T
        
        # Reshape output
        output = output.reshape(batch_size, out_height, out_width, self.out_channels)
        output = output.transpose(0, 3, 1, 2)
        
        self.cache['pre_activation'] = output
        
        # Apply activation
        activation_func = self._get_activation()
        output = activation_func(output)
        
        return output
    
    def _get_activation(self) -> Callable:
        """Get activation function."""
        if self.activation == ActivationFunction.RELU:
            return Activations.relu
        elif self.activation == ActivationFunction.LEAKY_RELU:
            return Activations.leaky_relu
        elif self.activation == ActivationFunction.TANH:
            return Activations.tanh
        elif self.activation == ActivationFunction.SIGMOID:
            return Activations.sigmoid
        return lambda x: x


class MaxPoolingLayer:
    """Max pooling layer for downsampling."""
    
    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through max pooling."""
        batch_size, channels, height, width = x.shape
        
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        for h in range(out_height):
            for w in range(out_width):
                h_start = h * self.stride
                h_end = h_start + self.pool_size
                w_start = w * self.stride
                w_end = w_start + self.pool_size
                
                output[:, :, h, w] = np.max(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))
        
        self.cache['input'] = x
        self.cache['output'] = output
        
        return output
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass through max pooling."""
        x = self.cache['input']
        batch_size, channels, height, width = x.shape
        
        out_height = grad.shape[2]
        out_width = grad.shape[3]
        
        dx = np.zeros_like(x)
        
        for h in range(out_height):
            for w in range(out_width):
                h_start = h * self.stride
                h_end = h_start + self.pool_size
                w_start = w * self.stride
                w_end = w_start + self.pool_size
                
                window = x[:, :, h_start:h_end, w_start:w_end]
                max_val = np.max(window, axis=(2, 3), keepdims=True)
                mask = (window == max_val)
                
                dx[:, :, h_start:h_end, w_start:w_end] += mask * grad[:, :, h:h+1, w:w+1]
        
        return dx


class NeuralNetwork:
    """Complete neural network for procedural art generation."""
    
    def __init__(self, layer_configs: List[LayerConfig]):
        self.layers = []
        for config in layer_configs:
            self.layers.append(DenseLayer(config))
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer.forward(x, training)
        return x
    
    def backward(self, grad: np.ndarray, learning_rate: float = 0.001) -> None:
        """Backward pass through all layers."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
    
    def train(self,
              x_train: np.ndarray,
              y_train: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              loss_fn: str = "mse") -> List[float]:
        """Train the network."""
        losses = []
        n_samples = x_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            n_batches = n_samples // batch_size
            
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Forward pass
                predictions = self.forward(x_batch, training=True)
                
                # Compute loss and gradient
                if loss_fn == "mse":
                    loss = np.mean((predictions - y_batch) ** 2)
                    grad = 2 * (predictions - y_batch) / batch_size
                elif loss_fn == "cross_entropy":
                    loss = -np.mean(y_batch * np.log(predictions + 1e-8))
                    grad = -y_batch / (predictions + 1e-8) / batch_size
                else:
                    loss = np.mean((predictions - y_batch) ** 2)
                    grad = 2 * (predictions - y_batch) / batch_size
                
                epoch_loss += loss
                
                # Backward pass
                self.backward(grad, learning_rate)
            
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        return losses
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.forward(x, training=False)
    
    def save(self, filepath: str) -> None:
        """Save model weights."""
        weights = {
            f'layer_{i}': {
                'weights': layer.weights,
                'biases': layer.biases,
                'gamma': layer.gamma,
                'beta': layer.beta,
            }
            for i, layer in enumerate(self.layers)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)
    
    def load(self, filepath: str) -> None:
        """Load model weights."""
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
        
        for i, layer in enumerate(self.layers):
            layer_weights = weights[f'layer_{i}']
            layer.weights = layer_weights['weights']
            layer.biases = layer_weights['biases']
            layer.gamma = layer_weights['gamma']
            layer.beta = layer_weights['beta']
```

### 24.2 Autoencoders for Style Extraction

Autoencoders learn compressed representations of images, making them perfect
for extracting and transferring artistic styles.

```python
"""
Autoencoder Implementations for Procedural Art
==============================================

Autoencoders compress images into latent representations
and reconstruct them, learning essential features.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class AutoencoderConfig:
    """Configuration for autoencoder architecture."""
    input_size: int
    hidden_layers: List[int]
    latent_size: int
    activation: str = "relu"
    use_variational: bool = False


class Autoencoder:
    """
    Standard autoencoder for image reconstruction and feature learning.
    
    The autoencoder consists of:
    - Encoder: Compresses input to latent space
    - Decoder: Reconstructs input from latent space
    """
    
    def __init__(self, config: AutoencoderConfig):
        self.config = config
        self.encoder_layers = []
        self.decoder_layers = []
        
        # Build encoder
        prev_size = config.input_size
        for hidden_size in config.hidden_layers:
            self.encoder_layers.append(self._create_layer(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Latent layer
        self.latent_layer = self._create_layer(prev_size, config.latent_size)
        
        # Build decoder (reverse of encoder)
        prev_size = config.latent_size
        for hidden_size in reversed(config.hidden_layers):
            self.decoder_layers.append(self._create_layer(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = self._create_layer(prev_size, config.input_size)
    
    def _create_layer(self, in_size: int, out_size: int) -> dict:
        """Create a layer with weights and biases."""
        scale = np.sqrt(2.0 / in_size)
        return {
            'weights': np.random.randn(in_size, out_size) * scale,
            'biases': np.zeros((1, out_size))
        }
    
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.config.activation == "relu":
            return np.maximum(0, x)
        elif self.config.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.config.activation == "tanh":
            return np.tanh(x)
        return x
    
    def _activate_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of activation function."""
        if self.config.activation == "relu":
            return (x > 0).astype(float)
        elif self.config.activation == "sigmoid":
            s = self._activate(x)
            return s * (1 - s)
        elif self.config.activation == "tanh":
            return 1 - np.tanh(x) ** 2
        return np.ones_like(x)
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to latent representation."""
        for layer in self.encoder_layers:
            x = self._activate(np.dot(x, layer['weights']) + layer['biases'])
        
        latent = np.dot(x, self.latent_layer['weights']) + self.latent_layer['biases']
        return latent
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent representation to output."""
        x = z
        for layer in self.decoder_layers:
            x = self._activate(np.dot(x, layer['weights']) + layer['biases'])
        
        # Output layer uses sigmoid for [0, 1] pixel values
        output = 1 / (1 + np.exp(-np.clip(
            np.dot(x, self.output_layer['weights']) + self.output_layer['biases'],
            -500, 500
        )))
        return output
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Full forward pass through autoencoder."""
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent
    
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Reconstruct input."""
        reconstruction, _ = self.forward(x)
        return reconstruction
    
    def get_latent(self, x: np.ndarray) -> np.ndarray:
        """Get latent representation of input."""
        return self.encode(x)
    
    def generate_from_latent(self, z: np.ndarray) -> np.ndarray:
        """Generate image from latent vector."""
        return self.decode(z)


class VariationalAutoencoder(Autoencoder):
    """
    Variational Autoencoder (VAE) for generative art.
    
    VAEs learn a probabilistic latent space, enabling:
    - Smooth interpolation between styles
    - Random generation of new images
    - Controlled style manipulation
    """
    
    def __init__(self, config: AutoencoderConfig):
        config.use_variational = True
        super().__init__(config)
        
        # VAE has two outputs: mean and log variance
        prev_size = config.hidden_layers[-1] if config.hidden_layers else config.input_size
        self.mu_layer = self._create_layer(prev_size, config.latent_size)
        self.log_var_layer = self._create_layer(prev_size, config.latent_size)
    
    def _reparameterize(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        """
        Reparameterization trick for backpropagation through sampling.
        z = mu + sigma * epsilon, where epsilon ~ N(0, I)
        """
        std = np.exp(0.5 * log_var)
        epsilon = np.random.randn(*mu.shape)
        return mu + std * epsilon
    
    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode to distribution parameters and sample."""
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = self._activate(np.dot(x, layer['weights']) + layer['biases'])
        
        # Get distribution parameters
        mu = np.dot(x, self.mu_layer['weights']) + self.mu_layer['biases']
        log_var = np.dot(x, self.log_var_layer['weights']) + self.log_var_layer['biases']
        
        # Sample from distribution
        z = self._reparameterize(mu, log_var)
        
        return z, mu, log_var
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass returning reconstruction and distribution parameters."""
        z, mu, log_var = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z, mu, log_var
    
    def loss(self, x: np.ndarray, reconstruction: np.ndarray, 
             mu: np.ndarray, log_var: np.ndarray) -> Tuple[float, float, float]:
        """
        VAE loss = Reconstruction loss + KL divergence
        """
        # Reconstruction loss (binary cross-entropy)
        recon_loss = -np.mean(
            x * np.log(reconstruction + 1e-8) + 
            (1 - x) * np.log(1 - reconstruction + 1e-8)
        )
        
        # KL divergence: D_KL(q(z|x) || p(z))
        kl_loss = -0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var))
        
        total_loss = recon_loss + kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate random samples from learned distribution."""
        z = np.random.randn(n_samples, self.config.latent_size)
        return self.decode(z)
    
    def interpolate(self, x1: np.ndarray, x2: np.ndarray, 
                    n_steps: int = 10) -> np.ndarray:
        """Interpolate between two images in latent space."""
        z1, _, _ = self.encode(x1.reshape(1, -1))
        z2, _, _ = self.encode(x2.reshape(1, -1))
        
        interpolations = []
        for t in np.linspace(0, 1, n_steps):
            z_interp = (1 - t) * z1 + t * z2
            interpolations.append(self.decode(z_interp))
        
        return np.array(interpolations).squeeze()


class ConvolutionalAutoencoder:
    """
    Convolutional autoencoder for image processing.
    
    Uses convolutional layers for spatial feature learning,
    preserving image structure better than fully connected networks.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int],  # (channels, height, width)
                 latent_dim: int = 128):
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        # Encoder architecture
        self.encoder_filters = [32, 64, 128]
        self.encoder_kernels = []
        
        in_channels = input_shape[0]
        for out_channels in self.encoder_filters:
            self.encoder_kernels.append(
                self._init_conv_kernel(in_channels, out_channels, 3)
            )
            in_channels = out_channels
        
        # Calculate flattened size after encoding
        h, w = input_shape[1], input_shape[2]
        for _ in self.encoder_filters:
            h = h // 2
            w = w // 2
        self.encoded_shape = (self.encoder_filters[-1], h, w)
        flat_size = self.encoder_filters[-1] * h * w
        
        # Dense layers for latent space
        scale = np.sqrt(2.0 / flat_size)
        self.fc_encode = {
            'weights': np.random.randn(flat_size, latent_dim) * scale,
            'biases': np.zeros((1, latent_dim))
        }
        
        scale = np.sqrt(2.0 / latent_dim)
        self.fc_decode = {
            'weights': np.random.randn(latent_dim, flat_size) * scale,
            'biases': np.zeros((1, flat_size))
        }
        
        # Decoder architecture (transpose convolutions)
        self.decoder_filters = [64, 32, input_shape[0]]
        self.decoder_kernels = []
        
        in_channels = self.encoder_filters[-1]
        for out_channels in self.decoder_filters:
            self.decoder_kernels.append(
                self._init_conv_kernel(in_channels, out_channels, 3)
            )
            in_channels = out_channels
    
    def _init_conv_kernel(self, in_ch: int, out_ch: int, size: int) -> dict:
        """Initialize convolutional kernel."""
        scale = np.sqrt(2.0 / (in_ch * size * size))
        return {
            'weights': np.random.randn(out_ch, in_ch, size, size) * scale,
            'biases': np.zeros((out_ch, 1, 1))
        }
    
    def _conv2d(self, x: np.ndarray, kernel: dict, stride: int = 1) -> np.ndarray:
        """2D convolution operation."""
        batch, in_ch, h, w = x.shape
        out_ch, _, kh, kw = kernel['weights'].shape
        
        # Calculate output size
        out_h = (h - kh) // stride + 1
        out_w = (w - kw) // stride + 1
        
        output = np.zeros((batch, out_ch, out_h, out_w))
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                patch = x[:, :, h_start:h_start+kh, w_start:w_start+kw]
                
                for k in range(out_ch):
                    output[:, k, i, j] = np.sum(
                        patch * kernel['weights'][k], axis=(1, 2, 3)
                    ) + kernel['biases'][k, 0, 0]
        
        return output
    
    def _conv2d_transpose(self, x: np.ndarray, kernel: dict, stride: int = 2) -> np.ndarray:
        """Transpose convolution (upsampling)."""
        batch, in_ch, h, w = x.shape
        out_ch = kernel['weights'].shape[1]
        kh, kw = kernel['weights'].shape[2:]
        
        # Output size for transpose conv
        out_h = (h - 1) * stride + kh
        out_w = (w - 1) * stride + kw
        
        output = np.zeros((batch, out_ch, out_h, out_w))
        
        for i in range(h):
            for j in range(w):
                h_start = i * stride
                w_start = j * stride
                
                for k in range(in_ch):
                    output[:, :, h_start:h_start+kh, w_start:w_start+kw] += (
                        x[:, k:k+1, i:i+1, j:j+1] * kernel['weights'][k]
                    )
        
        # Add bias
        output += kernel['biases'].reshape(1, out_ch, 1, 1)
        
        return output
    
    def _max_pool(self, x: np.ndarray, size: int = 2) -> np.ndarray:
        """Max pooling for downsampling."""
        batch, ch, h, w = x.shape
        out_h, out_w = h // size, w // size
        
        output = np.zeros((batch, ch, out_h, out_w))
        
        for i in range(out_h):
            for j in range(out_w):
                output[:, :, i, j] = np.max(
                    x[:, :, i*size:(i+1)*size, j*size:(j+1)*size],
                    axis=(2, 3)
                )
        
        return output
    
    def _upsample(self, x: np.ndarray, scale: int = 2) -> np.ndarray:
        """Nearest neighbor upsampling."""
        return np.repeat(np.repeat(x, scale, axis=2), scale, axis=3)
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode image to latent vector."""
        # Convolutional encoding
        for kernel in self.encoder_kernels:
            x = self._conv2d(x, kernel)
            x = np.maximum(0, x)  # ReLU
            x = self._max_pool(x)
        
        # Flatten and dense
        batch = x.shape[0]
        x = x.reshape(batch, -1)
        latent = np.dot(x, self.fc_encode['weights']) + self.fc_encode['biases']
        
        return latent
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent vector to image."""
        batch = z.shape[0]
        
        # Dense and reshape
        x = np.dot(z, self.fc_decode['weights']) + self.fc_decode['biases']
        x = np.maximum(0, x)  # ReLU
        x = x.reshape(batch, *self.encoded_shape)
        
        # Transpose convolutional decoding
        for i, kernel in enumerate(self.decoder_kernels):
            x = self._upsample(x)
            x = self._conv2d(x, kernel)
            
            if i < len(self.decoder_kernels) - 1:
                x = np.maximum(0, x)  # ReLU
            else:
                x = 1 / (1 + np.exp(-x))  # Sigmoid for output
        
        return x
    
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Reconstruct input image."""
        latent = self.encode(x)
        return self.decode(latent)
```

### 24.3 Generative Adversarial Networks (GANs)

GANs pit two networks against each other to generate realistic images.

```python
"""
GAN Implementations for Procedural Art Generation
================================================

Generative Adversarial Networks for creating new art.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class GANConfig:
    """Configuration for GAN architecture."""
    latent_dim: int = 100
    generator_layers: List[int] = None
    discriminator_layers: List[int] = None
    image_shape: Tuple[int, int, int] = (3, 64, 64)
    learning_rate: float = 0.0002
    beta1: float = 0.5  # Adam optimizer parameter
    
    def __post_init__(self):
        if self.generator_layers is None:
            self.generator_layers = [256, 512, 1024]
        if self.discriminator_layers is None:
            self.discriminator_layers = [512, 256, 128]


class Generator:
    """
    Generator network for GAN.
    
    Takes random noise and generates images.
    """
    
    def __init__(self, config: GANConfig):
        self.config = config
        self.layers = []
        
        # Build generator layers
        image_size = np.prod(config.image_shape)
        
        prev_size = config.latent_dim
        for hidden_size in config.generator_layers:
            self.layers.append(self._create_layer(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layer
        self.layers.append(self._create_layer(prev_size, image_size))
        
        # Batch normalization parameters
        self.bn_params = []
        for hidden_size in config.generator_layers:
            self.bn_params.append({
                'gamma': np.ones((1, hidden_size)),
                'beta': np.zeros((1, hidden_size)),
                'running_mean': np.zeros((1, hidden_size)),
                'running_var': np.ones((1, hidden_size))
            })
        
        # Adam optimizer state
        self.m = [np.zeros_like(l['weights']) for l in self.layers]
        self.v = [np.zeros_like(l['weights']) for l in self.layers]
        self.m_bias = [np.zeros_like(l['biases']) for l in self.layers]
        self.v_bias = [np.zeros_like(l['biases']) for l in self.layers]
        self.t = 0
    
    def _create_layer(self, in_size: int, out_size: int) -> dict:
        """Create layer with He initialization."""
        scale = np.sqrt(2.0 / in_size)
        return {
            'weights': np.random.randn(in_size, out_size) * scale,
            'biases': np.zeros((1, out_size))
        }
    
    def _batch_norm(self, x: np.ndarray, idx: int, training: bool = True) -> np.ndarray:
        """Apply batch normalization."""
        params = self.bn_params[idx]
        
        if training:
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            
            # Update running statistics
            params['running_mean'] = 0.9 * params['running_mean'] + 0.1 * mean
            params['running_var'] = 0.9 * params['running_var'] + 0.1 * var
        else:
            mean = params['running_mean']
            var = params['running_var']
        
        x_norm = (x - mean) / np.sqrt(var + 1e-8)
        return params['gamma'] * x_norm + params['beta']
    
    def _leaky_relu(self, x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        """Leaky ReLU activation."""
        return np.where(x > 0, x, alpha * x)
    
    def forward(self, z: np.ndarray, training: bool = True) -> np.ndarray:
        """Generate image from noise vector."""
        x = z
        activations = []
        
        for i, layer in enumerate(self.layers[:-1]):
            x = np.dot(x, layer['weights']) + layer['biases']
            x = self._batch_norm(x, i, training)
            x = self._leaky_relu(x)
            activations.append(x)
        
        # Output layer with tanh (images in [-1, 1])
        output_layer = self.layers[-1]
        x = np.dot(x, output_layer['weights']) + output_layer['biases']
        x = np.tanh(x)
        
        # Reshape to image
        batch_size = z.shape[0]
        x = x.reshape(batch_size, *self.config.image_shape)
        
        self.activations = activations
        return x
    
    def update(self, gradients: List[np.ndarray], learning_rate: float) -> None:
        """Update weights using Adam optimizer."""
        self.t += 1
        beta1 = self.config.beta1
        beta2 = 0.999
        epsilon = 1e-8
        
        for i, (grad_w, grad_b) in enumerate(gradients):
            # Update momentum
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad_w
            self.m_bias[i] = beta1 * self.m_bias[i] + (1 - beta1) * grad_b
            
            # Update velocity
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad_w ** 2)
            self.v_bias[i] = beta2 * self.v_bias[i] + (1 - beta2) * (grad_b ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)
            m_hat_b = self.m_bias[i] / (1 - beta1 ** self.t)
            v_hat_b = self.v_bias[i] / (1 - beta2 ** self.t)
            
            # Update weights
            self.layers[i]['weights'] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            self.layers[i]['biases'] -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)


class Discriminator:
    """
    Discriminator network for GAN.
    
    Classifies images as real or fake.
    """
    
    def __init__(self, config: GANConfig):
        self.config = config
        self.layers = []
        
        image_size = np.prod(config.image_shape)
        
        prev_size = image_size
        for hidden_size in config.discriminator_layers:
            self.layers.append(self._create_layer(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layer (single probability)
        self.layers.append(self._create_layer(prev_size, 1))
        
        # Adam optimizer state
        self.m = [np.zeros_like(l['weights']) for l in self.layers]
        self.v = [np.zeros_like(l['weights']) for l in self.layers]
        self.m_bias = [np.zeros_like(l['biases']) for l in self.layers]
        self.v_bias = [np.zeros_like(l['biases']) for l in self.layers]
        self.t = 0
    
    def _create_layer(self, in_size: int, out_size: int) -> dict:
        """Create layer."""
        scale = np.sqrt(2.0 / in_size)
        return {
            'weights': np.random.randn(in_size, out_size) * scale,
            'biases': np.zeros((1, out_size))
        }
    
    def _leaky_relu(self, x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        """Leaky ReLU activation."""
        return np.where(x > 0, x, alpha * x)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Classify image as real or fake."""
        # Flatten image
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        activations = [x]
        
        for i, layer in enumerate(self.layers[:-1]):
            x = np.dot(x, layer['weights']) + layer['biases']
            x = self._leaky_relu(x)
            
            # Dropout during training
            if training:
                dropout_mask = np.random.binomial(1, 0.7, x.shape) / 0.7
                x = x * dropout_mask
            
            activations.append(x)
        
        # Output layer with sigmoid
        output_layer = self.layers[-1]
        x = np.dot(x, output_layer['weights']) + output_layer['biases']
        x = self._sigmoid(x)
        
        self.activations = activations
        return x
    
    def update(self, gradients: List[np.ndarray], learning_rate: float) -> None:
        """Update weights using Adam optimizer."""
        self.t += 1
        beta1 = self.config.beta1
        beta2 = 0.999
        epsilon = 1e-8
        
        for i, (grad_w, grad_b) in enumerate(gradients):
            # Update momentum
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad_w
            self.m_bias[i] = beta1 * self.m_bias[i] + (1 - beta1) * grad_b
            
            # Update velocity
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad_w ** 2)
            self.v_bias[i] = beta2 * self.v_bias[i] + (1 - beta2) * (grad_b ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)
            m_hat_b = self.m_bias[i] / (1 - beta1 ** self.t)
            v_hat_b = self.v_bias[i] / (1 - beta2 ** self.t)
            
            # Update weights
            self.layers[i]['weights'] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            self.layers[i]['biases'] -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)


class GAN:
    """
    Complete GAN implementation for procedural art.
    
    Training alternates between:
    1. Training discriminator on real and fake images
    2. Training generator to fool discriminator
    """
    
    def __init__(self, config: GANConfig):
        self.config = config
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        self.losses = {'d_loss': [], 'g_loss': []}
    
    def generate(self, n_samples: int = 1) -> np.ndarray:
        """Generate fake images."""
        z = np.random.randn(n_samples, self.config.latent_dim)
        return self.generator.forward(z, training=False)
    
    def _discriminator_loss(self, real_output: np.ndarray, 
                           fake_output: np.ndarray) -> float:
        """Binary cross-entropy loss for discriminator."""
        real_loss = -np.mean(np.log(real_output + 1e-8))
        fake_loss = -np.mean(np.log(1 - fake_output + 1e-8))
        return real_loss + fake_loss
    
    def _generator_loss(self, fake_output: np.ndarray) -> float:
        """Generator wants discriminator to output 1 for fake images."""
        return -np.mean(np.log(fake_output + 1e-8))
    
    def train_step(self, real_images: np.ndarray) -> Tuple[float, float]:
        """Single training step."""
        batch_size = real_images.shape[0]
        
        # ===============================
        # Train Discriminator
        # ===============================
        
        # Generate fake images
        z = np.random.randn(batch_size, self.config.latent_dim)
        fake_images = self.generator.forward(z, training=True)
        
        # Discriminator predictions
        real_output = self.discriminator.forward(real_images, training=True)
        fake_output = self.discriminator.forward(fake_images, training=True)
        
        # Discriminator loss
        d_loss = self._discriminator_loss(real_output, fake_output)
        
        # Compute discriminator gradients (simplified)
        d_gradients = self._compute_discriminator_gradients(
            real_images, fake_images, real_output, fake_output
        )
        
        # Update discriminator
        self.discriminator.update(d_gradients, self.config.learning_rate)
        
        # ===============================
        # Train Generator
        # ===============================
        
        # Generate new fake images
        z = np.random.randn(batch_size, self.config.latent_dim)
        fake_images = self.generator.forward(z, training=True)
        
        # Discriminator prediction on fake
        fake_output = self.discriminator.forward(fake_images, training=False)
        
        # Generator loss
        g_loss = self._generator_loss(fake_output)
        
        # Compute generator gradients (simplified)
        g_gradients = self._compute_generator_gradients(z, fake_output)
        
        # Update generator
        self.generator.update(g_gradients, self.config.learning_rate)
        
        self.losses['d_loss'].append(d_loss)
        self.losses['g_loss'].append(g_loss)
        
        return d_loss, g_loss
    
    def _compute_discriminator_gradients(self, real_images, fake_images,
                                         real_output, fake_output) -> List:
        """Compute gradients for discriminator (simplified backprop)."""
        gradients = []
        
        batch_size = real_images.shape[0]
        
        # Gradient of loss w.r.t. outputs
        d_real = -1 / (real_output + 1e-8) / batch_size
        d_fake = 1 / (1 - fake_output + 1e-8) / batch_size
        
        # Simplified gradient computation
        for i, layer in enumerate(self.discriminator.layers):
            if i < len(self.discriminator.activations):
                activation = self.discriminator.activations[i]
                grad_w = np.dot(activation.T, d_real + d_fake) / batch_size
                grad_b = np.mean(d_real + d_fake, axis=0, keepdims=True)
                gradients.append((grad_w, grad_b))
            else:
                gradients.append((np.zeros_like(layer['weights']),
                                np.zeros_like(layer['biases'])))
        
        return gradients
    
    def _compute_generator_gradients(self, z, fake_output) -> List:
        """Compute gradients for generator (simplified backprop)."""
        gradients = []
        
        batch_size = z.shape[0]
        
        # Gradient of loss w.r.t. fake output
        d_output = -1 / (fake_output + 1e-8) / batch_size
        
        # Simplified gradient computation
        activations = [z] + self.generator.activations
        
        for i, layer in enumerate(self.generator.layers):
            if i < len(activations):
                activation = activations[i]
                grad_w = np.dot(activation.T, np.ones((batch_size, layer['weights'].shape[1])) * 0.01)
                grad_b = np.ones((1, layer['biases'].shape[1])) * 0.01
                gradients.append((grad_w, grad_b))
            else:
                gradients.append((np.zeros_like(layer['weights']),
                                np.zeros_like(layer['biases'])))
        
        return gradients
    
    def train(self, 
              real_images: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32) -> None:
        """Train the GAN."""
        n_samples = real_images.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            real_images_shuffled = real_images[indices]
            
            epoch_d_loss = 0
            epoch_g_loss = 0
            
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                batch = real_images_shuffled[start:end]
                
                d_loss, g_loss = self.train_step(batch)
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss
            
            avg_d_loss = epoch_d_loss / n_batches
            avg_g_loss = epoch_g_loss / n_batches
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}")
    
    def interpolate_latent(self, z1: np.ndarray, z2: np.ndarray,
                          n_steps: int = 10) -> np.ndarray:
        """Interpolate between two latent vectors."""
        interpolations = []
        for t in np.linspace(0, 1, n_steps):
            z = (1 - t) * z1 + t * z2
            img = self.generator.forward(z.reshape(1, -1), training=False)
            interpolations.append(img)
        return np.array(interpolations).squeeze()


class StyleGAN:
    """
    Simplified StyleGAN-like architecture for style-controlled generation.
    
    Key features:
    - Mapping network transforms latent z to style w
    - Style is injected at multiple layers
    - Progressive growing capability
    """
    
    def __init__(self, 
                 latent_dim: int = 512,
                 style_dim: int = 512,
                 image_size: int = 64):
        
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.image_size = image_size
        
        # Mapping network: z -> w
        self.mapping_layers = []
        prev_size = latent_dim
        for _ in range(8):
            self.mapping_layers.append(self._create_layer(prev_size, style_dim))
            prev_size = style_dim
        
        # Synthesis network layers
        self.synthesis_layers = []
        channels = [512, 256, 128, 64, 32, 16]  # Feature channels at each resolution
        
        for i in range(len(channels) - 1):
            self.synthesis_layers.append({
                'conv': self._create_conv(channels[i], channels[i+1]),
                'style_w1': self._create_layer(style_dim, channels[i]),
                'style_w2': self._create_layer(style_dim, channels[i+1]),
                'noise_scale': np.ones((1, channels[i+1], 1, 1)) * 0.1
            })
        
        # To RGB layer
        self.to_rgb = self._create_conv(channels[-1], 3, kernel_size=1)
        
        # Constant input
        self.const_input = np.random.randn(1, 512, 4, 4) * 0.1
    
    def _create_layer(self, in_size: int, out_size: int) -> dict:
        """Create dense layer."""
        scale = np.sqrt(2.0 / in_size)
        return {
            'weights': np.random.randn(in_size, out_size) * scale,
            'biases': np.zeros((1, out_size))
        }
    
    def _create_conv(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> dict:
        """Create convolutional layer."""
        scale = np.sqrt(2.0 / (in_ch * kernel_size * kernel_size))
        return {
            'weights': np.random.randn(out_ch, in_ch, kernel_size, kernel_size) * scale,
            'biases': np.zeros((out_ch, 1, 1))
        }
    
    def mapping_network(self, z: np.ndarray) -> np.ndarray:
        """Transform latent code z to style code w."""
        w = z
        for layer in self.mapping_layers:
            w = np.dot(w, layer['weights']) + layer['biases']
            w = np.maximum(0.2 * w, w)  # Leaky ReLU
        return w
    
    def _adaptive_instance_norm(self, x: np.ndarray, style: np.ndarray) -> np.ndarray:
        """Apply AdaIN: normalize then scale/shift with style."""
        # Normalize
        mean = np.mean(x, axis=(2, 3), keepdims=True)
        std = np.std(x, axis=(2, 3), keepdims=True) + 1e-8
        x_norm = (x - mean) / std
        
        # Style modulation
        scale = style[:, :x.shape[1], np.newaxis, np.newaxis]
        shift = style[:, x.shape[1]:2*x.shape[1], np.newaxis, np.newaxis] if style.shape[1] >= 2*x.shape[1] else 0
        
        return scale * x_norm + shift
    
    def _upsample(self, x: np.ndarray) -> np.ndarray:
        """Bilinear upsampling."""
        return np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)
    
    def _conv2d(self, x: np.ndarray, kernel: dict) -> np.ndarray:
        """Simple 2D convolution."""
        batch, in_ch, h, w = x.shape
        out_ch, _, kh, kw = kernel['weights'].shape
        
        # Pad for same size
        pad = kh // 2
        x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        
        output = np.zeros((batch, out_ch, h, w))
        
        for i in range(h):
            for j in range(w):
                patch = x_padded[:, :, i:i+kh, j:j+kw]
                for k in range(out_ch):
                    output[:, k, i, j] = np.sum(
                        patch * kernel['weights'][k], axis=(1, 2, 3)
                    ) + kernel['biases'][k, 0, 0]
        
        return output
    
    def synthesize(self, w: np.ndarray) -> np.ndarray:
        """Generate image from style code."""
        batch_size = w.shape[0]
        
        # Start with constant input
        x = np.tile(self.const_input, (batch_size, 1, 1, 1))
        
        for layer in self.synthesis_layers:
            # Upsample
            x = self._upsample(x)
            
            # Style modulation
            style = np.dot(w, layer['style_w1']['weights']) + layer['style_w1']['biases']
            x = self._adaptive_instance_norm(x, style)
            
            # Convolution
            x = self._conv2d(x, layer['conv'])
            
            # Add noise
            noise = np.random.randn(*x.shape) * layer['noise_scale']
            x = x + noise
            
            # Activation
            x = np.maximum(0.2 * x, x)  # Leaky ReLU
        
        # To RGB
        x = self._conv2d(x, self.to_rgb)
        x = np.tanh(x)
        
        return x
    
    def generate(self, n_samples: int = 1) -> np.ndarray:
        """Generate images from random latent codes."""
        z = np.random.randn(n_samples, self.latent_dim)
        w = self.mapping_network(z)
        return self.synthesize(w)
    
    def style_mixing(self, z1: np.ndarray, z2: np.ndarray,
                    mixing_layer: int = 4) -> np.ndarray:
        """Mix styles from two latent codes at specified layer."""
        w1 = self.mapping_network(z1)
        w2 = self.mapping_network(z2)
        
        # Use w1 for early layers, w2 for later layers
        # This is a simplified version - full StyleGAN uses w at each layer
        w_mixed = np.where(
            np.arange(self.style_dim) < self.style_dim * mixing_layer / 8,
            w1, w2
        )
        
        return self.synthesize(w_mixed)
```


### 24.4 Neural Style Transfer

Transfer artistic styles between images using neural networks.

```python
"""
Neural Style Transfer for Procedural Art
========================================

Transfer the style of one image to the content of another.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class StyleTransferConfig:
    """Configuration for neural style transfer."""
    content_weight: float = 1.0
    style_weight: float = 1000.0
    total_variation_weight: float = 0.01
    image_size: Tuple[int, int] = (256, 256)
    n_iterations: int = 500
    learning_rate: float = 2.0
    content_layers: List[str] = None
    style_layers: List[str] = None
    
    def __post_init__(self):
        if self.content_layers is None:
            self.content_layers = ['conv4_2']
        if self.style_layers is None:
            self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']


class VGGFeatureExtractor:
    """
    Simplified VGG-like feature extractor for style transfer.
    
    Real implementation would load pretrained VGG19 weights.
    This provides the architecture for educational purposes.
    """
    
    def __init__(self):
        # Layer configurations (channels)
        self.layer_configs = {
            'conv1_1': (3, 64), 'conv1_2': (64, 64),
            'conv2_1': (64, 128), 'conv2_2': (128, 128),
            'conv3_1': (128, 256), 'conv3_2': (256, 256),
            'conv3_3': (256, 256), 'conv3_4': (256, 256),
            'conv4_1': (256, 512), 'conv4_2': (512, 512),
            'conv4_3': (512, 512), 'conv4_4': (512, 512),
            'conv5_1': (512, 512), 'conv5_2': (512, 512),
            'conv5_3': (512, 512), 'conv5_4': (512, 512),
        }
        
        # Initialize kernels (would normally load pretrained weights)
        self.kernels = {}
        for name, (in_ch, out_ch) in self.layer_configs.items():
            scale = np.sqrt(2.0 / (in_ch * 9))
            self.kernels[name] = {
                'weights': np.random.randn(out_ch, in_ch, 3, 3) * scale,
                'biases': np.zeros((out_ch,))
            }
    
    def _conv2d(self, x: np.ndarray, kernel: dict) -> np.ndarray:
        """2D convolution with same padding."""
        batch, in_ch, h, w = x.shape
        out_ch = kernel['weights'].shape[0]
        
        # Pad input
        x_padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='reflect')
        
        output = np.zeros((batch, out_ch, h, w))
        
        for i in range(h):
            for j in range(w):
                patch = x_padded[:, :, i:i+3, j:j+3]
                for k in range(out_ch):
                    output[:, k, i, j] = np.sum(
                        patch * kernel['weights'][k], axis=(1, 2, 3)
                    ) + kernel['biases'][k]
        
        return output
    
    def _max_pool(self, x: np.ndarray) -> np.ndarray:
        """2x2 max pooling."""
        batch, ch, h, w = x.shape
        return x.reshape(batch, ch, h//2, 2, w//2, 2).max(axis=(3, 5))
    
    def forward(self, x: np.ndarray, layers: List[str]) -> Dict[str, np.ndarray]:
        """Extract features from specified layers."""
        features = {}
        
        # Block 1
        x = np.maximum(0, self._conv2d(x, self.kernels['conv1_1']))
        if 'conv1_1' in layers: features['conv1_1'] = x.copy()
        x = np.maximum(0, self._conv2d(x, self.kernels['conv1_2']))
        if 'conv1_2' in layers: features['conv1_2'] = x.copy()
        x = self._max_pool(x)
        
        # Block 2
        x = np.maximum(0, self._conv2d(x, self.kernels['conv2_1']))
        if 'conv2_1' in layers: features['conv2_1'] = x.copy()
        x = np.maximum(0, self._conv2d(x, self.kernels['conv2_2']))
        if 'conv2_2' in layers: features['conv2_2'] = x.copy()
        x = self._max_pool(x)
        
        # Block 3
        x = np.maximum(0, self._conv2d(x, self.kernels['conv3_1']))
        if 'conv3_1' in layers: features['conv3_1'] = x.copy()
        x = np.maximum(0, self._conv2d(x, self.kernels['conv3_2']))
        if 'conv3_2' in layers: features['conv3_2'] = x.copy()
        x = np.maximum(0, self._conv2d(x, self.kernels['conv3_3']))
        if 'conv3_3' in layers: features['conv3_3'] = x.copy()
        x = np.maximum(0, self._conv2d(x, self.kernels['conv3_4']))
        if 'conv3_4' in layers: features['conv3_4'] = x.copy()
        x = self._max_pool(x)
        
        # Block 4
        x = np.maximum(0, self._conv2d(x, self.kernels['conv4_1']))
        if 'conv4_1' in layers: features['conv4_1'] = x.copy()
        x = np.maximum(0, self._conv2d(x, self.kernels['conv4_2']))
        if 'conv4_2' in layers: features['conv4_2'] = x.copy()
        x = np.maximum(0, self._conv2d(x, self.kernels['conv4_3']))
        if 'conv4_3' in layers: features['conv4_3'] = x.copy()
        x = np.maximum(0, self._conv2d(x, self.kernels['conv4_4']))
        if 'conv4_4' in layers: features['conv4_4'] = x.copy()
        x = self._max_pool(x)
        
        # Block 5
        x = np.maximum(0, self._conv2d(x, self.kernels['conv5_1']))
        if 'conv5_1' in layers: features['conv5_1'] = x.copy()
        x = np.maximum(0, self._conv2d(x, self.kernels['conv5_2']))
        if 'conv5_2' in layers: features['conv5_2'] = x.copy()
        x = np.maximum(0, self._conv2d(x, self.kernels['conv5_3']))
        if 'conv5_3' in layers: features['conv5_3'] = x.copy()
        x = np.maximum(0, self._conv2d(x, self.kernels['conv5_4']))
        if 'conv5_4' in layers: features['conv5_4'] = x.copy()
        
        return features


class GramMatrix:
    """
    Gram matrix computation for style representation.
    
    The Gram matrix captures style by measuring correlations
    between feature map channels.
    """
    
    @staticmethod
    def compute(features: np.ndarray) -> np.ndarray:
        """
        Compute Gram matrix from feature maps.
        
        Args:
            features: Shape (batch, channels, height, width)
        
        Returns:
            Gram matrix of shape (batch, channels, channels)
        """
        batch, channels, height, width = features.shape
        
        # Reshape to (batch, channels, height * width)
        features_flat = features.reshape(batch, channels, -1)
        
        # Compute Gram matrix: G = F * F^T
        gram = np.matmul(features_flat, features_flat.transpose(0, 2, 1))
        
        # Normalize by number of elements
        gram = gram / (channels * height * width)
        
        return gram


class NeuralStyleTransfer:
    """
    Neural style transfer implementation.
    
    Combines content from one image with style from another
    by optimizing a generated image to minimize content and style losses.
    """
    
    def __init__(self, config: StyleTransferConfig):
        self.config = config
        self.feature_extractor = VGGFeatureExtractor()
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for VGG."""
        # Ensure shape is (1, 3, H, W)
        if image.ndim == 3:
            image = image[np.newaxis, ...]
        if image.shape[-1] == 3:  # HWC -> CHW
            image = image.transpose(0, 3, 1, 2)
        
        # Normalize (VGG normalization)
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        
        return (image - mean) / std
    
    def _deprocess(self, image: np.ndarray) -> np.ndarray:
        """Convert back to displayable image."""
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        # CHW -> HWC
        image = image.transpose(0, 2, 3, 1)
        
        return image.squeeze()
    
    def content_loss(self, 
                    generated_features: Dict[str, np.ndarray],
                    content_features: Dict[str, np.ndarray]) -> float:
        """
        Compute content loss.
        
        Measures how different the generated image is from the content image
        at specific layers (typically conv4_2).
        """
        loss = 0.0
        
        for layer in self.config.content_layers:
            gen_feat = generated_features[layer]
            content_feat = content_features[layer]
            loss += np.mean((gen_feat - content_feat) ** 2)
        
        return loss * self.config.content_weight
    
    def style_loss(self,
                  generated_features: Dict[str, np.ndarray],
                  style_features: Dict[str, np.ndarray]) -> float:
        """
        Compute style loss.
        
        Compares Gram matrices of generated and style images,
        measuring style similarity across multiple layers.
        """
        loss = 0.0
        
        for layer in self.config.style_layers:
            gen_gram = GramMatrix.compute(generated_features[layer])
            style_gram = GramMatrix.compute(style_features[layer])
            
            loss += np.mean((gen_gram - style_gram) ** 2)
        
        return loss * self.config.style_weight / len(self.config.style_layers)
    
    def total_variation_loss(self, image: np.ndarray) -> float:
        """
        Total variation loss for smoothness.
        
        Encourages spatial smoothness to reduce noise artifacts.
        """
        # Horizontal variation
        h_var = np.mean((image[:, :, 1:, :] - image[:, :, :-1, :]) ** 2)
        # Vertical variation
        v_var = np.mean((image[:, :, :, 1:] - image[:, :, :, :-1]) ** 2)
        
        return (h_var + v_var) * self.config.total_variation_weight
    
    def _compute_gradient(self, 
                         generated: np.ndarray,
                         content_features: Dict[str, np.ndarray],
                         style_grams: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute gradient of total loss w.r.t. generated image.
        
        This is simplified - real implementation uses automatic differentiation.
        """
        # Numerical gradient estimation
        epsilon = 1e-4
        grad = np.zeros_like(generated)
        
        # Sample gradient at random locations for efficiency
        n_samples = 1000
        indices = [
            (0, np.random.randint(3), 
             np.random.randint(generated.shape[2]),
             np.random.randint(generated.shape[3]))
            for _ in range(n_samples)
        ]
        
        for idx in indices:
            # Compute loss at x + epsilon
            generated_plus = generated.copy()
            generated_plus[idx] += epsilon
            
            features_plus = self.feature_extractor.forward(
                generated_plus,
                self.config.content_layers + self.config.style_layers
            )
            
            loss_plus = self._compute_loss(
                features_plus, content_features, style_grams, generated_plus
            )
            
            # Compute loss at x - epsilon
            generated_minus = generated.copy()
            generated_minus[idx] -= epsilon
            
            features_minus = self.feature_extractor.forward(
                generated_minus,
                self.config.content_layers + self.config.style_layers
            )
            
            loss_minus = self._compute_loss(
                features_minus, content_features, style_grams, generated_minus
            )
            
            # Central difference
            grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        
        return grad
    
    def _compute_loss(self,
                     gen_features: Dict[str, np.ndarray],
                     content_features: Dict[str, np.ndarray],
                     style_grams: Dict[str, np.ndarray],
                     generated: np.ndarray) -> float:
        """Compute total loss."""
        # Content loss
        c_loss = 0.0
        for layer in self.config.content_layers:
            c_loss += np.mean((gen_features[layer] - content_features[layer]) ** 2)
        c_loss *= self.config.content_weight
        
        # Style loss
        s_loss = 0.0
        for layer in self.config.style_layers:
            gen_gram = GramMatrix.compute(gen_features[layer])
            s_loss += np.mean((gen_gram - style_grams[layer]) ** 2)
        s_loss *= self.config.style_weight / len(self.config.style_layers)
        
        # Total variation loss
        tv_loss = self.total_variation_loss(generated)
        
        return c_loss + s_loss + tv_loss
    
    def transfer(self,
                content_image: np.ndarray,
                style_image: np.ndarray,
                initial_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform style transfer.
        
        Args:
            content_image: Image providing content (HWC, 0-1)
            style_image: Image providing style (HWC, 0-1)
            initial_image: Starting point for optimization (default: content image)
        
        Returns:
            Stylized image (HWC, 0-1)
        """
        # Preprocess images
        content = self._preprocess(content_image)
        style = self._preprocess(style_image)
        
        # Extract content and style features
        all_layers = list(set(self.config.content_layers + self.config.style_layers))
        
        content_features = self.feature_extractor.forward(content, self.config.content_layers)
        style_features = self.feature_extractor.forward(style, self.config.style_layers)
        
        # Compute style Gram matrices
        style_grams = {
            layer: GramMatrix.compute(style_features[layer])
            for layer in self.config.style_layers
        }
        
        # Initialize generated image
        if initial_image is not None:
            generated = self._preprocess(initial_image)
        else:
            generated = content.copy()
        
        # Add small noise
        generated = generated + np.random.randn(*generated.shape) * 0.01
        
        # Optimization loop
        for i in range(self.config.n_iterations):
            # Extract generated features
            gen_features = self.feature_extractor.forward(generated, all_layers)
            
            # Compute loss
            total_loss = self._compute_loss(
                gen_features, content_features, style_grams, generated
            )
            
            if i % 50 == 0:
                print(f"Iteration {i}, Loss: {total_loss:.4f}")
            
            # Compute gradient (simplified)
            grad = self._compute_gradient(
                generated, content_features, style_grams
            )
            
            # Update image
            generated = generated - self.config.learning_rate * grad
            
            # Clip to valid range
            generated = np.clip(generated, -2, 2)
        
        return self._deprocess(generated)


class FastStyleTransfer:
    """
    Fast style transfer using a trained feed-forward network.
    
    Instead of optimizing per image, train a network to apply
    a specific style in a single forward pass.
    """
    
    def __init__(self, image_size: int = 256):
        self.image_size = image_size
        
        # Encoder
        self.encoder = [
            self._create_conv(3, 32, 9, 1),
            self._create_conv(32, 64, 3, 2),
            self._create_conv(64, 128, 3, 2),
        ]
        
        # Residual blocks
        self.residual_blocks = [
            self._create_residual_block(128) for _ in range(5)
        ]
        
        # Decoder
        self.decoder = [
            self._create_deconv(128, 64, 3, 2),
            self._create_deconv(64, 32, 3, 2),
            self._create_conv(32, 3, 9, 1),
        ]
    
    def _create_conv(self, in_ch: int, out_ch: int, 
                    kernel_size: int, stride: int) -> dict:
        """Create convolutional layer."""
        scale = np.sqrt(2.0 / (in_ch * kernel_size * kernel_size))
        return {
            'weights': np.random.randn(out_ch, in_ch, kernel_size, kernel_size) * scale,
            'biases': np.zeros((out_ch,)),
            'stride': stride,
            'padding': kernel_size // 2
        }
    
    def _create_deconv(self, in_ch: int, out_ch: int,
                      kernel_size: int, scale: int) -> dict:
        """Create transposed convolution layer."""
        scale_init = np.sqrt(2.0 / (in_ch * kernel_size * kernel_size))
        return {
            'weights': np.random.randn(in_ch, out_ch, kernel_size, kernel_size) * scale_init,
            'biases': np.zeros((out_ch,)),
            'scale': scale
        }
    
    def _create_residual_block(self, channels: int) -> dict:
        """Create residual block."""
        scale = np.sqrt(2.0 / (channels * 9))
        return {
            'conv1': {
                'weights': np.random.randn(channels, channels, 3, 3) * scale,
                'biases': np.zeros((channels,))
            },
            'conv2': {
                'weights': np.random.randn(channels, channels, 3, 3) * scale,
                'biases': np.zeros((channels,))
            },
            'in1_gamma': np.ones((channels,)),
            'in1_beta': np.zeros((channels,)),
            'in2_gamma': np.ones((channels,)),
            'in2_beta': np.zeros((channels,)),
        }
    
    def _conv2d(self, x: np.ndarray, layer: dict) -> np.ndarray:
        """Apply convolution."""
        batch, in_ch, h, w = x.shape
        out_ch, _, kh, kw = layer['weights'].shape
        stride = layer['stride']
        padding = layer['padding']
        
        # Pad
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                         mode='reflect')
        
        out_h = (h + 2 * padding - kh) // stride + 1
        out_w = (w + 2 * padding - kw) // stride + 1
        
        output = np.zeros((batch, out_ch, out_h, out_w))
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                patch = x_padded[:, :, h_start:h_start+kh, w_start:w_start+kw]
                
                for k in range(out_ch):
                    output[:, k, i, j] = np.sum(
                        patch * layer['weights'][k], axis=(1, 2, 3)
                    ) + layer['biases'][k]
        
        return output
    
    def _upsample_conv(self, x: np.ndarray, layer: dict) -> np.ndarray:
        """Upsample then convolve (sub-pixel convolution alternative)."""
        scale = layer['scale']
        
        # Nearest neighbor upsample
        x = np.repeat(np.repeat(x, scale, axis=2), scale, axis=3)
        
        # Then convolve
        batch, in_ch, h, w = x.shape
        out_ch = layer['weights'].shape[1]
        kh, kw = layer['weights'].shape[2:]
        
        padding = kh // 2
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                         mode='reflect')
        
        output = np.zeros((batch, out_ch, h, w))
        
        for i in range(h):
            for j in range(w):
                patch = x_padded[:, :, i:i+kh, j:j+kw]
                for k in range(out_ch):
                    output[:, k, i, j] = np.sum(
                        patch * layer['weights'][:, k], axis=(1, 2, 3)
                    ) + layer['biases'][k]
        
        return output
    
    def _instance_norm(self, x: np.ndarray, gamma: np.ndarray, 
                       beta: np.ndarray) -> np.ndarray:
        """Instance normalization."""
        mean = np.mean(x, axis=(2, 3), keepdims=True)
        var = np.var(x, axis=(2, 3), keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + 1e-5)
        
        return gamma.reshape(1, -1, 1, 1) * x_norm + beta.reshape(1, -1, 1, 1)
    
    def _residual_block_forward(self, x: np.ndarray, block: dict) -> np.ndarray:
        """Forward pass through residual block."""
        residual = x
        
        # Conv -> IN -> ReLU
        out = self._conv2d(x, {'weights': block['conv1']['weights'],
                              'biases': block['conv1']['biases'],
                              'stride': 1, 'padding': 1})
        out = self._instance_norm(out, block['in1_gamma'], block['in1_beta'])
        out = np.maximum(0, out)
        
        # Conv -> IN
        out = self._conv2d(out, {'weights': block['conv2']['weights'],
                                'biases': block['conv2']['biases'],
                                'stride': 1, 'padding': 1})
        out = self._instance_norm(out, block['in2_gamma'], block['in2_beta'])
        
        return out + residual
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply style transfer to input image.
        
        Args:
            x: Input image (batch, 3, H, W), values in [0, 1]
        
        Returns:
            Stylized image (batch, 3, H, W)
        """
        # Encode
        for layer in self.encoder:
            x = self._conv2d(x, layer)
            x = np.maximum(0, x)  # ReLU
        
        # Residual blocks
        for block in self.residual_blocks:
            x = self._residual_block_forward(x, block)
        
        # Decode
        for i, layer in enumerate(self.decoder):
            if i < len(self.decoder) - 1:
                x = self._upsample_conv(x, layer)
                x = np.maximum(0, x)  # ReLU
            else:
                x = self._conv2d(x, layer)
                x = np.tanh(x) * 0.5 + 0.5  # Sigmoid-like output
        
        return x
    
    def stylize(self, image: np.ndarray) -> np.ndarray:
        """
        Stylize a single image.
        
        Args:
            image: Input image (H, W, 3), values in [0, 1]
        
        Returns:
            Stylized image (H, W, 3)
        """
        # Add batch dimension and convert to CHW
        x = image.transpose(2, 0, 1)[np.newaxis, ...]
        
        # Forward pass
        stylized = self.forward(x)
        
        # Convert back to HWC
        return stylized.squeeze().transpose(1, 2, 0)
```

## Chapter 25: Intelligent Parameter Selection

### 25.1 Automatic Parameter Optimization

Automatically find optimal parameters for artistic effects.

```python
"""
Intelligent Parameter Selection for Procedural Art
==================================================

Automatically determine optimal parameters for artistic effects
using optimization algorithms and machine learning.
"""

import numpy as np
from typing import Callable, Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum


class OptimizationMethod(Enum):
    """Available optimization methods."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"


@dataclass
class ParameterRange:
    """Definition of a parameter's valid range."""
    name: str
    min_value: float
    max_value: float
    step: Optional[float] = None
    log_scale: bool = False
    integer: bool = False
    
    def sample(self) -> float:
        """Sample a random value from this range."""
        if self.log_scale:
            log_min = np.log(self.min_value)
            log_max = np.log(self.max_value)
            value = np.exp(np.random.uniform(log_min, log_max))
        else:
            value = np.random.uniform(self.min_value, self.max_value)
        
        if self.integer:
            value = int(round(value))
        
        if self.step:
            value = round(value / self.step) * self.step
        
        return value
    
    def grid_values(self, n_points: int = 10) -> List[float]:
        """Generate grid of values for this parameter."""
        if self.log_scale:
            values = np.logspace(
                np.log10(self.min_value),
                np.log10(self.max_value),
                n_points
            )
        else:
            values = np.linspace(self.min_value, self.max_value, n_points)
        
        if self.integer:
            values = np.unique(np.round(values).astype(int))
        
        if self.step:
            values = np.unique(np.round(values / self.step) * self.step)
        
        return list(values)


class GridSearch:
    """
    Exhaustive grid search over parameter space.
    
    Best for small parameter spaces with few dimensions.
    """
    
    def __init__(self, 
                 parameters: List[ParameterRange],
                 objective_fn: Callable[[Dict[str, float]], float],
                 n_points_per_param: int = 10):
        self.parameters = parameters
        self.objective_fn = objective_fn
        self.n_points = n_points_per_param
        self.results = []
    
    def search(self) -> Tuple[Dict[str, float], float]:
        """Perform grid search and return best parameters."""
        # Generate grid for each parameter
        grids = [p.grid_values(self.n_points) for p in self.parameters]
        
        best_params = None
        best_score = float('-inf')
        
        # Iterate over all combinations
        indices = [0] * len(self.parameters)
        
        while True:
            # Create current parameter set
            params = {
                self.parameters[i].name: grids[i][indices[i]]
                for i in range(len(self.parameters))
            }
            
            # Evaluate
            score = self.objective_fn(params)
            self.results.append((params.copy(), score))
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
            
            # Increment indices
            for i in range(len(indices) - 1, -1, -1):
                indices[i] += 1
                if indices[i] < len(grids[i]):
                    break
                indices[i] = 0
                if i == 0:
                    return best_params, best_score
        
        return best_params, best_score


class RandomSearch:
    """
    Random search over parameter space.
    
    Often more efficient than grid search for high-dimensional spaces.
    """
    
    def __init__(self,
                 parameters: List[ParameterRange],
                 objective_fn: Callable[[Dict[str, float]], float],
                 n_iterations: int = 100):
        self.parameters = parameters
        self.objective_fn = objective_fn
        self.n_iterations = n_iterations
        self.results = []
    
    def search(self) -> Tuple[Dict[str, float], float]:
        """Perform random search and return best parameters."""
        best_params = None
        best_score = float('-inf')
        
        for _ in range(self.n_iterations):
            # Sample random parameters
            params = {p.name: p.sample() for p in self.parameters}
            
            # Evaluate
            score = self.objective_fn(params)
            self.results.append((params.copy(), score))
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
        
        return best_params, best_score


class BayesianOptimizer:
    """
    Bayesian optimization using Gaussian Process surrogate model.
    
    Efficiently explores parameter space by balancing exploration
    and exploitation using acquisition functions.
    """
    
    def __init__(self,
                 parameters: List[ParameterRange],
                 objective_fn: Callable[[Dict[str, float]], float],
                 n_iterations: int = 50,
                 n_initial: int = 5):
        self.parameters = parameters
        self.objective_fn = objective_fn
        self.n_iterations = n_iterations
        self.n_initial = n_initial
        
        # Observed data
        self.X = []  # Parameter vectors
        self.y = []  # Observed scores
        
        # GP hyperparameters
        self.length_scale = 1.0
        self.signal_variance = 1.0
        self.noise_variance = 0.01
    
    def _params_to_vector(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dict to normalized vector."""
        vector = []
        for p in self.parameters:
            value = params[p.name]
            # Normalize to [0, 1]
            if p.log_scale:
                normalized = (np.log(value) - np.log(p.min_value)) / \
                            (np.log(p.max_value) - np.log(p.min_value))
            else:
                normalized = (value - p.min_value) / (p.max_value - p.min_value)
            vector.append(normalized)
        return np.array(vector)
    
    def _vector_to_params(self, vector: np.ndarray) -> Dict[str, float]:
        """Convert normalized vector back to parameters."""
        params = {}
        for i, p in enumerate(self.parameters):
            normalized = vector[i]
            if p.log_scale:
                value = np.exp(
                    np.log(p.min_value) + normalized * 
                    (np.log(p.max_value) - np.log(p.min_value))
                )
            else:
                value = p.min_value + normalized * (p.max_value - p.min_value)
            
            if p.integer:
                value = int(round(value))
            params[p.name] = value
        
        return params
    
    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """RBF (squared exponential) kernel."""
        diff = x1 - x2
        return self.signal_variance * np.exp(
            -0.5 * np.sum(diff ** 2) / (self.length_scale ** 2)
        )
    
    def _compute_kernel_matrix(self, X: List[np.ndarray]) -> np.ndarray:
        """Compute kernel matrix for all observed points."""
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self._rbf_kernel(X[i], X[j])
        return K + self.noise_variance * np.eye(n)
    
    def _predict(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Predict mean and variance at point x using GP.
        """
        if len(self.X) == 0:
            return 0.0, self.signal_variance
        
        # Kernel matrix and vector
        K = self._compute_kernel_matrix(self.X)
        k = np.array([self._rbf_kernel(x, xi) for xi in self.X])
        
        # Solve K * alpha = y
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
            v = np.linalg.solve(L, k)
        except np.linalg.LinAlgError:
            # Fall back to regularized inverse
            K_reg = K + 1e-6 * np.eye(len(K))
            alpha = np.linalg.solve(K_reg, self.y)
            v = np.linalg.solve(K_reg, k)
        
        # Predictive mean and variance
        mean = np.dot(k, alpha)
        variance = self.signal_variance - np.dot(v, v)
        variance = max(variance, 1e-6)  # Ensure positive
        
        return mean, variance
    
    def _expected_improvement(self, x: np.ndarray, xi: float = 0.01) -> float:
        """
        Expected Improvement acquisition function.
        
        Balances exploration (high variance) and exploitation (high mean).
        """
        if len(self.y) == 0:
            return 1.0
        
        mean, variance = self._predict(x)
        std = np.sqrt(variance)
        
        best_y = max(self.y)
        
        if std < 1e-8:
            return 0.0
        
        z = (mean - best_y - xi) / std
        
        # Standard normal CDF and PDF
        cdf = 0.5 * (1 + np.tanh(z * 0.7978845608))  # Approximation
        pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
        
        ei = (mean - best_y - xi) * cdf + std * pdf
        
        return ei
    
    def _sample_acquisition(self, n_samples: int = 1000) -> np.ndarray:
        """Find point that maximizes acquisition function."""
        best_x = None
        best_ei = float('-inf')
        
        for _ in range(n_samples):
            x = np.random.rand(len(self.parameters))
            ei = self._expected_improvement(x)
            
            if ei > best_ei:
                best_ei = ei
                best_x = x
        
        return best_x
    
    def search(self) -> Tuple[Dict[str, float], float]:
        """Perform Bayesian optimization."""
        # Initial random samples
        for _ in range(self.n_initial):
            params = {p.name: p.sample() for p in self.parameters}
            x = self._params_to_vector(params)
            score = self.objective_fn(params)
            
            self.X.append(x)
            self.y.append(score)
        
        # Bayesian optimization loop
        for i in range(self.n_iterations - self.n_initial):
            # Find next point to evaluate
            x_next = self._sample_acquisition()
            params_next = self._vector_to_params(x_next)
            
            # Evaluate
            score = self.objective_fn(params_next)
            
            self.X.append(x_next)
            self.y.append(score)
            
            if (i + 1) % 10 == 0:
                print(f"Iteration {i + 1}, Best score: {max(self.y):.4f}")
        
        # Return best found
        best_idx = np.argmax(self.y)
        best_params = self._vector_to_params(self.X[best_idx])
        
        return best_params, self.y[best_idx]


class GeneticOptimizer:
    """
    Genetic algorithm for parameter optimization.
    
    Good for complex, non-convex optimization landscapes.
    """
    
    def __init__(self,
                 parameters: List[ParameterRange],
                 objective_fn: Callable[[Dict[str, float]], float],
                 population_size: int = 50,
                 n_generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_fraction: float = 0.1):
        
        self.parameters = parameters
        self.objective_fn = objective_fn
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = int(elite_fraction * population_size)
    
    def _create_individual(self) -> np.ndarray:
        """Create random individual (genome)."""
        return np.array([p.sample() for p in self.parameters])
    
    def _decode_individual(self, genome: np.ndarray) -> Dict[str, float]:
        """Convert genome to parameter dict."""
        return {p.name: genome[i] for i, p in enumerate(self.parameters)}
    
    def _evaluate_population(self, population: List[np.ndarray]) -> np.ndarray:
        """Evaluate fitness of all individuals."""
        return np.array([
            self.objective_fn(self._decode_individual(ind))
            for ind in population
        ])
    
    def _select_parents(self, population: List[np.ndarray], 
                       fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tournament selection for two parents."""
        def tournament(k: int = 3) -> np.ndarray:
            indices = np.random.choice(len(population), k, replace=False)
            best_idx = indices[np.argmax(fitness[indices])]
            return population[best_idx]
        
        return tournament(), tournament()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Single-point crossover."""
        if np.random.random() > self.crossover_rate:
            return parent1.copy()
        
        point = np.random.randint(1, len(parent1))
        child = np.concatenate([parent1[:point], parent2[point:]])
        return child
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Gaussian mutation."""
        mutated = individual.copy()
        
        for i, p in enumerate(self.parameters):
            if np.random.random() < self.mutation_rate:
                # Gaussian perturbation
                std = (p.max_value - p.min_value) * 0.1
                mutated[i] += np.random.normal(0, std)
                
                # Clip to valid range
                mutated[i] = np.clip(mutated[i], p.min_value, p.max_value)
                
                if p.integer:
                    mutated[i] = round(mutated[i])
        
        return mutated
    
    def search(self) -> Tuple[Dict[str, float], float]:
        """Perform genetic algorithm optimization."""
        # Initialize population
        population = [self._create_individual() for _ in range(self.population_size)]
        
        best_individual = None
        best_fitness = float('-inf')
        
        for generation in range(self.n_generations):
            # Evaluate fitness
            fitness = self._evaluate_population(population)
            
            # Track best
            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_individual = population[gen_best_idx].copy()
            
            if generation % 10 == 0:
                print(f"Generation {generation}, Best: {best_fitness:.4f}, "
                      f"Avg: {np.mean(fitness):.4f}")
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness)[-self.elite_count:]
            new_population = [population[i].copy() for i in elite_indices]
            
            # Create rest of population through selection, crossover, mutation
            while len(new_population) < self.population_size:
                parent1, parent2 = self._select_parents(population, fitness)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population
        
        return self._decode_individual(best_individual), best_fitness


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization (PSO).
    
    Particles explore the parameter space, guided by personal
    best positions and global best position.
    """
    
    def __init__(self,
                 parameters: List[ParameterRange],
                 objective_fn: Callable[[Dict[str, float]], float],
                 n_particles: int = 30,
                 n_iterations: int = 100,
                 w: float = 0.7,      # Inertia weight
                 c1: float = 1.5,     # Cognitive (personal) weight
                 c2: float = 1.5):    # Social (global) weight
        
        self.parameters = parameters
        self.objective_fn = objective_fn
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        self.n_dims = len(parameters)
    
    def _init_particles(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize particle positions and velocities."""
        positions = np.zeros((self.n_particles, self.n_dims))
        velocities = np.zeros((self.n_particles, self.n_dims))
        
        for i, p in enumerate(self.parameters):
            # Random initial positions
            positions[:, i] = np.random.uniform(
                p.min_value, p.max_value, self.n_particles
            )
            # Small random initial velocities
            velocity_range = (p.max_value - p.min_value) * 0.1
            velocities[:, i] = np.random.uniform(
                -velocity_range, velocity_range, self.n_particles
            )
        
        return positions, velocities
    
    def _evaluate_particles(self, positions: np.ndarray) -> np.ndarray:
        """Evaluate fitness at all particle positions."""
        scores = np.zeros(self.n_particles)
        
        for j in range(self.n_particles):
            params = {
                self.parameters[i].name: positions[j, i]
                for i in range(self.n_dims)
            }
            scores[j] = self.objective_fn(params)
        
        return scores
    
    def _clip_to_bounds(self, positions: np.ndarray) -> np.ndarray:
        """Clip positions to valid bounds."""
        clipped = positions.copy()
        for i, p in enumerate(self.parameters):
            clipped[:, i] = np.clip(clipped[:, i], p.min_value, p.max_value)
        return clipped
    
    def search(self) -> Tuple[Dict[str, float], float]:
        """Perform PSO optimization."""
        # Initialize
        positions, velocities = self._init_particles()
        
        # Evaluate initial positions
        scores = self._evaluate_particles(positions)
        
        # Personal best positions and scores
        personal_best_positions = positions.copy()
        personal_best_scores = scores.copy()
        
        # Global best
        global_best_idx = np.argmax(scores)
        global_best_position = positions[global_best_idx].copy()
        global_best_score = scores[global_best_idx]
        
        for iteration in range(self.n_iterations):
            # Update velocities
            r1 = np.random.random((self.n_particles, self.n_dims))
            r2 = np.random.random((self.n_particles, self.n_dims))
            
            cognitive = self.c1 * r1 * (personal_best_positions - positions)
            social = self.c2 * r2 * (global_best_position - positions)
            
            velocities = self.w * velocities + cognitive + social
            
            # Update positions
            positions = positions + velocities
            positions = self._clip_to_bounds(positions)
            
            # Evaluate new positions
            scores = self._evaluate_particles(positions)
            
            # Update personal bests
            improved = scores > personal_best_scores
            personal_best_positions[improved] = positions[improved]
            personal_best_scores[improved] = scores[improved]
            
            # Update global best
            if np.max(scores) > global_best_score:
                global_best_idx = np.argmax(scores)
                global_best_position = positions[global_best_idx].copy()
                global_best_score = scores[global_best_idx]
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Best: {global_best_score:.4f}")
        
        best_params = {
            self.parameters[i].name: global_best_position[i]
            for i in range(self.n_dims)
        }
        
        return best_params, global_best_score


class SimulatedAnnealing:
    """
    Simulated annealing optimization.
    
    Probabilistically accepts worse solutions to escape local optima,
    with acceptance probability decreasing as temperature drops.
    """
    
    def __init__(self,
                 parameters: List[ParameterRange],
                 objective_fn: Callable[[Dict[str, float]], float],
                 n_iterations: int = 1000,
                 initial_temp: float = 100.0,
                 final_temp: float = 0.1,
                 cooling_rate: float = 0.995):
        
        self.parameters = parameters
        self.objective_fn = objective_fn
        self.n_iterations = n_iterations
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
    
    def _initial_solution(self) -> np.ndarray:
        """Generate random initial solution."""
        return np.array([p.sample() for p in self.parameters])
    
    def _neighbor(self, solution: np.ndarray, temp: float) -> np.ndarray:
        """Generate neighboring solution with temperature-scaled step."""
        neighbor = solution.copy()
        
        # Modify one random parameter
        idx = np.random.randint(len(self.parameters))
        p = self.parameters[idx]
        
        # Step size scales with temperature
        step_scale = temp / self.initial_temp
        step = (p.max_value - p.min_value) * 0.1 * step_scale
        
        neighbor[idx] += np.random.normal(0, step)
        neighbor[idx] = np.clip(neighbor[idx], p.min_value, p.max_value)
        
        if p.integer:
            neighbor[idx] = round(neighbor[idx])
        
        return neighbor
    
    def _evaluate(self, solution: np.ndarray) -> float:
        """Evaluate solution."""
        params = {
            self.parameters[i].name: solution[i]
            for i in range(len(self.parameters))
        }
        return self.objective_fn(params)
    
    def _acceptance_probability(self, current_score: float, 
                               new_score: float, temp: float) -> float:
        """Calculate acceptance probability."""
        if new_score > current_score:
            return 1.0
        return np.exp((new_score - current_score) / temp)
    
    def search(self) -> Tuple[Dict[str, float], float]:
        """Perform simulated annealing optimization."""
        # Initialize
        current = self._initial_solution()
        current_score = self._evaluate(current)
        
        best = current.copy()
        best_score = current_score
        
        temp = self.initial_temp
        
        for iteration in range(self.n_iterations):
            # Generate neighbor
            neighbor = self._neighbor(current, temp)
            neighbor_score = self._evaluate(neighbor)
            
            # Accept or reject
            if np.random.random() < self._acceptance_probability(
                current_score, neighbor_score, temp
            ):
                current = neighbor
                current_score = neighbor_score
                
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
            
            # Cool down
            temp = max(self.final_temp, temp * self.cooling_rate)
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Temp: {temp:.2f}, Best: {best_score:.4f}")
        
        best_params = {
            self.parameters[i].name: best[i]
            for i in range(len(self.parameters))
        }
        
        return best_params, best_score
```

### 25.2 Image Quality Assessment

Automatically evaluate image quality for objective optimization.

```python
"""
Image Quality Assessment for Procedural Art
============================================

Objective metrics to evaluate generated art quality.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Collection of quality metrics for an image."""
    sharpness: float
    contrast: float
    colorfulness: float
    saturation: float
    noise_level: float
    edge_density: float
    texture_complexity: float
    color_harmony: float
    composition_score: float
    overall_score: float


class ImageQualityAssessor:
    """
    Assess image quality using multiple metrics.
    
    Provides objective scoring for procedural art optimization.
    """
    
    def __init__(self):
        self.weights = {
            'sharpness': 0.15,
            'contrast': 0.1,
            'colorfulness': 0.15,
            'saturation': 0.1,
            'noise_level': 0.1,  # Lower is better
            'edge_density': 0.1,
            'texture_complexity': 0.1,
            'color_harmony': 0.1,
            'composition_score': 0.1
        }
    
    def assess(self, image: np.ndarray) -> QualityMetrics:
        """
        Compute comprehensive quality metrics.
        
        Args:
            image: RGB image (H, W, 3), values in [0, 1]
        
        Returns:
            QualityMetrics with all computed scores
        """
        # Ensure correct format
        if image.max() > 1.0:
            image = image / 255.0
        
        # Compute individual metrics
        sharpness = self._compute_sharpness(image)
        contrast = self._compute_contrast(image)
        colorfulness = self._compute_colorfulness(image)
        saturation = self._compute_saturation(image)
        noise = self._compute_noise_level(image)
        edges = self._compute_edge_density(image)
        texture = self._compute_texture_complexity(image)
        harmony = self._compute_color_harmony(image)
        composition = self._compute_composition_score(image)
        
        # Compute overall score
        overall = (
            self.weights['sharpness'] * sharpness +
            self.weights['contrast'] * contrast +
            self.weights['colorfulness'] * colorfulness +
            self.weights['saturation'] * saturation +
            self.weights['noise_level'] * (1 - noise) +  # Invert noise
            self.weights['edge_density'] * edges +
            self.weights['texture_complexity'] * texture +
            self.weights['color_harmony'] * harmony +
            self.weights['composition_score'] * composition
        )
        
        return QualityMetrics(
            sharpness=sharpness,
            contrast=contrast,
            colorfulness=colorfulness,
            saturation=saturation,
            noise_level=noise,
            edge_density=edges,
            texture_complexity=texture,
            color_harmony=harmony,
            composition_score=composition,
            overall_score=overall
        )
    
    def _compute_sharpness(self, image: np.ndarray) -> float:
        """
        Compute sharpness using Laplacian variance.
        
        Higher variance indicates sharper image.
        """
        # Convert to grayscale
        gray = np.dot(image, [0.299, 0.587, 0.114])
        
        # Laplacian kernel
        laplacian = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ])
        
        # Convolve (simplified)
        h, w = gray.shape
        result = np.zeros((h-2, w-2))
        
        for i in range(h-2):
            for j in range(w-2):
                result[i, j] = np.sum(gray[i:i+3, j:j+3] * laplacian)
        
        # Variance of Laplacian
        variance = np.var(result)
        
        # Normalize to [0, 1]
        return min(1.0, variance / 0.1)
    
    def _compute_contrast(self, image: np.ndarray) -> float:
        """
        Compute global contrast using RMS contrast.
        """
        gray = np.dot(image, [0.299, 0.587, 0.114])
        
        # RMS contrast
        rms = np.sqrt(np.mean((gray - np.mean(gray)) ** 2))
        
        # Normalize (assuming max RMS ~ 0.5 for high contrast)
        return min(1.0, rms / 0.35)
    
    def _compute_colorfulness(self, image: np.ndarray) -> float:
        """
        Compute colorfulness metric (Hasler & Süsstrunk).
        """
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        # Color opponent dimensions
        rg = r - g
        yb = 0.5 * (r + g) - b
        
        # Standard deviation and mean
        std_rg = np.std(rg)
        std_yb = np.std(yb)
        mean_rg = np.mean(np.abs(rg))
        mean_yb = np.mean(np.abs(yb))
        
        std_root = np.sqrt(std_rg ** 2 + std_yb ** 2)
        mean_root = np.sqrt(mean_rg ** 2 + mean_yb ** 2)
        
        colorfulness = std_root + 0.3 * mean_root
        
        # Normalize (assuming max colorfulness ~ 1.0)
        return min(1.0, colorfulness)
    
    def _compute_saturation(self, image: np.ndarray) -> float:
        """
        Compute average saturation in HSV space.
        """
        # RGB to HSV conversion (simplified)
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        
        # Saturation = (max - min) / max
        saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)
        
        return np.mean(saturation)
    
    def _compute_noise_level(self, image: np.ndarray) -> float:
        """
        Estimate noise level using median absolute deviation.
        """
        gray = np.dot(image, [0.299, 0.587, 0.114])
        
        # High-pass filter (difference from local median)
        h, w = gray.shape
        noise_estimate = np.zeros((h-2, w-2))
        
        for i in range(h-2):
            for j in range(w-2):
                patch = gray[i:i+3, j:j+3]
                noise_estimate[i, j] = abs(gray[i+1, j+1] - np.median(patch))
        
        # Median absolute deviation
        mad = np.median(np.abs(noise_estimate - np.median(noise_estimate)))
        noise_sigma = 1.4826 * mad  # Convert MAD to sigma estimate
        
        # Normalize
        return min(1.0, noise_sigma / 0.1)
    
    def _compute_edge_density(self, image: np.ndarray) -> float:
        """
        Compute edge density using Sobel operator.
        """
        gray = np.dot(image, [0.299, 0.587, 0.114])
        
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        h, w = gray.shape
        gx = np.zeros((h-2, w-2))
        gy = np.zeros((h-2, w-2))
        
        for i in range(h-2):
            for j in range(w-2):
                patch = gray[i:i+3, j:j+3]
                gx[i, j] = np.sum(patch * sobel_x)
                gy[i, j] = np.sum(patch * sobel_y)
        
        # Edge magnitude
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        
        # Edge density (fraction of strong edges)
        threshold = 0.1
        edge_density = np.mean(magnitude > threshold)
        
        return edge_density
    
    def _compute_texture_complexity(self, image: np.ndarray) -> float:
        """
        Compute texture complexity using GLCM-like features.
        """
        gray = np.dot(image, [0.299, 0.587, 0.114])
        
        # Quantize to 8 levels
        gray_quantized = (gray * 7).astype(int)
        
        # Compute co-occurrence (horizontal neighbors)
        h, w = gray_quantized.shape
        cooccurrence = np.zeros((8, 8))
        
        for i in range(h):
            for j in range(w - 1):
                cooccurrence[gray_quantized[i, j], gray_quantized[i, j+1]] += 1
        
        # Normalize
        cooccurrence = cooccurrence / cooccurrence.sum()
        
        # Entropy as complexity measure
        entropy = -np.sum(cooccurrence * np.log2(cooccurrence + 1e-10))
        
        # Normalize (max entropy ~ 6 for 8 levels)
        return min(1.0, entropy / 5.0)
    
    def _compute_color_harmony(self, image: np.ndarray) -> float:
        """
        Compute color harmony based on hue distribution.
        
        Harmonious images have hues clustered in complementary
        or analogous patterns.
        """
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        # RGB to Hue (simplified)
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        delta = max_rgb - min_rgb
        
        hue = np.zeros_like(r)
        
        mask = delta > 0.001
        
        # Red is max
        red_max = mask & (max_rgb == r)
        hue[red_max] = ((g[red_max] - b[red_max]) / delta[red_max]) % 6
        
        # Green is max
        green_max = mask & (max_rgb == g)
        hue[green_max] = (b[green_max] - r[green_max]) / delta[green_max] + 2
        
        # Blue is max
        blue_max = mask & (max_rgb == b)
        hue[blue_max] = (r[blue_max] - g[blue_max]) / delta[blue_max] + 4
        
        hue = hue / 6.0  # Normalize to [0, 1]
        
        # Histogram of hues
        hist, _ = np.histogram(hue[mask], bins=12, range=(0, 1))
        hist = hist / (hist.sum() + 1e-10)
        
        # Harmony: check for complementary or analogous patterns
        # Complementary: peaks at opposite sides
        # Analogous: peaks close together
        
        # Entropy-based measure (lower entropy = more harmony)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Invert and normalize (lower entropy = higher harmony)
        harmony = 1.0 - (entropy / np.log2(12))
        
        return max(0.0, harmony)
    
    def _compute_composition_score(self, image: np.ndarray) -> float:
        """
        Evaluate composition using rule of thirds and visual balance.
        """
        gray = np.dot(image, [0.299, 0.587, 0.114])
        h, w = gray.shape
        
        # Edge magnitude for saliency
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        saliency = np.zeros((h-2, w-2))
        
        for i in range(h-2):
            for j in range(w-2):
                patch = gray[i:i+3, j:j+3]
                gx = np.sum(patch * sobel_x)
                gy = np.sum(patch * sobel_y)
                saliency[i, j] = np.sqrt(gx ** 2 + gy ** 2)
        
        # Rule of thirds: check saliency at intersection points
        thirds_h = [saliency.shape[0] // 3, 2 * saliency.shape[0] // 3]
        thirds_w = [saliency.shape[1] // 3, 2 * saliency.shape[1] // 3]
        
        # Sample saliency around thirds intersection
        roi_size = min(saliency.shape) // 6
        thirds_score = 0.0
        
        for th in thirds_h:
            for tw in thirds_w:
                h_start = max(0, th - roi_size)
                h_end = min(saliency.shape[0], th + roi_size)
                w_start = max(0, tw - roi_size)
                w_end = min(saliency.shape[1], tw + roi_size)
                
                roi_saliency = saliency[h_start:h_end, w_start:w_end]
                thirds_score += np.mean(roi_saliency)
        
        thirds_score /= 4.0  # Average of 4 intersection points
        
        # Visual balance: compare saliency in quadrants
        mid_h, mid_w = saliency.shape[0] // 2, saliency.shape[1] // 2
        
        quadrants = [
            saliency[:mid_h, :mid_w].mean(),
            saliency[:mid_h, mid_w:].mean(),
            saliency[mid_h:, :mid_w].mean(),
            saliency[mid_h:, mid_w:].mean()
        ]
        
        # Balance: lower variance in quadrant saliency = better balance
        balance = 1.0 - np.std(quadrants) / (np.mean(quadrants) + 1e-10)
        balance = max(0.0, balance)
        
        # Combine thirds and balance scores
        composition = 0.5 * min(1.0, thirds_score / 0.3) + 0.5 * balance
        
        return composition
```


### 25.3 Style Analysis and Classification

Analyze artistic styles to guide procedural generation.

```python
"""
Style Analysis for Procedural Art
=================================

Automatically analyze and classify artistic styles
to guide AI-driven procedural art generation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ArtMovement(Enum):
    """Major art movements for classification."""
    IMPRESSIONISM = "impressionism"
    EXPRESSIONISM = "expressionism"
    CUBISM = "cubism"
    SURREALISM = "surrealism"
    ABSTRACT = "abstract"
    POP_ART = "pop_art"
    MINIMALISM = "minimalism"
    PHOTOREALISM = "photorealism"
    WATERCOLOR = "watercolor"
    OIL_PAINTING = "oil_painting"
    DIGITAL_ART = "digital_art"
    PIXEL_ART = "pixel_art"


@dataclass
class StyleFeatures:
    """Extracted style features from an image."""
    # Color features
    dominant_colors: List[Tuple[int, int, int]]
    color_palette_size: int
    color_temperature: float  # -1 (cool) to 1 (warm)
    saturation_mean: float
    saturation_variance: float
    
    # Texture features
    brush_stroke_size: float
    texture_regularity: float
    edge_sharpness: float
    
    # Composition features
    symmetry_score: float
    complexity: float
    focal_points: List[Tuple[int, int]]
    
    # Style classification
    predicted_style: ArtMovement
    style_confidence: float


class ColorAnalyzer:
    """
    Analyze color characteristics of an image.
    """
    
    @staticmethod
    def extract_dominant_colors(image: np.ndarray, 
                               n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors using k-means clustering.
        """
        # Reshape image to pixel list
        pixels = image.reshape(-1, 3)
        
        # Simple k-means implementation
        # Initialize centroids randomly
        indices = np.random.choice(len(pixels), n_colors, replace=False)
        centroids = pixels[indices].astype(float)
        
        for _ in range(20):  # K-means iterations
            # Assign pixels to nearest centroid
            distances = np.zeros((len(pixels), n_colors))
            for i, centroid in enumerate(centroids):
                distances[:, i] = np.sum((pixels - centroid) ** 2, axis=1)
            
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(n_colors):
                mask = labels == i
                if np.any(mask):
                    new_centroids[i] = pixels[mask].mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        # Sort by cluster size
        cluster_sizes = [np.sum(labels == i) for i in range(n_colors)]
        sorted_indices = np.argsort(cluster_sizes)[::-1]
        
        return [tuple(centroids[i].astype(int)) for i in sorted_indices]
    
    @staticmethod
    def compute_color_temperature(image: np.ndarray) -> float:
        """
        Compute color temperature (-1 cool to +1 warm).
        """
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        # Warm colors: high red, low blue
        # Cool colors: low red, high blue
        warmth = (r.mean() - b.mean()) / 255.0
        
        return np.clip(warmth, -1, 1)
    
    @staticmethod
    def compute_palette_size(image: np.ndarray, threshold: float = 10) -> int:
        """
        Estimate effective palette size.
        
        Counts distinct colors (within threshold distance).
        """
        # Quantize to reduce computation
        quantized = (image // 16) * 16
        
        # Get unique colors
        pixels = quantized.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        
        return len(unique_colors)
    
    @staticmethod
    def analyze_saturation(image: np.ndarray) -> Tuple[float, float]:
        """
        Compute mean and variance of saturation.
        """
        r, g, b = image[:, :, 0] / 255, image[:, :, 1] / 255, image[:, :, 2] / 255
        
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        
        saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)
        
        return float(np.mean(saturation)), float(np.var(saturation))


class TextureAnalyzer:
    """
    Analyze texture characteristics of an image.
    """
    
    @staticmethod
    def estimate_brush_stroke_size(image: np.ndarray) -> float:
        """
        Estimate brush stroke size from texture patterns.
        
        Uses autocorrelation to find characteristic scale.
        """
        gray = np.dot(image, [0.299, 0.587, 0.114]) / 255.0
        
        # Compute local variance at different scales
        scales = [3, 5, 7, 11, 15, 21]
        variances = []
        
        for scale in scales:
            # Simple local variance
            h, w = gray.shape
            local_var = np.zeros((h - scale + 1, w - scale + 1))
            
            for i in range(h - scale + 1):
                for j in range(w - scale + 1):
                    patch = gray[i:i+scale, j:j+scale]
                    local_var[i, j] = np.var(patch)
            
            variances.append(np.mean(local_var))
        
        # Find scale with maximum variance (characteristic texture size)
        max_idx = np.argmax(variances)
        
        # Normalize to [0, 1] range
        return scales[max_idx] / 21.0
    
    @staticmethod
    def compute_texture_regularity(image: np.ndarray) -> float:
        """
        Compute texture regularity (periodic vs random).
        
        Regular patterns have strong frequency peaks.
        """
        gray = np.dot(image, [0.299, 0.587, 0.114]) / 255.0
        
        # 2D FFT
        fft = np.fft.fft2(gray)
        magnitude = np.abs(fft)
        
        # Remove DC component
        magnitude[0, 0] = 0
        
        # Regularity: ratio of peaks to average
        sorted_mag = np.sort(magnitude.flatten())[::-1]
        top_10_percent = sorted_mag[:int(len(sorted_mag) * 0.1)]
        rest = sorted_mag[int(len(sorted_mag) * 0.1):]
        
        regularity = top_10_percent.mean() / (rest.mean() + 1e-10)
        
        # Normalize
        return min(1.0, regularity / 10.0)
    
    @staticmethod
    def compute_edge_sharpness(image: np.ndarray) -> float:
        """
        Measure edge sharpness (sharp vs soft/blurred).
        """
        gray = np.dot(image, [0.299, 0.587, 0.114]) / 255.0
        
        # Laplacian
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        
        h, w = gray.shape
        result = np.zeros((h-2, w-2))
        
        for i in range(h-2):
            for j in range(w-2):
                result[i, j] = np.abs(np.sum(gray[i:i+3, j:j+3] * laplacian))
        
        # Higher values = sharper edges
        return min(1.0, np.mean(result) * 10)


class CompositionAnalyzer:
    """
    Analyze composition of an image.
    """
    
    @staticmethod
    def compute_symmetry(image: np.ndarray) -> float:
        """
        Compute bilateral symmetry score.
        """
        gray = np.dot(image, [0.299, 0.587, 0.114])
        
        # Compare left and right halves
        h, w = gray.shape
        mid = w // 2
        
        left = gray[:, :mid]
        right = gray[:, -mid:][:, ::-1]  # Flip right side
        
        # Ensure same size
        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w]
        right = right[:, :min_w]
        
        # Normalized cross-correlation
        correlation = np.sum(left * right) / (
            np.sqrt(np.sum(left ** 2) * np.sum(right ** 2)) + 1e-10
        )
        
        return max(0.0, correlation)
    
    @staticmethod
    def compute_complexity(image: np.ndarray) -> float:
        """
        Compute visual complexity using compression ratio.
        
        More complex images compress less.
        """
        # Quantize image
        quantized = (image // 32) * 32
        
        # Count unique pixel patterns
        h, w, c = quantized.shape
        
        # 3x3 pattern diversity
        patterns = set()
        
        for i in range(0, h - 2, 3):
            for j in range(0, w - 2, 3):
                patch = quantized[i:i+3, j:j+3].tobytes()
                patterns.add(patch)
        
        # More patterns = more complex
        max_patterns = (h // 3) * (w // 3)
        complexity = len(patterns) / max_patterns
        
        return complexity
    
    @staticmethod
    def find_focal_points(image: np.ndarray, n_points: int = 3) -> List[Tuple[int, int]]:
        """
        Find visual focal points using saliency.
        """
        gray = np.dot(image, [0.299, 0.587, 0.114]) / 255.0
        
        # Compute gradient magnitude as saliency
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        h, w = gray.shape
        saliency = np.zeros((h-2, w-2))
        
        for i in range(h-2):
            for j in range(w-2):
                patch = gray[i:i+3, j:j+3]
                gx = np.sum(patch * sobel_x)
                gy = np.sum(patch * sobel_y)
                saliency[i, j] = np.sqrt(gx ** 2 + gy ** 2)
        
        # Find local maxima
        focal_points = []
        block_size = max(h, w) // 6
        
        for _ in range(n_points):
            if saliency.max() < 0.01:
                break
            
            # Find global max
            max_idx = np.unravel_index(np.argmax(saliency), saliency.shape)
            focal_points.append((int(max_idx[0]), int(max_idx[1])))
            
            # Suppress surrounding area
            y, x = max_idx
            y_start = max(0, y - block_size)
            y_end = min(saliency.shape[0], y + block_size)
            x_start = max(0, x - block_size)
            x_end = min(saliency.shape[1], x + block_size)
            saliency[y_start:y_end, x_start:x_end] = 0
        
        return focal_points


class StyleClassifier:
    """
    Classify image into art movement/style categories.
    """
    
    def __init__(self):
        # Style signatures (simplified feature templates)
        self.style_signatures = {
            ArtMovement.IMPRESSIONISM: {
                'brush_stroke': (0.4, 0.7),  # Medium brush strokes
                'edge_sharpness': (0.2, 0.5),  # Soft edges
                'saturation': (0.4, 0.8),  # Moderate saturation
                'color_temp': (-0.3, 0.5),  # Any temperature
            },
            ArtMovement.EXPRESSIONISM: {
                'brush_stroke': (0.5, 1.0),  # Bold strokes
                'edge_sharpness': (0.3, 0.7),  # Variable edges
                'saturation': (0.5, 1.0),  # High saturation
                'color_temp': (-0.5, 0.8),  # Often warm
            },
            ArtMovement.CUBISM: {
                'brush_stroke': (0.3, 0.6),
                'edge_sharpness': (0.6, 1.0),  # Sharp geometric edges
                'saturation': (0.3, 0.7),
                'regularity': (0.3, 0.7),  # Geometric patterns
            },
            ArtMovement.MINIMALISM: {
                'brush_stroke': (0.0, 0.3),  # Few brush strokes
                'complexity': (0.0, 0.3),  # Low complexity
                'saturation': (0.2, 0.5),  # Muted colors
            },
            ArtMovement.POP_ART: {
                'saturation': (0.7, 1.0),  # High saturation
                'edge_sharpness': (0.7, 1.0),  # Sharp edges
                'color_temp': (0.0, 0.7),  # Often warm/bright
            },
            ArtMovement.PHOTOREALISM: {
                'edge_sharpness': (0.8, 1.0),  # Very sharp
                'brush_stroke': (0.0, 0.2),  # Invisible strokes
                'complexity': (0.6, 1.0),  # High detail
            },
            ArtMovement.WATERCOLOR: {
                'edge_sharpness': (0.1, 0.4),  # Soft edges
                'saturation': (0.3, 0.6),  # Transparent colors
                'brush_stroke': (0.2, 0.5),
            },
            ArtMovement.PIXEL_ART: {
                'regularity': (0.7, 1.0),  # Very regular
                'palette_size': (2, 64),  # Limited palette
                'edge_sharpness': (0.9, 1.0),  # Pixel-perfect
            },
        }
    
    def classify(self, 
                 color_features: dict,
                 texture_features: dict,
                 composition_features: dict) -> Tuple[ArtMovement, float]:
        """
        Classify image style based on extracted features.
        """
        scores = {}
        
        for style, signature in self.style_signatures.items():
            score = 0.0
            count = 0
            
            # Check each feature in signature
            for feature, (min_val, max_val) in signature.items():
                value = None
                
                if feature == 'brush_stroke':
                    value = texture_features.get('brush_stroke_size')
                elif feature == 'edge_sharpness':
                    value = texture_features.get('edge_sharpness')
                elif feature == 'saturation':
                    value = color_features.get('saturation_mean')
                elif feature == 'color_temp':
                    value = color_features.get('color_temperature')
                elif feature == 'regularity':
                    value = texture_features.get('texture_regularity')
                elif feature == 'complexity':
                    value = composition_features.get('complexity')
                elif feature == 'palette_size':
                    value = color_features.get('palette_size')
                    # Normalize palette size for comparison
                    value = min(1.0, (value - min_val) / (max_val - min_val))
                    min_val, max_val = 0.0, 1.0
                
                if value is not None:
                    # Score: 1 if in range, 0 if outside, interpolated at edges
                    if min_val <= value <= max_val:
                        score += 1.0
                    elif value < min_val:
                        score += max(0, 1 - (min_val - value) * 5)
                    else:
                        score += max(0, 1 - (value - max_val) * 5)
                    count += 1
            
            if count > 0:
                scores[style] = score / count
            else:
                scores[style] = 0.0
        
        # Find best match
        best_style = max(scores, key=scores.get)
        confidence = scores[best_style]
        
        return best_style, confidence


class StyleAnalyzer:
    """
    Complete style analysis pipeline.
    """
    
    def __init__(self):
        self.color_analyzer = ColorAnalyzer()
        self.texture_analyzer = TextureAnalyzer()
        self.composition_analyzer = CompositionAnalyzer()
        self.classifier = StyleClassifier()
    
    def analyze(self, image: np.ndarray) -> StyleFeatures:
        """
        Perform comprehensive style analysis.
        
        Args:
            image: RGB image (H, W, 3), values in [0, 255]
        
        Returns:
            StyleFeatures with complete analysis
        """
        # Color analysis
        dominant_colors = self.color_analyzer.extract_dominant_colors(image)
        palette_size = self.color_analyzer.compute_palette_size(image)
        color_temp = self.color_analyzer.compute_color_temperature(image)
        sat_mean, sat_var = self.color_analyzer.analyze_saturation(image)
        
        color_features = {
            'dominant_colors': dominant_colors,
            'palette_size': palette_size,
            'color_temperature': color_temp,
            'saturation_mean': sat_mean,
            'saturation_variance': sat_var,
        }
        
        # Texture analysis
        brush_stroke = self.texture_analyzer.estimate_brush_stroke_size(image)
        regularity = self.texture_analyzer.compute_texture_regularity(image)
        sharpness = self.texture_analyzer.compute_edge_sharpness(image)
        
        texture_features = {
            'brush_stroke_size': brush_stroke,
            'texture_regularity': regularity,
            'edge_sharpness': sharpness,
        }
        
        # Composition analysis
        symmetry = self.composition_analyzer.compute_symmetry(image)
        complexity = self.composition_analyzer.compute_complexity(image)
        focal_points = self.composition_analyzer.find_focal_points(image)
        
        composition_features = {
            'symmetry_score': symmetry,
            'complexity': complexity,
            'focal_points': focal_points,
        }
        
        # Style classification
        style, confidence = self.classifier.classify(
            color_features, texture_features, composition_features
        )
        
        return StyleFeatures(
            dominant_colors=dominant_colors,
            color_palette_size=palette_size,
            color_temperature=color_temp,
            saturation_mean=sat_mean,
            saturation_variance=sat_var,
            brush_stroke_size=brush_stroke,
            texture_regularity=regularity,
            edge_sharpness=sharpness,
            symmetry_score=symmetry,
            complexity=complexity,
            focal_points=focal_points,
            predicted_style=style,
            style_confidence=confidence
        )


class StyleTransferGuide:
    """
    Guide procedural generation based on style analysis.
    
    Maps style features to procedural generation parameters.
    """
    
    def __init__(self):
        self.parameter_mappings = {
            'blur_radius': {
                'source': 'edge_sharpness',
                'transform': lambda x: (1 - x) * 5,  # Soft edges = more blur
            },
            'noise_amount': {
                'source': 'brush_stroke_size',
                'transform': lambda x: x * 0.3,
            },
            'color_jitter': {
                'source': 'saturation_variance',
                'transform': lambda x: x * 0.5,
            },
            'contrast': {
                'source': 'edge_sharpness',
                'transform': lambda x: 0.5 + x * 0.5,
            },
            'saturation_boost': {
                'source': 'saturation_mean',
                'transform': lambda x: x,
            },
        }
    
    def generate_parameters(self, style_features: StyleFeatures) -> Dict[str, float]:
        """
        Generate procedural parameters from style features.
        """
        parameters = {}
        
        for param_name, mapping in self.parameter_mappings.items():
            source_attr = mapping['source']
            transform = mapping['transform']
            
            source_value = getattr(style_features, source_attr, 0.5)
            parameters[param_name] = transform(source_value)
        
        # Add style-specific parameters
        if style_features.predicted_style == ArtMovement.IMPRESSIONISM:
            parameters['brush_density'] = 0.7
            parameters['color_bleeding'] = 0.3
        elif style_features.predicted_style == ArtMovement.PIXEL_ART:
            parameters['pixelation_size'] = 4
            parameters['palette_limit'] = 16
        elif style_features.predicted_style == ArtMovement.WATERCOLOR:
            parameters['wet_edges'] = 0.5
            parameters['paper_texture'] = 0.4
        
        return parameters
```

## Chapter 26: Procedural Generation Strategies

### 26.1 Evolutionary Art Systems

Use genetic algorithms to evolve artwork.

```python
"""
Evolutionary Art Systems
========================

Generate art through evolutionary processes,
selecting and breeding successful visual genomes.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import copy


@dataclass
class Gene:
    """A single gene in an art genome."""
    name: str
    value: float
    min_value: float
    max_value: float
    mutation_rate: float = 0.1
    mutation_scale: float = 0.2
    
    def mutate(self) -> 'Gene':
        """Create mutated copy of this gene."""
        if np.random.random() < self.mutation_rate:
            # Gaussian mutation
            delta = np.random.normal(0, self.mutation_scale * (self.max_value - self.min_value))
            new_value = np.clip(self.value + delta, self.min_value, self.max_value)
            return Gene(self.name, new_value, self.min_value, self.max_value,
                       self.mutation_rate, self.mutation_scale)
        return Gene(self.name, self.value, self.min_value, self.max_value,
                   self.mutation_rate, self.mutation_scale)


@dataclass
class ArtGenome:
    """Complete genome for evolutionary art."""
    genes: Dict[str, Gene]
    
    def mutate(self) -> 'ArtGenome':
        """Create mutated copy."""
        new_genes = {name: gene.mutate() for name, gene in self.genes.items()}
        return ArtGenome(new_genes)
    
    def crossover(self, other: 'ArtGenome') -> Tuple['ArtGenome', 'ArtGenome']:
        """Create two children through crossover."""
        child1_genes = {}
        child2_genes = {}
        
        for name in self.genes:
            if np.random.random() < 0.5:
                child1_genes[name] = copy.deepcopy(self.genes[name])
                child2_genes[name] = copy.deepcopy(other.genes[name])
            else:
                child1_genes[name] = copy.deepcopy(other.genes[name])
                child2_genes[name] = copy.deepcopy(self.genes[name])
        
        return ArtGenome(child1_genes), ArtGenome(child2_genes)
    
    def get_parameters(self) -> Dict[str, float]:
        """Convert genome to parameter dictionary."""
        return {name: gene.value for name, gene in self.genes.items()}


class ArtRenderer(ABC):
    """Abstract base for art rendering from genome."""
    
    @abstractmethod
    def render(self, genome: ArtGenome, size: Tuple[int, int]) -> np.ndarray:
        """Render image from genome."""
        pass


class ShapeBasedRenderer(ArtRenderer):
    """
    Render art using procedural shapes.
    
    Genome controls shape properties and composition.
    """
    
    def __init__(self, n_shapes: int = 50):
        self.n_shapes = n_shapes
    
    @staticmethod
    def create_genome(n_shapes: int = 50) -> ArtGenome:
        """Create random genome for shape-based art."""
        genes = {
            'background_r': Gene('background_r', np.random.random(), 0, 1),
            'background_g': Gene('background_g', np.random.random(), 0, 1),
            'background_b': Gene('background_b', np.random.random(), 0, 1),
            'shape_type_bias': Gene('shape_type_bias', np.random.random(), 0, 1),
            'size_mean': Gene('size_mean', np.random.uniform(0.05, 0.3), 0.01, 0.5),
            'size_variance': Gene('size_variance', np.random.uniform(0, 0.2), 0, 0.3),
            'opacity_mean': Gene('opacity_mean', np.random.uniform(0.3, 0.9), 0.1, 1),
            'color_harmony': Gene('color_harmony', np.random.random(), 0, 1),
            'position_bias_x': Gene('position_bias_x', 0.5, 0, 1),
            'position_bias_y': Gene('position_bias_y', 0.5, 0, 1),
            'rotation_range': Gene('rotation_range', np.random.random(), 0, 1),
        }
        
        # Individual shape genes
        for i in range(n_shapes):
            genes[f'shape_{i}_x'] = Gene(f'shape_{i}_x', np.random.random(), 0, 1)
            genes[f'shape_{i}_y'] = Gene(f'shape_{i}_y', np.random.random(), 0, 1)
            genes[f'shape_{i}_hue'] = Gene(f'shape_{i}_hue', np.random.random(), 0, 1)
            genes[f'shape_{i}_size'] = Gene(f'shape_{i}_size', np.random.random(), 0, 1)
        
        return ArtGenome(genes)
    
    def render(self, genome: ArtGenome, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Render shapes from genome."""
        h, w = size
        params = genome.get_parameters()
        
        # Create background
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :, 0] = params['background_r']
        image[:, :, 1] = params['background_g']
        image[:, :, 2] = params['background_b']
        
        # Render each shape
        for i in range(self.n_shapes):
            x = params.get(f'shape_{i}_x', np.random.random())
            y = params.get(f'shape_{i}_y', np.random.random())
            hue = params.get(f'shape_{i}_hue', np.random.random())
            shape_size = params.get(f'shape_{i}_size', np.random.random())
            
            # Convert to pixel coordinates
            cx = int(x * w)
            cy = int(y * h)
            
            # Size based on genome parameters
            base_size = params['size_mean'] * min(h, w)
            variance = params['size_variance'] * base_size
            radius = int(base_size + (shape_size - 0.5) * 2 * variance)
            radius = max(1, radius)
            
            # Color from hue (with harmony based on base color)
            base_hue = params['color_harmony']
            actual_hue = (base_hue + hue * (1 - params['color_harmony'])) % 1.0
            r, g, b = self._hsv_to_rgb(actual_hue, 0.7, 0.9)
            
            # Opacity
            opacity = params['opacity_mean']
            
            # Draw shape based on type bias
            if params['shape_type_bias'] < 0.33:
                self._draw_circle(image, cx, cy, radius, r, g, b, opacity)
            elif params['shape_type_bias'] < 0.66:
                self._draw_rectangle(image, cx, cy, radius, r, g, b, opacity)
            else:
                self._draw_triangle(image, cx, cy, radius, r, g, b, opacity)
        
        return np.clip(image, 0, 1)
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[float, float, float]:
        """Convert HSV to RGB."""
        if s == 0:
            return v, v, v
        
        h = h * 6
        i = int(h)
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        if i == 0: return v, t, p
        elif i == 1: return q, v, p
        elif i == 2: return p, v, t
        elif i == 3: return p, q, v
        elif i == 4: return t, p, v
        else: return v, p, q
    
    def _draw_circle(self, image: np.ndarray, cx: int, cy: int, 
                     radius: int, r: float, g: float, b: float, opacity: float):
        """Draw circle with alpha blending."""
        h, w = image.shape[:2]
        
        y, x = np.ogrid[:h, :w]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        
        # Soft edges
        distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        alpha = np.clip(1 - (distances - radius + 2) / 4, 0, 1) * opacity
        alpha = alpha[:, :, np.newaxis]
        
        mask_3d = mask[:, :, np.newaxis]
        color = np.array([r, g, b])
        
        image[mask] = (1 - alpha[mask]) * image[mask] + alpha[mask] * color
    
    def _draw_rectangle(self, image: np.ndarray, cx: int, cy: int,
                        size: int, r: float, g: float, b: float, opacity: float):
        """Draw rectangle with alpha blending."""
        h, w = image.shape[:2]
        
        x1 = max(0, cx - size)
        x2 = min(w, cx + size)
        y1 = max(0, cy - size)
        y2 = min(h, cy + size)
        
        color = np.array([r, g, b])
        
        for yi in range(y1, y2):
            for xi in range(x1, x2):
                image[yi, xi] = (1 - opacity) * image[yi, xi] + opacity * color
    
    def _draw_triangle(self, image: np.ndarray, cx: int, cy: int,
                       size: int, r: float, g: float, b: float, opacity: float):
        """Draw triangle with alpha blending."""
        h, w = image.shape[:2]
        
        # Triangle vertices
        points = [
            (cx, cy - size),
            (cx - size, cy + size),
            (cx + size, cy + size)
        ]
        
        color = np.array([r, g, b])
        
        # Bounding box
        min_x = max(0, min(p[0] for p in points))
        max_x = min(w, max(p[0] for p in points))
        min_y = max(0, min(p[1] for p in points))
        max_y = min(h, max(p[1] for p in points))
        
        # Simple point-in-triangle test
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        for yi in range(min_y, max_y):
            for xi in range(min_x, max_x):
                d1 = sign((xi, yi), points[0], points[1])
                d2 = sign((xi, yi), points[1], points[2])
                d3 = sign((xi, yi), points[2], points[0])
                
                has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
                has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
                
                if not (has_neg and has_pos):
                    image[yi, xi] = (1 - opacity) * image[yi, xi] + opacity * color


class MathematicalRenderer(ArtRenderer):
    """
    Render art using mathematical functions.
    
    Creates abstract art from equation parameters.
    """
    
    @staticmethod
    def create_genome() -> ArtGenome:
        """Create genome for mathematical art."""
        genes = {
            # Function parameters
            'freq_x': Gene('freq_x', np.random.uniform(1, 10), 0.1, 20),
            'freq_y': Gene('freq_y', np.random.uniform(1, 10), 0.1, 20),
            'phase_x': Gene('phase_x', np.random.uniform(0, 2*np.pi), 0, 2*np.pi),
            'phase_y': Gene('phase_y', np.random.uniform(0, 2*np.pi), 0, 2*np.pi),
            'amplitude': Gene('amplitude', np.random.uniform(0.3, 1), 0.1, 1),
            
            # Color mapping
            'color_offset_r': Gene('color_offset_r', np.random.random(), 0, 1),
            'color_offset_g': Gene('color_offset_g', np.random.random(), 0, 1),
            'color_offset_b': Gene('color_offset_b', np.random.random(), 0, 1),
            'color_freq_r': Gene('color_freq_r', np.random.uniform(0.5, 3), 0.1, 5),
            'color_freq_g': Gene('color_freq_g', np.random.uniform(0.5, 3), 0.1, 5),
            'color_freq_b': Gene('color_freq_b', np.random.uniform(0.5, 3), 0.1, 5),
            
            # Complexity
            'harmonics': Gene('harmonics', np.random.randint(1, 6), 1, 8),
            'distortion': Gene('distortion', np.random.uniform(0, 0.5), 0, 1),
            'rotation': Gene('rotation', np.random.uniform(0, np.pi), 0, np.pi),
        }
        
        return ArtGenome(genes)
    
    def render(self, genome: ArtGenome, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Render mathematical art."""
        h, w = size
        params = genome.get_parameters()
        
        # Create coordinate grids
        x = np.linspace(-np.pi, np.pi, w)
        y = np.linspace(-np.pi, np.pi, h)
        X, Y = np.meshgrid(x, y)
        
        # Apply rotation
        theta = params['rotation']
        X_rot = X * np.cos(theta) - Y * np.sin(theta)
        Y_rot = X * np.sin(theta) + Y * np.cos(theta)
        
        # Generate base pattern
        pattern = np.zeros((h, w))
        n_harmonics = int(params['harmonics'])
        
        for i in range(1, n_harmonics + 1):
            pattern += (1 / i) * np.sin(
                params['freq_x'] * i * X_rot + params['phase_x']
            ) * np.cos(
                params['freq_y'] * i * Y_rot + params['phase_y']
            )
        
        # Apply distortion
        distortion = params['distortion']
        pattern += distortion * np.sin(pattern * 3)
        
        # Normalize to [0, 1]
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-10)
        pattern = pattern * params['amplitude']
        
        # Map to colors
        image = np.zeros((h, w, 3))
        
        image[:, :, 0] = 0.5 + 0.5 * np.sin(
            pattern * params['color_freq_r'] * np.pi + params['color_offset_r'] * np.pi
        )
        image[:, :, 1] = 0.5 + 0.5 * np.sin(
            pattern * params['color_freq_g'] * np.pi + params['color_offset_g'] * np.pi
        )
        image[:, :, 2] = 0.5 + 0.5 * np.sin(
            pattern * params['color_freq_b'] * np.pi + params['color_offset_b'] * np.pi
        )
        
        return np.clip(image, 0, 1)


class EvolutionaryArtSystem:
    """
    Complete evolutionary art system with selection and breeding.
    """
    
    def __init__(self,
                 renderer: ArtRenderer,
                 population_size: int = 20,
                 elite_size: int = 4,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.7):
        
        self.renderer = renderer
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.population: List[ArtGenome] = []
        self.fitness_scores: List[float] = []
        self.generation = 0
    
    def initialize_population(self, genome_factory: Callable[[], ArtGenome]):
        """Create initial random population."""
        self.population = [genome_factory() for _ in range(self.population_size)]
        self.fitness_scores = [0.0] * self.population_size
        self.generation = 0
    
    def evaluate_population(self, 
                           fitness_fn: Callable[[np.ndarray], float],
                           image_size: Tuple[int, int] = (256, 256)):
        """Evaluate fitness of all individuals."""
        for i, genome in enumerate(self.population):
            image = self.renderer.render(genome, image_size)
            self.fitness_scores[i] = fitness_fn(image)
    
    def select_parents(self) -> Tuple[ArtGenome, ArtGenome]:
        """Tournament selection for two parents."""
        def tournament(k: int = 3) -> ArtGenome:
            indices = np.random.choice(len(self.population), k, replace=False)
            best_idx = indices[np.argmax([self.fitness_scores[i] for i in indices])]
            return self.population[best_idx]
        
        return tournament(), tournament()
    
    def evolve_generation(self):
        """Create next generation through selection, crossover, mutation."""
        # Sort by fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        
        new_population = []
        
        # Elitism: keep best individuals
        for i in range(self.elite_size):
            new_population.append(copy.deepcopy(self.population[sorted_indices[i]]))
        
        # Create rest through breeding
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # Mutation
            if np.random.random() < self.mutation_rate:
                child1 = child1.mutate()
            if np.random.random() < self.mutation_rate:
                child2 = child2.mutate()
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = new_population
        self.fitness_scores = [0.0] * self.population_size
        self.generation += 1
    
    def get_best(self) -> Tuple[ArtGenome, float]:
        """Get the best individual and its fitness."""
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx], self.fitness_scores[best_idx]
    
    def run_evolution(self,
                      genome_factory: Callable[[], ArtGenome],
                      fitness_fn: Callable[[np.ndarray], float],
                      n_generations: int = 50,
                      image_size: Tuple[int, int] = (256, 256),
                      verbose: bool = True) -> ArtGenome:
        """
        Run complete evolutionary process.
        
        Args:
            genome_factory: Function to create initial genomes
            fitness_fn: Function to evaluate image fitness (higher is better)
            n_generations: Number of generations to evolve
            image_size: Size of images for fitness evaluation
            verbose: Print progress
        
        Returns:
            Best evolved genome
        """
        # Initialize
        self.initialize_population(genome_factory)
        
        best_overall = None
        best_fitness_overall = float('-inf')
        
        for gen in range(n_generations):
            # Evaluate
            self.evaluate_population(fitness_fn, image_size)
            
            # Track best
            best_genome, best_fitness = self.get_best()
            
            if best_fitness > best_fitness_overall:
                best_fitness_overall = best_fitness
                best_overall = copy.deepcopy(best_genome)
            
            if verbose:
                avg_fitness = np.mean(self.fitness_scores)
                print(f"Generation {gen}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")
            
            # Evolve (except last generation)
            if gen < n_generations - 1:
                self.evolve_generation()
        
        return best_overall


class InteractiveEvolution:
    """
    Interactive evolutionary art with human fitness evaluation.
    
    Users select preferred images to guide evolution.
    """
    
    def __init__(self, renderer: ArtRenderer, grid_size: int = 3):
        self.renderer = renderer
        self.grid_size = grid_size
        self.population_size = grid_size * grid_size
        
        self.population: List[ArtGenome] = []
        self.selected_indices: List[int] = []
        self.history: List[List[ArtGenome]] = []
    
    def initialize(self, genome_factory: Callable[[], ArtGenome]):
        """Create initial population for display."""
        self.population = [genome_factory() for _ in range(self.population_size)]
        self.selected_indices = []
    
    def render_grid(self, image_size: int = 256) -> np.ndarray:
        """Render population as image grid."""
        grid = np.zeros((
            self.grid_size * image_size,
            self.grid_size * image_size,
            3
        ))
        
        for i, genome in enumerate(self.population):
            row = i // self.grid_size
            col = i % self.grid_size
            
            image = self.renderer.render(genome, (image_size, image_size))
            
            y_start = row * image_size
            x_start = col * image_size
            
            grid[y_start:y_start+image_size, x_start:x_start+image_size] = image
        
        return grid
    
    def select(self, indices: List[int]):
        """Record user selections."""
        self.selected_indices = indices
    
    def breed_next_generation(self):
        """Create next generation based on selections."""
        if not self.selected_indices:
            return
        
        # Save current population to history
        self.history.append(copy.deepcopy(self.population))
        
        # Selected genomes become parents
        parents = [self.population[i] for i in self.selected_indices]
        
        new_population = []
        
        # Create children through crossover and mutation
        while len(new_population) < self.population_size:
            # Random parent pair
            p1, p2 = np.random.choice(len(parents), 2, replace=True)
            parent1, parent2 = parents[p1], parents[p2]
            
            # Crossover
            if np.random.random() < 0.7:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1 = copy.deepcopy(parent1)
                child2 = copy.deepcopy(parent2)
            
            # Mutation
            child1 = child1.mutate()
            child2 = child2.mutate()
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = new_population
        self.selected_indices = []
    
    def undo(self):
        """Revert to previous generation."""
        if self.history:
            self.population = self.history.pop()
            self.selected_indices = []
```


---

# PART X: PROJECT TEMPLATES AND REFERENCE

## Chapter 27: Complete Project Templates

### 27.1 Image Processing Pipeline Template

A complete, production-ready image processing pipeline.

```python
"""
Complete Image Processing Pipeline
==================================

Production-ready template for building image processing applications.
Includes loading, processing, effects, and export capabilities.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import hashlib
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the processing pipeline."""
    # Input/Output
    input_formats: List[str] = field(default_factory=lambda: ['png', 'jpg', 'jpeg', 'bmp', 'tiff'])
    output_format: str = 'png'
    output_quality: int = 95
    
    # Processing
    max_image_size: int = 8192
    auto_orient: bool = True
    preserve_metadata: bool = False
    
    # Performance
    use_cache: bool = True
    cache_dir: str = '.cache'
    num_workers: int = 4
    
    # Memory
    max_memory_mb: int = 2048
    chunk_size: int = 1024
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'input_formats': self.input_formats,
            'output_format': self.output_format,
            'output_quality': self.output_quality,
            'max_image_size': self.max_image_size,
            'auto_orient': self.auto_orient,
            'preserve_metadata': self.preserve_metadata,
            'use_cache': self.use_cache,
            'cache_dir': self.cache_dir,
            'num_workers': self.num_workers,
            'max_memory_mb': self.max_memory_mb,
            'chunk_size': self.chunk_size,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# IMAGE CONTAINER
# ============================================================================

class ColorSpace(Enum):
    """Supported color spaces."""
    RGB = "rgb"
    RGBA = "rgba"
    GRAYSCALE = "grayscale"
    HSV = "hsv"
    LAB = "lab"
    CMYK = "cmyk"


@dataclass
class ImageMetadata:
    """Image metadata container."""
    width: int
    height: int
    channels: int
    color_space: ColorSpace
    bit_depth: int = 8
    source_path: Optional[str] = None
    exif: Optional[Dict[str, Any]] = None
    custom: Dict[str, Any] = field(default_factory=dict)


class Image:
    """
    Main image container class.
    
    Wraps numpy array with metadata and utility methods.
    """
    
    def __init__(self, 
                 data: np.ndarray,
                 color_space: ColorSpace = ColorSpace.RGB,
                 metadata: Optional[ImageMetadata] = None):
        
        self._data = self._validate_data(data)
        self._color_space = color_space
        
        if metadata is None:
            h, w = data.shape[:2]
            c = data.shape[2] if data.ndim == 3 else 1
            self._metadata = ImageMetadata(
                width=w, height=h, channels=c, color_space=color_space
            )
        else:
            self._metadata = metadata
    
    def _validate_data(self, data: np.ndarray) -> np.ndarray:
        """Validate and normalize image data."""
        if data.ndim not in (2, 3):
            raise ValueError(f"Image must be 2D or 3D array, got {data.ndim}D")
        
        # Normalize to float32 [0, 1]
        if data.dtype == np.uint8:
            return data.astype(np.float32) / 255.0
        elif data.dtype == np.uint16:
            return data.astype(np.float32) / 65535.0
        elif data.dtype in (np.float32, np.float64):
            return data.astype(np.float32)
        else:
            raise ValueError(f"Unsupported dtype: {data.dtype}")
    
    @property
    def data(self) -> np.ndarray:
        """Get image data."""
        return self._data
    
    @property
    def width(self) -> int:
        return self._metadata.width
    
    @property
    def height(self) -> int:
        return self._metadata.height
    
    @property
    def channels(self) -> int:
        return self._metadata.channels
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape
    
    @property
    def color_space(self) -> ColorSpace:
        return self._color_space
    
    @property
    def metadata(self) -> ImageMetadata:
        return self._metadata
    
    def copy(self) -> 'Image':
        """Create a deep copy."""
        return Image(
            self._data.copy(),
            self._color_space,
            ImageMetadata(**{
                k: v for k, v in self._metadata.__dict__.items()
            })
        )
    
    def to_uint8(self) -> np.ndarray:
        """Convert to uint8 format."""
        return (np.clip(self._data, 0, 1) * 255).astype(np.uint8)
    
    def to_uint16(self) -> np.ndarray:
        """Convert to uint16 format."""
        return (np.clip(self._data, 0, 1) * 65535).astype(np.uint16)
    
    def compute_hash(self) -> str:
        """Compute hash of image data."""
        return hashlib.md5(self._data.tobytes()).hexdigest()
    
    def __repr__(self) -> str:
        return f"Image({self.width}x{self.height}, {self.channels}ch, {self.color_space.value})"


# ============================================================================
# PROCESSING OPERATIONS
# ============================================================================

class Operation(ABC):
    """Base class for all image operations."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self._enabled = True
        self._parameters: Dict[str, Any] = {}
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters.copy()
    
    def set_parameter(self, name: str, value: Any):
        """Set operation parameter."""
        self._parameters[name] = value
    
    @abstractmethod
    def apply(self, image: Image) -> Image:
        """Apply operation to image."""
        pass
    
    def __repr__(self) -> str:
        params = ', '.join(f"{k}={v}" for k, v in self._parameters.items())
        return f"{self.name}({params})"


class ResizeOperation(Operation):
    """Resize image to specified dimensions."""
    
    def __init__(self, 
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 scale: Optional[float] = None,
                 method: str = 'bilinear'):
        super().__init__('Resize')
        self._parameters = {
            'width': width,
            'height': height,
            'scale': scale,
            'method': method
        }
    
    def apply(self, image: Image) -> Image:
        if not self._enabled:
            return image
        
        data = image.data
        h, w = data.shape[:2]
        
        # Determine target size
        if self._parameters['scale']:
            new_w = int(w * self._parameters['scale'])
            new_h = int(h * self._parameters['scale'])
        else:
            new_w = self._parameters['width'] or w
            new_h = self._parameters['height'] or h
        
        # Simple bilinear interpolation
        resized = self._bilinear_resize(data, new_h, new_w)
        
        return Image(resized, image.color_space)
    
    def _bilinear_resize(self, data: np.ndarray, 
                        new_h: int, new_w: int) -> np.ndarray:
        """Bilinear interpolation resize."""
        old_h, old_w = data.shape[:2]
        
        # Create coordinate maps
        x_ratio = old_w / new_w
        y_ratio = old_h / new_h
        
        # Output array
        if data.ndim == 3:
            output = np.zeros((new_h, new_w, data.shape[2]), dtype=data.dtype)
        else:
            output = np.zeros((new_h, new_w), dtype=data.dtype)
        
        for i in range(new_h):
            for j in range(new_w):
                # Source coordinates
                x = j * x_ratio
                y = i * y_ratio
                
                x1 = int(x)
                y1 = int(y)
                x2 = min(x1 + 1, old_w - 1)
                y2 = min(y1 + 1, old_h - 1)
                
                # Fractional parts
                x_frac = x - x1
                y_frac = y - y1
                
                # Bilinear interpolation
                output[i, j] = (
                    data[y1, x1] * (1 - x_frac) * (1 - y_frac) +
                    data[y1, x2] * x_frac * (1 - y_frac) +
                    data[y2, x1] * (1 - x_frac) * y_frac +
                    data[y2, x2] * x_frac * y_frac
                )
        
        return output


class CropOperation(Operation):
    """Crop image to specified region."""
    
    def __init__(self,
                 x: int = 0,
                 y: int = 0,
                 width: Optional[int] = None,
                 height: Optional[int] = None):
        super().__init__('Crop')
        self._parameters = {
            'x': x, 'y': y, 'width': width, 'height': height
        }
    
    def apply(self, image: Image) -> Image:
        if not self._enabled:
            return image
        
        data = image.data
        x = self._parameters['x']
        y = self._parameters['y']
        w = self._parameters['width'] or (image.width - x)
        h = self._parameters['height'] or (image.height - y)
        
        # Validate bounds
        x = max(0, min(x, image.width - 1))
        y = max(0, min(y, image.height - 1))
        w = min(w, image.width - x)
        h = min(h, image.height - y)
        
        cropped = data[y:y+h, x:x+w]
        return Image(cropped.copy(), image.color_space)


class BrightnessContrastOperation(Operation):
    """Adjust brightness and contrast."""
    
    def __init__(self, brightness: float = 0.0, contrast: float = 1.0):
        super().__init__('BrightnessContrast')
        self._parameters = {'brightness': brightness, 'contrast': contrast}
    
    def apply(self, image: Image) -> Image:
        if not self._enabled:
            return image
        
        data = image.data.copy()
        brightness = self._parameters['brightness']
        contrast = self._parameters['contrast']
        
        # Apply contrast around midpoint
        data = (data - 0.5) * contrast + 0.5
        
        # Apply brightness
        data = data + brightness
        
        return Image(np.clip(data, 0, 1), image.color_space)


class SaturationOperation(Operation):
    """Adjust color saturation."""
    
    def __init__(self, saturation: float = 1.0):
        super().__init__('Saturation')
        self._parameters = {'saturation': saturation}
    
    def apply(self, image: Image) -> Image:
        if not self._enabled:
            return image
        
        if image.channels < 3:
            return image
        
        data = image.data.copy()
        saturation = self._parameters['saturation']
        
        # Convert to grayscale for desaturation reference
        gray = np.dot(data[..., :3], [0.299, 0.587, 0.114])
        gray = gray[..., np.newaxis]
        
        # Interpolate between grayscale and original
        data[..., :3] = gray + saturation * (data[..., :3] - gray)
        
        return Image(np.clip(data, 0, 1), image.color_space)


class GaussianBlurOperation(Operation):
    """Apply Gaussian blur."""
    
    def __init__(self, radius: float = 2.0):
        super().__init__('GaussianBlur')
        self._parameters = {'radius': radius}
    
    def apply(self, image: Image) -> Image:
        if not self._enabled:
            return image
        
        radius = self._parameters['radius']
        kernel_size = int(radius * 3) * 2 + 1
        
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, radius)
        
        # Apply separable convolution
        data = image.data
        
        if data.ndim == 3:
            result = np.zeros_like(data)
            for c in range(data.shape[2]):
                result[:, :, c] = self._convolve_2d(data[:, :, c], kernel)
        else:
            result = self._convolve_2d(data, kernel)
        
        return Image(result, image.color_space)
    
    def _create_gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Create 2D Gaussian kernel."""
        x = np.arange(size) - size // 2
        kernel_1d = np.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        return kernel_2d
    
    def _convolve_2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """2D convolution with zero padding."""
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        
        h, w = image.shape
        result = np.zeros_like(image)
        
        for i in range(h):
            for j in range(w):
                result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
        
        return result


class SharpenOperation(Operation):
    """Sharpen image using unsharp mask."""
    
    def __init__(self, amount: float = 1.0, radius: float = 1.0, threshold: float = 0.0):
        super().__init__('Sharpen')
        self._parameters = {
            'amount': amount,
            'radius': radius,
            'threshold': threshold
        }
    
    def apply(self, image: Image) -> Image:
        if not self._enabled:
            return image
        
        amount = self._parameters['amount']
        radius = self._parameters['radius']
        threshold = self._parameters['threshold']
        
        # Create blurred version
        blur_op = GaussianBlurOperation(radius)
        blurred = blur_op.apply(image)
        
        # Compute difference
        data = image.data
        diff = data - blurred.data
        
        # Apply threshold
        if threshold > 0:
            mask = np.abs(diff) > threshold
            diff = diff * mask
        
        # Add sharpening
        result = data + amount * diff
        
        return Image(np.clip(result, 0, 1), image.color_space)


class NoiseReductionOperation(Operation):
    """Reduce image noise using bilateral filter approximation."""
    
    def __init__(self, strength: float = 10.0, spatial_sigma: float = 5.0):
        super().__init__('NoiseReduction')
        self._parameters = {
            'strength': strength,
            'spatial_sigma': spatial_sigma
        }
    
    def apply(self, image: Image) -> Image:
        if not self._enabled:
            return image
        
        strength = self._parameters['strength'] / 255.0
        spatial = self._parameters['spatial_sigma']
        
        data = image.data
        kernel_size = int(spatial * 3) * 2 + 1
        
        if data.ndim == 3:
            result = np.zeros_like(data)
            for c in range(data.shape[2]):
                result[:, :, c] = self._bilateral_filter(
                    data[:, :, c], kernel_size, spatial, strength
                )
        else:
            result = self._bilateral_filter(data, kernel_size, spatial, strength)
        
        return Image(result, image.color_space)
    
    def _bilateral_filter(self, image: np.ndarray, 
                         kernel_size: int,
                         sigma_space: float,
                         sigma_color: float) -> np.ndarray:
        """Simple bilateral filter implementation."""
        pad = kernel_size // 2
        padded = np.pad(image, pad, mode='reflect')
        
        h, w = image.shape
        result = np.zeros_like(image)
        
        # Precompute spatial weights
        x = np.arange(kernel_size) - pad
        spatial_kernel = np.exp(-x**2 / (2 * sigma_space**2))
        spatial_weights = np.outer(spatial_kernel, spatial_kernel)
        
        for i in range(h):
            for j in range(w):
                center = image[i, j]
                patch = padded[i:i+kernel_size, j:j+kernel_size]
                
                # Color weights
                color_diff = patch - center
                color_weights = np.exp(-color_diff**2 / (2 * sigma_color**2))
                
                # Combined weights
                weights = spatial_weights * color_weights
                weights_sum = weights.sum()
                
                if weights_sum > 0:
                    result[i, j] = (patch * weights).sum() / weights_sum
                else:
                    result[i, j] = center
        
        return result


class ColorCurveOperation(Operation):
    """Apply color curves adjustment."""
    
    def __init__(self, 
                 curve_points: Optional[List[Tuple[float, float]]] = None,
                 channel: str = 'all'):
        super().__init__('ColorCurve')
        self._parameters = {
            'curve_points': curve_points or [(0, 0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1, 1)],
            'channel': channel
        }
    
    def apply(self, image: Image) -> Image:
        if not self._enabled:
            return image
        
        points = self._parameters['curve_points']
        channel = self._parameters['channel']
        
        # Create lookup table from curve
        lut = self._create_lut(points)
        
        data = image.data.copy()
        
        if channel == 'all':
            if data.ndim == 3:
                for c in range(min(3, data.shape[2])):
                    data[:, :, c] = self._apply_lut(data[:, :, c], lut)
            else:
                data = self._apply_lut(data, lut)
        elif channel in ('r', 'red') and data.ndim == 3:
            data[:, :, 0] = self._apply_lut(data[:, :, 0], lut)
        elif channel in ('g', 'green') and data.ndim == 3:
            data[:, :, 1] = self._apply_lut(data[:, :, 1], lut)
        elif channel in ('b', 'blue') and data.ndim == 3:
            data[:, :, 2] = self._apply_lut(data[:, :, 2], lut)
        
        return Image(data, image.color_space)
    
    def _create_lut(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """Create lookup table from curve control points."""
        lut = np.zeros(256)
        
        # Sort points by x
        points = sorted(points, key=lambda p: p[0])
        
        # Linear interpolation between points
        for i in range(256):
            x = i / 255.0
            
            # Find surrounding points
            lower = (0, 0)
            upper = (1, 1)
            
            for j, point in enumerate(points):
                if point[0] <= x:
                    lower = point
                if point[0] >= x:
                    upper = point
                    break
            
            # Interpolate
            if upper[0] == lower[0]:
                lut[i] = lower[1]
            else:
                t = (x - lower[0]) / (upper[0] - lower[0])
                lut[i] = lower[1] + t * (upper[1] - lower[1])
        
        return np.clip(lut, 0, 1)
    
    def _apply_lut(self, data: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """Apply lookup table to data."""
        indices = (np.clip(data, 0, 1) * 255).astype(int)
        return lut[indices]


# ============================================================================
# PROCESSING PIPELINE
# ============================================================================

class Pipeline:
    """
    Image processing pipeline.
    
    Chains multiple operations together with caching and undo support.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.operations: List[Operation] = []
        self._cache: Dict[str, Image] = {}
        self._history: List[Image] = []
        self._max_history = 20
    
    def add(self, operation: Operation) -> 'Pipeline':
        """Add operation to pipeline."""
        self.operations.append(operation)
        logger.info(f"Added operation: {operation}")
        return self
    
    def remove(self, index: int) -> 'Pipeline':
        """Remove operation at index."""
        if 0 <= index < len(self.operations):
            removed = self.operations.pop(index)
            logger.info(f"Removed operation: {removed}")
        return self
    
    def insert(self, index: int, operation: Operation) -> 'Pipeline':
        """Insert operation at index."""
        self.operations.insert(index, operation)
        return self
    
    def clear(self) -> 'Pipeline':
        """Remove all operations."""
        self.operations.clear()
        return self
    
    def process(self, image: Image) -> Image:
        """
        Process image through all operations.
        """
        # Save to history
        self._add_to_history(image)
        
        result = image
        
        for i, operation in enumerate(self.operations):
            if not operation.enabled:
                continue
            
            # Check cache
            cache_key = self._compute_cache_key(result, operation)
            
            if self.config.use_cache and cache_key in self._cache:
                result = self._cache[cache_key]
                logger.debug(f"Cache hit for {operation.name}")
            else:
                logger.info(f"Applying {operation.name}")
                result = operation.apply(result)
                
                if self.config.use_cache:
                    self._cache[cache_key] = result
        
        return result
    
    def _compute_cache_key(self, image: Image, operation: Operation) -> str:
        """Compute cache key for image+operation."""
        image_hash = image.compute_hash()
        op_str = repr(operation)
        return hashlib.md5(f"{image_hash}_{op_str}".encode()).hexdigest()
    
    def _add_to_history(self, image: Image):
        """Add image to undo history."""
        self._history.append(image.copy())
        if len(self._history) > self._max_history:
            self._history.pop(0)
    
    def undo(self) -> Optional[Image]:
        """Get previous image from history."""
        if self._history:
            return self._history.pop()
        return None
    
    def clear_cache(self):
        """Clear operation cache."""
        self._cache.clear()
    
    def get_preview(self, image: Image, 
                   max_size: int = 512) -> Image:
        """Generate quick preview at reduced resolution."""
        # Resize for preview
        scale = min(1.0, max_size / max(image.width, image.height))
        
        if scale < 1.0:
            resize = ResizeOperation(scale=scale)
            preview = resize.apply(image)
        else:
            preview = image.copy()
        
        return self.process(preview)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize pipeline to dictionary."""
        return {
            'config': self.config.to_dict(),
            'operations': [
                {
                    'type': op.__class__.__name__,
                    'name': op.name,
                    'enabled': op.enabled,
                    'parameters': op.parameters
                }
                for op in self.operations
            ]
        }
    
    def save(self, filepath: str):
        """Save pipeline configuration to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Pipeline saved to {filepath}")


# ============================================================================
# BATCH PROCESSING
# ============================================================================

class BatchProcessor:
    """
    Process multiple images with same pipeline.
    """
    
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.results: List[Dict[str, Any]] = []
    
    def process_files(self, 
                     input_paths: List[str],
                     output_dir: str,
                     callback: Optional[Callable[[int, int, str], None]] = None) -> List[str]:
        """
        Process multiple image files.
        
        Args:
            input_paths: List of input file paths
            output_dir: Directory for output files
            callback: Progress callback(current, total, filename)
        
        Returns:
            List of output file paths
        """
        output_paths = []
        total = len(input_paths)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, input_path in enumerate(input_paths):
            try:
                filename = Path(input_path).stem
                output_path = str(Path(output_dir) / f"{filename}_processed.png")
                
                if callback:
                    callback(i, total, filename)
                
                # Load image (simplified - real implementation would use Pillow/OpenCV)
                image = self._load_image(input_path)
                
                # Process
                result = self.pipeline.process(image)
                
                # Save
                self._save_image(result, output_path)
                
                output_paths.append(output_path)
                
                self.results.append({
                    'input': input_path,
                    'output': output_path,
                    'status': 'success'
                })
                
            except Exception as e:
                logger.error(f"Failed to process {input_path}: {e}")
                self.results.append({
                    'input': input_path,
                    'output': None,
                    'status': 'error',
                    'error': str(e)
                })
        
        return output_paths
    
    def _load_image(self, path: str) -> Image:
        """Load image from file (placeholder)."""
        # In real implementation, use Pillow or OpenCV
        # This is a placeholder that creates a test image
        return Image(np.random.rand(512, 512, 3).astype(np.float32))
    
    def _save_image(self, image: Image, path: str):
        """Save image to file (placeholder)."""
        # In real implementation, use Pillow or OpenCV
        logger.info(f"Would save image to {path}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def create_example_pipeline() -> Pipeline:
    """Create example processing pipeline."""
    pipeline = Pipeline()
    
    # Add operations
    pipeline.add(ResizeOperation(width=1920, height=1080))
    pipeline.add(BrightnessContrastOperation(brightness=0.05, contrast=1.1))
    pipeline.add(SaturationOperation(saturation=1.2))
    pipeline.add(SharpenOperation(amount=0.5, radius=1.0))
    pipeline.add(NoiseReductionOperation(strength=5.0))
    
    return pipeline


def main():
    """Example usage."""
    # Create pipeline
    pipeline = create_example_pipeline()
    
    # Create test image
    test_image = Image(np.random.rand(800, 600, 3).astype(np.float32))
    
    # Process
    result = pipeline.process(test_image)
    
    print(f"Input: {test_image}")
    print(f"Output: {result}")
    
    # Save pipeline configuration
    pipeline.save("pipeline_config.json")


if __name__ == "__main__":
    main()
```

### 27.2 Procedural Texture Generator Template

Complete template for generating procedural textures.

```python
"""
Procedural Texture Generator
============================

Complete system for generating tileable procedural textures
with extensive customization options.
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json


# ============================================================================
# CONFIGURATION
# ============================================================================

class BlendMode(Enum):
    """Layer blend modes."""
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    SOFT_LIGHT = "soft_light"
    HARD_LIGHT = "hard_light"
    ADD = "add"
    SUBTRACT = "subtract"


@dataclass
class TextureConfig:
    """Configuration for texture generation."""
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    tileable: bool = True
    output_format: str = "png"
    bit_depth: int = 8


# ============================================================================
# NOISE GENERATORS
# ============================================================================

class NoiseGenerator(ABC):
    """Base class for noise generators."""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._rng = np.random.default_rng(seed)
    
    @abstractmethod
    def generate(self, width: int, height: int) -> np.ndarray:
        """Generate noise array."""
        pass


class PerlinNoiseGenerator(NoiseGenerator):
    """Perlin noise generator with tileable support."""
    
    def __init__(self, 
                 seed: Optional[int] = None,
                 scale: float = 4.0,
                 octaves: int = 4,
                 persistence: float = 0.5,
                 lacunarity: float = 2.0):
        super().__init__(seed)
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        
        # Generate permutation table
        self._perm = np.arange(256, dtype=np.int32)
        self._rng.shuffle(self._perm)
        self._perm = np.tile(self._perm, 2)
    
    def _fade(self, t: np.ndarray) -> np.ndarray:
        """Smoothstep interpolation."""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def _lerp(self, a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Linear interpolation."""
        return a + t * (b - a)
    
    def _grad(self, hash_val: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Gradient dot product."""
        h = hash_val & 3
        result = np.zeros_like(x)
        
        mask0 = h == 0
        mask1 = h == 1
        mask2 = h == 2
        mask3 = h == 3
        
        result[mask0] = x[mask0] + y[mask0]
        result[mask1] = -x[mask1] + y[mask1]
        result[mask2] = x[mask2] - y[mask2]
        result[mask3] = -x[mask3] - y[mask3]
        
        return result
    
    def _perlin(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate single octave of Perlin noise."""
        # Integer coordinates
        xi = x.astype(np.int32) & 255
        yi = y.astype(np.int32) & 255
        
        # Fractional coordinates
        xf = x - x.astype(np.int32)
        yf = y - y.astype(np.int32)
        
        # Fade curves
        u = self._fade(xf)
        v = self._fade(yf)
        
        # Hash coordinates
        aa = self._perm[self._perm[xi] + yi]
        ab = self._perm[self._perm[xi] + yi + 1]
        ba = self._perm[self._perm[xi + 1] + yi]
        bb = self._perm[self._perm[xi + 1] + yi + 1]
        
        # Gradient dot products
        x1 = self._lerp(
            self._grad(aa, xf, yf),
            self._grad(ba, xf - 1, yf),
            u
        )
        x2 = self._lerp(
            self._grad(ab, xf, yf - 1),
            self._grad(bb, xf - 1, yf - 1),
            u
        )
        
        return self._lerp(x1, x2, v)
    
    def generate(self, width: int, height: int, tileable: bool = True) -> np.ndarray:
        """Generate Perlin noise texture."""
        noise = np.zeros((height, width))
        
        amplitude = 1.0
        frequency = self.scale
        max_value = 0.0
        
        for _ in range(self.octaves):
            # Create coordinate grids
            if tileable:
                # Use sinusoidal mapping for tileable noise
                u = np.linspace(0, 1, width, endpoint=False)
                v = np.linspace(0, 1, height, endpoint=False)
                U, V = np.meshgrid(u, v)
                
                # Map to torus
                angle_u = U * 2 * np.pi
                angle_v = V * 2 * np.pi
                
                x = frequency * (1 + np.cos(angle_u))
                y = frequency * (1 + np.sin(angle_u))
                z = frequency * (1 + np.cos(angle_v))
                w = frequency * (1 + np.sin(angle_v))
                
                # Use 2D slice of 4D noise for seamless tiling
                # Simplified: use x+z and y+w as coordinates
                coords_x = x + z
                coords_y = y + w
            else:
                x = np.linspace(0, frequency, width, endpoint=False)
                y = np.linspace(0, frequency, height, endpoint=False)
                coords_x, coords_y = np.meshgrid(x, y)
            
            noise += amplitude * self._perlin(coords_x, coords_y)
            max_value += amplitude
            
            amplitude *= self.persistence
            frequency *= self.lacunarity
        
        # Normalize to [0, 1]
        noise = (noise / max_value + 1) / 2
        
        return np.clip(noise, 0, 1)


class WorleyNoiseGenerator(NoiseGenerator):
    """Worley (cellular) noise generator."""
    
    def __init__(self,
                 seed: Optional[int] = None,
                 n_points: int = 64,
                 distance_type: str = 'euclidean',
                 feature_type: str = 'f1'):
        super().__init__(seed)
        self.n_points = n_points
        self.distance_type = distance_type
        self.feature_type = feature_type
    
    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Compute distance based on type."""
        diff = p1 - p2
        
        if self.distance_type == 'euclidean':
            return np.sqrt(np.sum(diff ** 2, axis=-1))
        elif self.distance_type == 'manhattan':
            return np.sum(np.abs(diff), axis=-1)
        elif self.distance_type == 'chebyshev':
            return np.max(np.abs(diff), axis=-1)
        
        return np.sqrt(np.sum(diff ** 2, axis=-1))
    
    def generate(self, width: int, height: int, tileable: bool = True) -> np.ndarray:
        """Generate Worley noise texture."""
        # Generate random points
        points = self._rng.random((self.n_points, 2))
        
        if tileable:
            # Tile points for seamless edges
            tiled_points = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    tiled_points.append(points + np.array([dx, dy]))
            points = np.vstack(tiled_points)
        
        # Create coordinate grid
        x = np.linspace(0, 1, width, endpoint=False)
        y = np.linspace(0, 1, height, endpoint=False)
        X, Y = np.meshgrid(x, y)
        coords = np.stack([X, Y], axis=-1)
        
        # Compute distances to all points
        distances = np.zeros((height, width, len(points)))
        for i, point in enumerate(points):
            distances[:, :, i] = self._distance(coords, point)
        
        # Sort distances
        distances.sort(axis=-1)
        
        # Select feature
        if self.feature_type == 'f1':
            result = distances[:, :, 0]
        elif self.feature_type == 'f2':
            result = distances[:, :, 1]
        elif self.feature_type == 'f2-f1':
            result = distances[:, :, 1] - distances[:, :, 0]
        else:
            result = distances[:, :, 0]
        
        # Normalize
        result = (result - result.min()) / (result.max() - result.min() + 1e-10)
        
        return result


class SimplexNoiseGenerator(NoiseGenerator):
    """Simplex noise generator."""
    
    def __init__(self,
                 seed: Optional[int] = None,
                 scale: float = 4.0,
                 octaves: int = 4):
        super().__init__(seed)
        self.scale = scale
        self.octaves = octaves
        
        # Skewing factors for 2D
        self.F2 = 0.5 * (np.sqrt(3.0) - 1.0)
        self.G2 = (3.0 - np.sqrt(3.0)) / 6.0
        
        # Gradient vectors
        self.gradients = np.array([
            [1, 1], [-1, 1], [1, -1], [-1, -1],
            [1, 0], [-1, 0], [0, 1], [0, -1]
        ], dtype=np.float32)
        
        # Permutation table
        self._perm = np.arange(256, dtype=np.int32)
        self._rng.shuffle(self._perm)
        self._perm = np.tile(self._perm, 2)
    
    def _simplex_2d(self, x: float, y: float) -> float:
        """Single 2D simplex noise value."""
        # Skew to simplex space
        s = (x + y) * self.F2
        i = int(np.floor(x + s))
        j = int(np.floor(y + s))
        
        # Unskew
        t = (i + j) * self.G2
        X0 = i - t
        Y0 = j - t
        x0 = x - X0
        y0 = y - Y0
        
        # Determine simplex
        if x0 > y0:
            i1, j1 = 1, 0
        else:
            i1, j1 = 0, 1
        
        x1 = x0 - i1 + self.G2
        y1 = y0 - j1 + self.G2
        x2 = x0 - 1.0 + 2.0 * self.G2
        y2 = y0 - 1.0 + 2.0 * self.G2
        
        # Hash coordinates
        ii = i & 255
        jj = j & 255
        
        gi0 = self._perm[ii + self._perm[jj]] % 8
        gi1 = self._perm[ii + i1 + self._perm[jj + j1]] % 8
        gi2 = self._perm[ii + 1 + self._perm[jj + 1]] % 8
        
        # Contributions
        n0 = n1 = n2 = 0.0
        
        t0 = 0.5 - x0*x0 - y0*y0
        if t0 >= 0:
            t0 *= t0
            n0 = t0 * t0 * np.dot(self.gradients[gi0], [x0, y0])
        
        t1 = 0.5 - x1*x1 - y1*y1
        if t1 >= 0:
            t1 *= t1
            n1 = t1 * t1 * np.dot(self.gradients[gi1], [x1, y1])
        
        t2 = 0.5 - x2*x2 - y2*y2
        if t2 >= 0:
            t2 *= t2
            n2 = t2 * t2 * np.dot(self.gradients[gi2], [x2, y2])
        
        return 70.0 * (n0 + n1 + n2)
    
    def generate(self, width: int, height: int, tileable: bool = True) -> np.ndarray:
        """Generate simplex noise texture."""
        noise = np.zeros((height, width))
        
        for i in range(height):
            for j in range(width):
                x = j / width * self.scale
                y = i / height * self.scale
                
                value = 0.0
                amplitude = 1.0
                frequency = 1.0
                max_val = 0.0
                
                for _ in range(self.octaves):
                    value += amplitude * self._simplex_2d(x * frequency, y * frequency)
                    max_val += amplitude
                    amplitude *= 0.5
                    frequency *= 2.0
                
                noise[i, j] = value / max_val
        
        # Normalize to [0, 1]
        noise = (noise + 1) / 2
        return np.clip(noise, 0, 1)


# ============================================================================
# TEXTURE LAYERS
# ============================================================================

@dataclass
class LayerParams:
    """Parameters for a texture layer."""
    name: str = "Layer"
    opacity: float = 1.0
    blend_mode: BlendMode = BlendMode.NORMAL
    visible: bool = True
    locked: bool = False


class TextureLayer:
    """Single layer in texture composition."""
    
    def __init__(self, 
                 data: np.ndarray,
                 params: Optional[LayerParams] = None):
        self.data = data.astype(np.float32)
        self.params = params or LayerParams()
        
        # Ensure 3 channels for consistent blending
        if self.data.ndim == 2:
            self.data = np.stack([self.data] * 3, axis=-1)
    
    @property
    def name(self) -> str:
        return self.params.name
    
    @property
    def opacity(self) -> float:
        return self.params.opacity
    
    @opacity.setter
    def opacity(self, value: float):
        self.params.opacity = np.clip(value, 0, 1)
    
    @property
    def blend_mode(self) -> BlendMode:
        return self.params.blend_mode
    
    @blend_mode.setter
    def blend_mode(self, mode: BlendMode):
        self.params.blend_mode = mode


class LayerStack:
    """Stack of texture layers with blending."""
    
    def __init__(self):
        self.layers: List[TextureLayer] = []
    
    def add_layer(self, layer: TextureLayer):
        """Add layer to top of stack."""
        self.layers.append(layer)
    
    def insert_layer(self, index: int, layer: TextureLayer):
        """Insert layer at index."""
        self.layers.insert(index, layer)
    
    def remove_layer(self, index: int):
        """Remove layer at index."""
        if 0 <= index < len(self.layers):
            self.layers.pop(index)
    
    def move_layer(self, from_index: int, to_index: int):
        """Move layer to new position."""
        if 0 <= from_index < len(self.layers):
            layer = self.layers.pop(from_index)
            self.layers.insert(to_index, layer)
    
    def _blend(self, base: np.ndarray, top: np.ndarray, 
               mode: BlendMode, opacity: float) -> np.ndarray:
        """Blend two layers."""
        if mode == BlendMode.NORMAL:
            result = top
        
        elif mode == BlendMode.MULTIPLY:
            result = base * top
        
        elif mode == BlendMode.SCREEN:
            result = 1 - (1 - base) * (1 - top)
        
        elif mode == BlendMode.OVERLAY:
            mask = base < 0.5
            result = np.where(
                mask,
                2 * base * top,
                1 - 2 * (1 - base) * (1 - top)
            )
        
        elif mode == BlendMode.SOFT_LIGHT:
            result = (1 - 2 * top) * base * base + 2 * top * base
        
        elif mode == BlendMode.HARD_LIGHT:
            mask = top < 0.5
            result = np.where(
                mask,
                2 * base * top,
                1 - 2 * (1 - base) * (1 - top)
            )
        
        elif mode == BlendMode.ADD:
            result = base + top
        
        elif mode == BlendMode.SUBTRACT:
            result = base - top
        
        else:
            result = top
        
        # Apply opacity
        result = base + opacity * (result - base)
        
        return np.clip(result, 0, 1)
    
    def flatten(self) -> np.ndarray:
        """Flatten all layers into single image."""
        if not self.layers:
            return np.zeros((512, 512, 3))
        
        # Start with first visible layer
        result = None
        
        for layer in self.layers:
            if not layer.params.visible:
                continue
            
            if result is None:
                result = layer.data * layer.opacity
            else:
                result = self._blend(
                    result,
                    layer.data,
                    layer.blend_mode,
                    layer.opacity
                )
        
        return result if result is not None else np.zeros_like(self.layers[0].data)


# ============================================================================
# TEXTURE GENERATOR
# ============================================================================

class ProceduralTextureGenerator:
    """
    Complete procedural texture generation system.
    """
    
    def __init__(self, config: Optional[TextureConfig] = None):
        self.config = config or TextureConfig()
        self.layer_stack = LayerStack()
        
        # Set random seed
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
    
    def add_noise_layer(self,
                       noise_type: str = 'perlin',
                       params: Optional[Dict[str, Any]] = None,
                       layer_params: Optional[LayerParams] = None) -> 'ProceduralTextureGenerator':
        """Add noise-based layer."""
        params = params or {}
        
        # Create noise generator
        if noise_type == 'perlin':
            generator = PerlinNoiseGenerator(
                seed=self.config.seed,
                scale=params.get('scale', 4.0),
                octaves=params.get('octaves', 4),
                persistence=params.get('persistence', 0.5),
                lacunarity=params.get('lacunarity', 2.0)
            )
        elif noise_type == 'worley':
            generator = WorleyNoiseGenerator(
                seed=self.config.seed,
                n_points=params.get('n_points', 64),
                distance_type=params.get('distance_type', 'euclidean'),
                feature_type=params.get('feature_type', 'f1')
            )
        elif noise_type == 'simplex':
            generator = SimplexNoiseGenerator(
                seed=self.config.seed,
                scale=params.get('scale', 4.0),
                octaves=params.get('octaves', 4)
            )
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Generate noise
        noise = generator.generate(
            self.config.width,
            self.config.height,
            tileable=self.config.tileable
        )
        
        # Create layer
        layer = TextureLayer(noise, layer_params)
        self.layer_stack.add_layer(layer)
        
        return self
    
    def add_gradient_layer(self,
                          gradient_type: str = 'linear',
                          colors: List[Tuple[float, Tuple[int, int, int]]] = None,
                          angle: float = 0.0,
                          layer_params: Optional[LayerParams] = None) -> 'ProceduralTextureGenerator':
        """Add gradient layer."""
        h, w = self.config.height, self.config.width
        
        # Default colors
        if colors is None:
            colors = [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))]
        
        # Create coordinate grid
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        X, Y = np.meshgrid(x, y)
        
        # Apply rotation
        if angle != 0:
            cos_a = np.cos(np.radians(angle))
            sin_a = np.sin(np.radians(angle))
            X_rot = X * cos_a - Y * sin_a
            Y_rot = X * sin_a + Y * cos_a
        else:
            X_rot, Y_rot = X, Y
        
        # Generate gradient value
        if gradient_type == 'linear':
            t = X_rot
        elif gradient_type == 'radial':
            cx, cy = 0.5, 0.5
            t = np.sqrt((X_rot - cx) ** 2 + (Y_rot - cy) ** 2) * np.sqrt(2)
        elif gradient_type == 'angular':
            cx, cy = 0.5, 0.5
            t = (np.arctan2(Y_rot - cy, X_rot - cx) + np.pi) / (2 * np.pi)
        else:
            t = X_rot
        
        t = np.clip(t, 0, 1)
        
        # Interpolate colors
        gradient = np.zeros((h, w, 3))
        
        for i in range(len(colors) - 1):
            t1, c1 = colors[i]
            t2, c2 = colors[i + 1]
            
            mask = (t >= t1) & (t <= t2)
            if not np.any(mask):
                continue
            
            local_t = (t[mask] - t1) / (t2 - t1)
            
            for c in range(3):
                gradient[mask, c] = c1[c] + local_t * (c2[c] - c1[c])
        
        gradient = gradient / 255.0
        
        layer = TextureLayer(gradient, layer_params)
        self.layer_stack.add_layer(layer)
        
        return self
    
    def add_pattern_layer(self,
                         pattern_type: str = 'checker',
                         params: Optional[Dict[str, Any]] = None,
                         layer_params: Optional[LayerParams] = None) -> 'ProceduralTextureGenerator':
        """Add geometric pattern layer."""
        params = params or {}
        h, w = self.config.height, self.config.width
        
        x = np.linspace(0, 1, w, endpoint=False)
        y = np.linspace(0, 1, h, endpoint=False)
        X, Y = np.meshgrid(x, y)
        
        if pattern_type == 'checker':
            size = params.get('size', 8)
            pattern = ((X * size).astype(int) + (Y * size).astype(int)) % 2
        
        elif pattern_type == 'stripes':
            size = params.get('size', 16)
            angle = params.get('angle', 0)
            X_rot = X * np.cos(np.radians(angle)) - Y * np.sin(np.radians(angle))
            pattern = (X_rot * size).astype(int) % 2
        
        elif pattern_type == 'dots':
            size = params.get('size', 32)
            radius = params.get('radius', 0.3)
            Xi = (X * size) % 1 - 0.5
            Yi = (Y * size) % 1 - 0.5
            pattern = (Xi ** 2 + Yi ** 2 < radius ** 2).astype(float)
        
        elif pattern_type == 'hexagons':
            size = params.get('size', 8)
            # Hexagonal grid
            sqrt3 = np.sqrt(3)
            q = (2/3 * X * size)
            r = (-1/3 * X + sqrt3/3 * Y) * size
            
            # Round to nearest hexagon
            x_hex = np.round(q)
            y_hex = np.round(r)
            
            pattern = ((x_hex + y_hex).astype(int) % 3) / 2.0
        
        else:
            pattern = np.zeros((h, w))
        
        layer = TextureLayer(pattern.astype(np.float32), layer_params)
        self.layer_stack.add_layer(layer)
        
        return self
    
    def apply_filter(self, 
                    filter_type: str,
                    params: Optional[Dict[str, Any]] = None) -> 'ProceduralTextureGenerator':
        """Apply filter to entire stack."""
        params = params or {}
        
        # Flatten current stack
        result = self.layer_stack.flatten()
        
        if filter_type == 'levels':
            black = params.get('black', 0.0)
            white = params.get('white', 1.0)
            gamma = params.get('gamma', 1.0)
            
            result = (result - black) / (white - black)
            result = np.clip(result, 0, 1)
            result = np.power(result, 1 / gamma)
        
        elif filter_type == 'invert':
            result = 1 - result
        
        elif filter_type == 'threshold':
            threshold = params.get('threshold', 0.5)
            result = (result > threshold).astype(float)
        
        elif filter_type == 'colorize':
            color = np.array(params.get('color', [255, 200, 100])) / 255.0
            gray = np.mean(result, axis=2, keepdims=True)
            result = gray * color
        
        elif filter_type == 'emboss':
            gray = np.mean(result, axis=2)
            kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            
            h, w = gray.shape
            embossed = np.zeros_like(gray)
            
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    embossed[i, j] = np.sum(gray[i-1:i+2, j-1:j+2] * kernel)
            
            embossed = (embossed + 1) / 2
            result = np.stack([embossed] * 3, axis=-1)
        
        # Clear stack and add result
        self.layer_stack.layers.clear()
        self.layer_stack.add_layer(TextureLayer(result))
        
        return self
    
    def generate(self) -> np.ndarray:
        """Generate final texture."""
        return self.layer_stack.flatten()
    
    def to_uint8(self) -> np.ndarray:
        """Generate texture as uint8."""
        return (np.clip(self.generate(), 0, 1) * 255).astype(np.uint8)


# ============================================================================
# PRESET TEXTURES
# ============================================================================

class TexturePresets:
    """Collection of preset texture generators."""
    
    @staticmethod
    def marble(config: Optional[TextureConfig] = None) -> np.ndarray:
        """Generate marble texture."""
        gen = ProceduralTextureGenerator(config)
        
        gen.add_noise_layer('perlin', {
            'scale': 2.0, 'octaves': 6, 'persistence': 0.6
        }, LayerParams(name='Base Marble'))
        
        gen.add_noise_layer('perlin', {
            'scale': 8.0, 'octaves': 4, 'persistence': 0.4
        }, LayerParams(name='Veins', opacity=0.3, blend_mode=BlendMode.OVERLAY))
        
        gen.apply_filter('colorize', {'color': [240, 230, 220]})
        
        return gen.generate()
    
    @staticmethod
    def wood(config: Optional[TextureConfig] = None) -> np.ndarray:
        """Generate wood grain texture."""
        gen = ProceduralTextureGenerator(config)
        config = config or TextureConfig()
        
        # Base color
        gen.add_gradient_layer('linear', [
            (0.0, (139, 90, 43)),
            (1.0, (160, 120, 60))
        ])
        
        # Add rings using noise
        gen.add_noise_layer('perlin', {
            'scale': 1.0, 'octaves': 1
        }, LayerParams(name='Rings', opacity=0.4, blend_mode=BlendMode.MULTIPLY))
        
        # Add grain detail
        gen.add_noise_layer('perlin', {
            'scale': 16.0, 'octaves': 2, 'persistence': 0.3
        }, LayerParams(name='Grain', opacity=0.2, blend_mode=BlendMode.OVERLAY))
        
        return gen.generate()
    
    @staticmethod
    def brick(config: Optional[TextureConfig] = None) -> np.ndarray:
        """Generate brick texture."""
        gen = ProceduralTextureGenerator(config)
        config = config or TextureConfig()
        h, w = config.height, config.width
        
        # Brick pattern
        brick_h = h // 8
        brick_w = w // 4
        
        pattern = np.zeros((h, w))
        
        for row in range(8):
            offset = (brick_w // 2) if row % 2 else 0
            for col in range(5):
                y_start = row * brick_h
                y_end = min((row + 1) * brick_h - 2, h)
                x_start = (col * brick_w + offset) % w
                x_end = min(x_start + brick_w - 2, w)
                
                if y_end > y_start and x_end > x_start:
                    pattern[y_start:y_end, x_start:x_end] = 1
        
        gen.layer_stack.add_layer(TextureLayer(
            pattern * 0.7 + 0.3,
            LayerParams(name='Brick Pattern')
        ))
        
        # Add color variation
        gen.add_noise_layer('perlin', {
            'scale': 8.0, 'octaves': 2
        }, LayerParams(name='Color Variation', opacity=0.3, blend_mode=BlendMode.OVERLAY))
        
        gen.apply_filter('colorize', {'color': [180, 80, 60]})
        
        return gen.generate()
    
    @staticmethod
    def stone(config: Optional[TextureConfig] = None) -> np.ndarray:
        """Generate stone/rock texture."""
        gen = ProceduralTextureGenerator(config)
        
        # Base cellular noise
        gen.add_noise_layer('worley', {
            'n_points': 32, 'feature_type': 'f1'
        }, LayerParams(name='Base Stone'))
        
        # Add surface detail
        gen.add_noise_layer('perlin', {
            'scale': 16.0, 'octaves': 4, 'persistence': 0.5
        }, LayerParams(name='Surface', opacity=0.4, blend_mode=BlendMode.OVERLAY))
        
        gen.apply_filter('colorize', {'color': [128, 128, 128]})
        
        return gen.generate()


# ============================================================================
# USAGE
# ============================================================================

def main():
    """Example usage."""
    config = TextureConfig(width=512, height=512, seed=42, tileable=True)
    
    # Custom texture
    gen = ProceduralTextureGenerator(config)
    gen.add_gradient_layer('radial', [
        (0.0, (50, 50, 80)),
        (1.0, (20, 20, 40))
    ])
    gen.add_noise_layer('perlin', {'scale': 4.0, 'octaves': 6},
                       LayerParams(opacity=0.3, blend_mode=BlendMode.SCREEN))
    gen.add_noise_layer('worley', {'n_points': 64, 'feature_type': 'f1'},
                       LayerParams(opacity=0.2, blend_mode=BlendMode.OVERLAY))
    
    texture = gen.to_uint8()
    print(f"Generated texture: {texture.shape}")
    
    # Presets
    marble = TexturePresets.marble(config)
    wood = TexturePresets.wood(config)
    brick = TexturePresets.brick(config)
    stone = TexturePresets.stone(config)
    
    print("Generated preset textures")


if __name__ == "__main__":
    main()
```


## Chapter 28: Quick Reference

### 28.1 Mathematical Functions Reference

```python
"""
Mathematical Functions Quick Reference
======================================

Essential mathematical functions for procedural art.
"""

import numpy as np
from typing import Callable, Tuple


# ============================================================================
# INTERPOLATION FUNCTIONS
# ============================================================================

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + t * (b - a)


def inverse_lerp(a: float, b: float, v: float) -> float:
    """Inverse linear interpolation (find t)."""
    return (v - a) / (b - a) if b != a else 0


def remap(value: float, in_min: float, in_max: float, 
          out_min: float, out_max: float) -> float:
    """Remap value from one range to another."""
    t = inverse_lerp(in_min, in_max, value)
    return lerp(out_min, out_max, t)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smooth Hermite interpolation."""
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(edge0: float, edge1: float, x: float) -> float:
    """Ken Perlin's improved smoothstep."""
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def cosine_interpolate(a: float, b: float, t: float) -> float:
    """Cosine interpolation (smoother than linear)."""
    t2 = (1 - np.cos(t * np.pi)) / 2
    return a * (1 - t2) + b * t2


def cubic_interpolate(y0: float, y1: float, y2: float, y3: float, t: float) -> float:
    """Cubic interpolation (using 4 points)."""
    a0 = y3 - y2 - y0 + y1
    a1 = y0 - y1 - a0
    a2 = y2 - y0
    a3 = y1
    return a0 * t**3 + a1 * t**2 + a2 * t + a3


def catmull_rom(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    """Catmull-Rom spline interpolation."""
    return 0.5 * (
        2 * p1 +
        (-p0 + p2) * t +
        (2*p0 - 5*p1 + 4*p2 - p3) * t**2 +
        (-p0 + 3*p1 - 3*p2 + p3) * t**3
    )


def bezier_quadratic(p0: float, p1: float, p2: float, t: float) -> float:
    """Quadratic Bezier interpolation."""
    return (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2


def bezier_cubic(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    """Cubic Bezier interpolation."""
    return (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3


# ============================================================================
# EASING FUNCTIONS
# ============================================================================

def ease_in_quad(t: float) -> float:
    """Quadratic ease-in."""
    return t * t


def ease_out_quad(t: float) -> float:
    """Quadratic ease-out."""
    return t * (2 - t)


def ease_in_out_quad(t: float) -> float:
    """Quadratic ease-in-out."""
    return 2*t*t if t < 0.5 else -1 + (4 - 2*t) * t


def ease_in_cubic(t: float) -> float:
    """Cubic ease-in."""
    return t * t * t


def ease_out_cubic(t: float) -> float:
    """Cubic ease-out."""
    return (t - 1)**3 + 1


def ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out."""
    return 4*t*t*t if t < 0.5 else (t - 1) * (2*t - 2)**2 + 1


def ease_in_sine(t: float) -> float:
    """Sine ease-in."""
    return 1 - np.cos(t * np.pi / 2)


def ease_out_sine(t: float) -> float:
    """Sine ease-out."""
    return np.sin(t * np.pi / 2)


def ease_in_out_sine(t: float) -> float:
    """Sine ease-in-out."""
    return -(np.cos(np.pi * t) - 1) / 2


def ease_in_expo(t: float) -> float:
    """Exponential ease-in."""
    return 0 if t == 0 else 2**(10 * (t - 1))


def ease_out_expo(t: float) -> float:
    """Exponential ease-out."""
    return 1 if t == 1 else 1 - 2**(-10 * t)


def ease_in_out_expo(t: float) -> float:
    """Exponential ease-in-out."""
    if t == 0: return 0
    if t == 1: return 1
    return 2**(20*t - 10) / 2 if t < 0.5 else (2 - 2**(-20*t + 10)) / 2


def ease_in_elastic(t: float) -> float:
    """Elastic ease-in."""
    c4 = (2 * np.pi) / 3
    if t == 0: return 0
    if t == 1: return 1
    return -2**(10*t - 10) * np.sin((t*10 - 10.75) * c4)


def ease_out_elastic(t: float) -> float:
    """Elastic ease-out."""
    c4 = (2 * np.pi) / 3
    if t == 0: return 0
    if t == 1: return 1
    return 2**(-10*t) * np.sin((t*10 - 0.75) * c4) + 1


def ease_in_bounce(t: float) -> float:
    """Bounce ease-in."""
    return 1 - ease_out_bounce(1 - t)


def ease_out_bounce(t: float) -> float:
    """Bounce ease-out."""
    n1 = 7.5625
    d1 = 2.75
    
    if t < 1/d1:
        return n1 * t * t
    elif t < 2/d1:
        t -= 1.5/d1
        return n1 * t * t + 0.75
    elif t < 2.5/d1:
        t -= 2.25/d1
        return n1 * t * t + 0.9375
    else:
        t -= 2.625/d1
        return n1 * t * t + 0.984375


# ============================================================================
# TRIGONOMETRIC UTILITIES
# ============================================================================

def deg_to_rad(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * np.pi / 180


def rad_to_deg(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * 180 / np.pi


def normalize_angle(angle: float) -> float:
    """Normalize angle to [0, 2π)."""
    return angle % (2 * np.pi)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two vectors in radians."""
    dot = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    return np.arccos(np.clip(dot / (mag1 * mag2), -1, 1))


def rotate_2d(point: Tuple[float, float], angle: float) -> Tuple[float, float]:
    """Rotate 2D point around origin."""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x, y = point
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)


def rotate_around(point: Tuple[float, float], center: Tuple[float, float],
                  angle: float) -> Tuple[float, float]:
    """Rotate point around center."""
    px, py = point
    cx, cy = center
    translated = (px - cx, py - cy)
    rotated = rotate_2d(translated, angle)
    return (rotated[0] + cx, rotated[1] + cy)


# ============================================================================
# VECTOR OPERATIONS
# ============================================================================

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    mag = np.linalg.norm(v)
    return v / mag if mag > 0 else v


def reflect(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Reflect vector across normal."""
    normal = normalize(normal)
    return v - 2 * np.dot(v, normal) * normal


def project(v: np.ndarray, onto: np.ndarray) -> np.ndarray:
    """Project vector onto another vector."""
    onto_normalized = normalize(onto)
    return np.dot(v, onto_normalized) * onto_normalized


def perpendicular_2d(v: np.ndarray) -> np.ndarray:
    """Get perpendicular vector in 2D."""
    return np.array([-v[1], v[0]])


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between points."""
    return np.linalg.norm(p2 - p1)


def manhattan_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Manhattan distance between points."""
    return np.sum(np.abs(p2 - p1))


def chebyshev_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Chebyshev distance between points."""
    return np.max(np.abs(p2 - p1))


# ============================================================================
# RANDOM UTILITIES
# ============================================================================

def random_in_circle(radius: float = 1.0) -> Tuple[float, float]:
    """Random point inside unit circle."""
    angle = np.random.random() * 2 * np.pi
    r = np.sqrt(np.random.random()) * radius
    return (r * np.cos(angle), r * np.sin(angle))


def random_on_circle(radius: float = 1.0) -> Tuple[float, float]:
    """Random point on circle perimeter."""
    angle = np.random.random() * 2 * np.pi
    return (radius * np.cos(angle), radius * np.sin(angle))


def random_in_sphere(radius: float = 1.0) -> Tuple[float, float, float]:
    """Random point inside sphere."""
    phi = np.random.random() * 2 * np.pi
    cos_theta = 2 * np.random.random() - 1
    sin_theta = np.sqrt(1 - cos_theta**2)
    r = np.cbrt(np.random.random()) * radius
    return (
        r * sin_theta * np.cos(phi),
        r * sin_theta * np.sin(phi),
        r * cos_theta
    )


def random_on_sphere(radius: float = 1.0) -> Tuple[float, float, float]:
    """Random point on sphere surface."""
    phi = np.random.random() * 2 * np.pi
    cos_theta = 2 * np.random.random() - 1
    sin_theta = np.sqrt(1 - cos_theta**2)
    return (
        radius * sin_theta * np.cos(phi),
        radius * sin_theta * np.sin(phi),
        radius * cos_theta
    )


def poisson_disk_sampling(width: float, height: float, 
                         min_distance: float, k: int = 30) -> list:
    """Generate blue noise distribution of points."""
    cell_size = min_distance / np.sqrt(2)
    grid_w = int(np.ceil(width / cell_size))
    grid_h = int(np.ceil(height / cell_size))
    grid = {}
    
    points = []
    active = []
    
    # First point
    x = np.random.random() * width
    y = np.random.random() * height
    points.append((x, y))
    active.append(0)
    grid[(int(x / cell_size), int(y / cell_size))] = 0
    
    while active:
        idx = np.random.randint(len(active))
        p = points[active[idx]]
        
        found = False
        for _ in range(k):
            angle = np.random.random() * 2 * np.pi
            dist = np.random.uniform(min_distance, 2 * min_distance)
            new_x = p[0] + dist * np.cos(angle)
            new_y = p[1] + dist * np.sin(angle)
            
            if 0 <= new_x < width and 0 <= new_y < height:
                cell = (int(new_x / cell_size), int(new_y / cell_size))
                
                valid = True
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        neighbor_cell = (cell[0] + di, cell[1] + dj)
                        if neighbor_cell in grid:
                            neighbor = points[grid[neighbor_cell]]
                            if distance(np.array([new_x, new_y]), 
                                       np.array(neighbor)) < min_distance:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    points.append((new_x, new_y))
                    active.append(len(points) - 1)
                    grid[cell] = len(points) - 1
                    found = True
                    break
        
        if not found:
            active.pop(idx)
    
    return points


# ============================================================================
# COLOR UTILITIES
# ============================================================================

def rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """RGB to HSV conversion (all values 0-1)."""
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    diff = max_c - min_c
    
    # Value
    v = max_c
    
    # Saturation
    s = 0 if max_c == 0 else diff / max_c
    
    # Hue
    if diff == 0:
        h = 0
    elif max_c == r:
        h = (g - b) / diff % 6
    elif max_c == g:
        h = (b - r) / diff + 2
    else:
        h = (r - g) / diff + 4
    
    h /= 6
    
    return (h, s, v)


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """HSV to RGB conversion (all values 0-1)."""
    if s == 0:
        return (v, v, v)
    
    h *= 6
    i = int(h)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    
    if i == 0: return (v, t, p)
    elif i == 1: return (q, v, p)
    elif i == 2: return (p, v, t)
    elif i == 3: return (p, q, v)
    elif i == 4: return (t, p, v)
    else: return (v, p, q)


def rgb_to_hsl(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """RGB to HSL conversion."""
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    l = (max_c + min_c) / 2
    
    if max_c == min_c:
        h = s = 0
    else:
        diff = max_c - min_c
        s = diff / (2 - max_c - min_c) if l > 0.5 else diff / (max_c + min_c)
        
        if max_c == r:
            h = (g - b) / diff + (6 if g < b else 0)
        elif max_c == g:
            h = (b - r) / diff + 2
        else:
            h = (r - g) / diff + 4
        
        h /= 6
    
    return (h, s, l)


def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
    """HSL to RGB conversion."""
    if s == 0:
        return (l, l, l)
    
    def hue_to_rgb(p, q, t):
        if t < 0: t += 1
        if t > 1: t -= 1
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: return p + (q - p) * (2/3 - t) * 6
        return p
    
    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    
    r = hue_to_rgb(p, q, h + 1/3)
    g = hue_to_rgb(p, q, h)
    b = hue_to_rgb(p, q, h - 1/3)
    
    return (r, g, b)


def rgb_to_lab(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """RGB to CIE LAB conversion."""
    # RGB to XYZ
    def gamma(c):
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92
    
    r, g, b = gamma(r), gamma(g), gamma(b)
    
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    # XYZ to Lab
    x /= 0.95047
    y /= 1.0
    z /= 1.08883
    
    def f(t):
        return t ** (1/3) if t > 0.008856 else 7.787 * t + 16/116
    
    l = 116 * f(y) - 16
    a = 500 * (f(x) - f(y))
    b = 200 * (f(y) - f(z))
    
    return (l, a, b)


def color_distance(c1: Tuple[float, float, float], 
                   c2: Tuple[float, float, float]) -> float:
    """Perceptual color distance using CIE76."""
    lab1 = rgb_to_lab(*c1)
    lab2 = rgb_to_lab(*c2)
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))


def generate_palette(base_color: Tuple[float, float, float],
                    scheme: str = 'complementary',
                    n_colors: int = 5) -> list:
    """Generate color palette from base color."""
    h, s, l = rgb_to_hsl(*base_color)
    colors = [base_color]
    
    if scheme == 'complementary':
        colors.append(hsl_to_rgb((h + 0.5) % 1, s, l))
    
    elif scheme == 'triadic':
        colors.append(hsl_to_rgb((h + 1/3) % 1, s, l))
        colors.append(hsl_to_rgb((h + 2/3) % 1, s, l))
    
    elif scheme == 'analogous':
        for i in range(1, n_colors):
            offset = (i - n_colors // 2) * 0.083  # 30 degrees
            colors.append(hsl_to_rgb((h + offset) % 1, s, l))
    
    elif scheme == 'split_complementary':
        colors.append(hsl_to_rgb((h + 0.417) % 1, s, l))  # 150 degrees
        colors.append(hsl_to_rgb((h + 0.583) % 1, s, l))  # 210 degrees
    
    elif scheme == 'tetradic':
        colors.append(hsl_to_rgb((h + 0.25) % 1, s, l))
        colors.append(hsl_to_rgb((h + 0.5) % 1, s, l))
        colors.append(hsl_to_rgb((h + 0.75) % 1, s, l))
    
    elif scheme == 'monochromatic':
        for i in range(1, n_colors):
            new_l = 0.1 + (i / (n_colors - 1)) * 0.8
            colors.append(hsl_to_rgb(h, s, new_l))
    
    return colors[:n_colors]


# ============================================================================
# GEOMETRY UTILITIES
# ============================================================================

def point_in_polygon(point: Tuple[float, float], 
                    polygon: list) -> bool:
    """Check if point is inside polygon (ray casting)."""
    x, y = point
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        
        j = i
    
    return inside


def line_intersection(p1: Tuple[float, float], p2: Tuple[float, float],
                     p3: Tuple[float, float], p4: Tuple[float, float]) -> Tuple[float, float]:
    """Find intersection point of two line segments."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return (x, y)


def polygon_area(vertices: list) -> float:
    """Calculate polygon area using shoelace formula."""
    n = len(vertices)
    area = 0
    
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    
    return abs(area) / 2


def polygon_centroid(vertices: list) -> Tuple[float, float]:
    """Calculate polygon centroid."""
    n = len(vertices)
    area = polygon_area(vertices)
    
    if area == 0:
        return (sum(v[0] for v in vertices) / n, sum(v[1] for v in vertices) / n)
    
    cx = cy = 0
    for i in range(n):
        j = (i + 1) % n
        factor = vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1]
        cx += (vertices[i][0] + vertices[j][0]) * factor
        cy += (vertices[i][1] + vertices[j][1]) * factor
    
    return (cx / (6 * area), cy / (6 * area))


def convex_hull(points: list) -> list:
    """Compute convex hull using Graham scan."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    points = sorted(points)
    
    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    return lower[:-1] + upper[:-1]


def bezier_point(control_points: list, t: float) -> Tuple[float, float]:
    """Calculate point on Bezier curve at parameter t."""
    n = len(control_points) - 1
    x = y = 0
    
    for i, (px, py) in enumerate(control_points):
        # Bernstein polynomial
        coeff = np.math.comb(n, i) * (1 - t)**(n - i) * t**i
        x += coeff * px
        y += coeff * py
    
    return (x, y)


def subdivide_bezier(control_points: list, n_segments: int = 100) -> list:
    """Subdivide Bezier curve into line segments."""
    return [bezier_point(control_points, t / n_segments) for t in range(n_segments + 1)]


def catmull_rom_spline(points: list, n_segments: int = 10) -> list:
    """Generate Catmull-Rom spline through points."""
    result = []
    
    for i in range(len(points) - 1):
        p0 = points[max(0, i - 1)]
        p1 = points[i]
        p2 = points[min(len(points) - 1, i + 1)]
        p3 = points[min(len(points) - 1, i + 2)]
        
        for j in range(n_segments):
            t = j / n_segments
            x = catmull_rom(p0[0], p1[0], p2[0], p3[0], t)
            y = catmull_rom(p0[1], p1[1], p2[1], p3[1], t)
            result.append((x, y))
    
    result.append(points[-1])
    return result
```

### 28.2 Convolution Kernels Reference

```python
"""
Convolution Kernels Quick Reference
===================================

Common kernels for image filtering.
"""

import numpy as np


# ============================================================================
# BLUR KERNELS
# ============================================================================

# Box blur (3x3)
BOX_BLUR_3x3 = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.float32) / 9

# Box blur (5x5)
BOX_BLUR_5x5 = np.ones((5, 5), dtype=np.float32) / 25

# Gaussian blur (3x3, sigma ≈ 0.85)
GAUSSIAN_3x3 = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32) / 16

# Gaussian blur (5x5, sigma ≈ 1.0)
GAUSSIAN_5x5 = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
], dtype=np.float32) / 256

# Gaussian blur (7x7, sigma ≈ 1.5)
GAUSSIAN_7x7 = np.array([
    [0, 0, 1, 2, 1, 0, 0],
    [0, 3, 13, 22, 13, 3, 0],
    [1, 13, 59, 97, 59, 13, 1],
    [2, 22, 97, 159, 97, 22, 2],
    [1, 13, 59, 97, 59, 13, 1],
    [0, 3, 13, 22, 13, 3, 0],
    [0, 0, 1, 2, 1, 0, 0]
], dtype=np.float32) / 1003


def create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create custom Gaussian kernel."""
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


# ============================================================================
# EDGE DETECTION KERNELS
# ============================================================================

# Sobel X (horizontal edges)
SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# Sobel Y (vertical edges)
SOBEL_Y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

# Scharr X (more accurate than Sobel)
SCHARR_X = np.array([
    [-3, 0, 3],
    [-10, 0, 10],
    [-3, 0, 3]
], dtype=np.float32)

# Scharr Y
SCHARR_Y = np.array([
    [-3, -10, -3],
    [0, 0, 0],
    [3, 10, 3]
], dtype=np.float32)

# Prewitt X
PREWITT_X = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

# Prewitt Y
PREWITT_Y = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
], dtype=np.float32)

# Roberts Cross X
ROBERTS_X = np.array([
    [1, 0],
    [0, -1]
], dtype=np.float32)

# Roberts Cross Y
ROBERTS_Y = np.array([
    [0, 1],
    [-1, 0]
], dtype=np.float32)

# Laplacian (edge detection)
LAPLACIAN_4 = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=np.float32)

LAPLACIAN_8 = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
], dtype=np.float32)

# Laplacian of Gaussian (LoG) approximation
LAPLACIAN_OF_GAUSSIAN = np.array([
    [0, 0, -1, 0, 0],
    [0, -1, -2, -1, 0],
    [-1, -2, 16, -2, -1],
    [0, -1, -2, -1, 0],
    [0, 0, -1, 0, 0]
], dtype=np.float32)


# ============================================================================
# SHARPENING KERNELS
# ============================================================================

# Basic sharpen
SHARPEN = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

# Strong sharpen
SHARPEN_STRONG = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]
], dtype=np.float32)

# Edge enhance
EDGE_ENHANCE = np.array([
    [0, 0, 0],
    [-1, 1, 0],
    [0, 0, 0]
], dtype=np.float32)

# Unsharp mask (subtract blurred from original)
UNSHARP_MASK = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, -476, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
], dtype=np.float32) / -256


# ============================================================================
# EMBOSS KERNELS
# ============================================================================

# Emboss (top-left lighting)
EMBOSS = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
], dtype=np.float32)

# Emboss (top lighting)
EMBOSS_TOP = np.array([
    [-1, -1, -1],
    [0, 1, 0],
    [1, 1, 1]
], dtype=np.float32)

# Emboss (left lighting)
EMBOSS_LEFT = np.array([
    [-1, 0, 1],
    [-1, 1, 1],
    [-1, 0, 1]
], dtype=np.float32)


# ============================================================================
# MORPHOLOGICAL KERNELS
# ============================================================================

# Square structuring element
MORPH_SQUARE_3 = np.ones((3, 3), dtype=np.uint8)
MORPH_SQUARE_5 = np.ones((5, 5), dtype=np.uint8)

# Cross structuring element
MORPH_CROSS = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)

# Diamond structuring element
MORPH_DIAMOND = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0]
], dtype=np.uint8)


def create_circular_kernel(radius: int) -> np.ndarray:
    """Create circular structuring element."""
    size = 2 * radius + 1
    y, x = np.ogrid[:size, :size]
    return ((x - radius)**2 + (y - radius)**2 <= radius**2).astype(np.uint8)


# ============================================================================
# SPECIAL EFFECT KERNELS
# ============================================================================

# Motion blur (horizontal)
MOTION_BLUR_H = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
], dtype=np.float32) / 5

# Motion blur (diagonal)
MOTION_BLUR_D = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
], dtype=np.float32) / 5

# High-pass filter
HIGH_PASS = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
], dtype=np.float32)

# Identity (no change)
IDENTITY = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
], dtype=np.float32)


# ============================================================================
# KERNEL UTILITIES
# ============================================================================

def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    """Normalize kernel to sum to 1."""
    s = kernel.sum()
    return kernel / s if s != 0 else kernel


def flip_kernel(kernel: np.ndarray) -> np.ndarray:
    """Flip kernel for convolution (vs correlation)."""
    return np.flipud(np.fliplr(kernel))


def separate_kernel(kernel: np.ndarray) -> tuple:
    """Attempt to separate 2D kernel into 1D kernels."""
    u, s, vt = np.linalg.svd(kernel)
    
    if s[1] / s[0] < 0.01:  # Approximately separable
        row = u[:, 0] * np.sqrt(s[0])
        col = vt[0, :] * np.sqrt(s[0])
        return row, col
    
    return None  # Not separable


def create_directional_kernel(angle: float, size: int = 5) -> np.ndarray:
    """Create directional blur kernel at given angle."""
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            dx = j - center
            dy = i - center
            
            # Distance from line at angle
            dist = abs(dx * np.sin(angle) - dy * np.cos(angle))
            
            if dist < 0.5:
                kernel[i, j] = 1
    
    return kernel / kernel.sum() if kernel.sum() > 0 else kernel
```

### 28.3 Blend Modes Reference

```python
"""
Blend Modes Quick Reference
===========================

All standard blend modes with implementations.
"""

import numpy as np


def blend_normal(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Normal blend - simple alpha composite."""
    return base * (1 - opacity) + top * opacity


def blend_multiply(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Multiply - darkens image."""
    result = base * top
    return blend_normal(base, result, opacity)


def blend_screen(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Screen - lightens image."""
    result = 1 - (1 - base) * (1 - top)
    return blend_normal(base, result, opacity)


def blend_overlay(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Overlay - combines multiply and screen."""
    mask = base < 0.5
    result = np.where(mask, 2 * base * top, 1 - 2 * (1 - base) * (1 - top))
    return blend_normal(base, result, opacity)


def blend_soft_light(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Soft light - gentle contrast adjustment."""
    result = (1 - 2 * top) * base * base + 2 * top * base
    return blend_normal(base, result, opacity)


def blend_hard_light(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Hard light - strong contrast (overlay with swapped inputs)."""
    mask = top < 0.5
    result = np.where(mask, 2 * base * top, 1 - 2 * (1 - base) * (1 - top))
    return blend_normal(base, result, opacity)


def blend_vivid_light(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Vivid light - extreme contrast."""
    mask = top < 0.5
    result = np.where(
        mask,
        1 - (1 - base) / (2 * top + 1e-10),
        base / (2 * (1 - top) + 1e-10)
    )
    return blend_normal(base, np.clip(result, 0, 1), opacity)


def blend_linear_light(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Linear light - linear dodge + linear burn."""
    result = base + 2 * top - 1
    return blend_normal(base, np.clip(result, 0, 1), opacity)


def blend_pin_light(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Pin light - replaces colors."""
    result = np.where(
        top < 0.5,
        np.minimum(base, 2 * top),
        np.maximum(base, 2 * (top - 0.5))
    )
    return blend_normal(base, result, opacity)


def blend_hard_mix(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Hard mix - posterize effect."""
    result = ((base + top) >= 1).astype(float)
    return blend_normal(base, result, opacity)


def blend_darken(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Darken - keeps darker pixels."""
    result = np.minimum(base, top)
    return blend_normal(base, result, opacity)


def blend_lighten(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Lighten - keeps lighter pixels."""
    result = np.maximum(base, top)
    return blend_normal(base, result, opacity)


def blend_color_dodge(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Color dodge - brightens base by top."""
    result = base / (1 - top + 1e-10)
    return blend_normal(base, np.clip(result, 0, 1), opacity)


def blend_color_burn(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Color burn - darkens base by top."""
    result = 1 - (1 - base) / (top + 1e-10)
    return blend_normal(base, np.clip(result, 0, 1), opacity)


def blend_linear_dodge(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Linear dodge (Add) - adds values."""
    result = base + top
    return blend_normal(base, np.clip(result, 0, 1), opacity)


def blend_linear_burn(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Linear burn - subtracts to darken."""
    result = base + top - 1
    return blend_normal(base, np.clip(result, 0, 1), opacity)


def blend_difference(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Difference - absolute difference."""
    result = np.abs(base - top)
    return blend_normal(base, result, opacity)


def blend_exclusion(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Exclusion - softer difference."""
    result = base + top - 2 * base * top
    return blend_normal(base, result, opacity)


def blend_subtract(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Subtract - subtracts top from base."""
    result = base - top
    return blend_normal(base, np.clip(result, 0, 1), opacity)


def blend_divide(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Divide - divides base by top."""
    result = base / (top + 1e-10)
    return blend_normal(base, np.clip(result, 0, 1), opacity)


def blend_hue(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Hue - takes hue from top, saturation/luminosity from base."""
    from .math_reference import rgb_to_hsl, hsl_to_rgb
    
    h_base, s_base, l_base = rgb_to_hsl(base[..., 0], base[..., 1], base[..., 2])
    h_top, _, _ = rgb_to_hsl(top[..., 0], top[..., 1], top[..., 2])
    
    result = np.stack(hsl_to_rgb(h_top, s_base, l_base), axis=-1)
    return blend_normal(base, result, opacity)


def blend_saturation(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Saturation - takes saturation from top."""
    from .math_reference import rgb_to_hsl, hsl_to_rgb
    
    h_base, s_base, l_base = rgb_to_hsl(base[..., 0], base[..., 1], base[..., 2])
    _, s_top, _ = rgb_to_hsl(top[..., 0], top[..., 1], top[..., 2])
    
    result = np.stack(hsl_to_rgb(h_base, s_top, l_base), axis=-1)
    return blend_normal(base, result, opacity)


def blend_color(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Color - takes hue and saturation from top."""
    from .math_reference import rgb_to_hsl, hsl_to_rgb
    
    _, _, l_base = rgb_to_hsl(base[..., 0], base[..., 1], base[..., 2])
    h_top, s_top, _ = rgb_to_hsl(top[..., 0], top[..., 1], top[..., 2])
    
    result = np.stack(hsl_to_rgb(h_top, s_top, l_base), axis=-1)
    return blend_normal(base, result, opacity)


def blend_luminosity(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Luminosity - takes luminosity from top."""
    from .math_reference import rgb_to_hsl, hsl_to_rgb
    
    h_base, s_base, _ = rgb_to_hsl(base[..., 0], base[..., 1], base[..., 2])
    _, _, l_top = rgb_to_hsl(top[..., 0], top[..., 1], top[..., 2])
    
    result = np.stack(hsl_to_rgb(h_base, s_base, l_top), axis=-1)
    return blend_normal(base, result, opacity)


# Blend mode lookup table
BLEND_MODES = {
    'normal': blend_normal,
    'multiply': blend_multiply,
    'screen': blend_screen,
    'overlay': blend_overlay,
    'soft_light': blend_soft_light,
    'hard_light': blend_hard_light,
    'vivid_light': blend_vivid_light,
    'linear_light': blend_linear_light,
    'pin_light': blend_pin_light,
    'hard_mix': blend_hard_mix,
    'darken': blend_darken,
    'lighten': blend_lighten,
    'color_dodge': blend_color_dodge,
    'color_burn': blend_color_burn,
    'linear_dodge': blend_linear_dodge,
    'linear_burn': blend_linear_burn,
    'difference': blend_difference,
    'exclusion': blend_exclusion,
    'subtract': blend_subtract,
    'divide': blend_divide,
    'hue': blend_hue,
    'saturation': blend_saturation,
    'color': blend_color,
    'luminosity': blend_luminosity,
}


def apply_blend(base: np.ndarray, top: np.ndarray, 
                mode: str = 'normal', opacity: float = 1.0) -> np.ndarray:
    """Apply blend mode by name."""
    blend_fn = BLEND_MODES.get(mode.lower(), blend_normal)
    return blend_fn(base, top, opacity)
```


---

# APPENDIX

## A. Common Image Formats

| Format | Extension | Compression | Alpha | Color Depth | Best For |
|--------|-----------|-------------|-------|-------------|----------|
| PNG | .png | Lossless | Yes | 8/16/24/32 bit | Graphics, screenshots |
| JPEG | .jpg/.jpeg | Lossy | No | 24 bit | Photos |
| WebP | .webp | Both | Yes | 24/32 bit | Web images |
| TIFF | .tif/.tiff | Both | Yes | 8/16/32 bit | Print, archival |
| BMP | .bmp | None | Yes | 1/4/8/24/32 bit | Legacy support |
| GIF | .gif | Lossless | 1-bit | 8 bit (256 colors) | Animations |
| PSD | .psd | None | Yes | 8/16/32 bit | Adobe workflow |
| EXR | .exr | Both | Yes | 16/32 bit float | HDR, VFX |
| RAW | .cr2/.nef/etc | None | No | 12/14 bit | Photography |

## B. Color Space Reference

| Space | Components | Range | Use Case |
|-------|------------|-------|----------|
| RGB | Red, Green, Blue | 0-255 or 0-1 | Display |
| RGBA | RGB + Alpha | 0-255 or 0-1 | Transparency |
| HSV | Hue, Sat, Value | H: 0-360, S/V: 0-1 | Color selection |
| HSL | Hue, Sat, Light | H: 0-360, S/L: 0-1 | Color adjustment |
| LAB | L, a, b | L: 0-100, a/b: -128-127 | Perceptual |
| CMYK | Cyan, Mag, Yel, K | 0-100% | Print |
| YCbCr | Luminance, Cb, Cr | Y: 16-235, C: 16-240 | Video |

## C. Performance Tips

1. **Memory Management**
   - Use float32 instead of float64 (half memory)
   - Process large images in tiles
   - Release unused arrays explicitly

2. **Computation Optimization**
   - Use numpy vectorized operations
   - Avoid Python loops for pixel operations
   - Consider numba for critical paths

3. **Caching Strategies**
   - Cache intermediate results
   - Use LRU cache for repeated operations
   - Implement lazy evaluation

4. **Parallelization**
   - Use multiprocessing for batch operations
   - Consider GPU acceleration for large images
   - Separate I/O from computation

## D. Further Reading

- Digital Image Processing by Gonzalez & Woods
- The Book of Shaders (thebookofshaders.com)
- GPU Gems series (NVIDIA)
- Texturing and Modeling: A Procedural Approach
- Real-Time Rendering by Akenine-Möller et al.

---

*End of Procedural Art Engineering Complete Guide*
*Total content: ~900KB of comprehensive procedural art engineering knowledge*


## E. Troubleshooting Guide

### E.1 Common Issues and Solutions

```python
"""
Troubleshooting Guide
=====================

Solutions to common procedural art problems.
"""

# ============================================================================
# IMAGE QUALITY ISSUES
# ============================================================================

class ImageQualityTroubleshooter:
    """
    Solutions for common image quality problems.
    """
    
    ISSUES = {
        "banding": {
            "symptoms": [
                "Visible steps in gradients",
                "Posterization in smooth areas",
                "Color stepping in sky/skin tones"
            ],
            "causes": [
                "Insufficient bit depth (8-bit processing)",
                "Over-compression (low JPEG quality)",
                "Aggressive color quantization"
            ],
            "solutions": [
                "Process in 16-bit or float32 mode",
                "Add subtle dithering before output",
                "Use PNG instead of JPEG for gradients",
                "Apply noise overlay to break up banding"
            ],
            "code_fix": """
# Add dithering to prevent banding
def add_dithering(image, strength=1/255):
    noise = np.random.uniform(-strength, strength, image.shape)
    return np.clip(image + noise, 0, 1)
"""
        },
        
        "color_shift": {
            "symptoms": [
                "Colors look different after save/load",
                "Images appear washed out or oversaturated",
                "Skin tones look wrong"
            ],
            "causes": [
                "Color space mismatch (sRGB vs Adobe RGB)",
                "Missing ICC profile",
                "Incorrect gamma handling"
            ],
            "solutions": [
                "Embed ICC profile in output",
                "Convert to sRGB before saving for web",
                "Apply gamma correction consistently"
            ],
            "code_fix": """
# Ensure sRGB output
def to_srgb(linear_rgb):
    '''Apply sRGB gamma curve.'''
    return np.where(
        linear_rgb <= 0.0031308,
        12.92 * linear_rgb,
        1.055 * np.power(linear_rgb, 1/2.4) - 0.055
    )
"""
        },
        
        "aliasing": {
            "symptoms": [
                "Jagged edges on diagonal lines",
                "Moire patterns in fine details",
                "Stairstepping on curves"
            ],
            "causes": [
                "Insufficient sampling rate",
                "Missing anti-aliasing",
                "Sharp cutoff in frequency domain"
            ],
            "solutions": [
                "Render at higher resolution, then downsample",
                "Use supersampling anti-aliasing (SSAA)",
                "Apply low-pass filter before downsampling"
            ],
            "code_fix": """
# Supersampling anti-aliasing
def render_with_ssaa(render_func, width, height, samples=4):
    '''Render at higher resolution and downsample.'''
    high_res = render_func(width * samples, height * samples)
    
    # Box filter downsample
    output = np.zeros((height, width, high_res.shape[2]))
    for i in range(samples):
        for j in range(samples):
            output += high_res[i::samples, j::samples]
    
    return output / (samples * samples)
"""
        },
        
        "noise_artifacts": {
            "symptoms": [
                "Grainy appearance in smooth areas",
                "Random speckles in output",
                "Visible sensor noise amplification"
            ],
            "causes": [
                "High ISO noise in source image",
                "Over-sharpening amplifying noise",
                "Insufficient denoising"
            ],
            "solutions": [
                "Apply noise reduction before sharpening",
                "Use bilateral or non-local means denoising",
                "Mask noise reduction to flat areas only"
            ],
            "code_fix": """
# Simple bilateral filter for noise reduction
def bilateral_denoise(image, sigma_space=5, sigma_color=0.1):
    h, w = image.shape[:2]
    output = np.zeros_like(image)
    
    for i in range(h):
        for j in range(w):
            window = get_window(image, i, j, int(sigma_space * 2))
            spatial_weights = gaussian_weights(window.shape, sigma_space)
            color_weights = np.exp(-((window - image[i,j])**2) / (2 * sigma_color**2))
            weights = spatial_weights * color_weights
            output[i, j] = np.sum(window * weights) / np.sum(weights)
    
    return output
"""
        },
        
        "halo_artifacts": {
            "symptoms": [
                "Light or dark edges around high-contrast boundaries",
                "Glowing outlines after sharpening",
                "Ring artifacts around edges"
            ],
            "causes": [
                "Over-sharpening",
                "Aggressive unsharp mask",
                "Large sharpening radius"
            ],
            "solutions": [
                "Reduce sharpening amount",
                "Use smaller radius",
                "Apply edge-aware sharpening",
                "Use deconvolution instead of unsharp mask"
            ],
            "code_fix": """
# Edge-aware sharpening to reduce halos
def edge_aware_sharpen(image, amount=1.0, radius=1.0):
    blurred = gaussian_blur(image, radius)
    detail = image - blurred
    
    # Edge mask to reduce sharpening at edges
    edges = detect_edges(image)
    edge_mask = 1 - edges  # Less sharpening at edges
    
    return image + amount * detail * edge_mask
"""
        }
    }
    
    @classmethod
    def diagnose(cls, image: 'np.ndarray') -> list:
        """Diagnose potential issues in an image."""
        issues = []
        
        # Check for banding
        if cls._check_banding(image):
            issues.append("banding")
        
        # Check for noise
        if cls._check_noise(image):
            issues.append("noise_artifacts")
        
        # Check for clipping
        if cls._check_clipping(image):
            issues.append("clipping")
        
        return issues
    
    @staticmethod
    def _check_banding(image: 'np.ndarray') -> bool:
        """Check for visible banding."""
        # Look for quantized gradient patterns
        gradients = np.diff(image, axis=0)
        unique_steps = len(np.unique(np.round(gradients * 255)))
        return unique_steps < 20
    
    @staticmethod
    def _check_noise(image: 'np.ndarray') -> bool:
        """Check for excessive noise."""
        # Estimate noise in flat regions
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        # Would apply laplacian and check variance
        return False
    
    @staticmethod
    def _check_clipping(image: 'np.ndarray') -> bool:
        """Check for highlight/shadow clipping."""
        clipped_high = np.sum(image > 0.99) / image.size
        clipped_low = np.sum(image < 0.01) / image.size
        return clipped_high > 0.05 or clipped_low > 0.05


# ============================================================================
# PERFORMANCE ISSUES
# ============================================================================

class PerformanceTroubleshooter:
    """
    Solutions for performance problems.
    """
    
    ISSUES = {
        "slow_processing": {
            "symptoms": [
                "Operations taking too long",
                "UI freezing during processing",
                "Memory usage growing over time"
            ],
            "causes": [
                "Python loops instead of vectorized ops",
                "Processing full resolution unnecessarily",
                "Memory leaks from unreleased arrays"
            ],
            "solutions": [
                "Use numpy vectorized operations",
                "Process preview at reduced resolution",
                "Use generators for large datasets",
                "Clear cache periodically"
            ],
            "code_fix": """
# Vectorized vs loop performance
import numpy as np
import time

# SLOW: Python loop
def process_slow(image):
    h, w, c = image.shape
    result = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                result[i, j, k] = image[i, j, k] * 1.5
    return result

# FAST: Vectorized
def process_fast(image):
    return image * 1.5

# Benchmark
image = np.random.rand(1000, 1000, 3)
# Vectorized is typically 100-1000x faster
"""
        },
        
        "memory_errors": {
            "symptoms": [
                "MemoryError exceptions",
                "System becoming unresponsive",
                "Swap usage increasing"
            ],
            "causes": [
                "Loading images at full resolution",
                "Not releasing intermediate arrays",
                "Creating copies unnecessarily"
            ],
            "solutions": [
                "Process in tiles/chunks",
                "Use in-place operations",
                "Delete arrays after use",
                "Use memory-mapped files for large images"
            ],
            "code_fix": """
# Process large image in tiles
def process_tiled(image, operation, tile_size=512, overlap=64):
    h, w = image.shape[:2]
    result = np.zeros_like(image)
    
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            # Extract tile with overlap
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = image[y:y_end, x:x_end]
            
            # Process tile
            processed = operation(tile)
            
            # Blend into result (handle overlap)
            out_y = y + overlap // 2 if y > 0 else 0
            out_x = x + overlap // 2 if x > 0 else 0
            out_ye = y_end - overlap // 2 if y_end < h else h
            out_xe = x_end - overlap // 2 if x_end < w else w
            
            tile_y = out_y - y
            tile_x = out_x - x
            tile_ye = tile_y + (out_ye - out_y)
            tile_xe = tile_x + (out_xe - out_x)
            
            result[out_y:out_ye, out_x:out_xe] = processed[tile_y:tile_ye, tile_x:tile_xe]
    
    return result
"""
        },
        
        "gpu_not_utilized": {
            "symptoms": [
                "GPU usage near 0%",
                "CPU bottlenecked at 100%",
                "No speedup from CUDA installation"
            ],
            "causes": [
                "Operations not GPU-compatible",
                "Data transfer overhead",
                "Wrong framework configuration"
            ],
            "solutions": [
                "Use CuPy instead of NumPy",
                "Batch operations to reduce transfers",
                "Check CUDA installation",
                "Use GPU-native libraries (OpenCV with CUDA)"
            ],
            "code_fix": """
# Using CuPy for GPU acceleration
try:
    import cupy as cp
    
    def process_gpu(image):
        # Transfer to GPU
        image_gpu = cp.asarray(image)
        
        # GPU operations
        result_gpu = cp.sqrt(image_gpu)  # Example operation
        
        # Transfer back
        return cp.asnumpy(result_gpu)
        
except ImportError:
    def process_gpu(image):
        # Fallback to CPU
        return np.sqrt(image)
"""
        }
    }


# ============================================================================
# PROCEDURAL GENERATION ISSUES
# ============================================================================

class ProceduralTroubleshooter:
    """
    Solutions for procedural generation problems.
    """
    
    ISSUES = {
        "visible_tiling": {
            "symptoms": [
                "Obvious seams when texture is tiled",
                "Repetitive patterns visible",
                "Edge discontinuities"
            ],
            "causes": [
                "Non-tileable base noise",
                "Incorrect coordinate wrapping",
                "Edge artifacts in processing"
            ],
            "solutions": [
                "Use tileable noise generation",
                "Apply seamless filter post-generation",
                "Blend edges using gradient masks"
            ],
            "code_fix": """
# Make texture tileable
def make_tileable(texture, blend_width=64):
    h, w = texture.shape[:2]
    
    # Create rolled versions
    rolled_h = np.roll(texture, h // 2, axis=0)
    rolled_w = np.roll(texture, w // 2, axis=1)
    rolled_both = np.roll(rolled_h, w // 2, axis=1)
    
    # Create blend masks
    y = np.linspace(0, 1, h)[:, np.newaxis]
    x = np.linspace(0, 1, w)[np.newaxis, :]
    
    mask_v = np.abs(y - 0.5) * 2  # 0 at center, 1 at edges
    mask_h = np.abs(x - 0.5) * 2
    
    # Blend
    result = texture * (1 - mask_v) + rolled_h * mask_v
    result = result * (1 - mask_h) + np.roll(result, w//2, axis=1) * mask_h
    
    return result
"""
        },
        
        "poor_randomness": {
            "symptoms": [
                "Patterns look artificial",
                "Visible regularity in 'random' features",
                "Same output every run (or different when shouldn't be)"
            ],
            "causes": [
                "Insufficient octaves in fractal noise",
                "Using wrong random distribution",
                "Not seeding RNG properly"
            ],
            "solutions": [
                "Add more octaves to noise",
                "Use appropriate distribution (uniform vs gaussian)",
                "Implement proper seeding system"
            ],
            "code_fix": """
# Proper seeding for reproducible randomness
import numpy as np

class ProceduralRandom:
    def __init__(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**31)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def reset(self):
        '''Reset to initial seed.'''
        self.rng = np.random.default_rng(self.seed)
    
    def uniform(self, low=0, high=1, size=None):
        return self.rng.uniform(low, high, size)
    
    def normal(self, mean=0, std=1, size=None):
        return self.rng.normal(mean, std, size)
    
    def hash_coordinates(self, x, y):
        '''Deterministic random based on coordinates.'''
        combined = x * 374761393 + y * 668265263 + self.seed
        combined = (combined ^ (combined >> 13)) * 1274126177
        return combined / 2**31
"""
        },
        
        "unrealistic_results": {
            "symptoms": [
                "Generated textures look fake",
                "Missing natural variation",
                "Too perfect/uniform appearance"
            ],
            "causes": [
                "Missing subtle imperfections",
                "Uniform parameters across image",
                "Lack of correlation between features"
            ],
            "solutions": [
                "Add subtle noise/imperfections",
                "Vary parameters spatially",
                "Use reference images to guide generation"
            ],
            "code_fix": """
# Add natural variation
def add_natural_variation(texture, variation_strength=0.1):
    h, w = texture.shape[:2]
    
    # Low-frequency variation (large-scale changes)
    low_freq = generate_perlin_noise(h, w, scale=2, octaves=2)
    
    # High-frequency variation (small imperfections)
    high_freq = generate_perlin_noise(h, w, scale=32, octaves=1)
    
    # Combine
    variation = 0.7 * low_freq + 0.3 * high_freq
    variation = (variation - 0.5) * 2 * variation_strength
    
    return np.clip(texture + variation[..., np.newaxis], 0, 1)
"""
        }
    }


# ============================================================================
# NEURAL NETWORK ISSUES
# ============================================================================

class NeuralNetTroubleshooter:
    """
    Solutions for neural network problems in procedural art.
    """
    
    ISSUES = {
        "mode_collapse_gan": {
            "symptoms": [
                "Generator produces same/similar outputs",
                "Lack of variety in generated images",
                "Training loss not improving"
            ],
            "causes": [
                "Discriminator too strong",
                "Insufficient latent space",
                "Training imbalance"
            ],
            "solutions": [
                "Add noise to discriminator inputs",
                "Use larger latent dimension",
                "Try different GAN architecture (WGAN, etc.)",
                "Implement minibatch discrimination"
            ]
        },
        
        "training_unstable": {
            "symptoms": [
                "Loss oscillating wildly",
                "NaN values appearing",
                "Model quality degrading over time"
            ],
            "causes": [
                "Learning rate too high",
                "Gradient explosion",
                "Batch size too small"
            ],
            "solutions": [
                "Reduce learning rate",
                "Add gradient clipping",
                "Use batch normalization",
                "Increase batch size"
            ]
        },
        
        "style_transfer_artifacts": {
            "symptoms": [
                "Checkerboard patterns",
                "Loss of content structure",
                "Unnatural color blobs"
            ],
            "causes": [
                "Upsampling artifacts",
                "Incorrect loss weighting",
                "Insufficient iterations"
            ],
            "solutions": [
                "Use resize-convolution instead of deconv",
                "Adjust content/style weight balance",
                "Increase total variation loss"
            ]
        }
    }
```

### E.2 Debugging Techniques

```python
"""
Debugging Techniques for Procedural Art
=======================================

Tools and methods for diagnosing issues.
"""

import numpy as np
from typing import Optional, Tuple


class ImageDebugger:
    """
    Debugging utilities for image processing.
    """
    
    @staticmethod
    def visualize_channels(image: np.ndarray) -> np.ndarray:
        """Split and display RGB channels separately."""
        if image.ndim != 3 or image.shape[2] < 3:
            return image
        
        h, w = image.shape[:2]
        result = np.zeros((h, w * 3, 3))
        
        # Red channel
        result[:, :w, 0] = image[:, :, 0]
        # Green channel
        result[:, w:2*w, 1] = image[:, :, 1]
        # Blue channel
        result[:, 2*w:, 2] = image[:, :, 2]
        
        return result
    
    @staticmethod
    def visualize_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
        """Create histogram visualization."""
        h_out, w_out = 200, 256
        result = np.zeros((h_out, w_out * 3, 3))
        
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        
        for c in range(min(3, image.shape[2] if image.ndim == 3 else 1)):
            channel = image[:, :, c] if image.ndim == 3 else image
            hist, _ = np.histogram(channel.flatten(), bins=bins, range=(0, 1))
            hist = hist / hist.max() * (h_out - 1)
            
            for i in range(bins):
                col = int(hist[i])
                result[h_out-col:h_out, c*w_out + i, c] = 1
        
        return result
    
    @staticmethod
    def visualize_gradients(image: np.ndarray) -> np.ndarray:
        """Visualize image gradients."""
        gray = np.mean(image, axis=2) if image.ndim == 3 else image
        
        # Sobel gradients
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        h, w = gray.shape
        gx = np.zeros((h-2, w-2))
        gy = np.zeros((h-2, w-2))
        
        for i in range(h-2):
            for j in range(w-2):
                gx[i, j] = np.sum(gray[i:i+3, j:j+3] * sobel_x)
                gy[i, j] = np.sum(gray[i:i+3, j:j+3] * sobel_y)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)
        
        # Encode direction as hue, magnitude as value
        hsv = np.zeros((h-2, w-2, 3))
        hsv[:, :, 0] = (direction + np.pi) / (2 * np.pi)  # Hue
        hsv[:, :, 1] = 1  # Saturation
        hsv[:, :, 2] = magnitude / magnitude.max()  # Value
        
        # Convert HSV to RGB (simplified)
        return hsv  # Would need hsv_to_rgb conversion
    
    @staticmethod
    def detect_anomalies(image: np.ndarray) -> dict:
        """Detect potential image anomalies."""
        results = {}
        
        # Check for NaN values
        results['has_nan'] = np.any(np.isnan(image))
        
        # Check for infinite values
        results['has_inf'] = np.any(np.isinf(image))
        
        # Check for out-of-range values
        results['min_value'] = float(np.min(image))
        results['max_value'] = float(np.max(image))
        results['out_of_range'] = results['min_value'] < 0 or results['max_value'] > 1
        
        # Check for clipping
        results['clipped_low'] = float(np.sum(image <= 0.001) / image.size)
        results['clipped_high'] = float(np.sum(image >= 0.999) / image.size)
        
        # Check for uniformity (suspicious)
        results['std_dev'] = float(np.std(image))
        results['possibly_uniform'] = results['std_dev'] < 0.01
        
        return results
    
    @staticmethod
    def compare_images(img1: np.ndarray, img2: np.ndarray) -> dict:
        """Compare two images and return metrics."""
        if img1.shape != img2.shape:
            return {'error': 'Shape mismatch'}
        
        diff = img1 - img2
        
        return {
            'mae': float(np.mean(np.abs(diff))),
            'mse': float(np.mean(diff ** 2)),
            'max_diff': float(np.max(np.abs(diff))),
            'psnr': float(10 * np.log10(1.0 / (np.mean(diff ** 2) + 1e-10))),
            'ssim': ImageDebugger._compute_ssim(img1, img2)
        }
    
    @staticmethod
    def _compute_ssim(img1: np.ndarray, img2: np.ndarray, 
                     window_size: int = 11) -> float:
        """Compute Structural Similarity Index."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Convert to grayscale if needed
        if img1.ndim == 3:
            img1 = np.mean(img1, axis=2)
            img2 = np.mean(img2, axis=2)
        
        # Simple SSIM (without gaussian weighting)
        mu1 = img1.mean()
        mu2 = img2.mean()
        var1 = img1.var()
        var2 = img2.var()
        covar = np.mean((img1 - mu1) * (img2 - mu2))
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * covar + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (var1 + var2 + C2))
        
        return float(ssim)


class ProgressTracker:
    """
    Track progress during long operations.
    """
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = None
    
    def update(self, n: int = 1):
        """Update progress by n steps."""
        import time
        
        if self.start_time is None:
            self.start_time = time.time()
        
        self.current += n
        
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0
        
        percent = self.current / self.total * 100
        bar_length = 30
        filled = int(bar_length * self.current / self.total)
        bar = '=' * filled + '-' * (bar_length - filled)
        
        print(f"\r{self.description}: [{bar}] {percent:.1f}% ({self.current}/{self.total}) "
              f"ETA: {eta:.1f}s", end='', flush=True)
        
        if self.current >= self.total:
            print()
    
    def finish(self):
        """Mark as complete."""
        self.current = self.total
        self.update(0)
```

## F. Keyboard Shortcuts Reference

### GIMP Shortcuts

| Action | Shortcut |
|--------|----------|
| New Image | Ctrl+N |
| Open | Ctrl+O |
| Save | Ctrl+S |
| Export As | Ctrl+Shift+E |
| Undo | Ctrl+Z |
| Redo | Ctrl+Y |
| Fit Image in Window | Shift+Ctrl+E |
| Zoom In | + |
| Zoom Out | - |
| Zoom 100% | 1 |
| Zoom to Fit | Shift+Ctrl+J |
| Select All | Ctrl+A |
| Select None | Ctrl+Shift+A |
| Invert Selection | Ctrl+I |
| Float Selection | Ctrl+Shift+L |
| New Layer | Ctrl+Shift+N |
| Flatten Image | Ctrl+Shift+M |
| Fill with FG Color | Alt+Backspace |
| Fill with BG Color | Ctrl+Backspace |
| Gaussian Blur | Filters > Blur > Gaussian |
| Curves | Colors > Curves |
| Levels | Colors > Levels |
| Hue-Saturation | Colors > Hue-Saturation |

### Photoshop Shortcuts

| Action | Shortcut |
|--------|----------|
| New Document | Ctrl+N |
| Open | Ctrl+O |
| Save | Ctrl+S |
| Save As | Ctrl+Shift+S |
| Export As | Alt+Shift+Ctrl+W |
| Undo | Ctrl+Z |
| Step Backward | Ctrl+Alt+Z |
| Fit on Screen | Ctrl+0 |
| Actual Pixels | Ctrl+1 |
| Zoom In | Ctrl++ |
| Zoom Out | Ctrl+- |
| Free Transform | Ctrl+T |
| Select All | Ctrl+A |
| Deselect | Ctrl+D |
| Inverse Selection | Ctrl+Shift+I |
| New Layer | Ctrl+Shift+N |
| Merge Visible | Ctrl+Shift+E |
| Flatten | Ctrl+Shift+Alt+E |
| Fill | Shift+F5 |
| Gaussian Blur | Filter > Blur > Gaussian |
| Curves | Ctrl+M |
| Levels | Ctrl+L |
| Hue/Saturation | Ctrl+U |
| Color Balance | Ctrl+B |
| Desaturate | Ctrl+Shift+U |

## G. Formula Reference

### Color Formulas

```
RGB to Grayscale (Luminance):
Y = 0.299*R + 0.587*G + 0.114*B

RGB to Grayscale (Average):
Y = (R + G + B) / 3

Contrast Adjustment:
output = (input - 0.5) * contrast + 0.5

Brightness Adjustment:
output = input + brightness

Gamma Correction:
output = input ^ (1/gamma)

Blend Multiply:
output = base * blend

Blend Screen:
output = 1 - (1 - base) * (1 - blend)

Blend Overlay:
output = base < 0.5 ? 2*base*blend : 1 - 2*(1-base)*(1-blend)
```

### Geometry Formulas

```
Distance (Euclidean):
d = sqrt((x2-x1)² + (y2-y1)²)

Distance (Manhattan):
d = |x2-x1| + |y2-y1|

Angle Between Vectors:
θ = acos((a·b) / (|a||b|))

Rotation:
x' = x*cos(θ) - y*sin(θ)
y' = x*sin(θ) + y*cos(θ)

Bezier Curve (Cubic):
P(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3

Circle Area:
A = π * r²

Polygon Area (Shoelace):
A = ½|Σ(x_i * y_{i+1} - x_{i+1} * y_i)|
```

### Noise Formulas

```
Perlin Fade Function:
f(t) = t³(t(t*6 - 15) + 10)

FBM (Fractal Brownian Motion):
value = Σ(amplitude_i * noise(frequency_i * x))
where amplitude_i = persistence^i
and frequency_i = lacunarity^i

Voronoi/Worley Distance:
F1 = distance to closest point
F2 = distance to second closest point

Domain Warping:
value = noise(x + noise(x), y + noise(y))
```

---

*This concludes the Procedural Art Engineering Complete Guide.*

*Generated as a comprehensive reference covering:*
- *Image fundamentals and color theory*
- *Convolution and filtering*
- *Noise and procedural generation*
- *Neural networks for art*
- *Plugin development for GIMP and Photoshop*
- *Optimization and parameter tuning*
- *Complete working templates*
- *Extensive reference material*

*Total size: ~900KB*
