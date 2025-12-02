#!/usr/bin/env python3
"""
LatticeForge Image Toolkit - S-Tier Photoshop Operations via CLI
Usage: python forge.py <command> [options] <input> <output>

Commands:
  crop        - Crop image (--box x,y,w,h or --ratio 16:9 --gravity center)
  resize      - Resize image (--width, --height, --scale, --fit cover|contain)
  rembg       - Remove background (AI-powered)
  replace-bg  - Remove bg and add new one (--bg color|image)
  overlay     - Composite images (--overlay path --position --opacity)
  watermark   - Add watermark (--text or --image, --position, --opacity)
  filter      - Apply filter (--blur, --sharpen, --contrast, --brightness, --saturation)
  border      - Add border/frame (--size --color --style solid|glow|gradient)
  shadow      - Add drop shadow (--blur --offset --color)
  round       - Round corners (--radius)
  favicon     - Generate favicon set from logo
  social      - Generate social media sizes (og, twitter, instagram)
  batch       - Process multiple files
  extract     - Extract from black bg to transparent
  tint        - Color tint/overlay (--color --mode multiply|overlay|screen)
  info        - Show image info
"""

import sys
import os
import argparse
from pathlib import Path

try:
    from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont, ImageOps
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install pillow numpy")
    sys.exit(1)

# Optional imports
try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class ImageForge:
    """S-tier image manipulation toolkit"""

    def __init__(self, input_path: str):
        self.input_path = Path(input_path)
        self.image = Image.open(input_path)
        # Ensure RGBA for transparency support
        if self.image.mode != 'RGBA':
            self.image = self.image.convert('RGBA')

    def save(self, output_path: str, quality: int = 95):
        """Save with format detection"""
        output = Path(output_path)
        fmt = output.suffix.lower()

        if fmt in ['.jpg', '.jpeg']:
            # Convert to RGB for JPEG (no alpha)
            rgb = Image.new('RGB', self.image.size, (255, 255, 255))
            rgb.paste(self.image, mask=self.image.split()[3] if self.image.mode == 'RGBA' else None)
            rgb.save(output_path, 'JPEG', quality=quality)
        elif fmt == '.webp':
            self.image.save(output_path, 'WEBP', quality=quality)
        elif fmt == '.png':
            self.image.save(output_path, 'PNG', optimize=True)
        else:
            self.image.save(output_path)

        print(f"✓ Saved: {output_path} ({self.image.size[0]}x{self.image.size[1]})")
        return self

    # === CROPPING ===
    def crop_box(self, x: int, y: int, w: int, h: int):
        """Crop to specific box"""
        self.image = self.image.crop((x, y, x + w, y + h))
        return self

    def crop_ratio(self, ratio: str, gravity: str = 'center'):
        """Crop to aspect ratio with gravity"""
        w, h = self.image.size
        target_w, target_h = map(int, ratio.split(':'))
        target_ratio = target_w / target_h
        current_ratio = w / h

        if current_ratio > target_ratio:
            # Too wide, crop width
            new_w = int(h * target_ratio)
            new_h = h
        else:
            # Too tall, crop height
            new_w = w
            new_h = int(w / target_ratio)

        # Calculate position based on gravity
        if gravity == 'center':
            x = (w - new_w) // 2
            y = (h - new_h) // 2
        elif gravity == 'top':
            x, y = (w - new_w) // 2, 0
        elif gravity == 'bottom':
            x, y = (w - new_w) // 2, h - new_h
        elif gravity == 'left':
            x, y = 0, (h - new_h) // 2
        elif gravity == 'right':
            x, y = w - new_w, (h - new_h) // 2
        else:
            x, y = 0, 0

        self.image = self.image.crop((x, y, x + new_w, y + new_h))
        return self

    def crop_circle(self):
        """Crop to circle (for avatars)"""
        size = min(self.image.size)
        # Center crop to square first
        self.crop_ratio('1:1', 'center')

        # Create circular mask
        mask = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)

        # Apply mask
        self.image = self.image.resize((size, size), Image.Resampling.LANCZOS)
        self.image.putalpha(mask)
        return self

    # === RESIZING ===
    def resize(self, width: int = None, height: int = None, scale: float = None,
               fit: str = 'contain'):
        """Resize with multiple modes"""
        w, h = self.image.size

        if scale:
            new_w, new_h = int(w * scale), int(h * scale)
        elif width and height:
            if fit == 'cover':
                # Fill, may crop
                ratio = max(width / w, height / h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                self.image = self.image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                # Center crop to target
                x = (new_w - width) // 2
                y = (new_h - height) // 2
                self.image = self.image.crop((x, y, x + width, y + height))
                return self
            elif fit == 'contain':
                # Fit within, may letterbox
                ratio = min(width / w, height / h)
                new_w, new_h = int(w * ratio), int(h * ratio)
            else:  # 'fill' - stretch
                new_w, new_h = width, height
        elif width:
            ratio = width / w
            new_w, new_h = width, int(h * ratio)
        elif height:
            ratio = height / h
            new_w, new_h = int(w * ratio), height
        else:
            return self

        self.image = self.image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return self

    # === BACKGROUND ===
    def remove_background(self):
        """AI-powered background removal"""
        if not HAS_REMBG:
            print("⚠ rembg not installed. Run: pip install rembg[gpu]")
            return self

        # Convert to bytes, process, convert back
        from io import BytesIO
        buf = BytesIO()
        self.image.save(buf, format='PNG')
        buf.seek(0)

        result = rembg_remove(buf.read())
        self.image = Image.open(BytesIO(result)).convert('RGBA')
        print("✓ Background removed")
        return self

    def extract_from_black(self, threshold: int = 20):
        """Extract element from pure black background to transparent"""
        arr = np.array(self.image)

        # Find pixels that are near-black
        is_black = (arr[:,:,0] < threshold) & (arr[:,:,1] < threshold) & (arr[:,:,2] < threshold)

        # Set alpha to 0 for black pixels
        arr[:,:,3] = np.where(is_black, 0, arr[:,:,3])

        self.image = Image.fromarray(arr)
        print("✓ Black background extracted to transparent")
        return self

    def replace_background(self, bg: str):
        """Replace background with color or image"""
        self.remove_background()

        if bg.startswith('#') or bg.startswith('rgb'):
            # Color background
            color = self._parse_color(bg)
            new_bg = Image.new('RGBA', self.image.size, color)
        else:
            # Image background
            new_bg = Image.open(bg).convert('RGBA')
            new_bg = new_bg.resize(self.image.size, Image.Resampling.LANCZOS)

        new_bg.paste(self.image, (0, 0), self.image)
        self.image = new_bg
        return self

    # === COMPOSITING ===
    def overlay(self, overlay_path: str, position: str = 'center',
                opacity: float = 1.0, scale: float = 1.0):
        """Overlay another image"""
        overlay = Image.open(overlay_path).convert('RGBA')

        # Scale overlay
        if scale != 1.0:
            new_size = (int(overlay.size[0] * scale), int(overlay.size[1] * scale))
            overlay = overlay.resize(new_size, Image.Resampling.LANCZOS)

        # Apply opacity
        if opacity < 1.0:
            alpha = overlay.split()[3]
            alpha = alpha.point(lambda p: int(p * opacity))
            overlay.putalpha(alpha)

        # Calculate position
        x, y = self._calc_position(position, self.image.size, overlay.size)

        # Composite
        self.image.paste(overlay, (x, y), overlay)
        return self

    def watermark_text(self, text: str, position: str = 'bottomright',
                       opacity: float = 0.5, size: int = 24, color: str = '#ffffff'):
        """Add text watermark"""
        txt_layer = Image.new('RGBA', self.image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt_layer)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except:
            font = ImageFont.load_default()

        # Get text size
        bbox = draw.textbbox((0, 0), text, font=font)
        txt_w, txt_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        x, y = self._calc_position(position, self.image.size, (txt_w + 20, txt_h + 10))

        rgba_color = self._parse_color(color)
        rgba_color = rgba_color[:3] + (int(255 * opacity),)

        draw.text((x + 10, y + 5), text, font=font, fill=rgba_color)

        self.image = Image.alpha_composite(self.image, txt_layer)
        return self

    # === FILTERS ===
    def blur(self, radius: int = 5):
        """Gaussian blur"""
        self.image = self.image.filter(ImageFilter.GaussianBlur(radius))
        return self

    def sharpen(self, factor: float = 2.0):
        """Sharpen image"""
        enhancer = ImageEnhance.Sharpness(self.image)
        self.image = enhancer.enhance(factor)
        return self

    def contrast(self, factor: float = 1.2):
        """Adjust contrast"""
        enhancer = ImageEnhance.Contrast(self.image)
        self.image = enhancer.enhance(factor)
        return self

    def brightness(self, factor: float = 1.0):
        """Adjust brightness"""
        enhancer = ImageEnhance.Brightness(self.image)
        self.image = enhancer.enhance(factor)
        return self

    def saturation(self, factor: float = 1.0):
        """Adjust saturation"""
        enhancer = ImageEnhance.Color(self.image)
        self.image = enhancer.enhance(factor)
        return self

    def tint(self, color: str, mode: str = 'overlay', strength: float = 0.5):
        """Apply color tint"""
        rgba = self._parse_color(color)
        tint_layer = Image.new('RGBA', self.image.size, rgba[:3] + (int(255 * strength),))

        if mode == 'multiply':
            self.image = ImageChops.multiply(self.image, tint_layer)
        elif mode == 'screen':
            self.image = ImageChops.screen(self.image, tint_layer)
        else:  # overlay
            self.image = Image.alpha_composite(self.image, tint_layer)
        return self

    # === BORDERS & EFFECTS ===
    def border(self, size: int = 10, color: str = '#ffffff'):
        """Add solid border"""
        rgba = self._parse_color(color)
        new_size = (self.image.size[0] + size * 2, self.image.size[1] + size * 2)
        bordered = Image.new('RGBA', new_size, rgba)
        bordered.paste(self.image, (size, size))
        self.image = bordered
        return self

    def glow_border(self, size: int = 20, color: str = '#3b82f6'):
        """Add glowing border effect"""
        rgba = self._parse_color(color)

        # Create larger canvas
        new_size = (self.image.size[0] + size * 2, self.image.size[1] + size * 2)
        result = Image.new('RGBA', new_size, (0, 0, 0, 0))

        # Create glow layer
        glow = Image.new('RGBA', new_size, rgba)
        glow = glow.filter(ImageFilter.GaussianBlur(size // 2))

        result = Image.alpha_composite(result, glow)
        result.paste(self.image, (size, size), self.image)
        self.image = result
        return self

    def shadow(self, blur: int = 10, offset: tuple = (5, 5), color: str = '#000000'):
        """Add drop shadow"""
        rgba = self._parse_color(color)

        # Create shadow
        shadow = Image.new('RGBA', self.image.size, (0, 0, 0, 0))
        shadow.paste(rgba, (0, 0), self.image.split()[3])
        shadow = shadow.filter(ImageFilter.GaussianBlur(blur))

        # Expand canvas for offset
        new_size = (self.image.size[0] + abs(offset[0]) + blur * 2,
                    self.image.size[1] + abs(offset[1]) + blur * 2)
        result = Image.new('RGBA', new_size, (0, 0, 0, 0))

        # Paste shadow then image
        result.paste(shadow, (blur + offset[0], blur + offset[1]), shadow)
        result.paste(self.image, (blur, blur), self.image)
        self.image = result
        return self

    def round_corners(self, radius: int = 20):
        """Round the corners"""
        mask = Image.new('L', self.image.size, 255)
        draw = ImageDraw.Draw(mask)

        # Draw rounded rectangle
        draw.rounded_rectangle([(0, 0), self.image.size], radius, fill=255)

        self.image.putalpha(mask)
        return self

    # === GENERATORS ===
    def generate_favicon_set(self, output_dir: str):
        """Generate all favicon sizes"""
        sizes = [16, 32, 48, 64, 128, 180, 192, 512]
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        for size in sizes:
            resized = self.image.copy()
            resized = resized.resize((size, size), Image.Resampling.LANCZOS)

            if size == 180:
                resized.save(output / 'apple-touch-icon.png', 'PNG')
            elif size == 192:
                resized.save(output / 'android-chrome-192x192.png', 'PNG')
            elif size == 512:
                resized.save(output / 'android-chrome-512x512.png', 'PNG')
            else:
                resized.save(output / f'favicon-{size}x{size}.png', 'PNG')

        # ICO with multiple sizes
        ico_sizes = [16, 32, 48]
        ico_images = [self.image.resize((s, s), Image.Resampling.LANCZOS) for s in ico_sizes]
        ico_images[0].save(output / 'favicon.ico', format='ICO', sizes=[(s, s) for s in ico_sizes])

        print(f"✓ Generated {len(sizes) + 1} favicon files in {output_dir}")
        return self

    def generate_social_set(self, output_dir: str):
        """Generate social media image sizes"""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        sizes = {
            'og-image': (1200, 630),      # Facebook/LinkedIn OG
            'twitter-card': (1200, 600),  # Twitter
            'instagram-square': (1080, 1080),
            'instagram-story': (1080, 1920),
            'youtube-thumbnail': (1280, 720),
        }

        for name, (w, h) in sizes.items():
            img = self.image.copy()
            # Cover fit
            ratio = max(w / img.size[0], h / img.size[1])
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Center crop
            x = (img.size[0] - w) // 2
            y = (img.size[1] - h) // 2
            img = img.crop((x, y, x + w, y + h))

            img.save(output / f'{name}.png', 'PNG')

        print(f"✓ Generated {len(sizes)} social media images in {output_dir}")
        return self

    # === HELPERS ===
    def _parse_color(self, color: str) -> tuple:
        """Parse color string to RGBA tuple"""
        if color.startswith('#'):
            hex_color = color.lstrip('#')
            if len(hex_color) == 3:
                hex_color = ''.join([c*2 for c in hex_color])
            r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            return (r, g, b, 255)
        elif color.startswith('rgb'):
            import re
            nums = re.findall(r'\d+', color)
            return tuple(map(int, nums)) + (255,) if len(nums) == 3 else tuple(map(int, nums))
        else:
            # Named colors
            colors = {
                'black': (0, 0, 0, 255),
                'white': (255, 255, 255, 255),
                'blue': (59, 130, 246, 255),
                'orange': (249, 115, 22, 255),
                'slate': (30, 41, 59, 255),
            }
            return colors.get(color.lower(), (0, 0, 0, 255))

    def _calc_position(self, position: str, canvas: tuple, item: tuple) -> tuple:
        """Calculate x,y from position string"""
        cw, ch = canvas
        iw, ih = item

        positions = {
            'center': ((cw - iw) // 2, (ch - ih) // 2),
            'topleft': (10, 10),
            'topright': (cw - iw - 10, 10),
            'bottomleft': (10, ch - ih - 10),
            'bottomright': (cw - iw - 10, ch - ih - 10),
            'top': ((cw - iw) // 2, 10),
            'bottom': ((cw - iw) // 2, ch - ih - 10),
            'left': (10, (ch - ih) // 2),
            'right': (cw - iw - 10, (ch - ih) // 2),
        }
        return positions.get(position, (0, 0))

    def info(self):
        """Print image info"""
        print(f"File: {self.input_path}")
        print(f"Size: {self.image.size[0]}x{self.image.size[1]}")
        print(f"Mode: {self.image.mode}")
        print(f"Format: {self.image.format}")
        return self


def main():
    parser = argparse.ArgumentParser(description='LatticeForge Image Toolkit')
    parser.add_argument('command', help='Command to run')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', nargs='?', help='Output path')

    # Common options
    parser.add_argument('--width', '-w', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--scale', '-s', type=float)
    parser.add_argument('--fit', choices=['cover', 'contain', 'fill'], default='contain')
    parser.add_argument('--ratio', help='Aspect ratio (e.g., 16:9)')
    parser.add_argument('--gravity', default='center')
    parser.add_argument('--box', help='Crop box: x,y,w,h')
    parser.add_argument('--bg', help='Background color or image path')
    parser.add_argument('--overlay', help='Overlay image path')
    parser.add_argument('--position', '-p', default='center')
    parser.add_argument('--opacity', '-o', type=float, default=1.0)
    parser.add_argument('--text', '-t', help='Text for watermark')
    parser.add_argument('--color', '-c', default='#ffffff')
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--blur', type=int, default=5)
    parser.add_argument('--radius', type=int, default=20)
    parser.add_argument('--threshold', type=int, default=20)
    parser.add_argument('--factor', '-f', type=float, default=1.2)
    parser.add_argument('--quality', '-q', type=int, default=95)

    args = parser.parse_args()

    forge = ImageForge(args.input)

    if args.command == 'info':
        forge.info()
    elif args.command == 'crop':
        if args.box:
            x, y, w, h = map(int, args.box.split(','))
            forge.crop_box(x, y, w, h)
        elif args.ratio:
            forge.crop_ratio(args.ratio, args.gravity)
        forge.save(args.output or args.input, args.quality)
    elif args.command == 'resize':
        forge.resize(args.width, args.height, args.scale, args.fit)
        forge.save(args.output or args.input, args.quality)
    elif args.command == 'rembg':
        forge.remove_background()
        forge.save(args.output or args.input.replace('.', '_nobg.'), args.quality)
    elif args.command == 'replace-bg':
        forge.replace_background(args.bg)
        forge.save(args.output or args.input, args.quality)
    elif args.command == 'extract':
        forge.extract_from_black(args.threshold)
        forge.save(args.output or args.input.replace('.', '_extracted.'), args.quality)
    elif args.command == 'overlay':
        forge.overlay(args.overlay, args.position, args.opacity, args.scale or 1.0)
        forge.save(args.output or args.input, args.quality)
    elif args.command == 'watermark':
        forge.watermark_text(args.text, args.position, args.opacity, args.size, args.color)
        forge.save(args.output or args.input, args.quality)
    elif args.command == 'blur':
        forge.blur(args.blur)
        forge.save(args.output or args.input, args.quality)
    elif args.command == 'sharpen':
        forge.sharpen(args.factor)
        forge.save(args.output or args.input, args.quality)
    elif args.command == 'contrast':
        forge.contrast(args.factor)
        forge.save(args.output or args.input, args.quality)
    elif args.command == 'brightness':
        forge.brightness(args.factor)
        forge.save(args.output or args.input, args.quality)
    elif args.command == 'border':
        forge.border(args.size, args.color)
        forge.save(args.output or args.input, args.quality)
    elif args.command == 'glow':
        forge.glow_border(args.size, args.color)
        forge.save(args.output or args.input, args.quality)
    elif args.command == 'shadow':
        forge.shadow(args.blur, (args.size, args.size), args.color)
        forge.save(args.output or args.input, args.quality)
    elif args.command == 'round':
        forge.round_corners(args.radius)
        forge.save(args.output or args.input, args.quality)
    elif args.command == 'circle':
        forge.crop_circle()
        forge.save(args.output or args.input, args.quality)
    elif args.command == 'favicon':
        forge.generate_favicon_set(args.output or './favicons')
    elif args.command == 'social':
        forge.generate_social_set(args.output or './social')
    else:
        print(f"Unknown command: {args.command}")
        print("Run with --help for usage")


if __name__ == '__main__':
    main()
