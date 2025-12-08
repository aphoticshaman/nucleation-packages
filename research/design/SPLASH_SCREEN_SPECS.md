# Splash Screen Design Specifications

**App:** VeilPath
**Purpose:** Launch screen (iOS) and Splash screen (Android)
**Date:** January 22, 2025

---

## Table of Contents

1. [Overview & Purpose](#overview--purpose)
2. [Design Requirements](#design-requirements)
3. [Technical Specifications](#technical-specifications)
4. [Design Concept](#design-concept)
5. [Color Palette](#color-palette)
6. [Layout & Composition](#layout--composition)
7. [Animation (Optional)](#animation-optional)
8. [Platform-Specific Guidelines](#platform-specific-guidelines)
9. [Accessibility](#accessibility)
10. [Deliverables](#deliverables)

---

## Overview & Purpose

### What is a Splash Screen?

**iOS (Launch Screen):**
- Shown briefly while app loads (~0.5-2 seconds)
- Should be simple, static, and fast to render
- Apple discourages splash screens with branding/marketing
- Purpose: Smooth transition, give impression of fast loading

**Android (Splash Screen):**
- Native splash screen API (Android 12+) shows app icon + background
- Custom splash screens allowed but must comply with Material Design
- Can include brief branding, logo, or tagline
- Purpose: Reinforce brand while app initializes

---

### VeilPath Splash Screen Goals

1. **Create calm, peaceful first impression** (set tone for mental wellness)
2. **Visually connect to app icon** (reinforce brand recognition)
3. **Load instantly** (lightweight, no heavy assets)
4. **Work in all orientations** (portrait, landscape)
5. **Support light and dark mode**
6. **Comply with iOS/Android guidelines**

---

## Design Requirements

### Must-Haves
- ✅ Simple, minimal design (fast rendering)
- ✅ Consistent with app icon aesthetic (purple, teal, moon/path theme)
- ✅ Readable app name "VeilPath"
- ✅ Works in portrait AND landscape orientations
- ✅ Dark mode variant (or works in both modes)
- ✅ No text cutoff on notched devices (iPhone X+, Android punch-hole)

### Should-Haves
- App icon or simplified version of icon
- Tagline or brief descriptor ("Your Journey to Mental Wellness")
- Smooth fade-in animation (optional, iOS only)

### Must-Avoids
- ❌ Complex illustrations (slow to load)
- ❌ Heavy gradients or effects (performance issues)
- ❌ Marketing messages or feature lists (against iOS guidelines)
- ❌ Text near edges (notch/punch-hole will cut off)
- ❌ Photos or realistic imagery

---

## Technical Specifications

### iOS Launch Screen

**File Format:** Storyboard (.storyboard) or single PNG (legacy)
**Recommended Approach:** Storyboard with solid color + icon + text (scalable)

**Sizes (if using PNG, legacy method):**
- iPhone 14 Pro Max (Portrait): 1290 x 2796 @3x
- iPhone 14 Pro (Portrait): 1179 x 2556 @3x
- iPhone 14 (Portrait): 1170 x 2532 @3x
- iPhone SE (Portrait): 750 x 1334 @2x
- iPad Pro 12.9" (Portrait): 2048 x 2732 @2x

**Safe Area:**
- Top: 44pt (132px @3x) — status bar
- Bottom: 34pt (102px @3x) — home indicator (iPhone X+)
- Sides: 20pt (60px @3x) — notch clearance

**Duration:** ~0.5-2 seconds (system controlled)

**Apple Guidelines:**
> "Design a launch screen that's nearly identical to the first screen of your app. If you include elements that look different when the app finishes launching, people can experience an unpleasant flash between the launch screen and the first screen of the app." — Apple HIG

**Translation:** Keep it simple, minimal, similar to loading state.

---

### Android Splash Screen

**Android 12+ (Native Splash Screen API):**
- App icon (adaptive icon foreground layer): 240 x 240 dp
- Background color: Single color (no gradient in native API)
- Animated icon (optional): Vector drawable or AnimatedVectorDrawable
- Branding image (optional): 200 x 80 dp (bottom of screen)

**Pre-Android 12 (Custom Splash via Theme):**
- Background: Solid color or drawable (XML)
- Icon/Logo: Centered, 192 x 192 dp
- Tagline: Optional, below logo

**Sizes (if using image assets):**
- MDPI (1x): 320 x 480
- HDPI (1.5x): 480 x 800
- XHDPI (2x): 720 x 1280
- XXHDPI (3x): 1080 x 1920
- XXXHDPI (4x): 1440 x 2560

**Safe Area:**
- Top: 24dp (status bar)
- Bottom: 48dp (navigation bar)
- Sides: 16dp

**Duration:** ~0.5-1.5 seconds (user-controlled via SplashScreen API)

---

## Design Concept

### Recommended Design: **Calm Awakening**

**Visual Description:**
- Gradient background (purple to midnight blue, same as icon)
- App icon centered (or simplified version)
- "VeilPath" wordmark below icon
- Optional tagline: "Your Journey to Mental Wellness" in small text
- Minimalist, peaceful, welcoming

**Layout:**
```
┌─────────────────────────────┐
│                             │
│         [Status Bar]        │ ← Top safe area (44pt)
│                             │
│                             │
│                             │
│          ┌─────┐            │
│          │     │            │ ← App Icon (120pt)
│          │ 🌙  │            │    Centered vertically
│          └─────┘            │
│                             │
│         VeilPath            │ ← App name (24pt, white)
│                             │
│   Your Journey to Mental    │ ← Tagline (14pt, white 70%)
│         Wellness            │
│                             │
│                             │
│                             │
│      [Home Indicator]       │ ← Bottom safe area (34pt)
└─────────────────────────────┘
```

---

### Alternative Design: **Minimal Gradient**

**Visual Description:**
- Full-screen gradient (purple to blue)
- No icon, no text
- Just pure color transition
- Fades directly into app

**Why this works:**
- Apple-recommended approach (minimal)
- Extremely fast to load
- Smooth transition to app (first screen also has gradient)
- Works perfectly in dark mode

**Layout:**
```
┌─────────────────────────────┐
│                             │
│                             │
│                             │
│    [Deep Purple #6B46C1]    │
│            ↓                │
│      [Gradient Fade]        │
│            ↓                │
│   [Midnight Blue #2D3748]   │
│                             │
│                             │
│                             │
└─────────────────────────────┘
```

**Note:** This is the safest, fastest option for iOS.

---

### Alternative Design: **Icon + Glow**

**Visual Description:**
- Solid deep purple background
- Large app icon in center (glowing effect)
- No text
- Subtle radial glow behind icon (teal/purple)

**Layout:**
```
┌─────────────────────────────┐
│                             │
│                             │
│                             │
│        ┌─────────┐          │
│        │  :::    │          │ ← App Icon (150pt)
│        │ :🌙::   │          │    With radial glow
│        │  :::    │          │    behind it
│        └─────────┘          │
│                             │
│                             │
│                             │
└─────────────────────────────┘
```

**Best for:** Android (allows more branding than iOS)

---

## Color Palette

### Background Colors

**Option 1: Gradient (Recommended for Visual Interest)**
- Top: Deep Purple `#6B46C1`
- Bottom: Midnight Blue `#2D3748`
- Gradient Type: Linear, top-to-bottom

**Option 2: Solid (Recommended for Speed)**
- Background: Deep Purple `#6B46C1`
- Simplest, fastest to render

**Dark Mode:**
- Background: Midnight Blue `#2D3748` (darker than light mode)
- Or: Keep same gradient (already dark)

---

### Text Colors

**App Name "VeilPath":**
- Color: White `#FFFFFF`
- Opacity: 100%
- Font: San Francisco (iOS), Roboto (Android), or custom brand font

**Tagline:**
- Color: White `#FFFFFF`
- Opacity: 70% (softer, secondary)

---

### Icon Treatment

**Option A: Full Color Icon**
- Use actual app icon (with path, moon, gradient)
- Size: 100-120pt (300-360px @3x)

**Option B: Simplified Icon**
- Just the crescent moon (no path, simplified)
- Single color: Teal `#4FD1C5` or Gold `#F6E05E`
- Faster to render, cleaner

**Option C: Monochrome Icon**
- Icon in white `#FFFFFF` (60% opacity)
- Blends with background, minimal

---

## Layout & Composition

### Vertical Centering

**For Portrait:**
- Icon: Centered horizontally, 40% from top
- Text: Centered horizontally, below icon with 20pt spacing

**For Landscape:**
- Icon: Smaller (80pt instead of 120pt)
- Text: Same size, 16pt spacing

---

### Typography

**App Name "VeilPath":**
- Font: SF Pro Display (iOS), Roboto (Android), or custom
- Weight: Semibold or Bold
- Size: 28pt (portrait), 24pt (landscape)
- Letter Spacing: -0.5pt (tighter, more refined)
- Alignment: Center

**Tagline:**
- Font: SF Pro Text (iOS), Roboto (Android)
- Weight: Regular
- Size: 16pt (portrait), 14pt (landscape)
- Letter Spacing: 0pt (default)
- Alignment: Center
- Color: White 70% opacity

---

### Spacing

**Vertical Rhythm:**
- Icon to Text: 24pt
- App Name to Tagline: 8pt
- Top Safe Area: 44pt (iOS), 24dp (Android)
- Bottom Safe Area: 34pt (iOS), 48dp (Android)

**Horizontal Margins:**
- Text: 40pt from edges (ensures readability on all devices)

---

## Animation (Optional)

### iOS (Subtle Fade-In)

Apple discourages complex animations, but a subtle fade is acceptable:

**Timeline:**
1. **0.0s:** Splash screen appears (static)
2. **0.3s:** Fade in icon (opacity 0% → 100%, duration 0.4s)
3. **0.5s:** Fade in text (opacity 0% → 100%, duration 0.3s)
4. **1.0s:** Transition to main app (crossfade)

**Implementation:** Use `UIView.animate` in Swift

```swift
// Pseudocode
UIView.animate(duration: 0.4, delay: 0.3) {
    iconView.alpha = 1.0
}
UIView.animate(duration: 0.3, delay: 0.5) {
    textView.alpha = 1.0
}
```

---

### Android 12+ (Icon Animation)

Android 12's native splash screen API supports animated icons:

**Icon Animation Options:**
1. **Pulse Glow:** Icon scales slightly (1.0 → 1.05 → 1.0) with glow
2. **Fade In:** Icon fades in (opacity 0% → 100%)
3. **Moon Phase:** Crescent moon "fills" from new moon to crescent

**Implementation:** AnimatedVectorDrawable (XML)

**Recommendation:** Keep it subtle (< 1 second animation)

---

### Pre-Android 12 (Custom Fade)

For older Android versions using custom splash Activity:

**Timeline:**
1. **0.0s:** Splash screen appears (static)
2. **0.5s:** Crossfade to main app

**Implementation:** Activity theme + exit animation

---

## Platform-Specific Guidelines

### iOS Launch Screen Best Practices

**Apple's Recommendations:**
- ✅ Use a static image or simple storyboard
- ✅ Make it nearly identical to the first app screen
- ✅ Avoid text (except essential app name)
- ✅ Avoid advertising or branding messages
- ❌ Don't use launch screen as splash screen (marketing)
- ❌ Don't include loading indicators (system shows progress)

**Our Approach:**
- Simple gradient background + icon + name
- No marketing text, no "Loading...", no spinners
- Matches first screen aesthetic (gradient background)

---

### Android Splash Screen Best Practices

**Material Design Recommendations:**
- ✅ Use Android 12+ native API when possible
- ✅ Keep branding minimal (icon + background color)
- ✅ Limit animation to < 1 second
- ✅ Provide dark mode variant
- ❌ Don't show splash > 2 seconds (bad UX)
- ❌ Don't use splash for ads or messages

**Our Approach:**
- Android 12+: Native splash screen API (icon + purple background)
- Pre-Android 12: Custom theme with drawable (gradient + icon + name)

---

## Accessibility

### High Contrast Mode

**iOS:**
- Increase text contrast (white 100% instead of 70% for tagline)
- Ensure 4.5:1 contrast ratio for text on gradient

**Android:**
- Provide high-contrast variant if using custom splash
- System handles contrast for native splash screen API

**Testing:**
- iOS: Settings → Accessibility → Display → Increase Contrast
- Android: Settings → Accessibility → High Contrast Text

---

### Reduced Motion

**iOS:**
- Detect `UIAccessibility.isReduceMotionEnabled`
- Skip fade-in animations if enabled
- Show static splash screen

**Android:**
- Detect `Settings.Global.TRANSITION_ANIMATION_SCALE == 0`
- Skip icon animation if reduce motion enabled

**Implementation (React Native):**
```javascript
import { AccessibilityInfo } from 'react-native';

AccessibilityInfo.isReduceMotionEnabled().then(enabled => {
  if (enabled) {
    // Skip animations
  }
});
```

---

### Dark Mode

**Automatic Detection:**
- iOS: Use `userInterfaceStyle` in storyboard (light/dark variants)
- Android: Use `night` qualifier in resources

**Dark Mode Splash:**
- Background: Midnight Blue `#2D3748` (darker)
- Text: White `#FFFFFF` (100% opacity)
- Icon: Same or slightly brighter

**Testing:**
- iOS: Settings → Developer → Dark Appearance
- Android: Settings → Display → Dark theme

---

## Deliverables

### iOS (Storyboard Approach - Recommended)

**File:** `LaunchScreen.storyboard`

**Contents:**
- View Controller with:
  - Background: Gradient view (purple to blue) OR solid purple
  - Image View: App icon (centered, 120pt x 120pt)
  - Label: "VeilPath" (centered below icon)
  - Label: "Your Journey to Mental Wellness" (centered below name, optional)

**Constraints:**
- Icon: Center X, Center Y with offset -60pt
- Name: Center X, Top to Icon Bottom + 24pt
- Tagline: Center X, Top to Name Bottom + 8pt
- Margins: Leading/Trailing ≥ 40pt

**Assets Needed:**
- `launch-icon.png` (360x360px @3x, 240x240px @2x, 120x120px @1x)

---

### iOS (Legacy PNG Approach)

**Files (all @3x):**
- `LaunchImage-iPhone14ProMax.png` (1290 x 2796)
- `LaunchImage-iPhone14Pro.png` (1179 x 2556)
- `LaunchImage-iPhone14.png` (1170 x 2532)
- `LaunchImage-iPhoneSE.png` (750 x 1334 @2x)
- `LaunchImage-iPadPro.png` (2048 x 2732 @2x)

**Format:** PNG, RGB, no transparency

---

### Android 12+ (Native Splash Screen API)

**Files:**
- `res/values/themes.xml` (splash theme config)
- `res/drawable/splash_icon.xml` (vector drawable, icon)
- `res/values/colors.xml` (background color `#6B46C1`)

**Configuration (themes.xml):**
```xml
<style name="Theme.VeilPath.Splash" parent="Theme.SplashScreen">
    <item name="windowSplashScreenBackground">@color/splash_background</item>
    <item name="windowSplashScreenAnimatedIcon">@drawable/splash_icon</item>
    <item name="windowSplashScreenAnimationDuration">500</item>
    <item name="postSplashScreenTheme">@style/Theme.VeilPath</item>
</style>
```

**Icon:** 240 x 240 dp (adaptive icon foreground layer)

---

### Android < 12 (Custom Splash)

**Files:**
- `res/drawable/splash_background.xml` (gradient or solid color)
- `res/layout/splash_screen.xml` (layout with icon + text)
- `res/values/styles.xml` (splash theme)

**Layout (splash_screen.xml):**
```xml
<LinearLayout
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:gravity="center"
    android:background="@drawable/splash_background">

    <ImageView
        android:layout_width="120dp"
        android:layout_height="120dp"
        android:src="@drawable/app_icon" />

    <TextView
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="VeilPath"
        android:textSize="24sp"
        android:textColor="#FFFFFF"
        android:layout_marginTop="24dp" />

    <TextView
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Your Journey to Mental Wellness"
        android:textSize="14sp"
        android:textColor="#B3FFFFFF"
        android:layout_marginTop="8dp" />
</LinearLayout>
```

---

## Implementation Checklist

### iOS
- [ ] Create `LaunchScreen.storyboard` in Xcode
- [ ] Add gradient background view (purple to blue)
- [ ] Add app icon image view (120pt, centered)
- [ ] Add "VeilPath" label (28pt, semibold, white)
- [ ] Add tagline label (16pt, regular, white 70%, optional)
- [ ] Configure constraints for all device sizes
- [ ] Test on iPhone SE, iPhone 14, iPhone 14 Pro Max
- [ ] Test in light and dark mode
- [ ] Test with "Increase Contrast" enabled

---

### Android
- [ ] Update `themes.xml` with SplashScreen theme (Android 12+)
- [ ] Create `splash_icon.xml` vector drawable
- [ ] Set `windowSplashScreenBackground` color to `#6B46C1`
- [ ] Set splash duration to 500ms
- [ ] Create fallback splash for Android < 12 (optional)
- [ ] Test on Pixel 6, Samsung Galaxy S22, older devices
- [ ] Test in light and dark mode
- [ ] Test with reduced motion enabled

---

## Testing Checklist

### Visual Testing
- [ ] Icon centered on all device sizes
- [ ] Text readable (not cut off by notch/punch-hole)
- [ ] Colors match app icon
- [ ] Gradient smooth (no banding)
- [ ] Works in portrait and landscape

### Performance Testing
- [ ] Loads in < 0.5 seconds (cold start)
- [ ] No flicker or flash between splash and app
- [ ] No layout shifts or reflows

### Accessibility Testing
- [ ] High contrast mode (text readable)
- [ ] Dark mode (colors appropriate)
- [ ] Reduce motion (animations skip)

---

## Recommended Approach (Summary)

**For fastest implementation:**

1. **iOS:** LaunchScreen.storyboard with:
   - Solid purple background `#6B46C1`
   - App icon centered (120pt)
   - "VeilPath" text below
   - No animations

2. **Android 12+:** Native SplashScreen API with:
   - Purple background `#6B46C1`
   - App icon (adaptive foreground layer)
   - 500ms duration

3. **Android < 12:** Theme-based splash with:
   - Gradient drawable (purple to blue)
   - Centered icon + text
   - Static (no custom Activity)

**Time Estimate:** 2-4 hours total

---

**Questions?** Contact: support@veilpath.app
**Last Updated:** January 22, 2025
