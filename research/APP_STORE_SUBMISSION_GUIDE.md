# App Store Submission Guide

**App**: VeilPath - Mental Wellness Tarot App
**Version**: 1.0.0
**Bundle ID (iOS)**: com.veilpath.app
**Package Name (Android)**: com.veilpath.app
**Last Updated**: 2025-11-22

---

## Table of Contents

1. [Pre-Submission Checklist](#pre-submission-checklist)
2. [Apple App Store Submission](#apple-app-store-submission)
3. [Google Play Store Submission](#google-play-store-submission)
4. [App Store Assets](#app-store-assets)
5. [App Store Optimization (ASO)](#app-store-optimization-aso)
6. [Legal & Compliance](#legal--compliance)
7. [Post-Submission Monitoring](#post-submission-monitoring)

---

## Pre-Submission Checklist

### Code & Build
- [ ] All features complete and tested
- [ ] No console.log() or debug code in production
- [ ] Environment variables properly configured
- [ ] API keys secured (not in code)
- [ ] Build configuration optimized for production
- [ ] Proguard/R8 enabled (Android)
- [ ] Bitcode enabled (iOS - if required)
- [ ] Version number correct in package.json, app.json, build configs
- [ ] Build number incremented

### Testing
- [ ] QA checklist 100% complete
- [ ] No critical or high-priority bugs
- [ ] Tested on minimum supported OS versions (iOS 13+, Android 8+)
- [ ] Tested on various screen sizes
- [ ] Performance targets met (<100ms screens, 60fps)
- [ ] Memory usage within limits (<150MB peak)
- [ ] No crashes in last 7 days of testing

### Legal & Privacy
- [ ] Privacy Policy finalized and accessible
- [ ] Terms of Service finalized and accessible
- [ ] Age rating determined
- [ ] Content rating questionnaire completed
- [ ] COPPA compliance verified (if applicable)
- [ ] GDPR compliance verified (if applicable)

### Analytics & Monitoring
- [ ] Crash reporting configured (Sentry, Firebase, etc.)
- [ ] Analytics configured (Firebase, Mixpanel, etc.)
- [ ] Performance monitoring enabled
- [ ] Error logging configured

---

## Apple App Store Submission

### 1. Apple Developer Account Setup

**Prerequisites**:
- Apple Developer Account ($99/year)
- App ID registered in App Store Connect
- Signing certificates configured
- Provisioning profiles created

**Required Certificates**:
- iOS Distribution Certificate
- Push Notification Certificate (if using notifications)

### 2. App Information

**App Store Connect > My Apps > App Information**

| Field | Value |
|-------|-------|
| Name | VeilPath |
| Subtitle | Mental Wellness Through Tarot |
| Primary Language | English (U.S.) |
| Bundle ID | com.veilpath.app |
| SKU | LUNATIQ-001 |
| Category | Primary: Health & Fitness<br>Secondary: Lifestyle |
| Content Rights | Does not contain third-party content |

### 3. Pricing & Availability

- **Price**: Free (with optional IAP for premium features)
- **Availability**: All territories
- **Pre-order**: No (for v1.0)

### 4. App Privacy

**Privacy Policy URL**: https://veilpath.app/privacy

**Data Collection** (Customize based on actual implementation):

| Data Type | Collected | Purpose | Linked to User |
|-----------|-----------|---------|----------------|
| Health & Fitness | ✅ | App Functionality, Analytics | No |
| User Content | ✅ | App Functionality | No |
| Usage Data | ✅ | Analytics, Product Personalization | No |
| Diagnostics | ✅ | App Functionality | No |
| Contact Info | ❌ | N/A | N/A |
| Location | ❌ | N/A | N/A |
| Financial Info | ❌ | N/A | N/A |

### 5. Age Rating

**App Store Connect > Age Rating Questionnaire**

Expected Rating: **12+**

Key Factors:
- Infrequent/Mild Realistic Violence: No
- Infrequent/Mild Cartoon or Fantasy Violence: No
- Infrequent/Mild Mature/Suggestive Themes: No
- Infrequent/Mild Horror/Fear Themes: Yes (Tarot imagery)
- Infrequent/Mild Medical/Treatment Information: Yes (Mental health)
- Gambling: No
- Unrestricted Web Access: No
- Made for Kids: No

### 6. Version Information

**Version**: 1.0.0
**Copyright**: © 2025 VeilPath, Inc.

**What's New in This Version**:
```
Welcome to VeilPath! 🌙

Discover mental wellness through the wisdom of tarot:

✨ TAROT READINGS
• Single card, Three Card, & Celtic Cross spreads
• AI-powered interpretations with Claude
• Beautiful card artwork and animations
• Save and favorite your readings

📝 JOURNALING
• Mood tracking and insights
• CBT distortion identification
• DBT skills practice
• Private entry protection

🧘 MINDFULNESS
• Guided breathing exercises
• Meditation sessions
• Daily affirmations

🎮 GAMIFICATION
• Level up and earn XP
• Unlock achievements
• Skill trees for CBT, DBT, and Mindfulness
• Track your progress

Start your journey to self-discovery today!
```

### 7. App Review Information

**Contact Information**:
- First Name: [Your First Name]
- Last Name: [Your Last Name]
- Phone Number: [Your Phone]
- Email: [Your Email]

**Sign-in Required**: No

**Demo Account** (if applicable):
- Username: demo@veilpath.app
- Password: [Demo password]

**Notes**:
```
Thank you for reviewing VeilPath!

This app combines traditional tarot with evidence-based mental wellness practices (CBT, DBT, Mindfulness).

Key Features to Test:
1. Try a Single Card reading (tap "New Reading")
2. Get an AI interpretation (requires internet)
3. Create a journal entry (tap "Journal" tab)
4. Try a breathing exercise (Mindfulness tab)

All data is stored locally. No account required.

Privacy: Users can mark journal entries as private, which excludes them from data exports.

AI Integration: We use Anthropic's Claude API for tarot interpretations. This is clearly labeled and optional.

If you have any questions, please contact [your email].
```

### 8. Build Upload

**Using Xcode**:
```bash
# 1. Archive the app
Product > Archive

# 2. Validate the archive
Window > Organizer > Archives > Validate App

# 3. Upload to App Store Connect
Window > Organizer > Archives > Distribute App > App Store Connect

# 4. Select build in App Store Connect
App Store Connect > TestFlight > Builds
```

**Using Fastlane** (Alternative):
```bash
fastlane ios release
```

### 9. App Store Assets (iOS)

#### App Icon
- **Sizes Required**:
  - 1024x1024 (App Store)
  - 180x180 (@3x iPhone)
  - 120x120 (@2x iPhone)
  - 167x167 (@2x iPad Pro)
  - 152x152 (@2x iPad)
  - 76x76 (@1x iPad)

#### Screenshots

**iPhone 6.7" (iPhone 14 Pro Max, 15 Pro Max)** - Required
- Resolution: 1290x2796
- Count: 3-10 screenshots
- Suggested:
  1. Home screen with card spread
  2. Reading detail with AI interpretation
  3. Journal entry with mood tracking
  4. Skill tree visualization
  5. Statistics dashboard

**iPhone 6.5" (iPhone 11 Pro Max, XS Max)** - Required
- Resolution: 1242x2688
- Count: 3-10 screenshots

**iPhone 5.5" (iPhone 8 Plus)** - Required
- Resolution: 1242x2208
- Count: 3-10 screenshots

**iPad Pro (12.9")** - Optional
- Resolution: 2048x2732
- Count: 3-10 screenshots

#### App Preview Videos (Optional)
- Duration: 15-30 seconds
- Formats: .mov, .m4v, .mp4
- Max size: 500 MB

### 10. App Description

**Description** (4000 character limit):
```
Discover clarity and cultivate mental wellness with VeilPath, the app that seamlessly blends ancient tarot wisdom with modern therapeutic practices.

Whether you're seeking daily guidance, exploring your emotions, or building healthier thought patterns, VeilPath offers a unique path to self-discovery and personal growth.

✨ TAROT READINGS
• Choose from Single Card, Three Card, or Celtic Cross spreads
• Beautiful, hand-illustrated tarot deck
• AI-powered interpretations using advanced Claude AI
• Save and organize your reading history
• Favorite meaningful readings for later reflection
• Track which cards appear most frequently in your journey

📝 INTELLIGENT JOURNALING
• Express your thoughts with mood tracking
• Identify cognitive distortions (CBT)
• Practice DBT skills in real-time
• Keep private entries completely secure
• Search and filter by mood, tags, or keywords
• Export your journal as beautifully formatted Markdown

🧘 MINDFULNESS & MEDITATION
• Guided breathing exercises (Box Breathing, 4-7-8, and more)
• Meditation sessions for every experience level
• Daily affirmations to start your day right
• Track mindfulness minutes and build consistency

🎮 ENGAGING GAMIFICATION
• Level up as you engage with the app
• Unlock 50+ achievements
• Earn skill points and unlock therapeutic skills
• Three skill trees: CBT, DBT, and Mindfulness
• Watch your streak grow with daily practice
• Detailed statistics and insights

📊 PROGRESS TRACKING
• Comprehensive statistics dashboard
• Mood trends over time
• Reading insights and patterns
• Milestone celebrations
• Export all your data anytime

🔒 PRIVACY FIRST
• All data stored locally on your device
• No account required
• Mark entries as private
• Complete control over your data
• Export or delete anytime

ABOUT THE APPROACH
VeilPath combines the reflective practice of tarot with evidence-based therapeutic techniques:
• CBT (Cognitive Behavioral Therapy) for identifying thought patterns
• DBT (Dialectical Behavior Therapy) for emotional regulation
• Mindfulness practices for present-moment awareness

Whether you're a tarot enthusiast, someone exploring mental wellness tools, or simply curious about self-reflection, VeilPath offers a judgment-free space for your journey.

PERFECT FOR:
• Daily reflection and meditation
• Mood tracking and emotional awareness
• Cognitive distortion identification
• Building mindfulness habits
• Personal growth and self-discovery
• Stress reduction and relaxation

Download VeilPath today and start your journey toward greater self-awareness and emotional wellness. 🌙✨

---

SUBSCRIPTION & PRICING
VeilPath is free to download and use. Optional premium features available via in-app purchase.

SUPPORT
Questions? Contact us at support@veilpath.app

FOLLOW US
Instagram: @veilpathapp
Twitter: @veilpathapp

Privacy Policy: https://veilpath.app/privacy
Terms of Service: https://veilpath.app/terms
```

**Keywords** (100 character limit):
```
tarot,mental health,wellness,meditation,mindfulness,journaling,CBT,DBT,self-care,therapy
```

**Support URL**: https://veilpath.app/support
**Marketing URL**: https://veilpath.app

---

## Google Play Store Submission

### 1. Google Play Console Setup

**Prerequisites**:
- Google Play Developer Account ($25 one-time)
- App created in Google Play Console
- Signing key configured
- App bundle or APK ready

### 2. Store Listing

**App Name**: VeilPath
**Short Description** (80 char max):
```
Mental wellness through tarot, CBT, DBT & mindfulness. Track mood & grow.
```

**Full Description** (4000 char max):
```
[Use same description as iOS, adjusted for Android features]
```

**App Category**:
- Primary: Health & Fitness
- Secondary: Lifestyle

**Tags**:
- Mental Health
- Wellness
- Meditation
- Self-Care
- Journaling

### 3. Graphic Assets (Android)

#### App Icon
- **Size**: 512x512
- **Format**: PNG (32-bit)
- **No transparency**
- **Full bleed** (fill entire area)

#### Feature Graphic
- **Size**: 1024x500
- **Format**: PNG or JPG
- **Purpose**: Top of store listing

#### Screenshots

**Phone** - Required (2-8 screenshots)
- Min: 320px
- Max: 3840px
- Aspect ratio: 16:9 or 9:16
- Suggested sizes:
  - 1080x1920 (Portrait)
  - 1920x1080 (Landscape)

**7-inch Tablet** - Optional
- 1200x1920 or 1920x1200

**10-inch Tablet** - Optional
- 1600x2560 or 2560x1600

#### Promo Video (Optional)
- YouTube URL
- Will appear in store listing

### 4. Content Rating

**Complete Content Rating Questionnaire**:

| Category | Response |
|----------|----------|
| Violence | None |
| Sexual Content | None |
| Profanity | None |
| Controlled Substances | None |
| User Interaction | Users can interact |
| User-Generated Content | Yes (Journal entries) |
| Share Location | No |
| Personal Info | No |
| Medical/Health | References mental health |

Expected Rating: **PEGI 12** / **ESRB Everyone 10+**

### 5. Pricing & Distribution

- **Price**: Free
- **Countries**: All available
- **Device Categories**: Phone, Tablet
- **Android Versions**: 8.0+ (API 26+)

### 6. App Content

**Privacy Policy URL**: https://veilpath.app/privacy

**Ads**: No (or Yes if implementing ads)
**In-App Purchases**: No (or Yes if implementing IAP)
**Target Audience**: Ages 13+

**Data Safety Section**:

| Data Type | Collected | Shared | Purpose |
|-----------|-----------|--------|---------|
| Personal info | No | No | N/A |
| Health and fitness | Yes | No | App functionality |
| Messages | Yes | No | App functionality (journal) |
| App activity | Yes | No | Analytics |
| App info and performance | Yes | No | Crash reports |

**Security Practices**:
- ✅ Data is encrypted in transit
- ✅ Users can request data deletion
- ✅ Data is not shared with third parties
- ❌ Data is not encrypted at rest (AsyncStorage)

### 7. Release Management

**Release Name**: 1.0.0 - Initial Release

**Release Notes**:
```
🎉 Welcome to VeilPath! 🌙

Discover mental wellness through the wisdom of tarot.

✨ Features:
• Tarot readings with AI interpretations
• Intelligent journaling with CBT/DBT support
• Guided meditation and breathing exercises
• Gamification with levels, achievements, and skill trees
• Comprehensive progress tracking
• Privacy-first design

Start your journey to self-discovery today!
```

**Release Type**:
- [ ] Internal Testing
- [ ] Closed Testing (Alpha/Beta)
- [x] Production

### 8. Build Upload

**Using Expo**:
```bash
# Build production AAB
eas build --platform android --profile production

# Download the AAB
eas build:download --platform android --latest

# Upload to Play Console manually or via Fastlane
```

**Using Android Studio**:
```bash
# Generate signed AAB
Build > Generate Signed Bundle/APK > Android App Bundle

# Upload to Play Console
Release > Production > Create New Release > Upload AAB
```

---

## App Store Assets

### Screenshot Guidelines

**Best Practices**:
- Use device frames for context
- Show key features in order
- Include captions/annotations
- Use consistent styling
- Localize for major markets
- Update seasonally (optional)

**Suggested Screenshot Flow**:
1. **Home/Hero**: Show beautiful card spread or welcome screen
2. **Reading**: Display AI interpretation with cards
3. **Journal**: Show mood tracking and CBT features
4. **Mindfulness**: Display breathing exercise or meditation
5. **Progress**: Show statistics or skill tree
6. **Value Prop**: Highlight key benefits

**Tools for Screenshots**:
- Figma (design tool)
- Screely (add device frames)
- AppLaunchpad (screenshot generator)
- Previewed (mockup generator)

### App Icon Design Guidelines

**iOS**:
- No alpha channel
- No rounded corners (iOS adds them)
- Simple, recognizable design
- Legible at small sizes
- Avoid text if possible

**Android**:
- Can include transparency
- Adaptive icon layers (foreground + background)
- Safe zone for circular masks
- Follow Material Design guidelines

**Suggested Concept**:
- Moon/crescent symbol
- Tarot card silhouette
- Mystical but approachable
- Purple/indigo color scheme

---

## App Store Optimization (ASO)

### Keyword Research

**High-Volume Keywords** (Suggested):
- tarot
- tarot reading
- mental health
- meditation
- mindfulness
- journaling
- self care
- CBT therapy
- wellness app
- mood tracker

**Long-Tail Keywords**:
- daily tarot card
- tarot journal
- mental health journal
- CBT exercises
- DBT skills
- guided meditation app

**Keyword Tools**:
- App Store Connect (iOS)
- Google Play Console (Android)
- Sensor Tower
- App Annie
- Mobile Action

### Title & Subtitle Optimization

**iOS Title** (30 char max): VeilPath

**iOS Subtitle** (30 char max):
- Option 1: Mental Wellness Through Tarot
- Option 2: Tarot, Journaling & Mindfulness
- Option 3: CBT, DBT & Tarot for Wellness

**Android Title** (50 char max):
- VeilPath: Tarot, Mental Health & Mindfulness

### Conversion Optimization

**A/B Testing Elements**:
- Screenshot order
- Feature highlights
- Icon variations
- Short description variants
- Video preview

**Metrics to Track**:
- Impression-to-install rate
- Install-to-active rate
- Day 1, 7, 30 retention
- Organic vs. paid traffic
- Keyword rankings

---

## Legal & Compliance

### Privacy Policy

**Must Include**:
- What data is collected
- How data is used
- How data is stored
- Data retention policy
- User rights (access, deletion, portability)
- Contact information
- Last updated date

**Template**: Use iubenda.com or termly.io for generation

**Accessibility**: Must be accessible in-app and via URL

### Terms of Service

**Must Include**:
- License grant
- Acceptable use policy
- Intellectual property rights
- Disclaimer of warranties
- Limitation of liability
- Governing law
- Dispute resolution

### Content Guidelines

**Apple App Store Review Guidelines**:
- Section 1.4: Physical Harm (Mental health claims)
- Section 2.5: Software Requirements
- Section 4.2: Minimum Functionality
- Section 5.1: Privacy

**Google Play Developer Policies**:
- Health & Medical apps policy
- User-Generated Content policy
- Privacy & Data Handling

### Age Rating & Content

**Ensure Compliance**:
- No medical claims or treatment promises
- Clear labeling of AI-generated content
- Disclaimers about not replacing professional help
- Age-appropriate content (12+)

---

## Post-Submission Monitoring

### Review Process Timeline

**Apple**:
- Typical: 24-48 hours
- Can be rejected and require resubmission
- Appeals process available

**Google**:
- Typical: 1-7 days
- Rolling review process
- Policy enforcement can be automated

### Common Rejection Reasons

**iOS**:
- Incomplete information
- Inaccurate screenshots
- Privacy policy issues
- Crashes on review
- Misleading marketing

**Android**:
- Policy violations
- Incomplete content rating
- Privacy policy missing/incorrect
- Technical issues

### Launch Day Checklist

**Pre-Launch** (1 day before):
- [ ] Monitor build status
- [ ] Prepare social media posts
- [ ] Brief customer support team
- [ ] Set up monitoring dashboards
- [ ] Test crash reporting

**Launch Day**:
- [ ] Verify app is live in stores
- [ ] Test download and install
- [ ] Monitor crash reports
- [ ] Track analytics
- [ ] Respond to reviews
- [ ] Post on social media

**Post-Launch** (Week 1):
- [ ] Daily crash report reviews
- [ ] Respond to all reviews
- [ ] Monitor performance metrics
- [ ] Collect user feedback
- [ ] Plan hotfix if needed

---

## Submission Commands Reference

### iOS (Expo/EAS)
```bash
# Build production IPA
eas build --platform ios --profile production

# Submit to App Store
eas submit --platform ios

# Check submission status
eas submission:list --platform ios
```

### Android (Expo/EAS)
```bash
# Build production AAB
eas build --platform android --profile production

# Submit to Play Store
eas submit --platform android

# Check submission status
eas submission:list --platform android
```

### Manual Submission
See platform-specific instructions above.

---

## Resources

**Official Documentation**:
- [App Store Review Guidelines](https://developer.apple.com/app-store/review/guidelines/)
- [Google Play Policy Center](https://play.google.com/about/developer-content-policy/)
- [Expo EAS Submit](https://docs.expo.dev/submit/introduction/)

**Tools**:
- [App Store Connect](https://appstoreconnect.apple.com/)
- [Google Play Console](https://play.google.com/console/)
- [Fastlane](https://fastlane.tools/)
- [Screenshot Creator](https://www.screenshot.rocks/)

**Communities**:
- r/iOSProgramming
- r/androiddev
- Indie Hackers
- Product Hunt

---

**Document Owner**: [Your Name]
**Last Review**: 2025-11-22
**Next Review**: Before submission
**Status**: Draft
