# VeilPath Community UX Design Specification

## Target Demographic
- **Primary**: Women 25-45, spiritually curious, wellness-focused
- **Secondary**: Gen Z (18-24), millennial spiritual seekers
- **Mindset**: Values authenticity, self-improvement, connection over competition
- **Usage Pattern**: Quick daily engagement (5-10 min), deeper weekend dives

---

## Design Principles

### 1. Sanctuary, Not Social Media
This isn't Twitter or Reddit. It's a **sacred gathering space**. The UX should feel:
- Calming, not stimulating
- Thoughtful, not reactive
- Intimate, not broadcast
- Supportive, not performative

### 2. Minimal Cognitive Load
Users come here to decompress, not doomscroll:
- Familiar patterns (reduce learning curve)
- Clean layouts, generous white space
- Content-first hierarchy
- No notification bombardment

### 3. Quality Over Quantity
Reward depth, not spam:
- Encourage thoughtful posts over hot takes
- Highlight wisdom, not volume
- No visible follower counts
- No algorithmic "virality"

---

## User Identity System

### Avatar: Card Back Display
- User's **equipped card back** displayed as thumbnail (32x48px)
- Shows collecting progress/taste
- Animated backs play subtle loop
- Placeholder for new users: `Celestial Default`

### Display Name
- Optional custom name (vs username)
- Filtered for inappropriate content
- 2-20 characters

### Title System (Earned Only - NOT Purchasable)
Users select from **unlocked titles** earned through achievements.

#### Title Categories:

**Progression Titles** (Level-based)
| Level | Title |
|-------|-------|
| 1-4 | Seeker |
| 5-9 | Initiate |
| 10-19 | Practitioner |
| 20-29 | Adept |
| 30-39 | Mystic |
| 40-49 | Oracle |
| 50+ | Sage |

**Activity Titles** (Engagement milestones)
| Achievement | Title |
|-------------|-------|
| 7-day streak | Dawn Devotee |
| 30-day streak | Moon Faithful |
| 100-day streak | Eternal Flame |
| 100 readings | Card Whisperer |
| 500 readings | Deck Master |
| 50 journal entries | Soul Scribe |
| 100 journal entries | Chronicle Keeper |
| First community post | Voice Awakened |
| 10 liked posts | Resonant Voice |
| 50 liked posts | Community Beacon |
| 100 liked posts | Guiding Light |

**Special Titles** (Rare achievements)
| Achievement | Title |
|-------------|-------|
| Full Major Arcana collection | Arcana Collector |
| All card backs unlocked | Back Collector |
| Beta tester | Founding Oracle |
| Premium subscriber | Patron of the Veil |
| Lifetime subscriber | Eternal Patron |
| 1-year anniversary | Year Walker |
| All spreads completed | Spread Savant |
| Found an Easter egg | Veil Piercer |

**Community Titles** (Earned through helpfulness)
| Achievement | Title |
|-------------|-------|
| 10 helpful replies | Kind Spirit |
| 50 helpful replies | Wisdom Sharer |
| Verified helpful answers | Community Guide |

### Title Display
```
[CardBack Thumbnail] DisplayName
        Title
```

Example:
```
[Animated Galaxy Back] MoonlitSoul
       Eternal Patron · Oracle
```

---

## Channel Architecture

### Channel Types

#### 1. Daily Rituals (High-frequency, quick engagement)
**Daily Card** `#daily_card`
- Share your daily draw + brief reflection
- Auto-attached card image from reading
- 280-character limit (Twitter-style brevity)
- Sorted: Recent first
- Gamified: "Share 7 days in a row" achievement

**Intentions** `#intentions`
- Morning intention setting
- "Today I will..." prompts
- Brief, affirmation-style posts
- Can tag your draw if desired

#### 2. Forum Discussions (Depth, async)
**Tarot Talk** `#tarot_talk`
- Card meanings, spread ideas, interpretation help
- Threaded replies
- Can request reading feedback
- Sorted: Last activity

**General** `#general`
- Off-topic but on-vibe
- Life updates, check-ins
- Community events/discussions

**Feature Requests** `#ideas`
- Suggest improvements
- Upvote system (not likes)
- Dev visibility flag
- Monthly "implemented" roundup

#### 3. Support Channels
**Help & Support** `#help`
- LLM-assisted (Vera responds automatically)
- Escalation to human support queue
- FAQ auto-responses
- Searchable archive

#### 4. Premium Channels
**Spreads & Rituals** `#spreads` (Premium)
- Advanced spread discussions
- Ritual sharing
- Deeper spiritual content

**Premium Lounge** `#lounge` (Premium)
- Exclusive community space
- Direct dev Q&A sessions
- Early feature previews

---

## Post Types

### 1. Text Post (Default)
- Title (optional, 100 char max)
- Body (2000 char max)
- Supports markdown: **bold**, *italic*, `code`
- No raw links (anti-spam)

### 2. Daily Card Share
- Auto-generated from reading
- Card image embedded
- Optional reflection text (280 char)
- Channel-specific (#daily_card only)

### 3. Question Post
- Marked with ? indicator
- Can be marked "Answered"
- Helpful reply gets highlighted
- Used in #tarot_talk, #help

### 4. Poll (Future)
- Simple yes/no or multiple choice
- Results visible after voting
- 48-hour duration

---

## Interaction Patterns

### Reactions (Not Likes)
Instead of generic "likes", contextual reactions:

| Reaction | Emoji | Meaning |
|----------|-------|---------|
| Resonate | ✨ | "I feel this" |
| Helpful | 🙏 | "This helped me" |
| Insightful | 💡 | "I learned something" |
| Support | 💜 | "Sending love" |

**Why not hearts?** Hearts create popularity contests. These reactions indicate *value type*, not social approval.

### Reply Threading
- Top-level replies shown by default
- Nested replies collapsed (click to expand)
- Max 3 levels deep (prevents rabbit holes)
- "View X more replies" pagination

### Bookmarks (Save for Later)
- Private collections
- Categories: "Helpful", "Inspiration", "To Try"
- No public "saved" counts

### Report System
- Hidden flag (not visible to reporter)
- Categories: Spam, Harassment, Harmful, Off-topic
- Optional description
- Routes to moderation queue

---

## Feed Algorithms

### Default: Chronological
- Most recent first
- No engagement-based boosting
- "Show posts from people I follow" filter option

### Trending (Opt-in Tab)
- Posts with high recent engagement
- Weighted toward ✨ and 💡 reactions
- De-weighted after 48 hours
- Never auto-shown

### Following
- Posts from followed users only
- Separate tab, not mixed

---

## Visual Design

### Color Usage
- Channel icons: Category-appropriate emoji
- Premium channels: Gold accent border
- Unread indicator: Subtle dot
- Author titles: Muted gold/brass
- Reactions: Contextual emoji (no color coding)

### Card Components

#### Post Card
```
┌────────────────────────────────────────────┐
│ [Avatar] DisplayName                    1h │
│          Title                             │
├────────────────────────────────────────────┤
│                                            │
│ Post content here with generous line       │
│ height for readability...                  │
│                                            │
│ [Embedded Card Image if Daily Share]       │
│                                            │
├────────────────────────────────────────────┤
│ ✨ 12   🙏 4   💡 2   💜 1    💬 3    🔖   │
└────────────────────────────────────────────┘
```

#### Channel List Item
```
┌──────────────────────────────────────────┐
│ 🌅  Daily Card                    • 24 new│
│     Share your daily draw                 │
└──────────────────────────────────────────┘
```

### Responsive Behavior

**Mobile (< 600px)**
- Full-width cards
- Bottom tab navigation (channels)
- Floating compose FAB
- Pull-to-refresh

**Tablet (600-1024px)**
- Channel sidebar (collapsible)
- Cards with padding
- Compose bar at bottom

**Desktop (> 1024px)**
- Three-column: Channels | Feed | Trending
- Sticky compose at top of feed
- Keyboard shortcuts

---

## Moderation

### Automated (Claude-Powered)
All posts pass through moderation before display:

1. **Quick Check** (< 100ms)
   - Keyword blocklist
   - Spam patterns
   - URL filtering

2. **Content Review** (< 2s for flagged)
   - Sentiment analysis
   - Mental health keyword detection
   - Harassment detection
   - Contextual appropriateness

### Moderation Actions
| Severity | Action | User Sees |
|----------|--------|-----------|
| Clean | Approve | Instant post |
| Sensitive | Warn | "Content may be sensitive" label |
| Mental Health | Resources | Post + crisis resources shown |
| Harmful | Block | "Unable to post. Please revise." |
| Spam | Shadow | User sees post, others don't |

### Human Escalation
- Appeals queue for blocked posts
- Report review (24hr response)
- Repeat offender review

### Crisis Protocol
If mental health crisis keywords detected:
1. Block post from publishing
2. Show immediate resources modal
3. Offer direct chat with Vera (supportive)
4. Optional: Connect to external hotline

---

## Onboarding

### First Visit
1. Welcome modal: "Welcome to The Gathering"
2. Brief community guidelines
3. Encouraged to set display name + title
4. Directed to #daily_card for first share

### Tooltips
- First post: "Great job sharing! Posts appear instantly."
- First reaction: "Reactions show how posts resonate"
- First reply: "Threaded replies keep conversations organized"

---

## Metrics We Care About

### Health Metrics (Monitor)
- Daily Active Posters
- Reply Rate (% posts with replies)
- Helpful Reaction Ratio
- Report Rate (should be < 1%)
- Time to First Reply

### Vanity Metrics (Ignore)
- Total post count
- Total users
- Likes per post
- Follower counts

### Anti-Metrics (Red Flags)
- Time spent (we want *quality* time)
- Notification clicks
- Rage engagement

---

## What We're NOT Building

| Feature | Why Not |
|---------|---------|
| Follower counts | Creates popularity hierarchy |
| Public profiles with post history | Privacy concerns |
| DMs | Harassment vector, scope creep |
| Stories/ephemeral content | Not aligned with thoughtful vibe |
| Live video | Scope creep, moderation nightmare |
| Hashtags | Creates spam incentives |
| Share to external | Keep community intimate |
| Algorithmic feed | Manipulation, engagement addiction |
| Notifications for likes | Dopamine hack, not value |

---

## Implementation Priority

### Phase 1: Foundation
- [ ] Channel structure with #daily_card, #general, #tarot_talk
- [ ] Basic post/reply functionality
- [ ] Reaction system
- [ ] Title selection from achievements
- [ ] Card back avatars

### Phase 2: Moderation
- [ ] Automated content review
- [ ] Report system
- [ ] Mental health detection + resources

### Phase 3: Quality of Life
- [ ] Bookmarks
- [ ] Search
- [ ] Following users
- [ ] #help LLM integration

### Phase 4: Premium
- [ ] Premium channels
- [ ] Dev visibility in #ideas
- [ ] Early access features

---

## Sources

Research synthesized from:
- [Higher Logic: Gamification in Community Forums](https://www.higherlogic.com/blog/gamification-community-forums/)
- [Trophy: Badges in Gamification Examples](https://trophy.so/blog/badges-feature-gamification-examples)
- [Game Developer: Community Badge Design](https://www.gamedeveloper.com/design/the-application-of-gamification-in-community-badge-design)
- [Grand View Research: Spiritual Wellness Apps Market](https://www.grandviewresearch.com/industry-analysis/spiritual-wellness-apps-market-report)
- [PMC: Skylight Spiritual Self-Care App Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10568400/)
- [Sendbird: Mobile App UX Best Practices](https://sendbird.com/blog/mobile-app-ux-best-practices)
- [CleverTap: App Gamification Examples](https://clevertap.com/blog/app-gamification-examples/)
