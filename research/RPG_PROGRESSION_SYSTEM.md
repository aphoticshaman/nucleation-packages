# RPG Progression System
## Quests, Currency, Reputation, and Story

**Monetization Model:**
- **FREE:** 2 readings/day (3-card or smaller), basic village access, taste of story
- **SUBSCRIPTION ($3.99-$4.99/month):** Full RPG unlocked - unlimited readings, all spreads, quests, currency, full NPC relationships, complete story

**Aesthetic:** Cute-core/Cottage-core meets Dark Carnival Gothic

---

## Free vs. Subscription

### Free Tier (Demo/Trial)

**What you get:**
- ✓ 2 readings per day (resets at midnight)
- ✓ Maximum 3-card spreads
- ✓ Access to village (explore, meet NPCs)
- ✓ Basic Oracle readings (truthful but not deep)
- ✓ Taste of story (Act 1 only)
- ✓ Can earn small amount of currency (capped)
- ✓ Basic dialog with NPCs (no relationship progression)

**What you DON'T get:**
- ✗ Unlimited readings
- ✗ Large spreads (Celtic Cross, Year Wheel, etc.)
- ✗ Quests (see them, can't accept)
- ✗ Currency earning (capped at 50 coins)
- ✗ NPC reputation system (locked)
- ✗ Story progression past Act 1
- ✗ Romance options
- ✗ Special events
- ✗ Advanced wagers
- ✗ Secret areas
- ✗ Seasonal content

**Luna to free player:**
> "You come with empty hands and ask for wisdom? I will grant you a taste, wanderer. Two questions a day, no more. If you wish to delve deeper... you must prove your commitment." [Subscribe button appears]

### Subscription Tier (Full Game)

**What unlocks:**
- ✓ **UNLIMITED readings** (no daily cap)
- ✓ **All spread types** (Celtic Cross, Year Wheel, custom layouts)
- ✓ **Quest system** (story-driven tasks)
- ✓ **Currency system** (earn, spend, invest)
- ✓ **NPC reputation** (build relationships, unlock stories)
- ✓ **Full story** (all acts, multiple endings)
- ✓ **Romance options** (Luna, NPCs - high stakes)
- ✓ **Advanced wagers** (soul fragments, memories, skills)
- ✓ **Secret areas** (unlock through quests/reputation)
- ✓ **Special events** (seasonal, lunar, story-triggered)
- ✓ **Deeper interpretations** (LLM has full context, history, character knowledge)
- ✓ **Community features** (shared readings, collective events)

**Luna to subscriber:**
> "You honor the exchange. The cards see your commitment. Come, let us walk deeper into mystery together. All secrets are open to you now."

---

## Currency System

### The Coin: "Veil Shards"

**Lore:**
- Currency minted from fragments of the Veil (boundary between worlds)
- Each coin shimmers with otherworldly light
- Accepted by all beings at the Crossroads
- Value is spiritual as much as material

**How to Earn:**

**Free Players (capped at 50 Veil Shards):**
- Daily login: 5 shards
- Complete a reading: 3 shards
- Talk to NPCs: 1 shard each (first time only)

**Subscribers (unlimited earning):**
- Daily login: 10 shards
- Complete reading: 5-10 shards (based on spread size)
- Complete quest: 20-100 shards (based on difficulty)
- Help NPC: 15 shards + reputation
- Find secrets: 25 shards
- Win wager: Variable (high risk, high reward)
- Seasonal events: 50-200 shards
- Achievements: 30-75 shards

**What to Spend On:**

```
SHOP (Masked Harlequin's Wagon):
├── Consumables
│   ├── Protective Charm (20 shards) - Soften one reversed card
│   ├── Blessing Incense (15 shards) - +10% positive reading chance
│   ├── Truth Serum (30 shards) - Deeper interpretation for next reading
│   ├── Fate Reroll (50 shards) - Draw different card (ONCE per reading)
│   └── Memory Vial (10 shards) - Save a reading permanently
│
├── Offerings for Oracle
│   ├── Black Rose (25 shards) - +5 Luna reputation
│   ├── Bottle of Absinthe (40 shards) - +10 Luna rep, unlock dialog
│   ├── Antique Tarot Book (60 shards) - +15 Luna rep, new spread
│   ├── Silk Scarf (35 shards) - +8 Luna rep, cosmetic change
│   └── Soul Fragment (100 shards) - +25 Luna rep, MAJOR unlock
│
├── NPC Gifts
│   ├── Sheet Music (15 shards) - Minstrel +10 rep
│   ├── Fire Oil (20 shards) - Fire-Eater +10 rep
│   ├── Herbal Tea (12 shards) - Cottage Witch +10 rep
│   ├── Rare Spice (30 shards) - Merchant +10 rep
│   └── Story Scroll (25 shards) - Any NPC +10 rep
│
├── Cosmetics
│   ├── Candle Colors (10 shards) - Change reading ambience
│   ├── Incense Scents (15 shards) - Lavender, Rose, Myrrh, etc.
│   ├── Card Backs (50 shards) - Unlock alternate designs
│   ├── Player Outfit (75 shards) - Change your appearance
│   └── Oracle Outfit (100 shards) - Unlock Luna's alternate looks
│
└── Unlocks
    ├── New Spread (80 shards) - Unlock advanced layout
    ├── Secret Area Key (120 shards) - Access hidden locations
    ├── NPC Backstory (50 shards) - Learn their full history
    ├── True Name (150 shards) - Learn an entity's power
    └── Companion Slot (200 shards) - Bring NPC on quests
```

### Investment System (Subscribers Only)

**"The Merchant's Gambit"**

Invest Veil Shards with the Caravan Merchant:
- Minimum investment: 50 shards
- Return time: 3 real days
- Risk/reward:
  - Safe (80% return in 3 days) - Get 90 shards back
  - Moderate (50% return, 50% loss) - Get 100 shards or 25 shards
  - Risky (25% 3x return, 75% total loss) - Get 150 shards or 0

Influenced by:
- Merchant reputation (higher = better odds)
- Recent readings (Fortune Wheel drawn = boost)
- Moon phase (full moon = lucky)
- Player karma (honor debts = blessed)

---

## Reputation System (Subscribers Only)

### How Reputation Works

**Track relationship with each NPC:**
```javascript
npc.reputation = {
  level: 0-100,
  status: "Stranger" / "Acquaintance" / "Friend" / "Confidant" / "Beloved",
  unlocked: {
    dialog: [],      // Unlocked conversation topics
    quests: [],      // Available quests from this NPC
    gifts: [],       // Special items they can give
    abilities: [],   // Buffs they provide
    romance: false,  // Romance path unlocked (if applicable)
  }
};
```

**Reputation Tiers:**

```
0-20: Stranger
- Basic dialog only
- No quests available
- Won't share personal info
- "I don't know you well enough for that."

21-40: Acquaintance
- Friendly dialog
- 1-2 simple quests available
- Small gifts accepted
- "You're alright. I've seen you around."

41-60: Friend
- Personal dialog unlocked
- 3-5 quests available (including story quests)
- Gives helpful items occasionally
- Teaches skills or secrets
- "I trust you. Let me tell you something..."

61-80: Confidant
- Deep personal stories revealed
- Major quests unlocked
- Powerful gifts given
- Provides significant buffs
- Defends you to other NPCs
- "You're like family to me now."

81-100: Beloved
- All secrets revealed
- Ultimate quest (character arc climax)
- Unique legendary item gifted
- Can call on them in crisis
- Romance available (if applicable)
- Ending influenced by this relationship
- "I would cross the Veil for you."
```

**How to Gain Reputation:**

- Talk to them regularly (+1 per day)
- Complete their quests (+10-30)
- Give them gifts they like (+5-15)
- Make choices they approve (+5-20)
- Help other NPCs they care about (+5)
- Heed Oracle's advice about them (+10)
- Draw cards that resonate with them (+3)

**How to Lose Reputation:**

- Ignore them for many days (-1 per 3 days)
- Refuse their quests (-15)
- Give insulting gifts (-10)
- Make choices they hate (-10-30)
- Harm NPCs they care about (-20)
- Break wagers/promises (-50, SEVERE)
- Betray their trust (-100, relationship destroyed)

### Key NPCs and Reputation Paths

**Luna (The Oracle)**

Reputation unlocks:
- 20: Learns your name, uses it
- 40: Offers deeper readings (more personal)
- 60: Shares piece of her history
- 80: Teaches you to read cards yourself
- 100: Romance option OR becomes mentor (player chooses)

**Kael (The Minstrel) - Cute-core NPC**

Appearance: Young wanderer with lute, messy dark hair, hopeful eyes, worn but colorful clothes

Personality: Optimistic, gentle, storyteller, believes in love and beauty despite world's darkness

Reputation unlocks:
- 20: Plays songs for you
- 40: Writes song about your journey
- 60: Teaches you music (buff: readings have musical accompaniment)
- 80: Reveals he's searching for lost love (quest chain)
- 100: Romance option (sweet, wholesome) OR helps reunite with lost love

**Rosalind (The Cottage Witch) - Cottage-core NPC**

Appearance: Plump older woman, flour on apron, warm smile, herbs in hair, cozy vibes

Personality: Motherly, nurturing, bakes constantly, offers tea and wisdom, safe haven

Reputation unlocks:
- 20: Offers you tea and cookies (healing item)
- 40: Teaches herbal magic (craft healing items)
- 60: Shares recipe book (cooking mini-game unlocked)
- 80: Reveals tragic past (lost daughter, you remind her)
- 100: Adopts you as surrogate child (ending option: stay in village)

**Vesper (The Fire-Eater) - Dark Carnival NPC**

Appearance: Scarred, intense, dark red costume, flames tattooed on arms, reckless energy

Personality: Thrill-seeker, addicted to danger, sold fear to Oracle, can't stop now

Reputation unlocks:
- 20: Teaches you to face fear (buff: better reversed card outcomes)
- 40: Tells story of wager (warning)
- 60: Asks you to help him feel SOMETHING again
- 80: Quest: Find a way to restore his fear (or accept he's doomed)
- 100: Either save him (restores fear, retires) OR witness death (tragic ending)

**Seraphine (The Contortionist) - Dark Carnival NPC**

Appearance: Impossibly flexible, pale, black/white costume, moves like liquid, unsettling grace

Personality: Mysterious, speaks in riddles, seems to know more than she says, alien beauty

Reputation unlocks:
- 20: Performs for you (mesmerizing, disturbing)
- 40: Teaches flexibility (skill: dodge attacks in quests)
- 60: Reveals she can "bend" reality, not just body
- 80: Offers to teach you (dangerous magic, high cost)
- 100: Romance option (VERY strange, reality-bending) OR learn forbidden truth

**Dorian (The Ringmaster) - Dark Carnival NPC**

Appearance: Tall, commanding, top hat, red-lined coat, white gloves, theatrical voice

Personality: Showman, controls access to Oracle, tests worthiness, has dark sense of humor

Reputation unlocks:
- 20: Lets you enter Oracle without interrogation
- 40: Shares carnival history (ancient, cursed, necessary)
- 60: Offers role in carnival (become performer)
- 80: Reveals he's bound here, cannot leave
- 100: Quest: Free him (breaks carnival, bad ending?) OR keep him (maintains order)

---

## Quest System (Subscribers Only)

### Types of Quests

**Main Story Quests:**
- Required to progress narrative
- Unlock new areas, NPCs, revelations
- Cannot be refused once started
- Multiple endings based on choices

**NPC Personal Quests:**
- Unlock at specific reputation levels
- Reveal character backstory
- Affect their fate (save, doom, transform)
- Influence ending

**Repeatable Quests:**
- Daily/weekly tasks
- Earn currency and reputation
- "Gather herbs for Rosalind"
- "Perform at tavern with Kael"
- "Help Merchant with inventory"

**Hidden Quests:**
- Discovered by exploration or specific card draws
- Unlock secrets, rare items, lore
- Optional but rewarding

**Seasonal Quests:**
- Available during events (Samhain, Solstice, etc.)
- Limited time, unique rewards
- Community-wide (all players contribute)

### Example Quest Chain: "The Fool's Errand"

**Act 1: The Lost Lute (Level 0, Free Tier)**

Kael the Minstrel: "My lute... I lost it in the woods. I'm too afraid to go alone. Will you help me?"

- Quest: Find Kael's lute in the Dark Woods
- Difficulty: Easy
- Reward: 20 shards, +15 Kael reputation
- Free players can complete this (intro quest)

**Act 2: The Broken String (Subscribers Only)**

Kael: "The lute is damaged. There's a craftsman in the city, but... he demands payment I can't afford."

- Quest: Earn 100 shards OR persuade craftsman OR find rare material
- Difficulty: Medium
- Choices affect outcome
- Reward: 50 shards, +20 Kael rep, Kael can perform (village ambience changes)

**Act 3: The Forgotten Song (Subscribers Only, Rep 60+)**

Kael: "I remember fragments of a song... from before. It's important, but I can't recall it all. The Oracle might know."

- Quest: Get reading from Luna about Kael's past
- Card drawn determines what's revealed
- Kael learns he's from noble family, exiled for loving wrong person
- Reward: 80 shards, +25 Kael rep, major lore reveal

**Act 4: The Choice (Subscribers Only, Rep 80+)**

Kael: "I can return home now. Reclaim my title. But... I'd have to leave the road. Leave music. Leave... you. What should I do?"

- Quest: Multiple outcomes based on player choice:
  - A) "Go home" → Kael leaves, sad but stable ending
  - B) "Stay and sing" → Kael remains, happy but poor ending
  - C) "I love you" (if romance unlocked) → Kael stays, romance ending
  - D) "Find your lost love" → Quest to reunite him, bittersweet ending

- Reward: 150 shards, +30 Kael rep, Kael's ending determined, major story impact

---

## Story Progression (Subscribers Only)

### The Overarching Narrative

**Core Mystery: Why does the Crossroads exist? What is Luna's true nature?**

**Act 1 (Free Tier):**
- Arrive at village
- Meet NPCs
- First Oracle reading
- Sense something is wrong (time is stuck, people seem trapped)
- Cliffhanger: Luna hints "We are all bound here, you included now."

**Act 2 (Subscribers, 0-40 total reputation):**
- Learn carnival history (ancient curse, cosmic prison)
- Discover NPCs are souls who wagered everything and lost
- Oracle is guardian/warden/priestess (unclear which)
- Can choose to leave (game over, "escape" ending) OR stay and help

**Act 3 (Subscribers, 41-70 total reputation):**
- Uncover truth: The Crossroads is a sacred space between life and death
- Those who come here are at turning points (crisis, despair, transformation)
- Oracle helps them face truth before they can move on
- But some get stuck (the NPCs), trapped by their wagers

**Act 4 (Subscribers, 71-100 total reputation):**
- Player discovers THEY are also at a crossroads in real life (meta-narrative)
- The game has been reflecting their actual journey
- Final choice: Save the NPCs (free them), OR maintain the Crossroads (it's necessary), OR take Luna's place (become new Oracle)

**Endings:**

1. **Escape Ending** - Leave early, reject the mystery (unsatisfying but valid)
2. **Liberator Ending** - Free all souls, destroy Crossroads (bittersweet, world loses sacred space)
3. **Guardian Ending** - Maintain Crossroads, improve it (hopeful, protect future seekers)
4. **Oracle Ending** - Replace Luna, become new guide (sacrifice, immortality)
5. **Romance Endings** - Stay with Luna/Kael/Seraphine (love conquers, personal happiness)
6. **Wanderer Ending** - Leave but changed, take lessons into real world (integration)

---

## Cute-core/Cottage-core Elements

**Balance darkness with warmth:**

### Cozy Spaces

**Rosalind's Cottage:**
- Warm fireplace, bookshelves, herbs drying
- Tea always brewing
- Soft music (acoustic, gentle)
- Safe from curses/darkness
- Can rest here (save point)

**Kael's Campfire:**
- Gather with other villagers
- Listen to stories and songs
- Roast marshmallows (yes, really)
- Pets allowed (mystical cats, foxes)
- Laughter, community, light

**The Meadow (Secret Area):**
- Flowers, sunshine (rare in perpetual twilight)
- Bunnies, butterflies
- Peace, restoration
- Can plant garden (mini-game)
- Unlocked at 50 total reputation

### Cute NPCs

**Pip (The Familiar) - Cottage-core Creature**

Appearance: Small fox with star-marked fur, violet eyes, too-intelligent gaze

Personality: Playful, loyal, knows secrets, can't speak but understands everything

Role: Companion, guides to secrets, occasionally steals shinies

Unlocked: Free for subscribers, can pet anytime

**The Baker's Ghost - Cute-core NPC**

Appearance: Translucent old man, flour everywhere, warm smile, smells like bread

Personality: Confused about being dead, keeps baking anyway, sweet and harmless

Role: Sells healing items (cookies, cakes), shares memories, comic relief

**Moth Collectors - Cute-core NPCs (Twin Children)**

Appearance: Two kids (boy/girl), nets, jars of glowing moths, wide curious eyes

Personality: Innocent, see magic everywhere, ask profound questions accidentally

Role: Teach wonder, offer rare items (caught moths grant visions), remind of joy

---

## Implementation Priority

### Phase 1: Monetization Gates
- [ ] Build reading limit system (2/day for free)
- [ ] Create subscription paywall UI
- [ ] Implement spread size restrictions (free = max 3 cards)
- [ ] Design "upgrade to unlock" prompts

### Phase 2: Currency System
- [ ] Create Veil Shard currency
- [ ] Build earning mechanics (login, readings, quests)
- [ ] Implement shop (Harlequin's wagon)
- [ ] Design currency UI (counter, transaction animations)

### Phase 3: Reputation System
- [ ] Track rep for each NPC (0-100)
- [ ] Create dialog unlocks based on rep
- [ ] Build gift-giving system
- [ ] Implement approval/disapproval mechanics

### Phase 4: Quest System
- [ ] Design quest log UI
- [ ] Create quest templates (fetch, choice, multi-stage)
- [ ] Build quest trigger system (rep-based, story-based)
- [ ] Implement quest rewards (shards, rep, items)

### Phase 5: Story Progression
- [ ] Write main story (4 acts)
- [ ] Create branching paths
- [ ] Implement ending system (6 endings)
- [ ] Build story gates (subscriber-only Acts 2-4)

### Phase 6: Cute-core Content
- [ ] Design Rosalind's Cottage (cozy safe space)
- [ ] Create Kael's campfire gathering
- [ ] Build The Meadow (secret cozy area)
- [ ] Design cute NPCs (Pip, Baker's Ghost, Moth Collectors)
- [ ] Implement petting/interaction mechanics

---

## Success Metrics

**Free-to-Subscriber Conversion:**
- Target: 15-25% of free users convert within 7 days
- Hook: Story cliffhanger + quest denial

**Subscriber Retention:**
- Target: 70-80% renew monthly
- Retention drivers: Ongoing quests, NPC relationships, story investment

**Engagement:**
- Target: Subscribers play 3-5 times/week
- Session length: 20-40 minutes
- Quest completion rate: 60%+

**Emotional Investment:**
- Target: 50%+ report caring about NPCs
- Target: 40%+ replay to see different endings

**Revenue:**
- $3.99/month tier: Majority of subscribers
- $4.99/month tier: 30-40% (premium features?)
- Average LTV (lifetime value): $30-50 (6-12 month retention)

---

## Final Note

**This is not just a tarot app with RPG elements tacked on.**

**This is an RPG where tarot is the core mechanic.**

- Readings guide quests
- Cards influence NPC relationships
- Spreads unlock story
- Wagers have narrative weight

**And the subscription unlocks:**
- Not just "more readings"
- But THE FULL EXPERIENCE

Free tier = "I'm curious"
Paid tier = "I'm committed"

**That commitment is rewarded with:**
- Deep story
- Meaningful relationships
- Real choices
- Lasting impact

**Worth $3.99-$4.99/month?**

**Absolutely.**
