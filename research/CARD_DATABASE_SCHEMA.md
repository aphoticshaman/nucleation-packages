# Card Database Schema
**VeilPath Tarot - Dark Fantasy Edition**

## Overview
This document defines the canonical data structure for all 78 tarot cards in the VeilPath Tarot app.

---

## Card Object Structure

```javascript
{
  // IDENTITY
  id: Number,              // 0-77 (The Fool = 0, Ace of Wands = 22, etc.)
  name: String,            // "The Fool", "Ace of Wands", "King of Cups"
  arcana: String,          // "Major" | "Minor"
  suit: String | null,     // "Wands" | "Cups" | "Swords" | "Pentacles" | null (Major Arcana)
  number: Number | String, // 0-21 (Major), 1-14 (Minor: 1-10, Jack/Knight/Queen/King)

  // CORRESPONDENCES
  element: String,         // "Fire" | "Water" | "Air" | "Earth" | "Spirit"
  astrology: String,       // Zodiac sign or planet (e.g., "Aries", "Mars", "Sun")
  numerology: Number,      // Numerological value (0-10)
  chakra: String | null,   // Associated chakra (e.g., "Root", "Crown")

  // MEANINGS
  keywords: {
    upright: String[],     // ["courage", "passion", "enthusiasm"]
    reversed: String[]     // ["recklessness", "burnout", "impulsiveness"]
  },

  meanings: {
    upright: String,       // Full upright interpretation (2-3 sentences)
    reversed: String       // Full reversed interpretation (2-3 sentences)
  },

  // SYMBOLISM
  symbols: String[],       // Visual symbols on card ["wand", "salamander", "flame"]
  colors: String[],        // Dominant colors ["red", "orange", "gold"]

  // THEMES
  themes: String[],        // Archetypal themes ["beginnings", "action", "creativity"]

  // CONTEXTUAL MEANINGS
  contexts: {
    love: String,          // Meaning in love readings
    career: String,        // Meaning in career readings
    spirituality: String   // Meaning in spiritual readings
  },

  // DARK FANTASY FLAVOR
  darkFantasy: {
    title: String,         // "The Wanderer at the Veil's Edge"
    description: String,   // Visual description in dark fantasy setting
    questTie: String,      // Connection to in-game quests/story
    npcReference: String   // Which NPC (Luna/Sol/etc.) relates to this card
  },

  // ADVICE
  advice: String,          // Actionable guidance (1 sentence)
  affirmation: String,     // Positive affirmation (1 sentence)

  // POLARITY
  polarity: String,        // "Positive" | "Negative" | "Neutral"

  // CARD BACK TIER (for unlocking progression)
  tier: String             // "common" | "uncommon" | "rare" | "epic" | "legendary"
}
```

---

## Major Arcana (0-21)

### ID Mapping
```javascript
0: "The Fool"
1: "The Magician"
2: "The High Priestess"
3: "The Empress"
4: "The Emperor"
5: "The Hierophant"
6: "The Lovers"
7: "The Chariot"
8: "Strength"
9: "The Hermit"
10: "Wheel of Fortune"
11: "Justice"
12: "The Hanged Man"
13: "Death"
14: "Temperance"
15: "The Devil"
16: "The Tower"
17: "The Star"
18: "The Moon"
19: "The Sun"
20: "Judgement"
21: "The World"
```

---

## Minor Arcana (22-77)

### Wands (22-35) - Fire Element
```javascript
22: "Ace of Wands"
23: "Two of Wands"
24: "Three of Wands"
25: "Four of Wands"
26: "Five of Wands"
27: "Six of Wands"
28: "Seven of Wands"
29: "Eight of Wands"
30: "Nine of Wands"
31: "Ten of Wands"
32: "Page of Wands"
33: "Knight of Wands"
34: "Queen of Wands"
35: "King of Wands"
```

### Cups (36-49) - Water Element
```javascript
36: "Ace of Cups"
37: "Two of Cups"
38: "Three of Cups"
39: "Four of Cups"
40: "Five of Cups"
41: "Six of Cups"
42: "Seven of Cups"
43: "Eight of Cups"
44: "Nine of Cups"
45: "Ten of Cups"
46: "Page of Cups"
47: "Knight of Cups"
48: "Queen of Cups"
49: "King of Cups"
```

### Swords (50-63) - Air Element
```javascript
50: "Ace of Swords"
51: "Two of Swords"
52: "Three of Swords"
53: "Four of Swords"
54: "Five of Swords"
55: "Six of Swords"
56: "Seven of Swords"
57: "Eight of Swords"
58: "Nine of Swords"
59: "Ten of Swords"
60: "Page of Swords"
61: "Knight of Swords"
62: "Queen of Swords"
63: "King of Swords"
```

### Pentacles (64-77) - Earth Element
```javascript
64: "Ace of Pentacles"
65: "Two of Pentacles"
66: "Three of Pentacles"
67: "Four of Pentacles"
68: "Five of Pentacles"
69: "Six of Pentacles"
70: "Seven of Pentacles"
71: "Eight of Pentacles"
72: "Nine of Pentacles"
73: "Ten of Pentacles"
74: "Page of Pentacles"
75: "Knight of Pentacles"
76: "Queen of Pentacles"
77: "King of Pentacles"
```

---

## Tier Assignment Rules

### Major Arcana Tiers
- **Legendary (5 cards):** The Fool, Death, The Tower, The Moon, The World
- **Epic (5 cards):** The Magician, The High Priestess, The Devil, The Star, The Sun
- **Rare (11 cards):** All others

### Minor Arcana Tiers
- **Epic (4 cards):** Aces (transformative power)
- **Rare (12 cards):** Court cards (Page/Knight/Queen/King × 4 suits)
- **Uncommon (20 cards):** 5-10 of each suit
- **Common (20 cards):** 2-4 of each suit

---

## Validation Rules

### Required Fields (All Cards)
- id, name, arcana, element, keywords, meanings

### Major Arcana Specific
- suit = null
- number = 0-21
- astrology must be present

### Minor Arcana Specific
- suit must be one of: Wands, Cups, Swords, Pentacles
- number = 1-14 (or Ace, Page, Knight, Queen, King)
- element matches suit (Wands=Fire, Cups=Water, Swords=Air, Pentacles=Earth)

---

## Dark Fantasy Theme Examples

### The Fool
```javascript
darkFantasy: {
  title: "The Wanderer at the Veil's Edge",
  description: "A hooded figure stands at the precipice of a moonlit chasm, one foot suspended over an endless void. Behind them, a white wolf with glowing eyes watches, protector or omen unclear. The Veil shimmers in the distance—a curtain between worlds.",
  questTie: "The First Step - Luna's initial quest to cross the Threshold",
  npcReference: "Luna guides new Fools through their first journey into shadow"
}
```

### Ace of Wands
```javascript
darkFantasy: {
  title: "The Ember That Birthed Flame",
  description: "A gnarled staff erupts from volcanic stone, its tip crowned with a single flame that casts dancing shadows across ancient runes. The air shimmers with potential—raw, untamed, hungry.",
  questTie: "Ignition - Quest to awaken dormant power within the Firelands",
  npcReference: "Sol's blessing for those who choose action over contemplation"
}
```

---

## Usage Examples

### Get Card by ID
```javascript
import { CARD_DATABASE } from './data/cardDatabase';

const card = CARD_DATABASE[0]; // The Fool
console.log(card.name); // "The Fool"
console.log(card.keywords.upright); // ["new beginnings", "innocence", ...]
```

### Filter by Suit
```javascript
const wands = Object.values(CARD_DATABASE).filter(card => card.suit === 'Wands');
```

### Get Dark Fantasy Description
```javascript
const fool = CARD_DATABASE[0];
console.log(fool.darkFantasy.description);
```

---

## File Location
`src/data/cardDatabase.js`

**Format:** ES6 Module Export
```javascript
export const CARD_DATABASE = {
  0: { /* The Fool */ },
  1: { /* The Magician */ },
  // ...
};
```

---

**Last Updated:** 2025-11-20
