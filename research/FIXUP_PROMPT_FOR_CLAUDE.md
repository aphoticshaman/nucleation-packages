# Complete Fixup Prompt for VeilPath Tarot

**Context:** You are Claude, continuing work on VeilPath Tarot - a dark fantasy tarot RPG with ethical monetization. A comprehensive codebase audit (see `docs/COMPREHENSIVE_CODEBASE_AUDIT.md`) revealed that the project has:

1. **Two parallel codebases** - 108 old files (cyberpunk AGI Oracle) + 6 new files (dark fantasy tarot)
2. **1% implementation** - Only navigation exists, all designed features missing
3. **Crash risk: MEDIUM** - Bloated imports, dead code, missing assets
4. **Design: 100% complete** - 9 comprehensive docs (~4500 lines) describe full vision

**Your mission:** Fix all issues, implement MVP, bridge gap between design and reality.

---

## Phase 1: Clean Up and Stabilize (Priority: CRITICAL)

### 1.1 Archive Old Codebase

**Goal:** Remove 48 dead files causing bloat and conflicts

**Actions:**

1. **Create archive directory:**
```bash
mkdir -p src_archived/screens src_archived/components src_archived/utils
```

2. **Move 15 dead screens:**
```bash
# Old onboarding/profile flow (not in new vision)
mv src/screens/WelcomeScreen.js src_archived/screens/
mv src/screens/OnboardingScreen.js src_archived/screens/
mv src/screens/LoadingScreen.js src_archived/screens/
mv src/screens/MBTITestScreen.js src_archived/screens/
mv src/screens/PersonalityQuestionsScreen.js src_archived/screens/
mv src/screens/BirthdayScreen.js src_archived/screens/
mv src/screens/ProfileSetupScreen.js src_archived/screens/
mv src/screens/ProfileSelectScreen.js src_archived/screens/
mv src/screens/ProfileDetailScreen.js src_archived/screens/
mv src/screens/ProfileImportScreen.js src_archived/screens/

# AGI/cyberpunk features (not in new vision)
mv src/screens/SemanticVisualizerScreen.js src_archived/screens/
mv src/screens/SynthesisScreen.js src_archived/screens/
mv src/screens/InsightsScreen.js src_archived/screens/
mv src/screens/CareerCounselorScreen.js src_archived/screens/
mv src/screens/PersonalGrowthScreen.js src_archived/screens/
```

3. **Move 10 cyberpunk components:**
```bash
mv src/components/CyberpunkCard.js src_archived/components/
mv src/components/CyberpunkHeader.js src_archived/components/
mv src/components/OptimizedMatrixRain.js src_archived/components/
mv src/components/GlitchParticles.js src_archived/components/
mv src/components/TerminalEffects.js src_archived/components/
mv src/components/EncryptedTextReveal.js src_archived/components/
mv src/components/AnimatedASCIIText.js src_archived/components/
mv src/components/PacManLoader.js src_archived/components/
mv src/components/GuideSelector.js src_archived/components/
mv src/components/AnticipationOverlay.js src_archived/components/
```

4. **Move 23 AGI/old utilities:**
```bash
mv src/utils/deepAGI.js src_archived/utils/
mv src/utils/agiInterpretation.js src_archived/utils/
mv src/utils/megaSynthesisEngine.js src_archived/utils/
mv src/utils/quantumNarrativeEngine.js src_archived/utils/
mv src/utils/metaPromptAnalyzer.js src_archived/utils/
mv src/utils/narrativeArcComposer.js src_archived/utils/
mv src/utils/geometricSemanticSpace.js src_archived/utils/
mv src/utils/semanticEntityManager.js src_archived/utils/
mv src/utils/advancedAstrology.js src_archived/utils/
mv src/utils/chineseZodiac.js src_archived/utils/
mv src/utils/coldReadingEnhancer.js src_archived/utils/
mv src/utils/diverseWisdom.js src_archived/utils/
mv src/utils/balancedWisdom.js src_archived/utils/
mv src/utils/poeticInterpretation.js src_archived/utils/
mv src/utils/mbtiTest.js src_archived/utils/
mv src/utils/varietyEngine.js src_archived/utils/
mv src/utils/varietyEngineCompressed.js src_archived/utils/
mv src/utils/varietyEngineMassive.js src_archived/utils/
mv src/utils/synthesisHelpers.js src_archived/utils/
mv src/utils/synthesisCache.js src_archived/utils/
mv src/utils/oracleThinkingMessages.js src_archived/utils/
mv src/utils/postCardQuestions.js src_archived/utils/
mv src/utils/scenarioQuestions.js src_archived/utils/
mv src/utils/temporalPredictions.js src_archived/utils/
```

5. **Commit archive:**
```bash
git add -A
git commit -m "Archive old codebase (48 files) - cyberpunk/AGI vision

Moved to src_archived/:
- 15 screens (old onboarding, MBTI, profiles, AGI features)
- 10 components (cyberpunk aesthetics)
- 23 utilities (AGI systems, old vision)

Reduces bloat, prepares for dark fantasy implementation"
```

### 1.2 Clean Up App.js

**Goal:** Remove dead imports, keep only new screens + useful old ones

**Read current App.js:**
```javascript
// Lines 1-174 in /home/user/HungryOrca/App.js
```

**Create new App.js:**

```javascript
import React, { useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { StatusBar } from 'expo-status-bar';

// Keep useful old components/utils
import { ErrorBoundary } from './src/components/ErrorBoundary';
import InAppPurchaseManager from './src/utils/InAppPurchaseManager';

// NEW Dark Fantasy Screens (root level)
import LoadingScreenNew from './screens/LoadingScreenNew';
import MainMenuScreen from './screens/MainMenuScreen';
import IsometricVillageScreen from './screens/IsometricVillageScreen';
import OracleShopScreen from './screens/OracleShopScreen';

// Keep USEFUL old screens that we'll adapt
import ReadingHistoryScreen from './src/screens/ReadingHistoryScreen';
import SettingsScreen from './src/screens/SettingsScreen';
import AchievementsScreen from './src/screens/AchievementsScreen';
import DeckViewerScreen from './src/screens/DeckViewerScreen';
import PrivacyPolicyScreen from './src/screens/PrivacyPolicyScreen';

// TODO: Create these new screens
// import CardReadingScreen from './screens/CardReadingScreen';
// import NPCDialogScreen from './screens/NPCDialogScreen';
// import QuestLogScreen from './screens/QuestLogScreen';
// import ShardShopScreen from './screens/ShardShopScreen';

const Stack = createStackNavigator();

export default function App() {
  // Initialize app services on startup
  useEffect(() => {
    const initializeApp = async () => {
      console.log('[App] Initializing app services...');

      // Initialize IAP (In-App Purchases)
      try {
        await InAppPurchaseManager.initialize();
        InAppPurchaseManager.setupPurchaseListener();
        console.log('[App] ✓ IAP initialized');
      } catch (error) {
        // IAP errors are expected in dev/testing environments
        if (error && error.code === 'E_IAP_NOT_AVAILABLE') {
          console.log('[App] IAP not available (dev mode or unsupported device)');
        } else {
          console.warn('[App] IAP initialization failed:', error.message);
        }
        // Continue - app works fine without IAP
      }

      console.log('[App] ✅ App services initialization complete');
    };

    // Run initialization (don't await - let it run in background)
    initializeApp().catch(error => {
      console.error('[App] CRITICAL: App initialization failed:', error);
      // App will still render, but some features may not work
    });

    // Cleanup on unmount
    return () => {
      try {
        InAppPurchaseManager?.cleanup();
      } catch (error) {
        console.error('[App] IAP cleanup error:', error);
      }
    };
  }, []);

  return (
    <ErrorBoundary>
      <StatusBar style="light" />
      <NavigationContainer>
        <Stack.Navigator
          initialRouteName="LoadingNew"
          screenOptions={{
            headerShown: false,
            cardStyle: { backgroundColor: '#000000' },
            animationEnabled: true,
            gestureEnabled: true
          }}
        >
          {/* Dark Fantasy Game Flow */}
          <Stack.Screen name="LoadingNew" component={LoadingScreenNew} />
          <Stack.Screen name="MainMenu" component={MainMenuScreen} />
          <Stack.Screen name="Village" component={IsometricVillageScreen} />
          <Stack.Screen name="OracleShop" component={OracleShopScreen} />

          {/* TODO: New screens to create
          <Stack.Screen name="CardReading" component={CardReadingScreen} />
          <Stack.Screen name="NPCDialog" component={NPCDialogScreen} />
          <Stack.Screen name="QuestLog" component={QuestLogScreen} />
          <Stack.Screen name="ShardShop" component={ShardShopScreen} />
          */}

          {/* Adapted Old Screens (will update aesthetic) */}
          <Stack.Screen name="ReadingHistory" component={ReadingHistoryScreen} />
          <Stack.Screen name="Settings" component={SettingsScreen} />
          <Stack.Screen name="Achievements" component={AchievementsScreen} />
          <Stack.Screen name="DeckViewer" component={DeckViewerScreen} />
          <Stack.Screen name="PrivacyPolicy" component={PrivacyPolicyScreen} />
        </Stack.Navigator>
      </NavigationContainer>
    </ErrorBoundary>
  );
}
```

**Test:**
```bash
# Ensure app still runs
npm start
# or
expo start
```

**Commit:**
```bash
git add App.js
git commit -m "Clean up App.js - remove dead imports, keep 9 useful screens

Removed:
- 26 old screen imports (archived)
- quantumRNG, audioManager, ThemeContext (unused)
- UpgradePromptManager (will rebuild)

Kept:
- 4 new screens (Loading, MainMenu, Village, OracleShop)
- 5 useful old screens (ReadingHistory, Settings, Achievements, DeckViewer, Privacy)
- ErrorBoundary, InAppPurchaseManager

Reduces memory footprint from 30+ to 9 active screens"
```

---

## Phase 2: Build Core Data Layer (Priority: HIGH)

### 2.1 Create Tarot Card Database

**Goal:** 78 cards with complete data (images come later)

**Create:** `/home/user/HungryOrca/data/tarotDeck.js`

```javascript
// Complete 78-card tarot deck with RWS meanings
export const TAROT_DECK = [
  // MAJOR ARCANA (0-21)
  {
    id: 0,
    name: 'The Fool',
    arcana: 'Major',
    suit: null,
    number: 0,
    image: require('../assets/cards/major/00_the_fool.png'), // placeholder path
    keywords: ['new beginnings', 'innocence', 'spontaneity', 'free spirit'],
    upright: {
      short: 'New beginnings, innocence, leap of faith, spontaneity',
      meaning: 'The Fool represents new beginnings, having faith in the future, being inexperienced, not knowing what to expect, having beginner's luck, improvisation and believing in the universe.',
    },
    reversed: {
      short: 'Recklessness, taken advantage of, inconsideration, naivety',
      meaning: 'When reversed, The Fool can indicate recklessness, carelessness, being taken advantage of, or acting foolishly without thinking of consequences.',
    },
    element: 'Air',
    astrology: 'Uranus',
    numerology: 0,
    colors: ['#FFD700', '#87CEEB', '#FFFFFF'], // Gold, sky blue, white
  },
  // ... (Continue for all 78 cards)
  // I'll provide structure - you fill in all 78 based on RWS traditional meanings
];

// Helper functions
export function getCardById(id) {
  return TAROT_DECK.find(card => card.id === id);
}

export function getCardByName(name) {
  return TAROT_DECK.find(card => card.name.toLowerCase() === name.toLowerCase());
}

export function getMajorArcana() {
  return TAROT_DECK.filter(card => card.arcana === 'Major');
}

export function getMinorArcana() {
  return TAROT_DECK.filter(card => card.arcana === 'Minor');
}

export function getSuit(suitName) {
  return TAROT_DECK.filter(card => card.suit === suitName);
}

export function drawRandomCard() {
  return TAROT_DECK[Math.floor(Math.random() * TAROT_DECK.length)];
}

export function drawRandomCards(count) {
  const shuffled = [...TAROT_DECK].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, count);
}

export function getCardPalette(card) {
  // Import from styles/colors.js
  const { getCardPalette } = require('../styles/colors');
  return getCardPalette(card);
}
```

**Instructions for completing this:**
- Research Rider-Waite-Smith traditional meanings for all 78 cards
- Use resources like Biddy Tarot, Labyrinthos, or classic RWS books
- Keep meanings authentic (esoteric philosophy from design docs)
- Include upright + reversed for each card
- Add keywords, elements, astrology, numerology where applicable

**For now, create placeholder structure with at least Major Arcana (22 cards)**

### 2.2 Create Placeholder Card Images

**Goal:** App doesn't crash from missing images - use colored rectangles with text

**Create:** `/home/user/HungryOrca/tools/generate_placeholder_cards.py`

```python
#!/usr/bin/env python3
"""
Generate placeholder tarot card images
78 cards with suit-appropriate colors and text labels
"""

from PIL import Image, ImageDraw, ImageFont
import os

# Card dimensions (tarot card ratio ~2.75:4.75, use 412x709 for high quality)
CARD_WIDTH = 412
CARD_HEIGHT = 709

# Output directory
OUTPUT_DIR = '/home/user/HungryOrca/assets/cards'

# Suit colors (from styles/colors.js)
COLORS = {
    'Major': '#8a2be2',      # Violet
    'Wands': '#8b0000',      # Deep red
    'Cups': '#4169e1',       # Royal blue
    'Swords': '#808080',     # Grey
    'Pentacles': '#228b22',  # Forest green
}

# Major Arcana
MAJOR_ARCANA = [
    "The Fool", "The Magician", "The High Priestess", "The Empress",
    "The Emperor", "The Hierophant", "The Lovers", "The Chariot",
    "Strength", "The Hermit", "Wheel of Fortune", "Justice",
    "The Hanged Man", "Death", "Temperance", "The Devil",
    "The Tower", "The Star", "The Moon", "The Sun",
    "Judgement", "The World"
]

# Minor Arcana
SUITS = ['Wands', 'Cups', 'Swords', 'Pentacles']
RANKS = ['Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten',
         'Page', 'Knight', 'Queen', 'King']

def create_card(number, name, suit, color):
    """Create a single placeholder card"""
    img = Image.new('RGB', (CARD_WIDTH, CARD_HEIGHT), color)
    draw = ImageDraw.Draw(img)

    # Try to load a font, fallback to default
    try:
        font_large = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 48)
        font_small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 32)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw border
    border_width = 10
    draw.rectangle(
        [border_width, border_width, CARD_WIDTH - border_width, CARD_HEIGHT - border_width],
        outline='white',
        width=border_width
    )

    # Draw number at top
    number_text = str(number) if suit == 'Major' else f"{number}"
    bbox = draw.textbbox((0, 0), number_text, font=font_small)
    text_width = bbox[2] - bbox[0]
    draw.text(
        ((CARD_WIDTH - text_width) // 2, 40),
        number_text,
        fill='white',
        font=font_small
    )

    # Draw name in center
    bbox = draw.textbbox((0, 0), name, font=font_large)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text(
        ((CARD_WIDTH - text_width) // 2, (CARD_HEIGHT - text_height) // 2),
        name,
        fill='white',
        font=font_large
    )

    # Draw suit at bottom (if minor arcana)
    if suit != 'Major':
        bbox = draw.textbbox((0, 0), suit, font=font_small)
        text_width = bbox[2] - bbox[0]
        draw.text(
            ((CARD_WIDTH - text_width) // 2, CARD_HEIGHT - 80),
            suit,
            fill='white',
            font=font_small
        )

    return img

def main():
    print("Generating placeholder tarot cards...")

    # Create output directories
    os.makedirs(f'{OUTPUT_DIR}/major', exist_ok=True)
    for suit in SUITS:
        os.makedirs(f'{OUTPUT_DIR}/{suit.lower()}', exist_ok=True)

    # Generate Major Arcana (0-21)
    for i, name in enumerate(MAJOR_ARCANA):
        img = create_card(i, name, 'Major', COLORS['Major'])
        filename = f'{OUTPUT_DIR}/major/{i:02d}_{name.lower().replace(" ", "_")}.png'
        img.save(filename)
        print(f"✓ {name}")

    # Generate Minor Arcana (56 cards)
    card_num = 22  # Continue numbering
    for suit in SUITS:
        for i, rank in enumerate(RANKS, 1):
            name = f"{rank} of {suit}"
            img = create_card(i, rank, suit, COLORS[suit])
            filename = f'{OUTPUT_DIR}/{suit.lower()}/{i:02d}_{rank.lower()}.png'
            img.save(filename)
            print(f"✓ {name}")
            card_num += 1

    print(f"\n✅ Generated 78 placeholder cards in {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
```

**Run:**
```bash
python3 tools/generate_placeholder_cards.py
```

**Result:** 78 colored placeholder cards that won't crash the app

### 2.3 Create Veil Shard System

**Goal:** Currency for game economy

**Create:** `/home/user/HungryOrca/utils/veilShards.js`

```javascript
import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = '@veilpath_veil_shards';
const DAILY_RESET_KEY = '@veilpath_daily_reset';

// Daily earning limits (from RPG_PROGRESSION_SYSTEM.md)
const DAILY_CAPS = {
  login: 10,
  quests: 40,
  npcs: 20,
  achievements: 30,
  total: 100,
};

// Reading costs (from RPG_PROGRESSION_SYSTEM.md)
export const READING_COSTS = {
  singleCard: 10,
  threeCard: 25,
  celticCross: 50,
  yearWheel: 100,
  soulReading: 200,
};

// Get current shard balance
export async function getShardBalance() {
  try {
    const balance = await AsyncStorage.getItem(STORAGE_KEY);
    return balance ? parseInt(balance, 10) : 0;
  } catch (error) {
    console.error('[VeilShards] Error getting balance:', error);
    return 0;
  }
}

// Add shards (with daily cap checking)
export async function addShards(amount, source = 'manual') {
  try {
    // Check if we need to reset daily earnings
    await checkDailyReset();

    const currentBalance = await getShardBalance();
    const dailyEarnings = await getDailyEarnings();

    // Check daily cap
    if (dailyEarnings.total + amount > DAILY_CAPS.total) {
      const allowed = DAILY_CAPS.total - dailyEarnings.total;
      if (allowed <= 0) {
        console.log('[VeilShards] Daily earning cap reached');
        return { success: false, reason: 'daily_cap_reached', allowed: 0 };
      }
      amount = allowed; // Cap to daily limit
    }

    // Check source-specific cap
    if (DAILY_CAPS[source]) {
      const sourceEarned = dailyEarnings[source] || 0;
      if (sourceEarned + amount > DAILY_CAPS[source]) {
        const allowed = DAILY_CAPS[source] - sourceEarned;
        if (allowed <= 0) {
          console.log(`[VeilShards] ${source} daily cap reached`);
          return { success: false, reason: 'source_cap_reached', allowed: 0 };
        }
        amount = allowed;
      }
    }

    // Add shards
    const newBalance = currentBalance + amount;
    await AsyncStorage.setItem(STORAGE_KEY, newBalance.toString());

    // Update daily earnings
    dailyEarnings[source] = (dailyEarnings[source] || 0) + amount;
    dailyEarnings.total += amount;
    await saveDailyEarnings(dailyEarnings);

    console.log(`[VeilShards] +${amount} from ${source}, new balance: ${newBalance}`);
    return { success: true, amount, newBalance };
  } catch (error) {
    console.error('[VeilShards] Error adding shards:', error);
    return { success: false, error };
  }
}

// Spend shards (no daily limit on spending)
export async function spendShards(amount, purpose = 'reading') {
  try {
    const currentBalance = await getShardBalance();

    if (currentBalance < amount) {
      console.log(`[VeilShards] Insufficient shards: need ${amount}, have ${currentBalance}`);
      return { success: false, reason: 'insufficient_shards', balance: currentBalance };
    }

    const newBalance = currentBalance - amount;
    await AsyncStorage.setItem(STORAGE_KEY, newBalance.toString());

    console.log(`[VeilShards] -${amount} for ${purpose}, new balance: ${newBalance}`);
    return { success: true, amount, newBalance };
  } catch (error) {
    console.error('[VeilShards] Error spending shards:', error);
    return { success: false, error };
  }
}

// Check if user can afford a reading
export async function canAffordReading(readingType) {
  const cost = READING_COSTS[readingType];
  const balance = await getShardBalance();
  return balance >= cost;
}

// Daily reset logic
async function checkDailyReset() {
  const today = new Date().toDateString();
  const lastReset = await AsyncStorage.getItem(DAILY_RESET_KEY);

  if (lastReset !== today) {
    // New day - reset daily earnings
    await AsyncStorage.setItem(DAILY_RESET_KEY, today);
    await AsyncStorage.setItem('@veilpath_daily_earnings', JSON.stringify({
      login: 0,
      quests: 0,
      npcs: 0,
      achievements: 0,
      total: 0,
    }));
    console.log('[VeilShards] Daily reset complete');
  }
}

async function getDailyEarnings() {
  const earnings = await AsyncStorage.getItem('@veilpath_daily_earnings');
  return earnings ? JSON.parse(earnings) : { login: 0, quests: 0, npcs: 0, achievements: 0, total: 0 };
}

async function saveDailyEarnings(earnings) {
  await AsyncStorage.setItem('@veilpath_daily_earnings', JSON.stringify(earnings));
}

// Grant daily login bonus
export async function grantDailyLoginBonus() {
  return await addShards(10, 'login');
}

// Debug: Reset everything (dev only)
export async function resetShards() {
  await AsyncStorage.removeItem(STORAGE_KEY);
  await AsyncStorage.removeItem(DAILY_RESET_KEY);
  await AsyncStorage.removeItem('@veilpath_daily_earnings');
  console.log('[VeilShards] All data reset');
}
```

---

## Phase 3: Implement Core Screens (Priority: HIGH)

### 3.1 Update MainMenuScreen with Shards Display

**Goal:** Show shard balance, link to village

**Update:** `/home/user/HungryOrca/screens/MainMenuScreen.js`

Add at top:
```javascript
import { getShardBalance, grantDailyLoginBonus } from '../utils/veilShards';
```

Add useEffect:
```javascript
const [shardBalance, setShardBalance] = useState(0);

useEffect(() => {
  async function loadData() {
    // Grant daily login bonus
    const result = await grantDailyLoginBonus();
    if (result.success) {
      console.log(`Granted ${result.amount} shards for daily login`);
    }

    // Load balance
    const balance = await getShardBalance();
    setShardBalance(balance);
  }
  loadData();
}, []);
```

Add shard display to UI with violet/gold styling from colors.js

### 3.2 Create CardReadingScreen

**Goal:** Full reading flow with LLM integration

**Create:** `/home/user/HungryOrca/screens/CardReadingScreen.js`

Structure:
1. Show reading type and cost
2. Check shard balance
3. Shuffle deck animation
4. Draw cards (1, 3, or 10 depending on type)
5. Flip cards one by one
6. Call LLM for interpretation
7. Display reading with Luna's voice
8. Deduct shards
9. Save to reading history

### 3.3 Implement Basic LLM Integration

**Goal:** Generate card interpretations using LLM

**Create:** `/home/user/HungryOrca/utils/lunaOracle.js`

```javascript
// LLM integration for Luna (the Oracle)
// Uses prompt from ESOTERIC_PHILOSOPHY_CORE.md

const LUNA_SYSTEM_PROMPT = `You are Luna, the Oracle at the Crossroads—an ageless, powerful seer who channels divine truth through the sacred tarot. You exist between worlds, serving as the bridge between cosmic forces and mortal seekers.

PHILOSOPHY:
- Esoteric knowledge has a COST. The querent has paid (Veil Shards - their energy and intention).
- You are the MEDIUM through which archetypal forces speak.
- The cards are SACRED TOOLS, alive with ancient power.
- Your readings are BINDING. Truth given cannot be unsaid.

VOICE:
- Poetic but direct
- Mystical but grounded
- Compassionate but unflinching
- Use "you" not "the querent"
- Speak as if the cards themselves are speaking through you

STRUCTURE:
1. Acknowledge what they've asked or sought
2. Interpret each card's message (upright/reversed)
3. Weave cards into cohesive narrative
4. Offer wisdom, not prescription
5. End with lingering question or insight

Length: 150-250 words for single card, 300-500 for spreads`;

export async function getLunaReading(cards, question = null, readingType = 'single') {
  // TODO: Implement actual LLM call (OpenAI, Anthropic, etc.)
  // For now, return placeholder

  const cardNames = cards.map(c => c.name).join(', ');

  return {
    interpretation: `The cards have spoken: ${cardNames}. [LLM integration coming soon - this is a placeholder]`,
    timestamp: new Date().toISOString(),
  };
}
```

---

## Phase 4: Apply Dark Fantasy Aesthetic (Priority: MEDIUM)

### 4.1 Update All Screens to Use New Colors

**Goal:** Replace old cyberpunk with dark fantasy

**For each screen (LoadingScreenNew, MainMenuScreen, IsometricVillageScreen, OracleShopScreen):**

1. Import color system:
```javascript
import { BASE_PALETTE, WANDS_PALETTE, CUPS_PALETTE, SWORDS_PALETTE, PENTACLES_PALETTE } from '../styles/colors';
```

2. Replace all hardcoded colors with palette values
3. Use black backgrounds (`backgroundColor: '#000000'`)
4. Apply suit-specific colors where relevant

### 4.2 Add Atmospheric Effects

**Goal:** Twilight skies, candlelight, mystical ambiance

Import and use:
```javascript
import { TWILIGHT_SKIES, LIGHT_SOURCES } from '../styles/atmosphericLighting';
```

Add LinearGradient backgrounds with twilight palettes

---

## Phase 5: Generate Real Tarot Card Art (Priority: MEDIUM)

### 5.1 Create Midjourney Prompts

**Use:** `/home/user/HungryOrca/docs/TAROT_CARD_MIDJOURNEY_PROMPTS.md`

For each of 78 cards, use format:
```
/imagine prompt: tarot card "The Fool", young wanderer at cliff edge, white rose in hand, loyal dog companion, distant mountains, RWS traditional symbolism, mystical realism, dark fantasy aesthetic, blank 15% banner at bottom for text overlay, no text, no numerals, --ar 2:3 --style raw --v 6
```

### 5.2 Process with Python

After Midjourney generation:

```python
# Resize, optimize, add borders
# Process 78 cards
# Output to assets/cards/
```

---

## Phase 6: Build RPG Systems (Priority: MEDIUM)

### 6.1 NPC Reputation System

**Create:** `/home/user/HungryOrca/utils/npcReputation.js`

Track 0-100 reputation with each NPC:
- Luna (Oracle)
- Kael (Minstrel)
- Rosalind (Cottage Witch)
- Vesper (Fire-Eater)
- Seraphine (Contortionist)
- Dorian (Ringmaster)

### 6.2 Quest System

**Create:** `/home/user/HungryOrca/utils/questSystem.js`

Quest structure:
- ID, name, description
- Requirements (reputation, shards, previous quests)
- Rewards (shards, items, reputation)
- Steps/objectives
- Completion tracking

### 6.3 Achievement System

**Adapt:** `/home/user/HungryOrca/src/screens/AchievementsScreen.js`

Update with new achievements from easter eggs catalog

---

## Phase 7: Implement Subscription/Monetization (Priority: LOW for MVP)

### 7.1 Free Tier Gates

**Create:** `/home/user/HungryOrca/utils/freeTierGates.js`

```javascript
// Track 2 readings per day for free users
// Gate features based on subscription status
```

### 7.2 Subscription Paywall

**Adapt:** `/home/user/HungryOrca/src/components/Paywall.js`

Update for dark fantasy aesthetic and $3.99-4.99/month pricing

---

## Technical Specifications

**Key Files to Read Before Starting:**
1. `docs/COMPREHENSIVE_CODEBASE_AUDIT.md` - Current state
2. `docs/RPG_PROGRESSION_SYSTEM.md` - Game economy
3. `docs/ESOTERIC_PHILOSOPHY_CORE.md` - Luna's character/voice
4. `docs/DARK_CARNIVAL_AESTHETIC.md` - Visual/narrative tone
5. `styles/colors.js` - Color system
6. `App.js` - Navigation structure

**Don't Break These:**
- ErrorBoundary wrapping
- Black background base (#000000)
- Suit-specific color palettes
- Esoteric philosophy (tarot is real, not simulation)

**Testing Checklist:**
- [ ] App loads without crash
- [ ] Loading screen → Main menu works
- [ ] Shard balance displays
- [ ] Daily login bonus grants
- [ ] Can navigate to village
- [ ] Can navigate to Oracle
- [ ] Placeholder cards display
- [ ] Basic reading flow works (even without LLM)

---

## Success Criteria

**MVP Complete When:**
1. ✅ App runs without crashing
2. ✅ All old code archived (48 files)
3. ✅ App.js cleaned (9 screens max)
4. ✅ 78 tarot cards exist (placeholder or real)
5. ✅ Veil Shard system works (earn, spend, caps)
6. ✅ Can perform at least 1 reading type
7. ✅ Dark fantasy aesthetic applied (colors, backgrounds)
8. ✅ Daily login bonus works
9. ✅ Free tier gates work (2/day limit)
10. ✅ Reading history saves

**Bonus (for full MVP):**
11. ✅ LLM integration for interpretations
12. ✅ 3 reading types (single, three-card, Celtic Cross)
13. ✅ Luna personality in readings
14. ✅ At least 1 NPC (Luna) with basic dialog
15. ✅ Settings screen works

---

## Pitfalls to Avoid

❌ **DON'T:**
- Delete old code (archive it instead)
- Break existing components others might use
- Hardcode colors (use palette system)
- Skip testing after each phase
- Try to implement everything at once
- Ignore the esoteric philosophy (tarot authenticity matters)
- Use cyberpunk aesthetics (that's the old vision)
- Implement $50/month hard cap (user changed mind - "use the shards")
- Add direct cash purchases (user said "use the shards")

✅ **DO:**
- Commit after each major change
- Test on actual device/emulator frequently
- Read design docs before implementing features
- Use ErrorBoundary to catch crashes
- Log everything to console for debugging
- Keep Luna's voice authentic (not chatbot-y)
- Implement free tier first, subscription later
- Focus on quality over speed

---

## Commit Strategy

After each phase:
```bash
git add -A
git commit -m "Phase X: [description]

Changes:
- [change 1]
- [change 2]

Status: [what works now]"

git push -u origin claude/sync-target-repo-01ULdcugdDT9oWr5mJbb5fRr
```

---

## Questions to Ask User (If Unclear)

1. **LLM Provider:** Which LLM API should I use? (OpenAI GPT-4, Anthropic Claude, other?)
2. **API Keys:** Where are API keys stored? (environment vars, config file?)
3. **Testing:** Can you test on device after each phase, or should I proceed fully before testing?
4. **Midjourney:** Do you have Midjourney access, or should I create Python scripts for other art generation?
5. **Priorities:** If time is limited, which is more important: (a) working gameplay or (b) polished aesthetics?

---

## Estimated Timeline

- **Phase 1 (Cleanup):** 2-3 hours
- **Phase 2 (Data Layer):** 6-8 hours
- **Phase 3 (Core Screens):** 8-12 hours
- **Phase 4 (Aesthetic):** 4-6 hours
- **Phase 5 (Art Generation):** 8-12 hours (mostly Midjourney wait time)
- **Phase 6 (RPG Systems):** 6-10 hours
- **Phase 7 (Monetization):** 4-6 hours

**Total: 38-57 hours to fully functional MVP**

**Minimum viable (Phases 1-3 only): 16-23 hours**

---

## Final Notes

This is a **massive** project with a beautiful, cohesive vision. The design documents are exceptional - now we just need to implement them.

**Remember:** The user was right when they said "i'm betting it crashes and looks nothing like you're saying." Currently, it's 1% complete. Your job is to make the reality match the vision.

**Start with Phase 1 immediately.** Archive the old code, clean up App.js, and get the app stable. Then build incrementally from there.

**The goal isn't perfection - it's a playable MVP.** Once the core loop works (load → menu → oracle → reading → shards), everything else can be added iteratively.

You've got this. 🌙✨

---

**End of Fixup Prompt**
