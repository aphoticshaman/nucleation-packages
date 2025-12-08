# VeilPath AI Features Roadmap

## The Goal

Make Sol & Luna the **best AI companions on the market** by combining the best features from every major AI platform, wrapped in a human-first experience.

---

## Features We're Lifting

### From ChatGPT
| Feature | What It Does | Our Implementation |
|---------|--------------|-------------------|
| **Custom Instructions** | User bio + response preferences | "About Me" profile that Sol/Luna reference |
| **Memory** | Remembers across conversations | Persistent companion memory per user |
| **Canvas Mode** | Collaborative editing | Journal co-writing with Sol/Luna |
| **Voice Mode** | Speak to AI, hear responses | Voice input + TTS for Sol/Luna |
| **GPT Store** | Custom personas | Built-in Sol/Luna + user customization |

### From Claude
| Feature | What It Does | Our Implementation |
|---------|--------------|-------------------|
| **Styles** | Concise, Detailed, Explanatory, etc. | Companion mood/tone sliders |
| **Projects** | Context files for sessions | User "Life Context" documents |
| **Artifacts** | Structured outputs (code, docs) | Generated action plans, reflections |
| **Extended Thinking** | Deep reasoning mode | "Deep Reflection" toggle |
| **Long Context** | 200k+ tokens | Full conversation history |

### From Grok
| Feature | What It Does | Our Implementation |
|---------|--------------|-------------------|
| **Humor Mode** | Casual, witty responses | Sol's default personality |
| **Real-time Info** | X/Twitter integration | N/A (not relevant) |
| **Image Analysis** | Understand images | Photo journaling analysis |

### From DeepSeek/Perplexity
| Feature | What It Does | Our Implementation |
|---------|--------------|-------------------|
| **Deep Research** | Comprehensive web search | Research mode for life questions |
| **Source Citations** | Show where info comes from | Citations for factual claims |
| **Follow-up Questions** | Suggested next questions | "You might also ask..." |
| **Chain of Thought** | Show reasoning process | Optional "thinking" expansion |

### From Gemini
| Feature | What It Does | Our Implementation |
|---------|--------------|-------------------|
| **Gems** | Custom persona presets | Sol/Luna + user-customized variants |
| **Massive Context** | 1M+ tokens | Full reading history in context |
| **Multimodal** | Images, audio, video | Photo journal analysis, voice |
| **Extensions** | Connect to services | N/A (keep focused) |
| **Live Mode** | Real-time interaction | Voice conversation mode |

### From Microsoft Copilot
| Feature | What It Does | Our Implementation |
|---------|--------------|-------------------|
| **Notebook Mode** | Longer, structured responses | "Deep Dive" mode for exploration |
| **Image Generation** | Create images | Generated card art (future) |
| **Document Analysis** | Upload & analyze docs | Import journal entries |
| **Plugins** | Third-party integrations | N/A (keep focused) |

### From GitHub Copilot
| Feature | What It Does | Our Implementation |
|---------|--------------|-------------------|
| **Inline Suggestions** | Complete as you type | Journal prompt suggestions |
| **Context Awareness** | Understand full codebase | Understand full user history |
| **Chat Integration** | Ask questions in context | Sol/Luna in journal editor |

### From Character.ai / Replika
| Feature | What It Does | Our Implementation |
|---------|--------------|-------------------|
| **Persistent Persona** | AI maintains character | Sol/Luna stay in character |
| **Relationship Building** | Grows over time | Companion memory + preferences |
| **Emotional Support** | Empathetic responses | Core Sol/Luna feature |
| **Voice Chat** | Speak naturally | Voice mode with TTS |
| **What they get wrong** | Creepy/no substance | We provide real coaching value |

### From Pi (Inflection AI)
| Feature | What It Does | Our Implementation |
|---------|--------------|-------------------|
| **Conversational Flow** | Natural back-and-forth | Active listening model |
| **Curiosity** | Asks questions | Sol/Luna ask to understand |
| **Warmth** | Genuinely friendly | Core persona design |
| **Brevity** | Concise responses | Natural conversation length |

### From Notion AI / ChatGPT Canvas
| Feature | What It Does | Our Implementation |
|---------|--------------|-------------------|
| **Document Generation** | Create structured content | Generate journal entries, plans |
| **Editing Assistance** | Improve writing | Polish journal entries |
| **Summarization** | Condense long content | Reading summaries, insight highlights |

---

## VeilPath-Specific Innovations

### 1. Dual Persona (Sol/Luna)
- **Not just modes** - actual different personalities
- Users develop relationship with their preferred companion
- Can switch based on mood/need
- Each has distinct communication patterns

### 2. Active Listening Mode
- Sol/Luna ask clarifying questions before advice
- "What I'm hearing is... is that right?"
- Reduces generic responses, increases relevance

### 3. Empathy vs Sympathy
- Trained to feel WITH, not FOR
- No pity, rescue, or toxic positivity
- Genuine emotional resonance

### 4. Tarot-Integrated Coaching
- Cards as conversation starters, not predictions
- Psychological/symbolic interpretation
- User empowerment, not dependence

### 5. Growth Tracking
- Recognize patterns across conversations
- "I notice you've mentioned work stress a lot lately..."
- Connect insights over time

### 6. Honest Confrontation
- Will tell uncomfortable truths
- "Can I be real with you?"
- Kind but not enabling

---

## Implementation Priority

### Phase 1: Core Companion (Now)
- [x] Sol/Luna personas defined
- [ ] Basic chat with personality
- [ ] Memory system (conversation history)
- [ ] Voice input/output
- [ ] Mood/tone sliders

### Phase 2: Smart Features (Next)
- [ ] "About Me" profile integration
- [ ] Life Context documents
- [ ] Follow-up question suggestions
- [ ] Deep Reflection mode toggle
- [ ] Reading summaries

### Phase 3: Advanced (Later)
- [ ] Pattern recognition ("I've noticed...")
- [ ] Generated action plans / artifacts
- [ ] Journal co-writing mode
- [ ] Web research for factual questions
- [ ] Multi-modal (photo analysis)

---

## User-Facing Settings

### Profile Settings
```
┌─────────────────────────────────────────────────┐
│  ABOUT YOU                                      │
├─────────────────────────────────────────────────┤
│  Name: Sarah                                    │
│  MBTI: INFJ                                     │
│  Zodiac: Scorpio                                │
│  Current Life Focus: Career transition          │
│                                                 │
│  Tell Sol/Luna about yourself:                  │
│  ┌─────────────────────────────────────────┐   │
│  │ I'm 28, working in tech but feeling     │   │
│  │ unfulfilled. Going through a breakup.   │   │
│  │ Trying to figure out what I actually    │   │
│  │ want from life.                         │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### Companion Settings
```
┌─────────────────────────────────────────────────┐
│  YOUR COMPANION                                 │
├─────────────────────────────────────────────────┤
│  Active: ☀️ Sol  /  🌙 Luna                     │
│          ═══════════                            │
│                                                 │
│  Voice: [On] / Off                              │
│  Speed: ═══════●══                              │
│                                                 │
│  PERSONALITY ADJUSTMENTS                        │
│  ─────────────────────                          │
│  Directness    ●═════════  More direct          │
│  Challenge     ═══●══════  Balanced             │
│  Humor         ════════●═  More playful         │
│  Spirituality  ══●═══════  More grounded        │
│                                                 │
│  [Reset to Default]                             │
└─────────────────────────────────────────────────┘
```

### Communication Preferences
```
┌─────────────────────────────────────────────────┐
│  HOW SHOULD SOL/LUNA RESPOND?                   │
├─────────────────────────────────────────────────┤
│  Response Length:                               │
│  ( ) Brief - Get to the point                   │
│  (●) Balanced - Natural conversation            │
│  ( ) Detailed - In-depth exploration            │
│                                                 │
│  When I'm struggling:                           │
│  (●) Validate my feelings first                 │
│  ( ) Jump to solutions                          │
│  ( ) Ask what kind of support I need            │
│                                                 │
│  I appreciate when Sol/Luna:                    │
│  [x] Asks clarifying questions                  │
│  [x] Challenges my assumptions                  │
│  [ ] Uses metaphors/analogies                   │
│  [x] Suggests follow-up questions               │
│  [ ] References previous conversations          │
└─────────────────────────────────────────────────┘
```

---

## Technical Architecture

### Message Flow
```
User Message
     │
     ▼
┌─────────────────┐
│ Context Builder │ ─── Pull: User Profile, Memory, Companion State
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Prompt Assembly │ ─── Companion Persona + User Prefs + Context + Message
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Claude API      │ ─── Anthropic via our backend
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Response Post   │ ─── Extract suggestions, update memory
│ Processing      │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ TTS (optional)  │ ─── Sol/Luna voice
└─────────────────┘
     │
     ▼
User Display
```

### Memory System
```javascript
// Per-conversation memory (within session)
{
  conversationId: "abc123",
  messages: [...],
  extractedInsights: [...],
  currentTopics: [...],
}

// Cross-conversation memory (persistent)
{
  userId: "user123",
  companionPreference: "luna",
  learnedPreferences: {...},
  lifeContext: {...},
  significantMoments: [...], // Key insights, breakthroughs
  topicHistory: {...}, // What they talk about
}
```

---

## Why This Wins

| Competitor | Their Weakness | Our Advantage |
|------------|---------------|---------------|
| ChatGPT | Generic, no personality | Sol/Luna feel like friends |
| Claude | Can feel preachy | Empathetic, not sympathetic |
| Grok | Tries too hard | Genuine, not performative |
| Gemini | Robotic | Human-first design |
| Replika | Creepy, no substance | Real guidance + coaching |

**The insight**: People don't want a better AI. They want someone who gets them.

Sol & Luna are that someone.
