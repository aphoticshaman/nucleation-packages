# LatticeForge UX Designer Specification

## Document Purpose

This specification provides comprehensive guidance for UX designers working on LatticeForge. It covers user research synthesis, information architecture, interaction patterns, user flows, wireframe specifications, and usability considerations. The goal is to create an AI-augmented research platform that feels intuitive despite its underlying complexity.

---

## 1. Product Context and User Mental Models

### 1.1 What LatticeForge Actually Does

LatticeForge is an AI-powered research intelligence platform that helps researchers, analysts, and knowledge workers synthesize information from multiple sources, generate insights, and track the evolution of their understanding over time. Think of it as the collision of three mental models:

1. **Research Assistant**: Like having a brilliant colleague who reads everything and surfaces connections
2. **Knowledge Graph**: Like a mind map that builds itself and finds unexpected links
3. **Audit Trail**: Like version control for your thinking process

Users don't need to understand the AI, the knowledge graphs, or the technical substrate. They need to feel like their research suddenly has a co-pilot.

### 1.2 Primary User Personas

**Dr. Sarah Chen - Academic Researcher**
- 42, Associate Professor of Computational Biology
- Manages 3 PhD students, publishes 4-6 papers/year
- Pain: Drowning in literature, missing cross-disciplinary connections
- Goal: Spend less time on literature review, more on actual research
- Tech comfort: Uses R, Python, comfortable with complex tools
- Quote: "I know there's a paper somewhere that connects to this, but I can't remember where"

**Marcus Williams - Intelligence Analyst**
- 34, Senior Analyst at a think tank
- Synthesizes geopolitical data from diverse sources
- Pain: Information overload, difficulty tracking narrative evolution
- Goal: Identify emerging patterns before they become obvious
- Tech comfort: Power Excel user, basic SQL, skeptical of black boxes
- Quote: "I need to explain my reasoning to stakeholders who weren't in the room"

**Elena Rodriguez - Product Strategy Lead**
- 38, VP of Strategy at a Series C startup
- Conducts market research, competitive analysis
- Pain: Research gets stale, team members duplicate efforts
- Goal: Build institutional knowledge that persists across projects
- Tech comfort: Heavy Notion/Airtable user, wants polish
- Quote: "Our research insights die in slide decks"

**James Park - PhD Student**
- 27, Third-year PhD in Materials Science
- Writing dissertation, needs to master adjacent fields quickly
- Pain: Imposter syndrome, fear of missing canonical papers
- Goal: Get up to speed in new domains without embarrassment
- Tech comfort: Native digital, expects modern UX
- Quote: "I don't know what I don't know"

### 1.3 Mental Model Transitions

Users come to LatticeForge with existing mental models that we must bridge:

| From (Familiar) | To (LatticeForge) | Bridge Concept |
|-----------------|-------------------|----------------|
| Google Scholar search | Continuous research monitoring | "Your research runs even when you're not looking" |
| Folder-based organization | Graph-based relationships | "Ideas connect themselves" |
| Static notes | Living synthesis documents | "Your notes grow with new evidence" |
| Manual citation tracking | Automatic provenance | "Every insight traces back to its source" |
| Single-user tools | Collaborative intelligence | "Build on each other's discoveries" |

---

## 2. Information Architecture

### 2.1 Primary Navigation Structure

```
LatticeForge
├── Dashboard (Home)
│   ├── Active Research Streams
│   ├── Recent Insights
│   ├── Team Activity (if applicable)
│   └── Quick Actions
│
├── Research Streams
│   ├── Stream List/Grid View
│   ├── Stream Detail
│   │   ├── Sources Panel
│   │   ├── Synthesis Canvas
│   │   ├── Graph Explorer
│   │   └── Timeline View
│   └── Create New Stream
│
├── Library
│   ├── All Sources
│   ├── Collections
│   ├── Annotations
│   └── Import Center
│
├── Insights
│   ├── Generated Insights
│   ├── Saved Insights
│   ├── Insight Chains
│   └── Export Center
│
├── Graph Explorer (Global)
│   ├── Full Knowledge Graph
│   ├── Concept Clusters
│   └── Connection Discovery
│
└── Settings
    ├── Profile & Preferences
    ├── Team Management
    ├── Integrations
    └── API Access
```

### 2.2 Object Hierarchy and Relationships

**Primary Objects:**

1. **Research Stream**: The main workspace for a research initiative
   - Contains: Sources, Syntheses, Insights, Graph Views
   - Metaphor: A dedicated workspace for a project

2. **Source**: Any input material (papers, articles, documents, URLs)
   - Properties: Content, metadata, annotations, extracted entities
   - Metaphor: A document in your filing cabinet

3. **Synthesis**: AI-generated or human-written summaries connecting sources
   - Properties: Text, source references, confidence indicators
   - Metaphor: Your research notes that cite sources

4. **Insight**: A discrete finding or connection
   - Properties: Statement, evidence chain, novelty score
   - Metaphor: A sticky note with a breakthrough idea

5. **Entity**: An extracted concept, person, organization, or term
   - Properties: Name, type, occurrences, relationships
   - Metaphor: Index card in a card catalog

**Relationship Types:**
- Source → Source: Citations, thematic links
- Source → Entity: Extraction (this source mentions this entity)
- Entity → Entity: Relationships (collaborates with, contradicts, etc.)
- Insight → Source: Evidence (this insight comes from these sources)
- Synthesis → Insight: Aggregation (this synthesis contains these insights)

### 2.3 URL Structure and Deep Linking

Every meaningful state should be addressable:

```
/dashboard
/streams
/streams/[stream-id]
/streams/[stream-id]/sources
/streams/[stream-id]/synthesis
/streams/[stream-id]/graph
/streams/[stream-id]/timeline
/library
/library/sources/[source-id]
/library/collections/[collection-id]
/insights
/insights/[insight-id]
/graph
/graph?focus=[entity-id]
/settings/[section]
```

Deep links should capture:
- Current view state
- Selected items
- Filter configurations
- Graph zoom/pan position (for sharing specific views)

---

## 3. Core User Flows

### 3.1 Onboarding Flow

**Goal**: Get user to first meaningful insight within 10 minutes

**Flow Steps:**

```
1. Sign Up / Sign In
   ├── Social auth (Google, GitHub, ORCID)
   ├── Email/password
   └── SSO (enterprise)

2. Welcome Screen
   ├── Brief value proposition (3 sentences max)
   ├── Skip option always visible
   └── Progress indicator (4 dots)

3. Research Focus Selection
   ├── "What do you research?" (free text)
   ├── Suggested domains based on input
   └── This primes AI assistance but isn't binding

4. First Source Import
   ├── Three options presented equally:
   │   ├── Paste a URL
   │   ├── Upload a PDF
   │   └── Connect Google Scholar
   ├── Processing indicator with educational content
   └── Success celebration (subtle, not patronizing)

5. First Insight Preview
   ├── Show automatically extracted entities
   ├── Show one AI-generated insight
   ├── "This is what LatticeForge can do"
   └── CTA: "Start your first research stream"

6. Dashboard (First-Time State)
   ├── Created stream visible
   ├── Contextual tooltips (dismissible)
   └── Empty state templates for other areas
```

**Critical Metrics:**
- Time to first source added: < 3 minutes
- Time to first insight seen: < 5 minutes
- Completion rate of onboarding: > 70%
- Skip rate: Track but don't optimize against

### 3.2 Research Stream Creation Flow

**Goal**: Set up a focused research workspace efficiently

```
1. Trigger
   ├── "New Stream" button (header, dashboard, empty state)
   └── Keyboard shortcut: Cmd/Ctrl + N

2. Stream Configuration (Single Modal)
   ├── Name (required, auto-focused)
   │   └── Placeholder: "e.g., CRISPR delivery mechanisms"
   ├── Description (optional, expandable)
   ├── Initial sources (optional)
   │   ├── Drag-drop zone
   │   ├── URL paste
   │   └── Select from library
   └── Visibility (private/team) if team features enabled

3. Stream Created
   ├── Redirect to stream detail view
   ├── If sources provided: Show processing state
   ├── If no sources: Show curated empty state with suggestions
   └── Contextual help: "Add sources to get started"
```

**Empty State for New Stream:**
```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │              [Import Icon]                              │   │
│  │                                                         │   │
│  │         Add sources to begin your research              │   │
│  │                                                         │   │
│  │   Drop files here, paste URLs, or browse your library   │   │
│  │                                                         │   │
│  │   ┌──────────┐ ┌──────────┐ ┌──────────────────────┐   │   │
│  │   │ Upload   │ │ Paste    │ │ Browse Library       │   │   │
│  │   │ Files    │ │ URL      │ │                      │   │   │
│  │   └──────────┘ └──────────┘ └──────────────────────┘   │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Suggested sources based on your research focus:               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [+] "Recent advances in [topic]" - Nature Reviews, 2024 │   │
│  │ [+] "[Related concept] systematic review" - 847 cites   │   │
│  │ [+] Import from your Zotero library                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Source Import and Processing Flow

**Goal**: Get sources into the system with minimal friction, clear status

```
1. Import Triggers (Multiple Entry Points)
   ├── Drag-drop anywhere in stream view
   ├── "Add Source" button
   ├── Browser extension (one-click from paper page)
   ├── Email forwarding (forward papers to your-stream@latticeforge.io)
   └── API/integrations (Zotero, Mendeley, etc.)

2. Import Modal (if not drag-drop)
   ├── Tab: Upload Files (PDFs, Word, etc.)
   ├── Tab: Paste URLs (one per line, or comma-separated)
   ├── Tab: Search (DOI, arXiv ID, title search)
   └── Tab: Library (select existing sources)

3. Processing State
   ├── Source appears immediately in list (optimistic UI)
   ├── Processing indicator shows current step:
   │   ├── "Uploading..." (for files)
   │   ├── "Fetching..." (for URLs)
   │   ├── "Extracting text..."
   │   ├── "Identifying entities..."
   │   └── "Finding connections..."
   ├── User can continue working (non-blocking)
   └── Error handling: Inline error with retry option

4. Processing Complete
   ├── Subtle notification (not disruptive)
   ├── Source card updates to show:
   │   ├── Title (extracted or from metadata)
   │   ├── Authors
   │   ├── Publication info
   │   ├── Entity count badge
   │   └── Thumbnail (first figure or generic icon)
   └── If part of batch: Progress bar updates

5. Optional: Quick Review
   ├── After processing, source can be expanded in-place
   ├── Shows extracted metadata for verification
   ├── "Edit" option for corrections
   └── Most users skip this (automatic extraction is good enough)
```

**Processing States Visual:**
```
┌────────────────────────────────────────────────────────────┐
│ ┌──────┐                                                   │
│ │ PDF  │  Understanding Deep Learning                      │
│ │ icon │  Processing: Extracting text...  [████░░░░] 45%  │
│ └──────┘                                                   │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ ┌──────┐                                                   │
│ │ PDF  │  Understanding Deep Learning                      │
│ │ icon │  Prince, S. (2023) · MIT Press · 42 entities     │
│ └──────┘  ● Ready                                          │
└────────────────────────────────────────────────────────────┘
```

### 3.4 Synthesis Generation Flow

**Goal**: Transform sources into coherent synthesis with user control

```
1. Trigger Synthesis
   ├── "Generate Synthesis" button in stream
   ├── Context: Which sources to include
   │   ├── All sources (default)
   │   ├── Selected sources only
   │   └── Sources matching filter
   └── Optional: Focus prompt ("Focus on methodology", etc.)

2. Configuration (Expandable, Not Required)
   ├── Synthesis type:
   │   ├── General overview (default)
   │   ├── Literature review style
   │   ├── Comparison/contrast
   │   ├── Gap analysis
   │   └── Custom prompt
   ├── Length preference: Brief / Standard / Comprehensive
   └── Include: Figures / Tables / Code (checkboxes)

3. Generation (Streamed)
   ├── Text appears word-by-word (streaming)
   ├── User can read as it generates
   ├── Inline citations appear as [1], [2], hoverable
   ├── Cancel button available throughout
   └── Edit available even while generating

4. Post-Generation
   ├── Synthesis saved automatically
   ├── Options:
   │   ├── Edit (rich text editor)
   │   ├── Regenerate (with same or different params)
   │   ├── Export (Markdown, Word, LaTeX)
   │   └── Add to Insight Chain
   └── Version history accessible
```

**Streaming Synthesis UI:**
```
┌────────────────────────────────────────────────────────────────┐
│ Synthesis: CRISPR Delivery Mechanisms                      [X] │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ Recent advances in CRISPR delivery have focused on three      │
│ primary vectors: lipid nanoparticles (LNPs), viral vectors,   │
│ and cell-penetrating peptides [1][2]. The choice of delivery  │
│ mechanism significantly impacts both editing efficiency and    │
│ off-target effects [3].                                       │
│                                                                │
│ LNPs have emerged as the leading non-viral approach,          │
│ particularly following their successful deployment in mRNA    │
│ vaccines [4]. However, tissue-specific targeting remains█     │
│                                                                │
│                              [Stop Generating]                 │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│ Sources: 12 included · Generated in 8.3s · [Edit] [Regenerate]│
└────────────────────────────────────────────────────────────────┘
```

### 3.5 Insight Discovery Flow

**Goal**: Surface non-obvious connections, let users capture and build on them

```
1. Passive Discovery (Background)
   ├── System continuously analyzes sources
   ├── Generates candidate insights
   ├── Ranks by novelty, confidence, relevance
   └── Surfaces top candidates in sidebar

2. Active Discovery (User-Initiated)
   ├── "Find Insights" button
   ├── Options:
   │   ├── Between selected sources
   │   ├── Connecting two entities
   │   ├── "What am I missing?"
   │   └── "What contradictions exist?"
   └── Results appear in dedicated panel

3. Insight Presentation
   ├── Card format with:
   │   ├── Insight statement (1-2 sentences)
   │   ├── Confidence indicator (visual, not numeric)
   │   ├── Evidence sources (clickable)
   │   ├── Related entities (tags)
   │   └── Actions: Save / Dismiss / Explore
   └── Expandable for full reasoning chain

4. Insight Actions
   ├── Save: Adds to saved insights, can organize into chains
   ├── Dismiss: Removes from suggestions (trains model)
   ├── Explore: Opens graph view centered on insight
   └── Challenge: "This seems wrong" → generates counter-evidence

5. Building Insight Chains
   ├── Saved insights can be connected
   ├── Create narrative flow: Insight A → Insight B → Conclusion
   ├── Export as research narrative
   └── Track which insights led to which conclusions
```

**Insight Card:**
```
┌────────────────────────────────────────────────────────────┐
│ 💡 Potential Connection                                    │
├────────────────────────────────────────────────────────────┤
│                                                            │
│ The protein folding mechanism described by Chen et al.     │
│ (2023) may explain the anomalous binding behavior in       │
│ your earlier source from Williams (2022).                  │
│                                                            │
│ ┌──────────────────────────────────────────────────────┐  │
│ │ ●●●●○ High Confidence                                │  │
│ └──────────────────────────────────────────────────────┘  │
│                                                            │
│ Evidence: [Chen 2023] [Williams 2022] [Park 2021]         │
│ Entities: protein folding, binding affinity, pH sensitivity│
│                                                            │
├────────────────────────────────────────────────────────────┤
│ [Save]  [Explore in Graph]  [Dismiss]  [Challenge]        │
└────────────────────────────────────────────────────────────┘
```

### 3.6 Graph Exploration Flow

**Goal**: Let users explore connections visually without getting lost

```
1. Entry Points
   ├── "View Graph" button in stream
   ├── "Explore" action on any entity or insight
   ├── Global graph explorer in navigation
   └── Click entity tag anywhere in app

2. Initial Graph View
   ├── Centered on entry point (source, entity, or insight)
   ├── 1-hop neighbors visible
   ├── Color coding by type (sources, entities, insights)
   ├── Edge thickness indicates relationship strength
   └── Subtle animation on load (graph settles)

3. Navigation
   ├── Click node: Select, show details in sidebar
   ├── Double-click node: Recenter graph on that node
   ├── Scroll: Zoom in/out
   ├── Drag background: Pan
   ├── Drag node: Reposition (force simulation adjusts)
   └── Right-click: Context menu (expand, hide, focus)

4. Filtering and Focus
   ├── Type toggles: Show/hide sources, entities, insights
   ├── Time range: Filter by when sources were added
   ├── Depth slider: 1-hop, 2-hop, 3-hop neighborhoods
   └── Search within graph: Highlight matching nodes

5. Graph Actions
   ├── Expand: Show more neighbors of selected node
   ├── Collapse: Hide children of selected node
   ├── Focus: Dim everything except path between two nodes
   ├── Cluster: Group highly-connected nodes
   └── Export: Image (PNG, SVG) or data (JSON)

6. Insight from Graph
   ├── Select multiple nodes: "What connects these?"
   ├── System generates insight about selection
   └── Can save result as new insight
```

**Graph View Wireframe:**
```
┌─────────────────────────────────────────────────────────────────────────┐
│ Graph Explorer                                           [Filter ▼] [X] │
├───────────────────────────────────────────────────────┬─────────────────┤
│                                                       │                 │
│            ○───────────○                              │ Selected:       │
│           /             \                             │ ┌─────────────┐ │
│          ○       ●       ○                            │ │ CRISPR-Cas9 │ │
│         / \     / \     / \                           │ │   (Entity)  │ │
│        ○   ○───●───○───○   ○                          │ └─────────────┘ │
│             \  │  /                                   │                 │
│              ○─●─○                                    │ Connections: 24 │
│                │                                      │ Sources: 8      │
│                ○                                      │ First seen:     │
│                                                       │ March 2024      │
│  ● Selected   ○ Entity   □ Source   ◇ Insight        │                 │
│                                                       │ [Expand]        │
│  Depth: [1] [2] [3]    Types: [●] [○] [□] [◇]       │ [Find Paths]    │
│                                                       │ [Hide]          │
└───────────────────────────────────────────────────────┴─────────────────┘
```

---

## 4. Interaction Patterns

### 4.1 Selection and Multi-Select

**Single Selection:**
- Click to select
- Selected item gets visual highlight (border, background)
- Selection persists until new selection or explicit deselect
- ESC clears selection

**Multi-Select:**
- Cmd/Ctrl + Click to add/remove from selection
- Shift + Click for range select (in lists)
- Drag rectangle in graph/canvas views
- Selection count badge when multiple selected
- Actions apply to all selected items

**Bulk Actions Bar:**
When 2+ items selected, a contextual bar appears:
```
┌────────────────────────────────────────────────────────────────┐
│ 5 sources selected   [Add to Collection ▼] [Export] [Delete]  │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 Drag and Drop

**Supported Interactions:**
| Drag | Drop Zone | Result |
|------|-----------|--------|
| File (PDF, etc.) | Stream view | Import source |
| File | Library | Import to library |
| Source card | Collection | Add to collection |
| Source card | Another stream | Copy to stream |
| Entity tag | Search bar | Filter by entity |
| Insight card | Insight chain | Add to chain |
| Graph node | Outside graph | Expand in new panel |

**Visual Feedback:**
- Drag preview: Semi-transparent clone of item
- Valid drop zones: Highlighted border, subtle pulse
- Invalid zones: No visual change (ignore, don't error)
- Drop success: Brief animation (settle into place)

### 4.3 Real-Time Updates

**WebSocket-Connected State:**
- Source processing progress
- New insights as they're generated
- Team member activity (if collaborative)
- Graph updates when new connections found

**Update Patterns:**
1. **Append**: New items appear at top/bottom of lists
2. **In-place**: Existing items update without layout shift
3. **Badge**: Count badges increment (e.g., "3 new insights")
4. **Toast**: Important events get brief notification

**Stale Data Handling:**
- Background tab: Queue updates, apply on focus
- Reconnection: Fetch delta, animate reconciliation
- Conflict: Last-write-wins for user edits, additive for system

### 4.4 Search and Filtering

**Global Search (Cmd/Ctrl + K):**
```
┌────────────────────────────────────────────────────────────────┐
│ 🔍 Search LatticeForge...                                      │
├────────────────────────────────────────────────────────────────┤
│ Recent:                                                        │
│   CRISPR delivery mechanisms (stream)                          │
│   protein folding (entity)                                     │
├────────────────────────────────────────────────────────────────┤
│ Actions:                                                       │
│   + Create new stream                                          │
│   ↑ Upload sources                                             │
│   ⚙ Settings                                                   │
└────────────────────────────────────────────────────────────────┘
```

**Search Behavior:**
- Instant results (debounced, 150ms)
- Fuzzy matching with highlighting
- Categories: Streams, Sources, Entities, Insights, Actions
- Keyboard navigation: Up/Down to select, Enter to go

**Contextual Filters:**
Every list view should have discoverable but non-intrusive filters:
```
Sources (47)  [Filter: Type ▼] [Sort: Recent ▼] [🔍 Search...]
```

Filter dropdowns are multi-select with counts:
```
┌─────────────────────────────┐
│ Type                        │
├─────────────────────────────┤
│ ☑ Research Paper (32)       │
│ ☑ Article (12)              │
│ ☐ Book Chapter (3)          │
│ ☑ Preprint (0)              │
├─────────────────────────────┤
│ [Clear] [Apply]             │
└─────────────────────────────┘
```

### 4.5 Keyboard Shortcuts

**Philosophy:**
- Common actions have single-key shortcuts
- Destructive actions require modifier
- All shortcuts shown in tooltips and command palette
- Customizable in settings

**Global Shortcuts:**
| Shortcut | Action |
|----------|--------|
| Cmd/Ctrl + K | Global search |
| Cmd/Ctrl + N | New stream |
| Cmd/Ctrl + U | Upload source |
| Cmd/Ctrl + / | Show all shortcuts |
| ESC | Close modal, clear selection, or go back |
| ? | Help overlay |

**View-Specific Shortcuts:**
| Context | Shortcut | Action |
|---------|----------|--------|
| Stream | G | Generate synthesis |
| Stream | I | Find insights |
| List | J/K | Move selection down/up |
| List | Enter | Open selected |
| Graph | +/- | Zoom in/out |
| Graph | 0 | Reset view |
| Graph | E | Expand selected node |

### 4.6 Progressive Disclosure

**Principle**: Show only what's needed at each moment, with clear paths to more.

**Patterns:**

1. **Expandable Sections:**
```
▶ Advanced Options
  [Click to expand to see: Focus prompt, length, format, etc.]
```

2. **Hover for Details:**
```
[Citation [1]] → hover shows: "Chen et al. (2023). Nature Methods."
```

3. **"See More" for Long Lists:**
```
Top 5 entities shown
[Show 17 more...]
```

4. **Contextual Help:**
```
┌─────────────────────────────────────────────────────────────┐
│ Confidence indicates how strongly the evidence supports     │
│ this insight. [Learn more]                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Empty States and Error Handling

### 5.1 Empty State Hierarchy

**Tier 1 - No Data Yet (New User):**
- Warm, encouraging tone
- Clear primary action
- Secondary options visible but de-emphasized
- Brief explanation of what will appear here

**Tier 2 - Filtered to Empty:**
- Acknowledge the filter
- Show how to clear/adjust filter
- Suggest related items if available

**Tier 3 - Search with No Results:**
- Confirm what was searched
- Suggest alternatives
- Offer to create based on search term

**Examples:**

New Stream (Tier 1):
```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│              📚                                            │
│                                                            │
│        Your research stream is ready                       │
│                                                            │
│   Add sources to start building your knowledge base.       │
│   LatticeForge will find connections automatically.        │
│                                                            │
│              [Add Your First Source]                       │
│                                                            │
│   Or: Import from Zotero · Browse sample streams           │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

Filtered Empty (Tier 2):
```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│   No sources match "Type: Book Chapter"                    │
│                                                            │
│   Your stream has 47 sources of other types.               │
│   [Clear Filter] or [Adjust Filters]                       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 5.2 Error States

**Transient Errors (Retry-able):**
```
┌────────────────────────────────────────────────────────────┐
│ ⚠️ Couldn't load insights                                  │
│                                                            │
│ There was a connection issue. [Try Again] or               │
│ the page will retry automatically in 10 seconds.           │
└────────────────────────────────────────────────────────────┘
```

**Processing Errors (Source-specific):**
```
┌────────────────────────────────────────────────────────────┐
│ ┌──────┐                                                   │
│ │ PDF  │  encrypted_file.pdf                               │
│ │ ⚠️   │  Could not process: File is password-protected   │
│ └──────┘  [Retry with Password] [Remove]                   │
└────────────────────────────────────────────────────────────┘
```

**Permission Errors:**
```
┌────────────────────────────────────────────────────────────┐
│ 🔒 You don't have access to this stream                    │
│                                                            │
│ This stream belongs to another workspace.                  │
│ [Request Access] [Go to Dashboard]                         │
└────────────────────────────────────────────────────────────┘
```

**Validation Errors:**
- Inline, next to the field
- Red text, but not red background (accessibility)
- Specific about what's wrong and how to fix
- Appear immediately on blur, not on submit

### 5.3 Loading States

**Skeleton Loading:**
For content that has a known structure:
```
┌────────────────────────────────────────────────────────────┐
│ ┌──────┐  ████████████████████████                         │
│ │      │  ██████████████                                   │
│ │ ░░░░ │  ████████████████████                             │
│ └──────┘                                                   │
├────────────────────────────────────────────────────────────┤
│ ┌──────┐  ████████████████████████                         │
│ │      │  ██████████████████                               │
│ │ ░░░░ │  ██████████████                                   │
│ └──────┘                                                   │
└────────────────────────────────────────────────────────────┘
```

**Spinner Loading:**
For actions where duration is unpredictable:
```
[○ Loading...] (animated spinner)
```

**Progress Loading:**
For multi-step processes:
```
Processing source: [████████░░░░░░░░] Extracting entities...
```

---

## 6. Mobile and Responsive Considerations

### 6.1 Responsive Breakpoints

| Breakpoint | Width | Primary Changes |
|------------|-------|-----------------|
| Desktop XL | ≥1440px | 3-column layouts, full graph |
| Desktop | 1024-1439px | 2-column layouts |
| Tablet | 768-1023px | Collapsible sidebars |
| Mobile | <768px | Single-column, bottom navigation |

### 6.2 Mobile Adaptations

**Navigation:**
- Bottom tab bar replaces sidebar
- 5 primary destinations max
- More actions in hamburger/sheet

**Graph Explorer:**
- Simplified view with fewer nodes visible
- Touch-optimized: Tap to select, long-press for menu
- Gestures: Pinch to zoom, drag to pan
- "Desktop view recommended" notice for complex graphs

**Input Adaptations:**
- URL paste works from share sheet
- Camera/files picker for source upload
- Voice input for search (system native)

**Interaction Changes:**
- No hover states; use tap + detail panel
- Larger touch targets (44px minimum)
- Pull-to-refresh in lists
- Swipe actions on cards (archive, delete)

### 6.3 Progressive Enhancement

Core functionality must work on all devices:
1. Create streams
2. Add sources (URL, file)
3. View syntheses and insights
4. Basic navigation

Enhanced features on capable devices:
1. Full graph exploration
2. Drag-and-drop organization
3. Keyboard shortcuts
4. Multi-select operations

---

## 7. Accessibility Requirements

### 7.1 WCAG 2.1 AA Compliance

**Perceivable:**
- Color contrast: 4.5:1 for normal text, 3:1 for large text
- Text alternatives for all non-text content
- Captions for any video content
- Content adaptable to different presentations

**Operable:**
- All functionality keyboard-accessible
- No keyboard traps
- Focus visible and logical
- Sufficient time for reading and interaction

**Understandable:**
- Language of page identified
- Consistent navigation
- Error identification and suggestion

**Robust:**
- Valid HTML
- Name, role, value exposed to assistive technology
- Status messages announced appropriately

### 7.2 Specific Requirements

**Focus Management:**
```css
/* Visible focus ring */
:focus {
  outline: 2px solid var(--focus-ring-color);
  outline-offset: 2px;
}

/* Remove outline for mouse users */
:focus:not(:focus-visible) {
  outline: none;
}
```

**Screen Reader Considerations:**
- Live regions for dynamic content updates
- Proper heading hierarchy (h1 → h2 → h3)
- ARIA labels for icon-only buttons
- Skip links for main content

**Motion Sensitivity:**
```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

### 7.3 Testing Protocol

1. **Automated**: Run aXe or Lighthouse on every page
2. **Keyboard**: Navigate entire app without mouse
3. **Screen Reader**: Test with VoiceOver (Mac), NVDA (Windows)
4. **Zoom**: Verify at 200% zoom
5. **Color**: Check with color blindness simulators

---

## 8. Collaboration and Social Features

### 8.1 Sharing Model

**Visibility Levels:**
1. **Private**: Only creator can access
2. **Team**: All workspace members can view/edit
3. **Link Sharing**: Anyone with link can view (optional)
4. **Public**: Discoverable by all users (optional future)

**Share Modal:**
```
┌────────────────────────────────────────────────────────────┐
│ Share "CRISPR Delivery Mechanisms"                     [X] │
├────────────────────────────────────────────────────────────┤
│                                                            │
│ Team Access                                                │
│ ┌────────────────────────────────────────────────────────┐ │
│ │ 👤 Sarah Chen (you)                            Owner   │ │
│ │ 👤 Marcus Williams                       Can edit  ▼  │ │
│ │ + Add people...                                        │ │
│ └────────────────────────────────────────────────────────┘ │
│                                                            │
│ Link Sharing ───────────────────────────────────── ○ Off   │
│                                                            │
│ [Copy Link] [Done]                                         │
└────────────────────────────────────────────────────────────┘
```

### 8.2 Presence and Activity

**Real-Time Presence:**
- Avatars show who's viewing same stream
- Cursor position shown for collaborators (optional)
- "Currently editing" indicator on items

**Activity Feed:**
```
Recent Activity
──────────────────────────────────────────
👤 Marcus added 3 sources                 2h ago
💡 New insight generated                  4h ago
📝 Sarah updated synthesis                Yesterday
```

### 8.3 Comments and Annotations

**Comment Anchoring:**
Comments can attach to:
- Sources (general comment)
- Specific passages (highlighted text)
- Insights (agree/disagree/build)
- Graph nodes (relationship comments)

**Comment Thread:**
```
┌────────────────────────────────────────────────────────────┐
│ ┌──────────────────────────────────────────────────────┐  │
│ │ The methodology section seems inconsistent with...   │  │
│ │                                      — Marcus, 2h ago │  │
│ └──────────────────────────────────────────────────────┘  │
│     ↳ ┌────────────────────────────────────────────────┐  │
│       │ Good catch. I'll flag this in the synthesis.  │  │
│       │                               — Sarah, 1h ago  │  │
│       └────────────────────────────────────────────────┘  │
│                                                            │
│ [Add reply...]                                             │
└────────────────────────────────────────────────────────────┘
```

---

## 9. Performance Perception

### 9.1 Speed Targets

| Interaction | Target | Measurement |
|-------------|--------|-------------|
| Page navigation | <200ms | Time to interactive |
| Search results | <100ms | First result visible |
| Source upload | <2s | Processing started indicator |
| Synthesis generation | Streaming | First token <500ms |
| Graph render | <500ms | Nodes visible |

### 9.2 Perceived Performance Techniques

**Optimistic Updates:**
User actions reflect immediately, sync in background:
- Adding source: Card appears, "Processing..." state
- Saving insight: Saved state shown, confirmation async
- Drag-drop: Item moves, revert if fails

**Progressive Loading:**
Above-fold content loads first:
```
[Header loads]
[Search loads]
[First 5 sources load]
[Remaining sources lazy-load on scroll]
```

**Streaming Content:**
AI-generated content streams in real-time:
- Synthesis: Words appear as generated
- Insights: Cards populate progressively
- Graph: Nodes animate in from center

**Background Processing:**
Heavy operations don't block UI:
- Source processing: Notification when done
- Batch imports: Progress indicator
- Graph calculations: Incremental updates

---

## 10. Design Deliverable Checklist

### 10.1 Per-Feature Deliverables

For each major feature, provide:

1. **User Story Map**
   - Who is the user?
   - What are they trying to accomplish?
   - What's the happy path?
   - What can go wrong?

2. **Flow Diagram**
   - Entry points
   - Decision points
   - Exit points
   - Error branches

3. **Wireframes**
   - Low-fidelity for structure
   - Key screens only (not every state)
   - Annotations for interactions

4. **Interaction Specifications**
   - Transitions and animations
   - Hover/focus/active states
   - Keyboard behavior
   - Touch equivalents

5. **Edge Case Documentation**
   - Empty states
   - Error states
   - Loading states
   - Permission variations

### 10.2 Handoff Format

**For Engineering:**
- Figma/Sketch files with auto-layout
- Component specifications
- Responsive behavior notes
- Accessibility annotations

**For Visual Design:**
- Wireframes as starting point
- Information hierarchy notes
- Content strategy guidance
- Interaction timing specs

**For Product:**
- User journey maps
- Success metrics recommendations
- Feature flag recommendations
- A/B test opportunities

---

## 11. Usability Testing Protocol

### 11.1 Continuous Testing Framework

**Weekly Unmoderated Tests:**
- 5 tasks per week, 5 participants
- Record completion rate, time, satisfaction
- Focus on recently shipped features

**Monthly Moderated Sessions:**
- 6-8 participants, 45-60 minutes
- Think-aloud protocol
- Focus on complex flows

**Quarterly Comprehensive Review:**
- Full journey testing
- Competitive benchmarking
- Persona validation

### 11.2 Key Metrics to Track

| Metric | Target | Collection Method |
|--------|--------|-------------------|
| Task Success Rate | >85% | Usability testing |
| Time on Task | Varies by task | Usability testing |
| Error Rate | <10% | Analytics |
| System Usability Scale | >70 | Quarterly survey |
| Net Promoter Score | >40 | Quarterly survey |
| Feature Adoption | >60% MAU | Analytics |

### 11.3 Test Task Bank

**Onboarding:**
1. Create an account and add your first source
2. Find where your imported sources are stored
3. Create a new research stream

**Core Workflows:**
1. Import 3 papers related to [topic] and generate a synthesis
2. Find the connection between [Entity A] and [Entity B]
3. Save an insight and add your own note to it
4. Share your research stream with a colleague

**Advanced:**
1. Use the graph to discover a non-obvious connection
2. Build an insight chain from 3 saved insights
3. Export your synthesis in a format suitable for a grant proposal

---

## 12. Appendix: Wireframe Templates

### 12.1 Page Templates

**Dashboard:**
```
┌──────────────────────────────────────────────────────────────────────────┐
│ [Logo]  Research Streams  Library  Insights  Graph     [Search] [Avatar]│
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Welcome back, Sarah                                                     │
│                                                                          │
│  ┌─ Active Streams ─────────────────────────────────────────────────┐   │
│  │                                                                   │   │
│  │  [Stream Card]  [Stream Card]  [Stream Card]  [+ New Stream]     │   │
│  │                                                                   │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─ Recent Insights ────────────────────────────────────────────────┐   │
│  │                                                                   │   │
│  │  [Insight Card]  [Insight Card]  [Insight Card]                  │   │
│  │                                                                   │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─ Team Activity ─────────────┐  ┌─ Quick Actions ─────────────────┐   │
│  │                             │  │                                 │   │
│  │  [Activity Item]            │  │  [Import Sources]               │   │
│  │  [Activity Item]            │  │  [Browse Public Streams]        │   │
│  │  [Activity Item]            │  │  [View Tutorial]                │   │
│  │                             │  │                                 │   │
│  └─────────────────────────────┘  └─────────────────────────────────┘   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Stream Detail (Three-Panel):**
```
┌──────────────────────────────────────────────────────────────────────────┐
│ [Logo]  ← Back  │ CRISPR Delivery Mechanisms        [Share] [⚙] [Avatar]│
├─────────────────┼─────────────────────────────────────┼──────────────────┤
│                 │                                     │                  │
│  Sources (24)   │     [Synthesis Canvas]              │  Insights (8)   │
│  ┌───────────┐  │                                     │  ┌────────────┐  │
│  │ Source    │  │     The field of CRISPR delivery   │  │ 💡 Insight │  │
│  │ Card      │  │     has evolved significantly...    │  │    Card    │  │
│  └───────────┘  │                                     │  └────────────┘  │
│  ┌───────────┐  │     [Full synthesis content here]   │  ┌────────────┐  │
│  │ Source    │  │                                     │  │ 💡 Insight │  │
│  │ Card      │  │                                     │  │    Card    │  │
│  └───────────┘  │                                     │  └────────────┘  │
│                 │                                     │                  │
│  [+ Add Source] │  [Regenerate] [Edit] [Export]      │  [Find More]     │
│                 │                                     │                  │
├─────────────────┴─────────────────────────────────────┴──────────────────┤
│  View: [Synthesis] [Graph] [Timeline]                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 12.2 Component Templates

**Source Card:**
```
┌────────────────────────────────────────────────────────────┐
│ ┌──────┐                                                   │
│ │      │  Title of the Paper or Document                   │
│ │ PDF  │  Authors · Publication · Year                     │
│ │ icon │  ○○○ 3 entities · ⏱ Added 2 days ago             │
│ └──────┘                                                   │
│                                                            │
│ [Open] [Add to Stream ▼] [···]                            │
└────────────────────────────────────────────────────────────┘
```

**Insight Card:**
```
┌────────────────────────────────────────────────────────────┐
│ 💡 Connection Found                              ●●●●○    │
├────────────────────────────────────────────────────────────┤
│                                                            │
│ Brief insight statement that fits in two to three lines    │
│ and conveys the key finding clearly.                       │
│                                                            │
│ Evidence: [Source 1] [Source 2] [Source 3]                │
│ Entities: entity1, entity2, entity3                        │
│                                                            │
├────────────────────────────────────────────────────────────┤
│ [Save]  [Explore]  [Dismiss]                              │
└────────────────────────────────────────────────────────────┘
```

---

*This specification should be treated as a living document. Update it as user research reveals new needs, as technical constraints emerge, and as the product vision evolves. The best UX comes from iteration grounded in real user behavior.*
