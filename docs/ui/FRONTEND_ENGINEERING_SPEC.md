# LatticeForge UI: Frontend Engineering Specification

**Audience:** Frontend Engineers, Full-Stack Engineers
**Prerequisites:** Rust fundamentals, reactive programming concepts, WebAssembly basics
**Related Docs:** LATTICEFORGE_2035_ARCHITECTURE.md, API Contract Spec (backend)
**Last Updated:** 2024-12

---

## 1. Overview and Technical Context

This document specifies the frontend architecture for LatticeForge. You're building a real-time intelligence dashboard that processes streaming signal data, renders interactive visualizations, and orchestrates LLM inference with live feedback.

The stack is unconventional: Rust compiled to WebAssembly via Leptos. If you're coming from React/Vue/Svelte, the mental model is similar (reactive, component-based) but the implementation differs significantly. This doc assumes you've done the Leptos tutorial and can write basic components.

### 1.1 Why This Stack

We covered this in the architecture doc, but the short version:

1. **Same language as backend** — Types are shared. No JSON schema drift. Refactoring touches both ends atomically.

2. **WASM performance** — Some components do client-side signal processing (entropy calculation, rolling statistics). JavaScript can't keep up at 60fps with 50+ concurrent streams.

3. **Memory control** — No garbage collection pauses. Predictable latency for animations and real-time updates.

4. **Type safety** — Rust's type system catches bugs at compile time that would be runtime errors in TypeScript.

The tradeoff: smaller ecosystem, steeper learning curve, longer compile times. We accept these for the benefits above.

### 1.2 What You're Building

The UI has four major surfaces:

1. **Dashboard** — Real-time signal grid with phase transition alerts, causal graphs, regime indicators
2. **Intelligence Briefings** — LLM-generated summaries with source fusion, confidence intervals, next-move recommendations
3. **Analysis Workspace** — Interactive tools for deep-dive investigation, scenario modeling, what-if analysis
4. **Admin Panel** — User management, org settings, API key management, usage analytics

Each surface has distinct performance characteristics and state management needs. This doc covers all four but emphasizes Dashboard (highest complexity) and Briefings (most novel).

---

## 2. Project Structure

```
packages/web-leptos/
├── Cargo.toml
├── src/
│   ├── main.rs              # Entry point, router setup
│   ├── app.rs               # Root <App/> component
│   ├── lib.rs               # WASM entry when compiled as library
│   │
│   ├── components/          # Reusable UI components
│   │   ├── mod.rs
│   │   ├── glass_card.rs    # Glassmorphism card container
│   │   ├── glass_button.rs  # Styled button variants
│   │   ├── signal_spark.rs  # Inline sparkline for signals
│   │   ├── entropy_gauge.rs # Radial gauge for entropy display
│   │   ├── risk_badge.rs    # Color-coded risk indicator
│   │   ├── causal_graph.rs  # D3-style force-directed graph
│   │   ├── phase_dial.rs    # Circular phase indicator
│   │   └── ...
│   │
│   ├── pages/               # Route-level components
│   │   ├── mod.rs
│   │   ├── dashboard.rs
│   │   ├── briefings.rs
│   │   ├── workspace.rs
│   │   ├── admin/
│   │   │   ├── mod.rs
│   │   │   ├── users.rs
│   │   │   ├── settings.rs
│   │   │   └── usage.rs
│   │   └── auth/
│   │       ├── login.rs
│   │       └── callback.rs
│   │
│   ├── state/               # Global state management
│   │   ├── mod.rs
│   │   ├── auth.rs          # User session, tokens
│   │   ├── signals.rs       # Real-time signal streams
│   │   ├── alerts.rs        # Active alerts, notifications
│   │   ├── preferences.rs   # User preferences, theme
│   │   └── websocket.rs     # WebSocket connection manager
│   │
│   ├── api/                 # Backend API clients
│   │   ├── mod.rs
│   │   ├── rest.rs          # REST endpoints
│   │   ├── ws.rs            # WebSocket message handling
│   │   └── types.rs         # Shared API types (from proto)
│   │
│   ├── compute/             # Client-side computation (WASM-heavy)
│   │   ├── mod.rs
│   │   ├── entropy.rs       # Token entropy calculation
│   │   ├── rolling_stats.rs # Rolling mean, variance, etc.
│   │   ├── interpolation.rs # Signal interpolation for smooth rendering
│   │   └── color_maps.rs    # Numeric-to-color mapping
│   │
│   └── utils/               # Utilities
│       ├── mod.rs
│       ├── format.rs        # Number/date formatting
│       ├── debounce.rs      # Input debouncing
│       └── storage.rs       # LocalStorage wrappers
│
├── style/
│   ├── main.css             # Global styles
│   ├── variables.css        # CSS custom properties
│   └── components/          # Component-specific styles
│
├── public/
│   ├── index.html           # Shell HTML
│   └── assets/              # Static assets
│
└── tests/
    ├── integration/         # Browser-based tests
    └── unit/                # Rust unit tests
```

### 2.1 Module Boundaries

The separation between `components/`, `pages/`, `state/`, and `compute/` is load-bearing.

**Components** are pure UI. They receive props, emit events, render DOM. No direct API calls. No global state mutations. If a component needs data, it gets it via props or context.

**Pages** are route handlers. They wire up state, make API calls, pass data to components. A page can be "smart" — it knows about global state and side effects.

**State** modules manage reactive signals that persist across page navigations. Auth state, active WebSocket connections, cached data. These use Leptos's `create_signal` and `provide_context` for dependency injection.

**Compute** modules are pure functions that run expensive calculations. They're designed to be called from `create_memo` or `create_effect` without blocking the render loop. Some are marked `#[wasm_bindgen]` for direct JS interop if needed.

Don't blur these boundaries. A component that makes API calls is a bug. A state module that renders DOM is a bug.

---

## 3. Reactive Primitives and State Management

Leptos uses fine-grained reactivity, similar to SolidJS. If you're from React, forget `useState` — this is closer to MobX or Svelte stores.

### 3.1 Core Primitives

```rust
// Signal: reactive value
let (count, set_count) = create_signal(0);

// Reading (tracks dependency)
let value = count.get();

// Writing (triggers updates)
set_count.set(value + 1);

// Derived signal (memoized)
let doubled = create_memo(move |_| count.get() * 2);

// Effect (side effects on change)
create_effect(move |_| {
    log::info!("Count changed to: {}", count.get());
});
```

Key insight: `count.get()` inside a reactive context (component body, memo, effect) creates a subscription. When `count` changes, only the specific subscribers re-run — not the whole component tree.

### 3.2 Global State Pattern

We use context for global state. Each state module exposes a struct and a provider.

```rust
// state/auth.rs
#[derive(Clone)]
pub struct AuthState {
    pub user: ReadSignal<Option<User>>,
    pub set_user: WriteSignal<Option<User>>,
    pub is_authenticated: Memo<bool>,
    pub role: Memo<UserRole>,
}

impl AuthState {
    pub fn new() -> Self {
        let (user, set_user) = create_signal(None::<User>);
        let is_authenticated = create_memo(move |_| user.get().is_some());
        let role = create_memo(move |_| {
            user.get().map(|u| u.role).unwrap_or(UserRole::Anonymous)
        });

        Self { user, set_user, is_authenticated, role }
    }
}

pub fn provide_auth_state() {
    provide_context(AuthState::new());
}

pub fn use_auth() -> AuthState {
    expect_context::<AuthState>()
}
```

Usage in components:

```rust
#[component]
fn UserMenu() -> impl IntoView {
    let auth = use_auth();

    view! {
        <Show when=move || auth.is_authenticated.get()>
            <div class="user-menu">
                {move || auth.user.get().map(|u| u.email)}
            </div>
        </Show>
    }
}
```

### 3.3 Signal Streams (Real-Time Data)

The dashboard receives streaming data via WebSocket. We model this as a reactive signal that updates on each message.

```rust
// state/signals.rs
pub struct SignalStreamState {
    // Map of signal_id -> latest data point
    signals: RwSignal<HashMap<String, SignalData>>,
    // Connection status
    status: RwSignal<ConnectionStatus>,
    // Error channel
    last_error: RwSignal<Option<String>>,
}

impl SignalStreamState {
    pub fn new() -> Self {
        Self {
            signals: create_rw_signal(HashMap::new()),
            status: create_rw_signal(ConnectionStatus::Disconnected),
            last_error: create_rw_signal(None),
        }
    }

    pub fn get_signal(&self, id: &str) -> Option<SignalData> {
        self.signals.get().get(id).cloned()
    }

    pub fn update_signal(&self, id: String, data: SignalData) {
        self.signals.update(|map| {
            map.insert(id, data);
        });
    }

    // Subscribe to a specific signal - returns a derived memo
    pub fn subscribe(&self, id: String) -> Memo<Option<SignalData>> {
        let signals = self.signals;
        create_memo(move |_| signals.get().get(&id).cloned())
    }
}
```

The WebSocket handler runs in a separate effect and pushes updates:

```rust
// In dashboard page setup
create_effect(move |_| {
    let stream_state = use_signal_stream();

    spawn_local(async move {
        let ws = connect_websocket("/api/ws/signals").await;
        stream_state.status.set(ConnectionStatus::Connected);

        while let Some(msg) = ws.next().await {
            match msg {
                WsMessage::SignalUpdate { id, data } => {
                    stream_state.update_signal(id, data);
                }
                WsMessage::Alert(alert) => {
                    // Push to alerts state
                }
                WsMessage::Error(e) => {
                    stream_state.last_error.set(Some(e));
                }
            }
        }

        stream_state.status.set(ConnectionStatus::Disconnected);
    });
});
```

### 3.4 Avoiding Re-Render Storms

Common mistake: putting large objects in signals and triggering updates on every field change.

Bad:
```rust
let (config, set_config) = create_signal(LargeConfig::default());

// Every field change re-renders all subscribers
set_config.update(|c| c.field_a = new_value);
```

Better: Split into granular signals or use `RwSignal` with targeted updates.

```rust
let config = create_rw_signal(LargeConfig::default());

// Memo for specific fields - only re-renders when that field changes
let field_a = create_memo(move |_| config.get().field_a);
```

For signal streams with 50+ concurrent updates per second, we batch updates:

```rust
// Collect updates for 16ms (one frame), then flush
let pending_updates = create_rw_signal(Vec::new());

create_effect(move |_| {
    let updates = pending_updates.get();
    if updates.is_empty() { return; }

    // Batch apply
    signal_state.signals.update(|map| {
        for (id, data) in updates {
            map.insert(id, data);
        }
    });

    pending_updates.set(Vec::new());
});

// On WS message, push to pending instead of immediate update
pending_updates.update(|v| v.push((id, data)));
```

---

## 4. Component Architecture

### 4.1 Component Conventions

Every component follows this structure:

```rust
use leptos::*;

/// Brief description of what this component does.
///
/// # Props
/// - `value`: The current value to display
/// - `on_change`: Called when user modifies value
///
/// # Example
/// ```rust
/// <MyComponent value=signal.get() on_change=|v| set_signal.set(v) />
/// ```
#[component]
pub fn MyComponent(
    /// The current value
    value: f64,
    /// Change handler
    #[prop(into)]
    on_change: Callback<f64>,
    /// Optional CSS class override
    #[prop(optional)]
    class: Option<String>,
) -> impl IntoView {
    // Local state (if any)
    let (local, set_local) = create_signal(value);

    // Effects (if any)
    create_effect(move |_| {
        // Sync local with prop when prop changes
        set_local.set(value);
    });

    // Event handlers
    let handle_input = move |ev: web_sys::Event| {
        let new_value = event_target_value(&ev).parse().unwrap_or(0.0);
        on_change.call(new_value);
    };

    // Render
    view! {
        <div class=format!("my-component {}", class.unwrap_or_default())>
            <input
                type="number"
                prop:value=move || local.get()
                on:input=handle_input
            />
        </div>
    }
}
```

Notes:
- Props are typed, documented, and have sensible defaults
- `#[prop(into)]` allows flexible callback types
- `#[prop(optional)]` for optional props
- Event handlers are defined before the view macro
- CSS classes support override/extension pattern

### 4.2 GlassCard Component

The primary container component. Implements the glassmorphism aesthetic.

```rust
#[derive(Clone, Copy, PartialEq)]
pub enum GlassBlur {
    Light,   // backdrop-blur-sm
    Medium,  // backdrop-blur-md
    Heavy,   // backdrop-blur-lg
}

#[component]
pub fn GlassCard(
    children: Children,
    #[prop(default = GlassBlur::Medium)]
    blur: GlassBlur,
    #[prop(optional)]
    class: Option<String>,
    #[prop(optional)]
    border_accent: Option<String>, // "cyan", "red", "amber", etc.
) -> impl IntoView {
    let blur_class = match blur {
        GlassBlur::Light => "backdrop-blur-sm",
        GlassBlur::Medium => "backdrop-blur-md",
        GlassBlur::Heavy => "backdrop-blur-lg",
    };

    let border_class = border_accent
        .map(|c| format!("border-l-4 border-{}-500", c))
        .unwrap_or_default();

    view! {
        <div class=format!(
            "bg-slate-900/70 {} border border-white/10 rounded-xl {} {}",
            blur_class,
            border_class,
            class.unwrap_or_default()
        )>
            {children()}
        </div>
    }
}
```

### 4.3 SignalSparkline Component

Renders a mini chart for signal data. Uses canvas for performance.

```rust
#[component]
pub fn SignalSparkline(
    /// Signal data points (last N values)
    data: Memo<Vec<f64>>,
    /// Width in pixels
    #[prop(default = 120)]
    width: u32,
    /// Height in pixels
    #[prop(default = 32)]
    height: u32,
    /// Line color
    #[prop(default = "#06b6d4".to_string())]
    color: String,
) -> impl IntoView {
    let canvas_ref = create_node_ref::<html::Canvas>();

    // Redraw on data change
    create_effect(move |_| {
        let Some(canvas) = canvas_ref.get() else { return };
        let points = data.get();

        draw_sparkline(&canvas, &points, &color, width, height);
    });

    view! {
        <canvas
            node_ref=canvas_ref
            width=width
            height=height
            class="signal-sparkline"
        />
    }
}

fn draw_sparkline(
    canvas: &HtmlCanvasElement,
    points: &[f64],
    color: &str,
    width: u32,
    height: u32,
) {
    let ctx = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<CanvasRenderingContext2d>()
        .unwrap();

    ctx.clear_rect(0.0, 0.0, width as f64, height as f64);

    if points.is_empty() { return; }

    let min = points.iter().copied().fold(f64::INFINITY, f64::min);
    let max = points.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(0.001);

    let x_step = width as f64 / (points.len() - 1).max(1) as f64;

    ctx.begin_path();
    ctx.set_stroke_style(&JsValue::from_str(color));
    ctx.set_line_width(1.5);

    for (i, &value) in points.iter().enumerate() {
        let x = i as f64 * x_step;
        let y = height as f64 - ((value - min) / range * height as f64);

        if i == 0 {
            ctx.move_to(x, y);
        } else {
            ctx.line_to(x, y);
        }
    }

    ctx.stroke();
}
```

### 4.4 CausalGraph Component

Force-directed graph visualization. This is complex — uses Web Workers for physics simulation to avoid blocking the main thread.

```rust
#[component]
pub fn CausalGraph(
    /// Nodes with positions
    nodes: Memo<Vec<GraphNode>>,
    /// Edges with weights
    edges: Memo<Vec<GraphEdge>>,
    /// Width
    #[prop(default = 600)]
    width: u32,
    /// Height
    #[prop(default = 400)]
    height: u32,
    /// Callback when node is clicked
    #[prop(optional, into)]
    on_node_click: Option<Callback<String>>,
) -> impl IntoView {
    let canvas_ref = create_node_ref::<html::Canvas>();
    let (simulation_state, set_simulation_state) = create_signal(SimulationState::default());

    // Initialize force simulation in worker
    create_effect(move |_| {
        let nodes = nodes.get();
        let edges = edges.get();

        spawn_local(async move {
            let worker = create_physics_worker();
            worker.post_message(&SimulationConfig { nodes, edges, width, height });

            // Receive position updates
            let mut receiver = worker.subscribe();
            while let Some(positions) = receiver.next().await {
                set_simulation_state.set(positions);
            }
        });
    });

    // Render loop
    create_effect(move |_| {
        let Some(canvas) = canvas_ref.get() else { return };
        let state = simulation_state.get();

        render_graph(&canvas, &state, width, height);
    });

    // Click handling
    let handle_click = move |ev: web_sys::MouseEvent| {
        let Some(callback) = on_node_click else { return };
        let state = simulation_state.get();

        let rect = canvas_ref.get().unwrap().get_bounding_client_rect();
        let x = ev.client_x() as f64 - rect.left();
        let y = ev.client_y() as f64 - rect.top();

        // Hit test nodes
        for node in &state.nodes {
            let dx = node.x - x;
            let dy = node.y - y;
            if (dx * dx + dy * dy).sqrt() < node.radius {
                callback.call(node.id.clone());
                return;
            }
        }
    };

    view! {
        <canvas
            node_ref=canvas_ref
            width=width
            height=height
            class="causal-graph cursor-pointer"
            on:click=handle_click
        />
    }
}
```

### 4.5 EntropyGauge Component

Radial gauge showing current entropy with micro-grokking indicator.

```rust
#[component]
pub fn EntropyGauge(
    /// Current entropy value (0.0 - 3.0 typical range)
    entropy: Memo<f64>,
    /// Whether micro-grokking was detected
    grokking_detected: Memo<bool>,
    /// Size in pixels
    #[prop(default = 80)]
    size: u32,
) -> impl IntoView {
    let normalized = create_memo(move |_| {
        (entropy.get() / 3.0).clamp(0.0, 1.0)
    });

    let color = create_memo(move |_| {
        let v = normalized.get();
        if v < 0.3 { "#10b981" }      // green - low entropy, confident
        else if v < 0.6 { "#f59e0b" } // amber - medium
        else { "#ef4444" }            // red - high entropy, confused
    });

    // SVG arc calculation
    let arc_path = create_memo(move |_| {
        let angle = normalized.get() * std::f64::consts::PI * 1.5; // 270 degree max
        let r = (size / 2 - 4) as f64;
        let cx = (size / 2) as f64;
        let cy = (size / 2) as f64;

        let start_angle = -std::f64::consts::PI * 0.75;
        let end_angle = start_angle + angle;

        let x1 = cx + r * start_angle.cos();
        let y1 = cy + r * start_angle.sin();
        let x2 = cx + r * end_angle.cos();
        let y2 = cy + r * end_angle.sin();

        let large_arc = if angle > std::f64::consts::PI { 1 } else { 0 };

        format!("M {x1} {y1} A {r} {r} 0 {large_arc} 1 {x2} {y2}")
    });

    view! {
        <div class="entropy-gauge relative" style=format!("width: {}px; height: {}px", size, size)>
            <svg viewBox=format!("0 0 {} {}", size, size) class="absolute inset-0">
                // Background arc
                <circle
                    cx=size/2
                    cy=size/2
                    r=size/2-4
                    fill="none"
                    stroke="rgba(255,255,255,0.1)"
                    stroke-width="4"
                />
                // Value arc
                <path
                    d=move || arc_path.get()
                    fill="none"
                    stroke=move || color.get()
                    stroke-width="4"
                    stroke-linecap="round"
                />
            </svg>

            // Center value
            <div class="absolute inset-0 flex items-center justify-center flex-col">
                <span class="text-lg font-mono" style=format!("color: {}", color.get())>
                    {move || format!("{:.2}", entropy.get())}
                </span>
                <Show when=move || grokking_detected.get()>
                    <span class="text-xs text-emerald-400 animate-pulse">"GROK"</span>
                </Show>
            </div>
        </div>
    }
}
```

---

## 5. Page Implementations

### 5.1 Dashboard Page

The dashboard is the most complex page. It displays:
- Grid of signal cards (configurable layout)
- Real-time causal graph
- Alert feed
- Phase transition timeline
- Regime status panel

```rust
#[component]
pub fn DashboardPage() -> impl IntoView {
    // Initialize WebSocket connection
    let signal_stream = use_signal_stream();
    let alerts = use_alerts();

    // Layout configuration
    let (layout, set_layout) = create_signal(DashboardLayout::default());

    // Selected signal for detail view
    let (selected_signal, set_selected_signal) = create_signal(None::<String>);

    // Fetch initial data
    create_effect(move |_| {
        spawn_local(async move {
            let config = fetch_dashboard_config().await;
            set_layout.set(config.layout);
        });
    });

    // Subscribe to signals based on layout
    let active_signals = create_memo(move |_| {
        layout.get().signal_ids.iter()
            .filter_map(|id| signal_stream.get_signal(id))
            .collect::<Vec<_>>()
    });

    // Causal graph data
    let graph_nodes = create_memo(move |_| {
        active_signals.get().iter().map(|s| GraphNode {
            id: s.id.clone(),
            label: s.name.clone(),
            value: s.latest_value,
            risk: s.risk_level,
        }).collect()
    });

    let graph_edges = create_memo(move |_| {
        // Derived from transfer entropy matrix
        signal_stream.get_causal_edges()
    });

    view! {
        <div class="dashboard-page h-full flex flex-col gap-4 p-4">
            // Header bar
            <div class="flex items-center justify-between">
                <h1 class="text-xl font-semibold text-white">"Signal Dashboard"</h1>
                <div class="flex items-center gap-2">
                    <ConnectionStatus status=signal_stream.status />
                    <LayoutSelector value=layout on_change=set_layout />
                </div>
            </div>

            // Main content grid
            <div class="flex-1 grid grid-cols-12 gap-4">
                // Signal cards (left 8 cols)
                <div class="col-span-8 grid grid-cols-3 gap-3 auto-rows-min">
                    <For
                        each=move || active_signals.get()
                        key=|s| s.id.clone()
                        children=move |signal| {
                            let id = signal.id.clone();
                            view! {
                                <SignalCard
                                    signal=signal.clone()
                                    selected=move || selected_signal.get() == Some(id.clone())
                                    on_click=move |_| set_selected_signal.set(Some(id.clone()))
                                />
                            }
                        }
                    />
                </div>

                // Right sidebar (4 cols)
                <div class="col-span-4 flex flex-col gap-4">
                    // Causal graph
                    <GlassCard class="p-4">
                        <h3 class="text-sm font-medium text-slate-400 mb-2">"Causal Structure"</h3>
                        <CausalGraph
                            nodes=graph_nodes
                            edges=graph_edges
                            on_node_click=move |id| set_selected_signal.set(Some(id))
                        />
                    </GlassCard>

                    // Alert feed
                    <GlassCard class="p-4 flex-1 overflow-hidden">
                        <h3 class="text-sm font-medium text-slate-400 mb-2">"Active Alerts"</h3>
                        <AlertFeed alerts=alerts.active />
                    </GlassCard>
                </div>
            </div>

            // Detail panel (slides up when signal selected)
            <Show when=move || selected_signal.get().is_some()>
                <SignalDetailPanel
                    signal_id=selected_signal.get().unwrap()
                    on_close=move || set_selected_signal.set(None)
                />
            </Show>
        </div>
    }
}
```

### 5.2 Briefings Page

Intelligence briefings with domain categories, executive summary, and next strategic move recommendations.

```rust
#[component]
pub fn BriefingsPage() -> impl IntoView {
    let (selected_preset, set_preset) = create_signal("global".to_string());
    let (briefing, set_briefing) = create_signal(None::<Briefing>);
    let (loading, set_loading) = create_signal(false);
    let (expanded_category, set_expanded_category) = create_signal(Some("Political & Security".to_string()));

    let load_briefing = move |_| {
        let preset = selected_preset.get();
        set_loading.set(true);

        spawn_local(async move {
            match fetch_briefing(&preset).await {
                Ok(b) => set_briefing.set(Some(b)),
                Err(e) => log::error!("Failed to load briefing: {}", e),
            }
            set_loading.set(false);
        });
    };

    view! {
        <div class="briefings-page h-full flex flex-col gap-6 p-6">
            // Header
            <div class="flex items-start justify-between border-b border-white/10 pb-4">
                <div class="flex items-center gap-3">
                    <div class="p-2 bg-cyan-500/20 rounded-lg">
                        <RadioIcon class="w-5 h-5 text-cyan-400" />
                    </div>
                    <div>
                        <h1 class="text-2xl font-bold text-white">"Intelligence Briefing"</h1>
                        <p class="text-slate-500 text-sm">
                            "Multi-source fusion • Real-time analysis"
                        </p>
                    </div>
                </div>
                <GlassButton on_click=|_| {}>"Glossary"</GlassButton>
            </div>

            // Preset tabs
            <PresetSelector
                selected=selected_preset
                on_select=move |p| {
                    set_preset.set(p);
                    set_briefing.set(None);
                }
            />

            // Content
            <Show
                when=move || briefing.get().is_some()
                fallback=move || view! {
                    <LoadBriefingPrompt loading=loading on_load=load_briefing />
                }
            >
                {move || {
                    let b = briefing.get().unwrap();
                    view! {
                        <div class="flex flex-col gap-6">
                            // Executive Summary
                            <GlassCard blur=GlassBlur::Heavy border_accent=Some("cyan".to_string()) class="p-6">
                                <div class="flex items-start justify-between mb-4">
                                    <div>
                                        <span class="text-xs font-medium text-cyan-400 uppercase tracking-wider">
                                            "Executive Summary"
                                        </span>
                                        <h2 class="text-xl font-semibold text-white mt-1">
                                            {format!("{}: Situation Assessment", selected_preset.get())}
                                        </h2>
                                    </div>
                                    <RiskBadge level=b.overall_risk />
                                </div>
                                <p class="text-slate-300 leading-relaxed">{b.summary}</p>
                            </GlassCard>

                            // Domain categories
                            <DomainAccordion
                                domains=b.domains
                                expanded=expanded_category
                                on_toggle=move |cat| {
                                    if expanded_category.get() == Some(cat.clone()) {
                                        set_expanded_category.set(None);
                                    } else {
                                        set_expanded_category.set(Some(cat));
                                    }
                                }
                            />

                            // Next Strategic Move
                            <Show when=move || b.nsm.is_some()>
                                <GlassCard
                                    blur=GlassBlur::Heavy
                                    class="p-6 border-2 border-cyan-500/30 bg-gradient-to-r from-cyan-500/10 to-blue-500/10"
                                >
                                    <div class="flex items-start gap-4">
                                        <div class="p-3 bg-cyan-500/20 rounded-xl">
                                            <TargetIcon class="w-6 h-6 text-cyan-400" />
                                        </div>
                                        <div>
                                            <h3 class="text-lg font-semibold text-cyan-300">
                                                "Next Strategic Move"
                                            </h3>
                                            <p class="text-slate-300 mt-2 whitespace-pre-line">
                                                {b.nsm.clone().unwrap()}
                                            </p>
                                        </div>
                                    </div>
                                </GlassCard>
                            </Show>
                        </div>
                    }
                }}
            </Show>
        </div>
    }
}
```

---

## 6. WebSocket Integration

### 6.1 Connection Management

```rust
// api/ws.rs
pub struct WebSocketManager {
    connection: RwSignal<Option<WebSocket>>,
    status: RwSignal<ConnectionStatus>,
    reconnect_attempts: RwSignal<u32>,
    message_handlers: RwSignal<Vec<Box<dyn Fn(WsMessage)>>>,
}

impl WebSocketManager {
    const MAX_RECONNECT_ATTEMPTS: u32 = 5;
    const RECONNECT_DELAY_MS: [u32; 5] = [1000, 2000, 4000, 8000, 16000];

    pub fn new() -> Self {
        Self {
            connection: create_rw_signal(None),
            status: create_rw_signal(ConnectionStatus::Disconnected),
            reconnect_attempts: create_rw_signal(0),
            message_handlers: create_rw_signal(Vec::new()),
        }
    }

    pub async fn connect(&self, url: &str) -> Result<(), JsValue> {
        self.status.set(ConnectionStatus::Connecting);

        let ws = WebSocket::new(url)?;

        // Set up handlers
        let status = self.status;
        let handlers = self.message_handlers;
        let reconnect_attempts = self.reconnect_attempts;
        let self_clone = self.clone();
        let url_owned = url.to_string();

        // On open
        let on_open = Closure::wrap(Box::new(move || {
            status.set(ConnectionStatus::Connected);
            reconnect_attempts.set(0);
        }) as Box<dyn Fn()>);
        ws.set_onopen(Some(on_open.as_ref().unchecked_ref()));
        on_open.forget();

        // On message
        let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
            if let Ok(text) = event.data().dyn_into::<js_sys::JsString>() {
                let text: String = text.into();
                if let Ok(msg) = serde_json::from_str::<WsMessage>(&text) {
                    for handler in handlers.get().iter() {
                        handler(msg.clone());
                    }
                }
            }
        }) as Box<dyn Fn(MessageEvent)>);
        ws.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        on_message.forget();

        // On close - attempt reconnect
        let on_close = Closure::wrap(Box::new(move || {
            status.set(ConnectionStatus::Disconnected);

            let attempts = reconnect_attempts.get();
            if attempts < Self::MAX_RECONNECT_ATTEMPTS {
                let delay = Self::RECONNECT_DELAY_MS[attempts as usize];
                reconnect_attempts.set(attempts + 1);

                let self_inner = self_clone.clone();
                let url_inner = url_owned.clone();
                spawn_local(async move {
                    gloo_timers::future::TimeoutFuture::new(delay).await;
                    let _ = self_inner.connect(&url_inner).await;
                });
            } else {
                status.set(ConnectionStatus::Failed);
            }
        }) as Box<dyn Fn()>);
        ws.set_onclose(Some(on_close.as_ref().unchecked_ref()));
        on_close.forget();

        self.connection.set(Some(ws));
        Ok(())
    }

    pub fn send(&self, msg: &WsMessage) {
        if let Some(ws) = self.connection.get() {
            if ws.ready_state() == WebSocket::OPEN {
                let json = serde_json::to_string(msg).unwrap();
                let _ = ws.send_with_str(&json);
            }
        }
    }

    pub fn on_message(&self, handler: impl Fn(WsMessage) + 'static) {
        self.message_handlers.update(|h| h.push(Box::new(handler)));
    }
}
```

### 6.2 Message Types

```rust
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum WsMessage {
    // Server -> Client
    SignalUpdate { id: String, data: SignalData },
    Alert(Alert),
    CausalGraphUpdate(CausalGraph),
    RegimeChange { signal_id: String, from: String, to: String },

    // Client -> Server
    Subscribe { signal_ids: Vec<String> },
    Unsubscribe { signal_ids: Vec<String> },
    SetAlertThreshold { signal_id: String, threshold: f64 },
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SignalData {
    pub id: String,
    pub name: String,
    pub latest_value: f64,
    pub timestamp: i64,
    pub history: Vec<f64>,  // Last N values
    pub risk_level: RiskLevel,
    pub regime: String,
    pub entropy: f64,
    pub trend: Trend,
}
```

---

## 7. Client-Side Computation

### 7.1 Entropy Calculation

Run in WASM for performance. Called on every token during LLM streaming.

```rust
// compute/entropy.rs
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn compute_token_entropy(logits: &[f32], temperature: f32) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }

    // Temperature-scaled softmax
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scaled: Vec<f32> = logits.iter()
        .map(|&l| ((l - max_logit) / temperature).exp())
        .collect();

    let sum: f32 = scaled.iter().sum();
    let probs: Vec<f32> = scaled.iter().map(|&s| s / sum).collect();

    // Shannon entropy
    -probs.iter()
        .filter(|&&p| p > 1e-10)
        .map(|&p| p * p.ln())
        .sum::<f32>()
}

#[wasm_bindgen]
pub fn detect_grokking(entropy_trace: &[f32], window_size: usize) -> Option<usize> {
    if entropy_trace.len() < window_size * 3 {
        return None;
    }

    // Compute second derivative
    let smoothed = smooth_signal(entropy_trace, window_size);
    let d1 = derivative(&smoothed);
    let d2 = derivative(&d1);

    // Find minimum (sharpest downward acceleration)
    let (min_idx, min_val) = d2.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?;

    if *min_val < -0.05 {
        Some(min_idx + window_size)  // Adjust for derivative offset
    } else {
        None
    }
}

fn smooth_signal(signal: &[f32], window: usize) -> Vec<f32> {
    signal.windows(window)
        .map(|w| w.iter().sum::<f32>() / window as f32)
        .collect()
}

fn derivative(signal: &[f32]) -> Vec<f32> {
    signal.windows(2)
        .map(|w| w[1] - w[0])
        .collect()
}
```

### 7.2 Rolling Statistics

```rust
// compute/rolling_stats.rs
#[wasm_bindgen]
pub struct RollingStats {
    window: Vec<f64>,
    capacity: usize,
    sum: f64,
    sum_sq: f64,
}

#[wasm_bindgen]
impl RollingStats {
    #[wasm_bindgen(constructor)]
    pub fn new(capacity: usize) -> Self {
        Self {
            window: Vec::with_capacity(capacity),
            capacity,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    pub fn push(&mut self, value: f64) {
        if self.window.len() >= self.capacity {
            let old = self.window.remove(0);
            self.sum -= old;
            self.sum_sq -= old * old;
        }

        self.window.push(value);
        self.sum += value;
        self.sum_sq += value * value;
    }

    pub fn mean(&self) -> f64 {
        if self.window.is_empty() {
            0.0
        } else {
            self.sum / self.window.len() as f64
        }
    }

    pub fn variance(&self) -> f64 {
        let n = self.window.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean = self.mean();
        (self.sum_sq / n) - (mean * mean)
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}
```

---

## 8. Performance Optimization

### 8.1 Render Batching

For high-frequency updates (signal streams), we batch renders to hit 60fps max.

```rust
pub struct RenderBatcher {
    pending: RwSignal<Vec<PendingUpdate>>,
    frame_scheduled: Cell<bool>,
}

impl RenderBatcher {
    pub fn queue_update(&self, update: PendingUpdate) {
        self.pending.update(|v| v.push(update));

        if !self.frame_scheduled.get() {
            self.frame_scheduled.set(true);

            request_animation_frame(move || {
                self.flush();
                self.frame_scheduled.set(false);
            });
        }
    }

    fn flush(&self) {
        let updates = self.pending.get();
        self.pending.set(Vec::new());

        // Apply all updates in single batch
        for update in updates {
            update.apply();
        }
    }
}
```

### 8.2 Virtualized Lists

For long lists (alert history, signal list), use virtual scrolling.

```rust
#[component]
pub fn VirtualList<T, V>(
    items: Memo<Vec<T>>,
    item_height: f64,
    render_item: impl Fn(T) -> V + Clone + 'static,
) -> impl IntoView
where
    T: Clone + 'static,
    V: IntoView,
{
    let container_ref = create_node_ref::<html::Div>();
    let (scroll_top, set_scroll_top) = create_signal(0.0);
    let (container_height, set_container_height) = create_signal(400.0);

    // Calculate visible range
    let visible_range = create_memo(move |_| {
        let start = (scroll_top.get() / item_height).floor() as usize;
        let visible_count = (container_height.get() / item_height).ceil() as usize + 2;
        let end = (start + visible_count).min(items.get().len());
        start..end
    });

    let total_height = create_memo(move |_| {
        items.get().len() as f64 * item_height
    });

    let visible_items = create_memo(move |_| {
        let range = visible_range.get();
        items.get()[range.clone()].to_vec()
    });

    let offset_y = create_memo(move |_| {
        visible_range.get().start as f64 * item_height
    });

    view! {
        <div
            node_ref=container_ref
            class="virtual-list overflow-auto"
            on:scroll=move |ev| {
                let el = event_target::<HtmlDivElement>(&ev);
                set_scroll_top.set(el.scroll_top());
            }
        >
            <div style=format!("height: {}px; position: relative", total_height.get())>
                <div style=format!("transform: translateY({}px)", offset_y.get())>
                    <For
                        each=move || visible_items.get()
                        key=|item| item.id.clone()
                        children=render_item.clone()
                    />
                </div>
            </div>
        </div>
    }
}
```

### 8.3 Canvas vs SVG Decision Tree

- **< 100 elements, infrequent updates:** SVG (easier styling, accessibility)
- **100-1000 elements, moderate updates:** SVG with `will-change: transform`
- **> 1000 elements OR > 30fps updates:** Canvas
- **> 10000 elements OR complex physics:** Canvas + Web Worker for computation

---

## 9. Testing Strategy

### 9.1 Component Tests

Use `wasm-bindgen-test` for browser-based component testing.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_entropy_gauge_renders() {
        let document = web_sys::window().unwrap().document().unwrap();
        let container = document.create_element("div").unwrap();
        document.body().unwrap().append_child(&container).unwrap();

        let entropy = create_memo(|_| 1.5);
        let grokking = create_memo(|_| false);

        mount_to(container.clone(), || view! {
            <EntropyGauge entropy=entropy grokking_detected=grokking />
        });

        // Assert DOM structure
        assert!(container.query_selector(".entropy-gauge").unwrap().is_some());
    }

    #[wasm_bindgen_test]
    fn test_entropy_gauge_updates_on_signal_change() {
        // ... similar setup

        // Update signal
        set_entropy.set(2.5);

        // Assert color changed to red
        let gauge = container.query_selector(".entropy-gauge span").unwrap().unwrap();
        let color = gauge.get_attribute("style").unwrap();
        assert!(color.contains("ef4444")); // red
    }
}
```

### 9.2 Integration Tests

Playwright for end-to-end testing.

```typescript
// tests/e2e/dashboard.spec.ts
test('dashboard loads and displays signals', async ({ page }) => {
    await page.goto('/dashboard');

    // Wait for WebSocket connection
    await expect(page.locator('[data-testid="connection-status"]'))
        .toHaveText('Connected');

    // Signal cards should appear
    await expect(page.locator('.signal-card')).toHaveCount(greaterThan(0));

    // Click a signal card
    await page.locator('.signal-card').first().click();

    // Detail panel should open
    await expect(page.locator('.signal-detail-panel')).toBeVisible();
});
```

---

## 10. Deployment

### 10.1 Build Pipeline

```bash
# Build WASM
wasm-pack build --target web --out-dir pkg

# Build Leptos app
cargo leptos build --release

# Output structure
dist/
├── index.html
├── pkg/
│   ├── latticeforge_web.js
│   ├── latticeforge_web_bg.wasm
│   └── ...
├── style/
│   └── main.css
└── assets/
```

### 10.2 CDN Configuration

- WASM files: Cache-Control max-age=31536000, immutable
- HTML: Cache-Control no-cache (for instant updates)
- CSS/JS bundles: Hashed filenames, long cache

### 10.3 Feature Flags

```rust
#[cfg(feature = "dev")]
fn dev_tools() -> impl IntoView {
    view! {
        <DevPanel />
        <StateInspector />
    }
}

#[cfg(not(feature = "dev"))]
fn dev_tools() -> impl IntoView {
    view! {}
}
```

---

## Appendix A: Common Patterns Cheat Sheet

```rust
// Conditional rendering
<Show when=move || condition.get()>
    <Content />
</Show>

// List rendering
<For
    each=move || items.get()
    key=|item| item.id
    children=|item| view! { <Item data=item /> }
/>

// Event handling
<button on:click=move |_| do_thing()>"Click"</button>

// Two-way binding
<input
    prop:value=move || value.get()
    on:input=move |ev| set_value.set(event_target_value(&ev))
/>

// Context access
let auth = use_context::<AuthState>().expect("AuthState not provided");

// Async data loading
let data = create_resource(
    move || params.get().id,
    |id| async move { fetch_data(id).await }
);

// Error boundaries
<ErrorBoundary fallback=|errors| view! { <ErrorDisplay errors /> }>
    <ComponentThatMightFail />
</ErrorBoundary>
```

---

*Questions? Ping #frontend-eng in Slack. For Leptos-specific issues, the Discord is responsive.*
