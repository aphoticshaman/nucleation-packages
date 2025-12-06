# LatticeForge UI: Junior Frontend Developer Guide

**Audience:** Junior/Mid-level Frontend Developers, new team members
**Prerequisites:** HTML, CSS, JavaScript basics. Rust experience helpful but not required.
**Time to Onboard:** ~2 weeks to first meaningful PR
**Mentor:** Assigned senior engineer (check your onboarding doc)

---

## Welcome

You're joining the frontend team on a project that looks intimidating. Rust. WebAssembly. Reactive programming. Phase transitions. Causal emergence.

Take a breath. You don't need to understand all of that on day one. This guide breaks down what you actually need to know, in the order you need to know it.

By the end of week one, you'll have:
- Set up your dev environment
- Made a small component change
- Understood the basic data flow
- Submitted your first PR

By the end of week two, you'll have:
- Built a new component from scratch
- Connected it to live data
- Written tests for it
- Understood why we chose this weird stack

Let's go.

---

## Part 1: Environment Setup (Day 1)

### 1.1 Install the Tools

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Add WASM target
rustup target add wasm32-unknown-unknown

# Leptos tooling
cargo install trunk
cargo install leptos_cli

# Node (for some tooling, not main development)
# Use nvm or your preferred method
nvm install 20
nvm use 20

# Clone the repo
git clone <repo-url>
cd nucleation-packages/packages/web-leptos
```

### 1.2 First Build

```bash
# This will take 5-10 minutes the first time
trunk serve

# Open http://localhost:8080
```

If you see the dashboard, you're good. If you see errors, check:
- Rust version: `rustc --version` (should be 1.75+)
- WASM target: `rustup target list | grep wasm32`
- Trunk version: `trunk --version`

Still stuck? Ping #frontend-eng. First-build issues are common.

### 1.3 Editor Setup

We use VS Code with:
- `rust-analyzer` extension (essential)
- `Even Better TOML` (for Cargo.toml)
- `Error Lens` (inline error display)

Your `settings.json` should include:
```json
{
  "rust-analyzer.cargo.target": "wasm32-unknown-unknown",
  "rust-analyzer.checkOnSave.allTargets": false,
  "editor.formatOnSave": true
}
```

The first setting is important — without it, rust-analyzer will complain about web_sys types.

---

## Part 2: Rust Crash Course for Frontend Devs (Days 1-2)

You don't need to be a Rust expert. You need to understand these concepts:

### 2.1 Variables and Ownership

```rust
// Immutable by default
let x = 5;
// x = 6; // ERROR

// Mutable with mut
let mut y = 5;
y = 6; // OK

// Ownership - values have one owner
let s1 = String::from("hello");
let s2 = s1; // s1 is now INVALID
// println!("{}", s1); // ERROR: s1 moved

// Clone to copy
let s3 = s2.clone();
println!("{} {}", s2, s3); // Both valid
```

Why this matters for UI: When you pass data to components, you need to think about whether you're moving it or cloning it.

### 2.2 References and Borrowing

```rust
// Borrowing - temporary access without ownership
fn print_length(s: &String) {
    println!("{}", s.len());
}

let s = String::from("hello");
print_length(&s); // Borrow s
println!("{}", s); // s still valid

// Mutable borrow
fn append_world(s: &mut String) {
    s.push_str(" world");
}

let mut s = String::from("hello");
append_world(&mut s);
println!("{}", s); // "hello world"
```

### 2.3 Structs and Enums

```rust
// Struct - like a JavaScript object but typed
struct User {
    name: String,
    email: String,
    age: u32,
}

let user = User {
    name: "Alice".to_string(),
    email: "alice@example.com".to_string(),
    age: 30,
};

// Enum - like TypeScript union types but better
enum Status {
    Loading,
    Success(String),    // Can hold data
    Error { code: u32, message: String }, // Named fields
}

let status = Status::Success("Data loaded".to_string());

// Pattern matching
match status {
    Status::Loading => println!("Loading..."),
    Status::Success(msg) => println!("Success: {}", msg),
    Status::Error { code, message } => println!("Error {}: {}", code, message),
}
```

### 2.4 Option and Result

No null in Rust. Instead:

```rust
// Option - might have a value
let maybe_name: Option<String> = Some("Alice".to_string());
let no_name: Option<String> = None;

// Unwrap carefully
if let Some(name) = maybe_name {
    println!("Hello, {}", name);
}

// Or use unwrap_or
let name = no_name.unwrap_or("Anonymous".to_string());

// Result - might have an error
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("Division by zero".to_string())
    } else {
        Ok(a / b)
    }
}

match divide(10.0, 2.0) {
    Ok(result) => println!("Result: {}", result),
    Err(e) => println!("Error: {}", e),
}
```

### 2.5 Closures

Like JavaScript arrow functions, but with type inference:

```rust
// Basic closure
let add = |a, b| a + b;
println!("{}", add(2, 3)); // 5

// Closure capturing environment
let multiplier = 3;
let multiply = |x| x * multiplier;
println!("{}", multiply(5)); // 15

// Move closure - takes ownership
let name = String::from("Alice");
let greet = move || println!("Hello, {}", name);
greet();
// name is now invalid here
```

In Leptos, you'll write closures constantly for event handlers and reactive computations:

```rust
let (count, set_count) = create_signal(0);

// This closure captures `count` and `set_count`
let increment = move |_| set_count.set(count.get() + 1);
```

---

## Part 3: Leptos Basics (Days 2-3)

### 3.1 Your First Component

Create a file `src/components/counter.rs`:

```rust
use leptos::*;

#[component]
pub fn Counter() -> impl IntoView {
    // Reactive state
    let (count, set_count) = create_signal(0);

    // Event handler
    let increment = move |_| set_count.set(count.get() + 1);
    let decrement = move |_| set_count.set(count.get() - 1);

    // JSX-like syntax
    view! {
        <div class="counter">
            <button on:click=decrement>"-"</button>
            <span>{count}</span>
            <button on:click=increment>"+"</button>
        </div>
    }
}
```

Register it in `src/components/mod.rs`:
```rust
mod counter;
pub use counter::Counter;
```

Use it in a page:
```rust
use crate::components::Counter;

view! {
    <Counter />
}
```

### 3.2 Props

Components receive data via props:

```rust
#[component]
pub fn Greeting(
    /// The name to greet
    name: String,
    /// Whether to be formal
    #[prop(default = false)]
    formal: bool,
) -> impl IntoView {
    view! {
        <p>
            {if formal { "Good day, " } else { "Hey, " }}
            {name}
            "!"
        </p>
    }
}

// Usage
view! {
    <Greeting name="Alice".to_string() />
    <Greeting name="Bob".to_string() formal=true />
}
```

### 3.3 Reactive Signals

The core of Leptos. Signals are reactive values:

```rust
// Create a signal
let (value, set_value) = create_signal(0);

// Read (subscribes to changes)
let current = value.get();

// Write (triggers updates)
set_value.set(42);

// Update based on current value
set_value.update(|v| *v += 1);
```

When you call `.get()` inside a reactive context (component body, effect, memo), Leptos tracks the dependency. When the signal changes, only that specific usage updates.

This is different from React's "re-render the whole component" model.

### 3.4 Derived Signals (Memos)

Compute values from other signals:

```rust
let (count, set_count) = create_signal(0);

// This recomputes only when count changes
let doubled = create_memo(move |_| count.get() * 2);

view! {
    <p>"Count: " {count} " Doubled: " {doubled}</p>
}
```

### 3.5 Effects

Side effects when signals change:

```rust
let (search, set_search) = create_signal("".to_string());

// Runs every time search changes
create_effect(move |_| {
    let query = search.get();
    log::info!("Search changed to: {}", query);
    // Could trigger API call here
});
```

### 3.6 Conditional Rendering

```rust
// Show/hide
<Show when=move || count.get() > 0>
    <p>"Count is positive"</p>
</Show>

// With fallback
<Show
    when=move || user.get().is_some()
    fallback=|| view! { <p>"Please log in"</p> }
>
    <p>"Welcome back!"</p>
</Show>

// Pattern matching
{move || match status.get() {
    Status::Loading => view! { <Spinner /> }.into_view(),
    Status::Success(data) => view! { <DataView data /> }.into_view(),
    Status::Error(e) => view! { <ErrorView error=e /> }.into_view(),
}}
```

### 3.7 List Rendering

```rust
let (items, set_items) = create_signal(vec!["a", "b", "c"]);

view! {
    <ul>
        <For
            each=move || items.get()
            key=|item| item.clone()
            children=move |item| view! {
                <li>{item}</li>
            }
        />
    </ul>
}
```

The `key` prop is crucial — it helps Leptos efficiently update the DOM when items change.

---

## Part 4: Project Structure Walkthrough (Day 3)

### 4.1 Where Things Live

```
src/
├── main.rs          # Entry point, router
├── app.rs           # Root component
├── components/      # Reusable pieces
├── pages/           # Route handlers
├── state/           # Global state
├── api/             # Backend communication
└── compute/         # Heavy calculations
```

### 4.2 Component vs Page

**Component:** Pure UI. Receives props, emits events. No API calls. No global state mutations.

```rust
// Good component
#[component]
fn UserCard(user: User, on_click: Callback<()>) -> impl IntoView {
    view! {
        <div class="user-card" on:click=move |_| on_click.call(())>
            <img src=user.avatar />
            <h3>{user.name}</h3>
        </div>
    }
}
```

**Page:** Wires up state, makes API calls, passes data to components.

```rust
// Good page
#[component]
fn UsersPage() -> impl IntoView {
    let (users, set_users) = create_signal(Vec::new());
    let (selected, set_selected) = create_signal(None::<User>);

    // API call
    create_effect(move |_| {
        spawn_local(async move {
            let fetched = fetch_users().await;
            set_users.set(fetched);
        });
    });

    view! {
        <div class="users-page">
            <For
                each=move || users.get()
                key=|u| u.id
                children=move |user| view! {
                    <UserCard
                        user=user.clone()
                        on_click=move |_| set_selected.set(Some(user.clone()))
                    />
                }
            />
        </div>
    }
}
```

### 4.3 State Management Pattern

Global state lives in `state/` modules:

```rust
// state/auth.rs
#[derive(Clone)]
pub struct AuthState {
    pub user: ReadSignal<Option<User>>,
    pub set_user: WriteSignal<Option<User>>,
    pub is_authenticated: Memo<bool>,
}

impl AuthState {
    pub fn new() -> Self {
        let (user, set_user) = create_signal(None);
        let is_authenticated = create_memo(move |_| user.get().is_some());
        Self { user, set_user, is_authenticated }
    }
}

// Provide at app root
pub fn provide_auth() {
    provide_context(AuthState::new());
}

// Access anywhere
pub fn use_auth() -> AuthState {
    expect_context::<AuthState>()
}
```

---

## Part 5: Your First Task (Days 3-4)

### 5.1 Warm-Up: Change a Button Color

Find `src/components/glass_button.rs`. Find the "primary" variant. Change the background color from cyan to blue.

1. Make the change
2. Check `trunk serve` for compile errors
3. Visually verify in browser
4. Commit: `git commit -m "change primary button to blue"`

This is a trivial change. The point is to verify your workflow works.

### 5.2 Small Feature: Add a Tooltip

We need tooltips on some icons. Here's your task:

1. Create `src/components/tooltip.rs`
2. Implement a simple tooltip that shows on hover
3. Export it from `src/components/mod.rs`
4. Use it somewhere in the dashboard

Here's a starting point:

```rust
use leptos::*;

#[component]
pub fn Tooltip(
    children: Children,
    text: String,
) -> impl IntoView {
    let (visible, set_visible) = create_signal(false);

    view! {
        <div
            class="relative inline-block"
            on:mouseenter=move |_| set_visible.set(true)
            on:mouseleave=move |_| set_visible.set(false)
        >
            {children()}
            <Show when=move || visible.get()>
                <div class="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-slate-800 text-white text-xs rounded whitespace-nowrap">
                    {text.clone()}
                </div>
            </Show>
        </div>
    }
}
```

Try to:
- Add an arrow pointing down
- Add a fade-in animation
- Handle edge cases (tooltip near screen edge)

### 5.3 PR Checklist

Before submitting:

```bash
# Format code
cargo fmt

# Check for warnings
cargo clippy -- -W clippy::all

# Run tests
cargo test

# Build for production (catches more issues)
trunk build --release
```

Write a clear PR description:
- What you changed
- Why you changed it
- How to test it
- Screenshot if visual

---

## Part 6: Understanding the Domain (Week 2)

Now that you can make changes, let's understand what you're building.

### 6.1 What LatticeForge Does

LatticeForge monitors data streams (signals) and detects:
- **Phase transitions** — when a system shifts from one state to another (like water freezing)
- **Causal relationships** — which signals affect which other signals
- **Regime changes** — when the "rules" of the system change

Think of it as a sophisticated early warning system.

### 6.2 Key Concepts for UI

**Signal:** A time series of data points. Could be stock price, temperature, social media sentiment, etc. Displayed as sparklines or charts.

**Entropy:** How "confused" or "uncertain" a prediction is. Low entropy = confident. High entropy = uncertain. Displayed as gauges.

**Micro-grokking:** When an AI model suddenly "gets it" — entropy drops sharply. We detect this in real-time and highlight it.

**Causal graph:** A network showing which signals affect which. Edges have weights based on "transfer entropy."

**Regime:** A labeled state the system is in. "Stable", "Transitioning", "Critical", etc.

### 6.3 The Dashboard

The main view shows:
- Grid of signal cards (each showing latest value, sparkline, regime)
- Causal graph visualization (force-directed)
- Alert feed (when detectors trigger)
- Phase timeline (history of transitions)

When a user clicks a signal card, a detail panel slides up showing:
- Full chart history
- Related signals
- Detected patterns
- Confidence intervals

### 6.4 The Briefings Page

AI-generated intelligence reports:
- Executive summary
- Domain-specific breakdowns (political, economic, tech, etc.)
- Next strategic move recommendation
- Sources and confidence levels

The tricky part: this content streams in via WebSocket as the AI generates it. You need to handle:
- Progressive rendering
- Highlighting as new content appears
- Error states if generation fails

---

## Part 7: Common Patterns and Gotchas

### 7.1 The Clone Dance

You'll clone things a lot. This is normal.

```rust
// Error: value moved
let name = "Alice".to_string();
view! {
    <p>{name}</p>
    <p>{name}</p> // ERROR: name already moved
}

// Fix: clone
let name = "Alice".to_string();
view! {
    <p>{name.clone()}</p>
    <p>{name}</p> // OK
}
```

For closures that capture and need to use a value multiple times:

```rust
let (data, _) = create_signal(vec![1, 2, 3]);

// Need to clone before closure
let data_for_len = data.clone();
let data_for_sum = data.clone();

let len = create_memo(move |_| data_for_len.get().len());
let sum = create_memo(move |_| data_for_sum.get().iter().sum::<i32>());
```

### 7.2 The Move Keyword

`move` makes a closure take ownership of captured values:

```rust
let count = create_signal(0).0;

// Without move: borrows count (might not live long enough)
// let handler = |_| println!("{}", count.get());

// With move: owns count (always works)
let handler = move |_| println!("{}", count.get());
```

In Leptos, use `move` for event handlers and reactive closures. It's almost always what you want.

### 7.3 Signal in Signal

Don't nest signals:

```rust
// Bad
let (outer, set_outer) = create_signal(create_signal(0));

// Good
#[derive(Clone)]
struct State {
    value: i32,
}
let (state, set_state) = create_signal(State { value: 0 });
```

### 7.4 Debugging Reactivity

When something isn't updating:

1. Check if you're using `.get()` inside a reactive context
2. Check if the signal is actually changing (add a log in an effect)
3. Check if you're comparing the right thing (`.get()` returns a snapshot)

```rust
// Add debug logging
create_effect(move |_| {
    log::debug!("Count changed to: {}", count.get());
});
```

### 7.5 Async in Components

Use `spawn_local` for async operations:

```rust
let (data, set_data) = create_signal(None);

create_effect(move |_| {
    spawn_local(async move {
        let result = fetch_something().await;
        set_data.set(Some(result));
    });
});
```

For data that depends on props, use `create_resource`:

```rust
let data = create_resource(
    move || props.id.clone(),  // Source signal
    |id| async move {          // Fetcher
        fetch_by_id(id).await
    }
);

view! {
    {move || match data.get() {
        None => view! { <p>"Loading..."</p> }.into_view(),
        Some(d) => view! { <DataView data=d /> }.into_view(),
    }}
}
```

---

## Part 8: Testing Your Code

### 8.1 Unit Tests

For pure logic:

```rust
// In the same file or a tests module
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(1234.5), "1,234.50");
    }
}
```

Run: `cargo test`

### 8.2 Component Tests

For components, we use browser tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn counter_increments() {
        // Mount component
        let document = web_sys::window().unwrap().document().unwrap();
        let container = document.create_element("div").unwrap();
        document.body().unwrap().append_child(&container).unwrap();

        mount_to(container.clone(), Counter);

        // Find and click button
        let button = container.query_selector("button").unwrap().unwrap();
        button.dyn_into::<HtmlElement>().unwrap().click();

        // Assert
        let span = container.query_selector("span").unwrap().unwrap();
        assert_eq!(span.text_content().unwrap(), "1");
    }
}
```

Run: `wasm-pack test --headless --chrome`

### 8.3 What to Test

- Logic-heavy functions (formatters, validators, calculations)
- Components with complex state
- Edge cases (empty data, error states, boundaries)

Don't test:
- Simple rendering (if it compiles, it renders)
- Framework behavior (Leptos is tested by Leptos)

---

## Part 9: Getting Help

### 9.1 When You're Stuck

1. **Read the error message.** Rust errors are verbose but helpful.
2. **Check the Leptos docs.** https://leptos.dev
3. **Search Leptos Discord.** Someone probably hit the same issue.
4. **Ask in #frontend-eng.** Include: what you tried, the error, your code.

### 9.2 Code Review Expectations

Your PRs will be reviewed for:
- **Correctness:** Does it work?
- **Style:** Does it match existing patterns?
- **Performance:** Any unnecessary clones or re-renders?
- **Tests:** Did you add them?

Expect feedback. It's not personal. Everyone's code gets reviewed.

### 9.3 Pairing Sessions

Book time with a senior engineer when:
- You're stuck for > 30 minutes
- You're about to make an architectural decision
- You want to understand why something is the way it is

Don't suffer in silence. Asking questions is part of the job.

---

## Part 10: Milestones

### Week 1 Checklist
- [ ] Environment running
- [ ] Made a trivial change (button color)
- [ ] Built a simple component (tooltip)
- [ ] Submitted first PR
- [ ] Understood signal/memo/effect basics

### Week 2 Checklist
- [ ] Built a component that uses real data
- [ ] Handled loading/error states
- [ ] Wrote tests for your component
- [ ] Understood the domain (signals, entropy, graphs)
- [ ] Can explain the folder structure to someone new

### Month 1 Checklist
- [ ] Owned a small feature end-to-end
- [ ] Debugged a reactivity issue
- [ ] Contributed to code review
- [ ] Improved something proactively (docs, tests, cleanup)

---

## Appendix: Quick Reference

### Leptos Cheat Sheet

```rust
// State
let (value, set_value) = create_signal(initial);
let computed = create_memo(move |_| value.get() * 2);

// Effects
create_effect(move |_| { /* runs on dependency change */ });

// Conditional
<Show when=move || condition.get()>...</Show>

// Lists
<For each=move || items.get() key=|i| i.id children=|i| view! {...} />

// Events
<button on:click=move |_| do_thing()>

// Two-way binding
<input prop:value=move || v.get() on:input=move |e| set_v.set(event_target_value(&e)) />

// Context
provide_context(MyState::new());
let state = expect_context::<MyState>();
```

### Common Errors and Fixes

| Error | Fix |
|-------|-----|
| "value moved" | Clone before using again |
| "closure may outlive" | Add `move` keyword |
| "cannot find in scope" | Check imports, use `use leptos::*;` |
| "expected impl IntoView" | Make sure view returns consistently typed views |
| "borrow later used" | Restructure to avoid overlapping borrows |

---

*You got this. Every senior engineer was once staring at their first Rust compiler error. Ping me if you need anything. — [Senior Engineer Name]*
