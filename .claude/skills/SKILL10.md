# RUST_SYSTEMS.skill.md

## Systems Programming in Rust: Ownership, Concurrency, and Performance

**Version**: 1.0
**Domain**: Systems Programming, Memory Safety, Concurrent Programming
**Prerequisites**: Programming fundamentals, basic memory concepts
**Complexity**: Intermediate to Advanced

---

## 1. EXECUTIVE SUMMARY

Rust provides memory safety without garbage collection through compile-time ownership tracking. This skill covers the ownership system, borrowing rules, lifetimes, concurrency primitives, async/await, FFI, and performance patterns.

**Core Insight**: Rust's learning curve is front-loaded. Once you internalize ownership, the compiler becomes an ally that catches bugs before runtime.

---

## 2. OWNERSHIP SYSTEM

### 2.1 The Three Rules

1. **Each value has exactly one owner**
2. **When owner goes out of scope, value is dropped**
3. **Ownership can be transferred (moved)**

```rust
fn main() {
    let s1 = String::from("hello");  // s1 owns the String
    let s2 = s1;                      // ownership moved to s2
    // println!("{}", s1);            // ERROR: s1 no longer valid
    println!("{}", s2);               // OK: s2 owns it
}  // s2 goes out of scope, String is dropped
```

### 2.2 Move Semantics

**Move** = transfer ownership. Original binding invalid.

```rust
let v1 = vec![1, 2, 3];
let v2 = v1;  // MOVE - v1 is now invalid

fn take_ownership(v: Vec<i32>) {
    // v owns the vector now
}  // v dropped here

take_ownership(v2);  // v2 moved into function
// v2 no longer valid
```

### 2.3 Copy Types

Types that implement `Copy` are duplicated instead of moved:
- All integer types
- `bool`, `char`
- Tuples of Copy types
- Arrays of Copy types

```rust
let x = 5;
let y = x;  // COPY, not move
println!("{} {}", x, y);  // both valid
```

### 2.4 Clone

Explicit deep copy:

```rust
let s1 = String::from("hello");
let s2 = s1.clone();  // deep copy
println!("{} {}", s1, s2);  // both valid
```

---

## 3. BORROWING

### 3.1 References

Borrow without taking ownership:

```rust
fn calculate_length(s: &String) -> usize {
    s.len()
}  // s goes out of scope but doesn't drop (it's a reference)

let s = String::from("hello");
let len = calculate_length(&s);  // borrow s
println!("{} has length {}", s, len);  // s still valid
```

### 3.2 Borrowing Rules

1. **Any number of immutable references OR exactly one mutable reference**
2. **References must always be valid (no dangling)**

```rust
let mut s = String::from("hello");

let r1 = &s;      // OK - immutable borrow
let r2 = &s;      // OK - multiple immutable
// let r3 = &mut s;  // ERROR - can't borrow mutably while immutable borrows exist

println!("{} {}", r1, r2);
// r1 and r2 no longer used after this

let r3 = &mut s;  // OK now - no other borrows active
r3.push_str(" world");
```

### 3.3 Mutable References

```rust
fn modify(s: &mut String) {
    s.push_str(" world");
}

let mut s = String::from("hello");
modify(&mut s);
println!("{}", s);  // "hello world"
```

### 3.4 Non-Lexical Lifetimes (NLL)

Borrows end when last used, not at scope end:

```rust
let mut s = String::from("hello");
let r1 = &s;
println!("{}", r1);  // last use of r1

let r2 = &mut s;  // OK - r1's borrow ended
r2.push_str(" world");
```

---

## 4. LIFETIMES

### 4.1 What Lifetimes Are

Lifetimes ensure references don't outlive their data. Usually inferred, sometimes explicit.

```rust
// Won't compile - what's the lifetime of return?
fn longest(x: &str, y: &str) -> &str {
    if x.len() > y.len() { x } else { y }
}

// Fixed with lifetime annotations
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

### 4.2 Lifetime Annotation Syntax

- `'a` - lifetime parameter
- `&'a T` - reference with lifetime 'a
- `&'a mut T` - mutable reference with lifetime 'a

### 4.3 Lifetime Elision Rules

Compiler infers lifetimes in common cases:

1. Each input reference gets its own lifetime
2. If one input lifetime, output gets that lifetime
3. If `&self` or `&mut self`, output gets self's lifetime

```rust
// These are equivalent:
fn first_word(s: &str) -> &str
fn first_word<'a>(s: &'a str) -> &'a str
```

### 4.4 Static Lifetime

`'static` - lives for entire program duration:

```rust
let s: &'static str = "hello";  // string literal
```

### 4.5 Lifetime in Structs

Structs holding references need lifetime parameters:

```rust
struct ImportantExcerpt<'a> {
    part: &'a str,
}

let novel = String::from("Call me Ishmael...");
let first_sentence = novel.split('.').next().unwrap();
let excerpt = ImportantExcerpt { part: first_sentence };
// excerpt can't outlive novel
```

---

## 5. SMART POINTERS

### 5.1 Box<T> - Heap Allocation

```rust
let b = Box::new(5);  // 5 stored on heap
println!("{}", b);    // deref automatically

// Use for:
// - Recursive types
// - Large data you want to transfer ownership without copying
// - Trait objects
```

### 5.2 Rc<T> - Reference Counting

Multiple owners (single-threaded):

```rust
use std::rc::Rc;

let a = Rc::new(vec![1, 2, 3]);
let b = Rc::clone(&a);  // increments count, not deep copy
let c = Rc::clone(&a);

println!("count: {}", Rc::strong_count(&a));  // 3
```

### 5.3 Arc<T> - Atomic Reference Counting

Multiple owners (thread-safe):

```rust
use std::sync::Arc;
use std::thread;

let data = Arc::new(vec![1, 2, 3]);

let handles: Vec<_> = (0..3).map(|_| {
    let data = Arc::clone(&data);
    thread::spawn(move || {
        println!("{:?}", data);
    })
}).collect();

for handle in handles {
    handle.join().unwrap();
}
```

### 5.4 RefCell<T> - Interior Mutability

Borrow checking at runtime:

```rust
use std::cell::RefCell;

let data = RefCell::new(5);

{
    let mut borrowed = data.borrow_mut();
    *borrowed += 1;
}

println!("{}", data.borrow());  // 6
```

### 5.5 Common Patterns

- `Rc<RefCell<T>>` - multiple owners with mutation (single-threaded)
- `Arc<Mutex<T>>` - multiple owners with mutation (multi-threaded)

---

## 6. ERROR HANDLING

### 6.1 Result<T, E>

```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}

fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("division by zero".to_string())
    } else {
        Ok(a / b)
    }
}
```

### 6.2 The ? Operator

Propagate errors concisely:

```rust
fn read_username() -> Result<String, io::Error> {
    let mut file = File::open("username.txt")?;  // returns early on Err
    let mut s = String::new();
    file.read_to_string(&mut s)?;
    Ok(s)
}
```

### 6.3 Option<T>

```rust
enum Option<T> {
    Some(T),
    None,
}

fn find_user(id: u32) -> Option<User> {
    // ...
}

if let Some(user) = find_user(42) {
    println!("Found: {}", user.name);
}
```

### 6.4 Error Type Design

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MyError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(#[from] std::num::ParseIntError),

    #[error("Custom error: {message}")]
    Custom { message: String },
}
```

---

## 7. TRAITS

### 7.1 Defining Traits

```rust
trait Summary {
    fn summarize(&self) -> String;

    // Default implementation
    fn summarize_author(&self) -> String {
        String::from("(anonymous)")
    }
}
```

### 7.2 Implementing Traits

```rust
struct Article {
    title: String,
    author: String,
    content: String,
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("{} by {}", self.title, self.author)
    }
}
```

### 7.3 Trait Bounds

```rust
fn notify<T: Summary>(item: &T) {
    println!("Breaking: {}", item.summarize());
}

// Multiple bounds
fn notify<T: Summary + Display>(item: &T) { ... }

// Where clause (cleaner for complex bounds)
fn notify<T>(item: &T)
where
    T: Summary + Display,
{ ... }
```

### 7.4 Trait Objects

Dynamic dispatch:

```rust
fn notify(item: &dyn Summary) {
    println!("Breaking: {}", item.summarize());
}

let articles: Vec<Box<dyn Summary>> = vec![
    Box::new(article1),
    Box::new(tweet1),
];
```

### 7.5 Important Standard Traits

- `Clone` - explicit duplication
- `Copy` - implicit duplication
- `Debug` - debug formatting
- `Display` - user-facing formatting
- `Default` - default values
- `PartialEq`, `Eq` - equality comparison
- `PartialOrd`, `Ord` - ordering comparison
- `Hash` - hashing
- `From`, `Into` - conversions
- `Iterator` - iteration

---

## 8. CONCURRENCY

### 8.1 Threads

```rust
use std::thread;

let handle = thread::spawn(|| {
    println!("Hello from thread!");
});

handle.join().unwrap();
```

### 8.2 Move Closures

Transfer ownership to thread:

```rust
let v = vec![1, 2, 3];

let handle = thread::spawn(move || {
    println!("{:?}", v);  // v moved into closure
});
```

### 8.3 Message Passing (Channels)

```rust
use std::sync::mpsc;

let (tx, rx) = mpsc::channel();

thread::spawn(move || {
    tx.send("hello").unwrap();
});

let received = rx.recv().unwrap();
println!("{}", received);
```

### 8.4 Shared State (Mutex)

```rust
use std::sync::{Arc, Mutex};

let counter = Arc::new(Mutex::new(0));

let handles: Vec<_> = (0..10).map(|_| {
    let counter = Arc::clone(&counter);
    thread::spawn(move || {
        let mut num = counter.lock().unwrap();
        *num += 1;
    })
}).collect();

for handle in handles {
    handle.join().unwrap();
}

println!("Result: {}", *counter.lock().unwrap());
```

### 8.5 RwLock

Multiple readers OR one writer:

```rust
use std::sync::RwLock;

let lock = RwLock::new(5);

// Multiple readers OK
{
    let r1 = lock.read().unwrap();
    let r2 = lock.read().unwrap();
}

// One writer
{
    let mut w = lock.write().unwrap();
    *w += 1;
}
```

---

## 9. ASYNC/AWAIT

### 9.1 Futures

```rust
async fn fetch_data() -> Result<String, Error> {
    // Returns a Future, doesn't execute yet
    Ok("data".to_string())
}

// Execute with runtime
#[tokio::main]
async fn main() {
    let data = fetch_data().await.unwrap();
}
```

### 9.2 Tokio Runtime

```rust
use tokio;

#[tokio::main]
async fn main() {
    let handle = tokio::spawn(async {
        // async work
    });

    handle.await.unwrap();
}
```

### 9.3 Concurrent Execution

```rust
use tokio;

async fn main() {
    let (r1, r2) = tokio::join!(
        fetch_url("http://a.com"),
        fetch_url("http://b.com"),
    );
}
```

### 9.4 Select

First future to complete:

```rust
use tokio::select;

select! {
    val = future1 => println!("future1 completed: {:?}", val),
    val = future2 => println!("future2 completed: {:?}", val),
}
```

### 9.5 Streams

Async iterators:

```rust
use tokio_stream::StreamExt;

let mut stream = tokio_stream::iter(vec![1, 2, 3]);

while let Some(value) = stream.next().await {
    println!("{}", value);
}
```

---

## 10. FFI (Foreign Function Interface)

### 10.1 Calling C from Rust

```rust
extern "C" {
    fn abs(input: i32) -> i32;
    fn strlen(s: *const c_char) -> usize;
}

fn main() {
    unsafe {
        println!("{}", abs(-5));
    }
}
```

### 10.2 Exposing Rust to C

```rust
#[no_mangle]
pub extern "C" fn rust_function(x: i32) -> i32 {
    x * 2
}
```

### 10.3 CString and CStr

```rust
use std::ffi::{CString, CStr};

// Rust String -> C string
let c_string = CString::new("hello").unwrap();
let ptr = c_string.as_ptr();

// C string -> Rust &str
unsafe {
    let c_str = CStr::from_ptr(ptr);
    let rust_str = c_str.to_str().unwrap();
}
```

### 10.4 PyO3 (Python FFI)

```rust
use pyo3::prelude::*;

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pymodule]
fn my_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
```

---

## 11. PERFORMANCE PATTERNS

### 11.1 Avoid Allocations

```rust
// Bad - allocates
let s = format!("{}{}", a, b);

// Better - reuse buffer
let mut s = String::with_capacity(a.len() + b.len());
s.push_str(a);
s.push_str(b);
```

### 11.2 Use Iterators

```rust
// Iterators are zero-cost abstractions
let sum: i32 = (1..1000)
    .filter(|x| x % 2 == 0)
    .map(|x| x * x)
    .sum();
```

### 11.3 Stack vs Heap

Prefer stack allocation when possible:

```rust
// Stack - fast
let arr = [0u8; 1024];

// Heap - slower
let vec = vec![0u8; 1024];
```

### 11.4 Inlining

```rust
#[inline]
fn small_function() { }

#[inline(always)]
fn critical_function() { }
```

### 11.5 SIMD

```rust
use std::simd::*;

let a = f32x4::from_array([1.0, 2.0, 3.0, 4.0]);
let b = f32x4::from_array([5.0, 6.0, 7.0, 8.0]);
let c = a + b;  // SIMD add
```

---

## 12. COMMON PATTERNS

### 12.1 Builder Pattern

```rust
struct ServerBuilder {
    host: String,
    port: u16,
}

impl ServerBuilder {
    fn new() -> Self {
        ServerBuilder {
            host: "localhost".to_string(),
            port: 8080,
        }
    }

    fn host(mut self, host: &str) -> Self {
        self.host = host.to_string();
        self
    }

    fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    fn build(self) -> Server {
        Server { host: self.host, port: self.port }
    }
}

let server = ServerBuilder::new()
    .host("0.0.0.0")
    .port(3000)
    .build();
```

### 12.2 Newtype Pattern

```rust
struct UserId(u64);
struct PostId(u64);

// Can't accidentally mix them up
fn get_user(id: UserId) -> User { }
fn get_post(id: PostId) -> Post { }
```

### 12.3 Type State Pattern

```rust
struct Locked;
struct Unlocked;

struct Door<State> {
    _state: PhantomData<State>,
}

impl Door<Locked> {
    fn unlock(self) -> Door<Unlocked> {
        Door { _state: PhantomData }
    }
}

impl Door<Unlocked> {
    fn lock(self) -> Door<Locked> {
        Door { _state: PhantomData }
    }

    fn open(&self) { }  // Only unlocked doors can open
}
```

---

## 13. TOOLING

### 13.1 Cargo

```bash
cargo new project_name      # new project
cargo build                  # build
cargo build --release        # optimized build
cargo run                    # build and run
cargo test                   # run tests
cargo doc --open             # generate docs
cargo clippy                 # linting
cargo fmt                    # formatting
```

### 13.2 Useful Crates

- `serde` - serialization
- `tokio` - async runtime
- `rayon` - data parallelism
- `clap` - CLI parsing
- `anyhow` - error handling
- `thiserror` - error definitions
- `tracing` - logging/tracing
- `criterion` - benchmarking

---

## 14. REFERENCES

**Books**:
- The Rust Programming Language (official book)
- Rust for Rustaceans (Jon Gjengset)
- Programming Rust (O'Reilly)

**Resources**:
- Rust by Example: https://doc.rust-lang.org/rust-by-example/
- Rustlings: https://github.com/rust-lang/rustlings

---

## 15. VERSION HISTORY

- **v1.0** (2024-11): Initial comprehensive version

---

*Rust's compiler is famously helpful. When you get an error, read it carefully - it usually tells you exactly how to fix it.*
