# CODING_BEST_PRACTICES.skill.md

## Production-Grade Code: Patterns That Scale

**Version**: 1.0
**Domain**: Software Engineering, Code Quality, Architecture, Maintainability
**Prerequisites**: Language-specific knowledge
**Output**: Clean, maintainable, reliable code

---

## 1. EXECUTIVE SUMMARY

Good code is not about cleverness—it's about clarity, correctness, and compassion for future readers (including future you). This skill covers patterns that produce maintainable, reliable, and efficient code across languages and paradigms.

**Core Principle**: Code is read 10x more than it's written. Optimize for reading.

---

## 2. CODE QUALITY FUNDAMENTALS

### 2.1 The Quality Triangle

```
                CORRECTNESS
                    /\
                   /  \
                  /    \
                 /      \
                /________\
        CLARITY          EFFICIENCY

PRIORITY ORDER:
1. Correctness: Wrong code is worthless
2. Clarity: Confusing code breeds bugs
3. Efficiency: Only optimize what matters

NEVER: Sacrifice correctness for speed
RARELY: Sacrifice clarity for speed
SOMETIMES: Accept slower code that's clearer
```

### 2.2 Readability Principles

```javascript
// BAD: Clever but obscure
const r = d.filter(x => x.t === 'A').reduce((a, c) => a + c.v, 0);

// GOOD: Clear and obvious
const activeItems = data.filter(item => item.type === 'ACTIVE');
const totalValue = activeItems.reduce(
  (sum, item) => sum + item.value,
  0
);

READABILITY RULES:
├── Meaningful names (not abbreviations)
├── One concept per line
├── No magic numbers
├── Functions do one thing
├── Comments explain WHY, not WHAT
```

### 2.3 The Boy Scout Rule

Leave code better than you found it:

```
ON EVERY CHANGE:
├── Fix obvious issues nearby
├── Improve naming if unclear
├── Add missing comments
├── Remove dead code
├── Update outdated comments

LIMITS:
├── Don't refactor unrelated code
├── Keep changes focused
├── Don't fix everything at once
```

---

## 3. NAMING CONVENTIONS

### 3.1 Naming Philosophy

Names should reveal intent:

```python
# BAD: What is this?
def proc(d):
    for i in d:
        if i.s == 1:
            i.s = 2

# GOOD: Intent is clear
def activate_pending_users(users):
    for user in users:
        if user.status == Status.PENDING:
            user.status = Status.ACTIVE

NAMING RULES:
├── Use domain vocabulary
├── Avoid abbreviations
├── Boolean names should read as questions
├── Functions should be verb phrases
├── Classes should be noun phrases
```

### 3.2 Scope-Appropriate Names

```
SCOPE       | LENGTH   | EXAMPLE
------------|----------|------------------
Loop index  | 1 char   | i, j, x
Lambda arg  | Short    | x, item
Local var   | Medium   | userCount
Field/Prop  | Medium   | maxRetries
Function    | Long     | calculateTotalOrderValue
Class       | Long     | UserAuthenticationService
```

### 3.3 Naming Patterns

```javascript
// Booleans: should read as true/false questions
const isActive = true;
const hasPermission = user.roles.includes('admin');
const canEdit = hasPermission && isActive;

// Functions: verb + noun
function calculateTotal() {}
function validateInput() {}
function fetchUserData() {}

// Transformers: xToY
function celsiusToFahrenheit() {}
function userToDTO() {}
function requestToCommand() {}

// Handlers: onEvent or handleEvent
function onClick() {}
function handleSubmit() {}
function onUserCreated() {}

// Factories: createX or buildX
function createUser() {}
function buildQuery() {}
```

---

## 4. FUNCTION DESIGN

### 4.1 Single Responsibility

Each function does one thing:

```python
# BAD: Does too much
def process_order(order):
    validate_order(order)
    calculate_total(order)
    apply_discount(order)
    charge_payment(order)
    send_confirmation(order)
    update_inventory(order)
    notify_warehouse(order)

# GOOD: Orchestrates single-purpose functions
def process_order(order):
    validated_order = validate(order)
    priced_order = calculate_price(validated_order)
    charged_order = process_payment(priced_order)
    fulfill(charged_order)
```

### 4.2 Function Size

```
GUIDELINES:
├── Max 20-30 lines (visible without scrolling)
├── Max 3-4 parameters
├── Max 2 levels of nesting
├── One level of abstraction

WHEN TOO LONG:
├── Extract helper functions
├── Convert to class with methods
├── Split into multiple functions
├── Use early returns
```

### 4.3 Pure Functions

Prefer pure functions where possible:

```javascript
// IMPURE: Depends on and modifies external state
let total = 0;
function addToTotal(amount) {
  total += amount; // Side effect
  console.log(total); // Another side effect
}

// PURE: Same input → same output, no side effects
function add(a, b) {
  return a + b;
}

PURE FUNCTION BENEFITS:
├── Easy to test
├── Easy to reason about
├── Cacheable
├── Parallelizable
├── No hidden dependencies
```

---

## 5. ERROR HANDLING

### 5.1 Fail Fast

Detect and report errors early:

```python
# BAD: Error discovered deep in execution
def process(data):
    # ... lots of work ...
    # Much later:
    if not data.required_field:
        raise Error("Missing required field")  # Too late!

# GOOD: Validate inputs immediately
def process(data):
    if not data.required_field:
        raise ValueError("Missing required field")
    # Now proceed with valid data
```

### 5.2 Error Categories

Handle different errors differently:

```javascript
try {
  await fetchData();
} catch (error) {
  if (error instanceof NetworkError) {
    // Retry with backoff
    return retry(fetchData);
  }
  if (error instanceof ValidationError) {
    // Show user message
    return showError(error.message);
  }
  if (error instanceof AuthenticationError) {
    // Redirect to login
    return redirectToLogin();
  }
  // Unknown error: log and rethrow
  console.error('Unexpected error:', error);
  throw error;
}
```

### 5.3 Error Messages

Good error messages are actionable:

```
BAD:  "Error occurred"
BAD:  "Invalid input"
BAD:  "Failed"

GOOD: "Email address 'foo' is invalid. Expected format: user@domain.com"
GOOD: "Cannot connect to database at localhost:5432. Is PostgreSQL running?"
GOOD: "Payment failed: Card declined. Please try a different payment method."

ERROR MESSAGE TEMPLATE:
[What went wrong] + [Why] + [How to fix]
```

---

## 6. TESTING PATTERNS

### 6.1 Test Structure

```python
def test_user_creation():
    # ARRANGE: Set up test data
    user_data = {"email": "test@example.com", "name": "Test User"}

    # ACT: Execute the code under test
    result = create_user(user_data)

    # ASSERT: Verify the results
    assert result.id is not None
    assert result.email == "test@example.com"
    assert result.created_at is not None
```

### 6.2 Test Coverage Strategy

```
TESTING PYRAMID:
        /\
       /  \  E2E Tests (few, slow, brittle)
      /____\
     /      \  Integration Tests (some, medium)
    /________\
   /          \ Unit Tests (many, fast, stable)
  /______________\

COVERAGE PRIORITIES:
├── Core business logic: 90%+ coverage
├── Edge cases: Explicit tests
├── Happy path: Always tested
├── Error paths: Tested for each error type
├── UI: Focus on critical flows
```

### 6.3 Test Quality

```python
# BAD: Tests implementation details
def test_user_repository():
    repo = UserRepository()
    repo._cache = {}  # Testing private state
    repo._db_connection = mock  # Tight coupling

# GOOD: Tests behavior
def test_create_user_returns_user_with_id():
    repo = UserRepository()
    user = repo.create(email="test@example.com")
    assert user.id is not None

# BAD: Test name is vague
def test_user():
    pass

# GOOD: Test name describes behavior
def test_create_user_with_duplicate_email_raises_conflict_error():
    pass
```

---

## 7. CODE ORGANIZATION

### 7.1 File Structure

```
PROJECT STRUCTURE:
src/
├── components/     # UI components
├── services/       # Business logic
├── repositories/   # Data access
├── models/         # Domain entities
├── utils/          # Pure utilities
├── hooks/          # React hooks (if applicable)
├── api/            # API route handlers
├── config/         # Configuration
└── types/          # TypeScript types

RULES:
├── One concept per file
├── Related files in same directory
├── Index files for public exports
├── Private files prefixed with _
```

### 7.2 Module Dependencies

```
DEPENDENCY DIRECTION:
├── UI → Services → Repositories → Database
├── High level → Low level
├── Concrete → Abstract

NEVER:
├── Repository → UI (inverse dependency)
├── Circular dependencies
├── God modules that know everything
```

### 7.3 Cohesion and Coupling

```
HIGH COHESION (good):
├── UserService handles all user operations
├── AuthController handles all auth endpoints
├── CartRepository handles all cart data

LOW COUPLING (good):
├── Services don't know about HTTP
├── Repositories don't know about business rules
├── UI doesn't know about database
```

---

## 8. DEFENSIVE CODING

### 8.1 Input Validation

Never trust input:

```typescript
function processUserInput(input: unknown): Result {
  // Validate type
  if (typeof input !== 'object' || input === null) {
    throw new ValidationError('Input must be an object');
  }

  // Validate required fields
  if (!input.email || typeof input.email !== 'string') {
    throw new ValidationError('Email is required');
  }

  // Validate format
  if (!isValidEmail(input.email)) {
    throw new ValidationError('Invalid email format');
  }

  // Sanitize
  const sanitizedEmail = sanitize(input.email.toLowerCase());

  // Now safe to use
  return process(sanitizedEmail);
}
```

### 8.2 Null Safety

Handle null/undefined explicitly:

```typescript
// BAD: Assumes data exists
function getUsername(user) {
  return user.profile.name; // Crashes if user or profile is null
}

// GOOD: Handles missing data
function getUsername(user: User | null): string {
  return user?.profile?.name ?? 'Anonymous';
}

// BETTER: Make invariants explicit
function getUsername(user: User): string {
  if (!user.profile) {
    throw new Error('User must have profile');
  }
  return user.profile.name;
}
```

### 8.3 Boundary Conditions

Always consider edges:

```python
def get_element(items, index):
    # Handle edge cases
    if not items:
        return None
    if index < 0:
        return None
    if index >= len(items):
        return None

    return items[index]

BOUNDARY CHECKLIST:
├── Empty collection
├── Single element
├── First and last element
├── Negative values
├── Maximum values
├── Null/undefined
├── Empty string
├── Unicode/special characters
```

---

## 9. PERFORMANCE PATTERNS

### 9.1 Optimization Guidelines

```
RULES:
1. Don't optimize prematurely
2. Measure before optimizing
3. Optimize the right thing (critical path)
4. Keep optimized code readable

OPTIMIZATION ORDER:
1. Algorithm complexity (O(n) vs O(n²))
2. Data structure choice
3. Caching
4. Batching
5. Parallelization
6. Low-level optimization (last resort)
```

### 9.2 Common Optimizations

```javascript
// BAD: O(n²) - nested loop with lookup
for (const order of orders) {
  const user = users.find(u => u.id === order.userId);
}

// GOOD: O(n) - build lookup map first
const userMap = new Map(users.map(u => [u.id, u]));
for (const order of orders) {
  const user = userMap.get(order.userId);
}

// BAD: Repeated computation
for (const item of items) {
  if (calculateExpensiveThing() > threshold) { ... }
}

// GOOD: Compute once
const expensiveResult = calculateExpensiveThing();
for (const item of items) {
  if (expensiveResult > threshold) { ... }
}
```

### 9.3 Memory Patterns

```javascript
// BAD: Creates large intermediate array
const result = largeArray
  .map(x => x * 2)      // New array
  .filter(x => x > 10)  // Another new array
  .map(x => x.toString()); // Yet another

// BETTER: Single pass with generator
function* processLargeArray(arr) {
  for (const x of arr) {
    const doubled = x * 2;
    if (doubled > 10) {
      yield doubled.toString();
    }
  }
}

// BEST: For very large data - stream processing
```

---

## 10. CODE REVIEW READY

### 10.1 Self-Review Checklist

Before submitting code:

```
FUNCTIONALITY:
├── [ ] Does it do what it's supposed to?
├── [ ] Are edge cases handled?
├── [ ] Are errors handled appropriately?

QUALITY:
├── [ ] Is it readable?
├── [ ] Are names meaningful?
├── [ ] Is duplication minimized?
├── [ ] Are functions focused?

TESTING:
├── [ ] Are tests included?
├── [ ] Do tests cover important cases?
├── [ ] Do all tests pass?

SECURITY:
├── [ ] Input validated?
├── [ ] No secrets in code?
├── [ ] No SQL/XSS injection risks?

PERFORMANCE:
├── [ ] No obvious inefficiencies?
├── [ ] No unnecessary work?
```

### 10.2 Code Review Response

How to respond to feedback:

```
GOOD RESPONSES:
├── "Good catch, fixed in [commit]"
├── "Intentional because [reason]. Added comment."
├── "Disagree, here's why: [reasoning]. Thoughts?"

BAD RESPONSES:
├── "Works on my machine"
├── "It's fine"
├── [Ignore and don't respond]
```

---

## 11. LANGUAGE-AGNOSTIC PATTERNS

### 11.1 Design Patterns (Use Sparingly)

```
FREQUENTLY USEFUL:
├── Factory: Creating objects with complex setup
├── Strategy: Swappable algorithms
├── Observer: Event-based communication
├── Adapter: Interface translation

RARELY NEEDED:
├── Singleton: Often a code smell
├── Abstract Factory: Usually overkill
├── Visitor: Complex and rarely clearer

PRINCIPLE: Use patterns to solve problems, not to show cleverness
```

### 11.2 SOLID Principles

```
S - Single Responsibility
    Each module/class has one reason to change

O - Open/Closed
    Open for extension, closed for modification

L - Liskov Substitution
    Subtypes must be substitutable for base types

I - Interface Segregation
    Many specific interfaces > one general interface

D - Dependency Inversion
    Depend on abstractions, not concretions

PRIORITY: S > D > L > O > I (roughly)
```

---

## 12. IMPLEMENTATION CHECKLIST

### For Every Function:
- [ ] Meaningful name
- [ ] Single responsibility
- [ ] Input validation
- [ ] Error handling
- [ ] Pure if possible
- [ ] Tested

### For Every File:
- [ ] One concept per file
- [ ] Clear exports
- [ ] No circular dependencies
- [ ] Consistent style

### For Every PR:
- [ ] Self-reviewed
- [ ] Tests pass
- [ ] No obvious issues
- [ ] Documentation updated

---

**Remember**: The best code is the code you don't have to debug. Write for the reader, test for the edge cases, and optimize only what matters.

Write simple. Write correct. Write once.
