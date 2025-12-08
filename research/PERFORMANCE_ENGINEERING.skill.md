# PERFORMANCE_ENGINEERING.skill.md

## High-Performance AI Agent Operations and Token-Optimal Execution

**Version**: 1.0
**Domain**: Performance Optimization, Latency Reduction, Token Efficiency, Parallel Execution
**Prerequisites**: Understanding of LLM mechanics, tool orchestration, context management
**Output**: Faster responses, lower costs, higher throughput

---

## 1. EXECUTIVE SUMMARY

Performance engineering for AI agents is not about making the LLM faster—it's about minimizing wasted work, parallelizing independent operations, and optimizing the information flow between agent and tools. Every token counts. Every tool call has latency. Every context window has limits.

**Core Principle**: The fastest operation is the one you don't do. The cheapest token is the one you don't generate.

---

## 2. TOKEN ECONOMICS

### 2.1 The Hidden Cost Model

```
TOKEN COST BREAKDOWN:
├── Input tokens (context) ──── $X per 1K (cheaper)
├── Output tokens (generation) ─ $Y per 1K (3-5x more expensive)
└── Tool calls ───────────────── Latency + round-trip tokens

COST OPTIMIZATION HIERARCHY:
1. Reduce output tokens (highest ROI)
2. Reduce redundant context
3. Batch operations
4. Cache repeated patterns
5. Speculative execution
```

### 2.2 Output Token Minimization

```
ANTI-PATTERN: Verbose responses
"I'll help you with that. Let me first understand what you're asking.
Based on your request, I'll need to examine the codebase. Let me start
by looking at the files. After careful analysis, I have found..."

PATTERN: Direct execution
[Tool call → Action → Minimal status]

SAVINGS: 50-80% reduction in output tokens
```

### 2.3 Context Window Management

```javascript
const CONTEXT_OPTIMIZATION = {
  // Read file summaries, not full files when exploring
  exploration: 'summary',

  // Only include relevant sections in context
  relevance_filtering: true,

  // Compress repeated patterns
  deduplication: true,

  // Drop stale information
  recency_weighting: true,

  // Hierarchical context (summary → detail on demand)
  hierarchical: true
};
```

---

## 3. PARALLEL EXECUTION PATTERNS

### 3.1 Independent Operations

When operations have no dependencies, execute them in parallel:

```
SERIAL (SLOW):
Read file A → Read file B → Read file C → Analyze
Total time: T_a + T_b + T_c + T_analyze

PARALLEL (FAST):
[Read file A]
[Read file B]  → All complete → Analyze
[Read file C]
Total time: max(T_a, T_b, T_c) + T_analyze
```

### 3.2 Dependency Graph Analysis

Before executing, build a dependency graph:

```python
def identify_parallel_opportunities(tasks):
    """
    Group tasks by dependency level.

    Level 0: No dependencies (execute first, in parallel)
    Level 1: Depends on Level 0 outputs
    Level N: Depends on Level N-1 outputs
    """
    dependency_graph = build_graph(tasks)
    levels = topological_sort(dependency_graph)

    # Execute each level in parallel
    for level in levels:
        execute_parallel(level.tasks)
```

### 3.3 Speculative Execution

For high-probability paths, speculatively prepare:

```
USER: "Fix the failing tests"

SPECULATIVE EXECUTION:
├── [Parallel] Run tests to identify failures
├── [Parallel] Read common test failure patterns
└── [Parallel] Prepare fix templates

When test results arrive → Immediately apply relevant fix
```

---

## 4. TOOL CALL OPTIMIZATION

### 4.1 Batching

Combine multiple tool calls when possible:

```
ANTI-PATTERN: Sequential file reads
Read(file1) → Read(file2) → Read(file3)
Latency: 3 × round-trip

PATTERN: Parallel batch reads
[Read(file1), Read(file2), Read(file3)]
Latency: 1 × round-trip (parallel execution)
```

### 4.2 Minimal Tool Selection

Choose the lightest tool that accomplishes the goal:

```
TOOL SELECTION HIERARCHY:
1. Glob (fastest) - When you know the pattern
2. Grep (fast) - When searching content
3. Read (medium) - When you need full file
4. Task/Agent (slow) - Only for complex exploration

DON'T: Spawn agent to find "class Foo"
DO: Grep for "class Foo" directly
```

### 4.3 Early Termination

Stop searching when answer is found:

```python
# ANTI-PATTERN: Exhaustive search
for file in all_files:
    if contains_answer(file):
        answers.append(file)
return answers  # Processed ALL files

# PATTERN: Early termination
for file in all_files:
    if contains_answer(file):
        return file  # Stop at first match
```

---

## 5. CACHING AND MEMOIZATION

### 5.1 Result Caching

Cache expensive operation results:

```
CACHEABLE OPERATIONS:
├── File contents (until modified)
├── Search results (within session)
├── Build outputs (until source changes)
├── Test results (until code changes)
└── API responses (with TTL)

CACHE KEY: hash(operation + inputs + relevant_state)
```

### 5.2 Pattern Recognition

Recognize recurring patterns to avoid re-derivation:

```
COMMON PATTERNS TO RECOGNIZE:
├── "Fix TypeScript errors" → Run tsc, parse errors, fix each
├── "Run tests" → npm test, check exit code, report
├── "Create PR" → git status, diff, push, gh pr create
├── "Commit changes" → git add, commit, status verification
```

### 5.3 Incremental Processing

Process only what changed:

```
FULL REBUILD (slow):
Read all files → Analyze all → Generate all

INCREMENTAL (fast):
Detect changes → Read only changed → Analyze delta → Update only affected
```

---

## 6. CONTEXT EFFICIENCY

### 6.1 Information Density

Maximize signal-to-noise ratio in context:

```
LOW DENSITY (wasteful):
"Here is the file content with all the boilerplate and comments..."
[1000 lines of code with 50 relevant lines]

HIGH DENSITY (efficient):
"Relevant sections of the file:"
[50 lines that matter]
"Full file available if needed."
```

### 6.2 Hierarchical Summarization

Build understanding in layers:

```
LEVEL 0: Directory structure
LEVEL 1: File purposes and relationships
LEVEL 2: Function/class signatures
LEVEL 3: Implementation details (on demand)

START at Level 0, DRILL DOWN only where needed
```

### 6.3 Relevance Filtering

Include only what's needed for current task:

```python
def filter_context(full_context, current_task):
    """
    Scoring function for context relevance.
    """
    scored_items = []
    for item in full_context:
        score = compute_relevance(item, current_task)
        if score > THRESHOLD:
            scored_items.append((score, item))

    # Return top-K most relevant
    return sorted(scored_items, reverse=True)[:MAX_CONTEXT_ITEMS]
```

---

## 7. RESPONSE OPTIMIZATION

### 7.1 Structured Output

Use structured formats for machine processing:

```
VERBOSE (expensive):
"I found three files that match your criteria. The first file is
located at src/components/Button.tsx and it contains a React
component. The second file..."

STRUCTURED (efficient):
Files matching criteria:
- src/components/Button.tsx
- src/components/Input.tsx
- src/components/Form.tsx
```

### 7.2 Progressive Disclosure

Provide summary first, details on demand:

```
FIRST RESPONSE:
"Found 3 TypeScript errors in 2 files. Want details?"

ON REQUEST:
"Errors:
1. src/api.ts:45 - Type mismatch
2. src/api.ts:78 - Missing property
3. src/utils.ts:12 - Undefined variable"
```

### 7.3 Action-Oriented Communication

Lead with action, not explanation:

```
BEFORE (explanation-heavy):
"I understand you want to fix the bug. Let me analyze the situation.
After looking at the code, I believe the issue is in the function..."

AFTER (action-oriented):
[Reads file]
"Bug is at line 45: missing null check. Fixing now."
[Edits file]
"Fixed. Run tests to verify."
```

---

## 8. LATENCY REDUCTION

### 8.1 Critical Path Optimization

Identify and optimize the critical path:

```
TASK: Fix bug and commit

CRITICAL PATH:
1. Identify bug location (can't parallelize)
2. Read relevant file (depends on 1)
3. Create fix (depends on 2)
4. Apply edit (depends on 3)
5. Verify fix works (depends on 4)

PARALLELIZABLE:
- While step 1: Prepare commit message template
- While step 4: Run linter in background
```

### 8.2 Precomputation

Prepare for likely next steps:

```python
def anticipate_next_steps(current_action):
    """
    Based on current action, prepare for likely next actions.
    """
    predictions = {
        'file_edit': ['run_tests', 'run_linter', 'git_diff'],
        'test_run': ['read_failure', 'edit_fix', 'rerun_tests'],
        'git_commit': ['git_push', 'create_pr'],
    }

    likely_next = predictions.get(current_action, [])
    for action in likely_next:
        precompute(action)  # Prepare in background
```

### 8.3 Streaming and Early Response

Start responding before full analysis complete:

```
BATCH RESPONSE:
[Wait for all analysis] → [Return everything at once]
User waits: 30 seconds

STREAMING RESPONSE:
[Return initial findings immediately]
[Update as more analysis completes]
User sees progress: feels faster
```

---

## 9. ERROR RECOVERY OPTIMIZATION

### 9.1 Fast Failure Detection

Detect errors early, before wasted work:

```python
def execute_with_fast_fail(steps):
    """
    Check preconditions before executing expensive operations.
    """
    for step in steps:
        # Quick validation first
        if not validate_preconditions(step):
            return Error(f"Precondition failed: {step}")

        # Only then execute expensive operation
        result = execute(step)
        if result.failed:
            return result  # Don't continue on failure
```

### 9.2 Automatic Retry with Backoff

Handle transient failures efficiently:

```python
async def retry_with_backoff(operation, max_retries=3):
    """
    Retry transient failures with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            return await operation()
        except TransientError:
            if attempt == max_retries - 1:
                raise
            await sleep(2 ** attempt)  # 1s, 2s, 4s
```

### 9.3 Fallback Strategies

Have fallback plans for common failures:

```
OPERATION: Search codebase for function
PRIMARY: Grep for function name
FALLBACK 1: Glob for likely file patterns
FALLBACK 2: Agent-based exploration
FALLBACK 3: Ask user for hints

Cascade through fallbacks only on failure
```

---

## 10. MEASUREMENT AND MONITORING

### 10.1 Key Metrics

Track what matters:

```
PERFORMANCE METRICS:
├── Time to first response (user-perceived)
├── Total task completion time
├── Token usage per task type
├── Tool call count per task
├── Error rate and retry count
├── Cache hit rate
└── Parallel execution ratio

TARGET IMPROVEMENTS:
- 50% reduction in average task time
- 40% reduction in token usage
- 80%+ cache hit rate for repeated patterns
```

### 10.2 Continuous Improvement

Learn from every interaction:

```python
def post_task_analysis(task_record):
    """
    Analyze completed task for optimization opportunities.
    """
    analysis = {
        'sequential_calls_that_could_parallelize': [],
        'redundant_tool_calls': [],
        'verbose_outputs': [],
        'cache_misses_that_could_hit': [],
    }

    # Feed back into optimization rules
    update_optimization_rules(analysis)
```

---

## 11. IMPLEMENTATION CHECKLIST

### For Every Task:
- [ ] Identify parallelizable operations before starting
- [ ] Use lightest tool that accomplishes goal
- [ ] Batch multiple tool calls when possible
- [ ] Check cache before expensive operations
- [ ] Minimize output tokens (action over explanation)
- [ ] Stream progress for long operations
- [ ] Have fallback strategies ready

### For Session Optimization:
- [ ] Build understanding hierarchically
- [ ] Cache file contents read in session
- [ ] Recognize and apply recurring patterns
- [ ] Track token usage across tasks
- [ ] Precompute likely next steps

---

## 12. PERFORMANCE ANTI-PATTERNS

```
DON'T: Read entire codebase to find one function
DO: Grep for function name, read only matching file

DON'T: Spawn agent for simple file lookup
DO: Use Glob/Grep directly

DON'T: Explain every step before doing it
DO: Execute, then summarize results

DON'T: Wait for one file read before starting next
DO: Read all needed files in parallel

DON'T: Re-read unchanged files
DO: Cache and reuse within session

DON'T: Generate verbose progress updates
DO: Show minimal status, detailed results

DON'T: Retry failed operations with same approach
DO: Try alternative strategies on failure
```

---

**Remember**: Performance is about respect—respect for the user's time, respect for resource costs, respect for the efficiency gains compounding across every interaction. Fast is kind. Efficient is professional.

Ship fast. Ship cheap. Ship often.
