# The 12-Step Claude Code Guide for Ryan
## Building Competition-Ready ARC Prize 2025 Submissions That Don't Fail

**Target**: Single `.py` file → `.ipynb` → Valid `submission.json` in 5-7 hours  
**Success**: No syntax errors, no format errors, no disqualification  
**Philosophy**: Official docs > clever algorithms. Validate > innovate.

---

## Step 0: The Mindset Reset

**Stop being a researcher. Start being a competition engineer.**

Your past submissions failed because:
- ❌ Didn't read official submission format specification
- ❌ Didn't validate JSON schema before submitting
- ❌ Didn't test full pipeline end-to-end locally
- ❌ Assumed things would "just work" in Kaggle
- ❌ Focused on algorithm sophistication over basic correctness

**New mindset checklist:**
```
□ Rules are absolute law, not suggestions
□ Validation comes before algorithms
□ Test realistically with full dataset
□ Never let exceptions kill the submission
□ Log everything for forensic analysis
```

**Time allocation:**
- 30% reading official docs & studying winners
- 40% building validation infrastructure & error handling  
- 20% actual solving algorithm
- 10% testing, debugging, submission prep

**Critical truth**: A valid submission with 5% accuracy beats a crashing submission with theoretical 95% accuracy.

---

## Step 1: Read Official Docs FIRST (30 min)

**DO THIS BEFORE WRITING ANY CODE. SERIOUSLY.**

### Required reading in order:

1. **ARC Prize 2025 Official Rules**  
   URL: `https://arcprize.org/compete`  
   Focus: Competition timeline, submission limits, disqualification criteria

2. **Kaggle Competition Overview**  
   URL: `https://www.kaggle.com/competitions/arc-prize-2025/overview`  
   Focus: Evaluation metrics, scoring, leaderboard rules

3. **Kaggle Data Description**  
   URL: `https://www.kaggle.com/competitions/arc-prize-2025/data`  
   Focus: File paths, data structure, what files are provided

4. **Submission Format Specification**  
   URL: `https://www.kaggle.com/competitions/arc-prize-2025/overview/evaluation`  
   Focus: EXACT JSON schema required, common format errors

5. **Kaggle Notebooks FAQ**  
   URL: `https://www.kaggle.com/docs/notebooks`  
   Focus: Time limits, resource constraints, file I/O patterns

### Critical information to extract:

```python
# Write these down explicitly:
SUBMISSION_SCHEMA = {
    "task_id": [
        {"attempt_1": [[int]], "attempt_2": [[int]]},  # Format?
        # OR
        [[[int]], [[int]]]  # Alternative format?
    ]
}

REQUIRED_PATHS = {
    "test_data": "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json",
    "training_data": "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json",
    "output": "/kaggle/working/submission.json"
}

CONSTRAINTS = {
    "max_time": "9 hours",
    "attempts_per_task": 2,
    "submissions_per_day": 1  # DON'T WASTE THEM
}
```

**Action item**: Open a text file, copy-paste the EXACT submission format from Kaggle docs.

---

## Step 2: Study Winning Notebooks (1 hour)

**Learn from people who already solved the format problem.**

### Find public notebooks:

1. Go to Kaggle competition page
2. Click "Code" tab
3. Filter by: "Most Votes" or "Highest Scoring"
4. Study top 3-5 notebooks

### What to extract:

**A) How they load data:**
```python
# Copy their EXACT data loading pattern
import json

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

# They probably handle both formats:
test_data = load_data('/kaggle/input/arc-prize-2025/...')
```

**B) How they structure predictions:**
```python
# Copy their EXACT submission format
submission = {}
for task_id, task in test_data.items():
    # How do they make 2 attempts?
    # What's the exact structure?
    submission[task_id] = [
        {"attempt_1": attempt1, "attempt_2": attempt2}
        # OR
        # [attempt1, attempt2]
    ]
```

**C) How they handle errors:**
```python
# They probably do this:
def solve_task(task):
    try:
        return actual_solving_logic(task)
    except Exception as e:
        print(f"Error: {e}")
        return default_fallback()  # NEVER crash
```

**D) How they validate:**
```python
# Do they validate before saving?
def validate_submission(submission):
    assert len(submission) == 240  # Or whatever
    for task_id, attempts in submission.items():
        assert len(attempts) == 2
        # etc
```

**Action item**: Create a file called `WINNING_PATTERNS.py` with copy-pasted snippets.

---

## Step 3: Build Validator First (1 hour)

**Write validation code BEFORE solving code.**

### Create `validator.py`:

```python
import json
from typing import Dict, List

def validate_grid(grid: List[List[int]]) -> bool:
    """Validate a single grid."""
    if not grid or not isinstance(grid, list):
        return False
    
    # Must be 2D list
    if not all(isinstance(row, list) for row in grid):
        return False
    
    # All rows same length
    if len(set(len(row) for row in grid)) != 1:
        return False
    
    # All values 0-9
    for row in grid:
        if not all(isinstance(x, int) and 0 <= x <= 9 for x in row):
            return False
    
    return True

def validate_attempt(attempt: List[List[int]]) -> bool:
    """Validate one attempt (a single grid)."""
    return validate_grid(attempt)

def validate_task_prediction(prediction) -> bool:
    """Validate predictions for one task."""
    # Check if it's the dict format
    if isinstance(prediction, dict):
        if "attempt_1" not in prediction or "attempt_2" not in prediction:
            return False
        return (validate_attempt(prediction["attempt_1"]) and 
                validate_attempt(prediction["attempt_2"]))
    
    # Check if it's the list format
    elif isinstance(prediction, list):
        if len(prediction) != 2:
            return False
        return validate_attempt(prediction[0]) and validate_attempt(prediction[1])
    
    return False

def validate_submission(submission: Dict) -> tuple[bool, str]:
    """Validate entire submission.json structure."""
    
    # Must be a dict
    if not isinstance(submission, dict):
        return False, "Submission must be a dictionary"
    
    # Must have correct number of tasks
    if len(submission) != 240:  # Check actual number from docs
        return False, f"Expected 240 tasks, got {len(submission)}"
    
    # Validate each task
    for task_id, prediction in submission.items():
        if not isinstance(task_id, str):
            return False, f"Task ID must be string, got {type(task_id)}"
        
        if not validate_task_prediction(prediction):
            return False, f"Invalid prediction for task {task_id}"
    
    return True, "Valid submission"

def validate_submission_file(filepath: str = "submission.json") -> bool:
    """Load and validate submission.json file."""
    try:
        with open(filepath, 'r') as f:
            submission = json.load(f)
        
        is_valid, message = validate_submission(submission)
        
        if is_valid:
            print(f"✅ VALID: {message}")
            print(f"   Tasks: {len(submission)}")
            print(f"   File size: {os.path.getsize(filepath):,} bytes")
            return True
        else:
            print(f"❌ INVALID: {message}")
            return False
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Test it immediately
if __name__ == "__main__":
    # Test with dummy data
    test_submission = {
        "task_001": [[[1, 0], [0, 1]], [[0, 1], [1, 0]]],
        # ... 239 more tasks
    }
    is_valid, msg = validate_submission(test_submission)
    print(f"Test: {msg}")
```

**Action item**: Run this validator on sample_submission.json from Kaggle to verify it works.

---

## Step 4: Build Safe Default Generator (30 min)

**Create a function that ALWAYS returns valid predictions, even if they're wrong.**

### Create `safe_defaults.py`:

```python
import random
from typing import List, Tuple

def get_input_shape(task: dict) -> Tuple[int, int]:
    """Get output shape from first test input."""
    try:
        test_input = task['test'][0]['input']
        return len(test_input), len(test_input[0])
    except:
        return 3, 3  # Fallback to 3x3

def generate_safe_default(task: dict) -> List[List[int]]:
    """
    Generate a safe default prediction.
    
    Strategy priority:
    1. Copy first test input (safest)
    2. Copy first training output
    3. Generate blank grid
    """
    try:
        # Strategy 1: Copy test input
        return task['test'][0]['input']
    except:
        pass
    
    try:
        # Strategy 2: Copy training output
        return task['train'][0]['output']
    except:
        pass
    
    # Strategy 3: Blank grid
    rows, cols = get_input_shape(task)
    return [[0 for _ in range(cols)] for _ in range(rows)]

def make_two_attempts(task: dict) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Make 2 attempts for a task.
    Attempt 1: Default strategy
    Attempt 2: Slight variation (or same)
    """
    attempt1 = generate_safe_default(task)
    
    # For attempt 2, try a simple variation
    attempt2 = generate_safe_default(task)
    
    return attempt1, attempt2

# Test it
if __name__ == "__main__":
    dummy_task = {
        'test': [{'input': [[1, 0], [0, 1]]}],
        'train': [{'output': [[0, 1], [1, 0]]}]
    }
    
    a1, a2 = make_two_attempts(dummy_task)
    print(f"Attempt 1: {a1}")
    print(f"Attempt 2: {a2}")
```

**Critical principle**: This code NEVER throws exceptions. It ALWAYS returns something valid.

---

## Step 5: Build End-to-End Skeleton (30 min)

**Wire everything together BEFORE adding smart algorithms.**

### Create `arc_solver_skeleton.py`:

```python
import json
import os
from typing import Dict
from validator import validate_submission_file
from safe_defaults import make_two_attempts

# Paths
TEST_PATH = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
TRAIN_PATH = '/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json'
OUTPUT_PATH = '/kaggle/working/submission.json'

def load_data(path: str) -> Dict:
    """Load JSON data with error handling."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return {}

def solve_task(task_id: str, task: dict) -> dict:
    """
    Solve a single task.
    Returns: {"attempt_1": grid, "attempt_2": grid}
    """
    try:
        # TODO: Add smart solving logic here
        # For now, use safe defaults
        attempt1, attempt2 = make_two_attempts(task)
        
        return {
            "attempt_1": attempt1,
            "attempt_2": attempt2
        }
    except Exception as e:
        print(f"❌ Error solving {task_id}: {e}")
        # Fallback to safe defaults
        attempt1, attempt2 = make_two_attempts(task)
        return {"attempt_1": attempt1, "attempt_2": attempt2}

def main():
    """Main pipeline."""
    print("="*70)
    print("ARC Prize 2025 Solver - Safe Skeleton")
    print("="*70)
    
    # Load test data
    print("\n📂 Loading test data...")
    test_data = load_data(TEST_PATH)
    print(f"   Loaded {len(test_data)} tasks")
    
    if len(test_data) == 0:
        print("❌ No test data loaded! Check paths.")
        return
    
    # Solve all tasks
    print("\n🧠 Solving tasks...")
    submission = {}
    
    for i, (task_id, task) in enumerate(test_data.items(), 1):
        if i % 50 == 0:
            print(f"   Progress: {i}/{len(test_data)}")
        
        submission[task_id] = solve_task(task_id, task)
    
    print(f"   ✅ Solved {len(submission)} tasks")
    
    # Save submission
    print("\n💾 Saving submission...")
    try:
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(submission, f)
        print(f"   ✅ Saved to {OUTPUT_PATH}")
    except Exception as e:
        print(f"   ❌ Error saving: {e}")
        return
    
    # Validate
    print("\n✅ Validating submission...")
    is_valid = validate_submission_file(OUTPUT_PATH)
    
    if is_valid:
        print("\n" + "="*70)
        print("🎉 SUCCESS! Submission is valid and ready!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ VALIDATION FAILED - Fix errors above")
        print("="*70)

if __name__ == "__main__":
    main()
```

**Action item**: Run this locally with dummy data to verify it completes without crashing.

---

## Step 6: Study Common Kaggle Errors (20 min)

**Learn from others' mistakes (documented in forums).**

### Top 10 ways to get disqualified:

1. **Wrong submission format** (dict vs list structure)
2. **Missing task IDs** (not all 240 tasks)
3. **Wrong number of attempts** (not exactly 2)
4. **Invalid grid values** (not integers 0-9)
5. **Ragged arrays** (rows with different lengths)
6. **Wrong file path** (saving to wrong location)
7. **Timeout** (exceeding 9-hour limit)
8. **Memory error** (exceeding RAM limits)
9. **Import errors** (using unavailable libraries)
10. **Runtime exceptions** (crashes before submission)

### For each error, add defensive code:

```python
# Error 1: Wrong format
# ALWAYS use the format from winning notebooks

# Error 2: Missing tasks
assert len(submission) == len(test_data), "Missing tasks!"

# Error 3: Wrong attempts
for task_id, pred in submission.items():
    if isinstance(pred, dict):
        assert "attempt_1" in pred and "attempt_2" in pred
    else:
        assert len(pred) == 2

# Error 4: Invalid values
def clamp_grid(grid):
    return [[max(0, min(9, int(x))) for x in row] for row in grid]

# Error 5: Ragged arrays
def fix_ragged(grid):
    max_len = max(len(row) for row in grid)
    return [row + [0]*(max_len - len(row)) for row in grid]

# Error 6: Wrong path
OUTPUT_PATH = '/kaggle/working/submission.json'  # NOT /tmp/ or ~/

# Error 7: Timeout
import time
start_time = time.time()
TIME_LIMIT = 8.5 * 3600  # Leave 30min buffer

def check_timeout():
    if time.time() - start_time > TIME_LIMIT:
        print("⏰ Approaching time limit, wrapping up...")
        return True
    return False

# Error 8: Memory
import gc
gc.collect()  # Periodically free memory

# Error 9: Imports
# Only use: json, os, time, numpy, typing
# Avoid: torch, tensorflow (unless you know they work)

# Error 10: Exceptions
# WRAP EVERYTHING IN try/except with safe fallbacks
```

**Action item**: Add these checks to your skeleton code.

---

## Step 7: Build Minimal Viable Solver (2 hours)

**Add ONE simple solving strategy that works on SOME tasks.**

### Strategy 1: Pattern matching

```python
def simple_pattern_solver(task: dict):
    """
    Try simple pattern matching:
    1. Check if output is horizontal/vertical flip of input
    2. Check if output is color swap
    3. Check if output is size change
    """
    train_examples = task.get('train', [])
    test_input = task['test'][0]['input']
    
    # Learn pattern from training
    for example in train_examples:
        inp = example['input']
        out = example['output']
        
        # Pattern 1: Horizontal flip?
        if is_horizontal_flip(inp, out):
            return horizontal_flip(test_input)
        
        # Pattern 2: Color swap?
        color_map = detect_color_swap(inp, out)
        if color_map:
            return apply_color_swap(test_input, color_map)
        
        # Pattern 3: Size change?
        if detect_size_pattern(inp, out):
            return apply_size_pattern(test_input, inp, out)
    
    # No pattern found, return safe default
    return generate_safe_default(task)

def is_horizontal_flip(inp, out):
    """Check if output is horizontal flip of input."""
    if len(inp) != len(out) or len(inp[0]) != len(out[0]):
        return False
    return all(inp[i] == out[i][::-1] for i in range(len(inp)))

def horizontal_flip(grid):
    """Flip grid horizontally."""
    return [row[::-1] for row in grid]

def detect_color_swap(inp, out):
    """Detect if there's a consistent color mapping."""
    if len(inp) != len(out) or len(inp[0]) != len(out[0]):
        return None
    
    color_map = {}
    for i in range(len(inp)):
        for j in range(len(inp[0])):
            in_color = inp[i][j]
            out_color = out[i][j]
            
            if in_color in color_map:
                if color_map[in_color] != out_color:
                    return None
            else:
                color_map[in_color] = out_color
    
    return color_map

def apply_color_swap(grid, color_map):
    """Apply color mapping to grid."""
    return [[color_map.get(cell, cell) for cell in row] for row in grid]

# Add more strategies as needed...
```

**Key principle**: Start with strategies that are:
- Simple to implement
- Easy to verify
- Work on at least 10-20% of tasks

**Don't** start with:
- Neural networks (overkill for step 1)
- Quantum mechanics (not real)
- "AGI consciousness" (not helpful)

---

## Step 8: Add Comprehensive Logging (30 min)

**Log EVERYTHING so you can debug failed submissions.**

### Create `logger.py`:

```python
import json
import time
from datetime import datetime

class SolverLogger:
    def __init__(self, log_file='solver_log.jsonl'):
        self.log_file = log_file
        self.start_time = time.time()
        self.stats = {
            'total_tasks': 0,
            'successful_solves': 0,
            'fallback_used': 0,
            'errors': [],
            'timing': {}
        }
    
    def log_task(self, task_id, status, strategy, error=None):
        """Log individual task result."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'task_id': task_id,
            'status': status,
            'strategy': strategy,
            'elapsed': time.time() - self.start_time
        }
        
        if error:
            entry['error'] = str(error)
            self.stats['errors'].append({'task_id': task_id, 'error': str(error)})
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        self.stats['total_tasks'] += 1
        if status == 'success':
            self.stats['successful_solves'] += 1
        elif status == 'fallback':
            self.stats['fallback_used'] += 1
    
    def print_summary(self):
        """Print final summary."""
        elapsed = time.time() - self.start_time
        print("\n" + "="*70)
        print("SOLVER SUMMARY")
        print("="*70)
        print(f"Total tasks: {self.stats['total_tasks']}")
        print(f"Successful: {self.stats['successful_solves']}")
        print(f"Fallbacks: {self.stats['fallback_used']}")
        print(f"Errors: {len(self.stats['errors'])}")
        print(f"Total time: {elapsed:.1f}s ({elapsed/3600:.2f}h)")
        
        if self.stats['errors']:
            print("\nError summary:")
            for err in self.stats['errors'][:5]:
                print(f"  - {err['task_id']}: {err['error'][:50]}")

# Use it in main solver
logger = SolverLogger()

def solve_task_with_logging(task_id, task):
    try:
        result = solve_task(task_id, task)
        logger.log_task(task_id, 'success', 'pattern_match')
        return result
    except Exception as e:
        logger.log_task(task_id, 'fallback', 'safe_default', error=e)
        return make_two_attempts(task)
```

**Why this matters**: When your submission scores poorly, you can analyze the log to see which tasks failed and why.

---

## Step 9: Build Local Test Harness (1 hour)

**Test your full pipeline locally before wasting a Kaggle submission.**

### Create `test_harness.py`:

```python
import json
import os
import shutil
from arc_solver_skeleton import main as run_solver
from validator import validate_submission_file

def setup_test_environment():
    """Create local mock Kaggle environment."""
    # Create mock Kaggle directories
    os.makedirs('/tmp/kaggle/input/arc-prize-2025', exist_ok=True)
    os.makedirs('/tmp/kaggle/working', exist_ok=True)
    
    # Copy real data files to mock location
    # (Download these from Kaggle first)
    shutil.copy(
        'real_data/arc-agi_test_challenges.json',
        '/tmp/kaggle/input/arc-prize-2025/'
    )
    
    print("✅ Test environment ready")

def test_full_pipeline():
    """Run full solver pipeline."""
    print("\n" + "="*70)
    print("TESTING FULL PIPELINE")
    print("="*70)
    
    # Temporarily override paths for local testing
    import arc_solver_skeleton
    arc_solver_skeleton.TEST_PATH = '/tmp/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
    arc_solver_skeleton.OUTPUT_PATH = '/tmp/kaggle/working/submission.json'
    
    # Run solver
    try:
        run_solver()
        print("\n✅ Solver completed without crashing")
    except Exception as e:
        print(f"\n❌ Solver crashed: {e}")
        return False
    
    # Check output exists
    if not os.path.exists('/tmp/kaggle/working/submission.json'):
        print("❌ submission.json not created")
        return False
    
    # Validate
    is_valid = validate_submission_file('/tmp/kaggle/working/submission.json')
    
    return is_valid

def test_with_subset(n_tasks=10):
    """Test with small subset for quick iteration."""
    # Load full data
    with open('real_data/arc-agi_test_challenges.json', 'r') as f:
        full_data = json.load(f)
    
    # Take subset
    subset = dict(list(full_data.items())[:n_tasks])
    
    # Save subset
    with open('/tmp/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json', 'w') as f:
        json.dump(subset, f)
    
    # Test
    print(f"\nTesting with {n_tasks} tasks...")
    return test_full_pipeline()

if __name__ == "__main__":
    # Quick test with 10 tasks
    setup_test_environment()
    
    print("\n1️⃣ Quick test (10 tasks)...")
    quick_pass = test_with_subset(n_tasks=10)
    
    if quick_pass:
        print("\n2️⃣ Full test (240 tasks)...")
        full_pass = test_full_pipeline()
        
        if full_pass:
            print("\n" + "="*70)
            print("🎉 ALL TESTS PASSED - READY FOR KAGGLE")
            print("="*70)
        else:
            print("\n❌ Full test failed")
    else:
        print("\n❌ Quick test failed - fix issues first")
```

**Action item**: Run this BEFORE submitting to Kaggle. It will catch 90% of issues.

---

## Step 10: Create Pre-Submission Checklist (10 min)

**A literal checklist to run through before clicking Submit.**

### Create `SUBMISSION_CHECKLIST.md`:

```markdown
# Pre-Submission Checklist

## Code Quality
- [ ] No syntax errors (ran locally without crashes)
- [ ] All imports available in Kaggle
- [ ] No hardcoded paths (uses Kaggle paths)
- [ ] Comprehensive error handling (no crashes)
- [ ] Safe fallbacks for all strategies

## Format Compliance
- [ ] Submission is a dict with 240 keys
- [ ] Each value has exactly 2 attempts
- [ ] All grids are valid (2D list of ints 0-9)
- [ ] No ragged arrays (all rows same length)
- [ ] Validated with validator.py

## Testing
- [ ] Tested locally with full dataset
- [ ] Tested with subset (quick test passed)
- [ ] Validated submission.json locally
- [ ] Checked file size (not suspiciously small/large)
- [ ] Reviewed solver_log.jsonl for errors

## Kaggle Environment
- [ ] Notebook has access to ARC Prize 2025 dataset
- [ ] Output directory is /kaggle/working/
- [ ] Time budget is reasonable (<8 hours)
- [ ] Memory usage is reasonable
- [ ] GPU/TPU not required (or properly configured)

## Double-Check
- [ ] Read latest Kaggle discussions for known issues
- [ ] Checked if format changed recently
- [ ] Verified I'm using correct competition
- [ ] Have submissions remaining today (1/day limit)

## Post-Submission
- [ ] Save notebook version that generated submission
- [ ] Save submission.json locally
- [ ] Save solver_log.jsonl for analysis
- [ ] Note submission time and score
```

**Action item**: Print this and check off each item before submitting.

---

## Step 11: Kaggle-Specific Optimizations (30 min)

**Make code work perfectly in Kaggle's environment.**

### Path handling:

```python
import os

def get_kaggle_paths():
    """Auto-detect if running in Kaggle."""
    if os.path.exists('/kaggle/input'):
        return {
            'test': '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json',
            'train': '/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json',
            'output': '/kaggle/working/submission.json'
        }
    else:
        # Local testing
        return {
            'test': 'data/arc-agi_test_challenges.json',
            'train': 'data/arc-agi_training_challenges.json',
            'output': 'submission.json'
        }

PATHS = get_kaggle_paths()
```

### Time management:

```python
import time

class TimeManager:
    def __init__(self, total_hours=8.5):
        self.start = time.time()
        self.limit = total_hours * 3600
        self.checkpoints = {}
    
    def elapsed(self):
        return time.time() - self.start
    
    def remaining(self):
        return self.limit - self.elapsed()
    
    def should_stop(self):
        return self.remaining() < 600  # Stop with 10min buffer
    
    def checkpoint(self, name):
        self.checkpoints[name] = self.elapsed()
    
    def print_timing(self):
        print("\nTiming breakdown:")
        prev_time = 0
        for name, t in self.checkpoints.items():
            duration = t - prev_time
            print(f"  {name}: {duration/60:.1f}min")
            prev_time = t

timer = TimeManager()

# Use in solver
for task_id, task in test_data.items():
    if timer.should_stop():
        print("⏰ Time limit approaching, finishing up...")
        break
    solve_task(task_id, task)

timer.print_timing()
```

### Memory management:

```python
import gc

def solve_with_memory_management(test_data):
    submission = {}
    
    for i, (task_id, task) in enumerate(test_data.items()):
        submission[task_id] = solve_task(task_id, task)
        
        # Free memory every 50 tasks
        if i % 50 == 0:
            gc.collect()
    
    return submission
```

### Progress tracking:

```python
from datetime import datetime

def solve_with_progress(test_data):
    total = len(test_data)
    submission = {}
    
    print(f"\nStarting solver at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total tasks: {total}")
    
    for i, (task_id, task) in enumerate(test_data.items(), 1):
        submission[task_id] = solve_task(task_id, task)
        
        # Progress updates
        if i % 20 == 0:
            pct = (i / total) * 100
            elapsed = timer.elapsed() / 60
            rate = i / (elapsed / 60)
            remaining = (total - i) / rate if rate > 0 else 0
            
            print(f"[{i:3}/{total}] {pct:5.1f}% | "
                  f"Elapsed: {elapsed:.1f}min | "
                  f"ETA: {remaining:.1f}min")
    
    return submission
```

---

## Step 12: Post-Submission Analysis (Ongoing)

**Learn from each submission to improve the next one.**

### After submitting, immediately:

1. **Download submission.json**
   ```bash
   # From Kaggle notebook output
   # Save locally with timestamp
   mv submission.json submission_2025-11-02_v1.json
   ```

2. **Download solver_log.jsonl**
   ```bash
   # This has all the debugging info
   mv solver_log.jsonl log_2025-11-02_v1.jsonl
   ```

3. **Check leaderboard score**
   - Public score (on public test set)
   - Note which percentile you're in

4. **Analyze failures**
   ```python
   # If score is low, analyze which tasks failed
   def analyze_failed_tasks(submission, test_data):
       # Compare your submission to sample_submission
       # Identify patterns in failures
       pass
   ```

5. **Read discussion forum**
   - What are top solutions doing?
   - Are there common bugs you hit?
   - Any format issues reported?

6. **Plan next submission**
   - What strategy to add next?
   - What bugs to fix?
   - What optimization to try?

### Create improvement tracker:

```markdown
# Submission History

## Submission 1 (2025-11-02)
- **Score**: 0.05
- **Strategy**: Safe defaults only
- **Issues**: None, just low accuracy
- **Next**: Add pattern matching

## Submission 2 (2025-11-03)
- **Score**: 0.12
- **Strategy**: Pattern matching (flip/rotate)
- **Issues**: Some timeouts
- **Next**: Optimize timing, add color swap

## Submission 3 (2025-11-04)
- **Score**: 0.18
- **Strategy**: + color mapping
- **Issues**: None
- **Next**: Add object detection
```

### Common improvement vectors:

1. **More strategies**: Add transformation types you haven't tried
2. **Better pattern recognition**: Improve matching accuracy
3. **Ensemble methods**: Combine multiple strategies
4. **Training data mining**: Extract more patterns from training set
5. **Time optimization**: Solve faster → try more strategies
6. **Error recovery**: Better fallbacks when strategies fail

---

## Final Mindset Mantras

**Before you code:**
1. "What does the official doc say?"
2. "How do winning notebooks do this?"
3. "How will I validate this works?"

**While you code:**
1. "Can this throw an exception? Add try/except."
2. "Is this the exact format required? Double-check."
3. "Will this work in Kaggle? Test the path/import."

**Before you submit:**
1. "Did I test locally with full dataset?"
2. "Did I validate submission.json?"
3. "Did I check the checklist?"

**After you submit:**
1. "What was my score?"
2. "Which tasks failed and why?"
3. "What's the ONE thing to improve next?"

---

## The 80/20 Rule for Competition Success

**80% of your score comes from:**
- Correct submission format (20%)
- No crashes/errors (20%)
- One good solving strategy (20%)
- Proper fallbacks (20%)

**20% of your score comes from:**
- Fancy algorithms (5%)
- Clever optimizations (5%)
- Ensemble methods (5%)
- Novel insights (5%)

**Focus on the 80% first. Perfect the basics before attempting AGI.**

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────┐
│  COMPETITION SURVIVAL CHECKLIST                 │
├─────────────────────────────────────────────────┤
│  □ Read official docs                           │
│  □ Study winning notebooks                      │
│  □ Build validator FIRST                        │
│  □ Test locally BEFORE submitting               │
│  □ Use exact paths: /kaggle/working/...         │
│  □ Handle ALL exceptions                        │
│  □ Validate submission.json                     │
│  □ Check you have 240 tasks × 2 attempts        │
│  □ Verify all grids are valid                   │
│  □ Save logs for debugging                      │
└─────────────────────────────────────────────────┘

GOLDEN RULES:
1. Docs > Intuition
2. Validation > Innovation  
3. Testing > Hoping
4. Fallbacks > Perfection
5. Simple > Clever

REMEMBER: Valid 5% > Crashing 95%
```

---

**You're ready. Go build a submission that WORKS, then make it GOOD.**