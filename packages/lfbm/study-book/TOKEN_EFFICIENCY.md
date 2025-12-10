# Token Efficiency & Format Discipline

## The Problem

Elle wastes tokens on:
1. Verbose preambles ("Let me analyze this...")
2. Malformed JSON (Guardian has to fix)
3. Unnecessary markdown in JSON values
4. Repeated information across keys

**Goal**: Perfect JSON on first generation. Zero Guardian intervention.

---

## Training Data Format Rules

### 1. NO Preambles
```
❌ BAD:
"I'll analyze the geopolitical situation. Based on the CIC framework..."

✅ GOOD:
{"political": "SUPERCOOLED phase. Ukraine 85% risk..."}
```

### 2. NO Markdown in JSON Values
```
❌ BAD:
{"summary": "## Global Assessment\n\n**Key findings**:\n- Point 1..."}

✅ GOOD:
{"summary": "Global assessment: SUPERCOOLED PHASE. Ukraine 85%, Iran 65%..."}
```

### 3. Compact Numbers
```
❌ BAD:
"The risk level is approximately 0.7632145..."

✅ GOOD:
"Risk 0.76, Φ=0.82"
```

### 4. NO Redundancy Across Keys
```
❌ BAD:
{"political": "Ukraine at 85% risk...",
 "summary": "Ukraine remains at 85% risk..."}  // REPEATED

✅ GOOD:
{"political": "Ukraine 85%, Taiwan 55%. SUPERCOOLED phase.",
 "summary": "3 nations critical. CIC F=0.72. Action: monitor UKR, TWN."}
```

---

## Training Data Generator Updates

Add these to `prepare_data_ultimate.py`:

```python
# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN EFFICIENCY CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum token budget per field
MAX_TOKENS = {
    'political': 80,
    'economic': 60,
    'security': 80,
    'military': 60,
    'financial': 50,
    'cyber': 50,
    'cic_assessment': 100,
    'phase_assessment': 80,
    'confidence_bounds': 60,
    'historical_parallel': 80,
    'summary': 100,
    'nsm': 80,
}

# Forbidden patterns (waste tokens)
FORBIDDEN_PATTERNS = [
    r'^(I |Let me |Based on |According to )',  # Preambles
    r'\*\*.*\*\*',  # Bold markdown
    r'^#{1,6} ',    # Headers
    r'```',          # Code blocks
    r'\n{2,}',       # Double newlines
    r'approximately|roughly|about',  # Hedging
    r'it is (important|worth noting|clear) that',  # Filler
]

# Abbreviation map (save tokens)
ABBREVIATIONS = {
    'approximately': '~',
    'percentage': '%',
    'temperature': 'T',
    'critical temperature': 'T_c',
    'integrated information': 'Φ',
    'representation entropy': 'H',
    'causality': 'C',
    'confidence': 'conf',
    'assessment': 'assess',
    'monitoring': 'monitor',
    'recommendation': 'rec',
}


def enforce_token_budget(text: str, max_tokens: int, tokenizer=None) -> str:
    """Truncate text to token budget, preserving meaning."""
    if tokenizer is None:
        # Rough estimate: 4 chars per token
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            return text[:max_chars-3] + "..."
        return text

    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        truncated = tokenizer.decode(tokens[:max_tokens-1])
        return truncated + "..."
    return text


def apply_abbreviations(text: str) -> str:
    """Replace verbose phrases with abbreviations."""
    for verbose, abbrev in ABBREVIATIONS.items():
        text = text.replace(verbose, abbrev)
    return text


def validate_format(output: dict) -> Tuple[bool, List[str]]:
    """Check if output follows format rules."""
    errors = []

    for key, value in output.items():
        if not isinstance(value, str):
            errors.append(f"{key}: not a string")
            continue

        # Check forbidden patterns
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                errors.append(f"{key}: contains forbidden pattern '{pattern}'")

        # Check token budget
        if key in MAX_TOKENS:
            estimated_tokens = len(value) / 4
            if estimated_tokens > MAX_TOKENS[key]:
                errors.append(f"{key}: exceeds {MAX_TOKENS[key]} token budget")

    return len(errors) == 0, errors
```

---

## System Prompt Update

Add to Elle's system prompt:

```
═══════════════════════════════════════════════════════════════════════════
TOKEN EFFICIENCY DOCTRINE
═══════════════════════════════════════════════════════════════════════════

You are a HIGH-FREQUENCY intelligence system. Every token costs compute.

RULES:
1. RAW JSON ONLY. No markdown, no preambles, no explanations.
2. ABBREVIATE: Φ not "integrated information", T_c not "critical temperature"
3. COMPACT NUMBERS: 0.76 not 0.7632145, 85% not "approximately 85 percent"
4. NO REDUNDANCY: Each key adds NEW information. Never repeat across keys.
5. ACTION-ORIENTED: End with concrete recommendation, not vague "monitoring"

TOKEN BUDGETS (HARD LIMITS):
  political: 80 tokens
  economic: 60 tokens
  security: 80 tokens
  summary: 100 tokens
  nsm: 80 tokens

FORBIDDEN:
  ❌ "I will analyze..." "Let me..." "Based on..."
  ❌ **bold** or ## headers or ``` code blocks
  ❌ "approximately" "roughly" "it appears that"
  ❌ Repeating information from previous keys

You generate JSON. Guardian validates. Minimize Guardian's workload.
```

---

## Format Enforcement at Inference

```python
def elle_generate(model, tokenizer, prompt: str) -> dict:
    """Generate with format enforcement."""

    # Force JSON start
    forced_prefix = '{"'

    # Bad token IDs to suppress
    bad_tokens = tokenizer.encode("Let me I will Based on ```")

    output = model.generate(
        tokenizer.encode(prompt + forced_prefix, return_tensors="pt"),
        max_new_tokens=800,
        bad_words_ids=[[t] for t in bad_tokens],
        # Force closing brace
        eos_token_id=tokenizer.encode("}")[0],
    )

    response = tokenizer.decode(output[0])

    # Parse JSON
    try:
        # Find JSON bounds
        start = response.find('{')
        end = response.rfind('}') + 1
        json_str = response[start:end]
        return json.loads(json_str)
    except:
        # Guardian fallback
        return {"error": "parse_failed", "raw": response}
```

---

## DB Entry Format

For storing articles/signals:

```typescript
interface DBEntry {
  // REQUIRED - minimal viable record
  id: string;          // UUID
  ts: number;          // Unix timestamp (not ISO string - saves bytes)
  src: string;         // Source code: "gdelt", "rss", "api"

  // CONTENT - compact
  title: string;       // Max 200 chars
  body?: string;       // Max 2000 chars, nullable

  // SIGNALS - numeric only
  tone: number;        // -10 to 10, 2 decimal places
  risk: number;        // 0 to 1, 2 decimal places

  // CLASSIFICATION - enum codes
  cat: string;         // Category code: "pol", "econ", "sec", "mil"
  geo: string;         // ISO country code: "UKR", "USA", "CHN"
  phase?: string;      // "CRYS", "SUPER", "NUC", "PLAS", "ANN"

  // CIC - precomputed
  phi?: number;        // Integrated information
  entropy?: number;    // Representation entropy
  cic_f?: number;      // F[T] value
}
```

**Size comparison**:
- Verbose: ~2KB per entry
- Compact: ~400 bytes per entry
- **5x storage savings**

---

## Validation Contract

Guardian checks:

```python
def guardian_validate(entry: dict) -> Tuple[bool, dict]:
    """
    Minimal validation. Elle should pass 99%+ first try.

    Returns:
        (is_valid, corrected_entry or errors)
    """
    required = ['id', 'ts', 'src', 'title', 'tone', 'risk', 'cat', 'geo']

    # Check required fields
    missing = [k for k in required if k not in entry]
    if missing:
        return False, {"error": "missing_fields", "fields": missing}

    # Type coercion (cheap)
    if isinstance(entry.get('ts'), str):
        entry['ts'] = int(datetime.fromisoformat(entry['ts']).timestamp())

    if isinstance(entry.get('risk'), str):
        entry['risk'] = float(entry['risk'].rstrip('%')) / 100

    # Range validation
    if not 0 <= entry['risk'] <= 1:
        entry['risk'] = max(0, min(1, entry['risk']))

    if not -10 <= entry['tone'] <= 10:
        entry['tone'] = max(-10, min(10, entry['tone']))

    return True, entry
```

---

## Training Data Examples

### Good Example (Token Efficient)
```json
{
  "political": "SUPERCOOLED. UKR 85%, TWN 55%, IRN 65%. 3 above threshold. Coalition fragmentation accelerating.",
  "economic": "Markets stable. VIX 18.2. USD strength persists. Supply chains nominal.",
  "security": "Alert count: 12. GDELT tone: -2.3. APT activity: baseline. No imminent threats.",
  "cic_assessment": "F[T]=0.72 (Φ=0.85, H=0.43, C=0.68). High integration, moderate entropy. UIPT: STABLE.",
  "phase_assessment": "SUPERCOOLED: T=0.42, Ψ=0.61, ν=0.23. Metastable. Watch for nucleation triggers.",
  "confidence_bounds": "Current: 0.78. 3mo: 0.56. 6mo: 0.41. Cluster cohesion: 0.92.",
  "summary": "3 critical nations, SUPERCOOLED phase, F=0.72. Elevated but stable. Monitor UKR escalation.",
  "nsm": "WATCH: UKR, IRN. ACTION: Update contingencies. Next review: 72h."
}
```

Token count: ~180 tokens total
Format violations: 0
Guardian intervention: NONE

---

## Metrics to Track

| Metric | Target | Current |
|--------|--------|---------|
| JSON parse success | 99.5% | ? |
| Token budget compliance | 95% | ? |
| Guardian intervention rate | <1% | ? |
| Avg tokens per briefing | <250 | ? |
| Forbidden pattern hits | 0 | ? |

Train until these targets are met.
