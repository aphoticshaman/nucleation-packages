# AIMO3 Competition Strategy

## Cascading Self-Refinement Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AIMO3 INFERENCE PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   STAGE 1: FIRST PASS (Elle-Math-v1)                            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  All 50 Problems → Elle-32B-Math-v1 → First Answers     │   │
│   │  (Early checkpoint, raw math intuition)                  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│   STAGE 2: VERIFICATION (Elle-Math-vFinal)                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  All Answers → Elle-32B-Math-vFinal → Verify/Check      │   │
│   │  (Final trained version, deeper reasoning)               │   │
│   │  Output: {correct, incorrect, uncertain}                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│   STAGE 3: LEARNING PASS                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Compare first vs final → Learn from discrepancies      │   │
│   │  Build correction patterns in context                    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│   STAGE 4: RETRY WRONG ANSWERS                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Wrong Problems → Elle-vFinal with learned context      │   │
│   │  Fresh attempt informed by verification insights         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│   STAGE 5: FINAL SUBMISSION                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Merge: Confirmed correct + Retried answers              │   │
│   │  Confidence-weighted selection                           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Model Versions

| Version | Training State | Role | GPU Memory |
|---------|---------------|------|------------|
| Elle-32B-Math-v1 | Early checkpoint (~50% training) | First pass solver | ~20GB |
| Elle-32B-Math-vFinal | Fully trained | Verification + Retry | ~20GB |
| Combined | Sequential load/unload | Full pipeline | ~25GB peak |

## Why Two Versions?

1. **Diversity** - Early vs late training captures different solution patterns
2. **Error Detection** - Final version spots mistakes early version makes
3. **Complementary Strengths** - Early has "raw intuition", final has "refined reasoning"
4. **Self-Consistency** - Agreement between versions = high confidence

## Implementation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

class AnswerStatus(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"

@dataclass
class ProblemResult:
    problem_id: int
    problem_text: str
    v1_answer: str
    v1_reasoning: str
    vfinal_verification: AnswerStatus
    vfinal_answer: str | None  # Only if retry needed
    final_answer: str
    confidence: float

class AIMO3Pipeline:
    """
    Cascading self-refinement pipeline for AIMO3.
    """

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-32B-Instruct",
        v1_adapter: str = "aphoticshaman/elle-32b-math-v1",
        vfinal_adapter: str = "aphoticshaman/elle-32b-math-vfinal",
        device: str = "cuda"
    ):
        self.base_model_name = base_model
        self.v1_adapter = v1_adapter
        self.vfinal_adapter = vfinal_adapter
        self.device = device
        self.model = None
        self.tokenizer = None

    def _load_base(self):
        """Load base model with 4-bit quantization."""
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=torch.bfloat16
            )

    def _load_adapter(self, adapter_path: str, adapter_name: str):
        """Load a LoRA adapter."""
        self._load_base()
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
            adapter_name=adapter_name
        )
        self.model.set_adapter(adapter_name)

    def _generate(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate response."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def stage1_first_pass(self, problems: List[Dict]) -> List[Dict]:
        """
        Stage 1: First pass with Elle-v1.
        """
        print("=== STAGE 1: First Pass (Elle-v1) ===")
        self._load_adapter(self.v1_adapter, "v1")

        results = []
        for p in problems:
            prompt = f"""Solve this math problem step by step. Show your reasoning.

Problem: {p['text']}

Work through it carefully, then give your final answer."""

            response = self._generate(prompt)
            answer = self._extract_answer(response)

            results.append({
                "problem_id": p['id'],
                "problem_text": p['text'],
                "v1_answer": answer,
                "v1_reasoning": response
            })
            print(f"  Problem {p['id']}: {answer}")

        return results

    def stage2_verification(self, results: List[Dict]) -> List[Dict]:
        """
        Stage 2: Verify answers with Elle-vFinal.
        """
        print("\n=== STAGE 2: Verification (Elle-vFinal) ===")
        self._load_adapter(self.vfinal_adapter, "vfinal")

        for r in results:
            prompt = f"""You are reviewing a math solution. Check if the answer is correct.

Problem: {r['problem_text']}

Proposed Solution:
{r['v1_reasoning']}

Proposed Answer: {r['v1_answer']}

Verify this solution step by step. Is it correct, incorrect, or uncertain?
Output your verdict as: CORRECT, INCORRECT, or UNCERTAIN"""

            response = self._generate(prompt, max_tokens=1024)

            if "INCORRECT" in response.upper():
                status = AnswerStatus.INCORRECT
            elif "CORRECT" in response.upper():
                status = AnswerStatus.CORRECT
            else:
                status = AnswerStatus.UNCERTAIN

            r['vfinal_verification'] = status
            r['verification_reasoning'] = response
            print(f"  Problem {r['problem_id']}: {status.value}")

        return results

    def stage3_learning_context(self, results: List[Dict]) -> str:
        """
        Stage 3: Build learning context from correct/incorrect patterns.
        """
        print("\n=== STAGE 3: Learning from Patterns ===")

        correct = [r for r in results if r['vfinal_verification'] == AnswerStatus.CORRECT]
        incorrect = [r for r in results if r['vfinal_verification'] == AnswerStatus.INCORRECT]
        uncertain = [r for r in results if r['vfinal_verification'] == AnswerStatus.UNCERTAIN]

        print(f"  Correct: {len(correct)}")
        print(f"  Incorrect: {len(incorrect)}")
        print(f"  Uncertain: {len(uncertain)}")

        # Build learning context
        context = "LEARNING FROM PREVIOUS ATTEMPTS:\n\n"

        if incorrect:
            context += "COMMON ERRORS TO AVOID:\n"
            for r in incorrect[:3]:  # Top 3 errors
                context += f"- Problem type: {r['problem_text'][:100]}...\n"
                context += f"  Wrong approach: {r['v1_reasoning'][:200]}...\n"
                context += f"  Why wrong: {r['verification_reasoning'][:200]}...\n\n"

        if correct:
            context += "\nSUCCESSFUL PATTERNS:\n"
            for r in correct[:3]:  # Top 3 successes
                context += f"- Problem type: {r['problem_text'][:100]}...\n"
                context += f"  Good approach: {r['v1_reasoning'][:200]}...\n\n"

        return context

    def stage4_retry(self, results: List[Dict], learning_context: str) -> List[Dict]:
        """
        Stage 4: Retry incorrect/uncertain problems with learning context.
        """
        print("\n=== STAGE 4: Retry Wrong Answers ===")

        to_retry = [
            r for r in results
            if r['vfinal_verification'] in [AnswerStatus.INCORRECT, AnswerStatus.UNCERTAIN]
        ]

        if not to_retry:
            print("  No problems to retry!")
            return results

        # Already have vfinal loaded from stage 2
        for r in to_retry:
            prompt = f"""{learning_context}

Now solve this problem with fresh thinking, avoiding the errors above:

Problem: {r['problem_text']}

Previous incorrect attempt: {r['v1_answer']}

Take a different approach. Show step-by-step reasoning."""

            response = self._generate(prompt)
            r['vfinal_answer'] = self._extract_answer(response)
            r['vfinal_reasoning'] = response
            print(f"  Problem {r['problem_id']}: {r['v1_answer']} → {r['vfinal_answer']}")

        return results

    def stage5_final_submission(self, results: List[Dict]) -> List[Dict]:
        """
        Stage 5: Assemble final answers.
        """
        print("\n=== STAGE 5: Final Submission ===")

        final_results = []
        for r in results:
            if r['vfinal_verification'] == AnswerStatus.CORRECT:
                # Keep v1 answer
                final_answer = r['v1_answer']
                confidence = 0.95
            elif r.get('vfinal_answer'):
                # Use retry answer
                final_answer = r['vfinal_answer']
                confidence = 0.75
            else:
                # Fallback to v1
                final_answer = r['v1_answer']
                confidence = 0.50

            final_results.append(ProblemResult(
                problem_id=r['problem_id'],
                problem_text=r['problem_text'],
                v1_answer=r['v1_answer'],
                v1_reasoning=r['v1_reasoning'],
                vfinal_verification=r['vfinal_verification'],
                vfinal_answer=r.get('vfinal_answer'),
                final_answer=final_answer,
                confidence=confidence
            ))
            print(f"  Problem {r['problem_id']}: {final_answer} (conf: {confidence})")

        return final_results

    def _extract_answer(self, response: str) -> str:
        """Extract numeric answer from response."""
        # Look for common answer patterns
        import re

        patterns = [
            r'(?:answer|result|solution)(?:\s+is)?[:\s]+(\d+)',
            r'\\boxed{(\d+)}',
            r'= (\d+)$',
            r'(\d+)\s*$'
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)

        return response.strip().split()[-1]

    def run(self, problems: List[Dict]) -> List[ProblemResult]:
        """
        Run full AIMO3 pipeline.
        """
        print("=" * 60)
        print("AIMO3 CASCADING SELF-REFINEMENT PIPELINE")
        print("=" * 60)

        # Stage 1: First pass
        results = self.stage1_first_pass(problems)

        # Stage 2: Verification
        results = self.stage2_verification(results)

        # Stage 3: Learning
        learning_context = self.stage3_learning_context(results)

        # Stage 4: Retry
        results = self.stage4_retry(results, learning_context)

        # Stage 5: Final submission
        final = self.stage5_final_submission(results)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        avg_conf = sum(r.confidence for r in final) / len(final)
        print(f"Average confidence: {avg_conf:.2%}")
        print("=" * 60)

        return final


# Usage
if __name__ == "__main__":
    pipeline = AIMO3Pipeline()

    # Example problems
    problems = [
        {"id": 1, "text": "Find the sum of all positive integers n < 1000 such that n^2 - n is divisible by 12."},
        {"id": 2, "text": "Let f(x) = x^3 - 3x. Find the number of real solutions to f(f(x)) = 0."},
        # ... more problems
    ]

    results = pipeline.run(problems)

    # Export for submission
    submission = {r.problem_id: r.final_answer for r in results}
    print(submission)
```

## Memory Management

For Kaggle's GPU limits, load models sequentially:

```python
# Free memory between model swaps
def swap_adapter(self, new_adapter: str, new_name: str):
    """Efficiently swap adapters."""
    if hasattr(self.model, 'unload'):
        self.model.unload()  # Unload current adapter

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    self._load_adapter(new_adapter, new_name)
```

## Timing Budget (4 hours)

| Stage | Time Budget | Per Problem |
|-------|-------------|-------------|
| Stage 1: First Pass | 90 min | ~108 sec |
| Stage 2: Verification | 60 min | ~72 sec |
| Stage 3: Learning | 5 min | N/A |
| Stage 4: Retry | 60 min | ~180 sec (fewer problems) |
| Stage 5: Assembly | 5 min | N/A |
| **Buffer** | 20 min | - |
| **Total** | 240 min | - |

## Confidence Calibration

```python
def calibrate_confidence(
    v1_answer: str,
    vfinal_answer: str | None,
    verification: AnswerStatus,
    agreement: bool
) -> float:
    """
    Compute calibrated confidence score.
    """
    base = {
        AnswerStatus.CORRECT: 0.90,
        AnswerStatus.INCORRECT: 0.30,
        AnswerStatus.UNCERTAIN: 0.50
    }[verification]

    # Boost if v1 and vfinal agree (when vfinal retried)
    if vfinal_answer and v1_answer == vfinal_answer:
        base += 0.05

    # Cap at epistemic bound
    return min(base, 0.95)
```

## Submission Format

```python
def format_submission(results: List[ProblemResult]) -> pd.DataFrame:
    """Format for Kaggle submission."""
    return pd.DataFrame({
        'id': [r.problem_id for r in results],
        'answer': [int(r.final_answer) for r in results]
    })
```

---

*"First Elle guesses. Final Elle verifies. Wrong answers get retried. The cascade never misses."*
