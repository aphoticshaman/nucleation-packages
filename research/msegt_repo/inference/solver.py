import re
import torch
from typing import List, Optional

from msegt.models.eagle_adapter import EagleWrapper
from msegt.models.prm_head import PRMHead, prm_probs
from msegt.models.symbolic_modules import classify_problem, symbolic_sanity_score

ANSWER_MIN, ANSWER_MAX = 0, 99999
FALLBACK_ANSWER = 0

def extract_answer(text: str) -> Optional[int]:
    boxes = re.findall(r"\\boxed\{(-?\d+)\}", text)
    if boxes:
        try:
            return int(boxes[-1])
        except ValueError:
            pass
    nums = re.findall(r"-?\d+", text)
    if nums:
        try:
            return int(nums[-1])
        except ValueError:
            pass
    return None

def generate_candidates(wrapper: EagleWrapper, problem: str, n_samples: int = 4, max_tokens: int = 512) -> List[str]:
    base_prompt = (
        "You are an expert olympiad mathematician. "
        "Solve this problem step-by-step and give the final integer answer in \\boxed{}."
        "\n\nProblem:\n" + problem + "\n"
    )
    outs = []
    for _ in range(n_samples):
        outs.append(wrapper.generate_text(base_prompt, max_new_tokens=max_tokens, temperature=0.4))
    return outs

@torch.no_grad()
def score_candidate(wrapper: EagleWrapper, prm_head: PRMHead, problem: str, candidate: str) -> float:
    lines = [ln.strip() for ln in candidate.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    texts = [problem + " [STEP] " + s for s in lines]
    embeddings = wrapper.forward_for_prm(texts)
    probs = prm_probs(prm_head, embeddings).cpu().numpy()
    avg_prm = float(probs.mean())
    problem_type = classify_problem(problem)
    sym_score = sum(symbolic_sanity_score(s, problem_type) for s in lines) / len(lines)
    return avg_prm + 0.1 * sym_score

def solve_with_prm(wrapper: EagleWrapper, prm_head: PRMHead, problem: str, n_samples: int = 4) -> int:
    candidates = generate_candidates(wrapper, problem, n_samples=n_samples)
    scored = [(score_candidate(wrapper, prm_head, problem, c), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_text = scored[0]
    ans = extract_answer(best_text)
    if ans is None:
        return FALLBACK_ANSWER
    return max(ANSWER_MIN, min(ANSWER_MAX, int(ans)))
