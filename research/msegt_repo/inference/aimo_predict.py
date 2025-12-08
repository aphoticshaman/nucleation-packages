from typing import Optional
import torch

from msegt.models.eagle_adapter import EagleWrapper
from msegt.models.prm_head import PRMHead
from msegt.inference.solver import solve_with_prm, extract_answer, FALLBACK_ANSWER

class AIMOModel:
    def __init__(self, model_name_or_path: str, prm_checkpoint: Optional[str] = None, device: str = "cuda"):
        self.wrapper = EagleWrapper(model_name_or_path=model_name_or_path, device=device)
        self.device = device
        if prm_checkpoint is not None:
            state = torch.load(prm_checkpoint, map_location=device)
            self.prm_head = PRMHead(hidden_dim=4096).to(device)
            self.prm_head.load_state_dict(state)
            self.prm_head.eval()
        else:
            self.prm_head = None

    def predict(self, problem: str) -> int:
        if self.prm_head is None:
            # naive single-shot fallback
            base = (
                "Solve the following problem and return only the final integer answer in \\boxed{}.\n\n"
                "Problem:\n" + problem + "\n"
            )
            text = self.wrapper.generate_text(base, max_new_tokens=512, temperature=0.3)
            ans = extract_answer(text)
            return ans if ans is not None else FALLBACK_ANSWER
        return solve_with_prm(self.wrapper, self.prm_head, problem, n_samples=4)
