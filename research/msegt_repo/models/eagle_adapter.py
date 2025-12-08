from typing import List
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class EagleWrapper(nn.Module):
    """
    Wrapper around NVIDIA GPT-OSS-120B Eagle3 (or compatible).
    Provides:
    - generate_text() for inference sampling
    - forward_for_prm() to extract embeddings for PRM training
    """
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.model.eval()

    @torch.no_grad()
    def generate_text(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def forward_for_prm(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states[-1]  # (B, T, D)
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled
