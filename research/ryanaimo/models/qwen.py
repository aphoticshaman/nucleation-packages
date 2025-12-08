"""
Qwen Math Model Wrapper
=======================

Thin wrapper around Qwen2.5-Math-72B-Instruct.
This is the ONLY place we use external libraries for inference.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Any

import torch


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    path: str
    quantization: str = "nf4"  # "nf4", "int8", "none"
    compute_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True


def find_model_path(base_path: str) -> str:
    """
    Find config.json in model directory.

    Kaggle can nest model files in subdirectories.
    """
    if os.path.exists(f"{base_path}/config.json"):
        return base_path

    if os.path.isdir(base_path):
        for item in os.listdir(base_path):
            subpath = f"{base_path}/{item}"
            if os.path.isdir(subpath) and os.path.exists(f"{subpath}/config.json"):
                return subpath

            # One more level
            if os.path.isdir(subpath):
                for subitem in os.listdir(subpath):
                    subsubpath = f"{subpath}/{subitem}"
                    if os.path.isdir(subsubpath) and os.path.exists(f"{subsubpath}/config.json"):
                        return subsubpath

    return base_path


def load_model(
    path: str,
    quantization: str = "nf4",
    compute_dtype: str = "bfloat16",
) -> tuple:
    """
    Load Qwen2.5-Math model with quantization.

    Args:
        path: Base path to model
        quantization: "nf4", "int8", or "none"
        compute_dtype: "bfloat16" or "float16"

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_path = find_model_path(path)
    print(f"[RYANAIMO] Loading model from {model_path}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Compute dtype
    dtype = torch.bfloat16 if compute_dtype == "bfloat16" else torch.float16

    # Quantization config
    if quantization == "nf4":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "int8":
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quant_config = None

    # Load model
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.eval()

    load_time = time.time() - start
    print(f"[RYANAIMO] Model loaded in {load_time:.1f}s")
    print(f"[RYANAIMO] Memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

    return model, tokenizer


class QwenMathModel:
    """
    Wrapper for Qwen2.5-Math-72B-Instruct.

    Provides:
    - Lazy loading
    - Extended thinking generation
    - Prompt templates
    """

    SYSTEM_PROMPT = """You are an expert mathematician solving olympiad problems.

For each problem:
1. First, think deeply about the problem in <think>...</think> tags
2. Identify the mathematical structures and techniques
3. Write clean Python code to compute the answer
4. Store the final answer in a variable called 'answer'
5. Answer must be an integer from 0 to 99999

Show your reasoning, then provide working code."""

    def __init__(
        self,
        path: str,
        quantization: str = "nf4",
        compute_dtype: str = "bfloat16",
        lazy_load: bool = True,
    ):
        self.config = ModelConfig(
            path=path,
            quantization=quantization,
            compute_dtype=compute_dtype,
        )
        self.model = None
        self.tokenizer = None
        self._loaded = False

        if not lazy_load:
            self.load()

    def load(self):
        """Load model if not already loaded."""
        if self._loaded:
            return

        self.model, self.tokenizer = load_model(
            path=self.config.path,
            quantization=self.config.quantization,
            compute_dtype=self.config.compute_dtype,
        )
        self._loaded = True

    def generate(
        self,
        problem: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate solution for a problem.

        Args:
            problem: The math problem text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            system_prompt: Override default system prompt

        Returns:
            Generated text
        """
        self.load()

        sys_prompt = system_prompt or self.SYSTEM_PROMPT

        # Build chat format
        prompt = (
            f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )

        return response

    def generate_with_think(
        self,
        problem: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate with explicit thinking phase.

        Uses a prompt that encourages deep reasoning before coding.
        """
        think_prompt = """You are an expert mathematician. Before writing code, you MUST think deeply.

Structure your response as:

<think>
[Your step-by-step reasoning here. Be thorough. Consider:
- What type of problem is this?
- What mathematical techniques apply?
- What are the key constraints?
- How can I verify my approach?]
</think>

Then write Python code with the answer stored in 'answer' variable."""

        return self.generate(
            problem=problem,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=think_prompt,
        )

    @property
    def device(self):
        """Get model device."""
        self.load()
        return self.model.device


__all__ = ["QwenMathModel", "load_model", "find_model_path"]
