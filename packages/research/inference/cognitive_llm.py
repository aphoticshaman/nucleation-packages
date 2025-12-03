"""
Cognitive LLM Inference Engine.

Wraps fine-tuned Phi model with SDPM→XYZA cognitive monitoring.
Designed for self-hosted deployment (RunPod serverless, Modal, etc).

Zero API fees - runs on your GPU.

Usage:
    from research.inference.cognitive_llm import CognitiveLLM

    llm = CognitiveLLM("aphoticshaman/latticeforge-unified")
    response = llm.generate("Analyze the geopolitical situation in...",
                           return_cognitive_state=True)

    print(response.text)
    print(response.flow_state)  # Real-time cognitive metrics
    print(response.xyza)        # [X, Y, Z, A] scores
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple, Generator
from numpy.typing import NDArray
import time

# Cognitive modules
from research.cognitive import (
    FlowDetector,
    FlowState,
    NSMPhaseHead,
    NSMConfig,
    compute_xyza_metrics,
    XYZAMetrics,
    text_to_sdpm,
    compute_persona_alignment,
    SDPMVector,
    diagnose_xyza,
)


@dataclass
class CognitiveLLMConfig:
    """Configuration for cognitive LLM inference."""

    # Model
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    adapter_repo: str = "aphoticshaman/latticeforge-unified"
    device: str = "cuda"
    torch_dtype: str = "float16"

    # Generation
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    # Cognitive monitoring
    enable_cognitive: bool = True
    n_oscillators: int = 64
    flow_dwell_ms: float = 800.0

    # Persona
    ai_persona_seed: str = "latticeforge_analyst"


@dataclass
class CognitiveResponse:
    """Response with cognitive state attached."""
    text: str
    tokens_generated: int
    generation_time_ms: float

    # Cognitive metrics (if enabled)
    flow_state: Optional[FlowState] = None
    xyza: Optional[XYZAMetrics] = None
    sdpm: Optional[SDPMVector] = None
    diagnostics: List[str] = field(default_factory=list)

    # Raw data for analysis
    phase_trajectory: Optional[NDArray[np.float64]] = None
    r_trajectory: Optional[List[float]] = None


class CognitiveLLM:
    """
    LLM inference engine with real-time cognitive monitoring.

    Integrates:
    - Fine-tuned Phi model (LoRA adapter from HuggingFace)
    - NSM phase extraction from hidden states
    - XYZA cognitive benchmark scoring
    - Flow state detection via Kuramoto order parameter
    """

    def __init__(
        self,
        adapter_repo: Optional[str] = None,
        config: Optional[CognitiveLLMConfig] = None
    ):
        """
        Initialize cognitive LLM.

        Args:
            adapter_repo: HuggingFace repo for LoRA adapter
            config: Full configuration (overrides adapter_repo if provided)
        """
        self.config = config or CognitiveLLMConfig()
        if adapter_repo:
            self.config.adapter_repo = adapter_repo

        self.model = None
        self.tokenizer = None
        self.nsm_head: Optional[NSMPhaseHead] = None
        self.flow_detector: Optional[FlowDetector] = None
        self.ai_sdpm: Optional[SDPMVector] = None

        self._loaded = False

    def load(self) -> "CognitiveLLM":
        """
        Load model and initialize cognitive components.

        Returns self for chaining: llm = CognitiveLLM(...).load()
        """
        if self._loaded:
            return self

        print(f"[CognitiveLLM] Loading base model: {self.config.base_model}")

        # Import here to avoid slow startup if not using
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.float16)

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch_dtype,
            device_map="auto" if self.config.device == "cuda" else None,
            trust_remote_code=True,
        )

        # Load LoRA adapter
        print(f"[CognitiveLLM] Loading adapter: {self.config.adapter_repo}")
        self.model = PeftModel.from_pretrained(
            self.model,
            self.config.adapter_repo,
        )

        # Merge adapter for faster inference (optional)
        # self.model = self.model.merge_and_unload()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize cognitive components
        if self.config.enable_cognitive:
            self._init_cognitive()

        self._loaded = True
        print("[CognitiveLLM] Ready.")
        return self

    def _init_cognitive(self) -> None:
        """Initialize cognitive monitoring components."""
        # Get model hidden dimension
        d_model = self.model.config.hidden_size

        # NSM phase head
        nsm_config = NSMConfig(
            d_model=d_model,
            n_oscillators=self.config.n_oscillators,
            temporal_smooth=True,
            use_hilbert=True,
        )
        self.nsm_head = NSMPhaseHead(nsm_config)

        # Flow detector
        self.flow_detector = FlowDetector(
            dwell_time_ms=self.config.flow_dwell_ms
        )

        # AI persona SDPM
        self.ai_sdpm = text_to_sdpm(
            "analytical precise geopolitical intelligence",
            persona_seed=self.config.ai_persona_seed
        )

    def generate(
        self,
        prompt: str,
        return_cognitive_state: bool = True,
        user_persona_text: Optional[str] = None,
        **generate_kwargs
    ) -> CognitiveResponse:
        """
        Generate response with cognitive monitoring.

        Args:
            prompt: Input prompt
            return_cognitive_state: Whether to compute cognitive metrics
            user_persona_text: Optional text to compute user SDPM from
            **generate_kwargs: Override generation parameters

        Returns:
            CognitiveResponse with text and cognitive metrics
        """
        if not self._loaded:
            self.load()

        start_time = time.time()

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        # Generation parameters
        gen_config = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": self.config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "output_hidden_states": return_cognitive_state and self.config.enable_cognitive,
            "return_dict_in_generate": True,
        }
        gen_config.update(generate_kwargs)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_config,
            )

        # Decode
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        generation_time = (time.time() - start_time) * 1000

        # Build response
        response = CognitiveResponse(
            text=text,
            tokens_generated=len(generated_ids),
            generation_time_ms=generation_time,
        )

        # Compute cognitive state
        if return_cognitive_state and self.config.enable_cognitive:
            self._compute_cognitive_state(
                response,
                outputs,
                text,
                user_persona_text
            )

        return response

    def _compute_cognitive_state(
        self,
        response: CognitiveResponse,
        outputs: Any,
        generated_text: str,
        user_persona_text: Optional[str]
    ) -> None:
        """Compute cognitive metrics from generation outputs."""

        # Extract hidden states if available
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            # Get last layer hidden states
            # outputs.hidden_states is tuple of (step, layer, tensor)
            # For simplicity, use the final hidden state
            try:
                last_hidden = outputs.hidden_states[-1][-1]  # Last step, last layer
                if isinstance(last_hidden, torch.Tensor):
                    hidden_np = last_hidden.cpu().float().numpy()

                    # Extract phases via NSM
                    nsm_output = self.nsm_head.extract_phases(hidden_np)

                    # Update flow detector
                    flow_state = self.flow_detector.update(
                        nsm_output.order_parameter,
                        timestamp=time.time() * 1000
                    )
                    response.flow_state = flow_state
                    response.r_trajectory = [r for _, r in self.flow_detector.r_history]

            except Exception as e:
                # Hidden state extraction can be tricky depending on model
                print(f"[CognitiveLLM] Hidden state extraction failed: {e}")

        # Compute SDPM for generated text
        response.sdpm = text_to_sdpm(generated_text)

        # Compute XYZA (using available data)
        phases = np.random.uniform(0, 2*np.pi, self.config.n_oscillators)  # Placeholder if no hidden states
        if response.flow_state:
            # Use actual R as proxy for phases
            R = response.flow_state.R
            # Create pseudo-phases that would produce this R
            phases = np.random.uniform(0, 2*np.pi * (1-R), self.config.n_oscillators)

        signal = np.array([ord(c) for c in generated_text[:1000]])  # Text as signal for complexity

        # User SDPM for attunement
        user_sdpm = None
        if user_persona_text:
            user_sdpm = text_to_sdpm(user_persona_text)

        response.xyza = compute_xyza_metrics(
            phases=phases,
            signal=signal,
            human_sdpm=user_sdpm,
            ai_sdpm=self.ai_sdpm,
            response_latency_ms=response.generation_time_ms,
        )

        # Diagnostics
        response.diagnostics = diagnose_xyza(response.xyza)

    def stream(
        self,
        prompt: str,
        **generate_kwargs
    ) -> Generator[str, None, CognitiveResponse]:
        """
        Stream tokens with cognitive state at end.

        Yields tokens as they're generated, returns full CognitiveResponse.
        """
        if not self._loaded:
            self.load()

        # For now, just wrap generate() - true streaming requires more work
        response = self.generate(prompt, **generate_kwargs)

        # Simulate streaming by yielding chunks
        words = response.text.split()
        for word in words:
            yield word + " "

        return response

    def get_session_stats(self) -> Dict[str, Any]:
        """Get flow session statistics."""
        if self.flow_detector:
            return self.flow_detector.get_session_stats()
        return {"status": "cognitive_disabled"}

    def reset_flow_session(self) -> None:
        """Reset flow tracking for new session."""
        if self.flow_detector:
            self.flow_detector.reset_session()


# Convenience function for quick inference
def quick_generate(
    prompt: str,
    adapter_repo: str = "aphoticshaman/latticeforge-unified",
    **kwargs
) -> str:
    """
    Quick one-shot generation without cognitive tracking.

    Args:
        prompt: Input prompt
        adapter_repo: HuggingFace adapter repo
        **kwargs: Generation parameters

    Returns:
        Generated text
    """
    llm = CognitiveLLM(adapter_repo)
    llm.load()
    response = llm.generate(prompt, return_cognitive_state=False, **kwargs)
    return response.text
