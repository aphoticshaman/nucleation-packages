"""
RunPod Serverless Handler.

This is the entry point for RunPod serverless deployment.
Handles cold starts efficiently by lazy-loading the model.

Deploy with:
    runpod deploy --gpu RTX4090 --handler runpod_handler.py

Or use the RunPod dashboard to deploy from Docker image.
"""

import runpod
import json
import time
import os

# Global model cache
_llm = None
_load_time = None


def get_llm():
    """Lazy load model on first request."""
    global _llm, _load_time

    if _llm is not None:
        return _llm

    print("[RunPod] Cold start - loading model...")
    start = time.time()

    from research.inference.cognitive_llm import CognitiveLLM, CognitiveLLMConfig

    config = CognitiveLLMConfig(
        adapter_repo=os.getenv("ADAPTER_REPO", "aphoticshaman/latticeforge-unified"),
        base_model=os.getenv("BASE_MODEL", "microsoft/Phi-3-mini-4k-instruct"),
        device="cuda",
        torch_dtype="float16",
    )

    _llm = CognitiveLLM(config=config).load()
    _load_time = time.time() - start
    print(f"[RunPod] Model loaded in {_load_time:.2f}s")

    return _llm


def handler(event):
    """
    RunPod serverless handler.

    Input event format:
    {
        "input": {
            "prompt": "Your prompt here",
            "max_tokens": 1024,
            "temperature": 0.7,
            "return_cognitive": true,
            "user_context": "optional user text"
        }
    }

    Returns:
    {
        "text": "Generated response",
        "tokens_generated": 150,
        "generation_time_ms": 2500,
        "flow_state": {...},
        "xyza": {...}
    }
    """
    try:
        # Extract input
        input_data = event.get("input", {})

        prompt = input_data.get("prompt")
        if not prompt:
            return {"error": "Missing required field: prompt"}

        # Get model
        llm = get_llm()

        # Generate
        response = llm.generate(
            prompt=prompt,
            return_cognitive_state=input_data.get("return_cognitive", True),
            user_persona_text=input_data.get("user_context"),
            max_new_tokens=input_data.get("max_tokens", 1024),
            temperature=input_data.get("temperature", 0.7),
        )

        # Build result
        result = {
            "text": response.text,
            "tokens_generated": response.tokens_generated,
            "generation_time_ms": response.generation_time_ms,
            "diagnostics": response.diagnostics,
        }

        # Add cognitive metrics
        if response.flow_state:
            result["flow_state"] = {
                "level": response.flow_state.level.name,
                "R": response.flow_state.R,
                "dR_dt": response.flow_state.dR_dt,
                "is_flow": response.flow_state.is_flow,
                "is_deep_flow": response.flow_state.is_deep_flow,
                "stability": response.flow_state.stability,
                "time_in_state_ms": response.flow_state.time_in_state_ms,
            }

        if response.xyza:
            result["xyza"] = {
                "coherence_x": response.xyza.coherence_x,
                "complexity_y": response.xyza.complexity_y,
                "reflection_z": response.xyza.reflection_z,
                "attunement_a": response.xyza.attunement_a,
                "combined_score": response.xyza.combined_score,
                "cognitive_level": response.xyza.cognitive_level,
            }

        if response.sdpm:
            result["sdpm"] = {
                "cognitive_mode": response.sdpm.cognitive_mode,
                "emotional_valence": response.sdpm.emotional_valence,
            }

        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# RunPod entry point
runpod.serverless.start({"handler": handler})
