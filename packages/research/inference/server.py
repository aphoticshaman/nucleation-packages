"""
FastAPI server for Cognitive LLM inference.

Deploy to RunPod Serverless, Modal, or any GPU instance.

Usage:
    # Local
    uvicorn research.inference.server:app --host 0.0.0.0 --port 8000

    # With GPU
    CUDA_VISIBLE_DEVICES=0 uvicorn research.inference.server:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /generate     - Generate with cognitive metrics
    POST /analyze      - Analyze text for SDPM/XYZA
    GET  /health       - Health check
    GET  /flow-status  - Current flow session stats
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import time
import os

# Lazy load to speed up cold starts
_llm = None


def get_llm():
    """Lazy load LLM on first request."""
    global _llm
    if _llm is None:
        from research.inference.cognitive_llm import CognitiveLLM, CognitiveLLMConfig

        config = CognitiveLLMConfig(
            adapter_repo=os.getenv("ADAPTER_REPO", "aphoticshaman/latticeforge-unified"),
            base_model=os.getenv("BASE_MODEL", "microsoft/Phi-3-mini-4k-instruct"),
            device="cuda" if os.getenv("USE_GPU", "1") == "1" else "cpu",
        )
        _llm = CognitiveLLM(config=config).load()
    return _llm


# FastAPI app
app = FastAPI(
    title="LatticeForge Cognitive LLM",
    description="Self-hosted LLM inference with SDPM→XYZA cognitive monitoring",
    version="1.0.0",
)

# CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://latticeforge.vercel.app",
        "https://*.vercel.app",
        os.getenv("FRONTEND_URL", "*"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(1024, description="Max tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    return_cognitive: bool = Field(True, description="Include cognitive metrics")
    user_context: Optional[str] = Field(None, description="User text for persona alignment")
    system_prompt: Optional[str] = Field(None, description="System prompt to prepend")


class FlowStateResponse(BaseModel):
    level: str
    R: float
    is_flow: bool
    is_deep_flow: bool
    stability: float
    time_in_state_ms: float


class XYZAResponse(BaseModel):
    coherence_x: float
    complexity_y: float
    reflection_z: float
    attunement_a: float
    combined_score: float
    cognitive_level: str


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    generation_time_ms: float
    flow_state: Optional[FlowStateResponse] = None
    xyza: Optional[XYZAResponse] = None
    diagnostics: List[str] = []


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    compute_sdpm: bool = Field(True)
    compute_complexity: bool = Field(True)


class AnalyzeResponse(BaseModel):
    sdpm_cognitive_mode: Optional[str] = None
    sdpm_emotional_valence: Optional[float] = None
    complexity_score: Optional[float] = None
    analysis: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    adapter: str
    gpu_available: bool


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    import torch
    return HealthResponse(
        status="ok",
        model_loaded=_llm is not None,
        adapter=os.getenv("ADAPTER_REPO", "aphoticshaman/latticeforge-unified"),
        gpu_available=torch.cuda.is_available(),
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text with cognitive monitoring.

    Returns generated text plus flow state and XYZA metrics.
    """
    try:
        llm = get_llm()

        # Build prompt
        prompt = request.prompt
        if request.system_prompt:
            prompt = f"{request.system_prompt}\n\n{prompt}"

        # Generate
        response = llm.generate(
            prompt,
            return_cognitive_state=request.return_cognitive,
            user_persona_text=request.user_context,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        # Build response
        result = GenerateResponse(
            text=response.text,
            tokens_generated=response.tokens_generated,
            generation_time_ms=response.generation_time_ms,
            diagnostics=response.diagnostics,
        )

        # Add cognitive metrics if available
        if response.flow_state:
            result.flow_state = FlowStateResponse(
                level=response.flow_state.level.name,
                R=response.flow_state.R,
                is_flow=response.flow_state.is_flow,
                is_deep_flow=response.flow_state.is_deep_flow,
                stability=response.flow_state.stability,
                time_in_state_ms=response.flow_state.time_in_state_ms,
            )

        if response.xyza:
            result.xyza = XYZAResponse(
                coherence_x=response.xyza.coherence_x,
                complexity_y=response.xyza.complexity_y,
                reflection_z=response.xyza.reflection_z,
                attunement_a=response.xyza.attunement_a,
                combined_score=response.xyza.combined_score,
                cognitive_level=response.xyza.cognitive_level,
            )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze text for cognitive properties without generation.

    Useful for analyzing user input or external content.
    """
    try:
        from research.cognitive import (
            text_to_sdpm,
            sdpm_to_cognitive_state,
            compute_complexity_y,
        )
        import numpy as np

        result = AnalyzeResponse()

        if request.compute_sdpm:
            sdpm = text_to_sdpm(request.text)
            result.sdpm_cognitive_mode = sdpm.cognitive_mode
            result.sdpm_emotional_valence = sdpm.emotional_valence
            result.analysis["cognitive_state"] = sdpm_to_cognitive_state(sdpm)

        if request.compute_complexity:
            signal = np.array([ord(c) for c in request.text])
            complexity, details = compute_complexity_y(signal)
            result.complexity_score = complexity
            result.analysis["complexity_details"] = details

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/flow-status")
async def get_flow_status():
    """Get current flow session statistics."""
    if _llm is None:
        return {"status": "model_not_loaded"}

    return _llm.get_session_stats()


@app.post("/flow-reset")
async def reset_flow():
    """Reset flow session tracking."""
    if _llm is None:
        return {"status": "model_not_loaded"}

    _llm.reset_flow_session()
    return {"status": "reset"}


# RunPod serverless handler
def handler(event):
    """
    RunPod serverless handler.

    Wraps FastAPI for serverless deployment.
    """
    import json

    method = event.get("method", "POST")
    path = event.get("path", "/generate")
    body = event.get("body", {})

    if isinstance(body, str):
        body = json.loads(body)

    # Route to appropriate endpoint
    if path == "/health":
        import torch
        return {
            "status": "ok",
            "model_loaded": _llm is not None,
            "gpu_available": torch.cuda.is_available(),
        }

    elif path == "/generate":
        llm = get_llm()
        response = llm.generate(
            body.get("prompt", ""),
            return_cognitive_state=body.get("return_cognitive", True),
            user_persona_text=body.get("user_context"),
            max_new_tokens=body.get("max_tokens", 1024),
            temperature=body.get("temperature", 0.7),
        )

        result = {
            "text": response.text,
            "tokens_generated": response.tokens_generated,
            "generation_time_ms": response.generation_time_ms,
            "diagnostics": response.diagnostics,
        }

        if response.flow_state:
            result["flow_state"] = {
                "level": response.flow_state.level.name,
                "R": response.flow_state.R,
                "is_flow": response.flow_state.is_flow,
            }

        if response.xyza:
            result["xyza"] = {
                "coherence_x": response.xyza.coherence_x,
                "complexity_y": response.xyza.complexity_y,
                "reflection_z": response.xyza.reflection_z,
                "attunement_a": response.xyza.attunement_a,
                "combined_score": response.xyza.combined_score,
            }

        return result

    return {"error": f"Unknown path: {path}"}


# For direct execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
