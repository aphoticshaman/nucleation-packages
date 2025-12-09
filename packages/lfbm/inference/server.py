"""
LFBM Inference Server for RunPod

Deploy as a serverless endpoint or always-on pod.
Provides OpenAI-compatible API for easy integration.

Usage:
    # Start server
    python server.py --model ./checkpoints/lfbm_final.pt --port 8000

    # Call from Vercel
    POST /v1/completions
    {
        "nations": [...],
        "signals": {...},
        "categories": {...}
    }
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.architecture import LFBM, LFBMConfig


# ============================================================
# Request/Response Models
# ============================================================

class NationInput(BaseModel):
    code: str
    risk: float = 0.5
    trend: float = 0.0


class BriefingRequest(BaseModel):
    nations: List[NationInput]
    signals: Dict[str, float] = {}
    categories: Dict[str, float] = {}
    max_tokens: int = 1024
    temperature: float = 0.7


class BriefingResponse(BaseModel):
    briefings: Dict[str, str]
    latency_ms: float
    tokens_generated: int
    model: str = "lfbm-v1"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    memory_used_gb: float


# ============================================================
# Inference Engine
# ============================================================

class LFBMInference:
    """Inference wrapper for LFBM model"""

    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.nation_to_idx = {}
        self.next_nation_idx = 0

        self._load_model(checkpoint_path)

    def _get_device(self, device: str) -> torch.device:
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)

    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint"""
        print(f"Loading model from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Get config from checkpoint or use default
        config = checkpoint.get('config', LFBMConfig())
        if isinstance(config, dict):
            config = LFBMConfig(**config)

        self.model = LFBM(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Simple tokenizer (should match training)
        from training.train import SimpleTokenizer
        self.tokenizer = SimpleTokenizer()

        print(f"Model loaded on {self.device}")
        print(f"Parameters: {self.model.count_parameters():,}")

    def _get_nation_idx(self, code: str) -> int:
        if code not in self.nation_to_idx:
            self.nation_to_idx[code] = self.next_nation_idx
            self.next_nation_idx += 1
        return self.nation_to_idx[code]

    @torch.no_grad()
    def generate(self, request: BriefingRequest) -> BriefingResponse:
        """Generate briefing from metrics"""
        start_time = time.time()

        # Prepare inputs
        nations = request.nations[:20]
        nation_codes = torch.zeros(1, 20, dtype=torch.long, device=self.device)
        nation_risks = torch.zeros(1, 20, device=self.device)
        nation_trends = torch.zeros(1, 20, device=self.device)

        for i, nation in enumerate(nations):
            nation_codes[0, i] = self._get_nation_idx(nation.code)
            nation_risks[0, i] = nation.risk
            nation_trends[0, i] = nation.trend + 0.5

        # Signals
        signals = request.signals
        signal_values = torch.tensor([[
            signals.get('gdelt_count', 50) / 200,
            (signals.get('avg_tone', 0) + 10) / 20,
            signals.get('alert_count', 5) / 50,
            0.5,
        ]], dtype=torch.float32, device=self.device)

        # Categories
        categories = request.categories
        category_risks = torch.tensor([[
            categories.get('political', 50) / 100,
            categories.get('economic', 50) / 100,
            categories.get('security', 50) / 100,
            categories.get('military', 50) / 100,
            categories.get('financial', 50) / 100,
            categories.get('cyber', 50) / 100,
            categories.get('health', 50) / 100,
            categories.get('scitech', 50) / 100,
            categories.get('energy', 50) / 100,
            categories.get('domestic', 50) / 100,
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5,
        ]], dtype=torch.float32, device=self.device)[:, :26]

        # Generate
        output_text = self.model.generate_briefing(
            nation_codes, nation_risks, nation_trends,
            signal_values, category_risks,
            self.tokenizer,
            max_length=request.max_tokens
        )

        # Parse JSON from output
        try:
            briefings = json.loads(output_text)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', output_text)
            if json_match:
                briefings = json.loads(json_match.group())
            else:
                briefings = {'error': 'Failed to parse output', 'raw': output_text}

        latency_ms = (time.time() - start_time) * 1000
        tokens = len(self.tokenizer.encode(json.dumps(briefings)))

        return BriefingResponse(
            briefings=briefings,
            latency_ms=latency_ms,
            tokens_generated=tokens
        )


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="LFBM Inference Server",
    description="LatticeForge Briefing Model - Metrics to Intelligence Prose",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine (loaded on startup)
engine: Optional[LFBMInference] = None


@app.on_event("startup")
async def startup():
    global engine
    model_path = os.environ.get('LFBM_MODEL_PATH', './checkpoints/lfbm_final.pt')
    device = os.environ.get('LFBM_DEVICE', 'auto')

    if os.path.exists(model_path):
        engine = LFBMInference(model_path, device)
    else:
        print(f"WARNING: Model not found at {model_path}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    memory_gb = 0
    if torch.cuda.is_available():
        memory_gb = torch.cuda.memory_allocated() / 1e9

    return HealthResponse(
        status="healthy" if engine else "model_not_loaded",
        model_loaded=engine is not None,
        device=str(engine.device) if engine else "none",
        memory_used_gb=memory_gb
    )


@app.post("/v1/completions", response_model=BriefingResponse)
async def generate_briefing(request: BriefingRequest):
    """Generate intelligence briefing from metrics"""
    if not engine:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return engine.generate(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/briefing", response_model=BriefingResponse)
async def generate_briefing_alt(request: BriefingRequest):
    """Alias for /v1/completions"""
    return await generate_briefing(request)


# ============================================================
# RunPod Serverless Handler
# ============================================================

def runpod_handler(event):
    """
    RunPod serverless handler.

    Deploy with:
        runpodctl deploy --name lfbm-inference --handler inference/server.py:runpod_handler
    """
    global engine

    if engine is None:
        model_path = os.environ.get('LFBM_MODEL_PATH', '/model/lfbm_final.pt')
        engine = LFBMInference(model_path)

    try:
        input_data = event.get('input', {})
        request = BriefingRequest(**input_data)
        response = engine.generate(request)
        return {
            'output': response.dict()
        }
    except Exception as e:
        return {
            'error': str(e)
        }


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./checkpoints/lfbm_final.pt')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()

    os.environ['LFBM_MODEL_PATH'] = args.model

    uvicorn.run(app, host=args.host, port=args.port)
