"""
LatticeForge Inference Client.

Lightweight client for calling the self-hosted LLM server.
Use this from Vercel/Next.js to call your GPU backend.

Usage:
    from research.inference.client import LatticeForgeClient

    client = LatticeForgeClient("https://your-runpod-endpoint.runpod.net")
    result = await client.generate("Analyze the situation in...")

    print(result.text)
    print(result.xyza.combined_score)
"""

import httpx
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import os


@dataclass
class FlowState:
    level: str
    R: float
    is_flow: bool
    is_deep_flow: bool
    stability: float
    time_in_state_ms: float


@dataclass
class XYZAMetrics:
    coherence_x: float
    complexity_y: float
    reflection_z: float
    attunement_a: float
    combined_score: float
    cognitive_level: str


@dataclass
class InferenceResult:
    text: str
    tokens_generated: int
    generation_time_ms: float
    flow_state: Optional[FlowState] = None
    xyza: Optional[XYZAMetrics] = None
    diagnostics: List[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceResult":
        flow = None
        if data.get("flow_state"):
            fs = data["flow_state"]
            flow = FlowState(
                level=fs.get("level", "NONE"),
                R=fs.get("R", 0.0),
                is_flow=fs.get("is_flow", False),
                is_deep_flow=fs.get("is_deep_flow", False),
                stability=fs.get("stability", 0.0),
                time_in_state_ms=fs.get("time_in_state_ms", 0.0),
            )

        xyza = None
        if data.get("xyza"):
            x = data["xyza"]
            xyza = XYZAMetrics(
                coherence_x=x.get("coherence_x", 0.0),
                complexity_y=x.get("complexity_y", 0.0),
                reflection_z=x.get("reflection_z", 0.0),
                attunement_a=x.get("attunement_a", 0.0),
                combined_score=x.get("combined_score", 0.0),
                cognitive_level=x.get("cognitive_level", "unknown"),
            )

        return cls(
            text=data.get("text", ""),
            tokens_generated=data.get("tokens_generated", 0),
            generation_time_ms=data.get("generation_time_ms", 0.0),
            flow_state=flow,
            xyza=xyza,
            diagnostics=data.get("diagnostics", []),
        )


class LatticeForgeClient:
    """
    Async client for LatticeForge inference server.

    Works with both:
    - Direct FastAPI server (local or cloud VM)
    - RunPod serverless endpoint
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """
        Initialize client.

        Args:
            endpoint: Server URL (or set LATTICEFORGE_ENDPOINT env var)
            api_key: API key for RunPod (or set RUNPOD_API_KEY env var)
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint or os.getenv("LATTICEFORGE_ENDPOINT", "http://localhost:8000")
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        self.timeout = timeout

        # Detect if this is a RunPod endpoint
        self.is_runpod = "runpod" in self.endpoint.lower()

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        return_cognitive: bool = True,
        user_context: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> InferenceResult:
        """
        Generate text with cognitive metrics.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_cognitive: Include cognitive metrics
            user_context: User text for persona alignment
            system_prompt: System prompt to prepend

        Returns:
            InferenceResult with text and cognitive metrics
        """
        if self.is_runpod:
            return await self._generate_runpod(
                prompt, max_tokens, temperature, return_cognitive, user_context
            )
        else:
            return await self._generate_direct(
                prompt, max_tokens, temperature, return_cognitive, user_context, system_prompt
            )

    async def _generate_direct(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        return_cognitive: bool,
        user_context: Optional[str],
        system_prompt: Optional[str],
    ) -> InferenceResult:
        """Call direct FastAPI endpoint."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.endpoint}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "return_cognitive": return_cognitive,
                    "user_context": user_context,
                    "system_prompt": system_prompt,
                },
            )
            response.raise_for_status()
            return InferenceResult.from_dict(response.json())

    async def _generate_runpod(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        return_cognitive: bool,
        user_context: Optional[str],
    ) -> InferenceResult:
        """Call RunPod serverless endpoint."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # RunPod uses /run endpoint
            response = await client.post(
                f"{self.endpoint}/run",
                headers=headers,
                json={
                    "input": {
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "return_cognitive": return_cognitive,
                        "user_context": user_context,
                    }
                },
            )
            response.raise_for_status()
            data = response.json()

            # RunPod wraps response in "output"
            if "output" in data:
                return InferenceResult.from_dict(data["output"])
            return InferenceResult.from_dict(data)

    async def analyze(
        self,
        text: str,
        compute_sdpm: bool = True,
        compute_complexity: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze text for cognitive properties.

        Args:
            text: Text to analyze
            compute_sdpm: Compute SDPM persona vector
            compute_complexity: Compute entropy-based complexity

        Returns:
            Analysis results
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.endpoint}/analyze",
                json={
                    "text": text,
                    "compute_sdpm": compute_sdpm,
                    "compute_complexity": compute_complexity,
                },
            )
            response.raise_for_status()
            return response.json()

    async def health(self) -> Dict[str, Any]:
        """Check server health."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.endpoint}/health")
            response.raise_for_status()
            return response.json()

    async def flow_status(self) -> Dict[str, Any]:
        """Get current flow session stats."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.endpoint}/flow-status")
            response.raise_for_status()
            return response.json()


# Sync wrapper for non-async contexts
class LatticeForgeClientSync:
    """Synchronous wrapper for LatticeForgeClient."""

    def __init__(self, *args, **kwargs):
        self._async_client = LatticeForgeClient(*args, **kwargs)

    def generate(self, *args, **kwargs) -> InferenceResult:
        import asyncio
        return asyncio.run(self._async_client.generate(*args, **kwargs))

    def analyze(self, *args, **kwargs) -> Dict[str, Any]:
        import asyncio
        return asyncio.run(self._async_client.analyze(*args, **kwargs))

    def health(self) -> Dict[str, Any]:
        import asyncio
        return asyncio.run(self._async_client.health())
