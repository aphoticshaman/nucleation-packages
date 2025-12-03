"""
LatticeForge on Modal - Secure Serverless GPU Inference.

Modal advantages:
- Firecracker microVM isolation (same as AWS Lambda)
- Built-in secrets management
- SOC 2 compliant
- Sub-second cold starts
- Native Python DX

Deploy:
    modal deploy modal_app.py

Test locally:
    modal run modal_app.py

Usage:
    curl -X POST https://aphoticshaman--latticeforge-generate.modal.run \\
        -H "Content-Type: application/json" \\
        -H "Authorization: Bearer $LATTICEFORGE_API_KEY" \\
        -d '{"prompt": "Analyze..."}'
"""

import modal
from modal import Image, Secret, web_endpoint, asgi_app
import os

# Modal app definition
app = modal.App("latticeforge")

# GPU image with all dependencies
inference_image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "accelerate>=0.25.0",
        "fastapi>=0.104.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "huggingface_hub>=0.19.0",
    )
    .copy_local_dir("../cognitive", "/app/research/cognitive")
    .copy_local_file("cognitive_llm.py", "/app/research/inference/cognitive_llm.py")
    .copy_local_file("security.py", "/app/research/inference/security.py")
    .env({"PYTHONPATH": "/app"})
)

# Secrets (stored securely in Modal dashboard)
# Set via: modal secret create latticeforge-secrets SIGNING_KEY=xxx HF_TOKEN=xxx
secrets = Secret.from_name("latticeforge-secrets")


@app.cls(
    gpu="A10G",  # or "A100" for production
    image=inference_image,
    secrets=[secrets],
    timeout=300,
    container_idle_timeout=60,  # Keep warm for 60s
    allow_concurrent_inputs=10,
)
class LatticeForgeInference:
    """
    Modal class for LatticeForge inference.

    Automatically scales, GPU-accelerated, secure.
    """

    def __init__(self):
        self.llm = None
        self.security = None

    @modal.enter()
    def load_model(self):
        """Load model on container start (cached across requests)."""
        from research.inference.cognitive_llm import CognitiveLLM, CognitiveLLMConfig
        from research.inference.security import SecurityMiddleware, SecurityConfig

        print("[Modal] Loading model...")

        config = CognitiveLLMConfig(
            adapter_repo=os.getenv("ADAPTER_REPO", "aphoticshaman/latticeforge-unified"),
            base_model=os.getenv("BASE_MODEL", "microsoft/Phi-3-mini-4k-instruct"),
            device="cuda",
            torch_dtype="float16",
        )

        self.llm = CognitiveLLM(config=config).load()

        # Initialize security
        sec_config = SecurityConfig(
            signing_key=os.getenv("SIGNING_KEY", ""),
            canary_token=os.getenv("CANARY_TOKEN", ""),
        )
        self.security = SecurityMiddleware(sec_config)

        print("[Modal] Ready.")

    @modal.method()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        return_cognitive: bool = True,
        client_id: str = "anonymous",
        signature: str = None,
    ) -> dict:
        """Generate with cognitive metrics."""

        # Security validation
        payload = {"prompt": prompt, "max_tokens": max_tokens}
        allowed, error = self.security.validate_request(
            client_id=client_id,
            payload=payload,
            signature=signature,
            require_signature=bool(os.getenv("REQUIRE_SIGNATURES", "0") == "1"),
        )

        if not allowed:
            return {"error": error, "status": 403}

        # Generate
        response = self.llm.generate(
            prompt=prompt,
            return_cognitive_state=return_cognitive,
            max_new_tokens=max_tokens,
            temperature=temperature,
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
                "stability": response.flow_state.stability,
            }

        if response.xyza:
            result["xyza"] = {
                "coherence_x": response.xyza.coherence_x,
                "complexity_y": response.xyza.complexity_y,
                "reflection_z": response.xyza.reflection_z,
                "attunement_a": response.xyza.attunement_a,
                "combined_score": response.xyza.combined_score,
            }

        # Audit log
        self.security.log_request(client_id, payload, result)

        return result

    @modal.method()
    def health(self) -> dict:
        """Health check."""
        import torch
        return {
            "status": "ok",
            "model_loaded": self.llm is not None,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "none",
        }


# Web endpoint (HTTPS, auth via header)
@app.function(
    image=inference_image,
    secrets=[secrets],
)
@web_endpoint(method="POST", docs=True)
def generate(request: dict) -> dict:
    """
    Public HTTPS endpoint for inference.

    Headers:
        Authorization: Bearer <api_key>
        X-Signature: <hmac_signature>

    Body:
        {
            "prompt": "...",
            "max_tokens": 1024,
            "temperature": 0.7
        }
    """
    # Get inference class
    inference = LatticeForgeInference()

    return inference.generate.remote(
        prompt=request.get("prompt", ""),
        max_tokens=request.get("max_tokens", 1024),
        temperature=request.get("temperature", 0.7),
        return_cognitive=request.get("return_cognitive", True),
        client_id=request.get("client_id", "anonymous"),
        signature=request.get("signature"),
    )


@app.function(image=inference_image)
@web_endpoint(method="GET")
def health() -> dict:
    """Health check endpoint."""
    inference = LatticeForgeInference()
    return inference.health.remote()


# Local testing
@app.local_entrypoint()
def main():
    """Test locally."""
    inference = LatticeForgeInference()

    result = inference.generate.remote(
        prompt="What is the current geopolitical situation in Eastern Europe?",
        max_tokens=256,
    )

    print("Response:", result.get("text", "")[:200])
    print("XYZA:", result.get("xyza"))
