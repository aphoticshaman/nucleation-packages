"""
MODAL DEPLOYMENT: Fine-tuned Phi-2 for Geopolitical Intelligence

Model: Microsoft Phi-2 (2.7B params) fine-tuned on 60k intel data points
Use case: Replace Anthropic for routine 10-minute intelligence summaries
Cost savings: ~$0.001/inference vs ~$0.01+ for Anthropic API

Deployment:
1. Upload model to Modal volume or HuggingFace
2. Run: modal deploy modal_phi2_intel.py
3. Call endpoint from Next.js API routes
"""

import modal
from typing import Optional

# Modal app setup
app = modal.App("latticeforge-phi2-intel")

# Model volume for persistent storage
model_volume = modal.Volume.from_name("phi2-intel-model", create_if_missing=True)

# Docker image with ML dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",  # For quantization
        "safetensors>=0.4.0",
        "sentencepiece>=0.1.99",
        "protobuf>=4.25.0",
        "fastapi",  # Required for web endpoints
        "peft>=0.7.0",  # For LoRA adapter loading
    )
    .run_commands(
        "pip install flash-attn --no-build-isolation || true"  # Optional, faster attention
    )
)


@app.cls(
    image=image,
    gpu="T4",  # T4 is cheapest, Phi-2 fits easily. Use A10G for faster inference.
    volumes={"/model": model_volume},
    secrets=[modal.Secret.from_name("huggingface")],  # For private HF model
    timeout=300,
    scaledown_window=120,  # Keep warm for 2 min after last request
)
@modal.concurrent(max_inputs=10)  # Handle multiple requests
class Phi2IntelModel:
    """
    Fine-tuned Phi-2 for geopolitical intelligence analysis.

    Replaces Anthropic API for routine analysis tasks:
    - 10-minute news summaries
    - Training data generation
    - Risk scoring narratives

    Keep Anthropic for:
    - Complex multi-step reasoning
    - Enterprise customer requests
    - Novel/unprecedented situations
    """

    @modal.enter()
    def load_model(self):
        """Load model on container startup (cold start ~60s first time)."""
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        # Check if merged model exists in volume (fastest)
        merged_path = "/model/latticeforge-merged"

        try:
            if os.path.exists(merged_path) and os.listdir(merged_path):
                # Load pre-merged model from volume
                print(f"Loading merged model from volume: {merged_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    merged_path,
                    trust_remote_code=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    merged_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                print("Loaded merged model from volume!")
            else:
                raise FileNotFoundError("No merged model in volume")
        except Exception as e:
            print(f"Volume load failed ({e}), loading from HuggingFace...")

            # Load base Phi-2 model
            base_model_id = "microsoft/phi-2"
            adapter_id = "aphoticshaman/latticeforge-unified"

            print(f"Loading base model: {base_model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                trust_remote_code=True,
            )

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

            # Load and apply LoRA adapter
            print(f"Loading LoRA adapter: {adapter_id}")
            self.model = PeftModel.from_pretrained(
                base_model,
                adapter_id,
                token=os.environ.get("HF_TOKEN"),
            )

            # Merge adapter into base model for faster inference
            print("Merging adapter into base model...")
            self.model = self.model.merge_and_unload()

            # Save merged model to volume for faster future loads
            print(f"Saving merged model to volume: {merged_path}")
            os.makedirs(merged_path, exist_ok=True)
            self.tokenizer.save_pretrained(merged_path)
            self.model.save_pretrained(merged_path)
            print("Merged model saved to volume!")

        self.model.eval()
        print("Model loaded and ready for inference")

    @modal.method()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """
        Generate intelligence analysis from prompt.

        Args:
            prompt: The analysis request
            max_new_tokens: Max output length
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter
            system_prompt: Optional system context

        Returns:
            dict with 'text', 'tokens_generated', 'latency_ms'
        """
        import torch
        import time

        start_time = time.time()

        # Format prompt (adjust based on your fine-tuning format)
        if system_prompt:
            full_prompt = f"""### System:
{system_prompt}

### User:
{prompt}

### Assistant:
"""
        else:
            full_prompt = f"""### User:
{prompt}

### Assistant:
"""

        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode (only new tokens)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        latency_ms = (time.time() - start_time) * 1000

        return {
            "text": response.strip(),
            "tokens_generated": len(generated_tokens),
            "latency_ms": round(latency_ms, 2),
            "model": "phi2-intel-60k",
        }

    @modal.method()
    def analyze_news(self, news_items: list[dict], domain: str) -> dict:
        """
        Analyze news items and generate training-quality output.

        This replaces the Anthropic call in /api/generate/training-data

        Args:
            news_items: List of {title, description, link, pubDate, domain}
            domain: Analysis domain (geopolitical, financial, cyber, etc.)

        Returns:
            dict with analysis and training example
        """
        # Build prompt matching your fine-tuning format
        news_text = "\n".join([
            f"- [{item.get('domain', domain)}] {item.get('title', '')}"
            for item in news_items[:5]
        ])

        prompt = f"""Analyze these {domain} news items and provide:
1. Risk assessment (CRITICAL/HIGH/ELEVATED/LOW)
2. Key indicators and signals
3. Cascade potential to related domains
4. Historical parallels
5. Recommended monitoring actions

NEWS ITEMS:
{news_text}

Provide expert analysis:"""

        system = f"You are a {domain} intelligence analyst with expertise in risk assessment, pattern recognition, and cross-domain cascade effects."

        result = self.generate(
            prompt=prompt,
            max_new_tokens=600,
            temperature=0.5,  # Lower temp for more consistent analysis
            system_prompt=system,
        )

        return {
            "analysis": result["text"],
            "domain": domain,
            "news_count": len(news_items),
            "model_latency_ms": result["latency_ms"],
        }

    @modal.method()
    def generate_briefing(
        self,
        preset: str,
        metrics: dict,
        alerts: list[dict],
    ) -> dict:
        """
        Generate intel briefing from pre-computed metrics.

        This replaces the Anthropic call in /api/intel-briefing

        Args:
            preset: global, nato, brics, conflict
            metrics: Pre-computed category risk metrics
            alerts: Top alerts

        Returns:
            dict with briefings per category
        """
        # Build metrics summary
        metrics_text = "\n".join([
            f"- {cat}: Risk {m.get('riskLevel', 50)}/100, Trend: {m.get('trend', 'stable')}"
            for cat, m in metrics.items()
        ])

        alerts_text = "\n".join([
            f"- [{a.get('severity', 'watch').upper()}] {a.get('category')}: {a.get('summary')}"
            for a in alerts[:5]
        ])

        prompt = f"""Generate concise intel briefings for the {preset.upper()} preset.

PRE-COMPUTED METRICS:
{metrics_text}

TOP ALERTS:
{alerts_text}

For each category, provide 1-2 sentences summarizing the current situation and key concerns.
Focus on actionable intelligence for decision-makers.

Output JSON format:
{{"political": "...", "economic": "...", "security": "...", "summary": "...", "nsm": "Next strategic move..."}}"""

        system = "You are an intelligence analyst providing concise briefings. Be specific, actionable, and avoid speculation."

        result = self.generate(
            prompt=prompt,
            max_new_tokens=800,
            temperature=0.6,
            system_prompt=system,
        )

        # Try to parse JSON from response
        import json
        try:
            # Find JSON in response
            text = result["text"]
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                briefings = json.loads(text[start:end])
            else:
                briefings = {"raw": text}
        except json.JSONDecodeError:
            briefings = {"raw": result["text"]}

        return {
            "briefings": briefings,
            "preset": preset,
            "model_latency_ms": result["latency_ms"],
        }


# ============================================
# WEB ENDPOINT (for Next.js to call)
# ============================================

@app.function(
    image=image,
    timeout=300,  # 5 min for cold start + inference
)
@modal.fastapi_endpoint(method="POST", docs=True)
def inference(request: dict) -> dict:
    """
    HTTP endpoint for inference.

    POST /inference
    {
        "action": "generate" | "analyze_news" | "generate_briefing",
        "prompt": "...",  // for generate
        "news_items": [...],  // for analyze_news
        "preset": "...",  // for generate_briefing
        "metrics": {...},  // for generate_briefing
        ...
    }
    """
    model = Phi2IntelModel()

    action = request.get("action", "generate")

    if action == "generate":
        return model.generate.remote(
            prompt=request.get("prompt", ""),
            max_new_tokens=request.get("max_new_tokens", 512),
            temperature=request.get("temperature", 0.7),
            system_prompt=request.get("system_prompt"),
        )

    elif action == "analyze_news":
        return model.analyze_news.remote(
            news_items=request.get("news_items", []),
            domain=request.get("domain", "geopolitical"),
        )

    elif action == "generate_briefing":
        return model.generate_briefing.remote(
            preset=request.get("preset", "global"),
            metrics=request.get("metrics", {}),
            alerts=request.get("alerts", []),
        )

    else:
        return {"error": f"Unknown action: {action}"}


# ============================================
# MODEL UPLOAD HELPER
# ============================================

@app.function(
    image=image,
    volumes={"/model": model_volume},
    timeout=3600,  # 1 hour for large uploads
)
def upload_model_from_gdrive(gdrive_file_id: str):
    """
    Download model from Google Drive and save to Modal volume.

    Run: modal run modal_phi2_intel.py::upload_model_from_gdrive --gdrive-file-id="YOUR_ID"

    Or better: Upload to HuggingFace first, then the model loads automatically.
    """
    import subprocess
    import os

    # Install gdown
    subprocess.run(["pip", "install", "gdown"], check=True)
    import gdown

    # Download
    output_path = "/model/phi2-intel-finetuned"
    os.makedirs(output_path, exist_ok=True)

    # If it's a folder, use folder download
    gdown.download_folder(
        id=gdrive_file_id,
        output=output_path,
        quiet=False,
    )

    # Commit to volume
    model_volume.commit()

    return {"status": "success", "path": output_path}


# ============================================
# LOCAL TESTING
# ============================================

@app.local_entrypoint()
def main():
    """Test the model locally."""
    model = Phi2IntelModel()

    # Test basic generation
    result = model.generate.remote(
        prompt="What are the key indicators of political instability in Eastern Europe?",
        max_new_tokens=256,
        temperature=0.7,
    )
    print(f"Generated ({result['latency_ms']}ms):")
    print(result["text"])
    print()

    # Test news analysis
    test_news = [
        {"title": "NATO announces increased presence in Baltic states", "domain": "defense"},
        {"title": "EU sanctions target Russian energy sector", "domain": "financial"},
        {"title": "Cyber attacks on critical infrastructure increase", "domain": "cyber"},
    ]

    analysis = model.analyze_news.remote(
        news_items=test_news,
        domain="geopolitical",
    )
    print(f"Analysis ({analysis['model_latency_ms']}ms):")
    print(analysis["analysis"])
