# Chapter 24: Self-Hosted Inference — Serverless LLMs with Rust, WASM, and Next.js

The major LLM providers charge per token. At scale, this becomes expensive. At edge, latency kills user experience. For privacy-sensitive applications, sending data to third parties is unacceptable.

This chapter shows how to build your own inference infrastructure using modern web technologies: serverless architecture, Rust compiled to WebAssembly, and Next.js for the application layer. No Python. No GPU servers. Just fast, portable, privacy-preserving inference.

---

## Why Self-Host Inference?

### The Cost Problem

API pricing at scale:

| Provider | Input (1M tokens) | Output (1M tokens) |
|----------|-------------------|---------------------|
| GPT-4 Turbo | $10.00 | $30.00 |
| Claude 3 Opus | $15.00 | $75.00 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| GPT-3.5 Turbo | $0.50 | $1.50 |

A moderately active application processing 100M tokens/month:
- GPT-4 Turbo: $4,000/month
- Self-hosted: ~$200/month (serverless compute)

### The Latency Problem

Round-trip to cloud APIs:
```
User → Your Server → API Provider → Your Server → User
      ~50ms            ~200-500ms        ~50ms

Total: 300-600ms minimum, often 1-3 seconds
```

Edge inference:
```
User → Edge Node → User
      ~10-50ms

Total: 10-50ms
```

For real-time applications (autocomplete, live translation, gaming), cloud APIs are too slow.

### The Privacy Problem

Every API call sends user data to:
- The provider's servers
- Their logging systems
- Potentially their training pipelines

For healthcare, legal, financial, or personal applications, this is unacceptable.

### The Control Problem

API providers can:
- Change pricing without notice
- Deprecate models
- Add content restrictions
- Rate limit during peak usage
- Go offline entirely

Self-hosting gives you control.

---

## Architecture Overview

### The Modern Stack

```
┌─────────────────────────────────────────────────────────┐
│                      Next.js App                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   React UI  │  │  API Routes │  │  Edge Runtime   │ │
│  └─────────────┘  └──────┬──────┘  └────────┬────────┘ │
└──────────────────────────┼──────────────────┼──────────┘
                           │                  │
                           ▼                  ▼
              ┌────────────────────┐  ┌───────────────┐
              │  Serverless Fn     │  │  WASM Module  │
              │  (Vercel/Lambda)   │  │  (Browser/Edge)│
              └─────────┬──────────┘  └───────┬───────┘
                        │                     │
                        ▼                     ▼
              ┌────────────────────────────────────────┐
              │         Rust Inference Engine          │
              │  (llama.cpp bindings / Candle / Burn)  │
              └────────────────────────────────────────┘
```

### Why Rust?

**Performance:** Near-C speed without manual memory management.

**Safety:** No null pointer exceptions, no data races, no buffer overflows.

**Portability:** Compiles to native binaries AND WebAssembly.

**Ecosystem:** Growing ML ecosystem (candle, burn, llama-cpp-rs).

### Why WebAssembly?

**Universal runtime:** Runs in browsers, edge functions, serverless platforms.

**Near-native speed:** 1.1-1.5x native performance in practice.

**Sandboxed:** Memory-safe by design, can't escape the sandbox.

**Portable:** Same binary runs everywhere WASM runs.

### Why Next.js?

**Full-stack:** Frontend and backend in one framework.

**Edge runtime:** Deploy inference to CDN edge nodes.

**Streaming:** Built-in support for streaming responses.

**Serverless:** Deploy to Vercel, Netlify, or self-host.

---

## Rust Inference Engines

### Option 1: llama-cpp-rs (Recommended for Production)

Rust bindings to llama.cpp, the industry standard for CPU inference.

```rust
// Cargo.toml
[dependencies]
llama-cpp-2 = "0.1"
tokio = { version = "1", features = ["full"] }

// src/inference.rs
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use std::num::NonZeroU32;

pub struct InferenceEngine {
    backend: LlamaBackend,
    model: LlamaModel,
}

impl InferenceEngine {
    pub fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let backend = LlamaBackend::init()?;

        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;

        Ok(Self { backend, model })
    }

    pub fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String, Box<dyn std::error::Error>> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(2048));

        let mut ctx = self.model.new_context(&self.backend, ctx_params)?;

        // Tokenize input
        let tokens = self.model.str_to_token(prompt, llama_cpp_2::model::AddBos::Always)?;

        // Create batch
        let mut batch = LlamaBatch::new(512, 1);

        for (i, token) in tokens.iter().enumerate() {
            batch.add(*token, i as i32, &[0], i == tokens.len() - 1)?;
        }

        // Decode initial prompt
        ctx.decode(&mut batch)?;

        // Generate tokens
        let mut output = String::new();
        let mut n_cur = tokens.len();

        for _ in 0..max_tokens {
            let candidates = ctx.candidates_ith(batch.n_tokens() - 1);
            let mut candidates_p = LlamaTokenDataArray::from_iter(candidates, false);

            // Sample next token
            let new_token_id = ctx.sample_token_greedy(&mut candidates_p);

            // Check for end of generation
            if self.model.is_eog_token(new_token_id) {
                break;
            }

            // Decode token to string
            let token_str = self.model.token_to_str(new_token_id, llama_cpp_2::token::Special::Tokenize)?;
            output.push_str(&token_str);

            // Prepare next batch
            batch.clear();
            batch.add(new_token_id, n_cur as i32, &[0], true)?;

            ctx.decode(&mut batch)?;
            n_cur += 1;
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_engine_creation() {
        // This test requires a model file
        // Skip in CI, run locally with actual model
        if std::env::var("CI").is_ok() {
            return;
        }

        let engine = InferenceEngine::new("models/llama-2-7b.Q4_K_M.gguf");
        assert!(engine.is_ok());
    }
}
```

### Option 2: Candle (Pure Rust, WASM-Compatible)

Hugging Face's pure Rust ML framework. No C dependencies means easy WASM compilation.

```rust
// Cargo.toml
[dependencies]
candle-core = "0.4"
candle-nn = "0.4"
candle-transformers = "0.4"
tokenizers = "0.15"
hf-hub = "0.3"

// src/candle_inference.rs
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Config, Llama, Cache};
use tokenizers::Tokenizer;

pub struct CandleEngine {
    model: Llama,
    tokenizer: Tokenizer,
    device: Device,
    cache: Cache,
}

impl CandleEngine {
    pub fn new(model_id: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Use CPU for portability (GPU available via features)
        let device = Device::Cpu;

        // Load from Hugging Face Hub
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(model_id.to_string());

        let tokenizer_path = repo.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;

        let config_path = repo.get("config.json")?;
        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;

        let weights_path = repo.get("model.safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F16, &device)? };

        let model = Llama::load(vb, &config)?;
        let cache = Cache::new(true, candle_core::DType::F16, &config, &device)?;

        Ok(Self { model, tokenizer, device, cache })
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String, Box<dyn std::error::Error>> {
        let encoding = self.tokenizer.encode(prompt, true)?;
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

        let mut output_tokens = Vec::new();

        for index in 0..max_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let context = &tokens[start_pos..];

            let input = Tensor::new(context, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos, &mut self.cache)?;

            let logits = logits.squeeze(0)?.squeeze(0)?;
            let next_token = self.sample_argmax(&logits)?;

            tokens.push(next_token);
            output_tokens.push(next_token);

            // Check for EOS
            if next_token == self.tokenizer.token_to_id("</s>").unwrap_or(2) {
                break;
            }
        }

        let output = self.tokenizer.decode(&output_tokens, true)?;
        Ok(output)
    }

    fn sample_argmax(&self, logits: &Tensor) -> Result<u32, Box<dyn std::error::Error>> {
        let logits_v: Vec<f32> = logits.to_vec1()?;
        let next_token = logits_v
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();
        Ok(next_token)
    }
}
```

### Option 3: Burn (Tensor Library with Backend Flexibility)

```rust
// Cargo.toml
[dependencies]
burn = "0.13"
burn-wgpu = "0.13"  # WebGPU backend for WASM
burn-ndarray = "0.13"  # CPU fallback

// src/burn_inference.rs
use burn::prelude::*;
use burn::tensor::backend::Backend;

// Burn allows switching backends at compile time
#[cfg(target_arch = "wasm32")]
type MyBackend = burn_wgpu::Wgpu;

#[cfg(not(target_arch = "wasm32"))]
type MyBackend = burn_ndarray::NdArray;

pub struct BurnEngine<B: Backend> {
    device: B::Device,
    // Model weights would be loaded here
}

impl<B: Backend> BurnEngine<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    // Implementation follows Burn's model patterns
}
```

---

## Compiling to WebAssembly

### Setup for WASM Target

```bash
# Install wasm-pack
cargo install wasm-pack

# Add WASM target
rustup target add wasm32-unknown-unknown

# For WASI (server-side WASM)
rustup target add wasm32-wasi
```

### WASM-Compatible Inference Module

```rust
// src/lib.rs
use wasm_bindgen::prelude::*;
use candle_core::{Device, Tensor};

#[wasm_bindgen]
pub struct WasmInference {
    // Simplified model for WASM
    // Full models need careful memory management
}

#[wasm_bindgen]
impl WasmInference {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmInference, JsValue> {
        // Initialize with limited model for browser
        Ok(WasmInference {})
    }

    #[wasm_bindgen]
    pub fn generate(&mut self, prompt: &str, max_tokens: u32) -> Result<String, JsValue> {
        // WASM inference implementation
        // Note: Browser memory limits apply (~2-4GB)
        Ok(format!("Generated response for: {}", prompt))
    }

    #[wasm_bindgen]
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, JsValue> {
        // Tokenization in WASM
        Ok(vec![1, 2, 3]) // Placeholder
    }
}

// Build with: wasm-pack build --target web
```

### Memory Considerations for WASM

Browser WASM has memory limits:

| Browser | Default Limit | Maximum |
|---------|---------------|---------|
| Chrome | 2GB | 4GB |
| Firefox | 2GB | 4GB |
| Safari | 1GB | 2GB |

This limits model size. Use quantization:

| Quantization | Model Size (7B params) | Memory Required |
|--------------|------------------------|-----------------|
| F32 | 28GB | Too large |
| F16 | 14GB | Too large |
| Q8_0 | 7GB | Too large |
| Q4_K_M | 4GB | Possible |
| Q4_0 | 3.5GB | Comfortable |
| Q2_K | 2.5GB | Comfortable |

For browser inference, target Q4_0 or Q2_K quantized models.

---

## Next.js Integration

### Project Structure

```
my-llm-app/
├── app/
│   ├── api/
│   │   └── generate/
│   │       └── route.ts       # API route for inference
│   ├── page.tsx               # Main UI
│   └── layout.tsx
├── lib/
│   ├── inference/
│   │   ├── worker.ts          # Web Worker for WASM
│   │   └── client.ts          # Client-side inference
│   └── wasm/
│       └── inference_bg.wasm  # Compiled WASM module
├── rust/
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs             # Rust inference code
├── next.config.js
└── package.json
```

### API Route (Serverless Function)

```typescript
// app/api/generate/route.ts
import { NextRequest, NextResponse } from 'next/server';

// For serverless, use pre-compiled native binary or WASI
// This example uses an external inference service
// In production, bundle the Rust binary with your deployment

interface GenerateRequest {
  prompt: string;
  maxTokens?: number;
  temperature?: number;
  stream?: boolean;
}

interface GenerateResponse {
  text: string;
  tokens: number;
  latencyMs: number;
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  const startTime = performance.now();

  try {
    const body: GenerateRequest = await request.json();
    const { prompt, maxTokens = 256, temperature = 0.7, stream = false } = body;

    if (!prompt || typeof prompt !== 'string') {
      return NextResponse.json(
        { error: 'Invalid prompt' },
        { status: 400 }
      );
    }

    if (stream) {
      return handleStreamingResponse(prompt, maxTokens, temperature);
    }

    // Non-streaming response
    const result = await runInference(prompt, maxTokens, temperature);

    const response: GenerateResponse = {
      text: result.text,
      tokens: result.tokenCount,
      latencyMs: performance.now() - startTime,
    };

    return NextResponse.json(response);

  } catch (error) {
    console.error('Inference error:', error);
    return NextResponse.json(
      { error: 'Inference failed' },
      { status: 500 }
    );
  }
}

async function runInference(
  prompt: string,
  maxTokens: number,
  temperature: number
): Promise<{ text: string; tokenCount: number }> {
  // Option 1: Call local binary via child_process (for Node.js runtime)
  // Option 2: Call WASI module
  // Option 3: Call external inference service

  // Example using child_process (Node.js runtime only)
  const { execSync } = await import('child_process');

  const result = execSync(
    `./inference-bin --prompt "${prompt.replace(/"/g, '\\"')}" --max-tokens ${maxTokens} --temperature ${temperature}`,
    { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 }
  );

  return {
    text: result.trim(),
    tokenCount: result.split(/\s+/).length, // Approximate
  };
}

function handleStreamingResponse(
  prompt: string,
  maxTokens: number,
  temperature: number
): NextResponse {
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      try {
        // Stream tokens as they're generated
        for await (const token of streamInference(prompt, maxTokens, temperature)) {
          const data = `data: ${JSON.stringify({ token })}\n\n`;
          controller.enqueue(encoder.encode(data));
        }
        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
      } catch (error) {
        controller.error(error);
      }
    },
  });

  return new NextResponse(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}

async function* streamInference(
  prompt: string,
  maxTokens: number,
  temperature: number
): AsyncGenerator<string> {
  // Implement streaming inference
  // This would call the Rust binary with streaming output
  yield 'Hello';
  yield ' world';
  yield '!';
}
```

### Edge Runtime (For Global Distribution)

```typescript
// app/api/generate-edge/route.ts
import { NextRequest } from 'next/server';

// Edge runtime for lower latency
export const runtime = 'edge';

export async function POST(request: NextRequest) {
  const { prompt } = await request.json();

  // Edge functions can't run native binaries
  // Options:
  // 1. Call a backend inference service
  // 2. Use WASM (with memory limits)
  // 3. Use a smaller model that fits in edge constraints

  // Example: Call backend service
  const response = await fetch(process.env.INFERENCE_SERVICE_URL!, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  });

  return response;
}
```

### Client-Side WASM Inference

```typescript
// lib/inference/client.ts
import init, { WasmInference } from '../wasm/inference';

let inference: WasmInference | null = null;
let initPromise: Promise<void> | null = null;

export async function initializeInference(): Promise<void> {
  if (inference) return;

  if (!initPromise) {
    initPromise = (async () => {
      await init();
      inference = new WasmInference();
    })();
  }

  await initPromise;
}

export async function generateClientSide(
  prompt: string,
  maxTokens: number = 256
): Promise<string> {
  await initializeInference();

  if (!inference) {
    throw new Error('Inference not initialized');
  }

  return inference.generate(prompt, maxTokens);
}

// Web Worker wrapper for non-blocking inference
export function createInferenceWorker(): Worker {
  return new Worker(
    new URL('./worker.ts', import.meta.url),
    { type: 'module' }
  );
}
```

### Web Worker for Background Inference

```typescript
// lib/inference/worker.ts
import init, { WasmInference } from '../wasm/inference';

let inference: WasmInference | null = null;

self.onmessage = async (event: MessageEvent) => {
  const { type, payload, id } = event.data;

  try {
    switch (type) {
      case 'init':
        await init();
        inference = new WasmInference();
        self.postMessage({ id, type: 'init_complete' });
        break;

      case 'generate':
        if (!inference) {
          throw new Error('Not initialized');
        }
        const result = inference.generate(payload.prompt, payload.maxTokens);
        self.postMessage({ id, type: 'result', payload: result });
        break;

      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (error) {
    self.postMessage({
      id,
      type: 'error',
      payload: error instanceof Error ? error.message : 'Unknown error'
    });
  }
};
```

### React Component

```tsx
// app/components/InferenceChat.tsx
'use client';

import { useState, useCallback, useRef, useEffect } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export default function InferenceChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [useClientSide, setUseClientSide] = useState(false);
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    // Initialize Web Worker for client-side inference
    if (useClientSide && !workerRef.current) {
      workerRef.current = new Worker(
        new URL('../lib/inference/worker.ts', import.meta.url),
        { type: 'module' }
      );

      workerRef.current.postMessage({ type: 'init', id: 'init' });
    }

    return () => {
      workerRef.current?.terminate();
    };
  }, [useClientSide]);

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      let response: string;

      if (useClientSide && workerRef.current) {
        // Client-side WASM inference
        response = await new Promise((resolve, reject) => {
          const id = crypto.randomUUID();

          const handler = (event: MessageEvent) => {
            if (event.data.id === id) {
              workerRef.current?.removeEventListener('message', handler);
              if (event.data.type === 'error') {
                reject(new Error(event.data.payload));
              } else {
                resolve(event.data.payload);
              }
            }
          };

          workerRef.current?.addEventListener('message', handler);
          workerRef.current?.postMessage({
            type: 'generate',
            id,
            payload: { prompt: input, maxTokens: 256 }
          });
        });
      } else {
        // Server-side inference via API
        const res = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: input, maxTokens: 256 }),
        });

        if (!res.ok) throw new Error('Inference failed');
        const data = await res.json();
        response = data.text;
      }

      const assistantMessage: Message = { role: 'assistant', content: response };
      setMessages(prev => [...prev, assistantMessage]);

    } catch (error) {
      console.error('Generation error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Error generating response. Please try again.'
      }]);
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, useClientSide]);

  return (
    <div className="flex flex-col h-screen max-w-2xl mx-auto p-4">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-bold">Self-Hosted Inference</h1>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={useClientSide}
            onChange={(e) => setUseClientSide(e.target.checked)}
          />
          <span className="text-sm">Client-side (WASM)</span>
        </label>
      </div>

      <div className="flex-1 overflow-y-auto space-y-4 mb-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`p-3 rounded-lg ${
              msg.role === 'user'
                ? 'bg-blue-100 ml-auto max-w-[80%]'
                : 'bg-gray-100 mr-auto max-w-[80%]'
            }`}
          >
            {msg.content}
          </div>
        ))}
        {isLoading && (
          <div className="bg-gray-100 p-3 rounded-lg mr-auto">
            <span className="animate-pulse">Generating...</span>
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          className="flex-1 p-2 border rounded"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </div>
  );
}
```

---

## Serverless Deployment Strategies

### Strategy 1: Vercel with Custom Runtime

```javascript
// next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverComponentsExternalPackages: ['@node-llama-cpp'],
  },
  webpack: (config, { isServer }) => {
    if (isServer) {
      // Include native binaries in server bundle
      config.externals.push({
        '@node-llama-cpp': 'commonjs @node-llama-cpp',
      });
    }

    // WASM support
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    };

    return config;
  },
};

module.exports = nextConfig;
```

### Strategy 2: Cloudflare Workers with WASM

```typescript
// workers/inference.ts
import wasmModule from '../lib/wasm/inference_bg.wasm';

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method !== 'POST') {
      return new Response('Method not allowed', { status: 405 });
    }

    const { prompt, maxTokens } = await request.json();

    // Initialize WASM
    const instance = await WebAssembly.instantiate(wasmModule);

    // Run inference
    // Note: Cloudflare Workers have 128MB memory limit
    // Use highly quantized models only

    return new Response(JSON.stringify({ text: 'response' }), {
      headers: { 'Content-Type': 'application/json' },
    });
  },
};
```

### Strategy 3: AWS Lambda with Container

```dockerfile
# Dockerfile for Lambda container
FROM public.ecr.aws/lambda/provided:al2

# Install Rust and build inference binary
RUN yum install -y gcc make
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

COPY rust/ /rust/
WORKDIR /rust
RUN cargo build --release

# Copy Lambda bootstrap
COPY bootstrap /var/runtime/bootstrap
RUN chmod +x /var/runtime/bootstrap

# Copy model (or download at runtime)
COPY models/ /opt/models/

ENTRYPOINT ["/var/runtime/bootstrap"]
```

```bash
# bootstrap
#!/bin/bash
set -euo pipefail

while true; do
  HEADERS="$(mktemp)"
  EVENT_DATA=$(curl -sS -LD "$HEADERS" "http://${AWS_LAMBDA_RUNTIME_API}/2018-06-01/runtime/invocation/next")
  REQUEST_ID=$(grep -Fi Lambda-Runtime-Aws-Request-Id "$HEADERS" | tr -d '[:space:]' | cut -d: -f2)

  # Parse event and run inference
  RESPONSE=$(/rust/target/release/inference-bin --event "$EVENT_DATA")

  curl -X POST "http://${AWS_LAMBDA_RUNTIME_API}/2018-06-01/runtime/invocation/$REQUEST_ID/response" -d "$RESPONSE"
done
```

### Strategy 4: Modal Labs (Recommended for Simplicity)

```python
# deploy.py - Modal deployment
# Note: Uses Python wrapper but calls Rust binary
import modal

app = modal.App("llm-inference")

image = (
    modal.Image.debian_slim()
    .apt_install("curl", "build-essential")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "~/.cargo/bin/cargo install --git https://github.com/ggerganov/llama.cpp llama-cli",
    )
    .pip_install("huggingface_hub")
)

@app.cls(
    image=image,
    gpu="T4",  # or "A10G" for faster inference
    container_idle_timeout=300,
)
class Inference:
    @modal.enter()
    def load_model(self):
        from huggingface_hub import hf_hub_download
        self.model_path = hf_hub_download(
            "TheBloke/Llama-2-7B-GGUF",
            "llama-2-7b.Q4_K_M.gguf"
        )

    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        import subprocess
        result = subprocess.run(
            [
                "llama-cli",
                "-m", self.model_path,
                "-p", prompt,
                "-n", str(max_tokens),
            ],
            capture_output=True,
            text=True,
        )
        return result.stdout

@app.local_entrypoint()
def main():
    inference = Inference()
    result = inference.generate.remote("What is the capital of France?")
    print(result)
```

---

## Model Selection and Quantization

### Choosing the Right Model

| Use Case | Recommended Model | Size | Quality |
|----------|-------------------|------|---------|
| Simple Q&A | Phi-3-mini | 2B | Good |
| General chat | Llama-3.2-3B | 3B | Good |
| Code generation | CodeLlama-7B | 7B | Very Good |
| Complex reasoning | Llama-3.1-8B | 8B | Excellent |
| Full capability | Llama-3.1-70B | 70B | Best |

### Quantization Trade-offs

```
Quality vs Size (7B model):

F16:     ████████████████████  100% quality, 14GB
Q8_0:    ███████████████████░   95% quality,  7GB
Q6_K:    ██████████████████░░   92% quality,  5.5GB
Q5_K_M:  █████████████████░░░   90% quality,  4.8GB
Q4_K_M:  ████████████████░░░░   87% quality,  4GB    ← Sweet spot
Q4_0:    ███████████████░░░░░   85% quality,  3.5GB
Q3_K_M:  ██████████████░░░░░░   82% quality,  3GB
Q2_K:    ████████████░░░░░░░░   75% quality,  2.5GB  ← Browser viable
```

### Downloading Quantized Models

```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf

# Using wget
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf
```

---

## Cost Analysis

### Self-Hosted vs API Pricing

**Scenario: 100M tokens/month**

| Option | Monthly Cost | Setup Effort | Latency |
|--------|--------------|--------------|---------|
| GPT-4 Turbo | $4,000 | None | 200-500ms |
| Claude 3.5 Sonnet | $1,800 | None | 200-500ms |
| Modal Labs (T4) | $150-300 | Low | 50-150ms |
| AWS Lambda | $100-200 | Medium | 100-300ms |
| Vercel Serverless | $200-400 | Low | 50-200ms |
| Self-hosted GPU | $500-800 | High | 20-100ms |
| Browser WASM | $0 (user's device) | Medium | 200-2000ms |

### Break-Even Analysis

```typescript
// cost_analysis.ts
interface CostConfig {
  tokensPerMonth: number;
  apiPricePerMToken: number;
  serverlessCostPerInvocation: number;
  avgTokensPerInvocation: number;
  serverlessBaseCost: number;
}

function calculateBreakEven(config: CostConfig): {
  apiMonthlyCost: number;
  serverlessMonthlyCost: number;
  savings: number;
  breakEvenTokens: number;
} {
  const {
    tokensPerMonth,
    apiPricePerMToken,
    serverlessCostPerInvocation,
    avgTokensPerInvocation,
    serverlessBaseCost
  } = config;

  const apiMonthlyCost = (tokensPerMonth / 1_000_000) * apiPricePerMToken;

  const invocationsPerMonth = tokensPerMonth / avgTokensPerInvocation;
  const serverlessMonthlyCost = serverlessBaseCost +
    (invocationsPerMonth * serverlessCostPerInvocation);

  const savings = apiMonthlyCost - serverlessMonthlyCost;

  // Break-even: API cost = Serverless cost
  // (tokens / 1M) * apiPrice = baseCost + (tokens / avgTokens) * invocationCost
  const breakEvenTokens = serverlessBaseCost /
    ((apiPricePerMToken / 1_000_000) - (serverlessCostPerInvocation / avgTokensPerInvocation));

  return { apiMonthlyCost, serverlessMonthlyCost, savings, breakEvenTokens };
}

// Example: GPT-4 vs Modal Labs
const analysis = calculateBreakEven({
  tokensPerMonth: 100_000_000,
  apiPricePerMToken: 40, // $10 input + $30 output avg
  serverlessCostPerInvocation: 0.001, // $0.001 per invocation
  avgTokensPerInvocation: 500,
  serverlessBaseCost: 50, // Base infrastructure
});

console.log(`API cost: $${analysis.apiMonthlyCost}`);
console.log(`Serverless cost: $${analysis.serverlessMonthlyCost}`);
console.log(`Monthly savings: $${analysis.savings}`);
console.log(`Break-even at: ${analysis.breakEvenTokens.toLocaleString()} tokens/month`);
```

---

## Security Considerations

### Input Sanitization

```typescript
// lib/security.ts
const MAX_PROMPT_LENGTH = 8192;
const BLOCKED_PATTERNS = [
  /ignore previous instructions/i,
  /system prompt/i,
  /\{\{.*\}\}/,  // Template injection
];

export function sanitizePrompt(prompt: string): {
  sanitized: string;
  warnings: string[]
} {
  const warnings: string[] = [];
  let sanitized = prompt;

  // Length check
  if (sanitized.length > MAX_PROMPT_LENGTH) {
    sanitized = sanitized.slice(0, MAX_PROMPT_LENGTH);
    warnings.push('Prompt truncated to maximum length');
  }

  // Pattern check
  for (const pattern of BLOCKED_PATTERNS) {
    if (pattern.test(sanitized)) {
      warnings.push(`Blocked pattern detected: ${pattern}`);
      sanitized = sanitized.replace(pattern, '[BLOCKED]');
    }
  }

  // Remove null bytes and control characters
  sanitized = sanitized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');

  return { sanitized, warnings };
}
```

### Rate Limiting

```typescript
// lib/rateLimit.ts
import { LRUCache } from 'lru-cache';

interface RateLimitConfig {
  windowMs: number;
  maxRequests: number;
}

const cache = new LRUCache<string, number[]>({
  max: 10000,
  ttl: 60 * 60 * 1000, // 1 hour
});

export function checkRateLimit(
  identifier: string,
  config: RateLimitConfig
): { allowed: boolean; remaining: number; resetAt: number } {
  const now = Date.now();
  const windowStart = now - config.windowMs;

  const requests = cache.get(identifier) || [];
  const recentRequests = requests.filter(t => t > windowStart);

  const allowed = recentRequests.length < config.maxRequests;
  const remaining = Math.max(0, config.maxRequests - recentRequests.length);
  const resetAt = recentRequests.length > 0
    ? recentRequests[0] + config.windowMs
    : now + config.windowMs;

  if (allowed) {
    recentRequests.push(now);
    cache.set(identifier, recentRequests);
  }

  return { allowed, remaining, resetAt };
}
```

### Model Output Filtering

```typescript
// lib/outputFilter.ts
const SENSITIVE_PATTERNS = [
  /\b\d{3}-\d{2}-\d{4}\b/,  // SSN
  /\b\d{16}\b/,             // Credit card
  /password\s*[:=]\s*\S+/i,  // Passwords
];

export function filterOutput(output: string): {
  filtered: string;
  redactions: number;
} {
  let filtered = output;
  let redactions = 0;

  for (const pattern of SENSITIVE_PATTERNS) {
    const matches = filtered.match(new RegExp(pattern, 'g'));
    if (matches) {
      redactions += matches.length;
      filtered = filtered.replace(new RegExp(pattern, 'g'), '[REDACTED]');
    }
  }

  return { filtered, redactions };
}
```

---

## Testing

### Unit Tests

```typescript
// __tests__/inference.test.ts
import { sanitizePrompt } from '../lib/security';
import { checkRateLimit } from '../lib/rateLimit';

describe('sanitizePrompt', () => {
  it('truncates long prompts', () => {
    const longPrompt = 'a'.repeat(10000);
    const { sanitized, warnings } = sanitizePrompt(longPrompt);
    expect(sanitized.length).toBe(8192);
    expect(warnings).toContain('Prompt truncated to maximum length');
  });

  it('blocks injection patterns', () => {
    const { sanitized, warnings } = sanitizePrompt('Please ignore previous instructions');
    expect(sanitized).toContain('[BLOCKED]');
    expect(warnings.length).toBeGreaterThan(0);
  });

  it('removes control characters', () => {
    const { sanitized } = sanitizePrompt('Hello\x00World');
    expect(sanitized).toBe('HelloWorld');
  });
});

describe('checkRateLimit', () => {
  it('allows requests under limit', () => {
    const result = checkRateLimit('test-user-1', {
      windowMs: 60000,
      maxRequests: 10,
    });
    expect(result.allowed).toBe(true);
    expect(result.remaining).toBe(9);
  });

  it('blocks requests over limit', () => {
    const config = { windowMs: 60000, maxRequests: 2 };
    checkRateLimit('test-user-2', config);
    checkRateLimit('test-user-2', config);
    const result = checkRateLimit('test-user-2', config);
    expect(result.allowed).toBe(false);
    expect(result.remaining).toBe(0);
  });
});
```

### Integration Tests

```typescript
// __tests__/api.integration.test.ts
import { createMocks } from 'node-mocks-http';
import { POST } from '../app/api/generate/route';

describe('Generate API', () => {
  it('returns 400 for missing prompt', async () => {
    const { req } = createMocks({
      method: 'POST',
      body: {},
    });

    const response = await POST(req as any);
    expect(response.status).toBe(400);
  });

  it('returns generated text for valid prompt', async () => {
    const { req } = createMocks({
      method: 'POST',
      body: { prompt: 'Hello', maxTokens: 10 },
    });

    const response = await POST(req as any);
    expect(response.status).toBe(200);

    const data = await response.json();
    expect(data).toHaveProperty('text');
    expect(data).toHaveProperty('latencyMs');
  });
});
```

### Load Tests

```typescript
// __tests__/load.test.ts
import autocannon from 'autocannon';

async function runLoadTest() {
  const result = await autocannon({
    url: 'http://localhost:3000/api/generate',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      prompt: 'What is 2+2?',
      maxTokens: 10,
    }),
    connections: 10,
    duration: 30,
  });

  console.log('Requests/sec:', result.requests.average);
  console.log('Latency (avg):', result.latency.average, 'ms');
  console.log('Latency (p99):', result.latency.p99, 'ms');
  console.log('Errors:', result.errors);

  // Assertions
  expect(result.errors).toBe(0);
  expect(result.latency.p99).toBeLessThan(5000);
}

describe('Load Test', () => {
  it('handles concurrent requests', runLoadTest, 60000);
});
```

---

## Complete Example: Production Setup

### Directory Structure

```
llm-app/
├── app/
│   ├── api/
│   │   ├── generate/
│   │   │   └── route.ts
│   │   └── health/
│   │       └── route.ts
│   ├── page.tsx
│   └── layout.tsx
├── lib/
│   ├── inference/
│   │   ├── engine.ts
│   │   ├── worker.ts
│   │   └── types.ts
│   ├── security/
│   │   ├── sanitize.ts
│   │   ├── rateLimit.ts
│   │   └── filter.ts
│   └── wasm/
│       ├── inference_bg.wasm
│       └── inference.js
├── rust/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       └── bin/
│           └── inference.rs
├── models/
│   └── .gitkeep
├── __tests__/
│   ├── unit/
│   ├── integration/
│   └── load/
├── scripts/
│   ├── download-model.sh
│   ├── build-wasm.sh
│   └── deploy.sh
├── next.config.js
├── package.json
├── tsconfig.json
└── README.md
```

### Setup Script

```bash
#!/bin/bash
# scripts/setup.sh

set -e

echo "Setting up LLM inference app..."

# Install Node dependencies
npm install

# Install Rust (if not present)
if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
fi

# Add WASM target
rustup target add wasm32-unknown-unknown

# Build Rust inference binary
cd rust
cargo build --release
cd ..

# Build WASM module
./scripts/build-wasm.sh

# Download default model (optional)
echo "Download a model? (y/n)"
read -r response
if [[ "$response" == "y" ]]; then
    ./scripts/download-model.sh
fi

echo "Setup complete!"
echo "Run 'npm run dev' to start development server"
```

### Build WASM Script

```bash
#!/bin/bash
# scripts/build-wasm.sh

set -e

cd rust

# Build with wasm-pack
wasm-pack build --target web --out-dir ../lib/wasm

# Optimize WASM
if command -v wasm-opt &> /dev/null; then
    wasm-opt -O3 ../lib/wasm/inference_bg.wasm -o ../lib/wasm/inference_bg.wasm
fi

echo "WASM build complete"
```

### Download Model Script

```bash
#!/bin/bash
# scripts/download-model.sh

MODEL_URL="https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf"
MODEL_PATH="models/llama-2-7b.Q4_K_M.gguf"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading model..."
    curl -L "$MODEL_URL" -o "$MODEL_PATH"
    echo "Model downloaded to $MODEL_PATH"
else
    echo "Model already exists at $MODEL_PATH"
fi
```

---

## Summary

Self-hosted inference gives you:

**Control:**
- Choose your models
- Set your safety policies
- Own your data

**Cost efficiency:**
- 10-20x cheaper at scale
- Predictable pricing
- No per-token charges

**Performance:**
- Sub-100ms latency at edge
- No cold starts (with warm pools)
- Global distribution

**Privacy:**
- Data never leaves your infrastructure
- No third-party logging
- GDPR/HIPAA compliance easier

The modern stack—Rust for performance, WASM for portability, Next.js for the application layer—makes this achievable without managing GPU servers.

Start small: deploy a quantized model on Modal or Vercel, measure your usage, then scale to dedicated infrastructure as needed.

The code in this chapter is available at the repository. Fork it, adapt it, make it yours.

---

*"The best API is the one you control."*
