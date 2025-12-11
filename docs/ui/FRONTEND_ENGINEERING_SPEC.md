# LatticeForge UI: Frontend Engineering Specification

**Stack:** Next.js 16 / React 19 / TypeScript / Tailwind CSS
**Audience:** Frontend Engineers, Full-Stack Engineers
**Last Updated:** December 2024

---

## 1. Overview and Technical Context

This document specifies the frontend architecture for LatticeForge. You're building a real-time intelligence dashboard that processes streaming signal data, renders interactive visualizations, and orchestrates LLM inference via RunPod serverless.

### 1.1 Technology Stack

| Layer | Technology | Notes |
|-------|------------|-------|
| Framework | Next.js 16 | App Router, React Server Components |
| UI Library | React 19 | Concurrent features, use() hook |
| Styling | Tailwind CSS 3.4 | Utility-first, glassmorphism theme |
| State | React hooks + Context | Server state via fetch, client state via useState |
| Auth | Supabase SSR | Session management, RLS |
| Payments | Stripe | Subscriptions, webhooks |
| LLM Inference | RunPod Serverless | Elle-72B + Guardian via vLLM |
| 3D | React Three Fiber | Three.js bindings for globe visualization |
| Charts | Recharts | Declarative charting |
| Icons | Lucide React | Consistent icon set |
| Rate Limiting | Upstash Redis | Guardian security layer |

### 1.2 Why This Stack

1. **Next.js 16** — Server components reduce client bundle, streaming SSR for fast TTFB
2. **React 19** — Concurrent rendering, improved Suspense, use() hook for async data
3. **Tailwind** — Rapid iteration, consistent design tokens, easy dark mode
4. **Supabase** — Real-time subscriptions, Row Level Security, managed Postgres
5. **RunPod** — Cost-effective GPU inference for Elle-72B (~$0.001/request vs $0.50+ for cloud LLMs)

### 1.3 What You're Building

The UI has four major surfaces:

1. **Dashboard** — Real-time signal grid with phase transition alerts, causal graphs, regime indicators
2. **Intelligence Briefings** — LLM-generated summaries with source fusion, confidence intervals
3. **PROMETHEUS Dashboard** — Research insights visualization with pipeline status
4. **Admin Panel** — User management, API monitoring, training data export

---

## 2. Project Structure

```
packages/web/
├── app/                        # Next.js App Router
│   ├── (auth)/                 # Auth routes (login, callback)
│   ├── (legal)/                # Legal pages (terms, privacy)
│   ├── admin/                  # Admin panel routes
│   │   ├── insights/           # Elle research insights
│   │   └── training/           # Training data management
│   ├── api/                    # API routes
│   │   ├── cron/               # Vercel cron jobs
│   │   ├── elle/               # Elle inference endpoints
│   │   │   ├── ask/            # Single question
│   │   │   ├── chat/           # Conversational
│   │   │   └── insights/       # Research insights
│   │   ├── intel-briefing/     # Briefing generation
│   │   └── webhooks/           # Stripe webhooks
│   ├── app/                    # Main app routes
│   │   └── navigator/          # Globe navigator
│   ├── dashboard/              # Dashboard routes
│   │   ├── llm-monitor/        # LLM pipeline health
│   │   └── prometheus/         # PROMETHEUS dashboard
│   ├── layout.tsx              # Root layout
│   └── page.tsx                # Landing page
├── components/                 # React components
│   ├── admin/                  # Admin-specific components
│   ├── ui/                     # Base UI components (Glass* theme)
│   │   ├── GlassCard.tsx       # Glassmorphism card
│   │   ├── GlassButton.tsx     # Styled button variants
│   │   ├── GlassInput.tsx      # Form inputs
│   │   ├── GlassTerm.tsx       # Terminal-style display
│   │   └── ConfidenceDisplay.tsx # K×f confidence viz
│   ├── ElleChat.tsx            # Chat interface for Elle
│   └── FeedbackButton.tsx      # User feedback collection
├── hooks/                      # Custom React hooks
│   └── useWasm.ts              # WASM module loading (optional compute)
├── lib/                        # Utilities and services
│   ├── inference/              # LLM inference client
│   │   └── LFBMClient.ts       # RunPod API client
│   ├── monitoring/             # Pipeline monitoring
│   │   └── LLMPipelineMonitor.ts
│   ├── reasoning/              # CIC reasoning engine
│   └── supabase/               # Supabase client setup
├── contexts/                   # React contexts
├── types/                      # TypeScript types
├── styles/                     # Global styles
├── public/                     # Static assets
│   └── wasm/                   # Optional WASM modules
└── supabase/                   # Supabase config
    └── migrations/             # Database migrations
```

---

## 3. Elle + Guardian Architecture

### 3.1 System Overview

```
User Request → Next.js API Route → Guardian (security) → RunPod vLLM → Elle-72B → Response
                                          ↓
                              Rate limiting, injection detection,
                              output filtering, cost tracking
```

### 3.2 Guardian Security Layer

Guardian protects against adversarial attacks on the LLM:

```typescript
// lib/inference/guardian.ts
import { Ratelimit } from '@upstash/ratelimit';
import { Redis } from '@upstash/redis';

const ratelimit = new Ratelimit({
  redis: Redis.fromEnv(),
  limiter: Ratelimit.slidingWindow(20, '1 m'),
});

const INJECTION_PATTERNS = [
  /ignore (previous|prior|above) instructions/i,
  /disregard (your|the) (instructions|programming)/i,
  /you are (now )?DAN/i,
  /repeat (your |the )?system prompt/i,
  /pretend (you're|to be) (an? )?(unrestricted|uncensored)/i,
];

export async function guardianCheck(
  userId: string,
  input: string
): Promise<{ allowed: boolean; reason?: string }> {
  // Rate limiting
  const { success } = await ratelimit.limit(userId);
  if (!success) {
    return { allowed: false, reason: 'rate_limited' };
  }

  // Prompt injection detection
  for (const pattern of INJECTION_PATTERNS) {
    if (pattern.test(input)) {
      return { allowed: false, reason: 'suspicious_input' };
    }
  }

  return { allowed: true };
}
```

### 3.3 LFBMClient (RunPod Integration)

```typescript
// lib/inference/LFBMClient.ts
export class LFBMClient {
  private endpoint: string;
  private apiKey: string;

  constructor() {
    this.endpoint = process.env.LFBM_ENDPOINT!;
    this.apiKey = process.env.RUNPOD_API_KEY!;
  }

  async chat(messages: Message[], options?: ChatOptions): Promise<string> {
    const response = await fetch(`${this.endpoint}/openai/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'aphoticshaman/elle-72b-ultimate',
        messages,
        max_tokens: options?.maxTokens ?? 2048,
        temperature: options?.temperature ?? 0.7,
        stream: options?.stream ?? false
      })
    });

    if (!response.ok) {
      throw new Error(`RunPod error: ${response.status}`);
    }

    const data = await response.json();
    return data.choices[0].message.content;
  }

  async stream(
    messages: Message[],
    onChunk: (chunk: string) => void
  ): Promise<void> {
    const response = await fetch(`${this.endpoint}/openai/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'aphoticshaman/elle-72b-ultimate',
        messages,
        max_tokens: 2048,
        stream: true
      })
    });

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(line => line.startsWith('data: '));

      for (const line of lines) {
        if (line === 'data: [DONE]') continue;
        try {
          const json = JSON.parse(line.slice(6));
          const content = json.choices[0]?.delta?.content;
          if (content) onChunk(content);
        } catch {}
      }
    }
  }
}
```

---

## 4. Component Architecture

### 4.1 Server vs Client Components

**Server Components** (default in App Router) — fetch data directly, no client hooks:

```tsx
// app/dashboard/page.tsx (Server Component)
import { createClient } from '@/lib/supabase/server';

export default async function DashboardPage() {
  const supabase = await createClient();
  const { data: signals } = await supabase
    .from('signals')
    .select('*')
    .order('updated_at', { ascending: false });

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-white">Dashboard</h1>
      <SignalGrid signals={signals} />
    </div>
  );
}
```

**Client Components** — use `'use client'` directive for interactivity:

```tsx
// components/SignalGrid.tsx
'use client';

import { useState, useEffect } from 'react';
import { createClient } from '@/lib/supabase/client';

export function SignalGrid({ signals: initialSignals }) {
  const [signals, setSignals] = useState(initialSignals);
  const supabase = createClient();

  useEffect(() => {
    const channel = supabase
      .channel('signals')
      .on('postgres_changes', {
        event: '*',
        schema: 'public',
        table: 'signals'
      }, (payload) => {
        setSignals(prev => /* update logic */);
      })
      .subscribe();

    return () => { supabase.removeChannel(channel); };
  }, []);

  return (
    <div className="grid grid-cols-3 gap-4">
      {signals.map(signal => (
        <SignalCard key={signal.id} signal={signal} />
      ))}
    </div>
  );
}
```

### 4.2 Glass Design System

```tsx
// components/ui/GlassCard.tsx
interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  blur?: 'sm' | 'md' | 'lg';
  borderAccent?: 'cyan' | 'red' | 'amber' | 'green';
}

export function GlassCard({
  children,
  className = '',
  blur = 'md',
  borderAccent
}: GlassCardProps) {
  const blurClass = {
    sm: 'backdrop-blur-sm',
    md: 'backdrop-blur-md',
    lg: 'backdrop-blur-lg'
  }[blur];

  const borderClass = borderAccent
    ? `border-l-4 border-${borderAccent}-500`
    : '';

  return (
    <div className={`
      bg-slate-900/70 ${blurClass}
      border border-white/10 rounded-xl
      ${borderClass} ${className}
    `}>
      {children}
    </div>
  );
}
```

---

## 5. API Routes

### 5.1 Elle Chat Endpoint

```typescript
// app/api/elle/chat/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/lib/supabase/server';
import { LFBMClient } from '@/lib/inference/LFBMClient';
import { guardianCheck } from '@/lib/inference/guardian';

export async function POST(request: NextRequest) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();

  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { messages } = await request.json();
  const lastMessage = messages[messages.length - 1].content;

  // Guardian security check
  const guardianResult = await guardianCheck(user.id, lastMessage);
  if (!guardianResult.allowed) {
    return NextResponse.json(
      { error: 'Request blocked', reason: guardianResult.reason },
      { status: 429 }
    );
  }

  // Elle inference via RunPod
  const client = new LFBMClient();
  const systemPrompt = `You are Elle, an advanced AI assistant created by Crystalline Labs. You specialize in mathematics, geopolitics, and strategic analysis. Provide clear, well-reasoned responses with appropriate uncertainty quantification.`;

  const fullMessages = [
    { role: 'system', content: systemPrompt },
    ...messages
  ];

  const response = await client.chat(fullMessages);

  return NextResponse.json({ response });
}
```

### 5.2 Intel Briefing Endpoint

```typescript
// app/api/intel-briefing/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/lib/supabase/server';
import { Redis } from '@upstash/redis';

const redis = Redis.fromEnv();
const CACHE_TTL = 30 * 60; // 30 minutes

export async function POST(request: NextRequest) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();

  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { preset } = await request.json();

  // Check cache
  const cacheKey = `briefing:${preset}`;
  const cached = await redis.get(cacheKey);
  if (cached) {
    return NextResponse.json(cached);
  }

  // Generate briefing via template engine (no LLM cost)
  const briefing = await generateBriefing(preset);

  // Cache result
  await redis.setex(cacheKey, CACHE_TTL, briefing);

  return NextResponse.json(briefing);
}
```

---

## 6. Environment Variables

```bash
# Supabase
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_ROLE_KEY=eyJ...

# Stripe
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_live_...

# RunPod (Elle inference)
RUNPOD_API_KEY=your-runpod-api-key
LFBM_ENDPOINT=https://api.runpod.ai/v2/YOUR_ENDPOINT

# Upstash Redis (rate limiting, caching)
UPSTASH_REDIS_REST_URL=https://xxx.upstash.io
UPSTASH_REDIS_REST_TOKEN=...

# Resend (email)
RESEND_API_KEY=re_...

# Anthropic (fallback LLM)
ANTHROPIC_API_KEY=sk-ant-...

# Internal
CRON_SECRET=random-32-char-string
INTERNAL_SERVICE_SECRET=random-32-char-string
```

---

## 7. RunPod Serverless Configuration

Elle-72B runs on RunPod Serverless with vLLM:

```json
{
  "name": "elle-72b-full-serverless",
  "docker_image": "runpod/worker-vllm:stable-cuda12.1.0",
  "env": {
    "MODEL_NAME": "aphoticshaman/elle-72b-ultimate",
    "HF_TOKEN": "${HF_TOKEN}",
    "DTYPE": "bfloat16",
    "GPU_MEMORY_UTILIZATION": "0.95",
    "MAX_MODEL_LEN": "8192",
    "TENSOR_PARALLEL_SIZE": "2"
  },
  "gpu_config": {
    "recommended": "2x H100 80GB",
    "alternative": "4x A100 80GB"
  },
  "scaling": {
    "min_workers": 0,
    "max_workers": 3,
    "idle_timeout": 300
  }
}
```

### Cost Comparison

| Provider | Model | Cost/1K tokens |
|----------|-------|----------------|
| RunPod (Elle) | elle-72b-ultimate | ~$0.001 |
| Anthropic | Claude Haiku | ~$0.25 |
| OpenAI | GPT-4o-mini | ~$0.15 |

---

## 8. Deployment

### 8.1 Vercel Configuration

```json
// vercel.json
{
  "crons": [
    { "path": "/api/cron/daily-alerts", "schedule": "0 * * * *" },
    { "path": "/api/cron/warm-cache", "schedule": "*/5 * * * *" },
    { "path": "/api/cron/rolling-country-update", "schedule": "*/30 * * * *" }
  ]
}
```

### 8.2 Build Commands

```bash
# Development
npm run dev

# Production build
npm run build

# Type check
npx tsc --noEmit

# Lint
npm run lint
```

---

## 9. Testing

```bash
# Unit tests
npm run test

# E2E tests
npm run test:e2e

# Type checking
npm run type-check
```

---

## Appendix A: Historical Note (Leptos Architecture)

The original architecture spec described a Rust/Leptos/WASM stack. This was a roadmap consideration for:
- Performance-critical real-time signal processing
- Client-side CIC computations
- Edge deployment scenarios

The current Next.js implementation was chosen for faster iteration and broader ecosystem support. WASM integration exists for specific compute modules (see `hooks/useWasm.ts`).

---

*Last updated: December 2024*
*Stack: Next.js 16 / React 19 / TypeScript / Tailwind CSS / RunPod Serverless*
