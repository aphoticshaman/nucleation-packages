# LatticeForge UI: Junior Frontend Developer Guide

**Stack:** Next.js 16 / React 19 / TypeScript / Tailwind CSS
**Audience:** Junior/Mid-level Frontend Developers, new team members
**Prerequisites:** HTML, CSS, JavaScript/TypeScript basics, React fundamentals
**Time to Onboard:** ~1 week to first meaningful PR

---

## Welcome

You're joining the frontend team on LatticeForge - a real-time intelligence dashboard that processes streaming signal data and orchestrates LLM inference via our Elle-72B-Ultimate model on RunPod.

The stack is familiar: Next.js, React, TypeScript, Tailwind. If you've done React before, you'll be productive quickly.

By the end of week one, you'll have:
- Set up your dev environment
- Made a small component change
- Understood the basic data flow
- Submitted your first PR

Let's go.

---

## Part 1: Environment Setup (Day 1)

### 1.1 Install the Tools

```bash
# Node.js 20+ (use nvm)
nvm install 20
nvm use 20

# Clone the repo
git clone <repo-url>
cd nucleation-packages/packages/web

# Install dependencies
npm install

# Copy environment template
cp .env.example .env.local
```

### 1.2 First Build

```bash
# Start development server
npm run dev

# Open http://localhost:3000
```

If you see the landing page, you're good. If you see errors, check:
- Node version: `node --version` (should be 20+)
- Dependencies installed: Check for `node_modules/`
- Environment variables: `.env.local` exists

### 1.3 Editor Setup

We use VS Code with:
- `ESLint` extension
- `Prettier` extension
- `Tailwind CSS IntelliSense` extension

Your `settings.json` should include:
```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "tailwindCSS.includeLanguages": {
    "typescript": "javascript",
    "typescriptreact": "javascript"
  }
}
```

---

## Part 2: Project Structure (Day 1-2)

### 2.1 Where Things Live

```
packages/web/
├── app/                    # Next.js App Router (routes)
│   ├── (auth)/             # Auth routes (login, signup)
│   ├── admin/              # Admin panel
│   ├── api/                # API routes (backend logic)
│   │   └── elle/           # Elle-72B-Ultimate inference
│   ├── dashboard/          # Dashboard pages
│   ├── layout.tsx          # Root layout
│   └── page.tsx            # Landing page
├── components/             # Reusable React components
│   ├── ui/                 # Base UI components (Glass* theme)
│   └── ElleChat.tsx        # Elle chat interface
├── lib/                    # Utilities and services
│   ├── inference/          # RunPod/Elle client
│   │   └── LFBMClient.ts   # Elle-72B-Ultimate API
│   ├── supabase/           # Database client
│   └── ...
├── hooks/                  # Custom React hooks
├── types/                  # TypeScript types
└── styles/                 # Global CSS
```

### 2.2 App Router Basics

Next.js 16 uses the App Router. Key concepts:

**File-based routing:**
- `app/page.tsx` → `/`
- `app/dashboard/page.tsx` → `/dashboard`
- `app/admin/insights/page.tsx` → `/admin/insights`

**Server vs Client Components:**
- Files are **Server Components** by default (can fetch data, no useState/useEffect)
- Add `'use client'` at the top for **Client Components** (interactivity)

```tsx
// Server Component (default)
export default async function Page() {
  const data = await fetchData(); // Can fetch directly!
  return <div>{data}</div>;
}

// Client Component
'use client';
import { useState } from 'react';

export default function Counter() {
  const [count, setCount] = useState(0);
  return <button onClick={() => setCount(c => c + 1)}>{count}</button>;
}
```

---

## Part 3: Component Patterns (Day 2-3)

### 3.1 Glass Design System

We use a glassmorphism aesthetic. The base components are in `components/ui/`:

```tsx
// Using GlassCard
import { GlassCard } from '@/components/ui/GlassCard';

export function MyComponent() {
  return (
    <GlassCard className="p-4">
      <h2 className="text-white font-semibold">Title</h2>
      <p className="text-slate-400">Content goes here</p>
    </GlassCard>
  );
}
```

### 3.2 Tailwind Basics

We use Tailwind CSS for styling. Common patterns:

```tsx
// Layout
<div className="flex items-center justify-between gap-4">

// Typography
<h1 className="text-2xl font-bold text-white">
<p className="text-sm text-slate-400">

// Spacing
<div className="p-4 m-2 space-y-4">

// Colors (our theme)
text-white        // Primary text
text-slate-400    // Secondary text
text-cyan-400     // Accent
bg-slate-900      // Background
border-white/10   // Subtle borders
```

### 3.3 Creating a Component

```tsx
// components/MyCard.tsx
interface MyCardProps {
  title: string;
  description: string;
  onAction?: () => void;
}

export function MyCard({ title, description, onAction }: MyCardProps) {
  return (
    <div className="bg-slate-900/70 backdrop-blur-md border border-white/10 rounded-xl p-4">
      <h3 className="text-lg font-semibold text-white">{title}</h3>
      <p className="text-slate-400 mt-2">{description}</p>
      {onAction && (
        <button
          onClick={onAction}
          className="mt-4 px-4 py-2 bg-cyan-500 hover:bg-cyan-600 rounded-lg text-white transition-colors"
        >
          Take Action
        </button>
      )}
    </div>
  );
}
```

---

## Part 4: Elle-72B-Ultimate Integration (Day 3-4)

### 4.1 What is Elle?

Elle-72B-Ultimate is our fine-tuned Qwen2.5-72B model running on RunPod Serverless. It powers:
- Intel briefings and analysis
- Training data generation
- Chat interactions
- Executive summaries

Model: `aphoticshaman/elle-72b-ultimate` on HuggingFace

### 4.2 Using Elle in Components

```tsx
// components/ElleChat.tsx
'use client';
import { useState } from 'react';

export function ElleChat() {
  const [input, setInput] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  async function askElle() {
    setLoading(true);
    try {
      const res = await fetch('/api/elle/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: input })
      });
      const data = await res.json();
      setResponse(data.response);
    } catch (err) {
      setResponse('Error communicating with Elle');
    }
    setLoading(false);
  }

  return (
    <div className="space-y-4">
      <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        className="w-full p-4 bg-slate-800 border border-white/10 rounded-lg text-white"
        placeholder="Ask Elle..."
      />
      <button
        onClick={askElle}
        disabled={loading}
        className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 rounded-lg text-white disabled:opacity-50"
      >
        {loading ? 'Thinking...' : 'Ask Elle'}
      </button>
      {response && (
        <div className="p-4 bg-slate-900/70 rounded-lg text-slate-300">
          {response}
        </div>
      )}
    </div>
  );
}
```

### 4.3 API Route for Elle

```typescript
// app/api/elle/ask/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { LFBMClient } from '@/lib/inference/LFBMClient';

export async function POST(request: NextRequest) {
  const { question } = await request.json();

  const client = new LFBMClient();
  const response = await client.chat([
    { role: 'system', content: 'You are Elle, an advanced AI assistant.' },
    { role: 'user', content: question }
  ]);

  return NextResponse.json({ response });
}
```

---

## Part 5: Data Fetching

### 5.1 Server-Side Fetching

For data that doesn't need real-time updates:

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

### 5.2 Client-Side Fetching with Real-Time

```tsx
// components/LiveData.tsx
'use client';
import { useState, useEffect } from 'react';
import { createClient } from '@/lib/supabase/client';

export function LiveData() {
  const [data, setData] = useState([]);
  const supabase = createClient();

  useEffect(() => {
    // Initial fetch
    supabase.from('signals').select('*').then(({ data }) => setData(data));

    // Real-time subscription
    const channel = supabase
      .channel('signals')
      .on('postgres_changes', {
        event: '*',
        schema: 'public',
        table: 'signals'
      }, (payload) => {
        console.log('Change:', payload);
      })
      .subscribe();

    return () => { supabase.removeChannel(channel); };
  }, []);

  return <div>{/* render data */}</div>;
}
```

---

## Part 6: Your First Task (Day 4-5)

### 6.1 Warm-Up: Change a Button Color

1. Find `components/ui/GlassButton.tsx`
2. Find the "primary" variant
3. Change the background from cyan to blue
4. Check the dev server for changes
5. Commit: `git commit -m "change primary button to blue"`

### 6.2 Small Feature: Add a Tooltip

Create a tooltip component:

```tsx
// components/ui/Tooltip.tsx
'use client';
import { useState } from 'react';

interface TooltipProps {
  children: React.ReactNode;
  text: string;
}

export function Tooltip({ children, text }: TooltipProps) {
  const [visible, setVisible] = useState(false);

  return (
    <div
      className="relative inline-block"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      {children}
      {visible && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-slate-800 text-white text-xs rounded whitespace-nowrap z-50">
          {text}
        </div>
      )}
    </div>
  );
}
```

### 6.3 PR Checklist

```bash
# Lint
npm run lint

# Type check
npx tsc --noEmit

# Build (catches more issues)
npm run build
```

---

## Part 7: Understanding the Domain

### 7.1 What LatticeForge Does

LatticeForge monitors data streams and detects:
- **Phase transitions** — when systems shift states
- **Causal relationships** — which signals affect others
- **Regime changes** — when the "rules" change

Think of it as an early warning system for geopolitical and market events.

### 7.2 Key Concepts

**Signal:** A time series (stock prices, sentiment, etc.). Displayed as sparklines.

**Elle-72B-Ultimate:** Our custom LLM on RunPod. Powers analysis and briefings.

**Guardian:** Security layer protecting Elle from prompt injection attacks.

**Intel Briefing:** AI-generated intelligence reports with executive summaries.

---

## Part 8: Common Patterns

### 8.1 Loading States

```tsx
if (loading) return <div className="animate-pulse">Loading...</div>;
if (error) return <div className="text-red-400">Error: {error.message}</div>;
return <div>{/* render data */}</div>;
```

### 8.2 Conditional Rendering

```tsx
{condition && <Component />}
{condition ? <ComponentA /> : <ComponentB />}
```

### 8.3 List Rendering

```tsx
{items.map(item => (
  <Card key={item.id} data={item} />
))}
```

---

## Part 9: Quick Reference

### Commands

```bash
npm run dev          # Start dev server
npm run build        # Production build
npm run lint         # Run linter
npx tsc --noEmit     # Type check
```

### Tailwind Cheat Sheet

```css
flex items-center justify-between gap-4
grid grid-cols-3 gap-4
p-4 m-2 space-y-4
text-lg font-bold text-white
bg-slate-900 text-cyan-400 border-white/10
transition-colors hover:bg-cyan-600
```

---

## Milestones

### Week 1
- [ ] Environment running
- [ ] Made a trivial change
- [ ] Built a simple component
- [ ] Submitted first PR
- [ ] Understood Server vs Client components

### Week 2
- [ ] Built component using real data
- [ ] Handled loading/error states
- [ ] Understood Elle integration
- [ ] Can explain folder structure

---

*Last updated: December 2024*
*Stack: Next.js 16 / React 19 / TypeScript / Tailwind CSS*
*Model: aphoticshaman/elle-72b-ultimate on RunPod*
