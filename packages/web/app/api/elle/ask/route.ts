import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { getLFBMClient, isLFBMEnabled } from '@/lib/inference/LFBMClient';

/**
 * Elle - LatticeForge's AI Assistant
 *
 * Handles user questions about the platform, features, and data.
 * Powered by LFBM (Qwen2.5 on RunPod) for cost-effective responses.
 */

// Elle's system prompt - friendly, helpful, knowledgeable
const ELLE_SYSTEM_PROMPT = `You are Elle, LatticeForge's AI assistant. You help users understand and navigate the platform.

About LatticeForge:
- A geopolitical intelligence platform for risk analysis
- Provides real-time briefings, risk scores, and historical pattern analysis
- Features include: nation risk dashboards, signal monitoring, cascade analysis, and custom alerts
- NSM = Nation Stability Metric (0-100, higher = more stable)
- Risk categories: Political, Economic, Security, Military, Cyber

Your personality:
- Friendly and professional
- Concise but thorough
- Use simple language, avoid jargon unless explaining it
- If you don't know something specific about LatticeForge, say so

Respond helpfully and concisely. Keep responses under 300 words unless more detail is needed.`;

// Rate limiting (simple in-memory for now)
const rateLimitMap = new Map<string, { count: number; resetAt: number }>();
const RATE_LIMIT = 20; // requests per hour
const RATE_WINDOW = 60 * 60 * 1000; // 1 hour

function checkRateLimit(userId: string): { allowed: boolean; remaining: number } {
  const now = Date.now();
  const key = userId;
  const entry = rateLimitMap.get(key);

  if (!entry || now > entry.resetAt) {
    rateLimitMap.set(key, { count: 1, resetAt: now + RATE_WINDOW });
    return { allowed: true, remaining: RATE_LIMIT - 1 };
  }

  if (entry.count >= RATE_LIMIT) {
    return { allowed: false, remaining: 0 };
  }

  entry.count++;
  return { allowed: true, remaining: RATE_LIMIT - entry.count };
}

// Get authenticated user (optional - allow anonymous with stricter limits)
async function getAuthUser() {
  try {
    const cookieStore = await cookies();
    const supabase = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          getAll() {
            return cookieStore.getAll();
          },
          setAll() {
            // Read-only
          },
        },
      }
    );

    const { data: { user }, error } = await supabase.auth.getUser();
    if (error || !user) return null;

    return user;
  } catch {
    return null;
  }
}

export async function POST(request: Request) {
  try {
    const user = await getAuthUser();
    const userId = user?.id || 'anonymous';

    // Rate limiting
    const rateCheck = checkRateLimit(userId);
    if (!rateCheck.allowed) {
      return NextResponse.json(
        { error: 'Rate limit exceeded. Please try again later.' },
        { status: 429 }
      );
    }

    const body = await request.json();
    const { question, pageUrl } = body;

    if (!question || typeof question !== 'string') {
      return NextResponse.json(
        { error: 'Question is required' },
        { status: 400 }
      );
    }

    if (question.length > 2000) {
      return NextResponse.json(
        { error: 'Question too long (max 2000 characters)' },
        { status: 400 }
      );
    }

    // Check if LFBM is enabled
    if (!isLFBMEnabled()) {
      // Return a helpful fallback response when LFBM is disabled
      return NextResponse.json({
        response: "Hi! I'm Elle, but I'm currently in limited mode. For help with LatticeForge, please check our documentation or submit a support ticket. I'll be fully available soon!",
        fallback: true,
      });
    }

    // Build context-aware prompt
    const contextHint = pageUrl
      ? `\n\nUser is currently viewing: ${pageUrl}`
      : '';

    const userMessage = `${question}${contextHint}`;

    // Call Elle via LFBM
    const lfbm = getLFBMClient();
    const response = await lfbm.generateRaw({
      systemPrompt: ELLE_SYSTEM_PROMPT,
      userMessage,
      max_tokens: 512,
      temperature: 0.7,
    });

    // Parse the response
    let elleResponse: string;
    try {
      const parsed = JSON.parse(response);
      // If it's a blocked response
      if (parsed.blocked) {
        elleResponse = "Hi! I'm Elle, but I'm currently in limited mode. For help with LatticeForge, please check our documentation or submit a support ticket.";
      } else if (parsed.raw) {
        elleResponse = parsed.raw;
      } else {
        elleResponse = response;
      }
    } catch {
      // Response is plain text
      elleResponse = response;
    }

    // Log for analytics (async, don't block)
    console.log(`[ELLE] Question from ${user?.email || 'anonymous'}: "${question.slice(0, 100)}..."`);

    return NextResponse.json({
      response: elleResponse,
      remaining: rateCheck.remaining,
    });
  } catch (error) {
    console.error('Elle API error:', error);
    return NextResponse.json(
      { error: 'Elle is temporarily unavailable. Please try again.' },
      { status: 500 }
    );
  }
}
