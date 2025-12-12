/**
 * Study Book Chat API
 *
 * Streaming chat endpoint for the admin Study Book.
 * Features:
 * - Real-time streaming responses
 * - Three research depths (instant/moderate/thorough)
 * - Tool execution
 * - GitHub integration
 * - Full conversation memory
 *
 * ADMIN ONLY - requires admin role
 */

import { NextRequest } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { getStudyClient, type StudyRequest, type StudyOptions } from '@/lib/study/client';
import { type ResearchDepth } from '@/lib/study/tools';
import { type StudyMode } from '@/lib/study/memory';

export const runtime = 'edge';
export const maxDuration = 300; // 5 minutes for thorough research

// =============================================================================
// REQUEST VALIDATION
// =============================================================================

interface ChatRequestBody {
  message: string;
  conversationId?: string;
  mode?: StudyMode;
  depth?: ResearchDepth;
  unrestricted?: boolean;
  bigBrain?: boolean;
  useTools?: boolean;
  githubRepo?: string;
  githubBranch?: string;
  systemPrompt?: string;
}

function validateRequest(body: unknown): ChatRequestBody {
  if (!body || typeof body !== 'object') {
    throw new Error('Invalid request body');
  }

  const b = body as Record<string, unknown>;

  if (!b.message || typeof b.message !== 'string' || b.message.trim().length === 0) {
    throw new Error('Message is required');
  }

  return {
    message: b.message.trim(),
    conversationId: typeof b.conversationId === 'string' ? b.conversationId : undefined,
    mode: ['chat', 'code', 'research', 'brief', 'analyze'].includes(b.mode as string)
      ? (b.mode as StudyMode)
      : 'chat',
    depth: ['instant', 'moderate', 'thorough'].includes(b.depth as string)
      ? (b.depth as ResearchDepth)
      : 'moderate',
    unrestricted: b.unrestricted === true,
    bigBrain: b.bigBrain === true,
    useTools: b.useTools !== false, // Default true
    githubRepo: typeof b.githubRepo === 'string' ? b.githubRepo : undefined,
    githubBranch: typeof b.githubBranch === 'string' ? b.githubBranch : undefined,
    systemPrompt: typeof b.systemPrompt === 'string' ? b.systemPrompt : undefined,
  };
}

// =============================================================================
// AUTH CHECK
// =============================================================================

async function checkAdminAuth(): Promise<{ userId: string; email: string } | null> {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseKey || !anonKey) {
    console.error('[Study] Missing Supabase config');
    return null;
  }

  // Get user from cookies
  const cookieStore = await cookies();
  const authClient = createServerClient(supabaseUrl, anonKey, {
    cookies: {
      getAll() {
        return cookieStore.getAll();
      },
      setAll() {},
    },
  });

  const { data: { user }, error: authError } = await authClient.auth.getUser();

  if (authError || !user) {
    console.log('[Study] Auth failed:', authError?.message || 'No user');
    return null;
  }

  // Check admin role with service client
  const serviceClient = createClient(supabaseUrl, supabaseKey);
  const { data: profile } = await serviceClient
    .from('profiles')
    .select('role')
    .eq('id', user.id)
    .single();

  if (!profile || profile.role !== 'admin') {
    console.log('[Study] Non-admin access attempt:', user.email);
    return null;
  }

  return { userId: user.id, email: user.email || '' };
}

// =============================================================================
// STREAMING RESPONSE
// =============================================================================

export async function POST(request: NextRequest) {
  const startTime = Date.now();

  try {
    // Auth check
    const auth = await checkAdminAuth();
    if (!auth) {
      return new Response(
        JSON.stringify({ error: 'Admin access required' }),
        { status: 403, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Parse and validate request
    let body: ChatRequestBody;
    try {
      const rawBody = await request.json();
      body = validateRequest(rawBody);
    } catch (e) {
      return new Response(
        JSON.stringify({ error: e instanceof Error ? e.message : 'Invalid request' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    console.log(`[Study] ${auth.email} - ${body.mode} mode, ${body.depth} depth`);

    // Build study request
    const studyRequest: StudyRequest = {
      message: body.message,
      conversationId: body.conversationId,
      options: {
        mode: body.mode!,
        depth: body.depth!,
        unrestricted: body.unrestricted!,
        bigBrain: body.bigBrain!,
        useTools: body.useTools!,
        githubContext: body.githubRepo
          ? {
              repo: body.githubRepo,
              branch: body.githubBranch,
            }
          : undefined,
      },
      systemPromptOverride: body.systemPrompt,
    };

    // Get study client and process
    const client = getStudyClient(auth.userId);
    const response = await client.chat(studyRequest);

    // Return response
    // For now, non-streaming - can upgrade to SSE later
    return new Response(
      JSON.stringify({
        success: true,
        content: response.content,
        thinking: response.thinking,
        conversationId: response.conversationId,
        messageId: response.messageId,
        model: response.model,
        tier: response.tier,
        latency_ms: response.latency_ms,
        research: response.research
          ? {
              depth: response.research.depth,
              sourcesFound: response.research.webResults.length,
              gdeltSignals: response.research.gdeltSignals.length,
              pagesAnalyzed: response.research.fetchedPages.length,
            }
          : undefined,
        totalLatency_ms: Date.now() - startTime,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  } catch (error) {
    console.error('[Study] Error:', error);
    return new Response(
      JSON.stringify({
        error: 'Internal server error',
        details: error instanceof Error ? error.message : 'Unknown error',
      }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}

// =============================================================================
// GET - List conversations
// =============================================================================

export async function GET(request: NextRequest) {
  try {
    const auth = await checkAdminAuth();
    if (!auth) {
      return new Response(
        JSON.stringify({ error: 'Admin access required' }),
        { status: 403, headers: { 'Content-Type': 'application/json' } }
      );
    }

    const { searchParams } = new URL(request.url);
    const mode = searchParams.get('mode') as StudyMode | null;
    const limit = parseInt(searchParams.get('limit') || '20');

    const client = getStudyClient(auth.userId);
    const conversations = await client['memory'].listConversations({
      mode: mode || undefined,
      limit,
    });

    return new Response(
      JSON.stringify({
        success: true,
        conversations,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  } catch (error) {
    console.error('[Study] List error:', error);
    return new Response(
      JSON.stringify({ error: 'Failed to list conversations' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
