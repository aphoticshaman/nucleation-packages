import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { createClient } from '@supabase/supabase-js';
import { getLFBMClient, isLFBMEnabled } from '@/lib/inference/LFBMClient';
import { getSecurityGuardian } from '@/lib/reasoning/security';
import { notifyNewTicket } from '@/lib/feedback-notifications';

/**
 * Elle Chat API - Intelligent conversation with ticket routing
 *
 * Elle handles all user interactions:
 * - Questions about the platform
 * - Bug reports
 * - Feature ideas
 * - General feedback
 *
 * Elle + Guardian work together to:
 * 1. Validate input (Guardian)
 * 2. Respond helpfully (Elle)
 * 3. Decide if a ticket should be created
 * 4. Route to support if needed
 */

// Elle's conversational system prompt - teaching + feature gap detection
const ELLE_CHAT_PROMPT = `You are Elle, LatticeForge's AI assistant. TEACH users to use the website, not depend on you.

APPROACH:
- Point to WHERE in the UI (not just answer)
- Phrases: "Click...", "Go to...", "Look in..."
- 1-2 sentences max
- Goal: They won't need to ask again

FEATURE GAP DETECTION:
If user asks for something that SHOULD be a UI feature but isn't:
- Note it as [IDEA] - you're detecting a gap
- Say "That's a great idea - I've noted this for the team"
- Example: "Can you compare two countries?" → If no compare feature exists, that's a feature gap

PLATFORM QUICK REF:
- NSM = stability score (0-100)
- Cmd/Ctrl+K = command palette
- Settings → Alerts
- Click nation → breakdown

STYLE:
- Ultra concise (save tokens)
- Direct: "Go to Settings → Alerts"
- Friendly, never frustrated
- Push them to explore the UI

End with:
[QUESTION] - taught them the UI
[BUG] - confirmed issue
[IDEA] - feature gap detected
[SUPPORT] - needs human
[CHAT] - general`;

// Rate limiting
const rateLimitMap = new Map<string, { count: number; resetAt: number }>();
const RATE_LIMIT = 30; // messages per hour
const RATE_WINDOW = 60 * 60 * 1000;

function checkRateLimit(userId: string): boolean {
  const now = Date.now();
  const entry = rateLimitMap.get(userId);

  if (!entry || now > entry.resetAt) {
    rateLimitMap.set(userId, { count: 1, resetAt: now + RATE_WINDOW });
    return true;
  }

  if (entry.count >= RATE_LIMIT) return false;
  entry.count++;
  return true;
}

// Get authenticated user
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

// Create a support ticket from the conversation
async function createTicketFromChat(
  userId: string | null,
  userEmail: string | undefined,
  classification: string,
  summary: string,
  conversationContext: string,
  pageUrl?: string
): Promise<string | null> {
  try {
    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );

    // Map classification to feedback type
    const typeMap: Record<string, string> = {
      BUG: 'bug',
      IDEA: 'idea',
      SUPPORT: 'question',
      QUESTION: 'question',
      SECURITY: 'bug', // Security issues go as bugs with critical priority
      QUOTA: 'other',  // Quota violations
    };

    // Priority mapping
    const priorityMap: Record<string, string> = {
      SECURITY: 'critical',
      QUOTA: 'high',
      BUG: 'normal',
      IDEA: 'low',
      SUPPORT: 'normal',
      QUESTION: 'low',
    };

    const { data, error } = await supabase
      .from('feedback')
      .insert({
        user_id: userId,
        type: typeMap[classification] || 'other',
        title: summary.slice(0, 200),
        description: conversationContext,
        page_url: pageUrl,
        metadata: {
          source: 'elle_chat',
          classification,
          created_by: 'elle',
          is_security_incident: classification === 'SECURITY',
          is_quota_violation: classification === 'QUOTA',
        },
        status: 'unread',
        priority: priorityMap[classification] || 'low',
      })
      .select('id')
      .single();

    if (error) {
      console.error('Failed to create ticket:', error);
      return null;
    }

    // Send notification (urgent for security/quota issues)
    void notifyNewTicket({
      id: data.id,
      type: typeMap[classification] || 'other',
      title: summary.slice(0, 200),
      description: conversationContext,
      priority: priorityMap[classification] || 'low',
      status: 'unread',
      pageUrl,
      userEmail,
    });

    console.log(`[ELLE] Created ticket ${data.id} (${classification}) from chat`);
    return data.id;
  } catch (e) {
    console.error('Ticket creation error:', e);
    return null;
  }
}

export async function POST(request: Request) {
  try {
    const user = await getAuthUser();
    const userId = user?.id || 'anonymous';

    const body = await request.json();

    // Rate limiting with quota violation tracking
    if (!checkRateLimit(userId)) {
      // Log quota violation and create ticket
      console.warn(`[ELLE/QUOTA] Rate limit exceeded for ${userId}`);

      await createTicketFromChat(
        user?.id || null,
        user?.email,
        'QUOTA',
        `Quota exceeded: ${user?.email || userId}`,
        `User hit rate limit (30 messages/hour)\nUser: ${user?.email || userId}\nTime: ${new Date().toISOString()}`,
        body?.pageUrl
      );

      return NextResponse.json(
        { error: 'Too many messages. Please wait a moment.', quotaExceeded: true },
        { status: 429 }
      );
    }

    const { message, history, pageUrl } = body;

    if (!message || typeof message !== 'string') {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      );
    }

    if (message.length > 2000) {
      return NextResponse.json(
        { error: 'Message too long' },
        { status: 400 }
      );
    }

    // Guardian: Validate input and detect security issues
    const guardian = getSecurityGuardian();
    const validation = guardian.validateInput(message, userId, 'consumer');

    if (!validation.allowed) {
      // Log security incident and create ticket if suspicious
      console.warn(`[ELLE/GUARDIAN] Blocked input from ${userId}: ${validation.reason}`);

      // Auto-create security ticket for suspicious activity
      await createTicketFromChat(
        user?.id || null,
        user?.email,
        'SECURITY',
        `Guardian blocked: ${validation.reason}`,
        `User attempted: "${message.slice(0, 500)}"\nReason: ${validation.reason}\nUser: ${user?.email || userId}`,
        pageUrl
      );

      return NextResponse.json({
        response: "I'm sorry, but I can't process that message. Could you rephrase your question?",
        ticketCreated: true, // Security ticket created
        securityFlag: true,
      });
    }

    // Check if LFBM is enabled
    if (!isLFBMEnabled()) {
      return NextResponse.json({
        response: "Hi! I'm Elle, but I'm in limited mode right now. For urgent help, please email support@latticeforge.ai. I'll be back to full capacity soon!",
        ticketCreated: false,
        fallback: true,
      });
    }

    // Build conversation context for Elle
    const conversationHistory = (history || [])
      .slice(-6) // Last 6 messages for context
      .map((m: { role: string; content: string }) => `${m.role === 'user' ? 'User' : 'Elle'}: ${m.content}`)
      .join('\n');

    const contextPrompt = conversationHistory
      ? `Previous conversation:\n${conversationHistory}\n\nUser: ${message}`
      : message;

    // Call Elle via LFBM
    const lfbm = getLFBMClient();
    const response = await lfbm.generateRaw({
      systemPrompt: ELLE_CHAT_PROMPT,
      userMessage: contextPrompt,
      max_tokens: 512,
      temperature: 0.7,
    });

    // Parse response
    let elleResponse: string;
    let classification = 'CHAT';

    try {
      const parsed = JSON.parse(response);
      if (parsed.blocked) {
        elleResponse = "I'm having a moment - could you try asking that again?";
      } else if (parsed.raw) {
        elleResponse = parsed.raw;
      } else {
        elleResponse = JSON.stringify(parsed);
      }
    } catch {
      elleResponse = response;
    }

    // Extract classification tag
    const classificationMatch = elleResponse.match(/\[(QUESTION|BUG|IDEA|SUPPORT|CHAT)\]/);
    if (classificationMatch) {
      classification = classificationMatch[1];
      // Remove the tag from the response
      elleResponse = elleResponse.replace(/\n?\[(QUESTION|BUG|IDEA|SUPPORT|CHAT)\]/, '').trim();
    }

    // Decide whether to create a ticket
    let ticketCreated = false;
    const shouldCreateTicket = ['BUG', 'IDEA', 'SUPPORT'].includes(classification);

    if (shouldCreateTicket) {
      // Build conversation summary for ticket
      const conversationContext = `User message: ${message}\n\nElle's response: ${elleResponse}`;

      const ticketId = await createTicketFromChat(
        user?.id || null,
        user?.email,
        classification,
        message.slice(0, 200), // Use user's message as title
        conversationContext,
        pageUrl
      );

      ticketCreated = !!ticketId;
    }

    // Log for analytics
    console.log(`[ELLE] Chat: ${classification} | User: ${user?.email || 'anonymous'} | "${message.slice(0, 50)}..."`);

    return NextResponse.json({
      response: elleResponse,
      classification,
      ticketCreated,
    });
  } catch (error) {
    console.error('Elle chat error:', error);
    return NextResponse.json(
      { error: 'Elle is having trouble. Please try again.' },
      { status: 500 }
    );
  }
}
