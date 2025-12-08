import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { createClient } from '@supabase/supabase-js';
import { notifyNewTicket } from '@/lib/feedback-notifications';

export const runtime = 'edge';

// Types
interface FeedbackSubmission {
  type: 'bug' | 'idea' | 'question' | 'other';
  title: string;
  description: string;
  pageUrl?: string;
  userAgent?: string;
  screenshotUrl?: string;
  metadata?: Record<string, unknown>;
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

// Check if user is admin or support
async function isAdminOrSupport(userId: string): Promise<boolean> {
  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );

  const { data: profile } = await supabase
    .from('profiles')
    .select('role')
    .eq('id', userId)
    .single();

  return profile?.role === 'admin' || profile?.role === 'support';
}

// POST: Submit new feedback
export async function POST(request: Request) {
  try {
    const user = await getAuthUser();

    // Allow anonymous feedback but prefer authenticated
    const userId = user?.id || null;

    const body: FeedbackSubmission = await request.json();

    // Validate required fields
    if (!body.type || !body.title || !body.description) {
      return NextResponse.json(
        { error: 'Missing required fields: type, title, description' },
        { status: 400 }
      );
    }

    // Validate type
    if (!['bug', 'idea', 'question', 'other'].includes(body.type)) {
      return NextResponse.json(
        { error: 'Invalid feedback type' },
        { status: 400 }
      );
    }

    // Validate lengths
    if (body.title.length > 200) {
      return NextResponse.json(
        { error: 'Title must be under 200 characters' },
        { status: 400 }
      );
    }

    if (body.description.length > 5000) {
      return NextResponse.json(
        { error: 'Description must be under 5000 characters' },
        { status: 400 }
      );
    }

    // Use service role to insert (bypasses RLS for anonymous users)
    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );

    // Build metadata with user context
    const metadata = {
      ...body.metadata,
      submitted_at: new Date().toISOString(),
      user_email: user?.email,
      user_tier: user?.user_metadata?.tier,
    };

    const { data, error } = await supabase
      .from('feedback')
      .insert({
        user_id: userId,
        type: body.type,
        title: body.title.trim(),
        description: body.description.trim(),
        page_url: body.pageUrl,
        user_agent: body.userAgent,
        screenshot_url: body.screenshotUrl,
        metadata,
        status: 'unread',
        priority: body.type === 'bug' ? 'normal' : 'low',
      })
      .select('id, created_at')
      .single();

    if (error) {
      console.error('Failed to insert feedback:', error);
      return NextResponse.json(
        { error: 'Failed to submit feedback' },
        { status: 500 }
      );
    }

    // Log for monitoring
    console.log(`[FEEDBACK] New ${body.type} submitted: ${data.id} by ${user?.email || 'anonymous'}`);

    // Send email notification to admins (async, don't block response)
    void notifyNewTicket({
      id: data.id,
      type: body.type,
      title: body.title.trim(),
      description: body.description.trim(),
      priority: body.type === 'bug' ? 'normal' : 'low',
      status: 'unread',
      pageUrl: body.pageUrl,
      userEmail: user?.email,
      userName: user?.user_metadata?.full_name,
    });

    return NextResponse.json({
      success: true,
      feedbackId: data.id,
      message: 'Thank you for your feedback!',
    });
  } catch (error) {
    console.error('Feedback POST error:', error);
    return NextResponse.json(
      { error: 'Failed to process feedback' },
      { status: 500 }
    );
  }
}

// GET: List feedback (admin/support only)
export async function GET(request: Request) {
  try {
    const user = await getAuthUser();

    if (!user) {
      return NextResponse.json({ error: 'Authentication required' }, { status: 401 });
    }

    // Verify admin or support role
    const hasAccess = await isAdminOrSupport(user.id);
    if (!hasAccess) {
      return NextResponse.json({ error: 'Admin access required' }, { status: 403 });
    }

    const { searchParams } = new URL(request.url);

    // Parse filters
    const status = searchParams.get('status'); // unread, acknowledged, etc.
    const type = searchParams.get('type'); // bug, idea, etc.
    const priority = searchParams.get('priority'); // low, normal, high, critical
    const limit = parseInt(searchParams.get('limit') || '50');
    const offset = parseInt(searchParams.get('offset') || '0');
    const sortBy = searchParams.get('sortBy') || 'created_at';
    const sortOrder = searchParams.get('sortOrder') || 'desc';

    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );

    // Build query
    let query = supabase
      .from('feedback')
      .select(`
        *,
        user:profiles!feedback_user_id_fkey(email, full_name, role, tier),
        assignee:profiles!feedback_assigned_to_fkey(email, full_name)
      `, { count: 'exact' });

    // Apply filters
    if (status) {
      query = query.eq('status', status);
    }
    if (type) {
      query = query.eq('type', type);
    }
    if (priority) {
      query = query.eq('priority', priority);
    }

    // Apply sorting
    const ascending = sortOrder === 'asc';
    query = query.order(sortBy, { ascending });

    // Apply pagination
    query = query.range(offset, offset + limit - 1);

    const { data, error, count } = await query;

    if (error) {
      console.error('Failed to fetch feedback:', error);
      return NextResponse.json(
        { error: 'Failed to fetch feedback' },
        { status: 500 }
      );
    }

    // Also get stats
    const { data: stats } = await supabase
      .from('feedback_stats')
      .select('*')
      .single();

    return NextResponse.json({
      feedback: data,
      total: count,
      stats,
      pagination: {
        limit,
        offset,
        hasMore: (count || 0) > offset + limit,
      },
    });
  } catch (error) {
    console.error('Feedback GET error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch feedback' },
      { status: 500 }
    );
  }
}
