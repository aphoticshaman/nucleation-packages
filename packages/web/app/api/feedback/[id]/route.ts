import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { createClient } from '@supabase/supabase-js';
import { notifyStatusChange, notifyAssignment } from '@/lib/feedback-notifications';

export const runtime = 'edge';

// Types
interface FeedbackUpdate {
  status?: 'unread' | 'acknowledged' | 'in_progress' | 'resolved' | 'wont_fix' | 'duplicate';
  priority?: 'low' | 'normal' | 'high' | 'critical';
  assigned_to?: string | null;
  admin_notes?: string;
  resolution_notes?: string;
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

// Check user role
async function getUserRole(userId: string): Promise<string | null> {
  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );

  const { data: profile } = await supabase
    .from('profiles')
    .select('role')
    .eq('id', userId)
    .single();

  return profile?.role || null;
}

// GET: Get single feedback item
export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const user = await getAuthUser();
    if (!user) {
      return NextResponse.json({ error: 'Authentication required' }, { status: 401 });
    }

    const { id } = await params;
    const role = await getUserRole(user.id);

    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );

    const { data, error } = await supabase
      .from('feedback')
      .select(`
        *,
        user:profiles!feedback_user_id_fkey(email, full_name, role, tier),
        assignee:profiles!feedback_assigned_to_fkey(email, full_name)
      `)
      .eq('id', id)
      .single();

    if (error || !data) {
      return NextResponse.json({ error: 'Feedback not found' }, { status: 404 });
    }

    // Check access: admins/support can see all, users can only see their own
    if (role !== 'admin' && role !== 'support' && data.user_id !== user.id) {
      return NextResponse.json({ error: 'Access denied' }, { status: 403 });
    }

    return NextResponse.json({ feedback: data });
  } catch (error) {
    console.error('Feedback GET error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch feedback' },
      { status: 500 }
    );
  }
}

// PATCH: Update feedback (admin/support only)
export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const user = await getAuthUser();
    if (!user) {
      return NextResponse.json({ error: 'Authentication required' }, { status: 401 });
    }

    const { id } = await params;
    const role = await getUserRole(user.id);

    // Only admin and support can update
    if (role !== 'admin' && role !== 'support') {
      return NextResponse.json({ error: 'Admin access required' }, { status: 403 });
    }

    const body: FeedbackUpdate = await request.json();

    // Validate status if provided
    const validStatuses = ['unread', 'acknowledged', 'in_progress', 'resolved', 'wont_fix', 'duplicate'];
    if (body.status && !validStatuses.includes(body.status)) {
      return NextResponse.json({ error: 'Invalid status' }, { status: 400 });
    }

    // Validate priority if provided
    const validPriorities = ['low', 'normal', 'high', 'critical'];
    if (body.priority && !validPriorities.includes(body.priority)) {
      return NextResponse.json({ error: 'Invalid priority' }, { status: 400 });
    }

    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );

    // Get current feedback first (for status change notifications)
    const { data: oldFeedback, error: fetchError } = await supabase
      .from('feedback')
      .select('*, user:profiles!feedback_user_id_fkey(email, full_name)')
      .eq('id', id)
      .single();

    if (fetchError || !oldFeedback) {
      return NextResponse.json({ error: 'Feedback not found' }, { status: 404 });
    }

    // Build update object
    const updates: Record<string, unknown> = {};

    if (body.status !== undefined) updates.status = body.status;
    if (body.priority !== undefined) updates.priority = body.priority;
    if (body.assigned_to !== undefined) updates.assigned_to = body.assigned_to;
    if (body.admin_notes !== undefined) updates.admin_notes = body.admin_notes;
    if (body.resolution_notes !== undefined) updates.resolution_notes = body.resolution_notes;

    // Add update metadata
    updates.metadata = {
      last_updated_by: user.id,
      last_updated_at: new Date().toISOString(),
    };

    const { data, error } = await supabase
      .from('feedback')
      .update(updates)
      .eq('id', id)
      .select()
      .single();

    if (error) {
      console.error('Failed to update feedback:', error);
      return NextResponse.json(
        { error: 'Failed to update feedback' },
        { status: 500 }
      );
    }

    console.log(`[FEEDBACK] Updated ${id}: status=${body.status}, priority=${body.priority} by ${user.email}`);

    // Send email notifications (async, don't block response)
    const feedbackData = {
      id,
      type: oldFeedback.type,
      title: oldFeedback.title,
      description: oldFeedback.description,
      priority: data.priority,
      status: data.status,
      pageUrl: oldFeedback.page_url,
      userEmail: oldFeedback.user?.email,
      userName: oldFeedback.user?.full_name,
    };

    // Notify user of status change
    if (body.status && body.status !== oldFeedback.status) {
      void notifyStatusChange(feedbackData, oldFeedback.status);
    }

    // Notify assignee if newly assigned
    if (body.assigned_to && body.assigned_to !== oldFeedback.assigned_to) {
      // Get assignee email
      const { data: assignee } = await supabase
        .from('profiles')
        .select('email')
        .eq('id', body.assigned_to)
        .single();

      if (assignee?.email) {
        void notifyAssignment(feedbackData, assignee.email);
      }
    }

    return NextResponse.json({
      success: true,
      feedback: data,
    });
  } catch (error) {
    console.error('Feedback PATCH error:', error);
    return NextResponse.json(
      { error: 'Failed to update feedback' },
      { status: 500 }
    );
  }
}

// DELETE: Delete feedback (admin only)
export async function DELETE(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const user = await getAuthUser();
    if (!user) {
      return NextResponse.json({ error: 'Authentication required' }, { status: 401 });
    }

    const { id } = await params;
    const role = await getUserRole(user.id);

    // Only admin can delete
    if (role !== 'admin') {
      return NextResponse.json({ error: 'Admin access required' }, { status: 403 });
    }

    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );

    const { error } = await supabase
      .from('feedback')
      .delete()
      .eq('id', id);

    if (error) {
      console.error('Failed to delete feedback:', error);
      return NextResponse.json(
        { error: 'Failed to delete feedback' },
        { status: 500 }
      );
    }

    console.log(`[FEEDBACK] Deleted ${id} by ${user.email}`);

    return NextResponse.json({
      success: true,
      message: 'Feedback deleted',
    });
  } catch (error) {
    console.error('Feedback DELETE error:', error);
    return NextResponse.json(
      { error: 'Failed to delete feedback' },
      { status: 500 }
    );
  }
}
