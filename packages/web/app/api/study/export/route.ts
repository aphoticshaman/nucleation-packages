/**
 * Study Book Training Export API
 *
 * Exports training examples in Axolotl JSONL format for fine-tuning Elle.
 * Also supports ShareGPT format as an alternative.
 *
 * ADMIN ONLY - requires admin role
 */

import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { getLearningEngine } from '@/lib/study/learning';

export const runtime = 'edge';

// =============================================================================
// AUTH CHECK
// =============================================================================

async function checkAdminAuth(): Promise<{ userId: string; email: string } | null> {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseKey || !anonKey) {
    console.error('[Export] Missing Supabase config');
    return null;
  }

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
    return null;
  }

  const serviceClient = createClient(supabaseUrl, supabaseKey);
  const { data: profile } = await serviceClient
    .from('profiles')
    .select('role')
    .eq('id', user.id)
    .single();

  if (!profile || profile.role !== 'admin') {
    return null;
  }

  return { userId: user.id, email: user.email || '' };
}

// =============================================================================
// GET - Export training data
// =============================================================================

export async function GET(request: NextRequest) {
  try {
    const auth = await checkAdminAuth();
    if (!auth) {
      return NextResponse.json({ error: 'Admin access required' }, { status: 403 });
    }

    const { searchParams } = new URL(request.url);
    const format = searchParams.get('format') || 'axolotl'; // axolotl | sharegpt
    const status = searchParams.get('status') as 'approved' | 'pending' | null;
    const minQuality = parseFloat(searchParams.get('minQuality') || '0');
    const domains = searchParams.get('domains')?.split(',').filter(Boolean);
    const limit = parseInt(searchParams.get('limit') || '1000');

    const learningEngine = getLearningEngine();

    let data: string;
    let contentType: string;
    let filename: string;

    if (format === 'sharegpt') {
      data = await learningEngine.exportShareGPT({
        minQuality: minQuality || undefined,
        limit,
      });
      contentType = 'application/json';
      filename = `elle-training-sharegpt-${Date.now()}.json`;
    } else {
      data = await learningEngine.exportForAxolotl({
        status: status || undefined,
        minQuality: minQuality || undefined,
        domains,
        limit,
      });
      contentType = 'application/jsonl';
      filename = `elle-training-axolotl-${Date.now()}.jsonl`;
    }

    const lines = data.split('\n').filter(Boolean).length;
    console.log(`[Export] ${auth.email} exported ${lines} training examples in ${format} format`);

    return new NextResponse(data, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Content-Disposition': `attachment; filename="${filename}"`,
        'X-Example-Count': lines.toString(),
      },
    });
  } catch (error) {
    console.error('[Export] Error:', error);
    return NextResponse.json(
      { error: 'Failed to export training data', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}

// =============================================================================
// POST - Batch extract training examples from rated interactions
// =============================================================================

interface ExtractRequest {
  minRating?: number;
  limit?: number;
}

export async function POST(request: NextRequest) {
  try {
    const auth = await checkAdminAuth();
    if (!auth) {
      return NextResponse.json({ error: 'Admin access required' }, { status: 403 });
    }

    const body = await request.json() as ExtractRequest;
    const minRating = body.minRating || 4;
    const limit = body.limit || 100;

    const learningEngine = getLearningEngine();
    const created = await learningEngine.extractTrainingExamples(minRating, limit);

    console.log(`[Export] ${auth.email} extracted ${created} new training examples`);

    return NextResponse.json({
      success: true,
      extracted: created,
      message: `Created ${created} new training examples from highly-rated interactions`,
    });
  } catch (error) {
    console.error('[Export] Extract error:', error);
    return NextResponse.json(
      { error: 'Failed to extract training examples' },
      { status: 500 }
    );
  }
}
