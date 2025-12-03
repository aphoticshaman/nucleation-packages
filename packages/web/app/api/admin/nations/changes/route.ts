import { NextResponse } from 'next/server';
import { createClient, createAdminClient } from '@/lib/supabase/server';

// GET: List nation changes history
export async function GET(request: Request) {
  const supabase = await createClient();
  const { searchParams } = new URL(request.url);

  // Verify admin
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { data: profile } = await supabase
    .from('profiles')
    .select('role')
    .eq('id', user.id)
    .single();

  if (profile?.role !== 'admin') {
    return NextResponse.json({ error: 'Admin access required' }, { status: 403 });
  }

  const nationCode = searchParams.get('nation');
  const changeType = searchParams.get('type');
  const limit = parseInt(searchParams.get('limit') || '50');

  let query = supabase
    .from('nation_changes')
    .select('*')
    .order('effective_date', { ascending: false })
    .limit(limit);

  if (nationCode) {
    query = query.or(`source_codes.cs.{${nationCode}},result_codes.cs.{${nationCode}}`);
  }

  if (changeType) {
    query = query.eq('change_type', changeType);
  }

  const { data, error } = await query;

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ changes: data });
}

// POST: Process a nation change
export async function POST(request: Request) {
  const supabase = await createClient();

  // Verify admin
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { data: profile } = await supabase
    .from('profiles')
    .select('role')
    .eq('id', user.id)
    .single();

  if (profile?.role !== 'admin') {
    return NextResponse.json({ error: 'Admin access required' }, { status: 403 });
  }

  const body = await request.json();
  const { changeType, ...params } = body;

  const serviceClient = createAdminClient();

  try {
    let result;

    switch (changeType) {
      case 'rename': {
        const { oldCode, newCode, newName, effectiveDate, description } = params;
        const { data, error } = await serviceClient.rpc('process_nation_rename', {
          p_old_code: oldCode,
          p_new_code: newCode,
          p_new_name: newName,
          p_effective_date: effectiveDate || new Date().toISOString().split('T')[0],
          p_description: description,
        });
        if (error) throw error;
        result = { changeId: data, type: 'rename' };
        break;
      }

      case 'split': {
        const { sourceCode, resultNations, effectiveDate, description, keepSource } = params;
        const { data, error } = await serviceClient.rpc('process_nation_split', {
          p_source_code: sourceCode,
          p_result_nations: resultNations,
          p_effective_date: effectiveDate || new Date().toISOString().split('T')[0],
          p_description: description,
          p_keep_source: keepSource || false,
        });
        if (error) throw error;
        result = { changeId: data, type: 'split' };
        break;
      }

      case 'merge': {
        const { sourceCodes, resultCode, resultName, resultRegion, effectiveDate, description } = params;
        const { data, error } = await serviceClient.rpc('process_nation_merge', {
          p_source_codes: sourceCodes,
          p_result_code: resultCode,
          p_result_name: resultName,
          p_result_region: resultRegion,
          p_effective_date: effectiveDate || new Date().toISOString().split('T')[0],
          p_description: description,
        });
        if (error) throw error;
        result = { changeId: data, type: 'merge' };
        break;
      }

      case 'takeover': {
        const { takenCode, takerCode, territoryName, effectiveDate, description, disputed } = params;
        const { data, error } = await serviceClient.rpc('process_nation_takeover', {
          p_taken_code: takenCode,
          p_taker_code: takerCode,
          p_territory_name: territoryName,
          p_effective_date: effectiveDate || new Date().toISOString().split('T')[0],
          p_description: description,
          p_disputed: disputed !== false,
        });
        if (error) throw error;
        result = { changeId: data, type: 'takeover' };
        break;
      }

      default:
        return NextResponse.json(
          { error: 'Invalid change type. Must be: rename, split, merge, or takeover' },
          { status: 400 }
        );
    }

    return NextResponse.json({
      success: true,
      ...result,
      message: `Nation ${changeType} processed successfully`,
    });
  } catch (error) {
    console.error('Nation change error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to process nation change' },
      { status: 500 }
    );
  }
}
