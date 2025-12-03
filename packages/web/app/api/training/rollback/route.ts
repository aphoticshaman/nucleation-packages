import { NextResponse } from 'next/server';
import { createClient, createAdminClient } from '@/lib/supabase/server';

// POST: Rollback to a specific backup
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
  const { backupId, action } = body;

  const serviceClient = createAdminClient();

  try {
    switch (action) {
      case 'rollback': {
        if (!backupId) {
          return NextResponse.json({ error: 'backupId required' }, { status: 400 });
        }

        // Execute rollback
        const { data, error } = await serviceClient.rpc('rollback_training_data', {
          p_backup_id: backupId,
          p_user_id: user.id,
        });

        if (error) throw error;

        return NextResponse.json({
          success: true,
          action: 'rollback',
          backupId,
          result: data?.[0] || { quarantined_count: 0 },
          message: `Rolled back to ${backupId}. Data created after backup moved to quarantine.`,
        });
      }

      case 'detect_anomalies': {
        const { data, error } = await serviceClient.rpc('detect_training_anomalies');

        if (error) throw error;

        return NextResponse.json({
          success: true,
          action: 'detect_anomalies',
          anomalies: data || [],
          count: data?.length || 0,
          message: `Found ${data?.length || 0} potential anomalies`,
        });
      }

      case 'quarantine': {
        const { exampleId, reason, type } = body;

        if (!exampleId || !reason) {
          return NextResponse.json({ error: 'exampleId and reason required' }, { status: 400 });
        }

        const { data, error } = await serviceClient.rpc('quarantine_training_example', {
          p_example_id: exampleId,
          p_reason: reason,
          p_quarantine_type: type || 'manual',
          p_user_id: user.id,
        });

        if (error) throw error;

        return NextResponse.json({
          success: true,
          action: 'quarantine',
          quarantineId: data,
          message: `Example ${exampleId} moved to quarantine`,
        });
      }

      case 'restore': {
        const { quarantineId, notes } = body;

        if (!quarantineId) {
          return NextResponse.json({ error: 'quarantineId required' }, { status: 400 });
        }

        const { data, error } = await serviceClient.rpc('restore_from_quarantine', {
          p_quarantine_id: quarantineId,
          p_user_id: user.id,
          p_notes: notes,
        });

        if (error) throw error;

        return NextResponse.json({
          success: true,
          action: 'restore',
          exampleId: data,
          message: `Example restored from quarantine`,
        });
      }

      case 'auto_quarantine_anomalies': {
        // First detect anomalies
        const { data: anomalies, error: detectError } = await serviceClient.rpc('detect_training_anomalies');

        if (detectError) throw detectError;

        if (!anomalies || anomalies.length === 0) {
          return NextResponse.json({
            success: true,
            action: 'auto_quarantine_anomalies',
            quarantined: 0,
            message: 'No anomalies found',
          });
        }

        // Quarantine each anomaly
        let quarantined = 0;
        const errors: string[] = [];

        for (const anomaly of anomalies) {
          try {
            await serviceClient.rpc('quarantine_training_example', {
              p_example_id: anomaly.example_id,
              p_reason: `Auto-detected: ${anomaly.anomaly_type} (score: ${anomaly.anomaly_score})`,
              p_quarantine_type: 'auto_anomaly',
              p_user_id: user.id,
            });
            quarantined++;
          } catch (e) {
            errors.push(`Failed to quarantine ${anomaly.example_id}: ${e}`);
          }
        }

        return NextResponse.json({
          success: true,
          action: 'auto_quarantine_anomalies',
          detected: anomalies.length,
          quarantined,
          errors: errors.length > 0 ? errors : undefined,
          message: `Quarantined ${quarantined} of ${anomalies.length} anomalies`,
        });
      }

      default:
        return NextResponse.json(
          { error: 'Invalid action. Use: rollback, detect_anomalies, quarantine, restore, auto_quarantine_anomalies' },
          { status: 400 }
        );
    }
  } catch (error) {
    console.error('Rollback/quarantine error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Operation failed' },
      { status: 500 }
    );
  }
}

// GET: List quarantined items
export async function GET(request: Request) {
  const supabase = await createClient();
  const { searchParams } = new URL(request.url);

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

  const resolved = searchParams.get('resolved');
  const domain = searchParams.get('domain');
  const limit = parseInt(searchParams.get('limit') || '100');

  let query = supabase
    .from('training_quarantine')
    .select('*')
    .order('quarantined_at', { ascending: false })
    .limit(limit);

  if (resolved !== null) {
    query = query.eq('resolved', resolved === 'true');
  }

  if (domain) {
    query = query.eq('domain', domain);
  }

  const { data, error } = await query;

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({
    quarantine: data,
    count: data?.length || 0,
    unresolvedCount: data?.filter(d => !d.resolved).length || 0,
  });
}
