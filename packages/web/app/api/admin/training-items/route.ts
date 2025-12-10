import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';

export const runtime = 'edge';

/**
 * Training Items API
 *
 * Granular, itemized training data management for Elle and Guardian.
 * Every action is immutably logged for auditability.
 *
 * GET: List training items with filters
 * POST: Create item, select/deselect, approve/reject, export batch
 */

interface TrainingItem {
  id: string;
  source_type: string;
  source_id: string | null;
  target_system: 'elle' | 'guardian' | 'both';
  domain: string | null;
  input_text: string;
  expected_output: string | null;
  actual_output: string | null;
  training_type: string;
  quality_score: number | null;
  difficulty_level: string | null;
  tags: string[] | null;
  selected_for_export: boolean;
  export_priority: number;
  status: string;
  created_at: string;
  created_by: string | null;
  reviewed_at: string | null;
  reviewed_by: string | null;
}

export async function GET(req: Request) {
  try {
    const cookieStore = await cookies();
    const supabase = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          get(name: string) {
            return cookieStore.get(name)?.value;
          },
          set() {},
          remove() {},
        },
      }
    );

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

    if (!profile || (profile as { role?: string }).role !== 'admin') {
      return NextResponse.json({ error: 'Admin access required' }, { status: 403 });
    }

    // Parse query params
    const url = new URL(req.url);
    const view = url.searchParams.get('view') || 'list';
    const targetSystem = url.searchParams.get('target');
    const domain = url.searchParams.get('domain');
    const status = url.searchParams.get('status');
    const selectedOnly = url.searchParams.get('selected') === 'true';
    const page = parseInt(url.searchParams.get('page') || '1');
    const limit = Math.min(parseInt(url.searchParams.get('limit') || '50'), 100);
    const offset = (page - 1) * limit;

    if (view === 'list') {
      // Build query
      let query = supabase
        .from('v_training_items_active')
        .select('*', { count: 'exact' });

      if (targetSystem) {
        query = query.eq('target_system', targetSystem);
      }
      if (domain) {
        query = query.eq('domain', domain);
      }
      if (status) {
        query = query.eq('status', status);
      }
      if (selectedOnly) {
        query = query.eq('selected_for_export', true);
      }

      query = query
        .order('created_at', { ascending: false })
        .range(offset, offset + limit - 1);

      const { data: items, error, count } = await query;

      if (error) {
        console.error('Error fetching training items:', error);
        throw error;
      }

      return NextResponse.json({
        items: items || [],
        pagination: {
          page,
          limit,
          total: count || 0,
          totalPages: Math.ceil((count || 0) / limit),
        },
      });
    }

    if (view === 'stats') {
      // Get statistics by domain
      const { data: stats } = await supabase
        .from('v_training_stats_by_domain')
        .select('*');

      // Get totals
      const { data: totals } = await supabase
        .from('v_training_items_active')
        .select('id, target_system, selected_for_export, status');

      const summary = {
        total: totals?.length || 0,
        elle: totals?.filter(t => t.target_system === 'elle' || t.target_system === 'both').length || 0,
        guardian: totals?.filter(t => t.target_system === 'guardian' || t.target_system === 'both').length || 0,
        selected: totals?.filter(t => t.selected_for_export).length || 0,
        approved: totals?.filter(t => t.status === 'approved').length || 0,
        exported: totals?.filter(t => t.status === 'exported').length || 0,
      };

      return NextResponse.json({ stats: stats || [], summary });
    }

    if (view === 'audit') {
      // Get recent audit log
      const { data: auditLog } = await supabase
        .from('v_training_audit_recent')
        .select('*')
        .limit(100);

      // Verify integrity
      const { data: integrity } = await supabase.rpc('verify_audit_log_integrity');

      return NextResponse.json({
        auditLog: auditLog || [],
        integrity: integrity?.[0] || { is_valid: true, total_entries: 0, verified_entries: 0 },
      });
    }

    if (view === 'export_batches') {
      const { data: batches } = await supabase
        .from('training_export_batches')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(50);

      return NextResponse.json({ batches: batches || [] });
    }

    return NextResponse.json({ error: 'Invalid view' }, { status: 400 });

  } catch (error) {
    console.error('Training Items API error:', error);
    return NextResponse.json({ error: 'Internal error' }, { status: 500 });
  }
}

export async function POST(req: Request) {
  try {
    const cookieStore = await cookies();
    const supabase = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          get(name: string) {
            return cookieStore.get(name)?.value;
          },
          set() {},
          remove() {},
        },
      }
    );

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

    if (!profile || (profile as { role?: string }).role !== 'admin') {
      return NextResponse.json({ error: 'Admin access required' }, { status: 403 });
    }

    const body = await req.json();
    const { action } = body;

    // ═══════════════════════════════════════════════════════════════════════════
    // CREATE ITEM - Add new training item manually
    // ═══════════════════════════════════════════════════════════════════════════
    if (action === 'create') {
      const { targetSystem, domain, inputText, expectedOutput, trainingType, tags, qualityScore, difficultyLevel } = body;

      if (!targetSystem || !inputText || !trainingType) {
        return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
      }

      const { data: item, error } = await supabase
        .from('training_items')
        .insert({
          source_type: 'human_authored',
          target_system: targetSystem,
          domain,
          input_text: inputText,
          expected_output: expectedOutput,
          training_type: trainingType,
          tags,
          quality_score: qualityScore,
          difficulty_level: difficultyLevel,
          created_by: user.id,
        })
        .select()
        .single();

      if (error) {
        console.error('Error creating training item:', error);
        return NextResponse.json({ error: 'Failed to create item' }, { status: 500 });
      }

      // Log action
      await supabase.rpc('log_training_action', {
        p_action: 'item_created',
        p_user_id: user.id,
        p_entity_type: 'training_item',
        p_entity_id: (item as TrainingItem).id,
        p_new_state: { target_system: targetSystem, domain, training_type: trainingType },
        p_reason: 'Manual item creation',
      });

      return NextResponse.json({ success: true, item });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SELECT/DESELECT - Toggle selection for export
    // ═══════════════════════════════════════════════════════════════════════════
    if (action === 'toggle_selection') {
      const { itemId, selected, reason } = body;

      // Get current state
      const { data: currentItem } = await supabase
        .from('training_items')
        .select('selected_for_export')
        .eq('id', itemId)
        .single();

      const { error } = await supabase
        .from('training_items')
        .update({ selected_for_export: selected })
        .eq('id', itemId);

      if (error) {
        return NextResponse.json({ error: 'Failed to update selection' }, { status: 500 });
      }

      // Log action
      await supabase.rpc('log_training_action', {
        p_action: selected ? 'item_selected' : 'item_deselected',
        p_user_id: user.id,
        p_entity_type: 'training_item',
        p_entity_id: itemId,
        p_previous_state: { selected_for_export: (currentItem as { selected_for_export: boolean })?.selected_for_export },
        p_new_state: { selected_for_export: selected },
        p_reason: reason || (selected ? 'Selected for training' : 'Deselected from training'),
      });

      return NextResponse.json({ success: true, selected });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // BULK SELECT - Select multiple items at once
    // ═══════════════════════════════════════════════════════════════════════════
    if (action === 'bulk_select') {
      const { itemIds, selected, reason } = body;

      if (!Array.isArray(itemIds) || itemIds.length === 0) {
        return NextResponse.json({ error: 'No items specified' }, { status: 400 });
      }

      const { error } = await supabase
        .from('training_items')
        .update({ selected_for_export: selected })
        .in('id', itemIds);

      if (error) {
        return NextResponse.json({ error: 'Failed to update selections' }, { status: 500 });
      }

      // Log bulk action
      await supabase.rpc('log_training_action', {
        p_action: selected ? 'bulk_selected' : 'bulk_deselected',
        p_user_id: user.id,
        p_entity_type: 'bulk_selection',
        p_entity_ids: itemIds,
        p_new_state: { selected_for_export: selected, count: itemIds.length },
        p_reason: reason || `Bulk ${selected ? 'selected' : 'deselected'} ${itemIds.length} items`,
      });

      return NextResponse.json({ success: true, updated: itemIds.length });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // APPROVE/REJECT - Review training item quality
    // ═══════════════════════════════════════════════════════════════════════════
    if (action === 'review') {
      const { itemId, status: newStatus, reason, qualityScore } = body;

      if (!['approved', 'rejected'].includes(newStatus)) {
        return NextResponse.json({ error: 'Invalid status' }, { status: 400 });
      }

      // Get current state
      const { data: currentItem } = await supabase
        .from('training_items')
        .select('status, quality_score')
        .eq('id', itemId)
        .single();

      const updateData: Record<string, unknown> = {
        status: newStatus,
        reviewed_at: new Date().toISOString(),
        reviewed_by: user.id,
      };

      if (qualityScore !== undefined) {
        updateData.quality_score = qualityScore;
      }

      const { error } = await supabase
        .from('training_items')
        .update(updateData)
        .eq('id', itemId);

      if (error) {
        return NextResponse.json({ error: 'Failed to review item' }, { status: 500 });
      }

      // Log action
      await supabase.rpc('log_training_action', {
        p_action: `item_${newStatus}`,
        p_user_id: user.id,
        p_entity_type: 'training_item',
        p_entity_id: itemId,
        p_previous_state: currentItem,
        p_new_state: { status: newStatus, quality_score: qualityScore },
        p_reason: reason || `Item ${newStatus}`,
      });

      return NextResponse.json({ success: true, status: newStatus });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // DELETE (SOFT) - Mark item as deleted
    // ═══════════════════════════════════════════════════════════════════════════
    if (action === 'delete') {
      const { itemId, reason } = body;

      // Get current state
      const { data: currentItem } = await supabase
        .from('training_items')
        .select('*')
        .eq('id', itemId)
        .single();

      const { error } = await supabase
        .from('training_items')
        .update({
          deleted_at: new Date().toISOString(),
          deleted_by: user.id,
          deletion_reason: reason,
        })
        .eq('id', itemId);

      if (error) {
        return NextResponse.json({ error: 'Failed to delete item' }, { status: 500 });
      }

      // Log action (immutable record of deletion)
      await supabase.rpc('log_training_action', {
        p_action: 'item_deleted',
        p_user_id: user.id,
        p_entity_type: 'training_item',
        p_entity_id: itemId,
        p_previous_state: currentItem,
        p_new_state: { deleted: true },
        p_reason: reason || 'Item deleted',
      });

      return NextResponse.json({ success: true, deleted: true });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EXPORT - Create export batch from selected items
    // ═══════════════════════════════════════════════════════════════════════════
    if (action === 'export') {
      const { exportName, targetSystem, exportFormat, filters, notes } = body;

      if (!exportName || !targetSystem || !exportFormat) {
        return NextResponse.json({ error: 'Missing export parameters' }, { status: 400 });
      }

      // Get selected items
      let query = supabase
        .from('v_training_items_selected')
        .select('*');

      if (targetSystem !== 'both') {
        query = query.or(`target_system.eq.${targetSystem},target_system.eq.both`);
      }

      const { data: selectedItems, error: selectError } = await query;

      if (selectError) {
        return NextResponse.json({ error: 'Failed to fetch selected items' }, { status: 500 });
      }

      if (!selectedItems || selectedItems.length === 0) {
        return NextResponse.json({ error: 'No items selected for export' }, { status: 400 });
      }

      // Create export batch
      const { data: batch, error: batchError } = await supabase
        .from('training_export_batches')
        .insert({
          export_name: exportName,
          target_system: targetSystem,
          export_format: exportFormat,
          total_items: selectedItems.length,
          elle_items: selectedItems.filter(i => i.target_system === 'elle' || i.target_system === 'both').length,
          guardian_items: selectedItems.filter(i => i.target_system === 'guardian' || i.target_system === 'both').length,
          filters_applied: filters,
          status: 'completed',
          created_by: user.id,
          completed_at: new Date().toISOString(),
          notes,
        })
        .select()
        .single();

      if (batchError) {
        return NextResponse.json({ error: 'Failed to create export batch' }, { status: 500 });
      }

      const batchId = (batch as { id: string }).id;

      // Link items to batch
      const itemExports = selectedItems.map(item => ({
        training_item_id: item.id,
        export_batch_id: batchId,
      }));

      await supabase.from('training_item_exports').insert(itemExports);

      // Mark items as exported
      const itemIds = selectedItems.map(i => i.id);
      await supabase
        .from('training_items')
        .update({
          status: 'exported',
          exported_at: new Date().toISOString(),
          exported_by: user.id,
        })
        .in('id', itemIds);

      // Format export data based on format
      const exportData = formatExportData(selectedItems, exportFormat, targetSystem);

      // Log action
      await supabase.rpc('log_training_action', {
        p_action: 'batch_exported',
        p_user_id: user.id,
        p_entity_type: 'export_batch',
        p_entity_id: batchId,
        p_entity_ids: itemIds,
        p_new_state: {
          batch_id: batchId,
          format: exportFormat,
          total_items: selectedItems.length,
          target_system: targetSystem,
        },
        p_reason: notes || `Exported ${selectedItems.length} items`,
        p_metadata: { export_name: exportName, filters },
      });

      return NextResponse.json({
        success: true,
        batch,
        data: exportData,
        itemCount: selectedItems.length,
      });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CREATE FROM EVALUATION - Generate training item from disagreement
    // ═══════════════════════════════════════════════════════════════════════════
    if (action === 'create_from_evaluation') {
      const { evaluationId } = body;

      const { data: itemId, error } = await supabase.rpc('create_training_item_from_evaluation', {
        p_evaluation_id: evaluationId,
        p_user_id: user.id,
      });

      if (error) {
        console.error('Error creating from evaluation:', error);
        return NextResponse.json({ error: error.message }, { status: 500 });
      }

      return NextResponse.json({ success: true, itemId });
    }

    return NextResponse.json({ error: 'Invalid action' }, { status: 400 });

  } catch (error) {
    console.error('Training Items API error:', error);
    return NextResponse.json({ error: 'Internal error' }, { status: 500 });
  }
}

/**
 * Format export data based on requested format
 */
function formatExportData(
  items: TrainingItem[],
  format: string,
  targetSystem: string
): string {
  const filteredItems = targetSystem === 'both'
    ? items
    : items.filter(i => i.target_system === targetSystem || i.target_system === 'both');

  switch (format) {
    case 'jsonl':
      // JSONL format for fine-tuning
      return filteredItems.map(item => JSON.stringify({
        instruction: item.input_text,
        output: item.expected_output || '',
        input: item.actual_output || '',
        domain: item.domain,
        type: item.training_type,
        quality: item.quality_score,
      })).join('\n');

    case 'alpaca':
      // Alpaca format
      return JSON.stringify(filteredItems.map(item => ({
        instruction: item.input_text,
        input: '',
        output: item.expected_output || '',
      })), null, 2);

    case 'sharegpt':
      // ShareGPT format for chat fine-tuning
      return JSON.stringify(filteredItems.map(item => ({
        conversations: [
          { from: 'human', value: item.input_text },
          { from: 'gpt', value: item.expected_output || '' },
        ],
        source: 'latticeforge',
        domain: item.domain,
      })), null, 2);

    case 'csv':
      // CSV format
      const header = 'input_text,expected_output,domain,training_type,quality_score';
      const rows = filteredItems.map(item =>
        `"${(item.input_text || '').replace(/"/g, '""')}","${(item.expected_output || '').replace(/"/g, '""')}","${item.domain || ''}","${item.training_type}","${item.quality_score || ''}"`
      );
      return [header, ...rows].join('\n');

    default:
      // Default JSON
      return JSON.stringify(filteredItems, null, 2);
  }
}
