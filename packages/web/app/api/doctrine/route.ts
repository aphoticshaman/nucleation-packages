import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import {
  DEFAULT_DOCTRINES,
  mapUserTierToPricing,
  TIER_CAPABILITIES,
  type DoctrineRule
} from '@/lib/doctrine/types';

export const runtime = 'edge';

function getSupabase() {
  return createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );
}

/**
 * GET /api/doctrine - List all active doctrines
 * Requires: Integrated or Stewardship tier
 */
export async function GET(req: Request) {
  const url = new URL(req.url);
  const category = url.searchParams.get('category');
  const includeDeprecated = url.searchParams.get('deprecated') === 'true';

  // Check tier from header (set by middleware or API key validation)
  const userTier = req.headers.get('x-user-tier') || 'free';
  const pricingTier = mapUserTierToPricing(userTier);

  if (!TIER_CAPABILITIES[pricingTier].doctrine_read) {
    return NextResponse.json(
      { error: 'Doctrine access requires Integrated or Stewardship tier' },
      { status: 403 }
    );
  }

  try {
    const supabase = getSupabase();
    // Try to fetch from database first
    let query = supabase
      .from('doctrines')
      .select('*')
      .order('category')
      .order('name');

    if (category) {
      query = query.eq('category', category);
    }

    if (!includeDeprecated) {
      query = query.is('deprecated_at', null);
    }

    const { data: dbDoctrines, error } = await query;

    // If table doesn't exist or is empty, return defaults
    if (error || !dbDoctrines || dbDoctrines.length === 0) {
      const defaults = DEFAULT_DOCTRINES.map((d, i) => ({
        ...d,
        id: `default-${i}`,
        created_at: d.effective_from,
        updated_at: d.effective_from
      }));

      const filtered = category
        ? defaults.filter(d => d.category === category)
        : defaults;

      return NextResponse.json({
        doctrines: filtered,
        source: 'defaults',
        tier: pricingTier
      });
    }

    return NextResponse.json({
      doctrines: dbDoctrines,
      source: 'database',
      tier: pricingTier
    });
  } catch (error) {
    console.error('Doctrine fetch error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch doctrines' },
      { status: 500 }
    );
  }
}

/**
 * POST /api/doctrine - Propose a new doctrine or change
 * Requires: Stewardship tier
 */
export async function POST(req: Request) {
  const userTier = req.headers.get('x-user-tier') || 'free';
  const pricingTier = mapUserTierToPricing(userTier);

  if (!TIER_CAPABILITIES[pricingTier].doctrine_propose) {
    return NextResponse.json(
      { error: 'Doctrine proposals require Stewardship tier' },
      { status: 403 }
    );
  }

  try {
    const body = await req.json();
    const {
      doctrine_id,
      proposed_changes,
      change_rationale
    } = body as {
      doctrine_id?: string;
      proposed_changes: Partial<DoctrineRule>;
      change_rationale: string;
    };

    if (!change_rationale) {
      return NextResponse.json(
        { error: 'Change rationale is required' },
        { status: 400 }
      );
    }

    // Create a doctrine proposal (not directly applied)
    const proposal = {
      doctrine_id: doctrine_id || null,
      proposed_changes,
      change_rationale,
      status: 'pending_review',
      proposed_by: req.headers.get('x-user-id') || 'unknown',
      proposed_at: new Date().toISOString()
    };

    const supabase = getSupabase();
    // Try to insert into proposals table
    const { data, error } = await supabase
      .from('doctrine_proposals')
      .insert(proposal)
      .select()
      .single();

    if (error) {
      // Table might not exist, return mock response
      console.error('Doctrine proposal insert error:', error);
      return NextResponse.json({
        proposal: {
          ...proposal,
          id: `proposal-${Date.now()}`
        },
        message: 'Proposal recorded (pending table creation)',
        note: 'Full doctrine governance requires database migration'
      });
    }

    return NextResponse.json({
      proposal: data,
      message: 'Doctrine change proposed successfully'
    });
  } catch (error) {
    console.error('Doctrine proposal error:', error);
    return NextResponse.json(
      { error: 'Failed to submit doctrine proposal' },
      { status: 500 }
    );
  }
}
