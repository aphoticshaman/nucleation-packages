import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { createCheckoutSession, getOrCreateCustomer, getPriceId, isStripeConfigured } from '@/lib/stripe';

export async function POST(req: Request) {
  try {
    // Check Stripe configuration first
    const stripeCheck = isStripeConfigured();
    if (!stripeCheck.ok) {
      console.error('Stripe not configured. Missing:', stripeCheck.missing);
      return NextResponse.json(
        {
          error: 'Payment system not configured',
          details: process.env.NODE_ENV === 'development'
            ? `Missing env vars: ${stripeCheck.missing.join(', ')}`
            : 'Please contact support',
        },
        { status: 503 }
      );
    }

    const { planId, startTrial } = await req.json();

    // Validate plan
    if (!['pro', 'team', 'enterprise'].includes(planId)) {
      return NextResponse.json({ error: 'Invalid plan' }, { status: 400 });
    }

    // Trial is 14 days for new Pro users - CC saved but not charged until trial ends
    const trialDays = startTrial && planId === 'pro' ? 14 : undefined;

    // Get current user
    const cookieStore = await cookies();
    const supabase = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          get(name: string) {
            return cookieStore.get(name)?.value;
          },
          set(name: string, value: string, options: Record<string, unknown>) {
            try {
              cookieStore.set({ name, value, ...options });
            } catch {
              // Server component - can't set cookies
            }
          },
          remove(name: string, options: Record<string, unknown>) {
            try {
              cookieStore.set({ name, value: '', ...options });
            } catch {
              // Server component - can't set cookies
            }
          },
        },
      }
    );

    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (!user) {
      return NextResponse.json({ error: 'Not authenticated' }, { status: 401 });
    }

    // Get user profile with role and org info
    const { data: profile } = await supabase
      .from('profiles')
      .select('organization_id, full_name, role, organizations(plan)')
      .eq('id', user.id)
      .single();

    // Block admin, enterprise, and support users from purchasing
    const blockedRoles = ['admin', 'enterprise', 'support'];
    if (profile?.role && blockedRoles.includes(profile.role)) {
      return NextResponse.json(
        { error: 'Your account type already has full access. No purchase required.' },
        { status: 403 }
      );
    }

    // Block users who already have enterprise plan
    const orgPlan = (profile?.organizations as { plan?: string } | null)?.plan;
    if (orgPlan === 'enterprise') {
      return NextResponse.json(
        { error: 'Your organization already has an enterprise plan.' },
        { status: 403 }
      );
    }

    let organizationId = profile?.organization_id;

    // Create org if needed
    if (!organizationId) {
      const { data: newOrg, error: orgError } = await supabase
        .from('organizations')
        .insert({
          name: profile?.full_name ? `${profile.full_name}'s Organization` : 'My Organization',
          slug: `org-${user.id.slice(0, 8)}`,
        })
        .select()
        .single();

      if (orgError) {
        console.error('Failed to create org:', orgError);
        return NextResponse.json({ error: 'Failed to create organization' }, { status: 500 });
      }

      organizationId = newOrg.id;

      // Link user to org
      await supabase.from('profiles').update({ organization_id: organizationId }).eq('id', user.id);
    }

    // Get org details
    const { data: org } = await supabase
      .from('organizations')
      .select('stripe_customer_id')
      .eq('id', organizationId)
      .single();

    // Get or create Stripe customer
    let customerId = org?.stripe_customer_id;

    if (!customerId) {
      const customer = await getOrCreateCustomer({
        email: user.email!,
        name: profile?.full_name || undefined,
        organizationId,
      });
      customerId = customer.id;

      // Save customer ID
      await supabase
        .from('organizations')
        .update({ stripe_customer_id: customerId })
        .eq('id', organizationId);
    }

    // Get price ID at runtime (not build time)
    const priceId = getPriceId(planId as 'pro' | 'team' | 'enterprise');

    if (!priceId) {
      console.error(`Price not configured for plan: ${planId}`);
      console.error('Available env vars:', {
        STRIPE_PRICE_STARTER: !!process.env.STRIPE_PRICE_STARTER,
        STRIPE_PRICE_PRO: !!process.env.STRIPE_PRICE_PRO,
        STRIPE_PRICE_ENTERPRISE: !!process.env.STRIPE_PRICE_ENTERPRISE,
      });
      return NextResponse.json({ error: 'Price not configured' }, { status: 500 });
    }

    // Create checkout session
    const origin = req.headers.get('origin') || 'http://localhost:3000';

    const session = await createCheckoutSession({
      priceId,
      customerId,
      organizationId,
      successUrl: `${origin}/dashboard?checkout=success`,
      cancelUrl: `${origin}/pricing?checkout=canceled`,
      trialDays,
    });

    return NextResponse.json({ url: session.url });
  } catch (error) {
    console.error('Checkout error:', error);
    return NextResponse.json({ error: 'Failed to create checkout session' }, { status: 500 });
  }
}
