import { headers } from 'next/headers';
import { NextResponse } from 'next/server';
import Stripe from 'stripe';
import { stripe, PLANS, PlanId } from '@/lib/stripe';
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import { sendEmail, emailTemplates } from '@/lib/email';

// Lazy-initialized Supabase admin client (avoids build-time env var access)
let _supabaseAdmin: SupabaseClient | null = null;

function getSupabaseAdmin(): SupabaseClient {
  if (!_supabaseAdmin) {
    _supabaseAdmin = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );
  }
  return _supabaseAdmin;
}

export async function POST(req: Request) {
  const body = await req.text();
  const headersList = await headers();
  const signature = headersList.get('stripe-signature');

  if (!signature) {
    return NextResponse.json({ error: 'No signature' }, { status: 400 });
  }

  let event: Stripe.Event;

  try {
    event = stripe.webhooks.constructEvent(body, signature, process.env.STRIPE_WEBHOOK_SECRET!);
  } catch (err) {
    console.error('Webhook signature verification failed:', err);
    return NextResponse.json({ error: 'Invalid signature' }, { status: 400 });
  }

  try {
    switch (event.type) {
      case 'checkout.session.completed': {
        const session = event.data.object as Stripe.Checkout.Session;
        await handleCheckoutComplete(session);
        break;
      }

      case 'customer.subscription.created':
      case 'customer.subscription.updated': {
        const subscription = event.data.object as Stripe.Subscription;
        await handleSubscriptionUpdate(subscription);
        break;
      }

      case 'customer.subscription.deleted': {
        const subscription = event.data.object as Stripe.Subscription;
        await handleSubscriptionCanceled(subscription);
        break;
      }

      case 'invoice.payment_succeeded': {
        const invoice = event.data.object as Stripe.Invoice;
        await handlePaymentSucceeded(invoice);
        break;
      }

      case 'invoice.payment_failed': {
        const invoice = event.data.object as Stripe.Invoice;
        await handlePaymentFailed(invoice);
        break;
      }

      default:
        console.log(`Unhandled event type: ${event.type}`);
    }

    return NextResponse.json({ received: true });
  } catch (error) {
    console.error('Webhook handler error:', error);
    return NextResponse.json({ error: 'Webhook handler failed' }, { status: 500 });
  }
}

async function handleCheckoutComplete(session: Stripe.Checkout.Session) {
  const organizationId = session.metadata?.organization_id;

  if (!organizationId) {
    console.error('No organization_id in checkout session metadata');
    return;
  }

  // Update organization with Stripe customer ID
  await getSupabaseAdmin()
    .from('organizations')
    .update({
      stripe_customer_id: session.customer as string,
    })
    .eq('id', organizationId);

  console.log(`Checkout complete for org ${organizationId}`);
}

async function handleSubscriptionUpdate(subscription: Stripe.Subscription) {
  const organizationId = subscription.metadata?.organization_id;

  if (!organizationId) {
    console.error('No organization_id in subscription metadata');
    return;
  }

  // Get the price to determine the plan
  const priceId = subscription.items.data[0]?.price.id;
  const plan = getPlanFromPriceId(priceId);
  const planLimits = PLANS[plan as PlanId]?.limits;

  // Update organization
  await getSupabaseAdmin()
    .from('organizations')
    .update({
      plan,
      plan_status: subscription.status,
      api_calls_limit: planLimits?.api_calls || 1000,
      team_seats_limit: planLimits?.team_seats || 5,
    })
    .eq('id', organizationId);

  // Update user role to enterprise
  await getSupabaseAdmin()
    .from('profiles')
    .update({ role: 'enterprise' })
    .eq('organization_id', organizationId);

  console.log(`Subscription updated for org ${organizationId}: ${plan}`);
}

async function handleSubscriptionCanceled(subscription: Stripe.Subscription) {
  const organizationId = subscription.metadata?.organization_id;

  if (!organizationId) {
    console.error('No organization_id in subscription metadata');
    return;
  }

  // Get organization details before update
  const { data: org } = await getSupabaseAdmin()
    .from('organizations')
    .select('id, name')
    .eq('id', organizationId)
    .single();

  // Downgrade to free
  await getSupabaseAdmin()
    .from('organizations')
    .update({
      plan: 'free',
      plan_status: 'canceled',
      api_calls_limit: 0,
      team_seats_limit: 1,
    })
    .eq('id', organizationId);

  // Downgrade users to consumer
  await getSupabaseAdmin()
    .from('profiles')
    .update({ role: 'consumer' })
    .eq('organization_id', organizationId);

  // Send cancellation email to admin
  if (org) {
    const { data: admin } = await getSupabaseAdmin()
      .from('profiles')
      .select('email')
      .eq('organization_id', organizationId)
      .limit(1)
      .single();

    if (admin?.email) {
      const template = emailTemplates.subscriptionCanceled(org.name || 'Your Organization');
      await sendEmail({
        to: admin.email,
        subject: template.subject,
        html: template.html,
        text: template.text,
      });
    }
  }

  console.log(`Subscription canceled for org ${organizationId}`);
}

async function handlePaymentSucceeded(invoice: Stripe.Invoice) {
  const customerId = invoice.customer as string;

  // Get organization by customer ID
  const { data: org } = await getSupabaseAdmin()
    .from('organizations')
    .select('id, name, plan')
    .eq('stripe_customer_id', customerId)
    .single();

  if (org) {
    // Reset API usage for the new billing period
    await getSupabaseAdmin().from('organizations').update({ api_calls_used: 0 }).eq('id', org.id);

    // Send payment receipt email
    const { data: admin } = await getSupabaseAdmin()
      .from('profiles')
      .select('email')
      .eq('organization_id', org.id)
      .limit(1)
      .single();

    if (admin?.email) {
      const template = emailTemplates.paymentSucceeded(
        org.name || 'Your Organization',
        invoice.amount_paid || 0,
        org.plan || 'Pro'
      );
      await sendEmail({
        to: admin.email,
        subject: template.subject,
        html: template.html,
        text: template.text,
      });
    }

    console.log(`Payment succeeded, reset usage for org ${org.id}`);
  }
}

async function handlePaymentFailed(invoice: Stripe.Invoice) {
  const customerId = invoice.customer as string;

  // Get organization by customer ID
  const { data: org } = await getSupabaseAdmin()
    .from('organizations')
    .select('id, name')
    .eq('stripe_customer_id', customerId)
    .single();

  if (org) {
    // Update status to past_due
    await getSupabaseAdmin()
      .from('organizations')
      .update({ plan_status: 'past_due' })
      .eq('id', org.id);

    // Send payment failed email notification
    const { data: admin } = await getSupabaseAdmin()
      .from('profiles')
      .select('email')
      .eq('organization_id', org.id)
      .limit(1)
      .single();

    if (admin?.email) {
      const template = emailTemplates.paymentFailed(
        org.name || 'Your Organization',
        invoice.amount_due || undefined
      );
      await sendEmail({
        to: admin.email,
        subject: template.subject,
        html: template.html,
        text: template.text,
      });
      console.log(`Payment failed email sent to ${admin.email} for org ${org.id}`);
    }

    console.log(`Payment failed for org ${org.id}`);
  }
}

function getPlanFromPriceId(priceId: string): string {
  if (priceId === process.env.STRIPE_PRICE_STARTER) return 'starter';
  if (priceId === process.env.STRIPE_PRICE_PRO) return 'pro';
  if (priceId === process.env.STRIPE_PRICE_ENTERPRISE) return 'enterprise';
  return 'free';
}
