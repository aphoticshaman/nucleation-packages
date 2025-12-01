// LatticeForge Billing - Stripe Integration
// Handles: checkout, portal, webhooks, tier upgrades
// Deploy: supabase functions deploy billing

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import Stripe from 'https://esm.sh/stripe@14.10.0?target=deno'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type, stripe-signature',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
}

// Price IDs from your Stripe dashboard
const STRIPE_PRICES: Record<string, string> = {
  pro_monthly: Deno.env.get('STRIPE_PRICE_PRO_MONTHLY') || '',
  pro_yearly: Deno.env.get('STRIPE_PRICE_PRO_YEARLY') || '',
  enterprise_monthly: Deno.env.get('STRIPE_PRICE_ENTERPRISE_MONTHLY') || '',
  enterprise_yearly: Deno.env.get('STRIPE_PRICE_ENTERPRISE_YEARLY') || '',
}

// Tier mapping from Stripe price to our tiers
const PRICE_TO_TIER: Record<string, 'pro' | 'enterprise'> = {}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  const stripeKey = Deno.env.get('STRIPE_SECRET_KEY')
  if (!stripeKey) {
    return jsonResponse({ error: 'Stripe not configured' }, 500)
  }

  const stripe = new Stripe(stripeKey, {
    apiVersion: '2023-10-16',
    httpClient: Stripe.createFetchHttpClient(),
  })

  const supabase = createClient(
    Deno.env.get('SUPABASE_URL') ?? '',
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
  )

  try {
    const url = new URL(req.url)
    const path = url.pathname.replace('/billing', '')

    switch (path) {
      case '/checkout':
        return handleCheckout(req, stripe, supabase)

      case '/portal':
        return handlePortal(req, stripe, supabase)

      case '/webhook':
        return handleWebhook(req, stripe, supabase)

      case '/status':
        return handleStatus(req, supabase)

      default:
        return jsonResponse({ error: 'Not found' }, 404)
    }
  } catch (error) {
    console.error('Billing error:', error)
    return jsonResponse({ error: 'Internal error' }, 500)
  }
})

// ============================================
// CHECKOUT - Create subscription
// ============================================

async function handleCheckout(req: Request, stripe: Stripe, supabase: any) {
  if (req.method !== 'POST') {
    return jsonResponse({ error: 'Method not allowed' }, 405)
  }

  const authHeader = req.headers.get('Authorization')
  if (!authHeader?.startsWith('Bearer ')) {
    return jsonResponse({ error: 'Authentication required' }, 401)
  }

  const { data: authData } = await supabase.auth.getUser(authHeader.replace('Bearer ', ''))
  if (!authData.user) {
    return jsonResponse({ error: 'Invalid token' }, 401)
  }

  const body = await req.json()
  const { price_id, success_url, cancel_url } = body

  if (!price_id || !STRIPE_PRICES[price_id]) {
    return jsonResponse({ error: 'Invalid price' }, 400)
  }

  // Get or create Stripe customer
  const { data: client } = await supabase
    .from('clients')
    .select('id, stripe_customer_id, email')
    .eq('user_id', authData.user.id)
    .single()

  if (!client) {
    return jsonResponse({ error: 'Client not found' }, 404)
  }

  let customerId = client.stripe_customer_id

  if (!customerId) {
    const customer = await stripe.customers.create({
      email: client.email,
      metadata: {
        client_id: client.id,
        supabase_user_id: authData.user.id,
      },
    })

    customerId = customer.id

    await supabase
      .from('clients')
      .update({ stripe_customer_id: customerId })
      .eq('id', client.id)
  }

  // Create checkout session
  const session = await stripe.checkout.sessions.create({
    customer: customerId,
    mode: 'subscription',
    payment_method_types: ['card'],
    line_items: [
      {
        price: STRIPE_PRICES[price_id],
        quantity: 1,
      },
    ],
    success_url: success_url || `${Deno.env.get('DASHBOARD_URL')}/billing?success=true`,
    cancel_url: cancel_url || `${Deno.env.get('DASHBOARD_URL')}/billing?canceled=true`,
    metadata: {
      client_id: client.id,
    },
  })

  return jsonResponse({ url: session.url })
}

// ============================================
// PORTAL - Customer billing portal
// ============================================

async function handlePortal(req: Request, stripe: Stripe, supabase: any) {
  if (req.method !== 'POST') {
    return jsonResponse({ error: 'Method not allowed' }, 405)
  }

  const authHeader = req.headers.get('Authorization')
  if (!authHeader?.startsWith('Bearer ')) {
    return jsonResponse({ error: 'Authentication required' }, 401)
  }

  const { data: authData } = await supabase.auth.getUser(authHeader.replace('Bearer ', ''))
  if (!authData.user) {
    return jsonResponse({ error: 'Invalid token' }, 401)
  }

  const { data: client } = await supabase
    .from('clients')
    .select('stripe_customer_id')
    .eq('user_id', authData.user.id)
    .single()

  if (!client?.stripe_customer_id) {
    return jsonResponse({ error: 'No billing account' }, 404)
  }

  const body = await req.json().catch(() => ({}))

  const session = await stripe.billingPortal.sessions.create({
    customer: client.stripe_customer_id,
    return_url: body.return_url || `${Deno.env.get('DASHBOARD_URL')}/billing`,
  })

  return jsonResponse({ url: session.url })
}

// ============================================
// WEBHOOK - Handle Stripe events
// ============================================

async function handleWebhook(req: Request, stripe: Stripe, supabase: any) {
  if (req.method !== 'POST') {
    return jsonResponse({ error: 'Method not allowed' }, 405)
  }

  const signature = req.headers.get('stripe-signature')
  const webhookSecret = Deno.env.get('STRIPE_WEBHOOK_SECRET')

  if (!signature || !webhookSecret) {
    return jsonResponse({ error: 'Missing signature' }, 400)
  }

  const body = await req.text()

  let event: Stripe.Event

  try {
    event = stripe.webhooks.constructEvent(body, signature, webhookSecret)
  } catch (err) {
    console.error('Webhook signature verification failed:', err)
    return jsonResponse({ error: 'Invalid signature' }, 400)
  }

  console.log('Stripe webhook:', event.type)

  switch (event.type) {
    case 'checkout.session.completed': {
      const session = event.data.object as Stripe.Checkout.Session
      await handleCheckoutComplete(session, stripe, supabase)
      break
    }

    case 'customer.subscription.updated': {
      const subscription = event.data.object as Stripe.Subscription
      await handleSubscriptionUpdate(subscription, supabase)
      break
    }

    case 'customer.subscription.deleted': {
      const subscription = event.data.object as Stripe.Subscription
      await handleSubscriptionCancel(subscription, supabase)
      break
    }

    case 'invoice.payment_failed': {
      const invoice = event.data.object as Stripe.Invoice
      await handlePaymentFailed(invoice, supabase)
      break
    }
  }

  return jsonResponse({ received: true })
}

async function handleCheckoutComplete(session: Stripe.Checkout.Session, stripe: Stripe, supabase: any) {
  const clientId = session.metadata?.client_id
  if (!clientId) return

  // Get subscription details
  if (session.subscription) {
    const subscription = await stripe.subscriptions.retrieve(session.subscription as string)
    const priceId = subscription.items.data[0]?.price.id
    const tier = getTierFromPrice(priceId)

    if (tier) {
      await supabase
        .from('clients')
        .update({
          tier,
          stripe_subscription_id: subscription.id,
          subscription_status: 'active',
        })
        .eq('id', clientId)

      console.log(`Client ${clientId} upgraded to ${tier}`)
    }
  }
}

async function handleSubscriptionUpdate(subscription: Stripe.Subscription, supabase: any) {
  const customerId = subscription.customer as string

  const { data: client } = await supabase
    .from('clients')
    .select('id')
    .eq('stripe_customer_id', customerId)
    .single()

  if (!client) return

  const priceId = subscription.items.data[0]?.price.id
  const tier = getTierFromPrice(priceId)
  const status = subscription.status

  await supabase
    .from('clients')
    .update({
      tier: tier || 'free',
      subscription_status: status,
    })
    .eq('id', client.id)
}

async function handleSubscriptionCancel(subscription: Stripe.Subscription, supabase: any) {
  const customerId = subscription.customer as string

  const { data: client } = await supabase
    .from('clients')
    .select('id')
    .eq('stripe_customer_id', customerId)
    .single()

  if (!client) return

  // Downgrade to free tier
  await supabase
    .from('clients')
    .update({
      tier: 'free',
      subscription_status: 'canceled',
    })
    .eq('id', client.id)

  console.log(`Client ${client.id} downgraded to free`)
}

async function handlePaymentFailed(invoice: Stripe.Invoice, supabase: any) {
  const customerId = invoice.customer as string

  const { data: client } = await supabase
    .from('clients')
    .select('id, email')
    .eq('stripe_customer_id', customerId)
    .single()

  if (!client) return

  // Create alert for failed payment
  await supabase.from('alerts').insert({
    client_id: client.id,
    type: 'billing',
    severity: 'critical',
    title: 'Payment Failed',
    message: 'Your subscription payment failed. Please update your payment method to avoid service interruption.',
  })
}

// ============================================
// STATUS - Get billing status
// ============================================

async function handleStatus(req: Request, supabase: any) {
  const authHeader = req.headers.get('Authorization')
  if (!authHeader?.startsWith('Bearer ')) {
    return jsonResponse({ error: 'Authentication required' }, 401)
  }

  const { data: authData } = await supabase.auth.getUser(authHeader.replace('Bearer ', ''))
  if (!authData.user) {
    return jsonResponse({ error: 'Invalid token' }, 401)
  }

  const { data: client } = await supabase
    .from('clients')
    .select('tier, subscription_status, stripe_customer_id')
    .eq('user_id', authData.user.id)
    .single()

  if (!client) {
    return jsonResponse({ error: 'Client not found' }, 404)
  }

  return jsonResponse({
    tier: client.tier,
    subscription_status: client.subscription_status,
    has_billing_account: !!client.stripe_customer_id,
  })
}

// ============================================
// HELPERS
// ============================================

function getTierFromPrice(priceId: string): 'pro' | 'enterprise' | null {
  if (priceId === STRIPE_PRICES.pro_monthly || priceId === STRIPE_PRICES.pro_yearly) {
    return 'pro'
  }
  if (priceId === STRIPE_PRICES.enterprise_monthly || priceId === STRIPE_PRICES.enterprise_yearly) {
    return 'enterprise'
  }
  return null
}

function jsonResponse(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  })
}
