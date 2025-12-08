# VeilPath Economy & Pricing Design

## Executive Summary

This document outlines the complete monetization strategy for VeilPath, including:
- Virtual currency (Veil Shards) pricing
- Subscription tiers
- Feature gating (free vs premium)
- Payment gateway strategy
- Accounting/compliance integration

**Goal**: Sustainable revenue while keeping the app accessible. Users should feel rewarded for engagement, not punished for being free.

---

## 1. Currency System

### Dual Currency Model

| Currency | Symbol | Type | Acquisition |
|----------|--------|------|-------------|
| **Veil Shards** | 💎 | Premium | Purchase with real money OR first-time bonuses |
| **Moonlight** | 🌙 | Soft | Earned through gameplay (readings, journals, streaks) |

### Why Dual Currency?
- **Veil Shards** (premium) = Clear real-money value, drives IAP revenue
- **Moonlight** (soft) = Engagement reward, no real value, keeps free users engaged
- Separation prevents "pay to play" feeling while maintaining monetization

### Veil Shards Pricing (IAP)

Based on industry research: $1 = ~100 premium currency is the psychological sweet spot.

| Package | Price | Shards | $/Shard | Bonus | Best Value? |
|---------|-------|--------|---------|-------|-------------|
| Starter | $0.99 | 100 | $0.0099 | 0% | No |
| Small | $4.99 | 550 | $0.0091 | 10% | No |
| Medium | $9.99 | 1,200 | $0.0083 | 20% | No |
| **Large** | $19.99 | 2,600 | $0.0077 | **30%** | **YES** ⭐ |
| Huge | $49.99 | 7,000 | $0.0071 | 40% | Whales only |
| Ultimate | $99.99 | 15,000 | $0.0067 | 50% | Whales only |

**First-Time Buyer Bonus**: 2x shards on first purchase (limited time offer psychology)

### Moonlight Earning Rates

| Action | Moonlight Earned |
|--------|------------------|
| Complete a reading | 10 🌙 |
| Write journal entry | 15 🌙 |
| 7-day streak | 50 🌙 |
| Complete mindfulness session | 20 🌙 |
| Daily login | 5 🌙 |
| Answer reflection questions | 5 🌙 |

---

## 2. Subscription Tiers

Based on Headspace ($12.99/mo), Calm ($14.99/mo), and wellness app benchmarks.

### Tier Structure

| Tier | Monthly | Annual | Lifetime | Features |
|------|---------|--------|----------|----------|
| **Free** | $0 | $0 | $0 | Basic readings, limited journal, ads |
| **Seeker** | $4.99 | $39.99 ($3.33/mo) | $99.99 | No ads, unlimited journal, basic Oracle |
| **Adept** | $9.99 | $79.99 ($6.67/mo) | $199.99 | All Seeker + Premium spreads, full Oracle, CBT/DBT tools |
| **Mystic** | $14.99 | $119.99 ($10/mo) | $299.99 | Everything + Voice narration, exclusive cosmetics, priority support |

### Why These Prices?
- **$4.99** = Impulse buy threshold (coffee price)
- **$9.99** = Sweet spot for engaged users (Netflix comparison)
- **$14.99** = Premium ceiling for wellness apps
- **Lifetime** = 1.5-2 years of annual (LTV optimization)

---

## 3. Feature Gating

### Free Tier Limitations

| Feature | Free Limit | Premium (Seeker+) |
|---------|------------|-------------------|
| Tarot readings | 3/day | Unlimited |
| Journal entries | 10 total | Unlimited |
| Oracle chats | 5/day | Unlimited (Adept+) |
| Card spreads | Single, 3-card only | All spreads |
| CBT/DBT tools | Preview only | Full access |
| Mindfulness | 2 exercises | Full library |
| Ad-free | ❌ (shows ads) | ✅ |
| Cloud sync | ❌ | ✅ |
| Voice narration | ❌ | ✅ (Mystic only) |
| Premium cosmetics | ❌ | ✅ |

### Monetization Flows

```
Free User Journey:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ First Open  │────▶│ Get 1000 💎 │────▶│ Explore App │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                          ┌────────────────────┴────────────────────┐
                          ▼                                         ▼
                  ┌───────────────┐                       ┌───────────────┐
                  │ Hit Daily Cap │                       │ Want Premium  │
                  │ (3 readings)  │                       │   Feature     │
                  └───────┬───────┘                       └───────┬───────┘
                          │                                       │
                          ▼                                       ▼
                  ┌───────────────┐                       ┌───────────────┐
                  │ Upsell Modal: │                       │ Upsell Modal: │
                  │ Subscribe for │                       │ Get Seeker    │
                  │ Unlimited     │                       │ for $4.99/mo  │
                  └───────────────┘                       └───────────────┘
```

---

## 4. Shop Pricing (Veil Shards)

### Card Backs

| Rarity | Price (💎) | Availability |
|--------|-----------|--------------|
| Common | Free | Default |
| Rare | 300 💎 | Shop |
| Epic | 750 💎 | Shop |
| Legendary | 1,500 💎 | Shop + Achievement |

### Flip Animations

| Rarity | Price (💎) | Availability |
|--------|-----------|--------------|
| Common | Free | Default |
| Rare | 200 💎 | Shop |
| Epic | 500 💎 | Shop |
| Legendary | 1,000 💎 | Achievement only |

### Themes

| Theme | Price (💎) |
|-------|-----------|
| Midnight Ritual | 1,500 💎 |
| Aurora Borealis | 1,500 💎 |
| Blood Moon | 2,000 💎 |
| Celestial Garden | 2,500 💎 |

### Oracle Consumables

| Item | Price (💎) | Effect |
|------|-----------|--------|
| Oracle Token | 50 💎 | 1 extra Oracle chat |
| Oracle Bundle (10) | 400 💎 | 10 Oracle chats |
| Deeper Insight | 100 💎 | Enhanced interpretation |
| Card Reveal | 75 💎 | Reveal hidden card aspects |

---

## 5. Payment Gateway Strategy

### Primary: Stripe

**Why Stripe?**
- Industry standard, trusted
- Web-first (no app store fees!)
- Supports 135+ currencies
- Built-in fraud detection
- PCI DSS compliant
- Connect for marketplace features
- Billing portal for subscriptions

**Supported Methods via Stripe:**
- Credit/Debit cards (Visa, Mastercard, Amex, Discover)
- Google Pay ✅
- Link (Stripe's one-click checkout)
- Bank transfers (ACH)
- Buy Now Pay Later (Klarna, Afterpay)

### Secondary: Crypto (via Coinbase Commerce or BTCPay)

**Supported Cryptocurrencies:**
- Bitcoin (BTC) ✅
- Ethereum (ETH) ✅
- Monero (XMR) ✅ (privacy-focused users love this)
- USDC/USDT (stablecoins for exact pricing)
- Litecoin (LTC)

**Implementation:**
```javascript
// Coinbase Commerce integration
const checkout = await CoinbaseCommerce.createCharge({
  name: "2,600 Veil Shards",
  description: "VeilPath Premium Currency",
  pricing_type: "fixed_price",
  local_price: {
    amount: "19.99",
    currency: "USD"
  },
  metadata: {
    user_id: userId,
    product_id: "veil_shards_large"
  }
});
```

### Alternative: PayPal

**Supported Methods:**
- PayPal balance
- Venmo (via PayPal checkout)
- Pay in 4 (installments)

### NOT Supported (Intentionally):

- ❌ Apple Pay (6-month token refresh BS, 30% app store cut)
- ❌ Samsung Pay (minimal web support)
- ❌ App Store IAP (30% cut + restrictions)
- ❌ Google Play IAP (same reasons)

### Cash App / Venmo Direct?

These require business verification and have higher fraud rates. Better to use PayPal which supports Venmo natively.

---

## 6. RevenueCat vs Alternatives

### RevenueCat Pros:
- ✅ Unified subscription management across platforms
- ✅ Webhook integrations
- ✅ Customer portal
- ✅ Analytics dashboard
- ✅ Free up to $2,500/mo MTR
- ✅ Handles receipt validation

### RevenueCat Cons:
- ❌ Primarily designed for app stores (less useful for web-only)
- ❌ 1% fee after $2,500/mo
- ❌ Adds complexity for web-first

### Recommendation for VeilPath (Web-First):

**Use Stripe directly + Stripe Billing** instead of RevenueCat because:
1. You're avoiding app stores entirely
2. Stripe Billing handles subscriptions natively
3. No additional fees (just Stripe's 2.9% + $0.30)
4. Better web integration
5. Built-in customer portal

**If you later add native apps**: Then add RevenueCat to handle app store IAP alongside Stripe for web.

---

## 7. Accounting Integrations

### Stripe + Accounting Software

| Integration | Cost | Features |
|-------------|------|----------|
| **Stripe + QuickBooks** | Free connector | Auto-sync transactions, invoices, revenue |
| **Stripe + Xero** | Free connector | Same as above |
| **Stripe + FreshBooks** | Free connector | Simpler, for freelancers |

### What Gets Synced:
- All payments/refunds
- Subscription revenue (MRR)
- Currency breakdowns
- Tax collected (if applicable)
- Payout reconciliation

### Tax Compliance

**Stripe Tax**: Automatically calculates and collects sales tax/VAT based on customer location.
- Enable in Stripe Dashboard
- Handles US state sales tax
- Handles EU VAT
- Handles international taxes

**Required for compliance:**
- Tax ID collection for B2B
- Invoice generation
- Revenue recognition (ASC 606)

---

## 8. Currency Sinks (Preventing Inflation)

To maintain Veil Shards value, we need ways to remove currency from circulation:

### Active Sinks (Player Choice)

| Sink | Cost | Description |
|------|------|-------------|
| Card backs | 300-1,500 💎 | Cosmetic purchases |
| Animations | 200-1,000 💎 | Cosmetic purchases |
| Themes | 1,500-2,500 💎 | Cosmetic purchases |
| Oracle tokens | 50-400 💎 | Consumable |
| Special spreads | 100-200 💎 | One-time access |
| Speed-up timers | 25 💎 | Skip cooldowns |

### Passive Sinks

| Sink | Description |
|------|-------------|
| Limited-time events | Exclusive items that disappear |
| Seasonal cosmetics | Only available during season |
| Achievement flex | Expensive "whale" items |

### Balance Target

- **Earning rate (free)**: ~50 💎/week from first-time bonuses, achievements
- **Earning rate (subscriber)**: ~100 💎/week from bonuses, events
- **Spending pressure**: New cosmetics monthly at 300-1,500 💎
- **Target**: Free users can buy rare items after 1-2 months of play

---

## 9. First-Time Bonus Economy Impact

### Bonus Breakdown

| Bonus | Amount | When |
|-------|--------|------|
| First shop visit | 1,000 💎 | Immediate hook |
| First reading | 250 💎 | First session |
| First interpretation | 150 💎 | First session |
| First questions | 100 💎 | First session |
| First journal | 200 💎 | First session |
| First 2-day streak | 300 💎 | Day 2 |
| MBTI completion | 500 💎 | Onboarding |
| MBTI selection | 100 💎 | Onboarding |
| **TOTAL** | **2,600 💎** | ~Day 2 |

### What Can They Buy?

With 2,600 💎, a new user can:
- 1 Legendary card back (1,500 💎) + 1 Epic animation (500 💎) + leftovers
- OR 2 Epic card backs (1,500 💎) + 1 Rare animation (200 💎)
- OR 8 Rare card backs (2,400 💎)
- OR 52 Oracle tokens (2,600 💎)

**Psychology**: "I got so much free stuff, this app is generous!" → converts to subscriber or IAP buyer later

---

## 10. Revenue Projections

### Assumptions (Conservative)

- 10,000 Monthly Active Users (MAU)
- 3% convert to paid subscriber
- 1% purchase IAP
- Average subscriber: Seeker ($4.99/mo)
- Average IAP: Medium package ($9.99)

### Monthly Revenue

| Source | Calculation | Revenue |
|--------|-------------|---------|
| Subscriptions | 300 users × $4.99 | $1,497 |
| IAP | 100 users × $9.99 | $999 |
| Ads (free users) | 9,600 users × $0.50 eCPM | ~$150 |
| **Total** | | **$2,646/mo** |

### Year 1 Target

| Metric | Target |
|--------|--------|
| MAU | 50,000 |
| Paid conversion | 5% |
| MRR | $12,500 |
| ARR | $150,000 |

---

## 11. Implementation Priority

### Phase 1 (Launch)
1. ✅ Welcome bonus system
2. ⏳ Stripe integration (subscriptions)
3. ⏳ Feature gating (free vs premium)
4. ⏳ Basic shop with Veil Shards

### Phase 2 (Month 2-3)
1. Veil Shards IAP packages
2. Crypto payments (Coinbase Commerce)
3. More cosmetics in shop
4. Subscription upgrade prompts

### Phase 3 (Month 4-6)
1. PayPal/Venmo integration
2. Oracle consumables
3. Seasonal events
4. Referral program

---

## Sources

- [Microtransactions: How Freemium Apps and Games Monetize in 2025](https://tyrads.com/microtransaction/)
- [Mobile Game Monetization Strategies 2024](https://lancaric.me/monetization-mobile-games/)
- [Mental Health App Monetization Strategies](https://sda.company/blog/category/mental-health/mental-health-app-monetization)
- [How to Create a Balanced Mobile Game Economy](https://www.blog.udonis.co/mobile-marketing/mobile-games/balanced-mobile-game-economy)
- [Mobile Game In-App Purchases Fraud Protection](https://irdeto.com/blog/mobile-game-in-app-purchases-how-to-protect-them-from-fraud)
- [Unity Game Economy Design Guide](https://unity.com/how-to/what-game-economy-guide-part-1)
