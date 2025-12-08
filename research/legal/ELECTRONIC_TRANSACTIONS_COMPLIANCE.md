# Electronic Transactions Compliance Documentation

## VeilPath LLC - Due Diligence for Digital Payments

**Last Updated**: November 2024
**Document Version**: 1.0
**Prepared For**: Internal Compliance Records

---

## 1. Executive Summary

This document establishes VeilPath's due diligence procedures for electronic transactions, virtual currency management, and payment processing. It demonstrates compliance with applicable regulations and industry best practices.

**Key Compliance Areas:**
- Payment Card Industry Data Security Standard (PCI DSS)
- Electronic Fund Transfer Act (EFTA)
- Consumer Financial Protection Bureau (CFPB) regulations
- State money transmitter laws
- Anti-Money Laundering (AML) requirements
- Virtual currency regulations

---

## 2. Payment Processing Architecture

### 2.1 Third-Party Payment Processors

VeilPath does NOT directly process payment card data. All payment processing is handled by PCI DSS Level 1 certified third-party processors:

| Processor | Certification | Purpose |
|-----------|---------------|---------|
| Stripe, Inc. | PCI DSS Level 1 | Primary payment processor |
| Coinbase Commerce | FinCEN registered MSB | Cryptocurrency payments |
| PayPal Holdings, Inc. | PCI DSS Level 1 | Alternative payments |

### 2.2 Data Handling

**We DO Store:**
- Transaction IDs (not card numbers)
- User purchase history
- Subscription status
- Virtual currency balances

**We DO NOT Store:**
- Credit/debit card numbers
- CVV/CVC codes
- Bank account numbers
- Payment processor tokens (beyond session)

### 2.3 PCI DSS Scope

VeilPath qualifies for **SAQ-A** (Self-Assessment Questionnaire A) because:
- All payment pages are hosted by Stripe (Checkout/Elements)
- No card data touches our servers
- All transmission is via HTTPS/TLS 1.3
- We redirect to payment processor for all transactions

---

## 3. Virtual Currency (Veil Shards) Compliance

### 3.1 Classification

"Veil Shards" are classified as **closed-loop virtual currency** under FinCEN guidance:

> "Virtual currency that can only be used within a particular virtual environment... and cannot be exchanged for fiat currency or used to purchase real goods/services outside the platform."

**Key Characteristics:**
- ✅ One-way conversion only (USD → Veil Shards)
- ✅ Cannot be cashed out or transferred between users
- ✅ Can only purchase in-app digital goods
- ✅ No real-world value outside VeilPath
- ✅ Non-transferable between accounts

### 3.2 Money Transmitter Analysis

Based on FinCEN's 2019 guidance on convertible virtual currencies, VeilPath is **NOT a money transmitter** because:

1. **No P2P Transfers**: Users cannot send Veil Shards to other users
2. **No Cash Out**: Veil Shards cannot be converted back to USD
3. **Closed Loop**: Currency only usable within our platform
4. **Non-Convertible**: Cannot be exchanged for other virtual currencies

**State-by-State Analysis**: Completed for all 50 states. No state money transmitter license required for closed-loop systems.

### 3.3 Consumer Protections

Even though not legally required, we implement consumer protections:

| Protection | Implementation |
|------------|----------------|
| Clear pricing | All prices shown before purchase |
| Refund policy | 14-day refund for unused currency |
| Terms of Service | Explicit virtual currency terms |
| No expiration | Veil Shards never expire |
| Balance visibility | Always displayed in-app |

---

## 4. Anti-Money Laundering (AML) Measures

### 4.1 Risk Assessment

**Overall AML Risk: LOW**

| Factor | Assessment |
|--------|------------|
| Transaction size | Max $99.99 single purchase |
| User base | Primarily US wellness consumers |
| Currency type | Closed-loop, non-transferable |
| Anonymity | Account required for purchase |
| Cross-border | Minimal (web platform) |

### 4.2 Monitoring Procedures

Despite low risk, we implement monitoring:

**Transaction Monitoring:**
- Flag purchases > $200/day from single account
- Flag > 5 purchases in 24 hours
- Flag mismatched billing/IP geolocation
- Flag rapid refund/repurchase patterns

**Account Monitoring:**
- Device fingerprinting to detect multi-accounting
- IP velocity checks
- Email domain analysis
- Signup pattern analysis

### 4.3 Suspicious Activity Response

If suspicious activity detected:

1. **Review**: Manual review within 24 hours
2. **Hold**: Temporarily suspend transactions if needed
3. **Document**: Log all suspicious activity
4. **Report**: File SAR if threshold met (>$5,000 suspicious)
5. **Terminate**: Close account if fraud confirmed

---

## 5. EFTA / Regulation E Compliance

### 5.1 Applicability

The Electronic Fund Transfer Act primarily applies to:
- Bank accounts
- Debit cards linked to accounts
- Prepaid cards

**VeilPath Veil Shards are NOT covered by EFTA** because:
- Not a prepaid access device
- Not redeemable for cash
- Not usable outside platform

### 5.2 Voluntary Compliance

Despite exemption, we voluntarily implement EFTA-like protections:

| EFTA Requirement | Our Implementation |
|------------------|-------------------|
| Error resolution | Support ticket within 48 hours |
| Unauthorized transfer protection | Device verification, 2FA optional |
| Periodic statements | Transaction history in-app |
| Initial disclosures | Terms of Service before purchase |

---

## 6. Cryptocurrency Payments

### 6.1 Structure

Cryptocurrency payments are processed through **Coinbase Commerce**, a registered Money Services Business (MSB) with FinCEN.

**Flow:**
1. User selects crypto payment
2. Redirected to Coinbase Commerce
3. Coinbase converts crypto → USD
4. USD settled to VeilPath
5. We receive fiat, not crypto

### 6.2 Compliance Benefits

- VeilPath never holds cryptocurrency
- No crypto custody = no crypto licensing needed
- Coinbase handles AML/KYC for crypto side
- We only receive USD

### 6.3 Supported Cryptocurrencies

| Currency | Rationale |
|----------|-----------|
| Bitcoin (BTC) | Most widely held |
| Ethereum (ETH) | Second largest |
| USDC | Stable pricing |
| Litecoin (LTC) | Lower fees |
| Monero (XMR) | Privacy-focused users |

---

## 7. Data Security

### 7.1 Encryption Standards

| Data Type | Encryption |
|-----------|------------|
| Data in transit | TLS 1.3 |
| Data at rest | AES-256 |
| User passwords | Argon2id (via Supabase Auth) |
| API keys | Encrypted environment variables |

### 7.2 Access Controls

| System | Access |
|--------|--------|
| Production database | 2 admins only, MFA required |
| Stripe Dashboard | 2 admins, MFA required |
| Supabase Admin | 2 admins, MFA required |
| User data exports | Admin approval required |

### 7.3 Incident Response Plan

1. **Detection**: Automated alerts via Sentry/monitoring
2. **Containment**: Immediate access revocation
3. **Assessment**: Determine scope within 4 hours
4. **Notification**: Users within 72 hours if PII affected
5. **Remediation**: Fix vulnerability, document learnings
6. **Reporting**: State AG notification if required

---

## 8. Terms of Service Requirements

### 8.1 Required Disclosures

Our Terms of Service must include:

```
VIRTUAL CURRENCY TERMS

1. Veil Shards are virtual currency for use only within VeilPath.
2. Veil Shards have no cash value and cannot be redeemed for cash.
3. Veil Shards are non-transferable between accounts.
4. Unused Veil Shards are non-refundable except as required by law.
5. We reserve the right to modify Veil Shards pricing at any time.
6. Veil Shards do not expire.
7. Upon account termination, unused Veil Shards are forfeited.
```

### 8.2 Refund Policy

```
REFUND POLICY

Subscriptions:
- Cancel anytime, access continues until period end
- No partial refunds for unused subscription time
- Refund within 14 days of purchase if unused

Virtual Currency (Veil Shards):
- Refundable within 14 days if completely unused
- No refund for partially used currency
- Chargebacks may result in account suspension

Digital Goods:
- All sales final once unlocked/used
- Technical issues handled case-by-case
```

---

## 9. Tax Compliance

### 9.1 Sales Tax

Using **Stripe Tax** for automatic calculation and collection:

| Jurisdiction | Tax Rate | Notes |
|--------------|----------|-------|
| US States | Varies (0-10%) | Auto-calculated by Stripe |
| EU/UK | 20-25% VAT | Digital services VAT applies |
| Canada | GST/HST varies | Province-dependent |
| Australia | 10% GST | For digital services |

### 9.2 Tax Reporting

- **1099-K**: Stripe provides if >$600 (2024+ threshold)
- **Income Recognition**: Revenue recognized when currency purchased
- **Deferred Revenue**: None (Veil Shards recognized immediately)

### 9.3 Record Retention

| Record Type | Retention Period |
|-------------|------------------|
| Transaction records | 7 years |
| Tax documents | 7 years |
| User agreements | Account lifetime + 3 years |
| Support tickets | 3 years |
| Security logs | 2 years |

---

## 10. Audit Trail

### 10.1 What We Log

Every transaction includes:

```json
{
  "transaction_id": "txn_abc123",
  "user_id": "user_xyz",
  "timestamp": "2024-11-26T10:30:00Z",
  "amount_usd": 19.99,
  "amount_shards": 2600,
  "payment_method": "stripe",
  "stripe_payment_intent": "pi_xxx",
  "ip_address": "hashed",
  "device_fingerprint": "hashed",
  "action": "purchase"
}
```

### 10.2 Immutable Ledger

All currency transactions stored in `currency_transactions` table:
- Append-only (no updates or deletes)
- Balance before/after recorded
- Full audit trail available
- Exportable for compliance audits

---

## 11. Compliance Checklist

### Pre-Launch

- [x] PCI DSS SAQ-A assessment
- [x] Terms of Service with virtual currency terms
- [x] Privacy Policy (GDPR/CCPA compliant)
- [x] Refund policy published
- [x] Stripe account verified
- [x] Tax collection enabled
- [x] Transaction logging implemented
- [x] Device fingerprinting for fraud

### Ongoing

- [ ] Quarterly transaction review
- [ ] Annual PCI DSS SAQ-A renewal
- [ ] Annual Terms of Service review
- [ ] Suspicious activity monitoring (continuous)
- [ ] User complaint tracking
- [ ] Refund rate monitoring (<2% target)

---

## 12. Contact Information

**Compliance Questions:**
- Email: compliance@veilpath.com
- Response time: 48 business hours

**Data Protection:**
- Email: privacy@veilpath.com
- GDPR/CCPA requests: Same email

**Legal:**
- Registered Agent: [TBD upon incorporation]
- Jurisdiction: Delaware, USA

---

## 13. Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Nov 2024 | Initial document |

---

## Certification

This document has been prepared to demonstrate VeilPath's commitment to compliance with applicable laws and regulations governing electronic transactions and virtual currencies.

**Prepared by**: [Founder Name]
**Date**: [Date]
**Next Review**: [Date + 1 year]

---

*This document is for internal compliance purposes and does not constitute legal advice. Consult with qualified legal counsel for specific compliance questions.*
