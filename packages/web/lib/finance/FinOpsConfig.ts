/**
 * FINANCIAL OPERATIONS (FinOps) CONFIGURATION
 *
 * Automated accounting/tax for Florida LLC with variable SaaS costs:
 *
 * YOUR BILLING SOURCES:
 * - Stripe: Customer payments + fees (~2.9% + $0.30)
 * - Vercel: Hosting ($20-200/mo based on usage)
 * - Supabase: Database ($25-100/mo)
 * - Anthropic: Claude API ($200+/mo variable)
 * - Upstash: Redis ($5-20/mo)
 * - Modal: ML inference ($30+/mo)
 * - Resend: Email ($0-20/mo)
 * - GitHub: Code hosting ($0-4/mo)
 * - Domain/DNS: (~$50/yr)
 *
 * RECOMMENDED TOOLS:
 * - Mercury: Business banking (free, built for startups)
 * - Pilot: Bookkeeping + tax ($200-400/mo)
 * - Bench: Bookkeeping ($250-500/mo)
 * - Finta: Financial data sync (free tier)
 * - QuickBooks: DIY accounting ($30/mo)
 *
 * FLORIDA LLC REQUIREMENTS:
 * - Annual Report: $138.75 (due May 1)
 * - Registered Agent: $100-150/yr
 * - No state income tax
 * - Self-employment tax: 15.3% (Social Security + Medicare)
 * - Federal income tax: Based on bracket
 */

// ============================================
// BILLING SOURCES
// ============================================

export interface BillingSource {
  id: string;
  name: string;
  type: 'revenue' | 'expense' | 'both';
  category: string;
  estimatedMonthly: string;
  variability: 'fixed' | 'usage' | 'tiered';
  taxDeductible: boolean;
  integrations: string[];
  notes: string;
}

export const BILLING_SOURCES: BillingSource[] = [
  // REVENUE
  {
    id: 'stripe',
    name: 'Stripe',
    type: 'both',
    category: 'Payment Processing',
    estimatedMonthly: 'Revenue - 2.9% + $0.30 per txn',
    variability: 'usage',
    taxDeductible: true,
    integrations: ['QuickBooks', 'Xero', 'Mercury', 'Pilot'],
    notes: 'Primary revenue source. Fees are deductible.',
  },

  // INFRASTRUCTURE
  {
    id: 'vercel',
    name: 'Vercel',
    type: 'expense',
    category: 'Cloud Infrastructure',
    estimatedMonthly: '$20-200',
    variability: 'tiered',
    taxDeductible: true,
    integrations: [],
    notes: 'Pro plan $20/mo base, scales with bandwidth/functions',
  },
  {
    id: 'supabase',
    name: 'Supabase',
    type: 'expense',
    category: 'Cloud Infrastructure',
    estimatedMonthly: '$25-100',
    variability: 'tiered',
    taxDeductible: true,
    integrations: [],
    notes: 'Pro plan $25/mo base, scales with DB size/bandwidth',
  },
  {
    id: 'upstash',
    name: 'Upstash',
    type: 'expense',
    category: 'Cloud Infrastructure',
    estimatedMonthly: '$5-20',
    variability: 'usage',
    taxDeductible: true,
    integrations: [],
    notes: 'Redis caching, billed through Vercel',
  },

  // AI/ML
  {
    id: 'anthropic-api',
    name: 'Anthropic (API)',
    type: 'expense',
    category: 'AI Services',
    estimatedMonthly: '$200-500+',
    variability: 'usage',
    taxDeductible: true,
    integrations: [],
    notes: 'Claude API usage for intelligence features',
  },
  {
    id: 'anthropic-claude-code',
    name: 'Anthropic (Claude Code)',
    type: 'expense',
    category: 'Development Tools',
    estimatedMonthly: '$200+',
    variability: 'usage',
    taxDeductible: true,
    integrations: [],
    notes: 'Claude Code subscription for development',
  },
  {
    id: 'modal',
    name: 'Modal',
    type: 'expense',
    category: 'AI Services',
    estimatedMonthly: '$30-100',
    variability: 'usage',
    taxDeductible: true,
    integrations: [],
    notes: 'ML inference, $30 free credits then usage',
  },
  {
    id: 'huggingface',
    name: 'HuggingFace',
    type: 'expense',
    category: 'AI Services',
    estimatedMonthly: '$0-50',
    variability: 'usage',
    taxDeductible: true,
    integrations: [],
    notes: 'Model hosting, inference endpoints',
  },

  // COMMUNICATION
  {
    id: 'resend',
    name: 'Resend',
    type: 'expense',
    category: 'Communication',
    estimatedMonthly: '$0-20',
    variability: 'tiered',
    taxDeductible: true,
    integrations: [],
    notes: 'Transactional email, free tier generous',
  },

  // DEVELOPMENT
  {
    id: 'github',
    name: 'GitHub',
    type: 'expense',
    category: 'Development Tools',
    estimatedMonthly: '$0-4',
    variability: 'fixed',
    taxDeductible: true,
    integrations: ['QuickBooks', 'Xero'],
    notes: 'Free for public, $4/mo for private features',
  },

  // DOMAINS
  {
    id: 'domains',
    name: 'Domains/DNS',
    type: 'expense',
    category: 'Infrastructure',
    estimatedMonthly: '~$4 (annual)',
    variability: 'fixed',
    taxDeductible: true,
    integrations: [],
    notes: '~$50/yr for domains, amortize monthly',
  },

  // COMPLIANCE (future)
  {
    id: 'vanta',
    name: 'Vanta (future)',
    type: 'expense',
    category: 'Compliance',
    estimatedMonthly: '$800-2000',
    variability: 'fixed',
    taxDeductible: true,
    integrations: ['QuickBooks', 'Xero'],
    notes: 'SOC 2 compliance platform - when ready',
  },
];

// ============================================
// MONTHLY COST ESTIMATE
// ============================================

export const MONTHLY_ESTIMATE = {
  // Core infrastructure
  infrastructure: {
    vercel: 50, // Average
    supabase: 50, // Average
    upstash: 10,
    domains: 4,
    subtotal: 114,
  },

  // AI/ML
  ai: {
    anthropic_api: 300, // Average
    anthropic_claude_code: 200,
    modal: 50,
    huggingface: 20,
    subtotal: 570,
  },

  // Communication
  communication: {
    resend: 10,
    subtotal: 10,
  },

  // Development
  development: {
    github: 4,
    subtotal: 4,
  },

  // Total before revenue
  totalExpenses: 698,

  // Stripe fees (estimated on revenue)
  // If $5000/mo revenue: ~$175 in fees
  stripeFees: '2.9% + $0.30 per transaction',

  // FLORIDA LLC OVERHEAD
  floridaLLC: {
    annualReport: 138.75 / 12, // ~$12/mo amortized
    registeredAgent: 125 / 12, // ~$10/mo amortized
    subtotal: 22,
  },

  // Potential future costs
  future: {
    soc2_compliance: 1000, // When ready
    insurance: 100, // E&O, Cyber liability
    legal: 200, // Contract reviews
  },
};

// ============================================
// ACCOUNTING AUTOMATION
// ============================================

export interface AccountingTool {
  name: string;
  type: 'bookkeeping' | 'banking' | 'tax' | 'sync';
  pricing: string;
  features: string[];
  stripeIntegration: boolean;
  recommended: boolean;
  forStage: 'bootstrap' | 'growth' | 'scale';
}

export const ACCOUNTING_TOOLS: AccountingTool[] = [
  // BANKING
  {
    name: 'Mercury',
    type: 'banking',
    pricing: 'Free',
    features: [
      'Business checking/savings',
      'Virtual cards',
      'ACH/Wire free',
      'Startup-friendly',
      'API access',
      'Accounting integrations',
    ],
    stripeIntegration: true,
    recommended: true,
    forStage: 'bootstrap',
  },
  {
    name: 'Relay',
    type: 'banking',
    pricing: 'Free',
    features: [
      'Business checking',
      'Profit First compatible',
      'Multiple accounts',
      'No minimums',
    ],
    stripeIntegration: true,
    recommended: false,
    forStage: 'bootstrap',
  },

  // BOOKKEEPING
  {
    name: 'Pilot',
    type: 'bookkeeping',
    pricing: '$200-400/mo',
    features: [
      'Full-service bookkeeping',
      'Tax prep included',
      'CFO services available',
      'Startup-focused',
      'SaaS metrics',
      'R&D tax credits',
    ],
    stripeIntegration: true,
    recommended: true,
    forStage: 'growth',
  },
  {
    name: 'Bench',
    type: 'bookkeeping',
    pricing: '$250-500/mo',
    features: [
      'Monthly bookkeeping',
      'Year-end tax prep',
      'Dedicated bookkeeper',
      'Accrual/cash basis',
    ],
    stripeIntegration: true,
    recommended: false,
    forStage: 'growth',
  },
  {
    name: 'QuickBooks Online',
    type: 'bookkeeping',
    pricing: '$30-200/mo',
    features: [
      'DIY accounting',
      'Invoicing',
      'Expense tracking',
      'Reports',
      'CPA access',
    ],
    stripeIntegration: true,
    recommended: true,
    forStage: 'bootstrap',
  },

  // DATA SYNC
  {
    name: 'Finta',
    type: 'sync',
    pricing: 'Free tier',
    features: [
      'Stripe → Sheets sync',
      'Financial dashboards',
      'Automated reports',
      'Multiple integrations',
    ],
    stripeIntegration: true,
    recommended: true,
    forStage: 'bootstrap',
  },
  {
    name: 'Ramp',
    type: 'banking',
    pricing: 'Free',
    features: [
      'Corporate cards',
      '1.5% cashback',
      'Expense management',
      'Bill pay',
      'Accounting sync',
    ],
    stripeIntegration: false,
    recommended: true,
    forStage: 'growth',
  },

  // TAX
  {
    name: 'TaxJar',
    type: 'tax',
    pricing: '$19-99/mo',
    features: [
      'Sales tax automation',
      'Multi-state filing',
      'Nexus tracking',
      'API for Stripe',
    ],
    stripeIntegration: true,
    recommended: false, // Not needed for B2B SaaS
    forStage: 'scale',
  },
];

// ============================================
// FLORIDA LLC TAX OBLIGATIONS
// ============================================

export const FLORIDA_LLC_TAXES = {
  // NO STATE INCOME TAX!
  stateIncomeTax: 0,

  // Federal self-employment tax
  selfEmploymentTax: {
    rate: 0.153, // 15.3%
    socialSecurity: 0.124, // 12.4% up to wage base
    medicare: 0.029, // 2.9%
    wageBase2024: 168600, // Social Security wage base
    additionalMedicare: 0.009, // Above $200k single
  },

  // Federal income tax brackets 2024 (single)
  federalBrackets: [
    { min: 0, max: 11600, rate: 0.10 },
    { min: 11600, max: 47150, rate: 0.12 },
    { min: 47150, max: 100525, rate: 0.22 },
    { min: 100525, max: 191950, rate: 0.24 },
    { min: 191950, max: 243725, rate: 0.32 },
    { min: 243725, max: 609350, rate: 0.35 },
    { min: 609350, max: Infinity, rate: 0.37 },
  ],

  // Quarterly estimated taxes
  quarterlyDates: [
    { quarter: 'Q1', due: 'April 15' },
    { quarter: 'Q2', due: 'June 15' },
    { quarter: 'Q3', due: 'September 15' },
    { quarter: 'Q4', due: 'January 15 (next year)' },
  ],

  // Florida requirements
  floridaRequirements: {
    annualReport: {
      due: 'May 1',
      fee: 138.75,
      lateFeee: 400,
      filingUrl: 'https://dos.myflorida.com/sunbiz/',
    },
    registeredAgent: {
      required: true,
      estimatedCost: '100-150/yr',
    },
    businessTax: {
      required: false, // Most counties don't require for online business
      note: 'Check county requirements',
    },
  },
};

// ============================================
// STRIPE INTEGRATION
// ============================================

export const STRIPE_CONFIG = {
  // Webhook events to track for accounting
  accountingEvents: [
    'invoice.paid',
    'invoice.payment_failed',
    'charge.succeeded',
    'charge.refunded',
    'customer.subscription.created',
    'customer.subscription.updated',
    'customer.subscription.deleted',
    'payout.paid',
  ],

  // Revenue recognition
  revenueRecognition: {
    // For SaaS, recognize monthly as service delivered
    method: 'accrual',
    subscriptionRecognition: 'monthly',
    // Deferred revenue for annual plans
    deferredRevenue: true,
  },

  // Fee tracking
  fees: {
    standard: { percentage: 2.9, fixed: 0.30 },
    international: { percentage: 3.9, fixed: 0.30 },
    currency_conversion: { percentage: 1.0, fixed: 0 },
  },

  // Reporting
  reports: [
    'balance_transactions',
    'payouts',
    'charges',
    'refunds',
    'disputes',
    'subscriptions',
  ],
};

// ============================================
// AUTOMATED FINANCIAL WORKFLOWS
// ============================================

export const FINANCIAL_WORKFLOWS = {
  // Daily
  daily: [],

  // Weekly
  weekly: [
    'Review Stripe dashboard for anomalies',
    'Check bank balance in Mercury',
    'Review any failed payments',
  ],

  // Monthly
  monthly: [
    'Reconcile Stripe payouts with bank deposits',
    'Categorize expenses in QuickBooks/Pilot',
    'Review MRR/ARR metrics',
    'Check usage-based costs (Anthropic, Vercel)',
    'Calculate burn rate',
  ],

  // Quarterly
  quarterly: [
    'Pay estimated federal taxes',
    'Review P&L statement',
    'Update financial projections',
    'Review pricing strategy',
    'Check if approaching next tier on any service',
  ],

  // Annually
  annually: [
    'File Florida Annual Report (by May 1)',
    'File federal taxes (or extension)',
    'Review and renew registered agent',
    'Update accounting integrations',
    'Review subscription costs for optimization',
    'Plan for major expenses (compliance, etc.)',
  ],
};

// ============================================
// CHART OF ACCOUNTS (QuickBooks compatible)
// ============================================

export const CHART_OF_ACCOUNTS = {
  // Revenue
  revenue: {
    '4000': 'Subscription Revenue',
    '4010': 'Annual Subscription Revenue',
    '4020': 'Monthly Subscription Revenue',
    '4100': 'Professional Services',
    '4200': 'API Revenue',
  },

  // Cost of Revenue
  cogs: {
    '5000': 'Cost of Revenue',
    '5010': 'Cloud Infrastructure (Vercel/Supabase)',
    '5020': 'AI/ML Services (Anthropic/Modal)',
    '5030': 'Payment Processing (Stripe fees)',
    '5040': 'Email Services (Resend)',
  },

  // Operating Expenses
  opex: {
    '6000': 'Operating Expenses',
    '6010': 'Software & Tools',
    '6020': 'Development Tools',
    '6030': 'Compliance & Security',
    '6040': 'Legal & Professional',
    '6050': 'Marketing',
    '6060': 'Travel',
    '6070': 'Office & Equipment',
    '6080': 'Insurance',
  },

  // Administrative
  admin: {
    '7000': 'Administrative',
    '7010': 'Bank Fees',
    '7020': 'Taxes & Licenses',
    '7030': 'Florida LLC Fees',
  },
};

// ============================================
// RECOMMENDED SETUP
// ============================================

export const RECOMMENDED_SETUP = {
  immediate: {
    name: 'Bootstrap Stack',
    totalCost: '$30-50/mo',
    tools: [
      'Mercury (free) - Business banking',
      'QuickBooks Simple Start ($30/mo) - DIY accounting',
      'Finta (free) - Stripe sync',
    ],
    tasks: [
      'Open Mercury business account',
      'Connect Stripe to Mercury for payouts',
      'Set up QuickBooks, connect Mercury',
      'Use Finta to sync Stripe → Google Sheets for metrics',
    ],
  },

  growth: {
    name: 'Growth Stack',
    totalCost: '$200-400/mo',
    tools: [
      'Mercury (free) - Banking',
      'Pilot ($200-400/mo) - Full-service bookkeeping + tax',
      'Ramp (free) - Corporate cards + expense management',
    ],
    tasks: [
      'Migrate to Pilot when revenue > $10k/mo',
      'Get Ramp card for expenses',
      'Automate expense categorization',
    ],
  },

  scale: {
    name: 'Scale Stack',
    totalCost: '$500-1000/mo',
    tools: [
      'Mercury - Banking',
      'Pilot CFO services - Strategic finance',
      'NetSuite - ERP (if needed)',
      'Stripe Revenue Recognition - Automated ASC 606',
    ],
    tasks: [
      'Implement proper revenue recognition',
      'Set up financial controls',
      'Prepare for audit',
    ],
  },
};

// ============================================
// QUICK CALC FUNCTIONS
// ============================================

/**
 * Calculate estimated monthly costs
 */
export function calculateMonthlyCosts(
  revenue: number,
  anthropicUsage: number = 300,
  vercelUsage: number = 50
): {
  grossRevenue: number;
  stripeFees: number;
  operatingCosts: number;
  netIncome: number;
  marginPercent: number;
} {
  // Stripe fees (average transaction ~$50)
  const avgTransactionSize = 50;
  const transactions = revenue / avgTransactionSize;
  const stripeFees = revenue * 0.029 + transactions * 0.3;

  // Operating costs
  const operatingCosts =
    vercelUsage + // Vercel
    50 + // Supabase
    10 + // Upstash
    anthropicUsage + // Anthropic API
    200 + // Claude Code
    50 + // Modal
    10 + // Resend
    4 + // GitHub
    4 + // Domains
    22; // Florida LLC overhead

  const netIncome = revenue - stripeFees - operatingCosts;
  const marginPercent = (netIncome / revenue) * 100;

  return {
    grossRevenue: revenue,
    stripeFees: Math.round(stripeFees * 100) / 100,
    operatingCosts: Math.round(operatingCosts),
    netIncome: Math.round(netIncome * 100) / 100,
    marginPercent: Math.round(marginPercent * 10) / 10,
  };
}

/**
 * Calculate quarterly estimated tax
 */
export function calculateQuarterlyTax(quarterlyIncome: number): {
  selfEmploymentTax: number;
  federalIncomeTax: number;
  totalEstimated: number;
} {
  const annualizedIncome = quarterlyIncome * 4;

  // Self-employment tax
  const seTaxableIncome = annualizedIncome * 0.9235; // 92.35% is taxable
  const seTax = Math.min(seTaxableIncome, FLORIDA_LLC_TAXES.selfEmploymentTax.wageBase2024) *
    FLORIDA_LLC_TAXES.selfEmploymentTax.socialSecurity +
    seTaxableIncome * FLORIDA_LLC_TAXES.selfEmploymentTax.medicare;

  // Federal income tax (simplified)
  let federalTax = 0;
  let remainingIncome = annualizedIncome - seTax * 0.5; // SE deduction

  for (const bracket of FLORIDA_LLC_TAXES.federalBrackets) {
    if (remainingIncome <= 0) break;
    const taxableInBracket = Math.min(
      remainingIncome,
      bracket.max - bracket.min
    );
    federalTax += taxableInBracket * bracket.rate;
    remainingIncome -= taxableInBracket;
  }

  return {
    selfEmploymentTax: Math.round(seTax / 4),
    federalIncomeTax: Math.round(federalTax / 4),
    totalEstimated: Math.round((seTax + federalTax) / 4),
  };
}
