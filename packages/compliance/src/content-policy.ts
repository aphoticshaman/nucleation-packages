/**
 * ContentPolicy
 *
 * Filters content that could be illegal, harmful, or violate ToS.
 * Helps avoid liability from processing problematic content.
 */

export type ContentCategory =
  | 'legal'
  | 'spam'
  | 'adult'
  | 'violence'
  | 'hate_speech'
  | 'illegal_activity'
  | 'financial_fraud'
  | 'market_manipulation'
  | 'insider_trading'
  | 'copyright'
  | 'pii_leak';

export interface ContentCheckResult {
  allowed: boolean;
  category?: ContentCategory;
  reason?: string;
  confidence: number;
}

export interface ContentPolicyConfig {
  blockCategories: ContentCategory[];
  keywords?: Record<ContentCategory, string[]>;
  customFilters?: ContentFilter[];
}

export interface ContentFilter {
  name: string;
  category: ContentCategory;
  check: (content: string) => boolean;
}

// Default blocked categories for financial/legal compliance
const DEFAULT_BLOCKED: ContentCategory[] = [
  'illegal_activity',
  'financial_fraud',
  'market_manipulation',
  'insider_trading',
  'pii_leak',
];

// Keyword patterns for basic detection
const DEFAULT_KEYWORDS: Record<ContentCategory, string[]> = {
  legal: [],
  spam: ['buy now', 'limited offer', 'act fast', 'click here'],
  adult: [], // Would need more sophisticated detection
  violence: ['kill', 'murder', 'attack', 'bomb threat'],
  hate_speech: [], // Would need ML-based detection
  illegal_activity: ['hack into', 'steal', 'counterfeit', 'launder money'],
  financial_fraud: ['guaranteed returns', 'risk-free', 'insider info', 'pump and dump'],
  market_manipulation: ['coordinated buy', 'short squeeze', 'pump this stock'],
  insider_trading: ['non-public information', 'before announcement', 'tip from inside'],
  copyright: [], // Would need content matching
  pii_leak: ['ssn', 'social security', 'credit card number', 'bank account'],
};

export class ContentPolicy {
  private config: ContentPolicyConfig;

  constructor(config: Partial<ContentPolicyConfig> = {}) {
    this.config = {
      blockCategories: config.blockCategories ?? DEFAULT_BLOCKED,
      keywords: { ...DEFAULT_KEYWORDS, ...config.keywords },
      customFilters: config.customFilters ?? [],
    };
  }

  /**
   * Check if content is allowed
   */
  check(content: string): ContentCheckResult {
    const lowerContent = content.toLowerCase();

    // Check keyword patterns
    for (const category of this.config.blockCategories) {
      const keywords = this.config.keywords?.[category] ?? [];

      for (const keyword of keywords) {
        if (lowerContent.includes(keyword.toLowerCase())) {
          return {
            allowed: false,
            category,
            reason: `Contains blocked keyword pattern: "${keyword}"`,
            confidence: 0.7,
          };
        }
      }
    }

    // Check custom filters
    for (const filter of this.config.customFilters ?? []) {
      if (this.config.blockCategories.includes(filter.category)) {
        if (filter.check(content)) {
          return {
            allowed: false,
            category: filter.category,
            reason: `Blocked by custom filter: ${filter.name}`,
            confidence: 0.8,
          };
        }
      }
    }

    return {
      allowed: true,
      confidence: 0.9,
    };
  }

  /**
   * Filter an array of content, returning only allowed items
   */
  filter<T extends { content: string }>(items: T[]): T[] {
    return items.filter((item) => this.check(item.content).allowed);
  }

  /**
   * Check multiple pieces of content
   */
  checkBatch(contents: string[]): ContentCheckResult[] {
    return contents.map((content) => this.check(content));
  }

  /**
   * Add a custom filter
   */
  addFilter(filter: ContentFilter): void {
    this.config.customFilters?.push(filter);
  }

  /**
   * Add keywords for a category
   */
  addKeywords(category: ContentCategory, keywords: string[]): void {
    const existing = this.config.keywords?.[category] ?? [];
    if (this.config.keywords) {
      this.config.keywords[category] = [...existing, ...keywords];
    }
  }

  /**
   * Update blocked categories
   */
  setBlockedCategories(categories: ContentCategory[]): void {
    this.config.blockCategories = categories;
  }

  /**
   * Get current configuration
   */
  getConfig(): ContentPolicyConfig {
    return { ...this.config };
  }
}

/**
 * Pre-built filter for SEC compliance
 * Blocks content that could constitute securities fraud
 */
export const SEC_COMPLIANCE_FILTER: ContentFilter = {
  name: 'sec-compliance',
  category: 'market_manipulation',
  check: (content: string): boolean => {
    const patterns = [
      /buy\s+now\s+before\s+it\s+(moons?|rockets?|explodes?)/i,
      /guaranteed\s+\d+%\s+returns?/i,
      /can't\s+lose|no\s+risk/i,
      /inside(r)?\s+(info|information|tip)/i,
      /coordinated\s+(buy|sell|pump|dump)/i,
    ];

    return patterns.some((p) => p.test(content));
  },
};

/**
 * Pre-built filter for financial advice disclaimer
 */
export const NOT_FINANCIAL_ADVICE_FILTER: ContentFilter = {
  name: 'financial-advice-disclaimer',
  category: 'financial_fraud',
  check: (content: string): boolean => {
    // Flag content that gives specific financial advice without disclaimers
    const advicePatterns = [
      /you\s+should\s+(buy|sell|invest)/i,
      /definitely\s+(buy|sell|invest)/i,
      /trust\s+me.+(buy|sell|invest)/i,
    ];

    const hasAdvice = advicePatterns.some((p) => p.test(content));
    const hasDisclaimer = /not\s+financial\s+advice|do\s+your\s+own\s+research|dyor/i.test(content);

    return hasAdvice && !hasDisclaimer;
  },
};
