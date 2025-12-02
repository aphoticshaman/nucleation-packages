/**
 * PiiScrubber
 *
 * Removes personally identifiable information from data.
 * Helps with GDPR/CCPA compliance.
 */

export interface ScrubResult {
  scrubbed: string;
  redactions: Redaction[];
}

export interface Redaction {
  type: PiiType;
  start: number;
  end: number;
  replacement: string;
}

export type PiiType =
  | 'email'
  | 'phone'
  | 'ssn'
  | 'credit_card'
  | 'ip_address'
  | 'name'
  | 'address'
  | 'date_of_birth';

export interface ScrubberConfig {
  types?: PiiType[];
  replacement?: string;
  preserveLength?: boolean;
}

// Patterns for PII detection
// Note: Patterns reviewed for ReDoS safety - linear time complexity patterns only
const PII_PATTERNS: Record<PiiType, RegExp> = {
  // Email: standard pattern - no nested quantifiers
  email: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g,
  // Phone: 10-digit US phone numbers - linear time, reviewed safe
  // eslint-disable-next-line security/detect-unsafe-regex
  phone: /\b\d{3}[- ]?\d{3}[- ]?\d{4}\b/g,
  // SSN: specific format - no nested quantifiers
  ssn: /\b\d{3}[- ]?\d{2}[- ]?\d{4}\b/g,
  // Credit card: 16-digit card numbers - linear time
  credit_card: /\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b/g,
  // IP address: explicit octets
  ip_address: /\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g,
  // Name: simple two-word capitalized pattern
  name: /\b[A-Z][a-z]+\s+[A-Z][a-z]+\b/g,
  // Address: street addresses - limited pattern, reviewed safe
  // eslint-disable-next-line security/detect-unsafe-regex
  address: /\b\d+\s+\w+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b/gi,
  // Date of birth: MM/DD/YYYY format
  date_of_birth: /\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b/g,
};

const DEFAULT_TYPES: PiiType[] = ['email', 'phone', 'ssn', 'credit_card', 'ip_address'];

export class PiiScrubber {
  private config: Required<ScrubberConfig>;

  constructor(config: ScrubberConfig = {}) {
    this.config = {
      types: config.types ?? DEFAULT_TYPES,
      replacement: config.replacement ?? '[REDACTED]',
      preserveLength: config.preserveLength ?? false,
    };
  }

  /**
   * Scrub PII from text
   */
  scrub(text: string): ScrubResult {
    const redactions: Redaction[] = [];
    let scrubbed = text;
    let offset = 0;

    for (const type of this.config.types) {
      const pattern = PII_PATTERNS[type];
      if (!pattern) continue;

      // Reset regex state
      pattern.lastIndex = 0;

      let match;
      while ((match = pattern.exec(text)) !== null) {
        const replacement = this.config.preserveLength
          ? '*'.repeat(match[0].length)
          : `[${type.toUpperCase()}]`;

        redactions.push({
          type,
          start: match.index,
          end: match.index + match[0].length,
          replacement,
        });
      }
    }

    // Sort by position (descending) to replace from end to start
    redactions.sort((a, b) => b.start - a.start);

    for (const redaction of redactions) {
      scrubbed =
        scrubbed.slice(0, redaction.start + offset) +
        redaction.replacement +
        scrubbed.slice(redaction.end + offset);

      // Adjust offset for length change
      offset += redaction.replacement.length - (redaction.end - redaction.start);
    }

    // Re-sort by position (ascending) for return
    redactions.sort((a, b) => a.start - b.start);

    return { scrubbed, redactions };
  }

  /**
   * Check if text contains PII
   */
  containsPii(text: string): boolean {
    for (const type of this.config.types) {
      const pattern = PII_PATTERNS[type];
      if (!pattern) continue;

      pattern.lastIndex = 0;
      if (pattern.test(text)) {
        return true;
      }
    }

    return false;
  }

  /**
   * Detect PII types present in text
   */
  detectPiiTypes(text: string): PiiType[] {
    const found: PiiType[] = [];

    for (const type of this.config.types) {
      const pattern = PII_PATTERNS[type];
      if (!pattern) continue;

      pattern.lastIndex = 0;
      if (pattern.test(text)) {
        found.push(type);
      }
    }

    return found;
  }

  /**
   * Scrub PII from object (recursive)
   */
  scrubObject<T extends Record<string, unknown>>(obj: T): T {
    const result = { ...obj };

    for (const [key, value] of Object.entries(result)) {
      if (typeof value === 'string') {
        (result as Record<string, unknown>)[key] = this.scrub(value).scrubbed;
      } else if (Array.isArray(value)) {
        (result as Record<string, unknown>)[key] = value.map((item) =>
          typeof item === 'string'
            ? this.scrub(item).scrubbed
            : typeof item === 'object' && item !== null
              ? this.scrubObject(item as Record<string, unknown>)
              : item
        );
      } else if (typeof value === 'object' && value !== null) {
        (result as Record<string, unknown>)[key] = this.scrubObject(
          value as Record<string, unknown>
        );
      }
    }

    return result;
  }

  /**
   * Update configuration
   */
  configure(config: Partial<ScrubberConfig>): void {
    if (config.types) this.config.types = config.types;
    if (config.replacement) this.config.replacement = config.replacement;
    if (config.preserveLength !== undefined) this.config.preserveLength = config.preserveLength;
  }
}
