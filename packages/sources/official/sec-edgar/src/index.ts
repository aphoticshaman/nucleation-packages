/**
 * SEC EDGAR Data Source
 *
 * Official US government source for securities filings.
 * Free, public domain, highly reliable.
 *
 * Rate limit: 10 requests per second (be respectful)
 * Attribution: None required (public domain)
 */

export interface EdgarFiling {
  accessionNumber: string;
  filingDate: string;
  reportDate: string;
  form: string;
  companyName: string;
  cik: string;
  fileNumber: string;
  filmNumber: string;
  items: string;
  size: number;
  isXBRL: boolean;
  primaryDocument: string;
  primaryDocDescription: string;
}

export interface EdgarCompanyFacts {
  cik: string;
  entityName: string;
  facts: {
    [taxonomy: string]: {
      [concept: string]: {
        label: string;
        description: string;
        units: {
          [unit: string]: Array<{
            val: number;
            accn: string;
            fy: number;
            fp: string;
            form: string;
            filed: string;
            end: string;
          }>;
        };
      };
    };
  };
}

export interface EdgarSearchParams {
  query?: string;
  cik?: string;
  ticker?: string;
  formType?: string;
  dateFrom?: string;
  dateTo?: string;
}

const BASE_URL = 'https://data.sec.gov';
const SUBMISSIONS_URL = 'https://data.sec.gov/submissions';
const _FULL_TEXT_URL = 'https://efts.sec.gov/LATEST/search-index';

// SEC requires a User-Agent header
const USER_AGENT = 'LatticeForge/1.0 (Crystalline Labs LLC)';

export class SecEdgarSource {
  private requestCount = 0;
  private lastRequestTime = 0;
  private minRequestInterval = 100; // 10 req/sec = 100ms between requests

  /**
   * Get recent filings for a company by CIK
   */
  async getFilings(cik: string): Promise<EdgarFiling[]> {
    await this.respectRateLimit();

    const paddedCik = cik.padStart(10, '0');
    const url = `${SUBMISSIONS_URL}/CIK${paddedCik}.json`;

    const response = await fetch(url, {
      headers: {
        'User-Agent': USER_AGENT,
        Accept: 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`SEC EDGAR error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json() as {
      filings: {
        recent: {
          accessionNumber: string[];
          filingDate: string[];
          reportDate: string[];
          form: string[];
          fileNumber: string[];
          filmNumber: string[];
          items: string[];
          size: number[];
          isXBRL: number[];
          primaryDocument: string[];
          primaryDocDescription: string[];
        };
      };
      name: string;
      cik: string;
    };

    const recent = data.filings.recent;
    const filings: EdgarFiling[] = [];

    for (let i = 0; i < recent.accessionNumber.length; i++) {
      filings.push({
        accessionNumber: recent.accessionNumber[i],
        filingDate: recent.filingDate[i],
        reportDate: recent.reportDate[i],
        form: recent.form[i],
        companyName: data.name,
        cik: data.cik,
        fileNumber: recent.fileNumber[i],
        filmNumber: recent.filmNumber[i],
        items: recent.items[i],
        size: recent.size[i],
        isXBRL: recent.isXBRL[i] === 1,
        primaryDocument: recent.primaryDocument[i],
        primaryDocDescription: recent.primaryDocDescription[i],
      });
    }

    return filings;
  }

  /**
   * Get company facts (XBRL data)
   */
  async getCompanyFacts(cik: string): Promise<EdgarCompanyFacts> {
    await this.respectRateLimit();

    const paddedCik = cik.padStart(10, '0');
    const url = `${BASE_URL}/api/xbrl/companyfacts/CIK${paddedCik}.json`;

    const response = await fetch(url, {
      headers: {
        'User-Agent': USER_AGENT,
        Accept: 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`SEC EDGAR error: ${response.status} ${response.statusText}`);
    }

    return response.json() as Promise<EdgarCompanyFacts>;
  }

  /**
   * Get filings by form type (e.g., 10-K, 10-Q, 8-K)
   */
  async getFilingsByForm(cik: string, formType: string): Promise<EdgarFiling[]> {
    const allFilings = await this.getFilings(cik);
    return allFilings.filter((f) => f.form === formType);
  }

  /**
   * Get recent 10-K filings (annual reports)
   */
  async getAnnualReports(cik: string): Promise<EdgarFiling[]> {
    return this.getFilingsByForm(cik, '10-K');
  }

  /**
   * Get recent 10-Q filings (quarterly reports)
   */
  async getQuarterlyReports(cik: string): Promise<EdgarFiling[]> {
    return this.getFilingsByForm(cik, '10-Q');
  }

  /**
   * Get recent 8-K filings (current reports / material events)
   */
  async getCurrentReports(cik: string): Promise<EdgarFiling[]> {
    return this.getFilingsByForm(cik, '8-K');
  }

  /**
   * Convert filings to numeric signal (filing frequency)
   */
  toSignal(filings: EdgarFiling[], daysBucket = 7): number[] {
    if (filings.length === 0) return [];

    // Sort by date
    const sorted = [...filings].sort(
      (a, b) => new Date(a.filingDate).getTime() - new Date(b.filingDate).getTime()
    );

    const start = new Date(sorted[0].filingDate);
    const end = new Date(sorted[sorted.length - 1].filingDate);
    const buckets: number[] = [];

    let current = new Date(start);
    while (current <= end) {
      const bucketEnd = new Date(current);
      bucketEnd.setDate(bucketEnd.getDate() + daysBucket);

      const count = sorted.filter((f) => {
        const date = new Date(f.filingDate);
        return date >= current && date < bucketEnd;
      }).length;

      buckets.push(count);
      current = bucketEnd;
    }

    return buckets;
  }

  /**
   * Get source metadata for compliance
   */
  getSourceMetadata() {
    return {
      sourceId: 'sec-edgar',
      name: 'SEC EDGAR',
      tier: 'official' as const,
      url: 'https://www.sec.gov/edgar',
      license: 'Public Domain',
      attribution: 'Data from SEC EDGAR',
      rateLimit: { requestsPerSecond: 10 },
    };
  }

  private async respectRateLimit(): Promise<void> {
    const now = Date.now();
    const timeSinceLastRequest = now - this.lastRequestTime;

    if (timeSinceLastRequest < this.minRequestInterval) {
      await new Promise((resolve) =>
        setTimeout(resolve, this.minRequestInterval - timeSinceLastRequest)
      );
    }

    this.lastRequestTime = Date.now();
    this.requestCount++;
  }
}

export default SecEdgarSource;
