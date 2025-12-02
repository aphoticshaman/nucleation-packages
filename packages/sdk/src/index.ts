/**
 * LatticeForge SDK
 *
 * "Your API design is clean but where's the 3-line integration?" - Patrick Collison
 *
 * Here it is:
 *
 * ```typescript
 * import LatticeForge from '@latticeforge/sdk';
 * const lf = new LatticeForge('lf_live_xxx');
 * const signal = await lf.fuse(['SEC', 'FRED', 'sentiment']);
 * ```
 *
 * That's it. Three lines.
 *
 * © 2025 Crystalline Labs LLC
 */

export interface LatticeForgeConfig {
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
}

export interface Signal {
  value: number;
  confidence: number;
  timestamp: Date;
  sources: string[];
  regime: 'risk-on' | 'risk-off' | 'transitional' | 'uncertain';
}

export interface Alert {
  id: string;
  type: 'phase_change' | 'anomaly' | 'cascade' | 'threshold';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  data: Record<string, unknown>;
  timestamp: Date;
}

export interface Usage {
  tokens_used: number;
  tokens_limit: number;
  period_end: Date;
}

/**
 * LatticeForge Client
 *
 * Stupid simple API for signal intelligence.
 */
export class LatticeForge {
  private apiKey: string;
  private baseUrl: string;
  private timeout: number;

  constructor(apiKey: string, options?: Partial<LatticeForgeConfig>) {
    if (!apiKey) {
      throw new Error('API key required. Get one at https://latticeforge.com');
    }

    this.apiKey = apiKey;
    this.baseUrl = options?.baseUrl ?? 'https://api.latticeforge.com/v1';
    this.timeout = options?.timeout ?? 30000;
  }

  // ========================================
  // CORE API - The 3 methods you actually need
  // ========================================

  /**
   * Fuse multiple data sources into a single signal.
   *
   * @example
   * const signal = await lf.fuse(['SEC', 'FRED', 'news']);
   * console.log(signal.value); // 0.72
   * console.log(signal.regime); // 'risk-on'
   */
  async fuse(sources: string[]): Promise<Signal> {
    return this.request<Signal>('POST', '/signals/fuse', { sources });
  }

  /**
   * Detect anomalies and phase changes.
   *
   * @example
   * const alerts = await lf.detect();
   * for (const alert of alerts) {
   *   console.log(`${alert.severity}: ${alert.message}`);
   * }
   */
  async detect(): Promise<Alert[]> {
    return this.request<Alert[]>('GET', '/signals/detect');
  }

  /**
   * Stream real-time signals.
   *
   * @example
   * for await (const signal of lf.stream(['SEC'])) {
   *   console.log(signal.value);
   * }
   */
  async *stream(sources: string[]): AsyncGenerator<Signal> {
    const response = await this.rawRequest('POST', '/signals/stream', { sources });

    if (!response.body) {
      throw new Error('Streaming not supported');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            yield this.parseSignal(data);
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  // ========================================
  // CONVENIENCE METHODS
  // ========================================

  /**
   * Get current regime (risk-on/risk-off).
   *
   * @example
   * const regime = await lf.regime();
   * if (regime === 'risk-off') console.log('Caution advised');
   */
  async regime(): Promise<'risk-on' | 'risk-off' | 'transitional' | 'uncertain'> {
    const signal = await this.fuse(['SEC', 'FRED', 'news']);
    return signal.regime;
  }

  /**
   * Check if phase change is imminent.
   *
   * @example
   * if (await lf.isPhaseChanging()) {
   *   console.log('Regime shift detected!');
   * }
   */
  async isPhaseChanging(): Promise<boolean> {
    const alerts = await this.detect();
    return alerts.some(a => a.type === 'phase_change' && a.severity !== 'low');
  }

  /**
   * Get a specific indicator.
   *
   * @example
   * const vix = await lf.indicator('volatility');
   */
  async indicator(name: string): Promise<number> {
    const response = await this.request<{ value: number }>('GET', `/indicators/${name}`);
    return response.value;
  }

  /**
   * Check API usage.
   *
   * @example
   * const usage = await lf.usage();
   * console.log(`${usage.tokens_used}/${usage.tokens_limit} tokens used`);
   */
  async usage(): Promise<Usage> {
    return this.request<Usage>('GET', '/usage');
  }

  /**
   * Verify API key is valid.
   */
  async verify(): Promise<boolean> {
    try {
      await this.request('GET', '/verify');
      return true;
    } catch {
      return false;
    }
  }

  // ========================================
  // ADVANCED METHODS
  // ========================================

  /**
   * Run custom analysis formula.
   *
   * @example
   * const result = await lf.analyze('phase_transition', {
   *   signals: ['SEC', 'FRED'],
   *   window: 30
   * });
   */
  async analyze(
    formula: string,
    params: Record<string, unknown> = {}
  ): Promise<Record<string, unknown>> {
    return this.request('POST', `/analyze/${formula}`, params);
  }

  /**
   * Get historical data.
   *
   * @example
   * const history = await lf.history('SEC', { days: 30 });
   */
  async history(
    source: string,
    options: { days?: number; startDate?: Date; endDate?: Date } = {}
  ): Promise<Array<{ timestamp: Date; value: number }>> {
    const params = new URLSearchParams();
    if (options.days) params.set('days', String(options.days));
    if (options.startDate) params.set('start', options.startDate.toISOString());
    if (options.endDate) params.set('end', options.endDate.toISOString());

    return this.request('GET', `/history/${source}?${params}`);
  }

  /**
   * Set up webhook for alerts.
   *
   * @example
   * await lf.webhook('https://myapp.com/alerts', ['phase_change', 'anomaly']);
   */
  async webhook(url: string, events: string[]): Promise<{ id: string }> {
    return this.request('POST', '/webhooks', { url, events });
  }

  // ========================================
  // INTERNAL
  // ========================================

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const response = await this.rawRequest(method, path, body);
    const data = await response.json();

    if (!response.ok) {
      throw new LatticeForgeError(
        data.error?.message ?? 'API request failed',
        data.error?.code ?? 'unknown_error',
        response.status
      );
    }

    return data as T;
  }

  private async rawRequest(
    method: string,
    path: string,
    body?: unknown
  ): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        method,
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json',
          'X-Client': 'latticeforge-sdk/0.1.0',
        },
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      return response;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  private parseSignal(data: Record<string, unknown>): Signal {
    return {
      value: data.value as number,
      confidence: data.confidence as number,
      timestamp: new Date(data.timestamp as string),
      sources: data.sources as string[],
      regime: data.regime as Signal['regime'],
    };
  }
}

/**
 * LatticeForge Error
 */
export class LatticeForgeError extends Error {
  code: string;
  status: number;

  constructor(message: string, code: string, status: number) {
    super(message);
    this.name = 'LatticeForgeError';
    this.code = code;
    this.status = status;
  }
}

// Default export for simple import
export default LatticeForge;

// ========================================
// USAGE EXAMPLES (for documentation)
// ========================================

/**
 * @example Basic usage - 3 lines
 * ```typescript
 * import LatticeForge from '@latticeforge/sdk';
 * const lf = new LatticeForge('lf_live_xxx');
 * const signal = await lf.fuse(['SEC', 'FRED', 'sentiment']);
 * ```
 *
 * @example Check regime
 * ```typescript
 * const regime = await lf.regime();
 * console.log(`Market is ${regime}`);
 * ```
 *
 * @example Real-time streaming
 * ```typescript
 * for await (const signal of lf.stream(['SEC'])) {
 *   if (signal.value > 0.8) {
 *     console.log('High signal detected!');
 *   }
 * }
 * ```
 *
 * @example Detect anomalies
 * ```typescript
 * const alerts = await lf.detect();
 * const critical = alerts.filter(a => a.severity === 'critical');
 * if (critical.length > 0) {
 *   sendSlackAlert(critical);
 * }
 * ```
 *
 * @example Custom analysis
 * ```typescript
 * const phaseState = await lf.analyze('phase_transition', {
 *   signals: ['SEC', 'FRED', 'CDC'],
 *   sensitivity: 0.7
 * });
 * console.log(`Phase: ${phaseState.phase}, Temp: ${phaseState.temperature}`);
 * ```
 */
