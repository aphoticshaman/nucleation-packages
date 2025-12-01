/**
 * LatticeForge Analytics
 *
 * "You have 6,000 lines of formula code and 0 lines of user analytics.
 *  Invert this ratio." - Eric Ries
 *
 * This module provides:
 * - Event tracking (Mixpanel/Amplitude/PostHog compatible)
 * - User identification
 * - Feature usage tracking
 * - Funnel analysis
 * - Session recording integration
 *
 * In production, swap the adapter for your analytics provider.
 */

export interface AnalyticsEvent {
  name: string;
  properties?: Record<string, unknown>;
  timestamp?: Date;
}

export interface UserTraits {
  email?: string;
  name?: string;
  tier?: string;
  company?: string;
  signupDate?: Date;
  [key: string]: unknown;
}

export interface AnalyticsAdapter {
  identify(userId: string, traits?: UserTraits): void;
  track(event: AnalyticsEvent): void;
  page(name: string, properties?: Record<string, unknown>): void;
  reset(): void;
}

/**
 * Console adapter for development (logs events to console)
 */
class ConsoleAdapter implements AnalyticsAdapter {
  private userId: string | null = null;

  identify(userId: string, traits?: UserTraits): void {
    this.userId = userId;
    console.log('[Analytics] Identify:', userId, traits);
  }

  track(event: AnalyticsEvent): void {
    console.log('[Analytics] Track:', event.name, {
      userId: this.userId,
      ...event.properties,
      timestamp: event.timestamp ?? new Date(),
    });
  }

  page(name: string, properties?: Record<string, unknown>): void {
    console.log('[Analytics] Page:', name, properties);
  }

  reset(): void {
    this.userId = null;
    console.log('[Analytics] Reset');
  }
}

/**
 * In-memory adapter for testing
 */
class MemoryAdapter implements AnalyticsAdapter {
  public events: AnalyticsEvent[] = [];
  public pages: Array<{ name: string; properties?: Record<string, unknown> }> = [];
  public userId: string | null = null;
  public traits: UserTraits | null = null;

  identify(userId: string, traits?: UserTraits): void {
    this.userId = userId;
    this.traits = traits ?? null;
  }

  track(event: AnalyticsEvent): void {
    this.events.push({ ...event, timestamp: event.timestamp ?? new Date() });
  }

  page(name: string, properties?: Record<string, unknown>): void {
    this.pages.push({ name, properties });
  }

  reset(): void {
    this.events = [];
    this.pages = [];
    this.userId = null;
    this.traits = null;
  }
}

/**
 * Mixpanel adapter
 */
class MixpanelAdapter implements AnalyticsAdapter {
  private mixpanel: { identify: Function; track: Function; reset: Function } | null = null;

  constructor(token: string) {
    // @ts-ignore - Mixpanel global
    if (typeof window !== 'undefined' && window.mixpanel) {
      // @ts-ignore
      this.mixpanel = window.mixpanel;
      this.mixpanel?.init?.(token);
    }
  }

  identify(userId: string, traits?: UserTraits): void {
    this.mixpanel?.identify(userId);
    if (traits) {
      // @ts-ignore
      this.mixpanel?.people?.set(traits);
    }
  }

  track(event: AnalyticsEvent): void {
    this.mixpanel?.track(event.name, event.properties);
  }

  page(name: string, properties?: Record<string, unknown>): void {
    this.mixpanel?.track('Page View', { page: name, ...properties });
  }

  reset(): void {
    this.mixpanel?.reset();
  }
}

/**
 * PostHog adapter (open-source alternative)
 */
class PostHogAdapter implements AnalyticsAdapter {
  private posthog: { identify: Function; capture: Function; reset: Function } | null = null;

  constructor(apiKey: string, apiHost: string = 'https://app.posthog.com') {
    // @ts-ignore - PostHog global
    if (typeof window !== 'undefined' && window.posthog) {
      // @ts-ignore
      this.posthog = window.posthog;
      this.posthog?.init?.(apiKey, { api_host: apiHost });
    }
  }

  identify(userId: string, traits?: UserTraits): void {
    this.posthog?.identify(userId, traits);
  }

  track(event: AnalyticsEvent): void {
    this.posthog?.capture(event.name, event.properties);
  }

  page(name: string, properties?: Record<string, unknown>): void {
    this.posthog?.capture('$pageview', { page: name, ...properties });
  }

  reset(): void {
    this.posthog?.reset();
  }
}

/**
 * Main Analytics class
 */
class Analytics {
  private adapter: AnalyticsAdapter;
  private queue: AnalyticsEvent[] = [];
  private initialized = false;

  constructor() {
    // Default to console in development
    this.adapter = new ConsoleAdapter();
  }

  /**
   * Initialize with adapter
   */
  init(adapter: AnalyticsAdapter): void {
    this.adapter = adapter;
    this.initialized = true;

    // Flush queued events
    for (const event of this.queue) {
      this.adapter.track(event);
    }
    this.queue = [];
  }

  /**
   * Initialize with Mixpanel
   */
  initMixpanel(token: string): void {
    this.init(new MixpanelAdapter(token));
  }

  /**
   * Initialize with PostHog
   */
  initPostHog(apiKey: string, apiHost?: string): void {
    this.init(new PostHogAdapter(apiKey, apiHost));
  }

  /**
   * Identify user
   */
  identify(userId: string, traits?: UserTraits): void {
    this.adapter.identify(userId, traits);
  }

  /**
   * Track event
   */
  track(name: string, properties?: Record<string, unknown>): void {
    const event: AnalyticsEvent = { name, properties, timestamp: new Date() };

    if (!this.initialized) {
      this.queue.push(event);
      return;
    }

    this.adapter.track(event);
  }

  /**
   * Track page view
   */
  page(name: string, properties?: Record<string, unknown>): void {
    this.adapter.page(name, properties);
  }

  /**
   * Reset (on logout)
   */
  reset(): void {
    this.adapter.reset();
  }

  // ========================================
  // LATTICEFORGE-SPECIFIC EVENTS
  // ========================================

  /**
   * Track login
   */
  login(userId: string, tier: string): void {
    this.identify(userId, { tier });
    this.track('User Logged In', { tier });
  }

  /**
   * Track logout
   */
  logout(): void {
    this.track('User Logged Out');
    this.reset();
  }

  /**
   * Track signal fusion
   */
  fusionRequested(sources: string[]): void {
    this.track('Fusion Requested', {
      sources,
      source_count: sources.length,
    });
  }

  /**
   * Track analysis run
   */
  analysisRun(formula: string, params: Record<string, unknown>): void {
    this.track('Analysis Run', {
      formula,
      params,
    });
  }

  /**
   * Track alert viewed
   */
  alertViewed(alertId: string, alertType: string, severity: string): void {
    this.track('Alert Viewed', {
      alert_id: alertId,
      alert_type: alertType,
      severity,
    });
  }

  /**
   * Track feature used
   */
  featureUsed(feature: string, context?: Record<string, unknown>): void {
    this.track('Feature Used', {
      feature,
      ...context,
    });
  }

  /**
   * Track error
   */
  error(errorType: string, message: string, context?: Record<string, unknown>): void {
    this.track('Error Occurred', {
      error_type: errorType,
      message,
      ...context,
    });
  }

  /**
   * Track upgrade intent
   */
  upgradeClicked(currentTier: string, targetTier: string): void {
    this.track('Upgrade Clicked', {
      current_tier: currentTier,
      target_tier: targetTier,
    });
  }

  /**
   * Track API key copied
   */
  apiKeyCopied(): void {
    this.track('API Key Copied');
  }

  /**
   * Track documentation viewed
   */
  docsViewed(section: string): void {
    this.track('Docs Viewed', { section });
  }
}

// Singleton instance
export const analytics = new Analytics();

// Export adapters for custom use
export { ConsoleAdapter, MemoryAdapter, MixpanelAdapter, PostHogAdapter };

// Default export
export default analytics;
