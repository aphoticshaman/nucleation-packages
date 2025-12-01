/**
 * Telegram public channel data source
 *
 * Accesses public Telegram channels and groups.
 * Requires no authentication for public data via web preview.
 */

import type {
  DataSource,
  SearchParams,
  SocialPost,
  AuthorInfo,
  EngagementMetrics,
} from '../types.js';

// Telegram web preview endpoints
const TELEGRAM_PREVIEW = 'https://t.me/s';

export class TelegramSource implements DataSource {
  readonly platform = 'telegram' as const;
  private ready = false;
  private channels: string[] = [];

  constructor(channels: string[] = []) {
    this.channels = channels;
  }

  async init(): Promise<void> {
    this.ready = true;
  }

  isReady(): boolean {
    return this.ready;
  }

  /**
   * Add channels to monitor
   */
  addChannels(channels: string[]): void {
    for (const channel of channels) {
      const normalized = channel.replace(/^@/, '').replace(/^https:\/\/t\.me\//, '');
      if (!this.channels.includes(normalized)) {
        this.channels.push(normalized);
      }
    }
  }

  async fetch(params: SearchParams): Promise<SocialPost[]> {
    if (!this.ready) {
      throw new Error('Telegram source not initialized');
    }

    const posts: SocialPost[] = [];
    const channelsToFetch =
      this.channels.length > 0 ? this.channels : this.getDefaultChannels(params.countries);

    for (const channel of channelsToFetch.slice(0, 10)) {
      try {
        const channelPosts = await this.fetchChannel(channel);
        posts.push(...this.filterPosts(channelPosts, params));
      } catch (error) {
        console.warn(`Failed to fetch Telegram channel ${channel}:`, error);
      }

      // Rate limiting between channels
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    return posts.slice(0, params.limit ?? 100);
  }

  private async fetchChannel(channelName: string): Promise<SocialPost[]> {
    const url = `${TELEGRAM_PREVIEW}/${channelName}`;

    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; NucleationBot/1.0)',
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch channel: ${response.status}`);
    }

    const html = await response.text();
    return this.parseChannelHtml(html, channelName);
  }

  private parseChannelHtml(html: string, channelName: string): SocialPost[] {
    const posts: SocialPost[] = [];

    // Parse message blocks from Telegram web preview
    // Format: <div class="tgme_widget_message" data-post="channel/123">
    const messageRegex =
      /class="tgme_widget_message[^"]*"[^>]*data-post="([^"]+)"[\s\S]*?<div class="tgme_widget_message_text[^"]*"[^>]*>([\s\S]*?)<\/div>/g;

    let match;
    while ((match = messageRegex.exec(html)) !== null) {
      const postId = match[1];
      const rawText = match[2];

      if (!postId || !rawText) continue;

      // Strip HTML tags (loop to handle nested tags)
      let text = rawText.replace(/<br\s*\/?>/gi, '\n');
      // Remove all HTML tags - loop until no more tags found
      let prevText = '';
      while (prevText !== text) {
        prevText = text;
        text = text.replace(/<[^>]+>/g, '');
      }
      // Decode HTML entities in correct order (amp LAST to prevent double-unescaping)
      text = text
        .replace(/&nbsp;/g, ' ')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'")
        .replace(/&amp;/g, '&') // Must be last to avoid double-unescaping
        .trim();

      if (text.length > 0) {
        // Extract timestamp
        const timeMatch = /datetime="([^"]+)"/.exec(html.slice(match.index, match.index + 1000));
        const timestamp = timeMatch && timeMatch[1] ? timeMatch[1] : new Date().toISOString();

        // Extract views
        const viewsMatch = /class="tgme_widget_message_views"[^>]*>([^<]+)/.exec(
          html.slice(match.index, match.index + 2000)
        );
        const views = viewsMatch && viewsMatch[1] ? this.parseViewCount(viewsMatch[1]) : undefined;

        const author: AuthorInfo = {
          id: channelName,
          name: channelName,
        };

        const engagement: EngagementMetrics = {};
        if (views !== undefined) {
          engagement.views = views;
        }

        posts.push({
          id: postId,
          platform: 'telegram',
          content: text,
          timestamp,
          author,
          engagement,
          raw: { channelName, postId },
        });
      }
    }

    return posts;
  }

  private parseViewCount(viewStr: string): number {
    const cleaned = viewStr.trim().toLowerCase();
    if (cleaned.endsWith('k')) {
      return parseFloat(cleaned) * 1000;
    }
    if (cleaned.endsWith('m')) {
      return parseFloat(cleaned) * 1000000;
    }
    return parseInt(cleaned, 10) || 0;
  }

  private filterPosts(posts: SocialPost[], params: SearchParams): SocialPost[] {
    return posts.filter((post) => {
      // Keyword filter
      if (params.keywords?.length) {
        const content = post.content.toLowerCase();
        const hasKeyword = params.keywords.some((kw) => content.includes(kw.toLowerCase()));
        if (!hasKeyword) return false;
      }

      // Time filter
      if (params.since) {
        if (new Date(post.timestamp) < params.since) return false;
      }
      if (params.until) {
        if (new Date(post.timestamp) > params.until) return false;
      }

      return true;
    });
  }

  /**
   * Get default channels based on country codes
   */
  private getDefaultChannels(countryCodes?: string[]): string[] {
    // Major news/information channels by region
    const channelsByRegion: Record<string, string[]> = {
      RU: ['rian_ru', 'medaborona', 'bbcrussian'],
      UA: ['ukrainenowenglish', 'nexaborona'],
      IR: ['bbcpersian'],
      BY: ['nexta_live'],
      // Add more regions as needed
    };

    if (!countryCodes?.length) {
      // Return global channels
      return ['bbcnews', 'reuters'];
    }

    const channels: string[] = [];
    for (const code of countryCodes) {
      const regional = channelsByRegion[code.toUpperCase()];
      if (regional) {
        channels.push(...regional);
      }
    }

    return channels.length > 0 ? channels : ['bbcnews'];
  }
}
