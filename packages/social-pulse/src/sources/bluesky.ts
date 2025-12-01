/**
 * Bluesky AT Protocol data source
 *
 * Fully free and open API via the AT Protocol.
 * No authentication required for public data.
 */

import type {
  DataSource,
  SearchParams,
  SocialPost,
  AuthorInfo,
  EngagementMetrics,
  SourceConfig,
} from '../types.js';

const DEFAULT_ENDPOINT = 'https://public.api.bsky.app';

interface BskyFeedPost {
  uri: string;
  cid: string;
  author: {
    did: string;
    handle: string;
    displayName?: string;
    avatar?: string;
    description?: string;
    createdAt?: string;
    followersCount?: number;
    followsCount?: number;
    postsCount?: number;
  };
  record: {
    text: string;
    createdAt: string;
    langs?: string[];
  };
  replyCount?: number;
  repostCount?: number;
  likeCount?: number;
  indexedAt: string;
}

interface BskySearchResponse {
  posts: BskyFeedPost[];
  cursor?: string;
}

export class BlueskySource implements DataSource {
  readonly platform = 'bluesky' as const;
  private endpoint: string;
  private ready = false;

  constructor(config: SourceConfig = { platform: 'bluesky' }) {
    this.endpoint = config.endpoint ?? DEFAULT_ENDPOINT;
  }

  async init(): Promise<void> {
    // Test connectivity
    try {
      const response = await fetch(
        `${this.endpoint}/xrpc/app.bsky.actor.searchActors?q=test&limit=1`
      );
      if (!response.ok && response.status !== 400) {
        throw new Error(`Bluesky API check failed: ${response.status}`);
      }
      this.ready = true;
    } catch (error) {
      throw new Error(`Failed to initialize Bluesky source: ${error}`);
    }
  }

  isReady(): boolean {
    return this.ready;
  }

  async fetch(params: SearchParams): Promise<SocialPost[]> {
    if (!this.ready) {
      throw new Error('Bluesky source not initialized');
    }

    const query = this.buildQuery(params);
    if (!query) {
      return [];
    }

    const limit = Math.min(params.limit ?? 100, 100);
    const url = new URL(`${this.endpoint}/xrpc/app.bsky.feed.searchPosts`);
    url.searchParams.set('q', query);
    url.searchParams.set('limit', limit.toString());

    if (params.since) {
      url.searchParams.set('since', params.since.toISOString());
    }
    if (params.until) {
      url.searchParams.set('until', params.until.toISOString());
    }
    if (params.languages?.length && params.languages[0]) {
      url.searchParams.set('lang', params.languages[0]);
    }

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`Bluesky search failed: ${response.status}`);
    }

    const data = (await response.json()) as BskySearchResponse;
    return data.posts.map((post) => this.transformPost(post));
  }

  async stream(
    params: SearchParams,
    callback: (post: SocialPost) => void
  ): Promise<{ stop: () => void }> {
    // Bluesky has a firehose but requires auth
    // For now, poll at intervals
    let running = true;
    let lastTimestamp = params.since ?? new Date();

    const poll = async (): Promise<void> => {
      while (running) {
        try {
          const posts = await this.fetch({
            ...params,
            since: lastTimestamp,
            limit: 50,
          });

          for (const post of posts) {
            callback(post);
            const postTime = new Date(post.timestamp);
            if (postTime > lastTimestamp) {
              lastTimestamp = postTime;
            }
          }
        } catch (error) {
          console.error('Bluesky poll error:', error);
        }

        // Wait 30 seconds between polls
        await new Promise((resolve) => setTimeout(resolve, 30000));
      }
    };

    poll().catch(console.error);

    return {
      stop: () => {
        running = false;
      },
    };
  }

  private buildQuery(params: SearchParams): string {
    const parts: string[] = [];

    if (params.keywords?.length) {
      parts.push(params.keywords.join(' OR '));
    }

    if (params.hashtags?.length) {
      parts.push(params.hashtags.map((h) => `#${h.replace(/^#/, '')}`).join(' OR '));
    }

    return parts.join(' ');
  }

  private transformPost(post: BskyFeedPost): SocialPost {
    const author: AuthorInfo = {
      id: post.author.did,
      name: post.author.handle,
      verified: false, // Bluesky doesn't have verification yet
    };

    // Only add optional properties if they exist
    if (post.author.createdAt) author.createdAt = post.author.createdAt;
    if (post.author.followersCount !== undefined) author.followers = post.author.followersCount;
    if (post.author.followsCount !== undefined) author.following = post.author.followsCount;
    if (post.author.postsCount !== undefined) author.postCount = post.author.postsCount;
    if (post.author.description) author.bio = post.author.description;

    const engagement: EngagementMetrics = {
      likes: post.likeCount ?? 0,
      reposts: post.repostCount ?? 0,
      replies: post.replyCount ?? 0,
    };

    const socialPost: SocialPost = {
      id: post.uri,
      platform: 'bluesky',
      content: post.record.text,
      timestamp: post.record.createdAt,
      author,
      engagement,
      raw: post,
    };

    // Only add language if it exists
    if (post.record.langs?.[0]) {
      socialPost.language = post.record.langs[0];
    }

    return socialPost;
  }
}
