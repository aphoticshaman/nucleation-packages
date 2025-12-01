/**
 * Authentication Middleware
 *
 * API key authentication with tier-based permissions.
 * Enterprise-grade auth for LatticeForge.
 */

import type { Context, Next } from 'hono';
import type { ApiClient, AuthContext, Permission } from '../types.js';
import { secureCompare, hashApiKey, getClientIp, generateRequestId } from './security.js';

// In-memory client store (replace with database in production)
const clientStore = new Map<string, ApiClient>();

/**
 * API Key authentication middleware
 */
export function authenticate() {
  return async (c: Context, next: Next) => {
    // Extract API key from header or query param
    const apiKey = extractApiKey(c);

    if (!apiKey) {
      return c.json(
        {
          success: false,
          error: {
            code: 'MISSING_API_KEY',
            message: 'API key required. Provide via X-API-Key header or api_key query parameter.',
          },
          meta: buildMeta(c),
        },
        401
      );
    }

    // Validate API key format
    if (!isValidKeyFormat(apiKey)) {
      return c.json(
        {
          success: false,
          error: {
            code: 'INVALID_API_KEY_FORMAT',
            message: 'Invalid API key format',
          },
          meta: buildMeta(c),
        },
        401
      );
    }

    // Find client by API key hash
    const client = findClientByKey(apiKey);

    if (!client) {
      return c.json(
        {
          success: false,
          error: {
            code: 'INVALID_API_KEY',
            message: 'API key not found or revoked',
          },
          meta: buildMeta(c),
        },
        401
      );
    }

    // Check if client is active
    if (!client.active) {
      return c.json(
        {
          success: false,
          error: {
            code: 'CLIENT_SUSPENDED',
            message: 'Account suspended. Contact support.',
          },
          meta: buildMeta(c),
        },
        403
      );
    }

    // Build auth context
    const authContext: AuthContext = {
      client,
      requestId: c.get('requestId') ?? generateRequestId(),
      timestamp: new Date().toISOString(),
      ip: getClientIp(c),
      userAgent: c.req.header('User-Agent') ?? 'unknown',
    };

    // Store auth context for downstream use
    c.set('auth', authContext);

    // Update last active timestamp
    client.lastActiveAt = authContext.timestamp;

    await next();
  };
}

/**
 * Permission checking middleware
 */
export function requirePermission(...permissions: Permission[]) {
  return async (c: Context, next: Next) => {
    const auth = c.get('auth') as AuthContext | undefined;

    if (!auth) {
      return c.json(
        {
          success: false,
          error: {
            code: 'NOT_AUTHENTICATED',
            message: 'Authentication required',
          },
          meta: buildMeta(c),
        },
        401
      );
    }

    const hasPermission = permissions.every((p) => auth.client.permissions.includes(p));

    if (!hasPermission) {
      return c.json(
        {
          success: false,
          error: {
            code: 'INSUFFICIENT_PERMISSIONS',
            message: `Required permissions: ${permissions.join(', ')}`,
          },
          meta: buildMeta(c),
        },
        403
      );
    }

    await next();
  };
}

/**
 * Tier requirement middleware
 */
export function requireTier(...tiers: ApiClient['tier'][]) {
  return async (c: Context, next: Next) => {
    const auth = c.get('auth') as AuthContext | undefined;

    if (!auth) {
      return c.json(
        {
          success: false,
          error: {
            code: 'NOT_AUTHENTICATED',
            message: 'Authentication required',
          },
          meta: buildMeta(c),
        },
        401
      );
    }

    if (!tiers.includes(auth.client.tier)) {
      return c.json(
        {
          success: false,
          error: {
            code: 'TIER_REQUIRED',
            message: `This endpoint requires ${tiers.join(' or ')} tier`,
          },
          meta: buildMeta(c),
        },
        403
      );
    }

    await next();
  };
}

// Helper functions

function extractApiKey(c: Context): string | null {
  // Check X-API-Key header first
  const headerKey = c.req.header('X-API-Key');
  if (headerKey) return headerKey;

  // Check Authorization header (Bearer token)
  const authHeader = c.req.header('Authorization');
  if (authHeader?.startsWith('Bearer ')) {
    return authHeader.slice(7);
  }

  // Check query parameter (least preferred)
  const queryKey = c.req.query('api_key');
  if (queryKey) return queryKey;

  return null;
}

function isValidKeyFormat(key: string): boolean {
  // LatticeForge keys start with lf_live_ or lf_test_
  return /^lf_(live|test)_[A-Za-z0-9_-]{32,64}$/.test(key);
}

function findClientByKey(apiKey: string): ApiClient | undefined {
  const keyHash = hashApiKey(apiKey);

  for (const client of clientStore.values()) {
    if (secureCompare(client.apiKeyHash, keyHash)) {
      return client;
    }
  }

  return undefined;
}

function buildMeta(c: Context) {
  const auth = c.get('auth') as AuthContext | undefined;

  return {
    requestId: c.get('requestId') ?? 'unknown',
    timestamp: new Date().toISOString(),
    version: '1.0.0',
    tier: auth?.client.tier ?? 'unknown',
    rateLimit: {
      remaining: -1,
      reset: 0,
    },
  };
}

// Client management functions (for admin use)

export function registerClient(client: Omit<ApiClient, 'apiKeyHash'>): ApiClient {
  const fullClient: ApiClient = {
    ...client,
    apiKeyHash: hashApiKey(client.apiKey),
  };

  clientStore.set(client.id, fullClient);
  return fullClient;
}

export function getClient(id: string): ApiClient | undefined {
  return clientStore.get(id);
}

export function revokeClient(id: string): boolean {
  const client = clientStore.get(id);
  if (client) {
    client.active = false;
    return true;
  }
  return false;
}

export function deleteClient(id: string): boolean {
  return clientStore.delete(id);
}

export function listClients(): ApiClient[] {
  return [...clientStore.values()];
}
