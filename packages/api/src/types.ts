/**
 * LatticeForge API Types
 * Enterprise-grade type definitions
 */

export type ClientTier = 'free' | 'pro' | 'enterprise' | 'government';

export interface ApiClient {
  id: string;
  name: string;
  tier: ClientTier;
  apiKey: string;
  apiKeyHash: string;
  createdAt: string;
  lastActiveAt: string;
  rateLimit: RateLimitConfig;
  permissions: Permission[];
  metadata: ClientMetadata;
  active: boolean;
}

export interface ClientMetadata {
  organization?: string;
  contactEmail?: string;
  industry?: string;
  contractId?: string;
  notes?: string;
}

export interface RateLimitConfig {
  requestsPerMinute: number;
  requestsPerDay: number;
  burstLimit: number;
}

export type Permission =
  | 'read:signals'
  | 'read:sources'
  | 'read:traces'
  | 'write:signals'
  | 'admin:clients'
  | 'admin:system';

export interface AuthContext {
  client: ApiClient;
  requestId: string;
  timestamp: string;
  ip: string;
  userAgent: string;
}

export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: ApiError;
  meta: ResponseMeta;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface ResponseMeta {
  requestId: string;
  timestamp: string;
  version: string;
  tier: ClientTier;
  rateLimit: {
    remaining: number;
    reset: number;
  };
}

export interface SignalRequest {
  sources: string[];
  startDate?: string;
  endDate?: string;
  normalize?: boolean;
  fuse?: boolean;
}

export interface SignalResponse {
  signals: Record<string, number[]>;
  fused?: number[];
  phase?: number;
  confidence?: number;
  provenance: ProvenanceRecord[];
}

export interface ProvenanceRecord {
  sourceId: string;
  tier: string;
  fetchedAt: string;
  attribution: string;
  hash: string;
}

// Tier configurations
export const TIER_LIMITS: Record<ClientTier, RateLimitConfig> = {
  free: {
    requestsPerMinute: 10,
    requestsPerDay: 100,
    burstLimit: 5,
  },
  pro: {
    requestsPerMinute: 60,
    requestsPerDay: 5000,
    burstLimit: 20,
  },
  enterprise: {
    requestsPerMinute: 300,
    requestsPerDay: 50000,
    burstLimit: 100,
  },
  government: {
    requestsPerMinute: 1000,
    requestsPerDay: 500000,
    burstLimit: 500,
  },
};

export const TIER_PERMISSIONS: Record<ClientTier, Permission[]> = {
  free: ['read:signals', 'read:sources'],
  pro: ['read:signals', 'read:sources', 'read:traces'],
  enterprise: ['read:signals', 'read:sources', 'read:traces', 'write:signals'],
  government: ['read:signals', 'read:sources', 'read:traces', 'write:signals', 'admin:clients'],
};
