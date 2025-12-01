/**
 * @latticeforge/api
 *
 * Enterprise API for LatticeForge signal intelligence.
 */

export { default as app } from './server.js';

// Types
export type {
  ApiClient,
  ClientTier,
  ClientMetadata,
  RateLimitConfig,
  Permission,
  AuthContext,
  ApiResponse,
  ApiError,
  ResponseMeta,
  SignalRequest,
  SignalResponse,
  ProvenanceRecord,
} from './types.js';

export { TIER_LIMITS, TIER_PERMISSIONS } from './types.js';

// Auth utilities
export {
  registerClient,
  getClient,
  revokeClient,
  deleteClient,
  listClients,
} from './middleware/auth.js';

// Security utilities
export {
  generateApiKey,
  hashApiKey,
  secureCompare,
  generateRequestId,
} from './middleware/security.js';
