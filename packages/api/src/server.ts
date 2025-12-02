/**
 * LatticeForge API Server
 *
 * Enterprise-grade signal intelligence API.
 * Fort Knox security. Crystalline clarity.
 */

import { Hono } from 'hono';
import { serve } from '@hono/node-server';
import {
  securityHeaders,
  corsMiddleware,
  validateRequest,
  requestLogger,
} from './middleware/security.js';
import { authenticate, registerClient } from './middleware/auth.js';
import { rateLimit } from './middleware/rate-limit.js';
import { TIER_LIMITS, TIER_PERMISSIONS } from './types.js';
import { generateApiKey } from './middleware/security.js';
import signals from './routes/signals.js';

const app = new Hono();

// ============ BRANDING ============

const BANNER = `
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ██╗      █████╗ ████████╗████████╗██╗ ██████╗███████╗         ║
║   ██║     ██╔══██╗╚══██╔══╝╚══██╔══╝██║██╔════╝██╔════╝         ║
║   ██║     ███████║   ██║      ██║   ██║██║     █████╗           ║
║   ██║     ██╔══██║   ██║      ██║   ██║██║     ██╔══╝           ║
║   ███████╗██║  ██║   ██║      ██║   ██║╚██████╗███████╗         ║
║   ╚══════╝╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚═╝ ╚═════╝╚══════╝         ║
║                                                                  ║
║   ███████╗ ██████╗ ██████╗  ██████╗ ███████╗                    ║
║   ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝                    ║
║   █████╗  ██║   ██║██████╔╝██║  ███╗█████╗                      ║
║   ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝                      ║
║   ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗                    ║
║   ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝                    ║
║                                                                  ║
║   The Crystallization of Meta-Insight                           ║
║   Signal Intelligence for the Enterprise                        ║
║                                                                  ║
║   Crystalline Labs LLC © 2025                                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
`;

// ============ GLOBAL MIDDLEWARE ============

// Security headers on all requests
app.use('*', securityHeaders());

// CORS - configure allowed origins in production
const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') ?? [];
app.use('*', corsMiddleware(allowedOrigins));

// Request validation
app.use('*', validateRequest());

// Request logging
app.use(
  '*',
  requestLogger((entry) => {
    const log = `[${entry.timestamp}] ${entry.method} ${entry.path} ${entry.status} ${entry.duration}ms - ${entry.ip}`;
    console.log(log);
  })
);

// ============ PUBLIC ROUTES ============

/**
 * Health check - no auth required
 */
app.get('/health', (c) => {
  return c.json({
    status: 'healthy',
    service: 'latticeforge-api',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
  });
});

/**
 * Root - API info
 */
app.get('/', (c) => {
  return c.json({
    name: 'LatticeForge API',
    tagline: 'The Crystallization of Meta-Insight',
    version: '1.0.0',
    documentation: '/docs',
    health: '/health',
    endpoints: {
      signals: '/v1/signals',
      sources: '/v1/sources',
      detect: '/v1/signals/detect',
    },
    company: {
      name: 'Crystalline Labs LLC',
      product: 'LatticeForge',
      contact: 'api@latticeforge.io',
    },
  });
});

/**
 * API Documentation
 */
app.get('/docs', (c) => {
  return c.json({
    openapi: '3.0.0',
    info: {
      title: 'LatticeForge API',
      description: 'Enterprise signal intelligence and phase transition detection',
      version: '1.0.0',
      contact: {
        name: 'Crystalline Labs LLC',
        email: 'api@latticeforge.io',
      },
    },
    servers: [
      { url: 'https://api.latticeforge.io/v1', description: 'Production' },
      { url: 'http://localhost:3000/v1', description: 'Development' },
    ],
    security: [{ apiKey: [] }],
    paths: {
      '/signals': {
        get: {
          summary: 'List available signal sources',
          security: [{ apiKey: [] }],
        },
        post: {
          summary: 'Fetch signals from multiple sources',
          security: [{ apiKey: [] }],
        },
      },
      '/signals/{sourceId}': {
        get: {
          summary: 'Get signals from a specific source',
          security: [{ apiKey: [] }],
        },
      },
      '/signals/detect': {
        post: {
          summary: 'Run phase detection on signals',
          security: [{ apiKey: [] }],
        },
      },
    },
    components: {
      securitySchemes: {
        apiKey: {
          type: 'apiKey',
          in: 'header',
          name: 'X-API-Key',
        },
      },
    },
  });
});

// ============ AUTHENTICATED ROUTES ============

// Apply auth and rate limiting to /v1/*
app.use('/v1/*', authenticate());
app.use('/v1/*', rateLimit());

// Mount signal routes
app.route('/v1/signals', signals);

// ============ ERROR HANDLING ============

app.onError((err, c) => {
  console.error(`[ERROR] ${err.message}`, err.stack);

  // Don't leak internal errors in production
  const message =
    process.env.NODE_ENV === 'production' ? 'An internal error occurred' : err.message;

  return c.json(
    {
      success: false,
      error: {
        code: 'INTERNAL_ERROR',
        message,
      },
      meta: {
        requestId: c.get('requestId') ?? 'unknown',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
      },
    },
    500
  );
});

app.notFound((c) => {
  return c.json(
    {
      success: false,
      error: {
        code: 'NOT_FOUND',
        message: `Endpoint not found: ${c.req.method} ${c.req.path}`,
      },
      meta: {
        requestId: c.get('requestId') ?? 'unknown',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
      },
    },
    404
  );
});

// ============ STARTUP ============

const port = parseInt(process.env.PORT ?? '3000', 10);

// Create demo client for development
if (process.env.NODE_ENV !== 'production') {
  const demoKey = generateApiKey();
  registerClient({
    id: 'demo-client',
    name: 'Demo Client',
    tier: 'enterprise',
    apiKey: demoKey,
    createdAt: new Date().toISOString(),
    lastActiveAt: new Date().toISOString(),
    rateLimit: TIER_LIMITS.enterprise,
    permissions: TIER_PERMISSIONS.enterprise,
    metadata: {
      organization: 'Demo Organization',
      contactEmail: 'demo@example.com',
    },
    active: true,
  });

  console.log(BANNER);
  console.log(`\n🔐 Demo API Key (for development only):`);
  console.log(`   ${demoKey}\n`);
}

console.log(`🚀 LatticeForge API starting on port ${port}`);
console.log(`   Health: http://localhost:${port}/health`);
console.log(`   Docs:   http://localhost:${port}/docs`);
console.log(`   API:    http://localhost:${port}/v1/signals\n`);

serve({
  fetch: app.fetch,
  port,
});

export default app;
