/**
 * LatticeForge Asset Registry
 *
 * Centralized asset management for the Dark Glass UI theme.
 * Assets are served from Vercel's global edge CDN (/public/images/).
 *
 * For Supabase Storage (future):
 * const SUPABASE_CDN = 'https://[project].supabase.co/storage/v1/object/public/assets';
 */

// Base path for static assets (Vercel edge CDN)
const ASSET_BASE = '/images';

/**
 * Asset categories matching the CATALOG.md structure
 */
export const assets = {
  // === BACKGROUNDS & TEXTURES ===
  backgrounds: {
    obsidian: `${ASSET_BASE}/bg/obsidian.png`,
    slateGradient: `${ASSET_BASE}/bg/slate-gradient.png`,
    navyGradient: `${ASSET_BASE}/bg/navy-gradient.png`,
    titaniumHex: `${ASSET_BASE}/bg/titanium-hex.png`,
  },

  // === GLOBE & GEOSPATIAL ===
  globe: {
    hero: `${ASSET_BASE}/hero/globe-wire.png`,
    hexbin: `${ASSET_BASE}/hero/hexbin-map.png`,
    wireframe: `${ASSET_BASE}/hero/wireframe-sphere.png`,
    dataGlobe: `${ASSET_BASE}/hero/data-globe.png`,
    earth: `${ASSET_BASE}/hero/earth.png`,
  },

  // === NETWORK & NODES ===
  network: {
    nodes: `${ASSET_BASE}/network/nodes.png`,
    constellation: `${ASSET_BASE}/network/constellation.png`,
    cluster: `${ASSET_BASE}/network/cluster.png`,
    causality: `${ASSET_BASE}/network/causality.png`,
  },

  // === ABSTRACT SHAPES ===
  shapes: {
    ring: `${ASSET_BASE}/shapes/ring-pulse.png`,
    hexagon: `${ASSET_BASE}/shapes/hexagon.png`,
    sphere: `${ASSET_BASE}/shapes/sphere.png`,
    crystal: `${ASSET_BASE}/shapes/crystal.png`,
    plasma: `${ASSET_BASE}/shapes/plasma.png`,
    infinity: `${ASSET_BASE}/shapes/infinity.png`,
  },

  // === DASHBOARD & ANALYTICS ===
  dashboard: {
    mockup: `${ASSET_BASE}/dashboard/mockup.png`,
    waterfall: `${ASSET_BASE}/dashboard/waterfall.png`,
    monitors: `${ASSET_BASE}/dashboard/monitors.png`,
    warRoom: `${ASSET_BASE}/dashboard/war-room.png`,
  },

  // === LOGOS & BRANDING ===
  brand: {
    appIcon: `${ASSET_BASE}/brand/app-icon.png`,
    logo: `${ASSET_BASE}/brand/logo.png`,
    monogram: `${ASSET_BASE}/brand/monogram.png`,
  },

  // === FEATURE ICONS ===
  icons: {
    analytics: `${ASSET_BASE}/icons/analytics.png`,
    api: `${ASSET_BASE}/icons/api.png`,
    export: `${ASSET_BASE}/icons/export.png`,
    security: `${ASSET_BASE}/icons/security.png`,
    simulation: `${ASSET_BASE}/icons/simulation.png`,
    team: `${ASSET_BASE}/icons/team.png`,
    webhook: `${ASSET_BASE}/icons/webhook.png`,
    growth: `${ASSET_BASE}/icons/growth.png`,
  },

  // === TIER BADGES ===
  badges: {
    trial: `${ASSET_BASE}/badges/trial.png`,
    starter: `${ASSET_BASE}/badges/starter.png`,
    pro: `${ASSET_BASE}/badges/pro.png`,
    enterprise: `${ASSET_BASE}/badges/enterprise.png`,
  },

  // === ERROR & EMPTY STATES ===
  states: {
    error404: `${ASSET_BASE}/states/404.png`,
    error500: `${ASSET_BASE}/states/500.png`,
    connectionError: `${ASSET_BASE}/states/connection-error.png`,
    empty: `${ASSET_BASE}/states/empty.png`,
    success: `${ASSET_BASE}/states/success.png`,
    emptyDashboard: `${ASSET_BASE}/states/empty-dashboard.png`,
  },

  // === ONBOARDING ===
  onboarding: {
    analysis: `${ASSET_BASE}/onboarding/analysis.png`,
    connecting: `${ASSET_BASE}/onboarding/connecting.png`,
    exporting: `${ASSET_BASE}/onboarding/exporting.png`,
    simulation: `${ASSET_BASE}/onboarding/simulation.png`,
  },

  // === ATMOSPHERIC ===
  atmosphere: {
    ocean: `${ASSET_BASE}/atmosphere/ocean.png`,
    sinkhole: `${ASSET_BASE}/atmosphere/sinkhole.png`,
    chessKing: `${ASSET_BASE}/atmosphere/chess-king.png`,
    mercuryEarth: `${ASSET_BASE}/atmosphere/mercury-earth.png`,
    workspace: `${ASSET_BASE}/atmosphere/workspace.png`,
    horizon: `${ASSET_BASE}/atmosphere/horizon.png`,
    topographic: `${ASSET_BASE}/atmosphere/topographic.png`,
  },

  // === SOCIAL MEDIA ===
  social: {
    ogCard: `${ASSET_BASE}/og/og-image.png`,
    twitterCard: `${ASSET_BASE}/social/twitter-card.png`,
  },

  // === RISK & ALERT ===
  risk: {
    energy: `${ASSET_BASE}/risk/energy.png`,
    shield: `${ASSET_BASE}/risk/shield.png`,
    bracket: `${ASSET_BASE}/risk/bracket.png`,
  },

  // === VIDEOS ===
  video: {
    hero1: `${ASSET_BASE}/video/hero-1.mp4`,
    hero2: `${ASSET_BASE}/video/hero-2.mp4`,
  },
} as const;

/**
 * Type for all asset paths
 */
export type AssetPath = typeof assets[keyof typeof assets][keyof typeof assets[keyof typeof assets]];

/**
 * Get asset URL with optional Supabase Storage fallback
 */
export function getAssetUrl(path: string, options?: {
  width?: number;
  quality?: number;
  format?: 'webp' | 'avif' | 'auto';
}): string {
  // For Vercel, use Next.js Image optimization
  // Assets are automatically served from edge CDN
  if (options?.width || options?.quality) {
    const params = new URLSearchParams();
    if (options.width) params.set('w', String(options.width));
    if (options.quality) params.set('q', String(options.quality));
    return `/_next/image?url=${encodeURIComponent(path)}&${params.toString()}`;
  }
  return path;
}

/**
 * Preload critical assets for better LCP
 */
export const criticalAssets = [
  assets.globe.hero,
  assets.backgrounds.titaniumHex,
  assets.brand.logo,
];

/**
 * Asset manifest for service worker caching
 */
export const assetManifest = Object.values(assets).flatMap(
  category => Object.values(category)
);
