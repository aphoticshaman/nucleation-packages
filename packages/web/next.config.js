/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable Turbopack (Next.js 16 default)
  turbopack: {},

  // Enable WASM support (webpack fallback for complex WASM needs)
  webpack: (config) => {
    // WASM support
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
      layers: true,
    };

    // Fix for WASM imports
    config.module.rules.push({
      test: /\.wasm$/,
      type: 'webassembly/async',
    });

    return config;
  },

  // Headers for SharedArrayBuffer (required for multi-threaded WASM)
  // IMPORTANT: Exclude auth routes - COOP/COEP breaks OAuth redirects
  async headers() {
    return [
      {
        // Apply COOP/COEP only to app routes that need WASM, not auth routes
        source: '/(app|dashboard|admin)/:path*',
        headers: [
          {
            key: 'Cross-Origin-Opener-Policy',
            value: 'same-origin',
          },
          {
            key: 'Cross-Origin-Embedder-Policy',
            value: 'require-corp',
          },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
