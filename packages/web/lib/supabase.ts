import { createBrowserClient } from '@supabase/ssr';

// Use ReturnType to get the correct type from createBrowserClient
type BrowserClient = ReturnType<typeof createBrowserClient>;

// Lazy-initialized Supabase client (avoids build-time env var access)
let _supabase: BrowserClient | null = null;

// Check if we're in a browser environment
const isBrowser = typeof window !== 'undefined';

// Cookie domain for cross-subdomain auth (auth.latticeforge.ai ↔ latticeforge.ai)
const COOKIE_DOMAIN = '.latticeforge.ai';

// Mock handler for build-time/SSR when env vars aren't available
const mockHandler: ProxyHandler<object> = {
  get(target, prop) {
    // Return mock implementations for common patterns
    if (prop === 'auth') {
      return new Proxy({}, mockHandler);
    }
    if (prop === 'from') {
      return () => new Proxy({}, mockHandler);
    }
    if (prop === 'channel') {
      return () => new Proxy({}, mockHandler);
    }
    if (prop === 'removeChannel') {
      return () => Promise.resolve();
    }
    // For method chains like .select(), .eq(), etc.
    if (typeof prop === 'string') {
      return (..._args: unknown[]) =>
        new Proxy(
          {},
          {
            get(_, p) {
              if (p === 'then') return undefined; // Not a promise
              if (p === 'data') return null;
              if (p === 'error') return null;
              return () => new Proxy({}, mockHandler);
            },
          }
        );
    }
    return undefined;
  },
};

function getSupabase(): BrowserClient {
  // Return cached client if available
  if (_supabase) {
    return _supabase;
  }

  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  // During build/SSR without env vars, return a mock that won't throw
  if (!supabaseUrl || !supabaseAnonKey) {
    if (!isBrowser) {
      // During SSR/build, return a mock that doesn't throw
      return new Proxy({}, mockHandler) as unknown as BrowserClient;
    }
    // On client without env vars (shouldn't happen in production)
    console.error('Supabase environment variables not configured');
    return new Proxy({}, mockHandler) as unknown as BrowserClient;
  }

  // Check if we're in production (latticeforge.ai)
  const isProduction = isBrowser && window.location.hostname.includes('latticeforge.ai');

  // Create browser client from @supabase/ssr - this syncs auth to cookies
  // so middleware can read the session server-side
  _supabase = createBrowserClient(supabaseUrl, supabaseAnonKey, {
    cookieOptions: isProduction
      ? {
          domain: COOKIE_DOMAIN,
          path: '/',
          sameSite: 'lax',
          secure: true,
        }
      : undefined,
  });
  return _supabase;
}

// Export the getter directly
export { getSupabase };

// For backwards compatibility, export a lazy accessor
export const supabase = new Proxy({} as BrowserClient, {
  get(_, prop) {
    const client = getSupabase();
    const value = (client as unknown as Record<string, unknown>)[prop as string];
    if (typeof value === 'function') {
      return value.bind(client);
    }
    return value;
  },
});

// Types for database tables
export interface Database {
  public: {
    Tables: {
      nations: {
        Row: {
          id: string;
          code: string;
          name: string;
          lat: number;
          lon: number;
          position: number[];
          velocity: number[];
          basin_strength: number;
          transition_risk: number;
          regime: number;
          influence_radius: number;
          metadata: Record<string, unknown>;
          created_at: string;
          updated_at: string;
        };
        Insert: Omit<
          Database['public']['Tables']['nations']['Row'],
          'id' | 'created_at' | 'updated_at'
        >;
        Update: Partial<Database['public']['Tables']['nations']['Insert']>;
      };
      influence_edges: {
        Row: {
          id: string;
          source_id: string;
          target_id: string;
          strength: number;
          geodesic_distance: number;
          esteem: number | null;
          simulation_id: string | null;
          created_at: string;
        };
        Insert: Omit<Database['public']['Tables']['influence_edges']['Row'], 'id' | 'created_at'>;
        Update: Partial<Database['public']['Tables']['influence_edges']['Insert']>;
      };
      esteem_relations: {
        Row: {
          id: string;
          source_id: string;
          target_id: string;
          esteem: number;
          created_at: string;
          updated_at: string;
        };
        Insert: Omit<
          Database['public']['Tables']['esteem_relations']['Row'],
          'id' | 'created_at' | 'updated_at'
        >;
        Update: Partial<Database['public']['Tables']['esteem_relations']['Insert']>;
      };
      simulations: {
        Row: {
          id: string;
          name: string | null;
          n_dims: number;
          interaction_decay: number;
          min_influence: number;
          dt: number;
          diffusion: number;
          sim_time: number;
          n_steps: number;
          status: string;
          user_id: string | null;
          metadata: Record<string, unknown>;
          created_at: string;
          updated_at: string;
        };
        Insert: Omit<
          Database['public']['Tables']['simulations']['Row'],
          'id' | 'created_at' | 'updated_at'
        >;
        Update: Partial<Database['public']['Tables']['simulations']['Insert']>;
      };
    };
    Functions: {
      compare_nations: {
        Args: { code1: string; code2: string };
        Returns: Record<string, unknown>;
      };
      nations_within_distance: {
        Args: { center_lat: number; center_lon: number; distance_km: number };
        Returns: Database['public']['Tables']['nations']['Row'][];
      };
    };
  };
}
