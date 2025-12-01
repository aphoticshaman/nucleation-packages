import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

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
        Insert: Omit<Database['public']['Tables']['nations']['Row'], 'id' | 'created_at' | 'updated_at'>;
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
        Insert: Omit<Database['public']['Tables']['esteem_relations']['Row'], 'id' | 'created_at' | 'updated_at'>;
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
          current_time: number;
          n_steps: number;
          status: string;
          user_id: string | null;
          metadata: Record<string, unknown>;
          created_at: string;
          updated_at: string;
        };
        Insert: Omit<Database['public']['Tables']['simulations']['Row'], 'id' | 'created_at' | 'updated_at'>;
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
