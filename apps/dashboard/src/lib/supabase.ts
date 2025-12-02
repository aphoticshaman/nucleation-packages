import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  console.warn('Supabase credentials not configured');
}

export const supabase = createClient(
  supabaseUrl || 'http://localhost:54321',
  supabaseAnonKey || 'dummy-key'
);

// API client that uses the Edge Function
export const api = {
  baseUrl: import.meta.env.VITE_API_URL || `${supabaseUrl}/functions/v1/api`,

  async request<T>(path: string, options: RequestInit = {}): Promise<T> {
    const {
      data: { session },
    } = await supabase.auth.getSession();

    // Get API key from localStorage or use session token
    const apiKey = localStorage.getItem('lf_api_key');

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string>),
    };

    if (apiKey) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    } else if (session?.access_token) {
      headers['Authorization'] = `Bearer ${session.access_token}`;
    }

    const response = await fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Request failed' }));
      throw new Error(error.error || 'Request failed');
    }

    return response.json();
  },

  // Convenience methods
  async fuse(sources: string[]) {
    return this.request('/signals/fuse', {
      method: 'POST',
      body: JSON.stringify({ sources }),
    });
  },

  async detect() {
    return this.request('/signals/detect');
  },

  async usage() {
    return this.request('/usage');
  },

  async indicator(name: string) {
    return this.request(`/indicators/${name}`);
  },

  async verify() {
    return this.request('/verify');
  },
};

// Auth helpers
export const auth = {
  async signUp(email: string, password: string) {
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
    });
    if (error) throw error;
    return data;
  },

  async signIn(email: string, password: string) {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });
    if (error) throw error;
    return data;
  },

  async signInWithMagicLink(email: string) {
    const { error } = await supabase.auth.signInWithOtp({
      email,
      options: {
        emailRedirectTo: `${window.location.origin}/auth/callback`,
      },
    });
    if (error) throw error;
  },

  async signOut() {
    const { error } = await supabase.auth.signOut();
    if (error) throw error;
    localStorage.removeItem('lf_api_key');
  },

  async getSession() {
    const {
      data: { session },
    } = await supabase.auth.getSession();
    return session;
  },

  onAuthStateChange(callback: (event: string, session: any) => void) {
    return supabase.auth.onAuthStateChange(callback);
  },
};

// Client/API key management
export const client = {
  async getClient() {
    const { data, error } = await supabase.from('clients').select('*').single();
    if (error) throw error;
    return data;
  },

  async getApiKeys() {
    const { data, error } = await supabase
      .from('api_keys')
      .select('*')
      .order('created_at', { ascending: false });
    if (error) throw error;
    return data;
  },

  async createApiKey(name: string = 'Default') {
    const clientData = await this.getClient();
    const { data, error } = await supabase.rpc('generate_api_key', {
      p_client_id: clientData.id,
      p_name: name,
    });
    if (error) throw error;
    return data[0]; // { key, key_id }
  },

  async revokeApiKey(keyId: string) {
    const { error } = await supabase.from('api_keys').update({ status: 'revoked' }).eq('id', keyId);
    if (error) throw error;
  },

  async getUsage() {
    const { data, error } = await supabase
      .from('usage_summary')
      .select('*')
      .order('billing_period', { ascending: false })
      .limit(12);
    if (error) throw error;
    return data;
  },
};
