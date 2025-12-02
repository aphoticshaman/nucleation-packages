'use client';

import { useState, useEffect, useCallback } from 'react';
import { supabase } from '@/lib/supabase';
import { Nation, InfluenceEdge } from '@/types';

interface UseSupabaseNationsResult {
  nations: Nation[];
  edges: InfluenceEdge[];
  loading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
  updateNation: (code: string, updates: Partial<Nation>) => Promise<void>;
}

export function useSupabaseNations(): UseSupabaseNationsResult {
  const [nations, setNations] = useState<Nation[]>([]);
  const [edges, setEdges] = useState<InfluenceEdge[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchNations = useCallback(async () => {
    try {
      setLoading(true);

      // Fetch nations
      const { data: nationsData, error: nationsError } = await supabase
        .from('nations')
        .select('*')
        .order('code');

      if (nationsError) throw nationsError;

      // Fetch edges with nation codes
      const { data: edgesData, error: edgesError } = await supabase
        .from('influence_edges')
        .select(`
          *,
          source:nations!source_id(code),
          target:nations!target_id(code)
        `)
        .gt('strength', 0.1);

      if (edgesError) throw edgesError;

      setNations(nationsData || []);
      setEdges(
        (edgesData || []).map((e: Record<string, unknown>) => ({
          ...e,
          source_code: (e.source as { code: string })?.code || '',
          target_code: (e.target as { code: string })?.code || '',
        })) as InfluenceEdge[]
      );
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to fetch nations'));
    } finally {
      setLoading(false);
    }
  }, []);

  const updateNation = useCallback(async (code: string, updates: Partial<Nation>) => {
    const { error } = await supabase
      .from('nations')
      .update(updates)
      .eq('code', code);

    if (error) throw error;

    // Optimistic update
    setNations((prev) =>
      prev.map((n) => (n.code === code ? { ...n, ...updates } : n))
    );
  }, []);

  useEffect(() => {
    void fetchNations();

    // Subscribe to realtime changes
    const channel = supabase
      .channel('nations-changes')
      .on(
        'postgres_changes',
        { event: '*', schema: 'public', table: 'nations' },
        () => {
          void fetchNations();
        }
      )
      .subscribe();

    return () => {
      void supabase.removeChannel(channel);
    };
  }, [fetchNations]);

  return {
    nations,
    edges,
    loading,
    error,
    refetch: fetchNations,
    updateNation,
  };
}
