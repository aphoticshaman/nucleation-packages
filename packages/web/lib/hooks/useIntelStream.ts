'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { createBrowserClient } from '@supabase/ssr';
import type { RealtimeChannel } from '@supabase/supabase-js';
import type { IntelPacket, ContextLayer } from '@/lib/types/causal';

interface UseIntelStreamOptions {
  /** Maximum number of items to keep in memory */
  maxItems?: number;
  /** Filter by context layer */
  contextFilter?: ContextLayer;
  /** Filter by category */
  categoryFilter?: string;
  /** Auto-start streaming on mount */
  autoStart?: boolean;
  /** Table to subscribe to (default: 'briefings') */
  table?: string;
}

interface UseIntelStreamReturn {
  /** Current stream of intel packets (newest first) */
  packets: IntelPacket[];
  /** Whether the stream is currently active */
  isLive: boolean;
  /** Start/resume the stream */
  start: () => void;
  /** Pause the stream (keeps data) */
  pause: () => void;
  /** Clear all packets */
  clear: () => void;
  /** Connection status */
  status: 'connecting' | 'connected' | 'disconnected' | 'error';
  /** Error message if any */
  error: string | null;
}

/**
 * useIntelStream - Real-time intelligence packet streaming via Supabase Realtime
 *
 * Subscribes to database changes and streams new intel packets as they arrive.
 * Supports filtering by context layer and category.
 *
 * Usage:
 * ```tsx
 * const { packets, isLive, start, pause } = useIntelStream({
 *   contextFilter: ContextLayer.SURFACE,
 *   maxItems: 50,
 * });
 * ```
 */
export function useIntelStream({
  maxItems = 100,
  contextFilter,
  categoryFilter,
  autoStart = true,
  table = 'briefings',
}: UseIntelStreamOptions = {}): UseIntelStreamReturn {
  const [packets, setPackets] = useState<IntelPacket[]>([]);
  const [isLive, setIsLive] = useState(false);
  const [status, setStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>(
    'disconnected'
  );
  const [error, setError] = useState<string | null>(null);

  const channelRef = useRef<RealtimeChannel | null>(null);
  const supabaseRef = useRef<ReturnType<typeof createBrowserClient> | null>(null);

  // Initialize Supabase client
  useEffect(() => {
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

    if (!supabaseUrl || !supabaseKey) {
      setError('Missing Supabase configuration');
      setStatus('error');
      return;
    }

    supabaseRef.current = createBrowserClient(supabaseUrl, supabaseKey);
  }, []);

  // Transform database row to IntelPacket
  const transformRow = useCallback(
    (row: Record<string, unknown>): IntelPacket | null => {
      try {
        const packet: IntelPacket = {
          id: String(row.id || row.uuid || crypto.randomUUID()),
          timestamp: String(row.created_at || row.timestamp || new Date().toISOString()),
          context: (row.context_layer as ContextLayer) || 'SURFACE',
          category: String(row.category || 'GENERAL'),
          header: String(row.title || row.header || 'Untitled'),
          summary: String(row.summary || row.description || ''),
          body: String(row.content || row.body || ''),
          coherence: Number(row.confidence || row.coherence || 0.8),
          source: String(row.source || 'System'),
        };

        // Apply filters
        if (contextFilter && packet.context !== contextFilter) {
          return null;
        }
        if (categoryFilter && packet.category !== categoryFilter) {
          return null;
        }

        return packet;
      } catch {
        console.error('Failed to transform row:', row);
        return null;
      }
    },
    [contextFilter, categoryFilter]
  );

  // Start streaming
  const start = useCallback(() => {
    if (!supabaseRef.current || channelRef.current) return;

    setStatus('connecting');
    setError(null);

    try {
      const channel = supabaseRef.current
        .channel(`intel-stream-${table}`)
        .on(
          'postgres_changes',
          {
            event: 'INSERT',
            schema: 'public',
            table: table,
          },
          (payload) => {
            const packet = transformRow(payload.new);
            if (packet) {
              setPackets((prev) => {
                const updated = [packet, ...prev];
                return updated.slice(0, maxItems);
              });
            }
          }
        )
        .on(
          'postgres_changes',
          {
            event: 'UPDATE',
            schema: 'public',
            table: table,
          },
          (payload) => {
            const packet = transformRow(payload.new);
            if (packet) {
              setPackets((prev) =>
                prev.map((p) => (p.id === packet.id ? packet : p))
              );
            }
          }
        )
        .subscribe((status) => {
          if (status === 'SUBSCRIBED') {
            setStatus('connected');
            setIsLive(true);
          } else if (status === 'CLOSED') {
            setStatus('disconnected');
            setIsLive(false);
          } else if (status === 'CHANNEL_ERROR') {
            setStatus('error');
            setError('Channel error');
            setIsLive(false);
          }
        });

      channelRef.current = channel;
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, [table, maxItems, transformRow]);

  // Pause streaming
  const pause = useCallback(() => {
    if (channelRef.current && supabaseRef.current) {
      supabaseRef.current.removeChannel(channelRef.current);
      channelRef.current = null;
    }
    setIsLive(false);
    setStatus('disconnected');
  }, []);

  // Clear packets
  const clear = useCallback(() => {
    setPackets([]);
  }, []);

  // Auto-start on mount
  useEffect(() => {
    if (autoStart) {
      start();
    }

    return () => {
      if (channelRef.current && supabaseRef.current) {
        supabaseRef.current.removeChannel(channelRef.current);
      }
    };
  }, [autoStart, start]);

  // Fetch initial data
  useEffect(() => {
    if (!supabaseRef.current) return;

    const fetchInitial = async () => {
      try {
        let query = supabaseRef.current!
          .from(table)
          .select('*')
          .order('created_at', { ascending: false })
          .limit(maxItems);

        if (contextFilter) {
          query = query.eq('context_layer', contextFilter);
        }
        if (categoryFilter) {
          query = query.eq('category', categoryFilter);
        }

        const { data, error: fetchError } = await query;

        if (fetchError) {
          console.error('Failed to fetch initial data:', fetchError);
          return;
        }

        if (data) {
          const transformed = data
            .map(transformRow)
            .filter((p): p is IntelPacket => p !== null);
          setPackets(transformed);
        }
      } catch (err) {
        console.error('Failed to fetch initial data:', err);
      }
    };

    fetchInitial();
  }, [table, maxItems, contextFilter, categoryFilter, transformRow]);

  return {
    packets,
    isLive,
    start,
    pause,
    clear,
    status,
    error,
  };
}

export default useIntelStream;
