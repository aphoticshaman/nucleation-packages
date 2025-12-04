'use client';

import { useState, useEffect } from 'react';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';
import { supabase } from '@/lib/supabase';

interface Simulation {
  id: string;
  name: string;
  description?: string;
  updated_at: string;
  created_at: string;
}

export default function SavedSimulationsPage() {
  const [simulations, setSimulations] = useState<Simulation[]>([]);
  const [loading, setLoading] = useState(true);
  const [userRole, setUserRole] = useState<string>('consumer');
  const [deleting, setDeleting] = useState<string | null>(null);

  const maxSlots = userRole === 'admin' ? 999 : 5; // Admin gets unlimited
  const usedSlots = simulations.length;

  useEffect(() => {
    async function fetchData() {
      try {
        const { data: { user } } = await supabase.auth.getUser();
        if (!user) return;

        // Get user role
        const { data: profile } = await supabase
          .from('profiles')
          .select('role')
          .eq('id', user.id)
          .single();

        if (profile) {
          setUserRole((profile as { role?: string }).role || 'consumer');
        }

        // Get simulations
        const { data: sims } = await supabase
          .from('saved_simulations')
          .select('*')
          .eq('user_id', user.id)
          .order('updated_at', { ascending: false });

        if (sims) {
          setSimulations(sims as Simulation[]);
        }
      } catch (err) {
        console.error('Failed to fetch saved simulations:', err);
      } finally {
        setLoading(false);
      }
    }
    void fetchData();
  }, []);

  const handleDelete = async (id: string) => {
    setDeleting(id);
    try {
      await supabase.from('saved_simulations').delete().eq('id', id);
      setSimulations(simulations.filter(s => s.id !== id));
    } catch (err) {
      console.error('Failed to delete simulation:', err);
    } finally {
      setDeleting(null);
    }
  };

  const isAdmin = userRole === 'admin';

  if (loading) {
    return (
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <div className="h-8 w-48 bg-white/10 rounded animate-pulse" />
            <div className="h-4 w-32 bg-white/5 rounded mt-2 animate-pulse" />
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3].map(i => (
            <GlassCard key={i} className="h-64 animate-pulse">
              <div className="h-full bg-white/5 rounded" />
            </GlassCard>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">My Simulations</h1>
          <p className="text-slate-400 mt-1">
            {isAdmin ? (
              <span className="text-amber-400">Admin - Unlimited storage</span>
            ) : (
              `${usedSlots} of ${maxSlots} save slots used`
            )}
          </p>
        </div>
        {(isAdmin || usedSlots < maxSlots) && (
          <GlassButton
            variant="primary"
            glow
            onClick={() => window.location.href = '/app'}
          >
            New Simulation
          </GlassButton>
        )}
      </div>

      {/* Simulations grid */}
      {simulations.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {simulations.map((sim) => (
            <GlassCard
              key={sim.id}
              interactive
              className="overflow-hidden"
            >
              {/* Preview placeholder */}
              <div className="h-40 bg-gradient-to-br from-slate-800/50 to-slate-900/50 flex items-center justify-center border-b border-white/[0.06] -m-4 sm:-m-6 mb-4">
                <span className="text-4xl">🗺️</span>
              </div>

              <div className="pt-4">
                <h3 className="font-medium text-white">{sim.name}</h3>
                {sim.description && (
                  <p className="text-sm text-slate-400 mt-1 line-clamp-2">{sim.description}</p>
                )}
                <div className="flex items-center justify-between mt-4">
                  <span className="text-xs text-slate-500">
                    {new Date(sim.updated_at).toLocaleDateString()}
                  </span>
                  <div className="flex gap-2">
                    <GlassButton
                      variant="secondary"
                      size="sm"
                      onClick={() => window.location.href = `/app?load=${sim.id}`}
                    >
                      Load
                    </GlassButton>
                    <GlassButton
                      variant="ghost"
                      size="sm"
                      loading={deleting === sim.id}
                      onClick={() => handleDelete(sim.id)}
                    >
                      Delete
                    </GlassButton>
                  </div>
                </div>
              </div>
            </GlassCard>
          ))}

          {/* Empty slots - only show for non-admins */}
          {!isAdmin && Array.from({ length: maxSlots - usedSlots }).map((_, i) => (
            <div
              key={`empty-${i}`}
              className="bg-[rgba(18,18,26,0.4)] backdrop-blur-sm rounded-xl border-2 border-dashed border-white/[0.08] h-64 flex items-center justify-center"
            >
              <div className="text-center">
                <p className="text-slate-500">Empty slot</p>
                <a href="/app" className="text-sm text-blue-400 hover:text-blue-300 mt-2 block">
                  Create simulation
                </a>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <GlassCard className="p-12 text-center">
          <span className="text-4xl">📭</span>
          <h3 className="text-lg font-medium text-white mt-4">No saved simulations</h3>
          <p className="text-slate-400 mt-2">Run a simulation and save it to access it later</p>
          <GlassButton
            variant="primary"
            glow
            className="mt-6"
            onClick={() => window.location.href = '/app'}
          >
            Start Exploring
          </GlassButton>
        </GlassCard>
      )}

      {/* Upgrade prompt - only for non-admins */}
      {!isAdmin && (
        <GlassCard accent glow className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div>
            <h3 className="font-medium text-white">Need more storage?</h3>
            <p className="text-sm text-slate-300 mt-1">
              Enterprise accounts get unlimited simulation saves
            </p>
          </div>
          <GlassButton
            variant="primary"
            onClick={() => window.location.href = '/pricing'}
          >
            Upgrade
          </GlassButton>
        </GlassCard>
      )}
    </div>
  );
}
