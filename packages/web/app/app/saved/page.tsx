import { createClient, requireConsumer } from '@/lib/auth';

export default async function SavedSimulationsPage() {
  const user = await requireConsumer();
  const supabase = await createClient();

  // Get user's saved simulations
  const { data: simulations } = await supabase
    .from('saved_simulations')
    .select('*')
    .eq('user_id', user.id)
    .order('updated_at', { ascending: false });

  const maxSlots = 5; // Consumer limit
  const usedSlots = simulations?.length || 0;

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">My Simulations</h1>
          <p className="text-slate-400 mt-1">
            {usedSlots} of {maxSlots} save slots used
          </p>
        </div>
        {usedSlots < maxSlots && (
          <a
            href="/app"
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition-colors"
          >
            New Simulation
          </a>
        )}
      </div>

      {/* Simulations grid */}
      {simulations && simulations.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {simulations.map((sim) => (
            <div
              key={sim.id}
              className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden hover:border-slate-700 transition-colors"
            >
              {/* Preview placeholder */}
              <div className="h-40 bg-slate-800 flex items-center justify-center">
                <span className="text-4xl">🗺️</span>
              </div>

              <div className="p-4">
                <h3 className="font-medium text-white">{sim.name}</h3>
                {sim.description && (
                  <p className="text-sm text-slate-400 mt-1 line-clamp-2">
                    {sim.description}
                  </p>
                )}
                <div className="flex items-center justify-between mt-4">
                  <span className="text-xs text-slate-500">
                    {new Date(sim.updated_at).toLocaleDateString()}
                  </span>
                  <div className="flex gap-2">
                    <button className="px-3 py-1 text-sm bg-slate-800 text-white rounded hover:bg-slate-700">
                      Load
                    </button>
                    <button className="px-3 py-1 text-sm bg-slate-800 text-slate-400 rounded hover:bg-slate-700">
                      Delete
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))}

          {/* Empty slots */}
          {Array.from({ length: maxSlots - usedSlots }).map((_, i) => (
            <div
              key={`empty-${i}`}
              className="bg-slate-900/50 rounded-xl border border-dashed border-slate-800 h-64 flex items-center justify-center"
            >
              <div className="text-center">
                <p className="text-slate-500">Empty slot</p>
                <a
                  href="/app"
                  className="text-sm text-blue-400 hover:text-blue-300 mt-2 block"
                >
                  Create simulation
                </a>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-12 text-center">
          <span className="text-4xl">📭</span>
          <h3 className="text-lg font-medium text-white mt-4">No saved simulations</h3>
          <p className="text-slate-400 mt-2">
            Run a simulation and save it to access it later
          </p>
          <a
            href="/app"
            className="inline-block mt-6 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500"
          >
            Start Exploring
          </a>
        </div>
      )}

      {/* Upgrade prompt */}
      <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-xl border border-blue-800/50 p-6 flex items-center justify-between">
        <div>
          <h3 className="font-medium text-white">Need more storage?</h3>
          <p className="text-sm text-slate-300 mt-1">
            Enterprise accounts get unlimited simulation saves
          </p>
        </div>
        <a
          href="/pricing"
          className="px-4 py-2 bg-white text-slate-900 rounded-lg font-medium hover:bg-slate-100"
        >
          Upgrade
        </a>
      </div>
    </div>
  );
}
