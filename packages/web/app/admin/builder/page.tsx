'use client';

import dynamic from 'next/dynamic';

// Dynamic import to avoid SSR issues with drag-and-drop
const DashboardBuilder = dynamic(
  () => import('@/components/dashboard-builder/DashboardBuilder'),
  {
    ssr: false,
    loading: () => (
      <div className="h-[calc(100vh-8rem)] bg-slate-900 rounded-xl border border-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-4" />
          <p className="text-slate-400">Loading Dashboard Builder...</p>
        </div>
      </div>
    ),
  }
);

export default function DashboardBuilderPage() {
  const handleSave = async (dashboard: Record<string, unknown>) => {
    // TODO: Save to Supabase
    console.log('Saving dashboard:', dashboard);
    // For now, save to localStorage as a demo
    localStorage.setItem('latticeforge_dashboard_draft', JSON.stringify(dashboard));
    alert('Dashboard saved to local storage. Supabase integration coming soon.');
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Dashboard Builder</h1>
          <p className="text-slate-400 mt-1">
            Drag and drop widgets to create custom dashboards for any role or tier
          </p>
        </div>
        <div className="flex gap-2">
          <button className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg text-sm">
            Load Saved
          </button>
          <button className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg text-sm">
            Export JSON
          </button>
        </div>
      </div>

      {/* Builder */}
      <div className="h-[calc(100vh-12rem)]">
        <DashboardBuilder onSave={handleSave} />
      </div>
    </div>
  );
}
