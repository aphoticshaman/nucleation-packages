'use client';

import dynamic from 'next/dynamic';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';
import { Upload, Download } from 'lucide-react';

// Dynamic import to avoid SSR issues with drag-and-drop
const DashboardBuilder = dynamic(
  () => import('@/components/dashboard-builder/DashboardBuilder'),
  {
    ssr: false,
    loading: () => (
      <GlassCard blur="heavy" className="h-[calc(100vh-8rem)] flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-4" />
          <p className="text-slate-400">Loading Dashboard Builder...</p>
        </div>
      </GlassCard>
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

  const handleLoad = () => {
    const saved = localStorage.getItem('latticeforge_dashboard_draft');
    if (saved) {
      alert('Dashboard loaded from local storage');
    } else {
      alert('No saved dashboard found');
    }
  };

  const handleExport = () => {
    const saved = localStorage.getItem('latticeforge_dashboard_draft');
    if (saved) {
      const blob = new Blob([saved], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'dashboard-config.json';
      a.click();
      URL.revokeObjectURL(url);
    } else {
      alert('No dashboard to export');
    }
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Dashboard Builder</h1>
          <p className="text-slate-400 mt-1">
            Drag and drop widgets to create custom dashboards for any role or tier
          </p>
        </div>
        <div className="flex gap-2">
          <GlassButton variant="secondary" size="sm" onClick={handleLoad}>
            <Upload className="w-4 h-4 mr-2" />
            Load Saved
          </GlassButton>
          <GlassButton variant="secondary" size="sm" onClick={handleExport}>
            <Download className="w-4 h-4 mr-2" />
            Export JSON
          </GlassButton>
        </div>
      </div>

      {/* Builder */}
      <div className="h-[calc(100vh-12rem)]">
        <DashboardBuilder onSave={handleSave} />
      </div>
    </div>
  );
}
