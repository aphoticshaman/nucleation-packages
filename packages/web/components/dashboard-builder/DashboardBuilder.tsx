'use client';

import { useState, useCallback } from 'react';
import type { Widget, Dashboard } from '@/lib/dashboard/types';
import DashboardCanvas from './DashboardCanvas';
import WidgetPalette from './WidgetPalette';
import PropertiesPanel from './PropertiesPanel';
import { Award, DollarSign, FileText } from 'lucide-react';

// Sample templates
const TEMPLATES: { name: string; icon: React.ReactNode; widgets: Widget[] }[] = [
  {
    name: 'Executive Overview',
    icon: <Award className="w-4 h-4" />,
    widgets: [
      { id: 'w1', type: 'stat_card', position: { x: 0, y: 0, w: 3, h: 2 }, title: 'Active Alerts', config: { value: '12', label: 'Critical items', trend: 'up' } },
      { id: 'w2', type: 'stat_card', position: { x: 3, y: 0, w: 3, h: 2 }, title: 'Countries', config: { value: '47', label: 'Monitored', trend: 'neutral' } },
      { id: 'w3', type: 'stat_card', position: { x: 6, y: 0, w: 3, h: 2 }, title: 'Sources', config: { value: '234', label: 'Active feeds', trend: 'up' } },
      { id: 'w4', type: 'stat_card', position: { x: 9, y: 0, w: 3, h: 2 }, title: 'Confidence', config: { value: '94%', label: 'Avg. accuracy', trend: 'up' } },
      { id: 'w5', type: 'signal_ticker', position: { x: 0, y: 2, w: 12, h: 1 } },
      { id: 'w6', type: 'chart_line', position: { x: 0, y: 3, w: 8, h: 4 }, title: 'Risk Trend - 30 Days' },
      { id: 'w7', type: 'alert_feed', position: { x: 8, y: 3, w: 4, h: 4 }, title: 'Priority Alerts' },
    ] as Widget[],
  },
  {
    name: 'Financial Dashboard',
    icon: <DollarSign className="w-4 h-4" />,
    widgets: [
      { id: 'w1', type: 'signal_ticker', position: { x: 0, y: 0, w: 12, h: 1 } },
      { id: 'w2', type: 'chart_line', position: { x: 0, y: 1, w: 6, h: 4 }, title: 'Portfolio Performance' },
      { id: 'w3', type: 'chart_pie', position: { x: 6, y: 1, w: 3, h: 4 }, title: 'Asset Allocation' },
      { id: 'w4', type: 'metric_gauge', position: { x: 9, y: 1, w: 3, h: 2 }, title: 'Risk Score', config: { value: 72, min: 0, max: 100 } },
      { id: 'w5', type: 'table', position: { x: 0, y: 5, w: 12, h: 4 }, title: 'Top Holdings' },
    ] as Widget[],
  },
  {
    name: 'Blank Canvas',
    icon: <FileText className="w-4 h-4" />,
    widgets: [],
  },
];

interface DashboardBuilderProps {
  initialDashboard?: Dashboard;
  onSave?: (dashboard: Partial<Dashboard>) => Promise<void>;
}

export default function DashboardBuilder({ initialDashboard, onSave }: DashboardBuilderProps) {
  const [widgets, setWidgets] = useState<Widget[]>(initialDashboard?.widgets || []);
  const [selectedWidgetId, setSelectedWidgetId] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState(true);
  const [isPaletteExpanded, setIsPaletteExpanded] = useState(true);
  const [dashboardName, setDashboardName] = useState(initialDashboard?.name || 'Untitled Dashboard');
  const [isSaving, setIsSaving] = useState(false);

  const selectedWidget = widgets.find((w) => w.id === selectedWidgetId) || null;

  const handleWidgetAdd = useCallback((widget: Widget) => {
    setWidgets((prev) => [...prev, widget]);
  }, []);

  const handleWidgetUpdate = useCallback((id: string, updates: Partial<Widget>) => {
    setWidgets((prev) =>
      prev.map((w) => (w.id === id ? { ...w, ...updates } : w))
    );
  }, []);

  const handleWidgetDelete = useCallback((id: string) => {
    setWidgets((prev) => prev.filter((w) => w.id !== id));
  }, []);

  const loadTemplate = (template: typeof TEMPLATES[0]) => {
    if (widgets.length > 0 && !confirm('This will replace your current layout. Continue?')) {
      return;
    }
    setWidgets(template.widgets.map(w => ({ ...w, id: `${w.id}-${Date.now()}` })));
    setSelectedWidgetId(null);
  };

  const handleSave = async () => {
    if (!onSave) return;
    setIsSaving(true);
    try {
      await onSave({
        name: dashboardName,
        widgets,
        updatedAt: new Date().toISOString(),
      });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Toolbar */}
      <div className="flex items-center justify-between p-4 bg-slate-900 border-b border-slate-800 rounded-t-xl">
        <div className="flex items-center gap-4">
          <input
            type="text"
            value={dashboardName}
            onChange={(e) => setDashboardName(e.target.value)}
            className="text-lg font-semibold bg-transparent text-white border-b border-transparent hover:border-slate-600 focus:border-blue-500 focus:outline-none px-1"
          />

          {/* Templates dropdown */}
          <div className="relative group">
            <button className="px-3 py-1.5 text-sm text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg">
              Templates
            </button>
            <div className="absolute top-full left-0 mt-1 w-48 bg-slate-800 border border-slate-700 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
              {TEMPLATES.map((template) => (
                <button
                  key={template.name}
                  onClick={() => loadTemplate(template)}
                  className="w-full px-4 py-2 text-left text-sm text-slate-300 hover:bg-slate-700 first:rounded-t-lg last:rounded-b-lg flex items-center gap-2"
                >
                  {template.icon}
                  {template.name}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsEditing(!isEditing)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              isEditing
                ? 'bg-blue-600 text-white'
                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
            }`}
          >
            {isEditing ? 'Editing' : 'Preview'}
          </button>

          {onSave && (
            <button
              onClick={() => void handleSave()}
              disabled={isSaving}
              className="px-4 py-2 bg-green-600 hover:bg-green-500 disabled:bg-green-800 text-white rounded-lg text-sm font-medium"
            >
              {isSaving ? 'Saving...' : 'Save Dashboard'}
            </button>
          )}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex gap-4 p-4 bg-slate-950 overflow-hidden">
        {/* Widget palette (left) */}
        {isEditing && (
          <WidgetPalette
            isExpanded={isPaletteExpanded}
            onToggle={() => setIsPaletteExpanded(!isPaletteExpanded)}
          />
        )}

        {/* Canvas (center) */}
        <div className="flex-1 overflow-auto">
          <DashboardCanvas
            widgets={widgets}
            onWidgetAdd={handleWidgetAdd}
            onWidgetUpdate={handleWidgetUpdate}
            onWidgetDelete={handleWidgetDelete}
            onWidgetSelect={setSelectedWidgetId}
            selectedWidgetId={selectedWidgetId}
            isEditing={isEditing}
          />
        </div>

        {/* Properties panel (right) */}
        {isEditing && (
          <PropertiesPanel
            widget={selectedWidget}
            onUpdate={(updates) => selectedWidgetId && handleWidgetUpdate(selectedWidgetId, updates)}
            onClose={() => setSelectedWidgetId(null)}
          />
        )}
      </div>

      {/* Status bar */}
      <div className="px-4 py-2 bg-slate-900 border-t border-slate-800 rounded-b-xl flex items-center justify-between text-xs text-slate-500">
        <span>{widgets.length} widgets</span>
        <span>Drag widgets from the palette • Click to select • Press Delete to remove</span>
        <span>{isEditing ? 'Edit mode' : 'Preview mode'}</span>
      </div>
    </div>
  );
}
