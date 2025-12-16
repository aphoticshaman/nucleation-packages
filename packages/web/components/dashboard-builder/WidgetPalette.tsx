'use client';

import type { DragEvent } from 'react';
import { WIDGET_CATALOG, type WidgetCatalogItem } from '@/lib/dashboard/types';
import { LayoutDashboard, TrendingUp, FileText, Award, Layers } from 'lucide-react';

const CATEGORIES = [
  { id: 'data', name: 'Data', icon: <LayoutDashboard className="w-4 h-4" /> },
  { id: 'visualization', name: 'Charts', icon: <TrendingUp className="w-4 h-4" /> },
  { id: 'content', name: 'Content', icon: <FileText className="w-4 h-4" /> },
  { id: 'intelligence', name: 'Intel', icon: <Award className="w-4 h-4" /> },
] as const;

interface WidgetPaletteProps {
  isExpanded: boolean;
  onToggle: () => void;
}

export default function WidgetPalette({ isExpanded, onToggle }: WidgetPaletteProps) {
  const handleDragStart = (e: DragEvent<HTMLDivElement>, item: WidgetCatalogItem) => {
    e.dataTransfer.setData('application/json', JSON.stringify(item));
    e.dataTransfer.effectAllowed = 'copy';
  };

  return (
    <div className={`bg-slate-900 border border-slate-800 rounded-xl transition-all ${
      isExpanded ? 'w-64' : 'w-14'
    }`}>
      {/* Toggle header */}
      <button
        onClick={onToggle}
        className="w-full p-4 flex items-center gap-3 border-b border-slate-800 hover:bg-slate-800/50"
      >
        <Layers className="w-5 h-5 text-slate-400" />
        {isExpanded && <span className="text-sm font-medium text-white">Widgets</span>}
      </button>

      {/* Widget categories */}
      {isExpanded ? (
        <div className="p-2 max-h-[calc(100vh-200px)] overflow-y-auto">
          {CATEGORIES.map((category) => (
            <div key={category.id} className="mb-4">
              <p className="text-xs text-slate-500 uppercase tracking-wider px-2 mb-2 flex items-center gap-2">
                {category.icon} {category.name}
              </p>
              <div className="space-y-1">
                {WIDGET_CATALOG.filter((w) => w.category === category.id).map((item) => (
                  <div
                    key={item.type}
                    draggable
                    onDragStart={(e) => handleDragStart(e, item)}
                    className="p-3 bg-slate-800 hover:bg-slate-700 rounded-lg cursor-grab active:cursor-grabbing transition-colors group"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{item.icon}</span>
                      <div>
                        <p className="text-sm text-white">{item.name}</p>
                        <p className="text-xs text-slate-400 group-hover:text-slate-300">
                          {item.description}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="p-2 space-y-2">
          {CATEGORIES.map((category) => (
            <button
              key={category.id}
              onClick={onToggle}
              className="w-10 h-10 flex items-center justify-center bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-white"
              title={category.name}
            >
              {category.icon}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
