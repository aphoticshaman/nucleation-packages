'use client';

import { useState, useCallback, type DragEvent } from 'react';
import type { Widget, WidgetPosition, WidgetCatalogItem } from '@/lib/dashboard/types';
import WidgetRenderer from './WidgetRenderer';

interface DashboardCanvasProps {
  widgets: Widget[];
  onWidgetAdd: (widget: Widget) => void;
  onWidgetUpdate: (id: string, updates: Partial<Widget>) => void;
  onWidgetDelete: (id: string) => void;
  onWidgetSelect: (id: string | null) => void;
  selectedWidgetId: string | null;
  isEditing: boolean;
  gridColumns?: number;
  rowHeight?: number;
  gap?: number;
}

export default function DashboardCanvas({
  widgets,
  onWidgetAdd,
  onWidgetUpdate,
  onWidgetDelete,
  onWidgetSelect,
  selectedWidgetId,
  isEditing,
  gridColumns = 12,
  rowHeight = 80,
  gap = 16,
}: DashboardCanvasProps) {
  const [dragOverCell, setDragOverCell] = useState<{ x: number; y: number } | null>(null);

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!isEditing) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * gridColumns);
    const y = Math.floor((e.clientY - rect.top) / (rowHeight + gap));

    setDragOverCell({ x: Math.min(x, gridColumns - 1), y });
  }, [isEditing, gridColumns, rowHeight, gap]);

  const handleDragLeave = useCallback(() => {
    setDragOverCell(null);
  }, []);

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!isEditing || !dragOverCell) return;

    const widgetData = e.dataTransfer.getData('application/json');
    if (!widgetData) return;

    try {
      const catalogItem: WidgetCatalogItem = JSON.parse(widgetData);

      // Create new widget
      const newWidget: Widget = {
        id: `widget-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        type: catalogItem.type,
        position: {
          x: dragOverCell.x,
          y: dragOverCell.y,
          w: Math.min(catalogItem.defaultSize.w, gridColumns - dragOverCell.x),
          h: catalogItem.defaultSize.h,
        },
        title: catalogItem.name,
        config: catalogItem.defaultConfig,
      } as Widget;

      onWidgetAdd(newWidget);
    } catch (err) {
      console.error('Failed to parse widget data:', err);
    }

    setDragOverCell(null);
  }, [isEditing, dragOverCell, gridColumns, onWidgetAdd]);

  // Calculate total rows needed
  const maxRow = widgets.reduce((max, w) => Math.max(max, w.position.y + w.position.h), 6);

  return (
    <div
      className={`relative bg-slate-950 rounded-xl border ${
        isEditing ? 'border-blue-500/50' : 'border-slate-800'
      } overflow-hidden`}
      style={{
        minHeight: `${maxRow * (rowHeight + gap)}px`,
      }}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Grid overlay when editing */}
      {isEditing && (
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            backgroundImage: `
              linear-gradient(to right, rgba(59, 130, 246, 0.1) 1px, transparent 1px),
              linear-gradient(to bottom, rgba(59, 130, 246, 0.1) 1px, transparent 1px)
            `,
            backgroundSize: `${100 / gridColumns}% ${rowHeight + gap}px`,
          }}
        />
      )}

      {/* Drop indicator */}
      {isEditing && dragOverCell && (
        <div
          className="absolute bg-blue-500/20 border-2 border-dashed border-blue-500 rounded-lg pointer-events-none transition-all"
          style={{
            left: `${(dragOverCell.x / gridColumns) * 100}%`,
            top: `${dragOverCell.y * (rowHeight + gap)}px`,
            width: `${(3 / gridColumns) * 100}%`,
            height: `${2 * rowHeight + gap}px`,
          }}
        />
      )}

      {/* Widgets */}
      <div className="relative p-4">
        {widgets.map((widget) => (
          <div
            key={widget.id}
            className={`absolute transition-all ${
              isEditing ? 'cursor-move' : ''
            } ${
              selectedWidgetId === widget.id
                ? 'ring-2 ring-blue-500 z-10'
                : ''
            }`}
            style={{
              left: `calc(${(widget.position.x / gridColumns) * 100}% + ${gap / 2}px)`,
              top: `${widget.position.y * (rowHeight + gap) + gap / 2}px`,
              width: `calc(${(widget.position.w / gridColumns) * 100}% - ${gap}px)`,
              height: `${widget.position.h * rowHeight + (widget.position.h - 1) * gap}px`,
            }}
            onClick={() => isEditing && onWidgetSelect(widget.id)}
            draggable={isEditing}
            onDragStart={(e) => {
              if (!isEditing) return;
              e.dataTransfer.setData('widget-move', widget.id);
            }}
          >
            <WidgetRenderer widget={widget} isEditing={isEditing} />

            {/* Delete button when selected */}
            {isEditing && selectedWidgetId === widget.id && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onWidgetDelete(widget.id);
                  onWidgetSelect(null);
                }}
                className="absolute -top-2 -right-2 w-6 h-6 bg-red-600 hover:bg-red-500 text-white rounded-full flex items-center justify-center text-xs z-20"
              >
                x
              </button>
            )}
          </div>
        ))}
      </div>

      {/* Empty state */}
      {widgets.length === 0 && isEditing && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <p className="text-slate-400 mb-2">Drag widgets from the palette to get started</p>
            <p className="text-sm text-slate-600">or click a template to load a preset layout</p>
          </div>
        </div>
      )}
    </div>
  );
}
