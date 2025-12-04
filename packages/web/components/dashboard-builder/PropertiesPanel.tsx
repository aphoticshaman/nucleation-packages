'use client';

import type { Widget, WidgetPosition } from '@/lib/dashboard/types';

interface PropertiesPanelProps {
  widget: Widget | null;
  onUpdate: (updates: Partial<Widget>) => void;
  onClose: () => void;
}

export default function PropertiesPanel({ widget, onUpdate, onClose }: PropertiesPanelProps) {
  if (!widget) {
    return (
      <div className="w-72 bg-slate-900 border border-slate-800 rounded-xl p-4">
        <p className="text-slate-400 text-sm text-center py-8">
          Select a widget to edit its properties
        </p>
      </div>
    );
  }

  const updatePosition = (key: keyof WidgetPosition, value: number) => {
    onUpdate({
      position: {
        ...widget.position,
        [key]: value,
      },
    });
  };

  return (
    <div className="w-72 bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        <h3 className="text-sm font-medium text-white">Properties</h3>
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-white text-sm"
        >
          Done
        </button>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4 max-h-[calc(100vh-200px)] overflow-y-auto">
        {/* Widget type */}
        <div>
          <label className="text-xs text-slate-400 uppercase tracking-wider">Type</label>
          <p className="text-white text-sm mt-1 capitalize">{widget.type.replace('_', ' ')}</p>
        </div>

        {/* Title */}
        <div>
          <label className="text-xs text-slate-400 uppercase tracking-wider">Title</label>
          <input
            type="text"
            value={widget.title || ''}
            onChange={(e) => onUpdate({ title: e.target.value })}
            className="w-full mt-1 px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500"
            placeholder="Widget title"
          />
        </div>

        {/* Position */}
        <div>
          <label className="text-xs text-slate-400 uppercase tracking-wider">Position & Size</label>
          <div className="grid grid-cols-2 gap-2 mt-2">
            <div>
              <label className="text-xs text-slate-500">Column</label>
              <input
                type="number"
                value={widget.position.x}
                onChange={(e) => updatePosition('x', Math.max(0, Math.min(11, parseInt(e.target.value) || 0)))}
                min={0}
                max={11}
                className="w-full px-2 py-1 bg-slate-800 border border-slate-700 rounded text-white text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-slate-500">Row</label>
              <input
                type="number"
                value={widget.position.y}
                onChange={(e) => updatePosition('y', Math.max(0, parseInt(e.target.value) || 0))}
                min={0}
                className="w-full px-2 py-1 bg-slate-800 border border-slate-700 rounded text-white text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-slate-500">Width</label>
              <input
                type="number"
                value={widget.position.w}
                onChange={(e) => updatePosition('w', Math.max(1, Math.min(12, parseInt(e.target.value) || 1)))}
                min={1}
                max={12}
                className="w-full px-2 py-1 bg-slate-800 border border-slate-700 rounded text-white text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-slate-500">Height</label>
              <input
                type="number"
                value={widget.position.h}
                onChange={(e) => updatePosition('h', Math.max(1, parseInt(e.target.value) || 1))}
                min={1}
                className="w-full px-2 py-1 bg-slate-800 border border-slate-700 rounded text-white text-sm"
              />
            </div>
          </div>
        </div>

        {/* Widget-specific config */}
        {widget.type === 'stat_card' && (
          <StatCardConfig
            config={(widget as { config: { value: string; label: string } }).config}
            onUpdate={(config) => onUpdate({ config } as Partial<Widget>)}
          />
        )}

        {widget.type === 'text' && (
          <TextConfig
            config={(widget as { config: { content: string; format: string } }).config}
            onUpdate={(config) => onUpdate({ config } as Partial<Widget>)}
          />
        )}

        {widget.type === 'embed' && (
          <EmbedConfig
            config={(widget as { config: { url: string } }).config}
            onUpdate={(config) => onUpdate({ config } as Partial<Widget>)}
          />
        )}

        {/* Data source */}
        <div>
          <label className="text-xs text-slate-400 uppercase tracking-wider">Data Source</label>
          <select
            className="w-full mt-1 px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm"
            value={widget.dataSource?.type || 'static'}
            onChange={(e) => onUpdate({
              dataSource: { ...widget.dataSource, type: e.target.value as 'static' | 'api' | 'supabase' }
            })}
          >
            <option value="static">Static (hardcoded)</option>
            <option value="api">API Endpoint</option>
            <option value="supabase">Supabase Table</option>
            <option value="realtime">Real-time Feed</option>
          </select>
        </div>
      </div>
    </div>
  );
}

// Widget-specific config components
function StatCardConfig({
  config,
  onUpdate
}: {
  config: { value: string | number; label: string; icon?: string; trend?: string };
  onUpdate: (config: Record<string, unknown>) => void;
}) {
  return (
    <div className="space-y-3 pt-3 border-t border-slate-800">
      <div>
        <label className="text-xs text-slate-500">Value</label>
        <input
          type="text"
          value={String(config.value)}
          onChange={(e) => onUpdate({ ...config, value: e.target.value })}
          className="w-full px-2 py-1 bg-slate-800 border border-slate-700 rounded text-white text-sm"
        />
      </div>
      <div>
        <label className="text-xs text-slate-500">Label</label>
        <input
          type="text"
          value={config.label}
          onChange={(e) => onUpdate({ ...config, label: e.target.value })}
          className="w-full px-2 py-1 bg-slate-800 border border-slate-700 rounded text-white text-sm"
        />
      </div>
      <div>
        <label className="text-xs text-slate-500">Icon (emoji)</label>
        <input
          type="text"
          value={config.icon || ''}
          onChange={(e) => onUpdate({ ...config, icon: e.target.value })}
          className="w-full px-2 py-1 bg-slate-800 border border-slate-700 rounded text-white text-sm"
          placeholder="📊"
        />
      </div>
      <div>
        <label className="text-xs text-slate-500">Trend</label>
        <select
          value={config.trend || 'neutral'}
          onChange={(e) => onUpdate({ ...config, trend: e.target.value })}
          className="w-full px-2 py-1 bg-slate-800 border border-slate-700 rounded text-white text-sm"
        >
          <option value="up">Up (green)</option>
          <option value="down">Down (red)</option>
          <option value="neutral">Neutral</option>
        </select>
      </div>
    </div>
  );
}

function TextConfig({
  config,
  onUpdate
}: {
  config: { content: string; format: string };
  onUpdate: (config: Record<string, unknown>) => void;
}) {
  return (
    <div className="space-y-3 pt-3 border-t border-slate-800">
      <div>
        <label className="text-xs text-slate-500">Content</label>
        <textarea
          value={config.content}
          onChange={(e) => onUpdate({ ...config, content: e.target.value })}
          rows={4}
          className="w-full px-2 py-1 bg-slate-800 border border-slate-700 rounded text-white text-sm resize-none"
        />
      </div>
      <div>
        <label className="text-xs text-slate-500">Format</label>
        <select
          value={config.format}
          onChange={(e) => onUpdate({ ...config, format: e.target.value })}
          className="w-full px-2 py-1 bg-slate-800 border border-slate-700 rounded text-white text-sm"
        >
          <option value="plain">Plain text</option>
          <option value="markdown">Markdown</option>
          <option value="html">HTML</option>
        </select>
      </div>
    </div>
  );
}

function EmbedConfig({
  config,
  onUpdate
}: {
  config: { url: string };
  onUpdate: (config: Record<string, unknown>) => void;
}) {
  return (
    <div className="space-y-3 pt-3 border-t border-slate-800">
      <div>
        <label className="text-xs text-slate-500">Embed URL</label>
        <input
          type="url"
          value={config.url}
          onChange={(e) => onUpdate({ ...config, url: e.target.value })}
          className="w-full px-2 py-1 bg-slate-800 border border-slate-700 rounded text-white text-sm"
          placeholder="https://..."
        />
      </div>
    </div>
  );
}
