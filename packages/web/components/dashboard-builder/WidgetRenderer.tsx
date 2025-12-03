'use client';

import type { Widget } from '@/lib/dashboard/types';

interface WidgetRendererProps {
  widget: Widget;
  isEditing: boolean;
}

export default function WidgetRenderer({ widget, isEditing }: WidgetRendererProps) {
  const baseClasses = 'h-full w-full rounded-lg bg-slate-900 border border-slate-800 overflow-hidden';

  const renderContent = () => {
    switch (widget.type) {
      case 'stat_card': {
        const config = (widget as { config: { value: string | number; label: string; change?: number; trend?: string; icon?: string } }).config;
        return (
          <div className="p-4 h-full flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-2xl">{config.icon || '📊'}</span>
              {config.change !== undefined && (
                <span className={`text-xs px-2 py-0.5 rounded ${
                  config.trend === 'up' ? 'bg-green-900/50 text-green-400' :
                  config.trend === 'down' ? 'bg-red-900/50 text-red-400' :
                  'bg-slate-800 text-slate-400'
                }`}>
                  {config.trend === 'up' ? '+' : ''}{config.change}%
                </span>
              )}
            </div>
            <div>
              <p className="text-2xl font-bold text-white">{config.value}</p>
              <p className="text-sm text-slate-400">{config.label}</p>
            </div>
          </div>
        );
      }

      case 'chart_line':
      case 'chart_bar':
      case 'chart_pie': {
        return (
          <div className="p-4 h-full flex flex-col">
            {widget.title && <h3 className="text-sm font-medium text-white mb-2">{widget.title}</h3>}
            <div className="flex-1 flex items-center justify-center border border-dashed border-slate-700 rounded">
              <span className="text-slate-500 text-sm">
                {widget.type === 'chart_line' ? '📈' : widget.type === 'chart_bar' ? '📊' : '🥧'}
                {' '}Chart - connect data source
              </span>
            </div>
          </div>
        );
      }

      case 'table': {
        return (
          <div className="p-4 h-full flex flex-col">
            {widget.title && <h3 className="text-sm font-medium text-white mb-2">{widget.title}</h3>}
            <div className="flex-1 overflow-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left py-2 px-2 text-slate-400 font-medium">Column 1</th>
                    <th className="text-left py-2 px-2 text-slate-400 font-medium">Column 2</th>
                    <th className="text-left py-2 px-2 text-slate-400 font-medium">Column 3</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-slate-800">
                    <td className="py-2 px-2 text-slate-300">--</td>
                    <td className="py-2 px-2 text-slate-300">--</td>
                    <td className="py-2 px-2 text-slate-300">--</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        );
      }

      case 'text': {
        const config = (widget as { config: { content: string } }).config;
        return (
          <div className="p-4 h-full">
            <p className="text-slate-300 text-sm">{config.content}</p>
          </div>
        );
      }

      case 'alert_feed': {
        return (
          <div className="p-4 h-full flex flex-col">
            <h3 className="text-sm font-medium text-white mb-2">Alert Feed</h3>
            <div className="flex-1 space-y-2 overflow-auto">
              <div className="p-2 bg-red-900/30 border border-red-800/50 rounded text-xs">
                <span className="text-red-400">CRITICAL</span>
                <p className="text-slate-300 mt-1">Sample alert message</p>
              </div>
              <div className="p-2 bg-yellow-900/30 border border-yellow-800/50 rounded text-xs">
                <span className="text-yellow-400">WARNING</span>
                <p className="text-slate-300 mt-1">Sample warning message</p>
              </div>
            </div>
          </div>
        );
      }

      case 'metric_gauge': {
        const config = (widget as { config: { value: number; min: number; max: number; unit?: string } }).config;
        const percentage = ((config.value - config.min) / (config.max - config.min)) * 100;
        return (
          <div className="p-4 h-full flex flex-col items-center justify-center">
            <div className="relative w-20 h-20">
              <svg className="w-full h-full -rotate-90" viewBox="0 0 36 36">
                <path
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="#334155"
                  strokeWidth="3"
                />
                <path
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="3"
                  strokeDasharray={`${percentage}, 100`}
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-lg font-bold text-white">{config.value}{config.unit}</span>
              </div>
            </div>
            {widget.title && <p className="text-xs text-slate-400 mt-2">{widget.title}</p>}
          </div>
        );
      }

      case 'signal_ticker': {
        return (
          <div className="h-full flex items-center px-4 overflow-hidden">
            <div className="flex gap-8 animate-pulse">
              <span className="text-green-400 text-sm">SPY +1.2%</span>
              <span className="text-red-400 text-sm">QQQ -0.4%</span>
              <span className="text-green-400 text-sm">GLD +0.8%</span>
              <span className="text-slate-400 text-sm">DXY 0.0%</span>
              <span className="text-green-400 text-sm">VIX -2.1%</span>
            </div>
          </div>
        );
      }

      case 'briefing_card': {
        return (
          <div className="p-4 h-full flex flex-col">
            <div className="flex items-center gap-2 mb-2">
              <span>🎖️</span>
              <h3 className="text-sm font-medium text-white">Intel Brief</h3>
            </div>
            <p className="text-xs text-slate-400 flex-1">
              AI-generated intelligence summary will appear here when connected to data source.
            </p>
            <div className="mt-2 pt-2 border-t border-slate-700">
              <span className="text-xs text-blue-400">Click to configure</span>
            </div>
          </div>
        );
      }

      case 'embed': {
        const config = (widget as { config: { url: string } }).config;
        if (!config.url) {
          return (
            <div className="h-full flex items-center justify-center">
              <span className="text-slate-500 text-sm">Configure embed URL</span>
            </div>
          );
        }
        return (
          <iframe
            src={config.url}
            className="w-full h-full border-0"
            allow="fullscreen"
          />
        );
      }

      default:
        return (
          <div className="p-4 h-full flex items-center justify-center">
            <span className="text-slate-500 text-sm">{widget.type}</span>
          </div>
        );
    }
  };

  return (
    <div className={`${baseClasses} ${isEditing ? 'hover:border-blue-500/50' : ''}`}>
      {renderContent()}
    </div>
  );
}
