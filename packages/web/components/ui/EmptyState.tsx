'use client';

import { type LucideIcon } from 'lucide-react';

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

/**
 * EmptyState - Enterprise-grade empty state display
 *
 * Design principles:
 * - No illustrations
 * - No cheerfulness
 * - Contextual explanation + next action
 * - Neutral, professional tone
 */
export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
}: EmptyStateProps) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-md p-8 text-center">
      {Icon && (
        <div className="inline-flex items-center justify-center w-10 h-10 rounded bg-slate-800 mb-4">
          <Icon className="w-5 h-5 text-slate-500" />
        </div>
      )}
      <h3 className="text-sm font-medium text-slate-300 mb-1">
        {title}
      </h3>
      <p className="text-sm text-slate-500 max-w-sm mx-auto">
        {description}
      </p>
      {action && (
        <button
          onClick={action.onClick}
          className="mt-4 px-4 py-2 text-sm font-medium text-blue-400 hover:text-blue-300 transition-colors"
        >
          {action.label}
        </button>
      )}
    </div>
  );
}
