'use client';

interface StatusIndicatorProps {
  status: 'healthy' | 'warning' | 'error' | 'inactive' | 'pending';
  label?: string;
  showLabel?: boolean;
  size?: 'sm' | 'md';
}

/**
 * StatusIndicator - Enterprise-grade status display
 *
 * Design principles:
 * - No glow effects
 * - No pulse animation (static = stable)
 * - Clear, semantic colors
 */
export function StatusIndicator({
  status,
  label,
  showLabel = true,
  size = 'md',
}: StatusIndicatorProps) {
  const dotSize = size === 'sm' ? 'w-2 h-2' : 'w-2.5 h-2.5';

  const statusConfig = {
    healthy: {
      dot: 'bg-emerald-500',
      text: 'text-emerald-500',
      label: label || 'Healthy',
    },
    warning: {
      dot: 'bg-amber-500',
      text: 'text-amber-500',
      label: label || 'Warning',
    },
    error: {
      dot: 'bg-red-500',
      text: 'text-red-500',
      label: label || 'Error',
    },
    inactive: {
      dot: 'bg-slate-500',
      text: 'text-slate-500',
      label: label || 'Inactive',
    },
    pending: {
      dot: 'bg-blue-500',
      text: 'text-blue-500',
      label: label || 'Pending',
    },
  };

  const config = statusConfig[status];

  return (
    <div className="inline-flex items-center gap-2">
      <span className={`${dotSize} rounded-full ${config.dot}`} />
      {showLabel && (
        <span className={`text-sm font-medium ${config.text}`}>
          {config.label}
        </span>
      )}
    </div>
  );
}
