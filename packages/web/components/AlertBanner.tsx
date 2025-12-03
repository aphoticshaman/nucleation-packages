'use client';

interface AlertBannerProps {
  level: string;
  message?: string; // Optional override for dynamic messages
}

const ALERT_CONFIG: Record<string, { bg: string; text: string; message: string }> = {
  normal: {
    bg: 'bg-green-900/80',
    text: 'text-green-400',
    message: 'System stable. All nations within normal attractor basins.',
  },
  elevated: {
    bg: 'bg-yellow-900/80',
    text: 'text-yellow-400',
    message: 'Elevated activity detected. Monitoring increased variance in attractor positions.',
  },
  high: {
    bg: 'bg-orange-900/80',
    text: 'text-orange-400',
    message: 'High alert: Critical slowing detected. Multiple nations approaching basin boundaries.',
  },
  warning: {
    bg: 'bg-orange-900/80',
    text: 'text-orange-400',
    message: 'Warning: Critical slowing detected. Multiple nations approaching basin boundaries.',
  },
  critical: {
    bg: 'bg-red-900/80',
    text: 'text-red-400',
    message: 'CRITICAL: Phase transition imminent. Immediate intervention recommended.',
  },
};

export default function AlertBanner({ level, message }: AlertBannerProps) {
  const config = ALERT_CONFIG[level] || ALERT_CONFIG.normal;

  // Only show banner for non-normal states
  if (level === 'normal') return null;

  // Use custom message if provided, otherwise use config default
  const displayMessage = message || config.message;

  return (
    <div
      className={`absolute top-0 left-0 right-0 ${config.bg} backdrop-blur-sm z-20 px-4 py-2 ${
        level === 'critical' || level === 'high' ? 'animate-pulse' : ''
      }`}
    >
      <div className="max-w-4xl mx-auto flex items-center gap-3">
        <div className={`${config.text} font-bold text-sm uppercase tracking-wide`}>{level}</div>
        <div className="text-slate-300 text-sm">{displayMessage}</div>
      </div>
    </div>
  );
}
