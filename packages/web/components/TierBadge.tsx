'use client';

import Image from 'next/image';

export type TierType = 'trial' | 'starter' | 'pro' | 'enterprise';

interface TierBadgeProps {
  tier: TierType;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  className?: string;
}

const TIER_CONFIG = {
  trial: {
    label: 'Trial',
    color: 'cyan',
    gradient: 'from-cyan-400 to-cyan-600',
    glow: 'rgba(34, 211, 238, 0.3)',
  },
  starter: {
    label: 'Starter',
    color: 'blue',
    gradient: 'from-blue-400 to-blue-600',
    glow: 'rgba(59, 130, 246, 0.3)',
  },
  pro: {
    label: 'Pro',
    color: 'orange',
    gradient: 'from-orange-400 to-amber-500',
    glow: 'rgba(251, 146, 60, 0.3)',
  },
  enterprise: {
    label: 'Enterprise',
    color: 'gold',
    gradient: 'from-amber-400 to-yellow-500',
    glow: 'rgba(251, 191, 36, 0.3)',
  },
};

const SIZES = {
  sm: { badge: 40, text: 'text-xs' },
  md: { badge: 56, text: 'text-sm' },
  lg: { badge: 80, text: 'text-base' },
};

export default function TierBadge({
  tier,
  size = 'md',
  showLabel = false,
  className = '',
}: TierBadgeProps) {
  const config = TIER_CONFIG[tier];
  const sizeConfig = SIZES[size];

  return (
    <div className={`inline-flex flex-col items-center gap-2 ${className}`}>
      {/* Badge with glow */}
      <div className="relative group" style={{ width: sizeConfig.badge, height: sizeConfig.badge }}>
        {/* Animated glow on hover */}
        <div
          className="absolute inset-0 rounded-full opacity-0 group-hover:opacity-60 blur-lg transition-opacity duration-300"
          style={{
            background: `radial-gradient(circle, ${config.glow} 0%, transparent 70%)`,
            transform: 'scale(1.5)',
          }}
        />

        {/* Static subtle glow */}
        <div
          className="absolute inset-0 rounded-full opacity-30 blur-md"
          style={{
            background: `radial-gradient(circle, ${config.glow} 0%, transparent 70%)`,
            transform: 'scale(1.2)',
          }}
        />

        {/* Badge image */}
        <Image
          src={`/images/badges/${tier}.png`}
          alt={`${config.label} tier badge`}
          width={sizeConfig.badge}
          height={sizeConfig.badge}
          className="relative z-10 object-contain drop-shadow-lg transition-transform duration-300 group-hover:scale-110"
        />
      </div>

      {/* Label */}
      {showLabel && (
        <span
          className={`font-semibold ${sizeConfig.text} bg-gradient-to-r ${config.gradient} bg-clip-text text-transparent`}
        >
          {config.label}
        </span>
      )}
    </div>
  );
}
