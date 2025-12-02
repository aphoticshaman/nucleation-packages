'use client';

import Image from 'next/image';

export type FeatureIconType = 'analytics' | 'export' | 'security' | 'simulation' | 'team' | 'webhook';

interface FeatureIconProps {
  icon: FeatureIconType;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
  glow?: boolean;
}

const SIZES = {
  sm: 32,
  md: 48,
  lg: 64,
  xl: 96,
};

export default function FeatureIcon({
  icon,
  size = 'md',
  className = '',
  glow = true
}: FeatureIconProps) {
  const dimension = SIZES[size];

  return (
    <div
      className={`relative inline-flex items-center justify-center ${className}`}
      style={{ width: dimension, height: dimension }}
    >
      {/* Glow effect */}
      {glow && (
        <div
          className="absolute inset-0 rounded-full opacity-40 blur-xl"
          style={{
            background: 'radial-gradient(circle, rgba(34, 211, 238, 0.4) 0%, transparent 70%)',
          }}
        />
      )}

      {/* Icon */}
      <Image
        src={`/images/icons/${icon}.png`}
        alt={`${icon} icon`}
        width={dimension}
        height={dimension}
        className="relative z-10 object-contain drop-shadow-lg"
        style={{
          filter: glow ? 'drop-shadow(0 0 8px rgba(34, 211, 238, 0.3))' : undefined,
        }}
      />
    </div>
  );
}
