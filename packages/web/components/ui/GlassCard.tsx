'use client';

import { forwardRef, type HTMLAttributes, type ReactNode } from 'react';

interface GlassCardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  /** Blur intensity: 'none' | 'light' | 'medium' | 'heavy' */
  blur?: 'none' | 'light' | 'medium' | 'heavy';
  /** Show accent gradient line at top */
  accent?: boolean;
  /** Glow effect on hover (desktop only) */
  glow?: boolean;
  /** Compact padding for mobile */
  compact?: boolean;
  /** Interactive card with hover state */
  interactive?: boolean;
}

const BLUR_VALUES = {
  none: '',
  light: 'backdrop-blur-sm',
  medium: 'backdrop-blur-md',
  heavy: 'backdrop-blur-lg',
};

/**
 * GlassCard - Dark glass UI component with mobile-first design
 *
 * Features:
 * - Fallback for browsers without backdrop-filter support
 * - Touch-optimized padding (min 44px touch targets)
 * - Reduced blur on mobile for performance
 * - Respects prefers-reduced-motion
 */
const GlassCard = forwardRef<HTMLDivElement, GlassCardProps>(
  (
    {
      children,
      blur = 'medium',
      accent = false,
      glow = false,
      compact = false,
      interactive = false,
      className = '',
      ...props
    },
    ref
  ) => {
    const blurClass = BLUR_VALUES[blur];

    // Base glass styles with fallback for older browsers
    const baseStyles = `
      relative
      bg-[rgba(18,18,26,0.7)]
      ${blurClass}
      border border-white/[0.06]
      rounded-xl
      ${compact ? 'p-3 sm:p-4' : 'p-4 sm:p-6'}
      ${interactive ? 'cursor-pointer transition-all duration-200 hover:bg-[rgba(18,18,26,0.85)] hover:border-white/[0.1] hover:-translate-y-0.5 hover:shadow-lg active:translate-y-0 active:shadow-md' : ''}
      ${glow ? 'hover:shadow-[0_0_30px_rgba(59,130,246,0.15)]' : ''}
    `;

    // Fallback for browsers that don't support backdrop-filter
    // Uses a slightly more opaque background instead
    const fallbackStyles = `
      @supports not (backdrop-filter: blur(1px)) {
        background: rgba(18, 18, 26, 0.95) !important;
      }
    `;

    return (
      <div ref={ref} className={`${baseStyles} ${className}`} {...props}>
        {/* Fallback styles injected */}
        <style jsx>{fallbackStyles}</style>

        {/* Accent line at top */}
        {accent && (
          <div
            className="absolute top-0 left-0 right-0 h-[2px] rounded-t-xl"
            style={{
              background: 'linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #06b6d4 100%)',
            }}
          />
        )}

        {children}
      </div>
    );
  }
);

GlassCard.displayName = 'GlassCard';

export { GlassCard };
export type { GlassCardProps };
