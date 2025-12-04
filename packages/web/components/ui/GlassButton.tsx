'use client';

import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react';

interface GlassButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  /** Full width on mobile, auto on desktop */
  fullWidthMobile?: boolean;
  /** Show loading state */
  loading?: boolean;
  /** Glow effect (primary only) */
  glow?: boolean;
}

/**
 * GlassButton - Touch-optimized button with glass aesthetic
 *
 * Mobile considerations:
 * - Minimum 44px touch target (WCAG 2.1)
 * - Active state feedback (no hover on touch)
 * - Reduced animations for performance
 */
const GlassButton = forwardRef<HTMLButtonElement, GlassButtonProps>(
  (
    {
      children,
      variant = 'primary',
      size = 'md',
      fullWidthMobile = false,
      loading = false,
      glow = false,
      disabled,
      className = '',
      ...props
    },
    ref
  ) => {
    // Size classes - ensuring minimum 44px touch target
    const sizeClasses = {
      sm: 'px-3 py-2 text-sm min-h-[36px] sm:min-h-[32px]',
      md: 'px-4 py-2.5 text-sm min-h-[44px] sm:min-h-[40px]',
      lg: 'px-6 py-3 text-base min-h-[48px] sm:min-h-[44px]',
    };

    // Variant styles
    const variantClasses = {
      primary: `
        bg-gradient-to-r from-blue-600 to-blue-500
        text-white font-medium
        border border-blue-500/50
        ${glow ? 'shadow-[0_0_20px_rgba(59,130,246,0.4)]' : 'shadow-md'}
        hover:from-blue-500 hover:to-blue-400
        active:from-blue-700 active:to-blue-600 active:scale-[0.98]
      `,
      secondary: `
        bg-[rgba(18,18,26,0.7)] backdrop-blur-md
        text-white
        border border-white/10
        hover:bg-[rgba(255,255,255,0.08)] hover:border-white/20
        active:bg-[rgba(255,255,255,0.12)] active:scale-[0.98]
      `,
      ghost: `
        bg-transparent
        text-slate-300
        border border-transparent
        hover:bg-white/5 hover:text-white
        active:bg-white/10 active:scale-[0.98]
      `,
      danger: `
        bg-red-500/20
        text-red-400
        border border-red-500/30
        hover:bg-red-500/30 hover:border-red-500/50
        active:bg-red-500/40 active:scale-[0.98]
      `,
    };

    const isDisabled = disabled || loading;

    return (
      <button
        ref={ref}
        disabled={isDisabled}
        className={`
          inline-flex items-center justify-center gap-2
          rounded-lg
          font-medium
          transition-all duration-150
          select-none
          touch-manipulation
          ${sizeClasses[size]}
          ${variantClasses[variant]}
          ${fullWidthMobile ? 'w-full sm:w-auto' : ''}
          ${isDisabled ? 'opacity-50 cursor-not-allowed pointer-events-none' : ''}
          ${className}
        `}
        {...props}
      >
        {loading ? (
          <>
            <svg
              className="animate-spin h-4 w-4"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
            <span>Loading...</span>
          </>
        ) : (
          children
        )}
      </button>
    );
  }
);

GlassButton.displayName = 'GlassButton';

export { GlassButton };
export type { GlassButtonProps };
