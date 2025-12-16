'use client';

import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  /** Show loading state */
  loading?: boolean;
}

/**
 * Button - Enterprise-grade button component
 *
 * Design principles:
 * - Solid colors (no gradients)
 * - No scale transforms
 * - No glow effects
 * - Subtle, immediate hover states
 * - Minimal border radius (6px)
 */
const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      children,
      variant = 'primary',
      size = 'md',
      loading = false,
      disabled,
      className = '',
      ...props
    },
    ref
  ) => {
    const sizeClasses = {
      sm: 'px-3 py-1.5 text-xs min-h-[32px]',
      md: 'px-4 py-2 text-sm min-h-[36px]',
      lg: 'px-5 py-2.5 text-sm min-h-[40px]',
    };

    const variantClasses = {
      primary: `
        bg-blue-600 text-white
        hover:bg-blue-700
        active:bg-blue-800
      `,
      secondary: `
        bg-slate-800 text-slate-200
        border border-slate-700
        hover:bg-slate-750 hover:border-slate-600
        active:bg-slate-700
      `,
      ghost: `
        bg-transparent text-slate-400
        hover:bg-slate-800 hover:text-slate-200
        active:bg-slate-750
      `,
      danger: `
        bg-red-600/10 text-red-400
        border border-red-600/20
        hover:bg-red-600/20 hover:border-red-600/30
        active:bg-red-600/30
      `,
    };

    const isDisabled = disabled || loading;

    return (
      <button
        ref={ref}
        disabled={isDisabled}
        className={`
          inline-flex items-center justify-center gap-2
          rounded-md
          font-medium
          transition-colors duration-100
          ${sizeClasses[size]}
          ${variantClasses[variant]}
          ${isDisabled ? 'opacity-50 cursor-not-allowed' : ''}
          ${className}
        `}
        {...props}
      >
        {loading ? (
          <>
            <span className="w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin" />
            <span>{children}</span>
          </>
        ) : (
          children
        )}
      </button>
    );
  }
);

Button.displayName = 'Button';

export { Button };
export type { ButtonProps };
