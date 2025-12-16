'use client';

import { forwardRef, type HTMLAttributes, type ReactNode } from 'react';

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  /** Padding size */
  padding?: 'none' | 'sm' | 'md' | 'lg';
  /** Interactive card with hover state */
  interactive?: boolean;
  /** Selected/active state */
  selected?: boolean;
}

const PADDING = {
  none: '',
  sm: 'p-3',
  md: 'p-4',
  lg: 'p-5',
};

/**
 * Card - Enterprise-grade card component
 *
 * Design principles:
 * - Solid backgrounds (no blur)
 * - Subtle borders
 * - Minimal border radius (6px)
 * - No glow, no gradients
 */
const Card = forwardRef<HTMLDivElement, CardProps>(
  (
    {
      children,
      padding = 'md',
      interactive = false,
      selected = false,
      className = '',
      ...props
    },
    ref
  ) => {
    const baseStyles = `
      bg-slate-900
      border border-slate-800
      rounded-md
      ${PADDING[padding]}
      ${interactive ? 'cursor-pointer transition-colors duration-100 hover:bg-slate-850 hover:border-slate-700' : ''}
      ${selected ? 'border-slate-600 bg-slate-850' : ''}
    `;

    return (
      <div ref={ref} className={`${baseStyles} ${className}`} {...props}>
        {children}
      </div>
    );
  }
);

Card.displayName = 'Card';

export { Card };
export type { CardProps };
