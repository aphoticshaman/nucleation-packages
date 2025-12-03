'use client';

import { forwardRef, type InputHTMLAttributes, type SelectHTMLAttributes } from 'react';

interface GlassInputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  hint?: string;
}

interface GlassSelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  error?: string;
  hint?: string;
  children: React.ReactNode;
}

/**
 * GlassInput - Touch-optimized input with glass aesthetic
 *
 * Mobile considerations:
 * - 16px font to prevent iOS zoom
 * - 44px minimum height for touch
 * - Large touch target for focus
 */
const GlassInput = forwardRef<HTMLInputElement, GlassInputProps>(
  ({ label, error, hint, className = '', ...props }, ref) => {
    return (
      <div className="w-full">
        {label && (
          <label className="block text-sm text-slate-400 mb-2">{label}</label>
        )}
        <input
          ref={ref}
          className={`
            w-full
            px-4 py-3
            min-h-[44px]
            text-base sm:text-sm
            text-white
            placeholder:text-slate-500
            bg-black/30
            border border-white/[0.08]
            rounded-lg
            backdrop-blur-sm
            transition-all duration-150
            focus:outline-none focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20
            disabled:opacity-50 disabled:cursor-not-allowed
            ${error ? 'border-red-500/50 focus:border-red-500/50 focus:ring-red-500/20' : ''}
            ${className}
          `}
          {...props}
        />
        {error && <p className="mt-1.5 text-sm text-red-400">{error}</p>}
        {hint && !error && (
          <p className="mt-1.5 text-xs text-slate-500">{hint}</p>
        )}
      </div>
    );
  }
);

GlassInput.displayName = 'GlassInput';

/**
 * GlassSelect - Touch-optimized select with glass aesthetic
 */
const GlassSelect = forwardRef<HTMLSelectElement, GlassSelectProps>(
  ({ label, error, hint, children, className = '', ...props }, ref) => {
    return (
      <div className="w-full">
        {label && (
          <label className="block text-sm text-slate-400 mb-2">{label}</label>
        )}
        <div className="relative">
          <select
            ref={ref}
            className={`
              w-full
              px-4 py-3
              min-h-[44px]
              text-base sm:text-sm
              text-white
              bg-black/30
              border border-white/[0.08]
              rounded-lg
              backdrop-blur-sm
              appearance-none
              cursor-pointer
              transition-all duration-150
              focus:outline-none focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20
              disabled:opacity-50 disabled:cursor-not-allowed
              ${error ? 'border-red-500/50' : ''}
              ${className}
            `}
            {...props}
          >
            {children}
          </select>
          {/* Custom dropdown arrow */}
          <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none">
            <svg
              className="w-5 h-5 text-slate-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </div>
        </div>
        {error && <p className="mt-1.5 text-sm text-red-400">{error}</p>}
        {hint && !error && (
          <p className="mt-1.5 text-xs text-slate-500">{hint}</p>
        )}
      </div>
    );
  }
);

GlassSelect.displayName = 'GlassSelect';

/**
 * GlassToggle - Touch-optimized toggle switch
 */
interface GlassToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  label?: string;
  description?: string;
}

function GlassToggle({
  checked,
  onChange,
  disabled = false,
  label,
  description,
}: GlassToggleProps) {
  return (
    <div className="flex items-center justify-between gap-4">
      {(label || description) && (
        <div className="flex-1 min-w-0">
          {label && <p className="text-white text-sm">{label}</p>}
          {description && (
            <p className="text-xs text-slate-400 mt-0.5">{description}</p>
          )}
        </div>
      )}
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={`
          relative
          w-12 h-7
          min-w-[48px]
          rounded-full
          transition-colors duration-200
          focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:ring-offset-2 focus:ring-offset-[#0a0a0f]
          ${checked ? 'bg-blue-600' : 'bg-slate-700'}
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
        `}
      >
        <span
          className={`
            absolute top-1 w-5 h-5
            bg-white rounded-full
            shadow-sm
            transition-transform duration-200
            ${checked ? 'translate-x-6' : 'translate-x-1'}
          `}
        />
      </button>
    </div>
  );
}

export { GlassInput, GlassSelect, GlassToggle };
export type { GlassInputProps, GlassSelectProps, GlassToggleProps };
