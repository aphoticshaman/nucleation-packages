'use client';

import Link from 'next/link';
import { AlertTriangle } from 'lucide-react';

interface ExecBriefButtonProps {
  variant?: 'full' | 'compact';
  className?: string;
}

/**
 * Executive Brief Button
 *
 * A prominent button linking to the cached global intelligence summary.
 * Styled to be immediately visible - the "big red button" for decision-makers.
 *
 * Tone: Global, neutral (Reuters/AP/BBC style) - not partisan or sensationalist.
 * Content: Cached executive summary updated by cron, never user-triggered.
 */
export default function ExecBriefButton({ variant = 'full', className = '' }: ExecBriefButtonProps) {
  if (variant === 'compact') {
    return (
      <Link
        href="/app/briefings"
        className={`
          inline-flex items-center gap-2 px-4 py-2
          bg-gradient-to-r from-red-600 to-red-700
          hover:from-red-500 hover:to-red-600
          text-white font-semibold rounded-lg
          shadow-[0_0_20px_rgba(220,38,38,0.4)]
          hover:shadow-[0_0_30px_rgba(220,38,38,0.6)]
          transition-all duration-200
          border border-red-500/50
          ${className}
        `}
      >
        <AlertTriangle className="w-4 h-4" />
        <span>EXEC BRIEF</span>
      </Link>
    );
  }

  return (
    <Link
      href="/app/briefings"
      className={`
        flex items-center gap-3 px-6 py-3
        bg-gradient-to-r from-red-600 via-red-700 to-red-800
        hover:from-red-500 hover:via-red-600 hover:to-red-700
        text-white font-bold rounded-xl
        shadow-[0_0_30px_rgba(220,38,38,0.5)]
        hover:shadow-[0_0_40px_rgba(220,38,38,0.7)]
        transition-all duration-200
        border border-red-500/50
        animate-pulse hover:animate-none
        ${className}
      `}
    >
      <AlertTriangle className="w-5 h-5" />
      <div className="flex flex-col">
        <span className="text-xs uppercase tracking-wider opacity-80">Global Intelligence</span>
        <span className="text-lg">EXECUTIVE BRIEF</span>
      </div>
      <div className="ml-2 w-2 h-2 rounded-full bg-white animate-ping" />
    </Link>
  );
}
