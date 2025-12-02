// LatticeForge "Dark Glass" Design System
// Neuro-Symbolic Color Coding:
// - Cyan/Neon Blue: Neural outputs (probabilistic, ML-derived)
// - Amber/Gold: Symbolic outputs (deterministic, rule-based)
// - Crimson/Red: High Risk/Threat (actionable alerts)

export const colors = {
  // Neural (ML/AI outputs)
  neural: {
    primary: '#06b6d4',      // cyan-500
    light: '#22d3ee',        // cyan-400
    dark: '#0891b2',         // cyan-600
    glow: 'rgba(6, 182, 212, 0.4)',
  },
  // Symbolic (Rule-based outputs)
  symbolic: {
    primary: '#f59e0b',      // amber-500
    light: '#fbbf24',        // amber-400
    dark: '#d97706',         // amber-600
    glow: 'rgba(245, 158, 11, 0.4)',
  },
  // Risk/Threat levels
  risk: {
    critical: '#ef4444',     // red-500
    high: '#f97316',         // orange-500
    elevated: '#eab308',     // yellow-500
    low: '#22c55e',          // green-500
    none: '#6b7280',         // gray-500
  },
  // Sentiment
  sentiment: {
    positive: '#10b981',     // emerald-500
    neutral: '#6b7280',      // gray-500
    negative: '#ef4444',     // red-500
  },
  // Base palette (Dark Glass)
  base: {
    bg: '#0f172a',           // slate-900
    bgDeep: '#020617',       // slate-950
    surface: '#1e293b',      // slate-800
    surfaceHover: '#334155', // slate-700
    border: '#334155',       // slate-700
    borderSubtle: '#1e293b', // slate-800
    text: '#f8fafc',         // slate-50
    textMuted: '#94a3b8',    // slate-400
    textSubtle: '#64748b',   // slate-500
  },
};

// Glassmorphism effects
export const glass = {
  // Standard glass panel
  panel: {
    background: 'rgba(15, 23, 42, 0.8)',
    backdropFilter: 'blur(12px)',
    border: '1px solid rgba(51, 65, 85, 0.5)',
  },
  // Elevated glass (modals, popovers)
  elevated: {
    background: 'rgba(30, 41, 59, 0.9)',
    backdropFilter: 'blur(16px)',
    border: '1px solid rgba(71, 85, 105, 0.5)',
    boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
  },
  // Subtle glass (cards)
  subtle: {
    background: 'rgba(15, 23, 42, 0.6)',
    backdropFilter: 'blur(8px)',
    border: '1px solid rgba(51, 65, 85, 0.3)',
  },
};

// Typography
export const typography = {
  fontMono: '"JetBrains Mono", "Fira Code", "SF Mono", monospace',
  fontSans: '"Inter", -apple-system, BlinkMacSystemFont, sans-serif',
  // Sizes
  xs: '0.75rem',    // 12px
  sm: '0.875rem',   // 14px
  base: '1rem',     // 16px
  lg: '1.125rem',   // 18px
  xl: '1.25rem',    // 20px
  '2xl': '1.5rem',  // 24px
  '3xl': '1.875rem', // 30px
};

// Animation durations
export const animation = {
  fast: '150ms',
  normal: '300ms',
  slow: '500ms',
  // Easing
  easeOut: 'cubic-bezier(0.16, 1, 0.3, 1)',
  easeInOut: 'cubic-bezier(0.65, 0, 0.35, 1)',
  spring: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
};

// Risk level utilities
export function getRiskColor(level: 'critical' | 'high' | 'elevated' | 'low' | 'none'): string {
  return colors.risk[level];
}

export function getRiskLabel(score: number): { level: string; color: string } {
  if (score >= 0.8) return { level: 'CRITICAL', color: colors.risk.critical };
  if (score >= 0.6) return { level: 'HIGH', color: colors.risk.high };
  if (score >= 0.4) return { level: 'ELEVATED', color: colors.risk.elevated };
  if (score >= 0.2) return { level: 'LOW', color: colors.risk.low };
  return { level: 'NONE', color: colors.risk.none };
}

export function getSentimentColor(score: number): string {
  if (score > 0.2) return colors.sentiment.positive;
  if (score < -0.2) return colors.sentiment.negative;
  return colors.sentiment.neutral;
}

// Source reliability (NATO Admiralty Code)
export const admiraltyCode = {
  reliability: {
    A: 'Completely reliable',
    B: 'Usually reliable',
    C: 'Fairly reliable',
    D: 'Not usually reliable',
    E: 'Unreliable',
    F: 'Reliability unknown',
  },
  credibility: {
    1: 'Confirmed',
    2: 'Probably true',
    3: 'Possibly true',
    4: 'Doubtfully true',
    5: 'Improbable',
    6: 'Truth cannot be judged',
  },
};

// Tailwind class helpers
export const tw = {
  // Glass panels
  glassPanel: 'bg-slate-900/80 backdrop-blur-xl border border-slate-700/50',
  glassElevated: 'bg-slate-800/90 backdrop-blur-2xl border border-slate-600/50 shadow-2xl',
  glassSubtle: 'bg-slate-900/60 backdrop-blur-lg border border-slate-700/30',

  // Text
  textPrimary: 'text-slate-50',
  textMuted: 'text-slate-400',
  textSubtle: 'text-slate-500',

  // Neural styling
  neural: 'text-cyan-400 border-cyan-500/50 bg-cyan-500/10',
  neuralGlow: 'shadow-[0_0_20px_rgba(6,182,212,0.3)]',

  // Symbolic styling
  symbolic: 'text-amber-400 border-amber-500/50 bg-amber-500/10',
  symbolicGlow: 'shadow-[0_0_20px_rgba(245,158,11,0.3)]',

  // Risk levels
  riskCritical: 'text-red-400 border-red-500/50 bg-red-500/10',
  riskHigh: 'text-orange-400 border-orange-500/50 bg-orange-500/10',
  riskElevated: 'text-yellow-400 border-yellow-500/50 bg-yellow-500/10',
  riskLow: 'text-green-400 border-green-500/50 bg-green-500/10',
};
