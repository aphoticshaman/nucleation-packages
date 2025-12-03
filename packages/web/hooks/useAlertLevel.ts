/**
 * ALERT LEVEL COMPUTATION
 *
 * Computes the global alert level based on nation states.
 * Levels: normal, elevated, high, critical
 */

import { useMemo } from 'react';

export type AlertLevel = 'normal' | 'elevated' | 'high' | 'critical';

export interface NationState {
  code: string;
  name: string;
  basin_strength: number;
  transition_risk: number;
  regime: number;
}

export interface AlertInfo {
  level: AlertLevel;
  message: string;
  nations_at_risk: string[];
  cascade_risk: boolean;
}

/**
 * Compute alert level from nation states
 */
export function computeAlertLevel(nations: NationState[]): AlertInfo {
  if (!nations || nations.length === 0) {
    return {
      level: 'normal',
      message: 'Monitoring global stability',
      nations_at_risk: [],
      cascade_risk: false,
    };
  }

  // Count nations by risk level
  const criticalNations = nations.filter(
    (n) => n.transition_risk > 0.8 || n.basin_strength < 0.2
  );
  const highRiskNations = nations.filter(
    (n) => n.transition_risk > 0.6 && n.transition_risk <= 0.8
  );
  const elevatedNations = nations.filter(
    (n) => n.transition_risk > 0.4 && n.transition_risk <= 0.6
  );

  // Check for cascade risk (multiple connected high-risk nations)
  const cascadeRisk = criticalNations.length >= 2 || highRiskNations.length >= 5;

  // Determine overall level
  let level: AlertLevel = 'normal';
  let message = 'Monitoring global stability';

  if (criticalNations.length >= 3 || (criticalNations.length >= 1 && cascadeRisk)) {
    level = 'critical';
    message = `Critical: ${criticalNations.length} nation${criticalNations.length > 1 ? 's' : ''} at extreme risk`;
  } else if (criticalNations.length >= 1 || highRiskNations.length >= 3) {
    level = 'high';
    const count = criticalNations.length + highRiskNations.length;
    message = `High alert: ${count} nation${count > 1 ? 's' : ''} showing instability`;
  } else if (highRiskNations.length >= 1 || elevatedNations.length >= 5) {
    level = 'elevated';
    message = `Elevated: Monitoring ${elevatedNations.length + highRiskNations.length} regions`;
  }

  return {
    level,
    message,
    nations_at_risk: [
      ...criticalNations.map((n) => n.name),
      ...highRiskNations.map((n) => n.name),
    ].slice(0, 5),
    cascade_risk: cascadeRisk,
  };
}

/**
 * Hook for reactive alert level computation
 */
export function useAlertLevel(nations: NationState[]): AlertInfo {
  return useMemo(() => computeAlertLevel(nations), [nations]);
}

/**
 * Get CSS classes for alert level
 */
export function getAlertLevelStyles(level: AlertLevel): {
  bg: string;
  text: string;
  border: string;
  pulse: boolean;
} {
  switch (level) {
    case 'critical':
      return {
        bg: 'bg-red-900/90',
        text: 'text-red-100',
        border: 'border-red-500',
        pulse: true,
      };
    case 'high':
      return {
        bg: 'bg-orange-900/90',
        text: 'text-orange-100',
        border: 'border-orange-500',
        pulse: true,
      };
    case 'elevated':
      return {
        bg: 'bg-yellow-900/80',
        text: 'text-yellow-100',
        border: 'border-yellow-600',
        pulse: false,
      };
    default:
      return {
        bg: 'bg-slate-800/80',
        text: 'text-slate-300',
        border: 'border-slate-600',
        pulse: false,
      };
  }
}
