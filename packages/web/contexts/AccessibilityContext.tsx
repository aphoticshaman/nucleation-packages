'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export interface AccessibilitySettings {
  // Visual
  colorblindMode: 'none' | 'deuteranopia' | 'protanopia' | 'tritanopia' | 'monochrome';
  highContrast: boolean;
  reducedMotion: boolean;
  largeText: boolean;

  // Cognitive
  simplifiedUI: boolean;  // Hides advanced features
  autoGlossary: boolean;  // Shows definitions on hover for technical terms
}

const DEFAULT_SETTINGS: AccessibilitySettings = {
  colorblindMode: 'none',
  highContrast: false,
  reducedMotion: false,
  largeText: false,
  simplifiedUI: false,
  autoGlossary: true,
};

interface AccessibilityContextType {
  settings: AccessibilitySettings;
  updateSetting: <K extends keyof AccessibilitySettings>(
    key: K,
    value: AccessibilitySettings[K]
  ) => void;
  resetSettings: () => void;
  isColorblindMode: boolean;
}

const AccessibilityContext = createContext<AccessibilityContextType | null>(null);

const STORAGE_KEY = 'lf-accessibility-settings';

export function AccessibilityProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<AccessibilitySettings>(DEFAULT_SETTINGS);
  const [mounted, setMounted] = useState(false);

  // Load settings from localStorage on mount
  useEffect(() => {
    setMounted(true);
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        setSettings({ ...DEFAULT_SETTINGS, ...parsed });
      }
    } catch {
      // Use defaults if storage fails
    }

    // Check system preferences
    if (typeof window !== 'undefined') {
      const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      const prefersHighContrast = window.matchMedia('(prefers-contrast: more)').matches;

      if (prefersReducedMotion || prefersHighContrast) {
        setSettings(prev => ({
          ...prev,
          reducedMotion: prefersReducedMotion || prev.reducedMotion,
          highContrast: prefersHighContrast || prev.highContrast,
        }));
      }
    }
  }, []);

  // Save settings to localStorage
  useEffect(() => {
    if (mounted) {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
      } catch {
        // Storage might be full or disabled
      }
    }
  }, [settings, mounted]);

  // Apply global CSS classes based on settings
  useEffect(() => {
    if (!mounted || typeof document === 'undefined') return;

    const root = document.documentElement;

    // High contrast mode
    root.classList.toggle('high-contrast', settings.highContrast);

    // Reduced motion
    root.classList.toggle('reduced-motion', settings.reducedMotion);

    // Large text
    root.classList.toggle('large-text', settings.largeText);

    // Colorblind modes
    root.classList.remove('colorblind-deuteranopia', 'colorblind-protanopia', 'colorblind-tritanopia', 'colorblind-monochrome');
    if (settings.colorblindMode !== 'none') {
      root.classList.add(`colorblind-${settings.colorblindMode}`);
    }

    // Set CSS custom property for components that read it
    root.style.setProperty('--accessibility-colorblind-mode', settings.colorblindMode);
  }, [settings, mounted]);

  const updateSetting = <K extends keyof AccessibilitySettings>(
    key: K,
    value: AccessibilitySettings[K]
  ) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const resetSettings = () => {
    setSettings(DEFAULT_SETTINGS);
  };

  const isColorblindMode = settings.colorblindMode !== 'none';

  return (
    <AccessibilityContext.Provider
      value={{ settings, updateSetting, resetSettings, isColorblindMode }}
    >
      {children}
    </AccessibilityContext.Provider>
  );
}

export function useAccessibility() {
  const context = useContext(AccessibilityContext);
  if (!context) {
    throw new Error('useAccessibility must be used within an AccessibilityProvider');
  }
  return context;
}

// Hook for components that just need to know about colorblind mode
export function useColorblindMode(): boolean {
  const context = useContext(AccessibilityContext);
  return context?.isColorblindMode ?? false;
}

// Export color palettes for use by other components
export const COLOR_PALETTES = {
  standard: {
    critical: '#DC2626',
    high: '#EA580C',
    moderate: '#CA8A04',
    low: '#0891B2',
    safe: '#059669',
  },
  colorblind: {
    critical: '#CC3311',
    high: '#EE7733',
    moderate: '#CCBB44',
    low: '#33BBEE',
    safe: '#0077BB',
  },
  monochrome: {
    critical: '#1F2937',
    high: '#4B5563',
    moderate: '#6B7280',
    low: '#9CA3AF',
    safe: '#D1D5DB',
  },
};
