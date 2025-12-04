'use client';

import { ReactNode } from 'react';
import { AccessibilityProvider } from '@/contexts/AccessibilityContext';

interface ClientProvidersProps {
  children: ReactNode;
}

export function ClientProviders({ children }: ClientProvidersProps) {
  return (
    <AccessibilityProvider>
      {children}
    </AccessibilityProvider>
  );
}
