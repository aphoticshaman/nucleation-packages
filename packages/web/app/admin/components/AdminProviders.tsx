'use client';

import type { ReactNode } from 'react';
import type { UserRole, UserTier } from '@/lib/auth';
import { ImpersonationProvider } from '@/contexts/ImpersonationContext';
import ImpersonationBanner from '@/components/admin/ImpersonationBanner';

interface AdminProvidersProps {
  children: ReactNode;
  userRole: UserRole;
  userTier: UserTier;
}

export default function AdminProviders({ children, userRole, userTier }: AdminProvidersProps) {
  return (
    <ImpersonationProvider userRole={userRole} userTier={userTier}>
      <ImpersonationBanner />
      {children}
    </ImpersonationProvider>
  );
}
