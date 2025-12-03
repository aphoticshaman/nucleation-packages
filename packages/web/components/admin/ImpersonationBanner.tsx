'use client';

import { useImpersonationSafe } from '@/contexts/ImpersonationContext';

export default function ImpersonationBanner() {
  const impersonation = useImpersonationSafe();

  if (!impersonation?.isImpersonating || !impersonation.viewAs) {
    return null;
  }

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-amber-600 text-black py-2 px-4">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="font-bold">PREVIEW MODE</span>
          <span className="text-amber-900">|</span>
          <span>
            Viewing as: <strong>{impersonation.viewAs.label}</strong>
            <span className="text-amber-800 ml-2">({impersonation.viewAs.description})</span>
          </span>
        </div>
        <button
          onClick={impersonation.stopImpersonation}
          className="px-4 py-1 bg-black text-white rounded-lg text-sm font-medium hover:bg-gray-800 transition-colors"
        >
          Exit Preview
        </button>
      </div>
    </div>
  );
}
