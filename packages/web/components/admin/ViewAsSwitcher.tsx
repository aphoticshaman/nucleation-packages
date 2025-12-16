'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Eye } from 'lucide-react';
import { useImpersonation, VIEW_PERSPECTIVES, type ViewPerspective } from '@/contexts/ImpersonationContext';

// Landing page mapping by role/tier
function getLandingPage(perspective: ViewPerspective): string {
  // Admin returns to admin dashboard
  if (perspective.role === 'admin') {
    return '/admin';
  }

  // Enterprise admin goes to enterprise dashboard
  if (perspective.role === 'enterprise') {
    return '/dashboard';
  }

  // Support staff goes to admin (with limited view)
  if (perspective.role === 'support') {
    return '/admin';
  }

  // All consumer tiers go to dashboard
  // The dashboard will show different features based on the tier
  return '/dashboard';
}

export default function ViewAsSwitcher() {
  const [isOpen, setIsOpen] = useState(false);
  const router = useRouter();
  const { isImpersonating, viewAs, startImpersonation, stopImpersonation } = useImpersonation();

  const handleSelect = (perspective: ViewPerspective) => {
    if (perspective.role === 'admin' && perspective.tier === 'enterprise_tier') {
      stopImpersonation();
      // Navigate back to admin dashboard
      router.push('/admin');
    } else {
      startImpersonation(perspective);
      // Navigate to the appropriate landing page for this perspective
      const landingPage = getLandingPage(perspective);
      router.push(landingPage);
    }
    setIsOpen(false);
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`w-full flex items-center justify-between p-3 rounded-lg transition-colors ${
          isImpersonating
            ? 'bg-amber-600 text-black'
            : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
        }`}
      >
        <div className="flex items-center gap-2">
          <Eye className="w-4 h-4" />
          <span className="text-sm font-medium">
            {isImpersonating ? `Viewing: ${viewAs?.label}` : 'View As...'}
          </span>
        </div>
        <svg
          className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />

          {/* Dropdown */}
          <div className="absolute bottom-full left-0 right-0 mb-2 bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-50 overflow-hidden">
            <div className="p-2 border-b border-slate-700">
              <p className="text-xs text-slate-400 px-2">Preview experience as:</p>
            </div>
            <div className="max-h-64 overflow-y-auto">
              {VIEW_PERSPECTIVES.map((perspective) => {
                const isActive =
                  (isImpersonating && viewAs?.label === perspective.label) ||
                  (!isImpersonating && perspective.role === 'admin');

                return (
                  <button
                    key={perspective.label}
                    onClick={() => handleSelect(perspective)}
                    className={`w-full text-left px-4 py-3 transition-colors ${
                      isActive
                        ? 'bg-blue-600 text-white'
                        : 'text-slate-300 hover:bg-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium">{perspective.label}</p>
                        <p className="text-xs text-slate-400">{perspective.description}</p>
                      </div>
                      {isActive && (
                        <span className="text-xs bg-blue-500 px-2 py-0.5 rounded">Active</span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
