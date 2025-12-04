'use client';

import dynamic from 'next/dynamic';

// Dynamic import to avoid SSR issues with the complex component
const PackageBuilder = dynamic(
  () => import('@/components/intelligence/PackageBuilder'),
  {
    ssr: false,
    loading: () => (
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-8 text-center">
        <div className="animate-pulse">
          <div className="h-8 bg-slate-800 rounded w-48 mx-auto mb-4" />
          <div className="h-4 bg-slate-800 rounded w-64 mx-auto" />
        </div>
      </div>
    )
  }
);

export default function PackagesPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Deliverable Packages</h1>
        <p className="text-slate-400 mt-1">
          Build mission-focused intelligence packages for any audience. Select a preset or customize your own.
        </p>
      </div>

      {/* Package Builder - single unified interface */}
      <PackageBuilder />
    </div>
  );
}
