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
          Build mission-focused intelligence packages for any audience
        </p>
      </div>

      {/* Quick templates */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 hover:border-blue-500 transition-colors cursor-pointer">
          <span className="text-2xl">🎖️</span>
          <p className="font-medium text-white mt-2">DoD Brief</p>
          <p className="text-xs text-slate-400 mt-1">Defense-focused analysis</p>
        </div>
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 hover:border-blue-500 transition-colors cursor-pointer">
          <span className="text-2xl">📋</span>
          <p className="font-medium text-white mt-2">Board Report</p>
          <p className="text-xs text-slate-400 mt-1">Executive summary</p>
        </div>
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 hover:border-blue-500 transition-colors cursor-pointer">
          <span className="text-2xl">🌐</span>
          <p className="font-medium text-white mt-2">NGO Brief</p>
          <p className="text-xs text-slate-400 mt-1">Humanitarian focus</p>
        </div>
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 hover:border-blue-500 transition-colors cursor-pointer">
          <span className="text-2xl">🔧</span>
          <p className="font-medium text-white mt-2">Technical</p>
          <p className="text-xs text-slate-400 mt-1">Full methodology</p>
        </div>
      </div>

      {/* Package Builder */}
      <PackageBuilder />
    </div>
  );
}
