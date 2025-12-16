'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Building2, FlaskConical, Shield, Briefcase, Mail, ArrowLeft, ExternalLink } from 'lucide-react';

type TabId = 'overview' | 'products' | 'research' | 'ip' | 'company';

const TABS: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: 'overview', label: 'Overview', icon: <Building2 className="w-4 h-4" /> },
  { id: 'products', label: 'Products', icon: <Briefcase className="w-4 h-4" /> },
  { id: 'research', label: 'Research', icon: <FlaskConical className="w-4 h-4" /> },
  { id: 'ip', label: 'Intellectual Property', icon: <Shield className="w-4 h-4" /> },
  { id: 'company', label: 'Company', icon: <Mail className="w-4 h-4" /> },
];

export default function CrystallineLabsPage() {
  const [activeTab, setActiveTab] = useState<TabId>('overview');

  return (
    <div className="min-h-screen bg-slate-950">
      {/* Header */}
      <header className="border-b border-slate-800/50">
        <div className="max-w-5xl mx-auto px-6 py-6">
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-slate-400 hover:text-white transition-colors text-sm mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to LatticeForge
          </Link>
          <h1 className="text-2xl font-semibold text-white tracking-tight">
            Crystalline Labs LLC
          </h1>
          <p className="text-slate-400 mt-1">
            Applied Research & Product Development
          </p>
        </div>
      </header>

      {/* Tabs */}
      <nav className="border-b border-slate-800/50 bg-slate-900/30">
        <div className="max-w-5xl mx-auto px-6">
          <div className="flex gap-1 overflow-x-auto">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                  activeTab === tab.id
                    ? 'border-cyan-500 text-white'
                    : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-600'
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Content */}
      <main className="max-w-5xl mx-auto px-6 py-12">
        {activeTab === 'overview' && <OverviewTab />}
        {activeTab === 'products' && <ProductsTab />}
        {activeTab === 'research' && <ResearchTab />}
        {activeTab === 'ip' && <IPTab />}
        {activeTab === 'company' && <CompanyTab />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/50 mt-auto">
        <div className="max-w-5xl mx-auto px-6 py-8">
          <p className="text-slate-500 text-sm">
            © {new Date().getFullYear()} Crystalline Labs LLC. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}

function OverviewTab() {
  return (
    <div className="prose prose-invert prose-slate max-w-none">
      <p className="text-lg text-slate-300 leading-relaxed">
        Crystalline Labs LLC is a privately held research and development company focused on
        deterministic intelligence systems, risk analytics, and decision-support architectures.
      </p>
      <p className="text-slate-400 leading-relaxed mt-6">
        The company develops proprietary analytic frameworks and software products designed
        for commercial and institutional use, with an emphasis on auditability, reproducibility,
        and governance.
      </p>
    </div>
  );
}

function ProductsTab() {
  return (
    <div className="space-y-8">
      <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-8">
        <h2 className="text-xl font-semibold text-white mb-4">LatticeForge</h2>
        <p className="text-slate-400 leading-relaxed mb-6">
          LatticeForge is a commercial geopolitical intelligence platform developed by Crystalline Labs.
          It provides deterministic, explainable risk assessments derived from structured data,
          governed analytic doctrine, and reproducible computation.
        </p>
        <p className="text-slate-400 leading-relaxed mb-6">
          LatticeForge is designed for commercial intelligence users in finance, insurance,
          energy, and enterprise risk.
        </p>
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-cyan-400 hover:text-cyan-300 transition-colors text-sm font-medium"
        >
          Visit latticeforge.io
          <ExternalLink className="w-4 h-4" />
        </Link>
      </div>
    </div>
  );
}

function ResearchTab() {
  return (
    <div className="prose prose-invert prose-slate max-w-none">
      <p className="text-slate-300 leading-relaxed">
        Crystalline Labs conducts applied research in deterministic analytics, evidence fusion,
        and large-scale decision systems.
      </p>

      <h3 className="text-lg font-medium text-white mt-8 mb-4">Research Focus Areas</h3>
      <ul className="space-y-3 text-slate-400">
        <li className="flex items-start gap-3">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-500 mt-2 flex-shrink-0" />
          Translating expert judgment into executable systems
        </li>
        <li className="flex items-start gap-3">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-500 mt-2 flex-shrink-0" />
          Improving auditability and replayability of analytic outputs
        </li>
        <li className="flex items-start gap-3">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-500 mt-2 flex-shrink-0" />
          Reducing operational and legal risk in intelligence workflows
        </li>
      </ul>

      <p className="text-slate-500 text-sm mt-8">
        The company&apos;s research program directly supports product development and proprietary system design.
      </p>
    </div>
  );
}

function IPTab() {
  return (
    <div className="prose prose-invert prose-slate max-w-none">
      <p className="text-slate-300 leading-relaxed">
        Crystalline Labs LLC develops and owns proprietary analytic systems, software architectures,
        and related intellectual property.
      </p>

      <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 mt-8">
        <h3 className="text-base font-medium text-white mb-3">IP Protection</h3>
        <p className="text-slate-400 text-sm leading-relaxed">
          Where appropriate, selected innovations are protected through patent filings.
          Other intellectual property is retained as trade secrets and proprietary know-how.
        </p>
      </div>

      <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 mt-4">
        <h3 className="text-base font-medium text-white mb-3">Licensing</h3>
        <p className="text-slate-400 text-sm leading-relaxed">
          The company licenses its intellectual property to its products and partners
          under commercial terms.
        </p>
      </div>
    </div>
  );
}

function CompanyTab() {
  return (
    <div className="prose prose-invert prose-slate max-w-none">
      <p className="text-slate-300 leading-relaxed">
        Crystalline Labs LLC is a privately held company.
      </p>

      <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 mt-8">
        <h3 className="text-base font-medium text-white mb-3">Contact</h3>
        <p className="text-slate-400 text-sm">
          For inquiries:
        </p>
        <a
          href="mailto:contact@crystallinelabs.com"
          className="text-cyan-400 hover:text-cyan-300 transition-colors text-sm"
        >
          contact@crystallinelabs.com
        </a>
      </div>
    </div>
  );
}
