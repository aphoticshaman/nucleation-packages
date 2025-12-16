import Link from 'next/link';
import { ArrowRight, Shield, Zap, Globe, BarChart3, Lock, Clock } from 'lucide-react';

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950">
      {/* Hero Section */}
      <header className="relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-900/20 via-transparent to-transparent" />
        <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-10" />

        <nav className="relative z-10 max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">LF</span>
            </div>
            <span className="text-white font-bold text-xl">LatticeForge</span>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/pricing" className="text-slate-400 hover:text-white text-sm transition-colors">
              Pricing
            </Link>
            <Link href="/login" className="text-slate-400 hover:text-white text-sm transition-colors">
              Sign In
            </Link>
            <Link
              href="/signup"
              className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              Get Started
            </Link>
          </div>
        </nav>

        <div className="relative z-10 max-w-7xl mx-auto px-6 py-24 md:py-32 text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-blue-500/10 border border-blue-500/20 rounded-full text-blue-400 text-sm mb-8">
            <Zap className="w-4 h-4" />
            <span>Zero-LLM Architecture • Deterministic Intelligence</span>
          </div>

          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold text-white mb-6 leading-tight">
            Geopolitical Intelligence
            <br />
            <span className="bg-gradient-to-r from-blue-400 to-cyan-400 text-transparent bg-clip-text">
              Without the Black Box
            </span>
          </h1>

          <p className="text-lg md:text-xl text-slate-400 max-w-3xl mx-auto mb-10">
            Real-time risk analysis across 190+ nations. Every insight traceable to source data.
            No hallucinations. No inference costs. Just transparent, deterministic intelligence.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/signup"
              className="group flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white px-6 py-3 rounded-lg font-medium transition-all"
            >
              Start Free Trial
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              href="/app"
              className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-white px-6 py-3 rounded-lg font-medium transition-colors"
            >
              <Globe className="w-4 h-4" />
              View Live Demo
            </Link>
          </div>

          {/* Trust indicators */}
          <div className="mt-16 flex flex-wrap items-center justify-center gap-8 text-slate-500 text-sm">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4" />
              <span>&lt;100ms response times</span>
            </div>
            <div className="flex items-center gap-2">
              <Lock className="w-4 h-4" />
              <span>SOC 2 compliant</span>
            </div>
            <div className="flex items-center gap-2">
              <Shield className="w-4 h-4" />
              <span>No data leaves your tenant</span>
            </div>
          </div>
        </div>
      </header>

      {/* Features Grid */}
      <section className="max-w-7xl mx-auto px-6 py-24">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Intelligence You Can Trust
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Every assessment backed by traceable evidence. Every prediction with explicit confidence intervals.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[
            {
              icon: Globe,
              title: '190+ Nations Monitored',
              description: 'Continuous coverage from GDELT, ACLED, World Bank, USGS, and 50+ other authoritative sources.',
            },
            {
              icon: Zap,
              title: 'Zero Inference Cost',
              description: 'All analysis via deterministic templates. No LLM calls means predictable costs and no hallucinations.',
            },
            {
              icon: BarChart3,
              title: 'Transparent Methodology',
              description: 'Dempster-Shafer evidence fusion with explicit uncertainty quantification. See exactly how conclusions are reached.',
            },
            {
              icon: Shield,
              title: 'Doctrine Registry',
              description: 'Enterprise customers can inspect and propose changes to the rules governing intelligence computation.',
            },
            {
              icon: Clock,
              title: 'Sub-100ms Latency',
              description: 'Edge-deployed caching ensures instant responses. No waiting for model inference.',
            },
            {
              icon: Lock,
              title: 'Audit-Ready',
              description: 'Complete decision audit trail. Every assessment traceable to source data and computation rules.',
            },
          ].map((feature) => (
            <div
              key={feature.title}
              className="p-6 bg-slate-900/50 border border-slate-800 rounded-xl hover:border-slate-700 transition-colors"
            >
              <feature.icon className="w-10 h-10 text-blue-400 mb-4" />
              <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
              <p className="text-slate-400 text-sm">{feature.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="max-w-7xl mx-auto px-6 py-24">
        <div className="bg-gradient-to-r from-blue-900/50 to-cyan-900/50 border border-blue-800/50 rounded-2xl p-8 md:p-12 text-center">
          <h2 className="text-2xl md:text-3xl font-bold text-white mb-4">
            Ready to See the World Clearly?
          </h2>
          <p className="text-slate-300 mb-8 max-w-2xl mx-auto">
            Join organizations that trust deterministic intelligence over black-box predictions.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/signup"
              className="bg-white text-slate-900 px-6 py-3 rounded-lg font-medium hover:bg-slate-100 transition-colors"
            >
              Start Free Trial
            </Link>
            <Link
              href="/pricing"
              className="text-white border border-white/20 px-6 py-3 rounded-lg font-medium hover:bg-white/10 transition-colors"
            >
              View Pricing
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-800 py-12">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-gradient-to-br from-blue-500 to-cyan-400 rounded flex items-center justify-center">
              <span className="text-white font-bold text-xs">LF</span>
            </div>
            <span className="text-slate-400 text-sm">© 2024 LatticeForge. All rights reserved.</span>
          </div>
          <div className="flex items-center gap-6 text-slate-400 text-sm">
            <Link href="/terms" className="hover:text-white transition-colors">Terms</Link>
            <Link href="/privacy" className="hover:text-white transition-colors">Privacy</Link>
            <a href="mailto:contact@latticeforge.ai" className="hover:text-white transition-colors">Contact</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
