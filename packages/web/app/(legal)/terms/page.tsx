'use client';

import Link from 'next/link';
import Image from 'next/image';
import { ArrowLeft } from 'lucide-react';

export default function TermsPage() {
  return (
    <div className="min-h-screen bg-[#0a0a0f] relative">
      {/* Atmospheric background */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div
          className="absolute inset-0"
          style={{
            background: 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(59, 130, 246, 0.1) 0%, transparent 50%)',
          }}
        />
        <div
          className="absolute inset-0 opacity-20"
          style={{
            backgroundImage: `
              linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
              linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px)
            `,
            backgroundSize: '60px 60px',
          }}
        />
      </div>

      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-[rgba(10,10,15,0.8)] backdrop-blur-xl border-b border-white/[0.06]">
        <div className="max-w-3xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link
            href="/"
            className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors min-h-[44px]"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Home</span>
          </Link>
          <Link href="/" className="flex items-center gap-2">
            <Image
              src="/images/brand/monogram.png"
              alt="LatticeForge"
              width={28}
              height={28}
            />
            <span className="font-bold text-white">LatticeForge</span>
          </Link>
          <Link
            href="/privacy"
            className="text-slate-400 hover:text-white transition-colors min-h-[44px] flex items-center"
          >
            Privacy
          </Link>
        </div>
      </nav>

      <div className="relative z-10 max-w-3xl mx-auto px-4 pt-24 pb-16">
        <div className="bg-[rgba(18,18,26,0.6)] backdrop-blur-xl rounded-2xl border border-white/[0.06] p-6 sm:p-8 lg:p-10">
          <h1 className="text-3xl sm:text-4xl font-bold text-white mb-2">Terms of Service</h1>
          <p className="text-slate-400 text-sm mb-8">Last updated: December 2024</p>

          <div className="prose prose-invert prose-slate max-w-none space-y-8">
            <section>
              <h2 className="text-xl font-semibold text-white mb-3">1. Acceptance of Terms</h2>
              <p className="text-slate-400">
                By accessing or using LatticeForge (&quot;the Service&quot;), you agree to be bound by
                these Terms of Service. If you do not agree to these terms, please do not use the
                Service.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">2. Description of Service</h2>
              <p className="text-slate-400">
                LatticeForge provides geopolitical analysis, simulation tools, and intelligence
                dashboards for research and educational purposes. The Service is provided &quot;as
                is&quot; without warranties of any kind.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">3. User Accounts</h2>
              <p className="text-slate-400">
                You are responsible for maintaining the confidentiality of your account credentials
                and for all activities that occur under your account. You must notify us immediately
                of any unauthorized use.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">4. Acceptable Use</h2>
              <p className="text-slate-400 mb-3">You agree not to:</p>
              <ul className="list-disc list-inside text-slate-400 space-y-1.5 ml-2">
                <li>Use the Service for any unlawful purpose</li>
                <li>Attempt to gain unauthorized access to any part of the Service</li>
                <li>Interfere with or disrupt the Service or servers</li>
                <li>Reverse engineer or attempt to extract source code</li>
                <li>Use the Service to harm others or spread misinformation</li>
              </ul>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">5. Intellectual Property</h2>
              <p className="text-slate-400">
                All content, features, and functionality of the Service are owned by LatticeForge and
                are protected by international copyright, trademark, and other intellectual property
                laws.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">6. Disclaimer</h2>
              <p className="text-slate-400">
                The analyses and simulations provided by LatticeForge are for informational and
                educational purposes only. They should not be construed as professional advice or used
                as the sole basis for any decision-making.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">7. Limitation of Liability</h2>
              <p className="text-slate-400">
                LatticeForge shall not be liable for any indirect, incidental, special, consequential,
                or punitive damages resulting from your use of the Service.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">8. Changes to Terms</h2>
              <p className="text-slate-400">
                We reserve the right to modify these terms at any time. Continued use of the Service
                after changes constitutes acceptance of the new terms.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">9. Contact</h2>
              <p className="text-slate-400">
                For questions about these Terms, please contact us at{' '}
                <a href="mailto:legal@latticeforge.ai" className="text-blue-400 hover:text-blue-300">
                  legal@latticeforge.ai
                </a>
              </p>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}
