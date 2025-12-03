'use client';

import Link from 'next/link';
import Image from 'next/image';
import { ArrowLeft } from 'lucide-react';

export default function PrivacyPage() {
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
            href="/terms"
            className="text-slate-400 hover:text-white transition-colors min-h-[44px] flex items-center"
          >
            Terms
          </Link>
        </div>
      </nav>

      <div className="relative z-10 max-w-3xl mx-auto px-4 pt-24 pb-16">
        <div className="bg-[rgba(18,18,26,0.6)] backdrop-blur-xl rounded-2xl border border-white/[0.06] p-6 sm:p-8 lg:p-10">
          <h1 className="text-3xl sm:text-4xl font-bold text-white mb-2">Privacy Policy</h1>
          <p className="text-slate-400 text-sm mb-8">Last updated: December 2024</p>

          <div className="prose prose-invert prose-slate max-w-none space-y-8">
            <section>
              <h2 className="text-xl font-semibold text-white mb-3">1. Information We Collect</h2>
              <p className="text-slate-400 mb-3">We collect information you provide directly:</p>
              <ul className="list-disc list-inside text-slate-400 space-y-1.5 ml-2">
                <li>Account information (email, name)</li>
                <li>Profile data you choose to provide</li>
                <li>Simulation configurations and saved data</li>
                <li>Communications with our support team</li>
              </ul>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">2. Automatically Collected Information</h2>
              <p className="text-slate-400 mb-3">When you use our Service, we may collect:</p>
              <ul className="list-disc list-inside text-slate-400 space-y-1.5 ml-2">
                <li>Device and browser information</li>
                <li>IP address and approximate location</li>
                <li>Usage patterns and feature interactions</li>
                <li>Error logs and performance data</li>
              </ul>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">3. How We Use Your Information</h2>
              <p className="text-slate-400 mb-3">We use collected information to:</p>
              <ul className="list-disc list-inside text-slate-400 space-y-1.5 ml-2">
                <li>Provide and maintain the Service</li>
                <li>Process transactions and send related information</li>
                <li>Send technical notices and security alerts</li>
                <li>Respond to your comments and questions</li>
                <li>Analyze usage to improve the Service</li>
              </ul>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">4. Information Sharing</h2>
              <p className="text-slate-400">
                We do not sell your personal information. We may share information with third-party
                service providers who assist in operating our Service (e.g., hosting, analytics),
                subject to confidentiality obligations.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">5. Data Security</h2>
              <p className="text-slate-400">
                We implement industry-standard security measures to protect your data, including
                encryption in transit and at rest. However, no method of transmission over the
                Internet is 100% secure.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">6. Data Retention</h2>
              <p className="text-slate-400">
                We retain your information for as long as your account is active or as needed to
                provide services. You may request deletion of your account and associated data at any
                time.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">7. Your Rights</h2>
              <p className="text-slate-400 mb-3">You have the right to:</p>
              <ul className="list-disc list-inside text-slate-400 space-y-1.5 ml-2">
                <li>Access your personal data</li>
                <li>Correct inaccurate data</li>
                <li>Request deletion of your data</li>
                <li>Object to processing of your data</li>
                <li>Export your data in a portable format</li>
              </ul>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">8. Cookies</h2>
              <p className="text-slate-400">
                We use essential cookies for authentication and session management. We may also use
                analytics cookies to understand how you use the Service. You can control cookie
                preferences through your browser settings.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">9. Children&apos;s Privacy</h2>
              <p className="text-slate-400">
                The Service is not intended for users under 18 years of age. We do not knowingly
                collect personal information from children.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">10. Changes to This Policy</h2>
              <p className="text-slate-400">
                We may update this Privacy Policy from time to time. We will notify you of any changes
                by posting the new policy on this page and updating the &quot;Last updated&quot; date.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">11. Contact Us</h2>
              <p className="text-slate-400">
                For questions about this Privacy Policy, please contact us at{' '}
                <a href="mailto:privacy@latticeforge.ai" className="text-blue-400 hover:text-blue-300">
                  privacy@latticeforge.ai
                </a>
              </p>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}
