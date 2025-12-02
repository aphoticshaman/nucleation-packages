'use client';

import Link from 'next/link';

export default function PrivacyPage() {
  return (
    <div className="min-h-screen bg-slate-950 py-16 px-4">
      <div className="max-w-3xl mx-auto">
        <div className="mb-8">
          <Link href="/" className="text-blue-400 hover:text-blue-300 text-sm">
            &larr; Back to Home
          </Link>
        </div>

        <h1 className="text-4xl font-bold text-white mb-8">Privacy Policy</h1>

        <div className="prose prose-invert prose-slate max-w-none">
          <p className="text-slate-300 text-lg mb-6">
            Last updated: December 2024
          </p>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-white mb-4">1. Information We Collect</h2>
            <p className="text-slate-400 mb-4">We collect information you provide directly:</p>
            <ul className="list-disc list-inside text-slate-400 space-y-2">
              <li>Account information (email, name)</li>
              <li>Profile data you choose to provide</li>
              <li>Simulation configurations and saved data</li>
              <li>Communications with our support team</li>
            </ul>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-white mb-4">2. Automatically Collected Information</h2>
            <p className="text-slate-400 mb-4">When you use our Service, we may collect:</p>
            <ul className="list-disc list-inside text-slate-400 space-y-2">
              <li>Device and browser information</li>
              <li>IP address and approximate location</li>
              <li>Usage patterns and feature interactions</li>
              <li>Error logs and performance data</li>
            </ul>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-white mb-4">3. How We Use Your Information</h2>
            <p className="text-slate-400 mb-4">We use collected information to:</p>
            <ul className="list-disc list-inside text-slate-400 space-y-2">
              <li>Provide and maintain the Service</li>
              <li>Process transactions and send related information</li>
              <li>Send technical notices and security alerts</li>
              <li>Respond to your comments and questions</li>
              <li>Analyze usage to improve the Service</li>
            </ul>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-white mb-4">4. Information Sharing</h2>
            <p className="text-slate-400">
              We do not sell your personal information. We may share information with third-party service
              providers who assist in operating our Service (e.g., hosting, analytics), subject to
              confidentiality obligations.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-white mb-4">5. Data Security</h2>
            <p className="text-slate-400">
              We implement industry-standard security measures to protect your data, including encryption
              in transit and at rest. However, no method of transmission over the Internet is 100% secure.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-white mb-4">6. Data Retention</h2>
            <p className="text-slate-400">
              We retain your information for as long as your account is active or as needed to provide
              services. You may request deletion of your account and associated data at any time.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-white mb-4">7. Your Rights</h2>
            <p className="text-slate-400 mb-4">You have the right to:</p>
            <ul className="list-disc list-inside text-slate-400 space-y-2">
              <li>Access your personal data</li>
              <li>Correct inaccurate data</li>
              <li>Request deletion of your data</li>
              <li>Object to processing of your data</li>
              <li>Export your data in a portable format</li>
            </ul>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-white mb-4">8. Cookies</h2>
            <p className="text-slate-400">
              We use essential cookies for authentication and session management. We may also use
              analytics cookies to understand how you use the Service. You can control cookie
              preferences through your browser settings.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-white mb-4">9. Children&apos;s Privacy</h2>
            <p className="text-slate-400">
              The Service is not intended for users under 18 years of age. We do not knowingly collect
              personal information from children.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-white mb-4">10. Changes to This Policy</h2>
            <p className="text-slate-400">
              We may update this Privacy Policy from time to time. We will notify you of any changes by
              posting the new policy on this page and updating the &quot;Last updated&quot; date.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-white mb-4">11. Contact Us</h2>
            <p className="text-slate-400">
              For questions about this Privacy Policy, please contact us at privacy@latticeforge.ai
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
