import type { Metadata } from 'next';
import './globals.css';
import { CookieConsent } from '@/components/CookieConsent';
import { NonProdApiToggle } from '@/components/admin/NonProdApiToggle';

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL || 'https://latticeforge.io'),
  title: 'LatticeForge - Know What Happens Next',
  description: 'AI watches global news so you can stay ahead. Daily briefings on what matters.',
  openGraph: {
    title: 'LatticeForge - Know What Happens Next',
    description:
      'AI-powered news intelligence. See what\'s happening around the world and what it means for you.',
    images: [
      {
        url: '/images/og/og-image.png',
        width: 1200,
        height: 630,
        alt: 'LatticeForge - See the world clearly',
      },
    ],
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'LatticeForge - Know What Happens Next',
    description: 'AI watches global news so you can stay ahead. Daily briefings on what matters.',
    images: ['/images/og/og-image.png'],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-white antialiased">
        {children}
        <CookieConsent />
        <NonProdApiToggle />
      </body>
    </html>
  );
}
