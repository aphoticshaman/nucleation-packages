import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL || 'https://latticeforge.io'),
  title: 'LatticeForge - Geopolitical Attractor Dynamics',
  description: 'Real-time visualization of nation-level attractor basins and phase transitions',
  openGraph: {
    title: 'LatticeForge - Geopolitical Intelligence Platform',
    description:
      'Real-time simulation and analysis of global political dynamics, regime stability, and phase transitions.',
    images: [
      {
        url: '/images/og/og-image.png',
        width: 1200,
        height: 630,
        alt: 'LatticeForge - Global Intelligence Visualization',
      },
    ],
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'LatticeForge - Geopolitical Intelligence Platform',
    description: 'Real-time simulation of global political dynamics and regime stability.',
    images: ['/images/og/og-image.png'],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-white antialiased">{children}</body>
    </html>
  );
}
