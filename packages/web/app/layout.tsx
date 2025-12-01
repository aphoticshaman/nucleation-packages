import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'LatticeForge - Geopolitical Attractor Dynamics',
  description: 'Real-time visualization of nation-level attractor basins and phase transitions',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-white antialiased">
        {children}
      </body>
    </html>
  );
}
