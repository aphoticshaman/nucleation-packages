import { requireAdmin } from '@/lib/auth';
import AdminNav from './components/AdminNav';
import AdminProviders from './components/AdminProviders';
import '@/styles/dark-glass.css';

export default async function AdminLayout({ children }: { children: React.ReactNode }) {
  const user = await requireAdmin();

  return (
    <AdminProviders userRole={user.role} userTier={user.tier}>
      <div className="min-h-screen bg-[#0a0a0f] relative">
        {/* Atmospheric background */}
        <div className="fixed inset-0 z-0 pointer-events-none">
          {/* Blue-purple gradient from top-left */}
          <div
            className="absolute inset-0"
            style={{
              background: 'radial-gradient(ellipse 80% 60% at 10% 10%, rgba(59, 130, 246, 0.12) 0%, transparent 50%)',
            }}
          />
          {/* Grid pattern */}
          <div
            className="absolute inset-0 opacity-20"
            style={{
              backgroundImage: `
                linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px)
              `,
              backgroundSize: '40px 40px',
            }}
          />
        </div>

        <AdminNav user={user} />
        <main className="pl-72 relative z-10">
          <div className="p-6 lg:p-8">{children}</div>
        </main>
      </div>
    </AdminProviders>
  );
}
