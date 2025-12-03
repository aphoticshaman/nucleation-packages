import { requireAdmin } from '@/lib/auth';
import AdminNav from './components/AdminNav';
import AdminProviders from './components/AdminProviders';

export default async function AdminLayout({ children }: { children: React.ReactNode }) {
  const user = await requireAdmin();

  return (
    <AdminProviders userRole={user.role} userTier={user.tier}>
      <div className="min-h-screen bg-slate-950">
        <AdminNav user={user} />
        <main className="pl-64">
          <div className="p-8">{children}</div>
        </main>
      </div>
    </AdminProviders>
  );
}
