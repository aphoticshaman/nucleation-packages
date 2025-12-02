import { requireEnterprise, createClient } from '@/lib/auth';
import EnterpriseNav from './components/EnterpriseNav';

export default async function DashboardLayout({ children }: { children: React.ReactNode }) {
  const user = await requireEnterprise();
  const supabase = await createClient();

  // Get organization info
  const { data: org } = await supabase
    .from('organizations')
    .select('*')
    .eq('id', user.organization_id)
    .single();

  return (
    <div className="min-h-screen bg-slate-950">
      <EnterpriseNav user={user} org={org} />
      {/* Responsive main content area */}
      {/* Mobile: full width with top padding for mobile nav bar */}
      {/* Desktop: left padding for sidebar */}
      <main className="pt-14 lg:pt-0 lg:pl-64 2xl:pl-72">
        <div className="p-4 md:p-6 lg:p-8 2xl:p-10 max-w-[2000px]">{children}</div>
      </main>
    </div>
  );
}
