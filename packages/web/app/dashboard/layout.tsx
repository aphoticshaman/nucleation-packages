import { requireEnterprise, createClient } from '@/lib/auth';
import EnterpriseNav from './components/EnterpriseNav';

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
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
      <main className="pl-64">
        <div className="p-8">{children}</div>
      </main>
    </div>
  );
}
