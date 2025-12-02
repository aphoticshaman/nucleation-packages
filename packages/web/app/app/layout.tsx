import { requireConsumer } from '@/lib/auth';
import ConsumerNav from './components/ConsumerNav';

export default async function ConsumerLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const user = await requireConsumer();

  return (
    <div className="min-h-screen bg-slate-950">
      <ConsumerNav user={user} />
      <main className="pt-16">
        <div className="max-w-7xl mx-auto px-4 py-8">{children}</div>
      </main>
    </div>
  );
}
