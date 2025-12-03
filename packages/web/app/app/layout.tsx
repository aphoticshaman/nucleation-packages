import { requireConsumer } from '@/lib/auth';
import ConsumerNav from './components/ConsumerNav';
import OnboardingGate from '@/components/setup/OnboardingGate';

export default async function ConsumerLayout({ children }: { children: React.ReactNode }) {
  const user = await requireConsumer();

  return (
    <div className="min-h-screen bg-slate-950">
      <OnboardingGate
        userId={user.id}
        userTier={user.tier}
        hasCompletedOnboarding={!!user.onboarding_completed_at}
      >
        <ConsumerNav user={user} />
        <main className="pt-16">
          {/* Full-width layout - no max-width constraint on desktop */}
          <div className="px-4 lg:px-6 xl:px-8 py-6">{children}</div>
        </main>
      </OnboardingGate>
    </div>
  );
}
