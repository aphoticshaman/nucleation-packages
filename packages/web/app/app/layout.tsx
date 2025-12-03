import { requireConsumer } from '@/lib/auth';
import ConsumerNav from './components/ConsumerNav';
import OnboardingGate from '@/components/setup/OnboardingGate';
import '@/styles/dark-glass.css';

export default async function ConsumerLayout({ children }: { children: React.ReactNode }) {
  const user = await requireConsumer();

  return (
    <div className="min-h-screen bg-[#0a0a0f] relative">
      {/* Atmospheric background with grid pattern */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        {/* Base gradient - blue glow from top */}
        <div
          className="absolute inset-0"
          style={{
            background: 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(59, 130, 246, 0.15) 0%, transparent 50%)',
          }}
        />

        {/* Grid pattern overlay */}
        <div
          className="absolute inset-0 opacity-30"
          style={{
            backgroundImage: `
              linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
              linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px)
            `,
            backgroundSize: '60px 60px',
            maskImage: 'radial-gradient(ellipse at center, black 0%, transparent 70%)',
            WebkitMaskImage: 'radial-gradient(ellipse at center, black 0%, transparent 70%)',
          }}
        />

        {/* Subtle noise texture for depth */}
        <div
          className="absolute inset-0 opacity-20 mix-blend-overlay"
          style={{
            backgroundImage: 'url(/images/bg/obsidian.png)',
            backgroundSize: 'cover',
            backgroundPosition: 'center',
          }}
        />
      </div>

      <OnboardingGate
        userId={user.id}
        userTier={user.tier}
        hasCompletedOnboarding={!!user.onboarding_completed_at}
      >
        <ConsumerNav user={user} />
        <main className="pt-16 relative z-10">
          <div className="px-4 lg:px-6 xl:px-8 py-6">{children}</div>
        </main>
      </OnboardingGate>
    </div>
  );
}
