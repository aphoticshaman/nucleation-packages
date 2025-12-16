'use client';

import { useState } from 'react';
import {
  ChevronRight,
  ChevronLeft,
  X,
  Globe,
  Zap,
  Shield,
  Users,
  Check,
  Sparkles,
  Target,
  BarChart3,
} from 'lucide-react';
import { Card, Button } from '@/components/ui';

interface OnboardingWizardProps {
  isOpen: boolean;
  onClose: () => void;
  onComplete: (preferences: OnboardingPreferences) => void;
}

interface OnboardingPreferences {
  primaryUseCase: string;
  interests: string[];
  alertPreferences: {
    email: boolean;
    push: boolean;
    digest: boolean;
  };
  watchlist: string[];
}

const USE_CASES = [
  {
    id: 'analyst',
    title: 'Intelligence Analyst',
    description: 'Monitor global events and produce briefings',
    icon: <Target className="w-6 h-6" />,
    features: ['Detailed country profiles', 'Cascade simulations', 'Custom watchlists'],
  },
  {
    id: 'executive',
    title: 'Executive / Decision Maker',
    description: 'Get high-level summaries and risk assessments',
    icon: <BarChart3 className="w-6 h-6" />,
    features: ['Executive summaries', 'Risk dashboards', 'Daily digests'],
  },
  {
    id: 'researcher',
    title: 'Researcher / Academic',
    description: 'Study patterns and trends in global stability',
    icon: <Sparkles className="w-6 h-6" />,
    features: ['Historical data', 'Export capabilities', 'API access'],
  },
  {
    id: 'security',
    title: 'Security Professional',
    description: 'Monitor threats and security events',
    icon: <Shield className="w-6 h-6" />,
    features: ['Threat alerts', 'Incident tracking', 'Integration webhooks'],
  },
];

const INTEREST_AREAS = [
  { id: 'geopolitical', name: 'Geopolitical Risk', icon: <Globe className="w-4 h-4" /> },
  { id: 'economic', name: 'Economic Indicators', icon: <BarChart3 className="w-4 h-4" /> },
  { id: 'security', name: 'Security Events', icon: <Shield className="w-4 h-4" /> },
  { id: 'conflict', name: 'Conflict Zones', icon: <Target className="w-4 h-4" /> },
  { id: 'elections', name: 'Elections & Politics', icon: <Users className="w-4 h-4" /> },
  { id: 'markets', name: 'Market Signals', icon: <Zap className="w-4 h-4" /> },
];

const FEATURED_NATIONS = [
  { iso3: 'USA', name: 'United States', flag: '🇺🇸' },
  { iso3: 'CHN', name: 'China', flag: '🇨🇳' },
  { iso3: 'RUS', name: 'Russia', flag: '🇷🇺' },
  { iso3: 'UKR', name: 'Ukraine', flag: '🇺🇦' },
  { iso3: 'TWN', name: 'Taiwan', flag: '🇹🇼' },
  { iso3: 'IRN', name: 'Iran', flag: '🇮🇷' },
  { iso3: 'ISR', name: 'Israel', flag: '🇮🇱' },
  { iso3: 'DEU', name: 'Germany', flag: '🇩🇪' },
  { iso3: 'GBR', name: 'United Kingdom', flag: '🇬🇧' },
  { iso3: 'IND', name: 'India', flag: '🇮🇳' },
  { iso3: 'JPN', name: 'Japan', flag: '🇯🇵' },
  { iso3: 'BRA', name: 'Brazil', flag: '🇧🇷' },
];

export default function OnboardingWizard({ isOpen, onClose, onComplete }: OnboardingWizardProps) {
  const [step, setStep] = useState(0);
  const [preferences, setPreferences] = useState<OnboardingPreferences>({
    primaryUseCase: '',
    interests: [],
    alertPreferences: {
      email: true,
      push: false,
      digest: true,
    },
    watchlist: [],
  });

  const steps = [
    { title: 'Welcome', subtitle: 'Let\'s personalize your experience' },
    { title: 'Your Role', subtitle: 'How will you use LatticeForge?' },
    { title: 'Interests', subtitle: 'What topics interest you most?' },
    { title: 'Watchlist', subtitle: 'Select nations to follow' },
    { title: 'Notifications', subtitle: 'How should we reach you?' },
    { title: 'Ready!', subtitle: 'Your workspace is configured' },
  ];

  const canProceed = () => {
    switch (step) {
      case 0: return true;
      case 1: return preferences.primaryUseCase !== '';
      case 2: return preferences.interests.length > 0;
      case 3: return true; // Watchlist is optional
      case 4: return true;
      default: return true;
    }
  };

  const handleNext = () => {
    if (step < steps.length - 1) {
      setStep(step + 1);
    } else {
      onComplete(preferences);
    }
  };

  const handleBack = () => {
    if (step > 0) {
      setStep(step - 1);
    }
  };

  const toggleInterest = (id: string) => {
    setPreferences((prev) => ({
      ...prev,
      interests: prev.interests.includes(id)
        ? prev.interests.filter((i) => i !== id)
        : [...prev.interests, id],
    }));
  };

  const toggleWatchlist = (iso3: string) => {
    setPreferences((prev) => ({
      ...prev,
      watchlist: prev.watchlist.includes(iso3)
        ? prev.watchlist.filter((i) => i !== iso3)
        : [...prev.watchlist, iso3],
    }));
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/70" />

      {/* Modal */}
      <div className="relative w-full max-w-2xl bg-[rgba(18,18,26,0.95)] border border-white/[0.08] rounded-2xl shadow-2xl overflow-hidden">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-2 text-slate-400 hover:text-white rounded-lg hover:bg-white/5 transition-colors z-10"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Progress bar */}
        <div className="absolute top-0 left-0 right-0 h-1 bg-white/5">
          <div
            className="h-full bg-gradient-to-r from-blue-600 to-cyan-500 transition-all duration-300"
            style={{ width: `${((step + 1) / steps.length) * 100}%` }}
          />
        </div>

        {/* Content */}
        <div className="p-8 pt-12">
          {/* Step header */}
          <div className="text-center mb-8">
            <p className="text-sm text-blue-400 font-medium mb-2">
              Step {step + 1} of {steps.length}
            </p>
            <h2 className="text-2xl font-bold text-white">{steps[step].title}</h2>
            <p className="text-slate-400 mt-1">{steps[step].subtitle}</p>
          </div>

          {/* Step content */}
          <div className="min-h-[300px]">
            {step === 0 && <WelcomeStep />}
            {step === 1 && (
              <UseCaseStep
                selected={preferences.primaryUseCase}
                onSelect={(id) => setPreferences((prev) => ({ ...prev, primaryUseCase: id }))}
              />
            )}
            {step === 2 && (
              <InterestsStep
                selected={preferences.interests}
                onToggle={toggleInterest}
              />
            )}
            {step === 3 && (
              <WatchlistStep
                selected={preferences.watchlist}
                onToggle={toggleWatchlist}
              />
            )}
            {step === 4 && (
              <NotificationsStep
                preferences={preferences.alertPreferences}
                onChange={(prefs) =>
                  setPreferences((prev) => ({ ...prev, alertPreferences: prefs }))
                }
              />
            )}
            {step === 5 && <ReadyStep preferences={preferences} />}
          </div>
        </div>

        {/* Footer */}
        <div className="px-8 py-4 border-t border-white/[0.06] flex items-center justify-between">
          <div>
            {step > 0 && (
              <Button variant="secondary" onClick={handleBack}>
                <ChevronLeft className="w-4 h-4 mr-1" />
                Back
              </Button>
            )}
          </div>
          <Button
            variant="secondary"
            onClick={handleNext}
            disabled={!canProceed()}
          >
            {step === steps.length - 1 ? 'Get Started' : 'Continue'}
            <ChevronRight className="w-4 h-4 ml-1" />
          </Button>
        </div>
      </div>
    </div>
  );
}

function WelcomeStep() {
  return (
    <div className="text-center space-y-6">
      <div className="w-20 h-20 mx-auto rounded-2xl bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center">
        <Globe className="w-10 h-10 text-white" />
      </div>
      <div>
        <h3 className="text-lg font-semibold text-white mb-2">Welcome to LatticeForge</h3>
        <p className="text-slate-400 max-w-md mx-auto">
          Your deterministic intelligence platform for monitoring global stability,
          predicting cascades, and making informed decisions.
        </p>
      </div>
      <div className="flex justify-center gap-8 pt-4">
        {[
          { icon: <Globe className="w-6 h-6" />, label: '195 Nations' },
          { icon: <Zap className="w-6 h-6" />, label: 'Real-time' },
          { icon: <Shield className="w-6 h-6" />, label: 'AI Analysis' },
        ].map((item, idx) => (
          <div key={idx} className="text-center">
            <div className="w-12 h-12 mx-auto rounded-xl bg-white/5 flex items-center justify-center text-blue-400 mb-2">
              {item.icon}
            </div>
            <span className="text-sm text-slate-400">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function UseCaseStep({
  selected,
  onSelect,
}: {
  selected: string;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {USE_CASES.map((useCase) => (
        <button
          key={useCase.id}
          onClick={() => onSelect(useCase.id)}
          className={`p-4 rounded-xl border text-left transition-all ${
            selected === useCase.id
              ? 'bg-blue-500/20 border-blue-500/50'
              : 'bg-black/20 border-white/[0.06] hover:border-white/[0.12]'
          }`}
        >
          <div className="flex items-start gap-3">
            <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
              selected === useCase.id ? 'bg-blue-500/30 text-blue-400' : 'bg-white/5 text-slate-400'
            }`}>
              {useCase.icon}
            </div>
            <div className="flex-1">
              <h4 className="text-white font-medium">{useCase.title}</h4>
              <p className="text-xs text-slate-500 mt-1">{useCase.description}</p>
            </div>
            {selected === useCase.id && (
              <Check className="w-5 h-5 text-blue-400" />
            )}
          </div>
        </button>
      ))}
    </div>
  );
}

function InterestsStep({
  selected,
  onToggle,
}: {
  selected: string[];
  onToggle: (id: string) => void;
}) {
  return (
    <div className="space-y-4">
      <p className="text-sm text-slate-400 text-center mb-4">
        Select all that apply. You can change these later.
      </p>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {INTEREST_AREAS.map((interest) => {
          const isSelected = selected.includes(interest.id);
          return (
            <button
              key={interest.id}
              onClick={() => onToggle(interest.id)}
              className={`p-3 rounded-xl border flex items-center gap-3 transition-all ${
                isSelected
                  ? 'bg-blue-500/20 border-blue-500/50 text-white'
                  : 'bg-black/20 border-white/[0.06] text-slate-400 hover:text-white'
              }`}
            >
              <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                isSelected ? 'bg-blue-500/30' : 'bg-white/5'
              }`}>
                {interest.icon}
              </div>
              <span className="text-sm font-medium">{interest.name}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function WatchlistStep({
  selected,
  onToggle,
}: {
  selected: string[];
  onToggle: (iso3: string) => void;
}) {
  return (
    <div className="space-y-4">
      <p className="text-sm text-slate-400 text-center mb-4">
        Add nations to your watchlist for priority monitoring. You can add more later.
      </p>
      <div className="grid grid-cols-3 md:grid-cols-4 gap-3">
        {FEATURED_NATIONS.map((nation) => {
          const isSelected = selected.includes(nation.iso3);
          return (
            <button
              key={nation.iso3}
              onClick={() => onToggle(nation.iso3)}
              className={`p-3 rounded-xl border text-center transition-all ${
                isSelected
                  ? 'bg-blue-500/20 border-blue-500/50'
                  : 'bg-black/20 border-white/[0.06] hover:border-white/[0.12]'
              }`}
            >
              <span className="text-2xl mb-1 block">{nation.flag}</span>
              <span className="text-xs text-slate-400">{nation.name}</span>
            </button>
          );
        })}
      </div>
      {selected.length > 0 && (
        <p className="text-center text-sm text-blue-400">
          {selected.length} nation{selected.length !== 1 ? 's' : ''} selected
        </p>
      )}
    </div>
  );
}

function NotificationsStep({
  preferences,
  onChange,
}: {
  preferences: OnboardingPreferences['alertPreferences'];
  onChange: (prefs: OnboardingPreferences['alertPreferences']) => void;
}) {
  return (
    <div className="space-y-4 max-w-md mx-auto">
      <Card className="p-4">
        <ToggleSwitch
          checked={preferences.email}
          onChange={(checked) => onChange({ ...preferences, email: checked })}
          label="Email Notifications"
          description="Receive critical alerts via email"
        />
      </Card>

      <Card className="p-4">
        <ToggleSwitch
          checked={preferences.push}
          onChange={(checked) => onChange({ ...preferences, push: checked })}
          label="Push Notifications"
          description="Browser notifications for urgent events"
        />
      </Card>

      <Card className="p-4">
        <ToggleSwitch
          checked={preferences.digest}
          onChange={(checked) => onChange({ ...preferences, digest: checked })}
          label="Daily Digest"
          description="Morning summary of key events"
        />
      </Card>
    </div>
  );
}

function ToggleSwitch({
  checked,
  onChange,
  label,
  description,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  description: string;
}) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <div className="text-sm font-medium text-white">{label}</div>
        <div className="text-xs text-slate-400">{description}</div>
      </div>
      <button
        onClick={() => onChange(!checked)}
        className={`w-12 h-6 rounded-full transition-colors ${
          checked ? 'bg-blue-500' : 'bg-slate-700'
        }`}
      >
        <div className={`w-5 h-5 bg-white rounded-full mt-0.5 transition-transform ${
          checked ? 'translate-x-6' : 'translate-x-0.5'
        }`} />
      </button>
    </div>
  );
}

function ReadyStep({ preferences }: { preferences: OnboardingPreferences }) {
  const useCase = USE_CASES.find((u) => u.id === preferences.primaryUseCase);

  return (
    <div className="text-center space-y-6">
      <div className="w-20 h-20 mx-auto rounded-full bg-green-500/20 flex items-center justify-center">
        <Check className="w-10 h-10 text-green-400" />
      </div>
      <div>
        <h3 className="text-lg font-semibold text-white mb-2">You&apos;re all set!</h3>
        <p className="text-slate-400 max-w-md mx-auto">
          Your workspace has been configured based on your preferences.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 max-w-sm mx-auto text-sm">
        <div className="p-3 bg-black/20 rounded-lg text-left">
          <p className="text-slate-500 mb-1">Role</p>
          <p className="text-white font-medium">{useCase?.title || 'Not set'}</p>
        </div>
        <div className="p-3 bg-black/20 rounded-lg text-left">
          <p className="text-slate-500 mb-1">Interests</p>
          <p className="text-white font-medium">{preferences.interests.length} selected</p>
        </div>
        <div className="p-3 bg-black/20 rounded-lg text-left">
          <p className="text-slate-500 mb-1">Watchlist</p>
          <p className="text-white font-medium">{preferences.watchlist.length} nations</p>
        </div>
        <div className="p-3 bg-black/20 rounded-lg text-left">
          <p className="text-slate-500 mb-1">Notifications</p>
          <p className="text-white font-medium">
            {[
              preferences.alertPreferences.email && 'Email',
              preferences.alertPreferences.push && 'Push',
              preferences.alertPreferences.digest && 'Digest',
            ]
              .filter(Boolean)
              .join(', ') || 'None'}
          </p>
        </div>
      </div>
    </div>
  );
}
