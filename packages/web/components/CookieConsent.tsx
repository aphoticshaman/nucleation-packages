'use client';

import { useState, useEffect } from 'react';
import { X, Cookie, Settings, Shield } from 'lucide-react';
import Link from 'next/link';

const COOKIE_CONSENT_KEY = 'lattice_cookie_consent';
const COOKIE_PREFERENCES_KEY = 'lattice_cookie_preferences';

interface CookiePreferences {
  essential: boolean; // Always true, required
  analytics: boolean;
  marketing: boolean;
}

const DEFAULT_PREFERENCES: CookiePreferences = {
  essential: true,
  analytics: false,
  marketing: false,
};

export function CookieConsent() {
  const [showBanner, setShowBanner] = useState(false);
  const [showPreferences, setShowPreferences] = useState(false);
  const [preferences, setPreferences] = useState<CookiePreferences>(DEFAULT_PREFERENCES);

  useEffect(() => {
    // Check if user has already consented
    const consent = localStorage.getItem(COOKIE_CONSENT_KEY);
    if (!consent) {
      // Small delay to avoid flash on page load
      const timer = setTimeout(() => setShowBanner(true), 500);
      return () => clearTimeout(timer);
    } else {
      // Load saved preferences
      const savedPrefs = localStorage.getItem(COOKIE_PREFERENCES_KEY);
      if (savedPrefs) {
        try {
          setPreferences(JSON.parse(savedPrefs));
        } catch {
          // Ignore parse errors
        }
      }
    }
  }, []);

  const saveConsent = (prefs: CookiePreferences) => {
    localStorage.setItem(COOKIE_CONSENT_KEY, 'true');
    localStorage.setItem(COOKIE_PREFERENCES_KEY, JSON.stringify(prefs));
    setPreferences(prefs);
    setShowBanner(false);
    setShowPreferences(false);

    // Dispatch event for analytics/marketing scripts to listen
    window.dispatchEvent(new CustomEvent('cookieConsentChanged', { detail: prefs }));
  };

  const acceptAll = () => {
    saveConsent({ essential: true, analytics: true, marketing: true });
  };

  const rejectNonEssential = () => {
    saveConsent({ essential: true, analytics: false, marketing: false });
  };

  const savePreferences = () => {
    saveConsent(preferences);
  };

  if (!showBanner) return null;

  return (
    <>
      {/* Overlay for preferences modal */}
      {showPreferences && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[9998]"
          onClick={() => setShowPreferences(false)}
        />
      )}

      {/* Main Banner */}
      <div className="fixed bottom-0 left-0 right-0 z-[9999] p-4 sm:p-6">
        <div className="max-w-4xl mx-auto">
          <div className="bg-[#12121a] border border-white/10 rounded-2xl shadow-2xl overflow-hidden">
            {!showPreferences ? (
              // Simple consent view
              <div className="p-4 sm:p-6">
                <div className="flex items-start gap-4">
                  <div className="hidden sm:flex w-12 h-12 rounded-xl bg-gradient-to-br from-amber-500/20 to-orange-500/20 items-center justify-center flex-shrink-0">
                    <Cookie className="w-6 h-6 text-amber-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="text-lg font-semibold text-white mb-1">We value your privacy</h3>
                    <p className="text-slate-400 text-sm leading-relaxed">
                      We use cookies to enhance your experience, analyze site traffic, and for marketing purposes.
                      By clicking &quot;Accept All&quot;, you consent to our use of cookies.
                      Read our{' '}
                      <Link href="/privacy" className="text-blue-400 hover:text-blue-300 underline">
                        Privacy Policy
                      </Link>{' '}
                      for more information.
                    </p>
                  </div>
                </div>

                <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 mt-5">
                  <button
                    onClick={() => setShowPreferences(true)}
                    className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg border border-white/10 text-slate-300 hover:text-white hover:bg-white/5 transition-colors text-sm font-medium"
                  >
                    <Settings className="w-4 h-4" />
                    Customize
                  </button>
                  <button
                    onClick={rejectNonEssential}
                    className="px-4 py-2.5 rounded-lg border border-white/10 text-slate-300 hover:text-white hover:bg-white/5 transition-colors text-sm font-medium"
                  >
                    Reject Non-Essential
                  </button>
                  <button
                    onClick={acceptAll}
                    className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-blue-600 to-blue-500 text-white font-medium text-sm hover:from-blue-500 hover:to-blue-400 transition-all shadow-lg shadow-blue-500/25"
                  >
                    Accept All
                  </button>
                </div>

                {/* CCPA/GDPR notice */}
                <p className="text-xs text-slate-500 mt-4 flex items-center gap-1.5">
                  <Shield className="w-3.5 h-3.5" />
                  California residents: You can opt-out of the sale of personal information.
                  EU residents: You have additional rights under GDPR.
                </p>
              </div>
            ) : (
              // Detailed preferences view
              <div className="p-4 sm:p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white">Cookie Preferences</h3>
                  <button
                    onClick={() => setShowPreferences(false)}
                    className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>

                <div className="space-y-4 mb-6">
                  {/* Essential - always on */}
                  <div className="flex items-start justify-between p-4 rounded-xl bg-white/[0.03] border border-white/5">
                    <div className="flex-1 mr-4">
                      <div className="flex items-center gap-2">
                        <h4 className="font-medium text-white">Essential Cookies</h4>
                        <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400">Required</span>
                      </div>
                      <p className="text-sm text-slate-400 mt-1">
                        These cookies are necessary for the website to function. They enable core functionality like security, authentication, and session management.
                      </p>
                    </div>
                    <div className="flex-shrink-0">
                      <div className="w-11 h-6 bg-emerald-500/30 rounded-full relative cursor-not-allowed">
                        <div className="absolute right-0.5 top-0.5 w-5 h-5 bg-emerald-500 rounded-full shadow" />
                      </div>
                    </div>
                  </div>

                  {/* Analytics */}
                  <div className="flex items-start justify-between p-4 rounded-xl bg-white/[0.03] border border-white/5">
                    <div className="flex-1 mr-4">
                      <h4 className="font-medium text-white">Analytics Cookies</h4>
                      <p className="text-sm text-slate-400 mt-1">
                        Help us understand how visitors interact with our website. This data helps us improve our services and user experience.
                      </p>
                    </div>
                    <button
                      onClick={() => setPreferences(p => ({ ...p, analytics: !p.analytics }))}
                      className={`flex-shrink-0 w-11 h-6 rounded-full relative transition-colors ${
                        preferences.analytics ? 'bg-blue-500/30' : 'bg-white/10'
                      }`}
                    >
                      <div className={`absolute top-0.5 w-5 h-5 rounded-full shadow transition-all ${
                        preferences.analytics ? 'right-0.5 bg-blue-500' : 'left-0.5 bg-slate-400'
                      }`} />
                    </button>
                  </div>

                  {/* Marketing */}
                  <div className="flex items-start justify-between p-4 rounded-xl bg-white/[0.03] border border-white/5">
                    <div className="flex-1 mr-4">
                      <h4 className="font-medium text-white">Marketing Cookies</h4>
                      <p className="text-sm text-slate-400 mt-1">
                        Used to track visitors across websites to display relevant advertisements. Also helps measure the effectiveness of our marketing campaigns.
                      </p>
                    </div>
                    <button
                      onClick={() => setPreferences(p => ({ ...p, marketing: !p.marketing }))}
                      className={`flex-shrink-0 w-11 h-6 rounded-full relative transition-colors ${
                        preferences.marketing ? 'bg-blue-500/30' : 'bg-white/10'
                      }`}
                    >
                      <div className={`absolute top-0.5 w-5 h-5 rounded-full shadow transition-all ${
                        preferences.marketing ? 'right-0.5 bg-blue-500' : 'left-0.5 bg-slate-400'
                      }`} />
                    </button>
                  </div>
                </div>

                <div className="flex flex-col sm:flex-row gap-3">
                  <button
                    onClick={rejectNonEssential}
                    className="flex-1 px-4 py-2.5 rounded-lg border border-white/10 text-slate-300 hover:text-white hover:bg-white/5 transition-colors text-sm font-medium"
                  >
                    Reject All Non-Essential
                  </button>
                  <button
                    onClick={savePreferences}
                    className="flex-1 px-6 py-2.5 rounded-lg bg-gradient-to-r from-blue-600 to-blue-500 text-white font-medium text-sm hover:from-blue-500 hover:to-blue-400 transition-all"
                  >
                    Save Preferences
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

// Hook for other components to check consent status
export function useCookieConsent() {
  const [preferences, setPreferences] = useState<CookiePreferences | null>(null);

  useEffect(() => {
    const loadPreferences = () => {
      const saved = localStorage.getItem(COOKIE_PREFERENCES_KEY);
      if (saved) {
        try {
          setPreferences(JSON.parse(saved));
        } catch {
          setPreferences(null);
        }
      }
    };

    loadPreferences();

    // Listen for changes
    const handleChange = (e: CustomEvent<CookiePreferences>) => {
      setPreferences(e.detail);
    };

    window.addEventListener('cookieConsentChanged', handleChange as EventListener);
    return () => window.removeEventListener('cookieConsentChanged', handleChange as EventListener);
  }, []);

  return {
    hasConsented: preferences !== null,
    analyticsAllowed: preferences?.analytics ?? false,
    marketingAllowed: preferences?.marketing ?? false,
  };
}
