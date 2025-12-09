'use client';

import { useState, useEffect, useCallback } from 'react';

interface ApiToggleState {
  lfbm: boolean;
  gdelt: boolean;
  worldbank: boolean;
  usgs: boolean;
  sentiment: boolean;
  training: boolean;
}

interface OverrideStatus {
  armed: boolean;
  armedSecondsRemaining: number;
  currentState: {
    lfbmEnabled: boolean;
    override: { active: boolean; remainingMinutes: number };
    environment: string;
    lfProdEnable: string;
  };
}

const DEFAULT_STATE: ApiToggleState = {
  lfbm: false,
  gdelt: false,
  worldbank: false,
  usgs: false,
  sentiment: false,
  training: false,
};

const STORAGE_KEY = 'latticeforge_nonprod_api_toggles';
const MIN_DURATION = 10;
const MAX_DURATION = 90;
const DEFAULT_DURATION = 30;

export function NonProdApiToggle() {
  const [isOpen, setIsOpen] = useState(false);
  const [toggles, setToggles] = useState<ApiToggleState>(DEFAULT_STATE);
  const [isNonProd, setIsNonProd] = useState(false);

  // Override state
  const [status, setStatus] = useState<OverrideStatus | null>(null);
  const [duration, setDuration] = useState(DEFAULT_DURATION);
  const [isLoading, setIsLoading] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);

  // Confirmation dialogs
  const [showEnableConfirm, setShowEnableConfirm] = useState(false);
  const [showDisableConfirm, setShowDisableConfirm] = useState(false);

  // Fetch status from API
  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/admin/api-override');
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
        if (data.currentState?.override?.active) {
          setCountdown(data.currentState.override.remainingMinutes);
        } else {
          setCountdown(null);
        }
      }
    } catch {
      // Ignore errors - might not be logged in
    }
  }, []);

  useEffect(() => {
    // Check if we're in non-production
    const env = process.env.NEXT_PUBLIC_VERCEL_ENV || process.env.NODE_ENV;
    setIsNonProd(env !== 'production');

    // Load saved state
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        setToggles(JSON.parse(saved));
      } catch {
        // Use defaults
      }
    }

    // Fetch initial status
    fetchStatus();

    // Poll status every 30 seconds
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  // Countdown timer for active override
  useEffect(() => {
    if (!countdown || countdown <= 0) return;

    const timer = setInterval(() => {
      setCountdown(prev => {
        if (!prev || prev <= 1) {
          fetchStatus();
          return null;
        }
        return prev - 1;
      });
    }, 60000); // Update every minute

    return () => clearInterval(timer);
  }, [countdown, fetchStatus]);

  // Don't render in production
  if (!isNonProd) return null;

  const handleToggle = (key: keyof ApiToggleState) => {
    const newToggles = { ...toggles, [key]: !toggles[key] };
    setToggles(newToggles);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(newToggles));
    document.cookie = `${STORAGE_KEY}=${JSON.stringify(newToggles)};path=/;max-age=86400`;
  };

  const enableAll = () => {
    const allEnabled: ApiToggleState = {
      lfbm: true,
      gdelt: true,
      worldbank: true,
      usgs: true,
      sentiment: true,
      training: true,
    };
    setToggles(allEnabled);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(allEnabled));
    document.cookie = `${STORAGE_KEY}=${JSON.stringify(allEnabled)};path=/;max-age=86400`;
  };

  const disableAll = () => {
    setToggles(DEFAULT_STATE);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(DEFAULT_STATE));
    document.cookie = `${STORAGE_KEY}=${JSON.stringify(DEFAULT_STATE)};path=/;max-age=86400`;
  };

  // ENABLE - arm + fire in one step with confirmation
  const handleEnable = async () => {
    setIsLoading(true);
    try {
      // First ARM
      const armRes = await fetch('/api/admin/api-override', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'arm' }),
      });

      if (!armRes.ok) {
        console.error('ARM failed');
        return;
      }

      // Then FIRE immediately
      const fireRes = await fetch('/api/admin/api-override', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'fire', minutes: duration }),
      });

      if (fireRes.ok) {
        setCountdown(duration);
        fetchStatus();
      }
    } finally {
      setIsLoading(false);
      setShowEnableConfirm(false);
    }
  };

  // DISABLE - disarm
  const handleDisable = async () => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/admin/api-override', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'disarm' }),
      });
      if (res.ok) {
        setCountdown(null);
        fetchStatus();
      }
    } finally {
      setIsLoading(false);
      setShowDisableConfirm(false);
    }
  };

  // Duration controls
  const incrementDuration = () => {
    setDuration(prev => Math.min(MAX_DURATION, prev + 5));
  };

  const decrementDuration = () => {
    setDuration(prev => Math.max(MIN_DURATION, prev - 5));
  };

  const handleDurationChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value, 10);
    if (!isNaN(val)) {
      setDuration(Math.max(MIN_DURATION, Math.min(MAX_DURATION, val)));
    }
  };

  const activeCount = Object.values(toggles).filter(Boolean).length;
  const isOverrideActive = status?.currentState?.override?.active ?? false;
  const lfbmEnabled = status?.currentState?.lfbmEnabled ?? false;

  return (
    <>
      {/* Floating Toggle Button - ALWAYS visible to admin in non-prod */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`fixed bottom-4 right-4 z-[9999] w-14 h-14 rounded-full shadow-lg flex items-center justify-center text-white font-bold text-sm transition-all ${
          isOverrideActive
            ? 'bg-red-500 hover:bg-red-600 animate-pulse'
            : activeCount > 0
            ? 'bg-amber-500 hover:bg-amber-600'
            : 'bg-slate-700 hover:bg-slate-600'
        }`}
        title={`API Controls ${isOverrideActive ? `(LIVE - ${countdown}m)` : '(Disabled)'}`}
        style={{ zIndex: 9999 }}
      >
        <span className="text-xl">
          {isOverrideActive ? '!' : 'API'}
        </span>
        {isOverrideActive && countdown && (
          <span className="absolute -top-1 -right-1 bg-white text-red-500 text-[10px] font-bold px-1 rounded">
            {countdown}m
          </span>
        )}
      </button>

      {/* Confirmation Dialogs */}
      {showEnableConfirm && (
        <div className="fixed inset-0 z-[10000] bg-black/70 flex items-center justify-center" style={{ zIndex: 10000 }}>
          <div className="bg-slate-800 border border-amber-500 rounded-lg p-6 max-w-sm mx-4">
            <h3 className="text-amber-400 font-bold text-lg mb-2">Enable RunPod APIs?</h3>
            <p className="text-slate-300 text-sm mb-4">
              This will enable RunPod/LFBM API calls for <strong>{duration} minutes</strong>.
              <br /><br />
              After this time, APIs will automatically be disabled again.
            </p>
            <div className="flex gap-2">
              <button
                onClick={handleEnable}
                disabled={isLoading}
                className="flex-1 bg-amber-600 hover:bg-amber-700 text-white py-2 rounded font-bold disabled:opacity-50"
              >
                {isLoading ? 'Enabling...' : `Enable for ${duration}m`}
              </button>
              <button
                onClick={() => setShowEnableConfirm(false)}
                className="flex-1 bg-slate-600 hover:bg-slate-700 text-white py-2 rounded"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {showDisableConfirm && (
        <div className="fixed inset-0 z-[10000] bg-black/70 flex items-center justify-center" style={{ zIndex: 10000 }}>
          <div className="bg-slate-800 border border-red-500 rounded-lg p-6 max-w-sm mx-4">
            <h3 className="text-red-400 font-bold text-lg mb-2">Disable RunPod APIs?</h3>
            <p className="text-slate-300 text-sm mb-4">
              This will immediately disable all RunPod/LFBM API calls.
              <br /><br />
              {countdown && countdown > 0 && (
                <span className="text-amber-400">
                  ({countdown} minutes were remaining on the current override)
                </span>
              )}
            </p>
            <div className="flex gap-2">
              <button
                onClick={handleDisable}
                disabled={isLoading}
                className="flex-1 bg-red-600 hover:bg-red-700 text-white py-2 rounded font-bold disabled:opacity-50"
              >
                {isLoading ? 'Disabling...' : 'Disable Now'}
              </button>
              <button
                onClick={() => setShowDisableConfirm(false)}
                className="flex-1 bg-slate-600 hover:bg-slate-700 text-white py-2 rounded"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Popover Panel - ALWAYS on top */}
      {isOpen && (
        <div
          className="fixed bottom-20 right-4 z-[9999] w-80 bg-slate-800 border border-slate-600 rounded-lg shadow-xl p-4 max-h-[80vh] overflow-y-auto"
          style={{ zIndex: 9999 }}
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-white font-bold text-sm">API Controls (Non-Prod)</h3>
            <button
              onClick={() => setIsOpen(false)}
              className="text-slate-400 hover:text-white"
            >
              X
            </button>
          </div>

          {/* Status Banner */}
          <div className={`p-3 rounded mb-3 ${
            lfbmEnabled ? 'bg-green-900 border border-green-600' : 'bg-red-900 border border-red-600'
          }`}>
            <div className={`font-bold text-sm ${lfbmEnabled ? 'text-green-200' : 'text-red-200'}`}>
              RunPod/LFBM: {lfbmEnabled ? 'ENABLED' : 'BLOCKED'}
            </div>
            {isOverrideActive && countdown && (
              <div className="text-white text-lg font-mono mt-1">
                {countdown} min remaining
              </div>
            )}
            <div className="text-[10px] opacity-75 mt-1 text-slate-300">
              LF_PROD_ENABLE: {status?.currentState?.lfProdEnable ?? 'unknown'}
            </div>
          </div>

          {/* Enable/Disable Controls */}
          <div className="bg-slate-900 rounded p-3 mb-3">
            <div className="text-slate-300 text-xs font-bold mb-2">
              Temporary Override
            </div>

            {/* Duration Picker */}
            <div className="flex items-center gap-2 mb-3">
              <span className="text-slate-400 text-xs">Duration:</span>
              <div className="flex items-center bg-slate-700 rounded">
                <button
                  onClick={decrementDuration}
                  className="px-3 py-1 text-white hover:bg-slate-600 rounded-l text-lg"
                  disabled={duration <= MIN_DURATION}
                >
                  -
                </button>
                <input
                  type="number"
                  value={duration}
                  onChange={handleDurationChange}
                  min={MIN_DURATION}
                  max={MAX_DURATION}
                  className="w-14 bg-slate-800 text-white text-center text-sm py-1 border-none outline-none [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                />
                <button
                  onClick={incrementDuration}
                  className="px-3 py-1 text-white hover:bg-slate-600 rounded-r text-lg"
                  disabled={duration >= MAX_DURATION}
                >
                  +
                </button>
              </div>
              <span className="text-slate-400 text-xs">min</span>
            </div>

            {/* Quick Duration Presets */}
            <div className="flex flex-wrap gap-1 mb-3">
              {[10, 15, 30, 45, 60, 90].map(mins => (
                <button
                  key={mins}
                  onClick={() => setDuration(mins)}
                  className={`px-2 py-1 text-xs rounded ${
                    duration === mins
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                  }`}
                >
                  {mins}m
                </button>
              ))}
            </div>

            {/* Enable/Disable Buttons */}
            <div className="flex gap-2">
              {!isOverrideActive ? (
                <button
                  onClick={() => setShowEnableConfirm(true)}
                  className="flex-1 bg-amber-600 hover:bg-amber-700 text-white text-sm py-2 rounded font-bold"
                >
                  ENABLE for {duration}m
                </button>
              ) : (
                <button
                  onClick={() => setShowDisableConfirm(true)}
                  className="flex-1 bg-red-600 hover:bg-red-700 text-white text-sm py-2 rounded font-bold"
                >
                  DISABLE
                </button>
              )}
            </div>
          </div>

          <p className="text-slate-400 text-xs mb-3">
            Individual toggles below (browser-only, doesn't affect server):
          </p>

          <div className="space-y-2 mb-4">
            {Object.entries(toggles).map(([key, enabled]) => (
              <label
                key={key}
                className="flex items-center justify-between cursor-pointer hover:bg-slate-700 p-1.5 rounded"
              >
                <span className="text-slate-200 text-sm capitalize">
                  {key === 'lfbm' ? 'LFBM (RunPod)' : key.toUpperCase()}
                </span>
                <button
                  onClick={() => handleToggle(key as keyof ApiToggleState)}
                  className={`w-10 h-5 rounded-full transition-colors ${
                    enabled ? 'bg-green-500' : 'bg-slate-600'
                  }`}
                >
                  <div
                    className={`w-4 h-4 bg-white rounded-full transition-transform mx-0.5 ${
                      enabled ? 'translate-x-5' : 'translate-x-0'
                    }`}
                  />
                </button>
              </label>
            ))}
          </div>

          <div className="flex gap-2">
            <button
              onClick={enableAll}
              className="flex-1 bg-green-600 hover:bg-green-700 text-white text-xs py-1.5 rounded"
            >
              All On
            </button>
            <button
              onClick={disableAll}
              className="flex-1 bg-slate-600 hover:bg-slate-700 text-white text-xs py-1.5 rounded"
            >
              All Off
            </button>
          </div>

          <p className="text-slate-500 text-[10px] mt-3 text-center">
            Non-prod environment • Crons not affected
          </p>
        </div>
      )}
    </>
  );
}

// Helper to check if APIs are enabled (for use in API routes)
export function isApiEnabled(apiName: keyof ApiToggleState): boolean {
  // In production, always enabled
  if (process.env.VERCEL_ENV === 'production' || process.env.NODE_ENV === 'production') {
    return true;
  }

  // Primary control: LF_PROD_ENABLE
  if (process.env.LF_PROD_ENABLE === 'true') {
    return true;
  }

  // Check for environment override
  if (process.env.ENABLE_APIS_IN_PREVIEW === 'true') {
    return true;
  }

  // For server-side, we can't check localStorage
  // Would need to pass via header or cookie
  return false;
}
