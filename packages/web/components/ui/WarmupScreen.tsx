'use client';

import { useState, useEffect, useCallback } from 'react';

// ============================================================================
// WARMUP SCREEN - Beautiful loading experience while cache warms
// ============================================================================
// Displays MJ-generated art loops (5-7 second perfect loops via Grok)
// while the intelligence briefing cache warms up.
//
// Video assets should be placed in: /public/warmup/
// Naming: warmup-{theme}-{number}.mp4 (e.g., warmup-cosmos-1.mp4)
// ============================================================================

export interface WarmupScreenProps {
  /** Estimated wait time in seconds */
  estimatedWaitSeconds?: number;
  /** Message to display */
  message?: string;
  /** Callback when warmup is complete */
  onComplete?: () => void;
  /** Preset being loaded (affects theme) */
  preset?: 'global' | 'nato' | 'brics' | 'conflict';
  /** Custom video URL (overrides theme selection) */
  videoUrl?: string;
  /** Polling interval in ms (default: 5000) */
  pollInterval?: number;
  /** API endpoint to poll */
  pollEndpoint?: string;
  /** Request body for polling */
  pollBody?: object;
}

// Video themes mapped to presets - MJ art loops
const THEME_VIDEOS: Record<string, string[]> = {
  global: [
    '/warmup/cosmos-network-1.mp4',
    '/warmup/earth-pulse-1.mp4',
    '/warmup/global-flow-1.mp4',
  ],
  nato: [
    '/warmup/atlantic-waves-1.mp4',
    '/warmup/alliance-grid-1.mp4',
  ],
  brics: [
    '/warmup/emerging-sunrise-1.mp4',
    '/warmup/silk-road-1.mp4',
  ],
  conflict: [
    '/warmup/tension-ripple-1.mp4',
    '/warmup/strategic-depth-1.mp4',
  ],
  default: [
    '/warmup/neural-flow-1.mp4',
    '/warmup/data-stream-1.mp4',
  ],
};

// Fallback gradients if videos aren't available
const THEME_GRADIENTS: Record<string, string> = {
  global: 'from-blue-900 via-purple-900 to-indigo-900',
  nato: 'from-blue-900 via-blue-800 to-slate-900',
  brics: 'from-amber-900 via-orange-900 to-red-900',
  conflict: 'from-red-900 via-slate-900 to-gray-900',
  default: 'from-slate-900 via-blue-900 to-purple-900',
};

// Loading phrases that rotate
const LOADING_PHRASES = [
  'Synthesizing intelligence streams...',
  'Correlating global signals...',
  'Analyzing geopolitical vectors...',
  'Processing nation-state indicators...',
  'Aggregating risk assessments...',
  'Computing basin dynamics...',
  'Mapping transition probabilities...',
  'Calibrating confidence intervals...',
  'Fusing multi-source intel...',
  'Rendering strategic landscape...',
];

export function WarmupScreen({
  estimatedWaitSeconds = 30,
  message,
  onComplete,
  preset = 'global',
  videoUrl,
  pollInterval = 5000,
  pollEndpoint = '/api/intel-briefing',
  pollBody,
}: WarmupScreenProps) {
  const [progress, setProgress] = useState(0);
  const [currentPhrase, setCurrentPhrase] = useState(0);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [videoError, setVideoError] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);

  // Select a random video from the theme on mount
  useEffect(() => {
    if (videoUrl) {
      setSelectedVideo(videoUrl);
    } else {
      const videos = THEME_VIDEOS[preset] || THEME_VIDEOS.default;
      const randomVideo = videos[Math.floor(Math.random() * videos.length)];
      setSelectedVideo(randomVideo);
    }
  }, [preset, videoUrl]);

  // Progress animation
  useEffect(() => {
    const interval = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1);
      setProgress((prev) => {
        // Asymptotic approach to 95% - never quite reaches 100% until complete
        const targetProgress = Math.min(95, (elapsedSeconds / estimatedWaitSeconds) * 100);
        return prev + (targetProgress - prev) * 0.1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [estimatedWaitSeconds, elapsedSeconds]);

  // Rotate loading phrases
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentPhrase((prev) => (prev + 1) % LOADING_PHRASES.length);
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  // Poll for completion
  const checkCache = useCallback(async () => {
    try {
      const response = await fetch(pollEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(pollBody || { preset }),
      });

      const data = await response.json();

      // If we got actual briefings (not "warming" status), we're done
      if (data.briefings && Object.keys(data.briefings).length > 0) {
        setProgress(100);
        onComplete?.();
        return true;
      }

      return false;
    } catch {
      // Network error - keep polling
      return false;
    }
  }, [pollEndpoint, pollBody, preset, onComplete]);

  useEffect(() => {
    // Start polling after initial delay
    const initialDelay = setTimeout(() => {
      checkCache();
    }, 2000);

    const interval = setInterval(() => {
      checkCache();
    }, pollInterval);

    return () => {
      clearTimeout(initialDelay);
      clearInterval(interval);
    };
  }, [checkCache, pollInterval]);

  const gradient = THEME_GRADIENTS[preset] || THEME_GRADIENTS.default;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center overflow-hidden">
      {/* Background video or gradient fallback */}
      {selectedVideo && !videoError ? (
        <video
          autoPlay
          loop
          muted
          playsInline
          className="absolute inset-0 w-full h-full object-cover"
          onError={() => setVideoError(true)}
        >
          <source src={selectedVideo} type="video/mp4" />
        </video>
      ) : (
        <div className={`absolute inset-0 bg-gradient-to-br ${gradient} animate-pulse`} />
      )}

      {/* Overlay for readability */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />

      {/* Content */}
      <div className="relative z-10 flex flex-col items-center justify-center px-8 text-center max-w-2xl">
        {/* Logo/Icon */}
        <div className="mb-8">
          <div className="relative w-24 h-24">
            {/* Pulsing rings */}
            <div className="absolute inset-0 rounded-full border-2 border-cyan-400/30 animate-ping" />
            <div className="absolute inset-2 rounded-full border border-cyan-400/50 animate-pulse" />
            <div className="absolute inset-4 rounded-full border border-cyan-400/70" />

            {/* Center icon */}
            <div className="absolute inset-0 flex items-center justify-center">
              <svg
                className="w-12 h-12 text-cyan-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                />
              </svg>
            </div>
          </div>
        </div>

        {/* Title */}
        <h2 className="text-2xl md:text-3xl font-light text-white mb-4 tracking-wide">
          {message || 'Initializing Intelligence Feed'}
        </h2>

        {/* Dynamic phrase */}
        <p className="text-cyan-300/80 text-lg mb-8 h-6 transition-all duration-500">
          {LOADING_PHRASES[currentPhrase]}
        </p>

        {/* Progress bar */}
        <div className="w-full max-w-md mb-4">
          <div className="h-1 bg-white/10 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full transition-all duration-1000 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Time estimate */}
        <p className="text-white/50 text-sm">
          {progress < 95 ? (
            <>
              Estimated time: ~{Math.max(1, Math.round(estimatedWaitSeconds - elapsedSeconds))}s
            </>
          ) : (
            'Almost ready...'
          )}
        </p>

        {/* Preset indicator */}
        <div className="mt-8 px-4 py-2 rounded-full bg-white/5 border border-white/10">
          <span className="text-white/60 text-xs uppercase tracking-widest">
            {preset.toUpperCase()} Intelligence Briefing
          </span>
        </div>
      </div>

      {/* Animated particles (optional visual flair) */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-cyan-400/30 rounded-full animate-float"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 5}s`,
              animationDuration: `${5 + Math.random() * 10}s`,
            }}
          />
        ))}
      </div>
    </div>
  );
}

export default WarmupScreen;
