'use client';

import { useEffect, useState, useMemo } from 'react';

interface SentimentBackgroundProps {
  sentiment: number; // -1 (negative) to 1 (positive)
  intensity?: number; // 0-1, how prominent the effect is
  animated?: boolean;
  children?: React.ReactNode;
}

// Component 05: Sentiment Ambient Background
export function SentimentBackground({
  sentiment,
  intensity = 0.3,
  animated = true,
  children,
}: SentimentBackgroundProps) {
  const [displaySentiment, setDisplaySentiment] = useState(sentiment);

  // Smooth transition of sentiment value
  useEffect(() => {
    if (!animated) {
      setDisplaySentiment(sentiment);
      return;
    }

    const step = (sentiment - displaySentiment) * 0.1;
    if (Math.abs(step) < 0.001) {
      setDisplaySentiment(sentiment);
      return;
    }

    const timer = requestAnimationFrame(() => {
      setDisplaySentiment(prev => prev + step);
    });

    return () => cancelAnimationFrame(timer);
  }, [sentiment, displaySentiment, animated]);

  // Calculate gradient colors based on sentiment
  const gradientColors = useMemo(() => {
    // Negative: Deep red/maroon
    // Neutral: Slate
    // Positive: Teal/green

    if (displaySentiment < -0.2) {
      // Negative
      const negIntensity = Math.min(1, Math.abs(displaySentiment));
      return {
        primary: `rgba(127, 29, 29, ${intensity * negIntensity})`, // red-900
        secondary: `rgba(153, 27, 27, ${intensity * negIntensity * 0.5})`, // red-800
        tertiary: `rgba(185, 28, 28, ${intensity * negIntensity * 0.3})`, // red-700
      };
    } else if (displaySentiment > 0.2) {
      // Positive
      const posIntensity = Math.min(1, displaySentiment);
      return {
        primary: `rgba(17, 94, 89, ${intensity * posIntensity})`, // teal-800
        secondary: `rgba(19, 78, 74, ${intensity * posIntensity * 0.5})`, // teal-900
        tertiary: `rgba(20, 184, 166, ${intensity * posIntensity * 0.2})`, // teal-500
      };
    } else {
      // Neutral
      return {
        primary: `rgba(30, 41, 59, ${intensity * 0.3})`, // slate-800
        secondary: `rgba(51, 65, 85, ${intensity * 0.2})`, // slate-700
        tertiary: `rgba(71, 85, 105, ${intensity * 0.1})`, // slate-600
      };
    }
  }, [displaySentiment, intensity]);

  // Generate mesh gradient positions
  const meshPositions = useMemo(() => {
    const seed = Math.abs(displaySentiment * 1000);
    return [
      { x: 10 + (seed % 20), y: 10 + (seed % 30) },
      { x: 70 + (seed % 25), y: 20 + (seed % 20) },
      { x: 30 + (seed % 15), y: 80 + (seed % 15) },
      { x: 80 + (seed % 10), y: 70 + (seed % 25) },
    ];
  }, [displaySentiment]);

  return (
    <div className="relative w-full h-full overflow-hidden">
      {/* Ambient gradient mesh */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: `
            radial-gradient(ellipse at ${meshPositions[0].x}% ${meshPositions[0].y}%, ${gradientColors.primary} 0%, transparent 50%),
            radial-gradient(ellipse at ${meshPositions[1].x}% ${meshPositions[1].y}%, ${gradientColors.secondary} 0%, transparent 40%),
            radial-gradient(ellipse at ${meshPositions[2].x}% ${meshPositions[2].y}%, ${gradientColors.tertiary} 0%, transparent 60%),
            radial-gradient(ellipse at ${meshPositions[3].x}% ${meshPositions[3].y}%, ${gradientColors.primary} 0%, transparent 45%)
          `,
          filter: 'blur(60px)',
          transition: animated ? 'all 2s ease-out' : 'none',
        }}
      />

      {/* Border glow */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          boxShadow: `
            inset 0 0 100px ${gradientColors.primary},
            inset 0 0 200px ${gradientColors.secondary}
          `,
          transition: animated ? 'all 1s ease-out' : 'none',
        }}
      />

      {/* Content */}
      <div className="relative z-10 w-full h-full">
        {children}
      </div>
    </div>
  );
}

// Compact border-only variant
export function SentimentBorder({
  sentiment,
  children,
  className = '',
}: {
  sentiment: number;
  children: React.ReactNode;
  className?: string;
}) {
  const borderColor = useMemo(() => {
    if (sentiment < -0.3) return 'border-red-500/50 shadow-red-500/20';
    if (sentiment > 0.3) return 'border-emerald-500/50 shadow-emerald-500/20';
    return 'border-slate-600/50';
  }, [sentiment]);

  return (
    <div
      className={`
        border-2 rounded-xl transition-all duration-500
        ${borderColor}
        ${Math.abs(sentiment) > 0.3 ? 'shadow-lg' : ''}
        ${className}
      `}
    >
      {children}
    </div>
  );
}

// Inline sentiment indicator
export function SentimentIndicator({
  sentiment,
  size = 'md',
  showLabel = false,
}: {
  sentiment: number;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
}) {
  const sizeClasses = {
    sm: 'w-16 h-1',
    md: 'w-24 h-1.5',
    lg: 'w-32 h-2',
  };

  const label = sentiment < -0.3 ? 'Negative' : sentiment > 0.3 ? 'Positive' : 'Neutral';
  const color = sentiment < -0.3 ? 'bg-red-500' : sentiment > 0.3 ? 'bg-emerald-500' : 'bg-slate-500';

  // Map -1 to 1 range to 0 to 100 for positioning
  const position = ((sentiment + 1) / 2) * 100;

  return (
    <div className="flex items-center gap-2">
      <div className={`relative ${sizeClasses[size]} bg-gradient-to-r from-red-900 via-slate-700 to-emerald-900 rounded-full overflow-hidden`}>
        {/* Position indicator */}
        <div
          className={`absolute top-0 bottom-0 w-1 ${color} transition-all duration-300`}
          style={{ left: `${position}%`, transform: 'translateX(-50%)' }}
        />
      </div>
      {showLabel && (
        <span className={`text-xs ${
          sentiment < -0.3 ? 'text-red-400' :
          sentiment > 0.3 ? 'text-emerald-400' :
          'text-slate-400'
        }`}>
          {label} ({sentiment.toFixed(2)})
        </span>
      )}
    </div>
  );
}

// Aggregate sentiment from multiple sources
export function AggregateSentiment({
  sources,
}: {
  sources: { name: string; sentiment: number; weight: number }[];
}) {
  const aggregate = useMemo(() => {
    const totalWeight = sources.reduce((a, s) => a + s.weight, 0);
    return sources.reduce((a, s) => a + s.sentiment * s.weight, 0) / totalWeight;
  }, [sources]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-400">Aggregate Sentiment</span>
        <SentimentIndicator sentiment={aggregate} size="sm" />
      </div>
      <div className="space-y-1">
        {sources.map(source => (
          <div key={source.name} className="flex items-center justify-between text-xs">
            <span className="text-slate-500">{source.name}</span>
            <div className="flex items-center gap-2">
              <span className="text-slate-600 font-mono">{source.weight.toFixed(1)}x</span>
              <SentimentIndicator sentiment={source.sentiment} size="sm" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
