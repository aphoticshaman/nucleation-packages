'use client';

import { useState, useRef, useEffect, useCallback, useMemo } from 'react';

interface TimelineEvent {
  id: string;
  timestamp: string;
  title: string;
  description?: string;
  category: string;
  severity?: 'low' | 'medium' | 'high' | 'critical';
  entities?: string[];
  coordinates?: { lat: number; lng: number };
  metadata?: Record<string, unknown>;
}

interface TimelinePlayerProps {
  events: TimelineEvent[];
  startDate?: Date;
  endDate?: Date;
  onTimeChange?: (currentTime: Date) => void;
  onEventClick?: (event: TimelineEvent) => void;
  autoPlay?: boolean;
  playbackSpeed?: number; // milliseconds per tick
  tickDuration?: number; // how much time passes per tick (in hours)
  showEventMarkers?: boolean;
  height?: number;
}

// Component 50: Animated Timeline with Playback Controls
export function TimelinePlayer({
  events,
  startDate,
  endDate,
  onTimeChange,
  onEventClick,
  autoPlay = false,
  playbackSpeed = 100,
  tickDuration = 24, // 1 day per tick
  showEventMarkers = true,
  height = 120,
}: TimelinePlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const [currentTime, setCurrentTime] = useState<Date | null>(null);
  const [speed, setSpeed] = useState(playbackSpeed);
  const [hoveredEvent, setHoveredEvent] = useState<string | null>(null);
  const playbackRef = useRef<NodeJS.Timeout | null>(null);

  // Calculate timeline bounds
  const bounds = useMemo(() => {
    if (startDate && endDate) {
      return { start: startDate, end: endDate };
    }

    const timestamps = events.map(e => new Date(e.timestamp).getTime());
    if (timestamps.length === 0) {
      const now = new Date();
      return { start: new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000), end: now };
    }

    const minTime = Math.min(...timestamps);
    const maxTime = Math.max(...timestamps);
    const padding = (maxTime - minTime) * 0.1;

    return {
      start: new Date(minTime - padding),
      end: new Date(maxTime + padding),
    };
  }, [events, startDate, endDate]);

  // Initialize current time
  useEffect(() => {
    if (!currentTime) {
      setCurrentTime(bounds.start);
    }
  }, [bounds, currentTime]);

  // Playback logic
  useEffect(() => {
    if (isPlaying && currentTime) {
      playbackRef.current = setInterval(() => {
        setCurrentTime(prev => {
          if (!prev) return bounds.start;

          const next = new Date(prev.getTime() + tickDuration * 60 * 60 * 1000);
          if (next >= bounds.end) {
            setIsPlaying(false);
            return bounds.end;
          }
          return next;
        });
      }, speed);
    }

    return () => {
      if (playbackRef.current) {
        clearInterval(playbackRef.current);
      }
    };
  }, [isPlaying, speed, tickDuration, bounds]);

  // Notify parent of time changes
  useEffect(() => {
    if (currentTime) {
      onTimeChange?.(currentTime);
    }
  }, [currentTime, onTimeChange]);

  // Calculate position from time
  const timeToPosition = useCallback((time: Date) => {
    const totalDuration = bounds.end.getTime() - bounds.start.getTime();
    const elapsed = time.getTime() - bounds.start.getTime();
    return (elapsed / totalDuration) * 100;
  }, [bounds]);

  // Calculate time from position
  const positionToTime = useCallback((position: number) => {
    const totalDuration = bounds.end.getTime() - bounds.start.getTime();
    const elapsed = (position / 100) * totalDuration;
    return new Date(bounds.start.getTime() + elapsed);
  }, [bounds]);

  // Handle scrubbing
  const handleScrub = useCallback((e: React.MouseEvent) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;

    const position = ((e.clientX - rect.left) / rect.width) * 100;
    const time = positionToTime(Math.max(0, Math.min(100, position)));
    setCurrentTime(time);
    setIsPlaying(false);
  }, [positionToTime]);

  // Speed controls
  const speedOptions = [
    { label: '0.5x', value: playbackSpeed * 2 },
    { label: '1x', value: playbackSpeed },
    { label: '2x', value: playbackSpeed / 2 },
    { label: '4x', value: playbackSpeed / 4 },
    { label: '8x', value: playbackSpeed / 8 },
  ];

  // Group events by severity for coloring
  const severityColors = {
    critical: '#ef4444',
    high: '#f59e0b',
    medium: '#06b6d4',
    low: '#22c55e',
  };

  // Get events visible at current time
  const visibleEvents = useMemo(() => {
    if (!currentTime) return [];
    return events.filter(e => new Date(e.timestamp) <= currentTime);
  }, [events, currentTime]);

  // Generate tick marks
  const ticks = useMemo(() => {
    const totalDuration = bounds.end.getTime() - bounds.start.getTime();
    const tickCount = 10;
    const tickInterval = totalDuration / tickCount;

    return Array.from({ length: tickCount + 1 }, (_, i) => {
      const time = new Date(bounds.start.getTime() + i * tickInterval);
      return {
        position: (i / tickCount) * 100,
        label: formatTickLabel(time, totalDuration),
      };
    });
  }, [bounds]);

  if (!currentTime) return null;

  const currentPosition = timeToPosition(currentTime);

  return (
    <div className="bg-slate-900/50 rounded-lg border border-slate-700 p-4" style={{ height }}>
      {/* Controls */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {/* Play/Pause */}
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="w-10 h-10 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center hover:bg-cyan-500/30 transition-colors"
          >
            {isPlaying ? '⏸' : '▶'}
          </button>

          {/* Skip controls */}
          <button
            onClick={() => setCurrentTime(bounds.start)}
            className="p-2 text-slate-400 hover:text-slate-200"
            title="Go to start"
          >
            ⏮
          </button>
          <button
            onClick={() => {
              const prev = new Date(currentTime.getTime() - tickDuration * 60 * 60 * 1000 * 10);
              setCurrentTime(prev < bounds.start ? bounds.start : prev);
            }}
            className="p-2 text-slate-400 hover:text-slate-200"
            title="Step back"
          >
            ⏪
          </button>
          <button
            onClick={() => {
              const next = new Date(currentTime.getTime() + tickDuration * 60 * 60 * 1000 * 10);
              setCurrentTime(next > bounds.end ? bounds.end : next);
            }}
            className="p-2 text-slate-400 hover:text-slate-200"
            title="Step forward"
          >
            ⏩
          </button>
          <button
            onClick={() => setCurrentTime(bounds.end)}
            className="p-2 text-slate-400 hover:text-slate-200"
            title="Go to end"
          >
            ⏭
          </button>
        </div>

        {/* Current time display */}
        <div className="text-center">
          <div className="text-sm font-mono text-cyan-400">
            {currentTime.toLocaleDateString()} {currentTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </div>
          <div className="text-xs text-slate-500">
            {visibleEvents.length} / {events.length} events
          </div>
        </div>

        {/* Speed control */}
        <div className="flex items-center gap-1">
          <span className="text-xs text-slate-500 mr-2">Speed:</span>
          {speedOptions.map(opt => (
            <button
              key={opt.label}
              onClick={() => setSpeed(opt.value)}
              className={`px-2 py-1 rounded text-xs transition-colors ${
                speed === opt.value
                  ? 'bg-cyan-500/20 text-cyan-400'
                  : 'bg-slate-700 text-slate-400 hover:text-slate-200'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Timeline track */}
      <div
        ref={containerRef}
        className="relative h-12 bg-slate-800 rounded-lg cursor-pointer"
        onClick={handleScrub}
      >
        {/* Tick marks */}
        {ticks.map((tick, i) => (
          <div
            key={i}
            className="absolute top-0 h-full flex flex-col items-center"
            style={{ left: `${tick.position}%` }}
          >
            <div className="w-px h-2 bg-slate-600" />
            <span className="text-xs text-slate-500 mt-auto mb-1 transform -translate-x-1/2">
              {tick.label}
            </span>
          </div>
        ))}

        {/* Progress bar */}
        <div
          className="absolute top-0 h-full bg-cyan-500/20 rounded-l-lg transition-all duration-100"
          style={{ width: `${currentPosition}%` }}
        />

        {/* Event markers */}
        {showEventMarkers && events.map(event => {
          const position = timeToPosition(new Date(event.timestamp));
          const color = severityColors[event.severity || 'low'];
          const isVisible = new Date(event.timestamp) <= currentTime;
          const isHovered = hoveredEvent === event.id;

          return (
            <div
              key={event.id}
              className="absolute top-1/2 transform -translate-y-1/2 -translate-x-1/2 z-10"
              style={{ left: `${position}%` }}
              onMouseEnter={() => setHoveredEvent(event.id)}
              onMouseLeave={() => setHoveredEvent(null)}
              onClick={(e) => {
                e.stopPropagation();
                onEventClick?.(event);
              }}
            >
              <div
                className={`w-3 h-3 rounded-full border-2 transition-all ${
                  isVisible ? 'opacity-100' : 'opacity-30'
                } ${isHovered ? 'scale-150' : ''}`}
                style={{
                  backgroundColor: isVisible ? color : 'transparent',
                  borderColor: color,
                }}
              />

              {/* Tooltip */}
              {isHovered && (
                <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 pointer-events-none z-20">
                  <div className="bg-slate-800 border border-slate-600 rounded-lg shadow-xl p-2 whitespace-nowrap">
                    <div className="text-xs font-medium text-slate-200">{event.title}</div>
                    <div className="text-xs text-slate-400">
                      {new Date(event.timestamp).toLocaleString()}
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}

        {/* Playhead */}
        <div
          className="absolute top-0 h-full w-0.5 bg-cyan-400 z-20 pointer-events-none"
          style={{ left: `${currentPosition}%` }}
        >
          <div className="absolute -top-1 left-1/2 transform -translate-x-1/2 w-3 h-3 bg-cyan-400 rounded-full" />
        </div>
      </div>
    </div>
  );
}

// Format tick label based on duration
function formatTickLabel(date: Date, totalDuration: number): string {
  const oneDay = 24 * 60 * 60 * 1000;
  const oneWeek = 7 * oneDay;
  const oneMonth = 30 * oneDay;

  if (totalDuration < oneDay) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } else if (totalDuration < oneWeek) {
    return date.toLocaleDateString([], { weekday: 'short', day: 'numeric' });
  } else if (totalDuration < oneMonth * 3) {
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  } else {
    return date.toLocaleDateString([], { month: 'short', year: '2-digit' });
  }
}

// Compact timeline for embedding
export function CompactTimeline({
  events,
  currentTime,
  onTimeChange,
  height = 40,
}: {
  events: TimelineEvent[];
  currentTime: Date;
  onTimeChange: (time: Date) => void;
  height?: number;
}) {
  const timestamps = events.map(e => new Date(e.timestamp).getTime());
  const minTime = Math.min(...timestamps);
  const maxTime = Math.max(...timestamps);
  const currentPos = ((currentTime.getTime() - minTime) / (maxTime - minTime)) * 100;

  return (
    <div className="relative bg-slate-800/50 rounded h-10" style={{ height }}>
      {events.map(event => {
        const pos = ((new Date(event.timestamp).getTime() - minTime) / (maxTime - minTime)) * 100;
        return (
          <div
            key={event.id}
            className="absolute top-1/2 w-1 h-3 bg-cyan-500/50 rounded transform -translate-y-1/2"
            style={{ left: `${pos}%` }}
          />
        );
      })}
      <div
        className="absolute top-0 h-full w-0.5 bg-cyan-400"
        style={{ left: `${currentPos}%` }}
      />
    </div>
  );
}

// Mock timeline events
export const mockTimelineEvents: TimelineEvent[] = [
  {
    id: '1',
    timestamp: '2024-01-01T08:00:00Z',
    title: 'Military buildup detected',
    category: 'military',
    severity: 'high',
    entities: ['Russia', 'Ukraine'],
  },
  {
    id: '2',
    timestamp: '2024-01-03T14:30:00Z',
    title: 'Diplomatic talks suspended',
    category: 'political',
    severity: 'medium',
    entities: ['NATO', 'Russia'],
  },
  {
    id: '3',
    timestamp: '2024-01-05T09:00:00Z',
    title: 'Energy supply disruption',
    category: 'economic',
    severity: 'high',
    entities: ['Europe', 'Russia'],
  },
  {
    id: '4',
    timestamp: '2024-01-08T16:45:00Z',
    title: 'Cyberattack on infrastructure',
    category: 'cyber',
    severity: 'critical',
    entities: ['Ukraine'],
  },
  {
    id: '5',
    timestamp: '2024-01-10T11:00:00Z',
    title: 'Humanitarian corridor established',
    category: 'humanitarian',
    severity: 'low',
    entities: ['UN', 'Ukraine'],
  },
  {
    id: '6',
    timestamp: '2024-01-12T19:00:00Z',
    title: 'Sanctions package announced',
    category: 'economic',
    severity: 'medium',
    entities: ['EU', 'US', 'Russia'],
  },
  {
    id: '7',
    timestamp: '2024-01-15T07:30:00Z',
    title: 'Offensive operation launched',
    category: 'military',
    severity: 'critical',
    entities: ['Ukraine', 'Russia'],
  },
];
