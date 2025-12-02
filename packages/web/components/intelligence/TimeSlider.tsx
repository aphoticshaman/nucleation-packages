'use client';

import { useState, useRef, useCallback, useEffect, useMemo } from 'react';

interface TimeEvent {
  timestamp: number;
  type: 'event' | 'milestone' | 'alert';
  label?: string;
}

interface TimeSliderProps {
  startTime: number; // Unix timestamp
  endTime: number;
  currentTime: number;
  onTimeChange: (time: number) => void;
  events?: TimeEvent[];
  predictedTime?: number; // Where predictions start
  isPlaying?: boolean;
  onPlayToggle?: () => void;
  playSpeed?: number; // 1x, 2x, 10x, etc.
  onSpeedChange?: (speed: number) => void;
}

// Component 21: Event Horizon Time Slider
export function TimeSlider({
  startTime,
  endTime,
  currentTime,
  onTimeChange,
  events = [],
  predictedTime,
  isPlaying = false,
  onPlayToggle,
  playSpeed = 1,
  onSpeedChange,
}: TimeSliderProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [hoverTime, setHoverTime] = useState<number | null>(null);

  const range = endTime - startTime;
  const currentPct = ((currentTime - startTime) / range) * 100;
  const predictedPct = predictedTime
    ? ((predictedTime - startTime) / range) * 100
    : 100;

  // Handle mouse/touch interactions
  const getTimeFromPosition = useCallback((clientX: number) => {
    if (!trackRef.current) return currentTime;
    const rect = trackRef.current.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    return startTime + pct * range;
  }, [startTime, range, currentTime]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true);
    onTimeChange(getTimeFromPosition(e.clientX));
  }, [getTimeFromPosition, onTimeChange]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const time = getTimeFromPosition(e.clientX);
    setHoverTime(time);
    if (isDragging) {
      onTimeChange(time);
    }
  }, [isDragging, getTimeFromPosition, onTimeChange]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setHoverTime(null);
    setIsDragging(false);
  }, []);

  // Format time display
  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    });
  };

  // Time markers for the track
  const markers = useMemo(() => {
    const count = 5;
    const step = range / (count - 1);
    return Array.from({ length: count }, (_, i) => ({
      time: startTime + i * step,
      pct: (i / (count - 1)) * 100,
    }));
  }, [startTime, range]);

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-4">
      {/* Controls bar */}
      <div className="flex items-center gap-4 mb-3">
        {/* Play/Pause */}
        <button
          onClick={onPlayToggle}
          className={`
            w-10 h-10 rounded-lg flex items-center justify-center transition-colors
            ${isPlaying
              ? 'bg-cyan-500/20 text-cyan-400'
              : 'bg-slate-800 text-slate-400 hover:text-white'
            }
          `}
        >
          {isPlaying ? '⏸' : '▶'}
        </button>

        {/* Speed selector */}
        <div className="flex items-center gap-1">
          {[1, 2, 10, 60].map(speed => (
            <button
              key={speed}
              onClick={() => onSpeedChange?.(speed)}
              className={`
                px-2 py-1 rounded text-xs font-mono transition-colors
                ${playSpeed === speed
                  ? 'bg-cyan-500/20 text-cyan-400'
                  : 'bg-slate-800 text-slate-500 hover:text-white'
                }
              `}
            >
              {speed}x
            </button>
          ))}
        </div>

        {/* Current time display */}
        <div className="flex-1 text-center">
          <span className="text-lg font-mono text-slate-200">
            {formatTime(currentTime)}
          </span>
          <span className="text-sm text-slate-500 ml-2">
            {formatDate(currentTime)}
          </span>
        </div>

        {/* Jump buttons */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => onTimeChange(startTime)}
            className="px-2 py-1 rounded text-xs bg-slate-800 text-slate-400 hover:text-white"
          >
            ⏮ Start
          </button>
          <button
            onClick={() => onTimeChange(Date.now())}
            className="px-2 py-1 rounded text-xs bg-slate-800 text-slate-400 hover:text-white"
          >
            Now
          </button>
          <button
            onClick={() => onTimeChange(endTime)}
            className="px-2 py-1 rounded text-xs bg-slate-800 text-slate-400 hover:text-white"
          >
            End ⏭
          </button>
        </div>
      </div>

      {/* Timeline track */}
      <div className="relative">
        {/* Track background */}
        <div
          ref={trackRef}
          className="relative h-10 rounded-lg overflow-hidden cursor-pointer"
          style={{
            background: 'linear-gradient(to right, #1e293b 0%, #1e293b 100%)',
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
        >
          {/* Historical region (solid) */}
          <div
            className="absolute top-0 left-0 bottom-0 bg-slate-700/50"
            style={{ width: `${Math.min(currentPct, predictedPct)}%` }}
          />

          {/* Predicted region (dashed pattern) */}
          {predictedTime && (
            <div
              className="absolute top-0 bottom-0"
              style={{
                left: `${predictedPct}%`,
                right: 0,
                backgroundImage: `repeating-linear-gradient(
                  90deg,
                  transparent,
                  transparent 4px,
                  rgba(6, 182, 212, 0.1) 4px,
                  rgba(6, 182, 212, 0.1) 8px
                )`,
              }}
            />
          )}

          {/* Prediction boundary line */}
          {predictedTime && (
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-cyan-500"
              style={{ left: `${predictedPct}%` }}
            >
              <div className="absolute -top-5 left-1/2 -translate-x-1/2 text-xs text-cyan-400 whitespace-nowrap">
                PREDICTIONS →
              </div>
            </div>
          )}

          {/* Event markers */}
          {events.map((event, i) => {
            const pct = ((event.timestamp - startTime) / range) * 100;
            if (pct < 0 || pct > 100) return null;

            return (
              <div
                key={i}
                className="absolute top-0 bottom-0 w-1 group"
                style={{ left: `${pct}%` }}
              >
                <div
                  className={`
                    w-full h-full
                    ${event.type === 'milestone' ? 'bg-amber-500' : ''}
                    ${event.type === 'alert' ? 'bg-red-500' : ''}
                    ${event.type === 'event' ? 'bg-cyan-500/50' : ''}
                  `}
                />
                {event.label && (
                  <div className="absolute bottom-full mb-1 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <div className="px-2 py-1 bg-slate-800 rounded text-xs text-slate-300 whitespace-nowrap">
                      {event.label}
                    </div>
                  </div>
                )}
              </div>
            );
          })}

          {/* Hover indicator */}
          {hoverTime !== null && (
            <div
              className="absolute top-0 bottom-0 w-px bg-slate-500 pointer-events-none"
              style={{ left: `${((hoverTime - startTime) / range) * 100}%` }}
            >
              <div className="absolute -top-6 left-1/2 -translate-x-1/2 px-2 py-0.5 bg-slate-700 rounded text-xs text-slate-300">
                {formatTime(hoverTime)}
              </div>
            </div>
          )}

          {/* Current position handle */}
          <div
            className="absolute top-0 bottom-0 w-1 transition-all"
            style={{ left: `${currentPct}%` }}
          >
            <div className="absolute inset-0 bg-white rounded-full shadow-lg" />
            <div
              className="absolute -top-1 -bottom-1 -left-2 -right-2 bg-white/20 rounded-full"
              style={{ boxShadow: '0 0 10px rgba(255,255,255,0.5)' }}
            />
          </div>
        </div>

        {/* Time markers */}
        <div className="flex justify-between mt-1">
          {markers.map((marker, i) => (
            <span key={i} className="text-xs text-slate-500 font-mono">
              {formatTime(marker.time)}
            </span>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-3 text-xs text-slate-500">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-slate-700/50 rounded" />
          <span>Historical</span>
        </div>
        <div className="flex items-center gap-1">
          <div
            className="w-3 h-3 rounded"
            style={{
              backgroundImage: `repeating-linear-gradient(
                90deg,
                transparent,
                transparent 2px,
                rgba(6, 182, 212, 0.3) 2px,
                rgba(6, 182, 212, 0.3) 4px
              )`,
            }}
          />
          <span>Predicted (Neural)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-amber-500 rounded" />
          <span>Milestone</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-red-500 rounded" />
          <span>Alert</span>
        </div>
      </div>
    </div>
  );
}

// Compact variant for embedding
export function TimeSliderMini({
  startTime,
  endTime,
  currentTime,
  onTimeChange,
}: {
  startTime: number;
  endTime: number;
  currentTime: number;
  onTimeChange: (time: number) => void;
}) {
  const pct = ((currentTime - startTime) / (endTime - startTime)) * 100;

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-slate-500 font-mono w-12">
        {new Date(currentTime).toLocaleTimeString('en-US', {
          hour: '2-digit',
          minute: '2-digit',
        })}
      </span>
      <input
        type="range"
        min={startTime}
        max={endTime}
        value={currentTime}
        onChange={(e) => onTimeChange(Number(e.target.value))}
        className="flex-1 h-1 bg-slate-700 rounded-full appearance-none cursor-pointer
          [&::-webkit-slider-thumb]:appearance-none
          [&::-webkit-slider-thumb]:w-3
          [&::-webkit-slider-thumb]:h-3
          [&::-webkit-slider-thumb]:bg-cyan-400
          [&::-webkit-slider-thumb]:rounded-full
          [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(6,182,212,0.5)]"
      />
    </div>
  );
}
