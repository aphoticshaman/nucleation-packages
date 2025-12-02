'use client';

import { useState, useEffect, useCallback, createContext, useContext } from 'react';

type ToastType = 'info' | 'success' | 'warning' | 'error' | 'critical';

interface Toast {
  id: string;
  type: ToastType;
  title: string;
  message?: string;
  duration?: number; // ms, 0 = no auto-dismiss
  action?: {
    label: string;
    onClick: () => void;
  };
  domain?: string;
  timestamp: number;
}

interface ToastContextType {
  toasts: Toast[];
  addToast: (toast: Omit<Toast, 'id' | 'timestamp'>) => string;
  removeToast: (id: string) => void;
  clearAll: () => void;
}

const ToastContext = createContext<ToastContextType | null>(null);

// Component 43: Smart Toast Notification Stack
export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = useCallback((toast: Omit<Toast, 'id' | 'timestamp'>) => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const newToast: Toast = {
      ...toast,
      id,
      timestamp: Date.now(),
      duration: toast.duration ?? (toast.type === 'critical' ? 0 : 5000),
    };

    setToasts(prev => [...prev, newToast]);
    return id;
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  const clearAll = useCallback(() => {
    setToasts([]);
  }, []);

  return (
    <ToastContext.Provider value={{ toasts, addToast, removeToast, clearAll }}>
      {children}
      <ToastContainer />
    </ToastContext.Provider>
  );
}

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within ToastProvider');
  }
  return context;
}

// Toast container (renders the stack)
function ToastContainer() {
  const { toasts, removeToast } = useToast();

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-md w-full pointer-events-none">
      {toasts.map((toast, index) => (
        <ToastItem
          key={toast.id}
          toast={toast}
          onDismiss={() => removeToast(toast.id)}
          index={index}
        />
      ))}
    </div>
  );
}

// Individual toast item
function ToastItem({
  toast,
  onDismiss,
  index,
}: {
  toast: Toast;
  onDismiss: () => void;
  index: number;
}) {
  const [isVisible, setIsVisible] = useState(false);
  const [isLeaving, setIsLeaving] = useState(false);

  // Entrance animation
  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), 50);
    return () => clearTimeout(timer);
  }, []);

  // Auto-dismiss
  useEffect(() => {
    if (toast.duration === 0) return;

    const timer = setTimeout(() => {
      setIsLeaving(true);
      setTimeout(onDismiss, 300);
    }, toast.duration);

    return () => clearTimeout(timer);
  }, [toast.duration, onDismiss]);

  const handleDismiss = () => {
    setIsLeaving(true);
    setTimeout(onDismiss, 300);
  };

  const config = getToastConfig(toast.type);

  return (
    <div
      className={`
        pointer-events-auto
        transform transition-all duration-300 ease-out
        ${isVisible && !isLeaving
          ? 'translate-x-0 opacity-100'
          : 'translate-x-full opacity-0'
        }
      `}
      style={{
        transitionDelay: `${index * 50}ms`,
      }}
    >
      <div
        className={`
          relative overflow-hidden rounded-lg border backdrop-blur-xl
          ${config.bg} ${config.border}
        `}
        style={{
          boxShadow: toast.type === 'critical'
            ? `0 0 30px ${config.glow}`
            : '0 10px 40px rgba(0,0,0,0.3)',
        }}
      >
        {/* Critical pulse overlay */}
        {toast.type === 'critical' && (
          <div className="absolute inset-0 bg-red-500/10 animate-pulse" />
        )}

        <div className="relative p-4">
          <div className="flex items-start gap-3">
            {/* Icon */}
            <div className={`text-xl ${config.iconColor}`}>
              {config.icon}
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <h4 className={`text-sm font-medium ${config.textColor}`}>
                  {toast.title}
                </h4>
                {toast.domain && (
                  <span className="text-xs text-slate-500 uppercase tracking-wider">
                    {toast.domain}
                  </span>
                )}
              </div>
              {toast.message && (
                <p className="text-sm text-slate-400 mt-1">
                  {toast.message}
                </p>
              )}
              {toast.action && (
                <button
                  onClick={toast.action.onClick}
                  className={`
                    mt-2 text-sm font-medium
                    ${config.actionColor}
                    hover:underline
                  `}
                >
                  {toast.action.label}
                </button>
              )}
            </div>

            {/* Dismiss button */}
            <button
              onClick={handleDismiss}
              className="text-slate-500 hover:text-slate-300 transition-colors"
            >
              ✕
            </button>
          </div>

          {/* Progress bar for auto-dismiss */}
          {toast.duration !== 0 && (
            <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-slate-700">
              <div
                className={`h-full ${config.progressColor} transition-all linear`}
                style={{
                  animation: `shrink ${toast.duration}ms linear forwards`,
                }}
              />
            </div>
          )}
        </div>
      </div>

      <style jsx>{`
        @keyframes shrink {
          from { width: 100%; }
          to { width: 0%; }
        }
      `}</style>
    </div>
  );
}

// Toast type configurations
function getToastConfig(type: ToastType) {
  switch (type) {
    case 'success':
      return {
        icon: '✓',
        bg: 'bg-emerald-950/90',
        border: 'border-emerald-500/50',
        iconColor: 'text-emerald-400',
        textColor: 'text-emerald-200',
        actionColor: 'text-emerald-400',
        progressColor: 'bg-emerald-500',
        glow: 'rgba(16, 185, 129, 0.2)',
      };
    case 'warning':
      return {
        icon: '⚠',
        bg: 'bg-amber-950/90',
        border: 'border-amber-500/50',
        iconColor: 'text-amber-400',
        textColor: 'text-amber-200',
        actionColor: 'text-amber-400',
        progressColor: 'bg-amber-500',
        glow: 'rgba(245, 158, 11, 0.2)',
      };
    case 'error':
      return {
        icon: '✗',
        bg: 'bg-red-950/90',
        border: 'border-red-500/50',
        iconColor: 'text-red-400',
        textColor: 'text-red-200',
        actionColor: 'text-red-400',
        progressColor: 'bg-red-500',
        glow: 'rgba(239, 68, 68, 0.2)',
      };
    case 'critical':
      return {
        icon: '🚨',
        bg: 'bg-red-950/95',
        border: 'border-red-500',
        iconColor: 'text-red-400',
        textColor: 'text-red-100',
        actionColor: 'text-red-300',
        progressColor: 'bg-red-500',
        glow: 'rgba(239, 68, 68, 0.5)',
      };
    default: // info
      return {
        icon: 'ℹ',
        bg: 'bg-slate-900/90',
        border: 'border-cyan-500/50',
        iconColor: 'text-cyan-400',
        textColor: 'text-slate-200',
        actionColor: 'text-cyan-400',
        progressColor: 'bg-cyan-500',
        glow: 'rgba(6, 182, 212, 0.2)',
      };
  }
}

// Standalone toast for non-provider use
export function InlineToast({
  type,
  title,
  message,
  onDismiss,
}: {
  type: ToastType;
  title: string;
  message?: string;
  onDismiss?: () => void;
}) {
  const config = getToastConfig(type);

  return (
    <div className={`p-4 rounded-lg border ${config.bg} ${config.border}`}>
      <div className="flex items-start gap-3">
        <span className={`text-xl ${config.iconColor}`}>{config.icon}</span>
        <div className="flex-1">
          <h4 className={`text-sm font-medium ${config.textColor}`}>{title}</h4>
          {message && <p className="text-sm text-slate-400 mt-1">{message}</p>}
        </div>
        {onDismiss && (
          <button onClick={onDismiss} className="text-slate-500 hover:text-slate-300">
            ✕
          </button>
        )}
      </div>
    </div>
  );
}
