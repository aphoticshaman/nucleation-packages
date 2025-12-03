'use client';

import { useState, ReactNode, createContext, useContext } from 'react';

// Panel nesting context
interface PanelContextValue {
  depth: number;
  parentId: string | null;
}

const PanelContext = createContext<PanelContextValue>({ depth: 0, parentId: null });

interface NestedPanelProps {
  id: string;
  title: string;
  icon?: string;
  badge?: string | number;
  badgeColor?: 'blue' | 'green' | 'amber' | 'red' | 'purple' | 'slate';
  children: ReactNode;
  defaultOpen?: boolean;
  collapsible?: boolean;
  variant?: 'default' | 'compact' | 'minimal';
  className?: string;
  headerAction?: ReactNode;
  onToggle?: (open: boolean) => void;
}

const BADGE_COLORS = {
  blue: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  green: 'bg-green-500/20 text-green-400 border-green-500/30',
  amber: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  red: 'bg-red-500/20 text-red-400 border-red-500/30',
  purple: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  slate: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
};

export function NestedPanel({
  id,
  title,
  icon,
  badge,
  badgeColor = 'slate',
  children,
  defaultOpen = false,
  collapsible = true,
  variant = 'default',
  className = '',
  headerAction,
  onToggle,
}: NestedPanelProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const { depth } = useContext(PanelContext);

  const handleToggle = () => {
    if (!collapsible) return;
    const newState = !isOpen;
    setIsOpen(newState);
    onToggle?.(newState);
  };

  // Adjust styling based on nesting depth
  const depthStyles = {
    0: 'bg-slate-900 border-slate-700',
    1: 'bg-slate-800/80 border-slate-600',
    2: 'bg-slate-700/60 border-slate-500',
    3: 'bg-slate-600/40 border-slate-400',
  };

  const bgStyle = depthStyles[Math.min(depth, 3) as keyof typeof depthStyles];

  const variantStyles = {
    default: 'rounded-xl border p-0',
    compact: 'rounded-lg border p-0',
    minimal: 'rounded-md border-l-2 border-t-0 border-r-0 border-b-0 pl-3',
  };

  return (
    <PanelContext.Provider value={{ depth: depth + 1, parentId: id }}>
      <div className={`${variantStyles[variant]} ${bgStyle} ${className} overflow-hidden`}>
        {/* Header */}
        <button
          onClick={handleToggle}
          className={`
            w-full flex items-center justify-between gap-2 text-left
            ${variant === 'default' ? 'px-4 py-3' : variant === 'compact' ? 'px-3 py-2' : 'py-2'}
            ${collapsible ? 'hover:bg-white/5 cursor-pointer' : 'cursor-default'}
            transition-colors
          `}
          disabled={!collapsible}
        >
          <div className="flex items-center gap-2 min-w-0">
            {icon && <span className="text-lg flex-shrink-0">{icon}</span>}
            <span className={`font-medium text-slate-200 truncate ${
              variant === 'compact' ? 'text-sm' : ''
            }`}>
              {title}
            </span>
            {badge !== undefined && (
              <span className={`
                px-1.5 py-0.5 text-xs rounded border flex-shrink-0
                ${BADGE_COLORS[badgeColor]}
              `}>
                {badge}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {headerAction}
            {collapsible && (
              <svg
                className={`w-4 h-4 text-slate-500 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            )}
          </div>
        </button>

        {/* Content */}
        <div className={`
          transition-all duration-200 ease-in-out overflow-hidden
          ${isOpen || !collapsible ? 'max-h-[2000px] opacity-100' : 'max-h-0 opacity-0'}
        `}>
          <div className={`
            ${variant === 'default' ? 'px-4 pb-4' : variant === 'compact' ? 'px-3 pb-3' : 'pb-2'}
            ${variant !== 'minimal' ? 'border-t border-slate-700/50' : ''}
          `}>
            <div className={variant !== 'minimal' ? 'pt-3' : ''}>
              {children}
            </div>
          </div>
        </div>
      </div>
    </PanelContext.Provider>
  );
}

// Grid of panels
interface PanelGridProps {
  children: ReactNode;
  cols?: 1 | 2 | 3 | 4;
  gap?: 'sm' | 'md' | 'lg';
}

export function PanelGrid({ children, cols = 2, gap = 'md' }: PanelGridProps) {
  const colClasses = {
    1: 'grid-cols-1',
    2: 'grid-cols-1 md:grid-cols-2',
    3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-4',
  };

  const gapClasses = {
    sm: 'gap-2',
    md: 'gap-4',
    lg: 'gap-6',
  };

  return (
    <div className={`grid ${colClasses[cols]} ${gapClasses[gap]}`}>
      {children}
    </div>
  );
}

// Tab panel for horizontal switching
interface TabPanelProps {
  tabs: Array<{
    id: string;
    label: string;
    icon?: string;
    badge?: string | number;
    content: ReactNode;
  }>;
  defaultTab?: string;
  variant?: 'default' | 'pills' | 'underline';
}

export function TabPanel({ tabs, defaultTab, variant = 'default' }: TabPanelProps) {
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.id);

  const activeContent = tabs.find(t => t.id === activeTab)?.content;

  const tabStyles = {
    default: {
      container: 'flex gap-1 bg-slate-800/50 rounded-lg p-1',
      tab: 'px-3 py-1.5 rounded-md text-sm transition-colors',
      active: 'bg-slate-700 text-white',
      inactive: 'text-slate-400 hover:text-white hover:bg-slate-700/50',
    },
    pills: {
      container: 'flex gap-2',
      tab: 'px-4 py-2 rounded-full text-sm transition-colors',
      active: 'bg-blue-600 text-white',
      inactive: 'text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700',
    },
    underline: {
      container: 'flex gap-4 border-b border-slate-700',
      tab: 'px-1 py-2 text-sm transition-colors border-b-2 -mb-px',
      active: 'border-blue-500 text-blue-400',
      inactive: 'border-transparent text-slate-400 hover:text-white',
    },
  };

  const styles = tabStyles[variant];

  return (
    <div className="space-y-4">
      <div className={styles.container}>
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`
              ${styles.tab}
              ${activeTab === tab.id ? styles.active : styles.inactive}
              flex items-center gap-1.5
            `}
          >
            {tab.icon && <span>{tab.icon}</span>}
            <span>{tab.label}</span>
            {tab.badge !== undefined && (
              <span className="px-1.5 py-0.5 text-xs rounded bg-white/10">
                {tab.badge}
              </span>
            )}
          </button>
        ))}
      </div>
      <div>{activeContent}</div>
    </div>
  );
}

// Quick action panel (always visible, opens overlay)
interface QuickPanelProps {
  trigger: ReactNode;
  children: ReactNode;
  position?: 'bottom' | 'right' | 'center';
  size?: 'sm' | 'md' | 'lg' | 'full';
}

export function QuickPanel({ trigger, children, position = 'center', size = 'md' }: QuickPanelProps) {
  const [isOpen, setIsOpen] = useState(false);

  const sizeClasses = {
    sm: 'max-w-sm',
    md: 'max-w-lg',
    lg: 'max-w-2xl',
    full: 'max-w-full mx-4',
  };

  const positionClasses = {
    bottom: 'items-end justify-center pb-4',
    right: 'items-center justify-end pr-4',
    center: 'items-center justify-center',
  };

  return (
    <>
      <div onClick={() => setIsOpen(true)} className="cursor-pointer">
        {trigger}
      </div>

      {isOpen && (
        <div
          className={`fixed inset-0 z-50 flex ${positionClasses[position]} bg-black/60 backdrop-blur-sm`}
          onClick={() => setIsOpen(false)}
        >
          <div
            className={`
              ${sizeClasses[size]} w-full bg-slate-900 rounded-xl border border-slate-700
              shadow-2xl animate-in fade-in slide-in-from-bottom-4 duration-200
            `}
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-end p-2 border-b border-slate-700">
              <button
                onClick={() => setIsOpen(false)}
                className="p-1 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="p-4 max-h-[80vh] overflow-y-auto">
              {children}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

// Splitview panel (side by side)
interface SplitPanelProps {
  left: ReactNode;
  right: ReactNode;
  defaultRatio?: number;
  minWidth?: number;
}

export function SplitPanel({ left, right, defaultRatio = 0.5 }: SplitPanelProps) {
  const [ratio, setRatio] = useState(defaultRatio);

  return (
    <div className="flex h-full">
      <div style={{ width: `${ratio * 100}%` }} className="overflow-auto">
        {left}
      </div>
      <div
        className="w-1 bg-slate-700 hover:bg-blue-500 cursor-col-resize flex-shrink-0"
        onMouseDown={(e) => {
          const startX = e.clientX;
          const startRatio = ratio;
          const container = e.currentTarget.parentElement;
          if (!container) return;
          const containerWidth = container.offsetWidth;

          const handleMouseMove = (e: MouseEvent) => {
            const delta = e.clientX - startX;
            const newRatio = Math.max(0.2, Math.min(0.8, startRatio + delta / containerWidth));
            setRatio(newRatio);
          };

          const handleMouseUp = () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
          };

          document.addEventListener('mousemove', handleMouseMove);
          document.addEventListener('mouseup', handleMouseUp);
        }}
      />
      <div style={{ width: `${(1 - ratio) * 100}%` }} className="overflow-auto">
        {right}
      </div>
    </div>
  );
}

export default NestedPanel;
