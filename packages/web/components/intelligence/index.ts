// LatticeForge Intelligence Components
// Neuro-Symbolic UI Components for Intelligence Analysis

// Design System
export * from '@/lib/design-system';

// Component 01: Neural Ticker Tape
export { NeuralTicker, mockTickerItems } from './NeuralTicker';
export type { } from './NeuralTicker';

// Component 03: Pulse Staleness Indicator
export { PulseIndicator, PulseIndicatorLarge, useStaleness } from './PulseIndicator';

// Component 04, 06, 07, 10: Signal Feed with Entity Chips & Source Badges
export { SignalFeed, SourceBadge, EntityChip, mockSignals } from './SignalFeed';

// Component 31: Logic Tree Inspector
export { LogicTree, mockLogicTreeData } from './LogicTree';

// Component 41: Command Palette (Ctrl+K)
export { CommandPalette, useCommandPalette } from './CommandPalette';
