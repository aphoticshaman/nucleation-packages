// LatticeForge Intelligence Components
// Neuro-Symbolic UI Components for Intelligence Analysis

// Design System
export * from '@/lib/design-system';

// Component 01: Neural Ticker Tape
export { NeuralTicker, mockTickerItems } from './NeuralTicker';

// Component 02: Velocity Sparklines
export { VelocitySparkline, SparklineCell, generateMockSparklineData } from './VelocitySparkline';

// Component 03: Pulse Staleness Indicator
export { PulseIndicator, PulseIndicatorLarge, useStaleness } from './PulseIndicator';

// Component 04, 06, 07, 10: Signal Feed with Entity Chips & Source Badges
export { SignalFeed, SourceBadge, EntityChip, mockSignals } from './SignalFeed';

// Component 18: Risk Weather Gauge
export { RiskGauge, RiskGaugeMini, DefconGauge } from './RiskGauge';

// Component 21: Event Horizon Time Slider
export { TimeSlider, TimeSliderMini } from './TimeSlider';

// Component 31: Logic Tree Inspector
export { LogicTree, mockLogicTreeData } from './LogicTree';

// Component 32: SHAP Value Waterfall
export { ShapWaterfall, ShapBar, mockShapFeatures } from './ShapWaterfall';

// Component 41: Command Palette (Ctrl+K)
export { CommandPalette, useCommandPalette } from './CommandPalette';
