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

// Component 05: Sentiment Ambient Background
export { SentimentBackground, SentimentBorder, SentimentIndicator, AggregateSentiment } from './SentimentBackground';

// Component 18: Risk Weather Gauge
export { RiskGauge, RiskGaugeMini, DefconGauge } from './RiskGauge';

// Component 21: Event Horizon Time Slider
export { TimeSlider, TimeSliderMini } from './TimeSlider';

// Component 22: Causal Chain Navigator
export { CausalChain, mockCausalNodes, mockCausalEdges } from './CausalChain';

// Component 31: Logic Tree Inspector
export { LogicTree, mockLogicTreeData } from './LogicTree';

// Component 32: SHAP Value Waterfall
export { ShapWaterfall, ShapBar, mockShapFeatures } from './ShapWaterfall';

// Component 33: Confidence Donut with Uncertainty
export { ConfidenceDonut, ConfidenceBar, ConfidenceComparison } from './ConfidenceDonut';

// Component 41: Command Palette (Ctrl+K)
export { CommandPalette, useCommandPalette } from './CommandPalette';

// Component 43: Toast Notification Stack
export { ToastProvider, useToast, InlineToast } from './ToastNotifications';

// Component 08: Jargon Decoder Tooltip
export { JargonDecoder, TermTooltip, getJargonDefinition, JARGON_DICTIONARY } from './JargonDecoder';

// Component 25: Prediction Confidence Cone
export { ConfidenceCone, generateMockForecast } from './ConfidenceCone';

// Component 34: Token Attention Highlighter
export { TokenAttention, TokenHeatmap, AttentionComparison, mockTokenWeights } from './TokenAttention';
