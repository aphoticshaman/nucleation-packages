/**
 * LATTICEFORGE INTELLIGENCE COMPONENT LIBRARY
 *
 * Neuro-Symbolic UI Components for Geopolitical Intelligence Analysis
 *
 * Design System:
 * - Neural (Cyan #06b6d4): AI/ML-driven insights, predictions, neural network outputs
 * - Symbolic (Amber #f59e0b): Rule-based logic, causal reasoning, symbolic AI
 * - Risk (Red #ef4444): Critical alerts, high-risk indicators, warnings
 * - Background: Slate-900 with subtle gradients
 *
 * Component Categories:
 * 1. Core Visualizations (1-10)
 * 2. Data Display (11-20)
 * 3. Interactive Controls (21-30)
 * 4. Epistemic/Uncertainty (31-40)
 * 5. Collaboration (41-50)
 * 6. Utilities (51+)
 *
 * Total: 28 specialized intelligence analysis components
 */

// =============================================================================
// DESIGN SYSTEM
// =============================================================================
export * from '@/lib/design-system';

// =============================================================================
// CORE VISUALIZATIONS (1-10)
// =============================================================================

// Component 01: Neural Ticker Tape
export { NeuralTicker, mockTickerItems } from './NeuralTicker';

// Component 02: Velocity Sparklines
export { VelocitySparkline, SparklineCell, generateMockSparklineData } from './VelocitySparkline';

// Component 03: Pulse Staleness Indicator
export { PulseIndicator, PulseIndicatorLarge, useStaleness } from './PulseIndicator';

// Component 04: Signal Feed with Entity Chips & Source Badges
export { SignalFeed, SourceBadge, EntityChip, mockSignals } from './SignalFeed';

// Component 05: Sentiment Ambient Background
export { SentimentBackground, SentimentBorder, SentimentIndicator, AggregateSentiment } from './SentimentBackground';

// Component 06: Risk Weather Gauge
export { RiskGauge, RiskGaugeMini, DefconGauge } from './RiskGauge';

// Component 07: Event Horizon Time Slider
export { TimeSlider, TimeSliderMini } from './TimeSlider';

// Component 08: Causal Chain Navigator
export { CausalChain, mockCausalNodes, mockCausalEdges } from './CausalChain';

// Component 09: Logic Tree Inspector
export { LogicTree, mockLogicTreeData } from './LogicTree';

// Component 10: SHAP Value Waterfall
export { ShapWaterfall, ShapBar, mockShapFeatures } from './ShapWaterfall';

// =============================================================================
// DATA DISPLAY (11-20)
// =============================================================================

// Component 11: Confidence Donut with Uncertainty
export { ConfidenceDonut, ConfidenceBar, ConfidenceComparison } from './ConfidenceDonut';

// Component 12: Command Palette (Ctrl+K)
export { CommandPalette, useCommandPalette } from './CommandPalette';

// Component 13: Toast Notification Stack
export { ToastProvider, useToast, InlineToast } from './ToastNotifications';

// Component 14: Jargon Decoder Tooltip (60+ terms)
export { JargonDecoder, TermTooltip, getJargonDefinition, JARGON_DICTIONARY } from './JargonDecoder';

// Component 15: Prediction Confidence Cone
export { ConfidenceCone, generateMockForecast } from './ConfidenceCone';

// Component 16: Token Attention Highlighter
export { TokenAttention, TokenHeatmap, AttentionComparison, mockTokenWeights } from './TokenAttention';

// Component 17: Sortable Data Table with Risk Coloring
export { DataTable, RiskCell, TrendCell, StatusCell } from './DataTable';

// Component 18: Citation Manager with NATO Admiralty Codes
export { CitationManager, CitationRef, mockCitations } from './CitationManager';

// Component 19: Force-Directed Network Graph
export { NetworkGraph, mockNetworkData } from './NetworkGraph';

// Component 20: Alert Threshold Configuration
export { AlertThresholds, mockThresholds } from './AlertThresholds';

// =============================================================================
// EPISTEMIC / UNCERTAINTY (21-30)
// =============================================================================

// Component 21: Epistemic Dashboard - Knowledge Quadrant Visualization
export {
  EpistemicDashboard,
  FuzzyDisplay,
  BoundedConfidence,
  mockEpistemicClaims,
  mockKnowledgeGaps
} from './EpistemicDashboard';

// Component 22: Source Reliability - NATO Admiralty Code Indicator
export {
  SourceReliability,
  ReliabilityMatrix,
  InlineReliability,
  AggregateReliability
} from './SourceReliability';

// =============================================================================
// INTERACTIVE CONTROLS (31-40)
// =============================================================================

// Component 31: Keyboard Shortcuts Reference
export {
  KeyboardShortcuts,
  useKeyboardShortcuts,
  ShortcutHint,
  CompactShortcutRef
} from './KeyboardShortcuts';

// Component 32: What-If Scenario Builder
export {
  ScenarioBuilder,
  mockScenarios,
  mockScenarioVariables
} from './ScenarioBuilder';

// Component 33: Timeline Player with Playback Controls
export {
  TimelinePlayer,
  CompactTimeline,
  mockTimelineEvents
} from './TimelinePlayer';

// =============================================================================
// GEOSPATIAL (41-45)
// =============================================================================

// Component 41: Hexbin Map - Geospatial Heatmap
export { HexbinMap, mockHexbinPoints } from './HexbinMap';

// =============================================================================
// COLLABORATION (46-50)
// =============================================================================

// Component 46: Annotation Layer - Collaborative Notes
export { AnnotationLayer, mockAnnotations } from './AnnotationLayer';

// Component 47: Watchlist Manager - Entity Tracking
export { WatchlistManager, mockWatchlists } from './WatchlistManager';

// =============================================================================
// RE-EXPORT EPISTEMIC ENGINE TYPES AND FUNCTIONS
// =============================================================================

export type {
  KnowledgeQuadrant,
  EpistemicClaim,
  HistoricalCorrelate,
  Hypothesis,
  Derivative,
  AblationResult,
  FuzzyNumber,
  KnowledgeGap,
  EpistemicProof,
} from '@/lib/epistemic-engine';

export {
  fuzzyAdd,
  fuzzyMultiply,
  fuzzyIntersect,
  defuzzify,
  fuzzyRisk,
  applyEpistemicBounds,
  detectKnowledgeQuadrant,
  EPISTEMIC_PROOFS,
  HISTORICAL_EVENTS_DATABASE,
} from '@/lib/epistemic-engine';
