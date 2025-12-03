/**
 * Dashboard Presets for LatticeForge
 *
 * Pre-configured dashboard setups so users don't face a blank slate.
 * Each preset is optimized for a specific use case and skill level.
 */

import { UserTier } from './powerUser';

// ============================================
// Preset Types
// ============================================

export type PresetId =
  | 'junior-analyst'
  | 'senior-analyst'
  | 'intelligence-officer'
  | 'executive-briefer'
  | 'crisis-monitor'
  | 'regional-specialist'
  | 'economic-analyst'
  | 'custom';

export interface WidgetConfig {
  id: string;
  type: WidgetType;
  title: string;
  position: { x: number; y: number; w: number; h: number };
  config: Record<string, unknown>;
  minimized?: boolean;
}

export type WidgetType =
  | 'executive-summary'
  | 'threat-matrix'
  | 'world-map'
  | 'regional-map'
  | 'tree-navigator'
  | 'timeline'
  | 'entity-list'
  | 'relation-graph'
  | 'news-feed'
  | 'alerts'
  | 'conflict-tracker'
  | 'resource-monitor'
  | 'demographic-chart'
  | 'sentiment-gauge'
  | 'cognitive-panel'
  | 'filter-panel'
  | 'bookmarks'
  | 'notes'
  | 'comparison-table'
  | 'trend-chart';

export interface FilterPreset {
  temporal: {
    range: 'week' | 'month' | 'quarter' | 'year' | 'custom';
    includeProjections: boolean;
  };
  categories: string[];
  regions: string[];
  severityThreshold: number;
  entityTypes: string[];
}

export interface NotificationPreset {
  emailDigest: 'none' | 'daily' | 'weekly';
  pushAlerts: boolean;
  alertSeverityThreshold: number;
  watchlistAlerts: boolean;
}

export interface DashboardPreset {
  id: PresetId;
  name: string;
  description: string;
  tagline: string;
  minTier: UserTier;
  widgets: WidgetConfig[];
  defaultFilters: FilterPreset;
  notifications: NotificationPreset;
  shortcuts: string[];
  tutorialSteps: TutorialStep[];
  recommendedFor: string[];
}

export interface TutorialStep {
  target: string; // CSS selector or widget ID
  title: string;
  content: string;
  action?: 'click' | 'hover' | 'scroll';
}

// ============================================
// Preset Definitions
// ============================================

export const DASHBOARD_PRESETS: Record<PresetId, DashboardPreset> = {
  'junior-analyst': {
    id: 'junior-analyst',
    name: 'Junior Analyst Desktop',
    description: 'Perfect starting point for those new to intelligence analysis. Focuses on the essentials without overwhelming you.',
    tagline: 'Start here. Learn the ropes.',
    minTier: 'explorer',
    widgets: [
      {
        id: 'exec-summary',
        type: 'executive-summary',
        title: 'Daily Brief',
        position: { x: 0, y: 0, w: 8, h: 4 },
        config: { detailLevel: 'simplified', maxItems: 5 },
      },
      {
        id: 'world-map',
        type: 'world-map',
        title: 'Global Overview',
        position: { x: 8, y: 0, w: 4, h: 4 },
        config: { defaultZoom: 1, showHotspots: true, simplified: true },
      },
      {
        id: 'news-feed',
        type: 'news-feed',
        title: 'Latest Intel',
        position: { x: 0, y: 4, w: 6, h: 4 },
        config: { maxItems: 10, autoRefresh: false },
      },
      {
        id: 'alerts',
        type: 'alerts',
        title: 'Important Alerts',
        position: { x: 6, y: 4, w: 6, h: 4 },
        config: { severityThreshold: 3, maxItems: 5 },
      },
    ],
    defaultFilters: {
      temporal: { range: 'week', includeProjections: false },
      categories: ['conflict', 'political', 'economic'],
      regions: ['Global'],
      severityThreshold: 2,
      entityTypes: ['country', 'organization'],
    },
    notifications: {
      emailDigest: 'daily',
      pushAlerts: false,
      alertSeverityThreshold: 4,
      watchlistAlerts: true,
    },
    shortcuts: ['?', 'r', 's'],
    tutorialSteps: [
      {
        target: '#exec-summary',
        title: 'Your Daily Brief',
        content: 'This is your executive summary. Every morning, start here to see what happened overnight and what matters today. Think of it as your "need to know" section.',
      },
      {
        target: '#world-map',
        title: 'The Big Picture',
        content: 'Red dots = active situations. Yellow = watch closely. Click any region to zoom in and see what\'s happening there.',
      },
      {
        target: '#news-feed',
        title: 'Intel Feed',
        content: 'Latest reports and analysis. We\'ve filtered out the noise so you see only what\'s relevant. Click any item to read the full report.',
      },
      {
        target: '#alerts',
        title: 'Alerts That Matter',
        content: 'When something big happens, it shows up here. Red = urgent. Orange = important. You can customize alert thresholds later.',
      },
    ],
    recommendedFor: [
      'New to intelligence analysis',
      'Students and researchers',
      'Journalists covering international affairs',
      'Business professionals monitoring global risks',
    ],
  },

  'senior-analyst': {
    id: 'senior-analyst',
    name: 'Senior Analyst Workstation',
    description: 'Full analytical toolkit with saved views, comparison tools, and deeper filtering. For those who know what they\'re looking for.',
    tagline: 'Dig deeper. Find patterns.',
    minTier: 'analyst',
    widgets: [
      {
        id: 'exec-summary',
        type: 'executive-summary',
        title: 'Executive Brief',
        position: { x: 0, y: 0, w: 6, h: 3 },
        config: { detailLevel: 'standard', maxItems: 8 },
      },
      {
        id: 'threat-matrix',
        type: 'threat-matrix',
        title: 'Threat Assessment',
        position: { x: 6, y: 0, w: 6, h: 3 },
        config: { showTrends: true, historicalComparison: true },
      },
      {
        id: 'tree-nav',
        type: 'tree-navigator',
        title: '3D Intel Explorer',
        position: { x: 0, y: 3, w: 6, h: 5 },
        config: { temporalDepth: 'month', showConnections: true },
      },
      {
        id: 'filter-panel',
        type: 'filter-panel',
        title: 'Analysis Filters',
        position: { x: 6, y: 3, w: 3, h: 5 },
        config: { showAdvanced: true, saveFilters: true },
      },
      {
        id: 'timeline',
        type: 'timeline',
        title: 'Event Timeline',
        position: { x: 9, y: 3, w: 3, h: 5 },
        config: { showProjections: true, linkRelated: true },
      },
      {
        id: 'entity-list',
        type: 'entity-list',
        title: 'Tracked Entities',
        position: { x: 0, y: 8, w: 4, h: 3 },
        config: { sortBy: 'activity', showRelations: true },
      },
      {
        id: 'comparison',
        type: 'comparison-table',
        title: 'Side-by-Side Analysis',
        position: { x: 4, y: 8, w: 4, h: 3 },
        config: { maxEntities: 4 },
      },
      {
        id: 'notes',
        type: 'notes',
        title: 'Analysis Notes',
        position: { x: 8, y: 8, w: 4, h: 3 },
        config: { autoSave: true, shareEnabled: false },
      },
    ],
    defaultFilters: {
      temporal: { range: 'month', includeProjections: true },
      categories: ['conflict', 'political', 'economic', 'military', 'social'],
      regions: ['Global'],
      severityThreshold: 1,
      entityTypes: ['country', 'organization', 'leader', 'military_unit'],
    },
    notifications: {
      emailDigest: 'daily',
      pushAlerts: true,
      alertSeverityThreshold: 3,
      watchlistAlerts: true,
    },
    shortcuts: ['?', 'r', 's', 'f', 'e', 'c', '1', '2', '3'],
    tutorialSteps: [
      {
        target: '#tree-nav',
        title: '3D Intel Explorer',
        content: 'Navigate intel in 3D space. Vertical axis = time (past below, future above). Rotate with mouse, click nodes to expand. This is where patterns emerge.',
      },
      {
        target: '#filter-panel',
        title: 'Power Filtering',
        content: 'Stack filters to drill down. Save filter combinations for reuse. Pro tip: Ctrl+click to add multiple values in the same category.',
      },
      {
        target: '#comparison',
        title: 'Compare Anything',
        content: 'Drag entities here to compare them side-by-side. Works with countries, organizations, leaders, or events.',
      },
    ],
    recommendedFor: [
      'Professional intelligence analysts',
      'Research institutions',
      'Strategic consultants',
      'Risk assessment teams',
    ],
  },

  'intelligence-officer': {
    id: 'intelligence-officer',
    name: 'Intelligence Officer Station',
    description: 'Full operational capability. Real-time feeds, relation mapping, predictive indicators, and team collaboration.',
    tagline: 'Operational intelligence. Real-time awareness.',
    minTier: 'strategist',
    widgets: [
      {
        id: 'exec-summary',
        type: 'executive-summary',
        title: 'Situation Report',
        position: { x: 0, y: 0, w: 4, h: 3 },
        config: { detailLevel: 'full', maxItems: 15, showSources: true },
      },
      {
        id: 'cognitive-panel',
        type: 'cognitive-panel',
        title: 'Analysis Confidence',
        position: { x: 4, y: 0, w: 2, h: 3 },
        config: { showXYZA: true, showFlow: true },
      },
      {
        id: 'regional-map',
        type: 'regional-map',
        title: 'Area of Operations',
        position: { x: 6, y: 0, w: 6, h: 4 },
        config: { layers: ['military', 'political', 'economic'], realTime: true },
      },
      {
        id: 'relation-graph',
        type: 'relation-graph',
        title: 'Entity Relations',
        position: { x: 0, y: 3, w: 6, h: 4 },
        config: { depth: 3, showStrength: true, temporal: true },
      },
      {
        id: 'conflict-tracker',
        type: 'conflict-tracker',
        title: 'Active Conflicts',
        position: { x: 6, y: 4, w: 6, h: 3 },
        config: { showProjections: true, resourceOverlay: true },
      },
      {
        id: 'alerts',
        type: 'alerts',
        title: 'Priority Alerts',
        position: { x: 0, y: 7, w: 3, h: 4 },
        config: { severityThreshold: 1, groupBy: 'region', realTime: true },
      },
      {
        id: 'timeline',
        type: 'timeline',
        title: 'Operational Timeline',
        position: { x: 3, y: 7, w: 5, h: 4 },
        config: { showProjections: true, showMilestones: true, interactive: true },
      },
      {
        id: 'resource-monitor',
        type: 'resource-monitor',
        title: 'Resource Flows',
        position: { x: 8, y: 7, w: 4, h: 4 },
        config: { types: ['energy', 'military', 'financial'], showDependencies: true },
      },
    ],
    defaultFilters: {
      temporal: { range: 'quarter', includeProjections: true },
      categories: ['conflict', 'political', 'economic', 'military', 'social', 'technological'],
      regions: ['Global'],
      severityThreshold: 0,
      entityTypes: ['country', 'organization', 'leader', 'military_unit', 'facility', 'event'],
    },
    notifications: {
      emailDigest: 'none',
      pushAlerts: true,
      alertSeverityThreshold: 2,
      watchlistAlerts: true,
    },
    shortcuts: ['?', 'r', 's', 'f', 'e', 'c', 'g', 'm', 't', 'a', '1', '2', '3', '4', '5'],
    tutorialSteps: [
      {
        target: '#relation-graph',
        title: 'Entity Relations',
        content: 'Interactive network graph. Double-click to expand connections. Edge thickness = relationship strength. Color = relationship type (green=alliance, red=hostile).',
      },
      {
        target: '#cognitive-panel',
        title: 'Analysis Confidence',
        content: 'XYZA metrics show how confident the system is. X=Coherence, Y=Complexity, Z=Reflection, A=Attunement. Watch the flow indicator for optimal analysis windows.',
      },
      {
        target: '#resource-monitor',
        title: 'Resource Dependencies',
        content: 'Sankey diagram showing resource flows. Click flows to see vulnerability points. Critical for understanding leverage and pressure points.',
      },
    ],
    recommendedFor: [
      'Government intelligence agencies',
      'Military planning staff',
      'National security teams',
      'Defense contractors',
    ],
  },

  'executive-briefer': {
    id: 'executive-briefer',
    name: 'Executive Briefer',
    description: 'Designed for those who brief leadership. Clean summaries, exportable reports, and presentation-ready visuals.',
    tagline: 'Brief up. Inform decisions.',
    minTier: 'analyst',
    widgets: [
      {
        id: 'exec-summary',
        type: 'executive-summary',
        title: 'Executive Summary',
        position: { x: 0, y: 0, w: 12, h: 4 },
        config: { detailLevel: 'executive', exportable: true, printOptimized: true },
      },
      {
        id: 'threat-matrix',
        type: 'threat-matrix',
        title: 'Threat Overview',
        position: { x: 0, y: 4, w: 6, h: 4 },
        config: { simplified: true, exportable: true },
      },
      {
        id: 'world-map',
        type: 'world-map',
        title: 'Global Situation',
        position: { x: 6, y: 4, w: 6, h: 4 },
        config: { presentationMode: true, annotations: true },
      },
      {
        id: 'trend-chart',
        type: 'trend-chart',
        title: 'Key Trends',
        position: { x: 0, y: 8, w: 8, h: 3 },
        config: { metrics: ['stability', 'threat', 'opportunity'], exportable: true },
      },
      {
        id: 'bookmarks',
        type: 'bookmarks',
        title: 'Key Items',
        position: { x: 8, y: 8, w: 4, h: 3 },
        config: { categories: ['brief', 'followup', 'background'] },
      },
    ],
    defaultFilters: {
      temporal: { range: 'week', includeProjections: true },
      categories: ['conflict', 'political', 'economic'],
      regions: ['Global'],
      severityThreshold: 3,
      entityTypes: ['country', 'organization', 'leader'],
    },
    notifications: {
      emailDigest: 'daily',
      pushAlerts: true,
      alertSeverityThreshold: 4,
      watchlistAlerts: true,
    },
    shortcuts: ['?', 'p', 'e', 'b'],
    tutorialSteps: [
      {
        target: '#exec-summary',
        title: 'The Brief',
        content: 'Click the export button to generate PDF or PowerPoint. Summaries auto-adjust based on your audience setting (board, executive, working level).',
      },
      {
        target: '#bookmarks',
        title: 'Prepare Your Brief',
        content: 'Save items here while researching. Drag to reorder. Export as appendix or background reading for your principals.',
      },
    ],
    recommendedFor: [
      'Executive assistants',
      'Chief of Staff offices',
      'Policy advisors',
      'Board briefers',
    ],
  },

  'crisis-monitor': {
    id: 'crisis-monitor',
    name: 'Crisis Monitor',
    description: 'Real-time crisis tracking and rapid response. Optimized for breaking situations.',
    tagline: 'When things go wrong. Fast.',
    minTier: 'analyst',
    widgets: [
      {
        id: 'alerts',
        type: 'alerts',
        title: 'FLASH TRAFFIC',
        position: { x: 0, y: 0, w: 12, h: 2 },
        config: { severityThreshold: 4, realTime: true, sound: true, fullWidth: true },
      },
      {
        id: 'regional-map',
        type: 'regional-map',
        title: 'Crisis Area',
        position: { x: 0, y: 2, w: 8, h: 5 },
        config: { realTime: true, trackMovement: true, showInfrastructure: true },
      },
      {
        id: 'timeline',
        type: 'timeline',
        title: 'Event Log',
        position: { x: 8, y: 2, w: 4, h: 5 },
        config: { realTime: true, autoScroll: true, showMinutes: true },
      },
      {
        id: 'news-feed',
        type: 'news-feed',
        title: 'Live Feed',
        position: { x: 0, y: 7, w: 6, h: 4 },
        config: { realTime: true, sources: ['osint', 'social', 'wire'], autoRefresh: true },
      },
      {
        id: 'entity-list',
        type: 'entity-list',
        title: 'Key Players',
        position: { x: 6, y: 7, w: 3, h: 4 },
        config: { sortBy: 'lastMentioned', showContact: true },
      },
      {
        id: 'notes',
        type: 'notes',
        title: 'Situation Log',
        position: { x: 9, y: 7, w: 3, h: 4 },
        config: { timestamped: true, autoSave: true },
      },
    ],
    defaultFilters: {
      temporal: { range: 'week', includeProjections: false },
      categories: ['conflict', 'political', 'military', 'humanitarian'],
      regions: ['Global'],
      severityThreshold: 3,
      entityTypes: ['country', 'organization', 'leader', 'military_unit', 'event'],
    },
    notifications: {
      emailDigest: 'none',
      pushAlerts: true,
      alertSeverityThreshold: 3,
      watchlistAlerts: true,
    },
    shortcuts: ['?', 'r', 'space', 'n', 'm'],
    tutorialSteps: [
      {
        target: '#alerts',
        title: 'Flash Traffic',
        content: 'Highest priority items appear here immediately. Sound alerts enabled by default. Press space to acknowledge and clear.',
      },
      {
        target: '#timeline',
        title: 'Live Event Log',
        content: 'Real-time event tracking. Click any event to see full details and related items. Auto-scrolls to newest.',
      },
    ],
    recommendedFor: [
      'Crisis response teams',
      'Situation rooms',
      'Emergency operations centers',
      'News desks during breaking events',
    ],
  },

  'regional-specialist': {
    id: 'regional-specialist',
    name: 'Regional Specialist',
    description: 'Deep dive into a specific region. Historical context, local entities, and cultural factors.',
    tagline: 'Know your AO inside and out.',
    minTier: 'analyst',
    widgets: [
      {
        id: 'regional-map',
        type: 'regional-map',
        title: 'Area of Responsibility',
        position: { x: 0, y: 0, w: 8, h: 5 },
        config: { defaultRegion: null, layers: ['political', 'ethnic', 'economic', 'infrastructure'] },
      },
      {
        id: 'exec-summary',
        type: 'executive-summary',
        title: 'Regional Brief',
        position: { x: 8, y: 0, w: 4, h: 5 },
        config: { detailLevel: 'regional', focusRegion: true },
      },
      {
        id: 'entity-list',
        type: 'entity-list',
        title: 'Key Actors',
        position: { x: 0, y: 5, w: 4, h: 3 },
        config: { filterByRegion: true, showBiographies: true },
      },
      {
        id: 'relation-graph',
        type: 'relation-graph',
        title: 'Local Networks',
        position: { x: 4, y: 5, w: 4, h: 3 },
        config: { filterByRegion: true, showTribes: true, showClans: true },
      },
      {
        id: 'demographic-chart',
        type: 'demographic-chart',
        title: 'Demographics',
        position: { x: 8, y: 5, w: 4, h: 3 },
        config: { showEthnic: true, showReligious: true, showAge: true },
      },
      {
        id: 'timeline',
        type: 'timeline',
        title: 'Regional History',
        position: { x: 0, y: 8, w: 8, h: 3 },
        config: { historicalDepth: 'decade', showCultural: true },
      },
      {
        id: 'notes',
        type: 'notes',
        title: 'Regional Notes',
        position: { x: 8, y: 8, w: 4, h: 3 },
        config: { categories: ['cultural', 'political', 'security'] },
      },
    ],
    defaultFilters: {
      temporal: { range: 'year', includeProjections: true },
      categories: ['conflict', 'political', 'economic', 'social', 'cultural'],
      regions: [], // Set during onboarding
      severityThreshold: 0,
      entityTypes: ['country', 'organization', 'leader', 'ethnic_group', 'religious_group'],
    },
    notifications: {
      emailDigest: 'daily',
      pushAlerts: true,
      alertSeverityThreshold: 2,
      watchlistAlerts: true,
    },
    shortcuts: ['?', 'r', 's', 'h', 'e'],
    tutorialSteps: [
      {
        target: '#regional-map',
        title: 'Your Region',
        content: 'Toggle map layers with the buttons in the corner. Ethnic, religious, economic, and infrastructure overlays help you understand the full picture.',
      },
      {
        target: '#relation-graph',
        title: 'Local Networks',
        content: 'Shows relationships between local actors including tribal, clan, and family ties. These connections often matter more than formal structures.',
      },
    ],
    recommendedFor: [
      'Country desk officers',
      'Regional specialists',
      'Cultural advisors',
      'Area studies researchers',
    ],
  },

  'economic-analyst': {
    id: 'economic-analyst',
    name: 'Economic Intelligence',
    description: 'Financial flows, trade dependencies, sanctions, and economic indicators.',
    tagline: 'Follow the money.',
    minTier: 'analyst',
    widgets: [
      {
        id: 'exec-summary',
        type: 'executive-summary',
        title: 'Economic Brief',
        position: { x: 0, y: 0, w: 6, h: 3 },
        config: { detailLevel: 'economic', categories: ['economic', 'trade', 'sanctions'] },
      },
      {
        id: 'trend-chart',
        type: 'trend-chart',
        title: 'Economic Indicators',
        position: { x: 6, y: 0, w: 6, h: 3 },
        config: { metrics: ['gdp', 'inflation', 'trade_balance', 'reserves'] },
      },
      {
        id: 'resource-monitor',
        type: 'resource-monitor',
        title: 'Trade Flows',
        position: { x: 0, y: 3, w: 6, h: 4 },
        config: { types: ['trade', 'energy', 'commodities'], showSanctions: true },
      },
      {
        id: 'relation-graph',
        type: 'relation-graph',
        title: 'Economic Networks',
        position: { x: 6, y: 3, w: 6, h: 4 },
        config: { relationTypes: ['trade', 'investment', 'sanctions'], showValue: true },
      },
      {
        id: 'entity-list',
        type: 'entity-list',
        title: 'Key Economic Actors',
        position: { x: 0, y: 7, w: 4, h: 4 },
        config: { types: ['corporation', 'bank', 'sovereign_fund'], showFinancials: true },
      },
      {
        id: 'alerts',
        type: 'alerts',
        title: 'Economic Alerts',
        position: { x: 4, y: 7, w: 4, h: 4 },
        config: { categories: ['economic', 'sanctions', 'trade'] },
      },
      {
        id: 'comparison',
        type: 'comparison-table',
        title: 'Economic Comparison',
        position: { x: 8, y: 7, w: 4, h: 4 },
        config: { metrics: ['gdp', 'growth', 'debt', 'trade'] },
      },
    ],
    defaultFilters: {
      temporal: { range: 'quarter', includeProjections: true },
      categories: ['economic', 'trade', 'sanctions', 'financial'],
      regions: ['Global'],
      severityThreshold: 1,
      entityTypes: ['country', 'organization', 'corporation', 'bank'],
    },
    notifications: {
      emailDigest: 'daily',
      pushAlerts: true,
      alertSeverityThreshold: 3,
      watchlistAlerts: true,
    },
    shortcuts: ['?', 'r', 's', 'e', 't'],
    tutorialSteps: [
      {
        target: '#resource-monitor',
        title: 'Trade Flow Analysis',
        content: 'Sankey diagram showing trade relationships. Width = volume. Red highlights = sanctioned flows. Click any flow to see commodity breakdown.',
      },
      {
        target: '#relation-graph',
        title: 'Economic Networks',
        content: 'Shows investment and trade relationships between entities. Useful for identifying dependencies and potential pressure points.',
      },
    ],
    recommendedFor: [
      'Economic intelligence analysts',
      'Sanctions compliance teams',
      'Trade policy advisors',
      'Financial crime investigators',
    ],
  },

  'custom': {
    id: 'custom',
    name: 'Custom Dashboard',
    description: 'Start with a blank canvas and build exactly what you need.',
    tagline: 'Your rules. Your way.',
    minTier: 'analyst',
    widgets: [],
    defaultFilters: {
      temporal: { range: 'month', includeProjections: true },
      categories: [],
      regions: ['Global'],
      severityThreshold: 0,
      entityTypes: [],
    },
    notifications: {
      emailDigest: 'weekly',
      pushAlerts: false,
      alertSeverityThreshold: 4,
      watchlistAlerts: false,
    },
    shortcuts: ['?'],
    tutorialSteps: [
      {
        target: '#widget-gallery',
        title: 'Add Widgets',
        content: 'Press "W" or click the + button to open the widget gallery. Drag widgets to your dashboard and resize them as needed.',
      },
      {
        target: '#settings',
        title: 'Configure Everything',
        content: 'Right-click any widget to access settings. Every aspect is customizable. Save layouts for different purposes.',
      },
    ],
    recommendedFor: [
      'Power users who know exactly what they want',
      'Teams with specific workflow requirements',
      'Anyone who\'s outgrown the presets',
    ],
  },
};

// ============================================
// Preset Selection Helpers
// ============================================

/**
 * Get presets available for a user tier
 */
export function getAvailablePresets(tier: UserTier): DashboardPreset[] {
  const tierRank: Record<UserTier, number> = {
    explorer: 0,
    analyst: 1,
    strategist: 2,
    architect: 3,
  };

  const userRank = tierRank[tier];

  return Object.values(DASHBOARD_PRESETS).filter(
    preset => tierRank[preset.minTier] <= userRank
  );
}

/**
 * Recommend presets based on user profile
 */
export function recommendPresets(profile: {
  tier: UserTier;
  experience: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  interests: string[];
  role?: string;
}): PresetId[] {
  const available = getAvailablePresets(profile.tier);
  const recommendations: PresetId[] = [];

  // Experience-based primary recommendation
  if (profile.experience === 'beginner') {
    recommendations.push('junior-analyst');
  } else if (profile.experience === 'intermediate') {
    recommendations.push('senior-analyst');
  } else if (profile.experience === 'advanced') {
    if (profile.tier === 'strategist' || profile.tier === 'architect') {
      recommendations.push('intelligence-officer');
    } else {
      recommendations.push('senior-analyst');
    }
  } else {
    recommendations.push('custom');
  }

  // Interest-based secondary recommendations
  if (profile.interests.includes('economic') || profile.interests.includes('financial')) {
    recommendations.push('economic-analyst');
  }
  if (profile.interests.includes('crisis') || profile.interests.includes('emergency')) {
    recommendations.push('crisis-monitor');
  }
  if (profile.interests.includes('briefing') || profile.interests.includes('executive')) {
    recommendations.push('executive-briefer');
  }
  if (profile.interests.includes('regional') || profile.interests.includes('area studies')) {
    recommendations.push('regional-specialist');
  }

  // Deduplicate and filter by availability
  const availableIds = new Set(available.map(p => p.id));
  return [...new Set(recommendations)].filter(id => availableIds.has(id));
}

/**
 * Get widget configurations for a preset
 */
export function getPresetWidgets(presetId: PresetId): WidgetConfig[] {
  const preset = DASHBOARD_PRESETS[presetId];
  if (!preset) return [];

  // Deep clone to prevent mutation
  return JSON.parse(JSON.stringify(preset.widgets));
}

/**
 * Merge custom widgets with preset base
 */
export function mergeWithPreset(
  presetId: PresetId,
  customWidgets: WidgetConfig[]
): WidgetConfig[] {
  const baseWidgets = getPresetWidgets(presetId);
  const customIds = new Set(customWidgets.map(w => w.id));

  // Keep preset widgets not overridden, add custom widgets
  const merged = baseWidgets.filter(w => !customIds.has(w.id));
  merged.push(...customWidgets);

  return merged;
}

// ============================================
// Preset Export/Import
// ============================================

export interface ExportedPreset {
  version: number;
  name: string;
  basePreset: PresetId;
  widgets: WidgetConfig[];
  filters: FilterPreset;
  notifications: NotificationPreset;
  shortcuts: string[];
  exportedAt: string;
}

/**
 * Export current dashboard config
 */
export function exportDashboardConfig(
  name: string,
  basePreset: PresetId,
  widgets: WidgetConfig[],
  filters: FilterPreset,
  notifications: NotificationPreset,
  shortcuts: string[]
): ExportedPreset {
  return {
    version: 1,
    name,
    basePreset,
    widgets,
    filters,
    notifications,
    shortcuts,
    exportedAt: new Date().toISOString(),
  };
}

/**
 * Import dashboard config
 */
export function importDashboardConfig(
  json: string
): { success: true; config: ExportedPreset } | { success: false; error: string } {
  try {
    const parsed = JSON.parse(json);

    // Validate structure
    if (!parsed.version || !parsed.widgets || !Array.isArray(parsed.widgets)) {
      return { success: false, error: 'Invalid preset format' };
    }

    if (parsed.version !== 1) {
      return { success: false, error: `Unsupported preset version: ${parsed.version}` };
    }

    // Validate base preset
    if (parsed.basePreset && !DASHBOARD_PRESETS[parsed.basePreset as PresetId]) {
      return { success: false, error: `Unknown base preset: ${parsed.basePreset}` };
    }

    return { success: true, config: parsed as ExportedPreset };
  } catch (e) {
    return { success: false, error: `JSON parse error: ${e}` };
  }
}
