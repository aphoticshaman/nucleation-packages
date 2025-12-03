/**
 * LatticeForge Power User Configuration System
 *
 * Tiered configurability:
 * - Explorer: Simple presets, guided experience
 * - Analyst: Custom filters, saved views, basic automation
 * - Strategist: Full API access, custom dashboards, advanced automation
 * - Architect: Everything + SDK access, custom integrations, white-label
 *
 * The difference between Explorer and Architect should be MASSIVE.
 */

// ============================================
// User Tier Definitions
// ============================================

export type UserTier = 'explorer' | 'analyst' | 'strategist' | 'architect';

export interface TierCapabilities {
  // Data access
  maxEntities: number;
  maxHistoricalDays: number;
  maxProjectionDays: number;
  realTimeUpdates: boolean;
  rawDataExport: boolean;

  // Views & Dashboards
  maxSavedViews: number;
  customDashboards: boolean;
  maxDashboards: number;
  widgetLibrary: boolean;
  customWidgets: boolean;

  // Automation
  maxAlertRules: number;
  webhooks: boolean;
  scheduledReports: boolean;
  customTriggers: boolean;
  macros: boolean;

  // API
  apiAccess: boolean;
  apiCallsPerDay: number;
  streamingApi: boolean;
  bulkOperations: boolean;

  // Advanced
  customModels: boolean;
  sdkAccess: boolean;
  whiteLabel: boolean;
  ssoIntegration: boolean;
  auditLogs: boolean;
  teamManagement: boolean;
  multiTenancy: boolean;
}

export const TIER_CAPABILITIES: Record<UserTier, TierCapabilities> = {
  explorer: {
    maxEntities: 50,
    maxHistoricalDays: 90,
    maxProjectionDays: 30,
    realTimeUpdates: false,
    rawDataExport: false,
    maxSavedViews: 3,
    customDashboards: false,
    maxDashboards: 0,
    widgetLibrary: false,
    customWidgets: false,
    maxAlertRules: 3,
    webhooks: false,
    scheduledReports: false,
    customTriggers: false,
    macros: false,
    apiAccess: false,
    apiCallsPerDay: 0,
    streamingApi: false,
    bulkOperations: false,
    customModels: false,
    sdkAccess: false,
    whiteLabel: false,
    ssoIntegration: false,
    auditLogs: false,
    teamManagement: false,
    multiTenancy: false,
  },
  analyst: {
    maxEntities: 195, // All nations
    maxHistoricalDays: 365,
    maxProjectionDays: 90,
    realTimeUpdates: true,
    rawDataExport: true,
    maxSavedViews: 25,
    customDashboards: true,
    maxDashboards: 5,
    widgetLibrary: true,
    customWidgets: false,
    maxAlertRules: 25,
    webhooks: true,
    scheduledReports: true,
    customTriggers: false,
    macros: false,
    apiAccess: true,
    apiCallsPerDay: 1000,
    streamingApi: false,
    bulkOperations: false,
    customModels: false,
    sdkAccess: false,
    whiteLabel: false,
    ssoIntegration: false,
    auditLogs: true,
    teamManagement: false,
    multiTenancy: false,
  },
  strategist: {
    maxEntities: -1, // Unlimited
    maxHistoricalDays: 3650, // 10 years
    maxProjectionDays: 365,
    realTimeUpdates: true,
    rawDataExport: true,
    maxSavedViews: -1,
    customDashboards: true,
    maxDashboards: 50,
    widgetLibrary: true,
    customWidgets: true,
    maxAlertRules: -1,
    webhooks: true,
    scheduledReports: true,
    customTriggers: true,
    macros: true,
    apiAccess: true,
    apiCallsPerDay: 50000,
    streamingApi: true,
    bulkOperations: true,
    customModels: true,
    sdkAccess: true,
    whiteLabel: false,
    ssoIntegration: true,
    auditLogs: true,
    teamManagement: true,
    multiTenancy: false,
  },
  architect: {
    maxEntities: -1,
    maxHistoricalDays: -1, // Full history
    maxProjectionDays: -1, // Unlimited projections
    realTimeUpdates: true,
    rawDataExport: true,
    maxSavedViews: -1,
    customDashboards: true,
    maxDashboards: -1,
    widgetLibrary: true,
    customWidgets: true,
    maxAlertRules: -1,
    webhooks: true,
    scheduledReports: true,
    customTriggers: true,
    macros: true,
    apiAccess: true,
    apiCallsPerDay: -1, // Unlimited
    streamingApi: true,
    bulkOperations: true,
    customModels: true,
    sdkAccess: true,
    whiteLabel: true,
    ssoIntegration: true,
    auditLogs: true,
    teamManagement: true,
    multiTenancy: true,
  },
};

// ============================================
// Keyboard Shortcuts & Macros
// ============================================

export interface KeyboardShortcut {
  id: string;
  keys: string[]; // e.g., ['ctrl', 'shift', 'f']
  action: string;
  description: string;
  customizable: boolean;
  category: 'navigation' | 'view' | 'filter' | 'action' | 'custom';
}

export interface Macro {
  id: string;
  name: string;
  description: string;
  trigger: MacroTrigger;
  actions: MacroAction[];
  enabled: boolean;
  lastRun?: Date;
  runCount: number;
}

export type MacroTrigger =
  | { type: 'shortcut'; keys: string[] }
  | { type: 'schedule'; cron: string }
  | { type: 'event'; eventType: string; conditions?: Record<string, unknown> }
  | { type: 'alert'; alertRuleId: string }
  | { type: 'manual' };

export type MacroAction =
  | { type: 'setFilter'; filterId: string; value: unknown }
  | { type: 'switchView'; viewId: string }
  | { type: 'exportData'; format: string; destination: string }
  | { type: 'sendAlert'; channel: string; template: string }
  | { type: 'runQuery'; query: string }
  | { type: 'callWebhook'; url: string; payload: Record<string, unknown> }
  | { type: 'executeScript'; scriptId: string; params?: Record<string, unknown> };

export const DEFAULT_SHORTCUTS: KeyboardShortcut[] = [
  // Navigation
  { id: 'nav_dashboard', keys: ['g', 'd'], action: 'navigate:dashboard', description: 'Go to Dashboard', customizable: true, category: 'navigation' },
  { id: 'nav_map', keys: ['g', 'm'], action: 'navigate:map', description: 'Go to Map', customizable: true, category: 'navigation' },
  { id: 'nav_tree', keys: ['g', 't'], action: 'navigate:tree', description: 'Go to 3D Tree', customizable: true, category: 'navigation' },
  { id: 'nav_search', keys: ['ctrl', 'k'], action: 'open:search', description: 'Open Search', customizable: false, category: 'navigation' },

  // View controls
  { id: 'view_zoom_in', keys: ['+'], action: 'view:zoomIn', description: 'Zoom In', customizable: true, category: 'view' },
  { id: 'view_zoom_out', keys: ['-'], action: 'view:zoomOut', description: 'Zoom Out', customizable: true, category: 'view' },
  { id: 'view_reset', keys: ['0'], action: 'view:reset', description: 'Reset View', customizable: true, category: 'view' },
  { id: 'view_fullscreen', keys: ['f'], action: 'view:fullscreen', description: 'Toggle Fullscreen', customizable: true, category: 'view' },

  // Filters
  { id: 'filter_temporal', keys: ['t'], action: 'filter:temporal', description: 'Toggle Temporal Filter', customizable: true, category: 'filter' },
  { id: 'filter_clear', keys: ['escape'], action: 'filter:clearAll', description: 'Clear All Filters', customizable: false, category: 'filter' },
  { id: 'filter_save', keys: ['ctrl', 's'], action: 'filter:save', description: 'Save Current Filters', customizable: true, category: 'filter' },

  // Actions
  { id: 'action_refresh', keys: ['r'], action: 'action:refresh', description: 'Refresh Data', customizable: true, category: 'action' },
  { id: 'action_export', keys: ['ctrl', 'e'], action: 'action:export', description: 'Export Current View', customizable: true, category: 'action' },
  { id: 'action_share', keys: ['ctrl', 'shift', 's'], action: 'action:share', description: 'Share View', customizable: true, category: 'action' },
];

// ============================================
// Custom Dashboard Widgets
// ============================================

export interface DashboardWidget {
  id: string;
  type: WidgetType;
  title: string;
  position: { x: number; y: number; w: number; h: number };
  config: WidgetConfig;
  refreshInterval?: number;
  dataSource?: DataSourceConfig;
}

export type WidgetType =
  | 'stat_card'
  | 'line_chart'
  | 'bar_chart'
  | 'radar_chart'
  | 'heat_map'
  | 'map_mini'
  | 'entity_list'
  | 'alert_feed'
  | 'news_feed'
  | 'conflict_tracker'
  | 'resource_flow'
  | 'relation_graph'
  | 'timeline'
  | 'comparison'
  | 'custom_html'
  | 'embedded_iframe';

export interface WidgetConfig {
  // Common
  showTitle?: boolean;
  backgroundColor?: string;
  borderColor?: string;

  // Chart-specific
  metrics?: string[];
  aggregation?: 'sum' | 'avg' | 'min' | 'max' | 'count';
  groupBy?: string;
  timeRange?: string;

  // Thresholds & alerts
  thresholds?: Array<{ value: number; color: string; label?: string }>;
  alertOnThreshold?: boolean;

  // Custom
  customCss?: string;
  customJs?: string;
  templateString?: string;
}

export interface DataSourceConfig {
  type: 'entities' | 'relations' | 'conflicts' | 'resources' | 'custom_query';
  query?: string;
  filters?: Record<string, unknown>;
  transform?: string; // JS expression for data transformation
}

export interface Dashboard {
  id: string;
  name: string;
  description?: string;
  ownerId: string;
  isPublic: boolean;
  isDefault: boolean;
  layout: 'grid' | 'freeform';
  gridColumns: number;
  widgets: DashboardWidget[];
  refreshInterval?: number;
  theme?: DashboardTheme;
  createdAt: Date;
  updatedAt: Date;
}

export interface DashboardTheme {
  primary: string;
  secondary: string;
  background: string;
  surface: string;
  text: string;
  accent: string;
  fontFamily?: string;
}

// ============================================
// Alert Rules & Automation
// ============================================

export interface AlertRule {
  id: string;
  name: string;
  description?: string;
  enabled: boolean;
  priority: 'low' | 'medium' | 'high' | 'critical';
  conditions: AlertCondition[];
  conditionLogic: 'all' | 'any';
  actions: AlertAction[];
  cooldownMinutes: number;
  lastTriggered?: Date;
  triggerCount: number;
}

export type AlertCondition =
  | { type: 'metric_threshold'; metric: string; operator: 'gt' | 'lt' | 'eq' | 'gte' | 'lte'; value: number }
  | { type: 'entity_change'; entityId: string; field: string; changeType: 'any' | 'increase' | 'decrease' }
  | { type: 'conflict_phase'; conflictId: string; phases: string[] }
  | { type: 'relation_change'; entityA: string; entityB: string; changeType: 'improved' | 'degraded' | 'any' }
  | { type: 'pattern_match'; pattern: string; field: string }
  | { type: 'custom_query'; query: string; expectedResult: unknown };

export type AlertAction =
  | { type: 'notification'; channel: 'email' | 'sms' | 'push' | 'slack' | 'teams' | 'webhook'; recipients: string[] }
  | { type: 'dashboard_highlight'; dashboardId: string; widgetId: string }
  | { type: 'log_event'; severity: string; category: string }
  | { type: 'trigger_macro'; macroId: string }
  | { type: 'api_call'; endpoint: string; method: string; body?: Record<string, unknown> }
  | { type: 'create_report'; templateId: string; recipients: string[] };

// ============================================
// User Preferences & Personalization
// ============================================

export interface UserPreferences {
  // Display
  theme: 'dark' | 'light' | 'system' | 'custom';
  customTheme?: DashboardTheme;
  density: 'compact' | 'comfortable' | 'spacious';
  fontSize: 'small' | 'medium' | 'large';
  animations: boolean;
  reducedMotion: boolean;

  // Default views
  defaultDashboard?: string;
  defaultMapView?: string;
  defaultFilters?: Record<string, unknown>;
  defaultTimeRange: string;
  defaultEntityGroups: string[];

  // Notifications
  emailNotifications: boolean;
  pushNotifications: boolean;
  notificationDigest: 'realtime' | 'hourly' | 'daily' | 'weekly';
  quietHoursStart?: string;
  quietHoursEnd?: string;

  // Data
  preferredUnits: 'metric' | 'imperial';
  dateFormat: string;
  timeFormat: '12h' | '24h';
  timezone: string;
  language: string;
  numberFormat: string;

  // Advanced
  developerMode: boolean;
  showRawData: boolean;
  debugOverlay: boolean;
  experimentalFeatures: boolean;
  keyboardShortcutsEnabled: boolean;
  customShortcuts: KeyboardShortcut[];
}

export const DEFAULT_PREFERENCES: UserPreferences = {
  theme: 'dark',
  density: 'comfortable',
  fontSize: 'medium',
  animations: true,
  reducedMotion: false,
  defaultTimeRange: '30d',
  defaultEntityGroups: ['global'],
  emailNotifications: true,
  pushNotifications: true,
  notificationDigest: 'daily',
  preferredUnits: 'metric',
  dateFormat: 'YYYY-MM-DD',
  timeFormat: '24h',
  timezone: 'UTC',
  language: 'en',
  numberFormat: 'en-US',
  developerMode: false,
  showRawData: false,
  debugOverlay: false,
  experimentalFeatures: false,
  keyboardShortcutsEnabled: true,
  customShortcuts: [],
};

// ============================================
// Saved Views & Templates
// ============================================

export interface SavedView {
  id: string;
  name: string;
  description?: string;
  ownerId: string;
  isPublic: boolean;
  viewType: 'map' | 'tree' | 'dashboard' | 'table' | 'timeline';
  filters: Record<string, unknown>;
  displaySettings: Record<string, unknown>;
  highlightedEntities?: string[];
  annotations?: ViewAnnotation[];
  createdAt: Date;
  updatedAt: Date;
  usageCount: number;
}

export interface ViewAnnotation {
  id: string;
  type: 'marker' | 'region' | 'line' | 'text' | 'arrow';
  position: { x: number; y: number } | { lat: number; lng: number };
  content: string;
  style?: Record<string, string>;
}

export interface ViewTemplate {
  id: string;
  name: string;
  category: 'crisis_response' | 'economic_analysis' | 'security_assessment' | 'resource_tracking' | 'custom';
  description: string;
  thumbnail?: string;
  baseView: Omit<SavedView, 'id' | 'ownerId' | 'createdAt' | 'updatedAt' | 'usageCount'>;
  requiredTier: UserTier;
  isOfficial: boolean;
}

// ============================================
// API Configuration
// ============================================

export interface ApiConfig {
  baseUrl: string;
  apiKey: string;
  rateLimitPerMinute: number;
  rateLimitPerDay: number;
  streamingEnabled: boolean;
  webhookSecret?: string;
  allowedOrigins: string[];
  ipWhitelist?: string[];
}

export interface ApiKeyInfo {
  id: string;
  name: string;
  prefix: string; // First 8 chars shown
  scopes: ApiScope[];
  createdAt: Date;
  expiresAt?: Date;
  lastUsed?: Date;
  usageCount: number;
}

export type ApiScope =
  | 'read:entities'
  | 'read:relations'
  | 'read:conflicts'
  | 'read:resources'
  | 'read:historical'
  | 'read:projections'
  | 'write:views'
  | 'write:alerts'
  | 'write:dashboards'
  | 'admin:team'
  | 'admin:billing';

// ============================================
// Team & Organization
// ============================================

export interface Team {
  id: string;
  name: string;
  organizationId: string;
  members: TeamMember[];
  sharedDashboards: string[];
  sharedViews: string[];
  sharedAlertRules: string[];
  createdAt: Date;
}

export interface TeamMember {
  userId: string;
  role: 'viewer' | 'editor' | 'admin' | 'owner';
  joinedAt: Date;
  invitedBy: string;
}

export interface Organization {
  id: string;
  name: string;
  tier: UserTier;
  domain?: string;
  ssoEnabled: boolean;
  ssoConfig?: SsoConfig;
  brandingConfig?: BrandingConfig;
  teams: string[];
  seats: { used: number; total: number };
  billingEmail: string;
  createdAt: Date;
}

export interface SsoConfig {
  provider: 'saml' | 'oidc' | 'azure_ad' | 'okta' | 'google';
  entityId?: string;
  ssoUrl?: string;
  certificate?: string;
  clientId?: string;
  clientSecret?: string;
}

export interface BrandingConfig {
  logoUrl?: string;
  faviconUrl?: string;
  primaryColor: string;
  accentColor: string;
  customCss?: string;
  emailFooter?: string;
  customDomain?: string;
}

// ============================================
// Utility Functions
// ============================================

export function canAccess(tier: UserTier, feature: keyof TierCapabilities): boolean {
  const caps = TIER_CAPABILITIES[tier];
  const value = caps[feature];

  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return value !== 0;
  return false;
}

export function getLimit(tier: UserTier, feature: keyof TierCapabilities): number {
  const value = TIER_CAPABILITIES[tier][feature];
  if (typeof value === 'number') return value;
  return value ? 1 : 0;
}

export function isUnlimited(tier: UserTier, feature: keyof TierCapabilities): boolean {
  const value = TIER_CAPABILITIES[tier][feature];
  return value === -1;
}

export function formatShortcut(keys: string[]): string {
  return keys
    .map((k) => {
      switch (k.toLowerCase()) {
        case 'ctrl': return '⌃';
        case 'cmd': return '⌘';
        case 'alt': return '⌥';
        case 'shift': return '⇧';
        case 'enter': return '↵';
        case 'escape': return 'Esc';
        case 'space': return 'Space';
        default: return k.toUpperCase();
      }
    })
    .join(' + ');
}

export function matchShortcut(event: KeyboardEvent, keys: string[]): boolean {
  const pressed = new Set<string>();

  if (event.ctrlKey) pressed.add('ctrl');
  if (event.metaKey) pressed.add('cmd');
  if (event.altKey) pressed.add('alt');
  if (event.shiftKey) pressed.add('shift');
  pressed.add(event.key.toLowerCase());

  if (pressed.size !== keys.length) return false;

  return keys.every((k) => pressed.has(k.toLowerCase()));
}

export function tierCompare(a: UserTier, b: UserTier): number {
  const order: UserTier[] = ['explorer', 'analyst', 'strategist', 'architect'];
  return order.indexOf(a) - order.indexOf(b);
}

export function getTierUpgrade(current: UserTier): UserTier | null {
  const order: UserTier[] = ['explorer', 'analyst', 'strategist', 'architect'];
  const idx = order.indexOf(current);
  return idx < order.length - 1 ? order[idx + 1] : null;
}

// ============================================
// Auth Tier Mapping
// ============================================

// Maps the auth system tiers to power user tiers
// Auth: 'free' | 'starter' | 'pro' | 'enterprise_tier'
// Power: 'explorer' | 'analyst' | 'strategist' | 'architect'
export type AuthTier = 'free' | 'starter' | 'pro' | 'enterprise_tier';

const AUTH_TO_POWER_TIER: Record<AuthTier, UserTier> = {
  free: 'explorer',
  starter: 'analyst',
  pro: 'strategist',
  enterprise_tier: 'architect',
};

const POWER_TO_AUTH_TIER: Record<UserTier, AuthTier> = {
  explorer: 'free',
  analyst: 'starter',
  strategist: 'pro',
  architect: 'enterprise_tier',
};

export function authTierToPowerTier(authTier: AuthTier): UserTier {
  return AUTH_TO_POWER_TIER[authTier] || 'explorer';
}

export function powerTierToAuthTier(powerTier: UserTier): AuthTier {
  return POWER_TO_AUTH_TIER[powerTier] || 'free';
}
