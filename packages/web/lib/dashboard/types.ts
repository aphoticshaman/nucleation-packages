// Dashboard Builder Type System

export type WidgetType =
  | 'stat_card'
  | 'chart_line'
  | 'chart_bar'
  | 'chart_pie'
  | 'table'
  | 'map'
  | 'text'
  | 'image'
  | 'alert_feed'
  | 'metric_gauge'
  | 'signal_ticker'
  | 'briefing_card'
  | 'countdown'
  | 'embed';

export interface WidgetPosition {
  x: number; // Grid column (0-11 for 12-col grid)
  y: number; // Grid row
  w: number; // Width in grid units (1-12)
  h: number; // Height in grid units
}

export interface WidgetBase {
  id: string;
  type: WidgetType;
  position: WidgetPosition;
  title?: string;
  dataSource?: DataSourceConfig;
  style?: WidgetStyle;
}

export interface WidgetStyle {
  backgroundColor?: string;
  borderColor?: string;
  borderRadius?: number;
  padding?: number;
  textColor?: string;
  fontSize?: number;
  shadow?: 'none' | 'sm' | 'md' | 'lg';
}

export interface DataSourceConfig {
  type: 'static' | 'api' | 'supabase' | 'realtime';
  endpoint?: string;
  table?: string;
  query?: string;
  refreshInterval?: number; // seconds
  transform?: string; // JSONata expression
}

// Specific widget configs
export interface StatCardWidget extends WidgetBase {
  type: 'stat_card';
  config: {
    value: string | number;
    label: string;
    change?: number;
    changeLabel?: string;
    icon?: string;
    trend?: 'up' | 'down' | 'neutral';
  };
}

export interface ChartWidget extends WidgetBase {
  type: 'chart_line' | 'chart_bar' | 'chart_pie';
  config: {
    data: unknown[];
    xKey?: string;
    yKey?: string;
    colors?: string[];
    showLegend?: boolean;
    showGrid?: boolean;
  };
}

export interface TableWidget extends WidgetBase {
  type: 'table';
  config: {
    columns: { key: string; label: string; width?: number }[];
    data: unknown[];
    pageSize?: number;
    sortable?: boolean;
    searchable?: boolean;
  };
}

export interface TextWidget extends WidgetBase {
  type: 'text';
  config: {
    content: string;
    format: 'plain' | 'markdown' | 'html';
    alignment?: 'left' | 'center' | 'right';
  };
}

export interface AlertFeedWidget extends WidgetBase {
  type: 'alert_feed';
  config: {
    maxItems: number;
    showTimestamp?: boolean;
    severityFilter?: ('critical' | 'warning' | 'info')[];
  };
}

export interface MetricGaugeWidget extends WidgetBase {
  type: 'metric_gauge';
  config: {
    value: number;
    min: number;
    max: number;
    thresholds?: { value: number; color: string }[];
    unit?: string;
  };
}

export interface EmbedWidget extends WidgetBase {
  type: 'embed';
  config: {
    url: string;
    allowFullscreen?: boolean;
  };
}

export type Widget =
  | StatCardWidget
  | ChartWidget
  | TableWidget
  | TextWidget
  | AlertFeedWidget
  | MetricGaugeWidget
  | EmbedWidget
  | WidgetBase;

// Dashboard definition
export interface Dashboard {
  id: string;
  name: string;
  description?: string;
  slug: string;
  widgets: Widget[];
  layout: {
    columns: 12;
    rowHeight: number;
    gap: number;
  };
  permissions: {
    viewRoles: string[];
    editRoles: string[];
    viewTiers: string[];
  };
  createdAt: string;
  updatedAt: string;
  createdBy: string;
  isPublished: boolean;
  isDefault?: boolean;
}

// Widget catalog for the palette
export interface WidgetCatalogItem {
  type: WidgetType;
  name: string;
  description: string;
  icon: string;
  category: 'data' | 'visualization' | 'content' | 'intelligence';
  defaultSize: { w: number; h: number };
  defaultConfig: Record<string, unknown>;
}

export const WIDGET_CATALOG: WidgetCatalogItem[] = [
  {
    type: 'stat_card',
    name: 'Stat Card',
    description: 'Display a key metric with optional trend',
    icon: '📊',
    category: 'data',
    defaultSize: { w: 3, h: 2 },
    defaultConfig: { value: '--', label: 'Metric', trend: 'neutral' },
  },
  {
    type: 'chart_line',
    name: 'Line Chart',
    description: 'Time series or trend visualization',
    icon: '📈',
    category: 'visualization',
    defaultSize: { w: 6, h: 4 },
    defaultConfig: { data: [], showLegend: true, showGrid: true },
  },
  {
    type: 'chart_bar',
    name: 'Bar Chart',
    description: 'Compare values across categories',
    icon: '📊',
    category: 'visualization',
    defaultSize: { w: 6, h: 4 },
    defaultConfig: { data: [], showLegend: true },
  },
  {
    type: 'chart_pie',
    name: 'Pie Chart',
    description: 'Show proportions of a whole',
    icon: '🥧',
    category: 'visualization',
    defaultSize: { w: 4, h: 4 },
    defaultConfig: { data: [], showLegend: true },
  },
  {
    type: 'table',
    name: 'Data Table',
    description: 'Tabular data with sorting and search',
    icon: '📋',
    category: 'data',
    defaultSize: { w: 6, h: 4 },
    defaultConfig: { columns: [], data: [], sortable: true, searchable: true },
  },
  {
    type: 'map',
    name: 'Map View',
    description: 'Geographic visualization',
    icon: '🗺️',
    category: 'visualization',
    defaultSize: { w: 6, h: 5 },
    defaultConfig: {},
  },
  {
    type: 'text',
    name: 'Text Block',
    description: 'Rich text or markdown content',
    icon: '📝',
    category: 'content',
    defaultSize: { w: 4, h: 2 },
    defaultConfig: { content: 'Enter text...', format: 'markdown' },
  },
  {
    type: 'image',
    name: 'Image',
    description: 'Display an image or logo',
    icon: '🖼️',
    category: 'content',
    defaultSize: { w: 3, h: 3 },
    defaultConfig: {},
  },
  {
    type: 'alert_feed',
    name: 'Alert Feed',
    description: 'Live stream of alerts and notifications',
    icon: '🚨',
    category: 'intelligence',
    defaultSize: { w: 4, h: 4 },
    defaultConfig: { maxItems: 10, showTimestamp: true },
  },
  {
    type: 'metric_gauge',
    name: 'Metric Gauge',
    description: 'Circular gauge for bounded metrics',
    icon: '⏱️',
    category: 'data',
    defaultSize: { w: 2, h: 2 },
    defaultConfig: { value: 0, min: 0, max: 100 },
  },
  {
    type: 'signal_ticker',
    name: 'Signal Ticker',
    description: 'Real-time market signal feed',
    icon: '📡',
    category: 'intelligence',
    defaultSize: { w: 12, h: 1 },
    defaultConfig: {},
  },
  {
    type: 'briefing_card',
    name: 'Briefing Card',
    description: 'AI-generated intelligence brief',
    icon: '🎖️',
    category: 'intelligence',
    defaultSize: { w: 4, h: 3 },
    defaultConfig: {},
  },
  {
    type: 'countdown',
    name: 'Countdown',
    description: 'Timer to a specific date/event',
    icon: '⏳',
    category: 'content',
    defaultSize: { w: 3, h: 2 },
    defaultConfig: {},
  },
  {
    type: 'embed',
    name: 'Embed',
    description: 'Embed external content via iframe',
    icon: '🔗',
    category: 'content',
    defaultSize: { w: 6, h: 4 },
    defaultConfig: { url: '', allowFullscreen: true },
  },
];
