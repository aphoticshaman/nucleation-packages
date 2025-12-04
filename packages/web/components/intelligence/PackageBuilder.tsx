'use client';

import { useState, useCallback, useMemo, useRef } from 'react';
import {
  FileText,
  Presentation,
  Table,
  Download,
  Save,
  FolderOpen,
  Plus,
  Trash2,
  GripVertical,
  Eye,
  Settings,
  Layout,
  Users,
  Shield,
  Building,
  Heart,
  ChevronDown,
  ChevronRight,
  Check,
  X,
  Loader2,
} from 'lucide-react';

/**
 * Deliverable Package Builder
 *
 * Palantir-inspired mission planning → deliverable export system.
 * User Story (from Lee, 10th SFG EOD):
 * "Enable end-to-end mission pipeline target package building where the user
 * decides pre-packaging what's useful, easy to trim/add/adjust/re-configure
 * and export for various audiences: DoD briefings, NGO crisis response, board meetings."
 *
 * Key Features:
 * 1. Component selection with drag/drop arrangement
 * 2. Audience presets (DoD, NGO, Board, Technical, Field)
 * 3. Visual layout editing
 * 4. Multiple export formats (PDF, PPTX, DOCX, JSON, CSV)
 * 5. Template save/load
 */

// === TYPES ===

type AudiencePreset = 'dod' | 'ngo' | 'board' | 'technical' | 'field' | 'custom';
type ExportFormat = 'pdf' | 'pptx' | 'docx' | 'json' | 'csv' | 'html';

interface PackageComponent {
  id: string;
  type: ComponentType;
  label: string;
  description: string;
  icon: string;
  enabled: boolean;
  order: number;
  config: ComponentConfig;
}

interface ComponentConfig {
  showConfidence?: boolean;
  showSources?: boolean;
  showTimestamps?: boolean;
  detailLevel?: 'summary' | 'standard' | 'detailed';
  // Classification stubbed - no authority to classify. All content is OSINT.
  classification?: 'UNCLASSIFIED';
  customNotes?: string;
}

type ComponentType =
  | 'executive_summary'
  | 'bluf'
  | 'threat_matrix'
  | 'key_developments'
  | 'risk_gauge'
  | 'map_view'
  | 'network_graph'
  | 'causal_chain'
  | 'timeline'
  | 'data_table'
  | 'recommendations'
  | 'sources'
  | 'appendix'
  | 'custom_section';

interface PackageTemplate {
  id: string;
  name: string;
  description: string;
  audience: AudiencePreset;
  components: PackageComponent[];
  createdAt: string;
  updatedAt: string;
}

interface PackageBuilderProps {
  /** Current intel briefing data */
  briefingData?: unknown;
  /** Available datasets for inclusion */
  availableDatasets?: string[];
  /** Callback when package is exported */
  onExport?: (format: ExportFormat, components: PackageComponent[]) => void;
  /** Callback when template is saved */
  onSaveTemplate?: (template: PackageTemplate) => void;
  /** Available templates to load */
  templates?: PackageTemplate[];
}

// === COMPONENT LIBRARY ===

const COMPONENT_LIBRARY: Omit<PackageComponent, 'order' | 'enabled'>[] = [
  {
    id: 'exec_summary',
    type: 'executive_summary',
    label: 'Executive Summary',
    description: 'POTUS-style overview with BLUF, risk assessment, and key findings',
    icon: '📋',
    config: { detailLevel: 'standard', showConfidence: true },
  },
  {
    id: 'bluf',
    type: 'bluf',
    label: 'BLUF',
    description: 'Bottom Line Up Front - the single most important takeaway',
    icon: '🎯',
    config: { detailLevel: 'summary' },
  },
  {
    id: 'threat_matrix',
    type: 'threat_matrix',
    label: 'Threat Assessment',
    description: 'Threat matrix with severity, probability, and timeframes',
    icon: '⚠️',
    config: { showConfidence: true, detailLevel: 'detailed' },
  },
  {
    id: 'key_devs',
    type: 'key_developments',
    label: 'Key Developments',
    description: 'Priority-ordered developments with urgency indicators',
    icon: '📰',
    config: { detailLevel: 'standard' },
  },
  {
    id: 'risk_gauge',
    type: 'risk_gauge',
    label: 'Risk Gauge',
    description: 'Visual risk indicator with confidence intervals',
    icon: '📊',
    config: { showConfidence: true },
  },
  {
    id: 'map',
    type: 'map_view',
    label: 'Geographic View',
    description: 'Map visualization of relevant areas and hotspots',
    icon: '🗺️',
    config: { detailLevel: 'standard' },
  },
  {
    id: 'network',
    type: 'network_graph',
    label: 'Network Analysis',
    description: 'Entity relationships and influence mapping',
    icon: '🕸️',
    config: { detailLevel: 'detailed' },
  },
  {
    id: 'causal',
    type: 'causal_chain',
    label: 'Causal Analysis',
    description: 'Event causality chains and contributing factors',
    icon: '🔗',
    config: { showConfidence: true, detailLevel: 'detailed' },
  },
  {
    id: 'timeline',
    type: 'timeline',
    label: 'Timeline',
    description: 'Chronological event sequence with projections',
    icon: '📅',
    config: { showTimestamps: true },
  },
  {
    id: 'data_table',
    type: 'data_table',
    label: 'Data Tables',
    description: 'Structured data with filtering and sorting',
    icon: '📑',
    config: { detailLevel: 'detailed' },
  },
  {
    id: 'recommendations',
    type: 'recommendations',
    label: 'Recommendations',
    description: 'Actionable recommendations with priority levels',
    icon: '💡',
    config: { detailLevel: 'standard' },
  },
  {
    id: 'sources',
    type: 'sources',
    label: 'Sources & Methods',
    description: 'Source attribution and reliability ratings',
    icon: '📚',
    config: { showSources: true },
  },
  {
    id: 'appendix',
    type: 'appendix',
    label: 'Appendix',
    description: 'Supporting documents and raw data',
    icon: '📎',
    config: { detailLevel: 'detailed' },
  },
];

// === AUDIENCE PRESETS ===

const AUDIENCE_PRESETS: Record<AudiencePreset, {
  name: string;
  description: string;
  icon: React.ReactNode;
  defaultComponents: string[];
  defaultConfig: Partial<ComponentConfig>;
}> = {
  dod: {
    name: 'DoD / IC Briefing',
    description: 'ODNI tradecraft standards format (OSINT only)',
    icon: <Shield className="w-4 h-4" />,
    defaultComponents: ['exec_summary', 'bluf', 'threat_matrix', 'key_devs', 'map', 'causal', 'recommendations', 'sources'],
    defaultConfig: { classification: 'UNCLASSIFIED', showConfidence: true, showSources: true, detailLevel: 'detailed' },
  },
  ngo: {
    name: 'NGO / Humanitarian',
    description: 'Crisis response focus with actionable field guidance',
    icon: <Heart className="w-4 h-4" />,
    defaultComponents: ['bluf', 'key_devs', 'map', 'timeline', 'recommendations'],
    defaultConfig: { classification: 'UNCLASSIFIED', showConfidence: false, detailLevel: 'standard' },
  },
  board: {
    name: 'Board / Executive',
    description: 'High-level strategic overview for decision makers',
    icon: <Building className="w-4 h-4" />,
    defaultComponents: ['exec_summary', 'bluf', 'risk_gauge', 'recommendations'],
    defaultConfig: { classification: 'UNCLASSIFIED', showConfidence: false, detailLevel: 'summary' },
  },
  technical: {
    name: 'Technical / Analyst',
    description: 'Full data access with methodology transparency',
    icon: <Settings className="w-4 h-4" />,
    defaultComponents: ['exec_summary', 'threat_matrix', 'key_devs', 'network', 'causal', 'data_table', 'sources', 'appendix'],
    defaultConfig: { showConfidence: true, showSources: true, showTimestamps: true, detailLevel: 'detailed' },
  },
  field: {
    name: 'Field Operator',
    description: 'Mission-critical info only, optimized for mobile',
    icon: <Users className="w-4 h-4" />,
    defaultComponents: ['bluf', 'threat_matrix', 'map', 'recommendations'],
    defaultConfig: { classification: 'UNCLASSIFIED', showConfidence: false, detailLevel: 'summary' },
  },
  custom: {
    name: 'Custom',
    description: 'Build your own package configuration',
    icon: <Layout className="w-4 h-4" />,
    defaultComponents: [],
    defaultConfig: { detailLevel: 'standard' },
  },
};

// === EXPORT FORMATS ===

const EXPORT_FORMATS: { format: ExportFormat; label: string; icon: React.ReactNode; description: string }[] = [
  { format: 'pdf', label: 'PDF', icon: <FileText className="w-4 h-4" />, description: 'Print-ready document' },
  { format: 'pptx', label: 'PowerPoint', icon: <Presentation className="w-4 h-4" />, description: 'Presentation slides' },
  { format: 'docx', label: 'Word', icon: <FileText className="w-4 h-4" />, description: 'Editable document' },
  { format: 'json', label: 'JSON', icon: <Table className="w-4 h-4" />, description: 'Structured data' },
  { format: 'csv', label: 'CSV', icon: <Table className="w-4 h-4" />, description: 'Spreadsheet data' },
  { format: 'html', label: 'HTML', icon: <FileText className="w-4 h-4" />, description: 'Web-ready report' },
];

// === EXPORT UTILITIES ===

function downloadFile(content: string | Blob, filename: string, mimeType: string) {
  const blob = content instanceof Blob ? content : new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function generatePackageContent(components: PackageComponent[], audience: AudiencePreset): {
  title: string;
  subtitle: string;
  generatedAt: string;
  sections: { id: string; title: string; icon: string; content: string; config: ComponentConfig }[];
} {
  const presetInfo = AUDIENCE_PRESETS[audience];
  const now = new Date();

  return {
    title: 'LatticeForge Intelligence Package',
    subtitle: `${presetInfo.name} - Generated ${now.toLocaleDateString()}`,
    generatedAt: now.toISOString(),
    sections: components.sort((a, b) => a.order - b.order).map(c => ({
      id: c.id,
      title: c.label,
      icon: c.icon,
      content: generateSectionContent(c),
      config: c.config,
    })),
  };
}

function generateSectionContent(component: PackageComponent): string {
  // Generate placeholder content based on component type
  const contentMap: Record<ComponentType, string> = {
    executive_summary: `Executive Summary\n\nThis intelligence package provides a comprehensive analysis of current geopolitical and market conditions. Key findings indicate elevated risk levels in monitored regions with potential for cascading effects across interconnected systems.\n\nConfidence Level: ${component.config.showConfidence ? 'HIGH (85%)' : 'See methodology'}`,
    bluf: `BOTTOM LINE UP FRONT\n\nElevated monitoring recommended. Multiple indicators suggest potential regime shift within 30-90 day window. Primary drivers: economic pressure, political instability, and external actor involvement.`,
    threat_matrix: `Threat Assessment Matrix\n\n| Threat | Severity | Probability | Timeframe |\n|--------|----------|-------------|----------|\n| Economic Instability | HIGH | 75% | 30 days |\n| Political Disruption | MEDIUM | 60% | 60 days |\n| Cascade Event | MEDIUM | 45% | 90 days |`,
    key_developments: `Key Developments (Last 7 Days)\n\n1. [PRIORITY] Economic indicators show stress signals\n2. [WATCH] Political rhetoric escalating in target region\n3. [INFO] New sanctions package announced\n4. [INFO] Military repositioning observed`,
    risk_gauge: `Overall Risk Assessment: ELEVATED (7.2/10)\n\nTrend: Increasing (+0.8 from previous period)\nConfidence Interval: 6.5 - 7.9`,
    map_view: `Geographic Analysis\n\nPrimary Areas of Interest:\n- Region Alpha: High activity\n- Region Beta: Moderate monitoring\n- Region Gamma: Baseline normal`,
    network_graph: `Network Analysis\n\nKey Entities Identified: 12\nRelationship Clusters: 4\nInfluence Score (Primary Actor): 0.87`,
    causal_chain: `Causal Chain Analysis\n\nRoot Cause → Economic Pressure\n  └─→ Currency Devaluation\n      └─→ Social Unrest\n          └─→ Political Response\n              └─→ Potential Escalation`,
    timeline: `Event Timeline\n\nT-90: Initial indicators detected\nT-60: Trend confirmation\nT-30: Acceleration phase\nT-0: Current state\nT+30: Projected critical window`,
    data_table: `Supporting Data\n\n| Metric | Value | Change | Status |\n|--------|-------|--------|--------|\n| Index A | 127.3 | +5.2% | Warning |\n| Index B | 89.1 | -2.1% | Normal |\n| Index C | 156.8 | +12.4% | Critical |`,
    recommendations: `Recommendations\n\n1. [HIGH PRIORITY] Increase monitoring frequency\n2. [MEDIUM] Prepare contingency protocols\n3. [STANDARD] Update stakeholder briefings\n4. [ONGOING] Continue baseline collection`,
    sources: `Sources and Methods\n\nOSINT Sources: 47\nReliability Rating: B (Generally Reliable)\nCollection Period: 2024-11-01 to present\n\nNote: All information derived from open sources. No classified materials.`,
    appendix: `Appendix\n\nA. Methodology Notes\nB. Data Collection Parameters\nC. Historical Comparisons\nD. Glossary of Terms`,
    custom_section: component.config.customNotes || 'Custom section content',
  };

  return contentMap[component.type] || 'Section content';
}

function exportAsJSON(components: PackageComponent[], audience: AudiencePreset): void {
  const content = generatePackageContent(components, audience);
  const json = JSON.stringify(content, null, 2);
  downloadFile(json, `latticeforge-package-${Date.now()}.json`, 'application/json');
}

function exportAsCSV(components: PackageComponent[], audience: AudiencePreset): void {
  const content = generatePackageContent(components, audience);
  const headers = ['Section', 'Title', 'Content', 'Detail Level', 'Show Confidence', 'Show Sources'];
  const rows = content.sections.map(s => [
    s.id,
    s.title,
    s.content.replace(/\n/g, ' ').replace(/,/g, ';'),
    s.config.detailLevel || 'standard',
    s.config.showConfidence ? 'Yes' : 'No',
    s.config.showSources ? 'Yes' : 'No',
  ]);

  const csv = [
    headers.join(','),
    ...rows.map(r => r.map(cell => `"${cell}"`).join(',')),
  ].join('\n');

  downloadFile(csv, `latticeforge-package-${Date.now()}.csv`, 'text/csv');
}

function exportAsHTML(components: PackageComponent[], audience: AudiencePreset): void {
  const content = generatePackageContent(components, audience);
  const presetInfo = AUDIENCE_PRESETS[audience];

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${content.title}</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; line-height: 1.6; padding: 2rem; }
    .container { max-width: 900px; margin: 0 auto; }
    header { border-bottom: 1px solid #334155; padding-bottom: 1rem; margin-bottom: 2rem; }
    h1 { font-size: 1.75rem; color: #f8fafc; margin-bottom: 0.5rem; }
    .subtitle { color: #94a3b8; font-size: 0.875rem; }
    .meta { display: flex; gap: 1rem; margin-top: 0.5rem; font-size: 0.75rem; color: #64748b; }
    section { background: #1e293b; border: 1px solid #334155; border-radius: 0.5rem; padding: 1.5rem; margin-bottom: 1rem; }
    section h2 { font-size: 1.125rem; color: #f8fafc; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
    section pre { white-space: pre-wrap; font-family: inherit; color: #cbd5e1; font-size: 0.875rem; }
    .badge { display: inline-block; padding: 0.125rem 0.5rem; background: #334155; border-radius: 0.25rem; font-size: 0.625rem; color: #94a3b8; margin-left: auto; }
    footer { text-align: center; color: #475569; font-size: 0.75rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #334155; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>${content.title}</h1>
      <p class="subtitle">${presetInfo.name}</p>
      <div class="meta">
        <span>Generated: ${new Date(content.generatedAt).toLocaleString()}</span>
        <span>Sections: ${content.sections.length}</span>
        <span>Classification: OSINT / UNCLASSIFIED</span>
      </div>
    </header>
    ${content.sections.map(s => `
    <section>
      <h2><span>${s.icon}</span> ${s.title} <span class="badge">${s.config.detailLevel || 'standard'}</span></h2>
      <pre>${s.content}</pre>
    </section>`).join('')}
    <footer>
      LatticeForge Intelligence Platform | ${new Date().getFullYear()} | OSINT Only
    </footer>
  </div>
</body>
</html>`;

  downloadFile(html, `latticeforge-package-${Date.now()}.html`, 'text/html');
}

function exportAsPDF(components: PackageComponent[], audience: AudiencePreset): void {
  // Generate HTML and use browser print to PDF
  const content = generatePackageContent(components, audience);
  const presetInfo = AUDIENCE_PRESETS[audience];

  const printWindow = window.open('', '_blank');
  if (!printWindow) {
    alert('Please allow popups to export PDF');
    return;
  }

  printWindow.document.write(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>${content.title}</title>
  <style>
    @media print {
      @page { margin: 1in; size: letter; }
      body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: 'Times New Roman', serif; color: #1a1a1a; line-height: 1.5; padding: 0; }
    .container { max-width: 100%; }
    header { border-bottom: 2px solid #1a1a1a; padding-bottom: 1rem; margin-bottom: 1.5rem; }
    h1 { font-size: 1.5rem; font-weight: bold; margin-bottom: 0.25rem; }
    .subtitle { font-size: 1rem; color: #444; }
    .meta { display: flex; gap: 2rem; margin-top: 0.5rem; font-size: 0.75rem; color: #666; }
    .classification { text-align: center; font-weight: bold; padding: 0.5rem; background: #e5e5e5; margin-bottom: 1rem; font-size: 0.875rem; }
    section { page-break-inside: avoid; margin-bottom: 1.5rem; }
    section h2 { font-size: 1.125rem; font-weight: bold; border-bottom: 1px solid #ccc; padding-bottom: 0.25rem; margin-bottom: 0.75rem; }
    section pre { white-space: pre-wrap; font-family: 'Courier New', monospace; font-size: 0.8125rem; color: #333; }
    .badge { display: inline-block; padding: 0.125rem 0.5rem; background: #e5e5e5; border-radius: 0.25rem; font-size: 0.625rem; margin-left: 0.5rem; }
    footer { text-align: center; color: #666; font-size: 0.75rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #ccc; }
  </style>
</head>
<body>
  <div class="container">
    <div class="classification">UNCLASSIFIED // OSINT</div>
    <header>
      <h1>${content.title}</h1>
      <p class="subtitle">${presetInfo.name}</p>
      <div class="meta">
        <span>Generated: ${new Date(content.generatedAt).toLocaleString()}</span>
        <span>Sections: ${content.sections.length}</span>
      </div>
    </header>
    ${content.sections.map(s => `
    <section>
      <h2>${s.icon} ${s.title} <span class="badge">${s.config.detailLevel || 'standard'}</span></h2>
      <pre>${s.content}</pre>
    </section>`).join('')}
    <footer>
      LatticeForge Intelligence Platform | ${new Date().getFullYear()} | OSINT Only - No Classification Authority
    </footer>
    <div class="classification" style="margin-top: 1rem;">UNCLASSIFIED // OSINT</div>
  </div>
  <script>
    window.onload = function() {
      setTimeout(function() {
        window.print();
        window.onafterprint = function() { window.close(); };
      }, 250);
    };
  </script>
</body>
</html>`);
  printWindow.document.close();
}

function exportAsDOCX(components: PackageComponent[], audience: AudiencePreset): void {
  // Generate a simple .doc (HTML-based) that Word can open
  const content = generatePackageContent(components, audience);
  const presetInfo = AUDIENCE_PRESETS[audience];

  const doc = `<!DOCTYPE html>
<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word">
<head>
  <meta charset="UTF-8">
  <title>${content.title}</title>
  <style>
    body { font-family: Calibri, sans-serif; font-size: 11pt; }
    h1 { font-size: 18pt; color: #1a1a1a; }
    h2 { font-size: 14pt; color: #333; border-bottom: 1pt solid #ccc; padding-bottom: 4pt; margin-top: 16pt; }
    pre { font-family: Consolas, monospace; font-size: 10pt; white-space: pre-wrap; background: #f5f5f5; padding: 8pt; }
    .meta { color: #666; font-size: 9pt; }
    .header { border-bottom: 2pt solid #1a1a1a; padding-bottom: 8pt; margin-bottom: 16pt; }
  </style>
</head>
<body>
  <div class="header">
    <h1>${content.title}</h1>
    <p><strong>${presetInfo.name}</strong></p>
    <p class="meta">Generated: ${new Date(content.generatedAt).toLocaleString()} | Classification: OSINT / UNCLASSIFIED</p>
  </div>
  ${content.sections.map(s => `
  <h2>${s.icon} ${s.title}</h2>
  <pre>${s.content}</pre>`).join('')}
  <p style="margin-top: 24pt; text-align: center; color: #666; font-size: 9pt;">
    LatticeForge Intelligence Platform | OSINT Only
  </p>
</body>
</html>`;

  const blob = new Blob([doc], { type: 'application/msword' });
  downloadFile(blob, `latticeforge-package-${Date.now()}.doc`, 'application/msword');
}

function exportAsPPTX(components: PackageComponent[], audience: AudiencePreset): void {
  // Generate HTML-based presentation that can be opened and converted
  const content = generatePackageContent(components, audience);
  const presetInfo = AUDIENCE_PRESETS[audience];

  // Create an HTML presentation format
  const slides = content.sections.map((s, i) => `
    <div class="slide">
      <div class="slide-number">${i + 1}</div>
      <h2>${s.icon} ${s.title}</h2>
      <pre>${s.content}</pre>
    </div>
  `);

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>${content.title} - Presentation</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; }
    .slide { width: 100%; min-height: 100vh; padding: 3rem; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); color: #e2e8f0; page-break-after: always; position: relative; }
    .slide-number { position: absolute; bottom: 2rem; right: 2rem; font-size: 0.875rem; color: #64748b; }
    .title-slide { display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; }
    .title-slide h1 { font-size: 3rem; color: #f8fafc; margin-bottom: 1rem; }
    .title-slide .subtitle { font-size: 1.5rem; color: #94a3b8; }
    .title-slide .meta { margin-top: 2rem; font-size: 1rem; color: #64748b; }
    h2 { font-size: 2rem; color: #f8fafc; margin-bottom: 2rem; border-bottom: 2px solid #3b82f6; padding-bottom: 0.5rem; }
    pre { font-family: inherit; white-space: pre-wrap; font-size: 1.125rem; line-height: 1.8; color: #cbd5e1; }
    @media print {
      .slide { height: 100vh; overflow: hidden; }
    }
  </style>
</head>
<body>
  <div class="slide title-slide">
    <h1>${content.title}</h1>
    <p class="subtitle">${presetInfo.name}</p>
    <p class="meta">Generated ${new Date(content.generatedAt).toLocaleDateString()}</p>
    <div class="slide-number">Title</div>
  </div>
  ${slides.join('')}
  <div class="slide title-slide">
    <h1>Questions?</h1>
    <p class="subtitle">LatticeForge Intelligence Platform</p>
    <p class="meta">OSINT Only - No Classification Authority</p>
    <div class="slide-number">End</div>
  </div>
</body>
</html>`;

  downloadFile(html, `latticeforge-presentation-${Date.now()}.html`, 'text/html');

  // Also alert user about conversion
  setTimeout(() => {
    alert('Presentation downloaded as HTML. For PowerPoint format:\n\n1. Open the HTML file in your browser\n2. Use Print → Save as PDF\n3. Or import into Google Slides / PowerPoint');
  }, 500);
}

function performExport(format: ExportFormat, components: PackageComponent[], audience: AudiencePreset): void {
  switch (format) {
    case 'json':
      exportAsJSON(components, audience);
      break;
    case 'csv':
      exportAsCSV(components, audience);
      break;
    case 'html':
      exportAsHTML(components, audience);
      break;
    case 'pdf':
      exportAsPDF(components, audience);
      break;
    case 'docx':
      exportAsDOCX(components, audience);
      break;
    case 'pptx':
      exportAsPPTX(components, audience);
      break;
  }
}

// === MAIN COMPONENT ===

export function PackageBuilder({
  briefingData,
  availableDatasets,
  onExport,
  onSaveTemplate,
  templates = [],
}: PackageBuilderProps) {
  // State
  const [selectedAudience, setSelectedAudience] = useState<AudiencePreset>('custom');
  const [components, setComponents] = useState<PackageComponent[]>([]);
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
  const [previewMode, setPreviewMode] = useState(false);
  const [templateName, setTemplateName] = useState('');
  const [showTemplateDialog, setShowTemplateDialog] = useState(false);
  const [showExportDialog, setShowExportDialog] = useState(false);
  const [draggedItem, setDraggedItem] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);

  // Apply audience preset
  const applyPreset = useCallback((preset: AudiencePreset) => {
    const presetConfig = AUDIENCE_PRESETS[preset];
    const newComponents: PackageComponent[] = COMPONENT_LIBRARY
      .filter(c => presetConfig.defaultComponents.includes(c.id))
      .map((c, idx) => ({
        ...c,
        enabled: true,
        order: idx,
        config: { ...c.config, ...presetConfig.defaultConfig },
      }));

    setComponents(newComponents);
    setSelectedAudience(preset);
  }, []);

  // Toggle component
  const toggleComponent = useCallback((componentId: string) => {
    setComponents(prev => {
      const existing = prev.find(c => c.id === componentId);
      if (existing) {
        return prev.filter(c => c.id !== componentId);
      }
      const template = COMPONENT_LIBRARY.find(c => c.id === componentId);
      if (!template) return prev;
      return [...prev, { ...template, enabled: true, order: prev.length }];
    });
  }, []);

  // Reorder components
  const moveComponent = useCallback((fromIndex: number, toIndex: number) => {
    setComponents(prev => {
      const result = [...prev];
      const [removed] = result.splice(fromIndex, 1);
      result.splice(toIndex, 0, removed);
      return result.map((c, idx) => ({ ...c, order: idx }));
    });
  }, []);

  // Update component config
  const updateComponentConfig = useCallback((componentId: string, config: Partial<ComponentConfig>) => {
    setComponents(prev =>
      prev.map(c =>
        c.id === componentId
          ? { ...c, config: { ...c.config, ...config } }
          : c
      )
    );
  }, []);

  // Save template
  const handleSaveTemplate = useCallback(() => {
    if (!templateName.trim()) return;

    const template: PackageTemplate = {
      id: `template-${Date.now()}`,
      name: templateName,
      description: `${AUDIENCE_PRESETS[selectedAudience].name} package`,
      audience: selectedAudience,
      components,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    onSaveTemplate?.(template);
    setShowTemplateDialog(false);
    setTemplateName('');
  }, [templateName, selectedAudience, components, onSaveTemplate]);

  // Load template
  const loadTemplate = useCallback((template: PackageTemplate) => {
    setComponents(template.components);
    setSelectedAudience(template.audience);
  }, []);

  // Export package
  const handleExport = useCallback((format: ExportFormat) => {
    if (components.length === 0) {
      alert('Please add at least one component to your package before exporting.');
      return;
    }

    setIsExporting(true);
    try {
      // Perform the actual export
      performExport(format, components, selectedAudience);
      // Also call optional callback if provided
      onExport?.(format, components);
    } catch (error) {
      console.error('Export error:', error);
      alert('Export failed. Please try again.');
    } finally {
      setIsExporting(false);
      setShowExportDialog(false);
    }
  }, [components, selectedAudience, onExport]);

  // Component count by enabled status
  const enabledCount = useMemo(() => components.length, [components]);

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-slate-800">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <span>📦</span>
              <span>Deliverable Package Builder</span>
            </h2>
            <p className="text-xs text-slate-500 mt-1">
              Configure, arrange, and export intelligence packages for your audience
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPreviewMode(!previewMode)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                previewMode
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
            >
              <Eye className="w-3.5 h-3.5" />
              Preview
            </button>
            <button
              onClick={() => setShowExportDialog(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 text-white rounded text-xs font-medium hover:bg-emerald-500 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              Export
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex">
        {/* Left Panel - Component Library */}
        <div className="w-64 border-r border-slate-800 p-4">
          {/* Audience Presets */}
          <div className="mb-6">
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
              Audience Preset
            </h3>
            <div className="space-y-1">
              {Object.entries(AUDIENCE_PRESETS).map(([key, preset]) => (
                <button
                  key={key}
                  onClick={() => applyPreset(key as AudiencePreset)}
                  className={`w-full flex items-center gap-2 px-3 py-2 rounded text-left transition-colors ${
                    selectedAudience === key
                      ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
                      : 'bg-slate-800/50 text-slate-400 hover:bg-slate-800 border border-transparent'
                  }`}
                >
                  {preset.icon}
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate">{preset.name}</div>
                    <div className="text-xs text-slate-500 truncate">{preset.description}</div>
                  </div>
                  {selectedAudience === key && <Check className="w-4 h-4 text-blue-400" />}
                </button>
              ))}
            </div>
          </div>

          {/* Component Library */}
          <div>
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
              Components ({enabledCount} selected)
            </h3>
            <div className="space-y-1 max-h-80 overflow-y-auto">
              {COMPONENT_LIBRARY.map(component => {
                const isEnabled = components.some(c => c.id === component.id);
                return (
                  <button
                    key={component.id}
                    onClick={() => toggleComponent(component.id)}
                    className={`w-full flex items-center gap-2 px-3 py-2 rounded text-left transition-colors ${
                      isEnabled
                        ? 'bg-emerald-600/20 text-emerald-400 border border-emerald-500/30'
                        : 'bg-slate-800/50 text-slate-400 hover:bg-slate-800 border border-transparent'
                    }`}
                  >
                    <span className="text-lg">{component.icon}</span>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium truncate">{component.label}</div>
                    </div>
                    {isEnabled ? (
                      <Check className="w-4 h-4 text-emerald-400" />
                    ) : (
                      <Plus className="w-4 h-4 text-slate-500" />
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Center Panel - Package Arrangement OR Preview */}
        <div className="flex-1 p-4 overflow-y-auto max-h-[600px]">
          {previewMode ? (
            /* Preview Mode - Show actual content */
            <div className="space-y-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-bold text-slate-300">
                  Preview - {AUDIENCE_PRESETS[selectedAudience].name}
                </h3>
                <span className="text-xs text-slate-500">
                  {components.length} sections
                </span>
              </div>

              {components.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-64 border-2 border-dashed border-slate-700 rounded-lg">
                  <div className="text-4xl mb-3">👁️</div>
                  <p className="text-slate-400 text-sm">No components to preview</p>
                  <p className="text-slate-500 text-xs mt-1">Add components or select a preset first</p>
                </div>
              ) : (
                <div className="bg-slate-950 border border-slate-700 rounded-lg overflow-hidden">
                  {/* Preview Header */}
                  <div className="bg-slate-800 px-4 py-3 border-b border-slate-700">
                    <h2 className="text-lg font-bold text-white">LatticeForge Intelligence Package</h2>
                    <p className="text-sm text-slate-400">{AUDIENCE_PRESETS[selectedAudience].name}</p>
                    <div className="flex gap-4 mt-2 text-xs text-slate-500">
                      <span>Generated: {new Date().toLocaleDateString()}</span>
                      <span>Classification: OSINT / UNCLASSIFIED</span>
                    </div>
                  </div>

                  {/* Preview Sections */}
                  <div className="divide-y divide-slate-800">
                    {components.sort((a, b) => a.order - b.order).map(component => (
                      <div key={component.id} className="p-4">
                        <h3 className="text-sm font-bold text-white flex items-center gap-2 mb-2">
                          <span>{component.icon}</span>
                          {component.label}
                          <span className="ml-auto text-xs px-2 py-0.5 bg-slate-800 text-slate-400 rounded">
                            {component.config.detailLevel || 'standard'}
                          </span>
                        </h3>
                        <pre className="text-xs text-slate-400 whitespace-pre-wrap font-mono bg-slate-900/50 p-3 rounded">
                          {generateSectionContent(component)}
                        </pre>
                      </div>
                    ))}
                  </div>

                  {/* Preview Footer */}
                  <div className="bg-slate-800 px-4 py-2 text-center text-xs text-slate-500">
                    LatticeForge Intelligence Platform | OSINT Only
                  </div>
                </div>
              )}
            </div>
          ) : (
          /* Edit Mode - Package Layout */
          <>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold text-slate-300">
              Package Layout
            </h3>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowTemplateDialog(true)}
                className="flex items-center gap-1.5 px-2 py-1 bg-slate-800 text-slate-400 rounded text-xs hover:bg-slate-700 transition-colors"
              >
                <Save className="w-3 h-3" />
                Save Template
              </button>
              {templates.length > 0 && (
                <div className="relative group">
                  <button className="flex items-center gap-1.5 px-2 py-1 bg-slate-800 text-slate-400 rounded text-xs hover:bg-slate-700 transition-colors">
                    <FolderOpen className="w-3 h-3" />
                    Load
                    <ChevronDown className="w-3 h-3" />
                  </button>
                  <div className="absolute right-0 top-full mt-1 w-48 bg-slate-800 border border-slate-700 rounded shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10">
                    {templates.map(template => (
                      <button
                        key={template.id}
                        onClick={() => loadTemplate(template)}
                        className="w-full text-left px-3 py-2 text-xs text-slate-300 hover:bg-slate-700"
                      >
                        {template.name}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Draggable Components */}
          {components.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 border-2 border-dashed border-slate-700 rounded-lg">
              <div className="text-4xl mb-3">📦</div>
              <p className="text-slate-400 text-sm">Select an audience preset or add components</p>
              <p className="text-slate-500 text-xs mt-1">Components can be reordered by dragging</p>
            </div>
          ) : (
            <div className="space-y-2">
              {components
                .sort((a, b) => a.order - b.order)
                .map((component, index) => (
                  <div
                    key={component.id}
                    draggable
                    onDragStart={() => setDraggedItem(component.id)}
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={() => {
                      if (draggedItem && draggedItem !== component.id) {
                        const fromIndex = components.findIndex(c => c.id === draggedItem);
                        moveComponent(fromIndex, index);
                      }
                      setDraggedItem(null);
                    }}
                    onClick={() => setSelectedComponent(
                      selectedComponent === component.id ? null : component.id
                    )}
                    className={`flex items-center gap-3 p-3 rounded-lg border transition-all cursor-pointer ${
                      selectedComponent === component.id
                        ? 'bg-blue-600/10 border-blue-500/50'
                        : 'bg-slate-800/50 border-slate-700 hover:border-slate-600'
                    } ${draggedItem === component.id ? 'opacity-50' : ''}`}
                  >
                    <GripVertical className="w-4 h-4 text-slate-600 cursor-grab" />
                    <span className="text-lg">{component.icon}</span>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-white">{component.label}</div>
                      <div className="text-xs text-slate-500">{component.description}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      {component.config.classification && (
                        <span className="text-xs px-1.5 py-0.5 bg-slate-700 text-slate-300 rounded">
                          {component.config.classification}
                        </span>
                      )}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleComponent(component.id);
                        }}
                        className="p-1 text-slate-500 hover:text-red-400 transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))}
            </div>
          )}
          </>
          )}
        </div>

        {/* Right Panel - Component Config */}
        {selectedComponent && (
          <div className="w-72 border-l border-slate-800 p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold text-slate-300">Component Settings</h3>
              <button
                onClick={() => setSelectedComponent(null)}
                className="text-slate-500 hover:text-slate-400"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {(() => {
              const component = components.find(c => c.id === selectedComponent);
              if (!component) return null;

              return (
                <div className="space-y-4">
                  {/* Detail Level */}
                  <div>
                    <label className="text-xs font-medium text-slate-400 block mb-2">
                      Detail Level
                    </label>
                    <div className="flex gap-1">
                      {(['summary', 'standard', 'detailed'] as const).map(level => (
                        <button
                          key={level}
                          onClick={() => updateComponentConfig(component.id, { detailLevel: level })}
                          className={`flex-1 px-2 py-1.5 text-xs rounded capitalize transition-colors ${
                            component.config.detailLevel === level
                              ? 'bg-blue-600 text-white'
                              : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                          }`}
                        >
                          {level}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Classification - STUBBED (no authority) */}
                  <div>
                    <label className="text-xs font-medium text-slate-400 block mb-2">
                      Classification
                    </label>
                    <div className="w-full bg-slate-800/50 border border-slate-700 text-slate-500 text-sm rounded px-3 py-2">
                      OSINT Only (Open Source)
                    </div>
                    <p className="text-xs text-slate-600 mt-1">No classification authority</p>
                  </div>

                  {/* Toggles */}
                  <div className="space-y-2">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={component.config.showConfidence ?? false}
                        onChange={(e) => updateComponentConfig(component.id, {
                          showConfidence: e.target.checked
                        })}
                        className="rounded bg-slate-800 border-slate-600"
                      />
                      <span className="text-sm text-slate-400">Show confidence levels</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={component.config.showSources ?? false}
                        onChange={(e) => updateComponentConfig(component.id, {
                          showSources: e.target.checked
                        })}
                        className="rounded bg-slate-800 border-slate-600"
                      />
                      <span className="text-sm text-slate-400">Show sources</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={component.config.showTimestamps ?? false}
                        onChange={(e) => updateComponentConfig(component.id, {
                          showTimestamps: e.target.checked
                        })}
                        className="rounded bg-slate-800 border-slate-600"
                      />
                      <span className="text-sm text-slate-400">Show timestamps</span>
                    </label>
                  </div>

                  {/* Custom Notes */}
                  <div>
                    <label className="text-xs font-medium text-slate-400 block mb-2">
                      Custom Notes
                    </label>
                    <textarea
                      value={component.config.customNotes || ''}
                      onChange={(e) => updateComponentConfig(component.id, {
                        customNotes: e.target.value
                      })}
                      placeholder="Add notes for this section..."
                      className="w-full bg-slate-800 border border-slate-700 text-slate-300 text-sm rounded px-3 py-2 h-20 resize-none"
                    />
                  </div>
                </div>
              );
            })()}
          </div>
        )}
      </div>

      {/* Save Template Dialog */}
      {showTemplateDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-slate-900 border border-slate-700 rounded-xl p-6 w-96 shadow-xl">
            <h3 className="text-lg font-bold text-white mb-4">Save Template</h3>
            <input
              type="text"
              value={templateName}
              onChange={(e) => setTemplateName(e.target.value)}
              placeholder="Template name..."
              className="w-full bg-slate-800 border border-slate-700 text-white rounded px-3 py-2 mb-4"
            />
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowTemplateDialog(false)}
                className="px-4 py-2 text-slate-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveTemplate}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-500 transition-colors"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Export Dialog */}
      {showExportDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-slate-900 border border-slate-700 rounded-xl p-6 w-96 shadow-xl">
            <h3 className="text-lg font-bold text-white mb-4">Export Package</h3>
            <p className="text-sm text-slate-400 mb-4">
              Select a format for your {components.length} component package:
            </p>
            <div className="grid grid-cols-2 gap-2 mb-4">
              {EXPORT_FORMATS.map(format => (
                <button
                  key={format.format}
                  onClick={() => handleExport(format.format)}
                  className="flex items-center gap-2 p-3 bg-slate-800 border border-slate-700 rounded-lg hover:border-blue-500 transition-colors text-left"
                >
                  {format.icon}
                  <div>
                    <div className="text-sm font-medium text-white">{format.label}</div>
                    <div className="text-xs text-slate-500">{format.description}</div>
                  </div>
                </button>
              ))}
            </div>
            <div className="flex justify-end">
              <button
                onClick={() => setShowExportDialog(false)}
                className="px-4 py-2 text-slate-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default PackageBuilder;
