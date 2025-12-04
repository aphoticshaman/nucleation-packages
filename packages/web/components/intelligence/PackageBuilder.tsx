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
  Mail,
  Send,
} from 'lucide-react';

// Import real intelligence data
import {
  MIDDLE_EAST_DISPUTES,
  EUROPE_DISPUTES,
  DOMESTIC_INSTABILITY,
  TRANSNATIONAL_THREATS,
  INTELLIGENCE_AGENCIES,
  getHighestRiskFlashpoints,
  getActiveConflicts,
  getCountriesByInstability,
} from '@/lib/global-flashpoints';

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

/**
 * Escape HTML to prevent XSS when injecting content into HTML templates.
 * This is critical for security when generating export documents.
 */
function escapeHtml(text: string): string {
  const htmlEscapes: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#x27;',
    '/': '&#x2F;',
  };
  return text.replace(/[&<>"'/]/g, char => htmlEscapes[char] || char);
}

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
  // Get real intelligence data
  const activeConflicts = getActiveConflicts();
  const highRiskFlashpoints = getHighestRiskFlashpoints(5);
  const unstableCountries = getCountriesByInstability();
  const now = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });

  const contentMap: Record<ComponentType, string> = {
    executive_summary: `EXECUTIVE SUMMARY
Assessment Date: ${now}
Classification: OSINT / UNCLASSIFIED

SITUATION OVERVIEW:
${activeConflicts.length} active conflicts currently monitored. ${highRiskFlashpoints.length} flashpoints exceed 70% escalation probability.

PRIMARY CONCERNS:
${activeConflicts.slice(0, 3).map((c, i) => `${i + 1}. ${c.name} (${c.region}) - Escalation Risk: ${Math.round(c.escalationRisk * 100)}%`).join('\n')}

KEY FINDINGS:
• Russia-Ukraine War remains largest European conflict since 1945
• Middle East multi-front escalation (Israel-Hamas, Lebanon, Iran, Houthis)
• European political instability accelerating (France, Germany coalition failures)
• US domestic polarization at historic levels ahead of transition

BOTTOM LINE:
Global instability elevated across all monitored vectors. Cascade risk between theaters is HIGH due to interconnected actors (Iran-Russia axis, China positioning).

${component.config.showConfidence ? 'Confidence: HIGH (Multiple source corroboration)\nAnalyst: LatticeForge OSINT Team' : ''}`,

    bluf: `BOTTOM LINE UP FRONT

WHAT: Multiple interconnected crises threatening global stability
WHO: Key actors - Russia, Iran, Israel, US, China, non-state actors (Hamas, Hezbollah, Houthis)
WHEN: Active NOW. Critical window: Next 90 days (US transition, Middle East trajectory)
WHERE: Primary theaters - Ukraine, Levant, Red Sea, Taiwan Strait (watch)
WHY: Great power competition + regional proxy conflicts + domestic instability convergence
HOW: Cascade dynamics - conflict in one theater enables/triggers others

IMMEDIATE CONCERNS:
1. Iran nuclear breakout timeline: 1-2 weeks to weapons-grade material
2. Ukraine war escalation spiral (ATACMS in Russia → new Russian missiles)
3. France government collapse - 4th Republic parallels
4. US transition period vulnerability

RECOMMENDED POSTURE: ELEVATED MONITORING`,

    threat_matrix: `THREAT ASSESSMENT MATRIX
Generated: ${now}

ACTIVE CONFLICTS (Immediate Threats):
${activeConflicts.map(c => `
${c.name}
├─ Status: ${c.status.replace('_', ' ').toUpperCase()}
├─ Escalation Risk: ${Math.round(c.escalationRisk * 100)}%
├─ Strategic Importance: ${Math.round(c.strategicImportance * 100)}%
├─ Parties: ${c.parties.map(p => p.name).join(' vs ')}
├─ Resources at Stake: ${c.resourcesAtStake.slice(0, 3).join(', ')}
└─ Last Incident: ${c.recentIncidents[0]?.description || 'N/A'}
`).join('\n')}

ESCALATION PROBABILITY RANKING:
${highRiskFlashpoints.map((f, i) => `${i + 1}. ${f.name}: ${Math.round(f.escalationRisk * 100)}%`).join('\n')}`,

    key_developments: `KEY DEVELOPMENTS
Period: Last 30 Days | Source: OSINT Collection

CRITICAL (Immediate Action Required):
${Object.values(MIDDLE_EAST_DISPUTES).flatMap(d => d.recentIncidents.filter(i => i.severity === 'critical').map(i => `• ${i.date}: ${i.description}`)).slice(0, 4).join('\n')}

${Object.values(EUROPE_DISPUTES).flatMap(d => d.recentIncidents.filter(i => i.severity === 'critical').map(i => `• ${i.date}: ${i.description}`)).slice(0, 3).join('\n')}

SERIOUS (Monitor Closely):
${Object.values(MIDDLE_EAST_DISPUTES).flatMap(d => d.recentIncidents.filter(i => i.severity === 'serious').map(i => `• ${i.date}: ${i.description}`)).slice(0, 3).join('\n')}

DOMESTIC INSTABILITY INDICATORS:
${unstableCountries.slice(0, 4).map(c => `• ${c.country} (${c.severity.toUpperCase()}): ${c.flashpoints[0]?.name || 'Multiple factors'}`).join('\n')}`,

    risk_gauge: `GLOBAL RISK ASSESSMENT
Date: ${now}

COMPOSITE RISK INDEX: ${(highRiskFlashpoints.reduce((sum, f) => sum + f.escalationRisk, 0) / highRiskFlashpoints.length * 10).toFixed(1)}/10

BY REGION:
• Middle East: ${Math.round(Object.values(MIDDLE_EAST_DISPUTES).reduce((sum, d) => sum + d.escalationRisk, 0) / Object.values(MIDDLE_EAST_DISPUTES).length * 100)}% (CRITICAL)
• Europe: ${Math.round(Object.values(EUROPE_DISPUTES).reduce((sum, d) => sum + d.escalationRisk, 0) / Object.values(EUROPE_DISPUTES).length * 100)}% (ELEVATED)
• Domestic (Major Powers): ${unstableCountries.filter(c => ['US', 'GB', 'FR', 'DE'].includes(c.countryCode)).length > 2 ? 'ELEVATED' : 'MODERATE'}

TREND: DETERIORATING
• 3 new critical incidents in past 30 days
• Iran-Israel direct exchanges crossed previous red lines
• European political instability accelerating

CASCADE RISK: HIGH
Interconnection density between theaters has increased. Conflict in one region more likely to trigger responses in others (Iran network activation pattern observed Oct 2023-present).`,

    map_view: `GEOGRAPHIC ANALYSIS
Primary Theaters of Concern

MIDDLE EAST / LEVANT:
${Object.values(MIDDLE_EAST_DISPUTES).map(d => `• ${d.name}
  Location: ${d.region}
  Status: ${d.status.replace('_', ' ')}
  Control: ${d.parties.map(p => `${p.name} (${p.controlPercentage}%)`).join(', ')}`).join('\n\n')}

EUROPE:
${Object.values(EUROPE_DISPUTES).map(d => `• ${d.name}
  Location: ${d.region}
  Status: ${d.status.replace('_', ' ')}
  Escalation: ${Math.round(d.escalationRisk * 100)}%`).join('\n\n')}

CRITICAL CHOKEPOINTS:
• Strait of Hormuz (20% global oil) - Iranian interdiction capability
• Bab el-Mandeb (Red Sea) - Active Houthi attacks on shipping
• Suwalki Gap (NATO) - Russia-NATO potential flashpoint
• Taiwan Strait - Chinese military activity elevated`,

    network_graph: `NETWORK ANALYSIS
Actor Relationships & Influence Mapping

AXIS STRUCTURE:
Iran-Russia-China Alignment
├─ Military: Russia-Iran drone/missile cooperation
├─ Economic: China-Russia energy partnership, sanctions evasion
├─ Proxy: Iran → Hezbollah, Hamas, Houthis, Iraqi militias
└─ Information: Coordinated narrative operations

PROXY NETWORKS:
Iranian Network (Active):
• Hezbollah (Lebanon) - 150,000+ rockets, precision missiles
• Hamas (Gaza) - Degraded but not eliminated
• Houthis (Yemen) - Disrupting 15% global shipping
• Iraqi PMF - Attacks on US bases

KEY INTELLIGENCE AGENCIES:
${Object.values(INTELLIGENCE_AGENCIES).filter(a => ['US', 'IL', 'RU', 'CN', 'GB'].includes(a.countryCode)).map(a => `• ${a.name} (${a.country})
  Capabilities: HUMINT ${Math.round(a.capabilities.humint * 100)}%, SIGINT ${Math.round(a.capabilities.sigint * 100)}%, Cyber ${Math.round(a.capabilities.cyber * 100)}%`).join('\n')}`,

    causal_chain: `CAUSAL CHAIN ANALYSIS

MIDDLE EAST ESCALATION PATHWAY:
Oct 7 Hamas Attack
└─→ Israeli Gaza Operation
    └─→ Hezbollah "Support Front"
        └─→ Israeli-Lebanon Escalation (Sep 2024)
            └─→ Iranian Direct Strikes (Apr, Oct 2024)
                └─→ [POTENTIAL] Full Regional War

UKRAINE-NATO ESCALATION PATHWAY:
Russian Invasion (Feb 2022)
└─→ Western Weapons Supply
    └─→ Ukrainian Deep Strikes
        └─→ Russian Escalation (new missiles)
            └─→ [POTENTIAL] NATO Article 5 trigger (Baltic incident)

EUROPEAN INSTABILITY CASCADE:
Energy Crisis (2022)
└─→ Inflation Surge
    └─→ Cost of Living Protests
        └─→ Government Instability (FR, DE)
            └─→ Far-Right Electoral Gains
                └─→ EU Cohesion Degradation

ROOT DRIVERS:
1. US-China strategic competition
2. Russian revisionism post-Cold War
3. Iranian regional ambitions
4. Climate-driven resource competition
5. Technology disruption of power structures`,

    timeline: `EVENT TIMELINE
Historical Context & Projections

PAST 24 MONTHS:
${[
  ...Object.values(MIDDLE_EAST_DISPUTES).flatMap(d => d.recentIncidents.map(i => ({ ...i, region: d.region }))),
  ...Object.values(EUROPE_DISPUTES).flatMap(d => d.recentIncidents.map(i => ({ ...i, region: d.region }))),
].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()).slice(0, 12).map(i => `${i.date}: ${i.description} [${i.severity.toUpperCase()}]`).join('\n')}

PROJECTED CRITICAL WINDOWS:
• T+30 days: US transition period (vulnerability window)
• T+60 days: Iranian nuclear decision point
• T+90 days: Ukraine winter offensive/defensive posture
• T+180 days: European elections cycle (potential shifts)

WATCH DATES:
• January 20, 2025: US inauguration
• Early 2025: German snap election
• 2025 H1: Potential Iran nuclear threshold`,

    data_table: `QUANTITATIVE INDICATORS

CONFLICT METRICS:
| Theater | Escalation Risk | Strategic Import | Active Parties |
|---------|-----------------|------------------|----------------|
${[...Object.values(MIDDLE_EAST_DISPUTES), ...Object.values(EUROPE_DISPUTES)].slice(0, 8).map(d => `| ${d.name.slice(0, 20)} | ${Math.round(d.escalationRisk * 100)}% | ${Math.round(d.strategicImportance * 100)}% | ${d.parties.length} |`).join('\n')}

DOMESTIC INSTABILITY:
| Country | Severity | Trajectory | Key Threat |
|---------|----------|------------|------------|
${unstableCountries.slice(0, 6).map(c => `| ${c.country} | ${c.severity.toUpperCase()} | ${c.trajectory} | ${c.threatTypes[0]} |`).join('\n')}

TRANSNATIONAL THREATS:
| Threat | Annual Value | Violence | State Capture |
|--------|--------------|----------|---------------|
${Object.values(TRANSNATIONAL_THREATS).slice(0, 4).map(t => `| ${t.name.slice(0, 25)} | ${t.estimatedAnnualValue} | ${t.violenceLevel} | ${t.stateCapture} |`).join('\n')}`,

    recommendations: `RECOMMENDATIONS

IMMEDIATE ACTIONS (0-30 days):
1. [CRITICAL] Monitor Iran nuclear indicators daily
2. [CRITICAL] Track Hezbollah reconstruction/repositioning
3. [HIGH] Assess US transition security posture
4. [HIGH] Monitor European government stability

MEDIUM-TERM (30-90 days):
1. Update contingency plans for Middle East regional war
2. Assess Ukraine conflict trajectory post-winter
3. Evaluate European far-right coalition scenarios
4. Review supply chain exposure to conflict zones

ONGOING COLLECTION PRIORITIES:
• Iranian nuclear program indicators
• Russian military capability regeneration
• Chinese Taiwan posture signals
• Cartel-state nexus developments
• Cyber threat actor campaigns

RISK MITIGATION:
• Diversify energy/commodity suppliers
• Stress-test financial exposure to sanctioned entities
• Update crisis communication protocols
• Review personnel security in elevated risk zones`,

    sources: `SOURCES AND METHODS

COLLECTION FRAMEWORK:
This assessment draws on open-source intelligence (OSINT) including:
• Government statements and official documents
• Major wire services (AP, Reuters, AFP)
• Regional media monitoring
• Academic and think tank analysis
• Social media analysis (verified accounts)
• Commercial satellite imagery analysis
• Financial market indicators

SOURCE RELIABILITY:
• Official government sources: Generally reliable (bias acknowledged)
• Major media: Reliable with verification
• Regional sources: Variable, cross-referenced
• Social media: Used for indications only

LIMITATIONS:
• No access to classified intelligence
• Potential information operations contamination
• Time lag on ground truth verification
• Language/cultural interpretation challenges

CONFIDENCE LEVELS USED:
• HIGH: Multiple independent sources, consistent pattern
• MODERATE: 2-3 sources, some gaps
• LOW: Single source or conflicting information

METHODOLOGY:
Structured analytic techniques including:
• Analysis of Competing Hypotheses
• Key Assumptions Check
• Red Team Analysis
• Scenario Planning`,

    appendix: `APPENDIX

A. GLOSSARY OF TERMS
• OSINT: Open Source Intelligence
• HUMINT: Human Intelligence
• SIGINT: Signals Intelligence
• BLUF: Bottom Line Up Front
• Escalation Risk: Probability of conflict intensification (0-100%)
• Cascade Effect: Secondary impacts triggered by primary event
• Axis: Aligned group of state/non-state actors
• Proxy: Actor operating on behalf of another power

B. REGIONAL PRIMERS
${Object.values(MIDDLE_EAST_DISPUTES).slice(0, 3).map(d => `
${d.name}:
${d.analystAssessment.trim()}`).join('\n')}

C. KEY ACTORS REFERENCE
${Object.values(INTELLIGENCE_AGENCIES).slice(0, 6).map(a => `• ${a.name}: ${a.notes}`).join('\n')}

D. DATA SOURCES
• LatticeForge platform indicators
• Global flashpoint database
• Historical pattern library
• Cascade simulation models`,

    custom_section: component.config.customNotes || 'Custom section - add analyst notes here.',
  };

  return contentMap[component.type] || 'Section content not available.';
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
  <title>${escapeHtml(content.title)}</title>
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
      <h1>${escapeHtml(content.title)}</h1>
      <p class="subtitle">${escapeHtml(presetInfo.name)}</p>
      <div class="meta">
        <span>Generated: ${escapeHtml(new Date(content.generatedAt).toLocaleString())}</span>
        <span>Sections: ${String(content.sections.length)}</span>
        <span>Classification: OSINT / UNCLASSIFIED</span>
      </div>
    </header>
    ${content.sections.map(s => `
    <section>
      <h2><span>${escapeHtml(s.icon)}</span> ${escapeHtml(s.title)} <span class="badge">${escapeHtml(s.config.detailLevel || 'standard')}</span></h2>
      <pre>${escapeHtml(s.content)}</pre>
    </section>`).join('')}
    <footer>
      LatticeForge Intelligence Platform | ${new Date().getFullYear()} | OSINT Only
    </footer>
  </div>
</body>
</html>`;

  downloadFile(html, `latticeforge-package-${Date.now()}.html`, 'text/html');
}

// nosemgrep: javascript.lang.security.audit.unknown-value-with-script-tag.unknown-value-with-script-tag
function exportAsPDF(components: PackageComponent[], audience: AudiencePreset): void {
  // Generate HTML and use browser print to PDF
  // Note: All dynamic content is sanitized via escapeHtml() before template insertion
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
  <title>${escapeHtml(content.title)}</title>
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
      <h1>${escapeHtml(content.title)}</h1>
      <p class="subtitle">${escapeHtml(presetInfo.name)}</p>
      <div class="meta">
        <span>Generated: ${escapeHtml(new Date(content.generatedAt).toLocaleString())}</span>
        <span>Sections: ${String(content.sections.length)}</span>
      </div>
    </header>
    ${content.sections.map(s => `
    <section>
      <h2>${escapeHtml(s.icon)} ${escapeHtml(s.title)} <span class="badge">${escapeHtml(s.config.detailLevel || 'standard')}</span></h2>
      <pre>${escapeHtml(s.content)}</pre>
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
  <title>${escapeHtml(content.title)}</title>
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
    <h1>${escapeHtml(content.title)}</h1>
    <p><strong>${escapeHtml(presetInfo.name)}</strong></p>
    <p class="meta">Generated: ${escapeHtml(new Date(content.generatedAt).toLocaleString())} | Classification: OSINT / UNCLASSIFIED</p>
  </div>
  ${content.sections.map(s => `
  <h2>${escapeHtml(s.icon)} ${escapeHtml(s.title)}</h2>
  <pre>${escapeHtml(s.content)}</pre>`).join('')}
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
      <h2>${escapeHtml(s.icon)} ${escapeHtml(s.title)}</h2>
      <pre>${escapeHtml(s.content)}</pre>
    </div>
  `);

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>${escapeHtml(content.title)} - Presentation</title>
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
    <h1>${escapeHtml(content.title)}</h1>
    <p class="subtitle">${escapeHtml(presetInfo.name)}</p>
    <p class="meta">Generated ${escapeHtml(new Date(content.generatedAt).toLocaleDateString())}</p>
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
  const [showEmailDialog, setShowEmailDialog] = useState(false);
  const [emailAddress, setEmailAddress] = useState('');
  const [emailOptions, setEmailOptions] = useState({
    includeTextBody: true,
    includePdfAttachment: false,
    includeJsonAttachment: false,
    includeMarkdownAttachment: true,
  });
  const [emailSending, setEmailSending] = useState(false);
  const [emailSent, setEmailSent] = useState(false);
  const [draggedItem, setDraggedItem] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  // Mobile accordion state
  const [mobileSection, setMobileSection] = useState<'audience' | 'components' | 'layout' | 'config' | null>('audience');

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
    // On mobile, auto-advance to layout after selecting preset
    if (window.innerWidth < 1024) {
      setMobileSection('layout');
    }
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

  // Email export
  const handleEmailSend = useCallback(async () => {
    if (components.length === 0) {
      alert('Please add at least one component to your package before emailing.');
      return;
    }

    if (!emailAddress.trim()) {
      alert('Please enter an email address.');
      return;
    }

    setEmailSending(true);
    try {
      const packageContent = generatePackageContent(components, selectedAudience);
      const response = await fetch('/api/export/email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          recipientEmail: emailAddress,
          subject: `LatticeForge Intelligence Package - ${AUDIENCE_PRESETS[selectedAudience].name}`,
          includeTextBody: emailOptions.includeTextBody,
          includePdfAttachment: emailOptions.includePdfAttachment,
          includeJsonAttachment: emailOptions.includeJsonAttachment,
          includeMarkdownAttachment: emailOptions.includeMarkdownAttachment,
          packageContent,
          audience: AUDIENCE_PRESETS[selectedAudience].name,
        }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || 'Failed to send email');
      }

      setEmailSent(true);
      setTimeout(() => {
        setShowEmailDialog(false);
        setEmailSent(false);
        setEmailAddress('');
      }, 2000);
    } catch (error) {
      console.error('Email send error:', error);
      alert(error instanceof Error ? error.message : 'Failed to send email. Please try again.');
    } finally {
      setEmailSending(false);
    }
  }, [components, selectedAudience, emailAddress, emailOptions]);

  // Component count by enabled status
  const enabledCount = useMemo(() => components.length, [components]);

  // Mobile accordion section component
  const MobileAccordion = ({ id, title, icon, children, badge }: { id: 'audience' | 'components' | 'layout' | 'config'; title: string; icon: React.ReactNode; children: React.ReactNode; badge?: string }) => (
    <div className="border-b border-slate-800 lg:hidden">
      <button
        onClick={() => setMobileSection(mobileSection === id ? null : id)}
        className="w-full flex items-center justify-between p-4 text-left"
      >
        <div className="flex items-center gap-3">
          {icon}
          <span className="font-medium text-white">{title}</span>
          {badge && <span className="text-xs px-2 py-0.5 bg-blue-600/20 text-blue-400 rounded-full">{badge}</span>}
        </div>
        <ChevronDown className={`w-5 h-5 text-slate-400 transition-transform ${mobileSection === id ? 'rotate-180' : ''}`} />
      </button>
      {mobileSection === id && (
        <div className="px-4 pb-4">
          {children}
        </div>
      )}
    </div>
  );

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
      {/* Header - Sticky on mobile */}
      <div className="sticky top-0 z-20 bg-slate-900/95 backdrop-blur-sm p-4 border-b border-slate-800">
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0 flex-1">
            <h2 className="text-base sm:text-lg font-bold text-white flex items-center gap-2">
              <span className="hidden sm:inline">📦</span>
              <span className="truncate">Package Builder</span>
            </h2>
            <p className="text-xs text-slate-500 mt-0.5 hidden sm:block">
              Configure, arrange, and export intelligence packages
            </p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={() => setPreviewMode(!previewMode)}
              className={`flex items-center gap-1.5 px-2 sm:px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                previewMode
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
            >
              <Eye className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Preview</span>
            </button>
            <button
              onClick={() => setShowExportDialog(true)}
              className="flex items-center gap-1.5 px-2 sm:px-3 py-1.5 bg-emerald-600 text-white rounded text-xs font-medium hover:bg-emerald-500 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Export</span>
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Layout - Accordion sections */}
      <div className="lg:hidden">
        {/* Audience Preset Accordion */}
        <MobileAccordion id="audience" title="Audience" icon={<Users className="w-4 h-4 text-blue-400" />} badge={AUDIENCE_PRESETS[selectedAudience].name}>
          <div className="space-y-2">
            {Object.entries(AUDIENCE_PRESETS).map(([key, preset]) => (
              <button
                key={key}
                onClick={() => applyPreset(key as AudiencePreset)}
                className={`w-full flex items-center gap-3 px-3 py-3 rounded-lg text-left transition-colors ${
                  selectedAudience === key
                    ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
                    : 'bg-slate-800/50 text-slate-400 active:bg-slate-800 border border-transparent'
                }`}
              >
                {preset.icon}
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium">{preset.name}</div>
                  <div className="text-xs text-slate-500 truncate">{preset.description}</div>
                </div>
                {selectedAudience === key && <Check className="w-4 h-4 text-blue-400" />}
              </button>
            ))}
          </div>
        </MobileAccordion>

        {/* Components Accordion */}
        <MobileAccordion id="components" title="Components" icon={<Layout className="w-4 h-4 text-emerald-400" />} badge={`${enabledCount} selected`}>
          <div className="grid grid-cols-2 gap-2">
            {COMPONENT_LIBRARY.map(component => {
              const isEnabled = components.some(c => c.id === component.id);
              return (
                <button
                  key={component.id}
                  onClick={() => toggleComponent(component.id)}
                  className={`flex items-center gap-2 px-3 py-2.5 rounded-lg text-left transition-colors ${
                    isEnabled
                      ? 'bg-emerald-600/20 text-emerald-400 border border-emerald-500/30'
                      : 'bg-slate-800/50 text-slate-400 active:bg-slate-800 border border-transparent'
                  }`}
                >
                  <span className="text-lg">{component.icon}</span>
                  <span className="text-xs font-medium truncate flex-1">{component.label}</span>
                  {isEnabled && <Check className="w-3 h-3 text-emerald-400 shrink-0" />}
                </button>
              );
            })}
          </div>
        </MobileAccordion>

        {/* Layout Accordion */}
        <MobileAccordion id="layout" title="Layout" icon={<GripVertical className="w-4 h-4 text-amber-400" />} badge={`${components.length} sections`}>
          {previewMode ? (
            /* Mobile Preview */
            <div className="space-y-3">
              {components.length === 0 ? (
                <div className="text-center py-8 text-slate-500 text-sm">
                  No components selected
                </div>
              ) : (
                <div className="bg-slate-950 border border-slate-700 rounded-lg overflow-hidden max-h-[60vh] overflow-y-auto">
                  <div className="bg-slate-800 px-3 py-2 border-b border-slate-700">
                    <h3 className="text-sm font-bold text-white">Package Preview</h3>
                    <p className="text-xs text-slate-400">{AUDIENCE_PRESETS[selectedAudience].name}</p>
                  </div>
                  <div className="divide-y divide-slate-800">
                    {components.sort((a, b) => a.order - b.order).map(component => (
                      <div key={component.id} className="p-3">
                        <h4 className="text-xs font-bold text-white flex items-center gap-2 mb-2">
                          <span>{component.icon}</span>
                          {component.label}
                        </h4>
                        <pre className="text-[10px] text-slate-400 whitespace-pre-wrap font-mono bg-slate-900/50 p-2 rounded max-h-32 overflow-y-auto">
                          {generateSectionContent(component).slice(0, 500)}...
                        </pre>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            /* Mobile Layout Editor */
            <div className="space-y-2">
              {components.length === 0 ? (
                <div className="text-center py-8 text-slate-500 text-sm">
                  Select components above
                </div>
              ) : (
                components.sort((a, b) => a.order - b.order).map((component, index) => (
                  <div
                    key={component.id}
                    className="flex items-center gap-2 p-3 bg-slate-800/50 border border-slate-700 rounded-lg"
                  >
                    <div className="flex flex-col gap-1">
                      <button
                        onClick={() => index > 0 && moveComponent(index, index - 1)}
                        disabled={index === 0}
                        className="p-1 text-slate-500 hover:text-white disabled:opacity-30"
                      >
                        <ChevronDown className="w-3 h-3 rotate-180" />
                      </button>
                      <button
                        onClick={() => index < components.length - 1 && moveComponent(index, index + 1)}
                        disabled={index === components.length - 1}
                        className="p-1 text-slate-500 hover:text-white disabled:opacity-30"
                      >
                        <ChevronDown className="w-3 h-3" />
                      </button>
                    </div>
                    <span className="text-lg">{component.icon}</span>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-white truncate">{component.label}</div>
                    </div>
                    <button
                      onClick={() => setSelectedComponent(selectedComponent === component.id ? null : component.id)}
                      className="p-1.5 text-slate-400 hover:text-blue-400"
                    >
                      <Settings className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => toggleComponent(component.id)}
                      className="p-1.5 text-slate-400 hover:text-red-400"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                ))
              )}
            </div>
          )}
        </MobileAccordion>

        {/* Config Accordion - only show when component selected */}
        {selectedComponent && (
          <MobileAccordion id="config" title="Settings" icon={<Settings className="w-4 h-4 text-purple-400" />}>
            {(() => {
              const component = components.find(c => c.id === selectedComponent);
              if (!component) return <p className="text-sm text-slate-500">Select a component</p>;

              return (
                <div className="space-y-4">
                  <div>
                    <label className="text-xs font-medium text-slate-400 block mb-2">Detail Level</label>
                    <div className="flex gap-1">
                      {(['summary', 'standard', 'detailed'] as const).map(level => (
                        <button
                          key={level}
                          onClick={() => updateComponentConfig(component.id, { detailLevel: level })}
                          className={`flex-1 px-2 py-2 text-xs rounded capitalize transition-colors ${
                            component.config.detailLevel === level
                              ? 'bg-blue-600 text-white'
                              : 'bg-slate-800 text-slate-400'
                          }`}
                        >
                          {level}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="space-y-3">
                    <label className="flex items-center gap-3 cursor-pointer py-1">
                      <input
                        type="checkbox"
                        checked={component.config.showConfidence ?? false}
                        onChange={(e) => updateComponentConfig(component.id, { showConfidence: e.target.checked })}
                        className="rounded bg-slate-800 border-slate-600 w-5 h-5"
                      />
                      <span className="text-sm text-slate-300">Show confidence</span>
                    </label>
                    <label className="flex items-center gap-3 cursor-pointer py-1">
                      <input
                        type="checkbox"
                        checked={component.config.showSources ?? false}
                        onChange={(e) => updateComponentConfig(component.id, { showSources: e.target.checked })}
                        className="rounded bg-slate-800 border-slate-600 w-5 h-5"
                      />
                      <span className="text-sm text-slate-300">Show sources</span>
                    </label>
                  </div>
                </div>
              );
            })()}
          </MobileAccordion>
        )}
      </div>

      {/* Desktop Layout - 3 column */}
      <div className="hidden lg:flex">
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

            {/* Email Export Option */}
            <div className="border-t border-slate-700 pt-4 mt-4">
              <button
                onClick={() => {
                  setShowExportDialog(false);
                  setShowEmailDialog(true);
                }}
                className="w-full flex items-center justify-center gap-2 p-3 bg-cyan-600/20 border border-cyan-500/30 rounded-lg hover:bg-cyan-600/30 transition-colors text-cyan-300"
              >
                <Mail className="w-4 h-4" />
                <span className="font-medium">Send via Email</span>
              </button>
              <p className="text-xs text-slate-500 text-center mt-2">
                Email package with text and/or file attachments
              </p>
            </div>

            <div className="flex justify-end mt-4">
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

      {/* Email Export Dialog */}
      {showEmailDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-slate-900 border border-slate-700 rounded-xl p-6 w-full max-w-md shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <Mail className="w-5 h-5 text-cyan-400" />
                Email Package
              </h3>
              <button
                onClick={() => setShowEmailDialog(false)}
                className="text-slate-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {emailSent ? (
              <div className="text-center py-8">
                <div className="w-16 h-16 bg-emerald-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Check className="w-8 h-8 text-emerald-400" />
                </div>
                <h4 className="text-lg font-medium text-white mb-2">Sent!</h4>
                <p className="text-slate-400">Package sent to {emailAddress}</p>
              </div>
            ) : (
              <>
                <div className="space-y-4">
                  {/* Recipient Email */}
                  <div>
                    <label className="text-sm font-medium text-slate-300 block mb-2">
                      Recipient Email
                    </label>
                    <input
                      type="email"
                      value={emailAddress}
                      onChange={(e) => setEmailAddress(e.target.value)}
                      placeholder="colleague@example.com"
                      className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder:text-slate-500 focus:outline-none focus:border-cyan-500"
                    />
                  </div>

                  {/* Content Options */}
                  <div>
                    <label className="text-sm font-medium text-slate-300 block mb-3">
                      Include in Email
                    </label>
                    <div className="space-y-2">
                      <label className="flex items-center gap-3 cursor-pointer p-2 rounded hover:bg-slate-800/50">
                        <input
                          type="checkbox"
                          checked={emailOptions.includeTextBody}
                          onChange={(e) => setEmailOptions(prev => ({ ...prev, includeTextBody: e.target.checked }))}
                          className="rounded bg-slate-800 border-slate-600 w-4 h-4 text-cyan-500"
                        />
                        <div>
                          <span className="text-sm text-slate-300">Full text in email body</span>
                          <p className="text-xs text-slate-500">Readable directly in email</p>
                        </div>
                      </label>
                      <label className="flex items-center gap-3 cursor-pointer p-2 rounded hover:bg-slate-800/50">
                        <input
                          type="checkbox"
                          checked={emailOptions.includeMarkdownAttachment}
                          onChange={(e) => setEmailOptions(prev => ({ ...prev, includeMarkdownAttachment: e.target.checked }))}
                          className="rounded bg-slate-800 border-slate-600 w-4 h-4 text-cyan-500"
                        />
                        <div>
                          <span className="text-sm text-slate-300">Markdown file (.md)</span>
                          <p className="text-xs text-slate-500">Formatted document attachment</p>
                        </div>
                      </label>
                      <label className="flex items-center gap-3 cursor-pointer p-2 rounded hover:bg-slate-800/50">
                        <input
                          type="checkbox"
                          checked={emailOptions.includeJsonAttachment}
                          onChange={(e) => setEmailOptions(prev => ({ ...prev, includeJsonAttachment: e.target.checked }))}
                          className="rounded bg-slate-800 border-slate-600 w-4 h-4 text-cyan-500"
                        />
                        <div>
                          <span className="text-sm text-slate-300">JSON data file (.json)</span>
                          <p className="text-xs text-slate-500">Structured data for import</p>
                        </div>
                      </label>
                    </div>
                  </div>

                  {/* Package Summary */}
                  <div className="bg-slate-800/50 rounded-lg p-3 text-sm">
                    <div className="flex justify-between text-slate-400 mb-1">
                      <span>Package:</span>
                      <span className="text-white">{AUDIENCE_PRESETS[selectedAudience].name}</span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>Sections:</span>
                      <span className="text-white">{components.length}</span>
                    </div>
                  </div>
                </div>

                <div className="flex gap-3 mt-6">
                  <button
                    onClick={() => setShowEmailDialog(false)}
                    className="flex-1 px-4 py-3 text-slate-400 hover:text-white transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleEmailSend}
                    disabled={emailSending || !emailAddress.trim()}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-cyan-600 text-white rounded-lg hover:bg-cyan-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {emailSending ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Sending...
                      </>
                    ) : (
                      <>
                        <Send className="w-4 h-4" />
                        Send Package
                      </>
                    )}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default PackageBuilder;
