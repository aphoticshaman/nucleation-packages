/**
 * AUTOMATED COMPLIANCE CONFIGURATION
 *
 * Federal compliance without full-time certified staff:
 *
 * AUTOMATED PLATFORMS (do the heavy lifting):
 * - Vanta: SOC 2, ISO 27001, HIPAA, FedRAMP prep ($10k-25k/yr)
 * - Drata: SOC 2, ISO 27001, HIPAA automation ($10k-20k/yr)
 * - Secureframe: SOC 2, HIPAA, ISO 27001 ($10k-15k/yr)
 * - Thoropass: FedRAMP-focused automation
 *
 * FREE/CHEAP OPTIONS:
 * - GitHub Advanced Security: SAST, secrets scanning (free for public repos)
 * - Semgrep: SAST (already in your pipeline)
 * - OWASP ZAP: DAST scanning (free)
 * - Trivy: Container/dependency scanning (free)
 * - ScoutSuite: AWS/GCP/Azure security audit (free)
 *
 * YOUR EXPIRED CERTS:
 * - Security+ (CompTIA): $370 renewal, 50 CEUs
 * - CISSP: $135/yr AMF + 40 CPEs
 * - CEH: ~$80/yr + 120 ECE credits
 *
 * RECOMMENDATION: Don't renew unless needed for specific contracts.
 * Use automated platforms + documented processes instead.
 */

// ============================================
// FEDERAL COMPLIANCE ROADMAP
// ============================================

export interface ComplianceFramework {
  id: string;
  name: string;
  fullName: string;
  requiredFor: string[];
  automatable: number; // 0-100% automatable
  estimatedCost: string;
  timeline: string;
  certRequired: boolean;
  selfAssessment: boolean;
  tools: string[];
}

export const FEDERAL_FRAMEWORKS: ComplianceFramework[] = [
  {
    id: 'fedramp',
    name: 'FedRAMP',
    fullName: 'Federal Risk and Authorization Management Program',
    requiredFor: ['DoD', 'DHS', 'ICE', 'CBP', 'All federal agencies'],
    automatable: 60,
    estimatedCost: '$250k-500k for Low, $500k-1M+ for Moderate',
    timeline: '12-18 months',
    certRequired: true, // Need 3PAO assessment
    selfAssessment: false,
    tools: ['Thoropass', 'Vanta', 'Palantir AIP'],
  },
  {
    id: 'fedramp-tailored',
    name: 'FedRAMP Tailored',
    fullName: 'FedRAMP Tailored for Low-Impact SaaS',
    requiredFor: ['Low-risk federal SaaS'],
    automatable: 70,
    estimatedCost: '$50k-150k',
    timeline: '6-9 months',
    certRequired: true,
    selfAssessment: false,
    tools: ['Vanta', 'Drata', 'Secureframe'],
  },
  {
    id: 'stateramp',
    name: 'StateRAMP',
    fullName: 'State Risk and Authorization Management Program',
    requiredFor: ['State and local government'],
    automatable: 70,
    estimatedCost: '$30k-100k',
    timeline: '4-8 months',
    certRequired: true,
    selfAssessment: false,
    tools: ['Vanta', 'Drata'],
  },
  {
    id: 'nist-800-53',
    name: 'NIST 800-53',
    fullName: 'NIST Special Publication 800-53',
    requiredFor: ['Federal contractors', 'Defense'],
    automatable: 65,
    estimatedCost: '$20k-100k',
    timeline: '3-6 months',
    certRequired: false, // Self-assessment possible
    selfAssessment: true,
    tools: ['OpenSCAP', 'Vanta', 'Drata'],
  },
  {
    id: 'nist-800-171',
    name: 'NIST 800-171',
    fullName: 'Protecting Controlled Unclassified Information',
    requiredFor: ['DoD contractors with CUI'],
    automatable: 60,
    estimatedCost: '$15k-75k',
    timeline: '2-4 months',
    certRequired: false,
    selfAssessment: true,
    tools: ['SPRS self-assessment', 'Vanta', 'Drata'],
  },
  {
    id: 'cmmc',
    name: 'CMMC 2.0',
    fullName: 'Cybersecurity Maturity Model Certification',
    requiredFor: ['DoD contractors'],
    automatable: 55,
    estimatedCost: 'Level 1: Self-attest, Level 2: $50k-150k',
    timeline: 'Level 1: 1-2 months, Level 2: 6-12 months',
    certRequired: true, // Level 2+ needs C3PAO
    selfAssessment: true, // Level 1 only
    tools: ['CyberAB marketplace', 'Vanta', 'Summit 7'],
  },
  {
    id: 'soc2',
    name: 'SOC 2 Type II',
    fullName: 'Service Organization Control 2',
    requiredFor: ['Enterprise sales', 'Federal preferred'],
    automatable: 85,
    estimatedCost: '$20k-50k',
    timeline: '3-6 months + 6 month observation',
    certRequired: true, // Need CPA firm
    selfAssessment: false,
    tools: ['Vanta', 'Drata', 'Secureframe', 'Laika'],
  },
  {
    id: 'iso27001',
    name: 'ISO 27001',
    fullName: 'Information Security Management System',
    requiredFor: ['International', 'Enterprise'],
    automatable: 75,
    estimatedCost: '$30k-80k',
    timeline: '6-12 months',
    certRequired: true,
    selfAssessment: false,
    tools: ['Vanta', 'Drata', 'OneTrust'],
  },
];

// ============================================
// AUTOMATED COMPLIANCE TOOLS
// ============================================

export interface ComplianceTool {
  name: string;
  type: 'platform' | 'scanner' | 'monitor' | 'audit';
  pricing: string;
  frameworks: string[];
  automationLevel: 'full' | 'partial' | 'manual';
  features: string[];
  recommendation: 'essential' | 'recommended' | 'optional';
  federalReady: boolean;
}

export const COMPLIANCE_TOOLS: ComplianceTool[] = [
  // Full Automation Platforms
  {
    name: 'Vanta',
    type: 'platform',
    pricing: '$10k-25k/yr',
    frameworks: ['SOC 2', 'ISO 27001', 'HIPAA', 'GDPR', 'FedRAMP prep'],
    automationLevel: 'full',
    features: [
      'Continuous monitoring',
      'Evidence collection',
      'Policy templates',
      'Vendor management',
      'Employee training',
      'Auditor portal',
    ],
    recommendation: 'essential',
    federalReady: true,
  },
  {
    name: 'Drata',
    type: 'platform',
    pricing: '$10k-20k/yr',
    frameworks: ['SOC 2', 'ISO 27001', 'HIPAA', 'PCI DSS', 'GDPR'],
    automationLevel: 'full',
    features: [
      'Continuous compliance',
      'Risk management',
      'Trust center',
      'Integrations (100+)',
      'Custom frameworks',
    ],
    recommendation: 'recommended',
    federalReady: true,
  },
  {
    name: 'Secureframe',
    type: 'platform',
    pricing: '$10k-15k/yr',
    frameworks: ['SOC 2', 'ISO 27001', 'HIPAA', 'PCI DSS'],
    automationLevel: 'full',
    features: [
      'Automated evidence',
      'Security awareness training',
      'Vendor risk',
      'Personnel security',
    ],
    recommendation: 'optional',
    federalReady: false,
  },
  {
    name: 'Thoropass',
    type: 'platform',
    pricing: '$15k-40k/yr',
    frameworks: ['FedRAMP', 'StateRAMP', 'SOC 2', 'HIPAA'],
    automationLevel: 'full',
    features: [
      'FedRAMP-specific workflows',
      '3PAO coordination',
      'ConMon automation',
      'POA&M tracking',
    ],
    recommendation: 'recommended',
    federalReady: true,
  },

  // Free Security Scanners
  {
    name: 'Semgrep',
    type: 'scanner',
    pricing: 'Free (OSS) / $40-400/mo (Team)',
    frameworks: ['OWASP Top 10', 'CWE'],
    automationLevel: 'full',
    features: [
      'SAST',
      'Custom rules',
      'CI/CD integration',
      'Supply chain',
    ],
    recommendation: 'essential',
    federalReady: true,
  },
  {
    name: 'Trivy',
    type: 'scanner',
    pricing: 'Free (OSS)',
    frameworks: ['CIS Benchmarks', 'CVE'],
    automationLevel: 'full',
    features: [
      'Container scanning',
      'Filesystem scanning',
      'Kubernetes',
      'SBOM generation',
    ],
    recommendation: 'essential',
    federalReady: true,
  },
  {
    name: 'OWASP ZAP',
    type: 'scanner',
    pricing: 'Free (OSS)',
    frameworks: ['OWASP Top 10'],
    automationLevel: 'partial',
    features: [
      'DAST',
      'API scanning',
      'Automated scans',
      'CI/CD integration',
    ],
    recommendation: 'recommended',
    federalReady: true,
  },
  {
    name: 'GitHub Advanced Security',
    type: 'scanner',
    pricing: 'Free (public) / $49/user/mo (GHAS)',
    frameworks: ['CWE', 'CVE'],
    automationLevel: 'full',
    features: [
      'CodeQL SAST',
      'Secret scanning',
      'Dependabot',
      'Security advisories',
    ],
    recommendation: 'recommended',
    federalReady: true,
  },
  {
    name: 'ScoutSuite',
    type: 'scanner',
    pricing: 'Free (OSS)',
    frameworks: ['CIS Benchmarks', 'Custom'],
    automationLevel: 'full',
    features: [
      'AWS audit',
      'GCP audit',
      'Azure audit',
      'Multi-cloud',
    ],
    recommendation: 'recommended',
    federalReady: true,
  },

  // Monitoring
  {
    name: 'Wazuh',
    type: 'monitor',
    pricing: 'Free (OSS) / $1k-10k/yr (Cloud)',
    frameworks: ['PCI DSS', 'HIPAA', 'GDPR', 'NIST 800-53'],
    automationLevel: 'full',
    features: [
      'SIEM',
      'XDR',
      'Vulnerability detection',
      'Compliance dashboards',
    ],
    recommendation: 'optional',
    federalReady: true,
  },
];

// ============================================
// YOUR COMPLIANCE ROADMAP
// ============================================

export const YOUR_ROADMAP = {
  // Phase 1: Foundation (You're here)
  phase1: {
    name: 'Security Foundation',
    timeline: 'Now',
    cost: '$0-500',
    tasks: [
      'Semgrep in CI/CD (done)',
      'Fix Supabase security findings (in progress)',
      'Implement security headers (done)',
      'Enable audit logging (done)',
      'Document security policies',
    ],
    outcome: 'Basic security posture documented',
  },

  // Phase 2: Automated Compliance
  phase2: {
    name: 'Compliance Automation',
    timeline: '1-2 months',
    cost: '$0-2k',
    tasks: [
      'Set up Vanta/Drata free trial',
      'Connect GitHub, Vercel, Supabase integrations',
      'Generate initial compliance report',
      'Identify gaps',
      'Create remediation plan',
    ],
    outcome: 'Compliance baseline established',
  },

  // Phase 3: SOC 2 Type II (Enterprise sales enabler)
  phase3: {
    name: 'SOC 2 Type II',
    timeline: '6-9 months',
    cost: '$20k-40k',
    tasks: [
      'Engage compliance platform (Vanta recommended)',
      'Complete readiness assessment',
      'Remediate gaps',
      'Begin 6-month observation period',
      'Engage auditor (CPA firm)',
      'Complete audit',
    ],
    outcome: 'SOC 2 Type II certified',
  },

  // Phase 4: Federal (DoD/DHS sales)
  phase4: {
    name: 'Federal Readiness',
    timeline: '12-18 months',
    cost: '$50k-250k',
    tasks: [
      'Complete NIST 800-171 self-assessment',
      'Register in SPRS',
      'Consider FedRAMP Tailored (if SaaS)',
      'Or: Pursue CMMC Level 2 (if DoD)',
      'Engage 3PAO/C3PAO for assessment',
    ],
    outcome: 'Federal contract eligible',
  },
};

// ============================================
// SELF-ASSESSMENT CHECKLIST
// ============================================

export interface ComplianceControl {
  id: string;
  family: string;
  control: string;
  description: string;
  implemented: boolean;
  automatable: boolean;
  evidenceType: string;
  yourStatus: 'complete' | 'partial' | 'missing';
}

export const NIST_800_171_CONTROLS: ComplianceControl[] = [
  // Access Control Family
  {
    id: '3.1.1',
    family: 'Access Control',
    control: 'Limit system access',
    description: 'Limit information system access to authorized users',
    implemented: true,
    automatable: true,
    evidenceType: 'Supabase RLS policies',
    yourStatus: 'complete',
  },
  {
    id: '3.1.2',
    family: 'Access Control',
    control: 'Limit system access by functions',
    description: 'Limit access to types of transactions and functions',
    implemented: true,
    automatable: true,
    evidenceType: 'RBAC in SecOpsConfig.ts',
    yourStatus: 'complete',
  },
  {
    id: '3.1.3',
    family: 'Access Control',
    control: 'Control CUI flow',
    description: 'Control the flow of CUI in accordance with approved authorizations',
    implemented: true,
    automatable: true,
    evidenceType: 'Data classification in SecOpsConfig.ts',
    yourStatus: 'complete',
  },
  {
    id: '3.1.5',
    family: 'Access Control',
    control: 'Least privilege',
    description: 'Employ the principle of least privilege',
    implemented: true,
    automatable: true,
    evidenceType: 'Tiered access model',
    yourStatus: 'complete',
  },
  {
    id: '3.1.8',
    family: 'Access Control',
    control: 'Unsuccessful logon attempts',
    description: 'Limit unsuccessful logon attempts',
    implemented: true,
    automatable: true,
    evidenceType: 'Rate limiting in SecOpsConfig.ts',
    yourStatus: 'complete',
  },

  // Audit & Accountability
  {
    id: '3.3.1',
    family: 'Audit',
    control: 'Audit events',
    description: 'Create and retain system audit logs',
    implemented: true,
    automatable: true,
    evidenceType: 'AuditLogger.ts implementation',
    yourStatus: 'complete',
  },
  {
    id: '3.3.2',
    family: 'Audit',
    control: 'User accountability',
    description: 'Ensure actions can be traced to users',
    implemented: true,
    automatable: true,
    evidenceType: 'Actor tracking in audit logs',
    yourStatus: 'complete',
  },

  // Configuration Management
  {
    id: '3.4.1',
    family: 'Configuration',
    control: 'Baseline configurations',
    description: 'Establish and maintain baseline configurations',
    implemented: true,
    automatable: true,
    evidenceType: 'Infrastructure as code',
    yourStatus: 'partial',
  },
  {
    id: '3.4.5',
    family: 'Configuration',
    control: 'Access restrictions for change',
    description: 'Define and enforce access restrictions for change',
    implemented: true,
    automatable: true,
    evidenceType: 'GitHub branch protection',
    yourStatus: 'complete',
  },

  // Identification & Authentication
  {
    id: '3.5.1',
    family: 'Authentication',
    control: 'Identify users',
    description: 'Identify system users and processes',
    implemented: true,
    automatable: true,
    evidenceType: 'Supabase Auth',
    yourStatus: 'complete',
  },
  {
    id: '3.5.3',
    family: 'Authentication',
    control: 'Multi-factor authentication',
    description: 'Use MFA for local and network access',
    implemented: false,
    automatable: true,
    evidenceType: 'Supabase MFA (needs enabling)',
    yourStatus: 'missing',
  },

  // System & Communications Protection
  {
    id: '3.13.1',
    family: 'Communications',
    control: 'Boundary protection',
    description: 'Monitor and control communications at boundaries',
    implemented: true,
    automatable: true,
    evidenceType: 'Vercel Edge Network',
    yourStatus: 'complete',
  },
  {
    id: '3.13.8',
    family: 'Communications',
    control: 'Cryptographic protection',
    description: 'Implement cryptographic mechanisms for CUI',
    implemented: true,
    automatable: true,
    evidenceType: 'TLS 1.3, AES-256',
    yourStatus: 'complete',
  },

  // System & Information Integrity
  {
    id: '3.14.1',
    family: 'Integrity',
    control: 'Flaw remediation',
    description: 'Identify and correct system flaws in a timely manner',
    implemented: true,
    automatable: true,
    evidenceType: 'Dependabot, Semgrep',
    yourStatus: 'complete',
  },
  {
    id: '3.14.2',
    family: 'Integrity',
    control: 'Malicious code protection',
    description: 'Provide protection from malicious code',
    implemented: true,
    automatable: true,
    evidenceType: 'SAST in CI/CD',
    yourStatus: 'complete',
  },
];

// ============================================
// QUICK COMPLIANCE SCORE
// ============================================

export function calculateComplianceScore(controls: ComplianceControl[]): {
  score: number;
  complete: number;
  partial: number;
  missing: number;
  readiness: string;
} {
  const complete = controls.filter(c => c.yourStatus === 'complete').length;
  const partial = controls.filter(c => c.yourStatus === 'partial').length;
  const missing = controls.filter(c => c.yourStatus === 'missing').length;
  const total = controls.length;

  const score = Math.round(((complete + partial * 0.5) / total) * 100);

  let readiness: string;
  if (score >= 90) {
    readiness = 'Ready for assessment';
  } else if (score >= 70) {
    readiness = 'Minor gaps - address before assessment';
  } else if (score >= 50) {
    readiness = 'Significant gaps - remediation needed';
  } else {
    readiness = 'Major work required';
  }

  return { score, complete, partial, missing, readiness };
}

// Calculate your current score
const yourScore = calculateComplianceScore(NIST_800_171_CONTROLS);
// Result: ~90% - Ready for assessment (once MFA enabled)

// ============================================
// BOTTOM LINE
// ============================================

/**
 * RECOMMENDATION FOR YOUR SITUATION:
 *
 * 1. DON'T RENEW CERTS - Use automated platforms instead
 *    - Security+, CISSP, CEH are nice-to-have for credibility
 *    - Not required for compliance if you use automation
 *    - Only renew if a specific contract requires them
 *
 * 2. START WITH VANTA/DRATA FREE TRIAL
 *    - Connect your GitHub, Vercel, Supabase
 *    - See your current compliance posture
 *    - Get automated evidence collection
 *
 * 3. FOR FEDERAL SALES:
 *    - Start with NIST 800-171 self-assessment (free)
 *    - Register in SPRS (required for DoD)
 *    - SOC 2 opens enterprise doors
 *    - FedRAMP is the golden ticket but expensive
 *
 * 4. YOUR CURRENT STATUS:
 *    - Security controls: ~85% implemented
 *    - Missing: MFA enforcement, formal policies
 *    - Timeline to SOC 2: 6-9 months
 *    - Timeline to FedRAMP Tailored: 9-12 months
 *
 * 5. ESTIMATED COSTS:
 *    - DIY with automation: $15-30k for SOC 2
 *    - Federal (FedRAMP Tailored): $50-150k
 *    - Full FedRAMP Moderate: $250k-500k
 */
