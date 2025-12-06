/**
 * Customer Vetting Protocol
 *
 * Anti-Complaint Spec Section 5.2 Implementation:
 * - Entity verification
 * - Use case disclosure
 * - Red flag screening (sanctions, human rights)
 * - Annual recertification
 *
 * Rejection triggers: Authoritarian government agencies, companies with
 * documented labor/human rights violations, entities under sanctions.
 */

// =============================================================================
// TYPES
// =============================================================================

export interface CustomerProfile {
  id: string;
  organizationName: string;
  legalEntityName: string;
  jurisdiction: string;
  registrationNumber?: string;

  // Classification
  entityType: EntityType;
  sector: string;
  subsector?: string;

  // Contacts
  primaryContact: ContactInfo;
  legalContact?: ContactInfo;
  technicalContact?: ContactInfo;

  // Verification status
  verificationStatus: VerificationStatus;
  verifiedAt?: Date;
  verifiedBy?: string;

  // Use case
  disclosedUseCases: DisclosedUseCase[];
  prohibitedUseAcknowledged: boolean;

  // Screening results
  screeningResults: ScreeningResult[];
  lastScreenedAt?: Date;

  // Certification
  certificationHistory: CertificationRecord[];
  nextRecertificationDue?: Date;

  // Risk score
  riskScore: number; // 0-100, higher = more risk
  riskFactors: RiskFactor[];
}

export type EntityType =
  | 'corporation'
  | 'government_agency'
  | 'non_profit'
  | 'educational'
  | 'military'
  | 'law_enforcement'
  | 'intelligence_agency'
  | 'individual'
  | 'other';

export interface ContactInfo {
  name: string;
  email: string;
  phone?: string;
  title?: string;
}

export type VerificationStatus =
  | 'pending'
  | 'in_review'
  | 'verified'
  | 'rejected'
  | 'suspended';

export interface DisclosedUseCase {
  id: string;
  description: string;
  category: UseCaseCategory;
  targetPopulations?: string[];
  dataSourcesRequired?: string[];
  approvalStatus: 'pending' | 'approved' | 'rejected' | 'requires_review';
  reviewNotes?: string;
}

export type UseCaseCategory =
  | 'threat_intelligence'
  | 'brand_protection'
  | 'executive_protection'
  | 'supply_chain_risk'
  | 'geopolitical_analysis'
  | 'competitive_intelligence'
  | 'fraud_detection'
  | 'regulatory_compliance'
  | 'research_academic'
  | 'law_enforcement'
  | 'national_security'
  | 'other';

export interface ScreeningResult {
  id: string;
  screeningType: ScreeningType;
  screenedAt: Date;
  source: string;
  matched: boolean;
  matchDetails?: string;
  riskLevel: 'none' | 'low' | 'medium' | 'high' | 'critical';
  requiresReview: boolean;
}

export type ScreeningType =
  | 'ofac_sanctions'
  | 'un_sanctions'
  | 'eu_sanctions'
  | 'uk_sanctions'
  | 'pep_list'
  | 'human_rights_watch'
  | 'amnesty_international'
  | 'press_freedom_index'
  | 'corruption_perception'
  | 'democracy_index'
  | 'adverse_media';

export interface CertificationRecord {
  id: string;
  certifiedAt: Date;
  expiresAt: Date;
  certifiedBy: string;
  useCasesApproved: string[];
  conditions?: string[];
  revoked: boolean;
  revokedAt?: Date;
  revocationReason?: string;
}

export interface RiskFactor {
  factor: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  mitigatable: boolean;
  mitigationSteps?: string[];
}

// =============================================================================
// PROHIBITED USE CASES (HARDCODED PER SPEC)
// =============================================================================

export const PROHIBITED_APPLICATIONS = [
  'mass_surveillance_of_protected_groups',
  'warrantless_location_tracking',
  'predictive_policing_on_individuals',
  'immigration_enforcement_targeting',
  'protest_monitoring',
  'union_organizing_surveillance',
  'journalist_source_identification',
  'political_opposition_research',
] as const;

export type ProhibitedApplication = typeof PROHIBITED_APPLICATIONS[number];

export const PROHIBITED_APPLICATION_DESCRIPTIONS: Record<ProhibitedApplication, string> = {
  mass_surveillance_of_protected_groups:
    'Bulk monitoring of individuals based on race, religion, ethnicity, political affiliation, or other protected characteristics',
  warrantless_location_tracking:
    'Tracking physical locations of individuals without valid legal authorization',
  predictive_policing_on_individuals:
    'Using data to predict criminal behavior of specific individuals before any crime occurs',
  immigration_enforcement_targeting:
    'Identifying individuals for immigration enforcement based on ethnicity or national origin',
  protest_monitoring:
    'Surveillance of lawful protest activities or identification of protest participants',
  union_organizing_surveillance:
    'Monitoring labor organizing activities or identifying union organizers',
  journalist_source_identification:
    'Attempting to identify confidential sources of journalists',
  political_opposition_research:
    'Surveillance of political opponents or opposition research beyond public records',
};

// =============================================================================
// HIGH-RISK JURISDICTIONS
// =============================================================================

export const HIGH_RISK_JURISDICTIONS = [
  // Based on Freedom House "Not Free" ratings + additional criteria
  'AF', // Afghanistan
  'BY', // Belarus
  'CN', // China
  'CU', // Cuba
  'ER', // Eritrea
  'IR', // Iran
  'KP', // North Korea
  'RU', // Russia
  'SA', // Saudi Arabia
  'SY', // Syria
  'TM', // Turkmenistan
  'VE', // Venezuela
] as const;

// =============================================================================
// VETTING ENGINE
// =============================================================================

export class CustomerVettingEngine {
  /**
   * Perform comprehensive customer screening.
   */
  async screenCustomer(profile: Partial<CustomerProfile>): Promise<{
    approved: boolean;
    riskScore: number;
    riskFactors: RiskFactor[];
    screeningResults: ScreeningResult[];
    requiresManualReview: boolean;
    rejectionReasons: string[];
  }> {
    const riskFactors: RiskFactor[] = [];
    const screeningResults: ScreeningResult[] = [];
    const rejectionReasons: string[] = [];

    // Check entity type
    if (this.isHighRiskEntityType(profile.entityType)) {
      riskFactors.push({
        factor: 'high_risk_entity_type',
        severity: 'high',
        description: `Entity type "${profile.entityType}" requires enhanced due diligence`,
        mitigatable: true,
        mitigationSteps: ['Obtain official use case documentation', 'Legal review required'],
      });
    }

    // Check jurisdiction
    if (profile.jurisdiction && this.isHighRiskJurisdiction(profile.jurisdiction)) {
      riskFactors.push({
        factor: 'high_risk_jurisdiction',
        severity: 'critical',
        description: `Jurisdiction "${profile.jurisdiction}" is designated high-risk`,
        mitigatable: false,
      });
      rejectionReasons.push(`Jurisdiction ${profile.jurisdiction} is on prohibited list`);
    }

    // Simulate sanctions screening
    const sanctionsResult = await this.screenSanctions(profile);
    screeningResults.push(sanctionsResult);
    if (sanctionsResult.matched) {
      riskFactors.push({
        factor: 'sanctions_match',
        severity: 'critical',
        description: sanctionsResult.matchDetails || 'Entity appears on sanctions list',
        mitigatable: false,
      });
      rejectionReasons.push('Entity or related parties appear on sanctions lists');
    }

    // Check disclosed use cases
    for (const useCase of profile.disclosedUseCases || []) {
      const violation = this.checkProhibitedUseCase(useCase);
      if (violation) {
        riskFactors.push({
          factor: 'prohibited_use_case',
          severity: 'critical',
          description: `Use case violates prohibition: ${violation}`,
          mitigatable: false,
        });
        rejectionReasons.push(`Disclosed use case matches prohibition: ${violation}`);
      }
    }

    // Calculate overall risk score
    const riskScore = this.calculateRiskScore(riskFactors);

    // Determine approval
    const hasCriticalRisk = riskFactors.some((f) => f.severity === 'critical' && !f.mitigatable);
    const approved = !hasCriticalRisk && rejectionReasons.length === 0;
    const requiresManualReview = riskScore > 30 || riskFactors.some((f) => f.severity === 'high');

    return {
      approved,
      riskScore,
      riskFactors,
      screeningResults,
      requiresManualReview,
      rejectionReasons,
    };
  }

  /**
   * Validate a use case against prohibited applications.
   */
  checkProhibitedUseCase(useCase: DisclosedUseCase): ProhibitedApplication | null {
    const description = useCase.description.toLowerCase();
    const targets = (useCase.targetPopulations || []).map((t) => t.toLowerCase());

    // Check for surveillance of protected groups
    const protectedGroupKeywords = [
      'ethnic', 'racial', 'religious', 'muslim', 'christian', 'jewish',
      'lgbtq', 'immigrant', 'refugee', 'minority',
    ];
    const surveillanceKeywords = ['monitor', 'track', 'surveil', 'identify', 'target'];

    const mentionsProtectedGroup = protectedGroupKeywords.some(
      (k) => description.includes(k) || targets.some((t) => t.includes(k))
    );
    const mentionsSurveillance = surveillanceKeywords.some((k) => description.includes(k));

    if (mentionsProtectedGroup && mentionsSurveillance) {
      return 'mass_surveillance_of_protected_groups';
    }

    // Check for location tracking
    if (description.includes('location') && description.includes('track')) {
      if (!description.includes('warrant') && !description.includes('consent')) {
        return 'warrantless_location_tracking';
      }
    }

    // Check for predictive policing
    if (
      description.includes('predict') &&
      (description.includes('crime') || description.includes('criminal'))
    ) {
      return 'predictive_policing_on_individuals';
    }

    // Check for protest monitoring
    if (description.includes('protest') || description.includes('demonstration')) {
      return 'protest_monitoring';
    }

    // Check for union surveillance
    if (description.includes('union') || description.includes('labor organiz')) {
      return 'union_organizing_surveillance';
    }

    // Check for journalist source identification
    if (description.includes('journalist') && description.includes('source')) {
      return 'journalist_source_identification';
    }

    return null;
  }

  /**
   * Screen against sanctions lists.
   */
  private async screenSanctions(profile: Partial<CustomerProfile>): Promise<ScreeningResult> {
    // In production, this would call actual sanctions screening APIs
    // (OFAC, UN, EU, UK, etc.)

    const entityName = profile.legalEntityName || profile.organizationName || '';

    // Simulate screening
    const result: ScreeningResult = {
      id: `scr_${Date.now()}`,
      screeningType: 'ofac_sanctions',
      screenedAt: new Date(),
      source: 'OFAC SDN List',
      matched: false,
      riskLevel: 'none',
      requiresReview: false,
    };

    // Demo: Flag certain patterns
    const riskyPatterns = ['defense ministry', 'state security', 'revolutionary guard'];
    if (riskyPatterns.some((p) => entityName.toLowerCase().includes(p))) {
      result.matched = true;
      result.matchDetails = `Entity name matches high-risk pattern`;
      result.riskLevel = 'critical';
      result.requiresReview = true;
    }

    return result;
  }

  private isHighRiskEntityType(entityType?: EntityType): boolean {
    const highRiskTypes: EntityType[] = [
      'government_agency',
      'military',
      'law_enforcement',
      'intelligence_agency',
    ];
    return entityType ? highRiskTypes.includes(entityType) : false;
  }

  private isHighRiskJurisdiction(jurisdiction: string): boolean {
    return HIGH_RISK_JURISDICTIONS.includes(
      jurisdiction.toUpperCase() as (typeof HIGH_RISK_JURISDICTIONS)[number]
    );
  }

  private calculateRiskScore(factors: RiskFactor[]): number {
    let score = 0;

    for (const factor of factors) {
      switch (factor.severity) {
        case 'critical':
          score += factor.mitigatable ? 30 : 50;
          break;
        case 'high':
          score += 20;
          break;
        case 'medium':
          score += 10;
          break;
        case 'low':
          score += 5;
          break;
      }
    }

    return Math.min(100, score);
  }

  /**
   * Check if customer needs recertification.
   */
  needsRecertification(profile: CustomerProfile): boolean {
    if (!profile.nextRecertificationDue) return true;
    return new Date() >= profile.nextRecertificationDue;
  }

  /**
   * Create recertification record.
   */
  createCertification(
    profile: CustomerProfile,
    certifiedBy: string,
    validityDays: number = 365
  ): CertificationRecord {
    const now = new Date();
    const expiresAt = new Date(now.getTime() + validityDays * 24 * 60 * 60 * 1000);

    return {
      id: `cert_${Date.now()}`,
      certifiedAt: now,
      expiresAt,
      certifiedBy,
      useCasesApproved: profile.disclosedUseCases
        .filter((u) => u.approvalStatus === 'approved')
        .map((u) => u.id),
      revoked: false,
    };
  }
}

// =============================================================================
// EXPORTS
// =============================================================================

export const vettingEngine = new CustomerVettingEngine();

export function createCustomerProfile(
  orgName: string,
  jurisdiction: string,
  entityType: EntityType,
  primaryContact: ContactInfo
): Partial<CustomerProfile> {
  return {
    id: `cust_${Date.now()}`,
    organizationName: orgName,
    legalEntityName: orgName,
    jurisdiction,
    entityType,
    primaryContact,
    verificationStatus: 'pending',
    disclosedUseCases: [],
    prohibitedUseAcknowledged: false,
    screeningResults: [],
    certificationHistory: [],
    riskScore: 0,
    riskFactors: [],
  };
}
