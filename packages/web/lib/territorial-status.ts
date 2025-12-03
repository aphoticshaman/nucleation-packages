/**
 * Territorial Status System - Blue Force Tracker Style
 *
 * Handles complex sovereignty vs control situations like:
 * - Crimea (legally Ukraine, occupied by Russia, contested)
 * - Iraq/Afghanistan during US occupation
 * - Kashmir (disputed between India/Pakistan/China)
 * - Taiwan (disputed sovereignty, de facto independent)
 *
 * Provides temporal tracking for historical analysis and
 * military asset overlay capabilities.
 */

// ============================================
// SOVEREIGNTY & CONTROL TYPES
// ============================================

export type SovereigntyStatus =
  | 'recognized'           // Internationally recognized
  | 'disputed'             // Multiple claimants
  | 'unrecognized'         // De facto state, not recognized (e.g., Somaliland)
  | 'annexed'              // Claimed annexed by occupier
  | 'occupied'             // Under military occupation
  | 'administered'         // UN/international administration
  | 'protectorate'         // Under protection of another state
  | 'mandate'              // Historical mandate territory
  | 'colonial';            // Colonial possession

export type ControlStatus =
  | 'full'                 // Complete governmental control
  | 'partial'              // Controls only portion of claimed territory
  | 'nominal'              // Legal claim but no effective control
  | 'contested'            // Active fighting over control
  | 'occupied'             // Under foreign military occupation
  | 'insurgent'            // Significant insurgent presence
  | 'failed'               // State collapse, no effective authority
  | 'transitional';        // Government in transition

export type DisputeType =
  | 'territorial'          // Border/territorial dispute
  | 'sovereignty'          // Who has sovereignty
  | 'recognition'          // Recognition dispute
  | 'maritime'             // Maritime boundary
  | 'airspace'             // Airspace claims
  | 'resource'             // Resource extraction rights
  | 'ethnic'               // Ethnic/separatist
  | 'historical';          // Historical claims

// ============================================
// GOVERNMENT & TRANSITION TYPES
// ============================================

export type GovernmentType =
  | 'democracy'            // Electoral democracy
  | 'authoritarian'        // Single-party/strongman
  | 'monarchy'             // Traditional monarchy
  | 'constitutional_monarchy'
  | 'theocracy'            // Religious rule
  | 'military_junta'       // Military government
  | 'transitional'         // Transitional government
  | 'occupation_admin'     // Occupation administration
  | 'failed_state'         // No functioning government
  | 'communist'            // Communist single-party
  | 'hybrid';              // Mixed/unclear

export type TransitionType =
  | 'election'             // Electoral transition
  | 'coup'                 // Military coup
  | 'revolution'           // Popular revolution
  | 'invasion'             // Foreign invasion
  | 'annexation'           // Territorial annexation
  | 'independence'         // Independence declaration
  | 'unification'          // State unification
  | 'partition'            // State partition
  | 'collapse'             // State collapse
  | 'restoration'          // Government restoration
  | 'negotiated';          // Negotiated transition

export type TransitionStage =
  | 'pre_transition'       // Signs of coming change
  | 'active'               // Transition underway
  | 'consolidating'        // New regime consolidating
  | 'stable'               // Transition complete, stable
  | 'regressing'           // Backsliding
  | 'contested';           // Disputed legitimacy

// ============================================
// BORDER STATUS & FLOW TYPES
// ============================================

export type BorderDomain =
  | 'land'                 // Terrestrial border
  | 'maritime'             // Sea/ocean boundary
  | 'riverine'             // River border
  | 'lacustrine'           // Lake border
  | 'airspace'             // Airspace boundary (ADIZ, FIR)
  | 'cyberspace'           // Cyber jurisdiction boundary
  | 'space';               // Orbital/space jurisdiction

export type BorderLegalStatus =
  | 'open'                 // Free movement (e.g., EU Schengen)
  | 'visa_required'        // Legal crossing requires documentation
  | 'restricted'           // Heavily restricted, limited crossings
  | 'closed'               // Officially closed
  | 'militarized'          // Heavily fortified/militarized
  | 'disputed'             // Border line itself disputed
  | 'demarcated'           // Clearly marked and recognized
  | 'undemarcated';        // No clear physical demarcation

export type MaritimeZoneType =
  | 'territorial_sea'      // 12nm from baseline
  | 'contiguous_zone'      // 24nm - customs/immigration enforcement
  | 'eez'                  // 200nm - Exclusive Economic Zone
  | 'continental_shelf'    // Extended continental shelf claims
  | 'high_seas'            // International waters
  | 'archipelagic'         // Archipelagic waters
  | 'historic_bay'         // Historic bay claims
  | 'disputed';            // Overlapping claims

export type AirspaceType =
  | 'sovereign'            // National airspace (above territorial land/sea)
  | 'fir'                  // Flight Information Region
  | 'adiz'                 // Air Defense Identification Zone
  | 'no_fly'               // Prohibited airspace
  | 'restricted'           // Restricted airspace
  | 'international';       // International airspace

export type BorderEnforcementLevel =
  | 'none'                 // No enforcement (collapsed state)
  | 'minimal'              // Token presence only
  | 'porous'               // Easily circumvented (US-Mexico historical)
  | 'moderate'             // Some enforcement, gaps exist
  | 'strict'               // Actively patrolled, crossings monitored
  | 'heavy'                // Major resources deployed
  | 'militarized'          // Military-level enforcement
  | 'shoot_to_kill';       // Lethal force authorized (e.g., Cold War Berlin)

export type BorderCrossingType =
  | 'official_port'        // Official port of entry
  | 'checkpoint'           // Military/police checkpoint
  | 'informal'             // Known informal crossing point
  | 'smuggling_route'      // Known smuggling route
  | 'sealed';              // No crossing possible

export type FlowType =
  | 'people'               // Human migration/travel
  | 'goods_legal'          // Legal trade
  | 'goods_contraband'     // Smuggled goods
  | 'narcotics'            // Drug trafficking
  | 'weapons'              // Arms trafficking
  | 'refugees'             // Refugee flows
  | 'military'             // Military movement
  | 'information';         // Information/signals

export type ExistentialThreatLevel =
  | 'none'                 // No existential threat
  | 'latent'               // Theoretical threat exists
  | 'potential'            // Threat could materialize
  | 'elevated'             // Increased threat activity
  | 'active'               // Active existential threat
  | 'imminent'             // Attack likely imminent
  | 'ongoing';             // Under active existential attack

// Conditional scenario analysis for realistic threat assessment
export interface ThreatScenario {
  id: string;
  name: string;
  description: string;
  probability: number;            // 0-1 baseline probability
  timeframe: string;              // e.g., "5 years", "10 years", "indefinite"

  // Conditions that must be true for scenario to occur
  requiredConditions: {
    condition: string;
    currentStatus: boolean;
    probabilityIfTrue: number;    // How this affects overall probability
  }[];

  // Conditions that would prevent scenario
  blockingConditions: {
    condition: string;
    currentStatus: boolean;
    blockingStrength: number;     // 0-1, how much this prevents scenario
  }[];

  // Dependencies on other actors
  actorDependencies: {
    actor: string;                // e.g., "US", "ASEAN", "Japan"
    requiredAction: string;       // What they must do/not do
    likelihoodOfAction: number;   // 0-1
  }[];

  // Outcome if scenario occurs
  outcomes: {
    outcome: string;
    probability: number;
  }[];
}

export interface ConditionalThreatAssessment {
  baseScenario: 'status_quo' | 'escalation' | 'de_escalation' | 'resolution';

  // Realistic assessment considering actor behavior
  scenarios: ThreatScenario[];

  // Key dependencies
  criticalDependencies: {
    factor: string;
    currentState: string;
    trendDirection: 'improving' | 'stable' | 'deteriorating';
    impact: 'enabling' | 'blocking' | 'neutral';
  }[];

  // Time-based probability curves
  probabilityOverTime: {
    years: number;
    baselineProbability: number;
    withUSDistraction: number;
    withPowerShift: number;
    withAllianceCollapse: number;
  }[];

  analystNotes: string;
  lastUpdated: string;
}

// Taiwan - Detailed Conditional Threat Assessment
export const TAIWAN_THREAT_ASSESSMENT: ConditionalThreatAssessment = {
  baseScenario: 'status_quo',

  scenarios: [
    {
      id: 'status-quo-indefinite',
      name: 'Indefinite Status Quo',
      description: 'No military action, continued ambiguity. Most likely near-term scenario.',
      probability: 0.65,
      timeframe: '5 years',
      requiredConditions: [
        { condition: 'US maintains credible deterrence posture', currentStatus: true, probabilityIfTrue: 0.8 },
        { condition: 'Taiwan avoids formal independence declaration', currentStatus: true, probabilityIfTrue: 0.9 },
        { condition: 'PRC economic growth continues', currentStatus: true, probabilityIfTrue: 0.7 },
      ],
      blockingConditions: [
        { condition: 'US-China direct military conflict elsewhere', currentStatus: false, blockingStrength: 0.9 },
        { condition: 'Major PRC domestic crisis requiring nationalist rally', currentStatus: false, blockingStrength: 0.8 },
      ],
      actorDependencies: [
        { actor: 'US', requiredAction: 'Maintain strategic ambiguity and deterrence', likelihoodOfAction: 0.75 },
        { actor: 'TW', requiredAction: 'Maintain de facto independence without de jure declaration', likelihoodOfAction: 0.85 },
        { actor: 'CN', requiredAction: 'Continue to calculate costs > benefits', likelihoodOfAction: 0.7 },
      ],
      outcomes: [
        { outcome: 'Continued cross-strait tension but no conflict', probability: 0.65 },
        { outcome: 'Gradual economic decoupling', probability: 0.25 },
        { outcome: 'Unexpected escalation trigger', probability: 0.1 },
      ],
    },
    {
      id: 'us-distraction-window',
      name: 'US Distraction Window Attack',
      description: 'PRC attacks while US is committed to major conflict elsewhere (Middle East, Europe, Korea)',
      probability: 0.12,
      timeframe: '10 years',
      requiredConditions: [
        { condition: 'US engaged in major conflict requiring Pacific assets', currentStatus: false, probabilityIfTrue: 0.4 },
        { condition: 'PRC military modernization reaches threshold capability', currentStatus: false, probabilityIfTrue: 0.6 },
        { condition: 'PRC leadership calculates success probability > 70%', currentStatus: false, probabilityIfTrue: 0.3 },
      ],
      blockingConditions: [
        { condition: 'Japan commits to Taiwan defense regardless of US', currentStatus: false, blockingStrength: 0.5 },
        { condition: 'Nuclear deterrence extended explicitly', currentStatus: false, blockingStrength: 0.7 },
        { condition: 'Semiconductor supply chain fully diversified', currentStatus: false, blockingStrength: 0.2 },
      ],
      actorDependencies: [
        { actor: 'US', requiredAction: 'Be substantially committed elsewhere', likelihoodOfAction: 0.3 },
        { actor: 'JP', requiredAction: 'Not independently commit to Taiwan defense', likelihoodOfAction: 0.5 },
        { actor: 'ASEAN', requiredAction: 'Remain neutral / not support US basing', likelihoodOfAction: 0.6 },
      ],
      outcomes: [
        { outcome: 'Successful blockade leading to negotiated absorption', probability: 0.35 },
        { outcome: 'Failed invasion with massive casualties', probability: 0.25 },
        { outcome: 'US intervention despite distraction, escalation', probability: 0.3 },
        { outcome: 'Nuclear threshold crossed', probability: 0.1 },
      ],
    },
    {
      id: 'power-shift-inevitability',
      name: 'Overwhelming Power Accumulation',
      description: 'PRC builds such overwhelming conventional superiority that US/allies calculate intervention futile',
      probability: 0.15,
      timeframe: '15-20 years',
      requiredConditions: [
        { condition: 'PRC navy exceeds US Pacific capability', currentStatus: false, probabilityIfTrue: 0.5 },
        { condition: 'A2/AD bubble makes intervention cost-prohibitive', currentStatus: false, probabilityIfTrue: 0.6 },
        { condition: 'Hypersonic/AI warfare gaps favor PRC', currentStatus: false, probabilityIfTrue: 0.4 },
        { condition: 'US domestic consensus against intervention crystallizes', currentStatus: false, probabilityIfTrue: 0.35 },
      ],
      blockingConditions: [
        { condition: 'US maintains technological edge through AUKUS etc', currentStatus: true, blockingStrength: 0.5 },
        { condition: 'Taiwan asymmetric deterrence sufficient', currentStatus: false, blockingStrength: 0.4 },
        { condition: 'PRC economic stagnation limits military spending', currentStatus: false, blockingStrength: 0.6 },
      ],
      actorDependencies: [
        { actor: 'US', requiredAction: 'Decline to intervene due to cost calculation', likelihoodOfAction: 0.25 },
        { actor: 'JP', requiredAction: 'Not independently oppose', likelihoodOfAction: 0.4 },
        { actor: 'AU', requiredAction: 'Not support US operations', likelihoodOfAction: 0.3 },
        { actor: 'ASEAN', requiredAction: 'Bandwagon with PRC', likelihoodOfAction: 0.5 },
      ],
      outcomes: [
        { outcome: 'Negotiated "peaceful reunification" under duress', probability: 0.5 },
        { outcome: 'Invasion with minimal resistance', probability: 0.3 },
        { outcome: 'Last-minute US intervention anyway', probability: 0.15 },
        { outcome: 'Taiwan nuclear breakout attempt', probability: 0.05 },
      ],
    },
    {
      id: 'alliance-collapse',
      name: 'Alliance System Collapse',
      description: 'US alliance system fragments, regional actors refuse to support Taiwan defense',
      probability: 0.08,
      timeframe: '10 years',
      requiredConditions: [
        { condition: 'US withdraws from or weakens regional commitments', currentStatus: false, probabilityIfTrue: 0.3 },
        { condition: 'Japan-US alliance significantly degraded', currentStatus: false, probabilityIfTrue: 0.2 },
        { condition: 'Philippines abrogates MDT or denies basing', currentStatus: false, probabilityIfTrue: 0.25 },
        { condition: 'Australia/UK decline AUKUS commitments', currentStatus: false, probabilityIfTrue: 0.15 },
      ],
      blockingConditions: [
        { condition: 'Bipartisan US commitment to Indo-Pacific', currentStatus: true, blockingStrength: 0.6 },
        { condition: 'Japanese remilitarization continues', currentStatus: true, blockingStrength: 0.5 },
      ],
      actorDependencies: [
        { actor: 'US', requiredAction: 'Signal non-commitment to regional defense', likelihoodOfAction: 0.2 },
        { actor: 'JP', requiredAction: 'Accommodate PRC / abandon Taiwan', likelihoodOfAction: 0.15 },
        { actor: 'PH', requiredAction: 'Deny US access, align with PRC', likelihoodOfAction: 0.3 },
      ],
      outcomes: [
        { outcome: 'PRC coercion campaign succeeds without invasion', probability: 0.4 },
        { outcome: 'Blockade/quarantine', probability: 0.35 },
        { outcome: 'Full invasion', probability: 0.2 },
        { outcome: 'Taiwan capitulates preemptively', probability: 0.05 },
      ],
    },
  ],

  criticalDependencies: [
    {
      factor: 'US Military Commitment to Indo-Pacific',
      currentState: 'Strong but stretched',
      trendDirection: 'stable',
      impact: 'blocking',
    },
    {
      factor: 'PLA Modernization Timeline',
      currentState: 'Accelerating, gaps closing',
      trendDirection: 'deteriorating',
      impact: 'enabling',
    },
    {
      factor: 'ASEAN Strategic Alignment',
      currentState: 'Hedging, economically tied to PRC',
      trendDirection: 'deteriorating',
      impact: 'enabling',
    },
    {
      factor: 'Semiconductor Supply Chain Concentration',
      currentState: 'Still Taiwan-centric (TSMC)',
      trendDirection: 'improving',
      impact: 'blocking',  // Diversification reduces leverage but increases invasion cost-benefit for PRC
    },
    {
      factor: 'Japanese Constitutional Constraints',
      currentState: 'Loosening under Kishida doctrine',
      trendDirection: 'improving',
      impact: 'blocking',
    },
    {
      factor: 'PRC Economic Trajectory',
      currentState: 'Slowing growth, property crisis',
      trendDirection: 'deteriorating',
      impact: 'neutral',  // Could enable (desperation) or block (can\'t afford)
    },
    {
      factor: 'Taiwan Asymmetric Defense Investment',
      currentState: 'Improving but insufficient',
      trendDirection: 'improving',
      impact: 'blocking',
    },
  ],

  probabilityOverTime: [
    { years: 1, baselineProbability: 0.02, withUSDistraction: 0.15, withPowerShift: 0.03, withAllianceCollapse: 0.10 },
    { years: 3, baselineProbability: 0.05, withUSDistraction: 0.25, withPowerShift: 0.08, withAllianceCollapse: 0.20 },
    { years: 5, baselineProbability: 0.08, withUSDistraction: 0.35, withPowerShift: 0.12, withAllianceCollapse: 0.30 },
    { years: 10, baselineProbability: 0.15, withUSDistraction: 0.50, withPowerShift: 0.25, withAllianceCollapse: 0.45 },
    { years: 15, baselineProbability: 0.20, withUSDistraction: 0.60, withPowerShift: 0.40, withAllianceCollapse: 0.55 },
    { years: 20, baselineProbability: 0.25, withUSDistraction: 0.70, withPowerShift: 0.55, withAllianceCollapse: 0.65 },
  ],

  analystNotes: `
REALISTIC ASSESSMENT: China will not attack Taiwan UNLESS one of the following occurs:
1. US is committed to a major conflict elsewhere that prevents Pacific intervention (Middle East escalation, European war, Korean conflict)
2. China accumulates sufficient power that US/allies calculate intervention costs > benefits
3. Alliance system collapses and Taiwan is isolated
4. Taiwan declares formal independence (self-inflicted trigger)

The status quo favors Taiwan indefinitely IF: (a) US maintains credible deterrence, (b) Taiwan maintains strategic patience, (c) regional allies continue hedging rather than fully bandwagoning with China.

Current trajectory suggests power balance shifting toward China but not yet at decision threshold. Key indicators to watch:
- PLA amphibious lift capability
- US shipbuilding vs. PLA Navy growth rates
- ASEAN basing access decisions
- Taiwan conscription/defense spending choices
- US domestic political consensus on Indo-Pacific

Time horizon for greatest risk: 2027-2035 (Xi's window of perceived opportunity)
Most likely trigger: US distracted by simultaneous crisis elsewhere
`,
  lastUpdated: '2024-12-01',
};

// ============================================
// CHINA COMPREHENSIVE TERRITORIAL DISPUTES
// ============================================

export interface TerritorialDispute {
  id: string;
  name: string;
  region: string;
  domain: BorderDomain;
  parties: {
    code: string;
    name: string;
    claimBasis: string;
    controlPercentage: number;
    militaryPresence: 'none' | 'minimal' | 'moderate' | 'heavy' | 'dominant';
  }[];
  status: 'frozen' | 'low_tension' | 'moderate_tension' | 'high_tension' | 'active_conflict';
  escalationRisk: number;         // 0-1
  strategicImportance: number;    // 0-1
  resourcesAtStake: string[];
  internationalLaw: {
    relevantTreaties: string[];
    icjRulings: string[];
    unResolutions: string[];
    status: 'clear' | 'ambiguous' | 'contested';
  };
  recentIncidents: {
    date: string;
    description: string;
    severity: 'minor' | 'moderate' | 'serious' | 'critical';
  }[];
  analystAssessment: string;
  lastUpdated: string;
}

export const CHINA_TERRITORIAL_DISPUTES: Record<string, TerritorialDispute> = {
  // South China Sea - Nine Dash Line
  'south_china_sea': {
    id: 'scs-nine-dash',
    name: 'South China Sea (Nine-Dash Line)',
    region: 'Southeast Asia',
    domain: 'maritime',
    parties: [
      {
        code: 'CN',
        name: 'China (PRC)',
        claimBasis: 'Historical "nine-dash line" claim (vague basis), 2013 ADIZ declaration',
        controlPercentage: 40,
        militaryPresence: 'dominant',
      },
      {
        code: 'VN',
        name: 'Vietnam',
        claimBasis: 'UNCLOS EEZ, historical administration of Paracels/Spratlys',
        controlPercentage: 15,
        militaryPresence: 'moderate',
      },
      {
        code: 'PH',
        name: 'Philippines',
        claimBasis: 'UNCLOS EEZ, 2016 PCA ruling in favor',
        controlPercentage: 10,
        militaryPresence: 'minimal',
      },
      {
        code: 'MY',
        name: 'Malaysia',
        claimBasis: 'UNCLOS EEZ, continental shelf claims',
        controlPercentage: 8,
        militaryPresence: 'minimal',
      },
      {
        code: 'BN',
        name: 'Brunei',
        claimBasis: 'UNCLOS EEZ',
        controlPercentage: 2,
        militaryPresence: 'none',
      },
      {
        code: 'TW',
        name: 'Taiwan (ROC)',
        claimBasis: 'Historical (same basis as PRC), Taiping Island presence',
        controlPercentage: 1,
        militaryPresence: 'minimal',
      },
    ],
    status: 'high_tension',
    escalationRisk: 0.55,
    strategicImportance: 0.95,
    resourcesAtStake: [
      '$3.4 trillion annual shipping transit',
      'Oil: 11 billion barrels (estimated)',
      'Natural gas: 190 trillion cubic feet (estimated)',
      'Fisheries: 10% of global catch',
      'Strategic sea lanes: Malacca Strait alternative',
    ],
    internationalLaw: {
      relevantTreaties: ['UNCLOS 1982', 'DOC 2002 (non-binding)'],
      icjRulings: [],
      unResolutions: [],
      status: 'contested',  // PCA ruled against China but China rejects
    },
    recentIncidents: [
      { date: '2024-06-17', description: 'PH-CN coast guard collision at Second Thomas Shoal', severity: 'serious' },
      { date: '2024-03-05', description: 'Chinese water cannon attack on PH vessels', severity: 'moderate' },
      { date: '2023-08-05', description: 'PH removes Chinese floating barrier at Scarborough', severity: 'moderate' },
    ],
    analystAssessment: `
China's "nine-dash line" claim has no basis in UNCLOS and was explicitly rejected by the 2016 PCA ruling.
However, China has achieved de facto control through island-building and militarization.
Key flashpoints: Second Thomas Shoal (BRP Sierra Madre), Scarborough Shoal, Whitsun Reef.
US MDT with Philippines is primary deterrent; escalation could trigger US involvement.
Risk trajectory: INCREASING due to Philippine pushback and US support.
    `,
    lastUpdated: '2024-12-01',
  },

  // Senkaku/Diaoyu Islands
  'senkaku_diaoyu': {
    id: 'senkaku-diaoyu',
    name: 'Senkaku Islands (Diaoyu Islands)',
    region: 'East China Sea',
    domain: 'maritime',
    parties: [
      {
        code: 'JP',
        name: 'Japan',
        claimBasis: 'Discovery 1885, terra nullius incorporation, effective administration since 1972',
        controlPercentage: 100,
        militaryPresence: 'moderate',  // Coast guard, JASDF nearby
      },
      {
        code: 'CN',
        name: 'China (PRC)',
        claimBasis: 'Historical records (Ming Dynasty maps), post-1969 resurvey (oil discovery)',
        controlPercentage: 0,
        militaryPresence: 'heavy',  // CCG/PLAN constant presence in vicinity
      },
      {
        code: 'TW',
        name: 'Taiwan (ROC)',
        claimBasis: 'Same historical basis as PRC (inherited from ROC)',
        controlPercentage: 0,
        militaryPresence: 'none',
      },
    ],
    status: 'high_tension',
    escalationRisk: 0.40,
    strategicImportance: 0.85,
    resourcesAtStake: [
      'Oil: Uncertain but potentially significant',
      'Natural gas deposits',
      'Fishing rights',
      'Strategic position for Taiwan contingency',
      'ADIZ overlap',
    ],
    internationalLaw: {
      relevantTreaties: ['San Francisco Treaty 1951 (ambiguous)', 'US-Japan Okinawa Reversion 1972'],
      icjRulings: [],
      unResolutions: [],
      status: 'ambiguous',
    },
    recentIncidents: [
      { date: '2024-02-04', description: 'Chinese coast guard vessels enter territorial waters (157th incursion 2024)', severity: 'moderate' },
      { date: '2023-12-26', description: 'Chinese survey ship conducts operations in contiguous zone', severity: 'minor' },
      { date: '2023-08-22', description: 'PLA Navy exercises near islands', severity: 'serious' },
    ],
    analystAssessment: `
Japan has administered islands since 1972 and US explicitly includes them under Article 5 of US-Japan Security Treaty.
China's claim appears primarily opportunistic (emerged after 1969 oil survey) rather than historically grounded.
Key risk: Gray zone operations (CCG incursions) could escalate if Japan Coast Guard responds forcefully.
Escalation pathway: CCG incursion -> JCG response -> PLAN involvement -> JSDF/USFJ activation.
US commitment is explicit but untested; China may probe during US distraction.
    `,
    lastUpdated: '2024-12-01',
  },

  // Sino-Indian Border (LAC)
  'sino_indian_lac': {
    id: 'sino-indian-lac',
    name: 'Sino-Indian Border (Line of Actual Control)',
    region: 'Himalayas',
    domain: 'land',
    parties: [
      {
        code: 'IN',
        name: 'India',
        claimBasis: 'McMahon Line 1914, British inheritance, Simla Convention',
        controlPercentage: 50,
        militaryPresence: 'heavy',
      },
      {
        code: 'CN',
        name: 'China (PRC)',
        claimBasis: 'Rejects McMahon Line, historical claims to "South Tibet" (Arunachal Pradesh)',
        controlPercentage: 50,
        militaryPresence: 'heavy',
      },
    ],
    status: 'high_tension',
    escalationRisk: 0.50,
    strategicImportance: 0.75,
    resourcesAtStake: [
      'Water sources (Brahmaputra headwaters)',
      'Strategic high ground',
      'Aksai Chin (Chinese road to Tibet)',
      'Arunachal Pradesh (90,000 km²)',
      'Doklam/Donglang plateau',
    ],
    internationalLaw: {
      relevantTreaties: ['Simla Convention 1914 (China rejected)', 'Panchsheel 1954', 'Various CBMs'],
      icjRulings: [],
      unResolutions: [],
      status: 'contested',
    },
    recentIncidents: [
      { date: '2024-10-21', description: 'Disengagement agreement reached at Demchok/Depsang', severity: 'moderate' },
      { date: '2022-12-09', description: 'Tawang sector clash, first shots since 1975', severity: 'serious' },
      { date: '2020-06-15', description: 'Galwan Valley clash, 20 Indian soldiers killed', severity: 'critical' },
    ],
    analystAssessment: `
Most dangerous land border dispute globally. Both nuclear powers with large conventional forces.
2020 Galwan clash fundamentally changed relationship - no return to pre-2020 status quo.
China's infrastructure build-up along LAC continues to shift balance.
India's response: Quad alignment, military modernization, reduced economic ties.
Key chokepoints: Depsang Plains, Galwan, Pangong Tso, Doklam.
Risk of escalation during China-Taiwan crisis (India may see opportunity/China may preempt).
    `,
    lastUpdated: '2024-12-01',
  },

  // Aksai Chin (subset of LAC)
  'aksai_chin': {
    id: 'aksai-chin',
    name: 'Aksai Chin',
    region: 'Western Himalayas',
    domain: 'land',
    parties: [
      {
        code: 'CN',
        name: 'China (PRC)',
        claimBasis: 'Effective control since 1962 war, G219 highway to Tibet',
        controlPercentage: 100,
        militaryPresence: 'moderate',
      },
      {
        code: 'IN',
        name: 'India',
        claimBasis: 'British inheritance (Jammu & Kashmir), never accepted 1962 seizure',
        controlPercentage: 0,
        militaryPresence: 'none',
      },
    ],
    status: 'frozen',
    escalationRisk: 0.15,
    strategicImportance: 0.60,
    resourcesAtStake: [
      'Strategic road (G219) connecting Xinjiang to Tibet',
      'High-altitude plateau (strategic depth)',
      '37,244 km² territory',
    ],
    internationalLaw: {
      relevantTreaties: ['None accepted by both parties'],
      icjRulings: [],
      unResolutions: [],
      status: 'contested',
    },
    recentIncidents: [],
    analystAssessment: `
Effectively a frozen conflict. China has controlled since 1962 and built critical infrastructure.
India officially claims but has no realistic path to recovering territory.
Primary importance: China's road access to Tibet; any Indian interdiction would be casus belli.
Status unlikely to change absent major war or political transformation in either country.
    `,
    lastUpdated: '2024-12-01',
  },

  // Arunachal Pradesh / South Tibet
  'arunachal_pradesh': {
    id: 'arunachal-south-tibet',
    name: 'Arunachal Pradesh ("South Tibet")',
    region: 'Eastern Himalayas',
    domain: 'land',
    parties: [
      {
        code: 'IN',
        name: 'India',
        claimBasis: 'McMahon Line, effective administration, 1.4M population, Indian state since 1987',
        controlPercentage: 100,
        militaryPresence: 'heavy',
      },
      {
        code: 'CN',
        name: 'China (PRC)',
        claimBasis: 'Rejects McMahon Line, claims as "South Tibet", historical/ethnic ties',
        controlPercentage: 0,
        militaryPresence: 'heavy',  // On Chinese side of LAC
      },
    ],
    status: 'moderate_tension',
    escalationRisk: 0.35,
    strategicImportance: 0.70,
    resourcesAtStake: [
      '83,743 km² territory',
      'Strategic Himalayan frontier',
      'Brahmaputra watershed',
      '1.4 million Indian citizens',
      'Tawang monastery (significant to Tibetan Buddhism)',
    ],
    internationalLaw: {
      relevantTreaties: ['McMahon Line 1914 (China rejects)'],
      icjRulings: [],
      unResolutions: [],
      status: 'contested',
    },
    recentIncidents: [
      { date: '2022-12-09', description: 'Tawang sector clash - first clash in this sector since 1962', severity: 'serious' },
    ],
    analystAssessment: `
Unlike Aksai Chin, India has full administrative control with population and infrastructure.
China periodically issues stapled visas to Arunachal residents, names features in Chinese.
2022 Tawang clash signals China may be more willing to use force than previously assumed.
Recovery by China would require major war - not a realistic near-term scenario.
Long-term risk: If China achieves regional dominance, may press claims more aggressively.
    `,
    lastUpdated: '2024-12-01',
  },

  // Bhutan-China Border
  'bhutan_china': {
    id: 'bhutan-china',
    name: 'Bhutan-China Border',
    region: 'Himalayas',
    domain: 'land',
    parties: [
      {
        code: 'BT',
        name: 'Bhutan',
        claimBasis: 'Historical administration, de facto borders',
        controlPercentage: 65,
        militaryPresence: 'minimal',
      },
      {
        code: 'CN',
        name: 'China (PRC)',
        claimBasis: 'Historical claims, recent encroachment (village building)',
        controlPercentage: 35,
        militaryPresence: 'moderate',
      },
    ],
    status: 'moderate_tension',
    escalationRisk: 0.25,
    strategicImportance: 0.55,
    resourcesAtStake: [
      'Doklam Plateau (strategic high ground)',
      'Northern territories',
      'China-India buffer zone',
    ],
    internationalLaw: {
      relevantTreaties: ['1998 Bilateral Agreement (peaceful resolution)'],
      icjRulings: [],
      unResolutions: [],
      status: 'ambiguous',
    },
    recentIncidents: [
      { date: '2021-06-01', description: 'Satellite imagery shows new Chinese village in disputed area', severity: 'moderate' },
      { date: '2017-06-16', description: 'Doklam standoff - China-India 73-day standoff over road building', severity: 'critical' },
    ],
    analystAssessment: `
Bhutan has no diplomatic relations with China - negotiations through intermediaries.
Doklam is critical because it's a "tri-junction" affecting India's Siliguri Corridor.
2017 standoff demonstrated India will intervene militarily to protect Bhutanese claims affecting Indian security.
China's "village building" strategy: create facts on ground without military confrontation.
Bhutan lacks capacity to resist alone - depends entirely on India security guarantee.
    `,
    lastUpdated: '2024-12-01',
  },

  // Paracel Islands
  'paracel_islands': {
    id: 'paracel-islands',
    name: 'Paracel Islands (Xisha)',
    region: 'South China Sea (Northern)',
    domain: 'maritime',
    parties: [
      {
        code: 'CN',
        name: 'China (PRC)',
        claimBasis: 'Seized from South Vietnam 1974, full administrative control, Sansha City HQ',
        controlPercentage: 100,
        militaryPresence: 'dominant',
      },
      {
        code: 'VN',
        name: 'Vietnam',
        claimBasis: 'Historical administration (pre-1974), French colonial inheritance',
        controlPercentage: 0,
        militaryPresence: 'none',
      },
      {
        code: 'TW',
        name: 'Taiwan (ROC)',
        claimBasis: 'Same basis as PRC (inherited claim)',
        controlPercentage: 0,
        militaryPresence: 'none',
      },
    ],
    status: 'frozen',
    escalationRisk: 0.20,
    strategicImportance: 0.65,
    resourcesAtStake: [
      'Fisheries',
      'Oil/gas (potential)',
      'Strategic military position',
      'Woody Island military base',
    ],
    internationalLaw: {
      relevantTreaties: ['None - seized by force 1974'],
      icjRulings: [],
      unResolutions: [],
      status: 'contested',
    },
    recentIncidents: [],
    analystAssessment: `
Effectively a settled dispute in China's favor through fait accompli (1974 seizure).
Vietnam has no realistic path to recovery. China has heavily militarized Woody Island.
Key lesson: Demonstrates China's willingness to use force when costs are low.
US did not intervene in 1974 (during Vietnam War withdrawal).
    `,
    lastUpdated: '2024-12-01',
  },

  // Taiwan (Political Entity)
  'taiwan': {
    id: 'taiwan',
    name: 'Taiwan (Republic of China)',
    region: 'Western Pacific',
    domain: 'maritime',
    parties: [
      {
        code: 'TW',
        name: 'Taiwan (ROC)',
        claimBasis: 'Continuous government since 1949, democratic legitimacy, de facto independence',
        controlPercentage: 100,
        militaryPresence: 'heavy',
      },
      {
        code: 'CN',
        name: 'China (PRC)',
        claimBasis: 'One China principle, UN Resolution 2758, ROC civil war successor state',
        controlPercentage: 0,
        militaryPresence: 'heavy',  // Opposite shore
      },
    ],
    status: 'high_tension',
    escalationRisk: 0.30,  // See TAIWAN_THREAT_ASSESSMENT for conditional analysis
    strategicImportance: 0.98,
    resourcesAtStake: [
      'TSMC - 90% of advanced semiconductors',
      '24 million population',
      'First Island Chain keystone',
      'Democratic values/credibility',
      '$800B GDP',
    ],
    internationalLaw: {
      relevantTreaties: ['Taiwan Relations Act 1979 (US domestic law)', 'Three Communiqués', 'Six Assurances'],
      icjRulings: [],
      unResolutions: ['UNGA 2758 (1971) - replaced ROC with PRC at UN'],
      status: 'ambiguous',  // Intentional strategic ambiguity
    },
    recentIncidents: [
      { date: '2024-05-23', description: 'PLA exercises "Joint Sword-2024A" surrounding Taiwan post-inauguration', severity: 'serious' },
      { date: '2023-04-08', description: 'PLA exercises following Tsai-McCarthy meeting', severity: 'serious' },
      { date: '2022-08-02', description: 'Pelosi visit triggers largest PLA exercises ever, median line crossed', severity: 'critical' },
    ],
    analystAssessment: `
See TAIWAN_THREAT_ASSESSMENT for detailed conditional scenario analysis.
Summary: Status quo likely to persist unless US distracted, power shift, or alliance collapse.
Xi has tied personal legacy to "reunification" but rationality still constrains action.
Semiconductor leverage works both ways - prevents invasion but also makes Taiwan target.
Time is NOT necessarily on China's side (demographic decline, economic challenges).
Key indicator: PLA amphibious lift capability timeline.
    `,
    lastUpdated: '2024-12-01',
  },
};

// Helper function to get all China disputes by risk level
export function getChinaDisputesByRisk(): TerritorialDispute[] {
  return Object.values(CHINA_TERRITORIAL_DISPUTES)
    .sort((a, b) => b.escalationRisk - a.escalationRisk);
}

// Helper function to get disputes by status
export function getChinaDisputesByStatus(status: TerritorialDispute['status']): TerritorialDispute[] {
  return Object.values(CHINA_TERRITORIAL_DISPUTES)
    .filter(d => d.status === status);
}

// ============================================
// INDIA-PAKISTAN DISPUTES (TWO NUCLEAR POWERS)
// ============================================

export const INDIA_PAKISTAN_DISPUTES: Record<string, TerritorialDispute> = {
  // Kashmir - Core Dispute
  'kashmir_loc': {
    id: 'kashmir-loc',
    name: 'Kashmir (Line of Control)',
    region: 'South Asia',
    domain: 'land',
    parties: [
      {
        code: 'IN',
        name: 'India',
        claimBasis: 'Instrument of Accession 1947, UN resolutions (interpreted as requiring Pakistan withdrawal first)',
        controlPercentage: 45,
        militaryPresence: 'heavy',
      },
      {
        code: 'PK',
        name: 'Pakistan',
        claimBasis: 'Two-Nation Theory, UN resolutions calling for plebiscite, majority Muslim population',
        controlPercentage: 35,
        militaryPresence: 'heavy',
      },
      {
        code: 'CN',
        name: 'China (PRC)',
        claimBasis: '1963 Sino-Pak agreement, 1962 war gains (Aksai Chin)',
        controlPercentage: 20,
        militaryPresence: 'moderate',
      },
    ],
    status: 'high_tension',
    escalationRisk: 0.60,
    strategicImportance: 0.90,
    resourcesAtStake: [
      'Water sources (Indus River system)',
      'Strategic high ground',
      'National identity/prestige (both sides)',
      '~14 million population under dispute',
      'Nuclear flashpoint',
    ],
    internationalLaw: {
      relevantTreaties: ['UNSCR 47 (1948) - plebiscite (never implemented)', 'Simla Agreement 1972', 'Lahore Declaration 1999'],
      icjRulings: [],
      unResolutions: ['UNSCR 47', 'UNSCR 80', 'UNSCR 91', 'UNSCR 122'],
      status: 'contested',
    },
    recentIncidents: [
      { date: '2019-08-05', description: 'India revokes Article 370, removes J&K special status', severity: 'critical' },
      { date: '2019-02-26', description: 'Indian Balakot airstrike following Pulwama attack', severity: 'critical' },
      { date: '2019-02-27', description: 'Pakistan shoots down Indian jet, captures pilot', severity: 'critical' },
      { date: '2016-09-29', description: 'Indian "surgical strikes" across LoC', severity: 'serious' },
    ],
    analystAssessment: `
MOST DANGEROUS NUCLEAR FLASHPOINT ON EARTH.
Four wars (1947, 1965, 1971, 1999), continuous ceasefire violations, two nuclear-armed states.
2019 Pulwama-Balakot crisis brought both to brink of nuclear war - closest since 1999 Kargil.
Key dynamics:
- Pakistan uses non-state actors (Lashkar, Jaish) for plausible deniability
- India's "Cold Start" doctrine threatens rapid conventional strike
- Pakistan's tactical nuclear weapons lower threshold
- Both have rejected international mediation
Water scarcity (Indus Waters Treaty under strain) could be future trigger.
No resolution in sight - both societies politically invested in conflict.
    `,
    lastUpdated: '2024-12-01',
  },

  // Siachen Glacier
  'siachen': {
    id: 'siachen-glacier',
    name: 'Siachen Glacier',
    region: 'Karakoram',
    domain: 'land',
    parties: [
      {
        code: 'IN',
        name: 'India',
        claimBasis: 'Operation Meghdoot 1984, effective control of heights',
        controlPercentage: 70,
        militaryPresence: 'heavy',
      },
      {
        code: 'PK',
        name: 'Pakistan',
        claimBasis: 'Simla Agreement ambiguity (line not demarcated above NJ9842)',
        controlPercentage: 30,
        militaryPresence: 'heavy',
      },
    ],
    status: 'frozen',
    escalationRisk: 0.25,
    strategicImportance: 0.55,
    resourcesAtStake: [
      'Strategic high ground (20,000+ feet)',
      'Glacial water sources',
      'National prestige',
      'Military lives (weather kills more than combat)',
    ],
    internationalLaw: {
      relevantTreaties: ['Simla Agreement 1972 (silent on area north of NJ9842)'],
      icjRulings: [],
      unResolutions: [],
      status: 'ambiguous',
    },
    recentIncidents: [
      { date: '2012-04-07', description: 'Pakistani avalanche kills 140 soldiers', severity: 'serious' },
    ],
    analystAssessment: `
"The world's highest battlefield" - more soldiers die from weather than combat.
India holds commanding heights since 1984 Operation Meghdoot.
Both sides lose hundreds to altitude, cold, avalanches annually.
Strategically minor but symbolically important - neither will withdraw first.
Demilitarization discussed but India refuses without authentication of current positions.
    `,
    lastUpdated: '2024-12-01',
  },

  // Sir Creek
  'sir_creek': {
    id: 'sir-creek',
    name: 'Sir Creek Maritime Boundary',
    region: 'Arabian Sea',
    domain: 'maritime',
    parties: [
      {
        code: 'IN',
        name: 'India',
        claimBasis: 'Mid-channel (thalweg) principle, 1914 Bombay Government resolution',
        controlPercentage: 50,
        militaryPresence: 'moderate',
      },
      {
        code: 'PK',
        name: 'Pakistan',
        claimBasis: '1914 map showing eastern shore as boundary',
        controlPercentage: 50,
        militaryPresence: 'moderate',
      },
    ],
    status: 'low_tension',
    escalationRisk: 0.15,
    strategicImportance: 0.40,
    resourcesAtStake: [
      'EEZ implications (affects ~35,000 km² of maritime zone)',
      'Fishing rights',
      'Oil/gas (potential)',
    ],
    internationalLaw: {
      relevantTreaties: ['UNCLOS'],
      icjRulings: [],
      unResolutions: [],
      status: 'ambiguous',
    },
    recentIncidents: [],
    analystAssessment: `
Most tractable India-Pakistan dispute - technical disagreement amenable to legal resolution.
Both sides have held multiple rounds of talks; solution technically achievable.
Political will lacking - neither wants to be seen as conceding.
Could be confidence-building measure if resolved, but unlikely given overall relationship.
    `,
    lastUpdated: '2024-12-01',
  },
};

// India-Pakistan Conditional Threat Assessment
export const INDIA_PAKISTAN_THREAT_ASSESSMENT: ConditionalThreatAssessment = {
  baseScenario: 'status_quo',

  scenarios: [
    {
      id: 'terror-attack-escalation',
      name: 'Terror Attack Escalation',
      description: 'Major terror attack in India (Mumbai 2008-style or worse) triggers military response cycle',
      probability: 0.25,
      timeframe: '5 years',
      requiredConditions: [
        { condition: 'Major attack on Indian soil with Pakistan links', currentStatus: false, probabilityIfTrue: 0.5 },
        { condition: 'Indian domestic pressure demands response', currentStatus: true, probabilityIfTrue: 0.8 },
        { condition: 'Pakistan unable/unwilling to act against perpetrators', currentStatus: true, probabilityIfTrue: 0.7 },
      ],
      blockingConditions: [
        { condition: 'Strong international pressure for restraint', currentStatus: false, blockingStrength: 0.3 },
        { condition: 'Clear Pakistani cooperation against attackers', currentStatus: false, blockingStrength: 0.6 },
      ],
      actorDependencies: [
        { actor: 'IN', requiredAction: 'Conduct strike on Pakistani territory', likelihoodOfAction: 0.7 },
        { actor: 'PK', requiredAction: 'Retaliate conventionally', likelihoodOfAction: 0.6 },
        { actor: 'US', requiredAction: 'Fail to de-escalate', likelihoodOfAction: 0.3 },
      ],
      outcomes: [
        { outcome: 'Limited strikes, managed de-escalation (2019 model)', probability: 0.50 },
        { outcome: 'Conventional war, stays below nuclear threshold', probability: 0.30 },
        { outcome: 'Escalation to nuclear use', probability: 0.10 },
        { outcome: 'International intervention prevents escalation', probability: 0.10 },
      ],
    },
    {
      id: 'kashmir-uprising',
      name: 'Kashmir Mass Uprising',
      description: 'Mass civilian uprising in Indian Kashmir triggers Pakistani intervention',
      probability: 0.15,
      timeframe: '10 years',
      requiredConditions: [
        { condition: 'Sustained mass protests in Kashmir', currentStatus: false, probabilityIfTrue: 0.4 },
        { condition: 'Indian security forces major atrocity', currentStatus: false, probabilityIfTrue: 0.6 },
        { condition: 'Pakistan sees opportunity for intervention', currentStatus: false, probabilityIfTrue: 0.3 },
      ],
      blockingConditions: [
        { condition: 'Indian restraint in response', currentStatus: false, blockingStrength: 0.5 },
        { condition: 'International monitors presence', currentStatus: false, blockingStrength: 0.4 },
      ],
      actorDependencies: [
        { actor: 'PK', requiredAction: 'Send "volunteers" across LoC', likelihoodOfAction: 0.4 },
        { actor: 'IN', requiredAction: 'Interpret as act of war', likelihoodOfAction: 0.8 },
      ],
      outcomes: [
        { outcome: 'Internationalization of dispute', probability: 0.35 },
        { outcome: 'Limited border war', probability: 0.40 },
        { outcome: 'Full-scale war', probability: 0.20 },
        { outcome: 'Nuclear exchange', probability: 0.05 },
      ],
    },
    {
      id: 'water-conflict',
      name: 'Indus Waters Crisis',
      description: 'Climate change / water scarcity triggers conflict over Indus River system',
      probability: 0.20,
      timeframe: '15 years',
      requiredConditions: [
        { condition: 'Severe drought in Indus basin', currentStatus: false, probabilityIfTrue: 0.5 },
        { condition: 'India builds upstream infrastructure affecting Pakistan flow', currentStatus: false, probabilityIfTrue: 0.4 },
        { condition: 'Indus Waters Treaty breaks down', currentStatus: false, probabilityIfTrue: 0.3 },
      ],
      blockingConditions: [
        { condition: 'World Bank arbitration succeeds', currentStatus: true, blockingStrength: 0.5 },
        { condition: 'Climate adaptation reduces dependence', currentStatus: false, blockingStrength: 0.3 },
      ],
      actorDependencies: [
        { actor: 'PK', requiredAction: 'Declare water diversion as casus belli', likelihoodOfAction: 0.5 },
        { actor: 'IN', requiredAction: 'Refuse to negotiate', likelihoodOfAction: 0.4 },
      ],
      outcomes: [
        { outcome: 'Negotiated water-sharing update', probability: 0.40 },
        { outcome: 'Low-intensity conflict over dams', probability: 0.35 },
        { outcome: 'Major war over water', probability: 0.20 },
        { outcome: 'Nuclear exchange', probability: 0.05 },
      ],
    },
  ],

  criticalDependencies: [
    {
      factor: 'Pakistani Military-Civilian Relations',
      currentState: 'Military dominant, civilian government weak',
      trendDirection: 'deteriorating',
      impact: 'enabling',
    },
    {
      factor: 'Indian Domestic Politics',
      currentState: 'Nationalist BJP government, hawkish',
      trendDirection: 'stable',
      impact: 'enabling',
    },
    {
      factor: 'Pakistan Nuclear Doctrine',
      currentState: 'First use, tactical weapons deployed',
      trendDirection: 'deteriorating',
      impact: 'enabling',
    },
    {
      factor: 'US-Pakistan Relations',
      currentState: 'Strained, limited leverage',
      trendDirection: 'stable',
      impact: 'neutral',
    },
    {
      factor: 'China-Pakistan Axis',
      currentState: 'Strong, CPEC deepening ties',
      trendDirection: 'stable',
      impact: 'enabling',  // Emboldens Pakistan
    },
    {
      factor: 'Kashmir Unrest',
      currentState: 'Suppressed but simmering post-Article 370',
      trendDirection: 'stable',
      impact: 'enabling',
    },
  ],

  probabilityOverTime: [
    { years: 1, baselineProbability: 0.05, withUSDistraction: 0.10, withPowerShift: 0.08, withAllianceCollapse: 0.12 },
    { years: 3, baselineProbability: 0.12, withUSDistraction: 0.20, withPowerShift: 0.15, withAllianceCollapse: 0.22 },
    { years: 5, baselineProbability: 0.20, withUSDistraction: 0.30, withPowerShift: 0.25, withAllianceCollapse: 0.35 },
    { years: 10, baselineProbability: 0.35, withUSDistraction: 0.45, withPowerShift: 0.40, withAllianceCollapse: 0.50 },
  ],

  analystNotes: `
ASSESSMENT: India-Pakistan is a perpetual crisis waiting to happen.

Key differences from China-Taiwan:
1. BOTH sides have nuclear weapons and delivery systems
2. Non-state actors provide plausible deniability for Pakistan
3. Multiple wars already fought - psychological barrier to conflict is LOW
4. Territorial dispute involves real populations and daily life
5. Water scarcity adds resource competition dimension
6. Pakistan's conventional inferiority lowers nuclear threshold

2019 Balakot-Pulwama crisis demonstrated:
- India WILL strike Pakistani territory after major attacks
- Pakistan WILL retaliate conventionally
- Nuclear threats were invoked by both sides
- De-escalation required US, Saudi, UAE intervention
- Next time may not de-escalate as easily

Most dangerous scenario: Pakistani state collapse + loose nukes + Indian intervention
Second most dangerous: Major terror attack + Indian "Cold Start" strike + Pakistani tactical nuclear use

Unlike Taiwan, there is NO "status quo that favors indefinite peace" here.
Both societies are politically committed to the conflict.
    `,
  lastUpdated: '2024-12-01',
};

// Helper function to get all disputes by risk level globally
export function getAllDisputesByRisk(): TerritorialDispute[] {
  return [
    ...Object.values(CHINA_TERRITORIAL_DISPUTES),
    ...Object.values(INDIA_PAKISTAN_DISPUTES),
  ].sort((a, b) => b.escalationRisk - a.escalationRisk);
}

export interface BorderSegment {
  id: string;

  // Domain - what type of boundary
  domain: BorderDomain;
  subdomains?: BorderDomain[];    // Additional domains (e.g., land + riverine)

  // Geographic definition
  coordinates: number[][];        // [[lat, lon], [lat, lon], ...] polyline
  length_km: number;

  // For maritime boundaries
  maritimeZones?: {
    type: MaritimeZoneType;
    claimedBy: string[];          // ISO codes of claimants
    baselineNm?: number;          // Nautical miles from baseline
  }[];

  // For airspace boundaries
  airspaceType?: AirspaceType;
  altitudeCeiling?: number;       // Feet or FL

  // Parties
  side_a: string;                 // ISO code
  side_b: string;                 // ISO code
  side_a_name: string;
  side_b_name: string;

  // Legal status
  legalStatus: BorderLegalStatus;
  treatyBasis?: string;           // Legal agreement governing border
  recognizedBy: 'both' | 'side_a' | 'side_b' | 'neither' | 'international';

  // Existential threat assessment
  existentialThreat?: {
    level: ExistentialThreatLevel;
    threatenedParty: string;      // ISO code of threatened entity
    threatSource: string;         // ISO code of threat source
    threatType: 'invasion' | 'blockade' | 'annexation' | 'absorption' | 'destruction';
    deterrents: string[];         // What's preventing escalation
    triggerConditions: string[];  // What could trigger escalation
    lastAssessment: string;
  };

  // Enforcement
  enforcement_a: BorderEnforcementLevel;  // Side A enforcement
  enforcement_b: BorderEnforcementLevel;  // Side B enforcement

  // Infrastructure
  physicalBarrier: 'none' | 'fence' | 'wall' | 'natural' | 'minefield' | 'multiple';
  barrierCoverage: number;        // 0-100% of segment

  // Crossings
  officialCrossings: number;      // Count of official ports
  informalCrossings: number;      // Estimated informal crossing points

  // Flow assessment
  flows: {
    type: FlowType;
    direction: 'a_to_b' | 'b_to_a' | 'bidirectional';
    volume: 'none' | 'minimal' | 'low' | 'moderate' | 'high' | 'extreme';
    trend: 'decreasing' | 'stable' | 'increasing' | 'surging';
  }[];

  // Temporal
  effectiveDate: string;
  lastAssessment: string;
  confidence: number;             // 0-1 intelligence confidence
}

export interface BorderCrossing {
  id: string;
  name: string;

  // Location
  position: { lat: number; lon: number };
  borderSegmentId: string;

  // Type
  type: BorderCrossingType;

  // Status
  status: 'open' | 'closed' | 'restricted' | 'destroyed';
  operatingHours?: string;        // e.g., "24/7" or "0600-2200"

  // Capacity
  vehicleCapacity?: number;       // Vehicles per hour
  pedestrianCapacity?: number;    // People per hour

  // Current conditions
  waitTime_a_to_b?: number;       // Minutes
  waitTime_b_to_a?: number;       // Minutes

  // Infrastructure
  hasCustoms: boolean;
  hasImmigration: boolean;
  hasMilitary: boolean;

  // Special designations
  designations: string[];         // e.g., ['commercial', 'passenger', 'refugee_processing']

  lastUpdated: string;
}

// Historical examples
export const BORDER_EXAMPLES: Record<string, BorderSegment> = {
  // US-Mexico border historical (pre-2016)
  'us_mexico_historical': {
    id: 'us-mx-pre2016',
    domain: 'land',
    subdomains: ['riverine'],     // Rio Grande portion
    coordinates: [],
    length_km: 3145,
    side_a: 'US',
    side_b: 'MX',
    side_a_name: 'United States',
    side_b_name: 'Mexico',
    legalStatus: 'demarcated',
    treatyBasis: 'Treaty of Guadalupe Hidalgo 1848, Gadsden Purchase 1854',
    recognizedBy: 'both',
    enforcement_a: 'porous',
    enforcement_b: 'minimal',
    physicalBarrier: 'fence',
    barrierCoverage: 35,
    officialCrossings: 48,
    informalCrossings: 200,
    flows: [
      { type: 'people', direction: 'b_to_a', volume: 'high', trend: 'stable' },
      { type: 'goods_legal', direction: 'bidirectional', volume: 'extreme', trend: 'increasing' },
      { type: 'narcotics', direction: 'b_to_a', volume: 'high', trend: 'increasing' },
      { type: 'weapons', direction: 'a_to_b', volume: 'moderate', trend: 'stable' },
    ],
    effectiveDate: '1990-01-01',
    lastAssessment: '2015-12-31',
    confidence: 0.8,
  },

  // Korean DMZ
  'korea_dmz': {
    id: 'kr-dmz',
    domain: 'land',
    coordinates: [],
    length_km: 250,
    side_a: 'KR',
    side_b: 'KP',
    side_a_name: 'South Korea',
    side_b_name: 'North Korea',
    legalStatus: 'militarized',
    treatyBasis: 'Korean Armistice Agreement 1953',
    recognizedBy: 'international',
    existentialThreat: {
      level: 'elevated',
      threatenedParty: 'KR',
      threatSource: 'KP',
      threatType: 'invasion',
      deterrents: ['US alliance', 'ROK military capability', 'nuclear umbrella', 'economic destruction mutual'],
      triggerConditions: ['regime collapse', 'succession crisis', 'provocation escalation', 'US withdrawal'],
      lastAssessment: '2024-01-01',
    },
    enforcement_a: 'militarized',
    enforcement_b: 'shoot_to_kill',
    physicalBarrier: 'multiple',
    barrierCoverage: 100,
    officialCrossings: 1,
    informalCrossings: 0,
    flows: [
      { type: 'people', direction: 'bidirectional', volume: 'none', trend: 'stable' },
      { type: 'information', direction: 'a_to_b', volume: 'minimal', trend: 'stable' },
    ],
    effectiveDate: '1953-07-27',
    lastAssessment: '2024-01-01',
    confidence: 0.95,
  },

  // EU Internal (Schengen)
  'eu_internal': {
    id: 'eu-schengen',
    domain: 'land',
    coordinates: [],
    length_km: 0,
    side_a: 'EU',
    side_b: 'EU',
    side_a_name: 'Schengen Member',
    side_b_name: 'Schengen Member',
    legalStatus: 'open',
    treatyBasis: 'Schengen Agreement 1985/1990',
    recognizedBy: 'both',
    enforcement_a: 'none',
    enforcement_b: 'none',
    physicalBarrier: 'none',
    barrierCoverage: 0,
    officialCrossings: 0,
    informalCrossings: 0,
    flows: [
      { type: 'people', direction: 'bidirectional', volume: 'extreme', trend: 'stable' },
      { type: 'goods_legal', direction: 'bidirectional', volume: 'extreme', trend: 'stable' },
    ],
    effectiveDate: '1995-03-26',
    lastAssessment: '2024-01-01',
    confidence: 1.0,
  },

  // India-Pakistan Line of Control
  'india_pakistan_loc': {
    id: 'in-pk-loc',
    domain: 'land',
    coordinates: [],
    length_km: 740,
    side_a: 'IN',
    side_b: 'PK',
    side_a_name: 'India',
    side_b_name: 'Pakistan',
    legalStatus: 'disputed',
    treatyBasis: 'Simla Agreement 1972 (not recognized as international border)',
    recognizedBy: 'neither',
    existentialThreat: {
      level: 'potential',
      threatenedParty: 'IN',       // Both sides have nuclear weapons
      threatSource: 'PK',
      threatType: 'invasion',
      deterrents: ['nuclear deterrence mutual', 'international pressure', 'economic costs'],
      triggerConditions: ['major terror attack', 'Kashmiri uprising', 'water crisis', 'political instability'],
      lastAssessment: '2024-01-01',
    },
    enforcement_a: 'militarized',
    enforcement_b: 'militarized',
    physicalBarrier: 'fence',
    barrierCoverage: 80,
    officialCrossings: 2,
    informalCrossings: 5,
    flows: [
      { type: 'people', direction: 'bidirectional', volume: 'minimal', trend: 'stable' },
      { type: 'military', direction: 'bidirectional', volume: 'moderate', trend: 'increasing' },
      { type: 'weapons', direction: 'b_to_a', volume: 'low', trend: 'stable' },
    ],
    effectiveDate: '1972-07-02',
    lastAssessment: '2024-01-01',
    confidence: 0.7,
  },

  // TAIWAN STRAIT - EXISTENTIAL THREAT EXAMPLE
  'taiwan_strait': {
    id: 'tw-strait',
    domain: 'maritime',
    subdomains: ['airspace'],
    coordinates: [],               // Median line coordinates
    length_km: 180,                // Width of strait at narrowest
    maritimeZones: [
      { type: 'territorial_sea', claimedBy: ['TW', 'CN'], baselineNm: 12 },
      { type: 'eez', claimedBy: ['TW', 'CN'], baselineNm: 200 },
      { type: 'disputed', claimedBy: ['CN'], baselineNm: 0 }, // PRC claims all
    ],
    airspaceType: 'adiz',
    side_a: 'TW',
    side_b: 'CN',
    side_a_name: 'Taiwan (ROC)',
    side_b_name: 'China (PRC)',
    legalStatus: 'disputed',
    treatyBasis: 'No formal treaty - ROC constitution claims mainland, PRC claims Taiwan',
    recognizedBy: 'neither',
    existentialThreat: {
      level: 'active',
      threatenedParty: 'TW',
      threatSource: 'CN',
      threatType: 'absorption',     // "Reunification" = state cessation
      deterrents: [
        'US Taiwan Relations Act',
        'Taiwan Strait geography',
        'TSMC semiconductor leverage',
        'International economic integration',
        'Taiwanese military modernization',
        'Uncertainty of US response',
        'Global economic consequences',
      ],
      triggerConditions: [
        'Taiwan independence declaration',
        'Foreign military basing',
        'Nuclear weapons development',
        'US-China conflict elsewhere',
        'PRC domestic instability requiring rally',
        'Perceived window of US weakness',
        'Taiwan internal political crisis',
      ],
      lastAssessment: '2024-01-01',
    },
    enforcement_a: 'militarized',
    enforcement_b: 'militarized',
    physicalBarrier: 'natural',     // Taiwan Strait itself
    barrierCoverage: 100,
    officialCrossings: 0,           // No formal crossings
    informalCrossings: 0,
    flows: [
      { type: 'people', direction: 'bidirectional', volume: 'moderate', trend: 'decreasing' },
      { type: 'goods_legal', direction: 'bidirectional', volume: 'high', trend: 'stable' },
      { type: 'military', direction: 'b_to_a', volume: 'high', trend: 'increasing' },  // PLA exercises
      { type: 'information', direction: 'b_to_a', volume: 'extreme', trend: 'increasing' }, // Cyber/info ops
    ],
    effectiveDate: '1949-10-01',
    lastAssessment: '2024-01-01',
    confidence: 0.85,
  },

  // South China Sea - 9-dash line
  'south_china_sea': {
    id: 'scs-nine-dash',
    domain: 'maritime',
    subdomains: ['airspace'],
    coordinates: [],
    length_km: 0,                   // Area claim, not line
    maritimeZones: [
      { type: 'disputed', claimedBy: ['CN', 'VN', 'PH', 'MY', 'BN', 'TW'], baselineNm: 0 },
      { type: 'eez', claimedBy: ['VN', 'PH', 'MY', 'BN'], baselineNm: 200 },
    ],
    side_a: 'CN',
    side_b: 'ASEAN',               // Multiple claimants
    side_a_name: 'China (PRC)',
    side_b_name: 'ASEAN Claimants',
    legalStatus: 'disputed',
    treatyBasis: 'UNCLOS (rejected by China for SCS), 2016 PCA ruling (rejected by China)',
    recognizedBy: 'neither',
    existentialThreat: {
      level: 'elevated',
      threatenedParty: 'PH',       // Most exposed
      threatSource: 'CN',
      threatType: 'blockade',
      deterrents: ['US MDT with Philippines', 'ASEAN unity', 'International shipping interests'],
      triggerConditions: ['Second Thomas Shoal escalation', 'Oil/gas development', 'Fishing incidents'],
      lastAssessment: '2024-01-01',
    },
    enforcement_a: 'heavy',         // CCG, maritime militia
    enforcement_b: 'moderate',      // Varies by claimant
    physicalBarrier: 'none',
    barrierCoverage: 0,
    officialCrossings: 0,
    informalCrossings: 0,
    flows: [
      { type: 'goods_legal', direction: 'bidirectional', volume: 'extreme', trend: 'stable' },
      { type: 'military', direction: 'bidirectional', volume: 'high', trend: 'increasing' },
    ],
    effectiveDate: '2009-05-07',    // China submitted 9-dash line to UN
    lastAssessment: '2024-01-01',
    confidence: 0.75,
  },

  // Ukraine-Russia (Donbas front line)
  'ukraine_donbas': {
    id: 'ua-donbas-2024',
    domain: 'land',
    coordinates: [],
    length_km: 1200,                // Approximate front line length
    side_a: 'UA',
    side_b: 'RU',
    side_a_name: 'Ukraine',
    side_b_name: 'Russia',
    legalStatus: 'disputed',
    treatyBasis: 'None - active conflict',
    recognizedBy: 'neither',
    existentialThreat: {
      level: 'ongoing',
      threatenedParty: 'UA',
      threatSource: 'RU',
      threatType: 'annexation',
      deterrents: ['Western military aid', 'Ukrainian resistance', 'International sanctions', 'NATO solidarity'],
      triggerConditions: ['Aid cutoff', 'NATO fracture', 'Regime change Kyiv', 'Russian mobilization surge'],
      lastAssessment: '2024-01-01',
    },
    enforcement_a: 'militarized',
    enforcement_b: 'militarized',
    physicalBarrier: 'multiple',    // Trenches, minefields, fortifications
    barrierCoverage: 100,
    officialCrossings: 0,
    informalCrossings: 0,
    flows: [
      { type: 'military', direction: 'bidirectional', volume: 'extreme', trend: 'stable' },
      { type: 'refugees', direction: 'a_to_b', volume: 'low', trend: 'decreasing' },  // Through Russia
    ],
    effectiveDate: '2022-02-24',
    lastAssessment: '2024-01-01',
    confidence: 0.7,                // Fog of war
  },

  // Gaza-Israel
  'gaza_israel': {
    id: 'gz-il',
    domain: 'land',
    subdomains: ['maritime', 'airspace'],
    coordinates: [],
    length_km: 51,
    side_a: 'IL',
    side_b: 'PS',                   // Palestinian territories
    side_a_name: 'Israel',
    side_b_name: 'Gaza Strip',
    legalStatus: 'militarized',
    treatyBasis: 'Oslo Accords (disputed applicability)',
    recognizedBy: 'neither',
    existentialThreat: {
      level: 'ongoing',
      threatenedParty: 'PS',        // Gaza
      threatSource: 'IL',
      threatType: 'destruction',
      deterrents: ['International pressure', 'Regional escalation risk', 'Hostage situation'],
      triggerConditions: ['Oct 7 attack triggered current operation'],
      lastAssessment: '2024-01-01',
    },
    enforcement_a: 'militarized',
    enforcement_b: 'militarized',   // Hamas
    physicalBarrier: 'wall',
    barrierCoverage: 100,
    officialCrossings: 2,           // Erez (people), Kerem Shalom (goods)
    informalCrossings: 0,           // Tunnels destroyed
    flows: [
      { type: 'military', direction: 'a_to_b', volume: 'extreme', trend: 'stable' },
      { type: 'refugees', direction: 'bidirectional', volume: 'none', trend: 'stable' }, // No exit
      { type: 'goods_legal', direction: 'a_to_b', volume: 'minimal', trend: 'decreasing' },
    ],
    effectiveDate: '2023-10-07',
    lastAssessment: '2024-01-01',
    confidence: 0.8,
  },
};

// ============================================
// MILITARY ASSET TYPES (NATO APP-6 Style)
// ============================================

export type MilitaryBranch =
  | 'army'
  | 'navy'
  | 'air_force'
  | 'marines'
  | 'special_forces'
  | 'paramilitary'
  | 'militia'
  | 'insurgent';

export type UnitType =
  // Ground Forces
  | 'infantry'
  | 'armor'                // Tanks
  | 'mechanized'           // Mechanized infantry
  | 'artillery'            // Field artillery
  | 'rocket_artillery'     // MLRS, HIMARS
  | 'air_defense'          // AA systems
  | 'engineer'             // Combat engineers
  | 'reconnaissance'       // Recon units
  | 'logistics'            // Supply/logistics
  // Air Assets
  | 'fighter'              // Air superiority
  | 'strike'               // Ground attack
  | 'bomber'               // Strategic bomber
  | 'transport'            // Air transport
  | 'helicopter_attack'    // Attack helicopters
  | 'helicopter_transport' // Transport helicopters
  | 'uav'                  // Drones
  | 'awacs'                // Airborne early warning
  // Naval
  | 'carrier'              // Aircraft carrier
  | 'submarine'            // Submarines
  | 'surface_combatant'    // Destroyers, frigates
  | 'amphibious'           // Amphibious assault
  // Special Capabilities
  | 'cyber'                // Cyber warfare
  | 'electronic_warfare'   // EW capabilities
  | 'nuclear'              // Nuclear capable
  | 'cbrn'                 // Chemical/Bio/Rad/Nuclear defense
  | 'special_operations';  // SOF

export type UnitSize =
  | 'team'                 // 4-5
  | 'squad'                // 8-12
  | 'platoon'              // 30-50
  | 'company'              // 100-200
  | 'battalion'            // 500-1000
  | 'regiment'             // 2000-3000
  | 'brigade'              // 3000-5000
  | 'division'             // 10000-20000
  | 'corps'                // 30000-50000
  | 'army'                 // 100000+
  | 'theater';             // Multiple armies

export type ThreatLevel =
  | 'none'                 // No threat
  | 'low'                  // Minimal activity
  | 'moderate'             // Some activity
  | 'elevated'             // Increased activity
  | 'high'                 // Active operations
  | 'critical';            // Imminent threat

export type ForceDisposition =
  | 'friendly'             // Allied/friendly
  | 'hostile'              // Enemy/hostile
  | 'neutral'              // Neutral party
  | 'unknown';             // Unknown affiliation

// ============================================
// CORE DATA STRUCTURES
// ============================================

export interface TerritorialClaim {
  claimantCode: string;           // ISO country code of claimant
  claimantName: string;
  claimType: SovereigntyStatus;
  claimBasis: string;             // Legal/historical basis
  internationalSupport: number;   // 0-1, % of UN recognizing
  effectiveControl: boolean;      // Does claimant control?
  controlPercentage: number;      // 0-100, % of territory controlled
}

export interface TerritorialStatus {
  // Core identification
  id: string;
  code: string;                   // ISO code or custom for disputed areas
  name: string;
  alternateNames: string[];       // Other names (e.g., "Crimea", "Republic of Crimea")

  // Geographic bounds
  bounds: {
    north: number;
    south: number;
    east: number;
    west: number;
  };
  centroid: { lat: number; lon: number };

  // Sovereignty situation
  legalSovereign: string;         // ISO code of legal sovereign
  legalSovereignName: string;
  deFactoController: string;      // ISO code of actual controller
  deFactoControllerName: string;

  // Status flags
  sovereigntyStatus: SovereigntyStatus;
  controlStatus: ControlStatus;
  isDisputed: boolean;
  isContested: boolean;           // Active conflict
  isOccupied: boolean;

  // Claims (for disputed territories)
  claims: TerritorialClaim[];

  // Dispute details
  disputes: {
    type: DisputeType;
    parties: string[];            // ISO codes
    description: string;
    startDate: string;            // ISO date
    ongoingConflict: boolean;
    internationalCases: string[]; // ICJ cases, UN resolutions
  }[];

  // Effective date of this status
  effectiveDate: string;          // ISO date
  endDate?: string;               // If status changed

  // Notes
  notes: string;
}

export interface GovernmentStatus {
  // Territory reference
  territoryCode: string;

  // Government details
  governmentType: GovernmentType;
  headOfState: string;
  headOfGovernment: string;
  rulingParty?: string;

  // Legitimacy
  internationalRecognition: number;  // 0-1
  domesticLegitimacy: number;        // 0-1 (estimated)
  electionFreedom: number;           // 0-1 (Freedom House style)

  // Transition state
  transitionStage: TransitionStage;
  lastTransition?: {
    type: TransitionType;
    date: string;
    description: string;
  };

  // Stability indicators
  stabilityScore: number;            // 0-1
  coupRisk: number;                  // 0-1
  civilWarRisk: number;              // 0-1

  // Effective dates
  effectiveDate: string;
  endDate?: string;
}

export interface MilitaryAsset {
  id: string;

  // Location
  position: { lat: number; lon: number };
  areaOfOperations?: {
    type: 'circle' | 'polygon';
    coordinates: number[] | number[][];  // [lat, lon, radius] or polygon points
  };

  // Unit identification
  designation: string;               // e.g., "3rd Armored Division"
  branch: MilitaryBranch;
  unitType: UnitType;
  unitSize: UnitSize;

  // Affiliation
  nation: string;                    // ISO code
  disposition: ForceDisposition;

  // Capability
  strength: number;                  // Personnel count (estimated)
  equipment: {
    type: string;
    count: number;
    variant?: string;
  }[];
  capabilities: UnitType[];          // Special capabilities
  readiness: number;                 // 0-1

  // Operational status
  operationalStatus: 'active' | 'reserve' | 'training' | 'deployed' | 'withdrawn';
  currentMission?: string;

  // Threat assessment
  threatLevel: ThreatLevel;

  // Temporal
  lastUpdated: string;
  confidence: number;                // 0-1, intelligence confidence
}

export interface ContestedZone {
  id: string;
  name: string;

  // Area definition
  type: 'point' | 'line' | 'polygon';
  coordinates: number[] | number[][] | number[][][];

  // Contest details
  contestType: 'active_combat' | 'ceasefire' | 'frozen' | 'hot' | 'low_intensity';
  primaryParties: string[];          // ISO codes

  // Intensity
  intensity: ThreatLevel;
  casualtyRate: 'none' | 'light' | 'moderate' | 'heavy' | 'severe';
  civilianImpact: 'minimal' | 'moderate' | 'severe' | 'catastrophic';

  // Temporal
  startDate: string;
  effectiveDate: string;             // Current assessment date
}

// ============================================
// VISUALIZATION STYLING
// ============================================

export interface TerritorialStyle {
  // Base fill (transparent overlay)
  fillColor: string;                 // Hex color
  fillOpacity: number;               // 0-1
  fillPattern?: 'solid' | 'stripes' | 'dots' | 'crosshatch' | 'diagonal';
  patternColor?: string;
  patternSpacing?: number;

  // Border
  strokeColor: string;
  strokeWidth: number;
  strokeStyle: 'solid' | 'dashed' | 'dotted' | 'dash-dot';

  // Labels
  labelText?: string;
  labelColor?: string;
}

export interface MilitarySymbol {
  // APP-6 style symbol components
  frame: 'friendly' | 'hostile' | 'neutral' | 'unknown';
  icon: string;                      // Icon identifier
  modifier1?: string;                // Top modifier
  modifier2?: string;                // Bottom modifier
  echelon?: UnitSize;                // Unit size indicator

  // Position
  position: { lat: number; lon: number };
  rotation?: number;                 // Degrees

  // Style overrides
  color?: string;
  size?: number;
}

// ============================================
// STYLE DEFINITIONS
// ============================================

export const SOVEREIGNTY_STYLES: Record<SovereigntyStatus, Partial<TerritorialStyle>> = {
  recognized: {
    fillOpacity: 0.1,
    fillPattern: 'solid',
    strokeStyle: 'solid',
    strokeWidth: 2,
  },
  disputed: {
    fillOpacity: 0.3,
    fillPattern: 'diagonal',
    strokeStyle: 'dashed',
    strokeWidth: 2,
  },
  unrecognized: {
    fillOpacity: 0.2,
    fillPattern: 'dots',
    strokeStyle: 'dotted',
    strokeWidth: 1,
  },
  annexed: {
    fillOpacity: 0.4,
    fillPattern: 'crosshatch',
    strokeStyle: 'solid',
    strokeWidth: 3,
  },
  occupied: {
    fillOpacity: 0.35,
    fillPattern: 'stripes',
    strokeStyle: 'dashed',
    strokeWidth: 2,
  },
  administered: {
    fillOpacity: 0.2,
    fillPattern: 'solid',
    strokeStyle: 'dash-dot',
    strokeWidth: 2,
  },
  protectorate: {
    fillOpacity: 0.15,
    fillPattern: 'solid',
    strokeStyle: 'dotted',
    strokeWidth: 1,
  },
  mandate: {
    fillOpacity: 0.2,
    fillPattern: 'diagonal',
    strokeStyle: 'dotted',
    strokeWidth: 1,
  },
  colonial: {
    fillOpacity: 0.25,
    fillPattern: 'stripes',
    strokeStyle: 'solid',
    strokeWidth: 2,
  },
};

export const DISPOSITION_COLORS: Record<ForceDisposition, string> = {
  friendly: '#3B82F6',    // Blue
  hostile: '#EF4444',     // Red
  neutral: '#22C55E',     // Green
  unknown: '#F59E0B',     // Yellow/Amber
};

export const THREAT_COLORS: Record<ThreatLevel, string> = {
  none: '#22C55E',        // Green
  low: '#84CC16',         // Lime
  moderate: '#F59E0B',    // Amber
  elevated: '#F97316',    // Orange
  high: '#EF4444',        // Red
  critical: '#7C2D12',    // Dark red
};

export const UNIT_ICONS: Record<UnitType, string> = {
  infantry: 'X',                    // Crossed rifles
  armor: '/',                       // Tank track
  mechanized: '/X',                 // Combined
  artillery: '.',                   // Dot (shell)
  rocket_artillery: '...',          // Multiple dots
  air_defense: '^',                 // Upward arrow
  engineer: 'E',                    // Engineer
  reconnaissance: 'R',              // Recon
  logistics: 'LOG',                 // Supply
  fighter: '=',                     // Wings
  strike: '=v',                     // Wings + bomb
  bomber: '=B',                     // Wings + B
  transport: '=T',                  // Wings + T
  helicopter_attack: '+v',          // Rotor + weapon
  helicopter_transport: '+T',       // Rotor + T
  uav: '-=',                        // Small wings
  awacs: '=O',                      // Wings + radar
  carrier: 'CV',                    // Carrier
  submarine: 'SS',                  // Sub
  surface_combatant: 'DD',          // Destroyer
  amphibious: 'LA',                 // Landing
  cyber: 'CY',                      // Cyber
  electronic_warfare: 'EW',         // EW
  nuclear: 'N',                     // Nuclear
  cbrn: 'CB',                       // CBRN
  special_operations: 'SF',         // Special Forces
};

export const SIZE_MODIFIERS: Record<UnitSize, string> = {
  team: '',                         // No modifier
  squad: '.',                       // One dot
  platoon: '..',                    // Two dots
  company: '|',                     // One bar
  battalion: '||',                  // Two bars
  regiment: '|||',                  // Three bars
  brigade: 'X',                     // One X
  division: 'XX',                   // Two X
  corps: 'XXX',                     // Three X
  army: 'XXXX',                     // Four X
  theater: 'XXXXX',                 // Five X
};

// ============================================
// HISTORICAL EXAMPLES
// ============================================

export const HISTORICAL_EXAMPLES: Record<string, TerritorialStatus> = {
  // Crimea 2014-present
  'crimea_2014': {
    id: 'crimea-2014',
    code: 'UA-43',                  // Ukraine's Crimea code
    name: 'Crimea',
    alternateNames: ['Republic of Crimea', 'Autonomous Republic of Crimea', 'Krym'],
    bounds: { north: 46.2, south: 44.4, east: 36.7, west: 32.5 },
    centroid: { lat: 45.3, lon: 34.1 },
    legalSovereign: 'UA',
    legalSovereignName: 'Ukraine',
    deFactoController: 'RU',
    deFactoControllerName: 'Russia',
    sovereigntyStatus: 'occupied',
    controlStatus: 'occupied',
    isDisputed: true,
    isContested: true,
    isOccupied: true,
    claims: [
      {
        claimantCode: 'UA',
        claimantName: 'Ukraine',
        claimType: 'recognized',
        claimBasis: 'Internationally recognized borders, Budapest Memorandum',
        internationalSupport: 0.87,  // Most UN members
        effectiveControl: false,
        controlPercentage: 0,
      },
      {
        claimantCode: 'RU',
        claimantName: 'Russia',
        claimType: 'annexed',
        claimBasis: '2014 referendum (disputed), historical claims',
        internationalSupport: 0.05,  // Only a few states recognize
        effectiveControl: true,
        controlPercentage: 100,
      },
    ],
    disputes: [
      {
        type: 'sovereignty',
        parties: ['UA', 'RU'],
        description: 'Russian annexation following 2014 invasion',
        startDate: '2014-02-27',
        ongoingConflict: true,
        internationalCases: ['UN GA Resolution 68/262', 'ICJ Ukraine v. Russia'],
      },
    ],
    effectiveDate: '2014-03-18',
    notes: 'Annexed by Russia following 2014 military intervention. Not recognized by most of international community.',
  },

  // Iraq under US occupation 2003-2011
  'iraq_2003': {
    id: 'iraq-2003',
    code: 'IQ',
    name: 'Iraq',
    alternateNames: ['Republic of Iraq'],
    bounds: { north: 37.4, south: 29.1, east: 48.6, west: 38.8 },
    centroid: { lat: 33.2, lon: 43.7 },
    legalSovereign: 'IQ',
    legalSovereignName: 'Iraq',
    deFactoController: 'US',        // Coalition Provisional Authority
    deFactoControllerName: 'United States (CPA)',
    sovereigntyStatus: 'occupied',
    controlStatus: 'partial',
    isDisputed: false,
    isContested: true,
    isOccupied: true,
    claims: [
      {
        claimantCode: 'IQ',
        claimantName: 'Iraq',
        claimType: 'recognized',
        claimBasis: 'Internationally recognized state',
        internationalSupport: 1.0,
        effectiveControl: false,
        controlPercentage: 0,
      },
    ],
    disputes: [
      {
        type: 'sovereignty',
        parties: ['IQ', 'US'],
        description: 'US-led occupation following 2003 invasion',
        startDate: '2003-03-20',
        ongoingConflict: true,
        internationalCases: ['UNSCR 1483'],
      },
    ],
    effectiveDate: '2003-05-01',
    endDate: '2004-06-28',          // Handover to Iraqi government
    notes: 'Coalition Provisional Authority period. Sovereignty nominally transferred June 28, 2004.',
  },

  // Kashmir - multi-party dispute
  'kashmir': {
    id: 'kashmir',
    code: 'DISP-KAS',
    name: 'Kashmir',
    alternateNames: ['Jammu and Kashmir', 'Azad Kashmir', 'Gilgit-Baltistan', 'Aksai Chin'],
    bounds: { north: 37.1, south: 32.2, east: 80.3, west: 73.3 },
    centroid: { lat: 34.5, lon: 76.0 },
    legalSovereign: 'DISP',         // Disputed
    legalSovereignName: 'Disputed',
    deFactoController: 'MULT',      // Multiple controllers
    deFactoControllerName: 'Multiple parties',
    sovereigntyStatus: 'disputed',
    controlStatus: 'contested',
    isDisputed: true,
    isContested: true,
    isOccupied: false,
    claims: [
      {
        claimantCode: 'IN',
        claimantName: 'India',
        claimType: 'recognized',
        claimBasis: 'Instrument of Accession 1947',
        internationalSupport: 0.4,
        effectiveControl: true,
        controlPercentage: 45,       // Jammu & Kashmir, Ladakh
      },
      {
        claimantCode: 'PK',
        claimantName: 'Pakistan',
        claimType: 'disputed',
        claimBasis: 'Two-Nation Theory, UNSCR plebiscite demand',
        internationalSupport: 0.3,
        effectiveControl: true,
        controlPercentage: 35,       // Azad Kashmir, Gilgit-Baltistan
      },
      {
        claimantCode: 'CN',
        claimantName: 'China',
        claimType: 'disputed',
        claimBasis: 'Historical claims, 1962 war gains',
        internationalSupport: 0.1,
        effectiveControl: true,
        controlPercentage: 20,       // Aksai Chin
      },
    ],
    disputes: [
      {
        type: 'territorial',
        parties: ['IN', 'PK', 'CN'],
        description: 'Ongoing territorial dispute since 1947 partition',
        startDate: '1947-10-22',
        ongoingConflict: true,
        internationalCases: ['UNSCR 47', 'UNSCR 80', 'Simla Agreement 1972'],
      },
    ],
    effectiveDate: '1947-10-22',
    notes: 'Complex three-way dispute. Line of Control between India/Pakistan. LAC between India/China.',
  },
};

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Get territorial status for a given date
 */
export function getStatusAtDate(
  statuses: TerritorialStatus[],
  date: string
): TerritorialStatus | undefined {
  const targetDate = new Date(date);

  return statuses.find(s => {
    const effective = new Date(s.effectiveDate);
    const end = s.endDate ? new Date(s.endDate) : new Date();
    return targetDate >= effective && targetDate <= end;
  });
}

/**
 * Get style for territory based on its status
 */
export function getTerrritorialStyle(
  status: TerritorialStatus,
  controllerColor: string
): TerritorialStyle {
  const baseStyle = SOVEREIGNTY_STYLES[status.sovereigntyStatus];

  return {
    fillColor: controllerColor,
    fillOpacity: baseStyle.fillOpacity || 0.2,
    fillPattern: baseStyle.fillPattern || 'solid',
    strokeColor: status.isContested ? THREAT_COLORS.high : controllerColor,
    strokeWidth: baseStyle.strokeWidth || 2,
    strokeStyle: baseStyle.strokeStyle || 'solid',
  };
}

/**
 * Generate military symbol for map display
 */
export function generateMilitarySymbol(asset: MilitaryAsset): MilitarySymbol {
  return {
    frame: asset.disposition,
    icon: UNIT_ICONS[asset.unitType],
    echelon: asset.unitSize,
    modifier1: SIZE_MODIFIERS[asset.unitSize],
    modifier2: asset.capabilities.includes('nuclear') ? 'N' : undefined,
    position: asset.position,
    color: DISPOSITION_COLORS[asset.disposition],
  };
}

/**
 * Calculate contested zone intensity based on military assets
 */
export function calculateContestIntensity(
  zone: ContestedZone,
  assets: MilitaryAsset[]
): ThreatLevel {
  // Count opposing forces in zone
  const hostileCount = assets.filter(a =>
    a.disposition === 'hostile' && isInZone(a.position, zone)
  ).length;

  const friendlyCount = assets.filter(a =>
    a.disposition === 'friendly' && isInZone(a.position, zone)
  ).length;

  const totalStrength = assets
    .filter(a => isInZone(a.position, zone))
    .reduce((sum, a) => sum + a.strength, 0);

  if (totalStrength === 0) return 'none';
  if (totalStrength < 1000) return 'low';
  if (totalStrength < 10000) return 'moderate';
  if (totalStrength < 50000) return 'elevated';
  if (totalStrength < 100000) return 'high';
  return 'critical';
}

/**
 * Check if position is within zone bounds
 */
function isInZone(
  position: { lat: number; lon: number },
  zone: ContestedZone
): boolean {
  if (zone.type === 'point') {
    const [lat, lon, radius = 0] = zone.coordinates as number[];
    const distance = Math.sqrt(
      Math.pow(position.lat - lat, 2) +
      Math.pow(position.lon - lon, 2)
    );
    return distance <= radius;
  }
  // Simplified polygon check - would use proper geo library in production
  return true; // Placeholder
}

/**
 * Generate temporal status timeline
 */
export interface StatusTimeline {
  territory: string;
  events: {
    date: string;
    type: TransitionType | 'status_change';
    from: Partial<TerritorialStatus | GovernmentStatus>;
    to: Partial<TerritorialStatus | GovernmentStatus>;
    description: string;
  }[];
}

export function generateTimeline(
  statuses: TerritorialStatus[],
  governments: GovernmentStatus[]
): StatusTimeline {
  const events: StatusTimeline['events'] = [];

  // Sort by date
  const sortedStatuses = [...statuses].sort(
    (a, b) => new Date(a.effectiveDate).getTime() - new Date(b.effectiveDate).getTime()
  );

  for (let i = 1; i < sortedStatuses.length; i++) {
    const prev = sortedStatuses[i - 1];
    const curr = sortedStatuses[i];

    events.push({
      date: curr.effectiveDate,
      type: 'status_change',
      from: {
        sovereigntyStatus: prev.sovereigntyStatus,
        deFactoController: prev.deFactoController,
      },
      to: {
        sovereigntyStatus: curr.sovereigntyStatus,
        deFactoController: curr.deFactoController,
      },
      description: `Control changed from ${prev.deFactoControllerName} to ${curr.deFactoControllerName}`,
    });
  }

  return {
    territory: statuses[0]?.code || '',
    events,
  };
}

// ============================================
// LAYER RENDERING HELPERS
// ============================================

export interface MapOverlay {
  id: string;
  type: 'territory' | 'asset' | 'zone' | 'line';
  priority: number;              // Render order
  visible: boolean;
  data: TerritorialStatus | MilitaryAsset | ContestedZone;
  style: Partial<TerritorialStyle | MilitarySymbol>;
}

/**
 * Generate all overlays for a region at a point in time
 */
export function generateOverlays(
  territories: TerritorialStatus[],
  assets: MilitaryAsset[],
  zones: ContestedZone[],
  asOfDate: string
): MapOverlay[] {
  const overlays: MapOverlay[] = [];
  const date = new Date(asOfDate);

  // Add territorial overlays (lowest priority - background)
  territories.forEach((t, i) => {
    const effectiveDate = new Date(t.effectiveDate);
    const endDate = t.endDate ? new Date(t.endDate) : new Date();

    if (date >= effectiveDate && date <= endDate) {
      overlays.push({
        id: `territory-${t.id}`,
        type: 'territory',
        priority: 10 + i,
        visible: true,
        data: t,
        style: SOVEREIGNTY_STYLES[t.sovereigntyStatus],
      });
    }
  });

  // Add contested zones (medium priority)
  zones.forEach((z, i) => {
    overlays.push({
      id: `zone-${z.id}`,
      type: 'zone',
      priority: 50 + i,
      visible: true,
      data: z,
      style: {
        fillColor: THREAT_COLORS[z.intensity],
        fillOpacity: 0.4,
        fillPattern: 'crosshatch',
        strokeColor: THREAT_COLORS[z.intensity],
        strokeWidth: 3,
        strokeStyle: 'dashed',
      },
    });
  });

  // Add military assets (highest priority - top)
  assets.forEach((a, i) => {
    overlays.push({
      id: `asset-${a.id}`,
      type: 'asset',
      priority: 100 + i,
      visible: true,
      data: a,
      style: generateMilitarySymbol(a),
    });
  });

  return overlays.sort((a, b) => a.priority - b.priority);
}
