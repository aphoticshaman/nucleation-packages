/**
 * Global Flashpoints - Comprehensive Threat Analysis
 *
 * Covers all major global instability vectors:
 * - Regional conflicts (Middle East, Europe, Africa, Americas)
 * - Domestic instability (polarization, separatism, civil unrest)
 * - Transnational threats (cartels, terrorism, organized crime)
 * - Great power competition
 * - Intelligence/security dynamics
 */

import type { TerritorialDispute, ConditionalThreatAssessment, BorderDomain } from './territorial-status';

// ============================================
// MIDDLE EAST FLASHPOINTS
// ============================================

export const MIDDLE_EAST_DISPUTES: Record<string, TerritorialDispute> = {
  // Israel-Palestine
  'israel_palestine': {
    id: 'israel-palestine',
    name: 'Israel-Palestine Conflict',
    region: 'Levant',
    domain: 'land',
    parties: [
      {
        code: 'IL',
        name: 'Israel',
        claimBasis: 'UN Partition 1947, 1948 War, 1967 War conquests, security requirements',
        controlPercentage: 85,
        militaryPresence: 'dominant',
      },
      {
        code: 'PS',
        name: 'Palestinian Authority / Hamas',
        claimBasis: 'Indigenous population, UN resolutions, 1967 borders',
        controlPercentage: 15,
        militaryPresence: 'moderate',
      },
    ],
    status: 'active_conflict',
    escalationRisk: 0.95,
    strategicImportance: 0.90,
    resourcesAtStake: [
      'Jerusalem (religious significance)',
      'Water resources',
      'Regional stability',
      'US credibility',
      'Arab-Israeli normalization',
    ],
    internationalLaw: {
      relevantTreaties: ['Oslo Accords 1993/1995', 'Camp David 1978'],
      icjRulings: ['2004 Wall Advisory Opinion', '2024 Occupation Advisory Opinion'],
      unResolutions: ['UNSCR 242', 'UNSCR 338', 'UNSCR 2334'],
      status: 'contested',
    },
    recentIncidents: [
      { date: '2023-10-07', description: 'Hamas attack on Israel, 1200 killed, hostages taken', severity: 'critical' },
      { date: '2023-10-27', description: 'Israeli ground invasion of Gaza begins', severity: 'critical' },
      { date: '2024-05-06', description: 'Rafah operation begins despite international opposition', severity: 'critical' },
    ],
    analystAssessment: `
ACTIVE WAR ZONE - Ongoing conflict since October 2023.
No viable two-state solution visible. Gaza devastation unprecedented.
Key dynamics:
- Hamas-Israel cycle appears unbreakable
- West Bank settler expansion continues
- Regional normalization (Abraham Accords) on hold
- Iran-backed axis (Hezbollah, Houthis) activated
- US domestic politics constrains policy options
Scenarios: Prolonged occupation, regional war, or international intervention.
    `,
    lastUpdated: '2024-12-01',
  },

  // Israel-Hezbollah/Lebanon
  'israel_lebanon': {
    id: 'israel-lebanon',
    name: 'Israel-Lebanon/Hezbollah',
    region: 'Levant',
    domain: 'land',
    parties: [
      {
        code: 'IL',
        name: 'Israel',
        claimBasis: 'Security requirements, Shebaa Farms dispute',
        controlPercentage: 0,
        militaryPresence: 'heavy', // Border
      },
      {
        code: 'LB',
        name: 'Lebanon/Hezbollah',
        claimBasis: 'Shebaa Farms, prisoner exchanges, resistance to occupation',
        controlPercentage: 100,
        militaryPresence: 'heavy',
      },
    ],
    status: 'high_tension',
    escalationRisk: 0.75,
    strategicImportance: 0.80,
    resourcesAtStake: [
      'Shebaa Farms',
      'Maritime gas fields (Karish)',
      'Regional stability',
      'Iran deterrence',
    ],
    internationalLaw: {
      relevantTreaties: ['UNSCR 1701 (2006 ceasefire)'],
      icjRulings: [],
      unResolutions: ['UNSCR 1559', 'UNSCR 1701'],
      status: 'contested',
    },
    recentIncidents: [
      { date: '2024-09-17', description: 'Pager attack on Hezbollah operatives', severity: 'critical' },
      { date: '2024-09-27', description: 'Nasrallah killed in Israeli airstrike', severity: 'critical' },
      { date: '2024-10-01', description: 'Israeli ground operation in southern Lebanon', severity: 'critical' },
    ],
    analystAssessment: `
ACTIVE CONFLICT - Low-intensity war since October 2023, escalated September 2024.
Hezbollah arsenal: 150,000+ rockets, precision-guided missiles, drones.
2006 war template inadequate - both sides much stronger now.
Hezbollah decapitation (Nasrallah death) may not prevent retaliation.
Iran's response calculation is key variable.
Full war would devastate both countries and potentially draw in Iran.
    `,
    lastUpdated: '2024-12-01',
  },

  // Israel-Iran
  'israel_iran': {
    id: 'israel-iran',
    name: 'Israel-Iran Shadow War',
    region: 'Middle East',
    domain: 'airspace', // Primarily conducted via air/cyber
    parties: [
      {
        code: 'IL',
        name: 'Israel',
        claimBasis: 'Existential threat from Iranian nuclear program and proxies',
        controlPercentage: 0,
        militaryPresence: 'none', // In Iran
      },
      {
        code: 'IR',
        name: 'Iran',
        claimBasis: 'Resistance to Zionism, regional hegemony, nuclear rights',
        controlPercentage: 0,
        militaryPresence: 'none', // Direct
      },
    ],
    status: 'high_tension',
    escalationRisk: 0.70,
    strategicImportance: 0.95,
    resourcesAtStake: [
      'Iranian nuclear program',
      'Regional proxy networks',
      'Oil transit (Strait of Hormuz)',
      'Global energy prices',
    ],
    internationalLaw: {
      relevantTreaties: ['NPT', 'JCPOA (defunct)'],
      icjRulings: [],
      unResolutions: ['Multiple Iran sanctions resolutions'],
      status: 'contested',
    },
    recentIncidents: [
      { date: '2024-04-13', description: 'Iranian drone/missile attack on Israel (Operation True Promise)', severity: 'critical' },
      { date: '2024-04-19', description: 'Israeli strike on Isfahan (limited response)', severity: 'serious' },
      { date: '2024-10-01', description: 'Iranian ballistic missile attack (Operation True Promise II)', severity: 'critical' },
    ],
    analystAssessment: `
First direct Iran-Israel military exchanges in 2024 crossed previous red lines.
Iran nuclear breakout time now ~1-2 weeks for weapons-grade material.
Israel has demonstrated ability to strike deep inside Iran.
US torn between restraining Israel and deterring Iran.
Full war would collapse oil markets, trigger global recession.
Neither side wants full escalation but miscalculation risk is HIGH.
    `,
    lastUpdated: '2024-12-01',
  },

  // Yemen - Houthis
  'yemen_houthis': {
    id: 'yemen-houthis',
    name: 'Yemen / Houthi Insurgency',
    region: 'Arabian Peninsula',
    domain: 'land',
    parties: [
      {
        code: 'YE',
        name: 'Yemen (Internationally Recognized)',
        claimBasis: 'UN recognition, Hadi government legitimacy',
        controlPercentage: 30,
        militaryPresence: 'moderate',
      },
      {
        code: 'YE-HTH',
        name: 'Houthis (Ansar Allah)',
        claimBasis: 'Revolutionary legitimacy, Zaydi representation',
        controlPercentage: 70,
        militaryPresence: 'heavy',
      },
      {
        code: 'SA',
        name: 'Saudi Arabia',
        claimBasis: 'Border security, restore legitimate government',
        controlPercentage: 0,
        militaryPresence: 'heavy', // Air campaign
      },
    ],
    status: 'active_conflict',
    escalationRisk: 0.65,
    strategicImportance: 0.75,
    resourcesAtStake: [
      'Bab el-Mandeb Strait control',
      'Global shipping routes',
      'Saudi security',
      'Iranian influence expansion',
    ],
    internationalLaw: {
      relevantTreaties: [],
      icjRulings: [],
      unResolutions: ['UNSCR 2216'],
      status: 'contested',
    },
    recentIncidents: [
      { date: '2024-01-11', description: 'US/UK strikes on Houthi targets after Red Sea attacks', severity: 'serious' },
      { date: '2024-03-06', description: 'Houthi attack sinks cargo ship (first since 2024 campaign)', severity: 'serious' },
      { date: '2024-06-12', description: 'Continued attacks on commercial shipping', severity: 'moderate' },
    ],
    analystAssessment: `
Houthis have disrupted ~15% of global shipping through Red Sea attacks.
US/UK strikes have not degraded Houthi capabilities significantly.
Iran provides missiles, drones, targeting data.
Saudi Arabia seeking exit but unwilling to accept Houthi victory.
Humanitarian catastrophe - worst in world by UN metrics.
Resolution requires Iran deal or Saudi capitulation.
    `,
    lastUpdated: '2024-12-01',
  },

  // Syria
  'syria': {
    id: 'syria',
    name: 'Syrian Civil War (Frozen)',
    region: 'Levant',
    domain: 'land',
    parties: [
      {
        code: 'SY',
        name: 'Assad Government',
        claimBasis: 'Incumbent government, Russian/Iranian backing',
        controlPercentage: 70,
        militaryPresence: 'heavy',
      },
      {
        code: 'SY-OPP',
        name: 'Opposition/SDF',
        claimBasis: 'Democratic legitimacy, Kurdish autonomy',
        controlPercentage: 25,
        militaryPresence: 'moderate',
      },
      {
        code: 'TR',
        name: 'Turkey',
        claimBasis: 'Border security, anti-PKK operations',
        controlPercentage: 5,
        militaryPresence: 'heavy',
      },
    ],
    status: 'frozen',
    escalationRisk: 0.40,
    strategicImportance: 0.70,
    resourcesAtStake: [
      'Iranian land bridge to Lebanon',
      'Russian Mediterranean base',
      'Kurdish autonomy precedent',
      'Refugee flows to Europe',
    ],
    internationalLaw: {
      relevantTreaties: [],
      icjRulings: [],
      unResolutions: ['UNSCR 2254'],
      status: 'contested',
    },
    recentIncidents: [
      { date: '2024-01-20', description: 'Israeli strikes on IRGC targets in Damascus', severity: 'moderate' },
    ],
    analystAssessment: `
Frozen conflict - Assad won but can't control all territory.
Multiple foreign forces: Russia, Iran, Turkey, US (small presence).
Israeli strikes continue targeting Iranian assets.
Normalization with Arab states proceeding slowly.
Kurdish autonomy question unresolved.
Reconstruction blocked by sanctions.
    `,
    lastUpdated: '2024-12-01',
  },

  // Gulf tensions
  'gulf_iran': {
    id: 'gulf-iran',
    name: 'Gulf States - Iran Rivalry',
    region: 'Persian Gulf',
    domain: 'maritime',
    parties: [
      {
        code: 'IR',
        name: 'Iran',
        claimBasis: 'Regional hegemon, Shia leadership, anti-Western resistance',
        controlPercentage: 50,
        militaryPresence: 'heavy',
      },
      {
        code: 'SA',
        name: 'Saudi Arabia (GCC)',
        claimBasis: 'Arab leadership, Sunni authority, oil security',
        controlPercentage: 50,
        militaryPresence: 'heavy',
      },
    ],
    status: 'moderate_tension',
    escalationRisk: 0.45,
    strategicImportance: 0.90,
    resourcesAtStake: [
      'Strait of Hormuz (20% of global oil)',
      'Oil production facilities',
      'Regional influence',
      'Sectarian balance',
    ],
    internationalLaw: {
      relevantTreaties: [],
      icjRulings: [],
      unResolutions: [],
      status: 'ambiguous',
    },
    recentIncidents: [
      { date: '2023-03-10', description: 'Saudi-Iran rapprochement via China mediation', severity: 'moderate' },
      { date: '2019-09-14', description: 'Abqaiq-Khurais attack (Iran-linked) on Saudi oil', severity: 'critical' },
    ],
    analystAssessment: `
2023 China-brokered rapprochement reduced direct tension.
Proxy conflicts continue (Yemen, Iraq, Syria, Lebanon).
Both sides hedging - Saudi seeking security guarantees, Iran sanctions relief.
Structural rivalry remains - Sunni-Shia, Arab-Persian, monarchy-theocracy.
Oil facilities remain vulnerable to Iranian retaliation if Israel attacks.
    `,
    lastUpdated: '2024-12-01',
  },
};

// ============================================
// EUROPE FLASHPOINTS
// ============================================

export const EUROPE_DISPUTES: Record<string, TerritorialDispute> = {
  // Russia-Ukraine (main theater)
  'ukraine_war': {
    id: 'ukraine-war',
    name: 'Russia-Ukraine War',
    region: 'Eastern Europe',
    domain: 'land',
    parties: [
      {
        code: 'UA',
        name: 'Ukraine',
        claimBasis: '1991 borders, international recognition, self-determination',
        controlPercentage: 82,
        militaryPresence: 'heavy',
      },
      {
        code: 'RU',
        name: 'Russia',
        claimBasis: 'Claimed annexation (Crimea, Donbas, Zaporizhzhia, Kherson), "denazification"',
        controlPercentage: 18,
        militaryPresence: 'heavy',
      },
    ],
    status: 'active_conflict',
    escalationRisk: 0.80,
    strategicImportance: 0.95,
    resourcesAtStake: [
      'European security architecture',
      'Nuclear deterrence credibility',
      'Global grain supply',
      'Energy markets',
      'NATO expansion',
    ],
    internationalLaw: {
      relevantTreaties: ['Budapest Memorandum 1994', 'UN Charter'],
      icjRulings: ['2022 Provisional Measures (Russia must cease)'],
      unResolutions: ['UNGA ES-11/1 (condemning invasion)'],
      status: 'clear', // Invasion clearly illegal
    },
    recentIncidents: [
      { date: '2024-08-06', description: 'Ukrainian incursion into Kursk Oblast', severity: 'critical' },
      { date: '2024-11-19', description: 'US authorizes ATACMS strikes in Russia', severity: 'serious' },
      { date: '2024-11-21', description: 'Russia launches new ICBM-derived missile', severity: 'critical' },
    ],
    analystAssessment: `
LARGEST WAR IN EUROPE SINCE 1945.
Escalation spiral ongoing - Western weapons in Russia, Russian escalation.
Nuclear risk elevated but not imminent (Putin rational actor).
Neither side can achieve maximal goals militarily.
Negotiations stalled - both sides believe time is on their side.
Key variables: US commitment (Trump return?), Russian sustainability, Ukrainian manpower.
    `,
    lastUpdated: '2024-12-01',
  },

  // Russia-NATO Baltic
  'russia_nato_baltic': {
    id: 'russia-nato-baltic',
    name: 'Russia-NATO Baltic Tension',
    region: 'Baltic Sea',
    domain: 'land',
    parties: [
      {
        code: 'RU',
        name: 'Russia',
        claimBasis: 'Security sphere, Kaliningrad access, Russian minorities',
        controlPercentage: 20, // Kaliningrad
        militaryPresence: 'heavy',
      },
      {
        code: 'NATO',
        name: 'NATO (Baltic States)',
        claimBasis: 'Article 5 collective defense, territorial integrity',
        controlPercentage: 80,
        militaryPresence: 'moderate', // Enhanced Forward Presence
      },
    ],
    status: 'high_tension',
    escalationRisk: 0.35,
    strategicImportance: 0.85,
    resourcesAtStake: [
      'Suwalki Gap (NATO-Kaliningrad land bridge)',
      'Baltic Sea control',
      'Article 5 credibility',
      'Russian enclave (Kaliningrad)',
    ],
    internationalLaw: {
      relevantTreaties: ['NATO Treaty Article 5', 'Helsinki Final Act'],
      icjRulings: [],
      unResolutions: [],
      status: 'clear',
    },
    recentIncidents: [
      { date: '2024-06-15', description: 'GPS jamming affecting Baltic aviation', severity: 'moderate' },
      { date: '2024-02-19', description: 'Russian aircraft violates Estonian airspace', severity: 'moderate' },
    ],
    analystAssessment: `
Most likely NATO-Russia direct confrontation zone after Ukraine.
Suwalki Gap (65km Poland-Lithuania corridor) is critical vulnerability.
Baltic states lack strategic depth - would be overrun in days without NATO.
Russia unlikely to attack while committed in Ukraine.
Post-Ukraine: depends on outcome and NATO posture.
    `,
    lastUpdated: '2024-12-01',
  },

  // Serbia-Kosovo
  'serbia_kosovo': {
    id: 'serbia-kosovo',
    name: 'Serbia-Kosovo',
    region: 'Balkans',
    domain: 'land',
    parties: [
      {
        code: 'RS',
        name: 'Serbia',
        claimBasis: 'Territorial integrity, UNSCR 1244, historical/religious claims',
        controlPercentage: 0,
        militaryPresence: 'none', // Except northern municipalities
      },
      {
        code: 'XK',
        name: 'Kosovo',
        claimBasis: '2008 independence declaration, ICJ advisory opinion, US/EU recognition',
        controlPercentage: 95,
        militaryPresence: 'moderate',
      },
    ],
    status: 'moderate_tension',
    escalationRisk: 0.35,
    strategicImportance: 0.50,
    resourcesAtStake: [
      'Kosovo recognition precedent',
      'Serbian EU path',
      'Trepca mines',
      'Regional stability',
    ],
    internationalLaw: {
      relevantTreaties: ['UNSCR 1244'],
      icjRulings: ['2010 Advisory Opinion (independence not illegal)'],
      unResolutions: ['UNSCR 1244'],
      status: 'ambiguous',
    },
    recentIncidents: [
      { date: '2023-09-24', description: 'Banjska attack by Serbian paramilitaries', severity: 'serious' },
      { date: '2024-01-01', description: 'Kosovo mandates euro, Serbian dinar phase-out', severity: 'moderate' },
    ],
    analystAssessment: `
Frozen conflict with periodic escalation.
Northern Kosovo (Serbian majority) is flashpoint.
Serbia won't recognize Kosovo but needs EU path.
Russia uses Serbia for Balkans influence.
Risk: localized violence that draws in NATO/KFOR.
    `,
    lastUpdated: '2024-12-01',
  },
};

// ============================================
// DOMESTIC INSTABILITY TYPES
// ============================================

export type DomesticThreatType =
  | 'polarization'           // Political polarization
  | 'separatism'             // Separatist movements
  | 'ethnic_conflict'        // Ethnic tensions
  | 'religious_conflict'     // Religious tensions
  | 'economic_crisis'        // Economic instability
  | 'institutional_decay'    // Democratic backsliding
  | 'civil_unrest'           // Protests/riots
  | 'terrorism_domestic'     // Domestic terrorism
  | 'organized_crime'        // Crime/cartel influence
  | 'migration_crisis'       // Immigration-related tensions
  | 'succession_crisis';     // Leadership transition issues

export interface DomesticInstability {
  id: string;
  country: string;
  countryCode: string;
  threatTypes: DomesticThreatType[];
  severity: 'low' | 'moderate' | 'elevated' | 'high' | 'critical';
  trajectory: 'improving' | 'stable' | 'deteriorating' | 'crisis';

  // Key indicators
  indicators: {
    name: string;
    value: string;
    trend: 'improving' | 'stable' | 'deteriorating';
  }[];

  // Flashpoints
  flashpoints: {
    name: string;
    description: string;
    riskLevel: 'low' | 'moderate' | 'high';
  }[];

  // Historical context
  recentEvents: {
    date: string;
    event: string;
    impact: 'minor' | 'moderate' | 'major';
  }[];

  analystNotes: string;
  lastUpdated: string;
}

export const DOMESTIC_INSTABILITY: Record<string, DomesticInstability> = {
  // United States
  'usa': {
    id: 'usa-domestic',
    country: 'United States',
    countryCode: 'US',
    threatTypes: ['polarization', 'institutional_decay', 'terrorism_domestic', 'migration_crisis'],
    severity: 'elevated',
    trajectory: 'deteriorating',
    indicators: [
      { name: 'Political Polarization Index', value: 'Historic high', trend: 'deteriorating' },
      { name: 'Trust in Institutions', value: '20% (Congress)', trend: 'deteriorating' },
      { name: 'Political Violence Incidents', value: '+300% since 2016', trend: 'deteriorating' },
      { name: 'Election Denial', value: '30% of population', trend: 'stable' },
    ],
    flashpoints: [
      { name: '2024 Election Aftermath', description: 'Contested results, potential violence', riskLevel: 'high' },
      { name: 'Border Crisis', description: 'Immigration policy as wedge issue', riskLevel: 'moderate' },
      { name: 'January 6 Precedent', description: 'Normalized political violence', riskLevel: 'high' },
      { name: 'State vs Federal Conflicts', description: 'Texas border standoff model', riskLevel: 'moderate' },
    ],
    recentEvents: [
      { date: '2024-07-13', event: 'Trump assassination attempt', impact: 'major' },
      { date: '2024-01-06', event: 'J6 anniversary, ongoing prosecutions', impact: 'moderate' },
      { date: '2023-12-19', event: 'Colorado removes Trump from ballot (later overturned)', impact: 'moderate' },
    ],
    analystNotes: `
US democratic institutions under significant stress.
Two-party system incentivizes zero-sum conflict.
Geographic/cultural sorting creates parallel realities.
Military remains professional/apolitical (key stabilizer).
Most likely scenario: Managed decline, not collapse.
Worst case: Contested 2024 results trigger state-level nullification attempts.
    `,
    lastUpdated: '2024-12-01',
  },

  // United Kingdom
  'uk': {
    id: 'uk-domestic',
    country: 'United Kingdom',
    countryCode: 'GB',
    threatTypes: ['migration_crisis', 'separatism', 'economic_crisis'],
    severity: 'moderate',
    trajectory: 'stable',
    indicators: [
      { name: 'Scottish Independence Support', value: '45-48%', trend: 'stable' },
      { name: 'Immigration Concern Polling', value: 'Top 3 issue', trend: 'deteriorating' },
      { name: 'Cost of Living Crisis', value: 'Real wages down 5%', trend: 'improving' },
    ],
    flashpoints: [
      { name: 'Scottish Independence', description: 'SNP seeking second referendum', riskLevel: 'moderate' },
      { name: 'Northern Ireland Protocol', description: 'Brexit border issues persist', riskLevel: 'low' },
      { name: 'Immigration Riots', description: 'Summer 2024 riots precedent', riskLevel: 'moderate' },
    ],
    recentEvents: [
      { date: '2024-08-03', event: 'Southport riots spread nationally', impact: 'major' },
      { date: '2024-07-04', event: 'Labour wins landslide election', impact: 'moderate' },
    ],
    analystNotes: `
Brexit aftermath continues to create friction.
Immigration became explosive issue in 2024.
Scottish independence not imminent but not resolved.
Northern Ireland stable but fragile.
New Labour government provides stability but faces fiscal constraints.
    `,
    lastUpdated: '2024-12-01',
  },

  // France
  'france': {
    id: 'france-domestic',
    country: 'France',
    countryCode: 'FR',
    threatTypes: ['polarization', 'migration_crisis', 'civil_unrest', 'terrorism_domestic'],
    severity: 'elevated',
    trajectory: 'deteriorating',
    indicators: [
      { name: 'Far-Right Support', value: '30%+ (RN)', trend: 'deteriorating' },
      { name: 'Banlieue Tension', value: 'High', trend: 'stable' },
      { name: 'Yellow Vest Legacy', value: 'Protest culture normalized', trend: 'stable' },
    ],
    flashpoints: [
      { name: 'Parliamentary Deadlock', description: 'No majority, unstable governments', riskLevel: 'high' },
      { name: 'Pension Reform', description: 'Austerity triggers mass protests', riskLevel: 'moderate' },
      { name: 'Banlieue Riots', description: 'Police shooting could trigger 2005-style unrest', riskLevel: 'moderate' },
    ],
    recentEvents: [
      { date: '2024-07-07', event: 'Snap election produces hung parliament', impact: 'major' },
      { date: '2024-12-04', event: 'Government falls via no-confidence vote', impact: 'major' },
      { date: '2023-06-27', event: 'Nationwide riots after police shooting', impact: 'major' },
    ],
    analystNotes: `
French political system in crisis - no stable majority possible.
Macron weakened, RN rising, left fragmented.
Street protest culture makes governance difficult.
Fifth Republic institutions creaking but holding.
2027 election could see RN victory.
    `,
    lastUpdated: '2024-12-01',
  },

  // Germany
  'germany': {
    id: 'germany-domestic',
    country: 'Germany',
    countryCode: 'DE',
    threatTypes: ['polarization', 'migration_crisis', 'economic_crisis'],
    severity: 'moderate',
    trajectory: 'deteriorating',
    indicators: [
      { name: 'AfD Support', value: '20%+ nationally, 30%+ in East', trend: 'deteriorating' },
      { name: 'Industrial Output', value: 'Declining (energy crisis)', trend: 'deteriorating' },
      { name: 'Coalition Stability', value: 'Traffic light collapsed', trend: 'deteriorating' },
    ],
    flashpoints: [
      { name: 'East-West Divide', description: 'AfD dominates eastern states', riskLevel: 'moderate' },
      { name: 'Deindustrialization', description: 'Energy costs driving industry abroad', riskLevel: 'high' },
      { name: 'Migration Policy', description: 'Asylum system under strain', riskLevel: 'moderate' },
    ],
    recentEvents: [
      { date: '2024-11-06', event: 'Traffic light coalition collapses', impact: 'major' },
      { date: '2024-09-01', event: 'AfD wins Thuringia, Saxony elections', impact: 'major' },
    ],
    analystNotes: `
Germany's post-war stability model under strain.
Economic model (cheap Russian gas, China exports) broken.
AfD rise threatens cordon sanitaire.
Early 2025 election likely - CDU/CSU favored but coalition math difficult.
Germany's weakness = European weakness.
    `,
    lastUpdated: '2024-12-01',
  },
};

// ============================================
// TRANSNATIONAL THREATS
// ============================================

export type TransnationalThreatCategory =
  | 'narcotics_trafficking'
  | 'arms_trafficking'
  | 'human_trafficking'
  | 'money_laundering'
  | 'cyber_crime'
  | 'terrorism'
  | 'piracy'
  | 'sanctions_evasion';

export interface TransnationalThreat {
  id: string;
  name: string;
  category: TransnationalThreatCategory;

  // Geographic scope
  primaryRegions: string[];
  transitRoutes: string[];
  destinationMarkets: string[];

  // Key actors
  actors: {
    name: string;
    type: 'cartel' | 'gang' | 'terrorist' | 'state' | 'network' | 'mafia';
    strength: 'weak' | 'moderate' | 'strong' | 'dominant';
    territory?: string;
  }[];

  // Scale
  estimatedAnnualValue: string;
  violenceLevel: 'low' | 'moderate' | 'high' | 'extreme';
  stateCapture: 'none' | 'local' | 'regional' | 'national';

  // Countermeasures
  enforcementAgencies: string[];
  effectivenessRating: number; // 0-1

  analystNotes: string;
  lastUpdated: string;
}

export const TRANSNATIONAL_THREATS: Record<string, TransnationalThreat> = {
  // Mexican Cartels
  'mexican_cartels': {
    id: 'mexican-cartels',
    name: 'Mexican Drug Cartels',
    category: 'narcotics_trafficking',
    primaryRegions: ['Mexico', 'Central America'],
    transitRoutes: ['US-Mexico border', 'Pacific shipping', 'Central American corridor'],
    destinationMarkets: ['United States', 'Canada', 'Europe'],
    actors: [
      { name: 'Sinaloa Cartel', type: 'cartel', strength: 'dominant', territory: 'Sinaloa, Durango, Chihuahua' },
      { name: 'CJNG (Jalisco New Generation)', type: 'cartel', strength: 'dominant', territory: 'Jalisco, expanding nationally' },
      { name: 'Gulf Cartel', type: 'cartel', strength: 'moderate', territory: 'Tamaulipas' },
      { name: 'Los Zetas remnants', type: 'cartel', strength: 'weak', territory: 'Fragmented' },
    ],
    estimatedAnnualValue: '$30-50 billion',
    violenceLevel: 'extreme',
    stateCapture: 'regional',
    enforcementAgencies: ['DEA', 'FBI', 'Mexican Marines', 'National Guard'],
    effectivenessRating: 0.25,
    analystNotes: `
Cartels effectively control significant Mexican territory.
Fentanyl has transformed economics - higher value, easier to produce.
US fentanyl deaths (70,000+/year) drive political pressure.
Cartel fragmentation creates more violence, not less.
State capture at local/state level extensive.
No military solution - demand-side problem.
    `,
    lastUpdated: '2024-12-01',
  },

  // European Organized Crime
  'european_mafia': {
    id: 'european-organized-crime',
    name: 'European Organized Crime Networks',
    category: 'narcotics_trafficking',
    primaryRegions: ['Western Europe', 'Balkans', 'Netherlands', 'Belgium'],
    transitRoutes: ['Rotterdam port', 'Antwerp port', 'Balkan route', 'Morocco-Spain'],
    destinationMarkets: ['EU internal market'],
    actors: [
      { name: "'Ndrangheta", type: 'mafia', strength: 'dominant', territory: 'Calabria, global reach' },
      { name: 'Mocro Maffia', type: 'network', strength: 'strong', territory: 'Netherlands, Belgium' },
      { name: 'Albanian Mafia', type: 'network', strength: 'strong', territory: 'Balkans, UK, EU' },
      { name: 'Camorra', type: 'mafia', strength: 'moderate', territory: 'Naples region' },
    ],
    estimatedAnnualValue: '$15-20 billion (cocaine alone)',
    violenceLevel: 'moderate',
    stateCapture: 'local',
    enforcementAgencies: ['Europol', 'National police forces', 'Eurojust'],
    effectivenessRating: 0.40,
    analystNotes: `
Rotterdam/Antwerp are cocaine gateways to Europe.
'Ndrangheta has become dominant cocaine wholesaler.
Dutch tolerance model created enforcement gaps.
Journalist assassinations (Peter de Vries) show brazenness.
Encrypted phone network takedowns (EncroChat, Sky) major successes.
Balkan route for heroin, synthetic drugs.
    `,
    lastUpdated: '2024-12-01',
  },

  // Global Terrorism
  'global_terrorism': {
    id: 'global-terrorism',
    name: 'Global Jihadist Networks',
    category: 'terrorism',
    primaryRegions: ['Sahel', 'Afghanistan', 'Syria/Iraq', 'Horn of Africa'],
    transitRoutes: ['Sahara', 'Turkey-Syria border', 'Pakistan-Afghanistan'],
    destinationMarkets: ['Western targets', 'Local populations'],
    actors: [
      { name: 'ISIS-Khorasan', type: 'terrorist', strength: 'moderate', territory: 'Afghanistan/Pakistan' },
      { name: 'Al-Qaeda affiliates', type: 'terrorist', strength: 'moderate', territory: 'Sahel, Yemen, Somalia' },
      { name: 'JNIM (Sahel)', type: 'terrorist', strength: 'strong', territory: 'Mali, Burkina Faso, Niger' },
      { name: 'Al-Shabaab', type: 'terrorist', strength: 'strong', territory: 'Somalia, Kenya border' },
    ],
    estimatedAnnualValue: 'N/A (ideology-driven)',
    violenceLevel: 'extreme',
    stateCapture: 'regional',
    enforcementAgencies: ['CIA', 'MI6', 'DGSE', 'Mossad', 'Local CT forces'],
    effectivenessRating: 0.55,
    analystNotes: `
Post-ISIS territorial defeat, threat dispersed but not eliminated.
Sahel is new epicenter - French withdrawal, Russian Wagner entry.
ISIS-K conducted Moscow attack (March 2024).
Lone wolf attacks in West remain persistent threat.
Taliban-Al Qaeda relationship complicates Afghanistan.
Sahel jihadist expansion threatening coastal West Africa.
    `,
    lastUpdated: '2024-12-01',
  },

  // Cyber Crime
  'cyber_crime': {
    id: 'cyber-crime-global',
    name: 'Global Cybercrime Networks',
    category: 'cyber_crime',
    primaryRegions: ['Russia', 'Eastern Europe', 'North Korea', 'China'],
    transitRoutes: ['Internet (global)', 'Cryptocurrency networks'],
    destinationMarkets: ['Global targets'],
    actors: [
      { name: 'LockBit', type: 'network', strength: 'strong', territory: 'Russia-linked' },
      { name: 'Lazarus Group', type: 'state', strength: 'strong', territory: 'North Korea' },
      { name: 'REvil', type: 'network', strength: 'moderate', territory: 'Russia-linked' },
      { name: 'APT groups (various)', type: 'state', strength: 'dominant', territory: 'China, Russia, Iran' },
    ],
    estimatedAnnualValue: '$6+ trillion (global cybercrime costs)',
    violenceLevel: 'low', // Physical
    stateCapture: 'none', // Criminal-state nexus different
    enforcementAgencies: ['FBI Cyber', 'NSA', 'GCHQ', 'Europol EC3'],
    effectivenessRating: 0.30,
    analystNotes: `
Ransomware-as-a-Service model lowered barriers.
Russia provides safe haven for criminals who don't target Russia.
North Korea uses cybercrime for sanctions evasion ($3B+ in crypto theft).
Critical infrastructure attacks (Colonial Pipeline) demonstrated vulnerabilities.
Attribution improving but prosecution difficult across borders.
AI will amplify both offense and defense.
    `,
    lastUpdated: '2024-12-01',
  },
};

// ============================================
// INTELLIGENCE AGENCIES
// ============================================

export interface IntelligenceAgency {
  id: string;
  name: string;
  country: string;
  countryCode: string;
  type: 'foreign_intelligence' | 'domestic_security' | 'signals' | 'military' | 'combined';

  // Capabilities
  capabilities: {
    humint: number;    // Human intelligence 0-1
    sigint: number;    // Signals intelligence 0-1
    cyber: number;     // Cyber operations 0-1
    covert: number;    // Covert action 0-1
    analysis: number;  // Analytical capacity 0-1
  };

  // Global reach
  globalPresence: 'limited' | 'regional' | 'extensive' | 'global';

  // Partnerships
  majorPartnerships: string[];  // Five Eyes, etc.

  // Recent operations (known/alleged)
  knownOperations: {
    name: string;
    type: string;
    date: string;
    target: string;
  }[];

  notes: string;
}

export const INTELLIGENCE_AGENCIES: Record<string, IntelligenceAgency> = {
  // United States
  'cia': {
    id: 'cia',
    name: 'Central Intelligence Agency (CIA)',
    country: 'United States',
    countryCode: 'US',
    type: 'foreign_intelligence',
    capabilities: { humint: 0.95, sigint: 0.70, cyber: 0.85, covert: 0.95, analysis: 0.90 },
    globalPresence: 'global',
    majorPartnerships: ['Five Eyes', 'NATO allies', 'Israel', 'Saudi Arabia', 'Japan'],
    knownOperations: [
      { name: 'Afghanistan withdrawal intel', type: 'collection', date: '2021', target: 'Taliban' },
      { name: 'Nord Stream investigation', type: 'analysis', date: '2022', target: 'Pipeline sabotage' },
    ],
    notes: 'World\'s largest intelligence agency. Strong HUMINT, extensive covert action history.',
  },
  'nsa': {
    id: 'nsa',
    name: 'National Security Agency (NSA)',
    country: 'United States',
    countryCode: 'US',
    type: 'signals',
    capabilities: { humint: 0.20, sigint: 1.0, cyber: 0.95, covert: 0.50, analysis: 0.85 },
    globalPresence: 'global',
    majorPartnerships: ['Five Eyes', 'SIGINT sharing globally'],
    knownOperations: [
      { name: 'Stuxnet', type: 'cyber', date: '2010', target: 'Iran nuclear program' },
      { name: 'Global surveillance (Snowden)', type: 'collection', date: '2013', target: 'Global' },
    ],
    notes: 'World\'s most capable SIGINT agency. Vast collection capabilities revealed by Snowden.',
  },
  'fbi': {
    id: 'fbi',
    name: 'Federal Bureau of Investigation (FBI)',
    country: 'United States',
    countryCode: 'US',
    type: 'domestic_security',
    capabilities: { humint: 0.85, sigint: 0.60, cyber: 0.80, covert: 0.70, analysis: 0.80 },
    globalPresence: 'extensive', // Legal attaches globally
    majorPartnerships: ['Five Eyes domestic services', 'Interpol'],
    knownOperations: [
      { name: 'Counterintelligence (China)', type: 'CI', date: 'ongoing', target: 'Chinese espionage' },
      { name: 'Domestic terrorism', type: 'CT', date: 'ongoing', target: 'Domestic extremists' },
    ],
    notes: 'Dual law enforcement and intelligence. Heavily focused on China threat.',
  },

  // United Kingdom
  'mi6': {
    id: 'mi6',
    name: 'Secret Intelligence Service (MI6/SIS)',
    country: 'United Kingdom',
    countryCode: 'GB',
    type: 'foreign_intelligence',
    capabilities: { humint: 0.90, sigint: 0.50, cyber: 0.75, covert: 0.85, analysis: 0.85 },
    globalPresence: 'global',
    majorPartnerships: ['Five Eyes', 'EU (historical)', 'Middle East partners'],
    knownOperations: [
      { name: 'Skripal case response', type: 'CI', date: '2018', target: 'Russian GRU' },
      { name: 'Christopher Steele dossier', type: 'collection', date: '2016', target: 'Russia/Trump' },
    ],
    notes: 'Historic HUMINT excellence. Strong Middle East networks. Close CIA partnership.',
  },
  'mi5': {
    id: 'mi5',
    name: 'Security Service (MI5)',
    country: 'United Kingdom',
    countryCode: 'GB',
    type: 'domestic_security',
    capabilities: { humint: 0.85, sigint: 0.40, cyber: 0.70, covert: 0.60, analysis: 0.80 },
    globalPresence: 'limited',
    majorPartnerships: ['Five Eyes domestic', 'EU security (historical)'],
    knownOperations: [
      { name: 'Northern Ireland intelligence', type: 'CT', date: 'ongoing', target: 'Dissident republicans' },
      { name: 'Islamist terrorism prevention', type: 'CT', date: 'ongoing', target: 'Jihadist networks' },
    ],
    notes: 'Strong CT capabilities developed during IRA era. Now focused on Islamist and far-right threats.',
  },
  'gchq': {
    id: 'gchq',
    name: 'Government Communications Headquarters (GCHQ)',
    country: 'United Kingdom',
    countryCode: 'GB',
    type: 'signals',
    capabilities: { humint: 0.20, sigint: 0.90, cyber: 0.90, covert: 0.40, analysis: 0.85 },
    globalPresence: 'extensive',
    majorPartnerships: ['Five Eyes', 'Particularly close NSA ties'],
    knownOperations: [
      { name: 'Tempora', type: 'collection', date: '2013', target: 'Global fiber optic tapping' },
      { name: 'Cyber operations vs ISIS', type: 'cyber', date: '2016+', target: 'ISIS networks' },
    ],
    notes: 'World-class SIGINT. Bletchley Park heritage. Strong cyber offensive capability.',
  },

  // Israel
  'mossad': {
    id: 'mossad',
    name: 'Mossad (Institute for Intelligence)',
    country: 'Israel',
    countryCode: 'IL',
    type: 'foreign_intelligence',
    capabilities: { humint: 0.95, sigint: 0.75, cyber: 0.90, covert: 1.0, analysis: 0.85 },
    globalPresence: 'extensive',
    majorPartnerships: ['CIA', 'MI6', 'Gulf states (expanding)'],
    knownOperations: [
      { name: 'Pager attack (Hezbollah)', type: 'covert', date: '2024', target: 'Hezbollah' },
      { name: 'Nasrallah assassination', type: 'covert', date: '2024', target: 'Hezbollah' },
      { name: 'Iranian nuclear scientist assassinations', type: 'covert', date: '2020-2021', target: 'Iran' },
      { name: 'Nuclear archive raid (Tehran)', type: 'covert', date: '2018', target: 'Iran' },
    ],
    notes: 'Exceptionally aggressive covert action. Willing to conduct operations others won\'t. Small but elite.',
  },
  'shin_bet': {
    id: 'shin-bet',
    name: 'Shin Bet (Israel Security Agency)',
    country: 'Israel',
    countryCode: 'IL',
    type: 'domestic_security',
    capabilities: { humint: 0.90, sigint: 0.70, cyber: 0.80, covert: 0.85, analysis: 0.80 },
    globalPresence: 'limited',
    majorPartnerships: ['Domestic focus', 'Palestinian territories'],
    knownOperations: [
      { name: 'West Bank intelligence', type: 'CI/CT', date: 'ongoing', target: 'Palestinian militants' },
      { name: 'Oct 7 failure', type: 'failure', date: '2023', target: 'Hamas (missed attack)' },
    ],
    notes: 'Highly effective internally but Oct 7 was catastrophic intelligence failure.',
  },

  // Russia
  'svr': {
    id: 'svr',
    name: 'Foreign Intelligence Service (SVR)',
    country: 'Russia',
    countryCode: 'RU',
    type: 'foreign_intelligence',
    capabilities: { humint: 0.85, sigint: 0.60, cyber: 0.80, covert: 0.80, analysis: 0.75 },
    globalPresence: 'global',
    majorPartnerships: ['China (limited)', 'Iran', 'Syria'],
    knownOperations: [
      { name: 'SolarWinds hack', type: 'cyber', date: '2020', target: 'US government' },
      { name: 'Illegals program', type: 'HUMINT', date: 'ongoing', target: 'Western countries' },
    ],
    notes: 'KGB successor for foreign intelligence. Strong HUMINT tradition. Cyber capabilities growing.',
  },
  'fsb': {
    id: 'fsb',
    name: 'Federal Security Service (FSB)',
    country: 'Russia',
    countryCode: 'RU',
    type: 'domestic_security',
    capabilities: { humint: 0.80, sigint: 0.75, cyber: 0.85, covert: 0.90, analysis: 0.70 },
    globalPresence: 'regional',
    majorPartnerships: ['CIS countries'],
    knownOperations: [
      { name: 'Navalny poisoning', type: 'covert', date: '2020', target: 'Opposition' },
      { name: 'Skripal poisoning', type: 'covert', date: '2018', target: 'Defectors' },
    ],
    notes: 'KGB domestic successor. Extensive domestic surveillance. Aggressive against perceived enemies.',
  },
  'gru': {
    id: 'gru',
    name: 'Main Intelligence Directorate (GRU)',
    country: 'Russia',
    countryCode: 'RU',
    type: 'military',
    capabilities: { humint: 0.75, sigint: 0.70, cyber: 0.90, covert: 0.85, analysis: 0.65 },
    globalPresence: 'extensive',
    majorPartnerships: ['Military operations globally'],
    knownOperations: [
      { name: 'NotPetya attack', type: 'cyber', date: '2017', target: 'Ukraine (global damage)' },
      { name: 'Election interference (2016)', type: 'influence', date: '2016', target: 'US election' },
      { name: 'Unit 29155 operations', type: 'covert', date: 'various', target: 'European targets' },
    ],
    notes: 'Military intelligence. Very aggressive cyber operations. Unit 29155 linked to assassinations, sabotage.',
  },

  // China
  'mss': {
    id: 'mss',
    name: 'Ministry of State Security (MSS)',
    country: 'China',
    countryCode: 'CN',
    type: 'combined',
    capabilities: { humint: 0.85, sigint: 0.80, cyber: 0.95, covert: 0.70, analysis: 0.80 },
    globalPresence: 'global',
    majorPartnerships: ['Limited formal partnerships'],
    knownOperations: [
      { name: 'OPM hack', type: 'cyber', date: '2015', target: 'US personnel records' },
      { name: 'Thousand Talents recruitment', type: 'HUMINT', date: 'ongoing', target: 'Western researchers' },
      { name: 'Belt and Road intelligence', type: 'collection', date: 'ongoing', target: 'Global' },
    ],
    notes: 'Massive scale operations. Particularly focused on technology theft. Uses diaspora networks.',
  },

  // International
  'interpol': {
    id: 'interpol',
    name: 'INTERPOL',
    country: 'International',
    countryCode: 'INT',
    type: 'combined',
    capabilities: { humint: 0.30, sigint: 0.20, cyber: 0.40, covert: 0.0, analysis: 0.60 },
    globalPresence: 'global',
    majorPartnerships: ['195 member countries'],
    knownOperations: [
      { name: 'Red Notices', type: 'coordination', date: 'ongoing', target: 'Wanted persons globally' },
      { name: 'Operation Lionfish', type: 'coordination', date: '2023', target: 'Drug trafficking' },
    ],
    notes: 'Coordination body, not operational. Red Notice system subject to political abuse. Limited enforcement power.',
  },
};

// ============================================
// HELPER FUNCTIONS
// ============================================

export function getFlashpointsByRegion(region: string): TerritorialDispute[] {
  const allDisputes = [
    ...Object.values(MIDDLE_EAST_DISPUTES),
    ...Object.values(EUROPE_DISPUTES),
  ];
  return allDisputes.filter(d => d.region.toLowerCase().includes(region.toLowerCase()));
}

export function getHighestRiskFlashpoints(limit: number = 10): TerritorialDispute[] {
  const allDisputes = [
    ...Object.values(MIDDLE_EAST_DISPUTES),
    ...Object.values(EUROPE_DISPUTES),
  ];
  return allDisputes
    .sort((a, b) => b.escalationRisk - a.escalationRisk)
    .slice(0, limit);
}

export function getActiveConflicts(): TerritorialDispute[] {
  const allDisputes = [
    ...Object.values(MIDDLE_EAST_DISPUTES),
    ...Object.values(EUROPE_DISPUTES),
  ];
  return allDisputes.filter(d => d.status === 'active_conflict');
}

export function getCountriesByInstability(): DomesticInstability[] {
  return Object.values(DOMESTIC_INSTABILITY)
    .sort((a, b) => {
      const severityOrder = { critical: 5, high: 4, elevated: 3, moderate: 2, low: 1 };
      return severityOrder[b.severity] - severityOrder[a.severity];
    });
}
