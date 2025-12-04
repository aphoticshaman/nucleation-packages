/**
 * Extended Global Flashpoints - Complete Coverage
 *
 * Covers regions not in main global-flashpoints.ts:
 * - Africa (Sahel, Horn, Great Lakes, Southern)
 * - Latin America (Venezuela, Colombia, Central America, Brazil)
 * - Extended Asia (North Korea, Myanmar, Philippines, Indonesia)
 * - Extended domestic instability
 * - More intelligence agencies
 * - International organizations
 */

import type { TerritorialDispute, BorderDomain, RegimeThreatAssessment } from './territorial-status';
import type { DomesticInstability, TransnationalThreat, IntelligenceAgency } from './global-flashpoints';

// ============================================
// AFRICA FLASHPOINTS
// ============================================

export const AFRICA_DISPUTES: Record<string, TerritorialDispute> = {
  // Sahel Crisis
  'sahel_insurgency': {
    id: 'sahel-insurgency',
    name: 'Sahel Jihadist Insurgency',
    region: 'Sahel (West Africa)',
    domain: 'land',
    parties: [
      {
        code: 'ML',
        name: 'Mali (Junta)',
        claimBasis: 'Territorial integrity, sovereignty',
        controlPercentage: 60,
        militaryPresence: 'moderate',
      },
      {
        code: 'BF',
        name: 'Burkina Faso (Junta)',
        claimBasis: 'Territorial integrity',
        controlPercentage: 50,
        militaryPresence: 'moderate',
      },
      {
        code: 'NE',
        name: 'Niger (Junta)',
        claimBasis: 'Territorial integrity',
        controlPercentage: 70,
        militaryPresence: 'moderate',
      },
      {
        code: 'JNIM',
        name: 'JNIM (Al-Qaeda affiliate)',
        claimBasis: 'Islamic governance, anti-colonial',
        controlPercentage: 25,
        militaryPresence: 'heavy',
      },
      {
        code: 'ISIS-S',
        name: 'ISIS-Sahel',
        claimBasis: 'Caliphate, Islamic governance',
        controlPercentage: 15,
        militaryPresence: 'moderate',
      },
    ],
    status: 'active_conflict',
    escalationRisk: 0.70,
    strategicImportance: 0.65,
    resourcesAtStake: [
      'Uranium (Niger - France dependency)',
      'Gold mining',
      'Migration routes to Europe',
      'Regional stability',
      'French/Western influence',
    ],
    internationalLaw: {
      relevantTreaties: [],
      icjRulings: [],
      unResolutions: ['MINUSMA mandate (ended 2023)'],
      status: 'contested',
    },
    recentIncidents: [
      { date: '2023-07-26', description: 'Niger coup - last French ally falls', severity: 'critical' },
      { date: '2024-01-28', description: 'Mali, Niger, Burkina form Alliance of Sahel States', severity: 'serious' },
      { date: '2024-09-17', description: 'Malian army suffers major defeat, 50+ killed', severity: 'serious' },
    ],
    analystAssessment: `
DETERIORATING - Jihadists gaining territory despite (because of?) Russian Wagner presence.
French withdrawal complete. US reducing footprint. Wagner replacing but less capable.
Coups created anti-Western bloc but security situation worsening.
Jihadist expansion toward coastal states (Benin, Togo, Ghana) accelerating.
Humanitarian crisis: millions displaced, food insecurity.
No military solution visible - governance failures at root.
    `,
    lastUpdated: '2024-12-01',
  },

  // Sudan Civil War
  'sudan_civil_war': {
    id: 'sudan-civil-war',
    name: 'Sudan Civil War (SAF vs RSF)',
    region: 'Horn of Africa',
    domain: 'land',
    parties: [
      {
        code: 'SD-SAF',
        name: 'Sudanese Armed Forces (Burhan)',
        claimBasis: 'Legitimate military, state continuity',
        controlPercentage: 45,
        militaryPresence: 'heavy',
      },
      {
        code: 'SD-RSF',
        name: 'Rapid Support Forces (Hemedti)',
        claimBasis: 'Revolutionary legitimacy, anti-Islamist',
        controlPercentage: 40,
        militaryPresence: 'heavy',
      },
    ],
    status: 'active_conflict',
    escalationRisk: 0.80,
    strategicImportance: 0.70,
    resourcesAtStake: [
      'Red Sea access',
      'Nile water control',
      'Gold reserves',
      'Regional stability',
      'Refugee flows (10M+ displaced)',
    ],
    internationalLaw: {
      relevantTreaties: [],
      icjRulings: [],
      unResolutions: ['Various humanitarian resolutions'],
      status: 'contested',
    },
    recentIncidents: [
      { date: '2023-04-15', description: 'War begins - RSF attacks SAF positions', severity: 'critical' },
      { date: '2024-10-20', description: 'RSF controls most of Darfur, ethnic cleansing reports', severity: 'critical' },
      { date: '2024-11-15', description: 'SAF retakes parts of Khartoum', severity: 'serious' },
    ],
    analystAssessment: `
CATASTROPHIC - Worst humanitarian crisis globally (per UN).
25+ million need aid, 10+ million displaced.
RSF backed by UAE, Wagner; SAF by Egypt, Iran.
Ethnic cleansing in Darfur echoing 2003 genocide.
State collapse scenario increasingly likely.
No peace process with traction. Both sides believe they can win.
    `,
    lastUpdated: '2024-12-01',
  },

  // Ethiopia - Tigray and beyond
  'ethiopia_conflicts': {
    id: 'ethiopia-conflicts',
    name: 'Ethiopia Internal Conflicts',
    region: 'Horn of Africa',
    domain: 'land',
    parties: [
      {
        code: 'ET',
        name: 'Ethiopian Federal Government',
        claimBasis: 'Constitutional authority, territorial integrity',
        controlPercentage: 75,
        militaryPresence: 'heavy',
      },
      {
        code: 'ET-TIG',
        name: 'Tigray (TPLF)',
        claimBasis: 'Regional autonomy, self-determination',
        controlPercentage: 5,
        militaryPresence: 'moderate',
      },
      {
        code: 'ET-OLA',
        name: 'Oromo Liberation Army',
        claimBasis: 'Oromo self-determination',
        controlPercentage: 10,
        militaryPresence: 'moderate',
      },
      {
        code: 'ET-AMH',
        name: 'Amhara Fano militias',
        claimBasis: 'Amhara nationalism, land claims',
        controlPercentage: 10,
        militaryPresence: 'moderate',
      },
    ],
    status: 'high_tension',
    escalationRisk: 0.55,
    strategicImportance: 0.70,
    resourcesAtStake: [
      'GERD dam (Nile water)',
      'Regional hegemon status',
      'Red Sea access (Eritrea deal)',
      '120M population stability',
    ],
    internationalLaw: {
      relevantTreaties: ['Pretoria Agreement 2022 (Tigray ceasefire)'],
      icjRulings: [],
      unResolutions: [],
      status: 'contested',
    },
    recentIncidents: [
      { date: '2022-11-02', description: 'Pretoria Agreement ends Tigray war', severity: 'serious' },
      { date: '2024-08-04', description: 'Amhara Fano insurgency intensifies', severity: 'serious' },
      { date: '2024-01-01', description: 'Ethiopia-Somaliland port deal angers Somalia', severity: 'moderate' },
    ],
    analystAssessment: `
Post-Tigray war: 600,000+ dead, ceasefire holding but fragile.
New insurgencies: Amhara Fano, OLA continuing.
Ethiopia seeking Red Sea access via Somaliland - risks Somalia war.
GERD filling continues - Egypt tensions manageable but structural.
Abiy's centralization vs. ethnic federalism tension unresolved.
State fragmentation risk if multiple fronts activate simultaneously.
    `,
    lastUpdated: '2024-12-01',
  },

  // Democratic Republic of Congo
  'drc_eastern': {
    id: 'drc-eastern',
    name: 'Eastern DRC Conflict',
    region: 'Great Lakes Africa',
    domain: 'land',
    parties: [
      {
        code: 'CD',
        name: 'DRC Government (FARDC)',
        claimBasis: 'Territorial integrity',
        controlPercentage: 50,
        militaryPresence: 'heavy',
      },
      {
        code: 'M23',
        name: 'M23 Rebels (Rwanda-backed)',
        claimBasis: 'Tutsi protection, political grievances',
        controlPercentage: 15,
        militaryPresence: 'heavy',
      },
      {
        code: 'ADF',
        name: 'ADF-ISIS',
        claimBasis: 'Islamic governance',
        controlPercentage: 5,
        militaryPresence: 'moderate',
      },
    ],
    status: 'active_conflict',
    escalationRisk: 0.65,
    strategicImportance: 0.75,
    resourcesAtStake: [
      'Coltan (80% of global reserves)',
      'Cobalt (critical minerals)',
      'Gold, diamonds',
      'Regional stability',
    ],
    internationalLaw: {
      relevantTreaties: [],
      icjRulings: [],
      unResolutions: ['MONUSCO mandate'],
      status: 'contested',
    },
    recentIncidents: [
      { date: '2024-02-07', description: 'M23 advances, captures Sake near Goma', severity: 'critical' },
      { date: '2024-11-01', description: 'Ceasefire announced but violations continue', severity: 'moderate' },
    ],
    analystAssessment: `
Decades-old conflict - world's deadliest since WWII (6M+ dead).
M23 resurgence backed by Rwanda (proven by UN experts).
Critical minerals (coltan, cobalt) fund armed groups.
MONUSCO withdrawing - security vacuum concerns.
DRC-Rwanda tensions at highest since 1998.
Risk of interstate war if DRC attacks Rwanda directly.
    `,
    lastUpdated: '2024-12-01',
  },

  // Somalia
  'somalia': {
    id: 'somalia',
    name: 'Somalia - Al-Shabaab Insurgency',
    region: 'Horn of Africa',
    domain: 'land',
    parties: [
      {
        code: 'SO',
        name: 'Federal Government of Somalia',
        claimBasis: 'Internationally recognized government',
        controlPercentage: 40,
        militaryPresence: 'moderate',
      },
      {
        code: 'AL-SHAB',
        name: 'Al-Shabaab',
        claimBasis: 'Islamic governance, Al-Qaeda affiliate',
        controlPercentage: 35,
        militaryPresence: 'heavy',
      },
      {
        code: 'ATMIS',
        name: 'African Union (ATMIS)',
        claimBasis: 'Peace enforcement mandate',
        controlPercentage: 15,
        militaryPresence: 'heavy',
      },
    ],
    status: 'active_conflict',
    escalationRisk: 0.55,
    strategicImportance: 0.60,
    resourcesAtStake: [
      'Gulf of Aden shipping lane',
      'Regional stability',
      'Piracy prevention',
      'Refugee flows',
    ],
    internationalLaw: {
      relevantTreaties: [],
      icjRulings: [],
      unResolutions: ['Various Somalia resolutions'],
      status: 'contested',
    },
    recentIncidents: [
      { date: '2024-08-02', description: 'Al-Shabaab attack on Lido Beach, Mogadishu', severity: 'serious' },
      { date: '2024-06-15', description: 'ATMIS drawdown continues - 2024 deadline', severity: 'moderate' },
    ],
    analystAssessment: `
Al-Shabaab remains Africa's most dangerous terrorist group.
Taxes extortion system gives financial sustainability.
Ethiopia-Somaliland deal threatens to destabilize further.
ATMIS withdrawal creating security gaps.
Clan politics complicate federal-regional dynamics.
No military victory in sight - political settlement needed.
    `,
    lastUpdated: '2024-12-01',
  },

  // Libya
  'libya': {
    id: 'libya',
    name: 'Libya Division',
    region: 'North Africa',
    domain: 'land',
    parties: [
      {
        code: 'LY-GNU',
        name: 'Government of National Unity (Tripoli)',
        claimBasis: 'UN-recognized government',
        controlPercentage: 30,
        militaryPresence: 'moderate',
      },
      {
        code: 'LY-LNA',
        name: 'Libyan National Army (Haftar/East)',
        claimBasis: 'House of Representatives backing',
        controlPercentage: 60,
        militaryPresence: 'heavy',
      },
    ],
    status: 'frozen',
    escalationRisk: 0.40,
    strategicImportance: 0.70,
    resourcesAtStake: [
      'Oil (Africa\'s largest reserves)',
      'Migration gateway to Europe',
      'Arms proliferation source',
      'Regional influence',
    ],
    internationalLaw: {
      relevantTreaties: ['Various UN-brokered agreements'],
      icjRulings: [],
      unResolutions: ['UNSMIL mandate'],
      status: 'contested',
    },
    recentIncidents: [
      { date: '2024-08-26', description: 'Central Bank crisis triggers oil shutdown', severity: 'serious' },
      { date: '2023-09-11', description: 'Derna floods kill 11,000+ - governance failure', severity: 'critical' },
    ],
    analystAssessment: `
Frozen conflict since 2020 ceasefire. No unified government.
Oil production hostage to political disputes.
Foreign forces remain (Turkey, Russia/Wagner, Egypt).
Elections perpetually postponed.
Migration flows continue - EU leverage point.
Haftar aging - succession question looms.
    `,
    lastUpdated: '2024-12-01',
  },

  // Mozambique
  'mozambique_cabo': {
    id: 'mozambique-cabo-delgado',
    name: 'Mozambique - Cabo Delgado Insurgency',
    region: 'Southern Africa',
    domain: 'land',
    parties: [
      {
        code: 'MZ',
        name: 'Mozambique Government',
        claimBasis: 'Territorial integrity',
        controlPercentage: 70,
        militaryPresence: 'heavy',
      },
      {
        code: 'ISIS-MOZ',
        name: 'ISIS-Mozambique (Al-Shabaab local)',
        claimBasis: 'Islamic governance',
        controlPercentage: 20,
        militaryPresence: 'moderate',
      },
      {
        code: 'SAMIM',
        name: 'SADC Mission (Rwanda troops)',
        claimBasis: 'Regional security mandate',
        controlPercentage: 10,
        militaryPresence: 'moderate',
      },
    ],
    status: 'active_conflict',
    escalationRisk: 0.45,
    strategicImportance: 0.60,
    resourcesAtStake: [
      'Natural gas ($50B+ LNG projects)',
      'Regional stability',
      'ISIS expansion in Africa',
    ],
    internationalLaw: {
      relevantTreaties: [],
      icjRulings: [],
      unResolutions: [],
      status: 'ambiguous',
    },
    recentIncidents: [
      { date: '2024-10-24', description: 'Post-election protests, 100+ killed', severity: 'serious' },
      { date: '2024-05-12', description: 'Insurgent attack on Macomia district', severity: 'moderate' },
    ],
    analystAssessment: `
LNG projects (TotalEnergies, Exxon) on hold due to security.
Rwandan troops most effective but expensive.
Insurgency contained but not defeated.
Political crisis (2024 election dispute) compounds security.
Southern Africa's first major jihadist threat.
    `,
    lastUpdated: '2024-12-01',
  },
};

// ============================================
// LATIN AMERICA FLASHPOINTS
// ============================================

export const LATIN_AMERICA_DISPUTES: Record<string, TerritorialDispute> = {
  // Venezuela Crisis - COMPREHENSIVE TRUMP 2.0 UPDATE
  'venezuela': {
    id: 'venezuela',
    name: 'Venezuela-US Confrontation (Trump 2.0)',
    region: 'South America / Caribbean',
    domain: 'land',
    parties: [
      {
        code: 'VE-MAD',
        name: 'Maduro Government',
        claimBasis: 'Constitutional continuity, 2024 election claim',
        controlPercentage: 95,
        militaryPresence: 'dominant',
      },
      {
        code: 'VE-OPP',
        name: 'Opposition (González/Machado)',
        claimBasis: '2024 election victory (documented by vote tallies), democratic legitimacy',
        controlPercentage: 0,
        militaryPresence: 'none',
      },
      {
        code: 'US',
        name: 'United States (Trump Administration)',
        claimBasis: 'Anti-narcoterrorism, Monroe Doctrine, regime change',
        controlPercentage: 0,
        militaryPresence: 'heavy', // Operation Southern Spear
      },
    ],
    status: 'active_conflict', // Upgraded from high_tension
    escalationRisk: 0.85, // CRITICAL - active military operations
    strategicImportance: 0.85,
    resourcesAtStake: [
      'Oil (world\'s largest reserves - 304B barrels)',
      'Regional democracy precedent',
      'Migration crisis (7.7M+ fled)',
      'US-China-Russia competition in Western Hemisphere',
      'Monroe Doctrine revival test',
      'Drug trafficking routes',
    ],
    internationalLaw: {
      relevantTreaties: ['OAS Democratic Charter', 'UN Charter Art. 2(4)'],
      icjRulings: [],
      unResolutions: ['Secretary-General: strikes "not compatible with international law"'],
      status: 'contested',
    },
    recentIncidents: [
      // 2024 Events
      { date: '2024-07-28', description: 'Election fraud - Maduro claims victory despite 67-30 opposition win in vote tallies', severity: 'critical' },
      { date: '2024-09-07', description: 'Edmundo González flees to Spain, arrest warrant issued', severity: 'serious' },
      // 2025 Trump Escalation
      { date: '2025-01-10', description: 'Trump recognizes González as "president-elect" on inauguration day', severity: 'serious' },
      { date: '2025-01-10', description: 'Maduro inaugurated for third term despite international rejection', severity: 'serious' },
      { date: '2025-01-20', description: 'González attends Trump inauguration as invited guest (Sen. Rick Scott)', severity: 'moderate' },
      { date: '2025-01-23', description: 'Secretary of State Rubio ratifies US recognition of González', severity: 'moderate' },
      // August-November Escalation
      { date: '2025-08-XX', description: 'US surges military assets to Caribbean (stated: anti-drug)', severity: 'serious' },
      { date: '2025-11-09', description: 'First boat strikes - 6 killed in 2 vessels', severity: 'serious' },
      { date: '2025-11-13', description: 'Operation Southern Spear officially announced by Pentagon', severity: 'critical' },
      { date: '2025-11-14', description: 'Gerald R. Ford Carrier Strike Group arrives in Caribbean (4,000 sailors, 12 aircraft)', severity: 'critical' },
      { date: '2025-11-15', description: '20th strike - 4 killed, total reaches 80+ dead', severity: 'serious' },
      { date: '2025-11-16', description: '21st strike - total confirmed: 82-83 killed in boat operations', severity: 'serious' },
      { date: '2025-11-21', description: 'Trump-Maduro phone call - Trump offers safe passage ultimatum (1 week)', severity: 'critical' },
      { date: '2025-11-24', description: 'Cartel de los Soles designated FTO (Maduro = terrorist organization head)', severity: 'critical' },
      { date: '2025-11-29', description: 'Trump declares Venezuelan airspace "closed" after Maduro rejects deal', severity: 'critical' },
      { date: '2025-12-01', description: 'Maduro loses Honduras, St Vincent allies at polls - regional isolation deepens', severity: 'moderate' },
      { date: '2025-12-02', description: 'Maduro rallies Caracas crowd - rejects "slave\'s peace", 4M militia mobilized', severity: 'serious' },
      { date: '2025-12-02', description: 'Trump signals land strikes: "much easier" than sea', severity: 'critical' },
      { date: '2025-12-02', description: 'WaPo reports "double-tap" strikes on survivors', severity: 'serious' },
      { date: '2025-12-03', description: 'Venezuela resumes US deportation flights despite airspace "closure"', severity: 'moderate' },
    ],
    analystAssessment: `
STATUS AS OF DECEMBER 3, 2025: ACTIVE LOW-INTENSITY CONFLICT

MILITARY SITUATION:
- Operation Southern Spear: 11-12 warships, 15,000+ US troops, Gerald R. Ford CSG
- 22nd MEU deployed, "anti-drug trafficking" mission
- 21+ kinetic strikes on vessels, 82-83 confirmed killed
- Pentagon admits "double-tap" strikes on survivors
- Maduro: 4M+ Bolivarian Militia mobilized, "maximum preparedness"

THE TRUMP ULTIMATUM (Nov 21):
- Offered Maduro safe passage for family
- Maduro countered: global immunity + retain military control → REJECTED
- "One week to leave" → expired Nov 28
- Nov 29: Trump declared airspace "closed"

CARTEL DE LOS SOLES DESIGNATION (Nov 24):
- Maduro designated head of FTO (Foreign Terrorist Organization)
- "Cartel of the Suns" = corrupt military/government officials since 1990s
- Legal experts: "designating a non-thing that is not a terror organization"
- Practical effect: Legal cover for asset targeting, infrastructure strikes
- Maduro denies cartel exists - calls it "fabrication"

STRATEGIC CALCULATIONS:
- US stated goal: Anti-narcoterrorism + block migration
- Actual goal (per analysts): Regime change, push out Russia/China/Iran
- Trump: "not about Maduro per se, about Russia/China/Iran in hemisphere"
- 70% of Americans oppose military action (CBS/YouGov)
- Experts: current forces insufficient for actual invasion
- Trump strategy: maximum pressure → voluntary departure

RUSSIA/CHINA RESPONSE:
- Russia: "resolutely condemns excessive military force"
- Russia: Emergency air defense weapons + technicians sent to Venezuela
- Russia: Sukhoi jets, helicopters, tanks, SAMs already in country
- China: Condemned as "foreign interference in regional affairs"
- China: Zero-tariff trade deal at Shanghai Expo 2025
- China: Venezuela's #1 oil buyer
- Venezuela requested: drones, missiles, radar from all three (RU/CN/IR)
- KEY: Neither Russia nor China showing appetite for direct confrontation

MADURO'S POSITION:
- Controls territory (95%+), military loyal, oil revenue (China/Russia)
- "We want peace with sovereignty, not a slave's peace"
- Conducting military drills, civilian militia mobilization
- Lost regional allies (Honduras, St Vincent) but core support intact
- Using crisis for nationalist mobilization

INTERVENTION SCENARIOS:
1. Maximum Pressure → Maduro Flees (Trump's stated goal)
   - Probability: 15-20%
   - Requires: Military defections, which haven't materialized

2. Limited Strikes → Stalemate
   - Probability: 50-55%
   - Current trajectory: Boat strikes, possible land strikes
   - No regime collapse, sustained pressure

3. Full Invasion
   - Probability: 10-15%
   - Requires: Force buildup 10x current, congressional pushback
   - CBS: 70% oppose, experts say current assets insufficient

4. De-escalation / Deal
   - Probability: 20-25%
   - Requires: Face-saving for both sides
   - Possible via third party (Pope Leo XVI offering)

WILDCARDS:
- Essequibo: Maduro could attack Guyana as distraction
- Iran: Could provide advanced missiles, change calculus
- Russia: Could send advisors, make strikes politically costly
- China: Could threaten economic retaliation vs US
- Military Coup: Unlikely - regime has purged dissenters

BOTTOM LINE:
First direct US military strikes in Latin America since Panama 1989.
Trump reviving Monroe Doctrine but without clear exit strategy.
Maduro not budging. Neither is Trump.
Most likely: sustained low-intensity conflict through 2026.
Risk of miscalculation into larger war: ELEVATED.
    `,
    lastUpdated: '2025-12-03',
  },

  // Haiti
  'haiti': {
    id: 'haiti',
    name: 'Haiti Gang Crisis / State Collapse',
    region: 'Caribbean',
    domain: 'land',
    parties: [
      {
        code: 'HT',
        name: 'Haitian Government (Transitional)',
        claimBasis: 'Transitional authority',
        controlPercentage: 20,
        militaryPresence: 'minimal',
      },
      {
        code: 'HT-GANG',
        name: 'Gang Coalitions (G9, G-Pep)',
        claimBasis: 'De facto territorial control',
        controlPercentage: 80,
        militaryPresence: 'heavy',
      },
      {
        code: 'KE-MSS',
        name: 'Kenya-led MSS Mission',
        claimBasis: 'UN-authorized security support',
        controlPercentage: 0,
        militaryPresence: 'minimal',
      },
    ],
    status: 'active_conflict',
    escalationRisk: 0.75,
    strategicImportance: 0.45,
    resourcesAtStake: [
      'Regional stability',
      'Migration to US/DR',
      'Humanitarian crisis',
      'Drug trafficking route',
    ],
    internationalLaw: {
      relevantTreaties: [],
      icjRulings: [],
      unResolutions: ['MSS authorization'],
      status: 'contested',
    },
    recentIncidents: [
      { date: '2024-02-29', description: 'Gangs storm prisons, PM resigns', severity: 'critical' },
      { date: '2024-06-25', description: 'Kenya police arrive - first MSS deployment', severity: 'moderate' },
      { date: '2024-11-07', description: 'Gang attacks continue despite MSS presence', severity: 'serious' },
    ],
    analystAssessment: `
Functional state collapse. Gangs control 80%+ of Port-au-Prince.
Kenya mission (1000 troops) far too small for 500+ gangs.
No elections possible in current conditions.
Dominican Republic border tensions rising.
US reluctant to intervene despite proximity.
Worst case: Somalia-style permanent failed state in Western Hemisphere.
    `,
    lastUpdated: '2024-12-01',
  },

  // Colombia
  'colombia': {
    id: 'colombia',
    name: 'Colombia Armed Groups',
    region: 'South America',
    domain: 'land',
    parties: [
      {
        code: 'CO',
        name: 'Colombian Government',
        claimBasis: 'Constitutional authority',
        controlPercentage: 80,
        militaryPresence: 'heavy',
      },
      {
        code: 'ELN',
        name: 'ELN (National Liberation Army)',
        claimBasis: 'Revolutionary, anti-imperialist',
        controlPercentage: 8,
        militaryPresence: 'moderate',
      },
      {
        code: 'DISS',
        name: 'FARC Dissidents (EMC, Segunda Marquetalia)',
        claimBasis: 'Rejected 2016 peace deal',
        controlPercentage: 7,
        militaryPresence: 'moderate',
      },
      {
        code: 'CLAN',
        name: 'Clan del Golfo (AGC)',
        claimBasis: 'Criminal enterprise (narco)',
        controlPercentage: 5,
        militaryPresence: 'moderate',
      },
    ],
    status: 'moderate_tension',
    escalationRisk: 0.45,
    strategicImportance: 0.65,
    resourcesAtStake: [
      'Cocaine production (world\'s largest)',
      'Regional stability',
      'US alliance',
      'Venezuela border',
    ],
    internationalLaw: {
      relevantTreaties: ['2016 FARC Peace Agreement'],
      icjRulings: [],
      unResolutions: [],
      status: 'ambiguous',
    },
    recentIncidents: [
      { date: '2024-09-18', description: 'ELN-government talks suspended after attacks', severity: 'moderate' },
      { date: '2024-06-01', description: 'EMC expands operations in Cauca', severity: 'moderate' },
    ],
    analystAssessment: `
2016 peace deal with FARC holds but implementation incomplete.
ELN negotiations stalled under Petro government.
Coca production at record levels despite eradication.
Clan del Golfo expanding - criminal not political.
Venezuela border is ungoverned space.
"Total peace" policy (negotiate with all) struggling.
    `,
    lastUpdated: '2024-12-01',
  },

  // Ecuador
  'ecuador': {
    id: 'ecuador',
    name: 'Ecuador Cartel Invasion',
    region: 'South America',
    domain: 'land',
    parties: [
      {
        code: 'EC',
        name: 'Ecuador Government',
        claimBasis: 'Constitutional authority',
        controlPercentage: 75,
        militaryPresence: 'heavy',
      },
      {
        code: 'CARTELS',
        name: 'Mexican/Colombian Cartels',
        claimBasis: 'Criminal territorial control',
        controlPercentage: 25,
        militaryPresence: 'heavy',
      },
    ],
    status: 'high_tension',
    escalationRisk: 0.60,
    strategicImportance: 0.50,
    resourcesAtStake: [
      'Cocaine transit route',
      'Port access (Guayaquil)',
      'Regional stability',
    ],
    internationalLaw: {
      relevantTreaties: [],
      icjRulings: [],
      unResolutions: [],
      status: 'clear',
    },
    recentIncidents: [
      { date: '2024-01-09', description: 'Gangs storm TV station live on air, state of emergency', severity: 'critical' },
      { date: '2024-01-08', description: 'Prison gang leaders escape, war declared', severity: 'critical' },
      { date: '2024-04-21', description: 'Noboa wins referendum for military against gangs', severity: 'moderate' },
    ],
    analystAssessment: `
Ecuador transformed from peaceful to cartel battleground in 5 years.
Mexican cartels (Sinaloa, CJNG) using Ecuador as transit hub.
Homicide rate 10x increase since 2018.
Government declared "internal armed conflict" - unprecedented.
Military deployed but facing learning curve.
Ports (Guayaquil) infiltrated - corruption endemic.
    `,
    lastUpdated: '2024-12-01',
  },

  // Nicaragua
  'nicaragua': {
    id: 'nicaragua',
    name: 'Nicaragua Authoritarianism',
    region: 'Central America',
    domain: 'land',
    parties: [
      {
        code: 'NI-ORTS',
        name: 'Ortega Government',
        claimBasis: 'Electoral (disputed) legitimacy',
        controlPercentage: 100,
        militaryPresence: 'dominant',
      },
      {
        code: 'NI-OPP',
        name: 'Opposition (exiled)',
        claimBasis: 'Democratic legitimacy',
        controlPercentage: 0,
        militaryPresence: 'none',
      },
    ],
    status: 'moderate_tension',
    escalationRisk: 0.30,
    strategicImportance: 0.40,
    resourcesAtStake: [
      'Central American stability',
      'China canal project (dormant)',
      'Migration flows',
      'Russian/Iranian presence',
    ],
    internationalLaw: {
      relevantTreaties: [],
      icjRulings: [],
      unResolutions: [],
      status: 'contested',
    },
    recentIncidents: [
      { date: '2024-02-09', description: 'Bishop Álvarez sentenced to 26 years', severity: 'moderate' },
      { date: '2023-09-12', description: 'Opposition figures stripped of citizenship', severity: 'moderate' },
    ],
    analystAssessment: `
Ortega has consolidated full authoritarian control.
Opposition crushed - leaders imprisoned or exiled.
Church persecution ongoing (Catholic opposition).
Russia, China, Iran expanding ties.
No internal threat to regime. Regional isolation increasing.
US sanctions have limited effect.
    `,
    lastUpdated: '2024-12-01',
  },

  // Guyana-Venezuela (Essequibo) - NOW LINKED TO US-VENEZUELA CRISIS
  'guyana_essequibo': {
    id: 'guyana-essequibo',
    name: 'Essequibo Dispute (Guyana-Venezuela)',
    region: 'South America',
    domain: 'land',
    parties: [
      {
        code: 'GY',
        name: 'Guyana',
        claimBasis: '1899 Arbitral Award, ICJ jurisdiction',
        controlPercentage: 100,
        militaryPresence: 'minimal',
      },
      {
        code: 'VE',
        name: 'Venezuela',
        claimBasis: 'Historical claim (pre-1899), 1966 Geneva Agreement',
        controlPercentage: 0,
        militaryPresence: 'heavy', // Border
      },
      {
        code: 'US',
        name: 'United States',
        claimBasis: 'Guyana defense commitment',
        controlPercentage: 0,
        militaryPresence: 'minimal', // But nearby for Operation Southern Spear
      },
    ],
    status: 'high_tension',
    escalationRisk: 0.55, // Elevated due to US-Venezuela crisis
    strategicImportance: 0.70,
    resourcesAtStake: [
      'Oil (ExxonMobil offshore discoveries - 11B+ barrels)',
      '160,000 km² territory (2/3 of Guyana)',
      'Regional precedent',
      'Potential Maduro "wag the dog" target',
    ],
    internationalLaw: {
      relevantTreaties: ['1899 Paris Award', '1966 Geneva Agreement'],
      icjRulings: ['2023 ICJ confirms jurisdiction over dispute', '2024 ICJ orders Venezuela to halt actions'],
      unResolutions: [],
      status: 'clear', // ICJ ruled for Guyana
    },
    recentIncidents: [
      { date: '2023-12-03', description: 'Venezuela referendum on annexation (sham, 95% "yes")', severity: 'serious' },
      { date: '2024-04-01', description: 'ICJ orders Venezuela to halt actions affecting Guyana', severity: 'moderate' },
      { date: '2024-07-08', description: 'Venezuela continues military buildup at border', severity: 'moderate' },
      { date: '2025-11-XX', description: 'US Operation Southern Spear creates nearby military presence', severity: 'serious' },
      { date: '2025-12-02', description: 'Analysts warn Maduro could use Essequibo as distraction', severity: 'moderate' },
    ],
    analystAssessment: `
STATUS AS OF DECEMBER 3, 2025: ELEVATED RISK DUE TO US-VENEZUELA CRISIS

CONTEXT SHIFT:
- Dormant claim reactivated 2023 after ExxonMobil discoveries
- NOW: Part of broader US-Venezuela confrontation
- Maduro facing Operation Southern Spear pressure
- Essequibo attack = possible "wag the dog" distraction

MILITARY BALANCE:
- Venezuela: 350,000 active military + 4M militia (mobilized)
- Guyana: ~3,400 active military (Coast Guard focus)
- Venezuela could take Essequibo in days without intervention
- BUT: US has 15,000 troops in Caribbean, carrier strike group

WAG THE DOG SCENARIO:
- If Maduro feels regime threatened, could attack Essequibo
- Goal: Rally nationalism, complicate US intervention
- Risk: Would trigger immediate US/UK response, accelerate regime end
- Probability: ~8-12% (see VENEZUELA_THREAT_ASSESSMENT)

INTERNATIONAL PROTECTION:
- US, UK, Brazil all signaled support for Guyana
- US has conducted military exercises with Guyana
- UK HMS Trent regularly patrols
- Brazil would not accept Venezuelan expansion

ICJ STATUS:
- 2024: ICJ ordered Venezuela to halt actions affecting Guyana
- Venezuela ignores ICJ, claims no jurisdiction
- Final ruling expected to favor Guyana

OIL FACTOR:
- ExxonMobil operating offshore Guyana (11B+ barrels discovered)
- Guyana now one of world's top oil producers per capita
- Venezuela's own production collapsed (200K bpd vs 3M peak)
- Envy + nationalism = dangerous combination
    `,
    lastUpdated: '2025-12-03',
  },
};

// ============================================
// EXTENDED ASIA FLASHPOINTS
// ============================================

export const EXTENDED_ASIA_DISPUTES: Record<string, TerritorialDispute> = {
  // North Korea
  'north_korea': {
    id: 'north-korea',
    name: 'Korean Peninsula',
    region: 'Northeast Asia',
    domain: 'land',
    parties: [
      {
        code: 'KR',
        name: 'South Korea',
        claimBasis: 'Constitutional claim to entire peninsula, democracy',
        controlPercentage: 45,
        militaryPresence: 'heavy',
      },
      {
        code: 'KP',
        name: 'North Korea',
        claimBasis: 'Constitutional claim to entire peninsula, revolutionary',
        controlPercentage: 45,
        militaryPresence: 'heavy',
      },
      {
        code: 'US',
        name: 'United States (USFK)',
        claimBasis: 'Alliance treaty, UN Command',
        controlPercentage: 0,
        militaryPresence: 'heavy',
      },
    ],
    status: 'high_tension',
    escalationRisk: 0.50,
    strategicImportance: 0.90,
    resourcesAtStake: [
      'Nuclear weapons (50+ warheads)',
      'Regional stability',
      'US alliance credibility',
      'Japan security',
      '25M hostage population (North)',
    ],
    internationalLaw: {
      relevantTreaties: ['1953 Armistice (no peace treaty)'],
      icjRulings: [],
      unResolutions: ['Multiple sanctions resolutions'],
      status: 'contested',
    },
    recentIncidents: [
      { date: '2024-10-15', description: 'North Korea sends troops to Russia for Ukraine war', severity: 'serious' },
      { date: '2024-07-31', description: 'Kim declares South Korea "enemy state" in constitution', severity: 'serious' },
      { date: '2024-01-05', description: 'North Korea shells near South Korean islands', severity: 'serious' },
    ],
    analystAssessment: `
Kim Jong Un has abandoned unification rhetoric - now "two hostile states."
Nuclear arsenal growing (60+ weapons by 2024), delivery improving (ICBMs, SLBMs, tactical).
Russia alliance deepening - arms for troops exchange.
South Korea considering own nuclear weapons (public support 70%+).
US extended deterrence being tested.
Scenarios: Status quo most likely, but window for miscalculation widening.
Regime collapse probability low but consequences would be catastrophic.
    `,
    lastUpdated: '2024-12-01',
  },

  // Myanmar
  'myanmar': {
    id: 'myanmar',
    name: 'Myanmar Civil War',
    region: 'Southeast Asia',
    domain: 'land',
    parties: [
      {
        code: 'MM-SAC',
        name: 'Military Junta (SAC)',
        claimBasis: 'Constitutional guardian role (self-declared)',
        controlPercentage: 40,
        militaryPresence: 'heavy',
      },
      {
        code: 'MM-NUG',
        name: 'National Unity Government (NUG/PDF)',
        claimBasis: 'Democratic legitimacy (2020 election)',
        controlPercentage: 25,
        militaryPresence: 'moderate',
      },
      {
        code: 'MM-EAO',
        name: 'Ethnic Armed Organizations',
        claimBasis: 'Ethnic self-determination',
        controlPercentage: 35,
        militaryPresence: 'heavy',
      },
    ],
    status: 'active_conflict',
    escalationRisk: 0.70,
    strategicImportance: 0.55,
    resourcesAtStake: [
      'China-Myanmar corridor (Belt and Road)',
      'Rare earths, jade, timber',
      'Regional refugee flows',
      'Drug production (meth)',
    ],
    internationalLaw: {
      relevantTreaties: [],
      icjRulings: ['Rohingya genocide case ongoing'],
      unResolutions: ['Various condemnations'],
      status: 'contested',
    },
    recentIncidents: [
      { date: '2023-10-27', description: 'Operation 1027 - resistance major offensive begins', severity: 'critical' },
      { date: '2024-08-03', description: 'Junta loses control of northern Shan State', severity: 'critical' },
      { date: '2024-11-12', description: 'Resistance captures more territory, junta weakening', severity: 'serious' },
    ],
    analystAssessment: `
Junta losing - Operation 1027 changed balance dramatically.
Multiple resistance groups cooperating (unprecedented).
Junta controls only ~40% of territory, major cities still held.
China backing both sides to hedge.
Regime collapse now plausible scenario.
Refugee crisis: 2M+ displaced internally, 1M+ to neighbors.
Meth production funding all sides.
    `,
    lastUpdated: '2024-12-01',
  },

  // Philippines - South China Sea (Detail)
  'philippines_scs': {
    id: 'philippines-scs',
    name: 'Philippines-China SCS Confrontation',
    region: 'South China Sea',
    domain: 'maritime',
    parties: [
      {
        code: 'PH',
        name: 'Philippines',
        claimBasis: 'UNCLOS, 2016 PCA ruling',
        controlPercentage: 20,
        militaryPresence: 'minimal',
      },
      {
        code: 'CN',
        name: 'China',
        claimBasis: 'Nine-dash line (rejected by PCA)',
        controlPercentage: 80,
        militaryPresence: 'dominant',
      },
    ],
    status: 'high_tension',
    escalationRisk: 0.60,
    strategicImportance: 0.85,
    resourcesAtStake: [
      'Fishing grounds',
      'Oil/gas (Reed Bank)',
      'US-Philippines MDT test',
      'Regional precedent',
    ],
    internationalLaw: {
      relevantTreaties: ['UNCLOS', 'US-PH Mutual Defense Treaty'],
      icjRulings: [],
      unResolutions: [],
      status: 'clear', // 2016 PCA ruling definitive
    },
    recentIncidents: [
      { date: '2024-06-17', description: 'PH sailor loses thumb in CCG confrontation', severity: 'serious' },
      { date: '2024-08-31', description: 'CCG rams PH vessel near Sabina Shoal', severity: 'serious' },
      { date: '2024-11-16', description: 'US reiterates MDT covers SCS', severity: 'moderate' },
    ],
    analystAssessment: `
Marcos Jr. has shifted from Duterte's accommodation to confrontation.
Second Thomas Shoal (BRP Sierra Madre) is primary flashpoint.
US MDT Article IV covers SCS - tested but not triggered.
Philippines using transparency strategy (documenting incidents).
Risk: Chinese miscalculation triggers MDT, escalation to US-China conflict.
Sabina Shoal emerging as second flashpoint.
    `,
    lastUpdated: '2024-12-01',
  },

  // Bangladesh
  'bangladesh': {
    id: 'bangladesh',
    name: 'Bangladesh Political Crisis',
    region: 'South Asia',
    domain: 'land',
    parties: [
      {
        code: 'BD-INT',
        name: 'Interim Government (Yunus)',
        claimBasis: 'Revolutionary legitimacy (student uprising)',
        controlPercentage: 100,
        militaryPresence: 'moderate',
      },
      {
        code: 'BD-AL',
        name: 'Awami League (exiled)',
        claimBasis: 'Electoral (disputed), Hasina in India',
        controlPercentage: 0,
        militaryPresence: 'none',
      },
    ],
    status: 'moderate_tension',
    escalationRisk: 0.45,
    strategicImportance: 0.55,
    resourcesAtStake: [
      'Garment industry ($50B+ exports)',
      'India-Bangladesh relations',
      'Rohingya refugees (1M+)',
      'Bay of Bengal access',
    ],
    internationalLaw: {
      relevantTreaties: [],
      icjRulings: [],
      unResolutions: [],
      status: 'ambiguous',
    },
    recentIncidents: [
      { date: '2024-08-05', description: 'Hasina flees after student uprising, 1000+ killed', severity: 'critical' },
      { date: '2024-08-08', description: 'Yunus sworn in as interim leader', severity: 'serious' },
      { date: '2024-10-20', description: 'Hindu minority attacks, India tensions rise', severity: 'moderate' },
    ],
    analystAssessment: `
Hasina's 15-year rule ended by student revolution.
170M population - 8th largest.
India lost key ally - watching nervously.
Economic stability key test (garment industry, remittances).
Hindu minority facing backlash - India-Bangladesh tension.
Transition to elections uncertain - military role unclear.
    `,
    lastUpdated: '2024-12-01',
  },

  // Afghanistan
  'afghanistan': {
    id: 'afghanistan',
    name: 'Afghanistan Under Taliban',
    region: 'Central/South Asia',
    domain: 'land',
    parties: [
      {
        code: 'AF-TAL',
        name: 'Taliban (Islamic Emirate)',
        claimBasis: 'Military victory 2021, Islamic legitimacy',
        controlPercentage: 95,
        militaryPresence: 'dominant',
      },
      {
        code: 'ISIS-K',
        name: 'ISIS-Khorasan',
        claimBasis: 'Caliphate, anti-Taliban',
        controlPercentage: 2,
        militaryPresence: 'minimal',
      },
      {
        code: 'NRF',
        name: 'National Resistance Front',
        claimBasis: 'Anti-Taliban, Panjshir-based',
        controlPercentage: 3,
        militaryPresence: 'minimal',
      },
    ],
    status: 'moderate_tension',
    escalationRisk: 0.35,
    strategicImportance: 0.60,
    resourcesAtStake: [
      'Rare earths ($1T+ estimated)',
      'Regional terrorism export',
      'Drug production (opium/heroin)',
      'Pakistan stability',
      'China access (Wakhan Corridor)',
    ],
    internationalLaw: {
      relevantTreaties: [],
      icjRulings: [],
      unResolutions: [],
      status: 'contested',
    },
    recentIncidents: [
      { date: '2024-03-22', description: 'ISIS-K Moscow attack planned from Afghanistan', severity: 'critical' },
      { date: '2024-09-15', description: 'Taliban-Pakistan border clashes continue', severity: 'moderate' },
    ],
    analystAssessment: `
Taliban consolidated control - most stable in 45 years (by brutal standards).
Women's rights erased - most restrictive in world.
ISIS-K remains threat (Moscow attack showed reach).
Al-Qaeda relationship continues despite Doha promises.
Pakistan-Taliban tensions rising (TTP safe haven).
China, Russia engaging pragmatically.
Recognition unlikely but de facto acceptance growing.
    `,
    lastUpdated: '2024-12-01',
  },
};

// ============================================
// EXTENDED DOMESTIC INSTABILITY
// ============================================

export const EXTENDED_DOMESTIC_INSTABILITY: Record<string, DomesticInstability> = {
  // Brazil
  'brazil': {
    id: 'brazil-domestic',
    country: 'Brazil',
    countryCode: 'BR',
    threatTypes: ['polarization', 'institutional_decay', 'organized_crime'],
    severity: 'moderate',
    trajectory: 'stable',
    indicators: [
      { name: 'Political Polarization', value: 'Extreme (Lula vs Bolsonaro)', trend: 'stable' },
      { name: 'Democratic Institutions', value: 'Tested but holding', trend: 'improving' },
      { name: 'Organized Crime', value: 'PCC, Comando Vermelho expanding', trend: 'deteriorating' },
    ],
    flashpoints: [
      { name: 'January 8 Aftermath', description: 'Bolsonarista coup attempt prosecutions', riskLevel: 'moderate' },
      { name: 'Amazon Deforestation', description: 'Environmental policy battleground', riskLevel: 'low' },
      { name: 'Organized Crime Expansion', description: 'PCC now in all states, international', riskLevel: 'high' },
    ],
    recentEvents: [
      { date: '2024-11-13', event: 'PCC-linked bomb plot at Supreme Court foiled', impact: 'major' },
      { date: '2023-01-08', event: 'Brasília riots (J8) - Brazilian J6', impact: 'major' },
    ],
    analystNotes: `
Lula stabilizing but polarization remains.
J8 rioters being prosecuted - institutions held.
Bolsonaro barred from office until 2030.
PCC (Primeiro Comando da Capital) is regional threat - operates in Paraguay, Bolivia.
Economy recovering but debt concerns.
2026 election will test whether polarization can be managed.
    `,
    lastUpdated: '2024-12-01',
  },

  // Mexico
  'mexico': {
    id: 'mexico-domestic',
    country: 'Mexico',
    countryCode: 'MX',
    threatTypes: ['organized_crime', 'institutional_decay', 'civil_unrest'],
    severity: 'high',
    trajectory: 'deteriorating',
    indicators: [
      { name: 'Cartel Control', value: '30%+ of territory', trend: 'deteriorating' },
      { name: 'Homicide Rate', value: '25/100k (stable but high)', trend: 'stable' },
      { name: 'Judicial Independence', value: 'Under attack (2024 reform)', trend: 'deteriorating' },
    ],
    flashpoints: [
      { name: 'Sinaloa Civil War', description: 'Cartel split after Zambada arrest', riskLevel: 'high' },
      { name: 'Judicial Reform', description: 'Elected judges = cartel capture risk', riskLevel: 'high' },
      { name: 'Chiapas Violence', description: 'Migration + cartel nexus', riskLevel: 'moderate' },
    ],
    recentEvents: [
      { date: '2024-09-15', event: 'Judicial reform passes despite protests', impact: 'major' },
      { date: '2024-07-25', event: 'Zambada captured, Sinaloa civil war begins', impact: 'major' },
      { date: '2024-10-01', event: 'Sheinbaum inaugurated', impact: 'moderate' },
    ],
    analystNotes: `
AMLO's "hugs not bullets" failed - violence unchanged.
Sheinbaum continuing approach despite evidence of failure.
Judicial reform (elected judges) could enable cartel capture of courts.
Sinaloa cartel civil war is bloodiest cartel conflict in years.
US relations strained over migration, fentanyl, judicial reform.
State capacity eroding in multiple regions.
    `,
    lastUpdated: '2024-12-01',
  },

  // South Africa
  'south_africa': {
    id: 'south-africa-domestic',
    country: 'South Africa',
    countryCode: 'ZA',
    threatTypes: ['economic_crisis', 'institutional_decay', 'civil_unrest', 'organized_crime'],
    severity: 'elevated',
    trajectory: 'deteriorating',
    indicators: [
      { name: 'Unemployment', value: '32% official, 42% expanded', trend: 'deteriorating' },
      { name: 'Power Grid', value: 'Rolling blackouts (loadshedding)', trend: 'stable' },
      { name: 'State Capture', value: 'Partial recovery from Zuma era', trend: 'improving' },
    ],
    flashpoints: [
      { name: 'GNU Stability', description: 'ANC-DA coalition fragile', riskLevel: 'moderate' },
      { name: 'Eskom Collapse', description: 'Power utility dysfunction', riskLevel: 'high' },
      { name: 'Xenophobic Violence', description: 'Anti-immigrant sentiment', riskLevel: 'moderate' },
    ],
    recentEvents: [
      { date: '2024-06-14', event: 'ANC loses majority, GNU formed with DA', impact: 'major' },
      { date: '2024-03-01', event: 'Loadshedding reaches Stage 6 (10+ hours/day)', impact: 'major' },
    ],
    analystNotes: `
ANC below 50% first time since 1994 - coalition government.
Eskom (power utility) is existential threat - grid collapse possible.
Youth unemployment 60%+ - social time bomb.
Organized crime (cash-in-transit heists) endemic.
Government capacity severely degraded.
Bright spot: GNU may force reforms ANC wouldn't do alone.
    `,
    lastUpdated: '2024-12-01',
  },

  // Turkey
  'turkey': {
    id: 'turkey-domestic',
    country: 'Turkey',
    countryCode: 'TR',
    threatTypes: ['institutional_decay', 'economic_crisis', 'ethnic_conflict'],
    severity: 'elevated',
    trajectory: 'stable',
    indicators: [
      { name: 'Inflation', value: '50%+ (down from 85%)', trend: 'improving' },
      { name: 'Democratic Backsliding', value: 'Authoritarian consolidation', trend: 'stable' },
      { name: 'Kurdish Issue', value: 'PKK conflict frozen', trend: 'stable' },
    ],
    flashpoints: [
      { name: 'Economic Crisis', description: 'Unorthodox policy consequences', riskLevel: 'high' },
      { name: 'Kurdish Issue', description: 'Syria, Iraq cross-border ops', riskLevel: 'moderate' },
      { name: 'Erdogan Succession', description: 'No clear successor, 2028 election', riskLevel: 'moderate' },
    ],
    recentEvents: [
      { date: '2024-03-31', event: 'Opposition wins Istanbul, Ankara in local elections', impact: 'moderate' },
      { date: '2024-10-23', event: 'PKK attack on defense company, 5 killed', impact: 'moderate' },
    ],
    analystNotes: `
Erdogan's grip firm but economic pain testing support.
Orthodox economics returning (Simsek) - inflation falling slowly.
Opposition won major cities - system not fully closed.
Syria presence continues - 35,000 troops.
PKK conflict occasionally flares but manageable.
Key question: succession planning (Erdogan 70).
    `,
    lastUpdated: '2024-12-01',
  },

  // Israel
  'israel': {
    id: 'israel-domestic',
    country: 'Israel',
    countryCode: 'IL',
    threatTypes: ['polarization', 'institutional_decay', 'ethnic_conflict'],
    severity: 'high',
    trajectory: 'deteriorating',
    indicators: [
      { name: 'Political Polarization', value: 'Extreme (judicial reform crisis)', trend: 'deteriorating' },
      { name: 'Arab-Jewish Relations', value: 'Strained by war', trend: 'deteriorating' },
      { name: 'Reservist Morale', value: 'Protests over conscription', trend: 'deteriorating' },
    ],
    flashpoints: [
      { name: 'Gaza War Continuation', description: 'No end strategy, hostages', riskLevel: 'high' },
      { name: 'Haredi Conscription', description: 'Draft exemption crisis', riskLevel: 'moderate' },
      { name: 'West Bank Settler Violence', description: 'US sanctions on extremists', riskLevel: 'moderate' },
    ],
    recentEvents: [
      { date: '2024-10-07', event: 'Oct 7 anniversary, war continues', impact: 'major' },
      { date: '2024-09-01', event: 'Hostage deal protests intensify', impact: 'moderate' },
    ],
    analystNotes: `
Pre-Oct 7 judicial reform crisis paused but not resolved.
War united society initially but divisions reemerging.
Haredi (ultra-Orthodox) draft exemption unsustainable.
Netanyahu's political survival tied to war continuation.
West Bank settler violence increasing (US sanctions).
Post-war political reckoning will be severe.
    `,
    lastUpdated: '2024-12-01',
  },

  // Iran
  'iran': {
    id: 'iran-domestic',
    country: 'Iran',
    countryCode: 'IR',
    threatTypes: ['polarization', 'civil_unrest', 'economic_crisis', 'succession_crisis'],
    severity: 'elevated',
    trajectory: 'stable',
    indicators: [
      { name: 'Regime Support', value: '15-20% hardcore, rest acquiescent', trend: 'deteriorating' },
      { name: 'Economic Pressure', value: 'Sanctions biting, inflation 40%+', trend: 'stable' },
      { name: 'Succession', value: 'Khamenei 85, no clear heir', trend: 'deteriorating' },
    ],
    flashpoints: [
      { name: 'Mahsa Protests Legacy', description: '2022 uprising suppressed but sentiment remains', riskLevel: 'moderate' },
      { name: 'Khamenei Succession', description: 'Unknown when/how transition occurs', riskLevel: 'high' },
      { name: 'Economic Collapse', description: 'Oil revenue dependent on China', riskLevel: 'moderate' },
    ],
    recentEvents: [
      { date: '2024-07-28', event: 'Pezeshkian elected president (reformist facade)', impact: 'moderate' },
      { date: '2024-05-19', event: 'Raisi helicopter crash, succession questions', impact: 'moderate' },
    ],
    analystNotes: `
Regime survived 2022 Woman Life Freedom protests.
Raisi death showed fragility but system adapted.
Pezeshkian's "reform" presidency is fig leaf.
Real power: Khamenei (85), IRGC.
Succession is key uncertainty - could trigger instability.
Society deeply alienated but no organized opposition.
Regional wars (Israel, Hezbollah) divert attention.
    `,
    lastUpdated: '2024-12-01',
  },

  // Russia
  'russia': {
    id: 'russia-domestic',
    country: 'Russia',
    countryCode: 'RU',
    threatTypes: ['institutional_decay', 'economic_crisis', 'succession_crisis'],
    severity: 'moderate',
    trajectory: 'stable',
    indicators: [
      { name: 'War Support', value: '70%+ (passive acceptance)', trend: 'stable' },
      { name: 'Economic Resilience', value: 'Better than expected but strains showing', trend: 'deteriorating' },
      { name: 'Elite Stability', value: 'Prigozhin eliminated, FSB dominant', trend: 'stable' },
    ],
    flashpoints: [
      { name: 'War Losses', description: '600K+ casualties, mobilization risk', riskLevel: 'moderate' },
      { name: 'Putin Succession', description: 'No mechanism, health rumors', riskLevel: 'high' },
      { name: 'Economic Pressure', description: 'Inflation, labor shortage', riskLevel: 'moderate' },
    ],
    recentEvents: [
      { date: '2024-03-17', event: 'Putin reelected (87%) in managed vote', impact: 'minor' },
      { date: '2024-08-06', event: 'Ukraine Kursk incursion humiliation', impact: 'moderate' },
    ],
    analystNotes: `
Regime stable - repression works, opposition crushed.
War economy booming but unsustainable long-term.
Prigozhin mutiny was warning but system adapted.
Navalny death, Kursk incursion - no visible impact on stability.
Succession is black box - Putin 72, no heir apparent.
Best guess: stable until Putin dies/incapacitated, then chaos.
    `,
    lastUpdated: '2024-12-01',
  },

  // China
  'china': {
    id: 'china-domestic',
    country: 'China',
    countryCode: 'CN',
    threatTypes: ['economic_crisis', 'institutional_decay', 'separatism'],
    severity: 'moderate',
    trajectory: 'deteriorating',
    indicators: [
      { name: 'Economic Growth', value: 'Slowing, property crisis', trend: 'deteriorating' },
      { name: 'Xi Control', value: 'Absolute (Mao-level)', trend: 'stable' },
      { name: 'Youth Unemployment', value: '20%+ (stopped publishing)', trend: 'deteriorating' },
    ],
    flashpoints: [
      { name: 'Property Crisis', description: 'Evergrande, local government debt', riskLevel: 'high' },
      { name: 'Xinjiang/Tibet', description: 'Repression stable but expensive', riskLevel: 'low' },
      { name: 'Hong Kong', description: 'Crushed but emigration wave', riskLevel: 'low' },
    ],
    recentEvents: [
      { date: '2024-09-26', event: 'Stimulus announcement after months of drift', impact: 'moderate' },
      { date: '2024-07-15', event: 'Third Plenum: no bold reforms', impact: 'moderate' },
    ],
    analystNotes: `
Xi has more power than any leader since Mao.
Economic model exhausted - property bubble, demographics, debt.
Youth unemployment politically sensitive (numbers hidden).
No visible elite opposition - purges effective.
Xinjiang, Tibet, Hong Kong - crushed resistance.
Taiwan obsession could trigger strategic error.
Key risk: economic failure + nationalism = foreign adventure.
    `,
    lastUpdated: '2024-12-01',
  },
};

// ============================================
// MORE INTELLIGENCE AGENCIES
// ============================================

export const EXTENDED_INTEL_AGENCIES: Record<string, IntelligenceAgency> = {
  // France
  'dgse': {
    id: 'dgse',
    name: 'Direction Générale de la Sécurité Extérieure (DGSE)',
    country: 'France',
    countryCode: 'FR',
    type: 'foreign_intelligence',
    capabilities: { humint: 0.80, sigint: 0.65, cyber: 0.75, covert: 0.80, analysis: 0.80 },
    globalPresence: 'extensive',
    majorPartnerships: ['Five Eyes (partial)', 'EU', 'Francophone Africa'],
    knownOperations: [
      { name: 'Rainbow Warrior', type: 'covert', date: '1985', target: 'Greenpeace' },
      { name: 'Sahel operations', type: 'CT', date: 'ongoing', target: 'Jihadists' },
    ],
    notes: 'Strong Africa presence (former colonies). Conducted controversial ops (Greenpeace). Active in Sahel until withdrawal.',
  },
  'dgsi': {
    id: 'dgsi',
    name: 'Direction Générale de la Sécurité Intérieure (DGSI)',
    country: 'France',
    countryCode: 'FR',
    type: 'domestic_security',
    capabilities: { humint: 0.75, sigint: 0.60, cyber: 0.70, covert: 0.50, analysis: 0.75 },
    globalPresence: 'limited',
    majorPartnerships: ['EU domestic services'],
    knownOperations: [
      { name: 'Bataclan response', type: 'CT', date: '2015', target: 'ISIS network' },
    ],
    notes: 'Created 2014 to improve domestic CT. Heavily tested by Islamist attacks.',
  },

  // Germany
  'bnd': {
    id: 'bnd',
    name: 'Bundesnachrichtendienst (BND)',
    country: 'Germany',
    countryCode: 'DE',
    type: 'foreign_intelligence',
    capabilities: { humint: 0.70, sigint: 0.75, cyber: 0.70, covert: 0.50, analysis: 0.80 },
    globalPresence: 'extensive',
    majorPartnerships: ['Five Eyes (close but not member)', 'EU', 'NATO'],
    knownOperations: [
      { name: 'Crypto AG operation', type: 'SIGINT', date: '1970s-2018', target: 'Global (with CIA)' },
    ],
    notes: 'Historically constrained by post-WWII restrictions. Crypto AG was major Cold War success.',
  },
  'bfv': {
    id: 'bfv',
    name: 'Bundesamt für Verfassungsschutz (BfV)',
    country: 'Germany',
    countryCode: 'DE',
    type: 'domestic_security',
    capabilities: { humint: 0.70, sigint: 0.50, cyber: 0.60, covert: 0.40, analysis: 0.75 },
    globalPresence: 'limited',
    majorPartnerships: ['EU domestic services'],
    knownOperations: [
      { name: 'AfD monitoring', type: 'CI', date: '2021+', target: 'Far-right' },
      { name: 'NSU failure', type: 'failure', date: '2011', target: 'Missed neo-Nazi terror cell' },
    ],
    notes: 'NSU failures led to major reforms. Now monitoring AfD for extremism.',
  },

  // Australia
  'asis': {
    id: 'asis',
    name: 'Australian Secret Intelligence Service (ASIS)',
    country: 'Australia',
    countryCode: 'AU',
    type: 'foreign_intelligence',
    capabilities: { humint: 0.75, sigint: 0.50, cyber: 0.65, covert: 0.70, analysis: 0.75 },
    globalPresence: 'regional',
    majorPartnerships: ['Five Eyes', 'AUKUS'],
    knownOperations: [
      { name: 'Timor-Leste bugging', type: 'SIGINT', date: '2004', target: 'Timor government' },
    ],
    notes: 'Five Eyes member. Asia-Pacific focus. Timor bugging scandal damaged reputation.',
  },
  'asd': {
    id: 'asd',
    name: 'Australian Signals Directorate (ASD)',
    country: 'Australia',
    countryCode: 'AU',
    type: 'signals',
    capabilities: { humint: 0.20, sigint: 0.85, cyber: 0.85, covert: 0.40, analysis: 0.80 },
    globalPresence: 'regional',
    majorPartnerships: ['Five Eyes', 'AUKUS'],
    knownOperations: [
      { name: 'Asia-Pacific SIGINT', type: 'collection', date: 'ongoing', target: 'Regional' },
    ],
    notes: 'Five Eyes SIGINT. AUKUS expanding cyber/nuclear cooperation.',
  },

  // Canada
  'csis': {
    id: 'csis',
    name: 'Canadian Security Intelligence Service (CSIS)',
    country: 'Canada',
    countryCode: 'CA',
    type: 'domestic_security',
    capabilities: { humint: 0.70, sigint: 0.40, cyber: 0.60, covert: 0.40, analysis: 0.75 },
    globalPresence: 'limited',
    majorPartnerships: ['Five Eyes'],
    knownOperations: [
      { name: 'Chinese interference investigation', type: 'CI', date: '2023+', target: 'PRC operations' },
    ],
    notes: 'Five Eyes member. Major focus on Chinese interference in Canadian politics.',
  },
  'cse': {
    id: 'cse',
    name: 'Communications Security Establishment (CSE)',
    country: 'Canada',
    countryCode: 'CA',
    type: 'signals',
    capabilities: { humint: 0.15, sigint: 0.80, cyber: 0.80, covert: 0.30, analysis: 0.75 },
    globalPresence: 'extensive',
    majorPartnerships: ['Five Eyes'],
    knownOperations: [
      { name: 'SIGINT collection', type: 'collection', date: 'ongoing', target: 'Global (Five Eyes sharing)' },
    ],
    notes: 'Five Eyes SIGINT partner. Cyber defence and offense capabilities.',
  },

  // India
  'raw': {
    id: 'raw',
    name: 'Research and Analysis Wing (R&AW)',
    country: 'India',
    countryCode: 'IN',
    type: 'foreign_intelligence',
    capabilities: { humint: 0.80, sigint: 0.60, cyber: 0.65, covert: 0.75, analysis: 0.70 },
    globalPresence: 'regional',
    majorPartnerships: ['US (improving)', 'Israel', 'Russia (historical)'],
    knownOperations: [
      { name: 'Bangladesh 1971', type: 'covert', date: '1971', target: 'Pakistan' },
      { name: 'Nijjar assassination (alleged)', type: 'covert', date: '2023', target: 'Khalistani leader in Canada' },
    ],
    notes: 'Strong Pakistan/China focus. Nijjar case damaged India-Canada relations. Growing US partnership via Quad.',
  },
  'ib': {
    id: 'ib',
    name: 'Intelligence Bureau (IB)',
    country: 'India',
    countryCode: 'IN',
    type: 'domestic_security',
    capabilities: { humint: 0.75, sigint: 0.55, cyber: 0.55, covert: 0.60, analysis: 0.65 },
    globalPresence: 'limited',
    majorPartnerships: ['Limited'],
    knownOperations: [
      { name: 'Kashmir operations', type: 'CI/CT', date: 'ongoing', target: 'Separatists' },
    ],
    notes: 'India\'s oldest intelligence agency. Heavy Kashmir focus. Coordination issues with R&AW.',
  },

  // Pakistan
  'isi': {
    id: 'isi',
    name: 'Inter-Services Intelligence (ISI)',
    country: 'Pakistan',
    countryCode: 'PK',
    type: 'military',
    capabilities: { humint: 0.85, sigint: 0.55, cyber: 0.50, covert: 0.90, analysis: 0.65 },
    globalPresence: 'regional',
    majorPartnerships: ['China', 'Saudi Arabia', 'US (strained)'],
    knownOperations: [
      { name: 'Taliban support', type: 'covert', date: '1990s-present', target: 'Afghanistan' },
      { name: 'Mumbai attacks (alleged)', type: 'covert', date: '2008', target: 'India' },
      { name: 'Bin Laden compound (disputed)', type: 'failure/complicity?', date: '2011', target: 'Unknown' },
    ],
    notes: 'State within a state. Created Taliban. Deep ties to militants. Bin Laden hiding raised questions. Now facing TTP blowback.',
  },

  // Saudi Arabia
  'gi': {
    id: 'saudi-gi',
    name: 'General Intelligence Presidency (GIP)',
    country: 'Saudi Arabia',
    countryCode: 'SA',
    type: 'combined',
    capabilities: { humint: 0.70, sigint: 0.65, cyber: 0.70, covert: 0.80, analysis: 0.60 },
    globalPresence: 'regional',
    majorPartnerships: ['US (complicated)', 'Egypt', 'UAE'],
    knownOperations: [
      { name: 'Khashoggi assassination', type: 'covert', date: '2018', target: 'Journalist' },
      { name: 'Yemen operations', type: 'covert/military', date: '2015+', target: 'Houthis' },
    ],
    notes: 'MBS consolidated control. Khashoggi murder was catastrophic PR. Heavy tech surveillance (Pegasus).',
  },

  // UAE
  'waed': {
    id: 'uae-waed',
    name: 'UAE Intelligence Services',
    country: 'UAE',
    countryCode: 'AE',
    type: 'combined',
    capabilities: { humint: 0.65, sigint: 0.70, cyber: 0.80, covert: 0.75, analysis: 0.65 },
    globalPresence: 'regional',
    majorPartnerships: ['US', 'Israel', 'Egypt'],
    knownOperations: [
      { name: 'Project Raven', type: 'cyber', date: '2016+', target: 'Journalists, activists' },
      { name: 'Sudan RSF support', type: 'covert', date: '2023+', target: 'Sudan' },
    ],
    notes: 'Punches above weight. Heavy cyber capabilities. Backing RSF in Sudan. Project Raven used ex-NSA.',
  },

  // North Korea
  'rgb': {
    id: 'rgb',
    name: 'Reconnaissance General Bureau (RGB)',
    country: 'North Korea',
    countryCode: 'KP',
    type: 'military',
    capabilities: { humint: 0.60, sigint: 0.40, cyber: 0.85, covert: 0.80, analysis: 0.50 },
    globalPresence: 'limited',
    majorPartnerships: ['China (limited)', 'Russia (growing)'],
    knownOperations: [
      { name: 'Sony hack', type: 'cyber', date: '2014', target: 'Sony Pictures' },
      { name: 'WannaCry ransomware', type: 'cyber', date: '2017', target: 'Global' },
      { name: 'Cryptocurrency theft', type: 'cyber', date: 'ongoing', target: 'Global ($3B+ stolen)' },
      { name: 'Kim Jong Nam assassination', type: 'covert', date: '2017', target: 'Kim\'s half-brother' },
    ],
    notes: 'Lazarus Group is RGB. World-class cyber for sanctions evasion. VX nerve agent in Malaysia airport.',
  },
};

// ============================================
// VENEZUELA CONDITIONAL THREAT ASSESSMENT
// ============================================

export const VENEZUELA_THREAT_ASSESSMENT: RegimeThreatAssessment = {
  target: 'Venezuela / Maduro Regime',
  threatActor: 'United States (Trump Administration)',
  baseAssessment: {
    level: 'critical',
    timeframe: 'months',
    confidence: 0.85,
    basis: 'Active military operations underway (Operation Southern Spear)',
  },
  conditionalScenarios: [
    // Scenario 1: Maximum Pressure Works - Maduro Flees
    {
      id: 'maduro-flees',
      name: 'Maximum Pressure Success',
      description: 'Maduro accepts safe passage and departs Venezuela voluntarily',
      conditions: [
        'Military leadership defects or signals willingness to defect',
        'Oil revenue collapses (China/Russia stop buying)',
        'Inner circle turns (family pressure, asset seizures)',
        'Third party (Pope, Brazil, Colombia) brokers face-saving deal',
      ],
      probability: 0.18,
      probabilityOverTime: [
        { months: 3, probability: 0.10 },
        { months: 6, probability: 0.15 },
        { months: 12, probability: 0.18 },
        { months: 24, probability: 0.20 },
      ],
      warningIndicators: [
        'Senior military figures seeking asylum',
        'Maduro family members leaving country',
        'China/Russia reducing oil imports significantly',
        'Maduro requesting negotiations through intermediaries',
      ],
      outcome: 'Peaceful transition - González government takes power. US claims victory. Russia/China accept fait accompli.',
    },
    // Scenario 2: Sustained Low-Intensity Conflict (MOST LIKELY)
    {
      id: 'sustained-pressure',
      name: 'Stalemate / Sustained Pressure',
      description: 'US continues strikes and sanctions, Maduro digs in, neither side escalates to full war',
      conditions: [
        'US domestic opposition prevents invasion (70% oppose)',
        'Maduro retains military loyalty',
        'Russia/China provide minimal support (not worth direct confrontation)',
        'No major incident triggers escalation',
      ],
      probability: 0.52,
      probabilityOverTime: [
        { months: 3, probability: 0.55 },
        { months: 6, probability: 0.52 },
        { months: 12, probability: 0.48 },
        { months: 24, probability: 0.42 },
      ],
      warningIndicators: [
        'Continued boat strikes at current tempo',
        'Sanctions tightened but no invasion preparation',
        'Both sides maintain public defiance',
        'No major change in force posture',
      ],
      outcome: 'Grinding stalemate. Venezuela isolated, impoverished. US maintains pressure but no regime change. Could last years.',
    },
    // Scenario 3: Full US Invasion
    {
      id: 'full-invasion',
      name: 'Operation Venezuelan Freedom',
      description: 'US launches full-scale military intervention to remove Maduro regime',
      conditions: [
        'Major incident (American casualties, terrorist attack on US soil attributed to Maduro)',
        'Congressional authorization or executive override',
        'Force buildup to 100,000+ troops (currently 15,000)',
        'Allied support (Brazil, Colombia) or unilateral action',
        'Trump prioritizes Venezuela over other crises',
      ],
      probability: 0.12,
      probabilityOverTime: [
        { months: 3, probability: 0.08 },
        { months: 6, probability: 0.12 },
        { months: 12, probability: 0.15 },
        { months: 24, probability: 0.10 }, // Decreases as other priorities emerge
      ],
      warningIndicators: [
        'Massive troop deployments to Caribbean, Colombia, Brazil borders',
        'Call-up of reserves',
        'Evacuation of US citizens from region',
        'Congressional debate on AUMF (Authorization for Use of Military Force)',
        'UN Security Council emergency sessions',
      ],
      outcome: 'Quick regime decapitation likely, but occupation would face insurgency. Iraq/Afghanistan lessons ignored at US peril.',
    },
    // Scenario 4: De-escalation / Negotiated Settlement
    {
      id: 'deescalation',
      name: 'Face-Saving Deal',
      description: 'Both sides find off-ramp through third-party mediation',
      conditions: [
        'Trump needs foreign policy "win" without casualties',
        'Maduro offered acceptable terms (amnesty but not power)',
        'Third party (Pope Leo XVI, Brazil Lula, Mexico) brokers deal',
        'Opposition (González) accepts power-sharing or elections',
        'Russia/China pressure Maduro to negotiate',
      ],
      probability: 0.22,
      probabilityOverTime: [
        { months: 3, probability: 0.15 },
        { months: 6, probability: 0.20 },
        { months: 12, probability: 0.25 },
        { months: 24, probability: 0.30 },
      ],
      warningIndicators: [
        'Secret back-channel talks reported',
        'Pope Leo XVI or other mediator visits',
        'Softening of rhetoric from both sides',
        'Maduro releasing political prisoners',
        'US quietly reducing strike tempo',
      ],
      outcome: 'Some form of transition government. Maduro exile. Neither side claims total victory.',
    },
    // Scenario 5: Regional Escalation - Essequibo Attack
    {
      id: 'essequibo-escalation',
      name: 'Maduro Attacks Guyana (Wag the Dog)',
      description: 'Maduro invades Essequibo to rally nationalist support and complicate US intervention',
      conditions: [
        'Maduro needs distraction from domestic pressure',
        'Calculates US won\'t fight two-front war',
        'Military needs "win" to maintain morale',
        'Believes Brazil/UK won\'t intervene quickly',
      ],
      probability: 0.08,
      probabilityOverTime: [
        { months: 3, probability: 0.05 },
        { months: 6, probability: 0.08 },
        { months: 12, probability: 0.10 },
        { months: 24, probability: 0.12 },
      ],
      warningIndicators: [
        'Major military movement toward Guyana border',
        'Nationalist rhetoric on Essequibo intensifies',
        'Venezuela closes border with Guyana',
        'ExxonMobil operations threatened',
      ],
      outcome: 'Would trigger immediate US/UK response. Likely Venezuelan defeat. Maduro regime end accelerated.',
    },
  ],
  actorDependencies: [
    {
      actor: 'Russia',
      influence: 0.25,
      currentStance: 'Verbal support, limited military aid, no appetite for direct confrontation',
      possibleShifts: [
        'Could send advisors to make US strikes more costly',
        'Could provide advanced SAMs if pressure mounts',
        'Unlikely to intervene directly - Ukraine consumes resources',
      ],
    },
    {
      actor: 'China',
      influence: 0.30,
      currentStance: 'Economic support (oil purchases), diplomatic criticism, avoiding military involvement',
      possibleShifts: [
        'Could threaten economic retaliation against US',
        'Could increase oil imports to keep Maduro afloat',
        'Will not risk US confrontation over Venezuela',
      ],
    },
    {
      actor: 'Iran',
      influence: 0.10,
      currentStance: 'Ideological ally, limited practical support',
      possibleShifts: [
        'Could provide drones/missiles (complicating factor)',
        'Already sending fuel and technical assistance',
        'Distracted by own regional crisis (Israel, Hezbollah)',
      ],
    },
    {
      actor: 'Brazil (Lula)',
      influence: 0.20,
      currentStance: 'Neutral mediator, opposes intervention, won\'t support Maduro openly',
      possibleShifts: [
        'Could offer mediation role',
        'Won\'t allow US to use Brazilian territory for invasion',
        'May accept refugees, provide humanitarian corridor',
      ],
    },
    {
      actor: 'Colombia (Petro)',
      influence: 0.15,
      currentStance: 'Left-leaning but won\'t support Maduro regime, border tensions',
      possibleShifts: [
        'Could become staging area for US forces (unlikely under Petro)',
        'May face internal pressure if conflict escalates',
        'ELN/FARC dissident safe havens complicate position',
      ],
    },
    {
      actor: 'Venezuelan Military (FANB)',
      influence: 0.90,
      currentStance: 'Loyal to Maduro - purged of dissenters, economically tied to regime',
      possibleShifts: [
        'Only internal actor that could change outcome',
        'If top generals defect, regime falls quickly',
        'Currently: No visible cracks, heavy surveillance of officers',
      ],
    },
  ],
  historicalAnalogs: [
    {
      name: 'Panama 1989 (Operation Just Cause)',
      relevance: 0.75,
      lessons: 'Quick decapitation possible. Noriega captured in 6 weeks. But Panama had 75K population, Venezuela has 28M.',
    },
    {
      name: 'Iraq 2003',
      relevance: 0.60,
      lessons: 'Regime fell quickly, but occupation/insurgency was catastrophic. Venezuela terrain + population = similar risk.',
    },
    {
      name: 'Libya 2011',
      relevance: 0.50,
      lessons: 'Intervention without ground troops. Regime fell but country collapsed into chaos. No follow-through.',
    },
    {
      name: 'Cuba 1961 (Bay of Pigs)',
      relevance: 0.65,
      lessons: 'Exile force failed when expected popular uprising didn\'t materialize. Maduro has suppressed opposition.',
    },
    {
      name: 'Venezuela 2019 (Guaidó Attempt)',
      relevance: 0.85,
      lessons: 'Recognition without force = nothing. Guaidó recognized by 50+ countries, still failed. Military stayed loyal.',
    },
  ],
  keyDates: [
    { date: '2025-11-28', event: 'Trump ultimatum expired' },
    { date: '2025-12-03', event: 'Current date - active low-intensity conflict' },
    { date: '2026-01-10', event: 'One year since Trump recognition of González' },
    { date: '2026-07-28', event: 'Two-year anniversary of stolen election' },
    { date: '2026-11-XX', event: 'US midterms - political pressure on Trump Venezuela policy' },
    { date: '2028-11-XX', event: 'Next US presidential election - policy continuity uncertain' },
  ],
  analystConfidence: {
    overall: 0.75,
    dataQuality: 'High - extensive open source reporting, but fog of war on ground truth',
    keyUncertainties: [
      'Venezuelan military loyalty under pressure',
      'Russia/China willingness to escalate',
      'Trump administration decision-making process',
      'Actual civilian casualties from boat strikes (claims of fishermen)',
    ],
  },
  lastUpdated: '2025-12-03',
};

// ============================================
// HELPER FUNCTIONS
// ============================================

export function getAllGlobalFlashpoints(): TerritorialDispute[] {
  return [
    ...Object.values(AFRICA_DISPUTES),
    ...Object.values(LATIN_AMERICA_DISPUTES),
    ...Object.values(EXTENDED_ASIA_DISPUTES),
  ];
}

export function getFlashpointsByRisk(minRisk: number = 0.5): TerritorialDispute[] {
  return getAllGlobalFlashpoints()
    .filter(d => d.escalationRisk >= minRisk)
    .sort((a, b) => b.escalationRisk - a.escalationRisk);
}

export function getAllDomesticInstability(): DomesticInstability[] {
  return Object.values(EXTENDED_DOMESTIC_INSTABILITY)
    .sort((a, b) => {
      const severityOrder = { critical: 5, high: 4, elevated: 3, moderate: 2, low: 1 };
      return severityOrder[b.severity] - severityOrder[a.severity];
    });
}

export function getAllIntelAgencies(): IntelligenceAgency[] {
  return Object.values(EXTENDED_INTEL_AGENCIES);
}

export function getIntelAgenciesByCountry(countryCode: string): IntelligenceAgency[] {
  return Object.values(EXTENDED_INTEL_AGENCIES)
    .filter(a => a.countryCode === countryCode);
}
