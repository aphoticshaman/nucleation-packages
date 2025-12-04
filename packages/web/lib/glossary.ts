/**
 * LatticeForge Terminology Glossary
 *
 * Three-tier definitions matching skill levels:
 * - simple: High school sophomore level
 * - standard: College/industry analyst level
 * - detailed: Post-PhD, CIA consultant level
 */

export interface GlossaryTerm {
  term: string;
  category: 'core' | 'metrics' | 'visualization' | 'methodology' | 'intelligence';
  simple: string;
  standard: string;
  detailed: string;
  related?: string[];
}

export const GLOSSARY: GlossaryTerm[] = [
  // ===== CORE CONCEPTS =====
  {
    term: 'Basin',
    category: 'core',
    simple: 'How stable a country is. Like a ball in a bowl - deep bowl = hard to knock out = stable.',
    standard: 'A stability attractor state. Countries in deep basins resist change; shallow basins indicate vulnerability to disruption.',
    detailed: 'A region in phase space where trajectories converge toward an equilibrium. Basin depth (Lyapunov exponent) quantifies resistance to perturbation. Shallow basins indicate proximity to bifurcation points.',
    related: ['Attractor', 'Stability', 'Phase Space'],
  },
  {
    term: 'Attractor',
    category: 'core',
    simple: 'The normal state a country tends to settle into, like water settling in a lake.',
    standard: 'The equilibrium state a nation tends toward. Changes in attractors signal regime shifts.',
    detailed: 'A set in phase space toward which a dynamical system evolves over time. Can be point attractors (stable states), limit cycles (oscillations), or strange attractors (chaotic dynamics).',
    related: ['Basin', 'Phase Space', 'Regime'],
  },
  {
    term: 'Transition',
    category: 'core',
    simple: 'When a country changes from one type of government or situation to another.',
    standard: 'A shift between stable states - e.g., from democracy to autocracy, or from peace to conflict.',
    detailed: 'Movement between attractor basins in phase space. Transitions can be continuous (gradual drift) or discontinuous (sudden jumps triggered by bifurcations or external shocks).',
    related: ['Basin', 'Phase Transition', 'Bifurcation'],
  },
  {
    term: 'Phase Space',
    category: 'core',
    simple: 'An imaginary map where we plot everything about a country to see its overall "health".',
    standard: 'A multi-dimensional space where each axis represents a different factor (economy, politics, etc). A nation\'s position shows its current state.',
    detailed: 'An n-dimensional manifold where each dimension represents an independent state variable. The system\'s complete state is a single point; its evolution traces a trajectory through this space.',
    related: ['State Vector', 'Trajectory', 'Manifold'],
  },
  {
    term: 'Regime',
    category: 'core',
    simple: 'The type of government a country has (democracy, dictatorship, etc).',
    standard: 'A political system classification based on how power is held and transferred. Measured on scales like Polity V.',
    detailed: 'Regime typology per V-Dem polyarchy indices: electoral democracy, liberal democracy, electoral autocracy, or closed autocracy. Includes executive constraints, competitive participation, and civil liberties scores.',
    related: ['Polity V', 'V-Dem', 'Autocracy Index'],
  },

  // ===== METRICS =====
  {
    term: 'Basin Depth',
    category: 'metrics',
    simple: 'How hard it is to destabilize a country. Deeper = more stable.',
    standard: 'A 0-100 score measuring resistance to disruption. Higher scores mean more stability.',
    detailed: 'Quantified via Lyapunov stability coefficient. Derived from eigenvalue analysis of the Jacobian at equilibrium. Expressed as normalized depth in arbitrary units.',
    related: ['Stability', 'Lyapunov Exponent'],
  },
  {
    term: 'Transition Risk',
    category: 'metrics',
    simple: 'The chance something big will change in the next few months (0-100%).',
    standard: 'Probability of a major political or economic shift in the 30/60/90-day horizon.',
    detailed: 'Monte Carlo-derived probability of basin exit within forecast window. Computed from 10K stochastic trajectory samples with empirical prior initialization.',
    related: ['Monte Carlo', 'Forecast Horizon', 'Probability Distribution'],
  },
  {
    term: 'Influence Coefficient',
    category: 'metrics',
    simple: 'How much one country affects another (like how the US affects Canada\'s economy).',
    standard: 'A measure of how changes in one nation propagate to others via trade, alliances, or proximity.',
    detailed: 'Edge weight in the bilateral influence graph. Derived from Granger causality tests on lagged state variables, trade dependency ratios, and alliance treaty obligations.',
    related: ['Coupling', 'Cascade', 'Contagion'],
  },
  {
    term: 'Eigenvector Centrality',
    category: 'metrics',
    simple: 'How important a country is in the global network. Being connected to important countries makes you important.',
    standard: 'A measure of influence based not just on connections, but on the importance of those connections.',
    detailed: 'Centrality score from the dominant eigenvector of the adjacency matrix. High scores indicate nodes connected to other high-scoring nodes.',
    related: ['Network Analysis', 'Influence', 'Graph Theory'],
  },

  // ===== VISUALIZATION =====
  {
    term: 'Color Saturation',
    category: 'visualization',
    simple: 'How bright or faded the color is. Brighter = more of that thing.',
    standard: 'Intensity of color encoding. Higher saturation indicates stronger signal.',
    detailed: 'HSL saturation channel mapped to primary metric value. Linear interpolation from 20% (minimum) to 100% (maximum) across the metric range.',
    related: ['Color Encoding', 'Visual Channel'],
  },
  {
    term: 'Node Radius',
    category: 'visualization',
    simple: 'How big the circle is on the map. Bigger = more important/influential.',
    standard: 'Circle size represents relative influence or importance in the network.',
    detailed: 'Radius proportional to sqrt(eigenvector_centrality) for perceptual area scaling. Minimum radius ensures visibility; maximum prevents occlusion.',
    related: ['Eigenvector Centrality', 'Visual Encoding'],
  },
  {
    term: 'Edge Weight',
    category: 'visualization',
    simple: 'How thick the line between countries is. Thicker = stronger connection.',
    standard: 'Line thickness shows strength of bilateral relationship or influence.',
    detailed: 'Stroke width mapped to bilateral coupling coefficient. Log-scaled to compress dynamic range. Opacity modulated by statistical significance.',
    related: ['Influence Coefficient', 'Network Graph'],
  },

  // ===== METHODOLOGY =====
  {
    term: 'Monte Carlo',
    category: 'methodology',
    simple: 'Running thousands of "what if" simulations to see what might happen.',
    standard: 'A simulation technique that runs many random scenarios to estimate probabilities.',
    detailed: 'Stochastic simulation via repeated random sampling from probability distributions. Used to propagate uncertainty through nonlinear dynamics and estimate transition probabilities.',
    related: ['Simulation', 'Stochastic', 'Probability'],
  },
  {
    term: 'Bayesian Update',
    category: 'methodology',
    simple: 'Updating our predictions as new information comes in.',
    standard: 'A method for revising probability estimates when new evidence arrives.',
    detailed: 'Posterior = (Likelihood × Prior) / Evidence. Applied to transition probability distributions after each data ingestion cycle.',
    related: ['Prior', 'Posterior', 'Likelihood'],
  },
  {
    term: 'OSINT',
    category: 'methodology',
    simple: 'Information gathered from public sources like news, social media, and government reports.',
    standard: 'Open Source Intelligence - publicly available information used for analysis.',
    detailed: 'Open Source Intelligence. Includes GDELT event data, ACLED conflict tracking, RSS/API news feeds, social media sentiment, and official government publications.',
    related: ['GDELT', 'ACLED', 'Sentiment Analysis'],
  },
  {
    term: 'HUMINT',
    category: 'methodology',
    simple: 'Information from talking to actual people on the ground.',
    standard: 'Human Intelligence - information gathered from human sources.',
    detailed: 'Human Intelligence. Ground-truth validation from subject matter experts, diplomatic cables, and field reports. Used to calibrate and validate automated assessments.',
    related: ['Ground Truth', 'Validation'],
  },
  {
    term: 'Brier Score',
    category: 'methodology',
    simple: 'A way to measure how good our predictions are. Lower = better.',
    standard: 'A metric for prediction accuracy. 0 = perfect, 1 = worst possible.',
    detailed: 'Mean squared error between probabilistic forecasts and binary outcomes. Decomposable into reliability, resolution, and uncertainty components. Used for forecast calibration.',
    related: ['Calibration', 'Accuracy', 'Forecast Verification'],
  },

  // ===== INTELLIGENCE TERMS =====
  {
    term: 'Flashpoint',
    category: 'intelligence',
    simple: 'A place where fighting or major conflict could start at any moment.',
    standard: 'A location or situation with high potential for rapid escalation to conflict.',
    detailed: 'Geographic or thematic focal point with elevated transition probability and high-stakes outcomes. Typically involves territorial disputes, ethnic tensions, or resource competition.',
    related: ['Escalation', 'Conflict', 'Hot Spot'],
  },
  {
    term: 'Gray Zone',
    category: 'intelligence',
    simple: 'Actions that aren\'t quite war but aren\'t quite peace either (like cyberattacks or economic pressure).',
    standard: 'Activities between peace and war - coercion, disinformation, cyber ops - that stay below the threshold of open conflict.',
    detailed: 'Competitive interactions below the threshold of armed conflict but above routine statecraft. Includes hybrid warfare, information operations, economic coercion, and proxy activities.',
    related: ['Hybrid Warfare', 'Escalation Ladder', 'Coercion'],
  },
  {
    term: 'Black Swan',
    category: 'intelligence',
    simple: 'A surprise event that nobody saw coming but has huge impact.',
    standard: 'A rare, unpredictable event with extreme consequences. Hard to forecast by definition.',
    detailed: 'High-impact, low-probability event outside normal expectation bounds. Characterized by extreme deviation from historical distributions. By definition, not capturable by standard forecasting.',
    related: ['Tail Risk', 'Discontinuity', 'Fat Tails'],
  },
  {
    term: 'Cascade',
    category: 'intelligence',
    simple: 'When problems in one country spread to others, like dominoes falling.',
    standard: 'A chain reaction where instability in one nation triggers instability in connected nations.',
    detailed: 'Propagation of state changes through the influence network. Modeled via threshold dynamics on weighted graphs. Can exhibit phase transitions from localized to systemic.',
    related: ['Contagion', 'Systemic Risk', 'Network Effects'],
  },
  {
    term: 'NSM (Next Strategic Move)',
    category: 'intelligence',
    simple: 'Our best guess about what a decision-maker should consider doing next.',
    standard: 'Actionable recommendation for policymakers based on current intelligence.',
    detailed: 'Decision-support synthesis integrating multi-domain assessments into prioritized action options. Weighted by impact magnitude, confidence level, and decision window urgency.',
    related: ['Actionable Intelligence', 'Decision Support'],
  },
];

// Group terms by category
export function getTermsByCategory(category: GlossaryTerm['category']): GlossaryTerm[] {
  return GLOSSARY.filter(term => term.category === category);
}

// Search terms
export function searchGlossary(query: string): GlossaryTerm[] {
  const q = query.toLowerCase();
  return GLOSSARY.filter(term =>
    term.term.toLowerCase().includes(q) ||
    term.simple.toLowerCase().includes(q) ||
    term.standard.toLowerCase().includes(q) ||
    term.detailed.toLowerCase().includes(q) ||
    term.related?.some(r => r.toLowerCase().includes(q))
  );
}

// Get related terms
export function getRelatedTerms(termName: string): GlossaryTerm[] {
  const term = GLOSSARY.find(t => t.term === termName);
  if (!term?.related) return [];
  return GLOSSARY.filter(t => term.related!.includes(t.term));
}

// ===== QUANT & FINANCIAL TERMS =====
const QUANT_TERMS: GlossaryTerm[] = [
  {
    term: 'Sharpe Ratio',
    category: 'metrics',
    simple: 'How much return you get for the risk you take. Higher is better.',
    standard: 'Return per unit of volatility. Measures risk-adjusted performance. Above 1 is good, above 2 is excellent.',
    detailed: 'Excess return over risk-free rate divided by standard deviation: (Rp - Rf) / σp. Assumes normal returns (problematic for fat tails). Annualize by multiplying by √252 for daily data.',
    related: ['Volatility', 'Alpha', 'Risk-Adjusted Return'],
  },
  {
    term: 'Value at Risk',
    category: 'metrics',
    simple: 'The most you could lose on a bad day (95% of the time).',
    standard: 'Maximum expected loss at a confidence level. 95% 1-day VaR means 19 of 20 days, losses stay below this number.',
    detailed: 'Quantile of P&L distribution at specified confidence (typically 95% or 99%). Does not capture tail severity. Supplement with CVaR/Expected Shortfall for tail risk.',
    related: ['Volatility', 'Drawdown', 'Tail Risk'],
  },
  {
    term: 'Correlation',
    category: 'metrics',
    simple: 'How two things move together. +1 = same direction, -1 = opposite, 0 = no relationship.',
    standard: 'Measure of co-movement between assets. Diversification needs low or negative correlations.',
    detailed: 'Pearson correlation coefficient ρ = Cov(X,Y) / (σX × σY). Ranges [-1, +1]. Warning: correlations spike in crises ("all correlations go to 1"). Use rolling windows and regime-switching models.',
    related: ['Diversification', 'Contagion', 'Beta'],
  },
  {
    term: 'Volatility',
    category: 'metrics',
    simple: 'How much prices swing around. High volatility = wild swings.',
    standard: 'Standard deviation of returns. Can be historical (past data) or implied (from options prices).',
    detailed: 'σ = √[Σ(ri - r̄)² / (n-1)]. Annualize by √252 for daily data. Vol clusters (GARCH effects). Implied vol from Black-Scholes inversion. VIX measures 30-day S&P 500 implied vol.',
    related: ['Value at Risk', 'Sharpe Ratio', 'Options'],
  },
  {
    term: 'Alpha',
    category: 'metrics',
    simple: 'Returns from skill, not just riding the market up.',
    standard: 'Excess return above what market exposure would explain. True alpha is rare and valuable.',
    detailed: 'Jensen\'s alpha: α = Rp - [Rf + β(Rm - Rf)]. Residual after accounting for factor exposures. Much "alpha" is actually unrecognized factor exposure. Decays as it gets arbitraged.',
    related: ['Beta', 'Sharpe Ratio', 'Factor Exposure'],
  },
  {
    term: 'Beta',
    category: 'metrics',
    simple: 'How much an investment moves compared to the whole market.',
    standard: 'Sensitivity to market movements. Beta 1.5 means 50% more volatile than market. Beta 0.5 is half as volatile.',
    detailed: 'β = Cov(Ri, Rm) / Var(Rm). Systematic risk that cannot be diversified away. Can be negative (gold vs. stocks sometimes). Rolling beta captures time variation.',
    related: ['Alpha', 'Volatility', 'CAPM'],
  },
  {
    term: 'Drawdown',
    category: 'metrics',
    simple: 'How far your investment fell from its peak before recovering.',
    standard: 'Peak-to-trough decline. Max drawdown is worst historical drop. Critical for survival - 50% loss requires 100% gain to recover.',
    detailed: 'DD(t) = (Peak(t) - Value(t)) / Peak(t). Max DD = max[DD(t)]. Recovery time also matters. Many strategies fail during drawdowns due to capital calls, redemptions, or psychological capitulation.',
    related: ['Value at Risk', 'Volatility', 'Risk Management'],
  },
  {
    term: 'Sovereign Spread',
    category: 'metrics',
    simple: 'Extra interest a risky country pays to borrow vs. a safe country.',
    standard: 'Yield premium of government bonds over risk-free benchmarks (usually US Treasuries). Measures perceived country risk.',
    detailed: 'Spread = Yield(country) - Yield(benchmark). Decomposes into default risk, currency risk, and liquidity premium. CDS spreads provide purer default risk signal. Widening spreads often precede crises.',
    related: ['Credit Risk', 'Sovereign Default', 'Capital Flight'],
  },
  {
    term: 'Contagion',
    category: 'intelligence',
    simple: 'When crisis in one place spreads to others through panic or real connections.',
    standard: 'Financial crisis transmission across markets/countries. Can be through actual linkages (trade, banking) or pure sentiment.',
    detailed: 'Contagion vs. interdependence: true contagion shows correlation breakdown (Forbes-Rigobon test). Channels: trade, banking exposure, common creditor, wake-up calls. Correlation spikes mask diversification failure.',
    related: ['Cascade', 'Correlation', 'Systemic Risk'],
  },
  {
    term: 'Capital Flight',
    category: 'intelligence',
    simple: 'When money rapidly leaves a country because investors are scared.',
    standard: 'Rapid outflow of capital due to crisis or loss of confidence. Often self-reinforcing through currency depreciation.',
    detailed: 'Measure via BOP financial account, bank deposit changes, or parallel FX premium. Early indicators: wealthy locals moving money abroad, real estate purchases in "safe" countries. Often precedes currency crisis.',
    related: ['Contagion', 'Currency Crisis', 'Sovereign Spread'],
  },
];

// Add quant terms to main glossary
QUANT_TERMS.forEach(term => GLOSSARY.push(term));

// Category labels
export const CATEGORY_LABELS: Record<GlossaryTerm['category'], string> = {
  core: 'Core Concepts',
  metrics: 'Metrics & Scores',
  visualization: 'Map & Charts',
  methodology: 'How It Works',
  intelligence: 'Intelligence Terms',
};

/**
 * Get tooltip text based on user skill level
 */
export function getTooltipForLevel(termName: string, level: 'simple' | 'standard' | 'detailed'): string | undefined {
  const term = GLOSSARY.find(t => t.term.toLowerCase() === termName.toLowerCase());
  return term?.[level];
}

/**
 * Get a quick one-liner for a term (uses simple level)
 */
export function getQuickTip(termName: string): string | undefined {
  return getTooltipForLevel(termName, 'simple');
}
