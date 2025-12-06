/**
 * Pricing Components - Anti-Complaint Spec Section 1 Implementation
 *
 * Implements Radical Price Transparency requirements:
 * - TCOCalculator: 5-year total cost of ownership projections
 * - ROIDashboard: Value demonstration with time savings, threat detection, FP rates
 * - APIUsageSimulator: API call pricing with workload-based estimation
 *
 * All pricing visible without sales contact.
 * No hidden costs. No "contact us for pricing".
 */

export { default as TCOCalculator } from './TCOCalculator';
export { default as ROIDashboard } from './ROIDashboard';
export { default as APIUsageSimulator } from './APIUsageSimulator';
