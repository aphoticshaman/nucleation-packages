/**
 * Input validation utilities for nucleation detectors
 */

import type { DetectorConfig, Sensitivity } from './types.js';

/**
 * Custom error class for nucleation-specific errors
 */
export class NucleationError extends Error {
  public readonly code: string;

  constructor(message: string, code: string) {
    super(message);
    this.name = 'NucleationError';
    this.code = code;

    // Maintains proper stack trace for where our error was thrown (only available on V8)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, NucleationError);
    }
  }
}

/**
 * Validate a numeric value for detector input
 *
 * @param value - The value to validate
 * @param name - Name of the parameter for error messages
 * @throws NucleationError if validation fails
 */
export function validateNumber(value: unknown, name = 'value'): number {
  if (typeof value !== 'number') {
    throw new NucleationError(`${name} must be a number, got ${typeof value}`, 'INVALID_TYPE');
  }

  if (!Number.isFinite(value)) {
    throw new NucleationError(`${name} must be a finite number, got ${value}`, 'INVALID_VALUE');
  }

  return value;
}

/**
 * Validate an array of numbers
 *
 * @param values - Array to validate
 * @param name - Name of the parameter for error messages
 * @throws NucleationError if validation fails
 */
export function validateNumberArray(values: unknown, name = 'values'): number[] {
  if (!Array.isArray(values) && !(values instanceof Float64Array)) {
    throw new NucleationError(`${name} must be an array or Float64Array`, 'INVALID_TYPE');
  }

  if (values.length === 0) {
    throw new NucleationError(`${name} cannot be empty`, 'EMPTY_ARRAY');
  }

  const arr = Array.isArray(values) ? values : Array.from(values);

  for (let i = 0; i < arr.length; i++) {
    const val = arr[i];
    if (typeof val !== 'number' || !Number.isFinite(val)) {
      throw new NucleationError(
        `${name}[${i}] must be a finite number, got ${val}`,
        'INVALID_VALUE'
      );
    }
  }

  return arr;
}

const VALID_SENSITIVITIES: Sensitivity[] = ['conservative', 'balanced', 'sensitive'];

/**
 * Validate detector configuration
 *
 * @param config - Configuration to validate
 * @returns Validated configuration (may be modified)
 * @throws NucleationError if validation fails
 */
export function validateConfig(config: DetectorConfig): DetectorConfig {
  const validated: DetectorConfig = { ...config };

  if (config.sensitivity !== undefined && !VALID_SENSITIVITIES.includes(config.sensitivity)) {
    throw new NucleationError(
      `sensitivity must be one of: ${VALID_SENSITIVITIES.join(', ')}`,
      'INVALID_CONFIG'
    );
  }

  if (config.windowSize !== undefined) {
    if (typeof config.windowSize !== 'number' || !Number.isInteger(config.windowSize)) {
      throw new NucleationError('windowSize must be an integer', 'INVALID_CONFIG');
    }
    if (config.windowSize < 2) {
      throw new NucleationError('windowSize must be at least 2', 'INVALID_CONFIG');
    }
    if (config.windowSize > 10000) {
      throw new NucleationError('windowSize cannot exceed 10000', 'INVALID_CONFIG');
    }
  }

  if (config.threshold !== undefined) {
    if (typeof config.threshold !== 'number' || !Number.isFinite(config.threshold)) {
      throw new NucleationError('threshold must be a finite number', 'INVALID_CONFIG');
    }
    if (config.threshold <= 0) {
      throw new NucleationError('threshold must be positive', 'INVALID_CONFIG');
    }
  }

  return validated;
}
