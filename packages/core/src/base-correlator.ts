/**
 * Base correlator class for multi-source correlation detection.
 *
 * Used for detecting convergence patterns across multiple data sources,
 * such as threat correlation, cohort analysis, or M&A integration monitoring.
 */

import { initialize, getModule, type ShepherdInstance } from './wasm-loader.js';
import { NucleationError } from './validation.js';

/**
 * Abstract base class for multi-source correlators.
 *
 * @typeParam TMetadata - Type for entity metadata
 */
export abstract class BaseCorrelator<TMetadata = Record<string, unknown>> {
  /** The underlying WASM Shepherd instance */
  protected shepherd: ShepherdInstance | null = null;

  /** Number of behavior categories to track */
  protected readonly categories: number;

  /** Whether the correlator has been initialized */
  private initialized = false;

  /** Registered entities with their metadata */
  protected readonly entities = new Map<string, TMetadata>();

  /**
   * Create a new correlator instance.
   *
   * @param categories - Number of behavior categories to track
   */
  constructor(categories = 10) {
    if (!Number.isInteger(categories) || categories < 1) {
      throw new NucleationError('categories must be a positive integer', 'INVALID_CONFIG');
    }
    if (categories > 1000) {
      throw new NucleationError('categories cannot exceed 1000', 'INVALID_CONFIG');
    }
    this.categories = categories;
  }

  /**
   * Initialize the correlator. Must be called before use.
   */
  async init(): Promise<void> {
    if (this.initialized) {
      return;
    }

    await initialize();
    const module = getModule();
    const { Shepherd } = module;

    this.shepherd = new Shepherd(this.categories);
    this.initialized = true;
  }

  /**
   * Ensure the correlator is initialized before use.
   */
  protected ensureInit(): void {
    if (!this.initialized || !this.shepherd) {
      throw new NucleationError(
        `${this.constructor.name} not initialized. Call init() first.`,
        'NOT_INITIALIZED'
      );
    }
  }

  /**
   * Register a new entity (source, user, team, etc.).
   *
   * @param entityId - Unique identifier for the entity
   * @param metadata - Optional metadata for the entity
   * @param initialProfile - Optional initial behavior distribution
   */
  registerEntity(
    entityId: string,
    metadata?: TMetadata,
    initialProfile?: Float64Array | null
  ): void {
    this.ensureInit();

    if (typeof entityId !== 'string' || entityId.length === 0) {
      throw new NucleationError('entityId must be a non-empty string', 'INVALID_VALUE');
    }

    this.shepherd!.registerActor(entityId, initialProfile ?? null);
    if (metadata !== undefined) {
      this.entities.set(entityId, metadata);
    }
  }

  /**
   * Update an entity with a new behavioral observation.
   *
   * @param entityId - Entity identifier
   * @param observation - Behavior distribution across categories
   * @param timestamp - Optional timestamp (defaults to now)
   * @returns Any alerts triggered by this update
   */
  updateEntity(entityId: string, observation: Float64Array, timestamp = Date.now()): unknown[] {
    this.ensureInit();

    if (!this.entities.has(entityId)) {
      this.registerEntity(entityId);
    }

    if (observation.length !== this.categories) {
      throw new NucleationError(
        `observation must have ${this.categories} elements, got ${observation.length}`,
        'INVALID_VALUE'
      );
    }

    return this.shepherd!.updateActor(entityId, observation, timestamp);
  }

  /**
   * Get correlation/conflict score between two entities.
   *
   * @param entityA - First entity ID
   * @param entityB - Second entity ID
   * @returns Correlation score, or undefined if not available
   */
  getCorrelation(entityA: string, entityB: string): number | undefined {
    this.ensureInit();
    return this.shepherd!.conflictPotential(entityA, entityB);
  }

  /**
   * Check all entity pairs for correlation alerts.
   *
   * @param timestamp - Optional timestamp (defaults to now)
   * @returns All triggered alerts
   */
  checkAllPairs(timestamp = Date.now()): unknown[] {
    this.ensureInit();
    return this.shepherd!.checkAllDyads(timestamp);
  }

  /**
   * Get all registered entity IDs.
   */
  getEntityIds(): string[] {
    return Array.from(this.entities.keys());
  }

  /**
   * Get metadata for an entity.
   *
   * @param entityId - Entity identifier
   * @returns Entity metadata, or undefined if not found
   */
  getEntityMetadata(entityId: string): TMetadata | undefined {
    return this.entities.get(entityId);
  }

  /**
   * Check if an entity is registered.
   */
  hasEntity(entityId: string): boolean {
    return this.entities.has(entityId);
  }

  /**
   * Get the number of registered entities.
   */
  get entityCount(): number {
    return this.entities.size;
  }
}
