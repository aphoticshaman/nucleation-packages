'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  hamiltonFilter,
  kimSmoother,
  detectRegimeChanges,
  expectedRegimeDuration,
  stationaryDistribution,
  forecastRegimes,
  generateRegimeVisualization,
  type FilteredState,
  type SmoothedState,
  type RegimeChangeEvent,
  type RegimeVisualizationData,
  type MarkovSwitchingConfig,
  type RegimeId,
  REGIME_NAMES,
} from '@/lib/physics/markov-switching';

export interface UseRegimeDetectionOptions {
  config?: Partial<MarkovSwitchingConfig>;
  forecastHorizons?: number[];
  autoUpdate?: boolean;
  updateInterval?: number;  // ms
}

export interface RegimeDetectionState {
  currentRegime: RegimeId;
  regimeName: string;
  regimeProbabilities: number[];
  transitionRisk: number;
  filtered: FilteredState[];
  smoothed: SmoothedState[];
  changes: RegimeChangeEvent[];
  forecasts: Map<number, number[]>;
  visualization: RegimeVisualizationData[];
  statistics: {
    expectedDurations: number[];
    stationaryDistribution: number[];
    mostLikelyTransition: { from: RegimeId; to: RegimeId; prob: number } | null;
  };
  isProcessing: boolean;
  lastUpdate: string;
}

const DEFAULT_CONFIG: MarkovSwitchingConfig = {
  numRegimes: 3,
  regimeParams: [
    { mean: 0.02, variance: 0.01, color: '#3b82f6' },
    { mean: 0.0, variance: 0.04, color: '#f59e0b' },
    { mean: -0.05, variance: 0.09, color: '#ef4444' },
  ],
  transitionMatrix: [
    [0.95, 0.04, 0.01],
    [0.10, 0.80, 0.10],
    [0.05, 0.15, 0.80],
  ],
  initialProbs: [0.7, 0.2, 0.1],
};

/**
 * Hook for Markov-switching regime detection
 *
 * Implements Hamilton filter and Kim smoother for real-time
 * regime probability estimation and forecasting.
 */
export function useRegimeDetection(
  observations: number[],
  options: UseRegimeDetectionOptions = {}
) {
  const {
    config: userConfig,
    forecastHorizons = [1, 5, 10, 30],
    autoUpdate = true,
    updateInterval = 5000,
  } = options;

  const config = useMemo<MarkovSwitchingConfig>(() => ({
    ...DEFAULT_CONFIG,
    ...userConfig,
  }), [userConfig]);

  const [state, setState] = useState<RegimeDetectionState>({
    currentRegime: 0,
    regimeName: REGIME_NAMES[0],
    regimeProbabilities: config.initialProbs,
    transitionRisk: 0,
    filtered: [],
    smoothed: [],
    changes: [],
    forecasts: new Map(),
    visualization: [],
    statistics: {
      expectedDurations: [],
      stationaryDistribution: [],
      mostLikelyTransition: null,
    },
    isProcessing: false,
    lastUpdate: new Date().toISOString(),
  });

  // Process observations
  const processObservations = useCallback(() => {
    if (observations.length < 10) {
      return;
    }

    setState(prev => ({ ...prev, isProcessing: true }));

    try {
      // Hamilton filter
      const filtered = hamiltonFilter(observations, config);

      // Kim smoother
      const smoothed = kimSmoother(filtered, config);

      // Detect regime changes
      const changes = detectRegimeChanges(smoothed);

      // Current state
      const currentState = smoothed[smoothed.length - 1];
      const currentRegime = currentState.mostLikelyRegime;
      const transitionRisk = 1 - config.transitionMatrix[currentRegime][currentRegime];

      // Forecasts
      const forecasts = forecastRegimes(currentState.smoothedProbs, forecastHorizons, config);

      // Visualization data
      const visualization = generateRegimeVisualization(smoothed, config);

      // Statistics
      const expectedDurations = expectedRegimeDuration(config);
      const stationary = stationaryDistribution(config);

      // Find most likely next transition
      let mostLikelyTransition: { from: RegimeId; to: RegimeId; prob: number } | null = null;
      let maxTransitionProb = 0;
      for (let from = 0; from < config.numRegimes; from++) {
        for (let to = 0; to < config.numRegimes; to++) {
          if (from !== to) {
            const prob = currentState.smoothedProbs[from] * config.transitionMatrix[from][to];
            if (prob > maxTransitionProb) {
              maxTransitionProb = prob;
              mostLikelyTransition = { from: from as RegimeId, to: to as RegimeId, prob };
            }
          }
        }
      }

      setState({
        currentRegime,
        regimeName: REGIME_NAMES[currentRegime],
        regimeProbabilities: currentState.smoothedProbs,
        transitionRisk,
        filtered,
        smoothed,
        changes,
        forecasts,
        visualization,
        statistics: {
          expectedDurations,
          stationaryDistribution: stationary,
          mostLikelyTransition,
        },
        isProcessing: false,
        lastUpdate: new Date().toISOString(),
      });
    } catch (error) {
      console.error('Regime detection error:', error);
      setState(prev => ({ ...prev, isProcessing: false }));
    }
  }, [observations, config, forecastHorizons]);

  // Initial processing and updates
  useEffect(() => {
    processObservations();
  }, [processObservations]);

  // Auto-update
  useEffect(() => {
    if (!autoUpdate) return;

    const interval = setInterval(() => {
      processObservations();
    }, updateInterval);

    return () => clearInterval(interval);
  }, [autoUpdate, updateInterval, processObservations]);

  // Manual refresh
  const refresh = useCallback(() => {
    processObservations();
  }, [processObservations]);

  // Get regime info
  const getRegimeInfo = useCallback((regimeId: RegimeId) => {
    return {
      id: regimeId,
      name: REGIME_NAMES[regimeId],
      params: config.regimeParams[regimeId],
      expectedDuration: state.statistics.expectedDurations[regimeId],
      stationaryProbability: state.statistics.stationaryDistribution[regimeId],
    };
  }, [config.regimeParams, state.statistics]);

  // Get forecast for specific horizon
  const getForecast = useCallback((horizon: number) => {
    return state.forecasts.get(horizon) || config.initialProbs;
  }, [state.forecasts, config.initialProbs]);

  return {
    ...state,
    refresh,
    getRegimeInfo,
    getForecast,
    config,
  };
}
