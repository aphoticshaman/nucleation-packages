'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  type PhaseState,
  type LandauGinzburgConfig,
  type TransitionEvent,
  langevinStep,
  detectTransitions,
  kramersEscapeRate,
  createPhaseState,
  stressToLandauCoefficient,
} from '@/lib/physics/landau-ginzburg';

export interface UsePhaseTransitionOptions {
  initialStability?: number;
  initialRegime?: 'stable' | 'volatile' | 'crisis';
  stressFactors?: {
    economic: number;
    military: number;
    political: number;
  };
  autoSimulate?: boolean;
  simulationSpeed?: number;  // Steps per second
  onTransition?: (event: TransitionEvent) => void;
}

export interface PhaseTransitionState {
  currentState: PhaseState;
  trajectory: PhaseState[];
  transitions: TransitionEvent[];
  escapeRate: number;
  isSimulating: boolean;
  config: LandauGinzburgConfig;
}

/**
 * Hook for real-time phase transition monitoring and simulation
 *
 * Uses Landau-Ginzburg potential dynamics to model regime stability
 * and predict phase transitions in geopolitical systems.
 */
export function usePhaseTransition(options: UsePhaseTransitionOptions = {}) {
  const {
    initialStability = 0.7,
    initialRegime = 'stable',
    stressFactors = { economic: 0.3, military: 0.2, political: 0.2 },
    autoSimulate = false,
    simulationSpeed = 30,
    onTransition,
  } = options;

  // Compute Landau coefficient from stress factors
  const landauA = stressToLandauCoefficient(
    stressFactors.economic,
    stressFactors.military,
    stressFactors.political
  );

  const configRef = useRef<LandauGinzburgConfig>({
    a: landauA,
    b: 1.0,
    damping: 0.5,
    noise: 0.1,
    dt: 0.01,
  });

  const [state, setState] = useState<PhaseTransitionState>(() => {
    const initial = createPhaseState(initialStability, initialRegime);
    return {
      currentState: initial,
      trajectory: [initial],
      transitions: [],
      escapeRate: kramersEscapeRate(configRef.current),
      isSimulating: autoSimulate,
      config: configRef.current,
    };
  });

  const animationRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);

  // Update config when stress factors change
  useEffect(() => {
    const newA = stressToLandauCoefficient(
      stressFactors.economic,
      stressFactors.military,
      stressFactors.political
    );

    configRef.current = {
      ...configRef.current,
      a: newA,
    };

    setState(prev => ({
      ...prev,
      escapeRate: kramersEscapeRate(configRef.current),
      config: configRef.current,
    }));
  }, [stressFactors.economic, stressFactors.military, stressFactors.political]);

  // Simulation loop
  const simulate = useCallback((timestamp: number) => {
    if (lastTimeRef.current === 0) {
      lastTimeRef.current = timestamp;
    }

    const elapsed = timestamp - lastTimeRef.current;
    const stepsPerFrame = Math.floor((elapsed / 1000) * simulationSpeed);

    if (stepsPerFrame > 0) {
      lastTimeRef.current = timestamp;

      setState(prev => {
        let currentState = prev.currentState;
        const newTrajectory = [...prev.trajectory];
        const newTransitions = [...prev.transitions];

        for (let i = 0; i < stepsPerFrame; i++) {
          const prevBasin = currentState.basinId;
          currentState = langevinStep(currentState, configRef.current);
          newTrajectory.push(currentState);

          // Check for transition
          if (currentState.basinId !== prevBasin) {
            const transitionEvent: TransitionEvent = {
              timestamp: newTrajectory.length,
              fromBasin: prevBasin,
              toBasin: currentState.basinId,
              orderParameter: currentState.orderParameter,
              transitionProbability: 1 - prev.currentState.stability,
            };
            newTransitions.push(transitionEvent);
            onTransition?.(transitionEvent);
          }
        }

        // Keep trajectory bounded
        const maxTrajectoryLength = 1000;
        const trimmedTrajectory = newTrajectory.length > maxTrajectoryLength
          ? newTrajectory.slice(-maxTrajectoryLength)
          : newTrajectory;

        return {
          ...prev,
          currentState,
          trajectory: trimmedTrajectory,
          transitions: newTransitions,
        };
      });
    }

    if (state.isSimulating) {
      animationRef.current = requestAnimationFrame(simulate);
    }
  }, [simulationSpeed, onTransition, state.isSimulating]);

  // Start/stop simulation
  useEffect(() => {
    if (state.isSimulating) {
      lastTimeRef.current = 0;
      animationRef.current = requestAnimationFrame(simulate);
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [state.isSimulating, simulate]);

  // Control functions
  const startSimulation = useCallback(() => {
    setState(prev => ({ ...prev, isSimulating: true }));
  }, []);

  const stopSimulation = useCallback(() => {
    setState(prev => ({ ...prev, isSimulating: false }));
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  }, []);

  const reset = useCallback(() => {
    const initial = createPhaseState(initialStability, initialRegime);
    setState({
      currentState: initial,
      trajectory: [initial],
      transitions: [],
      escapeRate: kramersEscapeRate(configRef.current),
      isSimulating: false,
      config: configRef.current,
    });
  }, [initialStability, initialRegime]);

  const step = useCallback(() => {
    setState(prev => {
      const newState = langevinStep(prev.currentState, configRef.current);
      const newTransitions = [...prev.transitions];

      if (newState.basinId !== prev.currentState.basinId) {
        const transitionEvent: TransitionEvent = {
          timestamp: prev.trajectory.length,
          fromBasin: prev.currentState.basinId,
          toBasin: newState.basinId,
          orderParameter: newState.orderParameter,
          transitionProbability: 1 - prev.currentState.stability,
        };
        newTransitions.push(transitionEvent);
        onTransition?.(transitionEvent);
      }

      return {
        ...prev,
        currentState: newState,
        trajectory: [...prev.trajectory, newState],
        transitions: newTransitions,
      };
    });
  }, [onTransition]);

  const setStress = useCallback((economic: number, military: number, political: number) => {
    const newA = stressToLandauCoefficient(economic, military, political);
    configRef.current = {
      ...configRef.current,
      a: newA,
    };
    setState(prev => ({
      ...prev,
      escapeRate: kramersEscapeRate(configRef.current),
      config: configRef.current,
    }));
  }, []);

  return {
    ...state,
    startSimulation,
    stopSimulation,
    reset,
    step,
    setStress,
  };
}
