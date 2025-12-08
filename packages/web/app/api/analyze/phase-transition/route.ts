import { NextRequest, NextResponse } from 'next/server';
import { requireAuth } from '@/lib/auth';
import {
  stressToLandauCoefficient,
  kramersEscapeRate,
  generatePotentialCurve,
  simulateTrajectory,
  detectTransitions,
  createPhaseState,
  type LandauGinzburgConfig,
} from '@/lib/physics/landau-ginzburg';
import { analyzeRegimes } from '@/lib/physics/markov-switching';

interface PhaseTransitionRequest {
  mode: 'analyze' | 'simulate' | 'forecast';
  // For analyze mode
  observations?: number[];
  // For simulate mode
  stressFactors?: {
    economic: number;
    military: number;
    political: number;
  };
  simulationSteps?: number;
  // For forecast mode
  currentRegime?: 'stable' | 'volatile' | 'crisis';
  stability?: number;
  horizons?: number[];
}

/**
 * POST /api/analyze/phase-transition
 *
 * Analyze and forecast regime transitions using Landau-Ginzburg
 * potential dynamics and Markov-switching models.
 */
export async function POST(request: NextRequest) {
  try {
    await requireAuth();

    const body: PhaseTransitionRequest = await request.json();
    const { mode } = body;

    if (mode === 'analyze') {
      // Markov-switching regime analysis
      const { observations } = body;

      if (!observations || observations.length < 50) {
        return NextResponse.json(
          { error: 'Need at least 50 observations for regime analysis' },
          { status: 400 }
        );
      }

      const analysis = analyzeRegimes(observations);

      return NextResponse.json({
        mode: 'analyze',
        currentRegime: analysis.currentRegime,
        regimeName: ['STABLE', 'VOLATILE', 'CRISIS'][analysis.currentRegime],
        transitionRisk: analysis.transitionRisk,
        regimeChanges: analysis.changes,
        expectedDurations: analysis.durations.map((d, i) => ({
          regime: ['STABLE', 'VOLATILE', 'CRISIS'][i],
          expectedDuration: d,
        })),
        stationaryDistribution: analysis.stationary.map((p, i) => ({
          regime: ['STABLE', 'VOLATILE', 'CRISIS'][i],
          probability: p,
        })),
        visualization: analysis.visualization,
      });
    }

    if (mode === 'simulate') {
      // Landau-Ginzburg simulation
      const {
        stressFactors = { economic: 0.3, military: 0.2, political: 0.2 },
        simulationSteps = 500,
      } = body;

      const a = stressToLandauCoefficient(
        stressFactors.economic,
        stressFactors.military,
        stressFactors.political
      );

      const config: LandauGinzburgConfig = {
        a,
        b: 1.0,
        damping: 0.5,
        noise: 0.1,
        dt: 0.01,
      };

      const initialState = createPhaseState(
        1 - Math.max(stressFactors.economic, stressFactors.military, stressFactors.political),
        a > 0.3 ? 'volatile' : 'stable'
      );

      const trajectory = simulateTrajectory(initialState, simulationSteps, config);
      const transitions = detectTransitions(trajectory);
      const escapeRate = kramersEscapeRate(config);
      const potentialCurve = generatePotentialCurve(config);

      // Downsample trajectory for response
      const step = Math.max(1, Math.floor(trajectory.length / 200));
      const sampledTrajectory = trajectory.filter((_, i) => i % step === 0);

      return NextResponse.json({
        mode: 'simulate',
        config: {
          landauA: a,
          stressLevel: (stressFactors.economic + stressFactors.military + stressFactors.political) / 3,
        },
        initialState,
        finalState: trajectory[trajectory.length - 1],
        transitions,
        escapeRate,
        trajectory: sampledTrajectory,
        potentialCurve,
        summary: {
          totalTransitions: transitions.length,
          timeInEachBasin: {
            left: trajectory.filter(s => s.basinId === 'left').length / trajectory.length,
            center: trajectory.filter(s => s.basinId === 'center').length / trajectory.length,
            right: trajectory.filter(s => s.basinId === 'right').length / trajectory.length,
          },
          averageStability: trajectory.reduce((s, t) => s + t.stability, 0) / trajectory.length,
        },
      });
    }

    if (mode === 'forecast') {
      // Combined forecast
      const {
        currentRegime = 'stable',
        stability = 0.7,
        horizons = [1, 5, 10, 30, 90],
      } = body;

      // Landau-Ginzburg escape rate
      const stressLevel = 1 - stability;
      const a = stressLevel * 2 - 1;
      const escapeRate = kramersEscapeRate({ a, b: 1.0, damping: 0.5, noise: 0.1, dt: 0.01 });

      // Markov-switching forecast
      const regimeIdx = currentRegime === 'stable' ? 0 : currentRegime === 'volatile' ? 1 : 2;
      const currentProbs = [0, 0, 0];
      currentProbs[regimeIdx] = 1;

      const transitionMatrix = [
        [0.95, 0.04, 0.01],
        [0.10, 0.80, 0.10],
        [0.05, 0.15, 0.80],
      ];

      // Compute forecasts
      const forecasts = horizons.map(h => {
        let probs = [...currentProbs];
        for (let step = 0; step < h; step++) {
          const newProbs = [0, 0, 0];
          for (let j = 0; j < 3; j++) {
            for (let i = 0; i < 3; i++) {
              newProbs[j] += probs[i] * transitionMatrix[i][j];
            }
          }
          probs = newProbs;
        }
        return {
          horizon: h,
          probabilities: {
            stable: probs[0],
            volatile: probs[1],
            crisis: probs[2],
          },
          mostLikely: probs[0] >= probs[1] && probs[0] >= probs[2]
            ? 'STABLE'
            : probs[1] >= probs[2]
            ? 'VOLATILE'
            : 'CRISIS',
        };
      });

      return NextResponse.json({
        mode: 'forecast',
        currentState: {
          regime: currentRegime.toUpperCase(),
          stability,
          escapeRate,
        },
        forecasts,
        riskAssessment: {
          shortTerm: forecasts[0]?.probabilities.crisis || 0,
          mediumTerm: forecasts[2]?.probabilities.crisis || 0,
          longTerm: forecasts[4]?.probabilities.crisis || 0,
          escalationProbability: 1 - Math.pow(1 - escapeRate, 30),
        },
      });
    }

    return NextResponse.json({ error: 'Invalid mode' }, { status: 400 });
  } catch (error) {
    console.error('Phase transition analysis error:', error);
    return NextResponse.json(
      { error: 'Analysis failed', details: (error as Error).message },
      { status: 500 }
    );
  }
}
