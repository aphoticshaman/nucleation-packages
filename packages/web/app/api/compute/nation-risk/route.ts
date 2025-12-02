import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

export const runtime = 'edge';

/**
 * Compute and update nation risk scores from ingested signals
 * Runs on schedule to keep risk values current
 */
export async function GET(req: Request) {
  const startTime = Date.now();

  // Verify cron secret
  const authHeader = req.headers.get('authorization');
  const cronSecret = process.env.CRON_SECRET;

  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );

  const results = {
    timestamp: new Date().toISOString(),
    nations_updated: 0,
    signals_processed: 0,
    errors: [] as string[],
  };

  try {
    // 1. Get latest GDELT tone signals by country (last 48 hours)
    const { data: gdeltSignals } = await supabase
      .from('learning_events')
      .select('data, timestamp')
      .eq('type', 'signal_observation')
      .eq('session_hash', 'gdelt_ingest')
      .neq('domain', 'global')
      .gte('timestamp', new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString())
      .order('timestamp', { ascending: false });

    // Aggregate GDELT tone by country
    const gdeltToneByCountry: Record<string, number[]> = {};
    for (const signal of gdeltSignals || []) {
      const data = signal.data as { numeric_features?: { gdelt_tone_risk?: number }; categorical_features?: { country_code?: string } };
      const country = data.categorical_features?.country_code;
      const toneRisk = data.numeric_features?.gdelt_tone_risk;
      if (country && toneRisk !== undefined) {
        if (!gdeltToneByCountry[country]) gdeltToneByCountry[country] = [];
        gdeltToneByCountry[country].push(toneRisk);
      }
    }

    // 2. Get latest economic indicators from country_signals
    const { data: econSignals } = await supabase
      .from('country_signals')
      .select('country_code, indicator, value, year')
      .order('year', { ascending: false });

    // Aggregate economic risk factors
    const econRiskByCountry: Record<string, {
      inflation?: number;
      unemployment?: number;
      debt_to_gdp?: number;
      gdp_growth?: number;
      current_account?: number;
    }> = {};

    for (const signal of econSignals || []) {
      if (!econRiskByCountry[signal.country_code]) {
        econRiskByCountry[signal.country_code] = {};
      }
      // Take most recent value for each indicator
      if (!econRiskByCountry[signal.country_code][signal.indicator as keyof typeof econRiskByCountry[string]]) {
        econRiskByCountry[signal.country_code][signal.indicator as keyof typeof econRiskByCountry[string]] = signal.value;
      }
    }

    // 3. Get current nation states
    const { data: nations } = await supabase
      .from('nations')
      .select('code, name, basin_strength, transition_risk, velocity');

    if (!nations) {
      return NextResponse.json({ error: 'Failed to fetch nations' }, { status: 500 });
    }

    // 4. Compute updated risk for each nation
    const updates: Array<{
      code: string;
      basin_strength: number;
      transition_risk: number;
      velocity: number[];
    }> = [];

    for (const nation of nations) {
      const gdeltTones = gdeltToneByCountry[nation.code] || [];
      const econ = econRiskByCountry[nation.code] || {};

      // Compute GDELT-based risk (0-1, higher = more negative news)
      let gdeltRisk = 0.5; // Default if no data
      if (gdeltTones.length > 0) {
        gdeltRisk = gdeltTones.reduce((a, b) => a + b, 0) / gdeltTones.length;
        results.signals_processed += gdeltTones.length;
      }

      // Compute economic risk factors
      let econRisk = 0.3; // Default
      let econFactors = 0;

      // High inflation = higher risk
      if (econ.inflation !== undefined) {
        const inflationRisk = Math.min(1, Math.max(0, econ.inflation / 20)); // 20% inflation = max risk
        econRisk += inflationRisk * 0.2;
        econFactors++;
      }

      // High unemployment = higher risk
      if (econ.unemployment !== undefined) {
        const unemploymentRisk = Math.min(1, Math.max(0, econ.unemployment / 25)); // 25% = max risk
        econRisk += unemploymentRisk * 0.15;
        econFactors++;
      }

      // High debt-to-GDP = higher risk
      if (econ.debt_to_gdp !== undefined) {
        const debtRisk = Math.min(1, Math.max(0, (econ.debt_to_gdp - 60) / 100)); // >160% = max risk
        econRisk += debtRisk * 0.15;
        econFactors++;
      }

      // Negative GDP growth = higher risk
      if (econ.gdp_growth !== undefined) {
        const growthRisk = Math.min(1, Math.max(0, (2 - econ.gdp_growth) / 10)); // -8% = max risk
        econRisk += growthRisk * 0.15;
        econFactors++;
      }

      // Large current account deficit = higher risk
      if (econ.current_account !== undefined) {
        const caRisk = Math.min(1, Math.max(0, (-econ.current_account - 2) / 10)); // -12% = max risk
        econRisk += caRisk * 0.1;
        econFactors++;
      }

      if (econFactors > 0) {
        econRisk = Math.min(1, econRisk);
        results.signals_processed += econFactors;
      }

      // Blend signals with current values (momentum/smoothing)
      // 30% new signals, 70% existing (prevents wild swings)
      const blendFactor = 0.3;

      // Hard-coded adjustments for current events (December 2024)
      // These should eventually come from real-time signals
      const currentEventAdjustments: Record<string, { riskDelta: number; reason: string }> = {
        // US and allies - political stress
        'USA': { riskDelta: 0.18, reason: 'Political transition, institutional stress, border crisis' },
        'FRA': { riskDelta: 0.10, reason: 'Political instability, government collapse' },
        'KOR': { riskDelta: 0.14, reason: 'Political crisis, martial law aftermath' },
        'GEO': { riskDelta: 0.12, reason: 'EU protests, political tensions' },
        'ROU': { riskDelta: 0.08, reason: 'Election controversies' },

        // Active conflicts
        'UKR': { riskDelta: 0.22, reason: 'Active war, territorial loss' },
        'ISR': { riskDelta: 0.20, reason: 'Multi-front conflict, Gaza, Lebanon, Yemen' },
        'SYR': { riskDelta: 0.18, reason: 'Regime collapse, power vacuum, Turkish/Kurdish tensions' },
        'LBN': { riskDelta: 0.15, reason: 'Hezbollah weakened, political vacuum, Israeli incursions' },
        'YEM': { riskDelta: 0.08, reason: 'Houthi attacks on shipping, US strikes' },
        'MMR': { riskDelta: 0.08, reason: 'Ongoing civil war, junta losing ground' },

        // Narco-terror corridor - US tensions
        'MEX': { riskDelta: 0.18, reason: 'Cartel violence, US military threats, fentanyl crisis' },
        'VEN': { riskDelta: 0.15, reason: 'TdA gang export, US sanctions, Maduro standoff' },
        'COL': { riskDelta: 0.10, reason: 'Cartel resurgence, coca production up, US pressure' },
        'ECU': { riskDelta: 0.14, reason: 'Cartel takeover, state of emergency, port violence' },
        'HTI': { riskDelta: 0.12, reason: 'Gang state, humanitarian collapse, US evacuation' },
        'HND': { riskDelta: 0.10, reason: 'Drug corridor, gang violence, migration pressure' },
        'GTM': { riskDelta: 0.08, reason: 'Cartel presence, migration hub, political instability' },
        'SLV': { riskDelta: 0.05, reason: 'Bukele crackdown working but authoritarian drift' },
        'NIC': { riskDelta: 0.08, reason: 'Ortega repression, US sanctions' },
        'PAN': { riskDelta: 0.06, reason: 'Darien Gap migration, cartel transit' },

        // Volatile regimes
        'ARG': { riskDelta: 0.12, reason: 'Milei shock therapy, hyperinflation legacy' },
        'BGD': { riskDelta: 0.15, reason: 'Post-uprising political transition' },
        'PAK': { riskDelta: 0.10, reason: 'Political instability, IMF dependency, terror' },
        'IRN': { riskDelta: 0.12, reason: 'Proxy losses, internal unrest, nuclear tensions' },
        'TUR': { riskDelta: 0.08, reason: 'Syria intervention, Kurdish issue, inflation' },

        // State collapse / fragile
        'SDN': { riskDelta: 0.05, reason: 'Civil war ongoing' },
        'SSD': { riskDelta: 0.03, reason: 'Fragile peace' },
        'CAF': { riskDelta: 0.04, reason: 'Wagner presence, low-level conflict' },
        'COD': { riskDelta: 0.06, reason: 'M23 offensive, mineral conflict' },
        'SOM': { riskDelta: 0.04, reason: 'Al-Shabaab, Ethiopian tensions' },
        'LBY': { riskDelta: 0.05, reason: 'Divided government, oil conflicts' },
        'AFG': { riskDelta: 0.03, reason: 'Taliban stable but repressive, ISIS-K threat' },
      };

      const eventAdj = currentEventAdjustments[nation.code];
      const eventRiskDelta = eventAdj?.riskDelta || 0;

      // New risk = blend of GDELT (40%), econ (30%), current events (30%)
      const signalRisk = gdeltRisk * 0.4 + econRisk * 0.3 + eventRiskDelta * 0.3 / 0.2; // Normalize event delta
      const newTransitionRisk = Math.min(0.98, Math.max(0.02,
        nation.transition_risk * (1 - blendFactor) + signalRisk * blendFactor + eventRiskDelta
      ));

      // Basin strength inversely related to risk (with lag)
      const newBasinStrength = Math.min(0.98, Math.max(0.02,
        nation.basin_strength * (1 - blendFactor * 0.5) + (1 - newTransitionRisk) * blendFactor * 0.5
      ));

      // Compute velocity (rate of change in phase space)
      const oldVelocity = nation.velocity || [0, 0, 0, 0];
      const riskVelocity = newTransitionRisk - nation.transition_risk;
      const basinVelocity = newBasinStrength - nation.basin_strength;
      const newVelocity = [
        oldVelocity[0] * 0.7 + riskVelocity * 0.3,
        oldVelocity[1] * 0.7 + basinVelocity * 0.3,
        oldVelocity[2] * 0.9, // Decay other dimensions
        oldVelocity[3] * 0.9,
      ];

      // Only update if meaningful change
      if (
        Math.abs(newTransitionRisk - nation.transition_risk) > 0.005 ||
        Math.abs(newBasinStrength - nation.basin_strength) > 0.005
      ) {
        updates.push({
          code: nation.code,
          basin_strength: Math.round(newBasinStrength * 1000) / 1000,
          transition_risk: Math.round(newTransitionRisk * 1000) / 1000,
          velocity: newVelocity.map(v => Math.round(v * 10000) / 10000),
        });
      }
    }

    // 5. Apply updates
    for (const update of updates) {
      const { error: updateError } = await supabase
        .from('nations')
        .update({
          basin_strength: update.basin_strength,
          transition_risk: update.transition_risk,
          velocity: update.velocity,
          updated_at: new Date().toISOString(),
        })
        .eq('code', update.code);

      if (updateError) {
        results.errors.push(`${update.code}: ${updateError.message}`);
      } else {
        results.nations_updated++;
      }
    }

    // 6. Record snapshot for trend analysis
    if (updates.length > 0) {
      const { error: historyError } = await supabase.rpc('record_nation_snapshot');
      if (historyError) {
        results.errors.push(`History: ${historyError.message}`);
      }
    }

    return NextResponse.json({
      ...results,
      latency_ms: Date.now() - startTime,
      updates_applied: updates.map(u => ({ code: u.code, risk: u.transition_risk })),
    });
  } catch (error) {
    console.error('Risk computation error:', error);
    return NextResponse.json(
      {
        error: 'Computation failed',
        details: error instanceof Error ? error.message : 'unknown',
        latency_ms: Date.now() - startTime,
      },
      { status: 500 }
    );
  }
}
