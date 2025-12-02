/**
 * Export training data from learning_events for fine-tuning
 * Run with: npx ts-node scripts/export-training-data.ts
 */

import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

interface TrainingExample {
  instruction: string;
  input: string;
  output: string;
  metadata?: Record<string, unknown>;
}

async function exportTrainingData() {
  console.log('Fetching learning events...');

  // Get LLM interactions (reasoning traces)
  const { data: llmEvents, error: llmError } = await supabase
    .from('learning_events')
    .select('*')
    .eq('type', 'llm_interaction')
    .order('timestamp', { ascending: false })
    .limit(50000);

  if (llmError) {
    console.error('Error fetching LLM events:', llmError);
    return;
  }

  // Get signal observations for context
  const { data: signalEvents } = await supabase
    .from('learning_events')
    .select('*')
    .eq('type', 'signal_observation')
    .order('timestamp', { ascending: false })
    .limit(10000);

  console.log(`Found ${llmEvents?.length || 0} LLM interactions`);
  console.log(`Found ${signalEvents?.length || 0} signal observations`);

  const trainingExamples: TrainingExample[] = [];

  // Convert LLM interactions to training format
  for (const event of llmEvents || []) {
    const data = event.data as {
      categorical_features?: {
        preset?: string;
        model?: string;
      };
      text_features?: {
        system_prompt?: string;
        user_prompt?: string;
        response?: string;
      };
      numeric_features?: Record<string, number>;
    };

    if (data.text_features?.user_prompt && data.text_features?.response) {
      trainingExamples.push({
        instruction: `You are a geopolitical intelligence analyst. Analyze the following situation and provide a risk assessment.`,
        input: data.text_features.user_prompt,
        output: data.text_features.response,
        metadata: {
          preset: data.categorical_features?.preset,
          timestamp: event.timestamp,
          domain: event.domain,
        },
      });
    }
  }

  // Create causal reasoning examples from signal patterns
  const signalsByDomain: Record<string, typeof signalEvents> = {};
  for (const event of signalEvents || []) {
    if (!signalsByDomain[event.domain]) {
      signalsByDomain[event.domain] = [];
    }
    signalsByDomain[event.domain]!.push(event);
  }

  // Generate synthetic causal reasoning examples
  for (const [domain, events] of Object.entries(signalsByDomain)) {
    if (!events || events.length < 5) continue;

    const recentEvents = events.slice(0, 5);
    const signals = recentEvents.map(e => {
      const data = e.data as { numeric_features?: Record<string, number> };
      return data.numeric_features || {};
    });

    // Create a reasoning trace example
    trainingExamples.push({
      instruction: 'Analyze the following signal data and identify causal patterns.',
      input: JSON.stringify({ domain, signals }, null, 2),
      output: `Domain: ${domain}\n\nSignal Analysis:\n- ${Object.keys(signals[0] || {}).slice(0, 3).join(', ')} show temporal correlation\n- Recommend monitoring for phase transition indicators\n- Confidence: Medium (based on ${events.length} observations)`,
      metadata: { synthetic: true, domain },
    });
  }

  console.log(`Generated ${trainingExamples.length} training examples`);

  // Export in different formats

  // 1. Alpaca format (for most fine-tuning)
  const alpacaFormat = trainingExamples.map(ex => ({
    instruction: ex.instruction,
    input: ex.input,
    output: ex.output,
  }));
  fs.writeFileSync('training_data_alpaca.json', JSON.stringify(alpacaFormat, null, 2));
  console.log('Wrote training_data_alpaca.json');

  // 2. ChatML format (for chat models)
  const chatmlFormat = trainingExamples.map(ex => ({
    messages: [
      { role: 'system', content: ex.instruction },
      { role: 'user', content: ex.input },
      { role: 'assistant', content: ex.output },
    ],
  }));
  fs.writeFileSync('training_data_chatml.json', JSON.stringify(chatmlFormat, null, 2));
  console.log('Wrote training_data_chatml.json');

  // 3. Simple prompt-completion (for basic fine-tuning)
  const simpleFormat = trainingExamples.map(ex => ({
    prompt: `${ex.instruction}\n\n${ex.input}`,
    completion: ex.output,
  }));
  fs.writeFileSync('training_data_simple.json', JSON.stringify(simpleFormat, null, 2));
  console.log('Wrote training_data_simple.json');

  // Stats
  console.log('\n=== Training Data Stats ===');
  console.log(`Total examples: ${trainingExamples.length}`);
  console.log(`From LLM interactions: ${llmEvents?.length || 0}`);
  console.log(`Synthetic from signals: ${trainingExamples.length - (llmEvents?.length || 0)}`);
  console.log(`\nReady for fine-tuning when you have 1000+ examples.`);
  console.log(`Current count: ${trainingExamples.length} (need ${Math.max(0, 1000 - trainingExamples.length)} more)`);
}

exportTrainingData().catch(console.error);
