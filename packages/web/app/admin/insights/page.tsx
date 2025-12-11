'use client';

import { Brain, Sparkles, FlaskConical, Code2, FileCheck } from 'lucide-react';

// Placeholder for Elle's Research Insights Dashboard
// TODO: Install shadcn/ui components (card, badge, button, tabs, select, dialog, textarea)
// then restore the full implementation

const STAGE_CONFIG = {
  latent_archaeology: {
    icon: Brain,
    label: 'Archaeology',
    color: 'bg-purple-500',
    description: 'Finding the gradient of ignorance',
  },
  novel_synthesis: {
    icon: Sparkles,
    label: 'Synthesis',
    color: 'bg-blue-500',
    description: 'Creating the fusion',
  },
  theoretical_validation: {
    icon: FlaskConical,
    label: 'Validation',
    color: 'bg-green-500',
    description: 'Proving with math',
  },
  xyza_operationalization: {
    icon: Code2,
    label: 'XYZA',
    color: 'bg-orange-500',
    description: 'Writing the code',
  },
  output_generation: {
    icon: FileCheck,
    label: 'Output',
    color: 'bg-red-500',
    description: 'Final dossier',
  },
};

export default function InsightsDashboard() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">
          Elle&apos;s Research Insights
        </h1>
        <p className="text-gray-400">
          Autonomous research capture via PROMETHEUS protocol
        </p>
      </div>

      {/* Pipeline Visualization */}
      <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6 mb-8">
        <h2 className="text-lg font-semibold mb-4">PROMETHEUS Pipeline</h2>
        <div className="flex items-center justify-between flex-wrap gap-4">
          {Object.entries(STAGE_CONFIG).map(([stage, config], index) => {
            const Icon = config.icon;
            return (
              <div key={stage} className="flex items-center">
                <div className="flex flex-col items-center">
                  <div className={`${config.color} p-3 rounded-full mb-2`}>
                    <Icon className="h-6 w-6 text-white" />
                  </div>
                  <span className="text-sm font-medium">{config.label}</span>
                  <span className="text-xs text-gray-400">{config.description}</span>
                </div>
                {index < Object.keys(STAGE_CONFIG).length - 1 && (
                  <div className="w-8 md:w-16 h-0.5 bg-gray-600 mx-2 hidden md:block" />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Coming Soon Notice */}
      <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-8 text-center">
        <Brain className="h-16 w-16 text-purple-400 mx-auto mb-4" />
        <h3 className="text-xl font-semibold mb-2">Insights Dashboard Coming Soon</h3>
        <p className="text-gray-400 max-w-md mx-auto">
          Elle&apos;s autonomous research insights will be displayed here once the
          PROMETHEUS protocol integration is complete. Check back soon for real-time
          research capture and validation.
        </p>
        <div className="mt-6 flex justify-center gap-4">
          <a
            href="/admin"
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
          >
            Back to Admin
          </a>
          <a
            href="/dashboard/prometheus"
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
          >
            View PROMETHEUS Dashboard
          </a>
        </div>
      </div>
    </div>
  );
}
