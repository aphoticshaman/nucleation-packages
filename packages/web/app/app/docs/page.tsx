'use client';

import { useState } from 'react';
import {
  Book,
  Code,
  Copy,
  Check,
  ChevronRight,
  Terminal,
  Globe,
  Key,
  Webhook,
  Zap,
  Shield,
  ExternalLink,
} from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';

const API_ENDPOINTS = [
  {
    method: 'GET',
    path: '/api/nations',
    description: 'List all nations with their current risk scores',
    auth: true,
    params: [
      { name: 'region', type: 'string', required: false, description: 'Filter by region (e.g., "europe", "asia")' },
      { name: 'minRisk', type: 'number', required: false, description: 'Minimum risk score (0-100)' },
      { name: 'limit', type: 'number', required: false, description: 'Max results (default: 50)' },
    ],
    response: `{
  "nations": [
    {
      "iso3": "USA",
      "name": "United States",
      "risk": 32,
      "regime": "stable",
      "lastUpdated": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 195
}`,
  },
  {
    method: 'GET',
    path: '/api/nations/:iso3',
    description: 'Get detailed risk data for a specific nation',
    auth: true,
    params: [
      { name: 'iso3', type: 'string', required: true, description: 'ISO 3166-1 alpha-3 country code' },
    ],
    response: `{
  "iso3": "USA",
  "name": "United States",
  "risk": 32,
  "regime": "stable",
  "factors": {
    "political": 28,
    "economic": 35,
    "social": 30,
    "security": 25
  },
  "trend": "stable",
  "lastUpdated": "2024-01-15T10:30:00Z"
}`,
  },
  {
    method: 'POST',
    path: '/api/cascade',
    description: 'Simulate a cascade event across nations',
    auth: true,
    params: [
      { name: 'origin', type: 'string', required: true, description: 'Origin nation ISO3 code' },
      { name: 'eventType', type: 'string', required: true, description: 'Type: "political", "economic", "conflict"' },
      { name: 'magnitude', type: 'number', required: true, description: 'Event magnitude (1-10)' },
    ],
    response: `{
  "id": "sim_abc123",
  "origin": "RUS",
  "affected": [
    { "iso3": "UKR", "impact": 85, "delay": 0 },
    { "iso3": "POL", "impact": 45, "delay": 24 },
    { "iso3": "DEU", "impact": 30, "delay": 48 }
  ],
  "totalImpact": 160,
  "peakTime": 72
}`,
  },
  {
    method: 'GET',
    path: '/api/alerts',
    description: 'Get active alerts and notifications',
    auth: true,
    params: [
      { name: 'severity', type: 'string', required: false, description: 'Filter: "low", "moderate", "high", "critical"' },
      { name: 'since', type: 'string', required: false, description: 'ISO timestamp for alerts after this time' },
    ],
    response: `{
  "alerts": [
    {
      "id": "alert_xyz",
      "severity": "high",
      "title": "Elevated tension in region",
      "summary": "Risk indicators show...",
      "timestamp": "2024-01-15T08:00:00Z",
      "nations": ["UKR", "RUS"]
    }
  ]
}`,
  },
  {
    method: 'POST',
    path: '/api/webhooks',
    description: 'Register a webhook for real-time alerts',
    auth: true,
    params: [
      { name: 'url', type: 'string', required: true, description: 'Your webhook endpoint URL' },
      { name: 'events', type: 'array', required: true, description: 'Events to subscribe to' },
      { name: 'secret', type: 'string', required: false, description: 'Signing secret for verification' },
    ],
    response: `{
  "id": "wh_abc123",
  "url": "https://your-api.com/webhook",
  "events": ["alert.created", "risk.changed"],
  "active": true,
  "createdAt": "2024-01-15T10:00:00Z"
}`,
  },
];

const CODE_EXAMPLES = {
  curl: `curl -X GET "https://api.latticeforge.com/api/nations?region=europe&limit=10" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json"`,
  javascript: `const response = await fetch('https://api.latticeforge.com/api/nations', {
  method: 'GET',
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json',
  },
  params: {
    region: 'europe',
    limit: 10,
  },
});

const data = await response.json();
console.log(data.nations);`,
  python: `import requests

response = requests.get(
    'https://api.latticeforge.com/api/nations',
    headers={
        'Authorization': 'Bearer YOUR_API_KEY',
        'Content-Type': 'application/json',
    },
    params={
        'region': 'europe',
        'limit': 10,
    }
)

data = response.json()
print(data['nations'])`,
};

const WEBHOOK_EVENTS = [
  { event: 'alert.created', description: 'New alert generated' },
  { event: 'alert.resolved', description: 'Alert resolved or expired' },
  { event: 'risk.changed', description: 'Nation risk score changed significantly' },
  { event: 'regime.shifted', description: 'Nation regime classification changed' },
  { event: 'cascade.detected', description: 'Cascade event detected' },
  { event: 'briefing.ready', description: 'New intel briefing available' },
];

export default function APIDocsPage() {
  const [selectedLang, setSelectedLang] = useState<'curl' | 'javascript' | 'python'>('curl');
  const [copiedEndpoint, setCopiedEndpoint] = useState<string | null>(null);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedEndpoint(id);
    setTimeout(() => setCopiedEndpoint(null), 2000);
  };

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center">
            <Book className="w-5 h-5 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-white">API Documentation</h1>
        </div>
        <p className="text-slate-400">
          Integrate LatticeForge intelligence into your applications
        </p>
      </div>

      {/* Quick Start */}
      <GlassCard blur="heavy">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5 text-yellow-400" />
          Quick Start
        </h2>
        <div className="space-y-4">
          <div className="flex items-start gap-3">
            <div className="w-6 h-6 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400 text-sm font-medium shrink-0">
              1
            </div>
            <div>
              <p className="text-white font-medium">Get your API key</p>
              <p className="text-sm text-slate-400">
                Go to <a href="/app/settings" className="text-blue-400 hover:underline">Settings</a> and generate an API key
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-6 h-6 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400 text-sm font-medium shrink-0">
              2
            </div>
            <div>
              <p className="text-white font-medium">Make your first request</p>
              <p className="text-sm text-slate-400">
                Use the examples below to query the API
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-6 h-6 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400 text-sm font-medium shrink-0">
              3
            </div>
            <div>
              <p className="text-white font-medium">Set up webhooks (optional)</p>
              <p className="text-sm text-slate-400">
                Receive real-time alerts at your endpoint
              </p>
            </div>
          </div>
        </div>
      </GlassCard>

      {/* Authentication */}
      <GlassCard blur="heavy">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Key className="w-5 h-5 text-amber-400" />
          Authentication
        </h2>
        <p className="text-slate-400 mb-4">
          All API requests require a Bearer token in the Authorization header:
        </p>
        <div className="bg-black/30 rounded-xl p-4 font-mono text-sm">
          <span className="text-slate-500">Authorization:</span>{' '}
          <span className="text-green-400">Bearer</span>{' '}
          <span className="text-blue-400">YOUR_API_KEY</span>
        </div>
        <div className="mt-4 p-3 bg-amber-500/10 border border-amber-500/20 rounded-xl">
          <p className="text-amber-300 text-sm flex items-start gap-2">
            <Shield className="w-4 h-4 shrink-0 mt-0.5" />
            Keep your API key secure. Never expose it in client-side code.
          </p>
        </div>
      </GlassCard>

      {/* Code Examples */}
      <GlassCard blur="heavy">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Code className="w-5 h-5 text-cyan-400" />
          Code Examples
        </h2>
        <div className="flex gap-2 mb-4">
          {(['curl', 'javascript', 'python'] as const).map((lang) => (
            <button
              key={lang}
              onClick={() => setSelectedLang(lang)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                selectedLang === lang
                  ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {lang === 'curl' ? 'cURL' : lang.charAt(0).toUpperCase() + lang.slice(1)}
            </button>
          ))}
        </div>
        <div className="relative">
          <pre className="bg-black/30 rounded-xl p-4 overflow-x-auto text-sm">
            <code className="text-slate-300">{CODE_EXAMPLES[selectedLang]}</code>
          </pre>
          <button
            onClick={() => copyToClipboard(CODE_EXAMPLES[selectedLang], 'example')}
            className="absolute top-3 right-3 p-2 text-slate-400 hover:text-white transition-colors"
          >
            {copiedEndpoint === 'example' ? (
              <Check className="w-4 h-4 text-green-400" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </button>
        </div>
      </GlassCard>

      {/* Endpoints */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Globe className="w-5 h-5 text-purple-400" />
          API Endpoints
        </h2>
        <div className="space-y-4">
          {API_ENDPOINTS.map((endpoint) => (
            <EndpointCard
              key={endpoint.path}
              endpoint={endpoint}
              copied={copiedEndpoint === endpoint.path}
              onCopy={() => copyToClipboard(endpoint.response, endpoint.path)}
            />
          ))}
        </div>
      </div>

      {/* Webhook Events */}
      <GlassCard blur="heavy">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Webhook className="w-5 h-5 text-orange-400" />
          Webhook Events
        </h2>
        <p className="text-slate-400 mb-4">
          Subscribe to these events to receive real-time notifications:
        </p>
        <div className="space-y-2">
          {WEBHOOK_EVENTS.map((evt) => (
            <div
              key={evt.event}
              className="flex items-center justify-between p-3 bg-black/20 rounded-lg"
            >
              <code className="text-blue-400 text-sm">{evt.event}</code>
              <span className="text-slate-400 text-sm">{evt.description}</span>
            </div>
          ))}
        </div>
      </GlassCard>

      {/* Rate Limits */}
      <GlassCard blur="heavy">
        <h2 className="text-lg font-semibold text-white mb-4">Rate Limits</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-500 border-b border-white/[0.06]">
                <th className="pb-3 font-medium">Plan</th>
                <th className="pb-3 font-medium">Requests/min</th>
                <th className="pb-3 font-medium">Requests/day</th>
              </tr>
            </thead>
            <tbody className="text-slate-300">
              <tr className="border-b border-white/[0.04]">
                <td className="py-3">Free</td>
                <td className="py-3">10</td>
                <td className="py-3">100</td>
              </tr>
              <tr className="border-b border-white/[0.04]">
                <td className="py-3">Starter</td>
                <td className="py-3">60</td>
                <td className="py-3">1,000</td>
              </tr>
              <tr className="border-b border-white/[0.04]">
                <td className="py-3">Pro</td>
                <td className="py-3">300</td>
                <td className="py-3">10,000</td>
              </tr>
              <tr>
                <td className="py-3">Enterprise</td>
                <td className="py-3">Custom</td>
                <td className="py-3">Unlimited</td>
              </tr>
            </tbody>
          </table>
        </div>
      </GlassCard>

      {/* SDKs */}
      <GlassCard blur="light" className="text-center">
        <p className="text-slate-400 mb-4">Need an SDK?</p>
        <div className="flex justify-center gap-4">
          <a
            href="https://github.com/latticeforge/sdk-js"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 font-medium"
          >
            JavaScript SDK
            <ExternalLink className="w-4 h-4" />
          </a>
          <a
            href="https://github.com/latticeforge/sdk-python"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 font-medium"
          >
            Python SDK
            <ExternalLink className="w-4 h-4" />
          </a>
        </div>
      </GlassCard>
    </div>
  );
}

function EndpointCard({
  endpoint,
  copied,
  onCopy,
}: {
  endpoint: (typeof API_ENDPOINTS)[0];
  copied: boolean;
  onCopy: () => void;
}) {
  const [expanded, setExpanded] = useState(false);

  const methodColors: Record<string, string> = {
    GET: 'bg-green-500/20 text-green-400',
    POST: 'bg-blue-500/20 text-blue-400',
    PUT: 'bg-amber-500/20 text-amber-400',
    DELETE: 'bg-red-500/20 text-red-400',
  };

  return (
    <GlassCard blur="heavy">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-4"
      >
        <span className={`px-2 py-1 rounded text-xs font-mono font-medium ${methodColors[endpoint.method]}`}>
          {endpoint.method}
        </span>
        <code className="text-white font-mono text-sm">{endpoint.path}</code>
        <span className="text-slate-400 text-sm flex-1 text-left">{endpoint.description}</span>
        <ChevronRight
          className={`w-4 h-4 text-slate-500 transition-transform ${expanded ? 'rotate-90' : ''}`}
        />
      </button>

      {expanded && (
        <div className="mt-4 pt-4 border-t border-white/[0.06] space-y-4">
          {/* Parameters */}
          <div>
            <h4 className="text-sm font-medium text-slate-400 mb-2">Parameters</h4>
            <div className="space-y-2">
              {endpoint.params.map((param) => (
                <div
                  key={param.name}
                  className="flex items-start gap-3 p-2 bg-black/20 rounded-lg text-sm"
                >
                  <code className="text-blue-400">{param.name}</code>
                  <span className="text-slate-500">{param.type}</span>
                  {param.required && (
                    <span className="text-red-400 text-xs">required</span>
                  )}
                  <span className="text-slate-400 flex-1">{param.description}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Response */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-slate-400">Response</h4>
              <button
                onClick={onCopy}
                className="p-1 text-slate-400 hover:text-white transition-colors"
              >
                {copied ? (
                  <Check className="w-4 h-4 text-green-400" />
                ) : (
                  <Copy className="w-4 h-4" />
                )}
              </button>
            </div>
            <pre className="bg-black/30 rounded-xl p-4 overflow-x-auto text-sm">
              <code className="text-slate-300">{endpoint.response}</code>
            </pre>
          </div>
        </div>
      )}
    </GlassCard>
  );
}
