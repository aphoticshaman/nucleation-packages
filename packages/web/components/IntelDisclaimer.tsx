'use client';

import { useState } from 'react';
import { ChevronDown, ChevronUp, AlertCircle } from 'lucide-react';

interface IntelDisclaimerProps {
  compact?: boolean;
}

export function IntelDisclaimer({ compact = false }: IntelDisclaimerProps) {
  const [expanded, setExpanded] = useState(false);

  if (compact) {
    return (
      <div className="text-[10px] text-slate-600 text-center py-2 border-t border-white/[0.04] mt-6">
        AI-generated analysis • Not financial/legal/medical advice • Independent research •
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-slate-500 hover:text-slate-400 ml-1 underline"
        >
          Full disclaimer
        </button>
        {expanded && <DisclaimerModal onClose={() => setExpanded(false)} />}
      </div>
    );
  }

  return (
    <div className="mt-8 border-t border-white/[0.06] pt-6">
      {/* Collapsed view */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between text-left text-xs text-slate-500 hover:text-slate-400 transition-colors"
      >
        <div className="flex items-center gap-2">
          <AlertCircle className="w-3 h-3" />
          <span>Important Disclaimers & Terms of Use</span>
        </div>
        {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="mt-4 space-y-4 text-xs">
          {/* Legal version */}
          <div className="p-4 bg-slate-900/50 rounded-lg border border-white/[0.04] text-slate-500 leading-relaxed space-y-3">
            <h4 className="text-slate-400 font-semibold uppercase tracking-wider text-[10px]">
              Legal Disclaimer
            </h4>
            <p>
              <strong>INDEPENDENCE & NON-AFFILIATION:</strong> LatticeForge operates as an independent research and analysis platform. We maintain no formal affiliations, partnerships, consulting arrangements, or employment relationships with any government agencies, intelligence organizations, defense contractors, financial institutions, law firms, medical institutions, academic institutions, or Fortune 500 corporations. Our analysis is produced independently and does not represent the views, positions, or policies of any external entity.
            </p>
            <p>
              <strong>NOT PROFESSIONAL ADVICE:</strong> The content provided on this platform, including all intelligence briefings, market analysis, security assessments, and strategic recommendations, is for informational and educational purposes only. This content does not constitute, and should not be construed as: (a) legal advice or the practice of law; (b) financial, investment, or fiduciary advice; (c) medical, health, or therapeutic guidance; (d) official government intelligence or policy recommendations; (e) defense or national security guidance; (f) professional consulting services of any kind. Users should consult qualified professionals in relevant fields before making decisions based on any content provided herein.
            </p>
            <p>
              <strong>AI-GENERATED CONTENT:</strong> This platform utilizes artificial intelligence systems to synthesize, analyze, and present information. AI systems, by their nature, may: (a) produce outputs that contain errors, inaccuracies, or hallucinations; (b) fail to reflect the most current information; (c) generate content that requires human verification; (d) exhibit biases present in training data. Users acknowledge that AI-generated content is inherently probabilistic and should be treated as one input among many in any decision-making process.
            </p>
            <p>
              <strong>NO WARRANTIES:</strong> All content is provided "AS IS" without warranties of any kind, express or implied, including but not limited to warranties of accuracy, completeness, timeliness, merchantability, or fitness for a particular purpose. We do not warrant that the service will be uninterrupted, secure, or error-free.
            </p>
            <p>
              <strong>LIMITATION OF LIABILITY:</strong> In no event shall LatticeForge, its operators, affiliates, or contributors be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from use of this platform or reliance on its content.
            </p>
            <p>
              <strong>VERIFICATION REQUIRED:</strong> Users are strongly encouraged to verify all information through multiple independent sources, including established fact-checking organizations and primary source documentation, before acting on any analysis or recommendations provided.
            </p>
          </div>

          {/* Plain English version */}
          <div className="p-4 bg-cyan-500/5 rounded-lg border border-cyan-500/10 text-slate-400 leading-relaxed">
            <h4 className="text-cyan-400 font-semibold uppercase tracking-wider text-[10px] mb-2">
              The Plain English Version
            </h4>
            <p>
              Here's the deal: We're an independent platform—not tied to any government, defense company, bank, law firm, or tech giant. We built this because we wanted better intelligence tools, not because anyone's paying us to push an agenda.
            </p>
            <p className="mt-2">
              <strong>What we're NOT:</strong> We're not lawyers, doctors, financial advisors, therapists, defense analysts, or government officials. We don't have security clearances. We're not here to tell you what to do with your money, your health, or your life.
            </p>
            <p className="mt-2">
              <strong>What we ARE:</strong> We're builders who think AI can help people understand what's happening in the world. We pull from open sources, apply analysis, and try to present it in useful ways.
            </p>
            <p className="mt-2">
              <strong>About the AI:</strong> Yes, it's AI. Yes, AI can be wrong—sometimes confidently wrong. It might miss context, misinterpret data, or just make stuff up (we call these "hallucinations" and they're a known issue with all AI systems). Don't take anything here as gospel.
            </p>
            <p className="mt-2">
              <strong>What you should do:</strong> Treat this as one tool in your toolkit. Cross-reference with trusted news sources. Check fact-checking sites like Snopes, PolitiFact, or FactCheck.org. Talk to actual humans—especially professionals if you're making important decisions. Stay skeptical, stay curious, and if something sounds crazy, it might be.
            </p>
            <p className="mt-2 text-slate-500 italic">
              We're here to help you think, not to think for you.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function DisclaimerModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70" onClick={onClose}>
      <div
        className="max-w-2xl max-h-[80vh] overflow-y-auto bg-slate-900 rounded-xl border border-white/[0.06] p-6"
        onClick={e => e.stopPropagation()}
      >
        <h3 className="text-lg font-semibold text-white mb-4">Disclaimer & Terms of Use</h3>
        <IntelDisclaimer />
        <button
          onClick={onClose}
          className="mt-4 px-4 py-2 bg-cyan-500/20 text-cyan-300 rounded-lg hover:bg-cyan-500/30 transition-colors"
        >
          Close
        </button>
      </div>
    </div>
  );
}

export default IntelDisclaimer;
