'use client';

import { useState, useEffect } from 'react';
import { useIntelBriefing, getRiskBadgeStyle } from '@/hooks/useIntelBriefing';
import {
  Globe, Shield, TrendingUp, AlertTriangle, RefreshCw, Target, BookOpen,
  Landmark, DollarSign, Factory, Cpu, Clock, ChevronRight,
  FileText, Users, Radio, Zap, X
} from 'lucide-react';
import Glossary from '@/components/Glossary';
import HelpTip from '@/components/HelpTip';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';
import { WarmupScreen } from '@/components/ui/WarmupScreen';
import { supabase } from '@/lib/supabase';

const PRESETS = [
  { id: 'global', name: 'Global Overview', icon: Globe, desc: 'All 195 nations' },
  { id: 'nato', name: 'NATO Alliance', icon: Shield, desc: '32 member states' },
  { id: 'brics', name: 'BRICS+', icon: TrendingUp, desc: 'Emerging powers bloc' },
  { id: 'conflict', name: 'Hot Spots', icon: AlertTriangle, desc: 'Active tension zones' },
];

// Domain categories for organized display
const DOMAIN_CATEGORIES = {
  'Political & Security': {
    icon: Landmark,
    color: 'text-red-400',
    bgColor: 'bg-red-500/10',
    borderColor: 'border-red-500/20',
    domains: ['political', 'security', 'military', 'terrorism', 'borders'],
  },
  'Economic & Markets': {
    icon: DollarSign,
    color: 'text-emerald-400',
    bgColor: 'bg-emerald-500/10',
    borderColor: 'border-emerald-500/20',
    domains: ['economic', 'financial', 'markets', 'employment', 'housing'],
  },
  'Tech & Cyber': {
    icon: Cpu,
    color: 'text-cyan-400',
    bgColor: 'bg-cyan-500/10',
    borderColor: 'border-cyan-500/20',
    domains: ['scitech', 'cyber', 'crypto', 'space', 'emerging'],
  },
  'Resources & Industry': {
    icon: Factory,
    color: 'text-amber-400',
    bgColor: 'bg-amber-500/10',
    borderColor: 'border-amber-500/20',
    domains: ['energy', 'minerals', 'resources', 'industry', 'logistics'],
  },
  'Society & Information': {
    icon: Users,
    color: 'text-purple-400',
    bgColor: 'bg-purple-500/10',
    borderColor: 'border-purple-500/20',
    domains: ['domestic', 'infoops', 'crime', 'health', 'education', 'religious'],
  },
};

const DOMAIN_LABELS: Record<string, string> = {
  political: 'Geopolitics',
  economic: 'Economy',
  security: 'Security',
  financial: 'Finance',
  health: 'Public Health',
  scitech: 'Science & Tech',
  resources: 'Resources',
  crime: 'Crime',
  cyber: 'Cyber',
  terrorism: 'Terrorism',
  domestic: 'Civil Society',
  borders: 'Borders & Migration',
  infoops: 'Information Ops',
  military: 'Defense',
  space: 'Space',
  industry: 'Industry',
  logistics: 'Logistics',
  minerals: 'Critical Minerals',
  energy: 'Energy',
  markets: 'Markets',
  religious: 'Religious Affairs',
  education: 'Education',
  employment: 'Labor',
  housing: 'Housing',
  crypto: 'Digital Assets',
  emerging: 'Emerging Threats',
};

// Parse briefing text into structured sections (5Ws+H + OUTLOOK + Position)
function parseBriefing(text: string) {
  const sections: { type: string; content: string }[] = [];

  // 5Ws + H pattern matching
  const whatMatch = text.match(/WHAT:([^]*?)(?=WHO:|WHERE:|WHEN:|WHY:|US IMPACT:|OUTLOOK:|Position:|$)/i);
  const whoMatch = text.match(/WHO:([^]*?)(?=WHERE:|WHEN:|WHY:|US IMPACT:|OUTLOOK:|Position:|$)/i);
  const whereMatch = text.match(/WHERE:([^]*?)(?=WHEN:|WHY:|US IMPACT:|OUTLOOK:|Position:|$)/i);
  const whenMatch = text.match(/WHEN:([^]*?)(?=WHY:|US IMPACT:|OUTLOOK:|Position:|$)/i);
  const whyMatch = text.match(/WHY:([^]*?)(?=US IMPACT:|OUTLOOK:|Position:|$)/i);
  const usImpactMatch = text.match(/US IMPACT:([^]*?)(?=OUTLOOK:|Position:|$)/i);
  const outlookMatch = text.match(/OUTLOOK:([^]*?)(?=Position:|$)/i);
  const positionMatch = text.match(/Position:([^]*?)$/i);

  // Legacy format support
  const currentMatch = text.match(/CURRENT:([^]*?)(?=OUTLOOK:|Position:|$)/i);

  // Add sections in display order
  if (whatMatch) sections.push({ type: 'what', content: whatMatch[1].trim() });
  if (whoMatch) sections.push({ type: 'who', content: whoMatch[1].trim() });
  if (whereMatch) sections.push({ type: 'where', content: whereMatch[1].trim() });
  if (whenMatch) sections.push({ type: 'when', content: whenMatch[1].trim() });
  if (whyMatch) sections.push({ type: 'why', content: whyMatch[1].trim() });
  if (usImpactMatch) sections.push({ type: 'usimpact', content: usImpactMatch[1].trim() });
  if (currentMatch && !whatMatch) sections.push({ type: 'current', content: currentMatch[1].trim() });
  if (outlookMatch) sections.push({ type: 'outlook', content: outlookMatch[1].trim() });
  if (positionMatch) sections.push({ type: 'position', content: positionMatch[1].trim() });

  // If no structured format, return as-is
  if (sections.length === 0) {
    sections.push({ type: 'content', content: text });
  }

  return sections;
}

export default function BriefingsPage() {
  const [selectedPreset, setSelectedPreset] = useState('global');
  const { briefings, metadata, loading, refetch, isWarming, warmupStatus } = useIntelBriefing(selectedPreset);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [showGlossary, setShowGlossary] = useState(false);
  const [expandedCategory, setExpandedCategory] = useState<string | null>('Political & Security');

  // Admin-only emergency refresh state
  const [isAdmin, setIsAdmin] = useState(false);
  const [showEmergencyModal, setShowEmergencyModal] = useState(false);
  const [emergencyLoading, setEmergencyLoading] = useState(false);
  const [emergencyResult, setEmergencyResult] = useState<{ success: boolean; message: string } | null>(null);

  // Check if user is admin on mount
  useEffect(() => {
    async function checkAdmin() {
      try {
        const { data: { user }, error: authError } = await supabase.auth.getUser();
        console.log('[BRIEFINGS] Auth check:', {
          userId: user?.id,
          email: user?.email,
          authError: authError?.message || null
        });

        if (authError) {
          console.error('[BRIEFINGS] Auth error:', authError);
          return;
        }

        if (!user) {
          console.log('[BRIEFINGS] No user session found');
          return;
        }

        const { data: profile, error: profileError } = await supabase
          .from('profiles')
          .select('role, email')
          .eq('id', user.id)
          .single();

        console.log('[BRIEFINGS] Profile check:', {
          profile,
          profileError: profileError?.message || null,
          profileCode: profileError?.code || null
        });

        if (profileError) {
          console.error('[BRIEFINGS] Profile query failed:', profileError);
          // RLS might be blocking - user can still use the app
          return;
        }

        const userRole = (profile as { role?: string })?.role;
        console.log('[BRIEFINGS] User role:', userRole);

        if (userRole === 'admin') {
          console.log('[BRIEFINGS] ✓ User is admin, showing Emergency Refresh button');
          setIsAdmin(true);
        } else {
          console.log('[BRIEFINGS] User role is not admin:', userRole);
        }
      } catch (err) {
        console.error('[BRIEFINGS] Exception in checkAdmin:', err);
      }
    }
    void checkAdmin();
  }, []);

  const handleLoad = async () => {
    setHasLoaded(true);
    await refetch();
  };

  // Emergency refresh handler - costs $1-3 per call
  const handleEmergencyRefresh = async () => {
    setEmergencyLoading(true);
    setEmergencyResult(null);
    try {
      const response = await fetch('/api/intel-briefing', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include', // Send auth cookies for server-side admin validation
        body: JSON.stringify({ preset: 'global' }), // Force fresh generation
      });

      // Handle non-JSON responses gracefully
      const contentType = response.headers.get('content-type');
      let data;

      if (contentType && contentType.includes('application/json')) {
        data = await response.json();
      } else {
        // API returned non-JSON (likely an error)
        const text = await response.text();
        console.error('Emergency refresh returned non-JSON:', text);
        data = { error: `API Error: ${text.substring(0, 200)}...` };
      }

      if (response.ok && data.success) {
        setEmergencyResult({ success: true, message: 'Data refreshed successfully! Reloading...' });
        // Reload the page after a short delay to show success
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      } else {
        const errorMsg = data.error || data.details || 'Failed to refresh data';
        setEmergencyResult({ success: false, message: errorMsg });
      }
    } catch (error) {
      console.error('Emergency refresh error:', error);
      setEmergencyResult({
        success: false,
        message: error instanceof Error ? error.message : 'Network error - check console'
      });
    } finally {
      setEmergencyLoading(false);
    }
  };

  const formatTimestamp = (ts: string) => {
    const date = new Date(ts);
    return date.toLocaleString('en-US', {
      month: 'numeric',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });
  };

  return (
    <div className="space-y-6">
      {/* Header - News Masthead Style */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 border-b border-white/[0.06] pb-4">
        <div>
          <div className="flex items-center gap-3">
            <div className="p-2 bg-cyan-500/20 rounded-lg">
              <Radio className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight">Intelligence Briefing</h1>
              <p className="text-slate-500 text-xs sm:text-sm flex items-center gap-2">
                <Clock className="w-3 h-3" />
                <span className="hidden xs:inline">Multi-source fusion • Real-time analysis • Actionable intelligence</span>
                <span className="xs:hidden">Real-time intel</span>
              </p>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {/* Emergency Refresh Button - ADMIN ONLY */}
          {isAdmin && (
            <button
              onClick={() => setShowEmergencyModal(true)}
              className="flex items-center gap-2 px-3 sm:px-4 py-2 min-h-[44px] bg-red-600 hover:bg-red-500 rounded-xl border border-red-400/50 text-white font-bold shadow-lg shadow-red-500/30 hover:shadow-red-500/50 transition-all animate-pulse hover:animate-none"
              title="Force refresh all data via LFBM (~$0.001) - ADMIN ONLY"
            >
              <Zap className="w-5 h-5" />
              <span className="text-xs sm:text-sm uppercase tracking-wide">Refresh</span>
            </button>
          )}
          <button
            onClick={() => setShowGlossary(true)}
            className="flex items-center gap-2 px-3 py-2 min-h-[44px] bg-[rgba(18,18,26,0.7)] backdrop-blur-sm rounded-xl border border-white/[0.06] text-slate-400 hover:text-white hover:border-white/[0.12] transition-all"
          >
            <BookOpen className="w-4 h-4" />
            <span className="text-sm">Glossary</span>
          </button>
        </div>
      </div>

      {/* Preset selector - tabs */}
      <div className="flex items-center gap-2 overflow-x-auto pb-2">
        {PRESETS.map((preset) => {
          const Icon = preset.icon;
          return (
            <button
              key={preset.id}
              onClick={() => {
                setSelectedPreset(preset.id);
                setHasLoaded(false);
              }}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg border whitespace-nowrap transition-all ${
                selectedPreset === preset.id
                  ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-300'
                  : 'bg-[rgba(18,18,26,0.7)] border-white/[0.06] text-slate-400 hover:border-white/[0.12]'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span className="text-sm font-medium">{preset.name}</span>
            </button>
          );
        })}
      </div>

      {/* Warmup Screen - shown when cache is warming */}
      {isWarming && warmupStatus && (
        <WarmupScreen
          estimatedWaitSeconds={warmupStatus.estimatedWaitSeconds}
          message={warmupStatus.message}
          preset={selectedPreset as 'global' | 'nato' | 'brics' | 'conflict'}
          onComplete={() => {
            // The hook handles refetching automatically
            console.log('[BRIEFINGS] Warmup complete!');
          }}
          pollEndpoint="/api/intel-briefing"
          pollBody={{ preset: selectedPreset }}
        />
      )}

      {/* Load button or briefing content */}
      {!hasLoaded && !isWarming ? (
        <GlassCard blur="heavy" className="p-8 text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-cyan-500/20 flex items-center justify-center">
            <FileText className="w-8 h-8 text-cyan-400" />
          </div>
          <h2 className="text-lg font-semibold text-white mb-2">
            {PRESETS.find(p => p.id === selectedPreset)?.name} Intel Brief
          </h2>
          <p className="text-slate-400 mb-6 max-w-md mx-auto">
            Generate a comprehensive intelligence assessment covering political, economic, security, technology, and social domains.
          </p>
          <GlassButton
            variant="primary"
            glow
            onClick={() => void handleLoad()}
            className="px-6"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Generate Briefing
          </GlassButton>
          <p className="text-xs text-slate-600 mt-4">
            Synthesized from wire services, OSINT, institutional research, and proprietary analysis
          </p>
        </GlassCard>
      ) : loading && !isWarming ? (
        <div className="space-y-4">
          <GlassCard blur="heavy" className="p-6">
            <div className="animate-pulse space-y-4">
              <div className="h-6 bg-white/10 rounded w-1/3" />
              <div className="h-4 bg-white/10 rounded w-full" />
              <div className="h-4 bg-white/10 rounded w-5/6" />
            </div>
          </GlassCard>
          {[...Array(3)].map((_, i) => (
            <GlassCard key={i} blur="light" className="p-4">
              <div className="animate-pulse space-y-3">
                <div className="h-4 bg-white/10 rounded w-40" />
                <div className="h-3 bg-white/10 rounded w-full" />
                <div className="h-3 bg-white/10 rounded w-4/5" />
              </div>
            </GlassCard>
          ))}
        </div>
      ) : !isWarming ? (
        <div className="space-y-6">
          {/* Executive Summary - Lead Story */}
          <GlassCard blur="heavy" className="p-6 border-l-4 border-cyan-500">
            <div className="flex items-start justify-between mb-4">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-medium text-cyan-400 uppercase tracking-wider">
                    Executive Summary
                  </span>
                  <span className="text-xs text-slate-600">•</span>
                  <span className="text-xs text-slate-500">
                    {metadata?.timestamp ? formatTimestamp(metadata.timestamp) : 'Just now'}
                  </span>
                </div>
                <h2 className="text-xl font-semibold text-white leading-tight">
                  {PRESETS.find(p => p.id === selectedPreset)?.name}: Situation Assessment
                </h2>
              </div>
              {metadata && (
                <div className="flex items-center gap-1 shrink-0">
                  <span className={`px-3 py-1.5 rounded-lg text-sm font-bold uppercase tracking-wide ${getRiskBadgeStyle(metadata.overallRisk)}`}>
                    {metadata.overallRisk} Risk
                  </span>
                </div>
              )}
            </div>
            <p className="text-slate-300 leading-relaxed text-[15px]">
              {briefings?.summary}
            </p>
          </GlassCard>

          {/* Domain Categories - Accordion */}
          <div className="space-y-3">
            {Object.entries(DOMAIN_CATEGORIES).map(([category, config]) => {
              const CategoryIcon = config.icon;
              const isExpanded = expandedCategory === category;
              const categoryDomains = config.domains.filter(d => briefings?.[d as keyof typeof briefings]);

              if (categoryDomains.length === 0) return null;

              return (
                <div key={category} className="rounded-xl overflow-hidden border border-white/[0.06]">
                  {/* Category Header */}
                  <button
                    onClick={() => setExpandedCategory(isExpanded ? null : category)}
                    className={`w-full flex items-center justify-between p-4 transition-colors ${
                      isExpanded ? `${config.bgColor}` : 'bg-[rgba(18,18,26,0.7)] hover:bg-[rgba(18,18,26,0.9)]'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <CategoryIcon className={`w-5 h-5 ${config.color}`} />
                      <span className="font-semibold text-white">{category}</span>
                      <span className="text-xs text-slate-500">
                        {categoryDomains.length} domains
                      </span>
                    </div>
                    <ChevronRight className={`w-5 h-5 text-slate-400 transition-transform ${isExpanded ? 'rotate-90' : ''}`} />
                  </button>

                  {/* Expanded Content */}
                  {isExpanded && (
                    <div className={`border-t ${config.borderColor} bg-black/20`}>
                      {categoryDomains.map((domain, idx) => {
                        const content = briefings?.[domain as keyof typeof briefings] as string;
                        const parsed = parseBriefing(content);

                        return (
                          <div
                            key={domain}
                            className={`p-4 ${idx !== categoryDomains.length - 1 ? 'border-b border-white/[0.04]' : ''}`}
                          >
                            <h4 className={`text-sm font-medium ${config.color} mb-3 uppercase tracking-wider`}>
                              {DOMAIN_LABELS[domain] || domain}
                            </h4>

                            {parsed.map((section, sIdx) => (
                              <div key={sIdx} className={sIdx > 0 ? 'mt-3' : ''}>
                                {/* 5Ws+H Sections */}
                                {section.type === 'what' && (
                                  <p className="text-sm text-slate-200 leading-relaxed">{section.content}</p>
                                )}
                                {section.type === 'who' && (
                                  <div className="flex items-start gap-2 text-sm">
                                    <span className="text-[10px] font-bold text-blue-400 bg-blue-500/20 px-1.5 py-0.5 rounded uppercase shrink-0 mt-0.5">
                                      Actors
                                    </span>
                                    <p className="text-slate-400 leading-relaxed">{section.content}</p>
                                  </div>
                                )}
                                {section.type === 'where' && (
                                  <div className="flex items-start gap-2 text-sm">
                                    <span className="text-[10px] font-bold text-purple-400 bg-purple-500/20 px-1.5 py-0.5 rounded uppercase shrink-0 mt-0.5">
                                      Where
                                    </span>
                                    <p className="text-slate-400 leading-relaxed">{section.content}</p>
                                  </div>
                                )}
                                {section.type === 'when' && (
                                  <div className="flex items-start gap-2 text-sm">
                                    <span className="text-[10px] font-bold text-orange-400 bg-orange-500/20 px-1.5 py-0.5 rounded uppercase shrink-0 mt-0.5">
                                      When
                                    </span>
                                    <p className="text-slate-400 leading-relaxed">{section.content}</p>
                                  </div>
                                )}
                                {section.type === 'why' && (
                                  <div className="flex items-start gap-2 text-sm">
                                    <span className="text-[10px] font-bold text-pink-400 bg-pink-500/20 px-1.5 py-0.5 rounded uppercase shrink-0 mt-0.5">
                                      Why
                                    </span>
                                    <p className="text-slate-400 leading-relaxed">{section.content}</p>
                                  </div>
                                )}
                                {section.type === 'usimpact' && (
                                  <div className="mt-2 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
                                    <div className="flex items-start gap-2">
                                      <span className="text-[10px] font-bold text-red-400 bg-red-500/20 px-1.5 py-0.5 rounded uppercase shrink-0 mt-0.5">
                                        🇺🇸 US
                                      </span>
                                      <p className="text-sm text-red-200 leading-relaxed">{section.content}</p>
                                    </div>
                                  </div>
                                )}
                                {/* Legacy and standard sections */}
                                {section.type === 'current' && (
                                  <div className="flex items-start gap-2">
                                    <span className="text-[10px] font-bold text-emerald-500 bg-emerald-500/20 px-1.5 py-0.5 rounded uppercase shrink-0 mt-0.5">
                                      Now
                                    </span>
                                    <p className="text-sm text-slate-300 leading-relaxed">{section.content}</p>
                                  </div>
                                )}
                                {section.type === 'outlook' && (
                                  <div className="flex items-start gap-2 mt-2">
                                    <span className="text-[10px] font-bold text-amber-500 bg-amber-500/20 px-1.5 py-0.5 rounded uppercase shrink-0 mt-0.5">
                                      Outlook
                                    </span>
                                    <p className="text-sm text-amber-200/80 leading-relaxed">{section.content}</p>
                                  </div>
                                )}
                                {section.type === 'position' && (
                                  <div className="flex items-start gap-2 mt-2 p-3 bg-cyan-500/10 border border-cyan-500/20 rounded-lg">
                                    <span className="text-[10px] font-bold text-cyan-400 bg-cyan-500/20 px-1.5 py-0.5 rounded uppercase shrink-0 mt-0.5">
                                      Action
                                    </span>
                                    <p className="text-sm text-cyan-200 leading-relaxed">{section.content}</p>
                                  </div>
                                )}
                                {section.type === 'content' && (
                                  <p className="text-sm text-slate-300 leading-relaxed">{section.content}</p>
                                )}
                              </div>
                            ))}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* NSM - Call to Action */}
          {briefings?.nsm && (
            <GlassCard blur="heavy" className="p-6 border-2 border-cyan-500/30 bg-gradient-to-r from-cyan-500/10 to-blue-500/10">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-cyan-500/20 rounded-xl shrink-0">
                  <Target className="w-6 h-6 text-cyan-400" />
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="text-lg font-semibold text-cyan-300">
                      Next Strategic Move
                    </h3>
                    <HelpTip term="NSM (Next Strategic Move)" skillLevel="standard" size={12} />
                  </div>
                  <p className="text-slate-300 leading-relaxed whitespace-pre-line">
                    {briefings.nsm}
                  </p>
                </div>
              </div>
            </GlassCard>
          )}

          {/* Footer */}
          <div className="flex items-center justify-between text-xs text-slate-600 px-2">
            <div className="flex items-center gap-2">
              <span>Sources: Wire services, OSINT, institutional research, proprietary signals</span>
            </div>
            <GlassButton
              variant="ghost"
              size="sm"
              onClick={() => {
                setHasLoaded(false);
                setTimeout(() => void handleLoad(), 100);
              }}
            >
              <RefreshCw className="w-3 h-3 mr-1" />
              Refresh
            </GlassButton>
          </div>
        </div>
      ) : null}

      {/* Glossary Modal */}
      <Glossary
        isOpen={showGlossary}
        onClose={() => setShowGlossary(false)}
        skillLevel="standard"
      />

      {/* Emergency Refresh Confirmation Modal */}
      {showEmergencyModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
          <div className="bg-slate-900 border-2 border-red-500/50 rounded-2xl p-6 max-w-md mx-4 shadow-2xl shadow-red-500/20">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-red-500/20 rounded-lg">
                  <Zap className="w-6 h-6 text-red-400" />
                </div>
                <h2 className="text-xl font-bold text-white">Emergency Refresh</h2>
              </div>
              <button
                onClick={() => {
                  setShowEmergencyModal(false);
                  setEmergencyResult(null);
                }}
                className="text-slate-500 hover:text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4">
              <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
                <p className="text-emerald-200 text-sm font-medium mb-2">This will:</p>
                <ul className="text-sm text-slate-300 space-y-1 ml-4 list-disc">
                  <li>Force a full data refresh (deterministic templates)</li>
                  <li>Regenerate all intelligence briefings</li>
                  <li>Cost approximately <span className="font-bold text-emerald-300">$0.00</span> (zero-LLM)</li>
                  <li>Replace all cached briefing data</li>
                </ul>
              </div>

              <p className="text-slate-400 text-sm">
                Use this only when the database is empty, fallback data is showing, or you need urgent fresh intel.
              </p>

              {emergencyResult && (
                <div className={`p-3 rounded-lg ${
                  emergencyResult.success
                    ? 'bg-emerald-500/20 border border-emerald-500/30 text-emerald-300'
                    : 'bg-red-500/20 border border-red-500/30 text-red-300'
                }`}>
                  {emergencyResult.message}
                </div>
              )}

              <div className="flex gap-3">
                <button
                  onClick={() => {
                    setShowEmergencyModal(false);
                    setEmergencyResult(null);
                  }}
                  className="flex-1 px-4 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-medium transition-colors"
                  disabled={emergencyLoading}
                >
                  Cancel
                </button>
                <button
                  onClick={() => void handleEmergencyRefresh()}
                  disabled={emergencyLoading}
                  className="flex-1 px-4 py-3 bg-red-600 hover:bg-red-500 text-white rounded-lg font-bold transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {emergencyLoading ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Refreshing...
                    </>
                  ) : (
                    <>
                      <Zap className="w-4 h-4" />
                      Confirm Refresh
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
