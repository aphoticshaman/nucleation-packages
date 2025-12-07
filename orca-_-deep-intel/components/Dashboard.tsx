
import React, { useState, useEffect, useRef } from 'react';
import { ContextLayer, EntropySource, IntelligencePacket, IntelCategory } from '../types';
import { fetchEntropy } from '../services/entropy';
import { Activity, ChevronDown, ChevronUp, Filter, Clock, Hash, Wifi, Radio } from 'lucide-react';

const CATEGORIES: IntelCategory[] = [
  'DEFENSE', 'CYBER', 'TECH', 'POLITICS', 'HEALTH', 
  'FINANCE', 'SPACE', 'CORP', 'AGRI', 'RESOURCES', 
  'HOUSING', 'EDU', 'CRIME', 'ENTERTAINMENT'
];

// Mock Data Pool - In production, this would be replaced by your GDELT/NewsAPI fetcher
const MOCK_DATA_POOL: Omit<IntelligencePacket, 'timestamp' | 'id'>[] = [
  { context: ContextLayer.SURFACE, category: 'DEFENSE', header: 'TAIWAN STRAIT MOBILIZATION', summary: 'PLAN fleet exercises extending beyond median line.', body: 'Satellite imagery confirms Type 055 destroyer group conducting live-fire anti-access/area denial (A2/AD) drills 12nm west of Penghu. Civilian shipping insurance premiums rose 15% overnight.', source: 'OSINT/Sat', coherence: 0.94 },
  { context: ContextLayer.SURFACE, category: 'DEFENSE', header: 'SUWALKI GAP LOGISTICS', summary: 'Rail transport increase in Kaliningrad corridor.', body: 'Heavy equipment flatbeds spotted moving nightly. Equipment appears to be engineering bridgelayers and counter-mobility assets, suggesting defensive hardening rather than offensive posturing.', source: 'RailNet', coherence: 0.88 },
  { context: ContextLayer.SURFACE, category: 'AGRI', header: 'MIDWEST AQUIFER DEPLETION', summary: 'Ogallala water table levels at historic lows.', body: 'USDA crop reports indicate a 12% reduction in projected corn yields for Nebraska/Kansas. Pivot irrigation usage restrictions likely to trigger localized insolvency events for mid-sized operations.', source: 'USDA/USGS', coherence: 0.91 },
  { context: ContextLayer.SURFACE, category: 'AGRI', header: 'PHOSPHATE SHORTAGE', summary: 'Morocco export quotas tightening fertilizer supply.', body: 'Geopolitical leverage applied via OCP Group. Global food prices correlated to DAP/MAP fertilizer index, predicting a Q3 CPI spike in developing nations reliant on imports.', source: 'TradeFlow', coherence: 0.85 },
  { context: ContextLayer.SURFACE, category: 'HOUSING', header: 'COMMERCIAL RE DEFAULT', summary: 'Metropolitan office vacancy triggering tranche failures.', body: 'San Francisco and Chicago downtown districts showing 35% vacancy. AAA-rated CMBS bonds being repriced. Regional banks holding paper face capital adequacy calls.', source: 'Bloomberg', coherence: 0.96 },
  { context: ContextLayer.SURFACE, category: 'TECH', header: 'TSMC ARIZONA DELAYS', summary: 'Fab 21 operational timeline pushed to 2026.', body: 'Labor disputes and supply chain bottlenecks for EUV lithography machines cited. US domestic semiconductor sovereignty goals slipping. Impact on Apple/Nvidia supply chains minimal near-term but risk elevated.', source: 'Reuters', coherence: 0.82 },
  { context: ContextLayer.SURFACE, category: 'SPACE', header: 'LEO DEBRIS CLOUD', summary: 'Fragmentation event in polar orbit threatens Starlink.', body: 'Defunct Soviet-era booster stage disintegrated. 400+ trackable fragments. ISS performing avoidance maneuver. Insurance underwriters halting policies for launches in 98-degree inclination.', source: 'USSPACECOM', coherence: 0.99 },
  { context: ContextLayer.SURFACE, category: 'RESOURCES', header: 'LITHIUM NATIONALIZATION', summary: 'Chilean government creates state-owned extraction entity.', body: 'SQM and Albemarle contracts under review. Global EV battery supply chain volatility increased. Spot prices stable, futures curve in backwardation suggesting long-term supply constraint fears.', source: 'MiningWatch', coherence: 0.89 },
  
  // DEEP
  { context: ContextLayer.DEEP, category: 'FINANCE', header: 'EURODOLLAR LIQUIDITY GAP', summary: 'Offshore collateral shortage spiking repo rates.', body: 'Shadow banking entities in Cayman/Virgin Islands showing stress. Reverse Repo facility draining faster than anticipated. Treasury General Account refill acting as liquidity vacuum. Systemic risk indicator flashing amber.', source: 'FedWire/OTC', coherence: 0.92 },
  { context: ContextLayer.DEEP, category: 'FINANCE', header: 'YEN CARRY UNWIND', summary: 'BOJ yield curve control tweak triggering capital flight.', body: 'Japanese institutional investors repatriating funds from US Treasuries and French OATs. Global liquidity contraction capability estimated at $2.4 Trillion USD equivalent.', source: 'FXFlows', coherence: 0.87 },
  { context: ContextLayer.DEEP, category: 'CORP', header: 'ESG SCORING FRAUD', summary: 'Algorithmic washing detected in major index funds.', body: 'Correlation analysis of carbon credits vs actual emissions reveals massive divergence. 40% of "Green" bonds funding traditional hydrocarbon extraction via shell subsidiaries.', source: 'ForensicAcct', coherence: 0.95 },
  { context: ContextLayer.DEEP, category: 'HEALTH', header: 'ANTIBIOTIC RESISTANCE', summary: 'Novel MRSA strain identified in Mumbai wastewater.', body: 'Genetic sequencing shows resistance to Colistin (last-resort). Spread vector modeled via international air travel hubs. Probability of pandemic classification < 5% currently but R0 is 1.4.', source: 'WHO/CDC-Internal', coherence: 0.84 },
  { context: ContextLayer.DEEP, category: 'CRIME', header: 'SYNTHETIC OPIOID FLOWS', summary: 'Precursor chemicals rerouted via Central America.', body: 'Sinaloa cartel shifting production labs to Guatemala border. Blockchain analysis links precursor payments to USDT wallets associated with East Asian chemical brokers.', source: 'DEA/FinCEN', coherence: 0.91 },
  { context: ContextLayer.DEEP, category: 'POLITICS', header: 'GERRYMANDER ALGORITHMS', summary: 'AI-generated redistricting maps maximizing incumbent retention.', body: 'New maps in key swing states show mathematical efficiency gap of 14%. Voter disenfranchisement localized to specific demographic clusters. Legal challenges pending but unlikely to resolve before midterms.', source: 'JudicialWatch', coherence: 0.88 },
  
  // DARK
  { context: ContextLayer.DARK, category: 'CYBER', header: 'CRITICAL INFRASTRUCTURE ZERO-DAY', summary: 'Dormant loader found in SCADA PLCs nationwide.', body: 'Polymorphic code "Voltz" detected in power grid controllers. Activation logic tied to specific NTP timestamp. Code provenance suggests state-sponsored actor (APT29 or similar). EDR solutions blind to signature.', source: 'SigInt/Active', coherence: 0.99 },
  { context: ContextLayer.DARK, category: 'CYBER', header: 'QUANTUM EXFILTRATION', summary: '"Store Now Decrypt Later" volume spiking.', body: 'Encrypted diplomatic cables being harvested at ISP level. Adversaries banking on 5-year timeline to CRQC (Cryptographically Relevant Quantum Computer). High-value target comms compromised retroactively.', source: 'NSA/Tao', coherence: 0.94 },
  { context: ContextLayer.DARK, category: 'RESOURCES', header: 'ILLEGAL GOLD MINING', summary: 'Mercury contamination used as biowarfare in Amazon.', body: 'Paramilitary groups poisoning indigenous water tables deliberately to clear land for extraction. Gold laundered through Swiss refineries. Satellite spectral analysis confirms mercury bloom.', source: 'SpecOps/Sat', coherence: 0.89 },
  { context: ContextLayer.DARK, category: 'SPACE', header: 'KILLER SATELLITE MANEUVER', summary: 'Russian "Inspector" sat synchronized orbit with KH-11.', body: 'Object Cosmos-2558 matched velocity and inclination of US Keyhole spy satellite. Range < 50km. Electronic warfare package likely jamming uplinks. Soft-kill capability active.', source: 'NRO', coherence: 0.97 },
  { context: ContextLayer.DARK, category: 'TECH', header: 'NEURAL LACE BACKDOOR', summary: 'BCI implant firmware contains remote override.', body: 'Medical device protocol vulnerability allows write-access to motor cortex stimulation. Theoretical capability to induce seizure or motor lock. Exploit circulating on dark web forums.', source: 'MedSec', coherence: 0.81 },
  { context: ContextLayer.DARK, category: 'CRIME', header: 'RED ROOM AUCTION', summary: 'Live-streamed torture signal triangulated to SE Asia.', body: 'Encrypted onion service hosting pay-per-view violence. Payment flows traced to Monero mixers. Geolocation narrowed to Mekong delta casino complex run by triad syndicates.', source: 'Interpol/Dark', coherence: 0.86 },
];

export const Dashboard: React.FC<{ context: ContextLayer }> = ({ context }) => {
  const [entropySources, setEntropySources] = useState<EntropySource[]>([]);
  const [intelStream, setIntelStream] = useState<IntelligencePacket[]>([]);
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());
  const [activeFilter, setActiveFilter] = useState<IntelCategory | 'ALL'>('ALL');
  const [isLive, setIsLive] = useState(true);
  const streamRef = useRef<HTMLDivElement>(null);

  const toggleExpand = (id: string) => {
    const newSet = new Set(expandedItems);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    setExpandedItems(newSet);
  };

  // 1. Fetch Live Entropy
  useEffect(() => {
    const loadEntropy = async () => {
      const data = await fetchEntropy();
      setEntropySources(data);
    };
    void loadEntropy();
    const interval = setInterval(() => void loadEntropy(), 60000);
    return () => clearInterval(interval);
  }, []);

  // 2. LIVE FEED SIMULATOR
  useEffect(() => {
    // Clear stream on context switch to simulate "tuning" a new channel
    setIntelStream([]);

    const getContextData = () => {
      return MOCK_DATA_POOL.filter(p => {
        // Simple context filter logic for demo
        if (context === ContextLayer.SURFACE) return p.context === ContextLayer.SURFACE;
        if (context === ContextLayer.DEEP) return p.context === ContextLayer.DEEP;
        return p.context === ContextLayer.DARK;
      });
    };

    const availablePackets = getContextData();
    let queueIndex = 0;

    // Initial Burst (Load 3 items immediately)
    const initial = availablePackets.slice(0, 3).map((p, i) => ({
      ...p,
      id: `init-${i}-${Date.now()}`,
      timestamp: new Date().toISOString().slice(11, 16) + 'Z'
    }));
    setIntelStream(initial);
    queueIndex = 3;

    // Interval to push new items one by one (Simulating live WebSocket)
    const interval = setInterval(() => {
      if (!isLive) return;

      if (queueIndex < availablePackets.length) {
        const nextPacket = availablePackets[queueIndex];
        const newPacket: IntelligencePacket = {
          ...nextPacket,
          id: `live-${queueIndex}-${Date.now()}`,
          timestamp: new Date().toISOString().slice(11, 16) + 'Z' // Real-time timestamp
        };

        setIntelStream(prev => [newPacket, ...prev]); // Add to TOP
        queueIndex++;
      } else {
        // Recycle packets with new timestamps to simulate continuous feed
        const randomPacket = availablePackets[Math.floor(Math.random() * availablePackets.length)];
        const recycledPacket: IntelligencePacket = {
           ...randomPacket,
           id: `recycle-${Date.now()}`,
           timestamp: new Date().toISOString().slice(11, 16) + 'Z',
           header: randomPacket.header + (Math.random() > 0.5 ? ' [UPDATE]' : ' [CONFIRMED]') // Add variation
        };
        setIntelStream(prev => [recycledPacket, ...prev]);
      }
    }, 4500); // New intel every 4.5 seconds

    return () => clearInterval(interval);
  }, [context, isLive]);

  // Handle Filtering
  const filteredStream = activeFilter === 'ALL' 
    ? intelStream 
    : intelStream.filter(p => p.category === activeFilter);

  return (
    <div className="flex flex-col h-full bg-app overflow-hidden">
      
      {/* 1. Header & Entropy Ticker */}
      <div className="h-14 shrink-0 border-b border-border-muted bg-surface/80 backdrop-blur flex items-center px-4 overflow-hidden">
        <div className="mr-4 flex items-center text-accent/80 font-mono text-xs font-bold whitespace-nowrap">
          <Activity size={14} className="mr-2 animate-pulse" />
          ENTROPY:
        </div>
        <div className="flex items-center space-x-6 overflow-x-auto scrollbar-none mask-gradient-right">
          {entropySources.map((source, i) => (
            <div key={i} className="flex items-center space-x-2 font-mono text-xs">
              <span className="text-text-muted">{source.label}</span>
              <span className={source.delta > 0 ? 'text-status-success' : 'text-status-error'}>
                {source.delta > 0 ? '▲' : '▼'} {Math.abs(source.delta).toFixed(2)}
              </span>
            </div>
          ))}
          {entropySources.length === 0 && <span className="text-xs text-text-muted font-mono">CALIBRATING SENSORS...</span>}
        </div>
      </div>

      {/* 2. Filter Bar & Live Control */}
      <div className="shrink-0 py-3 px-4 border-b border-border-muted flex items-center justify-between bg-surface">
        <div className="flex items-center gap-2 overflow-x-auto scrollbar-thin scrollbar-thumb-border-strong flex-1 mr-4">
          <div className="flex items-center text-text-muted text-xs font-mono mr-2">
            <Filter size={12} className="mr-1" /> FILTER:
          </div>
          <button
            onClick={() => setActiveFilter('ALL')}
            className={`px-3 py-1 rounded text-[10px] font-mono font-medium tracking-wide transition-colors whitespace-nowrap border ${
              activeFilter === 'ALL' 
                ? 'bg-primary/20 text-primary border-primary/30' 
                : 'bg-surface-raised text-text-secondary border-border-muted hover:border-text-muted'
            }`}
          >
            ALL SIGNALS
          </button>
          {CATEGORIES.map(cat => (
            <button
              key={cat}
              onClick={() => setActiveFilter(cat)}
              className={`px-3 py-1 rounded text-[10px] font-mono font-medium tracking-wide transition-colors whitespace-nowrap border ${
                activeFilter === cat 
                  ? 'bg-accent/20 text-accent border-accent/30' 
                  : 'bg-surface-raised text-text-secondary border-border-muted hover:border-text-muted'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>

        {/* Live Feed Toggle */}
        <button 
          onClick={() => setIsLive(!isLive)}
          className={`flex items-center gap-2 px-3 py-1 rounded border text-[10px] font-mono font-bold transition-all ${isLive ? 'bg-status-success/10 text-status-success border-status-success/30' : 'bg-surface-raised text-text-muted border-border-muted'}`}
        >
           {isLive ? <Radio size={14} className="animate-pulse" /> : <Wifi size={14} className="opacity-50" />}
           {isLive ? 'LIVE FEED ACTIVE' : 'FEED PAUSED'}
        </button>
      </div>

      {/* 3. Main Content - Chronological Accordion List */}
      <div className="flex-1 overflow-y-auto p-0 scrollbar-thin scrollbar-thumb-border-strong relative" ref={streamRef}>
        <div className="max-w-5xl mx-auto w-full">
          {filteredStream.length === 0 && (
            <div className="flex flex-col items-center justify-center h-64 text-text-muted">
               <Radio size={32} className="animate-spin mb-2 opacity-50" />
               <span className="font-mono text-xs">ACQUIRING SIGNAL...</span>
            </div>
          )}

          {filteredStream.map((packet, index) => {
            const isExpanded = expandedItems.has(packet.id);
            // New items animate in
            const isNew = index === 0 && isLive;
            
            return (
              <div 
                key={packet.id} 
                className={`border-b border-border-muted transition-all duration-500 ${
                  isNew ? 'bg-primary/10 animate-fade-in' : isExpanded ? 'bg-surface-raised/50' : 'hover:bg-surface-raised/30 bg-app'
                }`}
              >
                {/* Header Row - Always Visible */}
                <div 
                  className="flex items-center py-3 px-4 cursor-pointer group"
                  onClick={() => toggleExpand(packet.id)}
                >
                  {/* Category Pill */}
                  <div className="w-24 shrink-0">
                    <span className="text-[10px] font-mono font-bold text-text-muted bg-surface-raised border border-border-muted px-1.5 py-0.5 rounded group-hover:border-text-muted transition-colors">
                      {packet.category}
                    </span>
                  </div>

                  {/* Headline */}
                  <div className="flex-1 min-w-0 pr-4">
                    <div className="flex items-baseline gap-3">
                      <h3 className={`font-mono text-sm truncate transition-colors ${isExpanded ? 'text-primary font-bold' : 'text-text-primary group-hover:text-primary/80'}`}>
                        {packet.header}
                      </h3>
                      <span className="text-xs text-text-secondary hidden sm:inline-block truncate opacity-70">
                        // {packet.summary}
                      </span>
                    </div>
                  </div>

                  {/* Metadata Right */}
                  <div className="flex items-center gap-4 shrink-0">
                    <div className="flex items-center text-text-muted text-[10px] font-mono hidden sm:flex">
                      <Clock size={10} className="mr-1" />
                      {packet.timestamp}
                    </div>
                    <div className="flex items-center text-accent text-[10px] font-mono hidden sm:flex bg-accent/5 px-2 py-0.5 rounded border border-accent/10">
                      <Hash size={10} className="mr-1" />
                      {packet.coherence.toFixed(2)}
                    </div>
                    <div className="text-text-muted group-hover:text-text-primary transition-colors">
                      {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </div>
                  </div>
                </div>

                {/* Expanded Body */}
                {isExpanded && (
                  <div className="px-4 pb-4 pl-4 sm:pl-28 animate-slide-up">
                    <div className="p-4 bg-surface rounded border border-border-muted shadow-inner">
                      
                      {/* Full Summary (Mobile view helper) */}
                      <div className="sm:hidden mb-3 pb-3 border-b border-border-muted">
                        <span className="text-xs font-bold text-text-primary block mb-1">SUMMARY</span>
                        <p className="text-sm text-text-secondary">{packet.summary}</p>
                      </div>

                      <div className="flex gap-6 flex-col md:flex-row">
                        <div className="flex-1">
                          <span className="text-[10px] font-mono text-primary mb-2 block tracking-widest uppercase">Analysis Body</span>
                          <p className="text-sm text-text-secondary leading-relaxed font-sans">
                            {packet.body}
                          </p>
                        </div>
                        
                        <div className="md:w-48 shrink-0 flex flex-col gap-2 pt-2 md:pt-0 md:border-l md:border-border-muted md:pl-4">
                          <div>
                            <span className="text-[10px] font-mono text-text-muted block">SOURCE</span>
                            <span className="text-xs font-mono text-text-primary">{packet.source}</span>
                          </div>
                          <div>
                            <span className="text-[10px] font-mono text-text-muted block">CONTEXT</span>
                            <span className="text-xs font-mono text-text-primary">{packet.context.split(' ')[0]}</span>
                          </div>
                          <div>
                            <span className="text-[10px] font-mono text-text-muted block">ID</span>
                            <span className="text-xs font-mono text-text-primary truncate">{packet.id}</span>
                          </div>
                          <div className="mt-2">
                             <button className="w-full py-1 bg-primary/10 hover:bg-primary/20 text-primary border border-primary/30 rounded text-[10px] font-mono transition-colors">
                               EXPORT PACKET
                             </button>
                          </div>
                        </div>
                      </div>

                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
