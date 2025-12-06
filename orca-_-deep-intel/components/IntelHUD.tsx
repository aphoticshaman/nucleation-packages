
import React from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, GitBranch, Zap, Layers } from 'lucide-react';
import { GENERATE_REGIME_DATA, MOCK_CIC_METRICS } from '../constants';
import { PhaseBasinViz } from './PhaseBasinViz';

const regimeData = GENERATE_REGIME_DATA();

export const IntelHUD: React.FC = () => {
  return (
    <div className="w-full h-full flex flex-col gap-4 p-4 overflow-y-auto scrollbar-thin scrollbar-thumb-border-strong">
      
      {/* 1. CIC Theory Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard 
            label="INTEGRATION (Φ)" 
            value={MOCK_CIC_METRICS.phi.toFixed(2)} 
            icon={Activity}
            trend="stable"
            color="text-primary"
            desc="System Coherence"
        />
        <MetricCard 
            label="ENTROPY H(T|X)" 
            value={MOCK_CIC_METRICS.entropy.toFixed(2)} 
            icon={Layers}
            trend="down"
            color="text-accent"
            desc="Compression Quality"
        />
        <MetricCard 
            label="CAUSAL (C_multi)" 
            value={MOCK_CIC_METRICS.causalMulti.toFixed(2)} 
            icon={GitBranch}
            trend="up"
            color="text-status-success"
            desc="Effective Power"
        />
        <MetricCard 
            label="FREE ENERGY F[T]" 
            value={MOCK_CIC_METRICS.freeEnergy.toFixed(1)} 
            icon={Zap}
            trend="up"
            color="text-status-warning"
            desc="Optimization State"
        />
      </div>

      {/* 2. Regime Detection Dashboard */}
      <div className="card h-64 flex flex-col relative overflow-hidden">
        <div className="flex justify-between items-center mb-4 relative z-10">
            <h3 className="text-xs font-mono font-bold text-text-secondary uppercase tracking-widest flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-status-error animate-pulse"></div>
                Markov-Switching Regime Probability
            </h3>
            <span className="text-[10px] font-mono text-status-error border border-status-error/30 px-2 py-0.5 rounded bg-status-error/10">
                PHASE TRANSITION: T-MINUS 48H
            </span>
        </div>
        <div className="flex-1 w-full min-h-0 relative z-10">
            <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={regimeData}>
                    <defs>
                        <linearGradient id="colorStable" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                        </linearGradient>
                        <linearGradient id="colorVolatile" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                        </linearGradient>
                        <linearGradient id="colorCrisis" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                        </linearGradient>
                    </defs>
                    <XAxis dataKey="timestamp" hide />
                    <YAxis hide />
                    <Tooltip 
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', fontSize: '12px' }}
                        itemStyle={{ color: '#94a3b8' }}
                    />
                    <Area type="monotone" dataKey="stable" stackId="1" stroke="#3b82f6" fill="url(#colorStable)" />
                    <Area type="monotone" dataKey="volatile" stackId="1" stroke="#f59e0b" fill="url(#colorVolatile)" />
                    <Area type="monotone" dataKey="crisis" stackId="1" stroke="#ef4444" fill="url(#colorCrisis)" />
                </AreaChart>
            </ResponsiveContainer>
        </div>
      </div>

      {/* 3. Phase Space / Attractor */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 flex-1 min-h-[200px]">
         <div className="card p-4 flex flex-col">
             <h3 className="text-xs font-mono font-bold text-text-secondary uppercase tracking-widest mb-2 flex items-center justify-between">
                <span>Phase Space Navigator</span>
                <span className="text-[10px] text-primary">BASIN A</span>
             </h3>
             <div className="flex-1 rounded overflow-hidden relative">
                <PhaseBasinViz stability={0.75} isCritical={false} />
             </div>
         </div>

         <div className="card p-4">
             <h3 className="text-xs font-mono font-bold text-text-secondary uppercase tracking-widest mb-4">Epistemic Humidity</h3>
             <div className="space-y-4">
                 <ProgressBar label="SOURCE RELIABILITY" value={85} color="bg-blue-500" />
                 <ProgressBar label="MODEL CONFIDENCE" value={62} color="bg-yellow-500" />
                 <ProgressBar label="EVIDENCE CONFLICT" value={24} color="bg-red-500" />
                 <ProgressBar label="CIC COHERENCE" value={91} color="bg-cyan-500" />
             </div>
         </div>
      </div>
    </div>
  );
};

const MetricCard = ({ label, value, icon: Icon, trend, color, desc }: any) => (
  <div className="card p-3 bg-surface/40 hover:bg-surface/60 border border-white/5 relative overflow-hidden">
     <div className="flex justify-between items-start mb-2">
        <span className="text-[10px] font-mono text-text-muted">{label}</span>
        <Icon size={14} className={`${color} opacity-80`} />
     </div>
     <div className={`text-2xl font-mono font-bold ${color} mb-1`}>{value}</div>
     <div className="flex items-center justify-between">
        <span className="text-[10px] text-text-secondary truncate">{desc}</span>
        <span className="text-[10px] font-mono text-text-muted">{trend === 'up' ? '▲' : trend === 'down' ? '▼' : '−'}</span>
     </div>
  </div>
);

const ProgressBar = ({ label, value, color }: any) => (
    <div>
        <div className="flex justify-between text-[10px] font-mono text-text-muted mb-1">
            <span>{label}</span>
            <span>{value}%</span>
        </div>
        <div className="h-1.5 w-full bg-surface-raised rounded-full overflow-hidden">
            <div className={`h-full ${color}`} style={{ width: `${value}%` }}></div>
        </div>
    </div>
);
