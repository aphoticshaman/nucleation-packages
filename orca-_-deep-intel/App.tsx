
import React, { useState } from 'react';
import { Navigation } from './components/Navigation';
import { Dashboard } from './components/Dashboard';
import { CausalCanvas } from './components/CausalCanvas';
import { IntelHUD } from './components/IntelHUD';
import { CommandPalette } from './components/CommandPalette';
import { ContextLayer } from './types';

const App: React.FC = () => {
  const [context, setContext] = useState<ContextLayer>(ContextLayer.SURFACE);
  const [view, setView] = useState<'causal' | 'workbench' | 'predictive'>('causal');

  return (
    <div className="flex h-screen w-full bg-app text-text-primary font-sans selection:bg-primary/30 selection:text-white overflow-hidden relative">
      
      {/* Background Ambient Effects - LatticeForge Theme */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-blue-900/10 rounded-full blur-[150px] animate-pulse-slow"></div>
        <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-cyan-900/10 rounded-full blur-[150px] animate-pulse-slow" style={{ animationDelay: '1.5s' }}></div>
      </div>

      <CommandPalette />

      <Navigation 
        context={context} 
        setContext={setContext} 
        currentView={view}
        setView={setView}
      />
      
      <main className="flex-1 relative z-10 flex flex-col h-full overflow-hidden">
        
        {/* Mobile Header */}
        <div className="h-14 border-b border-border-default flex items-center justify-between px-6 bg-surface/50 backdrop-blur-sm lg:hidden">
            <span className="font-mono font-bold text-primary tracking-widest">LATTICE</span>
            <div className="text-[10px] font-mono text-text-muted">{view.toUpperCase()}</div>
        </div>

        <div className="flex-1 relative overflow-hidden flex flex-col lg:flex-row">
             {view === 'causal' && (
               <>
                 {/* Main Canvas Area */}
                 <div className="flex-1 relative h-full">
                    <CausalCanvas />
                    
                    {/* Floating HUD Overlay */}
                    <div className="absolute bottom-4 right-4 w-80 pointer-events-none">
                       {/* Mini Status or Alerts could go here */}
                    </div>
                 </div>
                 
                 {/* Right Sidebar - Intel HUD (Regimes + CIC) */}
                 <div className="w-full lg:w-96 h-[40vh] lg:h-full border-t lg:border-t-0 lg:border-l border-border-default bg-surface/30 backdrop-blur-sm z-20">
                    <IntelHUD />
                 </div>
               </>
             )}

             {view === 'workbench' && (
                 <div className="w-full h-full">
                    <Dashboard context={context} />
                 </div>
             )}

             {view === 'predictive' && (
                 <div className="w-full h-full flex items-center justify-center">
                    <div className="text-center opacity-50">
                        <div className="text-4xl font-mono text-primary mb-4">PREDICTIVE OPS</div>
                        <div className="font-mono text-sm">SPARSE GAUSSIAN PROCESS :: INITIALIZING</div>
                    </div>
                 </div>
             )}
        </div>
      </main>
    </div>
  );
};

export default App;
