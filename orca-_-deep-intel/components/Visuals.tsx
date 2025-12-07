import React, { useState } from 'react';
import { GoogleGenAI } from "@google/genai";
import { Image as ImageIcon, Sparkles, Download, Aperture, Layers, Settings, BookOpen } from 'lucide-react';

const ORCA_PROMPTS = [
  "A futuristic cybernetic orca whale breaching from a digital ocean of data, neon blue and cyan aesthetics, 4k highly detailed",
  "Blueprint schematic of a mechanical orca, white lines on dark navy background, technical drawing style",
  "A Pod of Orcas swimming through a submerged ruined city, bioluminescent plants, atmospheric lighting",
  "Abstract geometric representation of Orca intelligence, nodes and connections, deep learning visualization style"
];

const PUBLIC_DOMAIN_PROMPTS = [
  "Vintage 19th-century scientific illustration of an Orca (Orcinus orca), highly detailed etching, black and white, public domain style",
  "Marine biology lithograph of a Killer Whale skeleton, anatomical accuracy, aged paper texture, academic illustration",
  "Victorian era naturalist sketch of marine life including orcas, pencil and ink, field study aesthetic",
  "Woodcut print of a whale breaching, Hokusai wave style, traditional ink wash"
];

type AspectRatio = '1:1' | '16:9' | '9:16' | '4:3' | '3:4';
type ImageStyle = 'NONE' | 'PHOTOREALISTIC' | 'CYBERNETIC' | 'BLUEPRINT' | 'SCIENTIFIC' | 'CINEMATIC';

export const Visuals: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [aspectRatio, setAspectRatio] = useState<AspectRatio>('1:1');
  const [style, setStyle] = useState<ImageStyle>('NONE');
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const getStyleModifier = (s: ImageStyle) => {
    switch (s) {
      case 'PHOTOREALISTIC': return ', photorealistic, 8k, highly detailed, vivid colors';
      case 'CYBERNETIC': return ', cyberpunk, neon nodes, data visualization, glowing circuits';
      case 'BLUEPRINT': return ', technical blueprint, white lines on blue background, schematic';
      case 'SCIENTIFIC': return ', vintage scientific illustration, lithograph style, detailed etching, public domain aesthetic';
      case 'CINEMATIC': return ', cinematic lighting, dramatic angle, movie still, depth of field';
      default: return '';
    }
  };

  const handleGenerate = async (overridePrompt?: string) => {
    const basePrompt = overridePrompt || prompt;
    if (!basePrompt) return;

    setLoading(true);
    setError(null);
    setGeneratedImage(null);

    const fullPrompt = `${basePrompt} ${getStyleModifier(style)}`;

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: {
          parts: [{ text: fullPrompt }]
        },
        config: {
          imageConfig: {
            aspectRatio: aspectRatio
          }
        }
      });

      if (response.candidates && response.candidates[0].content.parts) {
         let foundImage = false;
         for (const part of response.candidates[0].content.parts) {
             if (part.inlineData && part.inlineData.data) {
                 const base64 = part.inlineData.data;
                 const mimeType = part.inlineData.mimeType || 'image/png';
                 setGeneratedImage(`data:${mimeType};base64,${base64}`);
                 foundImage = true;
                 break;
             }
         }
         if (!foundImage) {
             setError("No visual data returned. The model may have returned text instead.");
         }
      } else {
          setError("Empty response from matrix.");
      }
    } catch (err: any) {
      console.error(err);
      setError("Synthesis failed. " + (err.message || "Unknown error"));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col p-6 overflow-y-auto">
      <div className="mb-8 border-b border-border-default pb-4 flex justify-between items-end">
        <div>
          <h1 className="text-2xl font-mono font-bold text-primary flex items-center gap-3">
            <Aperture className="animate-spin-slow" />
            ORCA SYNTHESIS
          </h1>
          <p className="text-sm text-text-secondary mt-2 font-mono">
            Advanced visual generation module. Create public domain reference art or futuristic projections.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 h-full">
        {/* Left Control Panel */}
        <div className="col-span-12 lg:col-span-4 space-y-6 flex flex-col h-full">
          
          {/* Settings Panel */}
          <div className="card bg-surface/50 backdrop-blur-sm border-border-muted p-4 space-y-4">
             <div className="flex items-center gap-2 text-xs font-mono text-primary font-bold uppercase tracking-wider mb-2">
                <Settings size={14} /> Configuration
             </div>
             
             <div className="grid grid-cols-2 gap-4">
               <div>
                 <label className="text-[10px] font-mono text-text-muted block mb-1">ASPECT RATIO</label>
                 <select 
                   value={aspectRatio}
                   onChange={(e) => setAspectRatio(e.target.value as AspectRatio)}
                   className="w-full bg-app/50 border border-border-strong rounded px-2 py-1.5 text-xs font-mono text-text-primary focus:border-primary outline-none"
                 >
                   <option value="1:1">1:1 (SQUARE)</option>
                   <option value="16:9">16:9 (LANDSCAPE)</option>
                   <option value="9:16">9:16 (PORTRAIT)</option>
                   <option value="4:3">4:3 (STANDARD)</option>
                   <option value="3:4">3:4 (VERTICAL)</option>
                 </select>
               </div>
               <div>
                 <label className="text-[10px] font-mono text-text-muted block mb-1">STYLE PRESET</label>
                 <select 
                   value={style}
                   onChange={(e) => setStyle(e.target.value as ImageStyle)}
                   className="w-full bg-app/50 border border-border-strong rounded px-2 py-1.5 text-xs font-mono text-text-primary focus:border-primary outline-none"
                 >
                   <option value="NONE">RAW INPUT</option>
                   <option value="SCIENTIFIC">PUBLIC DOMAIN REF</option>
                   <option value="CYBERNETIC">CYBERNETIC</option>
                   <option value="BLUEPRINT">BLUEPRINT</option>
                   <option value="PHOTOREALISTIC">PHOTOREALISTIC</option>
                   <option value="CINEMATIC">CINEMATIC</option>
                 </select>
               </div>
             </div>
          </div>

          {/* Prompt Input */}
          <div className="card bg-surface/50 backdrop-blur-sm border-border-muted flex-shrink-0">
            <label className="text-xs font-mono text-primary font-bold mb-2 block uppercase tracking-wider">
              Prompt Vector
            </label>
            <textarea
              className="w-full bg-app/50 border border-border-strong rounded-md p-3 text-sm font-mono text-text-primary focus:border-primary focus:ring-1 focus:ring-primary outline-none resize-none h-24"
              placeholder="Describe the visual target..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
            <button
              disabled={loading || !prompt}
              onClick={() => void handleGenerate()}
              className="w-full mt-3 bg-primary/20 hover:bg-primary/30 text-primary border border-primary/50 py-2 rounded font-mono text-sm font-bold transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <span className="animate-pulse">SYNTHESIZING...</span>
              ) : (
                <>
                  <Sparkles size={16} /> GENERATE
                </>
              )}
            </button>
          </div>

          {/* Reference Archives */}
          <div className="flex-1 overflow-y-auto space-y-4 pr-1 scrollbar-thin scrollbar-thumb-border-strong">
            {/* Public Domain Section */}
            <div>
              <h3 className="text-[10px] font-mono text-text-muted font-bold mb-2 uppercase tracking-wider flex items-center gap-2 sticky top-0 bg-app/90 backdrop-blur py-1 z-10">
                <BookOpen size={10} /> Public Domain Reference
              </h3>
              <div className="space-y-2">
                {PUBLIC_DOMAIN_PROMPTS.map((p, i) => (
                  <button
                    key={i}
                    onClick={() => {
                        setPrompt(p);
                        setStyle('SCIENTIFIC');
                    }}
                    disabled={loading}
                    className="w-full text-left px-3 py-2 text-[10px] font-mono text-text-secondary bg-surface-raised hover:bg-accent/10 hover:text-accent rounded border border-transparent hover:border-accent/30 transition-colors truncate"
                    title={p}
                  >
                    PD_REF_{i}: {p.substring(0, 35)}...
                  </button>
                ))}
              </div>
            </div>

            {/* Standard Orca Prompts */}
            <div>
               <h3 className="text-[10px] font-mono text-text-muted font-bold mb-2 uppercase tracking-wider flex items-center gap-2 sticky top-0 bg-app/90 backdrop-blur py-1 z-10">
                 <Layers size={10} /> System Patterns
               </h3>
               <div className="space-y-2">
                 {ORCA_PROMPTS.map((p, i) => (
                   <button
                     key={i}
                     onClick={() => {
                         setPrompt(p);
                         setStyle('CYBERNETIC');
                     }}
                     disabled={loading}
                     className="w-full text-left px-3 py-2 text-[10px] font-mono text-text-secondary bg-surface-raised hover:bg-primary/10 hover:text-primary rounded border border-transparent hover:border-primary/30 transition-colors truncate"
                     title={p}
                   >
                     SYS_PATT_{i}: {p.substring(0, 35)}...
                   </button>
                 ))}
               </div>
            </div>
          </div>
        </div>

        {/* Right Display Area */}
        <div className="col-span-12 lg:col-span-8 flex flex-col h-full">
            <div className="flex-1 flex items-center justify-center bg-app/50 border border-border-default rounded-xl overflow-hidden relative group min-h-[400px]">
                {/* Background Grid Pattern */}
                <div className="absolute inset-0 opacity-10 pointer-events-none" 
                     style={{ backgroundImage: 'radial-gradient(circle, #334155 1px, transparent 1px)', backgroundSize: '20px 20px' }}>
                </div>

                {loading && (
                    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/60 backdrop-blur-sm">
                        <div className="w-16 h-16 border-4 border-primary/30 border-t-primary rounded-full animate-spin mb-4"></div>
                        <div className="font-mono text-primary text-sm animate-pulse">PROCESSING NEURAL VECTORS...</div>
                    </div>
                )}

                {generatedImage ? (
                    <div className="relative w-full h-full flex items-center justify-center bg-black/40">
                        <img src={generatedImage} alt="Generated output" className="max-w-full max-h-full object-contain shadow-2xl" />
                        <div className="absolute bottom-4 right-4 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                             <a 
                               href={generatedImage} 
                               download={`ORCA_SYNTH_${Date.now()}.png`}
                               className="bg-black/80 hover:bg-primary text-white p-2 rounded border border-white/20"
                             >
                                 <Download size={20} />
                             </a>
                        </div>
                    </div>
                ) : (
                    <div className="text-center opacity-30">
                        <ImageIcon size={64} className="mx-auto mb-4 text-text-muted" />
                        <div className="font-mono text-text-muted text-sm">AWAITING INPUT</div>
                        <div className="font-mono text-text-muted text-xs mt-2">SECURE CHANNEL READY</div>
                    </div>
                )}

                {error && (
                    <div className="absolute bottom-8 left-8 right-8 p-4 bg-red-900/80 border border-red-500/50 rounded text-red-100 font-mono text-xs">
                        ERROR: {error}
                    </div>
                )}
            </div>
            
            <div className="mt-4 flex justify-between items-center text-xs font-mono text-text-muted">
                <div className="flex gap-4">
                  <span>MODEL: GEMINI-2.5-FLASH-IMAGE</span>
                  <span>RATIO: {aspectRatio}</span>
                  <span>STYLE: {style}</span>
                </div>
                <div>SECURE CONNECTION // ENCRYPTED</div>
            </div>
        </div>
      </div>
    </div>
  );
};