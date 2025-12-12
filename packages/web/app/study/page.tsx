'use client';

import { useState, useRef, useEffect } from 'react';
import {
  MessageSquare,
  Code,
  Search,
  FileText,
  BarChart3,
  Send,
  Zap,
  Clock,
  Brain,
  Sparkles,
  GitBranch,
  Settings,
  ChevronDown,
  Loader2,
  Copy,
  ThumbsUp,
  ThumbsDown,
  RefreshCw,
} from 'lucide-react';

// =============================================================================
// TYPES
// =============================================================================

type StudyMode = 'chat' | 'code' | 'research' | 'brief' | 'analyze';
type ResearchDepth = 'instant' | 'moderate' | 'thorough';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  thinking?: string;
  timestamp: Date;
  model?: string;
  tier?: 'workhorse' | 'elle';
  latency_ms?: number;
  research?: {
    depth: ResearchDepth;
    sourcesFound: number;
    gdeltSignals: number;
    pagesAnalyzed: number;
  };
}

interface Conversation {
  id: string;
  title: string;
  mode: StudyMode;
  lastMessage: Date;
}

// =============================================================================
// MODE CONFIG
// =============================================================================

const MODE_CONFIG: Record<StudyMode, { icon: typeof MessageSquare; label: string; description: string }> = {
  chat: {
    icon: MessageSquare,
    label: 'Chat',
    description: 'General conversation with Elle',
  },
  code: {
    icon: Code,
    label: 'Code',
    description: 'Code assistant with GitHub integration',
  },
  research: {
    icon: Search,
    label: 'Research',
    description: 'Deep research with web + GDELT',
  },
  brief: {
    icon: FileText,
    label: 'Brief',
    description: 'Generate intel briefings',
  },
  analyze: {
    icon: BarChart3,
    label: 'Analyze',
    description: 'Analyze documents and data',
  },
};

const DEPTH_CONFIG: Record<ResearchDepth, { icon: typeof Zap; label: string; time: string; description: string }> = {
  instant: {
    icon: Zap,
    label: 'Instant',
    time: '~2s',
    description: 'Quick response, cached data',
  },
  moderate: {
    icon: Clock,
    label: 'Moderate',
    time: '~15s',
    description: '1-3 searches, synthesis',
  },
  thorough: {
    icon: Brain,
    label: 'Thorough',
    time: '2-5min',
    description: 'Deep multi-source research',
  },
};

// =============================================================================
// COMPONENT
// =============================================================================

export default function StudyPage() {
  // State
  const [mode, setMode] = useState<StudyMode>('chat');
  const [depth, setDepth] = useState<ResearchDepth>('moderate');
  const [unrestricted, setUnrestricted] = useState(true);
  const [bigBrain, setBigBrain] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showThinking, setShowThinking] = useState<string | null>(null);

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = `${inputRef.current.scrollHeight}px`;
    }
  }, [input]);

  // Send message
  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/study/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage.content,
          conversationId,
          mode,
          depth,
          unrestricted,
          bigBrain,
          useTools: true,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to send message');
      }

      // Update conversation ID
      if (data.conversationId && !conversationId) {
        setConversationId(data.conversationId);
      }

      // Add assistant message
      const assistantMessage: Message = {
        id: data.messageId || `assistant-${Date.now()}`,
        role: 'assistant',
        content: data.content,
        thinking: data.thinking,
        timestamp: new Date(),
        model: data.model,
        tier: data.tier,
        latency_ms: data.latency_ms,
        research: data.research,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      // Add error message
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle key press
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Copy message
  const copyMessage = (content: string) => {
    navigator.clipboard.writeText(content);
  };

  // New conversation
  const newConversation = () => {
    setMessages([]);
    setConversationId(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white">
      {/* Header */}
      <header className="border-b border-white/10 bg-black/20 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-lg">Study Book</h1>
              <p className="text-xs text-slate-400">Elle - Your Intelligence Analyst</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={newConversation}
              className="px-3 py-1.5 text-sm rounded-lg bg-white/5 hover:bg-white/10 transition flex items-center gap-2"
            >
              <RefreshCw className="w-4 h-4" />
              New
            </button>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition"
            >
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto flex">
        {/* Sidebar - Mode & Settings */}
        <aside className="w-64 border-r border-white/10 p-4 hidden lg:block">
          {/* Mode Selection */}
          <div className="mb-6">
            <h3 className="text-xs font-semibold text-slate-400 uppercase mb-2">Mode</h3>
            <div className="space-y-1">
              {(Object.entries(MODE_CONFIG) as [StudyMode, typeof MODE_CONFIG.chat][]).map(([key, config]) => (
                <button
                  key={key}
                  onClick={() => setMode(key)}
                  className={`w-full px-3 py-2 rounded-lg text-left flex items-center gap-2 transition ${
                    mode === key
                      ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                      : 'hover:bg-white/5 text-slate-300'
                  }`}
                >
                  <config.icon className="w-4 h-4" />
                  <span className="text-sm">{config.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Research Depth */}
          <div className="mb-6">
            <h3 className="text-xs font-semibold text-slate-400 uppercase mb-2">Research Depth</h3>
            <div className="space-y-1">
              {(Object.entries(DEPTH_CONFIG) as [ResearchDepth, typeof DEPTH_CONFIG.instant][]).map(([key, config]) => (
                <button
                  key={key}
                  onClick={() => setDepth(key)}
                  className={`w-full px-3 py-2 rounded-lg text-left flex items-center justify-between transition ${
                    depth === key
                      ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                      : 'hover:bg-white/5 text-slate-300'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <config.icon className="w-4 h-4" />
                    <span className="text-sm">{config.label}</span>
                  </div>
                  <span className="text-xs text-slate-500">{config.time}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Toggles */}
          <div className="space-y-3">
            <label className="flex items-center justify-between cursor-pointer">
              <span className="text-sm text-slate-300">Unrestricted</span>
              <button
                onClick={() => setUnrestricted(!unrestricted)}
                className={`w-10 h-6 rounded-full transition ${
                  unrestricted ? 'bg-orange-500' : 'bg-slate-700'
                }`}
              >
                <div
                  className={`w-4 h-4 rounded-full bg-white transition-transform ${
                    unrestricted ? 'translate-x-5' : 'translate-x-1'
                  }`}
                />
              </button>
            </label>

            <label className="flex items-center justify-between cursor-pointer">
              <span className="text-sm text-slate-300">Big Brain</span>
              <button
                onClick={() => setBigBrain(!bigBrain)}
                className={`w-10 h-6 rounded-full transition ${
                  bigBrain ? 'bg-purple-500' : 'bg-slate-700'
                }`}
              >
                <div
                  className={`w-4 h-4 rounded-full bg-white transition-transform ${
                    bigBrain ? 'translate-x-5' : 'translate-x-1'
                  }`}
                />
              </button>
            </label>
          </div>
        </aside>

        {/* Main Chat Area */}
        <main className="flex-1 flex flex-col h-[calc(100vh-57px)]">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500/20 to-cyan-400/20 flex items-center justify-center mb-4">
                  <Sparkles className="w-8 h-8 text-blue-400" />
                </div>
                <h2 className="text-xl font-semibold mb-2">Welcome to Study Book</h2>
                <p className="text-slate-400 max-w-md">
                  I&apos;m Elle, your intelligence analyst. Ask me anything - I have full access to
                  web search, GDELT intel feeds, code execution, and your GitHub repos.
                </p>
                <div className="mt-6 flex flex-wrap gap-2 justify-center">
                  {[
                    'What\'s happening in Eastern Europe today?',
                    'Analyze this code for security issues',
                    'Generate a brief on China-Taiwan tensions',
                    'Help me write a Python script',
                  ].map((suggestion, i) => (
                    <button
                      key={i}
                      onClick={() => setInput(suggestion)}
                      className="px-3 py-1.5 text-sm rounded-lg bg-white/5 hover:bg-white/10 transition text-slate-300"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                      message.role === 'user'
                        ? 'bg-blue-500/20 border border-blue-500/30'
                        : 'bg-white/5 border border-white/10'
                    }`}
                  >
                    {/* Thinking (if Big Brain) */}
                    {message.thinking && (
                      <div className="mb-3">
                        <button
                          onClick={() => setShowThinking(showThinking === message.id ? null : message.id)}
                          className="text-xs text-purple-400 flex items-center gap-1 hover:text-purple-300"
                        >
                          <Brain className="w-3 h-3" />
                          {showThinking === message.id ? 'Hide' : 'Show'} thinking
                          <ChevronDown className={`w-3 h-3 transition ${showThinking === message.id ? 'rotate-180' : ''}`} />
                        </button>
                        {showThinking === message.id && (
                          <div className="mt-2 p-2 rounded-lg bg-purple-500/10 border border-purple-500/20 text-sm text-purple-200">
                            {message.thinking}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Content */}
                    <div className="prose prose-invert prose-sm max-w-none">
                      <p className="whitespace-pre-wrap">{message.content}</p>
                    </div>

                    {/* Metadata */}
                    {message.role === 'assistant' && (
                      <div className="mt-2 pt-2 border-t border-white/10 flex items-center justify-between">
                        <div className="flex items-center gap-2 text-xs text-slate-500">
                          {message.tier && (
                            <span className={message.tier === 'elle' ? 'text-cyan-400' : 'text-slate-400'}>
                              {message.tier === 'elle' ? 'Elle-72B' : 'Workhorse'}
                            </span>
                          )}
                          {message.latency_ms && (
                            <span>{(message.latency_ms / 1000).toFixed(1)}s</span>
                          )}
                          {message.research && (
                            <span className="text-green-400">
                              {message.research.sourcesFound} sources
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-1">
                          <button
                            onClick={() => copyMessage(message.content)}
                            className="p-1 hover:bg-white/10 rounded"
                          >
                            <Copy className="w-3 h-3" />
                          </button>
                          <button className="p-1 hover:bg-white/10 rounded text-green-400">
                            <ThumbsUp className="w-3 h-3" />
                          </button>
                          <button className="p-1 hover:bg-white/10 rounded text-red-400">
                            <ThumbsDown className="w-3 h-3" />
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}

            {/* Loading indicator */}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-white/5 border border-white/10 rounded-2xl px-4 py-3">
                  <div className="flex items-center gap-2 text-slate-400">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="text-sm">
                      {depth === 'thorough' ? 'Deep research in progress...' :
                       depth === 'moderate' ? 'Searching and synthesizing...' :
                       'Thinking...'}
                    </span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="border-t border-white/10 p-4 bg-black/20 backdrop-blur-xl">
            <div className="max-w-4xl mx-auto">
              {/* Mobile mode selector */}
              <div className="flex gap-2 mb-3 lg:hidden overflow-x-auto pb-2">
                {(Object.entries(MODE_CONFIG) as [StudyMode, typeof MODE_CONFIG.chat][]).map(([key, config]) => (
                  <button
                    key={key}
                    onClick={() => setMode(key)}
                    className={`px-3 py-1.5 rounded-lg text-sm whitespace-nowrap flex items-center gap-1.5 ${
                      mode === key
                        ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                        : 'bg-white/5 text-slate-400'
                    }`}
                  >
                    <config.icon className="w-3.5 h-3.5" />
                    {config.label}
                  </button>
                ))}
              </div>

              <div className="relative">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyPress}
                  placeholder={`Ask Elle anything... (${MODE_CONFIG[mode].description})`}
                  rows={1}
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 pr-12 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500/50 placeholder-slate-500"
                  style={{ maxHeight: '200px' }}
                />
                <button
                  onClick={sendMessage}
                  disabled={!input.trim() || isLoading}
                  className="absolute right-2 bottom-2 p-2 rounded-lg bg-blue-500 hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition"
                >
                  {isLoading ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Send className="w-5 h-5" />
                  )}
                </button>
              </div>

              {/* Status bar */}
              <div className="mt-2 flex items-center justify-between text-xs text-slate-500">
                <div className="flex items-center gap-3">
                  <span className="flex items-center gap-1">
                    <span className={`w-2 h-2 rounded-full ${unrestricted ? 'bg-orange-400' : 'bg-slate-600'}`} />
                    {unrestricted ? 'Unrestricted' : 'Restricted'}
                  </span>
                  <span className="flex items-center gap-1">
                    <span className={`w-2 h-2 rounded-full ${bigBrain ? 'bg-purple-400' : 'bg-slate-600'}`} />
                    {bigBrain ? 'Big Brain' : 'Standard'}
                  </span>
                  <span className="flex items-center gap-1">
                    {(() => {
                      const DepthIcon = DEPTH_CONFIG[depth].icon;
                      return DepthIcon ? <DepthIcon className="w-3 h-3" /> : null;
                    })()}
                    {DEPTH_CONFIG[depth].label}
                  </span>
                </div>
                <span>
                  Press Enter to send, Shift+Enter for newline
                </span>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
