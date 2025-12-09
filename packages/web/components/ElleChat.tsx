'use client';

import { useState, useEffect, useRef } from 'react';
import { Sparkles, X, Send, Loader2, Minimize2 } from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'elle';
  content: string;
  timestamp: Date;
  ticketCreated?: boolean;
}

interface ElleChatProps {
  /** Position of the button */
  position?: 'bottom-right' | 'bottom-left';
  /** Additional class names */
  className?: string;
}

export default function ElleChat({ position = 'bottom-right', className = '' }: ElleChatProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Add welcome message when chat opens for first time
  useEffect(() => {
    if (isOpen && messages.length === 0) {
      setMessages([
        {
          id: 'welcome',
          role: 'elle',
          content: "Hey! I'm Elle. I won't just answer - I'll show you how to find answers yourself. Ask me anything and I'll point you in the right direction so you won't need me next time!",
          timestamp: new Date(),
        },
      ]);
    }
  }, [isOpen, messages.length]);

  // Handle sending a message
  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/elle/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          message: userMessage.content,
          history: messages.slice(-10).map((m) => ({
            role: m.role === 'elle' ? 'assistant' : 'user',
            content: m.content,
          })),
          pageUrl: window.location.href,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to get response');
      }

      const elleMessage: Message = {
        id: `elle-${Date.now()}`,
        role: 'elle',
        content: data.response,
        timestamp: new Date(),
        ticketCreated: data.ticketCreated,
      };

      setMessages((prev) => [...prev, elleMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'elle',
        content: "I'm having trouble connecting right now. Please try again in a moment, or you can reach us at support@latticeforge.ai.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle key press
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Position classes
  const positionClasses = position === 'bottom-right'
    ? 'right-4 bottom-4'
    : 'left-4 bottom-4';

  return (
    <>
      {/* Floating Button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className={`fixed ${positionClasses} z-40 p-4 bg-gradient-to-br from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white rounded-full shadow-lg shadow-cyan-500/30 hover:shadow-cyan-500/50 transition-all hover:scale-110 ${className}`}
          aria-label="Chat with Elle"
        >
          <Sparkles className="w-6 h-6" />
        </button>
      )}

      {/* Chat Window */}
      {isOpen && (
        <div
          className={`fixed ${positionClasses} z-50 flex flex-col bg-slate-900 border border-white/10 rounded-2xl shadow-2xl overflow-hidden transition-all duration-200 ${
            isMinimized ? 'w-72 h-14' : 'w-96 h-[500px] max-h-[80vh]'
          }`}
        >
          {/* Header */}
          <div
            className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-cyan-600/20 to-blue-600/20 border-b border-white/10 cursor-pointer"
            onClick={() => isMinimized && setIsMinimized(false)}
          >
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
                <Sparkles className="w-4 h-4 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-white text-sm">Elle</h3>
                <p className="text-xs text-slate-400">LatticeForge Assistant</p>
              </div>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={(e) => { e.stopPropagation(); setIsMinimized(!isMinimized); }}
                className="p-1.5 text-slate-400 hover:text-white transition-colors rounded-lg hover:bg-white/10"
              >
                <Minimize2 className="w-4 h-4" />
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); setIsOpen(false); }}
                className="p-1.5 text-slate-400 hover:text-white transition-colors rounded-lg hover:bg-white/10"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Messages */}
          {!isMinimized && (
            <>
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[85%] rounded-2xl px-4 py-2.5 ${
                        msg.role === 'user'
                          ? 'bg-cyan-600 text-white'
                          : 'bg-slate-800 text-slate-200 border border-white/5'
                      }`}
                    >
                      <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                      {msg.ticketCreated && (
                        <p className="text-xs text-cyan-300 mt-2 pt-2 border-t border-white/10">
                          ✓ Ticket created - we'll follow up soon
                        </p>
                      )}
                    </div>
                  </div>
                ))}

                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-slate-800 text-slate-400 rounded-2xl px-4 py-3 border border-white/5">
                      <div className="flex items-center gap-2">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span className="text-sm">Elle is typing...</span>
                      </div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              {/* Input */}
              <div className="p-3 border-t border-white/10 bg-slate-900/50">
                <div className="flex items-end gap-2">
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyPress}
                    placeholder="How do I... / What is... / Where can I..."
                    rows={1}
                    className="flex-1 px-4 py-2.5 bg-slate-800 border border-white/10 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 resize-none text-sm max-h-24 overflow-y-auto"
                    style={{ minHeight: '42px' }}
                  />
                  <button
                    onClick={handleSend}
                    disabled={!input.trim() || isLoading}
                    className="p-2.5 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded-xl transition-colors"
                  >
                    <Send className="w-5 h-5" />
                  </button>
                </div>
                <p className="text-xs text-slate-500 mt-2 text-center">
                  Learn to explore on your own
                </p>
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
}
