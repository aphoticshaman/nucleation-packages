'use client';

import { useState, useEffect } from 'react';
import { MessageSquarePlus, Bug, Lightbulb, HelpCircle, FileText, X, Send, CheckCircle, Loader2, Sparkles } from 'lucide-react';

// Feedback types - Elle handles questions directly
const FEEDBACK_TYPES = [
  { id: 'bug', label: 'Bug Report', icon: Bug, color: 'text-red-400', bgColor: 'bg-red-500/20' },
  { id: 'idea', label: 'Feature Idea', icon: Lightbulb, color: 'text-amber-400', bgColor: 'bg-amber-500/20' },
  { id: 'question', label: 'Ask Elle', icon: Sparkles, color: 'text-cyan-400', bgColor: 'bg-cyan-500/20', elleMode: true },
  { id: 'other', label: 'Other Feedback', icon: FileText, color: 'text-slate-400', bgColor: 'bg-slate-500/20' },
] as const;

type FeedbackType = typeof FEEDBACK_TYPES[number]['id'];

interface FeedbackButtonProps {
  /** Position of the button */
  position?: 'bottom-right' | 'bottom-left';
  /** Additional class names */
  className?: string;
}

export default function FeedbackButton({ position = 'bottom-right', className = '' }: FeedbackButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [step, setStep] = useState<'type' | 'form' | 'success' | 'elle'>('type');
  const [selectedType, setSelectedType] = useState<FeedbackType | null>(null);
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elleResponse, setElleResponse] = useState<string | null>(null);

  // Reset state when modal closes
  useEffect(() => {
    if (!isOpen) {
      // Delay reset to allow close animation
      const timer = setTimeout(() => {
        setStep('type');
        setSelectedType(null);
        setTitle('');
        setDescription('');
        setError(null);
        setElleResponse(null);
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [isOpen]);

  // Handle type selection
  const handleTypeSelect = (type: FeedbackType) => {
    setSelectedType(type);
    // Questions go to Elle mode, others to regular form
    setStep(type === 'question' ? 'elle' : 'form');
  };

  // Handle Elle question submission
  const handleElleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!description.trim()) return;

    setIsSubmitting(true);
    setError(null);

    try {
      // Call Elle API
      const response = await fetch('/api/elle/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          question: description.trim(),
          pageUrl: window.location.href,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Elle is currently unavailable');
      }

      setElleResponse(data.response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedType || !title.trim() || !description.trim()) return;

    setIsSubmitting(true);
    setError(null);

    try {
      const response = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          type: selectedType,
          title: title.trim(),
          description: description.trim(),
          pageUrl: window.location.href,
          userAgent: navigator.userAgent,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to submit feedback');
      }

      setStep('success');

      // Auto-close after success
      setTimeout(() => {
        setIsOpen(false);
      }, 2500);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Position classes
  const positionClasses = position === 'bottom-right'
    ? 'right-4 bottom-4'
    : 'left-4 bottom-4';

  return (
    <>
      {/* Floating Button */}
      <button
        onClick={() => setIsOpen(true)}
        className={`fixed ${positionClasses} z-40 p-3 bg-cyan-600 hover:bg-cyan-500 text-white rounded-full shadow-lg shadow-cyan-500/30 hover:shadow-cyan-500/50 transition-all hover:scale-110 ${className}`}
        aria-label="Send feedback"
      >
        <MessageSquarePlus className="w-6 h-6" />
      </button>

      {/* Modal Overlay */}
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => setIsOpen(false)}
          />

          {/* Modal */}
          <div className="relative w-full max-w-md bg-slate-900 border border-white/10 rounded-2xl shadow-2xl overflow-hidden animate-in fade-in zoom-in duration-200">
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
              <h2 className="text-lg font-semibold text-white">
                {step === 'type' && 'Send Feedback'}
                {step === 'form' && FEEDBACK_TYPES.find(t => t.id === selectedType)?.label}
                {step === 'elle' && (
                  <span className="flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-cyan-400" />
                    Ask Elle
                  </span>
                )}
                {step === 'success' && 'Thank You!'}
              </h2>
              <button
                onClick={() => setIsOpen(false)}
                className="p-1 text-slate-400 hover:text-white transition-colors rounded-lg hover:bg-white/10"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Content */}
            <div className="p-6">
              {/* Step 1: Select Type */}
              {step === 'type' && (
                <div className="space-y-3">
                  <p className="text-sm text-slate-400 mb-4">
                    What would you like to share with us?
                  </p>
                  {FEEDBACK_TYPES.map((type) => {
                    const Icon = type.icon;
                    return (
                      <button
                        key={type.id}
                        onClick={() => handleTypeSelect(type.id)}
                        className={`w-full flex items-center gap-4 p-4 rounded-xl border border-white/10 hover:border-white/20 ${type.bgColor} transition-all hover:scale-[1.02]`}
                      >
                        <Icon className={`w-6 h-6 ${type.color}`} />
                        <span className="text-white font-medium">{type.label}</span>
                      </button>
                    );
                  })}
                </div>
              )}

              {/* Step 2: Form */}
              {step === 'form' && (
                <form onSubmit={handleSubmit} className="space-y-4">
                  {/* Back button */}
                  <button
                    type="button"
                    onClick={() => setStep('type')}
                    className="text-sm text-slate-400 hover:text-white transition-colors"
                  >
                    &larr; Change type
                  </button>

                  {/* Title */}
                  <div>
                    <label htmlFor="feedback-title" className="block text-sm font-medium text-slate-300 mb-2">
                      Title
                    </label>
                    <input
                      id="feedback-title"
                      type="text"
                      value={title}
                      onChange={(e) => setTitle(e.target.value)}
                      placeholder={selectedType === 'bug' ? 'Brief description of the issue' : 'Brief summary'}
                      maxLength={200}
                      required
                      className="w-full px-4 py-3 bg-slate-800 border border-white/10 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50"
                    />
                    <p className="text-xs text-slate-500 mt-1 text-right">{title.length}/200</p>
                  </div>

                  {/* Description */}
                  <div>
                    <label htmlFor="feedback-description" className="block text-sm font-medium text-slate-300 mb-2">
                      Details
                    </label>
                    <textarea
                      id="feedback-description"
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      placeholder={
                        selectedType === 'bug'
                          ? 'Steps to reproduce, expected behavior, actual behavior...'
                          : 'Tell us more about your feedback...'
                      }
                      rows={4}
                      maxLength={5000}
                      required
                      className="w-full px-4 py-3 bg-slate-800 border border-white/10 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50 resize-none"
                    />
                    <p className="text-xs text-slate-500 mt-1 text-right">{description.length}/5000</p>
                  </div>

                  {/* Error */}
                  {error && (
                    <div className="p-3 bg-red-500/20 border border-red-500/30 rounded-lg text-red-300 text-sm">
                      {error}
                    </div>
                  )}

                  {/* Submit */}
                  <button
                    type="submit"
                    disabled={isSubmitting || !title.trim() || !description.trim()}
                    className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 disabled:cursor-not-allowed text-white font-medium rounded-xl transition-colors"
                  >
                    {isSubmitting ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Sending...
                      </>
                    ) : (
                      <>
                        <Send className="w-5 h-5" />
                        Send Feedback
                      </>
                    )}
                  </button>

                  <p className="text-xs text-slate-500 text-center">
                    Your feedback helps us improve LatticeForge
                  </p>
                </form>
              )}

              {/* Elle Mode: Ask a question */}
              {step === 'elle' && (
                <div className="space-y-4">
                  {/* Back button */}
                  <button
                    type="button"
                    onClick={() => { setStep('type'); setElleResponse(null); }}
                    className="text-sm text-slate-400 hover:text-white transition-colors"
                  >
                    &larr; Back
                  </button>

                  {!elleResponse ? (
                    <form onSubmit={handleElleSubmit} className="space-y-4">
                      <div className="flex items-center gap-3 p-3 bg-cyan-500/10 border border-cyan-500/20 rounded-xl">
                        <Sparkles className="w-5 h-5 text-cyan-400 flex-shrink-0" />
                        <p className="text-sm text-cyan-200">
                          Hi! I'm Elle, your LatticeForge assistant. Ask me anything about using the platform.
                        </p>
                      </div>

                      <div>
                        <label htmlFor="elle-question" className="block text-sm font-medium text-slate-300 mb-2">
                          Your Question
                        </label>
                        <textarea
                          id="elle-question"
                          value={description}
                          onChange={(e) => setDescription(e.target.value)}
                          placeholder="How do I interpret the risk scores? What does NSM mean?"
                          rows={4}
                          maxLength={2000}
                          required
                          className="w-full px-4 py-3 bg-slate-800 border border-white/10 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50 resize-none"
                        />
                      </div>

                      {error && (
                        <div className="p-3 bg-red-500/20 border border-red-500/30 rounded-lg text-red-300 text-sm">
                          {error}
                        </div>
                      )}

                      <button
                        type="submit"
                        disabled={isSubmitting || !description.trim()}
                        className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 disabled:cursor-not-allowed text-white font-medium rounded-xl transition-colors"
                      >
                        {isSubmitting ? (
                          <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            Elle is thinking...
                          </>
                        ) : (
                          <>
                            <Sparkles className="w-5 h-5" />
                            Ask Elle
                          </>
                        )}
                      </button>
                    </form>
                  ) : (
                    <div className="space-y-4">
                      {/* User's question */}
                      <div className="p-3 bg-slate-800 rounded-xl">
                        <p className="text-xs text-slate-400 mb-1">You asked:</p>
                        <p className="text-white">{description}</p>
                      </div>

                      {/* Elle's response */}
                      <div className="p-4 bg-cyan-500/10 border border-cyan-500/20 rounded-xl">
                        <div className="flex items-center gap-2 mb-2">
                          <Sparkles className="w-4 h-4 text-cyan-400" />
                          <span className="text-sm font-medium text-cyan-300">Elle</span>
                        </div>
                        <p className="text-slate-200 whitespace-pre-wrap">{elleResponse}</p>
                      </div>

                      {/* Actions */}
                      <div className="flex gap-3">
                        <button
                          onClick={() => { setDescription(''); setElleResponse(null); }}
                          className="flex-1 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-xl transition-colors text-sm"
                        >
                          Ask Another
                        </button>
                        <button
                          onClick={() => setIsOpen(false)}
                          className="flex-1 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl transition-colors text-sm"
                        >
                          Done
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Step 3: Success */}
              {step === 'success' && (
                <div className="text-center py-6">
                  <div className="w-16 h-16 mx-auto mb-4 bg-emerald-500/20 rounded-full flex items-center justify-center">
                    <CheckCircle className="w-8 h-8 text-emerald-400" />
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">
                    Feedback Received!
                  </h3>
                  <p className="text-slate-400 text-sm">
                    We appreciate you taking the time to help us improve.
                    {selectedType === 'bug' && " We'll look into this issue."}
                    {selectedType === 'idea' && " We'll consider your suggestion."}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
