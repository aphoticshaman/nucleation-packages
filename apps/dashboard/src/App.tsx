import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { Dashboard } from './pages/Dashboard';
import { Settings } from './pages/Settings';
import { AuthGate } from './components/AuthGate';
import { supabase, auth } from './lib/supabase';
import type { Session } from '@supabase/supabase-js';

export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setIsLoading(false);
    });

    // Listen for auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
    });

    return () => subscription.unsubscribe();
  }, []);

  const handleAuthenticate = () => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
    });
  };

  const handleLogout = async () => {
    await auth.signOut();
    setSession(null);
  };

  // Show loading state
  if (isLoading) {
    return (
      <div className="min-h-screen bg-surface-900 flex items-center justify-center">
        <div className="text-lattice-400">
          <svg className="animate-spin w-8 h-8" viewBox="0 0 24 24">
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
              fill="none"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        </div>
      </div>
    );
  }

  // Show auth gate if not authenticated
  if (!session) {
    return <AuthGate onAuthenticate={handleAuthenticate} />;
  }

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-surface-900 bg-grid">
        <Header onLogout={handleLogout} user={session.user} />
        <div className="flex">
          <Sidebar />
          <main className="flex-1 p-6 ml-64 mt-16">
            <Routes>
              <Route path="/" element={<Dashboard session={session} />} />
              <Route path="/settings" element={<Settings session={session} />} />
              <Route
                path="/signals"
                element={
                  <ComingSoon
                    title="Signals"
                    description="Real-time signal monitoring and analysis"
                  />
                }
              />
              <Route
                path="/sources"
                element={
                  <ComingSoon
                    title="Data Sources"
                    description="Configure and manage your data integrations"
                  />
                }
              />
              <Route
                path="/detection"
                element={
                  <ComingSoon
                    title="Anomaly Detection"
                    description="AI-powered anomaly detection and alerts"
                  />
                }
              />
              <Route
                path="/trace"
                element={
                  <ComingSoon
                    title="Audit Trace"
                    description="Complete audit trail of all system activity"
                  />
                }
              />
              <Route
                path="/docs"
                element={
                  <ComingSoon
                    title="API Documentation"
                    description="Complete API reference and guides"
                  />
                }
              />
              <Route
                path="/auth/callback"
                element={<AuthCallback onAuthenticate={handleAuthenticate} />}
              />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}

// Placeholder for pages not yet built
function ComingSoon({ title, description }: { title: string; description: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-[60vh] text-center">
      <div className="w-16 h-16 rounded-2xl bg-surface-800 border border-surface-600 flex items-center justify-center mb-4">
        <svg
          className="w-8 h-8 text-lattice-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"
          />
        </svg>
      </div>
      <h1 className="text-2xl font-bold text-white mb-2">{title}</h1>
      <p className="text-surface-400 mb-6">{description}</p>
      <span className="px-3 py-1 rounded-full bg-lattice-500/10 text-lattice-400 text-xs font-medium border border-lattice-500/20">
        Coming Soon
      </span>
    </div>
  );
}

// Handle OAuth callback
function AuthCallback({ onAuthenticate }: { onAuthenticate: () => void }) {
  useEffect(() => {
    onAuthenticate();
  }, [onAuthenticate]);

  return (
    <div className="flex items-center justify-center h-[60vh]">
      <div className="text-lattice-400">
        <svg className="animate-spin w-8 h-8" viewBox="0 0 24 24">
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
            fill="none"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
      </div>
    </div>
  );
}
