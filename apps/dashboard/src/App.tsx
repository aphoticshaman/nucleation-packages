import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { Dashboard } from './pages/Dashboard';
import { AuthGate } from './components/AuthGate';
import { supabase, auth } from './lib/supabase';
import type { Session } from '@supabase/supabase-js';

export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [apiKey, setApiKey] = useState<string | null>(null);

  useEffect(() => {
    // Check for stored API key
    const storedKey = localStorage.getItem('lf_api_key');
    if (storedKey) {
      setApiKey(storedKey);
    }

    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setIsLoading(false);
    });

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setSession(session);
        if (!session) {
          // Clear API key on logout
          localStorage.removeItem('lf_api_key');
          setApiKey(null);
        }
      }
    );

    return () => subscription.unsubscribe();
  }, []);

  const handleAuthenticate = () => {
    // Re-check session or API key
    const storedKey = localStorage.getItem('lf_api_key');
    if (storedKey) {
      setApiKey(storedKey);
    }
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
    });
  };

  const handleLogout = async () => {
    await auth.signOut();
    setSession(null);
    setApiKey(null);
  };

  // Show loading state
  if (isLoading) {
    return (
      <div className="min-h-screen bg-surface-900 flex items-center justify-center">
        <div className="text-lattice-400">
          <svg className="animate-spin w-8 h-8" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
        </div>
      </div>
    );
  }

  // Show auth gate if not authenticated
  const isAuthenticated = session || apiKey;

  if (!isAuthenticated) {
    return <AuthGate onAuthenticate={handleAuthenticate} />;
  }

  return (
    <div className="min-h-screen bg-surface-900 bg-grid">
      <Header onLogout={handleLogout} user={session?.user} />
      <div className="flex">
        <Sidebar />
        <main className="flex-1 p-6 ml-64 mt-16">
          <Dashboard apiKey={apiKey} session={session} />
        </main>
      </div>
    </div>
  );
}
