import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { Dashboard } from './pages/Dashboard';
import { AuthGate } from './components/AuthGate';

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [apiKey, setApiKey] = useState<string | null>(null);

  useEffect(() => {
    // Check for stored API key
    const storedKey = localStorage.getItem('lf_api_key');
    if (storedKey) {
      setApiKey(storedKey);
      setIsAuthenticated(true);
    }
  }, []);

  const handleAuthenticate = (key: string) => {
    localStorage.setItem('lf_api_key', key);
    setApiKey(key);
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem('lf_api_key');
    setApiKey(null);
    setIsAuthenticated(false);
  };

  if (!isAuthenticated) {
    return <AuthGate onAuthenticate={handleAuthenticate} />;
  }

  return (
    <div className="min-h-screen bg-surface-900 bg-grid">
      <Header onLogout={handleLogout} />
      <div className="flex">
        <Sidebar />
        <main className="flex-1 p-6 ml-64 mt-16">
          <Dashboard apiKey={apiKey!} />
        </main>
      </div>
    </div>
  );
}
