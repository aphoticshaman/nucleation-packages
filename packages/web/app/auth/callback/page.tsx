'use client';

import { useEffect, Suspense, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { supabase } from '@/lib/supabase';

function AuthCallbackHandler() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const redirect = searchParams.get('redirect') || '/app';
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const handleCallback = async () => {
      try {
        // Check for error in URL params (OAuth provider errors)
        const errorParam = searchParams.get('error');
        const errorDescription = searchParams.get('error_description');
        if (errorParam) {
          console.error('OAuth error:', errorParam, errorDescription);
          router.push(`/login?error=${encodeURIComponent(errorDescription || errorParam)}`);
          return;
        }

        // For PKCE flow, check for code in URL and exchange it
        const code = searchParams.get('code');
        if (code) {
          // Exchange the code for a session
          const { data, error: exchangeError } = await supabase.auth.exchangeCodeForSession(code);
          if (exchangeError) {
            console.error('Code exchange error:', exchangeError);
            setError(exchangeError.message);
            setTimeout(() => router.push('/login?error=code_exchange_failed'), 2000);
            return;
          }
          if (data.session) {
            router.push(redirect);
            return;
          }
        }

        // For implicit flow, Supabase client automatically picks up the hash fragment
        // We just need to wait for the session to be established
        const { data: { session }, error: sessionError } = await supabase.auth.getSession();

        if (sessionError) {
          console.error('Auth callback error:', sessionError);
          setError(sessionError.message);
          setTimeout(() => router.push('/login?error=auth_failed'), 2000);
          return;
        }

        if (session) {
          router.push(redirect);
        } else {
          // If no session yet, listen for auth state change
          const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
            if (event === 'SIGNED_IN' && session) {
              subscription.unsubscribe();
              router.push(redirect);
            }
          });

          // Timeout fallback - if nothing happens in 5 seconds, redirect to login
          setTimeout(() => {
            subscription.unsubscribe();
            router.push('/login?error=timeout');
          }, 5000);
        }
      } catch (err) {
        console.error('Unexpected auth callback error:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        setTimeout(() => router.push('/login?error=unexpected'), 2000);
      }
    };

    void handleCallback();
  }, [router, redirect, searchParams]);

  return (
    <div className="text-center">
      {error ? (
        <>
          <div className="text-red-500 text-4xl mb-4">!</div>
          <p className="text-red-400 mb-2">Authentication Error</p>
          <p className="text-slate-500 text-sm">{error}</p>
          <p className="text-slate-500 text-xs mt-2">Redirecting to login...</p>
        </>
      ) : (
        <>
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-slate-400">Completing sign in...</p>
        </>
      )}
    </div>
  );
}

function CallbackFallback() {
  return (
    <div className="text-center">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mx-auto mb-4"></div>
      <p className="text-slate-400">Loading...</p>
    </div>
  );
}

export default function AuthCallbackPage() {
  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center">
      <Suspense fallback={<CallbackFallback />}>
        <AuthCallbackHandler />
      </Suspense>
    </div>
  );
}
