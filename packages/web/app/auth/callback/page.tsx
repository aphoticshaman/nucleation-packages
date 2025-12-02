'use client';

import { useEffect, Suspense, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { supabase } from '@/lib/supabase';

function AuthCallbackHandler() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const redirect = searchParams.get('redirect') || '/app';
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('Completing sign in...');

  useEffect(() => {
    const ensureProfileAndRedirect = async () => {
      setStatus('Setting up your account...');
      try {
        await fetch('/api/auth/ensure-profile', {
          method: 'POST',
          credentials: 'include',
        });
      } catch {
        // Continue anyway
      }
      router.push(redirect);
    };

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

        // First, check if we already have a session (Supabase might have handled the exchange)
        setStatus('Checking authentication...');
        const { data: { session: existingSession } } = await supabase.auth.getSession();

        if (existingSession) {
          // Session already exists - just ensure profile and redirect
          await ensureProfileAndRedirect();
          return;
        }

        // Try to exchange code if present (PKCE flow)
        const code = searchParams.get('code');
        if (code) {
          setStatus('Completing sign in...');
          try {
            const { data, error: exchangeError } = await supabase.auth.exchangeCodeForSession(code);
            if (!exchangeError && data.session) {
              await ensureProfileAndRedirect();
              return;
            }
            // If exchange fails (e.g., verifier missing), check session again
            // The session might have been established via cookies from Supabase auth server
            if (exchangeError) {
              console.warn('Code exchange failed, checking for existing session:', exchangeError.message);
              const { data: { session: retrySession } } = await supabase.auth.getSession();
              if (retrySession) {
                await ensureProfileAndRedirect();
                return;
              }
            }
          } catch (err) {
            console.warn('Code exchange threw, checking session:', err);
            const { data: { session: fallbackSession } } = await supabase.auth.getSession();
            if (fallbackSession) {
              await ensureProfileAndRedirect();
              return;
            }
          }
        }

        // Check hash fragment for implicit flow
        if (window.location.hash) {
          setStatus('Processing authentication...');
          // Give Supabase time to process the hash
          await new Promise(resolve => setTimeout(resolve, 500));
          const { data: { session: hashSession } } = await supabase.auth.getSession();
          if (hashSession) {
            await ensureProfileAndRedirect();
            return;
          }
        }

        // Last resort: listen for auth state change
        setStatus('Waiting for authentication...');
        const { data: { subscription } } = supabase.auth.onAuthStateChange(async (event, session) => {
          if (event === 'SIGNED_IN' && session) {
            subscription.unsubscribe();
            await ensureProfileAndRedirect();
          }
        });

        // Timeout fallback
        setTimeout(() => {
          subscription.unsubscribe();
          // One final check before giving up
          supabase.auth.getSession().then(({ data: { session: finalSession } }) => {
            if (finalSession) {
              ensureProfileAndRedirect();
            } else {
              router.push('/login?error=timeout');
            }
          });
        }, 5000);
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
          <p className="text-slate-400">{status}</p>
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
