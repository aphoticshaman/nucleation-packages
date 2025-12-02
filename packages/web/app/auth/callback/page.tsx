'use client';

import { useEffect, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { supabase } from '@/lib/supabase';

function AuthCallbackHandler() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const redirect = searchParams.get('redirect') || '/app';

  useEffect(() => {
    const handleCallback = async () => {
      // For implicit flow, Supabase client automatically picks up the hash fragment
      // We just need to wait for the session to be established
      const { data: { session }, error } = await supabase.auth.getSession();

      if (error) {
        console.error('Auth callback error:', error);
        router.push('/login?error=auth_failed');
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
    };

    void handleCallback();
  }, [router, redirect]);

  return (
    <div className="text-center">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mx-auto mb-4"></div>
      <p className="text-slate-400">Completing sign in...</p>
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
