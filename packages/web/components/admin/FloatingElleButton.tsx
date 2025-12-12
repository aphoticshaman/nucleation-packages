'use client';

import { useState, useEffect } from 'react';
import { createBrowserClient } from '@supabase/ssr';
import { useStudyBook } from '@/lib/study/StudyBookContext';
import { Maximize2 } from 'lucide-react';

export function FloatingElleButton() {
  const [isAdmin, setIsAdmin] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const { isOpen, isMinimized, toggle, maximize } = useStudyBook();

  useEffect(() => {
    const checkAdmin = async () => {
      const supabase = createBrowserClient(
        process.env.NEXT_PUBLIC_SUPABASE_URL!,
        process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
      );

      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        setIsAdmin(false);
        return;
      }

      const { data: profile } = await supabase
        .from('profiles')
        .select('role')
        .eq('id', user.id)
        .single();

      setIsAdmin(profile?.role === 'admin');
    };

    void checkAdmin();
  }, []);

  if (!isAdmin) return null;

  // If overlay is open and NOT minimized, hide the button
  if (isOpen && !isMinimized) return null;

  // Show minimized indicator if Study Book is minimized
  if (isOpen && isMinimized) {
    return (
      <button
        onClick={maximize}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        className={`
          fixed bottom-6 right-6 z-[9999]
          flex items-center gap-2
          px-4 py-3 rounded-full
          bg-gradient-to-r from-purple-600 to-blue-600
          hover:from-purple-500 hover:to-blue-500
          text-white font-medium
          shadow-lg shadow-purple-500/25
          hover:shadow-xl hover:shadow-purple-500/40
          transition-all duration-300 ease-out
          hover:scale-105
          border border-purple-400/30
          backdrop-blur-sm
          animate-pulse
        `}
        title="Restore Study Book"
      >
        <Maximize2 className="w-5 h-5" />
        <span
          className={`
            overflow-hidden whitespace-nowrap
            transition-all duration-300 ease-out
            ${isHovered ? 'max-w-40 opacity-100' : 'max-w-0 opacity-0'}
          `}
        >
          Restore Elle
        </span>
      </button>
    );
  }

  // Default state - open Study Book
  return (
    <button
      onClick={toggle}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={`
        fixed bottom-6 right-6 z-[9999]
        flex items-center gap-2
        px-4 py-3 rounded-full
        bg-gradient-to-r from-cyan-600 to-blue-600
        hover:from-cyan-500 hover:to-blue-500
        text-white font-medium
        shadow-lg shadow-cyan-500/25
        hover:shadow-xl hover:shadow-cyan-500/40
        transition-all duration-300 ease-out
        hover:scale-105
        border border-cyan-400/30
        backdrop-blur-sm
        group
      `}
      title="Chat with Elle"
    >
      <span className="text-xl">✨</span>
      <span
        className={`
          overflow-hidden whitespace-nowrap
          transition-all duration-300 ease-out
          ${isHovered ? 'max-w-32 opacity-100' : 'max-w-0 opacity-0'}
        `}
      >
        Chat with Elle
      </span>
      {/* Pulse ring animation */}
      <span className="absolute inset-0 rounded-full bg-cyan-400/20 animate-ping opacity-75 pointer-events-none" />
    </button>
  );
}
