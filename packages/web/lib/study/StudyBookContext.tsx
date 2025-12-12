'use client';

import { createContext, useContext, useState, useCallback, ReactNode } from 'react';

interface StudyBookContextType {
  isOpen: boolean;
  isMinimized: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
  minimize: () => void;
  maximize: () => void;
}

const StudyBookContext = createContext<StudyBookContextType | null>(null);

export function StudyBookProvider({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);

  const open = useCallback(() => {
    setIsOpen(true);
    setIsMinimized(false);
  }, []);

  const close = useCallback(() => {
    setIsOpen(false);
    setIsMinimized(false);
  }, []);

  const toggle = useCallback(() => {
    if (isOpen && !isMinimized) {
      // If open and maximized, minimize
      setIsMinimized(true);
    } else if (isOpen && isMinimized) {
      // If minimized, maximize
      setIsMinimized(false);
    } else {
      // If closed, open
      setIsOpen(true);
      setIsMinimized(false);
    }
  }, [isOpen, isMinimized]);

  const minimize = useCallback(() => {
    setIsMinimized(true);
  }, []);

  const maximize = useCallback(() => {
    setIsMinimized(false);
  }, []);

  return (
    <StudyBookContext.Provider
      value={{ isOpen, isMinimized, open, close, toggle, minimize, maximize }}
    >
      {children}
    </StudyBookContext.Provider>
  );
}

export function useStudyBook() {
  const context = useContext(StudyBookContext);
  if (!context) {
    throw new Error('useStudyBook must be used within a StudyBookProvider');
  }
  return context;
}
