'use client';

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-white flex items-center justify-center px-4">
        <div className="text-center">
          {/* Error Display */}
          <div className="mb-8">
            <h1 className="text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-red-400 to-orange-400">
              Error
            </h1>
            <div className="mt-2 text-xl text-slate-400">Something went wrong</div>
          </div>

          {/* Description */}
          <p className="text-slate-500 mb-8 max-w-md mx-auto">
            An unexpected error occurred. Our team has been notified.
          </p>

          {/* Error digest for debugging */}
          {error.digest && (
            <p className="text-xs text-slate-600 mb-4 font-mono">
              Error ID: {error.digest}
            </p>
          )}

          {/* Actions */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={() => reset()}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-500 transition-colors"
            >
              Try Again
            </button>
            <a
              href="/"
              className="px-6 py-3 bg-slate-800 text-slate-300 rounded-lg font-medium hover:bg-slate-700 transition-colors"
            >
              Go Home
            </a>
          </div>
        </div>
      </body>
    </html>
  );
}
