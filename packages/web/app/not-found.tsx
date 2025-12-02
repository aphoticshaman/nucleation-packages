import Link from 'next/link';

export default function NotFound() {
  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center px-4">
      <div className="text-center">
        {/* 404 Display */}
        <div className="mb-8">
          <h1 className="text-8xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">
            404
          </h1>
          <div className="mt-2 text-xl text-slate-400">Page not found</div>
        </div>

        {/* Description */}
        <p className="text-slate-500 mb-8 max-w-md mx-auto">
          The page you&apos;re looking for doesn&apos;t exist or has been moved.
          Let&apos;s get you back on track.
        </p>

        {/* Actions */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link
            href="/"
            className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-500 transition-colors"
          >
            Go Home
          </Link>
          <Link
            href="/login"
            className="px-6 py-3 bg-slate-800 text-slate-300 rounded-lg font-medium hover:bg-slate-700 transition-colors"
          >
            Sign In
          </Link>
        </div>

        {/* Decorative element */}
        <div className="mt-16 flex justify-center gap-2">
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              className="w-2 h-2 rounded-full bg-slate-800"
              style={{
                animation: `pulse 2s ease-in-out ${i * 0.2}s infinite`,
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
