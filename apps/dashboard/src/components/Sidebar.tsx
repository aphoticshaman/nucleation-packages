import {
  DashboardIcon,
  SignalsIcon,
  SourcesIcon,
  DetectionIcon,
  TraceIcon,
  SettingsIcon,
  DocsIcon,
} from './Icons';

const navigation = [
  { name: 'Dashboard', icon: DashboardIcon, href: '#', current: true },
  { name: 'Signals', icon: SignalsIcon, href: '#signals', current: false },
  { name: 'Sources', icon: SourcesIcon, href: '#sources', current: false },
  { name: 'Detection', icon: DetectionIcon, href: '#detection', current: false },
  { name: 'Audit Trace', icon: TraceIcon, href: '#trace', current: false },
];

const secondaryNav = [
  { name: 'API Docs', icon: DocsIcon, href: '#docs' },
  { name: 'Settings', icon: SettingsIcon, href: '#settings' },
];

export function Sidebar() {
  return (
    <aside className="fixed left-0 top-16 bottom-0 w-64 bg-surface-800/50 border-r border-surface-600/50 backdrop-blur-xl">
      <nav className="flex flex-col h-full p-4">
        {/* Main Navigation */}
        <div className="space-y-1">
          <p className="px-3 text-[10px] font-semibold text-lattice-500 uppercase tracking-wider mb-2">
            Navigation
          </p>
          {navigation.map((item) => (
            <a
              key={item.name}
              href={item.href}
              className={`
                flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all
                ${
                  item.current
                    ? 'bg-lattice-500/10 text-lattice-300 border border-lattice-500/20'
                    : 'text-surface-400 hover:text-lattice-300 hover:bg-surface-700/50'
                }
              `}
            >
              <item.icon className="w-5 h-5" />
              {item.name}
              {item.current && (
                <span className="ml-auto w-1.5 h-1.5 rounded-full bg-lattice-400" />
              )}
            </a>
          ))}
        </div>

        {/* Divider */}
        <div className="my-6 border-t border-surface-600/50" />

        {/* Secondary Navigation */}
        <div className="space-y-1">
          <p className="px-3 text-[10px] font-semibold text-lattice-500 uppercase tracking-wider mb-2">
            Resources
          </p>
          {secondaryNav.map((item) => (
            <a
              key={item.name}
              href={item.href}
              className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium text-surface-400 hover:text-lattice-300 hover:bg-surface-700/50 transition-all"
            >
              <item.icon className="w-5 h-5" />
              {item.name}
            </a>
          ))}
        </div>

        {/* Bottom - Usage */}
        <div className="mt-auto">
          <div className="glass-card p-4">
            <p className="text-xs font-semibold text-lattice-400 mb-2">API Usage</p>
            <div className="space-y-2">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-surface-400">Requests Today</span>
                  <span className="text-lattice-300 font-mono">1,247 / 50,000</span>
                </div>
                <div className="h-1.5 bg-surface-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-lattice-500 to-crystal-500 rounded-full"
                    style={{ width: '2.5%' }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-surface-400">Rate Limit</span>
                  <span className="text-emerald-400 font-mono">298 / 300</span>
                </div>
                <div className="h-1.5 bg-surface-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 rounded-full"
                    style={{ width: '99%' }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </nav>
    </aside>
  );
}
