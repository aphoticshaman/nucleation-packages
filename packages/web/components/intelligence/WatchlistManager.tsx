'use client';

import { useState, useMemo, useCallback } from 'react';
import { User, Building2, MapPin, Calendar, MessageSquare, Package, Users, Trash2, X, ArrowUp, ArrowDown, ArrowRight, ClipboardList, AlertTriangle, Eye, Target, Search, Globe, Briefcase, Shield, Zap } from 'lucide-react';

type EntityType = 'person' | 'organization' | 'location' | 'event' | 'topic' | 'asset';

interface WatchlistEntity {
  id: string;
  name: string;
  type: EntityType;
  description?: string;
  aliases?: string[];
  riskScore?: number;
  lastActivity?: string;
  signalCount?: number;
  trend?: 'rising' | 'falling' | 'stable';
  metadata?: Record<string, unknown>;
  addedAt: string;
  addedBy: string;
}

interface Watchlist {
  id: string;
  name: string;
  description?: string;
  color: string;
  icon?: string;
  entities: WatchlistEntity[];
  shared: boolean;
  createdBy: string;
  createdAt: string;
  updatedAt: string;
}

interface WatchlistManagerProps {
  watchlists: Watchlist[];
  onWatchlistCreate?: (watchlist: Omit<Watchlist, 'id' | 'createdAt' | 'updatedAt'>) => void;
  onWatchlistUpdate?: (id: string, changes: Partial<Watchlist>) => void;
  onWatchlistDelete?: (id: string) => void;
  onEntityAdd?: (watchlistId: string, entity: Omit<WatchlistEntity, 'id' | 'addedAt'>) => void;
  onEntityRemove?: (watchlistId: string, entityId: string) => void;
  onEntityClick?: (entity: WatchlistEntity) => void;
  currentUser?: { id: string; name: string };
}

// Icon mapping helper
const getIconComponent = (iconId?: string) => {
  const iconMap: Record<string, React.ReactNode> = {
    clipboard: <ClipboardList className="w-4 h-4" />,
    alert: <AlertTriangle className="w-4 h-4" />,
    eye: <Eye className="w-4 h-4" />,
    target: <Target className="w-4 h-4" />,
    search: <Search className="w-4 h-4" />,
    globe: <Globe className="w-4 h-4" />,
    briefcase: <Briefcase className="w-4 h-4" />,
    shield: <Shield className="w-4 h-4" />,
    zap: <Zap className="w-4 h-4" />,
  };
  return iconMap[iconId || 'clipboard'] || iconMap.clipboard;
};

// Component 48: Watchlist Entity Manager
export function WatchlistManager({
  watchlists,
  onWatchlistCreate,
  onWatchlistUpdate,
  onWatchlistDelete,
  onEntityAdd,
  onEntityRemove,
  onEntityClick,
  currentUser = { id: 'user-1', name: 'Analyst' },
}: WatchlistManagerProps) {
  const [selectedWatchlist, setSelectedWatchlist] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [isAddingEntity, setIsAddingEntity] = useState(false);
  const [sortBy, setSortBy] = useState<'name' | 'risk' | 'activity'>('risk');

  const typeConfig: Record<EntityType, { icon: React.ReactNode; color: string; label: string }> = {
    person: { icon: <User className="w-5 h-5" />, color: 'cyan', label: 'Person' },
    organization: { icon: <Building2 className="w-5 h-5" />, color: 'purple', label: 'Organization' },
    location: { icon: <MapPin className="w-5 h-5" />, color: 'green', label: 'Location' },
    event: { icon: <Calendar className="w-5 h-5" />, color: 'amber', label: 'Event' },
    topic: { icon: <MessageSquare className="w-5 h-5" />, color: 'blue', label: 'Topic' },
    asset: { icon: <Package className="w-5 h-5" />, color: 'slate', label: 'Asset' },
  };

  const activeWatchlist = watchlists.find(w => w.id === selectedWatchlist);

  // Filter and sort entities
  const filteredEntities = useMemo(() => {
    if (!activeWatchlist) return [];

    let entities = activeWatchlist.entities;

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      entities = entities.filter(e =>
        e.name.toLowerCase().includes(query) ||
        e.description?.toLowerCase().includes(query) ||
        e.aliases?.some(a => a.toLowerCase().includes(query))
      );
    }

    // Sort
    entities = [...entities].sort((a, b) => {
      switch (sortBy) {
        case 'risk':
          return (b.riskScore ?? 0) - (a.riskScore ?? 0);
        case 'activity':
          return new Date(b.lastActivity || 0).getTime() - new Date(a.lastActivity || 0).getTime();
        default:
          return a.name.localeCompare(b.name);
      }
    });

    return entities;
  }, [activeWatchlist, searchQuery, sortBy]);

  // Stats for selected watchlist
  const watchlistStats = useMemo(() => {
    if (!activeWatchlist) return null;

    const entities = activeWatchlist.entities;
    const avgRisk = entities.length > 0
      ? entities.reduce((sum, e) => sum + (e.riskScore ?? 0), 0) / entities.length
      : 0;
    const risingCount = entities.filter(e => e.trend === 'rising').length;
    const criticalCount = entities.filter(e => (e.riskScore ?? 0) >= 0.8).length;

    return { avgRisk, risingCount, criticalCount, total: entities.length };
  }, [activeWatchlist]);

  return (
    <div className="bg-slate-900/50 rounded-lg border border-slate-700 h-full flex">
      {/* Sidebar - Watchlist list */}
      <div className="w-64 border-r border-slate-700 flex flex-col">
        <div className="p-3 border-b border-slate-700">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-slate-200">Watchlists</h3>
            <button
              onClick={() => setIsCreating(true)}
              className="p-1 text-cyan-400 hover:bg-cyan-500/10 rounded transition-colors"
              title="Create watchlist"
            >
              +
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          {watchlists.map(watchlist => {
            const entityCount = watchlist.entities.length;
            const criticalCount = watchlist.entities.filter(e => (e.riskScore ?? 0) >= 0.8).length;
            const isSelected = selectedWatchlist === watchlist.id;

            return (
              <button
                key={watchlist.id}
                onClick={() => setSelectedWatchlist(watchlist.id)}
                className={`w-full p-3 text-left border-b border-slate-800 transition-colors ${
                  isSelected ? 'bg-cyan-500/10 border-l-2 border-l-cyan-500' : 'hover:bg-slate-800/50'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: watchlist.color }}
                  />
                  <span className={`text-sm flex items-center gap-2 ${isSelected ? 'text-cyan-400' : 'text-slate-200'}`}>
                    {getIconComponent(watchlist.icon)} {watchlist.name}
                  </span>
                  {watchlist.shared && (
                    <Users className="ml-auto w-3 h-3 text-slate-500" />
                  )}
                </div>
                <div className="flex items-center gap-3 mt-1 text-xs text-slate-500">
                  <span>{entityCount} entities</span>
                  {criticalCount > 0 && (
                    <span className="text-red-400">{criticalCount} critical</span>
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col">
        {activeWatchlist ? (
          <>
            {/* Header */}
            <div className="p-4 border-b border-slate-700">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <span
                    className="w-4 h-4 rounded-full"
                    style={{ backgroundColor: activeWatchlist.color }}
                  />
                  <h2 className="text-lg font-medium text-slate-200 flex items-center gap-2">
                    {getIconComponent(activeWatchlist.icon)} {activeWatchlist.name}
                  </h2>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setIsAddingEntity(true)}
                    className="px-3 py-1.5 bg-cyan-500/20 text-cyan-400 rounded text-sm font-medium hover:bg-cyan-500/30"
                  >
                    + Add Entity
                  </button>
                  <button
                    onClick={() => onWatchlistDelete?.(activeWatchlist.id)}
                    className="p-1.5 text-slate-400 hover:text-red-400 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {activeWatchlist.description && (
                <p className="text-xs text-slate-500 mb-3">{activeWatchlist.description}</p>
              )}

              {/* Stats */}
              {watchlistStats && (
                <div className="grid grid-cols-4 gap-3">
                  <div className="bg-slate-800/50 rounded p-2">
                    <div className="text-xs text-slate-500">Total</div>
                    <div className="text-lg font-bold text-slate-200">{watchlistStats.total}</div>
                  </div>
                  <div className="bg-slate-800/50 rounded p-2">
                    <div className="text-xs text-slate-500">Avg Risk</div>
                    <div className={`text-lg font-bold ${
                      watchlistStats.avgRisk >= 0.7 ? 'text-red-400' :
                      watchlistStats.avgRisk >= 0.4 ? 'text-amber-400' : 'text-green-400'
                    }`}>
                      {(watchlistStats.avgRisk * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className="bg-slate-800/50 rounded p-2">
                    <div className="text-xs text-slate-500">Rising</div>
                    <div className="text-lg font-bold text-amber-400">{watchlistStats.risingCount}</div>
                  </div>
                  <div className="bg-slate-800/50 rounded p-2">
                    <div className="text-xs text-slate-500">Critical</div>
                    <div className="text-lg font-bold text-red-400">{watchlistStats.criticalCount}</div>
                  </div>
                </div>
              )}
            </div>

            {/* Search and filter */}
            <div className="p-3 border-b border-slate-800 flex items-center gap-3">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search entities..."
                className="flex-1 px-3 py-2 bg-slate-800 border border-slate-700 rounded text-sm text-slate-200"
              />
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
                className="px-3 py-2 bg-slate-800 border border-slate-700 rounded text-sm text-slate-200"
              >
                <option value="risk">Sort by Risk</option>
                <option value="activity">Sort by Activity</option>
                <option value="name">Sort by Name</option>
              </select>
            </div>

            {/* Entity list */}
            <div className="flex-1 overflow-y-auto">
              {filteredEntities.length === 0 ? (
                <div className="flex items-center justify-center h-full text-slate-500 text-sm">
                  {searchQuery ? 'No matching entities' : 'No entities in this watchlist'}
                </div>
              ) : (
                <div className="divide-y divide-slate-800">
                  {filteredEntities.map(entity => (
                    <EntityRow
                      key={entity.id}
                      entity={entity}
                      typeConfig={typeConfig[entity.type]}
                      onClick={() => onEntityClick?.(entity)}
                      onRemove={() => onEntityRemove?.(activeWatchlist.id, entity.id)}
                    />
                  ))}
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="flex items-center justify-center h-full text-slate-500">
            Select a watchlist to view entities
          </div>
        )}
      </div>

      {/* Create watchlist modal */}
      {isCreating && (
        <CreateWatchlistModal
          currentUser={currentUser}
          onClose={() => setIsCreating(false)}
          onCreate={(data) => {
            onWatchlistCreate?.(data);
            setIsCreating(false);
          }}
        />
      )}

      {/* Add entity modal */}
      {isAddingEntity && activeWatchlist && (
        <AddEntityModal
          typeConfig={typeConfig}
          currentUser={currentUser}
          onClose={() => setIsAddingEntity(false)}
          onAdd={(data) => {
            onEntityAdd?.(activeWatchlist.id, data);
            setIsAddingEntity(false);
          }}
        />
      )}
    </div>
  );
}

// Entity row component
function EntityRow({
  entity,
  typeConfig,
  onClick,
  onRemove,
}: {
  entity: WatchlistEntity;
  typeConfig: { icon: React.ReactNode; color: string; label: string };
  onClick: () => void;
  onRemove: () => void;
}) {
  const trendConfig = {
    rising: { icon: <ArrowUp className="w-4 h-4" />, color: 'text-red-400' },
    falling: { icon: <ArrowDown className="w-4 h-4" />, color: 'text-green-400' },
    stable: { icon: <ArrowRight className="w-4 h-4" />, color: 'text-slate-400' },
  };

  const trend = entity.trend ? trendConfig[entity.trend] : null;

  return (
    <div
      className="p-3 hover:bg-slate-800/50 cursor-pointer transition-colors flex items-center gap-3"
      onClick={onClick}
    >
      {/* Type icon */}
      <div
        className="w-10 h-10 rounded-lg flex items-center justify-center text-lg"
        style={{ backgroundColor: `var(--${typeConfig.color}-500/20)` }}
      >
        {typeConfig.icon}
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-slate-200 truncate">{entity.name}</span>
          <span className="px-1.5 py-0.5 bg-slate-700 rounded text-xs text-slate-400">
            {typeConfig.label}
          </span>
        </div>
        {entity.description && (
          <p className="text-xs text-slate-500 truncate mt-0.5">{entity.description}</p>
        )}
        <div className="flex items-center gap-3 mt-1">
          {entity.signalCount !== undefined && (
            <span className="text-xs text-slate-500">{entity.signalCount} signals</span>
          )}
          {entity.lastActivity && (
            <span className="text-xs text-slate-500">
              Last: {new Date(entity.lastActivity).toLocaleDateString()}
            </span>
          )}
        </div>
      </div>

      {/* Risk score */}
      <div className="text-right">
        {entity.riskScore !== undefined && (
          <div className="flex items-center gap-1">
            <div
              className={`text-lg font-bold font-mono ${
                entity.riskScore >= 0.8 ? 'text-red-400' :
                entity.riskScore >= 0.5 ? 'text-amber-400' : 'text-green-400'
              }`}
            >
              {(entity.riskScore * 100).toFixed(0)}%
            </div>
            {trend && (
              <span className={trend.color}>{trend.icon}</span>
            )}
          </div>
        )}
      </div>

      {/* Remove button */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onRemove();
        }}
        className="p-2 text-slate-500 hover:text-red-400 transition-colors"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}

// Create watchlist modal
function CreateWatchlistModal({
  currentUser,
  onClose,
  onCreate,
}: {
  currentUser: { id: string; name: string };
  onClose: () => void;
  onCreate: (data: Omit<Watchlist, 'id' | 'createdAt' | 'updatedAt'>) => void;
}) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [color, setColor] = useState('#06b6d4');
  const [icon, setIcon] = useState('clipboard');
  const [shared, setShared] = useState(false);

  const colors = ['#06b6d4', '#f59e0b', '#ef4444', '#22c55e', '#8b5cf6', '#ec4899'];
  const iconOptions = [
    { id: 'clipboard', icon: <ClipboardList className="w-4 h-4" /> },
    { id: 'alert', icon: <AlertTriangle className="w-4 h-4" /> },
    { id: 'eye', icon: <Eye className="w-4 h-4" /> },
    { id: 'target', icon: <Target className="w-4 h-4" /> },
    { id: 'search', icon: <Search className="w-4 h-4" /> },
    { id: 'globe', icon: <Globe className="w-4 h-4" /> },
    { id: 'briefcase', icon: <Briefcase className="w-4 h-4" /> },
    { id: 'shield', icon: <Shield className="w-4 h-4" /> },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div className="bg-slate-800 rounded-lg p-6 w-96" onClick={(e) => e.stopPropagation()}>
        <h3 className="text-lg font-medium text-slate-200 mb-4">Create Watchlist</h3>

        <div className="space-y-4">
          <div>
            <label className="block text-xs text-slate-400 mb-1">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-slate-200"
              placeholder="My Watchlist"
            />
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-1">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-slate-200 resize-none"
              rows={2}
            />
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-2">Icon</label>
            <div className="flex gap-2">
              {iconOptions.map(opt => (
                <button
                  key={opt.id}
                  onClick={() => setIcon(opt.id)}
                  className={`w-8 h-8 rounded flex items-center justify-center ${
                    icon === opt.id ? 'bg-cyan-500/20 ring-2 ring-cyan-500' : 'bg-slate-700'
                  }`}
                >
                  {opt.icon}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-2">Color</label>
            <div className="flex gap-2">
              {colors.map(c => (
                <button
                  key={c}
                  onClick={() => setColor(c)}
                  className={`w-8 h-8 rounded-full ${color === c ? 'ring-2 ring-white' : ''}`}
                  style={{ backgroundColor: c }}
                />
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="shared"
              checked={shared}
              onChange={(e) => setShared(e.target.checked)}
              className="rounded border-slate-600"
            />
            <label htmlFor="shared" className="text-sm text-slate-300">Share with team</label>
          </div>
        </div>

        <div className="flex justify-end gap-2 mt-6">
          <button onClick={onClose} className="px-3 py-1.5 text-sm text-slate-400">
            Cancel
          </button>
          <button
            onClick={() => onCreate({
              name,
              description,
              color,
              icon,
              entities: [],
              shared,
              createdBy: currentUser.id,
            })}
            disabled={!name.trim()}
            className="px-4 py-1.5 bg-cyan-500 text-slate-900 rounded text-sm font-medium disabled:opacity-50"
          >
            Create
          </button>
        </div>
      </div>
    </div>
  );
}

// Add entity modal
function AddEntityModal({
  typeConfig,
  currentUser,
  onClose,
  onAdd,
}: {
  typeConfig: Record<EntityType, { icon: React.ReactNode; color: string; label: string }>;
  currentUser: { id: string; name: string };
  onClose: () => void;
  onAdd: (data: Omit<WatchlistEntity, 'id' | 'addedAt'>) => void;
}) {
  const [name, setName] = useState('');
  const [type, setType] = useState<EntityType>('person');
  const [description, setDescription] = useState('');
  const [aliases, setAliases] = useState('');

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div className="bg-slate-800 rounded-lg p-6 w-96" onClick={(e) => e.stopPropagation()}>
        <h3 className="text-lg font-medium text-slate-200 mb-4">Add Entity</h3>

        <div className="space-y-4">
          <div>
            <label className="block text-xs text-slate-400 mb-1">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-slate-200"
            />
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-2">Type</label>
            <div className="grid grid-cols-3 gap-2">
              {(Object.entries(typeConfig) as [EntityType, typeof typeConfig[EntityType]][]).map(([t, config]) => (
                <button
                  key={t}
                  onClick={() => setType(t)}
                  className={`p-2 rounded text-xs ${
                    type === t ? 'bg-cyan-500/20 text-cyan-400' : 'bg-slate-700 text-slate-400'
                  }`}
                >
                  {config.icon} {config.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-1">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-slate-200 resize-none"
              rows={2}
            />
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-1">Aliases (comma separated)</label>
            <input
              type="text"
              value={aliases}
              onChange={(e) => setAliases(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-slate-200"
            />
          </div>
        </div>

        <div className="flex justify-end gap-2 mt-6">
          <button onClick={onClose} className="px-3 py-1.5 text-sm text-slate-400">
            Cancel
          </button>
          <button
            onClick={() => onAdd({
              name,
              type,
              description,
              aliases: aliases.split(',').map(a => a.trim()).filter(Boolean),
              addedBy: currentUser.id,
            })}
            disabled={!name.trim()}
            className="px-4 py-1.5 bg-cyan-500 text-slate-900 rounded text-sm font-medium disabled:opacity-50"
          >
            Add
          </button>
        </div>
      </div>
    </div>
  );
}

// Mock data
export const mockWatchlists: Watchlist[] = [
  {
    id: '1',
    name: 'Priority Actors',
    description: 'Key state and non-state actors requiring continuous monitoring',
    color: '#ef4444',
    icon: 'target',
    shared: true,
    createdBy: 'user-1',
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-15T00:00:00Z',
    entities: [
      {
        id: 'e1',
        name: 'Vladimir Putin',
        type: 'person',
        description: 'President of Russia',
        riskScore: 0.92,
        signalCount: 342,
        trend: 'stable',
        lastActivity: '2024-01-15T10:00:00Z',
        addedAt: '2024-01-01T00:00:00Z',
        addedBy: 'user-1',
      },
      {
        id: 'e2',
        name: 'Xi Jinping',
        type: 'person',
        description: 'President of China',
        riskScore: 0.75,
        signalCount: 256,
        trend: 'rising',
        lastActivity: '2024-01-14T15:00:00Z',
        addedAt: '2024-01-01T00:00:00Z',
        addedBy: 'user-1',
      },
      {
        id: 'e3',
        name: 'Wagner Group',
        type: 'organization',
        description: 'Russian private military company',
        aliases: ['PMC Wagner', 'Vagner'],
        riskScore: 0.88,
        signalCount: 128,
        trend: 'falling',
        lastActivity: '2024-01-13T09:00:00Z',
        addedAt: '2024-01-02T00:00:00Z',
        addedBy: 'user-2',
      },
    ],
  },
  {
    id: '2',
    name: 'Critical Infrastructure',
    description: 'Key infrastructure assets and systems',
    color: '#f59e0b',
    icon: 'zap',
    shared: false,
    createdBy: 'user-1',
    createdAt: '2024-01-05T00:00:00Z',
    updatedAt: '2024-01-10T00:00:00Z',
    entities: [
      {
        id: 'e4',
        name: 'Nord Stream Pipeline',
        type: 'asset',
        description: 'Natural gas pipeline from Russia to Europe',
        riskScore: 0.45,
        signalCount: 45,
        trend: 'stable',
        lastActivity: '2024-01-10T00:00:00Z',
        addedAt: '2024-01-05T00:00:00Z',
        addedBy: 'user-1',
      },
    ],
  },
];
