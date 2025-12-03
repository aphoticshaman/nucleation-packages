'use client';

import { useState, useRef, useCallback, useMemo } from 'react';

type AnnotationType = 'note' | 'highlight' | 'question' | 'alert' | 'link';
type AnnotationPriority = 'low' | 'medium' | 'high' | 'critical';

interface Annotation {
  id: string;
  type: AnnotationType;
  content: string;
  author: {
    id: string;
    name: string;
    avatar?: string;
  };
  position: {
    x: number;
    y: number;
    anchorId?: string; // ID of element being annotated
  };
  priority: AnnotationPriority;
  tags: string[];
  replies: AnnotationReply[];
  resolved: boolean;
  createdAt: string;
  updatedAt: string;
  linkedEntityId?: string;
  linkedEntityType?: string;
}

interface AnnotationReply {
  id: string;
  content: string;
  author: {
    id: string;
    name: string;
    avatar?: string;
  };
  createdAt: string;
}

interface AnnotationLayerProps {
  annotations: Annotation[];
  onAnnotationCreate?: (annotation: Omit<Annotation, 'id' | 'createdAt' | 'updatedAt'>) => void;
  onAnnotationUpdate?: (id: string, changes: Partial<Annotation>) => void;
  onAnnotationDelete?: (id: string) => void;
  onReplyCreate?: (annotationId: string, reply: Omit<AnnotationReply, 'id' | 'createdAt'>) => void;
  currentUser?: { id: string; name: string; avatar?: string };
  visible?: boolean;
  editable?: boolean;
  width?: number;
  height?: number;
}

// Component 47: Collaborative Annotation Layer
export function AnnotationLayer({
  annotations,
  onAnnotationCreate,
  onAnnotationUpdate,
  onAnnotationDelete,
  onReplyCreate,
  currentUser = { id: 'user-1', name: 'Analyst' },
  visible = true,
  editable = true,
  width = 800,
  height = 600,
}: AnnotationLayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedAnnotation, setSelectedAnnotation] = useState<string | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [createPosition, setCreatePosition] = useState<{ x: number; y: number } | null>(null);
  const [filterType, setFilterType] = useState<AnnotationType | 'all'>('all');
  const [showResolved, setShowResolved] = useState(false);

  const typeConfig: Record<AnnotationType, { icon: string; color: string; label: string }> = {
    note: { icon: '📝', color: 'cyan', label: 'Note' },
    highlight: { icon: '✨', color: 'amber', label: 'Highlight' },
    question: { icon: '❓', color: 'purple', label: 'Question' },
    alert: { icon: '⚠', color: 'red', label: 'Alert' },
    link: { icon: '🔗', color: 'green', label: 'Link' },
  };

  const priorityConfig: Record<AnnotationPriority, { bg: string; text: string }> = {
    low: { bg: 'bg-slate-500/20', text: 'text-slate-400' },
    medium: { bg: 'bg-cyan-500/20', text: 'text-cyan-400' },
    high: { bg: 'bg-amber-500/20', text: 'text-amber-400' },
    critical: { bg: 'bg-red-500/20', text: 'text-red-400' },
  };

  // Filter annotations
  const filteredAnnotations = useMemo(() => {
    return annotations.filter(a => {
      if (!showResolved && a.resolved) return false;
      if (filterType !== 'all' && a.type !== filterType) return false;
      return true;
    });
  }, [annotations, filterType, showResolved]);

  // Handle canvas click for creating
  const handleCanvasClick = useCallback((e: React.MouseEvent) => {
    if (!editable) return;

    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Check if clicking on existing annotation
    const clickedAnnotation = filteredAnnotations.find(a => {
      const dx = a.position.x - x;
      const dy = a.position.y - y;
      return Math.sqrt(dx * dx + dy * dy) < 20;
    });

    if (clickedAnnotation) {
      setSelectedAnnotation(clickedAnnotation.id);
    } else {
      setCreatePosition({ x, y });
      setIsCreating(true);
    }
  }, [editable, filteredAnnotations]);

  if (!visible) return null;

  return (
    <div className="relative">
      {/* Toolbar */}
      <div className="absolute top-4 right-4 z-20 flex items-center gap-2 bg-slate-800/90 rounded-lg p-2 border border-slate-700">
        {/* Type filter */}
        <select
          value={filterType}
          onChange={(e) => setFilterType(e.target.value as AnnotationType | 'all')}
          className="px-2 py-1 bg-slate-700 border border-slate-600 rounded text-xs text-slate-200"
        >
          <option value="all">All Types</option>
          {Object.entries(typeConfig).map(([type, config]) => (
            <option key={type} value={type}>{config.icon} {config.label}</option>
          ))}
        </select>

        {/* Show resolved toggle */}
        <button
          onClick={() => setShowResolved(!showResolved)}
          className={`px-2 py-1 rounded text-xs transition-colors ${
            showResolved ? 'bg-cyan-500/20 text-cyan-400' : 'bg-slate-700 text-slate-400'
          }`}
        >
          {showResolved ? '✓ Resolved' : 'Resolved'}
        </button>

        {/* Count */}
        <span className="text-xs text-slate-500">
          {filteredAnnotations.length} / {annotations.length}
        </span>
      </div>

      {/* Canvas */}
      <div
        ref={containerRef}
        className="relative bg-transparent cursor-crosshair"
        style={{ width, height }}
        onClick={handleCanvasClick}
      >
        {/* Annotation markers */}
        {filteredAnnotations.map(annotation => {
          const config = typeConfig[annotation.type];
          const isSelected = selectedAnnotation === annotation.id;

          return (
            <div
              key={annotation.id}
              className="absolute transform -translate-x-1/2 -translate-y-1/2 z-10"
              style={{ left: annotation.position.x, top: annotation.position.y }}
            >
              {/* Marker */}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedAnnotation(isSelected ? null : annotation.id);
                }}
                className={`
                  w-8 h-8 rounded-full flex items-center justify-center
                  transition-all duration-200 shadow-lg
                  ${annotation.resolved ? 'opacity-50' : ''}
                  ${isSelected ? 'ring-2 ring-white scale-110' : 'hover:scale-105'}
                `}
                style={{
                  backgroundColor: `var(--${config.color}-500)`,
                  boxShadow: `0 0 10px var(--${config.color}-500)`,
                }}
              >
                <span className="text-sm">{config.icon}</span>
              </button>

              {/* Reply count badge */}
              {annotation.replies.length > 0 && (
                <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-purple-500 text-white text-xs flex items-center justify-center">
                  {annotation.replies.length}
                </span>
              )}

              {/* Expanded card */}
              {isSelected && (
                <AnnotationCard
                  annotation={annotation}
                  config={config}
                  priorityConfig={priorityConfig}
                  currentUser={currentUser}
                  editable={editable}
                  onUpdate={(changes) => onAnnotationUpdate?.(annotation.id, changes)}
                  onDelete={() => onAnnotationDelete?.(annotation.id)}
                  onReply={(content) => onReplyCreate?.(annotation.id, { content, author: currentUser })}
                  onClose={() => setSelectedAnnotation(null)}
                />
              )}
            </div>
          );
        })}
      </div>

      {/* Create annotation modal */}
      {isCreating && createPosition && (
        <CreateAnnotationModal
          position={createPosition}
          typeConfig={typeConfig}
          currentUser={currentUser}
          onClose={() => {
            setIsCreating(false);
            setCreatePosition(null);
          }}
          onCreate={(data) => {
            onAnnotationCreate?.({
              ...data,
              position: createPosition,
              author: currentUser,
              replies: [],
              resolved: false,
            });
            setIsCreating(false);
            setCreatePosition(null);
          }}
        />
      )}
    </div>
  );
}

// Annotation detail card
function AnnotationCard({
  annotation,
  config,
  priorityConfig,
  currentUser,
  editable,
  onUpdate,
  onDelete,
  onReply,
  onClose,
}: {
  annotation: Annotation;
  config: { icon: string; color: string; label: string };
  priorityConfig: Record<AnnotationPriority, { bg: string; text: string }>;
  currentUser: { id: string; name: string };
  editable: boolean;
  onUpdate: (changes: Partial<Annotation>) => void;
  onDelete: () => void;
  onReply: (content: string) => void;
  onClose: () => void;
}) {
  const [replyContent, setReplyContent] = useState('');
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState(annotation.content);

  const isAuthor = annotation.author.id === currentUser.id;
  const priority = priorityConfig[annotation.priority];

  return (
    <div
      className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 z-20"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="w-80 bg-slate-800 border border-slate-600 rounded-lg shadow-xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-3 bg-slate-700/50 border-b border-slate-600">
          <div className="flex items-center gap-2">
            <span>{config.icon}</span>
            <span className="text-sm font-medium text-slate-200">{config.label}</span>
            <span className={`px-1.5 py-0.5 rounded text-xs ${priority.bg} ${priority.text}`}>
              {annotation.priority.toUpperCase()}
            </span>
          </div>
          <div className="flex items-center gap-1">
            {editable && isAuthor && (
              <>
                <button
                  onClick={() => setIsEditing(!isEditing)}
                  className="p-1 text-slate-400 hover:text-slate-200"
                >
                  ✎
                </button>
                <button
                  onClick={onDelete}
                  className="p-1 text-slate-400 hover:text-red-400"
                >
                  ✕
                </button>
              </>
            )}
            <button
              onClick={onClose}
              className="p-1 text-slate-400 hover:text-slate-200 ml-2"
            >
              ×
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-3">
          {isEditing ? (
            <div className="space-y-2">
              <textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                className="w-full px-2 py-1.5 bg-slate-900 border border-slate-600 rounded text-sm text-slate-200 resize-none"
                rows={3}
              />
              <div className="flex justify-end gap-2">
                <button
                  onClick={() => setIsEditing(false)}
                  className="px-2 py-1 text-xs text-slate-400"
                >
                  Cancel
                </button>
                <button
                  onClick={() => {
                    onUpdate({ content: editContent });
                    setIsEditing(false);
                  }}
                  className="px-2 py-1 bg-cyan-500 text-slate-900 rounded text-xs font-medium"
                >
                  Save
                </button>
              </div>
            </div>
          ) : (
            <p className="text-sm text-slate-300">{annotation.content}</p>
          )}

          {/* Author & meta */}
          <div className="flex items-center justify-between mt-3 pt-2 border-t border-slate-700">
            <div className="flex items-center gap-2">
              <div className="w-5 h-5 rounded-full bg-slate-600 flex items-center justify-center text-xs text-slate-300">
                {annotation.author.name.charAt(0)}
              </div>
              <span className="text-xs text-slate-400">{annotation.author.name}</span>
            </div>
            <span className="text-xs text-slate-500">
              {new Date(annotation.createdAt).toLocaleDateString()}
            </span>
          </div>

          {/* Tags */}
          {annotation.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {annotation.tags.map(tag => (
                <span key={tag} className="px-1.5 py-0.5 bg-slate-700 rounded text-xs text-slate-400">
                  #{tag}
                </span>
              ))}
            </div>
          )}

          {/* Resolve button */}
          {editable && (
            <button
              onClick={() => onUpdate({ resolved: !annotation.resolved })}
              className={`w-full mt-3 py-1.5 rounded text-xs font-medium transition-colors ${
                annotation.resolved
                  ? 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                  : 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
              }`}
            >
              {annotation.resolved ? 'Reopen' : '✓ Mark Resolved'}
            </button>
          )}
        </div>

        {/* Replies */}
        {annotation.replies.length > 0 && (
          <div className="border-t border-slate-600">
            <div className="max-h-32 overflow-y-auto">
              {annotation.replies.map(reply => (
                <div key={reply.id} className="p-2 border-b border-slate-700 last:border-b-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-medium text-slate-300">{reply.author.name}</span>
                    <span className="text-xs text-slate-500">
                      {new Date(reply.createdAt).toLocaleDateString()}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400">{reply.content}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Reply input */}
        {editable && (
          <div className="p-2 border-t border-slate-600">
            <div className="flex gap-2">
              <input
                type="text"
                value={replyContent}
                onChange={(e) => setReplyContent(e.target.value)}
                placeholder="Add reply..."
                className="flex-1 px-2 py-1 bg-slate-900 border border-slate-600 rounded text-xs text-slate-200"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && replyContent.trim()) {
                    onReply(replyContent.trim());
                    setReplyContent('');
                  }
                }}
              />
              <button
                onClick={() => {
                  if (replyContent.trim()) {
                    onReply(replyContent.trim());
                    setReplyContent('');
                  }
                }}
                className="px-2 py-1 bg-cyan-500/20 text-cyan-400 rounded text-xs"
              >
                →
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Create annotation modal
function CreateAnnotationModal({
  position,
  typeConfig,
  currentUser,
  onClose,
  onCreate,
}: {
  position: { x: number; y: number };
  typeConfig: Record<AnnotationType, { icon: string; color: string; label: string }>;
  currentUser: { id: string; name: string };
  onClose: () => void;
  onCreate: (data: Omit<Annotation, 'id' | 'createdAt' | 'updatedAt' | 'position' | 'author' | 'replies' | 'resolved'>) => void;
}) {
  const [type, setType] = useState<AnnotationType>('note');
  const [content, setContent] = useState('');
  const [priority, setPriority] = useState<AnnotationPriority>('medium');
  const [tags, setTags] = useState('');

  const handleSubmit = () => {
    if (!content.trim()) return;

    onCreate({
      type,
      content: content.trim(),
      priority,
      tags: tags.split(',').map(t => t.trim()).filter(Boolean),
    });
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={onClose}
    >
      <div
        className="bg-slate-800 rounded-lg border border-slate-600 p-4 w-80"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 className="text-sm font-medium text-slate-200 mb-4">New Annotation</h3>

        {/* Type selection */}
        <div className="flex gap-1 mb-4">
          {(Object.entries(typeConfig) as [AnnotationType, typeof typeConfig[AnnotationType]][]).map(([t, config]) => (
            <button
              key={t}
              onClick={() => setType(t)}
              className={`flex-1 py-2 rounded text-sm transition-colors ${
                type === t
                  ? `bg-${config.color}-500/20 text-${config.color}-400 border border-${config.color}-500/50`
                  : 'bg-slate-700 text-slate-400 border border-transparent'
              }`}
            >
              {config.icon}
            </button>
          ))}
        </div>

        {/* Content */}
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="Write your annotation..."
          className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-slate-200 resize-none mb-3"
          rows={3}
          autoFocus
        />

        {/* Priority */}
        <div className="flex gap-2 mb-3">
          {(['low', 'medium', 'high', 'critical'] as AnnotationPriority[]).map(p => (
            <button
              key={p}
              onClick={() => setPriority(p)}
              className={`flex-1 py-1 rounded text-xs capitalize transition-colors ${
                priority === p
                  ? 'bg-cyan-500/20 text-cyan-400'
                  : 'bg-slate-700 text-slate-400'
              }`}
            >
              {p}
            </button>
          ))}
        </div>

        {/* Tags */}
        <input
          type="text"
          value={tags}
          onChange={(e) => setTags(e.target.value)}
          placeholder="Tags (comma separated)"
          className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-slate-200 mb-4"
        />

        {/* Actions */}
        <div className="flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-sm text-slate-400 hover:text-slate-200"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={!content.trim()}
            className="px-4 py-1.5 bg-cyan-500 text-slate-900 rounded text-sm font-medium disabled:opacity-50"
          >
            Create
          </button>
        </div>
      </div>
    </div>
  );
}

// Mock annotations for demo
export const mockAnnotations: Annotation[] = [
  {
    id: '1',
    type: 'alert',
    content: 'Unusual troop movement patterns detected in this region. Cross-reference with satellite data.',
    author: { id: 'user-1', name: 'Analyst A' },
    position: { x: 150, y: 200 },
    priority: 'high',
    tags: ['military', 'urgent'],
    replies: [
      {
        id: 'r1',
        content: 'Confirmed via OSINT sources. Escalating to senior analyst.',
        author: { id: 'user-2', name: 'Analyst B' },
        createdAt: '2024-01-15T11:00:00Z',
      },
    ],
    resolved: false,
    createdAt: '2024-01-15T10:30:00Z',
    updatedAt: '2024-01-15T11:00:00Z',
  },
  {
    id: '2',
    type: 'question',
    content: 'Why is there no economic data for Q4? Data gap or collection issue?',
    author: { id: 'user-2', name: 'Analyst B' },
    position: { x: 400, y: 300 },
    priority: 'medium',
    tags: ['data-quality'],
    replies: [],
    resolved: false,
    createdAt: '2024-01-14T15:00:00Z',
    updatedAt: '2024-01-14T15:00:00Z',
  },
  {
    id: '3',
    type: 'note',
    content: 'Historical precedent: Similar patterns observed in 2014 prior to Crimea annexation.',
    author: { id: 'user-1', name: 'Analyst A' },
    position: { x: 250, y: 150 },
    priority: 'low',
    tags: ['historical', 'pattern'],
    replies: [],
    resolved: true,
    createdAt: '2024-01-13T09:00:00Z',
    updatedAt: '2024-01-13T09:00:00Z',
  },
];
