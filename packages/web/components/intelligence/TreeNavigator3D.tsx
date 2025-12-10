'use client';

import { useRef, useState, useMemo, useCallback, Suspense, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Billboard, Line, Html } from '@react-three/drei';
import * as THREE from 'three';

/**
 * Temporal-Spanning 3D Tree Navigator
 *
 * A revolutionary UI where intel branches like a tree:
 * - Trunk = Executive Summary (core truth)
 * - Canopy (upward) = Future projections & emerging trends
 * - Roots (downward) = Historical analysis & causation chains
 * - Branches = Intel categories spreading spatially
 * - Time flows along the Y-axis: past (bottom) -> present (center) -> future (top)
 *
 * Navigate through 4D space: 3D spatial + temporal dimension.
 */

// Intel node representing a piece of information
export interface IntelNode {
  id: string;
  label: string;
  category: IntelCategory;
  content: string;
  timestamp: Date;
  temporalState: 'historical' | 'current' | 'projected';
  confidence: number; // 0-1
  risk?: 'low' | 'moderate' | 'elevated' | 'high' | 'critical';
  children?: IntelNode[];
  parentId?: string;
  position?: [number, number, number]; // Computed position in 3D space
}

export type IntelCategory =
  | 'executive'
  | 'political'
  | 'economic'
  | 'security'
  | 'military'
  | 'cyber'
  | 'terrorism'
  | 'health'
  | 'scitech'
  | 'resources'
  | 'space'
  | 'emerging';

// Category configuration with spatial positioning
const CATEGORY_CONFIG: Record<IntelCategory, {
  color: string;
  angle: number; // Radians around Y axis
  emoji: string;
  label: string;
}> = {
  executive: { color: '#fbbf24', angle: 0, emoji: '🎯', label: 'Executive' },
  political: { color: '#f59e0b', angle: Math.PI * 0.2, emoji: '🏛️', label: 'Political' },
  economic: { color: '#22c55e', angle: Math.PI * 0.4, emoji: '📈', label: 'Economic' },
  security: { color: '#ef4444', angle: Math.PI * 0.6, emoji: '⚔️', label: 'Security' },
  military: { color: '#78716c', angle: Math.PI * 0.8, emoji: '🎖️', label: 'Military' },
  cyber: { color: '#a855f7', angle: Math.PI, emoji: '💻', label: 'Cyber' },
  terrorism: { color: '#dc2626', angle: Math.PI * 1.2, emoji: '⚡', label: 'Terrorism' },
  health: { color: '#ec4899', angle: Math.PI * 1.4, emoji: '🏥', label: 'Health' },
  scitech: { color: '#06b6d4', angle: Math.PI * 1.6, emoji: '🔬', label: 'Sci/Tech' },
  resources: { color: '#10b981', angle: Math.PI * 1.8, emoji: '🌿', label: 'Resources' },
  space: { color: '#8b5cf6', angle: Math.PI * 1.9, emoji: '🛰️', label: 'Space' },
  emerging: { color: '#f97316', angle: Math.PI * 0.1, emoji: '🔮', label: 'Emerging' },
};

interface TreeNavigatorProps {
  nodes: IntelNode[];
  onNodeSelect?: (node: IntelNode) => void;
  selectedNodeId?: string;
  temporalFocus?: 'historical' | 'current' | 'projected' | 'all';
  showConnections?: boolean;
}

// Single node in 3D space
function IntelNodeMesh({
  node,
  isSelected,
  onSelect,
  temporalFocus,
}: {
  node: IntelNode;
  isSelected: boolean;
  onSelect: () => void;
  temporalFocus: string;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  const config = CATEGORY_CONFIG[node.category];

  // Calculate opacity based on temporal focus
  const opacity = useMemo(() => {
    if (temporalFocus === 'all') return 1;
    if (temporalFocus === node.temporalState) return 1;
    return 0.2;
  }, [temporalFocus, node.temporalState]);

  // Pulse animation for selected/hovered
  useFrame((state) => {
    if (!meshRef.current) return;
    const scale = isSelected
      ? 1.2 + Math.sin(state.clock.elapsedTime * 3) * 0.1
      : hovered
        ? 1.1
        : 1;
    meshRef.current.scale.setScalar(scale);
  });

  // Size based on confidence and hierarchy
  const size = useMemo(() => {
    const baseSize = node.category === 'executive' ? 0.4 : 0.25;
    return baseSize * (0.5 + node.confidence * 0.5);
  }, [node.category, node.confidence]);

  // Risk-based glow
  const emissiveIntensity = useMemo(() => {
    switch (node.risk) {
      case 'critical': return 0.8;
      case 'high': return 0.5;
      case 'elevated': return 0.3;
      default: return 0.1;
    }
  }, [node.risk]);

  return (
    <group position={node.position}>
      {/* Main node sphere */}
      <mesh
        ref={meshRef}
        onClick={onSelect}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <sphereGeometry args={[size, 16, 16]} />
        <meshStandardMaterial
          color={config.color}
          emissive={config.color}
          emissiveIntensity={emissiveIntensity}
          transparent
          opacity={opacity}
          roughness={0.3}
          metalness={0.7}
        />
      </mesh>

      {/* Label billboard - always faces camera */}
      <Billboard position={[0, size + 0.15, 0]} follow lockX={false} lockY={false} lockZ={false}>
        <Text
          fontSize={0.12}
          color={hovered || isSelected ? '#ffffff' : '#94a3b8'}
          anchorX="center"
          anchorY="bottom"
          outlineWidth={0.01}
          outlineColor="#0f172a"
        >
          {config.emoji} {node.label}
        </Text>
      </Billboard>

      {/* Temporal indicator ring */}
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, -size - 0.05, 0]}>
        <ringGeometry args={[size * 0.8, size, 32]} />
        <meshBasicMaterial
          color={
            node.temporalState === 'historical'
              ? '#64748b'
              : node.temporalState === 'projected'
                ? '#3b82f6'
                : '#22c55e'
          }
          transparent
          opacity={opacity * 0.5}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Hover info panel */}
      {(hovered || isSelected) && (
        <Html position={[size + 0.3, 0, 0]} distanceFactor={3}>
          <div className="bg-slate-900/95 border border-slate-700 rounded-lg p-3 min-w-[200px] max-w-[300px] backdrop-blur-sm pointer-events-none">
            <div className="flex items-center gap-2 mb-2">
              <span>{config.emoji}</span>
              <span className="font-medium text-white text-sm">{node.label}</span>
              {node.risk && (
                <span
                  className={`text-xs px-1.5 py-0.5 rounded ${
                    node.risk === 'critical'
                      ? 'bg-red-500/20 text-red-400'
                      : node.risk === 'high'
                        ? 'bg-orange-500/20 text-orange-400'
                        : node.risk === 'elevated'
                          ? 'bg-yellow-500/20 text-yellow-400'
                          : 'bg-green-500/20 text-green-400'
                  }`}
                >
                  {node.risk.toUpperCase()}
                </span>
              )}
            </div>
            <p className="text-xs text-slate-400 leading-relaxed line-clamp-3">
              {node.content}
            </p>
            <div className="flex items-center gap-2 mt-2 text-xs text-slate-500">
              <span>
                {node.temporalState === 'historical' ? '📜' : node.temporalState === 'projected' ? '🔮' : '📍'}
              </span>
              <span className="capitalize">{node.temporalState}</span>
              <span>•</span>
              <span>{Math.round(node.confidence * 100)}% confidence</span>
            </div>
          </div>
        </Html>
      )}
    </group>
  );
}

// Connection lines between nodes
function NodeConnections({
  nodes,
  opacity,
}: {
  nodes: IntelNode[];
  opacity: number;
}) {
  const connections = useMemo(() => {
    const result: Array<{ from: [number, number, number]; to: [number, number, number]; color: string }> = [];

    nodes.forEach((node) => {
      if (node.parentId && node.position) {
        const parent = nodes.find((n) => n.id === node.parentId);
        if (parent?.position) {
          result.push({
            from: parent.position,
            to: node.position,
            color: CATEGORY_CONFIG[node.category].color,
          });
        }
      }
    });

    return result;
  }, [nodes]);

  return (
    <>
      {connections.map((conn, i) => (
        <Line
          key={i}
          points={[conn.from, conn.to]}
          color={conn.color}
          lineWidth={1}
          transparent
          opacity={opacity * 0.4}
        />
      ))}
    </>
  );
}

// Central trunk/axis representing time
function TemporalAxis() {
  return (
    <group>
      {/* Main trunk */}
      <mesh position={[0, 0, 0]}>
        <cylinderGeometry args={[0.08, 0.15, 6, 32]} />
        <meshStandardMaterial
          color="#1e293b"
          transparent
          opacity={0.6}
          roughness={0.8}
        />
      </mesh>

      {/* Time markers */}
      {[-2, 0, 2].map((y, i) => (
        <group key={i} position={[0, y, 0]}>
          <mesh>
            <torusGeometry args={[0.3, 0.03, 8, 32]} />
            <meshStandardMaterial
              color={y < 0 ? '#64748b' : y > 0 ? '#3b82f6' : '#22c55e'}
              emissive={y < 0 ? '#64748b' : y > 0 ? '#3b82f6' : '#22c55e'}
              emissiveIntensity={0.3}
            />
          </mesh>
          <Billboard position={[0.5, 0, 0]}>
            <Text fontSize={0.1} color="#64748b">
              {y < 0 ? 'PAST' : y > 0 ? 'FUTURE' : 'NOW'}
            </Text>
          </Billboard>
        </group>
      ))}
    </group>
  );
}

// FPS-style WASD + Mouse controller for desktop
function FPSController({
  enabled,
  targetPosition,
}: {
  enabled: boolean;
  targetPosition?: [number, number, number];
}) {
  const { camera, gl } = useThree();
  const moveState = useRef({ forward: false, backward: false, left: false, right: false, up: false, down: false });
  const euler = useRef(new THREE.Euler(0, 0, 0, 'YXZ'));
  const isLocked = useRef(false);
  const MOVE_SPEED = 8;
  const MOUSE_SENSITIVITY = 0.002;

  // Handle keyboard
  useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.code) {
        case 'KeyW': case 'ArrowUp': moveState.current.forward = true; break;
        case 'KeyS': case 'ArrowDown': moveState.current.backward = true; break;
        case 'KeyA': case 'ArrowLeft': moveState.current.left = true; break;
        case 'KeyD': case 'ArrowRight': moveState.current.right = true; break;
        case 'Space': moveState.current.up = true; break;
        case 'ShiftLeft': case 'ShiftRight': moveState.current.down = true; break;
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      switch (e.code) {
        case 'KeyW': case 'ArrowUp': moveState.current.forward = false; break;
        case 'KeyS': case 'ArrowDown': moveState.current.backward = false; break;
        case 'KeyA': case 'ArrowLeft': moveState.current.left = false; break;
        case 'KeyD': case 'ArrowRight': moveState.current.right = false; break;
        case 'Space': moveState.current.up = false; break;
        case 'ShiftLeft': case 'ShiftRight': moveState.current.down = false; break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [enabled]);

  // Handle mouse look (pointer lock)
  useEffect(() => {
    if (!enabled) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!isLocked.current) return;

      euler.current.setFromQuaternion(camera.quaternion);
      euler.current.y -= e.movementX * MOUSE_SENSITIVITY;
      euler.current.x -= e.movementY * MOUSE_SENSITIVITY;
      euler.current.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, euler.current.x));
      camera.quaternion.setFromEuler(euler.current);
    };

    const handleClick = () => {
      gl.domElement.requestPointerLock();
    };

    const handleLockChange = () => {
      isLocked.current = document.pointerLockElement === gl.domElement;
    };

    gl.domElement.addEventListener('click', handleClick);
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('pointerlockchange', handleLockChange);

    return () => {
      gl.domElement.removeEventListener('click', handleClick);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('pointerlockchange', handleLockChange);
    };
  }, [enabled, camera, gl]);

  // Movement update
  useFrame((_, delta) => {
    if (!enabled) return;

    const direction = new THREE.Vector3();
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);

    if (moveState.current.forward) direction.add(forward);
    if (moveState.current.backward) direction.sub(forward);
    if (moveState.current.left) direction.sub(right);
    if (moveState.current.right) direction.add(right);
    if (moveState.current.up) direction.y += 1;
    if (moveState.current.down) direction.y -= 1;

    direction.normalize().multiplyScalar(MOVE_SPEED * delta);
    camera.position.add(direction);

    // Fly to target on selection
    if (targetPosition) {
      const target = new THREE.Vector3(...targetPosition);
      target.add(new THREE.Vector3(2, 1, 3));
      camera.position.lerp(target, 0.02);
    }
  });

  return null;
}

// Camera controller with smooth transitions (fallback for mobile)
function CameraController({
  targetPosition,
}: {
  targetPosition?: [number, number, number];
}) {
  const { camera } = useThree();
  const targetRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 8));

  useEffect(() => {
    if (targetPosition) {
      targetRef.current.set(...targetPosition);
    }
  }, [targetPosition]);

  useFrame(() => {
    // Smooth camera movement
    camera.position.lerp(
      new THREE.Vector3(
        targetRef.current.x + 5,
        targetRef.current.y + 3,
        targetRef.current.z + 8
      ),
      0.02
    );
    camera.lookAt(targetRef.current);
  });

  return null;
}

// Main 3D scene
function TreeScene({
  nodes,
  onNodeSelect,
  selectedNodeId,
  temporalFocus = 'all',
  showConnections = true,
}: TreeNavigatorProps) {
  // Compute 3D positions for nodes
  const positionedNodes = useMemo(() => {
    return nodes.map((node) => {
      const config = CATEGORY_CONFIG[node.category];

      // Y position based on temporal state
      let y = 0;
      switch (node.temporalState) {
        case 'historical':
          y = -2 + Math.random() * 0.5;
          break;
        case 'current':
          y = -0.5 + Math.random();
          break;
        case 'projected':
          y = 1.5 + Math.random() * 0.5;
          break;
      }

      // Radial position based on category - SPREAD OUT for better navigation
      const radius = node.category === 'executive' ? 0.5 : 5 + Math.random() * 2;
      const angle = config.angle + (Math.random() - 0.5) * 0.4;

      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;

      return {
        ...node,
        position: [x, y, z] as [number, number, number],
      };
    });
  }, [nodes]);

  const selectedNode = positionedNodes.find((n) => n.id === selectedNodeId);

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#3b82f6" />

      {/* Central temporal axis */}
      <TemporalAxis />

      {/* Node connections */}
      {showConnections && (
        <NodeConnections
          nodes={positionedNodes}
          opacity={temporalFocus === 'all' ? 1 : 0.3}
        />
      )}

      {/* Intel nodes */}
      {positionedNodes.map((node) => (
        <IntelNodeMesh
          key={node.id}
          node={node}
          isSelected={node.id === selectedNodeId}
          onSelect={() => onNodeSelect?.(node)}
          temporalFocus={temporalFocus}
        />
      ))}

      {/* FPS Controls for desktop (WASD + Mouse) */}
      <FPSController enabled={true} targetPosition={selectedNode?.position} />

      {/* OrbitControls as fallback - disabled when FPS is active */}
      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={2}
        maxDistance={30}
        maxPolarAngle={Math.PI * 0.85}
        enabled={false}
      />

      {/* Background stars for depth */}
      <mesh>
        <sphereGeometry args={[50, 32, 32]} />
        <meshBasicMaterial color="#020617" side={THREE.BackSide} />
      </mesh>
    </>
  );
}

// Loading fallback
function LoadingFallback() {
  return (
    <div className="w-full h-full flex items-center justify-center bg-slate-900">
      <div className="text-center">
        <div className="w-12 h-12 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-slate-400 text-sm">Initializing neural tree...</p>
      </div>
    </div>
  );
}

// Control panel overlay
function ControlPanel({
  temporalFocus,
  setTemporalFocus,
  showConnections,
  setShowConnections,
}: {
  temporalFocus: 'historical' | 'current' | 'projected' | 'all';
  setTemporalFocus: (f: 'historical' | 'current' | 'projected' | 'all') => void;
  showConnections: boolean;
  setShowConnections: (s: boolean) => void;
}) {
  return (
    <div className="absolute top-4 left-4 bg-slate-900/90 backdrop-blur-sm rounded-lg border border-slate-700 p-3 space-y-3">
      {/* Temporal focus */}
      <div>
        <p className="text-xs text-slate-400 mb-2">Temporal Focus</p>
        <div className="flex gap-1">
          {(['all', 'historical', 'current', 'projected'] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTemporalFocus(t)}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                temporalFocus === t
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
            >
              {t === 'all' ? '4D' : t === 'historical' ? '📜' : t === 'projected' ? '🔮' : '📍'}
            </button>
          ))}
        </div>
      </div>

      {/* Toggle connections */}
      <label className="flex items-center gap-2 text-xs text-slate-400 cursor-pointer">
        <input
          type="checkbox"
          checked={showConnections}
          onChange={(e) => setShowConnections(e.target.checked)}
          className="rounded bg-slate-800 border-slate-600"
        />
        Show connections
      </label>

      {/* Legend */}
      <div className="pt-2 border-t border-slate-800">
        <p className="text-xs text-slate-500 mb-1">Drag to rotate • Scroll to zoom</p>
      </div>
    </div>
  );
}

// Main exported component
export function TreeNavigator3D({
  nodes,
  onNodeSelect,
  selectedNodeId,
  temporalFocus: initialTemporalFocus = 'all',
  showConnections: initialShowConnections = true,
}: TreeNavigatorProps) {
  const [temporalFocus, setTemporalFocus] = useState(initialTemporalFocus);
  const [showConnections, setShowConnections] = useState(initialShowConnections);

  return (
    <div className="relative w-full h-full min-h-[400px]">
      <Suspense fallback={<LoadingFallback />}>
        <Canvas
          camera={{ position: [0, 5, 15], fov: 60 }}
          className="bg-slate-950"
          dpr={[1, 2]}
        >
          <TreeScene
            nodes={nodes}
            onNodeSelect={onNodeSelect}
            selectedNodeId={selectedNodeId}
            temporalFocus={temporalFocus}
            showConnections={showConnections}
          />
        </Canvas>
      </Suspense>

      {/* Control overlay */}
      <ControlPanel
        temporalFocus={temporalFocus}
        setTemporalFocus={setTemporalFocus}
        showConnections={showConnections}
        setShowConnections={setShowConnections}
      />

      {/* Instructions */}
      <div className="absolute bottom-4 right-4 text-xs text-slate-400 bg-slate-900/90 backdrop-blur-sm rounded-lg px-3 py-2 border border-slate-700">
        <div className="font-medium text-slate-300 mb-1">Controls</div>
        <div>WASD = Move • Mouse = Look • Space/Shift = Up/Down</div>
        <div className="text-slate-500 mt-1">Click canvas to enable mouse look • ESC to release</div>
      </div>
    </div>
  );
}

export default TreeNavigator3D;

// Helper to convert flat briefings to tree nodes
export function briefingsToTreeNodes(
  briefings: Record<string, string>,
  metadata?: { timestamp?: string; overallRisk?: string }
): IntelNode[] {
  const now = metadata?.timestamp ? new Date(metadata.timestamp) : new Date();
  const nodes: IntelNode[] = [];

  // Executive summary as root
  if (briefings.summary) {
    nodes.push({
      id: 'executive-summary',
      label: 'Executive Summary',
      category: 'executive',
      content: briefings.summary,
      timestamp: now,
      temporalState: 'current',
      confidence: 0.95,
      risk: (metadata?.overallRisk as IntelNode['risk']) || 'moderate',
    });
  }

  // Map briefing categories
  const categoryMap: Array<{ key: string; category: IntelCategory; label: string }> = [
    { key: 'political', category: 'political', label: 'Political Landscape' },
    { key: 'economic', category: 'economic', label: 'Economic Outlook' },
    { key: 'security', category: 'security', label: 'Security Posture' },
    { key: 'military', category: 'military', label: 'Military Activity' },
    { key: 'cyber', category: 'cyber', label: 'Cyber Threats' },
    { key: 'terrorism', category: 'terrorism', label: 'Terrorism Risk' },
    { key: 'health', category: 'health', label: 'Health Indicators' },
    { key: 'scitech', category: 'scitech', label: 'Tech Developments' },
    { key: 'resources', category: 'resources', label: 'Resource Status' },
    { key: 'space', category: 'space', label: 'Space Domain' },
    { key: 'emerging', category: 'emerging', label: 'Emerging Trends' },
  ];

  categoryMap.forEach(({ key, category, label }) => {
    if (briefings[key]) {
      nodes.push({
        id: `${category}-current`,
        label,
        category,
        content: briefings[key],
        timestamp: now,
        temporalState: 'current',
        confidence: 0.8 + Math.random() * 0.15,
        parentId: 'executive-summary',
      });
    }
  });

  // Add NSM as projected future node
  if (briefings.nsm) {
    nodes.push({
      id: 'nsm-projection',
      label: 'Next Strategic Move',
      category: 'executive',
      content: briefings.nsm,
      timestamp: new Date(now.getTime() + 30 * 24 * 60 * 60 * 1000), // 30 days out
      temporalState: 'projected',
      confidence: 0.7,
      parentId: 'executive-summary',
    });
  }

  return nodes;
}
