import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  getBezierPath,
  MiniMap,
  Panel,
  useNodesState,
  useEdgesState,
  useReactFlow,
  addEdge,
  type ColorMode,
  type OnConnect,
  type Connection,
  type Node,
  type Edge,
  type EdgeProps,
  type Viewport,
} from "@xyflow/react";

import NodeSidebar, { NODE_CATEGORIES } from "./NodeSidebar";
import NodeConfigPanel from "./NodeConfigPanel";
import WorkflowNode from "./WorkflowNode";
import { getHandlesForNodeType, HANDLE_COLORS, areTypesCompatible, type HandleDataType } from "./handleTypes";

import "@xyflow/react/dist/style.css";

/* Inject edge animation styles */
const EDGE_ANIMATION_CSS = `
@keyframes edgeFlowDash {
  from { stroke-dashoffset: 20; }
  to { stroke-dashoffset: 0; }
}
.edge-flow-animation {
  animation: edgeFlowDash 0.6s linear infinite;
}
.react-flow__edge.invalid-connection path {
  stroke: #ef4444 !important;
  stroke-dasharray: 4 4;
}
/* Relationships mode: fade unrelated nodes and edges */
.relationships-mode .react-flow__node {
  opacity: 0.2;
  transition: opacity 200ms ease;
}
.relationships-mode .react-flow__node.rel-highlighted {
  opacity: 1;
}
.relationships-mode .react-flow__edge {
  opacity: 0.2;
  transition: opacity 200ms ease;
}
.relationships-mode .react-flow__edge.rel-highlighted {
  opacity: 1;
}
`;

if (typeof document !== "undefined") {
  const id = "orbiter-edge-anim";
  if (!document.getElementById(id)) {
    const style = document.createElement("style");
    style.id = id;
    style.textContent = EDGE_ANIMATION_CSS;
    document.head.appendChild(style);
  }
}

/* ------------------------------------------------------------------ */
/* Custom node types                                                    */
/* ------------------------------------------------------------------ */

const nodeTypes = { workflow: WorkflowNode };

/* ------------------------------------------------------------------ */
/* Custom edge with handle-type coloring                                */
/* ------------------------------------------------------------------ */

interface TypedEdgeData {
  color?: string;
  animated?: boolean;
  label?: string;
}

function TypedBezierEdge(props: EdgeProps) {
  const edgeData = props.data as TypedEdgeData | undefined;
  const color = edgeData?.color ?? "var(--zen-muted, #999)";
  const isAnimated = edgeData?.animated ?? false;
  const dataLabel = edgeData?.label ?? "data";
  const [hovered, setHovered] = useState(false);

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX: props.sourceX,
    sourceY: props.sourceY,
    sourcePosition: props.sourcePosition,
    targetX: props.targetX,
    targetY: props.targetY,
    targetPosition: props.targetPosition,
  });

  return (
    <g
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* Invisible wide path for easier hover detection */}
      <path
        d={edgePath}
        fill="none"
        stroke="transparent"
        strokeWidth={16}
      />
      {/* Visible edge */}
      <path
        d={edgePath}
        fill="none"
        stroke={color}
        strokeWidth={2}
        className={isAnimated ? "edge-animated" : undefined}
        markerEnd={props.markerEnd}
      />
      {/* Animated overlay when executing */}
      {isAnimated && (
        <path
          d={edgePath}
          fill="none"
          stroke={color}
          strokeWidth={3}
          strokeDasharray="6 4"
          className="edge-flow-animation"
          style={{ opacity: 0.8 }}
        />
      )}
      {/* Hover label */}
      {hovered && (
        <foreignObject
          x={labelX - 40}
          y={labelY - 14}
          width={80}
          height={28}
          style={{ pointerEvents: "none", overflow: "visible" }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              padding: "3px 8px",
              borderRadius: 6,
              fontSize: 10,
              fontWeight: 500,
              fontFamily: "'Bricolage Grotesque', sans-serif",
              background: "var(--zen-paper, #f2f0e3)",
              border: `1px solid ${color}`,
              color: "var(--zen-dark, #2e2e2e)",
              boxShadow: "0 1px 4px rgba(0,0,0,0.12)",
              whiteSpace: "nowrap",
            }}
          >
            {dataLabel}
          </div>
        </foreignObject>
      )}
    </g>
  );
}

const edgeTypes = { typed: TypedBezierEdge };

const SAVE_DEBOUNCE_MS = 2000;

const GRID_SIZE = 20;
const SNAP_GRID: [number, number] = [GRID_SIZE, GRID_SIZE];

/* ------------------------------------------------------------------ */
/* Theme hook                                                          */
/* ------------------------------------------------------------------ */

function useThemeColorMode(): ColorMode {
  const [colorMode, setColorMode] = useState<ColorMode>(() => {
    if (typeof document === "undefined") return "light";
    return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  });

  useEffect(() => {
    const observer = new MutationObserver(() => {
      const theme = document.documentElement.dataset.theme;
      setColorMode(theme === "dark" ? "dark" : "light");
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    return () => observer.disconnect();
  }, []);

  return colorMode;
}

/* ------------------------------------------------------------------ */
/* Undo / Redo history                                                 */
/* ------------------------------------------------------------------ */

interface HistoryEntry {
  nodes: Node[];
  edges: Edge[];
}

const MAX_HISTORY = 50;

function useUndoRedo(nodes: Node[], edges: Edge[]) {
  const past = useRef<HistoryEntry[]>([]);
  const future = useRef<HistoryEntry[]>([]);
  const skipRecord = useRef(false);

  /** Record current state before a change. */
  const record = useCallback(() => {
    if (skipRecord.current) {
      skipRecord.current = false;
      return;
    }
    past.current = [
      ...past.current.slice(-(MAX_HISTORY - 1)),
      { nodes: structuredClone(nodes), edges: structuredClone(edges) },
    ];
    future.current = [];
  }, [nodes, edges]);

  const canUndo = past.current.length > 0;
  const canRedo = future.current.length > 0;

  return { past, future, record, canUndo, canRedo, skipRecord };
}

/* ------------------------------------------------------------------ */
/* Toolbar button                                                      */
/* ------------------------------------------------------------------ */

function ToolbarButton({
  onClick,
  title,
  disabled,
  active,
  children,
}: {
  onClick: () => void;
  title: string;
  disabled?: boolean;
  active?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      disabled={disabled}
      className="nodrag nopan"
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: 32,
        height: 32,
        border: "none",
        borderRadius: 6,
        background: active
          ? "var(--zen-coral, #F76F53)"
          : "transparent",
        color: active
          ? "#fff"
          : disabled
            ? "var(--zen-muted, #999)"
            : "var(--zen-dark, #2e2e2e)",
        cursor: disabled ? "default" : "pointer",
        opacity: disabled ? 0.4 : 1,
        transition: "background 150ms, color 150ms, opacity 150ms",
        padding: 0,
      }}
    >
      {children}
    </button>
  );
}

function Separator() {
  return (
    <div
      style={{
        width: 1,
        height: 20,
        background: "var(--zen-muted, #ccc)",
        opacity: 0.3,
        margin: "0 2px",
      }}
    />
  );
}

/* ------------------------------------------------------------------ */
/* SVG icons (inline, 18x18)                                           */
/* ------------------------------------------------------------------ */

const icons = {
  zoomIn: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /><line x1="11" y1="8" x2="11" y2="14" /><line x1="8" y1="11" x2="14" y2="11" />
    </svg>
  ),
  zoomOut: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /><line x1="8" y1="11" x2="14" y2="11" />
    </svg>
  ),
  fitView: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M15 3h6v6" /><path d="M9 21H3v-6" /><path d="M21 3l-7 7" /><path d="M3 21l7-7" />
    </svg>
  ),
  lock: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="11" width="18" height="11" rx="2" /><path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  ),
  unlock: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="11" width="18" height="11" rx="2" /><path d="M7 11V7a5 5 0 0 1 9.9-1" />
    </svg>
  ),
  undo: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
    </svg>
  ),
  redo: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="23 4 23 10 17 10" /><path d="M20.49 15a9 9 0 1 1-2.13-9.36L23 10" />
    </svg>
  ),
  save: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" /><polyline points="17 21 17 13 7 13 7 21" /><polyline points="7 3 7 8 15 8" />
    </svg>
  ),
  export: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  ),
};

/* ------------------------------------------------------------------ */
/* Detect macOS for shortcut labels                                    */
/* ------------------------------------------------------------------ */

const isMac =
  typeof navigator !== "undefined" && /Mac|iPod|iPhone|iPad/.test(navigator.userAgent);
const mod = isMac ? "\u2318" : "Ctrl+";

/* ------------------------------------------------------------------ */
/* Auto-save hook                                                      */
/* ------------------------------------------------------------------ */

type SaveStatus = "saved" | "saving" | "unsaved";

function useAutoSave(
  workflowId: string | undefined,
  nodes: Node[],
  edges: Edge[],
  viewportRef: React.RefObject<Viewport>,
  loaded: boolean,
) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const savingRef = useRef(false);
  const [saveStatus, setSaveStatus] = useState<SaveStatus>("saved");

  const save = useCallback(() => {
    if (!workflowId || savingRef.current || !loaded) return;
    savingRef.current = true;
    setSaveStatus("saving");
    fetch(`/api/workflows/${workflowId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        nodes_json: JSON.stringify(nodes),
        edges_json: JSON.stringify(edges),
        viewport_json: JSON.stringify(viewportRef.current),
      }),
    })
      .then((res) => {
        setSaveStatus(res.ok ? "saved" : "unsaved");
      })
      .catch(() => {
        setSaveStatus("unsaved");
      })
      .finally(() => {
        savingRef.current = false;
      });
  }, [workflowId, nodes, edges, viewportRef, loaded]);

  const scheduleSave = useCallback(() => {
    if (!workflowId || !loaded) return;
    setSaveStatus("unsaved");
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(save, SAVE_DEBOUNCE_MS);
  }, [workflowId, save, loaded]);

  /** Flush pending debounce and save immediately. */
  const saveNow = useCallback(() => {
    if (!workflowId || !loaded) return;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = null;
    save();
  }, [workflowId, loaded, save]);

  /* Cleanup on unmount */
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  return { scheduleSave, saveNow, saveStatus };
}

/* ------------------------------------------------------------------ */
/* Relationships mode — BFS to find upstream/downstream nodes          */
/* ------------------------------------------------------------------ */

interface RelationshipSets {
  upstream: Set<string>;
  downstream: Set<string>;
  connectedEdges: Set<string>;
}

function computeRelationships(
  rootId: string,
  edges: Edge[],
): RelationshipSets {
  const upstream = new Set<string>();
  const downstream = new Set<string>();
  const connectedEdges = new Set<string>();

  // Build adjacency lists
  const childrenOf = new Map<string, { nodeId: string; edgeId: string }[]>();
  const parentsOf = new Map<string, { nodeId: string; edgeId: string }[]>();
  for (const e of edges) {
    if (!childrenOf.has(e.source)) childrenOf.set(e.source, []);
    childrenOf.get(e.source)!.push({ nodeId: e.target, edgeId: e.id });
    if (!parentsOf.has(e.target)) parentsOf.set(e.target, []);
    parentsOf.get(e.target)!.push({ nodeId: e.source, edgeId: e.id });
  }

  // BFS upstream (ancestors)
  const queue: string[] = [rootId];
  const visited = new Set<string>([rootId]);
  while (queue.length > 0) {
    const current = queue.shift()!;
    for (const { nodeId, edgeId } of parentsOf.get(current) ?? []) {
      connectedEdges.add(edgeId);
      if (!visited.has(nodeId)) {
        visited.add(nodeId);
        upstream.add(nodeId);
        queue.push(nodeId);
      }
    }
  }

  // BFS downstream (descendants)
  const queue2: string[] = [rootId];
  const visited2 = new Set<string>([rootId]);
  while (queue2.length > 0) {
    const current = queue2.shift()!;
    for (const { nodeId, edgeId } of childrenOf.get(current) ?? []) {
      connectedEdges.add(edgeId);
      if (!visited2.has(nodeId)) {
        visited2.add(nodeId);
        downstream.add(nodeId);
        queue2.push(nodeId);
      }
    }
  }

  return { upstream, downstream, connectedEdges };
}

/* ------------------------------------------------------------------ */
/* Canvas flow component                                               */
/* ------------------------------------------------------------------ */

function CanvasFlow({ workflowId }: { workflowId?: string }) {
  const colorMode = useThemeColorMode();
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const { zoomIn, zoomOut, fitView, setViewport, screenToFlowPosition } = useReactFlow();
  const [locked, setLocked] = useState(false);
  const [loaded, setLoaded] = useState(!workflowId);
  const viewportRef = useRef<Viewport>({ x: 0, y: 0, zoom: 1 });
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [relationshipNodeId, setRelationshipNodeId] = useState<string | null>(null);

  /* History tracking */
  const { past, future, record, canUndo, canRedo, skipRecord } =
    useUndoRedo(nodes, edges);

  /* Workflow metadata (name, description) */
  const [workflowName, setWorkflowName] = useState("");
  const [workflowDescription, setWorkflowDescription] = useState("");
  const [metaEditing, setMetaEditing] = useState(false);
  const metaTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /** Save name/description to backend (debounced). */
  const saveMetadata = useCallback(
    (name: string, description: string) => {
      if (!workflowId) return;
      if (metaTimerRef.current) clearTimeout(metaTimerRef.current);
      metaTimerRef.current = setTimeout(() => {
        fetch(`/api/workflows/${workflowId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, description }),
        });
      }, 800);
    },
    [workflowId],
  );

  /* Auto-save */
  const { scheduleSave, saveNow, saveStatus } = useAutoSave(workflowId, nodes, edges, viewportRef, loaded);

  /* Export workflow as JSON download */
  const exportWorkflow = useCallback(() => {
    if (!workflowId) return;
    fetch(`/api/workflows/${workflowId}/export`, { method: "POST" })
      .then((res) => {
        if (!res.ok) throw new Error("Export failed");
        return res.json();
      })
      .then((data) => {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = (workflowName || "workflow").replace(/\s+/g, "_").toLowerCase() + ".json";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      })
      .catch((err) => {
        alert("Export error: " + err.message);
      });
  }, [workflowId, workflowName]);

  /* Load canvas state from backend */
  useEffect(() => {
    if (!workflowId) return;
    fetch(`/api/workflows/${workflowId}`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load workflow");
        return res.json();
      })
      .then((data) => {
        /* Store workflow metadata */
        setWorkflowName(data.name ?? "");
        setWorkflowDescription(data.description ?? "");

        const rawNodes: Node[] = JSON.parse(data.nodes_json || "[]");
        const rawEdges: Edge[] = JSON.parse(data.edges_json || "[]");
        const vp: Viewport = JSON.parse(
          data.viewport_json || '{"x":0,"y":0,"zoom":1}',
        );
        /* Migrate older nodes to workflow type */
        const loadedNodes = rawNodes.map((n) =>
          n.type === "default" && (n.data as { nodeType?: string }).nodeType
            ? { ...n, type: "workflow" }
            : n,
        );
        /* Migrate edges to typed Bezier and add color + label data */
        const nodeMap = new Map(loadedNodes.map((n) => [n.id, n]));
        const loadedEdges = rawEdges.map((e) => {
          if (e.type === "typed" && (e.data as { color?: string } | undefined)?.color) return e;
          const srcNode = nodeMap.get(e.source);
          let color = HANDLE_COLORS.any;
          let label: string = "any";
          if (srcNode) {
            const nt = (srcNode.data as { nodeType?: string }).nodeType ?? "default";
            const handles = getHandlesForNodeType(nt);
            const h = handles.find((h) => h.id === (e.sourceHandle ?? "output"));
            if (h) {
              color = HANDLE_COLORS[h.dataType];
              label = h.dataType;
            }
          }
          return { ...e, type: "typed" as const, data: { ...((e.data ?? {}) as Record<string, unknown>), color, label } };
        });
        setNodes(loadedNodes);
        setEdges(loadedEdges);
        viewportRef.current = vp;
        setViewport(vp);
        setLoaded(true);
      })
      .catch(() => {
        setLoaded(true);
      });
  }, [workflowId, setNodes, setEdges, setViewport]);

  /* Trigger auto-save on node/edge changes */
  useEffect(() => {
    if (loaded) scheduleSave();
  }, [nodes, edges, loaded, scheduleSave]);

  /* Track viewport changes */
  const onMoveEnd = useCallback(
    (_event: unknown, vp: Viewport) => {
      viewportRef.current = vp;
      scheduleSave();
    },
    [scheduleSave],
  );

  /* Compute relationship sets when in relationships mode */
  const relationships = useMemo(() => {
    if (!relationshipNodeId) return null;
    return computeRelationships(relationshipNodeId, edges);
  }, [relationshipNodeId, edges]);

  /* Apply relationship classes to nodes and edges */
  const displayNodes = useMemo(() => {
    if (!relationships || !relationshipNodeId) return nodes;
    return nodes.map((n) => {
      const isRoot = n.id === relationshipNodeId;
      const isUpstream = relationships.upstream.has(n.id);
      const isDownstream = relationships.downstream.has(n.id);
      const isHighlighted = isRoot || isUpstream || isDownstream;
      const tint = isRoot ? "root" : isUpstream ? "upstream" : isDownstream ? "downstream" : null;
      return {
        ...n,
        className: isHighlighted ? "rel-highlighted" : undefined,
        data: { ...n.data, _relTint: tint },
      };
    });
  }, [nodes, relationships, relationshipNodeId]);

  const displayEdges = useMemo(() => {
    if (!relationships) return edges;
    return edges.map((e) => ({
      ...e,
      className: relationships.connectedEdges.has(e.id) ? "rel-highlighted" : undefined,
    }));
  }, [edges, relationships]);

  /* Node selection — open config panel, Shift+click for relationships mode */
  const onNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
    if (_event.shiftKey) {
      setRelationshipNodeId((prev) => (prev === node.id ? null : node.id));
      return;
    }
    setSelectedNodeId(node.id);
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
    setRelationshipNodeId(null);
  }, []);

  /* Update a node's data (used by config panel) */
  const handleNodeDataUpdate = useCallback(
    (id: string, data: Record<string, unknown>) => {
      setNodes((nds) =>
        nds.map((n) => (n.id === id ? { ...n, data } : n)),
      );
    },
    [setNodes],
  );

  /* Derive selected node object from current nodes */
  const selectedNode = selectedNodeId
    ? nodes.find((n) => n.id === selectedNodeId) ?? null
    : null;

  /* Resolve handle data type for a node+handle pair */
  const getHandleDataType = useCallback(
    (nodeId: string, handleId: string | null, fallbackType: "source" | "target"): HandleDataType => {
      const node = nodes.find((n) => n.id === nodeId);
      if (!node) return "any";
      const nodeType = (node.data as { nodeType?: string }).nodeType ?? "default";
      const handles = getHandlesForNodeType(nodeType);
      const defaultId = fallbackType === "source" ? "output" : "input";
      const handle = handles.find((h) => h.id === (handleId ?? defaultId));
      return handle?.dataType ?? "any";
    },
    [nodes],
  );

  /* Look up the data type of a source handle for edge coloring */
  const getSourceHandleColor = useCallback(
    (sourceId: string, sourceHandle: string | null): string => {
      const dt = getHandleDataType(sourceId, sourceHandle, "source");
      return HANDLE_COLORS[dt];
    },
    [getHandleDataType],
  );

  /* Validate connections: check data type compatibility */
  const isValidConnection = useCallback(
    (connection: Edge | Connection): boolean => {
      const sourceType = getHandleDataType(connection.source, connection.sourceHandle ?? null, "source");
      const targetType = getHandleDataType(connection.target, connection.targetHandle ?? null, "target");
      return areTypesCompatible(sourceType, targetType);
    },
    [getHandleDataType],
  );

  /* Record state before connection changes */
  const onConnect: OnConnect = useCallback(
    (params) => {
      record();
      const color = getSourceHandleColor(params.source, params.sourceHandle ?? null);
      const sourceDataType = getHandleDataType(params.source, params.sourceHandle ?? null, "source");
      setEdges((eds) =>
        addEdge(
          { ...params, type: "typed", data: { color, label: sourceDataType } },
          eds,
        ),
      );
    },
    [setEdges, record, getSourceHandleColor, getHandleDataType],
  );

  /* Wrap onNodesChange to record history on structural changes */
  const handleNodesChange = useCallback(
    (changes: Parameters<typeof onNodesChange>[0]) => {
      const hasStructural = changes.some(
        (c) => c.type === "remove" || c.type === "add",
      );
      if (hasStructural) record();
      onNodesChange(changes);
    },
    [onNodesChange, record],
  );

  const handleEdgesChange = useCallback(
    (changes: Parameters<typeof onEdgesChange>[0]) => {
      const hasStructural = changes.some(
        (c) => c.type === "remove" || c.type === "add",
      );
      if (hasStructural) record();
      onEdgesChange(changes);
    },
    [onEdgesChange, record],
  );

  /* Undo */
  const undo = useCallback(() => {
    if (past.current.length === 0) return;
    const prev = past.current.pop()!;
    future.current.push({
      nodes: structuredClone(nodes),
      edges: structuredClone(edges),
    });
    skipRecord.current = true;
    setNodes(prev.nodes);
    setEdges(prev.edges);
  }, [nodes, edges, past, future, skipRecord, setNodes, setEdges]);

  /* Redo */
  const redo = useCallback(() => {
    if (future.current.length === 0) return;
    const next = future.current.pop()!;
    past.current.push({
      nodes: structuredClone(nodes),
      edges: structuredClone(edges),
    });
    skipRecord.current = true;
    setNodes(next.nodes);
    setEdges(next.edges);
  }, [nodes, edges, past, future, skipRecord, setNodes, setEdges]);

  /* Select all */
  const selectAll = useCallback(() => {
    setNodes((nds) => nds.map((n) => ({ ...n, selected: true })));
    setEdges((eds) => eds.map((e) => ({ ...e, selected: true })));
  }, [setNodes, setEdges]);

  /* Delete selected */
  const deleteSelected = useCallback(() => {
    const selectedNodes = nodes.filter((n) => n.selected);
    const selectedEdges = edges.filter((e) => e.selected);
    if (selectedNodes.length === 0 && selectedEdges.length === 0) return;
    record();
    const nodeIds = new Set(selectedNodes.map((n) => n.id));
    /* Close config panel if the selected node is being deleted */
    if (selectedNodeId && nodeIds.has(selectedNodeId)) {
      setSelectedNodeId(null);
    }
    setNodes((nds) => nds.filter((n) => !n.selected));
    setEdges((eds) =>
      eds.filter(
        (e) =>
          !e.selected && !nodeIds.has(e.source) && !nodeIds.has(e.target),
      ),
    );
  }, [nodes, edges, record, setNodes, setEdges, selectedNodeId]);

  /* Keyboard shortcuts */
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const meta = e.metaKey || e.ctrlKey;

      /* Delete / Backspace — remove selected */
      if (e.key === "Delete" || e.key === "Backspace") {
        /* Don't intercept if focus is in an input/textarea */
        const tag = (e.target as HTMLElement)?.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        e.preventDefault();
        deleteSelected();
        return;
      }

      /* Cmd+Z — undo, Cmd+Shift+Z — redo */
      if (meta && e.key === "z") {
        e.preventDefault();
        if (e.shiftKey) {
          redo();
        } else {
          undo();
        }
        return;
      }

      /* Cmd+A — select all */
      if (meta && e.key === "a") {
        const tag = (e.target as HTMLElement)?.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA") return;
        e.preventDefault();
        selectAll();
        return;
      }

      /* Cmd+S — manual save */
      if (meta && e.key === "s") {
        e.preventDefault();
        saveNow();
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [deleteSelected, undo, redo, selectAll, saveNow]);

  /* Force re-render for canUndo/canRedo badge state.
     The refs don't trigger re-renders, so we use a counter. */
  const [, forceUpdate] = useState(0);
  const tick = useCallback(() => forceUpdate((n) => n + 1), []);

  /* After any node/edge state change, tick to re-check undo/redo. */
  useEffect(() => {
    tick();
  }, [nodes, edges, tick]);

  /* Look up category color for a node type */
  const getNodeColor = useCallback((typeId: string): string => {
    for (const cat of NODE_CATEGORIES) {
      if (cat.types.some((t) => t.id === typeId)) return cat.color;
    }
    return "#999";
  }, []);

  /* Drop handler — create a new node when dragging from sidebar */
  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const nodeType = e.dataTransfer.getData("application/reactflow-type");
      const label = e.dataTransfer.getData("application/reactflow-label");
      if (!nodeType) return;

      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      const color = getNodeColor(nodeType);

      record();
      const newNode: Node = {
        id: `${nodeType}_${Date.now()}`,
        type: "workflow",
        position,
        data: {
          label,
          nodeType,
          categoryColor: color,
        },
      };
      setNodes((nds) => [...nds, newNode]);
    },
    [screenToFlowPosition, record, setNodes, getNodeColor],
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Workflow metadata header */}
      {workflowId && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            padding: "8px 16px",
            borderBottom: "1px solid var(--zen-subtle, #e0ddd0)",
            background: "var(--zen-paper, #f2f0e3)",
            flexShrink: 0,
          }}
        >
          {metaEditing ? (
            <>
              <input
                type="text"
                value={workflowName}
                onChange={(e) => {
                  setWorkflowName(e.target.value);
                  saveMetadata(e.target.value, workflowDescription);
                }}
                onBlur={() => {
                  if (!workflowName.trim()) return;
                  setMetaEditing(false);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") setMetaEditing(false);
                  if (e.key === "Escape") setMetaEditing(false);
                }}
                placeholder="Workflow name"
                autoFocus
                style={{
                  fontSize: 15,
                  fontWeight: 600,
                  fontFamily: "'Bricolage Grotesque', sans-serif",
                  border: "1px solid var(--zen-subtle, #e0ddd0)",
                  borderRadius: 6,
                  padding: "2px 8px",
                  background: "var(--zen-paper, #f2f0e3)",
                  color: "var(--zen-dark, #2e2e2e)",
                  outline: "none",
                  minWidth: 120,
                }}
              />
              <input
                type="text"
                value={workflowDescription}
                onChange={(e) => {
                  setWorkflowDescription(e.target.value);
                  saveMetadata(workflowName, e.target.value);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") setMetaEditing(false);
                  if (e.key === "Escape") setMetaEditing(false);
                }}
                placeholder="Add a description\u2026"
                style={{
                  fontSize: 13,
                  fontFamily: "'Bricolage Grotesque', sans-serif",
                  border: "1px solid var(--zen-subtle, #e0ddd0)",
                  borderRadius: 6,
                  padding: "2px 8px",
                  background: "var(--zen-paper, #f2f0e3)",
                  color: "var(--zen-muted, #999)",
                  outline: "none",
                  flex: 1,
                  minWidth: 100,
                }}
              />
            </>
          ) : (
            <div
              onClick={() => setMetaEditing(true)}
              style={{ cursor: "pointer", display: "flex", alignItems: "baseline", gap: 8, minWidth: 0 }}
              title="Click to edit name and description"
            >
              <span
                style={{
                  fontSize: 15,
                  fontWeight: 600,
                  fontFamily: "'Bricolage Grotesque', sans-serif",
                  color: "var(--zen-dark, #2e2e2e)",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
              >
                {workflowName || "Untitled Workflow"}
              </span>
              {workflowDescription && (
                <span
                  style={{
                    fontSize: 12,
                    color: "var(--zen-muted, #999)",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                >
                  {workflowDescription}
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {/* Canvas */}
      <div style={{ flex: 1, minHeight: 0 }} className={relationshipNodeId ? "relationships-mode" : undefined}>
    <ReactFlow
      nodes={displayNodes}
      edges={displayEdges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      defaultEdgeOptions={{ type: "typed" }}
      onNodesChange={handleNodesChange}
      onEdgesChange={handleEdgesChange}
      onConnect={onConnect}
      isValidConnection={isValidConnection}
      onMoveEnd={onMoveEnd}
      onDragOver={onDragOver}
      onDrop={onDrop}
      onNodeClick={onNodeClick}
      onPaneClick={onPaneClick}
      colorMode={colorMode}
      snapToGrid
      snapGrid={SNAP_GRID}
      fitView={!workflowId}
      nodesDraggable={!locked}
      nodesConnectable={!locked}
      elementsSelectable={!locked}
      panOnDrag={!locked}
      zoomOnScroll={!locked}
      zoomOnPinch={!locked}
      zoomOnDoubleClick={!locked}
    >
      <Background variant={BackgroundVariant.Dots} gap={GRID_SIZE} size={1.5} />

      {/* Node type sidebar */}
      <NodeSidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed((v) => !v)}
      />

      {/* Custom toolbar */}
      <Panel position="top-center">
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 2,
            padding: "4px 6px",
            borderRadius: 10,
            background: "var(--zen-paper, #f2f0e3)",
            border: "1px solid var(--zen-subtle, #e0ddd0)",
            boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
          }}
        >
          <ToolbarButton onClick={() => zoomIn({ duration: 200 })} title={`Zoom In (${mod}+)`}>
            {icons.zoomIn}
          </ToolbarButton>
          <ToolbarButton onClick={() => zoomOut({ duration: 200 })} title={`Zoom Out (${mod}-)`}>
            {icons.zoomOut}
          </ToolbarButton>
          <ToolbarButton onClick={() => fitView({ padding: 0.2, duration: 300 })} title="Fit View">
            {icons.fitView}
          </ToolbarButton>

          <Separator />

          <ToolbarButton
            onClick={() => setLocked((v) => !v)}
            title={locked ? "Unlock Canvas" : "Lock Canvas"}
            active={locked}
          >
            {locked ? icons.lock : icons.unlock}
          </ToolbarButton>

          <Separator />

          <ToolbarButton onClick={undo} title={`Undo (${mod}Z)`} disabled={!canUndo}>
            {icons.undo}
          </ToolbarButton>
          <ToolbarButton onClick={redo} title={`Redo (${mod}Shift+Z)`} disabled={!canRedo}>
            {icons.redo}
          </ToolbarButton>

          {workflowId && (
            <>
              <Separator />
              <ToolbarButton onClick={saveNow} title={`Save (${mod}S)`}>
                {icons.save}
              </ToolbarButton>
              <span
                className="nodrag nopan"
                style={{
                  fontSize: 11,
                  fontWeight: 500,
                  fontFamily: "'Bricolage Grotesque', sans-serif",
                  color:
                    saveStatus === "saved"
                      ? "var(--zen-green, #63f78b)"
                      : saveStatus === "saving"
                        ? "var(--zen-muted, #999)"
                        : "var(--zen-coral, #F76F53)",
                  padding: "0 4px",
                  whiteSpace: "nowrap",
                }}
              >
                {saveStatus === "saved"
                  ? "Saved"
                  : saveStatus === "saving"
                    ? "Saving\u2026"
                    : "Unsaved changes"}
              </span>
              <ToolbarButton onClick={exportWorkflow} title="Export JSON">
                {icons.export}
              </ToolbarButton>
            </>
          )}
        </div>
      </Panel>

      <MiniMap
        position="bottom-right"
        pannable
        zoomable
        style={{ width: 160, height: 120 }}
      />

      {/* Node config panel */}
      <NodeConfigPanel
        node={selectedNode}
        onClose={() => setSelectedNodeId(null)}
        onNodeUpdate={handleNodeDataUpdate}
      />
    </ReactFlow>
      </div>
    </div>
  );
}

export default function CanvasIsland({ workflowId }: { workflowId?: string }) {
  return (
    <ReactFlowProvider>
      <CanvasFlow workflowId={workflowId} />
    </ReactFlowProvider>
  );
}
