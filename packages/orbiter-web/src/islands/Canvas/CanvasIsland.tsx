import { useCallback, useEffect, useRef, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  MiniMap,
  Panel,
  useNodesState,
  useEdgesState,
  useReactFlow,
  addEdge,
  type ColorMode,
  type OnConnect,
  type Node,
  type Edge,
  type Viewport,
} from "@xyflow/react";

import NodeSidebar, { NODE_CATEGORIES } from "./NodeSidebar";

import "@xyflow/react/dist/style.css";

const SAVE_DEBOUNCE_MS = 500;

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

function useAutoSave(
  workflowId: string | undefined,
  nodes: Node[],
  edges: Edge[],
  viewportRef: React.RefObject<Viewport>,
  loaded: boolean,
) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const savingRef = useRef(false);

  const save = useCallback(() => {
    if (!workflowId || savingRef.current || !loaded) return;
    savingRef.current = true;
    fetch(`/api/workflows/${workflowId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        nodes_json: JSON.stringify(nodes),
        edges_json: JSON.stringify(edges),
        viewport_json: JSON.stringify(viewportRef.current),
      }),
    }).finally(() => {
      savingRef.current = false;
    });
  }, [workflowId, nodes, edges, viewportRef, loaded]);

  const scheduleSave = useCallback(() => {
    if (!workflowId || !loaded) return;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(save, SAVE_DEBOUNCE_MS);
  }, [workflowId, save, loaded]);

  /* Cleanup on unmount */
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  return { scheduleSave };
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

  /* History tracking */
  const { past, future, record, canUndo, canRedo, skipRecord } =
    useUndoRedo(nodes, edges);

  /* Auto-save */
  const { scheduleSave } = useAutoSave(workflowId, nodes, edges, viewportRef, loaded);

  /* Load canvas state from backend */
  useEffect(() => {
    if (!workflowId) return;
    fetch(`/api/workflows/${workflowId}`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load workflow");
        return res.json();
      })
      .then((data) => {
        const loadedNodes: Node[] = JSON.parse(data.nodes_json || "[]");
        const loadedEdges: Edge[] = JSON.parse(data.edges_json || "[]");
        const vp: Viewport = JSON.parse(
          data.viewport_json || '{"x":0,"y":0,"zoom":1}',
        );
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

  /* Record state before connection changes */
  const onConnect: OnConnect = useCallback(
    (params) => {
      record();
      setEdges((eds) => addEdge(params, eds));
    },
    [setEdges, record],
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
    setNodes((nds) => nds.filter((n) => !n.selected));
    setEdges((eds) =>
      eds.filter(
        (e) =>
          !e.selected && !nodeIds.has(e.source) && !nodeIds.has(e.target),
      ),
    );
  }, [nodes, edges, record, setNodes, setEdges]);

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
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [deleteSelected, undo, redo, selectAll]);

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
        type: "default",
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
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={handleNodesChange}
      onEdgesChange={handleEdgesChange}
      onConnect={onConnect}
      onMoveEnd={onMoveEnd}
      onDragOver={onDragOver}
      onDrop={onDrop}
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
        </div>
      </Panel>

      <MiniMap
        position="bottom-right"
        pannable
        zoomable
        style={{ width: 160, height: 120 }}
      />
    </ReactFlow>
  );
}

export default function CanvasIsland({ workflowId }: { workflowId?: string }) {
  return (
    <ReactFlowProvider>
      <CanvasFlow workflowId={workflowId} />
    </ReactFlowProvider>
  );
}
