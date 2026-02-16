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
} from "@xyflow/react";

import "@xyflow/react/dist/style.css";

const initialNodes: Node[] = [];
const initialEdges: Edge[] = [];

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
/* Canvas flow component                                               */
/* ------------------------------------------------------------------ */

function CanvasFlow() {
  const colorMode = useThemeColorMode();
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const { zoomIn, zoomOut, fitView } = useReactFlow();
  const [locked, setLocked] = useState(false);

  /* History tracking */
  const { past, future, record, canUndo, canRedo, skipRecord } =
    useUndoRedo(nodes, edges);

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

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={handleNodesChange}
      onEdgesChange={handleEdgesChange}
      onConnect={onConnect}
      colorMode={colorMode}
      snapToGrid
      snapGrid={SNAP_GRID}
      fitView
      nodesDraggable={!locked}
      nodesConnectable={!locked}
      elementsSelectable={!locked}
      panOnDrag={!locked}
      zoomOnScroll={!locked}
      zoomOnPinch={!locked}
      zoomOnDoubleClick={!locked}
    >
      <Background variant={BackgroundVariant.Dots} gap={GRID_SIZE} size={1.5} />

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

export default function CanvasIsland() {
  return (
    <ReactFlowProvider>
      <CanvasFlow />
    </ReactFlowProvider>
  );
}
