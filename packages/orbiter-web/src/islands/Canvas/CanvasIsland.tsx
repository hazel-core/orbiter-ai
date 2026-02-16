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
import NodeInspectionPanel from "./NodeInspectionPanel";
import ValidationPanel from "./ValidationPanel";
import WorkflowNode from "./WorkflowNode";
import { getHandlesForNodeType, HANDLE_COLORS, areTypesCompatible, type HandleDataType } from "./handleTypes";
import { validateWorkflow, type ValidationResult } from "./validation";

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
.react-flow__edge.cycle-edge path {
  stroke: #ef4444 !important;
  stroke-width: 3px;
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
/* Execution status: pulsing coral border for running nodes */
@keyframes execPulse {
  0%, 100% { box-shadow: 0 0 0 2px rgba(247, 111, 83, 0.3); }
  50% { box-shadow: 0 0 0 6px rgba(247, 111, 83, 0.15); }
}
.exec-running {
  animation: execPulse 1.5s ease-in-out infinite;
}
/* Debug paused: pulsing purple border */
@keyframes debugPausePulse {
  0%, 100% { box-shadow: 0 0 0 4px rgba(168, 85, 247, 0.35); }
  50% { box-shadow: 0 0 0 8px rgba(168, 85, 247, 0.15); }
}
.debug-paused {
  animation: debugPausePulse 1.5s ease-in-out infinite;
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
const VALIDATION_DEBOUNCE_MS = 1000;

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
  validate: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  ),
  play: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" stroke="none">
      <polygon points="6,3 20,12 6,21" />
    </svg>
  ),
  stop: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" stroke="none">
      <rect x="5" y="5" width="14" height="14" rx="2" />
    </svg>
  ),
  debug: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a4 4 0 0 0-4 4v2" /><path d="M16 8V6a4 4 0 0 0-4-4" />
      <rect x="6" y="8" width="12" height="12" rx="3" />
      <line x1="6" y1="14" x2="4" y2="14" /><line x1="20" y1="14" x2="18" y2="14" />
      <line x1="12" y1="8" x2="12" y2="20" />
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
/* Execution state                                                     */
/* ------------------------------------------------------------------ */

type NodeExecStatus = "running" | "completed" | "failed";
type RunStatus = "idle" | "pending" | "running" | "completed" | "failed" | "cancelled";

interface ExecutionState {
  runId: string | null;
  status: RunStatus;
  nodeStatuses: Record<string, NodeExecStatus>;
  startTime: number | null;
  completedCount: number;
  totalNodes: number;
}

const INITIAL_EXEC_STATE: ExecutionState = {
  runId: null,
  status: "idle",
  nodeStatuses: {},
  startTime: null,
  completedCount: 0,
  totalNodes: 0,
};

function useWorkflowExecution(workflowId: string | undefined, totalNodes: number) {
  const [exec, setExec] = useState<ExecutionState>(INITIAL_EXEC_STATE);
  const wsRef = useRef<WebSocket | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  /** Start a workflow run. */
  const startRun = useCallback(async () => {
    if (!workflowId) return;

    setExec({
      runId: null,
      status: "pending",
      nodeStatuses: {},
      startTime: Date.now(),
      completedCount: 0,
      totalNodes,
    });
    setElapsed(0);

    try {
      const res = await fetch(`/api/workflows/${workflowId}/run`, { method: "POST" });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Failed to start run" }));
        setExec((s) => ({ ...s, status: "failed", runId: null }));
        alert(err.detail || "Failed to start run");
        return;
      }
      const { run_id } = await res.json();

      setExec((s) => ({ ...s, runId: run_id, status: "running" }));

      // Start elapsed timer
      timerRef.current = setInterval(() => {
        setElapsed((e) => e + 1);
      }, 1000);

      // Connect WebSocket for live events
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${proto}//${window.location.host}/api/workflows/${workflowId}/runs/${run_id}/stream`);
      wsRef.current = ws;

      ws.onmessage = (ev) => {
        const event = JSON.parse(ev.data);
        if (event.type === "node_started") {
          setExec((s) => ({
            ...s,
            nodeStatuses: { ...s.nodeStatuses, [event.node_id]: "running" },
          }));
        } else if (event.type === "node_completed") {
          setExec((s) => ({
            ...s,
            nodeStatuses: { ...s.nodeStatuses, [event.node_id]: "completed" },
            completedCount: s.completedCount + 1,
          }));
        } else if (event.type === "node_failed") {
          setExec((s) => ({
            ...s,
            nodeStatuses: { ...s.nodeStatuses, [event.node_id]: "failed" },
            completedCount: s.completedCount + 1,
          }));
        } else if (event.type === "execution_completed") {
          setExec((s) => ({
            ...s,
            status: event.status as RunStatus,
          }));
          clearTimer();
          ws.close();
        }
      };

      ws.onerror = () => {
        setExec((s) => ({ ...s, status: "failed" }));
        clearTimer();
      };
    } catch {
      setExec((s) => ({ ...s, status: "failed" }));
      clearTimer();
    }
  }, [workflowId, totalNodes, clearTimer]);

  /** Cancel the current run. */
  const cancelRun = useCallback(async () => {
    if (!workflowId || !exec.runId) return;

    try {
      await fetch(`/api/workflows/${workflowId}/runs/${exec.runId}`, { method: "DELETE" });
    } catch {
      // Ignore cancel errors
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    clearTimer();
    setExec((s) => ({ ...s, status: "cancelled" }));
  }, [workflowId, exec.runId, clearTimer]);

  /** Reset back to idle. */
  const resetExec = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    clearTimer();
    setExec(INITIAL_EXEC_STATE);
    setElapsed(0);
  }, [clearTimer]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
      clearTimer();
    };
  }, [clearTimer]);

  const isRunning = exec.status === "running" || exec.status === "pending";

  return { exec, elapsed, isRunning, startRun, cancelRun, resetExec };
}

/* ------------------------------------------------------------------ */
/* Debug execution state                                               */
/* ------------------------------------------------------------------ */

interface DebugState {
  active: boolean;
  runId: string | null;
  pausedNodeId: string | null;
  variables: Record<string, unknown>;
  breakpoints: Set<string>;
  nodeStatuses: Record<string, NodeExecStatus>;
  completedCount: number;
  totalNodes: number;
  status: "idle" | "pending" | "paused" | "running" | "completed" | "failed" | "cancelled";
}

const INITIAL_DEBUG_STATE: DebugState = {
  active: false,
  runId: null,
  pausedNodeId: null,
  variables: {},
  breakpoints: new Set(),
  nodeStatuses: {},
  completedCount: 0,
  totalNodes: 0,
  status: "idle",
};

function useDebugExecution(workflowId: string | undefined, totalNodes: number) {
  const [dbg, setDbg] = useState<DebugState>(INITIAL_DEBUG_STATE);
  const wsRef = useRef<WebSocket | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  /** Start a debug session. */
  const startDebug = useCallback(async () => {
    if (!workflowId) return;

    setDbg({
      ...INITIAL_DEBUG_STATE,
      active: true,
      status: "pending",
      totalNodes,
    });
    setElapsed(0);

    try {
      const res = await fetch(`/api/workflows/${workflowId}/debug`, { method: "POST" });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Failed to start debug" }));
        setDbg((s) => ({ ...s, status: "failed" }));
        alert(err.detail || "Failed to start debug session");
        return;
      }
      const { run_id } = await res.json();

      setDbg((s) => ({ ...s, runId: run_id, status: "running" }));

      // Start elapsed timer
      timerRef.current = setInterval(() => {
        setElapsed((e) => e + 1);
      }, 1000);

      // Connect debug WebSocket
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${proto}//${window.location.host}/api/workflows/${workflowId}/runs/${run_id}/debug`);
      wsRef.current = ws;

      ws.onmessage = (ev) => {
        const event = JSON.parse(ev.data);
        if (event.type === "debug_paused") {
          setDbg((s) => ({
            ...s,
            status: "paused",
            pausedNodeId: event.node_id,
            variables: event.variables ?? s.variables,
            breakpoints: new Set(event.breakpoints ?? []),
          }));
        } else if (event.type === "node_started") {
          setDbg((s) => ({
            ...s,
            status: "running",
            pausedNodeId: null,
            nodeStatuses: { ...s.nodeStatuses, [event.node_id]: "running" },
          }));
        } else if (event.type === "node_completed") {
          setDbg((s) => ({
            ...s,
            nodeStatuses: { ...s.nodeStatuses, [event.node_id]: "completed" },
            variables: event.variables ?? s.variables,
            completedCount: s.completedCount + 1,
          }));
        } else if (event.type === "node_failed") {
          setDbg((s) => ({
            ...s,
            nodeStatuses: { ...s.nodeStatuses, [event.node_id]: "failed" },
            completedCount: s.completedCount + 1,
          }));
        } else if (event.type === "node_skipped") {
          setDbg((s) => ({
            ...s,
            nodeStatuses: { ...s.nodeStatuses, [event.node_id]: "completed" },
            completedCount: s.completedCount + 1,
          }));
        } else if (event.type === "execution_completed") {
          setDbg((s) => ({
            ...s,
            status: (event.status ?? "completed") as DebugState["status"],
            pausedNodeId: null,
            variables: event.variables ?? s.variables,
          }));
          clearTimer();
          ws.close();
        }
      };

      ws.onerror = () => {
        setDbg((s) => ({ ...s, status: "failed" }));
        clearTimer();
      };
    } catch {
      setDbg((s) => ({ ...s, status: "failed" }));
      clearTimer();
    }
  }, [workflowId, totalNodes, clearTimer]);

  /** Send a debug command. */
  const sendCommand = useCallback((cmd: Record<string, unknown>) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(cmd));
    }
  }, []);

  const continueExec = useCallback(() => sendCommand({ action: "continue" }), [sendCommand]);
  const skipNode = useCallback(() => sendCommand({ action: "skip" }), [sendCommand]);
  const stopDebug = useCallback(() => {
    sendCommand({ action: "stop" });
    clearTimer();
    setDbg((s) => ({ ...s, status: "cancelled", pausedNodeId: null }));
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, [sendCommand, clearTimer]);

  const toggleBreakpoint = useCallback((nodeId: string) => {
    sendCommand({ action: "set_breakpoint", node_id: nodeId });
    setDbg((s) => {
      const next = new Set(s.breakpoints);
      if (next.has(nodeId)) next.delete(nodeId);
      else next.add(nodeId);
      return { ...s, breakpoints: next };
    });
  }, [sendCommand]);

  const setVariable = useCallback((name: string, value: unknown) => {
    sendCommand({ action: "set_variable", name, value });
    setDbg((s) => ({
      ...s,
      variables: { ...s.variables, [name]: value },
    }));
  }, [sendCommand]);

  /** Reset back to idle. */
  const resetDebug = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    clearTimer();
    setDbg(INITIAL_DEBUG_STATE);
    setElapsed(0);
  }, [clearTimer]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
      clearTimer();
    };
  }, [clearTimer]);

  const isDebugging = dbg.active && dbg.status !== "idle";

  return {
    dbg,
    debugElapsed: elapsed,
    isDebugging,
    startDebug,
    continueExec,
    skipNode,
    stopDebug,
    toggleBreakpoint,
    setVariable,
    resetDebug,
  };
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

/* ------------------------------------------------------------------ */
/* Debug variable inspector panel                                      */
/* ------------------------------------------------------------------ */

function DebugVariablePanel({
  variables,
  onSetVariable,
  nodes,
}: {
  variables: Record<string, unknown>;
  onSetVariable: (name: string, value: unknown) => void;
  nodes: Node[];
}) {
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");

  const nodeLabels = useMemo(() => {
    const map: Record<string, string> = {};
    for (const n of nodes) {
      const d = n.data as { label?: string; nodeType?: string };
      map[n.id] = d.label ?? d.nodeType ?? n.id;
    }
    return map;
  }, [nodes]);

  const startEdit = (key: string, val: unknown) => {
    setEditingKey(key);
    setEditValue(typeof val === "string" ? val : JSON.stringify(val));
  };

  const commitEdit = () => {
    if (editingKey === null) return;
    let parsed: unknown = editValue;
    try {
      parsed = JSON.parse(editValue);
    } catch {
      // keep as string
    }
    onSetVariable(editingKey, parsed);
    setEditingKey(null);
    setEditValue("");
  };

  const cancelEdit = () => {
    setEditingKey(null);
    setEditValue("");
  };

  return (
    <div
      style={{
        maxWidth: 480,
        maxHeight: 200,
        overflowY: "auto",
        padding: "8px 12px",
        borderRadius: 8,
        background: "var(--zen-paper, #f2f0e3)",
        border: "1px solid var(--zen-subtle, #e0ddd0)",
        boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
        fontFamily: "'Bricolage Grotesque', sans-serif",
        fontSize: 11,
      }}
    >
      <div
        style={{
          fontWeight: 600,
          fontSize: 11,
          color: "#a855f7",
          marginBottom: 6,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}
      >
        Variables
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <tbody>
          {Object.entries(variables).map(([key, val]) => (
            <tr key={key} style={{ borderBottom: "1px solid var(--zen-subtle, #e0ddd0)" }}>
              <td
                style={{
                  padding: "4px 8px 4px 0",
                  fontWeight: 600,
                  color: "var(--zen-dark, #2e2e2e)",
                  whiteSpace: "nowrap",
                  verticalAlign: "top",
                }}
                title={key}
              >
                {nodeLabels[key] ?? key}
              </td>
              <td style={{ padding: "4px 0", color: "var(--zen-muted, #999)", wordBreak: "break-all" }}>
                {editingKey === key ? (
                  <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                    <input
                      autoFocus
                      value={editValue}
                      onChange={(e) => setEditValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") commitEdit();
                        if (e.key === "Escape") cancelEdit();
                      }}
                      style={{
                        flex: 1,
                        padding: "2px 4px",
                        fontSize: 11,
                        border: "1px solid #a855f7",
                        borderRadius: 4,
                        background: "var(--zen-paper, #f2f0e3)",
                        color: "var(--zen-dark, #2e2e2e)",
                        fontFamily: "monospace",
                        outline: "none",
                      }}
                    />
                    <button
                      onClick={commitEdit}
                      style={{
                        padding: "1px 6px",
                        fontSize: 10,
                        borderRadius: 4,
                        border: "1px solid var(--zen-green, #63f78b)",
                        background: "var(--zen-green, #63f78b)",
                        color: "#fff",
                        cursor: "pointer",
                      }}
                    >
                      OK
                    </button>
                    <button
                      onClick={cancelEdit}
                      style={{
                        padding: "1px 6px",
                        fontSize: 10,
                        borderRadius: 4,
                        border: "1px solid var(--zen-subtle, #e0ddd0)",
                        background: "transparent",
                        color: "var(--zen-dark, #2e2e2e)",
                        cursor: "pointer",
                      }}
                    >
                      Esc
                    </button>
                  </div>
                ) : (
                  <span
                    onClick={() => startEdit(key, val)}
                    title="Click to edit"
                    style={{ cursor: "pointer", fontFamily: "monospace" }}
                  >
                    {typeof val === "string"
                      ? val.length > 80
                        ? val.slice(0, 80) + "\u2026"
                        : val
                      : JSON.stringify(val).slice(0, 80)}
                  </span>
                )}
              </td>
              <td style={{ padding: "4px 0 4px 4px", width: 24 }}>
                <button
                  title="Copy value"
                  onClick={() => {
                    const text = typeof val === "string" ? val : JSON.stringify(val, null, 2);
                    navigator.clipboard.writeText(text);
                  }}
                  style={{
                    padding: 2,
                    border: "none",
                    background: "transparent",
                    cursor: "pointer",
                    color: "var(--zen-muted, #999)",
                    fontSize: 11,
                  }}
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                  </svg>
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
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
  const [inspectedNodeId, setInspectedNodeId] = useState<string | null>(null);
  const [relationshipNodeId, setRelationshipNodeId] = useState<string | null>(null);
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [validationPanelOpen, setValidationPanelOpen] = useState(false);
  const validationTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /* History tracking */
  const { past, future, record, canUndo, canRedo, skipRecord } =
    useUndoRedo(nodes, edges);

  /* Execution state */
  const { exec, elapsed, isRunning, startRun, cancelRun, resetExec } =
    useWorkflowExecution(workflowId, nodes.length);

  /* Debug execution state */
  const {
    dbg, debugElapsed, isDebugging, startDebug, continueExec, skipNode,
    stopDebug, toggleBreakpoint, setVariable, resetDebug,
  } = useDebugExecution(workflowId, nodes.length);

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

  /* Navigate to a node (used by validation panel) */
  const navigateToNode = useCallback(
    (nodeId: string) => {
      const node = nodes.find((n) => n.id === nodeId);
      if (!node) return;
      // Center viewport on the node and select it
      setSelectedNodeId(nodeId);
      fitView({
        nodes: [{ id: nodeId }],
        padding: 0.5,
        duration: 400,
      });
    },
    [nodes, fitView],
  );

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

  /* Debounced validation on canvas changes */
  useEffect(() => {
    if (!loaded) return;
    if (validationTimerRef.current) clearTimeout(validationTimerRef.current);
    validationTimerRef.current = setTimeout(() => {
      setValidationResult(validateWorkflow(nodes, edges));
    }, VALIDATION_DEBOUNCE_MS);
    return () => {
      if (validationTimerRef.current) clearTimeout(validationTimerRef.current);
    };
  }, [nodes, edges, loaded]);

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

  /* Apply relationship classes + validation decorations + execution status to nodes and edges */
  const displayNodes = useMemo(() => {
    return nodes.map((n) => {
      // Relationship mode
      const isRoot = relationshipNodeId ? n.id === relationshipNodeId : false;
      const isUpstream = relationships?.upstream.has(n.id) ?? false;
      const isDownstream = relationships?.downstream.has(n.id) ?? false;
      const isHighlighted = isRoot || isUpstream || isDownstream;
      const tint = isRoot ? "root" : isUpstream ? "upstream" : isDownstream ? "downstream" : null;
      const relClass = relationshipNodeId ? (isHighlighted ? "rel-highlighted" : undefined) : undefined;

      // Validation decorations
      const hasMissingConfig = validationResult?.missingConfig.has(n.id) ?? false;
      const isUnreachable = validationResult?.unreachableNodes.has(n.id) ?? false;
      const disconnectedHandles = validationResult?.disconnectedInputs.get(n.id);
      const disconnectedArr = disconnectedHandles ? [...disconnectedHandles] : [];

      // Execution status (use debug statuses when debugging)
      const execStatus = isDebugging
        ? (dbg.nodeStatuses[n.id] ?? null)
        : (exec.nodeStatuses[n.id] ?? null);
      const isDebugPaused = isDebugging && dbg.pausedNodeId === n.id;
      const hasBreakpoint = isDebugging && dbg.breakpoints.has(n.id);
      const execClass = isDebugPaused
        ? "debug-paused"
        : execStatus === "running"
          ? "exec-running"
          : undefined;
      const cls = [relClass, execClass].filter(Boolean).join(" ") || undefined;

      return {
        ...n,
        className: cls,
        data: {
          ...n.data,
          _relTint: tint,
          _missingConfig: hasMissingConfig,
          _unreachable: isUnreachable,
          _disconnectedInputs: disconnectedArr,
          _execStatus: execStatus,
          _debugPaused: isDebugPaused,
          _hasBreakpoint: hasBreakpoint,
        },
      };
    });
  }, [nodes, relationships, relationshipNodeId, validationResult, exec.nodeStatuses, isDebugging, dbg.nodeStatuses, dbg.pausedNodeId, dbg.breakpoints]);

  const displayEdges = useMemo(() => {
    return edges.map((e) => {
      const relClass = relationships
        ? (relationships.connectedEdges.has(e.id) ? "rel-highlighted" : undefined)
        : undefined;
      const cycleClass = validationResult?.cycleEdges.has(e.id) ? "cycle-edge" : undefined;
      const cls = [relClass, cycleClass].filter(Boolean).join(" ") || undefined;

      // Animate edges where source is completed and target is running
      const nodeStatuses = isDebugging ? dbg.nodeStatuses : exec.nodeStatuses;
      const srcStatus = nodeStatuses[e.source];
      const tgtStatus = nodeStatuses[e.target];
      const execAnimated = srcStatus === "completed" && tgtStatus === "running";

      const edgeData = (e.data ?? {}) as Record<string, unknown>;
      return {
        ...e,
        className: cls,
        data: {
          ...edgeData,
          animated: execAnimated || edgeData.animated,
        },
      };
    });
  }, [edges, relationships, validationResult, exec.nodeStatuses, isDebugging, dbg.nodeStatuses]);

  /* Track whether execution has finished (data available to inspect) */
  const hasExecutionData = exec.runId !== null && !isRunning;

  /* Node selection — open config or inspection panel, Shift+click for relationships mode */
  const onNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
    if (_event.shiftKey) {
      setRelationshipNodeId((prev) => (prev === node.id ? null : node.id));
      return;
    }
    /* Alt+click toggles breakpoint in debug mode */
    if (_event.altKey && isDebugging) {
      toggleBreakpoint(node.id);
      return;
    }
    /* If execution data available and this node was executed, show inspection panel */
    if (hasExecutionData && exec.nodeStatuses[node.id]) {
      setInspectedNodeId(node.id);
      setSelectedNodeId(null);
    } else {
      setSelectedNodeId(node.id);
      setInspectedNodeId(null);
    }
  }, [hasExecutionData, exec.nodeStatuses, isDebugging, toggleBreakpoint]);

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
    setInspectedNodeId(null);
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

  /* Derive inspected node object for execution inspection panel */
  const inspectedNode = inspectedNodeId
    ? nodes.find((n) => n.id === inspectedNodeId) ?? null
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
    /* Close config/inspection panel if the selected node is being deleted */
    if (selectedNodeId && nodeIds.has(selectedNodeId)) {
      setSelectedNodeId(null);
    }
    if (inspectedNodeId && nodeIds.has(inspectedNodeId)) {
      setInspectedNodeId(null);
    }
    setNodes((nds) => nds.filter((n) => !n.selected));
    setEdges((eds) =>
      eds.filter(
        (e) =>
          !e.selected && !nodeIds.has(e.source) && !nodeIds.has(e.target),
      ),
    );
  }, [nodes, edges, record, setNodes, setEdges, selectedNodeId, inspectedNodeId]);

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

          <Separator />
          <ToolbarButton
            onClick={() => setValidationPanelOpen((v) => !v)}
            title="Toggle Validation Panel"
            active={validationPanelOpen}
          >
            {icons.validate}
          </ToolbarButton>
          {validationResult && validationResult.issues.length > 0 && (
            <span
              className="nodrag nopan"
              style={{
                fontSize: 10,
                fontWeight: 600,
                fontFamily: "'Bricolage Grotesque', sans-serif",
                color: validationResult.issues.some((i) => i.severity === "error")
                  ? "#ef4444"
                  : "#f59e0b",
                padding: "0 2px",
              }}
            >
              {validationResult.issues.length}
            </span>
          )}

          {workflowId && (
            <>
              <Separator />
              {isRunning ? (
                <ToolbarButton onClick={cancelRun} title="Cancel Execution">
                  {icons.stop}
                </ToolbarButton>
              ) : isDebugging ? (
                <ToolbarButton onClick={stopDebug} title="Stop Debug">
                  {icons.stop}
                </ToolbarButton>
              ) : (
                <>
                  <ToolbarButton
                    onClick={startRun}
                    title="Run Workflow"
                    disabled={nodes.length === 0}
                  >
                    {icons.play}
                  </ToolbarButton>
                  <ToolbarButton
                    onClick={startDebug}
                    title="Debug Workflow"
                    disabled={nodes.length === 0}
                  >
                    {icons.debug}
                  </ToolbarButton>
                </>
              )}
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

      {/* Validation summary panel */}
      <ValidationPanel
        issues={validationResult?.issues ?? []}
        open={validationPanelOpen}
        onToggle={() => setValidationPanelOpen((v) => !v)}
        onNavigateToNode={navigateToNode}
      />

      {/* Execution status bar */}
      {exec.status !== "idle" && (
        <Panel position="bottom-center">
          <div
            className="nodrag nopan"
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              padding: "8px 16px",
              borderRadius: 10,
              background: "var(--zen-paper, #f2f0e3)",
              border: "1px solid var(--zen-subtle, #e0ddd0)",
              boxShadow: "0 -1px 4px rgba(0,0,0,0.08)",
              fontFamily: "'Bricolage Grotesque', sans-serif",
              fontSize: 12,
              fontWeight: 500,
            }}
          >
            {/* Status indicator dot */}
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background:
                  exec.status === "running" || exec.status === "pending"
                    ? "var(--zen-coral, #F76F53)"
                    : exec.status === "completed"
                      ? "var(--zen-green, #63f78b)"
                      : "#ef4444",
                animation:
                  exec.status === "running" || exec.status === "pending"
                    ? "execPulse 1.5s ease-in-out infinite"
                    : undefined,
                flexShrink: 0,
              }}
            />

            {/* Status label */}
            <span style={{ color: "var(--zen-dark, #2e2e2e)", textTransform: "capitalize" }}>
              {exec.status}
            </span>

            {/* Separator */}
            <div style={{ width: 1, height: 14, background: "var(--zen-muted, #ccc)", opacity: 0.3 }} />

            {/* Elapsed time */}
            <span style={{ color: "var(--zen-muted, #999)" }}>
              {Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, "0")}
            </span>

            {/* Separator */}
            <div style={{ width: 1, height: 14, background: "var(--zen-muted, #ccc)", opacity: 0.3 }} />

            {/* Step count */}
            <span style={{ color: "var(--zen-muted, #999)" }}>
              {exec.completedCount}/{exec.totalNodes} steps
            </span>

            {/* Close/dismiss button when done */}
            {!isRunning && (
              <button
                onClick={resetExec}
                style={{
                  marginLeft: 4,
                  padding: "2px 8px",
                  fontSize: 11,
                  borderRadius: 6,
                  border: "1px solid var(--zen-subtle, #e0ddd0)",
                  background: "transparent",
                  color: "var(--zen-dark, #2e2e2e)",
                  cursor: "pointer",
                  fontFamily: "'Bricolage Grotesque', sans-serif",
                }}
              >
                Dismiss
              </button>
            )}
          </div>
        </Panel>
      )}

      {/* Debug control bar — shown above status bar when debugging */}
      {isDebugging && (
        <Panel position="bottom-center">
          <div
            className="nodrag nopan"
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 8,
              marginBottom: 8,
            }}
          >
            {/* Debug controls */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "8px 16px",
                borderRadius: 10,
                background: "var(--zen-paper, #f2f0e3)",
                border: "1px solid #a855f7",
                boxShadow: "0 -1px 8px rgba(168, 85, 247, 0.15)",
                fontFamily: "'Bricolage Grotesque', sans-serif",
                fontSize: 12,
                fontWeight: 500,
              }}
            >
              {/* Status indicator */}
              <div
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: dbg.status === "paused" ? "#a855f7" : "var(--zen-coral, #F76F53)",
                  animation: dbg.status === "paused" ? undefined : "execPulse 1.5s ease-in-out infinite",
                  flexShrink: 0,
                }}
              />

              <span style={{ color: "var(--zen-dark, #2e2e2e)", textTransform: "capitalize" }}>
                {dbg.status === "paused" ? `Paused at ${dbg.pausedNodeId ?? "node"}` : dbg.status}
              </span>

              <div style={{ width: 1, height: 14, background: "var(--zen-muted, #ccc)", opacity: 0.3 }} />

              <span style={{ color: "var(--zen-muted, #999)" }}>
                {Math.floor(debugElapsed / 60)}:{String(debugElapsed % 60).padStart(2, "0")}
              </span>

              <div style={{ width: 1, height: 14, background: "var(--zen-muted, #ccc)", opacity: 0.3 }} />

              <span style={{ color: "var(--zen-muted, #999)" }}>
                {dbg.completedCount}/{dbg.totalNodes} steps
              </span>

              <div style={{ width: 1, height: 14, background: "var(--zen-muted, #ccc)", opacity: 0.3 }} />

              {/* Flow control buttons */}
              {dbg.status === "paused" && (
                <>
                  <button
                    onClick={continueExec}
                    style={{
                      padding: "4px 12px",
                      fontSize: 11,
                      fontWeight: 600,
                      borderRadius: 6,
                      border: "1px solid var(--zen-green, #63f78b)",
                      background: "var(--zen-green, #63f78b)",
                      color: "#fff",
                      cursor: "pointer",
                      fontFamily: "'Bricolage Grotesque', sans-serif",
                    }}
                  >
                    Continue
                  </button>
                  <button
                    onClick={skipNode}
                    style={{
                      padding: "4px 12px",
                      fontSize: 11,
                      fontWeight: 600,
                      borderRadius: 6,
                      border: "1px solid var(--zen-blue, #6287f5)",
                      background: "var(--zen-blue, #6287f5)",
                      color: "#fff",
                      cursor: "pointer",
                      fontFamily: "'Bricolage Grotesque', sans-serif",
                    }}
                  >
                    Skip
                  </button>
                </>
              )}
              <button
                onClick={stopDebug}
                style={{
                  padding: "4px 12px",
                  fontSize: 11,
                  fontWeight: 600,
                  borderRadius: 6,
                  border: "1px solid #ef4444",
                  background: "#ef4444",
                  color: "#fff",
                  cursor: "pointer",
                  fontFamily: "'Bricolage Grotesque', sans-serif",
                }}
              >
                Stop
              </button>

              {/* Dismiss when done */}
              {(dbg.status === "completed" || dbg.status === "failed" || dbg.status === "cancelled") && (
                <button
                  onClick={resetDebug}
                  style={{
                    marginLeft: 4,
                    padding: "2px 8px",
                    fontSize: 11,
                    borderRadius: 6,
                    border: "1px solid var(--zen-subtle, #e0ddd0)",
                    background: "transparent",
                    color: "var(--zen-dark, #2e2e2e)",
                    cursor: "pointer",
                    fontFamily: "'Bricolage Grotesque', sans-serif",
                  }}
                >
                  Dismiss
                </button>
              )}
            </div>

            {/* Variable inspector panel — visible when paused */}
            {dbg.status === "paused" && Object.keys(dbg.variables).length > 0 && (
              <DebugVariablePanel
                variables={dbg.variables}
                onSetVariable={setVariable}
                nodes={nodes}
              />
            )}
          </div>
        </Panel>
      )}

      {/* Node config panel */}
      <NodeConfigPanel
        node={selectedNode}
        onClose={() => setSelectedNodeId(null)}
        onNodeUpdate={handleNodeDataUpdate}
      />

      {/* Node inspection panel (post-execution or debug) */}
      {workflowId && (exec.runId || dbg.runId) && (
        <NodeInspectionPanel
          node={inspectedNode}
          workflowId={workflowId}
          runId={(exec.runId || dbg.runId)!}
          onClose={() => setInspectedNodeId(null)}
        />
      )}
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
