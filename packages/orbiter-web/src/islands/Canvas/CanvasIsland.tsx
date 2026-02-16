import { useCallback, useEffect, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
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

function CanvasFlow() {
  const colorMode = useThemeColorMode();
  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onConnect: OnConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges],
  );

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      colorMode={colorMode}
      snapToGrid
      snapGrid={SNAP_GRID}
      fitView
    >
      <Background variant={BackgroundVariant.Dots} gap={GRID_SIZE} size={1.5} />
      <Controls />
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
