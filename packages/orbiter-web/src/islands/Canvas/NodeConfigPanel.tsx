import { useState, useCallback, useEffect, useRef } from "react";
import type { Node } from "@xyflow/react";
import { NODE_CATEGORIES } from "./NodeSidebar";
import LlmCallConfig from "./LlmCallConfig";
import AgentNodeConfig from "./AgentNodeConfig";

/* ------------------------------------------------------------------ */
/* Types                                                                */
/* ------------------------------------------------------------------ */

interface NodeConfigPanelProps {
  node: Node | null;
  onClose: () => void;
  onNodeUpdate: (id: string, data: Record<string, unknown>) => void;
}

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

/** Look up display info for a node type */
function getNodeTypeInfo(nodeType: string) {
  for (const cat of NODE_CATEGORIES) {
    const found = cat.types.find((t) => t.id === nodeType);
    if (found) return { label: found.label, category: cat.label, color: cat.color, icon: found.icon };
  }
  return { label: nodeType, category: "Unknown", color: "#999", icon: null };
}

/** Node types that have type-specific config panels */
const AGENT_TYPES = new Set(["agent_node", "sub_agent"]);

/** Render the type-specific configuration for a node */
function renderTypeConfig(
  node: Node,
  onDataChange: (updates: Record<string, unknown>) => void,
) {
  const nodeType = node.data.nodeType as string;

  if (nodeType === "llm_call") {
    return (
      <LlmCallConfig
        data={{
          model_provider: node.data.model_provider as string | undefined,
          model_name: node.data.model_name as string | undefined,
          prompt: node.data.prompt as string | undefined,
          temperature: node.data.temperature as number | undefined,
          max_tokens: node.data.max_tokens as number | undefined,
          response_format: node.data.response_format as "text" | "json" | undefined,
        }}
        onChange={onDataChange}
      />
    );
  }

  if (AGENT_TYPES.has(nodeType)) {
    return (
      <AgentNodeConfig
        data={{
          agent_id: node.data.agent_id as string | undefined,
          inline: node.data.inline as boolean | undefined,
          inline_name: node.data.inline_name as string | undefined,
          inline_model_provider: node.data.inline_model_provider as string | undefined,
          inline_model_name: node.data.inline_model_name as string | undefined,
          inline_instructions: node.data.inline_instructions as string | undefined,
          inline_tools: node.data.inline_tools as string[] | undefined,
        }}
        onChange={onDataChange}
      />
    );
  }

  /* Default placeholder for other node types */
  return (
    <div
      style={{
        padding: "20px 12px",
        textAlign: "center",
        fontSize: 12,
        color: "var(--zen-muted, #999)",
        border: "1px dashed var(--zen-subtle, #e0ddd0)",
        borderRadius: 8,
      }}
    >
      <div style={{ marginBottom: 4, fontSize: 16 }}>
        <svg
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ display: "inline-block", opacity: 0.5 }}
        >
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
        </svg>
      </div>
      Configuration for this node type coming soon
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Panel component                                                      */
/* ------------------------------------------------------------------ */

export default function NodeConfigPanel({ node, onClose, onNodeUpdate }: NodeConfigPanelProps) {
  const [name, setName] = useState("");
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  /* Sync name field when selected node changes */
  useEffect(() => {
    if (node) {
      setName((node.data.label as string) || "");
    }
  }, [node]);

  /* Debounced update — fires 500ms after last change */
  const scheduleUpdate = useCallback(
    (newName: string) => {
      if (!node) return;
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        onNodeUpdate(node.id, { ...node.data, label: newName });
      }, 500);
    },
    [node, onNodeUpdate],
  );

  /* Debounced data update for type-specific config fields */
  const scheduleDataUpdate = useCallback(
    (updates: Record<string, unknown>) => {
      if (!node) return;
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        onNodeUpdate(node.id, { ...node.data, ...updates });
      }, 500);
    },
    [node, onNodeUpdate],
  );

  /* Cleanup debounce timer on unmount */
  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  /* Handle Escape key to close */
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        /* Don't close if focus is in an input (let it blur first) */
        const tag = (e.target as HTMLElement)?.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA") {
          (e.target as HTMLElement).blur();
          return;
        }
        onClose();
      }
    };
    if (node) {
      window.addEventListener("keydown", handler);
      return () => window.removeEventListener("keydown", handler);
    }
  }, [node, onClose]);

  if (!node) return null;

  const typeInfo = getNodeTypeInfo(node.data.nodeType as string);

  const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = e.target.value;
    setName(v);
    scheduleUpdate(v);
  };

  return (
    <div
      ref={panelRef}
      className="nodrag nopan nowheel"
      style={{
        position: "absolute",
        top: 8,
        right: 8,
        bottom: 8,
        width: 300,
        zIndex: 10,
        display: "flex",
        flexDirection: "column",
        background: "var(--zen-paper, #f2f0e3)",
        border: "1px solid var(--zen-subtle, #e0ddd0)",
        borderRadius: 12,
        boxShadow: "0 2px 12px rgba(0,0,0,0.12)",
        overflow: "hidden",
        animation: "slideInRight 200ms ease-out",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "12px 14px 10px",
          borderBottom: "1px solid var(--zen-subtle, #e0ddd0)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, flex: 1, minWidth: 0 }}>
          <span
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 28,
              height: 28,
              borderRadius: 6,
              background: `${typeInfo.color}20`,
              color: typeInfo.color,
              flexShrink: 0,
            }}
          >
            {typeInfo.icon}
          </span>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div
              style={{
                fontSize: 10,
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                color: typeInfo.color,
                lineHeight: 1.2,
              }}
            >
              {typeInfo.label}
            </div>
          </div>
        </div>

        <button
          onClick={onClose}
          title="Close panel (Esc)"
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: 26,
            height: 26,
            border: "none",
            borderRadius: 6,
            background: "transparent",
            color: "var(--zen-muted, #999)",
            cursor: "pointer",
            flexShrink: 0,
          }}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      {/* Body */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "14px",
        }}
      >
        {/* Node name */}
        <div style={{ marginBottom: 16 }}>
          <label
            style={{
              display: "block",
              fontSize: 11,
              fontWeight: 600,
              color: "var(--zen-muted, #999)",
              marginBottom: 4,
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}
          >
            Name
          </label>
          <input
            type="text"
            value={name}
            onChange={handleNameChange}
            placeholder="Node name"
            style={{
              width: "100%",
              padding: "8px 10px",
              fontSize: 13,
              fontWeight: 500,
              border: "1px solid var(--zen-subtle, #e0ddd0)",
              borderRadius: 8,
              background: "var(--zen-paper, #f2f0e3)",
              color: "var(--zen-dark, #2e2e2e)",
              outline: "none",
              boxSizing: "border-box",
              transition: "border-color 150ms",
            }}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = "var(--zen-coral, #F76F53)";
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = "var(--zen-subtle, #e0ddd0)";
            }}
          />
        </div>

        {/* Node info */}
        <div style={{ marginBottom: 16 }}>
          <label
            style={{
              display: "block",
              fontSize: 11,
              fontWeight: 600,
              color: "var(--zen-muted, #999)",
              marginBottom: 4,
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}
          >
            Category
          </label>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              fontSize: 13,
              color: "var(--zen-dark, #2e2e2e)",
            }}
          >
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: typeInfo.color,
                flexShrink: 0,
              }}
            />
            {typeInfo.category}
          </div>
        </div>

        {/* Node ID (read-only) */}
        <div style={{ marginBottom: 16 }}>
          <label
            style={{
              display: "block",
              fontSize: 11,
              fontWeight: 600,
              color: "var(--zen-muted, #999)",
              marginBottom: 4,
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}
          >
            Node ID
          </label>
          <div
            style={{
              fontSize: 11,
              fontFamily: "monospace",
              color: "var(--zen-muted, #999)",
              padding: "6px 8px",
              background: "var(--zen-subtle, #e0ddd0)",
              borderRadius: 6,
              wordBreak: "break-all",
            }}
          >
            {node.id}
          </div>
        </div>

        {/* Divider */}
        <div
          style={{
            height: 1,
            background: "var(--zen-subtle, #e0ddd0)",
            margin: "16px 0",
          }}
        />

        {/* Node-type-specific configuration */}
        {renderTypeConfig(node, scheduleDataUpdate)}
      </div>

      {/* Inline keyframes for slide-in animation */}
      <style>{`
        @keyframes slideInRight {
          from { transform: translateX(100%); opacity: 0; }
          to { transform: translateX(0); opacity: 1; }
        }
      `}</style>
    </div>
  );
}
