# PRD: Orbiter Web — AI Agent Platform

## Introduction

Orbiter Web is a comprehensive web UI platform for creating, configuring, running, monitoring, and managing AI agents built on the Orbiter framework. It combines the best features from six leading platforms — N8N (visual workflow canvas, 400+ integrations), Manus (autonomous execution, live sandbox view, artifact library), Azure AI Foundry (enterprise monitoring, evaluation, connected agents), LangFlow (node-based visual builder, component inspection, MCP support), CrewAI (AI-assisted crew builder, role-based agents, real-time tracing), and Dify (five app types, prompt IDE, plugin marketplace, RAG pipeline, annotation system, real-time debugging).

The platform serves both developers who build Orbiter agents in Python and no-code users who design agents entirely through the UI. It ships as a single deployable unit: an Astro 5.x static shell with interactive islands backed by an embedded Python server that wraps the Orbiter framework packages directly.

**Package location:** `packages/orbiter-web/`

## Goals

- Provide a complete visual platform for building, testing, and deploying Orbiter agents without writing code
- Support advanced users with code editors, Python tool creation, and direct framework integration
- Offer five application types (Chatbot, Chatflow, Workflow, Agent, Text Generator) matching Dify's versatility
- Include a node-based visual workflow canvas (Xyflow/React Flow island) with 15+ node types
- Provide real-time agent execution with streaming, step-through debugging, and live sandbox view
- Ship a Prompt IDE with model comparison, variable inspection, and template management
- Deliver production-grade monitoring with token usage, cost tracking, latency metrics, and alerting
- Support multi-agent orchestration via supervisor/delegation, crews, and planner/executor/verifier patterns
- Include a built-in RAG pipeline with document ingestion, vector search, and knowledge retrieval nodes
- Provide a plugin marketplace for community tools, models, and agent strategies
- Enable one-click deployment as API endpoint, embeddable chatbot, or internal tool
- Integrate with Orbiter's observability package for structured logging, metrics, traces, and cost estimation
- Support all Orbiter model providers (OpenAI, Anthropic, Gemini, Vertex) with multi-key load balancing
- Follow the design system from `docs/frontend-design.md` — warm paper/dark theme, Bricolage Grotesque + Junicode fonts, CSS-only animations, section-divider patterns

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  orbiter-web                         │
│                                                     │
│  ┌────────────────────┐  ┌───────────────────────┐  │
│  │   Astro 5.x Shell  │  │  Embedded Python API  │  │
│  │                     │  │    (FastAPI/Litestar)  │  │
│  │  - Static pages     │  │                       │  │
│  │  - Astro islands    │  │  - Wraps orbiter-core │  │
│  │    (Xyflow canvas)  │  │  - Wraps orbiter-     │  │
│  │  - Tailwind CSS v4  │  │    models             │  │
│  │  - TypeScript       │  │  - Wraps orbiter-     │  │
│  │  - WebSocket client │  │    context            │  │
│  │                     │  │  - Wraps orbiter-     │  │
│  │  Design system:     │  │    observability      │  │
│  │  docs/frontend-     │  │  - WebSocket server   │  │
│  │  design.md          │  │  - REST API           │  │
│  └────────────────────┘  └───────────────────────┘  │
│                                                     │
│  Single deployable unit                             │
└─────────────────────────────────────────────────────┘
```

**Frontend:** Astro 5.x (static output by default, islands for interactive components). Tailwind CSS v4 via `@tailwindcss/vite`. TypeScript strict mode. The workflow canvas uses Xyflow (React Flow) embedded as an Astro island — this is the only React dependency, scoped to the canvas. All other interactivity is vanilla `<script>` or CSS-only, consistent with the design system.

**Backend:** Thin Python API server (FastAPI or Litestar) embedded alongside the Astro build. Calls Orbiter packages directly — no intermediate orbiter-server package needed. Exposes REST endpoints for CRUD operations and WebSocket endpoints for streaming agent execution, real-time debugging, and live updates.

**Data:** SQLite for local development (agent configs, run history, credentials). PostgreSQL option for production. File-based storage for artifacts, documents, and plugin code.

---

## User Stories

### Section 1: Project & Workspace Foundation

#### US-001: Package scaffolding and build setup
**Description:** As a developer, I need the orbiter-web package initialized with Astro 5.x, Tailwind CSS v4, and TypeScript so that development can begin.

**Acceptance Criteria:**
- [ ] `packages/orbiter-web/` directory created with `package.json`, `tsconfig.json`, `astro.config.mjs`
- [ ] Astro 5.x configured with static output mode
- [ ] Tailwind CSS v4 integrated via `@tailwindcss/vite` (NOT `@astrojs/tailwind`)
- [ ] Design system tokens from `docs/frontend-design.md` copied into `src/styles/global.css` inside `@theme {}`
- [ ] Color variables (--zen-paper, --zen-dark, --zen-muted, --zen-subtle, coral, zen-blue, zen-green) configured
- [ ] Bricolage Grotesque (via @fontsource, 400-700) and Junicode (self-hosted woff2) font setup
- [ ] `cn()` utility in `src/utils/merge.ts` using clsx + tailwind-merge
- [ ] BaseLayout.astro with theme detection script, animation observer, SEOHead
- [ ] `pyproject.toml` for the embedded Python API server with FastAPI dependency
- [ ] `uv sync` succeeds with the new package in the workspace
- [ ] `npm run dev` starts both Astro dev server and Python API server

#### US-002: Design system primitives
**Description:** As a developer, I need reusable UI components matching the design system so all pages are visually consistent.

**Acceptance Criteria:**
- [ ] `Button.astro` — polymorphic (a/button), Primary/Bordered/Default variants, hover:scale micro-interaction
- [ ] `Card.astro` — polymorphic element tag, rounded-xl bg-subtle/80, hover lift effect
- [ ] `Input.astro` — text input with label, error state, dark mode support
- [ ] `Select.astro` — dropdown select with the design system styling
- [ ] `Badge.astro` — pill/badge component (rounded-full, configurable color)
- [ ] `Modal.astro` — overlay dialog with backdrop blur, slide-in animation
- [ ] `Tabs.astro` — tab navigation with active indicator, CSS-only where possible
- [ ] `Toast.astro` — notification toast with success/error/info variants
- [ ] Section pattern with `section-divider` class and container (`mx-auto max-w-6xl px-4 sm:px-6 lg:px-8`)
- [ ] All components use `cn()` for class merging and accept a `class` prop
- [ ] Dark mode works via CSS custom properties and `[data-theme="dark"]`
- [ ] Verify in browser using dev-browser skill

#### US-003: Application shell and navigation
**Description:** As a user, I need a persistent application shell with sidebar navigation so I can move between platform sections.

**Acceptance Criteria:**
- [ ] PageLayout.astro with collapsible left sidebar (not top navbar — this is an app, not a marketing site)
- [ ] Sidebar sections: Projects, Agents, Workflows, Playground, Tools, Models, Knowledge Base, Monitoring, Plugins, Settings
- [ ] Sidebar collapses to icon-only on smaller screens, hamburger on mobile
- [ ] Top bar with: current project name, search (Cmd+K), theme toggle, user menu
- [ ] Breadcrumb navigation showing current location
- [ ] Sidebar uses the design system colors: bg-paper for light, bg-dark for dark mode
- [ ] Active nav item highlighted with coral accent
- [ ] Keyboard shortcut Cmd+B to toggle sidebar
- [ ] Responsive: sidebar becomes bottom tab bar on mobile
- [ ] Verify in browser using dev-browser skill

#### US-004: Project management
**Description:** As a user, I want to organize my work into projects so I can manage multiple agent configurations separately.

**Acceptance Criteria:**
- [ ] Projects list page with card grid showing: name, description, agent count, last modified
- [ ] Create project dialog with name and description fields
- [ ] Project detail page showing all agents, workflows, and tools within the project
- [ ] Rename and delete project (with confirmation dialog)
- [ ] Project-scoped settings (default model, default provider)
- [ ] REST API: `GET/POST /api/projects`, `GET/PUT/DELETE /api/projects/:id`
- [ ] SQLite schema for projects table
- [ ] Verify in browser using dev-browser skill

#### US-005: Authentication and workspace settings
**Description:** As a user, I need basic authentication and workspace configuration to secure the platform.

**Acceptance Criteria:**
- [ ] Login page with email/password authentication
- [ ] Session management via HTTP-only cookies
- [ ] Settings page with sections: Profile, Appearance, API Keys, Workspace
- [ ] Appearance settings: theme preference (light/dark/system), sidebar position
- [ ] Workspace settings: default project, default model provider, concurrent run limit
- [ ] Password hashing with bcrypt
- [ ] REST API: `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/auth/me`
- [ ] Protected route middleware that redirects unauthenticated users to login
- [ ] Verify in browser using dev-browser skill

---

### Section 2: Model Provider Configuration

#### US-006: Credential store and provider management
**Description:** As a user, I need to configure API credentials for LLM providers so agents can call models.

**Acceptance Criteria:**
- [ ] Settings > Models page listing all configured providers
- [ ] Provider cards showing: provider name, icon, status (connected/error), model count, key count
- [ ] Support for: OpenAI, Anthropic, Google Gemini, Google Vertex, Ollama (local), custom OpenAI-compatible
- [ ] Each provider matches Orbiter's `ModelConfig`: provider, model_name, api_key, base_url, max_retries, timeout
- [ ] API keys encrypted at rest in the database (Fernet symmetric encryption)
- [ ] "Test Connection" button that validates the API key by making a lightweight API call
- [ ] REST API: `GET/POST /api/providers`, `GET/PUT/DELETE /api/providers/:id`, `POST /api/providers/:id/test`
- [ ] Verify in browser using dev-browser skill

#### US-007: Multi-key load balancing
**Description:** As a user, I want to add multiple API keys per provider with automatic load balancing and failover, so I can avoid rate limits (inspired by Dify).

**Acceptance Criteria:**
- [ ] Each provider can have multiple API keys with optional labels (e.g., "Production", "Backup")
- [ ] Load balancing strategy selector: round-robin, random, or least-recently-used
- [ ] Auto-failover: when one key hits rate limit (429), automatically switch to next key
- [ ] Key health status: green (working), yellow (rate limited, cooling down), red (invalid/expired)
- [ ] Per-key usage stats: total requests, total tokens, error count, last used timestamp
- [ ] Cooldown timer display for rate-limited keys
- [ ] REST API: `GET/POST /api/providers/:id/keys`, `DELETE /api/providers/:id/keys/:keyId`
- [ ] Verify in browser using dev-browser skill

#### US-008: Model discovery and selection
**Description:** As a user, I want to browse available models per provider so I can select the right model for each agent.

**Acceptance Criteria:**
- [ ] Model catalog page showing all available models across configured providers
- [ ] Filter by: provider, capability (chat, embedding, vision, reasoning), context window size
- [ ] Model cards showing: name, provider, context window, pricing (if known), capabilities tags
- [ ] "Fetch Models" button per provider that queries the provider API for available models
- [ ] Manual model entry for custom/fine-tuned models
- [ ] Model comparison view: select 2-3 models side-by-side showing specs
- [ ] Default model setting at workspace and project level
- [ ] REST API: `GET /api/models`, `POST /api/providers/:id/discover`
- [ ] Verify in browser using dev-browser skill

---

### Section 3: Application Types

#### US-009: Application type selection
**Description:** As a user, I want to create different types of AI applications (like Dify's five types) so I can choose the right pattern for my use case.

**Acceptance Criteria:**
- [ ] "New Application" dialog with five type cards:
  - **Chatbot**: Single-agent conversational app with memory
  - **Chatflow**: Multi-turn conversational workflow with branching
  - **Workflow**: DAG-based automation pipeline (no chat interface)
  - **Agent**: Autonomous agent with tools and reasoning (ReAct or function-calling)
  - **Text Generator**: Single-shot text generation with templates
- [ ] Each type card shows: icon, name, description, use case examples
- [ ] Selection determines which builder UI opens (form-based vs canvas-based)
- [ ] Application list page showing all apps with type badge, status, last run timestamp
- [ ] Duplicate application function
- [ ] REST API: `POST /api/applications` with `type` field
- [ ] Verify in browser using dev-browser skill

---

### Section 4: Agent Builder & Configuration

#### US-010: Agent creation form
**Description:** As a user, I want to create and configure agents through a form interface matching Orbiter's AgentConfig.

**Acceptance Criteria:**
- [ ] Agent builder page with tabbed form: Basic, Model, Tools, Handoffs, Hooks, Advanced
- [ ] Basic tab: name (required), description, instructions/system prompt (multi-line editor with syntax highlighting)
- [ ] Model tab: provider selector, model selector (populated from configured providers), temperature slider (0.0-2.0), max_tokens input, preview of `provider:model_name` string
- [ ] Tools tab: assigned tools list with add/remove, drag to reorder priority
- [ ] Handoffs tab: list of other agents this agent can hand off to, with add/remove
- [ ] Hooks tab: configure lifecycle hooks (HookPoint enum values from Orbiter), attach hook functions
- [ ] Advanced tab: max_steps (1-100), output_type (JSON schema editor for structured output), memory toggle, context toggle
- [ ] Save creates/updates agent via REST API
- [ ] Form validation with inline error messages
- [ ] REST API: `GET/POST /api/agents`, `GET/PUT/DELETE /api/agents/:id`
- [ ] Verify in browser using dev-browser skill

#### US-011: System prompt editor with Prompt IDE
**Description:** As a user, I want an advanced prompt editor with variable injection, templates, and model comparison (inspired by Dify's Prompt IDE).

**Acceptance Criteria:**
- [ ] Rich text editor with syntax highlighting for prompt variables (`{{variable_name}}`)
- [ ] Variable panel on the right showing all detected variables with type and default value inputs
- [ ] Prompt template library: save/load reusable prompt templates
- [ ] "Test Prompt" button that sends the prompt to the selected model with sample variables filled in
- [ ] Model comparison mode: send same prompt to 2-3 models side-by-side, compare outputs
- [ ] Token counter showing estimated input tokens for the current prompt
- [ ] Version history for prompts (diff view between versions)
- [ ] Import/export prompts as JSON
- [ ] Verify in browser using dev-browser skill

#### US-012: Role-based agent configuration (CrewAI-style)
**Description:** As a user, I want to define agents with role, goal, and backstory fields for multi-agent crews (inspired by CrewAI).

**Acceptance Criteria:**
- [ ] Optional "Persona" section in agent builder with: Role, Goal, Backstory fields
- [ ] Role: short descriptor (e.g., "Senior Research Analyst")
- [ ] Goal: what the agent aims to achieve
- [ ] Backstory: context about the agent's expertise and approach
- [ ] These fields are injected into the system prompt automatically when set
- [ ] Persona templates: pre-built role templates (Researcher, Writer, Coder, Reviewer, Planner)
- [ ] Preview of the full system prompt with persona fields expanded
- [ ] Verify in browser using dev-browser skill

#### US-013: AI-assisted agent builder
**Description:** As a user, I want to describe what I need in natural language and have the platform generate agent configurations (inspired by CrewAI Studio and n8n).

**Acceptance Criteria:**
- [ ] "AI Builder" button in the agent creation flow
- [ ] Natural language input: "I need an agent that researches topics and writes blog posts"
- [ ] AI generates: agent name, role/goal/backstory, system prompt, suggested tools, suggested model
- [ ] Generated configuration shown in a review panel before applying
- [ ] User can edit any field before confirming
- [ ] For multi-agent scenarios: AI generates a full crew with multiple agents, tasks, and tool assignments
- [ ] Voice input support via Web Speech API (microphone icon)
- [ ] Verify in browser using dev-browser skill

---

### Section 5: Visual Workflow Canvas Builder

#### US-014: Canvas foundation with Xyflow
**Description:** As a developer, I need the Xyflow (React Flow) canvas embedded as an Astro island so the visual workflow builder has a solid foundation.

**Acceptance Criteria:**
- [ ] Xyflow (React Flow) installed and configured as a React island in Astro
- [ ] Canvas renders within the application shell with proper sizing (fills available space)
- [ ] Pan and zoom controls (mouse wheel zoom, click-drag pan, minimap)
- [ ] Grid background with snap-to-grid for node placement
- [ ] Canvas toolbar: zoom in/out, fit view, lock/unlock, undo/redo
- [ ] Keyboard shortcuts: Delete (remove selected), Cmd+Z (undo), Cmd+Shift+Z (redo), Cmd+A (select all)
- [ ] Canvas state persisted to REST API on save
- [ ] Dark mode compatible (canvas background matches theme)
- [ ] Verify in browser using dev-browser skill

#### US-015: Node type system
**Description:** As a user, I need a rich set of node types so I can build complex workflows visually.

**Acceptance Criteria:**
- [ ] Node sidebar/panel with categorized node types that can be dragged onto canvas:
  - **Triggers**: Chat Input, Webhook, Schedule, Manual
  - **LLM**: LLM Call, Prompt Template, Model Selector
  - **Agent**: Agent Node (wraps full Orbiter Agent), Sub-Agent (delegation target)
  - **Tools**: Function Tool, HTTP Request, Code (Python), Code (JavaScript)
  - **Logic**: Conditional (if/else), Switch (multi-branch), Loop/Iterator, Aggregator
  - **Data**: Variable Assigner, Template (Jinja), JSON Transform, Text Splitter
  - **Knowledge**: Knowledge Retrieval, Document Loader, Embedding Node
  - **Output**: Chat Response, API Response, File Output, Notification
  - **Integration**: Webhook Call, MCP Client
- [ ] Each node type has a distinct icon and color accent
- [ ] Node search/filter in the sidebar
- [ ] Drag from sidebar onto canvas to create node instance
- [ ] Verify in browser using dev-browser skill

#### US-016: Node configuration panels
**Description:** As a user, I need to configure each node's properties through a detail panel.

**Acceptance Criteria:**
- [ ] Clicking a node opens a right-side configuration panel (slide-in, does not obscure canvas)
- [ ] Panel shows: node name (editable), node type label, configuration fields specific to node type
- [ ] LLM Call node config: model selector, prompt (with variable support), temperature, max_tokens, response format
- [ ] Conditional node config: condition expression editor, true/false output handles
- [ ] Code node config: embedded code editor (CodeMirror or Monaco) with Python/JS syntax highlighting
- [ ] HTTP Request node config: method, URL, headers, body, auth selector
- [ ] Agent node config: select existing agent or create inline, tool assignments
- [ ] Knowledge Retrieval node config: select knowledge base, top-k, similarity threshold
- [ ] All config changes auto-save with debounce
- [ ] Validation errors shown inline with red borders and messages
- [ ] Verify in browser using dev-browser skill

#### US-017: Edge connections and data flow
**Description:** As a user, I need to connect nodes with edges that represent data flow and see how data moves through the workflow.

**Acceptance Criteria:**
- [ ] Nodes have typed input/output handles (ports) on left/right sides
- [ ] Handle types: message, text, number, boolean, json, any — with color coding
- [ ] Drag from output handle to input handle to create edge
- [ ] Type validation: prevent connecting incompatible types (show error tooltip)
- [ ] Animated edges during workflow execution showing data flow direction
- [ ] Edge labels showing data preview on hover
- [ ] Click edge to select/delete it
- [ ] Bezier curve edges with smooth routing
- [ ] Conditional nodes have labeled outputs (True/False or named branches)
- [ ] Verify in browser using dev-browser skill

#### US-018: Workflow save, export, and import
**Description:** As a user, I want to save workflows and export/import them as JSON for sharing and version control.

**Acceptance Criteria:**
- [ ] Auto-save to backend on every canvas change (debounced 2s)
- [ ] Manual save button with Cmd+S shortcut
- [ ] Export workflow as JSON file (includes all node configs, edges, positions)
- [ ] Import workflow from JSON file (validates schema, places nodes on canvas)
- [ ] Workflow metadata: name, description, version, created/modified timestamps
- [ ] Workflow list page showing all workflows with status, node count, last run
- [ ] Duplicate workflow function
- [ ] REST API: `GET/POST /api/workflows`, `GET/PUT/DELETE /api/workflows/:id`, `POST /api/workflows/:id/export`, `POST /api/workflows/import`
- [ ] Verify in browser using dev-browser skill

#### US-019: Relationships panel
**Description:** As a user, I want to highlight connected nodes when selecting a node so I can trace data flow in complex workflows (inspired by Dify).

**Acceptance Criteria:**
- [ ] Hold Shift + click a node to activate relationships mode
- [ ] Connected nodes and edges are highlighted; unrelated nodes fade to 20% opacity
- [ ] Upstream nodes (ancestors) shown with blue tint
- [ ] Downstream nodes (descendants) shown with green tint
- [ ] The selected node shown with coral accent
- [ ] Click empty canvas to exit relationships mode
- [ ] Verify in browser using dev-browser skill

#### US-020: Canvas real-time validation
**Description:** As a user, I want the canvas to validate my workflow in real-time so I catch errors before running (inspired by LangFlow).

**Acceptance Criteria:**
- [ ] Nodes with missing required configuration show yellow warning badge
- [ ] Disconnected required inputs show red dot on the handle
- [ ] Cycle detection: if user creates a cycle, show error toast and prevent save
- [ ] Unreachable nodes (not connected to any trigger) shown with dashed border
- [ ] Validation summary panel (toggle) listing all issues with click-to-navigate
- [ ] Validation runs on every canvas change (debounced)
- [ ] Verify in browser using dev-browser skill

---

### Section 6: Interactive Playground & Chat

#### US-021: Chat playground interface
**Description:** As a user, I want to test agents in a conversational interface with streaming responses.

**Acceptance Criteria:**
- [ ] Full-screen chat interface accessible from agent detail or Playground nav item
- [ ] Agent selector dropdown at top (choose which agent to chat with)
- [ ] Message input with send button and Enter to submit (Shift+Enter for newline)
- [ ] Streaming response display: text appears token-by-token as received via WebSocket
- [ ] Message bubbles: user messages right-aligned (bg-subtle), agent messages left-aligned
- [ ] Markdown rendering in agent responses (code blocks, tables, lists, headings)
- [ ] Copy message button on hover
- [ ] Clear conversation button
- [ ] Conversation history persisted and resumable
- [ ] WebSocket endpoint: `ws://api/playground/:agentId/chat`
- [ ] Verify in browser using dev-browser skill

#### US-022: Tool call and execution trace panel
**Description:** As a user, I want to see exactly what the agent is doing during a conversation — tool calls, reasoning, token usage — in a side panel.

**Acceptance Criteria:**
- [ ] Collapsible right-side panel (toggle with button or Cmd+I)
- [ ] For each agent response, show expandable trace:
  - Tool calls: tool name, arguments (formatted JSON), result, duration
  - Model calls: model used, prompt tokens, completion tokens, total tokens, latency
  - Reasoning content: if the model returns chain-of-thought (reasoning_content from ModelResponse)
  - Finish reason: stop, tool_calls, length, content_filter
- [ ] Token usage summary per message and cumulative for conversation
- [ ] Cost estimate per message (based on model pricing configuration)
- [ ] Error display: if a tool call or model call fails, show error with stack trace
- [ ] Click on a tool call to expand full input/output details
- [ ] Verify in browser using dev-browser skill

#### US-023: Multi-model chat comparison
**Description:** As a user, I want to send the same message to multiple models side-by-side to compare responses (inspired by Dify's Prompt IDE).

**Acceptance Criteria:**
- [ ] "Compare" mode toggle in playground
- [ ] Select 2-4 models for comparison
- [ ] User message sent to all selected models simultaneously
- [ ] Responses displayed in columns side-by-side
- [ ] Per-model metrics shown: response time, token count, cost estimate
- [ ] Thumbs up/down rating for each response
- [ ] Export comparison results as CSV/JSON
- [ ] Verify in browser using dev-browser skill

#### US-024: Voice input and text-to-speech
**Description:** As a user, I want voice input for chat messages and text-to-speech for agent responses.

**Acceptance Criteria:**
- [ ] Microphone button next to message input
- [ ] Uses Web Speech API (SpeechRecognition) for speech-to-text
- [ ] Live transcription displayed as user speaks
- [ ] Auto-send on voice input completion (configurable)
- [ ] Text-to-speech toggle for agent responses using SpeechSynthesis API
- [ ] Language/voice selector in playground settings
- [ ] Visual indicator when recording (pulsing red dot)
- [ ] Verify in browser using dev-browser skill

#### US-025: Conversation management and threading
**Description:** As a user, I want to manage multiple conversation threads with an agent and view conversation history (inspired by Azure's thread model).

**Acceptance Criteria:**
- [ ] Left sidebar in playground showing conversation threads (like a chat app)
- [ ] Create new thread button
- [ ] Thread list shows: first message preview, timestamp, message count
- [ ] Switch between threads without losing history
- [ ] Delete thread with confirmation
- [ ] Search across all threads
- [ ] Thread metadata: agent used, total tokens, total cost, duration
- [ ] REST API: `GET/POST /api/threads`, `GET/DELETE /api/threads/:id`, `GET /api/threads/:id/messages`
- [ ] Verify in browser using dev-browser skill

---

### Section 7: Workflow Execution & Debugging

#### US-026: Workflow execution engine
**Description:** As a user, I want to run workflows from the canvas and see execution progress in real-time.

**Acceptance Criteria:**
- [ ] "Run" button in canvas toolbar
- [ ] Execution proceeds in DAG dependency order (topological sort)
- [ ] During execution: currently running node pulses with coral border, completed nodes show green checkmark, failed nodes show red X
- [ ] Edge animation shows data flowing between nodes during execution
- [ ] Execution status bar at bottom: running/completed/failed, elapsed time, step count
- [ ] Cancel button to stop mid-execution
- [ ] WebSocket endpoint streams execution events: `node_started`, `node_completed`, `node_failed`, `execution_completed`
- [ ] REST API: `POST /api/workflows/:id/run`, `DELETE /api/workflows/:id/run/:runId` (cancel)
- [ ] Verify in browser using dev-browser skill

#### US-027: Per-node output inspection
**Description:** As a user, I want to click any node after execution to inspect its input, output, and timing (inspired by n8n and LangFlow).

**Acceptance Criteria:**
- [ ] After execution, click any node to open inspection panel
- [ ] Panel shows tabs: Input, Output, Logs, Timing
- [ ] Input tab: data received from upstream nodes (formatted JSON)
- [ ] Output tab: data produced by this node (formatted JSON, with copy button)
- [ ] Logs tab: any log messages emitted during node execution
- [ ] Timing tab: start time, end time, duration, token usage (for LLM nodes)
- [ ] For LLM nodes: full prompt and response visible
- [ ] For Code nodes: stdout/stderr output
- [ ] Node inspection data persisted per run (not just latest)
- [ ] Verify in browser using dev-browser skill

#### US-028: Real-time debugging with step-through execution
**Description:** As a user, I want to step through workflow execution one node at a time, inspecting and modifying variables between steps (inspired by Dify 1.5.0).

**Acceptance Criteria:**
- [ ] "Debug" button (next to Run) that starts step-through mode
- [ ] Execution pauses before each node, showing a "Continue" / "Skip" / "Stop" control bar
- [ ] Variable inspect panel at bottom of canvas showing all variables and their current values
- [ ] Variables can be edited in the inspect panel to test different scenarios
- [ ] Modify a variable mid-flow and continue — downstream nodes use the modified value
- [ ] Breakpoint support: click node edge to set breakpoint (red dot), execution auto-pauses there
- [ ] Single-node execution: right-click node > "Run This Node" to test in isolation with mock inputs
- [ ] Verify in browser using dev-browser skill

#### US-029: Variable inspect panel
**Description:** As a user, I need a panel showing all workflow variables and their current values in real-time (inspired by Dify).

**Acceptance Criteria:**
- [ ] Bottom-of-canvas collapsible panel (toggle with Cmd+J)
- [ ] Table view: variable name, current value, type, source node
- [ ] Variables update in real-time during execution
- [ ] Click variable to highlight the node that produced it
- [ ] Filter variables by node or by type
- [ ] Edit variable values during debug mode
- [ ] Copy variable value to clipboard
- [ ] Verify in browser using dev-browser skill

#### US-030: Execution history and replay
**Description:** As a user, I want to view past workflow executions and replay them for debugging.

**Acceptance Criteria:**
- [ ] Execution history panel (sidebar or separate page) showing past runs
- [ ] Each run shows: status (success/failed/cancelled), timestamp, duration, trigger info
- [ ] Click a run to load its execution state onto the canvas (nodes colored by result)
- [ ] Replay: re-run a past execution with the same inputs
- [ ] Replay with modifications: edit inputs before replaying
- [ ] Filter history: by status, date range, trigger type
- [ ] Run comparison: select two runs to diff inputs/outputs
- [ ] REST API: `GET /api/workflows/:id/runs`, `GET /api/workflows/:id/runs/:runId`
- [ ] Verify in browser using dev-browser skill

---

### Section 8: Tool Management & Plugin System

#### US-031: Tool library and browsing
**Description:** As a user, I want to browse available tools, see their schemas, and assign them to agents.

**Acceptance Criteria:**
- [ ] Tools page with grid/list view of all available tools
- [ ] Tool card: name, description, category icon, parameter count, usage count
- [ ] Categories: Search, Code, File, Data, Communication, Custom
- [ ] Search and filter by category, name, or tag
- [ ] Tool detail page showing: full description, JSON schema, parameter descriptions, example usage
- [ ] "Add to Agent" action that opens agent selector
- [ ] REST API: `GET /api/tools`, `GET /api/tools/:id`
- [ ] Verify in browser using dev-browser skill

#### US-032: Custom tool creation via Python editor
**Description:** As a user, I want to create custom tools by writing Python functions in the browser, matching Orbiter's `@tool` decorator pattern.

**Acceptance Criteria:**
- [ ] "Create Tool" page with embedded code editor (CodeMirror with Python syntax highlighting)
- [ ] Pre-filled template using Orbiter's `@tool` decorator pattern:
  ```python
  from orbiter.tool import tool

  @tool
  def my_tool(param: str) -> str:
      """Tool description here."""
      return result
  ```
- [ ] Auto-detected parameters from function signature shown in a parameter table
- [ ] JSON schema preview generated from the function signature
- [ ] "Test Tool" button that executes the function with sample inputs
- [ ] Test output panel showing result or error with traceback
- [ ] Save tool to library
- [ ] REST API: `POST /api/tools/custom`, `PUT /api/tools/custom/:id`, `POST /api/tools/custom/:id/test`
- [ ] Verify in browser using dev-browser skill

#### US-033: Visual tool schema editor
**Description:** As a user, I want to create tools by defining their schema visually (without code) for simple API-wrapping tools.

**Acceptance Criteria:**
- [ ] Schema editor form: tool name, description, parameters list
- [ ] For each parameter: name, type (string/number/boolean/object/array), description, required toggle, default value
- [ ] Nested object support (expandable sub-parameters)
- [ ] Enum support for string parameters (dropdown of allowed values)
- [ ] Live JSON schema preview panel
- [ ] For HTTP-based tools: method, URL template, header mapping, body template
- [ ] OpenAPI import: paste OpenAPI spec URL or JSON to auto-generate tool definitions
- [ ] REST API: `POST /api/tools/schema`
- [ ] Verify in browser using dev-browser skill

#### US-034: Tool Mode toggle for workflow components
**Description:** As a user, I want to expose any workflow component as a tool that agents can call (inspired by LangFlow).

**Acceptance Criteria:**
- [ ] "Tool Mode" toggle switch on every workflow node
- [ ] When enabled, the node appears in the tool library as an available tool
- [ ] Tool name derived from node name, schema from node inputs/outputs
- [ ] Agents using this tool trigger the workflow node's logic
- [ ] Tool Mode nodes show a wrench icon badge on the canvas
- [ ] Verify in browser using dev-browser skill

#### US-035: Plugin marketplace
**Description:** As a user, I want to browse and install community plugins (tools, models, agent strategies) from a marketplace (inspired by Dify).

**Acceptance Criteria:**
- [ ] Plugins page with marketplace view: grid of plugin cards
- [ ] Plugin card: name, author, description, install count, rating, category badge
- [ ] Categories: Models, Tools, Agent Strategies, Extensions, Bundles
- [ ] Plugin detail page: full description, screenshots, changelog, permissions required
- [ ] One-click "Install" button
- [ ] Installed plugins tab showing currently installed with enable/disable toggle
- [ ] Plugin isolation: each plugin runs in its own subprocess with defined permissions
- [ ] Plugin manifest format (`plugin.json`) defining: name, version, type, permissions, entry point
- [ ] Local plugin development: "Load from directory" for development
- [ ] REST API: `GET /api/plugins/marketplace`, `POST /api/plugins/install`, `DELETE /api/plugins/:id`
- [ ] Verify in browser using dev-browser skill

---

### Section 9: Multi-Agent Orchestration

#### US-036: Agent hierarchy visualization
**Description:** As a user, I want to see how agents relate to each other (handoffs, delegation chains) in a visual hierarchy view.

**Acceptance Criteria:**
- [ ] Agent graph view (using Xyflow) showing all agents and their connections
- [ ] Agent nodes show: name, model, tool count, status
- [ ] Handoff connections shown as directed edges with "handoff" label
- [ ] Delegation connections shown as dashed edges with "delegates to" label
- [ ] Click agent node to open agent detail/config
- [ ] Auto-layout: hierarchical top-to-bottom layout for supervisor patterns
- [ ] Drag to rearrange agent positions
- [ ] Verify in browser using dev-browser skill

#### US-037: Supervisor/orchestrator pattern builder
**Description:** As a user, I want to configure a supervisor agent that delegates to sub-agents as tools (inspired by n8n and Azure Connected Agents).

**Acceptance Criteria:**
- [ ] "Create Supervisor" wizard: select a primary agent, then select sub-agents to delegate to
- [ ] Sub-agents automatically wrapped as tools on the supervisor (Orbiter handoff mechanism)
- [ ] Supervisor's system prompt auto-includes descriptions of available sub-agents
- [ ] Visual representation on canvas: supervisor node at top with connection lines to sub-agent nodes
- [ ] Add/remove sub-agents without recreating the supervisor
- [ ] Routing rules: optional conditions for when to delegate to which sub-agent
- [ ] Test supervisor in playground with live delegation trace
- [ ] Verify in browser using dev-browser skill

#### US-038: Crew builder (CrewAI-style)
**Description:** As a user, I want to organize agents into crews with defined tasks, sequential or parallel execution, and agent delegation.

**Acceptance Criteria:**
- [ ] "Create Crew" page with: crew name, description, process type (sequential/parallel)
- [ ] Agent assignment: add existing agents to crew, define their task within the crew
- [ ] Task definition per agent: what to accomplish, expected output format, dependencies on other tasks
- [ ] Sequential mode: tasks execute in order, output of one feeds into next
- [ ] Parallel mode: independent tasks run simultaneously
- [ ] Agent delegation: toggle allowing agents to create sub-tasks for other crew members
- [ ] Crew execution view: timeline showing task progress per agent
- [ ] REST API: `GET/POST /api/crews`, `GET/PUT/DELETE /api/crews/:id`, `POST /api/crews/:id/run`
- [ ] Verify in browser using dev-browser skill

#### US-039: Planner/Executor/Verifier pattern (Manus-style)
**Description:** As a user, I want to set up autonomous task decomposition where a planner breaks down goals, executors handle sub-tasks, and verifiers check results.

**Acceptance Criteria:**
- [ ] "Autonomous Mode" toggle on any agent or crew
- [ ] When enabled, a Planner agent is auto-created that:
  - Receives the user's goal as input
  - Generates a numbered plan with steps, descriptions, and dependencies
  - Assigns steps to executor agents
- [ ] Plan display: numbered step list with status icons (pending/running/completed/failed)
- [ ] Verification step: after each execution, a verifier agent checks the output
- [ ] User can modify the plan mid-execution (add/remove/reorder steps)
- [ ] One action per iteration: system awaits result before proceeding to next step
- [ ] Plan versioning: when the plan changes, keep history
- [ ] Verify in browser using dev-browser skill

#### US-040: Workflow-embedded agent nodes
**Description:** As a user, I want to embed agent nodes within workflows so agents can be part of larger automation pipelines.

**Acceptance Criteria:**
- [ ] Agent node type on the workflow canvas that wraps a full Orbiter Agent
- [ ] Agent node config: select existing agent, or configure inline (name, model, tools, instructions)
- [ ] Agent node input: accepts messages/text from upstream nodes
- [ ] Agent node output: produces agent response text and metadata (tool calls, token usage)
- [ ] Agent nodes support the full tool loop (multiple LLM-tool round-trips within a single node execution)
- [ ] Streaming output from agent nodes visible in the execution trace
- [ ] Nested agents: an agent node's tools can include other agent nodes (supervisor pattern on canvas)
- [ ] Verify in browser using dev-browser skill

---

### Section 10: Knowledge Base & RAG

#### US-041: Knowledge base management
**Description:** As a user, I want to create and manage knowledge bases for RAG (retrieval-augmented generation).

**Acceptance Criteria:**
- [ ] Knowledge Base page with list of knowledge bases
- [ ] Create knowledge base: name, description, embedding model selector, chunk size, chunk overlap
- [ ] Knowledge base detail page showing: document list, stats (total docs, total chunks, index size)
- [ ] Delete knowledge base with confirmation
- [ ] REST API: `GET/POST /api/knowledge-bases`, `GET/PUT/DELETE /api/knowledge-bases/:id`
- [ ] Verify in browser using dev-browser skill

#### US-042: Document ingestion
**Description:** As a user, I want to upload documents to a knowledge base with automatic chunking and embedding.

**Acceptance Criteria:**
- [ ] Upload UI: drag-and-drop zone + file picker
- [ ] Supported formats: PDF, DOCX, TXT, MD, CSV, HTML, PPTX
- [ ] Upload progress bar with status (uploading, processing, chunking, embedding, complete)
- [ ] Chunking preview: show first few chunks after processing
- [ ] Per-document settings: custom chunk size override, metadata tags
- [ ] Bulk upload: multiple files at once
- [ ] Document list showing: filename, type, chunk count, upload date, status
- [ ] Delete document (re-indexes automatically)
- [ ] REST API: `POST /api/knowledge-bases/:id/documents`, `GET /api/knowledge-bases/:id/documents`, `DELETE /api/knowledge-bases/:id/documents/:docId`
- [ ] Verify in browser using dev-browser skill

#### US-043: Knowledge retrieval configuration
**Description:** As a user, I want to configure how knowledge is retrieved for RAG queries.

**Acceptance Criteria:**
- [ ] Retrieval settings per knowledge base: search type (semantic/keyword/hybrid), top-k (1-20), similarity threshold (0.0-1.0)
- [ ] Reranking toggle with reranker model selection
- [ ] Test retrieval: enter a query, see retrieved chunks with relevance scores
- [ ] Chunk preview: click a chunk to see full text with source document highlighted
- [ ] Retrieval node in workflow canvas that connects to knowledge bases
- [ ] Agent-level knowledge base assignment: attach knowledge bases to agents for automatic RAG
- [ ] REST API: `POST /api/knowledge-bases/:id/search` (test query)
- [ ] Verify in browser using dev-browser skill

#### US-044: Vector store configuration
**Description:** As a user, I want to configure which vector store backend is used for embeddings.

**Acceptance Criteria:**
- [ ] Settings > Knowledge > Vector Store section
- [ ] Supported backends: built-in SQLite-VSS (default), Milvus/Zilliz, Qdrant, ChromaDB, Pinecone
- [ ] Connection configuration per backend: host, port, API key, collection name
- [ ] "Test Connection" button
- [ ] Migration tool: move embeddings between vector stores
- [ ] Status indicator showing connected backend and index health
- [ ] Verify in browser using dev-browser skill

---

### Section 11: Context Engine UI

#### US-045: Context configuration panel
**Description:** As a user, I want to configure Orbiter's Context engine for agents through the UI.

**Acceptance Criteria:**
- [ ] Context tab in agent builder with configuration for Orbiter's ContextConfig
- [ ] Automation level selector: Pilot (fully autonomous), Copilot (suggests, user approves), Navigator (plans only)
- [ ] Token budget settings: max tokens per step, max total tokens
- [ ] Memory toggle with memory type selector (conversation, summary, sliding window)
- [ ] Workspace toggle: enable artifact storage for the agent
- [ ] These settings map directly to Orbiter's `ContextConfig` class
- [ ] Verify in browser using dev-browser skill

#### US-046: Neuron pipeline builder
**Description:** As a user, I want to visually configure the neuron pipeline that builds prompts for agents (Orbiter's composable prompt building system).

**Acceptance Criteria:**
- [ ] Neuron pipeline editor: ordered list of neurons that compose the final prompt
- [ ] Available neuron types: System Instruction, Context State, Tool Results, Conversation History, Knowledge Retrieval, Custom Template
- [ ] Drag to reorder neurons in the pipeline
- [ ] Per-neuron configuration (template text, max tokens, priority)
- [ ] Preview panel showing the assembled prompt from the neuron pipeline
- [ ] Token budget visualization: bar chart showing how many tokens each neuron consumes
- [ ] Save/load neuron pipeline configurations
- [ ] Verify in browser using dev-browser skill

#### US-047: Context state inspector
**Description:** As a user, I want to inspect the hierarchical context state during agent execution.

**Acceptance Criteria:**
- [ ] Context inspector panel accessible from playground and run monitoring
- [ ] Tree view of context state: parent-child hierarchy matching Orbiter's ContextState
- [ ] Each state node shows: task_id, key-value pairs, token usage
- [ ] Expand/collapse state nodes
- [ ] Highlight changes between steps (new/modified values in green, removed in red)
- [ ] Fork visualization: when a context is forked, show the fork in the tree
- [ ] Merge visualization: show merge events with net token delta
- [ ] Verify in browser using dev-browser skill

#### US-048: Checkpoint management
**Description:** As a user, I want to save and restore checkpoints for long-running agent tasks.

**Acceptance Criteria:**
- [ ] Checkpoint list per agent run showing: checkpoint name, timestamp, state size, step number
- [ ] "Save Checkpoint" button during execution
- [ ] Auto-checkpoint at configurable intervals (every N steps)
- [ ] "Restore Checkpoint" button that resumes execution from that point
- [ ] Checkpoint diff: compare two checkpoints to see state changes
- [ ] Delete old checkpoints (with retention policy setting)
- [ ] REST API: `GET/POST /api/runs/:runId/checkpoints`, `POST /api/runs/:runId/checkpoints/:cpId/restore`
- [ ] Verify in browser using dev-browser skill

---

### Section 12: Run Monitoring & Observability

#### US-049: Monitoring dashboard
**Description:** As a user, I want a central dashboard showing aggregate metrics for all agent runs.

**Acceptance Criteria:**
- [ ] Dashboard page with time-range selector (1h, 24h, 7d, 30d, custom)
- [ ] Key metric cards: total runs, success rate, avg latency, total tokens, total cost
- [ ] Charts (built with vanilla JS + SVG, no chart library dependency):
  - Runs over time (bar chart)
  - Token usage over time (line chart)
  - Cost over time (line chart)
  - Latency distribution (histogram)
  - Success/failure ratio (donut chart)
- [ ] Per-agent breakdown table: agent name, run count, success rate, avg tokens, avg cost
- [ ] Click agent name to drill into agent-specific metrics
- [ ] Auto-refresh toggle (every 30s)
- [ ] Leverages orbiter-observability package metrics
- [ ] Verify in browser using dev-browser skill

#### US-050: Real-time execution tracing
**Description:** As a user, I want to view live execution traces for running agents with per-step detail.

**Acceptance Criteria:**
- [ ] Runs list page showing: all runs across all agents, sortable by status/time/agent/cost
- [ ] Run detail view with timeline visualization:
  - Each step as a row: step number, type (LLM call/tool call/handoff), duration, token count
  - Expandable step detail: full input, full output, error (if any)
  - Tool calls within a step shown as nested sub-rows
- [ ] Live streaming for in-progress runs (new steps appear as they execute)
- [ ] Run summary: total steps, total tokens, total cost, total duration, finish reason
- [ ] Filter by: agent, status, date range, minimum cost
- [ ] REST API: `GET /api/runs`, `GET /api/runs/:id`, WebSocket `ws://api/runs/:id/stream`
- [ ] Verify in browser using dev-browser skill

#### US-051: Cost estimation and tracking
**Description:** As a user, I want to see estimated and actual costs for all LLM usage.

**Acceptance Criteria:**
- [ ] Model pricing configuration: set $/1K tokens for input and output per model
- [ ] Pre-configured pricing for common models (GPT-4o, Claude 3.5, Gemini 1.5 Pro, etc.)
- [ ] Cost displayed: per message in playground, per step in traces, per run in history, per agent in dashboard
- [ ] Cost budget alerts: set monthly/daily budget per agent or workspace, warn at 80%, pause at 100%
- [ ] Cost breakdown chart: by agent, by model, by time period
- [ ] Leverages orbiter-observability's cost estimation module
- [ ] REST API: `GET /api/costs/summary`, `GET/PUT /api/costs/pricing`, `GET/PUT /api/costs/budgets`
- [ ] Verify in browser using dev-browser skill

#### US-052: Structured logging viewer
**Description:** As a user, I want to view structured logs from agent executions with filtering and search.

**Acceptance Criteria:**
- [ ] Logs page with streaming log viewer (newest at top)
- [ ] Log entry: timestamp, level (debug/info/warn/error), source (agent/tool/model/system), message
- [ ] Structured metadata expandable per log entry (JSON key-value pairs)
- [ ] Filter by: level, source, agent, time range, search text
- [ ] Log severity color coding: debug=gray, info=blue, warn=yellow, error=red
- [ ] Auto-scroll toggle for live log streaming
- [ ] Export logs as JSON or CSV
- [ ] Leverages orbiter-observability structured logging
- [ ] REST API: `GET /api/logs` with query params for filtering, WebSocket `ws://api/logs/stream`
- [ ] Verify in browser using dev-browser skill

#### US-053: Alerting and health checks
**Description:** As a user, I want to be notified when agents fail, exceed budgets, or have health issues.

**Acceptance Criteria:**
- [ ] Alert rules configuration: condition (error rate > X%, latency > Yms, cost > $Z), action (toast, email, webhook)
- [ ] Alert list page showing: triggered alerts with severity, time, affected agent, message
- [ ] Alert acknowledgment: mark as reviewed, assign to team member
- [ ] Health check status page: per-agent health (last run status, average success rate, last error)
- [ ] Provider health: API key status, latency per provider, rate limit status
- [ ] SLO tracking: define SLOs (e.g., 99% success rate) and track compliance
- [ ] Leverages orbiter-observability alerting hooks and health check system
- [ ] REST API: `GET/POST /api/alerts/rules`, `GET /api/alerts`, `GET /api/health`
- [ ] Verify in browser using dev-browser skill

---

### Section 13: Human-in-the-Loop

#### US-054: Approval gates in workflows
**Description:** As a user, I want to add approval steps to workflows where execution pauses until a human approves (inspired by n8n and Azure).

**Acceptance Criteria:**
- [ ] "Approval Gate" node type in the workflow canvas
- [ ] When execution reaches this node, it pauses and sends a notification
- [ ] Approval UI: modal showing the pending action, context from previous nodes, approve/reject buttons
- [ ] Optional: require comment with rejection
- [ ] Timeout setting: auto-reject after N minutes/hours if no response
- [ ] Approval history: who approved, when, with what comment
- [ ] Notification channels: in-app notification, email, webhook
- [ ] REST API: `GET /api/approvals/pending`, `POST /api/approvals/:id/respond`
- [ ] Verify in browser using dev-browser skill

#### US-055: Real-time agent takeover
**Description:** As a user, I want to take over agent execution in real-time, injecting messages or stopping the agent mid-run (inspired by Manus).

**Acceptance Criteria:**
- [ ] During live agent execution in playground, "Take Over" button appears
- [ ] Clicking pauses the agent's autonomous loop
- [ ] User can: inject a message (overriding the agent's next action), modify the current plan, resume, or stop
- [ ] "Resume" returns control to the agent with the injected context
- [ ] "Stop" terminates execution immediately with partial results saved
- [ ] Takeover history logged in the execution trace
- [ ] Visual indicator: agent status badge changes from "Running" to "Paused (Human Control)"
- [ ] Verify in browser using dev-browser skill

#### US-056: Annotation and feedback system
**Description:** As a user, I want to annotate agent responses with feedback and curate improved responses (inspired by Dify).

**Acceptance Criteria:**
- [ ] Thumbs up/down buttons on every agent response in playground and logs
- [ ] "Annotate" button that opens an editor to write an improved response
- [ ] Annotation Reply: when a semantically similar query comes in, serve the annotated response instead of calling the LLM
- [ ] Annotation management page: list all annotations, search, edit, delete
- [ ] Bulk import: upload Q&A pairs as CSV for annotation
- [ ] Annotation match threshold setting (similarity score)
- [ ] Cost savings display: how much the annotation system saved vs LLM calls
- [ ] REST API: `GET/POST /api/annotations`, `PUT/DELETE /api/annotations/:id`, `POST /api/annotations/import`
- [ ] Verify in browser using dev-browser skill

#### US-057: Message editing in conversation logs
**Description:** As a user, I want to edit or delete individual messages in conversation logs for debugging and data curation (inspired by LangFlow and Dify).

**Acceptance Criteria:**
- [ ] In conversation thread view, hover menu on each message with: edit, delete, copy, replay from here
- [ ] Edit opens inline editor with the original message text
- [ ] "Replay from here": re-runs the conversation from the edited message onward
- [ ] Delete removes message and re-indexes subsequent messages
- [ ] Edit history tracked (show original vs edited)
- [ ] Verify in browser using dev-browser skill

---

### Section 14: Sandbox Execution

#### US-058: Code interpreter sandbox
**Description:** As a user, I want agents to execute Python code in a sandboxed environment (inspired by Azure Code Interpreter and Manus).

**Acceptance Criteria:**
- [ ] Code Interpreter tool available for agents: executes Python in an isolated subprocess
- [ ] Sandbox restrictions: no network access, no filesystem access outside workspace, memory limit, CPU timeout
- [ ] Common libraries pre-installed: pandas, numpy, matplotlib, requests, json, csv
- [ ] Code output captured: stdout, stderr, generated files (images, CSVs)
- [ ] Generated files displayed inline in chat (images rendered, files downloadable)
- [ ] Sandbox lifecycle: created per-run, destroyed after run completes
- [ ] Configuration: allowed libraries, timeout, memory limit
- [ ] REST API: `POST /api/sandbox/execute` (for testing), sandbox management in run context
- [ ] Verify in browser using dev-browser skill

#### US-059: Live sandbox view
**Description:** As a user, I want to watch the agent working in its sandbox in real-time (inspired by Manus's "Computer" view).

**Acceptance Criteria:**
- [ ] "Sandbox View" tab in the run detail page
- [ ] Real-time display of: files being created/modified (file tree), terminal output (log stream)
- [ ] File preview: click any file in the sandbox to preview contents
- [ ] Terminal view: shows command history and output as if watching a terminal
- [ ] For browser-based tasks: embedded iframe showing the agent's browser session (optional, requires headless browser)
- [ ] Artifact collection: all files generated during the run collected and downloadable
- [ ] Verify in browser using dev-browser skill

---

### Section 15: Artifact Management

#### US-060: Artifact library
**Description:** As a user, I want a centralized library of all files and artifacts generated by agents (inspired by Manus's Library).

**Acceptance Criteria:**
- [ ] Artifacts page with grid/list view of all generated files
- [ ] File preview: images rendered inline, text files with syntax highlighting, PDFs embedded
- [ ] Filter by: file type, generating agent, date range, run ID
- [ ] Download individual files or bulk download as ZIP
- [ ] Delete artifacts with confirmation
- [ ] Artifact metadata: source agent, source run, generation timestamp, file size
- [ ] Link back to the run that generated each artifact
- [ ] REST API: `GET /api/artifacts`, `GET /api/artifacts/:id`, `GET /api/artifacts/:id/download`, `DELETE /api/artifacts/:id`
- [ ] Verify in browser using dev-browser skill

#### US-061: Artifact editing
**Description:** As a user, I want to edit generated artifacts directly in the platform (inspired by Manus).

**Acceptance Criteria:**
- [ ] Text files: inline editor with syntax highlighting (CodeMirror)
- [ ] Markdown: preview mode with rendered markdown
- [ ] JSON: formatted editor with validation
- [ ] Version history: each edit creates a new version, diff view between versions
- [ ] Auto-save with debounce
- [ ] "Regenerate" button: send the artifact back to the agent with modification instructions
- [ ] Verify in browser using dev-browser skill

---

### Section 16: Evaluation & Quality

#### US-062: Evaluation framework
**Description:** As a user, I want to run automated evaluations on agent responses to measure quality (inspired by Azure AI Foundry and CrewAI).

**Acceptance Criteria:**
- [ ] Evaluation page: create evaluation suites with test cases
- [ ] Test case: input message, expected output (or criteria), evaluator type
- [ ] Built-in evaluators: exact match, contains, regex match, LLM-as-judge (uses another model to score), semantic similarity
- [ ] Run evaluation: sends all test cases to the agent, collects responses, scores them
- [ ] Results table: test case, expected, actual, score, pass/fail
- [ ] Aggregate metrics: overall score, pass rate, average similarity
- [ ] Evaluation history: track scores over time to detect regressions
- [ ] REST API: `GET/POST /api/evaluations`, `POST /api/evaluations/:id/run`, `GET /api/evaluations/:id/results`
- [ ] Verify in browser using dev-browser skill

#### US-063: Agent benchmarking
**Description:** As a user, I want to benchmark agents against each other on the same evaluation suite (inspired by CrewAI + AgentOps).

**Acceptance Criteria:**
- [ ] "Benchmark" mode: select multiple agents and an evaluation suite
- [ ] Run all agents against same test cases
- [ ] Comparison table: agent vs agent scores per test case
- [ ] Leaderboard view: agents ranked by overall score
- [ ] Cost-efficiency metric: score per dollar spent
- [ ] Charts: score comparison bar chart, cost comparison, latency comparison
- [ ] Export benchmark results as CSV/JSON
- [ ] Verify in browser using dev-browser skill

#### US-064: Safety evaluation and red-teaming
**Description:** As a user, I want to test agents for safety issues and harmful outputs (inspired by Azure AI Foundry's red-teaming).

**Acceptance Criteria:**
- [ ] Safety evaluation suite with pre-built adversarial test cases
- [ ] Categories: prompt injection, jailbreak attempts, PII leakage, harmful content generation, bias detection
- [ ] Red-team mode: AI-generated adversarial inputs targeting the agent
- [ ] Safety scores per category with pass/fail thresholds
- [ ] Detailed report: flagged responses with explanation of the safety concern
- [ ] Safety policy configuration: define what the agent should and shouldn't do
- [ ] REST API: `POST /api/evaluations/:id/safety-run`
- [ ] Verify in browser using dev-browser skill

---

### Section 17: Deployment & Sharing

#### US-065: One-click API deployment
**Description:** As a user, I want to deploy agents and workflows as API endpoints with a single click (inspired by Dify and LangFlow).

**Acceptance Criteria:**
- [ ] "Deploy as API" button on agent and workflow detail pages
- [ ] Generates REST endpoint: `POST /api/deployed/:id/run` accepting input messages
- [ ] API key authentication for deployed endpoints
- [ ] API documentation auto-generated (OpenAPI/Swagger spec)
- [ ] Rate limiting configuration per deployment
- [ ] Deployment management page: list all deployments, status, usage stats, enable/disable
- [ ] Streaming support via SSE (Server-Sent Events) for deployed endpoints
- [ ] REST API: `POST /api/deployments`, `GET /api/deployments`, `DELETE /api/deployments/:id`
- [ ] Verify in browser using dev-browser skill

#### US-066: Embeddable chatbot widget
**Description:** As a user, I want to deploy an agent as an embeddable chat widget for websites (inspired by Dify).

**Acceptance Criteria:**
- [ ] "Deploy as Widget" option generating a `<script>` embed code
- [ ] Widget: floating chat bubble (bottom-right), opens into chat panel
- [ ] Customization: primary color, position, welcome message, avatar
- [ ] Widget uses the deployed API endpoint for communication
- [ ] Widget preview in the deployment settings page
- [ ] Copy embed code button
- [ ] Widget respects CORS configuration
- [ ] Verify in browser using dev-browser skill

#### US-067: Agent templates and sharing
**Description:** As a user, I want to save agents as templates and share them with others.

**Acceptance Criteria:**
- [ ] "Save as Template" button on agent detail page
- [ ] Template includes: agent config, tools, prompt, but not credentials
- [ ] Template library page: browse, search, and use community templates
- [ ] "Use Template" button that creates a new agent from the template
- [ ] Template detail: description, required tools, required models, creator
- [ ] Import/export templates as JSON files
- [ ] Template versioning: update a template while keeping old versions accessible
- [ ] REST API: `GET/POST /api/templates`, `POST /api/templates/:id/instantiate`
- [ ] Verify in browser using dev-browser skill

#### US-068: Version control for agent configurations
**Description:** As a user, I want version history for all agent and workflow configurations with diff and rollback (inspired by Azure's CI/CD approach).

**Acceptance Criteria:**
- [ ] Every save to an agent or workflow creates a new version automatically
- [ ] Version history page: list of versions with timestamp, author, change summary
- [ ] Diff view between any two versions (side-by-side, highlighting changes)
- [ ] Rollback: restore any previous version as the current configuration
- [ ] Version tags: mark specific versions with labels (e.g., "production", "v2.1")
- [ ] Export version as JSON
- [ ] REST API: `GET /api/agents/:id/versions`, `POST /api/agents/:id/versions/:vId/rollback`
- [ ] Verify in browser using dev-browser skill

---

### Section 18: AI-Assisted Features

#### US-069: Natural language to workflow generation
**Description:** As a user, I want to describe a workflow in natural language and have the AI generate it on the canvas (inspired by n8n's AI Workflow Builder).

**Acceptance Criteria:**
- [ ] "AI Generate" button on empty canvas or in toolbar
- [ ] Text input: "Create a workflow that takes user input, searches the knowledge base, passes results to an LLM, and returns a formatted response"
- [ ] AI generates: nodes, connections, and basic configuration
- [ ] Generated workflow appears on canvas for review and editing
- [ ] "Refine" button: describe modifications in natural language to update the workflow
- [ ] Generation uses the configured default model
- [ ] Verify in browser using dev-browser skill

#### US-070: AI-assisted prompt optimization
**Description:** As a user, I want AI help improving my system prompts based on evaluation results.

**Acceptance Criteria:**
- [ ] "Optimize Prompt" button in the Prompt IDE
- [ ] AI analyzes: current prompt, evaluation results (if available), conversation logs
- [ ] Suggests: improved prompt with tracked changes (diff view)
- [ ] Multiple optimization strategies: clarity, specificity, safety, conciseness
- [ ] User can accept all changes, selectively apply, or reject
- [ ] Optimization history: track which changes improved scores
- [ ] Verify in browser using dev-browser skill

---

### Section 19: Integrations

#### US-071: MCP client support
**Description:** As a user, I want agents to connect to external MCP servers as tool providers (inspired by LangFlow).

**Acceptance Criteria:**
- [ ] Settings > Integrations > MCP Servers section
- [ ] Add MCP server: URL, authentication, name
- [ ] "Discover Tools" button that queries the MCP server for available tools
- [ ] Discovered tools appear in the tool library with "MCP" badge
- [ ] MCP tools assignable to agents like any other tool
- [ ] Connection health monitoring
- [ ] REST API: `GET/POST /api/integrations/mcp`, `POST /api/integrations/mcp/:id/discover`
- [ ] Verify in browser using dev-browser skill

#### US-072: Webhook triggers and notifications
**Description:** As a user, I want to trigger workflows via webhooks and send notifications to external services.

**Acceptance Criteria:**
- [ ] Webhook Trigger node: generates a unique URL, workflow executes when URL receives a POST
- [ ] Webhook URL management: view, regenerate, disable
- [ ] Incoming webhook payload available as variables in the workflow
- [ ] Notification node: send HTTP POST to a configured URL when workflow completes or fails
- [ ] Pre-built notification templates: Slack (webhook), Discord (webhook), email (SMTP)
- [ ] Webhook request log: recent requests with payload and response status
- [ ] REST API: auto-generated webhook URLs under `/api/webhooks/:workflowId/:hookId`
- [ ] Verify in browser using dev-browser skill

#### US-073: External observability integration
**Description:** As a user, I want to export traces and logs to external observability platforms (Langfuse, LangSmith, Datadog).

**Acceptance Criteria:**
- [ ] Settings > Integrations > Observability section
- [ ] Support for: Langfuse, LangSmith, Datadog, Opik, custom webhook
- [ ] Per-integration configuration: API key, endpoint URL, project/dataset name
- [ ] Toggle per integration to enable/disable trace export
- [ ] Trace export includes: all LLM calls, tool calls, agent steps, token usage, latency
- [ ] "Test Integration" button that sends a sample trace
- [ ] Leverages orbiter-observability's existing integration support
- [ ] Verify in browser using dev-browser skill

---

### Section 20: Responsive Design & Mobile

#### US-074: Mobile monitoring dashboard
**Description:** As a user, I want to monitor agent runs from my phone (inspired by Manus's mobile support).

**Acceptance Criteria:**
- [ ] Dashboard responsive: metric cards stack vertically on mobile
- [ ] Run list: simplified card view on mobile (status, agent, duration, cost)
- [ ] Run detail: scrollable trace view optimized for mobile
- [ ] Active run monitoring: streaming status updates on mobile
- [ ] Pull-to-refresh on all list pages
- [ ] Bottom navigation bar (replaces sidebar on mobile)
- [ ] Touch-optimized: large tap targets, swipe gestures for common actions
- [ ] Verify in browser using dev-browser skill (with mobile viewport)

#### US-075: Mobile playground
**Description:** As a user, I want a simplified chat playground usable on mobile.

**Acceptance Criteria:**
- [ ] Chat interface fills screen on mobile
- [ ] Agent selector in a compact header
- [ ] Trace panel accessible via swipe-right gesture or toggle button
- [ ] Voice input button prominently displayed for mobile
- [ ] Message input stays pinned to bottom with keyboard avoidance
- [ ] Smooth scrolling with new message auto-scroll
- [ ] Verify in browser using dev-browser skill (with mobile viewport)

---

### Section 21: Documentation & Onboarding

#### US-076: In-app documentation
**Description:** As a user, I need built-in documentation and contextual help.

**Acceptance Criteria:**
- [ ] Help icon (?) in the sidebar opening a docs panel
- [ ] Contextual help: tooltip help buttons on key forms that open relevant docs
- [ ] Searchable documentation (Astro content collection for docs pages)
- [ ] Getting started guide: step-by-step walkthrough of creating first agent
- [ ] API reference: auto-generated from REST endpoint definitions
- [ ] Keyboard shortcut reference (Cmd+/ to open)
- [ ] Verify in browser using dev-browser skill

#### US-077: Interactive onboarding tour
**Description:** As a new user, I want a guided tour that walks me through the platform features.

**Acceptance Criteria:**
- [ ] First-login trigger: show onboarding tour prompt
- [ ] Guided tour: highlighting UI elements with step-by-step explanations
- [ ] Steps: create project, configure model provider, create agent, test in playground
- [ ] Skip button on every step
- [ ] "Show tour again" button in settings
- [ ] Tour state persisted (don't re-show after completion)
- [ ] Verify in browser using dev-browser skill

---

## Functional Requirements

- FR-1: The system must support five application types: Chatbot, Chatflow, Workflow, Agent, Text Generator
- FR-2: The visual workflow canvas must support 15+ node types with typed connections
- FR-3: Agent configuration must map directly to Orbiter's `AgentConfig` (name, model, instructions, temperature, max_tokens, max_steps)
- FR-4: Model configuration must map to Orbiter's `ModelConfig` (provider, model_name, api_key, base_url, max_retries, timeout)
- FR-5: The system must support all Orbiter model providers: OpenAI, Anthropic, Gemini, Vertex, custom OpenAI-compatible
- FR-6: Multi-key load balancing must support round-robin, random, and least-recently-used strategies
- FR-7: The playground must stream responses via WebSocket with real-time token-by-token display
- FR-8: Workflow execution must follow DAG topological order with cycle detection
- FR-9: The system must provide per-step execution traces with input, output, token usage, and duration
- FR-10: Custom tools must be definable via Python code editor using Orbiter's `@tool` decorator
- FR-11: The system must support three multi-agent patterns: supervisor/delegation, crew-based, planner/executor/verifier
- FR-12: Knowledge bases must support document ingestion (PDF, DOCX, TXT, MD, CSV, HTML, PPTX)
- FR-13: RAG retrieval must support semantic, keyword, and hybrid search modes
- FR-14: The context engine UI must expose Orbiter's three automation levels: pilot, copilot, navigator
- FR-15: Approval gates must pause workflow execution and notify the user
- FR-16: The annotation system must serve cached responses for semantically similar queries
- FR-17: Code execution must be sandboxed with resource limits (memory, CPU timeout, network isolation)
- FR-18: All API keys must be encrypted at rest
- FR-19: Evaluations must support exact match, contains, regex, LLM-as-judge, and semantic similarity
- FR-20: Deployments must support REST API and embeddable chat widget formats
- FR-21: Version history must be tracked for all agent and workflow configurations
- FR-22: The system must be responsive with mobile-optimized views for monitoring and chat
- FR-23: The system must integrate with Orbiter's observability package for structured logging, metrics, and traces
- FR-24: Plugin marketplace must isolate plugin execution with defined permissions
- FR-25: The system must support MCP client connections for external tool discovery

## Non-Goals (Out of Scope)

- **No custom frontend framework**: The canvas uses Xyflow/React as an island; all other UI is Astro + vanilla JS. No full SPA framework.
- **No user-facing agent training/fine-tuning**: The platform configures and runs agents, not trains models.
- **No multi-tenant SaaS features**: No billing, subscription management, or organization hierarchy. Single workspace deployment.
- **No native mobile app**: Responsive web only, no iOS/Android apps.
- **No real-time collaboration**: No simultaneous multi-user editing (Google Docs style). Single user per workflow editor.
- **No data pipeline/ETL features**: The workflow builder is for agent orchestration, not general data engineering.
- **No model hosting**: Models are accessed via provider APIs, not hosted locally (except Ollama integration for users who run their own).

## Design Considerations

### Design System (from `docs/frontend-design.md`)
- **Colors**: Paper/dark warm palette with coral (#F76F53) primary accent, zen-blue (#6287f5) secondary, zen-green (#63f78b) tertiary
- **Typography**: Bricolage Grotesque (body, 400-700) + Junicode (display headings, serif)
- **Layout**: Application shell with collapsible sidebar (not the marketing site's Navbar pattern)
- **Components**: rounded-xl corners, subtle borders (border-dark/[0.06-0.08]), hover lift effects, blur-based animations
- **Dark mode**: CSS custom properties, `[data-theme="dark"]`, warm charcoal (not pure black)
- **Animations**: CSS-only with IntersectionObserver for scroll triggers, `prefers-reduced-motion` respected

### Component Reuse
- Reuse `global.css` (color tokens, font faces, animation keyframes)
- Reuse `cn()` utility, `Button.astro`, `Card.astro` patterns
- Adapt `BaseLayout.astro` for the app shell (different from marketing PageLayout)
- The workflow canvas is the one exception to the "no React" rule — Xyflow requires React, contained in an Astro island

## Technical Considerations

### Architecture
- **Frontend**: Astro 5.x, Tailwind CSS v4 via `@tailwindcss/vite`, TypeScript strict
- **Canvas island**: React 19 + Xyflow, isolated in `src/islands/Canvas/`
- **Backend**: FastAPI (Python) embedded alongside Astro build, wraps orbiter-core, orbiter-models, orbiter-context, orbiter-observability directly
- **Database**: SQLite for local dev (via aiosqlite), PostgreSQL option for production
- **WebSocket**: FastAPI WebSocket endpoints for streaming, live execution, log streaming
- **File storage**: Local filesystem with configurable path for artifacts, documents, plugin code

### Dependencies on Other Orbiter Packages
- `orbiter-core`: Agent, Tool, types, config, registry, events, hooks
- `orbiter-models`: ModelProvider, model_registry, OpenAI/Anthropic/Gemini/Vertex providers
- `orbiter-context`: Context, ContextConfig, ContextState, Neurons, Checkpoints
- `orbiter-observability`: Structured logging, metrics, traces, cost estimation, health checks, alerting, SLO tracking

### Performance Requirements
- Canvas must handle 100+ nodes without jank (60fps pan/zoom)
- Chat streaming must display tokens within 50ms of receipt
- Dashboard must load within 2 seconds
- API responses for CRUD operations under 200ms
- WebSocket reconnection with exponential backoff

### Security
- API key encryption at rest (Fernet/AES-256)
- JWT-based authentication with HTTP-only refresh token cookies
- CORS configuration for widget deployments
- Sandbox isolation for code execution via Docker containers with resource limits (CPU, memory, disk, network)
- Plugin isolation via Docker containers — each plugin runs in its own container with defined permissions
- Input sanitization for all user-provided content
- Content Security Policy headers
- CSRF protection on all state-changing endpoints

## Success Metrics

- User can create and test an agent within 5 minutes of first login (onboarding flow)
- Workflow canvas supports 100+ nodes without performance degradation
- Agent playground streaming latency < 50ms from backend receipt to screen render
- All six platform patterns implemented: visual canvas (n8n/LangFlow/Dify), autonomous execution (Manus), enterprise monitoring (Azure), crew management (CrewAI)
- Plugin marketplace supports at least 10 community-contributed plugins at launch
- Mobile monitoring usable for run tracking and basic chat

## Resolved Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Canvas library | **Xyflow/React Flow** (React Astro island) | Proven, battle-tested canvas used by LangFlow and others. Provides drag-and-drop, minimap, node types, and connection handling out of the box. |
| 2 | Code editor | **CodeMirror 6 everywhere** (~200KB) | Lighter than Monaco (~2MB), fast, extensible. Consistent experience across tool creation, code nodes, and prompt editing. |
| 3 | Real-time architecture | **Single global WebSocket + multiplexed channels** | One WebSocket connection for the entire app with channel-based multiplexing (chat, logs, execution, sandbox). Efficient, fewer connections. |
| 4 | Plugin sandboxing | **Docker containers** | Each plugin runs in its own container with resource limits. Strong isolation, proven approach. |
| 5 | Offline support | **None** | Platform requires an active connection at all times. Simpler architecture, no sync complexity. |
| 6 | Python server embedding | **Astro outputs static files served by FastAPI directly** | FastAPI serves the Astro build output as static files and provides the API. Single process, simplest deployment. |
| 7 | API contract | **Included in this PRD** | Full REST + WebSocket API contract defined below so frontend and backend can be built in parallel. |

---

## API Contract

This section defines the complete API that the embedded Python server must implement. All REST endpoints are prefixed with `/api/v1`. The frontend consumes this API exclusively.

### Authentication

All endpoints except `/api/v1/auth/*` require a valid JWT in the `Authorization: Bearer <token>` header.

```
POST   /api/v1/auth/login            { email, password } → { token, refresh_token, user }
POST   /api/v1/auth/refresh           { refresh_token }   → { token, refresh_token }
POST   /api/v1/auth/logout            { refresh_token }   → 204
POST   /api/v1/auth/oauth/:provider   { code, state }     → { token, refresh_token, user }
```

**JWT payload:**
```json
{
  "sub": "user_id",
  "email": "user@example.com",
  "role": "admin | developer | viewer",
  "workspace_id": "ws_xxx",
  "exp": 1700000000
}
```

### Error Response Format

All errors return a consistent JSON envelope:
```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Agent with id 'ag_xxx' not found",
    "details": {}
  }
}
```

Standard error codes: `VALIDATION_ERROR` (400), `UNAUTHORIZED` (401), `FORBIDDEN` (403), `RESOURCE_NOT_FOUND` (404), `CONFLICT` (409), `RATE_LIMITED` (429), `INTERNAL_ERROR` (500).

### Pagination

All list endpoints support cursor-based pagination:
```
GET /api/v1/agents?cursor=xxx&limit=20&sort=created_at&order=desc
```

Response envelope:
```json
{
  "data": [...],
  "pagination": {
    "next_cursor": "xxx",
    "has_more": true,
    "total": 42
  }
}
```

---

### REST Endpoints

#### Agents

```
GET    /api/v1/agents                       → list agents (paginated, filterable by status/model)
POST   /api/v1/agents                       → create agent
GET    /api/v1/agents/:id                   → get agent by id
PUT    /api/v1/agents/:id                   → update agent
DELETE /api/v1/agents/:id                   → delete agent
POST   /api/v1/agents/:id/duplicate         → clone agent
GET    /api/v1/agents/:id/versions          → list version history
GET    /api/v1/agents/:id/versions/:ver     → get specific version
POST   /api/v1/agents/:id/versions/:ver/restore → rollback to version
POST   /api/v1/agents/:id/publish           → promote version to live
```

**Agent schema:**
```json
{
  "id": "ag_xxx",
  "name": "Research Assistant",
  "model": "anthropic:claude-sonnet-4-5-20250929",
  "instructions": "You are a research assistant...",
  "temperature": 1.0,
  "max_tokens": null,
  "max_steps": 10,
  "output_type": null,
  "tools": ["tl_xxx", "tl_yyy"],
  "handoffs": ["ag_yyy"],
  "hooks": [{ "point": "before_tool_call", "hook_id": "hk_xxx" }],
  "context_config": {
    "automation_level": "copilot",
    "token_budget": 4096,
    "checkpoint_frequency": 5
  },
  "memory_enabled": false,
  "status": "active | draft | archived",
  "version": 3,
  "live_version": 2,
  "created_at": "2026-02-16T00:00:00Z",
  "updated_at": "2026-02-16T12:00:00Z",
  "created_by": "user_xxx"
}
```

#### Workflows

```
GET    /api/v1/workflows                    → list workflows
POST   /api/v1/workflows                    → create workflow
GET    /api/v1/workflows/:id                → get workflow (includes full canvas JSON)
PUT    /api/v1/workflows/:id                → update workflow
DELETE /api/v1/workflows/:id                → delete workflow
POST   /api/v1/workflows/:id/duplicate      → clone workflow
POST   /api/v1/workflows/:id/export         → export as portable JSON
POST   /api/v1/workflows/import             → import from JSON
GET    /api/v1/workflows/:id/versions       → version history
POST   /api/v1/workflows/:id/run            → execute workflow
POST   /api/v1/workflows/:id/debug          → start debug (step-through) run
```

**Workflow schema:**
```json
{
  "id": "wf_xxx",
  "name": "RAG Pipeline",
  "description": "...",
  "app_type": "workflow | chatbot | chatflow | agent | text_generator",
  "canvas": {
    "nodes": [
      {
        "id": "node_1",
        "type": "llm | tool | conditional | code | knowledge | transform | iterator | http | input | output | approval | question_classifier | variable_assigner",
        "position": { "x": 100, "y": 200 },
        "config": {},
        "data": { "label": "GPT-4o Call", "status": "idle" }
      }
    ],
    "edges": [
      {
        "id": "edge_1",
        "source": "node_1",
        "source_handle": "output_0",
        "target": "node_2",
        "target_handle": "input_0",
        "data_type": "text | tool_call | structured | any"
      }
    ]
  },
  "variables": {},
  "version": 1,
  "status": "draft | published",
  "created_at": "...",
  "updated_at": "..."
}
```

#### Tools

```
GET    /api/v1/tools                        → list tools (filterable by category: builtin/custom/marketplace)
POST   /api/v1/tools                        → create custom tool
GET    /api/v1/tools/:id                    → get tool with full schema
PUT    /api/v1/tools/:id                    → update tool
DELETE /api/v1/tools/:id                    → delete tool
POST   /api/v1/tools/:id/test               → execute tool with sample input → result
POST   /api/v1/tools/import-openapi         → import tools from OpenAPI spec
GET    /api/v1/tools/:id/usage              → usage statistics
```

**Tool schema:**
```json
{
  "id": "tl_xxx",
  "name": "web_search",
  "description": "Search the web for information",
  "category": "builtin | custom | marketplace",
  "source_code": "...",
  "parameters": {
    "type": "object",
    "properties": {
      "query": { "type": "string", "description": "Search query" }
    },
    "required": ["query"]
  },
  "return_type": { "type": "array", "items": { "type": "object" } },
  "usage_count": 142,
  "created_at": "..."
}
```

#### Model Providers

```
GET    /api/v1/providers                    → list all providers with status
GET    /api/v1/providers/:name              → get provider detail (openai|anthropic|gemini|vertex|ollama|custom)
PUT    /api/v1/providers/:name              → update provider config
GET    /api/v1/providers/:name/models       → list available models for provider
POST   /api/v1/providers/:name/test         → test provider connectivity
```

**Credentials (sub-resource):**
```
GET    /api/v1/providers/:name/credentials           → list credentials (keys masked)
POST   /api/v1/providers/:name/credentials           → add credential
PUT    /api/v1/providers/:name/credentials/:id       → update credential
DELETE /api/v1/providers/:name/credentials/:id       → delete credential
POST   /api/v1/providers/:name/credentials/:id/test  → test specific key
```

**Provider schema:**
```json
{
  "name": "openai",
  "display_name": "OpenAI",
  "status": "connected | error | unconfigured",
  "credentials": [
    {
      "id": "cred_xxx",
      "label": "Production Key",
      "api_key_masked": "sk-...xxxx",
      "base_url": null,
      "max_retries": 3,
      "timeout": 30.0,
      "status": "connected | invalid | untested",
      "request_count": 1542,
      "token_count": 2450000,
      "error_count": 12
    }
  ],
  "load_balancing": {
    "enabled": true,
    "strategy": "round_robin | least_used | random",
    "cooldown_seconds": 60
  },
  "models": ["gpt-4o", "gpt-4o-mini", "o1", "o3-mini"]
}
```

#### Runs

```
GET    /api/v1/runs                         → list runs (paginated, filterable by status/agent/date)
GET    /api/v1/runs/:id                     → get run detail with execution trace
DELETE /api/v1/runs/:id                     → delete run
POST   /api/v1/runs/:id/cancel              → cancel a running execution
GET    /api/v1/runs/:id/steps               → list all steps in a run
GET    /api/v1/runs/:id/steps/:step_id      → get step detail (full input/output/raw API)
GET    /api/v1/runs/:id/artifacts           → list artifacts produced by run
GET    /api/v1/runs/:id/artifacts/:name     → download specific artifact
```

**Run schema:**
```json
{
  "id": "run_xxx",
  "agent_id": "ag_xxx",
  "workflow_id": null,
  "status": "running | success | failed | cancelled",
  "started_at": "2026-02-16T12:00:00Z",
  "completed_at": "2026-02-16T12:00:05Z",
  "duration_ms": 5200,
  "total_tokens": { "prompt": 1200, "completion": 450 },
  "estimated_cost_usd": 0.0234,
  "step_count": 3,
  "error": null,
  "steps": [
    {
      "id": "step_1",
      "type": "llm_call | tool_call | handoff",
      "started_at": "...",
      "duration_ms": 1200,
      "tokens": { "prompt": 800, "completion": 200 },
      "input": {},
      "output": {},
      "model": "anthropic:claude-sonnet-4-5-20250929",
      "finish_reason": "stop | tool_calls | length",
      "error": null
    }
  ]
}
```

#### Conversations (Playground)

```
GET    /api/v1/conversations                         → list conversations
POST   /api/v1/conversations                         → create conversation
GET    /api/v1/conversations/:id                     → get conversation with messages
PUT    /api/v1/conversations/:id                     → rename conversation
DELETE /api/v1/conversations/:id                     → delete conversation
POST   /api/v1/conversations/:id/messages             → send message (triggers agent run)
GET    /api/v1/conversations/:id/messages             → list messages
PUT    /api/v1/conversations/:id/messages/:msg_id    → edit message
DELETE /api/v1/conversations/:id/messages/:msg_id    → delete message
POST   /api/v1/conversations/:id/messages/:msg_id/regenerate → regenerate response
POST   /api/v1/conversations/:id/messages/:msg_id/feedback   → thumbs up/down
POST   /api/v1/conversations/:id/messages/:msg_id/annotate   → add annotation reply
```

**Message schema:**
```json
{
  "id": "msg_xxx",
  "conversation_id": "conv_xxx",
  "role": "user | assistant | system | tool",
  "content": "Hello, can you help me...",
  "metadata": {
    "model": "anthropic:claude-sonnet-4-5-20250929",
    "tokens": { "prompt": 120, "completion": 85 },
    "latency_ms": 1450,
    "run_id": "run_xxx",
    "tool_calls": [
      { "name": "web_search", "arguments": {}, "result": "..." }
    ],
    "finish_reason": "stop"
  },
  "feedback": { "rating": "up | down | null", "comment": "" },
  "annotation": null,
  "created_at": "..."
}
```

#### Knowledge Bases

```
GET    /api/v1/knowledge-bases                        → list knowledge bases
POST   /api/v1/knowledge-bases                        → create knowledge base
GET    /api/v1/knowledge-bases/:id                    → get detail
PUT    /api/v1/knowledge-bases/:id                    → update config
DELETE /api/v1/knowledge-bases/:id                    → delete
POST   /api/v1/knowledge-bases/:id/documents          → upload document(s) (multipart)
GET    /api/v1/knowledge-bases/:id/documents          → list documents
DELETE /api/v1/knowledge-bases/:id/documents/:doc_id  → delete document
POST   /api/v1/knowledge-bases/:id/search              → query the knowledge base
```

**Knowledge Base schema:**
```json
{
  "id": "kb_xxx",
  "name": "Product Documentation",
  "embedding_model": "openai:text-embedding-3-small",
  "chunking_strategy": "fixed_size | semantic",
  "chunk_size": 512,
  "chunk_overlap": 50,
  "vector_store": "builtin | milvus | pinecone | weaviate | qdrant | chroma",
  "document_count": 24,
  "total_chunks": 1580,
  "status": "ready | processing | error",
  "created_at": "..."
}
```

**Search request/response:**
```json
// POST /api/v1/knowledge-bases/:id/search
{ "query": "How do I configure an agent?", "top_k": 5, "threshold": 0.7, "rerank": true, "mode": "semantic | keyword | hybrid" }

// Response
{
  "results": [
    { "chunk_text": "...", "score": 0.92, "document_name": "getting-started.md", "chunk_index": 14 }
  ]
}
```

#### Evaluations

```
GET    /api/v1/evaluations                            → list evaluation suites
POST   /api/v1/evaluations                            → create suite
GET    /api/v1/evaluations/:id                        → get suite
PUT    /api/v1/evaluations/:id                        → update suite config
DELETE /api/v1/evaluations/:id                        → delete suite
POST   /api/v1/evaluations/:id/run                     → execute evaluation
GET    /api/v1/evaluations/:id/results                 → list results
GET    /api/v1/evaluations/:id/results/:result_id      → get detailed result
POST   /api/v1/evaluations/:id/results/:result_id/export → export as CSV/JSON
```

**Evaluation schema:**
```json
{
  "id": "eval_xxx",
  "name": "Quality Check v2",
  "target_agent_id": "ag_xxx",
  "type": "quality | safety | custom | red_team | benchmark",
  "evaluators": ["relevance", "coherence", "groundedness", "fluency", "toxicity"],
  "custom_evaluator_code": null,
  "dataset": [
    { "input": "What is Orbiter?", "expected_output": "Orbiter is a multi-agent framework..." }
  ],
  "last_run_at": "...",
  "last_score": 0.87
}
```

#### Swarms / Crews

```
GET    /api/v1/swarms                                 → list swarms
POST   /api/v1/swarms                                 → create swarm
GET    /api/v1/swarms/:id                             → get swarm config
PUT    /api/v1/swarms/:id                             → update swarm
DELETE /api/v1/swarms/:id                             → delete swarm
POST   /api/v1/swarms/:id/run                          → execute swarm
GET    /api/v1/swarms/:id/runs                         → list swarm run history
```

**Swarm schema:**
```json
{
  "id": "sw_xxx",
  "name": "Research Crew",
  "pattern": "supervisor | planner_executor_verifier | crew | custom",
  "agents": [
    {
      "agent_id": "ag_xxx",
      "role": "Lead Researcher",
      "goal": "Find and synthesize information",
      "backstory": "You are an experienced research analyst...",
      "can_delegate": true
    }
  ],
  "tasks": [
    { "id": "task_1", "description": "Research the topic", "assigned_agent_id": "ag_xxx", "expected_output": "A summary", "dependencies": [] }
  ],
  "execution_mode": "sequential | parallel",
  "canvas": { "nodes": [], "edges": [] },
  "groups": [
    { "id": "grp_1", "type": "parallel | serial", "agent_ids": ["ag_xxx", "ag_yyy"] }
  ]
}
```

#### Plugins / Marketplace

```
GET    /api/v1/marketplace                             → browse marketplace (paginated, filterable)
GET    /api/v1/marketplace/:id                         → get listing detail
POST   /api/v1/marketplace/:id/install                 → install plugin
DELETE /api/v1/marketplace/:id/uninstall                → uninstall plugin
GET    /api/v1/plugins                                  → list installed plugins
GET    /api/v1/plugins/:id                              → get installed plugin detail
POST   /api/v1/plugins/publish                          → submit plugin for review
GET    /api/v1/plugins/publish/:id/status               → check review status
```

**Plugin schema:**
```json
{
  "id": "plg_xxx",
  "name": "Slack Integration",
  "author": "orbiter-community",
  "version": "1.2.0",
  "category": "tools | models | agent_templates | workflow_templates | extensions",
  "permissions": ["network", "filesystem:read"],
  "install_count": 450,
  "rating": 4.7,
  "status": "installed | available | pending_review",
  "docker_image": "orbiter-plugins/slack:1.2.0",
  "manifest": { "entry_point": "main.py", "resources": { "cpu": "0.5", "memory": "256Mi" } }
}
```

#### Prompts

```
GET    /api/v1/prompts                                 → list saved prompts
POST   /api/v1/prompts                                 → create prompt
GET    /api/v1/prompts/:id                             → get prompt with version history
PUT    /api/v1/prompts/:id                             → update prompt
DELETE /api/v1/prompts/:id                             → delete prompt
GET    /api/v1/prompts/:id/versions                    → list versions
POST   /api/v1/prompts/:id/versions/:ver/restore       → restore version
POST   /api/v1/prompts/:id/test                         → test against 1-4 models simultaneously
```

#### Annotations

```
GET    /api/v1/annotations                              → list all annotations
POST   /api/v1/annotations                              → create annotation
PUT    /api/v1/annotations/:id                          → update
DELETE /api/v1/annotations/:id                          → delete
POST   /api/v1/annotations/import                       → bulk import from CSV
GET    /api/v1/annotations/export                       → export as CSV
POST   /api/v1/annotations/match                        → test annotation matching for a query
```

#### Deployments

```
GET    /api/v1/deployments                              → list deployments
POST   /api/v1/deployments                              → create deployment (API or widget)
GET    /api/v1/deployments/:id                          → get detail (URL, status, usage)
PUT    /api/v1/deployments/:id                          → update config
DELETE /api/v1/deployments/:id                          → delete
POST   /api/v1/deployments/:id/toggle                   → enable/disable
GET    /api/v1/deployments/:id/usage                    → usage metrics
GET    /api/v1/deployments/:id/openapi                  → auto-generated OpenAPI spec
```

#### Context & Checkpoints

```
GET    /api/v1/contexts/:id                             → get context state tree
GET    /api/v1/contexts/:id/state                       → flat state key-values
GET    /api/v1/contexts/:id/children                    → list child contexts
GET    /api/v1/contexts/:id/checkpoints                 → list checkpoints
POST   /api/v1/contexts/:id/checkpoints/:ckpt_id/restore → restore
GET    /api/v1/contexts/:id/checkpoints/diff?from=x&to=y → diff two checkpoints
```

#### Sandbox

```
POST   /api/v1/sandbox/create                           → create sandbox session (Docker container)
GET    /api/v1/sandbox/:id                              → get sandbox status
POST   /api/v1/sandbox/:id/exec                         → execute command
GET    /api/v1/sandbox/:id/files                        → list files
GET    /api/v1/sandbox/:id/files/:path                  → download file
DELETE /api/v1/sandbox/:id                              → destroy sandbox
```

#### Monitoring & Analytics

```
GET    /api/v1/analytics/overview      → dashboard metrics (runs, success rate, tokens, cost, latency)
GET    /api/v1/analytics/usage         → token/cost breakdown by provider/model/agent over time
GET    /api/v1/analytics/latency       → latency percentiles (p50, p90, p99)
GET    /api/v1/analytics/errors        → error rate and recent errors
GET    /api/v1/analytics/providers     → per-provider health and usage
```

All analytics endpoints accept: `from`, `to`, `granularity` (1h|1d|1w), `agent_id`, `provider`.

#### Health, SLOs & Alerts

```
GET    /api/v1/health                  → system health check
GET    /api/v1/health/providers        → per-provider connectivity status
GET    /api/v1/slos                    → list SLO configurations
POST   /api/v1/slos                    → create SLO
PUT    /api/v1/slos/:id               → update SLO
DELETE /api/v1/slos/:id               → delete SLO
GET    /api/v1/slos/:id/status        → current status + burn rate

GET    /api/v1/alerts/rules           → list alert rules
POST   /api/v1/alerts/rules           → create rule
PUT    /api/v1/alerts/rules/:id       → update rule
DELETE /api/v1/alerts/rules/:id       → delete rule
GET    /api/v1/alerts/history          → list triggered alerts
POST   /api/v1/alerts/history/:id/resolve → resolve alert
POST   /api/v1/alerts/history/:id/snooze  → snooze alert
```

#### Settings & Workspace

```
GET    /api/v1/settings                → workspace settings
PUT    /api/v1/settings                → update settings
GET    /api/v1/settings/team           → list team members
POST   /api/v1/settings/team/invite    → invite member
PUT    /api/v1/settings/team/:user_id  → update role
DELETE /api/v1/settings/team/:user_id  → remove member
GET    /api/v1/settings/api-keys       → list API keys
POST   /api/v1/settings/api-keys       → generate key
DELETE /api/v1/settings/api-keys/:id   → revoke key
GET    /api/v1/settings/audit-log      → paginated audit log
GET    /api/v1/settings/environments   → list environments
PUT    /api/v1/settings/environments/:env → update env config
POST   /api/v1/settings/environments/:env/promote → promote to next
GET    /api/v1/settings/vector-stores  → get vector store config
PUT    /api/v1/settings/vector-stores  → update vector store config
POST   /api/v1/settings/vector-stores/test → test connection
```

#### Global Search

```
GET    /api/v1/search?q=xxx&types=agents,workflows,tools → search across all entities
```

#### Notifications

```
GET    /api/v1/notifications           → list notifications (paginated)
PUT    /api/v1/notifications/:id/read  → mark as read
POST   /api/v1/notifications/read-all  → mark all as read
```

#### AI Builder

```
POST   /api/v1/ai-builder/agent    { description } → generated AgentConfig
POST   /api/v1/ai-builder/workflow { description } → generated workflow canvas JSON
POST   /api/v1/ai-builder/crew     { description } → generated swarm config
POST   /api/v1/ai-builder/refine   { entity_type, entity_id, instruction } → refined config
```

#### Version Control

```
GET    /api/v1/versions/:entity_type/:entity_id            → list all versions
GET    /api/v1/versions/:entity_type/:entity_id/:ver       → get specific version
POST   /api/v1/versions/:entity_type/:entity_id/:ver/restore → restore
GET    /api/v1/versions/:entity_type/:entity_id/diff?from=1&to=3 → diff two versions
POST   /api/v1/versions/:entity_type/:entity_id/branch     → create draft branch
POST   /api/v1/versions/:entity_type/:entity_id/merge      → merge to live
```

#### CI/CD Webhooks

```
POST   /api/v1/webhooks                → register webhook
GET    /api/v1/webhooks                → list webhooks
DELETE /api/v1/webhooks/:id            → delete webhook
POST   /api/v1/ci/deploy               → deploy agent/workflow via CLI
POST   /api/v1/ci/evaluate             → run evaluation via CLI
POST   /api/v1/ci/promote              → promote environment via CLI
```

---

### WebSocket Protocol

Single global connection at `ws://host/api/v1/ws`. Authenticate via `Authorization` header on upgrade or as the first message.

**Channel multiplexing** — each message is a JSON envelope:

```json
{ "channel": "chat | logs | execution | sandbox | notifications | system", "action": "...", "payload": {} }
```

#### Channel: `system`
```json
// Authenticate on connect
{ "channel": "system", "action": "auth", "payload": { "token": "jwt_xxx" } }
→ { "channel": "system", "action": "auth_result", "payload": { "success": true } }

// Heartbeat (server sends every 30s)
{ "channel": "system", "action": "ping" }
→ { "channel": "system", "action": "pong" }
```

#### Channel: `chat`
```json
// Send message
{ "channel": "chat", "action": "message", "payload": { "conversation_id": "conv_xxx", "content": "Hello" } }

// Streaming token (server → client)
{ "channel": "chat", "action": "token", "payload": { "conversation_id": "conv_xxx", "message_id": "msg_xxx", "delta": "The ", "done": false } }

// Stream complete
{ "channel": "chat", "action": "complete", "payload": { "conversation_id": "conv_xxx", "message_id": "msg_xxx", "message": {} } }

// Stop generation
{ "channel": "chat", "action": "stop", "payload": { "conversation_id": "conv_xxx" } }

// Takeover (pause agent)
{ "channel": "chat", "action": "takeover", "payload": { "run_id": "run_xxx" } }

// Resume after takeover
{ "channel": "chat", "action": "resume", "payload": { "run_id": "run_xxx", "modified_context": {} } }
```

#### Channel: `execution`
```json
// Subscribe to run
{ "channel": "execution", "action": "subscribe", "payload": { "run_id": "run_xxx" } }

// Step started (server → client)
{ "channel": "execution", "action": "step_started", "payload": { "run_id": "run_xxx", "step": { "id": "step_1", "type": "llm_call", "node_id": "node_3" } } }

// Step completed
{ "channel": "execution", "action": "step_completed", "payload": { "run_id": "run_xxx", "step": {} } }

// Run completed
{ "channel": "execution", "action": "run_completed", "payload": { "run_id": "run_xxx", "status": "success" } }

// Approval requested (HITL gate)
{ "channel": "execution", "action": "approval_requested", "payload": { "run_id": "run_xxx", "step_id": "step_3", "details": {} } }

// Approve/reject
{ "channel": "execution", "action": "approval_response", "payload": { "run_id": "run_xxx", "step_id": "step_3", "approved": true } }

// Debug step-through controls
{ "channel": "execution", "action": "debug_step", "payload": { "run_id": "run_xxx" } }
{ "channel": "execution", "action": "debug_continue", "payload": { "run_id": "run_xxx" } }
{ "channel": "execution", "action": "debug_set_variable", "payload": { "run_id": "run_xxx", "key": "query", "value": "new" } }
```

#### Channel: `logs`
```json
// Subscribe
{ "channel": "logs", "action": "subscribe", "payload": { "filters": { "agent_id": "ag_xxx", "level": "INFO" } } }

// Log entry (server → client)
{ "channel": "logs", "action": "entry", "payload": { "timestamp": "...", "level": "INFO", "logger": "orbiter.agent", "message": "Starting tool call: web_search", "agent_id": "ag_xxx", "run_id": "run_xxx" } }
```

#### Channel: `sandbox`
```json
// Subscribe
{ "channel": "sandbox", "action": "subscribe", "payload": { "sandbox_id": "sb_xxx" } }

// Output (server → client)
{ "channel": "sandbox", "action": "output", "payload": { "sandbox_id": "sb_xxx", "stream": "stdout | stderr", "data": "Hello, World!\n" } }

// Send input (takeover mode)
{ "channel": "sandbox", "action": "input", "payload": { "sandbox_id": "sb_xxx", "data": "ls -la\n" } }

// File event
{ "channel": "sandbox", "action": "file_event", "payload": { "sandbox_id": "sb_xxx", "event": "created", "path": "/output/chart.png" } }

// Browser screenshot
{ "channel": "sandbox", "action": "screenshot", "payload": { "sandbox_id": "sb_xxx", "image_base64": "..." } }
```

#### Channel: `notifications`
```json
// New notification (server → client)
{ "channel": "notifications", "action": "new", "payload": { "id": "notif_xxx", "type": "alert | evaluation | approval | error | deployment", "title": "SLO breach", "body": "Success rate below 95%", "link": "/dashboard/slos/slo_xxx" } }
```
